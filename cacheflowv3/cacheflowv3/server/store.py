# SPDX-License-Identifier: Apache-2.0
"""Prefix-cache store on the BlueField-3: pool allocator, index, LRU, leases.

Pure bookkeeping (no I/O), so it is unit-testable anywhere. All methods must be
called with the owning server's lock held (the server is single-threaded on
the control plane, so in practice from the asyncio loop).

Keys are "<chunk-hash>:<rank>". A chunk hash already chains the whole prefix,
so equal keys mean equal KV. Entries move WRITING -> READY -> (evicted/freed).
A READY entry can be pinned by leases; pinned entries are never evicted.
"""

from __future__ import annotations

import bisect
import itertools
import time
from collections import OrderedDict
from dataclasses import dataclass, field

WRITING, READY = "writing", "ready"


class ExtentAllocator:
    """First-fit allocator of contiguous page runs with coalescing frees."""

    def __init__(self, total_bytes: int, page_bytes: int):
        self.page = page_bytes
        self.total_pages = total_bytes // page_bytes
        self._starts: list[int] = [0]  # sorted free-extent starts (in pages)
        self._lens: dict[int, int] = {0: self.total_pages}
        self.used_pages = 0

    def pages_for(self, nbytes: int) -> int:
        return -(-nbytes // self.page)

    def alloc(self, nbytes: int) -> int | None:
        """Returns a byte offset, or None if no contiguous run fits."""
        need = self.pages_for(nbytes)
        for i, s in enumerate(self._starts):
            ln = self._lens[s]
            if ln >= need:
                del self._lens[s]
                self._starts.pop(i)
                if ln > need:
                    ns = s + need
                    self._starts.insert(i, ns)
                    self._lens[ns] = ln - need
                self.used_pages += need
                return s * self.page
        return None

    def free(self, offset: int, nbytes: int) -> None:
        s, n = offset // self.page, self.pages_for(nbytes)
        self.used_pages -= n
        i = bisect.bisect_left(self._starts, s)
        # merge with the following extent
        if i < len(self._starts) and self._starts[i] == s + n:
            n += self._lens.pop(self._starts.pop(i))
        # merge with the preceding extent
        if i > 0 and self._starts[i - 1] + self._lens[self._starts[i - 1]] == s:
            prev = self._starts[i - 1]
            self._lens[prev] += n
            return
        self._starts.insert(i, s)
        self._lens[s] = n

    @property
    def used_bytes(self) -> int:
        return self.used_pages * self.page

    @property
    def num_extents(self) -> int:
        return len(self._starts)


@dataclass
class Entry:
    key: str
    offset: int
    size: int
    state: str
    owner: str | None = None  # session that is writing it
    pins: int = 0
    retired: bool = False  # replaced by a newer version, free once unpinned
    created: float = field(default_factory=time.monotonic)


@dataclass
class Lease:
    lease_id: int
    keys: list[str]
    expires: float
    owner: str


class Store:
    def __init__(
        self, pool_bytes: int, page_bytes: int, capacity_bytes: int | None = None
    ):
        self.alloc = ExtentAllocator(pool_bytes, page_bytes)
        self.pool_bytes = self.alloc.total_pages * page_bytes
        self.capacity_bytes = min(capacity_bytes or self.pool_bytes, self.pool_bytes)
        self.index: dict[str, Entry] = {}  # READY entries
        self.lru: OrderedDict[str, None] = OrderedDict()  # READY keys, oldest first
        self.writing: dict[tuple[str, str], Entry] = {}  # (key, session) -> entry
        self.retired: list[Entry] = []  # replaced but still pinned
        self.leases: dict[int, Lease] = {}
        self._lease_ids = itertools.count(1)
        self.stats = {
            "lookups": 0,
            "lookup_keys": 0,
            "lookup_hit_keys": 0,
            "alloc_new": 0,
            "alloc_exists": 0,
            "alloc_busy": 0,
            "alloc_full": 0,
            "commits": 0,
            "aborts": 0,
            "evictions": 0,
            "evicted_bytes": 0,
            "leases_created": 0,
            "leases_expired": 0,
            "flushes": 0,
        }

    # ----------------------------------------------------------------- space
    def _free_entry(self, e: Entry) -> None:
        self.alloc.free(e.offset, e.size)

    def _evict_one(self) -> bool:
        for key in self.lru:
            e = self.index[key]
            if e.pins == 0:
                del self.lru[key]
                del self.index[key]
                self._free_entry(e)
                self.stats["evictions"] += 1
                self.stats["evicted_bytes"] += e.size
                return True
        return False

    def _fits_capacity(self, nbytes: int) -> bool:
        need = self.alloc.pages_for(nbytes) * self.alloc.page
        return self.alloc.used_bytes + need <= self.capacity_bytes

    def _allocate(self, nbytes: int) -> int | None:
        """Allocate, evicting unpinned LRU entries as needed."""
        while True:
            if self._fits_capacity(nbytes):
                off = self.alloc.alloc(nbytes)
                if off is not None:
                    return off
            if not self._evict_one():
                return None

    def set_capacity(self, capacity_bytes: int) -> int:
        self.capacity_bytes = max(0, min(int(capacity_bytes), self.pool_bytes))
        while self.alloc.used_bytes > self.capacity_bytes and self._evict_one():
            pass
        return self.capacity_bytes

    def flush(self) -> int:
        """Drop every unpinned READY entry (cold-cache experiments)."""
        dropped = 0
        for key in list(self.lru):
            e = self.index[key]
            if e.pins == 0:
                del self.lru[key]
                del self.index[key]
                self._free_entry(e)
                dropped += 1
        self.stats["flushes"] += 1
        return dropped

    # ---------------------------------------------------------------- lookup
    def lookup(
        self, keys: list[str], ranks: int, pin: bool, lease_ms: int, owner: str
    ) -> tuple[int, Lease | None, list[list[Entry]]]:
        """Longest prefix of `keys` present (READY) for every rank.

        Returns (hits, lease, entries[rank][chunk]) where entries cover the hit
        prefix. With pin=True the hit entries are pinned under a new lease.
        """
        self.stats["lookups"] += 1
        self.stats["lookup_keys"] += len(keys)
        found: list[list[Entry]] = [[] for _ in range(ranks)]
        hits = 0
        for k in keys:
            row = [self.index.get(f"{k}:{r}") for r in range(ranks)]
            if any(e is None for e in row):
                break
            for r, e in enumerate(row):
                found[r].append(e)
            hits += 1
        self.stats["lookup_hit_keys"] += hits
        for r in range(ranks):
            for e in found[r]:
                self.lru.move_to_end(e.key)
        lease = None
        if pin and hits:
            full = [e.key for r in range(ranks) for e in found[r]]
            for r in range(ranks):
                for e in found[r]:
                    e.pins += 1
            lease = Lease(
                next(self._lease_ids), full, time.monotonic() + lease_ms / 1000.0, owner
            )
            self.leases[lease.lease_id] = lease
            self.stats["leases_created"] += 1
        return hits, lease, found

    def _unpin(self, key: str) -> None:
        e = self.index.get(key)
        if e is not None and e.pins > 0:
            e.pins -= 1
            return
        # the pinned version may have been retired by a newer commit
        for i, r in enumerate(self.retired):
            if r.key == key and r.pins > 0:
                r.pins -= 1
                if r.pins == 0:
                    self._free_entry(r)
                    self.retired.pop(i)
                return

    def release(self, lease_id: int) -> bool:
        lease = self.leases.pop(int(lease_id), None)
        if lease is None:
            return False
        for k in lease.keys:
            self._unpin(k)
        return True

    def expire_leases(self, now: float | None = None) -> int:
        now = time.monotonic() if now is None else now
        dead = [lid for lid, ls in self.leases.items() if ls.expires <= now]
        for lid in dead:
            self.release(lid)
        self.stats["leases_expired"] += len(dead)
        return len(dead)

    # ----------------------------------------------------------------- write
    def allocate_for_write(
        self, keys: list[str], size: int, session: str, skip_existing: bool
    ) -> list[tuple[str, Entry | None]]:
        """Per key: ("new", entry) | ("exists", None) | ("busy", None) |
        ("full", None)."""
        out: list[tuple[str, Entry | None]] = []
        for k in keys:
            if skip_existing and k in self.index:
                self.lru.move_to_end(k)
                self.stats["alloc_exists"] += 1
                out.append(("exists", None))
                continue
            if skip_existing and any(wk == k for wk, _ in self.writing):
                self.stats["alloc_busy"] += 1
                out.append(("busy", None))
                continue
            if (k, session) in self.writing:
                self.stats["alloc_busy"] += 1
                out.append(("busy", None))
                continue
            off = self._allocate(size)
            if off is None:
                self.stats["alloc_full"] += 1
                out.append(("full", None))
                continue
            e = Entry(k, off, size, WRITING, owner=session)
            self.writing[(k, session)] = e
            self.stats["alloc_new"] += 1
            out.append(("new", e))
        return out

    def commit(self, keys: list[str], session: str) -> int:
        n = 0
        for k in keys:
            e = self.writing.pop((k, session), None)
            if e is None:
                continue
            old = self.index.get(k)
            if old is not None:
                del self.lru[k]
                if old.pins:
                    old.retired = True
                    self.retired.append(old)
                else:
                    self._free_entry(old)
            e.state, e.owner = READY, None
            self.index[k] = e
            self.lru[k] = None
            n += 1
        self.stats["commits"] += n
        return n

    def abort(self, keys: list[str], session: str) -> int:
        n = 0
        for k in keys:
            e = self.writing.pop((k, session), None)
            if e is not None:
                self._free_entry(e)
                n += 1
        self.stats["aborts"] += n
        return n

    def drop_session(self, session: str) -> int:
        """Abort everything a disconnected session was writing."""
        mine = [k for (k, s) in self.writing if s == session]
        return self.abort(mine, session)

    # ----------------------------------------------------------------- stats
    def snapshot(self) -> dict:
        return {
            **self.stats,
            "entries": len(self.index),
            "writing": len(self.writing),
            "retired": len(self.retired),
            "leases_active": len(self.leases),
            "used_bytes": self.alloc.used_bytes,
            "capacity_bytes": self.capacity_bytes,
            "pool_bytes": self.pool_bytes,
            "page_bytes": self.alloc.page,
            "free_extents": self.alloc.num_extents,
        }
