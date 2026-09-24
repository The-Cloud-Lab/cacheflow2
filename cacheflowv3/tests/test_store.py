# SPDX-License-Identifier: Apache-2.0
import random

from cacheflowv3.server.store import ExtentAllocator, Store

MB = 1 << 20


def test_allocator_coalesces():
    a = ExtentAllocator(16 * MB, MB)
    offs = [a.alloc(MB) for _ in range(16)]
    assert sorted(offs) == [i * MB for i in range(16)]
    assert a.alloc(1) is None
    for o in offs[::2]:
        a.free(o, MB)
    assert a.alloc(2 * MB) is None  # fragmented
    for o in offs[1::2]:
        a.free(o, MB)
    assert a.num_extents == 1 and a.used_pages == 0
    assert a.alloc(16 * MB) == 0


def test_allocator_random_consistency():
    rng = random.Random(0)
    a = ExtentAllocator(64 * MB, MB)
    live = {}
    for _ in range(2000):
        if live and rng.random() < 0.5:
            o = rng.choice(list(live))
            a.free(o, live.pop(o))
        else:
            n = rng.randint(1, 6) * MB - rng.randint(0, MB - 1)
            o = a.alloc(n)
            if o is not None:
                assert all(o + n <= p or p + q <= o for p, q in live.items())
                live[o] = n
    assert a.used_pages == sum(a.pages_for(n) for n in live.values())


def _write(st, keys, sess="s1", size=4 * MB, skip=True):
    res = st.allocate_for_write(keys, size, sess, skip)
    st.commit([k for k, (s, _) in zip(keys, res) if s == "new"], sess)
    return [s for s, _ in res]


def test_lookup_longest_prefix_across_ranks():
    st = Store(64 * MB, MB)
    _write(st, ["a:0", "b:0", "c:0", "a:1", "b:1"])
    hits, lease, found = st.lookup(
        ["a", "b", "c"], ranks=2, pin=True, lease_ms=1000, owner="x"
    )
    assert hits == 2 and lease is not None
    assert [e.key for e in found[1]] == ["a:1", "b:1"]
    assert st.index["a:0"].pins == 1 and st.index["c:0"].pins == 0
    assert st.release(lease.lease_id) and st.index["a:0"].pins == 0
    assert not st.release(lease.lease_id)


def test_lru_eviction_respects_pins():
    st = Store(8 * MB, MB)  # room for two 4 MB entries
    _write(st, ["a:0", "b:0"])
    _, lease, _ = st.lookup(["a"], 1, pin=True, lease_ms=1000, owner="x")
    assert _write(st, ["c:0"]) == ["new"]  # evicts b (a is pinned)
    assert "a:0" in st.index and "b:0" not in st.index and "c:0" in st.index
    assert _write(st, ["d:0"]) == ["new"]  # evicts c
    _, lease2, _ = st.lookup(["d"], 1, pin=True, lease_ms=1000, owner="x")
    assert _write(st, ["e:0"]) == ["full"]  # a and d both pinned: nothing evictable
    st.release(lease.lease_id)
    assert _write(st, ["e:0"]) == ["new"]  # a unpinned now -> evicted
    assert "a:0" not in st.index and "d:0" in st.index
    st.release(lease2.lease_id)
    assert st.stats["evictions"] == 3


def test_writing_entries_are_not_evictable():
    st = Store(8 * MB, MB)
    res = st.allocate_for_write(["a:0", "b:0", "c:0"], 4 * MB, "s1", True)
    assert [s for s, _ in res] == ["new", "new", "full"]


def test_skip_existing_and_versioning():
    st = Store(64 * MB, MB)
    assert _write(st, ["a:0"]) == ["new"]
    assert _write(st, ["a:0"]) == ["exists"]
    _, lease, found = st.lookup(["a"], 1, pin=True, lease_ms=1000, owner="x")
    old = found[0][0]
    assert _write(st, ["a:0"], skip=False) == ["new"]  # new version while old is pinned
    assert st.index["a:0"] is not old and old in st.retired
    used = st.alloc.used_bytes
    st.release(lease.lease_id)
    assert not st.retired and st.alloc.used_bytes < used


def test_abort_and_session_drop_free_space():
    st = Store(64 * MB, MB)
    st.allocate_for_write(["a:0", "b:0"], 4 * MB, "s1", True)
    assert st.alloc.used_bytes == 8 * MB
    assert st.drop_session("s1") == 2 and st.alloc.used_bytes == 0
    assert st.lookup(["a"], 1, False, 0, "x")[0] == 0


def test_capacity_and_flush_and_lease_expiry():
    st = Store(64 * MB, MB, capacity_bytes=8 * MB)
    for k in ("a:0", "b:0", "c:0"):
        assert _write(st, [k]) == ["new"]
    assert len(st.index) == 2 and "a:0" not in st.index  # capacity forced an eviction
    st.set_capacity(4 * MB)
    assert len(st.index) == 1
    _, lease, _ = st.lookup([k.split(":")[0] for k in st.index], 1, True, 10, "x")
    assert st.flush() == 0  # pinned entry survives a flush
    assert st.expire_leases(now=1e18) == 1
    assert st.flush() == 1 and st.alloc.used_bytes == 0
