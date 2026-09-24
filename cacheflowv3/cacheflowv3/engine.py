# SPDX-License-Identifier: Apache-2.0
"""Worker-side transfer engine: moves KV chunks between vLLM's paged cache and
the BlueField-3 pool.

Save path (per request, per step):
  save_kv_layer(l) -> GPU gathers layer l of the new chunks into staging and
  records a CUDA event -> the saver thread allocates entries on the BF3 (once),
  waits for the event, posts the RDMA for that layer (overlap_dma_with_copy) or
  for all layers after the last one, then commits the entries.

Load path:
  start_load_kv -> post RDMA reads of the leased entries into staging (one
  completion group per layer if layerwise_load) -> wait_for_layer_load(l)
  waits for group l and scatters layer l into the paged cache.
"""

from __future__ import annotations

import contextlib
import itertools
import json
import logging
import os
import queue
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import torch

from . import rdma
from .config import CacheFlowConfig, GiB
from .kv_access import KVAccessor
from .layout import ChunkGeometry, plan_ops
from .protocol import ControlClient
from .staging import StagingPool, StagingSlice

log = logging.getLogger("vllm.cacheflowv3.engine")


class TransferError(RuntimeError):
    pass


class _PushWait:
    """Completion state for a server-driven (push) transfer."""

    def __init__(self, n_groups: int):
        self.events = [threading.Event() for _ in range(n_groups)]
        self.error: str | None = None

    def fail(self, err: str) -> None:
        self.error = err
        for e in self.events:
            e.set()

    def wait(self, group: int, timeout_s: float) -> None:
        if not self.events[group].wait(timeout_s):
            raise TransferError(f"push group {group} timed out")
        if self.error:
            raise TransferError(f"push transfer failed: {self.error}")


class DpuSession:
    """Control connection (+ RDMA endpoint for workers) to the CacheFlow server."""

    def __init__(self, cfg: CacheFlowConfig, name: str, want_rdma: bool):
        self.cfg = cfg
        self._push: dict[str, _PushWait] = {}
        self._push_lock = threading.Lock()
        self.ctl = ControlClient(
            cfg.server_host,
            cfg.server_port,
            timeout=cfg.connect_timeout_s,
            on_event=self._on_event,
            client_name=name,
        )
        self.info = self.ctl.session
        self.dev: rdma.Device | None = None
        self.ep: rdma.Endpoint | None = None
        if want_rdma:
            self.dev = rdma.Device(cfg.server_host, is_peer=True)
            self.ep = rdma.Endpoint.connect(
                self.dev,
                cfg.server_host,
                int(self.info["rdma_port"]),
                self.info["session"].encode(),
                timeout_ms=int(cfg.connect_timeout_s * 1000),
            )

    @property
    def rkey(self) -> int:
        return int(self.info["rkey"])

    def _on_event(self, msg: dict) -> None:
        if msg.get("event") == "push":
            with self._push_lock:
                w = self._push.get(msg["xfer"])
            if w is None:
                return
            if not msg.get("ok", False):
                w.fail(msg.get("error", "unknown"))
            else:
                w.events[msg["group"]].set()
        elif msg.get("event") == "disconnected":
            with self._push_lock:
                waits = list(self._push.values())
            for w in waits:
                w.fail("control connection lost")

    def push(self, xfer: str, n_groups: int, **fields) -> _PushWait:
        w = _PushWait(n_groups)
        with self._push_lock:
            self._push[xfer] = w
        try:
            self.ctl.request("push", xfer=xfer, **fields)
        except Exception as e:
            w.fail(str(e))
        return w

    def forget_push(self, xfer: str) -> None:
        with self._push_lock:
            self._push.pop(xfer, None)

    @property
    def healthy(self) -> bool:
        return self.ctl.alive and (self.ep is None or self.ep.error == 0)

    def close(self) -> None:
        if self.ep is not None:
            self.ep.close()
        self.ctl.close()
        if self.dev is not None:
            self.dev.close()


@dataclass
class SaveJob:
    req_id: str
    keys: list[str]
    slice: StagingSlice
    blk: torch.Tensor  # (n, C)
    off: torch.Tensor
    t_start: float = field(default_factory=time.perf_counter)
    layers_enqueued: int = 0
    done: threading.Event = field(default_factory=threading.Event)
    ok: bool = False


@dataclass
class LoadJob:
    req_id: str
    lease: int | None
    slice: StagingSlice
    n_chunks: int
    blk: torch.Tensor  # (T,) destination block per loaded token
    off: torch.Tensor
    jdx: torch.Tensor  # (T,) chunk position of each token in staging
    pdx: torch.Tensor  # (T,) position inside the chunk
    failed_blocks: set[int]
    tickets: list[int] | None = None
    push: _PushWait | None = None
    xfer: str | None = None
    layers_done: int = 0
    failed: bool = False
    t_start: float = field(default_factory=time.perf_counter)
    wait_ms: float = 0.0


class TransferEngine:
    def __init__(
        self, cfg: CacheFlowConfig, rank: int, world: int, device: torch.device
    ):
        self.cfg, self.rank, self.world, self.device = cfg, rank, world, device
        self.session = DpuSession(cfg, f"worker-r{rank}", want_rdma=True)
        info = self.session.info
        log.info(
            "CacheFlow v3 worker r%d/%d connected to %s:%d "
            "(server %s on %s, pool %.1f GiB, "
            "rdma %s -> %s)",
            rank,
            world,
            cfg.server_host,
            cfg.server_port,
            info.get("server_version"),
            info.get("device"),
            info["pool_bytes"] / GiB,
            self.session.dev.name,
            cfg.server_host,
        )
        self.staging = StagingPool(
            self.session.dev,
            int(cfg.staging_pool_gb * GiB),
            cfg.use_registered_buffer_pool,
            device,
        )
        _warn_if_memory_tight()
        if not self.staging.pinned_ok:
            log.warning(
                "staging memory is not seen as pinned by torch; "
                "staged copies will be synchronous"
            )
        if rank == 0:
            if cfg.reset_dpu_cache_on_start:
                n = self.session.ctl.request("flush")["dropped"]
                log.info("reset_dpu_cache_on_start: flushed %d entries", n)
            if cfg.dpu_capacity_gb > 0:
                cap = self.session.ctl.request(
                    "set_capacity", bytes=int(cfg.dpu_capacity_gb * GiB)
                )
                log.info(
                    "DPU pool capacity set to %.2f GiB", cap["capacity_bytes"] / GiB
                )

        self.geom: ChunkGeometry | None = None
        self.accessor: KVAccessor | None = None
        self.block_size = 0
        self._copy_streams = [
            torch.cuda.Stream(device) for _ in range(cfg.copy_stream_pool_size)
        ]
        self._stream_rr = itertools.cycle(range(len(self._copy_streams)))
        self._xfer_ids = itertools.count()

        self._save_q: queue.Queue = queue.Queue()
        self._saver = threading.Thread(
            target=self._save_loop, name="cacheflow-saver", daemon=True
        )
        self._saver.start()
        self._bg = ThreadPoolExecutor(1, thread_name_prefix="cacheflow-bg")
        self._deferred: list[tuple[torch.cuda.Event, StagingSlice]] = []
        self._stats_lock = threading.Lock()
        self.stats: dict[str, float] = {
            "loads": 0,
            "load_failures": 0,
            "load_bytes": 0,
            "load_tokens": 0,
            "load_wait_ms": 0.0,
            "load_total_ms": 0.0,
            "load_first_layer_ms": 0.0,
            "saves": 0,
            "save_failures": 0,
            "save_bytes": 0,
            "save_chunks_new": 0,
            "save_chunks_exists": 0,
            "save_chunks_busy": 0,
            "save_chunks_full": 0,
            "save_staging_rejected": 0,
            "save_alloc_ms": 0.0,
            "save_rdma_ms": 0.0,
            "save_commit_ms": 0.0,
            "save_total_ms": 0.0,
            "staging_setup_ms": 0.0,
        }

    # ------------------------------------------------------------- geometry
    def set_geometry(
        self, accessor: KVAccessor, num_layers: int, block_size: int
    ) -> None:
        self.accessor = accessor
        self.block_size = block_size
        self.geom = ChunkGeometry(
            num_layers, self.cfg.chunk_tokens * accessor.token_bytes
        )
        log.info(
            "KV geometry: %s, %d layers, %d B/token/layer, "
            "chunk %d tokens = %.1f MiB/entry",
            accessor.describe(),
            num_layers,
            accessor.token_bytes,
            self.cfg.chunk_tokens,
            self.geom.entry_bytes / 2**20,
        )

    def _layer_view(self, flat_u8: torch.Tensor, n: int) -> torch.Tensor:
        """(n, planes, C, F) staging bytes of one layer -> (planes, n, C, F) view."""
        a = self.accessor
        return (
            flat_u8.view(a.dtype)
            .view(n, a.planes, self.cfg.chunk_tokens, a.feat)
            .permute(1, 0, 2, 3)
        )

    def _stat(self, **inc) -> None:
        with self._stats_lock:
            for k, v in inc.items():
                self.stats[k] += v

    def _reap_deferred(self, block: bool = False) -> None:
        keep = []
        for ev, sl in self._deferred:
            if block:
                ev.synchronize()
            if block or ev.query():
                self.staging.release(sl)
            else:
                keep.append((ev, sl))
        self._deferred = keep

    # ----------------------------------------------------------------- save
    def begin_save(
        self, req_id: str, keys: list[str], blk: torch.Tensor, off: torch.Tensor
    ) -> SaveJob | None:
        self._reap_deferred()  # return finished loads' staging before allocating
        n = len(keys)
        sl = self.staging.acquire(n * self.geom.entry_bytes, wait_ms=0)
        if sl is None:
            self._stat(save_staging_rejected=1)
            return None
        self._stat(staging_setup_ms=sl.setup_ms)
        return SaveJob(req_id, keys, sl, blk, off)

    def save_layer(self, job: SaveJob, layer: int, kv: torch.Tensor) -> None:
        """Gather `layer` of the job's chunks into staging (async on the GPU)."""
        n = len(job.keys)
        S = self.geom.layer_bytes
        region = slice(layer * n * S, (layer + 1) * n * S)
        cur = torch.cuda.current_stream(self.device)
        keep = None
        if self.cfg.data_path == "mapped":
            self.accessor.gather(
                kv, job.blk, job.off, self._layer_view(job.slice.cuda[region], n)
            )
            ev = torch.cuda.Event()
            ev.record(cur)
        else:
            tmp = torch.empty(n * S, dtype=torch.uint8, device=self.device)
            self.accessor.gather(kv, job.blk, job.off, self._layer_view(tmp, n))
            g = torch.cuda.Event()
            g.record(cur)
            s = self._copy_streams[next(self._stream_rr)]
            s.wait_event(g)
            with torch.cuda.stream(s):
                job.slice.cpu[region].copy_(tmp, non_blocking=True)
                ev = torch.cuda.Event()
                ev.record(s)
            tmp.record_stream(s)
            keep = tmp
        job.layers_enqueued += 1
        self._save_q.put((job, layer, ev, keep))

    def end_save_step(self, job: SaveJob) -> None:
        """No more layers will arrive for this job in this step. If some never
        did (e.g. a layer without its own KV), the save is aborted."""
        if job.layers_enqueued < self.geom.num_layers:
            self._save_q.put((job, -1, None, None))

    def _save_loop(self) -> None:
        """Saver thread: allocate, move each layer when its gather is done, commit."""
        state: dict[int, dict] = {}
        while True:
            item = self._save_q.get()
            if item is None:
                return
            job, layer, ev, _keep = item
            L = self.geom.num_layers
            st = state.get(id(job))
            if st is None:
                try:
                    st = self._save_start(job)
                except Exception as e:
                    log.warning("save of %s: allocation failed: %s", job.req_id, e)
                    st = {
                        "new": [],
                        "addrs": [],
                        "rkey": 0,
                        "layers": 0,
                        "failed": True,
                        "tickets": [],
                        "pushes": [],
                        "t_rdma": None,
                    }
                state[id(job)] = st
            if layer < 0:  # finalize marker: some layers never came
                st["failed"] = True
                self._save_finish(job, st)
                del state[id(job)]
                continue
            try:
                ev.synchronize()  # the GPU gather of this layer is complete
                if st["new"] and not st["failed"]:
                    if self.cfg.overlap_dma_with_copy:
                        self._save_post(job, st, (layer, layer + 1))
                    elif st["layers"] + 1 == L:
                        self._save_post(job, st, (0, L))
            except Exception as e:
                log.warning("save of %s failed at layer %d: %s", job.req_id, layer, e)
                st["failed"] = True
            st["layers"] += 1
            if st["layers"] == L:
                self._save_finish(job, st)
                del state[id(job)]

    def _save_start(self, job: SaveJob) -> dict:
        t = time.perf_counter()
        r = self.session.ctl.request(
            "alloc",
            keys=job.keys,
            rank=self.rank,
            size=self.geom.entry_bytes,
            skip_existing=self.cfg.skip_save_if_prefix_cached,
        )
        alloc_ms = (time.perf_counter() - t) * 1e3
        new = [i for i, s in enumerate(r["status"]) if s == "new"]
        counts = {s: r["status"].count(s) for s in ("new", "exists", "busy", "full")}
        self._stat(
            save_alloc_ms=alloc_ms,
            save_chunks_new=counts["new"],
            save_chunks_exists=counts["exists"],
            save_chunks_busy=counts["busy"],
            save_chunks_full=counts["full"],
        )
        return {
            "new": new,
            "addrs": [r["addrs"][i] for i in new],
            "rkey": r["rkey"],
            "layers": 0,
            "failed": False,
            "tickets": [],
            "pushes": [],
            "t_rdma": None,
        }

    def _save_post(self, job: SaveJob, st: dict, layers: tuple[int, int]) -> None:
        if st["t_rdma"] is None:
            st["t_rdma"] = time.perf_counter()
        n = len(job.keys)
        if self.cfg.transfer_mode == "pull":
            (ops,) = plan_ops(
                self.geom,
                job.slice.addr,
                job.slice.lkey,
                st["addrs"],
                st["rkey"],
                layer_groups=False,
                staging_chunks=st["new"],
                n_staging=n,
                layers=layers,
            )
            st["tickets"].append(self.session.ep.post(rdma.OP_WRITE, ops))
        else:
            xfer = f"s{self.rank}-{next(self._xfer_ids)}"
            w = self.session.push(
                xfer,
                1,
                dir="save",
                entries=st["addrs"],
                num_layers=self.geom.num_layers,
                layer_bytes=self.geom.layer_bytes,
                staging_base=job.slice.addr,
                staging_rkey=job.slice.rkey,
                layer_groups=False,
                staging_chunks=st["new"],
                n_staging=n,
                layers=list(layers),
            )
            st["pushes"].append((xfer, w))

    def _save_finish(self, job: SaveJob, st: dict) -> None:
        timeout_ms = self.cfg.transfer_timeout_ms
        keys = [job.keys[i] for i in st.get("new", [])]
        ok = not st.get("failed", False)
        # Drain every posted transfer, even when aborting: entries and staging
        # must not be reused while the NIC may still be touching them.
        for t in st.get("tickets", []):
            try:
                self.session.ep.wait(t, timeout_ms)
            except Exception as e:
                ok = False
                log.warning("save of %s: RDMA write failed: %s", job.req_id, e)
        for xfer, w in st.get("pushes", []):
            try:
                w.wait(0, timeout_ms / 1000)
            except Exception as e:
                ok = False
                log.warning("save of %s: push failed: %s", job.req_id, e)
            self.session.forget_push(xfer)
        try:
            rdma_ms = (
                (time.perf_counter() - st["t_rdma"]) * 1e3 if st.get("t_rdma") else 0.0
            )
            if keys:
                t = time.perf_counter()
                self.session.ctl.request(
                    "commit" if ok else "abort", keys=keys, rank=self.rank
                )
                self._stat(save_commit_ms=(time.perf_counter() - t) * 1e3)
            if ok:
                self._stat(
                    saves=1,
                    save_bytes=len(keys) * self.geom.entry_bytes,
                    save_rdma_ms=rdma_ms,
                    save_total_ms=(time.perf_counter() - job.t_start) * 1e3,
                )
        except Exception as e:
            ok = False
            log.warning("save of %s failed at completion: %s", job.req_id, e)
            with contextlib.suppress(Exception):
                self.session.ctl.request("abort", keys=keys, rank=self.rank)
        if not ok:
            self._stat(save_failures=1)
        self.staging.release(job.slice)
        job.ok = ok
        job.done.set()

    # ----------------------------------------------------------------- load
    def begin_load(
        self,
        req_id: str,
        lease: int | None,
        entries: list[int],
        rkey: int,
        chunk0: int,
        tok_start: int,
        tok_stop: int,
        block_ids: list[int],
    ) -> LoadJob:
        self._reap_deferred()
        C, B = self.cfg.chunk_tokens, self.block_size
        n = len(entries)
        failed_blocks = {
            block_ids[t // B]
            for t in range(tok_start, tok_stop)
            if t // B < len(block_ids)
        }
        t = torch.arange(tok_start, tok_stop, dtype=torch.int64)
        table = torch.tensor(block_ids, dtype=torch.int64)
        dev = self.device
        blk = table[t // B].to(dev, non_blocking=True)
        off = (t % B).to(dev, non_blocking=True)
        jdx = (t // C - chunk0).to(dev, non_blocking=True)
        pdx = (t % C).to(dev, non_blocking=True)
        need = n * self.geom.entry_bytes
        if need > self.staging.capacity and self.staging.registered_pool:
            raise TransferError(
                f"a {n}-chunk load needs {need / GiB:.2f} GiB but staging_pool_gb is "
                f"{self.staging.capacity / GiB:.2f} GiB; raise staging_pool_gb"
            )
        sl = self.staging.acquire(need, wait_ms=self.cfg.load_staging_wait_ms)
        if sl is None:
            raise TransferError(f"no staging memory for a {n}-chunk load")
        self._stat(staging_setup_ms=sl.setup_ms)
        job = LoadJob(req_id, lease, sl, n, blk, off, jdx, pdx, failed_blocks)
        try:
            if self.cfg.transfer_mode == "pull":
                groups = plan_ops(
                    self.geom,
                    sl.addr,
                    sl.lkey,
                    entries,
                    rkey,
                    layer_groups=self.cfg.layerwise_load,
                )
                job.tickets = [self.session.ep.post(rdma.OP_READ, g) for g in groups]
            else:
                job.xfer = f"l{self.rank}-{next(self._xfer_ids)}"
                n_groups = self.geom.num_layers if self.cfg.layerwise_load else 1
                job.push = self.session.push(
                    job.xfer,
                    n_groups,
                    dir="load",
                    entries=entries,
                    num_layers=self.geom.num_layers,
                    layer_bytes=self.geom.layer_bytes,
                    staging_base=sl.addr,
                    staging_rkey=sl.rkey,
                    layer_groups=self.cfg.layerwise_load,
                )
        except Exception:
            self.staging.release(sl)
            raise
        return job

    def load_layer(self, job: LoadJob, layer: int, kv: torch.Tensor) -> None:
        """Wait for `layer` of the job to land, then scatter it into kv."""
        group = layer if self.cfg.layerwise_load else 0
        t = time.perf_counter()
        if job.layers_done == 0 or self.cfg.layerwise_load:
            timeout = self.cfg.transfer_timeout_ms
            if job.tickets is not None:
                self.session.ep.wait(job.tickets[group], timeout)
            else:
                job.push.wait(group, timeout / 1000)
        waited = (time.perf_counter() - t) * 1e3
        job.wait_ms += waited
        if job.layers_done == 0:
            self._stat(load_first_layer_ms=(time.perf_counter() - job.t_start) * 1e3)
        n, S = job.n_chunks, self.geom.layer_bytes
        region = slice(layer * n * S, (layer + 1) * n * S)
        cur = torch.cuda.current_stream(self.device)
        if self.cfg.data_path == "mapped":
            src = self._layer_view(job.slice.cuda[region], n)
        else:
            s = self._copy_streams[next(self._stream_rr)]
            with torch.cuda.stream(s):
                # Allocated on the copy stream, so the caching allocator won't
                # hand this memory to another copy until `cur` has consumed it.
                tmp = torch.empty(n * S, dtype=torch.uint8, device=self.device)
                tmp.copy_(job.slice.cpu[region], non_blocking=True)
                ev = torch.cuda.Event()
                ev.record(s)
            cur.wait_event(ev)
            tmp.record_stream(cur)
            src = self._layer_view(tmp, n)
        self.accessor.scatter(kv, job.blk, job.off, src[:, job.jdx, job.pdx])
        job.layers_done += 1

    def finish_load(self, job: LoadJob) -> None:
        """Called once all layers are scattered (or the load failed)."""
        if job.failed:
            # Reads for later layers may still be landing in staging; drain them
            # before the staging memory or the lease can be reused.
            timeout = self.cfg.transfer_timeout_ms
            try:
                if job.tickets:
                    self.session.ep.wait(job.tickets[-1], timeout)
                elif job.push is not None:
                    job.push.wait(len(job.push.events) - 1, timeout / 1000)
            except Exception:
                pass
        ev = torch.cuda.Event()
        ev.record(torch.cuda.current_stream(self.device))
        self._deferred.append((ev, job.slice))
        if job.xfer:
            self.session.forget_push(job.xfer)
        if job.lease is not None:
            self._bg.submit(self._release_lease, job.lease)
        if job.failed:
            self._stat(load_failures=1)
        else:
            self._stat(
                loads=1,
                load_bytes=job.n_chunks * self.geom.entry_bytes,
                load_tokens=int(job.blk.numel()),
                load_wait_ms=job.wait_ms,
                load_total_ms=(time.perf_counter() - job.t_start) * 1e3,
            )

    def _release_lease(self, lease: int) -> None:
        try:
            self.session.ctl.request("release", lease=lease)
        except Exception as e:
            log.debug("lease release failed: %s", e)

    # ------------------------------------------------------------ lifecycle
    def drain_saves(self, jobs: list[SaveJob], timeout_s: float | None = None) -> None:
        for j in jobs:
            j.done.wait(timeout_s)

    def snapshot(self) -> dict:
        with self._stats_lock:
            s = dict(self.stats)
        s["staging"] = dict(self.staging.stats)
        if self.session.ep is not None:
            s["rdma"] = self.session.ep.stats()
        return s

    def write_stats(self, path: str, extra: dict | None = None) -> None:
        try:
            tmp = f"{path}.tmp"
            with open(tmp, "w") as f:
                json.dump(
                    {
                        "config": self.cfg.as_dict(),
                        "rank": self.rank,
                        "stats": self.snapshot(),
                        **(extra or {}),
                    },
                    f,
                    indent=1,
                )
            os.replace(tmp, path)
        except OSError as e:
            log.debug("could not write stats: %s", e)

    def close(self) -> None:
        self._save_q.put(None)
        self._saver.join(timeout=self.cfg.transfer_timeout_ms / 1000)
        self._bg.shutdown(wait=True)
        with contextlib.suppress(Exception):
            torch.cuda.synchronize(self.device)
        self._reap_deferred(block=True)
        self.staging.close()
        self.session.close()


def _warn_if_memory_tight(min_available_gb: float = 4.0) -> None:
    """On unified-memory hosts (DGX Spark) the pinned staging pool competes with
    vLLM for the same DRAM. When the system is left with little available
    memory it starts reclaiming/swapping, which shows up as multi-second GPU
    stalls. Warn loudly: this silently corrupts benchmark results."""
    try:
        with open("/proc/meminfo") as f:
            info = {line.split(":")[0]: int(line.split()[1]) for line in f}
    except OSError:
        return
    avail = info.get("MemAvailable", 0) / 2**20
    huge = info.get("HugePages_Total", 0) * info.get("Hugepagesize", 0) / 2**20
    if avail < min_available_gb:
        log.warning(
            "only %.1f GiB of system memory available after allocating CacheFlow "
            "staging "
            "(%.1f GiB is reserved as hugepages). Expect reclaim/swap stalls; lower "
            "staging_pool_gb or --gpu-memory-utilization, or free hugepages.",
            avail,
            huge,
        )


def natural_layer_order(names: list[str]) -> list[str]:
    """Order layer names by their numeric parts (model.layers.2 < model.layers.10)."""
    import re

    def key(s: str):
        return [int(p) if p.isdigit() else p for p in re.split(r"(\d+)", s)]

    return sorted(names, key=key)


def ids_array(x) -> np.ndarray:  # pragma: no cover - small helper for tests
    return np.asarray(x, dtype=np.int64)
