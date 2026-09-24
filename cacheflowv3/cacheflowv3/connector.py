# SPDX-License-Identifier: Apache-2.0
"""CacheFlowConnectorV3: vLLM KV connector for the BlueField-3 prefix-cache pool.

Out-of-tree connector; no vLLM changes are needed:

    vllm serve <model> --kv-transfer-config '{
        "kv_connector": "CacheFlowConnectorV3",
        "kv_connector_module_path": "cacheflowv3.connector",
        "kv_role": "kv_both",
        "kv_load_failure_policy": "recompute",
        "kv_connector_extra_config": {"server_host": "10.0.1.2"}
    }'

Scheduler role
  * hashes each prompt into chained chunk keys (hashing.py)
  * get_num_new_matched_tokens: asks the BF3 for the longest cached prefix and
    pins it under a lease (one control round trip; the index lives on the NIC)
  * build_connector_meta: emits LoadSpecs (lease + entry addresses + target
    vLLM blocks) and SaveSpecs for chunks whose KV is complete after this step
    (chunked prefill is handled incrementally)

Worker role
  * start_load_kv posts the RDMA reads; wait_for_layer_load(l) waits for layer
    l only (layerwise_load) and scatters it into the paged cache
  * save_kv_layer gathers each layer into NIC-registered memory; a background
    thread moves it to the BF3 and commits it (engine.py)
"""

from __future__ import annotations

import contextlib
import functools
import logging
import os
import threading
import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)

from .config import CacheFlowConfig
from .engine import DpuSession, LoadJob, SaveJob, TransferEngine, natural_layer_order
from .hashing import chunk_keys, namespace_seed
from .kv_access import KVAccessor

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

try:
    from vllm.logger import init_logger

    logger = init_logger("vllm.cacheflowv3.connector")
except Exception:  # pragma: no cover
    logger = logging.getLogger(__name__)


def _timed(fn):
    """Accumulate the CPU time vLLM spends inside a connector hook (for the
    overhead breakdown in the stats)."""
    name = fn.__name__

    @functools.wraps(fn)
    def wrapper(self, *args, **kwargs):
        t = time.perf_counter()
        try:
            return fn(self, *args, **kwargs)
        finally:
            self._hook_ms[name] = (
                self._hook_ms.get(name, 0.0) + (time.perf_counter() - t) * 1e3
            )
            self._hook_calls[name] = self._hook_calls.get(name, 0) + 1

    return wrapper


# ============================================================================
# Scheduler -> worker metadata
# ============================================================================


@dataclass
class LoadSpec:
    lease: int | None
    entries: list[
        list[int]
    ]  # [rank][chunk] server addresses, chunks [chunk0, chunk0+n)
    rkey: int
    chunk0: int
    tok_start: int  # load tokens [tok_start, tok_stop) of the prompt
    tok_stop: int
    block_ids: list[int]


@dataclass
class SaveSpec:
    keys: list[str]  # keys of chunks [chunk0, chunk0+len(keys))
    chunk0: int
    block_ids: list[int]


@dataclass
class CacheFlowV3Metadata(KVConnectorMetadata):
    loads: dict[str, LoadSpec] = field(default_factory=dict)
    saves: dict[str, SaveSpec] = field(default_factory=dict)


# ============================================================================
# Scheduler-side request state
# ============================================================================


@dataclass
class _Plan:
    lease: int | None
    entries: list[list[int]]
    rkey: int
    chunk0: int
    tok_start: int
    tok_stop: int
    created: float


@dataclass
class _ReqState:
    prompt_len: int
    keys: list[str]
    dpu_hits: int = 0  # leading chunks known to be in the pool
    saved_upto: int = 0  # chunks [0, saved_upto) handled by saves
    plan: _Plan | None = None
    plan_computed: int = -1
    load_emitted: bool = False
    block_ids: list[int] = field(default_factory=list)


def _token_ids(request: Request) -> list[int]:
    ids = getattr(request, "prompt_token_ids", None)
    return list(ids) if ids is not None else []


class CacheFlowConnectorV3(KVConnectorBase_V1):
    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig | None = None,
    ):
        super().__init__(vllm_config, role, kv_cache_config)
        self._hook_ms: dict[str, float] = {}
        self._hook_calls: dict[str, int] = {}
        extra = (
            getattr(self._kv_transfer_config, "kv_connector_extra_config", None) or {}
        )
        self.cfg = CacheFlowConfig.from_extra_config(extra)
        pc, mc = vllm_config.parallel_config, vllm_config.model_config
        self.world = pc.tensor_parallel_size * pc.pipeline_parallel_size
        self.block_size = vllm_config.cache_config.block_size
        self._seed = namespace_seed(
            mc.model,
            mc.dtype,
            vllm_config.cache_config.cache_dtype,
            mc.get_total_num_kv_heads(),
            mc.get_head_size(),
            mc.get_total_num_hidden_layers(),
            mc.use_mla,
            pc.tensor_parallel_size,
            pc.pipeline_parallel_size,
            self.cfg.chunk_tokens,
        )
        self._enabled = True
        if kv_cache_config is not None and len(kv_cache_config.kv_cache_groups) > 1:
            logger.warning(
                "CacheFlow v3: %d KV cache groups (hybrid/sliding-window model) are "
                "not "
                "supported; the connector is disabled.",
                len(kv_cache_config.kv_cache_groups),
            )
            self._enabled = False

        # scheduler state
        self._states: dict[str, _ReqState] = {}
        self._sched_session: DpuSession | None = None
        self._sched_retry_at = 0.0
        self._sched_bg = ThreadPoolExecutor(1, thread_name_prefix="cacheflow-sched")
        self._sched_stats = {
            "requests": 0,
            "lookups": 0,
            "lookup_ms": 0.0,
            "lookup_errors": 0,
            "hit_tokens": 0,
            "load_specs": 0,
            "save_specs": 0,
        }
        self._last_stats_write = time.monotonic()

        # worker state
        self.engine: TransferEngine | None = None
        self.rank = 0
        self._kv: dict[str, torch.Tensor] = {}
        self._layer_idx: dict[str, int] = {}
        self._loads: dict[str, LoadJob] = {}
        self._saves: dict[str, SaveJob] = {}
        self._save_skipped: set[str] = set()
        self._step_saves: list[SaveJob] = []
        self._load_errors: set[int] = set()
        self._lock = threading.Lock()
        # CACHEFLOW_PROFILE=1: per-step CPU/GPU timing, dumped next to the stats
        self._profile = os.environ.get("CACHEFLOW_PROFILE", "0") == "1"
        self._prof_steps: list[dict] = []
        self._prof_cur: dict | None = None
        if self._profile:
            import faulthandler
            import signal

            faulthandler.register(
                signal.SIGUSR1, all_threads=True
            )  # kill -USR1 dumps stacks

        if role == KVConnectorRole.WORKER and self._enabled:
            from vllm.distributed.parallel_state import (
                get_pp_group,
                get_tensor_model_parallel_rank,
            )

            self.rank = (
                get_pp_group().rank_in_group * pc.tensor_parallel_size
                + get_tensor_model_parallel_rank()
            )
            self.engine = TransferEngine(
                self.cfg,
                self.rank,
                self.world,
                torch.device("cuda", torch.cuda.current_device()),
            )
        logger.info(
            "CacheFlowConnectorV3 (%s) config: %s", role.name, self.cfg.as_dict()
        )

    @classmethod
    def requires_piecewise_for_cudagraph(cls, extra_config: dict[str, Any]) -> bool:
        return True  # per-layer hooks must run between graph pieces

    @classmethod
    def get_required_kvcache_layout(cls, vllm_config: VllmConfig) -> str | None:
        return None

    # ======================================================================
    # Scheduler side
    # ======================================================================

    def _session(self) -> DpuSession | None:
        s = self._sched_session
        if s is not None and s.ctl.alive:
            return s
        if time.monotonic() < self._sched_retry_at:
            return None
        try:
            self._sched_session = DpuSession(self.cfg, "scheduler", want_rdma=False)
            logger.info(
                "CacheFlow v3 scheduler connected to %s:%d",
                self.cfg.server_host,
                self.cfg.server_port,
            )
            return self._sched_session
        except Exception as e:
            self._sched_session = None
            self._sched_retry_at = time.monotonic() + 5.0
            logger.warning(
                "CacheFlow v3 server unreachable (%s); running without it for 5s", e
            )
            return None

    def _state(self, request: Request) -> _ReqState:
        rid = request.request_id
        st = self._states.get(rid)
        if st is None:
            ids = _token_ids(request)
            n = self.cfg.cacheable_tokens(len(ids)) // self.cfg.chunk_tokens
            st = _ReqState(
                len(ids), chunk_keys(ids, self.cfg.chunk_tokens, n, self._seed)
            )
            self._states[rid] = st
            self._sched_stats["requests"] += 1
        return st

    def _drop_plan(self, st: _ReqState) -> None:
        if st.plan is not None and st.plan.lease is not None and not st.load_emitted:
            self._release_async(st.plan.lease)
        st.plan, st.plan_computed = None, -1

    def _release_async(self, lease: int) -> None:
        s = self._sched_session
        if s is not None:
            self._sched_bg.submit(self._release, s, lease)

    @staticmethod
    def _release(s: DpuSession, lease: int) -> None:
        with contextlib.suppress(Exception):
            s.ctl.request("release", lease=lease)

    @_timed
    def get_num_new_matched_tokens(
        self, request: Request, num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        if not self._enabled:
            return 0, False
        st = self._state(request)
        if not st.keys:
            return 0, False
        C = self.cfg.chunk_tokens
        # Reuse a fresh plan if the scheduler asks again for the same request.
        if (
            st.plan is not None
            and st.plan_computed == num_computed_tokens
            and not st.load_emitted
            and (time.monotonic() - st.plan.created) * 1000 < self.cfg.lease_ms / 2
        ):
            return st.plan.tok_stop - st.plan.tok_start, False
        self._drop_plan(st)
        s = self._session()
        if s is None:
            return 0, False
        # If vLLM's local prefix cache already covers every cacheable chunk, a
        # lookup is still useful (it tells the save path what is already in the
        # pool), but there is nothing to pin.
        useful = self.cfg.enable_load and len(st.keys) * C > num_computed_tokens
        t = time.perf_counter()
        try:
            r = s.ctl.request(
                "lookup",
                keys=st.keys,
                ranks=self.world,
                pin=useful,
                lease_ms=self.cfg.lease_ms,
            )
        except Exception as e:
            self._sched_stats["lookup_errors"] += 1
            logger.warning("CacheFlow v3 lookup failed: %s", e)
            return 0, False
        self._sched_stats["lookups"] += 1
        self._sched_stats["lookup_ms"] += (time.perf_counter() - t) * 1e3
        hits, lease = int(r["hits"]), r.get("lease")
        st.dpu_hits = max(st.dpu_hits, hits)
        if not self.cfg.enable_load:
            hits = 0
        # vLLM must compute at least one token itself.
        loadable = min(hits * C, st.prompt_len - 1)
        ext = loadable - num_computed_tokens
        if ext <= 0:
            if lease is not None:
                self._release_async(lease)
            return 0, False
        c0 = num_computed_tokens // C
        c1 = -(-(num_computed_tokens + ext) // C)
        st.plan = _Plan(
            lease,
            [row[c0:c1] for row in r["entries"]],
            int(r["rkey"]),
            c0,
            num_computed_tokens,
            num_computed_tokens + ext,
            time.monotonic(),
        )
        st.plan_computed = num_computed_tokens
        st.load_emitted = False
        self._sched_stats["hit_tokens"] += ext
        self._maybe_write_sched_stats()
        return ext, False

    @_timed
    def update_state_after_alloc(
        self, request: Request, blocks: KVCacheBlocks, num_external_tokens: int
    ):
        st = self._states.get(request.request_id)
        if st is not None and num_external_tokens == 0 and st.plan is not None:
            self._drop_plan(st)

    @_timed
    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        meta = CacheFlowV3Metadata()
        if not self._enabled:
            return meta
        sched = scheduler_output.num_scheduled_tokens
        for req in scheduler_output.scheduled_new_reqs:
            st = self._states.get(req.req_id)
            if st is None:
                continue
            st.block_ids = list(req.block_ids[0]) if req.block_ids else []
            self._emit(
                req.req_id, st, req.num_computed_tokens, sched.get(req.req_id, 0), meta
            )

        cached = scheduler_output.scheduled_cached_reqs
        resumed_ids = getattr(cached, "resumed_req_ids", None)
        resumed_flags = getattr(cached, "resumed_from_preemption", None)
        for i, rid in enumerate(cached.req_ids):
            st = self._states.get(rid)
            if st is None:
                continue
            nb = cached.new_block_ids[i]
            resumed = (
                (rid in resumed_ids)
                if resumed_ids is not None
                else bool(resumed_flags[i] if resumed_flags else False)
            )
            if resumed:
                st.block_ids = list(nb[0]) if nb else []
            elif nb:
                st.block_ids.extend(nb[0])
            self._emit(rid, st, cached.num_computed_tokens[i], sched.get(rid, 0), meta)
        return meta

    def _emit(
        self,
        rid: str,
        st: _ReqState,
        num_computed: int,
        num_sched: int,
        meta: CacheFlowV3Metadata,
    ) -> None:
        p = st.plan
        if p is not None and not st.load_emitted:
            meta.loads[rid] = LoadSpec(
                p.lease,
                p.entries,
                p.rkey,
                p.chunk0,
                p.tok_start,
                p.tok_stop,
                list(st.block_ids),
            )
            st.load_emitted = True
            self._sched_stats["load_specs"] += 1
        if not st.keys or not self.cfg.enable_save:
            return
        done = min(num_computed + num_sched, st.prompt_len)
        upto = min(done // self.cfg.chunk_tokens, len(st.keys))
        start = st.saved_upto
        if self.cfg.skip_save_if_prefix_cached:
            start = max(start, st.dpu_hits)
        if upto > start:
            meta.saves[rid] = SaveSpec(st.keys[start:upto], start, list(st.block_ids))
            self._sched_stats["save_specs"] += 1
        st.saved_upto = max(st.saved_upto, upto)

    @_timed
    def request_finished(
        self, request: Request, block_ids: list[int]
    ) -> tuple[bool, dict[str, Any] | None]:
        st = self._states.pop(request.request_id, None)
        if st is not None and st.plan is not None and st.plan.lease is not None:
            self._release_async(st.plan.lease)  # idempotent with the worker's release
        return False, None

    def take_events(self) -> Iterable[Any]:
        return ()

    def _maybe_write_sched_stats(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_stats_write < self.cfg.stats_log_interval_s:
            return
        self._last_stats_write = now
        s = self._sched_stats
        logger.info(
            "CacheFlow v3 scheduler: %d requests, %d lookups (avg %.3f ms), "
            "%d hit tokens, "
            "%d load specs, %d save specs",
            s["requests"],
            s["lookups"],
            s["lookup_ms"] / max(s["lookups"], 1),
            s["hit_tokens"],
            s["load_specs"],
            s["save_specs"],
        )
        _write_json(
            self.cfg.stats_path.format(role="scheduler", rank=0),
            {
                "config": self.cfg.as_dict(),
                "stats": dict(s),
                "hook_ms": dict(self._hook_ms),
                "hook_calls": dict(self._hook_calls),
            },
        )

    # ======================================================================
    # Worker side
    # ======================================================================

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        if self.engine is None:
            return
        names = natural_layer_order(list(kv_caches))
        first = kv_caches[names[0]]
        accessor = KVAccessor(first, self.block_size)
        for n in names[1:]:
            t = kv_caches[n]
            if t.shape != first.shape or t.dtype != first.dtype:
                logger.warning(
                    "CacheFlow v3: non-uniform KV layers (%s vs %s); disabled",
                    tuple(t.shape),
                    tuple(first.shape),
                )
                self._enabled = False
                return
        self._kv = kv_caches
        self._layer_idx = {n: i for i, n in enumerate(names)}
        self.engine.set_geometry(accessor, len(names), self.block_size)

    def _meta(self) -> CacheFlowV3Metadata | None:
        if not self.has_connector_metadata():
            return None
        m = self._get_connector_metadata()
        return m if isinstance(m, CacheFlowV3Metadata) else None

    @_timed
    def start_load_kv(self, forward_context: ForwardContext, **kwargs: Any) -> None:
        meta = self._meta()
        if self._profile and self.engine is not None:
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            self._prof_cur = {
                "t0": time.perf_counter(),
                "ev0": ev,
                "loads": len(meta.loads) if meta else 0,
                "saves": len(meta.saves) if meta else 0,
            }
        if meta is None or not meta.loads or self.engine is None or not self._enabled:
            return
        for rid, spec in meta.loads.items():
            try:
                self._loads[rid] = self.engine.begin_load(
                    rid,
                    spec.lease,
                    spec.entries[self.rank],
                    spec.rkey,
                    spec.chunk0,
                    spec.tok_start,
                    spec.tok_stop,
                    spec.block_ids,
                )
            except Exception as e:
                logger.warning("CacheFlow v3: load for %s could not start: %s", rid, e)
                B = self.block_size
                self._load_errors.update(
                    spec.block_ids[t // B]
                    for t in range(spec.tok_start, spec.tok_stop)
                    if t // B < len(spec.block_ids)
                )
                if spec.lease is not None:
                    self.engine._bg.submit(self.engine._release_lease, spec.lease)

    @_timed
    def wait_for_layer_load(self, layer_name: str) -> None:
        if not self._loads:
            return
        li = self._layer_idx.get(layer_name)
        if li is None:
            return
        kv = self._kv[layer_name]
        for rid, job in self._loads.items():
            if job.failed:
                continue
            try:
                self.engine.load_layer(job, li, kv)
            except Exception as e:
                logger.warning(
                    "CacheFlow v3: load of %s failed at layer %d: %s", rid, li, e
                )
                job.failed = True
                self._load_errors.update(job.failed_blocks)
        if li == len(self._layer_idx) - 1:
            self._finish_loads()

    def _finish_loads(self) -> None:
        for job in self._loads.values():
            if not job.failed and job.layers_done < len(self._layer_idx):
                job.failed = True  # some layers were never injected
                self._load_errors.update(job.failed_blocks)
            self.engine.finish_load(job)
        self._loads.clear()

    @_timed
    def save_kv_layer(
        self, layer_name: str, kv_layer: torch.Tensor, attn_metadata: Any, **kwargs: Any
    ) -> None:
        meta = self._meta()
        if meta is None or not meta.saves or self.engine is None or not self._enabled:
            return
        li = self._layer_idx.get(layer_name)
        if li is None:
            return
        C = self.cfg.chunk_tokens
        for rid, spec in meta.saves.items():
            job = self._saves.get(rid)
            if job is None:
                if rid in self._save_skipped:
                    continue
                n = len(spec.keys)
                t = torch.arange(
                    spec.chunk0 * C, (spec.chunk0 + n) * C, dtype=torch.int64
                )
                table = torch.tensor(spec.block_ids, dtype=torch.int64)
                if int(t[-1]) // self.block_size >= len(spec.block_ids):
                    logger.warning(
                        "CacheFlow v3: save for %s lacks blocks; skipped", rid
                    )
                    self._save_skipped.add(rid)
                    continue
                dev = kv_layer.device
                blk = table[t // self.block_size].view(n, C).to(dev, non_blocking=True)
                off = (t % self.block_size).view(n, C).to(dev, non_blocking=True)
                job = self.engine.begin_save(rid, spec.keys, blk, off)
                if job is None:
                    self._save_skipped.add(rid)
                    continue
                self._saves[rid] = job
                self._step_saves.append(job)
            self.engine.save_layer(job, li, kv_layer)

    @_timed
    def wait_for_save(self):
        if self.engine is None:
            return
        if self._prof_cur is not None:
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            self._prof_cur.update(t1=time.perf_counter(), ev1=ev)
            self._prof_steps.append(self._prof_cur)
            self._prof_cur = None
        if self._loads:  # a forward that skipped some layers
            self._finish_loads()
        jobs, self._step_saves = self._step_saves, []
        for job in jobs:
            self.engine.end_save_step(job)
        if jobs and not self.cfg.async_transfers:
            self.engine.drain_saves(jobs, self.cfg.transfer_timeout_ms / 1000)
        self._saves.clear()
        self._save_skipped.clear()
        now = time.monotonic()
        if now - self._last_stats_write >= self.cfg.stats_log_interval_s:
            self._last_stats_write = now
            self._log_worker_stats()

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[set[str] | None, set[str] | None]:
        return None, None

    def get_block_ids_with_load_errors(self) -> set[int]:
        errs, self._load_errors = self._load_errors, set()
        return errs

    def _log_worker_stats(self) -> None:
        s = self.engine.snapshot()
        gbs = lambda b, ms: b / ms / 1e6 if ms else 0.0  # noqa: E731
        logger.info(
            "CacheFlow v3 worker r%d: loads %d (fail %d, %.2f GB, avg wait %.1f ms, "
            "%.2f GB/s), saves %d (fail %d, %.2f GB, new/exists/full chunks %d/%d/%d, "
            "%.2f GB/s)",
            self.rank,
            s["loads"],
            s["load_failures"],
            s["load_bytes"] / 1e9,
            s["load_wait_ms"] / max(s["loads"], 1),
            gbs(s["load_bytes"], s["load_total_ms"]),
            s["saves"],
            s["save_failures"],
            s["save_bytes"] / 1e9,
            s["save_chunks_new"],
            s["save_chunks_exists"],
            s["save_chunks_full"],
            gbs(s["save_bytes"], s["save_rdma_ms"]),
        )
        logger.info(
            "CacheFlow v3 worker r%d hook CPU time (ms): %s",
            self.rank,
            {k: round(v, 1) for k, v in self._hook_ms.items()},
        )
        self.engine.write_stats(
            self.cfg.stats_path.format(role="worker", rank=self.rank),
            extra={
                "hook_ms": dict(self._hook_ms),
                "hook_calls": dict(self._hook_calls),
            },
        )

    def _dump_profile(self) -> None:
        rows = []
        torch.cuda.synchronize()
        for st in self._prof_steps:
            rows.append(
                {
                    "loads": st["loads"],
                    "saves": st["saves"],
                    "t0": st["t0"],
                    "t1": st["t1"],
                    "cpu_ms": (st["t1"] - st["t0"]) * 1e3,
                    "gpu_ms": st["ev0"].elapsed_time(st["ev1"]),
                }
            )
        _write_json(
            self.cfg.stats_path.format(role="profile", rank=self.rank), {"steps": rows}
        )

    def shutdown(self):
        if self._profile and self._prof_steps:
            self._dump_profile()
        if self.engine is not None:
            self._log_worker_stats()
            self.engine.close()
            self.engine = None
        if self._sched_session is not None:
            self._maybe_write_sched_stats(force=True)
            self._sched_session.close()
            self._sched_session = None
        self._sched_bg.shutdown(wait=False)


def _write_json(path: str, obj: dict) -> None:
    import json
    import os

    try:
        tmp = f"{path}.tmp"
        with open(tmp, "w") as f:
            json.dump(obj, f, indent=1)
        os.replace(tmp, path)
    except OSError:
        pass
