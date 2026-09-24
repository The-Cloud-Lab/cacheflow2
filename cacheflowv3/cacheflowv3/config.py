# SPDX-License-Identifier: Apache-2.0
"""Connector configuration (from kv_transfer_config.kv_connector_extra_config).

Every design choice that the paper's ablations toggle is a flag here, and the
v1 option names are accepted as aliases so existing configs keep working.

  flag                          ablation it drives
  ----------------------------  ---------------------------------------------
  use_registered_buffer_pool    pre-registered staging pool vs pin+register
    (v1: use_doca_buffer_pool)  per transfer ("DOCA buffer pooling")
  async_transfers               saves complete in the background vs
                                wait_for_save() blocking until RDMA+commit
  overlap_dma_with_copy         post each layer's RDMA as soon as its GPU
                                gather finishes vs after the last layer
  layerwise_load                per-layer load completion + injection vs
                                wait for the whole prefix before layer 0
  transfer_mode                 "pull": GPU host NIC drives RDMA;
                                "push": the BlueField-3 drives RDMA
  data_path                     "mapped": GPU gathers straight into NIC-
                                registered memory (zero-copy on GB10);
                                "staged": GPU gather + copy-engine D2H/H2D
  enable_save / enable_load     isolate one direction (e.g. warm the pool,
                                then measure load-only)
  skip_save_if_prefix_cached    dedup saves of chunks already in the pool
  copy_stream_pool_size         CUDA streams for staged copies
  staging_pool_gb               staging memory (v1: num_staging_buffers x
                                block_size)
  dpu_capacity_gb               pool capacity used for this run
                                (v1: max_blocks x block_size)
  reset_dpu_cache_on_start      cold-cache runs (flush the pool at startup)
  chunk_tokens (v1: tokens_per_block)       hashing / caching granularity
  max_cached_tokens (v1: common_prefix_num_tokens)  cap per request
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .protocol import DEFAULT_PORT

GiB = 1 << 30

_ALIASES = {
    "use_doca_buffer_pool": "use_registered_buffer_pool",
    "tokens_per_block": "chunk_tokens",
    "common_prefix_num_tokens": "max_cached_tokens",
    "dpu_host": "server_host",
    "dpu_port": "server_port",
}


def _as_bool(v: Any) -> bool:
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "on")
    return bool(v)


@dataclass
class CacheFlowConfig:
    server_host: str = "10.0.1.2"
    server_port: int = DEFAULT_PORT
    connect_timeout_s: float = 10.0

    chunk_tokens: int = 256
    max_cached_tokens: int = 0  # 0 = no cap
    offload_full_prompt: bool = False  # v1 compat: True ignores max_cached_tokens
    min_cached_tokens: int = 0

    enable_save: bool = True  # False: never write to the pool (load-only runs)
    enable_load: bool = True  # False: never read from the pool (save-only runs)
    skip_save_if_prefix_cached: bool = True
    async_transfers: bool = True
    overlap_dma_with_copy: bool = True
    layerwise_load: bool = True
    transfer_mode: str = "pull"
    data_path: str = "mapped"
    use_registered_buffer_pool: bool = True
    staging_pool_gb: float = (
        4.0  # must hold one request's prefix (20k tok x ~150 KB/tok = 3 GB)
    )
    copy_stream_pool_size: int = 4

    dpu_capacity_gb: float = 0.0  # 0 = leave the server's capacity alone
    reset_dpu_cache_on_start: bool = False
    lease_ms: int = 60_000
    transfer_timeout_ms: int = 30_000
    load_staging_wait_ms: int = 5_000

    stats_path: str = "/tmp/cacheflowv3_stats_{role}_{rank}.json"
    stats_log_interval_s: float = 30.0

    @classmethod
    def from_extra_config(cls, extra: dict[str, Any] | None) -> CacheFlowConfig:
        extra = dict(extra or {})
        for old, new in _ALIASES.items():
            if old in extra and new not in extra:
                extra[new] = extra[old]
        cfg = cls()
        for name, default in asdict(cfg).items():
            if name not in extra or extra[name] is None:
                continue
            v = extra[name]
            if isinstance(default, bool):
                v = _as_bool(v)
            elif isinstance(default, int):
                v = int(v)
            elif isinstance(default, float):
                v = float(v)
            setattr(cfg, name, v)

        # v1 compat: staging = num_staging_buffers x block_size,
        #            capacity = max_blocks x block_size
        if (
            "staging_pool_gb" not in extra
            and "num_staging_buffers" in extra
            and "block_size" in extra
        ):
            cfg.staging_pool_gb = (
                int(extra["num_staging_buffers"]) * int(extra["block_size"]) / GiB
            )
        if (
            "dpu_capacity_gb" not in extra
            and "max_blocks" in extra
            and "block_size" in extra
        ):
            cfg.dpu_capacity_gb = (
                int(extra["max_blocks"]) * int(extra["block_size"]) / GiB
            )
        if cfg.offload_full_prompt:
            cfg.max_cached_tokens = 0
        cfg.validate()
        return cfg

    def validate(self) -> None:
        if self.transfer_mode not in ("pull", "push"):
            raise ValueError(
                f"transfer_mode must be 'pull' or 'push', got {self.transfer_mode!r}"
            )
        if self.data_path not in ("mapped", "staged"):
            raise ValueError(
                f"data_path must be 'mapped' or 'staged', got {self.data_path!r}"
            )
        if self.chunk_tokens <= 0:
            raise ValueError("chunk_tokens must be positive")
        if self.copy_stream_pool_size <= 0:
            raise ValueError("copy_stream_pool_size must be positive")

    def cacheable_tokens(self, prompt_len: int) -> int:
        """Prompt tokens eligible for caching, rounded down to whole chunks."""
        n = (
            prompt_len
            if self.max_cached_tokens <= 0
            else min(prompt_len, self.max_cached_tokens)
        )
        n = (n // self.chunk_tokens) * self.chunk_tokens
        return n if n >= max(self.min_cached_tokens, 1) else 0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)
