# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
CacheFlowConnectorV1: KV Cache Offload to BlueField DPU via DOCA

This connector offloads KV cache to a BlueField DPU using DOCA DMA transfers.
Since RTX 4090 doesn't support GPUDirect RDMA, we use:
    GPU VRAM -> Host Pinned RAM (CUDA) -> DPU RAM (DOCA DMA)

Usage:
    vllm serve ... --kv-transfer-config '{
        "kv_connector": "CacheFlowConnectorV1",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {
            "dpu_pci_addr": "0000:03:00.0",
            "block_size": 67108864,
            "max_blocks": 256,
            "tokens_per_block": 1024
        }
    }'
"""

import hashlib
import math
from collections import OrderedDict
import logging
import sys
import threading
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional
import json
import os

import torch

from vllm.config import VllmConfig
from vllm.distributed.kv_events import BlockStored, KVCacheEvent
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.logger import init_logger
from vllm.v1.attention.backend import AttentionMetadata
from vllm.model_executor.layers.attention.mla_attention import MLACommonMetadata
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.outputs import KVConnectorOutput

if TYPE_CHECKING:
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)


def compute_prefix_hash(token_ids: list[int], num_tokens: int) -> str:
    """Compute a hash for a token prefix."""
    if num_tokens <= 0:
        prefix = ()
    else:
        prefix = tuple(token_ids[:num_tokens])
    return hashlib.sha256(str(prefix).encode()).hexdigest()[:16]


# ============================================================================
# Metadata Classes
# ============================================================================


@dataclass
class LoadSpec:
    """Specification for loading KV cache from DPU."""
    vllm_cached_tokens: int
    dpu_cached_tokens: int
    can_load: bool
    block_id: Optional[int] = None
    prefix_hash: Optional[str] = None
    block_ids: list[int] = field(default_factory=list)


@dataclass
class SaveSpec:
    """Specification for saving KV cache to DPU."""
    skip_leading_tokens: int
    can_save: bool
    req_id: str = ""
    token_ids: list[int] = field(default_factory=list)
    block_ids: list[int] = field(default_factory=list)


@dataclass
class CacheFlowConnectorMetadata(KVConnectorMetadata):
    """Metadata passed from scheduler to workers."""
    load_specs: dict[str, LoadSpec] = field(default_factory=dict)
    save_specs: dict[str, SaveSpec] = field(default_factory=dict)
    request_row_indices: dict[str, int] = field(default_factory=dict)


# ============================================================================
# Request Tracking
# ============================================================================


@dataclass
class RequestTracker:
    """Track per-request state for KV cache offloading."""
    req_id: str
    prompt_len: int
    token_ids: list[int]
    allocated_block_ids: list[int]
    num_saved_tokens: int = 0
    prefix_hash: Optional[str] = None
    cached_tokens: int = 0
    is_loading: bool = False
    is_saving: bool = False
    dpu_block_id: Optional[int] = None


# ============================================================================
# Lazy DOCA Backend Loader
# ============================================================================


class DOCABackendLoader:
    """Lazily loads the DOCA backend to avoid import errors if not available."""

    _manager = None
    _initialized = False
    _lock = threading.Lock()

    @classmethod
    def get_manager(
        cls,
        pci_addr: Optional[str],
        block_size: int,
        max_blocks: int,
        num_staging_buffers: int,
        async_transfers: bool,
    ):
        """Get or create the KVOffloadManager singleton."""
        with cls._lock:
            if cls._manager is not None:
                return cls._manager

            # Add DOCA_Backend to path
            doca_backend_path = Path(__file__).parent.parent.parent.parent.parent.parent / "DOCA_Backend" / "python"
            if doca_backend_path.exists() and str(doca_backend_path) not in sys.path:
                sys.path.insert(0, str(doca_backend_path))
                logger.info(f"Added DOCA_Backend path: {doca_backend_path}")

            try:
                from kv_offload_manager import KVOffloadManager
                cls._manager = KVOffloadManager(
                    pci_addr=pci_addr,
                    block_size=block_size,
                    max_blocks=max_blocks,
                    num_staging_buffers=num_staging_buffers,
                    async_transfers=async_transfers,
                )
                cls._initialized = True
                logger.info("DOCA KVOffloadManager initialized successfully")
                return cls._manager
            except ImportError as e:
                logger.error(f"Failed to import DOCA backend: {e}")
                raise RuntimeError(
                    "DOCA backend not available. Ensure DOCA_Backend is built "
                    "and libhost_provider.so is accessible."
                ) from e
            except Exception as e:
                logger.error(f"Failed to initialize DOCA manager: {e}")
                raise

    @classmethod
    def close(cls):
        """Close the manager if initialized."""
        with cls._lock:
            if cls._manager is not None:
                cls._manager.close()
                cls._manager = None
                cls._initialized = False


# ============================================================================
# Main Connector
# ============================================================================


class CacheFlowConnectorV1(KVConnectorBase_V1):
    """
    KV Cache Connector for BlueField DPU offloading via DOCA.

    This connector manages KV cache offloading from GPU to DPU:
    - Scheduler side: tracks requests, computes prefix hashes, determines cache hits
    - Worker side: loads/saves KV cache layers using DOCA DMA transfers
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: Optional["KVCacheConfig"] = None,
    ):
        super().__init__(vllm_config, role, kv_cache_config)

        # Extract configuration from kv_transfer_config
        extra_config = {}
        if self._kv_transfer_config is not None:
            extra_config = getattr(self._kv_transfer_config, 'kv_connector_extra_config', {}) or {}

        self.dpu_pci_addr = extra_config.get('dpu_pci_addr', None)
        self.block_size = extra_config.get('block_size', 64 * 1024 * 1024)  # 64MB
        self.max_blocks = extra_config.get('max_blocks', 256)
        self.num_staging_buffers = extra_config.get('num_staging_buffers', 4)
        self.async_transfers = extra_config.get('async_transfers', True)
        self.tokens_per_block = extra_config.get('tokens_per_block', 1024)
        self._skip_save_if_prefix_cached = bool(
            extra_config.get("skip_save_if_prefix_cached", True)
        )
        self._offload_full_prompt = bool(
            extra_config.get("offload_full_prompt", False)
        )
        self._min_cached_tokens = int(
            extra_config.get("min_cached_tokens", 0)
        )
        common_prefix_num_tokens = extra_config.get("common_prefix_num_tokens")
        if common_prefix_num_tokens is None:
            common_prefix_num_tokens = getattr(
                self._kv_transfer_config, "common_prefix_num_tokens", 0
            )
        self.common_prefix_num_tokens = int(common_prefix_num_tokens or 0)
        self._load_prefix_map_enabled = bool(
            extra_config.get("load_prefix_map", False)
        )
        self._prefix_map_mtime: float | None = None

        # Model configuration
        self.num_layers = vllm_config.model_config.get_num_layers(
            vllm_config.parallel_config
        )
        self.num_kv_heads = vllm_config.model_config.get_num_kv_heads(
            vllm_config.parallel_config
        )
        self.head_size = vllm_config.model_config.get_head_size()

        # KV cache block size
        if kv_cache_config is not None and hasattr(kv_cache_config, 'block_size'):
            self.kv_block_size = kv_cache_config.block_size
        elif vllm_config.cache_config is not None:
            self.kv_block_size = vllm_config.cache_config.block_size
        else:
            self.kv_block_size = 16
            logger.warning("Could not determine KV block size, defaulting to 16")

        # Initialize manager only on worker side to avoid duplicate initialization
        self._manager = None
        if role == KVConnectorRole.WORKER:
            self._manager = DOCABackendLoader.get_manager(
                pci_addr=self.dpu_pci_addr,
                block_size=self.block_size,
                max_blocks=self.max_blocks,
                num_staging_buffers=self.num_staging_buffers,
                async_transfers=self.async_transfers,
            )

        # Scheduler-side state
        self._request_trackers: dict[str, RequestTracker] = {}
        self._lock = threading.RLock()

        # Worker-side state
        self._load_errors: set[int] = set()
        self._pending_loads: dict[str, bool] = {}
        self._pending_saves: dict[str, set[str]] = {}  # req_id -> set of layer names
        self._pending_offloads: dict[str, set[int]] = {}
        self._save_batches: dict[str, dict[str, torch.Tensor]] = {}
        self._save_batch_cached_tokens: dict[str, int] = {}
        self._kv_caches: dict[str, torch.Tensor] = {}
        self._hash_to_dpu_block: dict[str, int] = {}
        self._block_id_to_hash: dict[int, str] = {}
        self._prefix_lru: OrderedDict[str, None] = OrderedDict()
        self._stats = {
            "save_attempts": 0,
            "save_success": 0,
            "save_fail": 0,
            "save_bytes_total": 0,
            "save_bytes_max": 0,
            "save_cached_tokens_total": 0,
            "save_cached_tokens_max": 0,
            "load_attempts": 0,
            "load_success": 0,
            "load_fail": 0,
        }

        self._prefix_map_path = self._get_prefix_map_path(extra_config)
        self._load_prefix_map(extra_config)

        logger.info(
            f"CacheFlowConnectorV1 initialized: role={role.name}, "
            f"num_layers={self.num_layers}, kv_block_size={self.kv_block_size}, "
            f"tokens_per_block={self.tokens_per_block}"
        )

    def _assign_dpu_block_id(self, prefix_hash: str) -> int:
        """Deterministically map a prefix hash to a DPU block ID."""
        return int(prefix_hash[:8], 16) % self.max_blocks

    def _record_prefix_use(self, prefix_hash: str) -> None:
        self._prefix_lru.pop(prefix_hash, None)
        self._prefix_lru[prefix_hash] = None

    def _evict_prefix(self, prefix_hash: str) -> None:
        block_id = self._hash_to_dpu_block.pop(prefix_hash, None)
        if block_id is not None:
            self._block_id_to_hash.pop(block_id, None)
        self._prefix_lru.pop(prefix_hash, None)

    def _assign_block_for_prefix(self, prefix_hash: str) -> tuple[int, list[str]]:
        """Assign a block for a prefix hash, avoiding collisions."""
        with self._lock:
            existing = self._hash_to_dpu_block.get(prefix_hash)
            if existing is not None:
                self._record_prefix_use(prefix_hash)
                return existing, []

            base = self._assign_dpu_block_id(prefix_hash)
            for offset in range(self.max_blocks):
                candidate = (base + offset) % self.max_blocks
                if candidate not in self._block_id_to_hash:
                    self._block_id_to_hash[candidate] = prefix_hash
                    self._hash_to_dpu_block[prefix_hash] = candidate
                    self._record_prefix_use(prefix_hash)
                    return candidate, []

            # No free blocks, evict the LRU prefix.
            evicted_hashes: list[str] = []
            if self._prefix_lru:
                evicted_hash = next(iter(self._prefix_lru))
                evicted_block = self._hash_to_dpu_block.get(evicted_hash)
                self._evict_prefix(evicted_hash)
                evicted_hashes.append(evicted_hash)
                if evicted_block is not None:
                    self._block_id_to_hash[evicted_block] = prefix_hash
                    self._hash_to_dpu_block[prefix_hash] = evicted_block
                    self._record_prefix_use(prefix_hash)
                    return evicted_block, evicted_hashes

            # Fallback: reuse deterministic mapping.
            self._block_id_to_hash[base] = prefix_hash
            self._hash_to_dpu_block[prefix_hash] = base
            self._record_prefix_use(prefix_hash)
            return base, evicted_hashes

    def _get_prefix_map_path(self, extra_config: dict[str, Any]) -> str:
        path = extra_config.get("prefix_map_path")
        if path:
            return path
        return os.path.join("/tmp", "cacheflow_prefix_map.json")

    def _load_prefix_map(self, extra_config: dict[str, Any]) -> None:
        if not extra_config.get("load_prefix_map", False):
            return
        if not os.path.exists(self._prefix_map_path):
            return
        try:
            self._prefix_map_mtime = os.path.getmtime(self._prefix_map_path)
            with open(self._prefix_map_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                self._hash_to_dpu_block = {
                    str(k): int(v) for k, v in data.items()
                }
                self._block_id_to_hash = {
                    int(v): str(k) for k, v in self._hash_to_dpu_block.items()
                }
                self._prefix_lru = OrderedDict(
                    (str(k), None) for k in self._hash_to_dpu_block.keys()
                )
                logger.info(
                    f"[CacheFlow] Loaded {len(self._hash_to_dpu_block)} prefix mappings "
                    f"from {self._prefix_map_path}"
                )
        except Exception as e:
            logger.warning(f"[CacheFlow] Failed to load prefix map: {e}")

    def _refresh_prefix_map_if_needed(self) -> None:
        if not self._load_prefix_map_enabled:
            return
        if not os.path.exists(self._prefix_map_path):
            return
        try:
            mtime = os.path.getmtime(self._prefix_map_path)
            if self._prefix_map_mtime is not None and mtime <= self._prefix_map_mtime:
                return
            self._prefix_map_mtime = mtime
            with open(self._prefix_map_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                self._hash_to_dpu_block = {
                    str(k): int(v) for k, v in data.items()
                }
                self._block_id_to_hash = {
                    int(v): str(k) for k, v in self._hash_to_dpu_block.items()
                }
                self._prefix_lru = OrderedDict(
                    (str(k), None) for k in self._hash_to_dpu_block.keys()
                )
                logger.info(
                    f"[CacheFlow] Refreshed prefix map "
                    f"({len(self._hash_to_dpu_block)} entries)"
                )
        except Exception as e:
            logger.warning(f"[CacheFlow] Failed to refresh prefix map: {e}")

    def _persist_prefix_map(self) -> None:
        try:
            tmp_path = f"{self._prefix_map_path}.tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self._hash_to_dpu_block, f)
            os.replace(tmp_path, self._prefix_map_path)
        except Exception as e:
            logger.warning(f"[CacheFlow] Failed to persist prefix map: {e}")

    def _resolve_row_index(
        self,
        block_table: torch.Tensor,
        block_ids: list[int],
        fallback_index: int,
    ) -> int:
        if not block_ids:
            return fallback_index
        target = block_ids[0]
        for idx in range(block_table.shape[0]):
            if int(block_table[idx, 0].item()) == target:
                return idx
        return fallback_index

    def _get_request_slot_mapping(
        self,
        attn_metadata: AttentionMetadata,
        layer_name: str,
        req_id: str,
        request_row_indices: dict[str, int],
    ) -> Optional[torch.Tensor]:
        meta = attn_metadata
        if isinstance(attn_metadata, dict):
            meta = attn_metadata.get(layer_name, attn_metadata)
        if meta is None:
            return None
        slot_mapping = getattr(meta, "slot_mapping", None)
        if slot_mapping is None:
            return None
        query_start_loc = getattr(meta, "query_start_loc", None)
        if query_start_loc is None:
            query_start_loc = getattr(meta, "query_start_loc_cpu", None)
        if query_start_loc is None:
            return slot_mapping
        row_index = request_row_indices.get(req_id, 0)
        if row_index + 1 >= int(query_start_loc.shape[0]):
            return slot_mapping
        start = int(query_start_loc[row_index].item())
        end = int(query_start_loc[row_index + 1].item())
        return slot_mapping[start:end]

    def _extract_kv_from_layer(
        self,
        kv_layer: torch.Tensor,
        slot_mapping: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        if isinstance(attn_metadata, MLACommonMetadata):
            num_pages, page_size = kv_layer.shape[0], kv_layer.shape[1]
            return kv_layer.reshape(num_pages * page_size, -1)[slot_mapping, ...]
        num_pages, page_size = kv_layer.shape[1], kv_layer.shape[2]
        return kv_layer.reshape(2, num_pages * page_size, -1)[:, slot_mapping, ...]

    def _allocate_kv_buffer(
        self,
        kv_layer: torch.Tensor,
        slot_mapping: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        if isinstance(attn_metadata, MLACommonMetadata):
            num_pages, page_size = kv_layer.shape[0], kv_layer.shape[1]
            flat_dim = kv_layer.reshape(num_pages * page_size, -1).shape[-1]
            shape = (int(slot_mapping.shape[0]), flat_dim)
        else:
            num_pages, page_size = kv_layer.shape[1], kv_layer.shape[2]
            flat_dim = kv_layer.reshape(2, num_pages * page_size, -1).shape[-1]
            shape = (2, int(slot_mapping.shape[0]), flat_dim)
        return torch.empty(shape, device=kv_layer.device, dtype=kv_layer.dtype)

    def _get_kv_cache_shape(
        self,
        kv_layer: torch.Tensor,
        slot_mapping: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> tuple[int, ...]:
        if isinstance(attn_metadata, MLACommonMetadata):
            num_pages, page_size = kv_layer.shape[0], kv_layer.shape[1]
            flat_dim = kv_layer.reshape(num_pages * page_size, -1).shape[-1]
            return (int(slot_mapping.shape[0]), flat_dim)
        num_pages, page_size = kv_layer.shape[1], kv_layer.shape[2]
        flat_dim = kv_layer.reshape(2, num_pages * page_size, -1).shape[-1]
        return (2, int(slot_mapping.shape[0]), flat_dim)

    def _get_layer_order(self) -> list[str]:
        if not self._kv_caches:
            return []
        return sorted(self._kv_caches.keys())

    def _inject_kv_into_layer(
        self,
        kv_layer: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> None:
        if isinstance(attn_metadata, MLACommonMetadata):
            num_pages, page_size = kv_layer.shape[0], kv_layer.shape[1]
            kv_layer.reshape(num_pages * page_size, -1)[slot_mapping, ...] = kv_cache
            return
        num_pages, page_size = kv_layer.shape[1], kv_layer.shape[2]
        kv_layer.reshape(2, num_pages * page_size, -1)[:, slot_mapping, ...] = kv_cache

    # ========================================================================
    # Scheduler-side methods
    # ========================================================================

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        """
        Check if request has cached KV on DPU.

        Returns:
            (num_matched_tokens, is_async): Number of tokens loadable from DPU
        """
        with self._lock:
            token_ids = self._extract_token_ids(request)
            if not token_ids:
                return 0, False

            req_id = getattr(request, 'request_id', str(id(request)))

            # Compute prefix hash (use common prefix length if configured)
            self._refresh_prefix_map_if_needed()
            hash_len = (
                len(token_ids)
                if self._offload_full_prompt
                else (self.common_prefix_num_tokens or len(token_ids))
            )
            cached_tokens = min(hash_len, len(token_ids))
            if cached_tokens <= num_computed_tokens:
                # Nothing new to load beyond local cache.
                if req_id in self._request_trackers:
                    tracker = self._request_trackers[req_id]
                    tracker.is_loading = False
                    tracker.dpu_block_id = None
                    tracker.cached_tokens = cached_tokens
                return 0, False

            prefix_hash = compute_prefix_hash(token_ids, cached_tokens)

            # Create tracker for new requests
            if req_id not in self._request_trackers:
                self._request_trackers[req_id] = RequestTracker(
                    req_id=req_id,
                    prompt_len=len(token_ids),
                    token_ids=token_ids.copy(),
                    allocated_block_ids=[],
                    prefix_hash=prefix_hash,
                    cached_tokens=cached_tokens,
                )
                logger.info(f"[CacheFlow] New request tracked: {req_id}, {len(token_ids)} tokens, hash={prefix_hash[:8]}...")
            else:
                self._request_trackers[req_id].prefix_hash = prefix_hash
                self._request_trackers[req_id].cached_tokens = cached_tokens

            tracker = self._request_trackers[req_id]
            dpu_block_id = self._hash_to_dpu_block.get(prefix_hash)
            if dpu_block_id is not None:
                self._record_prefix_use(prefix_hash)
                tracker.is_loading = True
                tracker.dpu_block_id = dpu_block_id
                logger.info(
                    f"[CacheFlow] Prefix hit for {req_id}: hash={prefix_hash[:8]}..., "
                    f"loading from DPU block {dpu_block_id}"
                )
                # Report matched tokens so vLLM's external hit rate reflects it.
                matched_tokens = cached_tokens - num_computed_tokens
                return matched_tokens, False

            # Clear any stale load state if the prefix is no longer mapped.
            if tracker.is_loading or tracker.dpu_block_id is not None:
                tracker.is_loading = False
                tracker.dpu_block_id = None

            return 0, False

    def _extract_token_ids(self, request: "Request") -> list[int]:
        """Extract token IDs from request object."""
        if hasattr(request, 'prompt_token_ids'):
            return list(request.prompt_token_ids)
        elif hasattr(request, 'get_token_ids'):
            return list(request.get_token_ids())
        elif hasattr(request, 'token_ids'):
            return list(request.token_ids)
        return []

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ):
        """Update state after vLLM allocates blocks."""
        with self._lock:
            req_id = getattr(request, 'request_id', str(id(request)))

            if req_id not in self._request_trackers:
                token_ids = self._extract_token_ids(request)
                self._request_trackers[req_id] = RequestTracker(
                    req_id=req_id,
                    prompt_len=len(token_ids),
                    token_ids=token_ids.copy(),
                    allocated_block_ids=[],
                )

            tracker = self._request_trackers[req_id]

            # Extract block IDs
            if hasattr(blocks, 'get_block_ids'):
                block_groups = blocks.get_block_ids()
                block_ids = list(block_groups[0]) if block_groups else []
            elif hasattr(blocks, 'block_ids'):
                block_ids = list(blocks.block_ids)
            elif isinstance(blocks, (list, tuple)):
                block_ids = list(blocks)
            else:
                block_ids = []

            tracker.allocated_block_ids = block_ids
            logger.info(f"[CacheFlow] Allocated blocks for {req_id}: {len(block_ids)} blocks")

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> CacheFlowConnectorMetadata:
        """Build metadata for this scheduling step."""
        with self._lock:
            # Keep prefix map in sync so we don't emit stale load specs.
            self._refresh_prefix_map_if_needed()
            load_specs: dict[str, LoadSpec] = {}
            save_specs: dict[str, SaveSpec] = {}
            request_row_indices: dict[str, int] = {}

            # scheduled_new_reqs is a list
            new_reqs = scheduler_output.scheduled_new_reqs or []
            # scheduled_cached_reqs is a CachedRequestData object with req_ids list
            cached_req_data = scheduler_output.scheduled_cached_reqs

            num_new_reqs = len(new_reqs)
            num_cached_reqs = len(cached_req_data.req_ids) if cached_req_data else 0

            if num_new_reqs > 0 or num_cached_reqs > 0:
                logger.info(f"[CacheFlow] build_connector_meta: {num_new_reqs} new reqs, {num_cached_reqs} cached reqs")

            # Process scheduled new requests
            for idx, req_data in enumerate(new_reqs):
                req_id = req_data.req_id
                request_row_indices[req_id] = idx

                if req_id in self._request_trackers:
                    tracker = self._request_trackers[req_id]

                    # Create save spec for offloading
                    save_specs[req_id] = SaveSpec(
                        skip_leading_tokens=tracker.num_saved_tokens,
                        can_save=True,
                        req_id=req_id,
                        token_ids=tracker.token_ids.copy(),
                        block_ids=tracker.allocated_block_ids.copy(),
                    )
                    logger.info(f"[CacheFlow] Created save_spec for {req_id}: {len(tracker.token_ids)} tokens, {len(tracker.allocated_block_ids)} blocks")

                    # Create load spec if we detected a prefix hit
                    if tracker.is_loading and tracker.dpu_block_id is not None:
                        # Revalidate mapping to avoid stale prefix hits.
                        current_block = self._hash_to_dpu_block.get(tracker.prefix_hash or "")
                        if current_block != tracker.dpu_block_id:
                            tracker.is_loading = False
                            tracker.dpu_block_id = None
                            continue
                        cached_tokens = tracker.cached_tokens or len(tracker.token_ids)
                        load_specs[req_id] = LoadSpec(
                            vllm_cached_tokens=0,
                            dpu_cached_tokens=cached_tokens,
                            can_load=True,
                            block_id=tracker.dpu_block_id,
                            prefix_hash=tracker.prefix_hash,
                            block_ids=tracker.allocated_block_ids.copy(),
                        )
                        logger.info(
                            f"[CacheFlow] Created load_spec for {req_id}: "
                            f"dpu_block_id={tracker.dpu_block_id}, hash={tracker.prefix_hash[:8]}..."
                        )

            # Process cached/running requests (CachedRequestData has parallel lists)
            if cached_req_data and cached_req_data.req_ids:
                for i, req_id in enumerate(cached_req_data.req_ids):
                    request_row_indices[req_id] = i
                    if req_id in self._request_trackers:
                        tracker = self._request_trackers[req_id]
                        # Update blocks if new ones allocated
                        if cached_req_data.new_block_ids and i < len(cached_req_data.new_block_ids):
                            new_blocks = cached_req_data.new_block_ids[i]
                            if new_blocks:
                                tracker.allocated_block_ids.extend(new_blocks[0])

                        if tracker.is_loading and tracker.dpu_block_id is not None and req_id not in load_specs:
                            # Revalidate mapping to avoid stale prefix hits.
                            current_block = self._hash_to_dpu_block.get(tracker.prefix_hash or "")
                            if current_block != tracker.dpu_block_id:
                                tracker.is_loading = False
                                tracker.dpu_block_id = None
                                continue
                            cached_tokens = tracker.cached_tokens or len(tracker.token_ids)
                            load_specs[req_id] = LoadSpec(
                                vllm_cached_tokens=0,
                                dpu_cached_tokens=cached_tokens,
                                can_load=True,
                                block_id=tracker.dpu_block_id,
                                prefix_hash=tracker.prefix_hash,
                                block_ids=tracker.allocated_block_ids.copy(),
                            )
                            logger.info(
                                f"[CacheFlow] Created load_spec for cached {req_id}: "
                                f"dpu_block_id={tracker.dpu_block_id}, hash={tracker.prefix_hash[:8]}..."
                            )

            if save_specs:
                logger.info(f"[CacheFlow] Returning metadata with {len(save_specs)} save_specs")

            return CacheFlowConnectorMetadata(
                load_specs=load_specs,
                save_specs=save_specs,
                request_row_indices=request_row_indices,
            )

    def update_connector_output(self, connector_output: KVConnectorOutput):
        """Update state from worker-side output."""
        # Handle finished requests
        pass

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        """Called when request finishes."""
        with self._lock:
            req_id = getattr(request, 'request_id', str(id(request)))

            if req_id in self._request_trackers:
                tracker = self._request_trackers[req_id]
                # Request finished, it will be offloaded in worker
                logger.debug(f"Request {req_id} finished")

            # Don't defer block freeing (synchronous offload)
            return False, None

    def take_events(self) -> Iterable[KVCacheEvent]:
        """Return KV cache events."""
        return ()

    # ========================================================================
    # Worker-side methods
    # ========================================================================

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        """Register KV caches with the connector."""
        self._kv_caches = kv_caches
        logger.info(f"[CacheFlow] Registered {len(kv_caches)} KV cache tensors: {list(kv_caches.keys())[:5]}...")

    def start_load_kv(
        self,
        forward_context: "ForwardContext",
        **kwargs: Any,
    ) -> None:
        """Start loading KV cache from DPU."""
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, CacheFlowConnectorMetadata):
            return

        if not metadata.load_specs:
            return

        if self._manager is None:
            logger.warning("[CacheFlow] DOCA manager not initialized, cannot load KV")
            return

        attn_metadata = forward_context.attn_metadata
        if isinstance(attn_metadata, list):
            attn_metadata = attn_metadata[0] if attn_metadata else {}

        layer_order = self._get_layer_order()
        for req_id, load_spec in metadata.load_specs.items():
            if not load_spec.can_load or load_spec.block_id is None:
                logger.info(f"[CacheFlow] Load spec for {req_id} not loadable, skipping")
                continue

            try:
                self._stats["load_attempts"] += 1
                if self._manager is not None and not self._manager.has_block(load_spec.block_id):
                    self._manager.wait_for_pending_transfers([load_spec.block_id])
                if self._manager is not None and not self._manager.has_block(load_spec.block_id):
                    logger.warning(
                        f"[CacheFlow] DPU block {load_spec.block_id} not present in "
                        f"current session; skipping load for {req_id}"
                    )
                    if load_spec.prefix_hash:
                        self._evict_prefix(load_spec.prefix_hash)
                        self._persist_prefix_map()
                    continue
                kv_layer_slices: list[tuple[str, torch.Tensor, torch.Tensor, tuple[int, ...]]] = []
                total_elems = 0
                for layer_name in layer_order:
                    kv_layer = self._kv_caches[layer_name]
                    slot_mapping = self._get_request_slot_mapping(
                        attn_metadata, layer_name, req_id, metadata.request_row_indices
                    )
                    if slot_mapping is None or slot_mapping.numel() == 0:
                        slot_mapping = None
                    cached_tokens = min(
                        load_spec.dpu_cached_tokens,
                        int(slot_mapping.numel()) if slot_mapping is not None else 0,
                    )
                    if cached_tokens <= 0:
                        cached_tokens = load_spec.dpu_cached_tokens
                    if cached_tokens <= 0:
                        continue
                    if slot_mapping is None or int(slot_mapping.numel()) < cached_tokens:
                        block_table = self._get_block_table(attn_metadata)
                        row_index = self._resolve_row_index(
                            block_table,
                            load_spec.block_ids,
                            metadata.request_row_indices.get(req_id, 0),
                        )
                        slot_mapping = self._build_prefix_slot_mapping(
                            block_table,
                            row_index,
                            cached_tokens,
                            kv_layer.device,
                        )
                        if slot_mapping is None or slot_mapping.numel() == 0:
                            logger.info(
                                f"[CacheFlow] No slot mapping for load {req_id}, skipping"
                            )
                            continue
                    slot_mapping = slot_mapping[:cached_tokens]
                    if slot_mapping.device != kv_layer.device:
                        slot_mapping = slot_mapping.to(kv_layer.device)

                    kv_shape = self._get_kv_cache_shape(
                        kv_layer, slot_mapping, attn_metadata
                    )
                    numel = int(math.prod(kv_shape))
                    kv_layer_slices.append((layer_name, kv_layer, slot_mapping, kv_shape))
                    total_elems += numel

                if total_elems <= 0:
                    continue

                first_layer = next(iter(self._kv_caches.values()))
                combined = torch.empty(
                    (total_elems,),
                    device=first_layer.device,
                    dtype=first_layer.dtype,
                )
                self._manager.fetch_tensor(load_spec.block_id, combined, sync=True)

                offset = 0
                for layer_name, kv_layer, slot_mapping, kv_shape in kv_layer_slices:
                    numel = int(math.prod(kv_shape))
                    kv_cache = combined[offset:offset + numel].view(kv_shape)
                    offset += numel
                    self._inject_kv_into_layer(
                        kv_layer, kv_cache, slot_mapping, attn_metadata
                    )

                self._pending_loads[req_id] = True
                self._stats["load_success"] += 1
                logger.info(f"[CacheFlow] Loaded KV for {req_id} from DPU block {load_spec.block_id}")
            except Exception as e:
                self._stats["load_fail"] += 1
                logger.error(f"[CacheFlow] Failed to load KV for {req_id}: {e}")
                self._load_errors.add(load_spec.block_id)

    def wait_for_layer_load(self, layer_name: str) -> None:
        """Wait for layer load to complete."""
        # Currently synchronous, no waiting needed
        return

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs: Any,
    ) -> None:
        """Save a layer's KV cache to DPU."""
        logger.info(f"[CacheFlow] save_kv_layer called for {layer_name}, kv_layer shape: {kv_layer.shape}")

        metadata = self._get_connector_metadata()
        if not isinstance(metadata, CacheFlowConnectorMetadata):
            logger.info(f"[CacheFlow] No CacheFlowConnectorMetadata, skipping save")
            return

        if not metadata.save_specs:
            logger.info(f"[CacheFlow] No save_specs in metadata, skipping save")
            return

        if not self._kv_caches:
            logger.info("[CacheFlow] No KV caches registered, skipping save")
            return

        if self._manager is None:
            logger.warning("[CacheFlow] DOCA manager not initialized, cannot save KV")
            return

        logger.info(f"[CacheFlow] Processing {len(metadata.save_specs)} save specs")

        # Process each request's save spec
        for req_id, save_spec in metadata.save_specs.items():
            if not save_spec.can_save:
                logger.info(f"[CacheFlow] Save spec for {req_id} has can_save=False, skipping")
                continue

            try:
                expected_cached_tokens = (
                    len(save_spec.token_ids)
                    if self._offload_full_prompt
                    else (self.common_prefix_num_tokens or len(save_spec.token_ids))
                )
                if expected_cached_tokens > 0:
                    prefix_hash = compute_prefix_hash(
                        save_spec.token_ids,
                        expected_cached_tokens,
                    )
                    if self._skip_save_if_prefix_cached:
                        cached_block = self._hash_to_dpu_block.get(prefix_hash)
                        if cached_block is not None and self._manager.has_block(cached_block):
                            logger.info(
                                f"[CacheFlow] Prefix already cached for {req_id}, "
                                f"skipping save to DPU block {cached_block}"
                            )
                            continue

                slot_mapping = self._get_request_slot_mapping(
                    attn_metadata, layer_name, req_id, metadata.request_row_indices
                )
                desired_tokens = (
                    len(save_spec.token_ids)
                    if self._offload_full_prompt
                    else (self.common_prefix_num_tokens or len(save_spec.token_ids))
                )
                cached_tokens = min(
                    desired_tokens,
                    int(slot_mapping.numel()) if slot_mapping is not None else 0,
                )
                if cached_tokens <= 0:
                    cached_tokens = desired_tokens
                if cached_tokens <= 0 or (
                    self._min_cached_tokens and cached_tokens < self._min_cached_tokens
                ):
                    continue
                if slot_mapping is None or int(slot_mapping.numel()) < cached_tokens:
                    block_table = self._get_block_table(attn_metadata)
                    row_index = self._resolve_row_index(
                        block_table,
                        save_spec.block_ids,
                        metadata.request_row_indices.get(req_id, 0),
                    )
                    slot_mapping = self._build_prefix_slot_mapping(
                        block_table,
                        row_index,
                        cached_tokens,
                        kv_layer.device,
                    )
                    if slot_mapping is None or slot_mapping.numel() == 0:
                        logger.info(f"[CacheFlow] No slot_mapping for {req_id}, skipping")
                        continue
                if cached_tokens <= 0:
                    continue
                slot_mapping = slot_mapping[:cached_tokens]
                if slot_mapping.device != kv_layer.device:
                    slot_mapping = slot_mapping.to(kv_layer.device)

                kv_block_data = self._extract_kv_from_layer(
                    kv_layer, slot_mapping, attn_metadata
                )
                if req_id not in self._save_batches:
                    self._save_batches[req_id] = {}
                self._save_batches[req_id][layer_name] = kv_block_data.contiguous()

                if req_id not in self._save_batch_cached_tokens:
                    self._save_batch_cached_tokens[req_id] = cached_tokens
                else:
                    self._save_batch_cached_tokens[req_id] = min(
                        self._save_batch_cached_tokens[req_id],
                        cached_tokens,
                    )

                layer_order = self._get_layer_order()
                if len(self._save_batches[req_id]) < len(layer_order):
                    continue

                cached_tokens_final = self._save_batch_cached_tokens[req_id]

                # Compute hash for prefix caching
                prefix_hash = compute_prefix_hash(
                    save_spec.token_ids,
                    cached_tokens_final,
                )

                # Assign DPU block ID with collision avoidance
                dpu_block_id, evicted_hashes = self._assign_block_for_prefix(
                    prefix_hash
                )

                # Concatenate all layers into one buffer for transfer
                ordered_blocks = [
                    (
                        self._save_batches[req_id][name][:cached_tokens_final, ...]
                        if self._save_batches[req_id][name].dim() == 2
                        else self._save_batches[req_id][name][:, :cached_tokens_final, ...]
                    ).reshape(-1)
                    for name in layer_order
                ]
                combined = torch.cat(ordered_blocks, dim=0)
                bytes_size = combined.numel() * combined.element_size()
                self._stats["save_bytes_total"] += bytes_size
                self._stats["save_bytes_max"] = max(
                    self._stats["save_bytes_max"],
                    bytes_size,
                )
                self._stats["save_cached_tokens_total"] += cached_tokens_final
                self._stats["save_cached_tokens_max"] = max(
                    self._stats["save_cached_tokens_max"],
                    cached_tokens_final,
                )

                logger.info(
                    f"[CacheFlow] Offloading batched KV for {req_id} to DPU block "
                    f"{dpu_block_id}, size: {bytes_size} bytes"
                )

                # Offload to DPU asynchronously
                self._stats["save_attempts"] += 1
                self._manager.offload_tensor(
                    block_id=dpu_block_id,
                    tensor=combined,
                    hash_key=prefix_hash,
                    sync=False,
                )

                self._persist_prefix_map()
                self._stats["save_success"] += 1

                if req_id not in self._pending_offloads:
                    self._pending_offloads[req_id] = set()
                self._pending_offloads[req_id].add(dpu_block_id)

                if req_id not in self._pending_saves:
                    self._pending_saves[req_id] = set()
                self._pending_saves[req_id].update(layer_order)

                del self._save_batches[req_id]
                del self._save_batch_cached_tokens[req_id]

            except Exception as e:
                self._stats["save_fail"] += 1
                logger.error(f"Failed to save KV for {req_id}: {e}")

    def _get_block_table(
        self,
        attn_metadata: AttentionMetadata
    ) -> Optional[torch.Tensor]:
        """Extract block table from attention metadata."""
        if hasattr(attn_metadata, 'block_table'):
            return attn_metadata.block_table
        elif hasattr(attn_metadata, 'block_tables'):
            return attn_metadata.block_tables
        return None

    def _build_prefix_slot_mapping(
        self,
        block_table: torch.Tensor,
        row_index: int,
        num_tokens: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        """Build a slot mapping for the first num_tokens using the block table."""
        if num_tokens <= 0:
            return None
        if block_table is None:
            return None
        if row_index < 0 or row_index >= int(block_table.shape[0]):
            return None

        # block_table contains block IDs per sequence; use kv_block_size to expand to slots.
        row = block_table[row_index]
        if row is None:
            return None
        block_ids = row.tolist()
        slot_indices: list[int] = []
        tokens_remaining = num_tokens
        for block_id in block_ids:
            if tokens_remaining <= 0:
                break
            if block_id is None or int(block_id) < 0:
                break
            base = int(block_id) * int(self.kv_block_size)
            take = min(tokens_remaining, int(self.kv_block_size))
            slot_indices.extend(range(base, base + take))
            tokens_remaining -= take

        if not slot_indices:
            return None
        return torch.tensor(slot_indices, device=device, dtype=torch.int32)

    def wait_for_save(self):
        """Wait for all save operations to complete."""
        if self._manager is None:
            return
        pending_block_ids: set[int] = set()
        for block_ids in self._pending_offloads.values():
            pending_block_ids.update(block_ids)
        if not pending_block_ids:
            return
        self._manager.wait_for_pending_transfers(list(pending_block_ids))
        self._pending_offloads.clear()

    def get_finished(
        self,
        finished_req_ids: set[str],
    ) -> tuple[set[str] | None, set[str] | None]:
        """Return IDs of requests that finished transfers."""
        extra_config = getattr(self._kv_transfer_config, "kv_connector_extra_config", None)
        if not isinstance(extra_config, dict):
            extra_config = {}
        report_finished = extra_config.get("report_finished_ids", False)
        finished_saves = set()
        finished_loads = set()

        for req_id in finished_req_ids:
            if req_id in self._pending_saves:
                finished_saves.add(req_id)
                del self._pending_saves[req_id]

            if req_id in self._pending_loads:
                finished_loads.add(req_id)
                del self._pending_loads[req_id]

            if req_id in self._pending_offloads:
                del self._pending_offloads[req_id]

            # Clean up scheduler-side tracker
            with self._lock:
                self._request_trackers.pop(req_id, None)

        if not report_finished:
            return (None, None)

        return (
            finished_saves if finished_saves else None,
            finished_loads if finished_loads else None,
        )

    def get_block_ids_with_load_errors(self) -> set[int]:
        """Return block IDs that failed to load."""
        return self._load_errors.copy()

    def shutdown(self):
        """Shutdown the connector."""
        logger.info("Shutting down CacheFlowConnectorV1")
        logger.info(
            "[CacheFlow] Stats: saves=%d/%d (fail=%d), loads=%d/%d (fail=%d)",
            self._stats["save_success"],
            self._stats["save_attempts"],
            self._stats["save_fail"],
            self._stats["load_success"],
            self._stats["load_attempts"],
            self._stats["load_fail"],
        )
        if self._stats["save_attempts"]:
            avg_bytes = self._stats["save_bytes_total"] / self._stats["save_attempts"]
            logger.info(
                "[CacheFlow] Save bytes: total=%d, avg=%.1f, max=%d",
                self._stats["save_bytes_total"],
                avg_bytes,
                self._stats["save_bytes_max"],
            )
            avg_tokens = (
                self._stats["save_cached_tokens_total"] / self._stats["save_attempts"]
            )
            logger.info(
                "[CacheFlow] Cached tokens: total=%d, avg=%.1f, max=%d",
                self._stats["save_cached_tokens_total"],
                avg_tokens,
                self._stats["save_cached_tokens_max"],
            )
        if self._manager is not None:
            try:
                mgr_stats = self._manager.get_stats()
                async_wait_ms_total = mgr_stats.get("async_wait_ms_total", 0.0)
                async_wait_ms_max = mgr_stats.get("async_wait_ms_max", 0.0)
                async_wait_count = mgr_stats.get("async_wait_count", 0)
                async_bytes_total = mgr_stats.get("async_bytes_total", 0)
                logger.info(
                    "[CacheFlow] Async waits: count=%d, total_ms=%.3f, max_ms=%.3f, bytes=%d",
                    async_wait_count,
                    async_wait_ms_total,
                    async_wait_ms_max,
                    async_bytes_total,
                )
            except Exception as e:
                logger.debug(f"[CacheFlow] Failed to fetch manager stats: {e}")
        self.wait_for_save()
        DOCABackendLoader.close()
        self._request_trackers.clear()
        self._pending_loads.clear()
        self._pending_saves.clear()
        self._pending_offloads.clear()
        self._save_batches.clear()
        self._save_batch_cached_tokens.clear()
        self._hash_to_dpu_block.clear()
        self._block_id_to_hash.clear()
        self._prefix_lru.clear()

    def reset_cache(self) -> bool:
        """Reset the cache state."""
        with self._lock:
            self._request_trackers.clear()
            self._pending_loads.clear()
            self._pending_saves.clear()
            self._load_errors.clear()
            self._pending_offloads.clear()
            self._save_batches.clear()
            self._save_batch_cached_tokens.clear()
            self._hash_to_dpu_block.clear()
            self._block_id_to_hash.clear()
            self._prefix_lru.clear()
        return True

    @classmethod
    def get_required_kvcache_layout(cls, vllm_config: VllmConfig) -> str | None:
        """Return required KV cache layout (None = any layout)."""
        return None
