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
import struct
from collections import OrderedDict
import logging
import sys
import threading
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Dict, List, Tuple
import json
import os

# Try to import xxhash for faster hashing (10x speedup)
try:
    import xxhash
    HAS_XXHASH = True
except ImportError:
    HAS_XXHASH = False

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
    """
    Compute a fast hash for a token prefix.

    Uses xxhash if available (10x faster), otherwise falls back to
    optimized SHA256 with byte packing (still 2-3x faster than str()).
    """
    if num_tokens <= 0:
        return "empty_prefix_000"

    # Limit to actual tokens available
    num_tokens = min(num_tokens, len(token_ids))

    if HAS_XXHASH:
        # Fast path: xxhash with direct byte packing
        packed = struct.pack(f'{num_tokens}i', *token_ids[:num_tokens])
        return xxhash.xxh3_64_hexdigest(packed)[:16]
    else:
        # Fallback: SHA256 with byte packing (still faster than str())
        packed = struct.pack(f'{num_tokens}i', *token_ids[:num_tokens])
        return hashlib.sha256(packed).hexdigest()[:16]


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


@dataclass
class PendingLoad:
    """Track an in-progress async load operation."""
    req_id: str
    block_id: int
    handle: Any  # AsyncTransferHandle from kv_offload_manager
    layer_slices: List[Tuple[str, torch.Tensor, torch.Tensor, Tuple[int, ...]]]
    combined_buffer: torch.Tensor
    cached_tokens: int
    load_started: bool = False
    load_completed: bool = False


@dataclass
class PendingSave:
    """Track an in-progress async save operation."""
    req_id: str
    block_id: int
    prefix_hash: str
    handle: Any  # AsyncTransferHandle from kv_offload_manager
    cached_tokens: int


@dataclass
class SaveBatch:
    """Pre-allocated buffer for zero-copy layer batching."""
    combined: torch.Tensor
    layer_shapes: Dict[str, Tuple[int, ...]]
    layer_offsets: Dict[str, int]
    layers_written: set
    cached_tokens: int


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
        copy_stream_pool_size: int = 4,
        overlap_dma: bool = True,
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
                    copy_stream_pool_size=copy_stream_pool_size,
                    overlap_dma=overlap_dma,
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
        self.num_staging_buffers = extra_config.get('num_staging_buffers', 16)
        self.async_transfers = extra_config.get('async_transfers', True)
        self.copy_stream_pool_size = extra_config.get('copy_stream_pool_size', 4)
        self.overlap_dma = extra_config.get('overlap_dma_with_copy', True)
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
                copy_stream_pool_size=self.copy_stream_pool_size,
                overlap_dma=self.overlap_dma,
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

        # Async transfer state (Phase 3/4 optimizations)
        self._pending_load_handles: Dict[str, PendingLoad] = {}
        self._pending_save_handles: Dict[str, PendingSave] = {}
        self._zero_copy_batches: Dict[str, SaveBatch] = {}  # Zero-copy save batches
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
                # NOTE: Load path disabled - KV data layout mismatch causes corruption.
                # Saves still work, but we don't report matched tokens to avoid loading.
                # TODO: Fix KV cache layout/injection to match vLLM's expected format.
                logger.info(
                    f"[CacheFlow] Prefix exists for {req_id}: hash={prefix_hash[:8]}..., "
                    f"DPU block {dpu_block_id} (load disabled, save-only mode)"
                )
                # Don't set is_loading or report matched tokens - save-only mode
                return 0, False

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
        """
        Start loading KV cache from DPU asynchronously.

        Phase 4 optimization: Start async fetch and return immediately.
        Actual data injection happens in wait_for_layer_load().
        """
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

        # Store attn_metadata for wait_for_layer_load
        self._current_attn_metadata = attn_metadata

        layer_order = self._get_layer_order()
        for req_id, load_spec in metadata.load_specs.items():
            if not load_spec.can_load or load_spec.block_id is None:
                continue

            try:
                self._stats["load_attempts"] += 1

                # Check if block exists on DPU
                if not self._manager.has_block(load_spec.block_id):
                    self._manager.wait_for_pending_transfers([load_spec.block_id])
                if not self._manager.has_block(load_spec.block_id):
                    logger.warning(
                        f"[CacheFlow] DPU block {load_spec.block_id} not present; "
                        f"skipping load for {req_id}"
                    )
                    if load_spec.prefix_hash:
                        self._evict_prefix(load_spec.prefix_hash)
                        self._persist_prefix_map()
                    continue

                # Compute layer slices and total size
                kv_layer_slices: List[Tuple[str, torch.Tensor, torch.Tensor, Tuple[int, ...]]] = []
                total_elems = 0
                cached_tokens = load_spec.dpu_cached_tokens

                for layer_name in layer_order:
                    kv_layer = self._kv_caches[layer_name]
                    slot_mapping = self._get_request_slot_mapping(
                        attn_metadata, layer_name, req_id, metadata.request_row_indices
                    )
                    if slot_mapping is None or slot_mapping.numel() == 0:
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

                # Allocate combined destination buffer
                first_layer = next(iter(self._kv_caches.values()))
                combined = torch.empty(
                    (total_elems,),
                    device=first_layer.device,
                    dtype=first_layer.dtype,
                )

                # Start async fetch (returns immediately if async available)
                handle = None
                if hasattr(self._manager, 'fetch_tensor_async'):
                    handle = self._manager.fetch_tensor_async(
                        block_id=load_spec.block_id,
                        dst_tensor=combined,
                    )
                    logger.info(
                        f"[CacheFlow] Started async load for {req_id} from DPU block "
                        f"{load_spec.block_id}, {total_elems} elements"
                    )
                else:
                    # Fallback to sync fetch
                    self._manager.fetch_tensor(load_spec.block_id, combined, sync=True)
                    logger.info(
                        f"[CacheFlow] Sync loaded KV for {req_id} from DPU block "
                        f"{load_spec.block_id}"
                    )

                # Store pending load for wait_for_layer_load
                self._pending_load_handles[req_id] = PendingLoad(
                    req_id=req_id,
                    block_id=load_spec.block_id,
                    handle=handle,
                    layer_slices=kv_layer_slices,
                    combined_buffer=combined,
                    cached_tokens=cached_tokens,
                    load_started=True,
                    load_completed=(handle is None),  # Completed if sync fallback
                )

            except Exception as e:
                self._stats["load_fail"] += 1
                logger.error(f"[CacheFlow] Failed to start load for {req_id}: {e}")
                self._load_errors.add(load_spec.block_id)

    def wait_for_layer_load(self, layer_name: str) -> None:
        """
        Wait for layer load to complete and inject data.

        Phase 4 optimization: Complete async transfer and inject all layers
        at once on first call.
        """
        if not self._pending_load_handles:
            return

        attn_metadata = getattr(self, '_current_attn_metadata', None)
        if attn_metadata is None:
            return

        # Process all pending loads
        completed_reqs = []
        for req_id, pending in self._pending_load_handles.items():
            if not pending.load_started:
                continue

            try:
                # Complete async transfer if not done
                if not pending.load_completed and pending.handle is not None:
                    if hasattr(self._manager, 'wait_fetch_complete'):
                        self._manager.wait_fetch_complete(pending.handle)
                    elif hasattr(self._manager, 'complete_fetch_async'):
                        self._manager.complete_fetch_async(pending.handle)
                    pending.load_completed = True

                # Inject all layer data from combined buffer
                offset = 0
                for lname, kv_layer, slot_mapping, kv_shape in pending.layer_slices:
                    numel = int(math.prod(kv_shape))
                    kv_cache = pending.combined_buffer[offset:offset + numel].view(kv_shape)
                    offset += numel
                    self._inject_kv_into_layer(
                        kv_layer, kv_cache, slot_mapping, attn_metadata
                    )

                self._pending_loads[req_id] = True
                self._stats["load_success"] += 1
                logger.info(
                    f"[CacheFlow] Completed async load for {req_id} from DPU block "
                    f"{pending.block_id}"
                )
                completed_reqs.append(req_id)

            except Exception as e:
                self._stats["load_fail"] += 1
                logger.error(f"[CacheFlow] Failed to complete load for {req_id}: {e}")
                self._load_errors.add(pending.block_id)
                completed_reqs.append(req_id)

        # Cleanup completed loads
        for req_id in completed_reqs:
            self._pending_load_handles.pop(req_id, None)

    # ========================================================================
    # Zero-Copy Save Methods (Phase 3 Optimization)
    # ========================================================================

    def _init_zero_copy_batch(
        self,
        req_id: str,
        save_spec: SaveSpec,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> Optional[SaveBatch]:
        """
        Pre-allocate a combined buffer for all layers.
        This eliminates the torch.cat() overhead in the save path.
        """
        layer_order = self._get_layer_order()
        if not layer_order:
            return None

        # Determine cached_tokens from slot mapping or config
        slot_mapping = self._get_request_slot_mapping(
            attn_metadata, layer_order[0], req_id, {}
        )
        desired_tokens = (
            len(save_spec.token_ids)
            if self._offload_full_prompt
            else (self.common_prefix_num_tokens or len(save_spec.token_ids))
        )
        cached_tokens = min(
            desired_tokens,
            int(slot_mapping.numel()) if slot_mapping is not None else desired_tokens,
        )
        if cached_tokens <= 0:
            return None
        if self._min_cached_tokens and cached_tokens < self._min_cached_tokens:
            return None

        # Calculate total size needed for all layers
        total_elements = 0
        layer_shapes: Dict[str, Tuple[int, ...]] = {}
        layer_offsets: Dict[str, int] = {}

        for layer_name in layer_order:
            ref_kv = self._kv_caches.get(layer_name)
            if ref_kv is None:
                continue

            # Build slot mapping for this layer
            layer_slot = self._get_request_slot_mapping(
                attn_metadata, layer_name, req_id, {}
            )
            if layer_slot is None or layer_slot.numel() == 0:
                continue
            layer_slot = layer_slot[:cached_tokens]

            shape = self._get_kv_cache_shape(ref_kv, layer_slot, attn_metadata)
            numel = int(math.prod(shape))

            layer_offsets[layer_name] = total_elements
            layer_shapes[layer_name] = shape
            total_elements += numel

        if total_elements <= 0:
            return None

        # Allocate single combined buffer
        combined = torch.empty(
            (total_elements,),
            device=kv_layer.device,
            dtype=kv_layer.dtype,
        )

        return SaveBatch(
            combined=combined,
            layer_shapes=layer_shapes,
            layer_offsets=layer_offsets,
            layers_written=set(),
            cached_tokens=cached_tokens,
        )

    def _write_layer_to_zero_copy_batch(
        self,
        batch: SaveBatch,
        layer_name: str,
        kv_layer: torch.Tensor,
        slot_mapping: torch.Tensor,
        attn_metadata: AttentionMetadata,
    ) -> bool:
        """
        Write layer KV data directly into pre-allocated combined buffer.
        Returns True if this was the last layer.
        """
        if layer_name not in batch.layer_offsets:
            return False

        offset = batch.layer_offsets[layer_name]
        shape = batch.layer_shapes[layer_name]
        numel = int(math.prod(shape))

        # Truncate slot mapping to cached_tokens
        slot_mapping = slot_mapping[:batch.cached_tokens]
        if slot_mapping.device != kv_layer.device:
            slot_mapping = slot_mapping.to(kv_layer.device)

        # Extract KV data and write directly to combined buffer slice
        kv_data = self._extract_kv_from_layer(kv_layer, slot_mapping, attn_metadata)
        batch.combined[offset:offset + numel] = kv_data.reshape(-1)
        batch.layers_written.add(layer_name)

        layer_order = self._get_layer_order()
        return len(batch.layers_written) >= len(layer_order)

    def _offload_zero_copy_batch_async(
        self,
        req_id: str,
        save_spec: SaveSpec,
        batch: SaveBatch,
    ) -> Optional[PendingSave]:
        """
        Start async offload of the zero-copy batch to DPU.
        Returns PendingSave handle for tracking.
        """
        if self._manager is None:
            return None

        # Compute prefix hash
        prefix_hash = compute_prefix_hash(
            save_spec.token_ids,
            batch.cached_tokens,
        )

        # Check if prefix already cached
        if self._skip_save_if_prefix_cached:
            cached_block = self._hash_to_dpu_block.get(prefix_hash)
            if cached_block is not None and self._manager.has_block(cached_block):
                logger.info(
                    f"[CacheFlow] Zero-copy: Prefix already cached for {req_id}, "
                    f"skipping save to DPU block {cached_block}"
                )
                return None

        # Assign DPU block with collision avoidance
        dpu_block_id, evicted_hashes = self._assign_block_for_prefix(prefix_hash)

        bytes_size = batch.combined.numel() * batch.combined.element_size()
        self._stats["save_bytes_total"] += bytes_size
        self._stats["save_bytes_max"] = max(self._stats["save_bytes_max"], bytes_size)
        self._stats["save_cached_tokens_total"] += batch.cached_tokens
        self._stats["save_cached_tokens_max"] = max(
            self._stats["save_cached_tokens_max"], batch.cached_tokens
        )
        self._stats["save_attempts"] += 1

        logger.info(
            f"[CacheFlow] Zero-copy offload for {req_id} to DPU block {dpu_block_id}, "
            f"size: {bytes_size} bytes, tokens: {batch.cached_tokens}"
        )

        # Use async offload if available
        handle = None
        if hasattr(self._manager, 'offload_tensor_async'):
            handle = self._manager.offload_tensor_async(
                block_id=dpu_block_id,
                tensor=batch.combined,
                hash_key=prefix_hash,
            )
        else:
            # Fallback to sync offload
            self._manager.offload_tensor(
                block_id=dpu_block_id,
                tensor=batch.combined,
                hash_key=prefix_hash,
                sync=False,
            )

        self._persist_prefix_map()
        self._stats["save_success"] += 1

        if req_id not in self._pending_offloads:
            self._pending_offloads[req_id] = set()
        self._pending_offloads[req_id].add(dpu_block_id)

        return PendingSave(
            req_id=req_id,
            block_id=dpu_block_id,
            prefix_hash=prefix_hash,
            handle=handle,
            cached_tokens=batch.cached_tokens,
        )

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs: Any,
    ) -> None:
        """
        Save a layer's KV cache to DPU using zero-copy optimization.

        Phase 3 optimization: Pre-allocate combined buffer and write layers
        directly into it, eliminating torch.cat() overhead.
        """
        metadata = self._get_connector_metadata()
        if not isinstance(metadata, CacheFlowConnectorMetadata):
            return

        if not metadata.save_specs:
            return

        if not self._kv_caches:
            return

        if self._manager is None:
            logger.warning("[CacheFlow] DOCA manager not initialized, cannot save KV")
            return

        # Process each request's save spec
        for req_id, save_spec in metadata.save_specs.items():
            if not save_spec.can_save:
                continue

            try:
                # Early skip check for already-cached prefixes
                expected_cached_tokens = (
                    len(save_spec.token_ids)
                    if self._offload_full_prompt
                    else (self.common_prefix_num_tokens or len(save_spec.token_ids))
                )
                if expected_cached_tokens > 0 and self._skip_save_if_prefix_cached:
                    prefix_hash = compute_prefix_hash(
                        save_spec.token_ids,
                        expected_cached_tokens,
                    )
                    cached_block = self._hash_to_dpu_block.get(prefix_hash)
                    if cached_block is not None and self._manager.has_block(cached_block):
                        continue

                # Initialize zero-copy batch on first layer for this request
                if req_id not in self._zero_copy_batches:
                    batch = self._init_zero_copy_batch(
                        req_id, save_spec, kv_layer, attn_metadata
                    )
                    if batch is None:
                        continue
                    self._zero_copy_batches[req_id] = batch

                batch = self._zero_copy_batches[req_id]

                # Get slot mapping for this layer
                slot_mapping = self._get_request_slot_mapping(
                    attn_metadata, layer_name, req_id, metadata.request_row_indices
                )
                if slot_mapping is None or slot_mapping.numel() == 0:
                    block_table = self._get_block_table(attn_metadata)
                    row_index = self._resolve_row_index(
                        block_table,
                        save_spec.block_ids,
                        metadata.request_row_indices.get(req_id, 0),
                    )
                    slot_mapping = self._build_prefix_slot_mapping(
                        block_table,
                        row_index,
                        batch.cached_tokens,
                        kv_layer.device,
                    )
                    if slot_mapping is None or slot_mapping.numel() == 0:
                        continue

                # Write this layer directly to combined buffer (zero-copy)
                is_last_layer = self._write_layer_to_zero_copy_batch(
                    batch, layer_name, kv_layer, slot_mapping, attn_metadata
                )

                # On last layer, trigger async offload
                if is_last_layer:
                    pending = self._offload_zero_copy_batch_async(
                        req_id, save_spec, batch
                    )
                    if pending is not None:
                        self._pending_save_handles[req_id] = pending

                        if req_id not in self._pending_saves:
                            self._pending_saves[req_id] = set()
                        self._pending_saves[req_id].update(self._get_layer_order())

                    # Cleanup batch
                    del self._zero_copy_batches[req_id]

            except Exception as e:
                self._stats["save_fail"] += 1
                logger.error(f"Failed to save KV for {req_id}: {e}")
                # Cleanup on error
                self._zero_copy_batches.pop(req_id, None)

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
        # Cleanup async state
        self._pending_load_handles.clear()
        self._pending_save_handles.clear()
        self._zero_copy_batches.clear()

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
            # Cleanup async state
            self._pending_load_handles.clear()
            self._pending_save_handles.clear()
            self._zero_copy_batches.clear()
        return True

    @classmethod
    def get_required_kvcache_layout(cls, vllm_config: VllmConfig) -> str | None:
        """Return required KV cache layout (None = any layout)."""
        return None
