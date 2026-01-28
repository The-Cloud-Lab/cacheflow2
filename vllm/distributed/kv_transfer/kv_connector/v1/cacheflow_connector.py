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
        self._kv_caches: dict[str, torch.Tensor] = {}
        self._hash_to_dpu_block: dict[str, int] = {}
        self._stats = {
            "save_attempts": 0,
            "save_success": 0,
            "save_fail": 0,
            "load_attempts": 0,
            "load_success": 0,
            "load_fail": 0,
        }

        self._prefix_map_path = self._get_prefix_map_path(extra_config)
        self._load_prefix_map()

        logger.info(
            f"CacheFlowConnectorV1 initialized: role={role.name}, "
            f"num_layers={self.num_layers}, kv_block_size={self.kv_block_size}, "
            f"tokens_per_block={self.tokens_per_block}"
        )

    def _assign_dpu_block_id(self, prefix_hash: str) -> int:
        """Deterministically map a prefix hash to a DPU block ID."""
        return int(prefix_hash[:8], 16) % self.max_blocks

    def _get_prefix_map_path(self, extra_config: dict[str, Any]) -> str:
        path = extra_config.get("prefix_map_path")
        if path:
            return path
        return os.path.join("/tmp", "cacheflow_prefix_map.json")

    def _load_prefix_map(self) -> None:
        if not os.path.exists(self._prefix_map_path):
            return
        try:
            with open(self._prefix_map_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                self._hash_to_dpu_block = {
                    str(k): int(v) for k, v in data.items()
                }
                logger.info(
                    f"[CacheFlow] Loaded {len(self._hash_to_dpu_block)} prefix mappings "
                    f"from {self._prefix_map_path}"
                )
        except Exception as e:
            logger.warning(f"[CacheFlow] Failed to load prefix map: {e}")

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

            # Compute prefix hash
            prefix_hash = compute_prefix_hash(token_ids, len(token_ids))

            # Create tracker for new requests
            if req_id not in self._request_trackers:
                self._request_trackers[req_id] = RequestTracker(
                    req_id=req_id,
                    prompt_len=len(token_ids),
                    token_ids=token_ids.copy(),
                    allocated_block_ids=[],
                    prefix_hash=prefix_hash,
                )
                logger.info(f"[CacheFlow] New request tracked: {req_id}, {len(token_ids)} tokens, hash={prefix_hash[:8]}...")
            else:
                self._request_trackers[req_id].prefix_hash = prefix_hash

            tracker = self._request_trackers[req_id]
            dpu_block_id = self._hash_to_dpu_block.get(prefix_hash)
            if dpu_block_id is not None:
                tracker.is_loading = True
                tracker.dpu_block_id = dpu_block_id
                logger.info(
                    f"[CacheFlow] Prefix hit for {req_id}: hash={prefix_hash[:8]}..., "
                    f"loading from DPU block {dpu_block_id}"
                )
                return 0, False

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
                        load_specs[req_id] = LoadSpec(
                            vllm_cached_tokens=0,
                            dpu_cached_tokens=len(tracker.token_ids),
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
                            load_specs[req_id] = LoadSpec(
                                vllm_cached_tokens=0,
                                dpu_cached_tokens=len(tracker.token_ids),
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

        for req_id, load_spec in metadata.load_specs.items():
            if not load_spec.can_load or load_spec.block_id is None:
                logger.info(f"[CacheFlow] Load spec for {req_id} not loadable, skipping")
                continue

            try:
                self._stats["load_attempts"] += 1
                row_index = metadata.request_row_indices.get(req_id, 0)
                for layer_name, kv_layer in self._kv_caches.items():
                    layer_attn = attn_metadata.get(layer_name) if isinstance(attn_metadata, dict) else None
                    block_table = self._get_block_table(layer_attn) if layer_attn is not None else None
                    if block_table is None or block_table.numel() == 0:
                        continue

                    row_index = self._resolve_row_index(
                        block_table, load_spec.block_ids, row_index
                    )
                    if row_index >= block_table.shape[0]:
                        logger.warning(
                            f"[CacheFlow] Row index {row_index} out of range for "
                            f"block_table shape {tuple(block_table.shape)} (req_id={req_id})"
                        )
                        continue

                    physical_block_id = int(block_table[row_index, 0].item())
                    dst_tensor = kv_layer[physical_block_id]

                    self._manager.fetch_tensor(load_spec.block_id, dst_tensor, sync=True)

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
                # Get block table from attention metadata
                block_table = self._get_block_table(attn_metadata)
                if block_table is None:
                    logger.info(f"[CacheFlow] No block_table in attn_metadata for {req_id}")
                    continue

                logger.info(f"[CacheFlow] block_table shape: {block_table.shape}")

                # Extract KV data for this request
                # block_table shape: [num_requests, max_blocks_per_request]
                if block_table.shape[0] > 0 and block_table.shape[1] > 0:
                    row_index = metadata.request_row_indices.get(req_id, 0)
                    row_index = self._resolve_row_index(
                        block_table, save_spec.block_ids, row_index
                    )
                    if row_index >= block_table.shape[0]:
                        logger.warning(
                            f"[CacheFlow] Row index {row_index} out of range for "
                            f"block_table shape {tuple(block_table.shape)} (req_id={req_id})"
                        )
                        continue

                    # Prefer scheduler-provided block IDs if available
                    if save_spec.block_ids:
                        physical_block_id = save_spec.block_ids[0]
                    else:
                        # Use per-request row mapping
                        physical_block_id = block_table[row_index, 0].item()
                    logger.info(f"[CacheFlow] physical_block_id: {physical_block_id}")

                    # Extract KV data
                    kv_block_data = kv_layer[physical_block_id]
                    logger.info(f"[CacheFlow] kv_block_data shape: {kv_block_data.shape}, dtype: {kv_block_data.dtype}")

                    # Compute hash for prefix caching
                    prefix_hash = compute_prefix_hash(
                        save_spec.token_ids,
                        len(save_spec.token_ids)
                    )

                    # Assign DPU block ID deterministically from prefix hash
                    dpu_block_id = self._assign_dpu_block_id(prefix_hash)

                    logger.info(f"[CacheFlow] Offloading {layer_name} for {req_id} to DPU block {dpu_block_id}, size: {kv_block_data.numel() * kv_block_data.element_size()} bytes")

                    # Offload to DPU
                    self._stats["save_attempts"] += 1
                    self._manager.offload_tensor(
                        block_id=dpu_block_id,
                        tensor=kv_block_data.contiguous(),
                        hash_key=prefix_hash,
                        sync=True,
                    )

                    self._hash_to_dpu_block[prefix_hash] = dpu_block_id
                    self._persist_prefix_map()
                    self._stats["save_success"] += 1

                    # Track saved layer
                    if req_id not in self._pending_saves:
                        self._pending_saves[req_id] = set()
                    self._pending_saves[req_id].add(layer_name)

                    logger.info(
                        f"[CacheFlow] Successfully saved {layer_name} for {req_id} to DPU block {dpu_block_id}"
                    )

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

    def wait_for_save(self):
        """Wait for all save operations to complete."""
        # Currently synchronous
        pass

    def get_finished(
        self,
        finished_req_ids: set[str],
    ) -> tuple[set[str] | None, set[str] | None]:
        """Return IDs of requests that finished transfers."""
        finished_saves = set()
        finished_loads = set()

        for req_id in finished_req_ids:
            if req_id in self._pending_saves:
                finished_saves.add(req_id)
                del self._pending_saves[req_id]

            if req_id in self._pending_loads:
                finished_loads.add(req_id)
                del self._pending_loads[req_id]

            # Clean up scheduler-side tracker
            with self._lock:
                self._request_trackers.pop(req_id, None)

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
        DOCABackendLoader.close()
        self._request_trackers.clear()
        self._pending_loads.clear()
        self._pending_saves.clear()

    def reset_cache(self) -> bool:
        """Reset the cache state."""
        with self._lock:
            self._request_trackers.clear()
            self._pending_loads.clear()
            self._pending_saves.clear()
            self._load_errors.clear()
        return True

    @classmethod
    def get_required_kvcache_layout(cls, vllm_config: VllmConfig) -> str | None:
        """Return required KV cache layout (None = any layout)."""
        return None
