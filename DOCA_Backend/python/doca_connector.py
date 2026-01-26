"""
DOCAConnectorV1: vLLM KVConnector for BlueField DPU Offload

Implements the vLLM KVConnectorBase_V1 interface, integrating the DOCA-based
KV offload system with vLLM's scheduler/worker architecture.

Architecture:
    Scheduler-side: Track requests, compute prefix hashes, determine cache hits
    Worker-side: Load/save KV cache layers during forward pass using DOCA DMA
"""

import threading
import hashlib
import logging
from typing import Any, Optional, Set, Dict, List, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1, KVConnectorMetadata, KVConnectorHandshakeMetadata, KVConnectorRole
)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.config import VllmConfig

# Import your offload manager
try:
    from .kv_offload_manager import KVOffloadManager, compute_prefix_hash
except ImportError:
    from kv_offload_manager import KVOffloadManager, compute_prefix_hash

logger = logging.getLogger(__name__)

# ---- Request Tracking ----
@dataclass
class RequestTracker:
    """Track per-request state for KV cache offloading."""
    req_id: str
    prompt_len: int
    token_ids: List[int]
    allocated_block_ids: List[int]
    num_saved_tokens: int = 0
    prefix_hash: Optional[str] = None
    is_decode_phase: bool = False
    
    def update_tokens(self, new_tokens: List[int]):
        """Add new tokens to the request."""
        self.token_ids.extend(new_tokens)
        
    def update_blocks(self, new_block_ids: List[int]):
        """Update allocated blocks."""
        self.allocated_block_ids = new_block_ids.copy()


@dataclass
class LoadSpec:
    """Specification for loading KV cache from DPU."""
    vllm_cached_tokens: int  # Tokens already in vLLM's cache
    dpu_cached_tokens: int   # Tokens available on DPU
    can_load: bool           # Whether scheduler allows loading
    block_id: Optional[int] = None  # DPU block to load from


@dataclass
class SaveSpec:
    """Specification for saving KV cache to DPU."""
    skip_leading_tokens: int  # Tokens already saved
    can_save: bool            # Whether scheduler allows saving
    

# ---- Metadata Classes ----
class DOCAKVConnectorMetadata(KVConnectorMetadata):
    """Metadata passed from scheduler to workers."""
    def __init__(self, 
                 load_specs: Dict[str, LoadSpec],
                 save_specs: Dict[str, SaveSpec]):
        self.load_specs = load_specs
        self.save_specs = save_specs

# ---- Main Connector ----
class DOCAConnectorV1(KVConnectorBase_V1):
    def __init__(self, vllm_config: VllmConfig, role: KVConnectorRole, kv_cache_config=None):
        super().__init__(vllm_config, role, kv_cache_config)
        
        # Extract configuration
        extra = self._kv_transfer_config.extra_config if hasattr(self._kv_transfer_config, 'extra_config') else {}
        dpu_pci_addr = extra.get('dpu_pci_addr', None)
        block_size = extra.get('block_size', 16 * 1024 * 1024)
        max_blocks = extra.get('max_blocks', 256)
        num_staging_buffers = extra.get('num_staging_buffers', 4)
        async_transfers = extra.get('async_transfers', True)
        self.tokens_per_block = extra.get('tokens_per_block', 1024)
        
        # Get model config for KV cache dimensions
        self.num_layers = vllm_config.model_config.get_num_layers(vllm_config.parallel_config)
        self.num_kv_heads = vllm_config.model_config.get_num_kv_heads(vllm_config.parallel_config)
        self.head_size = vllm_config.model_config.get_head_size()
        
        # KV cache block size from vLLM
        if kv_cache_config and hasattr(kv_cache_config, 'block_size'):
            self.kv_block_size = kv_cache_config.block_size
        elif hasattr(vllm_config, 'cache_config') and vllm_config.cache_config:
            self.kv_block_size = vllm_config.cache_config.block_size
        else:
            self.kv_block_size = 16  # Default fallback
            logger.warning("Could not determine block_size from config, defaulting to 16")
        
        # Initialize offload manager
        self._manager = KVOffloadManager(
            pci_addr=dpu_pci_addr,
            block_size=block_size,
            max_blocks=max_blocks,
            num_staging_buffers=num_staging_buffers,
            async_transfers=async_transfers,
        )
        
        # Request tracking (scheduler-side)
        self._request_trackers: Dict[str, RequestTracker] = {}
        self._lock = threading.RLock()
        
        # Worker-side state
        self._load_errors: Set[int] = set()
        self._pending_loads: Dict[str, Dict[str, torch.Tensor]] = {}  # req_id -> layer_name -> tensor
        self._pending_saves: Dict[str, Set[str]] = {}  # req_id -> set of layer_names
        
        logger.info(f"DOCAConnectorV1 initialized: role={role}, num_layers={self.num_layers}, "
                   f"kv_block_size={self.kv_block_size}, tokens_per_block={self.tokens_per_block}")

    # ---- Scheduler-side ----
    def get_num_new_matched_tokens(self, request, num_computed_tokens: int) -> Tuple[int, bool]:
        """
        Check if request has cached KV on DPU.
        
        Returns:
            (num_matched_tokens, is_async): Number of tokens that can be loaded from DPU
        """
        with self._lock:
            # Get token IDs from request
            token_ids = self._extract_token_ids(request)
            if not token_ids:
                return 0, False
            
            req_id = getattr(request, 'request_id', str(id(request)))
            logger.info(f"[DIAG] get_num_new_matched_tokens called for {req_id}, num_tokens={len(token_ids)}")
            
            # Compute prefix hash for the full prompt
            prefix_hash = compute_prefix_hash(token_ids, len(token_ids))
            
            # --- FIX START: Always create tracker for new requests ---
            if req_id not in self._request_trackers:
                self._request_trackers[req_id] = RequestTracker(
                    req_id=req_id,
                    prompt_len=len(token_ids),
                    token_ids=token_ids.copy(),
                    allocated_block_ids=[],
                    prefix_hash=prefix_hash
                )
                logger.info(f"[DIAG] Created tracker for {req_id}, hash={prefix_hash[:16]}...")
            # --- FIX END ---

            # Check if we have this prefix cached on DPU
            if self._manager.has_prefix(prefix_hash):
                block_id = self._manager.find_by_hash(prefix_hash)
                if block_id is not None:
                    # We have a cache hit! Return number of tokens available
                    matched_tokens = self.tokens_per_block
                    logger.debug(f"Cache hit for request {req_id}: {matched_tokens} tokens on DPU")
                    return matched_tokens, False  # Synchronous for now
            
            # No cache hit
            return 0, False
    
    def _extract_token_ids(self, request) -> List[int]:
        """Extract token IDs from request object."""
        if hasattr(request, 'prompt_token_ids'):
            return request.prompt_token_ids
        elif hasattr(request, 'get_token_ids'):
            return request.get_token_ids()
        elif hasattr(request, 'token_ids'):
            return request.token_ids
        return []

    def update_state_after_alloc(self, request, blocks, num_external_tokens: int):
        """Update state after vLLM allocates blocks for external tokens."""
        with self._lock:
            req_id = getattr(request, 'request_id', str(id(request)))
            
            if req_id in self._request_trackers:
                tracker = self._request_trackers[req_id]
                
                # Extract block IDs from blocks object
                if hasattr(blocks, 'block_ids'):
                    block_ids = blocks.block_ids
                elif isinstance(blocks, (list, tuple)):
                    block_ids = list(blocks)
                else:
                    block_ids = []
                
                tracker.update_blocks(block_ids)
                logger.debug(f"Updated blocks for request {req_id}: {len(block_ids)} blocks allocated")

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> DOCAKVConnectorMetadata:
        """
        Build metadata for this scheduling step.
        
        This metadata is sent to workers to guide KV cache loading/saving.
        """
        with self._lock:
            load_specs: Dict[str, LoadSpec] = {}
            save_specs: Dict[str, SaveSpec] = {}
            logger.info(f"[DIAG] build_connector_meta called, trackers={len(self._request_trackers)}")
            
            # Process scheduled requests
            for req_data in scheduler_output.scheduled_new_reqs:
                req_id = req_data.req_id
                logger.info(f"[DIAG] Processing scheduled req {req_id}")
                
                if req_id in self._request_trackers:
                    tracker = self._request_trackers[req_id]
                    
                    # Determine if we can load from DPU
                    if tracker.prefix_hash and self._manager.has_prefix(tracker.prefix_hash):
                        block_id = self._manager.find_by_hash(tracker.prefix_hash)
                        load_specs[req_id] = LoadSpec(
                            vllm_cached_tokens=req_data.num_computed_tokens,
                            dpu_cached_tokens=self.tokens_per_block,
                            can_load=True,
                            block_id=block_id
                        )
                    
                    # Determine if we should save to DPU
                    save_specs[req_id] = SaveSpec(
                        skip_leading_tokens=tracker.num_saved_tokens,
                        can_save=True
                    )
                    logger.info(f"[DIAG] Created save_spec for {req_id}")
                else:
                    logger.warning(f"[DIAG] No tracker found for {req_id}!")
            
            logger.info(f"[DIAG] Returning metadata with {len(save_specs)} save_specs")
            return DOCAKVConnectorMetadata(load_specs, save_specs)

    def request_finished(self, request, block_ids: List[int]) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Called when request finishes. Offload KV cache to DPU.
        
        Returns:
            (should_defer_free, kv_transfer_params)
        """
        with self._lock:
            req_id = getattr(request, 'request_id', str(id(request)))
            
            if req_id not in self._request_trackers:
                return False, None
            
            tracker = self._request_trackers[req_id]
            
            # Compute prefix hash if not already done
            if not tracker.prefix_hash:
                tracker.prefix_hash = compute_prefix_hash(tracker.token_ids, len(tracker.token_ids))
            
            # Mark for async offload
            # The actual offload happens in worker-side save_kv_layer
            logger.info(f"Request {req_id} finished, will offload to DPU")
            
            # Don't defer block freeing for now (synchronous offload)
            return False, None

    # ---- Worker-side ----
    def start_load_kv(self, forward_context, **kwargs):
        """
        Start loading KV cache from DPU to vLLM's paged buffer.
        
        This is called before the forward pass. We extract the metadata
        and initiate DMA transfers from DPU to GPU.
        """
        metadata = self._get_connector_metadata()
        if not metadata or not metadata.load_specs:
            return
        
        # Get attention metadata from forward context
        attn_metadata = forward_context.attn_metadata
        if isinstance(attn_metadata, dict):
            # V1 API: one metadata per layer
            # We'll handle loading in wait_for_layer_load
            pass
        else:
            # Single metadata for all layers
            self._load_kv_from_metadata(attn_metadata, metadata.load_specs)

    def wait_for_layer_load(self, layer_name: str):
        """
        Wait for a specific layer's KV cache to be loaded.
        
        For synchronous transfers, this is a no-op.
        For async transfers, this would wait for the DMA to complete.
        """
        # Currently synchronous, so nothing to wait for
        pass
    
    def _load_kv_from_metadata(self, attn_metadata, load_specs: Dict[str, LoadSpec]):
        """
        Load KV cache from DPU based on load specifications.
        
        This extracts the KV cache blocks from vLLM's paged buffer and
        fills them with data from the DPU.
        """
        for req_id, load_spec in load_specs.items():
            if not load_spec.can_load or load_spec.block_id is None:
                continue
            
            try:
                # Get the block table for this request from attention metadata
                # The block table maps logical blocks to physical blocks in KV cache
                block_table = self._extract_block_table(attn_metadata, req_id)
                if block_table is None:
                    continue
                
                # Fetch KV data from DPU into the allocated vLLM blocks
                # For now, we'll do this layer by layer in save_kv_layer
                # Store the load spec for later use
                if req_id not in self._pending_loads:
                    self._pending_loads[req_id] = {}
                
                logger.debug(f"Prepared load for request {req_id} from DPU block {load_spec.block_id}")
                
            except Exception as e:
                logger.error(f"Failed to load KV for request {req_id}: {e}")
                if load_spec.block_id:
                    self._load_errors.add(load_spec.block_id)
    
    def _extract_block_table(self, attn_metadata, req_id: str):
        """Extract block table from attention metadata."""
        logger.info(f"[DIAG] _extract_block_table: attn_metadata type={type(attn_metadata)}")
        
        # vLLM V1 uses 'block_table' (singular), not 'block_tables' (plural)
        if hasattr(attn_metadata, 'block_table'):
            bt = attn_metadata.block_table
            logger.info(f"[DIAG] Found block_table: type={type(bt)}, shape={bt.shape if hasattr(bt, 'shape') else 'N/A'}")
            return bt
        
        logger.warning(f"[DIAG] No block_table attribute found in attn_metadata")
        return None

    def save_kv_layer(self, layer_name: str, kv_layer: torch.Tensor, 
                     attn_metadata, **kwargs):
        """
        Save a layer's KV cache to DPU.
        
        Args:
            layer_name: Name of the attention layer
            kv_layer: The paged KV cache tensor for this layer
            attn_metadata: Attention metadata with block information
        """
        metadata = self._get_connector_metadata()
        logger.info(f"[DIAG] save_kv_layer called for {layer_name}, metadata={metadata}, save_specs={getattr(metadata, 'save_specs', None) if metadata else None}")
        if not metadata or not metadata.save_specs:
            logger.warning(f"[DIAG] No metadata or save_specs, skipping save for {layer_name}")
            return
        
        # kv_layer shape: [num_blocks, block_size, num_kv_heads, head_size]
        # We need to extract the relevant blocks for each request
        
        logger.info(f"[DIAG] Processing {len(metadata.save_specs)} save_specs in save_kv_layer")
        for req_id, save_spec in metadata.save_specs.items():
            logger.info(f"[DIAG] Processing save for req_id={req_id}, can_save={save_spec.can_save}")
            if not save_spec.can_save:
                logger.warning(f"[DIAG] Skipping {req_id} - can_save=False")
                continue
            
            try:
                # Get block table for this request
                logger.info(f"[DIAG] Extracting block table for {req_id}, attn_metadata type={type(attn_metadata)}")
                block_table = self._extract_block_table(attn_metadata, req_id)
                logger.info(f"[DIAG] block_table for {req_id}: {block_table}")
                if block_table is None:
                    logger.warning(f"[DIAG] block_table is None for {req_id}, skipping")
                    continue
                
                # Extract KV data for this request's blocks
                # For simplicity, we'll offload the first block
                # block_table is 2D: [num_requests, max_blocks_per_request]
                logger.info(f"[DIAG] block_table shape: {block_table.shape}")
                
                # Get the first request's block IDs (assuming single request or first in batch)
                # block_table[0] gives us all block IDs for the first request
                # block_table[0, 0] gives us the first physical block ID for that request
                if block_table.shape[0] > 0 and block_table.shape[1] > 0:
                    # Extract first block ID from first row
                    first_block_ids = block_table[0]  # Shape: [max_blocks]
                    physical_block_id = first_block_ids[0].item() if torch.is_tensor(first_block_ids[0]) else first_block_ids[0]
                    logger.info(f"[DIAG] physical_block_id={physical_block_id}, kv_layer shape={kv_layer.shape}")
                    
                    # Extract the KV data for this block
                    kv_block_data = kv_layer[physical_block_id]  # [block_size, num_kv_heads, head_size]
                    logger.info(f"[DIAG] kv_block_data shape={kv_block_data.shape}")
                    
                    # Compute hash for this request
                    if req_id in self._request_trackers:
                        tracker = self._request_trackers[req_id]
                        if not tracker.prefix_hash:
                            tracker.prefix_hash = compute_prefix_hash(tracker.token_ids, len(tracker.token_ids))
                        
                        # Offload to DPU
                        dpu_block_id = hash(req_id) % 1000  # Simple block ID assignment
                        logger.info(f"[DIAG] Calling offload_tensor: block_id={dpu_block_id}, hash={tracker.prefix_hash[:16]}...")
                        self._manager.offload_tensor(
                            block_id=dpu_block_id,
                            tensor=kv_block_data.contiguous(),
                            hash_key=tracker.prefix_hash,
                            sync=True
                        )
                        logger.info(f"[DIAG] offload_tensor completed successfully for {req_id}")
                        
                        # Track saved layers
                        if req_id not in self._pending_saves:
                            self._pending_saves[req_id] = set()
                        self._pending_saves[req_id].add(layer_name)
                        
                        logger.info(f"[DIAG] Saved layer {layer_name} for request {req_id} to DPU block {dpu_block_id}")
                    else:
                        logger.warning(f"[DIAG] req_id {req_id} not in _request_trackers!")
                else:
                    logger.warning(f"[DIAG] block_table is empty for {req_id}")
                
            except Exception as e:
                logger.error(f"[DIAG] Exception in save_kv_layer for {req_id}: {e}", exc_info=True)

    def wait_for_save(self):
        """Wait for all pending save operations to complete."""
        # Currently synchronous, so nothing to wait for
        # In async mode, we would wait for all pending DMA transfers
        pass

    def get_finished(self, finished_req_ids: Set[str]) -> Tuple[Optional[Set[str]], Optional[Set[str]]]:
        """
        Return IDs of requests that have finished async transfers.
        
        Returns:
            (finished_saves, finished_loads)
        """
        finished_saves = set()
        finished_loads = set()
        
        for req_id in finished_req_ids:
            if req_id in self._pending_saves:
                finished_saves.add(req_id)
                del self._pending_saves[req_id]
            
            if req_id in self._pending_loads:
                finished_loads.add(req_id)
                del self._pending_loads[req_id]
        
        return finished_saves if finished_saves else None, finished_loads if finished_loads else None

    def get_block_ids_with_load_errors(self) -> Set[int]:
        """Return block IDs that failed to load."""
        return self._load_errors.copy()

    def shutdown(self):
        """Shutdown the connector and cleanup resources."""
        logger.info("Shutting down DOCAConnectorV1")
        self._manager.close()
        self._request_trackers.clear()
        self._pending_loads.clear()
        self._pending_saves.clear()

    # ---- Additional Methods ----
    def _get_connector_metadata(self) -> Optional[DOCAKVConnectorMetadata]:
        """Get the current connector metadata (set by scheduler)."""
        return self._connector_metadata
    
    @classmethod
    def get_required_kvcache_layout(cls, vllm_config):
        """Specify required KV cache layout. None means any layout is acceptable."""
        return None

    def reset_cache(self):
        """Reset the cache state."""
        with self._lock:
            self._request_trackers.clear()
            self._pending_loads.clear()
            self._pending_saves.clear()
            self._load_errors.clear()
        return True
    
    def register_kv_caches(self, kv_caches: Dict[str, torch.Tensor]):
        """
        Register KV caches with the connector.
        
        This is called during initialization to give the connector
        access to the KV cache tensors.
        """
        # Store reference to KV caches if needed
        logger.debug(f"Registered {len(kv_caches)} KV cache tensors")


# Auto-register this connector when the module is imported
def _auto_register():
    """Auto-register DOCAConnectorV1 with vLLM's factory."""
    try:
        from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
        
        # Clear any existing registration to ensure we use the correct module path
        if 'DOCAConnectorV1' in KVConnectorFactory._registry:
            del KVConnectorFactory._registry['DOCAConnectorV1']
        
        # Use doca_connector_wrapper to avoid full path resolution issues
        module_path = 'doca_connector_wrapper'
        class_name = 'DOCAConnectorV1'
        
        KVConnectorFactory.register_connector(
            'DOCAConnectorV1',
            module_path,
            class_name
        )
    except Exception as e:
        # Silently fail if vLLM is not available
        pass

# Execute auto-registration when module is imported
_auto_register()


# End of doca_connector.py
