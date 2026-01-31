"""
KV Cache Offload Manager for vLLM/Cacheflow

This module provides the main integration layer between vLLM's KV cache
and the DOCA-based DPU offload system.

Similar to LMCache, this enables:
- Offloading KV cache blocks to DPU between requests
- Prefix caching across requests sharing common prompts
- Reducing GPU memory pressure for long sequences

Architecture:
    vLLM KV Cache (GPU) <-> KVOffloadManager <-> DOCA Client <-> DPU

Usage:
    from kv_offload_manager import KVOffloadManager

    manager = KVOffloadManager(pci_addr="0000:03:00.0")

    # Offload a KV block to DPU
    manager.offload_block(block_id=42, kv_tensor=tensor)

    # Later, fetch it back
    tensor = manager.fetch_block(block_id=42, dst_tensor=empty_tensor)

    manager.close()
"""

import threading
import time
from typing import Dict, Optional, Tuple, List, Any, TYPE_CHECKING
from dataclasses import dataclass, field
from enum import Enum
import logging

if TYPE_CHECKING:
    import torch

# Local imports
try:
    from .doca_cuda_utils import (
        PinnedBuffer, PinnedBufferPool, CUDARuntime,
        CUDAEvent, StreamEventPool
    )
    from .doca_kv_offload import DOCAKVOffloadClient, find_bluefield_pci_address
except ImportError:
    from doca_cuda_utils import (
        PinnedBuffer, PinnedBufferPool, CUDARuntime,
        CUDAEvent, StreamEventPool
    )
    from doca_kv_offload import DOCAKVOffloadClient, find_bluefield_pci_address

# Try to import torch
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logger = logging.getLogger(__name__)


class BlockState(Enum):
    """State of a KV cache block."""
    ON_GPU = "gpu"
    TRANSFERRING_TO_DPU = "to_dpu"
    ON_DPU = "dpu"
    TRANSFERRING_TO_GPU = "to_gpu"
    INVALID = "invalid"


@dataclass
class BlockInfo:
    """Metadata for a KV cache block."""
    block_id: int
    size: int
    state: BlockState
    buffer_id: Optional[int] = None  # DOCA buffer ID
    pinned_buffer: Optional[PinnedBuffer] = None
    refcount: int = 0  # Number of sequences using this block
    last_access: float = 0.0
    hash_key: Optional[str] = None  # For prefix matching
    pending_transfer_id: Optional[int] = None
    pending_transfer_start: Optional[float] = None
    pending_transfer_size: int = 0


@dataclass
class AsyncTransferHandle:
    """Handle for tracking async transfers with CUDA events."""
    block_id: int
    copy_stream: int
    copy_event: CUDAEvent
    doca_transfer_id: Optional[int] = None
    start_time: float = 0.0
    size_bytes: int = 0
    stage: str = "pending"  # pending, copying, transferring, complete
    pinned_buffer: Optional[PinnedBuffer] = None
    # For fetch operations
    dst_tensor: Optional[Any] = None  # torch.Tensor placeholder


class KVOffloadManager:
    """
    Manages KV cache offloading between GPU and DPU.

    This class coordinates:
    1. GPU to host pinned memory copies (CUDA)
    2. Host to DPU DMA transfers (DOCA)
    3. Block tracking and state management
    4. Prefix caching with hash-based lookup

    Similar in concept to LMCache but uses DPU instead of CPU memory.
    """

    def __init__(
        self,
        pci_addr: Optional[str] = None,
        block_size: int = 64 * 1024 * 1024,  # 64MB default block size
        max_blocks: int = 256,
        num_staging_buffers: int = 16,  # Increased from 4
        async_transfers: bool = True,
        copy_stream_pool_size: int = 4,  # NEW: pool size for async copies
        overlap_dma: bool = True,  # NEW: overlap GPU->Host with Host->DPU
    ):
        """
        Initialize the KV offload manager.

        Args:
            pci_addr: PCI address of BlueField DPU (auto-detect if None)
            block_size: Size of each KV block in bytes
            max_blocks: Maximum number of blocks to manage
            num_staging_buffers: Number of pinned buffers for staging
            async_transfers: Enable async transfer mode
        """
        self.block_size = block_size
        self.max_blocks = max_blocks
        self.async_transfers = async_transfers

        # Auto-detect DPU if not specified
        if pci_addr is None:
            pci_addr = find_bluefield_pci_address()
            if pci_addr is None:
                raise RuntimeError("No BlueField DPU found. Specify pci_addr manually.")
            logger.info(f"Auto-detected BlueField DPU at {pci_addr}")

        self.pci_addr = pci_addr

        # Initialize DOCA client
        self._doca_client = DOCAKVOffloadClient(pci_addr)

        # Buffer pool for GPU <-> Host staging
        self._buffer_pool = PinnedBufferPool(block_size, num_staging_buffers)

        # Block tracking
        self._blocks: Dict[int, BlockInfo] = {}
        self._hash_to_block: Dict[str, int] = {}  # Hash -> block_id for prefix matching
        self._lock = threading.RLock()

        # CUDA stream for async operations (legacy)
        if async_transfers:
            self._cuda_stream = CUDARuntime.stream_create()
        else:
            self._cuda_stream = None

        # NEW: Stream/Event pool for true async operations
        self._stream_event_pool = StreamEventPool(copy_stream_pool_size)
        self._overlap_dma = overlap_dma

        # NEW: Track pending async operations
        self._pending_async: Dict[int, AsyncTransferHandle] = {}

        # LRU tracking for eviction
        self._lru_order: List[int] = []
        self._async_wait_ms_total = 0.0
        self._async_wait_ms_max = 0.0
        self._async_wait_count = 0
        self._async_bytes_total = 0

        logger.info(
            f"KVOffloadManager initialized: block_size={block_size}, "
            f"max_blocks={max_blocks}, pci_addr={pci_addr}"
        )

    def _get_or_create_block(self, block_id: int, size: int) -> BlockInfo:
        """Get existing block info or create new one."""
        if block_id not in self._blocks:
            if len(self._blocks) >= self.max_blocks:
                # Need to evict
                self._evict_lru_block()

            self._blocks[block_id] = BlockInfo(
                block_id=block_id,
                size=size,
                state=BlockState.ON_GPU,
                last_access=time.time()
            )

        block = self._blocks[block_id]
        block.last_access = time.time()
        self._update_lru(block_id)
        return block

    def _update_lru(self, block_id: int) -> None:
        """Update LRU order for a block."""
        if block_id in self._lru_order:
            self._lru_order.remove(block_id)
        self._lru_order.append(block_id)

    def _evict_lru_block(self) -> None:
        """Evict the least recently used block."""
        with self._lock:
            for block_id in self._lru_order:
                block = self._blocks.get(block_id)
                if block and block.refcount == 0:
                    self._remove_block(block_id)
                    return

            # If all blocks are in use, raise error
            raise RuntimeError("Cannot evict: all blocks are in use")

    def _remove_block(self, block_id: int) -> None:
        """Remove a block from tracking."""
        if block_id not in self._blocks:
            return

        block = self._blocks[block_id]

        # Unregister from DOCA if registered
        if block.buffer_id is not None:
            try:
                self._doca_client.unregister_buffer(block.buffer_id)
            except Exception as e:
                logger.warning(f"Error unregistering buffer: {e}")

        # Release pinned buffer if held
        if block.pinned_buffer is not None:
            self._buffer_pool.release(block.pinned_buffer)

        # Remove from hash index
        if block.hash_key and block.hash_key in self._hash_to_block:
            del self._hash_to_block[block.hash_key]

        # Remove from tracking
        del self._blocks[block_id]
        if block_id in self._lru_order:
            self._lru_order.remove(block_id)

    def offload_block(
        self,
        block_id: int,
        gpu_ptr: int,
        size: int,
        hash_key: Optional[str] = None,
        sync: bool = True
    ) -> None:
        """
        Offload a KV cache block from GPU to DPU.

        Args:
            block_id: Unique block identifier
            gpu_ptr: GPU memory pointer to the KV data
            size: Size of the data in bytes
            hash_key: Optional hash for prefix matching
            sync: Wait for transfer completion
        """
        with self._lock:
            block = self._get_or_create_block(block_id, size)

            if block.state == BlockState.ON_DPU:
                logger.debug(f"Block {block_id} already on DPU")
                return

            block.state = BlockState.TRANSFERRING_TO_DPU

            # Acquire a staging buffer
            pinned_buf = self._buffer_pool.acquire()
            block.pinned_buffer = pinned_buf

            # Copy GPU -> Host (pinned)
            CUDARuntime.memcpy_dtoh(
                pinned_buf.address, gpu_ptr, size,
                stream=self._cuda_stream
            )

            if self._cuda_stream:
                CUDARuntime.stream_synchronize(self._cuda_stream)

            # Register with DOCA if not already
            if block.buffer_id is None:
                block.buffer_id = self._doca_client.register_buffer(
                    pinned_buf.address, size
                )

            # Transfer Host -> DPU
            transfer_id = self._doca_client.transfer(
                block.buffer_id, offset=0, length=size
            )

            if sync:
                self._doca_client.wait_transfer(transfer_id)
                block.state = BlockState.ON_DPU
                # Keep pinned buffer alive while data is on DPU.
                # DPU->Host load writes into this same buffer.
            else:
                # Store transfer_id for later completion check
                block.pending_transfer_id = transfer_id
                block.pending_transfer_start = time.perf_counter()
                block.pending_transfer_size = size

            # Update hash index for prefix matching
            if hash_key:
                block.hash_key = hash_key
                self._hash_to_block[hash_key] = block_id

            logger.debug(f"Block {block_id} offloaded to DPU (size={size})")

    def offload_tensor(
        self,
        block_id: int,
        tensor: "torch.Tensor",
        hash_key: Optional[str] = None,
        sync: bool = True
    ) -> None:
        """
        Offload a PyTorch tensor to DPU.

        Args:
            tensor: CUDA tensor containing KV data
            block_id: Block identifier
            hash_key: Optional hash for prefix matching
            sync: Wait for completion
        """
        if not HAS_TORCH:
            raise RuntimeError("PyTorch not available")

        if not tensor.is_cuda:
            raise ValueError("Tensor must be on CUDA device")

        # Ensure contiguous memory layout
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()

        size = tensor.numel() * tensor.element_size()
        self.offload_block(block_id, tensor.data_ptr(), size, hash_key, sync)

    def wait_for_pending_transfers(self, block_ids: Optional[List[int]] = None) -> None:
        """Wait for any pending offload transfers to complete."""
        with self._lock:
            targets = block_ids if block_ids is not None else list(self._blocks.keys())
            for block_id in targets:
                block = self._blocks.get(block_id)
                if block is None:
                    continue
                if block.pending_transfer_id is None:
                    continue
                try:
                    start = block.pending_transfer_start
                    self._doca_client.wait_transfer(block.pending_transfer_id)
                    if start is not None:
                        elapsed_ms = (time.perf_counter() - start) * 1000.0
                        self._async_wait_ms_total += elapsed_ms
                        self._async_wait_ms_max = max(
                            self._async_wait_ms_max, elapsed_ms
                        )
                        self._async_wait_count += 1
                        self._async_bytes_total += block.pending_transfer_size
                    block.pending_transfer_id = None
                    block.pending_transfer_start = None
                    block.pending_transfer_size = 0
                    block.state = BlockState.ON_DPU
                except Exception as e:
                    logger.warning(f"Failed waiting for transfer of block {block_id}: {e}")

    def fetch_block(
        self,
        block_id: int,
        gpu_ptr: int,
        size: int,
        sync: bool = True
    ) -> None:
        """
        Fetch a KV cache block from DPU back to GPU.

        Note: This requires DPU -> Host transfer first (future enhancement).
        Currently, we keep the data in the pinned buffer for faster access.

        Args:
            block_id: Block identifier
            gpu_ptr: Destination GPU pointer
            size: Size to fetch
            sync: Wait for completion
        """
        with self._lock:
            if block_id not in self._blocks:
                raise KeyError(f"Block {block_id} not found")

            block = self._blocks[block_id]

            if block.state == BlockState.ON_GPU:
                logger.debug(f"Block {block_id} already on GPU")
                return

            if block.state != BlockState.ON_DPU:
                raise RuntimeError(f"Block {block_id} in invalid state: {block.state}")

            if block.pinned_buffer is None:
                raise RuntimeError(
                    "Block data not in pinned buffer. "
                    "Cannot perform DPU->Host load."
                )

            block.state = BlockState.TRANSFERRING_TO_GPU

            # DPU -> Host (pinned)
            transfer_id = self._doca_client.load(
                block.buffer_id, offset=0, length=size
            )
            if sync:
                self._doca_client.wait_transfer(transfer_id)

            # Copy Host -> GPU
            CUDARuntime.memcpy_htod(
                gpu_ptr, block.pinned_buffer.address, size,
                stream=self._cuda_stream
            )

            if sync and self._cuda_stream:
                CUDARuntime.stream_synchronize(self._cuda_stream)

            block.state = BlockState.ON_GPU
            logger.debug(f"Block {block_id} fetched to GPU")

    def fetch_tensor(
        self,
        block_id: int,
        dst_tensor: "torch.Tensor",
        sync: bool = True
    ) -> None:
        """
        Fetch KV data from DPU back into a PyTorch CUDA tensor.
        """
        if not HAS_TORCH:
            raise RuntimeError("PyTorch not available")

        if not dst_tensor.is_cuda:
            raise ValueError("Destination tensor must be on CUDA")

        # Ensure the tensor is contiguous for DMA
        if not dst_tensor.is_contiguous():
            dst_tensor = dst_tensor.contiguous()

        # Calculate exact byte size (e.g., [16, 8, 128] * 2 bytes for bfloat16)
        size = dst_tensor.numel() * dst_tensor.element_size()
        
        # Trigger the DPU -> Host -> GPU pipeline defined in fetch_block
        self.fetch_block(block_id, dst_tensor.data_ptr(), size, sync)
    # =========================================================================
    # Async Transfer Methods (Optimized for TTFT)
    # =========================================================================

    def offload_tensor_async(
        self,
        block_id: int,
        tensor: "torch.Tensor",
        hash_key: Optional[str] = None,
    ) -> AsyncTransferHandle:
        """
        Non-blocking offload of a PyTorch tensor to DPU.

        This method returns immediately after starting the transfer pipeline:
        1. Start async GPU->Host copy
        2. Record CUDA event (no synchronization!)
        3. Start DOCA DMA transfer (overlapped with remaining copy if enabled)
        4. Return handle for later synchronization

        Args:
            block_id: Block identifier
            tensor: CUDA tensor containing KV data
            hash_key: Optional hash for prefix matching

        Returns:
            AsyncTransferHandle for tracking and completing the transfer
        """
        if not HAS_TORCH:
            raise RuntimeError("PyTorch not available")

        if not tensor.is_cuda:
            raise ValueError("Tensor must be on CUDA device")

        # Ensure contiguous memory layout
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()

        size = tensor.numel() * tensor.element_size()

        with self._lock:
            block = self._get_or_create_block(block_id, size)
            block.state = BlockState.TRANSFERRING_TO_DPU

            # Acquire stream/event from pool
            copy_stream, copy_event = self._stream_event_pool.acquire()

            # Acquire staging buffer
            pinned_buf = self._buffer_pool.acquire()
            block.pinned_buffer = pinned_buf

            # Async GPU->Host copy (NO synchronization!)
            CUDARuntime.memcpy_dtoh(
                pinned_buf.address, tensor.data_ptr(), size,
                stream=copy_stream
            )

            # Record event on copy stream - this marks copy completion
            copy_event.record(copy_stream)

            # Create handle
            handle = AsyncTransferHandle(
                block_id=block_id,
                copy_stream=copy_stream,
                copy_event=copy_event,
                start_time=time.perf_counter(),
                size_bytes=size,
                stage="copying",
                pinned_buffer=pinned_buf,
            )

            if self._overlap_dma:
                # Wait for copy and immediately start DMA (overlapped approach)
                copy_event.synchronize()
                self._start_dma_for_handle(block, handle, size, hash_key)
            else:
                # Store for later DMA start
                self._pending_async[block_id] = handle

            return handle

    def _start_dma_for_handle(
        self,
        block: BlockInfo,
        handle: AsyncTransferHandle,
        size: int,
        hash_key: Optional[str]
    ) -> None:
        """Start DOCA DMA transfer for a handle."""
        # Register buffer with DOCA if not already done
        if block.buffer_id is None:
            block.buffer_id = self._doca_client.register_buffer(
                handle.pinned_buffer.address, size
            )

        # Start Host->DPU transfer (async DOCA)
        handle.doca_transfer_id = self._doca_client.transfer(
            block.buffer_id, offset=0, length=size
        )
        handle.stage = "transferring"

        # Update hash index for prefix matching
        if hash_key:
            block.hash_key = hash_key
            self._hash_to_block[hash_key] = handle.block_id

        self._pending_async[handle.block_id] = handle

    def complete_offload_async(self, handle: AsyncTransferHandle) -> None:
        """
        Complete an async offload operation.

        Waits for both GPU->Host copy and Host->DPU DMA to finish.

        Args:
            handle: Handle from offload_tensor_async
        """
        with self._lock:
            # Ensure copy is complete
            if handle.stage == "copying":
                handle.copy_event.synchronize()
                block = self._blocks.get(handle.block_id)
                if block:
                    self._start_dma_for_handle(
                        block, handle, handle.size_bytes, block.hash_key
                    )

            # Wait for DMA
            if handle.doca_transfer_id is not None:
                self._doca_client.wait_transfer(handle.doca_transfer_id)

            # Update stats
            elapsed_ms = (time.perf_counter() - handle.start_time) * 1000.0
            self._async_wait_ms_total += elapsed_ms
            self._async_wait_ms_max = max(self._async_wait_ms_max, elapsed_ms)
            self._async_wait_count += 1
            self._async_bytes_total += handle.size_bytes

            # Update block state
            block = self._blocks.get(handle.block_id)
            if block:
                block.state = BlockState.ON_DPU

            handle.stage = "complete"

            # Return stream/event to pool
            self._stream_event_pool.release(handle.copy_stream, handle.copy_event)

            # Remove from pending
            self._pending_async.pop(handle.block_id, None)

    def fetch_tensor_async(
        self,
        block_id: int,
        dst_tensor: "torch.Tensor",
    ) -> AsyncTransferHandle:
        """
        Start async fetch from DPU to GPU tensor.

        This method returns immediately after starting the DPU->Host transfer.
        Call complete_fetch_async() to finish the Host->GPU copy.

        Args:
            block_id: Block identifier
            dst_tensor: Destination CUDA tensor

        Returns:
            AsyncTransferHandle for tracking and completing the fetch
        """
        if not HAS_TORCH:
            raise RuntimeError("PyTorch not available")

        if not dst_tensor.is_cuda:
            raise ValueError("Destination tensor must be on CUDA")

        size = dst_tensor.numel() * dst_tensor.element_size()

        with self._lock:
            if block_id not in self._blocks:
                raise KeyError(f"Block {block_id} not found")

            block = self._blocks[block_id]

            if block.state != BlockState.ON_DPU:
                raise RuntimeError(f"Block {block_id} not on DPU (state={block.state})")

            if block.pinned_buffer is None:
                raise RuntimeError("Block has no pinned buffer for DPU->Host load")

            # Acquire stream/event from pool
            copy_stream, copy_event = self._stream_event_pool.acquire()

            block.state = BlockState.TRANSFERRING_TO_GPU

            # Start DPU->Host transfer (async DOCA)
            doca_transfer_id = self._doca_client.load(
                block.buffer_id, offset=0, length=size
            )

            # Create handle
            handle = AsyncTransferHandle(
                block_id=block_id,
                copy_stream=copy_stream,
                copy_event=copy_event,
                doca_transfer_id=doca_transfer_id,
                start_time=time.perf_counter(),
                size_bytes=size,
                stage="dpu_to_host",
                pinned_buffer=block.pinned_buffer,
                dst_tensor=dst_tensor,
            )

            self._pending_async[block_id] = handle
            return handle

    def complete_fetch_async(self, handle: AsyncTransferHandle) -> None:
        """
        Complete an async fetch operation.

        Waits for DPU->Host DMA and performs Host->GPU copy.

        Args:
            handle: Handle from fetch_tensor_async
        """
        with self._lock:
            # Wait for DPU->Host
            if handle.doca_transfer_id is not None:
                self._doca_client.wait_transfer(handle.doca_transfer_id)

            # Now do async Host->GPU copy
            if handle.dst_tensor is not None and handle.pinned_buffer is not None:
                CUDARuntime.memcpy_htod(
                    handle.dst_tensor.data_ptr(),
                    handle.pinned_buffer.address,
                    handle.size_bytes,
                    stream=handle.copy_stream
                )

                # Record completion event
                handle.copy_event.record(handle.copy_stream)

            handle.stage = "host_to_gpu"

            # Update block state
            block = self._blocks.get(handle.block_id)
            if block:
                block.state = BlockState.ON_GPU

    def wait_fetch_complete(self, handle: AsyncTransferHandle) -> None:
        """
        Wait for fetch to fully complete (including Host->GPU copy).

        Args:
            handle: Handle from fetch_tensor_async
        """
        if handle.stage == "dpu_to_host":
            self.complete_fetch_async(handle)

        # Wait for Host->GPU copy
        handle.copy_event.synchronize()

        # Update stats
        elapsed_ms = (time.perf_counter() - handle.start_time) * 1000.0
        self._async_wait_ms_total += elapsed_ms
        self._async_wait_ms_max = max(self._async_wait_ms_max, elapsed_ms)
        self._async_wait_count += 1
        self._async_bytes_total += handle.size_bytes

        handle.stage = "complete"

        # Return stream/event to pool
        with self._lock:
            self._stream_event_pool.release(handle.copy_stream, handle.copy_event)
            self._pending_async.pop(handle.block_id, None)

    def find_by_hash(self, hash_key: str) -> Optional[int]:
        """
        Find a block by its hash key (for prefix matching).

        Args:
            hash_key: Hash of the prefix tokens

        Returns:
            block_id if found, None otherwise
        """
        with self._lock:
            return self._hash_to_block.get(hash_key)

    def has_block(self, block_id: int) -> bool:
        """Check if a block exists and is on DPU."""
        with self._lock:
            block = self._blocks.get(block_id)
            return block is not None and block.state == BlockState.ON_DPU

    def has_prefix(self, hash_key: str) -> bool:
        """Check if a prefix is cached on the DPU."""
        block_id = self.find_by_hash(hash_key)
        if block_id is None:
            return False

        block = self._blocks.get(block_id)
        return block is not None and block.state == BlockState.ON_DPU

    def increment_refcount(self, block_id: int) -> None:
        """Increment reference count for a block."""
        with self._lock:
            if block_id in self._blocks:
                self._blocks[block_id].refcount += 1

    def decrement_refcount(self, block_id: int) -> None:
        """Decrement reference count for a block."""
        with self._lock:
            if block_id in self._blocks:
                self._blocks[block_id].refcount = max(0, self._blocks[block_id].refcount - 1)

    def get_stats(self) -> Dict[str, Any]:
        """Get offload statistics."""
        stats = self._doca_client.get_stats()
        with self._lock:
            blocks_on_dpu = sum(1 for b in self._blocks.values() if b.state == BlockState.ON_DPU)
            blocks_on_gpu = sum(1 for b in self._blocks.values() if b.state == BlockState.ON_GPU)

        return {
            "total_transfers": stats.total_transfers,
            "total_bytes": stats.total_bytes,
            "failed_transfers": stats.failed_transfers,
            "avg_latency_us": stats.avg_latency_us,
            "peak_bandwidth_gbps": stats.peak_bandwidth_gbps,
            "blocks_on_dpu": blocks_on_dpu,
            "blocks_on_gpu": blocks_on_gpu,
            "total_blocks": len(self._blocks),
            "async_wait_ms_total": self._async_wait_ms_total,
            "async_wait_ms_max": self._async_wait_ms_max,
            "async_wait_count": self._async_wait_count,
            "async_bytes_total": self._async_bytes_total,
        }

    def close(self) -> None:
        """Cleanup resources."""
        logger.info("Closing KVOffloadManager")

        with self._lock:
            # Cleanup all blocks
            for block_id in list(self._blocks.keys()):
                self._remove_block(block_id)

        # Destroy CUDA stream
        if self._cuda_stream:
            CUDARuntime.stream_destroy(self._cuda_stream)
            self._cuda_stream = None

        # Close stream/event pool
        if hasattr(self, '_stream_event_pool'):
            self._stream_event_pool.close()

        # Close buffer pool
        self._buffer_pool.close()

        # Close DOCA client
        self._doca_client.close()

        logger.info("KVOffloadManager closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


# Utility function for computing prefix hashes
def compute_prefix_hash(token_ids: List[int], num_tokens: int) -> str:
    """
    Compute a hash for a token prefix.

    Args:
        token_ids: List of token IDs
        num_tokens: Number of tokens to include in hash

    Returns:
        Hash string for the prefix
    """
    import hashlib
    prefix = tuple(token_ids[:num_tokens])
    return hashlib.sha256(str(prefix).encode()).hexdigest()[:16]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Test basic functionality
    pci = find_bluefield_pci_address()
    if pci:
        print(f"Found BlueField at: {pci}")
        print("Use KVOffloadManager(pci_addr='{pci}') to initialize")
    else:
        print("No BlueField DPU found on this system")
