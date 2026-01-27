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
from typing import Dict, Optional, Tuple, List, Any
from dataclasses import dataclass
from enum import Enum
import logging

# Local imports
try:
    from .doca_cuda_utils import PinnedBuffer, PinnedBufferPool, CUDARuntime
    from .doca_kv_offload import DOCAKVOffloadClient, find_bluefield_pci_address
except ImportError:
    from doca_cuda_utils import PinnedBuffer, PinnedBufferPool, CUDARuntime
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
        num_staging_buffers: int = 4,
        async_transfers: bool = True,
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

        # CUDA stream for async operations
        if async_transfers:
            self._cuda_stream = CUDARuntime.stream_create()
        else:
            self._cuda_stream = None

        # LRU tracking for eviction
        self._lru_order: List[int] = []

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

                # We can release the pinned buffer now
                self._buffer_pool.release(pinned_buf)
                block.pinned_buffer = None
            else:
                # Store transfer_id for later completion check
                block._pending_transfer_id = transfer_id

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

            # For now, we need the pinned buffer to still hold the data
            # Full DPU->Host fetch would require bidirectional DMA (future)
            if block.pinned_buffer is None:
                raise RuntimeError(
                    "Block data not in pinned buffer. "
                    "DPU->Host fetch not yet implemented."
                )

            block.state = BlockState.TRANSFERRING_TO_GPU

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
        Fetch KV data into a PyTorch tensor.

        Args:
            block_id: Block identifier
            dst_tensor: Destination CUDA tensor
            sync: Wait for completion
        """
        if not HAS_TORCH:
            raise RuntimeError("PyTorch not available")

        if not dst_tensor.is_cuda:
            raise ValueError("Destination tensor must be on CUDA")

        size = dst_tensor.numel() * dst_tensor.element_size()
        self.fetch_block(block_id, dst_tensor.data_ptr(), size, sync)

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
