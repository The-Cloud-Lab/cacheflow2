"""
DOCA KV Cache Offload Backend for vLLM/Cacheflow

This package provides KV cache offloading to BlueField DPU via DOCA.
"""

from .doca_cuda_utils import (
    CUDARuntime,
    PinnedBuffer,
    PinnedBufferPool,
)
from .doca_kv_offload import (
    DOCAKVOffloadClient,
    find_bluefield_pci_address,
)
from .kv_offload_manager import (
    KVOffloadManager,
    BlockState,
    BlockInfo,
    compute_prefix_hash,
)

__all__ = [
    "CUDARuntime",
    "PinnedBuffer",
    "PinnedBufferPool",
    "DOCAKVOffloadClient",
    "find_bluefield_pci_address",
    "KVOffloadManager",
    "BlockState",
    "BlockInfo",
    "compute_prefix_hash",
]
