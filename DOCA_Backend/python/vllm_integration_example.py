#!/usr/bin/env python3
"""
vLLM/Cacheflow Integration Example for DOCA KV Cache Offload

This example demonstrates how to integrate the DOCA KV cache offload
with vLLM's serving infrastructure for prefix caching.

Key Concepts:
    1. KV blocks are offloaded to DPU when sequences complete
    2. Prefix hashes are used to identify reusable KV data
    3. On new requests, check if prefix exists on DPU and fetch if needed

Usage:
    This is a reference implementation. Integrate these patterns into
    your vLLM/cacheflow fork's KV cache manager.

Integration Points in vLLM:
    - vllm/core/block_manager.py: Block allocation and eviction
    - vllm/worker/cache_engine.py: Physical KV cache operations
    - vllm/engine/llm_engine.py: Request processing hooks
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import hashlib
import logging

# Add local imports
sys.path.insert(0, str(Path(__file__).parent))

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("Warning: PyTorch not available. Some examples will not work.")

from kv_offload_manager import KVOffloadManager, compute_prefix_hash, BlockState

logger = logging.getLogger(__name__)


@dataclass
class KVBlockMetadata:
    """Metadata for a KV cache block."""
    block_id: int
    layer_idx: int
    num_tokens: int
    token_hash: str
    is_prefix: bool


class VLLMKVOffloadIntegration:
    """
    Integration layer between vLLM's KV cache and DOCA offload.

    This class shows how to hook into vLLM's KV cache lifecycle:
    1. When blocks are evicted from GPU, offload to DPU
    2. When requests match cached prefixes, restore from DPU
    3. Track prefix hashes for efficient lookup
    """

    def __init__(
        self,
        pci_addr: Optional[str] = None,
        num_layers: int = 32,
        block_size_tokens: int = 16,
        hidden_size: int = 4096,
        num_kv_heads: int = 32,
        head_dim: int = 128,
        dtype_size: int = 2,  # float16 = 2 bytes
    ):
        """
        Initialize the vLLM KV offload integration.

        Args:
            pci_addr: BlueField PCI address
            num_layers: Number of transformer layers
            block_size_tokens: Tokens per KV block (vLLM block_size)
            hidden_size: Model hidden dimension
            num_kv_heads: Number of KV attention heads
            head_dim: Dimension per head
            dtype_size: Bytes per element (2 for float16)
        """
        self.num_layers = num_layers
        self.block_size_tokens = block_size_tokens
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype_size = dtype_size

        # Calculate KV block size
        # Shape: [2, num_kv_heads, block_size, head_dim] per layer
        # 2 for K and V
        self.kv_block_size_per_layer = (
            2 * num_kv_heads * block_size_tokens * head_dim * dtype_size
        )
        self.total_kv_block_size = self.kv_block_size_per_layer * num_layers

        logger.info(
            f"KV block size: {self.kv_block_size_per_layer / 1024:.1f} KB/layer, "
            f"{self.total_kv_block_size / (1024 * 1024):.1f} MB total"
        )

        # Initialize offload manager
        self._offload_manager = KVOffloadManager(
            pci_addr=pci_addr,
            block_size=self.total_kv_block_size,
        )

        # Track blocks by sequence
        self._seq_to_blocks: Dict[int, List[int]] = {}  # seq_id -> block_ids
        self._prefix_cache: Dict[str, List[int]] = {}   # hash -> block_ids

    def compute_token_hash(self, token_ids: List[int]) -> str:
        """Compute hash for a sequence of tokens."""
        return compute_prefix_hash(token_ids, len(token_ids))

    def register_sequence_blocks(self, seq_id: int, block_ids: List[int]) -> None:
        """
        Register blocks allocated for a sequence.

        Call this when vLLM allocates blocks for a new sequence.
        """
        self._seq_to_blocks[seq_id] = block_ids.copy()
        logger.debug(f"Registered {len(block_ids)} blocks for sequence {seq_id}")

    def offload_sequence_prefix(
        self,
        seq_id: int,
        token_ids: List[int],
        kv_cache: List["torch.Tensor"],
        num_prefix_tokens: int,
    ) -> None:
        """
        Offload the prefix portion of a sequence's KV cache.

        Call this when a sequence completes or when GPU memory pressure
        requires eviction.

        Args:
            seq_id: Sequence identifier
            token_ids: Token IDs for the sequence
            kv_cache: List of KV tensors per layer
            num_prefix_tokens: Number of tokens to treat as prefix
        """
        if not HAS_TORCH:
            logger.warning("PyTorch not available, skipping offload")
            return

        if seq_id not in self._seq_to_blocks:
            logger.warning(f"Sequence {seq_id} not registered")
            return

        # Calculate number of blocks for prefix
        num_prefix_blocks = (num_prefix_tokens + self.block_size_tokens - 1) // self.block_size_tokens
        block_ids = self._seq_to_blocks[seq_id][:num_prefix_blocks]

        # Compute hash for prefix
        prefix_hash = self.compute_token_hash(token_ids[:num_prefix_tokens])

        logger.info(
            f"Offloading prefix: seq={seq_id}, tokens={num_prefix_tokens}, "
            f"blocks={len(block_ids)}, hash={prefix_hash[:8]}..."
        )

        # Offload each block
        for i, block_id in enumerate(block_ids):
            # Get KV data for this block across all layers
            # In vLLM, KV cache is typically [num_layers, 2, num_heads, block_size, head_dim]
            # We need to pack this into a contiguous buffer

            # This is a simplified example - actual implementation would
            # gather the KV data from the appropriate positions in the cache
            block_start_token = i * self.block_size_tokens
            block_end_token = min(block_start_token + self.block_size_tokens, num_prefix_tokens)

            # Create combined hash for this block
            block_hash = f"{prefix_hash}_{i}"

            # For demonstration, assume kv_cache contains per-block tensors
            # In real vLLM, you'd slice the appropriate region
            if i < len(kv_cache):
                kv_tensor = kv_cache[i]
                self._offload_manager.offload_tensor(
                    block_id=block_id,
                    tensor=kv_tensor,
                    hash_key=block_hash,
                )

        # Store prefix info
        self._prefix_cache[prefix_hash] = block_ids
        logger.info(f"Prefix offloaded successfully")

    def check_prefix_hit(self, token_ids: List[int]) -> Optional[Tuple[str, List[int]]]:
        """
        Check if a token prefix is cached on the DPU.

        Returns:
            (hash, block_ids) if found, None otherwise
        """
        # Try different prefix lengths
        for prefix_len in [len(token_ids), len(token_ids) // 2, 128, 64, 32]:
            if prefix_len > len(token_ids) or prefix_len == 0:
                continue

            prefix_hash = self.compute_token_hash(token_ids[:prefix_len])
            if prefix_hash in self._prefix_cache:
                block_ids = self._prefix_cache[prefix_hash]
                # Verify blocks are still on DPU
                all_on_dpu = all(
                    self._offload_manager.has_prefix(f"{prefix_hash}_{i}")
                    for i in range(len(block_ids))
                )
                if all_on_dpu:
                    logger.info(f"Prefix hit: {prefix_len} tokens, {len(block_ids)} blocks")
                    return prefix_hash, block_ids

        return None

    def restore_prefix_blocks(
        self,
        prefix_hash: str,
        block_ids: List[int],
        dst_kv_cache: List["torch.Tensor"],
    ) -> int:
        """
        Restore prefix blocks from DPU to GPU.

        Returns:
            Number of tokens restored
        """
        if not HAS_TORCH:
            logger.warning("PyTorch not available, skipping restore")
            return 0

        logger.info(f"Restoring {len(block_ids)} prefix blocks from DPU")

        for i, block_id in enumerate(block_ids):
            block_hash = f"{prefix_hash}_{i}"
            if i < len(dst_kv_cache):
                self._offload_manager.fetch_tensor(
                    block_id=block_id,
                    dst_tensor=dst_kv_cache[i],
                )

        tokens_restored = len(block_ids) * self.block_size_tokens
        logger.info(f"Restored {tokens_restored} tokens")
        return tokens_restored

    def get_stats(self) -> Dict:
        """Get offload statistics."""
        return self._offload_manager.get_stats()

    def close(self) -> None:
        """Cleanup resources."""
        self._offload_manager.close()


def example_usage():
    """
    Example showing how to use the integration with vLLM.

    This is pseudocode showing the integration points - actual implementation
    would hook into vLLM's block_manager and cache_engine.
    """
    print("=" * 60)
    print("vLLM Integration Example (Pseudocode)")
    print("=" * 60)

    print("""
# In vllm/core/block_manager.py:

from doca_kv_offload.python.vllm_integration_example import VLLMKVOffloadIntegration

class BlockManager:
    def __init__(self, ...):
        # ... existing initialization ...

        # Add DOCA offload integration
        self.kv_offload = VLLMKVOffloadIntegration(
            pci_addr="0000:03:00.0",  # Your BlueField PCI address
            num_layers=model_config.num_layers,
            block_size_tokens=cache_config.block_size,
            num_kv_heads=model_config.num_kv_heads,
            head_dim=model_config.head_dim,
        )

    def allocate(self, seq_group):
        # ... existing allocation ...

        # Register blocks for tracking
        self.kv_offload.register_sequence_blocks(
            seq_id=seq.seq_id,
            block_ids=[b.block_number for b in seq.logical_blocks]
        )

    def can_allocate(self, seq_group):
        # Check if prefix is cached on DPU first
        for seq in seq_group.get_seqs():
            prefix_hit = self.kv_offload.check_prefix_hit(seq.get_token_ids())
            if prefix_hit:
                # Restore from DPU instead of computing
                hash_key, block_ids = prefix_hit
                self.kv_offload.restore_prefix_blocks(
                    hash_key, block_ids, self.gpu_cache
                )
                return True  # Can allocate with cached prefix

        # Fall back to normal allocation
        return self._can_allocate_normal(seq_group)

    def free(self, seq):
        # Before freeing, check if this is a good prefix to cache
        token_ids = seq.get_token_ids()

        if self._should_cache_prefix(seq):
            # Offload to DPU for future reuse
            self.kv_offload.offload_sequence_prefix(
                seq_id=seq.seq_id,
                token_ids=token_ids,
                kv_cache=self._get_kv_cache_for_seq(seq),
                num_prefix_tokens=self._get_prefix_len(seq),
            )

        # ... existing free logic ...
""")

    print("\n# In vllm/engine/llm_engine.py:")
    print("""
class LLMEngine:
    def add_request(self, request_id, prompt, ...):
        # Check for prefix cache hit before processing
        prefix_hit = self.block_manager.kv_offload.check_prefix_hit(
            self.tokenizer.encode(prompt)
        )

        if prefix_hit:
            logger.info(f"Prefix cache hit for request {request_id}")
            # Restore and skip prefix computation
            ...
""")

    print("\n" + "=" * 60)


def demo_with_mock_data():
    """
    Demo with mock KV cache data (no actual GPU required).
    """
    print("\n=== Demo with Mock Data ===\n")

    # This would work with actual DPU connection
    from doca_kv_offload import find_bluefield_pci_address

    pci_addr = find_bluefield_pci_address()
    if not pci_addr:
        print("No BlueField DPU found. Skipping live demo.")
        return

    print(f"Found BlueField at: {pci_addr}")
    print("Demo would proceed with actual DMA transfers...")

    # In a real scenario:
    # integration = VLLMKVOffloadIntegration(pci_addr=pci_addr)
    # ... use the integration ...
    # integration.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    example_usage()
    demo_with_mock_data()
