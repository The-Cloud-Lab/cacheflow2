# SPDX-License-Identifier: Apache-2.0
"""Gather/scatter between vLLM's paged KV cache and CacheFlow staging memory.

Supported per-layer KV tensor layouts (B = vLLM block size):
  kv_first     (2, num_blocks, B, H, D)   FlashAttention-style
  blocks_first (num_blocks, 2, B, H, D)   FlashInfer-style
  mla          (num_blocks, B, D)         MLA latent cache

Everything is expressed with block/offset index tensors, so no layout needs a
contiguous flattened view (which some stride orders don't allow).

A staging tensor for one layer is always viewed as (planes, n_chunks, C, F),
planes = 2 (K, V) or 1 (MLA), F = elements per token per plane.
"""

from __future__ import annotations

import torch


class UnsupportedKVLayout(RuntimeError):
    pass


class KVAccessor:
    def __init__(self, kv: torch.Tensor, block_size: int):
        s = tuple(kv.shape)
        if kv.dim() == 5 and s[0] == 2 and s[2] == block_size:
            self.kind, self.planes, self.head_shape = "kv_first", 2, s[3:]
        elif kv.dim() == 5 and s[1] == 2 and s[2] == block_size:
            self.kind, self.planes, self.head_shape = "blocks_first", 2, s[3:]
        elif kv.dim() == 3 and s[1] == block_size:
            self.kind, self.planes, self.head_shape = "mla", 1, s[2:]
        else:
            raise UnsupportedKVLayout(
                f"unsupported KV cache shape {s} for block size {block_size}"
            )
        self.feat = 1
        for d in self.head_shape:
            self.feat *= int(d)
        self.dtype = kv.dtype
        self.elem = kv.element_size()

    @property
    def token_bytes(self) -> int:
        """Bytes per token per layer (all planes)."""
        return self.planes * self.feat * self.elem

    def describe(self) -> str:
        return f"{self.kind}:planes={self.planes}:feat={self.feat}:{self.dtype}"

    def gather(
        self, kv: torch.Tensor, blk: torch.Tensor, off: torch.Tensor, out: torch.Tensor
    ) -> None:
        """out[p, i, j, :] = plane p of token (blk[i,j], off[i,j]).

        out has shape (planes, n, C, F)."""
        if self.kind == "kv_first":
            out.copy_(kv[:, blk, off].flatten(3))
        elif self.kind == "blocks_first":
            out.copy_(kv[blk, :, off].flatten(3).permute(2, 0, 1, 3))
        else:
            out[0].copy_(kv[blk, off])

    def scatter(
        self, kv: torch.Tensor, blk: torch.Tensor, off: torch.Tensor, src: torch.Tensor
    ) -> None:
        """Inverse of gather. blk/off may have any shape; src is
        (planes, *blk.shape, F)."""
        if self.kind == "kv_first":
            kv[:, blk, off] = src.unflatten(-1, self.head_shape)
        elif self.kind == "blocks_first":
            kv[blk, :, off] = src.movedim(0, -2).unflatten(-1, self.head_shape)
        else:
            kv[blk, off] = src[0].unflatten(-1, self.head_shape)


def slots_for_tokens(
    block_ids: list[int], block_size: int, start: int, stop: int, device
) -> tuple[torch.Tensor, torch.Tensor]:
    """(block index, in-block offset) tensors for token positions [start, stop)."""
    t = torch.arange(start, stop, dtype=torch.int64)
    table = torch.tensor(block_ids, dtype=torch.int64)
    blk = table[t // block_size]
    off = t % block_size
    return blk.to(device, non_blocking=True), off.to(device, non_blocking=True)
