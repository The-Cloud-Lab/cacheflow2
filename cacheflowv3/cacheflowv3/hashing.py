# SPDX-License-Identifier: Apache-2.0
"""Chained chunk hashes used as prefix-cache keys.

key[i] = H(key[i-1] || tokens[i*C:(i+1)*C]), key[-1] = namespace seed.

Chaining makes key[i] identify the entire prefix up to chunk i, so the pool can
answer "longest cached prefix" by walking keys in order, and partial prefix
hits fall out naturally. The namespace seed binds keys to the model and KV
format so different models can safely share one pool.
"""

from __future__ import annotations

import hashlib
from array import array
from collections.abc import Sequence

try:
    import xxhash

    def _h(data: bytes) -> bytes:
        return xxhash.xxh3_128_digest(data)

    HASH_NAME = "xxh3_128"
except ImportError:  # pragma: no cover - fallback path

    def _h(data: bytes) -> bytes:
        return hashlib.blake2b(data, digest_size=16).digest()

    HASH_NAME = "blake2b_128"


def namespace_seed(*parts: object) -> bytes:
    """Seed binding keys to a model / KV format. Pass everything that changes
    the bytes of a cached chunk (model, dtype, layers, heads, chunk size...)."""
    text = "|".join(str(p) for p in ("cacheflowv3", HASH_NAME, *parts))
    return hashlib.blake2b(text.encode(), digest_size=16).digest()


def chunk_keys(
    token_ids: Sequence[int], chunk_tokens: int, n_chunks: int, seed: bytes
) -> list[str]:
    """Keys for the first n_chunks full chunks of token_ids."""
    n_chunks = min(n_chunks, len(token_ids) // chunk_tokens)
    if n_chunks <= 0:
        return []
    raw = array("q", token_ids[: n_chunks * chunk_tokens]).tobytes()
    step = chunk_tokens * 8
    keys: list[str] = []
    prev = seed
    for i in range(n_chunks):
        prev = _h(prev + raw[i * step : (i + 1) * step])
        keys.append(prev.hex())
    return keys
