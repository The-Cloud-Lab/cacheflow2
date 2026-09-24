# SPDX-License-Identifier: Apache-2.0
"""Byte layouts of cached KV and the RDMA op plans that move it.

Unit of caching = one *chunk* of `chunk_tokens` prompt tokens. On the server a
chunk is one contiguous entry, laid out layer by layer:

    entry = [layer 0][layer 1] ... [layer L-1],   each layer = layer_bytes

and inside a layer the bytes are (planes, chunk_tokens, per_token_plane_bytes),
planes = 2 (K then V) for standard attention or 1 for MLA.

On the client the same data for a request's n chunks sits in one contiguous
staging region, laid out *layer-major* so that one layer of all chunks is
contiguous (one GPU gather/scatter per layer):

    staging = [layer 0: chunk 0 .. chunk n-1][layer 1: ...] ...

Both sides build the op list for a transfer from this module, so the control
plane only has to carry a compact descriptor. Imported on the BF3: numpy only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .rdma import OP_DTYPE


@dataclass(frozen=True)
class ChunkGeometry:
    num_layers: int
    layer_bytes: int  # bytes of one layer of one chunk

    @property
    def entry_bytes(self) -> int:
        return self.num_layers * self.layer_bytes

    def staging_offset(self, n_chunks: int, layer: int, chunk: int) -> int:
        return (layer * n_chunks + chunk) * self.layer_bytes


def plan_ops(
    geom: ChunkGeometry,
    staging_base: int,
    staging_key: int,
    entries: list[int] | np.ndarray,
    entry_key: int,
    layer_groups: bool,
    staging_is_local: bool = True,
    staging_chunks: list[int] | np.ndarray | None = None,
    n_staging: int | None = None,
    layers: tuple[int, int] | None = None,
) -> list[np.ndarray]:
    """Ops moving chunks between a client staging region and server entries.

    staging_is_local=True  : the vLLM worker posts the ops ("pull" mode);
                             local = staging, remote = server entries.
    staging_is_local=False : the BF3 server posts the ops ("push" mode);
                             local = server entries, remote = client staging.
    The keys are the lkey/rkey matching each side in that orientation.

    entries[i] pairs with staging chunk position staging_chunks[i] (default i)
    in a staging area laid out for n_staging chunks (default len(entries)), so
    a transfer can cover a subset of a request's chunks. `layers` restricts the
    plan to [start, stop).

    Returns one op array per layer if `layer_groups`, else a single array.
    Ops are ordered layer by layer either way, so completions (which are in
    order on an RC queue pair) also arrive layer by layer.
    """
    ent = np.asarray(entries, dtype=np.uint64)
    n = len(ent)
    pos = (
        np.arange(n, dtype=np.uint64)
        if staging_chunks is None
        else np.asarray(staging_chunks, dtype=np.uint64)
    )
    n_stage = np.uint64(n if n_staging is None else n_staging)
    S = geom.layer_bytes
    l0, l1 = layers if layers is not None else (0, geom.num_layers)
    nl = l1 - l0
    lay = np.repeat(np.arange(l0, l1, dtype=np.uint64), n)
    idx = np.tile(np.arange(n), nl)
    ops = np.zeros(n * nl, dtype=OP_DTYPE)
    staging = np.uint64(staging_base) + (lay * n_stage + pos[idx]) * np.uint64(S)
    entry = ent[idx] + lay * np.uint64(S)
    if staging_is_local:
        ops["laddr"], ops["lkey"] = staging, staging_key
        ops["raddr"], ops["rkey"] = entry, entry_key
    else:
        ops["laddr"], ops["lkey"] = entry, entry_key
        ops["raddr"], ops["rkey"] = staging, staging_key
    ops["len"] = S
    if not layer_groups:
        return [ops]
    return [ops[i * n : (i + 1) * n] for i in range(nl)]
