#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Cross-host check: one GPU host saves, another loads, through one BF3 pool.

The KV data is generated on the CPU from a seed, so both hosts can build the
same expected tensors independently.

    # on host A (e.g. spark2, link 10.0.1.x)
    python tests/multinode_check.py save --server 10.0.1.2 --tag run1
    # on host B (e.g. Spark1, link 10.0.2.x)
    python tests/multinode_check.py load --server 10.0.2.2 --tag run1

The load side exits non-zero unless every loaded byte matches.
"""

from __future__ import annotations

import argparse
import sys
import time

import torch
from cacheflowv3.config import CacheFlowConfig
from cacheflowv3.engine import TransferEngine
from cacheflowv3.hashing import chunk_keys, namespace_seed
from cacheflowv3.kv_access import KVAccessor

L, B, H, D, C, N_CHUNKS = 28, 16, 8, 128, 256, 16  # Qwen3-0.6B-like, 4096 tokens
NB = N_CHUNKS * C // B + 8


def source_kv(seed: int) -> list[torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return [
        torch.randn(2, NB, B, H, D, generator=g, dtype=torch.float32)
        .to(torch.bfloat16)
        .cuda()
        for _ in range(L)
    ]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("role", choices=["save", "load"])
    p.add_argument("--server", required=True)
    p.add_argument("--tag", required=True, help="same value on both hosts")
    p.add_argument("--mode", default="pull", choices=["pull", "push"])
    a = p.parse_args()

    seed = sum(map(ord, a.tag))
    tokens = torch.randint(
        0, 32000, (N_CHUNKS * C,), generator=torch.Generator().manual_seed(seed)
    ).tolist()
    keys = chunk_keys(tokens, C, N_CHUNKS, namespace_seed("multinode", a.tag))
    cfg = CacheFlowConfig.from_extra_config(
        {
            "server_host": a.server,
            "chunk_tokens": C,
            "transfer_mode": a.mode,
            "staging_pool_gb": 2,
            "stats_log_interval_s": 1e9,
        }
    )
    engine = TransferEngine(cfg, rank=0, world=1, device=torch.device("cuda", 0))
    try:
        src = source_kv(seed)
        engine.set_geometry(KVAccessor(src[0], B), L, B)
        blocks = list(range(N_CHUNKS * C // B))
        t = torch.arange(N_CHUNKS * C)
        if a.role == "save":
            blk = (t // B).view(N_CHUNKS, C).cuda()
            off = (t % B).view(N_CHUNKS, C).cuda()
            t0 = time.perf_counter()
            job = engine.begin_save("mn", keys, blk, off)
            for layer in range(L):
                engine.save_layer(job, layer, src[layer])
            engine.end_save_step(job)
            job.done.wait(120)
            s = engine.snapshot()
            print(
                f"saved {s['save_chunks_new']} new / {s['save_chunks_exists']} "
                f"existing chunks ({s['save_bytes'] / 1e9:.2f} GB) in "
                f"{time.perf_counter() - t0:.2f}s, ok={job.ok}"
            )
            return 0 if job.ok else 1

        r = engine.session.ctl.request("lookup", keys=keys, ranks=1, pin=True)
        print(f"lookup: {r['hits']}/{len(keys)} chunks present in the shared pool")
        if r["hits"] != len(keys):
            return 1
        dst = [torch.zeros_like(x) for x in src]
        # load into a different block placement than the saver used
        dst_blocks = list(reversed(blocks))
        t0 = time.perf_counter()
        job = engine.begin_load(
            "mn", r["lease"], r["entries"][0], r["rkey"], 0, 0, N_CHUNKS * C, dst_blocks
        )
        for layer in range(L):
            engine.load_layer(job, layer, dst[layer])
        engine.finish_load(job)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        sb = torch.tensor(blocks)[t // B]
        db = torch.tensor(dst_blocks)[t // B]
        off = t % B
        bad = [
            layer
            for layer in range(L)
            if not torch.equal(dst[layer][:, db, off], src[layer][:, sb, off])
        ]
        gb = N_CHUNKS * engine.geom.entry_bytes / 1e9
        print(
            f"loaded {gb:.2f} GB in {dt:.2f}s ({gb / dt:.2f} GB/s); "
            f"{'ALL LAYERS MATCH' if not bad else f'MISMATCH in layers {bad}'}"
        )
        return 1 if bad else 0
    finally:
        engine.close()


if __name__ == "__main__":
    sys.exit(main())
