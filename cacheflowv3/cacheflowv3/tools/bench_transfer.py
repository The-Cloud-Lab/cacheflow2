# SPDX-License-Identifier: Apache-2.0
"""Transfer microbenchmark: save/load latency and bandwidth vs prefix length.

Uses the real worker engine against a live server with a synthetic paged KV
cache shaped like a model, so the numbers include the GPU gather/scatter, the
RDMA transfer and the control-plane calls, but no model compute. Gives T_DPU(N)
for the efficiency model in the paper (Section 3.4) and the I/O ablation.

    python -m cacheflowv3.tools.bench_transfer --model-preset qwen3-4b \\
        --tokens 1024,4096,16384 --modes pull:mapped,push:mapped,pull:staged \\
        --out io.csv
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
import uuid

import torch

from ..config import CacheFlowConfig
from ..engine import TransferEngine
from ..hashing import chunk_keys, namespace_seed
from ..kv_access import KVAccessor

# layers, kv heads, head dim (bf16)
PRESETS = {
    "qwen3-0.6b": (28, 8, 128),
    "qwen3-4b": (36, 8, 128),
    "llama-3.2-3b": (28, 8, 128),
    "ministral-3b": (26, 8, 128),
}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument("--host", default="10.0.1.2")
    p.add_argument("--model-preset", default="qwen3-4b", choices=sorted(PRESETS))
    p.add_argument("--tokens", default="1024,4096,8192,16384,20480")
    p.add_argument(
        "--modes",
        default="pull:mapped,push:mapped,pull:staged,push:staged",
        help="comma list of transfer_mode:data_path",
    )
    p.add_argument("--chunk-tokens", type=int, default=256)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--staging-gb", type=float, default=4.0)
    p.add_argument("--out", help="CSV output path")
    a = p.parse_args(argv)

    L, H, D = PRESETS[a.model_preset]
    B, C = 16, a.chunk_tokens
    token_counts = [int(t) // C * C for t in a.tokens.split(",")]
    max_tok = max(token_counts)
    nb = max_tok // B + 1
    dev = torch.device("cuda", 0)
    kv = [
        torch.randn(2, nb, B, H, D, device=dev, dtype=torch.bfloat16) for _ in range(L)
    ]
    rows = []
    for mode in a.modes.split(","):
        tmode, dpath = mode.split(":")
        cfg = CacheFlowConfig.from_extra_config(
            {
                "server_host": a.host,
                "chunk_tokens": C,
                "transfer_mode": tmode,
                "data_path": dpath,
                "staging_pool_gb": a.staging_gb,
                "stats_log_interval_s": 1e9,
                "skip_save_if_prefix_cached": False,
            }
        )
        eng = TransferEngine(cfg, rank=0, world=1, device=dev)
        eng.set_geometry(KVAccessor(kv[0], B), L, B)
        try:
            for ntok in token_counts:
                n = ntok // C
                t = torch.arange(ntok)
                blk, off = (t // B).view(n, C).to(dev), (t % B).view(n, C).to(dev)
                block_ids = list(range(nb))
                save_ms, load_ms, first_ms = [], [], []
                for _ in range(a.repeats):
                    keys = chunk_keys(
                        list(range(ntok)), C, n, namespace_seed("bench", uuid.uuid4())
                    )
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    job = eng.begin_save("bench", keys, blk, off)
                    if job is None:
                        raise RuntimeError("staging pool too small; raise --staging-gb")
                    for layer in range(L):
                        eng.save_layer(job, layer, kv[layer])
                    job.done.wait(600)
                    save_ms.append((time.perf_counter() - t0) * 1e3)
                    r = eng.session.ctl.request("lookup", keys=keys, ranks=1, pin=True)
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    lj = eng.begin_load(
                        "bench",
                        r["lease"],
                        r["entries"][0],
                        r["rkey"],
                        0,
                        0,
                        ntok,
                        block_ids,
                    )
                    for layer in range(L):
                        eng.load_layer(lj, layer, kv[layer])
                        if layer == 0:
                            torch.cuda.synchronize()
                            first_ms.append((time.perf_counter() - t0) * 1e3)
                    torch.cuda.synchronize()
                    load_ms.append((time.perf_counter() - t0) * 1e3)
                    eng.finish_load(lj)
                gb = n * eng.geom.entry_bytes / 1e9
                row = {
                    "mode": tmode,
                    "data_path": dpath,
                    "model": a.model_preset,
                    "tokens": ntok,
                    "gb": round(gb, 4),
                    "save_ms": round(statistics.median(save_ms), 2),
                    "load_ms": round(statistics.median(load_ms), 2),
                    "load_first_layer_ms": round(statistics.median(first_ms), 2),
                    "save_gbps": round(gb / statistics.median(save_ms) * 1e3, 3),
                    "load_gbps": round(gb / statistics.median(load_ms) * 1e3, 3),
                }
                rows.append(row)
                print(
                    f"{tmode:4s} {dpath:6s} {ntok:6d} tok {gb:7.3f} GB  "
                    f"save {row['save_ms']:9.1f} ms ({row['save_gbps']:.2f} GB/s)  "
                    f"load {row['load_ms']:9.1f} ms ({row['load_gbps']:.2f} GB/s, "
                    f"layer0 ready {row['load_first_layer_ms']:.1f} ms)",
                    flush=True,
                )
        finally:
            eng.close()
    if a.out and rows:
        with open(a.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
