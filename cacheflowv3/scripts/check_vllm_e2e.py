#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Correctness + TTFT check of a running vLLM server (with or without CacheFlow).

Sends greedy requests whose prompts share long prefixes and records TTFT and
the generated text. Run it once against plain vLLM and once against vLLM with
CacheFlowConnectorV3 (and --no-enable-prefix-caching, so every reuse has to
come from the BlueField-3), then compare:

    python check_vllm_e2e.py --out base.json ...
    python check_vllm_e2e.py --out cf.json --compare base.json ...

Requests:  A (cold)  ->  A again (full prefix hit)  ->  A[:k] + new suffix
(partial hit)  ->  B (unrelated, miss).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import urllib.request


def make_prompt(seed: int, words: int) -> str:
    rng = random.Random(seed)
    vocab = [
        "alpha",
        "bravo",
        "charlie",
        "delta",
        "echo",
        "foxtrot",
        "golf",
        "hotel",
        "india",
        "juliet",
        "kilo",
        "lima",
        "mike",
        "november",
        "oscar",
        "papa",
    ]
    return " ".join(f"{rng.choice(vocab)}{rng.randint(0, 999)}" for _ in range(words))


def complete(base: str, model: str, prompt: str, max_tokens: int) -> dict:
    body = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": True,
            "ignore_eos": True,
        }
    ).encode()
    req = urllib.request.Request(
        f"{base}/v1/completions", body, {"Content-Type": "application/json"}
    )
    t0 = time.perf_counter()
    ttft, text = None, []
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            line = line.strip()
            if not line.startswith(b"data:") or line == b"data: [DONE]":
                continue
            chunk = json.loads(line[5:])
            piece = chunk["choices"][0].get("text", "")
            if piece and ttft is None:
                ttft = time.perf_counter() - t0
            text.append(piece)
    return {
        "ttft_ms": (ttft or 0) * 1e3,
        "total_ms": (time.perf_counter() - t0) * 1e3,
        "text": "".join(text),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="http://localhost:8100")
    p.add_argument("--model", required=True)
    p.add_argument(
        "--words",
        type=int,
        default=1200,
        help="prompt length in words (~2 tokens each)",
    )
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument("--out", required=True)
    p.add_argument("--compare", help="baseline JSON to compare generated text with")
    p.add_argument(
        "--cases",
        default="A_cold=A,A_hit=A,A_partial=A_partial,B_miss=B",
        help="comma list of label=prompt; prompts: A, A_partial, B, C",
    )
    a = p.parse_args()

    A, B = make_prompt(1, a.words), make_prompt(2, a.words)
    partial = (
        " ".join(A.split()[: a.words * 2 // 3]) + " " + make_prompt(3, a.words // 3)
    )
    prompts = {"A": A, "A_partial": partial, "B": B, "C": make_prompt(4, a.words)}
    cases = [c.split("=", 1) for c in a.cases.split(",")]
    res = {}
    for name, pname in cases:
        r = complete(a.base, a.model, prompts[pname], a.max_tokens)
        r["prompt"] = pname
        res[name] = r
        print(
            f"{name:10s} TTFT {r['ttft_ms']:8.1f} ms  "
            f"total {r['total_ms']:8.1f} ms  {r['text'][:60]!r}"
        )
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    if a.compare:
        with open(a.compare) as f:
            base = json.load(f)
        # match by prompt (falls back to the label for older baseline files)
        ref = {v.get("prompt", k): v["text"] for k, v in base.items()}
        bad = [
            k for k, v in res.items() if v["text"] != ref.get(v["prompt"], ref.get(k))
        ]
        print("outputs identical to baseline" if not bad else f"MISMATCH in {bad}")
        return 1 if bad else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
