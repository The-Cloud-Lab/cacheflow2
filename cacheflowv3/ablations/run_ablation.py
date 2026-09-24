#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run a CacheFlow v3 ablation study end to end.

A study (ablations/configs/*.json) fixes a model, a vLLM command line, a
prefix_repetition workload and a list of request rates, and lists variants.
For every (variant, rate) the runner:

  1. prepares the BF3 pool (flush for cold runs, set capacity, optional prewarm
     pass followed by a vLLM restart for warm-restart runs),
  2. starts `vllm serve` with the variant's KV connector config,
  3. runs `vllm bench serve --dataset-name prefix_repetition`,
  4. stops vLLM and collects the benchmark JSON, CacheFlow worker/scheduler
     stats, server stats and the vLLM log into
     results/<study>/<variant>/rate_<r>/,

then writes results/<study>/summary.csv.

    python ablations/run_ablation.py ablations/configs/components.json
    python ablations/run_ablation.py ablations/configs/components.json \
        --only full,sync_saves --rates 1
    python ablations/run_ablation.py ablations/configs/components.json --dry-run

Variant fields (all optional except name):
  system        "cacheflow" (default) | "vllm" (no connector) | "custom"
  extra         overrides for kv_connector_extra_config
  kv_transfer_config  full config for system="custom" (e.g. LMCache)
  vllm_args     extra vllm serve arguments
  workload      overrides of the study workload
  capacity_gb   BF3 pool capacity for the run (default: whole pool)
  cold          flush the pool first (default true)
  prewarm       run the workload once, restart vLLM, then measure (warm restart)
  env           extra environment variables for vllm serve
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT_CF_CONFIG = ROOT / "configs" / "cacheflowv3.json"
STATS_GLOB = "/tmp/cacheflowv3_stats_{role}_{rank}.json"


def vllm_bin(study: dict) -> str:
    """The study's vllm executable, else the one next to this Python."""
    if study.get("vllm_bin"):
        return study["vllm_bin"]
    local = Path(sys.executable).parent / "vllm"
    return str(local) if local.exists() else "vllm"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def cfctl(python: str, host: str, *args: str) -> str:
    out = subprocess.run(
        [python, "-m", "cacheflowv3.tools.cfctl", "--host", host, *args],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if out.returncode:
        raise RuntimeError(f"cfctl {' '.join(args)} failed: {out.stderr.strip()}")
    return out.stdout


class VllmServer:
    def __init__(self, study: dict, variant: dict, logfile: Path):
        self.study, self.variant, self.logfile = study, variant, logfile
        self.proc: subprocess.Popen | None = None
        self.port = int(study.get("port", 8100))

    def command(self) -> list[str]:
        s, v = self.study, self.variant
        cmd = [
            vllm_bin(s),
            "serve",
            s["model"],
            "--port",
            str(self.port),
            *s.get("vllm_args", []),
            *v.get("vllm_args", []),
        ]
        system = v.get("system", "cacheflow")
        if system == "cacheflow":
            cfg = json.loads(DEFAULT_CF_CONFIG.read_text())
            cfg["kv_connector_extra_config"].update(s.get("base_extra_config", {}))
            cfg["kv_connector_extra_config"].update(v.get("extra", {}))
            cmd += ["--kv-transfer-config", json.dumps(cfg)]
        elif system == "custom":
            cmd += ["--kv-transfer-config", json.dumps(v["kv_transfer_config"])]
        return cmd

    def start(self, timeout_s: int = 900) -> None:
        env = {
            **os.environ,
            "PYTHONHASHSEED": "0",
            **self.study.get("env", {}),
            **self.variant.get("env", {}),
        }
        self.logfile.parent.mkdir(parents=True, exist_ok=True)
        # Popen dups the fd, so the file can be closed right after starting
        with open(self.logfile, "a") as out:
            self.proc = subprocess.Popen(
                self.command(),
                stdout=out,
                stderr=subprocess.STDOUT,
                env=env,
                cwd="/tmp",
                start_new_session=True,
            )
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(
                    f"vllm exited with {self.proc.returncode}; see {self.logfile}"
                )
            try:
                urllib.request.urlopen(
                    f"http://localhost:{self.port}/health", timeout=2
                )
                return
            except Exception:
                time.sleep(2)
        raise RuntimeError(f"vllm not healthy after {timeout_s}s; see {self.logfile}")

    def stop(self) -> None:
        if self.proc is None or self.proc.poll() is not None:
            return
        os.killpg(self.proc.pid, signal.SIGINT)  # graceful: lets connectors write stats
        try:
            self.proc.wait(timeout=90)
        except subprocess.TimeoutExpired:
            os.killpg(self.proc.pid, signal.SIGKILL)
            self.proc.wait()
        time.sleep(5)  # GPU memory release


def bench_cmd(
    study: dict, variant: dict, rate: float, outdir: Path, fname: str
) -> list[str]:
    w = {**study["workload"], **variant.get("workload", {})}
    return [
        vllm_bin(study),
        "bench",
        "serve",
        "--backend",
        "vllm",
        "--model",
        study["model"],
        "--port",
        str(study.get("port", 8100)),
        "--dataset-name",
        "prefix_repetition",
        "--prefix-repetition-prefix-len",
        str(w["prefix_len"]),
        "--prefix-repetition-suffix-len",
        str(w["suffix_len"]),
        "--prefix-repetition-num-prefixes",
        str(w["num_prefixes"]),
        "--prefix-repetition-output-len",
        str(w["output_len"]),
        "--num-prompts",
        str(w["num_prompts"]),
        "--request-rate",
        str(rate),
        "--burstiness",
        str(w.get("burstiness", 1.0)),
        "--seed",
        str(w.get("seed", 0)),
        "--ignore-eos",
        "--percentile-metrics",
        "ttft,tpot,itl,e2el",
        "--metric-percentiles",
        "50,90,99",
        "--save-result",
        "--result-dir",
        str(outdir),
        "--result-filename",
        f"{fname}.json",
    ]


def run_bench(study, variant, rate, outdir: Path, fname: str) -> None:
    env = {**os.environ, "PYTHONHASHSEED": "0"}
    with open(outdir / f"{fname}.log", "w") as f:
        rc = subprocess.run(
            bench_cmd(study, variant, rate, outdir, fname),
            stdout=f,
            stderr=subprocess.STDOUT,
            env=env,
            cwd="/tmp",
        ).returncode
    if rc:
        raise RuntimeError(f"vllm bench serve failed ({rc}); see {outdir / fname}.log")


def run_one(study: dict, variant: dict, rate: float, outdir: Path, python: str) -> None:
    host = study.get("server_host", "10.0.1.2")
    uses_cf = variant.get("system", "cacheflow") == "cacheflow"
    for f in Path("/tmp").glob("cacheflowv3_stats_*.json"):
        f.unlink()
    if uses_cf:
        if variant.get("cold", True):
            log(f"  flush: {cfctl(python, host, 'flush').strip()}")
        cap = variant.get("capacity_gb", study.get("capacity_gb"))
        stats = json.loads(cfctl(python, host, "stats"))
        cap_bytes = int(cap * 2**30) if cap else stats["store"]["pool_bytes"]
        cfctl(python, host, "set-capacity", str(cap_bytes / 2**30))
    server = VllmServer(study, variant, outdir / "vllm.log")
    try:
        if variant.get("prewarm"):
            log("  prewarm pass (then restart vLLM)")
            server.start()
            run_bench(study, variant, rate, outdir, "prewarm")
            server.stop()
        server.start()
        log(f"  benchmark at {rate} req/s")
        run_bench(study, variant, rate, outdir, "bench")
    finally:
        server.stop()
    for f in Path("/tmp").glob("cacheflowv3_stats_*.json"):
        shutil.copy(f, outdir / f.name.replace("cacheflowv3_stats_", ""))
    if uses_cf:
        (outdir / "server_stats.json").write_text(cfctl(python, host, "stats"))


SUMMARY_FIELDS = [
    "mean_ttft_ms",
    "median_ttft_ms",
    "p90_ttft_ms",
    "p99_ttft_ms",
    "mean_tpot_ms",
    "p90_tpot_ms",
    "p99_tpot_ms",
    "mean_itl_ms",
    "output_throughput",
    "total_token_throughput",
    "request_throughput",
    "completed",
]


def summarize(study_dir: Path) -> Path:
    rows = []
    for bench in sorted(study_dir.glob("*/rate_*/bench.json")):
        r = json.loads(bench.read_text())
        row = {"variant": bench.parent.parent.name, "rate": bench.parent.name[5:]}
        row.update({k: r.get(k) for k in SUMMARY_FIELDS})
        ws = bench.parent / "worker_0.json"
        if ws.exists():
            s = json.loads(ws.read_text())["stats"]
            row.update(
                cf_loads=s["loads"],
                cf_load_failures=s["load_failures"],
                cf_load_gb=s["load_bytes"] / 1e9,
                cf_saves=s["saves"],
                cf_save_gb=s["save_bytes"] / 1e9,
                cf_load_gbps=s["load_bytes"] / s["load_total_ms"] / 1e6
                if s["load_total_ms"]
                else None,
            )
        rows.append(row)
    out = study_dir / "summary.csv"
    if rows:
        keys = sorted(
            {k for r in rows for k in r},
            key=lambda k: (k not in ("variant", "rate"), k),
        )
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
    return out


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument("study")
    p.add_argument("--only", help="comma-separated variant names")
    p.add_argument("--rates", help="comma-separated request rates (override the study)")
    p.add_argument("--results", default=str(ROOT / "results"))
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="skip runs that already have bench.json",
    )
    p.add_argument("--dry-run", action="store_true", help="print the commands only")
    a = p.parse_args()

    study = json.loads(Path(a.study).read_text())
    python = study.get("python", sys.executable)
    variants = study["variants"]
    if a.only:
        keep = set(a.only.split(","))
        variants = [v for v in variants if v["name"] in keep]
    rates = (
        [float(r) for r in a.rates.split(",")] if a.rates else study["request_rates"]
    )
    study_dir = Path(a.results) / study["name"]

    for v in variants:
        for rate in rates:
            outdir = study_dir / v["name"] / f"rate_{rate:g}"
            if a.dry_run:
                print(" ".join(VllmServer(study, v, outdir / "vllm.log").command()))
                print(" ".join(bench_cmd(study, v, rate, outdir, "bench")))
                continue
            if a.skip_existing and (outdir / "bench.json").exists():
                continue
            outdir.mkdir(parents=True, exist_ok=True)
            (outdir / "variant.json").write_text(
                json.dumps(
                    {
                        "study": {k: study[k] for k in study if k != "variants"},
                        "variant": v,
                        "rate": rate,
                    },
                    indent=1,
                )
            )
            log(f"{study['name']}: variant {v['name']} rate {rate:g}")
            try:
                run_one(copy.deepcopy(study), v, rate, outdir, python)
            except Exception as e:
                log(f"  FAILED: {e}")
                (outdir / "FAILED").write_text(str(e))
    if not a.dry_run:
        log(f"summary: {summarize(study_dir)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
