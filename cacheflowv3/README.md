# CacheFlow v3

A prefix KV-cache pool that lives in **BlueField-3 DRAM** and is reached by the GPU
host **over the network (RoCE RDMA)**, integrated into vLLM as an ordinary KV
connector. v1 (`DOCA_Backend/`, `CacheFlowConnectorV1`) offloaded to a DPU in the
*same* host over PCIe with DOCA Comm Channel + DOCA DMA; v3 targets a GPU host
and a SmartNIC that are separate machines.

```
 DGX Spark (GB10, unified memory)                         BlueField-3 (Arm, 32 GB DDR5)
 ┌──────────────────────────────────────┐   QSFP 100G     ┌──────────────────────────────┐
 │ vLLM ── CacheFlowConnectorV3         │   RoCE v2       │ cacheflowv3.server           │
 │   scheduler: chunk hashes ──lookup──────── TCP ctrl ───▶ │  prefix index + LRU + leases │
 │   worker: gather/scatter paged KV    │                 │  registered pool (16 GiB)    │
 │     ⇅ NIC-registered staging memory  │ ◀── RDMA R/W ──▶ │                              │
 │   ConnectX-7 (mlx5_0) 10.0.1.1       │                 │ p1 ⇄ SF (mlx5_3) 10.0.1.2    │
 └──────────────────────────────────────┘                 └──────────────────────────────┘
```

## What is different from v1

| | v1 | v3 |
|---|---|---|
| Transport | DOCA Comm Channel + DOCA DMA over PCIe (same host) | RDMA (verbs/rdma_cm) over RoCE between machines |
| Prefix index | JSON file shared by scheduler and worker | on the BF3; the scheduler asks the NIC (one RTT) |
| Cache unit | one whole prefix (`common_prefix_num_tokens`) | chained 256-token chunks: partial-prefix hits, multi-model safe |
| Load | waits for the whole prefix at layer 0 | real layer-wise pipelining (`layerwise_load`) |
| GPU memory | cudaMemcpy to pinned buffers | GPU gathers straight into NIC-registered memory (`data_path=mapped`) |
| Who moves bytes | DPU DMA engine | GPU-host NIC (`pull`) or BF3 (`push`), switchable |
| Chunked prefill | saved a truncated prefix under the full hash | saves incrementally as chunks complete |
| vLLM | in-tree (needs this fork) | out-of-tree via `kv_connector_module_path`; works with stock vLLM 0.20+ |

GB10 has no GPUDirect RDMA / dma-buf export (`cudaMalloc` memory cannot be
`ibv_reg_mr`'d). Because GPU and CPU share DRAM, `data_path=mapped` is the
closest equivalent: the NIC and the GPU operate on the same registered buffer
with no copy in between.

## Layout

```
csrc/cfrdma.{c,h}          RDMA data plane (C, verbs + rdma_cm), built for Spark and BF3
cacheflowv3/rdma.py        ctypes binding
cacheflowv3/protocol.py    control protocol (length-prefixed JSON over TCP)
cacheflowv3/layout.py      chunk byte layout + RDMA op planning (shared by both ends)
cacheflowv3/server/        BF3 server: store.py (allocator/index/LRU/leases), server.py
cacheflowv3/connector.py   CacheFlowConnectorV3 (vLLM KVConnectorBase_V1)
cacheflowv3/engine.py      worker transfer engine (save pipeline, layer-wise loads)
cacheflowv3/staging.py     NIC+CUDA registered staging memory
cacheflowv3/kv_access.py   gather/scatter for FlashAttention / FlashInfer / MLA layouts
cacheflowv3/tools/         cfctl (stats/flush/capacity/ping), bench_transfer (I/O microbench)
ablations/                 study runner + one JSON per ablation
scripts/                   network setup, BF3 deploy, vLLM correctness check
tests/                     unit tests + live end-to-end tests (all 32 data-path modes)
```

## Setup

```bash
# 1. Network (non-persistent; re-run after reboots)
ssh routenic-bf3 'sudo bash -s bf3' < scripts/setup_network.sh
sudo scripts/setup_network.sh spark
scripts/setup_network.sh check

# 2. Server on the BF3 (syncs this directory, builds libcfrdma, starts as root)
scripts/deploy_bf3.sh start --pool-gb 16        # also: stop | logs | sync

# 3. Connector on the GPU host
make -C csrc
/home/spark2/vllm_env/bin/pip install -e .       # into the vLLM environment

# 4. Serve
vllm serve Qwen/Qwen3-4B --kv-transfer-config "$(cat configs/cacheflowv3.json)"
```

`configs/cacheflowv3.json` loads the connector with
`kv_connector_module_path=cacheflowv3.connector`, so stock vLLM works; the fork
also registers the name `CacheFlowConnectorV3`. Keep
`"kv_load_failure_policy": "recompute"` so a failed or evicted load falls back to
recomputation instead of failing the request.

## Multi-node: several GPU hosts, one BF3 pool

The server can listen on several BF3 links at once (`--bind 10.0.1.2,10.0.2.2`, the
default in `deploy_bf3.sh`). It registers one pool on every RDMA device and keeps
one prefix index, so a prefix saved by one host is a cache hit on every other host
serving the same model. Keys include the model and KV format, so hosts serving
different models never collide. Each host talks to the BF3 address on its own
link; `scripts/make_config.py` picks it from the hostname:

```bash
# on each Spark
scripts/start_vllm.sh Qwen/Qwen3-0.6B /tmp/vllm.log --kv-transfer-config "$(python3 scripts/make_config.py)"
```

Checks (2026-09-24, Qwen3-0.6B, vLLM prefix caching off so all reuse comes from the BF3):

- `tests/multinode_check.py`: spark2 saves 16 chunks and Spark1 loads them bit-exact, in both
  pull and push mode (4.5 GB/s on Spark1's 40G link).
- vLLM: spark2 served prompts A and B. Spark1 then served A, a partial overlap of A, and B from the
  BF3, and its outputs were identical to its own baseline. TTFT for B was 158 ms, against 414 ms
  when recomputing. With both hosts serving at the same time there were no load failures and no
  leaked leases.

## Configuration and the ablations it enables

All options go in `kv_connector_extra_config` (see `cacheflowv3/config.py`). v1
names (`use_doca_buffer_pool`, `tokens_per_block`, `common_prefix_num_tokens`,
`num_staging_buffers`×`block_size`, `max_blocks`×`block_size`) are accepted.

| Option (default) | Ablation |
|---|---|
| `use_registered_buffer_pool` (true) | pre-registered staging vs pin+register per transfer |
| `async_transfers` (true) | saves finish in the background vs `wait_for_save` blocks |
| `overlap_dma_with_copy` (true) | post each layer's RDMA as soon as its gather is done |
| `layerwise_load` (true) | per-layer load completion vs whole prefix before layer 0 |
| `transfer_mode` (pull) | `pull`: GPU host NIC drives RDMA · `push`: the BF3 drives it |
| `data_path` (mapped) | `mapped`: zero-copy into NIC memory · `staged`: copy-engine D2H/H2D |
| `skip_save_if_prefix_cached` (true) | dedup saves of chunks already in the pool |
| `copy_stream_pool_size` (4), `staging_pool_gb` (4) | staging resources |
| `dpu_capacity_gb` | pool capacity for the run (eviction pressure) |
| `reset_dpu_cache_on_start` (false) | cold pool at startup |
| `enable_save` / `enable_load` (true) | isolate one direction |
| `chunk_tokens` (256), `max_cached_tokens` (0 = all) | caching granularity / cap |

Studies in `ablations/configs/` (run with `ablations/run_ablation.py <study.json>`):

| Study | Question |
|---|---|
| `main` | vLLM vs LMCache vs CacheFlow over QPS 1–10 (Table 1); LMCache needs `pip install lmcache` |
| `components` | leave-one-out of each Section 3.3 optimization |
| `alpha_sweep` | 20k prefix + growing suffix → measured speedup vs the α\* model |
| `capacity` | pool capacity vs hit rate under LRU eviction |
| `datapath` | pull/push × mapped/staged |
| `staging` | copy streams and staging size |
| `restart` | cold pool vs a pool warmed by a previous vLLM process |
| `smoke` | 5-minute sanity run of the runner with Qwen3-0.6B |

Each run stores the `vllm bench serve` JSON, the vLLM log, CacheFlow worker and
scheduler stats (including per-hook CPU time) and server stats in
`results/<study>/<variant>/rate_<r>/`, plus `results/<study>/summary.csv`.
`CACHEFLOW_PROFILE=1` adds per-step CPU/GPU timings.

I/O microbenchmark (gives T_DPU for the efficiency model):

```bash
python -m cacheflowv3.tools.bench_transfer --model-preset qwen3-4b --tokens 1024,4096,16384,20480 --out io.csv
```

## Tests

```bash
pytest -q tests/                                   # unit tests (store, layout, hashing, config, KV layouts)
CACHEFLOW_SERVER=10.0.1.2 pytest -q tests/         # + live save/load round trips in all 32 mode combinations
python scripts/check_vllm_e2e.py --model Qwen/Qwen3-0.6B --out cf.json --compare base.json
```

The vLLM check compares greedy outputs against a baseline. Run the baseline with
`--compilation-config '{"cudagraph_mode": "PIECEWISE"}'`: the connector forces
PIECEWISE CUDA graphs (per-layer hooks), and FULL graphs alone change bf16
numerics enough to flip near-tie tokens.

## Testbed notes

The current state of the testbed, measured bandwidths, and open hardware issues are in
[`../TESTBED_TOPOLOGY.md`](../TESTBED_TOPOLOGY.md). Before benchmarking, check:

1. The ConnectX-7 PCIe link on each Spark is at 32 GT/s (`sudo lspci -vv -s 0000:01:00.0 | grep LnkSta`).
   spark2's was once stuck at Gen1, which capped RDMA below 1 GB/s.
2. `MemAvailable` is comfortably above vLLM's share plus `staging_pool_gb`. Low memory causes swap
   stalls that look like CacheFlow overhead, and the engine warns at startup.
3. `python -m cacheflowv3.tools.rdma_bw --host <BF3 address>` shows the expected raw bandwidth
   (spark2: ~12 GB/s each way; Spark1: 4.9 GB/s down, 1.7 GB/s up, an open issue).

Other notes: OVS is masked on the BF3, so `setup_network.sh` forwards each uplink to its Arm SF with
hardware `tc` rules. The BF3 server runs as root because of the locked-memory limit. CUDA can't register
hugetlbfs pages on GB10, so staging uses 2 MB transparent huge pages instead. perftest versions
differ between hosts, so use `rdma_bw` instead of `ib_*_bw`.

## Current limitations

- One KV cache group only (hybrid / sliding-window models disable the connector).
- A request's loaded prefix must fit in `staging_pool_gb`.
- The control plane trusts clients (research prototype; no auth, no tenant isolation).
- The pool is DRAM only: a server restart empties it.
