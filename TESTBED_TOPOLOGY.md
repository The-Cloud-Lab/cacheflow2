# CacheFlow testbed topology

State as of **2026-09-24** (both Sparks attached to the BF3). Parts of this setup reset on reboot; see [Restoring after a reboot](#restoring-after-a-reboot).

## Diagram

```
                      Management network 172.16.3.0/23  (SSH, TCP control traffic)
  ════════╤══════════════════════════╤════════════════════════════╤══════════════════════════╤════════
          │ .227                     │ .100                       │ .234                     │ .189
          │ enP7s7                   │ enP7s7                     │ enp6s0                   │ oob_net0
 ┌────────┴──────────────┐  ┌────────┴──────────────┐  ┌──────────┴───────────────────────────┴──────────┐
 │ DGX Spark "spark2"    │  │ DGX Spark "Spark1"    │  │ cloudlab x86 server                             │
 │ gx10-ee53             │  │ spark-e1d8            │  │                                                 │
 │ GB10, 121 GB unified  │  │ GB10, 122 GB free     │  │   PCIe ┌──────────────────────────────────────┐ │
 │ 121 GB available      │  │ no hugepages          │  │  ══════│ BlueField-3 DPU (DPU / switchdev     │ │
 │                       │  │                       │  │  host  │ mode)                                │ │
 │                       │  │                       │  │  sees  │ 16 Arm cores, 32 GB DDR5             │ │
 │ vLLM +                │  │ vLLM +                │  │  mlx5_0│ OVS masked (disabled on purpose)     │ │
 │ CacheFlowConnectorV3  │  │ CacheFlowConnectorV3  │  │  mlx5_1│                                      │ │
 │                       │  │                       │  │  (no IP│  CacheFlow v3 server (root): one     │ │
 │ ConnectX-7            │  │ ConnectX-7            │  │  ,     │  16 GiB pool + one prefix index,     │ │
 │  PCIe 32GT/s x4 OK    │  │  PCIe 32GT/s x4 OK    │  │  unused│  shared by both Sparks. TCP :18515,  │ │
 │  (was Gen1; fixed)    │  │                       │  │  )     │  RDMA :18516 on both addresses       │ │
 │                       │  │                       │  │        │                                      │ │
 │                       │  │                       │  │        │  enp3s0f1s0 (SF, mlx5_3) 10.0.1.2    │ │
 │                       │  │                       │  │        │     ▲ tc redirect (hw offload)       │ │
 │ enp1s0f0np0 (mlx5_0)  │  │                       │  │        │  en3f1pf1sf0 ◄──► p1 ─────────┐      │ │
 │ 10.0.1.1  MTU 9000 ●──┼──┼───────────────────────┼──┼────────┼───────── 100G QSFP ───────────┘      │ │
 │                       │  │ enp1s0f1np1           │  │        │                                      │ │
 │                       │  │ 10.0.2.1 MTU 9000 ●───┼──┼────────┼───────── 40G ─────────────────┐      │ │
 │                       │  │                       │  │        │  en3f0pf0sf0 ◄──► p0 ─────────┘      │ │
 │                       │  │                       │  │        │     ▼ tc redirect (hw offload)       │ │
 │ enp1s0f1np1: no cable │  │ enp1s0f0np0: down     │  │        │  enp3s0f0s0 (SF, mlx5_2) 10.0.2.2    │ │
 └───────────────────────┘  └───────────────────────┘  │        │  tmfifo_net0 192.168.100.2 (to host) │ │
                                                       │        └──────────────────────────────────────┘ │
                                                       └─────────────────────────────────────────────────┘
```

## Machines

| Machine | Management IP | Login | Role |
|---|---|---|---|
| DGX Spark **spark2** (`gx10-ee53`) | 172.16.3.227 | `spark2` | GPU host that runs vLLM with CacheFlow v3 |
| DGX Spark **Spark1** (`spark-e1d8`) | 172.16.3.100 | `spark1` | Second GPU host that runs vLLM with CacheFlow v3 |
| **BlueField-3** Arm (`localhost.localdomain`) | 172.16.3.189 | `ubuntu` | Runs the CacheFlow v3 server and holds the prefix-cache pool |
| **cloudlab x86 server** | 172.16.3.234 | `cloudlab` | Only hosts the BF3 card over PCIe; CacheFlow does not use it |

All four accept SSH key logins and have passwordless sudo. Ask a lab admin for access.

## Links

| Link | Speed | Addresses | State |
|---|---|---|---|
| spark2 `enp1s0f0np0` ↔ BF3 `p1` | 100G QSFP | 10.0.1.1 ↔ 10.0.1.2 | **CacheFlow data path** (RoCE, MTU 9000). ~12 GB/s both directions. |
| Spark1 `enp1s0f1np1` ↔ BF3 `p0` | 40G | 10.0.2.1 ↔ 10.0.2.2 | **CacheFlow data path** (RoCE, MTU 9000). 4.8 GB/s BF3→Spark1; Spark1→BF3 capped at 1.7 GB/s (see known issues). |
| All four machines ↔ 172.16.3.0/23 | — | see above | SSH and management |
| x86 server ↔ BF3 | PCIe | none | Host functions `mlx5_0/1` visible, no IPs, unused |

## How the BF3 side is wired

The BF3 runs in DPU (switchdev) mode. Traffic arriving on a physical port (`p0`/`p1`) does not reach
the Arm cores unless the embedded switch forwards it. RoCE cannot terminate on the uplink port
itself, so each link uses its own **scalable function (SF)** on the Arm side:

```
spark2 ─ wire ─ p1 ─[tc redirect, in NIC hardware]─ en3f1pf1sf0 ─ enp3s0f1s0 (mlx5_3) 10.0.1.2 ─┐
Spark1 ─ wire ─ p0 ─[tc redirect, in NIC hardware]─ en3f0pf0sf0 ─ enp3s0f0s0 (mlx5_2) 10.0.2.2 ─┤
                                                                                               │
                                        CacheFlow v3 server: one pool, registered on both devices,
                                        one shared prefix index (a prefix saved by one Spark is a
                                        cache hit for the other)
```

OVS is masked on this BF3 on purpose (other projects rely on that), so the forwarding uses `tc`
rules instead of an OVS bridge. Do not unmask or start `openvswitch-switch` without checking
with the other users of this card.

## Notes

- On each DGX Spark, every physical ConnectX-7 port shows up as two interfaces (`enp1s0…` and
  `enP2p1s0…`), one per half of a split PCIe connection. They are the same port on the wire.
- spark2 also has Tailscale (100.105.244.64) and Docker (172.17.0.1) interfaces, left out of the diagram.

## Known issues (check before benchmarking)

Measured 2026-09-24 with `python -m cacheflowv3.tools.rdma_bw` (raw RDMA) and
`python -m cacheflowv3.tools.bench_transfer` (full KV save/load path, Qwen3-4B geometry).

| Path | Raw RDMA | KV save / load (pull mode) |
|---|---|---|
| spark2 ↔ BF3 (100G) | 12.1 GB/s both ways (~97% of line rate) | 11.8 / 11.8 GB/s |
| Spark1 ↔ BF3 (40G) | **1.68 GB/s Spark1→BF3**, 4.86 GB/s BF3→Spark1 | **1.66** / 4.8 GB/s |

1. **Spark1 → BF3 traffic is capped at ~1.68 GB/s** (~13 Gb/s of 40G), flat across message
   sizes and independent of which side drives the RDMA. No drops, pause frames, CNPs or
   retransmits; no rate limits in devlink or `mlnx_qos` on either end; same NIC firmware
   (28.45.4028) on both Sparks. Still unexplained. It only slows Spark1's (asynchronous) saves.
   Next things to try: swap which BF3 port/SF Spark1 uses, or check BF3 `p0` buffer settings.
2. **BF3-initiated writes to spark2 ("push" mode loads) reach only ~3.3 GB/s**, while spark2's
   own reads of the same data reach 11.8 GB/s. Only affects the push-mode ablation on spark2.
3. The BF3 limits locked memory to ~4 GB for normal users, so the CacheFlow server runs as root.
4. perftest versions differ between the Sparks (6.28) and the BF3 (6.27), so `ib_write_bw` and similar
   tools fail between them. Use `cacheflowv3.tools.rdma_bw` or `rping`.

Fixed on 2026-09-24 (check again after hardware changes):

- spark2's ConnectX-7 PCIe link was stuck at Gen1 (2.5 GT/s, ~0.4–0.9 GB/s). A reboot retrained it
  to 32 GT/s. Check with `sudo lspci -vv -s 0000:01:00.0 | grep LnkSta`.
- spark2 had 92 GB reserved as hugepages (DPDK leftover, runtime-only); gone after the reboot.
- Staging memory on 4 KB pages capped inbound RDMA on the Sparks at 3.3 GB/s (IOMMU translation per
  page). `libcfrdma` now allocates 2 MB-aligned transparent huge pages. The BF3 runs a 64 KB-page
  kernel, so its pool is unaffected.

## Software on the GPU hosts

Both Sparks have the same vLLM environment at `~/vllm_env` (Python 3.12, vLLM 0.20.1, torch 2.11.0+cu130)
with `cacheflowv3` installed from `~/cacheflow2/cacheflowv3`. Spark1 also needed `python3.12-dev`
(Triton compiles a C helper at startup). Spark1 has Qwen3-0.6B and Qwen3-4B cached.

## Restoring after a reboot

The 10.0.1.x addresses, MTU 9000 and the BF3 `tc` forwarding are **not persistent**. From spark2, in this repo:

```bash
ssh ubuntu@172.16.3.189 'sudo bash -s bf3' < cacheflowv3/scripts/setup_network.sh   # BF3: both links
sudo cacheflowv3/scripts/setup_network.sh spark                                     # spark2 (picks its link by hostname)
ssh spark1@172.16.3.100 'sudo bash -s spark' < cacheflowv3/scripts/setup_network.sh # Spark1
cacheflowv3/scripts/setup_network.sh check                                          # ping + RoCE MTU
CF_BF3_SSH=ubuntu@172.16.3.189 cacheflowv3/scripts/deploy_bf3.sh start --pool-gb 16  # CacheFlow server
```

(`CF_BF3_SSH` can be any SSH destination for the BF3, such as an alias from your `~/.ssh/config`.)

See `cacheflowv3/README.md` for running CacheFlow v3 itself.
