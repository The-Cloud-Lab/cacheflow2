# CacheFlow

**Hardware-accelerated prefix caching for LLM inference on NVIDIA BlueField SmartNICs**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)](pyproject.toml)
[![DOCA](https://img.shields.io/badge/NVIDIA%20DOCA-2.0%2B-green)](https://developer.nvidia.com/networking/doca)

---

## Overview

CacheFlow extends [vLLM](https://github.com/vllm-project/vllm) with a **DOCA-powered KV cache offloading backend** that runs natively on NVIDIA BlueField-3 SmartNICs. By offloading prefix KV caches from GPU memory to the DPU over PCIe via hardware DMA, CacheFlow reduces GPU memory pressure and increases effective cache capacity.
The data path is fully hardware accelerated.


### What CacheFlow adds to vLLM

| Capability | vLLM (upstream) | CacheFlow |
|---|---|---|
| Prefix caching | In-GPU memory | Extended to BlueField DPU via DMA |
| KV cache capacity | Limited by GPU VRAM | Expanded to DPU RAM |
| Transfer mechanism | CPU memcpy | Hardware DMA (zero-copy on host) |
| SmartNIC integration | — | DOCA Backend (`DOCAConnectorV1`) |
| Async offload | — | Overlapped DMA + CUDA execution |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                          Host  (x86)                            │
│                                                                 │
│   ┌──────────────┐    CUDA     ┌──────────────────────────┐    │
│   │  GPU VRAM    │ ──────────► │  Host Pinned RAM          │    │
│   │  (RTX 4090)  │            │  (Registered with DOCA)   │    │
│   └──────────────┘            └────────────┬─────────────┘    │
│                                            │                   │
│   vLLM + DOCAConnectorV1                   │                   │
│   ┌──────────────────────────┐             │                   │
│   │  Scheduler               │             │  PCIe             │
│   │  • Prefix hash matching  │             │  (DOCA DMA)       │
│   │  • Cache hit detection   │             │                   │
│   │  • Request tracking      │             │                   │
│   └──────────────────────────┘             │                   │
└───────────────────────────────────────────┼───────────────────┘
                                            │
┌───────────────────────────────────────────┼───────────────────┐
│                   BlueField-3 DPU (ARM)    │                   │
│                                            │                   │
│                              ┌─────────────▼──────────┐       │
│                              │  DPU RAM                │       │
│                              │  (Prefix KV Cache Store)│       │
│                              └────────────────────────┘       │
│                                                                 │
│   DOCA_Backend / dpu_offloader                                  │
│   • DMA engine management                                       │
│   • Memory import & export                                      │
│   • Comm Channel (control plane)                                │
└─────────────────────────────────────────────────────────────────┘
```

### Components

| Component | Location | Description |
|---|---|---|
| `DOCAConnectorV1` | `vllm/distributed/kv_transfer/` | vLLM KVConnector plugin — scheduler & worker side |
| Host Provider | `DOCA_Backend/host/` | x86 DOCA DMA host agent |
| DPU Offloader | `DOCA_Backend/dpu/` | BlueField ARM service (runs on DPU) |
| Python Wrapper | `DOCA_Backend/python/` | pybind11 bridge to PyTorch / vLLM |

---

## Prerequisites

### Hardware

- **Host**: x86\_64 server with an available PCIe slot
- **SmartNIC**: NVIDIA BlueField-3 DPU (BlueField-2 also supported)
- **GPU**: Any NVIDIA GPU (GPUDirect RDMA **not** required; RTX 4090 tested)

### Software

| Dependency | Version | Notes |
|---|---|---|
| Ubuntu | 20.04 / 22.04 | or RHEL 8/9 |
| NVIDIA DOCA SDK | ≥ 2.0.0 | Install on both host and DPU |
| GCC / Clang | ≥ 9.0 / ≥ 10.0 | C++17 required |
| CMake | ≥ 3.18 | |
| Python | 3.10 – 3.13 | |
| PyTorch | ≥ 2.0 | With CUDA support |
| CUDA Toolkit | ≥ 11.8 | |

---

## Installation

### 1. Install NVIDIA DOCA SDK

Install DOCA on **both the host and the BlueField DPU** by following the official guide for your OS:

```
https://developer.nvidia.com/networking/doca
```

The default installation path is `/opt/mellanox/doca`.

### 2. Clone the Repository

```bash
git clone https://github.com/your-org/cacheflow.git
cd cacheflow
```

### 3. Install vLLM Dependencies

CacheFlow is built on top of vLLM. Install the base Python dependencies first:

```bash
pip install -r requirements/common.txt
```

### 4. Build the DOCA Backend

The DOCA Backend must be compiled separately on the **host** and the **DPU**.

#### On the Host (x86\_64)

```bash
cd DOCA_Backend

# Option A — convenience Makefile
make all-host

# Option B — CMake directly
mkdir -p build/host && cd build/host
cmake ../.. -DBUILD_HOST=ON -DBUILD_DPU=OFF
make -j$(nproc)
cd ../..
```

#### On the BlueField DPU (ARM / aarch64)

Cross-compile on the host or build natively on the DPU:

```bash
cd DOCA_Backend

# Option A — convenience Makefile
make all-dpu

# Option B — CMake directly
mkdir -p build/dpu && cd build/dpu
cmake ../.. -DBUILD_HOST=OFF -DBUILD_DPU=ON
make -j$(nproc)
cd ../..
```

### 5. Install the Python Wrapper (Host only)

```bash
cd DOCA_Backend/python
pip install -e .
cd ../..
```

### 6. Install CacheFlow / vLLM

```bash
# Development install from source
pip install -e .

# Or build a wheel
pip install --upgrade pip setuptools wheel
python setup.py bdist_wheel
pip install dist/vllm-*.whl
```

---

## Configuration

### Step 1 — Identify PCI Addresses

**On the host**, find the BlueField DPU's PCI address:

```bash
lspci | grep -i mellanox
# Example: 0c:00.0 Ethernet controller: Mellanox Technologies BlueField-3 ...
```

**On the DPU**, verify the host-facing interface:

```bash
lspci | grep -i mellanox
```

### Step 2 — Configure the KV Transfer

Edit `kv_transfer_config.json` (or pass inline to `LLM()`):

```json
{
  "kv_connector": "DOCAConnectorV1",
  "kv_role": "kv_both",
  "kv_rank": 0,
  "kv_parallel_size": 1,
  "kv_buffer_device": "cuda",
  "kv_buffer_size": 2000000000,
  "kv_connector_extra_config": {
    "dpu_pci_addr": "0000:0c:00.0",
    "tokens_per_block": 256,
    "block_size": 536870912,
    "max_blocks": 20,
    "common_prefix_num_tokens": 1000,
    "num_staging_buffers": 16,
    "async_transfers": true,
    "copy_stream_pool_size": 4,
    "overlap_dma_with_copy": true,
    "use_doca_buffer_pool": true,
    "load_prefix_map": true,
    "prefix_map_path": "/tmp/cacheflow_prefix_map.json"
  }
}
```

Key parameters:

| Parameter | Description |
|---|---|
| `dpu_pci_addr` | PCI address of the BlueField DPU (from `lspci`) |
| `block_size` | Size of each KV cache block in bytes |
| `max_blocks` | Maximum number of cached blocks on DPU |
| `async_transfers` | Overlap DMA transfers with CUDA execution |
| `common_prefix_num_tokens` | Token length threshold for prefix caching |

---

## Usage

### Step 1 — Start the DPU Offloader Service

On the **BlueField DPU**, run the offloader as root:

```bash
sudo ./DOCA_Backend/build/dpu/dpu_offloader --server 0.0.0.0:6789
```

Expected output:

```
[INFO] DOCA KV Cache Offloader - DPU Service
[INFO] Server address: 0.0.0.0:6789
[INFO] DPU offloader initialized successfully
[INFO] Waiting for connections from host...
```

### Step 2 — Run vLLM with the DOCA Connector

#### Using the config file

```bash
vllm serve meta-llama/Llama-3-8B \
    --kv-transfer-config kv_transfer_config.json
```

#### Inline Python

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3-8B",
    kv_transfer_config={
        "kv_connector": "DOCAConnectorV1",
        "kv_role": "kv_both",
        "kv_connector_extra_config": {
            "dpu_pci_addr": "0000:0c:00.0",
            "block_size": 536870912,
            "max_blocks": 20,
            "async_transfers": True,
        },
    },
)

prompts = ["Explain transformer attention in detail:"]
params  = SamplingParams(temperature=0.8, max_tokens=512)
outputs = llm.generate(prompts, params)

for output in outputs:
    print(output.outputs[0].text)
```

### Step 3 — Using the DOCA Python API Directly

For lower-level access or custom integration:

```python
import torch
import doca_kv_offload

# Initialise with DPU PCI address
offloader = doca_kv_offload.DocaKVCacheOffloader()
offloader.init("0000:0c:00.0")

# Create a pinned tensor (simulates a KV cache block)
kv_block = torch.randn(1024 * 1024 * 64, dtype=torch.float16, pin_memory=True)

# Register and transfer synchronously
buffer_id = offloader.register_torch_tensor(kv_block)
offloader.transfer_sync(buffer_id, offset=0, length=kv_block.nbytes)

# Or transfer asynchronously and overlap with other work
transfer_id = offloader.transfer(buffer_id, 0, kv_block.nbytes)
# ... do other work here ...
offloader.wait_transfer(transfer_id)

# Inspect bandwidth statistics
stats = offloader.get_stats()
print(f"Peak bandwidth: {stats['peak_bandwidth_gbps']:.2f} Gbps")

offloader.unregister_buffer(buffer_id)
```

---

## Performance

Achieved on PCIe Gen4 with BlueField-3:

| Metric | Value |
|---|---|
| Peak DMA bandwidth (PCIe Gen4 x16) | ~28 GB/s |
| Sustained throughput (PCIe Gen4 x8) | 12–15 GB/s |
| Transfer latency | 50–200 μs |
| Async overlap efficiency | ~90% (DMA hidden behind CUDA) |

Run the built-in benchmark:

```bash
# Requires DPU offloader running
./DOCA_Backend/build/host/host_provider_test

# Python benchmark
python DOCA_Backend/python/example_usage.py
```

---

## Troubleshooting

**"Failed to open DOCA device"**

```bash
# List DOCA-compatible devices
doca_device_query

# Fix permissions
sudo chmod 666 /dev/infiniband/uverbs*
```

**"Failed to connect to host server"**

```bash
# Verify port is open on host
netstat -tulpn | grep 6789

# Allow port through firewall
sudo ufw allow 6789
```

**"Tensor must be pinned memory"**

```python
# Always pin tensors before registering with DOCA
tensor = tensor.pin_memory()
buffer_id = offloader.register_torch_tensor(tensor)
```

Enable verbose DOCA logging for deeper diagnostics:

```bash
export DOCA_LOG_LEVEL=DEBUG
./DOCA_Backend/build/dpu/dpu_offloader --server 0.0.0.0:6789 2>&1 | tee dpu.log
```

---

## Project Structure

```
cacheflow/
├── DOCA_Backend/          # DOCA DMA host + DPU components
│   ├── host/              # x86 host provider (C++)
│   ├── dpu/               # BlueField DPU offloader service (C++)
│   ├── python/            # pybind11 Python wrapper
│   ├── common/            # Shared protocol definitions
│   └── scripts/           # Build helpers
├── vllm/                  # vLLM core (extended with CacheFlow connector)
│   └── distributed/
│       └── kv_transfer/   # DOCAConnectorV1 plugin
├── examples/              # Offline and online inference examples
├── benchmarks/            # Performance benchmarking tools
├── docs/                  # Full documentation (MkDocs)
├── tests/                 # Test suite
├── kv_transfer_config.json  # Reference connector configuration
└── CMakeLists.txt         # Root build configuration
```

---

## Acknowledgements

CacheFlow is built on top of **[vLLM](https://github.com/vllm-project/vllm)**, an open-source, high-throughput LLM inference engine originally developed at the [Sky Computing Lab](https://sky.cs.berkeley.edu) at UC Berkeley. vLLM provides the PagedAttention memory manager, continuous batching engine, and KVConnector plugin API that CacheFlow's DOCA backend extends. We are grateful to the entire vLLM community for their foundational work.

If you use this project in academic work, please also cite the original vLLM paper:

```bibtex
@inproceedings{kwon2023efficient,
  title     = {Efficient Memory Management for Large Language Model Serving with PagedAttention},
  author    = {Woosuk Kwon and Zhuohan Li and Siyuan Zhuang and Ying Sheng and Lianmin Zheng
               and Cody Hao Yu and Joseph E. Gonzalez and Hao Zhang and Ion Stoica},
  booktitle = {Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles},
  year      = {2023}
}
```

We also thank the NVIDIA DOCA and BlueField teams for the hardware acceleration platform that makes this work possible.

---

## Resources

- [NVIDIA DOCA SDK](https://developer.nvidia.com/networking/doca)
- [DOCA DMA Programming Guide](https://docs.nvidia.com/doca/sdk/doca+dma/index.html)
- [DOCA Comm Channel Guide](https://docs.nvidia.com/doca/sdk/doca+comm+channel/index.html)
- [BlueField DPU Documentation](https://docs.nvidia.com/networking/display/bluefielddpuos)
- [vLLM Documentation](https://docs.vllm.ai/)
- [vLLM GitHub](https://github.com/vllm-project/vllm)

---

## License

CacheFlow is released under the [Apache License 2.0](LICENSE), consistent with the upstream vLLM project.
