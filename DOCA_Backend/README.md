# DOCA KV Cache Offload for vLLM

A high-performance KV cache offloading solution that uses NVIDIA DOCA to transfer data from the host (with consumer GPUs like RTX 4090) to BlueField DPU over PCIe, bypassing the CPU.

**🎉 NEW: Complete vLLM Integration** - Now includes full KVConnector implementation similar to LMCache, enabling seamless KV cache offloading to DPU!

## Quick Links

- **[Quick Start Guide](QUICKSTART.md)** - Get running in 5 minutes
- **[Build & Install Guide](BUILD_AND_INSTALL.md)** - Detailed setup instructions
- **[Example Usage](python/example_vllm_integration.py)** - vLLM integration example

## 🎯 Overview

This project enables **zero-copy** KV cache offloading from vLLM to a BlueField DPU using DOCA DMA and Communication Channel libraries. Since consumer GPUs (like RTX 4090) lack GPUDirect RDMA support, the data flow is:

```
GPU VRAM → [CUDA] → Host Pinned RAM → [DOCA DMA] → DPU RAM
```

### Key Features

- ⚡ **Hardware-accelerated DMA**: Uses DPU's DMA engine for PCIe transfers
- 🔄 **Zero-copy on host**: Direct pointer passing from PyTorch to DOCA (no extra memcpy)
- 🐍 **Python integration**: Seamless PyTorch/vLLM integration via pybind11
- 📊 **Performance monitoring**: Built-in transfer statistics and bandwidth tracking
- 🛡️ **Production-ready**: Robust error handling and logging

## 📋 Prerequisites

### Hardware Requirements

- **Host**: x86_64 server with PCIe slot
- **DPU**: NVIDIA BlueField-2 or BlueField-3 DPU
- **GPU**: NVIDIA RTX 4090 or similar (no GPUDirect RDMA required)
- **Connection**: DPU installed in host PCIe slot

### Software Requirements

- **DOCA SDK**: Version 2.0.0 or later ([Download](https://developer.nvidia.com/networking/doca))
- **OS**: Ubuntu 20.04/22.04 or RHEL 8/9
- **Compilers**:
  - GCC 9.0+ or Clang 10.0+
  - CMake 3.18+
- **Python**: 3.8+ (for Python wrapper)
- **PyTorch**: 2.0+ (for vLLM integration)

## 🚀 Installation

### 1. Install DOCA SDK

On both host and DPU:

```bash
# Download from https://developer.nvidia.com/networking/doca
# Follow NVIDIA's installation guide for your OS
# Default installation path: /opt/mellanox/doca
```

### 2. Clone and Build

#### On the Host (x86_64)

```bash
cd /home/biasbuster/cacheflow/DOCA_Backend

# Build host provider and Python wrapper
make all-host

# Or use CMake directly
mkdir -p build/host && cd build/host
cmake ../.. -DBUILD_HOST=ON -DBUILD_DPU=OFF
make -j$(nproc)
```

#### On the DPU (ARM/aarch64)

```bash
cd /home/biasbuster/cacheflow/DOCA_Backend

# Build DPU offloader
make all-dpu

# Or use CMake directly
mkdir -p build/dpu && cd build/dpu
cmake ../.. -DBUILD_HOST=OFF -DBUILD_DPU=ON
make -j$(nproc)
```

### 3. Install Python Package (Host only)

```bash
cd python
pip install -e .
```

## 🔧 Configuration

### Finding PCI Addresses

On the **host**, find the DPU's PCI address:

```bash
lspci | grep Mellanox
# Example output: 03:00.0 Ethernet controller: Mellanox Technologies MT42822 BlueField-2
```

On the **DPU**, find the host-facing interface:

```bash
lspci | grep Mellanox
# The DPU sees itself through a PCI interface
```

### Network Configuration

The communication channel uses TCP/IP. Ensure the host and DPU can communicate:

```bash
# On DPU: Check IP address
ip addr show

# On Host: Test connectivity
ping <DPU_IP>
```

## 🎮 Usage

### Step 1: Start DPU Offloader Service

On the **DPU**, start the offloader service:

```bash
# Run as root or with appropriate permissions
sudo ./build/dpu/dpu_offloader --server 0.0.0.0:6789
```

Expected output:
```
[INFO] DOCA KV Cache Offloader - DPU Service
[INFO] Server address: 0.0.0.0:6789
[INFO] DPU offloader initialized successfully
[INFO] Waiting for connections from host...
```

### Step 2: Use from Python/vLLM

On the **host**, use the Python API:

```python
import torch
import doca_kv_offload

# Initialize offloader with DPU PCI address
offloader = doca_kv_offload.DocaKVCacheOffloader()
offloader.init("0000:03:00.0")  # Adjust PCI address

# Create a pinned PyTorch tensor (simulating KV cache)
kv_cache = torch.randn(1024 * 1024 * 64, dtype=torch.float32, pin_memory=True)

# Register the tensor
buffer_id = offloader.register_torch_tensor(kv_cache)

# Transfer to DPU (blocking)
offloader.transfer_sync(buffer_id, offset=0, length=kv_cache.nbytes)

# Or transfer asynchronously
transfer_id = offloader.transfer(buffer_id, 0, kv_cache.nbytes)
# ... do other work ...
offloader.wait_transfer(transfer_id)

# Get statistics
stats = offloader.get_stats()
print(f"Bandwidth: {stats['peak_bandwidth_gbps']:.2f} Gbps")

# Cleanup
offloader.unregister_buffer(buffer_id)
```

### Step 3: Integration with vLLM

Example integration in your vLLM code:

```python
from vllm import LLM, SamplingParams
import doca_kv_offload

# Initialize DOCA offloader
offloader = doca_kv_offload.DocaKVCacheOffloader()
offloader.init("0000:03:00.0")

# Your vLLM setup
llm = LLM(model="meta-llama/Llama-2-7b-hf")

# Hook into KV cache management
class KVCacheOffloadManager:
    def __init__(self, offloader):
        self.offloader = offloader
        self.registered_buffers = {}
    
    def offload_kv_cache(self, kv_cache_tensor):
        """Offload KV cache tensor to DPU"""
        # Register if not already registered
        tensor_id = id(kv_cache_tensor)
        if tensor_id not in self.registered_buffers:
            buffer_id = self.offloader.register_torch_tensor(kv_cache_tensor)
            self.registered_buffers[tensor_id] = buffer_id
        else:
            buffer_id = self.registered_buffers[tensor_id]
        
        # Transfer to DPU
        self.offloader.transfer_sync(buffer_id, 0, kv_cache_tensor.nbytes)
        
        # Can now free GPU memory if needed
        return buffer_id

manager = KVCacheOffloadManager(offloader)

# During inference, when you need to offload:
# manager.offload_kv_cache(kv_cache_from_vllm)
```

## 📊 Performance

### Expected Bandwidth

- **PCIe Gen4 x8**: ~12-15 GB/s
- **PCIe Gen3 x16**: ~10-12 GB/s
- **Latency**: 50-200 μs (depending on transfer size)

### Benchmarking

Run the provided test programs:

```bash
# On host (requires DPU service running)
./build/host/host_provider_test

# Or use Python examples
python python/example_usage.py
```

## 🏗️ Architecture

### Components

1. **Host Provider** (`host/host_provider.cpp`)
   - Runs on x86 host alongside vLLM
   - Manages memory registration and export
   - Uses DOCA Comm Channel for control messages

2. **DPU Offloader** (`dpu/dpu_offloader.cpp`)
   - Runs on BlueField DPU ARM cores
   - Imports host memory and executes DMA
   - Uses DOCA DMA for hardware-accelerated transfers

3. **Python Wrapper** (`python/doca_wrapper.cpp`)
   - pybind11 bridge for PyTorch integration
   - Zero-copy pointer passing
   - Pythonic API

### Communication Protocol

The host and DPU communicate via DOCA Comm Channel with a custom protocol:

```
┌─────────┐                              ┌─────────┐
│  Host   │                              │   DPU   │
│Provider │                              │Offloader│
└────┬────┘                              └────┬────┘
     │                                        │
     │  MSG_REGISTER_BUFFER                   │
     │  (export descriptor + metadata)        │
     ├───────────────────────────────────────>│
     │                                        │
     │  MSG_ACK                               │
     │<───────────────────────────────────────┤
     │                                        │
     │  MSG_TRANSFER_REQUEST                  │
     │  (buffer_id, offset, length)           │
     ├───────────────────────────────────────>│
     │                                        │
     │                           [DMA Transfer]
     │                                        │
     │  MSG_TRANSFER_COMPLETE                 │
     │  (status, bytes transferred)           │
     │<───────────────────────────────────────┤
     │                                        │
```

### Memory Layout

```
┌─────────────────────────────────────────────────────┐
│                    Host x86                          │
│                                                      │
│  ┌──────────────┐         ┌──────────────┐          │
│  │  GPU VRAM    │  CUDA   │ Pinned RAM   │          │
│  │  (RTX 4090)  ├────────>│ (Registered) │          │
│  └──────────────┘         └──────┬───────┘          │
│                                  │                   │
└──────────────────────────────────┼───────────────────┘
                                   │ PCIe (DOCA DMA)
┌──────────────────────────────────┼───────────────────┐
│                    DPU ARM        │                   │
│                                  │                   │
│                           ┌──────▼───────┐           │
│                           │  DPU RAM     │           │
│                           │  (Cached KV) │           │
│                           └──────────────┘           │
│                                                      │
└─────────────────────────────────────────────────────┘
```

## 🐛 Troubleshooting

### Common Issues

#### 1. "Failed to open DOCA device"

**Cause**: Incorrect PCI address or device not accessible

**Solution**:
```bash
# List DOCA-compatible devices
doca_device_query

# Check permissions
sudo chmod 666 /dev/infiniband/uverbs*
```

#### 2. "Failed to connect to host server"

**Cause**: DPU cannot reach host or port blocked

**Solution**:
```bash
# On host: Check if port is open
netstat -tulpn | grep 6789

# Check firewall
sudo ufw allow 6789
```

#### 3. "Tensor must be pinned memory"

**Cause**: Trying to register non-pinned PyTorch tensor

**Solution**:
```python
# Always use pin_memory=True for tensors you want to offload
tensor = torch.randn(size, pin_memory=True)

# Or pin existing tensor
tensor = tensor.pin_memory()
```

#### 4. "Export descriptor too large"

**Cause**: DOCA export descriptor exceeds buffer size

**Solution**: This is a bug in the protocol. Increase `export_desc` size in `protocol.h`:
```c
uint8_t export_desc[512]; // Increase from 256
```

### Debugging

Enable debug logging:

```bash
# Set DOCA log level
export DOCA_LOG_LEVEL=DEBUG

# Run with verbose output
./build/dpu/dpu_offloader --server 0.0.0.0:6789 2>&1 | tee dpu.log
```

## 📚 API Reference

### Python API

#### `DocaKVCacheOffloader`

**Methods**:

- `init(pci_addr: str) -> None`
  - Initialize with DPU PCI address
  
- `register_buffer(tensor_ptr: int, size: int) -> int`
  - Register raw memory buffer, returns buffer ID
  
- `register_torch_tensor(tensor: torch.Tensor) -> int`
  - Register PyTorch tensor (convenience method)
  
- `transfer(buffer_id: int, offset: int, length: int) -> int`
  - Async transfer, returns transfer ID
  
- `wait_transfer(transfer_id: int, timeout_ms: int = 5000) -> None`
  - Wait for transfer completion
  
- `transfer_sync(buffer_id: int, offset: int, length: int, timeout_ms: int = 5000) -> None`
  - Blocking transfer
  
- `unregister_buffer(buffer_id: int) -> None`
  - Unregister buffer
  
- `get_stats() -> dict`
  - Get transfer statistics

### C/C++ API

See header files:
- `host/host_provider.h` - Host provider API
- `dpu/dpu_offloader.h` - DPU offloader API
- `common/protocol.h` - Communication protocol

## 🔬 Testing

Run the test suite:

```bash
# Host tests (requires DPU service running)
make test

# Or run specific tests
./build/host/host_provider_test 0000:03:00.0

# Python examples
python python/example_usage.py
```

## 📈 Roadmap

- [ ] Asynchronous batched transfers
- [ ] Compression support
- [ ] Multi-DPU support
- [ ] GPU memory pooling integration
- [ ] Prefetching and caching strategies
- [ ] Direct integration with vLLM PagedAttention

## 🔗 vLLM KVConnector Integration

This project includes a **complete vLLM KVConnectorBase_V1 implementation** (`DOCAConnectorV1`) that integrates the DOCA offload system with vLLM's distributed KV cache architecture, similar to how LMCache works.

### Features

✅ **Scheduler-side operations:**
- Prefix hash computation and matching
- Request tracking across forward passes
- Cache hit detection on DPU
- Metadata generation for workers

✅ **Worker-side operations:**
- KV cache layer extraction from vLLM's paged buffers
- Asynchronous DMA transfers to/from DPU
- Layer-by-layer loading and saving
- Error handling and recovery

✅ **Integration:**
- Automatic connector registration with vLLM
- Compatible with vLLM v1 API
- Support for both prefill and decode phases
- Seamless prefix caching across requests

### Configuration

To use DOCA offloading in vLLM, configure the KV transfer settings:

```python
from vllm import LLM

# Configure vLLM with DOCA connector
llm = LLM(
    model="meta-llama/Llama-2-7b-hf",
    kv_transfer_config={
        "kv_connector": "DOCAConnectorV1",
        "extra_config": {
            "dpu_pci_addr": "0000:03:00.0",  # Your DPU PCI address
            "block_size": 16 * 1024 * 1024,   # 16MB blocks
            "max_blocks": 256,                # Max cached blocks
            "async_transfers": True,          # Enable async transfers
        }
    }
)
```

### How It Works

1. **Prefix Caching**: The connector automatically detects and caches KV cache prefixes on the DPU
2. **Automatic Offloading**: Completed sequences are offloaded to DPU for future reuse
3. **Zero-Copy Loading**: Matching prefixes are loaded directly from DPU to GPU memory
4. **Performance**: Reduces GPU memory pressure and improves throughput for long sequences

### Architecture

```
vLLM Scheduler → DOCAConnectorV1 → KVOffloadManager → DOCA Client → DPU
     ↓
vLLM Worker → DOCAConnectorV1 → DMA Transfer → GPU Memory
```

The connector implements the full `KVConnectorBase_V1` interface, supporting both scheduler-side (prefix matching) and worker-side (load/save operations) functionality.

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch
3. Make your changes with tests
4. Submit a pull request

## 📄 License

This project is part of the CacheFlow system. See main repository for license details.

## 🙏 Acknowledgments

- NVIDIA DOCA team for the excellent documentation
- vLLM team for the inspiration
- BlueField DPU community

## 📞 Support

For issues and questions:
- Open an issue in the repository
- Check NVIDIA DOCA documentation: https://docs.nvidia.com/doca/
- BlueField community forums

## 📖 Additional Resources

- [DOCA DMA Programming Guide](https://docs.nvidia.com/doca/sdk/doca+dma/index.html)
- [DOCA Comm Channel Guide](https://docs.nvidia.com/doca/sdk/doca+comm+channel/index.html)
- [BlueField DPU Documentation](https://docs.nvidia.com/networking/display/bluefielddpuos)
- [vLLM Documentation](https://docs.vllm.ai/)

---

**Note**: This is an experimental project. Performance and reliability depend on your specific hardware configuration and DOCA version. Always test thoroughly in your environment before production use.
