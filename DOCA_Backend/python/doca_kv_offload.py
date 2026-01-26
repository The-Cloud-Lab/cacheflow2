"""
DOCA KV Cache Offload Client for vLLM/Cacheflow

This module provides Python bindings for the DOCA-based KV cache offload system.
It enables offloading KV blocks from GPU to BlueField DPU over PCIe using DMA.

Data Flow (since RTX 4090 doesn't support GPUDirect RDMA):
    GPU VRAM (KV cache) -> Host Pinned RAM (CUDA memcpy) -> DPU RAM (DOCA DMA)

Usage:
    from doca_kv_offload import DOCAKVOffloadClient

    client = DOCAKVOffloadClient(pci_addr="0000:03:00.0")

    # Register a pinned buffer
    buffer_id = client.register_buffer(pinned_ptr, size)

    # Transfer to DPU
    transfer_id = client.transfer(buffer_id, offset=0, length=size)
    client.wait_transfer(transfer_id)

    client.close()
"""

import ctypes
import os
from ctypes import (
    c_void_p, c_char_p, c_size_t, c_uint64, c_uint32, c_double,
    POINTER, Structure, byref
)
from typing import Optional, Tuple
from pathlib import Path

# Find the shared library
def _find_library() -> str:
    """Find the libhost_provider shared library."""
    # Check common locations
    search_paths = [
        Path(__file__).parent.parent / "build" / "host" / "libhost_provider.so",
        Path(__file__).parent.parent / "build" / "libhost_provider.so",
        Path("/usr/local/lib/libhost_provider.so"),
        Path("/opt/mellanox/doca/lib/libhost_provider.so"),
    ]

    # Also check LD_LIBRARY_PATH
    ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    for path in ld_path.split(":"):
        if path:
            search_paths.append(Path(path) / "libhost_provider.so")

    for path in search_paths:
        if path.exists():
            return str(path)

    # Try to find .a and suggest building .so
    static_lib = Path(__file__).parent.parent / "build" / "host" / "libhost_provider.a"
    if static_lib.exists():
        raise FileNotFoundError(
            f"Found static library at {static_lib}, but need shared library (.so). "
            "Rebuild with: cmake -DBUILD_SHARED_LIBS=ON .. && make"
        )

    raise FileNotFoundError(
        "Could not find libhost_provider.so. Build with: cd build && cmake .. && make"
    )


# DOCA error codes (from doca_error.h)
class DOCAError:
    SUCCESS = 0
    UNKNOWN = 1
    INVALID_VALUE = 2
    NO_MEMORY = 3
    IO_FAILED = 4
    NOT_FOUND = 5
    TIME_OUT = 6
    IN_PROGRESS = 7
    INITIALIZATION = 8
    AGAIN = 9

    @classmethod
    def check(cls, result: int, msg: str = ""):
        """Check DOCA result and raise exception on error."""
        if result != cls.SUCCESS:
            error_names = {
                cls.INVALID_VALUE: "Invalid value",
                cls.NO_MEMORY: "No memory",
                cls.IO_FAILED: "IO failed",
                cls.NOT_FOUND: "Not found",
                cls.TIME_OUT: "Timeout",
                cls.IN_PROGRESS: "In progress",
                cls.INITIALIZATION: "Initialization failed",
                cls.AGAIN: "Try again",
            }
            error_name = error_names.get(result, f"Unknown error ({result})")
            raise RuntimeError(f"DOCA error: {error_name}. {msg}")


# Statistics structure matching C definition
class TransferStats(Structure):
    """Transfer statistics from DOCA backend."""
    _fields_ = [
        ("total_transfers", c_uint64),
        ("total_bytes", c_uint64),
        ("failed_transfers", c_uint64),
        ("avg_latency_us", c_double),
        ("peak_bandwidth_gbps", c_double),
    ]

    def __repr__(self):
        return (
            f"TransferStats(transfers={self.total_transfers}, "
            f"bytes={self.total_bytes}, failed={self.failed_transfers}, "
            f"avg_latency={self.avg_latency_us:.2f}us, "
            f"peak_bw={self.peak_bandwidth_gbps:.2f}Gbps)"
        )


class DOCAKVOffloadClient:
    """
    DOCA-based KV cache offload client for vLLM.

    This client manages the host-side of the KV cache offload system,
    communicating with the DPU offloader service over DOCA ComCh.

    Args:
        pci_addr: PCI address of the BlueField DPU (e.g., "0000:03:00.0")
        lib_path: Optional path to libhost_provider.so
    """

    def __init__(self, pci_addr: str, lib_path: Optional[str] = None):
        self.pci_addr = pci_addr
        self._provider = None
        self._registered_buffers = {}  # buffer_id -> (ptr, size)

        # Load the library
        if lib_path is None:
            lib_path = _find_library()

        self._lib = ctypes.CDLL(lib_path)
        self._setup_function_signatures()

        # Initialize the provider
        self._init_provider()

    def _setup_function_signatures(self):
        """Set up ctypes function signatures for the C library."""
        lib = self._lib

        # host_provider_init
        lib.host_provider_init.argtypes = [POINTER(c_void_p), c_char_p]
        lib.host_provider_init.restype = c_uint32

        # host_provider_register_buffer
        lib.host_provider_register_buffer.argtypes = [
            c_void_p, c_void_p, c_size_t, POINTER(c_uint64)
        ]
        lib.host_provider_register_buffer.restype = c_uint32

        # host_provider_transfer
        lib.host_provider_transfer.argtypes = [
            c_void_p, c_uint64, c_uint64, c_size_t, POINTER(c_uint64)
        ]
        lib.host_provider_transfer.restype = c_uint32

        # host_provider_wait_transfer
        lib.host_provider_wait_transfer.argtypes = [
            c_void_p, c_uint64, c_uint32
        ]
        lib.host_provider_wait_transfer.restype = c_uint32

        # host_provider_unregister_buffer
        lib.host_provider_unregister_buffer.argtypes = [c_void_p, c_uint64]
        lib.host_provider_unregister_buffer.restype = c_uint32

        # host_provider_get_stats
        lib.host_provider_get_stats.argtypes = [c_void_p, POINTER(TransferStats)]
        lib.host_provider_get_stats.restype = c_uint32

        # host_provider_destroy
        lib.host_provider_destroy.argtypes = [c_void_p]
        lib.host_provider_destroy.restype = None

    def _init_provider(self):
        """Initialize the DOCA host provider."""
        provider_ptr = c_void_p()
        result = self._lib.host_provider_init(
            byref(provider_ptr),
            self.pci_addr.encode('utf-8')
        )
        DOCAError.check(result, f"Failed to initialize provider at {self.pci_addr}")
        self._provider = provider_ptr

    def register_buffer(self, host_addr: int, size: int) -> int:
        """
        Register a host memory buffer for DMA access by the DPU.

        The buffer MUST be pinned memory (page-locked) for DMA to work.
        Use cuda.cuMemHostAlloc or torch.cuda.pin_memory() to allocate.

        Args:
            host_addr: Host memory address (must be pinned)
            size: Size of the buffer in bytes

        Returns:
            buffer_id: Unique ID for the registered buffer
        """
        if self._provider is None:
            raise RuntimeError("Provider not initialized")

        buffer_id = c_uint64()
        result = self._lib.host_provider_register_buffer(
            self._provider,
            c_void_p(host_addr),
            c_size_t(size),
            byref(buffer_id)
        )
        DOCAError.check(result, f"Failed to register buffer at 0x{host_addr:x}")

        self._registered_buffers[buffer_id.value] = (host_addr, size)
        return buffer_id.value

    def transfer(self, buffer_id: int, offset: int = 0,
                 length: Optional[int] = None) -> int:
        """
        Request a DMA transfer from host to DPU.

        Args:
            buffer_id: ID of the registered buffer
            offset: Offset within the buffer (bytes)
            length: Number of bytes to transfer (None = entire buffer from offset)

        Returns:
            transfer_id: ID for tracking the transfer
        """
        if self._provider is None:
            raise RuntimeError("Provider not initialized")

        if buffer_id not in self._registered_buffers:
            raise ValueError(f"Buffer {buffer_id} not registered")

        _, buf_size = self._registered_buffers[buffer_id]
        if length is None:
            length = buf_size - offset

        if offset + length > buf_size:
            raise ValueError(
                f"Transfer exceeds buffer size: offset={offset}, "
                f"length={length}, buffer_size={buf_size}"
            )

        transfer_id = c_uint64()
        result = self._lib.host_provider_transfer(
            self._provider,
            c_uint64(buffer_id),
            c_uint64(offset),
            c_size_t(length),
            byref(transfer_id)
        )
        DOCAError.check(result, f"Failed to initiate transfer for buffer {buffer_id}")

        return transfer_id.value

    def wait_transfer(self, transfer_id: int, timeout_ms: int = 5000) -> None:
        """
        Wait for a transfer to complete.

        Args:
            transfer_id: Transfer ID from transfer()
            timeout_ms: Timeout in milliseconds (0 = no timeout)

        Raises:
            RuntimeError: If transfer fails or times out
        """
        if self._provider is None:
            raise RuntimeError("Provider not initialized")

        result = self._lib.host_provider_wait_transfer(
            self._provider,
            c_uint64(transfer_id),
            c_uint32(timeout_ms)
        )
        DOCAError.check(result, f"Transfer {transfer_id} failed")

    def transfer_sync(self, buffer_id: int, offset: int = 0,
                      length: Optional[int] = None,
                      timeout_ms: int = 5000) -> None:
        """
        Synchronous transfer: initiate and wait for completion.

        Args:
            buffer_id: ID of the registered buffer
            offset: Offset within the buffer
            length: Number of bytes to transfer
            timeout_ms: Timeout in milliseconds
        """
        transfer_id = self.transfer(buffer_id, offset, length)
        self.wait_transfer(transfer_id, timeout_ms)

    def unregister_buffer(self, buffer_id: int) -> None:
        """
        Unregister a buffer.

        Args:
            buffer_id: ID of the buffer to unregister
        """
        if self._provider is None:
            raise RuntimeError("Provider not initialized")

        result = self._lib.host_provider_unregister_buffer(
            self._provider,
            c_uint64(buffer_id)
        )
        DOCAError.check(result, f"Failed to unregister buffer {buffer_id}")

        if buffer_id in self._registered_buffers:
            del self._registered_buffers[buffer_id]

    def get_stats(self) -> TransferStats:
        """
        Get transfer statistics.

        Returns:
            TransferStats with total_transfers, total_bytes, etc.
        """
        if self._provider is None:
            raise RuntimeError("Provider not initialized")

        stats = TransferStats()
        result = self._lib.host_provider_get_stats(
            self._provider,
            byref(stats)
        )
        DOCAError.check(result, "Failed to get statistics")
        return stats

    def close(self) -> None:
        """Close the connection and cleanup resources."""
        if self._provider is not None:
            self._lib.host_provider_destroy(self._provider)
            self._provider = None
            self._registered_buffers.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def __del__(self):
        self.close()


# Convenience function for detecting DPU PCI address
def find_bluefield_pci_address() -> Optional[str]:
    """
    Auto-detect BlueField DPU PCI address.

    Returns:
        PCI address string (e.g., "0000:03:00.0") or None if not found
    """
    import subprocess
    try:
        result = subprocess.run(
            ["lspci", "-D"],
            capture_output=True,
            text=True,
            check=True
        )
        for line in result.stdout.splitlines():
            if "BlueField" in line or "ConnectX" in line:
                # Extract PCI address (first field)
                pci_addr = line.split()[0]
                return pci_addr
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return None


if __name__ == "__main__":
    # Quick test
    pci = find_bluefield_pci_address()
    if pci:
        print(f"Found BlueField at: {pci}")
    else:
        print("No BlueField DPU found")
