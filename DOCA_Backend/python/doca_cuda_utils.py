"""
CUDA Memory Utilities for DOCA KV Cache Offload

This module provides utilities for managing pinned host memory and
GPU-to-host transfers needed for the DOCA offload pipeline.

Data Flow:
    GPU VRAM (KV cache) -> Host Pinned RAM -> DPU RAM (via DOCA DMA)

Since RTX 4090 doesn't support GPUDirect RDMA, we need an intermediate
pinned buffer on the host for the DMA transfer.
"""

import ctypes
from ctypes import c_void_p, c_size_t, c_int, POINTER
from typing import Optional, Dict, Tuple, List
import threading


# Try to import torch for tensor operations
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# CUDA Runtime API bindings
class CUDARuntime:
    """CUDA Runtime API bindings via ctypes."""

    # CUDA error codes
    cudaSuccess = 0
    cudaErrorMemoryAllocation = 2

    # Memory copy kinds
    cudaMemcpyHostToHost = 0
    cudaMemcpyHostToDevice = 1
    cudaMemcpyDeviceToHost = 2
    cudaMemcpyDeviceToDevice = 3

    # Host alloc flags
    cudaHostAllocDefault = 0
    cudaHostAllocPortable = 1
    cudaHostAllocMapped = 2
    cudaHostAllocWriteCombined = 4

    _lib = None

    @classmethod
    def _load_library(cls):
        """Load the CUDA runtime library."""
        if cls._lib is not None:
            return cls._lib

        lib_names = [
            "libcudart.so",
            "libcudart.so.12",
            "libcudart.so.11",
            "/usr/local/cuda/lib64/libcudart.so",
        ]

        for name in lib_names:
            try:
                cls._lib = ctypes.CDLL(name)
                cls._setup_functions()
                return cls._lib
            except OSError:
                continue

        raise RuntimeError(
            "Could not load CUDA runtime library. "
            "Make sure CUDA is installed and in LD_LIBRARY_PATH"
        )

    @classmethod
    def _setup_functions(cls):
        """Set up function signatures."""
        lib = cls._lib

        # cudaMallocHost
        lib.cudaMallocHost.argtypes = [POINTER(c_void_p), c_size_t]
        lib.cudaMallocHost.restype = c_int

        # cudaHostAlloc (with flags)
        lib.cudaHostAlloc.argtypes = [POINTER(c_void_p), c_size_t, c_int]
        lib.cudaHostAlloc.restype = c_int

        # cudaFreeHost
        lib.cudaFreeHost.argtypes = [c_void_p]
        lib.cudaFreeHost.restype = c_int

        # cudaMemcpy
        lib.cudaMemcpy.argtypes = [c_void_p, c_void_p, c_size_t, c_int]
        lib.cudaMemcpy.restype = c_int

        # cudaMemcpyAsync
        lib.cudaMemcpyAsync.argtypes = [c_void_p, c_void_p, c_size_t, c_int, c_void_p]
        lib.cudaMemcpyAsync.restype = c_int

        # cudaStreamSynchronize
        lib.cudaStreamSynchronize.argtypes = [c_void_p]
        lib.cudaStreamSynchronize.restype = c_int

        # cudaStreamCreate
        lib.cudaStreamCreate.argtypes = [POINTER(c_void_p)]
        lib.cudaStreamCreate.restype = c_int

        # cudaStreamDestroy
        lib.cudaStreamDestroy.argtypes = [c_void_p]
        lib.cudaStreamDestroy.restype = c_int

        # cudaDeviceSynchronize
        lib.cudaDeviceSynchronize.argtypes = []
        lib.cudaDeviceSynchronize.restype = c_int

    @classmethod
    def check_error(cls, result: int, msg: str = ""):
        """Check CUDA result and raise on error."""
        if result != cls.cudaSuccess:
            raise RuntimeError(f"CUDA error {result}: {msg}")

    @classmethod
    def malloc_host(cls, size: int, flags: int = 0) -> int:
        """
        Allocate pinned (page-locked) host memory.

        Args:
            size: Size in bytes
            flags: Allocation flags (default=0 for standard pinned)

        Returns:
            Host pointer as integer
        """
        lib = cls._load_library()
        ptr = c_void_p()

        if flags == 0:
            result = lib.cudaMallocHost(ctypes.byref(ptr), c_size_t(size))
        else:
            result = lib.cudaHostAlloc(ctypes.byref(ptr), c_size_t(size), c_int(flags))

        cls.check_error(result, f"cudaMallocHost({size} bytes)")
        return ptr.value

    @classmethod
    def free_host(cls, ptr: int) -> None:
        """Free pinned host memory."""
        lib = cls._load_library()
        result = lib.cudaFreeHost(c_void_p(ptr))
        cls.check_error(result, f"cudaFreeHost(0x{ptr:x})")

    @classmethod
    def memcpy_dtoh(cls, dst: int, src: int, size: int, stream: Optional[int] = None) -> None:
        """
        Copy from device to host.

        Args:
            dst: Destination host pointer
            src: Source device pointer
            size: Size in bytes
            stream: Optional CUDA stream (None for synchronous)
        """
        lib = cls._load_library()

        if stream is None:
            result = lib.cudaMemcpy(
                c_void_p(dst), c_void_p(src), c_size_t(size),
                c_int(cls.cudaMemcpyDeviceToHost)
            )
            cls.check_error(result, f"cudaMemcpy D2H ({size} bytes)")
        else:
            result = lib.cudaMemcpyAsync(
                c_void_p(dst), c_void_p(src), c_size_t(size),
                c_int(cls.cudaMemcpyDeviceToHost), c_void_p(stream)
            )
            cls.check_error(result, f"cudaMemcpyAsync D2H ({size} bytes)")

    @classmethod
    def memcpy_htod(cls, dst: int, src: int, size: int, stream: Optional[int] = None) -> None:
        """
        Copy from host to device.

        Args:
            dst: Destination device pointer
            src: Source host pointer
            size: Size in bytes
            stream: Optional CUDA stream
        """
        lib = cls._load_library()

        if stream is None:
            result = lib.cudaMemcpy(
                c_void_p(dst), c_void_p(src), c_size_t(size),
                c_int(cls.cudaMemcpyHostToDevice)
            )
            cls.check_error(result, f"cudaMemcpy H2D ({size} bytes)")
        else:
            result = lib.cudaMemcpyAsync(
                c_void_p(dst), c_void_p(src), c_size_t(size),
                c_int(cls.cudaMemcpyHostToDevice), c_void_p(stream)
            )
            cls.check_error(result, f"cudaMemcpyAsync H2D ({size} bytes)")

    @classmethod
    def stream_synchronize(cls, stream: int) -> None:
        """Synchronize a CUDA stream."""
        lib = cls._load_library()
        result = lib.cudaStreamSynchronize(c_void_p(stream))
        cls.check_error(result, "cudaStreamSynchronize")

    @classmethod
    def device_synchronize(cls) -> None:
        """Synchronize all device operations."""
        lib = cls._load_library()
        result = lib.cudaDeviceSynchronize()
        cls.check_error(result, "cudaDeviceSynchronize")

    @classmethod
    def stream_create(cls) -> int:
        """Create a CUDA stream."""
        lib = cls._load_library()
        stream = c_void_p()
        result = lib.cudaStreamCreate(ctypes.byref(stream))
        cls.check_error(result, "cudaStreamCreate")
        return stream.value

    @classmethod
    def stream_destroy(cls, stream: int) -> None:
        """Destroy a CUDA stream."""
        lib = cls._load_library()
        result = lib.cudaStreamDestroy(c_void_p(stream))
        cls.check_error(result, "cudaStreamDestroy")


class PinnedBuffer:
    """
    A pinned host memory buffer for DMA transfers.

    This buffer can be used as an intermediate staging area for
    GPU -> Host -> DPU transfers.
    """

    def __init__(self, size: int, alignment: int = 64):
        """
        Allocate a pinned buffer.

        Args:
            size: Size in bytes
            alignment: Alignment for DMA (default 64 bytes for DOCA)
        """
        # Align size up
        self.size = (size + alignment - 1) & ~(alignment - 1)
        self.alignment = alignment

        # Allocate pinned memory
        self.ptr = CUDARuntime.malloc_host(self.size)
        self._freed = False

    @property
    def address(self) -> int:
        """Get the buffer address as an integer."""
        return self.ptr

    def copy_from_device(self, device_ptr: int, size: Optional[int] = None,
                         offset: int = 0, stream: Optional[int] = None) -> None:
        """
        Copy data from GPU to this buffer.

        Args:
            device_ptr: GPU memory pointer
            size: Size to copy (default: buffer size)
            offset: Offset in this buffer
            stream: Optional CUDA stream for async copy
        """
        if self._freed:
            raise RuntimeError("Buffer has been freed")

        if size is None:
            size = self.size - offset

        if offset + size > self.size:
            raise ValueError(f"Copy exceeds buffer: offset={offset}, size={size}, buf_size={self.size}")

        CUDARuntime.memcpy_dtoh(self.ptr + offset, device_ptr, size, stream)

    def copy_to_device(self, device_ptr: int, size: Optional[int] = None,
                       offset: int = 0, stream: Optional[int] = None) -> None:
        """
        Copy data from this buffer to GPU.

        Args:
            device_ptr: GPU memory pointer
            size: Size to copy
            offset: Offset in this buffer
            stream: Optional CUDA stream
        """
        if self._freed:
            raise RuntimeError("Buffer has been freed")

        if size is None:
            size = self.size - offset

        if offset + size > self.size:
            raise ValueError(f"Copy exceeds buffer")

        CUDARuntime.memcpy_htod(device_ptr, self.ptr + offset, size, stream)

    def free(self) -> None:
        """Free the pinned buffer."""
        if not self._freed:
            CUDARuntime.free_host(self.ptr)
            self._freed = True
            self.ptr = 0

    def __del__(self):
        self.free()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.free()
        return False


class PinnedBufferPool:
    """
    Pool of pinned buffers for efficient reuse.

    Instead of allocating/freeing pinned memory for each transfer,
    this pool maintains a set of reusable buffers.
    """

    def __init__(self, buffer_size: int, num_buffers: int = 4):
        """
        Create a buffer pool.

        Args:
            buffer_size: Size of each buffer
            num_buffers: Number of buffers to pre-allocate
        """
        self.buffer_size = buffer_size
        self._lock = threading.Lock()
        self._available: List[PinnedBuffer] = []
        self._in_use: Dict[int, PinnedBuffer] = {}

        # Pre-allocate buffers
        for _ in range(num_buffers):
            buf = PinnedBuffer(buffer_size)
            self._available.append(buf)

    def acquire(self) -> PinnedBuffer:
        """
        Acquire a buffer from the pool.

        Returns:
            A PinnedBuffer. Call release() when done.
        """
        with self._lock:
            if self._available:
                buf = self._available.pop()
            else:
                # Pool exhausted, allocate a new one
                buf = PinnedBuffer(self.buffer_size)

            self._in_use[buf.ptr] = buf
            return buf

    def release(self, buf: PinnedBuffer) -> None:
        """Return a buffer to the pool."""
        with self._lock:
            if buf.ptr in self._in_use:
                del self._in_use[buf.ptr]
                self._available.append(buf)

    def close(self) -> None:
        """Free all buffers in the pool."""
        with self._lock:
            for buf in self._available:
                buf.free()
            for buf in self._in_use.values():
                buf.free()
            self._available.clear()
            self._in_use.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


# PyTorch integration utilities
if HAS_TORCH:
    def get_tensor_data_ptr(tensor: torch.Tensor) -> int:
        """Get the raw data pointer from a PyTorch tensor."""
        return tensor.data_ptr()

    def tensor_to_pinned_buffer(tensor: torch.Tensor, buf: PinnedBuffer,
                                stream: Optional[int] = None) -> None:
        """
        Copy a CUDA tensor to a pinned buffer.

        Args:
            tensor: Source CUDA tensor
            buf: Destination pinned buffer
            stream: Optional CUDA stream
        """
        if not tensor.is_cuda:
            raise ValueError("Tensor must be on CUDA device")

        size = tensor.numel() * tensor.element_size()
        if size > buf.size:
            raise ValueError(f"Tensor size ({size}) exceeds buffer size ({buf.size})")

        buf.copy_from_device(tensor.data_ptr(), size, stream=stream)

    def pinned_buffer_to_tensor(buf: PinnedBuffer, tensor: torch.Tensor,
                                size: Optional[int] = None,
                                stream: Optional[int] = None) -> None:
        """
        Copy from pinned buffer to a CUDA tensor.

        Args:
            buf: Source pinned buffer
            tensor: Destination CUDA tensor
            size: Size to copy (default: tensor size)
            stream: Optional CUDA stream
        """
        if not tensor.is_cuda:
            raise ValueError("Tensor must be on CUDA device")

        if size is None:
            size = tensor.numel() * tensor.element_size()

        buf.copy_to_device(tensor.data_ptr(), size, stream=stream)


if __name__ == "__main__":
    # Test pinned buffer allocation
    print("Testing CUDA pinned memory allocation...")

    buf = PinnedBuffer(1024 * 1024)  # 1MB
    print(f"Allocated pinned buffer at 0x{buf.address:x}, size={buf.size}")

    buf.free()
    print("Buffer freed successfully")

    # Test pool
    print("\nTesting buffer pool...")
    with PinnedBufferPool(1024 * 1024, num_buffers=2) as pool:
        b1 = pool.acquire()
        b2 = pool.acquire()
        print(f"Acquired buffers: 0x{b1.address:x}, 0x{b2.address:x}")
        pool.release(b1)
        pool.release(b2)

    print("Pool test completed")
