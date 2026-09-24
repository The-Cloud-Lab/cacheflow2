# SPDX-License-Identifier: Apache-2.0
"""NIC-registered staging memory on the GPU host.

On the DGX Spark (GB10) the GPU and CPU share LPDDR5X, and GPUDirect RDMA into
cudaMalloc memory is not available. Staging memory is therefore host memory
that is (1) registered with the NIC (ibv_reg_mr) and (2) registered with CUDA
(cudaHostRegister, mapped), which gives two views of the same bytes:

  cuda  - a CUDA tensor the GPU can gather into / scatter from directly
          (data_path="mapped": the NIC and the GPU share one buffer)
  cpu   - a pinned CPU tensor for copy-engine D2H/H2D (data_path="staged")

use_registered_buffer_pool=True  : one large region registered once at
                                   startup, sub-allocated per transfer
use_registered_buffer_pool=False : every transfer allocates, pins and
                                   registers its own buffer (ablation baseline)
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import torch

from . import rdma
from .server.store import ExtentAllocator

_PAGE = 2 << 20
_CUDA_HOST_REGISTER_PORTABLE_MAPPED = 3


class _CudaArray:
    """Minimal __cuda_array_interface__ exporter for a raw device pointer."""

    def __init__(self, ptr: int, nbytes: int):
        self.__cuda_array_interface__ = {
            "shape": (nbytes,),
            "typestr": "|u1",
            "data": (ptr, False),
            "version": 3,
            "strides": None,
        }


class _PinnedRegion:
    """Host memory registered with both CUDA and the NIC."""

    def __init__(self, dev: rdma.Device, nbytes: int, device: torch.device):
        # No hugetlbfs pages here: cudaHostRegister rejects them on GB10 (driver
        # 580). Transparent huge pages are still requested via madvise.
        self.buf = rdma.HostBuffer(nbytes, try_hugepages=False)
        err = torch.cuda.cudart().cudaHostRegister(
            self.buf.addr, nbytes, _CUDA_HOST_REGISTER_PORTABLE_MAPPED
        )
        if int(err) != 0:
            self.buf.close()
            raise RuntimeError(f"cudaHostRegister failed: {err}")
        try:
            self.mr = dev.register(self.buf.addr, nbytes)
        except Exception:
            torch.cuda.cudart().cudaHostUnregister(self.buf.addr)
            self.buf.close()
            raise
        with torch.cuda.device(device):
            self.cuda = torch.as_tensor(
                _CudaArray(self.buf.addr, nbytes), device=device
            )
        self.cpu = torch.frombuffer(self.buf.as_numpy(), dtype=torch.uint8)
        self.addr, self.nbytes = self.buf.addr, nbytes

    def close(self) -> None:
        self.mr.close()
        torch.cuda.cudart().cudaHostUnregister(self.buf.addr)
        self.cuda = self.cpu = None
        self.buf.close()


@dataclass
class StagingSlice:
    """A contiguous piece of staging memory handed to one transfer."""

    addr: int
    nbytes: int
    lkey: int
    rkey: int
    cuda: torch.Tensor  # uint8 CUDA view
    cpu: torch.Tensor  # uint8 pinned CPU view
    _offset: int = -1  # offset in the shared pool, or -1 for a private region
    _region: _PinnedRegion | None = None
    setup_ms: float = 0.0  # time spent pinning/registering (private regions)


class StagingPool:
    def __init__(
        self, dev: rdma.Device, nbytes: int, registered_pool: bool, device: torch.device
    ):
        self.dev, self.device = dev, device
        self.registered_pool = registered_pool
        self._cv = threading.Condition()
        self._region: _PinnedRegion | None = None
        self._alloc: ExtentAllocator | None = None
        if registered_pool:
            t = time.perf_counter()
            self._region = _PinnedRegion(dev, nbytes, device)
            self._alloc = ExtentAllocator(nbytes, _PAGE)
            self.setup_ms = (time.perf_counter() - t) * 1e3
        self.capacity = nbytes
        self.stats = {
            "acquired": 0,
            "private": 0,
            "waits": 0,
            "rejected": 0,
            "private_setup_ms": 0.0,
        }

    @property
    def pinned_ok(self) -> bool:
        return self._region is None or bool(self._region.cpu.is_pinned())

    def acquire(self, nbytes: int, wait_ms: int = 0) -> StagingSlice | None:
        """Get nbytes of staging memory, waiting up to wait_ms for space.
        Returns None if it cannot be satisfied."""
        if not self.registered_pool:
            t = time.perf_counter()
            region = _PinnedRegion(self.dev, nbytes, self.device)
            ms = (time.perf_counter() - t) * 1e3
            self.stats["private"] += 1
            self.stats["private_setup_ms"] += ms
            return StagingSlice(
                region.addr,
                nbytes,
                region.mr.lkey,
                region.mr.rkey,
                region.cuda,
                region.cpu,
                -1,
                region,
                ms,
            )
        if nbytes > self.capacity:
            self.stats["rejected"] += 1
            return None
        deadline = time.monotonic() + wait_ms / 1000.0
        with self._cv:
            while True:
                off = self._alloc.alloc(nbytes)
                if off is not None:
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self.stats["rejected"] += 1
                    return None
                self.stats["waits"] += 1
                self._cv.wait(remaining)
        r = self._region
        self.stats["acquired"] += 1
        return StagingSlice(
            r.addr + off,
            nbytes,
            r.mr.lkey,
            r.mr.rkey,
            r.cuda[off : off + nbytes],
            r.cpu[off : off + nbytes],
            off,
        )

    def release(self, s: StagingSlice) -> None:
        if s._region is not None:
            s._region.close()
            return
        with self._cv:
            self._alloc.free(s._offset, s.nbytes)
            self._cv.notify_all()

    def close(self) -> None:
        if self._region is not None:
            self._region.close()
            self._region = None
