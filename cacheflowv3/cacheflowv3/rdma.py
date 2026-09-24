# SPDX-License-Identifier: Apache-2.0
"""ctypes binding for libcfrdma (csrc/cfrdma.c).

Imported by both the BlueField-3 server and the vLLM worker, so it must not
import torch or vllm. numpy is used only to build op arrays quickly.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np

OP_WRITE = 0
OP_READ = 1
MAX_PRIVATE_DATA = 56

# Mirrors struct cf_op (40 bytes, naturally aligned).
OP_DTYPE = np.dtype(
    [
        ("laddr", np.uint64),
        ("raddr", np.uint64),
        ("lkey", np.uint32),
        ("rkey", np.uint32),
        ("len", np.uint32),
        ("_pad", np.uint32),
    ]
)


class _EpStats(ctypes.Structure):
    _fields_ = [
        ("ops_posted", ctypes.c_uint64),
        ("ops_completed", ctypes.c_uint64),
        ("bytes_posted", ctypes.c_uint64),
        ("sq_full_stalls", ctypes.c_uint64),
    ]


class RdmaError(RuntimeError):
    pass


def _find_library() -> str:
    candidates = []
    if os.environ.get("CACHEFLOW_RDMA_LIB"):
        candidates.append(Path(os.environ["CACHEFLOW_RDMA_LIB"]))
    here = Path(__file__).resolve().parent
    candidates += [
        here / "lib" / "libcfrdma.so",
        here.parent / "build" / "libcfrdma.so",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    raise RdmaError(
        "libcfrdma.so not found; build it with `make -C cacheflowv3/csrc` "
        f"(looked in: {', '.join(map(str, candidates))})"
    )


_lib = None


def lib() -> ctypes.CDLL:
    global _lib
    if _lib is not None:
        return _lib
    L = ctypes.CDLL(_find_library())
    vp, c_int, c_char_p = ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p
    sigs = {
        "cf_last_error": (c_char_p, []),
        "cf_dev_open": (vp, [c_char_p, c_int]),
        "cf_dev_name": (c_char_p, [vp]),
        "cf_dev_close": (None, [vp]),
        "cf_reg_mr": (vp, [vp, vp, ctypes.c_size_t]),
        "cf_mr_lkey": (ctypes.c_uint32, [vp]),
        "cf_mr_rkey": (ctypes.c_uint32, [vp]),
        "cf_dereg_mr": (None, [vp]),
        "cf_alloc": (vp, [ctypes.c_size_t, c_int, ctypes.POINTER(c_int)]),
        "cf_free": (None, [vp, ctypes.c_size_t]),
        "cf_listen": (vp, [vp, c_char_p, c_int, c_int]),
        "cf_accept": (vp, [vp, c_int]),
        "cf_listener_close": (None, [vp]),
        "cf_connect": (vp, [vp, c_char_p, c_int, c_char_p, c_int, c_int]),
        "cf_ep_private_data": (c_int, [vp, vp, c_int]),
        "cf_post": (c_int, [vp, c_int, vp, c_int, ctypes.POINTER(ctypes.c_uint64)]),
        "cf_wait": (c_int, [vp, ctypes.c_uint64, c_int]),
        "cf_test": (c_int, [vp, ctypes.c_uint64]),
        "cf_ep_error": (c_int, [vp]),
        "cf_ep_get_stats": (None, [vp, ctypes.POINTER(_EpStats)]),
        "cf_ep_close": (None, [vp]),
    }
    for name, (res, args) in sigs.items():
        fn = getattr(L, name)
        fn.restype = res
        fn.argtypes = args
    _lib = L
    return L


def last_error() -> str:
    return lib().cf_last_error().decode(errors="replace")


class Device:
    """An RDMA device plus protection domain, located by IP address."""

    def __init__(self, ip: str, is_peer: bool = False):
        self._h = lib().cf_dev_open(ip.encode(), int(is_peer))
        if not self._h:
            raise RdmaError(f"cannot open RDMA device for {ip}: {last_error()}")

    @property
    def name(self) -> str:
        return lib().cf_dev_name(self._h).decode()

    def register(self, addr: int, length: int) -> MemoryRegion:
        return MemoryRegion(self, addr, length)

    def close(self) -> None:
        if self._h:
            lib().cf_dev_close(self._h)
            self._h = None


class MemoryRegion:
    def __init__(self, dev: Device, addr: int, length: int):
        self.addr = int(addr)
        self.length = int(length)
        self._h = lib().cf_reg_mr(dev._h, ctypes.c_void_p(self.addr), self.length)
        if not self._h:
            raise RdmaError(f"memory registration failed: {last_error()}")
        self.lkey = lib().cf_mr_lkey(self._h)
        self.rkey = lib().cf_mr_rkey(self._h)

    def close(self) -> None:
        if self._h:
            lib().cf_dereg_mr(self._h)
            self._h = None


class HostBuffer:
    """Page-aligned anonymous memory (hugepage-backed when possible)."""

    def __init__(self, length: int, try_hugepages: bool = True):
        huge = ctypes.c_int(0)
        self.length = int(length)
        self.addr = lib().cf_alloc(self.length, int(try_hugepages), ctypes.byref(huge))
        if not self.addr:
            raise RdmaError(f"allocation of {length} bytes failed: {last_error()}")
        self.hugepages = bool(huge.value)

    def as_numpy(self) -> np.ndarray:
        buf = (ctypes.c_uint8 * self.length).from_address(self.addr)
        return np.frombuffer(buf, dtype=np.uint8)

    def close(self) -> None:
        if self.addr:
            lib().cf_free(ctypes.c_void_p(self.addr), self.length)
            self.addr = 0


def make_ops(n: int) -> np.ndarray:
    return np.zeros(n, dtype=OP_DTYPE)


class Endpoint:
    """One RC connection. post() is non-blocking; wait()/test() track tickets."""

    def __init__(self, handle: int):
        self._h = handle

    @classmethod
    def connect(
        cls,
        dev: Device,
        server_ip: str,
        port: int,
        private_data: bytes = b"",
        timeout_ms: int = 5000,
    ) -> Endpoint:
        if len(private_data) > MAX_PRIVATE_DATA:
            raise ValueError("private data too long")
        h = lib().cf_connect(
            dev._h,
            server_ip.encode(),
            int(port),
            private_data,
            len(private_data),
            timeout_ms,
        )
        if not h:
            raise RdmaError(
                f"RDMA connect to {server_ip}:{port} failed: {last_error()}"
            )
        return cls(h)

    def private_data(self) -> bytes:
        buf = ctypes.create_string_buffer(MAX_PRIVATE_DATA)
        n = lib().cf_ep_private_data(self._h, buf, MAX_PRIVATE_DATA)
        return buf.raw[:n]

    def post(self, opcode: int, ops: np.ndarray) -> int:
        """Queue ops (an OP_DTYPE array). Returns the ticket for the batch."""
        if ops.dtype != OP_DTYPE:
            raise TypeError("ops must use OP_DTYPE")
        ops = np.ascontiguousarray(ops)
        ticket = ctypes.c_uint64(0)
        rc = lib().cf_post(
            self._h, opcode, ops.ctypes.data, len(ops), ctypes.byref(ticket)
        )
        if rc:
            raise RdmaError(f"post failed ({rc}): {last_error()}")
        return ticket.value

    def wait(self, ticket: int, timeout_ms: int = -1) -> None:
        rc = lib().cf_wait(self._h, ticket, timeout_ms)
        if rc:
            raise RdmaError(f"wait for ticket {ticket} failed ({rc}): {last_error()}")

    def test(self, ticket: int) -> bool:
        rc = lib().cf_test(self._h, ticket)
        if rc < 0:
            raise RdmaError(f"endpoint failed: {last_error()}")
        return rc == 1

    @property
    def error(self) -> int:
        return lib().cf_ep_error(self._h) if self._h else -1

    def stats(self) -> dict:
        s = _EpStats()
        lib().cf_ep_get_stats(self._h, ctypes.byref(s))
        return {k: getattr(s, k) for k, _ in _EpStats._fields_}

    def close(self) -> None:
        if self._h:
            lib().cf_ep_close(self._h)
            self._h = None


class Listener:
    def __init__(self, dev: Device, bind_ip: str, port: int, backlog: int = 64):
        self._h = lib().cf_listen(dev._h, bind_ip.encode(), int(port), backlog)
        if not self._h:
            raise RdmaError(f"RDMA listen on {bind_ip}:{port} failed: {last_error()}")

    def accept(self, timeout_ms: int = 1000) -> Endpoint | None:
        """Returns None on timeout, raises on other errors."""
        h = lib().cf_accept(self._h, timeout_ms)
        if h:
            return Endpoint(h)
        err = last_error()
        if err == "timeout":
            return None
        raise RdmaError(f"accept failed: {err}")

    def close(self) -> None:
        if self._h:
            lib().cf_listener_close(self._h)
            self._h = None
