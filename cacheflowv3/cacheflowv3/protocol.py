# SPDX-License-Identifier: Apache-2.0
"""Control-plane protocol shared by the BlueField-3 server and its clients.

Framing: 4-byte big-endian length + UTF-8 JSON object. Every request carries an
integer "id"; the reply echoes it. The server may also send unsolicited
notifications (no "id", a "event" field) on the same connection, e.g. completion
of a server-driven ("push") transfer group.

Imported on the BF3, so no torch/vllm imports here.
"""

from __future__ import annotations

import contextlib
import json
import socket
import struct
import threading
from collections.abc import Callable
from typing import Any

DEFAULT_PORT = 18515  # TCP control plane
DEFAULT_RDMA_PORT = 18516  # rdma_cm data plane
PROTOCOL_VERSION = 1
MAX_FRAME = 64 << 20

_HDR = struct.Struct("!I")


class ProtocolError(RuntimeError):
    pass


class ServerError(RuntimeError):
    """The server answered a request with {"ok": false}."""


def encode(msg: dict) -> bytes:
    body = json.dumps(msg, separators=(",", ":")).encode()
    return _HDR.pack(len(body)) + body


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("control connection closed by peer")
        buf += chunk
    return bytes(buf)


def recv_msg(sock: socket.socket) -> dict:
    (n,) = _HDR.unpack(_recv_exact(sock, _HDR.size))
    if n > MAX_FRAME:
        raise ProtocolError(f"frame too large: {n}")
    return json.loads(_recv_exact(sock, n))


class ControlClient:
    """Thread-safe request/response client with a background reader.

    Many threads may call request() concurrently; replies are matched by id.
    Notifications are passed to `on_event` from the reader thread.
    """

    def __init__(
        self,
        host: str,
        port: int = DEFAULT_PORT,
        timeout: float = 10.0,
        on_event: Callable[[dict], None] | None = None,
        client_name: str = "",
    ):
        self.host, self.port, self.timeout = host, port, timeout
        self._on_event = on_event
        self._lock = threading.Lock()
        self._send_lock = threading.Lock()
        self._next_id = 1
        self._pending: dict[int, list] = {}  # id -> [event, reply]
        self._sock: socket.socket | None = None
        self._reader: threading.Thread | None = None
        self._closed = False
        self.session: dict = {}
        self._connect(client_name)

    def _connect(self, client_name: str) -> None:
        sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.settimeout(None)
        self._sock = sock
        self._reader = threading.Thread(
            target=self._read_loop, name="cacheflow-ctl-reader", daemon=True
        )
        self._reader.start()
        self.session = self.request(
            "hello", version=PROTOCOL_VERSION, client=client_name
        )

    @property
    def alive(self) -> bool:
        return self._sock is not None and not self._closed

    def _read_loop(self) -> None:
        sock = self._sock
        err: Exception | None = None
        try:
            while True:
                msg = recv_msg(sock)
                rid = msg.get("id")
                if rid is None:
                    if self._on_event is not None:
                        self._on_event(msg)
                    continue
                with self._lock:
                    slot = self._pending.get(rid)
                if slot is not None:
                    slot[1] = msg
                    slot[0].set()
        except Exception as e:  # connection closed or broken
            err = e
        finally:
            self._closed = True
            with self._lock:
                pending = list(self._pending.values())
            for slot in pending:
                if slot[1] is None:
                    slot[1] = {"ok": False, "error": f"connection lost: {err}"}
                slot[0].set()
            if self._on_event is not None:
                self._on_event({"event": "disconnected", "error": str(err)})

    def request(self, op: str, timeout: float | None = None, **fields: Any) -> dict:
        if self._closed:
            raise ConnectionError("control connection is closed")
        slot = [threading.Event(), None]
        with self._lock:
            rid = self._next_id
            self._next_id += 1
            self._pending[rid] = slot
        try:
            frame = encode({"id": rid, "op": op, **fields})
            with self._send_lock:
                self._sock.sendall(frame)
            if not slot[0].wait(self.timeout if timeout is None else timeout):
                raise TimeoutError(f"control request {op!r} timed out")
            reply = slot[1]
        finally:
            with self._lock:
                self._pending.pop(rid, None)
        if not reply.get("ok", False):
            raise ServerError(f"{op}: {reply.get('error', 'unknown error')}")
        return reply

    def close(self) -> None:
        self._closed = True
        if self._sock is not None:
            with contextlib.suppress(OSError):
                self._sock.shutdown(socket.SHUT_RDWR)
            self._sock.close()
