# SPDX-License-Identifier: Apache-2.0
"""CacheFlow v3 prefix-cache server (runs on the BlueField-3 Arm cores).

  * Control plane: asyncio TCP server speaking cacheflowv3.protocol. The prefix
    index, allocator, LRU and leases live here (see store.py), so lookups by
    the vLLM scheduler are answered by the SmartNIC.
  * Data plane: one memory pool (the prefix cache itself), registered on every
    RDMA device the server binds to, and one rdma_cm listener per device.
    Clients read/write entries with one-sided RDMA ("pull"), or ask the server
    to drive the transfer ("push"), in which case the BF3 posts RDMA WRITE/READ
    against the client's registered staging memory.
  * Several GPU hosts can attach through different ports (one per link); they
    all share the same index and pool, so a prefix saved by one host is a hit
    for every other host serving the same model.

Only numpy + libcfrdma are needed; no torch, no vLLM.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from .. import __version__, rdma
from ..layout import ChunkGeometry, plan_ops
from ..protocol import MAX_FRAME, PROTOCOL_VERSION, encode
from .store import Store

log = logging.getLogger("cacheflowv3.server")


@dataclass
class ServerConfig:
    binds: list[str]  # one IP per RDMA-capable netdev (one per link)
    port: int
    rdma_port: int
    pool_bytes: int
    page_bytes: int
    capacity_bytes: int | None = None
    hugepages: bool = True
    default_lease_ms: int = 60_000
    push_threads: int = 4


@dataclass
class Port:
    """One RDMA device the pool is exposed on (one per link to a GPU host)."""

    ip: str
    dev: rdma.Device
    mr: rdma.MemoryRegion
    listener: rdma.Listener


@dataclass
class Session:
    token: str
    client: str
    peer: str
    writer: asyncio.StreamWriter
    port: Port | None  # the link the client came in on (None: control only)
    started: float = field(default_factory=time.monotonic)
    requests: int = 0


class CacheFlowServer:
    def __init__(self, cfg: ServerConfig):
        self.cfg = cfg
        t0 = time.monotonic()
        self.pool = rdma.HostBuffer(cfg.pool_bytes, try_hugepages=cfg.hugepages)
        self.ports: list[Port] = []
        for ip in cfg.binds:
            dev = rdma.Device(ip)
            mr = dev.register(self.pool.addr, self.pool.length)
            self.ports.append(Port(ip, dev, mr, rdma.Listener(dev, ip, cfg.rdma_port)))
            log.info("port %s: device %s, rkey %d", ip, dev.name, mr.rkey)
        log.info(
            "pool: %.1f GiB at 0x%x (hugepages=%s) on %d port(s), ready in %.1fs",
            cfg.pool_bytes / 2**30,
            self.pool.addr,
            self.pool.hugepages,
            len(self.ports),
            time.monotonic() - t0,
        )
        self._port_by_ip = {p.ip: p for p in self.ports}
        self.store = Store(cfg.pool_bytes, cfg.page_bytes, cfg.capacity_bytes)
        self.sessions: dict[str, Session] = {}
        self._eps: dict[str, rdma.Endpoint] = {}
        self._eps_lock = threading.Lock()
        self._stop = threading.Event()
        self._push_pool = ThreadPoolExecutor(
            cfg.push_threads, thread_name_prefix="cf-push"
        )
        self._loop: asyncio.AbstractEventLoop | None = None
        self.started = time.monotonic()
        self.push_stats = {"transfers": 0, "bytes": 0, "errors": 0}

    # ------------------------------------------------------------ RDMA side
    def _accept_loop(self, port: Port) -> None:
        while not self._stop.is_set():
            try:
                ep = port.listener.accept(timeout_ms=500)
            except rdma.RdmaError as e:
                log.warning("RDMA accept error on %s: %s", port.ip, e)
                continue
            if ep is None:
                continue
            token = ep.private_data().rstrip(b"\0").decode(errors="replace")
            sess = self.sessions.get(token)
            if sess is None:
                log.warning("RDMA connection with unknown session %r rejected", token)
                ep.close()
                continue
            if sess.port is not port:
                # Memory keys are per device: data must use the control link's port.
                log.warning(
                    "session %s: RDMA via %s but control via %s; rejected",
                    token,
                    port.ip,
                    sess.port.ip if sess.port else "none",
                )
                ep.close()
                continue
            with self._eps_lock:
                old = self._eps.pop(token, None)
                self._eps[token] = ep
            if old is not None:
                old.close()
            log.info(
                "RDMA endpoint up for session %s (%s)",
                token,
                self.sessions[token].client,
            )

    def _ep(self, token: str) -> rdma.Endpoint | None:
        with self._eps_lock:
            return self._eps.get(token)

    def _drop_ep(self, token: str) -> None:
        with self._eps_lock:
            ep = self._eps.pop(token, None)
        if ep is not None:
            ep.close()

    def _in_pool(self, addr: int, size: int) -> bool:
        return (
            self.pool.addr <= addr and addr + size <= self.pool.addr + self.pool.length
        )

    # ---------------------------------------------------------- control side
    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        peer = writer.get_extra_info("peername")
        sess: Session | None = None
        try:
            while True:
                hdr = await reader.readexactly(4)
                n = int.from_bytes(hdr, "big")
                if n > MAX_FRAME:
                    raise ValueError(f"frame too large ({n})")
                msg = json.loads(await reader.readexactly(n))
                rid, op = msg.get("id"), msg.get("op")
                if op == "hello":
                    local_ip = writer.get_extra_info("sockname")[0]
                    sess = Session(
                        os.urandom(8).hex(),
                        msg.get("client", ""),
                        str(peer),
                        writer,
                        self._port_by_ip.get(local_ip),
                    )
                    self.sessions[sess.token] = sess
                    reply = self._hello(sess, msg)
                    log.info(
                        "session %s opened by %s (%s)", sess.token, sess.client, peer
                    )
                elif sess is None:
                    reply = {"ok": False, "error": "send hello first"}
                else:
                    sess.requests += 1
                    try:
                        reply = self._dispatch(sess, op, msg)
                    except Exception as e:  # report, keep the session alive
                        log.exception("request %s failed", op)
                        reply = {"ok": False, "error": f"{type(e).__name__}: {e}"}
                reply["id"] = rid
                writer.write(encode(reply))
                await writer.drain()
        except (asyncio.IncompleteReadError, ConnectionError):
            pass
        except Exception:
            log.exception("control connection from %s failed", peer)
        finally:
            if sess is not None:
                self.sessions.pop(sess.token, None)
                self._drop_ep(sess.token)
                aborted = self.store.drop_session(sess.token)
                log.info(
                    "session %s closed (aborted %d in-flight writes)",
                    sess.token,
                    aborted,
                )
            writer.close()

    def _hello(self, sess: Session, msg: dict) -> dict:
        if msg.get("version") != PROTOCOL_VERSION:
            return {
                "ok": False,
                "error": f"protocol version mismatch: server {PROTOCOL_VERSION}",
            }
        return {
            "ok": True,
            "session": sess.token,
            "rdma_port": self.cfg.rdma_port,
            "rkey": self._rkey(sess),
            "pool_addr": self.pool.addr,
            "pool_bytes": self.store.pool_bytes,
            "capacity_bytes": self.store.capacity_bytes,
            "page_bytes": self.cfg.page_bytes,
            "server_version": __version__,
            "device": sess.port.dev.name if sess.port else None,
            "ports": [p.ip for p in self.ports],
        }

    @staticmethod
    def _rkey(sess: Session) -> int | None:
        return sess.port.mr.rkey if sess.port else None

    def _dispatch(self, sess: Session, op: str, m: dict) -> dict:
        st = self.store
        base = self.pool.addr
        if op == "ping":
            return {"ok": True, "t": time.time()}
        if op == "lookup":
            ranks = int(m.get("ranks", 1))
            hits, lease, found = st.lookup(
                m["keys"],
                ranks,
                bool(m.get("pin", False)),
                int(m.get("lease_ms", self.cfg.default_lease_ms)),
                sess.token,
            )
            return {
                "ok": True,
                "hits": hits,
                "lease": lease.lease_id if lease else None,
                "entries": [[base + e.offset for e in row] for row in found],
                "rkey": self._rkey(sess),
            }
        if op == "release":
            return {"ok": True, "released": st.release(m["lease"])}
        if op == "alloc":
            size = int(m["size"])
            keys = [f"{k}:{int(m.get('rank', 0))}" for k in m["keys"]]
            res = st.allocate_for_write(
                keys, size, sess.token, bool(m.get("skip_existing", True))
            )
            return {
                "ok": True,
                "status": [s for s, _ in res],
                "addrs": [base + e.offset if e is not None else 0 for _, e in res],
                "rkey": self._rkey(sess),
            }
        if op in ("commit", "abort"):
            keys = [f"{k}:{int(m.get('rank', 0))}" for k in m["keys"]]
            fn = st.commit if op == "commit" else st.abort
            return {"ok": True, "n": fn(keys, sess.token)}
        if op == "push":
            return self._start_push(sess, m)
        if op == "flush":
            return {"ok": True, "dropped": st.flush()}
        if op == "set_capacity":
            return {"ok": True, "capacity_bytes": st.set_capacity(int(m["bytes"]))}
        if op == "stats":
            return {"ok": True, "store": st.snapshot(), "server": self._server_stats()}
        return {"ok": False, "error": f"unknown op {op!r}"}

    def _server_stats(self) -> dict:
        with self._eps_lock:
            eps = {t: ep.stats() for t, ep in self._eps.items()}
        return {
            "uptime_s": time.monotonic() - self.started,
            "sessions": {t: s.client for t, s in self.sessions.items()},
            "rdma_endpoints": eps,
            "push": dict(self.push_stats),
            "ports": {p.ip: p.dev.name for p in self.ports},
            "session_ports": {
                t: s.port.ip if s.port else None for t, s in self.sessions.items()
            },
            "version": __version__,
        }

    # ----------------------------------------------------------- push mode
    def _start_push(self, sess: Session, m: dict) -> dict:
        ep = self._ep(sess.token)
        if ep is None or sess.port is None:
            return {"ok": False, "error": "no RDMA endpoint for this session"}
        geom = ChunkGeometry(int(m["num_layers"]), int(m["layer_bytes"]))
        entries = [int(a) for a in m["entries"]]
        for a in entries:
            if not self._in_pool(a, geom.entry_bytes):
                return {"ok": False, "error": f"entry 0x{a:x} outside the pool"}
        opcode = rdma.OP_WRITE if m["dir"] == "load" else rdma.OP_READ
        layers = m.get("layers")
        groups = plan_ops(
            geom,
            int(m["staging_base"]),
            int(m["staging_rkey"]),
            entries,
            sess.port.mr.lkey,
            layer_groups=bool(m.get("layer_groups", False)),
            staging_is_local=False,
            staging_chunks=m.get("staging_chunks"),
            n_staging=m.get("n_staging"),
            layers=tuple(layers) if layers else None,
        )
        xfer = m["xfer"]
        writer, loop = sess.writer, self._loop
        nbytes = geom.entry_bytes * len(entries)

        def notify(msg: dict) -> None:
            loop.call_soon_threadsafe(writer.write, encode(msg))

        def run() -> None:
            try:
                tickets = [ep.post(opcode, g) for g in groups]
                for gi, t in enumerate(tickets):
                    ep.wait(t, timeout_ms=30_000)
                    notify(
                        {
                            "event": "push",
                            "xfer": xfer,
                            "group": gi,
                            "last": gi == len(tickets) - 1,
                            "ok": True,
                        }
                    )
                self.push_stats["transfers"] += 1
                self.push_stats["bytes"] += nbytes
            except Exception as e:
                self.push_stats["errors"] += 1
                log.warning("push %s failed: %s", xfer, e)
                notify(
                    {
                        "event": "push",
                        "xfer": xfer,
                        "group": -1,
                        "last": True,
                        "ok": False,
                        "error": str(e),
                    }
                )

        self._push_pool.submit(run)
        return {"ok": True, "groups": len(groups)}

    # ------------------------------------------------------------ lifecycle
    async def _housekeeping(self) -> None:
        while True:
            await asyncio.sleep(1.0)
            self.store.expire_leases()
            with self._eps_lock:
                dead = [t for t, ep in self._eps.items() if ep.error]
            for t in dead:
                log.info(
                    "RDMA endpoint for session %s failed (error %s); dropping",
                    t,
                    self._ep(t).error if self._ep(t) else "?",
                )
                self._drop_ep(t)

    async def serve(self) -> None:
        self._loop = asyncio.get_running_loop()
        for p in self.ports:
            threading.Thread(
                target=self._accept_loop,
                args=(p,),
                name=f"cf-accept-{p.ip}",
                daemon=True,
            ).start()
        server = await asyncio.start_server(self._handle, self.cfg.binds, self.cfg.port)
        asyncio.create_task(self._housekeeping())
        log.info(
            "CacheFlow v3 server %s: control tcp port %d, rdma port %d on %s, "
            "capacity %.1f GiB",
            __version__,
            self.cfg.port,
            self.cfg.rdma_port,
            ", ".join(f"{p.ip} ({p.dev.name})" for p in self.ports),
            self.store.capacity_bytes / 2**30,
        )
        async with server:
            await server.serve_forever()

    def close(self) -> None:
        self._stop.set()
        self._push_pool.shutdown(wait=False, cancel_futures=True)
        with self._eps_lock:
            eps, self._eps = list(self._eps.values()), {}
        for ep in eps:
            ep.close()
        for p in self.ports:
            p.listener.close()
            p.mr.close()
        self.pool.close()
        for p in self.ports:
            p.dev.close()
