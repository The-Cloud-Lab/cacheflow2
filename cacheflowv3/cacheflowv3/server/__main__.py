# SPDX-License-Identifier: Apache-2.0
"""Run the CacheFlow v3 prefix-cache server on the BlueField-3.

python3 -m cacheflowv3.server --bind 10.0.1.2,10.0.2.2 --pool-gb 16
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal

from ..protocol import DEFAULT_PORT, DEFAULT_RDMA_PORT
from .server import CacheFlowServer, ServerConfig


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument(
        "--bind",
        required=True,
        help="comma-separated IPs of RDMA-capable netdevs (one per link, e.g. BF3 SFs)",
    )
    p.add_argument("--port", type=int, default=DEFAULT_PORT, help="TCP control port")
    p.add_argument(
        "--rdma-port", type=int, default=DEFAULT_RDMA_PORT, help="rdma_cm port"
    )
    p.add_argument("--pool-gb", type=float, default=16.0, help="registered pool size")
    p.add_argument(
        "--capacity-gb",
        type=float,
        default=None,
        help="usable capacity (<= pool); change at runtime with cfctl set-capacity",
    )
    p.add_argument("--page-kb", type=int, default=1024, help="allocation granularity")
    p.add_argument(
        "--no-hugepages", action="store_true", help="don't try 2MB hugepages"
    )
    p.add_argument("--lease-ms", type=int, default=60_000, help="default lookup lease")
    p.add_argument("--push-threads", type=int, default=4)
    p.add_argument("--log-level", default="INFO")
    a = p.parse_args()

    logging.basicConfig(
        level=a.log_level.upper(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cfg = ServerConfig(
        binds=[ip.strip() for ip in a.bind.split(",") if ip.strip()],
        port=a.port,
        rdma_port=a.rdma_port,
        pool_bytes=int(a.pool_gb * 2**30),
        page_bytes=a.page_kb * 1024,
        capacity_bytes=int(a.capacity_gb * 2**30) if a.capacity_gb else None,
        hugepages=not a.no_hugepages,
        default_lease_ms=a.lease_ms,
        push_threads=a.push_threads,
    )
    server = CacheFlowServer(cfg)
    loop = asyncio.new_event_loop()
    task = loop.create_task(server.serve())
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, task.cancel)
    try:
        loop.run_until_complete(task)
    except asyncio.CancelledError:
        pass
    finally:
        server.close()
        logging.getLogger("cacheflowv3.server").info("server stopped")


if __name__ == "__main__":
    main()
