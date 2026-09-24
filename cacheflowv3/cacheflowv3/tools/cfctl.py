# SPDX-License-Identifier: Apache-2.0
"""cfctl: talk to a running CacheFlow v3 server.

cfctl [--host 10.0.1.2] stats              # server + pool statistics (JSON)
cfctl flush                                # drop every unpinned entry (cold cache)
cfctl set-capacity 8                       # usable pool capacity in GiB
cfctl ping [-n 1000]                       # control-plane round-trip latency
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time

from ..protocol import DEFAULT_PORT, ControlClient


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="cfctl", description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument("--host", default="10.0.1.2")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("stats")
    sub.add_parser("flush")
    sc = sub.add_parser("set-capacity")
    sc.add_argument("gib", type=float)
    pg = sub.add_parser("ping")
    pg.add_argument("-n", type=int, default=1000)
    a = p.parse_args(argv)

    c = ControlClient(a.host, a.port, client_name="cfctl")
    try:
        if a.cmd == "stats":
            r = c.request("stats")
            r.pop("id", None)
            r.pop("ok", None)
            print(json.dumps(r, indent=2))
        elif a.cmd == "flush":
            print(f"dropped {c.request('flush')['dropped']} entries")
        elif a.cmd == "set-capacity":
            r = c.request("set_capacity", bytes=int(a.gib * 2**30))
            print(f"capacity = {r['capacity_bytes'] / 2**30:.2f} GiB")
        elif a.cmd == "ping":
            lat = []
            for _ in range(a.n):
                t = time.perf_counter()
                c.request("ping")
                lat.append((time.perf_counter() - t) * 1e6)
            lat.sort()
            print(
                f"control RTT over {a.n}: mean {statistics.mean(lat):.1f} us, "
                f"p50 {lat[len(lat) // 2]:.1f} us, "
                f"p99 {lat[int(len(lat) * 0.99)]:.1f} us"
            )
    finally:
        c.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
