# SPDX-License-Identifier: Apache-2.0
"""Raw RDMA bandwidth between this host and a running CacheFlow v3 server.

Borrows a scratch region of the server's pool (allocated and then aborted, so
nothing is cached) and times one-sided RDMA WRITE (host -> BF3) and READ
(BF3 -> host) at several message sizes. No torch; isolates the network path
from the KV connector.

    python -m cacheflowv3.tools.rdma_bw --host 10.0.2.2
"""

from __future__ import annotations

import argparse
import sys
import time
import uuid

import numpy as np

from .. import rdma
from ..config import CacheFlowConfig
from ..engine import DpuSession


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument("--host", default="10.0.1.2")
    p.add_argument(
        "--total-mb", type=int, default=1024, help="bytes moved per measurement"
    )
    p.add_argument("--sizes-kb", default="64,256,1024,4096,32768")
    p.add_argument("--repeats", type=int, default=3)
    a = p.parse_args(argv)

    total = a.total_mb << 20
    sess = DpuSession(CacheFlowConfig(server_host=a.host), "rdma_bw", want_rdma=True)
    key = f"rdma-bw-scratch-{uuid.uuid4().hex}"
    r = sess.ctl.request("alloc", keys=[key], rank=0, size=total, skip_existing=False)
    if r["status"][0] != "new":
        status = r["status"][0]
        print(f"server could not provide a {a.total_mb} MB scratch region ({status})")
        return 1
    raddr, rkey = r["addrs"][0], r["rkey"]
    buf = rdma.HostBuffer(total)
    mr = sess.dev.register(buf.addr, total)
    print(
        f"{sess.dev.name} -> {a.host} ({sess.info.get('device')}), "
        f"host buffer hugepage-backed={buf.hugepages or 'THP'}"
    )
    try:
        for kb in (int(s) for s in a.sizes_kb.split(",")):
            size = kb << 10
            n = total // size
            ops = rdma.make_ops(n)
            ops["laddr"] = buf.addr + np.arange(n, dtype=np.uint64) * size
            ops["raddr"] = raddr + np.arange(n, dtype=np.uint64) * size
            ops["lkey"], ops["rkey"], ops["len"] = mr.lkey, rkey, size
            res = {}
            for name, op in (("write", rdma.OP_WRITE), ("read", rdma.OP_READ)):
                best = 0.0
                for _ in range(a.repeats):
                    t = time.perf_counter()
                    sess.ep.wait(sess.ep.post(op, ops), 60_000)
                    best = max(best, n * size / (time.perf_counter() - t) / 1e9)
                res[name] = best
            print(
                f"  {kb:6d} KiB  WRITE (host->BF3) {res['write']:6.2f} GB/s   "
                f"READ (BF3->host) {res['read']:6.2f} GB/s",
                flush=True,
            )
    finally:
        sess.ctl.request("abort", keys=[key], rank=0)
        mr.close()
        buf.close()
        sess.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
