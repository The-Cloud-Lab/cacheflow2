#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Print a --kv-transfer-config JSON for this host.

Starts from configs/cacheflowv3.json, points server_host at the BF3 address on
this host's link (see scripts/setup_network.sh), and applies overrides:

    vllm serve ... --kv-transfer-config "$(python make_config.py)"
    python make_config.py --set transfer_mode=push --set layerwise_load=false
"""

from __future__ import annotations

import argparse
import json
import socket
from pathlib import Path

# hostname -> BF3 address on that host's direct link
BF3_ADDR = {"gx10-ee53": "10.0.1.2", "spark-e1d8": "10.0.2.2"}


def parse_value(v: str):
    try:
        return json.loads(v)
    except json.JSONDecodeError:
        return v


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument(
        "--base",
        default=str(
            Path(__file__).resolve().parent.parent / "configs" / "cacheflowv3.json"
        ),
    )
    p.add_argument("--server-host", default=BF3_ADDR.get(socket.gethostname()))
    p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    a = p.parse_args()
    cfg = json.loads(Path(a.base).read_text())
    extra = cfg["kv_connector_extra_config"]
    if a.server_host:
        extra["server_host"] = a.server_host
    for kv in a.set:
        k, v = kv.split("=", 1)
        extra[k] = parse_value(v)
    print(json.dumps(cfg))


if __name__ == "__main__":
    main()
