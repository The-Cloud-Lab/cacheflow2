#!/usr/bin/env bash
# CacheFlow v3 network setup: direct QSFP links between DGX Sparks and one BlueField-3.
#
#   link "spark2":  spark2 enp1s0f0np0 10.0.1.1 ── 100G ── BF3 p1 ⇄ en3f1pf1sf0 ⇄ enp3s0f1s0 (mlx5_3) 10.0.1.2
#   link "spark1":  Spark1 enp1s0f1np1 10.0.2.1 ── 40G  ── BF3 p0 ⇄ en3f0pf0sf0 ⇄ enp3s0f0s0 (mlx5_2) 10.0.2.2
#
# The BF3 runs in DPU (switchdev) mode, so each uplink (p0/p1) is forwarded in
# the eswitch to an Arm-side scalable function (SF). RoCE cannot terminate on
# the uplink representor itself (rdma_bind_addr fails with ENODEV). The
# CacheFlow server listens on every SF address (one RDMA device per link).
#
# Changes are NOT persistent (they revert on reboot). Re-run after a reboot.
#
# Usage:
#   sudo ./setup_network.sh bf3 [link ...]           # on the BF3 Arm (default: all links)
#   sudo ./setup_network.sh bf3-teardown [link ...]  # undo on the BF3
#   sudo ./setup_network.sh spark [link]             # on a Spark (default: chosen by hostname)
#   ./setup_network.sh check [link]                  # on a Spark: ping + jumbo + RoCE MTU
#
# To add a link, add a case to link_vars below.
set -euo pipefail

MTU="${CF_MTU:-9000}"
ALL_LINKS="spark2 spark1"
BF3_BRIDGE="${CF_BF3_BRIDGE:-roce-br}"

link_vars() {
    case "$1" in
        spark2)
            SPARK_NETDEV=enp1s0f0np0 SPARK_IP=10.0.1.1/24
            BF3_UPLINK=p1 BF3_SF_REP=en3f1pf1sf0 BF3_SF_NETDEV=enp3s0f1s0 BF3_IP=10.0.1.2/24 ;;
        spark1)
            SPARK_NETDEV=enp1s0f1np1 SPARK_IP=10.0.2.1/24
            BF3_UPLINK=p0 BF3_SF_REP=en3f0pf0sf0 BF3_SF_NETDEV=enp3s0f0s0 BF3_IP=10.0.2.2/24 ;;
        *) echo "unknown link '$1' (known: $ALL_LINKS)"; exit 1 ;;
    esac
}

link_for_host() {
    case "$(hostname)" in
        gx10-ee53) echo spark2 ;;
        spark-e1d8) echo spark1 ;;
        *) echo "cannot pick a link for host $(hostname); pass one of: $ALL_LINKS" >&2; exit 1 ;;
    esac
}

has_ip() { ip -4 -o addr show dev "$1" | grep -q " ${2%/*}/"; }
rdma_dev() { rdma link show | awk -v n="$1" '$0 ~ "netdev "n"( |$)" {split($2,a,"/"); print a[1]}'; }

# Forward uplink <-> SF representor.
#   ovs : add both ports to an OVS bridge (needs ovs-vswitchd running)
#   tc  : two hardware-offloaded tc redirect rules; works when OVS is stopped or
#         masked (the default on this testbed, where OVS is deliberately masked)
BF3_FWD_MODE="${CF_BF3_FWD_MODE:-auto}"

forward_ovs() {
    ovs-vsctl --timeout=10 --may-exist add-br "$BF3_BRIDGE"
    for port in "$BF3_UPLINK" "$BF3_SF_REP"; do
        owner="$(ovs-vsctl port-to-br "$port" 2>/dev/null || true)"
        if [[ -n "$owner" && "$owner" != "$BF3_BRIDGE" ]]; then
            echo "moving $port from bridge $owner to $BF3_BRIDGE"
            ovs-vsctl --timeout=10 del-port "$owner" "$port"
        fi
        ovs-vsctl --timeout=10 --may-exist add-port "$BF3_BRIDGE" "$port"
    done
    ip link set dev "$BF3_BRIDGE" up
}

forward_tc() {
    for dev in "$BF3_UPLINK" "$BF3_SF_REP"; do
        tc qdisc replace dev "$dev" ingress
        tc filter del dev "$dev" ingress 2>/dev/null || true
    done
    tc filter add dev "$BF3_UPLINK" ingress protocol all prio 1 flower skip_sw \
        action mirred egress redirect dev "$BF3_SF_REP"
    tc filter add dev "$BF3_SF_REP" ingress protocol all prio 1 flower skip_sw \
        action mirred egress redirect dev "$BF3_UPLINK"
    tc filter show dev "$BF3_UPLINK" ingress | grep -q in_hw || {
        echo "tc rule was not offloaded to hardware"; exit 1; }
}

setup_bf3_link() {
    link_vars "$1"
    for dev in "$BF3_UPLINK" "$BF3_SF_REP" "$BF3_SF_NETDEV"; do
        ip link set dev "$dev" mtu "$MTU" up
    done
    local mode="$BF3_FWD_MODE"
    if [[ "$mode" == auto ]]; then
        pgrep -x ovs-vswitchd >/dev/null && mode=ovs || mode=tc
    fi
    "forward_$mode"
    # The IP must live on the SF netdev only, never on the uplink.
    ip addr del "$BF3_IP" dev "$BF3_UPLINK" 2>/dev/null || true
    has_ip "$BF3_SF_NETDEV" "$BF3_IP" || ip addr add "$BF3_IP" dev "$BF3_SF_NETDEV"
    echo "link $1: $BF3_UPLINK <-> $BF3_SF_REP via $mode; $BF3_SF_NETDEV = $BF3_IP (RDMA $(rdma_dev "$BF3_SF_NETDEV"))"
}

teardown_bf3_link() {
    link_vars "$1"
    for dev in "$BF3_UPLINK" "$BF3_SF_REP"; do
        tc qdisc del dev "$dev" ingress 2>/dev/null || true
    done
    ip addr del "$BF3_IP" dev "$BF3_SF_NETDEV" 2>/dev/null || true
    echo "link $1: removed tc forwarding and $BF3_IP"
}

setup_spark() {
    link_vars "$1"
    ip link set dev "$SPARK_NETDEV" mtu "$MTU" up
    has_ip "$SPARK_NETDEV" "$SPARK_IP" || ip addr add "$SPARK_IP" dev "$SPARK_NETDEV"
    echo "link $1: $SPARK_NETDEV = $SPARK_IP (RDMA $(rdma_dev "$SPARK_NETDEV"))"
}

check() {
    link_vars "$1"
    local peer="${BF3_IP%/*}"
    ping -c 3 -W 1 "$peer"
    ping -c 2 -W 1 -M do -s $((MTU - 28)) "$peer" && echo "jumbo frames OK"
    ibv_devinfo -d "$(rdma_dev "$SPARK_NETDEV")" | grep -E "active_mtu|state"
}

cmd="${1:-}"
shift || true
case "$cmd" in
    bf3) for l in ${*:-$ALL_LINKS}; do setup_bf3_link "$l"; done ;;
    bf3-teardown) for l in ${*:-$ALL_LINKS}; do teardown_bf3_link "$l"; done ;;
    spark) setup_spark "${1:-$(link_for_host)}" ;;
    check) check "${1:-$(link_for_host)}" ;;
    *) echo "usage: $0 {bf3|bf3-teardown|spark|check} [link ...]"; exit 1 ;;
esac
