#!/usr/bin/env bash
# Copy cacheflowv3 to the BlueField-3, build libcfrdma there, and (optionally)
# (re)start the server in a detached tmux-free background process.
#
#   ./deploy_bf3.sh                 # sync + build
#   ./deploy_bf3.sh start [args]    # sync + build + restart server (extra args go to the server)
#   ./deploy_bf3.sh stop            # stop the server
#   ./deploy_bf3.sh logs            # tail the server log
#
# Env: CF_BF3_SSH (ssh alias, default routenic-bf3), CF_BF3_DIR (default ~/cacheflowv3),
#      CF_BF3_IP (server bind IP, default 10.0.1.2)
#
# The server runs as root: registering a multi-GB pool with the NIC needs more
# locked memory than the BF3's default RLIMIT_MEMLOCK (~4 GB) for normal users.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"
SSH="${CF_BF3_SSH:-routenic-bf3}"
DIR="${CF_BF3_DIR:-cacheflowv3}"
IP="${CF_BF3_IP:-10.0.1.2,10.0.2.2}"  # one address per Spark link
LOG="/tmp/cacheflowv3-server.log"

sync_build() {
    rsync -a --delete --exclude 'lib/' --exclude '__pycache__' --exclude '*.egg-info' \
        "$HERE/" "$SSH:$DIR/"
    ssh "$SSH" "make -s -C $DIR/csrc"
}

stop() {
    ssh "$SSH" "sudo pkill -f '[c]acheflowv3.server' && sleep 1 || true"
}

case "${1:-sync}" in
    sync) sync_build ;;
    start)
        shift
        sync_build
        stop
        # root does not see the user's ~/.local site-packages (numpy), so pass them on
        SITE="$(ssh "$SSH" "python3 -c 'import site; print(site.getusersitepackages())'")"
        ssh -n "$SSH" "cd $DIR && setsid -f sudo PYTHONPATH=$SITE python3 -m cacheflowv3.server --bind $IP $* > $LOG 2>&1 < /dev/null"
        for _ in $(seq 60); do
            if ssh "$SSH" "grep -q 'CacheFlow v3 server' $LOG"; then
                ssh "$SSH" "tail -3 $LOG"; exit 0
            fi
            if ! ssh "$SSH" "pgrep -f '[c]acheflowv3.server' >/dev/null"; then
                ssh "$SSH" "tail -20 $LOG"; echo "server failed to start"; exit 1
            fi
            sleep 1
        done
        echo "server did not report ready in 60s"; exit 1 ;;
    stop) stop ;;
    logs) ssh "$SSH" "tail -n 50 -f $LOG" ;;
    *) echo "usage: $0 {sync|start [server args]|stop|logs}"; exit 1 ;;
esac
