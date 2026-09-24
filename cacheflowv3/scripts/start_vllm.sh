#!/usr/bin/env bash
# Start `vllm serve` in the background and wait until it is ready.
#
#   start_vllm.sh <model> <logfile> [extra vllm serve args...]
#
# Writes the server PID to <logfile>.pid; stop it with `kill $(cat <logfile>.pid)`.
# Env: VLLM_BIN (default ~/vllm_env/bin/vllm), PORT (8100), GPU_UTIL (0.12),
#      MAX_LEN (8192), READY_TIMEOUT (600 s).
set -uo pipefail
MODEL=$1; LOG=$2; shift 2
VLLM_BIN="${VLLM_BIN:-$HOME/vllm_env/bin/vllm}"
PORT="${PORT:-8100}"
cd /tmp
nohup "$VLLM_BIN" serve "$MODEL" --port "$PORT" \
    --gpu-memory-utilization "${GPU_UTIL:-0.12}" --max-model-len "${MAX_LEN:-8192}" \
    "$@" > "$LOG" 2>&1 < /dev/null &
echo $! > "$LOG.pid"
for i in $(seq "${READY_TIMEOUT:-600}"); do
    if curl -sf "http://localhost:$PORT/health" > /dev/null; then echo "ready after ${i}s"; exit 0; fi
    if ! kill -0 "$(cat "$LOG.pid")" 2>/dev/null; then echo "vllm exited"; tail -30 "$LOG"; exit 1; fi
    sleep 1
done
echo "timeout"; tail -20 "$LOG"; exit 1
