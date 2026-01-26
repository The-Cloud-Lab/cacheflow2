#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Startup script for DOCA DPU offloader service

set -e

# Configuration
DPU_SERVICE_NAME="doca_kv_cache"
DPU_OFFLOADER_BIN="/opt/cacheflow/bin/dpu_offloader"
LOG_DIR="/var/log/cacheflow"
LOG_FILE="${LOG_DIR}/dpu_offloader.log"
PID_FILE="/var/run/dpu_offloader.pid"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
    if [ -d "$LOG_DIR" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] [INFO] $1" >> "$LOG_FILE"
    fi
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
    if [ -d "$LOG_DIR" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] [WARN] $1" >> "$LOG_FILE"
    fi
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
    if [ -d "$LOG_DIR" ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] [ERROR] $1" >> "$LOG_FILE"
    fi
}

# Check if running on DPU (ARM64)
check_dpu() {
    if [ "$(uname -m)" != "aarch64" ]; then
        log_error "This script must run on the DPU (ARM64 architecture)"
        exit 1
    fi
    log_info "Running on DPU (ARM64)"
}

# Check DOCA installation
check_doca() {
    if [ ! -d "/opt/mellanox/doca" ]; then
        log_error "DOCA not installed. Please install DOCA SDK first"
        exit 1
    fi
    log_info "DOCA installation found"
}

# Create necessary directories
setup_directories() {
    mkdir -p "$LOG_DIR"
    chmod 755 "$LOG_DIR"
    log_info "Log directory: $LOG_DIR"
}

# Check if DPU offloader binary exists
check_binary() {
    if [ ! -f "$DPU_OFFLOADER_BIN" ]; then
        # Try build directory
        DPU_OFFLOADER_BIN="$(dirname "$0")/../build/dpu/dpu_offloader"
        if [ ! -f "$DPU_OFFLOADER_BIN" ]; then
            log_error "DPU offloader binary not found. Please build first:"
            echo "  cd DOCA_Backend && make dpu"
            exit 1
        fi
    fi
    log_info "DPU offloader binary: $DPU_OFFLOADER_BIN"
}

# Check if service is already running
check_running() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p "$PID" > /dev/null 2>&1; then
            log_warn "DPU offloader already running (PID: $PID)"
            return 0
        else
            log_warn "Stale PID file found, removing"
            rm -f "$PID_FILE"
        fi
    fi
    return 1
}

# Start DPU offloader service
start_service() {
    log_info "Starting DPU offloader service..."

    # Set library path for DOCA
    export LD_LIBRARY_PATH="/opt/mellanox/doca/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH"

    # Start service in background
    nohup "$DPU_OFFLOADER_BIN" \
        --server-name "$DPU_SERVICE_NAME" \
        >> "$LOG_FILE" 2>&1 &

    PID=$!
    echo "$PID" > "$PID_FILE"

    sleep 2

    if ps -p "$PID" > /dev/null 2>&1; then
        log_info "DPU offloader started successfully (PID: $PID)"
        log_info "Service name: $DPU_SERVICE_NAME"
        log_info "Log file: $LOG_FILE"
        return 0
    else
        log_error "Failed to start DPU offloader"
        rm -f "$PID_FILE"
        return 1
    fi
}

# Stop DPU offloader service
stop_service() {
    if [ ! -f "$PID_FILE" ]; then
        log_warn "DPU offloader not running (no PID file)"
        return 0
    fi

    PID=$(cat "$PID_FILE")
    log_info "Stopping DPU offloader (PID: $PID)..."

    if ps -p "$PID" > /dev/null 2>&1; then
        kill -TERM "$PID"

        # Wait for process to stop (max 10 seconds)
        for i in {1..10}; do
            if ! ps -p "$PID" > /dev/null 2>&1; then
                break
            fi
            sleep 1
        done

        # Force kill if still running
        if ps -p "$PID" > /dev/null 2>&1; then
            log_warn "Process did not stop gracefully, force killing..."
            kill -KILL "$PID"
        fi

        rm -f "$PID_FILE"
        log_info "DPU offloader stopped"
    else
        log_warn "Process not running, removing stale PID file"
        rm -f "$PID_FILE"
    fi
}

# Show service status
show_status() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p "$PID" > /dev/null 2>&1; then
            log_info "DPU offloader is RUNNING (PID: $PID)"
            echo "Service name: $DPU_SERVICE_NAME"
            echo "Log file: $LOG_FILE"

            # Show recent logs
            echo ""
            echo "Recent logs:"
            tail -n 20 "$LOG_FILE"
            return 0
        else
            log_warn "DPU offloader is NOT running (stale PID file)"
            return 1
        fi
    else
        log_warn "DPU offloader is NOT running"
        return 1
    fi
}

# Main script
main() {
    case "${1:-start}" in
        start)
            check_dpu
            check_doca
            setup_directories
            check_binary

            if check_running; then
                exit 0
            fi

            start_service
            ;;

        stop)
            stop_service
            ;;

        restart)
            stop_service
            sleep 2
            main start
            ;;

        status)
            show_status
            ;;

        *)
            echo "Usage: $0 {start|stop|restart|status}"
            echo ""
            echo "Commands:"
            echo "  start   - Start DPU offloader service"
            echo "  stop    - Stop DPU offloader service"
            echo "  restart - Restart DPU offloader service"
            echo "  status  - Show service status"
            exit 1
            ;;
    esac
}

main "$@"
