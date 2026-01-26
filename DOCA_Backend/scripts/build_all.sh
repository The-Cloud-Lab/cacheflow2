#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Build script for DOCA Backend (host and DPU components)

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

echo_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

echo_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Detect architecture
ARCH=$(uname -m)
echo_info "Architecture: $ARCH"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo_info "Project root: $PROJECT_ROOT"

# Check DOCA installation
if [ ! -d "/opt/mellanox/doca" ]; then
    echo_error "DOCA not installed. Please install DOCA SDK first."
    echo_info "Visit: https://developer.nvidia.com/networking/doca"
    exit 1
fi

echo_success "DOCA installation found"

# Create build directory
BUILD_DIR="$PROJECT_ROOT/build"
mkdir -p "$BUILD_DIR"

cd "$BUILD_DIR"

# Configure with CMake
echo_info "Configuring with CMake..."
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_HOST=ON \
    -DBUILD_DPU=ON \
    -DBUILD_EXAMPLES=ON \
    -DDOCA_INSTALL_PATH=/opt/mellanox/doca

# Build
echo_info "Building..."
make -j$(nproc)

# Show build artifacts
echo ""
echo_success "Build complete!"
echo ""
echo "Build artifacts:"

if [ "$ARCH" = "x86_64" ]; then
    echo "  Host provider library: $BUILD_DIR/host/libhost_provider.a"
    if [ -f "$BUILD_DIR/host/host_provider_test" ]; then
        echo "  Host test: $BUILD_DIR/host/host_provider_test"
    fi
elif [ "$ARCH" = "aarch64" ]; then
    echo "  DPU offloader: $BUILD_DIR/dpu/dpu_offloader"
    if [ -f "$BUILD_DIR/dpu/dpu_standalone_test" ]; then
        echo "  DPU standalone test: $BUILD_DIR/dpu/dpu_standalone_test"
    fi
fi

echo ""
echo "Next steps:"
if [ "$ARCH" = "x86_64" ]; then
    echo "  1. Copy libhost_provider.a to host machine"
    echo "  2. Run: python -m vllm.v1.kv_offload.backends.doca_dma_client"
elif [ "$ARCH" = "aarch64" ]; then
    echo "  1. Install service: sudo ./scripts/install_service.sh"
    echo "  2. Start service: sudo ./scripts/start_dpu_service.sh start"
    echo "  3. Check status: sudo ./scripts/start_dpu_service.sh status"
fi

echo ""
