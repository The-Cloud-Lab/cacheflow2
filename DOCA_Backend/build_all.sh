#!/bin/bash
# Build script for DOCA Backend - Host and DPU components

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Building DOCA KV Cache Offload Backend${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Check if DOCA is installed
# Set PKG_CONFIG_PATH to include DOCA's pkgconfig directory
export PKG_CONFIG_PATH=/opt/mellanox/doca/lib/x86_64-linux-gnu/pkgconfig:$PKG_CONFIG_PATH

# Check for doca-common (base DOCA package)
if ! pkg-config --exists doca-common 2>/dev/null; then
    # Fallback: check if DOCA directory exists
    if [ ! -d "/opt/mellanox/doca" ]; then
        echo -e "${RED}ERROR: DOCA SDK not found${NC}"
        echo "Please install DOCA SDK from https://developer.nvidia.com/networking/doca"
        exit 1
    else
        echo -e "${YELLOW}⚠${NC} pkg-config can't find DOCA, but DOCA directory exists"
        echo -e "${YELLOW}⚠${NC} Will proceed with build (may need manual library paths)"
        DOCA_VERSION="installed (version unknown)"
    fi
else
    DOCA_VERSION=$(pkg-config --modversion doca-common 2>/dev/null || echo "unknown")
fi
echo -e "${GREEN}✓${NC} Found DOCA SDK version: $DOCA_VERSION"

# Determine what to build
BUILD_HOST=1
BUILD_DPU=0

# Check if we're on ARM (likely DPU)
ARCH=$(uname -m)
if [ "$ARCH" = "aarch64" ]; then
    echo -e "${YELLOW}Detected ARM architecture - building DPU components${NC}"
    BUILD_HOST=0
    BUILD_DPU=1
else
    echo -e "${YELLOW}Detected x86_64 architecture - building host components${NC}"
fi

# Allow override via arguments
if [ "$1" = "host" ]; then
    BUILD_HOST=1
    BUILD_DPU=0
    echo -e "${YELLOW}Building host components only${NC}"
elif [ "$1" = "dpu" ]; then
    BUILD_HOST=0
    BUILD_DPU=1
    echo -e "${YELLOW}Building DPU components only${NC}"
elif [ "$1" = "all" ]; then
    BUILD_HOST=1
    BUILD_DPU=1
    echo -e "${YELLOW}Building both host and DPU components${NC}"
fi

# Create build directory
BUILD_DIR="$SCRIPT_DIR/build"
mkdir -p "$BUILD_DIR"

# Build host components
if [ $BUILD_HOST -eq 1 ]; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Building Host Provider${NC}"
    echo -e "${GREEN}========================================${NC}"
    
    cd "$BUILD_DIR"
    
    # Configure with CMake
    echo "Configuring CMake..."
    cmake .. \
        -DBUILD_HOST=ON \
        -DBUILD_DPU=OFF \
        -DBUILD_SHARED_LIBS=ON \
        -DCMAKE_BUILD_TYPE=Release
    
    # Build
    echo "Building..."
    make -j$(nproc)
    
    # Check if library was built
    if [ -f "libhost_provider.so" ]; then
        echo -e "${GREEN}✓${NC} Host provider library built successfully"
        ls -lh libhost_provider.so
    else
        echo -e "${RED}✗${NC} Failed to build host provider library"
        exit 1
    fi
    
    # Check if test programs were built
    if [ -f "host_provider_test" ]; then
        echo -e "${GREEN}✓${NC} Host test program built"
    fi
fi

# Build DPU components
if [ $BUILD_DPU -eq 1 ]; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Building DPU Offloader${NC}"
    echo -e "${GREEN}========================================${NC}"
    
    DPU_BUILD_DIR="$SCRIPT_DIR/build_dpu"
    mkdir -p "$DPU_BUILD_DIR"
    cd "$DPU_BUILD_DIR"
    
    # Configure with CMake
    echo "Configuring CMake for DPU..."
    cmake .. \
        -DBUILD_HOST=OFF \
        -DBUILD_DPU=ON \
        -DCMAKE_BUILD_TYPE=Release
    
    # Build
    echo "Building..."
    make -j$(nproc)
    
    # Check if binary was built
    if [ -f "dpu_offloader" ]; then
        echo -e "${GREEN}✓${NC} DPU offloader built successfully"
        ls -lh dpu_offloader
    else
        echo -e "${RED}✗${NC} Failed to build DPU offloader"
        exit 1
    fi
fi

# Install Python package
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Installing Python Package${NC}"
echo -e "${GREEN}========================================${NC}"

cd "$SCRIPT_DIR/python"

# Check if we're in a virtual environment
if [ -n "$VIRTUAL_ENV" ]; then
    echo "Detected virtual environment: $VIRTUAL_ENV"
    echo "Installing Python package in development mode..."
    pip install -e .
    echo -e "${GREEN}✓${NC} Python package installed in virtual environment"
elif command -v pip &> /dev/null; then
    echo "Installing Python package in development mode..."
    if pip install -e . 2>&1 | grep -q "externally-managed-environment"; then
        echo -e "${YELLOW}⚠${NC} System pip is externally managed"
        echo -e "${YELLOW}⚠${NC} Please activate a virtual environment and run: pip install -e $SCRIPT_DIR/python"
    else
        echo -e "${GREEN}✓${NC} Python package installed"
    fi
else
    echo -e "${YELLOW}⚠${NC} pip not found, skipping Python package installation"
    echo -e "${YELLOW}⚠${NC} Please install manually: pip install -e $SCRIPT_DIR/python"
fi

# Summary
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Build Summary${NC}"
echo -e "${GREEN}========================================${NC}"

if [ $BUILD_HOST -eq 1 ]; then
    echo -e "${GREEN}✓${NC} Host components built in: $BUILD_DIR"
    echo "   - libhost_provider.so"
    echo "   - host_provider_test"
fi

if [ $BUILD_DPU -eq 1 ]; then
    echo -e "${GREEN}✓${NC} DPU components built in: $DPU_BUILD_DIR"
    echo "   - dpu_offloader"
fi

echo ""
echo -e "${GREEN}Next steps:${NC}"

if [ $BUILD_DPU -eq 1 ]; then
    echo "1. Start the DPU service:"
    echo "   sudo $DPU_BUILD_DIR/dpu_offloader --server 0.0.0.0:6789"
fi

if [ $BUILD_HOST -eq 1 ]; then
    echo "2. Set environment variables:"
    echo "   export LD_LIBRARY_PATH=$BUILD_DIR:\$LD_LIBRARY_PATH"
    echo "   export PYTHONPATH=$SCRIPT_DIR/python:\$PYTHONPATH"
    echo ""
    echo "3. Test the installation:"
    echo "   python3 $SCRIPT_DIR/test_integration.py"
    echo ""
    echo "4. Run vLLM example:"
    echo "   python3 $SCRIPT_DIR/python/example_vllm_integration.py"
fi

echo ""
echo -e "${GREEN}Build completed successfully!${NC}"
