#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Quick start script for CacheFlow DOCA Backend

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}╔═══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║        CacheFlow DOCA Backend - Quick Start              ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════╝${NC}"
echo ""

# Detect architecture
ARCH=$(uname -m)
echo -e "${GREEN}[INFO]${NC} Detected architecture: $ARCH"

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if [ "$ARCH" = "aarch64" ]; then
    echo -e "${GREEN}[INFO]${NC} Running on DPU (ARM64)"
    echo ""
    echo "This script will:"
    echo "  1. Build DPU offloader"
    echo "  2. Run unit tests"
    echo "  3. Start DPU service"
    echo ""
    read -p "Continue? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 0
    fi

    # Build
    echo -e "${BLUE}[STEP 1/3]${NC} Building DPU components..."
    cd "$PROJECT_ROOT"
    ./scripts/build_all.sh

    # Test
    echo ""
    echo -e "${BLUE}[STEP 2/3]${NC} Running unit tests..."
    if [ -f "build/dpu/dpu_standalone_test" ]; then
        sudo ./build/dpu/dpu_standalone_test
    else
        echo -e "${YELLOW}[WARN]${NC} Test binary not found, skipping tests"
    fi

    # Start service
    echo ""
    echo -e "${BLUE}[STEP 3/3]${NC} Starting DPU service..."
    sudo ./scripts/start_dpu_service.sh start

    echo ""
    echo -e "${GREEN}✓ DPU setup complete!${NC}"
    echo ""
    echo "Check service status:"
    echo "  sudo systemctl status doca-kv-offloader"
    echo ""
    echo "View logs:"
    echo "  sudo journalctl -u doca-kv-offloader -f"

elif [ "$ARCH" = "x86_64" ]; then
    echo -e "${GREEN}[INFO]${NC} Running on Host (x86_64)"
    echo ""
    echo "This script will:"
    echo "  1. Build host provider library"
    echo "  2. Install CacheFlow Python package"
    echo "  3. Verify DPU connectivity"
    echo ""
    read -p "Continue? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 0
    fi

    # Build
    echo -e "${BLUE}[STEP 1/3]${NC} Building host components..."
    cd "$PROJECT_ROOT"
    ./scripts/build_all.sh

    # Install Python package
    echo ""
    echo -e "${BLUE}[STEP 2/3]${NC} Installing CacheFlow Python package..."
    cd "$PROJECT_ROOT/.."
    pip install -e .

    # Test connectivity
    echo ""
    echo -e "${BLUE}[STEP 3/3]${NC} Testing DPU connectivity..."
    echo "Enter DPU PCI address (e.g., 0000:03:00.0):"
    read -r DPU_PCI_ADDR

    if [ -z "$DPU_PCI_ADDR" ]; then
        echo -e "${YELLOW}[WARN]${NC} No PCI address provided, skipping connectivity test"
    else
        python3 << EOF
from vllm.v1.kv_offload.backends.doca_dma_client import DOCADMAClient

print("Testing connectivity to DPU at $DPU_PCI_ADDR...")

try:
    client = DOCADMAClient(dpu_pci_addr="$DPU_PCI_ADDR")
    if client.connect():
        print("✓ Connected to DPU successfully")

        stats = client.get_stats()
        if stats:
            print("✓ DPU statistics retrieved:")
            for key, value in stats.items():
                print(f"    {key}: {value}")

        client.disconnect()
        print("\n✓✓✓ All connectivity tests passed!")
    else:
        print("✗ Failed to connect to DPU")
        print("\nTroubleshooting:")
        print("  1. Verify DPU service is running:")
        print("     ssh ubuntu@DPU_IP 'systemctl status doca-kv-offloader'")
        print("  2. Check PCI address:")
        print("     lspci -D | grep Mellanox")
        print("  3. Verify DOCA installation:")
        print("     ls /opt/mellanox/doca")
except Exception as e:
    print(f"✗ Error: {e}")
    print("\nPlease check:")
    print("  - DOCA SDK is installed")
    print("  - DPU is accessible")
    print("  - PCI address is correct")
EOF
    fi

    echo ""
    echo -e "${GREEN}✓ Host setup complete!${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Configure vLLM to use DOCA backend"
    echo "  2. Run multi-turn benchmark:"
    echo "     cd benchmarks/multiturn"
    echo "     python benchmark_multiturn.py --backend doca"
    echo ""
    echo "See TESTING_GUIDE.md for detailed testing procedures"

else
    echo -e "${RED}[ERROR]${NC} Unsupported architecture: $ARCH"
    echo "This script supports x86_64 (host) and aarch64 (DPU) only"
    exit 1
fi

echo ""
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
