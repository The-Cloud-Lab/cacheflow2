#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Host setup script for CacheFlow DOCA Backend

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}╔═══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║        CacheFlow DOCA Backend - Host Setup               ║${NC}"
echo -e "${BLUE}╚═══════════════════════════════════════════════════════════╝${NC}"
echo ""

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo -e "${GREEN}[INFO]${NC} Project root: $PROJECT_ROOT"

# Step 1: Build DOCA components (already done)
echo ""
echo -e "${GREEN}✓ DOCA host library already built${NC}"
echo "  Location: $PROJECT_ROOT/build/host/libhost_provider.a"

# Step 2: Set up Python environment
echo ""
echo -e "${BLUE}[STEP 1/2]${NC} Setting up Python environment..."

# Check CUDA
if [ -z "$CUDA_HOME" ]; then
    if [ -d "/usr/local/cuda" ]; then
        export CUDA_HOME=/usr/local/cuda
    elif [ -d "/usr/local/cuda-12" ]; then
        export CUDA_HOME=/usr/local/cuda-12
    fi
fi

if [ -n "$CUDA_HOME" ]; then
    echo -e "${GREEN}[INFO]${NC} CUDA_HOME: $CUDA_HOME"
    export PATH=$CUDA_HOME/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
else
    echo -e "${YELLOW}[WARN]${NC} CUDA_HOME not set. To install vLLM, please set:"
    echo "  export CUDA_HOME=/usr/local/cuda"
fi

# Step 3: Find DPU
echo ""
echo -e "${BLUE}[STEP 2/2]${NC} Finding DPU..."

DPU_PCI_ADDR=$(lspci -D | grep -i "mellanox\|bluefield" | head -n1 | awk '{print $1}')

if [ -n "$DPU_PCI_ADDR" ]; then
    echo -e "${GREEN}[INFO]${NC} Found DPU at: $DPU_PCI_ADDR"
else
    echo -e "${YELLOW}[WARN]${NC} No DPU found. Run: lspci -D | grep Mellanox"
fi

# Summary
echo ""
echo -e "${GREEN}✓ Host setup complete!${NC}"
echo ""
echo -e "${BLUE}Next Steps:${NC}"
echo ""
echo "1. Install vLLM (if not already installed):"
echo "   ${YELLOW}export CUDA_HOME=/usr/local/cuda${NC}"
echo "   cd $(dirname $PROJECT_ROOT)"
echo "   pip install -e ."
echo ""
echo "2. Test DOCA library directly (C test):"
echo "   ${YELLOW}sudo $PROJECT_ROOT/build/host/host_provider_test $DPU_PCI_ADDR${NC}"
echo "   (Requires DPU service running)"
echo ""
echo "3. Once vLLM is installed, test Python integration:"
echo '   python << EOF'
echo 'from vllm.v1.kv_offload.backends.doca_dma_client import DOCADMAClient'
echo 'client = DOCADMAClient(dpu_pci_addr="'$DPU_PCI_ADDR'")'
echo 'if client.connect():'
echo '    print("✓ Connected!")'
echo '    client.disconnect()'
echo 'EOF'
echo ""
echo "4. Run multi-turn benchmark:"
echo "   cd $(dirname $PROJECT_ROOT)/benchmarks/multiturn"
echo "   python benchmark_multiturn.py --backend doca"
echo ""
echo "Documentation:"
echo "  - $PROJECT_ROOT/TESTING_GUIDE.md"
echo "  - $PROJECT_ROOT/DEPLOYMENT.md"
echo "  - $(dirname $PROJECT_ROOT)/GETTING_STARTED.md"

