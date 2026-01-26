#!/bin/bash
# Setup script for DPU side
# Run this on the BlueField DPU

set -e

echo "========================================="
echo "DOCA KV Cache Offload - DPU Setup"
echo "========================================="

# Check if running on DPU (ARM)
ARCH=$(uname -m)
if [ "$ARCH" != "aarch64" ]; then
    echo "Error: This script must run on the DPU (ARM architecture)"
    echo "Current architecture: $ARCH"
    exit 1
fi

# Check if DOCA is installed
DOCA_PATH=${DOCA_INSTALL_PATH:-/opt/mellanox/doca}
if [ ! -d "$DOCA_PATH" ]; then
    echo "Error: DOCA not found at $DOCA_PATH"
    echo "Please install DOCA SDK first"
    exit 1
fi

echo "✓ Running on DPU (aarch64)"
echo "✓ DOCA found at $DOCA_PATH"

# Install dependencies
echo ""
echo "Installing dependencies..."
apt-get update
apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    libnuma-dev

echo "✓ Dependencies installed"

# Build DPU offloader
echo ""
echo "Building DPU offloader..."
cd "$(dirname "$0")/.."
make clean
make dpu

if [ $? -eq 0 ]; then
    echo "✓ Build successful"
else
    echo "✗ Build failed"
    exit 1
fi

# Create systemd service (optional)
echo ""
read -p "Install as systemd service? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    cat > /etc/systemd/system/doca-kv-offloader.service << EOF
[Unit]
Description=DOCA KV Cache Offloader Service
After=network.target

[Service]
Type=simple
ExecStart=$(pwd)/build/dpu/dpu_offloader --server 0.0.0.0:6789
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

    systemctl daemon-reload
    systemctl enable doca-kv-offloader.service
    echo "✓ Systemd service installed"
    echo ""
    echo "Start service with: sudo systemctl start doca-kv-offloader"
    echo "View logs with: sudo journalctl -u doca-kv-offloader -f"
fi

echo ""
echo "========================================="
echo "DPU setup complete!"
echo "========================================="
echo ""
echo "To start manually:"
echo "  sudo ./build/dpu/dpu_offloader --server 0.0.0.0:6789"
echo ""
