#!/bin/bash
# Qwen Training Engine - GCP VM Setup Script
# Usage: ssh into your VM, then run: bash setup_vm.sh
#
# Prerequisites: GCP VM with Ubuntu 22.04+, firewall rule allowing TCP port 8000

set -e

echo "=== Qwen Training Engine Setup ==="

# Install Python if needed
if ! command -v python3 &>/dev/null; then
    echo "[1/5] Installing Python..."
    sudo apt update && sudo apt install -y python3 python3-pip python3-venv
else
    echo "[1/5] Python already installed: $(python3 --version)"
fi

# Setup project
echo "[2/5] Setting up project..."
PROJECT_DIR=~/qwen-training
mkdir -p "$PROJECT_DIR"

if [ ! -f "$PROJECT_DIR/server/main.py" ]; then
    echo ""
    echo "Server files not found. Cloning repo..."
    cd "$PROJECT_DIR"
    git clone https://github.com/kmalarifi97/qwen-training.git .
fi

# Create venv and install deps
echo "[3/5] Installing dependencies..."
cd "$PROJECT_DIR"
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Create data directories
mkdir -p data/uploads data/datasets data/adapters data/db

# Create systemd service
echo "[4/5] Creating systemd service..."
sudo tee /etc/systemd/system/qwen-training.service > /dev/null <<EOF
[Unit]
Description=Qwen Training Engine
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$PROJECT_DIR
ExecStart=$PROJECT_DIR/venv/bin/uvicorn server.main:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=5
Environment=GEMINI_API_KEY=${GEMINI_API_KEY:-}
Environment=DATA_DIR=/home/$USER/qwen-training/data

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable qwen-training
sudo systemctl start qwen-training

echo "[5/5] Server started!"
echo ""
echo "=== Setup Complete ==="
echo ""
echo "Server running on port 8000"
echo "Check status:  sudo systemctl status qwen-training"
echo "View logs:     sudo journalctl -u qwen-training -f"
echo "Restart:       sudo systemctl restart qwen-training"
echo ""
echo "Open: http://$(curl -s ifconfig.me 2>/dev/null || echo 'YOUR_VM_IP'):8000"
