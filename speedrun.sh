#!/bin/bash

# --- Mamba-Integer V2 Portable Speedrun Script ---
# Designed to run on any Ubuntu/Debian VM with a GPU.

set -e 

# 1. Determine base directory
REPO_ROOT=$(pwd)
echo "Running from: $REPO_ROOT"

echo "--- Phase 0: Dependency Check ---"
# Check for NVCC (CUDA Compiler)
if ! command -v nvcc &> /dev/null; then
    echo "Error: nvcc not found. CUDA Toolkit is required."
    echo "Attempting to find it in common locations..."
    export PATH=$PATH:/usr/local/cuda/bin
    if ! command -v nvcc &> /dev/null; then
        echo "Still not found. Please install CUDA Toolkit."
        exit 1
    fi
fi
nvcc --version

# Check for build tools
if ! command -v make &> /dev/null; then
    echo "Error: make not found. Please install build-essential."
    exit 1
fi

echo "--- Phase 1: Environment Setup ---"
# Install uv if missing
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.cargo/bin:$PATH"

# Setup Virtual Env
[ -d ".venv" ] || uv venv
source .venv/bin/activate

# Install dependencies (portable)
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
uv pip install triton numpy sympy

echo "--- Phase 2: Path Configuration ---"
# Attempt to find bitnet-odp relative to this folder
# We check if it's a neighbor or inside. 
if [ -d "../bitnet-odp" ]; then
    BITNET_PATH="$(cd ../bitnet-odp/src && pwd)"
elif [ -d "./bitnet-odp" ]; then
    BITNET_PATH="$(cd ./bitnet-odp/src && pwd)"
else
    echo "Warning: bitnet-odp not found. Standard BitLinear may fail."
    BITNET_PATH=""
fi

# Ensure src is in PYTHONPATH
export PYTHONPATH="$PYTHONPATH:$REPO_ROOT/src:$BITNET_PATH"
echo "PYTHONPATH set to: $PYTHONPATH"

echo "--- Phase 3: Kernel Preparation ---"
# Build CUDA kernels
echo "Building CUDA kernels..."
if [ -d "$REPO_ROOT/src/cuda_kernels" ]; then
    cd "$REPO_ROOT/src/cuda_kernels"
    make clean
    make
    cd "$REPO_ROOT"
else
    echo "Error: src/cuda_kernels directory not found."
    exit 1
fi

echo "--- Phase 4: Data Check ---"
if [ ! -f "scripts/tinystories_train.bin" ]; then
    echo "Dataset not found at scripts/tinystories_train.bin"
    echo "Please ensure data is prepared before running."
    # exit 1 # Optional: auto-download logic could go here
fi

echo "--- Phase 5: Launch Training ---"
# Using -u for unbuffered logs (critical for nohup)
PYTHONUNBUFFERED=1 python3 -u scripts/train_mamba_integer.py