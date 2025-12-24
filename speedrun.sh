#!/bin/bash

# --- Mamba-Integer V2 Speedrun Script ---
# Optimized for A100/H100 GPUs with Triton parallel scan kernels

set -e

# 1. Determine base directory
REPO_ROOT=$(pwd)
echo "=== Mamba-Integer Training ==="
echo "Running from: $REPO_ROOT"

echo "--- Phase 0: Dependency Check ---"
# Check for NVCC (CUDA Compiler) - optional now since we use Triton
if ! command -v nvcc &> /dev/null; then
    echo "Warning: nvcc not found. Checking common locations..."
    export PATH=$PATH:/usr/local/cuda/bin
    if command -v nvcc &> /dev/null; then
        nvcc --version
    else
        echo "NVCC not found - CUDA kernels disabled, using Triton only (recommended)"
    fi
else
    nvcc --version
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
# src/rational_bitnet.py is self-contained, no external dependency needed
export PYTHONPATH="$PYTHONPATH:$REPO_ROOT/src"
echo "PYTHONPATH set to: $PYTHONPATH"

echo "--- Phase 3: Kernel Preparation ---"
# CUDA kernels are optional - Triton kernels are preferred and faster
if [ -d "$REPO_ROOT/src/cuda_kernels" ] && command -v nvcc &> /dev/null; then
    echo "Building optional CUDA kernels..."
    cd "$REPO_ROOT/src/cuda_kernels"
    make clean 2>/dev/null || true
    make 2>/dev/null || echo "CUDA kernel build failed - using Triton kernels instead"
    cd "$REPO_ROOT"
else
    echo "Using Triton kernels (parallel scan enabled)"
fi

echo "--- Phase 4: Data Check ---"
if [ ! -f "scripts/tinystories_train.bin" ]; then
    echo "Dataset not found at scripts/tinystories_train.bin"
    echo "Please ensure data is prepared before running."
fi

echo "--- Phase 5: Performance Settings ---"
# Optimal settings for A100/H100
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR="$REPO_ROOT/.triton_cache"

echo "--- Phase 6: Launch Training ---"
echo "Starting training with optimizations:"
echo "  - Parallel prefix scan (Triton)"
echo "  - BitLinear quantization caching"
echo "  - torch.compile enabled"
echo ""

# Using -u for unbuffered logs
PYTHONUNBUFFERED=1 python3 -u scripts/train_mamba_integer.py 2>&1 | tee training_speedrun.log