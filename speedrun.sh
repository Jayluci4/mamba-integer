#!/bin/bash

# --- Mamba-Integer V2 Speedrun Script ---
# Optimized for A100/H100 GPUs with Triton parallel scan kernels

set -e

# 1. Determine base directory
REPO_ROOT=$(pwd)
echo "=== Mamba-Integer Training ==="
echo "Running from: $REPO_ROOT"

echo "--- Phase 0: Dependency Check ---"
# Check for NVCC (CUDA Compiler)
if ! command -v nvcc &> /dev/null; then
    echo "Error: nvcc not found. Attempting auto-installation of CUDA Toolkit..."
    # Attempt to add NVIDIA repo and install
    if command -v sudo &> /dev/null; then
        sudo apt-get update
        sudo apt-get install -y cuda-toolkit-12-1 || sudo apt-get install -y nvidia-cuda-toolkit
        export PATH=$PATH:/usr/local/cuda/bin
    else
        echo "Sudo not found. Please install CUDA toolkit manually: apt-get install nvidia-cuda-toolkit"
        exit 1
    fi
fi

# Final check for nvcc
if ! command -v nvcc &> /dev/null; then
    # Try common location
    if [ -f "/usr/local/cuda/bin/nvcc" ]; then
        export PATH=$PATH:/usr/local/cuda/bin
    else
        echo "CUDA installation failed or nvcc not in path. Please fix manually."
        exit 1
    fi
fi
nvcc --version

echo "--- Phase 1: Environment Setup ---"
# Install uv if missing
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.cargo/bin:$PATH"

# Setup Virtual Env
[ -d ".venv" ] || uv venv
source .venv/bin/activate

# Install dependencies (portable)
# setuptools and wheel are required for some legacy packages and CUDA extensions
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
uv pip install triton numpy sympy setuptools wheel

echo "--- Phase 2: Path Configuration ---"
# Hardcoded relative path for current environment
if [ -d "../gemma-intelligent/conv/src/bitnet-odp" ]; then
    BITNET_PATH="$(cd ../gemma-intelligent/conv/src/bitnet-odp/src && pwd)"
elif [ -d "../bitnet-odp" ]; then
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

echo "--- Phase 3: Build & Data Preparation ---"
# 1. Build Rust Tokenizer if needed
if [ ! -f "src/cuda_kernels/librustbpe.so" ]; then
    echo "Building Rust Tokenizer..."
    if [ -d "src/rust_tokenizer" ]; then
        cd src/rust_tokenizer
        cargo build --release
        cp target/release/librustbpe.so ../cuda_kernels/
        cd ../..
    else
        echo "Error: src/rust_tokenizer not found. Tokenizer build skipped."
    fi
fi

# 2. Build CUDA kernels
echo "Building CUDA kernels..."
if [ -d "$REPO_ROOT/src/cuda_kernels" ]; then
    cd "$REPO_ROOT/src/cuda_kernels"
    make clean
    make
    cd "$REPO_ROOT"
fi

# 3. Check for dataset (.bin)
if [ ! -f "scripts/tinystories_train.bin" ]; then
    echo "Dataset .bin not found. Running offline tokenization..."
    # We assume raw data exists or is handled by the script
    if [ -f "scripts/tokenize_dataset_offline.py" ]; then
        python3 scripts/tokenize_dataset_offline.py
    else
        echo "Warning: scripts/tokenize_dataset_offline.py not found. Cannot build .bin."
    fi
fi

echo "--- Phase 4: Launch Training ---"

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