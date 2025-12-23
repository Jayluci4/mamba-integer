#!/bin/bash

# --- Mamba-Integer V2 Speedrun Script ---
# This script automates the setup and training of the Integer-Only Mamba model.
# Designed for L4 / A100 / H100 instances.

set -e # Exit on error

echo "--- Phase 1: Environment Setup ---"
# 1. Install uv (Fast Python manager)
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.cargo/bin:$PATH"

# 2. Create and activate venv
[ -d ".venv" ] || uv venv
source .venv/bin/activate

# 3. Install core dependencies
# Need torch (cuda), triton, and other basics
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
uv pip install triton numpy sympy

echo "--- Phase 2: Toolchain Setup (Rust & Tokenizer) ---"
# 4. Install Rust
command -v cargo &> /dev/null || (curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y)
source "$HOME/.cargo/env"

# 5. Build Tokenizer
# We assume the rust_tokenizer.py uses ctypes to load a built .so or similar.
# For our current setup, we just need to ensure the binary is ready.
cd scripts
# python3 prepare_data.py # This is usually done once
cd ..

echo "--- Phase 3: Hardware Optimization (Triton) ---"
# 6. Setup PYTHONPATH for the optimized kernels
export PYTHONPATH=$PYTHONPATH:$(pwd)/src:$(pwd)/../gemma-intelligent/conv/src/bitnet-odp/src

# 7. Check for custom CUDA kernels (Dyadic Scan)
if [ ! -f "src/cuda_kernels/libdyadic_mamba.so" ]; then
    echo "Warning: libdyadic_mamba.so not found. Attempting to build..."
    cd src/cuda_kernels && make || echo "Failed to build CUDA kernel. Will fallback to Triton Scan."
    cd ../..
fi

echo "--- Phase 4: Launching Training (V2 Optimized) ---"
# Launch training with unbuffered output
# torch.compile will take 1-2 minutes to JIT the first time.
PYTHONUNBUFFERED=1 python3 -u scripts/train_mamba_integer.py | tee training_speedrun.log

echo "--- Speedrun Complete ---"
