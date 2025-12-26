#!/bin/bash

# =============================================================================
# Mamba-Integer V2 Setup and Training Script
# Portable setup that works on any fresh VM with NVIDIA GPU
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# =============================================================================
# Configuration
# =============================================================================

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$REPO_ROOT/.venv"
CONFIG_DIR="$REPO_ROOT/configs"
SCRIPTS_DIR="$REPO_ROOT/scripts"
SRC_DIR="$REPO_ROOT/src"
CUDA_KERNELS_DIR="$SRC_DIR/cuda_kernels"
RUST_TOKENIZER_DIR="$SRC_DIR/rust_tokenizer"

echo "=============================================="
echo "  Mamba-Integer V2 Setup"
echo "=============================================="
echo "Repository: $REPO_ROOT"
echo ""

# =============================================================================
# Phase 0: System Requirements Check
# =============================================================================

log_info "Phase 0: Checking system requirements..."

# Check for NVIDIA GPU
if ! command -v nvidia-smi &> /dev/null; then
    log_error "nvidia-smi not found. NVIDIA GPU and drivers required."
    exit 1
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)
log_success "GPU detected: $GPU_NAME"

# Detect GPU architecture for CUDA compilation
detect_cuda_arch() {
    local gpu_name="$1"
    case "$gpu_name" in
        *"A100"*) echo "80" ;;
        *"A10"*) echo "86" ;;
        *"L4"*) echo "89" ;;
        *"L40"*) echo "89" ;;
        *"H100"*) echo "90" ;;
        *"4090"*|*"4080"*|*"4070"*) echo "89" ;;
        *"3090"*|*"3080"*|*"3070"*) echo "86" ;;
        *"V100"*) echo "70" ;;
        *"T4"*) echo "75" ;;
        *) echo "80" ;; # Default to A100 arch
    esac
}

CUDA_ARCH=$(detect_cuda_arch "$GPU_NAME")
log_info "Target CUDA architecture: sm_$CUDA_ARCH"

# =============================================================================
# Phase 1: Install System Dependencies
# =============================================================================

log_info "Phase 1: Installing system dependencies..."

# Install uv if missing
if ! command -v uv &> /dev/null; then
    log_info "Installing uv package manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi
log_success "uv available: $(uv --version 2>/dev/null || echo 'installed')"

# Check/Install Rust
if ! command -v cargo &> /dev/null; then
    log_info "Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
fi
log_success "Rust available: $(cargo --version 2>/dev/null || echo 'installed')"

# Check/Install CUDA Toolkit (nvcc)
if ! command -v nvcc &> /dev/null; then
    log_warn "nvcc not found. Attempting to locate or install CUDA toolkit..."

    # Try common locations
    for cuda_path in /usr/local/cuda/bin /usr/local/cuda-12/bin /usr/local/cuda-11/bin; do
        if [ -f "$cuda_path/nvcc" ]; then
            export PATH="$cuda_path:$PATH"
            log_success "Found nvcc at $cuda_path"
            break
        fi
    done

    # If still not found, try to install
    if ! command -v nvcc &> /dev/null; then
        if command -v apt-get &> /dev/null; then
            log_info "Installing CUDA toolkit via apt..."
            sudo apt-get update
            sudo apt-get install -y nvidia-cuda-toolkit || {
                log_warn "apt install failed, trying cuda-toolkit-12-1..."
                sudo apt-get install -y cuda-toolkit-12-1 2>/dev/null || true
            }
        fi
    fi
fi

if command -v nvcc &> /dev/null; then
    log_success "nvcc available: $(nvcc --version | grep release | awk '{print $6}')"
else
    log_error "nvcc not found. Please install CUDA toolkit manually."
    log_info "Try: sudo apt-get install nvidia-cuda-toolkit"
    exit 1
fi

# =============================================================================
# Phase 2: Python Environment Setup
# =============================================================================

log_info "Phase 2: Setting up Python environment..."

# Create venv if not exists
if [ ! -d "$VENV_DIR" ]; then
    log_info "Creating virtual environment..."
    uv venv "$VENV_DIR"
fi

# Activate venv
source "$VENV_DIR/bin/activate"
log_success "Virtual environment activated"

# Install Python dependencies
log_info "Installing Python dependencies..."
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
uv pip install triton numpy sympy setuptools wheel datasets

log_success "Python dependencies installed"

# =============================================================================
# Phase 3: Build CUDA Kernels
# =============================================================================

log_info "Phase 3: Building CUDA kernels..."

if [ -d "$CUDA_KERNELS_DIR" ]; then
    cd "$CUDA_KERNELS_DIR"

    # Create Makefile with detected architecture
    cat > Makefile << EOF
NVCC = nvcc
NVCC_FLAGS = -O3 --compiler-options '-fPIC' -shared \\
    -gencode arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH} \\
    -gencode arch=compute_${CUDA_ARCH},code=compute_${CUDA_ARCH} \\
    --use_fast_math

all: libdyadic_mamba.so libbitshift_norm.so libsquareplus.so

libdyadic_mamba.so: dyadic_mamba_kernel.cu
	\$(NVCC) \$(NVCC_FLAGS) \$< -o \$@

libbitshift_norm.so: bitshift_norm.cu
	\$(NVCC) \$(NVCC_FLAGS) \$< -o \$@

libsquareplus.so: squareplus_kernel.cu
	\$(NVCC) \$(NVCC_FLAGS) \$< -o \$@

# Optional: warp scan kernel if exists
ifneq (,\$(wildcard warp_scan_kernel.cu))
all: libwarp_scan.so
libwarp_scan.so: warp_scan_kernel.cu
	\$(NVCC) \$(NVCC_FLAGS) \$< -o \$@
endif

clean:
	rm -f *.so
EOF

    make clean 2>/dev/null || true
    make
    cd "$REPO_ROOT"
    log_success "CUDA kernels built for sm_$CUDA_ARCH"
else
    log_warn "CUDA kernels directory not found at $CUDA_KERNELS_DIR"
fi

# =============================================================================
# Phase 4: Build Rust Tokenizer
# =============================================================================

log_info "Phase 4: Building Rust tokenizer..."

if [ -d "$RUST_TOKENIZER_DIR" ]; then
    cd "$RUST_TOKENIZER_DIR"

    # Verify cargo is available
    if ! command -v cargo &> /dev/null; then
        log_error "cargo not found. Please install Rust: curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
        exit 1
    fi

    # Build with error checking
    if ! cargo build --release; then
        log_error "Rust tokenizer build failed"
        exit 1
    fi

    # Ensure target directory exists
    mkdir -p "$CUDA_KERNELS_DIR"

    # Copy library with verification
    if [ -f "target/release/librustbpe.so" ]; then
        cp target/release/librustbpe.so "$CUDA_KERNELS_DIR/"
        if [ -f "$CUDA_KERNELS_DIR/librustbpe.so" ]; then
            log_success "Rust tokenizer built and copied to $CUDA_KERNELS_DIR/"
        else
            log_error "Failed to copy librustbpe.so to $CUDA_KERNELS_DIR/"
            exit 1
        fi
    elif [ -f "target/release/librustbpe.dylib" ]; then
        cp target/release/librustbpe.dylib "$CUDA_KERNELS_DIR/"
        log_success "Rust tokenizer built (macOS)"
    else
        log_error "librustbpe library not found after build. Check Cargo.toml crate-type."
        exit 1
    fi
    cd "$REPO_ROOT"
else
    log_error "Rust tokenizer directory not found at $RUST_TOKENIZER_DIR"
    exit 1
fi

# =============================================================================
# Phase 5: Prepare Dataset
# =============================================================================

log_info "Phase 5: Preparing dataset..."

DATASET_BIN="$SCRIPTS_DIR/tinystories_train.bin"

if [ ! -f "$DATASET_BIN" ]; then
    log_info "Dataset not found. Running offline tokenization..."

    if [ -f "$SCRIPTS_DIR/tokenize_dataset_offline.py" ]; then
        cd "$SCRIPTS_DIR"
        python tokenize_dataset_offline.py
        cd "$REPO_ROOT"

        if [ -f "$SCRIPTS_DIR/tinystories_train.bin" ]; then
            log_success "Dataset tokenized successfully"
        else
            log_error "Tokenization failed. Check for errors above."
            exit 1
        fi
    else
        log_error "tokenize_dataset_offline.py not found"
        exit 1
    fi
else
    DATASET_SIZE=$(du -h "$DATASET_BIN" | cut -f1)
    log_success "Dataset found: $DATASET_BIN ($DATASET_SIZE)"
fi

# =============================================================================
# Phase 6: Interactive Configuration
# =============================================================================

log_info "Phase 6: Configuration..."

# List available configs
echo ""
echo "Available configuration files:"
CONFIG_FILES=($(ls "$CONFIG_DIR"/*.json 2>/dev/null))

if [ ${#CONFIG_FILES[@]} -eq 0 ]; then
    log_error "No configuration files found in $CONFIG_DIR"
    exit 1
fi

for i in "${!CONFIG_FILES[@]}"; do
    filename=$(basename "${CONFIG_FILES[$i]}")
    echo "  [$((i+1))] $filename"
done

echo ""
read -p "Select config (1-${#CONFIG_FILES[@]}) [1]: " CONFIG_CHOICE
CONFIG_CHOICE=${CONFIG_CHOICE:-1}

if [ "$CONFIG_CHOICE" -lt 1 ] || [ "$CONFIG_CHOICE" -gt ${#CONFIG_FILES[@]} ]; then
    log_warn "Invalid choice, using default (1)"
    CONFIG_CHOICE=1
fi

SELECTED_CONFIG="${CONFIG_FILES[$((CONFIG_CHOICE-1))]}"
log_success "Selected: $(basename "$SELECTED_CONFIG")"

# Show current training config
echo ""
echo "Current training configuration:"
python3 -c "
import json
with open('$SELECTED_CONFIG') as f:
    cfg = json.load(f)
train = cfg.get('training', {})
print(f\"  Sequence Length: {train.get('seq_len', 512)}\")
print(f\"  Batch Size: {train.get('batch_size', 2)}\")
print(f\"  Gradient Accumulation: {train.get('gradient_accumulation_steps', 32)}\")
print(f\"  Total Steps: {train.get('total_steps', 15000)}\")
print(f\"  Learning Rate: {train.get('learning_rate', 1e-3)}\")
"

echo ""
read -p "Modify training parameters? (y/N): " MODIFY_CONFIG
MODIFY_CONFIG=${MODIFY_CONFIG:-N}

if [[ "$MODIFY_CONFIG" =~ ^[Yy]$ ]]; then
    # Read current values
    CURRENT_SEQ_LEN=$(python3 -c "import json; print(json.load(open('$SELECTED_CONFIG')).get('training', {}).get('seq_len', 512))")
    CURRENT_BATCH=$(python3 -c "import json; print(json.load(open('$SELECTED_CONFIG')).get('training', {}).get('batch_size', 2))")
    CURRENT_ACCUM=$(python3 -c "import json; print(json.load(open('$SELECTED_CONFIG')).get('training', {}).get('gradient_accumulation_steps', 32))")
    CURRENT_STEPS=$(python3 -c "import json; print(json.load(open('$SELECTED_CONFIG')).get('training', {}).get('total_steps', 15000))")
    CURRENT_LR=$(python3 -c "import json; print(json.load(open('$SELECTED_CONFIG')).get('training', {}).get('learning_rate', 0.001))")

    read -p "Sequence length [$CURRENT_SEQ_LEN]: " NEW_SEQ_LEN
    read -p "Batch size [$CURRENT_BATCH]: " NEW_BATCH
    read -p "Gradient accumulation steps [$CURRENT_ACCUM]: " NEW_ACCUM
    read -p "Total training steps [$CURRENT_STEPS]: " NEW_STEPS
    read -p "Learning rate [$CURRENT_LR]: " NEW_LR

    NEW_SEQ_LEN=${NEW_SEQ_LEN:-$CURRENT_SEQ_LEN}
    NEW_BATCH=${NEW_BATCH:-$CURRENT_BATCH}
    NEW_ACCUM=${NEW_ACCUM:-$CURRENT_ACCUM}
    NEW_STEPS=${NEW_STEPS:-$CURRENT_STEPS}
    NEW_LR=${NEW_LR:-$CURRENT_LR}

    # Update config
    python3 << EOF
import json
with open('$SELECTED_CONFIG', 'r') as f:
    cfg = json.load(f)

if 'training' not in cfg:
    cfg['training'] = {}

cfg['training']['seq_len'] = int($NEW_SEQ_LEN)
cfg['training']['batch_size'] = int($NEW_BATCH)
cfg['training']['gradient_accumulation_steps'] = int($NEW_ACCUM)
cfg['training']['total_steps'] = int($NEW_STEPS)
cfg['training']['learning_rate'] = float($NEW_LR)

with open('$SELECTED_CONFIG', 'w') as f:
    json.dump(cfg, f, indent=2)

print("Configuration updated!")
EOF
fi

# =============================================================================
# Phase 7: Pre-flight Checks
# =============================================================================

log_info "Phase 7: Pre-flight checks..."

# Set environment
export PYTHONPATH="$REPO_ROOT:$SRC_DIR:$PYTHONPATH"
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR="$REPO_ROOT/.triton_cache"

# Quick import test
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA device: {torch.cuda.get_device_name(0)}')
import triton
print(f'Triton: {triton.__version__}')
" || {
    log_error "Python environment check failed"
    exit 1
}

log_success "All checks passed"

# =============================================================================
# Phase 8: Launch Training
# =============================================================================

echo ""
echo "=============================================="
echo "  Ready to Start Training"
echo "=============================================="
echo ""
echo "Training will use:"
echo "  - Config: $(basename "$SELECTED_CONFIG")"
echo "  - GPU: $GPU_NAME (sm_$CUDA_ARCH)"
echo "  - Dataset: $DATASET_BIN"
echo ""

read -p "Start training now? (Y/n): " START_TRAINING
START_TRAINING=${START_TRAINING:-Y}

if [[ "$START_TRAINING" =~ ^[Yy]$ ]]; then
    echo ""
    read -p "Run in background with nohup? (y/N): " USE_NOHUP
    USE_NOHUP=${USE_NOHUP:-N}

    LOG_FILE="$REPO_ROOT/training_$(date +%Y%m%d_%H%M%S).log"

    if [[ "$USE_NOHUP" =~ ^[Yy]$ ]]; then
        log_info "Starting training in background..."
        nohup python -u "$SCRIPTS_DIR/train_mamba_integer.py" > "$LOG_FILE" 2>&1 &
        TRAIN_PID=$!
        log_success "Training started with PID: $TRAIN_PID"
        log_info "Log file: $LOG_FILE"
        log_info "Monitor with: tail -f $LOG_FILE"
    else
        log_info "Starting training in foreground..."
        PYTHONUNBUFFERED=1 python -u "$SCRIPTS_DIR/train_mamba_integer.py" 2>&1 | tee "$LOG_FILE"
    fi
else
    log_info "Training skipped. Run manually with:"
    echo ""
    echo "  source $VENV_DIR/bin/activate"
    echo "  export PYTHONPATH=\"$REPO_ROOT:$SRC_DIR:\$PYTHONPATH\""
    echo "  python $SCRIPTS_DIR/train_mamba_integer.py"
    echo ""
fi

log_success "Setup complete!"
