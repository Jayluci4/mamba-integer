# Mamba-Integer: Verifiable Dyadic-Cayley Architecture

Mamba-Integer is a research-grade implementation of a purely integer-native State Space Model (SSM). By replacing transcendental operations (Exp, Sin, Cos, Sqrt) with algebraic equivalents (Shifts, Adds, Polynomials), Mamba-Integer enables high-performance, verifiable AI inference compatible with Zero-Knowledge (ZK) proofs and Fully Homomorphic Encryption (FHE).

## 🚀 Key Features

*   **Dyadic-Cayley Transform:** Hardware-friendly spectral transforms using the 3-shear lifting scheme.
*   **Integer-Only Selective Scan:** Linear recurrence implemented via dyadic rational multipliers ($h_t = (h_{t-1} \cdot n) \gg k$), eliminating `exp()` bottlenecks.
*   **Rational Squareplus Activation:** Division-free polynomial activation ($0.5(x + \sqrt{x^2+4})$) using Newton-RSQRT. Replaces potentially explosive $x^2$ or transcendental SiLU.
*   **BitShift Norm:** Power-of-2 normalization with learnable integer scalar. Replaces RMSNorm.
*   **Rust-Based BPE Tokenizer:** High-performance, dependency-free tokenizer exposed via C-API for maximum portability.
*   **ZK-Optimal Architecture:** 16x reduction in circuit depth (LogRows 17) and 100% elimination of lookup tables.
*   **Hardware Independence:** Includes a standalone C++ Inference Engine that runs without PyTorch.

## ⚠️ Known Scientific Findings (The "Stiff Memory" Trap)
During the initial 15k step training run, we observed a phenomenon where the model's `decay_nums` (memory retention parameters) clustered around `0.9` (Long-Term Memory), effectively destroying the "Fast Decay" heads required for local syntax.
*   **Cause:** Initial gradients during the stabilization phase pushed all decay parameters towards saturation (32,767).
*   **Effect:** The model learns global semantic clusters ("Lily", "Forest") but struggles with local grammar coherence.
*   **Fix for V2:** Freeze `base_decay_nums` for the first 1000 steps (Warmup) or clamp the `decay_mod` range to preserve the HiPPO initialization spectrum.

## 📂 Repository Structure

*   `src/`: Core implementation.
    *   `mamba_integer_model.py`: The main model definition (Stable v3).
    *   `dyadic_hippo.py`: Initialization logic.
    *   `rust_tokenizer.py`: Python wrapper for the Rust BPE library.
    *   `monitor.py`: Mission Control dashboard.
    *   `cuda_kernels/`: Pure CUDA C++ implementations (`.cu`) and compiled libraries (`.so`).
    *   `rust_tokenizer/`: Source code for the high-performance BPE tokenizer.
    *   `cpp_engine/`: Standalone C++ inference runtime.
*   `scripts/`: Utilities.
    *   `train_mamba_integer.py`: Pre-training loop (TinyStories).
    *   `prepare_rust_bpe.py`: Script to train the BPE vocabulary.
    *   `sft_mamba_integer.py`: Instruction Tuning loop.
    *   `export_int8.py`: Export trained model to binary format.
    *   `inference.py`: Python-based inference and analysis.
*   `configs/`: Model architecture definitions and BPE merge files.
*   `tests/`: Unit tests and verification scripts.
*   `archive/`: Older experimental scripts.
*   `docs/`: Technical reports.

## 🛠 Setup

### Prerequisites
*   NVIDIA GPU (L4 or better recommended)
*   CUDA Toolkit 11.5+
*   Rust (cargo)
*   PyTorch 2.x

### Build Kernels & Tokenizer
```bash
# Build CUDA Kernels
cd src/cuda_kernels
nvcc -shared -Xcompiler -fPIC -o libdyadic_mamba.so dyadic_mamba_kernel.cu
nvcc -shared -Xcompiler -fPIC -o libbitshift_norm.so bitshift_norm.cu
nvcc -shared -Xcompiler -fPIC -o libsquareplus.so squareplus_kernel.cu
nvcc -shared -Xcompiler -fPIC -o libbitlinear.so bitlinear.cu
nvcc -shared -Xcompiler -fPIC -o libconv1d_step.so conv1d_step.cu

# Build Rust Tokenizer
cd ../rust_tokenizer
cargo build --release
cp target/release/librustbpe.so ../cuda_kernels/
```

## 📖 Usage

### 1. Prepare Tokenizer
```bash
python scripts/prepare_rust_bpe.py
```

### 2. Pre-Training (from Scratch)
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
python scripts/train_mamba_integer.py
```

### 3. Deployment (C++ Engine)
Export to binary and run:
```bash
python scripts/export_int8.py
cd src/cpp_engine
nvcc -o mamba_infer mamba_infer.cpp -L../cuda_kernels -ldyadic_mamba -lbitshift_norm -lsquareplus -lbitlinear -lconv1d_step
./mamba_infer
```

## 📜 License
See `LICENSE` file. Strictly proprietary.
