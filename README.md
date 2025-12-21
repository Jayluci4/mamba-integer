# Mamba-Integer: Verifiable Dyadic-Cayley Architecture

Mamba-Integer is a research-grade implementation of a purely integer-native State Space Model (SSM). By replacing transcendental operations (Exp, Sin, Cos, Sqrt) with algebraic equivalents (Shifts, Adds, Polynomials), Mamba-Integer enables high-performance, verifiable AI inference compatible with Zero-Knowledge (ZK) proofs and Fully Homomorphic Encryption (FHE).

## 🚀 Key Features

*   **Dyadic-Cayley Transform:** Hardware-friendly spectral transforms using the 3-shear lifting scheme.
*   **Integer-Only Selective Scan:** Linear recurrence implemented via dyadic rational multipliers ($h_t = (h_{t-1} \cdot n) \gg k$), eliminating `exp()` bottlenecks.
*   **Rational Squareplus Activation:** Division-free polynomial activation ($0.5(x + \sqrt{x^2+4})$) using Newton-RSQRT. Replaces potentially explosive $x^2$ or transcendental SiLU.
*   **BitShift Norm:** Power-of-2 normalization with learnable integer scalar. Replaces RMSNorm.
*   **ZK-Optimal Architecture:** 16x reduction in circuit depth (LogRows 17) and 100% elimination of lookup tables.
*   **BitNet Integration:** Compatible with 1.58-bit ternary weights for massive memory savings.

## 📂 Repository Structure

*   `src/`: Core implementation of Dyadic-Cayley components and the Mamba-Integer model.
    *   `mamba_integer_model.py`: The main model definition (Stable v3).
    *   `dyadic_hippo.py`: Initialization logic.
    *   `monitor.py`: Mission Control dashboard for stability tracking.
    *   `cuda_kernels/`: Pure CUDA C++ implementations of algebraic primitives (Scan, Norm, Activation).
*   `scripts/`: Utilities for training and inference.
    *   `train_mamba_integer.py`: Training loop for TinyStories.
    *   `inference.py`: Generation script with model state analysis.
*   `configs/`: Model architecture definitions (e.g., `config_mamba_integer_l4.json`).
*   `tests/`: Unit tests for individual kernels.
*   `archive/`: Older experimental scripts (Surgery, ZK Benchmarks).
*   `docs/`: Technical reports.

## 🛠 Setup

### Prerequisites
*   NVIDIA GPU (L4 or better recommended)
*   CUDA Toolkit 11.5+
*   PyTorch 2.x

### Build Kernels
```bash
cd src/cuda_kernels
nvcc -shared -Xcompiler -fPIC -o libdyadic_mamba.so dyadic_mamba_kernel.cu
nvcc -shared -Xcompiler -fPIC -o libbitshift_norm.so bitshift_norm.cu
nvcc -shared -Xcompiler -fPIC -o libsquareplus.so squareplus_kernel.cu
# (Optional) dyadic_rope_kernel.cu if using Transformer attention
```

## 📖 Usage

### Training from Scratch
```bash
# Ensure PYTHONPATH includes src/
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
python scripts/train_mamba_integer.py
```

### Inference
```bash
python scripts/inference.py
```

## 📜 License
See `LICENSE` file. Strictly proprietary.