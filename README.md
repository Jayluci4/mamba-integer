# Mamba-Integer: Verifiable Dyadic-Cayley Architecture

Mamba-Integer is a research-grade implementation of a purely integer-native State Space Model (SSM). By replacing transcendental operations (Exp, Sin, Cos, Sqrt) with algebraic equivalents (Shifts, Adds, Polynomials), Mamba-Integer enables high-performance, verifiable AI inference compatible with Zero-Knowledge (ZK) proofs and Fully Homomorphic Encryption (FHE).

## 🚀 Key Features

*   **Dyadic-Cayley Transform:** Hardware-friendly spectral transforms using the 3-shear lifting scheme.
*   **Integer-Only Selective Scan:** Linear recurrence implemented via dyadic rational multipliers, eliminating `exp()` bottlenecks.
*   **ZK-Optimal Architecture:** 16x reduction in circuit depth (LogRows 17) and 100% elimination of lookup tables.
*   **BitNet Integration:** Compatible with 1.58-bit ternary weights for massive memory savings.
*   **Custom CUDA Kernels:** Optimized high-performance kernels for Dyadic Scan, BitShift Norm, and Square Activation.

## 📂 Repository Structure

*   `src/`: Core implementation of Dyadic-Cayley components and the Mamba-Integer model.
    *   `cuda_kernels/`: Pure CUDA C++ implementations of algebraic primitives.
*   `scripts/`: Utilities for training from scratch, model surgery, and ZK benchmarking.
*   `docs/`: Technical reports and mathematical summaries of the Dyadic-Cayley approach.
*   `tests/`: Stress tests and verification scripts against ground truth implementations.
*   `configs/`: Model architecture definitions.

## 🛠 Setup

### Prerequisites
*   NVIDIA GPU (L4 or better recommended)
*   CUDA Toolkit 11.5+
*   PyTorch 2.x

### Build Kernels
```bash
cd src/cuda_kernels
nvcc -shared -Xcompiler -fPIC -o libdyadic_mamba.so dyadic_mamba_kernel.cu
# Repeat for other kernels
```

## 📖 Usage

### Training from Scratch
```bash
python scripts/train_mamba_integer.py
```

### Model Surgery (Convert existing Llama/BitNet models)
```bash
python scripts/surgery_dyadic.py --model_id <hf_model_id>
```

## 📜 License
MIT

## 📧 Contact
[Your Name/Organization]
