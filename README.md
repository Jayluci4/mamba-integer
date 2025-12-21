# Mamba-Integer: Verifiable Dyadic-Cayley Architecture

Mamba-Integer is a research-grade implementation of a purely integer-native State Space Model (SSM). By replacing transcendental operations (Exp, Sin, Cos, Sqrt) with algebraic equivalents (Shifts, Adds, Polynomials), Mamba-Integer enables high-performance, verifiable AI inference compatible with Zero-Knowledge (ZK) proofs and Fully Homomorphic Encryption (FHE).

## 🚀 Key Features

*   **Dyadic-Cayley Transform:** Hardware-friendly spectral transforms using the 3-shear lifting scheme.
*   **Integer-Only Selective Scan:** Linear recurrence implemented via dyadic rational multipliers ($h_t = (h_{t-1} \cdot n) \gg k$), eliminating `exp()` bottlenecks.
*   **Rational Squareplus Activation:** Division-free polynomial activation ($0.5(x + \sqrt{x^2+4})$) using Newton-RSQRT. Replaces potentially explosive $x^2$ or transcendental SiLU.
*   **BitShift Norm:** Power-of-2 normalization with learnable integer scalar. Replaces RMSNorm.
*   **ZK-Optimal Architecture:** 16x reduction in circuit depth (LogRows 17) and 100% elimination of lookup tables.
*   **BitNet Integration:** Compatible with 1.58-bit ternary weights for massive memory savings.
*   **Hardware Independence:** Includes a standalone C++ Inference Engine that runs without PyTorch.

## 📂 Repository Structure

*   `src/`: Core implementation.
    *   `mamba_integer_model.py`: The main model definition (Stable v3).
    *   `dyadic_hippo.py`: Initialization logic.
    *   `monitor.py`: Mission Control dashboard.
    *   `cuda_kernels/`: Pure CUDA C++ implementations (Scan, Norm, Activation, BitLinear).
    *   `cpp_engine/`: Standalone C++ inference runtime.
*   `scripts/`: Utilities.
    *   `train_mamba_integer.py`: Pre-training loop (TinyStories).
    *   `sft_mamba_integer.py`: Instruction Tuning loop (Alpaca).
    *   `export_int8.py`: Export trained model to binary format.
    *   `inference.py`: Python-based inference and analysis.
*   `configs/`: Model architecture definitions (e.g., `config_mamba_integer_l4.json`).
*   `tests/`: Unit tests for individual kernels.
*   `archive/`: Older experimental scripts.
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
nvcc -shared -Xcompiler -fPIC -o libbitlinear.so bitlinear.cu
nvcc -shared -Xcompiler -fPIC -o libconv1d_step.so conv1d_step.cu
```

## 📖 Usage

### 1. Pre-Training (from Scratch)
```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/src
python scripts/train_mamba_integer.py
```

### 2. Instruction Tuning (SFT)
```bash
python scripts/sft_mamba_integer.py
```

### 3. Deployment (C++ Engine)
First, export the model to binary:
```bash
python scripts/export_int8.py
```
Then build and run the C++ engine:
```bash
cd src/cpp_engine
nvcc -o mamba_infer mamba_infer.cpp -L../cuda_kernels -ldyadic_mamba -lbitshift_norm -lsquareplus -lbitlinear -lconv1d_step
./mamba_infer
```

## 📜 License
See `LICENSE` file. Strictly proprietary.
