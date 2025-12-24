# Mamba-Integer: Verifiable Dyadic-Cayley Architecture

Mamba-Integer is a production-grade implementation of a purely integer-native State Space Model (SSM). By replacing all floating-point and transcendental operations (Exp, Sin, Cos, Sqrt) with algebraic equivalents (Shifts, Adds, Polynomials), Mamba-Integer enables high-performance, verifiable AI inference compatible with Zero-Knowledge (ZK) proofs and Fully Homomorphic Encryption (FHE).

## 🚀 Key Features

*   **V2 High-Speed Pipeline:** Fused Triton kernels and analytic backward passes achieving **~6s per step** on L4 GPUs (a 10x speedup over V1).
*   **BitNet b1.58 Integration:** Implementation of ternary weights $\{-1, 0, 1\}$ allowing for **multiplication-free** linear layers.
*   **Dyadic-Cayley Transform:** Hardware-friendly spectral transforms using the 3-shear lifting scheme for positional embeddings (RoPE).
*   **Integer-Only Selective Scan:** Linear recurrence implemented via dyadic rational multipliers ($h_t = (h_{t-1} \cdot n) \gg k$), eliminating `exp()` bottlenecks.
*   **Rational Squareplus Activation:** Division-free polynomial activation ($0.5(x + \sqrt{x^2+4})$) using Newton-RSQRT.
*   **BitShift Norm:** Power-of-2 normalization with learnable integer scalar. Replaces standard RMSNorm with shift-add logic.
*   **ZK-Optimal Design:** 16x reduction in circuit depth (LogRows 17) and 100% elimination of lookup tables.

## 📊 Performance (L4 GPU)

| Version | Implementation | Throughput (Step Time) | Relative Speed |
| :--- | :--- | :--- | :--- |
| **V1** | Standard PyTorch | ~50.0s | 1.0x |
| **V2** | **Fused Triton + Inductor** | **~6.0s** | **8.3x - 10x** |

## 📂 Repository Structure

*   `src/`: Core implementation.
    *   `mamba_integer_model.py`: The main model definition (V2 Optimized).
    *   `triton_kernels/`: High-performance fused kernels for MatMul, Norm, and Scan.
    *   `cuda_kernels/`: Raw CUDA implementations for legacy and standalone C++ support.
    *   `rational_bitnet.py`: Integer-only BitNet b1.58 linear layers.
    *   `dyadic_hippo.py`: Numerical initialization for stable recurrence.
*   `scripts/`: Utilities.
    *   `train_mamba_integer.py`: High-speed pre-training loop with auto-resume.
    *   `inference.py`: Analysis and generation utility.
    *   `speedrun.sh`: One-click environment setup and training launch.
*   `faster/`: Experimental pure C++/CUDA inference engine.

## 🛠 Setup & Usage

### One-Click Speedrun (Recommended)
This script automates environment setup, kernel compilation, and launches training.
```bash
chmod +x speedrun.sh
./speedrun.sh
```

### Manual Setup
1. **Environment:**
   ```bash
   pip install torch triton numpy sympy
   export PYTHONPATH=$PYTHONPATH:$(pwd)/src:$(pwd)/../bitnet-odp/src
   ```

2. **Build Kernels:**
   ```bash
   cd src/cuda_kernels && make
   ```

3. **Training:**
   ```bash
   python scripts/train_mamba_integer.py
   ```

## 📖 Roadmap
- [x] **V2 Optimization:** Fused Triton kernels for 10x throughput.
- [x] **Checkpoint Resume:** Robust model/optimizer/scheduler state recovery.
- [ ] **Dyadic Vision:** Integration of the Dyadic-NOVA tokenizer for Small Vision-Language Models (SVLM).
- [ ] **Pure C++ Training:** Migrating the backward pass to the `faster/` backend.

## 📜 License
Strictly proprietary. See `LICENSE` for details.