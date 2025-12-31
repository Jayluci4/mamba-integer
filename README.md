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

## 📊 Performance

### A100 80GB (Current)

| Version | Implementation | Step Time | Throughput | Speedup |
| :--- | :--- | :--- | :--- | :--- |
| V2 (baseline) | Naive SSD backward | 6.47s | 22.8k tok/s | 1.0x |
| **V2 (optimized)** | **Memory-efficient SSD backward** | **3.0s** | **53.4k tok/s** | **2.3x** |

### L4 GPU (Legacy)

| Version | Implementation | Step Time | Speedup |
| :--- | :--- | :--- | :--- |
| V1 | Standard PyTorch | ~50.0s | 1.0x |
| V2 | Fused Triton + Inductor | ~6.0s | 8.3x |

## 🔧 SSD Backward Pass Optimization

The key bottleneck in Mamba-2 SSD training is the backward pass, which was 20x slower than forward (86% of training time). We optimized this by avoiding materialization of the full hidden state tensor.

### The Problem

Original backward computed the hidden state `h` explicitly:
```python
# h shape: [B, n_heads, n_chunks, chunk_size, d_state, d_head]
# For B=10, n_heads=24, n_chunks=16, cs=64, d_state=64, d_head=64
# This is 10 * 24 * 16 * 64 * 64 * 64 * 4 bytes = 10GB per forward pass!
h = einsum('bhnij,bhnjs,bhnjd->bhnisd', L, B, X)
grad_C = einsum('bhnid,bhnisd->bhnis', grad_Y, h)
```

### The Solution

Use the SSD identity `Y = (L * CB) @ X` where `CB = C @ B.T` has shape `[cs, cs]` instead of `[d_state, d_head]`:

```python
# CB shape: [B, n_heads, n_chunks, cs, cs] = 10 * 24 * 16 * 64 * 64 * 4 = 157MB
CB = einsum('bhnis,bhnjs->bhnij', C, B)
L_CB = L * CB
grad_CB = grad_L_CB * L
grad_C = einsum('bhnij,bhnjs->bhnis', grad_CB, B)  # Direct, no h needed
```

### Results

| Metric | Before | After | Improvement |
| :--- | :--- | :--- | :--- |
| SSD fwd+bwd kernel | 52ms | 9ms | 5.6x faster |
| Backward pass | 1548ms | 518ms | 3.0x faster |
| Training step | 1794ms | 767ms | 2.3x faster |
| Peak memory | 58.7GB | 44.6GB | -14GB |

See `src/triton_kernels/ssd_multihead.py` for the implementation.

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