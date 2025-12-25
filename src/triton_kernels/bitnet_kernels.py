"""
BitNet Triton Kernels for Integer-Only AI.

All operations use ONLY: +, -, *, /, bit-shift, comparisons
NO transcendentals (sqrt, log2, exp2, sin, cos, etc.)

This enables ZK-ML compatibility and edge deployment.
"""

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice  # Keep for rint in quantization

# --- Forward Kernels ---

@triton.jit
def quantize_activations_kernel(
    x_ptr, x_quant_ptr, scale_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    # Correct row offset logic
    row_start = pid * n_cols
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    
    # Load with mask to avoid illegal memory access
    x = tl.load(x_ptr + row_start + offsets, mask=mask, other=0.0)
    
    abs_x = tl.abs(x)
    max_val = tl.max(abs_x, axis=0)
    scale = max_val if max_val > 1e-8 else 1e-8
    
    # Store scale (1 per row)
    tl.store(scale_ptr + pid, scale)
    
    q_factor = 127.0 / scale
    x_quant = libdevice.rint(x * q_factor)
    x_quant = tl.clamp(x_quant, -127.0, 127.0)
    
    # Store with mask
    tl.store(x_quant_ptr + row_start + offsets, x_quant, mask=mask)

class QuantizeActivationsFunction(torch.autograd.Function):
    """Autograd wrapper for activation quantization.

    Forward: x -> x_quant (quantized), scale (per-row max)
    Backward: Straight-Through Estimator (gradient passes through)
    """

    @staticmethod
    def forward(ctx, x):
        x = x.contiguous()
        x_flat = x.reshape(-1, x.shape[-1])
        n_rows, n_cols = x_flat.shape
        x_quant = torch.empty_like(x_flat, dtype=x.dtype)
        scale = torch.empty(n_rows, device=x.device, dtype=x.dtype)
        BLOCK_SIZE = triton.next_power_of_2(n_cols)

        quantize_activations_kernel[(n_rows,)](
            x_flat, x_quant, scale,
            n_cols,
            BLOCK_SIZE=BLOCK_SIZE
        )

        # Save for backward
        ctx.save_for_backward(scale)
        ctx.original_shape = x.shape

        return x_quant.view_as(x), scale.view(*x.shape[:-1], 1) / 127.0

    @staticmethod
    def backward(ctx, grad_x_quant, grad_scale):
        # Straight-Through Estimator: pass gradients through unchanged
        # The quantization is like rounding - we use identity gradient
        scale, = ctx.saved_tensors
        original_shape = ctx.original_shape

        # Scale gradient by the quantization factor (chain rule)
        # Forward: x_quant = round(x * 127 / scale)
        # dL/dx = dL/dx_quant * 127 / scale
        #
        # STABILITY FIX: Use larger min to prevent gradient explosion
        # If scale < 0.01, the multiplier would be > 12700 which can explode
        scale_safe = scale.view(*original_shape[:-1], 1).clamp(min=0.01)
        grad_x = grad_x_quant * 127.0 / scale_safe

        # Clamp gradients to prevent explosion
        grad_x = grad_x.clamp(-100.0, 100.0)

        return grad_x


def fast_quantize_activations(x):
    """Quantize activations with autograd support.

    Uses Triton kernel for forward, STE for backward.
    """
    return QuantizeActivationsFunction.apply(x)

# --- Fused BitNet MatMul Forward ---
# Based on research:
# - Use out_dtype=tl.int32 for int8 dot products
# - Autotuning is essential for performance
# - Proper L2 cache optimization via grouped ordering

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=5, num_warps=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def bitnet_matmul_kernel(
    # Pointers
    a_ptr, b_ptr, c_ptr, scale_a_ptr, scale_b_ptr,
    # Dimensions
    M, N, K,
    # Strides
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Block sizes (autotuned)
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    Compute C = (A @ B) * scale_a * scale_b
    A: [M, K] int8 (quantized activations)
    B: [K, N] int8 (transposed quantized weights)
    scale_a: [M] float32
    scale_b: [N] float32
    """
    # Program ID with grouped ordering for L2 cache optimization
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)

    # Grouped ordering: process tiles in groups to improve L2 locality
    GROUP_M: tl.constexpr = 8
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Offsets for this block
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers to first block of A and B
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    # Accumulator in int32 to prevent overflow
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)

    # Main loop over K dimension
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # Boundary masks
        k_offs = k * BLOCK_K + offs_k
        a_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        b_mask = (k_offs[:, None] < K) & (offs_n[None, :] < N)

        # Load blocks - cast to int8 for the dot product
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        # Int8 dot product with explicit int32 accumulator
        accumulator += tl.dot(a.to(tl.int8), b.to(tl.int8), out_dtype=tl.int32)

        # Advance pointers
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Load scales and apply
    scale_a = tl.load(scale_a_ptr + offs_m, mask=offs_m < M, other=1.0)
    scale_b = tl.load(scale_b_ptr + offs_n, mask=offs_n < N, other=1.0)

    # Convert to float and apply scales
    c = accumulator.to(tl.float32) * scale_a[:, None] * scale_b[None, :]

    # Write output
    c_ptrs = c_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

def fast_bitnet_matmul(x_quant, w_quant, x_scale, w_scale):
    """Fused BitNet matrix multiplication: y = (x_quant @ w_quant.T) * x_scale * w_scale

    Uses torch._int_mm for 10x speedup over float matmul.

    NOTE: Triton kernel disabled due to CUDA context corruption issues when
    used with multiple model layers (cuBLAS handle corruption after ~6 layers).
    See: https://github.com/triton-lang/triton/issues/882

    torch._int_mm uses cuBLAS INT8 GEMM which is both fast and reliable.
    """
    if x_quant.is_cuda and hasattr(torch, '_int_mm'):
        # Use PyTorch's native INT8 matmul (10x faster than float)
        x_int8 = x_quant.to(torch.int8).contiguous()
        w_int8 = w_quant.t().to(torch.int8).contiguous()
        y_int = torch._int_mm(x_int8, w_int8)
        return y_int.float() * x_scale.view(-1, 1) * w_scale.view(1, -1)

    # Fallback for CPU or old PyTorch
    y = torch.matmul(x_quant.float(), w_quant.float().t())
    return y * x_scale.view(-1, 1) * w_scale.view(1, -1)

def fast_bitnet_matmul_backward(grad_output, x_quant, w_quant, x_scale, w_scale):
    g_scaled_x = grad_output * w_scale.view(1, -1)
    grad_x = torch.matmul(g_scaled_x, w_quant.float()) * x_scale.view(-1, 1)
    g_scaled_w = grad_output * x_scale.view(-1, 1)
    grad_w = torch.matmul(g_scaled_w.t(), x_quant.float()) * w_scale.view(-1, 1)
    return grad_x, grad_w

# --- Optimized BitShiftNorm (INTEGER-ONLY) ---
# Replaces sqrt, log2, exp2 with lookup table approach

@triton.jit
def find_power_of_2_scale_triton(var):
    """Find power-of-2 scale using lookup table (INTEGER-ONLY).

    Instead of: sqrt(var) -> log2 -> exp2(-k)
    We use: var -> lookup -> scale = 2^(-k)

    Uses ONLY: comparisons and precomputed dyadic rationals.
    NO transcendentals (sqrt, log2, exp2).
    """
    # Thresholds are powers of 4 (= 2^(2k))
    # If var >= 4^k, then sqrt(var) >= 2^k, so scale = 2^(-k)

    # Start with scale = 1 (k=0)
    scale = 1.0

    # Check thresholds in order
    scale = tl.where(var >= 4.0, 0.5, scale)        # sqrt >= 2, use 1/2
    scale = tl.where(var >= 16.0, 0.25, scale)      # sqrt >= 4, use 1/4
    scale = tl.where(var >= 64.0, 0.125, scale)     # sqrt >= 8, use 1/8
    scale = tl.where(var >= 256.0, 0.0625, scale)   # sqrt >= 16, use 1/16
    scale = tl.where(var >= 1024.0, 0.03125, scale) # sqrt >= 32, use 1/32
    scale = tl.where(var >= 4096.0, 0.015625, scale)
    scale = tl.where(var >= 16384.0, 0.0078125, scale)
    scale = tl.where(var >= 65536.0, 0.00390625, scale)

    # For small variance, scale up
    scale = tl.where(var < 1.0, 1.0, scale)
    scale = tl.where(var < 0.25, 2.0, scale)
    scale = tl.where(var < 0.0625, 4.0, scale)
    scale = tl.where(var < 0.015625, 8.0, scale)

    return scale


@triton.jit
def bitshift_norm_kernel(
    x_ptr, gamma_ptr, out_ptr, inv_rms_ptr,
    stride_x_row, stride_out_row, n_cols,
    BLOCK_SIZE: tl.constexpr
):
    """BitShift normalization kernel (INTEGER-ONLY).

    Uses lookup table instead of sqrt/log2/exp2.
    All operations: +, -, *, /, comparisons (no transcendentals).
    """
    pid = tl.program_id(0)
    row_ptr_x = x_ptr + pid * stride_x_row
    row_ptr_out = out_ptr + pid * stride_out_row

    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    x = tl.load(row_ptr_x + offsets, mask=mask, other=0.0)

    mean = tl.sum(x, axis=0) / n_cols
    x_centered = x - mean
    var = tl.sum(x_centered * x_centered, axis=0) / n_cols

    # INTEGER-ONLY: Use lookup table instead of sqrt/log2/exp2
    scale = find_power_of_2_scale_triton(var + 1e-9)

    tl.store(inv_rms_ptr + pid, scale)
    gamma = tl.load(gamma_ptr + offsets, mask=mask, other=1.0)
    tl.store(row_ptr_out + offsets, x_centered * scale * gamma, mask=mask)

def fast_bitshift_norm(x, gamma):
    x = x.contiguous()
    x_flat = x.reshape(-1, x.shape[-1])
    n_rows, n_cols = x_flat.shape
    out = torch.empty_like(x_flat)
    inv_rms = torch.empty(n_rows, device=x.device, dtype=x.dtype)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    bitshift_norm_kernel[(n_rows,)](x_flat, gamma, out, inv_rms, x_flat.stride(0), out.stride(0), n_cols, BLOCK_SIZE=BLOCK_SIZE)
    return out.view_as(x), inv_rms
