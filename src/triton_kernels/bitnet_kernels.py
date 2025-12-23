
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

# --- Forward Kernels ---

@triton.jit
def quantize_activations_kernel(
    x_ptr, x_quant_ptr, scale_ptr,
    n_elements, BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start = pid * BLOCK_SIZE
    offsets = row_start + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offsets)
    
    abs_x = tl.abs(x)
    max_val = tl.max(abs_x, axis=0)
    scale = max_val if max_val > 1e-8 else 1e-8
    tl.store(scale_ptr + pid, scale)
    
    q_factor = 127.0 / scale
    x_quant = libdevice.rint(x * q_factor)
    x_quant = tl.clamp(x_quant, -127.0, 127.0)
    tl.store(x_quant_ptr + offsets, x_quant)

def fast_quantize_activations(x):
    x_flat = x.view(-1, x.shape[-1])
    n_rows, hidden_dim = x_flat.shape
    x_quant = torch.empty_like(x_flat, dtype=x.dtype)
    scale = torch.empty(n_rows, device=x.device, dtype=x.dtype)
    BLOCK_SIZE = triton.next_power_of_2(hidden_dim)
    quantize_activations_kernel[(n_rows,)](x_flat, x_quant, scale, n_rows * hidden_dim, BLOCK_SIZE=BLOCK_SIZE)
    return x_quant.view_as(x), scale.view(*x.shape[:-1], 1) / 127.0

# --- Fused BitNet MatMul Forward ---

@triton.jit
def bitnet_matmul_kernel(
    a_ptr, b_ptr, c_ptr, scale_a_ptr, scale_b_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m
    
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs).to(tl.int8)
        b = tl.load(b_ptrs).to(tl.int8)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
        
    scale_a = tl.load(scale_a_ptr + offs_am)
    scale_b = tl.load(scale_b_ptr + offs_bn)
    
    c = accumulator.to(tl.float32) * scale_a[:, None] * scale_b[None, :]
    
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

# --- NEW: BitNet MatMul Backward (dL/dX) ---
# Note: In training, we need dL/dX and dL/dW. 
# Since W is ternary, grad flows to latent W. 
# This is essentially standard MatMul but with scales applied.

def fast_bitnet_matmul_backward(grad_output, x_quant, w_quant, x_scale, w_scale):
    """
    dL/dX = (grad_output * w_scale) @ w_quant
    dL/dW = (grad_output * x_scale)^T @ x_quant
    """
    # 1. Rescale grad_output for input gradient
    # grad_output is [M, N], w_scale is [N]
    # We need to apply scales before the core MatMul to keep it efficient
    g_scaled_x = grad_output * w_scale.view(1, -1)
    grad_x = torch.matmul(g_scaled_x, w_quant.float()) * x_scale.view(-1, 1)
    
    # 2. Rescale grad_output for weight gradient
    g_scaled_w = grad_output * x_scale.view(-1, 1)
    grad_w = torch.matmul(g_scaled_w.t(), x_quant.float()) * w_scale.view(-1, 1)
    
    return grad_x, grad_w

# --- Optimized BitShiftNorm ---

@triton.jit
def bitshift_norm_kernel(
    x_ptr, gamma_ptr, out_ptr, inv_rms_ptr,
    stride_x_row, stride_out_row, n_cols,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    row_ptr_x = x_ptr + pid * stride_x_row
    row_ptr_out = out_ptr + pid * stride_out_row
    
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    x = tl.load(row_ptr_x + offsets, mask=mask, other=0.0)
    
    mean = tl.sum(x, axis=0) / n_cols
    x_centered = x - mean
    var = tl.sum(x_centered * x_centered, axis=0) / n_cols
    std = tl.sqrt(var + 1e-9)
    k = libdevice.rint(tl.log2(std))
    if k < 0.0: k = 0.0
    scale = tl.exp2(-k)
    
    # Store inv_rms for backward (reuse the dyadic scale)
    tl.store(inv_rms_ptr + pid, scale)
    
    gamma = tl.load(gamma_ptr + offsets, mask=mask, other=1.0)
    tl.store(row_ptr_out + offsets, x_centered * scale * gamma, mask=mask)

def fast_bitshift_norm(x, gamma):
    x_flat = x.view(-1, x.shape[-1])
    n_rows, n_cols = x_flat.shape
    out = torch.empty_like(x_flat)
    inv_rms = torch.empty(n_rows, device=x.device, dtype=x.dtype)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    bitshift_norm_kernel[(n_rows,)](x_flat, gamma, out, inv_rms, x_flat.stride(0), out.stride(0), n_cols, BLOCK_SIZE=BLOCK_SIZE)
    return out.view_as(x), inv_rms
