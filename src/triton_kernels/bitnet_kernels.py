import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

@triton.jit
def quantize_activations_kernel(
    x_ptr,
    x_quant_ptr,
    scale_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Quantizes a block of activations (per-token or per-tensor depending on grid).
    This kernel handles a single 1D vector (row) if launched row-wise.
    """
    # Simple row-wise parallelization
    # Each program processes one row (token)
    pid = tl.program_id(0)
    
    # Offset for this row
    row_start = pid * BLOCK_SIZE
    
    # Load data
    offsets = row_start + tl.arange(0, BLOCK_SIZE)
    # Mask not strictly needed if BLOCK_SIZE == hidden_dim and padded
    # But for safety assuming hidden_dim fits in block or is masked
    # We assume standard sizes (512, 1024) fit in block
    x = tl.load(x_ptr + offsets)
    
    # 1. Compute Max Abs
    abs_x = tl.abs(x)
    max_val = tl.max(abs_x, axis=0)
    
    # Avoid zero division
    scale = max_val
    if scale < 1e-8:
        scale = 1e-8
        
    # Store scale
    tl.store(scale_ptr + pid, scale)
    
    # 2. Quantize: x * 127 / scale
    # 127.0 is for int8
    q_factor = 127.0 / scale
    x_scaled = x * q_factor
    x_rounded = libdevice.rint(x_scaled)
    
    # Clamp
    x_clamped = tl.clamp(x_rounded, -127.0, 127.0)
    
    # Store
    tl.store(x_quant_ptr + offsets, x_clamped)


def fast_quantize_activations(x):
    """
    Triton wrapper for dynamic activation quantization.
    x: [Batch*Seq, Hidden] or [Batch, Seq, Hidden]
    """
    x_flat = x.view(-1, x.shape[-1])
    n_rows, hidden_dim = x_flat.shape
    
    # Output tensors
    x_quant = torch.empty_like(x_flat)
    scale = torch.empty(n_rows, device=x.device, dtype=x.dtype)
    
    # Grid: One program per row
    # Block size must cover hidden_dim (next power of 2)
    BLOCK_SIZE = triton.next_power_of_2(hidden_dim)
    
    # Launch
    quantize_activations_kernel[(n_rows,)](
        x_flat,
        x_quant,
        scale,
        n_rows * hidden_dim,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return x_quant.view_as(x), scale.view(*x.shape[:-1], 1) / 127.0


@triton.jit
def bitshift_norm_kernel(
    x_ptr,
    gamma_ptr,
    out_ptr,
    stride_x_row,
    stride_out_row,
    n_cols,
    BLOCK_SIZE: tl.constexpr
):
    """
    BitShiftNorm in Triton.
    1. Compute Variance (Mean Centered?)
       BitShiftNorm usually assumes x is centered or centers it.
       Original code: x = x - x.mean()
    2. Compute k = round(log2(sqrt(var)))
    3. Shift x >> k
    4. Multiply by gamma
    """
    pid = tl.program_id(0)
    
    # Pointers for this row
    row_ptr_x = x_ptr + pid * stride_x_row
    row_ptr_out = out_ptr + pid * stride_out_row
    
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    
    x = tl.load(row_ptr_x + offsets, mask=mask, other=0.0)
    gamma = tl.load(gamma_ptr + offsets, mask=mask, other=1.0)
    
    # 1. Mean Centering
    mean = tl.sum(x, axis=0) / n_cols
    x_centered = x - mean
    
    # 2. Variance
    x_sq = x_centered * x_centered
    var = tl.sum(x_sq, axis=0) / n_cols
    
    # 3. Calculate Scale k
    # sqrt(var)
    std = tl.sqrt(var + 1e-9)
    # log2(std)
    log_std = tl.log2(std)
    # round
    k = libdevice.rint(log_std)
    # clamp min 0
    if k < 0.0:
        k = 0.0
        
    # Scale factor = 1 / 2^k
    scale = tl.exp2(-k)
    
    # 4. Apply
    out = x_centered * scale * gamma
    
    tl.store(row_ptr_out + offsets, out, mask=mask)

def fast_bitshift_norm(x, gamma):
    """
    Triton wrapper for BitShiftNorm.
    """
    x_flat = x.view(-1, x.shape[-1])
    n_rows, n_cols = x_flat.shape
    
    out = torch.empty_like(x_flat)
    
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    
    bitshift_norm_kernel[(n_rows,)](
        x_flat,
        gamma,
        out,
        x_flat.stride(0),
        out.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out.view_as(x)

# --- Fused BitNet MatMul ---

@triton.jit
def bitnet_matmul_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Pointers to scales
    scale_a_ptr, scale_b_ptr,
    # Matrix dimensions
    M, N, K,
    # Strides
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    """
    Matrix Multiplication C = (A @ B) * scale_a * scale_b
    A: [M, K] int8 (activations)
    B: [K, N] int8 (weights, transposed?) -> usually weights are [N, K] stored as [K, N] for computation?
       Let's assume standard row-major A and col-major B logic, or standard PyTorch Linear: y = x @ w.T
       If w is [Out, In], w.T is [In, Out].
       Here M=Batch, N=Out, K=In.
       A is [M, K]. B is [K, N] (which is w.T).
       
    scale_a: [M] float (per-token scale)
    scale_b: [N] float (per-channel scale for weights)
    """
    # 1. Program ID
    pid = tl.program_id(axis=0)
    
    # 2. Grid Mapping
    # Number of blocks along M direction
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    
    # Map 1D pid to 2D grid
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m
    
    # 3. Create block pointers
    # Offsets for A [M, K]
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    
    # Offsets for B [K, N]
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
    
    # 4. Main Loop
    # Accumulator in int32 for precision with int8 inputs
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)
    
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load A and B blocks
        # A needs mask if M is not multiple of BLOCK, but we used modulo.
        # However, K loop must use mask?
        # Actually standard matmul tutorial assumes masked load not needed if size aligned or padded.
        # We'll use mask for safety on K dimension if needed, but usually we iterate up to K.
        # Let's assume K is multiple of BLOCK_SIZE_K for simplicity (usually 32/64).
        # Or check bounds.
        
        # Load A (int8) - cast to FP32? NO, keep int8 for dot.
        # But triton dot expects specific types.
        # int8 inputs -> int32 accumulator is supported on Ampere+.
        # On older GPUs, we might need float conversion.
        # Let's assume inputs are loaded as int8.
        
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        
        # Matrix Multiply
        # accumulator += dot(a, b)
        accumulator += tl.dot(a, b)
        
        # Advance pointers
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
        
    # 5. Epilogue: Rescale and Store
    
    # Load scales
    # scale_a is vector [M], corresponding to rows of C
    scale_a = tl.load(scale_a_ptr + offs_am) # [BLOCK_M]
    
    # scale_b is vector [N], corresponding to cols of C
    scale_b = tl.load(scale_b_ptr + offs_bn) # [BLOCK_N]
    
    # Broadcast scales
    # C = acc * scale_a[:, None] * scale_b[None, :]
    c = accumulator.to(tl.float32)
    c = c * scale_a[:, None]
    c = c * scale_b[None, :]
    
    # Store C
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    
    # Mask for output (handle edge cases)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    
    # Convert to output dtype (handled by store usually, but let's be explicit if needed)
    # Assuming output ptr is fp16 or fp32
    tl.store(c_ptrs, c, mask=c_mask)

def fast_bitnet_matmul(x_quant, w_quant, x_scale, w_scale):
    """
    Wrapper for Fused BitNet MatMul.
    
    Args:
        x_quant: [B, K] int8 (Activations) -> Usually flattened [Batch*Seq, Hidden]
        w_quant: [N, K] int8 (Weights) -> Linear layer weights are usually [Out, In]
        x_scale: [B] float
        w_scale: [N] float
        
    Returns:
        y: [B, N] float
    """
    # Check inputs
    assert x_quant.dtype == torch.float32 or x_quant.dtype == torch.int8 # Triton loads int8?
    # Actually, PyTorch 'int8' is what we want.
    # Note: Triton tutorials often load fp16 for tensor cores.
    # For Int8 Tensor Cores, inputs MUST be int8.
    
    M, K = x_quant.shape
    N, K_w = w_quant.shape
    assert K == K_w, "Inner dimensions must match"
    
    # Weights are [N, K]. We need [K, N] for the kernel logic (B matrix).
    # We can transpose w_quant logically by swapping strides.
    # Logic in kernel: B is [K, N].
    # w_quant is [N, K]. Transpose(w_quant) is [K, N].
    # stride_bk should be stride of K in w_quant.T -> stride of K in w_quant -> 1
    # stride_bn should be stride of N in w_quant.T -> stride of N in w_quant -> K
    
    # Wait, usually Linear weights are Row-Major [Out, In] -> [N, K].
    # stride(0) = K, stride(1) = 1.
    # If we treat it as B matrix [K, N]:
    # It is effectively Transposed.
    # So we pass B_ptr = w_quant.
    # stride_bk (stride along K dim of logic) = stride along 2nd dim of w_quant = 1.
    # stride_bn (stride along N dim of logic) = stride along 1st dim of w_quant = K.
    
    # Output
    y = torch.empty((M, N), device=x_quant.device, dtype=torch.float32) # or fp16
    
    # Grid
    BLOCK_M = 128
    BLOCK_N = 128
    BLOCK_K = 32
    
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), )
    
    bitnet_matmul_kernel[grid](
        # Pointers
        x_quant, w_quant, y,
        x_scale, w_scale,
        # Dimensions
        M, N, K,
        # Strides
        x_quant.stride(0), x_quant.stride(1), # stride_am, stride_ak
        w_quant.stride(1), w_quant.stride(0), # stride_bk, stride_bn (See note above)
        y.stride(0), y.stride(1),             # stride_cm, stride_cn
        # Meta-parameters
        BLOCK_SIZE_M=BLOCK_M, BLOCK_SIZE_N=BLOCK_N, BLOCK_SIZE_K=BLOCK_K
    )
    
    return y