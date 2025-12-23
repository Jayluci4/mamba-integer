
import torch
import triton
import triton.language as tl

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
    x_rounded = tl.libdevice.round(x_scaled)
    
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
    k = tl.libdevice.round(log_std)
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
