
import torch
import triton
import triton.language as tl

@triton.jit
def dyadic_scan_fwd_kernel(
    u_ptr,      # [B, L, D]
    nums_ptr,   # [B, L, D]
    shifts_ptr, # [B, L, D]
    h_ptr,      # [B, L, D]
    stride_b, stride_l, stride_d,
    B, L, D,
    scale_bits: tl.constexpr
):
    # Map program ID to Batch and Dimension
    # We parallelize over Batch and Dimension. Scan is sequential over L.
    pid_b = tl.program_id(0) # Batch idx
    pid_d = tl.program_id(1) # Dim idx
    
    # Pointers to the start of the sequence for this (b, d)
    offset = pid_b * stride_b + pid_d * stride_d
    u_ptr += offset
    nums_ptr += offset
    shifts_ptr += offset
    h_ptr += offset
    
    # Accumulator (Integer State)
    # We need to scale u from float to int
    scale = float(1 << scale_bits)
    h_int = 0.0 # Using float for simplicity in Triton v2, or int64?
    # Triton handles float nicely. Let's stick to the dyadic logic.
    # h = h * (n / 2^k) + u
    
    h_curr = 0.0
    
    for t in range(L):
        # Load u, num, shift
        # Assuming contiguous in L for speed? stride_l usually 1 or D.
        # u_ptr is [B, L, D]. stride_l = D.
        ptr_off = t * stride_l
        
        u_val = tl.load(u_ptr + ptr_off)
        num_val = tl.load(nums_ptr + ptr_off)
        shift_val = tl.load(shifts_ptr + ptr_off)
        
        # Dyadic Decay: num / 2^shift
        # h = h * num >> shift
        # In float world: h * num * 2^-shift
        
        # Compute decay factor
        # We can't do bitshift on float.
        # decay = num_val * tl.exp2(-shift_val.to(tl.float32))
        
        # To match C++ "Integer" logic exactly:
        # We need to cast h to int?
        # Let's stay in Float domain for Triton efficiency on H100 (Tensor Cores)
        # but enforce the algebraic property.
        
        decay = num_val.to(tl.float32) * tl.exp2(-shift_val.to(tl.float32))
        
        h_curr = h_curr * decay + u_val
        
        tl.store(h_ptr + ptr_off, h_curr)

def dyadic_scan_triton(u, nums, shifts, scale_bits=15):
    B, L, D = u.shape
    h = torch.empty_like(u)
    
    grid = (B, D)
    dyadic_scan_fwd_kernel[grid](
        u, nums, shifts, h,
        u.stride(0), u.stride(1), u.stride(2),
        B, L, D,
        scale_bits
    )
    return h
