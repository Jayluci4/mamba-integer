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
    pid_b = tl.program_id(0) # Batch idx
    pid_d = tl.program_id(1) # Dim idx
    
    # Pointers
    offset = pid_b * stride_b + pid_d * stride_d
    u_ptr += offset
    nums_ptr += offset
    shifts_ptr += offset
    h_ptr += offset
    
    h_curr = 0.0
    
    for t in range(L):
        ptr_off = t * stride_l
        
        u_val = tl.load(u_ptr + ptr_off)
        num_val = tl.load(nums_ptr + ptr_off)
        shift_val = tl.load(shifts_ptr + ptr_off)
        
        decay = num_val.to(tl.float32) * tl.exp2(-shift_val.to(tl.float32))
        
        h_curr = h_curr * decay + u_val
        
        tl.store(h_ptr + ptr_off, h_curr)

@triton.jit
def dyadic_scan_bwd_kernel(
    grad_h_ptr, h_ptr, nums_ptr, shifts_ptr,
    grad_u_ptr, grad_nums_ptr,
    stride_b, stride_l, stride_d,
    B, L, D,
    scale_bits: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)
    
    offset = pid_b * stride_b + pid_d * stride_d
    grad_h_ptr += offset
    h_ptr += offset
    nums_ptr += offset
    shifts_ptr += offset
    grad_u_ptr += offset
    grad_nums_ptr += offset
    
    # Reverse Accumulator
    d_h_curr = 0.0
    
    # Iterate backwards
    for t in range(L - 1, -1, -1):
        ptr_off = t * stride_l
        
        # Load gradients from layer above
        gh_val = tl.load(grad_h_ptr + ptr_off)
        
        d_h_curr += gh_val
        
        # 1. grad_u = d_h_curr
        tl.store(grad_u_ptr + ptr_off, d_h_curr)
        
        # 2. grad_num
        shift_val = tl.load(shifts_ptr + ptr_off)
        decay_scale = tl.exp2(-shift_val.to(tl.float32))
        
        h_prev = 0.0
        if t > 0:
            h_prev = tl.load(h_ptr + (t - 1) * stride_l)
            
        g_num = d_h_curr * h_prev * decay_scale
        tl.store(grad_nums_ptr + ptr_off, g_num)
        
        # 3. Update d_h for next step (t-1)
        num_val = tl.load(nums_ptr + ptr_off)
        decay = num_val.to(tl.float32) * decay_scale
        
        d_h_curr = d_h_curr * decay

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

def dyadic_scan_backward_triton(grad_h, h, nums, shifts, scale_bits=15):
    B, L, D = grad_h.shape
    grad_u = torch.empty_like(grad_h)
    grad_nums = torch.empty_like(grad_h)
    
    grid = (B, D)
    dyadic_scan_bwd_kernel[grid](
        grad_h, h, nums, shifts,
        grad_u, grad_nums,
        grad_h.stride(0), grad_h.stride(1), grad_h.stride(2),
        B, L, D,
        scale_bits
    )
    return grad_u, grad_nums