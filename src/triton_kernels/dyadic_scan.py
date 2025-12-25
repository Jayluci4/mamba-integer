import torch
import triton
import triton.language as tl

# Associative scan combiner for linear recurrence: h[t] = a[t] * h[t-1] + b[t]
# State is (a, b) representing transformation h -> a*h + b
# Combiner: (a1, b1) ⊕ (a2, b2) = (a2*a1, a2*b1 + b2)
@triton.jit
def combine_fn(a1, b1, a2, b2):
    return a2 * a1, a2 * b1 + b2


@triton.jit
def dyadic_scan_parallel_kernel(
    u_ptr,      # [B, L, D] - input
    nums_ptr,   # [B, L, D] - decay numerators
    shifts_ptr, # [B, L, D] - decay shifts
    h_ptr,      # [B, L, D] - output
    stride_b, stride_l, stride_d,
    B: tl.constexpr,
    L: tl.constexpr,
    D: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    """Parallel associative scan for linear recurrence using Triton's built-in scan."""
    # Each program handles one (batch, dim) pair
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)

    # Base offset for this (batch, dim)
    base_offset = pid_b * stride_b + pid_d * stride_d

    # Process in blocks along L dimension
    # For each block, we do a parallel scan within the block
    # Then carry forward the final state to the next block

    # Running state from previous blocks
    carry_a = 1.0  # Identity for multiplication
    carry_b = 0.0  # Identity for addition

    for block_start in range(0, L, BLOCK_L):
        # Indices for this block
        block_size = tl.minimum(BLOCK_L, L - block_start)
        offs = tl.arange(0, BLOCK_L)
        mask = offs < block_size

        # Load data
        ptr_offs = base_offset + (block_start + offs) * stride_l
        u_vals = tl.load(u_ptr + ptr_offs, mask=mask, other=0.0)
        num_vals = tl.load(nums_ptr + ptr_offs, mask=mask, other=0)
        shift_vals = tl.load(shifts_ptr + ptr_offs, mask=mask, other=0)

        # Compute decay values: decay = num * 2^(-shift)
        decay = num_vals.to(tl.float32) * tl.exp2(-shift_vals.to(tl.float32))

        # For parallel scan, each position contributes (decay, u)
        # Apply carry from previous blocks first
        # h[0] = carry_a * carry_b + decay[0] * (carry_a * h_prev) + u[0]
        # But since h_prev = carry_b / (1 - carry_a) conceptually, we need to adjust

        # Actually, we apply the carry as: new_u[0] = carry_a * u[0] + carry_b
        # Then run scan normally
        # This is wrong - let me think again...

        # For linear recurrence h[t] = a[t] * h[t-1] + b[t]:
        # We can represent the full transformation as composition of (a, b) pairs
        # Starting with h[-1] = 0 (or carry from previous block)
        #
        # To apply carry: we modify the first element
        # h[0] = decay[0] * carry_b + u[0] with adjusted decay
        # Actually: h[t] = (∏_{i<=t} a[i]) * h[-1] + cumulative sum terms

        # Simplest approach: run parallel scan within block, then add carry contribution

        # Run associative scan: (a, b) ⊕ (a', b') = (a'*a, a'*b + b')
        # This computes cumulative products/sums
        scan_a, scan_b = tl.associative_scan((decay, u_vals), axis=0, combine_fn=combine_fn)

        # scan_a[t] = ∏_{i=0}^{t} decay[i]
        # scan_b[t] = Σ_{i=0}^{t} u[i] * ∏_{j=i+1}^{t} decay[j]

        # Add contribution from carry (h[-1] = carry_b with cumulative decay carry_a)
        # h[t] = scan_a[t] * carry_b + scan_b[t]
        h_vals = scan_a * carry_b + scan_b

        # Store results
        tl.store(h_ptr + ptr_offs, h_vals, mask=mask)

        # Update carry for next block
        # carry represents the transformation: h_out = carry_a * h_in + carry_b
        # After this block: the full transformation from input h[-1] to output h[block_end]
        # is (scan_a[-1], scan_b[-1]) applied to the carry transformation
        # New carry = (carry_a, carry_b) ⊕ (scan_a[-1], scan_b[-1])
        #           = (scan_a[-1] * carry_a, scan_a[-1] * carry_b + scan_b[-1])
        # But carry_a is not used, we only track carry_b (the hidden state)

        # Actually simpler: carry_b is just h[block_end] = h_vals[-1]
        # And carry_a for next block multiplication is 1 (identity, since scan_a already includes it)

        # Get the last valid value
        last_idx = block_size - 1
        carry_b = tl.sum(tl.where(offs == last_idx, h_vals, 0.0))


@triton.jit
def dyadic_scan_bwd_parallel_kernel(
    grad_h_ptr, h_ptr, nums_ptr, shifts_ptr,
    grad_u_ptr, grad_nums_ptr,
    stride_b, stride_l, stride_d,
    B: tl.constexpr,
    L: tl.constexpr,
    D: tl.constexpr,
    BLOCK_L: tl.constexpr,
):
    """Backward pass using parallel scan (reversed direction).

    For recurrence h[t] = a[t] * h[t-1] + u[t], the backward pass is:
    d_u[t] = d_h_acc[t]
    d_h_acc[t-1] = a[t] * d_h_acc[t]  (gradient flows backward through decay)

    This is also a linear recurrence, just reversed!
    """
    pid_b = tl.program_id(0)
    pid_d = tl.program_id(1)

    base_offset = pid_b * stride_b + pid_d * stride_d

    # Carry for gradient accumulation across blocks
    carry_grad = 0.0

    # Process blocks in reverse order
    num_blocks = (L + BLOCK_L - 1) // BLOCK_L

    # Initialize carry_h_prev: for the first iteration (last block in forward order),
    # we need h[last_block_start - 1] where last_block_start = (num_blocks-1) * BLOCK_L
    first_block_start = (num_blocks - 1) * BLOCK_L
    if first_block_start > 0:
        carry_h_prev = tl.load(h_ptr + base_offset + (first_block_start - 1) * stride_l)
    else:
        carry_h_prev = 0.0

    for block_idx in range(num_blocks - 1, -1, -1):
        block_start = block_idx * BLOCK_L
        block_size = tl.minimum(BLOCK_L, L - block_start)

        # Indices for this block (in original order, we'll reverse in computation)
        offs = tl.arange(0, BLOCK_L)
        mask = offs < block_size

        # Load data for this block
        ptr_offs = base_offset + (block_start + offs) * stride_l
        grad_h_vals = tl.load(grad_h_ptr + ptr_offs, mask=mask, other=0.0)
        num_vals = tl.load(nums_ptr + ptr_offs, mask=mask, other=0)
        shift_vals = tl.load(shifts_ptr + ptr_offs, mask=mask, other=0)

        # Compute decay
        decay_scale = tl.exp2(-shift_vals.to(tl.float32))
        decay = num_vals.to(tl.float32) * decay_scale

        # Backward scan: d_h_acc[t] = grad_h[t] + decay[t+1] * d_h_acc[t+1]
        # In reversed order, this becomes:
        # d_h_acc[rev_t] = grad_h[rev_t] + decay[rev_t] * d_h_acc[rev_t-1] (where rev_t = L-1-t)
        #
        # To use forward parallel scan, we reverse the arrays:
        # reversed_grad_h, reversed_decay
        # Then scan, then reverse back

        # Reverse within block for parallel scan
        rev_offs = block_size - 1 - offs
        rev_mask = offs < block_size

        # Gather reversed values
        rev_grad_h = tl.load(grad_h_ptr + base_offset + (block_start + rev_offs) * stride_l, mask=rev_mask, other=0.0)
        rev_num = tl.load(nums_ptr + base_offset + (block_start + rev_offs) * stride_l, mask=rev_mask, other=0)
        rev_shift = tl.load(shifts_ptr + base_offset + (block_start + rev_offs) * stride_l, mask=rev_mask, other=0)
        rev_decay = rev_num.to(tl.float32) * tl.exp2(-rev_shift.to(tl.float32))

        # Run parallel scan on reversed data
        # Note: For backward, we want d_h_acc[t] = sum of future gradients weighted by cumulative decay
        # The combiner is the same: (a1, b1) ⊕ (a2, b2) = (a2*a1, a2*b1 + b2)
        scan_a, scan_b = tl.associative_scan((rev_decay, rev_grad_h), axis=0, combine_fn=combine_fn)

        # Add carry from previous (later in forward order) block
        d_h_acc_rev = scan_a * carry_grad + scan_b

        # Reverse back to original order by scattering
        # d_h_acc[t] = d_h_acc_rev[block_size - 1 - t]
        # Store grad_u = d_h_acc
        tl.store(grad_u_ptr + base_offset + (block_start + rev_offs) * stride_l, d_h_acc_rev, mask=rev_mask)

        # Compute grad_nums: g_num[t] = d_h_acc[t] * h[t-1] * decay_scale[t]
        # For forward position (block_start + rev_offs), we need h[block_start + rev_offs - 1]
        rev_decay_scale = tl.exp2(-rev_shift.to(tl.float32))

        # Load h[t-1] for each position
        # For rev_offs > 0: h[block_start + rev_offs - 1] is within the block
        # For rev_offs == 0: h[block_start - 1] comes from carry_h_prev (previous block)
        rev_h_prev = tl.load(h_ptr + base_offset + (block_start + rev_offs - 1) * stride_l,
                             mask=(rev_mask & (rev_offs > 0)), other=0.0)
        # For rev_offs == 0, use carry_h_prev which holds h[block_start - 1]
        rev_h_prev = tl.where(rev_offs == 0, carry_h_prev, rev_h_prev)

        g_nums_rev = d_h_acc_rev * rev_h_prev * rev_decay_scale
        tl.store(grad_nums_ptr + base_offset + (block_start + rev_offs) * stride_l, g_nums_rev, mask=rev_mask)

        # Update carry for next block (earlier in forward order)
        # carry_grad = d_h_acc at position 0 (which is at the end of reversed array)
        last_idx = block_size - 1
        carry_grad = tl.sum(tl.where(offs == last_idx, d_h_acc_rev, 0.0))

        # Update h_prev carry for the NEXT iteration (earlier block in forward order)
        # We need h[block_start - 1] for the next block's rev_offs==0 case
        # But we load it NOW because next iteration won't have access to current block_start
        if block_start > 0:
            carry_h_prev = tl.load(h_ptr + base_offset + (block_start - 1) * stride_l)
        else:
            carry_h_prev = 0.0


def dyadic_scan_triton(u, nums, shifts, scale_bits=15):
    B, L, D = u.shape
    h = torch.empty_like(u)

    # Choose block size based on sequence length
    # Larger blocks = more parallelism within block, fewer cross-block carries
    if L <= 64:
        BLOCK_L = 64
    elif L <= 128:
        BLOCK_L = 128
    elif L <= 256:
        BLOCK_L = 256
    else:
        BLOCK_L = 512  # Max block size for good occupancy

    grid = (B, D)
    dyadic_scan_parallel_kernel[grid](
        u, nums, shifts, h,
        u.stride(0), u.stride(1), u.stride(2),
        B, L, D,
        BLOCK_L,
    )
    return h


def dyadic_scan_backward_triton(grad_h, h, nums, shifts, scale_bits=15):
    B, L, D = grad_h.shape
    grad_u = torch.empty_like(grad_h)
    grad_nums = torch.empty_like(grad_h)

    if L <= 64:
        BLOCK_L = 64
    elif L <= 128:
        BLOCK_L = 128
    elif L <= 256:
        BLOCK_L = 256
    else:
        BLOCK_L = 512

    grid = (B, D)
    dyadic_scan_bwd_parallel_kernel[grid](
        grad_h, h, nums, shifts,
        grad_u, grad_nums,
        grad_h.stride(0), grad_h.stride(1), grad_h.stride(2),
        B, L, D,
        BLOCK_L,
    )
    return grad_u, grad_nums
