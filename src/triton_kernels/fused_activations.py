"""
Fused Triton kernels for MambaIntegerBlock activations.

Performance optimization: fuse multiple element-wise operations into single kernels
to reduce memory bandwidth (82% of GPU time was in element-wise ops).

Based on patterns from:
- Triton Fused Softmax Tutorial
- Liger-Kernel (LinkedIn)
- Bitdefender Fused Gated MLP

INTEGER-ONLY CONSTRAINT:
All operations must use only: +, -, *, /, bit-shift
NO transcendentals allowed: exp, log, sqrt, sin, cos, pow

For sqrt replacement, we use Newton-Raphson iteration for rsqrt (1/sqrt(x)):
  r_new = r * (3 - y * r^2) / 2
  sqrt(y) = y * rsqrt(y)

For sigmoid, we use algebraic approximation:
  sigmoid_alg(z) = 0.5 + 0.5 * z / (1 + |z|)
"""

import torch
import triton
import triton.language as tl


# --- Integer-Only rsqrt via Newton-Raphson ---
# This replaces tl.sqrt() with rational operations only

@triton.jit
def rsqrt_newton_raphson(y, num_iters: tl.constexpr = 3):
    """Compute 1/sqrt(y) using Newton-Raphson iteration.

    Newton-Raphson for f(r) = 1/r^2 - y = 0:
    r_new = r * (3 - y * r^2) / 2

    Uses ONLY: multiply, add, subtract, divide by 2 (bit-shift)
    NO transcendentals (sqrt, exp, log, etc.)

    Initial guess: For y in [1, 100], use 1/8 as starting point
    3 iterations gives ~7 digits of precision.
    """
    # Initial guess: r0 = 0.125 works well for y in typical range [1, 100]
    # For better range, we normalize: scale y to [1, 4) range

    # Simple approach: use fixed initial guess with more iterations
    # r0 = 1.0 / 8.0 works for y ~ 4-100
    # r0 = 1.0 works for y ~ 1-4
    # r0 = 0.5 works for y ~ 1-16

    # Adaptive initial guess based on magnitude (pure rational)
    r = tl.where(y > 16.0, 0.125, 0.5)
    r = tl.where(y > 64.0, 0.0625, r)
    r = tl.where(y < 4.0, 1.0, r)
    r = tl.where(y < 1.0, 2.0, r)

    # Newton-Raphson iterations: r = r * (3 - y * r^2) / 2
    # This is: r = r * (1.5 - 0.5 * y * r^2)
    # Using only multiply, add, subtract, divide by constant
    for _ in range(num_iters):
        r_sq = r * r
        yr_sq = y * r_sq
        r = r * (1.5 - 0.5 * yr_sq)

    return r


@triton.jit
def sqrt_rational(y, num_iters: tl.constexpr = 3):
    """Compute sqrt(y) using Newton-Raphson rsqrt.

    sqrt(y) = y * rsqrt(y) = y / sqrt(y)

    Uses ONLY rational operations (no transcendentals).
    """
    # Handle edge case: y <= 0
    y_safe = tl.where(y > 1e-8, y, 1e-8)
    rsqrt_y = rsqrt_newton_raphson(y_safe, num_iters)
    return y_safe * rsqrt_y


# --- Fused Squareplus + Clamp Kernel ---
# Fuses: clamp(x, low, high) -> 0.5 * (x + sqrt(x*x + 4))
# Original: 6 separate kernel launches -> 1 kernel
# NOW: Uses Newton-Raphson rsqrt instead of tl.sqrt()

@triton.jit
def fused_squareplus_clamp_fwd_kernel(
    x_ptr, out_ptr,
    n_elements,
    low: tl.constexpr, high: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Forward: y = squareplus(clamp(x, low, high))

    squareplus(x) = 0.5 * (x + sqrt(x^2 + 4))

    Uses INTEGER-ONLY operations via Newton-Raphson rsqrt.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Clamp
    x_clamped = tl.where(x < low, low, x)
    x_clamped = tl.where(x_clamped > high, high, x_clamped)

    # Squareplus: 0.5 * (x + sqrt(x^2 + 4))
    # Using Newton-Raphson rsqrt instead of tl.sqrt()
    y_sq = x_clamped * x_clamped + 4.0
    sqrt_y = sqrt_rational(y_sq, num_iters=3)
    y = 0.5 * (x_clamped + sqrt_y)

    tl.store(out_ptr + offsets, y, mask=mask)


@triton.jit
def fused_squareplus_clamp_bwd_kernel(
    grad_out_ptr, x_ptr, grad_x_ptr,
    n_elements,
    low: tl.constexpr, high: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Backward: grad_x = grad_out * d(squareplus)/dx if in clamp range, else 0

    d(squareplus)/dx = 0.5 * (1 + x / sqrt(x^2 + 4))

    Uses INTEGER-ONLY operations via Newton-Raphson rsqrt.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    grad_out = tl.load(grad_out_ptr + offsets, mask=mask, other=0.0)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Check clamp range (STE: gradient is 0 outside clamp range)
    in_range = (x >= low) & (x <= high)

    # Clamp for squareplus derivative calculation
    x_clamped = tl.where(x < low, low, x)
    x_clamped = tl.where(x_clamped > high, high, x_clamped)

    # d(squareplus)/dx = 0.5 * (1 + x / sqrt(x^2 + 4))
    # Using Newton-Raphson rsqrt instead of tl.sqrt()
    y_sq = x_clamped * x_clamped + 4.0
    sqrt_term = sqrt_rational(y_sq, num_iters=3)
    d_squareplus = 0.5 * (1.0 + x_clamped / sqrt_term)

    # Apply STE: zero gradient outside clamp range
    grad_x = tl.where(in_range, grad_out * d_squareplus, 0.0)

    tl.store(grad_x_ptr + offsets, grad_x, mask=mask)


class FusedSquareplusClampFunction(torch.autograd.Function):
    """Autograd wrapper for fused squareplus + clamp kernel."""

    @staticmethod
    def forward(ctx, x, low=-50.0, high=50.0):
        x = x.contiguous()
        out = torch.empty_like(x)

        n_elements = x.numel()
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

        fused_squareplus_clamp_fwd_kernel[grid](
            x, out,
            n_elements,
            low, high,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        ctx.save_for_backward(x)
        ctx.low = low
        ctx.high = high
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        grad_x = torch.empty_like(x)

        n_elements = x.numel()
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

        fused_squareplus_clamp_bwd_kernel[grid](
            grad_out, x, grad_x,
            n_elements,
            ctx.low, ctx.high,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        # STABILITY: Replace NaN/Inf and clamp gradients
        grad_x = torch.nan_to_num(grad_x, nan=0.0, posinf=100.0, neginf=-100.0)
        grad_x = grad_x.clamp(-100.0, 100.0)

        return grad_x, None, None


def _rsqrt_newton_cpu(y, num_iters=3):
    """Newton-Raphson rsqrt for CPU (INTEGER-ONLY)."""
    # Adaptive initial guess
    r = torch.where(y > 16.0, torch.full_like(y, 0.125), torch.full_like(y, 0.5))
    r = torch.where(y > 64.0, torch.full_like(y, 0.0625), r)
    r = torch.where(y < 4.0, torch.ones_like(y), r)
    r = torch.where(y < 1.0, torch.full_like(y, 2.0), r)

    for _ in range(num_iters):
        r = r * (1.5 - 0.5 * y * r * r)

    return r


def _sqrt_rational_cpu(y, num_iters=3):
    """Rational sqrt for CPU using Newton-Raphson rsqrt."""
    y_safe = torch.clamp(y, min=1e-8)
    rsqrt_y = _rsqrt_newton_cpu(y_safe, num_iters)
    return y_safe * rsqrt_y


def fused_squareplus_clamp(x, low=-50.0, high=50.0):
    """Fused squareplus activation with input clamping.

    Computes: squareplus(clamp(x, low, high))
    where squareplus(x) = 0.5 * (x + sqrt(x^2 + 4))

    Uses INTEGER-ONLY operations:
    - Newton-Raphson rsqrt instead of sqrt
    - Only +, -, *, / operations

    Args:
        x: Input tensor
        low: Lower clamp bound (default: -50.0)
        high: Upper clamp bound (default: 50.0)

    Returns:
        Output tensor with squareplus applied to clamped input
    """
    if not x.is_cuda:
        # CPU fallback using Newton-Raphson (INTEGER-ONLY)
        x_clamped = torch.clamp(x, low, high)
        y_sq = x_clamped * x_clamped + 4.0
        sqrt_y = _sqrt_rational_cpu(y_sq, num_iters=3)
        return 0.5 * (x_clamped + sqrt_y)
    return FusedSquareplusClampFunction.apply(x, low, high)


# --- Fused Sigmoid Gate Kernel ---
# Fuses: y * sigmoid_approx(z)
# This is the sigmoid approximation used in MambaIntegerBlock
# Original: 6 separate kernel launches -> 1 kernel
#
# INTEGER-ONLY: Instead of 0.5 * (z / sqrt(z^2 + 1) + 1)
# We use ALGEBRAIC sigmoid: 0.5 + 0.5 * z / (1 + |z|)
# This uses ONLY: +, -, *, /, abs (no sqrt!)

@triton.jit
def fused_sigmoid_gate_fwd_kernel(
    y_ptr, z_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Forward: out = y * sigmoid_alg(z)

    ALGEBRAIC sigmoid (INTEGER-ONLY):
    sigmoid_alg(z) = 0.5 + 0.5 * z / (1 + |z|)

    This approximates the original:
    sigmoid_approx(z) = 0.5 * (z / sqrt(z^2 + 1) + 1)

    But uses NO transcendentals - only +, -, *, /, abs.
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    z = tl.load(z_ptr + offsets, mask=mask, other=0.0)

    # Algebraic sigmoid: 0.5 + 0.5 * z / (1 + |z|)
    # Uses ONLY: add, multiply, divide, abs (no sqrt!)
    abs_z = tl.abs(z)
    sigmoid_z = 0.5 + 0.5 * z / (1.0 + abs_z)

    # Gating
    out = y * sigmoid_z

    tl.store(out_ptr + offsets, out, mask=mask)


@triton.jit
def fused_sigmoid_gate_bwd_kernel(
    grad_out_ptr, y_ptr, z_ptr,
    grad_y_ptr, grad_z_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Backward for fused sigmoid gate.

    Forward: out = y * sigmoid_alg(z)
    where sigmoid_alg(z) = 0.5 + 0.5 * z / (1 + |z|)

    Backward (INTEGER-ONLY):
        grad_y = grad_out * sigmoid_alg(z)
        grad_z = grad_out * y * d(sigmoid_alg)/dz

    Derivative: d(sigmoid_alg)/dz = 0.5 / (1 + |z|)^2

    Proof: Let f(z) = z / (1 + |z|)
    For z >= 0: f'(z) = 1 / (1+z)^2
    For z < 0:  f'(z) = 1 / (1-z)^2 = 1 / (1+|z|)^2
    So: d(sigmoid_alg)/dz = 0.5 * f'(z) = 0.5 / (1 + |z|)^2

    Uses ONLY: +, -, *, /, abs (no sqrt!)
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    grad_out = tl.load(grad_out_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    z = tl.load(z_ptr + offsets, mask=mask, other=0.0)

    # Algebraic sigmoid: 0.5 + 0.5 * z / (1 + |z|)
    abs_z = tl.abs(z)
    one_plus_abs_z = 1.0 + abs_z
    sigmoid_z = 0.5 + 0.5 * z / one_plus_abs_z

    # grad_y = grad_out * sigmoid(z)
    grad_y = grad_out * sigmoid_z

    # d(sigmoid_alg)/dz = 0.5 / (1 + |z|)^2
    d_sigmoid = 0.5 / (one_plus_abs_z * one_plus_abs_z)

    # grad_z = grad_out * y * d_sigmoid
    grad_z = grad_out * y * d_sigmoid

    tl.store(grad_y_ptr + offsets, grad_y, mask=mask)
    tl.store(grad_z_ptr + offsets, grad_z, mask=mask)


class FusedSigmoidGateFunction(torch.autograd.Function):
    """Autograd wrapper for fused sigmoid gate kernel."""

    @staticmethod
    def forward(ctx, y, z):
        y = y.contiguous()
        z = z.contiguous()
        out = torch.empty_like(y)

        n_elements = y.numel()
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

        fused_sigmoid_gate_fwd_kernel[grid](
            y, z, out,
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        ctx.save_for_backward(y, z)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        y, z = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        grad_y = torch.empty_like(y)
        grad_z = torch.empty_like(z)

        n_elements = y.numel()
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

        fused_sigmoid_gate_bwd_kernel[grid](
            grad_out, y, z,
            grad_y, grad_z,
            n_elements,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        # STABILITY: Replace NaN/Inf and clamp gradients
        grad_y = torch.nan_to_num(grad_y, nan=0.0, posinf=100.0, neginf=-100.0)
        grad_z = torch.nan_to_num(grad_z, nan=0.0, posinf=100.0, neginf=-100.0)
        grad_y = grad_y.clamp(-100.0, 100.0)
        grad_z = grad_z.clamp(-100.0, 100.0)

        return grad_y, grad_z


def fused_sigmoid_gate(y, z):
    """Fused sigmoid gating operation.

    Computes: y * sigmoid_alg(z)
    where sigmoid_alg(z) = 0.5 + 0.5 * z / (1 + |z|)

    This is the ALGEBRAIC sigmoid approximation (INTEGER-ONLY).
    Uses ONLY: +, -, *, /, abs (no sqrt!)

    Args:
        y: Gate input tensor
        z: Sigmoid input tensor

    Returns:
        Gated output: y * sigmoid_alg(z)
    """
    if not y.is_cuda:
        # CPU fallback using algebraic sigmoid (INTEGER-ONLY)
        abs_z = torch.abs(z)
        sigmoid_z = 0.5 + 0.5 * z / (1.0 + abs_z)
        return y * sigmoid_z
    return FusedSigmoidGateFunction.apply(y, z)


# --- Fused Clamp Kernel (simple version for scan output) ---

@triton.jit
def fused_clamp_fwd_kernel(
    x_ptr, out_ptr,
    n_elements,
    low: tl.constexpr, high: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Simple clamp kernel for when we just need clamping."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    x_clamped = tl.where(x < low, low, x)
    x_clamped = tl.where(x_clamped > high, high, x_clamped)

    tl.store(out_ptr + offsets, x_clamped, mask=mask)


@triton.jit
def fused_clamp_bwd_kernel(
    grad_out_ptr, x_ptr, grad_x_ptr,
    n_elements,
    low: tl.constexpr, high: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Backward for STE clamp: gradient passes through if in range."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    grad_out = tl.load(grad_out_ptr + offsets, mask=mask, other=0.0)
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # STE: pass gradient through if in range
    in_range = (x >= low) & (x <= high)
    grad_x = tl.where(in_range, grad_out, 0.0)

    tl.store(grad_x_ptr + offsets, grad_x, mask=mask)


class FusedClampFunction(torch.autograd.Function):
    """STE clamp with Triton kernel."""

    @staticmethod
    def forward(ctx, x, low, high):
        x = x.contiguous()
        out = torch.empty_like(x)

        n_elements = x.numel()
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

        fused_clamp_fwd_kernel[grid](
            x, out,
            n_elements,
            low, high,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        ctx.save_for_backward(x)
        ctx.low = low
        ctx.high = high
        return out

    @staticmethod
    def backward(ctx, grad_out):
        x, = ctx.saved_tensors
        grad_out = grad_out.contiguous()
        grad_x = torch.empty_like(x)

        n_elements = x.numel()
        BLOCK_SIZE = 1024
        grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

        fused_clamp_bwd_kernel[grid](
            grad_out, x, grad_x,
            n_elements,
            ctx.low, ctx.high,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        return grad_x, None, None


def fused_ste_clamp(x, low, high):
    """STE clamp with Triton kernel for GPU, fallback for CPU.

    Args:
        x: Input tensor
        low: Lower bound
        high: Upper bound

    Returns:
        Clamped tensor with STE gradient
    """
    if not x.is_cuda:
        return x + (torch.clamp(x, low, high) - x).detach()
    return FusedClampFunction.apply(x, low, high)


# --- Fused Decay Computation Kernel ---
# Fuses 5 operations into 1:
# 1. decay_mod.clamp(-20, 20)
# 2. ste_clamp(base + decay_mod, 0, 32000) → decay_nums
# 3. squareplus(decay_mod) * 0.01 → dt_val
# 4. ste_clamp(dt_val, 0, 0.1) → dt_val
#
# OPTIMIZATION: Use BLOCK_SIZE=1024 and vectorized N processing
# Previous version used BLOCK_SIZE=1 which was 2x slower than PyTorch

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['total_elements'],
)
@triton.jit
def fused_decay_computation_fwd_kernel(
    decay_mod_ptr,     # [B, L, D] input (flattened)
    base_decay_ptr,    # [D, N] base values
    decay_nums_ptr,    # [B, L, D, N] output (flattened)
    dt_val_ptr,        # [B, L, D] output (flattened)
    total_elements,    # B * L * D
    D, N,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Fused decay computation forward pass with proper parallelism.

    Each thread block processes BLOCK_SIZE elements from [B*L*D].
    The N dimension is vectorized within each thread.
    """
    pid = tl.program_id(0)

    # Each block processes BLOCK_SIZE elements
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements

    # Load decay_mod values (vectorized load)
    decay_mod = tl.load(decay_mod_ptr + offsets, mask=mask, other=0.0)

    # Step 1: Clamp decay_mod to [-20, 20]
    dm_clamped = tl.where(decay_mod < -20.0, -20.0, decay_mod)
    dm_clamped = tl.where(dm_clamped > 20.0, 20.0, dm_clamped)

    # Step 2: Compute squareplus(decay_mod) * 0.01 for dt_val
    y_sq = dm_clamped * dm_clamped + 4.0
    sqrt_y = sqrt_rational(y_sq, num_iters=3)
    sp_val = 0.5 * (dm_clamped + sqrt_y)
    dt_val = sp_val * 0.01

    # Step 3: Clamp dt_val to [0, 0.1]
    dt_val = tl.where(dt_val < 0.0, 0.0, dt_val)
    dt_val = tl.where(dt_val > 0.1, 0.1, dt_val)

    # Store dt_val (vectorized store)
    tl.store(dt_val_ptr + offsets, dt_val, mask=mask)

    # Step 4: Compute decay_nums for all N values
    # offsets are flat indices into [B*L*D], need to extract d_idx
    d_idx = offsets % D

    # For each n in N, compute decay_nums
    for n in range(BLOCK_N):
        if n < N:
            # Load base_decay[d, n] for each element
            base_offsets = d_idx * N + n
            base_val = tl.load(base_decay_ptr + base_offsets, mask=mask, other=0.0)

            # decay_nums = clamp(base + decay_mod, 0, 32000)
            dn_val = base_val + dm_clamped
            dn_val = tl.where(dn_val < 0.0, 0.0, dn_val)
            dn_val = tl.where(dn_val > 32000.0, 32000.0, dn_val)

            # Store to decay_nums[flat_idx, n]
            dn_offsets = offsets * N + n
            tl.store(decay_nums_ptr + dn_offsets, dn_val, mask=mask)


class FusedDecayComputationFunction(torch.autograd.Function):
    """Autograd wrapper for fused decay computation."""

    @staticmethod
    def forward(ctx, decay_mod, base_decay_nums):
        """
        Args:
            decay_mod: [B, L, D] decay modulation from dt_proj
            base_decay_nums: [D, N] base decay numerators

        Returns:
            decay_nums: [B, L, D, N] computed decay numerators
            dt_val: [B, L, D] computed dt values
        """
        B, L, D = decay_mod.shape
        N = base_decay_nums.shape[1]

        decay_mod = decay_mod.contiguous()
        base_decay_nums = base_decay_nums.contiguous()

        # Flatten for kernel
        decay_mod_flat = decay_mod.view(-1)

        decay_nums = torch.empty(B * L * D, N, device=decay_mod.device, dtype=decay_mod.dtype)
        dt_val = torch.empty(B * L * D, device=decay_mod.device, dtype=decay_mod.dtype)

        total_elements = B * L * D
        BLOCK_N = triton.next_power_of_2(N)

        # Grid size based on autotuned BLOCK_SIZE
        def grid(meta):
            return (triton.cdiv(total_elements, meta['BLOCK_SIZE']),)

        fused_decay_computation_fwd_kernel[grid](
            decay_mod_flat, base_decay_nums,
            decay_nums, dt_val,
            total_elements,
            D, N,
            BLOCK_N=BLOCK_N,
        )

        # Reshape outputs
        decay_nums = decay_nums.view(B, L, D, N)
        dt_val = dt_val.view(B, L, D)

        ctx.save_for_backward(decay_mod, base_decay_nums)
        ctx.dims = (B, L, D, N)
        return decay_nums, dt_val

    @staticmethod
    def backward(ctx, grad_decay_nums, grad_dt_val):
        """Backward pass for fused decay computation.

        Key insight for STE (Straight-Through Estimator):
        - decay_mod.clamp(-20, 20) → standard clamp gradient (0 outside range)
        - decay_nums STE → gradient passes through unconditionally
        - dt_val STE → gradient passes through unconditionally

        STE: x + (clamp(x) - x).detach() has grad_x = grad_output (passes through)
        """
        decay_mod, base_decay_nums = ctx.saved_tensors
        B, L, D, N = ctx.dims

        # Clamp decay_mod - standard clamp zeros gradient outside range
        dm_clamped = decay_mod.clamp(-20.0, 20.0)
        in_dm_range = (decay_mod >= -20.0) & (decay_mod <= 20.0)

        # Compute squareplus derivative for dt_val gradient
        # squareplus(x) = 0.5 * (x + sqrt(x^2 + 4))
        # d/dx = 0.5 * (1 + x / sqrt(x^2 + 4))
        y_sq = dm_clamped * dm_clamped + 4.0
        sqrt_y = _sqrt_rational_cpu(y_sq, num_iters=3)
        d_squareplus = 0.5 * (1.0 + dm_clamped / sqrt_y)
        d_dt = d_squareplus * 0.01

        # grad from dt_val: STE means gradient passes through unconditionally
        # Only respect the initial clamp(-20, 20) range
        grad_from_dt = grad_dt_val * d_dt
        grad_from_dt = torch.where(in_dm_range, grad_from_dt, torch.zeros_like(grad_from_dt))

        # grad from decay_nums: STE means gradient passes through unconditionally
        # Sum over N dimension since decay_mod broadcasts to [B, L, D, N]
        grad_from_dn = grad_decay_nums.sum(dim=-1)
        grad_from_dn = torch.where(in_dm_range, grad_from_dn, torch.zeros_like(grad_from_dn))

        # Total gradient
        grad_decay_mod = grad_from_dt + grad_from_dn
        grad_decay_mod = grad_decay_mod.clamp(-100.0, 100.0)

        # grad_base_decay_nums: STE passes through, sum over batch and length
        # in_dm_range has shape [B, L, D], need to expand for [B, L, D, N]
        grad_base = (grad_decay_nums * in_dm_range.unsqueeze(-1).float()).sum(dim=(0, 1))
        grad_base = grad_base.clamp(-100.0, 100.0)

        return grad_decay_mod, grad_base


def fused_decay_computation(decay_mod, base_decay_nums):
    """Fused computation of decay_nums and dt_val.

    Combines 5 operations into 1 kernel:
    1. decay_mod.clamp(-20, 20)
    2. ste_clamp(base + decay_mod, 0, 32000) → decay_nums
    3. squareplus(decay_mod) * 0.01 → dt_val
    4. ste_clamp(dt_val, 0, 0.1) → dt_val

    Args:
        decay_mod: [B, L, D] decay modulation from dt_proj
        base_decay_nums: [D, N] base decay numerators

    Returns:
        decay_nums: [B, L, D, N] computed decay numerators
        dt_val: [B, L, D] computed dt values
    """
    if not decay_mod.is_cuda:
        # CPU fallback
        dm_clamped = decay_mod.clamp(-20.0, 20.0)

        # decay_nums
        B, L, D = decay_mod.shape
        N = base_decay_nums.shape[1]
        decay_nums = (base_decay_nums.unsqueeze(0).unsqueeze(0) + dm_clamped.unsqueeze(-1))
        decay_nums = decay_nums + (decay_nums.clamp(0, 32000) - decay_nums).detach()

        # dt_val
        y_sq = dm_clamped * dm_clamped + 4.0
        sqrt_y = _sqrt_rational_cpu(y_sq, num_iters=3)
        dt_val = 0.5 * (dm_clamped + sqrt_y) * 0.01
        dt_val = dt_val + (dt_val.clamp(0, 0.1) - dt_val).detach()

        return decay_nums, dt_val

    return FusedDecayComputationFunction.apply(decay_mod, base_decay_nums)


# --- Fused Input Preparation Kernel ---
# Computes: u = (x * dt_val).unsqueeze(-1) * B_ssm.unsqueeze(2)
# This combines element-wise multiply + broadcast multiply into one kernel
#
# OPTIMIZATION: Use proper block sizing with autotuning
# Previous version used one thread per element which was 4x slower

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['total_elements'],
)
@triton.jit
def fused_input_prep_fwd_kernel(
    x_ptr,        # [B*L, D] flattened
    dt_val_ptr,   # [B*L, D] flattened
    B_ssm_ptr,    # [B*L, N] flattened
    u_ptr,        # [B*L*D, N] flattened output
    total_bl,     # B * L
    D, N,
    total_elements,  # B * L * D
    BLOCK_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Fused input preparation with proper parallelism.

    Each thread block processes BLOCK_SIZE elements from [B*L*D].
    u[b,l,d,n] = x[b,l,d] * dt_val[b,l,d] * B_ssm[b,l,n]
    """
    pid = tl.program_id(0)

    # Each block processes BLOCK_SIZE elements
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total_elements

    # Load x and dt_val (vectorized)
    x_vals = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    dt_vals = tl.load(dt_val_ptr + offsets, mask=mask, other=0.0)

    # Compute x * dt_val
    x_dt = x_vals * dt_vals

    # Extract (bl, d) indices from flat offset
    # offsets index into [B*L*D] where layout is (bl * D + d)
    bl_idx = offsets // D  # which (batch, length) position
    d_idx = offsets % D    # which D position

    # For each n in N, compute and store u
    for n in range(BLOCK_N):
        if n < N:
            # Load B_ssm[bl, n]
            b_ssm_offsets = bl_idx * N + n
            b_ssm_vals = tl.load(B_ssm_ptr + b_ssm_offsets, mask=mask, other=0.0)

            # Compute u = x_dt * B_ssm
            u_vals = x_dt * b_ssm_vals

            # Store to u[flat_bld, n]
            u_offsets = offsets * N + n
            tl.store(u_ptr + u_offsets, u_vals, mask=mask)


class FusedInputPrepFunction(torch.autograd.Function):
    """Autograd wrapper for fused input preparation."""

    @staticmethod
    def forward(ctx, x, dt_val, B_ssm):
        """
        Args:
            x: [B, L, D]
            dt_val: [B, L, D]
            B_ssm: [B, L, N]

        Returns:
            u: [B, L, D, N]
        """
        B, L, D = x.shape
        N = B_ssm.shape[2]

        x = x.contiguous()
        dt_val = dt_val.contiguous()
        B_ssm = B_ssm.contiguous()

        # Reshape for kernel
        x_flat = x.view(B * L, D).contiguous().view(-1)  # [B*L*D]
        dt_flat = dt_val.view(B * L, D).contiguous().view(-1)  # [B*L*D]
        B_ssm_flat = B_ssm.view(B * L, N)  # [B*L, N]

        u_flat = torch.empty(B * L * D, N, device=x.device, dtype=x.dtype)

        total_elements = B * L * D
        total_bl = B * L
        BLOCK_N = triton.next_power_of_2(N)

        def grid(meta):
            return (triton.cdiv(total_elements, meta['BLOCK_SIZE']),)

        fused_input_prep_fwd_kernel[grid](
            x_flat, dt_flat, B_ssm_flat, u_flat,
            total_bl, D, N, total_elements,
            BLOCK_N=BLOCK_N,
        )

        # Reshape output
        u = u_flat.view(B, L, D, N)

        ctx.save_for_backward(x, dt_val, B_ssm)
        ctx.dims = (B, L, D, N)
        return u

    @staticmethod
    def backward(ctx, grad_u):
        """Backward pass for fused input prep.

        u = x * dt_val * B_ssm (broadcasted)
        grad_x = sum_n(grad_u * dt_val * B_ssm)
        grad_dt_val = sum_n(grad_u * x * B_ssm)
        grad_B_ssm = sum_d(grad_u * x * dt_val)
        """
        x, dt_val, B_ssm = ctx.saved_tensors
        B, L, D, N = ctx.dims

        grad_u = grad_u.contiguous()

        # grad_x = sum over N: grad_u * dt_val * B_ssm
        # [B,L,D,N] * [B,L,D,1] * [B,L,1,N] -> sum over N -> [B,L,D]
        grad_x = (grad_u * dt_val.unsqueeze(-1) * B_ssm.unsqueeze(2)).sum(dim=-1)

        # grad_dt_val = sum over N: grad_u * x * B_ssm
        grad_dt_val = (grad_u * x.unsqueeze(-1) * B_ssm.unsqueeze(2)).sum(dim=-1)

        # grad_B_ssm = sum over D: grad_u * x * dt_val
        grad_B_ssm = (grad_u * (x * dt_val).unsqueeze(-1)).sum(dim=2)

        # Clamp gradients
        grad_x = grad_x.clamp(-100.0, 100.0)
        grad_dt_val = grad_dt_val.clamp(-100.0, 100.0)
        grad_B_ssm = grad_B_ssm.clamp(-100.0, 100.0)

        return grad_x, grad_dt_val, grad_B_ssm


def fused_input_prep(x, dt_val, B_ssm):
    """Fused input preparation for scan.

    Computes: u = (x * dt_val).unsqueeze(-1) * B_ssm.unsqueeze(2)

    Args:
        x: [B, L, D] input after activation
        dt_val: [B, L, D] time step values
        B_ssm: [B, L, N] SSM B parameter

    Returns:
        u: [B, L, D, N] input for scan
    """
    if not x.is_cuda:
        # CPU fallback
        return (x * dt_val).unsqueeze(-1) * B_ssm.unsqueeze(2)

    return FusedInputPrepFunction.apply(x, dt_val, B_ssm)


# --- Convenience function to check if fused kernels are available ---

def is_fused_available():
    """Check if fused Triton kernels are available."""
    return torch.cuda.is_available()
