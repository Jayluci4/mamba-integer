"""
Fused Triton kernels for MambaIntegerBlock activations.

Performance optimization: fuse multiple element-wise operations into single kernels
to reduce memory bandwidth (82% of GPU time was in element-wise ops).

Based on patterns from:
- Triton Fused Softmax Tutorial
- Liger-Kernel (LinkedIn)
- Bitdefender Fused Gated MLP
"""

import torch
import triton
import triton.language as tl


# --- Fused Squareplus + Clamp Kernel ---
# Fuses: clamp(x, low, high) -> 0.5 * (x + sqrt(x*x + 4))
# Original: 6 separate kernel launches -> 1 kernel

@triton.jit
def fused_squareplus_clamp_fwd_kernel(
    x_ptr, out_ptr,
    n_elements,
    low: tl.constexpr, high: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Forward: y = squareplus(clamp(x, low, high))"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Clamp
    x_clamped = tl.where(x < low, low, x)
    x_clamped = tl.where(x_clamped > high, high, x_clamped)

    # Squareplus: 0.5 * (x + sqrt(x^2 + 4))
    y = 0.5 * (x_clamped + tl.sqrt(x_clamped * x_clamped + 4.0))

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
    sqrt_term = tl.sqrt(x_clamped * x_clamped + 4.0)
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

        return grad_x, None, None


def fused_squareplus_clamp(x, low=-50.0, high=50.0):
    """Fused squareplus activation with input clamping.

    Computes: squareplus(clamp(x, low, high))
    where squareplus(x) = 0.5 * (x + sqrt(x^2 + 4))

    Args:
        x: Input tensor
        low: Lower clamp bound (default: -50.0)
        high: Upper clamp bound (default: 50.0)

    Returns:
        Output tensor with squareplus applied to clamped input
    """
    if not x.is_cuda:
        # CPU fallback
        x_clamped = torch.clamp(x, low, high)
        return 0.5 * (x_clamped + torch.sqrt(x_clamped * x_clamped + 4.0))
    return FusedSquareplusClampFunction.apply(x, low, high)


# --- Fused Sigmoid Gate Kernel ---
# Fuses: y * (0.5 * (z / sqrt(z^2 + 1) + 1))
# This is the sigmoid approximation used in MambaIntegerBlock
# Original: 6 separate kernel launches -> 1 kernel

@triton.jit
def fused_sigmoid_gate_fwd_kernel(
    y_ptr, z_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Forward: out = y * sigmoid_approx(z)

    sigmoid_approx(z) = 0.5 * (z / sqrt(z^2 + 1) + 1)
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    z = tl.load(z_ptr + offsets, mask=mask, other=0.0)

    # Sigmoid approximation: 0.5 * (z / sqrt(z^2 + 1) + 1)
    sqrt_term = tl.sqrt(z * z + 1.0)
    sigmoid_z = 0.5 * (z / sqrt_term + 1.0)

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

    Forward: out = y * sigmoid_approx(z)

    Backward:
        grad_y = grad_out * sigmoid_approx(z)
        grad_z = grad_out * y * d(sigmoid_approx)/dz

    where d(sigmoid_approx)/dz = 0.5 * (1 / sqrt(z^2 + 1) - z^2 / (z^2 + 1)^1.5)
                                = 0.5 / (z^2 + 1)^1.5
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    grad_out = tl.load(grad_out_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    z = tl.load(z_ptr + offsets, mask=mask, other=0.0)

    # Compute sigmoid approximation
    z_sq_plus_1 = z * z + 1.0
    sqrt_term = tl.sqrt(z_sq_plus_1)
    sigmoid_z = 0.5 * (z / sqrt_term + 1.0)

    # grad_y = grad_out * sigmoid(z)
    grad_y = grad_out * sigmoid_z

    # d(sigmoid)/dz = 0.5 / (z^2 + 1)^1.5
    d_sigmoid = 0.5 / (sqrt_term * z_sq_plus_1)  # = 0.5 / (z^2+1)^1.5

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

        return grad_y, grad_z


def fused_sigmoid_gate(y, z):
    """Fused sigmoid gating operation.

    Computes: y * sigmoid_approx(z)
    where sigmoid_approx(z) = 0.5 * (z / sqrt(z^2 + 1) + 1)

    This is a smooth approximation of sigmoid that's numerically stable.

    Args:
        y: Gate input tensor
        z: Sigmoid input tensor

    Returns:
        Gated output: y * sigmoid_approx(z)
    """
    if not y.is_cuda:
        # CPU fallback
        sigmoid_z = 0.5 * (z / torch.sqrt(z * z + 1.0) + 1.0)
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


# --- Convenience function to check if fused kernels are available ---

def is_fused_available():
    """Check if fused Triton kernels are available."""
    return torch.cuda.is_available()
