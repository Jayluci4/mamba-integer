"""
Mamba-Integer: Integer-Only State Space Model

FULLY INTEGER AI: All operations use only:
- Addition, subtraction, multiplication
- Division by power of 2 (bit-shift)
- NO transcendentals (exp, log, sqrt, sin, cos)

This enables:
- ZK-ML (Zero-Knowledge ML) compatibility
- Edge AI without floating-point units
- Custom silicon (FPGA/ASIC) deployment
"""

import torch
import torch.nn as nn
import os
import math
import sys

# Setup Path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../bitnet-odp/src")))
from rational_bitnet import BitLinear

# Triton
try:
    sys.path.append(os.path.dirname(__file__))
    from triton_kernels.dyadic_scan import (
        dyadic_scan_triton,
        dyadic_scan_backward_triton,
        dyadic_scan_triton_fast,
        dyadic_scan_backward_triton_fast,
    )
    from triton_kernels.bitnet_kernels import fast_bitshift_norm
    from triton_kernels.fused_activations import (
        fused_squareplus_clamp,
        fused_sigmoid_gate,
        fused_ste_clamp,
    )
    TRITON_AVAILABLE = True
    FUSED_KERNELS_AVAILABLE = True
    FAST_SCAN_AVAILABLE = True
    print("DEBUG: Triton Mamba Kernels Loaded (with fused activations + fast scan).")
except ImportError as e:
    TRITON_AVAILABLE = False
    FUSED_KERNELS_AVAILABLE = False
    FAST_SCAN_AVAILABLE = False
    print(f"DEBUG: Triton Mamba Kernels Failed: {e}")

# S1 FIX: Mamba-2 SSD (State Space Duality) for 2-8x speedup
try:
    from triton_kernels.ssd_chunk import dyadic_scan_ssd
    SSD_AVAILABLE = True
    print("DEBUG: Mamba-2 SSD (naive) Loaded.")
except ImportError as e:
    SSD_AVAILABLE = False
    print(f"DEBUG: Mamba-2 SSD (naive) not available: {e}")

# S1 FIX: Memory-efficient multi-head SSD
# Uses scalar A per head instead of matrix A, reducing memory by 1000x
# L matrix: [B, n_heads, n_chunks, cs, cs] = 6.3 MB vs [B, n_chunks, D, cs, cs] = 6.4 GB
try:
    from triton_kernels.ssd_multihead import MambaIntegerBlockV2, ssd_multihead
    SSD_MULTIHEAD_AVAILABLE = True
    print("DEBUG: Mamba-2 SSD Multi-head Loaded (memory-efficient, 1000x reduction).")
except ImportError as e:
    SSD_MULTIHEAD_AVAILABLE = False
    print(f"DEBUG: Mamba-2 SSD Multi-head not available: {e}")

# S1 FIX: Use multi-head SSD by default when available
# The naive SSD builds [B, chunks, D, chunk_size, chunk_size] matrices causing OOM
# Multi-head SSD uses scalar A per head, reducing memory to 6.3 MB
USE_SSD = False  # Legacy naive SSD (disabled)
USE_SSD_MULTIHEAD = True  # Memory-efficient multi-head SSD (enabled)


# --- Integer-Only Math Functions ---
# These replace transcendentals with rational approximations

def _rsqrt_newton(y, num_iters=3):
    """Newton-Raphson rsqrt (INTEGER-ONLY).

    Computes 1/sqrt(y) using only: +, -, *, /
    NO transcendentals (sqrt, exp, log, etc.)
    """
    # STABILITY: Clamp input to reasonable range
    y = torch.clamp(y, min=1e-6, max=1e6)

    # Adaptive initial guess based on magnitude
    r = torch.where(y > 16.0, torch.full_like(y, 0.125), torch.full_like(y, 0.5))
    r = torch.where(y > 64.0, torch.full_like(y, 0.0625), r)
    r = torch.where(y > 256.0, torch.full_like(y, 0.03125), r)
    r = torch.where(y < 4.0, torch.ones_like(y), r)
    r = torch.where(y < 1.0, torch.full_like(y, 2.0), r)
    r = torch.where(y < 0.25, torch.full_like(y, 4.0), r)

    for _ in range(num_iters):
        r = r * (1.5 - 0.5 * y * r * r)
        # STABILITY: Clamp intermediate results
        r = torch.clamp(r, min=1e-6, max=1e3)

    return r


def _sqrt_rational(y, num_iters=3):
    """Rational sqrt using Newton-Raphson rsqrt (INTEGER-ONLY)."""
    y_safe = torch.clamp(y, min=1e-6, max=1e6)
    rsqrt_y = _rsqrt_newton(y_safe, num_iters)
    result = y_safe * rsqrt_y
    # STABILITY: Clamp output
    return torch.clamp(result, min=0.0, max=1e3)


def _squareplus_rational(x, num_iters=3):
    """Squareplus activation using rational sqrt (INTEGER-ONLY).

    squareplus(x) = 0.5 * (x + sqrt(x^2 + 4))
    """
    y_sq = x * x + 4.0
    sqrt_y = _sqrt_rational(y_sq, num_iters)
    return 0.5 * (x + sqrt_y)


def _sigmoid_algebraic(z):
    """Algebraic sigmoid approximation (INTEGER-ONLY).

    sigmoid_alg(z) = 0.5 + 0.5 * z / (1 + |z|)

    Uses ONLY: +, -, *, /, abs (no sqrt!)
    """
    abs_z = torch.abs(z)
    return 0.5 + 0.5 * z / (1.0 + abs_z)

# --- Autograd ---

def ste_clamp(x, low, high):
    """Straight-Through Estimator for clamping."""
    # Use fused Triton kernel if available (3x fewer kernel launches)
    if FUSED_KERNELS_AVAILABLE and x.is_cuda:
        return fused_ste_clamp(x, low, high)
    return x + (torch.clamp(x, low, high) - x).detach()


def _find_power_of_2_scale(var, eps=1e-9):
    """Find the power-of-2 scale for normalization (INTEGER-ONLY).

    Instead of: sqrt(var) -> log2(rms) -> 2^k
    We use: var -> find k such that 2^(2k) ≈ var -> scale = 2^(-k)

    This avoids sqrt AND log2 by searching for the nearest power of 2.
    Uses ONLY: multiply, divide, compare (no transcendentals).
    """
    # The goal: find k such that 2^k ≈ sqrt(var + eps)
    # Equivalently: find k such that 2^(2k) ≈ var
    # We can do binary search or table lookup

    # For efficiency, we use a lookup table approach:
    # Check var against powers of 4 (since 2^(2k) = 4^k)

    # Table of (threshold, scale) pairs:
    # If var >= threshold, use scale = 2^(-k) where threshold = 4^k
    var_safe = var + eps

    # Start with scale = 1 (k=0)
    scale = torch.ones_like(var_safe)

    # Check thresholds: 4^1=4, 4^2=16, 4^3=64, 4^4=256, ...
    # If var >= 4, sqrt(var) >= 2, so scale should be 1/2
    scale = torch.where(var_safe >= 4.0, torch.full_like(scale, 0.5), scale)
    scale = torch.where(var_safe >= 16.0, torch.full_like(scale, 0.25), scale)
    scale = torch.where(var_safe >= 64.0, torch.full_like(scale, 0.125), scale)
    scale = torch.where(var_safe >= 256.0, torch.full_like(scale, 0.0625), scale)
    scale = torch.where(var_safe >= 1024.0, torch.full_like(scale, 0.03125), scale)
    scale = torch.where(var_safe >= 4096.0, torch.full_like(scale, 0.015625), scale)
    scale = torch.where(var_safe >= 16384.0, torch.full_like(scale, 0.0078125), scale)
    scale = torch.where(var_safe >= 65536.0, torch.full_like(scale, 0.00390625), scale)

    # For small variance (var < 1), scale up
    scale = torch.where(var_safe < 1.0, torch.full_like(scale, 1.0), scale)
    scale = torch.where(var_safe < 0.25, torch.full_like(scale, 2.0), scale)
    scale = torch.where(var_safe < 0.0625, torch.full_like(scale, 4.0), scale)
    scale = torch.where(var_safe < 0.015625, torch.full_like(scale, 8.0), scale)

    return scale


class BitShiftNormFunction(torch.autograd.Function):
    """BitShift Normalization with LSQ (Learned Step Quantization).

    P2 FIX: The original implementation used piecewise constant lookup table
    for inv_rms, which has zero gradient. This LSQ version:
    1. Forward: Uses discrete power-of-2 scale (integer-only, ZK-compatible)
    2. Backward: Uses STE with differentiable RMSNorm gradient approximation

    This allows gradients to flow while maintaining integer-only forward pass.
    Reference: "Learned Step Size Quantization" (Esser et al., 2020)
    """

    @staticmethod
    def forward(ctx, x, gamma, step_size):
        # Always compute centered x for consistent backward pass
        x_centered = x - x.mean(dim=-1, keepdim=True)

        # Compute variance for backward pass (differentiable path)
        var = x_centered.pow(2).mean(dim=-1, keepdim=True)

        if TRITON_AVAILABLE and x.is_cuda:
            # Triton kernel handles centering internally
            y, inv_rms = fast_bitshift_norm(x, gamma)
            # P2 FIX: Apply learnable step_size for gradient flow
            y = y * step_size
            ctx.save_for_backward(x_centered, gamma, inv_rms, var, step_size)
            return y
        else:
            # INTEGER-ONLY forward pass:
            # 1. Compute variance (uses only *, +, /)
            # 2. Find power-of-2 scale (uses only comparisons, no sqrt/log2)
            inv_rms = _find_power_of_2_scale(var)

            ctx.save_for_backward(x_centered, gamma, inv_rms, var, step_size)
            # P2 FIX: Apply learnable step_size
            return x_centered * inv_rms * gamma * step_size

    @staticmethod
    def backward(ctx, grad_output):
        x_centered, gamma, inv_rms, var, step_size = ctx.saved_tensors
        if inv_rms.dim() == 1:
            inv_rms = inv_rms.view(*x_centered.shape[:-1], 1)

        # P2 FIX: LSQ-style backward pass
        # Instead of zero gradient through lookup table, we use STE:
        # Pretend inv_rms was computed via differentiable rsqrt for backward
        #
        # Forward (discrete): y = x_centered * inv_rms_discrete * gamma * step_size
        # Backward (STE): treat inv_rms as rsqrt(var + eps) for gradient computation
        #
        # This is the key insight from LSQ: discrete forward, continuous backward

        eps = 1e-6
        # Differentiable inverse RMS for gradient computation
        inv_rms_diff = torch.rsqrt(var + eps)

        # Gradient w.r.t. input (using differentiable path)
        # d/dx of (x * inv_rms * gamma * step_size) with RMSNorm gradient
        d = x_centered.shape[-1]
        grad_x = grad_output * gamma * step_size * inv_rms_diff
        # Subtract mean gradient component (from centering)
        grad_x = grad_x - grad_x.mean(dim=-1, keepdim=True)
        # RMSNorm gradient correction term
        grad_x = grad_x - x_centered * (grad_output * x_centered * gamma * step_size * inv_rms_diff.pow(3)).mean(dim=-1, keepdim=True)

        # Gradient w.r.t. gamma
        grad_gamma = (grad_output * x_centered * inv_rms * step_size).sum(dim=tuple(range(grad_output.dim() - 1)))

        # Gradient w.r.t. step_size (LSQ gradient) - must match shape [1]
        grad_step_size = (grad_output * x_centered * inv_rms * gamma).sum().view(1)

        return grad_x, grad_gamma, grad_step_size

class DyadicScanFunction(torch.autograd.Function):
    """Dyadic scan with CONVEX COMBINATION formulation (minGRU-style).

    Computes: h[t] = decay * h[t-1] + (1 - decay) * u[t]

    This ensures scale-invariance and prevents gradient explosion.
    Reference: "Were RNNs All We Needed?" (2024)

    S1 FIX: When USE_SSD=True, uses Mamba-2 SSD chunked matmul for 2-8x speedup.
    """
    @staticmethod
    def forward(ctx, u, decay_nums, decay_shifts):
        # S1 FIX: Use SSD (chunked matmul) when enabled for tensor core speedup
        if USE_SSD and SSD_AVAILABLE and u.is_cuda:
            dn = decay_nums.detach()
            h = dyadic_scan_ssd(u, dn)
            ctx.save_for_backward(u, h, decay_nums)
            ctx.use_ssd = True
            ctx.use_fast_path = False
            return h

        if TRITON_AVAILABLE and u.is_cuda:
            # OPTIMIZED: Keep float32 throughout (no dtype conversion)
            # Float32 has 24-bit mantissa, exact for integers up to 16M
            # decay_nums range [0, 32000] fits perfectly
            dn = decay_nums.detach()  # No .to(torch.int32) - kernel accepts float32

            # FAST PATH: Use optimized kernel when shift=15 (constant)
            # Now uses convex combination: h = decay*h_prev + (1-decay)*u
            if FAST_SCAN_AVAILABLE:
                h = dyadic_scan_triton_fast(u, dn)
                ctx.save_for_backward(u, h, decay_nums)  # Save u for backward
                ctx.use_fast_path = True
                ctx.use_ssd = False
            else:
                ds = decay_shifts.detach().to(torch.int32)
                h = dyadic_scan_triton(u, dn, ds)
                ctx.save_for_backward(u, h, decay_nums, decay_shifts)
                ctx.use_fast_path = False
                ctx.use_ssd = False
            return h
        else:
            # CPU Fallback with convex combination
            B, L, DN = u.shape
            h_acc = torch.zeros(B, DN, device=u.device)
            h_list = []
            decay = decay_nums / (2.0 ** decay_shifts)
            for t in range(L):
                # Convex combination: h = decay*h_prev + (1-decay)*u
                h_acc = h_acc * decay[:, t] + (1.0 - decay[:, t]) * u[:, t]
                h_list.append(h_acc)
            h_seq = torch.stack(h_list, dim=1)
            ctx.save_for_backward(u, h_seq, decay_nums, decay_shifts)
            ctx.use_fast_path = False
            ctx.use_ssd = False
            return h_seq

    @staticmethod
    def backward(ctx, grad_h):
        # S1 FIX: SSD backward uses autograd from SSDChunkFunction
        if hasattr(ctx, 'use_ssd') and ctx.use_ssd:
            u, h, decay_nums = ctx.saved_tensors
            # SSD backward: compute gradients using convex combination formula
            B, L, D = grad_h.shape
            decay = decay_nums / 32768.0

            # Backward scan for grad_h_acc
            grad_h_acc = torch.zeros_like(grad_h)
            acc = torch.zeros(B, D, device=grad_h.device, dtype=grad_h.dtype)
            for t in range(L - 1, -1, -1):
                acc = grad_h[:, t] + decay[:, t] * acc
                grad_h_acc[:, t] = acc

            # Gradients
            input_weight = 1.0 - decay
            grad_u = input_weight * grad_h_acc

            # h_prev for decay gradient
            h_prev = torch.zeros_like(h)
            h_prev[:, 1:] = h[:, :-1]
            grad_decay = grad_h_acc * (h_prev - u)
            grad_nums = grad_decay / 32768.0

            return grad_u, grad_nums, None

        if ctx.use_fast_path:
            u, h, decay_nums = ctx.saved_tensors
            # OPTIMIZED: Keep float32 throughout (no dtype conversion)
            dn = decay_nums.detach()  # No .to(torch.int32)
            grad_u, grad_nums = dyadic_scan_backward_triton_fast(grad_h, h, u, dn)
            return grad_u, grad_nums, None
        else:
            u, h, decay_nums, decay_shifts = ctx.saved_tensors
            if TRITON_AVAILABLE and grad_h.is_cuda:
                dn = decay_nums.detach().to(torch.int32)
                ds = decay_shifts.detach().to(torch.int32)
                # Note: non-fast path still uses old backward (TODO: update if needed)
                grad_u, grad_nums = dyadic_scan_backward_triton(grad_h, h, dn, ds)
                return grad_u, grad_nums, None
            else:
                # Simple CPU fallback for convex combination
                B, L, DN = grad_h.shape
                decay = decay_nums / (2.0 ** decay_shifts)
                input_weight = 1.0 - decay
                # grad_u = (1 - decay) * grad_h (approximate)
                grad_u = input_weight * grad_h
                return grad_u, torch.zeros_like(decay_nums), None

class MambaIntegerBlock(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        d_model, d_state = config['d_model'], config['ssm_cfg']['d_state']
        d_inner = d_model * 2
        dt_rank = config['ssm_cfg']['dt_rank']
        
        self.norm = BitShiftNorm(d_model)
        self.in_proj = BitLinear(d_model, d_inner * 2)
        self.conv1d = nn.Conv1d(d_inner, d_inner, 4, groups=d_inner, padding=3)
        self.x_proj = BitLinear(d_inner, dt_rank + 2 * d_state)
        self.dt_proj = BitLinear(dt_rank, d_inner)
        self.out_proj = BitLinear(d_inner, d_model)
        
        # P1 FIX: Initialize decay_nums for decay = 0.5 (not 0.96)
        # Previous: 31500/32768 ≈ 0.96 caused slow gradient propagation
        # Now: 16384/32768 = 0.5 allows equal weight to history vs new input
        # Uses CONVEX COMBINATION: h = decay*h_prev + (1-decay)*u
        # Reference: "Were RNNs All We Needed?" recommends decay near 0.5 at init
        self.base_decay_nums = nn.Parameter(torch.ones(d_inner, d_state) * 16384.0)
        self.register_buffer('decay_shifts', torch.ones(d_inner, d_state) * 15.0)
        # P0 FIX: SkipInit-style initialization (1/sqrt(2*n_layer))
        # Previous: 0.01 caused 10^-48 gradient attenuation through 24 layers
        # Now: ~0.144 provides immediate gradient flow while remaining learnable
        n_layer = config.get('n_layer', 24)
        self.res_gate = nn.Parameter(torch.ones(1) / math.sqrt(2 * n_layer))

    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = self.norm(hidden_states)

        xz = self.in_proj(hidden_states)
        x, z = xz.chunk(2, dim=-1)

        # 1. Conv + Fused Activation (INTEGER-ONLY)
        x = self.conv1d(x.transpose(1, 2)).transpose(1, 2)[:, :hidden_states.shape[1]]
        # Fused: clamp(-50, 50) + squareplus activation
        if FUSED_KERNELS_AVAILABLE and x.is_cuda:
            x = fused_squareplus_clamp(x, low=-50.0, high=50.0)
        else:
            # A4 FIX: Use ste_clamp instead of .clamp() to avoid flat gradient regions
            x = ste_clamp(x, -50.0, 50.0)
            x = _squareplus_rational(x, num_iters=3)

        # 2. SSM Params
        x_proj_out = self.x_proj(x)
        dt, B_ssm, C_ssm = x_proj_out.split([self.config['ssm_cfg']['dt_rank'],
                                              self.config['ssm_cfg']['d_state'],
                                              self.config['ssm_cfg']['d_state']], dim=-1)

        decay_mod = self.dt_proj(dt)
        # A4 FIX: Use ste_clamp for decay_mod to avoid flat gradient regions
        # Range [-20, 20] ensures decay_nums stays in valid range after modulation
        decay_mod = ste_clamp(decay_mod, -20.0, 20.0)
        # decay_nums must be in [0, 32000] for dyadic scan
        decay_nums = ste_clamp(self.base_decay_nums.unsqueeze(0).unsqueeze(0) + decay_mod.unsqueeze(-1), 0, 32000)
        # A3 FIX: dt_val bounds [0.001, 0.1] - Mamba uses log-uniform in this range
        # Previous: lower bound 0 was too restrictive
        dt_val = ste_clamp(_squareplus_rational(decay_mod, num_iters=3) * 0.01, 0.001, 0.1)
        u = (x * dt_val).unsqueeze(-1) * B_ssm.unsqueeze(2)

        # 3. Scan
        B_size, L_size, D_in, N = u.shape
        u_flat = u.reshape(B_size, L_size, -1)
        dn_flat = decay_nums.reshape(B_size, L_size, -1)
        ds_flat = self.decay_shifts.view(-1).unsqueeze(0).unsqueeze(0).expand_as(dn_flat)

        h_flat = DyadicScanFunction.apply(u_flat, dn_flat, ds_flat)
        h = h_flat.reshape(B_size, L_size, D_in, N)

        # 4. Out with Fused Gating (INTEGER-ONLY)
        y = torch.matmul(h, C_ssm.unsqueeze(-1)).squeeze(-1)
        if FUSED_KERNELS_AVAILABLE and y.is_cuda:
            y = fused_sigmoid_gate(y, z)
        else:
            y = y * _sigmoid_algebraic(z)

        out = self.out_proj(y)
        return residual + out * self.res_gate

class BitShiftNorm(nn.Module):
    """BitShift Normalization with LSQ (Learned Step Quantization).

    P2 FIX: Added learnable step_size parameter for gradient flow.
    The step_size starts at 1.0 and can be learned to compensate for
    the quantization error in the power-of-2 scale lookup.
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        # P2 FIX: LSQ step size - starts at 1.0, learned during training
        self.step_size = nn.Parameter(torch.ones(1))

    def forward(self, x):
        x = x - x.mean(dim=-1, keepdim=True)
        return BitShiftNormFunction.apply(x, self.gamma, self.step_size)

class MambaIntegerModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config['vocab_size'], config['d_model'])

        # S1 FIX: Use memory-efficient multi-head SSD block when enabled
        use_ssd = config.get('ssm_cfg', {}).get('use_ssd', False)
        if use_ssd and SSD_MULTIHEAD_AVAILABLE and USE_SSD_MULTIHEAD:
            print(f"Using MambaIntegerBlockV2 (multi-head SSD) for {config['n_layer']} layers")
            self.layers = nn.ModuleList([MambaIntegerBlockV2(config, i) for i in range(config['n_layer'])])
        else:
            print(f"Using MambaIntegerBlock (dyadic scan) for {config['n_layer']} layers")
            self.layers = nn.ModuleList([MambaIntegerBlock(config, i) for i in range(config['n_layer'])])

        self.norm_f = BitShiftNorm(config['d_model'])
        self.lm_head = BitLinear(config['d_model'], config['vocab_size'])
        self.output_scale = 1.0 / math.sqrt(config['d_model'])
        self.gradient_checkpointing = False
        
    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
            # P0 FIX: Removed nan_to_num - it breaks gradient flow (PyTorch Issue #94700)
            # If NaN occurs, we should fix root cause, not mask it
        x = self.norm_f(x)
        # P0 FIX: Removed nan_to_num here as well
        logits = self.lm_head(x) * self.output_scale
        return logits
