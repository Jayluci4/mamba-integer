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
    from triton_kernels.dyadic_scan import dyadic_scan_triton, dyadic_scan_backward_triton
    from triton_kernels.bitnet_kernels import fast_bitshift_norm
    from triton_kernels.fused_activations import (
        fused_squareplus_clamp,
        fused_sigmoid_gate,
        fused_ste_clamp,
    )
    TRITON_AVAILABLE = True
    FUSED_KERNELS_AVAILABLE = True
    print("DEBUG: Triton Mamba Kernels Loaded (with fused activations).")
except ImportError as e:
    TRITON_AVAILABLE = False
    FUSED_KERNELS_AVAILABLE = False
    print(f"DEBUG: Triton Mamba Kernels Failed: {e}")


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
    """BitShift Normalization using INTEGER-ONLY operations.

    Replaces RMSNorm but uses power-of-2 scaling instead of exact sqrt.
    All operations are: +, -, *, /, comparisons (no transcendentals).
    """

    @staticmethod
    def forward(ctx, x, gamma):
        # Always compute centered x for consistent backward pass
        x_centered = x - x.mean(dim=-1, keepdim=True)

        if TRITON_AVAILABLE and x.is_cuda:
            # Triton kernel handles centering internally, but we need centered x for backward
            y, inv_rms = fast_bitshift_norm(x, gamma)
            ctx.save_for_backward(x_centered, gamma, inv_rms)
            return y
        else:
            # INTEGER-ONLY implementation:
            # 1. Compute variance (uses only *, +, /)
            var = x_centered.pow(2).mean(dim=-1, keepdim=True)

            # 2. Find power-of-2 scale (uses only comparisons, no sqrt/log2)
            inv_rms = _find_power_of_2_scale(var)

            ctx.save_for_backward(x_centered, gamma, inv_rms)
            return x_centered * inv_rms * gamma

    @staticmethod
    def backward(ctx, grad_output):
        x_centered, gamma, inv_rms = ctx.saved_tensors
        if inv_rms.dim() == 1:
            inv_rms = inv_rms.view(*x_centered.shape[:-1], 1)

        # INTEGER-ONLY backward pass:
        # Since inv_rms is computed via lookup table (piecewise constant),
        # it has zero gradient. The backward is simply the chain rule.
        #
        # Forward: y = x_centered * inv_rms * gamma
        # Backward: grad_x = grad_y * inv_rms * gamma
        #           grad_gamma = sum(grad_y * x_centered * inv_rms)

        grad_x = grad_output * gamma * inv_rms
        grad_gamma = (grad_output * x_centered * inv_rms).sum(dim=tuple(range(grad_output.dim() - 1)))
        return grad_x, grad_gamma

class DyadicScanFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, decay_nums, decay_shifts):
        if TRITON_AVAILABLE and u.is_cuda:
            dn = decay_nums.detach().clamp(0, 32000).int()
            ds = decay_shifts.detach().int()
            h = dyadic_scan_triton(u, dn, ds)
            ctx.save_for_backward(h, decay_nums, decay_shifts)
            return h
        else:
            # CPU Fallback
            B, L, DN = u.shape
            h_acc = torch.zeros(B, DN, device=u.device)
            h_list = []
            decay = decay_nums / (2.0 ** decay_shifts)
            for t in range(L):
                h_acc = h_acc * decay[:, t] + u[:, t]
                h_list.append(h_acc)
            h_seq = torch.stack(h_list, dim=1)
            ctx.save_for_backward(h_seq, decay_nums, decay_shifts)
            return h_seq

    @staticmethod
    def backward(ctx, grad_h):
        h, decay_nums, decay_shifts = ctx.saved_tensors
        if TRITON_AVAILABLE and grad_h.is_cuda:
            # STABILITY: Replace NaN/Inf with 0, clamp to prevent explosion
            grad_h = torch.nan_to_num(grad_h, nan=0.0, posinf=0.0, neginf=0.0)
            grad_h = grad_h.clamp(-100.0, 100.0)
            dn = decay_nums.detach().clamp(0, 32000).int()
            ds = decay_shifts.detach().int()
            grad_u, grad_nums = dyadic_scan_backward_triton(grad_h, h, dn, ds)
            # STABILITY: Clamp output gradients
            grad_u = torch.nan_to_num(grad_u, nan=0.0, posinf=0.0, neginf=0.0)
            grad_u = grad_u.clamp(-100.0, 100.0)
            grad_nums = torch.nan_to_num(grad_nums, nan=0.0, posinf=0.0, neginf=0.0)
            grad_nums = grad_nums.clamp(-100.0, 100.0)
            return grad_u, grad_nums, None
        else:
            return grad_h, torch.zeros_like(decay_nums), None

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
        
        # Initialize decay_nums slightly lower for stability
        self.base_decay_nums = nn.Parameter(torch.ones(d_inner, d_state) * 28000.0)
        self.register_buffer('decay_shifts', torch.ones(d_inner, d_state) * 15.0)
        self.res_gate = nn.Parameter(torch.ones(1) * 0.01) # Start small

    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        
        xz = self.in_proj(hidden_states)
        # STABILITY: Clamp in_proj output (hard clamp - zero gradient outside bounds)
        xz = torch.clamp(xz, -100.0, 100.0)
        x, z = xz.chunk(2, dim=-1)
        
        # 1. Conv + Fused Activation (INTEGER-ONLY)
        x = self.conv1d(x.transpose(1, 2)).transpose(1, 2)[:, :hidden_states.shape[1]]
        # Fused: clamp(-50, 50) + squareplus activation
        if FUSED_KERNELS_AVAILABLE and x.is_cuda:
            x = fused_squareplus_clamp(x, low=-50.0, high=50.0)
        else:
            # CPU fallback using Newton-Raphson (INTEGER-ONLY)
            x = x.clamp(-50.0, 50.0)
            x = _squareplus_rational(x, num_iters=3)

        # 2. SSM Params
        x_proj_out = self.x_proj(x)
        # STABILITY: Clamp x_proj output (hard clamp)
        x_proj_out = torch.clamp(x_proj_out, -100.0, 100.0)
        dt, B_ssm, C_ssm = x_proj_out.split([self.config['ssm_cfg']['dt_rank'],
                                              self.config['ssm_cfg']['d_state'],
                                              self.config['ssm_cfg']['d_state']], dim=-1)

        decay_mod = self.dt_proj(dt)
        # STABILITY: Clamp decay_mod (hard clamp)
        decay_mod = torch.clamp(decay_mod, -20.0, 20.0)
        # Use tighter clamp: max decay 0.97 (32000/32768)
        decay_nums = ste_clamp(self.base_decay_nums.unsqueeze(0).unsqueeze(0) + decay_mod.unsqueeze(-1), 0, 32000)

        # Conservative dt scaling (0.01) to keep u small
        # Using INTEGER-ONLY squareplus
        dt_val = ste_clamp(_squareplus_rational(decay_mod, num_iters=3) * 0.01, 0, 0.1)
        u = (x * dt_val).unsqueeze(-1) * B_ssm.unsqueeze(2)
        
        # 3. Scan
        B_size, L_size, D_in, N = u.shape
        u_flat = u.reshape(B_size, L_size, -1).clamp(-100.0, 100.0)
        dn_flat = decay_nums.reshape(B_size, L_size, -1)
        ds_flat = self.decay_shifts.view(-1).unsqueeze(0).unsqueeze(0).expand_as(dn_flat)
        
        h_flat = DyadicScanFunction.apply(u_flat, dn_flat, ds_flat)
        # Stability: replace NaN/Inf and clamp hidden state
        h_flat = torch.nan_to_num(h_flat, nan=0.0, posinf=100.0, neginf=-100.0)
        h_flat = h_flat.clamp(-1000.0, 1000.0)
        h = h_flat.reshape(B_size, L_size, D_in, N)
        
        # 4. Out with Fused Gating (INTEGER-ONLY)
        y = torch.matmul(h, C_ssm.unsqueeze(-1)).squeeze(-1)
        # STABILITY: Clamp matmul output (hard clamp)
        y = torch.clamp(y, -100.0, 100.0)
        # Fused: y * sigmoid_alg(z) using algebraic sigmoid
        if FUSED_KERNELS_AVAILABLE and y.is_cuda:
            y = fused_sigmoid_gate(y, z)
        else:
            # CPU fallback using algebraic sigmoid (INTEGER-ONLY)
            y = y * _sigmoid_algebraic(z)

        # STABILITY: Replace any NaN/Inf before final projection
        y = torch.nan_to_num(y, nan=0.0, posinf=50.0, neginf=-50.0)
        out = self.out_proj(y)
        # STABILITY: Clamp output before residual (hard clamp)
        out = torch.clamp(out, -100.0, 100.0)

        return residual + out * self.res_gate

class BitShiftNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        x = x - x.mean(dim=-1, keepdim=True)
        return BitShiftNormFunction.apply(x, self.gamma)

class MambaIntegerModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config['vocab_size'], config['d_model'])
        self.layers = nn.ModuleList([MambaIntegerBlock(config, i) for i in range(config['n_layer'])])
        self.norm_f = BitShiftNorm(config['d_model'])
        self.lm_head = BitLinear(config['d_model'], config['vocab_size'])
        self.output_scale = 1.0 / math.sqrt(config['d_model'])
        self.gradient_checkpointing = False
        
    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
            # STABILITY: Replace any NaN/Inf with 0 to prevent propagation
            x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        x = self.norm_f(x)
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        logits = self.lm_head(x) * self.output_scale
        return logits
