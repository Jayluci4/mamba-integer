
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
    TRITON_AVAILABLE = True
    print("DEBUG: Triton Mamba Kernels Loaded.")
except ImportError as e:
    TRITON_AVAILABLE = False
    print(f"DEBUG: Triton Mamba Kernels Failed: {e}")

# --- Autograd ---

def ste_clamp(x, low, high):
    """Straight-Through Estimator for clamping."""
    return x + (torch.clamp(x, low, high) - x).detach()

class BitShiftNormFunction(torch.autograd.Function):
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
            var = x_centered.pow(2).mean(dim=-1, keepdim=True)
            rms = torch.sqrt(var + 1e-6)
            k = torch.round(torch.log2(rms)).int().clamp(min=0)
            inv_rms = 1.0 / (2.0 ** k.float())
            ctx.save_for_backward(x_centered, gamma, inv_rms)
            return x_centered * inv_rms * gamma

    @staticmethod
    def backward(ctx, grad_output):
        x_centered, gamma, inv_rms = ctx.saved_tensors
        if inv_rms.dim() == 1:
            inv_rms = inv_rms.view(*x_centered.shape[:-1], 1)

        grad_output_scaled = grad_output * gamma
        D = x_centered.size(-1)
        term1 = grad_output_scaled * inv_rms
        dot = (grad_output_scaled * x_centered).sum(dim=-1, keepdim=True)
        term2 = x_centered * dot * (inv_rms ** 3).clamp(-10.0, 10.0) / D
        grad_gamma = (grad_output * x_centered * inv_rms).sum(dim=(0, 1))
        return term1 - term2, grad_gamma

class DyadicScanFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, decay_nums, decay_shifts):
        if TRITON_AVAILABLE and u.is_cuda:
            # Stricter clamp for forward stability
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
            # Replace NaN with 0, use wider clamp to preserve gradient magnitude
            grad_h = torch.nan_to_num(grad_h, 0.0).clamp(-100.0, 100.0)
            dn = decay_nums.detach().clamp(0, 32000).int()
            ds = decay_shifts.detach().int()
            grad_u, grad_nums = dyadic_scan_backward_triton(grad_h, h, dn, ds)
            # Clamp output gradients to prevent explosion but preserve learning
            grad_u = torch.nan_to_num(grad_u, 0.0).clamp(-100.0, 100.0)
            grad_nums = torch.nan_to_num(grad_nums, 0.0).clamp(-100.0, 100.0)
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
        x, z = xz.chunk(2, dim=-1)
        
        # 1. Conv
        x = self.conv1d(x.transpose(1, 2)).transpose(1, 2)[:, :hidden_states.shape[1]]
        # Stability: clamp conv output
        x = x.clamp(-50.0, 50.0)
        x = 0.5 * (x + torch.sqrt(x*x + 4.0)) # Squareplus
        
        # 2. SSM Params
        dt, B_ssm, C_ssm = self.x_proj(x).split([self.config['ssm_cfg']['dt_rank'], 
                                                 self.config['ssm_cfg']['d_state'], 
                                                 self.config['ssm_cfg']['d_state']], dim=-1)
        
        decay_mod = self.dt_proj(dt)
        # Use tighter clamp: max decay 0.97 (32000/32768)
        decay_nums = ste_clamp(self.base_decay_nums.unsqueeze(0).unsqueeze(0) + decay_mod.unsqueeze(-1), 0, 32000)
        
        # Conservative dt scaling (0.01) to keep u small
        dt_val = ste_clamp(0.5 * (decay_mod + torch.sqrt(decay_mod*decay_mod + 4.0)) * 0.01, 0, 0.1)
        u = (x * dt_val).unsqueeze(-1) * B_ssm.unsqueeze(2)
        
        # 3. Scan
        B_size, L_size, D_in, N = u.shape
        u_flat = u.reshape(B_size, L_size, -1).clamp(-100.0, 100.0)
        dn_flat = decay_nums.reshape(B_size, L_size, -1)
        ds_flat = self.decay_shifts.view(-1).unsqueeze(0).unsqueeze(0).expand_as(dn_flat)
        
        h_flat = DyadicScanFunction.apply(u_flat, dn_flat, ds_flat)
        # Stability: clamp hidden state
        h_flat = h_flat.clamp(-1000.0, 1000.0)
        h = h_flat.reshape(B_size, L_size, D_in, N)
        
        # 4. Out
        y = torch.matmul(h, C_ssm.unsqueeze(-1)).squeeze(-1)
        y = y * (0.5 * (z / torch.sqrt(z*z + 1.0) + 1.0)) # Sigmoid
        
        out = self.out_proj(y)

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
        logits = self.lm_head(self.norm_f(x)) * self.output_scale
        return logits
