import torch
import torch.nn as nn
import ctypes
import os
import math
import sys

# Add path for rational_bitnet
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../bitnet-odp/src'))
# Add parent directory to path for dyadic_hippo
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from rational_bitnet import BitLinear

# --- Load Kernels ---
def load_lib(name):
    path = os.path.join(os.path.dirname(__file__), f"cuda_kernels/lib{name}.so")
    if os.path.exists(path):
        return ctypes.CDLL(path)
    return None

lib_bitshift = load_lib("bitshift_norm")
lib_square = load_lib("square_activation")
lib_scan_path = os.path.join(os.path.dirname(__file__), "cuda/libdyadic_mamba.so")
lib_scan = ctypes.CDLL(lib_scan_path) if os.path.exists(lib_scan_path) else None

# --- Autograd Functions ---

class BitShiftNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gamma, scale_bits=0):
        ctx.save_for_backward(x, gamma)
        ctx.scale_bits = scale_bits
        
        B, S, D = x.shape
        out = torch.empty_like(x)
        
        if lib_bitshift and x.is_cuda:
            lib_bitshift.launch_bitshift_norm(
                ctypes.c_void_p(x.data_ptr()),
                ctypes.c_void_p(gamma.data_ptr()),
                ctypes.c_void_p(out.data_ptr()),
                ctypes.c_int(B), ctypes.c_int(S), ctypes.c_int(D),
                ctypes.c_int(scale_bits)
            )
        else:
            # CPU Fallback
            var = x.pow(2).mean(dim=-1, keepdim=True)
            k = torch.floor(torch.log2(torch.sqrt(var + 1e-9))).int()
            k = torch.clamp(k, min=0)
            scale = 1.0 / (2.0 ** k.float())
            out = x * scale * gamma
            
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, gamma = ctx.saved_tensors
        var = x.pow(2).mean(dim=-1, keepdim=True)
        k = torch.floor(torch.log2(torch.sqrt(var + 1e-9)))
        k = torch.clamp(k, min=0)
        scale = 1.0 / (2.0 ** k)
        
        grad_x = grad_output * scale * gamma
        grad_gamma = (grad_output * x * scale).sum(dim=(0, 1))
        
        return grad_x, grad_gamma, None

class SquareActivationFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, shift_k=0):
        ctx.save_for_backward(x)
        ctx.shift_k = shift_k
        
        B, S, D = x.shape
        out = torch.empty_like(x)
        
        if lib_square and x.is_cuda:
            lib_square.launch_square_activation(
                ctypes.c_void_p(x.data_ptr()),
                ctypes.c_void_p(out.data_ptr()),
                ctypes.c_int(B), ctypes.c_int(S), ctypes.c_int(D),
                ctypes.c_int(shift_k)
            )
        else:
            scale = 1.0 / (2.0 ** shift_k)
            out = (x * x) * scale
            
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        scale = 1.0 / (2.0 ** ctx.shift_k)
        grad_x = grad_output * (2.0 * x * scale)
        return grad_x, None

# --- Modules ---

class BitShiftNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        
    def forward(self, x):
        return BitShiftNormFunction.apply(x, self.gamma)

class SquareActivation(nn.Module):
    def __init__(self, shift_k=0):
        super().__init__()
        self.shift_k = shift_k
        
    def forward(self, x):
        return SquareActivationFunction.apply(x, self.shift_k)

# --- Full Block ---

class MambaIntegerBlock(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        d_model = config['d_model']
        d_inner = d_model * 2 
        dt_rank = config['ssm_cfg']['dt_rank']
        d_state = config['ssm_cfg']['d_state']
        
        self.norm = BitShiftNorm(d_model)
        self.in_proj = BitLinear(d_model, d_inner * 2)
        
        self.conv1d = nn.Conv1d(
            in_channels=d_inner,
            out_channels=d_inner,
            bias=True,
            kernel_size=4,
            groups=d_inner,
            padding=3,
        )
        
        self.activation = SquareActivation()
        
        self.x_proj = BitLinear(d_inner, dt_rank + 2 * d_state)
        self.dt_proj = BitLinear(dt_rank, d_inner)
        self.out_proj = BitLinear(d_inner, d_model)
        
        # Dyadic HiPPO Initialization
        from dyadic_hippo import get_hippo_s4d_real, project_to_dyadic
        A_ref = get_hippo_s4d_real(d_state)
        nums, shifts = project_to_dyadic(A_ref, scale_bits=15)
        
        # Expand to [D_inner, D_state]
        # Store as float for learnability (will be rounded in forward)
        self.base_decay_nums = nn.Parameter(nums.float().unsqueeze(0).repeat(d_inner, 1))
        self.register_buffer('decay_shifts', shifts.unsqueeze(0).repeat(d_inner, 1))

    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        
        # Project
        xz = self.in_proj(hidden_states)
        x, z = xz.chunk(2, dim=-1)
        
        # Conv1d
        x_t = x.transpose(1, 2)
        x_t = self.conv1d(x_t)[:, :, :hidden_states.shape[1]]
        x = x_t.transpose(1, 2)
        
        x = self.activation(x)
        
        # SSM Parameters
        x_dbl = self.x_proj(x)
        dt_rank = self.config['ssm_cfg']['dt_rank']
        d_state = self.config['ssm_cfg']['d_state']
        
        dt, B_ssm, C_ssm = x_dbl.split([dt_rank, d_state, d_state], dim=-1)
        
        # Decay Modulation
        decay_mod = self.dt_proj(dt)
        decay_nums = self.base_decay_nums.unsqueeze(0).unsqueeze(0) + decay_mod.unsqueeze(-1)
        decay_nums = torch.clamp(decay_nums, 0, 32767).int()
        
        # Input Expansion
        u = x.unsqueeze(-1) * B_ssm.unsqueeze(2)
        
        # Flatten
        B_size, L_size, D_in = x.shape
        flat_dim = D_in * d_state
        u_flat = u.reshape(B_size, L_size, flat_dim)
        decay_nums_flat = decay_nums.reshape(B_size, L_size, flat_dim)
        decay_shifts_flat = self.decay_shifts.view(-1).unsqueeze(0).unsqueeze(0).expand(B_size, L_size, flat_dim)
        
        # Dyadic Scan
        h_flat = torch.empty_like(u_flat)
        if lib_scan and u_flat.is_cuda:
            lib_scan.launch_dyadic_scan(
                ctypes.c_void_p(u_flat.contiguous().data_ptr()),
                ctypes.c_void_p(decay_nums_flat.contiguous().data_ptr()),
                ctypes.c_void_p(decay_shifts_flat.contiguous().data_ptr()),
                ctypes.c_void_p(h_flat.data_ptr()),
                ctypes.c_int(B_size),
                ctypes.c_int(L_size),
                ctypes.c_int(flat_dim),
                ctypes.c_int(15)
            )
        else:
            h_flat = u_flat # Fallback
            
        # Output Projection
        h = h_flat.reshape(B_size, L_size, D_in, d_state)
        y = torch.einsum('bldn,bln->bld', h, C_ssm)
        y = y * self.activation(z)
        out = self.out_proj(y)
        
        return out + residual

from torch.utils.checkpoint import checkpoint

class MambaIntegerModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config['vocab_size'], config['d_model'])
        self.layers = nn.ModuleList([
            MambaIntegerBlock(config, i) for i in range(config['n_layer'])
        ])
        self.norm_f = BitShiftNorm(config['d_model'])
        self.lm_head = BitLinear(config['d_model'], config['vocab_size'])
        self.gradient_checkpointing = True # Enable by default for training
        
    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(layer, x, use_reentrant=False)
            else:
                x = layer(x)
        x = self.norm_f(x)
        logits = self.lm_head(x)
        return logits