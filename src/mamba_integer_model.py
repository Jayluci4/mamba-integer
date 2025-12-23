
import torch
import torch.nn as nn
import ctypes
import os
import math
import sys

# Add path for rational_bitnet
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../bitnet-odp/src'))
sys.path.insert(0, os.path.dirname(__file__)) 
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from rational_bitnet import BitLinear

# --- Load Kernels ---
def load_lib(name):
    path = os.path.join(os.path.dirname(__file__), f"cuda_kernels/lib{name}.so")
    # Disable ctypes for torch.compile compatibility
    return None 
    # if os.path.exists(path):
    #    return ctypes.CDLL(path)
    # return None

lib_bitshift = load_lib("bitshift_norm")
lib_squareplus = load_lib("squareplus")
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "triton_kernels"))
from dyadic_scan import dyadic_scan_triton, dyadic_scan_backward_triton
try:
    from bitnet_kernels import fast_bitshift_norm
    BITSHIFT_TRITON_AVAILABLE = True
except ImportError:
    BITSHIFT_TRITON_AVAILABLE = False

# Setup library path for custom CUDA kernels
LIB_PATH = os.path.join(os.path.dirname(__file__), "cuda_kernels/libdyadic_mamba.so")
lib_scan = None
# if os.path.exists(LIB_PATH):
#    print(f"DEBUG: Found CUDA Library at {LIB_PATH}")
#    lib_scan = ctypes.CDLL(LIB_PATH)
# else:
#    print(f"DEBUG: CUDA Library NOT FOUND at {LIB_PATH}")
#    lib_scan = None

# --- Autograd Functions ---

class BitShiftNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gamma, scale_bits=0):
        ctx.save_for_backward(x, gamma)
        ctx.scale_bits = scale_bits
        
        B, S, D = x.shape
        out = torch.empty_like(x)
        
        if BITSHIFT_TRITON_AVAILABLE and x.is_cuda:
            # Use Fused Triton Kernel (Fast)
            return fast_bitshift_norm(x, gamma)
        elif lib_bitshift and x.is_cuda:
            lib_bitshift.launch_bitshift_norm(
                ctypes.c_void_p(x.contiguous().data_ptr()),
                ctypes.c_void_p(gamma.contiguous().data_ptr()),
                ctypes.c_void_p(out.data_ptr()),
                ctypes.c_int(B), ctypes.c_int(S), ctypes.c_int(D),
                ctypes.c_int(scale_bits)
            )
        else:
            # CPU Fallback
            var = x.pow(2).mean(dim=-1, keepdim=True)
            k = torch.round(torch.log2(torch.sqrt(var + 1e-9))).int()
            k = torch.clamp(k, min=0)
            scale = 1.0 / (2.0 ** k.float())
            out = x * scale * gamma
            
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, gamma = ctx.saved_tensors
        # Use analytic RMSNorm gradient for stability
        x_fp = x.float()
        var = x_fp.pow(2).mean(dim=-1, keepdim=True)
        rms = torch.sqrt(var + 1e-9)
        inv_rms = 1.0 / rms
        
        D = x.size(-1)
        grad_output_scaled = grad_output * gamma
        term1 = grad_output_scaled * inv_rms
        dot = (grad_output_scaled * x_fp).sum(dim=-1, keepdim=True)
        term2 = x_fp * dot * (inv_rms * inv_rms * inv_rms) / D
        grad_x = term1 - term2
        
        # Gamma grad
        k = torch.round(torch.log2(rms)).int().clamp(min=0)
        scale_fwd = 1.0 / (2.0 ** k.float())
        grad_gamma = (grad_output * x * scale_fwd).sum(dim=(0, 1))
        
        return grad_x, grad_gamma, None

class SquareplusFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        B, S, D = x.shape
        out = torch.empty_like(x)
        if lib_squareplus and x.is_cuda:
            lib_squareplus.launch_squareplus(
                ctypes.c_void_p(x.contiguous().data_ptr()),
                ctypes.c_void_p(out.data_ptr()),
                ctypes.c_int(B), ctypes.c_int(S), ctypes.c_int(D)
            )
        else:
            out = 0.5 * (x + torch.sqrt(x*x + 4.0))
        return out

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        z = torch.sqrt(x*x + 4.0)
        grad_x = grad_output * 0.5 * (1.0 + x / z)
        return grad_x

class RationalSigmoidFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        z = torch.sqrt(x*x + 1.0)
        out = 0.5 * (1.0 + x / z)
        ctx.save_for_backward(x, z)
        return out
    @staticmethod
    def backward(ctx, grad_output):
        x, z = ctx.saved_tensors
        grad_x = grad_output * 0.5 / (z * z * z)
        return grad_x

class DyadicScanFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, decay_nums_float, decay_shifts, scale_bits=15):
        ctx.scale_bits = scale_bits
        B, L, D = u.shape
        
        u_c = u.contiguous()
        d_n_c = decay_nums_float.contiguous().int()
        d_s_c = decay_shifts.contiguous().int()
        
        # Always use Triton for speed + compilation support
        h = dyadic_scan_triton(u_c, d_n_c, d_s_c, scale_bits)
            
        ctx.save_for_backward(h, decay_nums_float, decay_shifts)
        return h

    @staticmethod
    def backward(ctx, grad_h):
        h, decay_nums_float, decay_shifts = ctx.saved_tensors
        
        # Ensure contiguous inputs
        grad_h_c = grad_h.contiguous()
        h_c = h.contiguous()
        d_n_c = decay_nums_float.contiguous().int()
        d_s_c = decay_shifts.contiguous().int()
        
        # Use Triton backward for full inductor fusion
        grad_u, grad_decay = dyadic_scan_backward_triton(
            grad_h_c, h_c, d_n_c, d_s_c, ctx.scale_bits
        )
        return grad_u, grad_decay, None, None

# --- Modules ---

class BitShiftNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer('last_k', torch.tensor(0.0))
    def forward(self, x):
        # Mean Centering for Stability (LayerNorm-like)
        x = x - x.mean(dim=-1, keepdim=True)
        # if self.training:
        #    with torch.no_grad():
        #        var = x.pow(2).mean(dim=-1)
        #        k = torch.log2(torch.sqrt(var + 1e-9)).mean()
        #        self.last_k.copy_(k)
        return BitShiftNormFunction.apply(x, self.gamma)

class DampenedSquareplus(nn.Module):
    def forward(self, x):
        # Dampen by 0.5 to prevent gain > 1
        return 0.5 * SquareplusFunction.apply(x)

class RationalSigmoid(nn.Module):
    def forward(self, x): return RationalSigmoidFunction.apply(x)

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
        self.conv1d = nn.Conv1d(in_channels=d_inner, out_channels=d_inner, bias=True, kernel_size=4, groups=d_inner, padding=3)
        
        # Rational Squareplus (Restored)
        self.activation = DampenedSquareplus()
        self.gate_activation = RationalSigmoid()
        self.register_buffer('last_act_max', torch.tensor(0.0))
        
        self.x_proj = BitLinear(d_inner, dt_rank + 2 * d_state)
        self.dt_proj = BitLinear(dt_rank, d_inner)
        nn.init.normal_(self.dt_proj.weight, mean=0.0, std=0.001)
        if self.dt_proj.bias is not None:
            nn.init.zeros_(self.dt_proj.bias)
        self.out_proj = BitLinear(d_inner, d_model)
        
        # Scale Output Weights (0.1)
        nn.init.normal_(self.out_proj.weight, mean=0.0, std=0.02 / math.sqrt(config['n_layer']))
        
        from dyadic_hippo import get_hippo_s4d_real, project_to_dyadic
        A_ref = get_hippo_s4d_real(d_state)
        nums, shifts = project_to_dyadic(A_ref, scale_bits=15)
        self.base_decay_nums = nn.Parameter(nums.float().unsqueeze(0).repeat(d_inner, 1))
        self.register_buffer('decay_shifts', shifts.unsqueeze(0).repeat(d_inner, 1))
        
        # ReZero
        self.res_gate = nn.Parameter(torch.zeros(1))

    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        
        xz = self.in_proj(hidden_states)
        x, z = xz.chunk(2, dim=-1)
        
        x_t = x.transpose(1, 2)
        x_t = self.conv1d(x_t)[:, :, :hidden_states.shape[1]]
        x = x_t.transpose(1, 2)
        
        x = self.activation(x)
        
        x_dbl = self.x_proj(x)
        dt_rank = self.config['ssm_cfg']['dt_rank']
        d_state = self.config['ssm_cfg']['d_state']
        dt, B_ssm, C_ssm = x_dbl.split([dt_rank, d_state, d_state], dim=-1)
        
        decay_mod = self.dt_proj(dt)
        decay_nums = self.base_decay_nums.unsqueeze(0).unsqueeze(0) + decay_mod.unsqueeze(-1)
        decay_nums = torch.clamp(decay_nums, 0, 32767)
        
        # Dampened dt - Aggressive 0.1 for Rescue Plan
        dt_val = self.activation(decay_mod) * 0.1
        
        u = (x * dt_val).unsqueeze(-1) * B_ssm.unsqueeze(2)
        # Monitoring disabled for speed (avoid graph breaks)
        # if self.training:
        #    with torch.no_grad():
        #        self.last_act_max.copy_(u.abs().max())
        
        B_size, L_size, D_in = x.shape
        flat_dim = D_in * d_state
        u_flat = u.reshape(B_size, L_size, flat_dim)
        decay_nums_flat = decay_nums.reshape(B_size, L_size, flat_dim)
        decay_shifts_flat = self.decay_shifts.view(-1).unsqueeze(0).unsqueeze(0).expand(B_size, L_size, flat_dim)
        
        h_flat = DyadicScanFunction.apply(u_flat, decay_nums_flat, decay_shifts_flat, 15)
        # Clamping
        h_flat = torch.clamp(h_flat, -32000, 32000)
            
        h = h_flat.reshape(B_size, L_size, D_in, d_state)
        y = torch.einsum('bldn,bln->bld', h, C_ssm)
        y = y * self.gate_activation(z)
        
        out = self.out_proj(y)
        out = torch.clamp(out, -32000, 32000)
        
        return residual + out * self.res_gate

from torch.utils.checkpoint import checkpoint

class MambaIntegerModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config['vocab_size'], config['d_model'])
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        self.layers = nn.ModuleList([MambaIntegerBlock(config, i) for i in range(config['n_layer'])])
        self.norm_f = BitShiftNorm(config['d_model'])
        self.lm_head = BitLinear(config['d_model'], config['vocab_size'])
        self.gradient_checkpointing = False
        
    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm_f(x)
        logits = self.lm_head(x)
        return logits
