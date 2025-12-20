
import torch
import torch.nn as nn
from transformers import MambaForCausalLM, AutoTokenizer
import math
import sys
import os

# Add path to find our kernel wrapper
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_mamba_scan import DyadicMambaScan

class DyadicMambaAdapter(nn.Module):
    """
    Adapts a standard Mamba Mixer to use the Dyadic Integer Scan.
    
    Standard Mamba:
    1. x -> conv1d -> x
    2. x, z = in_proj(x)
    3. A, D, dt = params
    4. dt = softplus(dt_proj(x))
    5. dA = exp(dt * A)
    6. h = scan(x, dA) ...
    
    Dyadic Mamba:
    ...    
    5. dA (float) is computed.
    6. Convert dA -> (num, shift)
    7. h = dyadic_scan(x, num, shift)
    """
    def __init__(self, original_mixer, config):
        super().__init__()
        self.config = config
        
        # Copy references to weights (don't duplicate memory)
        self.in_proj = original_mixer.in_proj
        self.conv1d = original_mixer.conv1d
        self.x_proj = original_mixer.x_proj
        self.dt_proj = original_mixer.dt_proj
        self.out_proj = original_mixer.out_proj
        
        self.A_log = original_mixer.A_log
        self.D = original_mixer.D
        
        self.activation = original_mixer.act
        
        # The Kernel
        self.scan_kernel = DyadicMambaScan(scale_bits=15) # Match our verification script
        
    def forward(self, hidden_states, cache_params=None, **kwargs):
        # 1. Project Inputs
        batch, seq, dim = hidden_states.shape
        
        # fused proj
        projected = self.in_proj(hidden_states)
        # split (x, z)
        d_inner = self.config.intermediate_size
        x, z = projected.chunk(2, dim=-1)
        
        # 2. Conv1d
        # x shape for conv: [B, D, L]
        x_t = x.transpose(1, 2)
        x_t = self.conv1d(x_t)[:, :, :seq] # Causal padding handled by conv usually
        x = x_t.transpose(1, 2)
        
        x = self.activation(x)
        
        # 3. Compute Delta and Parameters
        # x_dbl = x_proj(x) -> (dt, B, C)
        # This part is Mamba specific.
        # dt_rank + d_state * 2
        
        x_dbl = self.x_proj(x)
        dt, B, C = x_dbl.split([self.config.time_step_rank, self.config.state_size, self.config.state_size], dim=-1)
        
        dt = self.dt_proj(dt)
        
        # 4. Discretize A (The Bridge)
        # A = -exp(A_log)
        A = -torch.exp(self.A_log.float()) # [D_inner, D_state]
        
        # dt is [Batch, Seq, D_inner]
        # dt = F.softplus(dt)
        dt = torch.nn.functional.softplus(dt)
        
        # Discrete A = exp(dt * A) 
        # A is broadcast over Batch, Seq. dt is broadcast over D_state?
        # A: [D_in, D_st], dt: [B, L, D_in]
        # dA: [B, L, D_in, D_st] -> effective decay per state element?
        # WAIT. Mamba standard scan is usually diagonal or simpler.
        # Transformers implementation details:
        # A is [D_inner, D_state]
        # But `selective_scan_fn` usually takes inputs.
        
        # Let's verify Mamba diagonal structure.
        # Often D_state is small (16).
        # We need decay factor per channel.
        # If A is full matrix, scan is matrix mul.
        # Mamba uses DIAGONAL A (S4D). A is effectively [D_inner, D_state] parameters, 
        # but interpreted as D_inner independent systems of size D_state?
        # Actually, Mamba 1:
        # x: [B, L, D_in]
        # B: [B, L, D_st]
        # C: [B, L, D_st]
        # A: [D_in, D_st]
        # Recurrence: h_t = A_bar * h_{t-1} + B_bar * x_t
        # h: [B, L, D_in, D_st]
        
        # My Dyadic Kernel `dyadic_mamba_kernel.cu` expects:
        # x: [B, S, D]
        # decay: [B, S, D]
        # h_out: [B, S, D]
        # It implements SINGLE-DIMENSION recurrence (scalar state).
        # h_t = a * h_{t-1} + x_t
        
        # Mamba's state is size D_state (e.g. 16).
        # So for each channel D_in, there are 16 states.
        # My kernel treats 'Dim' as independent channels.
        # To support Mamba, I must treat (D_inner * D_state) as the "Dim" of my kernel?
        
        # Yes. We can flatten the state dimensions.
        # Effective Dim = D_inner * D_state.
        # Input x must be broadcasted to this dim.
        # Decay A is [D_inner, D_state].
        
        # Let's prepare inputs for Dyadic Kernel
        
        # A_broad: [B, L, D_in, D_st] = exp(dt.unsqueeze(-1) * A)
        dA = torch.exp(torch.einsum('bld,ds->blds', dt, A))
        
        # X_broad: [B, L, D_in, D_st] = x.unsqueeze(-1) * B.unsqueeze(-2)?
        # No, discretization of B is: B_bar = (dt * B) ? Or similar.
        # Standard: B_bar = x * B * dt?
        # The recurrence input is `delta * B * x`.
        
        # Let's map accurately:
        # dx = softplus(dt)
        # dA = exp(dx * A)
        # dB = dx * B (approx, assuming zero-order hold)
        # u = dB * x
        # h_t = dA * h_{t-1} + u
        
        # So my kernel's "x" input should be `u`.
        # My kernel's "decay" should be `dA`.
        
        # Compute u:
        # B: [Batch, Seq, D_state]
        # x: [Batch, Seq, D_inner]
        # u: [Batch, Seq, D_inner, D_state] = x.unsqueeze(-1) * B.unsqueeze(-2) * dt.unsqueeze(-1)
        
        u = torch.einsum('bld,bln->bldn', x, B)
        u = u * dt.unsqueeze(-1)
        
        # Flatten for Kernel: [Batch, Seq, D_inner * D_state]
        B_size, L_size, Din, Dst = u.shape
        flat_dim = Din * Dst
        
        u_flat = u.reshape(B_size, L_size, flat_dim)
        dA_flat = dA.reshape(B_size, L_size, flat_dim)
        
        # Convert dA_flat to Dyadic (Num, Shift)
        scale_bits = 15
        scale = 2**scale_bits
        
        # dA is in [0, 1].
        # num = round(dA * scale)
        # shift = scale_bits
        
        decay_nums = (dA_flat * scale).clamp(0, 32767).int()
        decay_shifts = torch.full_like(decay_nums, scale_bits)
        
        # Run Kernel
        # h_flat: [B, L, flat_dim]
        h_flat = self.scan_kernel(u_flat, decay_nums, decay_shifts)
        
        # Reshape back
        h = h_flat.reshape(B_size, L_size, Din, Dst)
        
        # Output projection: y = h * C
        # C: [B, L, D_st]
        # y: [B, L, D_in] = sum_s (h * C)
        y = torch.einsum('bldn,bln->bld', h, C)
        
        # Add D residual
        # D is [D_inner]
        y = y + x * self.D
        
        # Gating
        y = y * self.activation(z)
        
        # Out projection
        out = self.out_proj(y)
        return out

def perform_mamba_surgery(model_id="state-spaces/mamba-130m-hf"):
    print(f"Loading {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = MambaForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32).cuda()
    
    print("Performing Surgery...")
    # Iterate and replace layers
    # Mamba structure: model.backbone.layers[i].mixer
    
    replaced = 0
    for i, layer in enumerate(model.backbone.layers):
        original_mixer = layer.mixer
        print(f"  Replacing Layer {i} Mixer...")
        
        # Create adapter
        new_mixer = DyadicMambaAdapter(original_mixer, model.config)
        
        # Move weights (if not shared)
        # new_mixer is already using references to original weights, so it points to CUDA tensors.
        
        # Replace
        layer.mixer = new_mixer
        replaced += 1
        
    print(f"Replaced {replaced} layers.")
    
    # Test Inference
    print("\n--- Inference Test ---")
    inputs = tokenizer("Hello, my name is", return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=20)
        
    print(f"Output: {tokenizer.decode(out[0])}")
    
    # Simple Perplexity Check
    print("Calculating Perplexity on 'The quick brown fox'...")
    text = "The quick brown fox jumps over the lazy dog."
    enc = tokenizer(text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        loss = model(enc.input_ids, labels=enc.input_ids).loss
    print(f"Perplexity: {math.exp(loss.item()):.4f}")

if __name__ == "__main__":
    perform_mamba_surgery()
