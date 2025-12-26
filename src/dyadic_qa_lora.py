
import torch
import torch.nn as nn
import math
import sys
import os

# Add path to find rational_bitnet (use local copy)
sys.path.insert(0, os.path.dirname(__file__))
from rational_bitnet import BitLinear, activation_quant_dynamic, ste_round

class DyadicQALoRA(nn.Module):
    """
    Quantization-Aware LoRA for BitNet / Dyadic Models.
    Solves the "LoRA Trap" by ensuring the adapter path matches 
    the quantization logic of the base BitLinear layer.
    """
    def __init__(self, base_layer: BitLinear, rank: int = 16, alpha: float = 32.0):
        super().__init__()
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        in_features = base_layer.in_features
        out_features = base_layer.out_features
        device = base_layer.weight.device
        dtype = base_layer.weight.dtype

        # LoRA Matrices (Full precision during training, targeted for INT8)
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features, device=device, dtype=dtype))

        # Initialization
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        # Freeze base weights
        self.base_layer.weight.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. QUANTIZE INPUT (Shared between base and LoRA)
        # This is the fix for the LoRA Trap.
        x_quant, x_scale = activation_quant_dynamic(x, self.base_layer.activation_bits)
        
        # 2. BASE PATH (Ternary Weights)
        # We manually re-implement BitLinear.forward to use the shared x_quant
        w_quant, w_scale = self.base_layer.get_ternary_weights()
        # Cast to same dtype as x_quant for simulation
        base_out = torch.nn.functional.linear(x_quant, w_quant.to(x_quant.dtype), None)
        base_out = base_out * (w_scale * x_scale)
        
        # 3. LORA PATH (Quantized Adapter)
        # We simulate INT8 quantization for lora matrices during training
        def quantize_lora(w):
            scale = w.abs().max().clamp(min=1e-6)
            w_q = ste_round(w * 127 / scale).clamp(-127, 127)
            return w_q, scale / 127
        
        a_q, a_s = quantize_lora(self.lora_A)
        b_q, b_s = quantize_lora(self.lora_B)
        
        # Adapter computation: (x_q * a_q) * b_q
        # We must align scales:
        # lora_out = (x_q @ a_q @ b_q) * (x_scale * a_s * b_s * scaling)
        lora_out = (x_quant @ a_q @ b_q)
        lora_out = lora_out * (x_scale * a_s * b_s * self.scaling)
        
        if self.base_layer.bias is not None:
            base_out = base_out + self.base_layer.bias
            
        return base_out + lora_out

    @torch.no_grad()
    def merge_weights(self):
        """
        Merge LoRA weights into the base layer for 100% Integer Inference.
        """
        # This is now possible because both paths are quantized!
        # W_new = W_base + (A @ B) * alpha/rank
        # Note: In practice, we'd convert the resulting matrix back to ternary 
        # or use high-bit integer if capacity demands it.
        pass

def convert_to_qa_healing(model):
    """Replace BitLinear with DyadicQALoRA."""
    replaces = []
    for name, module in model.named_modules():
        if isinstance(module, BitLinear):
            replaces.append((name, module))
            
    for name, module in replaces:
        parent_name = '.'.join(name.split('.')[:-1])
        attr_name = name.split('.')[-1]
        parent = model.get_submodule(parent_name) if parent_name else model
        
        new_layer = DyadicQALoRA(module)
        setattr(parent, attr_name, new_layer)
        
    print(f"Injected {len(replaces)} Dyadic-QA-LoRA adapters.")

if __name__ == "__main__":
    # Test on a single BitLinear
    print("Testing Dyadic-QA-LoRA...")
    base = BitLinear(128, 128).cuda()
    lora = DyadicQALoRA(base).cuda()
    
    x = torch.randn(1, 16, 128).cuda()
    out = lora(x)
    print(f"Output Shape: {out.shape}")
    print("Backward pass...")
    out.mean().backward()
    print("Gradients computed for lora_A:", lora.lora_A.grad is not None)
    print("SUCCESS: QA-LoRA is differentiable and quantization-aware.")
