
import torch
import torch.nn as nn
import os
import sys
import copy

# Add necessary paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../gemma-intelligent/conv/src/bitnet-odp/src'))

from mamba_integer_model import MambaIntegerBlock
import rational_bitnet
import mamba_integer_model

def test_v2_block_correctness():
    device = "cuda"
    torch.manual_seed(42)
    
    # 1. Config for a small block
    config = {
        'd_model': 128,
        'n_layer': 1,
        'ssm_cfg': {
            'd_state': 16,
            'dt_rank': 32
        }
    }
    
    # 2. Create two identical blocks
    block_v1 = MambaIntegerBlock(config, layer_idx=0).to(device)
    block_v2 = copy.deepcopy(block_v1)
    
    # Input data
    B, L, D = 2, 16, 128
    x = torch.randn(B, L, D, device=device, requires_grad=True)
    x_v1 = x.clone().detach().requires_grad_(True)
    x_v2 = x.clone().detach().requires_grad_(True)
    
    # --- RUN V1 (Baseline: Triton Disabled) ---
    print("\n--- Testing V1 (Standard PyTorch Fallback) ---")
    rational_bitnet.BITNET_TRITON_AVAILABLE = False
    mamba_integer_model.BITSHIFT_TRITON_AVAILABLE = False
    
    out_v1 = block_v1(x_v1)
    loss_v1 = out_v1.sum()
    loss_v1.backward()
    
    grad_x_v1 = x_v1.grad.clone()
    grad_w_v1 = block_v1.in_proj.weight.grad.clone()
    
    # --- RUN V2 (Optimized: Triton Enabled) ---
    print("--- Testing V2 (Triton Kernels) ---")
    rational_bitnet.BITNET_TRITON_AVAILABLE = True
    mamba_integer_model.BITSHIFT_TRITON_AVAILABLE = True
    print(f"  Triton Active in rational_bitnet: {rational_bitnet.BITNET_TRITON_AVAILABLE}")
    print(f"  Triton Active in mamba_integer_model: {mamba_integer_model.BITSHIFT_TRITON_AVAILABLE}")
    
    out_v2 = block_v2(x_v2)
    loss_v2 = out_v2.sum()
    loss_v2.backward()
    
    grad_x_v2 = x_v2.grad.clone()
    grad_w_v2 = block_v2.in_proj.weight.grad.clone()
    
    # --- COMPARISON ---
    print("\n--- Final Results ---")
    
    # Forward Match
    out_diff = (out_v1 - out_v2).abs().max().item()
    print(f"Forward Max Diff:  {out_diff:.2e}")
    
    # Input Gradient Match
    grad_x_diff = (grad_x_v1 - grad_x_v2).abs().max().item()
    print(f"Grad X Max Diff:   {grad_x_diff:.2e}")
    
    # Weight Gradient Match
    grad_w_diff = (grad_w_v1 - grad_w_v2).abs().max().item()
    print(f"Grad W Max Diff:   {grad_w_diff:.2e}")
    
    # Tolerances: 
    # Forward should be very close (~1e-6)
    # Gradients might vary slightly more due to Int8 vs Float matmul but should be < 1e-3
    if out_diff < 1e-4 and grad_x_diff < 1e-3:
        print("\n✅ V2 Correctness Test PASSED")
    else:
        print("\n❌ V2 Correctness Test FAILED")
        sys.exit(1)

if __name__ == "__main__":
    try:
        test_v2_block_correctness()
    except Exception as e:
        print(f"\nCRITICAL ERROR during test: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
