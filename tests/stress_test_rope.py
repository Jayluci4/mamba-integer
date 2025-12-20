
import torch
import math
import numpy as np
import ctypes
import os
import sys

# Add path to find rational_bitnet
sys.path.append("../bitnet-odp/src")
from rational_bitnet import DyadicRoPE

def standard_rope(q, k, position_ids, dim, base=10000.0):
    # Ground Truth implementation using transcendental sin/cos
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(q.device) / dim))
    t = position_ids.unsqueeze(-1).float() * inv_freq
    
    cos = torch.cos(t).unsqueeze(1)
    sin = torch.sin(t).unsqueeze(1)
    
    def apply(x, c, s):
        x1 = x[..., :dim//2]
        x2 = x[..., dim//2:]
        return torch.cat([x1 * c - x2 * s, x1 * s + x2 * c], dim=-1)
    
    return apply(q, cos, sin), apply(k, cos, sin)

def stress_test():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- STRESS TESTING DYADIC ROPE ({device}) ---")
    
    # Parameters
    B, H, D = 4, 8, 128
    S_MAX = 16384 # 16k context
    
    rope = DyadicRoPE(D, max_position=S_MAX).to(device)
    
    # 1. Correctness vs Ground Truth
    print(f"\n1. Correctness vs Standard RoPE (S={S_MAX}):")
    q = torch.randn(B, H, S_MAX, D, device=device)
    k = torch.randn(B, H, S_MAX, D, device=device)
    pos_ids = torch.arange(S_MAX, device=device).unsqueeze(0)
    
    with torch.no_grad():
        q_std, k_std = standard_rope(q, k, pos_ids, D)
        q_dya, k_dya = rope(q, k, pos_ids)
        
    l1_err = (q_std - q_dya).abs().mean().item()
    max_err = (q_std - q_dya).abs().max().item()
    
    print(f"   Mean L1 Error: {l1_err:.8f}")
    print(f"   Max Error:     {max_err:.8f}")
    
    if l1_err < 1e-4:
        print("   ✅ SUCCESS: Ground Truth Match")
    else:
        print("   ❌ FAILURE: Error too high")

    # 2. Energy Preservation (Orthogonality)
    print("\n2. Energy Preservation (Norm Stability):")
    q_norm_in = q.norm(dim=-1).mean().item()
    q_norm_out = q_dya.norm(dim=-1).mean().item()
    drift = abs(q_norm_in - q_norm_out)
    
    print(f"   Mean Norm Before: {q_norm_in:.6f}")
    print(f"   Mean Norm After:  {q_norm_out:.6f}")
    print(f"   Norm Drift:       {drift:.8f}")
    
    if drift < 1e-5:
        print("   ✅ SUCCESS: Perfectly Orthogonal")
    else:
        print("   ❌ FAILURE: Energy Leakage")

    # 3. Singularity / Pole Stress
    print("\n3. Singularity Stress (Testing pi-crossings):")
    # Force a position where theta is exactly pi
    # theta = pos * freq. pos = pi / freq
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, D, 2).float() / D))
    # Pick a freq and find matching pos
    freq = inv_freq[0].item()
    special_pos = int(math.pi / freq)
    
    # Test a small window around the pole
    pos_special = torch.arange(special_pos - 5, special_pos + 5, device=device).unsqueeze(0)
    q_spec = torch.ones(1, 1, 10, D, device=device)
    k_spec = torch.ones(1, 1, 10, D, device=device)
    
    q_rot, _ = rope(q_spec, k_spec, pos_special)
    
    has_nan = torch.isnan(q_rot).any().item()
    has_inf = torch.isinf(q_rot).any().item()
    
    print(f"   NaN detected: {has_nan}")
    print(f"   Inf detected: {has_inf}")
    
    if not (has_nan or has_inf):
        print("   ✅ SUCCESS: Angle reduction handled singularities")
    else:
        print("   ❌ FAILURE: Singularity blew up")

if __name__ == "__main__":
    stress_test()
