
import torch
import torch.nn as nn
import ezkl
import os
import json
import math
import onnx
from torch.onnx import export

# Define the ONNX-friendly Dyadic RoPE
class DyadicRoPE_ONNX(nn.Module):
    def __init__(self, dim, max_position=2048, base=10000.0):
        super().__init__()
        self.dim = dim
        self.max_position = max_position
        self.base = base
        self.scale_bits = 20
        self.scale = 2.0**self.scale_bits
        
        # Precompute params (Same logic as CUDA)
        self._precompute_params()
        
    def _precompute_params(self):
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        position = torch.arange(self.max_position).float()
        theta = torch.einsum("i,j->ij", position, inv_freq)
        
        # Normalize to [-pi, pi]
        theta_norm = (theta + math.pi) % (2 * math.pi) - math.pi
        
        # Branchless signs
        signs = torch.ones_like(theta_norm)
        mask_pos = theta_norm > (math.pi / 2)
        theta_norm[mask_pos] -= math.pi
        signs[mask_pos] *= -1.0
        
        mask_neg = theta_norm < (-math.pi / 2)
        theta_norm[mask_neg] += math.pi
        signs[mask_neg] *= -1.0
        
        # 3-Shear Coefficients
        t = torch.tan(theta_norm / 2)
        s = torch.sin(theta_norm)
        
        # Store as floats scaled by 1/scale for direct multiplication in ONNX
        # This matches the integer logic: (y * lambda) >> k  <==> y * (lambda/2^k)
        lambdas = torch.round(-t * self.scale) / self.scale
        gammas = torch.round(s * self.scale) / self.scale
        
        self.register_buffer("lambdas", lambdas)
        self.register_buffer("gammas", gammas)
        self.register_buffer("signs", signs)

    def forward(self, x):
        # Input x: [1, Seq, Dim]
        # We process pairs.
        # To make it ONNX friendly and static, we assume x is flattened or we reshape.
        # Let's assume standard [1, S, D]
        
        B, S, D = x.shape
        half_dim = D // 2
        
        x1 = x[..., :half_dim]
        x2 = x[..., half_dim:]
        
        # Get params for this sequence length
        lam = self.lambdas[:S, :] # [S, D/2]
        gam = self.gammas[:S, :]
        sgn = self.signs[:S, :]
        
        # Broadcast batch dim
        lam = lam.unsqueeze(0)
        gam = gam.unsqueeze(0)
        sgn = sgn.unsqueeze(0)
        
        # 3-Shear Rotation (Arithmetic only)
        # 1. Shear X
        x1 = x1 + x2 * lam
        # 2. Shear Y
        x2 = x2 + x1 * gam
        # 3. Shear X
        x1 = x1 + x2 * lam
        
        # Sign flip (Branchless)
        x1 = x1 * sgn
        x2 = x2 * sgn
        
        return torch.cat([x1, x2], dim=-1)

def run_benchmark():
    print("--- ZK Benchmark: Dyadic-Cayley RoPE ---")
    
    # 1. Configuration
    SEQ_LEN = 128  # Keep small for quick benchmark
    DIM = 64
    model = DyadicRoPE_ONNX(dim=DIM, max_position=SEQ_LEN)
    model.eval()
    
    # 2. Export ONNX
    print("Exporting ONNX...")
    x = torch.randn(1, SEQ_LEN, DIM)
    onnx_path = "dyadic_rope.onnx"
    
    export(
        model,
        x,
        onnx_path,
        opset_version=12, # Downgrade for EZKL compatibility
        input_names=['input'],
        output_names=['output'],
        # dynamic_axes={'input': {0: 'batch'}}, # REMOVED: EZKL requires static shapes
    )
    print(f"Saved {onnx_path}")
    
    # Simplify ONNX (Crucial for EZKL)
    try:
        from onnxsim import simplify
        print("Simplifying ONNX...")
        onnx_model = onnx.load(onnx_path)
        model_simp, check = simplify(onnx_model)
        if check:
            onnx.save(model_simp, onnx_path)
            print("ONNX Simplified successfully.")
        else:
            print("ONNX Simplification check failed.")
    except Exception as e:
        print(f"ONNX Simplification failed: {e}")
    
    # 3. Verify ONNX Ops
    print("Verifying ONNX Graph...")
    onnx_model = onnx.load(onnx_path)
    op_counts = {}
    for node in onnx_model.graph.node:
        op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1
    
    print("Operation Counts:", json.dumps(op_counts, indent=2))
    
    # Pitfall Check: Are there Sin/Cos?
    if 'Sin' in op_counts or 'Cos' in op_counts:
        print("❌ FAILED: Graph contains Sin/Cos nodes!")
    else:
        print("✅ SUCCESS: Graph is purely arithmetic (No Sin/Cos).")

    # 4. EZKL Benchmark
    print("\nRunning EZKL Benchmark...")
    
    settings_path = "settings.json"
    data_path = "input.json"
    
    # Generate random input for calibration
    input_data = dict(input_data=[x.flatten().tolist()])
    json.dump(input_data, open(data_path, 'w'))
    
    # Gen settings
    py_run_args = ezkl.PyRunArgs()
    py_run_args.input_visibility = "public"
    py_run_args.output_visibility = "public"
    py_run_args.param_visibility = "fixed" # Params are fixed constants (lambdas/gammas)
    
    res = ezkl.gen_settings(onnx_path, settings_path, py_run_args=py_run_args)
    
    if res:
        # Calibrate settings
        await_calibration = ezkl.calibrate_settings(
            data_path, settings_path, target="resources"
        )
        
        # Get Stats
        # Since 'calibrate_settings' might take time or async, we can load settings.json
        # to see estimates.
        
        with open(settings_path, 'r') as f:
            settings = json.load(f)
            
        print("\nEZKL Circuit Estimates:")
        print(f"  LogRows: {settings['run_args']['logrows']}")
        
        # Compile circuit to get precise stats
        model_path = "network.ezkl"
        compiled_res = ezkl.compile_circuit(onnx_path, model_path, settings_path)
        
        # Get gate counts
        print("  Getting circuit stats...")
        stats = ezkl.get_circuit_stats(model_path, settings_path)
        print("Circuit Stats:", stats)
        
        # Extract Lookup count
        # Usually under "lookup_count" or similar
        # Since EZKL output varies, we look for 'total_lookups' if available
        # Or deduce from ops.
        
    else:
        print("❌ EZKL Settings Generation Failed")

if __name__ == "__main__":
    run_benchmark()
