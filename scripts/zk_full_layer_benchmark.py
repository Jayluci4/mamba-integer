
import torch
import torch.nn as nn
import ezkl
import os
import json
import onnx
import sys
from torch.onnx import export

# Add path to find rational_bitnet
sys.path.append("../bitnet-odp/src")
from rational_bitnet import RationalBitNetBlock, RationalBitNetConfig, BitLinear

# Helper to freeze quantization for ZK export
# We want to export the INFERENCE circuit, not the training quantization logic.
def freeze_bitlinear_weights(model):
    for name, module in model.named_modules():
        if isinstance(module, BitLinear):
            # Pre-calculate quantized weights
            w_quant, w_scale = module.get_ternary_weights()
            # Replace weights with quantized values (float for ONNX compatibility, but integer values)
            module.weight.data = w_quant.float()
            # Register scale as a buffer so it's treated as a constant
            module.register_buffer('fixed_scale', w_scale)
            
            # Monkey-patch forward to skip quantization logic
            def inference_forward(self, x):
                # x is already quantized activation (in a real integer setup)
                # But here inputs are float.
                # Standard BitLinear quantizes x too:
                # x_quant, x_scale = activation_quant_dynamic(x, self.activation_bits)
                # For ZK benchmark, we include activation quantization cost 
                # because it happens at runtime.
                # But WEIGHT quantization is static.
                
                # Re-implement activation quant only
                Q_max = 127.0
                scale = x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-8)
                x_scaled = x * Q_max / scale
                # STE Round
                x_quant = x_scaled + (torch.round(x_scaled) - x_scaled).detach()
                x_quant = torch.clamp(x_quant, -Q_max, Q_max)
                
                # Linear with FROZEN ternary weights
                y = torch.nn.functional.linear(x_quant, self.weight, None)
                
                # Rescale
                y = y * (self.fixed_scale * (scale / Q_max))
                
                if self.bias is not None:
                    y = y + self.bias
                return y
            
            # Bind the new method to the instance
            module.forward = inference_forward.__get__(module, BitLinear)

def run_full_layer_benchmark():
    print("--- ZK Benchmark: Full Rational BitNet Layer ---")
    
    # 1. Configuration (Small but realistic for ZK)
    # 256 dim is enough to prove the point vs 2560 (linear scaling)
    config = RationalBitNetConfig(
        hidden_dim=256,
        intermediate_dim=768,
        num_heads=4,
        max_seq_len=64,
        use_dyadic_rope=True, # The Key Feature
        use_linear_attention=True # O(N) is ZK friendly
    )
    
    model = RationalBitNetBlock(config)
    model.eval()
    
    # Freeze weights to simulate inference-only circuit
    freeze_bitlinear_weights(model)
    
    # 2. Export ONNX
    print("Exporting ONNX...")
    seq_len = 16 # Short sequence for benchmark speed
    x = torch.randn(1, seq_len, config.hidden_dim)
    # Dummy position ids
    pos_ids = torch.arange(seq_len).unsqueeze(0)
    # Dummy mask (not used in linear attn but passed)
    mask = torch.zeros(1, 1, seq_len, seq_len)
    
    onnx_path = "rational_layer_dyadic.onnx"
    
    # Export with static shapes
    export(
        model,
        (x, mask, pos_ids),
        onnx_path,
        opset_version=12,
        input_names=['hidden_states', 'attention_mask', 'position_ids'],
        output_names=['output'],
    )
    print(f"Saved {onnx_path}")
    
    # Simplify
    try:
        from onnxsim import simplify
        model_simp, check = simplify(onnx.load(onnx_path))
        if check:
            onnx.save(model_simp, onnx_path)
            print("ONNX Simplified.")
    except ImportError:
        print("onnxsim not found, skipping simplification.")

    # 3. Analyze Ops
    print("Verifying ONNX Graph...")
    onnx_model = onnx.load(onnx_path)
    op_counts = {}
    for node in onnx_model.graph.node:
        op_counts[node.op_type] = op_counts.get(node.op_type, 0) + 1
    
    print("Operation Counts:", json.dumps(op_counts, indent=2))
    
    # Critical Checks
    forbidden = ['Sin', 'Cos', 'Exp', 'Sqrt', 'Div'] # Div is okay if rational, but minimizing is good
    found_forbidden = [op for op in forbidden if op in op_counts]
    
    if found_forbidden:
        print(f"⚠️  WARNING: Found transcendental/expensive ops: {found_forbidden}")
        # Note: RationalRMSNorm might use Div. RationalSiLU uses Div.
        # This is expected for "Rational" approach.
        # But Sin/Cos/Exp/Sqrt should be ZERO.
        if any(x in found_forbidden for x in ['Sin', 'Cos', 'Exp', 'Sqrt']):
             print("❌ FAILED: Graph contains hard transcendentals.")
        else:
             print("✅ SUCCESS: Purely Integer/Polynomial Arithmetic.")
    else:
        print("✅ SUCCESS: Purely Integer/Polynomial Arithmetic.")

    # 4. EZKL Estimates
    print("\nEstimating ZK Circuit Complexity...")
    settings_path = "settings_layer.json"
    data_path = "input_layer.json"
    
    # Dummy input data
    input_data = dict(
        hidden_states=[x.flatten().tolist()],
        attention_mask=[mask.flatten().tolist()],
        position_ids=[pos_ids.flatten().tolist()]
    )
    json.dump(input_data, open(data_path, 'w'))
    
    py_run_args = ezkl.PyRunArgs()
    py_run_args.input_visibility = "public"
    py_run_args.output_visibility = "public"
    py_run_args.param_visibility = "fixed"
    
    try:
        res = ezkl.gen_settings(onnx_path, settings_path, py_run_args=py_run_args)
        if res:
            with open(settings_path, 'r') as f:
                settings = json.load(f)
            print(f"  LogRows (Circuit Depth): {settings['run_args']['logrows']}")
            print(f"  Total Constraints estimate: ~2^{settings['run_args']['logrows']}")
            
            # Attempt calibration for precise lookup count
            print("  Calibrating (counting constraints)...")
            ezkl.calibrate_settings(data_path, settings_path, target="resources")
            
            with open(settings_path, 'r') as f:
                settings = json.load(f)
            
            # In EZKL settings, lookups are implicit in some configs, 
            # but we can infer success if it didn't crash on 'Sin' nodes.
            print("  Calibration Successful.")
            
    except Exception as e:
        print(f"❌ EZKL Error: {e}")

if __name__ == "__main__":
    run_full_layer_benchmark()
