
import torch
import struct
import os
import sys
import json

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../bitnet-odp/src'))
sys.path.insert(0, os.path.dirname(__file__)) 
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mamba_integer_model import MambaIntegerModel
from rational_bitnet import weight_quant_ternary

CONFIG_PATH = "config_mamba_integer_l4.json"
# Use the stable 1000 step checkpoint
CHECKPOINT_PATH = "/home/jayantlohia16/experiment/gemma-intelligent/conv/src/dyadic_experiment/mamba/mamba_integer_step_1000.pt"
OUTPUT_BIN = "mamba_integer_1000.bin"

def export_model():
    print(f"--- Exporting Dyadic Mamba to Binary ({OUTPUT_BIN}) ---")
    
    # Load Model
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)
    model = MambaIntegerModel(config).cuda()
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location="cuda"))
    model.eval()
    
    with open(OUTPUT_BIN, "wb") as f:
        # 1. Header / Config
        # Magic number: 0xDAD1C
        f.write(struct.pack("I", 0xDAD1C))
        f.write(struct.pack("I", config['d_model']))
        f.write(struct.pack("I", config['n_layer']))
        f.write(struct.pack("I", config['vocab_size']))
        f.write(struct.pack("I", config['ssm_cfg']['d_state']))
        f.write(struct.pack("I", config['ssm_cfg']['dt_rank']))
        
        # 2. Embedding
        # Embeddings are usually float or high precision. 
        # For "Integer-Only", we can quantize to int8?
        # Let's keep Embedding as Float32 for now to ensure quality in this POC.
        # Or int8. Standard BitNet keeps input/output high precision.
        print("Exporting Embeddings...")
        embed = model.embedding.weight.detach().cpu().numpy()
        f.write(embed.tobytes())
        
        # 3. Layers
        for i, layer in enumerate(model.layers):
            print(f"Exporting Layer {i}...")
            
            # BitShiftNorm Gamma (Float)
            gamma = layer.norm.gamma.detach().cpu().numpy()
            f.write(gamma.tobytes())
            
            # InProj (BitLinear) -> Export Quantized Int8 Weights + Float Scale
            # Weights are {-1, 0, 1}. Stored as int8.
            w_quant, w_scale = layer.in_proj.get_ternary_weights()
            w_int8 = w_quant.cpu().to(torch.int8).numpy()
            f.write(struct.pack("f", w_scale.item())) # Scale
            f.write(w_int8.tobytes()) # Weights [Out, In]
            
            # Conv1d (Standard Float)
            # We didn't bit-quantize conv. It's float.
            conv_w = layer.conv1d.weight.detach().cpu().numpy()
            conv_b = layer.conv1d.bias.detach().cpu().numpy()
            f.write(conv_w.tobytes())
            f.write(conv_b.tobytes())
            
            # X_proj (BitLinear)
            w_quant, w_scale = layer.x_proj.get_ternary_weights()
            f.write(struct.pack("f", w_scale.item()))
            f.write(w_quant.cpu().to(torch.int8).numpy().tobytes())
            
            # DT_proj (BitLinear)
            w_quant, w_scale = layer.dt_proj.get_ternary_weights()
            f.write(struct.pack("f", w_scale.item()))
            f.write(w_quant.cpu().to(torch.int8).numpy().tobytes())
            # DT bias (float)
            if layer.dt_proj.bias is not None:
                f.write(layer.dt_proj.bias.detach().cpu().numpy().tobytes())
            else:
                # write zeros
                zeros = torch.zeros(layer.dt_proj.out_features).numpy()
                f.write(zeros.tobytes())

            # Out_proj (BitLinear)
            w_quant, w_scale = layer.out_proj.get_ternary_weights()
            f.write(struct.pack("f", w_scale.item()))
            f.write(w_quant.cpu().to(torch.int8).numpy().tobytes())
            
            # Decay Params (Int32)
            # base_decay_nums was Float parameter, but represents Integers.
            # We need to round and save as int32.
            # Note: During training it was float.
            nums = layer.base_decay_nums.detach().round().int().cpu().numpy()
            shifts = layer.decay_shifts.cpu().int().numpy()
            f.write(nums.tobytes())
            f.write(shifts.tobytes())
            
            # ReZero (Float)
            gate = layer.res_gate.detach().cpu().numpy()
            f.write(gate.tobytes())

        # 4. Final Norm
        print("Exporting Final Norm...")
        gamma = model.norm_f.gamma.detach().cpu().numpy()
        f.write(gamma.tobytes())
        
        # 5. LM Head (BitLinear)
        print("Exporting LM Head...")
        w_quant, w_scale = model.lm_head.get_ternary_weights()
        f.write(struct.pack("f", w_scale.item()))
        f.write(w_quant.cpu().to(torch.int8).numpy().tobytes())
        
    print(f"Successfully exported to {OUTPUT_BIN}")

if __name__ == "__main__":
    export_model()
