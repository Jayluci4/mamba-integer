import torch
import torch.nn as nn
from mamba_integer_model import MambaIntegerModel
import json
import os
import sys
import numpy as np

# Add path for src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

# --- Config ---
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../configs/config_mamba_integer_l4.json")
MODEL_PATH = "/home/jayantlohia16/experiment/gemma-intelligent/conv/src/dyadic_experiment/mamba/mamba_integer_step_5500.pt"

if not os.path.exists(MODEL_PATH):
    print(f"Error: Could not find {MODEL_PATH}")

with open(CONFIG_PATH, 'r') as f:
    config = json.load(f)

class SimpleTokenizer:
    def __init__(self, vocab_size=4096):
        self.vocab_size = vocab_size
    def encode(self, text):
        return [min(b, self.vocab_size - 1) for b in text.encode('utf-8')]
    def decode(self, ids):
        return bytes([i for i in ids if i < 256]).decode('utf-8', errors='ignore')

def analyze_model_state(model):
    print("\n--- Model State Analysis ---")
    
    # 1. Check Weights Magnitude
    print("Checking Layer 0 Weights:")
    layer0 = model.layers[0]
    
    # BitLinear Weights (Should be small float before quantization)
    w = layer0.in_proj.weight
    print(f"  InProj Weight: Mean={w.mean():.4f}, Std={w.std():.4f}, Max={w.abs().max():.4f}")
    
    # Decay Params
    decay = layer0.base_decay_nums
    print(f"  Decay Nums: Mean={decay.mean():.4f}, Min={decay.min():.4f}, Max={decay.max():.4f}")
    
    # Norm Gamma
    gamma = layer0.norm.gamma
    print(f"  Norm Gamma: Mean={gamma.mean():.4f}, Max={gamma.max():.4f}")
    
    # ReZero
    gate = layer0.res_gate
    print(f"  ReZero Gate: {gate.item():.4f}")
    
    # 2. Check for NaNs
    has_nan = False
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"  ❌ NaN detected in {name}")
            has_nan = True
            
    if not has_nan:
        print("  ✅ No NaNs in parameters.")

def run_inference():
    print(f"--- Inference: Mamba-Integer-L4 ({MODEL_PATH}) ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load Model
    model = MambaIntegerModel(config).to(device)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Analyze before running
    analyze_model_state(model)
    
    tokenizer = SimpleTokenizer(config['vocab_size'])
    model.eval()
    
    # 2. Prompts
    prompts = [
        "Once upon a time",
        "The quick brown fox",
    ]
    
    print("\n--- Generation ---")
    
    for prompt in prompts:
        print(f"\nPrompt: '{prompt}'")
        input_ids = torch.tensor([tokenizer.encode(prompt)]).to(device)
        
        generated = input_ids
        model.gradient_checkpointing = False
        
        with torch.no_grad():
            for i in range(100):
                logits = model(generated)
                next_token_logits = logits[:, -1, :]
                
                # Print Logits Stats for first step
                if i == 0:
                    print(f"  Logits: Mean={next_token_logits.mean():.2f}, Max={next_token_logits.max():.2f}, Min={next_token_logits.min():.2f}")
                
                # Temperature Sampling
                probs = torch.softmax(next_token_logits / 0.8, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                generated = torch.cat([generated, next_token], dim=-1)
                
        output_text = tokenizer.decode(generated[0].tolist())
        print(f"Output: {output_text}")

if __name__ == "__main__":
    run_inference()