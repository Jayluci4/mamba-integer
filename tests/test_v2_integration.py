
import torch
import torch.nn as nn
import torch.optim as optim
import os
import sys
import time

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../gemma-intelligent/conv/src/bitnet-odp/src'))

from mamba_integer_model import MambaIntegerModel
import rational_bitnet
import mamba_integer_model

def test_v2_integration_stability():
    print("--- V2 Integration & Stability Test (with torch.compile) ---")
    device = "cuda"
    
    # 1. Setup V2
    rational_bitnet.BITNET_TRITON_AVAILABLE = True
    mamba_integer_model.BITSHIFT_TRITON_AVAILABLE = True
    
    # 2. Mini Model Config
    config = {
        'vocab_size': 100,
        'd_model': 128,
        'n_layer': 2,
        'ssm_cfg': {
            'd_state': 16,
            'dt_rank': 32
        }
    }
    
    model = MambaIntegerModel(config).to(device)
    
    # 3. Compile Model
    print("Compiling model (Inductor)...")
    compiled_model = torch.compile(model)
    
    optimizer = optim.AdamW(compiled_model.parameters(), lr=1e-3)
    
    # 4. Synthetic Training Loop
    B, L = 4, 64
    print(f"Starting 50 steps of synthetic training...")
    
    losses = []
    start_time = time.time()
    
    for i in range(50):
        input_ids = torch.randint(0, config['vocab_size'], (B, L), device=device)
        targets = torch.randint(0, config['vocab_size'], (B, L), device=device)
        
        optimizer.zero_grad()
        
        # Forward
        logits = compiled_model(input_ids)
        loss = nn.functional.cross_entropy(logits.view(-1, config['vocab_size']), targets.view(-1))
        
        # Backward
        loss.backward()
        
        # Check for NaNs
        if torch.isnan(loss):
            print(f"❌ NaN Loss detected at step {i}")
            sys.exit(1)
            
        # Grad Clip & Step
        torch.nn.utils.clip_grad_norm_(compiled_model.parameters(), 1.0)
        optimizer.step()
        
        losses.append(loss.item())
        
        if i == 0:
            print(f"  Step 0 complete (Compilation step). Time: {time.time() - start_time:.2f}s")
        if (i + 1) % 10 == 0:
            print(f"  Step {i+1}/50 | Loss: {loss.item():.4f}")

    end_time = time.time()
    avg_step_time = (end_time - start_time) / 50
    
    print(f"\nTraining Complete.")
    print(f"Average Step Time: {avg_step_time*1000:.2f} ms")
    
    # 5. Final Checks
    if losses[-1] < losses[0] * 1.1: # Allow some noise in synthetic, but shouldn't explode
        print("✅ Stability Check: PASSED (Loss is stable/decreasing)")
    else:
        print("❌ Stability Check: FAILED (Loss exploded)")
        sys.exit(1)
        
    print("✅ Integration Test: PASSED")

if __name__ == "__main__":
    test_v2_integration_stability()
