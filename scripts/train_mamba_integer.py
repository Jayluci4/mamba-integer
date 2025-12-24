
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
import sys
# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from mamba_integer_model import MambaIntegerModel
import json
import time
import numpy as np
import torch

# Add path for src
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# --- Config ---
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../configs/config_mamba_integer_l4.json")
with open(CONFIG_PATH, 'r') as f:
    config = json.load(f)

# --- Fast Binary Dataset ---
class BinaryDataset(Dataset):
    def __init__(self, bin_path, seq_len):
        self.seq_len = seq_len
        # Use mmap for zero-copy access
        self.data = np.memmap(bin_path, dtype=np.uint16, mode='r')
        self.n_tokens = len(self.data)
        print(f"Loaded {self.n_tokens:,} tokens from {bin_path}")

    def __len__(self):
        # We can draw many samples by random offsets
        return 1000000 # Virtual length

    def __getitem__(self, idx):
        # Random offset
        offset = np.random.randint(0, self.n_tokens - self.seq_len - 1)
        chunk = self.data[offset : offset + self.seq_len + 1].astype(np.int64)
        
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        
        # Loss masking: Padding is not relevant in packed binary, 
        # but we use 0 as document separator. Let's mask 0 targets.
        y_masked = y.clone()
        y_masked[y == 0] = -100
        
        return x, y_masked

def train():
    print("--- High Efficiency Mamba-Integer V2 Training ---")
    # Enable TF32 for Ampere+ GPUs (A100/H100)
    torch.set_float32_matmul_precision('high')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Model
    model = MambaIntegerModel(config).to(device)
    # Gradient Checkpointing is disabled because it can cause OOM with torch.compile on some GPUs
    model.gradient_checkpointing = False
    # torch.compile for speed (fuses Triton kernels and analytic gradients)
    model = torch.compile(model)
    
    # 2. Data
    bin_path = os.path.join(os.path.dirname(__file__), "tinystories_train.bin")
    if not os.path.exists(bin_path):
        print(f"Error: {bin_path} not found. Run tokenize_dataset_offline.py first.")
        return
        
    dataset = BinaryDataset(bin_path, seq_len=512)
    # BS=2 is verified safe for L4 VRAM (24GB) with Inductor
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0, pin_memory=True)
    
    # 3. Optimizer: Stiff Memory Protocol
    decay_params = []
    other_params = []
    for name, param in model.named_parameters():
        if "base_decay_nums" in name:
            decay_params.append(param)
        else:
            other_params.append(param)
            
    # Ultra-stable settings for BitNet-Mamba
    optimizer = optim.AdamW([
        {'params': decay_params, 'lr': 1e-3, 'weight_decay': 0.0},
        {'params': other_params, 'lr': 1e-4, 'weight_decay': 0.01}
    ])
    
    total_opt_steps = 15000
    gradient_accumulation_steps = 32 # Total Batch 64 (2 * 32)
    
    # Linear Warmup + Cosine
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=[1e-3, 1e-4], 
        total_steps=total_opt_steps, 
        pct_start=0.1,
        anneal_strategy='cos',
        div_factor=10.0,
        final_div_factor=100.0
    )
    
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    # Monitor
    from monitor import DyadicMonitor
    monitor = DyadicMonitor()
    
    # --- Auto-Resume Logic ---
    start_step = 0
    # Find latest checkpoint
    import glob
    ckpt_files = glob.glob(os.path.join(os.path.dirname(__file__), "mamba_integer_step_*.pt"))
    if ckpt_files:
        # Extract steps
        steps = []
        for f in ckpt_files:
            try:
                # filename format: mamba_integer_step_123.pt
                step_str = f.split("_step_")[-1].split(".")[0]
                steps.append(int(step_str))
            except: pass
            
        if steps:
            max_step = max(steps)
            latest_ckpt = os.path.join(os.path.dirname(__file__), f"mamba_integer_step_{max_step}.pt")
            print(f"Resuming from checkpoint: {latest_ckpt} (Step {max_step})")
            
            try:
                # Handle torch.compile prefix logic
                state_dict = torch.load(latest_ckpt, map_location=device)
                
                # If checkpoint is full dict (model+opt), extract. If just state_dict, load directly.
                if "model_state_dict" in state_dict:
                    # New format (future proof)
                    model_state = state_dict["model_state_dict"]
                    optimizer.load_state_dict(state_dict["optimizer_state_dict"])
                    scheduler.load_state_dict(state_dict["scheduler_state_dict"])
                    start_step = state_dict["step"] + 1 # Start next step
                else:
                    # Old format (just weights)
                    # Handle _orig_mod prefix if needed
                    new_state_dict = {}
                    for k, v in state_dict.items():
                        if k.startswith("_orig_mod."):
                            new_state_dict[k[10:]] = v
                        else:
                            new_state_dict[k] = v
                    
                    model.load_state_dict(new_state_dict, strict=False) # strict=False for safety
                    start_step = max_step + 1
                    
                    # Cold-start optimizer but advance scheduler
                    # Scheduler advance requires loop or internal state hack
                    print("Fast-forwarding scheduler...")
                    for _ in range(start_step):
                        scheduler.step()
                        
            except Exception as e:
                print(f"Warning: Failed to load checkpoint {latest_ckpt}: {e}")
                print("Starting from scratch.")

    # 4. Loop
    model.train()
    start_time = time.time()
    optimizer.zero_grad()
    
    print(f"Starting High-Speed Loop from Step {start_step}...")
    
    data_iter = iter(dataloader)
    
    for opt_step in range(start_step, total_opt_steps):
        step_loss = 0
        for _ in range(gradient_accumulation_steps):
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                x, y = next(data_iter)
                
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            
            logits = model(x)
            loss = criterion(logits.view(-1, config['vocab_size']), y.view(-1))
            loss = loss / gradient_accumulation_steps
            loss.backward()
            step_loss += loss.item()
            
        # Optimizer Step
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        if opt_step % 10 == 0:
            elapsed = time.time() - start_time
            print(f"Step {opt_step}/{total_opt_steps} | Loss: {step_loss:.4f} | Time: {elapsed:.2f}s")
            try:
                # Log stats (using uncompiled model refs via layers)
                monitor.log_step(opt_step, model, step_loss)
            except: pass
            
        if opt_step > 0 and opt_step % 500 == 0:
            ckpt_path = f"mamba_integer_step_{opt_step}.pt"
            # Save Full State for better resuming next time
            full_state = {
                'step': opt_step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict()
            }
            torch.save(full_state, ckpt_path)
            # Also save lightweight weight-only for inference
            # torch.save(model.state_dict(), f"mamba_integer_weights_{opt_step}.pt")
            
    print("Training Complete.")
    torch.save(model.state_dict(), "mamba_integer_final.pt")

if __name__ == "__main__":
    train()
