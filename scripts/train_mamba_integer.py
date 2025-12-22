
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from mamba_integer_model import MambaIntegerModel
import json
import time
import os
import sys
import numpy as np

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
    print("--- High Efficiency Mamba-Integer-L4 Training ---")
    torch.set_float32_matmul_precision('high')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Model
    model = MambaIntegerModel(config).to(device)
    # torch.compile for speed (fuses non-custom ops)
    model = torch.compile(model)
    
    # 2. Data
    bin_path = os.path.join(os.path.dirname(__file__), "tinystories_train.bin")
    if not os.path.exists(bin_path):
        print(f"Error: {bin_path} not found. Run tokenize_dataset_offline.py first.")
        return
        
    dataset = BinaryDataset(bin_path, seq_len=512)
    # num_workers=0 avoids torch.compile deadlocks and is fast enough for mmap
    # BS=4 is safer for L4 VRAM (24GB)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=0, pin_memory=True)
    
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
    gradient_accumulation_steps = 16 # Total Batch 64
    
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
    
    # 4. Loop
    model.train()
    start_time = time.time()
    optimizer.zero_grad()
    
    print(f"Starting High-Speed Loop (Effective Batch: 64)...")
    
    data_iter = iter(dataloader)
    
    for opt_step in range(total_opt_steps):
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
            torch.save(model.state_dict(), ckpt_path)
            
    print("Training Complete.")
    torch.save(model.state_dict(), "mamba_integer_final.pt")

if __name__ == "__main__":
    train()
