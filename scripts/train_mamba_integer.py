import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
from mamba_integer_model import MambaIntegerModel
import json
import time
import os
import sys

# Add path for src
sys.path.insert(0, os.path.dirname(__file__))
# Add parent for dyadic_hippo etc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# --- Config ---
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../configs/config_mamba_integer_l4.json")
with open(CONFIG_PATH, 'r') as f:
    config = json.load(f)

# --- Rust Tokenizer ---
from rust_tokenizer import get_rust_tokenizer
MERGES_PATH = os.path.join(os.path.dirname(__file__), "../configs/rust_bpe_merges.txt")
rust_tokenizer = get_rust_tokenizer()
if os.path.exists(MERGES_PATH):
    print(f"Loading merges from {MERGES_PATH}")
    rust_tokenizer.load(MERGES_PATH)
else:
    print("WARNING: merges file not found. Running with empty vocab!")

def train():
    print("--- Training Mamba-Integer-L4 on TinyStories (Rust BPE) ---")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # 1. Model
    model = MambaIntegerModel(config).to(device)
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 2. Data
    print("Loading TinyStories...")
    dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    
    def collate_fn(batch):
        max_len = 512
        input_ids = []
        for item in batch:
            ids = rust_tokenizer.encode(item['text'])[:max_len]
            # Pad
            if len(ids) < max_len:
                ids += [0] * (max_len - len(ids))
            input_ids.append(torch.tensor(ids))
        return torch.stack(input_ids)
        
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    
    # 3. Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # Monitor
    from monitor import DyadicMonitor
    monitor = DyadicMonitor()
    
    # 4. Loop
    model.train()
    total_steps = 15000 
    start_time = time.time()
    
    print("Starting Training Loop...")
    for step, batch in enumerate(dataloader):
        if step >= total_steps: break
        
        inputs = batch.to(device)
        targets = inputs.clone()
        
        # Forward
        logits = model(inputs)
        
        # Shift for loss
        shift_logits = logits[:, :-1, :].contiguous().view(-1, config['vocab_size'])
        shift_targets = targets[:, 1:].contiguous().view(-1)
        
        loss = criterion(shift_logits, shift_targets)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 10 == 0:
            elapsed = time.time() - start_time
            print(f"Step {step}/{total_steps} | Loss: {loss.item():.4f} | Time: {elapsed:.2f}s")
            monitor.log_step(step, model, loss.item())
            
        if step > 0 and step % 500 == 0:
            ckpt_path = f"mamba_integer_step_{step}.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")
            
    print("Training Complete.")
    torch.save(model.state_dict(), "mamba_integer_final.pt")
    
    # 5. Generation Test
    print("\n--- Generation Test ---")
    model.eval()
    prompt = "Once upon a time"
    input_ids = torch.tensor([rust_tokenizer.encode(prompt)]).to(device)
    
    generated = input_ids
    for _ in range(50):
        logits = model(generated)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=-1)
        
    print(f"Prompt: {prompt}")
    print(f"Output: {rust_tokenizer.decode(generated[0].tolist())}")

if __name__ == "__main__":
    train()