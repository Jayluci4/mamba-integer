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
    
    # Gradient Accumulation
    gradient_accumulation_steps = 16 # Effective batch size = 2 * 16 = 32
    
    def collate_fn(batch):
        max_len = 512
        input_ids = []
        target_ids = []
        
        for item in batch:
            ids = rust_tokenizer.encode(item['text'])[:max_len]
            # Create inputs and targets
            # Inputs: [A, B, C]
            # Targets: [B, C, EOS] (but we shift later, so here just pad)
            
            length = len(ids)
            if length < max_len:
                # Pad inputs with 0
                pad_len = max_len - length
                inp = ids + [0] * pad_len
                # Pad targets with -100 (Ignore Index)
                tar = ids + [-100] * pad_len
            else:
                inp = ids
                tar = ids
                
            input_ids.append(torch.tensor(inp))
            target_ids.append(torch.tensor(tar))
            
        return torch.stack(input_ids), torch.stack(target_ids)
        
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn, num_workers=0)
    
    # 3. Optimizer
    # Separate parameters for specific learning rates
    decay_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if "base_decay_nums" in name:
            decay_params.append(param)
        else:
            other_params.append(param)
            
    optimizer = optim.AdamW([
        {'params': decay_params, 'lr': 5e-2, 'weight_decay': 0.0}, # Boosted LR (x500 relative to 1e-4) & NO Decay
        {'params': other_params, 'lr': 1e-4, 'weight_decay': 0.01}
    ])
    
    # Scheduler: Warmup + Cosine
    # 15000 steps * 16 accum = 240,000 micro steps? No, we step optimizer every 16.
    # Total optimizer steps = 15000. 
    # Total micro steps = 15000 * 16.
    total_opt_steps = 15000
    
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=[5e-2, 1e-4], 
        total_steps=total_opt_steps, 
        pct_start=0.06, # ~1000 steps
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
    
    print(f"Starting Training Loop (Accumulation: {gradient_accumulation_steps})...")
    
    optimizer.zero_grad()
    
    for step, batch in enumerate(dataloader):
        if step >= total_opt_steps * gradient_accumulation_steps: break
        
        # Unpack
        inputs, targets_raw = batch
        inputs = inputs.to(device)
        targets_raw = targets_raw.to(device)
        
        logits = model(inputs)
        
        # Shift for loss
        # logits: [B, L, V] -> predict next token
        # targets: [B, L]
        # Shift: logits[:, :-1] predicts targets[:, 1:]
        
        shift_logits = logits[:, :-1, :].contiguous().view(-1, config['vocab_size'])
        shift_targets = targets_raw[:, 1:].contiguous().view(-1)
        
        loss = criterion(shift_logits, shift_targets)
        loss = loss / gradient_accumulation_steps # Scale loss
        
        loss.backward()
        
        if (step + 1) % gradient_accumulation_steps == 0:
            # Clip Gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            # Log every 10 optimizer steps
            opt_step = (step + 1) // gradient_accumulation_steps
            if opt_step % 10 == 0:
                elapsed = time.time() - start_time
                # Re-scale loss for display
                print(f"Step {opt_step}/{total_opt_steps} | Loss: {loss.item() * gradient_accumulation_steps:.4f} | Time: {elapsed:.2f}s")
                try:
                    monitor.log_step(opt_step, model, loss.item() * gradient_accumulation_steps)
                except Exception as e:
                    print(f"Monitor error: {e}")
            
            if opt_step > 0 and opt_step % 500 == 0:
                ckpt_path = f"mamba_integer_step_{opt_step}.pt"
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