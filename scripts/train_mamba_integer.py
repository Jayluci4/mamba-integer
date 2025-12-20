
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from datasets import load_dataset
from mamba_integer_model import MambaIntegerModel
import json
import time
import os

# --- Config ---
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config_mamba_integer_l4.json")
with open(CONFIG_PATH, 'r') as f:
    config = json.load(f)

# --- Tokenizer (Simple Character Level for Proof of Concept) ---
# Or load a pretrained one. Since vocab is 4096, we can use a BPE.
# For simplicity and speed, let's use a char-level tokenizer if we don't have a BPE handy.
# Or better: Use GPT-2 tokenizer and limit vocab?
# Let's use a simple custom tokenizer for TinyStories.

class SimpleTokenizer:
    def __init__(self, vocab_size=4096):
        self.vocab_size = vocab_size
        self.char_to_idx = {}
        self.idx_to_char = {}
        # Basic ASCII
        for i in range(256):
            c = chr(i)
            self.char_to_idx[c] = i
            self.idx_to_char[i] = c
            
    def encode(self, text):
        # Fallback to ASCII bytes
        return [min(b, self.vocab_size - 1) for b in text.encode('utf-8')]
        
    def decode(self, ids):
        return bytes([i for i in ids if i < 256]).decode('utf-8', errors='ignore')

def train():
    print("--- Training Mamba-Integer-L4 on TinyStories ---")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # 1. Model
    model = MambaIntegerModel(config).to(device)
    print(f"Model Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 2. Data
    print("Loading TinyStories...")
    dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    tokenizer = SimpleTokenizer(config['vocab_size'])
    
    def collate_fn(batch):
        # Dynamic padding or fixed length
        max_len = 512
        input_ids = []
        for item in batch:
            ids = tokenizer.encode(item['text'])[:max_len]
            # Pad
            if len(ids) < max_len:
                ids += [0] * (max_len - len(ids))
            input_ids.append(torch.tensor(ids))
        return torch.stack(input_ids)
        
    dataloader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn) # Reduced from 8
    # Streaming dataset is iterable.
    # Convert to iterable dataloader logic.
    
    # 3. Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # 4. Loop
    model.train()
    total_steps = 15000 # Increased for full training
    start_time = time.time()
    
    print("Starting Training Loop...")
    for step, batch in enumerate(dataloader):
        if step >= total_steps: break
        
        # Prepare inputs
        inputs = batch.to(device)
        targets = inputs.clone()
        
        # Forward
        logits = model(inputs)
        
        # Shift for loss
        # logits: [B, L, V], targets: [B, L]
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
            
    print("Training Complete.")
    
    # 5. Generation Test
    print("\n--- Generation Test ---")
    model.eval()
    prompt = "Once upon a time"
    input_ids = torch.tensor([tokenizer.encode(prompt)]).to(device)
    
    generated = input_ids
    for _ in range(50):
        logits = model(generated)
        next_token_logits = logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=-1)
        
    print(f"Prompt: {prompt}")
    print(f"Output: {tokenizer.decode(generated[0].tolist())}")

if __name__ == "__main__":
    train()
