
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

# Add path for rational_bitnet
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../bitnet-odp/src'))
# Add path for src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

# --- Config ---
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../../configs/config_mamba_integer_l4.json")
# Use the pre-trained base model
PRETRAINED_PATH = "mamba_integer_final.pt"

if not os.path.exists(CONFIG_PATH):
    # Fallback to local config if moved
    CONFIG_PATH = "config_mamba_integer_l4.json"

with open(CONFIG_PATH, 'r') as f:
    config = json.load(f)

# --- Tokenizer (Same as Pretraining) ---
class SimpleTokenizer:
    def __init__(self, vocab_size=4096):
        self.vocab_size = vocab_size
        
    def encode(self, text):
        return [min(b, self.vocab_size - 1) for b in text.encode('utf-8')]
        
    def decode(self, ids):
        return bytes([i for i in ids if i < 256]).decode('utf-8', errors='ignore')

def train_sft():
    print("--- DyadicSFT: Instruction Tuning ---")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Load Pretrained Model
    print(f"Loading Base Model from {PRETRAINED_PATH}...")
    model = MambaIntegerModel(config).to(device)
    if os.path.exists(PRETRAINED_PATH):
        model.load_state_dict(torch.load(PRETRAINED_PATH, map_location=device))
        print("Base weights loaded.")
    else:
        print("Warning: Pretrained checkpoint not found. Starting from scratch (not recommended for SFT).")
    
    # 2. Data (Alpaca)
    print("Loading Alpaca Dataset...")
    try:
        dataset = load_dataset("tatsu-lab/alpaca", split="train")
    except:
        print("Alpaca not found, falling back to dummy data for POC.")
        dataset = [{"instruction": "Hi", "output": "Hello"}] * 100

    tokenizer = SimpleTokenizer(config['vocab_size'])
    
    def collate_fn(batch):
        max_len = 512
        input_ids_batch = []
        labels_batch = []
        
        for item in batch:
            # Format: User: {inst}\n\nAssistant: {out}
            inst = f"User: {item['instruction']}\n\nAssistant: "
            resp = item['output']
            
            enc_inst = tokenizer.encode(inst)
            enc_resp = tokenizer.encode(resp)
            
            full_ids = enc_inst + enc_resp
            
            # Create Labels: -100 for instruction (ignore), token IDs for response
            full_labels = [-100] * len(enc_inst) + enc_resp
            
            # Truncate / Pad
            if len(full_ids) > max_len:
                full_ids = full_ids[:max_len]
                full_labels = full_labels[:max_len]
            else:
                pad_len = max_len - len(full_ids)
                full_ids += [0] * pad_len
                full_labels += [-100] * pad_len
                
            input_ids_batch.append(torch.tensor(full_ids))
            labels_batch.append(torch.tensor(full_labels))
            
        return torch.stack(input_ids_batch), torch.stack(labels_batch)
        
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn, shuffle=True)
    
    # 3. Optimizer (Lower LR for Fine-tuning)
    optimizer = optim.AdamW(model.parameters(), lr=1e-5)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    # 4. Loop
    model.train()
    # SFT usually fewer steps (1-3 epochs). For POC: 100 steps.
    total_steps = 100 
    
    print("Starting SFT Loop...")
    for step, (inputs, labels) in enumerate(dataloader):
        if step >= total_steps: break
        
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        logits = model(inputs)
        
        # Shift
        shift_logits = logits[:, :-1, :].contiguous().view(-1, config['vocab_size'])
        shift_labels = labels[:, 1:].contiguous().view(-1)
        
        loss = criterion(shift_logits, shift_labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if step % 10 == 0:
            print(f"SFT Step {step}/{total_steps} | Loss: {loss.item():.4f}")
            
    print("SFT Complete.")
    torch.save(model.state_dict(), "mamba_integer_sft.pt")
    
    # 5. Chat Test
    print("\n--- Chat Test ---")
    model.eval()
    prompt = "User: Tell me a story.\n\nAssistant: "
    input_ids = torch.tensor([tokenizer.encode(prompt)]).to(device)
    
    generated = input_ids
    for _ in range(50):
        logits = model(generated)
        next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        generated = torch.cat([generated, next_token], dim=-1)
        
    print(f"{tokenizer.decode(generated[0].tolist())}")

if __name__ == "__main__":
    train_sft()
