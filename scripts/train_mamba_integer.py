
import torch
# Enable TF32 BEFORE any other torch operations for maximum performance
torch.set_float32_matmul_precision('high')
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

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
        
        # Create tensors directly (faster than from_numpy for small arrays)
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        
        # Loss masking: Padding is not relevant in packed binary, 
        # but we use 0 as document separator. Let's mask 0 targets.
        y_masked = y.clone()
        y_masked[y == 0] = -100
        
        return x, y_masked

def train():
    print("--- High Efficiency Mamba-Integer V2 Training ---")
    print(f"TF32 enabled: matmul={torch.backends.cuda.matmul.allow_tf32}, cudnn={torch.backends.cudnn.allow_tf32}")
    # Enable cuDNN benchmarking for consistent input sizes
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Model
    model = MambaIntegerModel(config).to(device)
    model.gradient_checkpointing = False
    
    # --- Auto-Resume Logic (Pre-Compile) ---
    start_step = 0
    import glob
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    ckpt_files = glob.glob(os.path.join(project_root, "mamba_integer_step_*.pt"))
    
    # Define optimizer early to load its state if needed
    decay_params = []
    other_params = []
    for name, param in model.named_parameters():
        if "base_decay_nums" in name:
            decay_params.append(param)
        else:
            other_params.append(param)
    optimizer = optim.AdamW([
        {'params': decay_params, 'lr': 1e-3, 'weight_decay': 0.0},
        {'params': other_params, 'lr': 1e-4, 'weight_decay': 0.01}
    ])
    
    # Scheduler needs total_steps, so define it
    total_opt_steps = 15000
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[1e-3, 1e-4], total_steps=total_opt_steps, pct_start=0.1, anneal_strategy='cos', div_factor=10.0, final_div_factor=100.0
    )

    if ckpt_files:
        steps = []
        for f in ckpt_files:
            try:
                fname = os.path.basename(f)
                step_str = fname.split("_step_")[-1].split(".")[0]
                steps.append(int(step_str))
            except: pass
            
        if steps:
            max_step = max(steps)
            latest_ckpt = os.path.join(project_root, f"mamba_integer_step_{max_step}.pt")
            print(f"Resuming from checkpoint: {latest_ckpt} (Step {max_step})")
            
            try:
                print(f"DEBUG: Attempting to load {latest_ckpt}")
                state_dict = torch.load(latest_ckpt, map_location=device)
                
                if "model_state_dict" in state_dict:
                    print("DEBUG: Detected nested checkpoint format.")
                    model.load_state_dict(state_dict["model_state_dict"])
                    optimizer.load_state_dict(state_dict["optimizer_state_dict"])
                    scheduler.load_state_dict(state_dict["scheduler_state_dict"])
                    start_step = state_dict["step"] + 1
                else:
                    print("DEBUG: Detected flat checkpoint format (weights only).")
                    new_state_dict = {}
                    for k, v in state_dict.items():
                        # Remove torch.compile prefix if present
                        key = k.replace("_orig_mod.", "")
                        new_state_dict[key] = v
                    
                    print(f"DEBUG: First key in processed dict: {list(new_state_dict.keys())[0]}")
                    msg = model.load_state_dict(new_state_dict, strict=False)
                    print(f"DEBUG: Load Result -> {msg}", flush=True)
                    
                    if len(msg.missing_keys) > 10:
                        print("CRITICAL: Too many missing keys! Weights likely did not load.", flush=True)
                    
                    start_step = max_step + 1
                    print("DEBUG: Fast-forwarding scheduler...", flush=True)
                    for _ in range(start_step):
                        scheduler.step()
                
                print(f"DEBUG: Checkpoint loaded successfully. Resuming from step {start_step}", flush=True)
            except Exception as e:
                print(f"ERROR during loading: {e}")
                import traceback
                traceback.print_exc()

    # Compile AFTER loading weights
    # Use default mode - 'reduce-overhead' causes very slow first-step compilation
    # First step will be slow (compilation), but subsequent steps will be fast
    model = torch.compile(model)
    
    # 2. Data
    bin_path = os.path.join(os.path.dirname(__file__), "tinystories_train.bin")
    if not os.path.exists(bin_path):
        print(f"Error: {bin_path} not found.")
        return
        
    dataset = BinaryDataset(bin_path, seq_len=512)
    # Batch size optimized for A100 80GB - reduce if OOM
    # Effective batch size = batch_size * gradient_accumulation_steps
    batch_size = 8  # Reduced from 16 to avoid OOM
    gradient_accumulation_steps = 8  # Effective batch = 64
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    # Monitor
    from monitor import DyadicMonitor
    monitor = DyadicMonitor()

    # 4. Loop
    model.train()
    start_time = time.time()
    optimizer.zero_grad() # Ensure grads are zeroed after load
    
    print(f"Starting High-Speed Loop from Step {start_step}...")
    
    data_iter = iter(dataloader)
    
    for opt_step in range(start_step, total_opt_steps):
        step_loss = torch.tensor(0.0, device=device)  # Keep on GPU
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
            # CRITICAL: Don't call .item() here - it causes CPU-GPU sync!
            # Accumulate loss tensor instead, only convert to float after backward
            step_loss += loss.detach()  # Keep on GPU, detach from graph
            
        # Optimizer Step
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        
        if opt_step % 10 == 0:
            elapsed = time.time() - start_time
            # Convert accumulated loss tensor to float (only once per logging)
            loss_val = step_loss.item() if isinstance(step_loss, torch.Tensor) else step_loss
            print(f"Step {opt_step}/{total_opt_steps} | Loss: {loss_val:.4f} | Time: {elapsed:.2f}s")
            try:
                # Log stats (using uncompiled model refs via layers)
                # Note: model.layers access might need model._orig_mod.layers if compiled
                # Or just try-except
                loss_for_monitor = loss_val if isinstance(step_loss, torch.Tensor) else step_loss
                monitor.log_step(opt_step, model, loss_for_monitor)
            except: pass
            
        if opt_step > 0 and opt_step % 500 == 0:
            ckpt_path = f"mamba_integer_step_{opt_step}.pt"
            # Save Full State
            full_state = {
                'step': opt_step,
                'model_state_dict': model._orig_mod.state_dict() if hasattr(model, '_orig_mod') else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict()
            }
            torch.save(full_state, ckpt_path)
            
    print("Training Complete.")
    torch.save(model.state_dict(), "mamba_integer_final.pt")

if __name__ == "__main__":
    # Log to both file and stdout for visibility
    log_file = open("training_INTERNAL_LOG.log", "w", buffering=1)
    
    class TeeOutput:
        def __init__(self, *files):
            self.files = files
        def write(self, obj):
            for f in self.files:
                f.write(obj)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()
        def isatty(self):
            # Check if any of the files is a TTY (usually stdout)
            return any(hasattr(f, 'isatty') and f.isatty() for f in self.files)
        def fileno(self):
            # Return the fileno of the first file (usually stdout)
            if self.files and hasattr(self.files[0], 'fileno'):
                return self.files[0].fileno()
            raise OSError("fileno not available")
    
    tee = TeeOutput(sys.stdout, log_file)
    sys.stdout = tee
    sys.stderr = tee
    train()
