import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import json
import time
import numpy as np
import glob

# Add src directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
sys.path.insert(0, "/home/jayantlohia16/experiment/gemma-intelligent/conv/src/bitnet-odp/src")

from mamba_integer_model import MambaIntegerModel

# --- Config ---
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../configs/config_mamba_integer_l4.json")
with open(CONFIG_PATH, 'r') as f:
    config = json.load(f)

# --- Fast Binary Dataset ---
class BinaryDataset(Dataset):
    def __init__(self, bin_path, seq_len):
        self.seq_len = seq_len
        self.data = np.memmap(bin_path, dtype=np.uint16, mode='r')
        self.n_tokens = len(self.data)
        print(f"Loaded {self.n_tokens:,} tokens from {bin_path}")

    def __len__(self):
        return 1000000 # Virtual length

    def __getitem__(self, idx):
        offset = np.random.randint(0, self.n_tokens - self.seq_len - 1)
        chunk = self.data[offset : offset + self.seq_len + 1].astype(np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        y_masked = y.clone()
        y_masked[y == 0] = -100
        return x, y_masked


def validate_optimizer_state(optimizer, checkpoint_opt_state, expected_params):
    """Validate that optimizer state is complete and consistent."""
    opt_state = checkpoint_opt_state.get('state', {})
    param_groups = checkpoint_opt_state.get('param_groups', [])

    # Count total params in checkpoint
    total_checkpoint_params = sum(len(pg.get('params', [])) for pg in param_groups)
    state_entries = len(opt_state)

    print(f"  Checkpoint optimizer: {total_checkpoint_params} params, {state_entries} state entries")
    print(f"  Current model: {expected_params} params")

    # Check if all params have state (they should after training)
    if state_entries < expected_params * 0.9:  # Allow 10% tolerance
        print(f"  WARNING: Only {state_entries}/{expected_params} params have optimizer state!")
        print(f"  This indicates corrupted/incomplete optimizer state.")
        return False

    # Check step consistency across params
    steps = [v.get('step', 0) for v in opt_state.values() if isinstance(v.get('step'), (int, float, torch.Tensor))]
    if steps:
        # Convert tensors to floats
        steps = [s.item() if isinstance(s, torch.Tensor) else s for s in steps]
        min_step, max_step = min(steps), max(steps)
        if max_step - min_step > 100:
            print(f"  WARNING: Inconsistent step counts in optimizer state: min={min_step}, max={max_step}")
            return False
        print(f"  Optimizer step count: {max_step}")

    return True


def validate_checkpoint_integrity(checkpoint, model, device):
    """Comprehensive checkpoint validation."""
    issues = []

    # 1. Check model state for NaN/Inf
    model_sd = checkpoint.get('model_state_dict', checkpoint)
    nan_params = []
    inf_params = []
    for k, v in model_sd.items():
        if isinstance(v, torch.Tensor):
            if torch.isnan(v).any():
                nan_params.append(k)
            if torch.isinf(v).any():
                inf_params.append(k)

    if nan_params:
        issues.append(f"NaN in parameters: {nan_params[:5]}...")
    if inf_params:
        issues.append(f"Inf in parameters: {inf_params[:5]}...")

    # 2. Check optimizer state
    if 'optimizer_state_dict' in checkpoint:
        expected_params = len(list(model.parameters()))
        if not validate_optimizer_state(None, checkpoint['optimizer_state_dict'], expected_params):
            issues.append("Optimizer state incomplete or corrupted")

    # 3. Check step consistency
    if 'step' in checkpoint and 'scheduler_state_dict' in checkpoint:
        ckpt_step = checkpoint['step']
        sched_epoch = checkpoint['scheduler_state_dict'].get('last_epoch', -1)
        if abs(ckpt_step - sched_epoch) > 10:
            issues.append(f"Step mismatch: checkpoint step={ckpt_step}, scheduler epoch={sched_epoch}")

    return issues


def train():
    print("--- High Efficiency Mamba-Integer V2 Training ---")
    torch.set_float32_matmul_precision('high')

    # Enable TF32 for better performance
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print(f"TF32 enabled: matmul={torch.backends.cuda.matmul.allow_tf32}, cudnn={torch.backends.cudnn.allow_tf32}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Model
    model = MambaIntegerModel(config).to(device)
    model.train()
    num_params = len(list(model.parameters()))
    print(f"Model has {num_params} parameter tensors")

    # 2. Optimizer & Scheduler Setup (before torch.compile)
    decay_params = []
    other_params = []
    for name, param in model.named_parameters():
        if "base_decay_nums" in name:
            decay_params.append(param)
        else:
            other_params.append(param)

    print(f"Decay params: {len(decay_params)}, Other params: {len(other_params)}")

    # torch.compile disabled - conflicts with custom Triton kernels
    # TODO: Re-enable after fixing kernel compatibility
    # print("Applying torch.compile (this may take a few minutes on first run)...")
    # model = torch.compile(model, mode='default')
    print("Running without torch.compile (custom Triton kernels active)")

    # Learning rates
    optimizer = optim.AdamW([
        {'params': decay_params, 'lr': 5e-3, 'weight_decay': 0.0},
        {'params': other_params, 'lr': 5e-4, 'weight_decay': 0.01}
    ])

    total_opt_steps = 15000
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[5e-3, 5e-4], total_steps=total_opt_steps, pct_start=0.1, anneal_strategy='cos', div_factor=10.0, final_div_factor=100.0
    )

    # --- Auto-Resume Logic (with validation) ---
    start_step = 0
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    ckpt_files = glob.glob(os.path.join(project_root, "mamba_integer_step_*.pt"))

    if ckpt_files:
        # Sort by step number and try checkpoints from newest to oldest
        ckpt_with_steps = []
        for f in ckpt_files:
            try:
                s = int(os.path.basename(f).replace(".pt", "").split("_")[-1])
                ckpt_with_steps.append((s, f))
            except:
                pass

        ckpt_with_steps.sort(reverse=True)  # Newest first

        checkpoint_loaded = False
        for step_num, ckpt_path in ckpt_with_steps:
            print(f"\nTrying checkpoint: {os.path.basename(ckpt_path)} (Step {step_num})")

            try:
                checkpoint = torch.load(ckpt_path, map_location=device)
                print(f"DEBUG: Attempting to load {ckpt_path}")

                # Determine checkpoint format
                is_nested = "model_state_dict" in checkpoint

                if is_nested:
                    print("DEBUG: Detected nested checkpoint format.")

                    # Validate checkpoint integrity
                    issues = validate_checkpoint_integrity(checkpoint, model, device)
                    if issues:
                        print(f"WARNING: Checkpoint has issues:")
                        for issue in issues:
                            print(f"  - {issue}")

                        # If optimizer is corrupted, skip this checkpoint
                        if any("Optimizer" in i or "optimizer" in i for i in issues):
                            print(f"Skipping checkpoint due to optimizer issues. Trying earlier checkpoint...")
                            continue

                    # Load model state
                    model.load_state_dict(checkpoint["model_state_dict"], strict=False)

                    # Load optimizer state with explicit error handling
                    try:
                        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                        print("DEBUG: Optimizer and scheduler loaded successfully")
                    except Exception as e:
                        print(f"ERROR loading optimizer/scheduler: {e}")
                        print("Creating fresh optimizer/scheduler and warming up...")
                        # Recreate optimizer and scheduler
                        optimizer = optim.AdamW([
                            {'params': decay_params, 'lr': 5e-3, 'weight_decay': 0.0},
                            {'params': other_params, 'lr': 5e-4, 'weight_decay': 0.01}
                        ])
                        scheduler = torch.optim.lr_scheduler.OneCycleLR(
                            optimizer, max_lr=[5e-3, 5e-4], total_steps=total_opt_steps,
                            pct_start=0.1, anneal_strategy='cos', div_factor=10.0, final_div_factor=100.0
                        )
                        # Fast-forward scheduler
                        for _ in range(step_num + 1):
                            scheduler.step()
                        print(f"Scheduler fast-forwarded to step {step_num + 1}")

                    start_step = checkpoint["step"] + 1

                else:
                    # Flat checkpoint (model weights only, likely from torch.compile)
                    print("DEBUG: Detected flat checkpoint format (model weights only).")
                    state_dict = checkpoint

                    # Remove _orig_mod. prefix from torch.compile
                    new_state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
                    model.load_state_dict(new_state_dict, strict=False)

                    # Fresh optimizer, fast-forward scheduler
                    start_step = step_num + 1
                    print(f"Fast-forwarding scheduler to step {start_step}...")
                    for _ in range(start_step):
                        scheduler.step()
                    print("NOTE: Using fresh optimizer (no momentum history from checkpoint)")

                print(f"DEBUG: Checkpoint loaded successfully. Resuming from step {start_step}")
                checkpoint_loaded = True
                break

            except Exception as e:
                print(f"ERROR loading checkpoint {ckpt_path}: {e}")
                continue

        if not checkpoint_loaded:
            print("WARNING: No valid checkpoint found. Starting from scratch.")
            start_step = 0

    # 3. Data
    bin_path = os.path.join(os.path.dirname(__file__), "tinystories_train.bin")
    dataset = BinaryDataset(bin_path, seq_len=512)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0, pin_memory=True)

    gradient_accumulation_steps = 32
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    # 4. Training Loop with NaN detection
    model.train()
    start_time = time.time()
    last_log_time = time.time()
    optimizer.zero_grad()

    # Track loss history for NaN detection
    nan_streak = 0
    max_nan_streak = 3  # Stop after 3 consecutive NaN losses
    last_valid_loss = float('inf')

    print(f"Starting High-Speed Loop from Step {start_step}...")
    data_iter = iter(dataloader)

    for opt_step in range(start_step, total_opt_steps):
        step_loss = torch.tensor(0.0, device=device)

        for _ in range(gradient_accumulation_steps):
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                x, y = next(data_iter)

            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits.view(-1, config['vocab_size']), y.view(-1))
            loss = loss / gradient_accumulation_steps
            loss.backward()
            step_loss += loss.detach()

        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        # Logging and NaN detection
        elapsed = time.time() - last_log_time
        last_log_time = time.time()
        loss_val = step_loss.item()

        # NaN/Inf detection
        if torch.isnan(step_loss) or torch.isinf(step_loss) or loss_val > 1e6:
            nan_streak += 1
            print(f"Step {opt_step}/{total_opt_steps} | Loss: {loss_val:.4f} | Time: {elapsed:.2f}s | ALERT: Invalid loss! Streak: {nan_streak}/{max_nan_streak}", flush=True)

            if nan_streak >= max_nan_streak:
                print(f"\nFATAL: {max_nan_streak} consecutive invalid losses. Stopping training.")
                print(f"Last valid loss was: {last_valid_loss:.4f}")
                print("Consider resuming from an earlier checkpoint.")
                # Save emergency checkpoint
                emergency_path = os.path.join(project_root, f"mamba_integer_EMERGENCY_step_{opt_step}.pt")
                torch.save({
                    'step': opt_step,
                    'model_state_dict': model.state_dict(),
                    'note': 'Emergency save due to NaN loss'
                }, emergency_path)
                print(f"Emergency checkpoint saved to: {emergency_path}")
                return
        else:
            nan_streak = 0
            last_valid_loss = loss_val

        # Regular logging
        current_lr = scheduler.get_last_lr()[0]
        print(f"Step {opt_step}/{total_opt_steps} | Loss: {loss_val:.4f} | LR: {current_lr:.2e} | GradNorm: {grad_norm:.2f} | Time: {elapsed:.2f}s", flush=True)

        # Checkpoint saving
        if opt_step > 0 and opt_step % 500 == 0:
            ckpt_path = os.path.join(project_root, f"mamba_integer_step_{opt_step}.pt")

            # Validate before saving
            has_nan = any(torch.isnan(p).any() for p in model.parameters())
            if has_nan:
                print(f"WARNING: Model has NaN parameters at step {opt_step}. Not saving checkpoint.")
            else:
                torch.save({
                    'step': opt_step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': loss_val
                }, ckpt_path)
                print(f"Checkpoint saved: {ckpt_path}")

    # Final save
    final_path = os.path.join(project_root, "mamba_integer_final.pt")
    torch.save(model.state_dict(), final_path)
    print(f"Training complete. Final model saved to: {final_path}")

if __name__ == "__main__":
    train()
