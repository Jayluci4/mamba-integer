#!/usr/bin/env python3
"""
50M Validation: Integer vs FP32 Comparison

Purpose: Validate that BitLinear quantization gap is acceptable.
Dataset: FineWeb-Edu only (simpler, faster)

Success criteria (from docs/validation_experiments.md):
  - Loss gap < 0.15 at 30k steps: PASS
  - Loss gap 0.15-0.4: CONCERNING
  - Loss gap > 0.4: FAIL

Run: python scripts/run_50m_comparison.py
Time: ~30 hours total (can run sequentially or parallel on 2 GPUs)
"""

import os
import sys
import json
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# ============================================================================
# Speed Optimizations
# ============================================================================
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')

# ============================================================================
# Configuration
# ============================================================================

# HuggingFace token - set HF_TOKEN environment variable if needed
HF_TOKEN = os.environ.get("HF_TOKEN", None)

TOTAL_STEPS = 30000
LOG_INTERVAL = 10  # More frequent logging
SAVE_INTERVAL = 5000
COMPARE_INTERVAL = 1000  # Compare losses every 1k steps

# Success/Fail thresholds
PASS_GAP = 0.15
CONCERN_GAP = 0.4
FAIL_GAP = 0.5

# Abort if gap exceeds this at 10k steps
ABORT_GAP_10K = 0.5


# ============================================================================
# Dataset: FineWeb-Edu Only
# ============================================================================

class FineWebLoader:
    """Simple FineWeb-Edu streaming loader."""

    def __init__(self, tokenizer, seq_len, buffer_size=100000):
        from datasets import load_dataset

        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.buffer = []
        self.buffer_size = buffer_size

        print("Loading FineWeb-Edu (streaming)...")
        self.dataset = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name="sample-10BT",
            split="train",
            streaming=True,
            token=HF_TOKEN
        )
        self.iterator = iter(self.dataset)
        print("Dataset ready.")

    def _fill_buffer(self, min_tokens=None):
        if min_tokens is None:
            min_tokens = self.buffer_size

        while len(self.buffer) < min_tokens:
            try:
                item = next(self.iterator)
                text = item.get('text', '')
                if text:
                    tokens = self.tokenizer.encode(text)
                    self.buffer.extend(tokens)
            except StopIteration:
                print("Dataset exhausted, restarting...")
                self.iterator = iter(self.dataset)

    def get_batch(self, batch_size, device):
        sequences_x = []
        sequences_y = []

        for _ in range(batch_size):
            if len(self.buffer) < self.seq_len + 1:
                self._fill_buffer()

            chunk = self.buffer[:self.seq_len + 1]
            self.buffer = self.buffer[self.seq_len:]
            sequences_x.append(chunk[:-1])
            sequences_y.append(chunk[1:])

        x = torch.tensor(sequences_x, dtype=torch.long, device=device)
        y = torch.tensor(sequences_y, dtype=torch.long, device=device)
        return x, y


# ============================================================================
# FP32 Linear (for baseline)
# ============================================================================

class FPLinear(nn.Module):
    """Standard FP32 Linear layer."""

    def __init__(self, in_features, out_features, bias=False, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features) / math.sqrt(in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


# ============================================================================
# Training Function
# ============================================================================

def train_model(model_type, config, tokenizer, device, total_steps, log_file):
    """Train either Integer or FP32 model."""

    from mamba_integer_model import MambaIntegerModel

    # Create model
    if model_type == "fp32":
        # Patch BitLinear with FPLinear
        import rational_bitnet
        original_bitlinear = rational_bitnet.BitLinear
        rational_bitnet.BitLinear = FPLinear

        # Reload model module
        import importlib
        import mamba_integer_model
        importlib.reload(mamba_integer_model)
        model = mamba_integer_model.MambaIntegerModel(config).to(device)

        # Restore
        rational_bitnet.BitLinear = original_bitlinear
    else:
        model = MambaIntegerModel(config).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"{model_type.upper()} model: {total_params:,} params")

    # Note: torch.compile disabled - incompatible with custom Triton kernels

    # Dataset
    dataset = FineWebLoader(tokenizer, config['training']['seq_len'])

    # Training setup
    batch_size = config['training']['batch_size']
    grad_accum = config['training']['gradient_accumulation_steps']
    lr = config['training']['learning_rate']

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=config['training']['weight_decay']
    )

    warmup_steps = 1000
    def get_lr(step):
        if step < warmup_steps:
            return lr * step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return lr * 0.5 * (1 + math.cos(math.pi * progress))

    scaler = GradScaler()

    # Training loop
    model.train()
    losses = []
    checkpoints = {}  # step -> loss

    start_time = time.time()

    for step in range(1, total_steps + 1):
        current_lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for _ in range(grad_accum):
            x, y = dataset.get_batch(batch_size, device)

            with autocast(dtype=torch.bfloat16):
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
                loss = loss / grad_accum

            scaler.scale(loss).backward()
            accum_loss += loss.item()

        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        losses.append(accum_loss)

        # Logging
        if step % LOG_INTERVAL == 0:
            elapsed = time.time() - start_time
            avg_loss = sum(losses[-100:]) / len(losses[-100:])
            log_msg = f"[{model_type.upper()}] Step {step}/{total_steps} | Loss: {accum_loss:.4f} | Avg: {avg_loss:.4f} | LR: {current_lr:.2e} | Time: {elapsed:.1f}s"
            print(log_msg)

            with open(log_file, 'a') as f:
                f.write(log_msg + '\n')

        # Record checkpoint losses
        if step % COMPARE_INTERVAL == 0:
            avg_loss = sum(losses[-COMPARE_INTERVAL:]) / COMPARE_INTERVAL
            checkpoints[step] = avg_loss

        # Save model checkpoint
        if step % SAVE_INTERVAL == 0:
            ckpt_path = os.path.join(os.path.dirname(__file__), '..', f'50m_{model_type}_step_{step}.pt')
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'loss': accum_loss,
            }, ckpt_path)

    return checkpoints, losses


# ============================================================================
# Main Comparison
# ============================================================================

def run_comparison(run_mode='sequential'):
    """Run Integer vs FP32 comparison."""

    print("=" * 70)
    print("50M VALIDATION: Integer vs FP32 Comparison")
    print("=" * 70)
    print(f"Dataset: FineWeb-Edu only")
    print(f"Steps: {TOTAL_STEPS:,}")
    print(f"Pass threshold: gap < {PASS_GAP}")
    print(f"Fail threshold: gap > {FAIL_GAP}")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load config
    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'config_mamba_integer_50m.json')
    with open(config_path) as f:
        config = json.load(f)

    # Initialize tokenizer
    from rust_tokenizer import get_rust_tokenizer
    tokenizer = get_rust_tokenizer()
    tokenizer.load(os.path.join(os.path.dirname(__file__), '..', 'configs', 'rust_bpe_merges.txt'))

    log_file = os.path.join(os.path.dirname(__file__), '..', '50m_comparison.log')

    # Clear log
    with open(log_file, 'w') as f:
        f.write(f"50M Comparison Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n")

    # Run Integer model
    print("\n" + "=" * 70)
    print("PHASE 1: Training Integer Model")
    print("=" * 70)
    integer_checkpoints, integer_losses = train_model(
        "integer", config, tokenizer, device, TOTAL_STEPS, log_file
    )

    # Run FP32 model
    print("\n" + "=" * 70)
    print("PHASE 2: Training FP32 Model")
    print("=" * 70)

    # Need to reload modules fresh
    import importlib
    import rational_bitnet
    import mamba_integer_model
    importlib.reload(rational_bitnet)
    importlib.reload(mamba_integer_model)

    fp32_checkpoints, fp32_losses = train_model(
        "fp32", config, tokenizer, device, TOTAL_STEPS, log_file
    )

    # Compare results
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)

    print(f"\n{'Step':>8} | {'Integer':>10} | {'FP32':>10} | {'Gap':>10} | Status")
    print("-" * 60)

    final_gap = None
    for step in sorted(integer_checkpoints.keys()):
        int_loss = integer_checkpoints[step]
        fp_loss = fp32_checkpoints.get(step, 0)
        gap = int_loss - fp_loss

        if step == max(integer_checkpoints.keys()):
            final_gap = gap

        if gap < PASS_GAP:
            status = "GOOD"
        elif gap < CONCERN_GAP:
            status = "CONCERN"
        else:
            status = "BAD"

        print(f"{step:>8} | {int_loss:>10.4f} | {fp_loss:>10.4f} | {gap:>+10.4f} | {status}")

    # Final verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    if final_gap is None:
        print("[ERROR] Could not compute final gap")
        return

    print(f"Final gap: {final_gap:+.4f}")

    if final_gap < PASS_GAP:
        print(f"[PASS] Gap {final_gap:.4f} < {PASS_GAP}")
        print("BitLinear quantization is acceptable.")
        print("NEXT STEP: Scale to 125M or continue 205M with confidence")
    elif final_gap < CONCERN_GAP:
        print(f"[CONCERNING] Gap {final_gap:.4f} between {PASS_GAP} and {CONCERN_GAP}")
        print("Some capacity loss from quantization.")
        print("NEXT STEP: Decide if tradeoff is acceptable for ZK-ML goals")
    else:
        print(f"[FAIL] Gap {final_gap:.4f} > {CONCERN_GAP}")
        print("BitLinear quantization too aggressive.")
        print("NEXT STEP: Relax quantization (try 4-bit or 8-bit weights)")

    # Save comparison data
    import numpy as np
    np.savez(
        os.path.join(os.path.dirname(__file__), '..', '50m_comparison_results.npz'),
        integer_losses=np.array(integer_losses),
        fp32_losses=np.array(fp32_losses),
        integer_checkpoints=integer_checkpoints,
        fp32_checkpoints=fp32_checkpoints,
    )
    print("\nResults saved to 50m_comparison_results.npz")


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--integer-only', action='store_true',
                        help='Only run Integer model (for testing)')
    parser.add_argument('--fp32-only', action='store_true',
                        help='Only run FP32 model (for testing)')
    args = parser.parse_args()

    if args.integer_only:
        print("Running Integer model only...")
        # Simplified single-model run
        device = torch.device('cuda')
        config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'config_mamba_integer_50m.json')
        with open(config_path) as f:
            config = json.load(f)
        from rust_tokenizer import get_rust_tokenizer
        tokenizer = get_rust_tokenizer()
        tokenizer.load(os.path.join(os.path.dirname(__file__), '..', 'configs', 'rust_bpe_merges.txt'))
        train_model("integer", config, tokenizer, device, TOTAL_STEPS, "50m_integer.log")

    elif args.fp32_only:
        print("Running FP32 model only...")
        device = torch.device('cuda')
        config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'config_mamba_integer_50m.json')
        with open(config_path) as f:
            config = json.load(f)
        from rust_tokenizer import get_rust_tokenizer
        tokenizer = get_rust_tokenizer()
        tokenizer.load(os.path.join(os.path.dirname(__file__), '..', 'configs', 'rust_bpe_merges.txt'))
        train_model("fp32", config, tokenizer, device, TOTAL_STEPS, "50m_fp32.log")

    else:
        run_comparison()
