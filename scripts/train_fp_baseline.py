#!/usr/bin/env python3
"""
FP Baseline Training Script

Purpose: Train the same Mamba architecture with standard FP Linear layers
         instead of BitLinear to establish a baseline for comparison.

This helps diagnose: Is the model architecture sound, or is BitLinear causing issues?

Expected outcome:
- If FP baseline loss is similar to BitLinear: Architecture is fine, BitLinear works
- If FP baseline loss is much lower: BitLinear may be too aggressive

Run: python scripts/train_fp_baseline.py
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

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Monkey-patch BitLinear BEFORE importing the model
# This replaces BitLinear with standard nn.Linear
class FPLinear(nn.Module):
    """Standard FP Linear layer with same interface as BitLinear."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        activation_bits: int = 8,  # Ignored - for interface compatibility
        is_output_layer: bool = False,  # Ignored
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Standard initialization
        self.weight = nn.Parameter(torch.randn(out_features, in_features) / math.sqrt(in_features))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)

# Patch the module before import
import rational_bitnet
rational_bitnet.BitLinear = FPLinear

# Now import the model (it will use FPLinear instead of BitLinear)
from mamba_integer_model import MambaIntegerModel
from rust_tokenizer import get_rust_tokenizer

# HuggingFace token - set HF_TOKEN environment variable if needed
HF_TOKEN = os.environ.get("HF_TOKEN", None)

# Training config - match the main training script
TOTAL_STEPS = 10000  # Short run for comparison
LOG_INTERVAL = 10
SAVE_INTERVAL = 2000
EVAL_INTERVAL = 500

def get_streaming_batch(dataloader_iter, dataloader, batch_size, seq_len, device):
    """Get a batch from streaming dataloader."""
    try:
        batch = next(dataloader_iter)
    except StopIteration:
        dataloader_iter = iter(dataloader)
        batch = next(dataloader_iter)

    x = batch['input_ids'][:, :-1].to(device)
    y = batch['input_ids'][:, 1:].to(device)
    return x, y, dataloader_iter


class SimpleStreamingDataset:
    """Simple streaming dataset for FP baseline - uses FineWeb-Edu only."""

    def __init__(self, tokenizer, seq_len=1024, buffer_size=100000):
        from datasets import load_dataset
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.buffer = []
        self.buffer_size = buffer_size

        print("Loading FineWeb-Edu dataset (streaming)...")
        self.dataset = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name="sample-10BT",
            split="train",
            streaming=True,
            token=HF_TOKEN
        )
        self.iterator = iter(self.dataset)

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
                self.iterator = iter(self.dataset)

    def get_batch(self, batch_size, device):
        """Get a batch of sequences."""
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


def train():
    print("=" * 60)
    print("FP BASELINE TRAINING")
    print("=" * 60)
    print("Purpose: Compare standard FP Linear vs BitLinear")
    print("=" * 60)

    # Load config
    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'config_mamba_integer_l4.json')
    with open(config_path) as f:
        config = json.load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Initialize model
    print("\nInitializing FP Baseline Model...")
    model = MambaIntegerModel(config).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Verify we're using FPLinear not BitLinear
    linear_type = type(model.lm_head).__name__
    print(f"Linear layer type: {linear_type}")
    assert linear_type == "FPLinear", f"Expected FPLinear, got {linear_type}"

    # Initialize tokenizer
    tokenizer = get_rust_tokenizer()
    tokenizer.load(os.path.join(os.path.dirname(__file__), '..', 'configs', 'rust_bpe_merges.txt'))

    # Initialize dataset
    seq_len = config['training']['seq_len']
    batch_size = config['training']['batch_size']
    grad_accum = config['training']['gradient_accumulation_steps']

    print(f"\nTraining config:")
    print(f"  Seq length: {seq_len}")
    print(f"  Batch size: {batch_size}")
    print(f"  Grad accumulation: {grad_accum}")
    print(f"  Effective batch: {batch_size * grad_accum}")
    print(f"  Total steps: {TOTAL_STEPS}")

    dataset = SimpleStreamingDataset(tokenizer, seq_len=seq_len)

    # Optimizer - match main training
    lr = config['training']['learning_rate']
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=0.1
    )

    # LR scheduler - cosine with warmup
    warmup_steps = 500
    def get_lr(step):
        if step < warmup_steps:
            return lr * step / warmup_steps
        progress = (step - warmup_steps) / (TOTAL_STEPS - warmup_steps)
        return lr * 0.5 * (1 + math.cos(math.pi * progress))

    # AMP scaler
    scaler = GradScaler()

    # Training loop
    print("\nStarting training...")
    model.train()

    log_file = os.path.join(os.path.dirname(__file__), '..', 'fp_baseline_training.log')

    losses = []
    start_time = time.time()

    for step in range(1, TOTAL_STEPS + 1):
        # Update LR
        current_lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        # Gradient accumulation
        optimizer.zero_grad()
        accum_loss = 0.0

        for _ in range(grad_accum):
            x, y = dataset.get_batch(batch_size, device)

            with autocast(dtype=torch.bfloat16):
                logits = model(x)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    y.view(-1)
                )
                loss = loss / grad_accum

            scaler.scale(loss).backward()
            accum_loss += loss.item()

        # Gradient clipping
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        scaler.step(optimizer)
        scaler.update()

        losses.append(accum_loss)

        # Logging
        if step % LOG_INTERVAL == 0:
            elapsed = time.time() - start_time
            avg_loss = sum(losses[-100:]) / len(losses[-100:])

            log_msg = f"Step {step}/{TOTAL_STEPS} | Loss: {accum_loss:.4f} | Avg: {avg_loss:.4f} | LR: {current_lr:.2e} | GradNorm: {grad_norm:.2f} | Time: {elapsed:.1f}s"
            print(log_msg)

            with open(log_file, 'a') as f:
                f.write(log_msg + '\n')

        # Save checkpoint
        if step % SAVE_INTERVAL == 0:
            ckpt_path = os.path.join(os.path.dirname(__file__), '..', f'fp_baseline_step_{step}.pt')
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': accum_loss,
                'losses': losses,
            }, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

    # Final summary
    print("\n" + "=" * 60)
    print("FP BASELINE TRAINING COMPLETE")
    print("=" * 60)
    print(f"Final loss: {losses[-1]:.4f}")
    print(f"Avg loss (last 500): {sum(losses[-500:]) / len(losses[-500:]):.4f}")
    print(f"Min loss: {min(losses):.4f}")
    print(f"Total time: {time.time() - start_time:.1f}s")

    # Save final losses for comparison
    import numpy as np
    np.save(os.path.join(os.path.dirname(__file__), '..', 'fp_baseline_losses.npy'), np.array(losses))
    print(f"Saved losses to fp_baseline_losses.npy")


if __name__ == '__main__':
    train()
