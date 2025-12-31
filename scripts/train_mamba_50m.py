#!/usr/bin/env python3
"""
Mamba-Integer 50M Training Script

Purpose: Fast validation of integer-only architecture
Target: Coherent outputs in ~32 hours (75k steps)

Model: 46.4M parameters
  - d_model: 640
  - n_layer: 20
  - n_heads: 10

Expected tokens for coherence: ~3B (75k steps * 64 batch * 1024 seq)

Run: python scripts/train_mamba_50m.py
"""

import os
import sys
import json
import time
import math
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from mamba_integer_model import MambaIntegerModel
from rust_tokenizer import get_rust_tokenizer

# HuggingFace token - set HF_TOKEN environment variable if needed
HF_TOKEN = os.environ.get("HF_TOKEN", None)

# Training config
LOG_INTERVAL = 10
SAVE_INTERVAL = 2500
EVAL_INTERVAL = 1000

# Use mixed datasets for better training
USE_MIXED_DATASETS = True

DATASET_MIX = {
    "fineweb_edu": {
        "weight": 0.60,
        "path": "HuggingFaceFW/fineweb-edu",
        "name": "sample-10BT",
        "text_field": "text",
    },
    "tiny_codes": {
        "weight": 0.25,
        "path": "nampdn-ai/tiny-codes",
        "text_field": "prompt",  # or "response"
    },
    "openwebmath": {
        "weight": 0.15,
        "path": "open-web-math/open-web-math",
        "text_field": "text",
    }
}


class MixedDatasetLoader:
    """Weighted sampling from multiple streaming datasets."""

    def __init__(self, tokenizer, seq_len, dataset_mix):
        from datasets import load_dataset

        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.dataset_mix = dataset_mix

        # Normalize weights
        total_weight = sum(cfg["weight"] for cfg in dataset_mix.values())
        self.weights = {name: cfg["weight"] / total_weight for name, cfg in dataset_mix.items()}

        # Initialize datasets and buffers
        self.datasets = {}
        self.iterators = {}
        self.buffers = {name: [] for name in dataset_mix}
        self.exhausted = set()

        print("Loading datasets...")
        for name, cfg in dataset_mix.items():
            try:
                if "name" in cfg:
                    ds = load_dataset(cfg["path"], name=cfg["name"], split="train", streaming=True, token=HF_TOKEN)
                else:
                    ds = load_dataset(cfg["path"], split="train", streaming=True, token=HF_TOKEN)
                self.datasets[name] = ds
                self.iterators[name] = iter(ds)
                print(f"  Loaded: {name} (weight={self.weights[name]:.2f})")
            except Exception as e:
                print(f"  Failed to load {name}: {e}")
                self.exhausted.add(name)

    def _fill_buffer(self, name, min_tokens=50000):
        """Fill buffer for a specific dataset."""
        if name in self.exhausted:
            return False

        cfg = self.dataset_mix[name]
        text_field = cfg.get("text_field", "text")

        try:
            while len(self.buffers[name]) < min_tokens:
                item = next(self.iterators[name])
                text = item.get(text_field, "")
                if text:
                    tokens = self.tokenizer.encode(text)
                    self.buffers[name].extend(tokens)
            return True
        except StopIteration:
            self.exhausted.add(name)
            return len(self.buffers[name]) >= self.seq_len + 1

    def get_batch(self, batch_size, device):
        """Get a batch with weighted sampling from datasets."""
        import random

        sequences_x = []
        sequences_y = []

        for _ in range(batch_size):
            # Weighted random selection
            available = [n for n in self.weights if n not in self.exhausted or self.buffers[n]]
            if not available:
                raise StopIteration("All datasets exhausted")

            weights = [self.weights[n] for n in available]
            total = sum(weights)
            weights = [w / total for w in weights]

            name = random.choices(available, weights=weights, k=1)[0]

            # Ensure buffer has enough tokens
            if len(self.buffers[name]) < self.seq_len + 1:
                if not self._fill_buffer(name):
                    continue

            # Extract chunk
            if len(self.buffers[name]) >= self.seq_len + 1:
                chunk = self.buffers[name][:self.seq_len + 1]
                self.buffers[name] = self.buffers[name][self.seq_len:]
                sequences_x.append(chunk[:-1])
                sequences_y.append(chunk[1:])

        if not sequences_x:
            raise StopIteration("No data available")

        x = torch.tensor(sequences_x, dtype=torch.long, device=device)
        y = torch.tensor(sequences_y, dtype=torch.long, device=device)
        return x, y


class SimpleDatasetLoader:
    """Simple single-dataset loader."""

    def __init__(self, tokenizer, seq_len):
        from datasets import load_dataset

        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.buffer = []

        print("Loading FineWeb-Edu dataset...")
        self.dataset = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name="sample-10BT",
            split="train",
            streaming=True,
            token=HF_TOKEN
        )
        self.iterator = iter(self.dataset)

    def _fill_buffer(self, min_tokens=50000):
        while len(self.buffer) < min_tokens:
            item = next(self.iterator)
            text = item.get('text', '')
            if text:
                tokens = self.tokenizer.encode(text)
                self.buffer.extend(tokens)

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


def evaluate(model, tokenizer, device):
    """Quick evaluation on held-out examples."""
    model.eval()

    test_cases = [
        ("CODE", "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)"),
        ("EDU", "Photosynthesis is the process by which plants convert sunlight into energy."),
        ("MATH", "The derivative of x squared is 2x."),
    ]

    results = {}
    for name, text in test_cases:
        ids = tokenizer.encode(text)
        if len(ids) < 2:
            continue

        input_t = torch.tensor([ids], device=device)
        with torch.no_grad(), autocast(dtype=torch.bfloat16):
            logits = model(input_t)

        shift_logits = logits[0, :-1, :].float()
        shift_labels = input_t[0, 1:]
        loss = F.cross_entropy(shift_logits, shift_labels)
        ppl = torch.exp(loss).item()
        results[name] = ppl

    model.train()
    return results


def generate_sample(model, tokenizer, prompt, device, max_tokens=50, temp=0.8):
    """Generate a sample for monitoring."""
    model.eval()

    ids = tokenizer.encode(prompt)
    input_t = torch.tensor([ids], device=device)
    generated = []

    with torch.no_grad():
        for _ in range(max_tokens):
            with autocast(dtype=torch.bfloat16):
                logits = model(input_t)

            logits = logits[0, -1, :].float()

            # Repetition penalty
            for tok in set(generated[-20:]):
                logits[tok] /= 1.3

            probs = torch.softmax(logits / temp, dim=-1)
            next_tok = torch.multinomial(probs, 1).item()

            if next_tok == 0:
                break

            generated.append(next_tok)
            input_t = torch.cat([input_t, torch.tensor([[next_tok]], device=device)], dim=1)

    model.train()
    return tokenizer.decode(generated)


def train():
    print("=" * 60)
    print("MAMBA-INTEGER 50M TRAINING")
    print("=" * 60)
    print("Purpose: Fast validation of integer-only architecture")
    print("Target: Coherent outputs in ~32 hours")
    print("=" * 60)

    # Load config
    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'config_mamba_integer_50m.json')
    with open(config_path) as f:
        config = json.load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Initialize model
    print("\nInitializing 50M model...")
    model = MambaIntegerModel(config).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,} ({total_params/1e6:.1f}M)")

    # Initialize tokenizer
    tokenizer = get_rust_tokenizer()
    tokenizer.load(os.path.join(os.path.dirname(__file__), '..', 'configs', 'rust_bpe_merges.txt'))

    # Training params
    seq_len = config['training']['seq_len']
    batch_size = config['training']['batch_size']
    grad_accum = config['training']['gradient_accumulation_steps']
    total_steps = config['training']['total_steps']
    lr = config['training']['learning_rate']

    tokens_per_step = batch_size * grad_accum * seq_len
    total_tokens = total_steps * tokens_per_step

    print(f"\nTraining config:")
    print(f"  Seq length: {seq_len}")
    print(f"  Batch size: {batch_size}")
    print(f"  Grad accumulation: {grad_accum}")
    print(f"  Effective batch: {batch_size * grad_accum}")
    print(f"  Tokens per step: {tokens_per_step:,}")
    print(f"  Total steps: {total_steps:,}")
    print(f"  Total tokens: {total_tokens/1e9:.2f}B")

    # Initialize dataset
    if USE_MIXED_DATASETS:
        dataset = MixedDatasetLoader(tokenizer, seq_len, DATASET_MIX)
    else:
        dataset = SimpleDatasetLoader(tokenizer, seq_len)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=config['training']['weight_decay']
    )

    # LR scheduler - cosine with warmup
    warmup_steps = 1000
    def get_lr(step):
        if step < warmup_steps:
            return lr * step / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return lr * 0.5 * (1 + math.cos(math.pi * progress))

    # AMP
    scaler = GradScaler()

    # Check for existing checkpoint
    start_step = 1
    ckpt_files = sorted([f for f in os.listdir(os.path.dirname(__file__) + '/..')
                         if f.startswith('mamba_50m_step_') and f.endswith('.pt')])

    if ckpt_files:
        latest_ckpt = ckpt_files[-1]
        print(f"\nResuming from {latest_ckpt}...")
        ckpt = torch.load(os.path.join(os.path.dirname(__file__), '..', latest_ckpt),
                          map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_step = ckpt['step'] + 1
        print(f"Resumed from step {start_step - 1}")

    # Training loop
    print("\nStarting training...")
    model.train()

    log_file = os.path.join(os.path.dirname(__file__), '..', 'training_50m.log')

    losses = []
    start_time = time.time()

    for step in range(start_step, total_steps + 1):
        step_start = time.time()

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
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config['training']['grad_clip'])

        scaler.step(optimizer)
        scaler.update()

        losses.append(accum_loss)
        step_time = time.time() - step_start

        # Logging
        if step % LOG_INTERVAL == 0:
            avg_loss = sum(losses[-100:]) / len(losses[-100:]) if losses else accum_loss

            log_msg = f"Step {step}/{total_steps} | Loss: {accum_loss:.4f} | LR: {current_lr:.2e} | GradNorm: {grad_norm:.2f} | Time: {step_time:.2f}s"
            print(log_msg)

            with open(log_file, 'a') as f:
                f.write(log_msg + '\n')

        # Evaluation
        if step % EVAL_INTERVAL == 0:
            ppl_results = evaluate(model, tokenizer, device)
            print(f"  PPL - CODE: {ppl_results.get('CODE', 0):.2f}, EDU: {ppl_results.get('EDU', 0):.2f}, MATH: {ppl_results.get('MATH', 0):.2f}")

            # Generate sample
            sample = generate_sample(model, tokenizer, "The function of", device)
            print(f"  Sample: {sample[:100]}...")

        # Save checkpoint
        if step % SAVE_INTERVAL == 0:
            ckpt_path = os.path.join(os.path.dirname(__file__), '..', f'mamba_50m_step_{step}.pt')
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': accum_loss,
            }, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

    # Final summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Final loss: {losses[-1]:.4f}")
    print(f"Avg loss (last 500): {sum(losses[-500:]) / len(losses[-500:]):.4f}")
    print(f"Total time: {elapsed/3600:.1f} hours")
    print(f"Tokens trained: {total_steps * tokens_per_step / 1e9:.2f}B")


if __name__ == '__main__':
    train()
