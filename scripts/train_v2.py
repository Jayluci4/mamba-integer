#!/usr/bin/env python3
"""
Train Mamba-Integer V2 with Surprise-Gated Retention

Key differences from V1:
1. Uses MambaIntegerBlockV2Surprise with adaptive retention
2. Logs surprise statistics during training
3. Validates surprise-retention correlation

Usage:
    python scripts/train_v2.py --config configs/config_mamba_integer_v2_small.json
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

# V2 imports - use FAST chunked version
from v2 import MambaIntegerBlockV2Chunked as MambaIntegerBlockV2Surprise


class MambaIntegerModelV2(nn.Module):
    """
    Mamba-Integer V2 with Surprise-Gated Retention.

    Architecture:
    - Embedding layer
    - N x MambaIntegerBlockV2Surprise (with surprise gates)
    - LM head
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        vocab_size = config['vocab_size']
        d_model = config['d_model']
        n_layer = config['n_layer']

        # Embedding
        self.embed = nn.Embedding(vocab_size, d_model)

        # Blocks with surprise gates
        self.blocks = nn.ModuleList([
            MambaIntegerBlockV2Surprise(config, layer_idx=i)
            for i in range(n_layer)
        ])

        # LM head
        self.norm_f = nn.RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Tie weights
        self.lm_head.weight = self.embed.weight

        # Initialize
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def forward(self, input_ids, return_surprise=False):
        """
        Forward pass.

        Args:
            input_ids: [B, L] token IDs
            return_surprise: Whether to return surprise statistics

        Returns:
            logits: [B, L, vocab_size]
            surprise_stats: Optional dict of surprise statistics
        """
        x = self.embed(input_ids)

        surprise_all = []
        for block in self.blocks:
            x, surprise = block(x, return_surprise=True)
            if surprise is not None:
                surprise_all.append(surprise)

        x = self.norm_f(x)
        logits = self.lm_head(x)

        if return_surprise and surprise_all:
            surprise_stats = {
                'mean': torch.stack([s.mean() for s in surprise_all]).mean().item(),
                'max': torch.stack([s.max() for s in surprise_all]).max().item(),
                'per_layer': [s.mean().item() for s in surprise_all],
            }
            return logits, surprise_stats

        return logits, None

    def get_surprise_stats(self):
        """Get surprise statistics from all blocks."""
        stats = []
        for block in self.blocks:
            stats.append(block.get_surprise_stats())
        return stats

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_synthetic_data(batch_size, seq_len, vocab_size, device):
    """Create synthetic training data with varying surprise levels."""
    # Mix of predictable (repeated) and surprising (random) tokens
    data = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    # Make some positions predictable (repeat previous)
    for i in range(1, seq_len):
        mask = torch.rand(batch_size, device=device) < 0.3  # 30% predictable
        data[mask, i] = data[mask, i-1]

    return data


def train_step(model, optimizer, input_ids, scaler=None):
    """Single training step."""
    model.train()

    # Shift for causal LM
    inputs = input_ids[:, :-1]
    targets = input_ids[:, 1:]

    # Forward
    logits, surprise_stats = model(inputs, return_surprise=True)

    # Loss
    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=-100
    )

    # Backward
    optimizer.zero_grad()
    loss.backward()

    # Gradient clipping
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

    optimizer.step()

    return loss.item(), grad_norm.item(), surprise_stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config_mamba_integer_v2_small.json')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        config = json.load(f)

    print("="*60)
    print("MAMBA-INTEGER V2 TRAINING")
    print("="*60)
    print(f"Config: {args.config}")
    print(f"Device: {args.device}")
    print(f"Model: d_model={config['d_model']}, n_layer={config['n_layer']}")
    print(f"SSM: d_state={config['ssm_cfg']['d_state']}, n_heads={config['ssm_cfg']['n_heads']}")
    print("="*60)

    # Create model
    model = MambaIntegerModelV2(config).to(args.device)
    n_params = model.count_parameters()
    print(f"Parameters: {n_params:,} ({n_params/1e6:.1f}M)")

    # Optimizer
    train_cfg = config['training']
    optimizer = AdamW(
        model.parameters(),
        lr=train_cfg['learning_rate'],
        weight_decay=train_cfg['weight_decay'],
    )

    # Training loop
    total_steps = train_cfg['total_steps']
    log_interval = train_cfg['log_interval']
    save_interval = train_cfg['save_interval']
    batch_size = train_cfg['batch_size']
    seq_len = train_cfg['seq_len']
    vocab_size = config['vocab_size']

    print(f"\nTraining for {total_steps} steps...")
    print(f"Batch size: {batch_size}, Seq len: {seq_len}")
    print()

    losses = []
    surprises = []

    for step in range(1, total_steps + 1):
        # Generate synthetic data
        input_ids = create_synthetic_data(batch_size, seq_len, vocab_size, args.device)

        # Train step
        t0 = time.time()
        loss, grad_norm, surprise_stats = train_step(model, optimizer, input_ids)
        dt = time.time() - t0

        losses.append(loss)
        if surprise_stats:
            surprises.append(surprise_stats['mean'])

        # Log
        if step % log_interval == 0:
            avg_loss = sum(losses[-log_interval:]) / log_interval
            avg_surprise = sum(surprises[-log_interval:]) / log_interval if surprises else 0

            print(f"Step {step}/{total_steps} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"GradNorm: {grad_norm:.4f} | "
                  f"Surprise: {avg_surprise:.4f} | "
                  f"Time: {dt:.2f}s")

            # Log per-layer surprise
            if train_cfg.get('log_surprise', False) and surprise_stats:
                per_layer = surprise_stats['per_layer']
                layer_str = ', '.join([f"L{i}:{s:.4f}" for i, s in enumerate(per_layer)])
                print(f"  Per-layer surprise: {layer_str}")

        # Save checkpoint
        if step % save_interval == 0:
            ckpt_path = f"checkpoints/mamba_v2_step_{step}.pt"
            os.makedirs('checkpoints', exist_ok=True)
            torch.save({
                'step': step,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'config': config,
                'loss': loss,
            }, ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path}")

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Final loss: {losses[-1]:.4f}")
    print(f"Final surprise: {surprises[-1]:.4f}")

    # Final surprise analysis
    print("\nSurprise Statistics by Layer:")
    stats = model.get_surprise_stats()
    for s in stats:
        print(f"  Layer {s['layer_idx']}: "
              f"α_base={s['alpha_base_mean']:.3f}, "
              f"β={s['beta_mean']:.3f}, "
              f"EMA={s['surprise_ema']:.4f}")


if __name__ == '__main__':
    main()
