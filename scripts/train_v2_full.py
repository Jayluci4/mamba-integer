#!/usr/bin/env python3
"""
Train Mamba-Integer V2 Full Architecture

All four phases:
1. Surprise-based retention
2. Multi-timescale memory (L0-L3)
3. Consolidation triggers
4. Selective forgetting + synaptic homeostasis

Usage:
    python scripts/train_v2_full.py --config configs/config_mamba_integer_v2_full.json
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import json
import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

from v2 import MambaIntegerModelV2Full


def create_synthetic_data(batch_size, seq_len, vocab_size, device):
    """Create synthetic training data with varying surprise levels."""
    data = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    # Make some positions predictable (repeat previous)
    for i in range(1, seq_len):
        mask = torch.rand(batch_size, device=device) < 0.3
        data[mask, i] = data[mask, i-1]

    return data


def train_step(model, optimizer, input_ids):
    """Single training step."""
    model.train()

    inputs = input_ids[:, :-1]
    targets = input_ids[:, 1:]

    logits, surprise_stats = model(inputs, return_surprise=True)

    loss = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        ignore_index=-100
    )

    optimizer.zero_grad()
    loss.backward()

    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    return loss.item(), grad_norm.item(), surprise_stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--steps', type=int, default=100)
    args = parser.parse_args()

    # Default config for full V2
    config = {
        'vocab_size': 4096,
        'd_model': 256,
        'n_layer': 4,
        'ssm_cfg': {
            'n_heads': 8,
            'd_head': 32,
            'd_state': 32,
            'chunk_size': 64,
            'n_persistent': 32,
            'chunk_memory_size': 128,
            'homeostasis_interval': 32,
            'capacity_threshold': 0.9,
            'surprise_threshold': 2.0,
            'time_threshold': 64,
        },
        'training': {
            'learning_rate': 3e-4,
            'weight_decay': 0.1,
            'batch_size': 4,
            'seq_len': 256,
            'total_steps': args.steps,
            'log_interval': 10,
        }
    }

    # Load config if provided
    if args.config and os.path.exists(args.config):
        with open(args.config) as f:
            config.update(json.load(f))

    print("=" * 60)
    print("MAMBA-INTEGER V2 FULL ARCHITECTURE TRAINING")
    print("=" * 60)
    print(f"Device: {args.device}")
    print(f"Model: d_model={config['d_model']}, n_layer={config['n_layer']}")
    print(f"SSM: d_state={config['ssm_cfg']['d_state']}, n_heads={config['ssm_cfg']['n_heads']}")
    print(f"Memory: n_persistent={config['ssm_cfg']['n_persistent']}, chunk_mem={config['ssm_cfg']['chunk_memory_size']}")
    print("=" * 60)

    # Create model
    model = MambaIntegerModelV2Full(config).to(args.device)
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
    batch_size = train_cfg['batch_size']
    seq_len = train_cfg['seq_len']
    vocab_size = config['vocab_size']

    print(f"\nTraining for {total_steps} steps...")
    print(f"Batch size: {batch_size}, Seq len: {seq_len}")
    print()

    losses = []
    surprises = []

    for step in range(1, total_steps + 1):
        input_ids = create_synthetic_data(batch_size, seq_len, vocab_size, args.device)

        t0 = time.time()
        loss, grad_norm, surprise_stats = train_step(model, optimizer, input_ids)
        dt = time.time() - t0

        losses.append(loss)
        if surprise_stats:
            surprises.append(surprise_stats['mean'])

        if step % log_interval == 0:
            avg_loss = sum(losses[-log_interval:]) / log_interval
            avg_surprise = sum(surprises[-log_interval:]) / log_interval if surprises else 0

            # Get detailed stats
            all_stats = model.get_all_stats()
            consolidation_total = sum(s['consolidation']['consolidation_count'] for s in all_stats)
            homeostasis_total = sum(s['homeostasis_count'] for s in all_stats)
            chunk_mem_total = sum(s['chunk_memory_filled'] for s in all_stats)

            print(f"Step {step}/{total_steps} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"GradNorm: {grad_norm:.4f} | "
                  f"Surprise: {avg_surprise:.4f} | "
                  f"Consol: {consolidation_total} | "
                  f"Homeo: {homeostasis_total} | "
                  f"ChunkMem: {chunk_mem_total} | "
                  f"Time: {dt:.2f}s")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Final loss: {losses[-1]:.4f}")
    if surprises:
        print(f"Final surprise: {surprises[-1]:.4f}")

    # Final statistics
    print("\nFinal Layer Statistics:")
    for stats in model.get_all_stats():
        print(f"  Layer {stats['layer_idx']}:")
        print(f"    Surprise EMA: {stats['surprise_ema']:.4f}")
        print(f"    Alpha base: {stats['alpha_base_mean']:.4f}")
        print(f"    Consolidations: {stats['consolidation']['consolidation_count']}")
        print(f"    Homeostasis: {stats['homeostasis_count']}")
        print(f"    Chunk Memory: {stats['chunk_memory_filled']}")


if __name__ == '__main__':
    main()
