#!/usr/bin/env python3
"""
Benchmark V1 vs V2 Full vs V2 Redesign

Compares:
1. Forward pass latency
2. Memory usage
3. Consolidation events
4. Training step time
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn.functional as F
import time
import gc


def get_memory_mb():
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0


def benchmark_model(model, input_ids, name, n_warmup=3, n_iter=10):
    """Benchmark a model's forward pass."""
    device = input_ids.device

    # Warmup
    for _ in range(n_warmup):
        with torch.no_grad():
            _ = model(input_ids)

    if device.type == 'cuda':
        torch.cuda.synchronize()

    # Benchmark forward
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    mem_before = get_memory_mb()

    t0 = time.time()
    for _ in range(n_iter):
        with torch.no_grad():
            out = model(input_ids)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t1 = time.time()

    mem_after = get_memory_mb()
    peak_mem = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0

    avg_time = (t1 - t0) / n_iter * 1000
    tokens_per_sec = (input_ids.numel() * n_iter) / (t1 - t0)

    return {
        'name': name,
        'avg_time_ms': avg_time,
        'tokens_per_sec': tokens_per_sec,
        'peak_memory_mb': peak_mem,
    }


def benchmark_training_step(model, input_ids, name):
    """Benchmark a single training step."""
    device = input_ids.device
    targets = input_ids[:, 1:].contiguous()
    inputs = input_ids[:, :-1].contiguous()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    def get_logits(out):
        """Extract logits from model output (handles tuple or tensor)."""
        if isinstance(out, tuple):
            return out[0]
        return out

    # Warmup
    for _ in range(2):
        optimizer.zero_grad()
        out = model(inputs)
        logits = get_logits(out)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        loss.backward()
        optimizer.step()

    if device.type == 'cuda':
        torch.cuda.synchronize()

    # Benchmark
    n_iter = 5
    t0 = time.time()
    for _ in range(n_iter):
        optimizer.zero_grad()
        out = model(inputs)
        logits = get_logits(out)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        loss.backward()
        optimizer.step()
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t1 = time.time()

    avg_time = (t1 - t0) / n_iter * 1000

    return {
        'name': name,
        'train_step_ms': avg_time,
    }


def main():
    print("=" * 70)
    print("BENCHMARK: V1 vs V2 Full vs V2 Redesign")
    print("=" * 70)
    print()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
    print()

    # Config
    config = {
        'vocab_size': 4096,
        'd_model': 512,
        'n_layer': 4,
        'ssm_cfg': {
            'n_heads': 8,
            'd_head': 64,
            'd_state': 64,
            'chunk_size': 64,
            'use_ssd': True,
        }
    }

    batch, seqlen = 4, 256
    input_ids = torch.randint(0, config['vocab_size'], (batch, seqlen), device=device)

    results = []

    # V1 Baseline
    print("Loading V1 (MambaIntegerModel)...")
    try:
        from mamba_integer_model import MambaIntegerModel
        model_v1 = MambaIntegerModel(config).to(device)
        n_params_v1 = sum(p.numel() for p in model_v1.parameters())
        print(f"  Parameters: {n_params_v1:,}")

        res = benchmark_model(model_v1, input_ids, "V1 Baseline")
        results.append(res)
        print(f"  Forward: {res['avg_time_ms']:.2f}ms, {res['tokens_per_sec']:.0f} tok/s")

        try:
            train_res = benchmark_training_step(model_v1, input_ids, "V1 Baseline")
            res['train_step_ms'] = train_res['train_step_ms']
            print(f"  Train step: {res['train_step_ms']:.2f}ms")
        except Exception as e:
            print(f"  Train step error: {e}")

        del model_v1
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"  Error: {e}")

    print()

    # V2 Full
    print("Loading V2 Full (MambaIntegerModelV2Full)...")
    try:
        from v2.full_v2_architecture import MambaIntegerModelV2Full
        model_v2_full = MambaIntegerModelV2Full(config).to(device)
        n_params_v2_full = model_v2_full.count_parameters()
        print(f"  Parameters: {n_params_v2_full:,}")

        res = benchmark_model(model_v2_full, input_ids, "V2 Full")
        results.append(res)
        print(f"  Forward: {res['avg_time_ms']:.2f}ms, {res['tokens_per_sec']:.0f} tok/s")

        # Get consolidation stats
        _, _ = model_v2_full(input_ids, return_surprise=True)
        stats = model_v2_full.get_all_stats()
        total_consol = sum(s['consolidation']['consolidation_count'] for s in stats)
        res['consolidations'] = total_consol
        print(f"  Consolidations: {total_consol}")

        try:
            train_res = benchmark_training_step(model_v2_full, input_ids, "V2 Full")
            res['train_step_ms'] = train_res['train_step_ms']
            print(f"  Train step: {res['train_step_ms']:.2f}ms")
        except Exception as e:
            print(f"  Train step error: {e}")

        del model_v2_full
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"  Error: {e}")

    print()

    # V2 Redesign
    print("Loading V2 Redesign (MambaIntegerModelV2Redesign)...")
    try:
        from v2.v2_redesign import MambaIntegerModelV2Redesign
        model_v2_redesign = MambaIntegerModelV2Redesign(config).to(device)
        n_params_v2_redesign = model_v2_redesign.count_parameters()
        print(f"  Parameters: {n_params_v2_redesign:,}")

        res = benchmark_model(model_v2_redesign, input_ids, "V2 Redesign")
        results.append(res)
        print(f"  Forward: {res['avg_time_ms']:.2f}ms, {res['tokens_per_sec']:.0f} tok/s")

        # Get consolidation stats
        _, _, stats_list = model_v2_redesign(input_ids, return_stats=True)
        total_consol = sum(s['consolidation_events'] for s in stats_list)
        res['consolidations'] = total_consol
        print(f"  Consolidations: {total_consol}")

        try:
            train_res = benchmark_training_step(model_v2_redesign, input_ids, "V2 Redesign")
            res['train_step_ms'] = train_res['train_step_ms']
            print(f"  Train step: {res['train_step_ms']:.2f}ms")
        except Exception as e:
            print(f"  Train step error: {e}")

        del model_v2_redesign
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"  Error: {e}")

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()

    # Print comparison table
    headers = ["Model", "Forward (ms)", "Tok/s", "Train (ms)", "Consol", "Overhead"]
    print(f"{'Model':<20} {'Forward':<12} {'Tok/s':<10} {'Train':<12} {'Consol':<8}")
    print("-" * 70)

    baseline_time = results[0]['avg_time_ms'] if results else 1

    for res in results:
        overhead = res['avg_time_ms'] / baseline_time
        consol = res.get('consolidations', 'N/A')
        train = res.get('train_step_ms', 'N/A')
        if isinstance(train, float):
            train = f"{train:.1f}ms"
        print(f"{res['name']:<20} {res['avg_time_ms']:<12.2f} {res['tokens_per_sec']:<10.0f} {train:<12} {consol:<8} {overhead:.2f}x")

    print()
    print("=" * 70)


if __name__ == '__main__':
    main()
