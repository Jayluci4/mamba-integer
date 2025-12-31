"""Profile training loop to identify bottlenecks."""

import os
import sys
import torch
import torch.nn as nn
import json
import time
from contextlib import contextmanager
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src/triton_kernels")))

from mamba_integer_model import MambaIntegerModel


class CUDATimer:
    """Accurate GPU timing using CUDA events."""

    def __init__(self):
        self.timings = defaultdict(list)

    @contextmanager
    def time(self, name):
        """Context manager for timing a block."""
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        yield
        end.record()

        torch.cuda.synchronize()
        self.timings[name].append(start.elapsed_time(end))

    def summary(self, title="TIMING SUMMARY"):
        """Print timing summary."""
        print("\n" + "=" * 70)
        print(title)
        print("=" * 70)

        total = 0
        results = []
        for name, times in self.timings.items():
            avg = sum(times) / len(times)
            results.append((name, avg, len(times)))
            if not name.startswith("  "):  # Top-level only
                total += avg

        # Sort by time (descending)
        results.sort(key=lambda x: -x[1])

        print(f"\n{'Component':<45} {'Avg (ms)':<12} {'%':<8} {'Calls':<8}")
        print("-" * 70)

        for name, avg, count in results:
            pct = (avg / total * 100) if total > 0 and not name.startswith("  ") else 0
            pct_str = f"{pct:.1f}%" if pct > 0 else ""
            print(f"{name:<45} {avg:>10.2f}  {pct_str:<8} {count:<8}")

        print("-" * 70)
        print(f"{'TOTAL (top-level)':<45} {total:>10.2f}ms")
        print("=" * 70)

        return results, total


def profile_kernel_operations(config, device):
    """Profile individual Triton kernel operations."""
    print("\n" + "=" * 70)
    print("TRITON KERNEL OPERATION PROFILE")
    print("=" * 70)

    from dyadic_scan import dyadic_scan_triton_fast, dyadic_scan_chunked
    from ssd_multihead import ssd_multihead
    from fused_activations import fused_squareplus_clamp, fused_sigmoid_gate

    timer = CUDATimer()

    # Training dimensions from config
    batch_size = config.get('training', {}).get('batch_size', 10)
    seq_len = config.get('training', {}).get('seq_len', 1024)
    d_model = config['d_model']
    n_heads = config['ssm_cfg']['n_heads']
    d_head = config['ssm_cfg']['d_head']
    d_state = config['ssm_cfg']['d_state']

    print(f"\nDimensions: B={batch_size}, L={seq_len}, d_model={d_model}")
    print(f"SSM: n_heads={n_heads}, d_head={d_head}, d_state={d_state}")

    n_iters = 20

    # 1. Dyadic Scan
    print("\n--- Dyadic Scan ---")
    u = torch.randn(batch_size, seq_len, d_model, device=device, dtype=torch.float32)
    nums = torch.randint(8000, 28000, (batch_size, seq_len, d_model), device=device, dtype=torch.float32)

    # Warmup
    for _ in range(3):
        _ = dyadic_scan_triton_fast(u, nums)
    torch.cuda.synchronize()

    for _ in range(n_iters):
        with timer.time("dyadic_scan_sequential"):
            _ = dyadic_scan_triton_fast(u, nums)

    # Chunked version
    for _ in range(3):
        _ = dyadic_scan_chunked(u, nums, chunk_size=64)
    torch.cuda.synchronize()

    for _ in range(n_iters):
        with timer.time("dyadic_scan_chunked_64"):
            _ = dyadic_scan_chunked(u, nums, chunk_size=64)

    # 2. SSD Multihead (forward only)
    print("\n--- SSD Multihead ---")
    X = torch.randn(batch_size, seq_len, n_heads, d_head, device=device, dtype=torch.float32)
    A = torch.randn(batch_size, seq_len, n_heads, device=device, dtype=torch.float32) * 0.1 - 0.5
    B = torch.randn(batch_size, seq_len, n_heads, d_state, device=device, dtype=torch.float32)
    C = torch.randn(batch_size, seq_len, n_heads, d_state, device=device, dtype=torch.float32)

    # Warmup
    for _ in range(3):
        _ = ssd_multihead(X, A, B, C, chunk_size=64)
    torch.cuda.synchronize()

    for _ in range(n_iters):
        with timer.time("ssd_multihead_forward"):
            _ = ssd_multihead(X, A, B, C, chunk_size=64)

    # SSD with backward
    X = X.clone().detach().requires_grad_(True)
    A = A.clone().detach().requires_grad_(True)
    B = B.clone().detach().requires_grad_(True)
    C = C.clone().detach().requires_grad_(True)

    for _ in range(n_iters):
        with timer.time("ssd_multihead_fwd+bwd"):
            Y = ssd_multihead(X, A, B, C, chunk_size=64)
            loss = Y.sum()
            loss.backward()
            X.grad = None
            A.grad = None
            B.grad = None
            C.grad = None

    # 3. Fused activations
    print("\n--- Fused Activations ---")
    x_act = torch.randn(batch_size, seq_len, d_model, device=device, dtype=torch.float32)
    gate = torch.randn(batch_size, seq_len, d_model, device=device, dtype=torch.float32)

    # Warmup
    for _ in range(3):
        _ = fused_squareplus_clamp(x_act)
        _ = fused_sigmoid_gate(x_act, gate)
    torch.cuda.synchronize()

    for _ in range(n_iters):
        with timer.time("squareplus_clamp"):
            _ = fused_squareplus_clamp(x_act)

    for _ in range(n_iters):
        with timer.time("fused_sigmoid_gate"):
            _ = fused_sigmoid_gate(x_act, gate)

    timer.summary("KERNEL TIMINGS")


def profile_training_step(model, config, device, n_iters=20):
    """Profile a full training step."""
    print("\n" + "=" * 70)
    print("FULL TRAINING STEP PROFILE")
    print("=" * 70)

    timer = CUDATimer()

    # Config
    seq_len = config.get('training', {}).get('seq_len', 1024)
    batch_size = config.get('training', {}).get('batch_size', 10)
    vocab_size = config['vocab_size']
    grad_accum = config.get('training', {}).get('gradient_accumulation_steps', 4)

    print(f"\nConfig: batch={batch_size}, seq={seq_len}, vocab={vocab_size}, grad_accum={grad_accum}")

    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    # Warmup
    print("Warming up...")
    for _ in range(3):
        x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        y = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = model(x)
            loss = criterion(logits.view(-1, vocab_size), y.view(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    torch.cuda.synchronize()

    # Profile iterations
    print(f"Profiling {n_iters} optimizer steps...")

    for i in range(n_iters):
        # Data generation (simulated)
        with timer.time("data_generation"):
            x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
            y = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

        optimizer.zero_grad()

        # Gradient accumulation loop
        for accum_step in range(grad_accum):
            # Forward pass
            with timer.time("forward_pass"):
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    logits = model(x)

            # Loss computation
            with timer.time("loss_computation"):
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    loss = criterion(logits.view(-1, vocab_size), y.view(-1))
                    loss = loss / grad_accum

            # Backward pass
            with timer.time("backward_pass"):
                loss.backward()

        # Gradient clipping
        with timer.time("grad_clipping"):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Optimizer step
        with timer.time("optimizer_step"):
            optimizer.step()

    torch.cuda.synchronize()

    # Summary
    results, total_ms = timer.summary("TRAINING STEP BREAKDOWN")

    # Calculate throughput
    tokens_per_step = batch_size * seq_len * grad_accum
    throughput = tokens_per_step / (total_ms / 1000)

    print(f"\nThroughput: {throughput:,.0f} tokens/sec")
    print(f"Time per optimizer step: {total_ms:.1f}ms ({total_ms/1000:.2f}s)")

    return results


def profile_per_layer(model, config, device):
    """Profile time spent in each layer."""
    print("\n" + "=" * 70)
    print("PER-LAYER PROFILE")
    print("=" * 70)

    timer = CUDATimer()

    seq_len = config.get('training', {}).get('seq_len', 1024)
    batch_size = config.get('training', {}).get('batch_size', 10)
    vocab_size = config['vocab_size']

    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    # Warmup
    for _ in range(3):
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            _ = model(x)
    torch.cuda.synchronize()

    # Profile each forward pass component
    n_iters = 10
    for _ in range(n_iters):
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            with timer.time("embedding"):
                h = model.embedding(x)

            for i, layer in enumerate(model.layers):
                with timer.time(f"layer_{i:02d}"):
                    h = layer(h)

            with timer.time("lm_head"):
                logits = model.lm_head(h)

    timer.summary("FORWARD PASS BY LAYER")


def profile_memory_usage(model, config, device):
    """Profile memory usage."""
    print("\n" + "=" * 70)
    print("MEMORY PROFILE")
    print("=" * 70)

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    seq_len = config.get('training', {}).get('seq_len', 1024)
    batch_size = config.get('training', {}).get('batch_size', 10)
    vocab_size = config['vocab_size']

    # Baseline memory
    baseline = torch.cuda.memory_allocated() / 1024**3
    print(f"\nBaseline (model loaded): {baseline:.2f} GB")

    # Forward pass
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    torch.cuda.reset_peak_memory_stats()
    with torch.amp.autocast('cuda', dtype=torch.bfloat16):
        logits = model(x)

    after_fwd = torch.cuda.memory_allocated() / 1024**3
    peak_fwd = torch.cuda.max_memory_allocated() / 1024**3
    print(f"After forward: {after_fwd:.2f} GB (peak: {peak_fwd:.2f} GB)")

    # Backward pass
    criterion = nn.CrossEntropyLoss()
    y = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

    torch.cuda.reset_peak_memory_stats()
    loss = criterion(logits.view(-1, vocab_size), y.view(-1))
    loss.backward()

    after_bwd = torch.cuda.memory_allocated() / 1024**3
    peak_bwd = torch.cuda.max_memory_allocated() / 1024**3
    print(f"After backward: {after_bwd:.2f} GB (peak: {peak_bwd:.2f} GB)")

    # Per-parameter gradient memory
    grad_memory = sum(p.grad.numel() * p.grad.element_size() for p in model.parameters() if p.grad is not None)
    print(f"Gradient memory: {grad_memory / 1024**3:.2f} GB")

    # Total GPU memory
    total_gpu = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"\nTotal GPU memory: {total_gpu:.1f} GB")
    print(f"Peak usage: {peak_bwd:.2f} GB ({peak_bwd/total_gpu*100:.1f}%)")


def main():
    print("=" * 70)
    print("MAMBA-INTEGER TRAINING PROFILER")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("ERROR: CUDA required for profiling")
        return

    # Load config
    config_path = os.path.join(os.path.dirname(__file__), "../configs/config_mamba_integer_l4.json")
    with open(config_path, 'r') as f:
        config = json.load(f)

    print(f"\nDevice: {torch.cuda.get_device_name(0)}")
    print(f"Config: {config_path}")

    # Enable TF32
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Load model
    print("\nLoading model...")
    model = MambaIntegerModel(config).to(device)
    model.train()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,} ({n_params/1e6:.1f}M)")

    # Run profiling
    profile_memory_usage(model, config, device)

    # Clear memory before kernel profiling
    model.zero_grad()
    torch.cuda.empty_cache()

    profile_kernel_operations(config, device)

    # Clear memory and reload model for training step
    torch.cuda.empty_cache()
    del model
    torch.cuda.empty_cache()

    model = MambaIntegerModel(config).to(device)
    model.train()

    profile_training_step(model, config, device, n_iters=10)

    print("\n" + "=" * 70)
    print("PROFILING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
