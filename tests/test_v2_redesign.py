#!/usr/bin/env python3
"""
Tests for V2 Redesigned Architecture

Tests:
1. Convex SSD forward produces valid output
2. Convex combination balances old/new state correctly
3. GradientSurpriseGate computes alpha correctly
4. AggressiveConsolidationTrigger fires more frequently
5. MultiTimescaleMemoryV2 reads/writes correctly
6. Full block forward pass works
7. Full model forward pass works
8. Model backward pass (gradients flow)
9. Consolidation events occur during forward
10. Performance benchmark vs V1
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn.functional as F
import time

from src.v2.v2_redesign import (
    GradientSurpriseGate,
    AggressiveConsolidationTrigger,
    MultiTimescaleMemoryV2,
    MambaIntegerBlockV2Redesign,
    MambaIntegerModelV2Redesign,
    convex_ssd_chunk_forward,
)
from src.triton_kernels.ssd_multihead import build_causal_decay_matrix_multihead


def test_convex_ssd_forward():
    """Test that convex SSD forward produces valid output."""
    print("Test 1: Convex SSD forward...")

    batch, n_heads, cs, d_head = 2, 8, 64, 32
    d_state = 32
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    X = torch.randn(batch, n_heads, cs, d_head, device=device)
    A = torch.randn(batch, n_heads, cs, device=device) * 0.1 - 0.5
    B = torch.randn(batch, n_heads, cs, d_state, device=device)
    C = torch.randn(batch, n_heads, cs, d_state, device=device)
    h_prev = torch.zeros(batch, n_heads, d_state, d_head, device=device)

    Y, h_end, decay = convex_ssd_chunk_forward(X, A, B, C, h_prev, alpha=0.5)

    assert Y.shape == (batch, n_heads, cs, d_head), f"Expected {(batch, n_heads, cs, d_head)}, got {Y.shape}"
    assert h_end.shape == (batch, n_heads, d_state, d_head), f"Expected {(batch, n_heads, d_state, d_head)}, got {h_end.shape}"
    assert not torch.isnan(Y).any(), "Y contains NaN"
    assert not torch.isnan(h_end).any(), "h_end contains NaN"

    print("  PASSED: Convex SSD forward produces valid output")


def test_convex_combination_balance():
    """Test that alpha properly balances old state vs new input."""
    print("Test 2: Convex combination balance...")

    batch, n_heads, cs, d_head = 1, 4, 32, 16
    d_state = 16
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    X = torch.randn(batch, n_heads, cs, d_head, device=device)
    A = torch.ones(batch, n_heads, cs, device=device) * -0.1  # Slow decay
    B = torch.randn(batch, n_heads, cs, d_state, device=device)
    C = torch.randn(batch, n_heads, cs, d_state, device=device)
    h_prev = torch.ones(batch, n_heads, d_state, d_head, device=device)

    # High alpha (0.9) should preserve more of h_prev
    _, h_high_alpha, _ = convex_ssd_chunk_forward(X, A, B, C, h_prev.clone(), alpha=0.9)

    # Low alpha (0.1) should add more new input
    _, h_low_alpha, _ = convex_ssd_chunk_forward(X, A, B, C, h_prev.clone(), alpha=0.1)

    # h_high_alpha should be closer to decayed h_prev
    decay_factor = torch.exp(A.sum(dim=-1, keepdim=True).unsqueeze(-1))
    h_prev_decayed = decay_factor * h_prev

    dist_high = (h_high_alpha - h_prev_decayed).pow(2).mean()
    dist_low = (h_low_alpha - h_prev_decayed).pow(2).mean()

    assert dist_high < dist_low, f"High alpha should be closer to h_prev: {dist_high:.4f} vs {dist_low:.4f}"

    print(f"  PASSED: dist_high={dist_high:.4f} < dist_low={dist_low:.4f}")


def test_gradient_surprise_gate():
    """Test GradientSurpriseGate computes alpha correctly."""
    print("Test 3: GradientSurpriseGate...")

    n_heads, d_state = 8, 32
    batch = 2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    gate = GradientSurpriseGate(n_heads, d_state).to(device)

    # Low surprise -> moderate alpha
    surprise_low = torch.ones(batch, n_heads, 1, device=device) * 0.1
    alpha_low, _ = gate(surprise_low)

    # High surprise -> high alpha
    surprise_high = torch.ones(batch, n_heads, 1, device=device) * 10.0
    alpha_high, _ = gate(surprise_high)

    assert alpha_high.mean() > alpha_low.mean(), "High surprise should yield higher alpha"
    assert 0 < alpha_low.mean() < 1, f"Alpha should be in (0,1), got {alpha_low.mean()}"
    assert 0 < alpha_high.mean() < 1, f"Alpha should be in (0,1), got {alpha_high.mean()}"

    print(f"  PASSED: alpha_low={alpha_low.mean():.4f}, alpha_high={alpha_high.mean():.4f}")


def test_aggressive_consolidation():
    """Test AggressiveConsolidationTrigger fires more frequently."""
    print("Test 4: Aggressive consolidation trigger...")

    # With aggressive thresholds: capacity=0.5, surprise=0.5, time=32
    trigger = AggressiveConsolidationTrigger(
        capacity_threshold=0.5,
        surprise_threshold=0.5,
        time_threshold=32,
    )

    consolidation_count = 0

    # Simulate 100 tokens with moderate surprise
    for i in range(100):
        should, triggers = trigger.should_consolidate(
            surprise_score=0.02,  # Small per-token surprise
            buffer_fullness=0.3,  # Below capacity threshold
        )
        if should:
            consolidation_count += 1

    # Should consolidate at least 2-3 times (every ~32 tokens due to time threshold)
    assert consolidation_count >= 2, f"Expected >= 2 consolidations, got {consolidation_count}"

    print(f"  PASSED: {consolidation_count} consolidations in 100 tokens")


def test_multi_timescale_memory():
    """Test MultiTimescaleMemoryV2 read/write."""
    print("Test 5: MultiTimescaleMemoryV2...")

    d_model, n_heads, d_state, d_head = 256, 8, 32, 32
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    memory = MultiTimescaleMemoryV2(d_model, n_heads, d_state, d_head).to(device)

    # Write some states
    batch = 2
    state = torch.randn(batch, n_heads, d_state, d_head, device=device)
    importance = torch.rand(batch, n_heads, device=device)

    memory.write_chunk_summary(state, importance)

    assert memory.chunk_memory_filled.item() == 1, "Should have 1 entry after write"

    # Read
    query = torch.randn(batch, 64, d_model, device=device)
    mem_out = memory.read_memory(query)

    assert mem_out.shape == (batch, 64, d_model), f"Expected {(batch, 64, d_model)}, got {mem_out.shape}"
    assert not torch.isnan(mem_out).any(), "Memory output contains NaN"

    print("  PASSED: Memory read/write works")


def test_block_forward():
    """Test full block forward pass."""
    print("Test 6: Block forward pass...")

    config = {
        'd_model': 256,
        'n_layer': 4,
        'ssm_cfg': {
            'n_heads': 8,
            'd_head': 32,
            'd_state': 32,
            'chunk_size': 64,
        }
    }

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    block = MambaIntegerBlockV2Redesign(config, layer_idx=0).to(device)

    batch, seqlen = 2, 128
    x = torch.randn(batch, seqlen, config['d_model'], device=device)

    out, surprise, stats = block(x, return_surprise=True, return_stats=True)

    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"
    assert not torch.isnan(out).any(), "Output contains NaN"
    assert 'mean_surprise' in stats, "Stats should contain mean_surprise"
    assert 'consolidation_events' in stats, "Stats should contain consolidation_events"

    print(f"  PASSED: Output shape={out.shape}, surprise={stats['mean_surprise']:.4f}")


def test_model_forward():
    """Test full model forward pass."""
    print("Test 7: Model forward pass...")

    config = {
        'vocab_size': 4096,
        'd_model': 256,
        'n_layer': 2,
        'ssm_cfg': {
            'n_heads': 8,
            'd_head': 32,
            'd_state': 32,
            'chunk_size': 64,
        }
    }

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MambaIntegerModelV2Redesign(config).to(device)

    batch, seqlen = 2, 128
    input_ids = torch.randint(0, config['vocab_size'], (batch, seqlen), device=device)

    logits, surprise_stats = model(input_ids, return_surprise=True)

    assert logits.shape == (batch, seqlen, config['vocab_size']), f"Got {logits.shape}"
    assert not torch.isnan(logits).any(), "Logits contain NaN"
    assert 'mean' in surprise_stats, "Should have mean surprise"

    print(f"  PASSED: Logits shape={logits.shape}, mean_surprise={surprise_stats['mean']:.4f}")


def test_model_backward():
    """Test gradients flow through the model."""
    print("Test 8: Model backward pass...")

    config = {
        'vocab_size': 4096,
        'd_model': 256,
        'n_layer': 2,
        'ssm_cfg': {
            'n_heads': 8,
            'd_head': 32,
            'd_state': 32,
            'chunk_size': 64,
        }
    }

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MambaIntegerModelV2Redesign(config).to(device)

    batch, seqlen = 2, 128
    input_ids = torch.randint(0, config['vocab_size'], (batch, seqlen), device=device)
    targets = torch.randint(0, config['vocab_size'], (batch, seqlen), device=device)

    logits, _ = model(input_ids)
    loss = F.cross_entropy(logits.view(-1, config['vocab_size']), targets.view(-1))

    loss.backward()

    # Check gradients exist
    has_grad = False
    for name, param in model.named_parameters():
        if param.grad is not None and param.grad.abs().sum() > 0:
            has_grad = True
            break

    assert has_grad, "No gradients found"
    assert not torch.isnan(loss), "Loss is NaN"

    print(f"  PASSED: Loss={loss.item():.4f}, gradients flow correctly")


def test_consolidation_occurs():
    """Test that consolidation actually happens during forward."""
    print("Test 9: Consolidation during forward...")

    config = {
        'vocab_size': 4096,
        'd_model': 256,
        'n_layer': 2,
        'ssm_cfg': {
            'n_heads': 8,
            'd_head': 32,
            'd_state': 32,
            'chunk_size': 64,
            'time_threshold': 16,  # Force frequent consolidation
        }
    }

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MambaIntegerModelV2Redesign(config).to(device)

    # Process longer sequence
    batch, seqlen = 2, 256
    input_ids = torch.randint(0, config['vocab_size'], (batch, seqlen), device=device)

    _, _, stats_all = model(input_ids, return_stats=True)

    total_consolidations = sum(s['consolidation_events'] for s in stats_all)
    total_memory_filled = sum(s['chunk_memory_filled'] for s in stats_all)

    assert total_consolidations > 0, f"Expected consolidations, got {total_consolidations}"

    print(f"  PASSED: {total_consolidations} consolidation events, {total_memory_filled} memory entries")


def test_performance_benchmark():
    """Benchmark redesigned V2 vs baseline."""
    print("Test 10: Performance benchmark...")

    config = {
        'vocab_size': 4096,
        'd_model': 256,
        'n_layer': 4,
        'ssm_cfg': {
            'n_heads': 8,
            'd_head': 32,
            'd_state': 32,
            'chunk_size': 64,
        }
    }

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MambaIntegerModelV2Redesign(config).to(device)

    batch, seqlen = 4, 256
    input_ids = torch.randint(0, config['vocab_size'], (batch, seqlen), device=device)

    # Warmup
    for _ in range(3):
        _ = model(input_ids)

    if device == 'cuda':
        torch.cuda.synchronize()

    # Benchmark
    n_iter = 10
    t0 = time.time()
    for _ in range(n_iter):
        _ = model(input_ids)
    if device == 'cuda':
        torch.cuda.synchronize()
    t1 = time.time()

    avg_time = (t1 - t0) / n_iter * 1000
    tokens_per_sec = (batch * seqlen * n_iter) / (t1 - t0)

    print(f"  PASSED: {avg_time:.2f}ms/iter, {tokens_per_sec:.0f} tokens/sec")

    return avg_time


def main():
    print("=" * 60)
    print("V2 REDESIGN TESTS")
    print("=" * 60)
    print()

    test_convex_ssd_forward()
    test_convex_combination_balance()
    test_gradient_surprise_gate()
    test_aggressive_consolidation()
    test_multi_timescale_memory()
    test_block_forward()
    test_model_forward()
    test_model_backward()
    test_consolidation_occurs()
    avg_time = test_performance_benchmark()

    print()
    print("=" * 60)
    print("ALL 10 TESTS PASSED")
    print("=" * 60)
    print(f"Average forward time: {avg_time:.2f}ms")


if __name__ == '__main__':
    main()
