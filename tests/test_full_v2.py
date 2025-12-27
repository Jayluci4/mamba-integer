"""
Test suite for Mamba-Integer V2 Full Architecture

Tests all four phases:
1. Surprise-based retention
2. Multi-timescale memory (L0-L3)
3. Consolidation triggers
4. Selective forgetting + synaptic homeostasis
"""

import sys
sys.path.insert(0, '/home/jayantlohia16/mamba-integer/src')

import torch
import torch.nn.functional as F
import time


# =============================================================================
# Test 1: Surprise Gate
# =============================================================================

def test_surprise_gate():
    """Test surprise-based retention computation."""
    print("\n" + "=" * 60)
    print("TEST 1: Surprise Gate")
    print("=" * 60)

    from v2.full_v2_architecture import SurpriseGate

    gate = SurpriseGate(n_heads=4, d_state=16)

    batch = 2
    n_heads = 4
    d_state = 16
    d_head = 8

    # Case 1: Low prediction error -> low surprise -> lower retention
    h_prev = torch.randn(batch, n_heads, d_state, d_head)
    decay = torch.ones(batch, n_heads) * 0.9
    h_actual = decay.unsqueeze(-1).unsqueeze(-1) * h_prev  # Exactly as expected

    error_low = gate.compute_prediction_error(h_actual, h_prev, decay)
    alpha_low, _ = gate(error_low)
    print(f"Low prediction error -> alpha = {alpha_low.mean():.4f}")

    # Case 2: High prediction error -> high surprise -> higher retention
    h_actual_high = h_prev + torch.randn_like(h_prev) * 2.0
    error_high = gate.compute_prediction_error(h_actual_high, h_prev, decay)
    alpha_high, _ = gate(error_high)
    print(f"High prediction error -> alpha = {alpha_high.mean():.4f}")

    assert alpha_high.mean() > alpha_low.mean(), "High surprise should increase retention"
    print("PASS: High surprise increases retention")

    return True


# =============================================================================
# Test 2: Multi-Timescale Memory
# =============================================================================

def test_multi_timescale_memory():
    """Test 4-level memory hierarchy."""
    print("\n" + "=" * 60)
    print("TEST 2: Multi-Timescale Memory")
    print("=" * 60)

    from v2.full_v2_architecture import MultiTimescaleMemory

    d_model = 64
    n_heads = 4
    d_state = 16
    d_head = 16

    memory = MultiTimescaleMemory(
        d_model=d_model,
        n_heads=n_heads,
        d_state=d_state,
        d_head=d_head,
        n_persistent=32,
        chunk_memory_size=64,
    ).cuda()

    batch = 2
    seqlen = 32

    # Test L3 (persistent) is initialized
    print(f"L3 persistent memory shape: {memory.persistent_memory.shape}")
    assert memory.persistent_memory.shape == (n_heads, 32, d_state)
    print("PASS: L3 persistent memory initialized")

    # Test writing to L2 (chunk memory)
    chunk_state = torch.randn(batch, n_heads, d_state, d_head).cuda()
    importance = torch.ones(batch, n_heads).cuda()

    memory.write_chunk_summary(chunk_state, importance)
    assert memory.chunk_memory_filled.item() == 1
    print("PASS: L2 chunk memory write")

    # Write multiple times
    for _ in range(5):
        memory.write_chunk_summary(chunk_state, importance)
    assert memory.chunk_memory_filled.item() == 6
    print(f"PASS: L2 chunk memory filled = {memory.chunk_memory_filled.item()}")

    # Test reading from memory
    query = torch.randn(batch, seqlen, d_model).cuda()
    memory_out = memory.read_memory(query)
    assert memory_out.shape == (batch, seqlen, d_model)
    print(f"PASS: Memory read output shape = {memory_out.shape}")

    # Test integration
    x = torch.randn(batch, seqlen, d_model).cuda()
    integrated = memory.integrate(x, memory_out)
    assert integrated.shape == x.shape
    print("PASS: Memory integration")

    return True


# =============================================================================
# Test 3: Consolidation Triggers
# =============================================================================

def test_consolidation_triggers():
    """Test three consolidation trigger mechanisms."""
    print("\n" + "=" * 60)
    print("TEST 3: Consolidation Triggers")
    print("=" * 60)

    from v2.full_v2_architecture import ConsolidationTrigger

    trigger = ConsolidationTrigger(
        capacity_threshold=0.9,
        surprise_threshold=2.0,
        time_threshold=10,
    )

    # Test time-based trigger
    # time_threshold=10 means trigger when tokens_since_consolidation > 10
    # So we need 11 calls (0-10 don't trigger, 11th does)
    for i in range(11):
        should, reasons = trigger.should_consolidate(0.1, 0.5)
        if should:
            print(f"  Triggered at step {i+1}: {reasons}")
            break

    # Should have triggered by now
    assert trigger.consolidation_count.item() >= 1, "Time trigger should have fired"
    print("PASS: Time-based trigger works")

    # Test surprise-based trigger (after reset)
    trigger.reset()
    for i in range(3):
        should, reasons = trigger.should_consolidate(0.5, 0.5)  # Cumulative = 1.5

    should, reasons = trigger.should_consolidate(0.6, 0.5)  # Cumulative = 2.1 > threshold
    assert should and 'surprise' in reasons
    print("PASS: Surprise-based trigger works")

    # Test capacity-based trigger
    trigger.reset()
    should, reasons = trigger.should_consolidate(0.1, 0.95)  # High capacity
    assert should and 'capacity' in reasons
    print("PASS: Capacity-based trigger works")

    print(f"Stats: {trigger.get_stats()}")
    return True


# =============================================================================
# Test 4: Importance Scorer
# =============================================================================

def test_importance_scorer():
    """Test importance scoring for consolidation."""
    print("\n" + "=" * 60)
    print("TEST 4: Importance Scorer")
    print("=" * 60)

    from v2.full_v2_architecture import ImportanceScorer

    n_heads = 4
    d_state = 16
    d_head = 8

    scorer = ImportanceScorer(n_heads, d_state, method='surprise_weighted')

    batch = 2
    states = torch.randn(batch, n_heads, d_state, d_head)
    surprise = torch.rand(batch, n_heads)

    importance = scorer.compute_importance(states, surprise)
    assert importance.shape == (batch, n_heads)
    print(f"PASS: Importance shape = {importance.shape}")

    # Higher surprise should lead to higher importance
    low_surprise = torch.ones(batch, n_heads) * 0.1
    high_surprise = torch.ones(batch, n_heads) * 2.0

    imp_low = scorer.compute_importance(states, low_surprise)
    imp_high = scorer.compute_importance(states, high_surprise)

    assert imp_high.mean() > imp_low.mean()
    print("PASS: Higher surprise -> higher importance")

    return True


# =============================================================================
# Test 5: Selective Forgetting
# =============================================================================

def test_selective_forgetting():
    """Test selective forgetting with importance protection."""
    print("\n" + "=" * 60)
    print("TEST 5: Selective Forgetting")
    print("=" * 60)

    from v2.full_v2_architecture import SelectiveForgetting

    d_model = 64
    n_heads = 4
    d_state = 16
    d_head = 8

    forget = SelectiveForgetting(d_model, n_heads).cuda()

    batch = 2
    h_prev = torch.randn(batch, n_heads, d_state, d_head).cuda()
    x = torch.randn(batch, 1, d_model).cuda()

    # Low importance -> should forget more
    low_importance = torch.ones(batch, n_heads).cuda() * 0.1
    h_low, gate_low = forget(h_prev, x, low_importance)

    # High importance -> should forget less (protection)
    high_importance = torch.ones(batch, n_heads).cuda() * 2.0
    h_high, gate_high = forget(h_prev, x, high_importance)

    # High importance should preserve more
    diff_low = (h_prev - h_low).abs().mean()
    diff_high = (h_prev - h_high).abs().mean()

    print(f"Low importance: state change = {diff_low:.4f}")
    print(f"High importance: state change = {diff_high:.4f}")

    assert diff_high <= diff_low + 1e-5  # High importance should change less
    print("PASS: High importance protects from forgetting")

    return True


# =============================================================================
# Test 6: Synaptic Homeostasis
# =============================================================================

def test_synaptic_homeostasis():
    """Test periodic downscaling based on importance."""
    print("\n" + "=" * 60)
    print("TEST 6: Synaptic Homeostasis")
    print("=" * 60)

    from v2.full_v2_architecture import SynapticHomeostasis

    n_heads = 4
    d_state = 16
    d_head = 8

    homeostasis = SynapticHomeostasis(
        n_heads, d_state,
        downscale_factor=0.9,
        homeostasis_interval=4,
    )

    batch = 2
    state = torch.ones(batch, n_heads, d_state, d_head) * 1.0

    # Low importance -> should downscale
    low_importance = torch.ones(batch, n_heads) * 0.1
    high_importance = torch.ones(batch, n_heads) * 1.0

    # Run until homeostasis triggers
    for _ in range(3):
        state_low = homeostasis.maybe_apply(state.clone(), low_importance)
        state_high = homeostasis.maybe_apply(state.clone(), high_importance)

    # 4th call triggers homeostasis
    state_low = homeostasis.maybe_apply(state.clone(), low_importance)

    print(f"Homeostasis count: {homeostasis.homeostasis_count.item()}")
    assert homeostasis.homeostasis_count.item() >= 1
    print("PASS: Homeostasis triggered")

    return True


# =============================================================================
# Test 7: Full V2 Block
# =============================================================================

def test_full_v2_block():
    """Test complete V2 block with all components."""
    print("\n" + "=" * 60)
    print("TEST 7: Full V2 Block Forward/Backward")
    print("=" * 60)

    from v2.full_v2_architecture import MambaIntegerBlockV2Full

    config = {
        'd_model': 64,
        'n_layer': 2,
        'ssm_cfg': {
            'n_heads': 4,
            'd_head': 16,
            'd_state': 16,
            'chunk_size': 32,
            'n_persistent': 16,
            'chunk_memory_size': 32,
            'homeostasis_interval': 8,
        }
    }

    block = MambaIntegerBlockV2Full(config, layer_idx=0).cuda()

    batch = 2
    seqlen = 64
    x = torch.randn(batch, seqlen, 64).cuda()
    x.requires_grad_(True)

    # Forward with stats
    out, surprise, stats = block(x, return_surprise=True, return_stats=True)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Surprise shape: {surprise.shape}")
    assert out.shape == x.shape
    print("PASS: Forward pass shapes correct")

    # Check stats
    print(f"\nStats:")
    print(f"  Mean surprise: {stats['mean_surprise']:.4f}")
    print(f"  Consolidation events: {stats['consolidation_events']}")
    print(f"  Homeostasis count: {stats['homeostasis_count']}")
    print(f"  Chunk memory filled: {stats['chunk_memory_filled']}")

    # Backward
    loss = out.sum()
    loss.backward()

    has_grads = any(p.grad is not None for p in block.parameters() if p.requires_grad)
    assert has_grads
    print("PASS: Backward pass completed")

    return True


# =============================================================================
# Test 8: Full V2 Model
# =============================================================================

def test_full_v2_model():
    """Test complete V2 model."""
    print("\n" + "=" * 60)
    print("TEST 8: Full V2 Model")
    print("=" * 60)

    from v2.full_v2_architecture import MambaIntegerModelV2Full

    config = {
        'vocab_size': 1000,
        'd_model': 64,
        'n_layer': 2,
        'ssm_cfg': {
            'n_heads': 4,
            'd_head': 16,
            'd_state': 16,
            'chunk_size': 32,
        }
    }

    model = MambaIntegerModelV2Full(config).cuda()

    print(f"Parameters: {model.count_parameters():,}")

    batch = 2
    seqlen = 64
    input_ids = torch.randint(0, 1000, (batch, seqlen)).cuda()

    # Forward
    logits, surprise_stats = model(input_ids, return_surprise=True)

    print(f"Logits shape: {logits.shape}")
    print(f"Surprise stats: {surprise_stats}")
    assert logits.shape == (batch, seqlen, 1000)
    print("PASS: Model forward")

    # Training step
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    for step in range(3):
        input_ids = torch.randint(0, 1000, (batch, seqlen)).cuda()
        logits, _ = model(input_ids)

        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, 1000),
            input_ids[:, 1:].reshape(-1)
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Step {step}: loss = {loss.item():.4f}")

    print("PASS: Training loop completed")

    # Get all stats
    all_stats = model.get_all_stats()
    print(f"\nLayer stats:")
    for s in all_stats:
        print(f"  Layer {s['layer_idx']}: surprise_ema={s['surprise_ema']:.4f}, "
              f"chunk_mem={s['chunk_memory_filled']}")

    return True


# =============================================================================
# Test 9: Speed Benchmark
# =============================================================================

def test_speed_benchmark():
    """Benchmark V2 full vs V2 chunked (surprise only)."""
    print("\n" + "=" * 60)
    print("TEST 9: Speed Benchmark")
    print("=" * 60)

    from v2.full_v2_architecture import MambaIntegerBlockV2Full
    from v2.chunked_surprise_ssd import MambaIntegerBlockV2Chunked

    config = {
        'd_model': 256,
        'n_layer': 12,
        'ssm_cfg': {
            'n_heads': 8,
            'd_head': 32,
            'd_state': 32,
            'chunk_size': 64,
        }
    }

    block_full = MambaIntegerBlockV2Full(config, layer_idx=0).cuda()
    block_simple = MambaIntegerBlockV2Chunked(config, layer_idx=0).cuda()

    batch = 4
    seqlen = 256
    x = torch.randn(batch, seqlen, 256).cuda()

    # Warmup
    for _ in range(3):
        _ = block_full(x)
        _ = block_simple(x)
    torch.cuda.synchronize()

    # Benchmark full V2
    n_iters = 10
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_iters):
        _ = block_full(x)
    torch.cuda.synchronize()
    t_full = (time.time() - t0) / n_iters * 1000

    # Benchmark simple V2
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n_iters):
        _ = block_simple(x)
    torch.cuda.synchronize()
    t_simple = (time.time() - t0) / n_iters * 1000

    print(f"Full V2 (all components): {t_full:.1f} ms")
    print(f"Simple V2 (surprise only): {t_simple:.1f} ms")
    print(f"Overhead: {t_full / t_simple:.2f}x")

    # Full should be slower but not by more than 10x
    # (includes memory hierarchy, consolidation, forgetting, homeostasis)
    assert t_full / t_simple < 10.0, "Full V2 overhead too high"
    print("PASS: Overhead within acceptable range")

    return True


# =============================================================================
# Test 10: Memory Consolidation Flow
# =============================================================================

def test_consolidation_flow():
    """Test that consolidation actually happens during forward pass."""
    print("\n" + "=" * 60)
    print("TEST 10: Consolidation Flow")
    print("=" * 60)

    from v2.full_v2_architecture import MambaIntegerBlockV2Full

    config = {
        'd_model': 64,
        'n_layer': 2,
        'ssm_cfg': {
            'n_heads': 4,
            'd_head': 16,
            'd_state': 16,
            'chunk_size': 32,
            'time_threshold': 2,  # Trigger consolidation quickly
            'chunk_memory_size': 16,
        }
    }

    block = MambaIntegerBlockV2Full(config, layer_idx=0).cuda()

    batch = 2
    x = torch.randn(batch, 128, 64).cuda()  # 4 chunks

    # Run forward
    _, _, stats = block(x, return_stats=True)

    print(f"Consolidation events: {stats['consolidation_events']}")
    print(f"Chunk memory filled: {stats['chunk_memory_filled']}")

    # Should have some consolidation
    assert stats['consolidation_events'] > 0 or stats['chunk_memory_filled'] > 0
    print("PASS: Consolidation occurred")

    return True


# =============================================================================
# Run All Tests
# =============================================================================

def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("MAMBA-INTEGER V2 FULL ARCHITECTURE TEST SUITE")
    print("=" * 60)

    tests = [
        ("Surprise Gate", test_surprise_gate),
        ("Multi-Timescale Memory", test_multi_timescale_memory),
        ("Consolidation Triggers", test_consolidation_triggers),
        ("Importance Scorer", test_importance_scorer),
        ("Selective Forgetting", test_selective_forgetting),
        ("Synaptic Homeostasis", test_synaptic_homeostasis),
        ("Full V2 Block", test_full_v2_block),
        ("Full V2 Model", test_full_v2_model),
        ("Speed Benchmark", test_speed_benchmark),
        ("Consolidation Flow", test_consolidation_flow),
    ]

    results = []
    for name, test_fn in tests:
        try:
            success = test_fn()
            results.append((name, "PASS" if success else "FAIL"))
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, f"ERROR: {e}"))

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for name, status in results:
        icon = "PASS" if status == "PASS" else "FAIL"
        print(f"[{icon}] {name}")

    all_passed = all(s == "PASS" for _, s in results)
    print("\n" + ("ALL TESTS PASSED!" if all_passed else "SOME TESTS FAILED"))

    return all_passed


if __name__ == "__main__":
    run_all_tests()
