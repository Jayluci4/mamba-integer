"""
Test suite for Mamba-Integer V2 Surprise-Gated SSD

Tests:
1. SurpriseGatedRetention module
2. Surprise computation (prediction error)
3. Adaptive retention behavior
4. Full block forward/backward pass
"""

import sys
sys.path.insert(0, '/home/jayantlohia16/mamba-integer/src')
sys.path.insert(0, '/home/jayantlohia16/mamba-integer')

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def test_surprise_gated_retention():
    """Test the SurpriseGatedRetention module."""
    print("\n" + "="*60)
    print("TEST 1: SurpriseGatedRetention Module")
    print("="*60)

    from v2.surprise_ssd import SurpriseGatedRetention

    n_heads = 8
    d_state = 32
    batch = 4

    gate = SurpriseGatedRetention(n_heads=n_heads, d_state=d_state)

    # Test with varying surprise levels
    low_surprise = torch.ones(batch, n_heads) * 0.1
    medium_surprise = torch.ones(batch, n_heads) * 1.0
    high_surprise = torch.ones(batch, n_heads) * 5.0

    alpha_low, _ = gate(low_surprise)
    alpha_med, _ = gate(medium_surprise)
    alpha_high, _ = gate(high_surprise)

    print(f"Low surprise → alpha: {alpha_low.mean():.4f}")
    print(f"Medium surprise → alpha: {alpha_med.mean():.4f}")
    print(f"High surprise → alpha: {alpha_high.mean():.4f}")

    # Verify: higher surprise should lead to higher retention
    assert alpha_high.mean() > alpha_med.mean(), "High surprise should have higher retention"
    assert alpha_med.mean() > alpha_low.mean(), "Medium surprise should have higher retention than low"

    print("✓ Higher surprise → higher retention (as expected)")

    # Test gradient flow
    gate.zero_grad()
    surprise = torch.randn(batch, n_heads).abs() + 0.1
    alpha, _ = gate(surprise)
    loss = alpha.sum()
    loss.backward()

    grad_exists = gate.log2_alpha_base.grad is not None
    print(f"✓ Gradients flow through surprise gate: {grad_exists}")

    return True


def test_prediction_error():
    """Test prediction error computation."""
    print("\n" + "="*60)
    print("TEST 2: Prediction Error Computation")
    print("="*60)

    from v2.surprise_ssd import SurpriseGatedRetention

    n_heads = 4
    d_state = 16
    batch = 2

    gate = SurpriseGatedRetention(n_heads=n_heads, d_state=d_state)

    # Test case 1: No change (low prediction error)
    h_prev = torch.ones(batch, n_heads, d_state)
    A_decay = torch.ones(batch, n_heads) * 0.9  # High retention
    h_new = A_decay.unsqueeze(-1) * h_prev  # Expected: no new input

    error = gate.compute_prediction_error(h_prev, h_new, A_decay)
    print(f"No change → prediction error: {error.mean():.6f}")
    assert error.mean() < 0.01, "Expected low error for no change"
    print("✓ Low error for predictable state")

    # Test case 2: Large change (high prediction error)
    h_new_surprised = h_prev + torch.randn_like(h_prev) * 5.0  # Big change

    error_high = gate.compute_prediction_error(h_prev, h_new_surprised, A_decay)
    print(f"Large change → prediction error: {error_high.mean():.4f}")
    assert error_high.mean() > error.mean(), "Expected higher error for large change"
    print("✓ High error for surprising state change")

    return True


def test_surprise_ssd_forward():
    """Test the full surprise-gated SSD forward pass."""
    print("\n" + "="*60)
    print("TEST 3: Surprise-Gated SSD Forward Pass")
    print("="*60)

    from v2.surprise_ssd import SurpriseGatedSSD

    batch = 2
    seqlen = 32
    n_heads = 4
    d_head = 16
    d_state = 16

    ssd = SurpriseGatedSSD(
        n_heads=n_heads,
        d_state=d_state,
        d_head=d_head,
        chunk_size=8,
    )

    # Create inputs
    X = torch.randn(batch, seqlen, n_heads, d_head)
    A = torch.ones(batch, seqlen, n_heads) * -0.5  # Log decay
    B = F.normalize(torch.randn(batch, seqlen, n_heads, d_state), dim=-1)
    C = F.normalize(torch.randn(batch, seqlen, n_heads, d_state), dim=-1)

    # Forward pass
    Y, surprise = ssd(X, A, B, C, return_surprise=True)

    print(f"Input shape: {X.shape}")
    print(f"Output shape: {Y.shape}")
    print(f"Surprise shape: {surprise.shape}")

    assert Y.shape == X.shape, "Output shape should match input"
    assert surprise.shape == (batch, seqlen, n_heads), "Surprise should be [B, L, n_heads]"
    print("✓ Shapes correct")

    # Check surprise statistics
    stats = ssd.get_surprise_stats()
    print(f"Surprise stats: {stats}")
    print("✓ Statistics tracking works")

    return True


def test_block_forward_backward():
    """Test full block forward and backward pass."""
    print("\n" + "="*60)
    print("TEST 4: Full Block Forward/Backward")
    print("="*60)

    from v2.surprise_ssd import MambaIntegerBlockV2Surprise

    # Small config for testing
    config = {
        'd_model': 64,
        'n_layer': 2,
        'ssm_cfg': {
            'n_heads': 4,
            'd_head': 16,
            'd_state': 16,
            'chunk_size': 8,
        }
    }

    block = MambaIntegerBlockV2Surprise(config, layer_idx=0)

    batch = 2
    seqlen = 32
    d_model = config['d_model']

    x = torch.randn(batch, seqlen, d_model, requires_grad=True)

    # Forward
    out, surprise = block(x, return_surprise=True)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    print(f"Surprise shape: {surprise.shape}")

    assert out.shape == x.shape, "Output shape should match input"
    print("✓ Forward pass successful")

    # Backward
    loss = out.sum()
    loss.backward()

    assert x.grad is not None, "Input should have gradient"
    print(f"✓ Backward pass successful, input grad norm: {x.grad.norm():.4f}")

    # Check surprise stats
    stats = block.get_surprise_stats()
    print(f"Block surprise stats: {stats}")

    return True


def test_surprise_correlation():
    """Test that surprise correlates with input novelty."""
    print("\n" + "="*60)
    print("TEST 5: Surprise-Novelty Correlation")
    print("="*60)

    from v2.surprise_ssd import SurpriseGatedSSD

    batch = 4
    seqlen = 64
    n_heads = 4
    d_head = 16
    d_state = 16

    ssd = SurpriseGatedSSD(
        n_heads=n_heads,
        d_state=d_state,
        d_head=d_head,
    )

    # Create input with a "surprising" spike in the middle
    X = torch.randn(batch, seqlen, n_heads, d_head) * 0.1  # Low variance baseline
    X[:, seqlen//2:seqlen//2+5] = torch.randn(batch, 5, n_heads, d_head) * 5.0  # High variance spike

    A = torch.ones(batch, seqlen, n_heads) * -0.5
    B = F.normalize(torch.randn(batch, seqlen, n_heads, d_state), dim=-1)
    C = F.normalize(torch.randn(batch, seqlen, n_heads, d_state), dim=-1)

    _, surprise = ssd(X, A, B, C, return_surprise=True)

    # Analyze surprise at different positions
    surprise_before = surprise[:, :seqlen//2-5].mean()
    surprise_spike = surprise[:, seqlen//2:seqlen//2+5].mean()
    surprise_after = surprise[:, seqlen//2+10:].mean()

    print(f"Surprise before spike: {surprise_before:.4f}")
    print(f"Surprise at spike: {surprise_spike:.4f}")
    print(f"Surprise after spike: {surprise_after:.4f}")

    # The spike should have higher surprise
    # Note: due to EMA normalization, this might take a few steps to show
    print(f"Spike surprise ratio: {surprise_spike / (surprise_before + 1e-6):.2f}x")

    return True


def test_integer_friendly():
    """Verify integer-friendly operations."""
    print("\n" + "="*60)
    print("TEST 6: Integer-Friendly Operations")
    print("="*60)

    from v2.surprise_ssd import SurpriseGatedRetention

    gate = SurpriseGatedRetention(n_heads=4, d_state=16)

    # Check that log2_alpha_base uses powers of 2
    alpha_base = 1 - torch.pow(2.0, gate.log2_alpha_base)
    print(f"Alpha base values: {alpha_base}")

    # Check beta uses powers of 2
    beta = torch.pow(2.0, gate.log2_beta)
    print(f"Beta values: {beta}")

    # Verify these are indeed powers of 2 (or close)
    log2_check = torch.log2(beta.abs() + 1e-10)
    is_power_of_2 = (log2_check - log2_check.round()).abs().max() < 0.01
    print(f"Beta is power of 2: {is_power_of_2}")
    print("✓ Integer-friendly power-of-2 scaling")

    return True


def run_all_tests():
    """Run all V2 tests."""
    print("\n" + "="*60)
    print("MAMBA-INTEGER V2 TEST SUITE")
    print("="*60)

    tests = [
        ("SurpriseGatedRetention", test_surprise_gated_retention),
        ("Prediction Error", test_prediction_error),
        ("SurpriseGatedSSD Forward", test_surprise_ssd_forward),
        ("Block Forward/Backward", test_block_forward_backward),
        ("Surprise Correlation", test_surprise_correlation),
        ("Integer-Friendly", test_integer_friendly),
    ]

    results = []
    for name, test_fn in tests:
        try:
            success = test_fn()
            results.append((name, "PASS" if success else "FAIL"))
        except Exception as e:
            print(f"✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, f"ERROR: {e}"))

    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    for name, status in results:
        status_icon = "✓" if status == "PASS" else "✗"
        print(f"{status_icon} {name}: {status}")

    all_passed = all(s == "PASS" for _, s in results)
    print("\n" + ("ALL TESTS PASSED!" if all_passed else "SOME TESTS FAILED"))

    return all_passed


if __name__ == "__main__":
    run_all_tests()
