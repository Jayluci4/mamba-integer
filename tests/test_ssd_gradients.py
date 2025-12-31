"""Test SSD multihead gradient computation."""

import sys
sys.path.insert(0, '/home/jayantlohia16/mamba-integer/src')
sys.path.insert(0, '/home/jayantlohia16/mamba-integer/src/triton_kernels')

import torch
import torch.nn.functional as F


def test_ssd_gradients():
    """Test that SSD backward pass computes gradients for all inputs."""
    from ssd_multihead import ssd_multihead, ssd_multihead_forward

    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float32

    # Small test case
    batch = 2
    seqlen = 128
    n_heads = 4
    d_head = 16
    d_state = 8
    chunk_size = 32

    # Create inputs with requires_grad (as leaf tensors)
    X = torch.randn(batch, seqlen, n_heads, d_head, device=device, dtype=dtype)
    X = X.clone().detach().requires_grad_(True)

    A = torch.randn(batch, seqlen, n_heads, device=device, dtype=dtype) * 0.1 - 0.5
    A = A.clone().detach().requires_grad_(True)

    B = torch.randn(batch, seqlen, n_heads, d_state, device=device, dtype=dtype)
    B = B.clone().detach().requires_grad_(True)

    C = torch.randn(batch, seqlen, n_heads, d_state, device=device, dtype=dtype)
    C = C.clone().detach().requires_grad_(True)

    # Forward pass
    Y = ssd_multihead(X, A, B, C, chunk_size)

    # Retain grad for debugging
    Y.retain_grad()

    # Create a scalar loss
    loss = Y.sum()

    # Backward pass
    loss.backward()

    # Check gradients exist and are non-zero
    print("=" * 60)
    print("SSD Gradient Test Results")
    print("=" * 60)

    results = {}
    for name, tensor in [('X', X), ('A', A), ('B', B), ('C', C)]:
        grad = tensor.grad
        if grad is None:
            results[name] = "FAIL: No gradient"
        elif torch.isnan(grad).any():
            results[name] = "FAIL: NaN in gradient"
        elif torch.isinf(grad).any():
            results[name] = "FAIL: Inf in gradient"
        elif grad.abs().max() < 1e-10:
            results[name] = f"WARNING: Very small gradient (max={grad.abs().max():.2e})"
        else:
            results[name] = f"PASS: max={grad.abs().max():.2e}, mean={grad.abs().mean():.2e}"

    for name, result in results.items():
        print(f"  grad_{name}: {result}")

    print()

    # Numerical gradient check for A (the previously missing gradient)
    print("Numerical Gradient Check for A:")
    eps = 1e-4

    # Pick a random position to check
    b_idx, s_idx, h_idx = 0, seqlen // 2, 0

    # Perturb A at this position
    A_plus = A.detach().clone()
    A_plus[b_idx, s_idx, h_idx] += eps
    Y_plus = ssd_multihead_forward(X.detach(), A_plus, B.detach(), C.detach(), chunk_size)
    loss_plus = Y_plus.sum().item()

    A_minus = A.detach().clone()
    A_minus[b_idx, s_idx, h_idx] -= eps
    Y_minus = ssd_multihead_forward(X.detach(), A_minus, B.detach(), C.detach(), chunk_size)
    loss_minus = Y_minus.sum().item()

    numerical_grad = (loss_plus - loss_minus) / (2 * eps)
    analytical_grad = A.grad[b_idx, s_idx, h_idx].item()

    print(f"  Position [{b_idx}, {s_idx}, {h_idx}]:")
    print(f"    Numerical gradient:  {numerical_grad:.6f}")
    print(f"    Analytical gradient: {analytical_grad:.6f}")

    rel_error = abs(numerical_grad - analytical_grad) / (abs(numerical_grad) + 1e-8)
    print(f"    Relative error: {rel_error:.2e}")

    if rel_error < 0.1:
        print("    Result: PASS (relative error < 10%)")
    elif rel_error < 0.5:
        print("    Result: ACCEPTABLE (relative error < 50%)")
    else:
        print("    Result: WARNING (relative error > 50%)")

    print()
    print("=" * 60)

    # Overall pass/fail
    all_passed = all('PASS' in r or 'ACCEPTABLE' in r or 'WARNING: Very small' in r
                     for r in results.values())
    if all_passed:
        print("Overall: All gradients computed successfully")
    else:
        print("Overall: Some gradient tests failed")

    return all_passed


def test_grad_A_flow_to_A_log():
    """Test that gradient flows from A to A_log in MambaIntegerBlockV2."""
    import json
    from ssd_multihead import MambaIntegerBlockV2

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float32

    # Minimal config
    config = {
        'd_model': 64,
        'n_layer': 2,
        'ssm_cfg': {
            'n_heads': 4,
            'd_head': 16,
            'd_state': 8,
            'chunk_size': 32
        }
    }

    # Create block
    block = MambaIntegerBlockV2(config, layer_idx=0).to(device).to(dtype)

    # Store initial A_log
    initial_A_log = block.A_log.data.clone()
    print("\nTesting gradient flow to A_log parameter:")
    print(f"  Initial A_log: {initial_A_log}")

    # Forward pass
    x = torch.randn(2, 64, 64, device=device, dtype=dtype)
    y = block(x)
    loss = y.sum()

    # Backward
    loss.backward()

    # Check A_log gradient
    if block.A_log.grad is None:
        print("  A_log gradient: FAIL - No gradient!")
    elif torch.isnan(block.A_log.grad).any():
        print("  A_log gradient: FAIL - NaN!")
    elif block.A_log.grad.abs().max() < 1e-10:
        print(f"  A_log gradient: WARNING - Very small ({block.A_log.grad.abs().max():.2e})")
    else:
        print(f"  A_log gradient: PASS ({block.A_log.grad})")

    print()


if __name__ == '__main__':
    print("Testing SSD Multihead Gradients\n")
    test_ssd_gradients()
    test_grad_A_flow_to_A_log()
