"""Test chunked dyadic scan correctness and performance."""

import sys
sys.path.insert(0, '/home/jayantlohia16/mamba-integer/src')
sys.path.insert(0, '/home/jayantlohia16/mamba-integer/src/triton_kernels')

import torch
import time


def test_chunked_scan_correctness():
    """Verify chunked scan produces same results as sequential scan."""
    from dyadic_scan import dyadic_scan_triton_fast, dyadic_scan_chunked

    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 60)
    print("Chunked Scan Correctness Test")
    print("=" * 60)

    test_cases = [
        (2, 128, 64, 32),   # Small
        (2, 256, 128, 64),  # Medium
        (2, 512, 256, 64),  # Larger
        (2, 1024, 512, 64), # Full training size
    ]

    all_passed = True
    for B, L, D, chunk_size in test_cases:
        # Generate test data
        u = torch.randn(B, L, D, device=device, dtype=torch.float32)
        # Decay numerators in range [0, 32000] for shift=15
        nums = torch.randint(8000, 28000, (B, L, D), device=device, dtype=torch.float32)

        # Run sequential scan
        h_seq = dyadic_scan_triton_fast(u, nums)

        # Run chunked scan
        h_chunked = dyadic_scan_chunked(u, nums, chunk_size=chunk_size)

        # Compare
        max_diff = (h_seq - h_chunked).abs().max().item()
        mean_diff = (h_seq - h_chunked).abs().mean().item()
        rel_error = max_diff / (h_seq.abs().max().item() + 1e-8)

        status = "PASS" if rel_error < 0.01 else "FAIL"
        if status == "FAIL":
            all_passed = False

        print(f"  B={B}, L={L}, D={D}, chunk={chunk_size}: "
              f"max_diff={max_diff:.2e}, rel_error={rel_error:.2e} [{status}]")

    print()
    return all_passed


def test_chunked_scan_gradients():
    """Verify gradients flow correctly through chunked scan."""
    from dyadic_scan import dyadic_scan_chunked_autograd

    torch.manual_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 60)
    print("Chunked Scan Gradient Test")
    print("=" * 60)

    B, L, D = 2, 256, 64
    chunk_size = 64

    u = torch.randn(B, L, D, device=device, dtype=torch.float32, requires_grad=True)
    nums = torch.randint(8000, 28000, (B, L, D), device=device, dtype=torch.float32)
    nums = nums.clone().detach().requires_grad_(True)

    # Forward
    h = dyadic_scan_chunked_autograd(u, nums, chunk_size)
    loss = h.sum()

    # Backward
    loss.backward()

    # Check gradients
    results = {}
    for name, tensor in [('u', u), ('nums', nums)]:
        grad = tensor.grad
        if grad is None:
            results[name] = "FAIL: No gradient"
        elif torch.isnan(grad).any():
            results[name] = "FAIL: NaN in gradient"
        elif grad.abs().max() < 1e-10:
            results[name] = f"WARNING: Very small ({grad.abs().max():.2e})"
        else:
            results[name] = f"PASS: max={grad.abs().max():.2e}"

    for name, result in results.items():
        print(f"  grad_{name}: {result}")

    print()
    return all('PASS' in r or 'WARNING' in r for r in results.values())


def benchmark_scan_performance():
    """Benchmark chunked vs sequential scan performance."""
    from dyadic_scan import dyadic_scan_triton_fast, dyadic_scan_chunked

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("Skipping benchmark (no CUDA)")
        return

    print("=" * 60)
    print("Performance Benchmark: Sequential vs Chunked")
    print("=" * 60)

    # Warmup
    u = torch.randn(2, 1024, 512, device=device, dtype=torch.float32)
    nums = torch.randint(8000, 28000, (2, 1024, 512), device=device, dtype=torch.float32)
    for _ in range(3):
        _ = dyadic_scan_triton_fast(u, nums)
        _ = dyadic_scan_chunked(u, nums, chunk_size=64)
    torch.cuda.synchronize()

    # Benchmark configurations
    configs = [
        (2, 512, 512),
        (2, 1024, 512),
        (4, 1024, 768),
        (8, 1024, 768),
    ]

    print(f"\n{'Config':<25} {'Sequential':<15} {'Chunked':<15} {'Speedup':<10}")
    print("-" * 65)

    for B, L, D in configs:
        u = torch.randn(B, L, D, device=device, dtype=torch.float32)
        nums = torch.randint(8000, 28000, (B, L, D), device=device, dtype=torch.float32)

        # Benchmark sequential
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(10):
            _ = dyadic_scan_triton_fast(u, nums)
        torch.cuda.synchronize()
        time_seq = (time.perf_counter() - start) / 10 * 1000  # ms

        # Benchmark chunked (try different chunk sizes)
        best_time_chunked = float('inf')
        best_chunk_size = 64
        for chunk_size in [32, 64, 128]:
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(10):
                _ = dyadic_scan_chunked(u, nums, chunk_size=chunk_size)
            torch.cuda.synchronize()
            time_chunked = (time.perf_counter() - start) / 10 * 1000

            if time_chunked < best_time_chunked:
                best_time_chunked = time_chunked
                best_chunk_size = chunk_size

        speedup = time_seq / best_time_chunked

        config_str = f"B={B}, L={L}, D={D}"
        print(f"{config_str:<25} {time_seq:.3f}ms{'':<8} {best_time_chunked:.3f}ms (cs={best_chunk_size}){'':<2} {speedup:.2f}x")

    print()


def test_large_sequence():
    """Test with sequence lengths similar to training."""
    from dyadic_scan import dyadic_scan_triton_fast, dyadic_scan_chunked

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cpu':
        print("Skipping large sequence test (no CUDA)")
        return

    print("=" * 60)
    print("Large Sequence Test (L=1024, D=768)")
    print("=" * 60)

    B, L, D = 2, 1024, 768

    u = torch.randn(B, L, D, device=device, dtype=torch.float32)
    nums = torch.randint(8000, 28000, (B, L, D), device=device, dtype=torch.float32)

    # Sequential
    h_seq = dyadic_scan_triton_fast(u, nums)

    # Chunked with chunk_size=64
    h_chunked = dyadic_scan_chunked(u, nums, chunk_size=64)

    max_diff = (h_seq - h_chunked).abs().max().item()
    rel_error = max_diff / (h_seq.abs().max().item() + 1e-8)

    print(f"  Max absolute difference: {max_diff:.2e}")
    print(f"  Relative error: {rel_error:.2e}")
    print(f"  Status: {'PASS' if rel_error < 0.01 else 'FAIL'}")
    print()


if __name__ == '__main__':
    print("\nTesting Chunked Dyadic Scan\n")

    correctness_ok = test_chunked_scan_correctness()
    gradient_ok = test_chunked_scan_gradients()
    test_large_sequence()
    benchmark_scan_performance()

    print("=" * 60)
    if correctness_ok and gradient_ok:
        print("All tests PASSED")
    else:
        print("Some tests FAILED")
    print("=" * 60)
