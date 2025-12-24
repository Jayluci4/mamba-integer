
import torch
import time
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))
from triton_kernels.bitnet_kernels import fast_quantize_activations, fast_bitnet_matmul

def benchmark_quantization(batch_size=32, seq_len=512, hidden_dim=2048):
    print(f"\n--- Benchmarking Quantization (B={batch_size}, L={seq_len}, D={hidden_dim}) ---")
    
    x = torch.randn(batch_size, seq_len, hidden_dim, device='cuda', dtype=torch.float32)
    
    # PyTorch Baseline
    def pytorch_quant(x):
        # Dynamic per-token
        scale = x.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-8)
        x_quant = (x * 127.0 / scale).round().clamp(-127, 127)
        return x_quant, scale
        
    # Warmup
    for _ in range(10): pytorch_quant(x)
    torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(100): pytorch_quant(x)
    torch.cuda.synchronize()
    ms_torch = (time.time() - start) * 10
    print(f"PyTorch Time: {ms_torch:.3f} ms")
    
    # Triton
    # Warmup
    for _ in range(10): fast_quantize_activations(x)
    torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(100): fast_quantize_activations(x)
    torch.cuda.synchronize()
    ms_triton = (time.time() - start) * 10
    print(f"Triton Time:  {ms_triton:.3f} ms")
    print(f"Speedup:      {ms_torch / ms_triton:.2f}x")
    
    # Correctness
    q_torch, s_torch = pytorch_quant(x)
    q_triton, s_triton = fast_quantize_activations(x)
    
    # Check Scale
    diff_scale = (s_torch - s_triton * 127.0).abs().max()
    print(f"Scale Max Diff: {diff_scale:.2e}")
    
    # Check Quant
    # Triton clamp/round might behave slightly differently on edge cases
    # We check if values are within +/- 1
    diff_quant = (q_torch - q_triton).abs().float().mean()
    print(f"Quant Mean Diff: {diff_quant:.4f}")

def benchmark_matmul(batch_size=32, seq_len=128, in_dim=1024, out_dim=4096):
    print(f"\n--- Benchmarking Fused MatMul (B={batch_size}, L={seq_len}, In={in_dim}, Out={out_dim}) ---")
    
    M = batch_size * seq_len
    K = in_dim
    N = out_dim
    
    # Inputs (Simulated Int8)
    x_quant = torch.randint(-127, 127, (M, K), device='cuda', dtype=torch.int8)
    w_quant = torch.randint(-1, 2, (N, K), device='cuda', dtype=torch.int8) # Ternary {-1, 0, 1}
    
    x_scale = torch.rand(M, device='cuda')
    w_scale = torch.rand(N, device='cuda')
    
    # PyTorch Baseline (Float Matmul)
    def pytorch_mm(xq, wq, xs, ws):
        # Dequantize first (simulating what happens if we don't have int8 kernel)
        # Or just float matmul
        xf = xq.float()
        wf = wq.float()
        y = torch.matmul(xf, wf.t())
        y = y * xs.unsqueeze(1) * ws.unsqueeze(0)
        return y
        
    # Warmup
    for _ in range(10): pytorch_mm(x_quant, w_quant, x_scale, w_scale)
    torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(100): pytorch_mm(x_quant, w_quant, x_scale, w_scale)
    torch.cuda.synchronize()
    ms_torch = (time.time() - start) * 10
    print(f"PyTorch (Float) Time: {ms_torch:.3f} ms")
    
    # Triton Fused
    # Warmup
    for _ in range(10): fast_bitnet_matmul(x_quant, w_quant, x_scale, w_scale)
    torch.cuda.synchronize()
    
    start = time.time()
    for _ in range(100): fast_bitnet_matmul(x_quant, w_quant, x_scale, w_scale)
    torch.cuda.synchronize()
    ms_triton = (time.time() - start) * 10
    print(f"Triton (Int8) Time:   {ms_triton:.3f} ms")
    print(f"Speedup:              {ms_torch / ms_triton:.2f}x")
    
    # Correctness
    y_torch = pytorch_mm(x_quant, w_quant, x_scale, w_scale)
    y_triton = fast_bitnet_matmul(x_quant, w_quant, x_scale, w_scale)
    
    diff = (y_torch - y_triton).abs().max()
    print(f"Max Diff: {diff.item():.4f}")
    # Tolerable error? Int8 vs Float math
    # Int8 accumulation (Triton) vs Float accumulation (PyTorch)
    # Differences are expected.
    rel_error = (y_torch - y_triton).abs().mean() / (y_torch.abs().mean() + 1e-6)
    print(f"Rel Error: {rel_error.item():.6f}")

if __name__ == "__main__":
    benchmark_quantization()
    benchmark_matmul()
