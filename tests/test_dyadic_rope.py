
import torch
import time
from rational_bitnet import DyadicRoPE, RationalRoPE, RationalBitNetConfig, RationalBitNetAttention

def benchmark_rope():
    print("=== Benchmarking Dyadic vs Rational RoPE ===")
    
    # Config
    batch_size = 32
    seq_len = 2048
    num_heads = 8
    hidden_dim = 512
    head_dim = hidden_dim // num_heads # 64
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("Warning: CUDA not available. Skipping benchmark.")
        return

    print(f"Shape: [B={batch_size}, H={num_heads}, S={seq_len}, D={head_dim}]")
    
    # Init Models
    rational_rope = RationalRoPE(head_dim, max_position=seq_len).to(device)
    dyadic_rope = DyadicRoPE(head_dim, max_position=seq_len).to(device)
    
    # Init Data
    q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    pos_ids = torch.arange(seq_len, device=device).unsqueeze(0) # [1, S]
    
    # Warmup
    for _ in range(10):
        _ = rational_rope(q, k, pos_ids)
        _ = dyadic_rope(q, k, pos_ids)
    
    torch.cuda.synchronize()
    
    # Test Rational
    start = time.time()
    for _ in range(100):
        _ = rational_rope(q, k, pos_ids)
    torch.cuda.synchronize()
    rational_time = (time.time() - start) / 100
    print(f"RationalRoPE (Float): {rational_time*1000:.3f} ms")
    
    # Test Dyadic
    start = time.time()
    for _ in range(100):
        _ = dyadic_rope(q, k, pos_ids)
    torch.cuda.synchronize()
    dyadic_time = (time.time() - start) / 100
    print(f"DyadicRoPE (Integer): {dyadic_time*1000:.3f} ms")
    
    print(f"Speedup: {rational_time / dyadic_time:.2f}x")
    
    # Accuracy Check
    q_rat, _ = rational_rope(q, k, pos_ids)
    q_dya, _ = dyadic_rope(q, k, pos_ids)
    
    # Dyadic is an approximation of the continuous rotation.
    # RationalRoPE is also an approximation (Cayley).
    # They should be close but not identical due to quantization.
    diff = (q_rat - q_dya).abs().mean()
    print(f"Mean L1 Difference: {diff.item():.6f}")
    
    # Check if attention still works with this
    config = RationalBitNetConfig(
        hidden_dim=hidden_dim, 
        num_heads=num_heads, 
        max_seq_len=seq_len,
        use_dyadic_rope=True
    )
    attn = RationalBitNetAttention(config).to(device)
    
    # Run Attention forward
    hidden = torch.randn(batch_size, seq_len, hidden_dim, device=device)
    out = attn(hidden)
    print(f"Attention Output Shape: {out.shape}")
    print("Integration Successful.")

if __name__ == "__main__":
    # Add path to sys to find rational_bitnet
    import sys
    sys.path.append("../bitnet-odp/src")
    benchmark_rope()
