
import torch
import ctypes
import os
import time

class DyadicMambaScan(torch.nn.Module):
    def __init__(self, scale_bits=20):
        super().__init__()
        self.scale_bits = scale_bits
        
        # Load CUDA lib
        lib_path = os.path.join(os.path.dirname(__file__), "cuda/libdyadic_mamba.so")
        if os.path.exists(lib_path):
            self.lib = ctypes.CDLL(lib_path)
            self.lib.launch_dyadic_scan.argtypes = [
                ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int
            ]
            print(f"Loaded Dyadic Mamba Kernel from {lib_path}")
        else:
            print("CUDA library not found. Using CPU fallback.")
            self.lib = None

    def forward(self, x, decay_nums, decay_shifts):
        # x: [B, S, D]
        if self.lib and x.is_cuda:
            B, S, D = x.shape
            h_out = torch.empty_like(x)
            
            self.lib.launch_dyadic_scan(
                ctypes.c_void_p(x.contiguous().data_ptr()),
                ctypes.c_void_p(decay_nums.contiguous().data_ptr()),
                ctypes.c_void_p(decay_shifts.contiguous().data_ptr()),
                ctypes.c_void_p(h_out.data_ptr()),
                ctypes.c_int(B),
                ctypes.c_int(S),
                ctypes.c_int(D),
                ctypes.c_int(self.scale_bits)
            )
            return h_out
        else:
            return self.python_forward(x, decay_nums, decay_shifts)

    def python_forward(self, x, nums, shifts):
        # Simulation of the integer logic
        B, S, D = x.shape
        h = torch.zeros(B, D, device=x.device, dtype=torch.double)
        output = []
        
        scale = 2.0**self.scale_bits
        
        for t in range(S):
            x_val = x[:, t, :] * scale
            n = nums[:, t, :]
            s = shifts[:, t, :]
            
            # Recurrence
            # h = (h * n) >> s + x
            # Since h is float here, we simulate >> s as / 2^s
            
            factor = n.double() / (2.0 ** s.double())
            h = h * factor + x_val
            
            output.append(h / scale)
            
        return torch.stack(output, dim=1)

def benchmark():
    print("--- Benchmarking Dyadic Mamba Scan ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    B, S, D = 32, 1024, 256
    model = DyadicMambaScan().to(device)
    
    # Inputs
    x = torch.randn(B, S, D, device=device)
    
    # Decays: A ~ 0.9.  0.9 ~ 29491 / 32768 (2^15)
    # Let's say we want variable decays.
    # nums: random integers 
    # shifts: constant 16
    decay_nums = torch.randint(20000, 32000, (B, S, D), dtype=torch.int32, device=device)
    decay_shifts = torch.full((B, S, D), 15, dtype=torch.int32, device=device)
    
    # Warmup
    _ = model(x, decay_nums, decay_shifts)
    torch.cuda.synchronize()
    
    # Run
    start = time.time()
    for _ in range(50):
        out = model(x, decay_nums, decay_shifts)
    torch.cuda.synchronize()
    gpu_time = (time.time() - start) / 50
    print(f"Dyadic Mamba (CUDA): {gpu_time*1000:.3f} ms")
    
    # CPU/Python Check (Small batch for speed)
    print("\nVerifying Correctness (Small Batch)...")
    B_small = 1
    x_s = x[:B_small, :100, :].contiguous()
    n_s = decay_nums[:B_small, :100, :].contiguous()
    s_s = decay_shifts[:B_small, :100, :].contiguous()
    
    out_cuda = model(x_s, n_s, s_s)
    out_py = model.python_forward(x_s, n_s, s_s)
    
    diff = (out_cuda - out_py).abs().max()
    print(f"Max Difference: {diff.item():.6f}")
    if diff < 1e-4:
        print("✅ SUCCESS: Matches Python logic")
    else:
        print("❌ FAILURE: Mismatch")

if __name__ == "__main__":
    benchmark()
