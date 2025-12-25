"""Profile training to find bottlenecks."""
import os
import sys
import torch
import torch.nn as nn
import json
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
sys.path.insert(0, "/home/jayantlohia16/experiment/gemma-intelligent/conv/src/bitnet-odp/src")

from mamba_integer_model import MambaIntegerModel

# Load config
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "../configs/config_mamba_integer_l4.json")
with open(CONFIG_PATH, 'r') as f:
    config = json.load(f)

print("Creating model...")
model = MambaIntegerModel(config).cuda()
model.train()

# Create sample data
x = torch.randint(0, config['vocab_size'], (2, 512), device='cuda')
y = torch.randint(0, config['vocab_size'], (2, 512), device='cuda')
criterion = nn.CrossEntropyLoss()

# Warmup
print("Warmup...")
for _ in range(3):
    logits = model(x)
    loss = criterion(logits.view(-1, config['vocab_size']), y.view(-1))
    loss.backward()
    model.zero_grad()
torch.cuda.synchronize()

# Profile with PyTorch profiler
print("\nProfiling forward + backward pass...")
with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ],
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    logits = model(x)
    loss = criterion(logits.view(-1, config['vocab_size']), y.view(-1))
    loss.backward()
    torch.cuda.synchronize()

# Print results sorted by CUDA time
print("\n" + "="*80)
print("TOP 30 OPERATIONS BY CUDA TIME")
print("="*80)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))

# Print results sorted by CPU time
print("\n" + "="*80)
print("TOP 20 OPERATIONS BY CPU TIME")
print("="*80)
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))

# Group by input shape to understand memory access patterns
print("\n" + "="*80)
print("TOP 20 BY SELF CUDA TIME (actual kernel time)")
print("="*80)
print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20))

# Export trace for visualization
trace_path = "/home/jayantlohia16/experiment/mamba-integer/profile_trace.json"
prof.export_chrome_trace(trace_path)
print(f"\nTrace exported to: {trace_path}")
print("View in Chrome at: chrome://tracing")
