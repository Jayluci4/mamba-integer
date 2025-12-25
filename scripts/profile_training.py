"""Profile training to find actual bottlenecks."""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity
import time

from mamba_integer_model import MambaIntegerModel

# Config matching training
config = {
    'vocab_size': 4096,
    'd_model': 768,
    'n_layer': 24,
    'ssm_cfg': {'d_state': 16, 'dt_rank': 48}
}

device = 'cuda'
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

print("Creating model...")
model = MambaIntegerModel(config).to(device)
model.train()

# Dummy data
B, L = 2, 512
x = torch.randint(0, 4096, (B, L), device=device)
y = torch.randint(0, 4096, (B, L), device=device)

# Warmup
print("Warming up...")
for _ in range(3):
    logits = model(x)
    loss = nn.functional.cross_entropy(logits.view(-1, 4096), y.view(-1))
    loss.backward()
    model.zero_grad()
torch.cuda.synchronize()

# Profile
print("\nProfiling forward + backward pass...")
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=False,
    with_stack=False,
) as prof:
    for _ in range(5):
        with record_function("forward"):
            logits = model(x)
        with record_function("loss"):
            loss = nn.functional.cross_entropy(logits.view(-1, 4096), y.view(-1))
        with record_function("backward"):
            loss.backward()
        with record_function("zero_grad"):
            model.zero_grad()
    torch.cuda.synchronize()

# Print results sorted by CUDA time
print("\n" + "="*80)
print("TOP 20 OPERATIONS BY CUDA TIME")
print("="*80)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

print("\n" + "="*80)
print("TOP 20 OPERATIONS BY CPU TIME")
print("="*80)
print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))

# Export chrome trace for detailed view
prof.export_chrome_trace("/home/jayantlohia16/experiment/mamba-integer/profile_trace.json")
print("\nChrome trace saved to profile_trace.json")
print("Open chrome://tracing and load the file for detailed visualization")
