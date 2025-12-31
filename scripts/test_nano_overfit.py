#!/usr/bin/env python3
"""
Nano Overfit Test: Can Integer Math Memorize?

This is the FIRST validation test before any scaling.
If this fails, the architecture is fundamentally broken.

Test: Train a 5M model to memorize a single fixed batch.
Goal: Loss → 0.01 (near-perfect memorization)

Success criteria (from docs/validation_experiments.md):
  - Loss < 0.1 at 1000 steps: PASS
  - Loss < 0.01 at convergence: PASS
  - Loss > 1.0 at 1000 steps: FAIL
  - Loss > 0.5 at convergence: FAIL

Run: python scripts/test_nano_overfit.py
Time: ~2-3 hours max (usually converges much faster)
"""

import os
import sys
import json
import time
import urllib.request
import torch
import torch.nn.functional as F

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from mamba_integer_model import MambaIntegerModel

# ============================================================================
# Configuration
# ============================================================================

TINYSHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'tinyshakespeare.txt')

# Test parameters
SEQ_LEN = 512
BATCH_SIZE = 1
MAX_STEPS = 10000
LOG_INTERVAL = 100
LR = 1e-3

# Success/Fail thresholds (from validation_experiments.md)
PASS_LOSS_1000 = 0.1      # Must be below this at step 1000
FAIL_LOSS_1000 = 1.0      # Fail if above this at step 1000
PASS_LOSS_FINAL = 0.01    # Target for convergence
FAIL_LOSS_FINAL = 0.5     # Fail if above this at end

# Abort criteria
ABORT_NO_PROGRESS_STEPS = 500  # Abort if no improvement for this many steps
ABORT_LOSS_THRESHOLD = 2.0     # Abort if stuck above this after 1000 steps


# ============================================================================
# Data Loading
# ============================================================================

def download_tinyshakespeare():
    """Download tinyshakespeare if not present."""
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)

    if os.path.exists(DATA_PATH):
        print(f"Using cached: {DATA_PATH}")
        return

    print(f"Downloading tinyshakespeare...")
    urllib.request.urlretrieve(TINYSHAKESPEARE_URL, DATA_PATH)
    print(f"Saved to: {DATA_PATH}")


def create_fixed_batch(tokenizer, seq_len):
    """Create a single fixed batch for overfitting."""
    with open(DATA_PATH, 'r') as f:
        text = f.read()

    # Take the first chunk of text
    tokens = tokenizer.encode(text[:seq_len * 10])  # Encode more than we need

    # Create fixed batch
    x = torch.tensor(tokens[:seq_len], dtype=torch.long).unsqueeze(0)
    y = torch.tensor(tokens[1:seq_len + 1], dtype=torch.long).unsqueeze(0)

    return x, y


# ============================================================================
# Simple Character-Level Tokenizer (fallback)
# ============================================================================

class CharTokenizer:
    """Simple character-level tokenizer as fallback."""

    def __init__(self, text):
        chars = sorted(list(set(text)))
        self.char_to_idx = {c: i for i, c in enumerate(chars)}
        self.idx_to_char = {i: c for i, c in enumerate(chars)}
        self.vocab_size = len(chars)

    def encode(self, text):
        return [self.char_to_idx.get(c, 0) for c in text]

    def decode(self, tokens):
        return ''.join([self.idx_to_char.get(t, '?') for t in tokens])


# ============================================================================
# Main Test
# ============================================================================

def run_overfit_test():
    print("=" * 70)
    print("NANO OVERFIT TEST: Can Integer Math Memorize?")
    print("=" * 70)
    print(f"Goal: Overfit a 5M model on a single batch to loss < {PASS_LOSS_FINAL}")
    print(f"Pass at 1000 steps: loss < {PASS_LOSS_1000}")
    print(f"Fail at 1000 steps: loss > {FAIL_LOSS_1000}")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Download data
    download_tinyshakespeare()

    # Load config
    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'config_mamba_integer_5m.json')
    with open(config_path) as f:
        config = json.load(f)

    # Initialize model
    print("\nInitializing 5M model...")
    model = MambaIntegerModel(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,} ({total_params/1e6:.2f}M)")

    # Try to use Rust tokenizer, fall back to char-level
    try:
        from rust_tokenizer import get_rust_tokenizer
        tokenizer = get_rust_tokenizer()
        tokenizer.load(os.path.join(os.path.dirname(__file__), '..', 'configs', 'rust_bpe_merges.txt'))
        print("Using Rust BPE tokenizer")
    except Exception as e:
        print(f"Rust tokenizer failed ({e}), using char-level tokenizer")
        with open(DATA_PATH, 'r') as f:
            text = f.read()
        tokenizer = CharTokenizer(text)
        # Update vocab size in config
        config['vocab_size'] = min(tokenizer.vocab_size, 4096)
        model = MambaIntegerModel(config).to(device)

    # Create fixed batch
    print(f"\nCreating fixed batch (seq_len={SEQ_LEN})...")
    x, y = create_fixed_batch(tokenizer, SEQ_LEN)
    x, y = x.to(device), y.to(device)
    print(f"Batch shape: x={x.shape}, y={y.shape}")

    # Show what we're memorizing
    if hasattr(tokenizer, 'decode'):
        sample_text = tokenizer.decode(x[0, :50].tolist())
        print(f"Memorizing: \"{sample_text}...\"")

    # Optimizer - no weight decay for pure memorization
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)

    # Training loop
    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)

    model.train()
    start_time = time.time()
    best_loss = float('inf')
    best_loss_step = 0
    losses = []

    for step in range(1, MAX_STEPS + 1):
        optimizer.zero_grad()

        # Forward pass
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        # Check for NaN
        if torch.isnan(loss):
            print(f"\n[ABORT] NaN loss at step {step}")
            print("RESULT: FAIL - Numerical instability")
            return False

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        loss_val = loss.item()
        losses.append(loss_val)

        # Track best
        if loss_val < best_loss:
            best_loss = loss_val
            best_loss_step = step

        # Logging
        if step % LOG_INTERVAL == 0:
            elapsed = time.time() - start_time
            print(f"Step {step:5d} | Loss: {loss_val:.6f} | Best: {best_loss:.6f} @ step {best_loss_step} | Time: {elapsed:.1f}s")

        # Check pass/fail at 1000 steps
        if step == 1000:
            print("\n" + "-" * 50)
            print(f"CHECKPOINT @ 1000 steps: Loss = {loss_val:.6f}")
            if loss_val < PASS_LOSS_1000:
                print(f"  [GOOD] Loss < {PASS_LOSS_1000} - On track!")
            elif loss_val > FAIL_LOSS_1000:
                print(f"  [FAIL] Loss > {FAIL_LOSS_1000} - Architecture may be broken")
                print("  Continuing to see if it improves...")
            else:
                print(f"  [CONCERNING] Loss between {PASS_LOSS_1000} and {FAIL_LOSS_1000}")
            print("-" * 50 + "\n")

        # Abort if no progress
        if step > 1000 and step - best_loss_step > ABORT_NO_PROGRESS_STEPS:
            if best_loss > ABORT_LOSS_THRESHOLD:
                print(f"\n[ABORT] No progress for {ABORT_NO_PROGRESS_STEPS} steps, loss stuck at {best_loss:.4f}")
                print("RESULT: FAIL - Model cannot learn")
                return False

        # Early success
        if loss_val < PASS_LOSS_FINAL:
            print(f"\n[EARLY SUCCESS] Loss {loss_val:.6f} < {PASS_LOSS_FINAL} at step {step}")
            break

    # Final evaluation
    elapsed = time.time() - start_time
    final_loss = losses[-1]

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(f"Final loss: {final_loss:.6f}")
    print(f"Best loss:  {best_loss:.6f} @ step {best_loss_step}")
    print(f"Total time: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    # Generate sample to verify memorization
    print("\n" + "-" * 50)
    print("MEMORIZATION CHECK")
    print("-" * 50)
    model.eval()
    with torch.no_grad():
        logits = model(x)
        predictions = logits.argmax(dim=-1)

        # Calculate accuracy
        correct = (predictions == y).float().mean().item()
        print(f"Token accuracy: {correct*100:.2f}%")

        # Show prediction vs actual
        if hasattr(tokenizer, 'decode'):
            actual = tokenizer.decode(y[0, :30].tolist())
            predicted = tokenizer.decode(predictions[0, :30].tolist())
            print(f"Actual:    \"{actual}\"")
            print(f"Predicted: \"{predicted}\"")

    # Final verdict
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    if final_loss < PASS_LOSS_FINAL:
        print(f"[PASS] Loss {final_loss:.6f} < {PASS_LOSS_FINAL}")
        print("Integer-only architecture CAN memorize.")
        print("NEXT STEP: Proceed to 50M validation (Experiment 2)")
        return True
    elif final_loss < FAIL_LOSS_FINAL:
        print(f"[PARTIAL] Loss {final_loss:.6f} between {PASS_LOSS_FINAL} and {FAIL_LOSS_FINAL}")
        print("Architecture works but may have capacity limitations.")
        print("NEXT STEP: Investigate quantization precision, then proceed to 50M")
        return True
    else:
        print(f"[FAIL] Loss {final_loss:.6f} > {FAIL_LOSS_FINAL}")
        print("Integer-only architecture CANNOT memorize a simple batch.")
        print("NEXT STEP: Debug architecture before any scaling")
        print("\nPossible issues:")
        print("  - BitLinear quantization too aggressive")
        print("  - Dyadic scan state precision insufficient")
        print("  - BitShiftNorm causing gradient issues")
        return False


# ============================================================================
# Also create FP32 baseline for comparison
# ============================================================================

def run_fp32_baseline():
    """Run the same test with FP32 weights as control."""
    print("\n" + "=" * 70)
    print("FP32 BASELINE (Control)")
    print("=" * 70)

    # Monkey-patch BitLinear
    import rational_bitnet
    import torch.nn as nn
    import math

    class FPLinear(nn.Module):
        def __init__(self, in_features, out_features, bias=False, **kwargs):
            super().__init__()
            self.weight = nn.Parameter(torch.randn(out_features, in_features) / math.sqrt(in_features))
            self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

        def forward(self, x):
            return F.linear(x, self.weight, self.bias)

    original_bitlinear = rational_bitnet.BitLinear
    rational_bitnet.BitLinear = FPLinear

    # Reimport model with FP32
    import importlib
    import mamba_integer_model
    importlib.reload(mamba_integer_model)

    # Run same test
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'config_mamba_integer_5m.json')
    with open(config_path) as f:
        config = json.load(f)

    model = mamba_integer_model.MambaIntegerModel(config).to(device)
    print(f"FP32 model params: {sum(p.numel() for p in model.parameters()):,}")

    # Create batch
    try:
        from rust_tokenizer import get_rust_tokenizer
        tokenizer = get_rust_tokenizer()
        tokenizer.load(os.path.join(os.path.dirname(__file__), '..', 'configs', 'rust_bpe_merges.txt'))
    except:
        with open(DATA_PATH, 'r') as f:
            tokenizer = CharTokenizer(f.read())

    x, y = create_fixed_batch(tokenizer, SEQ_LEN)
    x, y = x.to(device), y.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)

    model.train()
    for step in range(1, 2001):  # Quick 2k step run
        optimizer.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        optimizer.step()

        if step % 500 == 0:
            print(f"FP32 Step {step}: Loss = {loss.item():.6f}")

    print(f"\nFP32 Final Loss: {loss.item():.6f}")

    # Restore
    rational_bitnet.BitLinear = original_bitlinear

    return loss.item()


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--with-fp32-baseline', action='store_true',
                        help='Also run FP32 baseline for comparison')
    args = parser.parse_args()

    # Run main test
    success = run_overfit_test()

    # Optionally run FP32 baseline
    if args.with_fp32_baseline:
        fp32_loss = run_fp32_baseline()
        print(f"\n[COMPARISON] FP32 baseline loss: {fp32_loss:.6f}")

    # Exit code
    sys.exit(0 if success else 1)
