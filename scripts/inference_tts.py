#!/usr/bin/env python3
"""
Test-Time Scaling Inference Script for Mamba-Integer.

Demonstrates TTS methods on arithmetic problems:
- Self-consistency (majority voting)
- Beam search with implicit PRM
- Adaptive strategy selection

Usage:
    python scripts/inference_tts.py --checkpoint mamba_integer_step_27000.pt
    python scripts/inference_tts.py --checkpoint mamba_integer_step_27000.pt --method beam
    python scripts/inference_tts.py --checkpoint mamba_integer_step_27000.pt --method adaptive
"""

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from mamba_integer_model import MambaIntegerModel
from rust_tokenizer import get_rust_tokenizer
from tts import SelfConsistency, BeamSearchPRM, AdaptiveStrategy


def parse_args():
    parser = argparse.ArgumentParser(description="TTS Inference for Mamba-Integer")

    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default="configs/config_mamba_integer_l4.json",
                        help="Model config path")

    parser.add_argument("--method", type=str, default="all",
                        choices=["greedy", "self_consistency", "beam", "adaptive", "all"],
                        help="TTS method to use")

    # Self-consistency params
    parser.add_argument("--n_samples", type=int, default=8,
                        help="Number of samples for self-consistency")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")

    # Beam search params
    parser.add_argument("--beam_width", type=int, default=4,
                        help="Beam width for beam search")
    parser.add_argument("--max_steps", type=int, default=8,
                        help="Max reasoning steps for beam search")

    parser.add_argument("--max_new_tokens", type=int, default=128,
                        help="Max tokens to generate")

    return parser.parse_args()


def load_model(config_path: str, checkpoint_path: str, device: str):
    """Load model from checkpoint."""
    print(f"Loading config from {config_path}")
    with open(config_path, 'r') as f:
        config = json.load(f)

    print(f"Initializing model...")
    model = MambaIntegerModel(config).to(device)

    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    # Handle torch.compile prefix
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            new_state_dict[k[10:]] = v
        else:
            new_state_dict[k] = v

    model.load_state_dict(new_state_dict, strict=False)
    model.eval()

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {num_params:,} parameters")

    return model


def greedy_generate(model, tokenizer, prompt: str, max_new_tokens: int, device: str) -> str:
    """Simple greedy generation."""
    input_ids = tokenizer.encode(prompt)
    input_tensor = torch.tensor([input_ids], device=device)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                logits = model(input_tensor)

            next_token = logits[0, -1, :].argmax().item()
            input_tensor = torch.cat([
                input_tensor,
                torch.tensor([[next_token]], device=device)
            ], dim=1)

            if next_token == 0:
                break

    output_ids = input_tensor[0].tolist()
    return tokenizer.decode(output_ids[len(input_ids):])


def run_test_problems(model, tokenizer, args, device):
    """Run TTS on test problems."""

    # Test problems (arithmetic)
    problems = [
        ("7 + 8 =", "15"),
        ("23 + 45 =", "68"),
        ("100 - 37 =", "63"),
        ("12 * 5 =", "60"),
        ("144 / 12 =", "12"),
        ("What is 25 + 17? Answer:", "42"),
        ("Calculate: 8 * 9 =", "72"),
        ("If I have 50 apples and eat 13, how many remain? Answer:", "37"),
    ]

    print("\n" + "=" * 70)
    print("TEST-TIME SCALING EVALUATION")
    print("=" * 70)

    # Initialize TTS methods
    sc = SelfConsistency(
        model, tokenizer,
        n_samples=args.n_samples,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens
    )

    beam = BeamSearchPRM(
        model, tokenizer,
        beam_width=args.beam_width,
        max_steps=args.max_steps,
        max_tokens_per_step=args.max_new_tokens // args.max_steps
    )

    adaptive = AdaptiveStrategy(
        model, tokenizer,
        easy_samples=2,
        medium_samples=args.n_samples,
        hard_beam_width=args.beam_width,
        max_new_tokens=args.max_new_tokens
    )

    results = {
        "greedy": {"correct": 0, "total": 0, "time": 0},
        "self_consistency": {"correct": 0, "total": 0, "time": 0},
        "beam_search": {"correct": 0, "total": 0, "time": 0},
        "adaptive": {"correct": 0, "total": 0, "time": 0},
    }

    for prompt, expected in problems:
        print(f"\n{'─' * 70}")
        print(f"Problem: {prompt}")
        print(f"Expected: {expected}")
        print(f"{'─' * 70}")

        # Greedy baseline
        if args.method in ["greedy", "all"]:
            start = time.time()
            greedy_output = greedy_generate(model, tokenizer, prompt, args.max_new_tokens, device)
            elapsed = time.time() - start
            results["greedy"]["time"] += elapsed

            # Check if answer is in output
            correct = expected in greedy_output
            results["greedy"]["total"] += 1
            if correct:
                results["greedy"]["correct"] += 1

            print(f"\n[Greedy] ({elapsed:.2f}s)")
            print(f"  Output: {greedy_output[:100]}...")
            print(f"  Correct: {correct}")

        # Self-consistency
        if args.method in ["self_consistency", "all"]:
            start = time.time()
            sc_result = sc.solve(prompt)
            elapsed = time.time() - start
            results["self_consistency"]["time"] += elapsed

            correct = sc_result.answer == expected
            results["self_consistency"]["total"] += 1
            if correct:
                results["self_consistency"]["correct"] += 1

            print(f"\n[Self-Consistency n={args.n_samples}] ({elapsed:.2f}s)")
            print(f"  Answer: {sc_result.answer}")
            print(f"  Confidence: {sc_result.confidence:.2f}")
            print(f"  Votes: {sc_result.vote_counts}")
            print(f"  Correct: {correct}")

        # Beam search
        if args.method in ["beam", "all"]:
            start = time.time()
            beam_result = beam.search(prompt)
            elapsed = time.time() - start
            results["beam_search"]["time"] += elapsed

            correct = beam_result.answer == expected
            results["beam_search"]["total"] += 1
            if correct:
                results["beam_search"]["correct"] += 1

            print(f"\n[Beam Search width={args.beam_width}] ({elapsed:.2f}s)")
            print(f"  Answer: {beam_result.answer}")
            print(f"  Score: {beam_result.score:.2f}")
            print(f"  Steps: {beam_result.n_steps}")
            print(f"  Correct: {correct}")

        # Adaptive
        if args.method in ["adaptive", "all"]:
            start = time.time()
            adapt_result = adaptive.solve(prompt)
            elapsed = time.time() - start
            results["adaptive"]["time"] += elapsed

            correct = adapt_result.answer == expected
            results["adaptive"]["total"] += 1
            if correct:
                results["adaptive"]["correct"] += 1

            print(f"\n[Adaptive] ({elapsed:.2f}s)")
            print(f"  Difficulty: {adapt_result.difficulty.level.value} ({adapt_result.difficulty.score:.2f})")
            print(f"  Strategy: {adapt_result.strategy_used}")
            print(f"  Answer: {adapt_result.answer}")
            print(f"  Confidence: {adapt_result.confidence:.2f}")
            print(f"  Correct: {correct}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for method, stats in results.items():
        if stats["total"] > 0:
            acc = stats["correct"] / stats["total"] * 100
            avg_time = stats["time"] / stats["total"]
            print(f"{method:20s}: {stats['correct']}/{stats['total']} ({acc:.1f}%) | Avg time: {avg_time:.2f}s")


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model
    model = load_model(args.config, args.checkpoint, device)

    # Load tokenizer
    tokenizer = get_rust_tokenizer()
    merges_path = "configs/rust_bpe_merges.txt"
    if os.path.exists(merges_path):
        tokenizer.load(merges_path)
    else:
        print("WARNING: Tokenizer merges not found")

    # Run tests
    run_test_problems(model, tokenizer, args, device)


if __name__ == "__main__":
    main()
