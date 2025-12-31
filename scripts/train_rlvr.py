#!/usr/bin/env python3
"""
RLVR Training Script for Mamba-Integer

Train Mamba-Integer with Reinforcement Learning from Verifiable Rewards.
Uses GRPO (Group Relative Policy Optimization) for efficient RL without critic model.

Usage:
    python scripts/train_rlvr.py --checkpoint mamba_integer_step_70000.pt

Prerequisites:
    - Pretrained Mamba-Integer checkpoint (recommend 70k+ steps)
    - CUDA GPU with sufficient memory

Reference:
    - DeepSeek-R1: https://arxiv.org/abs/2501.12948
"""

import argparse
import json
import os
import sys
import torch

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../src'))

from mamba_integer_model import MambaIntegerModel
from rust_tokenizer import get_rust_tokenizer
from rlvr import (
    GRPOTrainer,
    GRPOConfig,
    MathProblemDataset,
    ArithmeticReward,
    CodeExecutionReward,
    MathProblemReward,
    CompositeReward,
    ZKIntegerArithmeticReward,
    ZKDyadicRationalReward,
    ZKCompositeReward
)


def parse_args():
    parser = argparse.ArgumentParser(description="RLVR Training for Mamba-Integer")

    # Model
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to pretrained checkpoint")
    parser.add_argument("--config", type=str, default="configs/config_mamba_integer_l4.json",
                        help="Model config path")

    # Training
    parser.add_argument("--num_samples", type=int, default=8,
                        help="Samples per prompt (G in GRPO)")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Prompts per batch")
    parser.add_argument("--lr", type=float, default=1e-6,
                        help="Learning rate")
    parser.add_argument("--kl_coef", type=float, default=0.04,
                        help="KL penalty coefficient")
    parser.add_argument("--total_steps", type=int, default=5000,
                        help="Total training steps")
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="Max generation length")

    # Reward
    parser.add_argument("--reward_type", type=str, default="zk_arithmetic",
                        choices=["arithmetic", "code", "math", "zk_arithmetic", "zk_composite"],
                        help="Reward function type")

    # Data
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to training data (JSONL)")
    parser.add_argument("--num_synthetic", type=int, default=10000,
                        help="Number of synthetic problems to generate")

    # Output
    parser.add_argument("--output_dir", type=str, default="rlvr_checkpoints",
                        help="Output directory")

    return parser.parse_args()


def load_model(config_path: str, checkpoint_path: str, device: str):
    """Load pretrained Mamba-Integer model."""
    print(f"Loading config from {config_path}")
    with open(config_path, 'r') as f:
        config = json.load(f)

    print(f"Initializing model...")
    model = MambaIntegerModel(config).to(device)

    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle nested format
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
    model.gradient_checkpointing = False

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {num_params:,} parameters")

    return model, config


def create_reward_function(reward_type: str):
    """Create reward function based on type."""
    if reward_type == "arithmetic":
        return ArithmeticReward(tolerance=0.0)

    elif reward_type == "code":
        return CodeExecutionReward(timeout=5.0)

    elif reward_type == "math":
        return MathProblemReward(tolerance=1e-6)

    elif reward_type == "zk_arithmetic":
        # ZK-friendly integer arithmetic
        return ZKIntegerArithmeticReward()

    elif reward_type == "zk_composite":
        # Combine multiple ZK rewards
        return ZKCompositeReward(
            rewards=[
                ZKIntegerArithmeticReward(),
                ZKDyadicRationalReward()
            ],
            mode="all"
        )

    else:
        raise ValueError(f"Unknown reward type: {reward_type}")


def create_dataset(args):
    """Create training dataset."""
    if args.data_path:
        print(f"Loading data from {args.data_path}")
        return MathProblemDataset.from_jsonl(args.data_path)
    else:
        print(f"Generating {args.num_synthetic} synthetic arithmetic problems")
        return MathProblemDataset.generate_arithmetic(
            num_problems=args.num_synthetic,
            max_value=1000,
            operations=["+", "-", "*"]
        )


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model
    model, config = load_model(args.config, args.checkpoint, device)

    # Load tokenizer
    tokenizer = get_rust_tokenizer()
    merges_path = "configs/rust_bpe_merges.txt"
    if os.path.exists(merges_path):
        tokenizer.load(merges_path)
    else:
        print("WARNING: Tokenizer merges not found")

    # Create reward function
    print(f"Using reward type: {args.reward_type}")
    reward_fn = create_reward_function(args.reward_type)

    # Create dataset
    train_dataset = create_dataset(args)
    print(f"Training dataset size: {len(train_dataset)}")

    # Create eval dataset (subset)
    eval_size = min(100, len(train_dataset) // 10)
    eval_problems = [train_dataset[i] for i in range(eval_size)]
    eval_dataset = MathProblemDataset(
        [{"prompt": p["prompt"], "ground_truth": p["ground_truth"]} for p in eval_problems]
    )

    # Create GRPO config
    grpo_config = GRPOConfig(
        num_samples_per_prompt=args.num_samples,
        max_new_tokens=args.max_new_tokens,
        temperature=0.7,
        top_p=0.9,
        learning_rate=args.lr,
        kl_coef=args.kl_coef,
        batch_size=args.batch_size,
        total_steps=args.total_steps,
        log_interval=10,
        save_interval=500,
        eval_interval=100
    )

    print("\n" + "="*60)
    print("GRPO Configuration")
    print("="*60)
    for key, value in grpo_config.__dict__.items():
        print(f"  {key}: {value}")
    print("="*60 + "\n")

    # Create trainer
    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        reward_fn=reward_fn,
        config=grpo_config,
        device=device
    )

    # Start training
    print("Starting RLVR training with GRPO...")
    os.makedirs(args.output_dir, exist_ok=True)

    try:
        trainer.train(
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            output_dir=args.output_dir
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving checkpoint...")
        trainer.save_checkpoint(args.output_dir)

    print("\nTraining complete!")

    # Final evaluation
    print("\nFinal evaluation:")
    final_metrics = trainer.evaluate(eval_dataset)
    for key, value in final_metrics.items():
        print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()
