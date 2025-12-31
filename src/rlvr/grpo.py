"""
GRPO: Group Relative Policy Optimization

Implementation based on DeepSeek-R1 and DeepSeekMath papers.
Optimized for Mamba-Integer's integer-only architecture.

Key Features:
1. No critic model needed (reduces memory by ~50%)
2. Group-relative reward normalization
3. KL divergence regularization to prevent drift
4. Compatible with verifiable rewards

References:
- DeepSeek-R1: arXiv:2501.12948
- DeepSeekMath: arXiv:2402.03300
- GRPO Guide: https://cameronrwolfe.substack.com/p/grpo
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm
import json
import os

from .rewards import VerifiableReward, RewardResult


@dataclass
class GRPOConfig:
    """Configuration for GRPO training."""

    # Sampling
    num_samples_per_prompt: int = 8       # G in the paper (DeepSeek uses 64)
    max_new_tokens: int = 512             # Maximum generation length
    temperature: float = 0.7              # Sampling temperature
    top_p: float = 0.9                    # Nucleus sampling

    # Training
    learning_rate: float = 1e-6           # Policy LR (DeepSeek: 1e-6)
    kl_coef: float = 0.04                 # KL penalty coefficient (DeepSeek: 0.04)
    clip_range: float = 0.2               # PPO-style clipping (optional)
    batch_size: int = 8                   # Prompts per batch
    gradient_accumulation_steps: int = 4

    # Optimization
    max_grad_norm: float = 1.0
    warmup_steps: int = 100
    total_steps: int = 10000

    # Logging
    log_interval: int = 10
    save_interval: int = 500
    eval_interval: int = 100


@dataclass
class GRPOBatch:
    """A batch for GRPO training."""
    prompts: List[str]
    responses: List[List[str]]          # [batch_size, num_samples]
    rewards: torch.Tensor               # [batch_size, num_samples]
    log_probs: torch.Tensor             # [batch_size, num_samples]
    ref_log_probs: torch.Tensor         # [batch_size, num_samples]
    advantages: torch.Tensor            # [batch_size, num_samples] normalized


class GRPOTrainer:
    """
    GRPO Trainer for Mamba-Integer.

    Algorithm Overview:
    1. For each prompt, sample G responses from current policy
    2. Compute verifiable rewards for each response
    3. Normalize rewards within each group (prompt)
    4. Compute policy gradient with KL regularization
    5. Update policy
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        reward_fn: VerifiableReward,
        config: GRPOConfig,
        ref_model: Optional[nn.Module] = None,
        device: str = "cuda"
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.reward_fn = reward_fn
        self.config = config
        self.device = device

        # Reference model for KL divergence (frozen copy of initial policy)
        if ref_model is not None:
            self.ref_model = ref_model
        else:
            # Create frozen copy
            self.ref_model = self._create_ref_model()

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01
        )

        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.total_steps,
            eta_min=config.learning_rate * 0.1
        )

        # Training state
        self.global_step = 0
        self.total_samples = 0
        self.metrics = {
            "loss": [],
            "reward_mean": [],
            "reward_std": [],
            "kl_div": [],
            "grad_norm": []
        }

    def _create_ref_model(self) -> nn.Module:
        """Create frozen reference model."""
        import copy
        ref = copy.deepcopy(self.model)
        ref.eval()
        for param in ref.parameters():
            param.requires_grad = False
        return ref

    @torch.no_grad()
    def generate_samples(
        self,
        prompt: str,
        num_samples: int
    ) -> Tuple[List[str], torch.Tensor]:
        """
        Generate multiple samples for a prompt.

        Returns:
            responses: List of generated text responses
            log_probs: Log probabilities of each response [num_samples]
        """
        self.model.eval()

        # Encode prompt
        input_ids = torch.tensor([self.tokenizer.encode(prompt)]).to(self.device)
        prompt_len = input_ids.shape[1]

        responses = []
        all_log_probs = []

        for _ in range(num_samples):
            generated = input_ids.clone()
            total_log_prob = 0.0

            for _ in range(self.config.max_new_tokens):
                logits = self.model(generated)
                next_logits = logits[:, -1, :]

                # Apply temperature
                next_logits = next_logits / self.config.temperature

                # Compute probabilities
                probs = F.softmax(next_logits, dim=-1)

                # Top-p sampling
                if self.config.top_p < 1.0:
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                    cumsum = torch.cumsum(sorted_probs, dim=-1)
                    mask = cumsum > self.config.top_p
                    mask[:, 0] = False  # Keep at least one token
                    sorted_probs[mask] = 0
                    probs = torch.zeros_like(probs).scatter_(1, sorted_indices, sorted_probs)
                    probs = probs / probs.sum(dim=-1, keepdim=True)

                # Sample
                next_token = torch.multinomial(probs, num_samples=1)

                # Accumulate log prob
                token_log_prob = torch.log(probs[0, next_token[0, 0]] + 1e-10)
                total_log_prob += token_log_prob.item()

                generated = torch.cat([generated, next_token], dim=-1)

                # Stop at EOS (assuming token 0 or check vocab)
                if next_token[0, 0].item() == 0:  # EOS token
                    break

            # Decode response (excluding prompt)
            response_ids = generated[0, prompt_len:].tolist()
            response_text = self.tokenizer.decode(response_ids)
            responses.append(response_text)
            all_log_probs.append(total_log_prob)

        self.model.train()
        return responses, torch.tensor(all_log_probs, device=self.device)

    @torch.no_grad()
    def compute_ref_log_probs(
        self,
        prompt: str,
        responses: List[str]
    ) -> torch.Tensor:
        """Compute log probabilities under reference model."""
        self.ref_model.eval()

        ref_log_probs = []

        for response in responses:
            full_text = prompt + response
            input_ids = torch.tensor([self.tokenizer.encode(full_text)]).to(self.device)
            prompt_ids = torch.tensor([self.tokenizer.encode(prompt)]).to(self.device)
            prompt_len = prompt_ids.shape[1]

            if input_ids.shape[1] <= prompt_len:
                ref_log_probs.append(0.0)
                continue

            logits = self.ref_model(input_ids)

            # Compute log probs for response tokens only
            response_logits = logits[:, prompt_len-1:-1, :]  # Shift by 1
            response_targets = input_ids[:, prompt_len:]

            log_probs = F.log_softmax(response_logits, dim=-1)
            token_log_probs = log_probs.gather(2, response_targets.unsqueeze(-1)).squeeze(-1)
            total_log_prob = token_log_probs.sum().item()

            ref_log_probs.append(total_log_prob)

        return torch.tensor(ref_log_probs, device=self.device)

    def compute_rewards(
        self,
        prompt: str,
        responses: List[str],
        ground_truth: Optional[str] = None
    ) -> Tuple[torch.Tensor, List[RewardResult]]:
        """Compute verifiable rewards for responses."""
        rewards = []
        results = []

        for response in responses:
            result = self.reward_fn.compute(prompt, response, ground_truth)
            rewards.append(result.reward)
            results.append(result)

        return torch.tensor(rewards, device=self.device), results

    def compute_advantages(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        Compute group-relative advantages.

        GRPO normalizes rewards within each group (prompt) to get advantages.
        This is the key innovation: no critic model needed.
        """
        # Normalize within group
        mean = rewards.mean()
        std = rewards.std() + 1e-8

        advantages = (rewards - mean) / std
        return advantages

    def compute_loss(
        self,
        prompt: str,
        responses: List[str],
        advantages: torch.Tensor,
        old_log_probs: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute GRPO loss.

        Loss = -E[advantage * log_prob] + kl_coef * KL(policy || ref_policy)
        """
        self.model.train()

        total_policy_loss = 0.0
        total_kl = 0.0

        for i, response in enumerate(responses):
            full_text = prompt + response
            input_ids = torch.tensor([self.tokenizer.encode(full_text)]).to(self.device)
            prompt_ids = torch.tensor([self.tokenizer.encode(prompt)]).to(self.device)
            prompt_len = prompt_ids.shape[1]

            if input_ids.shape[1] <= prompt_len:
                continue

            # Forward pass
            logits = self.model(input_ids)

            # Compute current log probs
            response_logits = logits[:, prompt_len-1:-1, :]
            response_targets = input_ids[:, prompt_len:]

            log_probs = F.log_softmax(response_logits, dim=-1)
            token_log_probs = log_probs.gather(2, response_targets.unsqueeze(-1)).squeeze(-1)
            current_log_prob = token_log_probs.sum()

            # Policy gradient loss (with advantage weighting)
            advantage = advantages[i]

            # Optional: PPO-style clipping
            if self.config.clip_range > 0:
                ratio = torch.exp(current_log_prob - old_log_probs[i])
                clipped_ratio = torch.clamp(ratio, 1 - self.config.clip_range, 1 + self.config.clip_range)
                policy_loss = -torch.min(ratio * advantage, clipped_ratio * advantage)
            else:
                policy_loss = -current_log_prob * advantage

            total_policy_loss += policy_loss

            # KL divergence (approximate)
            with torch.no_grad():
                ref_logits = self.ref_model(input_ids)
                ref_response_logits = ref_logits[:, prompt_len-1:-1, :]
                ref_log_probs = F.log_softmax(ref_response_logits, dim=-1)
                ref_token_log_probs = ref_log_probs.gather(2, response_targets.unsqueeze(-1)).squeeze(-1)
                ref_log_prob = ref_token_log_probs.sum()

            kl = current_log_prob - ref_log_prob
            total_kl += kl

        # Average over samples
        num_samples = len(responses)
        avg_policy_loss = total_policy_loss / num_samples
        avg_kl = total_kl / num_samples

        # Total loss with KL penalty
        loss = avg_policy_loss + self.config.kl_coef * avg_kl

        metrics = {
            "policy_loss": avg_policy_loss.item(),
            "kl_div": avg_kl.item(),
            "total_loss": loss.item()
        }

        return loss, metrics

    def train_step(
        self,
        prompts: List[str],
        ground_truths: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """
        Single GRPO training step.

        For each prompt:
        1. Generate G samples
        2. Compute rewards
        3. Normalize to advantages
        4. Compute loss and update
        """
        self.model.train()
        self.optimizer.zero_grad()

        all_metrics = {
            "loss": 0.0,
            "reward_mean": 0.0,
            "reward_std": 0.0,
            "kl_div": 0.0,
            "correct_ratio": 0.0
        }

        total_loss = 0.0

        for i, prompt in enumerate(prompts):
            gt = ground_truths[i] if ground_truths else None

            # 1. Generate samples
            responses, log_probs = self.generate_samples(
                prompt,
                self.config.num_samples_per_prompt
            )

            # 2. Compute rewards
            rewards, results = self.compute_rewards(prompt, responses, gt)

            # 3. Compute advantages (group-relative normalization)
            advantages = self.compute_advantages(rewards)

            # 4. Compute loss
            loss, metrics = self.compute_loss(prompt, responses, advantages, log_probs)

            total_loss += loss

            # Accumulate metrics
            all_metrics["reward_mean"] += rewards.mean().item()
            all_metrics["reward_std"] += rewards.std().item()
            all_metrics["kl_div"] += metrics["kl_div"]
            all_metrics["correct_ratio"] += sum(1 for r in results if r.correct) / len(results)

        # Average metrics
        batch_size = len(prompts)
        for key in all_metrics:
            all_metrics[key] /= batch_size

        # Backward and update
        total_loss = total_loss / batch_size
        total_loss.backward()

        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.max_grad_norm
        )

        self.optimizer.step()
        self.scheduler.step()

        all_metrics["loss"] = total_loss.item()
        all_metrics["grad_norm"] = grad_norm.item()
        all_metrics["lr"] = self.scheduler.get_last_lr()[0]

        self.global_step += 1
        self.total_samples += batch_size * self.config.num_samples_per_prompt

        return all_metrics

    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        output_dir: str = "grpo_checkpoints"
    ):
        """
        Full GRPO training loop.

        Args:
            train_dataset: Dataset yielding (prompt, ground_truth) pairs
            eval_dataset: Optional evaluation dataset
            output_dir: Where to save checkpoints
        """
        os.makedirs(output_dir, exist_ok=True)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True
        )

        pbar = tqdm(total=self.config.total_steps, desc="GRPO Training")

        for epoch in range(1000):  # Iterate until total_steps reached
            for batch in train_loader:
                if self.global_step >= self.config.total_steps:
                    break

                # Extract prompts and ground truths
                if isinstance(batch, dict):
                    prompts = batch["prompt"]
                    ground_truths = batch.get("ground_truth", None)
                else:
                    prompts = batch[0]
                    ground_truths = batch[1] if len(batch) > 1 else None

                # Training step
                metrics = self.train_step(prompts, ground_truths)

                # Logging
                if self.global_step % self.config.log_interval == 0:
                    self._log_metrics(metrics)
                    pbar.set_postfix({
                        "loss": f"{metrics['loss']:.4f}",
                        "reward": f"{metrics['reward_mean']:.3f}",
                        "correct": f"{metrics['correct_ratio']:.2%}"
                    })

                # Evaluation
                if eval_dataset and self.global_step % self.config.eval_interval == 0:
                    eval_metrics = self.evaluate(eval_dataset)
                    print(f"\nEval @ step {self.global_step}: {eval_metrics}")

                # Checkpointing
                if self.global_step % self.config.save_interval == 0:
                    self.save_checkpoint(output_dir)

                pbar.update(1)

            if self.global_step >= self.config.total_steps:
                break

        pbar.close()
        self.save_checkpoint(output_dir, final=True)

    @torch.no_grad()
    def evaluate(self, eval_dataset: Dataset) -> Dict[str, float]:
        """Evaluate model on held-out data."""
        self.model.eval()

        total_reward = 0.0
        total_correct = 0
        total_samples = 0

        eval_loader = DataLoader(eval_dataset, batch_size=self.config.batch_size)

        for batch in eval_loader:
            if isinstance(batch, dict):
                prompts = batch["prompt"]
                ground_truths = batch.get("ground_truth", None)
            else:
                prompts = batch[0]
                ground_truths = batch[1] if len(batch) > 1 else None

            for i, prompt in enumerate(prompts):
                gt = ground_truths[i] if ground_truths else None

                # Generate single sample (greedy)
                responses, _ = self.generate_samples(prompt, 1)
                rewards, results = self.compute_rewards(prompt, responses, gt)

                total_reward += rewards[0].item()
                total_correct += 1 if results[0].correct else 0
                total_samples += 1

        self.model.train()

        return {
            "eval_reward_mean": total_reward / total_samples,
            "eval_accuracy": total_correct / total_samples
        }

    def _log_metrics(self, metrics: Dict[str, float]):
        """Store metrics for logging."""
        for key, value in metrics.items():
            if key not in self.metrics:
                self.metrics[key] = []
            self.metrics[key].append(value)

    def save_checkpoint(self, output_dir: str, final: bool = False):
        """Save training checkpoint."""
        suffix = "final" if final else f"step_{self.global_step}"
        path = os.path.join(output_dir, f"grpo_{suffix}.pt")

        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "total_samples": self.total_samples,
            "metrics": self.metrics,
            "config": self.config.__dict__
        }, path)

        print(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint["global_step"]
        self.total_samples = checkpoint["total_samples"]
        self.metrics = checkpoint["metrics"]

        print(f"Loaded checkpoint from step {self.global_step}")


# =============================================================================
# MATH PROBLEM DATASET
# =============================================================================

class MathProblemDataset(Dataset):
    """
    Dataset of math problems for RLVR training.

    Format: {"prompt": "What is 2 + 2?", "ground_truth": "4"}
    """

    def __init__(self, problems: List[Dict[str, str]]):
        self.problems = problems

    def __len__(self):
        return len(self.problems)

    def __getitem__(self, idx):
        p = self.problems[idx]
        return {
            "prompt": p["prompt"],
            "ground_truth": p.get("ground_truth", p.get("answer", None))
        }

    @classmethod
    def from_jsonl(cls, path: str) -> "MathProblemDataset":
        """Load from JSONL file."""
        problems = []
        with open(path, 'r') as f:
            for line in f:
                problems.append(json.loads(line))
        return cls(problems)

    @classmethod
    def generate_arithmetic(
        cls,
        num_problems: int = 1000,
        max_value: int = 1000,
        operations: List[str] = None
    ) -> "MathProblemDataset":
        """Generate synthetic arithmetic problems."""
        import random

        if operations is None:
            operations = ["+", "-", "*"]

        problems = []

        for _ in range(num_problems):
            a = random.randint(1, max_value)
            b = random.randint(1, max_value)
            op = random.choice(operations)

            if op == "+":
                answer = a + b
            elif op == "-":
                answer = a - b
            elif op == "*":
                answer = a * b
            elif op == "//":
                b = random.randint(1, 100)  # Smaller divisor
                answer = a // b
            elif op == "%":
                b = random.randint(1, 100)
                answer = a % b
            else:
                continue

            problems.append({
                "prompt": f"What is {a} {op} {b}?",
                "ground_truth": str(answer)
            })

        return cls(problems)
