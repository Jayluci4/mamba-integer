"""
IV-GRPO: Integer-Verified Group Relative Policy Optimization

Novel contribution from the post-training playbook:
  R = R_correctness + α * exp(-β * KL(P_float || P_integer))

Extends standard GRPO with a dual-path forward pass that measures
distributional divergence between the float training path and
integer-only inference path. Low KL → high reward → trains toward
parameter regions where quantization causes minimal shift.

No existing work trains for quantization robustness through RL rewards.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
import math
import json
import os

from .rewards import VerifiableReward, RewardResult
from .grpo import GRPOConfig, GRPOTrainer
from .domain_verifiers import IVGRPOReward, create_verifier


@dataclass
class IVGRPOConfig(GRPOConfig):
    """Configuration for Integer-Verified GRPO."""

    # IV-GRPO specific
    iv_alpha: float = 0.3           # Weight of integer-consistency reward
    iv_beta: float = 1.0            # KL sensitivity (higher = more penalty)
    iv_max_kl: float = 10.0         # Clamp KL for numerical stability
    iv_temperature: float = 1.0     # Temperature for KL computation
    iv_warmup_steps: int = 50       # Steps before enabling IV reward
    iv_anneal: bool = True          # Anneal alpha up from 0 during warmup
    iv_eval_only: bool = False      # Only log KL, don't use in reward

    # Integer model
    int_quantize_fn: Optional[str] = None  # Name of quantization function


class IVGRPOTrainer(GRPOTrainer):
    """
    Integer-Verified GRPO Trainer.

    Extends GRPOTrainer with dual-path KL divergence measurement between
    the float training path and integer-only inference path.

    The integer model is derived from the current model by applying
    quantization (BitNet 1.58b ternary). During training:
    1. Generate responses using the float model (standard GRPO)
    2. Compute R_correctness using domain verifiers
    3. Forward both float and integer paths on the same sequences
    4. Compute KL(P_float || P_integer)
    5. R_total = R_correctness + α * exp(-β * KL)
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        reward_fn: VerifiableReward,
        config: IVGRPOConfig,
        ref_model: Optional[nn.Module] = None,
        quantize_fn: Optional[Callable] = None,
        device: str = "cuda",
    ):
        """
        Args:
            model: The policy model (float path for training)
            tokenizer: Tokenizer instance
            reward_fn: Domain-specific verifier for R_correctness
            config: IVGRPOConfig with both GRPO and IV params
            ref_model: Frozen reference model (for standard GRPO KL penalty)
            quantize_fn: Function(model) -> quantized model for integer path.
                         If None, uses _default_quantize which applies
                         BitNet 1.58b ternary quantization.
            device: Device to run on
        """
        super().__init__(model, tokenizer, reward_fn, config, ref_model, device)

        self.iv_config = config
        self.quantize_fn = quantize_fn or self._default_quantize

        # Additional metrics tracking
        self.metrics["kl_float_int"] = []
        self.metrics["r_consistency"] = []
        self.metrics["iv_alpha_effective"] = []

    @staticmethod
    def _default_quantize(model: nn.Module) -> nn.Module:
        """
        Default quantization: simulate BitNet 1.58b ternary.

        For each linear layer weight W:
            W_ternary = sign(W) * round(|W| / scale)
        where scale = mean(|W|) and values are clamped to {-1, 0, 1}.

        This is a detached operation — no gradients flow through quantization.
        """
        import copy
        int_model = copy.deepcopy(model)
        int_model.eval()

        with torch.no_grad():
            for name, param in int_model.named_parameters():
                if param.dim() >= 2:  # Weight matrices only
                    scale = param.abs().mean()
                    if scale > 0:
                        # Ternary quantization: {-1, 0, 1}
                        quantized = torch.sign(param) * torch.clamp(
                            torch.round(param.abs() / (scale + 1e-8)),
                            max=1.0,
                        )
                        param.copy_(quantized * scale)
                param.requires_grad = False

        return int_model

    def _get_effective_alpha(self) -> float:
        """Get current IV alpha, with optional warmup annealing."""
        if self.iv_config.iv_eval_only:
            return 0.0

        alpha = self.iv_config.iv_alpha

        if self.iv_config.iv_anneal and self.global_step < self.iv_config.iv_warmup_steps:
            # Linear ramp from 0 to alpha over warmup steps
            progress = self.global_step / max(self.iv_config.iv_warmup_steps, 1)
            alpha = alpha * progress

        return alpha

    @torch.no_grad()
    def compute_float_int_kl(
        self,
        prompt: str,
        responses: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute KL(P_float || P_integer) for each response.

        Runs both the current float model and a freshly-quantized integer
        model on the same prompt+response sequences, then measures the
        distributional divergence.

        Returns:
            kl_values: [num_responses] KL divergence per response
            r_consistency: [num_responses] consistency reward per response
        """
        # Create integer model from current weights
        int_model = self.quantize_fn(self.model)
        int_model = int_model.to(self.device)

        self.model.eval()

        kl_values = []
        r_consistency_values = []

        for response in responses:
            full_text = prompt + response
            input_ids = torch.tensor(
                [self.tokenizer.encode(full_text)]
            ).to(self.device)
            prompt_ids = torch.tensor(
                [self.tokenizer.encode(prompt)]
            ).to(self.device)
            prompt_len = prompt_ids.shape[1]

            if input_ids.shape[1] <= prompt_len:
                kl_values.append(0.0)
                r_consistency_values.append(1.0)
                continue

            # Float path logits
            float_logits = self.model(input_ids)
            float_response_logits = float_logits[:, prompt_len - 1:-1, :]

            # Integer path logits
            int_logits = int_model(input_ids)
            int_response_logits = int_logits[:, prompt_len - 1:-1, :]

            # Apply temperature
            t = self.iv_config.iv_temperature
            float_log_probs = F.log_softmax(float_response_logits / t, dim=-1)
            int_log_probs = F.log_softmax(int_response_logits / t, dim=-1)

            # KL(P_float || P_integer)
            float_probs = float_log_probs.exp()
            kl = (float_probs * (float_log_probs - int_log_probs)).sum(dim=-1)
            kl_mean = kl.mean().item()

            # Clamp
            kl_mean = max(0.0, min(kl_mean, self.iv_config.iv_max_kl))

            # Consistency reward: exp(-β * KL)
            r_cons = math.exp(-self.iv_config.iv_beta * kl_mean)

            kl_values.append(kl_mean)
            r_consistency_values.append(r_cons)

        # Clean up integer model
        del int_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.model.train()

        return (
            torch.tensor(kl_values, device=self.device),
            torch.tensor(r_consistency_values, device=self.device),
        )

    def compute_iv_rewards(
        self,
        prompt: str,
        responses: List[str],
        ground_truth: Optional[str] = None,
    ) -> Tuple[torch.Tensor, List[RewardResult], Dict[str, float]]:
        """
        Compute combined IV-GRPO rewards.

        R_total = R_correctness + α * R_consistency
        Normalized to [0, 1] by dividing by (1 + α).

        Returns:
            rewards: [num_responses] combined rewards
            results: List of RewardResult from base verifier
            iv_metrics: Dict with kl_mean, r_consistency_mean, alpha_effective
        """
        # Base correctness rewards
        base_rewards, results = self.compute_rewards(prompt, responses, ground_truth)

        # Float-vs-integer KL
        kl_values, r_consistency = self.compute_float_int_kl(prompt, responses)

        # Effective alpha (with warmup)
        alpha = self._get_effective_alpha()

        # Combined reward: R = R_correctness + α * R_consistency
        combined = base_rewards + alpha * r_consistency

        # Normalize to [0, 1]
        if alpha > 0:
            combined = combined / (1.0 + alpha)

        iv_metrics = {
            "kl_float_int": kl_values.mean().item(),
            "r_consistency": r_consistency.mean().item(),
            "iv_alpha_effective": alpha,
        }

        return combined, results, iv_metrics

    def train_step(
        self,
        prompts: List[str],
        ground_truths: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        IV-GRPO training step.

        Extends standard GRPO with integer-consistency rewards.
        """
        self.model.train()
        self.optimizer.zero_grad()

        all_metrics = {
            "loss": 0.0,
            "reward_mean": 0.0,
            "reward_std": 0.0,
            "kl_div": 0.0,
            "correct_ratio": 0.0,
            "kl_float_int": 0.0,
            "r_consistency": 0.0,
            "iv_alpha_effective": 0.0,
        }

        total_loss = 0.0

        for i, prompt in enumerate(prompts):
            gt = ground_truths[i] if ground_truths else None

            # 1. Generate samples (standard GRPO)
            responses, log_probs = self.generate_samples(
                prompt, self.config.num_samples_per_prompt
            )

            # 2. Compute IV-GRPO rewards (correctness + consistency)
            rewards, results, iv_metrics = self.compute_iv_rewards(
                prompt, responses, gt
            )

            # 3. Group-relative advantage normalization
            advantages = self.compute_advantages(rewards)

            # 4. Compute loss (standard GRPO policy gradient + KL penalty)
            loss, loss_metrics = self.compute_loss(
                prompt, responses, advantages, log_probs
            )

            total_loss += loss

            # Accumulate metrics
            all_metrics["reward_mean"] += rewards.mean().item()
            all_metrics["reward_std"] += rewards.std().item()
            all_metrics["kl_div"] += loss_metrics["kl_div"]
            all_metrics["correct_ratio"] += (
                sum(1 for r in results if r.correct) / len(results)
            )
            all_metrics["kl_float_int"] += iv_metrics["kl_float_int"]
            all_metrics["r_consistency"] += iv_metrics["r_consistency"]
            all_metrics["iv_alpha_effective"] = iv_metrics["iv_alpha_effective"]

        # Average metrics
        batch_size = len(prompts)
        for key in all_metrics:
            if key != "iv_alpha_effective":
                all_metrics[key] /= batch_size

        # Backward and update
        total_loss = total_loss / batch_size
        total_loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.max_grad_norm
        )

        self.optimizer.step()
        self.scheduler.step()

        all_metrics["loss"] = total_loss.item()
        all_metrics["grad_norm"] = grad_norm.item()
        all_metrics["lr"] = self.scheduler.get_last_lr()[0]

        self.global_step += 1
        self.total_samples += batch_size * self.config.num_samples_per_prompt

        return all_metrics

    def _log_metrics(self, metrics: Dict[str, float]):
        """Store metrics including IV-GRPO specific ones."""
        super()._log_metrics(metrics)


# =============================================================================
# MULTI-DOMAIN DATASET
# =============================================================================

class DomainDataset(Dataset):
    """
    Multi-domain dataset for GRPO/IV-GRPO training.

    Each example has:
    - prompt: str
    - ground_truth: Optional[str]
    - domain: str (for selecting the appropriate verifier)

    Can load from JSONL or from multiple domain-specific sources.
    """

    def __init__(self, examples: List[Dict[str, Any]]):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        return {
            "prompt": ex["prompt"],
            "ground_truth": ex.get("ground_truth", None),
            "domain": ex.get("domain", "math"),
        }

    @classmethod
    def from_jsonl(cls, path: str) -> "DomainDataset":
        """Load from JSONL file."""
        examples = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    examples.append(json.loads(line))
        return cls(examples)

    @classmethod
    def from_multiple_jsonl(cls, paths: Dict[str, str]) -> "DomainDataset":
        """Load from multiple domain-specific JSONL files.

        Args:
            paths: Dict mapping domain name to JSONL file path
        """
        examples = []
        for domain, path in paths.items():
            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        ex = json.loads(line)
                        ex["domain"] = domain
                        examples.append(ex)
        return cls(examples)

    @classmethod
    def generate_mixed(
        cls,
        num_per_domain: int = 200,
        domains: Optional[List[str]] = None,
    ) -> "DomainDataset":
        """Generate synthetic mixed-domain training data."""
        import random

        if domains is None:
            domains = ["math", "sql"]

        examples = []

        if "math" in domains:
            ops = ["+", "-", "*"]
            for _ in range(num_per_domain):
                a = random.randint(1, 1000)
                b = random.randint(1, 1000)
                op = random.choice(ops)
                if op == "+":
                    ans = a + b
                elif op == "-":
                    ans = a - b
                else:
                    ans = a * b
                examples.append({
                    "prompt": f"What is {a} {op} {b}?",
                    "ground_truth": str(ans),
                    "domain": "math",
                })

        if "sql" in domains:
            templates = [
                (
                    "Write a SQL query to select all users older than {age}.",
                    "SELECT * FROM users WHERE age > {age};",
                ),
                (
                    "Write a SQL query to count employees in each department.",
                    "SELECT department, COUNT(*) FROM employees GROUP BY department;",
                ),
                (
                    "Write a SQL query to find the maximum salary.",
                    "SELECT MAX(salary) FROM employees;",
                ),
            ]
            for _ in range(num_per_domain):
                tmpl = random.choice(templates)
                age = random.randint(18, 65)
                prompt = tmpl[0].format(age=age)
                gt = tmpl[1].format(age=age)
                examples.append({
                    "prompt": prompt,
                    "ground_truth": gt,
                    "domain": "sql",
                })

        random.shuffle(examples)
        return cls(examples)


# =============================================================================
# MULTI-DOMAIN IV-GRPO TRAINER
# =============================================================================

class MultiDomainIVGRPOTrainer(IVGRPOTrainer):
    """
    IV-GRPO trainer with automatic domain verifier selection.

    Uses the domain field in each example to select the appropriate
    verifier from the registry. This is the main entry point for
    training across Solidity, SQL, Math, SEC/Finance, and English.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        config: IVGRPOConfig,
        verifier_configs: Optional[Dict[str, Dict]] = None,
        ref_model: Optional[nn.Module] = None,
        quantize_fn: Optional[Callable] = None,
        device: str = "cuda",
    ):
        """
        Args:
            model: Policy model
            tokenizer: Tokenizer
            config: IVGRPOConfig
            verifier_configs: Optional per-domain verifier kwargs.
                e.g. {"solidity": {"required_functions": ["transfer"]}}
            ref_model: Frozen reference model
            quantize_fn: Quantization function for integer path
            device: Device
        """
        # Use a math verifier as default (will be overridden per-domain)
        default_reward = create_verifier("math")
        super().__init__(
            model, tokenizer, default_reward, config,
            ref_model, quantize_fn, device,
        )

        self.verifier_configs = verifier_configs or {}
        self._verifier_cache: Dict[str, VerifiableReward] = {}

    def _get_verifier(self, domain: str) -> VerifiableReward:
        """Get or create a cached verifier for the domain."""
        if domain not in self._verifier_cache:
            kwargs = self.verifier_configs.get(domain, {})
            self._verifier_cache[domain] = create_verifier(domain, **kwargs)
        return self._verifier_cache[domain]

    def compute_rewards_for_domain(
        self,
        prompt: str,
        responses: List[str],
        domain: str,
        ground_truth: Optional[str] = None,
    ) -> Tuple[torch.Tensor, List[RewardResult]]:
        """Compute rewards using domain-specific verifier."""
        verifier = self._get_verifier(domain)
        rewards = []
        results = []
        for response in responses:
            result = verifier.compute(prompt, response, ground_truth)
            rewards.append(result.reward)
            results.append(result)
        return torch.tensor(rewards, device=self.device), results

    def train_step(
        self,
        prompts: List[str],
        ground_truths: Optional[List[str]] = None,
        domains: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """
        Multi-domain IV-GRPO training step.

        Each prompt can have a different domain verifier.
        """
        self.model.train()
        self.optimizer.zero_grad()

        if domains is None:
            domains = ["math"] * len(prompts)

        all_metrics = {
            "loss": 0.0,
            "reward_mean": 0.0,
            "reward_std": 0.0,
            "kl_div": 0.0,
            "correct_ratio": 0.0,
            "kl_float_int": 0.0,
            "r_consistency": 0.0,
            "iv_alpha_effective": 0.0,
        }

        total_loss = 0.0

        for i, prompt in enumerate(prompts):
            gt = ground_truths[i] if ground_truths else None
            domain = domains[i]

            # 1. Generate samples
            responses, log_probs = self.generate_samples(
                prompt, self.config.num_samples_per_prompt
            )

            # 2. Domain-specific correctness rewards
            base_rewards, results = self.compute_rewards_for_domain(
                prompt, responses, domain, gt
            )

            # 3. Float-vs-integer KL consistency
            kl_values, r_consistency = self.compute_float_int_kl(
                prompt, responses
            )

            # 4. Combine rewards
            alpha = self._get_effective_alpha()
            combined = base_rewards + alpha * r_consistency
            if alpha > 0:
                combined = combined / (1.0 + alpha)

            # 5. Advantages
            advantages = self.compute_advantages(combined)

            # 6. Loss
            loss, loss_metrics = self.compute_loss(
                prompt, responses, advantages, log_probs
            )
            total_loss += loss

            # Metrics
            all_metrics["reward_mean"] += combined.mean().item()
            all_metrics["reward_std"] += combined.std().item()
            all_metrics["kl_div"] += loss_metrics["kl_div"]
            all_metrics["correct_ratio"] += (
                sum(1 for r in results if r.correct) / len(results)
            )
            all_metrics["kl_float_int"] += kl_values.mean().item()
            all_metrics["r_consistency"] += r_consistency.mean().item()
            all_metrics["iv_alpha_effective"] = alpha

        # Average
        batch_size = len(prompts)
        for key in all_metrics:
            if key != "iv_alpha_effective":
                all_metrics[key] /= batch_size

        # Backward
        total_loss = total_loss / batch_size
        total_loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.max_grad_norm
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
        output_dir: str = "iv_grpo_checkpoints",
    ):
        """
        Full multi-domain IV-GRPO training loop.

        Dataset items must have 'prompt', optional 'ground_truth' and 'domain'.
        """
        from torch.utils.data import DataLoader
        from tqdm import tqdm

        os.makedirs(output_dir, exist_ok=True)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
        )

        pbar = tqdm(total=self.config.total_steps, desc="IV-GRPO Training")

        for epoch in range(1000):
            for batch in train_loader:
                if self.global_step >= self.config.total_steps:
                    break

                if isinstance(batch, dict):
                    prompts = batch["prompt"]
                    ground_truths = batch.get("ground_truth", None)
                    domains = batch.get("domain", None)
                    # Convert list of None to None
                    if ground_truths is not None and all(
                        g is None for g in ground_truths
                    ):
                        ground_truths = None
                    if domains is not None:
                        domains = list(domains)
                else:
                    prompts = batch[0]
                    ground_truths = batch[1] if len(batch) > 1 else None
                    domains = batch[2] if len(batch) > 2 else None

                metrics = self.train_step(
                    list(prompts),
                    list(ground_truths) if ground_truths else None,
                    domains,
                )

                if self.global_step % self.config.log_interval == 0:
                    self._log_metrics(metrics)
                    pbar.set_postfix({
                        "loss": f"{metrics['loss']:.4f}",
                        "reward": f"{metrics['reward_mean']:.3f}",
                        "kl_fi": f"{metrics['kl_float_int']:.4f}",
                        "r_con": f"{metrics['r_consistency']:.3f}",
                        "α": f"{metrics['iv_alpha_effective']:.2f}",
                    })

                if eval_dataset and self.global_step % self.config.eval_interval == 0:
                    eval_metrics = self.evaluate(eval_dataset)
                    print(f"\nEval @ step {self.global_step}: {eval_metrics}")

                if self.global_step % self.config.save_interval == 0:
                    self.save_checkpoint(output_dir)

                pbar.update(1)

            if self.global_step >= self.config.total_steps:
                break

        pbar.close()
        self.save_checkpoint(output_dir, final=True)
