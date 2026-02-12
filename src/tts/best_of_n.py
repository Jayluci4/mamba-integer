"""
Best-of-N with Verifiable Rewards (Phase 4a).

Generate N completions, verify each with domain-specific verifiers,
return the best verified response. Unlike self-consistency (majority vote),
this uses formal verification — a single correct output among N is sufficient.

Key advantage for ternary models: the model runs in milliseconds,
so N=32 adds negligible latency. Proven more compute-efficient than
scaling parameters for small models.

Reference: Post-training playbook Phase 4a
"""

from dataclasses import dataclass, field
from typing import List, Optional, Callable, Tuple
import time

import torch
import torch.nn.functional as F


@dataclass
class BestOfNResult:
    """Result from Best-of-N with verification."""
    best_response: str
    best_reward: float
    verified: bool  # Whether best response passed verification
    all_responses: List[str]
    all_rewards: List[float]
    all_verified: List[bool]
    n_verified: int  # How many passed verification
    n_total: int
    min_n_needed: int  # First N where a verified response appeared
    wall_time_ms: float = 0.0


class BestOfNVerified:
    """
    Best-of-N inference with verifiable rewards.

    Generates N completions and selects the best one that passes
    domain-specific verification. Falls back to highest-reward
    unverified response if none pass.

    Args:
        model: Language model with forward(input_ids) -> logits
        tokenizer: Tokenizer with encode/decode methods
        verifier: Verifier with verify(prompt, response, ground_truth) method
                  returning an object with .correct and .score attributes
        n_samples: Number of completions to generate
        temperature: Sampling temperature
        top_p: Nucleus sampling threshold
        max_new_tokens: Maximum tokens per completion
        early_stop: Stop generating once a verified response is found
    """

    def __init__(
        self,
        model,
        tokenizer,
        verifier,
        n_samples: int = 32,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_new_tokens: int = 256,
        early_stop: bool = False,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.verifier = verifier
        self.n_samples = n_samples
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.early_stop = early_stop

    @torch.no_grad()
    def _generate_one(self, input_ids: torch.Tensor) -> Tuple[List[int], float]:
        """Generate one sample with temperature sampling. Returns (tokens, logprob)."""
        device = input_ids.device
        generated = input_ids.clone()
        total_logprob = 0.0

        for _ in range(self.max_new_tokens):
            logits = self.model(generated)
            next_logits = logits[0, -1, :].float() / self.temperature

            # Top-p filtering
            sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > self.top_p
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = False
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            next_logits[indices_to_remove] = float('-inf')

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            total_logprob += torch.log(probs[next_token.item()] + 1e-10).item()

            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
            if next_token.item() == 0:
                break

        return generated[0].tolist(), total_logprob

    @torch.no_grad()
    def generate_and_verify(
        self,
        prompt: str,
        ground_truth: Optional[str] = None,
    ) -> BestOfNResult:
        """
        Generate N samples and verify each.

        Args:
            prompt: The problem/prompt
            ground_truth: Expected answer (for verifiers that need it)

        Returns:
            BestOfNResult with best verified response
        """
        start_time = time.time()
        device = next(self.model.parameters()).device

        input_ids = self.tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids], device=device)

        all_responses = []
        all_rewards = []
        all_verified = []
        min_n_needed = -1

        for i in range(self.n_samples):
            # Generate
            output_ids, logprob = self._generate_one(input_tensor)
            response = self.tokenizer.decode(output_ids[len(input_ids):])
            all_responses.append(response)

            # Verify
            result = self.verifier.verify(prompt, response, ground_truth)
            reward = result.score if hasattr(result, 'score') else (1.0 if result.correct else 0.0)
            all_rewards.append(reward)
            all_verified.append(result.correct)

            if result.correct and min_n_needed < 0:
                min_n_needed = i + 1

            # Early stop if we found a verified response
            if self.early_stop and result.correct:
                break

        n_verified = sum(all_verified)
        wall_time_ms = (time.time() - start_time) * 1000

        # Select best response
        if n_verified > 0:
            # Among verified responses, pick highest reward
            best_idx = max(
                (i for i, v in enumerate(all_verified) if v),
                key=lambda i: all_rewards[i]
            )
            verified = True
        else:
            # No verified responses — fall back to highest reward
            best_idx = max(range(len(all_rewards)), key=lambda i: all_rewards[i])
            verified = False
            min_n_needed = -1

        return BestOfNResult(
            best_response=all_responses[best_idx],
            best_reward=all_rewards[best_idx],
            verified=verified,
            all_responses=all_responses,
            all_rewards=all_rewards,
            all_verified=all_verified,
            n_verified=n_verified,
            n_total=len(all_responses),
            min_n_needed=min_n_needed,
            wall_time_ms=wall_time_ms,
        )

    def pass_at_k(
        self,
        prompt: str,
        ground_truth: Optional[str] = None,
        k_values: Optional[List[int]] = None,
    ) -> dict:
        """
        Compute pass@k for various k values.

        Generates n_samples completions, then computes the probability
        of at least one correct answer in a random subset of size k.

        Returns:
            Dict mapping k -> pass@k probability
        """
        if k_values is None:
            k_values = [1, 2, 4, 8, 16, 32]

        result = self.generate_and_verify(prompt, ground_truth)
        n = result.n_total
        c = result.n_verified

        pass_rates = {}
        for k in k_values:
            if k > n:
                continue
            if c == 0:
                pass_rates[k] = 0.0
            elif c >= n:
                pass_rates[k] = 1.0
            else:
                # Unbiased pass@k estimator: 1 - C(n-c, k) / C(n, k)
                # Use log-space for numerical stability
                import math
                log_numerator = sum(math.log(n - c - i) for i in range(k) if n - c - i > 0)
                log_denominator = sum(math.log(n - i) for i in range(k))
                if n - c < k:
                    pass_rates[k] = 1.0
                else:
                    pass_rates[k] = 1.0 - math.exp(log_numerator - log_denominator)

        return pass_rates


class BestOfNWithReranking(BestOfNVerified):
    """
    Best-of-N with two-stage reranking.

    Stage 1: Generate N samples, verify with domain verifier
    Stage 2: Among verified samples, rerank by log-probability
             (higher logprob = model is more confident = likely better quality)
    """

    @torch.no_grad()
    def generate_and_verify(
        self,
        prompt: str,
        ground_truth: Optional[str] = None,
    ) -> BestOfNResult:
        """Generate, verify, then rerank by logprob."""
        start_time = time.time()
        device = next(self.model.parameters()).device

        input_ids = self.tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids], device=device)

        all_responses = []
        all_rewards = []
        all_verified = []
        all_logprobs = []
        min_n_needed = -1

        for i in range(self.n_samples):
            output_ids, logprob = self._generate_one(input_tensor)
            response = self.tokenizer.decode(output_ids[len(input_ids):])
            all_responses.append(response)
            all_logprobs.append(logprob)

            result = self.verifier.verify(prompt, response, ground_truth)
            reward = result.score if hasattr(result, 'score') else (1.0 if result.correct else 0.0)
            all_rewards.append(reward)
            all_verified.append(result.correct)

            if result.correct and min_n_needed < 0:
                min_n_needed = i + 1

            if self.early_stop and result.correct:
                break

        n_verified = sum(all_verified)
        wall_time_ms = (time.time() - start_time) * 1000

        if n_verified > 0:
            # Among verified: rerank by reward first, then logprob as tiebreaker
            best_idx = max(
                (i for i, v in enumerate(all_verified) if v),
                key=lambda i: (all_rewards[i], all_logprobs[i])
            )
            verified = True
        else:
            # No verified — pick by logprob (model confidence)
            best_idx = max(range(len(all_logprobs)), key=lambda i: all_logprobs[i])
            verified = False
            min_n_needed = -1

        return BestOfNResult(
            best_response=all_responses[best_idx],
            best_reward=all_rewards[best_idx],
            verified=verified,
            all_responses=all_responses,
            all_rewards=all_rewards,
            all_verified=all_verified,
            n_verified=n_verified,
            n_total=len(all_responses),
            min_n_needed=min_n_needed,
            wall_time_ms=wall_time_ms,
        )


def compute_optimal_n(
    model,
    tokenizer,
    verifier,
    prompts: List[str],
    ground_truths: List[Optional[str]],
    max_n: int = 128,
    temperature: float = 0.7,
    max_new_tokens: int = 256,
) -> List[dict]:
    """
    For a set of prompts, find the minimum N needed for verification.

    This generates the training data for the Meta-Inference Router (Phase 5d):
    for each prompt, we find the minimum N needed to get a verified-correct response.

    Args:
        model: Language model
        tokenizer: Tokenizer
        verifier: Domain verifier
        prompts: List of prompts
        ground_truths: List of expected answers
        max_n: Maximum samples per prompt
        temperature: Sampling temperature
        max_new_tokens: Max tokens per completion

    Returns:
        List of dicts with {prompt, ground_truth, min_n, pass_rate, domain}
    """
    solver = BestOfNVerified(
        model=model,
        tokenizer=tokenizer,
        verifier=verifier,
        n_samples=max_n,
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        early_stop=False,  # Need full data for analysis
    )

    results = []
    for prompt, gt in zip(prompts, ground_truths):
        result = solver.generate_and_verify(prompt, gt)
        domain = getattr(verifier, 'name', 'unknown')

        results.append({
            'prompt': prompt,
            'ground_truth': gt,
            'min_n': result.min_n_needed,
            'n_verified': result.n_verified,
            'n_total': result.n_total,
            'pass_rate': result.n_verified / result.n_total if result.n_total > 0 else 0,
            'domain': domain,
            'wall_time_ms': result.wall_time_ms,
        })

    return results
