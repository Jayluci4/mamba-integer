"""
Adaptive Strategy Selection for Test-Time Scaling.

Allocates compute based on problem difficulty:
- Easy problems: Greedy or low-sample self-consistency
- Medium problems: Standard self-consistency
- Hard problems: Beam search with PRM

This provides optimal compute allocation vs uniform sampling.

Reference: Snell et al. "Scaling LLM Test-Time Compute Optimally"
"""

import re
from dataclasses import dataclass
from typing import Optional, Union
from enum import Enum

import torch
import torch.nn.functional as F

from .self_consistency import SelfConsistency, ConsistencyResult
from .beam_search import BeamSearchPRM, BeamSearchResult


class Difficulty(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


@dataclass
class DifficultyEstimate:
    """Difficulty estimate for a problem."""
    level: Difficulty
    score: float  # 0-1, higher = harder
    reason: str


@dataclass
class AdaptiveResult:
    """Result from adaptive strategy."""
    answer: str
    strategy_used: str
    difficulty: DifficultyEstimate
    confidence: float
    raw_result: Union[ConsistencyResult, BeamSearchResult]


class DifficultyEstimator:
    """
    Estimate problem difficulty using model uncertainty.

    Methods:
    1. Perplexity-based: High perplexity on problem = harder
    2. Answer variance: Low agreement = harder
    3. Heuristic: Problem features (length, operations, digits)
    """

    def __init__(
        self,
        model,
        tokenizer,
        n_probe_samples: int = 4,
        use_perplexity: bool = True,
        use_heuristics: bool = True
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.n_probe_samples = n_probe_samples
        self.use_perplexity = use_perplexity
        self.use_heuristics = use_heuristics

    def _compute_perplexity(self, text: str) -> float:
        """Compute perplexity of text under model."""
        device = next(self.model.parameters()).device
        tokens = self.tokenizer.encode(text)

        if len(tokens) < 2:
            return 1.0

        input_tensor = torch.tensor([tokens], device=device)

        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            logits = self.model(input_tensor)

        # Compute cross-entropy loss
        shift_logits = logits[0, :-1, :]
        shift_labels = input_tensor[0, 1:]

        loss = F.cross_entropy(
            shift_logits.float(),
            shift_labels,
            reduction='mean'
        )

        return torch.exp(loss).item()

    def _heuristic_difficulty(self, problem: str) -> float:
        """Estimate difficulty from problem features."""
        score = 0.0

        # Length factor
        word_count = len(problem.split())
        if word_count > 50:
            score += 0.2
        elif word_count > 100:
            score += 0.4

        # Number of operations
        ops = len(re.findall(r'[\+\-\*\/\^]', problem))
        score += min(ops * 0.1, 0.3)

        # Large numbers
        numbers = re.findall(r'\d+', problem)
        if numbers:
            max_digits = max(len(n) for n in numbers)
            if max_digits > 3:
                score += 0.2
            if max_digits > 5:
                score += 0.2

        # Multi-step indicators
        multi_step_words = ['then', 'after', 'next', 'finally', 'first', 'second']
        for word in multi_step_words:
            if word in problem.lower():
                score += 0.1
                break

        # Algebra/equation indicators
        if re.search(r'[xyz]=|solve|equation', problem.lower()):
            score += 0.3

        return min(score, 1.0)

    @torch.no_grad()
    def _probe_answer_variance(self, problem: str) -> float:
        """
        Quick probe: generate few samples, measure answer variance.

        High variance = harder problem.
        """
        from .self_consistency import SelfConsistency

        sc = SelfConsistency(
            self.model,
            self.tokenizer,
            n_samples=self.n_probe_samples,
            temperature=0.8,
            max_new_tokens=64  # Short probe
        )

        result = sc.solve(problem)

        if not result.all_answers:
            return 1.0  # No valid answers = hard

        # Variance from majority agreement
        if result.confidence < 0.5:
            return 0.9  # Low agreement = hard
        elif result.confidence < 0.75:
            return 0.5
        else:
            return 0.2  # High agreement = easy

    def estimate(self, problem: str) -> DifficultyEstimate:
        """
        Estimate problem difficulty.

        Combines multiple signals into a single difficulty estimate.
        """
        scores = []
        reasons = []

        # Perplexity-based
        if self.use_perplexity:
            ppl = self._compute_perplexity(problem)
            ppl_score = min(ppl / 10.0, 1.0)  # Normalize
            scores.append(ppl_score * 0.3)
            reasons.append(f"perplexity={ppl:.1f}")

        # Heuristic-based
        if self.use_heuristics:
            h_score = self._heuristic_difficulty(problem)
            scores.append(h_score * 0.3)
            reasons.append(f"heuristic={h_score:.2f}")

        # Probe-based (most expensive)
        if self.n_probe_samples > 0:
            v_score = self._probe_answer_variance(problem)
            scores.append(v_score * 0.4)
            reasons.append(f"variance={v_score:.2f}")

        total_score = sum(scores)

        # Classify
        if total_score < 0.3:
            level = Difficulty.EASY
        elif total_score < 0.6:
            level = Difficulty.MEDIUM
        else:
            level = Difficulty.HARD

        return DifficultyEstimate(
            level=level,
            score=total_score,
            reason=", ".join(reasons)
        )


class AdaptiveStrategy:
    """
    Adaptive test-time scaling strategy.

    Selects compute allocation based on problem difficulty:
    - Easy: Greedy or n=2 self-consistency
    - Medium: n=8 self-consistency
    - Hard: Beam search with implicit PRM
    """

    def __init__(
        self,
        model,
        tokenizer,
        easy_samples: int = 2,
        medium_samples: int = 8,
        hard_beam_width: int = 4,
        hard_max_steps: int = 8,
        difficulty_estimator: Optional[DifficultyEstimator] = None,
        max_new_tokens: int = 256
    ):
        self.model = model
        self.tokenizer = tokenizer

        # Strategy configs
        self.easy_samples = easy_samples
        self.medium_samples = medium_samples
        self.hard_beam_width = hard_beam_width
        self.hard_max_steps = hard_max_steps
        self.max_new_tokens = max_new_tokens

        # Difficulty estimator
        self.estimator = difficulty_estimator or DifficultyEstimator(
            model, tokenizer, n_probe_samples=4
        )

        # Pre-build strategies
        self._easy_sc = SelfConsistency(
            model, tokenizer,
            n_samples=easy_samples,
            temperature=0.5,
            max_new_tokens=max_new_tokens
        )

        self._medium_sc = SelfConsistency(
            model, tokenizer,
            n_samples=medium_samples,
            temperature=0.7,
            max_new_tokens=max_new_tokens
        )

        self._hard_beam = BeamSearchPRM(
            model, tokenizer,
            beam_width=hard_beam_width,
            max_steps=hard_max_steps,
            max_tokens_per_step=max_new_tokens // hard_max_steps
        )

    def solve(
        self,
        problem: str,
        force_strategy: Optional[str] = None
    ) -> AdaptiveResult:
        """
        Solve problem with adaptive strategy selection.

        Args:
            problem: The problem to solve
            force_strategy: Override difficulty estimation
                           ("easy", "medium", "hard", or None for auto)

        Returns:
            AdaptiveResult with answer and strategy info
        """
        # Get difficulty estimate (or use forced strategy)
        if force_strategy:
            difficulty = DifficultyEstimate(
                level=Difficulty(force_strategy),
                score=0.5,
                reason=f"forced={force_strategy}"
            )
        else:
            difficulty = self.estimator.estimate(problem)

        # Select and run strategy
        if difficulty.level == Difficulty.EASY:
            result = self._easy_sc.solve(problem)
            strategy = f"self_consistency(n={self.easy_samples})"
            answer = result.answer
            confidence = result.confidence

        elif difficulty.level == Difficulty.MEDIUM:
            result = self._medium_sc.solve(problem)
            strategy = f"self_consistency(n={self.medium_samples})"
            answer = result.answer
            confidence = result.confidence

        else:  # HARD
            result = self._hard_beam.search(problem)
            strategy = f"beam_search(width={self.hard_beam_width})"
            answer = result.answer
            # Confidence from beam score (normalized)
            confidence = min(1.0, max(0.0, (result.score + 10) / 20))

        return AdaptiveResult(
            answer=answer,
            strategy_used=strategy,
            difficulty=difficulty,
            confidence=confidence,
            raw_result=result
        )

    def solve_batch(
        self,
        problems: list,
        force_strategy: Optional[str] = None
    ) -> list:
        """Solve multiple problems."""
        return [self.solve(p, force_strategy) for p in problems]


class ComputeBudgetStrategy(AdaptiveStrategy):
    """
    Variant with explicit compute budget.

    Given a fixed compute budget (in FLOPs or samples),
    allocates optimally across problems.
    """

    def __init__(
        self,
        model,
        tokenizer,
        total_budget: int = 100,  # Total samples across all problems
        **kwargs
    ):
        super().__init__(model, tokenizer, **kwargs)
        self.total_budget = total_budget

    def solve_batch_with_budget(self, problems: list) -> list:
        """
        Solve problems with budget allocation.

        Harder problems get more compute.
        """
        # First pass: estimate difficulties
        difficulties = [self.estimator.estimate(p) for p in problems]

        # Allocate budget based on difficulty
        total_difficulty = sum(d.score for d in difficulties)
        if total_difficulty == 0:
            total_difficulty = len(problems)

        allocations = []
        for d in difficulties:
            # Proportional allocation with minimum
            share = max(2, int(self.total_budget * d.score / total_difficulty))
            allocations.append(share)

        # Solve with allocated compute
        results = []
        for problem, n_samples in zip(problems, allocations):
            # Create custom self-consistency with allocated samples
            sc = SelfConsistency(
                self.model,
                self.tokenizer,
                n_samples=n_samples,
                temperature=0.7,
                max_new_tokens=self.max_new_tokens
            )
            raw_result = sc.solve(problem)

            results.append(AdaptiveResult(
                answer=raw_result.answer,
                strategy_used=f"self_consistency(n={n_samples})",
                difficulty=difficulties[problems.index(problem)],
                confidence=raw_result.confidence,
                raw_result=raw_result
            ))

        return results
