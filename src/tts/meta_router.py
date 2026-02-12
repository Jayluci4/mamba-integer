"""
Meta-Inference Router (Phase 5d): Learned Compute Allocation.

A tiny router network (~100K params) that predicts per-prompt optimal
compute budget. Replaces heuristic difficulty estimation with a learned
model trained on verifier outcomes.

Training the router is FREE for verifiable domains:
1. For 10K prompts, generate N=128 completions
2. Record minimum N needed to find a correct completion (via verifier)
3. This gives (prompt, min_N) supervised pairs at zero labeling cost
4. Train router: prompt features -> (N, temperature, strategy)

At inference:
- Easy prompts: N=2, greedy -> millisecond response
- Hard prompts: N=64, beam search with verification pruning

Compute savings: If 60% easy (N=2), 30% medium (N=8), 10% hard (N=64)
-> average N=10 vs fixed N=32. ~3.2x inference speedup at equal accuracy.

Reference: Post-training playbook Phase 5d
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Configuration
# =============================================================================

class InferenceStrategy(Enum):
    BEST_OF_N = "best_of_n"
    SELF_CONSISTENCY = "self_consistency"
    BEAM_SEARCH = "beam_search"


@dataclass
class RouterConfig:
    """Configuration for the Meta-Inference Router."""
    # Prompt encoding
    vocab_size: int = 8192
    d_embed: int = 64
    d_hidden: int = 128
    n_encoder_layers: int = 2
    max_prompt_tokens: int = 128

    # Router output
    n_strategies: int = 3  # best-of-n, self-consistency, beam-search
    n_buckets: int = 8     # N prediction buckets: [1, 2, 4, 8, 16, 32, 64, 128]
    n_temp_bins: int = 5   # Temperature bins: [0.1, 0.3, 0.5, 0.7, 1.0]

    # Training
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    n_epochs: int = 50
    batch_size: int = 64

    @property
    def n_buckets_list(self) -> List[int]:
        return [1, 2, 4, 8, 16, 32, 64, 128][:self.n_buckets]

    @property
    def temp_bins_list(self) -> List[float]:
        return [0.1, 0.3, 0.5, 0.7, 1.0][:self.n_temp_bins]


@dataclass
class RouterDecision:
    """Output from the router."""
    n_samples: int
    temperature: float
    strategy: InferenceStrategy
    confidence: float
    predicted_difficulty: float  # 0=easy, 1=hard
    raw_logits: Optional[Dict[str, torch.Tensor]] = field(default=None, repr=False)


# =============================================================================
# Prompt Encoder
# =============================================================================

class PromptEncoder(nn.Module):
    """
    Encode prompt into a fixed-size feature vector.

    Uses lightweight embedding + 1D convolution + pooling.
    ~50K params — captures prompt structure without heavy attention.
    """

    def __init__(self, config: RouterConfig):
        super().__init__()
        self.config = config

        self.embed = nn.Embedding(config.vocab_size, config.d_embed)
        self.conv1 = nn.Conv1d(config.d_embed, config.d_hidden, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(config.d_hidden, config.d_hidden, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.norm = nn.LayerNorm(config.d_hidden)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embed.weight, std=0.02)
        for conv in [self.conv1, self.conv2]:
            nn.init.kaiming_normal_(conv.weight)
            nn.init.zeros_(conv.bias)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Encode prompt tokens to feature vector.

        Args:
            input_ids: [batch, seq_len] token IDs

        Returns:
            [batch, d_hidden] feature vector
        """
        # Clamp to vocab size
        input_ids = input_ids.clamp(0, self.config.vocab_size - 1)

        x = self.embed(input_ids)  # [B, L, d_embed]
        x = x.transpose(1, 2)     # [B, d_embed, L]

        x = F.relu(self.conv1(x))  # [B, d_hidden, L]
        x = F.relu(self.conv2(x))  # [B, d_hidden, L]

        x = self.pool(x).squeeze(-1)  # [B, d_hidden]
        x = self.norm(x)

        return x


# =============================================================================
# Meta-Inference Router Network
# =============================================================================

class MetaInferenceRouter(nn.Module):
    """
    Learned compute router (~100K params).

    Predicts per-prompt:
    1. N bucket (how many samples to generate)
    2. Temperature (sampling diversity)
    3. Strategy (best-of-n, self-consistency, beam-search)

    Trained on (prompt, min_N_needed) data from verification runs.
    """

    def __init__(self, config: Optional[RouterConfig] = None):
        super().__init__()
        self.config = config or RouterConfig()

        # Prompt encoder
        self.encoder = PromptEncoder(self.config)

        # Prediction heads
        d = self.config.d_hidden

        # N-bucket predictor
        self.n_head = nn.Sequential(
            nn.Linear(d, d),
            nn.ReLU(),
            nn.Linear(d, self.config.n_buckets),
        )

        # Temperature predictor
        self.temp_head = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.ReLU(),
            nn.Linear(d // 2, self.config.n_temp_bins),
        )

        # Strategy predictor
        self.strategy_head = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.ReLU(),
            nn.Linear(d // 2, self.config.n_strategies),
        )

        # Difficulty regressor (auxiliary — predicts pass@1 rate)
        self.difficulty_head = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.ReLU(),
            nn.Linear(d // 2, 1),
            nn.Sigmoid(),
        )

        self._init_heads()

    def _init_heads(self):
        for module in [self.n_head, self.temp_head, self.strategy_head, self.difficulty_head]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_normal_(layer.weight, gain=0.5)
                    nn.init.zeros_(layer.bias)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        input_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            input_ids: [batch, seq_len] prompt token IDs

        Returns:
            n_logits: [batch, n_buckets] — logits over N buckets
            temp_logits: [batch, n_temp_bins] — logits over temperature bins
            strategy_logits: [batch, n_strategies] — logits over strategies
            difficulty: [batch, 1] — predicted difficulty (0-1)
        """
        features = self.encoder(input_ids)  # [B, d_hidden]

        n_logits = self.n_head(features)
        temp_logits = self.temp_head(features)
        strategy_logits = self.strategy_head(features)
        difficulty = self.difficulty_head(features)

        return n_logits, temp_logits, strategy_logits, difficulty

    def predict(
        self,
        input_ids: torch.Tensor,
    ) -> RouterDecision:
        """
        Make a routing decision for a single prompt.

        Args:
            input_ids: [1, seq_len] or [seq_len] token IDs

        Returns:
            RouterDecision with N, temperature, strategy
        """
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        self.eval()
        with torch.no_grad():
            n_logits, temp_logits, strat_logits, difficulty = self(input_ids)

        # Decode N bucket
        n_probs = F.softmax(n_logits[0], dim=-1)
        n_idx = n_probs.argmax().item()
        n_samples = self.config.n_buckets_list[n_idx]
        n_confidence = n_probs[n_idx].item()

        # Decode temperature
        temp_probs = F.softmax(temp_logits[0], dim=-1)
        temp_idx = temp_probs.argmax().item()
        temperature = self.config.temp_bins_list[temp_idx]

        # Decode strategy
        strat_probs = F.softmax(strat_logits[0], dim=-1)
        strat_idx = strat_probs.argmax().item()
        strategies = [InferenceStrategy.BEST_OF_N,
                      InferenceStrategy.SELF_CONSISTENCY,
                      InferenceStrategy.BEAM_SEARCH]
        strategy = strategies[strat_idx]

        return RouterDecision(
            n_samples=n_samples,
            temperature=temperature,
            strategy=strategy,
            confidence=n_confidence,
            predicted_difficulty=difficulty[0, 0].item(),
            raw_logits={
                'n_logits': n_logits[0],
                'temp_logits': temp_logits[0],
                'strategy_logits': strat_logits[0],
            },
        )


# =============================================================================
# Router Training
# =============================================================================

@dataclass
class RouterTrainingExample:
    """One training example for the router."""
    prompt_ids: List[int]
    min_n: int          # Minimum N needed for correct answer
    pass_rate: float    # Fraction of N=128 that passed
    domain: str
    best_temperature: float = 0.7  # Can be determined from data
    best_strategy: int = 0         # Index into strategy list


class RouterTrainer:
    """
    Train the Meta-Inference Router from verification data.

    Training data is FREE for verifiable domains:
    - Generate N=128 completions per prompt
    - Record min N needed to find correct answer
    - This gives (prompt, min_N) supervised pairs
    """

    def __init__(
        self,
        router: MetaInferenceRouter,
        config: Optional[RouterConfig] = None,
    ):
        self.router = router
        self.config = config or router.config
        self.train_losses = []

    def _n_to_bucket(self, min_n: int) -> int:
        """Convert min_N to bucket index."""
        buckets = self.config.n_buckets_list
        for i, b in enumerate(buckets):
            if min_n <= b:
                return i
        return len(buckets) - 1

    def _pass_rate_to_difficulty(self, pass_rate: float) -> float:
        """Convert pass@N rate to difficulty score. Low pass rate = hard."""
        return 1.0 - pass_rate

    def _prepare_batch(
        self,
        examples: List[RouterTrainingExample],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare a batch of training examples."""
        max_len = self.config.max_prompt_tokens

        # Pad/truncate prompt IDs
        batch_ids = []
        for ex in examples:
            ids = ex.prompt_ids[:max_len]
            ids = ids + [0] * (max_len - len(ids))
            batch_ids.append(ids)

        input_ids = torch.tensor(batch_ids, device=device)

        # Targets
        n_targets = torch.tensor(
            [self._n_to_bucket(ex.min_n) for ex in examples],
            device=device, dtype=torch.long
        )
        temp_targets = torch.tensor(
            [ex.best_temperature for ex in examples],
            device=device, dtype=torch.float
        )
        strategy_targets = torch.tensor(
            [ex.best_strategy for ex in examples],
            device=device, dtype=torch.long
        )
        difficulty_targets = torch.tensor(
            [self._pass_rate_to_difficulty(ex.pass_rate) for ex in examples],
            device=device, dtype=torch.float
        ).unsqueeze(1)

        return input_ids, n_targets, temp_targets, strategy_targets, difficulty_targets

    def train(
        self,
        examples: List[RouterTrainingExample],
        val_examples: Optional[List[RouterTrainingExample]] = None,
        device: Optional[torch.device] = None,
    ) -> Dict[str, List[float]]:
        """
        Train the router on verification data.

        Args:
            examples: Training examples from compute_optimal_n
            val_examples: Optional validation examples
            device: Device to train on

        Returns:
            Dict with training metrics history
        """
        if device is None:
            device = next(self.router.parameters()).device

        self.router.to(device)
        self.router.train()

        optimizer = torch.optim.AdamW(
            self.router.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.n_epochs
        )

        history = {
            'train_loss': [], 'n_loss': [], 'difficulty_loss': [],
            'val_loss': [], 'n_accuracy': [],
        }

        batch_size = self.config.batch_size

        for epoch in range(self.config.n_epochs):
            # Shuffle
            import random
            indices = list(range(len(examples)))
            random.shuffle(indices)

            epoch_loss = 0.0
            epoch_n_loss = 0.0
            epoch_diff_loss = 0.0
            n_batches = 0

            for start in range(0, len(indices), batch_size):
                batch_indices = indices[start:start + batch_size]
                batch = [examples[i] for i in batch_indices]

                input_ids, n_targets, temp_targets, strat_targets, diff_targets = \
                    self._prepare_batch(batch, device)

                # Forward
                n_logits, temp_logits, strat_logits, difficulty = self.router(input_ids)

                # Losses
                # Primary: N-bucket classification (most important)
                n_loss = F.cross_entropy(n_logits, n_targets)

                # Auxiliary: difficulty regression
                diff_loss = F.mse_loss(difficulty, diff_targets)

                # Strategy classification (when we have labels)
                strat_loss = F.cross_entropy(strat_logits, strat_targets)

                # Combined loss
                loss = n_loss + 0.5 * diff_loss + 0.3 * strat_loss

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.router.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                epoch_n_loss += n_loss.item()
                epoch_diff_loss += diff_loss.item()
                n_batches += 1

            scheduler.step()

            avg_loss = epoch_loss / max(n_batches, 1)
            avg_n_loss = epoch_n_loss / max(n_batches, 1)
            avg_diff_loss = epoch_diff_loss / max(n_batches, 1)

            history['train_loss'].append(avg_loss)
            history['n_loss'].append(avg_n_loss)
            history['difficulty_loss'].append(avg_diff_loss)

            # Validation
            if val_examples:
                val_loss, n_acc = self._evaluate(val_examples, device)
                history['val_loss'].append(val_loss)
                history['n_accuracy'].append(n_acc)

        self.train_losses = history['train_loss']
        return history

    @torch.no_grad()
    def _evaluate(
        self,
        examples: List[RouterTrainingExample],
        device: torch.device,
    ) -> Tuple[float, float]:
        """Evaluate router on examples. Returns (loss, n_bucket_accuracy)."""
        self.router.eval()

        input_ids, n_targets, temp_targets, strat_targets, diff_targets = \
            self._prepare_batch(examples, device)

        n_logits, temp_logits, strat_logits, difficulty = self.router(input_ids)

        n_loss = F.cross_entropy(n_logits, n_targets)
        n_preds = n_logits.argmax(dim=-1)
        n_accuracy = (n_preds == n_targets).float().mean().item()

        self.router.train()
        return n_loss.item(), n_accuracy


# =============================================================================
# Routed Inference (End-to-End)
# =============================================================================

class RoutedInference:
    """
    End-to-end inference with learned compute routing.

    Uses the Meta-Inference Router to predict compute budget,
    then dispatches to the appropriate strategy.

    Args:
        model: Language model
        tokenizer: Tokenizer
        router: Trained MetaInferenceRouter
        verifier: Domain verifier (for Best-of-N verification)
        max_prompt_tokens: Max tokens for router input
    """

    def __init__(
        self,
        model,
        tokenizer,
        router: MetaInferenceRouter,
        verifier=None,
        max_prompt_tokens: int = 128,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.router = router
        self.verifier = verifier
        self.max_prompt_tokens = max_prompt_tokens

    def _encode_prompt(self, prompt: str) -> torch.Tensor:
        """Encode prompt for router."""
        device = next(self.router.parameters()).device
        ids = self.tokenizer.encode(prompt)[:self.max_prompt_tokens]
        return torch.tensor([ids], device=device)

    def solve(
        self,
        prompt: str,
        ground_truth: Optional[str] = None,
        override_n: Optional[int] = None,
        override_strategy: Optional[InferenceStrategy] = None,
    ) -> Dict:
        """
        Solve a prompt using router-guided inference.

        Args:
            prompt: The problem to solve
            ground_truth: Expected answer (for verification)
            override_n: Override router's N prediction
            override_strategy: Override router's strategy prediction

        Returns:
            Dict with answer, strategy used, router decision, etc.
        """
        # Get router decision
        input_ids = self._encode_prompt(prompt)
        decision = self.router.predict(input_ids)

        # Apply overrides
        n = override_n or decision.n_samples
        strategy = override_strategy or decision.strategy

        # Dispatch to strategy
        if strategy == InferenceStrategy.BEST_OF_N and self.verifier:
            from .best_of_n import BestOfNVerified
            solver = BestOfNVerified(
                model=self.model,
                tokenizer=self.tokenizer,
                verifier=self.verifier,
                n_samples=n,
                temperature=decision.temperature,
            )
            result = solver.generate_and_verify(prompt, ground_truth)
            answer = result.best_response
            verified = result.verified
            confidence = result.best_reward

        elif strategy == InferenceStrategy.SELF_CONSISTENCY:
            from .self_consistency import SelfConsistency
            solver = SelfConsistency(
                model=self.model,
                tokenizer=self.tokenizer,
                n_samples=n,
                temperature=decision.temperature,
            )
            result = solver.solve(prompt)
            answer = result.answer
            verified = result.confidence > 0.5
            confidence = result.confidence

        elif strategy == InferenceStrategy.BEAM_SEARCH:
            from .beam_search import BeamSearchPRM
            beam_width = min(n, 8)  # Beam width from N, capped
            solver = BeamSearchPRM(
                model=self.model,
                tokenizer=self.tokenizer,
                beam_width=beam_width,
                temperature=decision.temperature,
            )
            result = solver.search(prompt)
            answer = result.answer
            verified = False  # Beam search doesn't verify
            confidence = min(1.0, max(0.0, (result.score + 10) / 20))

        else:
            # Fallback: Best-of-N without verifier
            from .self_consistency import SelfConsistency
            solver = SelfConsistency(
                model=self.model,
                tokenizer=self.tokenizer,
                n_samples=n,
                temperature=decision.temperature,
            )
            result = solver.solve(prompt)
            answer = result.answer
            verified = False
            confidence = result.confidence

        return {
            'answer': answer,
            'verified': verified,
            'confidence': confidence,
            'strategy': strategy.value,
            'n_samples': n,
            'temperature': decision.temperature,
            'predicted_difficulty': decision.predicted_difficulty,
            'router_confidence': decision.confidence,
            'raw_result': result,
        }

    def solve_batch(
        self,
        prompts: List[str],
        ground_truths: Optional[List[str]] = None,
    ) -> List[Dict]:
        """Solve multiple prompts with per-prompt routing."""
        if ground_truths is None:
            ground_truths = [None] * len(prompts)
        return [
            self.solve(p, gt) for p, gt in zip(prompts, ground_truths)
        ]

    def compute_savings(self, decisions: List[RouterDecision]) -> Dict:
        """
        Compute theoretical speedup from routing vs fixed N.

        Returns:
            Dict with average_n, baseline_n, speedup_factor
        """
        if not decisions:
            return {'average_n': 0, 'baseline_n': 32, 'speedup_factor': 1.0}

        avg_n = sum(d.n_samples for d in decisions) / len(decisions)
        baseline_n = 32  # Default fixed N

        return {
            'average_n': avg_n,
            'baseline_n': baseline_n,
            'speedup_factor': baseline_n / max(avg_n, 1),
            'n_distribution': {
                n: sum(1 for d in decisions if d.n_samples == n)
                for n in sorted(set(d.n_samples for d in decisions))
            },
        }
