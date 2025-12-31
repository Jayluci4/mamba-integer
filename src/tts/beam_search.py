"""
Beam Search with Implicit Process Reward Model (PRM).

Uses the model's own log-probabilities as step-level rewards,
avoiding the need to train a separate reward model.

Key insight: A well-calibrated model assigns higher probability
to correct reasoning steps than incorrect ones.

Reference:
- Snell et al. "Scaling LLM Test-Time Compute Optimally"
- DeepSeek-R1: Implicit PRM via log-prob ratios
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Callable
import copy

import torch
import torch.nn.functional as F


@dataclass
class BeamState:
    """State of a single beam during search."""
    tokens: List[int]
    score: float  # Cumulative log-prob or reward
    steps: List[str] = field(default_factory=list)  # Reasoning steps
    step_scores: List[float] = field(default_factory=list)
    finished: bool = False

    def __lt__(self, other):
        return self.score < other.score


@dataclass
class BeamSearchResult:
    """Result from beam search."""
    answer: str
    best_response: str
    score: float
    all_beams: List[BeamState]
    n_steps: int


class BeamSearchPRM:
    """
    Beam search with implicit Process Reward Model.

    Scores each reasoning step using log-probability,
    maintaining top-k beams at each step.
    """

    def __init__(
        self,
        model,
        tokenizer,
        beam_width: int = 4,
        max_steps: int = 10,
        max_tokens_per_step: int = 50,
        step_delimiter: str = "\n",
        temperature: float = 0.7,
        length_penalty: float = 0.0,
        use_reference: bool = False,
        reference_model=None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.beam_width = beam_width
        self.max_steps = max_steps
        self.max_tokens_per_step = max_tokens_per_step
        self.step_delimiter = step_delimiter
        self.temperature = temperature
        self.length_penalty = length_penalty
        self.use_reference = use_reference
        self.reference_model = reference_model

    @torch.no_grad()
    def _get_logprobs(self, model, input_ids: torch.Tensor) -> torch.Tensor:
        """Get log-probabilities for next token."""
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            logits = model(input_ids)
        return F.log_softmax(logits[0, -1, :] / self.temperature, dim=-1)

    def _score_step(self, tokens: List[int], step_tokens: List[int]) -> float:
        """
        Score a reasoning step using implicit PRM.

        If use_reference is True, computes:
            score = mean(policy_logprob - reference_logprob)

        Otherwise, just uses raw log-probability.
        """
        device = next(self.model.parameters()).device

        if not step_tokens:
            return 0.0

        # Build context with step
        full_tokens = tokens + step_tokens
        input_tensor = torch.tensor([tokens], device=device)

        total_score = 0.0
        for i, token in enumerate(step_tokens):
            # Get policy log-prob
            policy_logprobs = self._get_logprobs(self.model, input_tensor)
            policy_score = policy_logprobs[token].item()

            if self.use_reference and self.reference_model is not None:
                # Get reference log-prob
                ref_logprobs = self._get_logprobs(self.reference_model, input_tensor)
                ref_score = ref_logprobs[token].item()
                step_score = policy_score - ref_score
            else:
                step_score = policy_score

            total_score += step_score

            # Extend context
            input_tensor = torch.cat([
                input_tensor,
                torch.tensor([[token]], device=device)
            ], dim=1)

        # Average score per token (with length penalty)
        avg_score = total_score / (len(step_tokens) ** self.length_penalty) if step_tokens else 0
        return avg_score

    def _generate_step_candidates(
        self,
        tokens: List[int],
        n_candidates: int
    ) -> List[Tuple[List[int], float]]:
        """
        Generate multiple candidate next steps.

        Returns list of (step_tokens, score) tuples.
        """
        device = next(self.model.parameters()).device
        candidates = []

        for _ in range(n_candidates):
            input_tensor = torch.tensor([tokens], device=device)
            step_tokens = []
            step_score = 0.0

            for _ in range(self.max_tokens_per_step):
                logprobs = self._get_logprobs(self.model, input_tensor)

                # Sample next token
                probs = torch.exp(logprobs)
                next_token = torch.multinomial(probs, num_samples=1).item()

                step_tokens.append(next_token)
                step_score += logprobs[next_token].item()

                input_tensor = torch.cat([
                    input_tensor,
                    torch.tensor([[next_token]], device=device)
                ], dim=1)

                # Check for step delimiter
                decoded = self.tokenizer.decode([next_token])
                if self.step_delimiter in decoded or next_token == 0:
                    break

            candidates.append((step_tokens, step_score))

        return candidates

    def _is_answer_step(self, step_text: str) -> bool:
        """Check if step contains final answer."""
        patterns = [
            r'\\boxed\{',
            r'answer\s+is',
            r'therefore',
            r'thus',
            r'finally',
            r'result\s*[:=]'
        ]
        return any(re.search(p, step_text.lower()) for p in patterns)

    def _extract_answer(self, response: str) -> str:
        """Extract final answer from response."""
        # Try boxed format
        boxed = re.search(r'\\boxed\{([^}]+)\}', response)
        if boxed:
            return boxed.group(1).strip()

        # Try "answer is X"
        answer_is = re.search(r'answer\s+is\s+[:\s]*(-?\d+(?:\.\d+)?)', response, re.I)
        if answer_is:
            return answer_is.group(1)

        # Last number
        numbers = re.findall(r'-?\d+(?:\.\d+)?', response)
        if numbers:
            return numbers[-1]

        return ""

    def search(self, prompt: str) -> BeamSearchResult:
        """
        Run beam search with implicit PRM.

        Args:
            prompt: Problem to solve

        Returns:
            BeamSearchResult with best answer and all beams
        """
        # Encode prompt
        prompt_tokens = self.tokenizer.encode(prompt)

        # Initialize beams
        beams = [BeamState(tokens=prompt_tokens.copy(), score=0.0)]

        for step_idx in range(self.max_steps):
            all_candidates = []

            for beam in beams:
                if beam.finished:
                    all_candidates.append(beam)
                    continue

                # Generate candidate next steps
                candidates = self._generate_step_candidates(
                    beam.tokens,
                    n_candidates=self.beam_width * 2
                )

                for step_tokens, raw_score in candidates:
                    # Score the step
                    step_score = self._score_step(beam.tokens, step_tokens)

                    # Create new beam state
                    new_tokens = beam.tokens + step_tokens
                    new_beam = BeamState(
                        tokens=new_tokens,
                        score=beam.score + step_score,
                        steps=beam.steps + [self.tokenizer.decode(step_tokens)],
                        step_scores=beam.step_scores + [step_score]
                    )

                    # Check if this step contains final answer
                    step_text = self.tokenizer.decode(step_tokens)
                    if self._is_answer_step(step_text) or step_idx == self.max_steps - 1:
                        new_beam.finished = True

                    all_candidates.append(new_beam)

            # Keep top beams
            all_candidates.sort(key=lambda b: b.score, reverse=True)
            beams = all_candidates[:self.beam_width]

            # Check if all beams finished
            if all(b.finished for b in beams):
                break

        # Get best beam
        best_beam = max(beams, key=lambda b: b.score)
        full_response = self.tokenizer.decode(best_beam.tokens[len(prompt_tokens):])
        answer = self._extract_answer(full_response)

        return BeamSearchResult(
            answer=answer,
            best_response=full_response,
            score=best_beam.score,
            all_beams=beams,
            n_steps=len(best_beam.steps)
        )


class StepwiseBeamSearch(BeamSearchPRM):
    """
    Variant that explicitly separates reasoning into steps.

    Uses step boundaries (like "Step 1:", "Step 2:") for cleaner
    beam expansion.
    """

    def __init__(self, *args, step_prompt: str = "Step {n}: ", **kwargs):
        super().__init__(*args, **kwargs)
        self.step_prompt = step_prompt

    def _generate_step_with_prompt(self, tokens: List[int], step_num: int) -> List[Tuple[List[int], float]]:
        """Generate step with explicit step prompt."""
        device = next(self.model.parameters()).device

        # Add step prompt
        step_prompt_text = self.step_prompt.format(n=step_num)
        step_prompt_tokens = self.tokenizer.encode(step_prompt_text)

        full_tokens = tokens + step_prompt_tokens
        return self._generate_step_candidates(full_tokens, self.beam_width * 2)

    def search(self, prompt: str) -> BeamSearchResult:
        """Run stepwise beam search."""
        prompt_tokens = self.tokenizer.encode(prompt + "\n\n")
        beams = [BeamState(tokens=prompt_tokens.copy(), score=0.0)]

        for step_num in range(1, self.max_steps + 1):
            all_candidates = []

            for beam in beams:
                if beam.finished:
                    all_candidates.append(beam)
                    continue

                # Generate with step prompt
                candidates = self._generate_step_with_prompt(beam.tokens, step_num)

                step_prompt_tokens = self.tokenizer.encode(
                    self.step_prompt.format(n=step_num)
                )

                for step_tokens, raw_score in candidates:
                    full_step = step_prompt_tokens + step_tokens
                    step_score = self._score_step(beam.tokens, full_step)

                    new_tokens = beam.tokens + full_step
                    step_text = self.tokenizer.decode(full_step)

                    new_beam = BeamState(
                        tokens=new_tokens,
                        score=beam.score + step_score,
                        steps=beam.steps + [step_text],
                        step_scores=beam.step_scores + [step_score]
                    )

                    # Check for answer
                    if self._is_answer_step(step_text):
                        new_beam.finished = True

                    all_candidates.append(new_beam)

            all_candidates.sort(key=lambda b: b.score, reverse=True)
            beams = all_candidates[:self.beam_width]

            if all(b.finished for b in beams):
                break

        best_beam = max(beams, key=lambda b: b.score)
        full_response = self.tokenizer.decode(best_beam.tokens[len(prompt_tokens):])
        answer = self._extract_answer(full_response)

        return BeamSearchResult(
            answer=answer,
            best_response=full_response,
            score=best_beam.score,
            all_beams=beams,
            n_steps=len(best_beam.steps)
        )
