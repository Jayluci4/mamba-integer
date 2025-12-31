"""
Self-Consistency: Majority voting over multiple samples.

Simple but effective TTS method. Generate N samples with temperature > 0,
extract the final answer from each, return the most common answer.

Reference: Wang et al. "Self-Consistency Improves Chain of Thought Reasoning"
"""

import re
from collections import Counter
from typing import List, Optional, Tuple, Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class ConsistencyResult:
    """Result from self-consistency voting."""
    answer: str
    confidence: float  # Fraction of samples agreeing
    all_answers: List[str]
    all_responses: List[str]
    vote_counts: dict


class SelfConsistency:
    """
    Self-consistency via majority voting.

    Generate multiple samples, extract answers, return majority vote.
    Higher temperature encourages diverse reasoning paths.
    """

    def __init__(
        self,
        model,
        tokenizer,
        n_samples: int = 8,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_new_tokens: int = 256,
        answer_extractor: Optional[Callable[[str], Optional[str]]] = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.n_samples = n_samples
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.answer_extractor = answer_extractor or self._default_extractor

    def _default_extractor(self, response: str) -> Optional[str]:
        """
        Extract numerical answer from response.

        Looks for patterns like:
        - "The answer is 42"
        - "= 42"
        - "\\boxed{42}"
        - Just the number at the end
        """
        response = response.strip()

        # Try boxed format (LaTeX)
        boxed = re.search(r'\\boxed\{([^}]+)\}', response)
        if boxed:
            return boxed.group(1).strip()

        # Try "answer is X" pattern
        answer_is = re.search(r'answer\s+is\s+[:\s]*(-?\d+(?:\.\d+)?)', response, re.I)
        if answer_is:
            return answer_is.group(1)

        # Try "= X" at end
        equals = re.search(r'=\s*(-?\d+(?:\.\d+)?)\s*$', response)
        if equals:
            return equals.group(1)

        # Try last number in response
        numbers = re.findall(r'-?\d+(?:\.\d+)?', response)
        if numbers:
            return numbers[-1]

        return None

    @torch.no_grad()
    def generate_samples(self, prompt: str) -> List[str]:
        """Generate N diverse samples for the prompt."""
        device = next(self.model.parameters()).device

        # Encode prompt
        input_ids = self.tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids], device=device)

        samples = []
        for _ in range(self.n_samples):
            # Generate with sampling
            output_ids = self._generate_one(input_tensor)
            response = self.tokenizer.decode(output_ids[len(input_ids):])
            samples.append(response)

        return samples

    def _generate_one(self, input_ids: torch.Tensor) -> List[int]:
        """Generate one sample with temperature sampling."""
        device = input_ids.device
        generated = input_ids.clone()

        for _ in range(self.max_new_tokens):
            # Forward pass
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                logits = self.model(generated)

            # Get next token logits
            next_logits = logits[0, -1, :] / self.temperature

            # Top-p filtering
            sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative prob above threshold
            sorted_indices_to_remove = cumulative_probs > self.top_p
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = False

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            next_logits[indices_to_remove] = float('-inf')

            # Sample
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)

            # Check for EOS (token 0 or newline patterns)
            if next_token.item() == 0:
                break

        return generated[0].tolist()

    def solve(self, prompt: str) -> ConsistencyResult:
        """
        Solve problem using self-consistency.

        Args:
            prompt: The problem to solve

        Returns:
            ConsistencyResult with majority answer and confidence
        """
        # Generate samples
        responses = self.generate_samples(prompt)

        # Extract answers
        answers = []
        for resp in responses:
            ans = self.answer_extractor(resp)
            if ans is not None:
                answers.append(ans)

        if not answers:
            return ConsistencyResult(
                answer="",
                confidence=0.0,
                all_answers=[],
                all_responses=responses,
                vote_counts={}
            )

        # Majority vote
        vote_counts = Counter(answers)
        majority_answer, majority_count = vote_counts.most_common(1)[0]
        confidence = majority_count / len(answers)

        return ConsistencyResult(
            answer=majority_answer,
            confidence=confidence,
            all_answers=answers,
            all_responses=responses,
            vote_counts=dict(vote_counts)
        )

    def solve_batch(self, prompts: List[str]) -> List[ConsistencyResult]:
        """Solve multiple problems."""
        return [self.solve(p) for p in prompts]


class WeightedSelfConsistency(SelfConsistency):
    """
    Self-consistency with log-probability weighting.

    Instead of equal votes, weight each answer by the
    log-probability of the full response.
    """

    @torch.no_grad()
    def generate_samples_with_logprobs(self, prompt: str) -> List[Tuple[str, float]]:
        """Generate samples and compute their log-probabilities."""
        device = next(self.model.parameters()).device

        input_ids = self.tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids], device=device)

        samples = []
        for _ in range(self.n_samples):
            output_ids, logprob = self._generate_one_with_logprob(input_tensor)
            response = self.tokenizer.decode(output_ids[len(input_ids):])
            samples.append((response, logprob))

        return samples

    def _generate_one_with_logprob(self, input_ids: torch.Tensor) -> Tuple[List[int], float]:
        """Generate one sample and track log-probability."""
        device = input_ids.device
        generated = input_ids.clone()
        total_logprob = 0.0

        for _ in range(self.max_new_tokens):
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                logits = self.model(generated)

            next_logits = logits[0, -1, :] / self.temperature

            # Top-p filtering
            sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            sorted_indices_to_remove = cumulative_probs > self.top_p
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = False

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            next_logits[indices_to_remove] = float('-inf')

            # Sample and track logprob
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Add log probability of chosen token
            token_logprob = torch.log(probs[next_token.item()] + 1e-10).item()
            total_logprob += token_logprob

            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)

            if next_token.item() == 0:
                break

        return generated[0].tolist(), total_logprob

    def solve(self, prompt: str) -> ConsistencyResult:
        """Solve with log-probability weighted voting."""
        samples = self.generate_samples_with_logprobs(prompt)

        # Extract answers with weights
        answer_weights = {}
        all_answers = []
        all_responses = []

        for response, logprob in samples:
            all_responses.append(response)
            ans = self.answer_extractor(response)
            if ans is not None:
                all_answers.append(ans)
                # Use exp(logprob) as weight (normalized later)
                weight = torch.exp(torch.tensor(logprob)).item()
                answer_weights[ans] = answer_weights.get(ans, 0) + weight

        if not answer_weights:
            return ConsistencyResult(
                answer="",
                confidence=0.0,
                all_answers=[],
                all_responses=all_responses,
                vote_counts={}
            )

        # Find highest weighted answer
        total_weight = sum(answer_weights.values())
        majority_answer = max(answer_weights, key=answer_weights.get)
        confidence = answer_weights[majority_answer] / total_weight if total_weight > 0 else 0

        return ConsistencyResult(
            answer=majority_answer,
            confidence=confidence,
            all_answers=all_answers,
            all_responses=all_responses,
            vote_counts={k: v/total_weight for k, v in answer_weights.items()}
        )
