"""
SFT Data Generation Pipeline via vLLM

Generates verified SFT training data using strong teacher models served by vLLM,
filtered through the domain verifier stack. Pipeline: teacher LLM generates →
domain verifiers filter → verified-correct examples become SFT data.

Usage:
    from src.rlvr.sft_generator import SFTGenerator, SFTGenerationConfig
    config = SFTGenerationConfig(model_name="Qwen/Qwen3-32B")
    gen = SFTGenerator(config)
    result = gen.generate_one("What is 7*8?", "math", "56")
"""

import hashlib
import json
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

from .domain_verifiers import create_verifier
from .rewards import RewardResult, VerifiableReward

logger = logging.getLogger(__name__)

try:
    import openai
except ImportError:
    openai = None  # Deferred error at usage time


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class SFTGenerationConfig:
    """Configuration for SFT data generation."""

    # vLLM endpoints
    default_endpoint: str = "http://localhost:8000/v1"
    domain_endpoints: Optional[Dict[str, str]] = None
    api_key: str = "EMPTY"
    model_name: Optional[str] = None

    # Generation parameters
    max_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.95
    n_completions: int = 8
    early_stop_on_verified: bool = True

    # Verification thresholds
    default_reward_threshold: float = 0.6
    domain_reward_thresholds: Dict[str, float] = field(default_factory=dict)

    # Execution
    batch_size: int = 16
    max_retries: int = 3

    # Output
    output_dir: str = "sft_data"
    enable_resume: bool = True

    # Router data collection
    collect_router_data: bool = True

    def get_reward_threshold(self, domain: str) -> float:
        """Get reward threshold for a domain, falling back to default."""
        return self.domain_reward_thresholds.get(domain, self.default_reward_threshold)

    def get_endpoint(self, domain: str) -> str:
        """Get endpoint for a domain, falling back to default."""
        if self.domain_endpoints and domain in self.domain_endpoints:
            return self.domain_endpoints[domain]
        return self.default_endpoint

    def to_json(self) -> str:
        """Serialize config to JSON."""
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, data: str) -> "SFTGenerationConfig":
        """Deserialize config from JSON string."""
        d = json.loads(data)
        return cls(**d)


@dataclass
class SFTGenerationResult:
    """Result of generating and verifying a single SFT example."""

    prompt: str
    response: str
    domain: str
    ground_truth: Optional[str]
    verified: bool
    reward: float
    reason: str
    n_attempts: int
    generation_time_ms: float


@dataclass
class PromptTemplate:
    """Per-domain system prompt template."""
    domain: str
    system_prompt: str


DOMAIN_PROMPT_TEMPLATES: Dict[str, PromptTemplate] = {
    "solidity": PromptTemplate(
        domain="solidity",
        system_prompt=(
            "You are an expert Solidity developer. Write secure, gas-efficient smart contracts. "
            "Always wrap your Solidity code in ```solidity code blocks. Include SPDX license "
            "identifiers and pragma statements."
        ),
    ),
    "sql": PromptTemplate(
        domain="sql",
        system_prompt=(
            "You are an expert SQL developer. Write correct, efficient SQL queries. "
            "Always wrap your SQL code in ```sql code blocks. Use standard SQL syntax."
        ),
    ),
    "math": PromptTemplate(
        domain="math",
        system_prompt=(
            "You are an expert mathematician. Solve problems step by step and state your "
            "final answer clearly using \\boxed{} notation or 'The answer is X' format."
        ),
    ),
    "sec_finance": PromptTemplate(
        domain="sec_finance",
        system_prompt=(
            "You are a financial analyst. Provide accurate financial summaries with specific "
            "dollar amounts and percentages. Use bullet points for key metrics."
        ),
    ),
    "english": PromptTemplate(
        domain="english",
        system_prompt=(
            "You are a skilled writer. Produce clear, coherent, well-structured prose. "
            "Use proper grammar and varied sentence structure."
        ),
    ),
}

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. Provide clear, accurate, and well-structured responses."
)


# =============================================================================
# VLLM CLIENT
# =============================================================================

class VLLMClient:
    """Thin wrapper around openai.OpenAI for vLLM's OpenAI-compatible API."""

    def __init__(self, config: SFTGenerationConfig):
        if openai is None:
            raise ImportError(
                "openai package is required for SFT generation. "
                "Install it with: pip install openai"
            )
        self.config = config
        self._clients: Dict[str, "openai.OpenAI"] = {}

    def _get_client(self, endpoint: str) -> "openai.OpenAI":
        """Get or create a cached OpenAI client for an endpoint."""
        if endpoint not in self._clients:
            self._clients[endpoint] = openai.OpenAI(
                base_url=endpoint,
                api_key=self.config.api_key,
            )
        return self._clients[endpoint]

    def generate(
        self,
        prompt: str,
        domain: str,
        n: int = 1,
        system_prompt: Optional[str] = None,
    ) -> List[str]:
        """Generate completions via vLLM.

        Args:
            prompt: User prompt
            domain: Domain name for endpoint routing
            n: Number of completions to generate
            system_prompt: Override system prompt (uses domain default if None)

        Returns:
            List of generated response strings
        """
        endpoint = self.config.get_endpoint(domain)
        client = self._get_client(endpoint)

        if system_prompt is None:
            template = DOMAIN_PROMPT_TEMPLATES.get(domain)
            system_prompt = template.system_prompt if template else DEFAULT_SYSTEM_PROMPT

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        last_error = None
        for attempt in range(self.config.max_retries):
            try:
                response = client.chat.completions.create(
                    model=self.config.model_name or "default",
                    messages=messages,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    n=n,
                )
                return [choice.message.content for choice in response.choices]
            except openai.APIConnectionError as e:
                raise ConnectionError(
                    f"Cannot connect to vLLM at {endpoint}. "
                    f"Start with: vllm serve <model> --host 0.0.0.0 --port 8000\n"
                    f"Error: {e}"
                ) from e
            except (openai.RateLimitError, openai.APITimeoutError, openai.APIStatusError) as e:
                last_error = e
                wait = 2 ** attempt
                logger.warning(f"API error (attempt {attempt + 1}/{self.config.max_retries}): {e}. Retrying in {wait}s...")
                time.sleep(wait)

        raise last_error


# =============================================================================
# SFT GENERATOR
# =============================================================================

class SFTGenerator:
    """Main class for generating verified SFT training data."""

    def __init__(self, config: SFTGenerationConfig):
        self.config = config
        self.client = VLLMClient(config)
        self._verifier_cache: Dict[str, VerifiableReward] = {}
        self._stats = {
            "total": 0,
            "verified": 0,
            "failed": 0,
            "errors": 0,
            "per_domain": {},
            "total_attempts": 0,
            "total_time_ms": 0.0,
        }

    def _get_verifier(self, domain: str) -> VerifiableReward:
        """Get or create a cached verifier for the domain."""
        if domain not in self._verifier_cache:
            self._verifier_cache[domain] = create_verifier(domain)
        return self._verifier_cache[domain]

    def _init_domain_stats(self, domain: str) -> None:
        """Initialize per-domain stats if needed."""
        if domain not in self._stats["per_domain"]:
            self._stats["per_domain"][domain] = {
                "total": 0,
                "verified": 0,
                "failed": 0,
                "errors": 0,
                "total_attempts": 0,
                "total_time_ms": 0.0,
            }

    def _update_stats(self, result: SFTGenerationResult) -> None:
        """Update stats counters from a result."""
        self._init_domain_stats(result.domain)
        ds = self._stats["per_domain"][result.domain]

        self._stats["total"] += 1
        ds["total"] += 1
        self._stats["total_attempts"] += result.n_attempts
        ds["total_attempts"] += result.n_attempts
        self._stats["total_time_ms"] += result.generation_time_ms
        ds["total_time_ms"] += result.generation_time_ms

        if result.verified:
            self._stats["verified"] += 1
            ds["verified"] += 1
        else:
            self._stats["failed"] += 1
            ds["failed"] += 1

    def generate_one(
        self,
        prompt: str,
        domain: str,
        ground_truth: Optional[str] = None,
    ) -> SFTGenerationResult:
        """Generate and verify a single SFT example.

        Uses Best-of-N sampling: generates completions and picks the first
        (early_stop) or best verified response.

        Args:
            prompt: Input prompt
            domain: Domain for verifier selection
            ground_truth: Optional ground truth for verification

        Returns:
            SFTGenerationResult with best response (verified or not)
        """
        verifier = self._get_verifier(domain)
        threshold = self.config.get_reward_threshold(domain)
        start_time = time.time()

        best_result = None
        best_reward = -1.0
        n_attempts = 0

        if self.config.early_stop_on_verified:
            # Generate one at a time, stop on first verified
            for i in range(self.config.n_completions):
                try:
                    responses = self.client.generate(prompt, domain, n=1)
                except Exception as e:
                    logger.warning(f"Generation error for prompt '{prompt[:50]}...': {e}")
                    n_attempts += 1
                    continue

                for resp in responses:
                    n_attempts += 1
                    try:
                        reward_result = verifier.compute(prompt, resp, ground_truth)
                    except Exception as e:
                        logger.warning(f"Verifier error: {e}")
                        continue

                    if reward_result.reward >= threshold and reward_result.reward > best_reward:
                        best_result = (resp, reward_result)
                        best_reward = reward_result.reward

                    if best_result is not None:
                        break  # Early stop on first verified

                if best_result is not None:
                    break
        else:
            # Generate all N at once, pick best verified
            try:
                responses = self.client.generate(
                    prompt, domain, n=self.config.n_completions
                )
            except Exception as e:
                logger.warning(f"Generation error for prompt '{prompt[:50]}...': {e}")
                elapsed = (time.time() - start_time) * 1000
                result = SFTGenerationResult(
                    prompt=prompt,
                    response="",
                    domain=domain,
                    ground_truth=ground_truth,
                    verified=False,
                    reward=0.0,
                    reason=f"Generation error: {e}",
                    n_attempts=0,
                    generation_time_ms=elapsed,
                )
                self._update_stats(result)
                return result

            for resp in responses:
                n_attempts += 1
                try:
                    reward_result = verifier.compute(prompt, resp, ground_truth)
                except Exception as e:
                    logger.warning(f"Verifier error: {e}")
                    continue

                if reward_result.reward >= threshold and reward_result.reward > best_reward:
                    best_result = (resp, reward_result)
                    best_reward = reward_result.reward

        elapsed = (time.time() - start_time) * 1000

        if best_result is not None:
            resp, rr = best_result
            result = SFTGenerationResult(
                prompt=prompt,
                response=resp,
                domain=domain,
                ground_truth=ground_truth,
                verified=True,
                reward=rr.reward,
                reason=rr.reason,
                n_attempts=n_attempts,
                generation_time_ms=elapsed,
            )
        else:
            result = SFTGenerationResult(
                prompt=prompt,
                response="",
                domain=domain,
                ground_truth=ground_truth,
                verified=False,
                reward=0.0,
                reason="No completion passed verification",
                n_attempts=n_attempts,
                generation_time_ms=elapsed,
            )

        self._update_stats(result)
        return result

    def generate_batch(
        self,
        examples: List[Dict[str, Any]],
    ) -> List[SFTGenerationResult]:
        """Generate SFT data for a batch of examples in parallel.

        Args:
            examples: List of dicts with "prompt", "domain", optional "ground_truth"

        Returns:
            List of SFTGenerationResult
        """
        results = []
        with ThreadPoolExecutor(max_workers=self.config.batch_size) as executor:
            futures = {}
            for i, ex in enumerate(examples):
                future = executor.submit(
                    self.generate_one,
                    ex["prompt"],
                    ex["domain"],
                    ex.get("ground_truth"),
                )
                futures[future] = i

            result_map = {}
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    result_map[idx] = future.result()
                except Exception as e:
                    logger.error(f"Batch item {idx} failed: {e}")
                    ex = examples[idx]
                    result_map[idx] = SFTGenerationResult(
                        prompt=ex["prompt"],
                        response="",
                        domain=ex["domain"],
                        ground_truth=ex.get("ground_truth"),
                        verified=False,
                        reward=0.0,
                        reason=f"Error: {e}",
                        n_attempts=0,
                        generation_time_ms=0.0,
                    )
                    self._stats["errors"] += 1

            # Preserve input order
            for i in range(len(examples)):
                results.append(result_map[i])

        return results

    def _prompt_hash(self, prompt: str) -> str:
        """SHA256 hash of a prompt for resume tracking."""
        return hashlib.sha256(prompt.encode()).hexdigest()

    def _load_progress(self, progress_path: str) -> set:
        """Load set of processed prompt hashes from progress file."""
        processed = set()
        if os.path.exists(progress_path):
            with open(progress_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        processed.add(line)
        return processed

    def _result_to_jsonl(self, result: SFTGenerationResult) -> str:
        """Convert a result to JSONL format compatible with DomainDataset."""
        entry = {
            "prompt": result.prompt,
            "domain": result.domain,
            "ground_truth": result.ground_truth,
            "response": result.response,
            "metadata": {
                "reward": result.reward,
                "reason": result.reason,
                "n_attempts": result.n_attempts,
                "generation_time_ms": result.generation_time_ms,
            },
        }
        return json.dumps(entry, ensure_ascii=False)

    def generate_from_jsonl(
        self,
        input_path: str,
        output_path: Optional[str] = None,
    ) -> str:
        """Generate SFT data from a JSONL file of prompts.

        Input JSONL format: {"prompt": "...", "domain": "...", "ground_truth": "..."}

        Args:
            input_path: Path to input JSONL file
            output_path: Path to output JSONL file (default: output_dir/sft_output.jsonl)

        Returns:
            Path to output JSONL file
        """
        os.makedirs(self.config.output_dir, exist_ok=True)
        if output_path is None:
            output_path = os.path.join(self.config.output_dir, "sft_output.jsonl")

        progress_path = output_path + ".progress"

        # Load progress for resume
        processed = set()
        if self.config.enable_resume:
            processed = self._load_progress(progress_path)
            if processed:
                logger.info(f"Resuming: {len(processed)} prompts already processed")

        # Read input
        examples = []
        with open(input_path, "r") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    ex = json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping malformed JSONL line {line_num}: {e}")
                    continue

                if "prompt" not in ex or "domain" not in ex:
                    logger.warning(f"Skipping line {line_num}: missing 'prompt' or 'domain'")
                    continue

                prompt_hash = self._prompt_hash(ex["prompt"])
                if prompt_hash in processed:
                    continue

                examples.append(ex)

        logger.info(f"Processing {len(examples)} prompts ({len(processed)} already done)")

        # Process in batches
        try:
            with open(output_path, "a") as out_f, open(progress_path, "a") as prog_f:
                for batch_start in range(0, len(examples), self.config.batch_size):
                    batch = examples[batch_start:batch_start + self.config.batch_size]
                    results = self.generate_batch(batch)

                    for ex, result in zip(batch, results):
                        if result.verified:
                            out_f.write(self._result_to_jsonl(result) + "\n")
                            out_f.flush()

                        prog_f.write(self._prompt_hash(ex["prompt"]) + "\n")
                        prog_f.flush()

                    done = min(batch_start + len(batch), len(examples))
                    logger.info(
                        f"Progress: {done}/{len(examples)} "
                        f"(verified: {self._stats['verified']}/{self._stats['total']})"
                    )
        except KeyboardInterrupt:
            logger.info("Interrupted — progress saved, partial results written.")

        # Write stats
        stats = self.get_stats()
        stats_path = os.path.join(self.config.output_dir, "generation_stats.json")
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Stats written to {stats_path}")

        return output_path

    def get_stats(self) -> Dict[str, Any]:
        """Get generation statistics."""
        total = self._stats["total"]
        verified = self._stats["verified"]
        stats = {
            "total_prompts": total,
            "verified": verified,
            "rate": verified / total if total > 0 else 0.0,
            "failed": self._stats["failed"],
            "errors": self._stats["errors"],
            "avg_attempts": (
                self._stats["total_attempts"] / total if total > 0 else 0.0
            ),
            "avg_time_ms": (
                self._stats["total_time_ms"] / total if total > 0 else 0.0
            ),
            "per_domain": {},
        }
        for domain, ds in self._stats["per_domain"].items():
            dt = ds["total"]
            dv = ds["verified"]
            stats["per_domain"][domain] = {
                "total": dt,
                "verified": dv,
                "rate": dv / dt if dt > 0 else 0.0,
                "avg_attempts": ds["total_attempts"] / dt if dt > 0 else 0.0,
                "avg_time_ms": ds["total_time_ms"] / dt if dt > 0 else 0.0,
            }
        return stats
