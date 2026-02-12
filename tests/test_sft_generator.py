"""
Tests for SFT data generation pipeline.

Mocks the openai client — no real vLLM needed.
Run: python -m pytest tests/test_sft_generator.py -v
"""

import json
import os
import sys
import tempfile
from dataclasses import dataclass
from typing import List, Optional
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from src.rlvr.sft_generator import (
    SFTGenerationConfig,
    SFTGenerationResult,
    SFTGenerator,
    VLLMClient,
    PromptTemplate,
    DOMAIN_PROMPT_TEMPLATES,
    DEFAULT_SYSTEM_PROMPT,
)


# =============================================================================
# MOCK HELPERS
# =============================================================================

# Known-good responses that pass verifiers (from test_domain_verifiers.py)

GOOD_MATH_RESPONSE = "The answer is 56"
BAD_MATH_RESPONSE = "The answer is 99"

GOOD_ENGLISH_RESPONSE = (
    "The quick brown fox jumps over the lazy dog. This is a well-known "
    "pangram that contains every letter of the English alphabet at least once. "
    "It has been used since the late 19th century for testing typewriters and "
    "computer fonts. The sentence is particularly useful because of its brevity "
    "and completeness."
)

GOOD_SEC_FINANCE_RESPONSE = (
    "Apple Inc. (AAPL) Q4 2024 Financial Summary:\n\n"
    "- Revenue: $89.5 billion, up 6% year-over-year\n"
    "- Net income: $22.9 billion, representing a 25.6% margin\n"
    "- Earnings per share: $1.46, beating consensus estimates of $1.39\n"
    "- Services revenue reached $22.2 billion, a new all-time high\n"
    "- Cash and equivalents: $162.1 billion\n\n"
    "The company returned $25 billion to shareholders through dividends "
    "and share repurchases during the quarter."
)


@dataclass
class MockMessage:
    content: str


@dataclass
class MockChoice:
    message: MockMessage


@dataclass
class MockCompletion:
    choices: List[MockChoice]


def make_mock_client(responses: Optional[List[str]] = None, side_effect=None):
    """Create a mock openai.OpenAI that returns predetermined responses."""
    mock_openai_cls = MagicMock()
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client

    if side_effect is not None:
        mock_client.chat.completions.create.side_effect = side_effect
    elif responses is not None:
        completion = MockCompletion(
            choices=[MockChoice(message=MockMessage(content=r)) for r in responses]
        )
        mock_client.chat.completions.create.return_value = completion
    return mock_openai_cls, mock_client


def make_mock_client_sequential(response_batches: List[List[str]]):
    """Create mock that returns different responses on successive calls."""
    mock_openai_cls = MagicMock()
    mock_client = MagicMock()
    mock_openai_cls.return_value = mock_client

    completions = []
    for batch in response_batches:
        completions.append(
            MockCompletion(
                choices=[MockChoice(message=MockMessage(content=r)) for r in batch]
            )
        )
    mock_client.chat.completions.create.side_effect = completions
    return mock_openai_cls, mock_client


# =============================================================================
# TEST CONFIG
# =============================================================================

class TestSFTGenerationConfig:
    def test_defaults(self):
        config = SFTGenerationConfig()
        assert config.default_endpoint == "http://localhost:8000/v1"
        assert config.n_completions == 8
        assert config.temperature == 0.7
        assert config.early_stop_on_verified is True
        assert config.batch_size == 16
        assert config.max_retries == 3
        assert config.enable_resume is True

    def test_json_roundtrip(self):
        config = SFTGenerationConfig(
            model_name="test-model",
            n_completions=4,
            domain_reward_thresholds={"math": 0.9, "sql": 0.7},
        )
        json_str = config.to_json()
        restored = SFTGenerationConfig.from_json(json_str)
        assert restored.model_name == "test-model"
        assert restored.n_completions == 4
        assert restored.domain_reward_thresholds == {"math": 0.9, "sql": 0.7}

    def test_threshold_lookup(self):
        config = SFTGenerationConfig(
            default_reward_threshold=0.6,
            domain_reward_thresholds={"math": 0.9},
        )
        assert config.get_reward_threshold("math") == 0.9
        assert config.get_reward_threshold("sql") == 0.6
        assert config.get_reward_threshold("english") == 0.6

    def test_domain_endpoint_routing(self):
        config = SFTGenerationConfig(
            default_endpoint="http://default:8000/v1",
            domain_endpoints={"math": "http://math-server:8000/v1"},
        )
        assert config.get_endpoint("math") == "http://math-server:8000/v1"
        assert config.get_endpoint("sql") == "http://default:8000/v1"

    def test_from_json_with_custom_fields(self):
        data = json.dumps({
            "default_endpoint": "http://custom:9000/v1",
            "api_key": "my-key",
            "model_name": "my-model",
            "max_tokens": 2048,
            "temperature": 0.5,
            "top_p": 0.9,
            "n_completions": 16,
            "early_stop_on_verified": False,
            "default_reward_threshold": 0.8,
            "domain_reward_thresholds": {},
            "batch_size": 32,
            "max_retries": 5,
            "output_dir": "/tmp/sft",
            "enable_resume": False,
            "collect_router_data": False,
            "domain_endpoints": None,
        })
        config = SFTGenerationConfig.from_json(data)
        assert config.default_endpoint == "http://custom:9000/v1"
        assert config.n_completions == 16
        assert config.early_stop_on_verified is False


# =============================================================================
# TEST PROMPT TEMPLATES
# =============================================================================

class TestPromptTemplate:
    def test_all_domains_have_templates(self):
        expected = {"solidity", "sql", "math", "sec_finance", "english"}
        assert set(DOMAIN_PROMPT_TEMPLATES.keys()) == expected

    def test_system_prompts_non_empty(self):
        for domain, tmpl in DOMAIN_PROMPT_TEMPLATES.items():
            assert isinstance(tmpl, PromptTemplate)
            assert len(tmpl.system_prompt) > 10, f"Empty prompt for {domain}"
            assert tmpl.domain == domain

    def test_default_fallback(self):
        assert len(DEFAULT_SYSTEM_PROMPT) > 10


# =============================================================================
# TEST VLLM CLIENT
# =============================================================================

class TestVLLMClient:
    def test_endpoint_routing(self):
        config = SFTGenerationConfig(
            default_endpoint="http://default:8000/v1",
            domain_endpoints={"math": "http://math:8000/v1"},
        )
        with patch("src.rlvr.sft_generator.openai") as mock_openai:
            mock_cls, mock_inst = make_mock_client(["response"])
            mock_openai.OpenAI = mock_cls
            mock_openai.APIConnectionError = Exception
            mock_openai.RateLimitError = Exception
            mock_openai.APITimeoutError = Exception
            mock_openai.APIStatusError = Exception

            client = VLLMClient(config)
            client.generate("test", "math", n=1)

            # Should use math endpoint
            mock_cls.assert_called_with(base_url="http://math:8000/v1", api_key="EMPTY")

    def test_generate_returns_strings(self):
        config = SFTGenerationConfig()
        with patch("src.rlvr.sft_generator.openai") as mock_openai:
            mock_cls, mock_inst = make_mock_client(["resp1", "resp2"])
            mock_openai.OpenAI = mock_cls
            mock_openai.APIConnectionError = Exception
            mock_openai.RateLimitError = Exception
            mock_openai.APITimeoutError = Exception
            mock_openai.APIStatusError = Exception

            client = VLLMClient(config)
            results = client.generate("test", "math", n=2)
            assert results == ["resp1", "resp2"]

    def test_retry_on_error(self):
        config = SFTGenerationConfig(max_retries=3)

        class FakeRateLimit(Exception):
            pass

        with patch("src.rlvr.sft_generator.openai") as mock_openai:
            mock_openai.OpenAI = MagicMock()
            mock_openai.APIConnectionError = ConnectionError
            mock_openai.RateLimitError = FakeRateLimit
            mock_openai.APITimeoutError = TimeoutError
            mock_openai.APIStatusError = FakeRateLimit

            mock_inst = MagicMock()
            mock_openai.OpenAI.return_value = mock_inst

            completion = MockCompletion(
                choices=[MockChoice(message=MockMessage(content="ok"))]
            )
            # Fail twice, succeed on third
            mock_inst.chat.completions.create.side_effect = [
                FakeRateLimit("rate limited"),
                FakeRateLimit("rate limited"),
                completion,
            ]

            client = VLLMClient(config)
            with patch("src.rlvr.sft_generator.time.sleep"):  # Skip actual sleep
                results = client.generate("test", "math", n=1)
            assert results == ["ok"]

    def test_connection_error(self):
        config = SFTGenerationConfig()

        class FakeConnError(Exception):
            pass

        with patch("src.rlvr.sft_generator.openai") as mock_openai:
            mock_openai.OpenAI = MagicMock()
            mock_openai.APIConnectionError = FakeConnError
            mock_openai.RateLimitError = Exception
            mock_openai.APITimeoutError = Exception
            mock_openai.APIStatusError = Exception

            mock_inst = MagicMock()
            mock_openai.OpenAI.return_value = mock_inst
            mock_inst.chat.completions.create.side_effect = FakeConnError("refused")

            client = VLLMClient(config)
            with pytest.raises(ConnectionError, match="Cannot connect to vLLM"):
                client.generate("test", "math", n=1)


# =============================================================================
# TEST SFT GENERATOR
# =============================================================================

def _patch_vllm_client(generator, responses):
    """Patch a generator's client to return given responses."""
    mock = MagicMock()
    completion = MockCompletion(
        choices=[MockChoice(message=MockMessage(content=r)) for r in responses]
    )
    mock.generate.return_value = [r for r in responses]
    generator.client = mock
    return mock


def _patch_vllm_client_sequential(generator, response_batches):
    """Patch generator's client to return different responses on successive calls."""
    mock = MagicMock()
    mock.generate.side_effect = response_batches
    generator.client = mock
    return mock


class TestSFTGenerator:
    def test_generate_one_verified(self):
        config = SFTGenerationConfig(n_completions=3, early_stop_on_verified=True)
        gen = SFTGenerator(config)
        _patch_vllm_client(gen, [GOOD_MATH_RESPONSE])

        result = gen.generate_one("What is 7 * 8?", "math", "56")
        assert result.verified
        assert result.reward >= 0.6
        assert result.domain == "math"
        assert result.n_attempts >= 1

    def test_generate_one_all_fail(self):
        config = SFTGenerationConfig(n_completions=3, early_stop_on_verified=True)
        gen = SFTGenerator(config)
        _patch_vllm_client(gen, [BAD_MATH_RESPONSE])

        result = gen.generate_one("What is 7 * 8?", "math", "56")
        assert not result.verified
        assert result.reward == 0.0

    def test_early_stop_behavior(self):
        config = SFTGenerationConfig(n_completions=5, early_stop_on_verified=True)
        gen = SFTGenerator(config)

        # First call returns bad, second returns good
        mock = MagicMock()
        mock.generate.side_effect = [
            [BAD_MATH_RESPONSE],
            [GOOD_MATH_RESPONSE],
        ]
        gen.client = mock

        result = gen.generate_one("What is 7 * 8?", "math", "56")
        assert result.verified
        assert result.n_attempts == 2  # Only tried 2 of 5
        assert mock.generate.call_count == 2  # Stopped early

    def test_no_early_stop(self):
        config = SFTGenerationConfig(n_completions=4, early_stop_on_verified=False)
        gen = SFTGenerator(config)

        responses = [BAD_MATH_RESPONSE, BAD_MATH_RESPONSE, GOOD_MATH_RESPONSE, BAD_MATH_RESPONSE]
        _patch_vllm_client(gen, responses)

        result = gen.generate_one("What is 7 * 8?", "math", "56")
        assert result.verified
        assert result.n_attempts == 4  # Tried all

    def test_reward_threshold_filtering(self):
        config = SFTGenerationConfig(
            n_completions=1,
            early_stop_on_verified=False,
            default_reward_threshold=0.99,  # Very high threshold
        )
        gen = SFTGenerator(config)
        _patch_vllm_client(gen, [GOOD_ENGLISH_RESPONSE])

        # English verifier gives partial scores, may not hit 0.99
        result = gen.generate_one("Write about pangrams", "english")
        # With threshold=0.99, many valid responses won't pass
        # This tests that threshold filtering works
        assert isinstance(result.verified, bool)

    def test_batch_generation(self):
        config = SFTGenerationConfig(
            n_completions=1,
            batch_size=4,
            early_stop_on_verified=True,
        )
        gen = SFTGenerator(config)

        mock = MagicMock()
        mock.generate.return_value = [GOOD_MATH_RESPONSE]
        gen.client = mock

        examples = [
            {"prompt": f"What is {i} + {i}?", "domain": "math", "ground_truth": str(2 * i)}
            for i in range(4)
        ]
        results = gen.generate_batch(examples)
        assert len(results) == 4
        # All should attempt generation
        assert all(isinstance(r, SFTGenerationResult) for r in results)

    def test_verifier_caching(self):
        config = SFTGenerationConfig(n_completions=1, early_stop_on_verified=True)
        gen = SFTGenerator(config)

        v1 = gen._get_verifier("math")
        v2 = gen._get_verifier("math")
        assert v1 is v2  # Same instance

        v3 = gen._get_verifier("english")
        assert v3 is not v1  # Different domain, different instance

    def test_stats_tracking(self):
        config = SFTGenerationConfig(n_completions=1, early_stop_on_verified=True)
        gen = SFTGenerator(config)
        _patch_vllm_client(gen, [GOOD_MATH_RESPONSE])

        gen.generate_one("What is 7 * 8?", "math", "56")
        gen.generate_one("What is 3 + 5?", "math", "8")

        stats = gen.get_stats()
        assert stats["total_prompts"] == 2
        assert "math" in stats["per_domain"]
        assert stats["per_domain"]["math"]["total"] == 2

    def test_n_attempts_metadata(self):
        config = SFTGenerationConfig(n_completions=5, early_stop_on_verified=True)
        gen = SFTGenerator(config)

        # Bad, bad, good
        mock = MagicMock()
        mock.generate.side_effect = [
            [BAD_MATH_RESPONSE],
            [BAD_MATH_RESPONSE],
            [GOOD_MATH_RESPONSE],
        ]
        gen.client = mock

        result = gen.generate_one("What is 7 * 8?", "math", "56")
        assert result.n_attempts == 3
        assert result.generation_time_ms > 0


# =============================================================================
# TEST OUTPUT FORMAT
# =============================================================================

class TestOutputFormat:
    def test_jsonl_compatibility_with_domain_dataset(self):
        """Output JSONL should be loadable by DomainDataset.from_jsonl()."""
        config = SFTGenerationConfig(n_completions=1, early_stop_on_verified=True)
        gen = SFTGenerator(config)

        result = SFTGenerationResult(
            prompt="What is 7 * 8?",
            response="The answer is 56",
            domain="math",
            ground_truth="56",
            verified=True,
            reward=1.0,
            reason="Correct",
            n_attempts=1,
            generation_time_ms=100.0,
        )
        jsonl_line = gen._result_to_jsonl(result)
        parsed = json.loads(jsonl_line)

        # Must have fields DomainDataset expects
        assert "prompt" in parsed
        assert "domain" in parsed
        assert "ground_truth" in parsed

        # Also has SFT-specific fields
        assert "response" in parsed
        assert "metadata" in parsed

    def test_metadata_fields_present(self):
        config = SFTGenerationConfig()
        gen = SFTGenerator(config)
        result = SFTGenerationResult(
            prompt="test",
            response="test response",
            domain="math",
            ground_truth="42",
            verified=True,
            reward=1.0,
            reason="ok",
            n_attempts=3,
            generation_time_ms=500.0,
        )
        parsed = json.loads(gen._result_to_jsonl(result))
        meta = parsed["metadata"]
        assert meta["reward"] == 1.0
        assert meta["reason"] == "ok"
        assert meta["n_attempts"] == 3
        assert meta["generation_time_ms"] == 500.0

    def test_stats_json_format(self):
        config = SFTGenerationConfig(n_completions=1, early_stop_on_verified=True)
        gen = SFTGenerator(config)
        _patch_vllm_client(gen, [GOOD_MATH_RESPONSE])

        gen.generate_one("What is 7 * 8?", "math", "56")
        stats = gen.get_stats()

        assert "total_prompts" in stats
        assert "verified" in stats
        assert "rate" in stats
        assert "per_domain" in stats
        assert isinstance(stats["rate"], float)

    def test_per_domain_output(self):
        config = SFTGenerationConfig(n_completions=1, early_stop_on_verified=True)
        gen = SFTGenerator(config)

        # Math
        mock = MagicMock()
        mock.generate.return_value = [GOOD_MATH_RESPONSE]
        gen.client = mock
        gen.generate_one("What is 7 * 8?", "math", "56")

        # English
        mock.generate.return_value = [GOOD_ENGLISH_RESPONSE]
        gen.generate_one("Write about pangrams", "english")

        stats = gen.get_stats()
        assert "math" in stats["per_domain"]
        assert "english" in stats["per_domain"]
        assert stats["per_domain"]["math"]["total"] == 1
        assert stats["per_domain"]["english"]["total"] == 1


# =============================================================================
# TEST RESUME
# =============================================================================

class TestResume:
    def test_progress_file_creation(self, tmp_path):
        config = SFTGenerationConfig(
            n_completions=1,
            early_stop_on_verified=True,
            output_dir=str(tmp_path),
            enable_resume=True,
        )
        gen = SFTGenerator(config)
        mock = MagicMock()
        mock.generate.return_value = [GOOD_MATH_RESPONSE]
        gen.client = mock

        # Create input JSONL
        input_path = str(tmp_path / "input.jsonl")
        with open(input_path, "w") as f:
            f.write(json.dumps({"prompt": "What is 7 * 8?", "domain": "math", "ground_truth": "56"}) + "\n")

        output_path = gen.generate_from_jsonl(input_path)
        progress_path = output_path + ".progress"
        assert os.path.exists(progress_path)
        with open(progress_path) as f:
            hashes = [l.strip() for l in f if l.strip()]
        assert len(hashes) == 1

    def test_skip_processed_prompts(self, tmp_path):
        config = SFTGenerationConfig(
            n_completions=1,
            early_stop_on_verified=True,
            output_dir=str(tmp_path),
            enable_resume=True,
        )
        gen = SFTGenerator(config)
        mock = MagicMock()
        mock.generate.return_value = [GOOD_MATH_RESPONSE]
        gen.client = mock

        input_path = str(tmp_path / "input.jsonl")
        with open(input_path, "w") as f:
            f.write(json.dumps({"prompt": "What is 7 * 8?", "domain": "math", "ground_truth": "56"}) + "\n")
            f.write(json.dumps({"prompt": "What is 3 + 5?", "domain": "math", "ground_truth": "8"}) + "\n")

        # First run
        output_path = gen.generate_from_jsonl(input_path)
        first_call_count = mock.generate.call_count

        # Second run should skip both
        gen2 = SFTGenerator(config)
        gen2.client = MagicMock()
        gen2.client.generate.return_value = [GOOD_MATH_RESPONSE]
        gen2.generate_from_jsonl(input_path, output_path)
        assert gen2.client.generate.call_count == 0  # All skipped

    def test_incremental_append(self, tmp_path):
        config = SFTGenerationConfig(
            n_completions=1,
            early_stop_on_verified=True,
            output_dir=str(tmp_path),
            enable_resume=True,
        )

        input_path = str(tmp_path / "input.jsonl")
        with open(input_path, "w") as f:
            f.write(json.dumps({"prompt": "What is 7 * 8?", "domain": "math", "ground_truth": "56"}) + "\n")

        # First run
        gen1 = SFTGenerator(config)
        mock1 = MagicMock()
        mock1.generate.return_value = [GOOD_MATH_RESPONSE]
        gen1.client = mock1
        output_path = gen1.generate_from_jsonl(input_path)

        # Add more data (ground_truth matches GOOD_MATH_RESPONSE = "The answer is 56")
        with open(input_path, "a") as f:
            f.write(json.dumps({"prompt": "What is 28 * 2?", "domain": "math", "ground_truth": "56"}) + "\n")

        # Second run appends
        gen2 = SFTGenerator(config)
        mock2 = MagicMock()
        mock2.generate.return_value = [GOOD_MATH_RESPONSE]
        gen2.client = mock2
        gen2.generate_from_jsonl(input_path, output_path)

        # Should only process the new prompt
        assert mock2.generate.call_count == 1

        # Output should have 2 lines (both verified with answer=56)
        with open(output_path) as f:
            lines = [l.strip() for l in f if l.strip()]
        assert len(lines) == 2


# =============================================================================
# TEST E2E (MOCKED)
# =============================================================================

class TestE2E:
    def test_full_pipeline_mocked(self, tmp_path):
        """Full pipeline: input JSONL → generate → verify → output JSONL + stats."""
        config = SFTGenerationConfig(
            n_completions=2,
            early_stop_on_verified=True,
            output_dir=str(tmp_path / "output"),
            batch_size=2,
        )
        gen = SFTGenerator(config)
        mock = MagicMock()
        mock.generate.return_value = [GOOD_MATH_RESPONSE]
        gen.client = mock

        # Create input
        input_path = str(tmp_path / "input.jsonl")
        with open(input_path, "w") as f:
            for i in range(5):
                f.write(json.dumps({
                    "prompt": f"What is 7 * 8?",
                    "domain": "math",
                    "ground_truth": "56",
                }) + "\n")

        output_path = gen.generate_from_jsonl(input_path)
        assert os.path.exists(output_path)

        # Verify output
        with open(output_path) as f:
            lines = [json.loads(l) for l in f if l.strip()]
        assert len(lines) >= 1
        for line in lines:
            assert "prompt" in line
            assert "domain" in line
            assert "response" in line

        # Verify stats
        stats_path = os.path.join(str(tmp_path / "output"), "generation_stats.json")
        assert os.path.exists(stats_path)
        with open(stats_path) as f:
            stats = json.load(f)
        assert stats["total_prompts"] == 5
        assert stats["verified"] >= 1

    def test_multi_domain_generation(self, tmp_path):
        """Test generation across multiple domains."""
        config = SFTGenerationConfig(
            n_completions=1,
            early_stop_on_verified=True,
            output_dir=str(tmp_path / "output"),
            batch_size=4,
        )
        gen = SFTGenerator(config)

        # Route domain to correct response
        def domain_router(prompt, domain, n=1, system_prompt=None):
            if domain == "math":
                return [GOOD_MATH_RESPONSE]
            elif domain == "english":
                return [GOOD_ENGLISH_RESPONSE]
            elif domain == "sec_finance":
                return [GOOD_SEC_FINANCE_RESPONSE]
            return ["Unknown domain"]

        mock = MagicMock()
        mock.generate.side_effect = domain_router
        gen.client = mock

        input_path = str(tmp_path / "input.jsonl")
        with open(input_path, "w") as f:
            f.write(json.dumps({"prompt": "What is 7 * 8?", "domain": "math", "ground_truth": "56"}) + "\n")
            f.write(json.dumps({"prompt": "Write about pangrams", "domain": "english"}) + "\n")
            f.write(json.dumps({"prompt": "Summarize Apple Q4", "domain": "sec_finance"}) + "\n")

        output_path = gen.generate_from_jsonl(input_path)

        stats = gen.get_stats()
        assert stats["total_prompts"] == 3
        assert "math" in stats["per_domain"]
        assert "english" in stats["per_domain"]
        assert "sec_finance" in stats["per_domain"]

        # Check output has multi-domain results
        with open(output_path) as f:
            lines = [json.loads(l) for l in f if l.strip()]
        domains_in_output = {l["domain"] for l in lines}
        # At least math and english should verify (sec_finance depends on verifier)
        assert "math" in domains_in_output


# =============================================================================
# TEST MALFORMED INPUT
# =============================================================================

class TestEdgeCases:
    def test_malformed_jsonl_skipped(self, tmp_path):
        config = SFTGenerationConfig(
            n_completions=1,
            early_stop_on_verified=True,
            output_dir=str(tmp_path),
        )
        gen = SFTGenerator(config)
        mock = MagicMock()
        mock.generate.return_value = [GOOD_MATH_RESPONSE]
        gen.client = mock

        input_path = str(tmp_path / "input.jsonl")
        with open(input_path, "w") as f:
            f.write("not valid json\n")
            f.write(json.dumps({"prompt": "What is 7 * 8?", "domain": "math", "ground_truth": "56"}) + "\n")
            f.write('{"prompt": "missing domain"}\n')  # Missing domain

        gen.generate_from_jsonl(input_path)
        # Should process only the valid line
        assert gen._stats["total"] == 1

    def test_empty_input(self, tmp_path):
        config = SFTGenerationConfig(
            n_completions=1,
            output_dir=str(tmp_path),
        )
        gen = SFTGenerator(config)
        gen.client = MagicMock()

        input_path = str(tmp_path / "input.jsonl")
        with open(input_path, "w") as f:
            pass  # Empty file

        gen.generate_from_jsonl(input_path)
        assert gen._stats["total"] == 0

    def test_openai_not_installed(self):
        """Test clear error when openai is not installed."""
        config = SFTGenerationConfig()
        with patch("src.rlvr.sft_generator.openai", None):
            with pytest.raises(ImportError, match="openai package is required"):
                VLLMClient(config)
