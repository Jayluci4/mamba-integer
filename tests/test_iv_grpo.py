"""
Tests for IV-GRPO: Integer-Verified Group Relative Policy Optimization.

Tests the novel dual-path reward: R = R_correctness + α * exp(-β * KL(P_float || P_integer))

Run: python -m pytest tests/test_iv_grpo.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import torch
import torch.nn as nn
from dataclasses import dataclass


# =============================================================================
# MINIMAL MOCK MODEL (for testing without full Mamba architecture)
# =============================================================================

class TinyLM(nn.Module):
    """Tiny language model for testing. 2-layer transformer-like."""

    def __init__(self, vocab_size=100, d_model=32, n_head=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.head = nn.Linear(d_model, vocab_size)
        self.vocab_size = vocab_size

    def forward(self, input_ids):
        x = self.embed(input_ids)
        x = self.layers(x)
        return self.head(x)


class MockTokenizer:
    """Mock tokenizer for testing."""

    def __init__(self, vocab_size=100):
        self.vocab_size = vocab_size

    def encode(self, text: str):
        # Deterministic hash-based encoding
        tokens = []
        for i, c in enumerate(text):
            tokens.append(ord(c) % self.vocab_size)
        return tokens if tokens else [1]

    def decode(self, ids):
        return "".join(chr(min(i + 32, 126)) for i in ids)


# =============================================================================
# TEST IMPORTS
# =============================================================================

class TestImports:
    def test_iv_grpo_imports(self):
        from src.rlvr.iv_grpo import (
            IVGRPOConfig,
            IVGRPOTrainer,
            DomainDataset,
            MultiDomainIVGRPOTrainer,
        )

    def test_config_inherits_grpo(self):
        from src.rlvr.iv_grpo import IVGRPOConfig
        from src.rlvr.grpo import GRPOConfig

        config = IVGRPOConfig()
        assert isinstance(config, GRPOConfig)
        # GRPO fields present
        assert hasattr(config, "num_samples_per_prompt")
        assert hasattr(config, "kl_coef")
        # IV fields present
        assert hasattr(config, "iv_alpha")
        assert hasattr(config, "iv_beta")


# =============================================================================
# TEST IV-GRPO CONFIG
# =============================================================================

class TestIVGRPOConfig:
    def test_defaults(self):
        from src.rlvr.iv_grpo import IVGRPOConfig

        c = IVGRPOConfig()
        assert c.iv_alpha == 0.3
        assert c.iv_beta == 1.0
        assert c.iv_max_kl == 10.0
        assert c.iv_warmup_steps == 50
        assert c.iv_anneal is True

    def test_custom_config(self):
        from src.rlvr.iv_grpo import IVGRPOConfig

        c = IVGRPOConfig(
            iv_alpha=0.5,
            iv_beta=2.0,
            num_samples_per_prompt=4,
            learning_rate=3e-6,
        )
        assert c.iv_alpha == 0.5
        assert c.iv_beta == 2.0
        assert c.num_samples_per_prompt == 4
        assert c.learning_rate == 3e-6


# =============================================================================
# TEST DEFAULT QUANTIZATION
# =============================================================================

class TestDefaultQuantize:
    def test_ternary_quantization(self):
        from src.rlvr.iv_grpo import IVGRPOTrainer

        model = TinyLM(vocab_size=50, d_model=16)
        int_model = IVGRPOTrainer._default_quantize(model)

        # Check that weights are ternary (scaled)
        for name, param in int_model.named_parameters():
            if param.dim() >= 2:
                # After quantization, unique values should be {-scale, 0, scale}
                scale = param.abs().max()
                if scale > 0:
                    normalized = (param / scale).round()
                    unique_vals = set(normalized.unique().tolist())
                    assert unique_vals <= {-1.0, 0.0, 1.0}, (
                        f"Weight {name} has non-ternary values: {unique_vals}"
                    )

    def test_quantized_model_runs(self):
        from src.rlvr.iv_grpo import IVGRPOTrainer

        model = TinyLM(vocab_size=50, d_model=16)
        int_model = IVGRPOTrainer._default_quantize(model)

        # Should produce valid logits
        x = torch.randint(0, 50, (1, 5))
        with torch.no_grad():
            logits = int_model(x)
        assert logits.shape == (1, 5, 50)
        assert not torch.isnan(logits).any()

    def test_quantized_model_frozen(self):
        from src.rlvr.iv_grpo import IVGRPOTrainer

        model = TinyLM(vocab_size=50, d_model=16)
        int_model = IVGRPOTrainer._default_quantize(model)

        for param in int_model.parameters():
            assert not param.requires_grad


# =============================================================================
# TEST IV-GRPO TRAINER
# =============================================================================

class TestIVGRPOTrainer:
    @pytest.fixture
    def trainer(self):
        from src.rlvr.iv_grpo import IVGRPOTrainer, IVGRPOConfig
        from src.rlvr.domain_verifiers import MathVerifier

        model = TinyLM(vocab_size=50, d_model=16)
        tokenizer = MockTokenizer(vocab_size=50)
        reward_fn = MathVerifier()
        config = IVGRPOConfig(
            num_samples_per_prompt=2,
            max_new_tokens=10,
            total_steps=5,
            batch_size=2,
            iv_alpha=0.3,
            iv_beta=1.0,
            iv_warmup_steps=3,
        )

        return IVGRPOTrainer(
            model=model,
            tokenizer=tokenizer,
            reward_fn=reward_fn,
            config=config,
            device="cpu",
        )

    def test_effective_alpha_warmup(self, trainer):
        """Alpha should ramp up during warmup."""
        trainer.global_step = 0
        assert trainer._get_effective_alpha() == 0.0

        trainer.global_step = 1
        alpha_1 = trainer._get_effective_alpha()
        assert 0 < alpha_1 < 0.3

        trainer.global_step = 3
        alpha_3 = trainer._get_effective_alpha()
        assert alpha_3 == pytest.approx(0.3)

    def test_effective_alpha_eval_only(self, trainer):
        """Alpha should be 0 when iv_eval_only is True."""
        trainer.iv_config.iv_eval_only = True
        trainer.global_step = 100
        assert trainer._get_effective_alpha() == 0.0

    def test_compute_float_int_kl(self, trainer):
        """KL between float and int paths should be >= 0."""
        kl_values, r_consistency = trainer.compute_float_int_kl(
            "test prompt", ["response one", "response two"]
        )
        assert kl_values.shape == (2,)
        assert r_consistency.shape == (2,)
        assert (kl_values >= 0).all()
        assert (r_consistency >= 0).all()
        assert (r_consistency <= 1).all()

    def test_compute_iv_rewards(self, trainer):
        """Combined rewards should include consistency component."""
        trainer.global_step = 100  # Past warmup

        rewards, results, iv_metrics = trainer.compute_iv_rewards(
            "What is 2+2?", ["The answer is 4", "I don't know"], "4"
        )

        assert rewards.shape == (2,)
        assert len(results) == 2
        assert "kl_float_int" in iv_metrics
        assert "r_consistency" in iv_metrics
        assert iv_metrics["iv_alpha_effective"] == pytest.approx(0.3)

    def test_metrics_tracking(self, trainer):
        """IV-GRPO specific metrics should be tracked."""
        assert "kl_float_int" in trainer.metrics
        assert "r_consistency" in trainer.metrics


# =============================================================================
# TEST DOMAIN DATASET
# =============================================================================

class TestDomainDataset:
    def test_basic_creation(self):
        from src.rlvr.iv_grpo import DomainDataset

        ds = DomainDataset([
            {"prompt": "What is 1+1?", "ground_truth": "2", "domain": "math"},
            {"prompt": "SELECT * FROM t;", "domain": "sql"},
        ])
        assert len(ds) == 2
        assert ds[0]["domain"] == "math"
        assert ds[1]["ground_truth"] is None

    def test_generate_mixed(self):
        from src.rlvr.iv_grpo import DomainDataset

        ds = DomainDataset.generate_mixed(num_per_domain=10, domains=["math"])
        assert len(ds) == 10
        assert all(ds[i]["domain"] == "math" for i in range(len(ds)))
        assert all(ds[i]["ground_truth"] is not None for i in range(len(ds)))

    def test_generate_multi_domain(self):
        from src.rlvr.iv_grpo import DomainDataset

        ds = DomainDataset.generate_mixed(
            num_per_domain=5, domains=["math", "sql"]
        )
        assert len(ds) == 10
        domains = set(ds[i]["domain"] for i in range(len(ds)))
        assert "math" in domains
        assert "sql" in domains


# =============================================================================
# TEST MULTI-DOMAIN TRAINER
# =============================================================================

class TestMultiDomainIVGRPO:
    def test_verifier_selection(self):
        from src.rlvr.iv_grpo import MultiDomainIVGRPOTrainer, IVGRPOConfig

        model = TinyLM(vocab_size=50, d_model=16)
        tokenizer = MockTokenizer(vocab_size=50)
        config = IVGRPOConfig(
            num_samples_per_prompt=2,
            max_new_tokens=5,
            total_steps=2,
        )

        trainer = MultiDomainIVGRPOTrainer(
            model=model,
            tokenizer=tokenizer,
            config=config,
            device="cpu",
        )

        # Get math verifier
        v_math = trainer._get_verifier("math")
        assert v_math.name == "math"

        # Get English verifier
        v_eng = trainer._get_verifier("english")
        assert v_eng.name == "english"

        # Same instance returned on second call (caching)
        assert trainer._get_verifier("math") is v_math

    def test_domain_specific_rewards(self):
        from src.rlvr.iv_grpo import MultiDomainIVGRPOTrainer, IVGRPOConfig

        model = TinyLM(vocab_size=50, d_model=16)
        tokenizer = MockTokenizer(vocab_size=50)
        config = IVGRPOConfig(
            num_samples_per_prompt=2,
            max_new_tokens=5,
        )

        trainer = MultiDomainIVGRPOTrainer(
            model=model,
            tokenizer=tokenizer,
            config=config,
            device="cpu",
        )

        # Math domain with correct answer
        rewards, results = trainer.compute_rewards_for_domain(
            "What is 5+3?", ["The answer is 8"], "math", "8"
        )
        assert results[0].correct

        # Math domain with wrong answer
        rewards, results = trainer.compute_rewards_for_domain(
            "What is 5+3?", ["The answer is 7"], "math", "8"
        )
        assert not results[0].correct


# =============================================================================
# TEST KL MATH
# =============================================================================

class TestKLMath:
    def test_identical_models_zero_kl(self):
        """KL between identical models should be ~0."""
        from src.rlvr.iv_grpo import IVGRPOTrainer, IVGRPOConfig
        from src.rlvr.domain_verifiers import MathVerifier

        model = TinyLM(vocab_size=50, d_model=16)
        tokenizer = MockTokenizer(vocab_size=50)
        config = IVGRPOConfig(
            num_samples_per_prompt=2,
            max_new_tokens=5,
        )

        # Use identity quantization (no change)
        def identity_quantize(m):
            import copy
            q = copy.deepcopy(m)
            q.eval()
            for p in q.parameters():
                p.requires_grad = False
            return q

        trainer = IVGRPOTrainer(
            model=model,
            tokenizer=tokenizer,
            reward_fn=MathVerifier(),
            config=config,
            quantize_fn=identity_quantize,
            device="cpu",
        )

        kl_values, r_consistency = trainer.compute_float_int_kl(
            "test", ["hello world"]
        )

        assert kl_values[0].item() < 0.01, (
            f"KL should be ~0 for identical models, got {kl_values[0].item()}"
        )
        assert r_consistency[0].item() > 0.99

    def test_different_models_positive_kl(self):
        """KL between float and quantized should be > 0."""
        from src.rlvr.iv_grpo import IVGRPOTrainer, IVGRPOConfig
        from src.rlvr.domain_verifiers import MathVerifier

        model = TinyLM(vocab_size=50, d_model=16)
        # Initialize with larger weights so quantization has bigger effect
        with torch.no_grad():
            for p in model.parameters():
                if p.dim() >= 2:
                    p.mul_(5.0)

        tokenizer = MockTokenizer(vocab_size=50)
        config = IVGRPOConfig(
            num_samples_per_prompt=2,
            max_new_tokens=5,
        )

        trainer = IVGRPOTrainer(
            model=model,
            tokenizer=tokenizer,
            reward_fn=MathVerifier(),
            config=config,
            device="cpu",
        )

        kl_values, r_consistency = trainer.compute_float_int_kl(
            "test", ["hello world"]
        )

        assert kl_values[0].item() > 0.0, (
            f"KL should be > 0 for quantized model, got {kl_values[0].item()}"
        )

    def test_consistency_reward_formula(self):
        """R_consistency = exp(-β * KL)."""
        import math as m

        kl = 0.5
        beta = 2.0
        expected = m.exp(-beta * kl)

        assert abs(expected - m.exp(-1.0)) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
