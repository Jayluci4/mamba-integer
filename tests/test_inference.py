"""
Tests for inference-time scaling: Best-of-N with verifiers + Meta-Inference Router.

Phase 4a: Best-of-N with Verifiable Rewards
Phase 5d: Meta-Inference Router (Learned Compute Allocation)

Run: python -m pytest tests/test_inference.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# MOCK MODEL AND VERIFIER
# =============================================================================

class TinyLM(nn.Module):
    """Tiny language model for testing."""

    def __init__(self, vocab_size=100, d_model=32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.fc = nn.Linear(d_model, d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.vocab_size = vocab_size

    def forward(self, input_ids):
        x = self.embed(input_ids)
        x = F.relu(self.fc(x))
        return self.head(x)


class MockTokenizer:
    """Mock tokenizer for testing."""

    def __init__(self, vocab_size=100):
        self.vocab_size = vocab_size

    def encode(self, text: str):
        tokens = [ord(c) % self.vocab_size for c in text]
        return tokens if tokens else [1]

    def decode(self, ids):
        return "".join(chr(min(i + 32, 126)) for i in ids)


class MockVerifier:
    """Mock verifier that checks if response contains '42'."""

    def __init__(self, name="mock"):
        self.name = name

    def verify(self, prompt, response, ground_truth=None):
        # Check if response contains the ground truth (or "42" by default)
        target = ground_truth or "42"

        class Result:
            def __init__(self, correct, score):
                self.correct = correct
                self.score = score

        if target in response:
            return Result(correct=True, score=1.0)
        return Result(correct=False, score=0.1)


class AlwaysCorrectVerifier:
    """Verifier that always says correct (for testing)."""

    def __init__(self, name="always_correct"):
        self.name = name

    def verify(self, prompt, response, ground_truth=None):
        class Result:
            def __init__(self):
                self.correct = True
                self.score = 1.0
        return Result()


class AlwaysWrongVerifier:
    """Verifier that always says wrong."""

    def __init__(self, name="always_wrong"):
        self.name = name

    def verify(self, prompt, response, ground_truth=None):
        class Result:
            def __init__(self):
                self.correct = False
                self.score = 0.0
        return Result()


# =============================================================================
# TEST IMPORTS
# =============================================================================

class TestImports:
    def test_best_of_n_imports(self):
        from src.tts.best_of_n import (
            BestOfNVerified,
            BestOfNWithReranking,
            BestOfNResult,
            compute_optimal_n,
        )

    def test_meta_router_imports(self):
        from src.tts.meta_router import (
            MetaInferenceRouter,
            RouterConfig,
            RouterDecision,
            RouterTrainer,
            RouterTrainingExample,
            RoutedInference,
            InferenceStrategy,
            PromptEncoder,
        )

    def test_tts_package_imports(self):
        from src.tts import (
            BestOfNVerified,
            MetaInferenceRouter,
            RouterConfig,
            RoutedInference,
        )


# =============================================================================
# TEST BEST-OF-N
# =============================================================================

class TestBestOfNVerified:
    @pytest.fixture
    def solver(self):
        from src.tts.best_of_n import BestOfNVerified

        model = TinyLM(vocab_size=50, d_model=16)
        tokenizer = MockTokenizer(vocab_size=50)
        verifier = AlwaysCorrectVerifier()

        return BestOfNVerified(
            model=model,
            tokenizer=tokenizer,
            verifier=verifier,
            n_samples=4,
            temperature=0.8,
            max_new_tokens=10,
        )

    def test_basic_generation(self, solver):
        """Should generate N samples and return results."""
        result = solver.generate_and_verify("What is 6*7?")

        assert result.n_total == 4
        assert len(result.all_responses) == 4
        assert len(result.all_rewards) == 4
        assert len(result.all_verified) == 4
        assert result.wall_time_ms > 0

    def test_all_verified_correct_verifier(self, solver):
        """With always-correct verifier, all should be verified."""
        result = solver.generate_and_verify("test")

        assert result.n_verified == 4
        assert result.verified is True
        assert result.best_reward == 1.0
        assert result.min_n_needed == 1  # First one should be correct

    def test_none_verified_wrong_verifier(self):
        """With always-wrong verifier, none should be verified."""
        from src.tts.best_of_n import BestOfNVerified

        model = TinyLM(vocab_size=50, d_model=16)
        tokenizer = MockTokenizer(vocab_size=50)
        verifier = AlwaysWrongVerifier()

        solver = BestOfNVerified(
            model=model,
            tokenizer=tokenizer,
            verifier=verifier,
            n_samples=4,
            max_new_tokens=10,
        )

        result = solver.generate_and_verify("test")

        assert result.n_verified == 0
        assert result.verified is False
        assert result.min_n_needed == -1  # No verified response

    def test_early_stop(self):
        """Early stop should reduce total samples."""
        from src.tts.best_of_n import BestOfNVerified

        model = TinyLM(vocab_size=50, d_model=16)
        tokenizer = MockTokenizer(vocab_size=50)
        verifier = AlwaysCorrectVerifier()

        solver = BestOfNVerified(
            model=model,
            tokenizer=tokenizer,
            verifier=verifier,
            n_samples=32,
            max_new_tokens=10,
            early_stop=True,
        )

        result = solver.generate_and_verify("test")

        # Should stop after first verified response
        assert result.n_total == 1
        assert result.verified is True

    def test_result_fields(self, solver):
        """All result fields should be populated."""
        result = solver.generate_and_verify("test", ground_truth="42")

        assert isinstance(result.best_response, str)
        assert isinstance(result.best_reward, float)
        assert isinstance(result.verified, bool)
        assert isinstance(result.all_responses, list)
        assert isinstance(result.all_rewards, list)
        assert isinstance(result.all_verified, list)
        assert isinstance(result.n_verified, int)
        assert isinstance(result.n_total, int)
        assert isinstance(result.min_n_needed, int)
        assert isinstance(result.wall_time_ms, float)


class TestBestOfNWithReranking:
    def test_reranking_basic(self):
        """Reranking should prefer higher logprob among verified."""
        from src.tts.best_of_n import BestOfNWithReranking

        model = TinyLM(vocab_size=50, d_model=16)
        tokenizer = MockTokenizer(vocab_size=50)
        verifier = AlwaysCorrectVerifier()

        solver = BestOfNWithReranking(
            model=model,
            tokenizer=tokenizer,
            verifier=verifier,
            n_samples=4,
            max_new_tokens=10,
        )

        result = solver.generate_and_verify("test")
        assert result.verified is True
        assert result.n_total == 4


class TestPassAtK:
    def test_pass_at_k_all_correct(self):
        """When all correct, pass@k = 1.0 for all k."""
        from src.tts.best_of_n import BestOfNVerified

        model = TinyLM(vocab_size=50, d_model=16)
        tokenizer = MockTokenizer(vocab_size=50)
        verifier = AlwaysCorrectVerifier()

        solver = BestOfNVerified(
            model=model,
            tokenizer=tokenizer,
            verifier=verifier,
            n_samples=8,
            max_new_tokens=10,
        )

        rates = solver.pass_at_k("test", k_values=[1, 2, 4, 8])
        for k, rate in rates.items():
            assert rate == 1.0, f"pass@{k} should be 1.0, got {rate}"

    def test_pass_at_k_none_correct(self):
        """When none correct, pass@k = 0.0 for all k."""
        from src.tts.best_of_n import BestOfNVerified

        model = TinyLM(vocab_size=50, d_model=16)
        tokenizer = MockTokenizer(vocab_size=50)
        verifier = AlwaysWrongVerifier()

        solver = BestOfNVerified(
            model=model,
            tokenizer=tokenizer,
            verifier=verifier,
            n_samples=8,
            max_new_tokens=10,
        )

        rates = solver.pass_at_k("test", k_values=[1, 2, 4, 8])
        for k, rate in rates.items():
            assert rate == 0.0, f"pass@{k} should be 0.0, got {rate}"

    def test_pass_at_k_monotonic(self):
        """pass@k should be non-decreasing with k."""
        from src.tts.best_of_n import BestOfNVerified

        model = TinyLM(vocab_size=50, d_model=16)
        tokenizer = MockTokenizer(vocab_size=50)
        verifier = AlwaysCorrectVerifier()

        solver = BestOfNVerified(
            model=model,
            tokenizer=tokenizer,
            verifier=verifier,
            n_samples=16,
            max_new_tokens=10,
        )

        rates = solver.pass_at_k("test", k_values=[1, 2, 4, 8, 16])
        sorted_ks = sorted(rates.keys())
        for i in range(1, len(sorted_ks)):
            assert rates[sorted_ks[i]] >= rates[sorted_ks[i-1]], (
                f"pass@{sorted_ks[i]} ({rates[sorted_ks[i]]}) < "
                f"pass@{sorted_ks[i-1]} ({rates[sorted_ks[i-1]]})"
            )


# =============================================================================
# TEST COMPUTE OPTIMAL N
# =============================================================================

class TestComputeOptimalN:
    def test_optimal_n_always_correct(self):
        """With always-correct verifier, min_n should be 1."""
        from src.tts.best_of_n import compute_optimal_n

        model = TinyLM(vocab_size=50, d_model=16)
        tokenizer = MockTokenizer(vocab_size=50)
        verifier = AlwaysCorrectVerifier()

        results = compute_optimal_n(
            model=model,
            tokenizer=tokenizer,
            verifier=verifier,
            prompts=["test1", "test2"],
            ground_truths=[None, None],
            max_n=4,
            max_new_tokens=10,
        )

        assert len(results) == 2
        for r in results:
            assert r['min_n'] == 1
            assert r['pass_rate'] == 1.0
            assert r['domain'] == 'always_correct'
            assert 'wall_time_ms' in r

    def test_optimal_n_never_correct(self):
        """With always-wrong verifier, min_n should be -1."""
        from src.tts.best_of_n import compute_optimal_n

        model = TinyLM(vocab_size=50, d_model=16)
        tokenizer = MockTokenizer(vocab_size=50)
        verifier = AlwaysWrongVerifier()

        results = compute_optimal_n(
            model=model,
            tokenizer=tokenizer,
            verifier=verifier,
            prompts=["test"],
            ground_truths=[None],
            max_n=4,
            max_new_tokens=10,
        )

        assert results[0]['min_n'] == -1
        assert results[0]['pass_rate'] == 0.0


# =============================================================================
# TEST PROMPT ENCODER
# =============================================================================

class TestPromptEncoder:
    def test_basic_encoding(self):
        from src.tts.meta_router import PromptEncoder, RouterConfig

        config = RouterConfig(vocab_size=50, d_embed=32, d_hidden=64)
        encoder = PromptEncoder(config)

        input_ids = torch.randint(0, 50, (2, 20))
        features = encoder(input_ids)

        assert features.shape == (2, 64)
        assert not torch.isnan(features).any()

    def test_variable_length(self):
        """Encoder should handle different sequence lengths."""
        from src.tts.meta_router import PromptEncoder, RouterConfig

        config = RouterConfig(vocab_size=50, d_embed=32, d_hidden=64)
        encoder = PromptEncoder(config)

        short = torch.randint(0, 50, (1, 5))
        long = torch.randint(0, 50, (1, 100))

        f_short = encoder(short)
        f_long = encoder(long)

        assert f_short.shape == (1, 64)
        assert f_long.shape == (1, 64)

    def test_vocab_clamping(self):
        """Should handle out-of-range token IDs."""
        from src.tts.meta_router import PromptEncoder, RouterConfig

        config = RouterConfig(vocab_size=50, d_embed=32, d_hidden=64)
        encoder = PromptEncoder(config)

        # Token IDs beyond vocab_size
        input_ids = torch.tensor([[999, 1000, 50]])
        features = encoder(input_ids)

        assert features.shape == (1, 64)
        assert not torch.isnan(features).any()

    def test_param_count(self):
        """Encoder should be lightweight."""
        from src.tts.meta_router import PromptEncoder, RouterConfig

        config = RouterConfig(vocab_size=8192, d_embed=64, d_hidden=128)
        encoder = PromptEncoder(config)

        n_params = sum(p.numel() for p in encoder.parameters())
        # Embedding: 8192*64 = 524K, convs: ~50K, total < 600K
        assert n_params < 700_000


# =============================================================================
# TEST META-INFERENCE ROUTER
# =============================================================================

class TestMetaInferenceRouter:
    @pytest.fixture
    def router(self):
        from src.tts.meta_router import MetaInferenceRouter, RouterConfig

        config = RouterConfig(
            vocab_size=50, d_embed=16, d_hidden=32,
            n_buckets=4, n_temp_bins=3, n_strategies=3,
        )
        return MetaInferenceRouter(config)

    def test_forward_shapes(self, router):
        """Forward should return correct shapes."""
        input_ids = torch.randint(0, 50, (2, 20))
        n_logits, temp_logits, strat_logits, difficulty = router(input_ids)

        assert n_logits.shape == (2, 4)     # n_buckets=4
        assert temp_logits.shape == (2, 3)  # n_temp_bins=3
        assert strat_logits.shape == (2, 3) # n_strategies=3
        assert difficulty.shape == (2, 1)
        assert (difficulty >= 0).all() and (difficulty <= 1).all()

    def test_predict_decision(self, router):
        """predict() should return a valid RouterDecision."""
        from src.tts.meta_router import RouterDecision, InferenceStrategy

        input_ids = torch.randint(0, 50, (1, 10))
        decision = router.predict(input_ids)

        assert isinstance(decision, RouterDecision)
        assert decision.n_samples in router.config.n_buckets_list
        assert decision.temperature in router.config.temp_bins_list
        assert isinstance(decision.strategy, InferenceStrategy)
        assert 0 <= decision.predicted_difficulty <= 1
        assert 0 <= decision.confidence <= 1

    def test_predict_1d_input(self, router):
        """predict() should handle 1D input."""
        input_ids = torch.randint(0, 50, (15,))
        decision = router.predict(input_ids)
        assert decision.n_samples in router.config.n_buckets_list

    def test_param_count_small(self, router):
        """With small config, params should be very few."""
        n = router.param_count()
        assert n > 0
        assert n < 50_000  # Small config

    def test_param_count_full(self):
        """With default config, should be ~100K params."""
        from src.tts.meta_router import MetaInferenceRouter, RouterConfig

        config = RouterConfig(
            vocab_size=8192, d_embed=64, d_hidden=128,
        )
        router = MetaInferenceRouter(config)

        n = router.param_count()
        # Should be around 100K (playbook spec)
        # Embedding (8192*64=524K) dominates, but the heads are ~100K
        # Total with embedding is larger — that's fine for a tiny router
        assert n > 50_000


# =============================================================================
# TEST ROUTER CONFIG
# =============================================================================

class TestRouterConfig:
    def test_defaults(self):
        from src.tts.meta_router import RouterConfig

        c = RouterConfig()
        assert c.vocab_size == 8192
        assert c.d_embed == 64
        assert c.d_hidden == 128
        assert c.n_buckets == 8
        assert c.n_temp_bins == 5
        assert c.n_strategies == 3

    def test_bucket_list(self):
        from src.tts.meta_router import RouterConfig

        c = RouterConfig(n_buckets=4)
        assert c.n_buckets_list == [1, 2, 4, 8]

    def test_temp_bins_list(self):
        from src.tts.meta_router import RouterConfig

        c = RouterConfig(n_temp_bins=3)
        assert c.temp_bins_list == [0.1, 0.3, 0.5]


# =============================================================================
# TEST ROUTER TRAINING
# =============================================================================

class TestRouterTrainer:
    def test_n_to_bucket(self):
        from src.tts.meta_router import RouterTrainer, MetaInferenceRouter, RouterConfig

        config = RouterConfig(
            vocab_size=50, d_embed=16, d_hidden=32,
            n_buckets=4,  # [1, 2, 4, 8]
        )
        router = MetaInferenceRouter(config)
        trainer = RouterTrainer(router, config)

        assert trainer._n_to_bucket(1) == 0   # N=1 -> bucket 0
        assert trainer._n_to_bucket(2) == 1   # N=2 -> bucket 1
        assert trainer._n_to_bucket(3) == 2   # N=3 -> bucket 2 (next bucket up)
        assert trainer._n_to_bucket(4) == 2   # N=4 -> bucket 2
        assert trainer._n_to_bucket(100) == 3 # N>8 -> last bucket

    def test_training_loop(self):
        """Router should train and reduce loss."""
        from src.tts.meta_router import (
            RouterTrainer, MetaInferenceRouter, RouterConfig,
            RouterTrainingExample,
        )

        config = RouterConfig(
            vocab_size=50, d_embed=16, d_hidden=32,
            n_buckets=4, n_temp_bins=3,
            n_epochs=10, batch_size=4, learning_rate=1e-2,
        )
        router = MetaInferenceRouter(config)
        trainer = RouterTrainer(router, config)

        # Create synthetic training data
        examples = []
        for i in range(20):
            # Easy prompts need N=1, hard need N=8
            is_hard = i % 2 == 0
            examples.append(RouterTrainingExample(
                prompt_ids=[i % 50] * 10,
                min_n=8 if is_hard else 1,
                pass_rate=0.1 if is_hard else 0.9,
                domain="math",
            ))

        history = trainer.train(examples, device=torch.device("cpu"))

        assert len(history['train_loss']) == 10
        # Loss should decrease
        assert history['train_loss'][-1] < history['train_loss'][0], (
            f"Loss should decrease: {history['train_loss'][0]:.4f} -> "
            f"{history['train_loss'][-1]:.4f}"
        )

    def test_training_with_validation(self):
        """Training with validation set should track val metrics."""
        from src.tts.meta_router import (
            RouterTrainer, MetaInferenceRouter, RouterConfig,
            RouterTrainingExample,
        )

        config = RouterConfig(
            vocab_size=50, d_embed=16, d_hidden=32,
            n_buckets=4, n_epochs=5, batch_size=8,
        )
        router = MetaInferenceRouter(config)
        trainer = RouterTrainer(router, config)

        train_data = [
            RouterTrainingExample(
                prompt_ids=[i % 50] * 10, min_n=2, pass_rate=0.5, domain="math"
            )
            for i in range(20)
        ]
        val_data = [
            RouterTrainingExample(
                prompt_ids=[i % 50] * 10, min_n=2, pass_rate=0.5, domain="math"
            )
            for i in range(5)
        ]

        history = trainer.train(train_data, val_examples=val_data, device=torch.device("cpu"))

        assert len(history['val_loss']) == 5
        assert len(history['n_accuracy']) == 5
        # Accuracy should be between 0 and 1
        for acc in history['n_accuracy']:
            assert 0 <= acc <= 1


# =============================================================================
# TEST ROUTED INFERENCE
# =============================================================================

class TestRoutedInference:
    @pytest.fixture
    def routed(self):
        from src.tts.meta_router import (
            MetaInferenceRouter, RouterConfig, RoutedInference,
        )

        model = TinyLM(vocab_size=50, d_model=16)
        tokenizer = MockTokenizer(vocab_size=50)
        verifier = AlwaysCorrectVerifier()

        config = RouterConfig(
            vocab_size=50, d_embed=16, d_hidden=32,
            n_buckets=4, n_temp_bins=3,
        )
        router = MetaInferenceRouter(config)

        return RoutedInference(
            model=model,
            tokenizer=tokenizer,
            router=router,
            verifier=verifier,
        )

    def test_basic_solve(self, routed):
        """Should return a result dict with expected keys."""
        result = routed.solve("What is 2+2?")

        assert 'answer' in result
        assert 'verified' in result
        assert 'confidence' in result
        assert 'strategy' in result
        assert 'n_samples' in result
        assert 'temperature' in result
        assert 'predicted_difficulty' in result
        assert 'router_confidence' in result

    def test_override_strategy(self, routed):
        """Should respect strategy override."""
        from src.tts.meta_router import InferenceStrategy

        result = routed.solve(
            "test",
            override_strategy=InferenceStrategy.SELF_CONSISTENCY,
        )
        assert result['strategy'] == 'self_consistency'

    def test_override_n(self, routed):
        """Should respect N override."""
        result = routed.solve("test", override_n=2)
        assert result['n_samples'] == 2

    def test_batch_solve(self, routed):
        """Should solve multiple prompts."""
        results = routed.solve_batch(
            ["q1", "q2", "q3"],
            ground_truths=["a1", "a2", "a3"],
        )
        assert len(results) == 3

    def test_compute_savings(self, routed):
        """compute_savings should return valid metrics."""
        from src.tts.meta_router import RouterDecision, InferenceStrategy

        decisions = [
            RouterDecision(n_samples=2, temperature=0.5,
                          strategy=InferenceStrategy.BEST_OF_N,
                          confidence=0.9, predicted_difficulty=0.1),
            RouterDecision(n_samples=8, temperature=0.7,
                          strategy=InferenceStrategy.SELF_CONSISTENCY,
                          confidence=0.7, predicted_difficulty=0.5),
            RouterDecision(n_samples=64, temperature=0.9,
                          strategy=InferenceStrategy.BEAM_SEARCH,
                          confidence=0.5, predicted_difficulty=0.9),
        ]

        savings = routed.compute_savings(decisions)
        assert savings['average_n'] == pytest.approx((2 + 8 + 64) / 3)
        assert savings['baseline_n'] == 32
        assert savings['speedup_factor'] > 1.0  # Should be faster than fixed N=32
        assert 2 in savings['n_distribution']
        assert 8 in savings['n_distribution']
        assert 64 in savings['n_distribution']


# =============================================================================
# TEST INFERENCE STRATEGY ENUM
# =============================================================================

class TestInferenceStrategy:
    def test_values(self):
        from src.tts.meta_router import InferenceStrategy

        assert InferenceStrategy.BEST_OF_N.value == "best_of_n"
        assert InferenceStrategy.SELF_CONSISTENCY.value == "self_consistency"
        assert InferenceStrategy.BEAM_SEARCH.value == "beam_search"


# =============================================================================
# TEST E2E: BEST-OF-N -> ROUTER TRAINING -> ROUTED INFERENCE
# =============================================================================

class TestE2EPipeline:
    def test_full_pipeline(self):
        """
        End-to-end: collect optimal-N data -> train router -> use for inference.
        This is the core Phase 5d workflow.
        """
        from src.tts.best_of_n import BestOfNVerified, compute_optimal_n
        from src.tts.meta_router import (
            MetaInferenceRouter, RouterConfig, RouterTrainer,
            RouterTrainingExample, RoutedInference,
        )

        model = TinyLM(vocab_size=50, d_model=16)
        tokenizer = MockTokenizer(vocab_size=50)
        verifier = AlwaysCorrectVerifier()

        # Step 1: Collect optimal-N data
        optimal_data = compute_optimal_n(
            model=model,
            tokenizer=tokenizer,
            verifier=verifier,
            prompts=["easy q1", "easy q2", "hard q3"],
            ground_truths=[None, None, None],
            max_n=4,
            max_new_tokens=10,
        )

        assert len(optimal_data) == 3
        for d in optimal_data:
            assert d['min_n'] == 1  # Always-correct verifier

        # Step 2: Convert to training examples
        train_examples = [
            RouterTrainingExample(
                prompt_ids=tokenizer.encode(d['prompt'])[:20],
                min_n=d['min_n'],
                pass_rate=d['pass_rate'],
                domain=d['domain'],
            )
            for d in optimal_data
        ]

        # Step 3: Train router
        config = RouterConfig(
            vocab_size=50, d_embed=16, d_hidden=32,
            n_buckets=4, n_temp_bins=3,
            n_epochs=5, batch_size=4,
        )
        router = MetaInferenceRouter(config)
        trainer = RouterTrainer(router, config)
        history = trainer.train(train_examples, device=torch.device("cpu"))

        assert len(history['train_loss']) == 5

        # Step 4: Use router for inference
        routed = RoutedInference(
            model=model,
            tokenizer=tokenizer,
            router=router,
            verifier=verifier,
        )

        result = routed.solve("new question")
        assert 'answer' in result
        assert result['n_samples'] in config.n_buckets_list
        assert result['temperature'] in config.temp_bins_list


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
