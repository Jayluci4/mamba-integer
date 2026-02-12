"""
Test-Time Scaling (TTS) Module for Mamba-Integer

Implements inference-time compute scaling strategies:
- Self-consistency: Majority voting over multiple samples
- Beam search with implicit PRM: Step-level scoring via log-prob ratios
- Adaptive strategy: Select method based on problem difficulty
- Best-of-N with verifiable rewards: Formal verification of completions
- Meta-inference router: Learned per-prompt compute allocation

Reference: DeepSeek-R1, Snell et al. "Scaling LLM Test-Time Compute"
"""

from .self_consistency import SelfConsistency
from .beam_search import BeamSearchPRM
from .adaptive import AdaptiveStrategy, DifficultyEstimator
from .best_of_n import BestOfNVerified, BestOfNWithReranking, BestOfNResult, compute_optimal_n
from .meta_router import (
    MetaInferenceRouter,
    RouterConfig,
    RouterDecision,
    RouterTrainer,
    RouterTrainingExample,
    RoutedInference,
    InferenceStrategy,
    PromptEncoder,
)

__all__ = [
    "SelfConsistency",
    "BeamSearchPRM",
    "AdaptiveStrategy",
    "DifficultyEstimator",
    "BestOfNVerified",
    "BestOfNWithReranking",
    "BestOfNResult",
    "compute_optimal_n",
    "MetaInferenceRouter",
    "RouterConfig",
    "RouterDecision",
    "RouterTrainer",
    "RouterTrainingExample",
    "RoutedInference",
    "InferenceStrategy",
    "PromptEncoder",
]
