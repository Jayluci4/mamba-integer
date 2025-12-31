"""
Test-Time Scaling (TTS) Module for Mamba-Integer

Implements inference-time compute scaling strategies:
- Self-consistency: Majority voting over multiple samples
- Beam search with implicit PRM: Step-level scoring via log-prob ratios
- Adaptive strategy: Select method based on problem difficulty

Reference: DeepSeek-R1, Snell et al. "Scaling LLM Test-Time Compute"
"""

from .self_consistency import SelfConsistency
from .beam_search import BeamSearchPRM
from .adaptive import AdaptiveStrategy, DifficultyEstimator

__all__ = [
    "SelfConsistency",
    "BeamSearchPRM",
    "AdaptiveStrategy",
    "DifficultyEstimator"
]
