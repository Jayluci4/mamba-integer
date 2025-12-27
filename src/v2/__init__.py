"""
Mamba-Integer V2: Surprise-Gated Multi-Timescale Memory

Key innovations:
1. Surprise-based retention: α_t = f(prediction_error)
2. Multi-timescale memory hierarchy (future)
3. Selective forgetting (future)

Two implementations:
- Sequential (surprise_ssd.py): Accurate per-token surprise, slow
- Chunked (chunked_surprise_ssd.py): Per-chunk surprise, fast (~80x speedup)

Usage:
    # Fast version (recommended for training)
    from v2 import MambaIntegerBlockV2Chunked, ChunkedSurpriseGatedSSD

    # Slow but accurate version (for analysis)
    from v2 import MambaIntegerBlockV2Surprise, SurpriseGatedSSD
"""

# Sequential (slow but accurate)
from .surprise_ssd import (
    SurpriseGatedRetention,
    SurpriseGatedSSD,
    MambaIntegerBlockV2Surprise,
    surprise_ssd_forward,
)

# Chunked (fast, recommended)
from .chunked_surprise_ssd import (
    ChunkedSurpriseGate,
    ChunkedSurpriseGatedSSD,
    MambaIntegerBlockV2Chunked,
    chunked_surprise_ssd_forward,
)

__all__ = [
    # Sequential
    'SurpriseGatedRetention',
    'SurpriseGatedSSD',
    'MambaIntegerBlockV2Surprise',
    'surprise_ssd_forward',
    # Chunked (fast)
    'ChunkedSurpriseGate',
    'ChunkedSurpriseGatedSSD',
    'MambaIntegerBlockV2Chunked',
    'chunked_surprise_ssd_forward',
]
