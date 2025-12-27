"""
Mamba-Integer V2: Surprise-Gated Multi-Timescale Memory

Key innovations:
1. Surprise-based retention: α_t = f(prediction_error)
2. Multi-timescale memory hierarchy (future)
3. Selective forgetting (future)

Usage:
    from v2 import MambaIntegerBlockV2Surprise, SurpriseGatedSSD
"""

from .surprise_ssd import (
    SurpriseGatedRetention,
    SurpriseGatedSSD,
    MambaIntegerBlockV2Surprise,
    surprise_ssd_forward,
)

__all__ = [
    'SurpriseGatedRetention',
    'SurpriseGatedSSD',
    'MambaIntegerBlockV2Surprise',
    'surprise_ssd_forward',
]
