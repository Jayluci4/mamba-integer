"""
Mamba-Integer V2: Surprise-Gated Multi-Timescale Memory

Key innovations:
1. Surprise-based retention: α_t = f(prediction_error)
2. Multi-timescale memory hierarchy (L0-L3)
3. Consolidation triggers (capacity/surprise/time)
4. Selective forgetting with synaptic homeostasis

Three implementations:
- Sequential (surprise_ssd.py): Accurate per-token surprise, slow
- Chunked (chunked_surprise_ssd.py): Per-chunk surprise, fast (~80x speedup)
- Full (full_v2_architecture.py): Complete 4-phase architecture

Usage:
    # Fast version (recommended for training)
    from v2 import MambaIntegerBlockV2Chunked

    # Full V2 with all components
    from v2 import MambaIntegerBlockV2Full, MambaIntegerModelV2Full

    # Slow but accurate version (for analysis)
    from v2 import MambaIntegerBlockV2Surprise
"""

# Sequential (slow but accurate)
from .surprise_ssd import (
    SurpriseGatedRetention,
    SurpriseGatedSSD,
    MambaIntegerBlockV2Surprise,
    surprise_ssd_forward,
)

# Chunked (fast, recommended for simple training)
from .chunked_surprise_ssd import (
    ChunkedSurpriseGate,
    ChunkedSurpriseGatedSSD,
    MambaIntegerBlockV2Chunked,
    chunked_surprise_ssd_forward,
)

# Full V2 architecture (all 4 phases)
from .full_v2_architecture import (
    # Core components
    SurpriseGate,
    MultiTimescaleMemory,
    ConsolidationTrigger,
    ImportanceScorer,
    SelectiveForgetting,
    SynapticHomeostasis,
    # Block and model
    MambaIntegerBlockV2Full,
    MambaIntegerModelV2Full,
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
    # Full V2 architecture
    'SurpriseGate',
    'MultiTimescaleMemory',
    'ConsolidationTrigger',
    'ImportanceScorer',
    'SelectiveForgetting',
    'SynapticHomeostasis',
    'MambaIntegerBlockV2Full',
    'MambaIntegerModelV2Full',
]
