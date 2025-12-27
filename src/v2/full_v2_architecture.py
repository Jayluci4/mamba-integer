"""
Mamba-Integer V2 Full Architecture

Complete implementation of the 4-phase V2 design:
1. Surprise-Based Retention (from chunked_surprise_ssd.py)
2. Multi-Timescale Memory Hierarchy (L0-L3)
3. Consolidation Triggers (capacity/surprise/time)
4. Selective Forgetting with Synaptic Homeostasis

Key innovations:
- 4-level memory hierarchy with different timescales
- Consolidation from fast to slow memory based on triggers
- Selective forgetting with importance-based protection
- Integer-only operations via BitLinear
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass

import sys
sys.path.insert(0, '/home/jayantlohia16/mamba-integer/src')

from triton_kernels.ssd_multihead import build_causal_decay_matrix_multihead

# Import BitLinear for integer-only operations
# Note: BitLinear has Triton compatibility issues in some environments
# We disable it by default and use nn.Linear for prototyping
# Set USE_BITLINEAR = True to enable after fixing Triton kernels
USE_BITLINEAR = False  # Disabled due to Triton rint compatibility

if USE_BITLINEAR:
    try:
        from rational_bitnet import BitLinear
        HAS_BITLINEAR = True
    except ImportError:
        HAS_BITLINEAR = False
else:
    HAS_BITLINEAR = False

# Fallback to nn.Linear
if not HAS_BITLINEAR:
    BitLinear = nn.Linear


# =============================================================================
# Phase 1: Surprise-Based Retention (from chunked_surprise_ssd.py)
# =============================================================================

class SurpriseGate(nn.Module):
    """
    Computes adaptive retention based on state prediction error.

    Surprise = ||h_actual - decay * h_prev||^2
    High surprise -> high retention (keep surprising info)
    Low surprise -> normal decay (predictable, ok to forget)
    """

    def __init__(
        self,
        n_heads: int,
        d_state: int,
        ema_decay: float = 0.99,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.d_state = d_state
        self.ema_decay = ema_decay

        # Learned base retention per head (initialize to ~0.9)
        self.log2_alpha_base = nn.Parameter(torch.ones(n_heads) * -0.15)

        # Learned surprise scaling factor per head
        self.log2_beta = nn.Parameter(torch.zeros(n_heads))

        # Running EMA for surprise normalization
        self.register_buffer('surprise_ema', torch.ones(n_heads))
        self.register_buffer('step_count', torch.tensor(0))

    def compute_prediction_error(
        self,
        h_actual: torch.Tensor,    # [B, n_heads, d_state, d_head]
        h_prev: torch.Tensor,      # Same shape
        decay: torch.Tensor,       # [B, n_heads]
    ) -> torch.Tensor:
        """Compute state prediction error."""
        # Expand decay for broadcasting
        decay_expanded = decay
        for _ in range(h_prev.dim() - decay.dim()):
            decay_expanded = decay_expanded.unsqueeze(-1)

        h_expected = decay_expanded * h_prev
        error = (h_actual - h_expected).pow(2)

        # Reduce to [B, n_heads]
        while error.dim() > 2:
            error = error.mean(dim=-1)

        return error

    def forward(
        self,
        prediction_error: torch.Tensor,  # [B, n_heads] or [B, n_heads, n_chunks]
        training: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute adaptive retention from prediction error.

        Returns:
            alpha: Retention coefficient (higher = more retention)
            normalized_surprise: For logging
        """
        if prediction_error.dim() == 2:
            prediction_error = prediction_error.unsqueeze(-1)

        # Update EMA
        if training:
            with torch.no_grad():
                mean_error = prediction_error.mean(dim=(0, 2))
                self.surprise_ema = self.ema_decay * self.surprise_ema + (1 - self.ema_decay) * mean_error
                self.step_count += 1

        # Normalize
        normalized = prediction_error / (self.surprise_ema.view(1, -1, 1) + 1e-6)

        # Base retention
        log2_clamped = self.log2_alpha_base.clamp(-3.32, -0.015)
        alpha_base = 1.0 - torch.pow(2.0, log2_clamped)

        # Surprise scaling
        beta = torch.pow(2.0, self.log2_beta.clamp(-2, 2))

        # Boost from surprise
        boost = torch.tanh(beta.view(1, -1, 1) * normalized)
        boost_positive = F.relu(boost)

        # Final alpha
        alpha = alpha_base.view(1, -1, 1) + (1 - alpha_base.view(1, -1, 1)) * boost_positive
        alpha = alpha.clamp(0.01, 0.999)

        return alpha, normalized


# =============================================================================
# Phase 2: Multi-Timescale Memory Hierarchy
# =============================================================================

class MultiTimescaleMemory(nn.Module):
    """
    4-level memory hierarchy with different timescales.

    Level 0: Activations (per-token, ephemeral)
    Level 1: SSM State (surprise-adaptive retention)
    Level 2: Chunk Memory (slow, high retention, τ ~ 256-1024)
    Level 3: Persistent Memory (frozen after training, τ = ∞)
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_state: int,
        d_head: int,
        n_persistent: int = 64,
        chunk_memory_size: int = 256,
        chunk_retention: float = 0.99,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_state = d_state
        self.d_head = d_head
        self.n_persistent = n_persistent
        self.chunk_memory_size = chunk_memory_size
        self.chunk_retention = chunk_retention

        # Level 2: Chunk memory (slow, high retention)
        # Stores compressed summaries of processed chunks
        self.register_buffer(
            'chunk_memory',
            torch.zeros(n_heads, chunk_memory_size, d_state)
        )
        self.register_buffer('chunk_memory_ptr', torch.tensor(0))
        self.register_buffer('chunk_memory_filled', torch.tensor(0))

        # Level 3: Persistent memory (fixed after training)
        # Stores stable knowledge patterns
        self.persistent_memory = nn.Parameter(
            torch.randn(n_heads, n_persistent, d_state) * 0.02
        )

        # Projection for reading from memory levels
        self.memory_query_proj = nn.Linear(d_model, n_heads * d_state)
        self.memory_out_proj = nn.Linear(n_heads * d_state, d_model)

        # Gating for memory integration
        self.memory_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )

    def write_chunk_summary(
        self,
        chunk_state: torch.Tensor,  # [B, n_heads, d_state, d_head]
        importance: torch.Tensor,    # [B, n_heads] importance score
    ):
        """
        Write chunk summary to L2 memory (circular buffer).
        Only writes if importance exceeds threshold.
        """
        batch = chunk_state.shape[0]

        # Compress state to [n_heads, d_state] by importance-weighted average
        # Sum over d_head dimension
        compressed = chunk_state.mean(dim=-1)  # [B, n_heads, d_state]

        # Average across batch, weighted by importance
        importance_norm = F.softmax(importance, dim=0)  # [B, n_heads]
        weighted = (compressed * importance_norm.unsqueeze(-1)).sum(dim=0)  # [n_heads, d_state]

        # Write to circular buffer with exponential moving average
        ptr = self.chunk_memory_ptr.item()
        self.chunk_memory[:, ptr] = (
            self.chunk_retention * self.chunk_memory[:, ptr] +
            (1 - self.chunk_retention) * weighted
        )

        # Update pointer
        self.chunk_memory_ptr = (self.chunk_memory_ptr + 1) % self.chunk_memory_size
        self.chunk_memory_filled = torch.clamp(
            self.chunk_memory_filled + 1,
            max=self.chunk_memory_size
        )

    def read_memory(
        self,
        query: torch.Tensor,  # [B, L, d_model]
    ) -> torch.Tensor:
        """
        Read from L2 (chunk) and L3 (persistent) memory.
        Uses attention-like mechanism to retrieve relevant memories.
        """
        batch, seqlen, _ = query.shape

        # Project query
        q = self.memory_query_proj(query)  # [B, L, n_heads * d_state]
        q = q.view(batch, seqlen, self.n_heads, self.d_state)  # [B, L, n_heads, d_state]

        # L2: Chunk memory (only filled portion)
        filled = max(1, self.chunk_memory_filled.item())
        chunk_mem = self.chunk_memory[:, :filled]  # [n_heads, filled, d_state]

        # L3: Persistent memory
        persist_mem = self.persistent_memory  # [n_heads, n_persistent, d_state]

        # Combine memories
        combined_mem = torch.cat([chunk_mem, persist_mem], dim=1)  # [n_heads, M, d_state]

        # Attention scores: q @ mem^T
        # q: [B, L, n_heads, d_state], mem: [n_heads, M, d_state]
        attn = torch.einsum('blhd,hmd->blhm', q, combined_mem)  # [B, L, n_heads, M]
        attn = attn / math.sqrt(self.d_state)
        attn = F.softmax(attn, dim=-1)

        # Retrieve values
        retrieved = torch.einsum('blhm,hmd->blhd', attn, combined_mem)  # [B, L, n_heads, d_state]
        retrieved = retrieved.reshape(batch, seqlen, -1)  # [B, L, n_heads * d_state]

        # Project back
        memory_out = self.memory_out_proj(retrieved)  # [B, L, d_model]

        return memory_out

    def integrate(
        self,
        x: torch.Tensor,           # [B, L, d_model] current representation
        memory_out: torch.Tensor,  # [B, L, d_model] retrieved memory
    ) -> torch.Tensor:
        """Gated integration of current representation with memory."""
        combined = torch.cat([x, memory_out], dim=-1)
        gate = self.memory_gate(combined)
        return x + gate * memory_out


# =============================================================================
# Phase 3: Consolidation Triggers
# =============================================================================

class ConsolidationTrigger(nn.Module):
    """
    Determines when to consolidate from fast (L1) to slow (L2) memory.

    Three trigger mechanisms:
    1. Capacity-based: Buffer fullness exceeds threshold
    2. Surprise-based: Cumulative surprise exceeds threshold
    3. Time-based: Tokens since last consolidation exceeds threshold
    """

    def __init__(
        self,
        capacity_threshold: float = 0.9,
        surprise_threshold: float = 2.0,
        time_threshold: int = 256,
    ):
        super().__init__()
        self.capacity_threshold = capacity_threshold
        self.surprise_threshold = surprise_threshold
        self.time_threshold = time_threshold

        # Track state
        self.register_buffer('cumulative_surprise', torch.tensor(0.0))
        self.register_buffer('tokens_since_consolidation', torch.tensor(0))
        self.register_buffer('consolidation_count', torch.tensor(0))

    def should_consolidate(
        self,
        surprise_score: float,
        buffer_fullness: float,
    ) -> Tuple[bool, List[str]]:
        """
        Check if consolidation should occur.

        Returns:
            should_trigger: Whether to consolidate
            triggered_by: List of trigger reasons
        """
        self.cumulative_surprise = self.cumulative_surprise + surprise_score
        self.tokens_since_consolidation = self.tokens_since_consolidation + 1

        triggers = []

        if buffer_fullness > self.capacity_threshold:
            triggers.append('capacity')

        if self.cumulative_surprise.item() > self.surprise_threshold:
            triggers.append('surprise')

        if self.tokens_since_consolidation.item() > self.time_threshold:
            triggers.append('time')

        if triggers:
            self.reset()
            self.consolidation_count = self.consolidation_count + 1
            return True, triggers

        return False, []

    def reset(self):
        """Reset triggers after consolidation."""
        self.cumulative_surprise.zero_()
        self.tokens_since_consolidation.zero_()

    def get_stats(self) -> Dict:
        return {
            'cumulative_surprise': self.cumulative_surprise.item(),
            'tokens_since_consolidation': self.tokens_since_consolidation.item(),
            'consolidation_count': self.consolidation_count.item(),
        }


class ImportanceScorer(nn.Module):
    """
    Computes importance scores for state elements.

    Used to determine what to consolidate and what to protect from forgetting.

    Methods:
    - surprise_weighted: Weight by surprise at encoding time
    - gradient_based: Use gradient magnitude (like SI/EWC)
    """

    def __init__(self, n_heads: int, d_state: int, method: str = 'surprise_weighted'):
        super().__init__()
        self.n_heads = n_heads
        self.d_state = d_state
        self.method = method

        # Running importance accumulator (for gradient-based)
        self.register_buffer('importance_accumulator', torch.zeros(n_heads, d_state))
        self.register_buffer('update_count', torch.tensor(0))

    def compute_importance(
        self,
        states: torch.Tensor,      # [B, n_heads, d_state, d_head]
        surprise: torch.Tensor,    # [B, n_heads]
        gradients: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute importance scores for state elements.

        Returns:
            importance: [B, n_heads] importance score per head
        """
        if self.method == 'surprise_weighted':
            # High surprise at encoding -> high importance
            state_magnitude = states.pow(2).mean(dim=(-1, -2))  # [B, n_heads]
            importance = state_magnitude * surprise

        elif self.method == 'gradient_based':
            # Use gradient magnitude (requires gradients to be passed)
            if gradients is not None:
                importance = gradients.pow(2).mean(dim=(-1, -2))
            else:
                importance = states.pow(2).mean(dim=(-1, -2))

        else:
            # Default: just state magnitude
            importance = states.pow(2).mean(dim=(-1, -2))

        return importance

    def update_accumulator(self, importance: torch.Tensor):
        """Update running importance (for EWC-like consolidation)."""
        with torch.no_grad():
            mean_importance = importance.mean(dim=0)  # [n_heads]
            # Expand to match accumulator shape
            if mean_importance.dim() == 1:
                mean_importance = mean_importance.unsqueeze(-1).expand(-1, self.d_state)
            self.importance_accumulator = (
                0.99 * self.importance_accumulator + 0.01 * mean_importance
            )
            self.update_count = self.update_count + 1


# =============================================================================
# Phase 4: Selective Forgetting
# =============================================================================

class SelectiveForgetting(nn.Module):
    """
    Learns WHAT to forget, not just how much.

    Key insight: Forgetting should be selective:
    - Forget noise, duplicates, outdated info
    - Keep surprising, novel, goal-relevant info

    Importance-gated: High importance overrides forget decision.
    """

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads

        # Forget gate: learns what to forget based on [state, input]
        self.forget_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.SiLU(),
            nn.Linear(d_model, n_heads),
            nn.Sigmoid()
        )

        # Learned importance threshold
        self.importance_threshold = nn.Parameter(torch.tensor(0.5))

    def forward(
        self,
        h_prev: torch.Tensor,        # [B, n_heads, d_state, d_head] previous state
        x: torch.Tensor,             # [B, L, d_model] current input
        importance: torch.Tensor,    # [B, n_heads] importance scores
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply selective forgetting to state.

        Returns:
            h_forgotten: State after selective forgetting
            forget_gate: The forget gate values for logging
        """
        batch, seqlen, _ = x.shape

        # Compute forget gate from last input token
        x_last = x[:, -1]  # [B, d_model]

        # Flatten state for gate computation
        h_flat = h_prev.mean(dim=(-1, -2))  # [B, n_heads] -> we need [B, d_model]
        # Project state to d_model
        h_proj = h_prev.reshape(batch, -1)[:, :self.d_model]  # Simple truncation
        if h_proj.shape[-1] < self.d_model:
            h_proj = F.pad(h_proj, (0, self.d_model - h_proj.shape[-1]))

        combined = torch.cat([h_proj, x_last], dim=-1)  # [B, d_model * 2]
        forget = self.forget_gate(combined)  # [B, n_heads]

        # Importance-modulated forgetting
        # High importance -> don't forget (override gate)
        importance_norm = importance / (importance.max() + 1e-6)
        importance_mask = (importance_norm > self.importance_threshold).float()

        # Effective forget: forget unless important
        effective_forget = forget * (1 - importance_mask)

        # Apply forgetting to state
        # expand for broadcasting: [B, n_heads] -> [B, n_heads, 1, 1]
        forget_expanded = effective_forget.unsqueeze(-1).unsqueeze(-1)
        h_forgotten = h_prev * (1 - forget_expanded)

        return h_forgotten, forget


class SynapticHomeostasis(nn.Module):
    """
    Periodic downscaling of unimportant state elements.

    Inspired by SHY (Synaptic Homeostasis Hypothesis):
    - During "wake": accumulate information
    - During "sleep" (periodic): downscale weak connections

    This creates room for new information and prevents runaway growth.
    """

    def __init__(
        self,
        n_heads: int,
        d_state: int,
        downscale_factor: float = 0.99,
        homeostasis_interval: int = 64,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.d_state = d_state
        self.downscale_factor = downscale_factor
        self.homeostasis_interval = homeostasis_interval

        self.register_buffer('step_counter', torch.tensor(0))
        self.register_buffer('homeostasis_count', torch.tensor(0))

    def maybe_apply(
        self,
        state: torch.Tensor,       # [B, n_heads, d_state, d_head]
        importance: torch.Tensor,  # [B, n_heads]
    ) -> torch.Tensor:
        """
        Apply homeostasis if interval has passed.

        Returns:
            state: Possibly downscaled state
        """
        self.step_counter = self.step_counter + 1

        if self.step_counter.item() % self.homeostasis_interval == 0:
            state = self.apply_homeostasis(state, importance)
            self.homeostasis_count = self.homeostasis_count + 1

        return state

    def apply_homeostasis(
        self,
        state: torch.Tensor,
        importance: torch.Tensor,
    ) -> torch.Tensor:
        """
        Scale down unimportant state elements.

        High importance -> scale = 1.0 (preserve)
        Low importance -> scale = downscale_factor (shrink)
        """
        # Normalize importance to [0, 1]
        importance_norm = importance / (importance.max() + 1e-6)

        # Scale: high importance = 1.0, low importance = downscale_factor
        scale = importance_norm + (1 - importance_norm) * self.downscale_factor

        # Expand for broadcasting
        scale_expanded = scale.unsqueeze(-1).unsqueeze(-1)

        return state * scale_expanded


# =============================================================================
# SSD Chunk Forward (same as chunked_surprise_ssd.py)
# =============================================================================

def ssd_chunk_forward(
    X_chunk: torch.Tensor,     # [B, n_heads, cs, d_head]
    A_chunk: torch.Tensor,     # [B, n_heads, cs] log decay
    B_chunk: torch.Tensor,     # [B, n_heads, cs, d_state]
    C_chunk: torch.Tensor,     # [B, n_heads, cs, d_state]
    h_prev: torch.Tensor,      # [B, n_heads, d_state, d_head]
    decay_scale: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Process a single chunk with SSD."""
    B, n_heads, cs, d_head = X_chunk.shape
    d_state = B_chunk.shape[-1]
    device = X_chunk.device
    dtype = X_chunk.dtype

    A_scaled = A_chunk * decay_scale
    A_cumsum = torch.cumsum(A_scaled, dim=-1)

    L = build_causal_decay_matrix_multihead(A_cumsum.unsqueeze(2))[:, :, 0]

    CB = torch.einsum('bhid,bhjd->bhij', C_chunk, B_chunk)
    L_CB = L * CB
    Y_intra = torch.einsum('bhij,bhjd->bhid', L_CB, X_chunk)

    decay_from_start = torch.exp(A_cumsum)
    Y_inter = torch.einsum('bhis,bhsd,bhi->bhid', C_chunk, h_prev, decay_from_start)

    Y_chunk = Y_intra + Y_inter

    decay_to_end = torch.exp(A_cumsum[:, :, -1:] - A_cumsum)
    h_chunk_contrib = torch.einsum('bhts,bhtd,bht->bhsd', B_chunk, X_chunk, decay_to_end)

    decay_total = torch.exp(A_cumsum[:, :, -1])
    h_end = decay_total.unsqueeze(-1).unsqueeze(-1) * h_prev + h_chunk_contrib

    return Y_chunk, h_end, decay_total


# =============================================================================
# Full V2 Block: Integrating All Components
# =============================================================================

class MambaIntegerBlockV2Full(nn.Module):
    """
    Complete Mamba-Integer V2 block with all four phases:

    1. Surprise-based retention
    2. Multi-timescale memory (L0-L3)
    3. Consolidation triggers
    4. Selective forgetting + synaptic homeostasis

    Uses BitLinear for integer-only operations.
    """

    def __init__(self, config: dict, layer_idx: int):
        super().__init__()

        self.config = config
        self.layer_idx = layer_idx

        d_model = config['d_model']
        ssm_cfg = config.get('ssm_cfg', {})

        self.n_heads = ssm_cfg.get('n_heads', 24)
        self.d_head = ssm_cfg.get('d_head', 64)
        self.d_state = ssm_cfg.get('d_state', 64)
        self.d_inner = self.n_heads * self.d_head
        self.chunk_size = ssm_cfg.get('chunk_size', 64)

        # Use BitLinear if available, else nn.Linear
        Linear = BitLinear if HAS_BITLINEAR else nn.Linear

        # Input projections
        self.norm = nn.RMSNorm(d_model)
        self.in_proj = Linear(d_model, self.d_inner * 2)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, 4,
            groups=self.d_inner, padding=3
        )

        # SSM projections
        proj_size = self.n_heads * (1 + 2 * self.d_state)
        self.x_proj = Linear(self.d_inner, proj_size)

        # Base decay (learned log scale)
        self.A_log = nn.Parameter(torch.log(torch.ones(self.n_heads) * 0.5))

        # Phase 1: Surprise gate
        self.surprise_gate = SurpriseGate(
            n_heads=self.n_heads,
            d_state=self.d_state,
        )

        # Phase 2: Multi-timescale memory
        self.multi_memory = MultiTimescaleMemory(
            d_model=d_model,
            n_heads=self.n_heads,
            d_state=self.d_state,
            d_head=self.d_head,
            n_persistent=ssm_cfg.get('n_persistent', 64),
            chunk_memory_size=ssm_cfg.get('chunk_memory_size', 256),
        )

        # Phase 3: Consolidation
        self.consolidation_trigger = ConsolidationTrigger(
            capacity_threshold=ssm_cfg.get('capacity_threshold', 0.9),
            surprise_threshold=ssm_cfg.get('surprise_threshold', 2.0),
            time_threshold=ssm_cfg.get('time_threshold', 256),
        )
        self.importance_scorer = ImportanceScorer(
            n_heads=self.n_heads,
            d_state=self.d_state,
        )

        # Phase 4: Selective forgetting
        self.selective_forget = SelectiveForgetting(
            d_model=d_model,
            n_heads=self.n_heads,
        )
        self.synaptic_homeostasis = SynapticHomeostasis(
            n_heads=self.n_heads,
            d_state=self.d_state,
            homeostasis_interval=ssm_cfg.get('homeostasis_interval', 64),
        )

        # Output projection
        self.out_proj = Linear(self.d_inner, d_model)

        # Residual gate
        n_layer = config.get('n_layer', 24)
        self.res_gate = nn.Parameter(torch.ones(1) / math.sqrt(2 * n_layer))

        # State tracking
        self.last_surprise = None
        self.last_importance = None

    def forward(
        self,
        hidden_states: torch.Tensor,  # [B, L, d_model]
        return_surprise: bool = False,
        return_stats: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Dict]]:
        """
        Forward pass with full V2 architecture.

        Returns:
            output: [B, L, d_model]
            surprise: Optional[Tensor] surprise values
            stats: Optional[Dict] detailed statistics
        """
        residual = hidden_states
        batch, seqlen, _ = hidden_states.shape

        # Normalize
        hidden_states = self.norm(hidden_states)

        # Project and split
        xz = self.in_proj(hidden_states)
        x, z = xz.chunk(2, dim=-1)

        # Conv
        x = self.conv1d(x.transpose(1, 2)).transpose(1, 2)[:, :seqlen]
        x = F.silu(x)

        # SSM projections
        x_proj_out = self.x_proj(x)
        x_proj_out = x_proj_out.view(batch, seqlen, self.n_heads, 1 + 2 * self.d_state)

        dt = x_proj_out[..., 0]
        B_ssm = x_proj_out[..., 1:1+self.d_state]
        C_ssm = x_proj_out[..., 1+self.d_state:]

        B_ssm = F.normalize(B_ssm, dim=-1)
        C_ssm = F.normalize(C_ssm, dim=-1)

        # Base decay
        A_log_clamped = self.A_log.clamp(-2.3, -0.01)
        A = A_log_clamped.view(1, 1, self.n_heads).expand(batch, seqlen, -1)

        dt_scale = F.softplus(dt.clamp(-10, 10)).clamp(0.01, 10.0)
        A = A * dt_scale
        A = A.clamp(-20.0, -0.001)

        X = x.view(batch, seqlen, self.n_heads, self.d_head)

        # Process with chunked SSD + surprise + memory + forgetting
        Y, surprise, importance, stats = self._full_forward(
            X, A, B_ssm, C_ssm, hidden_states
        )

        self.last_surprise = surprise
        self.last_importance = importance

        # Reshape output
        y = Y.reshape(batch, seqlen, self.d_inner)
        y = y.clamp(-100, 100)
        y = y * torch.sigmoid(z)

        # Output projection
        out = self.out_proj(y)
        out = out.clamp(-100, 100)

        # Residual connection
        output = residual + out * self.res_gate

        if return_stats:
            return output, surprise, stats
        elif return_surprise:
            return output, surprise
        return output, None

    def _full_forward(
        self,
        X: torch.Tensor,      # [B, L, n_heads, d_head]
        A: torch.Tensor,      # [B, L, n_heads]
        B: torch.Tensor,      # [B, L, n_heads, d_state]
        C: torch.Tensor,      # [B, L, n_heads, d_state]
        hidden_states: torch.Tensor,  # [B, L, d_model] for memory
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
        """
        Full V2 forward with all components.
        """
        batch, seqlen, n_heads, d_head = X.shape
        d_state = B.shape[-1]
        device = X.device
        dtype = X.dtype

        # Pad to chunk size
        orig_seqlen = seqlen
        if seqlen % self.chunk_size != 0:
            pad_len = self.chunk_size - (seqlen % self.chunk_size)
            X = F.pad(X, (0, 0, 0, 0, 0, pad_len))
            A = F.pad(A, (0, 0, 0, pad_len))
            B = F.pad(B, (0, 0, 0, 0, 0, pad_len))
            C = F.pad(C, (0, 0, 0, 0, 0, pad_len))
            hidden_states = F.pad(hidden_states, (0, 0, 0, pad_len))
            seqlen = X.shape[1]

        n_chunks = seqlen // self.chunk_size

        # Reshape to chunks
        X_chunks = X.view(batch, n_chunks, self.chunk_size, n_heads, d_head).permute(0, 3, 1, 2, 4)
        A_chunks = A.view(batch, n_chunks, self.chunk_size, n_heads).permute(0, 3, 1, 2)
        B_chunks = B.view(batch, n_chunks, self.chunk_size, n_heads, d_state).permute(0, 3, 1, 2, 4)
        C_chunks = C.view(batch, n_chunks, self.chunk_size, n_heads, d_state).permute(0, 3, 1, 2, 4)

        # Initialize
        h_prev = torch.zeros(batch, n_heads, d_state, d_head, device=device, dtype=dtype)

        Y_chunks_list = []
        h_history = [h_prev]
        decay_history = []
        surprise_list = []
        importance_list = []
        consolidation_events = []

        # Importance buffer for consolidation
        importance_buffer = []
        state_buffer = []

        for c in range(n_chunks):
            X_c = X_chunks[:, :, c]
            A_c = A_chunks[:, :, c]
            B_c = B_chunks[:, :, c]
            C_c = C_chunks[:, :, c]

            # Compute surprise from previous chunk (delayed)
            if c == 0:
                decay_scale = 1.0
                surprise_c = torch.zeros(batch, n_heads, device=device, dtype=dtype)
            else:
                h_actual = h_history[-1]
                h_before = h_history[-2]
                decay_prev = decay_history[-1]

                surprise_c = self.surprise_gate.compute_prediction_error(
                    h_actual, h_before, decay_prev
                )

                alpha_c, _ = self.surprise_gate(surprise_c.unsqueeze(-1))
                alpha_c = alpha_c.squeeze(-1)
                decay_scale = (1.0 - alpha_c).mean().item()

            surprise_list.append(surprise_c)

            # Compute importance
            if c > 0:
                importance_c = self.importance_scorer.compute_importance(
                    h_history[-1], surprise_c
                )
            else:
                importance_c = torch.ones(batch, n_heads, device=device, dtype=dtype)
            importance_list.append(importance_c)

            # Phase 4: Selective forgetting (before processing new chunk)
            if c > 0:
                chunk_start = c * self.chunk_size
                chunk_hidden = hidden_states[:, chunk_start:chunk_start+1]
                h_prev, forget_gate = self.selective_forget(
                    h_prev, chunk_hidden, importance_c
                )

                # Synaptic homeostasis
                h_prev = self.synaptic_homeostasis.maybe_apply(h_prev, importance_c)

            # Process chunk
            Y_c, h_end, decay_total = ssd_chunk_forward(
                X_c, A_c, B_c, C_c, h_prev, decay_scale
            )

            Y_chunks_list.append(Y_c)
            h_history.append(h_end)
            decay_history.append(decay_total)
            h_prev = h_end

            # Track for consolidation
            importance_buffer.append(importance_c.mean().item())
            state_buffer.append(h_end.detach())

            # Phase 3: Check consolidation triggers
            buffer_fullness = len(state_buffer) / 8  # Assume buffer size 8
            should_consolidate, triggers = self.consolidation_trigger.should_consolidate(
                surprise_c.mean().item(),
                buffer_fullness,
            )

            if should_consolidate and len(state_buffer) > 0:
                # Write to chunk memory (L2)
                avg_state = torch.stack(state_buffer, dim=0).mean(dim=0)
                avg_importance = torch.tensor(importance_buffer, device=device).mean()
                avg_importance = avg_importance.expand(batch, n_heads)

                self.multi_memory.write_chunk_summary(avg_state, avg_importance)

                consolidation_events.append({
                    'chunk': c,
                    'triggers': triggers,
                    'importance': sum(importance_buffer) / len(importance_buffer),
                })

                # Clear buffers
                importance_buffer = []
                state_buffer = []

        # Stack outputs
        Y_chunks = torch.stack(Y_chunks_list, dim=2)
        Y = Y_chunks.permute(0, 2, 3, 1, 4).reshape(batch, seqlen, n_heads, d_head)
        Y = Y[:, :orig_seqlen]

        # Phase 2: Read from multi-timescale memory and integrate
        memory_out = self.multi_memory.read_memory(hidden_states[:, :orig_seqlen])

        # Integrate memory into output (reshape for integration)
        Y_flat = Y.reshape(batch, orig_seqlen, -1)[:, :, :hidden_states.shape[-1]]
        if Y_flat.shape[-1] < hidden_states.shape[-1]:
            Y_flat = F.pad(Y_flat, (0, hidden_states.shape[-1] - Y_flat.shape[-1]))
        Y_integrated = self.multi_memory.integrate(Y_flat, memory_out)

        # Reshape back
        Y = Y_integrated.view(batch, orig_seqlen, n_heads, d_head)

        # Collect stats
        surprise = torch.stack(surprise_list, dim=-1)
        importance = torch.stack(importance_list, dim=-1)

        stats = {
            'layer_idx': self.layer_idx,
            'n_chunks': n_chunks,
            'mean_surprise': surprise.mean().item(),
            'max_surprise': surprise.max().item(),
            'mean_importance': importance.mean().item(),
            'consolidation_events': len(consolidation_events),
            'consolidation_triggers': self.consolidation_trigger.get_stats(),
            'homeostasis_count': self.synaptic_homeostasis.homeostasis_count.item(),
            'chunk_memory_filled': self.multi_memory.chunk_memory_filled.item(),
        }

        return Y, surprise, importance, stats

    def get_surprise_stats(self) -> Dict:
        """Get surprise statistics."""
        gate = self.surprise_gate
        stats = {
            'layer_idx': self.layer_idx,
            'surprise_ema': gate.surprise_ema.mean().item(),
            'alpha_base_mean': (1 - torch.pow(2.0, gate.log2_alpha_base)).mean().item(),
            'beta_mean': torch.pow(2.0, gate.log2_beta).mean().item(),
            'consolidation': self.consolidation_trigger.get_stats(),
            'homeostasis_count': self.synaptic_homeostasis.homeostasis_count.item(),
            'chunk_memory_filled': self.multi_memory.chunk_memory_filled.item(),
        }
        if self.last_surprise is not None:
            stats['last_surprise_mean'] = self.last_surprise.mean().item()
        if self.last_importance is not None:
            stats['last_importance_mean'] = self.last_importance.mean().item()
        return stats


# =============================================================================
# Full V2 Model
# =============================================================================

class MambaIntegerModelV2Full(nn.Module):
    """
    Complete Mamba-Integer V2 model with all four phases.
    """

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        vocab_size = config['vocab_size']
        d_model = config['d_model']
        n_layer = config['n_layer']

        # Embedding
        self.embed = nn.Embedding(vocab_size, d_model)

        # V2 blocks with full architecture
        self.blocks = nn.ModuleList([
            MambaIntegerBlockV2Full(config, layer_idx=i)
            for i in range(n_layer)
        ])

        # Output
        self.norm_f = nn.RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Tie weights
        self.lm_head.weight = self.embed.weight

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        return_surprise: bool = False,
        return_stats: bool = False,
    ):
        """Forward pass."""
        x = self.embed(input_ids)

        surprise_all = []
        stats_all = []

        for block in self.blocks:
            if return_stats:
                x, surprise, stats = block(x, return_surprise=True, return_stats=True)
                stats_all.append(stats)
            else:
                x, surprise = block(x, return_surprise=True)

            if surprise is not None:
                surprise_all.append(surprise)

        x = self.norm_f(x)
        logits = self.lm_head(x)

        if return_stats:
            return logits, surprise_all, stats_all
        elif return_surprise:
            surprise_stats = {
                'mean': torch.stack([s.mean() for s in surprise_all]).mean().item() if surprise_all else 0,
                'max': torch.stack([s.max() for s in surprise_all]).max().item() if surprise_all else 0,
                'per_layer': [s.mean().item() for s in surprise_all],
            }
            return logits, surprise_stats

        return logits, None

    def get_all_stats(self) -> List[Dict]:
        """Get statistics from all blocks."""
        return [block.get_surprise_stats() for block in self.blocks]

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Core components
    'SurpriseGate',
    'MultiTimescaleMemory',
    'ConsolidationTrigger',
    'ImportanceScorer',
    'SelectiveForgetting',
    'SynapticHomeostasis',
    # Block and model
    'MambaIntegerBlockV2Full',
    'MambaIntegerModelV2Full',
    # Utility
    'ssd_chunk_forward',
]
