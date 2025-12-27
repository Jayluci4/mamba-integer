"""
Surprise-Gated SSD (State Space Duality) for Mamba-Integer V2

Core insight from neuroscience:
- High surprise (prediction error) → HIGH retention (remember this!)
- Low surprise (expected) → LOW retention (ok to forget)

Mathematical formulation:
    prediction_error_t = ||h_t - A @ h_{t-1}||²
    normalized_surprise_t = prediction_error_t / EMA(prediction_error)
    α_t = α_base + (1 - α_base) * tanh(β * surprise_t)
    h_t = α_t * h_{t-1} + (1-α_t) * u_t

This is a minimal delta from ssd_multihead.py - only the retention is adaptive.

PERFORMANCE NOTE:
    Current implementation is SEQUENTIAL (loops per-token) for correctness.
    This is ~100x slower than chunked SSD.

    TODO for production:
    1. Implement chunked surprise computation (per-chunk, not per-token)
    2. Use Triton kernels for parallel surprise computation
    3. Approximate surprise with chunk-level statistics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class SurpriseGatedRetention(nn.Module):
    """
    Computes adaptive retention based on prediction error (surprise).

    Integer-friendly: uses dyadic rationals (powers of 2) where possible.

    Args:
        n_heads: Number of attention heads
        d_state: Dimension of SSM state
        ema_decay: Decay for running EMA of surprise (default 0.99)
        surprise_scale: Scaling factor β for surprise → retention (default 1.0)
    """

    def __init__(
        self,
        n_heads: int,
        d_state: int,
        ema_decay: float = 0.99,
        surprise_scale: float = 1.0,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.d_state = d_state
        self.ema_decay = ema_decay

        # Learned base retention per head (in log2 space for integer-friendly ops)
        # Initialize to ~0.5 retention: log2(0.5) = -1
        self.log2_alpha_base = nn.Parameter(torch.ones(n_heads) * -1.0)

        # Learned surprise scaling factor β per head
        self.log2_beta = nn.Parameter(torch.zeros(n_heads))

        # Running EMA of surprise for normalization (not trained)
        self.register_buffer('surprise_ema', torch.ones(n_heads))

        # Track statistics for logging
        self.register_buffer('step_count', torch.tensor(0))
        self.register_buffer('surprise_history', torch.zeros(100))  # Rolling buffer

    def compute_prediction_error(
        self,
        h_prev: torch.Tensor,  # [B, n_heads, d_state]
        h_new: torch.Tensor,   # [B, n_heads, d_state]
        A_decay: torch.Tensor, # [B, n_heads] or [n_heads]
    ) -> torch.Tensor:
        """
        Compute prediction error: how much did the state change unexpectedly?

        prediction_error = ||h_new - decay * h_prev||²

        Returns:
            prediction_error: [B, n_heads]
        """
        # Expected state if no new input
        if A_decay.dim() == 1:
            A_decay = A_decay.unsqueeze(0)  # [1, n_heads]
        h_expected = A_decay.unsqueeze(-1) * h_prev  # [B, n_heads, d_state]

        # Prediction error = squared difference
        error = (h_new - h_expected).pow(2).mean(dim=-1)  # [B, n_heads]

        return error

    def forward(
        self,
        prediction_error: torch.Tensor,  # [B, n_heads]
        training: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute adaptive retention from prediction error.

        Args:
            prediction_error: [B, n_heads] squared prediction error per head
            training: Whether to update running statistics

        Returns:
            alpha: [B, n_heads] adaptive retention (higher = more retention)
            normalized_surprise: [B, n_heads] for logging
        """
        batch = prediction_error.shape[0]

        # Update running EMA of surprise (only during training)
        if training:
            with torch.no_grad():
                batch_surprise = prediction_error.mean(dim=0)  # [n_heads]
                self.surprise_ema = (
                    self.ema_decay * self.surprise_ema +
                    (1 - self.ema_decay) * batch_surprise
                )
                # Update step count and history
                idx = self.step_count % 100
                self.surprise_history[idx] = batch_surprise.mean()
                self.step_count += 1

        # Normalize surprise by running EMA
        # High values = more surprising than average
        normalized_surprise = prediction_error / (self.surprise_ema.unsqueeze(0) + 1e-6)

        # Base retention: α_base = 1 - 2^(log2_alpha_base)
        # Clamped to valid range [0.1, 0.99]
        log2_clamped = self.log2_alpha_base.clamp(-3.32, -0.015)  # 2^-3.32 ≈ 0.1, 2^-0.015 ≈ 0.99
        alpha_base = 1.0 - torch.pow(2.0, log2_clamped)  # [n_heads]

        # Surprise scaling: β = 2^log2_beta
        beta = torch.pow(2.0, self.log2_beta.clamp(-2, 2))  # [n_heads], range [0.25, 4]

        # Surprise boost: tanh(β * surprise) ∈ [-1, 1]
        # High surprise → boost close to 1 → α close to 1 (high retention)
        boost = torch.tanh(beta.unsqueeze(0) * normalized_surprise)  # [B, n_heads]

        # Final retention: α = α_base + (1 - α_base) * max(0, boost)
        # Only positive boost increases retention (negative surprise = "bored")
        boost_positive = F.relu(boost)
        alpha = alpha_base.unsqueeze(0) + (1 - alpha_base.unsqueeze(0)) * boost_positive

        # Clamp for numerical stability
        alpha = alpha.clamp(0.01, 0.999)

        return alpha, normalized_surprise


def surprise_ssd_forward(
    X: torch.Tensor,      # [B, L, n_heads, d_head]
    A: torch.Tensor,      # [B, L, n_heads] log decay (base, will be modulated)
    B: torch.Tensor,      # [B, L, n_heads, d_state]
    C: torch.Tensor,      # [B, L, n_heads, d_state]
    surprise_gate: SurpriseGatedRetention,
    chunk_size: int = 64,
    return_surprise: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    SSD forward with surprise-gated adaptive retention.

    Key difference from standard SSD:
    - A (decay) is modulated per-token based on prediction error
    - High surprise → high retention (low decay)
    - Low surprise → low retention (high decay, ok to forget)

    Args:
        X: Input [B, L, n_heads, d_head]
        A: Base log decay [B, L, n_heads]
        B: Input projection [B, L, n_heads, d_state]
        C: Output projection [B, L, n_heads, d_state]
        surprise_gate: Module that computes adaptive retention
        chunk_size: Chunk size for matmul
        return_surprise: Whether to return surprise scores

    Returns:
        Y: Output [B, L, n_heads, d_head]
        surprise_scores: Optional [B, L, n_heads] normalized surprise
    """
    batch, seqlen, n_heads, d_head = X.shape
    d_state = B.shape[-1]
    device = X.device
    dtype = X.dtype

    # State h has shape [B, n_heads, d_state, d_head]
    # It accumulates outer products: h += B^T @ X
    h = torch.zeros(batch, n_heads, d_state, d_head, device=device, dtype=dtype)

    # For surprise computation, we track a reduced state (mean over d_head)
    h_reduced = torch.zeros(batch, n_heads, d_state, device=device, dtype=dtype)

    # Storage for outputs and surprise
    Y_list = []
    surprise_list = []

    # Convert A from log space to decay factor
    A_decay_base = torch.exp(A)  # [B, L, n_heads], values in (0, 1)

    for t in range(seqlen):
        # Get inputs for this timestep
        x_t = X[:, t]  # [B, n_heads, d_head]
        B_t = B[:, t]  # [B, n_heads, d_state]
        C_t = C[:, t]  # [B, n_heads, d_state]
        A_decay_t = A_decay_base[:, t]  # [B, n_heads]

        # Compute input contribution: u = B^T @ x (outer product style)
        # u[s, d] = B[s] * x[d] (broadcasting)
        u_t = B_t.unsqueeze(-1) * x_t.unsqueeze(-2)  # [B, n_heads, d_state, d_head]

        # For surprise: reduced version (mean over d_head)
        u_reduced = u_t.mean(dim=-1)  # [B, n_heads, d_state]

        # Compute prediction error on reduced state
        h_expected = A_decay_t.unsqueeze(-1) * h_reduced
        h_new_reduced = h_expected + (1 - A_decay_t.unsqueeze(-1)) * u_reduced
        pred_error = (h_new_reduced - h_expected).pow(2).mean(dim=-1)  # [B, n_heads]

        # Get adaptive retention based on surprise
        alpha, normalized_surprise = surprise_gate(pred_error, training=surprise_gate.training)

        # Apply surprise-modulated retention to full state
        # h_new = alpha * h + (1-alpha) * u
        alpha_expanded = alpha.unsqueeze(-1).unsqueeze(-1)  # [B, n_heads, 1, 1]
        h = alpha_expanded * h + (1 - alpha_expanded) * u_t

        # Update reduced state for next iteration
        h_reduced = h.mean(dim=-1)

        # Output: y[d] = sum_s C[s] * h[s, d]
        # C: [B, n_heads, d_state], h: [B, n_heads, d_state, d_head]
        y_t = torch.einsum('bhs,bhsd->bhd', C_t, h)  # [B, n_heads, d_head]

        Y_list.append(y_t)
        surprise_list.append(normalized_surprise)

    # Stack outputs
    Y = torch.stack(Y_list, dim=1)  # [B, L, n_heads, d_head]

    if return_surprise:
        surprise_scores = torch.stack(surprise_list, dim=1)  # [B, L, n_heads]
        return Y, surprise_scores
    return Y, None


class SurpriseGatedSSD(nn.Module):
    """
    SSD module with surprise-gated adaptive retention.

    Drop-in replacement for ssd_multihead that adds:
    1. Per-token prediction error computation
    2. Surprise-based retention modulation
    3. Logging of surprise statistics
    """

    def __init__(
        self,
        n_heads: int,
        d_state: int,
        d_head: int,
        chunk_size: int = 64,
        ema_decay: float = 0.99,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.d_state = d_state
        self.d_head = d_head
        self.chunk_size = chunk_size

        # Surprise gate
        self.surprise_gate = SurpriseGatedRetention(
            n_heads=n_heads,
            d_state=d_state,
            ema_decay=ema_decay,
        )

    def forward(
        self,
        X: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        return_surprise: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with surprise-gated retention.

        Args:
            X: [B, L, n_heads, d_head] input
            A: [B, L, n_heads] log decay
            B: [B, L, n_heads, d_state] input projection
            C: [B, L, n_heads, d_state] output projection
            return_surprise: Whether to return surprise scores

        Returns:
            Y: [B, L, n_heads, d_head] output
            surprise: Optional [B, L, n_heads] surprise scores
        """
        return surprise_ssd_forward(
            X, A, B, C,
            self.surprise_gate,
            self.chunk_size,
            return_surprise,
        )

    def get_surprise_stats(self) -> dict:
        """Get surprise statistics for logging."""
        return {
            'surprise_ema': self.surprise_gate.surprise_ema.mean().item(),
            'surprise_history_mean': self.surprise_gate.surprise_history.mean().item(),
            'alpha_base_mean': (1 - torch.pow(2.0, self.surprise_gate.log2_alpha_base)).mean().item(),
            'beta_mean': torch.pow(2.0, self.surprise_gate.log2_beta).mean().item(),
        }


class MambaIntegerBlockV2Surprise(nn.Module):
    """
    Mamba-Integer V2 block with surprise-gated retention.

    Key differences from MambaIntegerBlockV2:
    1. Uses SurpriseGatedSSD instead of fixed-decay SSD
    2. Adaptive retention based on prediction error
    3. Logs surprise statistics for analysis
    """

    def __init__(self, config: dict, layer_idx: int):
        super().__init__()

        self.config = config
        self.layer_idx = layer_idx

        d_model = config['d_model']
        ssm_cfg = config['ssm_cfg']

        # Multi-head dimensions
        self.n_heads = ssm_cfg.get('n_heads', 24)
        self.d_head = ssm_cfg.get('d_head', 64)
        self.d_state = ssm_cfg.get('d_state', 64)
        self.d_inner = self.n_heads * self.d_head

        # Use nn.Linear for prototype (BitLinear requires Triton warmup)
        # TODO: Switch to BitLinear for production
        Linear = nn.Linear

        # Normalization (use simple RMSNorm for prototype)
        self.norm = nn.RMSNorm(d_model)

        # Projections
        self.in_proj = Linear(d_model, self.d_inner * 2)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, 4,
            groups=self.d_inner, padding=3
        )

        # SSM projections
        proj_size = self.n_heads * (1 + 2 * self.d_state)
        self.x_proj = Linear(self.d_inner, proj_size)

        # Base A (log decay) - will be modulated by surprise gate
        self.A_log = nn.Parameter(torch.log(torch.ones(self.n_heads) * 0.5))

        # Surprise-gated SSD (THE KEY V2 COMPONENT)
        self.ssd = SurpriseGatedSSD(
            n_heads=self.n_heads,
            d_state=self.d_state,
            d_head=self.d_head,
            chunk_size=ssm_cfg.get('chunk_size', 64),
        )

        # Output projection
        self.out_proj = Linear(self.d_inner, d_model)

        # Residual gate
        n_layer = config.get('n_layer', 24)
        self.res_gate = nn.Parameter(torch.ones(1) / math.sqrt(2 * n_layer))

        # Store last surprise for logging
        self.last_surprise = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        return_surprise: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with surprise-gated retention.

        Args:
            hidden_states: [B, L, d_model]
            return_surprise: Whether to return surprise scores

        Returns:
            output: [B, L, d_model]
            surprise: Optional [B, L, n_heads]
        """
        residual = hidden_states
        batch, seqlen, _ = hidden_states.shape

        # Normalize
        hidden_states = self.norm(hidden_states)

        # Input projection
        xz = self.in_proj(hidden_states)
        x, z = xz.chunk(2, dim=-1)

        # Conv1d
        x = self.conv1d(x.transpose(1, 2)).transpose(1, 2)[:, :seqlen]

        # Activation (squareplus for integer-only)
        x = F.silu(x)  # Use SiLU for prototype, replace with squareplus later

        # SSM projections
        x_proj_out = self.x_proj(x)
        x_proj_out = x_proj_out.view(batch, seqlen, self.n_heads, 1 + 2 * self.d_state)

        # Split
        dt = x_proj_out[..., 0]
        B_ssm = x_proj_out[..., 1:1+self.d_state]
        C_ssm = x_proj_out[..., 1+self.d_state:]

        # Normalize B and C
        B_ssm = F.normalize(B_ssm, dim=-1)
        C_ssm = F.normalize(C_ssm, dim=-1)

        # Compute A (base decay, will be modulated by surprise gate)
        A_log_clamped = self.A_log.clamp(-2.3, -0.01)
        A = A_log_clamped.view(1, 1, self.n_heads).expand(batch, seqlen, -1)

        # Modulate with dt
        dt_scale = F.softplus(dt.clamp(-10, 10)).clamp(0.01, 10.0)
        A = A * dt_scale
        A = A.clamp(-20.0, -0.001)

        # Reshape X
        X = x.view(batch, seqlen, self.n_heads, self.d_head)

        # Run surprise-gated SSD (THE V2 MAGIC)
        Y, surprise = self.ssd(X, A, B_ssm, C_ssm, return_surprise=True)

        # Store surprise for logging
        self.last_surprise = surprise

        # Reshape back
        y = Y.reshape(batch, seqlen, self.d_inner)
        y = y.clamp(-100, 100)

        # Gate with z
        y = y * torch.sigmoid(z)

        # Output projection
        out = self.out_proj(y)
        out = out.clamp(-100, 100)

        output = residual + out * self.res_gate

        if return_surprise:
            return output, surprise
        return output, None

    def get_surprise_stats(self) -> dict:
        """Get surprise statistics for logging."""
        stats = self.ssd.get_surprise_stats()
        stats['layer_idx'] = self.layer_idx
        if self.last_surprise is not None:
            stats['last_surprise_mean'] = self.last_surprise.mean().item()
            stats['last_surprise_max'] = self.last_surprise.max().item()
        return stats
