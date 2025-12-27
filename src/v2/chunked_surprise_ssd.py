"""
Chunked Surprise-Gated SSD for Mamba-Integer V2 (FIXED VERSION)

Key fix: Use DELAYED STATE-BASED surprise, not input energy.

Previous (wrong):
    surprise = ||B @ X||²  # Input energy - measures loudness, not novelty

Fixed (correct):
    surprise_c = ||h_end[c-1] - decay * h_end[c-2]||²  # State prediction error
    alpha_c = f(surprise_{c-1})  # Use PREVIOUS chunk's surprise for THIS chunk

This is causal: surprise from chunk c-1 modulates retention in chunk c.
Neuroscience analogy: "That was surprising → pay more attention NOW"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List

import sys
sys.path.insert(0, '/home/jayantlohia16/mamba-integer/src')
from triton_kernels.ssd_multihead import build_causal_decay_matrix_multihead


class ChunkedSurpriseGate(nn.Module):
    """
    Computes per-chunk adaptive retention based on STATE prediction error.

    Key insight: Surprise should measure how much the state deviated from
    expectation, not how loud the input was.
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

        # Learned base retention per head
        # Initialize to ~0.9 retention (high baseline, surprise boosts further)
        self.log2_alpha_base = nn.Parameter(torch.ones(n_heads) * -0.15)

        # Learned surprise scaling factor β per head
        self.log2_beta = nn.Parameter(torch.zeros(n_heads))

        # Running EMA of surprise for normalization
        self.register_buffer('surprise_ema', torch.ones(n_heads))

        # Statistics tracking
        self.register_buffer('step_count', torch.tensor(0))

    def compute_state_prediction_error(
        self,
        h_actual: torch.Tensor,    # [B, n_heads, d_state] or [B, n_heads, d_state, d_head]
        h_prev: torch.Tensor,      # Same shape as h_actual
        decay: torch.Tensor,       # [B, n_heads] or scalar
    ) -> torch.Tensor:
        """
        Compute state prediction error: ||h_actual - decay * h_prev||²

        This measures how much the state changed BEYOND what decay would predict.
        High values = surprising (state changed more than expected)
        Low values = predictable (state evolved as expected)
        """
        # Expected state (if no new information entered)
        # Need to broadcast decay over all dimensions of h_prev
        if decay.dim() == 0:
            h_expected = decay * h_prev
        elif decay.dim() == 2:
            # decay: [B, n_heads]
            # h_prev: [B, n_heads, d_state] or [B, n_heads, d_state, d_head]
            decay_expanded = decay
            for _ in range(h_prev.dim() - decay.dim()):
                decay_expanded = decay_expanded.unsqueeze(-1)
            h_expected = decay_expanded * h_prev
        else:
            h_expected = decay * h_prev

        # Prediction error (squared L2 norm)
        error = (h_actual - h_expected).pow(2)

        # Reduce to [B, n_heads]
        while error.dim() > 2:
            error = error.mean(dim=-1)

        return error

    def forward(
        self,
        prediction_errors: torch.Tensor,  # [B, n_heads, n_chunks] or [B, n_heads]
        training: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute adaptive retention from prediction errors.

        Args:
            prediction_errors: State prediction errors per chunk
            training: Whether to update EMA

        Returns:
            alpha: Adaptive retention (higher = more retention)
            normalized_surprise: For logging
        """
        # Handle different input shapes
        if prediction_errors.dim() == 2:
            prediction_errors = prediction_errors.unsqueeze(-1)

        # Update EMA
        if training:
            with torch.no_grad():
                mean_error = prediction_errors.mean(dim=(0, 2))  # [n_heads]
                self.surprise_ema = self.ema_decay * self.surprise_ema + (1 - self.ema_decay) * mean_error
                self.step_count += 1

        # Normalize by running statistics
        normalized = prediction_errors / (self.surprise_ema.view(1, -1, 1) + 1e-6)

        # Base retention: α_base ∈ [0.1, 0.99]
        log2_clamped = self.log2_alpha_base.clamp(-3.32, -0.015)
        alpha_base = 1.0 - torch.pow(2.0, log2_clamped)  # [n_heads]

        # Surprise scaling: β ∈ [0.25, 4]
        beta = torch.pow(2.0, self.log2_beta.clamp(-2, 2))  # [n_heads]

        # Boost from surprise: high surprise → high retention
        boost = torch.tanh(beta.view(1, -1, 1) * normalized)
        boost_positive = F.relu(boost)  # Only positive boost (surprise increases retention)

        # Final alpha: α = α_base + (1 - α_base) * boost
        alpha = alpha_base.view(1, -1, 1) + (1 - alpha_base.view(1, -1, 1)) * boost_positive
        alpha = alpha.clamp(0.01, 0.999)

        return alpha, normalized


def ssd_chunk_forward(
    X_chunk: torch.Tensor,     # [B, n_heads, cs, d_head]
    A_chunk: torch.Tensor,     # [B, n_heads, cs] log decay
    B_chunk: torch.Tensor,     # [B, n_heads, cs, d_state]
    C_chunk: torch.Tensor,     # [B, n_heads, cs, d_state]
    h_prev: torch.Tensor,      # [B, n_heads, d_state, d_head] incoming state
    decay_scale: float = 1.0,  # Multiplier for decay (from surprise modulation)
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Process a single chunk with SSD.

    Returns:
        Y_chunk: [B, n_heads, cs, d_head] output
        h_end: [B, n_heads, d_state, d_head] final state
        decay_total: [B, n_heads] total decay across chunk
    """
    B, n_heads, cs, d_head = X_chunk.shape
    d_state = B_chunk.shape[-1]
    device = X_chunk.device
    dtype = X_chunk.dtype

    # Apply decay scaling
    A_scaled = A_chunk * decay_scale

    # Cumulative sum for decay matrix
    A_cumsum = torch.cumsum(A_scaled, dim=-1)  # [B, n_heads, cs]

    # Build L matrix for this chunk
    L = build_causal_decay_matrix_multihead(A_cumsum.unsqueeze(2))[:, :, 0]  # [B, n_heads, cs, cs]

    # Intra-chunk: Y = (L * CB) @ X
    CB = torch.einsum('bhid,bhjd->bhij', C_chunk, B_chunk)  # [B, n_heads, cs, cs]
    L_CB = L * CB
    Y_intra = torch.einsum('bhij,bhjd->bhid', L_CB, X_chunk)  # [B, n_heads, cs, d_head]

    # Inter-chunk contribution from h_prev
    decay_from_start = torch.exp(A_cumsum)  # [B, n_heads, cs]
    # Y_inter[i] = C[i] @ h_prev * decay_from_start[i]
    Y_inter = torch.einsum('bhis,bhsd,bhi->bhid', C_chunk, h_prev, decay_from_start)

    Y_chunk = Y_intra + Y_inter

    # Compute final state h_end
    decay_to_end = torch.exp(A_cumsum[:, :, -1:] - A_cumsum)  # [B, n_heads, cs]
    # h_contribution[t] = B[t] @ X[t] * decay_to_end[t]
    h_chunk_contrib = torch.einsum('bhts,bhtd,bht->bhsd', B_chunk, X_chunk, decay_to_end)

    # h_end = decay_total * h_prev + h_chunk_contrib
    decay_total = torch.exp(A_cumsum[:, :, -1])  # [B, n_heads]
    h_end = decay_total.unsqueeze(-1).unsqueeze(-1) * h_prev + h_chunk_contrib

    return Y_chunk, h_end, decay_total


def chunked_surprise_ssd_forward(
    X: torch.Tensor,      # [B, L, n_heads, d_head]
    A: torch.Tensor,      # [B, L, n_heads] base log decay
    B: torch.Tensor,      # [B, L, n_heads, d_state]
    C: torch.Tensor,      # [B, L, n_heads, d_state]
    surprise_gate: ChunkedSurpriseGate,
    chunk_size: int = 64,
    return_surprise: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Chunked SSD with DELAYED STATE-BASED surprise.

    Key mechanism:
    1. Process chunk c
    2. Compute prediction error: ||h_end[c] - decay * h_end[c-1]||²
    3. Use this error to modulate retention for chunk c+1

    This is causal: we use PAST surprise to modulate CURRENT retention.
    """
    batch, seqlen, n_heads, d_head = X.shape
    d_state = B.shape[-1]
    device = X.device
    dtype = X.dtype

    # Pad to multiple of chunk_size
    orig_seqlen = seqlen
    if seqlen % chunk_size != 0:
        pad_len = chunk_size - (seqlen % chunk_size)
        X = F.pad(X, (0, 0, 0, 0, 0, pad_len), value=0.0)
        A = F.pad(A, (0, 0, 0, pad_len), value=0.0)
        B = F.pad(B, (0, 0, 0, 0, 0, pad_len), value=0.0)
        C = F.pad(C, (0, 0, 0, 0, 0, pad_len), value=0.0)
        seqlen = X.shape[1]

    n_chunks = seqlen // chunk_size

    # Reshape to chunks and transpose
    X_chunks = X.view(batch, n_chunks, chunk_size, n_heads, d_head).permute(0, 3, 1, 2, 4)
    A_chunks = A.view(batch, n_chunks, chunk_size, n_heads).permute(0, 3, 1, 2)
    B_chunks = B.view(batch, n_chunks, chunk_size, n_heads, d_state).permute(0, 3, 1, 2, 4)
    C_chunks = C.view(batch, n_chunks, chunk_size, n_heads, d_state).permute(0, 3, 1, 2, 4)
    # All now: [B, n_heads, n_chunks, cs, ...]

    # Initialize state
    h_prev = torch.zeros(batch, n_heads, d_state, d_head, device=device, dtype=dtype)

    # Storage
    Y_chunks_list = []
    h_history = [h_prev]  # Track states for surprise computation
    decay_history = []
    surprise_list = []

    # Process chunks with delayed surprise
    for c in range(n_chunks):
        # Get chunk data
        X_c = X_chunks[:, :, c]  # [B, n_heads, cs, d_head]
        A_c = A_chunks[:, :, c]  # [B, n_heads, cs]
        B_c = B_chunks[:, :, c]  # [B, n_heads, cs, d_state]
        C_c = C_chunks[:, :, c]  # [B, n_heads, cs, d_state]

        # Compute decay scale from PREVIOUS chunk's surprise (delayed)
        if c == 0:
            # First chunk: use baseline (no previous surprise)
            decay_scale = 1.0
            surprise_c = torch.zeros(batch, n_heads, device=device, dtype=dtype)
        else:
            # Compute prediction error from previous chunk
            # surprise = ||h_end[c-1] - decay[c-1] * h_end[c-2]||²
            h_actual = h_history[-1]      # h_end[c-1]
            h_before = h_history[-2]      # h_end[c-2]
            decay_prev = decay_history[-1]  # decay across chunk c-1

            surprise_c = surprise_gate.compute_state_prediction_error(
                h_actual, h_before, decay_prev
            )  # [B, n_heads]

            # Get adaptive retention
            alpha_c, _ = surprise_gate(surprise_c.unsqueeze(-1), training=surprise_gate.training)
            alpha_c = alpha_c.squeeze(-1)  # [B, n_heads]

            # Convert alpha to decay scale
            # Higher alpha = more retention = LESS decay = scale closer to 0
            # Lower alpha = less retention = MORE decay = scale closer to 1
            decay_scale = (1.0 - alpha_c).mean().item()  # Scalar for simplicity
            # TODO: Per-head decay scaling for full expressiveness

        surprise_list.append(surprise_c)

        # Process chunk with modulated decay
        Y_c, h_end, decay_total = ssd_chunk_forward(
            X_c, A_c, B_c, C_c, h_prev, decay_scale
        )

        Y_chunks_list.append(Y_c)
        h_history.append(h_end)
        decay_history.append(decay_total)
        h_prev = h_end

    # Stack outputs
    Y_chunks = torch.stack(Y_chunks_list, dim=2)  # [B, n_heads, n_chunks, cs, d_head]
    Y = Y_chunks.permute(0, 2, 3, 1, 4).reshape(batch, seqlen, n_heads, d_head)

    # Remove padding
    Y = Y[:, :orig_seqlen]

    if return_surprise:
        surprise = torch.stack(surprise_list, dim=-1)  # [B, n_heads, n_chunks]
        return Y, surprise
    return Y, None


class ChunkedSurpriseGatedSSD(nn.Module):
    """
    Fast SSD with CORRECT delayed state-based surprise.
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

        self.surprise_gate = ChunkedSurpriseGate(
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
        return chunked_surprise_ssd_forward(
            X, A, B, C,
            self.surprise_gate,
            self.chunk_size,
            return_surprise,
        )

    def get_surprise_stats(self) -> dict:
        gate = self.surprise_gate
        return {
            'surprise_ema': gate.surprise_ema.mean().item(),
            'alpha_base_mean': (1 - torch.pow(2.0, gate.log2_alpha_base)).mean().item(),
            'beta_mean': torch.pow(2.0, gate.log2_beta).mean().item(),
            'step_count': gate.step_count.item(),
        }


class MambaIntegerBlockV2Chunked(nn.Module):
    """
    Mamba-Integer V2 block with CORRECT delayed state-based surprise.
    """

    def __init__(self, config: dict, layer_idx: int):
        super().__init__()

        self.config = config
        self.layer_idx = layer_idx

        d_model = config['d_model']
        ssm_cfg = config['ssm_cfg']

        self.n_heads = ssm_cfg.get('n_heads', 24)
        self.d_head = ssm_cfg.get('d_head', 64)
        self.d_state = ssm_cfg.get('d_state', 64)
        self.d_inner = self.n_heads * self.d_head
        self.chunk_size = ssm_cfg.get('chunk_size', 64)

        Linear = nn.Linear

        self.norm = nn.RMSNorm(d_model)
        self.in_proj = Linear(d_model, self.d_inner * 2)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, 4, groups=self.d_inner, padding=3)

        proj_size = self.n_heads * (1 + 2 * self.d_state)
        self.x_proj = Linear(self.d_inner, proj_size)

        self.A_log = nn.Parameter(torch.log(torch.ones(self.n_heads) * 0.5))

        self.ssd = ChunkedSurpriseGatedSSD(
            n_heads=self.n_heads,
            d_state=self.d_state,
            d_head=self.d_head,
            chunk_size=self.chunk_size,
        )

        self.out_proj = Linear(self.d_inner, d_model)

        n_layer = config.get('n_layer', 24)
        self.res_gate = nn.Parameter(torch.ones(1) / math.sqrt(2 * n_layer))

        self.last_surprise = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        return_surprise: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        residual = hidden_states
        batch, seqlen, _ = hidden_states.shape

        hidden_states = self.norm(hidden_states)

        xz = self.in_proj(hidden_states)
        x, z = xz.chunk(2, dim=-1)

        x = self.conv1d(x.transpose(1, 2)).transpose(1, 2)[:, :seqlen]
        x = F.silu(x)

        x_proj_out = self.x_proj(x)
        x_proj_out = x_proj_out.view(batch, seqlen, self.n_heads, 1 + 2 * self.d_state)

        dt = x_proj_out[..., 0]
        B_ssm = x_proj_out[..., 1:1+self.d_state]
        C_ssm = x_proj_out[..., 1+self.d_state:]

        B_ssm = F.normalize(B_ssm, dim=-1)
        C_ssm = F.normalize(C_ssm, dim=-1)

        A_log_clamped = self.A_log.clamp(-2.3, -0.01)
        A = A_log_clamped.view(1, 1, self.n_heads).expand(batch, seqlen, -1)

        dt_scale = F.softplus(dt.clamp(-10, 10)).clamp(0.01, 10.0)
        A = A * dt_scale
        A = A.clamp(-20.0, -0.001)

        X = x.view(batch, seqlen, self.n_heads, self.d_head)

        Y, surprise = self.ssd(X, A, B_ssm, C_ssm, return_surprise=True)

        self.last_surprise = surprise

        y = Y.reshape(batch, seqlen, self.d_inner)
        y = y.clamp(-100, 100)
        y = y * torch.sigmoid(z)

        out = self.out_proj(y)
        out = out.clamp(-100, 100)

        output = residual + out * self.res_gate

        if return_surprise:
            return output, surprise
        return output, None

    def get_surprise_stats(self) -> dict:
        stats = self.ssd.get_surprise_stats()
        stats['layer_idx'] = self.layer_idx
        if self.last_surprise is not None:
            stats['last_surprise_mean'] = self.last_surprise.mean().item()
            stats['last_surprise_max'] = self.last_surprise.max().item()
        return stats
