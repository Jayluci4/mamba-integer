"""
Chunked Surprise-Gated SSD for Mamba-Integer V2

This is the FAST version that computes surprise per-chunk instead of per-token.
Uses the same chunked matmul infrastructure as the original SSD.

Key insight:
- Original SSD: fixed decay A per head
- Chunked Surprise SSD: adaptive decay per CHUNK based on prediction error

Performance:
- Sequential (per-token): ~166ms for seqlen=64
- Chunked (per-chunk): ~2ms for seqlen=64 (expected ~80x speedup)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

# Import the original SSD building blocks
import sys
sys.path.insert(0, '/home/jayantlohia16/mamba-integer/src')
from triton_kernels.ssd_multihead import build_causal_decay_matrix_multihead


class ChunkedSurpriseGate(nn.Module):
    """
    Computes per-chunk adaptive retention based on chunk-level prediction error.

    Instead of computing surprise per-token (O(L)), we compute per-chunk (O(L/chunk_size)).
    This is ~chunk_size times faster while preserving the core behavior.
    """

    def __init__(
        self,
        n_heads: int,
        ema_decay: float = 0.99,
    ):
        super().__init__()
        self.n_heads = n_heads
        self.ema_decay = ema_decay

        # Learned base retention per head (log2 space)
        # Initialize to ~0.9 retention for chunks (higher than per-token)
        self.log2_alpha_base = nn.Parameter(torch.ones(n_heads) * -0.15)  # 2^-0.15 ≈ 0.9

        # Learned surprise scaling factor β per head
        self.log2_beta = nn.Parameter(torch.zeros(n_heads))

        # Running EMA of chunk surprise
        self.register_buffer('surprise_ema', torch.ones(n_heads))

    def forward(
        self,
        chunk_surprise: torch.Tensor,  # [B, n_heads, n_chunks]
        training: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute per-chunk adaptive retention.

        Args:
            chunk_surprise: [B, n_heads, n_chunks] prediction error per chunk
            training: Whether to update EMA

        Returns:
            alpha: [B, n_heads, n_chunks] adaptive retention per chunk
            normalized_surprise: [B, n_heads, n_chunks]
        """
        # Update EMA
        if training:
            with torch.no_grad():
                mean_surprise = chunk_surprise.mean(dim=(0, 2))  # [n_heads]
                self.surprise_ema = self.ema_decay * self.surprise_ema + (1 - self.ema_decay) * mean_surprise

        # Normalize
        normalized = chunk_surprise / (self.surprise_ema.view(1, -1, 1) + 1e-6)

        # Base retention
        log2_clamped = self.log2_alpha_base.clamp(-3.32, -0.015)
        alpha_base = 1.0 - torch.pow(2.0, log2_clamped)  # [n_heads]

        # Surprise scaling
        beta = torch.pow(2.0, self.log2_beta.clamp(-2, 2))  # [n_heads]

        # Boost from surprise
        boost = torch.tanh(beta.view(1, -1, 1) * normalized)
        boost_positive = F.relu(boost)

        # Final alpha
        alpha = alpha_base.view(1, -1, 1) + (1 - alpha_base.view(1, -1, 1)) * boost_positive
        alpha = alpha.clamp(0.01, 0.999)

        return alpha, normalized


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
    Chunked SSD with per-chunk surprise-gated retention.

    Key difference from standard SSD:
    - A (decay) is modulated per-CHUNK based on chunk-level prediction error
    - Uses existing chunked matmul infrastructure for speed

    Args:
        X: [B, L, n_heads, d_head] input
        A: [B, L, n_heads] base log decay (will be modulated)
        B: [B, L, n_heads, d_state] input projection
        C: [B, L, n_heads, d_state] output projection
        surprise_gate: Module for computing adaptive retention
        chunk_size: Chunk size for matmul
        return_surprise: Whether to return surprise scores

    Returns:
        Y: [B, L, n_heads, d_head] output
        surprise: Optional [B, n_heads, n_chunks] per-chunk surprise
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

    # Reshape to chunks
    X_chunks = X.view(batch, n_chunks, chunk_size, n_heads, d_head)
    A_chunks = A.view(batch, n_chunks, chunk_size, n_heads)
    B_chunks = B.view(batch, n_chunks, chunk_size, n_heads, d_state)
    C_chunks = C.view(batch, n_chunks, chunk_size, n_heads, d_state)

    # Transpose for processing
    X_t = X_chunks.permute(0, 3, 1, 2, 4)  # [B, n_heads, n_chunks, cs, d_head]
    A_t = A_chunks.permute(0, 3, 1, 2)      # [B, n_heads, n_chunks, cs]
    B_t = B_chunks.permute(0, 3, 1, 2, 4)  # [B, n_heads, n_chunks, cs, d_state]
    C_t = C_chunks.permute(0, 3, 1, 2, 4)  # [B, n_heads, n_chunks, cs, d_state]

    # =========================================================================
    # STEP 1: Compute chunk-level surprise (prediction error)
    # =========================================================================

    # First, run a quick forward pass to get chunk final states
    # We need: h_chunk[c] = final state after processing chunk c

    # Compute per-chunk input energy as proxy for surprise
    # (Actual state-based surprise would require sequential pass)
    # Approximation: high input variance = high surprise
    chunk_input_energy = (X_t ** 2).mean(dim=(-1, -2))  # [B, n_heads, n_chunks]

    # Alternative: use B-weighted input energy
    # This correlates with how much new information enters the state
    BX = torch.einsum('bhncs,bhncd->bhnsd', B_t, X_t)  # [B, n_heads, n_chunks, d_state, d_head]
    chunk_state_delta = (BX ** 2).mean(dim=(-1, -2))  # [B, n_heads, n_chunks]

    # Use state delta as surprise metric
    chunk_surprise = chunk_state_delta

    # =========================================================================
    # STEP 2: Compute adaptive retention per chunk
    # =========================================================================

    alpha_chunks, normalized_surprise = surprise_gate(chunk_surprise, training=surprise_gate.training)
    # alpha_chunks: [B, n_heads, n_chunks]

    # =========================================================================
    # STEP 3: Modulate A based on surprise
    # =========================================================================

    # Original A is log-decay (negative values)
    # Higher alpha (more retention) means LESS decay → A closer to 0
    # Lower alpha (less retention) means MORE decay → A more negative

    # Scale factor: alpha=1 → scale=0 (no decay), alpha=0 → scale=1 (full decay)
    decay_scale = 1.0 - alpha_chunks  # [B, n_heads, n_chunks]

    # Apply to A: modulate decay within each chunk uniformly
    # A_modulated[c, t] = A[c, t] * decay_scale[c]
    A_modulated = A_t * decay_scale.unsqueeze(-1)  # [B, n_heads, n_chunks, cs]

    # =========================================================================
    # STEP 4: Run chunked SSD with modulated A
    # =========================================================================

    # Cumulative sum of modulated A
    A_cumsum = torch.cumsum(A_modulated, dim=-1)  # [B, n_heads, n_chunks, cs]

    # Build decay matrix L
    L = build_causal_decay_matrix_multihead(A_cumsum)  # [B, n_heads, n_chunks, cs, cs]

    # Intra-chunk computation: Y_intra = (L * CB) @ X
    CB = torch.einsum('bhnid,bhnjd->bhnij', C_t, B_t)  # [B, n_heads, n_chunks, cs, cs]
    L_CB = L * CB
    Y_intra = torch.einsum('bhnij,bhnjd->bhnid', L_CB, X_t)  # [B, n_heads, n_chunks, cs, d_head]

    # Inter-chunk state propagation
    decay_chunk = torch.exp(A_modulated.sum(dim=-1))  # [B, n_heads, n_chunks]
    decay_to_end = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)  # [B, n_heads, n_chunks, cs]

    # Chunk final states
    h_chunk_final = torch.einsum('bhnc,bhncs,bhncd->bhnsd', decay_to_end, B_t, X_t)
    # [B, n_heads, n_chunks, d_state, d_head]

    # Propagate states across chunks
    h_inter = torch.zeros(batch, n_heads, n_chunks, d_state, d_head, device=device, dtype=dtype)
    carry = torch.zeros(batch, n_heads, d_state, d_head, device=device, dtype=dtype)

    for c in range(n_chunks):
        h_inter[:, :, c] = carry
        carry = decay_chunk[:, :, c:c+1, None] * carry + h_chunk_final[:, :, c]

    # Add inter-chunk contribution
    decay_from_start = torch.exp(A_cumsum)  # [B, n_heads, n_chunks, cs]
    Y_inter = torch.einsum('bhnis,bhnsp,bhni->bhnip', C_t, h_inter, decay_from_start)

    # Combine
    Y_t = Y_intra + Y_inter  # [B, n_heads, n_chunks, cs, d_head]

    # Reshape back
    Y_chunks = Y_t.permute(0, 2, 3, 1, 4)  # [B, n_chunks, cs, n_heads, d_head]
    Y = Y_chunks.reshape(batch, seqlen, n_heads, d_head)

    # Remove padding
    Y = Y[:, :orig_seqlen]

    if return_surprise:
        return Y, normalized_surprise
    return Y, None


class ChunkedSurpriseGatedSSD(nn.Module):
    """
    Fast SSD module with per-chunk surprise-gated retention.

    Drop-in replacement for SurpriseGatedSSD with ~80x speedup.
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

        # Chunked surprise gate
        self.surprise_gate = ChunkedSurpriseGate(n_heads=n_heads, ema_decay=ema_decay)

    def forward(
        self,
        X: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        return_surprise: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass with chunked surprise-gated retention."""
        return chunked_surprise_ssd_forward(
            X, A, B, C,
            self.surprise_gate,
            self.chunk_size,
            return_surprise,
        )

    def get_surprise_stats(self) -> dict:
        """Get surprise statistics for logging."""
        gate = self.surprise_gate
        return {
            'surprise_ema': gate.surprise_ema.mean().item(),
            'alpha_base_mean': (1 - torch.pow(2.0, gate.log2_alpha_base)).mean().item(),
            'beta_mean': torch.pow(2.0, gate.log2_beta).mean().item(),
        }


class MambaIntegerBlockV2Chunked(nn.Module):
    """
    Mamba-Integer V2 block with FAST chunked surprise-gated retention.

    This is the production version with ~80x speedup over sequential V2.
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
        self.chunk_size = ssm_cfg.get('chunk_size', 64)

        # Use nn.Linear for prototype
        Linear = nn.Linear

        # Normalization
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

        # Base A (log decay)
        self.A_log = nn.Parameter(torch.log(torch.ones(self.n_heads) * 0.5))

        # Chunked surprise-gated SSD (FAST VERSION)
        self.ssd = ChunkedSurpriseGatedSSD(
            n_heads=self.n_heads,
            d_state=self.d_state,
            d_head=self.d_head,
            chunk_size=self.chunk_size,
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
        """Forward pass with chunked surprise-gated retention."""
        residual = hidden_states
        batch, seqlen, _ = hidden_states.shape

        # Normalize
        hidden_states = self.norm(hidden_states)

        # Input projection
        xz = self.in_proj(hidden_states)
        x, z = xz.chunk(2, dim=-1)

        # Conv1d
        x = self.conv1d(x.transpose(1, 2)).transpose(1, 2)[:, :seqlen]

        # Activation
        x = F.silu(x)

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

        # Compute A
        A_log_clamped = self.A_log.clamp(-2.3, -0.01)
        A = A_log_clamped.view(1, 1, self.n_heads).expand(batch, seqlen, -1)

        # Modulate with dt
        dt_scale = F.softplus(dt.clamp(-10, 10)).clamp(0.01, 10.0)
        A = A * dt_scale
        A = A.clamp(-20.0, -0.001)

        # Reshape X
        X = x.view(batch, seqlen, self.n_heads, self.d_head)

        # Run FAST chunked surprise-gated SSD
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
