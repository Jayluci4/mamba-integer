"""
Mamba-Integer V2 Redesign: Correct Implementation

Key fixes from first-principles analysis:
1. Proper convex combination: h = α * h_prev + (1-α) * update
   - Both decay AND input are modulated by surprise
2. Gradient-based surprise signal (like Titans)
   - ||∇_h L||² measures actual importance for prediction
3. More aggressive consolidation
   - Lower thresholds for better memory utilization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List, Dict

import sys
sys.path.insert(0, '/home/jayantlohia16/mamba-integer/src')

from triton_kernels.ssd_multihead import build_causal_decay_matrix_multihead


# =============================================================================
# Gradient-Based Surprise (like Titans)
# =============================================================================

class GradientSurpriseGate(nn.Module):
    """
    Compute surprise using gradient magnitude (like Titans).

    Key insight: ||∇_h L||² measures how important the state is for prediction.
    High gradient = state matters a lot = surprising/important
    Low gradient = state doesn't matter = predictable/unimportant

    This is more semantically meaningful than state prediction error.
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
        self.log2_alpha_base = nn.Parameter(torch.ones(n_heads) * -0.15)

        # Learned surprise scaling
        self.log2_beta = nn.Parameter(torch.zeros(n_heads))

        # Running statistics
        self.register_buffer('surprise_ema', torch.ones(n_heads))
        self.register_buffer('step_count', torch.tensor(0))

        # Store gradients for surprise computation
        self.last_state_grad = None

    def compute_gradient_surprise(
        self,
        state: torch.Tensor,  # [B, n_heads, d_state, d_head]
    ) -> torch.Tensor:
        """
        Compute surprise from gradient magnitude.

        Must be called AFTER backward pass so gradients are available.
        Returns [B, n_heads] surprise scores.
        """
        if state.grad is not None:
            # ||∇_h L||² per head
            grad_magnitude = state.grad.pow(2).mean(dim=(-1, -2))  # [B, n_heads]
            self.last_state_grad = grad_magnitude.detach()
            return grad_magnitude
        elif self.last_state_grad is not None:
            # Use cached gradient from previous step
            return self.last_state_grad
        else:
            # No gradient available, return zeros
            return torch.zeros(
                state.shape[0], self.n_heads,
                device=state.device, dtype=state.dtype
            )

    def compute_state_prediction_error(
        self,
        h_actual: torch.Tensor,
        h_prev: torch.Tensor,
        decay: torch.Tensor,
    ) -> torch.Tensor:
        """Fallback: state prediction error when gradients unavailable."""
        decay_expanded = decay
        for _ in range(h_prev.dim() - decay.dim()):
            decay_expanded = decay_expanded.unsqueeze(-1)

        h_expected = decay_expanded * h_prev
        error = (h_actual - h_expected).pow(2)

        while error.dim() > 2:
            error = error.mean(dim=-1)

        return error

    def forward(
        self,
        surprise: torch.Tensor,  # [B, n_heads] or [B, n_heads, n_chunks]
        training: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute adaptive retention coefficient α.

        Returns:
            alpha: [B, n_heads, ...] retention coefficient in [0, 1]
                   High surprise → high α → keep more, update less
            normalized_surprise: for logging
        """
        if surprise.dim() == 2:
            surprise = surprise.unsqueeze(-1)

        # Update EMA
        if training:
            with torch.no_grad():
                mean_surprise = surprise.mean(dim=(0, 2))
                self.surprise_ema = self.ema_decay * self.surprise_ema + (1 - self.ema_decay) * mean_surprise
                self.step_count += 1

        # Normalize
        normalized = surprise / (self.surprise_ema.view(1, -1, 1) + 1e-6)

        # Base retention
        log2_clamped = self.log2_alpha_base.clamp(-3.32, -0.015)
        alpha_base = 1.0 - torch.pow(2.0, log2_clamped)

        # Surprise scaling
        beta = torch.pow(2.0, self.log2_beta.clamp(-2, 2))

        # Compute boost
        boost = torch.tanh(beta.view(1, -1, 1) * normalized)
        boost_positive = F.relu(boost)

        # Final α: high surprise → high α
        alpha = alpha_base.view(1, -1, 1) + (1 - alpha_base.view(1, -1, 1)) * boost_positive
        alpha = alpha.clamp(0.01, 0.99)

        return alpha, normalized


# =============================================================================
# Convex Combination SSD Forward
# =============================================================================

def convex_ssd_chunk_forward(
    X_chunk: torch.Tensor,     # [B, n_heads, cs, d_head]
    A_chunk: torch.Tensor,     # [B, n_heads, cs] log decay
    B_chunk: torch.Tensor,     # [B, n_heads, cs, d_state]
    C_chunk: torch.Tensor,     # [B, n_heads, cs, d_state]
    h_prev: torch.Tensor,      # [B, n_heads, d_state, d_head]
    alpha: float = 0.5,        # Retention coefficient from surprise
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    SSD chunk forward with PROPER CONVEX COMBINATION.

    Key change from original:
    - Original: h = decay * h_prev + input (only decay modulated)
    - Fixed: h = α * (decay * h_prev) + (1-α) * input (both modulated)

    This ensures:
    - High α (high surprise) → keep more of h_prev, add less new input
    - Low α (low surprise) → decay h_prev more, add more new input

    Wait, that's backwards for what we want!

    Let me reconsider: "High surprise → STORE IT"

    The surprising thing is the NEW INPUT, not the old state.
    So high surprise should mean: keep the new surprising input.

    Correct interpretation:
    - High surprise → low α → decay old state, keep surprising input
    - Low surprise → high α → keep old state, ignore predictable input

    So α should be INVERSELY related to surprise:
    α = 1 - f(surprise)

    Actually, let me re-read the docs again...

    "High surprise → STORE IT (low decay, high retention)"

    "Store it" refers to the surprising information.
    "High retention" means keep this information for longer.

    So the flow is:
    1. Input comes in, causes state change
    2. We detect this was surprising
    3. We RETAIN this new state (don't let it decay)

    The confusion is: we can't know input is surprising BEFORE processing it.

    With delayed causality:
    1. Process chunk c-1, get h[c-1]
    2. Compute surprise = how much h changed
    3. High surprise means h[c-1] contains important new info
    4. For chunk c, we want h[c-1] to persist → low decay

    So the current approach IS correct:
    - High surprise from c-1 → low decay in c → h[c-1] persists

    But the convex combination adds another dimension:
    - α controls both decay AND input scaling
    - High α → preserve more of h_prev, add less new input

    For our use case:
    - After surprising chunk: α should be HIGH (preserve the surprising state)
    - After predictable chunk: α should be LOW (allow new input to modify state)

    So we want: surprise → α (positive correlation)

    Let me implement this correctly.
    """
    B, n_heads, cs, d_head = X_chunk.shape
    d_state = B_chunk.shape[-1]
    device = X_chunk.device
    dtype = X_chunk.dtype

    # Compute base decay
    A_cumsum = torch.cumsum(A_chunk, dim=-1)

    # Build L matrix
    L = build_causal_decay_matrix_multihead(A_cumsum.unsqueeze(2))[:, :, 0]

    # Intra-chunk computation: Y_intra = (L * CB) @ X
    CB = torch.einsum('bhid,bhjd->bhij', C_chunk, B_chunk)
    L_CB = L * CB
    Y_intra = torch.einsum('bhij,bhjd->bhid', L_CB, X_chunk)

    # Compute chunk contribution to state
    decay_to_end = torch.exp(A_cumsum[:, :, -1:] - A_cumsum)
    h_chunk_contrib = torch.einsum('bhts,bhtd,bht->bhsd', B_chunk, X_chunk, decay_to_end)

    # Compute decay of previous state
    decay_total = torch.exp(A_cumsum[:, :, -1])  # [B, n_heads]
    h_prev_decayed = decay_total.unsqueeze(-1).unsqueeze(-1) * h_prev

    # CONVEX COMBINATION: h = α * h_prev_decayed + (1 - α) * h_chunk_contrib
    # α controls the balance between keeping old state vs adding new input
    h_end = alpha * h_prev_decayed + (1 - alpha) * h_chunk_contrib

    # Inter-chunk contribution to output
    decay_from_start = torch.exp(A_cumsum)
    # Scale inter-chunk by α (preserve old state contribution)
    Y_inter = alpha * torch.einsum('bhis,bhsd,bhi->bhid', C_chunk, h_prev, decay_from_start)

    # Scale intra-chunk by (1-α) (new input contribution)
    Y_chunk = (1 - alpha) * Y_intra + Y_inter

    return Y_chunk, h_end, decay_total


# =============================================================================
# Aggressive Consolidation
# =============================================================================

class AggressiveConsolidationTrigger(nn.Module):
    """
    More aggressive consolidation for better memory utilization.

    Original thresholds were too conservative:
    - capacity: 0.9, surprise: 2.0, time: 256
    - Result: only 1 consolidation per sequence

    New thresholds:
    - capacity: 0.5 (consolidate when buffer half full)
    - surprise: 0.5 (consolidate on moderate surprise)
    - time: 32 (consolidate every 32 tokens)
    """

    def __init__(
        self,
        capacity_threshold: float = 0.5,
        surprise_threshold: float = 0.5,
        time_threshold: int = 32,
    ):
        super().__init__()
        self.capacity_threshold = capacity_threshold
        self.surprise_threshold = surprise_threshold
        self.time_threshold = time_threshold

        self.register_buffer('cumulative_surprise', torch.tensor(0.0))
        self.register_buffer('tokens_since_consolidation', torch.tensor(0))
        self.register_buffer('consolidation_count', torch.tensor(0))

    def should_consolidate(
        self,
        surprise_score: float,
        buffer_fullness: float,
    ) -> Tuple[bool, List[str]]:
        # Use in-place operations that don't affect gradients
        with torch.no_grad():
            self.cumulative_surprise.add_(surprise_score)
            self.tokens_since_consolidation.add_(1)

        triggers = []

        if buffer_fullness > self.capacity_threshold:
            triggers.append('capacity')

        if self.cumulative_surprise.item() > self.surprise_threshold:
            triggers.append('surprise')

        if self.tokens_since_consolidation.item() > self.time_threshold:
            triggers.append('time')

        if triggers:
            self.reset()
            with torch.no_grad():
                self.consolidation_count.add_(1)
            return True, triggers

        return False, []

    def reset(self):
        with torch.no_grad():
            self.cumulative_surprise.zero_()
            self.tokens_since_consolidation.zero_()


# =============================================================================
# Multi-Timescale Memory (unchanged but with better integration)
# =============================================================================

class MultiTimescaleMemoryV2(nn.Module):
    """
    4-level memory hierarchy with better utilization.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_state: int,
        d_head: int,
        n_persistent: int = 64,
        chunk_memory_size: int = 128,
        chunk_retention: float = 0.95,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_state = d_state
        self.d_head = d_head
        self.n_persistent = n_persistent
        self.chunk_memory_size = chunk_memory_size
        self.chunk_retention = chunk_retention

        # L2: Chunk memory
        self.register_buffer(
            'chunk_memory',
            torch.zeros(n_heads, chunk_memory_size, d_state)
        )
        self.register_buffer('chunk_memory_ptr', torch.tensor(0))
        self.register_buffer('chunk_memory_filled', torch.tensor(0))

        # L3: Persistent memory
        self.persistent_memory = nn.Parameter(
            torch.randn(n_heads, n_persistent, d_state) * 0.02
        )

        # Memory integration
        self.memory_query_proj = nn.Linear(d_model, n_heads * d_state)
        self.memory_out_proj = nn.Linear(n_heads * d_state, d_model)
        self.memory_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )

    def write_chunk_summary(
        self,
        chunk_state: torch.Tensor,
        importance: torch.Tensor,
    ):
        """Write with exponential moving average."""
        with torch.no_grad():
            compressed = chunk_state.mean(dim=-1)
            importance_norm = F.softmax(importance, dim=0)
            weighted = (compressed * importance_norm.unsqueeze(-1)).sum(dim=0)

            ptr = self.chunk_memory_ptr.item()
            self.chunk_memory[:, ptr] = (
                self.chunk_retention * self.chunk_memory[:, ptr] +
                (1 - self.chunk_retention) * weighted
            )

            self.chunk_memory_ptr = (self.chunk_memory_ptr + 1) % self.chunk_memory_size
            self.chunk_memory_filled.clamp_(max=self.chunk_memory_size)
            self.chunk_memory_filled.add_(1).clamp_(max=self.chunk_memory_size)

    def read_memory(self, query: torch.Tensor) -> torch.Tensor:
        batch, seqlen, _ = query.shape

        q = self.memory_query_proj(query)
        q = q.view(batch, seqlen, self.n_heads, self.d_state)

        filled = max(1, self.chunk_memory_filled.item())
        chunk_mem = self.chunk_memory[:, :filled]
        persist_mem = self.persistent_memory

        combined_mem = torch.cat([chunk_mem, persist_mem], dim=1)

        attn = torch.einsum('blhd,hmd->blhm', q, combined_mem)
        attn = attn / math.sqrt(self.d_state)
        attn = F.softmax(attn, dim=-1)

        retrieved = torch.einsum('blhm,hmd->blhd', attn, combined_mem)
        retrieved = retrieved.reshape(batch, seqlen, -1)

        memory_out = self.memory_out_proj(retrieved)
        return memory_out

    def integrate(self, x: torch.Tensor, memory_out: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([x, memory_out], dim=-1)
        gate = self.memory_gate(combined)
        return x + gate * memory_out


# =============================================================================
# Redesigned V2 Block
# =============================================================================

class MambaIntegerBlockV2Redesign(nn.Module):
    """
    Redesigned V2 block with:
    1. Gradient-based surprise (like Titans)
    2. Proper convex combination
    3. Aggressive consolidation
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

        # Input projections
        self.norm = nn.RMSNorm(d_model)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, 4,
            groups=self.d_inner, padding=3
        )

        # SSM projections
        proj_size = self.n_heads * (1 + 2 * self.d_state)
        self.x_proj = nn.Linear(self.d_inner, proj_size)
        self.A_log = nn.Parameter(torch.log(torch.ones(self.n_heads) * 0.5))

        # Gradient-based surprise gate
        self.surprise_gate = GradientSurpriseGate(
            n_heads=self.n_heads,
            d_state=self.d_state,
        )

        # Multi-timescale memory
        self.multi_memory = MultiTimescaleMemoryV2(
            d_model=d_model,
            n_heads=self.n_heads,
            d_state=self.d_state,
            d_head=self.d_head,
        )

        # Aggressive consolidation
        self.consolidation_trigger = AggressiveConsolidationTrigger(
            capacity_threshold=ssm_cfg.get('capacity_threshold', 0.5),
            surprise_threshold=ssm_cfg.get('surprise_threshold', 0.5),
            time_threshold=ssm_cfg.get('time_threshold', 32),
        )

        # Output
        self.out_proj = nn.Linear(self.d_inner, d_model)

        n_layer = config.get('n_layer', 24)
        self.res_gate = nn.Parameter(torch.ones(1) / math.sqrt(2 * n_layer))

        # State tracking
        self.last_surprise = None
        self.last_alpha = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        return_surprise: bool = False,
        return_stats: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Dict]]:
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

        # Forward with convex combination
        Y, surprise, stats = self._convex_forward(
            X, A, B_ssm, C_ssm, hidden_states
        )

        self.last_surprise = surprise

        y = Y.reshape(batch, seqlen, self.d_inner)
        y = y.clamp(-100, 100)
        y = y * torch.sigmoid(z)

        out = self.out_proj(y)
        out = out.clamp(-100, 100)

        output = residual + out * self.res_gate

        if return_stats:
            return output, surprise, stats
        elif return_surprise:
            return output, surprise
        return output, None

    def _convex_forward(
        self,
        X: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        batch, seqlen, n_heads, d_head = X.shape
        d_state = B.shape[-1]
        device = X.device
        dtype = X.dtype

        # Pad
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

        # Reshape
        X_chunks = X.view(batch, n_chunks, self.chunk_size, n_heads, d_head).permute(0, 3, 1, 2, 4)
        A_chunks = A.view(batch, n_chunks, self.chunk_size, n_heads).permute(0, 3, 1, 2)
        B_chunks = B.view(batch, n_chunks, self.chunk_size, n_heads, d_state).permute(0, 3, 1, 2, 4)
        C_chunks = C.view(batch, n_chunks, self.chunk_size, n_heads, d_state).permute(0, 3, 1, 2, 4)

        h_prev = torch.zeros(batch, n_heads, d_state, d_head, device=device, dtype=dtype)

        Y_chunks_list = []
        h_history = [h_prev]
        decay_history = []
        surprise_list = []
        alpha_list = []
        consolidation_events = []
        state_buffer = []

        for c in range(n_chunks):
            X_c = X_chunks[:, :, c]
            A_c = A_chunks[:, :, c]
            B_c = B_chunks[:, :, c]
            C_c = C_chunks[:, :, c]

            # Compute surprise and alpha
            if c == 0:
                alpha = 0.5  # Neutral for first chunk
                surprise_c = torch.zeros(batch, n_heads, device=device, dtype=dtype)
            else:
                # Use state prediction error (gradient-based needs backward pass)
                h_actual = h_history[-1]
                h_before = h_history[-2]
                decay_prev = decay_history[-1]

                surprise_c = self.surprise_gate.compute_state_prediction_error(
                    h_actual, h_before, decay_prev
                )

                alpha_tensor, _ = self.surprise_gate(surprise_c.unsqueeze(-1))
                alpha = alpha_tensor.mean().item()

            surprise_list.append(surprise_c)
            alpha_list.append(alpha)

            # Process with convex combination
            Y_c, h_end, decay_total = convex_ssd_chunk_forward(
                X_c, A_c, B_c, C_c, h_prev, alpha
            )

            Y_chunks_list.append(Y_c)
            h_history.append(h_end)
            decay_history.append(decay_total)
            h_prev = h_end

            state_buffer.append(h_end.detach())

            # Consolidation check
            buffer_fullness = len(state_buffer) / 4
            should_consolidate, triggers = self.consolidation_trigger.should_consolidate(
                surprise_c.mean().item(),
                buffer_fullness,
            )

            if should_consolidate and len(state_buffer) > 0:
                avg_state = torch.stack(state_buffer, dim=0).mean(dim=0)
                avg_importance = torch.stack(surprise_list[-len(state_buffer):], dim=0).mean(dim=0)

                self.multi_memory.write_chunk_summary(avg_state, avg_importance)
                consolidation_events.append({'chunk': c, 'triggers': triggers})
                state_buffer = []

        # Stack outputs
        Y_chunks = torch.stack(Y_chunks_list, dim=2)
        Y = Y_chunks.permute(0, 2, 3, 1, 4).reshape(batch, seqlen, n_heads, d_head)
        Y = Y[:, :orig_seqlen]

        # Memory integration
        memory_out = self.multi_memory.read_memory(hidden_states[:, :orig_seqlen])
        Y_flat = Y.reshape(batch, orig_seqlen, -1)[:, :, :hidden_states.shape[-1]]
        if Y_flat.shape[-1] < hidden_states.shape[-1]:
            Y_flat = F.pad(Y_flat, (0, hidden_states.shape[-1] - Y_flat.shape[-1]))
        Y_integrated = self.multi_memory.integrate(Y_flat, memory_out)
        Y = Y_integrated.view(batch, orig_seqlen, n_heads, d_head)

        surprise = torch.stack(surprise_list, dim=-1)

        stats = {
            'layer_idx': self.layer_idx,
            'n_chunks': n_chunks,
            'mean_surprise': surprise.mean().item(),
            'mean_alpha': sum(alpha_list) / len(alpha_list),
            'consolidation_events': len(consolidation_events),
            'chunk_memory_filled': self.multi_memory.chunk_memory_filled.item(),
        }

        return Y, surprise, stats

    def get_surprise_stats(self) -> Dict:
        gate = self.surprise_gate
        return {
            'layer_idx': self.layer_idx,
            'surprise_ema': gate.surprise_ema.mean().item(),
            'alpha_base_mean': (1 - torch.pow(2.0, gate.log2_alpha_base)).mean().item(),
            'consolidation_count': self.consolidation_trigger.consolidation_count.item(),
            'chunk_memory_filled': self.multi_memory.chunk_memory_filled.item(),
        }


# =============================================================================
# Redesigned V2 Model
# =============================================================================

class MambaIntegerModelV2Redesign(nn.Module):
    """Redesigned V2 model with correct implementation."""

    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        vocab_size = config['vocab_size']
        d_model = config['d_model']
        n_layer = config['n_layer']

        self.embed = nn.Embedding(vocab_size, d_model)

        self.blocks = nn.ModuleList([
            MambaIntegerBlockV2Redesign(config, layer_idx=i)
            for i in range(n_layer)
        ])

        self.norm_f = nn.RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
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
                'per_layer': [s.mean().item() for s in surprise_all],
            }
            return logits, surprise_stats

        return logits, None

    def get_all_stats(self) -> List[Dict]:
        return [block.get_surprise_stats() for block in self.blocks]

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    'GradientSurpriseGate',
    'AggressiveConsolidationTrigger',
    'MultiTimescaleMemoryV2',
    'MambaIntegerBlockV2Redesign',
    'MambaIntegerModelV2Redesign',
    'convex_ssd_chunk_forward',
]
