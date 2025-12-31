"""
Memory-Efficient Mamba-2 SSD (State Space Duality) with Multi-Head Architecture

S1 FIX: Implements Mamba-2's memory-efficient SSD with:
- Scalar A per head (not matrix per position)
- L matrix: [B, n_heads, n_chunks, cs, cs] instead of [B, n_chunks, D, cs, cs]
- Memory reduction: 6.4 GB -> 6.3 MB (1000x improvement)

Reference: https://tridao.me/blog/2024/mamba2-part3-algorithm/
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def segsum(x):
    """Compute segment cumulative sums for log-space products.

    Used to compute L[i,j] = exp(A_cumsum[i] - A_cumsum[j])

    Args:
        x: [..., T] log-decay values

    Returns:
        x_segsum: [..., T, T] segment sums where x_segsum[i,j] = sum(x[j+1:i+1])
    """
    T = x.size(-1)
    device = x.device
    dtype = x.dtype

    # Expand to [..., T, T]
    x_expanded = x.unsqueeze(-1).expand(*x.shape, T)  # [..., T, T]

    # Create lower triangular mask (excluding diagonal)
    mask = torch.tril(torch.ones(T, T, device=device, dtype=dtype), diagonal=-1)
    x_masked = x_expanded * mask

    # Cumulative sum along rows
    x_segsum = torch.cumsum(x_masked, dim=-2)

    # Apply causal mask (including diagonal)
    causal_mask = torch.tril(torch.ones(T, T, device=device, dtype=dtype), diagonal=0)
    x_segsum = x_segsum.masked_fill(~causal_mask.bool(), float('-inf'))

    return x_segsum


def build_causal_decay_matrix_multihead(A_cumsum):
    """Build causal decay matrix L from cumulative log-decay.

    Args:
        A_cumsum: [B, n_heads, n_chunks, chunk_size] cumulative log-decay

    Returns:
        L: [B, n_heads, n_chunks, chunk_size, chunk_size] causal decay matrix
    """
    B, n_heads, n_chunks, cs = A_cumsum.shape
    device = A_cumsum.device
    dtype = A_cumsum.dtype

    # L[i,j] = exp(A_cumsum[i] - A_cumsum[j]) for i >= j
    idx_i = torch.arange(cs, device=device).view(1, 1, 1, cs, 1)
    idx_j = torch.arange(cs, device=device).view(1, 1, 1, 1, cs)

    # Expand A_cumsum for broadcasting
    A_i = A_cumsum.unsqueeze(-1)  # [B, n_heads, n_chunks, cs, 1]
    A_j = A_cumsum.unsqueeze(-2)  # [B, n_heads, n_chunks, 1, cs]

    # Compute log differences
    log_diff = A_i - A_j  # [B, n_heads, n_chunks, cs, cs]

    # Apply causal mask BEFORE exp to avoid overflow
    # For upper triangle (i < j), log_diff is positive and large
    # Set upper triangle to -inf before exp so exp(-inf) = 0
    causal_mask = (idx_i >= idx_j)
    log_diff = log_diff.masked_fill(~causal_mask, float('-inf'))

    # Clamp log_diff for numerical stability before exp
    log_diff = torch.clamp(log_diff, min=-100.0, max=100.0)

    # Exponentiate (upper triangle becomes 0)
    L = torch.exp(log_diff)

    # Ensure no NaN/Inf
    L = torch.nan_to_num(L, nan=0.0, posinf=1.0, neginf=0.0)

    return L


def ssd_multihead_forward(X, A, B, C, chunk_size=64):
    """Memory-efficient SSD forward with multi-head architecture.

    Args:
        X: [B, L, n_heads, d_head] input
        A: [B, L, n_heads] scalar decay per head (log space, negative)
        B: [B, L, n_heads, d_state] input projection
        C: [B, L, n_heads, d_state] output projection
        chunk_size: Chunk size for matmul (default 64)

    Returns:
        Y: [B, L, n_heads, d_head] output
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

    # Reshape to chunks: [B, n_chunks, chunk_size, n_heads, ...]
    X_chunks = X.view(batch, n_chunks, chunk_size, n_heads, d_head)
    A_chunks = A.view(batch, n_chunks, chunk_size, n_heads)
    B_chunks = B.view(batch, n_chunks, chunk_size, n_heads, d_state)
    C_chunks = C.view(batch, n_chunks, chunk_size, n_heads, d_state)

    # Transpose A to [B, n_heads, n_chunks, chunk_size] for cumsum
    A_t = A_chunks.permute(0, 3, 1, 2)  # [B, n_heads, n_chunks, cs]
    A_cumsum = torch.cumsum(A_t, dim=-1)  # Cumulative sum within chunks

    # Initialize output
    Y_output = torch.zeros_like(X_chunks)

    # Process all heads together (memory-efficient due to scalar A per head)
    # L matrix: [B, n_heads, n_chunks, cs, cs] = 6.3 MB for B=2, n_heads=24, n_chunks=8, cs=64
    L = build_causal_decay_matrix_multihead(A_cumsum)  # [B, n_heads, n_chunks, cs, cs]

    # Permute for efficient einsum
    # X: [B, n_chunks, cs, n_heads, d_head] -> [B, n_heads, n_chunks, cs, d_head]
    X_t = X_chunks.permute(0, 3, 1, 2, 4)  # [B, n_heads, n_chunks, cs, d_head]
    B_t = B_chunks.permute(0, 3, 1, 2, 4)  # [B, n_heads, n_chunks, cs, d_state]
    C_t = C_chunks.permute(0, 3, 1, 2, 4)  # [B, n_heads, n_chunks, cs, d_state]

    # Step 1: Compute weighted input (for convex combination)
    # decay_weights[i] = L[i,i] = 1 (diagonal)
    # input_weights[i,j] = (1 - exp(A[j])) * L[i,j] for j < i
    # For simplicity, use standard formulation: contribution from u[j] at position i

    # Step 2: Intra-chunk computation via matmul
    # Y_intra[i] = sum_j L[i,j] * B[j] * X[j]
    # Compute B * X: [B, n_heads, n_chunks, cs, d_state] * [B, n_heads, n_chunks, cs, d_head]
    # -> need to do this properly

    # Simplified: compute h directly via matmul with L
    # h_intra = L @ X (treating as linear system)
    # Then output = h @ C

    # For Mamba-2 SSD:
    # h[i] = sum_{j<=i} L[i,j] * B[j].T @ X[j]  (state accumulation)
    # Y[i] = C[i] @ h[i]

    # Using einsum for clarity (will optimize with Triton later)
    # BX = B.T @ X: [B, n_heads, n_chunks, cs, d_state, d_head] (outer product)
    # But this is expensive. Better to use the SSD identity:
    # Y = C @ (sum over j: L[i,j] * B[j].T @ X[j])
    #   = sum over j: L[i,j] * (C[i] @ B[j].T) @ X[j]
    #   = sum over j: L[i,j] * (C[i] . B[j]) * X[j]  (when d_state is small)

    # Compute C @ B.T for each position pair within chunk
    # CB[i,j] = sum_s C[i,s] * B[j,s] = dot(C[i], B[j])
    # CB: [B, n_heads, n_chunks, cs_i, cs_j]
    CB = torch.einsum('bhnid,bhnjd->bhnij', C_t, B_t)  # [B, n_heads, n_chunks, cs, cs]

    # Combine with decay matrix: L * CB
    # Then apply to X
    L_CB = L * CB  # [B, n_heads, n_chunks, cs, cs]

    # Y_intra = (L * CB) @ X
    Y_intra = torch.einsum('bhnij,bhnjd->bhnid', L_CB, X_t)  # [B, n_heads, n_chunks, cs, d_head]

    # Step 3: Inter-chunk state propagation
    # Need to propagate state across chunk boundaries

    # Compute final state of each chunk
    # h_final[c] = sum_j L[-1,j] * B[j].T @ X[j] within chunk c
    # This is already captured in Y_intra at the last position

    # Decay product for each chunk: exp(sum of log-decay across chunk)
    decay_chunk = torch.exp(A_t.sum(dim=-1))  # [B, n_heads, n_chunks]

    # State at end of each chunk: accumulate B.T @ X weighted by decay
    # For convex combination, state is: h = decay * h_prev + (1-decay) * u
    # Final state h_final = running accumulation

    # Compute per-chunk final states
    # h_chunk_final: [B, n_heads, n_chunks, d_state, d_head]
    # h_chunk_final[c] = sum_t decay_from_t_to_end[t] * B[t].T @ X[t]

    # Decay from position t to end of chunk
    decay_to_end = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)  # [B, n_heads, n_chunks, cs]

    # Weighted B and X
    # h_chunk_final = sum_t decay_to_end[t] * outer(B[t], X[t])
    # = einsum('bhnc,bhncs,bhncd->bhnsd', decay_to_end, B_t, X_t)
    h_chunk_final = torch.einsum('bhnc,bhncs,bhncd->bhnsd', decay_to_end, B_t, X_t)
    # Shape: [B, n_heads, n_chunks, d_state, d_head]

    # Propagate states across chunks (sequential, but only n_chunks iterations)
    h_inter = torch.zeros(batch, n_heads, n_chunks, d_state, d_head, device=device, dtype=dtype)
    carry = torch.zeros(batch, n_heads, d_state, d_head, device=device, dtype=dtype)

    for c in range(n_chunks):
        h_inter[:, :, c] = carry
        # Update carry: new_carry = decay_chunk[c] * carry + h_chunk_final[c]
        carry = decay_chunk[:, :, c:c+1, None] * carry + h_chunk_final[:, :, c]

    # Step 4: Add inter-chunk contribution to intra-chunk result
    # For position i in chunk c, add C[i] @ h_inter[c] * decay_from_start[i]

    # Decay from start of chunk to position i
    decay_from_start = torch.exp(A_cumsum)  # [B, n_heads, n_chunks, cs]

    # Y_inter[i] = C[i] @ h_inter[c] * decay_from_start[i]
    # C_t: [B, n_heads, n_chunks, cs, d_state]
    # h_inter: [B, n_heads, n_chunks, d_state, d_head]
    # decay_from_start: [B, n_heads, n_chunks, cs]
    # Y_inter[i,p] = sum_s C[i,s] * h_inter[s,p] * decay_from_start[i]
    # = einsum('bhnis,bhnsp,bhni->bhnip', C_t, h_inter, decay_from_start)
    Y_inter = torch.einsum('bhnis,bhnsp,bhni->bhnip', C_t, h_inter, decay_from_start)

    # Combine
    Y_t = Y_intra + Y_inter  # [B, n_heads, n_chunks, cs, d_head]

    # Permute back: [B, n_heads, n_chunks, cs, d_head] -> [B, n_chunks, cs, n_heads, d_head]
    Y_chunks = Y_t.permute(0, 2, 3, 1, 4)

    # Reshape: [B, L, n_heads, d_head]
    Y = Y_chunks.reshape(batch, seqlen, n_heads, d_head)

    # Remove padding
    Y = Y[:, :orig_seqlen]

    return Y


class SSDMultiheadFunction(torch.autograd.Function):
    """Autograd function for memory-efficient multi-head SSD."""

    @staticmethod
    def forward(ctx, X, A, B, C, chunk_size=64):
        """Forward pass.

        Args:
            X: [B, L, n_heads, d_head] input
            A: [B, L, n_heads] scalar decay per head (log space)
            B: [B, L, n_heads, d_state] input projection
            C: [B, L, n_heads, d_state] output projection
        """
        Y = ssd_multihead_forward(X, A, B, C, chunk_size)
        ctx.save_for_backward(X, A, B, C)
        ctx.chunk_size = chunk_size
        return Y

    @staticmethod
    def backward(ctx, grad_Y):
        """Memory-efficient backward pass - avoids materializing full h tensor."""
        X, A, B, C = ctx.saved_tensors
        chunk_size = ctx.chunk_size

        batch, seqlen, n_heads, d_head = X.shape
        d_state = B.shape[-1]
        device = X.device
        orig_dtype = X.dtype

        # Cast to float32 for stability
        X = X.float()
        A = A.float()
        B = B.float()
        C = C.float()
        grad_Y = grad_Y.float()

        # Pad sequences
        orig_seqlen = seqlen
        if seqlen % chunk_size != 0:
            pad_len = chunk_size - (seqlen % chunk_size)
            X = F.pad(X, (0, 0, 0, 0, 0, pad_len), value=0.0)
            A = F.pad(A, (0, 0, 0, pad_len), value=0.0)
            B = F.pad(B, (0, 0, 0, 0, 0, pad_len), value=0.0)
            C = F.pad(C, (0, 0, 0, 0, 0, pad_len), value=0.0)
            grad_Y = F.pad(grad_Y, (0, 0, 0, 0, 0, pad_len), value=0.0)
            seqlen = X.shape[1]

        n_chunks = seqlen // chunk_size
        cs = chunk_size

        # Reshape to [B, n_heads, n_chunks, cs, ...]
        X_t = X.view(batch, n_chunks, cs, n_heads, d_head).permute(0, 3, 1, 2, 4)
        A_t = A.view(batch, n_chunks, cs, n_heads).permute(0, 3, 1, 2)
        B_t = B.view(batch, n_chunks, cs, n_heads, d_state).permute(0, 3, 1, 2, 4)
        C_t = C.view(batch, n_chunks, cs, n_heads, d_state).permute(0, 3, 1, 2, 4)
        grad_Y_t = grad_Y.view(batch, n_chunks, cs, n_heads, d_head).permute(0, 3, 1, 2, 4)

        # Build L matrix for all chunks
        A_cumsum = torch.cumsum(A_t, dim=-1)
        L = build_causal_decay_matrix_multihead(A_cumsum)

        # ===== MEMORY-EFFICIENT GRADIENT COMPUTATION =====
        # Instead of computing full h tensor, use the identity:
        # Y = (L * (C @ B.T)) @ X
        # This lets us compute gradients without materializing h[d_state, d_head]

        # CB = C @ B.T: [B, n_heads, n_chunks, cs, cs]
        CB = torch.einsum('bhnis,bhnjs->bhnij', C_t, B_t)
        L_CB = L * CB

        # grad w.r.t. L_CB from Y = L_CB @ X
        # grad_L_CB = grad_Y @ X.T
        grad_L_CB = torch.einsum('bhnid,bhnjd->bhnij', grad_Y_t, X_t)

        # grad_X from Y = L_CB @ X
        # grad_X = L_CB.T @ grad_Y
        grad_X_t = torch.einsum('bhnij,bhnid->bhnjd', L_CB, grad_Y_t)

        # grad w.r.t. L and CB
        grad_L = grad_L_CB * CB
        grad_CB = grad_L_CB * L

        # grad_C from CB = C @ B.T
        # grad_C = grad_CB @ B
        grad_C_t = torch.einsum('bhnij,bhnjs->bhnis', grad_CB, B_t)

        # grad_B from CB = C @ B.T (note: CB[i,j] = sum_s C[i,s] * B[j,s])
        # grad_B[j,s] = sum_i grad_CB[i,j] * C[i,s]
        grad_B_t = torch.einsum('bhnij,bhnis->bhnjs', grad_CB, C_t)

        # grad_A from L[i,j] = exp(A_cumsum[i] - A_cumsum[j])
        M = grad_L * L

        # Vectorized grad_A computation
        M_cumsum_j = torch.cumsum(M, dim=-1)
        M_suffix = torch.flip(torch.cumsum(torch.flip(M_cumsum_j, dims=[-2]), dim=-2), dims=[-2])

        grad_A_intra = torch.zeros(batch, n_heads, n_chunks, cs, device=device, dtype=torch.float32)
        diag_indices = torch.arange(1, cs, device=device)
        grad_A_intra[:, :, :, 1:] = M_suffix[:, :, :, diag_indices, diag_indices - 1]

        # ===== INTER-CHUNK GRADIENT =====
        decay_from_start = torch.exp(A_cumsum)
        decay_chunk = torch.exp(A_t.sum(dim=-1))
        decay_to_end = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum)
        h_chunk_final = torch.einsum('bhnc,bhncs,bhncd->bhnsd', decay_to_end, B_t, X_t)

        # Sequential carry (only n_chunks iterations)
        h_inter = torch.zeros(batch, n_heads, n_chunks, d_state, d_head, device=device, dtype=torch.float32)
        carry = torch.zeros(batch, n_heads, d_state, d_head, device=device, dtype=torch.float32)
        for c in range(n_chunks):
            h_inter[:, :, c] = carry
            carry = decay_chunk[:, :, c:c+1, None] * carry + h_chunk_final[:, :, c]

        grad_decay_from_start = torch.einsum('bhnip,bhnis,bhnsp->bhni', grad_Y_t, C_t, h_inter)
        grad_A_cumsum_inter = grad_decay_from_start * decay_from_start
        grad_A_inter = torch.flip(torch.cumsum(torch.flip(grad_A_cumsum_inter, dims=[-1]), dim=-1), dims=[-1])

        grad_A_t = grad_A_intra + grad_A_inter

        # ===== RESHAPE AND RETURN =====
        grad_X_out = grad_X_t.permute(0, 2, 3, 1, 4).reshape(batch, seqlen, n_heads, d_head)
        grad_A_out = grad_A_t.permute(0, 2, 3, 1).reshape(batch, seqlen, n_heads)
        grad_B_out = grad_B_t.permute(0, 2, 3, 1, 4).reshape(batch, seqlen, n_heads, d_state)
        grad_C_out = grad_C_t.permute(0, 2, 3, 1, 4).reshape(batch, seqlen, n_heads, d_state)

        # Remove padding
        grad_X_out = grad_X_out[:, :orig_seqlen]
        grad_A_out = grad_A_out[:, :orig_seqlen]
        grad_B_out = grad_B_out[:, :orig_seqlen]
        grad_C_out = grad_C_out[:, :orig_seqlen]

        # Cast back to original dtype
        return (grad_X_out.to(orig_dtype), grad_A_out.to(orig_dtype),
                grad_B_out.to(orig_dtype), grad_C_out.to(orig_dtype), None)


def ssd_multihead(X, A, B, C, chunk_size=64):
    """Memory-efficient SSD with multi-head architecture.

    Drop-in replacement for Mamba-2 SSD that uses 1000x less memory.

    Args:
        X: [B, L, n_heads, d_head] input
        A: [B, L, n_heads] scalar decay per head (log space)
        B: [B, L, n_heads, d_state] input projection
        C: [B, L, n_heads, d_state] output projection

    Returns:
        Y: [B, L, n_heads, d_head] output
    """
    return SSDMultiheadFunction.apply(X, A, B, C, chunk_size)


class MambaIntegerBlockV2(nn.Module):
    """Mamba-2 style block with multi-head SSD for memory efficiency.

    Key differences from MambaIntegerBlock:
    1. Multi-head architecture: n_heads=24, d_head=64
    2. Scalar A per head instead of matrix A
    3. Uses memory-efficient SSD (6.3 MB vs 6.4 GB)
    4. Larger d_state=64 (was 16) enabled by SSD efficiency
    """

    def __init__(self, config, layer_idx):
        super().__init__()
        from rational_bitnet import BitLinear

        self.config = config
        d_model = config['d_model']  # 768
        ssm_cfg = config['ssm_cfg']

        # Multi-head dimensions
        self.n_heads = ssm_cfg.get('n_heads', 24)
        self.d_head = ssm_cfg.get('d_head', 64)
        self.d_state = ssm_cfg.get('d_state', 64)
        self.d_inner = self.n_heads * self.d_head  # 1536

        # Normalization
        # Import at function level to avoid circular dependency
        self.norm = _create_bitshift_norm(d_model)

        # Projections
        self.in_proj = BitLinear(d_model, self.d_inner * 2)  # x and z
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, 4,
                                groups=self.d_inner, padding=3)

        # SSM projections - output per head: dt (1), B (d_state), C (d_state)
        # dt is scalar per head, B and C are vectors of size d_state
        proj_size = self.n_heads * (1 + 2 * self.d_state)  # 24 * (1 + 128) = 3096
        self.x_proj = BitLinear(self.d_inner, proj_size)

        # Scalar A per head (Mamba-2 constraint) in log space
        # Initialize for decay around 0.5: A = log(0.5) = -0.693
        self.A_log = nn.Parameter(torch.log(torch.ones(self.n_heads) * 0.5))

        # Output projection
        self.out_proj = BitLinear(self.d_inner, d_model)

        # Residual gate (SkipInit)
        n_layer = config.get('n_layer', 24)
        self.res_gate = nn.Parameter(torch.ones(1) / math.sqrt(2 * n_layer))

        # Chunk size for SSD
        self.chunk_size = ssm_cfg.get('chunk_size', 64)

    def forward(self, hidden_states):
        residual = hidden_states
        batch, seqlen, _ = hidden_states.shape

        # Normalize
        hidden_states = self.norm(hidden_states)

        # Input projection: split into x and gate z
        xz = self.in_proj(hidden_states)
        x, z = xz.chunk(2, dim=-1)  # [B, L, d_inner]

        # Conv1d
        x = self.conv1d(x.transpose(1, 2)).transpose(1, 2)[:, :seqlen]

        # Activation (integer-only squareplus)
        x = _squareplus_activation(x)

        # SSM parameters: dt, B, C per head
        x_proj_out = self.x_proj(x)  # [B, L, n_heads * (1 + 2*d_state)]

        # Reshape to per-head: [B, L, n_heads, 1 + 2*d_state]
        x_proj_out = x_proj_out.view(batch, seqlen, self.n_heads, 1 + 2 * self.d_state)

        # Split
        dt = x_proj_out[..., 0]  # [B, L, n_heads]
        B_ssm = x_proj_out[..., 1:1+self.d_state]  # [B, L, n_heads, d_state]
        C_ssm = x_proj_out[..., 1+self.d_state:]  # [B, L, n_heads, d_state]

        # Normalize B and C for stability (unit vectors)
        B_ssm = B_ssm / (B_ssm.norm(dim=-1, keepdim=True) + 1e-6)
        C_ssm = C_ssm / (C_ssm.norm(dim=-1, keepdim=True) + 1e-6)

        # Compute A (log-decay per head, broadcasted)
        # A_log is [n_heads], need [B, L, n_heads]
        # Clamp A_log for stability (decay in range [0.1, 0.99])
        A_log_clamped = torch.clamp(self.A_log, min=-2.3, max=-0.01)
        A = A_log_clamped.view(1, 1, self.n_heads).expand(batch, seqlen, -1)

        # Modulate A with dt (softplus of dt scales the decay)
        # Clamp dt for stability
        dt_clamped = torch.clamp(dt, -10.0, 10.0)
        dt_scale = F.softplus(dt_clamped)  # [B, L, n_heads]
        dt_scale = torch.clamp(dt_scale, min=0.01, max=10.0)
        A = A * dt_scale  # Scaled log-decay
        # Clamp final A for numerical stability
        A = torch.clamp(A, min=-20.0, max=-0.001)

        # Reshape x to multi-head: [B, L, n_heads, d_head]
        X = x.view(batch, seqlen, self.n_heads, self.d_head)

        # Run SSD
        Y = ssd_multihead(X, A, B_ssm, C_ssm, self.chunk_size)
        # Y: [B, L, n_heads, d_head]

        # Reshape back: [B, L, d_inner]
        y = Y.reshape(batch, seqlen, self.d_inner)

        # Clamp for stability before gating
        y = torch.clamp(y, -100.0, 100.0)

        # Gate with z
        y = y * _sigmoid_gate(z)

        # Output projection
        out = self.out_proj(y)

        # Clamp output for stability
        out = torch.clamp(out, -100.0, 100.0)

        return residual + out * self.res_gate


class BitShiftNormV2(nn.Module):
    """Simplified BitShift Normalization for SSD blocks.

    Uses power-of-2 scaling for integer-only normalization.
    """

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.step_size = nn.Parameter(torch.ones(1))
        self.eps = 1e-6

    def forward(self, x):
        # Center
        x = x - x.mean(dim=-1, keepdim=True)
        # Compute variance
        var = x.pow(2).mean(dim=-1, keepdim=True)
        # Power-of-2 scale lookup (integer-only)
        scale = self._find_power_of_2_scale(var)
        return x * scale * self.gamma * self.step_size

    def _find_power_of_2_scale(self, var):
        """Find power-of-2 scale for normalization."""
        var_safe = var + self.eps
        scale = torch.ones_like(var_safe)

        # Thresholds for power-of-2 scaling
        scale = torch.where(var_safe >= 4.0, torch.full_like(scale, 0.5), scale)
        scale = torch.where(var_safe >= 16.0, torch.full_like(scale, 0.25), scale)
        scale = torch.where(var_safe >= 64.0, torch.full_like(scale, 0.125), scale)
        scale = torch.where(var_safe >= 256.0, torch.full_like(scale, 0.0625), scale)
        scale = torch.where(var_safe >= 1024.0, torch.full_like(scale, 0.03125), scale)
        scale = torch.where(var_safe >= 4096.0, torch.full_like(scale, 0.015625), scale)
        scale = torch.where(var_safe < 1.0, torch.full_like(scale, 1.0), scale)
        scale = torch.where(var_safe < 0.25, torch.full_like(scale, 2.0), scale)
        scale = torch.where(var_safe < 0.0625, torch.full_like(scale, 4.0), scale)

        return scale


def _create_bitshift_norm(dim):
    """Create BitShiftNorm for SSD blocks."""
    return BitShiftNormV2(dim)


def _squareplus_activation(x):
    """Squareplus activation (integer-only).

    squareplus(x) = 0.5 * (x + sqrt(x^2 + 4))
    Always positive for all x.
    """
    x = torch.clamp(x, -50.0, 50.0)
    y_sq = x * x + 4.0

    # Better initial guess based on magnitude
    # 1/sqrt(y) for various ranges
    r = torch.ones_like(y_sq)
    r = torch.where(y_sq >= 4.0, 0.5 * torch.ones_like(r), r)
    r = torch.where(y_sq >= 16.0, 0.25 * torch.ones_like(r), r)
    r = torch.where(y_sq >= 64.0, 0.125 * torch.ones_like(r), r)
    r = torch.where(y_sq >= 256.0, 0.0625 * torch.ones_like(r), r)
    r = torch.where(y_sq >= 1024.0, 0.03125 * torch.ones_like(r), r)
    r = torch.where(y_sq >= 4096.0, 0.015625 * torch.ones_like(r), r)

    # More Newton-Raphson iterations for better convergence
    for _ in range(5):
        r = r * (1.5 - 0.5 * y_sq * r * r)
        r = torch.clamp(r, min=1e-6, max=10.0)

    sqrt_y = y_sq * r  # sqrt(y) = y * rsqrt(y)
    result = 0.5 * (x + sqrt_y)

    # Squareplus is always >= 1 for x=0 (sqrt(4)/2 = 1)
    # and always positive. Clamp for safety.
    return torch.clamp(result, min=0.0)


def _sigmoid_gate(z):
    """Algebraic sigmoid gate (integer-only)."""
    # sigmoid_alg(z) = 0.5 + 0.5 * z / (1 + |z|)
    abs_z = torch.abs(z)
    return 0.5 + 0.5 * z / (1.0 + abs_z)
