# Mamba-Integer V2: Surprise-Gated Multi-Timescale Memory

**Date:** 2025-12-27
**Status:** Design Document
**Authors:** Research synthesis from neuroscience + ML literature

---

## Executive Summary

Mamba-Integer V2 extends the integer-only SSM architecture with three key innovations inspired by neuroscience and recent ML research:

1. **Surprise-Based Retention**: Dynamically adjust state decay based on prediction error
2. **Multi-Timescale Memory**: Fast/slow memory systems with consolidation
3. **Forgetting as Feature**: Selective forgetting for regularization and capacity management

---

## 1. Core Insight: Why Surprise Matters

### Neuroscience Evidence

From our research synthesis:

| Finding | Mechanism | Source |
|---------|-----------|--------|
| Waking sharp-wave ripples "tag" surprising experiences | Hippocampal SWRs mark what to consolidate | Science 2024 |
| Reward prediction errors enhance memory | Dopamine scales with |actual - expected| | PMC 2021 |
| Novel stimuli trigger enhanced encoding | Dopamine D1/D2 activation | Multiple 2024 studies |
| Emotional salience prioritizes consolidation | Amygdala-hippocampus interactions | Frontiers 2025 |

### The Key Equation

From predictive coding (Friston's Free Energy Principle):

```
Prediction Error = |actual_state - predicted_state|

Surprise = -log P(observation | model)  # Information-theoretic

TD Error (Schultz): δ = r + γV(s') - V(s)  # Dopamine signal
```

### Connection to SSMs

Current Mamba uses **fixed** decay:
```python
h_t = α * h_{t-1} + (1-α) * u_t   # α is constant
```

The brain uses **adaptive** retention:
```
High surprise → STORE IT (low decay, high retention)
Low surprise → OK TO FORGET (high decay, low retention)
```

---

## 2. Surprise-Based Retention Gate

### Mathematical Formulation

**Step 1: Compute Prediction Error**
```python
# At layer l, position t
h_predicted = A @ h_{t-1}              # What we expect
h_actual = h_predicted + B @ x_t       # What we observe
prediction_error = ||h_actual - h_predicted||²
```

**Step 2: Compute Surprise Score**
```python
# Normalize by running statistics (EMA of prediction errors)
surprise_ema = 0.99 * surprise_ema + 0.01 * prediction_error
normalized_surprise = prediction_error / (surprise_ema + eps)

# Alternative: Use gradient magnitude (like SI/Titans)
surprise_grad = ||∇_h L||²  # Gradient magnitude as importance
```

**Step 3: Adaptive Retention**
```python
# Base retention from learned parameters
α_base = sigmoid(W_α @ [layer_emb, position_emb])

# Surprise modulation: high surprise → high retention
α_t = α_base + (1 - α_base) * tanh(β * normalized_surprise)

# Update state with adaptive retention
h_t = α_t * h_{t-1} + (1 - α_t) * u_t
```

### Integer-Only Implementation

```python
class SurpriseGatedRetention(nn.Module):
    """
    Integer-friendly surprise-based retention.
    All operations use dyadic rationals (divide by powers of 2).
    """
    def __init__(self, d_state, n_heads):
        self.d_state = d_state
        self.n_heads = n_heads

        # Learned base retention per head (log2 scale)
        self.log2_alpha_base = nn.Parameter(torch.zeros(n_heads))

        # Surprise scaling factor (also power of 2)
        self.log2_beta = nn.Parameter(torch.zeros(n_heads))

        # Running EMA for normalization
        self.register_buffer('surprise_ema', torch.ones(n_heads))

    def forward(self, h_prev, u, prediction_error):
        # Base retention: α_base = 1 - 2^(-log2_alpha_base)
        alpha_base = 1.0 - torch.pow(2.0, -self.log2_alpha_base.abs())

        # Normalize surprise
        with torch.no_grad():
            self.surprise_ema = 0.99 * self.surprise_ema + 0.01 * prediction_error.mean(-1)
        normalized = prediction_error / (self.surprise_ema.unsqueeze(-1) + 1e-6)

        # Surprise boost (clamped for stability)
        beta = torch.pow(2.0, self.log2_beta.clamp(-4, 4))
        boost = torch.tanh(beta.unsqueeze(-1) * normalized)

        # Final retention: high surprise → α closer to 1
        alpha = alpha_base.unsqueeze(-1) + (1 - alpha_base.unsqueeze(-1)) * boost

        # State update (dyadic rational friendly)
        h_new = alpha * h_prev + (1 - alpha) * u
        return h_new, normalized  # Return surprise for logging
```

---

## 3. Multi-Timescale Memory Hierarchy

### Inspired by Complementary Learning Systems

From neuroscience research:
- **Hippocampus**: Fast learning, sparse representations, temporary storage
- **Neocortex**: Slow learning, distributed representations, permanent storage
- **Sleep**: Transfers from hippocampus → neocortex via replay

### Four Memory Levels

```
Level 0: Activations      τ = 1 step       (per-forward, ephemeral)
Level 1: SSM State        τ ~ tokens       (surprise-adaptive retention)
Level 2: Chunk Memory     τ ~ 256-1024     (slow memory within context)
Level 3: Persistent Mem   τ = ∞            (fixed after training, like Titans)
```

### Implementation

```python
class MultiTimescaleMemory(nn.Module):
    """
    4-level memory hierarchy with different timescales.
    """
    def __init__(self, dim, n_heads, chunk_size=256, n_persistent=64):
        # Level 1: SSM state (per-token, surprise-adaptive)
        self.ssm_state = None  # (batch, n_heads, d_state)

        # Level 2: Chunk memory (slow, high-retention)
        self.chunk_memory = nn.Parameter(torch.zeros(n_heads, chunk_size, dim))
        self.chunk_retention = 0.99  # Very high retention

        # Level 3: Persistent memory (frozen after training)
        self.persistent_memory = nn.Parameter(torch.randn(n_persistent, dim) * 0.02)

        # Cross-level attention
        self.level_attention = nn.MultiheadAttention(dim, n_heads)

    def forward(self, x, ssm_state):
        batch, seq_len, dim = x.shape

        # Level 1: SSM state update (handled by SurpriseGatedRetention)
        # ...

        # Level 2: Chunk memory update (every chunk_size tokens)
        if seq_len % self.chunk_size == 0:
            chunk_summary = x[:, -self.chunk_size:].mean(dim=1)
            self.chunk_memory = self.chunk_retention * self.chunk_memory + \
                               (1 - self.chunk_retention) * chunk_summary

        # Level 3: Persistent memory (read-only at inference)
        persistent = self.persistent_memory.unsqueeze(0).expand(batch, -1, -1)

        # Cross-level integration
        memory_context = torch.cat([
            ssm_state.flatten(-2),  # Level 1
            self.chunk_memory.unsqueeze(0).expand(batch, -1, -1, -1).flatten(1, 2),  # Level 2
            persistent  # Level 3
        ], dim=1)

        # Attend over all memory levels
        x_attended, _ = self.level_attention(x, memory_context, memory_context)

        return x_attended
```

---

## 4. Consolidation Triggers

### When Does the Brain Consolidate?

From research:
1. **Capacity-based**: When hippocampal buffer is full
2. **Surprise-based**: Sharp-wave ripples triggered by high-surprise events
3. **Time-based**: Sleep cycles (NREM → REM alternation)
4. **Goal-based**: Task-relevant information prioritized

### Three Trigger Mechanisms for V2

```python
class ConsolidationTrigger:
    """
    Determines when to consolidate from fast → slow memory.
    """
    def __init__(self, capacity_threshold=0.9, surprise_threshold=2.0,
                 time_threshold=1000):
        self.capacity_threshold = capacity_threshold
        self.surprise_threshold = surprise_threshold  # Standard deviations
        self.time_threshold = time_threshold  # Tokens since last consolidation

        self.buffer = []
        self.cumulative_surprise = 0.0
        self.tokens_since_consolidation = 0

    def should_consolidate(self, surprise_score, buffer_fullness):
        """
        Returns True if any trigger condition is met.
        """
        self.cumulative_surprise += surprise_score
        self.tokens_since_consolidation += 1

        triggers = {
            'capacity': buffer_fullness > self.capacity_threshold,
            'surprise': self.cumulative_surprise > self.surprise_threshold,
            'time': self.tokens_since_consolidation > self.time_threshold
        }

        if any(triggers.values()):
            triggered_by = [k for k, v in triggers.items() if v]
            self.reset()
            return True, triggered_by
        return False, []

    def reset(self):
        self.cumulative_surprise = 0.0
        self.tokens_since_consolidation = 0
```

### What Gets Consolidated?

Inspired by EWC and SI:

```python
def compute_importance(states, gradients, method='surprise_weighted'):
    """
    Compute importance scores for state elements.

    Methods:
    - 'fisher': Fisher Information (EWC)
    - 'path_integral': Cumulative gradient contribution (SI)
    - 'surprise_weighted': Our method - weight by surprise at encoding
    """
    if method == 'fisher':
        # F_i = E[∂log p / ∂θ_i]²
        importance = (gradients ** 2).mean(dim=0)

    elif method == 'path_integral':
        # Ω_k = Σ (ω_k^μ / Δ_k²)
        # Track gradient * change during training
        importance = gradients * (states - states.detach())

    elif method == 'surprise_weighted':
        # Weight by surprise score at encoding time
        # High surprise → high importance
        importance = states * surprise_scores.unsqueeze(-1)

    return importance
```

---

## 5. Forgetting as Feature

### Why Forgetting Helps

From research synthesis:

| Benefit | Mechanism | Reference |
|---------|-----------|-----------|
| Regularization | Prevent overfitting by discarding noise | Dropout connection |
| Generalization | Forget task-specific, keep transferable | SHY (synaptic homeostasis) |
| Capacity | Make room for new information | PackNet, Progressive NN |
| Privacy | Remove sensitive data influence | Machine unlearning |
| Efficiency | Sparse memory → faster retrieval | Memory layers, FoX |

### Selective Forgetting Gate

Inspired by LSTM forget gate and FoX (Forgetting Transformer, 2025):

```python
class SelectiveForgetting(nn.Module):
    """
    Learns WHAT to forget, not just how much.

    Key insight: Forgetting should be selective:
    - Forget noise, duplicates, outdated info
    - Keep surprising, novel, goal-relevant info
    """
    def __init__(self, dim, n_heads):
        self.forget_gate = nn.Sequential(
            nn.Linear(dim * 2, dim),  # [state, input]
            nn.SiLU(),
            nn.Linear(dim, n_heads),
            nn.Sigmoid()
        )

        # Importance threshold (learned)
        self.importance_threshold = nn.Parameter(torch.tensor(0.5))

    def forward(self, h_prev, x, importance_scores):
        """
        Args:
            h_prev: Previous state
            x: Current input
            importance_scores: Per-element importance (from consolidation)
        """
        # Compute forget gate
        combined = torch.cat([h_prev, x], dim=-1)
        forget = self.forget_gate(combined)  # (batch, n_heads)

        # Importance-modulated forgetting
        # High importance → don't forget (override gate)
        importance_mask = (importance_scores > self.importance_threshold).float()

        # Final forget decision
        effective_forget = forget * (1 - importance_mask)

        # Apply forgetting
        h_forgotten = h_prev * (1 - effective_forget.unsqueeze(-1))

        return h_forgotten, forget  # Return gate for analysis
```

### Synaptic Homeostasis for SSM

Inspired by SHY (downscale weak synapses, preserve relevant ones):

```python
def synaptic_homeostasis(state, importance, downscale_factor=0.99):
    """
    Scale down unimportant state elements.
    Mimics sleep-dependent synaptic downscaling.

    Called periodically (e.g., every N tokens or at chunk boundaries)
    """
    # Normalize importance to [0, 1]
    importance_norm = importance / (importance.max() + 1e-6)

    # Downscale inversely proportional to importance
    # High importance → scale = 1 (preserve)
    # Low importance → scale = downscale_factor (shrink)
    scale = importance_norm + (1 - importance_norm) * downscale_factor

    return state * scale
```

---

## 6. Complete V2 Block Architecture

```python
class MambaIntegerV2Block(nn.Module):
    """
    Complete V2 block with:
    1. Surprise-based retention
    2. Multi-timescale memory
    3. Consolidation triggers
    4. Selective forgetting
    """
    def __init__(self, dim, d_state, n_heads, chunk_size=256):
        super().__init__()

        # Standard components
        self.norm = BitShiftNorm(dim)
        self.in_proj = BitLinear(dim, dim * 2)
        self.conv1d = nn.Conv1d(dim, dim, kernel_size=4, groups=dim)

        # V2 components
        self.surprise_gate = SurpriseGatedRetention(d_state, n_heads)
        self.multi_memory = MultiTimescaleMemory(dim, n_heads, chunk_size)
        self.consolidation = ConsolidationTrigger()
        self.selective_forget = SelectiveForgetting(dim, n_heads)

        # SSM projections
        self.x_proj = BitLinear(dim, d_state * 2)  # B, C
        self.dt_proj = BitLinear(dim, n_heads)

        # Output
        self.out_proj = BitLinear(dim, dim)

    def forward(self, x, state=None):
        batch, seq_len, dim = x.shape

        # Normalize
        x_norm = self.norm(x)

        # Project and split
        xz = self.in_proj(x_norm)
        x, z = xz.chunk(2, dim=-1)

        # Conv
        x = self.conv1d(x.transpose(1, 2)).transpose(1, 2)

        # SSM projections
        BC = self.x_proj(x)
        B, C = BC.chunk(2, dim=-1)
        dt = F.softplus(self.dt_proj(x))

        # Initialize state
        if state is None:
            state = torch.zeros(batch, self.n_heads, self.d_state, device=x.device)

        outputs = []
        importance_buffer = []

        for t in range(seq_len):
            # Compute prediction error
            h_predicted = self.A @ state  # What we expect
            u = B[:, t] * x[:, t:t+1]
            h_actual = h_predicted + u
            prediction_error = (h_actual - h_predicted).pow(2).mean(-1)

            # Surprise-gated retention
            state, surprise = self.surprise_gate(state, u, prediction_error)

            # Check consolidation trigger
            should_consolidate, triggers = self.consolidation.should_consolidate(
                surprise.mean().item(),
                len(importance_buffer) / self.chunk_size
            )

            if should_consolidate:
                # Compute importance and consolidate
                importance = compute_importance(
                    torch.stack(importance_buffer),
                    method='surprise_weighted'
                )
                # Transfer to slow memory
                self.multi_memory.consolidate(importance_buffer, importance)
                importance_buffer = []
            else:
                importance_buffer.append(state.detach())

            # Selective forgetting (periodic)
            if t % 64 == 0:
                state, forget_gate = self.selective_forget(
                    state, x[:, t],
                    importance=surprise  # Use surprise as importance proxy
                )

            # Output
            y = (C[:, t] * state).sum(-1)
            outputs.append(y)

        y = torch.stack(outputs, dim=1)

        # Gate and project
        y = y * F.silu(z)
        y = self.out_proj(y)

        return y + x_norm, state
```

---

## 7. Key Equations Summary

### Surprise Computation
```
prediction_error_t = ||h_t - A @ h_{t-1}||²
normalized_surprise_t = prediction_error_t / EMA(prediction_error)
```

### Adaptive Retention
```
α_t = α_base + (1 - α_base) * tanh(β * surprise_t)
h_t = α_t * h_{t-1} + (1-α_t) * u_t
```

### Consolidation Decision
```
consolidate = (capacity > θ_cap) OR (Σsurprise > θ_surp) OR (t > θ_time)
```

### Importance-Weighted Forgetting
```
forget_t = σ(W_f @ [h_{t-1}, x_t]) * (1 - importance_mask)
h_forgotten = h_{t-1} * (1 - forget_t)
```

---

## 8. Connection to MIRAS/Titans/Nested Learning

### How V2 Fits the Frameworks

| Framework | V2 Implementation |
|-----------|-------------------|
| **MIRAS Memory Architecture** | Multi-timescale (vector L1, matrix L2, MLP L3) |
| **MIRAS Retention Gate** | Surprise-adaptive α_t |
| **Titans Surprise Metric** | ||∇_M L|| → prediction error |
| **Titans Memory Write** | High surprise triggers consolidation |
| **Nested Learning Timescales** | 4 levels with different τ |
| **ODP Compatibility** | All ops remain integer-friendly |

### The Unifying Principle

```
All sequence models = memory systems with:
1. What to store (surprise/importance selection)
2. How long to store (retention/decay)
3. When to consolidate (capacity/time/surprise triggers)
4. What to forget (importance-weighted pruning)

V2 makes all four EXPLICIT and LEARNABLE.
```

---

## 9. Implementation Roadmap

### Phase 1: Core Surprise Gate
- [ ] Implement SurpriseGatedRetention module
- [ ] Add prediction error computation to forward pass
- [ ] Validate on toy sequences (high surprise → high retention)

### Phase 2: Multi-Timescale Memory
- [ ] Add chunk memory (L2) alongside SSM state (L1)
- [ ] Implement persistent memory (L3) for stable knowledge
- [ ] Test memory transfer during "sleep" (offline consolidation)

### Phase 3: Consolidation Logic
- [ ] Implement three trigger mechanisms
- [ ] Add importance scoring (surprise-weighted)
- [ ] Test that important information survives longer

### Phase 4: Selective Forgetting
- [ ] Add forget gate
- [ ] Implement importance-gated override
- [ ] Validate generalization improvement on held-out data

### Phase 5: Integer-Only Optimization
- [ ] Replace all ops with dyadic rationals
- [ ] Ensure ZK/FHE compatibility
- [ ] Benchmark vs V1

---

## 10. Expected Benefits

| Metric | V1 | V2 (Expected) | Improvement |
|--------|----|--------------:|-------------|
| Long-range dependency | ~1024 tokens | ~4096+ tokens | 4x |
| Catastrophic forgetting | Present | Mitigated | EWC-like |
| Inference efficiency | Good | Better | Sparse memory |
| Generalization | Baseline | Improved | Forgetting regularization |
| ZK/FHE compatibility | Full | Full | Maintained |

---

## References

### Neuroscience
- Girardeau et al. (2024). "Selection of experience for memory by hippocampal sharp wave ripples." Science.
- Tononi & Cirelli. "Synaptic Homeostasis Hypothesis (SHY)."
- LePort et al. (2017). "Highly Superior Autobiographical Memory."

### Machine Learning
- Kirkpatrick et al. (2017). "Overcoming catastrophic forgetting with EWC."
- Zenke et al. (2017). "Continual Learning through Synaptic Intelligence."
- Peng et al. (2025). "Forgetting Transformer (FoX)."
- Gu & Dao (2024). "Mamba: Linear-Time Sequence Modeling."
- Behrouz et al. (2024). "Titans: Learning to Memorize at Test Time."

### Our Prior Work
- MIRAS Framework (docs/miras-framework.md)
- Nested Learning (docs/nested-learning.md)
- ODP x Google (docs/ODPxGOOGLE.md)
