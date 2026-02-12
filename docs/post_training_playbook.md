# Mamba-Integer Post-Training Playbook

## Overview

Post-training strategy for the 46M parameter Mamba-Integer model — a ternary-weight ({-1, 0, 1}),
integer-only SSM targeting ZK-ML and FHE-compatible inference. This document covers mid-training
interventions, SFT, RL, novel meta-learned algorithms, and inference-time techniques.

---

## Design Advantages

1. **46M ternary = millisecond inference.** Millions of samples can be generated cheaply — self-play and rejection sampling have near-zero marginal cost.
2. **3 of 5 domains have verifiable rewards** — Solidity compiles or doesn't, SQL executes or doesn't, math is checkable. No expensive human labels needed.
3. **192GB MI300X VRAM = 100+ copies simultaneously.** Ensemble methods impractical for 70B models are trivial at this scale.
4. **ZK-ML target means verifiability IS the product.** RL rewards can be designed around verifiability itself, not just correctness.
5. **Ternary weight dynamics** — ~9MB of effective information (46M × 1.58 bits), but boundary crossings during SFT/RL enable surgical reshaping of behavior.
6. **Integer-only inference** — zero transcendentals in the inference path (algebraic sigmoid, Cayley RoPE, Newton-Raphson rsqrt).

---

## Phase 0: Mid-Training Interventions

### 0a. Data Annealing
In the final 20% of pretraining steps, shift the data mix toward highest quality:
- Increase FineWeb-Edu and OpenWebMath weights
- Drop lower quality sources
- Technique used by Llama-3, DeepSeek, and most frontier models

### 0b. Replay Buffer
Periodically re-expose the model to short-sequence examples from earlier curriculum stages.
Prevents catastrophic forgetting of short-context patterns as the curriculum reaches 1024.

---

## Phase 1: SFT — Two-Stage Curriculum

Drawing from Two-Stage Curriculum SFT (cognitive science inspired):

### Stage 1a: Chain-of-Thought Distillation
- Use Claude/GPT-4 as teacher to generate step-by-step reasoning traces
- Domain-specific examples:
  - **Solidity**: contract request → reasoning → code
  - **Math**: problem → step 1 → step 2 → answer
  - **SQL**: natural language → schema analysis → query
  - **SEC**: filing section → key metrics → summary
- Train on reasoning traces so the model learns structured thinking, not pattern matching
- Selective about which reasoning patterns to teach — capacity is severely limited

### Stage 1b: Direct Instruction Tuning
- Standard prompt → response pairs without chain-of-thought
- The model has internalized reasoning from 1a, now learns to produce clean outputs
- Domain-specific instruction sets:
  - "Write a Solidity contract that..."
  - "Generate a SQL query for..."
  - "Summarize this 10-K filing..."

---

## Phase 2: RLVR — Reinforcement Learning with Verifiable Rewards

The core post-training technique. Uses GRPO (Group Relative Policy Optimization, from
DeepSeek-R1) with domain-specific verifiers.

### Verifier Stack

| Domain       | Verifier                     | Reward Signal                              |
|-------------|-----------------------------|--------------------------------------------|
| Solidity    | `solc` compiler + Hardhat   | Compiles? Passes tests? Gas efficient?     |
| SQL         | SQLite/Postgres executor     | Executes? Returns expected rows?           |
| Math        | SymPy / numerical checker    | Correct answer? Valid steps?               |
| SEC/Finance | Regex + entity extraction    | Valid format? Realistic figures?            |
| English     | Perplexity from teacher      | Fluent? Coherent?                           |

### GRPO Loop
1. Sample N completions per prompt (N=64 is cheap at this model size)
2. Score each with domain-specific verifier (binary: correct/incorrect)
3. Compute group-relative advantages (correct → positive, wrong → negative)
4. Update policy to amplify correct completions

### Novel: Integer-Verified GRPO (IV-GRPO)
Since the model targets ZK-ML, add a second reward signal: does the output match between
the float training path and the integer-only inference path?

```
R = R_correctness + α * exp(-β * KL(P_float || P_integer))
```

Where `KL(P_float || P_integer)` measures distributional divergence between the float and
integer inference paths. Low KL = high reward. This trains the model toward regions of
parameter space where integer quantization causes minimal distributional shift.

Novel contribution — no existing work trains models for quantization robustness through
RL rewards rather than post-hoc compression.

---

## Phase 3: Self-Play and Ensemble Amplification

### 3a. SPIN (Self-Play Fine-Tuning)
- Model generates completions
- Train discriminator: "is this human-written or model-generated?"
- Use discrimination signal to improve generator
- Iterate until model can't distinguish its own outputs from human data
- Note: limited utility at 46M ternary — outputs are trivially distinguishable from human
  text, so discriminator signal provides weak learning pressure. Deprioritized.

### 3b. Ensemble Self-Distillation
With 100+ model copies fitting in VRAM simultaneously:
1. Train K models (K=8–128) with different random seeds and data orderings
2. For each prompt, generate from all K models
3. Ensemble: majority vote for verifiable domains, average logits for others
4. Distill the ensemble's outputs back into a single model
5. Repeat

Ternary quantization is highly sensitive to initialization — each seed produces a
different ternary configuration. Large K (possible only at this model size) captures
the full distribution of viable configurations, and distillation extracts the optimal
single configuration.

Related work: SPEQ (Stochastic Precision Ensemble for Quantized Networks)

---

## Phase 4: Inference-Time Scaling

### 4a. Best-of-N with Verifiable Rewards
- For Solidity/SQL/Math: generate N outputs, pick the one that passes verification
- Model runs in milliseconds, so N=32 adds negligible latency
- Proven more compute-efficient than scaling parameters

### 4b. Self-Consistency Voting
- Generate N reasoning chains, take majority answer
- Especially powerful for math and structured outputs

### 4c. ZK-Verified Chain of Thought
Multi-step generation where each intermediate step is verifiable:
1. Model generates step 1 → ZK proof of step 1
2. Model generates step 2 (conditioned on verified step 1) → ZK proof of step 2
3. Final answer comes with a full ZK proof chain

Unique capability — cryptographically verifiable reasoning, enabled by the integer-only
inference path.

---

## Phase 5: Meta-Discovered Training

### Core Thesis

Every phase of post-training involves meta-decisions that humans make with heuristics:
SFT → which data, what order, what weighting. RL → what update rule, what domain balance.
Distillation → what objective, what temperature. Inference → how many samples, when to verify.

For large models, suboptimal meta-decisions are absorbed by sheer capacity. **For a 9MB
ternary model, every meta-decision is a bit allocation problem.** Each training step competes
for the same ~9MB of representational capacity. Meta-learning finds the optimal allocation.

Inspired by DeepMind's [DiscoRL](https://github.com/google-deepmind/disco_rl) (Nature 2025)
— which used meta-learning to discover RL algorithms that beat all human-designed methods
(PPO, SAC, MuZero) on Atari — the same paradigm is applied across the entire post-training
pipeline.

### 5a. Meta-Discovered Ternary RL (MDTR)

**Problem:** The ternary policy landscape is piecewise-constant. Most gradient updates move
the latent float weight but the ternary value {-1, 0, 1} stays the same. Then one update
crosses a boundary and behavior jumps discontinuously. No existing RL algorithm was designed
for this landscape.

**Approach:** Reframe the optimizer as an RL agent and the training process as the environment.

| MDP Component | Mapping |
|---------------|---------|
| State         | Boundary distances, flip rates, per-domain verifier scores, grad norms |
| Action        | Per-layer LR multipliers, boundary zone widths, domain weights |
| Reward        | Verifier score improvement over K training steps |
| Trajectory    | Sequence of (generate → score → update → measure) cycles |

**Meta-network architecture:** 2-layer LSTM (128 hidden units).

Input per step:
- Per-layer boundary distance histograms [16 layers]
- Per-layer recent flip rates [16]
- Per-layer gradient norms [16]
- Per-domain verifier scores [5 domains]
- Training progress, loss delta [2]

Output per step:
- Per-layer LR multipliers [16]
- Per-layer boundary zone widths (ε) [16]
- Per-domain loss weights [5]
- Per-layer flip penalty (regularize excessive boundary crossings) [16]
- **Novel prediction targets [16]** — semantically undefined, following DiscoRL's design.
  The meta-learner decides what these predict. DiscoRL's undefined predictions independently
  discovered value-function-like concepts plus future policy entropy — quantities no human
  designed. The equivalent discovery for ternary models could yield entirely new concepts
  around quantization robustness.

**Meta-training loop:**
1. Run 50 parallel copies of the 5M model (~50MB each with optimizer = 2.5GB, fits in 192GB)
2. Each copy does a 2000-step GRPO run, modulated by the meta-network's output
3. Evaluate all copies on held-out verifier suite
4. Meta-gradients update the meta-network to maximize final verifier scores
5. Repeat ~200 meta-steps
6. Estimated compute: ~2.5 days on single MI300X
7. Transfer discovered update rule from 5M → 50M model (DiscoRL demonstrated cross-scale transfer)

**Why meta-discovery beats hand-crafted rules:**
Hand-crafted boundary-aware GRPO assumes "amplify near-boundary, dampen far-from-boundary."
A meta-learner can discover non-obvious strategies:
- Some boundary-adjacent weights are load-bearing and should be frozen
- Some deep-interior weights should be pushed toward boundaries to unlock capabilities
- Cross-layer coupling (e.g., layer 7 benefits from amplification only when layer 12 recently flipped)
- Domain-specific boundary strategies (Solidity vs SQL vs math have different optimal flip patterns)

### 5b. Meta-SFT: Learned Data Curator

**Problem:** At 9MB capacity (3.6% of GPT-2's 248MB), each SFT example must be chosen
surgically. A wrong example can flip boundary weights destructively, erasing capacity
allocated to other domains (catastrophic interference via ternary boundary crossings).

**Approach:** A small data selector network (<1M params) observes model state and selects
which training example to show next.

Input to selector:
- Per-domain loss distribution (which domains are improving vs plateauing)
- Boundary distance statistics (how many weights are near flipping)
- Recent gradient norms per layer
- Training step / total steps

Output: probability distribution over domain + difficulty buckets

Meta-objective: maximize held-out verifier scores after K SFT steps.

**Implementation:**
- Maintain a pool of ~5000 curated examples across 5 domains
- Data selector outputs softmax over domain + difficulty buckets
- At each step, sample from the selector's distribution
- Every 500 steps, evaluate on held-out verifiers and compute meta-gradient
- Lightweight enough to run inline with SFT

The key advantage over standard curriculum learning: the selector has access to ternary
boundary information and can learn which examples cause beneficial flips (expanding
domain capability) vs harmful flips (catastrophic interference).

Related work: Meta-Rater (ACL 2025), DFT: Dynamic Fine-Tuning (ICLR 2026),
Curriculum Learning with Quality-Driven Data Selection

### 5c. Meta-Distillation: Learned Transfer Objective

**Problem:** Standard distillation minimizes KL(teacher || student) on logits, assuming the
student can represent the teacher's distribution. A ternary student cannot — its outputs are
constrained to what {-1, 0, 1} weights can produce. Much of the KL signal is noise the
student will never represent, creating gradient interference.

**Approach:** Learn a transformation T of teacher logits before computing the objective:

```
L_distill = D(T(teacher_logits, domain, difficulty), student_logits)
```

Where T is a small learned network that can:
- Sharpen teacher logits to focus on top-k tokens (ignore the long tail the student can't capture)
- Apply domain-specific temperature (Solidity may need hard targets, English may need soft)
- Re-weight token positions (early tokens may matter more for ternary capacity allocation)
- Suppress logit dimensions where the student's integer path diverges most from float path

Meta-objective: student's verifier scores after N distillation steps — not the distillation
loss itself. This decouples "how well you match the teacher" from "how well you actually perform."

The optimal distillation target for a ternary student is not the teacher's full distribution.
It's a simplified projection of the teacher's knowledge onto the representable subspace of
ternary weights. The meta-learner discovers this projection.

Related work: BitDistill, Collaborative Multi-Teacher KD for Low Bit-Width

### 5d. Meta-Inference: Learned Compute Router

**Problem:** Fixed Best-of-N wastes compute on easy prompts and may be insufficient for
hard ones. "Write an ERC20 token" needs N=2. "Write a cross-chain atomic swap with flash
loan protection" needs N=128.

**Approach:** A tiny router network (~100K params) that predicts per-prompt optimal compute budget.

Training the router (free for verifiable domains):
1. For 10K prompts across all domains, generate N=128 completions
2. Record the minimum N needed to find a correct completion (via verifier)
3. This gives (prompt, min_N) supervised training pairs at zero labeling cost
4. Train router: prompt features → predicted (N, temperature, strategy)

At inference:
1. Router predicts difficulty → outputs (N, temperature, strategy)
2. Strategy selection: {best-of-N, beam-search, self-consistency} based on predicted difficulty
3. Easy prompts: N=2, greedy → millisecond response
4. Hard prompts: N=64, beam search with verification pruning

Failure modes are domain-specific and predictable (Solidity fails on complex inheritance,
SQL fails on nested subqueries), making the router's task well-structured.

Compute savings estimate: if 60% of prompts are easy (N=2), 30% medium (N=8), 10% hard
(N=64) → average N = 10 vs fixed N=32. ~3.2x inference speedup at equal or better accuracy.

Related work: Input-Adaptive LM Compute Allocation (ICLR 2025), Adaptive Test-Time Compute
via Learned Heuristics, Latency-Aware Test-Time Compute

### 5e. Meta-Ensemble: Learned Aggregation

**Problem:** Standard ensembles average logits or majority-vote with equal weights. But
ternary quantization sensitivity to initialization means each seed produces a different
configuration that specializes differently across domains.

**Approach:** A learned aggregation function that produces optimally-weighted combinations:

```
logits_final = Σᵢ wᵢ(prompt, domain) × logitsᵢ
```

Where w is a learned function (not uniform 1/K). The aggregator learns:
- Which models are experts for which domains
- Which model disagreements signal genuine ambiguity vs noise
- When to trust a confident minority over the majority

Training data is free — generate from all K models on validation prompts, verify with
domain-specific verifiers, learn which models are correct under which conditions.

### 5f. Meta-Pipeline: Learned Phase Sequencing

**Problem:** The standard pipeline (Pretraining → SFT → RL → Distillation → Deploy) is
sequential and hand-designed. It assumes each phase completes before the next begins.

**Hypothesis:** For ternary models, each phase reshapes the ternary configuration through
boundary crossings. The order of these crossings may matter — like a combination lock, Phase A
might need to happen before Phase B because A creates the boundary conditions B exploits.

**Approach:** Define the pipeline as a sequence of (phase, duration, hyperparams) tuples.
Use evolutionary search on the 5M model:

1. Generate 100 random pipeline configurations
2. Each pipeline runs on the 5M model (hours, not days)
3. Evaluate final model on verifier suite
4. Evolve: mutate top performers, crossover, repeat
5. After ~50 generations, extract the best pipeline
6. Apply to 50M model

Search space:
- Phase type: {SFT, GRPO, distillation, ensemble-round, MDTR}
- Duration: 100–5000 steps per phase
- Domain focus: per-phase domain emphasis
- Learning rate: per-phase
- Data source: per-phase data pool

This is the most speculative technique — included as a research direction.

---

## Unifying Principle: Every Meta-Decision is Bit Allocation

For a 9MB ternary model, capacity is the binding constraint. Every training step, every
data example, every RL update competes for the same ~9MB of representational space.

| Phase | Meta-Decision | What's Being Allocated |
|-------|---------------|----------------------|
| SFT | Which example to train on | Which domain gets capacity |
| RL | Which weights to update | Which ternary bits to flip |
| Distillation | What knowledge to transfer | Which teacher patterns fit in 9MB |
| Inference | How many samples to generate | Compute budget per query |
| Pipeline | What order to train | Which boundary crossings happen first |

Large models have enough capacity that suboptimal allocation still works.
At 9MB, meta-learning is the tool that solves the bit allocation problem.

---

## Priority Matrix

| Technique                          | Effort | Impact      | Priority              |
|------------------------------------|--------|-------------|----------------------|
| Data annealing (Phase 0)           | Low    | Medium      | **Immediate**        |
| Domain-specific SFT (Phase 1)     | Medium | High        | **First**            |
| GRPO with verifiers (Phase 2)     | Medium | **Massive** | **Core**             |
| IV-GRPO integer consistency        | Medium | High        | **Novel differentiator** |
| MDTR meta-discovered RL (5a)      | High   | **Massive** | **After basic GRPO** |
| Meta-SFT data curator (5b)        | Medium | High        | **During SFT**       |
| Meta-distillation objective (5c)  | Medium | High        | **During distillation** |
| Meta-inference router (5d)        | Low    | High        | **Low-hanging fruit** |
| Ensemble self-distillation (3b)   | Medium | High        | After GRPO           |
| Meta-ensemble aggregation (5e)    | Low    | Medium      | After ensemble       |
| Meta-pipeline sequencing (5f)     | High   | Unknown     | Research             |
| Best-of-N inference (4a)          | Low    | High        | **Low-hanging fruit** |
| ZK chain-of-thought (4c)          | High   | **Massive** | Long-term            |
| SPIN self-play (3a)               | Medium | Low         | Deprioritized        |

Notes:
- BT-SFT (Boundary-Targeted SFT) is subsumed by MDTR — instead of hand-crafting boundary
  targeting rules, the meta-learner discovers the optimal boundary strategy.
- SPIN is deprioritized because at 46M ternary, model outputs are trivially distinguishable
  from human text, providing weak discriminator signal.

---

## Frontier Model Research

### From Kimi K2.5 (Moonshot AI)
- **Parallel-Agent RL (PARL)**: Orchestrator agent decomposes tasks into parallelizable subtasks
  executed by frozen sub-agents. Reward: 80% task quality + 20% critical path efficiency.
- **Long2Short Distillation**: Train long-CoT model first, then distill into short-CoT model
  via length penalty RL + model merging. Achieves SOTA short-CoT results (60.8 AIME, 94.6 MATH500).
- **Partial Rollouts**: Reuse previously computed trajectories during RL to improve efficiency.
  Scale RL context to 128K tokens.
- **Quantization-Aware Training**: Native INT4 during training, not post-hoc compression.
  Validates the ternary-native approach already used in this project.

### From GLM-5 (Zhipu AI / Z.ai)
- **Slime Framework**: Asynchronous RL infrastructure using SGLang (rollout) + Megatron (training)
  + Ray (resource management). Enables fine-grained post-training iterations.
- **Async Agent RL**: Model continuously learns from long-range interactions, unlocking
  potential of pre-trained models for agentic tasks.
- **Record Low Hallucination**: Achieved through careful RL reward design focusing on factuality.
- **744B MoE / 40B active**: Trained on 28.5T tokens entirely on Huawei Ascend chips.

### From DeepSeek-R1
- **GRPO**: Group Relative Policy Optimization — no value function needed, just binary rewards
  + group-relative advantage estimation. Simpler and more stable than PPO.
- **RLVR at Scale**: Verifiable rewards (math, code) enable massive RL scaling without reward hacking.
- **Emergence of Reasoning**: RL training causes spontaneous emergence of CoT, planning, reflection.

### From DeepMind DiscoRL (Nature 2025)
- **Meta-Learned RL Algorithms**: A backward LSTM meta-network discovers RL update rules
  by processing agent trajectories. Discovered algorithm (Disco103) beat PPO, SAC, and MuZero
  on Atari, reaching MuZero performance with 40% less compute.
- **Novel Prediction Semantics**: The meta-network outputs semantically undefined prediction
  targets that evolved to capture future policy entropy and upcoming large-reward events —
  quantities no human designed but optimal for learning.
- **Cross-Environment Transfer**: Algorithms discovered on Atari generalized to ProcGen,
  Crafter, DMLab-30 without retraining. Also transfers across model scales.
- **Adaptation**: MDTR (Phase 5a) applies DiscoRL's paradigm to ternary weight optimization,
  reframing the optimizer as an RL agent navigating a piecewise-constant loss landscape.

### Applicability Summary

| Technique             | Applicable? | Notes                                          |
|----------------------|------------|------------------------------------------------|
| Long2Short distill   | **Yes**    | Train long-CoT, compress to short outputs      |
| Partial rollouts     | **Yes**    | Efficiency win for RL phase                    |
| PARL agent swarm     | Maybe      | Model may be too small for orchestration       |
| Slime async RL       | **Yes**    | Adapt for Triton kernel setup                  |
| GRPO                 | **Yes**    | Core RL algorithm for verifiable domains       |
| QAT approach         | Already    | Ternary-native training already in use         |
| DiscoRL meta-learn   | **Adapted**| MDTR: discover ternary-specific update rules   |
| Meta-SFT curriculum  | **Yes**    | Learned data selection critical at 9MB capacity|
| Meta-distillation    | **Yes**    | Learned objective for ternary student          |
| Adaptive inference   | **Yes**    | Difficulty-aware compute allocation            |

---

## References

- [GRPO: Group Relative Policy Optimization](https://arxiv.org/html/2503.06639v1)
- [SPIN: Self-Play Fine-Tuning](https://arxiv.org/abs/2401.01335)
- [SeRL: Self-Play RL with Limited Data](https://arxiv.org/abs/2505.20347)
- [Test-Time Compute Scaling](https://arxiv.org/abs/2512.02008)
- [Art of Scaling Test-Time Compute](https://arxiv.org/abs/2512.02008)
- [Falcon-Edge BitNet Fine-Tuning](https://falcon-lm.github.io/blog/falcon-edge/)
- [BitNet Distillation (BitDistill)](https://arxiv.org/html/2510.13998v1)
- [Two-Stage Curriculum SFT](https://arxiv.org/html/2509.26497)
- [RLVR Across Diverse Domains](https://arxiv.org/pdf/2503.23829)
- [SPEQ: Stochastic Precision Ensemble](https://cdn.aaai.org/ojs/16839/16839-13-20333-1-2-20210518.pdf)
- [Kimi k1.5: Scaling RL with LLMs](https://arxiv.org/abs/2501.12599)
- [Kimi K2: Open Agentic Intelligence](https://arxiv.org/abs/2507.20534)
- [Kimi K2.5 Tech Blog](https://www.kimi.com/blog/kimi-k2-5.html)
- [GLM-5 HuggingFace](https://huggingface.co/zai-org/GLM-5)
- [Slime: SGLang-Native Post-Training Framework](https://github.com/THUDM/slime)
- [GLM-5 VentureBeat Coverage](https://venturebeat.com/technology/z-ais-open-source-glm-5-achieves-record-low-hallucination-rate-and-leverages)
- [State of LLMs 2025](https://magazine.sebastianraschka.com/p/state-of-llms-2025)
- [DiscoRL: Discovering SOTA RL Algorithms (Nature 2025)](https://www.nature.com/articles/s41586-025-09761-x)
- [DiscoRL GitHub](https://github.com/google-deepmind/disco_rl)
- [DiscoRL Project Page](https://google-deepmind.github.io/disco_rl/)
- [Input-Adaptive LM Compute Allocation (ICLR 2025)](https://proceedings.iclr.cc/paper_files/paper/2025/file/ff414825df833edb8b1839e3d5d495e9-Paper-Conference.pdf)
- [Adaptive Test-Time Compute via Learned Heuristics](https://arxiv.org/html/2602.03975)
- [Latency-Aware Test-Time Compute](https://arxiv.org/pdf/2509.09864)
- [DFT: Dynamic Fine-Tuning (ICLR 2026)](https://arxiv.org/abs/2508.05629)
- [Meta-Rater: Multi-dimensional Data Selection (ACL 2025)](https://aclanthology.org/2025.acl-long.533.pdf)
- [Curriculum Learning with Quality-Driven Data Selection](https://arxiv.org/abs/2407.00102)
- [Collaborative Multi-Teacher KD for Low Bit-Width (WACV 2023)](https://openaccess.thecvf.com/content/WACV2023/papers/Pham_Collaborative_Multi-Teacher_Knowledge_Distillation_for_Learning_Low_Bit-Width_Deep_Neural_WACV_2023_paper.pdf)
