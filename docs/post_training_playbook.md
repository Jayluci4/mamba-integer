# Mamba-Integer Post-Training Playbook

## Overview

Post-training strategy for the 46M parameter Mamba-Integer model — a ternary-weight ({-1, 0, 1}),
integer-only SSM targeting ZK-ML and FHE-compatible inference. This document covers mid-training
interventions, SFT, RL, and novel algorithms designed to make our model punch above its weight class.

---

## Our Unfair Advantages (First Principles)

1. **46M ternary = runs in milliseconds.** We can generate millions of samples cheaply. Self-play and rejection sampling are essentially free.
2. **3 of our 5 domains have VERIFIABLE rewards** — Solidity compiles or doesn't, SQL executes or doesn't, math is checkable. No expensive human labels needed.
3. **192GB MI300X VRAM = 100+ copies simultaneously.** Ensemble methods that are impractical for 70B models are trivial for us.
4. **ZK-ML target means verifiability IS the product.** We can design RL rewards around verifiability itself — not just correctness.
5. **Ternary weight dynamics** — only ~9MB of effective information, but boundary crossings during SFT/RL let us surgically reshape behavior.
6. **Integer-only inference** — zero transcendentals. Unique capability no peer has.

---

## Phase 0: Mid-Training Interventions (During Remaining Pretraining Steps)

### 0a. Data Annealing
In the final 20% of pretraining steps, shift the data mix toward highest quality:
- Increase FineWeb-Edu and OpenWebMath weights
- Drop lower quality sources
- This technique is used by Llama-3, DeepSeek, and most frontier models

### 0b. Replay Buffer
Periodically re-expose the model to short-sequence examples from earlier curriculum stages.
Prevents catastrophic forgetting of short-context patterns as the curriculum reaches 1024.

---

## Phase 1: SFT — Two-Stage Curriculum

Drawing from Two-Stage Curriculum SFT (cognitive science inspired):

### Stage 1a: Chain-of-Thought Distillation
- Use Claude/GPT-4 as teacher to generate step-by-step reasoning traces
- Domain-specific examples:
  - **Solidity**: "Here's a contract request → here's my reasoning → here's the code"
  - **Math**: "Here's the problem → step 1 → step 2 → answer"
  - **SQL**: "Here's the natural language → schema analysis → query"
  - **SEC**: "Here's a filing section → key metrics → summary"
- Train on reasoning traces so the model learns to THINK, not just pattern-match
- Be selective about WHICH reasoning patterns to teach (capacity is limited)

### Stage 1b: Direct Instruction Tuning
- Standard prompt→response pairs WITHOUT chain-of-thought
- The model has internalized reasoning from 1a, now learns to produce clean outputs
- Domain-specific instruction sets:
  - "Write a Solidity contract that..."
  - "Generate a SQL query for..."
  - "Summarize this 10-K filing..."

### Novel: Boundary-Targeted SFT (BT-SFT)
In ternary training, most weights are far from quantization boundaries — gradients update the
latent weight but the ternary value stays the same. Only weights near boundaries change behavior.

Custom optimizer that:
1. Identifies weights within ε of a ternary boundary
2. Amplifies gradients for these weights (they're the ones that matter)
3. Dampens gradients for weights deep inside a quantization bin (wasted compute)

Expected benefit: **2-3x more sample-efficient SFT** for ternary models.

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
| SEC/Finance | Regex + entity extraction    | Valid format? Real-looking numbers?         |
| English     | Perplexity from teacher      | Fluent? Coherent?                           |

### GRPO Loop
1. Sample N completions per prompt from our model (N=64 is cheap for us)
2. Score each with domain-specific verifier (binary: correct/incorrect)
3. Compute group-relative advantages (correct → positive, wrong → negative)
4. Update policy to amplify correct completions

### Novel: Integer-Verified GRPO (IV-GRPO)
Since our model targets ZK-ML, add a SECOND reward signal: **does the output match between
the float training path and the integer-only inference path?**

```
R = R_correctness * (1 + α * R_integer_consistency)
```

Where `R_integer_consistency = 1` if float and integer paths agree, `0` if they diverge.
This trains the model to produce outputs robust to integer quantization artifacts.

**This is a novel contribution — no one else trains models to be "ZK-friendly."**

---

## Phase 3: Self-Play Amplification

### 3a. SPIN (Self-Play Fine-Tuning)
Reference: [SPIN paper](https://arxiv.org/abs/2401.01335)
- Model generates completions
- Train discriminator: "is this human-written or model-generated?"
- Use discrimination signal to improve generator
- Iterate until model can't distinguish its own outputs from human data
- Perfect for general English quality — no verifier needed

### 3b. Ensemble Self-Distillation (Novel)
Our killer advantage — can fit 100+ model copies in VRAM:
1. Train 8 models with different random seeds / data orderings
2. For each prompt, generate from all 8 models
3. Ensemble: majority vote for verifiable domains, average logits for others
4. Distill the ensemble's outputs back into a single model
5. Repeat

Related work: SPEQ (Stochastic Precision Ensemble for Quantized Networks)

Each iteration squeezes more performance from the same parameter budget.

---

## Phase 4: Inference-Time Scaling

### 4a. Best-of-N with Verifiable Rewards
- For Solidity/SQL/Math: generate N outputs, pick the one that passes verification
- Our model runs in milliseconds, so N=32 adds negligible latency
- Proven more compute-efficient than scaling parameters

### 4b. Self-Consistency Voting
- Generate N reasoning chains, take majority answer
- Especially powerful for math and structured outputs

### 4c. Novel: ZK-Verified Chain of Thought
Multi-step generation where each intermediate step is verifiable:
1. Model generates step 1 → ZK proof of step 1
2. Model generates step 2 (conditioned on verified step 1) → ZK proof of step 2
3. Final answer comes with a full ZK proof chain

**Unique capability no other model offers — cryptographically verifiable reasoning.**

---

## Priority Order & Estimated Impact

| Technique                          | Effort | Impact      | Priority              |
|------------------------------------|--------|-------------|----------------------|
| Data annealing (Phase 0)           | Low    | Medium      | **Do NOW**           |
| Domain-specific SFT (Phase 1)     | Medium | High        | **First**            |
| GRPO with verifiers (Phase 2)     | Medium | **Massive** | **Core**             |
| IV-GRPO integer consistency        | Medium | High        | **Novel differentiator** |
| BT-SFT boundary targeting         | High   | Medium      | Research             |
| Ensemble self-distillation         | Medium | High        | After GRPO           |
| SPIN self-play                     | Medium | Medium      | After SFT            |
| Best-of-N inference                | Low    | High        | **Free lunch**       |
| ZK chain-of-thought               | High   | **Massive** | Long-term            |

---

## Key Insights from Frontier Model Research

### From Kimi K2.5 (Moonshot AI)
- **Parallel-Agent RL (PARL)**: Orchestrator agent decomposes tasks into parallelizable subtasks
  executed by frozen sub-agents. Reward: 80% task quality + 20% critical path efficiency.
- **Long2Short Distillation**: Train long-CoT model first, then distill into short-CoT model
  via length penalty RL + model merging. Achieves SOTA short-CoT results (60.8 AIME, 94.6 MATH500).
- **Partial Rollouts**: Reuse previously computed trajectories during RL to improve efficiency.
  Scale RL context to 128K tokens.
- **Quantization-Aware Training**: Native INT4 during training, not post-hoc compression.
  We already do this with ternary — validates our approach.

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

### Applicability to Our Model
| Technique             | Applicable? | Notes                                          |
|----------------------|------------|------------------------------------------------|
| Long2Short distill   | **Yes**    | Train long-CoT, compress to short outputs      |
| Partial rollouts     | **Yes**    | Huge efficiency win for our RL phase            |
| PARL agent swarm     | Maybe      | Our model is too small for orchestration        |
| Slime async RL       | **Yes**    | Adapt for our Triton kernel setup               |
| GRPO                 | **Yes**    | Core RL algorithm, perfect for verifiable domains|
| QAT approach         | Already    | We already train ternary natively               |

---

## References

- [GRPO: Group Relative Policy Optimization](https://arxiv.org/html/2503.06639v1)
- [SPIN: Self-Play Fine-Tuning](https://arxiv.org/abs/2401.01335)
- [SeRL: Self-Play RL with Limited Data](https://arxiv.org/abs/2505.20347)
- [Test-Time Compute Scaling](https://arxiv.org/abs/2512.02008)
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
