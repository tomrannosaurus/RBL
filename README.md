# Retrospective Beam-Lookahead (RBL) Optimizer

A novel optimizer combining multi-path beam search with gradient-based lookahead and retrospective self-correction for neural network training.

## Idea

Traditional optimizers like Adam are reactive, they adjust based on past gradients. Some, like lookahead, are proactive in their state-projection paradigm. RBL is proactive in that it simulates multiple future trajectories and picks the best one before committing, and retroactive in that it tests states that it could have followed but did not in previous time steps. RBL sees multiple future paths, picks the best one, and maintains abandoned branches to allow self-correction.

### Algorithm (4 Stages)

> **Terminology note:** "Stages" refer to the algorithm's components below. "Phases" (in the Roadmap section) refer to project milestones.

**Stage 1: Branching**

Spawn $B$ trajectory clones from current weights $\theta_t$, each with a different strategy (e.g., varying learning rates):

$$\theta_{t+k}^{(b)} = \text{Unroll}(\theta_t, \text{Strategy}_b, k \text{ steps}) \quad \forall b \in \{1...B\}$$

**Stage 2: Retrospective Branch Continuation** *(load-bearing novel contribution)*

Key innovation: also continue the $B-1$ branches **not selected** at time $t-1$. If at $t-1$ we selected branch $b^*$ but abandoned branches $\{b_1, b_2, ...\}$, we also unroll:

$$\theta_{t+k}^{(b_i)} = \text{Unroll}(\theta_{t-1}^{(b_i)}, \text{Strategy}_{b_i}, k \text{ steps}) \quad \forall b_i \notin \{b^*\}$$

This allows RBL to self-correct if an abandoned path would have led somewhere better.

> **Implementation note:** Stage 2 is not yet implemented in the current PoC. The PoC performs branching, argmin selection, and consistency modulation, but does not retain abandoned branches across steps. See `TODO.md`.

**Stage 3: Discrete Selection**

Evaluate loss at all endpoints (current branches + continued abandoned branches):

$$b^* = \underset{b}{\text{argmin}} \ \mathcal{L}(\theta_{t+k}^{(b)})$$

**Stage 4: Consistency Modulation** *(prior art — see HGM below)*

Compute alignment between proposed update $\Delta\theta^* = \theta_{t+k}^{(b^*)} - \theta_t$ and previous update $\Delta\theta_{t-1}$:

$$C = \text{CosineSimilarity}(\Delta\theta^*, \Delta\theta_{t-1})$$

Modulation factor $\lambda = f(C) \in [0.5, 1.5]$ boosts aligned updates, dampens contradictory ones.

**Final Update:**

$$\theta_{t+1} = \theta_t + \alpha \cdot \lambda \cdot (\theta_{t+k}^{(b^*)} - \theta_t)$$

> **Note on Stage 4 novelty:** Hindsight-Guided Momentum (HGM, Sarkar 2025, arXiv:2506.22479) independently proposed using cosine similarity between a current direction and a historical direction to modulate learning rate. HGM compares the current gradient against accumulated momentum; RBL compares the selected branch update against the previous step's update. The core idea (cosine-similarity-based LR modulation) is prior art. Stage 4 is retained in the algorithm for empirical benefit but is **not claimed as a novel contribution**.

### Visual Example

```
Time 1:  θ_0 ──┬── Branch1_1 ──→ (not selected, keep tracking)
               ├── Branch2_1 ──→ θ_1 (SELECTED)
               └── Branch3_1 ──→ (not selected, keep tracking)

Time 2:  θ_1 ──┬── Branch1_2 ──→ evaluate
               ├── Branch2_2 ──→ evaluate
               └── Branch3_2 ──→ evaluate
         
         Branch1_1 ────────────→ Branch1_1' ──→ evaluate (retrospective)
         Branch3_1 ────────────→ Branch3_1' ──→ evaluate (retrospective)
         
         Select best among ALL 5 candidates
```

## Novelty Claim

RBL's novel contribution is the **retrospective branch continuation** mechanism (Stage 2): maintaining a moving window of B k-step unrolled trajectories in continuous weight space, including trajectories that were not selected at prior outer steps, and updating by argmin across all live endpoints. This transfers the "recovering" concept from discrete beam search (Della Croce & T'kindt, 2002; Pirnay & Grimm, 2024) to continuous neural network optimization.

The combination of argmin selection across diverse k-step unrolled trajectories including stale (retrospective) ones, applied to NN weight updates, has not been found published.

**What is not novel:**
- Multi-branch k-step unrolling alone (Stage 1) — precedented by Lookahead (B=1), Lookaround (B>1 with averaging), and PBT (coarser granularity).
- Cosine-similarity modulation of step size (Stage 4) — precedented by HGM (Sarkar, 2025).

**What appears novel:**
- Retrospective continuation of abandoned branches across outer steps (Stage 2), combined with argmin selection (Stage 3), in continuous weight space.

## Related Work

### Core Foundations (Direct Building Blocks)

| Authors | Paper | Link | Connection | How RBL Differs |
|---------|-------|------|------------|-----------------|
| Zhang et al., 2019 | Lookahead Optimizer | [arXiv:1907.08610](https://arxiv.org/abs/1907.08610) | $k$-step unrolling, slow/fast weights | RBL uses $B$ parallel paths with discrete selection, not single-path averaging |
| Dozat, 2016 | Nadam | [ICLR Workshop](https://openreview.net/forum?id=OM0jvwB8jIp57ZJjtNEZ) | Future-peeking via Nesterov momentum | RBL looks $k$ steps ahead across $B$ branches, not 1-step linear extrapolation |
| Salimans et al., 2017 | Evolution Strategies | [arXiv:1703.03864](https://arxiv.org/abs/1703.03864) | Multi-path exploration of weight space | RBL uses gradient-guided trajectories, not random noise perturbations |

### Closest Direct Neighbors

| Authors | Paper | Link | Connection | How RBL Differs |
|---------|-------|------|------------|-----------------|
| Zhang et al., 2023 | **Lookaround Optimizer** (NeurIPS 2023) | [arXiv:2306.07684](https://arxiv.org/abs/2306.07684) | Runs B parallel networks from a common checkpoint for k steps with different data augmentations — structurally closest published optimizer to RBL | Lookaround **averages** branch endpoints (not argmin), and does **not** retain abandoned branches across outer steps. RBL uses discrete selection and retrospective continuation. |
| Sarkar, 2025 | **Hindsight-Guided Momentum (HGM)** | [arXiv:2506.22479](https://arxiv.org/abs/2506.22479) | Same core idea as RBL Stage 4: cosine similarity between a current direction and a historical direction to modulate learning rate | HGM compares gradient vs. accumulated momentum in a single-trajectory optimizer. RBL applies this modulation to the output of multi-branch selection. The cosine-similarity LR-modulation idea is HGM's contribution; RBL does not claim it as novel. |

### Beam Search in ML (Conceptual Ancestors)

| Authors | Paper | Link | Connection | How RBL Differs |
|---------|-------|------|------------|-----------------|
| Kool et al., 2019 | Stochastic Beam Search | [arXiv:1903.06059](https://arxiv.org/abs/1903.06059) | Gumbel-Top-k for diverse beam sampling without replacement | Applied to discrete sequence decoding; RBL applies to continuous weight optimization during training |
| Meister et al., 2021 | Conditional Poisson Stochastic Beams | [ACL Anthology](https://aclanthology.org/2021.emnlp-main.52/) | Improved stochastic beam search with better statistical estimators | Sequence decoding for NLP; RBL optimizes neural network parameters |
| Della Croce & T'kindt, 2002 | Recovering Beam Search | [JSTOR:822814](https://www.jstor.org/stable/822814) | **Key precedent**: beam search with recovering step that can override previous decisions | Applied to discrete scheduling; uses dominance properties, not gradient-guided continuous optimization |

### Neural Combinatorial Optimization (Closest Conceptual Neighbors)

| Authors | Paper | Link | Connection | How RBL Differs |
|---------|-------|------|------------|-----------------|
| Pirnay & Grimm, 2024 | Gumbeldore | [OpenReview](https://openreview.net/forum?id=agT8ojoH0X) | Self-improvement via SBS rounds; select best trajectory as pseudo-expert | For CO inference (TSP, CVRP); RBL is an optimizer for NN training |
| Pirnay & Grimm, 2024 | Take a Step and Reconsider | [arXiv:2407.17206](https://arxiv.org/abs/2407.17206) | Follow best path for $s$ steps, then reconsider abandoned alternatives — conceptual ancestor of RBL's retrospective continuation | Discrete sequence space for CO; RBL operates in continuous weight space with gradient unrolling |
| Choo et al., 2022 | Simulation-guided Beam Search | [arXiv:2207.06190](https://arxiv.org/abs/2207.06190) | Neural policy + rollout simulation to guide beam search | For CO inference; RBL is a training-time optimizer |

### Additional Related Work

| Authors | Paper | Link | Connection | How RBL Differs |
|---------|-------|------|------------|-----------------|
| Jaderberg et al., 2017+ | Population Based Training (PBT) + follow-ups | [arXiv:1711.09846](https://arxiv.org/abs/1711.09846) | Population of models with periodic exploit+explore | PBT operates at coarse checkpoint granularity with copy-and-perturb, not per-step k-step unroll-and-argmin |
| Shea & Schmidt, 2024 | Plane Search | [arXiv:2406.17954](https://arxiv.org/abs/2406.17954) | Picks best (LR, momentum) per layer per iteration via 2D subspace search — closest "best of multiple proposals per step" optimizer | No k-step unroll, no inter-step branch retention |
| Or, 2026 | Automatic Stability and Recovery | [arXiv:2601.17483](https://arxiv.org/abs/2601.17483) | Monitors optimizer updates and rolls back on instability — same "don't blindly accept" spirit | Single rollback vs. argmin over live alternatives |
| Zhang et al., 2018/2021 | BPGrad | [CVPR 2018](https://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_BPGrad_Towards_Global_CVPR_2018_paper.pdf), [arXiv:2104.01730](https://arxiv.org/abs/2104.01730) | Branch-and-bound on parameter space using Lipschitz constants | Branches regions, not trajectories; no argmin over endpoints |
| Izmailov et al., 2018 | Stochastic Weight Averaging (SWA) | [arXiv:1803.05407](https://arxiv.org/abs/1803.05407) | Averaging iterates from a single trajectory | Single trajectory averaging, no multi-branch selection |

### Literature Review

The "Recovering Beam Search" paper (Della Croce & T'kindt, 2002) is the closest conceptual ancestor to RBL. It introduced the idea of a beam search that can revisit and override previous decisions via a recovering step. However, it was applied to discrete scheduling problems using dominance properties.

We transfer this "recovering" concept to continuous neural network optimization, using gradient-guided trajectory unrolling instead of discrete dominance checks, and applying it to the training process rather than inference/decoding.

**Most important related work to understand:** Lookaround (Zhang et al., 2023) is the structurally closest published optimizer. Like RBL, it trains B parallel networks from a common checkpoint for k steps. The critical differences are: (1) Lookaround averages endpoints rather than selecting by argmin, and (2) Lookaround does not retain abandoned branches across outer steps. These are precisely the mechanisms that differentiate RBL.

## Current Results (Proof of Concept)

| Benchmark | Adam | RBL | Winner |
|-----------|------|-----|--------|
| Rosenbrock (2D) | 5.53 | 0.11 | **RBL** (98% better) |
| Neural Net (synthetic) | 0.0082 | 0.0000 | **RBL** |
| Rastrigin (2D) | 7.96 | 7.97 | Tie (both stuck in local min) |

### Observations
- RBL shows dramatic improvement on Rosenbrock
- Faster convergence on neural network training
- Branch selection skews toward aggressive strategy (2x LR)
- Both optimizers fail equally on Rastrigin's local minima

## Files

```
rbl_optimizer_poc.py   # Main implementation + tests
requirements.txt       # Dependencies (torch, matplotlib, numpy)
run_rbl.sh            # Install & run script
```

## Quick Start

```bash
pip install -r requirements.txt
python rbl_optimizer_poc.py
```

## Known Limitations / Critique Preparation

1. **Stage 2 not yet implemented:** The current PoC does not retain abandoned branches across steps — the core novelty claim is untested in code.
2. Compute cost: $B \times k$ forward passes per step vs Adam's 1
3. Same-batch selection: May overfit to current batch
4. Toy scale: Only tested on 2D functions and small networks
5. No statistical rigor: Single runs, no error bars
6. Hyperparameter advantage: RBL has more tuning knobs
7. Stage 4 (consistency modulation) is prior art (HGM, arXiv:2506.22479) — not a novel contribution

## Roadmap

### Phase 0: Novelty Gate
- [x] Literature review for related work (see `docs/novelty_review.md`)
- [x] Identify subsumed claims (Stage 4 subsumed by HGM)
- [ ] Update README with narrowed novelty claim and new related work ✓
- [ ] Implement Stage 2 (retrospective branch continuation) in PoC
- [ ] Decide whether to keep or drop Stage 4 as a non-novel mechanism

### Phase 1: Strengthen PoC
- [x] Implement core RBL algorithm (Stages 1, 3, 4 only)
- [x] Test on Rosenbrock function
- [x] Test on simple neural network
- [x] Test on Rastrigin (multi-modal)
- [x] Generate comparison plots
- [ ] Implement Stage 2 (retrospective branch continuation)

### Phase 2: Fair Comparison (*Current*)
- [ ] Compute-matched baseline: Adam with $B \times k$ more steps
- [ ] Lookaround baseline (B branches, k steps, averaging) — required comparison
- [ ] Tune Adam LR to match RBL's preferred branch (2x)
- [ ] Multiple random seeds (5-10) with std dev reporting
- [ ] Wall-clock time comparison

### Phase 3: Scale Up
- [ ] Test on CIFAR-10 with ResNet/CNN
- [ ] Test on real NLP task (small transformer)
- [ ] Memory profiling at scale
- [ ] Investigate surrogate models for cheaper evaluation

### Phase 4: Ablations
- [ ] RBL without retrospective continuation (most critical ablation)
- [ ] RBL with $B=1$ (isolate lookahead contribution)
- [ ] RBL without consistency check (Stage 4) vs. plain HGM modulation
- [ ] Vary $k$ and $B$ systematically
- [ ] Different branch strategies beyond LR variation

### Phase 5: Publication Prep
- [ ] Convergence analysis (theoretical)
- [ ] Write up methodology and results
- [ ] Open source release with clean API


## License

Licensed under the Apache License, Version 2.0 (the "License");
you may not use these files except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0