# Retrospective Beam-Lookahead (RBL) Optimizer

A novel optimizer combining multi-path beam search with gradient-based lookahead for neural network training.

## Idea

Traditional optimizers like Adam are reactive, they adjust based on past gradients. Some, like lookahead, are proactive in their state-projection paradigm. RBL is proactive in that it simulates multiple future trajectories and picks the best one before committing, and retroactive in that it tests states that it could have followed but did not in previous time steps. RBL sees multiple future paths, picks the best one, and maintains abandoned branches to allow self-correction.

### Algorithm (4 Stages)

**Stage 1: Branching**

Spawn $B$ trajectory clones from current weights $\theta_t$, each with a different strategy (e.g., varying learning rates):

$$\theta_{t+k}^{(b)} = \text{Unroll}(\theta_t, \text{Strategy}_b, k \text{ steps}) \quad \forall b \in \{1...B\}$$

**Stage 2: Retrospective Branch Continuation**

Key innovation: also continue the $B-1$ branches **not selected** at time $t-1$. If at $t-1$ we selected branch $b^*$ but abandoned branches $\{b_1, b_2, ...\}$, we also unroll:

$$\theta_{t+k}^{(b_i)} = \text{Unroll}(\theta_{t-1}^{(b_i)}, \text{Strategy}_{b_i}, k \text{ steps}) \quad \forall b_i \notin \{b^*\}$$

This allows RBL to self-correct if an abandoned path leads somewhere better.

**Stage 3: Discrete Selection**

Evaluate loss at all endpoints (current branches + continued abandoned branches):

$$b^* = \underset{b}{\text{argmin}} \ \mathcal{L}(\theta_{t+k}^{(b)})$$

**Stage 4: Retrospective Consistency Check**

Compute alignment between proposed update $\Delta\theta^* = \theta_{t+k}^{(b^*)} - \theta_t$ and momentum history $m_{t-1}$:

$$C = \text{CosineSimilarity}(\Delta\theta^*, m_{t-1})$$

Modulation factor $\lambda = f(C) \in [0.5, 1.5]$ boosts aligned updates, dampens contradictory ones.

**Final Update:**

$$\theta_{t+1} = \theta_t + \alpha \cdot \lambda \cdot (\theta_{t+k}^{(b^*)} - \theta_t)$$

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

The synthesis of:
- **Lookahead** (Zhang et al., 2019) — $k$-step unrolling
- **Beam Search** — $B$ parallel paths with discrete selection
- **Retrospective Correction** — abandoned branch continuation + history-dependent modulation

No existing optimizer combines gradient-guided multi-path exploration with discrete selection and retrospective self-correction for weight updates.

## Foundational Work (Citations)

| Technique | Paper | What We Use |
|-----------|-------|-------------|
| Lookahead | Zhang et al., 2019 | $k$-step unrolling, slow/fast weights |
| Nadam | Dozat, 2016 | Future-peeking concept |
| Evolutionary Strategies | Salimans et al., 2017 | Multi-path exploration (but we use gradients, not noise) |

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

1. Compute cost: $B \times k$ forward passes per step vs Adam's 1
2. Same-batch selection: May overfit to current batch
3. Toy scale: Only tested on 2D functions and small networks
4. No statistical rigor: Single runs, no error bars
5. Hyperparameter advantage: RBL has more tuning knobs

## Roadmap

### Phase 1: Strengthen PoC
- [x] Implement core RBL algorithm
- [x] Test on Rosenbrock function
- [x] Test on simple neural network
- [x] Test on Rastrigin (multi-modal)
- [x] Generate comparison plots

### Phase 2: Fair Comparison (Next)
- [ ] Compute-matched baseline: Adam with $B \times k$ more steps
- [ ] Tune Adam LR to match RBL's preferred branch (2x)
- [ ] Multiple random seeds (5-10) with std dev reporting
- [ ] Wall-clock time comparison

### Phase 3: Scale Up
- [ ] Test on CIFAR-10 with ResNet/CNN
- [ ] Test on real NLP task (small transformer)
- [ ] Memory profiling at scale
- [ ] Investigate surrogate models for cheaper evaluation

### Phase 4: Ablations
- [ ] RBL without retrospective check (isolate beam search contribution)
- [ ] RBL with $B=1$ (isolate lookahead contribution)
- [ ] Vary $k$ and $B$ systematically
- [ ] Different branch strategies beyond LR variation

### Phase 5: Publication Prep
- [ ] Literature review for any missed related work
- [ ] Convergence analysis (theoretical)
- [ ] Write up methodology and results
- [ ] Open source release with clean API


## License

Apache 2.0
