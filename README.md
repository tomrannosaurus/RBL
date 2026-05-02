# Retrospective Beam-Lookahead (RBL) Optimizer

Multi-path beam search with gradient-based lookahead and retrospective self-correction for neural network training.

> **See `docs/terminology.md`** for canonical Stage/Phase definitions and **`docs/references.md`** for the full paper corpus with IDs and verdicts.

## Idea

Traditional optimizers like Adam are reactive. Lookahead is proactive in its state-projection paradigm. RBL is both proactive (simulates multiple future trajectories, picks the best) and retroactive (continues abandoned branches from previous steps to allow self-correction).

### Algorithm (4 Stages)

**Stage 1: Branching** — Spawn $B$ trajectory clones from $\theta_t$, each with a different strategy (e.g., varying learning rates), unrolled $k$ steps:

$$\theta_{t+k}^{(b)} = \text{Unroll}(\theta_t, \text{Strategy}_b, k \text{ steps}) \quad \forall b \in \{1...B\}$$

**Stage 2: Retrospective Branch Continuation** *(load-bearing novelty)* — Continue the $B-1$ branches **not selected** at $t-1$ alongside fresh branches:

$$\theta_{t+k}^{(b_i)} = \text{Unroll}(\theta_{t-1}^{(b_i)}, \text{Strategy}_{b_i}, k \text{ steps}) \quad \forall b_i \notin \{b^*\}$$

> Not yet implemented in the current PoC. See `TODO.md`.

**Stage 3: Discrete Selection** — Argmin over all live endpoints (fresh + retrospective):

$$b^* = \underset{b}{\text{argmin}} \ \mathcal{L}(\theta_{t+k}^{(b)})$$

**Stage 4: Consistency Modulation** *(prior art — see HGM below]* — Scale the final update by alignment with previous direction:

$$C = \text{CosineSimilarity}(\Delta\theta^*, \Delta\theta_{t-1}), \quad \lambda = f(C) \in [0.5, 1.5]$$

$$\theta_{t+1} = \theta_t + \alpha \cdot \lambda \cdot (\theta_{t+k}^{(b^*)} - \theta_t)$$

> HGM ([2506.22479](https://arxiv.org/abs/2506.22479)) independently proposed cosine-similarity LR modulation. Stage 4 uses different inputs (branch update vs. prev update) but the core idea is HGM's. **Not claimed as novel.**

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

**Novel:** Stage 2 (retrospective branch continuation) + Stage 3 (argmin over live endpoints including stale ones) in continuous weight space. This transfers the "recovering" concept from discrete beam search (Della Croce & T'kindt 2002; Pirnay & Grimm 2024) to NN optimization.

**Not novel:** Stage 1 (multi-branch unrolling — LOOKAROUND, PBT); Stage 4 (cosine-similarity LR modulation — HGM).

**Marginally novel:** Stage 3 alone (argmin over diverse k-step strategies including stale ones; vs. Lookaround's averaging, vs. line search's single-step).

## Related Work

### Subsuming / Direct Threats

| Reference ID | Paper | Overlaps | Key difference from RBL |
|-------------|-------|----------|--------------------------|
| LOOKAROUND | Zhang et al. 2023 [NeurIPS] | Stage 1 (branching) | Averages endpoints (not argmin); no abandoned-branch retention. **Must be a baseline.** |
| HGM | Sarkar 2025 | Stage 4 (consistency modulation) | Single-trajectory; uses gradient vs. momentum. **Stage 4 is HGM's idea.** |

### Conceptual Ancestors

| Reference ID | Paper | Connection |
|-------------|-------|------------|
| RECOVERING-BS | Della Croce & T'kindt 2002 | Beam search with recovering step → discrete ancestor of Stage 2 |
| RECONSIDER | Pirnay & Grimm 2024 | Follow best, then reconsider → discrete analog of Stage 2 |
| LOOKAHEAD | Zhang et al. 2019 | k-step unrolling, slow/fast weights → RBL generalizes B=1 → B>1 with selection |
| PBT | Jaderberg et al. 2017+ | Population with exploit+explore → coarser-granularity multi-branch |
| PLANE-SEARCH | Shea & Schmidt 2024 | Best (LR, momentum) per iteration → closest "best-of-proposals per step" optimizer |

### Additional Context

| Reference ID | Paper | Connection |
|-------------|-------|------------|
| BPGRAD | Zhang et al. 2018/2021 | B&B on parameter space (branches regions, not trajectories) |
| AUTOSTAB | Or 2026 | Rollback on instability (single rollback, not argmin over alternatives) |
| SWA | Izmailov et al. 2018 | Single-trajectory averaging |
| SBS / POISSON-SBS / GUMBELDORE / SIM-BS | Various | Discrete beam search for CO inference |
| NADAM | Dozat 2016 | Nesterov-style future-peeking |
| ES | Salimans et al. 2017 | Population-based weight-space exploration |

Full citations with arXiv links in `docs/references.md`.

## Current Results (Proof of Concept)

| Benchmark | Adam | RBL | Winner |
|-----------|------|-----|--------|
| Rosenbrock (2D) | 5.53 | 0.11 | **RBL** (98% better) |
| Neural Net (synthetic) | 0.0082 | 0.0000 | **RBL** |
| Rastrigin (2D) | 7.96 | 7.97 | Tie |

**Observations:** RBL shows strong improvement on Rosenbrock; branch selection skews toward aggressive strategy (2× LR); both optimizers fail equally on Rastrigin's local minima.

## Known Limitations

1. **Stage 2 not yet implemented** — core novelty claim untested in code
2. Compute cost: $B \times k$ forward passes per step vs. 1 for Adam
3. Same-batch selection may overfit
4. Toy scale only; no statistical rigor (single runs, no error bars)
5. Stage 4 is prior art (HGM)

## Files

```
rbl_optimizer_poc.py   # Implementation (Stages 1, 3, 4 only; Stage 2 pending)
requirements.txt       # torch, matplotlib, numpy
run_rbl.sh             # Install & run
docs/CONVENTIONS.md    # File-class rules (current-truth vs. lab-notebook vs. changelog)
docs/terminology.md    # Canonical Stage/Phase definitions
docs/references.md     # Paper corpus with IDs and verdicts
docs/novelty_review.md # Novelty assessment (current truth)
NOTES.md               # Lab notebook (chronological reasoning, search logs)
CHANGELOG.md           # Structured change log
TODO.md                # Work tracker
```

## Quick Start

```bash
pip install -r requirements.txt
python rbl_optimizer_poc.py
```

## Roadmap

### Phase 0: Novelty Gate
- [x] Literature review (see `docs/novelty_review.md`)
- [x] Identify subsumed claims (Stage 4 → HGM)
- [ ] Implement Stage 2 in PoC
- [ ] Decide: keep Stage 4 (cite HGM) or drop

### Phase 1: Strengthen PoC
- [x] Core RBL implementation (Stages 1, 3, 4)
- [x] Rosenbrock, neural net, Rastrigin tests
- [ ] Implement Stage 2 (retrospective branch continuation)

### Phase 2: Fair Comparison
- [ ] Compute-matched Adam baseline (`B × k` more steps)
- [ ] Lookaround baseline (required)
- [ ] Multi-seed + wall-clock reporting

### Phase 3: Scale Up
- [ ] CIFAR-10 / ResNet
- [ ] Memory profiling
- [ ] Small NLP task

### Phase 4: Ablations
- [ ] No retrospective continuation (critical)
- [ ] B=1 (Lookahead equivalent)
- [ ] Stage 4 vs. plain HGM modulation
- [ ] k and B sweeps

### Phase 5: Publication Prep
- [ ] Convergence analysis
- [ ] Methodology + results draft
- [ ] Clean API + open source release

## License

Apache License 2.0. See http://www.apache.org/licenses/LICENSE-2.0