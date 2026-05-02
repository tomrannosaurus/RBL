# RBL Terminology

Canonical definitions. Check here before using terms in code, docs, or papers.

## Stages (algorithm components)

| # | Short name | What it does | Novel? |
|---|-----------|--------------|--------|
| 1 | Branching | Spawn B clones from θ_t with different strategies, unroll k steps | **No — precedented by Lookahead (B=1), Lookaround (B>1 w/ avg)** |
| 2 | Retrospective Branch Continuation | Continue B−1 abandoned branches from t−1 alongside fresh at t | **Yes — novel in continuous weight space** (discrete analogs: RECOVERING-BS, RECONSIDER) |
| 3 | Discrete Selection | argmin loss across all live endpoints | Marginal — novel only combined with Stage 2; argmin alone is line/plane search |
| 4 | Consistency Modulation | CosSim(selected update, prev update) → λ ∈ [0.5, 1.5] | **No — subsumed by HGM** |

Do not renumber. Stage 2 is not yet in the PoC (only Stages 1, 3, 4 implemented).

## Phases (project milestones)

| # | Name | Gate |
|---|------|------|
| 0 | Novelty Gate | Proceed / pivot / abandon |
| 1 | Strengthen PoC | Phase 0 verdict = proceed |
| 2 | Fair Comparison | Phase 1 complete |
| 3 | Scale Up | Phases 1+2 healthy |
| 4 | Ablations | Before publication |
| 5 | Publication Prep | After ablations |

Never use "Stage" for a milestone or "Phase" for an algorithm component.

## Notation

| Symbol | Meaning |
|--------|---------|
| B | Beam width (# parallel trajectories) |
| k | Lookahead horizon (gradient steps per branch per outer step) |
| α | Interpolation factor (default 0.5) |
| λ | Stage 4 modulation factor, range [0.5, 1.5] |
| θ_t | Parameters at outer step t |
| θ_{t+k}^{(b)} | Branch b endpoint after k steps from θ_t |
| b* | Selected branch index (argmin) |
| Δθ* | Selected update: θ_{t+k}^{(b*)} − θ_t |

## Novelty classification

- **Novel (defensible):** Stage 2 + Stage 3 in continuous weight space
- **Not novel:** Stage 1 (Lookaround, PBT); Stage 4 (HGM)
- **Marginally novel:** Stage 3 alone (argmin vs. Lookaround's averaging, vs. line search's single-step)

Never claim Stage 1 or Stage 4 as novel. Always note Lookaround/PBT precedence for Stage 1 and HGM precedence for Stage 4.

## Preferred terms

| Use | Avoid | Reason |
|-----|-------|--------|
| Retrospective branch continuation | recovery, reconsideration step | Avoids confusion with discrete-decoding terms |
| Discrete selection | beam search step, argmin step | Unambiguous |
| Consistency Modulation | retrospective consistency check | "Retrospective" is reserved for Stage 2 |
| Outer step / inner step | macro/micro step, slow/fast step | Aligns with Lookahead terminology |
| Abandoned branch | stale branch, old branch | Canonical term for non-selected branches |
| Live endpoints | candidates | Includes fresh + abandoned endpoints |

## PoC code label corrections

`rbl_optimizer_poc.py` mislabels stages:

| Code comment | Correct |
|---|---|
| `# stage 1: Branching and Unrolling` | Stage 1: Branching |
| `# stage 2: Discrete Selection (Beam Search)` | Stage 3: Discrete Selection |
| `# stage 3: Retrospective Consistency Check` | Stage 4: Consistency Modulation |
| `# stage 4: Final Slow Weight Update` | Part of Stage 4 |