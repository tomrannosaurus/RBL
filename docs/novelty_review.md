# RBL Novelty Review

Verdict: **Proceed with narrowed contribution.**

- Stage 4 (Consistency Modulation) subsumed by HGM ([2506.22479](https://arxiv.org/abs/2506.22479)). Same concept (cosine similarity → LR modulation), different inputs. **Do not claim as novel.**
- Lookaround ([2306.07684](https://arxiv.org/abs/2306.07684)) is the closest neighbor. B branches, k steps, but averages (not argmin) and no abandoned-branch retention. RBL's argmin + retrospective continuation remain differentiators.
- Stage 2 (Retrospective Branch Continuation) is the load-bearing novelty. **Not yet implemented in the PoC.**
- Surviving novelty: **"keep B parallel k-step unrolled trajectories alive across outer optimizer steps, including those not selected, and pick the next iterate by argmin loss across all live trajectories"** in continuous weight space.

## Stage-by-Stage

| Stage | Mechanism | Closest prior art | Novel? |
|-------|-----------|-------------------|--------|
| 1 | B trajectories from θ_t, different strategies | Lookaround, PBT, Lookahead (B=1) | No |
| 2 | Continue abandoned branches from t−1 alongside fresh at t | RECONSIDER, RECOVERING-BS (discrete only) | **Yes** |
| 3 | argmin loss across all live endpoints | Line/plane search, Lookaround (averages) | Marginal |
| 4 | CosSim(branch update, prev update) → λ ∈ [0.5, 1.5] | HGM — CosSim(gradient, momentum) → LR | **No** |

## HGM vs. Stage 4

**Same:** Cosine similarity between current and historical direction → scale step size. Both boost aligned, dampen oscillatory regions.

**Different:** Inputs (HGM: gradient + momentum buffer; RBL: branch update + prev update). Context (HGM: standalone single-trajectory; RBL: post-hoc after beam selection).

**Verdict:** Keep Stage 4 (may help empirically). Cite HGM. Do not claim novelty.

## Implementation Gap

PoC implements Stages 1, 3, 4. **Stage 2 is missing** (no `self.abandoned_branches`). See `docs/terminology.md` for PoC label corrections.