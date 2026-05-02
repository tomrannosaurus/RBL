# RBL Lab Notes

Append-only. Never edit or delete past entries.

## 2026-05-02 — Novelty review

### Search queries run

arXiv, Google Scholar, OpenReview, Semantic Scholar, GitHub. 16 query families: beam-search-as-optimizer, multi-trajectory-NN-training, parallel-rollout, retrospective/abandoned-branches, lookahead-variants, PBT-variants, best-of-k, line/plane-search, branch-and-bound-NN, speculative-execution, trust-region-candidates, MCTS-NN-training, learned-optimizers, cosine-modulation, diverse-multi-trajectory, optimizer-survey. Forward-citation crawls on Lookahead, ES, PBT, Recovering-BS, Reconsider.

Queries that produced no overlap: speculative execution+GD, MCTS for NN training, top-k pooling, trust region methods, Andrychowicz-style learned optimizers.

### Key findings reasoning

**HGM subsumption:** HGM (arXiv:2506.22479) computes CosSim(gradient, momentum_buffer) → adaptive LR scaling. RBL Stage 4 computes CosSim(selected_branch_update, prev_step_update) → λ ∈ [0.5, 1.5] scaling. Same concept (cosine similarity between current and historical direction → modulate step size), different inputs. Initial Claude Code review phrased this as "cosine-similarity LR modulation against momentum — the exact thing RBL claimed as Stage 4" which was confusing because the reviewer used "against" ambiguously and mixed Stages/Phases. Clarified: Same concept, different specific inputs and application context (post-selection vs. standalone). Verdict: keep Stage 4 in algorithm, cite HGM, do not claim novelty.

**Lookaround as closest neighbor:** B parallel branches with data augmentation, k steps, then averages (not argmin). No abandoned-branch retention across outer steps. Structurally closest published optimizer. Must be a baseline.

**Terminology confusion:** Original plan used "phases" for project milestones. Claude Code reviewer introduced "stages" for algorithm components. Resolved: Stages = algorithm (1–4), Phases = roadmap (0–5). Documented in terminology.md.

**PoC gap:** Stage 2 (the load-bearing novelty) is not implemented. PoC has Stages 1, 3, 4 only. Code comments mislabel stages (off by one after Stage 1). See terminology.md for correction table.

**Decision on Stage 4:** Keep it in the algorithm (may help empirically), cite HGM, classify as prior art. Dropping it changes the method without benefit.

### Limitations at time of review

Full papers not read end-to-end (especially LOOKAROUND, HGM algorithm sections). English-language search only. No PhD-thesis or patent search.

## 2026-05-02 — File restructuring

Established current-truth / lab-notebook / changelog convention. Extracted all historical content from current-truth files into NOTES.md. Compressed all current-truth files to contain only present understanding.