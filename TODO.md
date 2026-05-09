# RBL TODO

See `docs/terminology.md` for Stage/Phase definitions, `docs/references.md` for paper IDs.

## 0. Novelty Gate

Complete. Verdict in `docs/novelty_review.md`.

### 0.1 Before §1

- [ ] **Implement Stage 2** in `rbl_optimizer_poc.py` — abandoned-branch tracking, branches not selected at t−1 continued at t. Core novelty, currently missing.
- [ ] **Decide on Stage 4** — keep (cite HGM) or drop. Cannot claim as novel.
- [ ] **Align PoC code labels** with `docs/terminology.md`.
- [ ] **Resolve discrete beam-search follow-up reads** — full-read `SBS`, `GUMBELDORE`, `SIM-BS`, and `RECONSIDER`; confirm they remain background/conceptual ancestors or update verdicts in `docs/references.md`.

## 1. Strengthen the PoC

- [ ] Implement Stage 2 (retrospective branch continuation)
- [ ] Compute-matched baseline: Adam with `B × k` more steps
- [ ] **Lookaround baseline** (B branches, k steps, averaging) — required
- [ ] Adam LR grid including RBL's preferred 2×
- [ ] 5–10 seeds per condition; mean ± std
- [ ] Wall-clock time alongside step count

## 2. Ablations

- [ ] `B=1` → ≈ Lookahead optimizer
- [ ] No retrospective continuation → **most critical ablation**
- [ ] No Stage 4 → quantify HGM-prior-art contribution
- [ ] Stage 4 replaced with plain HGM modulation → direct comparison
- [ ] Random branch strategies vs. LR-only
- [ ] Sweep `k ∈ {1, 3, 5, 10}`, `B ∈ {2, 3, 5, 8}`

## 3. Scale up

- [ ] CIFAR-10 / small ResNet
- [ ] Memory profile (B copies of weights)
- [ ] Surrogate evaluation
- [ ] Small NLP task

## 4. Theory / write-up

- [ ] Convergence sketch (B=1, k=1 doesn't break base optimizer)
- [ ] Clean-room reimplementation with documented API
- [ ] Draft methodology + results

## Open questions

- Same-batch vs. held-out batch for Stage 3
- How long to keep abandoned branches alive? (Currently: one step. Decaying window?)
- Does Stage 4 help beyond beam selection? (Ablations will answer.)
- If HGM-style modulation ≡ RBL Stage 4, frame as "HGM applied post-selection"
