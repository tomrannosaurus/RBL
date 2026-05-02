# RBL TODO

Tracks work picked up after the December PoC. Granular companion to the high-level roadmap in `README.md`. Items are ordered; do not start a phase until the prior one's gate is cleared.

---

## 0. Novelty Gate (do this first)

**Status (2026-05-02): first pass complete.** See `docs/novelty_review.md` for the full review. Verdict: **proceed with narrowed contribution.** Stage 4 (cosine-similarity modulation) is subsumed by HGM (arXiv:2506.22479) and must be dropped or re-attributed. Lookaround (NeurIPS 2023) is a closer ancestor than the README acknowledges and must be added as a baseline. The retrospective-branch-continuation mechanic in continuous weight space appears to be the load-bearing novel contribution. Before §1, do the README revisions listed in `docs/novelty_review.md` §4.

### Original plan (kept for reference):


**Goal:** Before spending more engineering effort, confirm RBL — *gradient-guided multi-path unrolling with discrete endpoint selection AND continuation of abandoned branches across optimizer steps* — is not already published. Output is a written verdict: **proceed / pivot / abandon**.

The existing `README.md` "Related Work" table is a starting point, not a literature review. It was written from memory and from the references that motivated the design; it has not been stress-tested against an adversarial search.

### 0.1 Search plan

Run each of these and log hits in `docs/novelty_review.md` (create the file). Cast a wide net — false positives are cheap, missed prior art is expensive.

- [ ] **arXiv full-text search** for each query below; review titles for the top 50 hits each, abstracts for anything plausibly related:
  - `"beam search" optimizer neural network training`
  - `multi-path OR multi-trajectory optimizer`
  - `lookahead optimizer` (and crawl forward-citations of Zhang 2019 via Semantic Scholar / Connected Papers)
  - `population based training` (Jaderberg 2017) — and follow-ups
  - `ensemble optimizer` / `ensemble of trajectories` for NN training
  - `branch and bound` continuous neural training
  - `rollout` / `unroll` based optimizer
  - `meta-learning optimizer` survey papers from 2022–2025 (catch anything we'd otherwise miss)
- [ ] **Google Scholar** the same queries; pay attention to thesis chapters and workshop papers, which are easy to miss.
- [ ] **OpenReview** search ICLR / NeurIPS 2023, 2024, 2025 submissions (including rejected) for the queries above. Rejected papers matter — they establish prior art and may explain why the idea didn't work.
- [ ] **Forward citations** on each "Core Foundations" row in the README (Lookahead, Nadam, ES). Use Semantic Scholar API or Connected Papers.
- [ ] **Optimizer benchmark / survey papers** (2023+) — they catalog the field and will surface anything we're missing.
- [ ] Specifically check for: the *retrospective branch continuation* mechanic (re-evaluating not-selected branches at later steps). This is the piece I believe is novel; if it exists anywhere, that's the most important finding.

### 0.2 Verdict template (fill into `docs/novelty_review.md`)

For each closely related paper found, record:
- Citation, link, year, venue
- One-paragraph summary of method
- Specific overlap with RBL (which of: multi-path / lookahead / discrete selection / retrospective continuation / consistency check)
- Specific differences
- Verdict on whether it subsumes RBL

Final section: **proceed / pivot / abandon**, with reasoning. If "pivot," sketch what the pivoted contribution would be.

### 0.3 Exit criterion

Either:
- (a) No paper found that subsumes RBL → proceed to §1.
- (b) A paper subsumes part of RBL → revise novelty claim in `README.md` and proceed with narrowed contribution.
- (c) A paper subsumes all of RBL → stop and tell me; we'll decide whether to abandon or extend in a different direction.

---

## 1. Strengthen the PoC (gated on §0 verdict = proceed)

These map to README's Phase 2. Don't start until §0 is closed.

- [ ] Compute-matched baseline: Adam with `B × k` more gradient steps per RBL step (currently RBL gets a free compute advantage).
- [ ] Tune Adam LR over a grid that includes RBL's preferred 2× LR branch — make sure RBL isn't just discovering "use a higher LR".
- [ ] 5–10 seeds per condition; report mean ± std, not single runs.
- [ ] Wall-clock time alongside step count (the honest cost story).
- [ ] Add a "compute-equalized" column to the results table in README.

## 2. Ablations (before scaling)

Ordered to isolate each mechanism. Each ablation is a one-line config change against the PoC.

- [ ] `B=1` → isolates lookahead contribution (should ≈ Lookahead optimizer).
- [ ] No retrospective continuation → isolates beam contribution.
- [ ] No consistency check (Stage 4) → isolates the modulation contribution.
- [ ] Random branch strategies vs. LR-only → does diversity matter beyond LR?
- [ ] Sweep `k ∈ {1, 3, 5, 10}` and `B ∈ {2, 3, 5, 8}`.

## 3. Scale up (gated on §1 + §2 looking healthy)

- [ ] CIFAR-10 with a small ResNet — first real test.
- [ ] Memory profile at scale (`B` copies of weights is the obvious bottleneck).
- [ ] Surrogate evaluation: can we score branches with a proxy cheaper than full forward passes?
- [ ] One small NLP task (e.g., char-level transformer on tiny shakespeare) to check generality.

## 4. Theory / write-up

- [ ] Convergence sketch — at minimum, show RBL doesn't break convergence guarantees of the base optimizer when `B=1, k=1`.
- [ ] Clean-room reimplementation with documented API.
- [ ] Draft methodology + results sections.

---

## Open questions to resolve along the way

- Same-batch selection vs. held-out batch for Stage 3 — current PoC may be overfitting to the current batch. Empirically test both.
- How long to keep abandoned branches alive? PoC keeps them one step; should it be a decaying window?
- Does the consistency check (Stage 4) actually help, or is it cosmetic? §2 will tell us.
