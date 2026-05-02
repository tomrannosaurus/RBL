# RBL TODO

Tracks work picked up after the December PoC. Granular companion to the high-level roadmap in `README.md`. Items are ordered; do not start a phase until the prior one's gate is cleared.

---

## 0. Novelty Gate (do this first)

**Status (2026-05-02): first pass complete, revised.** See `docs/novelty_review.md` for the full review. Verdict: **proceed with narrowed contribution.**

Key findings:
- **Stage 4 (Retrospective Consistency Check)** is subsumed by HGM (arXiv:2506.22479). Both use cosine similarity between a current direction and a historical direction to modulate learning rate. HGM uses gradient vs. momentum buffer; RBL uses branch update vs. previous update — different inputs, same core idea. Stage 4 must not be claimed as novel. Either drop it or cite HGM and acknowledge it as prior art.
- **Lookaround** (NeurIPS 2023, arXiv:2306.07684) is the closest direct neighbor. It runs B parallel branches with different data augmentations for k steps but **averages** (not argmin) and does **not** retain abandoned branches. RBL's argmin + retrospective continuation are still differentiators.
- **Stage 2 (Retrospective Branch Continuation)** — the load-bearing novel contribution — **is not yet implemented in the PoC.** The current PoC only implements branching, argmin selection, and consistency modulation. Retrospective continuation must be added before experiments can validate the novelty claim.
- The narrowed novelty claim is: **maintaining a moving window of B k-step unrolled trajectories in continuous weight space, including those not selected at prior steps, and updating by argmin across all live endpoints.**

### 0.1 Before proceeding to §1

- [ ] **Implement Stage 2 (Retrospective Branch Continuation)** in the PoC. Add abandoned-branch tracking so that branches not selected at step t−1 are continued alongside fresh branches at step t. Without this, the core novelty claim cannot be tested.
- [ ] **Update `README.md`** with revised Related Work and narrowed Novelty Claim per `docs/novelty_review.md` §6.
- [ ] **Align PoC code comments** with README stage labels (currently off by one after Stage 1).
- [ ] **Decide on Stage 4**: Drop it from the novelty claim (cite HGM as prior art), or keep the mechanism but attribute it to HGM. Either way, it cannot be claimed as novel.

### 0.2 Verdict template (filled in `docs/novelty_review.md`)

For each closely related paper found, record:
- Citation, link, year, venue
- One-paragraph summary of method
- Specific overlap with RBL (which of: multi-path / lookahead / discrete selection / retrospective continuation / consistency modulation)
- Specific differences
- Verdict on whether it subsumes RBL

Final section: **proceed with narrowed contribution**, with reasoning. See `docs/novelty_review.md`.

### 0.3 Exit criterion

- (b) Papers subsume part of RBL → revise novelty claim in `README.md` and proceed with narrowed contribution. **Current status.**

---

## 1. Strengthen the PoC (gated on §0 revisions)

These map to README's Phase 2. Don't start until §0 revisions are complete.

- [ ] Implement Stage 2 (retrospective branch continuation) in `rbl_optimizer_poc.py`.
- [ ] Compute-matched baseline: Adam with `B × k` more gradient steps per RBL step (currently RBL gets a free compute advantage).
- [ ] **Lookaround baseline**: compute-matched comparison against Lookaround (B branches, k steps, averaging). This is the closest published neighbor and must be included.
- [ ] Tune Adam LR over a grid that includes RBL's preferred 2× LR branch — make sure RBL isn't just discovering "use a higher LR".
- [ ] 5–10 seeds per condition; report mean ± std, not single runs.
- [ ] Wall-clock time alongside step count (the honest cost story).
- [ ] Add a "compute-equalized" column to the results table in README.

## 2. Ablations (before scaling)

Ordered to isolate each mechanism. Each ablation is a one-line config change against the PoC.

- [ ] `B=1` → isolates lookahead contribution (should ≈ Lookahead optimizer).
- [ ] No retrospective continuation → isolates beam contribution. **This is the most important ablation** — if removing retrospective continuation doesn't hurt, the novelty claim is in trouble.
- [ ] No consistency check (Stage 4) → isolates the modulation contribution. Since HGM subsumes this, quantify how much it adds on top of beam selection alone.
- [ ] Consistency check replaced with plain HGM modulation (gradient-vs-momentum cosine similarity) → direct comparison to HGM's mechanism.
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
- Does the consistency check (Stage 4) actually help, or is it cosmetic? §2 ablations will tell us. Given HGM prior art, the answer determines whether we keep or drop it.
- Stage 4 comparison: if HGM-style modulation (gradient vs. momentum) performs equivalently to RBL's formulation (branch update vs. previous update), then Stage 4 is truly just HGM applied post-selection and should be framed accordingly.