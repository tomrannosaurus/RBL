# RBL Reference Corpus

Check here before searching or citing. If a paper is listed with a verdict, use it. Do not re-derive novelty assessments.

## Subsuming / Direct Threats

| ID | Citation | arXiv | Method | Overlaps stage | Verdict |
|----|----------|-------|--------|-----------------|---------|
| LOOKAROUND | Zhang et al. 2023 (NeurIPS) | [2306.07684](https://arxiv.org/abs/2306.07684) | B parallel networks, k steps, diff data augs, average endpoints | Stage 1 | No argmin, no abandoned-branch retention. **Must be baseline.** |
| HGM | Sarkar 2025 | [2506.22479](https://arxiv.org/abs/2506.22479) | CosSim(gradient, momentum) → modulate LR | Stage 4 | **Stage 4 fully subsumed.** Do not claim as novel. |
| LOOKBEHIND-SAM | — | [2307.16704](https://arxiv.org/abs/2307.16704) | k ascent steps for SAM sharpness | Adjacent | Single trajectory, different objective. Not a competitor. |
| PBT | Jaderberg et al. 2017+ | [1711.09846](https://arxiv.org/abs/1711.09846), [2109.13800](https://arxiv.org/abs/2109.13800), [2002.02518](https://arxiv.org/abs/2002.02518), [2506.03225](https://arxiv.org/abs/2506.03225) | Population, periodic exploit+explore | Stage 1 (coarse granularity) | Different granularity. Cite as related. |
| BPGRAD | Zhang et al. 2018/2021 | [CVPR 2018](https://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_BPGrad_Towards_Global_CVPR_2018_paper.pdf), [2104.01730](https://arxiv.org/abs/2104.01730) | B&B on parameter space via Lipschitz constants | Name overlap only | Branches regions, not trajectories. Cite for honesty. |

## Prior Art (Cite, Does Not Subsume)

| ID | Citation | arXiv | Method | Relation to RBL |
|----|----------|-------|--------|-----------------|
| LOOKAHEAD | Zhang et al. 2019 | [1907.08610](https://arxiv.org/abs/1907.08610) | k inner steps, slow/fast averaging | RBL generalizes B=1 → B>1 with selection |
| RECONSIDER | Pirnay & Grimm 2024 | [2407.17206](https://arxiv.org/abs/2407.17206) | Follow best, then reconsider abandoned | Discrete analog of Stage 2 |
| RECOVERING-BS | Della Croce & T'kindt 2002 | [JSTOR:822814](https://www.jstor.org/stable/822814) | Beam search with recovering step | Discrete ancestor of Stage 2 |
| SWA | Izmailov et al. 2018 | [1803.05407](https://arxiv.org/abs/1803.05407) | Average iterates along single trajectory | Different mechanism. One-line cite. |
| PLANE-SEARCH | Shea & Schmidt 2024 | [2406.17954](https://arxiv.org/abs/2406.17954) | Best (LR, momentum) per layer per iteration | Closest "best-of-proposals per step" optimizer. No unroll, no retention. |
| LINE-SEARCH-NN | Kenneweg et al. | [2403.18519](https://arxiv.org/abs/2403.18519) | 1D analog of Plane Search | Not a competitor. |
| AUTOSTAB | Or 2026 | [2601.17483](https://arxiv.org/abs/2601.17483) | Roll back on instability | Same "don't blindly accept" spirit. Single rollback. |
| DUALOPT | — | [2604.22838](https://arxiv.org/abs/2604.22838) | Weight rollback for fine-tuning | Different problem. |
| IOMT-DOIT | Chen et al. 2025/2026 | (survey) | Surrogate-model optimizer selection | Coarse granularity. One-line cite. |
| FRANKENSTEIN | — | [2503.02147](https://arxiv.org/abs/2503.02147) | Combination of optimizer tricks | Not fully reviewed. Confirm no overlap. |
| LOOKAHEAD-THEORY | — | [2509.15776](https://arxiv.org/abs/2509.15776) | Lookahead convergence theory | Useful for RBL theory, not a competitor. |
| GUIDED-DESCENT | — | [2512.18373](https://arxiv.org/abs/2512.18373) | Recent (Dec 2025), not fully reviewed | Worth follow-up. |

## Pre-Review Citations (Already in README)

| ID | Citation | arXiv |
|----|----------|-------|
| LOOKAHEAD | Zhang et al. 2019 | [1907.08610](https://arxiv.org/abs/1907.08610) |
| NADAM | Dozat 2016 | [ICLR Workshop](https://openreview.net/forum?id=OM0jvwB8jIp57ZJjtNEZ) |
| ES | Salimans et al. 2017 | [1703.03864](https://arxiv.org/abs/1703.03864) |
| SBS | Kool et al. 2019 | [1903.06059](https://arxiv.org/abs/1903.06059) |
| POISSON-SBS | Meister et al. 2021 | [ACL Anthology](https://aclanthology.org/2021.emnlp-main.52/) |
| RECOVERING-BS | Della Croce & T'kindt 2002 | [JSTOR:822814](https://www.jstor.org/stable/822814) |
| GUMBELDORE | Pirnay & Grimm 2024 | [OpenReview](https://openreview.net/forum?id=agT8ojoH0X) |
| RECONSIDER | Pirnay & Grimm 2024 | [2407.17206](https://arxiv.org/abs/2407.17206) |
| SIM-BS | Choo et al. 2022 | [2207.06190](https://arxiv.org/abs/2207.06190) |

## Full-Read Status

- [ ] LOOKAROUND — confirm averaging vs. argmin, no abandoned-branch retention
- [ ] HGM — confirm per-parameter vs. per-step modulation; step-size vs. direction
- [ ] RECONSIDER — confirm discrete precedent for Stage 2
- [ ] FRANKENSTEIN — rule out overlap
- [ ] GUIDED-DESCENT — recent, not fully reviewed