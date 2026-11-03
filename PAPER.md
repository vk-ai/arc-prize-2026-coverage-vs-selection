# Coverage, Not Selection: Where a Qwen3-4B Test-Time-Training Pipeline Loses Points on ARC-AGI-2

**ARC Prize 2026 Paper Track.** Author: Vinay Kumar K M (Kaggle: vinaykm).
Linked code submission: https://www.kaggle.com/code/vinaykm/arc-f1-smoke-py311 (Version 4, public LB 27.64).
License: CC0 1.0 for all of my own code. Code and reports: https://github.com/vk-ai/arc-prize-2026-coverage-vs-selection

## Abstract
Strong open ARC-AGI-2 pipelines fine-tune a small LLM on each task at test time (TTT), sample candidate grids, and pick two answers by scoring candidates under several augmented "views" (a product of experts, PoE). We saved every candidate and its per-view scores from Kaggle runs of a Qwen3-4B pipeline (NVARC `qwen3_4b_grids15_sft139`) on the public evaluation set. We then asked which part loses more points: generating a correct candidate (coverage) or choosing it (selection). On held-out shards, calibrated, NVARC-style and min-aug selectors gain at most +1 to +3 outputs over the default selector, and several gain nothing. Adding more decoding views (views4) or pooling candidates from several runs raises coverage more consistently. Most test outputs never get a correct candidate at all. We built F1, a deadline-aware notebook: a full baseline pass, then a views4 pass on the least-confident tasks, with kgmon for attempt 1 and a second selector for attempt 2. F1 scored 27.64 on the public leaderboard (LB), which is inside the 26.9–33.9 range that unmodified public notebooks get from run-to-run noise.

## 1. Introduction
ARC-AGI tests skill acquisition on novel grid puzzles from a few examples. Current Kaggle leaders use the ARChitects/NVARC recipe: a small LLM, per-task TTT, sampling, and augmentation-based PoE scoring (Franzen et al., 2025). Most public tuning targets the selector, so we started there. Our offline data says the selector is close to saturated and the larger loss is upstream, in generation.

## 2. Prior work
- **ARChitects / Product of Experts** (Franzen et al., arXiv:2505.07859): augmented-view scoring for both candidate generation and selection. We reuse it.
- **NVARC (ARC Prize 2025):** scores a candidate by its occurrence count adjusted by the geometric mean of augmented log-probabilities. The exact formula is unpublished, so we tested reconstructions of it (marked HYPOTHESIS).
- **MindsAI + Tufa Labs (ARC Prize 2025):** a 660M CodeT5 model with TTT and about 10k augmented votes per task. They put massive voting into coverage; we study a cheap version of that trade-off.
- **Public 2026 notebooks** (e.g. mikelou1, "ARC AGI2 LB33.89 Minimal Perfpatch"): the same model scores 26.9–33.9 on the LB depending on the run.

## 3. Approach
**Pipeline.** Qwen3-4B (grids15 SFT139) loaded with unsloth, per-task TTT, DFS/beam sampling, and re-scoring of each candidate under 8 views (D8 transform, colour permutation, example shuffle). The default selector, *kgmon*, ranks a candidate by its sample count minus its mean NLL across views.

**Instrumentation.** We ran the notebook in two shards of the public evaluation set and saved every candidate with its per-view NLLs (shard0 and shard1, which share no tasks). Our offline re-implementation of kgmon matches Kaggle's logged score exactly on shard0 (13.00/54).

**Coverage levers.**
- *views4:* 4 decoding views instead of 2.
- *ttt32:* 32 TTT augmentations instead of 16.
- *Pooling:* merge candidates from several runs before selection.

**Selection levers.**
- *calibCV:* per-view PoE weights fitted by L-BFGS-B with task-level 5-fold CV, or frozen from the other shard.
- *probmul3:* a product-of-probabilities score.
- *km1 + X-2:* kgmon for attempt 1 and selector X for attempt 2. Attempt 1 is untouched, so this cannot lose a kgmon top-1 hit.
- *NVARC reconstruction:* log(n) − mean NLL.
- *min-aug:* the worst-view NLL as a feature.
- *Symbolic filters:* output shape and colour.

**F1 submission notebook.**
1. Run a full baseline pass.
2. Rank tasks by confidence.
3. Run views4 on the least-confident tasks, stopping 20 minutes before the deadline.
4. Pool the candidates from both passes.
5. Pick attempt 1 with kgmon and attempt 2 with the second selector.

The notebook runs on 4×L4 with internet off, Python 3.11 and the pinned 2025-07-08 environment, and sets `PYTHONHASHSEED`.

## 4. Results (public evaluation, offline; pass@2 counts of test outputs)
**Coverage levers, shard0 (kgmon selector):**

| Run | Coverage | Top-2 correct | Task score |
|---|---|---|---|
| baseline | 23 | 18 | 13.0 |
| ttt32 | 23 | 18 | 14.0 |
| **views4** | **25** | **20** | **15.0** |
| pool baseline+views4 | 27 | 20 | 14.5 |
| pool all three | 28 | 20 | 14.5 |

Pooling lifts coverage from 23 to 28. Top-2 correct moves only from 18 to 20, so the extra correct candidates are mostly not picked.

**Selection levers, pooled shard0 (82 outputs, oracle 28):**

| Selector | Top-2 correct | Task score |
|---|---|---|
| kgmon | 20 | 14.5 |
| calibCV (5-fold by task) | 22 | 17.0 |
| min-aug + calibCV | 22 | 17.0 (no gain) |

calibCV converts 4 of the 8 covered-but-missed outputs but loses 2 that kgmon had.

**Cross-shard held-out (weights from the opposite shard):**

| Selector | shard0 (81) | shard1 (80) |
|---|---|---|
| kgmon | 20 | 23 |
| km1 + probmul3-2 | 21 | 24 |
| km1 + calibCV-2 | 21 | 24 |
| pure calibCV | 21 | 23 (loses 1 task) |

**NVARC-style reconstruction** (pooled shard0 baseline+ttt32): 20/80, against 19/80 for kgmon. On shard1 it scores 24/79, the same as kgmon. The symbolic filters change no picks, and the shape filter removes correct answers (1 on shard0, 5 on shard1).

**Earlier full-evaluation run** (108 tasks, 151 outputs):
- kgmon pass@2 is 29.32%, against an oracle coverage of 34.72%.
- Uniform PoE scores 28.86%, and calibrated per-view weights score 28.40%.
- A small DSL verifier expert added no correct candidates.
- 100 of the 151 outputs (66%) have no correct candidate. An exploratory logistic-regression detector predicts that "none-correct" condition with CV AUROC 0.853. We have not tested whether using it to trigger extra compute helps.

**Leaderboard:**

| Entry | Public LB |
|---|---|
| F1 smoke commit (shard0 offline) | 14.50 / 60 |
| **F1, Version 4 (my method)** | **27.64** |
| Unchanged fork of mikelou1's public Perfpatch notebook (not my method) | 31.81 (best, rank about 584 of 3,627) |

The 31.81 entry is credited fully to mikelou1. It is reported only for context and does not support our claims. Its gap of about 4 points to F1 is inside the ±3 rerun noise we observed. The public LB uses about 50% of the hidden tasks, and private scores are *[pending, announced Dec 4]*.

## 5. Negative results
- Calibrated selection did not beat kgmon on the full-evaluation run (28.40% vs 29.32%).
- Pure calibCV overfits. The two shards disagree on which views matter (beam weight dominates on shard0, aug7 on shard1).
- min-aug, the NVARC reconstruction and the symbolic filters gave about 0 gain.
- ttt32 did not raise coverage on shard0.
- F1 did not beat public notebooks on the LB.

## 6. Limitations
- **Tiny samples:** about 80 outputs per shard, so 1 output is roughly 1.2 points and most of the differences above are within noise.
- **One base model and pipeline.**
- **Excluded tasks:** 5 tasks were dropped because the Kaggle and GitHub test-input counts differ.
- **Hash-salted colour permutations:** these hurt reproducibility, which is why we fixed `PYTHONHASHSEED`.
- **Untested idea:** the confidence-triggered second pass (detector-targeted vs random control) was planned but never run on GPU.

## 7. Conclusion
For this TTT pipeline, the oracle-to-pick gap is small: 5–8 outputs per shard. The safe win is km1 + X-2, at +1 per shard with no losses. The larger loss is in coverage: two-thirds of outputs never receive a correct candidate. Extra test-time compute should go into generating more diverse candidates, aimed at low-confidence tasks, rather than into better selectors.

## Acknowledgements and reproducibility
- Credits: NVARC (model), the ARChitects (PoE recipe), mikelou1 (Perfpatch notebook), kgmon (selector), and unsloth.
- Code: https://www.kaggle.com/code/vinaykm/arc-f1-smoke-py311.
- Candidate dumps and scoring scripts: https://github.com/vk-ai/arc-prize-2026-coverage-vs-selection (CC0 1.0).

## References
Franzen, D., Disselhoff, J., Hartmann, D. "Product of Experts with LLMs: Boosting Performance on ARC Is a Matter of Perspective." arXiv:2505.07859, 2025.
MindsAI & Tufa Labs, ARC Prize 2025 solution writeup (Kaggle).
