# Cross-shard selector validation — baseline+views4 pools (OFFLINE, nothing submitted)

Date: 2026-10-25. Script: `results/cross_shard/xshard.py` (run with `PYTHONPATH=.`).
Metrics JSON: `results/cross_shard/metrics.json`. Frozen weights: `results/cross_shard/theta_train_shard0.json`, `theta_train_shard1.json`.

Pools = baseline shard candidates ∪ views4 shard candidates, grouped by (task_id, test_idx).
calibCV weights are **fit on one shard's pool and applied to the other** (shards share 0 tasks, so this is a true held-out test). probmul3 = kgmon attempt1 + probmul3 attempt2. All counts are pass@2 (correct grid in the chosen top-2) over scorable outputs.

## Results

### Test shard1 (80 outputs, pool coverage 31), weights trained on shard0
| selector | pass@2 | gained vs kgmon | lost vs kgmon |
|---|---|---|---|
| kgmon (baseline selector) | 23 | — | — |
| probmul3 top2 | 24 | a6f40cea_0 | — |
| **km1 + probmul3-2** | **24** | a6f40cea_0 | — |
| calibCV top2 (frozen shard0) | 23 | a6f40cea_0 | 4c3d4a41_0 |
| **km1 + calibCV-2 (frozen shard0)** | **24** | a6f40cea_0 | — |

### Test shard0 (81 outputs, pool coverage 27), weights trained on shard1
| selector | pass@2 | gained vs kgmon | lost vs kgmon |
|---|---|---|---|
| kgmon (baseline selector) | 20 | — | — |
| probmul3 top2 | 21 | b5ca7ac4_0 | — |
| **km1 + probmul3-2** | **21** | b5ca7ac4_0 | — |
| calibCV top2 (frozen shard1) | 21 | 8f3a5a89_0 | — |
| **km1 + calibCV-2 (frozen shard1)** | **21** | 8f3a5a89_0 | — |

## Read
- Pooling baseline+views4 lifts **coverage** to 31 (shard1) / 27 (shard0) vs kgmon pass@2 23/20 — the gap is selection, not coverage. kgmon leaves 8 covered-but-unpicked on shard1 and 7 on shard0.
- Every alternative beats kgmon by **+1 on each shard** at most. The gains are small and real; do not oversell.
- **Pure calibCV is the only selector that LOSES a task cross-shard** (4c3d4a41_0 on shard1 when weights come from shard0). It overfits its attempt-1 slot. Keeping kgmon as attempt-1 and using the calibrated model only for attempt-2 removes that loss.
- `km1+probmul3-2` and `km1+calibCV-2` are tied: both +1 on each shard, **zero losses**, on truly held-out weights.

## Recommended selector
**km1 + calibCV-2** (kgmon for attempt 1; calibCV, with weights frozen from the opposite shard, for attempt 2), OR the equally-scoring **km1 + probmul3-2**. Both give +1 pass@2 per shard over kgmon with no regressions. Prefer **km1 + probmul3-2** if you want zero trained parameters (nothing to overfit, nothing to ship); prefer **km1 + calibCV-2** if you want the calibrated probabilities for the confidence-driven compute reallocation (cell 9). Do **not** ship pure calibCV top-2.

Frozen weights for the combined notebook (train on shard0, score is applied to the hidden test):
- theta (shard0): beam 0.39, aug7 0.19, aug0 0.18, aug5 0.15, aug6 0.06, aug1 0.04, aug2 0.02; rest 0.
- theta (shard1): aug7 0.29, aug3 0.13, aug4 0.04, count 0.03, aug6 0.02, aug5 0.01; rest 0.
The two shards disagree on which views matter (beam dominates shard0, aug7 shard1), which is why pure calibCV is fragile. Average the two for the ship weights, or use km1+probmul3-2 and skip weights entirely.
