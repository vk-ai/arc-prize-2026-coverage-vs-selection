| Variant | pass@2 | oracle gap | ECE |
|---|---|---|---|
| A0 kgmon (notebook default) | 29.32% | 5.40% | – |
| A0' full_probmul_3 | 29.32% | 5.40% | – |
| A1u uniform PoE | 28.86% | 5.86% | – |
| A1 calibrated per-view | 28.40% | 6.33% | 0.433 |
| A2 + verifier | 28.40% | 6.33% | 0.433 |

Oracle coverage: LLM candidates 34.72%, +verifier 34.72%. Tasks 108, test outputs 151.
| HYPOTHESIS offline pool shard0 (baseline+ttt32), kgmon | 80 outputs/58 tasks | coverage 27, top-2 19 (23.8%) vs 18 each run alone | not a live run |
| HYPOTHESIS offline pooled shard0, calib PoE 5-fold CV (shard1+other folds) | 80 outputs | top-2 22 (27.5%) vs kgmon 19 (23.8%); +4 converted / -1 lost | not a live run |
| HYPOTHESIS offline pooled shard0, probmul3 | 80 outputs | top-2 20 vs kgmon 19; +1/-0 | not a live run |
