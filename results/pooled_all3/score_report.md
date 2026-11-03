# 3-way pool (baseline ∪ ttt32 ∪ views4), shard0 — offline, nothing submitted
**Conclusion:** Best honest selector = 5-fold CV calibrated PoE (refit per fold, shard1 + other folds): **pass@2 22/82, task_score 17.0** vs kgmon 20 / 14.5 on the same pool. Converts 4 of 8 covered-but-missed, loses 2 kgmon hits. Oracle ceiling = coverage 28. Verifier expert is inert (0 exact-fit programs on 59 tasks) → adds nothing. Gap to ceiling is 6; selection gains are small and noisy (±2) → next lever is coverage (views4 on shard1 + pooling), with calibCV as selector.

Pool: 82 outputs / 59 tasks, coverage 28, 1413 unique cands.

| selector | pass@2 | task_score | converted (of 8) | lost vs kgmon |
|---|---|---|---|---|
| kgmon | 20 | 14.5 | 0 | 0 |
| probmul3 | 21 | 15.5 | 1 | 0 |
| rrf(km,pm) | 21 | 15.5 | 1 | 0 |
| km1+pm2 split | 21 | 15.5 | 1 | 0 |
| **calibCV PoE** | **22** | **17.0** | 4 | 2 |
| km1+calibCV2 | 21 | 16.0 | 2 | 1 |
| calibCV+verifier | 22 | 17.0 | 4 | 2 |
| calibCV+ver+inject | 22 | 17.0 | 4 | 2 |
| oracle | 28 | 20.5 | 8 | 0 |

Verifier: 20 s budget/task, 12.9 s total, 0 exact-fit programs, verifier weight fit = 0.0 in all folds. Script: score3.py.
Safe zero-loss option: km1+pm2 (21, no losses).

## Min-over-augmentation feature (2026-10-25, offline) — script minaug.py, metrics_minaug.json
**Conclusion:** min-aug adds nothing. Fused with calibCV it ties calibCV (22/82, 17.0, same 4 converted/2 lost); alone it ≈ kgmon (20). Keep calibCV.

| selector (5-fold CV, refit per fold) | pass@2 | task_score | converted (of 8) | lost |
|---|---|---|---|---|
| kgmon | 20 | 14.5 | 0 | 0 |
| calibCV | 22 | 17.0 | 4 | 2 |
| minaug alone (min over 8 view logp) | 20 | 15.0 | 2 | 2 |
| minaug + count | 19 | 14.5 | 1 | 2 |
| minaug_raw (per-guess min score_aug) + count | 20 | 15.0 | 1 | 1 |
| calibCV + minaug | 22 | 17.0 | 4 | 2 |
| calibCV + minaug_raw | 22 | 17.0 | 4 | 2 |
