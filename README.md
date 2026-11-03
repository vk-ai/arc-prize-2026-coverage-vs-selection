# Coverage, Not Selection — ARC Prize 2026 (ARC-AGI-2) Paper Track

Code and reports for the paper by Vinay Kumar K M (Kaggle: vinaykm). Full text: [PAPER.md](PAPER.md).

**Thesis:** in a Qwen3-4B test-time-training pipeline (NVARC `qwen3_4b_grids15_sft139`), most lost points come from never generating a correct candidate (coverage), not from choosing the wrong one (selection). Better selectors (calibrated, NVARC-style, min-aug) gain only +1 to +3 outputs on held-out shards; more decoding views (views4) and pooling runs raise coverage more reliably.

**F1 notebook** (full baseline pass + deadline-aware views4 pass on least-confident tasks; kgmon attempt 1, second selector attempt 2): https://www.kaggle.com/code/vinaykm/arc-f1-smoke-py311 — Version 4, public LB 27.64. Offline smoke (Version 2, shard0): 20 correct outputs, task score 14.50/60.

## Layout
- `notebooks/f1/` — F1 notebook (`arc-f1-smoke.ipynb`) and its cells as files; `BUILD.md` describes assembly.
- `selector/`, `tests/` — candidate ingest, calibration, selection, reallocation, verifier.
- `fit_calib.py`, `verifier_coverage.py`, `run_eval.py`, `results.py`, `detector.py`, `check_real.py` — scoring scripts.
- `results/` — scoring reports (.md/.txt/.json) and plots.

## License
My own code and reports: **CC0 1.0 Universal** (see `LICENSE`).

## Third-party credits (not covered by CC0)
- Solver/loader cells (`02`–`05`) are adapted from public Kaggle notebooks in the ARChitects/NVARC lineage (public Kaggle notebooks are Apache 2.0); the ARChitects recipe (Franzen et al.) and NVARC model remain under their own licenses.
- Runtime dependencies: unsloth (Apache 2.0), transformers (Apache 2.0); kgmon selector idea credited per its public notebook.
- mikelou1's "ARC AGI2 LB33.89 Minimal Perfpatch" (our 31.81 LB entry was an unchanged fork) is **not included**: https://www.kaggle.com/code/mikelou1/arc-agi2-lb33-89-minimal-perfpatch
- Not included: model weights, candidate dumps, ARC data, credentials.
