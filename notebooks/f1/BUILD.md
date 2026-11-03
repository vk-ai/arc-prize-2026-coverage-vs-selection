# F1 notebook build — paste order (DRAFT; NOT RUN on Kaggle, NOT SUBMITTED)

Plan: full baseline pass → views4 pass on least-confident tasks until 20 min before deadline → pool → attempt_1 kgmon, attempt_2 probmul3 → `/kaggle/working/submission.json`.
Start from a copy of the live baseline notebook (`failed-in-aimo / arc-exp-views4-shard0-v2` lineage). Each `NN_*.py` file = one code cell; files starting with `%%writefile` must keep that first line.

## Sources (picked on the box)
- True baseline solver = `tmp0/arc_solver.py` (== `tmp1/arc_solver.py`). Differs from ttt32 only at L327 (TTT augment n=16 vs 32) and from views4 at L362 (decode views n=2 vs 4) + batching block + unequal-length fallback.
- `arc_loader.py`, `arc_decoder.py` identical (md5) across tmp0 / ttt32 / views4. `starter.py` identical across all three.
- F1 edits (only diffs vs source): L300 `dir_outputs = os.environ.get("ARC_OUT_DIR", ...)`; after `if key is None: break` skip the puzzle if `< ARC_START_GUARD_S` (1200 s) to `end_time`. starter: queue from `ARC_TASK_ORDER`; **no default shard** (old starter silently ran only shard 0 on non-rerun commits).

## Paste order
| # | file | cell |
|---|---|---|
| 0 | `00_setup.py` | first code cell: PYTHONHASHSEED, `global_end_time` (kept if already defined), `TEST_CHALLENGES_PATH`, selector dir |
| 1 | `01_KEEP_live_install_cells.md` | keep live notebook's install / imports cells as-is |
| 2 | `02_arc_loader.py` | `%%writefile arc_loader.py` |
| 3 | `03_arc_decoder.py` | `%%writefile arc_decoder.py` |
| 4 | `04_arc_solver_base.py` | `%%writefile arc_solver_base.py` |
| 5 | `05_arc_solver_views4.py` | `%%writefile arc_solver_views4.py` |
| 6 | `06_starter.py` | `%%writefile starter.py` |
| 7 | `07_selector_arch_ingest.py` | writes `/kaggle/working/selector/arch_ingest.py` |
| 8 | `08_selector_calibrate.py` | writes `/kaggle/working/selector/calibrate.py` (no arc-selector dataset needed) |
| 9 | `09_select_km1_attempt2.py` | writes `/kaggle/working/select_km1_attempt2.py` |
| 10 | `10_f1_confidence.py` | writes `/kaggle/working/f1_confidence.py` |
| 11 | `11_run_passes.py` | runs pass A then pass B (replaces the old `!python starter.py --end-time ...` cell) |
| 12 | `12_submission.py` | builds submission.json (replaces old selection/ArcDecoder cell; delete old cell 9 realloc) |
Delete from the live notebook: its old `%%writefile arc_solver.py`, `starter.py`, the old run cell, the old submission cell. Cell 11 copies the chosen solver to `arc_solver.py` before each pass.

## Box-verified
- All cells compile. Cells 07–10 + 12 run end-to-end on box data (shard1 baseline+views4 dirs): 120 tasks written, shape guard OK, 24 correct outputs = expected km1+pm2 shard1 number. Confidence order file written.
- NOT verifiable on the box: cells 04–06/11 (need GPUs, model, Kaggle paths).

## Check against the live notebook (needs Kaggle UI)
1. Model input attached at `/kaggle/input/qwen3_4b_grids15_sft139/transformers/bfloat16/1` (hard-coded in solver L265) and the LoRA/other inputs the install cells use.
2. Whether the live first cell already sets `global_end_time` and with what margin (cell 00 keeps an existing one).
3. Competition data root `/kaggle/input/competitions/arc-prize-2026-arc-agi-2/` (as in starter.py).
4. Install cells: what the old run cell did besides `!python starter.py` (env vars, `cd`, cleanup) — replicate in cell 11 if any.
5. Old submission cell: confirm Kaggle expects `/kaggle/working/submission.json` (the box copies of the run dirs contain `submission.json`, consistent).
6. Accelerator 4×L4, internet off.

## Runtime / risk notes
- A non-rerun commit now processes **all 120 eval tasks** (~12h). For a cheaper smoke commit set `ARC_SHARD=0` in cell 00 (60 tasks) — only with Vinay's OK; no run has been launched.
- Pass B gets ≈5h50m (≈78% of tasks), extrapolated from 60-task shard timings; model reload between passes unmeasured.

## Update 2026-10-25 — built `arc-f1-smoke.ipynb` from live `arc-exp-views4-shard0-v2.ipynb` (NOT RUN, NOT SUBMITTED)
Resolved checks (from the live notebook):
- `global_end_time = time.time() + 12*3600 - 600` (10 min margin) in live cell 0 — kept as notebook cell 0; cell 00 keeps it.
- Model path `/kaggle/input/qwen3_4b_grids15_sft139/transformers/bfloat16/1` and competition root `/kaggle/input/competitions/arc-prize-2026-arc-agi-2/` are both in the live cells and match ours.
- Live run cell: `!UNSLOTH_DISABLE_STATISTICS=1 TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas OMP_NUM_THREADS=12 python starter.py --end-time {global_end_time}` — these env vars are now set in cell 11's subprocess env.
- Live env cell set ARC_SHARD=1, ARC_NSHARD=2, ARC_GEN_N=4, ARC_GEN_SEED=2, PYTHONHASHSEED=0. ARC_GEN_N / ARC_GEN_SEED / ARC_NSHARD are read by NO file (views4 is hard-coded n=4) → dropped. Live cell 4 == patch/lb/arc_solver_views4.py exactly (so "live" was the views4 solver; baseline comes from tmp0).
- Live cells 2/3/5 (loader/decoder/starter) == box copies. Live cell 9 copied /kaggle/inference_outputs → /kaggle/working/cands; F1 writes straight into /kaggle/working/cands_base, cands_v4. Live cell 10 (ArcDecoder kgmon + reload score) replaced by cell 12 (km1+pm2 + smoke score print).
- No realloc cell existed in the live notebook. Raw cell 6 (stray `ARC_SHARD="1"`) dropped.
- Smoke: `ARC_SHARD="0"`; cell 11 restricts pass-B order to the same shard (otherwise shard-1 tasks with 0 candidates would rank first).
