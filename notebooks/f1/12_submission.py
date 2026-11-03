# 07 — F1 submission cell (DRAFT). Pools pass A + pass B, attempt_1 = kgmon, attempt_2 = probmul3.
# ARC Prize Kaggle expects /kaggle/working/submission.json ({task: [{"attempt_1":grid,"attempt_2":grid}, ...]}),
# not a CSV. Falls back to pass-A-only kgmon top-2 if anything fails.
import os, sys, json, traceback
sys.path.insert(0, "/kaggle/working")
os.environ.setdefault("ARC_ATTEMPT2", "probmul3")
tasks = json.load(open(TEST_CHALLENGES_PATH))
try:
    from select_km1_attempt2 import build
    sub = build(tasks, ["/kaggle/working/cands_base", "/kaggle/working/cands_v4"])
except Exception:
    traceback.print_exc(); print("F1 SELECT FAILED -> kgmon top-2 on pass A")
    from selector.arch_ingest import load_pickles, baseline_submission
    sub = baseline_submission(load_pickles("/kaggle/working/cands_base"), tasks)
for tid, t in tasks.items():                         # shape guard: one entry per test input
    if len(sub.get(tid, [])) != len(t["test"]):
        sub[tid] = [{"attempt_1": tt["input"], "attempt_2": tt["input"]} for tt in t["test"]]
json.dump(sub, open("/kaggle/working/submission.json", "w")); print("wrote submission.json", len(sub))

# smoke only (non-rerun): score against eval solutions, as the live notebook's reload check did
if not os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
    sol = json.load(open("/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_solutions.json"))
    ids = sorted(tasks)[int(os.environ["ARC_SHARD"])::2] if os.environ.get("ARC_SHARD") else sorted(tasks)
    n = sum(1 for t in ids for i, a in enumerate(sub[t]) if sol[t][i] in (a["attempt_1"], a["attempt_2"]))
    sc = sum(sum(sol[t][i] in (a["attempt_1"], a["attempt_2"]) for i, a in enumerate(sub[t])) / len(sub[t]) for t in ids)
    print(f"*** F1 smoke: {n} correct outputs, task score {sc:.2f}/{len(ids)}")
