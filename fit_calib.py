"""Local: python fit_calib.py <cands_dir(s)...>  (downloaded /kaggle/working/cands from calibration runs)
Fits theta with 5-fold CV by task on ARC-AGI-2 public eval; prints A0 (kgmon), A0' (probmul_3), A1 (uniform PoE), A2 (calibrated)."""
import sys, numpy as np, json
from arc.data import load_dir
from arc.score import score
from selector.arch_ingest import load_pickles, to_rows, check_alignment, baseline_submission, getter_kgmon, getter_full_probmul_3
from selector.calibrate import features, fit, ece, save, probs
from selector.select import select_group
tasks = load_dir("data_repo/data/evaluation"); groups = {}
for d in sys.argv[1:]: groups.update(load_pickles(d))
print("groups", len(groups), "alignment issues", len(check_alignment(groups)))
tasks = {k: v for k, v in tasks.items() if any(key[0] == k for key in groups)}
rows = to_rows(groups); G = {}
for r in rows: G.setdefault((r["task_id"], r["test_idx"]), []).append(r)
views = sorted({v for r in rows for v in r["logprobs"]})
gt = {(k, i): t["output"] for k, t in tasks.items() for i, t in enumerate(tasks[k]["test"])}
def data_for(keys): return [(features(G[k], views), np.array([float(c["grid"] == gt[k]) for c in G[k]])) for k in keys if k in G]
def sub_with(theta):
    s = {}
    for tid, t in tasks.items():
        s[tid] = []
        for i, tt in enumerate(t["test"]):
            if (tid, i) in G: a1, a2, _, _ = select_group(G[(tid, i)], views, theta); s[tid].append({"attempt_1": a1, "attempt_2": a2})
            else: s[tid].append({"attempt_1": tt["input"], "attempt_2": tt["input"]})
    return s
oracle = np.mean([any(c["grid"] == gt[k] for c in G.get(k, [])) for k in gt])
print(f"oracle coverage {oracle:.2%}")
print("A0  kgmon      ", f"{score(baseline_submission(groups, tasks, getter_kgmon), tasks):.2%}")
print("A0' probmul_3  ", f"{score(baseline_submission(groups, tasks, getter_full_probmul_3), tasks):.2%}")
uni = np.r_[np.full(len(views), 1.0 / len(views)), 0.0]; print("A1  uniform PoE", f"{score(sub_with(uni), tasks):.2%}")
ids = sorted(tasks); folds = [ids[i::5] for i in range(5)]; cv = {}; confs = []; corr = []
for f in folds:
    tr = [k for k in G if k[0] not in f]; th = fit(data_for(tr), len(views))
    for tid in f: cv[tid] = th
    for k in G:
        if k[0] in f:
            p = probs(features(G[k], views), th); j = int(p.argmax()); confs.append(p[j]); corr.append(float(G[k][j]["grid"] == gt[k]))
s = {}
for tid in tasks: s.update({tid: sub_with(cv.get(tid, uni))[tid]})
print("A2  calibrated ", f"{score(s, tasks):.2%}", "ECE", f"{ece(confs, corr):.3f}")
th = fit(data_for(list(G)), len(views)); save(th, views, "theta.json"); print("saved theta.json (all-data fit)")
