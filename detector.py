"""None-correct detector + reallocation curve + pass-2 task list (offline, real dumps)."""
import json, random, numpy as np
from arc.data import load_dir
from selector.arch_ingest import load_pickles, to_rows, getter_kgmon
from selector.verifier import Verifier
EXC = set("4a21e3da,abc82100,b6f77b65,f560132c,faa9f03d".split(","))
groups = {}
for d in ["cands_shard0", "cands_shard1"]: groups.update(load_pickles(d))
groups = {k: v for k, v in groups.items() if k[0] not in EXC}
tasks = {k: v for k, v in load_dir("data_repo/data/evaluation").items() if any(g[0] == k for g in groups)}
G = {}
for r in to_rows(groups): G.setdefault((r["task_id"], r["test_idx"]), []).append(r)
vfit = {t: (Verifier(tasks[t], 2.0).progs[:1] or [(0, 0, 0.0)])[0][2] for t in tasks}
NAMES = ["log_n_cand", "log_n_samples", "top1_share", "best_beam", "top1_aug_mean", "top1_aug_min", "top1_aug_spread",
         "margin12", "agree_top2", "log_out_cells", "n_train", "n_test", "verifier_trainfit", "missing"]
keys, F, none, top2ok = [], [], [], []
for t, task in tasks.items():
    for i, tt in enumerate(task["test"]):
        cs = G.get((t, i), []); gt = tt["output"]
        if not cs:
            f = [0] * 13 + [1]; nc = 1; ok = 0
        else:
            sc = [getter_kgmon([{"score_aug": a} for a in c["_score_aug"]]) for c in cs]
            o = np.argsort(sc)[::-1]; c1 = cs[o[0]]; a1 = np.array([np.mean(x) for x in zip(*c1["_score_aug"])])
            ns = sum(c["n_samples"] for c in cs)
            m = sc[o[0]] - sc[o[1]] if len(cs) > 1 else 5.0
            s2 = (c1["n_samples"] + (cs[o[1]]["n_samples"] if len(cs) > 1 else 0)) / ns
            f = [np.log(len(cs)), np.log(ns), c1["n_samples"] / ns, min(c1["_beam_scores"] or [9]), a1.mean(), a1.min(), a1.max() - a1.min(),
                 m, s2, np.log(np.size(np.array(c1["grid"]))), len(task["train"]), len(task["test"]), vfit[t], 0]
            nc = int(not any(c["grid"] == gt for c in cs)); ok = int(any(cs[j]["grid"] == gt for j in o[:2]))
        keys.append((t, i)); F.append(f); none.append(nc); top2ok.append(ok)
F = np.array(F, float); none = np.array(none); top2ok = np.array(top2ok)

def auroc(s, y):
    y = np.asarray(y); r = np.argsort(np.argsort(s)) + 1.0
    # average ranks for ties
    from collections import defaultdict
    d = defaultdict(list)
    for j, v in enumerate(s): d[v].append(j)
    for js in d.values(): r[js] = np.mean(r[js])
    P, N = y.sum(), len(y) - y.sum()
    return float((r[y == 1].sum() - P * (P + 1) / 2) / (P * N)) if P and N else float("nan")
def logreg(X, y, l2=1.0, it=3000, lr=0.1):
    mu, sd = X.mean(0), X.std(0) + 1e-9; Z = (X - mu) / sd; w = np.zeros(Z.shape[1]); b = 0.0
    for _ in range(it):
        p = 1 / (1 + np.exp(-(Z @ w + b))); w -= lr * (Z.T @ (p - y) / len(y) + l2 * w / len(y)); b -= lr * np.mean(p - y)
    return lambda X2: 1 / (1 + np.exp(-(((X2 - mu) / sd) @ w + b))), w
def cv(X, y, mask=None, seed=0, k=5):
    idx = np.arange(len(y)) if mask is None else np.where(mask)[0]
    tl = sorted({keys[j][0] for j in idx}); random.Random(seed).shuffle(tl); fold = {t: n % k for n, t in enumerate(tl)}
    p = np.full(len(y), np.nan)
    for f in range(k):
        tr = [j for j in idx if fold[keys[j][0]] != f]; te = [j for j in idx if fold[keys[j][0]] == f]
        pr, _ = logreg(X[tr], y[tr]); p[te] = pr(X[te])
    return p
out = {"n_outputs": len(y := none), "n_tasks": len(tasks), "n_uncovered": int(none.sum()), "n_covered": int((1 - none).sum())}
ps = [cv(F, none, seed=s) for s in range(5)]; p_none = np.mean(ps, 0)
out["auroc_none_cv"] = [auroc(p, none) for p in ps]; out["auroc_none_cv_mean"] = float(np.mean(out["auroc_none_cv"]))
out["auroc_single"] = {n: max(auroc(F[:, j], none), 1 - auroc(F[:, j], none)) for j, n in enumerate(NAMES) if F[:, j].std() > 0}
out["auroc_single_signed"] = {n: auroc(F[:, j], none) for j, n in enumerate(NAMES) if F[:, j].std() > 0}
_, w = logreg(F, none); out["weights_full_std"] = dict(zip(NAMES, map(float, w)))
# reallocation curve
o = np.argsort(-p_none); curve = []
rng = np.random.RandomState(0)
for K in [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
    n = int(round(K / 100 * len(none))); hit = none[o[:n]].sum() / none.sum()
    curve.append({"K%": K, "n_outputs": n, "frac_uncovered_hit": float(hit), "frac_covered_rerun": float((1 - none[o[:n]]).sum() / (1 - none).sum())})
out["realloc_curve"] = curve
# selection confidence over covered outputs: P(top-2 contains correct | covered)
cov = none == 0
pc = [cv(F, top2ok, mask=cov, seed=s) for s in range(5)]
out["n_covered_top2ok"] = int(top2ok[cov].sum())
out["auroc_select_cv"] = [auroc(p[cov], top2ok[cov]) for p in pc]
out["auroc_select_margin"] = auroc(F[cov, 7], top2ok[cov])
# task-level confidence: P(task not fully solved) ~ max over outputs
tconf = {}
for j, (t, i) in enumerate(keys): tconf[t] = max(tconf.get(t, 0), p_none[j])
MIN, GPUS, BUDGET_H, OVER_H = 20, 4, 6.0, 0.75   # ~20 min/puzzle/GPU (no per-puzzle timing in dumps), 45 min load/overhead
N = int((BUDGET_H - OVER_H) * 60 * GPUS / MIN)
rank = sorted(tconf, key=lambda t: -tconf[t]); sel = rank[:N]
ctrl = random.Random(2026).sample(sorted(tasks), N)
open("pass2_tasks.txt", "w").write(",".join(sel) + "\n"); open("pass2_random_control.txt", "w").write(",".join(ctrl) + "\n")
unc_t = {t for j, (t, i) in enumerate(keys) if none[j]}
out["pass2"] = {"n_tasks": N, "est_runtime_h": OVER_H + N * MIN / 60 / GPUS, "assumptions": f"{MIN} min/puzzle/GPU x {GPUS} GPUs + {OVER_H} h overhead",
                "uncovered_tasks_total": len(unc_t), "uncovered_tasks_in_list": len(unc_t & set(sel)), "uncovered_tasks_in_random": len(unc_t & set(ctrl)),
                "overlap_list_random": len(set(sel) & set(ctrl))}
json.dump(out, open("results/detector.json", "w"), indent=1)
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
xs = [0] + [c["K%"] for c in curve]; plt.plot(xs, [0] + [c["frac_uncovered_hit"] * 100 for c in curve], "o-", label="detector")
plt.plot([0, 100], [0, 100], "--", label="random"); plt.xlabel("% outputs re-run (least confident first)"); plt.ylabel("% uncovered outputs targeted"); plt.legend(); plt.savefig("results/realloc_curve.png", dpi=120)
print(json.dumps({k: v for k, v in out.items() if k != "weights_full_std"}, indent=1))
