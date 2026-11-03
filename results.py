"""Turnkey results: python results.py cands_shard0 cands_shard1 [--tasks data_repo/data/evaluation] [--out results/]
Produces results/ablation.md, results/metrics.json, results/reliability.png, theta.json, and paper/WRITEUP_FILLED.md
(every [RESULT: key] placeholder in paper/WRITEUP.md replaced from metrics.json; unknown keys left as-is)."""
import argparse, json, os, re, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from arc.data import load_dir
from arc.score import score
from selector.arch_ingest import load_pickles, to_rows, check_alignment, baseline_submission, getter_kgmon, getter_full_probmul_3
from selector.calibrate import features, fit, ece, probs, save
from selector.select import select_group
from selector.verifier import Verifier

def main(dirs, tdir, out, ver_budget=2.0, folds=5, exclude=()):
    os.makedirs(out, exist_ok=True)
    tasks = load_dir(tdir); groups = {}
    for d in dirs: groups.update(load_pickles(d))
    groups = {k: v for k, v in groups.items() if k[0] not in set(exclude)}
    tasks = {k: v for k, v in tasks.items() if any(key[0] == k for key in groups)}
    G = {}
    for r in to_rows(groups): G.setdefault((r["task_id"], r["test_idx"]), []).append(r)
    views = sorted({v for cs in G.values() for c in cs for v in c["logprobs"]})
    gt = {(k, i): tt["output"] for k, t in tasks.items() for i, tt in enumerate(t["test"])}
    # verifier scores + injected candidates
    vers = {}
    for tid, t in tasks.items():
        V = Verifier(t, ver_budget)
        for i in range(len(t["test"])):
            grp = G.setdefault((tid, i), [])
            for g in V.predictions(i):
                if all(c["grid"] != g for c in grp): grp.append({"grid": g, "logprobs": {}, "_injected": True})
            vers[(tid, i)] = [V.score(i, c["grid"]) for c in grp]
    def X(k, use_ver): return features(G[k], views, vers[k] if use_ver else None)
    def y(k): return np.array([float(c["grid"] == gt[k]) for c in G[k]])
    def sub_theta(thetas, use_ver):
        s = {}
        for tid, t in tasks.items():
            s[tid] = []
            for i, tt in enumerate(t["test"]):
                k = (tid, i)
                if G.get(k):
                    a1, a2, _, _ = select_group(G[k], views, thetas[tid], vers[k] if use_ver else None)
                    s[tid].append({"attempt_1": a1, "attempt_2": a2})
                else: s[tid].append({"attempt_1": tt["input"], "attempt_2": tt["input"]})
        return s
    ids = sorted(tasks); fl = [ids[i::folds] for i in range(folds)]
    def cv(use_ver):
        th = {}; cf = []; co = []
        for f in fl:
            fs = set(f); tr = [(X(k, use_ver), y(k)) for k in G if k[0] not in fs and G[k]]
            t = fit(tr, len(views), lr=0.5, iters=3000)
            if not use_ver: t[-1] = 0.0
            for tid in f: th[tid] = t
            for k in G:
                if k[0] in fs and G[k]:
                    p = probs(X(k, use_ver), t); j = int(p.argmax()); cf.append(float(p[j])); co.append(float(G[k][j]["grid"] == gt[k]))
        return th, cf, co
    llm_only = {k: [c for c in G[k] if not c.get("_injected")] for k in G}
    def _ora(src): return float(np.mean([np.mean([any(c["grid"] == gt[(t, i)] for c in src.get((t, i), [])) for i in range(len(tasks[t]["test"]))]) for t in tasks]))
    oracle_llm = _ora(llm_only)
    oracle_all = _ora(G)
    uni = np.r_[np.full(len(views), 1 / len(views)), 0.0]
    m = {"n_tasks": len(tasks), "n_test_outputs": len(gt), "n_views": len(views),
         "alignment_issues": len(check_alignment(groups)), "oracle_llm": oracle_llm, "oracle_all": oracle_all,
         "A0_pass2": score(baseline_submission(groups, tasks, getter_kgmon), tasks),
         "A0p_pass2": score(baseline_submission(groups, tasks, getter_full_probmul_3), tasks),
         "A1u_pass2": score(sub_theta({t: uni for t in tasks}, False), tasks)}
    th1, c1, o1 = cv(False); th2, c2, o2 = cv(True)
    m.update({"A1_pass2": score(sub_theta(th1, False), tasks), "A1_ece": ece(c1, o1),
              "A2_pass2": score(sub_theta(th2, True), tasks), "A2_ece": ece(c2, o2),
              "injected_hits": int(sum(1 for k in G for c in G[k] if c.get("_injected") and c["grid"] == gt[k]))})
    for a in ("A0", "A0p", "A1u", "A1", "A2"): m[f"{a}_gap"] = oracle_all - m[f"{a}_pass2"]
    full = fit([(X(k, True), y(k)) for k in G if G[k]], len(views), lr=0.5, iters=3000); save(full, views, "theta.json")
    m["theta"] = dict(zip(views + ["verifier"], map(float, full)))
    json.dump(m, open(f"{out}/metrics.json", "w"), indent=1)
    # reliability diagram
    fig, ax = plt.subplots(figsize=(4.5, 4.5)); edges = np.linspace(0, 1, 11)
    for cf, co, lab in ((c1, o1, "A1 calibrated"), (c2, o2, "A2 +verifier")):
        cf, co = np.array(cf), np.array(co); xs, ys = [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            s = (cf >= lo) & (cf <= hi) if lo == 0 else (cf > lo) & (cf <= hi)
            if s.any(): xs.append(cf[s].mean()); ys.append(co[s].mean())
        ax.plot(xs, ys, "o-", label=lab)
    ax.plot([0, 1], [0, 1], "k--", lw=1); ax.set_xlabel("top-1 confidence"); ax.set_ylabel("accuracy"); ax.legend(); ax.set_title("Reliability (5-fold CV)")
    fig.tight_layout(); fig.savefig(f"{out}/reliability.png", dpi=150)
    pct = lambda v: f"{100*v:.2f}%"
    rows = [("A0 kgmon (notebook default)", "A0"), ("A0' full_probmul_3", "A0p"), ("A1u uniform PoE", "A1u"), ("A1 calibrated per-view", "A1"), ("A2 + verifier", "A2")]
    md = ["| Variant | pass@2 | oracle gap | ECE |", "|---|---|---|---|"]
    for name, k in rows: md.append(f"| {name} | {pct(m[k+'_pass2'])} | {pct(m[k+'_gap'])} | {m.get(k+'_ece', float('nan')):.3f} |".replace("nan", "–"))
    md.append(f"\nOracle coverage: LLM candidates {pct(oracle_llm)}, +verifier {pct(oracle_all)}. Tasks {len(tasks)}, test outputs {len(gt)}.")
    open(f"{out}/ablation.md", "w").write("\n".join(md))
    # fill placeholders
    fill = {"A0 pass@2": pct(m["A0_pass2"]), "A0' pass@2": pct(m["A0p_pass2"]), "A1u pass@2": pct(m["A1u_pass2"]),
            "A1 pass@2": pct(m["A1_pass2"]), "A2 pass@2": pct(m["A2_pass2"]), "A1 ECE": f"{m['A1_ece']:.3f}", "A2 ECE": f"{m['A2_ece']:.3f}",
            "oracle coverage": pct(oracle_llm), "oracle coverage +verifier": pct(oracle_all), "A1 oracle gap": pct(m["A1_gap"]),
            "A2 oracle gap": pct(m["A2_gap"]), "A0 oracle gap": pct(m["A0_gap"]), "injected hits": str(m["injected_hits"]),
            "n tasks": str(len(tasks)), "ablation table": "\n" + "\n".join(md) + "\n", "verifier weight": f"{m['theta']['verifier']:.3f}"}
    if os.path.exists("paper/WRITEUP.md"):
        txt = open("paper/WRITEUP.md").read()
        txt = re.sub(r"\[RESULT: ([^\]]+)\]", lambda x: fill.get(x.group(1), x.group(0)), txt)
        open("paper/WRITEUP_FILLED_v1_selection.md", "w").write(txt)
        left = sorted(set(re.findall(r"\[RESULT: ([^\]]+)\]", txt))); m["unfilled"] = left
        print("unfilled placeholders:", left)
    print(open(f"{out}/ablation.md").read())
    return m

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("dirs", nargs="+")
    ap.add_argument("--tasks", default="data_repo/data/evaluation"); ap.add_argument("--out", default="results")
    ap.add_argument("--ver-budget", type=float, default=2.0)
    ap.add_argument("--exclude", default="4a21e3da,abc82100,b6f77b65,f560132c,faa9f03d", help="tasks whose Kaggle-2026 eval test count differs from GitHub ARC-AGI-2")
    a = ap.parse_args()
    main(a.dirs, a.tasks, a.out, a.ver_budget, exclude=[x for x in a.exclude.split(',') if x])
