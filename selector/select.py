"""Select 2 attempts per test output and write submission.json.
Usage: python -m selector.select --cands cands.jsonl --tasks challenges.json [--theta theta.json] [--verifier] --out submission.json"""
import argparse, json, numpy as np
from .schema import load, views as all_views
from .calibrate import features, probs, load as load_theta
from .verifier import Verifier

def _shape(g): return (len(g), len(g[0]) if g else 0)

def select_group(group, views, theta, ver_scores=None):
    X = features(group, views, ver_scores); p = probs(X, theta); order = np.argsort(-p)
    a1 = group[order[0]]["grid"]
    # diverse attempt_2: best candidate != a1, prefer a different shape if its prob is within 50% of runner-up
    rest = [i for i in order[1:] if group[i]["grid"] != a1]
    a2 = a1
    if rest:
        a2i = rest[0]
        diff = [i for i in rest if _shape(group[i]["grid"]) != _shape(a1)]
        if diff and p[diff[0]] >= 0.5 * p[a2i]: a2i = diff[0]
        a2 = group[a2i]["grid"]
    return a1, a2, p, order

def run(cands_path, tasks, theta=None, views=None, use_verifier=False, ver_budget=2.0):
    groups = load(cands_path); views = views or all_views(groups)
    if theta is None: theta = np.r_[np.full(len(views), 1.0 / max(len(views), 1)), 1.0 if use_verifier else 0.0]
    vers = {}; sub = {}; meta = {}
    for tid, t in tasks.items():
        sub[tid] = []
        for ti, tt in enumerate(t["test"]):
            group = groups.get((tid, ti), [])
            ver = None
            if use_verifier:
                vers.setdefault(tid, Verifier(t, ver_budget)); V = vers[tid]
                for g in V.predictions(ti):  # verifier may inject candidates the LLM never produced
                    if all(c["grid"] != g for c in group): group.append({"grid": g, "logprobs": {}})
                ver = [V.score(ti, c["grid"]) for c in group]
            if not group:
                sub[tid].append({"attempt_1": tt["input"], "attempt_2": tt["input"]}); continue
            a1, a2, p, order = select_group(group, views, theta, ver)
            sub[tid].append({"attempt_1": a1, "attempt_2": a2}); meta[(tid, ti)] = float(p[order[0]])
    return sub, meta

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cands", required=True); ap.add_argument("--tasks", required=True)
    ap.add_argument("--theta"); ap.add_argument("--verifier", action="store_true"); ap.add_argument("--out", default="submission.json")
    a = ap.parse_args(); tasks = json.load(open(a.tasks))
    th, vw = load_theta(a.theta) if a.theta else (None, None)
    sub, _ = run(a.cands, tasks, th, vw, a.verifier)
    json.dump(sub, open(a.out, "w")); print("wrote", a.out, len(sub), "tasks")
