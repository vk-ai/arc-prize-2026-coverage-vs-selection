%%writefile /kaggle/working/select_km1_attempt2.py
# Combined-notebook selection cell (DRAFT, NOT SUBMITTED). Pools ALL candidate pickles from the baseline
# and views4 generation passes, then picks attempt1 = kgmon, attempt2 = calibCV (frozen) or probmul3.
import json, os, numpy as np
from selector.arch_ingest import load_pickles, getter_kgmon, getter_full_probmul_3, to_rows
from selector.calibrate import features, probs
SEL = os.environ.get("ARC_ATTEMPT2", "probmul3")          # "probmul3" (no params) or "calibcv"
V = theta = None
if SEL == "calibcv":                                      # theta only needed for the calibCV variant
    TH = json.load(open(os.environ.get("ARC_THETA", "calibcv_theta_frozen.json")))
    V, theta = TH["views"], np.array(TH["theta"])
def pooled(dirs):
    o = {}
    for d in dirs:
        if not os.path.isdir(d): continue
        for k, s in load_pickles(d).items():
            for h, x in s.items(): o.setdefault(k, {}).setdefault(h, {"grid": x["grid"], "guesses": []})["guesses"] += x["guesses"]
    return o
def safe(f, g):
    try: return f(g)
    except Exception: return -1e9
def pick(sols):
    sols = list(sols.values())
    if not sols: return []
    km = sorted(range(len(sols)), key=lambda i: -safe(getter_kgmon, sols[i]["guesses"]))
    if SEL == "calibcv":
        rows = to_rows({("x", 0): {str(i): s for i, s in enumerate(sols)}}, 8)   # order preserved
        sc = probs(features(rows, V), theta)
    else:
        sc = np.array([safe(getter_full_probmul_3, s["guesses"]) for s in sols])
    a2 = [i for i in np.argsort(-sc, kind="stable") if i != km[0]][:1]
    return [sols[i]["grid"] for i in [km[0]] + a2]
def build(tasks, dirs):
    P = pooled(dirs); sub = {}
    for tid, t in tasks.items():
        sub[tid] = []
        for ti, tt in enumerate(t["test"]):
            g = pick(P.get((tid, ti), {})) or [tt["input"]]
            while len(g) < 2: g.append(g[0])
            sub[tid].append({"attempt_1": g[0], "attempt_2": g[1]})
    return sub
