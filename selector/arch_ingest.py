"""Ingest ARChitects-style decoder outputs (bz2 pickles in /kaggle/inference_outputs) into our schema.

Each pickle holds candidates for one subkey; each candidate is {beam_score, score_aug, solution}
(beam_score, score_aug entries are NLLs -> lower is better). We do not have the exact pickle container
from the notebook, so the loader accepts: a dict {subkey: [cand,...]}, a dict {subkey: cand}, a list of cands,
or a single cand. Subkey -> (task_id, test_idx) via '<task>_<idx>' (extra suffixes after the idx are ignored).

ALIGNMENT: score_aug is a positional list. Our per-view weights assume position j is the same view for every
candidate. check_alignment() flags groups whose score_aug lengths differ; it cannot detect a reordering, so
verify in arc_solver.calc_scores that the aug list is built from a fixed, seeded order (not resampled per candidate).
"""
import bz2, glob, hashlib, json, os, pickle, re
from collections import defaultdict
import numpy as np

def _cands(obj, default_key):
    if isinstance(obj, dict) and "solution" in obj: yield default_key, obj; return
    if isinstance(obj, dict):
        for k, v in obj.items():
            for kk, c in _cands(v, str(k)): yield kk, c
        return
    if isinstance(obj, (list, tuple)):
        for v in obj:
            for kk, c in _cands(v, default_key): yield kk, c

def parse_subkey(sk):
    m = re.match(r"^([0-9a-f]{8})(?:_(\d+))?", os.path.basename(str(sk)))
    return (m.group(1), int(m.group(2) or 0)) if m else (str(sk), 0)

def _grid(sol):
    a = np.asarray(sol); return a.astype(int).tolist()

def _h(grid): return hashlib.md5(json.dumps(grid).encode()).hexdigest()

def load_pickles(root):
    """-> {(task_id, test_idx): {hash: {"grid", "guesses": [cand,...]}}}"""
    out = defaultdict(dict)
    files = sorted(glob.glob(os.path.join(root, "**", "*"), recursive=True))
    for f in files:
        if os.path.isdir(f): continue
        try:
            with (bz2.open(f, "rb") if f.endswith(".bz2") or open(f, "rb").read(3) == b"BZh" else open(f, "rb")) as fh:
                obj = pickle.load(fh)
        except Exception:
            continue
        rel = os.path.relpath(f, root).split(os.sep)
        default = rel[0] if len(rel) > 1 else os.path.splitext(os.path.splitext(rel[0])[0])[0]
        for sk, c in _cands(obj, default):
            key = parse_subkey(sk); g = _grid(c["solution"]); h = _h(g)
            out[key].setdefault(h, {"grid": g, "guesses": []})["guesses"].append(c)
    return dict(out)

def check_alignment(groups):
    """Returns list of (key, sorted set of score_aug lengths) where lengths differ."""
    bad = []
    for key, sols in groups.items():
        L = {len(c.get("score_aug", [])) for s in sols.values() for c in s["guesses"]}
        if len(L) > 1: bad.append((key, sorted(L)))
    return bad

def to_rows(groups, n_views=None):
    """Our schema: logprobs[view] = -mean NLL at aug position j over guesses of this solution;
    plus 'beam' view = -min beam_score, and meta fields used by the A0 baselines."""
    rows = []
    for (tid, ti), sols in groups.items():
        for h, s in sols.items():
            G = s["guesses"]; aug = [np.asarray(c.get("score_aug", []), float) for c in G]
            L = n_views or max((len(a) for a in aug), default=0)
            lp = {}
            for j in range(L):
                vals = [a[j] for a in aug if len(a) > j]
                if vals: lp[f"aug{j}"] = -float(np.mean(vals))
            bs = [float(c["beam_score"]) for c in G if c.get("beam_score") is not None]
            if bs: lp["beam"] = -float(min(bs))
            lp["count"] = float(len(G))  # number of sampled hits (the kgmon "inf_score" term)
            rows.append({"task_id": tid, "test_idx": ti, "grid": s["grid"], "logprobs": lp,
                         "n_samples": len(G),
                         "_beam_scores": bs, "_score_aug": [a.tolist() for a in aug]})
    return rows

# --- A0 / A0' baselines: reproduce notebook getters (higher = better) ---
def getter_kgmon(guesses):
    """len(guesses) - mean over guesses of mean(score_aug)."""
    return len(guesses) - float(np.mean([np.mean(g["score_aug"]) for g in guesses]))

def getter_full_probmul_3(guesses):
    """sum over guesses of (3 - beam_score) + mean over guesses of sum(3 - score_aug)."""
    a = sum(3 - float(g["beam_score"]) for g in guesses)
    b = float(np.mean([np.sum(3 - np.asarray(g["score_aug"], float)) for g in guesses]))
    return a + b

def baseline_submission(groups, tasks, getter=getter_kgmon):
    sub = {}
    for tid, t in tasks.items():
        sub[tid] = []
        for ti, tt in enumerate(t["test"]):
            sols = list(groups.get((tid, ti), {}).values())
            sols.sort(key=lambda s: -getter(s["guesses"]))
            gs = [s["grid"] for s in sols[:2]] or [tt["input"]]
            while len(gs) < 2: gs.append(gs[0])
            sub[tid].append({"attempt_1": gs[0], "attempt_2": gs[1]})
    return sub
