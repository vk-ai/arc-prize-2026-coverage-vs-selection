%%writefile /kaggle/working/f1_confidence.py
# F1 confidence ranking (DRAFT). Reads pass-A pickles, writes task order for pass B (least confident first).
# conf(output) = share of all sampled guesses that hit the kgmon top-1 grid (0 if no candidates).
# conf(task)   = min over its test outputs. Ties broken by task id (deterministic).
import json, os, sys
from selector.arch_ingest import load_pickles, getter_kgmon
def task_conf(dirs, task_ids):
    G = {}
    for d in dirs:
        if os.path.isdir(d):
            for k, s in load_pickles(d).items():
                for h, x in s.items(): G.setdefault(k, {}).setdefault(h, {"grid": x["grid"], "guesses": []})["guesses"] += x["guesses"]
    conf = {t: 1.0 for t in task_ids}; seen = set()
    for (t, i), sols in G.items():
        if t not in conf: continue
        seen.add(t); S = list(sols.values()); n = sum(len(s["guesses"]) for s in S)
        top = max(S, key=lambda s: getter_kgmon(s["guesses"]))
        conf[t] = min(conf[t], len(top["guesses"]) / n if n else 0.0)
    for t in task_ids:
        if t not in seen: conf[t] = 0.0   # pass A produced nothing -> most in need
    return conf
def write_order(dirs, task_ids, path="/kaggle/working/f1_order.json"):
    c = task_conf(dirs, task_ids); order = sorted(task_ids, key=lambda t: (c[t], t))
    json.dump({"order": order, "conf": c}, open(path, "w")); return order, c
if __name__ == "__main__":
    ids = json.load(open(sys.argv[1])).keys(); print(write_order(sys.argv[2:], sorted(ids))[0][:10])
