import numpy as np, sys, json
from arc.data import load_dir
from selector.arch_ingest import load_pickles, getter_kgmon, getter_full_probmul_3
tasks = load_dir("data_repo/data/evaluation")
DIFF = {'4a21e3da', 'abc82100', 'b6f77b65', 'f560132c', 'faa9f03d'}
def acc(groups, getter, mapping):
    ntp = {}
    for (t, i) in groups: ntp[t] = max(ntp.get(t, 0), i + 1)
    s = 0.0
    for (t, i), sols in groups.items():
        if t in DIFF:
            if mapping is None or i != mapping: continue
            gt = tasks[t]["test"][0]["output"]
        else: gt = tasks[t]["test"][i]["output"]
        top = sorted(sols.values(), key=lambda x: -getter(x["guesses"]))[:2]
        if any(x["grid"] == gt for x in top): s += 1 / ntp[t]
    return s, len(ntp)
for d in sys.argv[1:]:
    g = load_pickles(d)
    print(d, "groups", len(g), "tasks", len({k[0] for k in g}), "diff-tasks present", sorted({k[0] for k in g} & DIFF))
    for m in (None, 0, 1):
        print(f"  gh test0 -> kaggle idx {m}: kgmon %.2f/%d" % acc(g, getter_kgmon, m), " probmul_3 %.2f/%d" % acc(g, getter_full_probmul_3, m))
    L = {}; same = []
    for v in g.values():
        for s in v.values():
            for c in s["guesses"]: L[len(c["score_aug"])] = L.get(len(c["score_aug"]), 0) + 1
            same.append(all(np.allclose(s["guesses"][0]["score_aug"], c["score_aug"]) for c in s["guesses"]))
    print("  score_aug lengths", L, "identical score_aug across guesses of same grid:", round(float(np.mean(same)), 3))
