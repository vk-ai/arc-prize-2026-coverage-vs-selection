"""Candidate dump schema (JSONL, one candidate per line):
{"task_id": str, "test_idx": int, "grid": [[int]], "logprobs": {view_id: float}, "n_samples": int}
logprobs[v] = sum of token log-probs of the candidate under augmentation view v (de-augmented back to canonical).
Missing views are allowed (treated as the per-view minimum observed for that group)."""
import json
from collections import defaultdict
def load(path):
    groups = defaultdict(list)
    for line in open(path):
        if line.strip():
            r = json.loads(line); groups[(r["task_id"], int(r["test_idx"]))].append(r)
    return dict(groups)
def dump(rows, path):
    with open(path, "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
def views(groups):
    return sorted({v for cs in groups.values() for c in cs for v in c["logprobs"]})
