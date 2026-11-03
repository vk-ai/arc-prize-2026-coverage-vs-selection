"""Confidence-driven compute allocation.
confidence(task) = min over its test outputs of calibrated top-1 prob; margin = p1 - p2.
Extra budget B (seconds or TTT steps) goes to tasks with conf < tau, proportionally to (1 - conf),
capped per task, never exceeding the remaining wall-clock budget."""
import numpy as np
def confidence(p):
    p = np.sort(np.asarray(p))[::-1]; return float(p[0]), float(p[0] - (p[1] if len(p) > 1 else 0.0))
def allocate(conf, budget, tau=0.6, cap=None, margin=None, mtau=0.2):
    """conf: {task_id: top1 prob}; margin: optional {task_id: margin}. Returns {task_id: extra units}."""
    need = {k: 1.0 - c for k, c in conf.items() if c < tau or (margin and margin.get(k, 1) < mtau)}
    tot = sum(need.values())
    if tot <= 0 or budget <= 0: return {k: 0.0 for k in conf}
    out = {k: 0.0 for k in conf}
    for k, n in need.items():
        out[k] = budget * n / tot
        if cap is not None: out[k] = min(out[k], cap)
    return out
