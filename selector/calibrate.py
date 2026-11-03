"""Fit per-view weights a_v (= w_v/T_v; w and T are not separately identifiable for a linear score)
and verifier weight w_ver by minimising softmax log-loss over candidates within each (task,test_idx) group.
s(c) = sum_v a_v * logp_v(c) + w_ver * ver(c). Plus ECE of the calibrated top-1 probability."""
import json, numpy as np

def features(group, views, ver=None):
    X = np.array([[c["logprobs"].get(v, np.nan) for v in views] for c in group], float)
    for j in range(X.shape[1]):  # missing view -> group min for that view
        col = X[:, j]; mn = np.nanmin(col) if np.isfinite(col).any() else -50.0
        col[np.isnan(col)] = mn; X[:, j] = col
    X = X - X.max(0, keepdims=True)  # per-view shift; softmax invariant, improves conditioning
    v = np.array([ver[i] if ver is not None else 0.0 for i in range(len(group))])[:, None]
    return np.hstack([X, v])

def _softmax(z): z = z - z.max(); e = np.exp(z); return e / e.sum()

def fit(data, n_views, l2=1e-3, lr=0.05, iters=500, init=None, ver_nonneg=True):
    """L-BFGS-B (scipy) with bounds: view weights >= 0, verifier weight >= 0 (an expert can only add evidence).
    Falls back to projected gradient descent if scipy is unavailable."""
    data = [(X, y) for X, y in data if y.sum() > 0 and len(y) > 1]
    x0 = np.array(init, float) if init is not None else np.r_[np.full(n_views, 1.0 / max(n_views, 1)), 0.0]
    try:
        from scipy.optimize import minimize
        def f(th):
            L = 0.5 * l2 * th @ th; g = l2 * th
            for X, y in data:
                z = X @ th; z = z - z.max(); e = np.exp(z); p = e / e.sum(); t = y / y.sum()
                L -= np.log(max(p[y > 0].sum(), 1e-300)) / len(data); g += X.T @ (p - t) / len(data)
            return L, g
        b = [(0, None)] * n_views + [((0, None) if ver_nonneg else (None, None))]
        return minimize(f, x0, jac=True, method="L-BFGS-B", bounds=b, options={"maxiter": 2000}).x
    except ImportError:
        return _fit_gd(data, n_views, l2, lr, iters, init)

def _fit_gd(data, n_views, l2=1e-3, lr=0.05, iters=500, init=None):
    """data: list of (X [n_cand, n_views+1], y [n_cand] 0/1). Groups with no positive are skipped."""
    data = [(X, y) for X, y in data if y.sum() > 0 and len(y) > 1]
    theta = np.array(init, float) if init is not None else np.r_[np.full(n_views, 1.0 / max(n_views, 1)), 0.0]
    for _ in range(iters):
        g = l2 * theta
        for X, y in data:
            p = _softmax(X @ theta); t = y / y.sum(); g += X.T @ (p - t) / len(data)
        theta -= lr * g
        theta[:n_views] = np.maximum(theta[:n_views], 0.0)  # keep view weights non-negative
    return theta

def log_loss(data, theta):
    L = [-np.log(max(_softmax(X @ theta)[y > 0].sum(), 1e-12)) for X, y in data if y.sum() > 0]
    return float(np.mean(L)) if L else float("nan")

def ece(confs, correct, bins=10):
    confs, correct = np.asarray(confs, float), np.asarray(correct, float)
    edges = np.linspace(0, 1, bins + 1); e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (confs > lo) & (confs <= hi) if lo > 0 else (confs >= lo) & (confs <= hi)
        if m.any(): e += m.mean() * abs(confs[m].mean() - correct[m].mean())
    return float(e)

def probs(X, theta): return _softmax(X @ theta)

def save(theta, views, path):
    json.dump({"views": views, "theta": list(map(float, theta))}, open(path, "w"))
def load(path):
    d = json.load(open(path)); return np.array(d["theta"]), d["views"]
