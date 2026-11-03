"""icecuber-lite verifier: DSL program search that must reproduce train pairs.
verify(task, grid) -> score in [0,1]: 1.0 if some program fits ALL train pairs exactly and produces `grid` on the test input;
otherwise best partial (mean pixel agreement on train pairs of a program whose test output == grid), else 0."""
import time, numpy as np
from itertools import product

def _bg(g):
    v, c = np.unique(g, return_counts=True); return int(v[c.argmax()])
def _crop(g, mask):
    if not mask.any(): return g
    r, c = np.any(mask, 1), np.any(mask, 0); return g[np.ix_(r, c)]
def _objects(g):
    """4-connected components of non-background cells (any colour)."""
    bg = _bg(g); seen = np.zeros(g.shape, bool); objs = []
    H, W = g.shape
    for i in range(H):
        for j in range(W):
            if g[i, j] != bg and not seen[i, j]:
                st = [(i, j)]; seen[i, j] = True; cells = []
                while st:
                    a, b = st.pop(); cells.append((a, b))
                    for da, db in ((1,0),(-1,0),(0,1),(0,-1)):
                        x, y = a+da, b+db
                        if 0 <= x < H and 0 <= y < W and not seen[x, y] and g[x, y] != bg:
                            seen[x, y] = True; st.append((x, y))
                m = np.zeros(g.shape, bool); m[tuple(zip(*cells))] = True; objs.append(m)
    return objs

def _obj_select(key):
    def f(g):
        objs = _objects(g)
        if not objs: return g
        m = key(g, objs); return _crop(np.where(m, g, _bg(g)), m)
    return f

def primitives():
    P = {"id": lambda g: g}
    for k in range(1, 4): P[f"rot{k}"] = (lambda k: lambda g: np.rot90(g, k))(k)
    P["flipud"] = np.flipud; P["fliplr"] = np.fliplr; P["T"] = lambda g: g.T
    P["crop_fg"] = lambda g: _crop(g, g != _bg(g))
    for r, c in product(range(1, 4), range(1, 4)):
        if (r, c) != (1, 1): P[f"tile{r}x{c}"] = (lambda r, c: lambda g: np.tile(g, (r, c)))(r, c)
    for s in (2, 3): P[f"up{s}"] = (lambda s: lambda g: np.kron(g, np.ones((s, s), int)))(s)
    for s in (2, 3): P[f"down{s}"] = (lambda s: lambda g: g[::s, ::s])(s)
    P["mirror_h"] = lambda g: np.hstack([g, np.fliplr(g)]); P["mirror_v"] = lambda g: np.vstack([g, np.flipud(g)])
    P["obj_largest"] = _obj_select(lambda g, o: max(o, key=lambda m: m.sum()))
    P["obj_smallest"] = _obj_select(lambda g, o: min(o, key=lambda m: m.sum()))
    P["obj_rarest_color"] = _obj_select(lambda g, o: min(o, key=lambda m: (np.isin(g, g[m]).sum(), -m.sum())))
    P["half_left"] = lambda g: g[:, : g.shape[1] // 2]; P["half_top"] = lambda g: g[: g.shape[0] // 2]
    return P

def _fit_colormap(outs, targets):
    m = {}
    for x, y in zip(outs, targets):
        if x.shape != y.shape: return None
        for a, b in zip(x.ravel(), y.ravel()):
            if m.setdefault(int(a), int(b)) != int(b): return None
    return m

def search(task, time_budget=2.0, depth=2):
    """Return list of (name, fn, train_score) for programs; exact-fit programs have score 1.0."""
    t0 = time.time(); P = primitives()
    ins = [np.array(p["input"]) for p in task["train"]]; outs = [np.array(p["output"]) for p in task["train"]]
    progs = [(n,) for n in P] + ([(a, b) for a in P for b in P if a != "id" and b != "id"] if depth >= 2 else [])
    found = []
    for prog in progs:
        if time.time() - t0 > time_budget: break
        def f(g, prog=prog):
            for n in prog: g = P[n](g)
            return g
        try: xs = [np.asarray(f(i)) for i in ins]
        except Exception: continue
        if all(x.shape == y.shape and (x == y).all() for x, y in zip(xs, outs)):
            found.append(("+".join(prog), f, 1.0)); continue
        cm = _fit_colormap(xs, outs)
        if cm is not None:
            g2 = (lambda f, cm: lambda g: np.vectorize(lambda v: cm.get(int(v), int(v)))(f(g)))(f, cm)
            found.append(("+".join(prog) + "+cmap", g2, 1.0)); continue
        if all(x.shape == y.shape for x, y in zip(xs, outs)):
            found.append(("+".join(prog), f, float(np.mean([(x == y).mean() for x, y in zip(xs, outs)]))))
    found.sort(key=lambda t: -t[2])
    return found

class Verifier:
    def __init__(self, task, time_budget=2.0):
        self.task = task; self.progs = search(task, time_budget)
        self._cache = {}
    def predictions(self, test_idx, k=2, exact_only=True):
        g = np.array(self.task["test"][test_idx]["input"]); res = []
        for name, f, s in self.progs:
            if exact_only and s < 1.0: break
            try: o = np.asarray(f(g)).tolist()
            except Exception: continue
            if o not in res: res.append(o)
            if len(res) == k: break
        return res
    def score(self, test_idx, grid):
        key = (test_idx, str(grid))
        if key in self._cache: return self._cache[key]
        g = np.array(self.task["test"][test_idx]["input"]); best = 0.0
        for name, f, s in self.progs:
            if s <= best: break
            try:
                if np.asarray(f(g)).tolist() == grid: best = s
            except Exception: pass
        self._cache[key] = best; return best
