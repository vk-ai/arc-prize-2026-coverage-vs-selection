import json, numpy as np, os, tempfile
from selector.calibrate import fit, features, probs, ece, log_loss
from selector.realloc import allocate, confidence
from selector.verifier import Verifier
from selector.select import run
from selector.schema import dump

def task_rot():
    a = [[1, 2], [3, 4]]; b = [[5, 0, 0], [6, 7, 0]]
    return {"train": [{"input": a, "output": np.rot90(a).tolist()}, {"input": b, "output": np.rot90(b).tolist()}],
            "test": [{"input": [[1, 0], [0, 2]], "output": np.rot90([[1, 0], [0, 2]]).tolist()}]}

def test_verifier_exact():
    t = task_rot(); V = Verifier(t)
    assert t["test"][0]["output"] in V.predictions(0)
    assert V.score(0, t["test"][0]["output"]) == 1.0
    assert V.score(0, [[9]]) == 0.0

def test_verifier_cmap():
    t = {"train": [{"input": [[1, 2]], "output": [[3, 4]]}, {"input": [[2, 1, 1]], "output": [[4, 3, 3]]}], "test": [{"input": [[1, 1, 2]]}]}
    assert [[3, 3, 4]] in Verifier(t).predictions(0)

def test_calibration_learns_good_view():
    rng = np.random.default_rng(0); data = []
    for _ in range(60):
        n = 5; y = np.zeros(n); y[rng.integers(n)] = 1
        good = np.where(y > 0, 0.0, -3.0) + rng.normal(0, .3, n); noise = rng.normal(0, 3, n)
        data.append((np.c_[good, noise, np.zeros(n)], y))
    th = fit(data, 2, iters=300)
    assert th[0] > th[1]
    assert log_loss(data, th) < log_loss(data, np.array([0.5, 0.5, 0.0]))

def test_ece():
    assert ece([1, 1, 0, 0], [1, 1, 0, 0]) == 0.0
    assert abs(ece([0.9] * 10, [1] * 5 + [0] * 5) - 0.4) < 1e-9

def test_realloc():
    out = allocate({"a": 0.9, "b": 0.2, "c": 0.4}, 100, tau=0.6)
    assert out["a"] == 0 and out["b"] > out["c"] > 0 and abs(sum(out.values()) - 100) < 1e-9
    assert allocate({"a": .1}, 100, cap=30)["a"] == 30
    assert np.allclose(confidence([0.2, 0.7, 0.1]), (0.7, 0.5))

def test_select_end_to_end():
    t = task_rot(); gt = t["test"][0]["output"]
    rows = [{"task_id": "t", "test_idx": 0, "grid": [[0]], "logprobs": {"v0": -1.0, "v1": -1.0}},
            {"task_id": "t", "test_idx": 0, "grid": gt, "logprobs": {"v0": -2.0, "v1": -1.5}}]
    p = os.path.join(tempfile.mkdtemp(), "c.jsonl"); dump(rows, p)
    sub, _ = run(p, {"t": t}); assert sub["t"][0]["attempt_2"] == gt  # attempt_2 keeps runner-up
    sub, _ = run(p, {"t": t}, use_verifier=True); assert sub["t"][0]["attempt_1"] == gt  # verifier flips top-1
