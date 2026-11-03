import bz2, pickle, os, tempfile, numpy as np
from selector.arch_ingest import load_pickles, to_rows, check_alignment, getter_kgmon, getter_full_probmul_3, baseline_submission

def _write(root, name, obj):
    with bz2.open(os.path.join(root, name), "wb") as f: pickle.dump(obj, f)

def test_ingest_and_baselines():
    r = tempfile.mkdtemp(); A = [[1, 2]]; B = [[3]]
    _write(r, "0934a4d8_0.pkl.bz2", [
        {"beam_score": 0.1, "score_aug": [0.2, 0.3, 0.1], "solution": np.array(A)},
        {"beam_score": 0.2, "score_aug": [0.2, 0.2, 0.2], "solution": A},
        {"beam_score": 0.05, "score_aug": [2.0, 2.0, 2.0], "solution": B}])
    g = load_pickles(r); key = ("0934a4d8", 0)
    assert key in g and len(g[key]) == 2 and not check_alignment(g)
    sols = {tuple(map(tuple, s["grid"])): s["guesses"] for s in g[key].values()}
    gA, gB = sols[((1, 2),)], sols[((3,),)]
    assert abs(getter_kgmon(gA) - (2 - np.mean([0.2, 0.2]))) < 1e-9
    assert getter_kgmon(gA) > getter_kgmon(gB) and getter_full_probmul_3(gA) > getter_full_probmul_3(gB)
    tasks = {"0934a4d8": {"test": [{"input": [[0]]}]}}
    s = baseline_submission(g, tasks); assert s["0934a4d8"][0]["attempt_1"] == A and s["0934a4d8"][0]["attempt_2"] == B
    rows = to_rows(g); ra = [x for x in rows if x["grid"] == A][0]
    assert set(ra["logprobs"]) == {"aug0", "aug1", "aug2", "beam", "count"} and ra["n_samples"] == 2

def test_alignment_flag():
    r = tempfile.mkdtemp()
    _write(r, "aa4ec2a5_0.pkl.bz2", [{"beam_score": .1, "score_aug": [1, 2], "solution": [[1]]},
                                     {"beam_score": .1, "score_aug": [1, 2, 3], "solution": [[2]]}])
    assert check_alignment(load_pickles(r))
