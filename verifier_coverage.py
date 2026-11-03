import sys, time, json
from arc.data import load_dir
from selector.verifier import Verifier
for split in sys.argv[1:] or ["evaluation", "training"]:
    tasks = load_dir(f"data_repo/data/{split}"); t0 = time.time(); fit = 0; solved = 0.0; anyprog = 0
    for k, t in tasks.items():
        V = Verifier(t, time_budget=2.0); ok = 0
        if V.progs and V.progs[0][2] == 1.0: fit += 1
        for i, tt in enumerate(t["test"]):
            if tt.get("output") in V.predictions(i): ok += 1
        solved += ok / len(t["test"])
    n = len(tasks)
    print(f"{split}: tasks={n} exact-fit-program={fit} ({fit/n:.1%}) pass@2={solved/n:.2%} time={time.time()-t0:.0f}s")
