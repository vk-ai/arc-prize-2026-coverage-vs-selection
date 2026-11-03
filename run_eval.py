import sys, json, time
from arc.data import load_dir
from arc.baseline import solve
from arc.score import score
from arc.submit import write_submission
split = sys.argv[1] if len(sys.argv) > 1 else "evaluation"
tasks = load_dir(f"data_repo/data/{split}")
t0 = time.time(); preds = {k: solve(t) for k, t in tasks.items()}
s = score(preds, tasks); write_submission(preds, f"submission_{split}.json")
print(f"{split}: {len(tasks)} tasks, score={s:.4f} ({s*100:.2f}%), {time.time()-t0:.1f}s")
