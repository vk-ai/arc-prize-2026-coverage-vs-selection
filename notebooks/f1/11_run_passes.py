# 07 — F1 orchestration cell (DRAFT, paste as ONE cell after the %%writefile cells; replaces the single
# `!python starter.py --end-time ...` call). Requires global_end_time from cell 1 (start + 12h - 600 s).
import os, sys, json, time, subprocess, shutil
os.chdir("/kaggle/working")
sys.path.insert(0, "/kaggle/working")
TEST = TEST_CHALLENGES_PATH                         # same path variable the notebook already uses
DIR_A, DIR_B = "/kaggle/working/cands_base", "/kaggle/working/cands_v4"
STOP_B = global_end_time - 20 * 60                  # pass B must be done 20 min before deadline
def run(solver, out_dir, end_t, order=None):
    shutil.copy(solver, "/kaggle/working/arc_solver.py")            # starter.py imports `arc_solver.worker`
    env = dict(os.environ, ARC_OUT_DIR=out_dir, PYTHONHASHSEED="0", UNSLOTH_DISABLE_STATISTICS="1",
               TRITON_PTXAS_PATH="/usr/local/cuda/bin/ptxas", OMP_NUM_THREADS="12")   # as live run cell 8
    env.pop("ARC_TASK_ORDER", None)
    if order: env["ARC_TASK_ORDER"] = order
    for r in range(4):                              # starter's rank handshake files from a prior pass
        try: os.remove(f"/kaggle/worker{r}")
        except FileNotFoundError: pass
    t0 = time.time()
    subprocess.run([sys.executable, "/kaggle/working/starter.py", "--end-time", str(end_t)], env=env, check=False,
                   timeout=max(60, int(end_t - time.time() + 300)))
    print(solver, "took", round((time.time() - t0) / 60, 1), "min; left", round((global_end_time - time.time()) / 60, 1), "min")
# Pass A: full baseline (ARC_GEN_N=2 solver), allowed to run up to STOP_B as well
run("arc_solver_base.py", DIR_A, STOP_B)
# Pass B: views4 on least-confident tasks first, until STOP_B
if STOP_B - time.time() > 1500:                     # need >= one puzzle cap + margin
    from f1_confidence import write_order
    ids = sorted(json.load(open(TEST)).keys())
    if not os.getenv("KAGGLE_IS_COMPETITION_RERUN") and os.environ.get("ARC_SHARD"):
        ids = ids[int(os.environ["ARC_SHARD"])::2]   # smoke: pass B only on the same shard as pass A
    order, conf = write_order([DIR_A], ids, "/kaggle/working/f1_order.json")
    print("pass B order head:", order[:8])
    run("arc_solver_views4.py", DIR_B, STOP_B, "/kaggle/working/f1_order.json")
else:
    print("skip pass B: not enough time")
