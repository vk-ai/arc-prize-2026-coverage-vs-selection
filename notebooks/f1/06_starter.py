import os
import sys
# Demote Kaggle utility that ships a broken torch/ so site-packages wins (F1 smoke v1 ImportError).
for _p in list(sys.path):
    if "pip_install_unsloth_flash_patch" in _p:
        try:
            sys.path.remove(_p)
            sys.path.append(_p)
        except ValueError:
            pass
import time
import json
import torch
import argparse
import torch.multiprocessing as mp


def local_worker(rank, queue, end_time):
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)

    torch.set_default_device("cpu")

    # Fix Unsloth patching issue
    if rank > 0:
        while not os.path.exists(f"/kaggle/worker{rank-1}"):
            time.sleep(5)
    
    from arc_solver import worker

    with open(f"/kaggle/worker{rank}", "w") as f:
        f.write("Ok")
    
    print(f"[Rank {rank}] start!")
    
    worker(rank, queue, end_time)
    
    print(f"[Rank {rank}] done!")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--end-time", type=float, default=0.0)
    args = parser.parse_args()

    rerun_mode = os.getenv("KAGGLE_IS_COMPETITION_RERUN")

    if rerun_mode:
        test_path = "/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json"
    else:
        test_path = "/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json"

    with open(test_path, "r") as f:
        data = json.load(f)

    queue = mp.Manager().Queue()

    order_file = os.environ.get("ARC_TASK_ORDER")  # F1: pass B queue, least-confident first
    if order_file and os.path.exists(order_file):
        keys = [k for k in json.load(open(order_file))["order"] if k in data]
    else:
        keys = sorted(data.keys())
        if not rerun_mode and os.environ.get("ARC_SHARD"):  # offline sharding only
            keys = keys[int(os.environ["ARC_SHARD"])::2]
    for key in keys:
        queue.put(key)
    for _ in range(4):
        queue.put(None)
    
    mp.spawn(local_worker, args=(queue, args.end_time), nprocs=4)
