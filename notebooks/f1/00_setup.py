# Cell 00 — F1 setup (paste as the FIRST code cell, or merge into the notebook's existing first cell).
# Keep the live notebook's own global_end_time if it already defines one at the very top; this only fills it in.
import os, time
os.environ["ARC_SHARD"] = "0"          # SMOKE: 60-task eval shard (ignored in competition rerun). Remove for full run.
os.environ["PYTHONHASHSEED"] = "0"
try: global_end_time
except NameError: global_end_time = time.time() + 12 * 3600 - 600
RERUN = bool(os.getenv("KAGGLE_IS_COMPETITION_RERUN"))
COMP = "/kaggle/input/competitions/arc-prize-2026-arc-agi-2"        # same root starter.py uses
TEST_CHALLENGES_PATH = f"{COMP}/arc-agi_test_challenges.json" if RERUN else f"{COMP}/arc-agi_evaluation_challenges.json"
os.makedirs("/kaggle/working/selector", exist_ok=True); open("/kaggle/working/selector/__init__.py", "a").close()
print("rerun:", RERUN, "| test:", TEST_CHALLENGES_PATH, "| hours left:", round((global_end_time - time.time()) / 3600, 2))
