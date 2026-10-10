"""Run the strict checks in one remote NPU process; stop on first failure."""
import gc
import argparse
import json
import runpy
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--warmup", type=int, default=0)
parser.add_argument("--repeat", type=int, default=0)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()

cases = [
    ("small", "1,1,4,16"),
    ("medium", "2,17,6,64"),
    ("large", "2,65,8,1024"),
    ("streaming_tail", "1,97,8,256"),
    ("k_tail", "1,3,6,1008"),
    ("unaligned", "1,3,4,20"),
    ("boundary", "1,1,4,1"),
]
completed = 0
for name, shape in cases:
    for dtype in ("fp16", "bf16"):
        for gamma in (0, 1):
            sys.argv = ["check.py", "--shape", shape, "--dtype", dtype,
                        "--gamma", str(gamma), "--seed", str(args.seed),
                        "--repeat", str(args.repeat), "--warmup", str(args.warmup)]
            print(json.dumps({"case": name, "argv": sys.argv}), flush=True)
            runpy.run_path(str(Path(__file__).with_name("check.py")), run_name="__main__")
            completed += 1
            gc.collect()
print(json.dumps({"matrix": "PASS", "cases": completed}), flush=True)
