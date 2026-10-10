"""Masked 32-element reduction tails, including reuse and generic n=2 coverage."""
import gc
import json
import runpy
import sys
from pathlib import Path

completed = 0
for n, d in ((6, 48), (2, 16)):
    for tokens in (1, 41, 129):
        for dtype in ("fp16", "bf16"):
            for gamma in (0, 1):
                sys.argv = ["check.py", "--shape", f"1,{tokens},{n},{d}", "--dtype", dtype,
                            "--gamma", str(gamma), "--seed", "81", "--warmup", "20", "--repeat", "100"]
                runpy.run_path(str(Path(__file__).with_name("check.py")), run_name="__main__")
                completed += 1
                gc.collect()
print(json.dumps({"tail_cases": "PASS", "cases": completed}), flush=True)
