"""Actual NPU checks of packed sigmoid saturation; unchanged reference/tolerances."""
import gc
import json
import runpy
import sys
from pathlib import Path

count = 0
for shape in ("1,1,4,16", "1,33,6,64", "1,3,8,256", "1,3,4,20", "1,3,2,16"):
    for dtype in ("fp16", "bf16"):
        for gamma in (0, 1):
            sys.argv = ["check.py", "--shape", shape, "--dtype", dtype,
                        "--gamma", str(gamma), "--mode", "sigmoid_extreme",
                        "--seed", "929", "--warmup", "2", "--repeat", "8"]
            runpy.run_path(str(Path(__file__).with_name("check.py")), run_name="__main__")
            count += 1
            gc.collect()
print(json.dumps({"activation_cases": "PASS", "cases": count}), flush=True)
