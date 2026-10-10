"""Additional branch coverage and seeds, using unchanged strict tolerances."""
import gc
import json
import runpy
import sys
from pathlib import Path

completed = 0
for shape in ("1,41,4,32", "1,41,6,16", "1,3,8,16", "1,41,4,240", "1,41,4,256"):
    for seed in (7, 2026):
        for dtype in ("fp16", "bf16"):
            for gamma in (0, 1):
                sys.argv = ["check.py", "--shape", shape, "--dtype", dtype, "--gamma", str(gamma),
                            "--seed", str(seed), "--warmup", "0", "--repeat", "0"]
                runpy.run_path(str(Path(__file__).with_name("check.py")), run_name="__main__")
                completed += 1
                gc.collect()
for shape in ("1,1,4,16", "1,3,4,20", "1,1,8,1024"):
    for dtype in ("fp16", "bf16"):
        sys.argv = ["check.py", "--shape", shape, "--dtype", dtype, "--mode", "zero",
                    "--warmup", "0", "--repeat", "0"]
        runpy.run_path(str(Path(__file__).with_name("check.py")), run_name="__main__")
        completed += 1
print(json.dumps({"stress": "PASS", "cases": completed}), flush=True)
