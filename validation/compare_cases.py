"""Identical aligned FP16 workloads for archived original and candidates."""
import argparse
import gc
import json
import runpy
import sys
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--benchmark-failed", action="store_true")
p.add_argument("--suite", choices=["comparison", "resident"], default="comparison")
a = p.parse_args()
failed = 0
shapes = ("1,1,4,16", "2,17,6,64", "1,97,8,256", "2,65,8,1024")
if a.suite == "resident":
    shapes = ("1,1,4,16", "1,32,4,16", "1,128,4,16", "1,1,4,128",
              "1,32,4,128", "1,32,6,16", "1,32,6,64", "1,32,8,16",
              "1,32,8,32", "1,32,4,256")
for shape in shapes:
    for gamma in (0, 1):
        sys.argv = ["check.py", "--shape", shape, "--dtype", "fp16", "--gamma", str(gamma),
                    "--warmup", "20", "--repeat", "100"]
        if a.benchmark_failed:
            sys.argv.append("--benchmark-failed")
        try:
            runpy.run_path(str(Path(__file__).with_name("check.py")), run_name="__main__")
        except SystemExit as e:
            if e.code != 2 or not a.benchmark_failed:
                raise
            failed += 1
        gc.collect()
print(json.dumps({"comparison_cases": len(shapes)*2, "correctness_failures": failed,
                  "diagnostic_baseline": a.benchmark_failed}), flush=True)
if failed:
    sys.exit(2)
