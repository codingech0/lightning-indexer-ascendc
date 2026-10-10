"""NPU correctness around paired-token and 40-core scheduling boundaries."""
import gc
import json
import runpy
import sys
from pathlib import Path

count = 0
for n, d in ((4, 16), (6, 16), (8, 32)):
    for tokens in (2, 3, 31, 32, 33, 39, 40, 41, 79, 80):
        for dtype in ("fp16", "bf16"):
            for gamma in (0, 1):
                sys.argv = ["check.py", "--shape", f"1,{tokens},{n},{d}",
                            "--dtype", dtype, "--gamma", str(gamma),
                            "--seed", "921", "--warmup", "2", "--repeat", "8"]
                runpy.run_path(str(Path(__file__).with_name("check.py")), run_name="__main__")
                count += 1
                gc.collect()
print(json.dumps({"tiling_edges": "PASS", "cases": count}), flush=True)
