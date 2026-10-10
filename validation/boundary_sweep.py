"""NPU-only stress of resident/streaming transitions and token reuse."""
import gc
import json
import runpy
import sys
from pathlib import Path

completed = 0
for n, ds in ((4, (16, 288, 304, 320)), (6, (16, 80, 96, 112)), (8, (16, 32, 48, 64))):
    for d in ds:
        for tokens in (1, 81, 257):
            for dtype in ("fp16", "bf16"):
                for gamma in (0, 1):
                    args = ["check.py", "--shape", f"1,{tokens},{n},{d}", "--dtype", dtype,
                            "--gamma", str(gamma), "--seed", "123", "--warmup", "2", "--repeat", "8"]
                    print(json.dumps({"sweep_case": completed, "args": args}), flush=True)
                    sys.argv = args
                    runpy.run_path(str(Path(__file__).with_name("check.py")), run_name="__main__")
                    completed += 1
                    gc.collect()
print(json.dumps({"boundary_sweep": "PASS", "cases": completed}), flush=True)
