"""Compare saved profiler JSONL only; never execute or time a kernel on CPU."""
import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("baseline")
p.add_argument("candidate")
a = p.parse_args()
base = [json.loads(line) for line in Path(a.baseline).read_text().splitlines() if line]
cand = [json.loads(line) for line in Path(a.candidate).read_text().splitlines() if line]
assert len(base) == len(cand)
for b, c in zip(base, cand):
    assert (b["inputs"], b["dtype"], b["samples"], b["warmup"]) == (
        c["inputs"], c["dtype"], c["samples"], c["warmup"])
    print(json.dumps({"inputs": b["inputs"], "dtype": b["dtype"],
        "baseline_us": b["mean_us"], "candidate_us": c["mean_us"],
        "latency_reduction_pct": 100 * (1 - c["mean_us"] / b["mean_us"]),
        "candidate_min_us": c["min_us"], "candidate_p99_us": c["p99_us"]}))
