"""Summarize real MhcPre task durations, excluding correctness and warmup."""
import argparse
import csv
import json
import statistics
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("root")
p.add_argument("--warmup", type=int, default=20)
p.add_argument("--repeat", type=int, default=100)
a = p.parse_args()
files = sorted(Path(a.root).rglob("op_summary_*.csv"))
if not files:
    raise SystemExit("No op_summary CSV: no latency result")
for path in files:
    with path.open() as f:
        rows = [r for r in csv.DictReader(f) if r.get("OP Type") == "MhcPre"]
    group_size = 1 + a.warmup + a.repeat
    if not rows or len(rows) % group_size:
        raise SystemExit(f"Unexpected MhcPre count {len(rows)} in {path}")
    for start in range(0, len(rows), group_size):
        group = rows[start:start + group_size]
        keys = ("Input Shapes", "Input Data Types", "Block Num")
        assert all(all(r[k] == group[0][k] for k in keys) for r in group)
        data = sorted(float(r["Task Duration(us)"]) for r in group[1+a.warmup:])
        def percentile(q):
            pos = (len(data)-1)*q
            lo = int(pos)
            return data[lo] + (data[min(lo+1, len(data)-1)]-data[lo])*(pos-lo)
        print(json.dumps(dict(source=str(path), inputs=group[0]["Input Shapes"],
            dtype=group[0]["Input Data Types"], block_dim=group[0]["Block Num"],
            samples=len(data), warmup=a.warmup, min_us=min(data),
            mean_us=statistics.mean(data), p50_us=percentile(.5),
            p90_us=percentile(.9), p99_us=percentile(.99))), flush=True)
