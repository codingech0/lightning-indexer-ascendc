"""Read NPU execution logs and summarize their recorded correctness evidence."""
import argparse
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("logs", nargs="+")
a = p.parse_args()
counts = {}
failures = []
errors = {}
for name in a.logs:
    for line in Path(name).read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(r, dict) or "correctness" not in r:
            continue
        stage = r.get("stage", "initial")
        counts[stage] = counts.get(stage, 0) + 1
        if r["correctness"] != "PASS":
            failures.append(r)
        for e in r["errors"]:
            out = errors.setdefault(e["output"], {})
            for metric in ("max_abs", "max_rel", "mean_abs"):
                out[metric] = max(out.get(metric, 0), e[metric])
print(json.dumps(dict(checks=counts, failures=failures, max_errors=errors), indent=2))
if not counts or failures:
    raise SystemExit(2)
