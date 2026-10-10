#!/usr/bin/env bash
# Run inside a Linux CANN environment after sourcing its set_env.sh.
# Builds only; does not install packages or submit to the contest.
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if ! command -v cmake >/dev/null 2>&1; then
    echo 'Missing cmake: use a Linux CANN development environment.' >&2
    exit 2
fi
cmake -S "$project_dir/code" -B "$project_dir/build"
cmake --build "$project_dir/build" --parallel 2
