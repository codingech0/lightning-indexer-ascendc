#!/usr/bin/env bash
# Test an already installed RUN package; does not build/install the operator.
set -euo pipefail
unset LD_PRELOAD
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
toolkit="${ASCEND_HOME_PATH:-${ASCEND_TOOLKIT_HOME:-}}"
if [[ -z "$toolkit" ]]; then
    echo 'Source your CANN set_env.sh or set ASCEND_HOME_PATH to the toolkit directory.' >&2
    exit 2
fi
deploy_dir="${LIGHTNING_DEPLOY_DIR:-$project_dir/deploy_run}"
api="$(find "$deploy_dir" -path '*/op_api/lib/libcust_opapi.so' -print -quit)"
if [[ -z "$api" ]]; then
    echo "No installed RUN package found under $deploy_dir" >&2
    exit 2
fi
vendor_dir="${api%/op_api/lib/libcust_opapi.so}"
export ASCEND_CUSTOM_OPP_PATH="$vendor_dir"
export LD_LIBRARY_PATH="$vendor_dir/op_api/lib:$toolkit/lib64:${LD_LIBRARY_PATH:-}"
mkdir -p "$project_dir/build_run"
echo "Custom package: $vendor_dir"
test_name="${LIGHTNING_TEST:-smoke}"
case "$test_name" in smoke|correctness) ;; *) echo 'LIGHTNING_TEST must be smoke or correctness' >&2; exit 2;; esac
g++ -std=c++17 -O0 -g "$project_dir/tests/npu_${test_name}.cpp" \
    -I"$toolkit/include" -I"$vendor_dir/op_api/include" \
    -L"$vendor_dir/op_api/lib" -L"$toolkit/lib64" \
    -Wl,-rpath,"$vendor_dir/op_api/lib" -Wl,-rpath,"$toolkit/lib64" \
    -lcust_opapi -lnnopbase -lascendcl -o "$project_dir/build_run/npu_${test_name}"
"$project_dir/build_run/npu_${test_name}" "${1:-0}"
