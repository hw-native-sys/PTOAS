#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; MODE="${1:-all}"; OUT="${HERE}/outputs"; mkdir -p "${OUT}"
set +u; source /home/jzhuang/cann_installed/9.1.0-beta.3/cann/set_env.sh; set -u
compile() {
  PYTHONPATH="${HERE}/fixtures" conda run -n cann91_dev python "${HERE}/fixtures/fixed_arguments.py" --emit-mlir > "${OUT}/fixed.mlir"
  grep -q '!pto.ptr<f32, gm>' "${OUT}/fixed.mlir"
  grep -q 'pto.mte_gm_ub' "${OUT}/fixed.mlir"
  grep -q 'DESIRED API' "${HERE}/fixtures/desired_pointer_table.py"
  conda run -n cann91_dev python "${HERE}/fixtures/indirect_api_negative.py" | tee "${OUT}/negative_api.txt"
  ACL_DEVICE_ID="${ACL_DEVICE_ID:-}" python3 "${HERE}/benchmark.py" --compile-only
  echo "PASS: stream-launchable CCE and fixed-argument VMI libraries built; typed pointer-table form is rejected"
}
run() { if [[ "${ACL_DEVICE_ID:-}" != "" ]]; then python3 "${HERE}/benchmark.py" | tee "${OUT}/results.txt"; else python3 "${HERE}/report.py" | tee "${OUT}/results.txt"; fi; }
case "$MODE" in compile) compile;; correctness|benchmark) run;; all) compile; run;; *) echo "usage: $0 [all|compile|correctness|benchmark]" >&2; exit 2;; esac
