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
  "${ASCEND_HOME_PATH}/tools/bisheng_compiler/bin/bisheng" -xcce -O2 -fPIC -std=c++17 \
    --cce-aicore-arch=dav-c310-vec --cce-aicore-only -c "${HERE}/fixtures/reference_cce.cpp" \
    -o "${OUT}/reference_device.o" -I"${ASCEND_HOME_PATH}/include" \
    -I"${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw" -I"${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/impl" \
    -I"${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/interface"
  echo "PASS: fixed ABI compiles; typed pointer-table form is rejected by current PTODSL"
}
run() { if [[ "${ACL_DEVICE_ID:-}" != "" ]]; then python3 "${HERE}/benchmark.py" | tee "${OUT}/results.txt"; else python3 "${HERE}/report.py" | tee "${OUT}/results.txt"; fi; }
case "$MODE" in compile) compile;; correctness|benchmark) run;; all) compile; run;; *) echo "usage: $0 [all|compile|correctness|benchmark]" >&2; exit 2;; esac
