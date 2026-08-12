#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; MODE="${1:-all}"; OUT="${HERE}/outputs"; mkdir -p "${OUT}"
CONDA_ENV="${CONDA_ENV:-cann91_dev}"
if [[ -n "${CANN_ENV:-}" ]]; then
  set +u
  source "${CANN_ENV}"
  set -u
elif [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
  echo "set CANN_ENV to the CANN 9.1.0-beta.3 set_env.sh (or source it first)" >&2
  exit 2
fi
if [[ "${CONDA_DEFAULT_ENV:-}" != "${CONDA_ENV}" ]] && command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV}"
fi
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
compile() {
  PYTHONPATH="${HERE}/fixtures" "${PYTHON_BIN}" "${HERE}/fixtures/fixed_arguments.py" --emit-mlir > "${OUT}/fixed.mlir"
  grep -q '!pto.ptr<f32, gm>' "${OUT}/fixed.mlir"
  grep -q 'pto.mte_gm_ub' "${OUT}/fixed.mlir"
  grep -q 'DESIRED API' "${HERE}/fixtures/desired_pointer_table.py"
  "${PYTHON_BIN}" "${HERE}/fixtures/indirect_api_negative.py" | tee "${OUT}/negative_api.txt"
  ACL_DEVICE_ID="${ACL_DEVICE_ID:-}" "${PYTHON_BIN}" "${HERE}/benchmark.py" --compile-only
  echo "PASS: stream-launchable direct-pointer CCE and stacked-buffer VMI libraries built; typed pointer-table form is rejected"
}
run() { if [[ "${ACL_DEVICE_ID:-}" != "" ]]; then task-submit --device "${ACL_DEVICE_ID}" --run "ACL_DEVICE_ID=${ACL_DEVICE_ID} '${PYTHON_BIN}' '${HERE}/benchmark.py'" | tee "${OUT}/results.txt"; else "${PYTHON_BIN}" "${HERE}/report.py" | tee "${OUT}/results.txt"; fi; }
case "$MODE" in compile) compile;; correctness|benchmark) run;; all) compile; run;; *) echo "usage: $0 [all|compile|correctness|benchmark]" >&2; exit 2;; esac
