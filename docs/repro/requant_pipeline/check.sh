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
PTOAS_BIN="${PTOAS_BIN:-$(command -v ptoas)}"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
export PTOAS_BIN
compile() {
  env -u PYTHONPATH "${PYTHON_BIN}" "${HERE}/benchmark.py" --compile-only
  test -s "${OUT}/unpack_stage.o"
  test -s "${OUT}/requant_stage.o"
  test -s "${OUT}/rescale_cce.o"
  echo "PASS: full CCE and composed VMI libraries compile"
}
run() {
  if [[ "${ACL_DEVICE_ID:-}" != "" ]]; then
    task-submit --device "${ACL_DEVICE_ID}" --run "ACL_DEVICE_ID=${ACL_DEVICE_ID} '${PYTHON_BIN}' '${HERE}/benchmark.py'" | tee "${OUT}/results.txt"
  else
    python3 "${HERE}/report.py" | tee "${OUT}/results.txt"
  fi
}
case "$MODE" in compile) compile;; correctness|benchmark) run;; all) compile; run;; *) echo "usage: $0 [all|compile|correctness|benchmark]" >&2; exit 2;; esac
