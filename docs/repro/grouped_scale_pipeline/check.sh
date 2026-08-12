#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-all}"
OUT="${HERE}/outputs"
mkdir -p "${OUT}"
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
PTOAS_BIN="${PTOAS_BIN:-$(command -v ptoas)}"
export PTOAS_BIN
compile() {
  # ``production_group_vmi.pto`` is executable PTODSL, not textual MLIR.
  # benchmark.py lowers it with the pinned PTODSL compiler, then invokes PTOAS
  # on the emitted MLIR.  Calling PTOAS on the Python source made this check
  # fail with ``custom op 'from' is unknown``.
  env -u PYTHONPATH ACL_DEVICE_ID="${ACL_DEVICE_ID:-}" \
    "${PYTHON_BIN}" "${HERE}/benchmark.py" --compile-only
  grep -q 'pto.vcmax' "${OUT}/grouped_vmi.mlir"
  echo "PASS: stock PTODSL/PTOAS and stream-launchable direct CCE kernels compile"
}
run() {
  if [[ "${ACL_DEVICE_ID:-}" != "" ]]; then
    task-submit --device "${ACL_DEVICE_ID}" --run "ACL_DEVICE_ID=${ACL_DEVICE_ID} '${PYTHON_BIN}' '${HERE}/benchmark.py'" | tee "${OUT}/results.txt"
  else
    python3 "${HERE}/report.py" | tee "${OUT}/results.txt"
  fi
}
case "${MODE}" in compile) compile;; correctness|benchmark) run;; all) compile; run;; *) echo "usage: $0 [all|compile|correctness|benchmark]" >&2; exit 2;; esac
