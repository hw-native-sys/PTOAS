#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="/home/jzhuang/.conda/envs/cann91_dev/bin/python"
ROOT="$(cd "${HERE}/../../.." && pwd)"
MODE="${1:-all}"
OUT="${HERE}/outputs"
mkdir -p "${OUT}"
set +u; source /home/jzhuang/cann_installed/9.1.0-beta.3/cann/set_env.sh; set -u
PTOAS_BIN="${PTOAS_BIN:-$(conda run -n cann91_dev which ptoas | tail -1)}"
compile() {
  # ``production_group_vmi.pto`` is executable PTODSL, not textual MLIR.
  # benchmark.py lowers it with the pinned PTODSL compiler, then invokes PTOAS
  # on the emitted MLIR.  Calling PTOAS on the Python source made this check
  # fail with ``custom op 'from' is unknown``.
  env -u PYTHONPATH ACL_DEVICE_ID="${ACL_DEVICE_ID:-}" \
    PATH="${CONDA_PREFIX:-/home/jzhuang/.conda/envs/cann91_dev}/bin:$PATH" \
    "${PYTHON_BIN}" "${HERE}/benchmark.py" --compile-only
  grep -q 'pto.vcmax' "${OUT}/grouped_vmi.mlir"
  echo "PASS: stock PTODSL/PTOAS and stream-launchable direct CCE kernels compile"
}
run() {
  if [[ "${ACL_DEVICE_ID:-}" != "" ]]; then
    task-submit --device "${ACL_DEVICE_ID}" --run "source /home/jzhuang/cann_installed/9.1.0-beta.3/cann/set_env.sh; source /home/jzhuang/miniconda/bin/activate cann91_dev; PATH=\$CONDA_PREFIX/bin:\$PATH python '${HERE}/benchmark.py'" | tee "${OUT}/results.txt"
  else
    python3 "${HERE}/report.py" | tee "${OUT}/results.txt"
  fi
}
case "${MODE}" in compile) compile;; correctness|benchmark) run;; all) compile; run;; *) echo "usage: $0 [all|compile|correctness|benchmark]" >&2; exit 2;; esac
