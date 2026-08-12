#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; MODE="${1:-all}"; OUT="${HERE}/outputs"; mkdir -p "${OUT}"
CANN_ENV="${CANN_ENV:-/home/jzhuang/cann_installed/9.1.0-beta.3/cann-9.1.0-beta.3/set_env.sh}"
set +u; source "${CANN_ENV}"; set -u
PTOAS_BIN="${PTOAS_BIN:-$(conda run -n cann91_dev which ptoas | tail -1)}"
compile() {
  env -u PYTHONPATH "${PTOAS_BIN}" --pto-arch=a5 --pto-backend=vpto --emit-vpto "${HERE}/fixtures/requant_vmi.pto" -o "${OUT}/requant.vpto"
  grep -q 'pto.vcgmax' "${OUT}/requant.vpto"; grep -q 'pto.vcvt' "${OUT}/requant.vpto"; grep -q 'pto.vmul' "${OUT}/requant.vpto"
  env -u PYTHONPATH python3 "${HERE}/benchmark.py" --compile-only
  echo "PASS: direct CCE and VMI libraries compile"
}
run() {
  if [[ "${ACL_DEVICE_ID:-}" != "" ]]; then
    task-submit --device "${ACL_DEVICE_ID}" --run "source '${CANN_ENV}'; source /home/jzhuang/miniconda/bin/activate cann91_dev; PATH=\$CONDA_PREFIX/bin:\$PATH python '${HERE}/benchmark.py'" | tee "${OUT}/results.txt"
  else
    python3 "${HERE}/report.py" | tee "${OUT}/results.txt"
  fi
}
case "$MODE" in compile) compile;; correctness|benchmark) run;; all) compile; run;; *) echo "usage: $0 [all|compile|correctness|benchmark]" >&2; exit 2;; esac
