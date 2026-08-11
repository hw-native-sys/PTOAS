#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; MODE="${1:-all}"; OUT="${HERE}/outputs"; mkdir -p "${OUT}"
set +u; source /home/jzhuang/cann_installed/9.1.0-beta.3/cann/set_env.sh; set -u
PTOAS_BIN="${PTOAS_BIN:-$(conda run -n cann91_dev which ptoas | tail -1)}"
compile() {
  env -u PYTHONPATH "${PTOAS_BIN}" --pto-arch=a5 --pto-backend=vpto --emit-vpto "${HERE}/fixtures/requant_vmi.pto" -o "${OUT}/requant.vpto"
  grep -q 'pto.vcgmax' "${OUT}/requant.vpto"; grep -q 'pto.vcvt' "${OUT}/requant.vpto"; grep -q 'pto.vmul' "${OUT}/requant.vpto"
  env -u PYTHONPATH ACL_DEVICE_ID="${ACL_DEVICE_ID:-}" python3 "${HERE}/benchmark.py" --cce-only
  echo "PASS: direct CCE library compiles, launches, and passes its host golden"
}
run() {
  if [[ "${ACL_DEVICE_ID:-}" != "" ]]; then
    python3 "${HERE}/benchmark.py" | tee "${OUT}/results.txt"
  else
    python3 "${HERE}/report.py" | tee "${OUT}/results.txt"
  fi
}
case "$MODE" in compile) compile;; correctness|benchmark) run;; all) compile; run;; *) echo "usage: $0 [all|compile|correctness|benchmark]" >&2; exit 2;; esac
