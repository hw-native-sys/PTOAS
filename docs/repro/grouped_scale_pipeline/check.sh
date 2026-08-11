#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${HERE}/../../.." && pwd)"
MODE="${1:-all}"
OUT="${HERE}/outputs"
mkdir -p "${OUT}"
set +u; source /home/jzhuang/cann_installed/9.1.0-beta.3/cann/set_env.sh; set -u
PTOAS_BIN="${PTOAS_BIN:-$(conda run -n cann91_dev which ptoas | tail -1)}"
compile() {
  env -u PYTHONPATH "${PTOAS_BIN}" --pto-arch=a5 --pto-backend=vpto --emit-vpto \
    "${HERE}/fixtures/production_group_vmi.pto" -o "${OUT}/grouped_scale.vpto"
  grep -q 'pto.vcgmax' "${OUT}/grouped_scale.vpto"
  grep -Eq 'pto.vselr|pto.vdup|pto.vlds.*BRC' "${OUT}/grouped_scale.vpto"
  env -u PYTHONPATH "${PTOAS_BIN}" --pto-arch=a5 --pto-backend=vpto --pto-level=level3 \
    "${HERE}/fixtures/production_group_vmi.pto" -o "${OUT}/grouped_scale_vmi.o"
  env -u PYTHONPATH ACL_DEVICE_ID="${ACL_DEVICE_ID:-}" PATH="${CONDA_PREFIX:-/home/jzhuang/.conda/envs/cann91_dev}/bin:$PATH" python3 "${HERE}/benchmark.py" --compile-only
  echo "PASS: stock PTOAS and stream-launchable direct CCE kernels compile"
}
run() {
  if [[ "${ACL_DEVICE_ID:-}" != "" ]]; then
    task-submit --device "${ACL_DEVICE_ID}" --run "source /home/jzhuang/cann_installed/9.1.0-beta.3/cann/set_env.sh; source /home/jzhuang/miniconda/bin/activate cann91_dev; PATH=\$CONDA_PREFIX/bin:\$PATH python '${HERE}/benchmark.py'" | tee "${OUT}/results.txt"
  else
    python3 "${HERE}/report.py" | tee "${OUT}/results.txt"
  fi
}
case "${MODE}" in compile) compile;; correctness|benchmark) run;; all) compile; run;; *) echo "usage: $0 [all|compile|correctness|benchmark]" >&2; exit 2;; esac
