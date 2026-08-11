#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; MODE="${1:-all}"; OUT="${HERE}/outputs"; mkdir -p "${OUT}"
set +u; source /home/jzhuang/cann_installed/9.1.0-beta.3/cann/set_env.sh; set -u
PTOAS_BIN="${PTOAS_BIN:-$(conda run -n cann91_dev which ptoas | tail -1)}"; BISHENG="${BISHENG:-${ASCEND_HOME_PATH}/tools/bisheng_compiler/bin/bisheng}"
compile() {
 env -u PYTHONPATH "${PTOAS_BIN}" --pto-arch=a5 --pto-backend=vpto --emit-vpto "${HERE}/fixtures/fused_roundtrip_vmi.pto" -o "${OUT}/fused.vpto"
 grep -q 'pto.vcgmax' "${OUT}/fused.vpto"; test "$(grep -c 'pto.vcvt' "${OUT}/fused.vpto")" -ge 3
 env -u PYTHONPATH "${PTOAS_BIN}" --pto-arch=a5 --pto-backend=vpto --pto-level=level3 "${HERE}/fixtures/fused_roundtrip_vmi.pto" -o "${OUT}/fused_vmi.o"
 "${BISHENG}" -xcce -O2 -fPIC -std=c++17 --cce-aicore-arch=dav-c310-vec --cce-aicore-only -c "${HERE}/fixtures/reference_cce.cpp" \
   -o "${OUT}/reference_device.o" -I"${ASCEND_HOME_PATH}/include" -I"${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw" \
   -I"${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/impl" -I"${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/interface"
 echo "PASS: full GM/UB VMI and direct CCE kernels compile"
}
run() { python3 "${HERE}/report.py" | tee "${OUT}/results.txt"; }
case "$MODE" in compile) compile;; correctness|benchmark) run;; all) compile; run;; *) echo "usage: $0 [all|compile|correctness|benchmark]" >&2; exit 2;; esac
