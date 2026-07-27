#!/usr/bin/env bash
# Try compiling the desired (1-lane) and target (1PT_B32) VMI forms; record results.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KERNEL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
FIXTURES="${KERNEL_ROOT}/fixtures"
OUT="${KERNEL_ROOT}/sim_outputs/check_desired"
mkdir -p "${OUT}"
LOG="${OUT}/compile_results.txt"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env.sh"

BISHENG="${BISHENG:-${ASCEND_HOME_PATH}/bin/bisheng}"
ASCEND="${ASCEND_HOME_PATH}"
NPU_ARCH="${ASCEND_NPU_ARCH:-dav-3510}"

: > "${LOG}"
echo "store_pad8 check_desired — $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "${LOG}"
echo "PTOAS_ROOT=${PTOAS_ROOT}" | tee -a "${LOG}"
echo | tee -a "${LOG}"

# --- desired_vmi.py: mask-1 store after vcadd (should compile via ptodsl) ---
echo "=== desired_vmi.py (ptodsl compile) ===" | tee -a "${LOG}"
set +e
python3 "${FIXTURES}/desired_vmi.py" > "${OUT}/desired_vmi.mlir" 2> "${OUT}/desired_vmi.err"
rc=$?
set -e
if [ "${rc}" -eq 0 ]; then
  echo "PASS: desired_vmi.py compiled (MLIR written to sim_outputs/check_desired/desired_vmi.mlir)" | tee -a "${LOG}"
else
  echo "FAIL: desired_vmi.py compile exit ${rc}" | tee -a "${LOG}"
  cat "${OUT}/desired_vmi.err" | tee -a "${LOG}"
fi
echo | tee -a "${LOG}"

# --- target_mi.pto: ptoas emit-vpto-llvm-ir + bisheng -c -x ir (expected to crash) ---
echo "=== target_mi.pto (ptoas -> LLVM IR -> bisheng object) ===" | tee -a "${LOG}"
TARGET_MI="${FIXTURES}/target_mi.pto"
LLVM_IR="${OUT}/target_mi.ll"
OBJ="${OUT}/target_mi.o"

set +e
if command -v ptoas >/dev/null 2>&1; then
  ptoas --cann-output-version=9.0.0 --pto-arch=a5 --pto-backend=vpto \
    --emit-vpto-llvm-ir "${TARGET_MI}" -o "${LLVM_IR}" > "${OUT}/target_mi_ptoas.log" 2>&1
  pto_rc=$?
else
  echo "SKIP: ptoas not on PATH" | tee -a "${LOG}"
  pto_rc=127
fi
set -e

if [ "${pto_rc}" -eq 0 ]; then
  echo "PASS: ptoas --emit-vpto-llvm-ir target_mi.pto" | tee -a "${LOG}"
  set +e
  "${BISHENG}" -O2 -fPIC -std=c++17 --npu-arch="${NPU_ARCH}" -c -x ir \
    "${LLVM_IR}" -o "${OBJ}" > "${OUT}/target_mi_bisheng.log" 2>&1
  b_rc=$?
  set -e
  if [ "${b_rc}" -eq 0 ] && [ -s "${OBJ}" ]; then
    echo "PASS: bisheng object compile (unexpected — gap may be closed)" | tee -a "${LOG}"
  else
    echo "FAIL: bisheng object compile exit ${b_rc} (expected on vmi-v0.1.3)" | tee -a "${LOG}"
    tail -20 "${OUT}/target_mi_bisheng.log" | tee -a "${LOG}"
  fi
else
  echo "FAIL: ptoas emit-vpto-llvm-ir exit ${pto_rc}" | tee -a "${LOG}"
  tail -20 "${OUT}/target_mi_ptoas.log" | tee -a "${LOG}"
fi
echo | tee -a "${LOG}"

# --- reference_asc_cce.asc: hand-written CCE ONEPT baseline (should compile) ---
echo "=== reference_asc_cce.asc (bisheng --cce-aicore-only) ===" | tee -a "${LOG}"
REF="${FIXTURES}/reference_asc_cce.asc"
REF_OBJ="${OUT}/reference_asc_cce.o"
set +e
"${BISHENG}" -O2 -fPIC -std=c++17 --npu-arch="${NPU_ARCH}" --cce-aicore-only -c \
  "${REF}" -o "${REF_OBJ}" \
  -I"${ASCEND}/include" \
  -I"${ASCEND}/compiler/tikcpp/tikcfw" \
  -I"${ASCEND}/compiler/tikcpp/tikcfw/impl" \
  -I"${ASCEND}/compiler/tikcpp/tikcfw/interface" \
  > "${OUT}/reference_asc_cce.log" 2>&1
ref_rc=$?
set -e
if [ "${ref_rc}" -eq 0 ] && [ -s "${REF_OBJ}" ]; then
  echo "PASS: reference_asc_cce.asc -> non-empty .o (CCE ONEPT path is legal)" | tee -a "${LOG}"
else
  echo "FAIL: reference_asc_cce.asc compile exit ${ref_rc}" | tee -a "${LOG}"
  tail -20 "${OUT}/reference_asc_cce.log" | tee -a "${LOG}"
fi

echo | tee -a "${LOG}"
echo "Full log: ${LOG}" | tee -a "${LOG}"
