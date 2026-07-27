#!/usr/bin/env bash
# Compile/lower L=8 ui8→ui16 widen fixtures; print PASS/FAIL for each path.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPRO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
FIXTURES="${REPRO_ROOT}/fixtures"
OUT="${REPRO_ROOT}/outputs/check_l8_widen"
LOG="${OUT}/compile_results.txt"
VMI_PASSES=(-vmi-lower-unified-to-legacy -vmi-to-vpto)

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env.sh"

mkdir -p "${OUT}"
: > "${LOG}"

BISHENG="${BISHENG:-${ASCEND_HOME_PATH}/tools/bisheng_compiler/bin/bisheng}"
ASCEND="${ASCEND_HOME_PATH}"
NPU_ARCH="${ASCEND_NPU_ARCH:-dav-3510}"
PTOAS_FLAGS=(--cann-output-version=9.0.0 --pto-arch=a5 --pto-backend=vpto)

if ! command -v pto-test-opt >/dev/null 2>&1; then
  echo "ERROR: pto-test-opt not found; build PTOAS at tag vmi-v0.1.3 first." | tee -a "${LOG}"
  exit 1
fi
if ! command -v ptoas >/dev/null 2>&1; then
  echo "ERROR: ptoas not found; build PTOAS at tag vmi-v0.1.3 first." | tee -a "${LOG}"
  exit 1
fi

echo "l8_widen compile check — $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "${LOG}"
echo "PTOAS_ROOT=${PTOAS_ROOT}" | tee -a "${LOG}"
echo "pto-test-opt=$(command -v pto-test-opt)" | tee -a "${LOG}"
echo "ptoas=$(command -v ptoas)" | tee -a "${LOG}"
echo | tee -a "${LOG}"

# --- current_slow_vmi.pto: L=256 ui8→ui16 VMI (known-good widen shape) ---
echo "=== current_slow_vmi.pto (VMI L=256 ui8→ui16 lower) ===" | tee -a "${LOG}"
CURRENT_OUT="${OUT}/current_slow_vmi_vpto.pto"
set +e
pto-test-opt "${FIXTURES}/current_slow_vmi.pto" "${VMI_PASSES[@]}" -o "${CURRENT_OUT}" \
  > "${OUT}/current_slow_vmi.log" 2>&1
rc=$?
set -e
if [ "${rc}" -eq 0 ] && [ -s "${CURRENT_OUT}" ]; then
  echo "PASS: current_slow_vmi.pto lowers to VPTO (L=256 vcvt EVEN/ODD)" | tee -a "${LOG}"
else
  echo "FAIL: current_slow_vmi.pto lower exit ${rc}" | tee -a "${LOG}"
  tail -20 "${OUT}/current_slow_vmi.log" | tee -a "${LOG}"
fi
echo | tee -a "${LOG}"

# --- desired_vmi.pto: L=8 ui8→ui16 VMI (illegalizes to extui on vmi-v0.1.3) ---
echo "=== desired_vmi.pto (VMI L=8 ui8→ui16 lower) ===" | tee -a "${LOG}"
DESIRED_OUT="${OUT}/desired_vmi_vpto.pto"
set +e
pto-test-opt "${FIXTURES}/desired_vmi.pto" "${VMI_PASSES[@]}" -o "${DESIRED_OUT}" \
  > "${OUT}/desired_vmi.log" 2>&1
rc=$?
set -e
if [ "${rc}" -eq 0 ] && [ -s "${DESIRED_OUT}" ]; then
  echo "PASS: desired_vmi.pto lowers to VPTO (L=8 widen gap may be closed)" | tee -a "${LOG}"
else
  echo "FAIL: desired_vmi.pto lower exit ${rc} (residual pto.vmi.extui on vmi-v0.1.3)" | tee -a "${LOG}"
  rg -n "extui|illegal|VMI-RESIDUAL" "${OUT}/desired_vmi.log" | tail -5 | tee -a "${LOG}" || tail -15 "${OUT}/desired_vmi.log" | tee -a "${LOG}"
fi
echo | tee -a "${LOG}"

# --- target_mi.pto: small-L MI shape (ptoas lower + device object) ---
echo "=== target_mi.pto (MI PAT_VL8 vcvt EVEN + ui16 store) ===" | tee -a "${LOG}"
TARGET_VPTO="${OUT}/target_mi_vpto.pto"
TARGET_LL="${OUT}/target_mi.ll"
TARGET_OBJ="${OUT}/target_mi.o"

set +e
ptoas "${PTOAS_FLAGS[@]}" --emit-vpto "${FIXTURES}/target_mi.pto" -o "${TARGET_VPTO}" \
  > "${OUT}/target_mi_emit_vpto.log" 2>&1
vpto_rc=$?
set -e
if [ "${vpto_rc}" -ne 0 ]; then
  echo "FAIL: ptoas --emit-vpto target_mi.pto exit ${vpto_rc}" | tee -a "${LOG}"
  tail -20 "${OUT}/target_mi_emit_vpto.log" | tee -a "${LOG}"
else
  echo "PASS: ptoas --emit-vpto target_mi.pto" | tee -a "${LOG}"
  set +e
  ptoas "${PTOAS_FLAGS[@]}" --emit-vpto-llvm-ir "${FIXTURES}/target_mi.pto" -o "${TARGET_LL}" \
    > "${OUT}/target_mi_emit_llvm.log" 2>&1
  llvm_rc=$?
  set -e
  if [ "${llvm_rc}" -ne 0 ]; then
    echo "FAIL: ptoas --emit-vpto-llvm-ir target_mi.pto exit ${llvm_rc}" | tee -a "${LOG}"
    tail -20 "${OUT}/target_mi_emit_llvm.log" | tee -a "${LOG}"
  else
    echo "PASS: ptoas --emit-vpto-llvm-ir target_mi.pto" | tee -a "${LOG}"
    set +e
    ptoas "${PTOAS_FLAGS[@]}" "${FIXTURES}/target_mi.pto" -o "${TARGET_OBJ}" \
      > "${OUT}/target_mi_ptoas.log" 2>&1
    obj_rc=$?
    set -e
    if [ "${obj_rc}" -eq 0 ] && [ -s "${TARGET_OBJ}" ]; then
      echo "PASS: ptoas device object compile for target_mi.pto" | tee -a "${LOG}"
    else
      echo "FAIL: ptoas device object compile exit ${obj_rc}" | tee -a "${LOG}"
      tail -20 "${OUT}/target_mi_ptoas.log" | tee -a "${LOG}"
    fi
  fi
fi
echo | tee -a "${LOG}"

# --- reference_asc_cce.asc: hand-written CCE baseline ---
echo "=== reference_asc_cce.asc (bisheng --cce-aicore-only) ===" | tee -a "${LOG}"
REF_OBJ="${OUT}/reference_asc_cce.o"
set +e
"${BISHENG}" -O2 -fPIC -std=c++17 --npu-arch="${NPU_ARCH}" --cce-aicore-only -c \
  "${FIXTURES}/reference_asc_cce.asc" -o "${REF_OBJ}" \
  -I"${ASCEND}/include" \
  -I"${ASCEND}/compiler/tikcpp/tikcfw" \
  -I"${ASCEND}/compiler/tikcpp/tikcfw/impl" \
  -I"${ASCEND}/compiler/tikcpp/tikcfw/interface" \
  > "${OUT}/reference_asc_cce.log" 2>&1
ref_rc=$?
set -e
if [ "${ref_rc}" -eq 0 ] && [ -s "${REF_OBJ}" ]; then
  echo "PASS: reference_asc_cce.asc -> non-empty .o (CCE PAT_VL8 widen baseline)" | tee -a "${LOG}"
else
  echo "FAIL: reference_asc_cce.asc compile exit ${ref_rc}" | tee -a "${LOG}"
  tail -20 "${OUT}/reference_asc_cce.log" | tee -a "${LOG}"
fi

echo | tee -a "${LOG}"
echo "Full log: outputs/check_l8_widen/compile_results.txt" | tee -a "${LOG}"
