#!/usr/bin/env bash
# Compile-check CCE stages=2 vs VMI stages=1/2 Persistent paths; record pass/fail.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KERNEL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
FIXTURES="${KERNEL_ROOT}/fixtures"
OUT="${KERNEL_ROOT}/sim_outputs/check_stages"
mkdir -p "${OUT}"
LOG="${OUT}/compile_results.txt"

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env.sh"

NPU_ARCH="${ASCEND_NPU_ARCH:-dav-3510}"

: > "${LOG}"
echo "pipelining check_stages — $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "${LOG}"
echo "PTOAS_ROOT=${PTOAS_ROOT}" | tee -a "${LOG}"
echo "BISHENG=${BISHENG}" | tee -a "${LOG}"
echo | tee -a "${LOG}"

# --- reference_asc_cce.asc: CCE stages=2 ping-pong (expect PASS) ---
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
  echo "PASS: reference_asc_cce.asc -> non-empty .o (CCE stages=2 ping-pong is legal)" | tee -a "${LOG}"
else
  echo "FAIL: reference_asc_cce.asc compile exit ${ref_rc}" | tee -a "${LOG}"
  tail -20 "${OUT}/reference_asc_cce.log" | tee -a "${LOG}"
fi
echo | tee -a "${LOG}"

# --- desired_vmi.py: ptodsl frontend + full lower (expect layout reject at lower) ---
echo "=== desired_vmi.py (ptodsl -> MLIR frontend) ===" | tee -a "${LOG}"
set +e
python3 "${FIXTURES}/desired_vmi.py" > "${OUT}/desired_vmi.mlir" 2> "${OUT}/desired_vmi_py.err"
py_rc=$?
set -e
if [ "${py_rc}" -eq 0 ]; then
  echo "PASS: desired_vmi.py emitted MLIR (frontend only)" | tee -a "${LOG}"
else
  echo "FAIL: desired_vmi.py compile exit ${py_rc}" | tee -a "${LOG}"
  tail -30 "${OUT}/desired_vmi_py.err" | tee -a "${LOG}"
fi
echo | tee -a "${LOG}"

echo "=== desired_vmi.py MLIR (ptoas -> LLVM IR, stages=2 block_k=1024) ===" | tee -a "${LOG}"
DESIRED_LL="${OUT}/desired_vmi.ll"
set +e
if command -v ptoas >/dev/null 2>&1 && [ -f "${OUT}/desired_vmi.mlir" ]; then
  ptoas --cann-output-version=9.0.0 --pto-arch=a5 --pto-backend=vpto \
    --emit-vpto-llvm-ir "${OUT}/desired_vmi.mlir" -o "${DESIRED_LL}" \
    > "${OUT}/desired_vmi_lower.log" 2>&1
  desired_lower_rc=$?
else
  desired_lower_rc=127
fi
set -e
if [ "${desired_lower_rc}" -eq 0 ]; then
  echo "PASS: desired_vmi full lower (unexpected — gap may be closed)" | tee -a "${LOG}"
else
  echo "FAIL: desired_vmi full lower exit ${desired_lower_rc} (layout reject on 1024-wide vmul)" | tee -a "${LOG}"
  tail -15 "${OUT}/desired_vmi_lower.log" | tee -a "${LOG}"
fi
echo | tee -a "${LOG}"

# --- desired_vmi.pto: ptoas --emit-vpto (expect layout / lower fail) ---
echo "=== desired_vmi.pto (ptoas --emit-vpto) ===" | tee -a "${LOG}"
DESIRED_PTO="${FIXTURES}/desired_vmi.pto"
DESIRED_VPTO="${OUT}/desired_vmi.vpto"
set +e
if command -v ptoas >/dev/null 2>&1; then
  ptoas --cann-output-version=9.0.0 --pto-arch=a5 --pto-backend=vpto \
    --emit-vpto "${DESIRED_PTO}" -o "${DESIRED_VPTO}" \
    > "${OUT}/desired_vmi_ptoas.log" 2>&1
  desired_pto_rc=$?
else
  echo "SKIP: ptoas not on PATH" | tee -a "${LOG}"
  desired_pto_rc=127
fi
set -e
if [ "${desired_pto_rc}" -eq 0 ] && [ -s "${DESIRED_VPTO}" ]; then
  echo "PASS: desired_vmi.pto --emit-vpto (unexpected — gap may be closed)" | tee -a "${LOG}"
else
  echo "FAIL: desired_vmi.pto --emit-vpto exit ${desired_pto_rc}" | tee -a "${LOG}"
  tail -15 "${OUT}/desired_vmi_ptoas.log" | tee -a "${LOG}"
fi
echo | tee -a "${LOG}"

# --- target_mi.pto: emit-vpto then LLVM IR -> bisheng (expect bisheng crash) ---
echo "=== target_mi.pto (ptoas --emit-vpto) ===" | tee -a "${LOG}"
TARGET_MI="${FIXTURES}/target_mi.pto"
TARGET_VPTO="${OUT}/target_mi.vpto"
LLVM_IR="${OUT}/target_mi.ll"
OBJ="${OUT}/target_mi.o"

set +e
if command -v ptoas >/dev/null 2>&1; then
  ptoas --cann-output-version=9.0.0 --pto-arch=a5 --pto-backend=vpto \
    --emit-vpto "${TARGET_MI}" -o "${TARGET_VPTO}" \
    > "${OUT}/target_mi_emit_vpto.log" 2>&1
  emit_vpto_rc=$?
else
  echo "SKIP: ptoas not on PATH" | tee -a "${LOG}"
  emit_vpto_rc=127
fi
set -e

if [ "${emit_vpto_rc}" -eq 0 ] && [ -s "${TARGET_VPTO}" ]; then
  echo "PASS: target_mi.pto --emit-vpto" | tee -a "${LOG}"
else
  echo "FAIL: target_mi.pto --emit-vpto exit ${emit_vpto_rc}" | tee -a "${LOG}"
  tail -20 "${OUT}/target_mi_emit_vpto.log" | tee -a "${LOG}"
fi
echo | tee -a "${LOG}"

echo "=== target_mi.pto (ptoas -> LLVM IR -> bisheng object) ===" | tee -a "${LOG}"
set +e
if command -v ptoas >/dev/null 2>&1; then
  ptoas --cann-output-version=9.0.0 --pto-arch=a5 --pto-backend=vpto \
    --emit-vpto-llvm-ir "${TARGET_MI}" -o "${LLVM_IR}" \
    > "${OUT}/target_mi_ptoas.log" 2>&1
  pto_rc=$?
else
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
    tail -10 "${OUT}/target_mi_bisheng.log" | tee -a "${LOG}"
  fi
else
  echo "FAIL: ptoas emit-vpto-llvm-ir exit ${pto_rc}" | tee -a "${LOG}"
  tail -20 "${OUT}/target_mi_ptoas.log" | tee -a "${LOG}"
fi
echo | tee -a "${LOG}"

# --- current_slow_vmi stages=1: lowered_vpto.pto is the working MI path ---
echo "=== current_slow_vmi.py (ptodsl -> MLIR frontend, stages=1 block_k=512) ===" | tee -a "${LOG}"
set +e
python3 "${FIXTURES}/current_slow_vmi.py" > "${OUT}/current_slow_vmi.mlir" 2> "${OUT}/current_slow_vmi_py.err"
slow_py_rc=$?
set -e
if [ "${slow_py_rc}" -eq 0 ]; then
  echo "PASS: current_slow_vmi.py emitted MLIR (frontend only)" | tee -a "${LOG}"
else
  echo "FAIL: current_slow_vmi.py compile exit ${slow_py_rc}" | tee -a "${LOG}"
  tail -20 "${OUT}/current_slow_vmi_py.err" | tee -a "${LOG}"
fi
echo | tee -a "${LOG}"

echo "=== lowered_vpto.pto (stages=1 chunked MI, ptoas -> LLVM IR) ===" | tee -a "${LOG}"
LOWERED_PTO="${FIXTURES}/lowered_vpto.pto"
SLOW_LL="${OUT}/lowered_vpto.ll"
set +e
if command -v ptoas >/dev/null 2>&1; then
  ptoas --cann-output-version=9.0.0 --pto-arch=a5 --pto-backend=vpto \
    --emit-vpto-llvm-ir "${LOWERED_PTO}" -o "${SLOW_LL}" \
    > "${OUT}/lowered_vpto_ptoas.log" 2>&1
  lowered_rc=$?
else
  lowered_rc=127
fi
set -e
if [ "${lowered_rc}" -eq 0 ] && [ -s "${SLOW_LL}" ]; then
  echo "PASS: lowered_vpto.pto -> LLVM IR (stages=1 working MI path)" | tee -a "${LOG}"
else
  echo "FAIL: lowered_vpto.pto emit-vpto-llvm-ir exit ${lowered_rc}" | tee -a "${LOG}"
  tail -20 "${OUT}/lowered_vpto_ptoas.log" | tee -a "${LOG}"
fi

echo | tee -a "${LOG}"
echo "Full log: ${LOG}" | tee -a "${LOG}"
