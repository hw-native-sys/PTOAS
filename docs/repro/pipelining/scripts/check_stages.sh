#!/usr/bin/env bash
# Compile-check CCE stages=2 vs VMI stages×block_k isolate matrix (kernel-agnostic).
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

compile_asc() {
  local name="$1"
  local src="${FIXTURES}/${name}"
  local obj="${OUT}/${name%.asc}.o"
  local logf="${OUT}/${name%.asc}.log"
  echo "=== ${name} (bisheng --cce-aicore-only) ===" | tee -a "${LOG}"
  set +e
  "${BISHENG}" -O2 -fPIC -std=c++17 --npu-arch="${NPU_ARCH}" --cce-aicore-only -c \
    "${src}" -o "${obj}" \
    -I"${ASCEND}/include" \
    -I"${ASCEND}/compiler/tikcpp/tikcfw" \
    -I"${ASCEND}/compiler/tikcpp/tikcfw/impl" \
    -I"${ASCEND}/compiler/tikcpp/tikcfw/interface" \
    > "${logf}" 2>&1
  local rc=$?
  set -e
  if [ "${rc}" -eq 0 ] && [ -s "${obj}" ]; then
    echo "PASS: ${name} -> non-empty .o" | tee -a "${LOG}"
  else
    echo "FAIL: ${name} compile exit ${rc}" | tee -a "${LOG}"
    tail -20 "${logf}" | tee -a "${LOG}"
  fi
  echo | tee -a "${LOG}"
}

emit_vpto() {
  local name="$1"
  local expect="$2"  # pass|fail
  local src="${FIXTURES}/${name}"
  local outf="${OUT}/${name%.pto}.vpto"
  local logf="${OUT}/${name%.pto}_emit.log"
  echo "=== ${name} (ptoas --emit-vpto) expect=${expect} ===" | tee -a "${LOG}"
  set +e
  if command -v ptoas >/dev/null 2>&1; then
    ptoas --cann-output-version=9.0.0 --pto-arch=a5 --pto-backend=vpto \
      --emit-vpto "${src}" -o "${outf}" > "${logf}" 2>&1
    local rc=$?
  else
    echo "SKIP: ptoas not on PATH" | tee -a "${LOG}"
    rc=127
  fi
  set -e
  if [ "${rc}" -eq 0 ] && [ -s "${outf}" ]; then
    if [ "${expect}" = "pass" ]; then
      echo "PASS: ${name} --emit-vpto" | tee -a "${LOG}"
    else
      echo "PASS: ${name} --emit-vpto (unexpected — gap may be closed)" | tee -a "${LOG}"
    fi
  else
    if [ "${expect}" = "fail" ]; then
      echo "FAIL: ${name} --emit-vpto exit ${rc} (expected layout/lower reject)" | tee -a "${LOG}"
    else
      echo "FAIL: ${name} --emit-vpto exit ${rc}" | tee -a "${LOG}"
    fi
    tail -12 "${logf}" | tee -a "${LOG}"
  fi
  echo | tee -a "${LOG}"
}

ptodsl_py() {
  local name="$1"
  local outf="${OUT}/${name%.py}.mlir"
  local errf="${OUT}/${name%.py}.err"
  echo "=== ${name} (ptodsl frontend) ===" | tee -a "${LOG}"
  set +e
  python3 "${FIXTURES}/${name}" > "${outf}" 2> "${errf}"
  local rc=$?
  set -e
  if [ "${rc}" -eq 0 ]; then
    echo "PASS: ${name} emitted MLIR (frontend only)" | tee -a "${LOG}"
  else
    echo "FAIL: ${name} exit ${rc}" | tee -a "${LOG}"
    tail -20 "${errf}" | tee -a "${LOG}"
  fi
  echo | tee -a "${LOG}"
}

# --- CCE baseline ---
compile_asc "reference_asc_cce.asc"

echo "=== device/scale_stages2.asc (bisheng --cce-aicore-only) ===" | tee -a "${LOG}"
set +e
"${BISHENG}" -O2 -fPIC -std=c++17 --npu-arch="${NPU_ARCH}" --cce-aicore-only -c \
  "${KERNEL_ROOT}/device/scale_stages2.asc" -o "${OUT}/scale_stages2_aicore.o" \
  -I"${ASCEND}/include" \
  -I"${ASCEND}/compiler/tikcpp/tikcfw" \
  -I"${ASCEND}/compiler/tikcpp/tikcfw/impl" \
  -I"${ASCEND}/compiler/tikcpp/tikcfw/interface" \
  > "${OUT}/scale_stages2_aicore.log" 2>&1
rc=$?
set -e
if [ "${rc}" -eq 0 ] && [ -s "${OUT}/scale_stages2_aicore.o" ]; then
  echo "PASS: device/scale_stages2.asc -> non-empty .o" | tee -a "${LOG}"
else
  echo "FAIL: device/scale_stages2.asc compile exit ${rc}" | tee -a "${LOG}"
  tail -20 "${OUT}/scale_stages2_aicore.log" | tee -a "${LOG}"
fi
echo | tee -a "${LOG}"

# --- VMI matrix: stages × block_k ---
ptodsl_py "current_slow_vmi.py"
emit_vpto "current_slow_vmi.pto" "pass"

echo "=== lowered_vpto.pto (stages=1 chunked MI -> LLVM IR) ===" | tee -a "${LOG}"
set +e
ptoas --cann-output-version=9.0.0 --pto-arch=a5 --pto-backend=vpto \
  --emit-vpto-llvm-ir "${FIXTURES}/lowered_vpto.pto" -o "${OUT}/lowered_vpto.ll" \
  > "${OUT}/lowered_vpto_ptoas.log" 2>&1
lowered_rc=$?
set -e
if [ "${lowered_rc}" -eq 0 ] && [ -s "${OUT}/lowered_vpto.ll" ]; then
  echo "PASS: lowered_vpto.pto -> LLVM IR" | tee -a "${LOG}"
else
  echo "FAIL: lowered_vpto.pto exit ${lowered_rc}" | tee -a "${LOG}"
  tail -15 "${OUT}/lowered_vpto_ptoas.log" | tee -a "${LOG}"
fi
echo | tee -a "${LOG}"

ptodsl_py "isolate_stages2_blockk512_vmi.py"
emit_vpto "isolate_stages2_blockk512_vmi.pto" "pass"

ptodsl_py "isolate_stages1_blockk1024_vmi.py"
emit_vpto "isolate_stages1_blockk1024_vmi.pto" "fail"

ptodsl_py "desired_vmi.py"
emit_vpto "desired_vmi.pto" "fail"

# target_mi: emit OK, bisheng object often crashes
echo "=== target_mi.pto (emit-vpto + LLVM IR + bisheng .o) ===" | tee -a "${LOG}"
set +e
ptoas --cann-output-version=9.0.0 --pto-arch=a5 --pto-backend=vpto \
  --emit-vpto "${FIXTURES}/target_mi.pto" -o "${OUT}/target_mi.vpto" \
  > "${OUT}/target_mi_emit_vpto.log" 2>&1
emit_rc=$?
set -e
if [ "${emit_rc}" -eq 0 ] && [ -s "${OUT}/target_mi.vpto" ]; then
  echo "PASS: target_mi.pto --emit-vpto" | tee -a "${LOG}"
else
  echo "FAIL: target_mi.pto --emit-vpto exit ${emit_rc}" | tee -a "${LOG}"
fi
set +e
ptoas --cann-output-version=9.0.0 --pto-arch=a5 --pto-backend=vpto \
  --emit-vpto-llvm-ir "${FIXTURES}/target_mi.pto" -o "${OUT}/target_mi.ll" \
  > "${OUT}/target_mi_ptoas.log" 2>&1
ll_rc=$?
set -e
if [ "${ll_rc}" -eq 0 ]; then
  echo "PASS: target_mi.pto --emit-vpto-llvm-ir" | tee -a "${LOG}"
  set +e
  "${BISHENG}" -O2 -fPIC -std=c++17 --npu-arch="${NPU_ARCH}" -c -x ir \
    "${OUT}/target_mi.ll" -o "${OUT}/target_mi.o" > "${OUT}/target_mi_bisheng.log" 2>&1
  b_rc=$?
  set -e
  if [ "${b_rc}" -eq 0 ] && [ -s "${OUT}/target_mi.o" ]; then
    echo "PASS: bisheng object (unexpected — gap may be closed)" | tee -a "${LOG}"
  else
    echo "FAIL: bisheng object exit ${b_rc} (expected on vmi-v0.1.3)" | tee -a "${LOG}"
    tail -8 "${OUT}/target_mi_bisheng.log" | tee -a "${LOG}"
  fi
else
  echo "FAIL: target_mi llvm-ir exit ${ll_rc}" | tee -a "${LOG}"
fi

echo | tee -a "${LOG}"
echo "Full log: ${LOG}" | tee -a "${LOG}"
