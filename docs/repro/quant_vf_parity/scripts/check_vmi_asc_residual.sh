#!/usr/bin/env bash
# Compile/lower AscendC vs VMI residual fixtures; print PASS/FAIL for each path.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPRO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
FIXTURES="${REPRO_ROOT}/fixtures"
OUT="${REPRO_ROOT}/outputs/check_vmi_asc_residual"
LOG="${OUT}/compile_results.txt"
# Lit-order pipeline: lower → mask → layout → VPTO (works for layouted and unlayouted).
VMI_PASSES=(
  -vmi-lower-unified-to-legacy
  -vmi-mask-granularity-assignment
  -vmi-layout-assignment
  -vmi-to-vpto
)

# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env.sh"

mkdir -p "${OUT}"
: > "${LOG}"

BISHENG="${BISHENG:-${ASCEND_HOME_PATH}/tools/bisheng_compiler/bin/bisheng}"
if [ ! -x "${BISHENG}" ]; then
  BISHENG="$(command -v bisheng || true)"
fi
ASCEND="${ASCEND_HOME_PATH}"
NPU_ARCH="${ASCEND_NPU_ARCH:-dav-3510}"

pass() { echo "PASS: $*" | tee -a "${LOG}"; }
fail() { echo "FAIL: $*" | tee -a "${LOG}"; }
note() { echo "NOTE: $*" | tee -a "${LOG}"; }

if ! command -v pto-test-opt >/dev/null 2>&1; then
  echo "ERROR: pto-test-opt not found; build this PTOAS tree (or set PTOAS_ROOT)." | tee -a "${LOG}"
  exit 1
fi

echo "vmi_asc_residual compile check — $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "${LOG}"
echo "PTOAS_ROOT=${PTOAS_ROOT}" | tee -a "${LOG}"
echo "pto-test-opt=$(command -v pto-test-opt)" | tee -a "${LOG}"
echo "ptoas=$(command -v ptoas || echo MISSING)" | tee -a "${LOG}"
echo "bisheng=${BISHENG:-MISSING}" | tee -a "${LOG}"
echo | tee -a "${LOG}"

check_asc() {
  local name="$1"
  echo "=== ${name} (bisheng) ===" | tee -a "${LOG}"
  if [ -z "${BISHENG}" ] || [ ! -x "${BISHENG}" ]; then
    fail "bisheng not found; skip ${name}"
    echo | tee -a "${LOG}"
    return
  fi
  local obj="${OUT}/${name%.asc}.o"
  set +e
  "${BISHENG}" -O2 -fPIC -std=c++17 --npu-arch="${NPU_ARCH}" --cce-aicore-only -c \
    "${FIXTURES}/${name}" -o "${obj}" \
    -I"${ASCEND}/include" \
    -I"${ASCEND}/compiler/tikcpp/tikcfw" \
    -I"${ASCEND}/compiler/tikcpp/tikcfw/impl" \
    -I"${ASCEND}/compiler/tikcpp/tikcfw/interface" \
    > "${OUT}/${name}.log" 2>&1
  local rc=$?
  set -e
  if [ "${rc}" -eq 0 ] && [ -s "${obj}" ]; then
    pass "${name} compiles with bisheng"
  else
    fail "${name} compile exit ${rc}"
    tail -30 "${OUT}/${name}.log" | tee -a "${LOG}"
  fi
  echo | tee -a "${LOG}"
}

check_vmi_lower() {
  local name="$1"
  echo "=== ${name} (pto-test-opt VMI→VPTO) ===" | tee -a "${LOG}"
  local outf="${OUT}/${name%.pto}.vpto"
  set +e
  pto-test-opt "${FIXTURES}/${name}" "${VMI_PASSES[@]}" -o "${outf}" \
    > "${OUT}/${name}.log" 2>&1
  local rc=$?
  set -e
  if [ "${rc}" -eq 0 ] && [ -s "${outf}" ]; then
    pass "${name} lowers to VPTO"
  else
    fail "${name} lower exit ${rc}"
    grep -nE "UNSUPPORTED|error:|illegal|requires" "${OUT}/${name}.log" \
      | tail -12 | tee -a "${LOG}" \
      || tail -20 "${OUT}/${name}.log" | tee -a "${LOG}"
  fi
  echo | tee -a "${LOG}"
}

# Ask 1: get_block_idx bound check around VF (expect expand FAIL today)
check_asc reference_asc_block_idx_vf.asc

echo "=== current_vmi_block_idx_vf.pto (ptoas device emit) ===" | tee -a "${LOG}"
BID="${FIXTURES}/current_vmi_block_idx_vf.pto"
BID_LOG="${OUT}/current_vmi_block_idx_vf.emit.log"
BID_OBJ="${OUT}/current_vmi_block_idx_vf.fatobj.o"
if ! command -v ptoas >/dev/null 2>&1; then
  fail "ptoas not found; skip Ask 1 emit"
else
  set +e
  ptoas --pto-arch=a5 --pto-backend=vpto --pto-level=level3 \
    --cann-output-version=9.1.0-beta.3 \
    "${BID}" -o "${BID_OBJ}" > "${BID_LOG}" 2>&1
  bid_rc=$?
  set -e
  if [ "${bid_rc}" -eq 0 ] && [ -s "${BID_OBJ}" ]; then
    pass "current_vmi_block_idx_vf.pto emits (Ask 1 may be closed)"
  else
    fail "current_vmi_block_idx_vf.pto emit exit ${bid_rc} (Ask 1 open if expand error)"
    if grep -q "Do not know how to expand the result of this operator" "${BID_LOG}"; then
      note "matched expected expand failure string"
    fi
    grep -nE "expand the result|error:|Error:" "${BID_LOG}" | tail -12 | tee -a "${LOG}" \
      || tail -25 "${BID_LOG}" | tee -a "${LOG}"
  fi
fi
echo | tee -a "${LOG}"

if [ -f "${FIXTURES}/current_vmi_multibuf_expand_full.pto" ] && command -v ptoas >/dev/null 2>&1; then
  echo "=== current_vmi_multibuf_expand_full.pto (ptoas device emit) ===" | tee -a "${LOG}"
  FULL_LOG="${OUT}/current_vmi_multibuf_expand_full.emit.log"
  FULL_OBJ="${OUT}/current_vmi_multibuf_expand_full.fatobj.o"
  set +e
  ptoas --pto-arch=a5 --pto-backend=vpto --pto-level=level3 \
    --cann-output-version=9.1.0-beta.3 \
    "${FIXTURES}/current_vmi_multibuf_expand_full.pto" -o "${FULL_OBJ}" \
    > "${FULL_LOG}" 2>&1
  full_rc=$?
  set -e
  if [ "${full_rc}" -eq 0 ] && [ -s "${FULL_OBJ}" ]; then
    pass "current_vmi_multibuf_expand_full.pto emits"
  else
    fail "current_vmi_multibuf_expand_full.pto emit exit ${full_rc}"
    if grep -q "Do not know how to expand the result of this operator" "${FULL_LOG}"; then
      note "matched expected expand failure string on full dump"
    fi
    grep -nE "expand the result|error:|Error:" "${FULL_LOG}" | tail -12 | tee -a "${LOG}" \
      || tail -20 "${FULL_LOG}" | tee -a "${LOG}"
  fi
  echo | tee -a "${LOG}"
fi

# Ask 2
check_asc reference_asc_dequant_dblbuf.asc
check_vmi_lower broken_vmi_dequant_dblbuf.pto
check_vmi_lower working_vmi_dequant_narrow.pto
note "Ask 2 device mismatch: see fixtures/RECORDED_DEVICE_NOTES.md"

# Ask 3
check_asc reference_asc_fp32_strip_amax.asc
check_vmi_lower current_vmi_fp32_strip_amax.pto
note "Ask 3 emit evidence: see fixtures/emit_compare_note.md"

echo "Full log: ${LOG}" | tee -a "${LOG}"
