#!/usr/bin/env bash
# Compile/lower AscendC vs VMI fixtures for the feature requests; print PASS/FAIL.
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
    -I"${FIXTURES}" \
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

# Feature request 1: get_block_idx controlling a vector body (expect expand FAIL today)
check_asc reference_asc_block_idx_vf.asc

echo "=== current_vmi_block_idx_vf.pto (ptoas device emit) ===" | tee -a "${LOG}"
BID="${FIXTURES}/current_vmi_block_idx_vf.pto"
BID_LOG="${OUT}/current_vmi_block_idx_vf.emit.log"
BID_OBJ="${OUT}/current_vmi_block_idx_vf.fatobj.o"
if ! command -v ptoas >/dev/null 2>&1; then
  fail "ptoas not found; skip feature request 1 emit"
else
  set +e
  ptoas --pto-arch=a5 --pto-backend=vpto --pto-level=level3 \
    --cann-output-version=9.1.0-beta.3 \
    "${BID}" -o "${BID_OBJ}" > "${BID_LOG}" 2>&1
  bid_rc=$?
  set -e
  if [ "${bid_rc}" -eq 0 ] && [ -s "${BID_OBJ}" ]; then
    pass "current_vmi_block_idx_vf.pto emits (feature request 1 may be closed)"
  else
    fail "current_vmi_block_idx_vf.pto emit exit ${bid_rc} (feature request 1 open if expand error)"
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

# Feature request 2
check_asc reference_asc_dequant_dblbuf.asc
check_vmi_lower broken_vmi_dequant_dblbuf.pto
check_vmi_lower working_vmi_dequant_narrow.pto
note "Feature request 2 on-device mismatch: see fixtures/RECORDED_DEVICE_NOTES.md"

# Backlog — former feature request 3 (solved; not a blocker)
note "Backlog FR3: issue solved, not a blocker now — kept for regression only"
check_asc reference_asc_fp32_block_quant.asc
if [ -f "${FIXTURES}/current_vmi_fp32_block_quant_8192x2048.ptodsl.py" ]; then
  echo "=== current_vmi_fp32_block_quant_8192x2048.ptodsl.py (syntax / import) ===" | tee -a "${LOG}"
  set +e
  PYTHONPATH="${PTOAS_ROOT}/ptodsl:${PYTHONPATH:-}" python3 -c "
import ast, pathlib
p = pathlib.Path('${FIXTURES}/current_vmi_fp32_block_quant_8192x2048.ptodsl.py')
ast.parse(p.read_text())
print('ast ok', p.name)
" > "${OUT}/current_vmi_fp32_block_quant_8192x2048.ptodsl.log" 2>&1
  ptodsl_rc=$?
  set -e
  if [ "${ptodsl_rc}" -eq 0 ]; then
    pass "current_vmi_fp32_block_quant_8192x2048.ptodsl.py parses"
  else
    fail "current_vmi_fp32_block_quant_8192x2048.ptodsl.py parse exit ${ptodsl_rc}"
    tail -20 "${OUT}/current_vmi_fp32_block_quant_8192x2048.ptodsl.log" | tee -a "${LOG}"
  fi
  echo | tee -a "${LOG}"
fi
note "Backlog FP32 µs (solved wall gap): fixtures/PERF_FINDINGS.md"
note "Optional backlog device run: ./scripts/run_fp32_block_quant_device.sh (skip if no NPU)"

check_asc reference_asc_fp32_strip_amax.asc
check_vmi_lower current_vmi_fp32_strip_amax.pto
note "Backlog strip fixtures are VF fragments only (not an open ask)"

echo "Full log: ${LOG}" | tee -a "${LOG}"
