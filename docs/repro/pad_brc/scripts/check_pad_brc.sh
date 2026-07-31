#!/usr/bin/env bash
# Compile/lower pad_brc fixtures; print PASS/FAIL for each path.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPRO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
FIXTURES="${REPRO_ROOT}/fixtures"
OUT="${REPRO_ROOT}/outputs/check_pad_brc"
LOG="${OUT}/compile_results.txt"
VMI_PASSES=(-vmi-lower-unified-to-legacy -vmi-to-vpto)

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

if ! command -v pto-test-opt >/dev/null 2>&1; then
  echo "ERROR: pto-test-opt not found; build this PTOAS tree (or set PTOAS_ROOT)." | tee -a "${LOG}"
  exit 1
fi

echo "pad_brc compile check — $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "${LOG}"
echo "PTOAS_ROOT=${PTOAS_ROOT}" | tee -a "${LOG}"
echo "pto-test-opt=$(command -v pto-test-opt)" | tee -a "${LOG}"
echo "bisheng=${BISHENG:-MISSING}" | tee -a "${LOG}"
echo | tee -a "${LOG}"

# --- current_pad_brc_vmi.pto ---
echo "=== current_pad_brc_vmi.pto (pad-32 store + brc reload) ===" | tee -a "${LOG}"
CURRENT_OUT="${OUT}/current_pad_brc_vmi.vpto"
set +e
pto-test-opt "${FIXTURES}/current_pad_brc_vmi.pto" "${VMI_PASSES[@]}" -o "${CURRENT_OUT}" \
  > "${OUT}/current_pad_brc_vmi.log" 2>&1
rc=$?
set -e
if [ "${rc}" -eq 0 ] && [ -s "${CURRENT_OUT}" ]; then
  pass "current_pad_brc_vmi.pto lowers to VPTO"
else
  fail "current_pad_brc_vmi.pto lower exit ${rc}"
  tail -20 "${OUT}/current_pad_brc_vmi.log" | tee -a "${LOG}"
fi
echo | tee -a "${LOG}"

# --- desired_compact_vmi.pto ---
echo "=== desired_compact_vmi.pto (in-register vbrc group=4) ===" | tee -a "${LOG}"
DESIRED_OUT="${OUT}/desired_compact_vmi.vpto"
set +e
pto-test-opt "${FIXTURES}/desired_compact_vmi.pto" "${VMI_PASSES[@]}" -o "${DESIRED_OUT}" \
  > "${OUT}/desired_compact_vmi.log" 2>&1
rc=$?
set -e
if [ "${rc}" -eq 0 ] && [ -s "${DESIRED_OUT}" ]; then
  pass "desired_compact_vmi.pto lowers to VPTO (feature may be closed)"
else
  fail "desired_compact_vmi.pto lower exit ${rc} (residual group_broadcast expected)"
  grep -nE "group_broadcast|VMI-UNSUPPORTED|UNSUPPORTED|illegal" "${OUT}/desired_compact_vmi.log" \
    | tail -8 | tee -a "${LOG}" \
    || tail -15 "${OUT}/desired_compact_vmi.log" | tee -a "${LOG}"
fi
echo | tee -a "${LOG}"

# --- reference_asc_cce.asc ---
echo "=== reference_asc_cce.asc (bisheng) ===" | tee -a "${LOG}"
if [ -z "${BISHENG}" ] || [ ! -x "${BISHENG}" ]; then
  fail "bisheng not found; skip ASC baseline compile"
else
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
    pass "reference_asc_cce.asc → .o (reg-local inverse baseline)"
  else
    fail "reference_asc_cce.asc compile exit ${ref_rc}"
    tail -30 "${OUT}/reference_asc_cce.log" | tee -a "${LOG}"
  fi
fi
echo | tee -a "${LOG}"

echo "Full log: ${LOG}" | tee -a "${LOG}"
