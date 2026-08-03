#!/usr/bin/env bash
# Compile + numeric check for ub_table_scatter_gather fixtures.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPRO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
FIXTURES="${REPRO_ROOT}/fixtures"
OUT="${REPRO_ROOT}/outputs/check_ub_table_scatter_gather"
LOG="${OUT}/results.txt"

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
PY="$(command -v python || command -v python3)"

pass() { echo "PASS: $*" | tee -a "${LOG}"; }
fail() { echo "FAIL: $*" | tee -a "${LOG}"; }
skip() { echo "SKIP: $*" | tee -a "${LOG}"; }

echo "ub_table_scatter_gather check — $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "${LOG}"
echo "PTOAS_ROOT=${PTOAS_ROOT}" | tee -a "${LOG}"
echo "python=${PY}" | tee -a "${LOG}"
echo "bisheng=${BISHENG:-MISSING}" | tee -a "${LOG}"
echo | tee -a "${LOG}"

echo "=== failing_vmi.py ===" | tee -a "${LOG}"
set +e
PYTHONPATH="${FIXTURES}:${PYTHONPATH:-}" "${PY}" - <<'PY' > "${OUT}/failing_vmi.log" 2>&1
from failing_vmi import ub_table_scatter_gather, run_numeric
compiled = ub_table_scatter_gather.compile()
print("COMPILE_OK", len(compiled.mlir_text()))
try:
    r = run_numeric()
except Exception as exc:
    print(f"LAUNCH_ERR {type(exc).__name__}: {exc}")
    raise SystemExit(0)
print("NUMERIC", r)
raise SystemExit(0 if r.get("ok") else 1)
PY
rc=$?
set -e
tee -a "${LOG}" < "${OUT}/failing_vmi.log"
if grep -q "COMPILE_OK" "${OUT}/failing_vmi.log"; then
  pass "failing_vmi.py compiles"
else
  fail "failing_vmi.py compile"
fi
if grep -qE "LAUNCH_ERR|NPU not available" "${OUT}/failing_vmi.log"; then
  skip "failing_vmi.py numeric (no device / launch error)"
elif grep -q "all_neg_inf.: True" "${OUT}/failing_vmi.log"; then
  fail "failing_vmi.py numeric — all lanes -inf (bug reproduced)"
elif [ "${rc}" -eq 0 ]; then
  pass "failing_vmi.py numeric identity (bug may be fixed)"
else
  fail "failing_vmi.py numeric mismatch"
fi
echo | tee -a "${LOG}"

for v in variant_no_barrier variant_uint_offsets variant_byte_offsets; do
  echo "=== ${v}.py (compile) ===" | tee -a "${LOG}"
  set +e
  PYTHONPATH="${FIXTURES}:${PYTHONPATH:-}" "${PY}" "${FIXTURES}/${v}.py" \
    > "${OUT}/${v}.log" 2>&1
  vrc=$?
  set -e
  if [ "${vrc}" -eq 0 ]; then
    pass "${v}.py compiles"
  else
    fail "${v}.py compile exit ${vrc}"
    tail -15 "${OUT}/${v}.log" | tee -a "${LOG}"
  fi
  echo | tee -a "${LOG}"
done

echo "=== reference_asc_cce.asc (bisheng) ===" | tee -a "${LOG}"
if [ -z "${BISHENG}" ] || [ ! -x "${BISHENG}" ]; then
  skip "bisheng not found"
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
    pass "reference_asc_cce.asc → .o"
  else
    fail "reference_asc_cce.asc compile exit ${ref_rc}"
    tail -30 "${OUT}/reference_asc_cce.log" | tee -a "${LOG}"
  fi
fi
echo | tee -a "${LOG}"
echo "Full log: ${LOG}" | tee -a "${LOG}"
