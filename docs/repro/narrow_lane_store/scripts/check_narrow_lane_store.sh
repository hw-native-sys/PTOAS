#!/usr/bin/env bash
# Compile + numeric check for narrow_lane_store fixtures.
# Each VMI fixture runs in its own Python process so a 507035 fault cannot
# poison the padded workaround launch.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPRO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
FIXTURES="${REPRO_ROOT}/fixtures"
OUT="${REPRO_ROOT}/outputs/check_narrow_lane_store"
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

echo "narrow_lane_store check — $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "${LOG}"
echo "PTOAS_ROOT=${PTOAS_ROOT}" | tee -a "${LOG}"
echo "python=${PY}" | tee -a "${LOG}"
echo "bisheng=${BISHENG:-MISSING}" | tee -a "${LOG}"
echo | tee -a "${LOG}"

run_fixture() {
  local name="$1"
  local expect="$2"  # fault | pass
  echo "=== ${name} ===" | tee -a "${LOG}"
  set +e
  PYTHONPATH="${FIXTURES}:${PYTHONPATH:-}" "${PY}" - <<PY > "${OUT}/${name}.log" 2>&1
import importlib
m = importlib.import_module("${name}")
kern = next(getattr(m, n) for n in dir(m) if n.startswith("narrow_lane_store_"))
compiled = kern.compile()
print("COMPILE_OK", len(compiled.mlir_text()))
try:
    r = m.run_numeric()
    print("NUMERIC", r)
    raise SystemExit(0 if r.get("ok") else 1)
except RuntimeError as exc:
    msg = str(exc)
    print(f"LAUNCH_ERR {type(exc).__name__}: {exc}")
    if "requires 'addr' operand" in msg or "pto-level=level3" in msg:
        print("NATIVE_BUILD_ERR")
    elif "507035" in msg or "vector core" in msg.lower():
        print("VECTOR_CORE_FAULT")
    raise SystemExit(0)
except Exception as exc:
    msg = str(exc)
    print(f"LAUNCH_ERR {type(exc).__name__}: {exc}")
    if "requires 'addr' operand" in msg or "pto-level=level3" in msg:
        print("NATIVE_BUILD_ERR")
    elif "507035" in msg or "vector core" in msg.lower():
        print("VECTOR_CORE_FAULT")
    raise SystemExit(0)
PY
  rc=$?
  set -e
  tee -a "${LOG}" < "${OUT}/${name}.log"
  if grep -q "COMPILE_OK" "${OUT}/${name}.log"; then
    pass "${name} compiles"
  else
    fail "${name} compile"
    echo | tee -a "${LOG}"
    return
  fi
  if grep -q "NATIVE_BUILD_ERR" "${OUT}/${name}.log"; then
    fail "${name} native build (level3/addr)"
  elif grep -qE "LAUNCH_ERR.*NPU not available" "${OUT}/${name}.log"; then
    skip "${name} numeric (no device)"
  elif grep -qE "VECTOR_CORE_FAULT|507035" "${OUT}/${name}.log"; then
    if [ "${expect}" = "fault" ]; then
      fail "${name} launch — vector core fault (bug reproduced)"
    else
      fail "${name} launch — unexpected vector core fault"
    fi
  elif grep -q "'ok': True" "${OUT}/${name}.log"; then
    pass "${name} numeric"
  else
    fail "${name} numeric mismatch/error (rc=${rc})"
  fi
  echo | tee -a "${LOG}"
}

# Padded first so a clean process records the workaround before the fault case.
run_fixture "padded_workaround_vmi" "pass"
run_fixture "faulting_vmi" "fault"

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
