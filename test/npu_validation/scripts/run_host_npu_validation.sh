#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

WORK_SPACE="${WORK_SPACE:-}"
ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-}"
PTO_ISA_ROOT="${PTO_ISA_ROOT:-}"
PTOAS_BIN="${PTOAS_BIN:-}"
PTOAS_FLAGS="${PTOAS_FLAGS:---pto-arch a5}"
A5VM_FLAGS="${A5VM_FLAGS:---pto-backend=a5vm --a5vm-emit-hivm-llvm}"
SAMPLE_NAME="${SAMPLE_NAME:-}"
TESTCASE_NAME="${TESTCASE_NAME:-}"

SOC_VERSION="${SOC_VERSION:-A5}"
AICORE_ARCH="${AICORE_ARCH:-dav-c310-vec}"
HOST_RUNNER="${HOST_RUNNER:-ssh root@localhost}"
RESULTS_TSV="${RESULTS_TSV:-}"
SEED="${SEED:-19}"
GOLDEN_MODE="${GOLDEN_MODE:-py}"  # py|skip
KERNEL_MODE="${KERNEL_MODE:-llvm}"  # llvm|emitc

log() {
  echo "[$(date +'%F %T')] $*"
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

require_env() {
  local name="$1"
  local value="$2"
  local example="$3"
  if [[ -z "${value}" ]]; then
    echo "ERROR: ${name} is required." >&2
    echo "Example: export ${name}=${example}" >&2
    exit 1
  fi
}

require_env "WORK_SPACE" "${WORK_SPACE}" "/tmp/ptoas-npu-validation"
require_env "ASCEND_HOME_PATH" "${ASCEND_HOME_PATH}" "/usr/local/Ascend/cann-9.0.0"
require_env "PTO_ISA_ROOT" "${PTO_ISA_ROOT}" "/path/to/pto-isa"
require_env "PTOAS_BIN" "${PTOAS_BIN}" "/path/to/build/tools/ptoas/ptoas"
require_env "SAMPLE_NAME" "${SAMPLE_NAME}" "Cmp"

[[ -x "${PTOAS_BIN}" ]] || die "PTOAS_BIN is not executable: ${PTOAS_BIN}"
[[ -d "${PTO_ISA_ROOT}/include" ]] || die "PTO_ISA_ROOT/include missing: ${PTO_ISA_ROOT}"
[[ -d "${ASCEND_HOME_PATH}" ]] || die "ASCEND_HOME_PATH missing: ${ASCEND_HOME_PATH}"

set +u
source "${ROOT_DIR}/env.sh" >/dev/null 2>&1
set -u

if [[ -f "${ASCEND_HOME_PATH}/set_env.sh" ]]; then
  set +u
  source "${ASCEND_HOME_PATH}/set_env.sh" >/dev/null 2>&1
  set -u
fi

command -v python3 >/dev/null 2>&1 || die "python3 not found"
command -v cmake >/dev/null 2>&1 || die "cmake not found"
command -v bisheng >/dev/null 2>&1 || die "bisheng not found in PATH"

SAMPLE_OUT_DIR="${WORK_SPACE}/emitc"
TESTCASE_ROOT="${WORK_SPACE}/testcase"
SAMPLE_VALIDATION_DIR="${ROOT_DIR}/test/samples/${SAMPLE_NAME}/npu_validation"
CASE_RUNNER="${SCRIPT_DIR}/run_host_npu_validation_case.sh"

[[ -x "${CASE_RUNNER}" ]] || die "missing case runner: ${CASE_RUNNER}"
[[ -d "${ROOT_DIR}/test/samples/${SAMPLE_NAME}" ]] || die "missing sample dir: ${ROOT_DIR}/test/samples/${SAMPLE_NAME}"
[[ -d "${SAMPLE_VALIDATION_DIR}" ]] || die "missing npu_validation dir: ${SAMPLE_VALIDATION_DIR}"

mkdir -p "${WORK_SPACE}"
RESULTS_TSV="${RESULTS_TSV:-${WORK_SPACE}/host_npu_validation_results.tsv}"
printf "testcase\tstatus\tstage\tinfo\n" > "${RESULTS_TSV}"

discover_testcases() {
  local golden
  if [[ -n "${TESTCASE_NAME}" ]]; then
    golden="${SAMPLE_VALIDATION_DIR}/${TESTCASE_NAME}/golden.py"
    [[ -f "${golden}" ]] || die "testcase is not wired for host validation: ${SAMPLE_NAME}/${TESTCASE_NAME}"
    printf "%s\n" "${TESTCASE_NAME}"
    return 0
  fi

  find "${SAMPLE_VALIDATION_DIR}" -mindepth 2 -maxdepth 2 -type f -name 'golden.py' \
    | sort \
    | while read -r golden; do
        basename "$(dirname "${golden}")"
      done
}

mapfile -t TESTCASES < <(discover_testcases)
[[ ${#TESTCASES[@]} -gt 0 ]] || die "no host validation testcase found under ${SAMPLE_VALIDATION_DIR}"

log "=== Host NPU Validation ==="
log "WORK_SPACE=${WORK_SPACE}"
log "ASCEND_HOME_PATH=${ASCEND_HOME_PATH}"
log "PTO_ISA_ROOT=${PTO_ISA_ROOT}"
log "PTOAS_BIN=${PTOAS_BIN}"
log "PTOAS_FLAGS=${PTOAS_FLAGS}"
log "A5VM_FLAGS=${A5VM_FLAGS}"
log "SAMPLE_NAME=${SAMPLE_NAME}"
log "TESTCASE_NAME=${TESTCASE_NAME:-<all>}"
log "HOST_RUNNER=${HOST_RUNNER}"
log "GOLDEN_MODE=${GOLDEN_MODE}"
log "KERNEL_MODE=${KERNEL_MODE}"
log "RESULTS_TSV=${RESULTS_TSV}"

log "Exporting sample ${SAMPLE_NAME} with runop.sh"
set +e
PTOAS_BIN="${PTOAS_BIN}" \
PTOAS_OUT_DIR="${SAMPLE_OUT_DIR}" \
PTOAS_FLAGS="${PTOAS_FLAGS}" \
  "${ROOT_DIR}/test/samples/runop.sh" -t "${SAMPLE_NAME}"
rc=$?
set -e
[[ ${rc} -eq 0 ]] || die "sample export failed for ${SAMPLE_NAME} (runop_exit=${rc})"

fail_count=0
for testcase in "${TESTCASES[@]}"; do
  log "Dispatching ${SAMPLE_NAME}/${testcase}"
  set +e
  SAMPLE_NAME="${SAMPLE_NAME}" \
  TESTCASE_NAME="${testcase}" \
  WORK_SPACE="${WORK_SPACE}" \
  ASCEND_HOME_PATH="${ASCEND_HOME_PATH}" \
  PTO_ISA_ROOT="${PTO_ISA_ROOT}" \
  PTOAS_BIN="${PTOAS_BIN}" \
  PTOAS_FLAGS="${PTOAS_FLAGS}" \
  A5VM_FLAGS="${A5VM_FLAGS}" \
  SOC_VERSION="${SOC_VERSION}" \
  AICORE_ARCH="${AICORE_ARCH}" \
  HOST_RUNNER="${HOST_RUNNER}" \
  RESULTS_TSV="${RESULTS_TSV}" \
  SEED="${SEED}" \
  GOLDEN_MODE="${GOLDEN_MODE}" \
  KERNEL_MODE="${KERNEL_MODE}" \
    "${CASE_RUNNER}"
  rc=$?
  set -e
  if [[ ${rc} -ne 0 ]]; then
    fail_count=$((fail_count + 1))
    log "Case failed: ${SAMPLE_NAME}/${testcase} (exit=${rc})"
  fi
done

if [[ ${fail_count} -ne 0 ]]; then
  die "${fail_count} testcase(s) failed for sample ${SAMPLE_NAME}. See ${RESULTS_TSV}"
fi

log "All ${#TESTCASES[@]} testcase(s) passed for sample ${SAMPLE_NAME}"
