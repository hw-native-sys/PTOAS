#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

WORK_SPACE="${WORK_SPACE:-}"
ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-}"
PTO_ISA_ROOT="${PTO_ISA_ROOT:-}"
PTOAS_BIN="${PTOAS_BIN:-}"

SOC_VERSION="${SOC_VERSION:-A5}"
AICORE_ARCH="${AICORE_ARCH:-dav-c310-vec}"
HOST_RUNNER="${HOST_RUNNER:-ssh root@localhost}"
RESULTS_TSV="${RESULTS_TSV:-}"
SEED="${SEED:-19}"
RUN_ONLY_CASES="${RUN_ONLY_CASES:-Cmp}"
SKIP_CASES="${SKIP_CASES:-}"

log() {
  echo "[$(date +'%F %T')] $*"
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

normalize_list() {
  local s="$1"
  s="${s//$'\n'/,}"
  s="${s//$'\t'/,}"
  s="${s// /,}"
  while [[ "$s" == *",,"* ]]; do
    s="${s//,,/,}"
  done
  s="${s#,}"
  s="${s%,}"
  echo "$s"
}

list_contains() {
  local list="$1"
  local item="$2"
  [[ -n "${item}" ]] || return 1
  [[ ",${list}," == *",${item},"* ]]
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

run_remote() {
  local cmd="$1"
  if [[ "${HOST_RUNNER}" == "ssh root@localhost" ]]; then
    ssh -o StrictHostKeyChecking=no root@localhost "${cmd}"
  else
    eval "${HOST_RUNNER} ${cmd@Q}"
  fi
}

require_env "WORK_SPACE" "${WORK_SPACE}" "/tmp/ptoas-npu-validation-cmp-llvm"
require_env "ASCEND_HOME_PATH" "${ASCEND_HOME_PATH}" "/usr/local/Ascend/cann-9.0.0"
require_env "PTO_ISA_ROOT" "${PTO_ISA_ROOT}" "/path/to/pto-isa"
require_env "PTOAS_BIN" "${PTOAS_BIN}" "/path/to/build/tools/ptoas/ptoas"

[[ -x "${PTOAS_BIN}" ]] || die "PTOAS_BIN is not executable: ${PTOAS_BIN}"
[[ -d "${PTO_ISA_ROOT}/include" ]] || die "PTO_ISA_ROOT/include missing: ${PTO_ISA_ROOT}"
[[ -d "${PTO_ISA_ROOT}/tests/common" ]] || die "PTO_ISA_ROOT/tests/common missing: ${PTO_ISA_ROOT}"
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

RUN_ONLY_CASES_NORM="$(normalize_list "${RUN_ONLY_CASES}")"
SKIP_CASES_NORM="$(normalize_list "${SKIP_CASES}")"

if [[ -n "${RUN_ONLY_CASES_NORM}" ]] && ! list_contains "${RUN_ONLY_CASES_NORM}" "Cmp"; then
  die "This script only supports Cmp. RUN_ONLY_CASES=${RUN_ONLY_CASES}"
fi
if [[ -n "${RUN_ONLY_CASES_NORM}" ]]; then
  for item in ${RUN_ONLY_CASES_NORM//,/ }; do
    [[ "${item}" == "Cmp" ]] || die "Unsupported testcase: ${item}. Only Cmp is supported."
  done
fi
if [[ -n "${SKIP_CASES_NORM}" ]] && list_contains "${SKIP_CASES_NORM}" "Cmp"; then
  log "Cmp is listed in SKIP_CASES; nothing to do."
  exit 0
fi

mkdir -p "${WORK_SPACE}"
RESULTS_TSV="${RESULTS_TSV:-${WORK_SPACE}/host_npu_validation_results.tsv}"
printf "testcase\tstatus\tstage\tinfo\n" > "${RESULTS_TSV}"

SAMPLE_OUT_DIR="${WORK_SPACE}/emitc"
TESTCASE_ROOT="${WORK_SPACE}/testcase"
TESTCASE_DIR="${TESTCASE_ROOT}/Cmp/cmp"
BUILD_DIR="${TESTCASE_DIR}/build"
REPACK_DIR="${WORK_SPACE}/llvm_ir_kernel_so/Cmp/cmp/repack"
REPACK_SO="${REPACK_DIR}/libcmp_kernel.so"
SAMPLE_CPP="${SAMPLE_OUT_DIR}/Cmp/cmp-pto.cpp"

record_result() {
  local status="$1"
  local stage="$2"
  local info="$3"
  printf "Cmp\t%s\t%s\t%s\n" "${status}" "${stage}" "${info}" >> "${RESULTS_TSV}"
}

log "=== Host NPU Validation (Cmp) ==="
log "WORK_SPACE=${WORK_SPACE}"
log "ASCEND_HOME_PATH=${ASCEND_HOME_PATH}"
log "PTO_ISA_ROOT=${PTO_ISA_ROOT}"
log "PTOAS_BIN=${PTOAS_BIN}"
log "HOST_RUNNER=${HOST_RUNNER}"
log "RESULTS_TSV=${RESULTS_TSV}"

stage="generate"
set +e
PTOAS_BIN="${PTOAS_BIN}" \
PTOAS_OUT_DIR="${SAMPLE_OUT_DIR}" \
  "${ROOT_DIR}/test/samples/runop.sh" -t Cmp
rc=$?
set -e
if [[ ${rc} -ne 0 ]]; then
  record_result "FAIL" "${stage}" "runop_exit=${rc}"
  die "sample export failed at stage=${stage}"
fi
[[ -f "${SAMPLE_CPP}" ]] || {
  record_result "FAIL" "${stage}" "missing_sample_cpp"
  die "missing generated sample cpp: ${SAMPLE_CPP}"
}

set +e
python3 "${ROOT_DIR}/test/npu_validation/scripts/generate_testcase.py" \
  --input "${SAMPLE_CPP}" \
  --testcase cmp \
  --output-root "${TESTCASE_ROOT}" \
  --run-mode sim \
  --soc-version "${SOC_VERSION}" \
  --aicore-arch "${AICORE_ARCH}"
rc=$?
set -e
if [[ ${rc} -ne 0 ]]; then
  record_result "FAIL" "${stage}" "generate_testcase_exit=${rc}"
  die "generate_testcase failed at stage=${stage}"
fi

stage="build"
set +e
cmake -S "${TESTCASE_DIR}" -B "${BUILD_DIR}" \
  -DSOC_VERSION="${SOC_VERSION}" \
  -DENABLE_SIM_GOLDEN=OFF \
  -DPTO_ISA_ROOT="${PTO_ISA_ROOT}"
rc=$?
set -e
if [[ ${rc} -ne 0 ]]; then
  record_result "FAIL" "${stage}" "cmake_configure_exit=${rc}"
  die "cmake configure failed at stage=${stage}"
fi

set +e
cmake --build "${BUILD_DIR}" --parallel
rc=$?
set -e
if [[ ${rc} -ne 0 ]]; then
  record_result "FAIL" "${stage}" "cmake_build_exit=${rc}"
  die "cmake build failed at stage=${stage}"
fi

stage="build_so"
set +e
WORK_SPACE="${WORK_SPACE}" \
ASCEND_HOME_PATH="${ASCEND_HOME_PATH}" \
PTO_ISA_ROOT="${PTO_ISA_ROOT}" \
PTOAS_BIN="${PTOAS_BIN}" \
SAMPLE_NAME="Cmp" \
TESTCASE_NAME="cmp" \
SOC_VERSION="${SOC_VERSION}" \
AICORE_ARCH="${AICORE_ARCH}" \
  "${ROOT_DIR}/test/npu_validation/scripts/build_llvm_ir_kernel_so.sh"
rc=$?
set -e
if [[ ${rc} -ne 0 ]]; then
  record_result "FAIL" "${stage}" "build_llvm_ir_kernel_so_exit=${rc}"
  die "llvm ir kernel so build failed at stage=${stage}"
fi
[[ -f "${REPACK_SO}" ]] || {
  record_result "FAIL" "${stage}" "missing_repack_so"
  die "missing llvm ir kernel so: ${REPACK_SO}"
}

stage="generate"
set +e
python3 "${ROOT_DIR}/test/samples/Cmp/npu_validation/golden.py" \
  --output-dir "${TESTCASE_DIR}" \
  --seed "${SEED}" \
  --src-elem-bytes 4
rc=$?
set -e
if [[ ${rc} -ne 0 ]]; then
  record_result "FAIL" "${stage}" "golden_script_exit=${rc}"
  die "golden generation failed at stage=${stage}"
fi

stage="run"
remote_run_cmd=$(cat <<EOF
cd "${TESTCASE_DIR}" && \
export ASCEND_HOME_PATH="${ASCEND_HOME_PATH}" && \
if [ -f "\$ASCEND_HOME_PATH/set_env.sh" ]; then source "\$ASCEND_HOME_PATH/set_env.sh" >/dev/null 2>&1; fi && \
LD_LIBRARY_PATH="${REPACK_DIR}:${BUILD_DIR}:\$ASCEND_HOME_PATH/lib64:\${LD_LIBRARY_PATH:-}" ./build/cmp
EOF
)
set +e
run_remote "${remote_run_cmd}"
rc=$?
set -e
if [[ ${rc} -ne 0 ]]; then
  record_result "FAIL" "${stage}" "runner_exit=${rc}"
  die "npu run failed at stage=${stage}"
fi

remote_ldd_cmd=$(cat <<EOF
cd "${TESTCASE_DIR}" && \
export ASCEND_HOME_PATH="${ASCEND_HOME_PATH}" && \
if [ -f "\$ASCEND_HOME_PATH/set_env.sh" ]; then source "\$ASCEND_HOME_PATH/set_env.sh" >/dev/null 2>&1; fi && \
LD_LIBRARY_PATH="${REPACK_DIR}:${BUILD_DIR}:\$ASCEND_HOME_PATH/lib64:\${LD_LIBRARY_PATH:-}" ldd ./build/cmp | grep libcmp_kernel.so
EOF
)
set +e
ldd_output="$(run_remote "${remote_ldd_cmd}")"
rc=$?
set -e
if [[ ${rc} -ne 0 ]]; then
  record_result "FAIL" "${stage}" "ldd_check_exit=${rc}"
  die "ldd check failed at stage=${stage}"
fi
if [[ "${ldd_output}" != *"${REPACK_SO}"* ]]; then
  record_result "FAIL" "${stage}" "unexpected_libcmp_kernel=${ldd_output}"
  die "cmp binary did not load llvm ir kernel so from ${REPACK_SO}"
fi

stage="compare"
set +e
compare_output="$(cd "${TESTCASE_DIR}" && COMPARE_STRICT=1 python3 ./compare.py 2>&1)"
rc=$?
set -e
if [[ ${rc} -ne 0 ]]; then
  record_result "FAIL" "${stage}" "$(printf '%s' "${compare_output}" | tail -n 1)"
  echo "${compare_output}" >&2
  die "compare failed at stage=${stage}"
fi

record_result "OK" "all" "seed=${SEED}"
log "Cmp host npu validation passed"
log "Loaded libcmp_kernel: ${ldd_output}"
printf "%s\n" "${compare_output}"
