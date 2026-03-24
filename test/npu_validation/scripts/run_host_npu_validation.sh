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
GOLDEN_MODE="${GOLDEN_MODE:-py}"  # py|skip
KERNEL_MODE="${KERNEL_MODE:-llvm}"  # llvm|emitc
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

configure_case() {
  local case_name="$1"
  case "${case_name}" in
    Cmp)
      SAMPLE_NAME="Cmp"
      TESTCASE_NAME="cmp"
      SAMPLE_GOLDEN_PY="${ROOT_DIR}/test/samples/Cmp/npu_validation/golden.py"
      GOLDEN_EXTRA_ARGS=(--src-elem-bytes 4)
      ;;
    Abs)
      SAMPLE_NAME="Abs"
      TESTCASE_NAME="abs"
      SAMPLE_GOLDEN_PY="${ROOT_DIR}/test/samples/Abs/npu_validation/golden.py"
      GOLDEN_EXTRA_ARGS=()
      ;;
    VectorAddition)
      SAMPLE_NAME="VectorAddition"
      TESTCASE_NAME="vadd_pto_ir"
      SAMPLE_GOLDEN_PY="${ROOT_DIR}/test/samples/VectorAddition/npu_validation/golden.py"
      GOLDEN_EXTRA_ARGS=()
      ;;
    Max)
      SAMPLE_NAME="Max"
      TESTCASE_NAME="max"
      SAMPLE_GOLDEN_PY="${ROOT_DIR}/test/samples/Max/npu_validation/golden.py"
      GOLDEN_EXTRA_ARGS=()
      ;;
    Sub)
      SAMPLE_NAME="Sub"
      TESTCASE_NAME="sub"
      SAMPLE_GOLDEN_PY="${ROOT_DIR}/test/samples/Sub/npu_validation/golden.py"
      GOLDEN_EXTRA_ARGS=()
      ;;
    Exp)
      SAMPLE_NAME="Exp"
      TESTCASE_NAME="exp"
      SAMPLE_GOLDEN_PY="${ROOT_DIR}/test/samples/Exp/npu_validation/golden.py"
      GOLDEN_EXTRA_ARGS=()
      ;;
    Mul)
      SAMPLE_NAME="Mul"
      TESTCASE_NAME="mul"
      SAMPLE_GOLDEN_PY="${ROOT_DIR}/test/samples/Mul/npu_validation/golden.py"
      GOLDEN_EXTRA_ARGS=()
      ;;
    Expands)
      SAMPLE_NAME="Expands"
      TESTCASE_NAME="expand"
      SAMPLE_GOLDEN_PY="${ROOT_DIR}/test/samples/Expands/npu_validation/golden.py"
      GOLDEN_EXTRA_ARGS=()
      ;;
    Reshape)
      SAMPLE_NAME="Reshape"
      TESTCASE_NAME="reshape"
      SAMPLE_GOLDEN_PY="${ROOT_DIR}/test/samples/Reshape/npu_validation/golden.py"
      GOLDEN_EXTRA_ARGS=()
      ;;
    Rowexpanddiv)
      SAMPLE_NAME="Rowexpanddiv"
      TESTCASE_NAME="rowexpanddiv"
      SAMPLE_GOLDEN_PY="${ROOT_DIR}/test/samples/Rowexpanddiv/npu_validation/golden.py"
      GOLDEN_EXTRA_ARGS=()
      ;;
    Rowexpandmul)
      SAMPLE_NAME="Rowexpandmul"
      TESTCASE_NAME="rowexpandmul"
      SAMPLE_GOLDEN_PY="${ROOT_DIR}/test/samples/Rowexpandmul/npu_validation/golden.py"
      GOLDEN_EXTRA_ARGS=()
      ;;
    Rowmax)
      SAMPLE_NAME="Rowmax"
      TESTCASE_NAME="rowmax"
      SAMPLE_GOLDEN_PY="${ROOT_DIR}/test/samples/Rowmax/npu_validation/golden.py"
      GOLDEN_EXTRA_ARGS=()
      ;;
    Rowsum)
      SAMPLE_NAME="Rowsum"
      TESTCASE_NAME="rowsum"
      SAMPLE_GOLDEN_PY="${ROOT_DIR}/test/samples/Rowsum/npu_validation/golden.py"
      GOLDEN_EXTRA_ARGS=()
      ;;
    *)
      return 1
      ;;
  esac

  SAMPLE_CPP="${SAMPLE_OUT_DIR}/${SAMPLE_NAME}/${TESTCASE_NAME}-pto.cpp"
  TESTCASE_DIR="${TESTCASE_ROOT}/${SAMPLE_NAME}/${TESTCASE_NAME}"
  BUILD_DIR="${TESTCASE_DIR}/build"
  REPACK_DIR="${WORK_SPACE}/llvm_ir_kernel_so/${SAMPLE_NAME}/${TESTCASE_NAME}/repack"
  REPACK_SO="${REPACK_DIR}/lib${TESTCASE_NAME}_kernel.so"
  return 0
}

record_result() {
  local case_name="$1"
  local status="$2"
  local stage="$3"
  local info="$4"
  printf "%s\t%s\t%s\t%s\n" "${case_name}" "${status}" "${stage}" "${info}" >> "${RESULTS_TSV}"
}

require_env "WORK_SPACE" "${WORK_SPACE}" "/tmp/ptoas-npu-validation-llvm"
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

SUPPORTED_CASES=(Cmp Abs VectorAddition Max Sub Exp Mul Expands Reshape Rowexpanddiv Rowexpandmul Rowmax Rowsum)
if [[ -n "${RUN_ONLY_CASES_NORM}" ]]; then
  for item in ${RUN_ONLY_CASES_NORM//,/ }; do
    case "${item}" in
      Cmp|Abs|VectorAddition|Max|Sub|Exp|Mul|Expands|Reshape|Rowexpanddiv|Rowexpandmul|Rowmax|Rowsum) ;;
      *) die "Unsupported testcase: ${item}." ;;
    esac
  done
fi

mkdir -p "${WORK_SPACE}"
RESULTS_TSV="${RESULTS_TSV:-${WORK_SPACE}/host_npu_validation_results.tsv}"
printf "testcase\tstatus\tstage\tinfo\n" > "${RESULTS_TSV}"

SAMPLE_OUT_DIR="${WORK_SPACE}/emitc"
TESTCASE_ROOT="${WORK_SPACE}/testcase"

log "=== Host NPU Validation ==="
log "WORK_SPACE=${WORK_SPACE}"
log "ASCEND_HOME_PATH=${ASCEND_HOME_PATH}"
log "PTO_ISA_ROOT=${PTO_ISA_ROOT}"
log "PTOAS_BIN=${PTOAS_BIN}"
log "HOST_RUNNER=${HOST_RUNNER}"
log "GOLDEN_MODE=${GOLDEN_MODE}"
log "KERNEL_MODE=${KERNEL_MODE}"
log "RESULTS_TSV=${RESULTS_TSV}"

for case_name in "${SUPPORTED_CASES[@]}"; do
  if [[ -n "${RUN_ONLY_CASES_NORM}" ]] && ! list_contains "${RUN_ONLY_CASES_NORM}" "${case_name}"; then
    continue
  fi
  if [[ -n "${SKIP_CASES_NORM}" ]] && list_contains "${SKIP_CASES_NORM}" "${case_name}"; then
    log "${case_name} is listed in SKIP_CASES; skipping."
    continue
  fi

  configure_case "${case_name}" || die "failed to configure testcase ${case_name}"

  log "=== Case ${case_name} ==="

  stage="generate"
  set +e
  PTOAS_BIN="${PTOAS_BIN}" \
  PTOAS_OUT_DIR="${SAMPLE_OUT_DIR}" \
    "${ROOT_DIR}/test/samples/runop.sh" -t "${SAMPLE_NAME}"
  rc=$?
  set -e
  if [[ ${rc} -ne 0 ]]; then
    record_result "${case_name}" "FAIL" "${stage}" "runop_exit=${rc}"
    die "sample export failed at stage=${stage} case=${case_name}"
  fi
  [[ -f "${SAMPLE_CPP}" ]] || {
    record_result "${case_name}" "FAIL" "${stage}" "missing_sample_cpp"
    die "missing generated sample cpp: ${SAMPLE_CPP}"
  }

  set +e
  python3 "${ROOT_DIR}/test/npu_validation/scripts/generate_testcase.py" \
    --input "${SAMPLE_CPP}" \
    --testcase "${TESTCASE_NAME}" \
    --output-root "${TESTCASE_ROOT}" \
    --run-mode sim \
    --soc-version "${SOC_VERSION}" \
    --aicore-arch "${AICORE_ARCH}"
  rc=$?
  set -e
  if [[ ${rc} -ne 0 ]]; then
    record_result "${case_name}" "FAIL" "${stage}" "generate_testcase_exit=${rc}"
    die "generate_testcase failed at stage=${stage} case=${case_name}"
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
    record_result "${case_name}" "FAIL" "${stage}" "cmake_configure_exit=${rc}"
    die "cmake configure failed at stage=${stage} case=${case_name}"
  fi

  set +e
  cmake --build "${BUILD_DIR}" --parallel
  rc=$?
  set -e
  if [[ ${rc} -ne 0 ]]; then
    record_result "${case_name}" "FAIL" "${stage}" "cmake_build_exit=${rc}"
    die "cmake build failed at stage=${stage} case=${case_name}"
  fi

  stage="build_so"
  if [[ "${KERNEL_MODE}" == "llvm" ]]; then
    set +e
    WORK_SPACE="${WORK_SPACE}" \
    ASCEND_HOME_PATH="${ASCEND_HOME_PATH}" \
    PTO_ISA_ROOT="${PTO_ISA_ROOT}" \
    PTOAS_BIN="${PTOAS_BIN}" \
    SAMPLE_NAME="${SAMPLE_NAME}" \
    TESTCASE_NAME="${TESTCASE_NAME}" \
    SOC_VERSION="${SOC_VERSION}" \
    AICORE_ARCH="${AICORE_ARCH}" \
      "${ROOT_DIR}/test/npu_validation/scripts/build_llvm_ir_kernel_so.sh"
    rc=$?
    set -e
    if [[ ${rc} -ne 0 ]]; then
      record_result "${case_name}" "FAIL" "${stage}" "build_llvm_ir_kernel_so_exit=${rc}"
      die "llvm ir kernel so build failed at stage=${stage} case=${case_name}"
    fi
    [[ -f "${REPACK_SO}" ]] || {
      record_result "${case_name}" "FAIL" "${stage}" "missing_repack_so"
      die "missing llvm ir kernel so: ${REPACK_SO}"
    }
  elif [[ "${KERNEL_MODE}" == "emitc" ]]; then
    REPACK_SO="${BUILD_DIR}/lib${TESTCASE_NAME}_kernel.so"
    [[ -f "${REPACK_SO}" ]] || {
      record_result "${case_name}" "FAIL" "${stage}" "missing_emitc_kernel_so"
      die "missing emitc kernel so: ${REPACK_SO}"
    }
  else
    die "Unsupported KERNEL_MODE=${KERNEL_MODE}. Expected llvm|emitc."
  fi

  stage="generate"
  case "${GOLDEN_MODE}" in
    py)
      [[ -f "${SAMPLE_GOLDEN_PY}" ]] || die "missing sample golden script: ${SAMPLE_GOLDEN_PY}"
      set +e
      python3 "${SAMPLE_GOLDEN_PY}" \
        --output-dir "${TESTCASE_DIR}" \
        --seed "${SEED}" \
        "${GOLDEN_EXTRA_ARGS[@]}"
      rc=$?
      set -e
      if [[ ${rc} -ne 0 ]]; then
        record_result "${case_name}" "FAIL" "${stage}" "golden_script_exit=${rc}"
        die "golden generation failed at stage=${stage} case=${case_name}"
      fi
      ;;
    skip)
      log "Skipping golden generation and compare because GOLDEN_MODE=skip (${case_name})"
      ;;
    *)
      die "Unsupported GOLDEN_MODE=${GOLDEN_MODE}. Expected py|skip."
      ;;
  esac

  stage="run"
  remote_run_cmd=$(cat <<EOF
cd "${TESTCASE_DIR}" && \
export ASCEND_HOME_PATH="${ASCEND_HOME_PATH}" && \
if [ -f "\$ASCEND_HOME_PATH/set_env.sh" ]; then source "\$ASCEND_HOME_PATH/set_env.sh" >/dev/null 2>&1; fi && \
LD_LIBRARY_PATH="${REPACK_DIR}:${BUILD_DIR}:\$ASCEND_HOME_PATH/lib64:\${LD_LIBRARY_PATH:-}" ./build/${TESTCASE_NAME}
EOF
)
  set +e
  run_remote "${remote_run_cmd}"
  rc=$?
  set -e
  if [[ ${rc} -ne 0 ]]; then
    record_result "${case_name}" "FAIL" "${stage}" "runner_exit=${rc}"
    die "npu run failed at stage=${stage} case=${case_name}"
  fi

  remote_ldd_cmd=$(cat <<EOF
cd "${TESTCASE_DIR}" && \
export ASCEND_HOME_PATH="${ASCEND_HOME_PATH}" && \
if [ -f "\$ASCEND_HOME_PATH/set_env.sh" ]; then source "\$ASCEND_HOME_PATH/set_env.sh" >/dev/null 2>&1; fi && \
LD_LIBRARY_PATH="${REPACK_DIR}:${BUILD_DIR}:\$ASCEND_HOME_PATH/lib64:\${LD_LIBRARY_PATH:-}" ldd ./build/${TESTCASE_NAME} | grep lib${TESTCASE_NAME}_kernel.so
EOF
)
  set +e
  ldd_output="$(run_remote "${remote_ldd_cmd}")"
  rc=$?
  set -e
  if [[ ${rc} -ne 0 ]]; then
    record_result "${case_name}" "FAIL" "${stage}" "ldd_check_exit=${rc}"
    die "ldd check failed at stage=${stage} case=${case_name}"
  fi
  if [[ "${ldd_output}" != *"${REPACK_SO}"* ]]; then
    record_result "${case_name}" "FAIL" "${stage}" "unexpected_kernel_so=${ldd_output}"
    die "${case_name} binary did not load llvm ir kernel so from ${REPACK_SO}"
  fi

  stage="compare"
  if [[ "${GOLDEN_MODE}" == "skip" ]]; then
    record_result "${case_name}" "OK" "all" "seed=${SEED},compare_skipped"
    log "${case_name} host npu validation ran with compare skipped"
    log "Loaded lib${TESTCASE_NAME}_kernel: ${ldd_output}"
    continue
  fi

  set +e
  compare_output="$(cd "${TESTCASE_DIR}" && COMPARE_STRICT=1 python3 ./compare.py 2>&1)"
  rc=$?
  set -e
  if [[ ${rc} -ne 0 ]]; then
    record_result "${case_name}" "FAIL" "${stage}" "$(printf '%s' "${compare_output}" | tail -n 1)"
    echo "${compare_output}" >&2
    die "compare failed at stage=${stage} case=${case_name}"
  fi

  record_result "${case_name}" "OK" "all" "seed=${SEED}"
  log "${case_name} host npu validation passed"
  log "Loaded lib${TESTCASE_NAME}_kernel: ${ldd_output}"
  printf "%s\n" "${compare_output}"
done
