#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

WORK_SPACE="${WORK_SPACE:-}"
ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-}"
PTO_ISA_ROOT="${PTO_ISA_ROOT:-}"
PTOAS_BIN="${PTOAS_BIN:-}"
PTOAS_FLAGS="${PTOAS_FLAGS:-}"
A5VM_FLAGS="${A5VM_FLAGS:-}"
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

record_result() {
  local status="$1"
  local stage="$2"
  local info="$3"
  printf "%s/%s\t%s\t%s\t%s\n" "${SAMPLE_NAME}" "${TESTCASE_NAME}" "${status}" "${stage}" "${info}" >> "${RESULTS_TSV}"
}

run_remote() {
  local cmd="$1"
  if [[ "${HOST_RUNNER}" == "ssh root@localhost" ]]; then
    ssh -o StrictHostKeyChecking=no root@localhost "${cmd}"
  else
    eval "${HOST_RUNNER} ${cmd@Q}"
  fi
}

[[ -n "${WORK_SPACE}" ]] || die "WORK_SPACE is required"
[[ -n "${ASCEND_HOME_PATH}" ]] || die "ASCEND_HOME_PATH is required"
[[ -n "${PTO_ISA_ROOT}" ]] || die "PTO_ISA_ROOT is required"
[[ -n "${PTOAS_BIN}" ]] || die "PTOAS_BIN is required"
[[ -n "${SAMPLE_NAME}" ]] || die "SAMPLE_NAME is required"
[[ -n "${TESTCASE_NAME}" ]] || die "TESTCASE_NAME is required"
[[ -n "${RESULTS_TSV}" ]] || die "RESULTS_TSV is required"

SAMPLE_CPP="${WORK_SPACE}/emitc/${SAMPLE_NAME}/${TESTCASE_NAME}-pto.cpp"
TESTCASE_DIR="${WORK_SPACE}/testcase/${SAMPLE_NAME}/${TESTCASE_NAME}"
BUILD_DIR="${TESTCASE_DIR}/build"
REPACK_DIR="${WORK_SPACE}/llvm_ir_kernel_so/${SAMPLE_NAME}/${TESTCASE_NAME}/repack"
REPACK_SO="${REPACK_DIR}/lib${TESTCASE_NAME}_kernel.so"
SAMPLE_GOLDEN_PY="${ROOT_DIR}/test/samples/${SAMPLE_NAME}/npu_validation/${TESTCASE_NAME}/golden.py"

stage="generate"
[[ -f "${SAMPLE_CPP}" ]] || {
  record_result "FAIL" "${stage}" "missing_sample_cpp"
  die "missing generated sample cpp: ${SAMPLE_CPP}"
}

set +e
python3 "${ROOT_DIR}/test/npu_validation/scripts/generate_testcase.py" \
  --input "${SAMPLE_CPP}" \
  --testcase "${TESTCASE_NAME}" \
  --output-root "${WORK_SPACE}/testcase" \
  --run-mode sim \
  --soc-version "${SOC_VERSION}" \
  --aicore-arch "${AICORE_ARCH}"
rc=$?
set -e
if [[ ${rc} -ne 0 ]]; then
  record_result "FAIL" "${stage}" "generate_testcase_exit=${rc}"
  die "generate_testcase failed at stage=${stage} case=${SAMPLE_NAME}/${TESTCASE_NAME}"
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
  die "cmake configure failed at stage=${stage} case=${SAMPLE_NAME}/${TESTCASE_NAME}"
fi

set +e
cmake --build "${BUILD_DIR}" --parallel
rc=$?
set -e
if [[ ${rc} -ne 0 ]]; then
  record_result "FAIL" "${stage}" "cmake_build_exit=${rc}"
  die "cmake build failed at stage=${stage} case=${SAMPLE_NAME}/${TESTCASE_NAME}"
fi

stage="build_so"
if [[ "${KERNEL_MODE}" == "llvm" ]]; then
  set +e
  WORK_SPACE="${WORK_SPACE}" \
  ASCEND_HOME_PATH="${ASCEND_HOME_PATH}" \
  PTO_ISA_ROOT="${PTO_ISA_ROOT}" \
  PTOAS_BIN="${PTOAS_BIN}" \
  PTOAS_FLAGS="${PTOAS_FLAGS}" \
  A5VM_FLAGS="${A5VM_FLAGS}" \
  SAMPLE_NAME="${SAMPLE_NAME}" \
  TESTCASE_NAME="${TESTCASE_NAME}" \
  SOC_VERSION="${SOC_VERSION}" \
  AICORE_ARCH="${AICORE_ARCH}" \
    "${ROOT_DIR}/test/npu_validation/scripts/build_llvm_ir_kernel_so.sh"
  rc=$?
  set -e
  if [[ ${rc} -ne 0 ]]; then
    record_result "FAIL" "${stage}" "build_llvm_ir_kernel_so_exit=${rc}"
    die "llvm ir kernel so build failed at stage=${stage} case=${SAMPLE_NAME}/${TESTCASE_NAME}"
  fi
  [[ -f "${REPACK_SO}" ]] || {
    record_result "FAIL" "${stage}" "missing_repack_so"
    die "missing llvm ir kernel so: ${REPACK_SO}"
  }
elif [[ "${KERNEL_MODE}" == "emitc" ]]; then
  REPACK_SO="${BUILD_DIR}/lib${TESTCASE_NAME}_kernel.so"
  [[ -f "${REPACK_SO}" ]] || {
    record_result "FAIL" "${stage}" "missing_emitc_kernel_so"
    die "missing emitc kernel so: ${REPACK_SO}"
  }
else
  record_result "FAIL" "${stage}" "unsupported_kernel_mode=${KERNEL_MODE}"
  die "Unsupported KERNEL_MODE=${KERNEL_MODE}. Expected llvm|emitc."
fi

stage="generate"
case "${GOLDEN_MODE}" in
  py)
    [[ -f "${SAMPLE_GOLDEN_PY}" ]] || {
      record_result "FAIL" "${stage}" "missing_sample_golden"
      die "missing sample golden script: ${SAMPLE_GOLDEN_PY}"
    }
    set +e
    python3 "${SAMPLE_GOLDEN_PY}" \
      --output-dir "${TESTCASE_DIR}" \
      --seed "${SEED}"
    rc=$?
    set -e
    if [[ ${rc} -ne 0 ]]; then
      record_result "FAIL" "${stage}" "golden_script_exit=${rc}"
      die "golden generation failed at stage=${stage} case=${SAMPLE_NAME}/${TESTCASE_NAME}"
    fi
    ;;
  skip)
    log "Skipping golden generation and compare because GOLDEN_MODE=skip (${SAMPLE_NAME}/${TESTCASE_NAME})"
    ;;
  *)
    record_result "FAIL" "${stage}" "unsupported_golden_mode=${GOLDEN_MODE}"
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
  record_result "FAIL" "${stage}" "runner_exit=${rc}"
  die "npu run failed at stage=${stage} case=${SAMPLE_NAME}/${TESTCASE_NAME}"
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
  record_result "FAIL" "${stage}" "ldd_check_exit=${rc}"
  die "ldd check failed at stage=${stage} case=${SAMPLE_NAME}/${TESTCASE_NAME}"
fi
if [[ "${ldd_output}" != *"${REPACK_SO}"* ]]; then
  record_result "FAIL" "${stage}" "unexpected_kernel_so=${ldd_output}"
  die "${SAMPLE_NAME}/${TESTCASE_NAME} binary did not load kernel so from ${REPACK_SO}"
fi

stage="compare"
if [[ "${GOLDEN_MODE}" == "skip" ]]; then
  record_result "OK" "all" "seed=${SEED},compare_skipped"
  log "${SAMPLE_NAME}/${TESTCASE_NAME} host npu validation ran with compare skipped"
  log "Loaded lib${TESTCASE_NAME}_kernel: ${ldd_output}"
  exit 0
fi

set +e
compare_output="$(cd "${TESTCASE_DIR}" && COMPARE_STRICT=1 python3 ./compare.py 2>&1)"
rc=$?
set -e
if [[ ${rc} -ne 0 ]]; then
  record_result "FAIL" "${stage}" "$(printf '%s' "${compare_output}" | tail -n 1)"
  echo "${compare_output}" >&2
  die "compare failed at stage=${stage} case=${SAMPLE_NAME}/${TESTCASE_NAME}"
fi

record_result "OK" "all" "seed=${SEED}"
log "${SAMPLE_NAME}/${TESTCASE_NAME} host npu validation passed"
log "Loaded lib${TESTCASE_NAME}_kernel: ${ldd_output}"
printf "%s\n" "${compare_output}"
