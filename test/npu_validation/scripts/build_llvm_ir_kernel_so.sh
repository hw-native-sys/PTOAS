#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

WORK_SPACE="${WORK_SPACE:-}"
ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-}"
PTO_ISA_ROOT="${PTO_ISA_ROOT:-}"
PTOAS_BIN="${PTOAS_BIN:-${ROOT_DIR}/build/tools/ptoas/ptoas}"
PTOAS_FLAGS="${PTOAS_FLAGS:---pto-arch a5}"
A5VM_FLAGS="${A5VM_FLAGS:---pto-backend=a5vm --a5vm-emit-hivm-llvm}"
SAMPLE_NAME="${SAMPLE_NAME:-}"
TESTCASE_NAME="${TESTCASE_NAME:-}"

SOC_VERSION="${SOC_VERSION:-A5}"
AICORE_ARCH="${AICORE_ARCH:-dav-c310-vec}"
ARTIFACTS_ROOT="${ARTIFACTS_ROOT:-}"
LLVM_OUT_ROOT="${LLVM_OUT_ROOT:-}"
FAIL_FAST="${FAIL_FAST:-0}"
RESULTS_TSV="${RESULTS_TSV:-}"
MODULE_ID="${MODULE_ID:-a5d60abf67864aa0}"

if [[ ! -x "${PTOAS_BIN}" ]]; then
  echo "ERROR: PTOAS_BIN not found: ${PTOAS_BIN}" >&2
  exit 1
fi
if [[ -z "${ASCEND_HOME_PATH}" ]]; then
  echo "ERROR: ASCEND_HOME_PATH is not set." >&2
  exit 1
fi
if [[ -z "${PTO_ISA_ROOT}" ]]; then
  echo "ERROR: PTO_ISA_ROOT is not set." >&2
  exit 1
fi
if [[ -z "${SAMPLE_NAME}" ]]; then
  echo "ERROR: SAMPLE_NAME is required." >&2
  exit 1
fi

set +u
source "${ROOT_DIR}/env.sh" >/dev/null 2>&1
set -u

if [[ -f "${ASCEND_HOME_PATH}/set_env.sh" ]]; then
  source "${ASCEND_HOME_PATH}/set_env.sh" >/dev/null 2>&1
fi

BISHENG_BIN="${BISHENG_BIN:-${ASCEND_HOME_PATH}/bin/bisheng}"
BISHENG_CC1_BIN="${BISHENG_CC1_BIN:-${ASCEND_HOME_PATH}/tools/bisheng_compiler/bin/bisheng}"
CCE_LD_BIN="${CCE_LD_BIN:-${ASCEND_HOME_PATH}/bin/cce-ld}"
LD_LLD_BIN="${LD_LLD_BIN:-${ASCEND_HOME_PATH}/bin/ld.lld}"
CLANG_RESOURCE_DIR="${CLANG_RESOURCE_DIR:-${ASCEND_HOME_PATH}/tools/bisheng_compiler/lib/clang/15.0.5}"
CCE_STUB_DIR="${CCE_STUB_DIR:-${CLANG_RESOURCE_DIR}/include/cce_stub}"

if ! command -v "${BISHENG_BIN}" >/dev/null 2>&1; then
  echo "ERROR: bisheng not found: ${BISHENG_BIN}" >&2
  exit 1
fi

readarray -t BISHENG_SYSTEM_INCLUDES < <(
  "${BISHENG_BIN}" -xc++ -E -v - </dev/null 2>&1 |
    awk '
      /#include <...> search starts here:/ {capture=1; next}
      /End of search list\./ {capture=0}
      capture && $0 ~ /^ / {sub(/^ +/, "", $0); print}
    '
)

if [[ "${#BISHENG_SYSTEM_INCLUDES[@]}" -eq 0 ]]; then
  echo "ERROR: failed to discover bisheng system include directories." >&2
  exit 1
fi

CC1_INCLUDE_FLAGS=()
for inc in "${BISHENG_SYSTEM_INCLUDES[@]}"; do
  if [[ "${inc}" == */include/c++/* || "${inc}" == */backward ]]; then
    CC1_INCLUDE_FLAGS+=(-internal-isystem "${inc}")
  elif [[ "${inc}" == "/usr/include" ]]; then
    CC1_INCLUDE_FLAGS+=(-internal-externc-isystem "${inc}")
  else
    CC1_INCLUDE_FLAGS+=(-internal-isystem "${inc}")
  fi
done

log() {
  echo "[$(date +'%F %T')] $*"
}

case_result() {
  local testcase="$1"
  local status="$2"
  local stage="$3"
  local info="$4"
  printf "%s\t%s\t%s\t%s\t%s\n" "${SAMPLE_NAME}" "${testcase}" "${status}" "${stage}" "${info}" >> "${RESULTS_TSV}"
}

should_fail_fast() {
  [[ "${FAIL_FAST}" == "1" ]]
}

if [[ -z "${WORK_SPACE}" ]]; then
  WORK_SPACE="$(mktemp -d "${TMPDIR:-/tmp}/llvm-ir-kernel-so-workspace.XXXXXX")"
fi
mkdir -p "${WORK_SPACE}"

TESTCASE_ROOT="${WORK_SPACE}/testcase/${SAMPLE_NAME}"
ARTIFACTS_ROOT="${ARTIFACTS_ROOT:-${WORK_SPACE}/llvm_ir_kernel_so/${SAMPLE_NAME}}"
LLVM_OUT_ROOT="${LLVM_OUT_ROOT:-${ARTIFACTS_ROOT}/llvm}"
RESULTS_TSV="${RESULTS_TSV:-${ARTIFACTS_ROOT}/build_results.tsv}"
RUNOP_LOG="${ARTIFACTS_ROOT}/runop.log"

mkdir -p "${ARTIFACTS_ROOT}"
printf "sample\ttestcase\tstatus\tstage\tinfo\n" > "${RESULTS_TSV}"

if [[ ! -d "${TESTCASE_ROOT}" ]]; then
  echo "ERROR: testcase root missing: ${TESTCASE_ROOT}" >&2
  exit 1
fi

declare -a TESTCASES=()
if [[ -n "${TESTCASE_NAME}" ]]; then
  TESTCASES=("${TESTCASE_NAME}")
else
  while IFS= read -r -d '' dir; do
    TESTCASES+=("$(basename "${dir}")")
  done < <(find "${TESTCASE_ROOT}" -mindepth 1 -maxdepth 1 -type d -print0 | sort -z)
fi

if [[ "${#TESTCASES[@]}" -eq 0 ]]; then
  echo "ERROR: no testcase directories found under ${TESTCASE_ROOT}" >&2
  exit 1
fi

log "step 1/4: export ${SAMPLE_NAME} as textual llvm ir"
mkdir -p "${LLVM_OUT_ROOT}"
set +e
PTOAS_BIN="${PTOAS_BIN}" \
PTOAS_OUT_DIR="${LLVM_OUT_ROOT}" \
PTOAS_FLAGS="${PTOAS_FLAGS} ${A5VM_FLAGS}" \
  "${ROOT_DIR}/test/samples/runop.sh" -t "${SAMPLE_NAME}" >"${RUNOP_LOG}" 2>&1
RUNOP_RC=$?
set -e
if [[ ${RUNOP_RC} -ne 0 ]]; then
  log "runop.sh exited with rc=${RUNOP_RC}; continuing if requested testcase IR was still generated"
  tail -n 80 "${RUNOP_LOG}" || true
fi

status=0
ok_count=0
fail_count=0

build_one() {
  local testcase="$1"
  local testcase_dir="${TESTCASE_ROOT}/${testcase}"
  local build_dir="${testcase_dir}/build"
  local kernel_src="${testcase_dir}/${testcase}_kernel.cpp"
  local launch_obj="${build_dir}/CMakeFiles/${testcase}_kernel.dir/launch.cpp.o"
  local llvm_ir_cpp="${LLVM_OUT_ROOT}/${SAMPLE_NAME}/${testcase}-pto.cpp"
  local llvm_device_obj="${LLVM_OUT_ROOT}/${SAMPLE_NAME}/${testcase}-pto.o"
  local case_artifacts_dir="${ARTIFACTS_ROOT}/${testcase}"
  local repack_dir="${case_artifacts_dir}/repack"
  local host_stub_obj="${repack_dir}/${testcase}_kernel_host_from_llvm.o"
  local repack_obj="${repack_dir}/${testcase}_kernel.cpp.o"
  local repack_so="${repack_dir}/lib${testcase}_kernel.so"
  local rc

  if [[ ! -d "${testcase_dir}" ]]; then
    case_result "${testcase}" "FAIL" "discover" "missing_testcase_dir"
    return 1
  fi
  if [[ ! -f "${kernel_src}" ]]; then
    case_result "${testcase}" "FAIL" "discover" "missing_kernel_src"
    return 1
  fi
  if [[ ! -f "${launch_obj}" ]]; then
    case_result "${testcase}" "FAIL" "discover" "missing_launch_obj"
    return 1
  fi

  if [[ ! -f "${llvm_ir_cpp}" ]]; then
    local info="missing_llvm_ir"
    if [[ ${RUNOP_RC} -ne 0 ]]; then
      info="runop_exit=${RUNOP_RC}"
      echo "---- runop log (${SAMPLE_NAME}) ----" >&2
      cat "${RUNOP_LOG}" >&2 || true
      echo "---- end runop log (${SAMPLE_NAME}) ----" >&2
    fi
    case_result "${testcase}" "FAIL" "export" "${info}"
    return 1
  fi

  log "step 2/4: compile llvm ir to aicore device object (${SAMPLE_NAME}/${testcase})"
  set +e
  "${BISHENG_BIN}" \
    --target=hiipu64-hisilicon-cce \
    -march="${AICORE_ARCH}" \
    --cce-aicore-arch="${AICORE_ARCH}" \
    --cce-aicore-only \
    -c -x ir "${llvm_ir_cpp}" \
    -o "${llvm_device_obj}"
  rc=$?
  set -e
  if [[ ${rc} -ne 0 ]]; then
    case_result "${testcase}" "FAIL" "compile_ir" "bisheng_exit=${rc}"
    return 1
  fi

  log "step 3/4: rebuild fatobj object from llvm device object (${SAMPLE_NAME}/${testcase})"
  mkdir -p "${repack_dir}"
  set +e
  "${BISHENG_CC1_BIN}" -cc1 \
    -triple aarch64-unknown-linux-gnu \
    -fcce-aicpu-legacy-launch \
    -fcce-is-host \
    -cce-launch-with-flagv2-impl \
    -fcce-aicore-arch "${AICORE_ARCH}" \
    -fcce-fatobj-compile \
    -emit-obj \
    --mrelax-relocations \
    -disable-free \
    -clear-ast-before-backend \
    -disable-llvm-verifier \
    -discard-value-names \
    -main-file-name "${testcase}_kernel.cpp" \
    -mrelocation-model pic \
    -pic-level 2 \
    -fhalf-no-semantic-interposition \
    -fenable-matrix \
    -mllvm -enable-matrix \
    -mframe-pointer=non-leaf \
    -fmath-errno \
    -ffp-contract=on \
    -fno-rounding-math \
    -mconstructor-aliases \
    -funwind-tables=2 \
    -target-cpu generic \
    -target-feature +neon \
    -target-feature +v8a \
    -target-abi aapcs \
    -fallow-half-arguments-and-returns \
    -mllvm -treat-scalable-fixed-error-as-warning \
    -fcoverage-compilation-dir="${ROOT_DIR}" \
    -resource-dir "${CLANG_RESOURCE_DIR}" \
    -include __clang_cce_runtime_wrapper.h \
    -D "${testcase}_kernel_EXPORTS" \
    -I "${PTO_ISA_ROOT}/include" \
    -I "${ASCEND_HOME_PATH}/include" \
    -I "${ASCEND_HOME_PATH}/pkg_inc" \
    -I "${ASCEND_HOME_PATH}/pkg_inc/profiling" \
    -I "${ASCEND_HOME_PATH}/pkg_inc/runtime/runtime" \
    -D _FORTIFY_SOURCE=2 \
    -D REGISTER_BASE \
    "${CC1_INCLUDE_FLAGS[@]}" \
    -O2 \
    -Wno-macro-redefined \
    -Wno-ignored-attributes \
    -std=c++17 \
    -fdeprecated-macro \
    -fdebug-compilation-dir="${ROOT_DIR}" \
    -ferror-limit 19 \
    -stack-protector 2 \
    -fno-signed-char \
    -fgnuc-version=4.2.1 \
    -fcxx-exceptions \
    -fexceptions \
    -vectorize-loops \
    -vectorize-slp \
    -mllvm -cce-aicore-stack-size=0x8000 \
    -mllvm -cce-aicore-function-stack-size=0x8000 \
    -mllvm -cce-aicore-record-overflow=true \
    -mllvm -cce-aicore-addr-transform \
    -mllvm -cce-aicore-dcci-insert-for-scalar=false \
    -fcce-include-aibinary "${llvm_device_obj}" \
    -fcce-device-module-id "${MODULE_ID}" \
    -target-feature +outline-atomics \
    -faddrsig \
    -D__GCC_HAVE_DWARF2_CFI_ASM=1 \
    -o "${host_stub_obj}" \
    -x cce "${kernel_src}"
  rc=$?
  set -e
  if [[ ${rc} -ne 0 ]]; then
    case_result "${testcase}" "FAIL" "repack" "host_stub_exit=${rc}"
    return 1
  fi

  set +e
  "${CCE_LD_BIN}" \
    "${LD_LLD_BIN}" \
    -x \
    -cce-lite-bin-module-id "${MODULE_ID}" \
    -cce-aicore-arch="${AICORE_ARCH}" \
    -r \
    -o "${repack_obj}" \
    -cce-stub-dir "${CCE_STUB_DIR}" \
    -cce-install-dir "$(dirname "${BISHENG_CC1_BIN}")" \
    -cce-inputs-number 1 \
    "${host_stub_obj}"
  rc=$?
  set -e
  if [[ ${rc} -ne 0 ]]; then
    case_result "${testcase}" "FAIL" "repack" "cce_ld_exit=${rc}"
    return 1
  fi

  log "step 4/4: link replacement lib${testcase}_kernel.so (${SAMPLE_NAME}/${testcase})"
  set +e
  (
    cd "${repack_dir}"
    "${BISHENG_BIN}" \
      -fPIC -s -Wl,-z,relro -Wl,-z,now --cce-fatobj-link \
      -shared -Wl,-soname,"lib${testcase}_kernel.so" \
      -o "${repack_so}" \
      "${repack_obj}" \
      "${launch_obj}"
  )
  rc=$?
  set -e
  if [[ ${rc} -ne 0 ]]; then
    case_result "${testcase}" "FAIL" "link" "bisheng_link_exit=${rc}"
    return 1
  fi

  case_result "${testcase}" "OK" "all" "${repack_so}"
  echo
  echo "Built LLVM-path artifacts for ${SAMPLE_NAME}/${testcase}:"
  echo "  workspace:     ${WORK_SPACE}"
  echo "  testcase dir:  ${testcase_dir}"
  echo "  artifacts dir: ${case_artifacts_dir}"
  echo "  LLVM IR:       ${llvm_ir_cpp}"
  echo "  device object: ${llvm_device_obj}"
  echo "  host stub:     ${host_stub_obj}"
  echo "  fat object:    ${repack_obj}"
  echo "  shared lib:    ${repack_so}"
  echo
  echo "Use it for validation with:"
  echo "  LD_LIBRARY_PATH=${repack_dir}:${build_dir}:${ASCEND_HOME_PATH}/lib64:\${LD_LIBRARY_PATH:-} ./build/${testcase}"
  return 0
}

for testcase in "${TESTCASES[@]}"; do
  if build_one "${testcase}"; then
    ok_count=$((ok_count + 1))
  else
    status=1
    fail_count=$((fail_count + 1))
    should_fail_fast && break
  fi
done

log "=== SUMMARY ==="
log "sample=${SAMPLE_NAME} OK=${ok_count} FAIL=${fail_count}"
log "RESULTS_TSV=${RESULTS_TSV}"
exit "${status}"
