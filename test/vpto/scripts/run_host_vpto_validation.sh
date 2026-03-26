#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
VPTO_ROOT="${ROOT_DIR}/test/vpto"
CASES_ROOT="${VPTO_ROOT}/cases"

WORK_SPACE="${WORK_SPACE:-}"
ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-}"
PTOAS_BIN="${PTOAS_BIN:-${ROOT_DIR}/build/tools/ptoas/ptoas}"
PTOAS_FLAGS="${PTOAS_FLAGS:---pto-arch a5}"
A5VM_FLAGS="${A5VM_FLAGS:---pto-backend=a5vm --a5vm-emit-hivm-llvm}"
AICORE_ARCH="${AICORE_ARCH:-dav-c310-vec}"
HOST_RUNNER="${HOST_RUNNER:-ssh root@localhost}"
CASE_NAME="${CASE_NAME:-}"
MODULE_ID="${MODULE_ID:-a5d60abf67864aa0}"

log() {
  echo "[$(date +'%F %T')] $*"
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

run_remote() {
  local cmd="$1"
  if [[ "${HOST_RUNNER}" == "ssh root@localhost" ]]; then
    ssh -o StrictHostKeyChecking=no root@localhost "${cmd}"
  else
    eval "${HOST_RUNNER} ${cmd@Q}"
  fi
}

require_env() {
  local name="$1"
  local value="$2"
  if [[ -z "${value}" ]]; then
    die "${name} is required"
  fi
}

require_env "WORK_SPACE" "${WORK_SPACE}"
require_env "ASCEND_HOME_PATH" "${ASCEND_HOME_PATH}"
[[ -x "${PTOAS_BIN}" ]] || die "PTOAS_BIN is not executable: ${PTOAS_BIN}"
[[ -d "${CASES_ROOT}" ]] || die "missing cases root: ${CASES_ROOT}"

if [[ -f "${ASCEND_HOME_PATH}/set_env.sh" ]]; then
  set +u
  source "${ASCEND_HOME_PATH}/set_env.sh" >/dev/null 2>&1
  set -u
fi

BISHENG_BIN="${BISHENG_BIN:-${ASCEND_HOME_PATH}/bin/bisheng}"
BISHENG_CC1_BIN="${BISHENG_CC1_BIN:-${ASCEND_HOME_PATH}/tools/bisheng_compiler/bin/bisheng}"
CCE_LD_BIN="${CCE_LD_BIN:-${ASCEND_HOME_PATH}/bin/cce-ld}"
LD_LLD_BIN="${LD_LLD_BIN:-${ASCEND_HOME_PATH}/bin/ld.lld}"
CLANG_RESOURCE_DIR="${CLANG_RESOURCE_DIR:-${ASCEND_HOME_PATH}/tools/bisheng_compiler/lib/clang/15.0.5}"
CCE_STUB_DIR="${CCE_STUB_DIR:-${CLANG_RESOURCE_DIR}/include/cce_stub}"

command -v "${BISHENG_BIN}" >/dev/null 2>&1 || die "bisheng not found: ${BISHENG_BIN}"
command -v python3 >/dev/null 2>&1 || die "python3 not found"

readarray -t BISHENG_SYSTEM_INCLUDES < <(
  "${BISHENG_BIN}" -xc++ -E -v - </dev/null 2>&1 |
    awk '
      /#include <...> search starts here:/ {capture=1; next}
      /End of search list\./ {capture=0}
      capture && $0 ~ /^ / {sub(/^ +/, "", $0); print}
    '
)

[[ "${#BISHENG_SYSTEM_INCLUDES[@]}" -gt 0 ]] || die "failed to discover bisheng system include directories"

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

mkdir -p "${WORK_SPACE}"

discover_cases() {
  if [[ -n "${CASE_NAME}" ]]; then
    [[ -d "${CASES_ROOT}/${CASE_NAME}" ]] || die "unknown case: ${CASE_NAME}"
    printf "%s\n" "${CASE_NAME}"
    return 0
  fi

  find "${CASES_ROOT}" -mindepth 1 -maxdepth 1 -type d | sort | while read -r dir; do
    basename "${dir}"
  done
}

readarray -t CASES < <(discover_cases)
[[ "${#CASES[@]}" -gt 0 ]] || die "no cases found under ${CASES_ROOT}"

build_launch_object() {
  local case_dir="$1"
  local out_obj="$2"

  "${BISHENG_BIN}" \
    -c -fPIC -xcce -fenable-matrix --cce-aicore-enable-tl \
    -fPIC -Xhost-start -Xhost-end \
    -mllvm -cce-aicore-stack-size=0x8000 \
    -mllvm -cce-aicore-function-stack-size=0x8000 \
    -mllvm -cce-aicore-record-overflow=true \
    -mllvm -cce-aicore-addr-transform \
    -mllvm -cce-aicore-dcci-insert-for-scalar=false \
    --cce-aicore-arch="${AICORE_ARCH}" \
    -DREGISTER_BASE \
    -std=c++17 \
    -Wno-macro-redefined -Wno-ignored-attributes \
    -I "${ASCEND_HOME_PATH}/include" \
    -I "${ASCEND_HOME_PATH}/pkg_inc" \
    -I "${ASCEND_HOME_PATH}/pkg_inc/profiling" \
    -I "${ASCEND_HOME_PATH}/pkg_inc/runtime/runtime" \
    "${case_dir}/launch.cpp" \
    -o "${out_obj}"
}

build_host_stub() {
  local case_dir="$1"
  local device_obj="$2"
  local stub_obj="$3"

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
    -main-file-name "stub.cpp" \
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
    -fcce-include-aibinary "${device_obj}" \
    -fcce-device-module-id "${MODULE_ID}" \
    -target-feature +outline-atomics \
    -faddrsig \
    -D__GCC_HAVE_DWARF2_CFI_ASM=1 \
    -o "${stub_obj}" \
    -x cce "${case_dir}/stub.cpp"
}

link_kernel_so() {
  local case_name="$1"
  local host_stub_obj="$2"
  local launch_obj="$3"
  local repack_obj="$4"
  local repack_so="$5"

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

  "${BISHENG_BIN}" \
    -fPIC -s -Wl,-z,relro -Wl,-z,now --cce-fatobj-link \
    -shared -Wl,-soname,"lib${case_name}_kernel.so" \
    -o "${repack_so}" \
    "${repack_obj}" \
    "${launch_obj}"
}

build_host_executable() {
  local case_name="$1"
  local case_dir="$2"
  local out_dir="$3"

  "${BISHENG_BIN}" \
    -xc++ -include stdint.h -include stddef.h -std=c++17 \
    "${case_dir}/main.cpp" \
    -I "${case_dir}" \
    -I "${ROOT_DIR}/test/npu_validation/common" \
    -I "${ASCEND_HOME_PATH}/include" \
    -L "${out_dir}" \
    -L "${ASCEND_HOME_PATH}/lib64" \
    -Wl,-rpath,"${out_dir}" \
    -Wl,-rpath,"${ASCEND_HOME_PATH}/lib64" \
    -o "${out_dir}/${case_name}" \
    -lruntime -lstdc++ -lascendcl -lm -ltiling_api -lplatform -lc_sec -ldl -lnnopbase \
    -l"${case_name}_kernel"
}

build_one() {
  local case_name="$1"
  local case_dir="${CASES_ROOT}/${case_name}"
  local out_dir="${WORK_SPACE}/${case_name}"
  local llvm_ir="${out_dir}/${case_name}.ll"
  local device_obj="${out_dir}/${case_name}.o"
  local launch_obj="${out_dir}/launch.o"
  local host_stub_obj="${out_dir}/kernel_host_from_llvm.o"
  local repack_obj="${out_dir}/${case_name}_stub.cpp.o"
  local repack_so="${out_dir}/lib${case_name}_kernel.so"

  [[ -f "${case_dir}/kernel.pto" ]] || die "missing kernel.pto for ${case_name}"
  [[ -f "${case_dir}/stub.cpp" ]] || die "missing stub.cpp for ${case_name}"
  [[ -f "${case_dir}/main.cpp" ]] || die "missing main.cpp for ${case_name}"
  [[ -f "${case_dir}/launch.cpp" ]] || die "missing launch.cpp for ${case_name}"
  [[ -f "${case_dir}/golden.py" ]] || die "missing golden.py for ${case_name}"
  [[ -f "${case_dir}/compare.py" ]] || die "missing compare.py for ${case_name}"

  rm -rf "${out_dir}"
  mkdir -p "${out_dir}"

  log "[$case_name] step 1/6: lower A5VM MLIR to LLVM IR"
  "${PTOAS_BIN}" ${PTOAS_FLAGS} ${A5VM_FLAGS} \
    "${case_dir}/kernel.pto" -o "${llvm_ir}"

  log "[$case_name] step 2/6: compile LLVM IR to device object"
  "${BISHENG_BIN}" \
    --target=hiipu64-hisilicon-cce \
    -march="${AICORE_ARCH}" \
    --cce-aicore-arch="${AICORE_ARCH}" \
    --cce-aicore-only \
    -c -x ir "${llvm_ir}" \
    -o "${device_obj}"

  log "[$case_name] step 3/6: build launch object and host fatobj stub"
  build_launch_object "${case_dir}" "${launch_obj}"
  build_host_stub "${case_dir}" "${device_obj}" "${host_stub_obj}"

  log "[$case_name] step 4/6: link kernel shared library"
  link_kernel_so "${case_name}" "${host_stub_obj}" "${launch_obj}" "${repack_obj}" "${repack_so}"

  log "[$case_name] step 5/6: build host executable and golden"
  build_host_executable "${case_name}" "${case_dir}" "${out_dir}"
  (
    cd "${out_dir}"
    python3 "${case_dir}/golden.py"
  )

  log "[$case_name] step 6/6: run NPU validation"
  local remote_run_cmd
  remote_run_cmd=$(cat <<EOF
cd "${out_dir}" && \
export ASCEND_HOME_PATH="${ASCEND_HOME_PATH}" && \
if [ -f "\$ASCEND_HOME_PATH/set_env.sh" ]; then source "\$ASCEND_HOME_PATH/set_env.sh" >/dev/null 2>&1; fi && \
LD_LIBRARY_PATH="${out_dir}:\$ASCEND_HOME_PATH/lib64:\${LD_LIBRARY_PATH:-}" "./${case_name}"
EOF
)
  run_remote "${remote_run_cmd}"

  local remote_ldd_cmd
  remote_ldd_cmd=$(cat <<EOF
cd "${out_dir}" && \
export ASCEND_HOME_PATH="${ASCEND_HOME_PATH}" && \
if [ -f "\$ASCEND_HOME_PATH/set_env.sh" ]; then source "\$ASCEND_HOME_PATH/set_env.sh" >/dev/null 2>&1; fi && \
LD_LIBRARY_PATH="${out_dir}:\$ASCEND_HOME_PATH/lib64:\${LD_LIBRARY_PATH:-}" ldd "./${case_name}" | grep "lib${case_name}_kernel.so"
EOF
)
  local ldd_output
  ldd_output="$(run_remote "${remote_ldd_cmd}")"
  [[ "${ldd_output}" == *"${repack_so}"* || "${ldd_output}" == *"lib${case_name}_kernel.so"* ]] || \
    die "${case_name} did not load expected kernel so: ${ldd_output}"

  (
    cd "${out_dir}"
    COMPARE_STRICT=1 python3 "${case_dir}/compare.py"
  )

  log "[$case_name] compare passed"
  log "[$case_name] output dir: ${out_dir}"
}

log "=== VPTO Host Validation ==="
log "WORK_SPACE=${WORK_SPACE}"
log "ASCEND_HOME_PATH=${ASCEND_HOME_PATH}"
log "PTOAS_BIN=${PTOAS_BIN}"
log "PTOAS_FLAGS=${PTOAS_FLAGS}"
log "A5VM_FLAGS=${A5VM_FLAGS}"
log "CASE_NAME=${CASE_NAME:-<all>}"

for case_name in "${CASES[@]}"; do
  build_one "${case_name}"
done

log "All ${#CASES[@]} VPTO case(s) passed"
