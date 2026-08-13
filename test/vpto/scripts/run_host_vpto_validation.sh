#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
VPTO_ROOT="${VPTO_ROOT:-${ROOT_DIR}/test/vpto/cases}"
CASES_ROOT="${CASES_ROOT:-${VPTO_ROOT}}"
NPU_VALIDATION_COMMON_DIR="${NPU_VALIDATION_COMMON_DIR:-${ROOT_DIR}/test/vpto/npu_validation/common}"

WORK_SPACE="${WORK_SPACE:-}"
ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-}"
PTOAS_BIN="${PTOAS_BIN:-${ROOT_DIR}/install/bin/ptoas}"
PTOAS_FLAGS="${PTOAS_FLAGS:---pto-arch a5 --pto-backend=vpto}"
# pto-isa include dir for VPTO pipe bridge cases (vpto_bridge.cpp).
# Default resolves to a pto-isa checkout in the parent of the PTOAS
# workspace.
PTO_ISA_INCLUDE_DIR="${PTO_ISA_INCLUDE_DIR:-${SCRIPT_DIR}/../../../../../pto-isa/include}"
# set he HOST_RUNNER to "ssh root@localhost" if must change user to root to access the device 
HOST_RUNNER="${HOST_RUNNER:-}"
CASE_NAME="${CASE_NAME:-}"
DEVICE="${DEVICE:-SIM}"
SIM_LIB_DIR="${SIM_LIB_DIR:-}"
COMPILE_ONLY="${COMPILE_ONLY:-0}"
PTODSL_SIM_SOC_VERSION="${PTODSL_SIM_SOC_VERSION:-Ascend950PR_9599}"

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
    bash -lc "${cmd}"
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

resolve_sim_lib_dir() {
  if [[ "${DEVICE}" != "SIM" ]]; then
    return 0
  fi

  if [[ -n "${SIM_LIB_DIR}" ]]; then
    [[ -d "${SIM_LIB_DIR}" ]] ||
      die "SIM_LIB_DIR is set but invalid: ${SIM_LIB_DIR}"
    return 0
  fi

  local -a candidates=()
  readarray -t candidates < <(
    find "${ASCEND_HOME_PATH}" -type d -path '*/simulator/dav_3510/lib' | sort
  )

  if [[ "${#candidates[@]}" -eq 1 ]]; then
    SIM_LIB_DIR="${candidates[0]}"
    log "SIM_LIB_DIR is unset; auto-selected: ${SIM_LIB_DIR}"
    return 0
  fi

  if [[ "${#candidates[@]}" -gt 1 ]]; then
    SIM_LIB_DIR="${candidates[0]}"
    log "SIM_LIB_DIR is unset; multiple dav_3510 simulator dirs found, using: ${SIM_LIB_DIR}"
    return 0
  fi

  die "SIM_LIB_DIR is required for DEVICE=SIM and no dav_3510 simulator lib dir was found under: ${ASCEND_HOME_PATH}"
}

BISHENG_BIN="${BISHENG_BIN:-${ASCEND_HOME_PATH}/bin/bisheng}"

command -v "${BISHENG_BIN}" >/dev/null 2>&1 || die "bisheng not found: ${BISHENG_BIN}"
command -v python3 >/dev/null 2>&1 || die "python3 not found"

mkdir -p "${WORK_SPACE}"
WORK_SPACE="$(cd "${WORK_SPACE}" && pwd)"

is_ptodsl_case_dir() {
  [[ -f "$1/kernel.py" ]]
}

is_ptodsl_case_file() {
  local case_path="$1"
  local base_name
  base_name="$(basename "${case_path}")"
  [[ -f "${case_path}" ]] &&
    [[ "${case_path}" == *.py ]] &&
    [[ "${base_name}" != "golden.py" ]] &&
    [[ "${base_name}" != "compare.py" ]] &&
    [[ "${base_name}" != "kernel.py" ]] &&
    [[ "${base_name}" != _* ]]
}

is_ptodsl_case_path() {
  is_ptodsl_case_dir "$1" || is_ptodsl_case_file "$1"
}

ptodsl_case_script() {
  local case_path="$1"
  if is_ptodsl_case_dir "${case_path}"; then
    printf '%s\n' "${case_path}/kernel.py"
    return 0
  fi
  if is_ptodsl_case_file "${case_path}"; then
    printf '%s\n' "${case_path}"
    return 0
  fi
  die "path is not a PTODSL case: ${case_path}"
}

validate_case_path() {
  local case_name="$1"
  local case_path="$2"

  if is_ptodsl_case_path "${case_path}"; then
    return 0
  fi
  [[ -d "${case_path}" ]] || die "case ${case_name} is neither a directory nor a PTODSL case file"
  for f in launch.cpp main.cpp golden.py compare.py; do
    [[ -f "${case_path}/${f}" ]] || die "case ${case_name} is missing ${f}"
  done
  [[ -f "${case_path}/kernel.pto" ]] || die "case ${case_name} must provide kernel.pto"
}

discover_cases() {
  local onboard_only_prefix="onboard-only/"

  if [[ -n "${CASE_NAME}" ]]; then
    [[ "${CASE_NAME}" != /* ]] || die "CASE_NAME must be relative to CASES_ROOT: ${CASE_NAME}"
    if [[ "${DEVICE}" == "SIM" && "${COMPILE_ONLY}" != "1" &&
          "${CASE_NAME}" == "${onboard_only_prefix}"* ]]; then
      die "case ${CASE_NAME} is onboard-only and cannot run with DEVICE=SIM"
    fi
    local requested_path="${CASES_ROOT}/${CASE_NAME}"
    [[ -e "${requested_path}" ]] || die "unknown case: ${CASE_NAME}"
    validate_case_path "${CASE_NAME}" "${requested_path}"
    printf "%s\n" "${CASE_NAME}"
    return 0
  fi

  find "${CASES_ROOT}" -mindepth 1 -type d | sort | while read -r dir; do
    [[ -f "${dir}/kernel.pto" || -f "${dir}/kernel.py" ]] || continue
    local rel="${dir#${CASES_ROOT}/}"
    if ! is_ptodsl_case_dir "${dir}"; then
      local ok=1
      for f in launch.cpp main.cpp golden.py compare.py; do
        if [[ ! -f "${dir}/${f}" ]]; then
          ok=0
          break
        fi
      done
      [[ "${ok}" -eq 1 ]] || continue
    fi
    if [[ "${DEVICE}" == "SIM" && "${COMPILE_ONLY}" != "1" &&
          "${rel}" == "${onboard_only_prefix}"* ]]; then
      continue
    fi
    printf "%s\n" "${rel}"
  done

  find "${CASES_ROOT}" -type f -name '*.py' | sort | while read -r path; do
    is_ptodsl_case_file "${path}" || continue
    local rel="${path#${CASES_ROOT}/}"
    if [[ "${DEVICE}" == "SIM" && "${COMPILE_ONLY}" != "1" &&
          "${rel}" == "${onboard_only_prefix}"* ]]; then
      continue
    fi
    printf "%s\n" "${rel}"
  done
}

if [[ "${DEVICE}" == "SIM" && "${COMPILE_ONLY}" != "1" &&
      "${CASE_NAME}" == onboard-only/* ]]; then
  die "case ${CASE_NAME} is onboard-only and cannot run with DEVICE=SIM"
fi

readarray -t CASES < <(discover_cases)
[[ "${#CASES[@]}" -gt 0 ]] || die "no cases found under ${CASES_ROOT}"

needs_legacy_sim_lib=0
if [[ "${DEVICE}" == "SIM" ]]; then
  for case_name in "${CASES[@]}"; do
    if ! is_ptodsl_case_path "${CASES_ROOT}/${case_name}"; then
      needs_legacy_sim_lib=1
      break
    fi
  done
fi
if [[ "${needs_legacy_sim_lib}" == "1" ]]; then
  resolve_sim_lib_dir
fi

case_output_token() {
  printf '%s' "$1" | sed 's#[/[:space:]]#_#g'
}

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
    --cce-aicore-arch=dav-c310 \
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

# Compiles the PTO-ISA template bridge (vpto_bridge.cpp) to per-target
# bitcode. ObjectEmission merges it with the VPTO device LLVM IR before
# Bisheng compiles the device object, so Cube and Vector must be compiled
# separately for their own dav-c310 target.
build_vpto_bridge_object() {
  local case_dir="$1"
  local target="$2"
  local out_obj="$3"
  [[ -d "${PTO_ISA_INCLUDE_DIR}" ]] ||
    die "PTO_ISA_INCLUDE_DIR is not a directory: ${PTO_ISA_INCLUDE_DIR}"
  "${BISHENG_BIN}" -O2 -std=c++17 -c -emit-llvm -xcce --cce-aicore-only \
    --cce-aicore-arch="dav-c310-${target}" -DREGISTER_BASE \
    -I "${PTO_ISA_INCLUDE_DIR}" "${case_dir}/vpto_bridge.cpp" -o "${out_obj}"
}

link_kernel_so() {
  local case_name="$1"
  local kernel_fatobj="$2"
  local launch_obj="$3"
  local kernel_so="$4"
  local extra_lib_dirs=()
  local extra_link_libs=()

  if [[ "${DEVICE}" == "SIM" ]]; then
    [[ -n "${SIM_LIB_DIR}" && -d "${SIM_LIB_DIR}" ]] ||
      die "SIM_LIB_DIR is not set or invalid for DEVICE=SIM: ${SIM_LIB_DIR}"
    extra_lib_dirs+=(-L "${SIM_LIB_DIR}" -Wl,-rpath,"${SIM_LIB_DIR}")
    extra_link_libs+=(-Wl,--no-as-needed -lruntime_camodel)
  else
    extra_link_libs+=(-Wl,--no-as-needed -lruntime)
  fi

  "${BISHENG_BIN}" \
    -fPIC -s -Wl,-z,relro -Wl,-z,now --cce-fatobj-link \
    -shared -Wl,-soname,"lib${case_name}_kernel.so" \
    -L "${ASCEND_HOME_PATH}/lib64" \
    "${extra_lib_dirs[@]}" \
    -Wl,-rpath,"${ASCEND_HOME_PATH}/lib64" \
    -o "${kernel_so}" \
    "${kernel_fatobj}" \
    "${launch_obj}" \
    "${extra_link_libs[@]}"
}

build_host_executable() {
  local case_token="$1"
  local case_dir="$2"
  local out_dir="$3"
  local extra_ldflags=()
  local extra_lib_dirs=()
  if [[ "${DEVICE}" == "SIM" ]]; then
    [[ -n "${SIM_LIB_DIR}" && -d "${SIM_LIB_DIR}" ]] ||
      die "SIM_LIB_DIR is not set or invalid for DEVICE=SIM: ${SIM_LIB_DIR}"
    extra_lib_dirs+=(-L "${SIM_LIB_DIR}" -Wl,-rpath,"${SIM_LIB_DIR}")
    extra_ldflags+=(-Wl,--allow-shlib-undefined -lruntime_camodel)
  else
    extra_ldflags+=(-Wl,--allow-shlib-undefined -lruntime)
  fi

  "${BISHENG_BIN}" \
    -xc++ -include stdint.h -include stddef.h -std=c++17 \
    "${case_dir}/main.cpp" \
    -I "${case_dir}" \
    -I "${NPU_VALIDATION_COMMON_DIR}" \
    -I "${ASCEND_HOME_PATH}/include" \
    -L "${out_dir}" \
    -L "${ASCEND_HOME_PATH}/lib64" \
    "${extra_lib_dirs[@]}" \
    -Wl,-rpath,"${out_dir}" \
    -Wl,-rpath,"${ASCEND_HOME_PATH}/lib64" \
    -o "${out_dir}/${case_token}" \
    -l"${case_token}_kernel" \
    "${extra_ldflags[@]}" \
    -lstdc++ -lascendcl -lm -ltiling_api -lplatform -lc_sec -ldl -lnnopbase
}

run_ptodsl_case() {
  local case_name="$1"
  local case_path="$2"
  local out_dir="$3"
  local case_script
  case_script="$(ptodsl_case_script "${case_path}")"

  log "[$case_name] run PTODSL source-backed case"
  (
    cd "${out_dir}"
    export PTODSL_CACHE_DIR="${out_dir}/ptodsl-cache"
    export PATH="$(dirname "${PTOAS_BIN}"):${PATH}"
    if [[ "${DEVICE}" == "SIM" ]]; then
      "${ROOT_DIR}/scripts/sim_dsl.sh" \
        --soc-version "${PTODSL_SIM_SOC_VERSION}" \
        --output "${out_dir}/msprof" \
        "${case_script}"
    else
      export LD_LIBRARY_PATH="${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH:-}"
      python3 "${case_script}"
    fi
  )
  log "[$case_name] output dir: ${out_dir}"
}

build_one_impl() {
  local case_name="$1"
  local case_path="${CASES_ROOT}/${case_name}"
  local case_token
  case_token="$(case_output_token "${case_name}")"
  local out_dir="${WORK_SPACE}/${case_token}"
  local launch_obj="${out_dir}/launch.o"
  local kernel_fatobj="${out_dir}/kernel.fatobj.o"
  local kernel_so="${out_dir}/lib${case_token}_kernel.so"
  local -a ptoas_args=()

  if is_ptodsl_case_path "${case_path}"; then
    run_ptodsl_case "${case_name}" "${case_path}" "${out_dir}"
    return 0
  fi
  local case_dir="${case_path}"

  [[ -f "${case_dir}/kernel.pto" ]] || die "missing kernel.pto for ${case_name}"

  [[ -f "${case_dir}/main.cpp" ]] || die "missing main.cpp for ${case_name}"
  [[ -f "${case_dir}/launch.cpp" ]] || die "missing launch.cpp for ${case_name}"
  [[ -f "${case_dir}/golden.py" ]] || die "missing golden.py for ${case_name}"
  [[ -f "${case_dir}/compare.py" ]] || die "missing compare.py for ${case_name}"

  if [[ -f "${case_dir}/ptoas.flags" ]]; then
    read -r -a ptoas_args < "${case_dir}/ptoas.flags"
  else
    read -r -a ptoas_args <<< "${PTOAS_FLAGS}"
  fi

  local bridge_cube_bc=""
  local bridge_vector_bc=""
  if [[ -f "${case_dir}/vpto_bridge.cpp" ]]; then
    bridge_cube_bc="${out_dir}/vpto_bridge_cube.bc"
    bridge_vector_bc="${out_dir}/vpto_bridge_vector.bc"
    log "[$case_name] step 0/4: build VPTO cube/vector bridge bitcode"
    build_vpto_bridge_object "${case_dir}" cube "${bridge_cube_bc}"
    build_vpto_bridge_object "${case_dir}" vec "${bridge_vector_bc}"
  fi

  log "[$case_name] step 1/4: emit kernel fatobj"
  PTOAS_VPTO_CUBE_BRIDGE_BITCODE="${bridge_cube_bc}" \
  PTOAS_VPTO_VECTOR_BRIDGE_BITCODE="${bridge_vector_bc}" \
    "${PTOAS_BIN}" "${ptoas_args[@]}" \
      "${case_dir}/kernel.pto" -o "${kernel_fatobj}"

  log "[$case_name] step 2/4: build launch object"
  build_launch_object "${case_dir}" "${launch_obj}"

  log "[$case_name] step 3/4: link kernel shared library"
  link_kernel_so "${case_token}" "${kernel_fatobj}" "${launch_obj}" "${kernel_so}"

  if [[ "${COMPILE_ONLY}" == "1" ]]; then
    log "[$case_name] compile-only mode: stop after kernel shared library"
    log "[$case_name] output dir: ${out_dir}"
    return 0
  fi

  log "[$case_name] step 4/4: build host executable and golden"
  build_host_executable "${case_token}" "${case_dir}" "${out_dir}"
  (
    cd "${out_dir}"
    python3 "${case_dir}/golden.py"
  )

  log "[$case_name] run NPU validation"
  local remote_run_cmd
  remote_run_cmd=$(cat <<EOF
cd "${out_dir}" && \
export ASCEND_HOME_PATH="${ASCEND_HOME_PATH}" && \
if [ -f "\$ASCEND_HOME_PATH/set_env.sh" ]; then source "\$ASCEND_HOME_PATH/set_env.sh" >/dev/null 2>&1; fi && \
LD_LIBRARY_PATH="${out_dir}:${SIM_LIB_DIR}:\$ASCEND_HOME_PATH/lib64:\${LD_LIBRARY_PATH:-}" "./${case_token}"
EOF
)
  run_remote "${remote_run_cmd}"

  local remote_ldd_cmd
  remote_ldd_cmd=$(cat <<EOF
cd "${out_dir}" && \
export ASCEND_HOME_PATH="${ASCEND_HOME_PATH}" && \
if [ -f "\$ASCEND_HOME_PATH/set_env.sh" ]; then source "\$ASCEND_HOME_PATH/set_env.sh" >/dev/null 2>&1; fi && \
LD_LIBRARY_PATH="${out_dir}:${SIM_LIB_DIR}:\$ASCEND_HOME_PATH/lib64:\${LD_LIBRARY_PATH:-}" ldd "./${case_token}" | grep "lib${case_token}_kernel.so"
EOF
)
  local ldd_output
  ldd_output="$(run_remote "${remote_ldd_cmd}")"
  [[ "${ldd_output}" == *"${kernel_so}"* || "${ldd_output}" == *"lib${case_token}_kernel.so"* ]] || \
    die "${case_name} did not load expected kernel so: ${ldd_output}"

  (
    cd "${out_dir}"
    COMPARE_STRICT=1 python3 "${case_dir}/compare.py"
  )

  log "[$case_name] compare passed"
  log "[$case_name] output dir: ${out_dir}"
}

build_one() {
  local case_name="$1"
  local case_token
  case_token="$(case_output_token "${case_name}")"
  local out_dir="${WORK_SPACE}/${case_token}"
  local case_log="${out_dir}/validation.log"

  rm -rf "${out_dir}"
  mkdir -p "${out_dir}"

  (
    build_one_impl "${case_name}"
  ) 2>&1 | tee "${case_log}"
}

log "=== VPTO Host Validation ==="
log "WORK_SPACE=${WORK_SPACE}"
log "ASCEND_HOME_PATH=${ASCEND_HOME_PATH}"
log "PTOAS_BIN=${PTOAS_BIN}"
log "PTOAS_FLAGS=${PTOAS_FLAGS}"
log "COMPILE_ONLY=${COMPILE_ONLY}"
log "CASE_NAME=${CASE_NAME:-<all>}"

for case_name in "${CASES[@]}"; do
  build_one "${case_name}"
done

log "All ${#CASES[@]} VPTO case(s) passed"
