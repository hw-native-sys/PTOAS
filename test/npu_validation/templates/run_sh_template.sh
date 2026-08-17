#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set -euo pipefail

RUN_MODE="${RUN_MODE:-@RUN_MODE@}"
export RUN_MODE
SOC_VERSION="${SOC_VERSION:-@SOC_VERSION@}"
export SOC_VERSION
GOLDEN_MODE="${GOLDEN_MODE:-npu}"  # sim|npu|skip
BUILD_DIR="${BUILD_DIR:-build}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

append_unique_colon_item() {
  local list="$1"
  local item="$2"
  [[ -n "${item}" && -d "${item}" ]] || {
    echo "${list}"
    return 0
  }
  if [[ -z "${list}" ]]; then
    echo "${item}"
    return 0
  fi
  case ":${list}:" in
    *":${item}:"*) echo "${list}" ;;
    *) echo "${list}:${item}" ;;
  esac
}

list_contains_file() {
  local list="$1"
  local file_name="$2"
  local dir
  IFS=':' read -r -a _pto_list_dirs <<< "${list}"
  for dir in "${_pto_list_dirs[@]}"; do
    [[ -n "${dir}" && -e "${dir}/${file_name}" ]] && return 0
  done
  return 1
}

host_lib_arch() {
  case "$(uname -m)" in
    aarch64 | arm64) echo "aarch64" ;;
    x86_64 | amd64) echo "x86_64" ;;
    *) uname -m ;;
  esac
}

collect_cann_host_link_dirs_for_root() {
  local root="$1"
  local arch="$2"
  local dirs="$3"
  local dir=""
  for dir in \
    "${root}/lib64" \
    "${root}/${arch}-linux/lib64" \
    "${root}/runtime/lib64" \
    "${root}/fwkacllib/lib64" \
    "${root}/${arch}-linux/devlib" \
    "${root}/${arch}-linux/devlib/linux/${arch}"; do
    [[ -d "${dir}" ]] || continue
    if [[ -e "${dir}/libnnopbase.so" || -e "${dir}/libascendcl.so" \
       || -e "${dir}/libplatform.so" || -e "${dir}/libtiling_api.a" \
       || -e "${dir}/libtiling_api.so" ]]; then
      dirs="$(append_unique_colon_item "${dirs}" "${dir}")"
    fi
  done
  echo "${dirs}"
}

discover_cann_host_link_dirs() {
  local arch="$1"
  local dirs=""
  local root=""
  local current_base=""
  local -a candidate_roots=()
  shopt -s nullglob

  if [[ -n "${ASCEND_HOME_PATH:-}" ]]; then
    dirs="$(collect_cann_host_link_dirs_for_root "${ASCEND_HOME_PATH}" "${arch}" "${dirs}")"
    current_base="$(basename "${ASCEND_HOME_PATH}")"
  fi
  if list_contains_file "${dirs}" "libnnopbase.so"; then
    shopt -u nullglob
    echo "${dirs}"
    return 0
  fi

  for root in \
    /usr/local/Ascend/cann \
    /usr/local/Ascend/cann-* \
    /usr/local/Ascend/ascend-toolkit/latest \
    /home/*/cann*/cann-* \
    /home/*/*/cann-* \
    /home/*/Ascend/*/cann-*; do
    [[ -d "${root}" ]] || continue
    [[ -n "${current_base}" && "${root}" == "${ASCEND_HOME_PATH:-}" ]] && continue
    if [[ -n "${current_base}" && "${root}" == *"/${current_base}" ]]; then
      candidate_roots+=("${root}")
    fi
  done
  for root in \
    /usr/local/Ascend/cann \
    /usr/local/Ascend/cann-* \
    /usr/local/Ascend/ascend-toolkit/latest \
    /home/*/cann*/cann-* \
    /home/*/*/cann-* \
    /home/*/Ascend/*/cann-*; do
    [[ -d "${root}" ]] || continue
    [[ "${root}" == "${ASCEND_HOME_PATH:-}" ]] && continue
    candidate_roots+=("${root}")
  done
  shopt -u nullglob

  for root in "${candidate_roots[@]}"; do
    dirs="$(collect_cann_host_link_dirs_for_root "${root}" "$(host_lib_arch)" "${dirs}")"
    if list_contains_file "${dirs}" "libnnopbase.so"; then
      break
    fi
  done
  echo "${dirs}"
}

collect_cann_host_include_dirs_for_root() {
  local root="$1"
  local arch="$2"
  local dirs="$3"
  local dir=""
  for dir in \
    "${root}/include" \
    "${root}/${arch}-linux/include" \
    "${root}/runtime/include" \
    "${root}/fwkacllib/include" \
    "${root}/${arch}-linux/pkg_inc" \
    "${root}/pkg_inc"; do
    [[ -d "${dir}" ]] || continue
    if [[ -e "${dir}/pto/npu/comm/async/sdma/sdma_workspace_manager.hpp" \
       || -e "${dir}/ccelib/common/runtime.h" \
       || -e "${dir}/runtime/rt.h" \
       || -e "${dir}/acl/acl.h" ]]; then
      dirs="$(append_unique_colon_item "${dirs}" "${dir}")"
    fi
  done
  echo "${dirs}"
}

discover_cann_host_include_dirs() {
  local arch="$1"
  local dirs=""
  local root=""
  local current_base=""
  local -a candidate_roots=()
  shopt -s nullglob

  if [[ -n "${ASCEND_HOME_PATH:-}" ]]; then
    dirs="$(collect_cann_host_include_dirs_for_root "${ASCEND_HOME_PATH}" "${arch}" "${dirs}")"
    current_base="$(basename "${ASCEND_HOME_PATH}")"
  fi

  for root in \
    /usr/local/Ascend/cann \
    /usr/local/Ascend/cann-* \
    /usr/local/Ascend/ascend-toolkit/latest \
    /home/*/cann*/cann-* \
    /home/*/*/cann-* \
    /home/*/Ascend/*/cann-*; do
    [[ -d "${root}" ]] || continue
    [[ -n "${current_base}" && "${root}" == "${ASCEND_HOME_PATH:-}" ]] && continue
    if [[ -n "${current_base}" && "${root}" == *"/${current_base}" ]]; then
      candidate_roots+=("${root}")
    fi
  done
  for root in \
    /usr/local/Ascend/cann \
    /usr/local/Ascend/cann-* \
    /usr/local/Ascend/ascend-toolkit/latest \
    /home/*/cann*/cann-* \
    /home/*/*/cann-* \
    /home/*/Ascend/*/cann-*; do
    [[ -d "${root}" ]] || continue
    [[ "${root}" == "${ASCEND_HOME_PATH:-}" ]] && continue
    candidate_roots+=("${root}")
  done
  shopt -u nullglob

  for root in "${candidate_roots[@]}"; do
    dirs="$(collect_cann_host_include_dirs_for_root "${root}" "${arch}" "${dirs}")"
  done
  echo "${dirs}"
}

cd "${ROOT_DIR}"
python3 "${ROOT_DIR}/golden.py"

# Best-effort resolve PTO_ISA_ROOT for generated CMakeLists.txt.
if [[ -z "${PTO_ISA_ROOT:-}" ]]; then
  search_dir="${ROOT_DIR}"
  for _ in {1..8}; do
    if [[ -d "${search_dir}/pto-isa/include" && -d "${search_dir}/pto-isa/tests/common" ]]; then
      PTO_ISA_ROOT="${search_dir}/pto-isa"
      break
    fi
    if [[ "${search_dir}" == "/" ]]; then
      break
    fi
    search_dir="$(dirname "${search_dir}")"
  done
  export PTO_ISA_ROOT="${PTO_ISA_ROOT:-}"
fi

# Best-effort load Ascend/CANN environment (toolchains + runtime). Be careful with set -euo pipefail.
if [[ -z "${ASCEND_HOME_PATH:-}" && -f "/usr/local/Ascend/cann/set_env.sh" ]]; then
  echo "[INFO] Sourcing /usr/local/Ascend/cann/set_env.sh"
  set +e
  set +u
  set +o pipefail
  source "/usr/local/Ascend/cann/set_env.sh" || true
  set -o pipefail
  set -u
  set -e
elif [[ -z "${ASCEND_HOME_PATH:-}" && -f "/usr/local/Ascend/ascend-toolkit/latest/set_env.sh" ]]; then
  echo "[INFO] Sourcing /usr/local/Ascend/ascend-toolkit/latest/set_env.sh"
  set +e
  set +u
  set +o pipefail
  source "/usr/local/Ascend/ascend-toolkit/latest/set_env.sh" || true
  set -o pipefail
  set -u
  set -e
fi

# Improve runtime linking robustness.
if [[ -n "${ASCEND_HOME_PATH:-}" ]]; then
  if [[ -z "${PTO_CANN_EXTRA_LINK_DIRS:-}" ]]; then
    PTO_CANN_EXTRA_LINK_DIRS="$(discover_cann_host_link_dirs "$(host_lib_arch)")"
  fi
  if [[ -z "${PTO_CANN_EXTRA_INCLUDE_DIRS:-}" ]]; then
    PTO_CANN_EXTRA_INCLUDE_DIRS="$(discover_cann_host_include_dirs "$(host_lib_arch)")"
  fi
  export LIBRARY_PATH="${ASCEND_HOME_PATH}/lib64${LIBRARY_PATH:+:${LIBRARY_PATH}}"
  export LD_LIBRARY_PATH="${ASCEND_HOME_PATH}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  if [[ -n "${PTO_CANN_EXTRA_LINK_DIRS:-}" ]]; then
    export PTO_CANN_EXTRA_LINK_DIRS
    export LIBRARY_PATH="${LIBRARY_PATH}:${PTO_CANN_EXTRA_LINK_DIRS}"
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${PTO_CANN_EXTRA_LINK_DIRS}"
  fi
  if [[ -n "${PTO_CANN_EXTRA_INCLUDE_DIRS:-}" ]]; then
    export PTO_CANN_EXTRA_INCLUDE_DIRS
    export CPATH="${PTO_CANN_EXTRA_INCLUDE_DIRS}${CPATH:+:${CPATH}}"
    export CPLUS_INCLUDE_PATH="${PTO_CANN_EXTRA_INCLUDE_DIRS}${CPLUS_INCLUDE_PATH:+:${CPLUS_INCLUDE_PATH}}"
  fi
fi

LD_LIBRARY_PATH_NPU="${LD_LIBRARY_PATH:-}"
LD_LIBRARY_PATH_SIM="${LD_LIBRARY_PATH_NPU}"
if [[ -n "${ASCEND_HOME_PATH:-}" ]]; then
  SIM_SOC_VERSION="${SOC_VERSION}"
  if [[ "${SOC_VERSION}" == "Ascend910" ]]; then
    if [[ -d "${ASCEND_HOME_PATH}/aarch64-linux/simulator/Ascend910A/lib" ]]; then
      SIM_SOC_VERSION="Ascend910A"
    elif [[ -d "${ASCEND_HOME_PATH}/aarch64-linux/simulator/Ascend910ProA/lib" ]]; then
      SIM_SOC_VERSION="Ascend910ProA"
    fi
  fi

  for d in \
    "${ASCEND_HOME_PATH}/aarch64-linux/simulator/${SIM_SOC_VERSION}/lib" \
    "${ASCEND_HOME_PATH}/x86_64-linux/simulator/${SIM_SOC_VERSION}/lib" \
    "${ASCEND_HOME_PATH}/simulator/${SIM_SOC_VERSION}/lib" \
    "${ASCEND_HOME_PATH}/tools/simulator/${SIM_SOC_VERSION}/lib"; do
    [[ -d "$d" ]] && LD_LIBRARY_PATH_SIM="$d:${LD_LIBRARY_PATH_SIM}"
  done
fi

mkdir -p "${ROOT_DIR}/${BUILD_DIR}"
cd "${ROOT_DIR}/${BUILD_DIR}"
ENABLE_SIM_GOLDEN="OFF"
[[ "${GOLDEN_MODE}" == "sim" ]] && ENABLE_SIM_GOLDEN="ON"
if [[ -n "${PTO_ISA_ROOT:-}" ]]; then
  cmake -DSOC_VERSION="${SIM_SOC_VERSION:-${SOC_VERSION}}" -DENABLE_SIM_GOLDEN="${ENABLE_SIM_GOLDEN}" -DPTO_ISA_ROOT="${PTO_ISA_ROOT}" ..
else
  cmake -DSOC_VERSION="${SIM_SOC_VERSION:-${SOC_VERSION}}" -DENABLE_SIM_GOLDEN="${ENABLE_SIM_GOLDEN}" ..
fi
make -j

cd "${ROOT_DIR}"

copy_outputs_as_golden() {
  if [[ -f "${ROOT_DIR}/outputs.txt" ]]; then
    while IFS= read -r name; do
      [[ -n "${name}" ]] || continue
      cp -f "${ROOT_DIR}/${name}.bin" "${ROOT_DIR}/golden_${name}.bin"
    done < "${ROOT_DIR}/outputs.txt"
    return 0
  fi
  # Fallback: copy every .bin (best-effort).
  for f in "${ROOT_DIR}"/*.bin; do
    [[ -f "$f" ]] || continue
    base="$(basename "$f")"
    cp -f "$f" "${ROOT_DIR}/golden_${base}"
  done
}

clear_golden_outputs() {
  rm -f "${ROOT_DIR}"/golden_*.bin
}

has_golden_outputs() {
  compgen -G "${ROOT_DIR}/golden_*.bin" > /dev/null
}

case "${GOLDEN_MODE}" in
  sim)
    clear_golden_outputs
    python3 "${ROOT_DIR}/golden.py"
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH_SIM}" "${ROOT_DIR}/${BUILD_DIR}/@EXECUTABLE@_sim"
    if ! has_golden_outputs; then
      copy_outputs_as_golden
    fi
    if [[ "${RUN_MODE}" == "npu" ]]; then
      LD_LIBRARY_PATH="${LD_LIBRARY_PATH_NPU}" "${ROOT_DIR}/${BUILD_DIR}/@EXECUTABLE@"
    fi
    COMPARE_STRICT=1 python3 "${ROOT_DIR}/compare.py"
    ;;
  npu)
    if [[ "${RUN_MODE}" != "npu" ]]; then
      echo "[ERROR] GOLDEN_MODE=npu requires RUN_MODE=npu" >&2
      exit 2
    fi
    clear_golden_outputs
    python3 "${ROOT_DIR}/golden.py"
    LD_LIBRARY_PATH="${LD_LIBRARY_PATH_NPU}" "${ROOT_DIR}/${BUILD_DIR}/@EXECUTABLE@"
    if ! has_golden_outputs; then
      echo "[WARN] no independent golden; validating NPU run-to-run determinism only"
      copy_outputs_as_golden
      python3 "${ROOT_DIR}/golden.py"
      LD_LIBRARY_PATH="${LD_LIBRARY_PATH_NPU}" "${ROOT_DIR}/${BUILD_DIR}/@EXECUTABLE@"
      echo "[INFO] DETERMINISM_ONLY: repeated NPU output will be compared"
    fi
    COMPARE_STRICT=1 python3 "${ROOT_DIR}/compare.py"
    ;;
  skip)
    if [[ "${RUN_MODE}" == "npu" ]]; then
      python3 "${ROOT_DIR}/golden.py"
      LD_LIBRARY_PATH="${LD_LIBRARY_PATH_NPU}" "${ROOT_DIR}/${BUILD_DIR}/@EXECUTABLE@"
    fi
    echo "[WARN] compare skipped (GOLDEN_MODE=skip)"
    ;;
  *)
    echo "[ERROR] Unknown GOLDEN_MODE=${GOLDEN_MODE} (expected: sim|npu|skip)" >&2
    exit 2
    ;;
esac
