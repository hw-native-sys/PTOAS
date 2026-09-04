#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set -euo pipefail

WORKSPACE="${1:-${WORKSPACE:-$(pwd)}}"
WORKSPACE="$(cd "${WORKSPACE}" && pwd)"
BUILD_ROOT="${BUILD_ROOT:-${WORKSPACE}/.work/gitcode-vpto-sim}"
ASCEND_3RD_LIB_PATH="${ASCEND_3RD_LIB_PATH:-/home/opensource}"
CASE_PREFIX="${VPTO_SIM_CASE_PREFIX:-}"
JOBS="${VPTO_SIM_JOBS:-$(nproc 2>/dev/null || printf '4')}"

[[ -d "${ASCEND_3RD_LIB_PATH}" ]] || {
  echo "ERROR: ASCEND_3RD_LIB_PATH is unavailable: ${ASCEND_3RD_LIB_PATH}" >&2
  exit 1
}

mkdir -p "${BUILD_ROOT}"
exec > >(tee "${BUILD_ROOT}/vpto-sim.log") 2>&1

if [[ -f "${ASCEND_HOME_PATH:-}/bin/setenv.bash" ]]; then
  # shellcheck disable=SC1091
  source "${ASCEND_HOME_PATH}/bin/setenv.bash"
fi

[[ -n "${ASCEND_HOME_PATH:-}" && -d "${ASCEND_HOME_PATH}" ]] || {
  echo "ERROR: ASCEND_HOME_PATH is required" >&2
  exit 1
}

BISHENG_BIN="${BISHENG_BIN:-${ASCEND_HOME_PATH}/bin/bisheng}"
MSPROF_BIN="${MSPROF_BIN:-${ASCEND_HOME_PATH}/bin/msprof}"
command -v "${BISHENG_BIN}" >/dev/null 2>&1 || {
  echo "ERROR: bisheng is unavailable: ${BISHENG_BIN}" >&2
  exit 1
}
command -v "${MSPROF_BIN}" >/dev/null 2>&1 || {
  echo "ERROR: msprof is unavailable: ${MSPROF_BIN}" >&2
  exit 1
}

readarray -t SIM_LIB_DIRS < <(
  find "${ASCEND_HOME_PATH}" -type d -path '*/simulator/dav_3510/lib' 2>/dev/null | sort
)
if [[ "${#SIM_LIB_DIRS[@]}" -eq 0 ]]; then
  echo "ERROR: dav_3510 simulator library is unavailable under ${ASCEND_HOME_PATH}" >&2
  exit 1
fi
SIM_LIB_DIR="${SIM_LIB_DIRS[0]}"

echo "CANN environment: ${ASCEND_HOME_PATH}"
echo "Bisheng: ${BISHENG_BIN}"
echo "msprof: ${MSPROF_BIN}"
echo "SIM_LIB_DIR: ${SIM_LIB_DIR}"

BUILD_JOBS="${BUILD_JOBS:-${JOBS}}"
bash "${WORKSPACE}/build.sh" --build \
  --cann_3rd_lib_path "${ASCEND_3RD_LIB_PATH}" \
  -j "${BUILD_JOBS}"

if [[ -f "${WORKSPACE}/build/ptoas-test-env.sh" ]]; then
  # shellcheck disable=SC1091
  source "${WORKSPACE}/build/ptoas-test-env.sh"
fi
PTOAS_BIN="${PTOAS_BIN:-${WORKSPACE}/build/tools/ptoas/ptoas}"
[[ -x "${PTOAS_BIN}" ]] || {
  echo "ERROR: built ptoas is unavailable: ${PTOAS_BIN}" >&2
  exit 1
}

export PTOAS_BIN ASCEND_HOME_PATH SIM_LIB_DIR DEVICE=SIM JOBS
export WORK_SPACE="${BUILD_ROOT}/cases"
export CASES_ROOT="${WORKSPACE}/test/vpto/cases"
export PATH="$(dirname "${PTOAS_BIN}"):${PATH}"

if [[ -n "${CASE_PREFIX}" ]]; then
  export CASE_PREFIX
fi

bash "${WORKSPACE}/test/vpto/scripts/run_host_vpto_validation_parallel.sh"
