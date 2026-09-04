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
BUILD_ROOT="${BUILD_ROOT:-${WORKSPACE}/.work/gitcode-build-and-test}"
ASCEND_3RD_LIB_PATH="${ASCEND_3RD_LIB_PATH:-/home/jenkins/opensource}"
LLVM_GIT_URL="${LLVM_GIT_URL:-https://gitcode.com/GitHub_Trending/ll/llvm-project.git}"
LLVM_GIT_REF="${LLVM_GIT_REF:-llvmorg-19.1.7}"
LLVM_VPTO_PATCH_URL="${LLVM_VPTO_PATCH_URL:-https://gitcode.com/cann-src-third-party/llvm/releases/download/19.1.7-h0/feature-vpto-last3.patch}"
LLVM_VPTO_PATCH_SHA256="${LLVM_VPTO_PATCH_SHA256:-a49c1d3dd8ab78e93264712bc0d46deb536196a54abb2c2ee02abd914cd385e2}"

if [[ -z "${BUILD_JOBS:-}" ]]; then
  BUILD_JOBS="$(nproc 2>/dev/null || printf '4')"
  (( BUILD_JOBS > 16 )) && BUILD_JOBS=16
fi

mkdir -p "${BUILD_ROOT}"
exec > >(tee "${BUILD_ROOT}/build-and-test.log") 2>&1

echo "Build-and-test workspace: ${WORKSPACE}"
echo "LLVM cache root: ${ASCEND_3RD_LIB_PATH}"
echo "Parallel jobs: ${BUILD_JOBS}"

bash "${WORKSPACE}/.gitcode/scripts/probe_vpto_sim_env.sh"

if [[ ! -d "${ASCEND_3RD_LIB_PATH}/llvm-19" ]]; then
  echo "ERROR: the compile image does not provide the LLVM cache at ${ASCEND_3RD_LIB_PATH}/llvm-19" >&2
  exit 1
fi

if [[ -d /opt/buildtools/python-3.10.2/bin ]]; then
  export PATH="/opt/buildtools/python-3.10.2/bin:${PATH}"
fi
if [[ -f /opt/rh/devtoolset-7/enable ]]; then
  # shellcheck disable=SC1091
  source /opt/rh/devtoolset-7/enable
fi

export ASCEND_3RD_LIB_PATH LLVM_GIT_URL LLVM_GIT_REF
export LLVM_VPTO_PATCH_URL LLVM_VPTO_PATCH_SHA256

echo "Building PTOAS with the compile image's cached LLVM/MLIR tree"
bash "${WORKSPACE}/build.sh" \
  --build \
  --cann_3rd_lib_path "${ASCEND_3RD_LIB_PATH}" \
  -j "${BUILD_JOBS}"

# build.sh records the exact cached LLVM and install paths used by this build.
# shellcheck disable=SC1091
source "${WORKSPACE}/build/ptoas-test-env.sh"
export PATH="${WORKSPACE}/build/tools/ptoas:${WORKSPACE}/build/tools/ptobc:${LLVM_BUILD_DIR}/bin:${PATH}"

echo "Running PTODSL tests"
ctest --test-dir "${WORKSPACE}/build" --output-on-failure -L PTODSL

echo "Running PTO-BC tests"
ctest --test-dir "${WORKSPACE}/build" --output-on-failure -R '^ptobc_'

echo "Preparing lit test dependencies"
ninja -C "${WORKSPACE}/build" -j "${BUILD_JOBS}" \
  ptoas_runtime_deps \
  pto-test-opt \
  pto-vpto-scheduler-tracker-test

echo "Running lit tests (excluding runop smoke tests)"
"${LLVM_BUILD_DIR}/bin/llvm-lit" \
  -sv \
  --filter-out 'npu_validation/runop_' \
  "${WORKSPACE}/build/test/lit"

echo "Build-and-test completed successfully."
