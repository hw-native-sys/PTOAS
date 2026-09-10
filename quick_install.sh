#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# Build and install PTOAS in editable mode against an existing LLVM/MLIR
# build. The persistent build directory can subsequently be used with Ninja,
# for example: ninja -C build check-pto.
#
# Usage: quick_install.sh [-v]
#   -v   verbose: trace commands and pass -v to every pip invocation.
#
# Optional env:
#   LLVM_BUILD_DIR   - default: <repo-parent>/llvm-project/build-shared
#   PTO_BUILD_DIR    - default: <repo>/build
#   PYTHON_BIN       - default: python

set -euo pipefail

VERBOSE=0
while getopts ":v" opt; do
  case "${opt}" in
    v) VERBOSE=1 ;;
    *) echo "usage: $0 [-v]" >&2; exit 2 ;;
  esac
done

PIP_VERBOSE=()
if [[ "${VERBOSE}" -eq 1 ]]; then
  set -x
  PIP_VERBOSE=(-v)
fi

PTO_SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLVM_SOURCE_DIR="${LLVM_SOURCE_DIR:-$(cd "${PTO_SOURCE_DIR}/.." && pwd)/llvm-project}"
LLVM_BUILD_DIR="${LLVM_BUILD_DIR:-${LLVM_SOURCE_DIR}/build-shared}"
PTO_BUILD_DIR="${PTO_BUILD_DIR:-${PTO_SOURCE_DIR}/build}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ ! -d "${LLVM_BUILD_DIR}/lib/cmake/llvm" || ! -d "${LLVM_BUILD_DIR}/lib/cmake/mlir" ]]; then
  echo "LLVM/MLIR CMake packages not found under: ${LLVM_BUILD_DIR}" >&2
  echo "Set LLVM_BUILD_DIR to an existing LLVM build directory." >&2
  exit 1
fi

if command -v ccache >/dev/null 2>&1; then
  export CMAKE_C_COMPILER_LAUNCHER="${CMAKE_C_COMPILER_LAUNCHER:-ccache}"
  export CMAKE_CXX_COMPILER_LAUNCHER="${CMAKE_CXX_COMPILER_LAUNCHER:-ccache}"
fi

"${PYTHON_BIN}" -m pip install \
  "${PIP_VERBOSE[@]}" \
  'scikit-build-core>=0.12.2,<2' \
  'pybind11<3' \
  'nanobind>=2.4'

LLVM_BUILD_DIR="${LLVM_BUILD_DIR}" \
  "${PYTHON_BIN}" -m pip install --editable "${PTO_SOURCE_DIR}" \
    --no-build-isolation "${PIP_VERBOSE[@]}" \
    --config-settings="build-dir=${PTO_BUILD_DIR}" \
    --config-settings="cmake.define.LLVM_DIR=${LLVM_BUILD_DIR}/lib/cmake/llvm" \
    --config-settings="cmake.define.MLIR_DIR=${LLVM_BUILD_DIR}/lib/cmake/mlir"

echo "PTOAS editable install complete."
echo "Build directory: ${PTO_BUILD_DIR}"
