#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
# Build the host shim over AsyncWorkspace as a shared library that the PTODSL ST
# harness loads with ctypes.
#
# No CANN headers or libraries are needed to build this: AsyncWorkspace resolves
# every toolkit symbol with dlopen, so the only link dependency is -ldl. The
# toolkit is required to run the result, not to produce it, which means this
# builds on a machine with no driver and no card.
#
#   test/comm/build_async_shim.sh [output_dir]
#
# Environment:
#   CXX   host C++ compiler; defaults to g++

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

OUT_DIR="${1:-${SCRIPT_DIR}/build}"
CXX_BIN="${CXX:-g++}"
LIB_NAME="libpto_async_shim.so"

if ! command -v -- "${CXX_BIN}" >/dev/null 2>&1; then
  echo "error: host C++ compiler not found: ${CXX_BIN} (override with CXX=)" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"

# AsyncSessionABI.h comes from include/, AsyncWorkspace.h from this directory.
cxx_args=(
  -std=c++17
  -O2
  -Wall
  -Wextra
  -fPIC
  -shared
  "-I${REPO_ROOT}/include"
  "-I${SCRIPT_DIR}"
  "${SCRIPT_DIR}/AsyncWorkspaceShim.cpp"
  -o "${OUT_DIR}/${LIB_NAME}"
  -Wl,-z,relro
  -Wl,-z,now
  -Wl,-z,noexecstack
  -ldl
)

"${CXX_BIN}" "${cxx_args[@]}"

echo "built ${OUT_DIR}/${LIB_NAME}"
