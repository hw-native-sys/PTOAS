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
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env.sh"

OUTPUT_DIR="${REPRO_ROOT}/outputs/reference_build"
FIXTURES="${REPRO_ROOT}/fixtures"
mkdir -p "${OUTPUT_DIR}"

if [ -z "${BISHENG}" ] || [ ! -x "${BISHENG}" ]; then
  echo "bisheng was not found after sourcing ${CANN_ENV}" >&2
  exit 1
fi

COMMON=(
  -O3 -std=gnu++17 -fPIC -Wno-macro-redefined
  -Wno-ignored-attributes -Wno-unknown-attributes
  --cce-aicore-arch=dav-c310-vec
)

"${BISHENG}" "${COMMON[@]}" -xcce \
  -Xhost-start -Xhost-end \
  -c "${FIXTURES}/reference_compact_store.cpp" \
  -o "${OUTPUT_DIR}/reference_compact_store.o" \
  -I"${ASCEND_HOME_PATH}/include" \
  -I"${FIXTURES}" \
  -I"${ASCEND_HOME_PATH}/aarch64-linux/asc/include" \
  -I"${ASCEND_HOME_PATH}/aarch64-linux/asc/include/basic_api" \
  -I"${ASCEND_HOME_PATH}/aarch64-linux/asc/include/interface" \
  -I"${ASCEND_HOME_PATH}/aarch64-linux/asc" \
  -I"${ASCEND_HOME_PATH}/aarch64-linux/asc/impl/basic_api" \
  -I"${ASCEND_HOME_PATH}/aarch64-linux/asc/impl" \
  >"${OUTPUT_DIR}/reference_compact_store.cpp.log" 2>&1

"${BISHENG}" --cce-fatobj-link -shared -fPIC \
  -Wl,--no-undefined \
  "${OUTPUT_DIR}/reference_compact_store.o" \
  -L"${ASCEND_HOME_PATH}/lib64" \
  -Wl,-rpath,"${ASCEND_HOME_PATH}/lib64" \
  -Wl,--no-as-needed -lruntime \
  -o "${OUTPUT_DIR}/libreference_compact_store.so" \
  >"${OUTPUT_DIR}/reference_compact_store.link.log" 2>&1

echo "${OUTPUT_DIR}/libreference_compact_store.so"
