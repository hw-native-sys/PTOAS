#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set -euo pipefail

SOC_VERSION="${SOC_VERSION:-Ascend910}"
BUILD_DIR="${BUILD_DIR:-build}"
ACL_DEVICE_ID_NPU="${ACL_DEVICE_ID:-}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

python3 "${ROOT_DIR}/golden.py"

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

if [[ -z "${ASCEND_HOME_PATH:-}" && -f "/usr/local/Ascend/cann/set_env.sh" ]]; then
  echo "[INFO] Sourcing /usr/local/Ascend/cann/set_env.sh"
  set +e
  set +u
  set +o pipefail
  source "/usr/local/Ascend/cann/set_env.sh" || true
  set -o pipefail
  set -u
  set -e
fi

if [[ -n "${ASCEND_HOME_PATH:-}" ]]; then
  export LD_LIBRARY_PATH="${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH:-}"
fi

mkdir -p "${ROOT_DIR}/${BUILD_DIR}"
cd "${ROOT_DIR}/${BUILD_DIR}"
if [[ -n "${PTO_ISA_ROOT:-}" ]]; then
  cmake -DSOC_VERSION="${SOC_VERSION}" -DPTO_ISA_ROOT="${PTO_ISA_ROOT}" ..
else
  cmake -DSOC_VERSION="${SOC_VERSION}" ..
fi
make -j

cd "${ROOT_DIR}"
if [[ -n "${ACL_DEVICE_ID_NPU}" ]]; then
  ACL_DEVICE_ID="${ACL_DEVICE_ID_NPU}" "${ROOT_DIR}/${BUILD_DIR}/subview_board"
else
  "${ROOT_DIR}/${BUILD_DIR}/subview_board"
fi

COMPARE_STRICT=1 python3 "${ROOT_DIR}/compare.py"
