#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# Build the EmitC host/device path, then compare real-device results.
set -euo pipefail
CASE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${CASE_ROOT}/../../.." && pwd)"
: "${PTOAS_BIN:=${REPO_ROOT}/build/tools/ptoas/ptoas}"
: "${PYTHON_BIN:=python3}"
: "${WORK_SPACE:=${REPO_ROOT}/build/issue1745-a5-runtime}"
: "${SOC_VERSION:=Ascend950}"
: "${BUILD_JOBS:=8}"
: "${PTO_ISA_ROOT:?Set PTO_ISA_ROOT to the PTO-ISA checkout}"
: "${ASCEND_HOME_PATH:?Source the CANN environment first}"
mkdir -p "${WORK_SPACE}/Issue1745Tf32"
for testcase in issue1745_tf32 issue1745_tf32_loop; do
    output="${WORK_SPACE}/Issue1745Tf32/${testcase}-pto.cpp"
    "${PTOAS_BIN}" --pto-arch=a5 --enable-insert-sync --check-sync-self-coverage \
        "${CASE_ROOT}/${testcase}.pto" -o "${output}"
    "${PYTHON_BIN}" "${REPO_ROOT}/test/npu_validation/scripts/generate_testcase.py" \
        --input "${output}" --testcase "${testcase}" --output-root "${WORK_SPACE}/generated" \
        --soc-version "${SOC_VERSION}" --aicore-arch dav-c310-cube
    case_dir="${WORK_SPACE}/generated/Issue1745Tf32/${testcase}"
    cmake -S "${case_dir}" -B "${case_dir}/build" -G Ninja \
        -DPTO_ISA_ROOT="${PTO_ISA_ROOT}" -DENABLE_SIM_GOLDEN=OFF
    cmake --build "${case_dir}/build" --parallel "${BUILD_JOBS}"
    products=2
    if [[ "${testcase}" == issue1745_tf32_loop ]]; then
        products=80
    fi
    (
        cd "${case_dir}"
        "${PYTHON_BIN}" "${CASE_ROOT}/npu_validation/golden.py" --products "${products}"
        timeout 120 "./build/${testcase}"
        "${PYTHON_BIN}" "${CASE_ROOT}/npu_validation/compare.py"
    )
done
