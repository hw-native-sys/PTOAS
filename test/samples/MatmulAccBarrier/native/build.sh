#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
set -euo pipefail
sample_dir=$(cd -- "$(dirname -- "$0")" && pwd)
output_dir=${1:-"$sample_dir/build"}
: "${PTO_ISA_ROOT:?Set PTO_ISA_ROOT to the selected PTO ISA checkout}"
: "${ASCEND_HOME_PATH:?Source the selected CANN set_env.sh first}"
mkdir -p -- "$output_dir"
"$ASCEND_HOME_PATH/bin/bisheng" -xcce -O2 -std=c++17 -fPIC -shared \
    -fenable-matrix --cce-aicore-enable-tl --cce-aicore-arch=dav-c310-cube \
    -D__DAV_CUBE__ --cce-fatobj-link -Wno-ignored-attributes \
    -mllvm -cce-aicore-stack-size=0x8000 \
    -mllvm -cce-aicore-function-stack-size=0x8000 \
    -mllvm -cce-aicore-addr-transform \
    -mllvm -cce-aicore-dcci-insert-for-scalar=false \
    -I"$PTO_ISA_ROOT/include" -I"$ASCEND_HOME_PATH/include" \
    "$sample_dir/barrier_kernel.cpp" \
    -L"$ASCEND_HOME_PATH/lib64" -lruntime -lascendcl \
    -o "$output_dir/libbarrier_benchmark.so"
