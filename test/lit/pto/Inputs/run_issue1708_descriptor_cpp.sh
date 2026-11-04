#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# Compile and execute generated C++ against the host descriptor model.
set -euo pipefail
if (( $# < 1 || $# > 2 )); then
    echo "usage: $0 generated.cpp [runtime.cpp]" >&2
    exit 2
fi
inputs=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
runtime=${2:-"$inputs/issue1708_descriptor_runtime.cpp"}
executable="$1.descriptor-test"
trap 'rm -f -- "$executable"' EXIT
read -r -a compiler <<< "${CXX:-c++}"
"${compiler[@]}" -std=c++17 -O0 -I "$inputs/issue1708-runtime" \
    -include "$1" "$runtime" -o "$executable"
"$executable"
