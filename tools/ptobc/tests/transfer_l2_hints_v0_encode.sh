#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# Reuse the compiler fixture to check every direction-specific enum, including
# explicit zero, through the existing v0 attribute-dictionary encoding.
set -euo pipefail
: "${PTOBC_BIN:?PTOBC_BIN must be set}"
: "${TEST_IR:?TEST_IR must be set}"
OUT_DIR="${PWD}/ptobc_transfer_l2_hints_out"
mkdir -p "${OUT_DIR}"
"${PTOBC_BIN}" encode "${TEST_IR}" -o "${OUT_DIR}/hints.ptobc"
"${PTOBC_BIN}" decode "${OUT_DIR}/hints.ptobc" -o "${OUT_DIR}/hints.pto"
for direction in load store; do
  for mode in normal_first_victim normal_last_victim normal_persistent not_alloc_clean; do
    grep -F "#pto.${direction}_l2_hint<${mode}>" "${OUT_DIR}/hints.pto" >/dev/null
  done
done
for mode in not_alloc_keep not_alloc_drop; do
  grep -F "#pto.load_l2_hint<${mode}>" "${OUT_DIR}/hints.pto" >/dev/null
done
"${PTOBC_BIN}" encode "${OUT_DIR}/hints.pto" -o "${OUT_DIR}/hints-again.ptobc"
cmp "${OUT_DIR}/hints.ptobc" "${OUT_DIR}/hints-again.ptobc"
