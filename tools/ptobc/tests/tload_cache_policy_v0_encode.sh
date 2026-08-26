#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set -euo pipefail

PTOBC_BIN=${PTOBC_BIN:-}
TESTDATA_DIR=${TESTDATA_DIR:-}
if [[ -z "${PTOBC_BIN}" || -z "${TESTDATA_DIR}" ]]; then
  echo "error: PTOBC_BIN and TESTDATA_DIR must be set" >&2
  exit 2
fi

IN="${TESTDATA_DIR}/tload_cache_policy_v0_roundtrip.pto"
OUT_DIR=${OUT_DIR:-"${PWD}/ptobc_tload_cache_policy_out"}
mkdir -p "${OUT_DIR}"

BC="${OUT_DIR}/tload_cache_policy_v0_roundtrip.ptobc"
ROUNDTRIP="${OUT_DIR}/tload_cache_policy_v0_roundtrip.roundtrip.pto"

"${PTOBC_BIN}" encode "${IN}" -o "${BC}"
"${PTOBC_BIN}" decode "${BC}" -o "${ROUNDTRIP}"

grep -F "pto.tload ins(" "${ROUNDTRIP}" >/dev/null
grep -F "cache_policy = #pto.load_cache_policy<l2_bypass>" "${ROUNDTRIP}" >/dev/null
