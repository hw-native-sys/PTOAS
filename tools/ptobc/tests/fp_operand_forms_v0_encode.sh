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
PTOAS_BIN=${PTOAS_BIN:-}
TESTDATA_DIR=${TESTDATA_DIR:-}
PYTHON_EXECUTABLE=${PYTHON_EXECUTABLE:-}
if [[ -z "${PTOBC_BIN}" || -z "${PTOAS_BIN}" || -z "${PYTHON_EXECUTABLE}" || -z "${TESTDATA_DIR}" ]]; then
  echo "error: PTOBC_BIN, PTOAS_BIN, PYTHON_EXECUTABLE, and TESTDATA_DIR must be set" >&2
  exit 2
fi

IN="${TESTDATA_DIR}/fp_operand_forms_v0_roundtrip.pto"
OUT_DIR=${OUT_DIR:-"${PWD}/ptobc_fp_operand_forms_out"}
mkdir -p "${OUT_DIR}"

BC="${OUT_DIR}/fp_operand_forms_v0_roundtrip.ptobc"
ROUNDTRIP="${OUT_DIR}/fp_operand_forms_v0_roundtrip.roundtrip.pto"

"${PTOBC_BIN}" encode "${IN}" -o "${BC}"
"${PTOBC_BIN}" decode "${BC}" -o "${ROUNDTRIP}"

"${PYTHON_EXECUTABLE}" - <<'PY' "${BC}"
from pathlib import Path
import sys

data = Path(sys.argv[1]).read_bytes()
expected = {
    b"\x22\x10": "textract_fp",
    b"\x2e\x10": "tinsert_fp",
    b"\x39\x10": "tmov.fp",
}
for encoding, name in expected.items():
    if encoding not in data:
        raise SystemExit(f"missing legacy {name} opcode encoding")
PY

grep -F "pto.textract ins(" "${ROUNDTRIP}" >/dev/null
grep -F "pto.tinsert ins(" "${ROUNDTRIP}" >/dev/null
grep -F "pto.tmov ins(" "${ROUNDTRIP}" >/dev/null
[[ $(grep -Fc " fp " "${ROUNDTRIP}") -eq 2 ]]
grep -F "pto.tmov ins(" "${ROUNDTRIP}" \
  | grep -F ", %" \
  | grep -F "!pto.tile_buf<scaling" >/dev/null

# Parsing and verification prove that decoder-reconstructed segment metadata is
# valid, including records encoded with the legacy FP wire operand ordering.
"${PTOAS_BIN}" --pto-arch=a3 --emit-pto-ir "${ROUNDTRIP}" -o /dev/null
