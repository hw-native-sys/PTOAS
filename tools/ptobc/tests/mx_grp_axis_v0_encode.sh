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
PYTHON_EXECUTABLE=${PYTHON_EXECUTABLE:-}
TESTDATA_DIR=${TESTDATA_DIR:-}
if [[ -z "${PTOBC_BIN}" || -z "${PTOAS_BIN}" ||
      -z "${PYTHON_EXECUTABLE}" || -z "${TESTDATA_DIR}" ]]; then
  echo "error: PTOBC_BIN, PTOAS_BIN, PYTHON_EXECUTABLE, and TESTDATA_DIR must be set" >&2
  exit 2
fi

IN="${TESTDATA_DIR}/mx_grp_axis_v0_roundtrip.pto"
TQUANT_ONLY_IN="${TESTDATA_DIR}/tquant_mx_v0_generic.pto"
OUT_DIR=${OUT_DIR:-"${PWD}/ptobc_mx_grp_axis_out"}
mkdir -p "${OUT_DIR}"

BC="${OUT_DIR}/mx_grp_axis_v0_roundtrip.ptobc"
ROUNDTRIP="${OUT_DIR}/mx_grp_axis_v0_roundtrip.roundtrip.pto"
TQUANT_ONLY_BC="${OUT_DIR}/tquant_mx_v0_generic.ptobc"

"${PTOBC_BIN}" encode "${IN}" -o "${BC}"
"${PTOBC_BIN}" decode "${BC}" -o "${ROUNDTRIP}"
"${PTOBC_BIN}" encode "${TQUANT_ONLY_IN}" -o "${TQUANT_ONLY_BC}"

grep -F "pto.tquant.mx ins(" "${ROUNDTRIP}" >/dev/null
grep -F "grpAxis = #pto<mx_group_axis axis0>" "${ROUNDTRIP}" >/dev/null
grep -F "interleave = true" "${ROUNDTRIP}" >/dev/null
[[ $(grep -Fc "pto.tmov ins(" "${ROUNDTRIP}") -eq 2 ]]
grep -F "!pto.tile_buf<scaling," "${ROUNDTRIP}" >/dev/null
grep -F "!pto.tile_buf<vec, 2x16xui8>" "${ROUNDTRIP}" >/dev/null

"${PTOAS_BIN}" --pto-arch=a5 --emit-pto-ir "${ROUNDTRIP}" -o /dev/null

"${PYTHON_EXECUTABLE}" - "${BC}" "${TQUANT_ONLY_BC}" <<'PY'
from pathlib import Path
import sys

combined = Path(sys.argv[1]).read_bytes()
tquant_only = Path(sys.argv[2]).read_bytes()
generic = b"\xff\xff"
legacy_tmov_fp = b"\x39\x10"

if combined.count(generic) < 2:
    raise SystemExit("expected generic v0 records for tquant.mx and X-to-ZZ tmov")
if combined.count(legacy_tmov_fp) != 1:
    raise SystemExit("expected exactly one legacy scaling-FP tmov opcode")
if b"pto.tquant.mx" not in combined or b"pto.tmov" not in combined:
    raise SystemExit("generic op names are missing from the v0 binary")
if generic not in tquant_only or b"pto.tquant.mx" not in tquant_only:
    raise SystemExit("tquant.mx-only fixture did not use kOpcodeGeneric (0xFFFF)")
if legacy_tmov_fp in tquant_only:
    raise SystemExit("tquant.mx-only fixture selected the scaling-FP tmov opcode")
PY
