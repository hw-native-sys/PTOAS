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

IN="${TESTDATA_DIR}/tfillpad_legacy_v0_roundtrip.pto"
OUT_DIR=${OUT_DIR:-"${PWD}/ptobc_tfillpad_legacy_out"}
mkdir -p "${OUT_DIR}"

CANONICAL="${OUT_DIR}/tfillpad_canonical.ptobc"
EXPAND="${OUT_DIR}/tfillpad_expand_legacy.ptobc"
INPLACE="${OUT_DIR}/tfillpad_inplace_legacy.ptobc"

"${PTOBC_BIN}" encode "${IN}" -o "${CANONICAL}"

"${PYTHON_EXECUTABLE}" - <<'PY' "${CANONICAL}" "${EXPAND}" "${INPLACE}"
from pathlib import Path
import sys

canonical = Path(sys.argv[1]).read_bytes()
wire_opcode = b"\x23\x10"
if canonical.count(wire_opcode) != 1:
    raise SystemExit("expected exactly one canonical tfillpad opcode")

Path(sys.argv[2]).write_bytes(canonical.replace(wire_opcode, b"\x24\x10", 1))
Path(sys.argv[3]).write_bytes(canonical.replace(wire_opcode, b"\x25\x10", 1))
PY

for kind in expand inplace; do
  bc="${OUT_DIR}/tfillpad_${kind}_legacy.ptobc"
  roundtrip="${OUT_DIR}/tfillpad_${kind}_legacy.roundtrip.pto"
  "${PTOBC_BIN}" decode "${bc}" -o "${roundtrip}"
  grep -F "pto.tfillpad ins(" "${roundtrip}" >/dev/null
  "${PTOAS_BIN}" --pto-arch=a5 --emit-pto-ir "${roundtrip}" -o /dev/null
done
