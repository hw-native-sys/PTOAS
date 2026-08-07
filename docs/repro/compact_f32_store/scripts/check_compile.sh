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

OUT="${REPRO_ROOT}/outputs/compile"
mkdir -p "${OUT}"
RESULTS="${OUT}/results.txt"
: >"${RESULTS}"

echo "PTOAS commit: $(git -C "${PTOAS_ROOT}" rev-parse HEAD)" | tee -a "${RESULTS}"
echo "CANN env: ${CANN_ENV}" | tee -a "${RESULTS}"
echo "Python: ${PYTHON_BIN}" | tee -a "${RESULTS}"
echo "PTOAS binary: ${PTOAS_BIN:-MISSING}" | tee -a "${RESULTS}"

reference_lib="$("${SCRIPT_DIR}/build_reference.sh")"
test -s "${reference_lib}"
echo "PASS native ASC/CCE reference build" | tee -a "${RESULTS}"

compile_vmi() {
  local fixture="$1"
  local stem="${fixture%.py}"
  "${PYTHON_BIN}" "${REPRO_ROOT}/fixtures/${fixture}" --compile-only \
    >"${OUT}/${stem}.log" 2>&1
  "${PYTHON_BIN}" "${REPRO_ROOT}/fixtures/${fixture}" --emit-mlir \
    >"${OUT}/${stem}.mlir" 2>>"${OUT}/${stem}.log"
  grep -q "pto.vmi.vstore" "${OUT}/${stem}.mlir"
  echo "PASS ${fixture} frontend compile" | tee -a "${RESULTS}"
}

compile_vmi desired_compact_store.py
compile_vmi workaround_padded_store.py

test -n "${PTOAS_BIN}" && test -x "${PTOAS_BIN}"
for stem in desired_compact_store workaround_padded_store; do
  "${PTOAS_BIN}" --pto-arch=a5 --pto-backend=vpto --pto-level=level3 \
    --enable-tile-op-expand "${OUT}/${stem}.mlir" -o "${OUT}/${stem}.o" \
    >"${OUT}/${stem}.ptoas.log" 2>&1
  test -s "${OUT}/${stem}.o"
done
echo "PASS both VMI fixtures lower through PTOAS native compilation" | tee -a "${RESULTS}"

"${PTOAS_BIN}" --pto-arch=a5 --pto-backend=vpto --pto-level=level3 \
  --enable-tile-op-expand --emit-vpto "${OUT}/desired_compact_store.mlir" \
  -o "${OUT}/desired_compact_store.vpto" \
  >"${OUT}/desired_compact_store.emit-vpto.log" 2>&1
"${PYTHON_BIN}" - "${OUT}/desired_compact_store.vpto" <<'PY'
from pathlib import Path
import sys

text = Path(sys.argv[1]).read_text(encoding="utf-8")
if 'dist = "BRC_B32"' not in text:
    raise SystemExit("expected the explicit scalar broadcast load in emitted VPTO")
store_lines = [line.strip() for line in text.splitlines() if "pto.vsts" in line]
if not store_lines:
    raise SystemExit("expected a lowered pto.vsts in emitted VPTO")
if any('dist = "1PT_B32"' in line for line in store_lines):
    raise SystemExit("compiler behavior changed: desired store now lowers to 1PT_B32; rerun device test")
if not any("{dist =" not in line for line in store_lines):
    raise SystemExit("unexpected desired-store lowering; inspect emitted VPTO")
print("KNOWN BUG: group=1 broadcast result lowered to dense vsts, not 1PT_B32")
PY
echo "REPRODUCED compile-side dense-store mislowering" | tee -a "${RESULTS}"

echo "Compile checks passed. Logs: ${OUT}" | tee -a "${RESULTS}"
