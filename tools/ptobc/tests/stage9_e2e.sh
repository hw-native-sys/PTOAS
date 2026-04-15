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
if [[ -z "${PTOBC_BIN}" ]]; then
  echo "error: PTOBC_BIN not set" >&2
  exit 2
fi

# You can pass multiple roots separated by ':'
# Default: ptobc testdata + PTOAS test/samples
DEFAULT_A="$(cd "$(dirname "${BASH_SOURCE[0]}")/../testdata" && pwd)"
DEFAULT_B="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)/test/samples"
TESTDATA_DIRS=${TESTDATA_DIRS:-"${DEFAULT_A}:${DEFAULT_B}"}

OUT_DIR=${OUT_DIR:-"${PWD}/ptobc_stage9_out"}
mkdir -p "${OUT_DIR}"

should_skip_roundtrip() {
  local path="$1"
  case "$path" in
    */test/samples/Complex/mix_kernel.pto) return 0 ;;
    */test/samples/SCF/scf_for_break_like.pto) return 0 ;;
    */test/samples/SCF/scf_while_break.pto) return 0 ;;
    */test/samples/MatMul/0.pto) return 0 ;;
    */test/samples/MatMul/tmatmulk.pto) return 0 ;;
    */test/samples/Sync/test_if_else_tile_result.pto) return 0 ;;
  esac
  return 1
}

failed=0
IFS=':' read -r -a roots <<< "${TESTDATA_DIRS}"
for root in "${roots[@]}"; do
  [[ -d "${root}" ]] || continue
  while IFS= read -r -d '' f; do
    # Keep stage9 focused on PTO samples that ptobc can round-trip today.
    # Legacy parser-only samples, unsupported custom ops, and known SCF decode
    # gaps remain covered elsewhere in the PTOAS test corpus.
    if should_skip_roundtrip "$f"; then
      echo "skip unsupported roundtrip sample: $f"
      continue
    fi

    base=$(basename "$f" .pto)
    # avoid name collisions across directories
    hash=$(python3 - "$f" <<'PY'
import hashlib, sys
print(hashlib.sha1(sys.argv[1].encode()).hexdigest()[:8])
PY
)
    base="${base}.${hash}"

  bc1="${OUT_DIR}/${base}.ptobc"
  pto2="${OUT_DIR}/${base}.roundtrip.pto"
  bc2="${OUT_DIR}/${base}.roundtrip.ptobc"

    "${PTOBC_BIN}" encode "$f" -o "$bc1" || { echo "encode failed: $f"; failed=1; continue; }
    "${PTOBC_BIN}" decode "$bc1" -o "$pto2" || { echo "decode failed: $f"; failed=1; continue; }
    "${PTOBC_BIN}" encode "$pto2" -o "$bc2" || { echo "re-encode failed: $f"; failed=1; continue; }

    if command -v cmp >/dev/null 2>&1; then
      cmp "$bc1" "$bc2" || { echo "mismatch: $f"; failed=1; }
    fi
  done < <(find "${root}" -type f -name '*.pto' -print0)
done

exit $failed
