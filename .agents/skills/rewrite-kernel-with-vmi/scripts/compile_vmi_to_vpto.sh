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
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
PTOAS_BIN="${PTOAS_BIN:-${REPO_ROOT}/install/bin/ptoas}"

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: $0 <input.pto> [output.mi.pto]" >&2
  exit 1
fi

INPUT="$1"
if [[ ! -f "${INPUT}" ]]; then
  echo "Input file not found: ${INPUT}" >&2
  exit 1
fi

if [[ ! -x "${PTOAS_BIN}" ]]; then
  echo "ptoas not found or not executable: ${PTOAS_BIN}" >&2
  exit 1
fi

if [[ $# -eq 2 ]]; then
  OUTPUT="$2"
else
  if [[ "${INPUT}" == *.vmi.pto ]]; then
    OUTPUT="${INPUT%.vmi.pto}.mi.pto"
  else
    OUTPUT="${INPUT%.pto}.mi.pto"
  fi
fi

echo "Compiling ${INPUT} -> ${OUTPUT}"
"${PTOAS_BIN}" \
  --pto-arch=a5 \
  --pto-backend=vpto \
  --pto-level=level3 \
  --emit-vpto \
  -o "${OUTPUT}" \
  "${INPUT}"

echo "Done: ${OUTPUT}"
