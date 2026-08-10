#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PYTHONPATH="${ROOT}/ptodsl:${PYTHONPATH:-}" "${PYTHON_BIN}" \
  "${ROOT}/docs/repro/indirect_buffer_abi/fixtures/fixed_arguments.py" --emit-mlir \
  | grep -q '!pto.ptr<f32, gm>'
grep -q 'DESIRED API' "${ROOT}/docs/repro/indirect_buffer_abi/fixtures/desired_pointer_table.py"
echo "PASS: fixed-pointer control compiles; desired indirect ABI is documented"
