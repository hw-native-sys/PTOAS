#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PTOAS_BIN="${PTOAS_BIN:-${ROOT}/build/tools/ptoas/ptoas}"
if [[ ! -x "${PTOAS_BIN}" ]]; then PTOAS_BIN="$(command -v ptoas || true)"; fi
if [[ ! -x "${PTOAS_BIN}" && -x "${ROOT}/../ptoas-main/build/tools/ptoas/ptoas" ]]; then
  PTOAS_BIN="${ROOT}/../ptoas-main/build/tools/ptoas/ptoas"
fi
if [[ ! -x "${PTOAS_BIN}" ]]; then echo "error: build ptoas or set PTOAS_BIN" >&2; exit 2; fi
OUT="$(mktemp)"
env -u PYTHONPATH "${PYTHON_BIN:-python3}" "${PTOAS_BIN}" --pto-arch=a5 --pto-backend=vpto --emit-vpto \
  "${ROOT}/docs/repro/grouped_scale_pipeline/fixtures/grouped_scale_vmi.pto" >"${OUT}"
grep -q 'pto.vcgmax' "${OUT}"
grep -Eq 'pto.vselr|pto.vdup|pto.vlds.*BRC' "${OUT}"
grep -q 'pto.vcvt' "${OUT}"
echo "PASS: grouped scale fixture lowers to VPTO"
