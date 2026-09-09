#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPRO_ROOT="$(cd "${HERE}/../.." && pwd)"
export PYTHONPATH="${REPRO_PYTHONPATH:-${REPRO_ROOT}/build/python:${REPRO_ROOT}/ptodsl:${TILELANG_ROOT:-}:${PYTHONPATH:-}}"
if [[ -z "${PTOAS_BIN:-}" && -x "${REPRO_ROOT}/build/tools/ptoas/ptoas" ]]; then
  export PTOAS_BIN="${REPRO_ROOT}/build/tools/ptoas/ptoas"
fi
export PTOAS_BIN="${PTOAS_BIN:-$(command -v ptoas || true)}"
export CANN_HOME="${CANN_HOME:-${ASCEND_HOME_PATH:-}}"
DEVICE="${REPRO_DEVICE:-${2:-0}}"
case "${1:-help}" in
  vmi-compile) exec python3 "${HERE}/run_vmi.py" --compile-only ;;
  vmi)
    set +e
    python3 "${HERE}/run_vmi.py" --device "npu:${DEVICE}"
    rc=$?
    set -e
    exit "${rc}"
    ;;
  asc) exec python3 "${HERE}/run_asc.py" --device "npu:${DEVICE}" ;;
  *) echo "usage: $0 {vmi-compile|vmi|asc} [device]" >&2; exit 64 ;;
esac
