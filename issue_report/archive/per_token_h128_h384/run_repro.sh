#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPRO_ROOT="$(cd "${HERE}/../.." && pwd)"
export PYTHONPATH="${REPRO_PYTHONPATH:-${REPRO_ROOT}/build/python:${REPRO_ROOT}/ptodsl:${TILELANG_ROOT:-}:${PYTHONPATH:-}}"
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
  *) echo "usage: $0 {vmi-compile|vmi} [device]" >&2; exit 64 ;;
esac
