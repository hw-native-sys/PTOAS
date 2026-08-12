#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; MODE="${1:-all}"
if [[ -n "${CANN_ENV:-}" ]]; then source "$CANN_ENV"; elif [[ -z "${ASCEND_HOME_PATH:-}" ]]; then echo "set CANN_ENV or source the CANN environment" >&2; exit 2; fi
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
compile() { "$PYTHON_BIN" "$HERE/benchmark.py" --compile-only; rg -q 'pto\.vmi\.vcvt' "$HERE/outputs/vmi.mlir"; echo "PASS: direct CCE and PTODSL VMI sources compile"; }
run() { if [[ -n "${ACL_DEVICE_ID:-}" ]]; then task-submit --device "$ACL_DEVICE_ID" --run "source '${CANN_ENV:-${ASCEND_HOME_PATH}/set_env.sh}'; ACL_DEVICE_ID=$ACL_DEVICE_ID '$PYTHON_BIN' '$HERE/benchmark.py'"; else "$PYTHON_BIN" "$HERE/benchmark.py"; fi; }
case "$MODE" in compile) compile;; correctness|benchmark) run;; all) compile; run;; *) echo "usage: $0 [all|compile|correctness|benchmark]" >&2; exit 2;; esac
