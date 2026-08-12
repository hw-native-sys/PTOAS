#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="/home/jzhuang/.conda/envs/cann91_dev/bin/python"
CANN_ENV="${CANN_ENV:-/home/jzhuang/cann_installed/9.1.0-beta.3/cann-9.1.0-beta.3/set_env.sh}"
set +u; source "${CANN_ENV}"; set -u
MODE="${1:-all}"
case "$MODE" in
  compile) ACL_DEVICE_ID="${ACL_DEVICE_ID:-0}" "$PYTHON_BIN" "$HERE/benchmark.py" --compile-only ;;
  benchmark|correctness) : "${ACL_DEVICE_ID:?set ACL_DEVICE_ID=0 or 1}"; task-submit --device "$ACL_DEVICE_ID" --run "source '${CANN_ENV}'; ACL_DEVICE_ID=$ACL_DEVICE_ID '$PYTHON_BIN' '$HERE/benchmark.py' --profile" ;;
  all) ACL_DEVICE_ID="${ACL_DEVICE_ID:-0}" "$PYTHON_BIN" "$HERE/benchmark.py" --compile-only; : "${ACL_DEVICE_ID:?set ACL_DEVICE_ID=0 or 1}"; task-submit --device "$ACL_DEVICE_ID" --run "source '${CANN_ENV}'; ACL_DEVICE_ID=$ACL_DEVICE_ID '$PYTHON_BIN' '$HERE/benchmark.py' --profile" ;;
  *) echo "usage: $0 [compile|correctness|benchmark|all]" >&2; exit 2 ;;
esac
