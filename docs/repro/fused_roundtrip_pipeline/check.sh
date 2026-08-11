#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source /home/jzhuang/cann_installed/9.1.0-beta.3/cann/set_env.sh
MODE="${1:-all}"
case "$MODE" in
  compile) ACL_DEVICE_ID="${ACL_DEVICE_ID:-0}" python3 "$HERE/benchmark.py" --compile-only ;;
  benchmark|correctness) : "${ACL_DEVICE_ID:?set ACL_DEVICE_ID=0 or 1}"; task-submit --device "$ACL_DEVICE_ID" --run "python3 '$HERE/benchmark.py'" ;;
  all) ACL_DEVICE_ID="${ACL_DEVICE_ID:-0}" python3 "$HERE/benchmark.py" --compile-only; : "${ACL_DEVICE_ID:?set ACL_DEVICE_ID=0 or 1}"; task-submit --device "$ACL_DEVICE_ID" --run "python3 '$HERE/benchmark.py'" ;;
  *) echo "usage: $0 [compile|correctness|benchmark|all]" >&2; exit 2 ;;
esac
