#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="/home/jzhuang/.conda/envs/cann91_dev/bin/python"
set +u; source /home/jzhuang/cann_installed/9.1.0-beta.3/cann/set_env.sh; set -u
MODE="${1:-all}"
case "$MODE" in
  compile) ACL_DEVICE_ID="${ACL_DEVICE_ID:-0}" "$PYTHON_BIN" "$HERE/benchmark.py" --compile-only ;;
  benchmark|correctness) : "${ACL_DEVICE_ID:?set ACL_DEVICE_ID=0 or 1}"; task-submit --device "$ACL_DEVICE_ID" --run "source /home/jzhuang/cann_installed/9.1.0-beta.3/cann/set_env.sh; source /home/jzhuang/miniconda/bin/activate cann91_dev; python3 '$HERE/benchmark.py'" ;;
  all) ACL_DEVICE_ID="${ACL_DEVICE_ID:-0}" "$PYTHON_BIN" "$HERE/benchmark.py" --compile-only; : "${ACL_DEVICE_ID:?set ACL_DEVICE_ID=0 or 1}"; task-submit --device "$ACL_DEVICE_ID" --run "source /home/jzhuang/cann_installed/9.1.0-beta.3/cann/set_env.sh; source /home/jzhuang/miniconda/bin/activate cann91_dev; python '$HERE/benchmark.py'" ;;
  *) echo "usage: $0 [compile|correctness|benchmark|all]" >&2; exit 2 ;;
esac
