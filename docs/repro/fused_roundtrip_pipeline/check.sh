#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV="${CONDA_ENV:-cann91_dev}"
if [[ -n "${CANN_ENV:-}" ]]; then
  set +u
  source "${CANN_ENV}"
  set -u
elif [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
  echo "set CANN_ENV to the CANN 9.1.0-beta.3 set_env.sh (or source it first)" >&2
  exit 2
fi
if [[ "${CONDA_DEFAULT_ENV:-}" != "${CONDA_ENV}" ]] && command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV}"
fi
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
MODE="${1:-all}"
case "$MODE" in
  compile) ACL_DEVICE_ID="${ACL_DEVICE_ID:-0}" "$PYTHON_BIN" "$HERE/benchmark.py" --compile-only ;;
  benchmark|correctness) : "${ACL_DEVICE_ID:?set ACL_DEVICE_ID=0 or 1}"; task-submit --device "$ACL_DEVICE_ID" --run "ACL_DEVICE_ID=$ACL_DEVICE_ID '$PYTHON_BIN' '$HERE/benchmark.py' --profile" ;;
  all) ACL_DEVICE_ID="${ACL_DEVICE_ID:-0}" "$PYTHON_BIN" "$HERE/benchmark.py" --compile-only; : "${ACL_DEVICE_ID:?set ACL_DEVICE_ID=0 or 1}"; task-submit --device "$ACL_DEVICE_ID" --run "ACL_DEVICE_ID=$ACL_DEVICE_ID '$PYTHON_BIN' '$HERE/benchmark.py' --profile" ;;
  *) echo "usage: $0 [compile|correctness|benchmark|all]" >&2; exit 2 ;;
esac
