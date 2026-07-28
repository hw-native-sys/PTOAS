#!/usr/bin/env bash
# Build and run CCE stages=2 on NPU. Optional: MSOPPROF=1 for msopprof wrap.
set -euo pipefail
DEVICE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash "${DEVICE_DIR}/build.sh"
# shellcheck disable=SC1091
source "${DEVICE_DIR}/../scripts/env.sh"
HOST="${DEVICE_DIR}/build/scale_stages2_host"
if [ "${MSOPPROF:-0}" = "1" ] && command -v msopprof >/dev/null 2>&1; then
  msopprof --application="${HOST}" --output="${DEVICE_DIR}/build/msopprof" || true
fi
exec "${HOST}"
