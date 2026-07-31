#!/usr/bin/env bash
# CANN + PTOAS env for pad_brc compile checks.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPRO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/cann-9.0.0}}"

# PTOAS repo root: pad_brc -> repro -> docs -> repo (three levels up).
PTOAS_ROOT="${PTOAS_ROOT:-$(cd "${REPRO_ROOT}/../../.." && pwd)}"

# shellcheck disable=SC1091
if [ -f "${ASCEND_HOME_PATH}/bin/setenv.bash" ]; then
  source "${ASCEND_HOME_PATH}/bin/setenv.bash"
fi
if [ -n "${PTOAS_ROOT}" ] && [ -f "${PTOAS_ROOT}/scripts/ptoas_env.sh" ]; then
  export PTOAS_ENV_SKIP_SMOKE_TEST="${PTOAS_ENV_SKIP_SMOKE_TEST:-1}"
  # shellcheck disable=SC1091
  source "${PTOAS_ROOT}/scripts/ptoas_env.sh"
fi
for _tool_dir in \
  "${PTOAS_ROOT}/build/tools/pto-test-opt" \
  "${PTOAS_ROOT}/build/tools/ptoas" \
  "${PTOAS_ROOT}/install/bin"; do
  if [ -d "${_tool_dir}" ]; then
    export PATH="${_tool_dir}:${PATH}"
  fi
done

export ASCEND_NPU_ARCH="${ASCEND_NPU_ARCH:-dav-3510}"
export PTOAS_HOST_TARGET_CPU="${PTOAS_HOST_TARGET_CPU:-znver3}"
unset CCC_OVERRIDE_OPTIONS || true

echo "REPRO_ROOT=${REPRO_ROOT}"
echo "ASCEND_HOME_PATH=${ASCEND_HOME_PATH}"
echo "PTOAS_ROOT=${PTOAS_ROOT}"
echo "ptoas=$(command -v ptoas || echo MISSING)"
echo "pto-test-opt=$(command -v pto-test-opt || echo MISSING)"
