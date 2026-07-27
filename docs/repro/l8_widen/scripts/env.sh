#!/usr/bin/env bash
# CANN + PTOAS env for l8_widen compile checks.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPRO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/cann-9.0.0}}"

# PTOAS repo root: l8_widen -> repro -> docs -> repo (three levels up).
PTOAS_ROOT="${PTOAS_ROOT:-$(cd "${REPRO_ROOT}/../../.." && pwd)}"

# shellcheck disable=SC1091
source "${ASCEND_HOME_PATH}/bin/setenv.bash"
if [ -n "${PTOAS_ROOT}" ] && [ -f "${PTOAS_ROOT}/scripts/ptoas_env.sh" ]; then
  export PTOAS_ENV_SKIP_SMOKE_TEST="${PTOAS_ENV_SKIP_SMOKE_TEST:-1}"
  # shellcheck disable=SC1091
  source "${PTOAS_ROOT}/scripts/ptoas_env.sh"
  for _tool_dir in \
    "${PTOAS_ROOT}/build/tools/pto-test-opt" \
    "${PTOAS_ROOT}/build/tools/ptoas"; do
    if [ -d "${_tool_dir}" ]; then
      export PATH="${_tool_dir}:${PATH}"
    fi
  done
fi

export ASCEND_NPU_ARCH="${ASCEND_NPU_ARCH:-dav-3510}"
export PTOAS_HOST_TARGET_CPU="${PTOAS_HOST_TARGET_CPU:-znver3}"
unset CCC_OVERRIDE_OPTIONS || true
