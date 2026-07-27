#!/usr/bin/env bash
# CANN + PTOAS env for pipelining stage compile checks.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KERNEL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/cann-9.0.0}}"

# pipelining -> repro -> docs -> repo (three levels up from repro root).
PTOAS_ROOT="${PTOAS_ROOT:-$(cd "${KERNEL_ROOT}/../../.." && pwd)}"

# shellcheck disable=SC1091
source "${ASCEND_HOME_PATH}/bin/setenv.bash"
if [ -n "${PTOAS_ROOT}" ] && [ -f "${PTOAS_ROOT}/scripts/ptoas_env.sh" ]; then
  export PTOAS_ENV_SKIP_SMOKE_TEST="${PTOAS_ENV_SKIP_SMOKE_TEST:-1}"
  # shellcheck disable=SC1091
  source "${PTOAS_ROOT}/scripts/ptoas_env.sh"
  [ -d "${PTOAS_ROOT}/build/tools/ptoas" ] && export PATH="${PTOAS_ROOT}/build/tools/ptoas:${PATH}"
  [ -d "${PTOAS_ROOT}/install/bin" ] && export PATH="${PTOAS_ROOT}/install/bin:${PATH}"
fi

export ASCEND_NPU_ARCH="${ASCEND_NPU_ARCH:-dav-3510}"
export BISHENG="${BISHENG:-${ASCEND_HOME_PATH}/tools/bisheng_compiler/bin/bisheng}"
export ASCEND="${ASCEND_HOME_PATH}"
