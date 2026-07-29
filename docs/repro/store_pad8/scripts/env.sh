#!/usr/bin/env bash
# CANN + PTOAS env for store_pad8 cannsim runs.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KERNEL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/cann-9.0.0}}"

# PTOAS repo root: store_pad8 -> repro -> docs -> repo (three levels up from repro root).
PTOAS_ROOT="${PTOAS_ROOT:-$(cd "${KERNEL_ROOT}/../../.." && pwd)}"

# shellcheck disable=SC1091
source "${ASCEND_HOME_PATH}/bin/setenv.bash"
if [ -n "${PTOAS_ROOT}" ] && [ -f "${PTOAS_ROOT}/scripts/ptoas_env.sh" ]; then
  export PTOAS_ENV_SKIP_SMOKE_TEST="${PTOAS_ENV_SKIP_SMOKE_TEST:-1}"
  # shellcheck disable=SC1091
  source "${PTOAS_ROOT}/scripts/ptoas_env.sh"
  # ptoas_env prepends build/tools/ptoas; that stub can clash with the pip/wheel
  # launcher (LLVM CL option double-register). Prefer the wheel on PATH.
  PATH="$(echo "${PATH}" | tr ':' '\n' | grep -v "/build/tools/ptoas$" | paste -sd:)"
  export PATH
fi

export ASCEND_NPU_ARCH="${ASCEND_NPU_ARCH:-dav-3510}"
ARCH=$(uname -m)-linux
CANN="${ASCEND_HOME_PATH}"
export LD_LIBRARY_PATH="${CANN}/${ARCH}/simulator/dav_3510/camodel:${CANN}/${ARCH}/simulator/dav_3510/lib:${CANN}/${ARCH}/lib64:${LD_LIBRARY_PATH:-}"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export PTOAS_HOST_TARGET_CPU="${PTOAS_HOST_TARGET_CPU:-znver3}"
unset CCC_OVERRIDE_OPTIONS || true
