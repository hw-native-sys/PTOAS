#!/usr/bin/env bash
# CANN + PTOAS/ptodsl env for ub_table_scatter_gather numeric checks.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPRO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-${ASCEND_TOOLKIT_HOME:-/home/jzhuang/cann_installed/9.1.0-beta.3/cann-9.1.0-beta.3}}"

# PTOAS repo root: slug -> repro -> docs -> repo
PTOAS_ROOT="${PTOAS_ROOT:-$(cd "${REPRO_ROOT}/../../.." && pwd)}"

# shellcheck disable=SC1091
if [ -f "${ASCEND_HOME_PATH}/bin/setenv.bash" ]; then
  source "${ASCEND_HOME_PATH}/bin/setenv.bash"
elif [ -f "${ASCEND_HOME_PATH}/../set_env.sh" ]; then
  # shellcheck disable=SC1091
  source "${ASCEND_HOME_PATH}/../set_env.sh"
elif [ -f "/home/jzhuang/cann_installed/9.1.0-beta.3/cann/set_env.sh" ]; then
  # shellcheck disable=SC1091
  source "/home/jzhuang/cann_installed/9.1.0-beta.3/cann/set_env.sh"
fi

if [ -n "${PTOAS_ROOT}" ] && [ -f "${PTOAS_ROOT}/scripts/ptoas_env.sh" ]; then
  export PTOAS_ENV_SKIP_SMOKE_TEST="${PTOAS_ENV_SKIP_SMOKE_TEST:-1}"
  # shellcheck disable=SC1091
  source "${PTOAS_ROOT}/scripts/ptoas_env.sh"
fi

# Prefer this tree's ptodsl on PYTHONPATH
export PYTHONPATH="${PTOAS_ROOT}/ptodsl${PYTHONPATH:+:${PYTHONPATH}}"
export NPU_TEST_DEVICE="${NPU_TEST_DEVICE:-npu:1}"
export ASCEND_NPU_ARCH="${ASCEND_NPU_ARCH:-dav-3510}"
unset CCC_OVERRIDE_OPTIONS || true

echo "REPRO_ROOT=${REPRO_ROOT}"
echo "ASCEND_HOME_PATH=${ASCEND_HOME_PATH}"
echo "PTOAS_ROOT=${PTOAS_ROOT}"
echo "NPU_TEST_DEVICE=${NPU_TEST_DEVICE}"
echo "python=$(command -v python || command -v python3 || echo MISSING)"
