#!/usr/bin/env bash
# PTOAS runtime environment bootstrap.
# Usage:
#   source scripts/ptoas_env.sh
#
# Optional overrides before sourcing:
#   export WORKSPACE_DIR=/path/to/workspace
#   export LLVM_BUILD_DIR=/path/to/llvm-project/build-shared
#   export PTO_SOURCE_DIR=/path/to/PTOAS
#   export PTO_INSTALL_DIR=/path/to/PTOAS/install
#   export PTO_PYTHON_BIN=/path/to/python3

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
	echo "This script must be sourced: source scripts/ptoas_env.sh"
	exit 1
fi

_PTOAS_ENV_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
_PTOAS_REPO_DIR="$(cd -- "${_PTOAS_ENV_SCRIPT_DIR}/.." && pwd)"

# Default layout:
#   <workspace>/
#     ├── PTOAS/
#     └── llvm-project/
export PTO_SOURCE_DIR="${PTO_SOURCE_DIR:-${_PTOAS_REPO_DIR}}"
export WORKSPACE_DIR="${WORKSPACE_DIR:-$(cd -- "${PTO_SOURCE_DIR}/.." && pwd)}"
export LLVM_SOURCE_DIR="${LLVM_SOURCE_DIR:-${WORKSPACE_DIR}/llvm-project}"
export LLVM_BUILD_DIR="${LLVM_BUILD_DIR:-${LLVM_SOURCE_DIR}/build-shared}"
export PTO_INSTALL_DIR="${PTO_INSTALL_DIR:-${PTO_SOURCE_DIR}/install}"
export PTO_ISA_PATH="${PTO_ISA_PATH:-${WORKSPACE_DIR}/pto-isa}"
export ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-${HOME}/cann}"

export MLIR_PYTHON_ROOT="${MLIR_PYTHON_ROOT:-${LLVM_BUILD_DIR}/tools/mlir/python_packages/mlir_core}"
export PTO_PYTHON_ROOT="${PTO_PYTHON_ROOT:-${PTO_INSTALL_DIR}}"
export PTO_PYTHON_BUILD_ROOT="${PTO_PYTHON_BUILD_ROOT:-${PTO_SOURCE_DIR}/build/python}"
export PYBIND11_CMAKE_DIR=$(python3 -m pybind11 --cmakedir)
export PTOAS_FLAGS="${PTOAS_FLAGS:-}"
export PTOAS_OUT_DIR=$PTO_SOURCE_DIR/build/output

_ptoas_prepend_path() {
	local var_name="$1"
	local value="$2"
	local current="${!var_name:-}"
	if [[ -z "${value}" ]]; then
		return 0
	fi
	if [[ ! -e "${value}" ]]; then
		return 0
	fi
	if [[ ":${current}:" == *":${value}:"* ]]; then
		return 0
	fi
	if [[ -z "${current}" ]]; then
		printf -v "${var_name}" '%s' "${value}"
	else
		printf -v "${var_name}" '%s:%s' "${value}" "${current}"
	fi
	export "${var_name}"
}

_ptoas_prepend_path PYTHONPATH "${MLIR_PYTHON_ROOT}"
_ptoas_prepend_path PYTHONPATH "${PTO_PYTHON_ROOT}"
_ptoas_prepend_path PYTHONPATH "${PTO_PYTHON_BUILD_ROOT}"

_ptoas_prepend_path LD_LIBRARY_PATH "${LLVM_BUILD_DIR}/lib"
_ptoas_prepend_path LD_LIBRARY_PATH "${PTO_INSTALL_DIR}/lib"
_ptoas_prepend_path LD_LIBRARY_PATH "${PTO_SOURCE_DIR}/build/lib"

_ptoas_prepend_path PATH "${PTO_SOURCE_DIR}/build/tools/ptoas"

if [[ -n "${PTO_PYTHON_BIN:-}" && -x "${PTO_PYTHON_BIN}" ]]; then
	alias ptoas-python="${PTO_PYTHON_BIN}"
fi

echo "[ptoas_env] PTO_SOURCE_DIR=${PTO_SOURCE_DIR}"
echo "[ptoas_env] LLVM_BUILD_DIR=${LLVM_BUILD_DIR}"
echo "[ptoas_env] PTO_INSTALL_DIR=${PTO_INSTALL_DIR}"
echo "[ptoas_env] PTO_ISA_PATH=${PTO_ISA_PATH}"
echo "[ptoas_env] ASCEND_HOME_PATH=${ASCEND_HOME_PATH}"
echo "[ptoas_env] PATH/PYTHONPATH/LD_LIBRARY_PATH updated"

unset _PTOAS_ENV_SCRIPT_DIR
unset _PTOAS_REPO_DIR
