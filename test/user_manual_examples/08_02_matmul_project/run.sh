#!/usr/bin/env bash

set -euo pipefail

RUN_MODE="${RUN_MODE:-npu}"
SOC_VERSION="${SOC_VERSION:-Ascend910B1}"
BUILD_DIR="${BUILD_DIR:-build}"
PTO_ARCH="${PTO_ARCH:-a3}"
PTOAS_FLAGS="${PTOAS_FLAGS:---enable-insert-sync}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PTO_AS_ROOT="$(cd "${ROOT_DIR}/../../.." && pwd)"

bootstrap_env() {
    if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
        for env_script in \
            /usr/local/Ascend/ascend-toolkit/set_env.sh \
            /usr/local/Ascend/cann/set_env.sh
        do
            if [[ -f "${env_script}" ]]; then
                # shellcheck disable=SC1090
                source "${env_script}" >/dev/null 2>&1
                break
            fi
        done
    fi

    if [[ -L "${HOME}/.ccache" ]]; then
        mkdir -p "$(readlink "${HOME}/.ccache")/tmp"
    else
        mkdir -p "${HOME}/.ccache/tmp"
    fi

    if [[ -z "${PTO_ISA_ROOT:-}" ]]; then
        for candidate in \
            "${PTO_AS_ROOT}/../pto-isa" \
            "${PTO_AS_ROOT}/../VPTO/pto-isa" \
            "${PWD}/../pto-isa"
        do
            if [[ -d "${candidate}" ]]; then
                export PTO_ISA_ROOT="${candidate}"
                break
            fi
        done
    fi

    if [[ -z "${PTO_ISA_ROOT:-}" || ! -d "${PTO_ISA_ROOT}" ]]; then
        echo "[ERROR] PTO_ISA_ROOT is not set. Please export PTO_ISA_ROOT=/path/to/pto-isa"
        exit 1
    fi
}

bootstrap_env

PTOAS_BIN="${PTOAS_BIN:-$(command -v ptoas || true)}"
if [[ -z "${PTOAS_BIN}" ]]; then
    for candidate in \
        "${PTO_AS_ROOT}/build/tools/ptoas/ptoas" \
        /usr/local/bin/ptoas-bin/ptoas
    do
        if [[ -x "${candidate}" ]]; then
            PTOAS_BIN="${candidate}"
            break
        fi
    done
fi

if [[ -z "${PTOAS_BIN}" || ! -x "${PTOAS_BIN}" ]]; then
    echo "[ERROR] Cannot find ptoas. Please export PTOAS_BIN=/path/to/ptoas or add ptoas to PATH"
    exit 1
fi

if ! command -v bisheng >/dev/null 2>&1; then
    echo "[ERROR] Cannot find bisheng. Please source Ascend set_env.sh first"
    exit 1
fi

cd "${ROOT_DIR}"
"${PTOAS_BIN}" "${ROOT_DIR}/kernel.pto" --pto-arch="${PTO_ARCH}" ${PTOAS_FLAGS} -o "${ROOT_DIR}/kernel.cpp"

mkdir -p "${ROOT_DIR}/${BUILD_DIR}"
cd "${ROOT_DIR}/${BUILD_DIR}"
cmake -DRUN_MODE="${RUN_MODE}" -DSOC_VERSION="${SOC_VERSION}" ..
make -j

cd "${ROOT_DIR}"
"${ROOT_DIR}/${BUILD_DIR}/matmul_project"
