#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# Shared environment for the compact f32 store reproducer.

set -euo pipefail

REPRO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PTOAS_ROOT="$(cd "${REPRO_ROOT}/../../.." && pwd)"
CANN_ENV="${CANN_ENV:-/home/jzhuang/cann_installed/9.1.0-beta.3/cann/set_env.sh}"
CONDA_ENV="${CONDA_ENV:-cann91_dev}"

if [ ! -f "${CANN_ENV}" ]; then
  echo "CANN environment script not found: ${CANN_ENV}" >&2
  return 1 2>/dev/null || exit 1
fi

# The toolkit script reads these variables while `set -u` is active.
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${PYTHONPATH:-}"
export CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH:-}"
# shellcheck disable=SC1090
source "${CANN_ENV}"

PYTHON_BIN="${PYTHON_BIN:-$(conda run -n "${CONDA_ENV}" which python)}"
export PATH="$(dirname "${PYTHON_BIN}"):${PATH}"
BISHENG="${BISHENG:-$(command -v bisheng || true)}"
PTOAS_BIN="${PTOAS_BIN:-${PTOAS_ROOT}/build/tools/ptoas/ptoas}"

if [ ! -x "${PTOAS_BIN}" ]; then
  # The source worktree may intentionally share a verified build from a sibling
  # checkout. Override PTOAS_BIN explicitly when using such a build.
  PTOAS_BIN="$(command -v ptoas || true)"
fi

export REPRO_ROOT PTOAS_ROOT CANN_ENV CONDA_ENV PYTHON_BIN BISHENG PTOAS_BIN
export PYTHONPATH="${PTOAS_ROOT}/build/python:${PTOAS_ROOT}/build/ptodsl:${PTOAS_ROOT}/ptodsl:${PTOAS_ROOT}/build/lib:${PYTHONPATH:-}"
export PTOAS_BIN
