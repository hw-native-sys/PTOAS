#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; MODE="${1:-all}"; OUT="${HERE}/outputs"; mkdir -p "${OUT}"
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
CANN_SET_ENV="${CANN_ENV:-${ASCEND_HOME_PATH}/set_env.sh}"
task_run() { task-submit --device "$ACL_DEVICE_ID" --run "source '$CANN_SET_ENV'; ACL_DEVICE_ID=$ACL_DEVICE_ID '$PYTHON_BIN' '$HERE/benchmark.py'"; }
compile() {
  PYTHONPATH="${HERE}/fixtures" "${PYTHON_BIN}" "${HERE}/fixtures/fixed_arguments.py" --emit-mlir > "${OUT}/fixed.mlir"
  grep -q '!pto.ptr<f32, gm>' "${OUT}/fixed.mlir"
  grep -q 'pto.mte_gm_ub' "${OUT}/fixed.mlir"
  PYTHONPATH="${HERE}/fixtures" "${PYTHON_BIN}" "${HERE}/fixtures/pointer_table_abi.py" --emit-mlir > "${OUT}/pointer_table.mlir"
  grep -q 'pto.ld_dev' "${OUT}/pointer_table.mlir"
  grep -q 'pto.castptr' "${OUT}/pointer_table.mlir"
  grep -q 'pto.mte_gm_ub' "${OUT}/pointer_table.mlir"
  ACL_DEVICE_ID="${ACL_DEVICE_ID:-}" "${PYTHON_BIN}" "${HERE}/benchmark.py" --compile-only
  echo "PASS: stream-launchable direct-pointer CCE and stacked-buffer VMI libraries built; GM i64 address-table ABI compiles"
}
run() { if [[ "${ACL_DEVICE_ID:-}" != "" ]]; then task_run | tee "${OUT}/results.txt"; else "${PYTHON_BIN}" "${HERE}/report.py" | tee "${OUT}/results.txt"; fi; }
case "$MODE" in compile) compile;; correctness|benchmark) run;; all) compile; run;; *) echo "usage: $0 [all|compile|correctness|benchmark]" >&2; exit 2;; esac
