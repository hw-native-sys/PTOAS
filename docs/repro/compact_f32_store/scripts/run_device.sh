#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set -euo pipefail

DEVICE="${1:-0}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env.sh"

OUT="${REPRO_ROOT}/outputs/device"
mkdir -p "${OUT}"
RESULTS="${OUT}/results.txt"
: >"${RESULTS}"

reference_lib="$("${SCRIPT_DIR}/build_reference.sh")"

run_isolated() {
  local name="$1"
  local command="$2"
  local log="${OUT}/${name}.log"
  local status_file="${OUT}/${name}.status"
  local wrapped
  wrapped="source '${CANN_ENV}' >/dev/null 2>&1 && export PYTHONPATH='${PYTHONPATH}' PTOAS_BIN='${PTOAS_BIN}' NPU_TEST_DEVICE='npu:${DEVICE}' && ${command}"
  set +e
  if command -v task-submit >/dev/null 2>&1; then
    task-submit --device "${DEVICE}" --max-time 1200 --run "${wrapped}" >"${log}" 2>&1 &
  else
    bash -lc "${wrapped}" >"${log}" 2>&1 &
  fi
  local job_pid=$!
  while kill -0 "${job_pid}" 2>/dev/null; do
    sleep 5
  done
  wait "${job_pid}"
  local rc=$?
  set -e
  echo "${rc}" >"${status_file}"
}

run_isolated reference "'${PYTHON_BIN}' '${REPRO_ROOT}/fixtures/run_reference.py' --library '${reference_lib}' --device 'npu:${DEVICE}'"
ref_rc="$(<"${OUT}/reference.status")"
if [ "${ref_rc}" -ne 0 ] || ! grep -q "PASS native ASC/CCE" "${OUT}/reference.log"; then
  echo "FAIL native ASC/CCE reference (rc=${ref_rc})" | tee -a "${RESULTS}"
  tail -80 "${OUT}/reference.log" | tee -a "${RESULTS}"
  exit 1
fi
echo "PASS native ASC/CCE reference" | tee -a "${RESULTS}"

run_isolated desired "'${PYTHON_BIN}' '${REPRO_ROOT}/fixtures/desired_compact_store.py' --device 'npu:${DEVICE}'"
desired_rc="$(<"${OUT}/desired.status")"
if [ "${desired_rc}" -eq 0 ] && grep -q "PASS desired compact VMI" "${OUT}/desired.log"; then
  echo "PASS desired compact VMI (issue is fixed)" | tee -a "${RESULTS}"
elif grep -q "507035" "${OUT}/desired.log"; then
  echo "REPRODUCED desired compact VMI device fault 507035" | tee -a "${RESULTS}"
else
  echo "FAIL desired compact VMI with an unexpected result (rc=${desired_rc})" | tee -a "${RESULTS}"
  tail -100 "${OUT}/desired.log" | tee -a "${RESULTS}"
  exit 1
fi

run_isolated workaround "'${PYTHON_BIN}' '${REPRO_ROOT}/fixtures/workaround_padded_store.py' --device 'npu:${DEVICE}'"
workaround_rc="$(<"${OUT}/workaround.status")"
if [ "${workaround_rc}" -ne 0 ] || ! grep -q "PASS padded VMI workaround" "${OUT}/workaround.log"; then
  echo "FAIL padded VMI workaround (rc=${workaround_rc})" | tee -a "${RESULTS}"
  tail -100 "${OUT}/workaround.log" | tee -a "${RESULTS}"
  exit 1
fi
echo "PASS padded VMI workaround" | tee -a "${RESULTS}"

echo "Device checks completed in independent processes. Logs: ${OUT}" | tee -a "${RESULTS}"
