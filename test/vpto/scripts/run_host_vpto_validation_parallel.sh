#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
VPTO_ROOT="${VPTO_ROOT:-${ROOT_DIR}/test/vpto/cases}"
CASES_ROOT="${CASES_ROOT:-${VPTO_ROOT}}"
SERIAL_SCRIPT="${SCRIPT_DIR}/run_host_vpto_validation.sh"

WORK_SPACE="${WORK_SPACE:-}"
CASE_NAME="${CASE_NAME:-}"
CASE_PREFIX="${CASE_PREFIX:-}"
JOBS="${JOBS:-}"

log() {
  echo "[$(date +'%F %T')] $*"
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

clean_tmp_inode_hotspots() {
  local -a targets=(
    /tmp/pto-microop-full
    /tmp/pto-microop-full-redownload
  )

  log "tmp inode usage before cleanup"
  df -ih /tmp

  for dir in "${targets[@]}"; do
    if [[ -e "${dir}" ]]; then
      log "remove ${dir}"
      rm -rf "${dir}"
    fi
  done

  log "tmp inode usage after cleanup"
  df -ih /tmp
}

clean_tmp_inode_hotspots

[[ -x "${SERIAL_SCRIPT}" ]] || die "missing serial validation script: ${SERIAL_SCRIPT}"
[[ -d "${CASES_ROOT}" ]] || die "missing cases root: ${CASES_ROOT}"
[[ -n "${WORK_SPACE}" ]] || die "WORK_SPACE is required"

if [[ -z "${JOBS}" ]]; then
  if command -v nproc >/dev/null 2>&1; then
    JOBS="$(nproc)"
  else
    JOBS=1
  fi
  if [[ "${JOBS}" -gt 1 ]]; then
    JOBS="$((JOBS / 2))"
  fi
fi

[[ "${JOBS}" =~ ^[0-9]+$ ]] || die "JOBS must be a positive integer, got: ${JOBS}"
[[ "${JOBS}" -ge 1 ]] || die "JOBS must be >= 1"

mkdir -p "${WORK_SPACE}"
WORK_SPACE="$(cd "${WORK_SPACE}" && pwd)"
SUMMARY_FILE="${WORK_SPACE}/parallel-summary.tsv"
RUNNER_LOG="${WORK_SPACE}/parallel-runner.log"

is_ptodsl_case_dir() {
  [[ -f "$1/kernel.py" ]]
}

is_unsupported_case_path() {
  [[ -f "$1/UNSUPPORTED" ]]
}

is_ptodsl_case_file() {
  local case_path="$1"
  local base_name
  base_name="$(basename "${case_path}")"
  [[ -f "${case_path}" ]] &&
    [[ "${case_path}" == *.py ]] &&
    [[ "${base_name}" != "golden.py" ]] &&
    [[ "${base_name}" != "compare.py" ]] &&
    [[ "${base_name}" != "kernel.py" ]] &&
    [[ "${base_name}" != _* ]]
}

is_ptodsl_case_path() {
  is_ptodsl_case_dir "$1" || is_ptodsl_case_file "$1"
}

validate_case_path() {
  local case_name="$1"
  local case_path="$2"

  if is_ptodsl_case_path "${case_path}"; then
    return 0
  fi
  [[ -d "${case_path}" ]] || die "case ${case_name} is neither a directory nor a PTODSL case file"
  for f in launch.cpp main.cpp golden.py compare.py; do
    [[ -f "${case_path}/${f}" ]] || die "case ${case_name} is missing ${f}"
  done
  [[ -f "${case_path}/kernel.pto" ]] || die "case ${case_name} must provide kernel.pto"
}

discover_cases() {
  local onboard_only_prefix="onboard-only/"

  if [[ -n "${CASE_NAME}" ]]; then
    if [[ "${DEVICE:-SIM}" == "SIM" && "${COMPILE_ONLY:-0}" != "1" &&
          "${CASE_NAME}" == "${onboard_only_prefix}"* ]]; then
      die "case ${CASE_NAME} is onboard-only and cannot run with DEVICE=SIM"
    fi
    local requested_path="${CASES_ROOT}/${CASE_NAME}"
    [[ -e "${requested_path}" ]] || die "unknown case: ${CASE_NAME}"
    validate_case_path "${CASE_NAME}" "${requested_path}"
    printf "%s\n" "${CASE_NAME}"
    return 0
  fi

  find "${CASES_ROOT}" -mindepth 1 -type d | sort | while read -r dir; do
    [[ -f "${dir}/kernel.pto" || -f "${dir}/kernel.py" ]] || continue
    local rel="${dir#${CASES_ROOT}/}"
    if is_unsupported_case_path "${dir}"; then
      continue
    fi
    if ! is_ptodsl_case_dir "${dir}"; then
      local ok=1
      for f in launch.cpp main.cpp golden.py compare.py; do
        if [[ ! -f "${dir}/${f}" ]]; then
          ok=0
          break
        fi
      done
      [[ "${ok}" -eq 1 ]] || continue
    fi
    if [[ "${DEVICE:-SIM}" == "SIM" && "${COMPILE_ONLY:-0}" != "1" &&
          "${rel}" == "${onboard_only_prefix}"* ]]; then
      continue
    fi
    if [[ -n "${CASE_PREFIX}" && "${rel}" != "${CASE_PREFIX}"* ]]; then
      continue
    fi
    printf "%s\n" "${rel}"
  done

  find "${CASES_ROOT}" -type f -name '*.py' | sort | while read -r path; do
    is_ptodsl_case_file "${path}" || continue
    local rel="${path#${CASES_ROOT}/}"
    if [[ "${DEVICE:-SIM}" == "SIM" && "${COMPILE_ONLY:-0}" != "1" &&
          "${rel}" == "${onboard_only_prefix}"* ]]; then
      continue
    fi
    if [[ -n "${CASE_PREFIX}" && "${rel}" != "${CASE_PREFIX}"* ]]; then
      continue
    fi
    printf "%s\n" "${rel}"
  done
}

if [[ "${DEVICE:-SIM}" == "SIM" && "${COMPILE_ONLY:-0}" != "1" &&
      "${CASE_NAME}" == onboard-only/* ]]; then
  die "case ${CASE_NAME} is onboard-only and cannot run with DEVICE=SIM"
fi

readarray -t CASES < <(discover_cases)
[[ "${#CASES[@]}" -gt 0 ]] || die "no cases found under ${CASES_ROOT}"

: > "${SUMMARY_FILE}"
: > "${RUNNER_LOG}"

declare -A PID_TO_CASE=()

launch_case() {
  local case_name="$1"

  log "[${case_name}] launch" | tee -a "${RUNNER_LOG}"
  (
    CASE_NAME="${case_name}" "${SERIAL_SCRIPT}"
  ) &

  local pid=$!
  PID_TO_CASE["${pid}"]="${case_name}"
}

reap_one() {
  local pid="$1"
  local case_name="${PID_TO_CASE[${pid}]}"
  local result="FAIL"
  local detail="1"

  if wait "${pid}"; then
    result="PASS"
    detail="0"
  fi

  printf '%s\t%s\t%s\n' "${case_name}" "${result}" "${detail}" >> "${SUMMARY_FILE}"
  log "[${case_name}] ${result} (${detail})" | tee -a "${RUNNER_LOG}"
  unset 'PID_TO_CASE['"${pid}"']'
}

log "=== VPTO Host Validation Parallel ===" | tee -a "${RUNNER_LOG}"
log "WORK_SPACE=${WORK_SPACE}" | tee -a "${RUNNER_LOG}"
log "CASE_NAME=${CASE_NAME:-<all>}" | tee -a "${RUNNER_LOG}"
log "CASE_PREFIX=${CASE_PREFIX:-<none>}" | tee -a "${RUNNER_LOG}"
log "JOBS=${JOBS}" | tee -a "${RUNNER_LOG}"
log "TOTAL_CASES=${#CASES[@]}" | tee -a "${RUNNER_LOG}"
if [[ -n "${SIM_LIB_DIR:-}" ]]; then
  log "SIM_LIB_DIR=${SIM_LIB_DIR}" | tee -a "${RUNNER_LOG}"
fi

next_index=0
while [[ "${next_index}" -lt "${#CASES[@]}" || "${#PID_TO_CASE[@]}" -gt 0 ]]; do
  while [[ "${next_index}" -lt "${#CASES[@]}" && "${#PID_TO_CASE[@]}" -lt "${JOBS}" ]]; do
    launch_case "${CASES[${next_index}]}"
    next_index="$((next_index + 1))"
  done

  if [[ "${#PID_TO_CASE[@]}" -eq 0 ]]; then
    continue
  fi

  while true; do
    for pid in "${!PID_TO_CASE[@]}"; do
      if ! kill -0 "${pid}" 2>/dev/null; then
        reap_one "${pid}"
        break 2
      fi
    done
    sleep 1
  done
done

pass_count="$(awk -F '\t' '$2 == "PASS" {count++} END {print count + 0}' "${SUMMARY_FILE}")"
fail_count="$(awk -F '\t' '$2 != "PASS" {count++} END {print count + 0}' "${SUMMARY_FILE}")"

log "PASS=${pass_count} FAIL=${fail_count}" | tee -a "${RUNNER_LOG}"
log "summary: ${SUMMARY_FILE}" | tee -a "${RUNNER_LOG}"

if [[ "${fail_count}" -ne 0 ]]; then
  die "parallel validation finished with ${fail_count} failing case(s)"
fi

log "All ${pass_count} case(s) passed" | tee -a "${RUNNER_LOG}"
