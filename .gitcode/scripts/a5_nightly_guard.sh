#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
# Three-stage guard for the A5 nightly.
#
#   --stage start   00:05  make sure the 00:00 run actually starts and survives
#                          its start watch window (30 min by default); restart
#                          it when it does not.
#   --stage middle  02:00  if it is running, judge progress; if it already died,
#                          restart and watch the restart for ten minutes.
#   --stage end     04:30  wait for completion (bounded) and report the outcome.
#
# The guard only takes automatable actions (start/restart, wait, report). Every
# decision is written to the guard log and, when A5_FEISHU_WEBHOOK_URL is set,
# pushed to Feishu so a human can take over for anything it cannot fix.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=a5_notify.sh
source "${SCRIPT_DIR}/a5_notify.sh"
REPO_DIR="${PTOAS_CI_REPO_DIR:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
CI_ROOT="${PTOAS_CI_ROOT:-${REPO_DIR}/.work/a5-board-ci}"
SERVICE="${PTOAS_A5_GUARD_SERVICE:-ptoas-a5-nightly.service}"
STATE="${CI_ROOT}/state/last-run.env"
MAX_RESTARTS="${PTOAS_A5_GUARD_MAX_RESTARTS:-2}"
STAGE=""
DRY_RUN=0
# Give up waiting for the end stage after this many minutes (04:30 + 3h).
END_DEADLINE_MIN="${PTOAS_A5_GUARD_END_DEADLINE_MIN:-180}"
# A run that has covered less than this percentage at the middle checkpoint is
# reported as slow. 710 cases in ~4 h means ~50% by 02:00.
SLOW_PERCENT="${PTOAS_A5_GUARD_SLOW_PERCENT:-30}"
# Observation windows; overridable so the logic can be exercised quickly.
START_WATCH_SEC="${PTOAS_A5_GUARD_START_WATCH_SEC:-1800}"
RESTART_WATCH_SEC="${PTOAS_A5_GUARD_RESTART_WATCH_SEC:-600}"

log() { printf '[%s] %s\n' "$(date +'%F %T')" "$*"; }
die() {
  log "ERROR: $*" >&2
  exit 1
}

alert() {
  local text="$1"
  log "ALERT: ${text}"
  a5_notify_text "${text}"
}

usage() {
  cat <<EOF
Usage: $(basename "$0") --stage start|middle|end [--ci-root DIR] [--dry-run]

  --stage start   verify the nightly starts and survives the start watch window
  --stage middle  check progress at 02:00, restart a dead run
  --stage end     wait for completion and report
  --dry-run       print decisions without starting the service
Environment:
  PTOAS_CI_ROOT, A5_FEISHU_WEBHOOK_URL, PTOAS_A5_GUARD_SERVICE,
  PTOAS_A5_GUARD_MAX_RESTARTS, PTOAS_A5_GUARD_SLOW_PERCENT
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage) STAGE="${2:-}"; shift 2 ;;
    --ci-root) CI_ROOT="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) die "unknown argument: $1 (see --help)" ;;
  esac
done

case "${STAGE}" in
  start|middle|end) ;;
  *) usage >&2; die "--stage is required" ;;
esac

STATE="${CI_ROOT}/state/last-run.env"

RUN_ID=""; RC="missing"; SCOPE=""; START=""; END=""; WORK_ROOT=""; DEVICE=""; COMMIT=""; RESULTS_TSV=""; REPORTED=0
load_state() {
  RUN_ID=""; RC="missing"; SCOPE=""; START=""; END=""; WORK_ROOT=""; DEVICE=""; COMMIT=""; RESULTS_TSV=""; REPORTED=0
  if [[ -f "${STATE}" ]]; then
    local line
    while IFS= read -r line; do
      [[ -z "${line}" ]] && continue
      [[ "${line}" =~ ^[A-Za-z_][A-Za-z0-9_]*=.*$ ]] \
        || die "invalid state line in ${STATE}: ${line}"
      case "${line}" in
        *'$('*|*'`'*|*';'*|*'|'*|*'&'*|*'>'*|*'<'*)
          die "unsafe state line in ${STATE}" ;;
      esac
    done < "${STATE}"
    # shellcheck disable=SC1090
    source "${STATE}"
  fi
}

# A successful state from a previous night (or from a manual/prepare run in a
# different state file) must never be reported as tonight's PASS.
is_current_nightly() {
  [[ "${SCOPE:-}" == "nightly" ]] || return 1
  [[ -n "${START}" ]] || return 1
  [[ "${PTOAS_A5_GUARD_ALLOW_PREVIOUS_DAY:-0}" == "1" ]] && return 0
  [[ "${START%% *}" == "$(date +'%F')" ]]
}

nightly_active() {
  local st
  st="$(systemctl is-active "${SERVICE}" 2>/dev/null || true)"
  [[ "${st}" == "active" || "${st}" == "activating" ]]
}

start_nightly() {
  if (( DRY_RUN )); then
    log "DRY-RUN: would start ${SERVICE}"
    return 0
  fi
  systemctl start --no-block "${SERVICE}"
}

wait_for_start() {
  local seconds="$1" waited=0
  while (( waited < seconds )); do
    nightly_active && return 0
    load_state
    [[ "${RC}" == "running" ]] && return 0
    sleep 10
    waited=$((waited + 10))
  done
  return 1
}

progress() {
  local blog="${WORK_ROOT}/board-validation.log"
  local manifest="${WORK_ROOT}/payload/test/samples/expected_npu_validation_cases.txt"
  DONE=0
  EXPECTED=0
  [[ -n "${WORK_ROOT}" && -f "${blog}" ]] && DONE="$(grep -c '=== CASE' "${blog}" 2>/dev/null || true)"
  [[ -n "${WORK_ROOT}" && -f "${manifest}" ]] && EXPECTED="$(wc -l < "${manifest}" 2>/dev/null || true)"
  DONE="${DONE:-0}"
  EXPECTED="${EXPECTED:-0}"
}

elapsed_min() {
  local start_epoch now
  [[ -n "${START}" ]] || {
    echo 0
    return
  }
  start_epoch="$(date -d "${START}" +%s 2>/dev/null || echo 0)"
  now="$(date +%s)"
  (( start_epoch > 0 )) && echo $(( (now - start_epoch) / 60 )) || echo 0
}

summarize_results() {
  local tsv="${RESULTS_TSV:-}"
  [[ -n "${tsv}" && -f "${tsv}" ]] || tsv="${WORK_ROOT}/remote_npu_validation_results.tsv"
  if [[ ! -f "${tsv}" ]]; then
    echo "results: missing (${tsv})"
    return 0
  fi
  awk -F'\t' 'NR>1 {c[$2]++} END {for (k in c) printf "%s=%d ", k, c[k]; print ""}' "${tsv}"
  local fails
  fails="$(awk -F'\t' '$2=="FAIL" {print $1": "$4}' "${tsv}" | head -5)"
  [[ -n "${fails}" ]] && printf 'failing cases:\n%s\n' "${fails}"
  return 0
}

# Classify why the last attempt died and, for the one failure we can fix from
# here, fix it. A guard that only restarts and says "rc=1" wastes a night: the
# alert must carry the cause and what happens next.
# Sets FAIL_KIND / FAIL_CAUSE / FAIL_ACTION / FAIL_NOTE.
classify_dead_run() {
  local log="${WORK_ROOT}/nightly.log"
  FAIL_KIND="unknown"; FAIL_CAUSE="未识别的失败"; FAIL_ACTION="看日志定位"; FAIL_NOTE=""
  IFS=$'	' read -r FAIL_KIND _ FAIL_CAUSE FAIL_ACTION < <(a5_classify_failure "${log}")
  if [[ "${FAIL_KIND}" == "disk-full" ]]; then
    local freed
    freed="$(reclaim_disk)"
    FAIL_NOTE="已清理旧运行的重目录（${freed:-0} 个），清理后 $(df -h "${CI_ROOT}" | awk 'NR==2 {print $3"/"$2" 已用"}')"
    FAIL_ACTION="清理后重跑；若再次写满，需要主机 owner 释放 /data（共享盘）"
  fi
}

# Remove npu-validation/payload from every previous run: they are regenerated
# and each costs ~17 GiB. Prints how many were removed.
reclaim_disk() {
  local dir sub removed=0
  while IFS= read -r dir; do
    [[ -n "${dir}" ]] || continue
    [[ "${dir}" == "${WORK_ROOT}" ]] && continue
    for sub in npu-validation payload; do
      [[ -n "${dir}" && "${dir}" != "/" ]] || continue
      [[ -d "${dir}/${sub}" ]] || continue
      rm -rf -- "${dir}/${sub}" 2>/dev/null && removed=$((removed + 1))
    done
  done < <(find "${CI_ROOT}/work" -mindepth 1 -maxdepth 1 -type d -name '20*-*' 2>/dev/null)
  printf '%s' "${removed}"
}

stage_start() {
  log "guard[start]: waiting for the 00:00 run to appear"
  if ! wait_for_start 300; then
    alert "PTOAS A5 板测：00:05 仍未见任务启动，正在手动拉起"
    start_nightly
    wait_for_start 120 || {
      alert "PTOAS A5 板测：手动拉起后仍未运行，需人工介入"
      exit 1
    }
  fi

  log "guard[start]: run is up; watching the first $((START_WATCH_SEC / 60)) min"
  if (( DRY_RUN )); then
    exit 0
  fi
  sleep "${START_WATCH_SEC}"

  load_state
  if nightly_active; then
    progress
    log "guard[start]: OK, still running (${DONE}/${EXPECTED} cases, $(elapsed_min) min)"
    exit 0
  fi
  if [[ "${RC}" == "0" ]] && is_current_nightly; then
    log "guard[start]: run ${RUN_ID} already finished successfully; nothing to restart"
    exit 0
  fi

  local attempt
  classify_dead_run
  for attempt in $(seq 1 "${MAX_RESTARTS}"); do
    alert "PTOAS A5 板测：启动观察窗（$((START_WATCH_SEC / 60)) 分钟）内退出（rc=${RC}，run ${RUN_ID}），第 ${attempt} 次重启
原因：${FAIL_CAUSE}（kind=${FAIL_KIND}）${FAIL_NOTE:+
${FAIL_NOTE}}
处置：${FAIL_ACTION}
log: ${WORK_ROOT}/nightly.log"
    start_nightly
    sleep "${RESTART_WATCH_SEC}"
    load_state
    if nightly_active; then
      log "guard[start]: restart ${attempt} survived ${RESTART_WATCH_SEC}s"
      exit 0
    fi
  done
  alert "PTOAS A5 板测：连续 ${MAX_RESTARTS} 次重启后仍未运行，需要人工介入
原因：${FAIL_CAUSE}（kind=${FAIL_KIND}）${FAIL_NOTE:+
${FAIL_NOTE}}
需要你：${FAIL_ACTION}
run: ${RUN_ID}  log: ${WORK_ROOT:-${CI_ROOT}}/nightly.log"
  exit 1
}

stage_middle() {
  load_state
  if nightly_active; then
    progress
    local mins pct
    mins="$(elapsed_min)"
    pct=0
    (( EXPECTED > 0 )) && pct=$(( DONE * 100 / EXPECTED ))
    log "guard[middle]: running ${DONE}/${EXPECTED} (${pct}%) after ${mins} min " \
        "(run ${RUN_ID} commit ${COMMIT:-unknown})"
    if (( EXPECTED > 0 && mins > 90 && pct < SLOW_PERCENT )); then
      alert "PTOAS A5 板测进度偏慢：${mins} 分钟只完成 ${pct}%（${DONE}/${EXPECTED}），可能有卡竞争
run: ${RUN_ID}
log: ${WORK_ROOT}/board-validation.log"
    fi
    exit 0
  fi

  if [[ "${RC}" == "0" ]] && is_current_nightly; then
    log "guard[middle]: run ${RUN_ID} already finished successfully"
    exit 0
  fi

  classify_dead_run
  if [[ "${RC}" == "0" ]]; then
    alert "PTOAS A5 板测：没有今晚的有效运行记录（最近一次 scope=${SCOPE:-unknown}，run ${RUN_ID}），重新拉起并观察 $((RESTART_WATCH_SEC / 60)) 分钟"
  else
    alert "PTOAS A5 板测：02:00 前已退出（rc=${RC}，run ${RUN_ID}），重启并观察 $((RESTART_WATCH_SEC / 60)) 分钟
原因：${FAIL_CAUSE}（kind=${FAIL_KIND}）${FAIL_NOTE:+
${FAIL_NOTE}}
处置：${FAIL_ACTION}
log: ${WORK_ROOT:-?}/nightly.log"
  fi
  start_nightly
  if (( DRY_RUN )); then
    exit 0
  fi
  sleep "${RESTART_WATCH_SEC}"
  load_state
  if nightly_active; then
    log "guard[middle]: restart survived ${RESTART_WATCH_SEC}s"
    exit 0
  fi
  classify_dead_run
  alert "PTOAS A5 板测：重启后 $((RESTART_WATCH_SEC / 60)) 分钟内再次退出（rc=${RC}），放弃自动重启
原因：${FAIL_CAUSE}（kind=${FAIL_KIND}）${FAIL_NOTE:+
${FAIL_NOTE}}
需要你：${FAIL_ACTION}
run: ${RUN_ID}  log: ${WORK_ROOT:-?}/nightly.log"
  exit 1
}

stage_end() {
  load_state
  log "guard[end]: run ${RUN_ID} rc=${RC} device=${DEVICE} ended=${END:-?}"
  local waited=0
  while nightly_active && (( waited < END_DEADLINE_MIN )); do
    sleep 60
    waited=$((waited + 1))
    (( waited % 10 == 0 )) && {
      load_state
      progress
      log "guard[end]: still running ${DONE}/${EXPECTED} cases, waited ${waited} min"
    }
  done

  load_state
  local summary
  summary="$(summarize_results)"
  if ! is_current_nightly; then
    alert "PTOAS A5 板测看护：没有今晚的有效运行记录（最近一次 scope=${SCOPE:-unknown}，run ${RUN_ID:-?}），无法判定结果
log: ${WORK_ROOT:-${CI_ROOT}}/nightly.log"
    exit 1
  fi
  if [[ "${RC}" == "0" ]]; then
    log "guard[end]: PASS after $(elapsed_min) min"
    if [[ "${REPORTED:-0}" == "1" ]]; then
      log "guard[end]: result already delivered by the run card; not repeating it"
      exit 0
    fi
    alert "PTOAS A5 板测完成：PASS（$(elapsed_min) 分钟）
${summary}
run: ${RUN_ID}
log: ${WORK_ROOT}/nightly.log"
    exit 0
  fi
  if nightly_active; then
    alert "PTOAS A5 板测看护：${END_DEADLINE_MIN} 分钟后仍在运行，超出预期
run: ${RUN_ID}
log: ${WORK_ROOT}/nightly.log"
    exit 1
  fi
  if [[ "${REPORTED:-0}" == "1" ]]; then
    log "guard[end]: failure already delivered by the run card; not repeating it"
    exit 1
  fi
  alert "PTOAS A5 板测失败：rc=${RC}
${summary}
run: ${RUN_ID}
log: ${WORK_ROOT}/nightly.log"
  exit 1
}

case "${STAGE}" in
  start) stage_start ;;
  middle) stage_middle ;;
  end) stage_end ;;
esac
