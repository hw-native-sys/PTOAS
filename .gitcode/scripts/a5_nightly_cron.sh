#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
# Nightly entry point for a host that schedules the A5 board validation itself
# (i.e. no GitCode runner). It updates the persistent checkout and then hands
# over to a5_board_nightly.sh.
#
# This is the fallback path described in docs/ci/a5-nightly-board.md. Run it
# from a systemd timer (install_a5_nightly_timer.sh) or by hand.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=a5_notify.sh
source "${SCRIPT_DIR}/a5_notify.sh"

log() { printf '[%s] %s\n' "$(date +'%F %T')" "$*"; }
die() {
  log "ERROR: $*" >&2
  exit 1
}

# Failures before the driver starts (checkout, watchdog) still need to reach a
# human; the driver's own EXIT trap covers everything after it takes over.
alert() {
  local text="$1"
  log "ALERT: ${text}"
  a5_notify_text "${text}"
}

REPO_DIR="${PTOAS_CI_REPO_DIR:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
CI_ROOT="${PTOAS_CI_ROOT:-${REPO_DIR}/.work/a5-board-ci}"
GIT_URL="${PTOAS_GIT_URL:-https://gitcode.com/cann/pto-as.git}"
GIT_REF="${PTOAS_GIT_REF:-origin/master}"
# Branch tip whose code the nightly validates. It must be the canonical
# upstream, not whatever the checkout's origin happens to be: 238 clones a
# personal fork (for auth) whose master can be weeks behind. Set
# PTOAS_UPSTREAM_URL="" to fall back to origin/master.
UPSTREAM_URL="${PTOAS_UPSTREAM_URL-https://gitcode.com/cann/pto-as.git}"
UPSTREAM_REMOTE="${PTOAS_UPSTREAM_REMOTE:-canon}"
# Root on a shared host often carries several tenants' tokens for the same
# host in ~/.git-credentials, and git's store helper answers with the first
# match. Never inherit that helper: use the CI-owned file when configured, and
# fall back to anonymous access (public repos) otherwise.
GIT_CREDENTIALS_FILE="${PTOAS_GIT_CREDENTIALS_FILE:-}"
CHECK_LAST_RUN_HOURS=""
MAX_SUCCESS_AGE_HOURS=24
git_opts=(-c credential.helper=)
if [[ -n "${GIT_CREDENTIALS_FILE}" ]]; then
  [[ -f "${GIT_CREDENTIALS_FILE}" ]] \
    || die "PTOAS_GIT_CREDENTIALS_FILE is set but missing: ${GIT_CREDENTIALS_FILE}"
  git_opts=(-c credential.helper= -c "credential.helper=store --file=${GIT_CREDENTIALS_FILE}")
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-dir) REPO_DIR="$2"; shift 2 ;;
    --ci-root) CI_ROOT="$2"; shift 2 ;;
    --git-url) GIT_URL="$2"; shift 2 ;;
    --git-ref) GIT_REF="$2"; shift 2 ;;
    --check-last-run)
      [ -n "${2:-}" ] || die "--check-last-run needs a max age in hours"
      CHECK_LAST_RUN_HOURS="$2"; shift 2 ;;
    --max-success-age-hours)
      [ -n "${2:-}" ] || die "--max-success-age-hours needs a value"
      MAX_SUCCESS_AGE_HOURS="$2"; shift 2 ;;
    --help|-h)
      cat <<EOF
Usage: $(basename "$0") [--repo-dir DIR] [--ci-root DIR] [--git-url URL] [--git-ref REF]
                        [--check-last-run HOURS] [--max-success-age-hours HOURS]

Remaining arguments are forwarded to a5_board_nightly.sh.

  --check-last-run HOURS  Watchdog mode: read \${CI_ROOT}/state/last-run.env and
                          exit non-zero when the last nightly is missing, not
                          a nightly run, still marked running past HOURS, or
                          failed.
  --max-success-age-hours HOURS
                          Watchdog mode: a successful run is stale when it
                          ended more than HOURS ago (default: 24).

Environment:
  PTOAS_GIT_URL           Repo used for the checkout and --git-ref
                          (default: the canonical GitCode repo).
  PTOAS_GIT_REF           Branch/ref to check out (default: origin/master).
  PTOAS_UPSTREAM_URL      Canonical repo whose master is merged into a
                          branch checkout before the run
                          (default: the canonical GitCode repo). Set empty
                          to merge origin/master instead.
  PTOAS_UPSTREAM_REMOTE   Remote name used for PTOAS_UPSTREAM_URL
                          (default: canon).
EOF
      exit 0
      ;;
    --)
      shift
      break
      ;;
    *) break ;;
  esac
done

# Watchdog mode: no checkout, no lock, just report on the last run. It is meant
# for a second timer that fires after the nightly should have finished.
[[ -n "${CI_ROOT}" && "${CI_ROOT}" == /* && "${CI_ROOT}" != "/" ]] \
  || die "--ci-root must be an absolute path other than / (got: ${CI_ROOT})"
load_state_file() {
  local file="$1" line
  while IFS= read -r line; do
    [[ -z "${line}" ]] && continue
    [[ "${line}" =~ ^[A-Za-z_][A-Za-z0-9_]*=.*$ ]] \
      || die "invalid state line in ${file}: ${line}"
    case "${line}" in
      *'$('*|*'`'*|*';'*|*'|'*|*'&'*|*'>'*|*'<'*)
        die "unsafe state line in ${file}" ;;
    esac
  done < "${file}"
  # shellcheck disable=SC1090
  source "${file}"
}

if [[ -n "${CHECK_LAST_RUN_HOURS:-}" ]]; then
  [[ "${CHECK_LAST_RUN_HOURS}" =~ ^[0-9]+$ ]] \
    || die "--check-last-run must be a non-negative integer"
  [[ "${MAX_SUCCESS_AGE_HOURS}" =~ ^[0-9]+$ ]] \
    || die "--max-success-age-hours must be a non-negative integer"
  state="${CI_ROOT}/state/last-run.env"
  if [[ ! -f "${state}" ]]; then
    alert "PTOAS A5 板测看护：没有 nightly 运行记录 (${state})，夜间任务可能从未启动"
    exit 1
  fi
  # A truncated or hand-edited state file must not crash the watchdog with
  # "unbound variable"; default everything it may reference.
  RUN_ID=""; RC="missing"; SCOPE=""; START=""; END=""
  WORK_ROOT=""; FAILURE_KIND=""; PHASE=""
  load_state_file "${state}"
  if [[ "${SCOPE:-}" != "nightly" ]]; then
    alert "PTOAS A5 板测看护：最近一次记录不是 nightly（scope=${SCOPE:-unknown}，run ${RUN_ID:-?}），今晚可能从未启动"
    exit 1
  fi
  now="$(date +%s)"
  # A run that dies before it can write its state leaves the previous night's
  # record on disk. Reading that as "the last run already reported, nothing to
  # say" is how 2026-10-30 went unnoticed: the record was 10-29's failed run
  # with REPORTED=1. Judge freshness by the start date before looking at RC.
  start_date="${START%% *}"
  if [[ "${start_date}" != "$(date +'%F')" ]]; then
    alert "PTOAS A5 板测看护：没有最近一晚的运行记录（最近一次 run ${RUN_ID:-?} 开始于 ${start_date:-unknown}），夜间任务可能没有启动
检查 journalctl -u ptoas-a5-nightly.service 与 ${CI_ROOT}/logs/nightly-cron.log"
    exit 1
  fi
  start_epoch=0
  [[ -n "${START:-}" ]] && start_epoch="$(date -d "${START}" +%s 2>/dev/null || echo 0)"
  end_epoch=0
  [[ -n "${END:-}" ]] && end_epoch="$(date -d "${END}" +%s 2>/dev/null || echo 0)"
  age_h=0
  (( start_epoch > 0 )) && age_h=$(( (now - start_epoch) / 3600 ))
  end_age_h=0
  (( end_epoch > 0 )) && end_age_h=$(( (now - end_epoch) / 3600 ))
  case "${RC:-unknown}" in
    0)
      if (( end_age_h > MAX_SUCCESS_AGE_HOURS )); then
        alert "PTOAS A5 板测看护：最近一次成功运行 ${RUN_ID} 结束于 ${end_age_h} 小时前，已过期"
        exit 1
      fi
      log "WATCHDOG: OK (run ${RUN_ID}, ended ${end_age_h}h ago)"
      exit 0
      ;;
    running)
      if (( age_h > CHECK_LAST_RUN_HOURS )); then
        alert "PTOAS A5 板测看护：run ${RUN_ID} 已运行 ${age_h} 小时仍标记为 running，疑似卡死"
        exit 1
      fi
      log "WATCHDOG: run ${RUN_ID} still in progress (${age_h}h)"
      exit 0
      ;;
    *)
      if [[ "${REPORTED:-0}" == "1" ]]; then
        log "WATCHDOG: run ${RUN_ID} failed (rc=${RC}, kind=${FAILURE_KIND:-?}, phase=${PHASE:-?}); " \
            "the run already sent its own card"
        exit 1
      fi
      alert "PTOAS A5 板测失败 (rc=${RC}, kind=${FAILURE_KIND:-?}, phase=${PHASE:-?})，run ${RUN_ID:-?}
$(a5_failure_summary "${WORK_ROOT:-${CI_ROOT}}/nightly.log")
log: ${WORK_ROOT:-?}/nightly.log"
      exit 1
      ;;
  esac
fi

mkdir -p "${CI_ROOT}/locks" "${CI_ROOT}/logs"
# Update and run under the same lock the driver uses, so a scheduled run, a
# manual run, and a GitCode runner job can never interleave on this tree.
exec 7>"${CI_ROOT}/locks/nightly.lock"
flock -n 7 || {
  log "another A5 nightly run holds ${CI_ROOT}/locks/nightly.lock; exiting"
  exit 0
}

if [[ ! -d "${REPO_DIR}/.git" ]]; then
  [[ -n "${REPO_DIR}" && "${REPO_DIR}" != "/" && "${REPO_DIR}" != "." && "${REPO_DIR}" != ".." ]] \
    || die "refusing unsafe --repo-dir: ${REPO_DIR}"
  log "cloning ${GIT_URL} into ${REPO_DIR}"
  rm -rf -- "${REPO_DIR}"
  git "${git_opts[@]}" clone --branch master "${GIT_URL}" "${REPO_DIR}"
fi

log "updating ${REPO_DIR} from ${GIT_REF}"
if git -C "${REPO_DIR}" remote get-url origin >/dev/null 2>&1; then
  # Keep the remote pinned to the configured URL so switching between the
  # GitCode canonical repo and a mirror is deterministic.
  git -C "${REPO_DIR}" remote set-url origin "${GIT_URL}"
else
  git -C "${REPO_DIR}" remote add origin "${GIT_URL}"
fi
# A previously narrowed refspec (single-branch clone, or a fetch refspec left
# over from a mirror) silently hides every branch except master, which breaks
# --git-ref for anything else. Widen it before fetching.
git -C "${REPO_DIR}" config remote.origin.fetch '+refs/heads/*:refs/remotes/origin/*'
# Transient network/DNS failures are worth retrying; an auth failure is not,
# but it is cheaper to retry three times than to lose a night.
attempt=1
until git -C "${REPO_DIR}" "${git_opts[@]}" fetch --prune origin; do
  if (( attempt >= 3 )); then
    alert "PTOAS A5 板测失败：更新代码失败（${GIT_URL}）
检查 ${GIT_CREDENTIALS_FILE:-CI 凭证文件} 里的账号/token；继承的 credential helper 已被禁用"
    die "git fetch from ${GIT_URL} failed after ${attempt} attempts; " \
        "check ${GIT_CREDENTIALS_FILE:-the CI-owned credentials file}"
  fi
  log "git fetch failed (attempt ${attempt}/3); retrying in 60s"
  sleep 60
  attempt=$((attempt + 1))
done
# A previous run may have died inside the throwaway merge below; clear that
# state before checking out so it can never block the next night.
git -C "${REPO_DIR}" merge --abort >/dev/null 2>&1 || true

# The checkout's origin may be a personal fork (238 uses one for auth), and a
# fork's master can be weeks behind the canonical repo. Fetch the upstream
# branch separately so the merge below validates real master, not the fork.
upstream_ref="origin/master"
if [[ -n "${UPSTREAM_URL}" && "${UPSTREAM_URL}" != "${GIT_URL}" ]]; then
  if git -C "${REPO_DIR}" remote get-url "${UPSTREAM_REMOTE}" >/dev/null 2>&1; then
    git -C "${REPO_DIR}" remote set-url "${UPSTREAM_REMOTE}" "${UPSTREAM_URL}"
  else
    git -C "${REPO_DIR}" remote add "${UPSTREAM_REMOTE}" "${UPSTREAM_URL}"
  fi
  git -C "${REPO_DIR}" config "remote.${UPSTREAM_REMOTE}.fetch" \
    "+refs/heads/*:refs/remotes/${UPSTREAM_REMOTE}/*"
  attempt=1
  until git -C "${REPO_DIR}" "${git_opts[@]}" fetch --prune --no-tags "${UPSTREAM_REMOTE}"; do
    if (( attempt >= 3 )); then
      alert "PTOAS A5 板测失败：无法从主线仓库更新代码（${UPSTREAM_URL}）
检查 ${GIT_CREDENTIALS_FILE:-CI 凭证文件} 里的账号/token"
      die "git fetch from ${UPSTREAM_URL} failed after ${attempt} attempts"
    fi
    log "git fetch ${UPSTREAM_REMOTE} failed (attempt ${attempt}/3); retrying in 60s"
    sleep 60
    attempt=$((attempt + 1))
  done
  upstream_ref="${UPSTREAM_REMOTE}/master"
fi

git -C "${REPO_DIR}" checkout -f "${GIT_REF}"
commit="$(git -C "${REPO_DIR}" rev-parse HEAD)"

# While the CI branch is unmerged it trails master, so checking out the branch
# alone validates a snapshot: a fix that landed on master does not show up and
# a known failure looks unfixed. Merge the upstream master into a throwaway
# local tree (never pushed) before every run, so each night exercises current
# code. Fail closed on a conflict: silently validating stale code is worse than
# a red run, because it misattributes the result.
tested_commit="${commit}"
ref_name="$(git -C "${REPO_DIR}" rev-parse --symbolic-full-name "${GIT_REF}" 2>/dev/null || true)"
upstream_name="$(git -C "${REPO_DIR}" rev-parse --symbolic-full-name "${upstream_ref}" 2>/dev/null || true)"
if [[ "${ref_name}" == refs/remotes/* || "${ref_name}" == refs/heads/* ]] \
   && [[ -n "${ref_name}" && "${ref_name}" != "${upstream_name}" ]]; then
  if git -C "${REPO_DIR}" -c user.name="PTOAS A5 CI" -c user.email="ptoas-ci@localhost" \
       merge --no-edit --no-stat "${upstream_ref}" >/dev/null 2>&1; then
    tested_commit="$(git -C "${REPO_DIR}" rev-parse "${upstream_ref}")"
    log "merged ${upstream_ref} (${tested_commit}) into ${GIT_REF} for this run"
  else
    conflicts="$(git -C "${REPO_DIR}" diff --name-only --diff-filter=U 2>/dev/null | tr '\n' ' ' || true)"
    git -C "${REPO_DIR}" merge --abort >/dev/null 2>&1 || true
    alert "PTOAS A5 板测失败：无法把 ${upstream_ref} 合入 ${GIT_REF}，本次拒绝验证旧快照
冲突文件: ${conflicts:-unknown}
处置: 在 CI 分支上解决并推送后重跑；在 ${REPO_DIR} 手工合并会被下一次 checkout 覆盖"
    die "cannot merge ${upstream_ref} into ${GIT_REF}; refusing to validate a stale snapshot"
  fi
fi
# The card/state carry the master commit under test, not the synthetic local
# merge commit, which does not exist on any remote.
log "running ${GIT_REF} tree at ${commit}, validating ${tested_commit}"

PTOAS_CI_COMMIT_SHA="${tested_commit}" \
  exec bash "${SCRIPT_DIR}/a5_board_nightly.sh" \
    --workspace "${REPO_DIR}" --no-sync --ci-root "${CI_ROOT}" \
    --night-window --skip-lock --scope nightly "$@"
