#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
# Install the host-side systemd timer for the A5 nightly board validation.
# Use this only when the GitCode host runner is not (yet) available; see
# docs/ci/a5-nightly-board.md.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
UNIT_TEMPLATE_DIR="${SCRIPT_DIR}/systemd"

log() { printf '[%s] %s\n' "$(date +'%F %T')" "$*"; }
die() {
  log "ERROR: $*" >&2
  exit 1
}

REPO_DIR="${PTOAS_CI_REPO_DIR:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
CI_ROOT="${PTOAS_CI_ROOT:-${REPO_DIR}/.work/a5-board-ci}"
DEST_DIR="/etc/systemd/system"
ENABLE=1
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-dir) REPO_DIR="$2"; shift 2 ;;
    --ci-root) CI_ROOT="$2"; shift 2 ;;
    --dest-dir) DEST_DIR="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --no-enable) ENABLE=0; shift ;;
    --help|-h)
      cat <<EOF
Usage: $(basename "$0") [--repo-dir DIR] [--ci-root DIR] [--dest-dir DIR]
                        [--dry-run] [--no-enable]

Installs ptoas-a5-nightly.service/.timer into ${DEST_DIR} and enables the
timer for 00:00 local time. --dry-run renders the units into a temporary
directory, prints them, and does not require root.
EOF
      exit 0
      ;;
    *) die "unknown argument: $1 (see --help)" ;;
  esac
done

if (( ! DRY_RUN )); then
  [[ "$(id -u)" -eq 0 ]] || die "run as root (system units)"
  [[ -d "${REPO_DIR}/.git" ]] || die "${REPO_DIR} is not a git checkout; run a5_nightly_cron.sh once first"
fi
[[ -f "${UNIT_TEMPLATE_DIR}/ptoas-a5-nightly.service.in" ]] || die "missing service template"
if [[ ! -f "${CI_ROOT}/.git-credentials" ]]; then
  log "WARN: ${CI_ROOT}/.git-credentials is missing; create it (chmod 600) or the nightly will fail before the build"
fi

if (( DRY_RUN )); then
  DEST_DIR="$(mktemp --directory)"
  trap 'rm -rf -- "${DEST_DIR}"' EXIT
elif systemctl list-timers --all --no-pager 2>/dev/null | grep -q 'ptoas-a5-nightly.timer'; then
  log "existing ptoas-a5-nightly.timer detected; it will be replaced"
fi

sed_escape() {
  printf '%s' "$1" | sed -e 's/[\\&|]/\\&/g'
}

render() {
  local src="$1"
  local dst="$2"
  local repo_escaped ci_escaped
  repo_escaped="$(sed_escape "${REPO_DIR}")"
  ci_escaped="$(sed_escape "${CI_ROOT}")"
  sed -e "s|@REPO_DIR@|${repo_escaped}|g" \
      -e "s|@CI_ROOT@|${ci_escaped}|g" "${src}" > "${dst}"
}

install -d "${DEST_DIR}"
render "${UNIT_TEMPLATE_DIR}/ptoas-a5-nightly.service.in" "${DEST_DIR}/ptoas-a5-nightly.service"
install -m 0644 "${UNIT_TEMPLATE_DIR}/ptoas-a5-nightly.timer" "${DEST_DIR}/ptoas-a5-nightly.timer"
# Watchdog: a second timer that reports a missing/failed/stuck nightly.
if [[ -f "${UNIT_TEMPLATE_DIR}/ptoas-a5-nightly-watchdog.service.in" ]]; then
  render "${UNIT_TEMPLATE_DIR}/ptoas-a5-nightly-watchdog.service.in" \
    "${DEST_DIR}/ptoas-a5-nightly-watchdog.service"
  install -m 0644 "${UNIT_TEMPLATE_DIR}/ptoas-a5-nightly-watchdog.timer" \
    "${DEST_DIR}/ptoas-a5-nightly-watchdog.timer"
fi
# Three-stage guard: start / middle / end checkpoints.
GUARD_TIMERS=()
if [[ -f "${UNIT_TEMPLATE_DIR}/ptoas-a5-guard@.service.in" ]]; then
  render "${UNIT_TEMPLATE_DIR}/ptoas-a5-guard@.service.in" \
    "${DEST_DIR}/ptoas-a5-guard@.service"
  for stage in start middle end; do
    install -m 0644 "${UNIT_TEMPLATE_DIR}/ptoas-a5-guard-${stage}.timer" \
      "${DEST_DIR}/ptoas-a5-guard-${stage}.timer"
    GUARD_TIMERS+=("ptoas-a5-guard-${stage}.timer")
  done
fi

if (( DRY_RUN )); then
  for unit in "${DEST_DIR}"/*; do
    echo "===== $(basename "${unit}")"
    cat "${unit}"
  done
  log "dry-run complete; nothing installed"
  exit 0
fi

systemctl daemon-reload
if [[ ${ENABLE} -eq 1 ]]; then
  systemctl enable --now ptoas-a5-nightly.timer
  [[ -f "${DEST_DIR}/ptoas-a5-nightly-watchdog.timer" ]] \
    && systemctl enable --now ptoas-a5-nightly-watchdog.timer
  for t in "${GUARD_TIMERS[@]}"; do
    systemctl enable --now "$t"
  done
  systemctl list-timers --all --no-pager | grep 'ptoas-a5' || true
else
  log "timer installed but not enabled (--no-enable)"
fi
log "installed: ${DEST_DIR}/ptoas-a5-nightly.{service,timer}"
