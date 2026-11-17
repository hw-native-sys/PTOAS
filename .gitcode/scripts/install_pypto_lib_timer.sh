#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
# Install the host-side systemd timer for the pypto-lib A5 board nightly.
# The unit reuses the PyPTO compatibility driver with `--stage lib`; see
# docs/ci/pypto-lib-a5.md for the prerequisites it expects.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
UNIT_TEMPLATE_DIR="${SCRIPT_DIR}/systemd"

log() { printf '[%s] %s\n' "$(date +'%F %T')" "$*"; }
die() {
  log "ERROR: $*" >&2
  exit 1
}

REPO_DIR="${PTOAS_CI_REPO_DIR:-$(cd -- "${SCRIPT_DIR}/../.." && pwd)}"
PYPTO_ROOT="${PTOAS_PYPTO_ROOT:-/data/ptoas-pypto}"
DEST_DIR="/etc/systemd/system"
ENABLE=1
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-dir) REPO_DIR="$2"; shift 2 ;;
    --pypto-root) PYPTO_ROOT="$2"; shift 2 ;;
    --dest-dir) DEST_DIR="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --no-enable) ENABLE=0; shift ;;
    --help|-h)
      cat <<EOF
Usage: $(basename "$0") [--repo-dir DIR] [--pypto-root DIR] [--dest-dir DIR]
                        [--dry-run] [--no-enable]

Installs ptoas-pypto-lib-a5.service/.timer into ${DEST_DIR} and enables the
daily 05:00 run. --dry-run renders the units into a temporary directory,
prints them, and does not require root.
EOF
      exit 0
      ;;
    *) die "unknown argument: $1 (see --help)" ;;
  esac
done

if (( ! DRY_RUN )); then
  [[ "$(id -u)" -eq 0 ]] || die "run as root (system units)"
  [[ -d "${REPO_DIR}/.git" ]] || die "${REPO_DIR} is not a git checkout"
  # The unit runs the driver through /bin/bash, so the exec bit is irrelevant
  # (the checkout carries it as 0644).
  [[ -f "${REPO_DIR}/.gitcode/scripts/pypto_compat.sh" ]] \
    || die "missing driver: ${REPO_DIR}/.gitcode/scripts/pypto_compat.sh"
  [[ -d "${PYPTO_ROOT}" ]] || die "missing PyPTO root: ${PYPTO_ROOT}"
fi
[[ -f "${UNIT_TEMPLATE_DIR}/ptoas-pypto-lib-a5.service.in" ]] || die "missing service template"

if (( DRY_RUN )); then
  DEST_DIR="$(mktemp --directory)"
  trap 'rm -rf -- "${DEST_DIR}"' EXIT
fi

sed_escape() {
  printf '%s' "$1" | sed -e 's/[\\&|]/\\&/g'
}

render() {
  local src="$1" dst="$2"
  local repo_escaped pypto_escaped
  repo_escaped="$(sed_escape "${REPO_DIR}")"
  pypto_escaped="$(sed_escape "${PYPTO_ROOT}")"
  sed -e "s|@REPO_DIR@|${repo_escaped}|g" \
      -e "s|@PYPTO_ROOT@|${pypto_escaped}|g" "${src}" > "${dst}"
}

if (( ! DRY_RUN )); then
  # systemd opens StandardOutput=append: before ExecStart; create the logging
  # directory here so a clean host does not fail at unit start.
  install -d "${PYPTO_ROOT}/logs"
fi
install -d "${DEST_DIR}"
render "${UNIT_TEMPLATE_DIR}/ptoas-pypto-lib-a5.service.in" \
  "${DEST_DIR}/ptoas-pypto-lib-a5.service"
install -m 0644 "${UNIT_TEMPLATE_DIR}/ptoas-pypto-lib-a5.timer" \
  "${DEST_DIR}/ptoas-pypto-lib-a5.timer"

if (( DRY_RUN )); then
  for unit in "${DEST_DIR}"/ptoas-pypto-lib-a5.*; do
    echo "===== $(basename "${unit}")"
    cat "${unit}"
  done
  log "dry-run complete; nothing installed"
  exit 0
fi

systemctl daemon-reload
if [[ ${ENABLE} -eq 1 ]]; then
  systemctl enable --now ptoas-pypto-lib-a5.timer
fi
log "installed; next run: $(systemctl list-timers ptoas-pypto-lib-a5.timer --no-pager 2>/dev/null | sed -n 2p)"
