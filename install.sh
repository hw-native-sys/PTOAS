#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set -euo pipefail

REPO_DEFAULT="zhangstevenunity/PTOAS"
REPO="${PTOAS_REPO:-$REPO_DEFAULT}"

TAG=""
TARGET=""
INSTALL_ROOT="${PTOAS_INSTALL_ROOT:-${HOME:-}/.local/ptoas}"
BIN_DIR="${PTOAS_BIN_DIR:-${HOME:-}/.local/bin}"
FORCE=0
DRY_RUN=0
INSTALL_LAUNCHER=1

usage() {
  cat <<'EOF'
Install prebuilt PTOAS binaries from GitHub Releases.

Usage:
  ./install.sh [options]

Options:
  --tag <tag>            Release tag to install (default: latest)
  --target <target>      Override detected target
                          Supported: linux-x86_64, linux-aarch64
  --install-root <dir>   Install root directory (default: $HOME/.local/ptoas)
  --bin-dir <dir>        Where to place a launcher script (default: $HOME/.local/bin)
  --no-launcher          Do not install launcher into --bin-dir
  --force                Overwrite existing install dir
  --dry-run              Print planned actions without changing the system
  -h, --help             Show this help

Env:
  PTOAS_REPO          GitHub repo (default: zhangstevenunity/PTOAS)
  PTOAS_INSTALL_ROOT  Same as --install-root
  PTOAS_BIN_DIR       Same as --bin-dir

Examples:
  # Install the latest release for your machine (Linux only for now)
  ./install.sh

  # Install a specific tag
  ./install.sh --tag v0.3

  # Test extraction on a non-Linux machine (downloads Linux binaries)
  ./install.sh --target linux-x86_64 --tag v0.3 --dry-run
EOF
}

die() {
  echo "error: $*" >&2
  exit 1
}

have_cmd() {
  command -v "$1" >/dev/null 2>&1
}

http_get() {
  local url="$1"
  if have_cmd curl; then
    curl -fsSL "$url"
  elif have_cmd wget; then
    wget -qO- "$url"
  else
    die "missing required tool: curl (or wget)"
  fi
}

download_file() {
  local url="$1"
  local out="$2"
  if have_cmd curl; then
    curl -fL "$url" -o "$out"
  elif have_cmd wget; then
    wget -q "$url" -O "$out"
  else
    die "missing required tool: curl (or wget)"
  fi
}

get_latest_tag() {
  local api="https://api.github.com/repos/${REPO}/releases/latest"
  local json tag
  json="$(http_get "$api")"
  tag="$(printf '%s\n' "$json" | sed -nE 's/.*"tag_name"[[:space:]]*:[[:space:]]*"([^"]+)".*/\1/p' | head -n 1)"
  [[ -n "$tag" ]] || die "failed to detect latest tag from $api"
  printf '%s\n' "$tag"
}

detect_target() {
  local os arch
  os="$(uname -s 2>/dev/null || true)"
  arch="$(uname -m 2>/dev/null || true)"

  case "$os" in
    Linux) ;;
    *)
      die "unsupported OS '$os' (only Linux is supported by prebuilt binaries right now). Use --target to override for testing, or build from source."
      ;;
  esac

  case "$arch" in
    x86_64|amd64) printf '%s\n' "linux-x86_64" ;;
    aarch64|arm64) printf '%s\n' "linux-aarch64" ;;
    *) die "unsupported arch '$arch' (supported: x86_64, aarch64)" ;;
  esac
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tag) TAG="${2:-}"; shift 2 ;;
    --target) TARGET="${2:-}"; shift 2 ;;
    --install-root) INSTALL_ROOT="${2:-}"; shift 2 ;;
    --bin-dir) BIN_DIR="${2:-}"; shift 2 ;;
    --no-launcher) INSTALL_LAUNCHER=0; shift ;;
    --force) FORCE=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *)
      die "unknown argument: $1 (use --help)"
      ;;
  esac
done

[[ -n "${INSTALL_ROOT}" ]] || die "--install-root is empty"
[[ -n "${BIN_DIR}" ]] || die "--bin-dir is empty"

if [[ -z "$TARGET" ]]; then
  TARGET="$(detect_target)"
fi

case "$TARGET" in
  linux-x86_64) ASSET="ptoas-bin-x86_64.tar.gz" ;;
  linux-aarch64) ASSET="ptoas-bin-aarch64.tar.gz" ;;
  *) die "unsupported --target '$TARGET' (supported: linux-x86_64, linux-aarch64)" ;;
esac

if [[ -z "$TAG" ]]; then
  TAG="$(get_latest_tag)"
fi

INSTALL_DIR="${INSTALL_ROOT%/}/${TAG}"
URL="https://github.com/${REPO}/releases/download/${TAG}/${ASSET}"

cat <<EOF
PTOAS install plan:
  repo:        ${REPO}
  tag:         ${TAG}
  target:      ${TARGET}
  asset:       ${ASSET}
  url:         ${URL}
  install_dir: ${INSTALL_DIR}
  launcher:    $( [[ $INSTALL_LAUNCHER -eq 1 ]] && echo "${BIN_DIR%/}/ptoas" || echo "(disabled)" )
EOF

if [[ $DRY_RUN -eq 1 ]]; then
  exit 0
fi

have_cmd tar || die "missing required tool: tar"
have_cmd mktemp || die "missing required tool: mktemp"

tmpdir="$(mktemp -d)"
cleanup() { rm -rf "$tmpdir"; }
trap cleanup EXIT

tarball="${tmpdir}/${ASSET}"
echo "Downloading: ${URL}"
download_file "$URL" "$tarball"

if [[ -e "$INSTALL_DIR" ]]; then
  if [[ $FORCE -eq 1 ]]; then
    echo "Removing existing install dir: ${INSTALL_DIR}"
    rm -rf "$INSTALL_DIR"
  else
    die "install dir already exists: ${INSTALL_DIR} (use --force to overwrite)"
  fi
fi

mkdir -p "$INSTALL_DIR"
tar -xzf "$tarball" -C "$INSTALL_DIR"

chmod +x "$INSTALL_DIR/ptoas" "$INSTALL_DIR/bin/ptoas" || true

if [[ $INSTALL_LAUNCHER -eq 1 ]]; then
  mkdir -p "$BIN_DIR"
  launcher="${BIN_DIR%/}/ptoas"
  cat >"$launcher" <<EOF
#!/usr/bin/env bash
set -euo pipefail
exec "${INSTALL_DIR}/ptoas" "\$@"
EOF
  chmod +x "$launcher"
fi

echo ""
echo "Installed PTOAS to: ${INSTALL_DIR}"
if [[ $INSTALL_LAUNCHER -eq 1 ]]; then
  echo "Launcher installed: ${BIN_DIR%/}/ptoas"
fi
echo ""
echo "Next steps:"
if [[ $INSTALL_LAUNCHER -eq 1 ]]; then
  echo "  - Try: ${BIN_DIR%/}/ptoas --help"
else
  echo "  - Try: ${INSTALL_DIR}/ptoas --help"
fi

case ":${PATH:-}:" in
  *":${BIN_DIR%/}:"*) ;;
  *)
    if [[ $INSTALL_LAUNCHER -eq 1 ]]; then
      echo "  - Add to PATH (example): export PATH=\"${BIN_DIR%/}:\$PATH\""
    fi
    ;;
esac
