#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# Collect ptoas binary and macOS dylib dependencies into a self-contained distribution.
#
# Usage: ./collect_ptoas_dist_mac.sh <output_directory>
#
# Required environment variables:
#   LLVM_BUILD_DIR  - Path to LLVM build directory
#   PTO_INSTALL_DIR - Path to PTO install directory
#   PTO_SOURCE_DIR  - Path to PTO source directory
#
# Output structure:
#   <output_directory>/
#     ptoas           - Wrapper script that sets up DYLD_LIBRARY_PATH
#     bin/ptoas       - The actual ptoas binary
#     lib/*.dylib     - Required shared library dependencies

set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Usage: $0 <output_directory>" >&2
  exit 1
fi

PTOAS_DIST_DIR="$1"

# Validate required environment variables
for var in LLVM_BUILD_DIR PTO_INSTALL_DIR PTO_SOURCE_DIR; do
  if [ -z "${!var:-}" ]; then
    echo "Error: $var environment variable is not set" >&2
    exit 1
  fi
done

PTOAS_BIN="${PTO_SOURCE_DIR}/build/tools/ptoas/ptoas"
PTOAS_DEPS_DIR="${PTOAS_DIST_DIR}/lib"
UNRESOLVED_NON_SYSTEM_COUNT=0

if [ ! -f "$PTOAS_BIN" ]; then
  echo "Error: ptoas binary not found at $PTOAS_BIN" >&2
  exit 1
fi

mkdir -p "${PTOAS_DIST_DIR}/bin" "${PTOAS_DEPS_DIR}"
cp -fL "$PTOAS_BIN" "${PTOAS_DIST_DIR}/bin/"
chmod +x "${PTOAS_DIST_DIR}/bin/ptoas"

# Resolve @rpath / @loader_path / @executable_path / absolute install names.
resolve_dep_path() {
  local owner="$1"
  local dep="$2"
  local owner_dir
  owner_dir="$(dirname "$owner")"

  # macOS GitHub runners use bash 3.2; avoid mapfile for compatibility.
  local owner_rpaths=()
  local rp_line
  while IFS= read -r rp_line; do
    [ -n "$rp_line" ] && owner_rpaths+=("$rp_line")
  done < <(
    otool -l "$owner" | awk '
      $1=="cmd" && $2=="LC_RPATH" {want=1; next}
      want && $1=="path" {print $2; want=0}
    '
  )

  local dep_tail="$dep"
  if [[ "$dep" == @rpath/* ]]; then
    dep_tail="${dep#@rpath/}"
  fi

  local candidates=()
  if [[ "$dep" = /* ]]; then
    candidates+=("$dep")
  fi
  if [[ "$dep" == @loader_path/* ]]; then
    candidates+=("${owner_dir}/${dep#@loader_path/}")
  fi
  if [[ "$dep" == @executable_path/* ]]; then
    candidates+=("${PTOAS_DIST_DIR}/bin/${dep#@executable_path/}")
  fi
  if [[ "$dep" == @rpath/* ]]; then
    for rp in "${owner_rpaths[@]:-}"; do
      case "$rp" in
        @loader_path/*) rp="${owner_dir}/${rp#@loader_path/}" ;;
        @executable_path/*) rp="${PTOAS_DIST_DIR}/bin/${rp#@executable_path/}" ;;
      esac
      candidates+=("${rp}/${dep_tail}")
    done
    candidates+=(
      "${LLVM_BUILD_DIR}/lib/${dep_tail}"
      "${PTO_INSTALL_DIR}/lib/${dep_tail}"
      "${owner_dir}/${dep_tail}"
    )
  fi

  local c
  for c in "${candidates[@]}"; do
    if [[ -f "$c" ]]; then
      echo "$c"
      return 0
    fi
  done
  return 1
}

collect_dylibs() {
  local bin="$1"
  while read -r dep; do
    [ -n "$dep" ] || continue
    local resolved
    resolved="$(resolve_dep_path "$bin" "$dep" || true)"
    if [ -z "$resolved" ]; then
      case "$dep" in
        /usr/lib/*|/System/Library/*)
          # Expected on macOS: system libs are provided by the host.
          ;;
        *)
          echo "WARN: unresolved non-system dep for $bin -> $dep"
          UNRESOLVED_NON_SYSTEM_COUNT=$((UNRESOLVED_NON_SYSTEM_COUNT + 1))
          ;;
      esac
      continue
    fi

    local base
    base="$(basename "$resolved")"
    if [ ! -f "${PTOAS_DEPS_DIR}/${base}" ]; then
      cp -fL "$resolved" "${PTOAS_DEPS_DIR}/${base}"
      install_name_tool -id "@loader_path/${base}" "${PTOAS_DEPS_DIR}/${base}" || true
      collect_dylibs "${PTOAS_DEPS_DIR}/${base}"
    fi
    install_name_tool -change "$dep" "@loader_path/../lib/${base}" "$bin" || true
  done < <(otool -L "$bin" | awk 'NR>1 {print $1}')
}

rewrite_packaged_install_names() {
  python3 - "${PTOAS_DIST_DIR}" "${PTOAS_DEPS_DIR}" <<'PY'
import os
import subprocess
import sys
from pathlib import Path

dist_dir = Path(sys.argv[1]).resolve()
deps_dir = Path(sys.argv[2]).resolve()
bin_dir = (dist_dir / "bin").resolve()
allowed_prefixes = (
    "@loader_path/",
    "@rpath/",
    "@executable_path/",
    "/usr/lib/",
    "/System/Library/",
)


def is_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def packaged_dep_ref(owner: Path, dep_base: str) -> str:
    if is_under(owner, bin_dir):
        return f"@loader_path/../lib/{dep_base}"
    if is_under(owner, deps_dir):
        return f"@loader_path/{dep_base}"
    return f"@loader_path/{dep_base}"


def iter_targets():
    for root in (bin_dir, deps_dir):
        if not root.exists():
            continue
        for base, _, files in os.walk(root):
            for name in sorted(files):
                if name == "ptoas" or name.endswith(".dylib"):
                    yield Path(base, name).resolve()


def iter_deps(target: Path):
    try:
        output = subprocess.check_output(
            ["otool", "-L", str(target)],
            stderr=subprocess.STDOUT,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        sys.stderr.write(
            f"ERROR: failed to inspect install names for {target}: "
            f"{exc.output.strip()}\n"
        )
        raise SystemExit(exc.returncode or 1)

    for line in output.splitlines()[1:]:
        dep = line.strip().split(" ", 1)[0]
        if dep:
            yield dep


for target in iter_targets():
    for dep in iter_deps(target):
        if dep.startswith(allowed_prefixes):
            continue

        dep_base = os.path.basename(dep)
        if not (deps_dir / dep_base).is_file():
            continue

        replacement = packaged_dep_ref(target, dep_base)
        if dep == replacement:
            continue

        print(f"rewrite install name: {target} :: {dep} -> {replacement}")
        try:
            subprocess.check_call(
                ["install_name_tool", "-change", dep, replacement, str(target)]
            )
        except subprocess.CalledProcessError as exc:
            sys.stderr.write(
                f"ERROR: failed to rewrite install name for {target}: {dep} -> "
                f"{replacement} (exit {exc.returncode})\n"
            )
            raise SystemExit(exc.returncode or 1)
PY
}

echo "Collecting dylib dependencies..."
collect_dylibs "${PTOAS_DIST_DIR}/bin/ptoas"

echo "Rewriting packaged install names..."
rewrite_packaged_install_names

echo "Validating packaged dependency install names..."
if ! python3 - "${PTOAS_DIST_DIR}" <<'PY'
import os
import subprocess
import sys

root = sys.argv[1]
allowed_prefixes = (
    "@loader_path/",
    "@rpath/",
    "@executable_path/",
    "/usr/lib/",
    "/System/Library/",
)

bad = []
for base, _, files in os.walk(root):
    for name in files:
        if name != "ptoas" and not name.endswith(".dylib"):
            continue
        path = os.path.join(base, name)
        try:
            output = subprocess.check_output(
                ["otool", "-L", path],
                stderr=subprocess.STDOUT,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            print(f"ERROR: failed to inspect {path}: {exc.output.strip()}",
                  file=sys.stderr)
            sys.exit(2)

        for line in output.splitlines()[1:]:
            dep = line.strip().split(" ", 1)[0]
            if dep.startswith(allowed_prefixes):
                continue
            bad.append((path, dep))

for path, dep in bad:
    print(f"ERROR: non-portable dependency in {path} -> {dep}", file=sys.stderr)

print(f"portable dependency scan checked {root} ({len(bad)} offending deps)")
sys.exit(1 if bad else 0)
PY
then
  echo "Error: found non-portable dependency install names" >&2
  exit 1
fi

if ! command -v codesign >/dev/null 2>&1; then
  echo "Error: codesign is required on macOS to sign packaged artifacts" >&2
  exit 1
fi

echo "Ad-hoc signing packaged binaries and dylibs..."
shopt -s nullglob
SIGN_TARGETS=("${PTOAS_DIST_DIR}/bin/ptoas" "${PTOAS_DEPS_DIR}"/*.dylib)
for target in "${SIGN_TARGETS[@]}"; do
  codesign --force --sign - --timestamp=none "$target"
done

echo "Verifying code signatures..."
for target in "${SIGN_TARGETS[@]}"; do
  codesign --verify --strict --verbose=2 "$target"
done
shopt -u nullglob

echo "Creating wrapper script..."
cat > "${PTOAS_DIST_DIR}/ptoas" << 'WRAPPER_EOF'
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export DYLD_LIBRARY_PATH="${SCRIPT_DIR}/lib:${DYLD_LIBRARY_PATH}"
exec "${SCRIPT_DIR}/bin/ptoas" "$@"
WRAPPER_EOF
chmod +x "${PTOAS_DIST_DIR}/ptoas"

echo "Smoke testing packaged ptoas dist..."
VERSION_OUTPUT="$(env -u PYTHONPATH -u DYLD_LIBRARY_PATH -u LD_LIBRARY_PATH \
  "${PTOAS_DIST_DIR}/ptoas" --version | tr -d '\r')"
echo "$VERSION_OUTPUT"
if [ -n "${PTOAS_VERSION:-}" ]; then
  EXPECTED_VERSION_OUTPUT="ptoas ${PTOAS_VERSION}"
  if [ "${VERSION_OUTPUT}" != "${EXPECTED_VERSION_OUTPUT}" ]; then
    echo "Error: expected '${EXPECTED_VERSION_OUTPUT}', got '${VERSION_OUTPUT}'" >&2
    exit 1
  fi
else
  echo "$VERSION_OUTPUT" | grep -Eq '^ptoas [0-9]+\.[0-9]+$'
fi
env -u DYLD_LIBRARY_PATH -u LD_LIBRARY_PATH \
  "${PTOAS_DIST_DIR}/ptoas" \
  "${PTO_SOURCE_DIR}/test/lit/pto/kernel_kind_vector_scf_while_emitc.pto" \
  >/dev/null

echo ""
echo "=== ptoas distribution contents ==="
ls -la "${PTOAS_DIST_DIR}/"
ls -la "${PTOAS_DIST_DIR}/bin/"
DYLIB_COUNT=$(find "${PTOAS_DEPS_DIR}" -name "*.dylib" 2>/dev/null | wc -l)
echo "=== Collected .dylib dependencies (${DYLIB_COUNT} files) ==="
du -sh "${PTOAS_DEPS_DIR}/"
echo "=== Unresolved non-system deps: ${UNRESOLVED_NON_SYSTEM_COUNT} ==="
echo ""
echo "Distribution created at: ${PTOAS_DIST_DIR}"
