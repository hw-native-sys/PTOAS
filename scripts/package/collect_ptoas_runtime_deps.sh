#!/usr/bin/env bash
# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------

set -euo pipefail

if [ $# -ne 4 ]; then
  echo "Usage: $0 <build_root> <source_binary> <staged_binary> <staged_lib_dir>" >&2
  exit 1
fi

BUILD_ROOT="$(cd "$1" && pwd -P)"
SOURCE_BINARY="$2"
STAGED_BINARY="$3"
STAGED_LIB_DIR="$4"
STAGED_REAL_BINARY="${STAGED_BINARY}.real"
LLVM_STRIP_BIN="${BUILD_ROOT}/llvm-project/build-shared/bin/llvm-strip"
LLVM_RUNTIME_LIB_DIR="${BUILD_ROOT}/llvm-project/build-shared/lib"
declare -a ALLOWED_ROOTS=("${BUILD_ROOT}")

[[ -f "${SOURCE_BINARY}" ]] || {
  echo "Error: source binary not found: ${SOURCE_BINARY}" >&2
  exit 1
}

mkdir -p "$(dirname "${STAGED_BINARY}")" "${STAGED_LIB_DIR}"
rm -f "${STAGED_BINARY}" "${STAGED_REAL_BINARY}"
find "${STAGED_LIB_DIR}" -mindepth 1 -maxdepth 1 -type f -delete 2>/dev/null || true

declare -A COPIED_BY_BASENAME=()
declare -A VISITED_SOURCE_PATHS=()

canonicalize_path() {
  local path="$1"
  if command -v readlink >/dev/null 2>&1 && readlink -f / >/dev/null 2>&1; then
    readlink -f "$path"
    return
  fi
  if command -v realpath >/dev/null 2>&1; then
    realpath "$path"
    return
  fi
  python3 -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "$path"
}

if [ -n "${PTO_INSTALL_DIR:-}" ] && [ -d "${PTO_INSTALL_DIR}" ]; then
  ALLOWED_ROOTS+=("$(canonicalize_path "${PTO_INSTALL_DIR}")")
fi

if [ -d "${LLVM_RUNTIME_LIB_DIR}" ]; then
  export LD_LIBRARY_PATH="${LLVM_RUNTIME_LIB_DIR}:${LD_LIBRARY_PATH:-}"
fi
if [ -n "${PTO_INSTALL_DIR:-}" ] && [ -d "${PTO_INSTALL_DIR}/lib" ]; then
  export LD_LIBRARY_PATH="${PTO_INSTALL_DIR}/lib:${LD_LIBRARY_PATH:-}"
fi

within_allowed_roots() {
  local path="$1"
  local allowed_root
  for allowed_root in "${ALLOWED_ROOTS[@]}"; do
    case "${path}" in
      "${allowed_root}"/*) return 0 ;;
    esac
  done
  return 1
}

read_rpath() {
  local path="$1"
  local dynamic_info
  if command -v patchelf >/dev/null 2>&1; then
    patchelf --print-rpath "$path" 2>/dev/null || true
    return
  fi
  dynamic_info="$(readelf -d "$path" 2>/dev/null || true)"
  awk '
    /(RPATH|RUNPATH)/ && !printed {
      line = $0
      sub(/^.*\[/, "", line)
      sub(/\].*$/, "", line)
      print line
      printed = 1
    }' <<< "${dynamic_info}"
}

has_rpath() {
  local path="$1"
  [[ -n "$(read_rpath "$path")" ]]
}

remove_rpath_with_cmake() {
  local path="$1"
  local current_rpath="$2"
  local script_file
  script_file="$(mktemp)"
  cat > "${script_file}" <<'EOF'
if(NOT DEFINED ELF_PATH OR NOT DEFINED OLD_RPATH)
  message(FATAL_ERROR "ELF_PATH and OLD_RPATH must be provided")
endif()
file(RPATH_CHANGE FILE "${ELF_PATH}" OLD_RPATH "${OLD_RPATH}" NEW_RPATH "")
EOF
  cmake -DELF_PATH="${path}" -DOLD_RPATH="${current_rpath}" -P "${script_file}"
  rm -f "${script_file}"
}

remove_rpath() {
  local path="$1"
  if ! has_rpath "$path"; then
    return
  fi
  if command -v patchelf >/dev/null 2>&1; then
    patchelf --remove-rpath "$path"
  fi
  if has_rpath "$path" && command -v chrpath >/dev/null 2>&1; then
    chrpath -d "$path"
  fi
  if has_rpath "$path"; then
    local current_rpath
    current_rpath="$(read_rpath "$path")"
    if [ -n "${current_rpath}" ] && command -v cmake >/dev/null 2>&1; then
      remove_rpath_with_cmake "$path" "${current_rpath}"
    fi
  fi
  if has_rpath "$path"; then
    echo "Error: failed to scrub RPATH/RUNPATH from ${path}" >&2
    exit 1
  fi
}

strip_binary() {
  local path="$1"
  local -a strip_bins=()
  if command -v strip >/dev/null 2>&1; then
    strip_bins+=("$(command -v strip)")
  fi
  if [ -x "${LLVM_STRIP_BIN}" ]; then
    strip_bins+=("${LLVM_STRIP_BIN}")
  elif command -v llvm-strip >/dev/null 2>&1; then
    strip_bins+=("$(command -v llvm-strip)")
  fi

  local strip_bin
  for strip_bin in "${strip_bins[@]}"; do
    if "${strip_bin}" --strip-unneeded "$path" 2>/dev/null; then
      return
    fi
    if "${strip_bin}" "$path" 2>/dev/null; then
      return
    fi
  done
}

assert_no_symtab() {
  local path="$1"
  local section_info
  section_info="$(readelf -S "$path" 2>/dev/null || true)"
  if grep -Eq '[[:space:]]\\.symtab[[:space:]]' <<< "${section_info}"; then
    echo "Error: symbol table still present in ${path}" >&2
    exit 1
  fi
}

assert_no_rpath() {
  local path="$1"
  if has_rpath "$path"; then
    echo "Error: runtime search path still present in ${path}" >&2
    exit 1
  fi
}

harden_elf() {
  local path="$1"
  remove_rpath "$path"
  strip_binary "$path"
  assert_no_symtab "$path"
  assert_no_rpath "$path"
}

copy_runtime_dep() {
  local source_path="$1"
  [[ -f "${source_path}" ]] || return 0
  source_path="$(canonicalize_path "${source_path}")"
  within_allowed_roots "${source_path}" || return 0

  if [[ -n "${VISITED_SOURCE_PATHS[${source_path}]:-}" ]]; then
    return 0
  fi
  VISITED_SOURCE_PATHS["${source_path}"]=1

  local name
  name="$(basename "${source_path}")"
  if [[ -n "${COPIED_BY_BASENAME[${name}]:-}" ]]; then
    if [[ "${COPIED_BY_BASENAME[${name}]}" != "${source_path}" ]]; then
      echo "Error: dependency basename collision for ${name}: ${COPIED_BY_BASENAME[${name}]} vs ${source_path}" >&2
      exit 1
    fi
    return 0
  fi

  local dest_path="${STAGED_LIB_DIR}/${name}"
  cp -L "${source_path}" "${dest_path}"
  COPIED_BY_BASENAME["${name}"]="${source_path}"
  harden_elf "${dest_path}"

  while read -r dep; do
    [ -n "${dep}" ] || continue
    copy_runtime_dep "${dep}"
  done < <(ldd "${source_path}" 2>/dev/null | awk '/=> \// {print $3}')
}

cp "${SOURCE_BINARY}" "${STAGED_REAL_BINARY}"
harden_elf "${STAGED_REAL_BINARY}"
chmod +x "${STAGED_REAL_BINARY}"

cat > "${STAGED_BINARY}" <<'EOF'
#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LD_LIBRARY_PATH="${SCRIPT_DIR}/../lib:${LD_LIBRARY_PATH:-}"
exec -a ptoas "${SCRIPT_DIR}/ptoas.real" "$@"
EOF
chmod +x "${STAGED_BINARY}"

while read -r dep; do
  [ -n "${dep}" ] || continue
  copy_runtime_dep "${dep}"
done < <(ldd "${SOURCE_BINARY}" 2>/dev/null | awk '/=> \// {print $3}')

while read -r staged_lib; do
  [ -n "${staged_lib}" ] || continue
  harden_elf "${staged_lib}"
done < <(find "${STAGED_LIB_DIR}" -type f | sort)
