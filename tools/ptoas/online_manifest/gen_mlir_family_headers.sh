#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
#
# Regenerate the frozen LLVM/MLIR C++ header manifest consumed by the online
# compilation of the version-sensitive native extensions: the `ptoas.mlir`
# pybind11 family (_mlir core, _mlirDialectsLLVM, _site_initialize_0) AND the
# full-featured `ptoas._core` (PTOModule.cpp + NativeModule.cpp under
# PTOAS_ONLINE_BUILD). All of them compile against the SAME real LLVM/MLIR C++
# header closure staged as `mlir_include/` in the wheel.
#
# This is a MAINTENANCE script: run it once per LLVM upgrade on a dev machine
# that has the LLVM/MLIR source + a configured build tree, and commit the
# resulting `mlir_family_headers.manifest`. The wheel build then copies exactly
# the listed headers (see tools/ptoas/CMakeLists.txt) -- no compiler needed on
# the packaging machine.
#
# The manifest holds include-root-relative header subpaths (e.g.
# `llvm/ADT/StringRef.h`, `mlir/IR/Types.h`, `mlir/Config/mlir-config.h`). It
# deliberately EXCLUDES:
#   - the private sibling headers next to the .cpp (Globals.h, IRModule.h,
#     Pass.h, PybindUtils.h, Rewrite.h) -- those ship with the sources;
#   - pybind11 and Python.h -- those come from the target interpreter's pip/dev.
#
# Usage:
#   LLVM_SRC=/path/to/llvm-project LLVM_BUILD=/path/to/llvm-project/build \
#     tools/ptoas/online_manifest/gen_mlir_family_headers.sh
#
# Defaults match the local dev layout when the env vars are unset.

set -euo pipefail

CXX="${CXX:-c++}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
out_manifest="$script_dir/mlir_family_headers.manifest"
repo_root="$(cd "$script_dir/../../.." && pwd)"

# Repo-owned include roots needed by the _core TUs (PTO/Transforms/*Service.h,
# PTO/Support/CodeConstants.h, pto-c/Dialect/*, and PTOModule.h).
repo_inc="$repo_root/include"
repo_pybind_src="$repo_root/lib/Bindings/Python"

# Include roots, longest first so prefix stripping is unambiguous.
# G.SCRIPT.04: variable derivations stay at most two levels deep; the
# build-tree include roots are therefore expanded inline from LLVM_BUILD.
LLVM_SRC="${LLVM_SRC:-$HOME/workspace/huawei/llvm-workspace/llvm-project}"
LLVM_BUILD="${LLVM_BUILD:-$LLVM_SRC/build-shared}"
inc_mlir_src="$LLVM_SRC/mlir/include"
inc_llvm_src="$LLVM_SRC/llvm/include"
py_root="$LLVM_SRC/mlir/lib/Bindings/Python"

pybind_inc="$(python3 -c 'import pybind11; print(pybind11.get_include())')"
python_inc="$(python3 -c 'import sysconfig; print(sysconfig.get_path("include"))')"

for d in "$inc_mlir_src" "$LLVM_BUILD/tools/mlir/include" "$inc_llvm_src" "$LLVM_BUILD/include" "$py_root"; do
  if [[ ! -d "$d" ]]; then
    echo "error: include/source dir not found: $d" >&2
    exit 1
  fi
done

# The _mlir core family sources, the LLVM dialect binding, and our registration.
sources=(
  "$py_root/MainModule.cpp"
  "$py_root/IRAffine.cpp"
  "$py_root/IRAttributes.cpp"
  "$py_root/IRCore.cpp"
  "$py_root/IRInterfaces.cpp"
  "$py_root/IRModule.cpp"
  "$py_root/IRTypes.cpp"
  "$py_root/Pass.cpp"
  "$py_root/Rewrite.cpp"
  "$py_root/DialectLLVM.cpp"
)
# PythonRegistration.cpp lives in the PTOAS repo, not the LLVM tree.
reg_cpp="$repo_root/lib/Bindings/Python/PythonRegistration.cpp"
if [[ -f "$reg_cpp" ]]; then
  sources+=("$reg_cpp")
fi

# The full-featured `ptoas._core` TUs. NativeModule.cpp is scanned with
# PTOAS_ONLINE_BUILD (below) so it forward-declares runPTOAS and does NOT pull
# the heavy ptoas.h driver closure; its remaining includes (mlir/CAPI/IR.h, the
# service headers -> mlir/IR/BuiltinOps.h, ...) contribute the C++ IR closure.
core_srcs=(
  "$repo_root/lib/Bindings/Python/PTOModule.cpp"
  "$repo_root/tools/ptoas/NativeModule.cpp"
)
for c in "${core_srcs[@]}"; do
  if [[ -f "$c" ]]; then
    sources+=("$c")
  fi
done

tmp_all="$(mktemp)"
trap 'rm -f "$tmp_all"' EXIT

for src in "${sources[@]}"; do
  echo "scanning: ${src##*/}" >&2
  # -M -MG emits the full dependency list even when the preprocess later hits a
  # fatal error (e.g. pybind11's Python version guard against a partial system
  # Python.h). The header list is what we want, so tolerate a nonzero exit.
  "$CXX" -std=c++17 -M -MG \
    -DMLIR_PYTHON_PACKAGE_PREFIX=ptoas.mlir. \
    -DPTOAS_ONLINE_BUILD=1 \
    -I"$py_root" \
    -I"$repo_inc" -I"$repo_pybind_src" \
    -I"$inc_mlir_src" -I"$LLVM_BUILD/tools/mlir/include" \
    -I"$inc_llvm_src" -I"$LLVM_BUILD/include" \
    -I"$pybind_inc" -I"$python_inc" \
    "$src" 2>/dev/null > "$tmp_all.one" || true
  tr ' ' '\n' < "$tmp_all.one" | sed 's/\\$//' >> "$tmp_all"
done
rm -f "$tmp_all.one"

# Keep only headers resolved from the four LLVM/MLIR include roots, then
# relativize to their include-root subpath.
{
  echo "# LLVM/MLIR C++ header closure for online-compiling the version-sensitive"
  echo "# native extensions: the ptoas.mlir pybind11 family AND ptoas._core."
  echo "# GENERATED by gen_mlir_family_headers.sh --"
  echo "# do not edit by hand; rerun the generator after an LLVM upgrade."
  echo "# Paths are include-root-relative; '#' and blank lines are ignored by"
  echo "# the wheel-build reader in tools/ptoas/CMakeLists.txt."
  {
    for root in "$inc_mlir_src" "$LLVM_BUILD/tools/mlir/include" "$inc_llvm_src" "$LLVM_BUILD/include"; do
      grep -F "$root/" "$tmp_all" || true
    done
  } | while IFS= read -r path; do
    [[ -z "$path" ]] && continue
    for root in "$inc_mlir_src" "$LLVM_BUILD/tools/mlir/include" "$inc_llvm_src" "$LLVM_BUILD/include"; do
      if [[ "$path" == "$root/"* ]]; then
        echo "${path#"$root"/}"
        break
      fi
    done
  done | sort -u
} > "$out_manifest"

echo "wrote $(grep -cv '^#\|^$' "$out_manifest") headers to $out_manifest" >&2
