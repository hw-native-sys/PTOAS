#!/bin/bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
# Build driver for the LLVM 19 snapshot tree (project ptoas, the GitHub PTOAS
# layout). This is the entry point used by the gitcode smoke pipeline
# (./build.sh --build / --pkg). It builds the external LLVM/MLIR 19 dependency
# (reusing a cached LLVM source/build when available) and then builds and
# installs PTOAS through the tree's native CMake build.

set -e

dotted_line="----------------------------------------------------------------"
COLOR_RESET="\033[0m"
COLOR_GREEN="\033[32m"
COLOR_RED="\033[31m"

export BASE_PATH=$(
  cd "$(dirname $0)"
  pwd
)
export BUILD_PATH="${BASE_PATH}/build"
export BUILD_OUT_PATH="${BASE_PATH}/build_out"
export INSTALL_PATH="${BASE_PATH}/install"
export PACKAGE_STAGE_PATH="${BUILD_PATH}/package_runtime"
export PTOAS_PRESMOKE_SKIP_RUNOP_MARKER="${BUILD_PATH}/.skip-presmoke-runop"
export LLVM_SOURCE_VERSION="19.1.7"
export PTOAS_GLIBCXX_ABI="${PTOAS_GLIBCXX_ABI:-0}"
# This branch is used to validate the native LLVM/PTOAS build and CANN package
# lifecycle while the CI image cannot install the Python wheel build backend.
# Keep the normal CMake/CPack package flow, but deliberately omit the wheel from
# the generated .run artifact.  The installer handles the empty wheel directory
# as a successful placeholder package.
export PTOAS_PLACEHOLDER_RUN_PACKAGE="${PTOAS_PLACEHOLDER_RUN_PACKAGE:-TRUE}"
# The PTOAS tree is built against the vpto-dev LLVM/MLIR 19 "feature-vpto"
# branch (source of custom calling conventions such as SimtEntry). Source it
# from GitHub by default; override with LLVM_GIT_URL / LLVM_GIT_REF when a
# mirror must be used.
export LLVM_GIT_URL="${LLVM_GIT_URL:-https://github.com/vpto-dev/llvm-project.git}"
export LLVM_GIT_REF="${LLVM_GIT_REF:-feature-vpto}"
# The vpto calling conventions (SimtEntry, float8) can also be produced by
# applying the feature-vpto patch to the upstream llvmorg-19.1.7 source that
# the CI cache (ASCEND_3RD_LIB_PATH) and cann-cmake download. When the cached
# source lacks SimtEntry we fetch the patch from the gitcode release asset and
# apply it with patch -p1, matching the PATCH_COMMAND added to cann-cmake's
# third_party/llvm.cmake.
export LLVM_VPTO_PATCH_URL="${LLVM_VPTO_PATCH_URL:-https://gitcode.com/cann-src-third-party/llvm/releases/download/19.1.7-h0/feature-vpto-last3.patch}"
export LLVM_VPTO_PATCH_SHA256="${LLVM_VPTO_PATCH_SHA256:-a49c1d3dd8ab78e93264712bc0d46deb536196a54abb2c2ee02abd914cd385e2}"
# Prefer ASCEND_3RD_LIB_PATH when it points to a valid LLVM source cache
# (CI images set this to /home/jenkins/opensource). Fall back to the in-tree
# third_party directory for local builds where it is unset.
if [ -n "${ASCEND_3RD_LIB_PATH}" ] && [ -d "${ASCEND_3RD_LIB_PATH}/llvm-19" ]; then
    CANN_3RD_LIB_PATH="${ASCEND_3RD_LIB_PATH}"
else
    CANN_3RD_LIB_PATH="${BASE_PATH}/third_party"
fi
HARDENING_CACHE_FILE="${BASE_PATH}/cmake/LinuxHardeningCache.cmake"
LLVM_PROJECT_URL="${LLVM_GIT_URL}"
# Only enable the CentOS7 devtoolset-7 sysroot + gcc-toolchain when the
# toolchain is actually present. Manylinux and non-CentOS7 images do not ship
# /opt/rh/devtoolset-7, and forcing these flags there breaks the build because
# clang cannot find the sysroot.
if [ -d "/opt/rh/devtoolset-7/root" ]; then
  DEVTOOLSET_TOOLCHAIN_FLAGS="--sysroot=/opt/rh/devtoolset-7/root --gcc-toolchain=/opt/rh/devtoolset-7/root/usr"
else
  DEVTOOLSET_TOOLCHAIN_FLAGS=""
fi

# Internal builds provide a pinned clang-15 toolchain under /opt/buildtools,
# while gitcode images provide clang-15 through PATH. Keep the internal
# toolchain as the first choice, but require both drivers to exist before
# selecting it so the same entry point works in both environments.
resolve_ptoas_toolchain() {
  local default_cc=""
  local default_cxx=""
  local internal_cc="/opt/buildtools/llvm-15.0.4/bin/clang"
  local internal_cxx="/opt/buildtools/llvm-15.0.4/bin/clang++"

  if [ -x "${internal_cc}" ] && [ -x "${internal_cxx}" ]; then
    default_cc="${internal_cc}"
    default_cxx="${internal_cxx}"
  elif command -v clang >/dev/null 2>&1 \
       && command -v clang++ >/dev/null 2>&1; then
    default_cc="$(command -v clang)"
    default_cxx="$(command -v clang++)"
  elif command -v clang-15 >/dev/null 2>&1 \
       && command -v clang++-15 >/dev/null 2>&1; then
    default_cc="$(command -v clang-15)"
    default_cxx="$(command -v clang++-15)"
  elif command -v gcc >/dev/null 2>&1 \
       && command -v g++ >/dev/null 2>&1; then
    default_cc="$(command -v gcc)"
    default_cxx="$(command -v g++)"
  fi

  PTOAS_CC="${PTOAS_CC:-${default_cc}}"
  PTOAS_CXX="${PTOAS_CXX:-${default_cxx}}"
  PTOAS_CC="$(command -v "${PTOAS_CC}" 2>/dev/null || true)"
  PTOAS_CXX="$(command -v "${PTOAS_CXX}" 2>/dev/null || true)"

  if [ -z "${PTOAS_CC}" ] || [ -z "${PTOAS_CXX}" ]; then
    echo "ERROR: no usable C/C++ compiler pair found" >&2
    echo "Set PTOAS_CC and PTOAS_CXX to executable compiler paths." >&2
    exit 1
  fi

  export PTOAS_CC PTOAS_CXX
  echo "Using PTOAS toolchain: CC=${PTOAS_CC}, CXX=${PTOAS_CXX}"
}

print_success() {
  echo
  echo $dotted_line
  echo -e "${COLOR_GREEN}[SUCCESS] ${msg}${COLOR_RESET}"
  echo $dotted_line
  echo
}

print_error() {
  echo
  echo $dotted_line
  local msg="$1"
  echo -e "${COLOR_RED}[ERROR] ${msg}${COLOR_RESET}"
  echo $dotted_line
  echo
}

usage() {
  echo "Usage:"
  echo ""
  echo "    -h, --help               Print usage"
  echo "    --build                  Build and run validation"
  echo "    --pkg                    Build and package through CANN CPack"
  echo "    --pkg-type=<TYPE>        Package type (run/rpm/deb/all); accepted for"
  echo "                             interface compatibility and forwarded to CPack"
  echo "    -j <N>                   Parallel jobs (default: nproc)"
  echo "    --cann_3rd_lib_path <d>  Override the third-party/LLVM cache root"
  echo ""
}

# ---------------------------------------------------------------------------
# LLVM/MLIR dependency handling
# ---------------------------------------------------------------------------
prepare_llvm_cache_layout() {
  mkdir -p "${CANN_3RD_LIB_PATH}"
  mkdir -p "${CANN_3RD_LIB_PATH}/lib_cache/llvm_${LLVM_SOURCE_VERSION}"

  export LLVM_SOURCE_DIR="${CANN_3RD_LIB_PATH}/llvm-19"
  # Keep an explicitly supplied build directory intact. Otherwise use the
  # conventional cache path; incompatible legacy caches are redirected to a
  # configuration-specific sibling by ensure_llvm_build().
  if [ -z "${LLVM_BUILD_DIR:-}" ]; then
    export PTOAS_LLVM_BUILD_DIR_EXPLICIT=FALSE
    export LLVM_BUILD_DIR="${CANN_3RD_LIB_PATH}/lib_cache/llvm_${LLVM_SOURCE_VERSION}/build-shared"
  else
    export PTOAS_LLVM_BUILD_DIR_EXPLICIT=TRUE
  fi
}

# Check whether the LLVM source carries the vpto custom calling conventions
# (llvm::CallingConv::SimtEntry). Upstream llvmorg-19.1.7 does not; the vpto
# fork and a patched upstream tree do.
llvm_has_simt_entry() {
  [ -f "${LLVM_SOURCE_DIR}/llvm/include/llvm/IR/CallingConv.h" ] \
    && grep -q "SimtEntry" "${LLVM_SOURCE_DIR}/llvm/include/llvm/IR/CallingConv.h"
}

# Download the feature-vpto patch and apply it to the upstream LLVM source so
# the tree gains the vpto calling conventions. The patch is a git-format-patch
# series rooted at llvm/, so patch -p1 is the correct strip level (the same
# PATCH_COMMAND used by cann-cmake's third_party/llvm.cmake).
apply_vpto_patch() {
  echo "${dotted_line}"
  echo "Applying feature-vpto patch to upstream LLVM source"
  local patch_file="${CANN_3RD_LIB_PATH}/pkg/feature-vpto-last3.patch"
  if [ -f "${CANN_3RD_LIB_PATH}/feature-vpto-last3.patch" ]; then
    patch_file="${CANN_3RD_LIB_PATH}/feature-vpto-last3.patch"
  elif [ -f "${CANN_3RD_LIB_PATH}/pkg/feature-vpto-last3.patch" ]; then
    patch_file="${CANN_3RD_LIB_PATH}/pkg/feature-vpto-last3.patch"
  else
    mkdir -p "${CANN_3RD_LIB_PATH}/pkg"
    echo "Downloading vpto patch from ${LLVM_VPTO_PATCH_URL}"
    curl -fL --retry 3 -o "${patch_file}" "${LLVM_VPTO_PATCH_URL}" || {
      echo "ERROR: failed to download vpto patch" >&2
      exit 1
    }
    local actual_sha
    actual_sha="$(sha256sum "${patch_file}" | cut -d' ' -f1)"
    if [ "${actual_sha}" != "${LLVM_VPTO_PATCH_SHA256}" ]; then
      echo "ERROR: vpto patch SHA256 mismatch: ${actual_sha}" >&2
      exit 1
    fi
  fi

  (cd "${LLVM_SOURCE_DIR}" && patch -p1 < "${patch_file}") || {
    echo "ERROR: failed to apply vpto patch to ${LLVM_SOURCE_DIR}" >&2
    exit 1
  }
  echo "Applied vpto patch: ${patch_file}"
}

# Ensure the LLVM 19 source (vpto "feature-vpto" branch) is present under
# ${LLVM_SOURCE_DIR}. Accepts an already-populated source tree (the usual CI
# cache layout where llvm-19/llvm holds the top-level CMakeLists.txt) or
# clones the vpto branch from ${LLVM_GIT_URL}. When the cached source is the
# upstream (unpatched) snapshot, apply the vpto patch so SimtEntry/float8
# resolve during the PTOAS build.
ensure_llvm_source() {
  if [ -f "${LLVM_SOURCE_DIR}/llvm/CMakeLists.txt" ]; then
    # Git checkout layout: the project root is ${LLVM_SOURCE_DIR}/llvm.
    export LLVM_CMAKE_SOURCE_DIR="${LLVM_SOURCE_DIR}/llvm"
    if ! llvm_has_simt_entry; then
      echo "${dotted_line}"
      echo "Cached LLVM source lacks SimtEntry; applying feature-vpto patch"
      apply_vpto_patch
    fi
    return 0
  fi
  if [ -f "${LLVM_SOURCE_DIR}/CMakeLists.txt" ]; then
    export LLVM_CMAKE_SOURCE_DIR="${LLVM_SOURCE_DIR}"
    if ! llvm_has_simt_entry; then
      echo "${dotted_line}"
      echo "Cached LLVM source lacks SimtEntry; applying feature-vpto patch"
      apply_vpto_patch
    fi
    return 0
  fi

  echo "${dotted_line}"
  echo "Cloning LLVM ${LLVM_SOURCE_VERSION} source (${LLVM_GIT_REF})"
  mkdir -p "${CANN_3RD_LIB_PATH}"
  git clone --depth 1 --single-branch \
    --branch "${LLVM_GIT_REF}" \
    "${LLVM_GIT_URL}" "${LLVM_SOURCE_DIR}"
  if [ -f "${LLVM_SOURCE_DIR}/llvm/CMakeLists.txt" ]; then
    export LLVM_CMAKE_SOURCE_DIR="${LLVM_SOURCE_DIR}/llvm"
  else
    export LLVM_CMAKE_SOURCE_DIR="${LLVM_SOURCE_DIR}"
  fi
}

# LLVM/MLIR and PTOAS are both built with the old libstdc++ ABI. Reject a
# cached LLVM tree that exports the new-ABI Twine::str symbol.
llvm_build_is_abi_compatible() {
  local support_lib="${LLVM_BUILD_DIR}/lib/libLLVMSupport.so.19.1"
  [ -f "${support_lib}" ] || return 1
  # _ZNK4llvm5Twine3strEv -> _GLIBCXX_USE_CXX11_ABI=0
  nm -D --defined-only "${support_lib}" 2>/dev/null \
    | grep -q "_ZNK4llvm5Twine3strEv"
}

# BSPUB changes LLVM data types at compile time, so a cache produced without
# the matching C/C++ definitions and LLVM option cannot be reused safely.
llvm_build_has_bspub_npu_data_type() {
  local cache_file="${LLVM_BUILD_DIR}/CMakeCache.txt"
  [ -f "${cache_file}" ] || return 1
  grep -Eq '^LLVM_BSPUB_NPU_DATA_TYPE:(BOOL|UNINITIALIZED)=ON$' "${cache_file}" \
    && grep -Eq '^CMAKE_C_FLAGS:[^=]*=.*-DBSPUB_NPU_DATA_TYPE([[:space:]]|$)' "${cache_file}" \
    && grep -Eq '^CMAKE_CXX_FLAGS:[^=]*=.*-DBSPUB_NPU_DATA_TYPE([[:space:]]|$)' "${cache_file}"
}

# PTOAS links LLVM and MLIR component targets directly. Keep those components
# shared so the Python runtime loads one copy of LLVM's command-line registry.
# A monolithic libLLVM alongside component DSOs can register options twice.
llvm_build_uses_shared_components() {
  local cache_file="${LLVM_BUILD_DIR}/CMakeCache.txt"
  [ -f "${cache_file}" ] || return 1
  grep -Eq '^BUILD_SHARED_LIBS:(BOOL|UNINITIALIZED)=ON$' "${cache_file}" \
    && grep -Eq '^LLVM_BUILD_LLVM_DYLIB:(BOOL|UNINITIALIZED)=OFF$' "${cache_file}" \
    && grep -Eq '^LLVM_LINK_LLVM_DYLIB:(BOOL|UNINITIALIZED)=OFF$' "${cache_file}"
}

# The BSPUB backport makes LLVMVectorize call llvm::Triple, but its LLVM 19
# component metadata omits TargetParser. Add that DSO through LLVM's
# target-specific linker-flags cache entry without modifying LLVM sources.
llvm_build_links_vectorize_target_parser() {
  local cache_file="${LLVM_BUILD_DIR}/CMakeCache.txt"
  [ -f "${cache_file}" ] || return 1
  grep -Eq '^LLVM_LLVMVectorize_LINKER_FLAGS:[^=]*=.*-lLLVMTargetParser([;[:space:]]|$)' \
    "${cache_file}"
}

llvm_vectorize_has_target_parser_dependency() {
  local vectorize_lib="${LLVM_BUILD_DIR}/lib/libLLVMVectorize.so.19.1"
  local readelf_bin
  [ -f "${vectorize_lib}" ] || return 1
  readelf_bin="$(command -v readelf || command -v llvm-readelf || true)"
  [ -n "${readelf_bin}" ] || return 1
  "${readelf_bin}" -d "${vectorize_lib}" 2>/dev/null \
    | grep -q 'libLLVMTargetParser\.so'
}

# Check every property that makes the shared LLVM cache safe for PTOAS. This
# is deliberately stricter than checking for LLVMConfig.cmake alone: a cache
# from the CANN toolchain may be complete but use the opposite libstdc++ ABI
# or omit the TargetParser dependency required by LLVMVectorize.
llvm_build_cache_is_usable() {
  [ -f "${LLVM_BUILD_DIR}/lib/cmake/llvm/LLVMConfig.cmake" ] \
    && [ -f "${LLVM_BUILD_DIR}/lib/cmake/mlir/MLIRConfig.cmake" ] \
    || return 1

  if llvm_has_simt_entry \
     && [ -f "${LLVM_BUILD_DIR}/include/llvm/IR/CallingConv.h" ] \
     && ! grep -q "SimtEntry" "${LLVM_BUILD_DIR}/include/llvm/IR/CallingConv.h"; then
    return 1
  fi
  llvm_build_is_abi_compatible || return 1
  llvm_build_has_bspub_npu_data_type || return 1
  llvm_build_uses_shared_components || return 1
  llvm_build_links_vectorize_target_parser || return 1
  llvm_vectorize_has_target_parser_dependency || return 1
}

# Build LLVM/MLIR 19 (shared components + MLIR Python bindings) if the cached build
# tree is not usable, mirroring the PTOAS development workflow.
ensure_llvm_build() {
  ensure_llvm_source

  local default_llvm_build_dir="${LLVM_BUILD_DIR}"
  if llvm_build_cache_is_usable; then
    echo "${dotted_line}"
    echo "Reusing cached LLVM/MLIR build at ${LLVM_BUILD_DIR}"
    return 0
  fi

  # Do not destroy a legacy cache shared by other jobs. Build the requested
  # configuration in a stable sibling directory so later PreSmoke runs can
  # reuse it instead of repeating the LLVM build.
  local keyed_llvm_build_dir="${default_llvm_build_dir}-ptoas-abi${PTOAS_GLIBCXX_ABI}-bspub-shared"
  if [ "${PTOAS_LLVM_BUILD_DIR_EXPLICIT:-FALSE}" != "TRUE" ]; then
    export LLVM_BUILD_DIR="${keyed_llvm_build_dir}"
    if llvm_build_cache_is_usable; then
      echo "${dotted_line}"
      echo "Reusing cached LLVM/MLIR build at ${LLVM_BUILD_DIR}"
      return 0
    fi
    echo "Cached LLVM/MLIR configuration is unavailable; building ${LLVM_BUILD_DIR}"
  fi

  rm -rf "${LLVM_BUILD_DIR}"

  echo "${dotted_line}"
  echo "Building LLVM/MLIR ${LLVM_SOURCE_VERSION} (this can take a while)"
  mkdir -p "${LLVM_BUILD_DIR}"

  local python_bin
  python_bin="$(command -v python3 || command -v python)"

  local pybind_dir
  pybind_dir="$("${python_bin}" -m pybind11 --cmakedir 2>/dev/null || true)"

  local cmake_args=(
    -G Ninja
    -S "${LLVM_CMAKE_SOURCE_DIR}"
    -B "${LLVM_BUILD_DIR}"
    # PTOAS consumes LLVM/MLIR through CMake; Clang is not a PTOAS dependency.
    # Keeping it out avoids building clangInterpreter, which is incompatible
    # with the GCC 7 libstdc++ headers used by the ARM CI image.
    -DLLVM_ENABLE_PROJECTS="mlir"
    -DCMAKE_CXX_FLAGS="-DBSPUB_NPU_DATA_TYPE -D_GLIBCXX_USE_CXX11_ABI=${PTOAS_GLIBCXX_ABI}"
    -DCMAKE_C_FLAGS="-DBSPUB_NPU_DATA_TYPE"
    -DLLVM_BSPUB_NPU_DATA_TYPE=ON
    -DBUILD_SHARED_LIBS=ON
    -DLLVM_BUILD_LLVM_DYLIB=OFF
    -DLLVM_LINK_LLVM_DYLIB=OFF
    -DLLVM_LLVMVectorize_LINKER_FLAGS="-L${LLVM_BUILD_DIR}/lib;-Wl,--no-as-needed;-lLLVMTargetParser;-Wl,--as-needed"
    -DLLVM_ENABLE_ASSERTIONS=ON
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON
    -DCMAKE_BUILD_TYPE=Release
    -DLLVM_TARGETS_TO_BUILD="host"
    -DLLVM_ENABLE_ZSTD=OFF
    -DLLVM_INCLUDE_TESTS=OFF
    -DLLVM_INCLUDE_BENCHMARKS=OFF
    -DLLVM_INCLUDE_EXAMPLES=OFF
    -DCMAKE_C_COMPILER="${PTOAS_CC}"
    -DCMAKE_CXX_COMPILER="${PTOAS_CXX}"
    -DPython3_EXECUTABLE="${python_bin}"
    -DPython_EXECUTABLE="${python_bin}"
  )
  if [ -n "${pybind_dir}" ]; then
    cmake_args+=( -Dpybind11_DIR="${pybind_dir}" )
  fi

  if [ -f "${HARDENING_CACHE_FILE}" ]; then
    # Process the BSPUB flags before the preload cache so the cache appends,
    # rather than loses, the delivery hardening flags.
    cmake "${cmake_args[@]}" -C "${HARDENING_CACHE_FILE}"
  else
    cmake "${cmake_args[@]}"
  fi
  # The linker-flags cache entry adds a linker input but not a Ninja target
  # edge, so materialize TargetParser before the parallel Vectorize link.
  cmake --build "${LLVM_BUILD_DIR}" --target LLVMTargetParser -- -j "${JOBS}"
  cmake --build "${LLVM_BUILD_DIR}" -- -j "${JOBS}"
  if ! llvm_vectorize_has_target_parser_dependency; then
    echo "ERROR: LLVMVectorize was built without a dependency on LLVMTargetParser" >&2
    exit 1
  fi
}

# ---------------------------------------------------------------------------
# PTOAS build + install
# ---------------------------------------------------------------------------
compiler_rt_has_muloti4() {
  local runtime_archive="$1"
  local nm_bin="${LLVM_BUILD_DIR}/bin/llvm-nm"

  [ -f "${runtime_archive}" ] || return 1
  if [ ! -x "${nm_bin}" ]; then
    nm_bin="$(command -v llvm-nm || command -v nm || true)"
  fi
  [ -n "${nm_bin}" ] || return 1
  "${nm_bin}" -g --defined-only "${runtime_archive}" 2>/dev/null \
    | grep -E '(^|[[:space:]])__muloti4$' >/dev/null
}

# Some internal aarch64 images ship clang without compiler-rt. Build only the
# builtins archive from the matching LLVM source tree as a cached fallback.
build_aarch64_compiler_rt() {
  local compiler_rt_src="${LLVM_SOURCE_DIR}/compiler-rt"
  local builtins_src="${compiler_rt_src}/lib/builtins"
  local compiler_rt_build="${CANN_3RD_LIB_PATH}/lib_cache/compiler_rt_${LLVM_SOURCE_VERSION}/build-aarch64"
  local compiler_target

  if [ ! -f "${builtins_src}/CMakeLists.txt" ]; then
    echo "ERROR: compiler-rt builtins source not found: ${builtins_src}" >&2
    exit 1
  fi

  compiler_target="$("${PTOAS_CC}" -print-target-triple 2>/dev/null || true)"
  if [ -z "${compiler_target}" ]; then
    compiler_target="$("${PTOAS_CC}" -dumpmachine 2>/dev/null || true)"
  fi
  case "${compiler_target}" in
    aarch64-*|arm64-*) ;;
    *)
      echo "ERROR: ${PTOAS_CC} reported non-aarch64 target: ${compiler_target:-unknown}" >&2
      exit 1
      ;;
  esac

  if [ -d "${compiler_rt_build}" ]; then
    PTOAS_COMPILER_RT="$(
      find "${compiler_rt_build}" -name 'libclang_rt.builtins-aarch64.a' \
        -type f 2>/dev/null | head -1 || true
    )"
    if compiler_rt_has_muloti4 "${PTOAS_COMPILER_RT}"; then
      export PTOAS_COMPILER_RT
      echo "Reusing LLVM compiler-rt: ${PTOAS_COMPILER_RT}"
      return 0
    fi
  fi

  echo "${dotted_line}"
  echo "Building compiler-rt builtins for ${compiler_target}"
  mkdir -p "${compiler_rt_build}"
  cmake \
    -G Ninja \
    -S "${builtins_src}" \
    -B "${compiler_rt_build}" \
    -DLLVM_CMAKE_DIR="${LLVM_BUILD_DIR}/lib/cmake/llvm" \
    -DLLVM_MAIN_SRC_DIR="${LLVM_SOURCE_DIR}/llvm" \
    -DCMAKE_C_COMPILER="${PTOAS_CC}" \
    -DCMAKE_ASM_COMPILER="${PTOAS_CC}" \
    -DCMAKE_C_COMPILER_TARGET="${compiler_target}" \
    -DCMAKE_ASM_COMPILER_TARGET="${compiler_target}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLVM_ENABLE_PER_TARGET_RUNTIME_DIR=OFF \
    -DCOMPILER_RT_DEFAULT_TARGET_ONLY=ON
  cmake --build "${compiler_rt_build}" --target builtins -- -j "${JOBS}"

  PTOAS_COMPILER_RT="$(
    find "${compiler_rt_build}" -name 'libclang_rt.builtins-aarch64.a' \
      -type f 2>/dev/null | head -1 || true
  )"
  if ! compiler_rt_has_muloti4 "${PTOAS_COMPILER_RT}"; then
    echo "ERROR: compiler-rt build did not produce an aarch64 archive with __muloti4" >&2
    exit 1
  fi

  export PTOAS_COMPILER_RT
  echo "Built LLVM compiler-rt: ${PTOAS_COMPILER_RT}"
}

# On aarch64, -ftrapv lowers __int128 multiplication to __muloti4 (compiler-rt).
# Link the matching compiler-rt builtins so the PTOAS executables/libraries
# resolve it. Accepts an explicit override via PTOAS_COMPILER_RT.
resolve_compiler_rt() {
  case "$(uname -m)" in
    aarch64|arm64)
      if [ -n "${PTOAS_COMPILER_RT:-}" ] && [ -f "${PTOAS_COMPILER_RT}" ]; then
        export PTOAS_COMPILER_RT
        echo "Using PTOAS_COMPILER_RT=${PTOAS_COMPILER_RT}"
        return 0
      fi

      local clang_res
      clang_res="$("${PTOAS_CC}" -print-resource-dir 2>/dev/null || true)"
      if [ -z "${clang_res}" ]; then
        echo "ERROR: failed to query compiler resource directory from ${PTOAS_CC}" >&2
        exit 1
      fi

      PTOAS_COMPILER_RT="${clang_res}/lib/linux/libclang_rt.builtins-aarch64.a"
      if [ ! -f "${PTOAS_COMPILER_RT}" ] && [ -d "${clang_res}" ]; then
        PTOAS_COMPILER_RT="$(
          find "${clang_res}" -name 'libclang_rt.builtins-aarch64.a' -type f 2>/dev/null \
            | head -1 || true
        )"
      fi
      if [ -z "${PTOAS_COMPILER_RT}" ] || [ ! -f "${PTOAS_COMPILER_RT}" ]; then
        local runtime_search_roots=()
        local runtime_root
        for runtime_root in /opt/buildtools /usr /usr/local; do
          [ -d "${runtime_root}" ] && runtime_search_roots+=("${runtime_root}")
        done
        PTOAS_COMPILER_RT="$(
          find "${runtime_search_roots[@]}" \
            -name 'libclang_rt.builtins-aarch64.a' -type f 2>/dev/null \
            | head -1 || true
        )"
      fi
      if [ -z "${PTOAS_COMPILER_RT}" ] || [ ! -f "${PTOAS_COMPILER_RT}" ]; then
        echo "compiler-rt not found on host; building from ${LLVM_SOURCE_DIR}/compiler-rt"
        build_aarch64_compiler_rt
      fi

      export PTOAS_COMPILER_RT
      echo "Using LLVM compiler-rt: ${PTOAS_COMPILER_RT}"
      ;;
    *)
      unset PTOAS_COMPILER_RT
      return 0
  esac
}

pip_install_runtime_deps() {
  local python_bin="$1"
  shift
  local index_url="${PTOAS_PIP_INDEX_URL:-}"
  local trusted_host="${PTOAS_PIP_TRUSTED_HOST:-}"

  # Internal CI selects its pinned compiler from /opt/buildtools and cannot
  # reach public PyPI. GitCode selects /usr/bin/clang and keeps pip defaults.
  if [ -z "${index_url}" ] && [[ "${PTOAS_CC}" == /opt/buildtools/* ]]; then
    index_url="http://mirrors.tools.huawei.com/pypi/simple"
    trusted_host="mirrors.tools.huawei.com"
  fi

  if [ -n "${index_url}" ]; then
    local index_args=(-i "${index_url}")
    if [ -n "${trusted_host}" ]; then
      index_args+=(--trusted-host "${trusted_host}")
    fi
    "${python_bin}" -m pip install --no-cache-dir "${index_args[@]}" "$@"
  else
    "${python_bin}" -m pip install --no-cache-dir "$@"
  fi
}

write_ptoas_test_env() {
  local env_file="${BUILD_PATH}/ptoas-test-env.sh"

  mkdir -p "${BUILD_PATH}"
  cat > "${env_file}" <<EOF
# Generated by build.sh. Source this file before running PTO-AS source-tree tests.
export LLVM_BUILD_DIR="${LLVM_BUILD_DIR}"
export MLIR_PYTHON_ROOT="${LLVM_BUILD_DIR}/tools/mlir/python_packages/mlir_core"
export PTO_INSTALL_DIR="${INSTALL_PATH}"
export PTO_PYTHON_ROOT="${INSTALL_PATH}"
export PYTHONPATH="\${MLIR_PYTHON_ROOT}:\${PTO_PYTHON_ROOT}:\${PYTHONPATH:-}"
export LD_LIBRARY_PATH="\${LLVM_BUILD_DIR}/lib:\${PTO_INSTALL_DIR}/lib:\${LD_LIBRARY_PATH:-}"
EOF
}

configure_ptoas() {
  local python_bin
  python_bin="$(command -v python3 || command -v python)"
  local pybind_dir
  pybind_dir="$("${python_bin}" -m pybind11 --cmakedir 2>/dev/null || true)"

  echo "Resetting PTOAS build tree: ${BUILD_PATH}"
  rm -rf "${BUILD_PATH}"
  mkdir -p "${BUILD_PATH}"
  local ptoas_cmake_args=(
    -G Ninja
    -S "${BASE_PATH}"
    -B "${BUILD_PATH}"
    -DLLVM_DIR="${LLVM_BUILD_DIR}/lib/cmake/llvm"
    -DMLIR_DIR="${LLVM_BUILD_DIR}/lib/cmake/mlir"
    -DPython3_EXECUTABLE="${python_bin}"
    -Dpybind11_DIR="${pybind_dir}"
    -DPTO_ENABLE_PYTHON_BINDING=ON
    -DBUILD_TESTING=ON
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_CXX_FLAGS="-DBSPUB_NPU_DATA_TYPE -D_GLIBCXX_USE_CXX11_ABI=0 -Wno-error=deprecated-declarations"
    -DCMAKE_INSTALL_PREFIX="${INSTALL_PATH}"
    -DCMAKE_C_COMPILER="${PTOAS_CC}"
    -DCMAKE_CXX_COMPILER="${PTOAS_CXX}"
    -DENABLE_PACKAGE="${ENABLE_PACKAGE:-FALSE}"
    -DPACKAGE_TYPE="${PACKAGE_TYPE:-run}"
    -DCANN_3RD_LIB_PATH="${CANN_3RD_LIB_PATH}"
  )
  if [ -n "${PTOAS_WHEEL_FILE:-}" ]; then
    ptoas_cmake_args+=("-DPTOAS_WHEEL_FILE=${PTOAS_WHEEL_FILE}")
  fi

  # Inject compiler-rt for __muloti4 (aarch64 -ftrapv __int128) into the
  # linker flags applied to every PTOAS target. The static archive must be
  # pulled in even though linker flags precede the object files: force the
  # symbol with -Wl,-u so the linker extracts __muloti4 from the archive.
  resolve_compiler_rt
  echo "PTOAS_COMPILER_RT=${PTOAS_COMPILER_RT:-NOT_USED}"
  if [ -n "${PTOAS_COMPILER_RT:-}" ]; then
    ptoas_cmake_args+=(
      -DCMAKE_EXE_LINKER_FLAGS="${CMAKE_EXE_LINKER_FLAGS:-} -Wl,-u,__muloti4 ${PTOAS_COMPILER_RT}"
      -DCMAKE_SHARED_LINKER_FLAGS="${CMAKE_SHARED_LINKER_FLAGS:-} -Wl,-u,__muloti4 ${PTOAS_COMPILER_RT}"
      -DCMAKE_MODULE_LINKER_FLAGS="${CMAKE_MODULE_LINKER_FLAGS:-} -Wl,-u,__muloti4 ${PTOAS_COMPILER_RT}"
    )
  fi

  if [ -f "${HARDENING_CACHE_FILE}" ]; then
    # Keep the PTOAS BSPUB definition together with the delivery hardening
    # flags from the preload cache.
    cmake "${ptoas_cmake_args[@]}" -C "${HARDENING_CACHE_FILE}"
  else
    cmake "${ptoas_cmake_args[@]}"
  fi
  write_ptoas_test_env
}

build_only() {
  echo $dotted_line
  echo "build ptoas"
  ensure_llvm_build
  configure_ptoas
  cmake --build "${BUILD_PATH}" -- -j "${JOBS}"
  cmake --install "${BUILD_PATH}"

  echo "execute samples success"
}

# Build the Python distribution used by the installed ptoas console entry.
# The current tree intentionally does not ship a native _runtime/bin/ptoas:
# pyproject.toml exposes ptoas._cli:main and the wheel owns the Python package,
# native extensions, TileOps resources, and console script as one unit.
stage_ptoas_wheel() {
  local python_bin
  python_bin="$(command -v python3 || command -v python)"
  local wheel_dist="${BUILD_PATH}/wheel-dist"
  local wheelhouse="${BUILD_PATH}/wheelhouse"
  local stable_wheelhouse="${BASE_PATH}/.ptoas-wheelhouse"
  local python_scripts
  python_scripts="$("${python_bin}" -c 'import sysconfig; print(sysconfig.get_path("scripts"))')"
  local user_scripts
  user_scripts="$("${python_bin}" -c 'import os, sysconfig; print(sysconfig.get_path("scripts", scheme=os.name + "_user"))')"
  local python_tool_path="${python_scripts}:${user_scripts}:${PATH}"
  local wheel_arch
  local wheel_feature_args=()
  local repair_plat
  local repair_succeeded=false

  case "$(uname -m)" in
    aarch64|arm64) wheel_arch="aarch64" ;;
    x86_64|amd64) wheel_arch="x86_64" ;;
    *)
      echo "ERROR: unsupported wheel architecture: $(uname -m)" >&2
      exit 1
      ;;
  esac

  rm -rf "${wheel_dist}" "${wheelhouse}" "${stable_wheelhouse}" "${PACKAGE_STAGE_PATH}"
  mkdir -p "${wheel_dist}" "${wheelhouse}" \
    "${PACKAGE_STAGE_PATH}/tools/ptoas/wheels"

  echo "Building PTOAS wheel"
  # CI buildtools python ships pip 21.2.4, which predates --config-settings
  # (required by scikit-build-core, added in pip 22.1). Upgrade pip first so
  # the scikit-build-core flags below are accepted.
  pip_install_runtime_deps "${python_bin}" --upgrade "pip>=22.1"
  pip_install_runtime_deps "${python_bin}" \
    numpy \
    'pybind11<3' \
    'scikit-build-core>=0.12.2,<2'
  # pip 21.2 copies local projects out of tree unless this transitional
  # feature is enabled, which conflicts with the existing CMake cache. Newer
  # pip builds in tree by default and has removed the feature flag entirely.
  if "${python_bin}" -m pip wheel \
       --use-feature=in-tree-build --help >/dev/null 2>&1; then
    wheel_feature_args+=(--use-feature=in-tree-build)
  fi
  CMAKE_BUILD_PARALLEL_LEVEL="${JOBS}" \
  SKBUILD_BUILD_DIR="${BUILD_PATH}" \
  LLVM_BUILD_DIR="${LLVM_BUILD_DIR}" \
    "${python_bin}" -m pip wheel "${BASE_PATH}" \
      "${wheel_feature_args[@]}" \
      --config-settings=wheel.py-api=cp37 \
      --no-build-isolation \
      --no-deps \
      --wheel-dir "${wheel_dist}"
  "${python_bin}" "${BASE_PATH}/docker/validate_wheel_payload.py" \
    "${wheel_dist}"

  # The GitCode build uses the cached shared LLVM tree. Repairing the wheel
  # makes those DSOs package-relative so the smoke job does not depend on the
  # build cache still being mounted when the .run artifact is installed.
  if ! "${python_bin}" -c 'import auditwheel' >/dev/null 2>&1 \
     || ! PATH="${python_tool_path}" command -v patchelf >/dev/null 2>&1; then
    pip_install_runtime_deps "${python_bin}" auditwheel patchelf
  fi
  if ! "${python_bin}" -c 'import auditwheel' >/dev/null 2>&1; then
    echo "ERROR: auditwheel is not importable by ${python_bin}" >&2
    exit 1
  fi
  if ! PATH="${python_tool_path}" command -v patchelf >/dev/null 2>&1; then
    echo "ERROR: patchelf not found in ${python_scripts} or ${user_scripts}" >&2
    exit 1
  fi
  # The shared LLVM cache is built by the GitCode image rather than the
  # manylinux_2_34 container used by the release workflow. Its versioned
  # symbols can therefore require a newer PEP 600 policy. Try supported
  # policies in compatibility order and keep the oldest one auditwheel can
  # honestly assign to this wheel.
  for repair_plat in \
    "manylinux_2_34_${wheel_arch}" \
    "manylinux_2_35_${wheel_arch}" \
    "manylinux_2_36_${wheel_arch}" \
    "manylinux_2_37_${wheel_arch}" \
    "manylinux_2_38_${wheel_arch}" \
    "manylinux_2_39_${wheel_arch}"; do
    echo "Trying auditwheel platform: ${repair_plat}"
    if PATH="${python_tool_path}" \
       LD_LIBRARY_PATH="${LLVM_BUILD_DIR}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
         "${python_bin}" -m auditwheel repair \
           --plat "${repair_plat}" \
           --wheel-dir "${wheelhouse}" \
           "${wheel_dist}"/ptoas*.whl; then
      repair_succeeded=true
      break
    fi
  done
  if [ "${repair_succeeded}" != true ]; then
    echo "ERROR: auditwheel could not repair the PTOAS wheel with a supported policy" >&2
    exit 1
  fi
  "${python_bin}" "${BASE_PATH}/docker/validate_wheel_payload.py" \
    "${wheelhouse}"

  mkdir -p "${stable_wheelhouse}"
  cp "${wheelhouse}"/ptoas*.whl "${stable_wheelhouse}/"
  PTOAS_WHEEL_FILE="$(realpath "${stable_wheelhouse}"/ptoas*.whl)"
  export PTOAS_WHEEL_FILE
  echo "staged ptoas wheel: ${PTOAS_WHEEL_FILE}"
}

# Package the wheel into a self-extracting .run installer under build_out. The
# GitCode smoke pipeline invokes the artifact with --full / --uninstall. Keep
# the wheel installer here because the legacy install-tree helper only extracts
# files and cannot create the Python console entry required by runop.sh.
package() {
  echo $dotted_line
  echo "package ptoas"
  ensure_llvm_build
  ENABLE_PACKAGE=FALSE configure_ptoas
  cmake --build "${BUILD_PATH}" -- -j "${JOBS}"

  # Distill the version used for the .run package name. The CANN product version
  # (9.2.0 for this release train) differs from project(ptoas VERSION 0.57), so
  # default to the packaging version and allow an explicit override.
  PTOAS_PACKAGE_VERSION="${PTOAS_PACKAGE_VERSION:-9.2.0}"

  # The placeholder package intentionally does not invoke pip/auditwheel.  The
  # native build above is still performed, and CMake/CPack owns generation of
  # the same .run/RPM/DEB package formats.  Keep the old wheel path available
  # behind an explicit opt-in for local release builds.
  rm -rf "${BUILD_OUT_PATH}"
  mkdir -p "${BUILD_OUT_PATH}"
  unset PTOAS_WHEEL_FILE
  if [ "${PTOAS_PLACEHOLDER_RUN_PACKAGE}" != "TRUE" ]; then
    stage_ptoas_wheel
  else
    echo "Building placeholder PTOAS package without a Python wheel"
  fi
  ENABLE_PACKAGE=TRUE
  configure_ptoas
  # configure_ptoas resets the build tree; rebuild all targets before the
  # install/CPack pass so generated install scripts reference real artifacts.
  cmake --build "${BUILD_PATH}" -- -j "${JOBS}"
  cmake --install "${BUILD_PATH}"
  cmake --build "${BUILD_PATH}" --target package -- -j "${JOBS}"
  echo "package staged under ${BUILD_OUT_PATH}"
  # Diagnostics: the OBS uploader reads build_out via the host path
  # /opt/cloud/slavespace/.../x86build/build_out; print what we actually
  # ~created so a path mismatch is visible in the CI log.
  echo "BUILD_OUT absolute: $(cd "${BUILD_OUT_PATH}" && pwd -P)"
  ls -la "${BUILD_OUT_PATH}"
}

# The CANN PreSmoke driver invokes test/samples/runop.sh after build.sh exits.
# Persist the decision in the build tree because exports from this child shell
# cannot affect that later command in the parent CI process.
write_presmoke_runop_policy() {
  if [ "${SMOKE_TYPE:-}" = "pre" ]; then
    mkdir -p "${BUILD_PATH}"
    touch "${PTOAS_PRESMOKE_SKIP_RUNOP_MARKER}"
    echo "PreSmoke runop smoke disabled: ${PTOAS_PRESMOKE_SKIP_RUNOP_MARKER}"
  else
    rm -f "${PTOAS_PRESMOKE_SKIP_RUNOP_MARKER}"
  fi
}

main() {
  JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"
  ENABLE_BUILD_ONLY=FALSE
  ENABLE_PACKAGE=FALSE

  while [ $# -gt 0 ]; do
    case "$1" in
      -h|--help)
        usage
        exit 0
        ;;
      --build)
        ENABLE_BUILD_ONLY=TRUE
        shift
        ;;
      --pkg)
        ENABLE_PACKAGE=TRUE
        shift
        ;;
      --pkg-type|--pkg-type=*)
        # Preserve the master package type interface for CPack.
        case "$1" in
          --pkg-type=*)
            PACKAGE_TYPE="${1#--pkg-type=}"
            shift
            ;;
          *)
            PACKAGE_TYPE="$2"
            shift 2
            ;;
        esac
        ;;
      -j)
        JOBS="$2"
        shift 2
        ;;
      -j=*)
        JOBS="${1#-j=}"
        shift
        ;;
      --cann_3rd_lib_path)
        CANN_3RD_LIB_PATH="$2"
        shift 2
        ;;
      --cann_3rd_lib_path=*)
        CANN_3RD_LIB_PATH="${1#--cann_3rd_lib_path=}"
        shift
        ;;
      *)
        usage
        exit 1
        ;;
    esac
  done

  if [ "$ENABLE_BUILD_ONLY" == "TRUE" ] || [ "$ENABLE_PACKAGE" == "TRUE" ]; then
    resolve_ptoas_toolchain
  fi

  prepare_llvm_cache_layout

  if [ "$ENABLE_BUILD_ONLY" == "TRUE" ]; then
    build_only
  fi
  if [ "$ENABLE_PACKAGE" == "TRUE" ]; then
    package
  fi
  if [ "$ENABLE_BUILD_ONLY" == "TRUE" ] || [ "$ENABLE_PACKAGE" == "TRUE" ]; then
    write_presmoke_runop_policy
  fi
  if [ "$ENABLE_BUILD_ONLY" != "TRUE" ] && [ "$ENABLE_PACKAGE" != "TRUE" ]; then
    usage
  fi
}

set -o pipefail
main "$@"
