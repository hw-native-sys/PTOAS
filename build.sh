#!/bin/bash
# --------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------
#
# Build driver for the LLVM 19 snapshot tree (project ptoas, the GitHub PTOAS
# layout). This is the entry point used by the gitcode smoke pipeline
# (./build.sh --build / --pkg). It builds the external LLVM/MLIR 19 dependency
# (reusing a cached LLVM source/build when available) and then builds and
# installs PTOAS through the tree's native CMake build. Both Actions and robot
# PreSmoke consume a cached or downloaded Compile installer; PreSmoke never
# builds from source.

set -e

dotted_line="----------------------------------------------------------------"
COLOR_RESET="\033[0m"
COLOR_GREEN="\033[32m"
COLOR_RED="\033[31m"

export BASE_PATH=$(
  cd "$(dirname "$0")"
  pwd
)
export BUILD_PATH="${BASE_PATH}/build"
export BUILD_OUT_PATH="${BASE_PATH}/build_out"
export INSTALL_PATH="${BASE_PATH}/install"
export PACKAGE_STAGE_PATH="${BUILD_PATH}/package_runtime"
export PTOAS_PRESMOKE_SKIP_RUNOP_MARKER="${BUILD_PATH}/.skip-presmoke-runop"
export LLVM_SOURCE_VERSION="19.1.7"
export PTOAS_GLIBCXX_ABI="${PTOAS_GLIBCXX_ABI:-0}"
# The LLVM 19 source (upstream llvmorg-19.1.7 tarball + feature-vpto patch)
# is fetched by cann-cmake's third_party/llvm.cmake via ExternalProject_Add.
# ensure_llvm_source() drives a minimal CMake project that includes the module
# and builds the third_party_llvm target, so the download+patch runs at build
# time. cann-cmake itself (providing llvm.cmake) is pulled by
# cmake/fetch_cann_cmake.cmake. See ensure_cann_cmake / ensure_llvm_source.
# Prefer ASCEND_3RD_LIB_PATH when it points to a valid LLVM source cache
# (CI images set this to /home/jenkins/opensource). Fall back to the in-tree
# third_party directory for local builds where it is unset.
if [ -n "${ASCEND_3RD_LIB_PATH}" ] && [ -d "${ASCEND_3RD_LIB_PATH}/llvm-19" ]; then
    CANN_3RD_LIB_PATH="${ASCEND_3RD_LIB_PATH}"
else
    CANN_3RD_LIB_PATH="${BASE_PATH}/third_party"
fi
HARDENING_CACHE_FILE="${BASE_PATH}/cmake/LinuxHardeningCache.cmake"
# Only enable the CentOS7 devtoolset-7 sysroot + gcc-toolchain when the
# toolchain is actually present AND the host glibc is newer than CentOS7.
# On a CentOS7 CI image the host glibc is already 2.17, so linking against the
# devtoolset sysroot is a no-op and would instead invalidate the pre-seeded
# LLVM cache (whose flags carry no --sysroot), forcing a full LLVM rebuild
# that times out. On newer hosts (Ubuntu 22.04 glibc 2.35, manylinux) the
# sysroot flags lower the packaged libraries' glibc floor from the host value
# down to 2.17.
#
# A /opt/rh/devtoolset-7 stub can also linger on non-CentOS7 CI images
# (e.g. ubuntu24.04_x86) without the actual GCC 7 tree. gating on the
# directory alone made those builds link libLLVMSupport/llvm-min-tblgen
# against a nonexistent libstdc++ and hang for hours, so the sysroot is
# only accepted when the arch-matching GCC 7 toolchain tree is complete
# AND a probe C++ link against it succeeds within a timeout.
devtoolset7_tree_is_usable() {
  local root="/opt/rh/devtoolset-7/root"
  [ -d "${root}" ] || return 1
  [ -x "${root}/usr/bin/gcc" ] || return 1
  { [ -f "${root}/lib64/libc.so.6" ] || [ -f "${root}/lib/libc.so.6" ] \
    || [ -f "${root}/lib/aarch64-linux-gnu/libc.so.6" ] \
    || [ -f "${root}/lib/x86_64-linux-gnu/libc.so.6" ]; } || return 1
  # The GCC 7 install triplet varies across images: redhat-style
  # (x86_64-redhat-linux / aarch64-unknown-linux-gnu) and the ubuntu-built
  # devtoolset on the X86 image (x86_64-pc-linux-gnu). Locate the actual
  # gcc/7 directory by globbing instead of hardcoding one triplet.
  local gcc_dir=""
  local arch_triplet=""
  local _candidate
  for _candidate in "${root}"/usr/lib/gcc/*/7; do
    [ -d "${_candidate}" ] || continue
    if [ -f "${_candidate}/crtbegin.o" ]         && ls "${_candidate}"/libstdc++.so* >/dev/null 2>&1; then
      gcc_dir="${_candidate}"
      arch_triplet="$(basename "$(dirname "${_candidate}")")"
      break
    fi
  done
  [ -n "${gcc_dir}" ] || return 1
  [ -d "${root}/usr/include/c++/7" ] || return 1
  # The arch-specific C++ header dir may be missing on some minimal trees;
  # accept the gcc install triplet or any arch subdir under c++/7.
  local _cxx_inc_arch="${root}/usr/include/c++/7/${arch_triplet}"
  if [ ! -d "${_cxx_inc_arch}" ]; then
    local _d
    for _d in "${root}"/usr/include/c++/7/*/; do
      case "$(basename "${_d}")" in
        backward|ext) continue ;;
      esac
      _cxx_inc_arch="${_d%/}"
      break
    done
  fi
  [ -d "${_cxx_inc_arch}" ] || return 1

  # Probe-link a tiny C++ program with the exact flags we will use. A stub
  # tree passes the layout checks but can still deadlock the linker (seen on
  # CI: llvm-min-tblgen link hung for hours), so require a real, timed link.
  # Locate a working C/C++ driver first (always present on build hosts).
  local cc_bin=""
  if [ -n "${PTOAS_CC:-}" ] && [ -x "${PTOAS_CC}" ]; then
    cc_bin="${PTOAS_CC}"
  elif command -v clang++ >/dev/null 2>&1; then
    cc_bin="$(command -v clang++)"
  elif command -v g++ >/dev/null 2>&1; then
    cc_bin="$(command -v g++)"
  fi
  [ -n "${cc_bin}" ] || return 1

  local probe_src="${TMPDIR:-/tmp}/ptoas_dts7_probe.cpp"
  local probe_bin="${TMPDIR:-/tmp}/ptoas_dts7_probe.$$"
  printf '#include <string>\nint main(){ std::string s("ok"); return (int)s.size()==2?0:1; }\n' \
    > "${probe_src}" 2>/dev/null || return 1
  # 30s is far beyond a healthy link; a hanging stub tree gets killed promptly.
  timeout 30 "${cc_bin}" "${probe_src}" -o "${probe_bin}" \
      --sysroot="${root}" --gcc-toolchain="${root}/usr" \
      >/dev/null 2>&1
  local link_rc=$?
  rm -f "${probe_src}" "${probe_bin}"
  [ "${link_rc}" -eq 0 ] || return 1
  return 0
}

DEVTOOLSET_TOOLCHAIN_FLAGS=""
resolve_devtoolset_toolchain() {
  if devtoolset7_tree_is_usable; then
    local glibc_version
    glibc_version="$(ldd --version 2>/dev/null)"
    # Read the complete output before selecting its first line. Piping ldd to
    # head can raise SIGPIPE and abort this function under set -o pipefail.
    if [[ "${glibc_version%%$'\n'*}" =~ ([0-9]+)\.([0-9]+) ]]; then
      local glibc_major="${BASH_REMATCH[1]}"
      local glibc_minor="${BASH_REMATCH[2]}"
      if [ "${glibc_major}" -gt 2 ] \
          || { [ "${glibc_major}" -eq 2 ] && [ "${glibc_minor}" -gt 17 ]; }; then
        DEVTOOLSET_TOOLCHAIN_FLAGS="--sysroot=/opt/rh/devtoolset-7/root --gcc-toolchain=/opt/rh/devtoolset-7/root/usr"
      fi
    fi
  fi
}

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
  echo "PreSmoke reuses a Compile run installer and never builds from source."
  echo "Downloads use obs_path (Actions) or GIT_PR_NUMBER/pr_id (robot)."
}

# Keep LLVM parallelism at nproc (or an explicit -j). PTOAS applies its
# separate limit at each native build entry point, including the wheel.
resolve_build_jobs() {
  local cpus requested
  cpus="$(nproc 2>/dev/null || echo 4)"
  requested="${JOBS:-${cpus}}"
  if [[ ! "${requested}" =~ ^[1-9][0-9]{0,5}$ ]]; then
    echo "ERROR: -j/JOBS must be a positive integer of at most six digits" >&2
    return 1
  fi
  JOBS="${requested}"
  export CMAKE_BUILD_PARALLEL_LEVEL="${JOBS}"
  echo "Build resources: nproc=${cpus}, requested=${requested}, jobs=${JOBS}"
}

# PTOAS has large TableGen-generated translation units. Bound their
# concurrency without reducing LLVM parallelism or using cgroup memory
# estimates. Honor explicit job counts below the limit.
ptoas_build_jobs() {
  echo "$(( JOBS > 16 ? 16 : JOBS ))"
}

report_build_failure() {
  local status="$1"
  [ "${status}" -ne 0 ] || return 0
  echo "Build exited with status ${status}; jobs=${JOBS:-unknown}" >&2
  local counter
  for counter in /sys/fs/cgroup/memory.events /sys/fs/cgroup/memory.peak \
      /sys/fs/cgroup/memory/memory.failcnt; do
    if [ -r "${counter}" ]; then
      echo "${counter}:" >&2
      cat "${counter}" >&2 || true
    fi
  done
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

# Ensure cann-cmake (the CANN CMake infrastructure repo providing
# third_party/llvm.cmake) is present under ${CANN_3RD_LIB_PATH}/cann-cmake.
# fetch_cann_cmake.cmake runs in cmake script mode and uses git clone, so it
# works without a project() declaration.
ensure_cann_cmake() {
  if [ -d "${CANN_3RD_LIB_PATH}/cann-cmake/third_party/llvm.cmake" ]; then
    return 0
  fi
  echo "${dotted_line}"
  echo "Fetching cann-cmake (CANN CMake infrastructure)"
  mkdir -p "${CANN_3RD_LIB_PATH}"
  cmake -DPROJECT_SOURCE_DIR="" -DCANN_3RD_LIB_PATH="${CANN_3RD_LIB_PATH}" \
    -P "${BASE_PATH}/cmake/fetch_cann_cmake.cmake" || {
      echo "ERROR: failed to fetch cann-cmake" >&2
      exit 1
    }
}

# Ensure the LLVM 19 source tree is present under ${LLVM_SOURCE_DIR}. A cached
# tree carrying SimtEntry is reused as-is. Otherwise the source is fetched via
# cann-cmake's third_party/llvm.cmake: a minimal CMake project includes the
# module, declaring an ExternalProject_Add target (third_party_llvm) that
# downloads the llvmorg-19.1.7 tarball, applies the feature-vpto patch, and
# extracts to ${LLVM_SOURCE_DIR}. The download+patch runs at build time, not
# configure time, matching the cann-cmake integration pattern used across CANN
# repos (add_cann_third_party + cmake --build --target <name>).
# Apply the feature-vpto patch in place to an existing (unpatched) LLVM
# source tree. CI images pre-seed the third-party cache with the pristine
# llvmorg-19.1.7 source; deleting and re-fetching the tree (as an earlier
# revision did) invalidates the pre-seeded LLVM build cache and forces a full
# LLVM rebuild that exceeds the smoke job time limit. Patching in place keeps
# the untouched files' build cache entries valid. The patch source and hash
# mirror cann-cmake third_party/llvm.cmake (LLVM_VPTO_PATCH_FILE logic).
apply_vpto_patch_inplace() {
  echo "${dotted_line}"
  echo "Cached LLVM source lacks SimtEntry; applying feature-vpto patch in place"
  local patch_file=""
  if [ -f "${CANN_3RD_LIB_PATH}/feature-vpto-last3.patch" ]; then
    patch_file="${CANN_3RD_LIB_PATH}/feature-vpto-last3.patch"
  elif [ -f "${CANN_3RD_LIB_PATH}/pkg/feature-vpto-last3.patch" ]; then
    patch_file="${CANN_3RD_LIB_PATH}/pkg/feature-vpto-last3.patch"
  else
    mkdir -p "${CANN_3RD_LIB_PATH}/pkg"
    patch_file="${CANN_3RD_LIB_PATH}/pkg/feature-vpto-last3.patch"
    echo "Downloading vpto patch from https://gitcode.com/cann-src-third-party/llvm/releases/download/19.1.7-h0/feature-vpto-last3.patch"
    curl -fL --retry 3 -o "${patch_file}"       "https://gitcode.com/cann-src-third-party/llvm/releases/download/19.1.7-h0/feature-vpto-last3.patch" || {
        echo "ERROR: failed to download vpto patch" >&2
        exit 1
      }
    local actual_sha
    actual_sha="$(sha256sum "${patch_file}" | cut -d ' ' -f1)"
    if [ "${actual_sha}" != "a49c1d3dd8ab78e93264712bc0d46deb536196a54abb2c2ee02abd914cd385e2" ]; then
      echo "ERROR: vpto patch SHA256 mismatch: ${actual_sha}" >&2
      exit 1
    fi
  fi
  (cd "${LLVM_SOURCE_DIR}" && patch -p1 < "${patch_file}") || {
    echo "ERROR: failed to apply vpto patch to ${LLVM_SOURCE_DIR}" >&2
    exit 1
  }
  if ! llvm_has_simt_entry; then
    echo "ERROR: LLVM source still lacks SimtEntry after patching" >&2
    exit 1
  fi
  echo "Applied vpto patch in place: ${patch_file}"
}

ensure_llvm_source() {
  if [ -f "${LLVM_SOURCE_DIR}/llvm/CMakeLists.txt" ]; then
    export LLVM_CMAKE_SOURCE_DIR="${LLVM_SOURCE_DIR}/llvm"
    if ! llvm_has_simt_entry; then
      apply_vpto_patch_inplace
    fi
    return 0
  fi
  if [ -f "${LLVM_SOURCE_DIR}/CMakeLists.txt" ]; then
    export LLVM_CMAKE_SOURCE_DIR="${LLVM_SOURCE_DIR}"
    if ! llvm_has_simt_entry; then
      apply_vpto_patch_inplace
    fi
    return 0
  fi

  ensure_cann_cmake

  echo "${dotted_line}"
  echo "Fetching LLVM ${LLVM_SOURCE_VERSION} source via cann-cmake third_party/llvm.cmake"
  local llvm_fetch_dir="${BUILD_PATH}/llvm_fetch"
  rm -rf "${llvm_fetch_dir}"
  mkdir -p "${llvm_fetch_dir}"
  cat > "${llvm_fetch_dir}/CMakeLists.txt" <<EOF
cmake_minimum_required(VERSION 3.20.0)
project(ptoas_llvm_fetch NONE)
set(CANN_3RD_LIB_PATH "${CANN_3RD_LIB_PATH}" CACHE PATH "")
include("${CANN_3RD_LIB_PATH}/cann-cmake/third_party/llvm.cmake")
EOF
  cmake -S "${llvm_fetch_dir}" -B "${llvm_fetch_dir}/build" || {
    echo "ERROR: cmake configure of LLVM fetch project failed" >&2
    exit 1
  }
  cmake --build "${llvm_fetch_dir}/build" --target third_party_llvm || {
    echo "ERROR: failed to fetch LLVM source via cann-cmake" >&2
    exit 1
  }

  if [ -f "${LLVM_SOURCE_DIR}/llvm/CMakeLists.txt" ]; then
    export LLVM_CMAKE_SOURCE_DIR="${LLVM_SOURCE_DIR}/llvm"
  elif [ -f "${LLVM_SOURCE_DIR}/CMakeLists.txt" ]; then
    export LLVM_CMAKE_SOURCE_DIR="${LLVM_SOURCE_DIR}"
  else
    echo "ERROR: LLVM source not found at ${LLVM_SOURCE_DIR} after fetch" >&2
    exit 1
  fi
  if ! llvm_has_simt_entry; then
    echo "ERROR: fetched LLVM source still lacks SimtEntry; patch may have failed" >&2
    exit 1
  fi
}

# LLVM/MLIR and PTOAS are both built with the old libstdc++ ABI. Reject a
# cached LLVM tree that exports the new-ABI Twine::str symbol.
llvm_build_is_abi_compatible() {
  local support_lib="${LLVM_BUILD_DIR}/lib/libLLVMSupport.so.19.1"
  [ -f "${support_lib}" ] || return 1
  # _ZNK4llvm5Twine3strEv -> _GLIBCXX_USE_CXX11_ABI=0
  # Drain nm output: grep -q can close the pipe early and make a valid cache
  # fail with SIGPIPE under pipefail, forcing LLVM to be rebuilt on every run.
  nm -D --defined-only "${support_lib}" 2>/dev/null \
    | grep -F "_ZNK4llvm5Twine3strEv" >/dev/null
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

# The CentOS7 devtoolset-7 sysroot lowers the GLIBC/GLIBCXX dependency floor of
# the packaged runtime libraries. A cache produced without it (e.g. built on a
# manylinux/Ubuntu image) still links against the host glibc and must not be
# reused when the devtoolset-7 toolchain is present: the resulting .run package
# would silently require the newer host libc at install time.
llvm_build_has_devtoolset_sysroot() {
  # Only enforced when the build is actually going to apply the devtoolset
  # sysroot flags (newer host glibc). On a CentOS7 CI image the flags are
  # disabled and the pre-seeded host-glibc LLVM cache stays valid.
  [ -n "${DEVTOOLSET_TOOLCHAIN_FLAGS}" ] || return 0
  [ -d "/opt/rh/devtoolset-7/root" ] || return 0
  local cache_file="${LLVM_BUILD_DIR}/CMakeCache.txt"
  [ -f "${cache_file}" ] || return 1
  grep -Eq '^CMAKE_C_FLAGS:[^=]*=.*--sysroot=/opt/rh/devtoolset-7/root' "${cache_file}" \
    && grep -Eq '^CMAKE_CXX_FLAGS:[^=]*=.*--sysroot=/opt/rh/devtoolset-7/root' "${cache_file}"
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
    | grep -F 'libLLVMTargetParser.so' >/dev/null
}

# Check every property that makes the shared LLVM cache safe for PTOAS. This
# is deliberately stricter than checking for LLVMConfig.cmake alone: a cache
# from the CANN toolchain may be complete but use the opposite libstdc++ ABI
# or omit the TargetParser dependency required by LLVMVectorize.
llvm_build_cache_is_usable() {
  local reason
  if [ ! -f "${LLVM_BUILD_DIR}/lib/cmake/llvm/LLVMConfig.cmake" ] \
     || [ ! -f "${LLVM_BUILD_DIR}/lib/cmake/mlir/MLIRConfig.cmake" ]; then
    reason="LLVM/MLIR CMake package files are missing"
  elif llvm_has_simt_entry \
       && [ -f "${LLVM_BUILD_DIR}/include/llvm/IR/CallingConv.h" ] \
       && ! grep -q "SimtEntry" "${LLVM_BUILD_DIR}/include/llvm/IR/CallingConv.h"; then
    reason="cached CallingConv.h lacks SimtEntry"
  elif ! llvm_build_is_abi_compatible; then
    reason="libLLVMSupport is missing, unreadable, or has an incompatible C++ ABI"
  elif ! llvm_build_has_bspub_npu_data_type; then
    reason="BSPUB_NPU_DATA_TYPE flags do not match"
  elif ! llvm_build_has_devtoolset_sysroot; then
    reason="devtoolset-7 sysroot flags do not match"
  elif ! llvm_build_uses_shared_components; then
    reason="LLVM shared-component settings do not match"
  elif ! llvm_build_links_vectorize_target_parser; then
    reason="LLVMVectorize linker flags lack LLVMTargetParser"
  elif ! llvm_vectorize_has_target_parser_dependency; then
    reason="LLVMVectorize is missing, unreadable, or lacks its LLVMTargetParser dependency"
  else
    return 0
  fi
  echo "LLVM/MLIR cache miss at ${LLVM_BUILD_DIR}: ${reason}" >&2
  return 1
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
  # reuse it instead of repeating the LLVM build. When the devtoolset sysroot
  # is active (newer host glibc) keep the low-glibc tree in its own keyed
  # directory; otherwise fall back to the default shared cache location.
  local _dts7_key=""
  [ -n "${DEVTOOLSET_TOOLCHAIN_FLAGS}" ] && _dts7_key="-dts7"
  local keyed_llvm_build_dir="${default_llvm_build_dir}-ptoas-abi${PTOAS_GLIBCXX_ABI}-bspub${_dts7_key}-shared"
  unset _dts7_key
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

  # CI's BuildAccelerate/NextCache injects itself through an LD_PRELOAD
  # exec hook (libxcache_hook.so) that intercepts clang/ld invocations. When
  # the devtoolset sysroot is active the intercepted build misbehaves, so
  # detach the whole cmake/ninja subtree from the hook by unsetting
  # LD_PRELOAD. CentOS7 CI (glibc already 2.17, no sysroot) keeps the hook
  # and its acceleration intact.
  if [ -n "${DEVTOOLSET_TOOLCHAIN_FLAGS}" ] && [ -n "${LD_PRELOAD:-}" ]; then
    _hook_lib="${LD_PRELOAD}"
    case "${_hook_lib}" in
      *libxcache_hook.so*|*nextbuild*)
        echo "Note: unsetting LD_PRELOAD (${_hook_lib}) to detach the build from the xcache exec hook"
        unset LD_PRELOAD
        ;;
    esac
    unset _hook_lib
  fi

  local python_bin
  python_bin="$(command -v python3 || command -v python)"

  local pybind_dir
  pybind_dir="$("${python_bin}" -m pybind11 --cmakedir 2>/dev/null || true)"

  local llvm_c_flags="-DBSPUB_NPU_DATA_TYPE"
  local llvm_cxx_flags="-DBSPUB_NPU_DATA_TYPE -D_GLIBCXX_USE_CXX11_ABI=${PTOAS_GLIBCXX_ABI}"
  local llvm_linker_flags=""
  if [ -n "${DEVTOOLSET_TOOLCHAIN_FLAGS}" ]; then
    # Lower the linked GLIBC/GLIBCXX floor of the runtime libraries to the
    # CentOS7 devtoolset-7 sysroot instead of the host libc.
    llvm_c_flags="${llvm_c_flags} ${DEVTOOLSET_TOOLCHAIN_FLAGS}"
    llvm_cxx_flags="${llvm_cxx_flags} ${DEVTOOLSET_TOOLCHAIN_FLAGS}"
    # clang 15 + the devtoolset-7 sysroot headers (glibc 2.17 __REDIRECT
    # fortify macros) miscompile llvm::sys::fs::readNativeFileSlice into an
    # infinite self-loop when -D_FORTIFY_SOURCE is active: the tblgen binary
    # then spins at 100% CPU forever on its first file read and the build
    # hangs (observed as the [185/202] llvm-min-tblgen "link" hang). The
    # LLVM build tools are not part of the delivered run package, so drop
    # FORTIFY for this build only. The delivered PTOAS libraries keep the
    # full hardening flags.
    llvm_c_flags="${llvm_c_flags} -U_FORTIFY_SOURCE -D_FORTIFY_SOURCE=0"
    llvm_cxx_flags="${llvm_cxx_flags} -U_FORTIFY_SOURCE -D_FORTIFY_SOURCE=0"
    # LLVM's sandbox/C utility binaries (bin/count etc.) are C programs linked
    # with the C driver, which never injects -lstdc++. With the devtoolset
    # sysroot in effect those executables pull libLLVMSupport.so (a C++ DSO)
    # and the link fails on the libstdc++ symbols unless stdc++ is provided
    # explicitly. Keep it in the runtime linker flags so both the C and C++
    # targets resolve their libstdc++ dependency against the sysroot.
    llvm_linker_flags="-fuse-ld=lld -lstdc++"
  fi

  local cmake_args=(
    -G Ninja
    -S "${LLVM_CMAKE_SOURCE_DIR}"
    -B "${LLVM_BUILD_DIR}"
    # PTOAS consumes LLVM/MLIR through CMake; Clang is not a PTOAS dependency.
    # Keeping it out avoids building clangInterpreter, which is incompatible
    # with the GCC 7 libstdc++ headers used by the ARM CI image.
    -DLLVM_ENABLE_PROJECTS="mlir"
    -DCMAKE_CXX_FLAGS="${llvm_cxx_flags}"
    -DCMAKE_C_FLAGS="${llvm_c_flags}"
    -DCMAKE_EXE_LINKER_FLAGS="${llvm_linker_flags}"
    -DCMAKE_SHARED_LINKER_FLAGS="${llvm_linker_flags}"
    -DCMAKE_MODULE_LINKER_FLAGS="${llvm_linker_flags}"
    -DLLVM_BSPUB_NPU_DATA_TYPE=ON
    -DBUILD_SHARED_LIBS=ON
    -DLLVM_BUILD_LLVM_DYLIB=OFF
    -DLLVM_LINK_LLVM_DYLIB=OFF
    -DLLVM_USE_LINKER=lld
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
  # The devtoolset sysroot link has been observed to hang indefinitely on
  # some CI ARM executors (no error, no CPU, [185/202] llvm-min-tblgen stuck
  # for hours) while the identical command completes in 0.1s elsewhere.
  # Build the small executable-link step serially under a timeout so a hang
  # is detected early instead of burning the 2h job limit; on timeout print
  # the exact link command and a process snapshot for diagnosis, then retry
  # once with a fully sanitized environment (env -i) which has been observed
  # to unstick similar exec-hook interactions.
  if [ -n "${DEVTOOLSET_TOOLCHAIN_FLAGS}" ]; then
    if ! timeout 300 cmake --build "${LLVM_BUILD_DIR}" --target llvm-min-tblgen -- -j 1 -v; then
      echo "WARNING: llvm-min-tblgen link timed out or failed; dumping diagnostics and retrying once" >&2
      ps -ef | grep -E "ld|lld|clang" | grep -v grep >&2 || true
      ninja -C "${LLVM_BUILD_DIR}" -t commands llvm-min-tblgen 2>/dev/null | tail -1 >&2 || true
      env -i PATH="${PATH}" HOME="${HOME}"         timeout 300 ninja -C "${LLVM_BUILD_DIR}" llvm-min-tblgen -j 1 -v || {
          echo "ERROR: llvm-min-tblgen link failed after sanitized retry" >&2
          exit 1
        }
    fi
  fi
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
  if is_presmoke; then
    touch "${PTOAS_PRESMOKE_SKIP_RUNOP_MARKER}"
  fi
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
    -DCMAKE_CXX_FLAGS="-DBSPUB_NPU_DATA_TYPE -D_GLIBCXX_USE_CXX11_ABI=0 -Wno-error=deprecated-declarations${DEVTOOLSET_TOOLCHAIN_FLAGS:+ ${DEVTOOLSET_TOOLCHAIN_FLAGS}}"
    -DCMAKE_INSTALL_PREFIX="${INSTALL_PATH}"
    -DCMAKE_C_COMPILER="${PTOAS_CC}"
    -DCMAKE_CXX_COMPILER="${PTOAS_CXX}"
    -DENABLE_PACKAGE="${ENABLE_PACKAGE:-FALSE}"
    -DPACKAGE_TYPE="${PACKAGE_TYPE:-run}"
    -DCANN_3RD_LIB_PATH="${CANN_3RD_LIB_PATH}"
  )
  if [ -n "${DEVTOOLSET_TOOLCHAIN_FLAGS:-}" ]; then
    # Inject the devtoolset-7 sysroot + gcc-toolchain into C compilation.
    # `-D` overrides the `-C` hardening preload, so carry the hardening flags
    # explicitly alongside the sysroot instead of letting them be dropped.
    ptoas_cmake_args+=(
      "-DCMAKE_C_FLAGS=-D_FORTIFY_SOURCE=2 -fstack-protector-strong -ftrapv ${DEVTOOLSET_TOOLCHAIN_FLAGS}"
    )
  fi
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
  local _ptoas_jobs
  _ptoas_jobs="$(ptoas_build_jobs)"
  echo "PTOAS build parallelism: jobs=${_ptoas_jobs} (requested=${JOBS})"
  cmake --build "${BUILD_PATH}" -- -j "${_ptoas_jobs}"
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
  # scikit-build-core reconfigures the PTOAS tree from scratch for the wheel.
  # Pass the devtoolset-7 sysroot + gcc-toolchain through CMake defines so the
  # wheel's native extension links against the CentOS7 libc floor, matching the
  # LLVM/MLIR runtime libraries packaged alongside it.
  if [ -n "${DEVTOOLSET_TOOLCHAIN_FLAGS}" ]; then
    # -ftrapv (from the hardening preload) on aarch64 emits __muloti4 calls
    # for __int128 multiplications; the devtoolset-7 GCC libgcc lacks that
    # symbol, so the wheel's native extension link fails unless compiler-rt
    # is provided. resolve_compiler_rt locates the archive (or builds it).
    resolve_compiler_rt
    local _wheel_rt_flags=""
    if [ -n "${PTOAS_COMPILER_RT:-}" ]; then
      _wheel_rt_flags="-Wl,-u,__muloti4 ${PTOAS_COMPILER_RT}"
    fi
    wheel_feature_args+=(
      "--config-settings=cmake.define.CMAKE_C_FLAGS=${DEVTOOLSET_TOOLCHAIN_FLAGS}"
      "--config-settings=cmake.define.CMAKE_CXX_FLAGS=-DBSPUB_NPU_DATA_TYPE -D_GLIBCXX_USE_CXX11_ABI=0 ${DEVTOOLSET_TOOLCHAIN_FLAGS}"
      "--config-settings=cmake.define.CMAKE_EXE_LINKER_FLAGS=-fuse-ld=lld -lstdc++ ${_wheel_rt_flags}"
      "--config-settings=cmake.define.CMAKE_SHARED_LINKER_FLAGS=-fuse-ld=lld -lstdc++ ${_wheel_rt_flags}"
      "--config-settings=cmake.define.CMAKE_MODULE_LINKER_FLAGS=-fuse-ld=lld -lstdc++ ${_wheel_rt_flags}"
    )
  fi
  CMAKE_BUILD_PARALLEL_LEVEL="$(ptoas_build_jobs)" \
  SKBUILD_BUILD_DIR="${BUILD_PATH}" \
  LLVM_BUILD_DIR="${LLVM_BUILD_DIR}" \
    "${python_bin}" -m pip wheel "${BASE_PATH}" \
      "${wheel_feature_args[@]}" \
      --config-settings=wheel.py-api=cp37 \
      --config-settings=cmake.define.PTOAS_ENABLE_ONLINE_CORE_COMPILE=ON \
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
  local _ptoas_jobs
  _ptoas_jobs="$(ptoas_build_jobs)"
  echo "PTOAS build parallelism: jobs=${_ptoas_jobs} (requested=${JOBS})"
  ENABLE_PACKAGE=FALSE configure_ptoas
  cmake --build "${BUILD_PATH}" -- -j "${_ptoas_jobs}"

  # Distill the version used for the .run package name. The CANN product version
  # (9.2.0 for this release train) differs from project(ptoas VERSION 0.57), so
  # default to the packaging version and allow an explicit override.
  PTOAS_PACKAGE_VERSION="${PTOAS_PACKAGE_VERSION:-9.2.0}"

  # Build and repair the wheel first, then reconfigure with its absolute path
  # so CMake/CPack owns run, RPM, and DEB payload generation uniformly.
  rm -rf "${BUILD_OUT_PATH}"
  mkdir -p "${BUILD_OUT_PATH}"
  stage_ptoas_wheel
  ENABLE_PACKAGE=TRUE
  configure_ptoas
  # configure_ptoas resets the build tree; rebuild all targets before the
  # install/CPack pass so generated install scripts reference real artifacts.
  cmake --build "${BUILD_PATH}" -- -j "${_ptoas_jobs}"
  cmake --install "${BUILD_PATH}"
  cmake --build "${BUILD_PATH}" --target package -- -j "${_ptoas_jobs}"
  echo "package staged under ${BUILD_OUT_PATH}"
  # Diagnostics: the OBS uploader reads build_out via the host path
  # /opt/cloud/slavespace/.../x86build/build_out; print what we actually
  # ~created so a path mismatch is visible in the CI log.
  echo "BUILD_OUT absolute: $(cd "${BUILD_OUT_PATH}" && pwd -P)"
  ls -la "${BUILD_OUT_PATH}"
}

# Actions marks PreSmoke with ST_PART=1. The robot exports task_name and may
# also pass SMOKE_TYPE=pre; accept the job name when that shell variable is lost.
is_presmoke() {
  [ "${SMOKE_TYPE:-}" = "pre" ] || [ "${ST_PART:-}" = "1" ] \
    || [[ "${task_name:-}" == PreSmoke_* ]]
}

# Persist the decision before building: the parent driver invokes runop.sh
# after this child exits, and cannot inherit exports from it.
prepare_presmoke_log_archive() {
  # The parent smoke driver archives this directory after the skipped board
  # commands, so prepare it for either CI entry point after package success.
  local log_dir="${PTOAS_PRESMOKE_LOG_DIR:-/root/ascend/log}"
  mkdir -p "${log_dir}"
  printf '%s\n' 'PTOAS PreSmoke package prepared; runop and board validation are disabled.' \
    > "${log_dir}/ptoas-presmoke.log"
}

write_presmoke_runop_policy() {
  if is_presmoke; then
    mkdir -p "${BUILD_PATH}"
    touch "${PTOAS_PRESMOKE_SKIP_RUNOP_MARKER}"
    echo "PreSmoke runop smoke disabled: ${PTOAS_PRESMOKE_SKIP_RUNOP_MARKER}"
    # Report the legacy success sentinel only after package preparation succeeds.
    if [ "${1:-}" = "complete" ]; then
      prepare_presmoke_log_archive
      echo "execute samples success (skipped in PreSmoke; covered by Build_and_test)"
    fi
  else
    rm -f "${PTOAS_PRESMOKE_SKIP_RUNOP_MARKER}"
  fi
}

# Accept exactly one nonempty installer for this host. Reject ambiguous output
# rather than letting the parent expand its wildcard to multiple installers.
find_presmoke_package() {
  local package_dir="$1"
  local package_arch="$2"
  local packages=()
  local candidate
  for candidate in "${package_dir}"/cann-pto-as*.run; do
    [ -e "${candidate}" ] || continue
    packages+=("${candidate}")
  done
  [ "${#packages[@]}" -eq 1 ] || return 1
  candidate="${packages[0]}"
  [ -f "${candidate}" ] && [ -s "${candidate}" ] || return 1
  case "${candidate##*/}" in
    *linux-"${package_arch}".run) printf '%s\n' "${candidate}" ;;
    *) return 1 ;;
  esac
}

# Only PreSmoke may consume a previous package. Compile and Build_and_test
# always build the requested source, and successful packaging refreshes this
# optional architecture-specific cache with an atomic rename.
cache_presmoke_package() {
  local package_arch="$1"
  local cached_dir="${CANN_3RD_LIB_PATH}/lib_cache/ptoas-presmoke/${package_arch}"
  local package_file
  package_file="$(find_presmoke_package "${BUILD_OUT_PATH}" "${package_arch}")" || return 0
  mkdir -p "${cached_dir}" || return 1
  local staging_file
  staging_file="$(mktemp "${cached_dir}/.package.XXXXXX")" || return 1
  if cp "${package_file}" "${staging_file}" && chmod 644 "${staging_file}" \
      && mv -f "${staging_file}" "${cached_dir}/cann-pto-as_linux-${package_arch}.run"; then
    echo "Cached PreSmoke installer under ${cached_dir}"
  else
    rm -f "${staging_file}"
    return 1
  fi
}

reuse_presmoke_package() {
  local package_arch="$1"
  local cached_dir="${CANN_3RD_LIB_PATH}/lib_cache/ptoas-presmoke/${package_arch}"
  local package_dir package_file
  for package_dir in "${BUILD_OUT_PATH}" "${BASE_PATH}" "${download_path:-${BASE_PATH}}" "${cached_dir}"; do
    package_file="$(find_presmoke_package "${package_dir}" "${package_arch}")" || continue
    if [ "${package_dir}" != "${BUILD_OUT_PATH}" ]; then
      # Never leave two installers for the parent's cann-pto-as*.run glob.
      local existing
      for existing in "${BUILD_OUT_PATH}"/cann-pto-as*.run; do
        [ ! -e "${existing}" ] || return 1
      done
      mkdir -p "${BUILD_OUT_PATH}" || return 1
      cp "${package_file}" "${BUILD_OUT_PATH}/cann-pto-as_linux-${package_arch}.run" || return 1
    fi
    echo "Reusing PreSmoke installer: ${package_file}; skipping LLVM/PTOAS build and packaging"
    return 0
  done
  return 1
}

# Actions exports obs_path; the robot publishes under pto-as/package and
# exports GIT_PR_NUMBER (older smoke wrappers use pr_id). Never guess a PR.
presmoke_artifact_path() {
  if [ -n "${obs_path:-}" ]; then
    if [[ "${obs_path}" =~ ^([A-Za-z0-9_-]+/pto-as/ci|pto-as)/package/[0-9]+$ ]]; then
      printf '%s\n' "${obs_path}"
      return 0
    fi
  elif [ "${ST_PART:-}" != "1" ]; then
    local robot_pr="${GIT_PR_NUMBER:-${pr_id:-}}"
    if [[ "${robot_pr}" =~ ^[0-9]+$ ]]; then
      printf 'pto-as/package/%s\n' "${robot_pr}"
      return 0
    fi
  fi
  echo "ERROR: no valid PreSmoke artifact path; set obs_path or the robot PR number" >&2
  return 1
}

# Both CI systems publish Compile installers to this existing public endpoint,
# but their PreSmoke nodes may have neither the package nor a shared cache.
download_presmoke_package() {
  local package_arch="$1" artifact_path
  artifact_path="$(presmoke_artifact_path)" || return 1
  if [[ ! "${package_arch}" =~ ^(x86_64|aarch64)$ ]]; then
    echo "ERROR: unsupported PreSmoke installer architecture" >&2
    return 1
  fi
  local package_name="cann-pto-as_linux-${package_arch}.run"
  local artifact_url="https://ascend-ci.obs.cn-north-4.myhuaweicloud.com/${artifact_path}/${package_name}"
  local existing staging_file first_line
  for existing in "${BUILD_OUT_PATH}"/cann-pto-as*.run; do
    [ ! -e "${existing}" ] || return 1
  done
  mkdir -p "${BUILD_OUT_PATH}" || return 1
  staging_file="$(mktemp "${BUILD_OUT_PATH}/.package.XXXXXX")" || return 1
  echo "Downloading Compile installer for PreSmoke: ${artifact_url}"
  if curl --fail --silent --show-error --location --proto '=https' --proto-redir '=https' \
      --connect-timeout 15 --max-time 300 --retry 2 --retry-max-time 300 \
      --output "${staging_file}" "${artifact_url}" && [ -s "${staging_file}" ]; then
    IFS= read -r first_line < "${staging_file}" || first_line=""
    case "${first_line}" in
      '#!/bin/sh'|'#!/bin/bash'|'#!/usr/bin/env bash')
        if chmod 644 "${staging_file}" && mv "${staging_file}" "${BUILD_OUT_PATH}/${package_name}"; then
          return 0
        fi
        ;;
    esac
  fi
  rm -f "${staging_file}"
  echo "ERROR: no valid Compile installer available for PreSmoke; source builds are disabled" >&2
  return 1
}

main() {
  JOBS="${JOBS:-}"
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
        [ $# -ge 2 ] || { echo "ERROR: -j requires a job count" >&2; return 1; }
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

  local package_arch
  package_arch="$(uname -m)"
  case "${package_arch}" in
    arm64) package_arch=aarch64 ;;
    amd64) package_arch=x86_64 ;;
  esac
  if [ "$ENABLE_BUILD_ONLY" == "TRUE" ] || [ "$ENABLE_PACKAGE" == "TRUE" ]; then
    write_presmoke_runop_policy
    if is_presmoke; then
      if [ "${PACKAGE_TYPE:-run}" != "run" ]; then
        echo "ERROR: PreSmoke requires a run installer; source builds are disabled" >&2
        return 1
      fi
      if ! reuse_presmoke_package "${package_arch}"; then
        download_presmoke_package "${package_arch}" || return 1
        reuse_presmoke_package "${package_arch}" || return 1
      fi
      write_presmoke_runop_policy complete
      return 0
    fi
    resolve_build_jobs
    trap 'report_build_failure "$?"' EXIT
    resolve_devtoolset_toolchain
    resolve_ptoas_toolchain
  fi

  prepare_llvm_cache_layout

  if [ "$ENABLE_BUILD_ONLY" == "TRUE" ]; then
    build_only
  fi
  if [ "$ENABLE_PACKAGE" == "TRUE" ]; then
    package
    if ! cache_presmoke_package "${package_arch}"; then
      echo "WARNING: could not refresh the optional PreSmoke package cache" >&2
    fi
  fi
  if [ "$ENABLE_BUILD_ONLY" == "TRUE" ] || [ "$ENABLE_PACKAGE" == "TRUE" ]; then
    write_presmoke_runop_policy complete
  fi
  if [ "$ENABLE_BUILD_ONLY" != "TRUE" ] && [ "$ENABLE_PACKAGE" != "TRUE" ]; then
    usage
  fi
}

set -o pipefail
main "$@"
