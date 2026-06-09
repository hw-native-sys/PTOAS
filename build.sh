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

set -e

dotted_line="----------------------------------------------------------------"
COLOR_RESET="\033[0m"
COLOR_GREEN="\033[32m"
COLOR_RED="\033[31m"

export BASE_PATH=$(
  cd "$(dirname $0)"
  pwd
)

export INCLUDE_PATH="${ASCEND_HOME_PATH}/include"
export ASCEND_ENV_PATH="${ASCEND_HOME_PATH}/bin"
export BUILD_PATH="${BASE_PATH}/build"
export BUILD_OUT_PATH="${BASE_PATH}/build_out"
CANN_3RD_LIB_PATH="${BASE_PATH}/third_party"
CMAKE_ARGS=""
HARDENING_CACHE_FILE="${BASE_PATH}/cmake/LinuxHardeningCache.cmake"
RUNTIME_DEPS_COLLECTOR="${BASE_PATH}/scripts/package/collect_ptoas_runtime_deps.sh"
FORTIFY_MARKER_SOURCE="${BASE_PATH}/scripts/package/fortify_marker.c"
LLVM_GIT_URL="https://gitcode.com/GitHub_Trending/ll/llvm-project.git"
LLVM_GIT_REF="llvmorg-19.1.7"
LLVM_CLONE_RETRY_COUNT=3
LLVM_CLONE_RETRY_INTERVAL=5
DEVTOOLSET_TOOLCHAIN_FLAGS="--sysroot=/opt/rh/devtoolset-7/root --gcc-toolchain=/opt/rh/devtoolset-7/root/usr"

#print usage message
usage() {
  echo "Usage:"
  echo ""
  echo "    -h, --help  Print usage"
  echo "    --pkg Build run package"
  echo ""
}

print_success() {
  echo
  echo $dotted_line
  local msg="$1"
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

ensure_hardening_cache() {
  if [ ! -f "${HARDENING_CACHE_FILE}" ]; then
    print_error "missing hardening cache: ${HARDENING_CACHE_FILE}"
    exit 1
  fi
}

prepare_fortify_marker_object() {
  local output_dir="$1"
  local marker_object="${output_dir}/fortify_marker.o"

  mkdir -p "${output_dir}"

  clang -O2 -D_FORTIFY_SOURCE=2 -fPIC -c "${FORTIFY_MARKER_SOURCE}" -o "${marker_object}"

  export PTOAS_FORTIFY_MARKER_OBJECT="${marker_object}"
}

compose_runtime_compiler_flags() {
  local existing_flags="$1"
  local merged_flags="${existing_flags:+${existing_flags} }${DEVTOOLSET_TOOLCHAIN_FLAGS}"

  # Do not add -D_FORTIFY_SOURCE=2 here: with the CentOS7 devtoolset
  # sysroot, LLVM's Expected-returning file read helpers can be miscompiled
  # into a busy loop. The package still links a dedicated fortify marker.
  for hardening_flag in -fstack-protector-all; do
    if [[ " ${merged_flags} " != *" ${hardening_flag} "* ]]; then
      merged_flags="${merged_flags} ${hardening_flag}"
    fi
  done

  echo "${merged_flags}"
}

harden_package_artifacts() {
  local build_root="${PTO_SOURCE_DIR}/build"
  local ptoas_bin="${build_root}/tools/ptoas/ptoas"
  local runtime_stage_root="${build_root}/package_runtime/tools/ptoas"
  local staged_bin="${runtime_stage_root}/bin/ptoas"
  local staged_lib_dir="${runtime_stage_root}/lib"

  if [ ! -x "${RUNTIME_DEPS_COLLECTOR}" ]; then
    chmod +x "${RUNTIME_DEPS_COLLECTOR}"
  fi

  if [ ! -f "${ptoas_bin}" ]; then
    print_error "missing ptoas binary for package staging: ${ptoas_bin}"
    exit 1
  fi

  rm -rf "${build_root}/package_runtime"
  mkdir -p "${runtime_stage_root}/bin" "${staged_lib_dir}"

  bash "${RUNTIME_DEPS_COLLECTOR}"     "${build_root}"     "${ptoas_bin}"     "${staged_bin}"     "${staged_lib_dir}"
}

clone_llvm_source() {
  local target_dir="$1"
  local attempt=1

  rm -rf "${target_dir}"

  if [ -d "${CANN_3RD_LIB_PATH}/llvm-19" ]; then
    cp -r "${CANN_3RD_LIB_PATH}/llvm-19" "${target_dir}"
    return 0
  fi

  while [ "${attempt}" -le "${LLVM_CLONE_RETRY_COUNT}" ]; do
    if git -c http.version=HTTP/1.1 clone       --depth 1       --single-branch       --branch "${LLVM_GIT_REF}"       "${LLVM_GIT_URL}"       "${target_dir}"; then
      return 0
    fi

    rm -rf "${target_dir}"

    if [ "${attempt}" -lt "${LLVM_CLONE_RETRY_COUNT}" ]; then
      sleep "${LLVM_CLONE_RETRY_INTERVAL}"
    fi

    attempt=$((attempt + 1))
  done

  print_error "failed to prepare llvm-project source"
  exit 1
}

configure_llvm_host_tools_build() {
  local cmake_args=("$@")
  cmake_args+=("-DLLVM_ENABLE_ZSTD=OFF")

  cmake -G Ninja -S llvm -B "${LLVM_NATIVE_BUILD_DIR}"     -DLLVM_ENABLE_PROJECTS="mlir"     -DBUILD_SHARED_LIBS=OFF     -DCMAKE_C_COMPILER=clang     -DCMAKE_CXX_COMPILER=clang++     -DLLVM_USE_LINKER=lld     -DMLIR_ENABLE_BINDINGS_PYTHON=OFF     -DPython3_EXECUTABLE="$(which python3)"     -DCMAKE_BUILD_TYPE=Release     -DLLVM_TARGETS_TO_BUILD="host"     -DLLVM_INCLUDE_TESTS=OFF     -DLLVM_INCLUDE_BENCHMARKS=OFF     -DLLVM_INCLUDE_EXAMPLES=OFF     "${cmake_args[@]}"
}

build_llvm_host_tools() {
  configure_llvm_host_tools_build
  ninja -C "${LLVM_NATIVE_BUILD_DIR}" llvm-min-tblgen llvm-tblgen mlir-tblgen
}

configure_llvm_runtime_build() {
  local cmake_args=("$@")
  local cmake_c_flags
  local cmake_cxx_flags
  cmake_c_flags="$(compose_runtime_compiler_flags "${CFLAGS:-}")"
  cmake_cxx_flags="$(compose_runtime_compiler_flags "${CXXFLAGS:-}")"
  cmake_args+=(
    "-DLLVM_ENABLE_ZSTD=OFF"
    "-DHAVE_LIBRT=ON"
    "-DLLVM_NATIVE_TOOL_DIR=${LLVM_NATIVE_BUILD_DIR}/bin"
    "-DLLVM_TABLEGEN=${LLVM_NATIVE_BUILD_DIR}/bin/llvm-tblgen"
    "-DMLIR_TABLEGEN_EXE=${LLVM_NATIVE_BUILD_DIR}/bin/mlir-tblgen"
  )
  if [ -n "${PTOAS_FORTIFY_MARKER_OBJECT:-}" ]; then
    cmake_args+=("-DPTOAS_FORTIFY_MARKER_OBJECT=${PTOAS_FORTIFY_MARKER_OBJECT}")
  fi

  cmake -C "${HARDENING_CACHE_FILE}" -G Ninja -S llvm -B "${LLVM_BUILD_DIR}"     -DLLVM_ENABLE_PROJECTS="mlir"     -DBUILD_SHARED_LIBS=ON     -DCMAKE_C_COMPILER=clang     -DCMAKE_CXX_COMPILER=clang++     -DCMAKE_C_FLAGS="${cmake_c_flags}"     -DCMAKE_CXX_FLAGS="${cmake_cxx_flags}"     -DLLVM_USE_LINKER=lld     -DMLIR_ENABLE_BINDINGS_PYTHON=ON     -DPython3_EXECUTABLE="$(which python3)"     -DCMAKE_BUILD_TYPE=Release     -DLLVM_TARGETS_TO_BUILD="host"     -DLLVM_INCLUDE_TESTS=OFF     -DLLVM_INCLUDE_BENCHMARKS=OFF     -DLLVM_INCLUDE_EXAMPLES=OFF     "${cmake_args[@]}"
}

configure_ptoas_build() {
  local cmake_args=("$@")
  local cmake_c_flags
  local cmake_cxx_flags
  cmake_c_flags="$(compose_runtime_compiler_flags "${CFLAGS:-}")"
  cmake_cxx_flags="$(compose_runtime_compiler_flags "${CXXFLAGS:-}")"
  if [ -n "${PTOAS_FORTIFY_MARKER_OBJECT:-}" ]; then
    cmake_args+=("-DPTOAS_FORTIFY_MARKER_OBJECT=${PTOAS_FORTIFY_MARKER_OBJECT}")
  fi

  cmake -C "${HARDENING_CACHE_FILE}" -G Ninja     -S .     -B build     -DLLVM_DIR="${LLVM_BUILD_DIR}/lib/cmake/llvm"     -DMLIR_DIR="${LLVM_BUILD_DIR}/lib/cmake/mlir"     -DPython3_EXECUTABLE="$(which python3)"     -DPython3_FIND_STRATEGY=LOCATION     -Dpybind11_DIR="${PYBIND11_CMAKE_DIR}"     -DMLIR_ENABLE_BINDINGS_PYTHON=ON     -DCMAKE_BUILD_TYPE=Release     -DCMAKE_C_COMPILER=clang     -DCMAKE_CXX_COMPILER=clang++     -DCMAKE_C_FLAGS="${cmake_c_flags}"     -DCMAKE_CXX_FLAGS="${cmake_cxx_flags}"     -DLLVM_USE_LINKER=lld     -DMLIR_PYTHON_PACKAGE_DIR="${LLVM_BUILD_DIR}/tools/mlir/python_packages/mlir_core"     -DCMAKE_INSTALL_PREFIX="${PTO_INSTALL_DIR}"     "${cmake_args[@]}"
}

checkopts() {
  ENABLE_BUILD_ALL=FALSE
  ENABLE_BUILD_ONLY=FALSE
  ENABLE_PACKAGE=FALSE

  parsed_args=$(getopt -a -o j:hvuO: -l help,pkg,build,cann_3rd_lib_path: -- "$@") || {
  usage
  exit 1
  }

  eval set -- "$parsed_args"

  while true; do
    case "$1" in
      -h | --help)
        usage
        exit 0
        ;;
      --build)
        shift
        ENABLE_BUILD_ONLY=TRUE
        ;;
      --cann_3rd_lib_path)
        shift
        CANN_3RD_LIB_PATH="$1"
        shift
        ;;
      --pkg)
        ENABLE_PACKAGE=TRUE
        shift
        ;;
      --)
        shift
        break
        ;;
      *)
        usage
        exit 1
        ;;
    esac
  done
  if [[ "$ENABLE_PACKAGE" == "TRUE" ]]; then
    CMAKE_ARGS="$CMAKE_ARGS -DENABLE_PACKAGE=TRUE"
  fi
  CMAKE_ARGS="$CMAKE_ARGS -DCANN_3RD_LIB_PATH=${CANN_3RD_LIB_PATH}"
}

build_only() {
  echo $dotted_line
  echo "build only"
  ensure_hardening_cache
  export LLVM_SOURCE_DIR=$WORKSPACE/llvm-project
  clone_llvm_source "${LLVM_SOURCE_DIR}"
  export LLVM_NATIVE_BUILD_DIR=$LLVM_SOURCE_DIR/build-native-tools
  export LLVM_BUILD_DIR=$LLVM_SOURCE_DIR/build-shared
  export PTO_SOURCE_DIR=$WORKSPACE
  export PTO_INSTALL_DIR=$PTO_SOURCE_DIR/install
  prepare_fortify_marker_object "${BASE_PATH}/build/fortify_marker"

  cd $LLVM_SOURCE_DIR
  rm -rf "${LLVM_NATIVE_BUILD_DIR}" "${LLVM_BUILD_DIR}"

  build_llvm_host_tools
  configure_llvm_runtime_build
  ninja -C $LLVM_BUILD_DIR

  cd $PTO_SOURCE_DIR
  export PYBIND11_CMAKE_DIR=$(python3 -m pybind11 --cmakedir)

  if [ -d "$CANN_3RD_LIB_PATH/llvm-19" ]; then
    configure_ptoas_build
  else
    configure_ptoas_build
  fi

  ninja -C build
  ninja -C build install

  export MLIR_PYTHON_ROOT=$LLVM_BUILD_DIR/tools/mlir/python_packages/mlir_core
  export PTO_PYTHON_ROOT=$PTO_INSTALL_DIR/
  export PYTHONPATH=$MLIR_PYTHON_ROOT:$PTO_PYTHON_ROOT:$PYTHONPATH
  export LD_LIBRARY_PATH=$LLVM_BUILD_DIR/lib:$PTO_INSTALL_DIR/lib:$LD_LIBRARY_PATH
  export PATH=$PTO_SOURCE_DIR/build/tools/ptoas:$PATH

  bash test/samples/runop.sh --enablebc all
  STAGE="${STAGE:-run}" RUN_MODE='npu' SOC_VERSION='Ascend910' SKIP_CASES='mix_kernel,vadd_validshape,vadd_validshape_dynamic,print' bash test/npu_validation/scripts/run_remote_npu_validation.sh

  echo "execute samples success"
}

clean_build() {
  if [ -d "${BUILD_PATH}" ]; then
    rm -rf ${BUILD_PATH}
  fi
}

clean_build_out() {
  if [ -d "${BUILD_OUT_PATH}" ]; then
    rm -rf ${BUILD_OUT_PATH}
  fi
}

package() {
  echo $dotted_line
  echo "package start"
  ensure_hardening_cache
  clean_build_out
  clean_build
  mkdir $BUILD_PATH
  mkdir $BUILD_OUT_PATH
  cd $BUILD_PATH
  export LLVM_SOURCE_DIR=$BUILD_PATH/llvm-project
  clone_llvm_source "${LLVM_SOURCE_DIR}"
  export LLVM_NATIVE_BUILD_DIR=$LLVM_SOURCE_DIR/build-native-tools
  export LLVM_BUILD_DIR=$LLVM_SOURCE_DIR/build-shared
  export PTO_SOURCE_DIR=$BASE_PATH
  export PTO_INSTALL_DIR=$PTO_SOURCE_DIR/install
  prepare_fortify_marker_object "${BUILD_PATH}/fortify_marker"

  cd $LLVM_SOURCE_DIR
  rm -rf "${LLVM_NATIVE_BUILD_DIR}" "${LLVM_BUILD_DIR}"

  build_llvm_host_tools
  configure_llvm_runtime_build
  ninja -C $LLVM_BUILD_DIR

  cd $PTO_SOURCE_DIR
  export PYBIND11_CMAKE_DIR=$(python3 -m pybind11 --cmakedir)
  mkdir -p "${BUILD_PATH}/package_runtime/tools/ptoas/bin" "${BUILD_PATH}/package_runtime/tools/ptoas/lib"

  if [ -d "$CANN_3RD_LIB_PATH/llvm-19" ]; then
    configure_ptoas_build ${CMAKE_ARGS}
  else
    configure_ptoas_build ${CMAKE_ARGS}
  fi

  ninja -C build
  harden_package_artifacts
  ninja -C build install
  cd $BUILD_PATH
  ninja package
}

main() {
  checkopts "$@"
  if [ "$ENABLE_BUILD_ONLY" == "TRUE" ]; then
    build_only
  fi
  if [ "$ENABLE_PACKAGE" == "TRUE" ]; then
    package
  fi
}

set -o pipefail
main "$@" | gawk '{print strftime("[%Y-%m-%d %H:%M:%S]"), $0}'
