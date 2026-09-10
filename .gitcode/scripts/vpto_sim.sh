#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set -euo pipefail

WORKSPACE="${1:-${WORKSPACE:-$(pwd)}}"
WORKSPACE="$(cd "${WORKSPACE}" && pwd)"
BUILD_ROOT="${BUILD_ROOT:-${WORKSPACE}/.work/gitcode-vpto-sim}"
ASCEND_3RD_LIB_PATH="${ASCEND_3RD_LIB_PATH:-/home/opensource}"
CANN_HOME="${CANN_HOME:-${ASCEND_HOME_PATH:-}}"
CASE_PREFIX="${VPTO_SIM_CASE_PREFIX:-}"
JOBS="${VPTO_SIM_JOBS:-$(nproc 2>/dev/null || printf '4')}"
PYPTO_REF="${PYPTO_REF:-ef6ce7cd8bd33b4c93b58dc34830a20b145736ac}"
PTO_ISA_COMMIT="${PTO_ISA_COMMIT:-927253b784344d4ecbed524fd44737097c2bc3e0}"
PYPTO_WORKSPACE="${PYPTO_WORKSPACE:-${BUILD_ROOT}/pypto-ci}"
PTO_ISA_ROOT="${PTO_ISA_ROOT:-${BUILD_ROOT}/pto-isa-ci}"
SUITE_STATUS=0
SIM_SUITE="${SIM_SUITE:-all}"

mkdir -p "${BUILD_ROOT}"
exec > >(tee "${BUILD_ROOT}/vpto-sim.log") 2>&1

skip_simulator() {
  echo "::warning::Skipping ${SIM_SUITE} simulator validation: $1"
  echo "Simulator environment is unavailable; no validation was executed."
  exit 0
}

[[ -d "${ASCEND_3RD_LIB_PATH}" ]] || {
  skip_simulator "ASCEND_3RD_LIB_PATH is unavailable: ${ASCEND_3RD_LIB_PATH}"
}

if [[ -f "${CANN_HOME}/set_env.sh" ]]; then
  # shellcheck disable=SC1091
  source "${CANN_HOME}/set_env.sh"
fi

ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-${CANN_HOME}}"

[[ -n "${ASCEND_HOME_PATH:-}" && -d "${ASCEND_HOME_PATH}" ]] || {
  skip_simulator "ASCEND_HOME_PATH is unavailable: ${ASCEND_HOME_PATH:-<unset>}"
}

BISHENG_BIN="${BISHENG_BIN:-${ASCEND_HOME_PATH}/bin/bisheng}"
MSPROF_BIN="${MSPROF_BIN:-${ASCEND_HOME_PATH}/bin/msprof}"
command -v "${BISHENG_BIN}" >/dev/null 2>&1 || {
  skip_simulator "bisheng is unavailable: ${BISHENG_BIN}"
}
command -v "${MSPROF_BIN}" >/dev/null 2>&1 || {
  skip_simulator "msprof is unavailable: ${MSPROF_BIN}"
}

readarray -t SIM_LIB_DIRS < <(
  find "${ASCEND_HOME_PATH}" -type d -path '*/simulator/dav_3510/lib' 2>/dev/null | sort
)
if [[ "${#SIM_LIB_DIRS[@]}" -eq 0 ]]; then
  skip_simulator "dav_3510 simulator library is unavailable under ${ASCEND_HOME_PATH}"
fi
SIM_LIB_DIR="${SIM_LIB_DIRS[0]}"

echo "CANN environment: ${ASCEND_HOME_PATH}"
echo "Bisheng: ${BISHENG_BIN}"
echo "msprof: ${MSPROF_BIN}"
echo "SIM_LIB_DIR: ${SIM_LIB_DIR}"

BUILD_JOBS="${BUILD_JOBS:-${JOBS}}"
bash "${WORKSPACE}/build.sh" --build \
  --cann_3rd_lib_path "${ASCEND_3RD_LIB_PATH}" \
  -j "${BUILD_JOBS}"

if [[ -f "${WORKSPACE}/build/ptoas-test-env.sh" ]]; then
  # shellcheck disable=SC1091
  source "${WORKSPACE}/build/ptoas-test-env.sh"
fi
PTOAS_BIN="${PTOAS_BIN:-${WORKSPACE}/build/tools/ptoas/ptoas}"
[[ -x "${PTOAS_BIN}" ]] || {
  echo "ERROR: built ptoas is unavailable: ${PTOAS_BIN}" >&2
  exit 1
}

export PTOAS_BIN ASCEND_HOME_PATH SIM_LIB_DIR DEVICE=SIM JOBS
export WORK_SPACE="${BUILD_ROOT}/cases"
export CASES_ROOT="${WORKSPACE}/test/vpto/cases"
export PATH="$(dirname "${PTOAS_BIN}"):${PATH}"

if [[ -n "${CASE_PREFIX}" ]]; then
  export CASE_PREFIX
fi

run_suite() {
  local suite_name="$1"
  shift
  echo "=== ${suite_name} ==="
  set +e
  "$@"
  local status=$?
  set -e
  if [[ "${status}" -ne 0 ]]; then
    echo "${suite_name}: FAILED (${status})"
    SUITE_STATUS=1
  else
    echo "${suite_name}: PASSED"
  fi
}

run_vpto() {
  bash "${WORKSPACE}/test/vpto/scripts/run_host_vpto_validation_parallel.sh"
}

run_tilelib_st() {
  mkdir -p "${BUILD_ROOT}/tilelib-st-a5"
  LLVM_BUILD_DIR="${LLVM_BUILD_DIR:-${WORKSPACE}/build}" \
  ASCEND_HOME_PATH="${ASCEND_HOME_PATH}" \
  PYTHON_BIN="${PYTHON_BIN:-python3}" \
  PTOAS_BIN="${PTOAS_BIN}" \
    bash "${WORKSPACE}/test/tilelang_st/script/run_ci.sh" -r sim -v a5 \
      --build-jobs "${JOBS}" --jobs "${JOBS}" --smoke \
      2>&1 | tee "${BUILD_ROOT}/tilelib-st.log"
}

run_ptodsl_st() {
  ASCEND_HOME_PATH="${ASCEND_HOME_PATH}" PTOAS_BIN="${PTOAS_BIN}" \
    bash "${WORKSPACE}/scripts/sim_dsl.sh" "${WORKSPACE}/test/dsl-st" \
      2>&1 | tee "${BUILD_ROOT}/ptodsl-dsl-st.log"
}

run_pypto_tests() {
  command -v git >/dev/null 2>&1 || return 1
  if [[ ! -d "${PYPTO_WORKSPACE}/.git" ]]; then
    git clone --depth 1 https://github.com/hw-native-sys/pypto.git "${PYPTO_WORKSPACE}" || return 1
    git -C "${PYPTO_WORKSPACE}" fetch --depth 1 origin "${PYPTO_REF}" || return 1
    git -C "${PYPTO_WORKSPACE}" checkout --force "${PYPTO_REF}" || return 1
  fi
  if [[ ! -d "${PTO_ISA_ROOT}/.git" ]]; then
    git clone --depth 1 https://github.com/hw-native-sys/pto-isa.git "${PTO_ISA_ROOT}" || return 1
    git -C "${PTO_ISA_ROOT}" fetch --depth 1 origin "${PTO_ISA_COMMIT}" || return 1
    git -C "${PTO_ISA_ROOT}" checkout --force "${PTO_ISA_COMMIT}" || return 1
  fi

  local python_bin="${PYTHON_BIN:-python3}"
  "${python_bin}" -m pip install --no-build-isolation --no-deps "${PYPTO_WORKSPACE}" || return 1
  env -u ASCEND_HOME_PATH "${python_bin}" -m pip install --no-build-isolation --no-deps \
    "${PYPTO_WORKSPACE}/runtime" || return 1
  local -a core_tests=(
    tests/st/examples/00_hello_world/test_hello_world.py
    tests/st/examples/02_intermediate/test_softmax.py::TestTileSoftmax::test_tile_softmax
    tests/st/examples/02_intermediate/test_rms_norm.py::TestRMSNormCore::test_rms_norm_core
    tests/st/runtime/ops/test_elementwise.py
    tests/st/runtime/ops/test_assemble.py
    tests/st/runtime/framework_and_models/test_jit.py::TestJITExecution::test_cache_hit_reuses_compiled_program
    tests/st/runtime/framework_and_models/test_jit.py::TestJITDynamicBatch::test_one_artifact_serves_multiple_batches
    tests/st/runtime/framework_and_models/test_compiled_program.py::TestManualWorkerExtraction::test_block_dim_override_runs
  )
  local -a fa_tests=(
    tests/st/runtime/ops/test_cast.py::TestCast::test_tile_cast_col_major_narrow
    tests/st/runtime/framework_and_models/test_paged_attention.py::TestPagedAttentionKernels::test_qk_matmul_ptoas[16-128-128]
    tests/st/runtime/framework_and_models/test_paged_attention.py::TestPagedAttentionKernels::test_softmax_prepare_ptoas[16-128]
    tests/st/runtime/framework_and_models/test_paged_attention.py::TestPagedAttentionKernels::test_softmax_prepare_unaligned_ptoas[16-128-100]
    tests/st/runtime/framework_and_models/test_paged_attention.py::TestPagedAttentionKernels::test_pv_matmul_ptoas[16-128-128]
    tests/st/runtime/framework_and_models/test_paged_attention.py::TestPagedAttentionKernels::test_online_update_ptoas[16-128-0-1]
  )
  local -a int8_tests=(tests/st/codegen/dsl/test_batch_matmul_pipeline.py::test_no_mat_to_mat_tmov)
  run_pypto_pytest() {
    local platform="$1"
    shift
    (cd "${PYPTO_WORKSPACE}" && "${python_bin}" -m pytest "$@" -v --platform="${platform}") \
      2>&1 | tee "${BUILD_ROOT}/pypto-${platform}.log"
  }
  local platform
  for platform in a5sim a2a3sim; do
    run_suite "PyPTO core ${platform}" run_pypto_pytest "${platform}" "${core_tests[@]}"
    run_suite "PyPTO FA ${platform}" run_pypto_pytest "${platform}" "${fa_tests[@]}"
  done
  run_suite "PyPTO INT8 a2a3sim" run_pypto_pytest a2a3sim "${int8_tests[@]}"
}

case "${SIM_SUITE}" in
  vpto)
    run_suite "VPTO SIM" run_vpto
    ;;
  tilelib)
    run_suite "TileLib ST" run_tilelib_st
    ;;
  ptodsl)
    run_suite "PTODSL/DSL ST" run_ptodsl_st
    ;;
  pypto)
    run_suite "PyPTO simulator suites" run_pypto_tests
    ;;
  all)
    run_suite "VPTO SIM" run_vpto
    run_suite "TileLib ST" run_tilelib_st
    run_suite "PTODSL/DSL ST" run_ptodsl_st
    run_suite "PyPTO simulator suites" run_pypto_tests
    ;;
  *)
    echo "ERROR: unsupported SIM_SUITE=${SIM_SUITE}; expected vpto, tilelib, ptodsl, pypto, or all" >&2
    exit 2
    ;;
esac

echo "=== SIM suite summary ==="
if [[ "${SUITE_STATUS}" -ne 0 ]]; then
  echo "One or more migrated simulator suites failed; this experimental job remains non-blocking."
else
  echo "All migrated simulator suites passed."
fi
exit "${SUITE_STATUS}"
