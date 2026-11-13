#!/usr/bin/env bash
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set -e

# Every path below is derived from WORKSPACE; fail loudly instead of expanding to
# an empty prefix (the script also removes PTOAS_OUT_DIR under it).
: "${WORKSPACE:?WORKSPACE must be set to the checked-out pto-as tree}"

echo "start run test case, please wait ..."

log() {
  local dt
  dt=$(date '+%Y%m%d.%H%M%S')
  echo "===================================================================="
  echo "$dt : $*"
  echo "===================================================================="
}

log "init test case, please wait ..."
rm -rf /root/ascend/log

# ==============================
# 确定要测试的 ops 列表
# ==============================
declare -a ops
ops=("is_finite")
echo $ops

# ==============================
# 运行测试主循环
# ==============================
log "start run test case, please wait ..."

export ASCEND_GLOBAL_LOG_LEVEL=2
export ASCEND_SLOG_PRINT_TO_STDOUT=0

# devtoolset-7 only exists on the CentOS based A2 image. The A5 image is Ubuntu 24.04,
# which ships gcc 13 and has no /opt/rh tree; sourcing it unconditionally aborts the
# whole smoke under `set -e`. build.sh already decides on its own whether the
# devtoolset sysroot is usable, so skipping the activation there is safe.
TOOLCHAIN_ENV=/opt/rh/devtoolset-7/enable
if [ -f "${TOOLCHAIN_ENV}" ]; then
  echo "using devtoolset-7 from ${TOOLCHAIN_ENV}"
else
  echo "devtoolset-7 not present (non-CentOS image), using the image default toolchain"
fi

# /usr/bin/python3.10 is what the PTODSL Python bindings are built against on the A2
# image; the A5 image also ships it. Fall back to whatever python3.10/python3 exists.
PY_BIN=/usr/bin/python3.10
if [ ! -x "${PY_BIN}" ]; then
  PY_BIN=$(command -v python3.10 || command -v python3)
fi
echo "python bindings interpreter: ${PY_BIN}"

# The third-party/LLVM cache lives under /home/opensource on both CI images, but let
# the runner override it when it advertises a different root.
CANN_3RD_LIB_PATH="${ASCEND_3RD_LIB_PATH:-/home/opensource}"
echo "third-party/LLVM cache: ${CANN_3RD_LIB_PATH}"

# build.sh configures LLVM with -DLLVM_USE_LINKER=lld unconditionally; the A2 image
# ships lld, the A5 (Ubuntu 24.04) image does not, and LLVM's configure step then
# fails with "Host compiler does not support '-fuse-ld=lld'". Only the A5 image has
# apt-get: keep this a no-op elsewhere so a missing package manager cannot abort the
# A2 smoke under `set -e`.
if ! command -v ld.lld >/dev/null 2>&1; then
  if command -v apt-get >/dev/null 2>&1; then
    echo "lld not found, installing from the image package source ..."
    sudo apt-get update -y
    sudo apt-get install -y lld
  else
    echo "lld not found and this image has no apt-get; leaving the toolchain as is"
  fi
fi
command -v ld.lld >/dev/null 2>&1 && ld.lld --version | head -1

sudo pip3 install ninja
(
if [ -f "${TOOLCHAIN_ENV}" ]; then
  source "${TOOLCHAIN_ENV}"
fi
mkdir -p /tmp/py310shim
ln -sf "${PY_BIN}" /tmp/py310shim/python3
ln -sf "${PY_BIN}" /tmp/py310shim/python
export PATH=/tmp/py310shim:$PATH
which python3
python3 --version
"${PY_BIN}" -m pip install pybind11==2.13.6
#      sudo apt-get update && sudo apt-get install lld -y
echo "source devtoolset"
# build.sh treats ST_PART=1/SMOKE_TYPE=pre as "PreSmoke" and then reuses the installer
# produced by the Compile stage instead of building. This entry point has no Compile
# stage behind it (the A5 job runs on its own runner), so keep ST_PART out of build.sh's
# environment and only ask the board validation below to skip the full sweep.
bash build.sh --pkg --cann_3rd_lib_path="${CANN_3RD_LIB_PATH}" 2>&1 | tee -a ./run_test.log
export LLVM_BUILD_DIR=${WORKSPACE}/build/llvm-project/build-shared
export PTO_INSTALL_DIR=${WORKSPACE}/install
export MLIR_PYTHON_ROOT=$LLVM_BUILD_DIR/tools/mlir/python_packages/mlir_core
export PTO_PYTHON_ROOT=$PTO_INSTALL_DIR/
export PYTHONPATH=$MLIR_PYTHON_ROOT:$PTO_PYTHON_ROOT:$PYTHONPATH
export LD_LIBRARY_PATH=$LLVM_BUILD_DIR/lib:$PTO_INSTALL_DIR/lib:$LD_LIBRARY_PATH
echo "bash ./build_out/cann-pto-as*.run --full --quiet --install-path=/usr/local/Ascend"
bash ./build_out/cann-pto-as*.run --full --quiet --install-path=/usr/local/Ascend
# The run package installs the ptoas wheel into the CANN tree instead of a
# system site-packages, so `import ptoas` only resolves after the toolkit's
# set_env.sh has put <version>/python/site-packages on PYTHONPATH. Without it
# every PTODSL sample dies with "ModuleNotFoundError: No module named 'ptoas'".
export ASCEND_PTO_AS_PATH=/usr/local/Ascend/cann/pto_as
for _cann_env in /usr/local/Ascend/ascend-toolkit/set_env.sh \
                 /usr/local/Ascend/cann/set_env.sh \
                 /usr/local/Ascend/cann-*/set_env.sh; do
  if [ -f "${_cann_env}" ]; then
    echo "sourcing CANN environment: ${_cann_env}"
    . "${_cann_env}"
    break
  fi
done
export PATH=/usr/local/Ascend/cann/tools/ptoas/bin:$PATH
echo "ptoas import: $("${PY_BIN}" -c 'import ptoas, sys; print(ptoas.__file__)' 2>&1 | tail -1)"
# runop.sh --enablebc drives every sample through ptobc, but resolve_ptobc_bin only
# looks at build/tools/ptobc, build/bin and PATH. `build.sh --pkg` builds through the
# wheel path, which leaves ptobc under build/wheel/tools/ptobc, so without this every
# case fails with "Missing executable: PTOBC_BIN".
for _cand_dir in "${WORKSPACE}/build/wheel" "${WORKSPACE}/build"; do
  if [ -x "${_cand_dir}/tools/ptobc/ptobc" ] || [ -x "${_cand_dir}/bin/ptobc" ]; then
    export PTO_BUILD_DIR="${_cand_dir}"
    break
  fi
done
if [ -z "${PTOBC_BIN:-}" ]; then
  _ptobc_found="$(find "${WORKSPACE}/build" -maxdepth 4 -type f -name ptobc -perm -u+x 2>/dev/null | head -1)"
  if [ -n "${_ptobc_found}" ]; then
    export PTOBC_BIN="${_ptobc_found}"
  fi
fi
echo "ptobc: ${PTOBC_BIN:-not found}; PTO_BUILD_DIR=${PTO_BUILD_DIR:-unset};" \
  "ptoas: $(command -v ptoas || echo not-found)"
# Pin runop's output directory: it writes the generated *-pto.cpp payloads and
# expected_npu_validation_cases.txt there, and the board validation enumerates
# cases from that same directory. Left to mktemp the path is random, so the
# validation falls back to the source tree, finds no *-pto.cpp, and aborts with
# "expected-case manifest is missing".
export PTOAS_OUT_DIR="${WORKSPACE}/build/ptoas-samples"
rm -rf "${PTOAS_OUT_DIR}"
echo "bash test/samples/runop.sh --enablebc all"
bash test/samples/runop.sh --enablebc all 2>&1 | tee -a ./run_test.log
echo "bash test/npu_validation/scripts/run_remote_npu_validation.sh"
ST_PART="${PTOAS_BOARD_ST_PART:-1}" STAGE="${STAGE:-run}" RUN_MODE='npu' SOC_VERSION="${SOC_VERSION:-Ascend910}" \
  PTOAS_SAMPLES_ROOT="${PTOAS_OUT_DIR}" \
  EXPECTED_CASES_FILE="${PTOAS_OUT_DIR}/expected_npu_validation_cases.txt" \
  RUN_ONLY_CASES="${PTOAS_BOARD_RUN_ONLY:-}" \
  SKIP_CASES='mix_kernel,vadd_validshape,vadd_validshape_dynamic,print' \
  bash test/npu_validation/scripts/run_remote_npu_validation.sh 2>&1 | tee -a ./run_test.log
    )

# ==============================
# 打包log
# ==============================
mkdir -p /root/ascend/log
slog_name="slog.tar.gz"
tar -zcf "${slog_name}" -C /root/ascend log

# upload plog
# if python3 /home/upload.py --bucket-name "ascend-ci" --action upload  --local-file "slog.tar.gz" --obs-object-key "${obs_path}/${slog_name}"; then
#   echo "::set-output var=plog_url:https://ascend-ci.obs.cn-north-4.myhuaweicloud.com/${obs_path}/slog.tar.gz"
# fi

# ==============================
# 检查 NPU 状态
# ==============================
log "checking NPU status ..."
mkdir -p ./npu_log
npu-smi info 2>&1 | tee ./npu_log/npu_info.log

# ==============================
# 检查测试结果
# ==============================
log "checking test results ..."

date_time=`date +%Y%m%d`"."`date +%H%M%S`
execution_success_found=false
fail_not_zero_found=false
while IFS= read -r line; do
  if [[ "$line" == *"execute samples success"* ]]; then
      execution_success_found=true
  fi
  if [[ "$line" =~ OK=[0-9]+\ +FAIL=([0-9]+)\ +SKIP=[0-9]+ ]]; then
      current_fail="${BASH_REMATCH[1]}"

      if [ "$current_fail" -ne 0 ]; then
          echo "$date_time : run test case failed (found FAIL=$current_fail in line: $line)"
          exit 1 # 发现非零 FAIL，立即退出
      fi
  fi
done < "./run_test.log"

# The A5 job opts into the board explicitly (PTOAS_BOARD_ST_PART=0). On that path a
# pre-smoke skip must fail the job: run_remote_npu_validation.sh's three skip
# conditions (SMOKE_TYPE=pre / ST_PART=1 / task_name=PreSmoke_*) print the legacy
# success sentinel and exit 0, so a skipped board would otherwise look green.
if [ "${PTOAS_BOARD_ST_PART:-1}" = "0" ]; then
  if ! grep -q "DEVICE_ID=" ./run_test.log; then
    echo "$date_time : board validation never started (pre-smoke skip path hit); refusing to report success"
    exit 1
  fi
  if ! grep -q "MATCHED=" ./run_test.log; then
    echo "$date_time : board validation produced no case summary; refusing to report success"
    exit 1
  fi
fi

if [ "$execution_success_found" = true ]; then
  echo "$date_time : run test case success"
else
  echo "$date_time : run test case failed ('execute samples success' not found)"
  exit 1
fi
