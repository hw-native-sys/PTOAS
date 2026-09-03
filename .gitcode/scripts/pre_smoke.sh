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

sudo pip3 install ninja
(
source /opt/rh/devtoolset-7/enable
mkdir -p /tmp/py310shim
ln -sf /usr/bin/python3.10 /tmp/py310shim/python3
ln -sf /usr/bin/python3.10 /tmp/py310shim/python
export PATH=/tmp/py310shim:$PATH
which python3
python3 --version
/usr/bin/python3.10 -m pip install pybind11==2.13.6
#      sudo apt-get update && sudo apt-get install lld -y
echo "source devtoolset"
bash build.sh --pkg --cann_3rd_lib_path=/home/opensource 2>&1 | tee -a ./run_test.log
export LLVM_BUILD_DIR=${WORKSPACE}/build/llvm-project/build-shared
export PTO_INSTALL_DIR=${WORKSPACE}/install
export MLIR_PYTHON_ROOT=$LLVM_BUILD_DIR/tools/mlir/python_packages/mlir_core
export PTO_PYTHON_ROOT=$PTO_INSTALL_DIR/
export PYTHONPATH=$MLIR_PYTHON_ROOT:$PTO_PYTHON_ROOT:$PYTHONPATH
export LD_LIBRARY_PATH=$LLVM_BUILD_DIR/lib:$PTO_INSTALL_DIR/lib:$LD_LIBRARY_PATH
echo "bash ./build_out/cann-pto-as*.run --full --quiet --install-path=/usr/local/Ascend"
bash ./build_out/cann-pto-as*.run --full --quiet --install-path=/usr/local/Ascend
export PATH=/usr/local/Ascend/cann/tools/ptoas/bin:$PATH
echo "bash test/samples/runop.sh --enablebc all"
bash test/samples/runop.sh --enablebc all 2>&1 | tee -a ./run_test.log
echo "bash test/npu_validation/scripts/run_remote_npu_validation.sh"
STAGE="${STAGE:-run}" RUN_MODE='npu' SOC_VERSION='Ascend910' SKIP_CASES='mix_kernel,vadd_validshape,vadd_validshape_dynamic,print' bash test/npu_validation/scripts/run_remote_npu_validation.sh 2>&1 | tee -a ./run_test.log
    )

# ==============================
# 打包log
# ==============================
mkdir -p /root/ascend
slog_name="slog.tar.gz"
tar -zcf "${slog_name}" -C /root/ascend log

# upload plog
if python3 /home/upload.py --bucket-name "ascend-ci" --action upload  --local-file "slog.tar.gz" --obs-object-key "${obs_path}/${slog_name}"; then
  echo "::set-output var=plog_url:https://ascend-ci.obs.cn-north-4.myhuaweicloud.com/${obs_path}/slog.tar.gz"
fi

# ==============================
# 检查 NPU 状态
# ==============================
log "checking NPU status ..."
mkdir -p ./npu_log
npu-smi info  2>&1 | tee ./npu_log/npu_info.log

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

if [ "$execution_success_found" = true ]; then
  echo "$date_time : run test case success"
else
  echo "$date_time : run test case failed ('execute samples success' not found)"
  exit 1
fi
