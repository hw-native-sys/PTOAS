#!/usr/bin/env bash
# On-device AscendC vs VMI FP32 block-quant bench (task-submit wrapper).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPRO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUT="${REPRO_ROOT}/outputs/fp32_block_quant"
NPU_DEVICE="${NPU_DEVICE:-1}"
PTOAS_ROOT="${PTOAS_ROOT:-/home/jzhuang/work_dir/vmi_work_0804/PTOAS-main}"

mkdir -p "${OUT}"

INNER="set -euo pipefail
source /home/jzhuang/miniconda/etc/profile.d/conda.sh
conda activate cann91_dev
set +u
source /home/jzhuang/cann_installed/9.1.0-beta.3/cann/set_env.sh
set -u
export ASCEND_RT_VISIBLE_DEVICES=0
export NPU_DEVICE=0
export RG_N_CORES=\${RG_N_CORES:-72}
export PTOAS_ROOT='${PTOAS_ROOT}'
export PYTHONPATH=\"\${PTOAS_ROOT}/ptodsl:\${PYTHONPATH:-}\"
export PATH=\"\${PTOAS_ROOT}/build/tools/ptoas:\${PTOAS_ROOT}/build/tools/pto-test-opt:\${PATH}\"
export PTO_FLAGS='--pto-backend=vpto --pto-level=level3 --cann-output-version=9.1.0-beta.3'
export PTODSL_CACHE_DIR='${OUT}/ptodsl_cache'
mkdir -p \"\${PTODSL_CACHE_DIR}\"
python '${SCRIPT_DIR}/bench_fp32_block_quant.py' --side both --shapes 8192x2048,512x2048 2>&1 | tee '${OUT}/bench.txt'
"

if command -v task-submit >/dev/null 2>&1; then
  echo "Using task-submit --device ${NPU_DEVICE}"
  task-submit --device "${NPU_DEVICE}" --max-time 2400 --run "bash -lc $(printf %q "${INNER}")"
else
  bash -lc "${INNER}"
fi

echo "Results: ${OUT}/bench.txt"
