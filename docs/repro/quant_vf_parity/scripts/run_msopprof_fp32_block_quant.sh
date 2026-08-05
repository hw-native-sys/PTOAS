#!/usr/bin/env bash
# msopprof PipeUtilization on AscendC vs VMI FP32 block-quant (8192×2048).
# Uses --kernel-name so torch.randn is not collected as the operator.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPRO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUT="${REPRO_ROOT}/outputs/fp32_block_quant/msopprof"
NPU_DEVICE="${NPU_DEVICE:-1}"
PTOAS_ROOT="${PTOAS_ROOT:-/home/jzhuang/work_dir/vmi_work_0804/PTOAS-main}"
SIDE="${1:-both}"

mkdir -p "${OUT}"

INNER="set -euo pipefail
source /home/jzhuang/miniconda/etc/profile.d/conda.sh
conda activate cann91_dev
set +u
source /home/jzhuang/cann_installed/9.1.0-beta.3/cann/set_env.sh
set -u
export ASCEND_RT_VISIBLE_DEVICES=0 NPU_DEVICE=0 RG_N_CORES=\${RG_N_CORES:-72}
export PTOAS_ROOT='${PTOAS_ROOT}'
export PYTHONPATH=\"\${PTOAS_ROOT}/ptodsl:\${PYTHONPATH:-}\"
export PATH=\"\${PTOAS_ROOT}/build/tools/ptoas:\${PATH}\"
export PTO_FLAGS='--pto-backend=vpto --pto-level=level3 --cann-output-version=9.1.0-beta.3'
export PTODSL_CACHE_DIR='${OUT}/ptodsl_cache'
mkdir -p \"\${PTODSL_CACHE_DIR}\"
PY='${SCRIPT_DIR}/bench_fp32_block_quant.py'
# Prewarm VMI compile outside the profiler.
python \"\${PY}\" --side vmi --shapes 8192x2048 --no-bench >/tmp/prewarm_vmi_fp32_bq.txt 2>&1 || true

profile_side() {
  local side=\"\$1\" kn
  local od='${OUT}/'\${side}_kern
  mkdir -p \"\${od}\"
  if [ \"\${side}\" = asc ]; then kn='per_block_cast_kernel'; else kn='fp32_block_quant'; fi
  echo \"=== msprof op side=\${side} kernel-name=\${kn} ===\"
  msprof op --output=\"\${od}\" --aic-metrics=Default,PipeUtilization \\
    --warm-up=3 --launch-count=1 --kill=on --kernel-name=\"\${kn}\" \\
    python \"\${PY}\" --side \"\${side}\" --shapes 8192x2048 --no-bench \\
    2>&1 | tee \"\${od}/msprof.log\" || true
  local d
  d=\$(ls -d \"\${od}\"/OPPROF_* 2>/dev/null | tail -1 || true)
  if [ -n \"\${d}\" ]; then
    echo \"--- OpBasicInfo ---\"
    cat \"\${d}/OpBasicInfo.csv\"
  fi
}

SIDE_ARG='${SIDE}'
case \"\${SIDE_ARG}\" in
  asc|vmi) profile_side \"\${SIDE_ARG}\" ;;
  both) profile_side asc; profile_side vmi ;;
  *) echo 'usage: asc|vmi|both'; exit 2 ;;
esac
"

if command -v task-submit >/dev/null 2>&1; then
  echo "Using task-submit --device ${NPU_DEVICE}"
  task-submit --device "${NPU_DEVICE}" --max-time 2400 --run "bash -lc $(printf %q "${INNER}")"
else
  bash -lc "${INNER}"
fi

echo "msopprof outputs under ${OUT}"
