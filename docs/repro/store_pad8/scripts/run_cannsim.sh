#!/usr/bin/env bash
# Run store_pad8 correctness under cannsim.
# Usage: STORE_PAD8_CASE=large|small ./scripts/run_cannsim.sh <backend: cce|vmi>
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KERNEL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env.sh"

BACKEND="${1:-vmi}"
export TLVF_VMI_BACKEND="${BACKEND}"
STORE_PAD8_CASE="${STORE_PAD8_CASE:-large}"
export STORE_PAD8_CASE

force_build() {
  python3 -c "
import sys
sys.path.insert(0, '${KERNEL_ROOT}')
from common.cce_vf_build import build_cce_root
from pathlib import Path
build_cce_root(Path('${KERNEL_ROOT}/cce'), force=True)
"
}

TEST="${KERNEL_ROOT}/test/test_store_pad8.py"
if [ "${BACKEND}" = "cce" ]; then
  force_build
fi
OUT="${KERNEL_ROOT}/sim_outputs/store_pad8_${STORE_PAD8_CASE}_${BACKEND}"
mkdir -p "${OUT}"
exec "${SCRIPT_DIR}/run_sim.sh" "${TEST}" "${OUT}"
