#!/usr/bin/env bash
# Three-way reduce store: CCE ONEPT vs VMI pad8 vs VMI mask1.
# Usage: STORE_PAD8_CASE=large|small ./scripts/run_three_way.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KERNEL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/env.sh"

CASE="${STORE_PAD8_CASE:-large}"
export STORE_PAD8_CASE="${CASE}"
TEST="${KERNEL_ROOT}/test/test_store_pad8.py"

echo "=== CCE ONEPT STORE_PAD8_CASE=${CASE} ==="
STORE_PAD8_CASE="${CASE}" "${SCRIPT_DIR}/run_cannsim.sh" cce || true

echo "=== VMI pad8 STORE_PAD8_CASE=${CASE} ==="
export STORE_PAD8_VARIANT=pad8
STORE_PAD8_CASE="${CASE}" "${SCRIPT_DIR}/run_cannsim.sh" vmi || true

echo "=== VMI mask1 STORE_PAD8_CASE=${CASE} ==="
export STORE_PAD8_VARIANT=mask1
OUT="${KERNEL_ROOT}/sim_outputs/store_pad8_${CASE}_vmi_mask1"
mkdir -p "${OUT}"
TLVF_VMI_BACKEND=vmi STORE_PAD8_VARIANT=mask1 STORE_PAD8_CASE="${CASE}" \
  "${SCRIPT_DIR}/run_sim.sh" "${TEST}" "${OUT}" || true

echo "=== metrics ==="
python3 "${SCRIPT_DIR}/cannsim_metrics.py" \
  "${KERNEL_ROOT}/sim_outputs/store_pad8_${CASE}_cce" \
  "${KERNEL_ROOT}/sim_outputs/store_pad8_${CASE}_vmi" \
  "${KERNEL_ROOT}/sim_outputs/store_pad8_${CASE}_vmi_mask1" \
  --table || true
