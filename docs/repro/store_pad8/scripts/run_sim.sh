#!/usr/bin/env bash
# cannsim record + report. Usage: run_sim.sh <test.py> [out_dir]
set -euo pipefail
SHARED="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="$(realpath "${1:?Usage: run_sim.sh <test.py> [out_dir]}")"
KERNEL_ROOT="$(cd "$(dirname "${SCRIPT}")/.." && pwd)"
OUT_DIR="${2:-${KERNEL_ROOT}/sim_outputs/$(basename "$SCRIPT" .py)}"
ENTRY="${SHARED}/run_sim_entry.sh"
mkdir -p "${OUT_DIR}"

# cannsim mkdirs log_ca next to the entry script before the entry runs.
for asset in log_ca instr.bin; do
  link="${SHARED}/${asset}"
  if [ -L "${link}" ] || [ -e "${link}" ]; then
    rm -rf "${link}"
  fi
done
rm -rf "${KERNEL_ROOT}/log_ca" "${KERNEL_ROOT}/instr.bin"
mkdir -p "${KERNEL_ROOT}/log_ca"

echo "==> cannsim record ${SCRIPT} -> ${OUT_DIR}"
set +e
cannsim record "${ENTRY}" -s Ascend950 --gen-report -o "${OUT_DIR}" -u "${SCRIPT}"
rc=$?
set -e
echo "cannsim exit code: ${rc}"

RUN="$(find "${OUT_DIR}" -maxdepth 1 -type d -name 'cannsim_*' 2>/dev/null | sort | tail -1 || true)"
if [ -n "${RUN}" ]; then
  [ -d "${KERNEL_ROOT}/log_ca" ] && [ ! -d "${RUN}/log_ca" ] && cp -a "${KERNEL_ROOT}/log_ca" "${RUN}/log_ca"
  if [ -f "${KERNEL_ROOT}/instr.bin" ]; then
    dest="${RUN}/instr.bin"
    if [ ! -f "${dest}" ] || [ -L "${dest}" ] || [ "${dest}" -ef "${KERNEL_ROOT}/instr.bin" ]; then
      cp -f "${KERNEL_ROOT}/instr.bin" "${RUN}/.instr.bin.tmp" && mv -f "${RUN}/.instr.bin.tmp" "${dest}"
    fi
  fi
  if [ -f "${RUN}/instr.bin" ]; then
    mkdir -p "${RUN}/report"
    cannsim report -e "${RUN}" -o "${RUN}/report" -n 0 || true
  fi
fi

python3 "${SHARED}/cannsim_metrics.py" "${OUT_DIR}" || true
exit 0
