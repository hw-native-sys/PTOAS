#!/usr/bin/env bash
# Real file (not a symlink): cannsim collects log_ca/instr.bin from this directory.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
KERNEL_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

mkdir -p "${KERNEL_ROOT}/log_ca"
rm -f "${KERNEL_ROOT}/instr.bin"
find "${KERNEL_ROOT}/log_ca" -mindepth 1 -delete 2>/dev/null || true

for asset in log_ca instr.bin; do
  ln -sfn "${KERNEL_ROOT}/${asset}" "${SCRIPT_DIR}/${asset}"
done

cd "${KERNEL_ROOT}"
exec python3 "$@"
