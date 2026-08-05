#!/usr/bin/env bash
# Build libfp32_block_quant.so from reference AscendC + ctypes host entry.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPRO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
FIXTURES="${REPRO_ROOT}/fixtures"
OUT_DIR="${FIXTURES}/fp32_block_quant_artifact"
OUT_SO="${OUT_DIR}/libfp32_block_quant.so"
NPU_ARCH="${ASCEND_NPU_ARCH:-dav-3510}"

ASCEND="${ASCEND_HOME_PATH:-${ASCEND_TOOLKIT_HOME:-}}"
if [ -z "${ASCEND}" ]; then
  echo "ERROR: set ASCEND_HOME_PATH" >&2
  exit 1
fi
BISHENG="${BISHENG:-${ASCEND}/tools/bisheng_compiler/bin/bisheng}"
if [ ! -x "${BISHENG}" ]; then
  BISHENG="$(command -v bisheng || true)"
fi
if [ -z "${BISHENG}" ] || [ ! -x "${BISHENG}" ]; then
  echo "ERROR: bisheng not found" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"
TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT
# Single TU: device kernel + host call_ (bisheng --shared expects one .asc-like unit).
{
  cat "${FIXTURES}/reference_asc_fp32_block_quant.asc"
  echo
  # Host stub without re-including kernel_operator / redefining fp8 alias.
  sed -e '/^#include "kernel_operator.h"/d' \
      -e '/^using fp8_e4_t/d' \
      "${FIXTURES}/fp32_block_quant_host.cpp"
} > "${TMP}/fp32_block_quant_lib.asc"

echo "==> ${BISHENG} --shared -> ${OUT_SO}"
"${BISHENG}" -O2 -fPIC --shared --npu-arch="${NPU_ARCH}" \
  "${TMP}/fp32_block_quant_lib.asc" -o "${OUT_SO}" \
  -I"${ASCEND}/include" \
  -I"${ASCEND}/compiler/tikcpp/tikcfw" \
  -I"${ASCEND}/compiler/tikcpp/tikcfw/impl" \
  -I"${ASCEND}/compiler/tikcpp/tikcfw/interface" \
  -Wno-unused-variable -Wno-ignored-attributes -Wno-unknown-attributes
ls -la "${OUT_SO}"
echo "Built ${OUT_SO}"
