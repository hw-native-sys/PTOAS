#!/usr/bin/env bash
set -euo pipefail

PTOBC_BIN=${PTOBC_BIN:-}
if [[ -z "${PTOBC_BIN}" ]]; then
  echo "error: PTOBC_BIN not set" >&2
  exit 2
fi

TESTDATA_DIR=${TESTDATA_DIR:-}
if [[ -z "${TESTDATA_DIR}" ]]; then
  echo "error: TESTDATA_DIR not set" >&2
  exit 2
fi

IN="${TESTDATA_DIR}/tprint_v0_roundtrip.pto"
OUT_DIR=${OUT_DIR:-"${PWD}/ptobc_tprint_out"}
mkdir -p "${OUT_DIR}"

BC="${OUT_DIR}/tprint_v0_roundtrip.ptobc"
ROUNDTRIP="${OUT_DIR}/tprint_v0_roundtrip.roundtrip.pto"

"${PTOBC_BIN}" encode "${IN}" -o "${BC}"
"${PTOBC_BIN}" decode "${BC}" -o "${ROUNDTRIP}"

grep -F "pto.tprint ins(" "${ROUNDTRIP}" >/dev/null
