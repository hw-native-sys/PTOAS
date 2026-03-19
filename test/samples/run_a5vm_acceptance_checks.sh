#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PTOAS_BIN="${PTOAS_BIN:-${ROOT_DIR}/build/tools/ptoas/ptoas}"
OUT_DIR="${PTOAS_SAMPLE_ACCEPTANCE_OUT:-${ROOT_DIR}/build/test-sample-acceptance}"

require_pattern() {
  local pattern="$1"
  local file="$2"
  local message="$3"
  if ! rg -n "${pattern}" "${file}" >/dev/null; then
    echo "error: ${message}" >&2
    echo "searched pattern: ${pattern}" >&2
    echo "in file: ${file}" >&2
    exit 1
  fi
}

if [[ ! -x "${PTOAS_BIN}" ]]; then
  echo "error: missing ptoas binary: ${PTOAS_BIN}" >&2
  exit 1
fi

mkdir -p "${OUT_DIR}"

# Shell environment for local Ascend/PTO setup.
if [[ -f "${ROOT_DIR}/env.sh" ]]; then
  # shellcheck disable=SC1091
  set +u
  source "${ROOT_DIR}/env.sh"
  set -u
fi

echo "sample acceptance: Abs"
ABS_OUT="${OUT_DIR}/abs"
rm -rf "${ABS_OUT}"
mkdir -p "${ABS_OUT}"
(
  cd "${ROOT_DIR}"
  PTOAS_BIN="${PTOAS_BIN}" \
  PTOAS_OUT_DIR="${ABS_OUT}" \
  PTOAS_FLAGS="--pto-arch a5 --pto-backend=a5vm --a5vm-print-ir" \
  ./test/samples/runop.sh -t Abs
) > "${ABS_OUT}/run.log" 2>&1
require_pattern 'Abs\(abs\.py\)[[:space:]]+OK' "${ABS_OUT}/run.log" \
  "Abs sample did not compile successfully"
ABS_CPP="${ABS_OUT}/Abs/abs-pto.cpp"
[[ -f "${ABS_CPP}" ]] || { echo "error: missing ${ABS_CPP}" >&2; exit 1; }
require_pattern 'a5vm\.copy_gm_to_ubuf' "${ABS_CPP}" \
  "Abs output lost TLOAD lowering"
require_pattern 'a5vm\.vabs' "${ABS_CPP}" \
  "Abs output lost TABS vector lowering"
require_pattern 'llvm\.loop\.aivector_scope' "${ABS_CPP}" \
  "Abs output lost vec-scope loop carrier"
require_pattern 'a5vm\.copy_ubuf_to_gm' "${ABS_CPP}" \
  "Abs output lost TSTORE lowering"

echo "sample acceptance: Sync/add_double_dynamic"
SYNC_OUT="${OUT_DIR}/sync"
rm -rf "${SYNC_OUT}"
mkdir -p "${SYNC_OUT}"
(
  cd "${ROOT_DIR}"
  PTOAS_BIN="${PTOAS_BIN}" \
  PTOAS_OUT_DIR="${SYNC_OUT}" \
  PTOAS_FLAGS="--pto-arch a5 --pto-backend=a5vm --a5vm-print-ir" \
  ./test/samples/runop.sh -t Sync
) > "${SYNC_OUT}/run.log" 2>&1 || true
require_pattern 'Sync\(add_double_dynamic\.py\)[[:space:]]+OK' "${SYNC_OUT}/run.log" \
  "Sync/add_double_dynamic sample did not compile successfully"
ADD_DOUBLE_OUT="${SYNC_OUT}/Sync/add_double_dynamic-pto.cpp"
[[ -f "${ADD_DOUBLE_OUT}" ]] || { echo "error: missing ${ADD_DOUBLE_OUT}" >&2; exit 1; }
require_pattern 'a5vm\.set_loop2_stride_outtoub' "${ADD_DOUBLE_OUT}" \
  "dynamic TLOAD path did not program copy loop registers"
require_pattern 'a5vm\.copy_gm_to_ubuf' "${ADD_DOUBLE_OUT}" \
  "dynamic TLOAD path did not lower to copy_gm_to_ubuf"
require_pattern 'a5vm\.vadd' "${ADD_DOUBLE_OUT}" \
  "dynamic elementwise body did not lower to a5vm vector ops"
require_pattern 'a5vm\.copy_ubuf_to_gm' "${ADD_DOUBLE_OUT}" \
  "dynamic TSTORE path did not lower to copy_ubuf_to_gm"

echo "sample acceptance: PASS"
