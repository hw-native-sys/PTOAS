#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PTOAS_BIN="${PTOAS_BIN:-${ROOT_DIR}/build/tools/ptoas/ptoas}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
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

require_no_pattern() {
  local pattern="$1"
  local file="$2"
  local message="$3"
  if rg -n "${pattern}" "${file}" >/dev/null; then
    echo "error: ${message}" >&2
    echo "unexpected pattern: ${pattern}" >&2
    echo "in file: ${file}" >&2
    exit 1
  fi
}

if [[ ! -x "${PTOAS_BIN}" ]]; then
  echo "error: missing ptoas binary: ${PTOAS_BIN}" >&2
  exit 1
fi
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "error: missing python binary: ${PYTHON_BIN}" >&2
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

echo "sample acceptance: strategy evidence (Abs/Div/Rowexpand/Adds/Maxs/Mins/Lrelu/Muls/Divs)"
STRATEGY_OUT="${OUT_DIR}/strategy-evidence"
rm -rf "${STRATEGY_OUT}"
mkdir -p "${STRATEGY_OUT}"

"${PYTHON_BIN}" "${ROOT_DIR}/test/samples/Abs/abs.py" > "${STRATEGY_OUT}/abs.pto"
"${PTOAS_BIN}" --pto-arch a5 --pto-backend=a5vm \
  --a5vm-print-ir \
  "${STRATEGY_OUT}/abs.pto" -o "${STRATEGY_OUT}/abs-post.cpp" \
  > "${STRATEGY_OUT}/abs-post.a5vm" 2>&1
"${PTOAS_BIN}" --pto-arch a5 --pto-backend=a5vm \
  --a5vm-lowering-strategy no-post-update \
  --a5vm-print-ir \
  "${STRATEGY_OUT}/abs.pto" -o "${STRATEGY_OUT}/abs-nopost.cpp" \
  > "${STRATEGY_OUT}/abs-nopost.a5vm" 2>&1
"${PTOAS_BIN}" --pto-arch a5 --pto-backend=a5vm \
  --a5vm-emit-hivm-llvm \
  "${STRATEGY_OUT}/abs.pto" -o "${STRATEGY_OUT}/abs-post.ll"
"${PTOAS_BIN}" --pto-arch a5 --pto-backend=a5vm \
  --a5vm-lowering-strategy no-post-update \
  --a5vm-emit-hivm-llvm \
  "${STRATEGY_OUT}/abs.pto" -o "${STRATEGY_OUT}/abs-nopost.ll"
require_pattern 'a5vm\.vlds_post ' "${STRATEGY_OUT}/abs-post.a5vm" \
  "Abs post-update A5VM IR lost post-update vlds mode"
require_pattern 'a5vm\.vsts_post ' "${STRATEGY_OUT}/abs-post.a5vm" \
  "Abs post-update A5VM IR lost post-update vsts mode"
require_no_pattern 'a5vm\.vlds_post|a5vm\.vsts_post' "${STRATEGY_OUT}/abs-nopost.a5vm" \
  "Abs no-post-update A5VM IR still contains post-update mode"
require_pattern 'llvm\.hivm\.vldsx1\.post\.v64f32' "${STRATEGY_OUT}/abs-post.ll" \
  "Abs post-update LLVM lost .post load intrinsic"
require_pattern 'llvm\.hivm\.vstsx1\.post\.v64f32' "${STRATEGY_OUT}/abs-post.ll" \
  "Abs post-update LLVM lost .post store intrinsic"
require_no_pattern 'llvm\.hivm\.vldsx1\.post\.v64f32|llvm\.hivm\.vstsx1\.post\.v64f32' "${STRATEGY_OUT}/abs-nopost.ll" \
  "Abs no-post-update LLVM still contains .post intrinsics"

"${PYTHON_BIN}" "${ROOT_DIR}/test/samples/Div/div.py" > "${STRATEGY_OUT}/div.pto"
"${PTOAS_BIN}" --pto-arch a5 --pto-backend=a5vm \
  --a5vm-print-ir \
  "${STRATEGY_OUT}/div.pto" -o "${STRATEGY_OUT}/div-post.cpp" \
  > "${STRATEGY_OUT}/div-post.a5vm" 2>&1
"${PTOAS_BIN}" --pto-arch a5 --pto-backend=a5vm \
  --a5vm-lowering-strategy no-post-update \
  --a5vm-print-ir \
  "${STRATEGY_OUT}/div.pto" -o "${STRATEGY_OUT}/div-nopost.cpp" \
  > "${STRATEGY_OUT}/div-nopost.a5vm" 2>&1
"${PTOAS_BIN}" --pto-arch a5 --pto-backend=a5vm \
  --a5vm-emit-hivm-llvm \
  "${STRATEGY_OUT}/div.pto" -o "${STRATEGY_OUT}/div-post.ll"
"${PTOAS_BIN}" --pto-arch a5 --pto-backend=a5vm \
  --a5vm-lowering-strategy no-post-update \
  --a5vm-emit-hivm-llvm \
  "${STRATEGY_OUT}/div.pto" -o "${STRATEGY_OUT}/div-nopost.ll"
require_pattern 'a5vm\.vlds_post ' "${STRATEGY_OUT}/div-post.a5vm" \
  "Div post-update A5VM IR lost post-update load mode"
require_pattern 'a5vm\.vsts_post ' "${STRATEGY_OUT}/div-post.a5vm" \
  "Div post-update A5VM IR lost post-update store mode"
require_no_pattern 'a5vm\.vlds_post|a5vm\.vsts_post' "${STRATEGY_OUT}/div-nopost.a5vm" \
  "Div no-post-update A5VM IR still contains post-update mode"
require_pattern 'llvm\.hivm\.vldsx1\.post\.v64f32' "${STRATEGY_OUT}/div-post.ll" \
  "Div post-update LLVM lost .post load intrinsic"
require_pattern 'llvm\.hivm\.vstsx1\.post\.v64f32' "${STRATEGY_OUT}/div-post.ll" \
  "Div post-update LLVM lost .post store intrinsic"
require_no_pattern 'llvm\.hivm\.vldsx1\.post\.v64f32|llvm\.hivm\.vstsx1\.post\.v64f32' "${STRATEGY_OUT}/div-nopost.ll" \
  "Div no-post-update LLVM still contains .post intrinsics"

"${PYTHON_BIN}" "${ROOT_DIR}/test/samples/Rowexpand/rowexpand.py" > "${STRATEGY_OUT}/rowexpand.pto"
"${PTOAS_BIN}" --pto-arch a5 --pto-backend=a5vm \
  --a5vm-print-ir \
  "${STRATEGY_OUT}/rowexpand.pto" -o "${STRATEGY_OUT}/rowexpand-post.cpp" \
  > "${STRATEGY_OUT}/rowexpand-post.a5vm" 2>&1
"${PTOAS_BIN}" --pto-arch a5 --pto-backend=a5vm \
  --a5vm-lowering-strategy no-post-update \
  --a5vm-print-ir \
  "${STRATEGY_OUT}/rowexpand.pto" -o "${STRATEGY_OUT}/rowexpand-nopost.cpp" \
  > "${STRATEGY_OUT}/rowexpand-nopost.a5vm" 2>&1
"${PTOAS_BIN}" --pto-arch a5 --pto-backend=a5vm \
  --a5vm-emit-hivm-llvm \
  "${STRATEGY_OUT}/rowexpand.pto" -o "${STRATEGY_OUT}/rowexpand-post.ll"
"${PTOAS_BIN}" --pto-arch a5 --pto-backend=a5vm \
  --a5vm-lowering-strategy no-post-update \
  --a5vm-emit-hivm-llvm \
  "${STRATEGY_OUT}/rowexpand.pto" -o "${STRATEGY_OUT}/rowexpand-nopost.ll"
require_pattern 'a5vm\.vlds_post ' "${STRATEGY_OUT}/rowexpand-post.a5vm" \
  "Rowexpand post-update A5VM IR lost post-update source load"
require_pattern 'a5vm\.vsts_post ' "${STRATEGY_OUT}/rowexpand-post.a5vm" \
  "Rowexpand post-update A5VM IR lost post-update destination store"
require_no_pattern 'a5vm\.vlds_post|a5vm\.vsts_post' "${STRATEGY_OUT}/rowexpand-nopost.a5vm" \
  "Rowexpand no-post-update A5VM IR still contains post-update mode"
require_pattern 'llvm\.hivm\.vldsx1\.post\.v64f32' "${STRATEGY_OUT}/rowexpand-post.ll" \
  "Rowexpand post-update LLVM lost .post load intrinsic"
require_pattern 'llvm\.hivm\.vstsx1\.post\.v64f32' "${STRATEGY_OUT}/rowexpand-post.ll" \
  "Rowexpand post-update LLVM lost .post store intrinsic"
require_no_pattern 'llvm\.hivm\.vldsx1\.post\.v64f32|llvm\.hivm\.vstsx1\.post\.v64f32' "${STRATEGY_OUT}/rowexpand-nopost.ll" \
  "Rowexpand no-post-update LLVM still contains .post intrinsics"

"${PYTHON_BIN}" "${ROOT_DIR}/test/samples/Adds/adds.py" > "${STRATEGY_OUT}/adds.pto"
"${PTOAS_BIN}" --pto-arch a5 --pto-backend=a5vm \
  --a5vm-print-ir \
  "${STRATEGY_OUT}/adds.pto" -o "${STRATEGY_OUT}/adds-post.cpp" \
  > "${STRATEGY_OUT}/adds-post.a5vm" 2>&1
"${PTOAS_BIN}" --pto-arch a5 --pto-backend=a5vm \
  --a5vm-lowering-strategy no-post-update \
  --a5vm-print-ir \
  "${STRATEGY_OUT}/adds.pto" -o "${STRATEGY_OUT}/adds-nopost.cpp" \
  > "${STRATEGY_OUT}/adds-nopost.a5vm" 2>&1
"${PTOAS_BIN}" --pto-arch a5 --pto-backend=a5vm \
  --a5vm-emit-hivm-llvm \
  "${STRATEGY_OUT}/adds.pto" -o "${STRATEGY_OUT}/adds-post.ll"
"${PTOAS_BIN}" --pto-arch a5 --pto-backend=a5vm \
  --a5vm-lowering-strategy no-post-update \
  --a5vm-emit-hivm-llvm \
  "${STRATEGY_OUT}/adds.pto" -o "${STRATEGY_OUT}/adds-nopost.ll"
require_pattern 'a5vm\.vlds_post ' "${STRATEGY_OUT}/adds-post.a5vm" \
  "Adds post-update A5VM IR lost post-update load mode"
require_pattern 'a5vm\.vsts_post ' "${STRATEGY_OUT}/adds-post.a5vm" \
  "Adds post-update A5VM IR lost post-update store mode"
require_no_pattern 'a5vm\.vlds_post|a5vm\.vsts_post' "${STRATEGY_OUT}/adds-nopost.a5vm" \
  "Adds no-post-update A5VM IR still contains post-update mode"
require_pattern 'llvm\.hivm\.vadds\.v64f32\.x' "${STRATEGY_OUT}/adds-post.ll" \
  "Adds post-update LLVM lost vadds intrinsic"
require_pattern 'llvm\.hivm\.vldsx1\.post\.v64f32' "${STRATEGY_OUT}/adds-post.ll" \
  "Adds post-update LLVM lost .post load intrinsic"
require_pattern 'llvm\.hivm\.vstsx1\.post\.v64f32' "${STRATEGY_OUT}/adds-post.ll" \
  "Adds post-update LLVM lost .post store intrinsic"
require_no_pattern 'llvm\.hivm\.vldsx1\.post\.v64f32|llvm\.hivm\.vstsx1\.post\.v64f32' "${STRATEGY_OUT}/adds-nopost.ll" \
  "Adds no-post-update LLVM still contains .post intrinsics"

"${PYTHON_BIN}" "${ROOT_DIR}/test/samples/Maxs/maxs.py" > "${STRATEGY_OUT}/maxs.pto"
"${PTOAS_BIN}" --pto-arch a5 --pto-backend=a5vm \
  --a5vm-print-ir \
  "${STRATEGY_OUT}/maxs.pto" -o "${STRATEGY_OUT}/maxs-post.cpp" \
  > "${STRATEGY_OUT}/maxs-post.a5vm" 2>&1
"${PTOAS_BIN}" --pto-arch a5 --pto-backend=a5vm \
  --a5vm-lowering-strategy no-post-update \
  --a5vm-print-ir \
  "${STRATEGY_OUT}/maxs.pto" -o "${STRATEGY_OUT}/maxs-nopost.cpp" \
  > "${STRATEGY_OUT}/maxs-nopost.a5vm" 2>&1
"${PTOAS_BIN}" --pto-arch a5 --pto-backend=a5vm \
  --a5vm-emit-hivm-llvm \
  "${STRATEGY_OUT}/maxs.pto" -o "${STRATEGY_OUT}/maxs-post.ll"
"${PTOAS_BIN}" --pto-arch a5 --pto-backend=a5vm \
  --a5vm-lowering-strategy no-post-update \
  --a5vm-emit-hivm-llvm \
  "${STRATEGY_OUT}/maxs.pto" -o "${STRATEGY_OUT}/maxs-nopost.ll"
require_pattern 'a5vm\.vlds_post ' "${STRATEGY_OUT}/maxs-post.a5vm" \
  "Maxs post-update A5VM IR lost post-update load mode"
require_pattern 'a5vm\.vsts_post ' "${STRATEGY_OUT}/maxs-post.a5vm" \
  "Maxs post-update A5VM IR lost post-update store mode"
require_no_pattern 'a5vm\.vlds_post|a5vm\.vsts_post' "${STRATEGY_OUT}/maxs-nopost.a5vm" \
  "Maxs no-post-update A5VM IR still contains post-update mode"
require_pattern 'llvm\.hivm\.vmaxs\.v64f32\.x' "${STRATEGY_OUT}/maxs-post.ll" \
  "Maxs post-update LLVM lost vmaxs intrinsic"
require_pattern 'llvm\.hivm\.vldsx1\.post\.v64f32' "${STRATEGY_OUT}/maxs-post.ll" \
  "Maxs post-update LLVM lost .post load intrinsic"
require_pattern 'llvm\.hivm\.vstsx1\.post\.v64f32' "${STRATEGY_OUT}/maxs-post.ll" \
  "Maxs post-update LLVM lost .post store intrinsic"
require_no_pattern 'llvm\.hivm\.vldsx1\.post\.v64f32|llvm\.hivm\.vstsx1\.post\.v64f32' "${STRATEGY_OUT}/maxs-nopost.ll" \
  "Maxs no-post-update LLVM still contains .post intrinsics"

"${PYTHON_BIN}" "${ROOT_DIR}/test/samples/Mins/mins.py" > "${STRATEGY_OUT}/mins.pto"
"${PTOAS_BIN}" --pto-arch a5 --pto-backend=a5vm \
  --a5vm-print-ir \
  "${STRATEGY_OUT}/mins.pto" -o "${STRATEGY_OUT}/mins-post.cpp" \
  > "${STRATEGY_OUT}/mins-post.a5vm" 2>&1
"${PTOAS_BIN}" --pto-arch a5 --pto-backend=a5vm \
  --a5vm-lowering-strategy no-post-update \
  --a5vm-print-ir \
  "${STRATEGY_OUT}/mins.pto" -o "${STRATEGY_OUT}/mins-nopost.cpp" \
  > "${STRATEGY_OUT}/mins-nopost.a5vm" 2>&1
"${PTOAS_BIN}" --pto-arch a5 --pto-backend=a5vm \
  --a5vm-emit-hivm-llvm \
  "${STRATEGY_OUT}/mins.pto" -o "${STRATEGY_OUT}/mins-post.ll"
"${PTOAS_BIN}" --pto-arch a5 --pto-backend=a5vm \
  --a5vm-lowering-strategy no-post-update \
  --a5vm-emit-hivm-llvm \
  "${STRATEGY_OUT}/mins.pto" -o "${STRATEGY_OUT}/mins-nopost.ll"
require_pattern 'a5vm\.vlds_post ' "${STRATEGY_OUT}/mins-post.a5vm" \
  "Mins post-update A5VM IR lost post-update load mode"
require_pattern 'a5vm\.vsts_post ' "${STRATEGY_OUT}/mins-post.a5vm" \
  "Mins post-update A5VM IR lost post-update store mode"
require_no_pattern 'a5vm\.vlds_post|a5vm\.vsts_post' "${STRATEGY_OUT}/mins-nopost.a5vm" \
  "Mins no-post-update A5VM IR still contains post-update mode"
require_pattern 'llvm\.hivm\.vmins\.v64f32\.x' "${STRATEGY_OUT}/mins-post.ll" \
  "Mins post-update LLVM lost vmins intrinsic"
require_pattern 'llvm\.hivm\.vldsx1\.post\.v64f32' "${STRATEGY_OUT}/mins-post.ll" \
  "Mins post-update LLVM lost .post load intrinsic"
require_pattern 'llvm\.hivm\.vstsx1\.post\.v64f32' "${STRATEGY_OUT}/mins-post.ll" \
  "Mins post-update LLVM lost .post store intrinsic"
require_no_pattern 'llvm\.hivm\.vldsx1\.post\.v64f32|llvm\.hivm\.vstsx1\.post\.v64f32' "${STRATEGY_OUT}/mins-nopost.ll" \
  "Mins no-post-update LLVM still contains .post intrinsics"

"${PYTHON_BIN}" "${ROOT_DIR}/test/samples/Lrelu/lrelu.py" > "${STRATEGY_OUT}/lrelu.pto"
"${PTOAS_BIN}" --pto-arch a5 --pto-backend=a5vm \
  --a5vm-print-ir \
  "${STRATEGY_OUT}/lrelu.pto" -o "${STRATEGY_OUT}/lrelu-post.cpp" \
  > "${STRATEGY_OUT}/lrelu-post.a5vm" 2>&1
"${PTOAS_BIN}" --pto-arch a5 --pto-backend=a5vm \
  --a5vm-lowering-strategy no-post-update \
  --a5vm-print-ir \
  "${STRATEGY_OUT}/lrelu.pto" -o "${STRATEGY_OUT}/lrelu-nopost.cpp" \
  > "${STRATEGY_OUT}/lrelu-nopost.a5vm" 2>&1
"${PTOAS_BIN}" --pto-arch a5 --pto-backend=a5vm \
  --a5vm-emit-hivm-llvm \
  "${STRATEGY_OUT}/lrelu.pto" -o "${STRATEGY_OUT}/lrelu-post.ll"
"${PTOAS_BIN}" --pto-arch a5 --pto-backend=a5vm \
  --a5vm-lowering-strategy no-post-update \
  --a5vm-emit-hivm-llvm \
  "${STRATEGY_OUT}/lrelu.pto" -o "${STRATEGY_OUT}/lrelu-nopost.ll"
require_pattern 'a5vm\.vlds_post ' "${STRATEGY_OUT}/lrelu-post.a5vm" \
  "Lrelu post-update A5VM IR lost post-update load mode"
require_pattern 'a5vm\.vsts_post ' "${STRATEGY_OUT}/lrelu-post.a5vm" \
  "Lrelu post-update A5VM IR lost post-update store mode"
require_no_pattern 'a5vm\.vlds_post|a5vm\.vsts_post' "${STRATEGY_OUT}/lrelu-nopost.a5vm" \
  "Lrelu no-post-update A5VM IR still contains post-update mode"
require_pattern 'llvm\.hivm\.vlrelu\.v64f32\.x' "${STRATEGY_OUT}/lrelu-post.ll" \
  "Lrelu post-update LLVM lost vlrelu intrinsic"
require_pattern 'llvm\.hivm\.vldsx1\.post\.v64f32' "${STRATEGY_OUT}/lrelu-post.ll" \
  "Lrelu post-update LLVM lost .post load intrinsic"
require_pattern 'llvm\.hivm\.vstsx1\.post\.v64f32' "${STRATEGY_OUT}/lrelu-post.ll" \
  "Lrelu post-update LLVM lost .post store intrinsic"
require_no_pattern 'llvm\.hivm\.vldsx1\.post\.v64f32|llvm\.hivm\.vstsx1\.post\.v64f32' "${STRATEGY_OUT}/lrelu-nopost.ll" \
  "Lrelu no-post-update LLVM still contains .post intrinsics"

"${PYTHON_BIN}" "${ROOT_DIR}/test/samples/Muls/muls.py" > "${STRATEGY_OUT}/muls.pto"
"${PTOAS_BIN}" --pto-arch a5 --pto-backend=a5vm \
  --a5vm-print-ir \
  "${STRATEGY_OUT}/muls.pto" -o "${STRATEGY_OUT}/muls-post.cpp" \
  > "${STRATEGY_OUT}/muls-post.a5vm" 2>&1
"${PTOAS_BIN}" --pto-arch a5 --pto-backend=a5vm \
  --a5vm-lowering-strategy no-post-update \
  --a5vm-print-ir \
  "${STRATEGY_OUT}/muls.pto" -o "${STRATEGY_OUT}/muls-nopost.cpp" \
  > "${STRATEGY_OUT}/muls-nopost.a5vm" 2>&1
"${PTOAS_BIN}" --pto-arch a5 --pto-backend=a5vm \
  --a5vm-emit-hivm-llvm \
  "${STRATEGY_OUT}/muls.pto" -o "${STRATEGY_OUT}/muls-post.ll"
"${PTOAS_BIN}" --pto-arch a5 --pto-backend=a5vm \
  --a5vm-lowering-strategy no-post-update \
  --a5vm-emit-hivm-llvm \
  "${STRATEGY_OUT}/muls.pto" -o "${STRATEGY_OUT}/muls-nopost.ll"
require_pattern 'a5vm\.vlds_post ' "${STRATEGY_OUT}/muls-post.a5vm" \
  "Muls post-update A5VM IR lost post-update load mode"
require_pattern 'a5vm\.vsts_post ' "${STRATEGY_OUT}/muls-post.a5vm" \
  "Muls post-update A5VM IR lost post-update store mode"
require_no_pattern 'a5vm\.vlds_post|a5vm\.vsts_post' "${STRATEGY_OUT}/muls-nopost.a5vm" \
  "Muls no-post-update A5VM IR still contains post-update mode"
require_pattern 'llvm\.hivm\.vldsx1\.post\.v64f32' "${STRATEGY_OUT}/muls-post.ll" \
  "Muls post-update LLVM lost .post load intrinsic"
require_pattern 'llvm\.hivm\.vstsx1\.post\.v64f32' "${STRATEGY_OUT}/muls-post.ll" \
  "Muls post-update LLVM lost .post store intrinsic"
require_no_pattern 'llvm\.hivm\.vldsx1\.post\.v64f32|llvm\.hivm\.vstsx1\.post\.v64f32' "${STRATEGY_OUT}/muls-nopost.ll" \
  "Muls no-post-update LLVM still contains .post intrinsics"

"${PYTHON_BIN}" "${ROOT_DIR}/test/samples/Divs/divs.py" > "${STRATEGY_OUT}/divs.pto"
"${PTOAS_BIN}" --pto-arch a5 --pto-backend=a5vm \
  --a5vm-print-ir \
  "${STRATEGY_OUT}/divs.pto" -o "${STRATEGY_OUT}/divs-post.cpp" \
  > "${STRATEGY_OUT}/divs-post.a5vm" 2>&1
"${PTOAS_BIN}" --pto-arch a5 --pto-backend=a5vm \
  --a5vm-lowering-strategy no-post-update \
  --a5vm-print-ir \
  "${STRATEGY_OUT}/divs.pto" -o "${STRATEGY_OUT}/divs-nopost.cpp" \
  > "${STRATEGY_OUT}/divs-nopost.a5vm" 2>&1
"${PTOAS_BIN}" --pto-arch a5 --pto-backend=a5vm \
  --a5vm-emit-hivm-llvm \
  "${STRATEGY_OUT}/divs.pto" -o "${STRATEGY_OUT}/divs-post.ll"
"${PTOAS_BIN}" --pto-arch a5 --pto-backend=a5vm \
  --a5vm-lowering-strategy no-post-update \
  --a5vm-emit-hivm-llvm \
  "${STRATEGY_OUT}/divs.pto" -o "${STRATEGY_OUT}/divs-nopost.ll"
require_pattern 'a5vm\.vlds_post ' "${STRATEGY_OUT}/divs-post.a5vm" \
  "Divs post-update A5VM IR lost post-update load mode"
require_pattern 'a5vm\.vsts_post ' "${STRATEGY_OUT}/divs-post.a5vm" \
  "Divs post-update A5VM IR lost post-update store mode"
require_no_pattern 'a5vm\.vlds_post|a5vm\.vsts_post' "${STRATEGY_OUT}/divs-nopost.a5vm" \
  "Divs no-post-update A5VM IR still contains post-update mode"
require_pattern 'llvm\.hivm\.vldsx1\.post\.v64f32' "${STRATEGY_OUT}/divs-post.ll" \
  "Divs post-update LLVM lost .post load intrinsic"
require_pattern 'llvm\.hivm\.vstsx1\.post\.v64f32' "${STRATEGY_OUT}/divs-post.ll" \
  "Divs post-update LLVM lost .post store intrinsic"
require_no_pattern 'llvm\.hivm\.vldsx1\.post\.v64f32|llvm\.hivm\.vstsx1\.post\.v64f32' "${STRATEGY_OUT}/divs-nopost.ll" \
  "Divs no-post-update LLVM still contains .post intrinsics"

echo "sample acceptance: Expands scalar fill"
EXPANDS_OUT="${OUT_DIR}/expands"
rm -rf "${EXPANDS_OUT}"
mkdir -p "${EXPANDS_OUT}"
"${PYTHON_BIN}" "${ROOT_DIR}/test/samples/Expands/expands.py" > "${EXPANDS_OUT}/expands.pto"
"${PTOAS_BIN}" --pto-arch a5 --pto-backend=a5vm \
  --a5vm-print-ir \
  "${EXPANDS_OUT}/expands.pto" -o "${EXPANDS_OUT}/expands.cpp" \
  > "${EXPANDS_OUT}/expands.a5vm" 2>&1
"${PTOAS_BIN}" --pto-arch a5 --pto-backend=a5vm \
  --a5vm-emit-hivm-llvm \
  "${EXPANDS_OUT}/expands.pto" -o "${EXPANDS_OUT}/expands.ll"
require_pattern 'a5vm\.vdup' "${EXPANDS_OUT}/expands.a5vm" \
  "Expands A5VM IR lost vdup scalar broadcast"
require_pattern 'a5vm\.vsts_post ' "${EXPANDS_OUT}/expands.a5vm" \
  "Expands A5VM IR lost post-update store branch"
require_pattern 'a5vm\.vsts .*: !a5vm\.vec<64xf32>, !llvm\.ptr<6>, !a5vm\.mask$' "${EXPANDS_OUT}/expands.a5vm" \
  "Expands A5VM IR lost indexed store branch"
require_pattern 'llvm\.hivm\.vdups\.v64f32\.z' "${EXPANDS_OUT}/expands.ll" \
  "Expands LLVM IR lost scalar broadcast intrinsic"
require_pattern 'llvm\.hivm\.vstsx1\.post\.v64f32' "${EXPANDS_OUT}/expands.ll" \
  "Expands LLVM IR lost post-update store intrinsic"
require_pattern 'llvm\.hivm\.vstsx1\.v64f32' "${EXPANDS_OUT}/expands.ll" \
  "Expands LLVM IR lost indexed store intrinsic"

echo "sample acceptance: Neg"
NEG_OUT="${OUT_DIR}/neg"
rm -rf "${NEG_OUT}"
mkdir -p "${NEG_OUT}"
(
  cd "${ROOT_DIR}"
  SOC_VERSION="A5" \
  PTOAS_BIN="${PTOAS_BIN}" \
  PTOAS_OUT_DIR="${NEG_OUT}" \
  PTOAS_FLAGS="--pto-arch a5 --pto-backend=a5vm --a5vm-print-ir" \
  ./test/samples/runop.sh -t Neg
) > "${NEG_OUT}/run.log" 2>&1
require_pattern 'Neg\(neg\.py\)[[:space:]]+OK' "${NEG_OUT}/run.log" \
  "Neg sample did not compile successfully"
NEG_CPP="${NEG_OUT}/Neg/neg-pto.cpp"
[[ -f "${NEG_CPP}" ]] || { echo "error: missing ${NEG_CPP}" >&2; exit 1; }
require_pattern 'a5vm\.vmuls' "${NEG_CPP}" \
  "Neg output lost PTO A5 TMULS-based lowering"
if rg -n 'pto\.tneg' "${NEG_CPP}" >/dev/null; then
  echo "error: Neg output still contains pto.tneg" >&2
  exit 1
fi

echo "sample acceptance: Lrelu"
LRELU_OUT="${OUT_DIR}/lrelu"
rm -rf "${LRELU_OUT}"
mkdir -p "${LRELU_OUT}"
(
  cd "${ROOT_DIR}"
  SOC_VERSION="A5" \
  PTOAS_BIN="${PTOAS_BIN}" \
  PTOAS_OUT_DIR="${LRELU_OUT}" \
  PTOAS_FLAGS="--pto-arch a5 --pto-backend=a5vm --a5vm-print-ir" \
  ./test/samples/runop.sh -t Lrelu
) > "${LRELU_OUT}/run.log" 2>&1
require_pattern 'Lrelu\(lrelu\.py\)[[:space:]]+OK' "${LRELU_OUT}/run.log" \
  "Lrelu sample did not compile successfully"
LRELU_CPP="${LRELU_OUT}/Lrelu/lrelu-pto.cpp"
[[ -f "${LRELU_CPP}" ]] || { echo "error: missing ${LRELU_CPP}" >&2; exit 1; }
require_pattern 'a5vm\.vlrelu' "${LRELU_CPP}" \
  "Lrelu output lost PTO A5 vlrelu lowering"
if rg -n 'pto\.tlrelu' "${LRELU_CPP}" >/dev/null; then
  echo "error: Lrelu output still contains pto.tlrelu" >&2
  exit 1
fi

echo "sample acceptance: Trans"
TRANS_OUT="${OUT_DIR}/trans"
rm -rf "${TRANS_OUT}"
mkdir -p "${TRANS_OUT}"
(
  cd "${ROOT_DIR}"
  SOC_VERSION="A5" \
  PTOAS_BIN="${PTOAS_BIN}" \
  PTOAS_OUT_DIR="${TRANS_OUT}" \
  PTOAS_FLAGS="--pto-arch a5 --pto-backend=a5vm --a5vm-print-ir" \
  ./test/samples/runop.sh -t Trans
) > "${TRANS_OUT}/run.log" 2>&1
require_pattern 'Trans\(trans\.py\)[[:space:]]+OK' "${TRANS_OUT}/run.log" \
  "Trans sample did not compile successfully"
TRANS_CPP="${TRANS_OUT}/Trans/trans-pto.cpp"
[[ -f "${TRANS_CPP}" ]] || { echo "error: missing ${TRANS_CPP}" >&2; exit 1; }
require_pattern 'a5vm\.vci' "${TRANS_CPP}" \
  "Trans output lost PTO A5 index-vector materialization"
require_pattern 'a5vm\.vgather2' "${TRANS_CPP}" \
  "Trans output lost PTO A5 gather lowering"
if rg -n 'pto\.ttrans' "${TRANS_CPP}" >/dev/null; then
  echo "error: Trans output still contains pto.ttrans" >&2
  exit 1
fi

echo "sample acceptance: Sort32"
SORT32_OUT="${OUT_DIR}/sort32"
rm -rf "${SORT32_OUT}"
mkdir -p "${SORT32_OUT}"
(
  cd "${ROOT_DIR}"
  SOC_VERSION="A5" \
  PTOAS_BIN="${PTOAS_BIN}" \
  PTOAS_OUT_DIR="${SORT32_OUT}" \
  PTOAS_FLAGS="--pto-arch a5 --pto-backend=a5vm --a5vm-print-ir" \
  ./test/samples/runop.sh -t Sort32
) > "${SORT32_OUT}/run.log" 2>&1
require_pattern 'Sort32\(sort32\.py\)[[:space:]]+OK' "${SORT32_OUT}/run.log" \
  "Sort32 sample did not compile successfully"
SORT32_CPP="${SORT32_OUT}/Sort32/sort32-pto.cpp"
[[ -f "${SORT32_CPP}" ]] || { echo "error: missing ${SORT32_CPP}" >&2; exit 1; }
require_pattern 'a5vm\.vbitsort' "${SORT32_CPP}" \
  "Sort32 output lost PTO A5 vbitsort lowering"
if rg -n 'pto\.(tsort32|pointer_cast|bind_tile)' "${SORT32_CPP}" >/dev/null; then
  echo "error: Sort32 output still contains residual PTO sort scaffold" >&2
  exit 1
fi

echo "sample acceptance: Mrgsort"
MRGSORT_OUT="${OUT_DIR}/mrgsort"
rm -rf "${MRGSORT_OUT}"
mkdir -p "${MRGSORT_OUT}"
(
  cd "${ROOT_DIR}"
  SOC_VERSION="A5" \
  PTOAS_BIN="${PTOAS_BIN}" \
  PTOAS_OUT_DIR="${MRGSORT_OUT}" \
  PTOAS_FLAGS="--pto-arch a5 --pto-backend=a5vm --a5vm-print-ir" \
  ./test/samples/runop.sh -t Mrgsort
) > "${MRGSORT_OUT}/run.log" 2>&1
require_pattern 'Mrgsort\(mrgsort\.py\)[[:space:]]+OK' "${MRGSORT_OUT}/run.log" \
  "Mrgsort sample did not compile successfully"
require_pattern 'Mrgsort\(mrgsort_a5\.py\)[[:space:]]+OK' "${MRGSORT_OUT}/run.log" \
  "Mrgsort A5 sample did not compile successfully"
require_pattern 'Mrgsort\(mrgsort_format2\.py\)[[:space:]]+OK' "${MRGSORT_OUT}/run.log" \
  "Mrgsort format2 sample did not compile successfully"
MRGSORT_CPP="${MRGSORT_OUT}/Mrgsort/mrgsort-pto.cpp"
MRGSORT_A5_CPP="${MRGSORT_OUT}/Mrgsort/mrgsort_a5-pto.cpp"
MRGSORT_FORMAT2_CPP="${MRGSORT_OUT}/Mrgsort/mrgsort_format2-pto.cpp"
[[ -f "${MRGSORT_CPP}" ]] || { echo "error: missing ${MRGSORT_CPP}" >&2; exit 1; }
[[ -f "${MRGSORT_A5_CPP}" ]] || { echo "error: missing ${MRGSORT_A5_CPP}" >&2; exit 1; }
[[ -f "${MRGSORT_FORMAT2_CPP}" ]] || { echo "error: missing ${MRGSORT_FORMAT2_CPP}" >&2; exit 1; }
require_pattern 'a5vm\.vmrgsort4' "${MRGSORT_CPP}" \
  "Mrgsort output lost PTO A5 vmrgsort4 lowering"
require_pattern 'a5vm\.vmrgsort4' "${MRGSORT_A5_CPP}" \
  "Mrgsort A5 output lost PTO A5 vmrgsort4 lowering"
require_pattern 'a5vm\.vmrgsort4' "${MRGSORT_FORMAT2_CPP}" \
  "Mrgsort format2 output lost PTO A5 vmrgsort4 lowering"
require_pattern 'a5vm\.copy_ubuf_to_ubuf' "${MRGSORT_FORMAT2_CPP}" \
  "Mrgsort format2 output lost PTO A5 intermediate ubuf copy"
if rg -n 'pto\.(tmrgsort|pointer_cast|bind_tile)' \
  "${MRGSORT_CPP}" "${MRGSORT_A5_CPP}" "${MRGSORT_FORMAT2_CPP}" >/dev/null; then
  echo "error: Mrgsort output still contains residual PTO sort scaffold" >&2
  exit 1
fi

echo "sample acceptance: Fillpad"
FILLPAD_OUT="${OUT_DIR}/fillpad"
rm -rf "${FILLPAD_OUT}"
mkdir -p "${FILLPAD_OUT}"
(
  cd "${ROOT_DIR}"
  SOC_VERSION="A5" \
  PTOAS_BIN="${PTOAS_BIN}" \
  PTOAS_OUT_DIR="${FILLPAD_OUT}" \
  PTOAS_FLAGS="--pto-arch a5 --pto-backend=a5vm --a5vm-print-ir" \
  ./test/samples/runop.sh -t Fillpad
) > "${FILLPAD_OUT}/run.log" 2>&1
require_pattern 'Fillpad\(fillpad\.py\)[[:space:]]+OK' "${FILLPAD_OUT}/run.log" \
  "Fillpad sample did not compile successfully"
require_pattern 'Fillpad\(fillpad_expand\.py\)[[:space:]]+OK' "${FILLPAD_OUT}/run.log" \
  "Fillpad expand sample did not compile successfully"
FILLPAD_CPP="${FILLPAD_OUT}/Fillpad/fillpad-pto.cpp"
FILLPAD_EXPAND_CPP="${FILLPAD_OUT}/Fillpad/fillpad_expand-pto.cpp"
[[ -f "${FILLPAD_CPP}" ]] || { echo "error: missing ${FILLPAD_CPP}" >&2; exit 1; }
[[ -f "${FILLPAD_EXPAND_CPP}" ]] || { echo "error: missing ${FILLPAD_EXPAND_CPP}" >&2; exit 1; }
require_pattern 'a5vm\.vsts' "${FILLPAD_CPP}" \
  "Fillpad output lost predicated valid-copy store lowering"
if rg -n 'pto\.tfillpad' "${FILLPAD_CPP}" >/dev/null; then
  echo "error: Fillpad output still contains pto.tfillpad" >&2
  exit 1
fi
require_pattern 'a5vm\.vdup' "${FILLPAD_EXPAND_CPP}" \
  "Fillpad expand output lost pad-vector materialization"
require_pattern 'a5vm\.vsts' "${FILLPAD_EXPAND_CPP}" \
  "Fillpad expand output lost predicated fillpad store lowering"
if rg -n 'pto\.tfillpad_expand' "${FILLPAD_EXPAND_CPP}" >/dev/null; then
  echo "error: Fillpad expand output still contains pto.tfillpad_expand" >&2
  exit 1
fi

echo "sample acceptance: Rsqrt"
RSQRT_OUT="${OUT_DIR}/rsqrt"
rm -rf "${RSQRT_OUT}"
mkdir -p "${RSQRT_OUT}"
(
  cd "${ROOT_DIR}"
  SOC_VERSION="A5" \
  PTOAS_BIN="${PTOAS_BIN}" \
  PTOAS_OUT_DIR="${RSQRT_OUT}" \
  PTOAS_FLAGS="--pto-arch a5 --pto-backend=a5vm --a5vm-print-ir" \
  ./test/samples/runop.sh -t Rsqrt
) > "${RSQRT_OUT}/run.log" 2>&1
require_pattern 'Rsqrt\(rsqrt\.py\)[[:space:]]+OK' "${RSQRT_OUT}/run.log" \
  "Rsqrt sample did not compile successfully"
RSQRT_CPP="${RSQRT_OUT}/Rsqrt/rsqrt-pto.cpp"
[[ -f "${RSQRT_CPP}" ]] || { echo "error: missing ${RSQRT_CPP}" >&2; exit 1; }
require_pattern 'a5vm\.vdup' "${RSQRT_CPP}" \
  "Rsqrt output lost one-vector materialization"
require_pattern 'a5vm\.vsqrt' "${RSQRT_CPP}" \
  "Rsqrt output lost sqrt lowering"
require_pattern 'a5vm\.vdiv' "${RSQRT_CPP}" \
  "Rsqrt output lost reciprocal divide lowering"
if rg -n 'pto\.trsqrt' "${RSQRT_CPP}" >/dev/null; then
  echo "error: Rsqrt output still contains pto.trsqrt" >&2
  exit 1
fi

echo "sample acceptance: Cvt"
CVT_OUT="${OUT_DIR}/cvt"
rm -rf "${CVT_OUT}"
mkdir -p "${CVT_OUT}"
(
  cd "${ROOT_DIR}"
  SOC_VERSION="A5" \
  PTOAS_BIN="${PTOAS_BIN}" \
  PTOAS_OUT_DIR="${CVT_OUT}" \
  PTOAS_FLAGS="--pto-arch a5 --pto-backend=a5vm --a5vm-print-ir" \
  ./test/samples/runop.sh -t Cvt
) > "${CVT_OUT}/run.log" 2>&1
require_pattern 'Cvt\(cvt\.py\)[[:space:]]+OK' "${CVT_OUT}/run.log" \
  "Cvt sample did not compile successfully"
CVT_CPP="${CVT_OUT}/Cvt/cvt-pto.cpp"
[[ -f "${CVT_CPP}" ]] || { echo "error: missing ${CVT_CPP}" >&2; exit 1; }
require_pattern 'a5vm\.vtrc' "${CVT_CPP}" \
  "Cvt output lost PTO A5 vtrc lowering"
if rg -n 'pto\.tcvt' "${CVT_CPP}" >/dev/null; then
  echo "error: Cvt output still contains pto.tcvt" >&2
  exit 1
fi

echo "sample acceptance: Subs"
SUBS_OUT="${OUT_DIR}/subs"
rm -rf "${SUBS_OUT}"
mkdir -p "${SUBS_OUT}"
"${PYTHON_BIN}" "${ROOT_DIR}/test/samples/Subs/subs.py" > "${SUBS_OUT}/subs.pto"
if "${PTOAS_BIN}" --pto-arch a5 --pto-backend=a5vm --a5vm-print-ir \
    "${SUBS_OUT}/subs.pto" -o "${SUBS_OUT}/subs-pto.cpp" \
    > "${SUBS_OUT}/run.log" 2>&1; then
  echo "error: Subs sample unexpectedly compiled despite unresolved A5 baseline" >&2
  exit 1
fi
require_pattern 'tsubs lowering is intentionally unresolved until the installed A5 PTO helper baseline is located and traced' "${SUBS_OUT}/run.log" \
  "Subs failure did not explain the unresolved installed A5 baseline"

echo "sample acceptance: Maxs"
MAXS_OUT="${OUT_DIR}/maxs"
rm -rf "${MAXS_OUT}"
mkdir -p "${MAXS_OUT}"
(
  cd "${ROOT_DIR}"
  SOC_VERSION="A5" \
  PTOAS_BIN="${PTOAS_BIN}" \
  PTOAS_OUT_DIR="${MAXS_OUT}" \
  PTOAS_FLAGS="--pto-arch a5 --pto-backend=a5vm --a5vm-print-ir" \
  ./test/samples/runop.sh -t Maxs
) > "${MAXS_OUT}/run.log" 2>&1
require_pattern 'Maxs\(maxs\.py\)[[:space:]]+OK' "${MAXS_OUT}/run.log" \
  "Maxs sample did not compile successfully"
MAXS_CPP="${MAXS_OUT}/Maxs/maxs-pto.cpp"
[[ -f "${MAXS_CPP}" ]] || { echo "error: missing ${MAXS_CPP}" >&2; exit 1; }
require_pattern 'a5vm\.vmaxs' "${MAXS_CPP}" \
  "Maxs output lost PTO A5 scalar-max lowering"
if rg -n 'pto\.tmaxs' "${MAXS_CPP}" >/dev/null; then
  echo "error: Maxs output still contains pto.tmaxs" >&2
  exit 1
fi

echo "sample acceptance: Sels"
SELS_OUT="${OUT_DIR}/sels"
rm -rf "${SELS_OUT}"
mkdir -p "${SELS_OUT}"
(
  cd "${ROOT_DIR}"
  SOC_VERSION="A5" \
  PTOAS_BIN="${PTOAS_BIN}" \
  PTOAS_OUT_DIR="${SELS_OUT}" \
  PTOAS_FLAGS="--pto-arch a5 --pto-backend=a5vm --a5vm-print-ir" \
  ./test/samples/runop.sh -t Sels
) > "${SELS_OUT}/run.log" 2>&1
require_pattern 'Sels\(sels\.py\)[[:space:]]+OK' "${SELS_OUT}/run.log" \
  "Sels sample did not compile successfully"
SELS_CPP="${SELS_OUT}/Sels/sels-pto.cpp"
[[ -f "${SELS_CPP}" ]] || { echo "error: missing ${SELS_CPP}" >&2; exit 1; }
require_pattern 'a5vm\.pset_b8' "${SELS_CPP}" \
  "Sels output lost PTO A5 predicate-materialization lowering"
require_pattern 'a5vm\.vsel' "${SELS_CPP}" \
  "Sels output lost PTO A5 select lowering"
if rg -n 'pto\.tsels' "${SELS_CPP}" >/dev/null; then
  echo "error: Sels output still contains pto.tsels" >&2
  exit 1
fi

echo "sample acceptance: Cmps/Cmp"
CMP_OUT="${OUT_DIR}/cmp-family"
rm -rf "${CMP_OUT}"
mkdir -p "${CMP_OUT}"
(
  cd "${ROOT_DIR}"
  SOC_VERSION="A5" \
  PTOAS_BIN="${PTOAS_BIN}" \
  PTOAS_OUT_DIR="${CMP_OUT}" \
  PTOAS_FLAGS="--pto-arch a5 --pto-backend=a5vm --a5vm-print-ir" \
  ./test/samples/runop.sh -t Cmps
) > "${CMP_OUT}/cmps.log" 2>&1
(
  cd "${ROOT_DIR}"
  SOC_VERSION="A5" \
  PTOAS_BIN="${PTOAS_BIN}" \
  PTOAS_OUT_DIR="${CMP_OUT}" \
  PTOAS_FLAGS="--pto-arch a5 --pto-backend=a5vm --a5vm-print-ir" \
  ./test/samples/runop.sh -t Cmp
) > "${CMP_OUT}/cmp.log" 2>&1
require_pattern 'Cmps\(cmps\.py\)[[:space:]]+OK' "${CMP_OUT}/cmps.log" \
  "Cmps sample did not compile successfully"
require_pattern 'Cmp\(cmp\.py\)[[:space:]]+OK' "${CMP_OUT}/cmp.log" \
  "Cmp sample did not compile successfully"
CMPS_CPP="${CMP_OUT}/Cmps/cmps-pto.cpp"
CMP_CPP="${CMP_OUT}/Cmp/cmp-pto.cpp"
[[ -f "${CMPS_CPP}" ]] || { echo "error: missing ${CMPS_CPP}" >&2; exit 1; }
[[ -f "${CMP_CPP}" ]] || { echo "error: missing ${CMP_CPP}" >&2; exit 1; }
require_pattern 'a5vm\.vcmps' "${CMPS_CPP}" \
  "Cmps output lost vector-scalar compare lowering"
require_pattern 'a5vm\.pdintlv_b8' "${CMPS_CPP}" \
  "Cmps output lost predicate interleave lowering"
require_pattern 'a5vm\.psts' "${CMPS_CPP}" \
  "Cmps output lost predicate store lowering"
if rg -n 'pto\.tcmps' "${CMPS_CPP}" >/dev/null; then
  echo "error: Cmps output still contains pto.tcmps" >&2
  exit 1
fi
require_pattern 'a5vm\.vcmp' "${CMP_CPP}" \
  "Cmp output lost vector compare lowering"
require_pattern 'a5vm\.pdintlv_b8' "${CMP_CPP}" \
  "Cmp output lost predicate interleave lowering"
require_pattern 'a5vm\.psts' "${CMP_CPP}" \
  "Cmp output lost predicate store lowering"
if rg -n 'pto\.tcmp' "${CMP_CPP}" >/dev/null; then
  echo "error: Cmp output still contains pto.tcmp" >&2
  exit 1
fi

echo "sample acceptance: Sel"
SEL_OUT="${OUT_DIR}/sel"
rm -rf "${SEL_OUT}"
mkdir -p "${SEL_OUT}"
(
  cd "${ROOT_DIR}"
  SOC_VERSION="A5" \
  PTOAS_BIN="${PTOAS_BIN}" \
  PTOAS_OUT_DIR="${SEL_OUT}" \
  PTOAS_FLAGS="--pto-arch a5 --pto-backend=a5vm --a5vm-print-ir" \
  ./test/samples/runop.sh -t Sel
) > "${SEL_OUT}/run.log" 2>&1
require_pattern 'Sel\(sel\.py\)[[:space:]]+OK' "${SEL_OUT}/run.log" \
  "Sel sample did not compile successfully"
require_pattern 'Sel\(sel_head\.py\)[[:space:]]+OK' "${SEL_OUT}/run.log" \
  "Sel head sample did not compile successfully"
SEL_CPP="${SEL_OUT}/Sel/sel-pto.cpp"
SEL_HEAD_CPP="${SEL_OUT}/Sel/sel_head-pto.cpp"
[[ -f "${SEL_CPP}" ]] || { echo "error: missing ${SEL_CPP}" >&2; exit 1; }
[[ -f "${SEL_HEAD_CPP}" ]] || { echo "error: missing ${SEL_HEAD_CPP}" >&2; exit 1; }
require_pattern 'a5vm\.plds' "${SEL_CPP}" \
  "Sel output lost predicate-load lowering"
require_pattern 'a5vm\.punpack' "${SEL_CPP}" \
  "Sel output lost predicate-unpack lowering"
require_pattern 'a5vm\.vsel' "${SEL_CPP}" \
  "Sel output lost vector select lowering"
require_pattern 'a5vm\.vsts' "${SEL_CPP}" \
  "Sel output lost predicated vector store lowering"
if rg -n 'pto\.tsel' "${SEL_CPP}" >/dev/null; then
  echo "error: Sel output still contains pto.tsel" >&2
  exit 1
fi
require_pattern 'a5vm\.pset_b16' "${SEL_HEAD_CPP}" \
  "Sel head output lost full-mask materialization lowering"
require_pattern 'a5vm\.pintlv_b16' "${SEL_HEAD_CPP}" \
  "Sel head output lost predicate interleave lowering"
require_pattern 'a5vm\.vsts' "${SEL_HEAD_CPP}" \
  "Sel head output lost predicated vector store lowering"
if rg -n 'pto\.tsel' "${SEL_HEAD_CPP}" >/dev/null; then
  echo "error: Sel head output still contains pto.tsel" >&2
  exit 1
fi

echo "sample acceptance: Sync/add_double_dynamic"
SYNC_OUT="${OUT_DIR}/sync"
rm -rf "${SYNC_OUT}"
mkdir -p "${SYNC_OUT}"
(
  cd "${ROOT_DIR}"
  SOC_VERSION="A5" \
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

require_pattern 'Sync\(test_dynamic_valid_shape\.py\)[[:space:]]+OK' "${SYNC_OUT}/run.log" \
  "Sync/test_dynamic_valid_shape sample did not compile successfully"
require_pattern 'Sync\(test_dynamic_valid_shape\.pto\)[[:space:]]+OK' "${SYNC_OUT}/run.log" \
  "Sync/test_dynamic_valid_shape.pto sample did not compile successfully"
DYNAMIC_VALID_OUT="${SYNC_OUT}/Sync/test_dynamic_valid_shape-pto.cpp"
[[ -f "${DYNAMIC_VALID_OUT}" ]] || { echo "error: missing ${DYNAMIC_VALID_OUT}" >&2; exit 1; }
require_pattern 'a5vm\.vrelu' "${DYNAMIC_VALID_OUT}" \
  "dynamic valid-shape sample lost relu lowering"
require_pattern 'a5vm\.vsts' "${DYNAMIC_VALID_OUT}" \
  "dynamic valid-shape sample lost tail-aware store lowering"
require_pattern 'llvm\.loop\.aivector_scope' "${DYNAMIC_VALID_OUT}" \
  "dynamic valid-shape sample lost vec-scope loop carrier"
if rg -n 'pto\.trelu' "${DYNAMIC_VALID_OUT}" >/dev/null; then
  echo "error: dynamic valid-shape output still contains pto.trelu" >&2
  exit 1
fi

require_pattern 'Sync\(test_a5_buf_sync\.py\)[[:space:]]+OK' "${SYNC_OUT}/run.log" \
  "Sync/test_a5_buf_sync sample did not compile successfully"
A5_BUF_SYNC_OUT="${SYNC_OUT}/Sync/test_a5_buf_sync.cpp"
[[ -f "${A5_BUF_SYNC_OUT}" ]] || { echo "error: missing ${A5_BUF_SYNC_OUT}" >&2; exit 1; }
require_pattern 'a5vm\.get_buf "PIPE_MTE2"' "${A5_BUF_SYNC_OUT}" \
  "A5 buffer sync output lost get_buf lowering for TLOAD"
require_pattern 'a5vm\.rls_buf "PIPE_MTE2"' "${A5_BUF_SYNC_OUT}" \
  "A5 buffer sync output lost rls_buf lowering for TLOAD"
require_pattern 'a5vm\.get_buf "PIPE_V"' "${A5_BUF_SYNC_OUT}" \
  "A5 buffer sync output lost get_buf lowering for TVEC"
require_pattern 'a5vm\.rls_buf "PIPE_V"' "${A5_BUF_SYNC_OUT}" \
  "A5 buffer sync output lost rls_buf lowering for TVEC"

echo "sample acceptance: Layout DN"
LAYOUT_DN_OUT="${OUT_DIR}/layout-dn"
rm -rf "${LAYOUT_DN_OUT}"
mkdir -p "${LAYOUT_DN_OUT}"
(
  cd "${ROOT_DIR}"
  SOC_VERSION="A5" \
  PTOAS_BIN="${PTOAS_BIN}" \
  PTOAS_OUT_DIR="${LAYOUT_DN_OUT}" \
  PTOAS_FLAGS="--pto-arch a5 --pto-backend=a5vm --a5vm-print-ir" \
  ./test/samples/runop.sh -t Layout
) > "${LAYOUT_DN_OUT}/run.log" 2>&1 || true
require_pattern 'Layout\(tensor_view_layout_dn\.py\)[[:space:]]+OK' "${LAYOUT_DN_OUT}/run.log" \
  "Layout/tensor_view_layout_dn sample did not compile successfully"
require_pattern 'Layout\(tensor_view_infer_layout_dn\.py\)[[:space:]]+OK' "${LAYOUT_DN_OUT}/run.log" \
  "Layout/tensor_view_infer_layout_dn sample did not compile successfully"
LAYOUT_DN_CPP="${LAYOUT_DN_OUT}/Layout/tensor_view_layout_dn-pto.cpp"
[[ -f "${LAYOUT_DN_CPP}" ]] || { echo "error: missing ${LAYOUT_DN_CPP}" >&2; exit 1; }
require_pattern 'a5vm\.copy_gm_to_ubuf.*"dn"' "${LAYOUT_DN_CPP}" \
  "Layout DN output lost DN TLOAD lowering"
require_pattern 'a5vm\.copy_ubuf_to_gm.*"dn"' "${LAYOUT_DN_CPP}" \
  "Layout DN output lost DN TSTORE lowering"
LAYOUT_DN_INFER_CPP="${LAYOUT_DN_OUT}/Layout/tensor_view_infer_layout_dn-pto.cpp"
[[ -f "${LAYOUT_DN_INFER_CPP}" ]] || { echo "error: missing ${LAYOUT_DN_INFER_CPP}" >&2; exit 1; }
require_pattern 'a5vm\.copy_gm_to_ubuf.*"dn"' "${LAYOUT_DN_INFER_CPP}" \
  "Layout infer-DN output did not infer DN TLOAD lowering"
require_pattern 'a5vm\.copy_ubuf_to_gm.*"dn"' "${LAYOUT_DN_INFER_CPP}" \
  "Layout infer-DN output did not infer DN TSTORE lowering"

echo "sample acceptance: Reshape"
RESHAPE_OUT="${OUT_DIR}/reshape"
rm -rf "${RESHAPE_OUT}"
mkdir -p "${RESHAPE_OUT}"
(
  cd "${ROOT_DIR}"
  SOC_VERSION="A5" \
  PTOAS_BIN="${PTOAS_BIN}" \
  PTOAS_OUT_DIR="${RESHAPE_OUT}" \
  PTOAS_FLAGS="--pto-arch a5 --pto-backend=a5vm --a5vm-print-ir" \
  ./test/samples/runop.sh -t Reshape
) > "${RESHAPE_OUT}/run.log" 2>&1 || true
require_pattern 'Reshape\(reshape\.py\)[[:space:]]+OK' "${RESHAPE_OUT}/run.log" \
  "Reshape/reshape sample did not compile successfully"
RESHAPE_CPP="${RESHAPE_OUT}/Reshape/reshape-pto.cpp"
[[ -f "${RESHAPE_CPP}" ]] || { echo "error: missing ${RESHAPE_CPP}" >&2; exit 1; }
require_pattern 'a5vm\.copy_gm_to_ubuf.*"nd"' "${RESHAPE_CPP}" \
  "Reshape output lost ND TLOAD lowering"
require_pattern 'a5vm\.copy_ubuf_to_gm.*"dn"' "${RESHAPE_CPP}" \
  "Reshape output lost DN TSTORE lowering after treshape"

echo "sample acceptance: Expands"
EXPANDS_OUT="${OUT_DIR}/expands"
rm -rf "${EXPANDS_OUT}"
mkdir -p "${EXPANDS_OUT}"
(
  cd "${ROOT_DIR}"
  SOC_VERSION="A5" \
  PTOAS_BIN="${PTOAS_BIN}" \
  PTOAS_OUT_DIR="${EXPANDS_OUT}" \
  PTOAS_FLAGS="--pto-arch a5 --pto-backend=a5vm --a5vm-print-ir" \
  ./test/samples/runop.sh -t Expands
) > "${EXPANDS_OUT}/run.log" 2>&1
require_pattern 'Expands\(expand\.py\)[[:space:]]+OK' "${EXPANDS_OUT}/run.log" \
  "Expands/expand sample did not compile successfully"
require_pattern 'Expands\(expands\.py\)[[:space:]]+OK' "${EXPANDS_OUT}/run.log" \
  "Expands/expands sample did not compile successfully"
EXPAND_CPP="${EXPANDS_OUT}/Expands/expand-pto.cpp"
EXPANDS_CPP="${EXPANDS_OUT}/Expands/expands-pto.cpp"
[[ -f "${EXPAND_CPP}" ]] || { echo "error: missing ${EXPAND_CPP}" >&2; exit 1; }
[[ -f "${EXPANDS_CPP}" ]] || { echo "error: missing ${EXPANDS_CPP}" >&2; exit 1; }
require_pattern 'a5vm\.vdup' "${EXPAND_CPP}" \
  "Expands output lost scalar broadcast lowering"
if rg -n 'pto\.texpands' "${EXPAND_CPP}" >/dev/null; then
  echo "error: Expands output still contains pto.texpands" >&2
  exit 1
fi
if rg -n 'pto\.texpands' "${EXPANDS_CPP}" >/dev/null; then
  echo "error: Expands output still contains pto.texpands" >&2
  exit 1
fi

echo "sample acceptance: Expand family"
EXPAND_FAMILY_OUT="${OUT_DIR}/expand-family"
rm -rf "${EXPAND_FAMILY_OUT}"
mkdir -p "${EXPAND_FAMILY_OUT}"
(
  cd "${ROOT_DIR}"
  SOC_VERSION="A5" \
  PTOAS_BIN="${PTOAS_BIN}" \
  PTOAS_OUT_DIR="${EXPAND_FAMILY_OUT}" \
  PTOAS_FLAGS="--pto-arch a5 --pto-backend=a5vm --a5vm-print-ir" \
  ./test/samples/runop.sh -t Rowexpand
) > "${EXPAND_FAMILY_OUT}/rowexpand.log" 2>&1
(
  cd "${ROOT_DIR}"
  SOC_VERSION="A5" \
  PTOAS_BIN="${PTOAS_BIN}" \
  PTOAS_OUT_DIR="${EXPAND_FAMILY_OUT}" \
  PTOAS_FLAGS="--pto-arch a5 --pto-backend=a5vm --a5vm-print-ir" \
  ./test/samples/runop.sh -t Colexpand
) > "${EXPAND_FAMILY_OUT}/colexpand.log" 2>&1
require_pattern 'Rowexpand\(rowexpand\.py\)[[:space:]]+OK' "${EXPAND_FAMILY_OUT}/rowexpand.log" \
  "Rowexpand sample did not compile successfully"
require_pattern 'Colexpand\(colexpand\.py\)[[:space:]]+OK' "${EXPAND_FAMILY_OUT}/colexpand.log" \
  "Colexpand sample did not compile successfully"
ROWEXPAND_CPP="${EXPAND_FAMILY_OUT}/Rowexpand/rowexpand-pto.cpp"
COLEXPAND_CPP="${EXPAND_FAMILY_OUT}/Colexpand/colexpand-pto.cpp"
[[ -f "${ROWEXPAND_CPP}" ]] || { echo "error: missing ${ROWEXPAND_CPP}" >&2; exit 1; }
[[ -f "${COLEXPAND_CPP}" ]] || { echo "error: missing ${COLEXPAND_CPP}" >&2; exit 1; }
require_pattern 'a5vm\.vdup' "${ROWEXPAND_CPP}" \
  "Rowexpand output lost vector broadcast lowering"
require_pattern 'llvm\.loop\.aivector_scope' "${ROWEXPAND_CPP}" \
  "Rowexpand output lost vec-scope loop carrier"
if rg -n 'pto\.trowexpand' "${ROWEXPAND_CPP}" >/dev/null; then
  echo "error: Rowexpand output still contains pto.trowexpand" >&2
  exit 1
fi
require_pattern 'a5vm\.vlds' "${COLEXPAND_CPP}" \
  "Colexpand output lost source row load lowering"
require_pattern 'a5vm\.vsts' "${COLEXPAND_CPP}" \
  "Colexpand output lost broadcast store lowering"
if rg -n 'pto\.tcolexpand' "${COLEXPAND_CPP}" >/dev/null; then
  echo "error: Colexpand output still contains pto.tcolexpand" >&2
  exit 1
fi

echo "sample acceptance: Muls"
MULS_OUT="${OUT_DIR}/muls"
rm -rf "${MULS_OUT}"
mkdir -p "${MULS_OUT}"
(
  cd "${ROOT_DIR}"
  SOC_VERSION="A5" \
  PTOAS_BIN="${PTOAS_BIN}" \
  PTOAS_OUT_DIR="${MULS_OUT}" \
  PTOAS_FLAGS="--pto-arch a5 --pto-backend=a5vm --a5vm-print-ir" \
  ./test/samples/runop.sh -t Muls
) > "${MULS_OUT}/run.log" 2>&1
require_pattern 'Muls\(muls\.py\)[[:space:]]+OK' "${MULS_OUT}/run.log" \
  "Muls sample did not compile successfully"
MULS_CPP="${MULS_OUT}/Muls/muls-pto.cpp"
[[ -f "${MULS_CPP}" ]] || { echo "error: missing ${MULS_CPP}" >&2; exit 1; }
require_pattern 'a5vm\.vmuls' "${MULS_CPP}" \
  "Muls output lost TMulS vector-scalar lowering"
if rg -n 'pto\.tmuls' "${MULS_CPP}" >/dev/null; then
  echo "error: Muls output still contains pto.tmuls" >&2
  exit 1
fi

echo "sample acceptance: PyPTOIRParser softmax prepare"
PARSER_OUT="${OUT_DIR}/py-parser"
rm -rf "${PARSER_OUT}"
mkdir -p "${PARSER_OUT}"
(
  cd "${ROOT_DIR}"
  SOC_VERSION="A5" \
  PTOAS_BIN="${PTOAS_BIN}" \
  PTOAS_OUT_DIR="${PARSER_OUT}" \
  PTOAS_FLAGS="--pto-arch a5 --pto-backend=a5vm --a5vm-print-ir" \
  ./test/samples/runop.sh -t PyPTOIRParser
) > "${PARSER_OUT}/run.log" 2>&1 || true
require_pattern 'PyPTOIRParser\(paged_attention_example_kernel_softmax_prepare\.py\)[[:space:]]+OK' "${PARSER_OUT}/run.log" \
  "PyPTOIRParser softmax prepare sample did not compile successfully"
PARSER_SOFTMAX_CPP="${PARSER_OUT}/PyPTOIRParser/paged_attention_example_kernel_softmax_prepare-pto.cpp"
[[ -f "${PARSER_SOFTMAX_CPP}" ]] || { echo "error: missing ${PARSER_SOFTMAX_CPP}" >&2; exit 1; }
require_pattern 'a5vm\.vmuls' "${PARSER_SOFTMAX_CPP}" \
  "PyPTOIRParser softmax prepare output lost TMulS vector-scalar lowering"
if rg -n 'pto\.tmuls' "${PARSER_SOFTMAX_CPP}" >/dev/null; then
  echo "error: PyPTOIRParser softmax prepare output still contains pto.tmuls" >&2
  exit 1
fi
require_pattern 'a5vm\.vcvt' "${PARSER_SOFTMAX_CPP}" \
  "PyPTOIRParser softmax prepare output lost TCVT lowering"
if rg -n 'pto\.tcvt' "${PARSER_SOFTMAX_CPP}" >/dev/null; then
  echo "error: PyPTOIRParser softmax prepare output still contains pto.tcvt" >&2
  exit 1
fi
require_pattern 'a5vm\.vldas' "${PARSER_SOFTMAX_CPP}" \
  "PyPTOIRParser softmax prepare output lost TROWEXPANDSUB align-load lowering"
require_pattern 'a5vm\.vldus' "${PARSER_SOFTMAX_CPP}" \
  "PyPTOIRParser softmax prepare output lost TROWEXPANDSUB unaligned-load lowering"
require_pattern 'a5vm\.vsub' "${PARSER_SOFTMAX_CPP}" \
  "PyPTOIRParser softmax prepare output lost TROWEXPANDSUB subtract lowering"
if rg -n 'pto\.trowexpandsub' "${PARSER_SOFTMAX_CPP}" >/dev/null; then
  echo "error: PyPTOIRParser softmax prepare output still contains pto.trowexpandsub" >&2
  exit 1
fi

echo "sample acceptance: PyPTOIRParser online update"
require_pattern 'PyPTOIRParser\(paged_attention_example_kernel_online_update\.py\)[[:space:]]+OK' "${PARSER_OUT}/run.log" \
  "PyPTOIRParser online update sample did not compile successfully"
PARSER_ONLINE_UPDATE_CPP="${PARSER_OUT}/PyPTOIRParser/paged_attention_example_kernel_online_update-pto.cpp"
[[ -f "${PARSER_ONLINE_UPDATE_CPP}" ]] || { echo "error: missing ${PARSER_ONLINE_UPDATE_CPP}" >&2; exit 1; }
require_pattern 'a5vm\.copy_gm_to_ubuf' "${PARSER_ONLINE_UPDATE_CPP}" \
  "PyPTOIRParser online update output lost TLOAD copy lowering"
require_pattern 'a5vm\.copy_ubuf_to_gm' "${PARSER_ONLINE_UPDATE_CPP}" \
  "PyPTOIRParser online update output lost TSTORE copy lowering"
require_pattern 'a5vm\.vmax' "${PARSER_ONLINE_UPDATE_CPP}" \
  "PyPTOIRParser online update output lost TMax lowering"
require_pattern 'a5vm\.vsub' "${PARSER_ONLINE_UPDATE_CPP}" \
  "PyPTOIRParser online update output lost TSub / TRowExpandSub lowering"
require_pattern 'a5vm\.vexp' "${PARSER_ONLINE_UPDATE_CPP}" \
  "PyPTOIRParser online update output lost TExp lowering"
require_pattern 'a5vm\.vmul' "${PARSER_ONLINE_UPDATE_CPP}" \
  "PyPTOIRParser online update output lost TMul / TRowExpandMul lowering"
require_pattern 'a5vm\.vadd' "${PARSER_ONLINE_UPDATE_CPP}" \
  "PyPTOIRParser online update output lost TAdd lowering"
require_pattern 'a5vm\.vdiv' "${PARSER_ONLINE_UPDATE_CPP}" \
  "PyPTOIRParser online update output lost TRowExpandDiv lowering"
require_pattern 'a5vm\.vsts' "${PARSER_ONLINE_UPDATE_CPP}" \
  "PyPTOIRParser online update output lost tail store lowering"
require_pattern 'llvm\.loop\.aivector_scope' "${PARSER_ONLINE_UPDATE_CPP}" \
  "PyPTOIRParser online update output lost vec-scope loop carrier"
if rg -n 'pto\.(pointer_cast|bind_tile|trowexpandmul|trowexpanddiv|tmax|tsub|texp|tmul|tadd)' "${PARSER_ONLINE_UPDATE_CPP}" >/dev/null; then
  echo "error: PyPTOIRParser online update output still contains residual PTO ops/scaffold" >&2
  exit 1
fi

echo "sample acceptance: Adds"
ADDS_OUT="${OUT_DIR}/adds"
rm -rf "${ADDS_OUT}"
mkdir -p "${ADDS_OUT}"
(
  cd "${ROOT_DIR}"
  SOC_VERSION="A5" \
  PTOAS_BIN="${PTOAS_BIN}" \
  PTOAS_OUT_DIR="${ADDS_OUT}" \
  PTOAS_FLAGS="--pto-arch a5 --pto-backend=a5vm --a5vm-print-ir" \
  ./test/samples/runop.sh -t Adds
) > "${ADDS_OUT}/run.log" 2>&1
require_pattern 'Adds\(adds\.py\)[[:space:]]+OK' "${ADDS_OUT}/run.log" \
  "Adds sample did not compile successfully"
ADDS_CPP="${ADDS_OUT}/Adds/adds-pto.cpp"
[[ -f "${ADDS_CPP}" ]] || { echo "error: missing ${ADDS_CPP}" >&2; exit 1; }
require_pattern 'a5vm\.vadds' "${ADDS_CPP}" \
  "Adds output lost TAddS vector-scalar lowering"
require_pattern 'llvm\.loop\.aivector_scope' "${ADDS_CPP}" \
  "Adds output lost vec-scope loop carrier"
if rg -n 'pto\.tadds' "${ADDS_CPP}" >/dev/null; then
  echo "error: Adds output still contains pto.tadds" >&2
  exit 1
fi

echo "sample acceptance: DataMovement"
DATAMOVEMENT_OUT="${OUT_DIR}/datamovement"
rm -rf "${DATAMOVEMENT_OUT}"
mkdir -p "${DATAMOVEMENT_OUT}"
(
  cd "${ROOT_DIR}"
  SOC_VERSION="A5" \
  PTOAS_BIN="${PTOAS_BIN}" \
  PTOAS_OUT_DIR="${DATAMOVEMENT_OUT}" \
  PTOAS_FLAGS="--pto-arch a5 --pto-backend=a5vm --a5vm-print-ir" \
  ./test/samples/runop.sh -t DataMovement
) > "${DATAMOVEMENT_OUT}/run.log" 2>&1
require_pattern 'DataMovement\(dataMovement\.py\)[[:space:]]+OK' "${DATAMOVEMENT_OUT}/run.log" \
  "DataMovement sample did not compile successfully"
DATAMOVEMENT_CPP="${DATAMOVEMENT_OUT}/DataMovement/dataMovement-pto.cpp"
[[ -f "${DATAMOVEMENT_CPP}" ]] || { echo "error: missing ${DATAMOVEMENT_CPP}" >&2; exit 1; }
require_pattern 'a5vm\.vadds' "${DATAMOVEMENT_CPP}" \
  "DataMovement output lost TAddS vector-scalar lowering"
require_pattern 'llvm\.loop\.aivector_scope' "${DATAMOVEMENT_CPP}" \
  "DataMovement output lost vec-scope loop carrier"
if rg -n 'pto\.tadds' "${DATAMOVEMENT_CPP}" >/dev/null; then
  echo "error: DataMovement output still contains pto.tadds" >&2
  exit 1
fi

echo "sample acceptance: Mins"
MINS_OUT="${OUT_DIR}/mins"
rm -rf "${MINS_OUT}"
mkdir -p "${MINS_OUT}"
(
  cd "${ROOT_DIR}"
  SOC_VERSION="A5" \
  PTOAS_BIN="${PTOAS_BIN}" \
  PTOAS_OUT_DIR="${MINS_OUT}" \
  PTOAS_FLAGS="--pto-arch a5 --pto-backend=a5vm --a5vm-print-ir" \
  ./test/samples/runop.sh -t Mins
) > "${MINS_OUT}/run.log" 2>&1
require_pattern 'Mins\(mins\.py\)[[:space:]]+OK' "${MINS_OUT}/run.log" \
  "Mins sample did not compile successfully"
MINS_CPP="${MINS_OUT}/Mins/mins-pto.cpp"
[[ -f "${MINS_CPP}" ]] || { echo "error: missing ${MINS_CPP}" >&2; exit 1; }
require_pattern 'a5vm\.vmins' "${MINS_CPP}" \
  "Mins output lost TMinS vector-scalar lowering"
require_pattern 'llvm\.loop\.aivector_scope' "${MINS_CPP}" \
  "Mins output lost vec-scope loop carrier"
if rg -n 'pto\.tmins' "${MINS_CPP}" >/dev/null; then
  echo "error: Mins output still contains pto.tmins" >&2
  exit 1
fi

echo "sample acceptance: Max/Min"
MAXMIN_OUT="${OUT_DIR}/max-min"
rm -rf "${MAXMIN_OUT}"
mkdir -p "${MAXMIN_OUT}"
(
  cd "${ROOT_DIR}"
  SOC_VERSION="A5" \
  PTOAS_BIN="${PTOAS_BIN}" \
  PTOAS_OUT_DIR="${MAXMIN_OUT}" \
  PTOAS_FLAGS="--pto-arch a5 --pto-backend=a5vm --a5vm-print-ir" \
  ./test/samples/runop.sh -t Max
) > "${MAXMIN_OUT}/max.log" 2>&1
(
  cd "${ROOT_DIR}"
  SOC_VERSION="A5" \
  PTOAS_BIN="${PTOAS_BIN}" \
  PTOAS_OUT_DIR="${MAXMIN_OUT}" \
  PTOAS_FLAGS="--pto-arch a5 --pto-backend=a5vm --a5vm-print-ir" \
  ./test/samples/runop.sh -t Min
) > "${MAXMIN_OUT}/min.log" 2>&1
require_pattern 'Max\(max\.py\)[[:space:]]+OK' "${MAXMIN_OUT}/max.log" \
  "Max sample did not compile successfully"
require_pattern 'Min\(min\.py\)[[:space:]]+OK' "${MAXMIN_OUT}/min.log" \
  "Min sample did not compile successfully"
MAX_CPP="${MAXMIN_OUT}/Max/max-pto.cpp"
MIN_CPP="${MAXMIN_OUT}/Min/min-pto.cpp"
[[ -f "${MAX_CPP}" ]] || { echo "error: missing ${MAX_CPP}" >&2; exit 1; }
[[ -f "${MIN_CPP}" ]] || { echo "error: missing ${MIN_CPP}" >&2; exit 1; }
require_pattern 'a5vm\.vmax' "${MAX_CPP}" \
  "Max output lost TMax vector lowering"
require_pattern 'llvm\.loop\.aivector_scope' "${MAX_CPP}" \
  "Max output lost vec-scope loop carrier"
if rg -n 'pto\.tmax' "${MAX_CPP}" >/dev/null; then
  echo "error: Max output still contains pto.tmax" >&2
  exit 1
fi
require_pattern 'a5vm\.vmin' "${MIN_CPP}" \
  "Min output lost TMin vector lowering"
require_pattern 'llvm\.loop\.aivector_scope' "${MIN_CPP}" \
  "Min output lost vec-scope loop carrier"
if rg -n 'pto\.tmin' "${MIN_CPP}" >/dev/null; then
  echo "error: Min output still contains pto.tmin" >&2
  exit 1
fi

echo "sample acceptance: Ci"
CI_OUT="${OUT_DIR}/ci"
rm -rf "${CI_OUT}"
mkdir -p "${CI_OUT}"
(
  cd "${ROOT_DIR}"
  SOC_VERSION="A5" \
  PTOAS_BIN="${PTOAS_BIN}" \
  PTOAS_OUT_DIR="${CI_OUT}" \
  PTOAS_FLAGS="--pto-arch a5 --pto-backend=a5vm --a5vm-print-ir" \
  ./test/samples/runop.sh -t Ci
) > "${CI_OUT}/run.log" 2>&1
require_pattern 'Ci\(ci\.py\)[[:space:]]+OK' "${CI_OUT}/run.log" \
  "Ci sample did not compile successfully"
CI_CPP="${CI_OUT}/Ci/ci-pto.cpp"
[[ -f "${CI_CPP}" ]] || { echo "error: missing ${CI_CPP}" >&2; exit 1; }
require_pattern 'scf\.for' "${CI_CPP}" \
  "Ci output lost software loop lowering"
require_pattern 'llvm\.store' "${CI_CPP}" \
  "Ci output lost llvm store lowering"
if rg -n 'pto\.tci' "${CI_CPP}" >/dev/null; then
  echo "error: Ci output still contains pto.tci" >&2
  exit 1
fi

echo "sample acceptance: Divs"
DIVS_OUT="${OUT_DIR}/divs"
rm -rf "${DIVS_OUT}"
mkdir -p "${DIVS_OUT}"
(
  cd "${ROOT_DIR}"
  SOC_VERSION="A5" \
  PTOAS_BIN="${PTOAS_BIN}" \
  PTOAS_OUT_DIR="${DIVS_OUT}" \
  PTOAS_FLAGS="--pto-arch a5 --pto-backend=a5vm --a5vm-print-ir" \
  ./test/samples/runop.sh -t Divs
) > "${DIVS_OUT}/run.log" 2>&1
require_pattern 'Divs\(divs\.py\)[[:space:]]+OK' "${DIVS_OUT}/run.log" \
  "Divs sample did not compile successfully"
DIVS_CPP="${DIVS_OUT}/Divs/divs-pto.cpp"
[[ -f "${DIVS_CPP}" ]] || { echo "error: missing ${DIVS_CPP}" >&2; exit 1; }
require_pattern 'a5vm\.vmuls' "${DIVS_CPP}" \
  "Divs output lost PTO A5 reciprocal-times-muls lowering"
require_pattern 'llvm\.loop\.aivector_scope' "${DIVS_CPP}" \
  "Divs output lost vec-scope loop carrier"
if rg -n 'pto\.tdivs' "${DIVS_CPP}" >/dev/null; then
  echo "error: Divs output still contains pto.tdivs" >&2
  exit 1
fi

echo "sample acceptance: Divs2"
DIVS2_OUT="${OUT_DIR}/divs2"
rm -rf "${DIVS2_OUT}"
mkdir -p "${DIVS2_OUT}"
(
  cd "${ROOT_DIR}"
  SOC_VERSION="A5" \
  PTOAS_BIN="${PTOAS_BIN}" \
  PTOAS_OUT_DIR="${DIVS2_OUT}" \
  PTOAS_FLAGS="--pto-arch a5 --pto-backend=a5vm --a5vm-print-ir" \
  ./test/samples/runop.sh -t Divs2
) > "${DIVS2_OUT}/run.log" 2>&1
require_pattern 'Divs2\(divs2\.py\)[[:space:]]+OK' "${DIVS2_OUT}/run.log" \
  "Divs2 sample did not compile successfully"
DIVS2_CPP="${DIVS2_OUT}/Divs2/divs2-pto.cpp"
[[ -f "${DIVS2_CPP}" ]] || { echo "error: missing ${DIVS2_CPP}" >&2; exit 1; }
require_pattern 'a5vm\.vmuls' "${DIVS2_CPP}" \
  "Divs2 output lost PTO A5 reciprocal-times-muls lowering"
require_pattern 'llvm\.loop\.aivector_scope' "${DIVS2_CPP}" \
  "Divs2 output lost vec-scope loop carrier"
if rg -n 'pto\.tdivs' "${DIVS2_CPP}" >/dev/null; then
  echo "error: Divs2 output still contains pto.tdivs" >&2
  exit 1
fi

echo "sample acceptance: Row reductions"
ROW_REDUCE_OUT="${OUT_DIR}/row-reductions"
rm -rf "${ROW_REDUCE_OUT}"
mkdir -p "${ROW_REDUCE_OUT}"
(
  cd "${ROOT_DIR}"
  SOC_VERSION="A5" \
  PTOAS_BIN="${PTOAS_BIN}" \
  PTOAS_OUT_DIR="${ROW_REDUCE_OUT}" \
  PTOAS_FLAGS="--pto-arch a5 --pto-backend=a5vm --a5vm-print-ir" \
  ./test/samples/runop.sh -t Rowmax
) > "${ROW_REDUCE_OUT}/rowmax.log" 2>&1
(
  cd "${ROOT_DIR}"
  SOC_VERSION="A5" \
  PTOAS_BIN="${PTOAS_BIN}" \
  PTOAS_OUT_DIR="${ROW_REDUCE_OUT}" \
  PTOAS_FLAGS="--pto-arch a5 --pto-backend=a5vm --a5vm-print-ir" \
  ./test/samples/runop.sh -t Rowmin
) > "${ROW_REDUCE_OUT}/rowmin.log" 2>&1
(
  cd "${ROOT_DIR}"
  SOC_VERSION="A5" \
  PTOAS_BIN="${PTOAS_BIN}" \
  PTOAS_OUT_DIR="${ROW_REDUCE_OUT}" \
  PTOAS_FLAGS="--pto-arch a5 --pto-backend=a5vm --a5vm-print-ir" \
  ./test/samples/runop.sh -t Rowsum
) > "${ROW_REDUCE_OUT}/rowsum.log" 2>&1
require_pattern 'Rowmax\(rowmax\.py\)[[:space:]]+OK' "${ROW_REDUCE_OUT}/rowmax.log" \
  "Rowmax sample did not compile successfully"
require_pattern 'Rowmin\(rowmin\.py\)[[:space:]]+OK' "${ROW_REDUCE_OUT}/rowmin.log" \
  "Rowmin sample did not compile successfully"
require_pattern 'Rowsum\(rowsum\.py\)[[:space:]]+OK' "${ROW_REDUCE_OUT}/rowsum.log" \
  "Rowsum sample did not compile successfully"

ROWMAX_CPP="${ROW_REDUCE_OUT}/Rowmax/rowmax-pto.cpp"
ROWMIN_CPP="${ROW_REDUCE_OUT}/Rowmin/rowmin-pto.cpp"
ROWSUM_CPP="${ROW_REDUCE_OUT}/Rowsum/rowsum-pto.cpp"
[[ -f "${ROWMAX_CPP}" ]] || { echo "error: missing ${ROWMAX_CPP}" >&2; exit 1; }
[[ -f "${ROWMIN_CPP}" ]] || { echo "error: missing ${ROWMIN_CPP}" >&2; exit 1; }
[[ -f "${ROWSUM_CPP}" ]] || { echo "error: missing ${ROWSUM_CPP}" >&2; exit 1; }

require_pattern 'a5vm\.vbr' "${ROWMAX_CPP}" \
  "Rowmax output lost vbr initialization"
require_pattern 'a5vm\.vcmax' "${ROWMAX_CPP}" \
  "Rowmax output lost vcmax reduction"
require_pattern 'a5vm\.vmax' "${ROWMAX_CPP}" \
  "Rowmax output lost vmax accumulation"
require_pattern 'llvm\.loop\.aivector_scope' "${ROWMAX_CPP}" \
  "Rowmax output lost vec-scope loop carrier"
if rg -n 'pto\.trowmax' "${ROWMAX_CPP}" >/dev/null; then
  echo "error: Rowmax output still contains pto.trowmax" >&2
  exit 1
fi

require_pattern 'a5vm\.vbr' "${ROWMIN_CPP}" \
  "Rowmin output lost vbr initialization"
require_pattern 'a5vm\.vcmin' "${ROWMIN_CPP}" \
  "Rowmin output lost vcmin reduction"
require_pattern 'a5vm\.vmin' "${ROWMIN_CPP}" \
  "Rowmin output lost vmin accumulation"
if rg -n 'pto\.trowmin' "${ROWMIN_CPP}" >/dev/null; then
  echo "error: Rowmin output still contains pto.trowmin" >&2
  exit 1
fi

require_pattern 'a5vm\.vbr' "${ROWSUM_CPP}" \
  "Rowsum output lost vbr initialization"
require_pattern 'a5vm\.vcadd' "${ROWSUM_CPP}" \
  "Rowsum output lost vcadd reduction"
require_pattern 'a5vm\.vadd' "${ROWSUM_CPP}" \
  "Rowsum output lost vadd accumulation"
if rg -n 'pto\.trowsum' "${ROWSUM_CPP}" >/dev/null; then
  echo "error: Rowsum output still contains pto.trowsum" >&2
  exit 1
fi

echo "sample acceptance: Col reductions"
COL_REDUCE_OUT="${OUT_DIR}/col-reductions"
rm -rf "${COL_REDUCE_OUT}"
mkdir -p "${COL_REDUCE_OUT}"
(
  cd "${ROOT_DIR}"
  SOC_VERSION="A5" \
  PTOAS_BIN="${PTOAS_BIN}" \
  PTOAS_OUT_DIR="${COL_REDUCE_OUT}" \
  PTOAS_FLAGS="--pto-arch a5 --pto-backend=a5vm --a5vm-print-ir" \
  ./test/samples/runop.sh -t Colmax
) > "${COL_REDUCE_OUT}/colmax.log" 2>&1
(
  cd "${ROOT_DIR}"
  SOC_VERSION="A5" \
  PTOAS_BIN="${PTOAS_BIN}" \
  PTOAS_OUT_DIR="${COL_REDUCE_OUT}" \
  PTOAS_FLAGS="--pto-arch a5 --pto-backend=a5vm --a5vm-print-ir" \
  ./test/samples/runop.sh -t Colmin
) > "${COL_REDUCE_OUT}/colmin.log" 2>&1
(
  cd "${ROOT_DIR}"
  SOC_VERSION="A5" \
  PTOAS_BIN="${PTOAS_BIN}" \
  PTOAS_OUT_DIR="${COL_REDUCE_OUT}" \
  PTOAS_FLAGS="--pto-arch a5 --pto-backend=a5vm --a5vm-print-ir" \
  ./test/samples/runop.sh -t Colsum
) > "${COL_REDUCE_OUT}/colsum.log" 2>&1
require_pattern 'Colmax\(colmax\.py\)[[:space:]]+OK' "${COL_REDUCE_OUT}/colmax.log" \
  "Colmax sample did not compile successfully"
require_pattern 'Colmin\(colmin\.py\)[[:space:]]+OK' "${COL_REDUCE_OUT}/colmin.log" \
  "Colmin sample did not compile successfully"
require_pattern 'Colsum\(colsum\.py\)[[:space:]]+OK' "${COL_REDUCE_OUT}/colsum.log" \
  "Colsum sample did not compile successfully"

COLMAX_CPP="${COL_REDUCE_OUT}/Colmax/colmax-pto.cpp"
COLMIN_CPP="${COL_REDUCE_OUT}/Colmin/colmin-pto.cpp"
COLSUM_CPP="${COL_REDUCE_OUT}/Colsum/colsum-pto.cpp"
[[ -f "${COLMAX_CPP}" ]] || { echo "error: missing ${COLMAX_CPP}" >&2; exit 1; }
[[ -f "${COLMIN_CPP}" ]] || { echo "error: missing ${COLMIN_CPP}" >&2; exit 1; }
[[ -f "${COLSUM_CPP}" ]] || { echo "error: missing ${COLSUM_CPP}" >&2; exit 1; }

require_pattern 'a5vm\.vmax' "${COLMAX_CPP}" \
  "Colmax output lost vmax accumulation"
require_pattern 'llvm\.loop\.aivector_scope' "${COLMAX_CPP}" \
  "Colmax output lost vec-scope loop carrier"
if rg -n 'pto\.tcolmax' "${COLMAX_CPP}" >/dev/null; then
  echo "error: Colmax output still contains pto.tcolmax" >&2
  exit 1
fi

require_pattern 'a5vm\.vmin' "${COLMIN_CPP}" \
  "Colmin output lost vmin accumulation"
if rg -n 'pto\.tcolmin' "${COLMIN_CPP}" >/dev/null; then
  echo "error: Colmin output still contains pto.tcolmin" >&2
  exit 1
fi

require_pattern 'a5vm\.vadd' "${COLSUM_CPP}" \
  "Colsum output lost vadd accumulation"
require_pattern 'a5vm\.vsts' "${COLSUM_CPP}" \
  "Colsum output lost tmp/dst vector stores"
if rg -n 'pto\.tcolsum' "${COLSUM_CPP}" >/dev/null; then
  echo "error: Colsum output still contains pto.tcolsum" >&2
  exit 1
fi

echo "sample acceptance: Part family"
PART_OUT="${OUT_DIR}/part-family"
rm -rf "${PART_OUT}"
mkdir -p "${PART_OUT}"
(
  cd "${ROOT_DIR}"
  SOC_VERSION="A5" \
  PTOAS_BIN="${PTOAS_BIN}" \
  PTOAS_OUT_DIR="${PART_OUT}" \
  PTOAS_FLAGS="--pto-arch a5 --pto-backend=a5vm --a5vm-print-ir" \
  ./test/samples/runop.sh -t Partadd
) > "${PART_OUT}/partadd.log" 2>&1
(
  cd "${ROOT_DIR}"
  SOC_VERSION="A5" \
  PTOAS_BIN="${PTOAS_BIN}" \
  PTOAS_OUT_DIR="${PART_OUT}" \
  PTOAS_FLAGS="--pto-arch a5 --pto-backend=a5vm --a5vm-print-ir" \
  ./test/samples/runop.sh -t Partmax
) > "${PART_OUT}/partmax.log" 2>&1
(
  cd "${ROOT_DIR}"
  SOC_VERSION="A5" \
  PTOAS_BIN="${PTOAS_BIN}" \
  PTOAS_OUT_DIR="${PART_OUT}" \
  PTOAS_FLAGS="--pto-arch a5 --pto-backend=a5vm --a5vm-print-ir" \
  ./test/samples/runop.sh -t Partmin
) > "${PART_OUT}/partmin.log" 2>&1
require_pattern 'Partadd\(partadd\.py\)[[:space:]]+OK' "${PART_OUT}/partadd.log" \
  "Partadd sample did not compile successfully"
require_pattern 'Partmax\(partmax\.py\)[[:space:]]+OK' "${PART_OUT}/partmax.log" \
  "Partmax sample did not compile successfully"
require_pattern 'Partmin\(partmin\.py\)[[:space:]]+OK' "${PART_OUT}/partmin.log" \
  "Partmin sample did not compile successfully"

PARTADD_CPP="${PART_OUT}/Partadd/partadd-pto.cpp"
PARTMAX_CPP="${PART_OUT}/Partmax/partmax-pto.cpp"
PARTMIN_CPP="${PART_OUT}/Partmin/partmin-pto.cpp"
[[ -f "${PARTADD_CPP}" ]] || { echo "error: missing ${PARTADD_CPP}" >&2; exit 1; }
[[ -f "${PARTMAX_CPP}" ]] || { echo "error: missing ${PARTMAX_CPP}" >&2; exit 1; }
[[ -f "${PARTMIN_CPP}" ]] || { echo "error: missing ${PARTMIN_CPP}" >&2; exit 1; }

require_pattern 'a5vm\.vadd' "${PARTADD_CPP}" \
  "Partadd output lost vadd lowering"
if rg -n 'pto\.tpartadd' "${PARTADD_CPP}" >/dev/null; then
  echo "error: Partadd output still contains pto.tpartadd" >&2
  exit 1
fi

require_pattern 'a5vm\.vbr' "${PARTMAX_CPP}" \
  "Partmax output lost pad initialization"
require_pattern 'a5vm\.vmax' "${PARTMAX_CPP}" \
  "Partmax output lost vmax lowering"
if rg -n 'pto\.tpartmax' "${PARTMAX_CPP}" >/dev/null; then
  echo "error: Partmax output still contains pto.tpartmax" >&2
  exit 1
fi

require_pattern 'a5vm\.vbr' "${PARTMIN_CPP}" \
  "Partmin output lost pad initialization"
require_pattern 'a5vm\.vmin' "${PARTMIN_CPP}" \
  "Partmin output lost vmin lowering"
if rg -n 'pto\.tpartmin' "${PARTMIN_CPP}" >/dev/null; then
  echo "error: Partmin output still contains pto.tpartmin" >&2
  exit 1
fi

if rg -n 'pto\.(pointer_cast|bind_tile)' "${OUT_DIR}" --glob '*-pto.cpp' >/dev/null; then
  echo "error: final A5VM backend output still contains pto.pointer_cast or pto.bind_tile" >&2
  exit 1
fi
echo "sample acceptance: PASS"
