#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
# A5 nightly board validation driver (GitCode CI).
#
# Runs the full A5 board path on a host that owns A5 (Ascend950) devices:
#   preflight -> PTOAS build -> sample payload -> pto-isa vendoring ->
#   device validation -> summary/report -> retention cleanup
#
# The script is deliberately self-contained so the same entry point works for
# the scheduled GitCode pipeline and for a manual/offline reproduction. It
# never assumes the source checkout lives on a large disk: the checkout is
# mirrored into a persistent CI root together with the LLVM cache and builds.
#
# Example:
#   .gitcode/scripts/a5_board_nightly.sh --ci-root /data/ptoas-ci --device 6
#
# The caller (workflow) only needs a checked-out source tree; everything else
# is derived or configurable through flags/environment.

set -euo pipefail

SCRIPT_NAME="$(basename "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck source=a5_notify.sh
source "${SCRIPT_DIR}/a5_notify.sh"

log() { printf '[%s] %s\n' "$(date +'%F %T')" "$*"; }
warn() { log "WARN: $*"; }
die() {
  log "ERROR: $*" >&2
  exit 1
}

host_arch() {
  case "$(uname -m)" in
    x86_64 | amd64) echo "x86_64" ;;
    aarch64 | arm64) echo "aarch64" ;;
    *) uname -m ;;
  esac
}

usage() {
  cat <<EOF
Usage: ${SCRIPT_NAME} [options]

Options:
  --workspace DIR        Source tree to validate (default: repository root)
  --ci-root DIR          Persistent CI root for LLVM/build/logs/payload
                         (default: \${PTOAS_CI_ROOT:-<repo>/.work/a5-board-ci})
  --device ID            Physical NPU id (default: auto-select a healthy idle A5)
  --soc NAME             Target SoC (default: Ascend950)
  --skip-cases LIST      Additional comma/space separated cases to skip
                         (appended to the built-in A5 defaults)
  --run-only LIST        Only run these cases (empty = all A5 cases)
  --scope SCOPE          Run scope: nightly (watchdog-visible) or manual
                         (default: manual; --prepare-only implies prepare)
  --night-window         Refuse to start between 06:00 and 22:00 local time;
                         the scheduled GitCode/cron entry points pass this
  --pto-isa-repo URL     pto-isa git URL (default: GitCode canonical repo)
  --pto-isa-commit SHA   pto-isa ref to pin; empty tracks the newest
                         pto-isa master (PTO_ISA_TRACK_LATEST=0 falls
                         back to the repository pin)
  --jobs N               Parallel build jobs (default: min(nproc, 64))
  --no-build             Reuse the existing PTOAS build in the CI root
  --no-sync              Build directly in --workspace instead of mirroring it
  --prepare-only         Prepare/verify the toolchain and exit (no validation)
  --no-report            Skip the summary/notification step
  --skip-lock            Caller already holds \${CI_ROOT}/locks/nightly.lock
  --wait-idle MIN        Wait up to MIN minutes for the device to be healthy
                         and idle before giving up (default: PTOAS_A5_WAIT_IDLE_MIN
                         or 30; 0 = fail immediately)
  --retention-days N     Delete whole run dirs older than N days (default: 7).
                         Heavy trees (npu-validation, payload) are pruned after
                         every run regardless; see PTOAS_A5_KEEP_HEAVY_RUNS.
  --help                 Show this help

Environment:
  PTOAS_CI_ROOT          Same as --ci-root
  PTOAS_A5_DEVICE_ID     Same as --device
  PTOAS_A5_WAIT_IDLE_MIN Same as --wait-idle
  PTOAS_A5_KEEP_HEAVY_RUNS
                         How many earlier runs keep their npu-validation and
                         payload trees (default: 0). Results, summaries and
                         logs are always kept; those two trees are ~17 GiB per
                         run and are regenerated on demand.
  A5_FEISHU_WEBHOOK_URL  Optional Feishu webhook for the result card and for
                         failure alerts (early failures included)
  A5_FEISHU_SECRET       Optional Feishu bot signing secret (签名校验)
  PTOAS_GIT_CREDENTIALS_FILE
                         Credential file for git. When set the file must exist;
                         when unset every git call runs with inherited
                         credential helpers disabled (anonymous access).
  PTOAS_CANN_CMAKE_DIR   Pre-populated cann-cmake tree (no network needed)
  PTOAS_LLVM_PATCH_FILE  Pre-downloaded feature-vpto-last3.patch
  PTOAS_RUNOP_JOBS       Parallel sample generation jobs (default: 1)
EOF
}

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
WORKSPACE="${REPO_ROOT}"
CI_ROOT="${PTOAS_CI_ROOT:-${REPO_ROOT}/.work/a5-board-ci}"
DEVICE_ID="${PTOAS_A5_DEVICE_ID:-auto}"
SOC_VERSION="${SOC_VERSION:-Ascend950}"
EXTRA_SKIP_CASES=""
RUN_ONLY_CASES="${RUN_ONLY_CASES:-}"
PTO_ISA_REPO="${PTO_ISA_REPO:-https://gitcode.com/cann/pto-isa.git}"
PTO_ISA_COMMIT="${PTO_ISA_COMMIT:-}"
# Track the newest pto-isa master instead of the repository pin. The pin has to
# be advanced through a PR plus a verified run, which lagged the fixes by weeks;
# pto-isa runs its own daily guard, so a broken isa commit is visible there.
# Set PTO_ISA_TRACK_LATEST=0 (or pass --pto-isa-commit) to go back to the pin.
PTO_ISA_TRACK_LATEST="${PTO_ISA_TRACK_LATEST:-1}"
JOBS=""
DO_BUILD=1
DO_SYNC=1
PREPARE_ONLY=0
DO_REPORT=1
SKIP_LOCK=0
RETENTION_DAYS="${PTOAS_A5_RETENTION_DAYS:-7}"
KEEP_HEAVY_RUNS="${PTOAS_A5_KEEP_HEAVY_RUNS:-0}"
WAIT_IDLE_MIN="${PTOAS_A5_WAIT_IDLE_MIN:-30}"
SCOPE="manual"
NIGHT_WINDOW=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --workspace) WORKSPACE="$2"; shift 2 ;;
    --ci-root) CI_ROOT="$2"; shift 2 ;;
    --device) DEVICE_ID="$2"; shift 2 ;;
    --soc) SOC_VERSION="$2"; shift 2 ;;
    --skip-cases) EXTRA_SKIP_CASES="$2"; shift 2 ;;
    --run-only) RUN_ONLY_CASES="$2"; shift 2 ;;
    --pto-isa-repo) PTO_ISA_REPO="$2"; shift 2 ;;
    --pto-isa-commit) PTO_ISA_COMMIT="$2"; shift 2 ;;
    --jobs) JOBS="$2"; shift 2 ;;
    --no-build) DO_BUILD=0; shift ;;
    --no-sync) DO_SYNC=0; shift ;;
    --prepare-only) PREPARE_ONLY=1; shift ;;
    --no-report) DO_REPORT=0; shift ;;
    --skip-lock) SKIP_LOCK=1; shift ;;
    --retention-days) RETENTION_DAYS="$2"; shift 2 ;;
    --wait-idle) WAIT_IDLE_MIN="$2"; shift 2 ;;
    --scope) SCOPE="$2"; shift 2 ;;
    --night-window) NIGHT_WINDOW=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) die "unknown argument: $1 (see --help)" ;;
  esac
done

case "${SCOPE}" in
  nightly|manual) ;;
  *) die "--scope must be nightly or manual, got: ${SCOPE}" ;;
esac
if [[ ${PREPARE_ONLY} -eq 1 ]]; then
  SCOPE="prepare"
fi
[[ -n "${CI_ROOT}" && "${CI_ROOT}" == /* && "${CI_ROOT}" != "/" ]] \
  || die "--ci-root must be an absolute path other than / (got: ${CI_ROOT})"

RUN_ID="$(date +%Y%m%d-%H%M%S)"
LOG_ROOT="${CI_ROOT}/logs"
WORK_ROOT="${CI_ROOT}/work/${RUN_ID}"
PAYLOAD_DIR="${WORK_ROOT}/payload"
OUTPUT_ROOT="${WORK_ROOT}/npu-validation"
RESULTS_TSV="${WORK_ROOT}/remote_npu_validation_results.tsv"
BOARD_LOG="${WORK_ROOT}/board-validation.log"
SAMPLE_BUILD_LOG="${WORK_ROOT}/sample-build.log"
SUMMARY_MD="${WORK_ROOT}/summary.md"
# "ok" / "failed" / "skipped" from the report script's Feishu delivery attempt.
FEISHU_STATUS="${WORK_ROOT}/feishu-status.txt"
# Cases deliberately excluded before generation never reach the manifest, so
# they are recorded here and reported by the summary/Feishu card.
NOT_GENERATED_TSV="${WORK_ROOT}/not-generated.tsv"
LLVM_SOURCE_VERSION="19.1.7"
A5_DEFAULT_SKIP_CASES="mix_kernel,vadd_validshape,vadd_validshape_dynamic,print,storefp"
A3_ONLY_CASES="partition5d,partition5d_dynamic,mrgsort,tmatmulk_autosync,movfp_fixpipe_reuse_a3"
A5_INCOMPATIBLE_LEVEL3_CASES="decode_projection_incore_0,rmsnorm_incore_0,"
A5_INCOMPATIBLE_LEVEL3_CASES+="test_tmov_col_major_16x1_align_a5,test_tmov_row_major_1x16_control_a5"
# timg2col_runtime was gated here while it still used the pre-valid_shape
# ConvTileBufType/TImg2colOp signatures. Master 3be6650d9 ported it to the
# current conv-tile API and `runop.sh -t Img2col` generates it again, so the
# gate is gone and the case runs on the board.
#
# matmul_mx_low_precision: master 67dd41eaf added MX tensor-view layouts whose
# dim 1 must be divisible by 16; this sample feeds 110/63, so generation stops
# the whole build. Gated, not hidden: it is written to not-generated.tsv and
# listed on the card until the sample's shapes are migrated.
#
# Keys are sample base names, not file names: runop.sh strips a trailing "-pto"
# before matching, so "matmul_mx_low_precision-pto" skips nothing and the case
# still fails the build.
A5_GENERATION_SKIP_CASES="${PTOAS_A5_GENERATION_SKIP_CASES-matmul_mx_low_precision}"

if ! mkdir -p "${LOG_ROOT}" "${WORK_ROOT}" "${CI_ROOT}/locks" "${CI_ROOT}/state"; then
  printf 'ERROR: cannot create the CI directories under %s\n' "${CI_ROOT}" >&2
  exit 1
fi
# Keep one log per run next to the artifacts.
exec > >(tee -a "${WORK_ROOT}/nightly.log") 2>&1

# NOTE: heavy-tree pruning happens in main(), after the state file and the EXIT
# trap exist. Calling it here executed before the function definition was
# reached (bash defines functions top-down), which exited 127 on 2026-10-30 and
# killed every attempt within seconds.

# ---------------------------------------------------------------------------
# Run state and failure notification
#
# A nightly that dies before the board stage (git auth, disk, device busy) used
# to fail silently: the run directory was never created, nothing was reported,
# and the first person to notice was whoever asked the next morning. Every exit
# path now records a state file (read by --check-last-run in the cron wrapper)
# and, when a Feishu webhook is configured, sends one short failure message.
# ---------------------------------------------------------------------------
case "${SCOPE}" in
  nightly) STATE_FILE="${CI_ROOT}/state/last-run.env" ;;
  manual) STATE_FILE="${CI_ROOT}/state/last-manual.env" ;;
  prepare) STATE_FILE="${CI_ROOT}/state/last-prepare.env" ;;
esac
RUN_STARTED_AT="$(date +'%F %T')"
RUN_DATE="$(date +'%F')"
# Set to 1 once the end-of-run Feishu card has been sent, so the failure alert
# in on_exit (and the guard/watchdog) do not post the same failure twice.
REPORTED=0
# Last pipeline stage that started, and the classified failure kind. rc=1 alone
# cannot tell "the board found failing cases" from "the driver crashed at
# startup"; these two fields can.
PHASE="init"
FAILURE_KIND="none"

write_state() {
  local rc="$1"
  local tmp="${STATE_FILE}.tmp.$$"
  {
    # %q keeps "START=2026-10-28 00:00:01" sourceable from the watchdog.
    printf 'SCOPE=%q\n' "${SCOPE}"
    printf 'RUN_DATE=%q\n' "${RUN_DATE}"
    printf 'RUN_ID=%q\n' "${RUN_ID}"
    printf 'COMMIT=%q\n' "${PTOAS_CI_COMMIT_SHA:-unknown}"
    printf 'DEVICE=%q\n' "${DEVICE_ID}"
    printf 'HOST=%q\n' "$(hostname)"
    printf 'START=%q\n' "${RUN_STARTED_AT}"
    printf 'END=%q\n' "$(date +'%F %T')"
    printf 'RC=%s\n' "${rc}"
    printf 'FAILURE_KIND=%q\n' "${FAILURE_KIND}"
    printf 'PHASE=%q\n' "${PHASE}"
    printf 'REPORTED=%s\n' "${REPORTED}"
    printf 'WORK_ROOT=%q\n' "${WORK_ROOT}"
    printf 'RESULTS_TSV=%q\n' "${RESULTS_TSV}"
  } > "${tmp}"
  mv -f "${tmp}" "${STATE_FILE}"
  chmod 600 "${STATE_FILE}" 2>/dev/null || true
}

notify_feishu() {
  a5_notify_text "$1"
}

# Copy the small, human-relevant artifacts to a stable directory so the GitCode
# upload step never has to guess which run directory belongs to this job. On an
# early failure only the files that exist are copied.
publish_latest_artifacts() {
  local dest="${CI_ROOT}/work/latest-artifacts"
  local f
  mkdir -p "${dest}" || return 0
  find "${dest}" -mindepth 1 -maxdepth 1 ! -type d -delete 2>/dev/null || true
  for f in remote_npu_validation_results.tsv board-validation.log \
           sample-build.log summary.md nightly.log not-generated.tsv; do
    [[ -f "${WORK_ROOT}/${f}" ]] && cp -f "${WORK_ROOT}/${f}" "${dest}/${f}" || true
  done
}

on_exit() {
  local rc=$?
  # Distinguish a real board verdict from an infrastructure failure. The state
  # file keeps the kind so the guard/watchdog can report it without guessing.
  if (( rc == 0 )); then
    FAILURE_KIND="none"
  elif [[ -s "${RESULTS_TSV}" ]] && grep -q $'	FAIL	' "${RESULTS_TSV}" 2>/dev/null; then
    FAILURE_KIND="board"
  else
    FAILURE_KIND="$(a5_classify_failure "${WORK_ROOT}/nightly.log" | cut -f1)"
  fi
  write_state "${rc}"
  publish_latest_artifacts
  if (( rc == 0 )); then
    log "RESULT: PASS (state: ${STATE_FILE})"
    return 0
  fi
  printf 'FAILED rc=%s at %s (log: %s/nightly.log)\n' \
    "${rc}" "$(date +'%F %T')" "${WORK_ROOT}" > "${WORK_ROOT}/FAILED"
  log "RESULT: FAILED rc=${rc}; state: ${STATE_FILE}; marker: ${WORK_ROOT}/FAILED"
  # A result card already describes this exact board outcome; only a failure
  # that happened after the card (rc != BOARD_RC) still needs its own alert.
  if [[ "${REPORTED}" == "1" && "${rc}" == "${BOARD_RC:-}" ]]; then
    log "failure already reported by the result card; skipping the extra alert"
    return 0
  fi
  # Persist the fact that a human was already told, so the 08:00 watchdog does
  # not repeat the same failure. A failed send keeps REPORTED=0, so the
  # watchdog still has a chance to surface the night.
  if [[ -n "${A5_FEISHU_WEBHOOK_URL:-}" ]]; then
    if notify_feishu "PTOAS A5 板测失败（exit ${rc}，stage=${PHASE}）
$(a5_failure_summary "${WORK_ROOT}/nightly.log")
run: ${RUN_ID}  host: $(hostname)  device: ${DEVICE_ID}
log: ${WORK_ROOT}/nightly.log"; then
      REPORTED=1
      write_state "${rc}"
    else
      log "WARN: failure alert could not be delivered; REPORTED stays 0"
    fi
  fi
}

trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
PHASE="startup"
write_state running

# Validate the workspace only after the state file and trap exist: a bad
# --workspace (or a checkout that vanished) must leave a recorded failure and an
# alert, not a silent exit with last night's state still on disk.
[[ -d "${WORKSPACE}" ]] || die "workspace does not exist: ${WORKSPACE}"
WORKSPACE="$(cd "${WORKSPACE}" && pwd)"
[[ -f "${WORKSPACE}/build.sh" ]] || die "not a PTOAS source tree: ${WORKSPACE}"

log "=== A5 nightly board validation ${RUN_ID} ==="
log "workspace=${WORKSPACE}"
log "ci_root=${CI_ROOT}"
log "work_root=${WORK_ROOT}"
log "host=$(hostname) arch=$(host_arch)"

# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------
require_command() {
  command -v "$1" >/dev/null 2>&1 || die "required command is missing: $1"
}

preflight_commands() {
  local cmd
  for cmd in bash git cmake ninja patch tar flock; do
    require_command "${cmd}"
  done
  # ccache is optional: build.sh falls back to a plain compiler invocation.
  if ! command -v ccache >/dev/null 2>&1; then
    warn "ccache is unavailable; continuing without a compiler cache"
  fi
  # rsync is only needed for the mirror step; tar is the fallback.
  if [[ ${DO_SYNC} -eq 1 ]] && ! command -v rsync >/dev/null 2>&1; then
    warn "rsync is unavailable; falling back to a tar-based mirror (stale files may remain)"
  fi
}

preflight_python() {
  local python_bin="${PTOAS_PYTHON_BIN:-}"
  if [[ -z "${python_bin}" && -x "${CI_ROOT}/venv/bin/python3" ]]; then
    python_bin="${CI_ROOT}/venv/bin/python3"
  fi
  if [[ -z "${python_bin}" ]]; then
    python_bin="$(command -v python3 || true)"
  fi
  [[ -n "${python_bin}" ]] || die "python3 is required"
  "${python_bin}" - <<'PY' || die "python dependencies missing (need numpy and pybind11<3; use a venv or pip install)"
import numpy  # noqa: F401
import pybind11

major = int(pybind11.__version__.split(".", 1)[0])
if major >= 3:
    raise SystemExit(
        "pybind11 %s is not compatible with MLIR 19 python bindings; pin pybind11==2.13.6"
        % pybind11.__version__
    )
PY
  PYTHON_BIN="${python_bin}"
  # Use the same interpreter for build, payload generation and the board run:
  # run_remote_npu_validation.sh invokes bare python3.
  PATH="$(dirname "${PYTHON_BIN}"):${PATH}"
  export PATH
  log "python=${PYTHON_BIN} ($("${PYTHON_BIN}" -V 2>&1))"
}

preflight_cann() {
  local cand
  if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
    for cand in /usr/local/Ascend/cann /usr/local/Ascend/ascend-toolkit/latest; do
      [[ -d "${cand}" ]] && { ASCEND_HOME_PATH="${cand}"; break; }
    done
  fi
  [[ -n "${ASCEND_HOME_PATH:-}" ]] || die "CANN toolkit not found (set ASCEND_HOME_PATH)"
  command -v npu-smi >/dev/null 2>&1 || die "npu-smi is required (Ascend driver)"
  local soc_lc
  soc_lc="$(printf '%s' "${SOC_VERSION}" | tr '[:upper:]' '[:lower:]')"
  case "${soc_lc}" in
    *950*|*a5*) ;;
    *) die "--soc must target A5 (Ascend950); got ${SOC_VERSION}" ;;
  esac
  local bisheng="${ASCEND_HOME_PATH}/bin/bisheng"
  if ! command -v bisheng >/dev/null 2>&1 && [[ ! -x "${bisheng}" ]]; then
    die "bisheng compiler not found under CANN: ${ASCEND_HOME_PATH}"
  fi
  log "ASCEND_HOME_PATH=${ASCEND_HOME_PATH}"
  export ASCEND_HOME_PATH
  log "driver_version=$(sed -n 's/^Version=//p' /usr/local/Ascend/driver/version.info 2>/dev/null | head -1)"
}

preflight_disk() {
  local avail_kb
  avail_kb="$(df -Pk "${CI_ROOT}" | awk 'NR==2 {print $4}')"
  # A cold run materialises LLVM (~6 GiB source+build) plus PTOAS and payloads.
  local need_kb=$((30 * 1024 * 1024))
  if [[ -z "${avail_kb}" || "${avail_kb}" -lt "${need_kb}" ]]; then
    die "insufficient free space under ${CI_ROOT}: $(( ${avail_kb:-0} / 1024 / 1024 )) GiB < 30 GiB"
  fi
  log "free_space_under_ci_root=$(( avail_kb / 1024 / 1024 )) GiB"
}

# ---------------------------------------------------------------------------
# Source mirroring
# ---------------------------------------------------------------------------
SRC_DIR="${WORKSPACE}"
mirror_source() {
  [[ ${DO_SYNC} -eq 1 ]] || { SRC_DIR="${WORKSPACE}"; return 0; }
  local dest="${CI_ROOT}/src"
  if [[ "${WORKSPACE}" == "${dest}" ]]; then
    SRC_DIR="${dest}"
    return 0
  fi
  local excluded=(--exclude=.git --exclude=build --exclude=build_out --exclude=install
                  --exclude=tmp --exclude=.work --exclude=node_modules)
  mkdir -p "${dest}"
  log "mirroring source tree into ${dest}"
  if command -v rsync >/dev/null 2>&1; then
    rsync -a --delete "${excluded[@]}" "${WORKSPACE}/" "${dest}/"
  else
    # Preserve build trees: only overlay the source files.
    ( cd "${WORKSPACE}" && tar -cf - "${excluded[@]}" . ) | ( cd "${dest}" && tar -xf - )
  fi
  SRC_DIR="${dest}"
}

# ---------------------------------------------------------------------------
# LLVM / third-party preparation
# ---------------------------------------------------------------------------
prepare_third_party() {
  THIRD_PARTY_DIR="${CI_ROOT}/3rd-party"
  mkdir -p "${THIRD_PARTY_DIR}"

  # build.sh only looks for <3rd-party>/cann-cmake and
  # <3rd-party>/feature-vpto-last3.patch. If the caller pre-populated them
  # elsewhere, link them into that layout so the documented variables are real.
  local cann_cmake_default="${THIRD_PARTY_DIR}/cann-cmake"
  local cann_cmake="${PTOAS_CANN_CMAKE_DIR:-${cann_cmake_default}}"
  if [[ "${cann_cmake}" != "${cann_cmake_default}" ]]; then
    [[ -f "${cann_cmake}/third_party/llvm.cmake" ]] \
      || die "PTOAS_CANN_CMAKE_DIR lacks third_party/llvm.cmake: ${cann_cmake}"
    if [[ -e "${cann_cmake_default}" && ! -L "${cann_cmake_default}" ]]; then
      warn "using the existing ${cann_cmake_default}; PTOAS_CANN_CMAKE_DIR is ignored"
    else
      ln -sfn "${cann_cmake}" "${cann_cmake_default}"
      log "linked ${cann_cmake_default} -> ${cann_cmake}"
    fi
  fi
  if [[ ! -f "${cann_cmake_default}/third_party/llvm.cmake" ]]; then
    warn "cann-cmake is not pre-populated at ${cann_cmake_default}; CMake will try to clone it"
    warn "if the clone fails, run: git clone --depth 1 -b master-055 " \
         "https://gitcode.com/cann/cmake.git ${cann_cmake_default}"
  fi

  # The vpto patch lives on a GitCode release URL. build.sh reads it from
  # <3rd-party>/feature-vpto-last3.patch (or pkg/); honour a pre-downloaded
  # copy at a different path by linking it into place.
  local patch_default="${THIRD_PARTY_DIR}/feature-vpto-last3.patch"
  local patch_src="${PTOAS_LLVM_PATCH_FILE:-${patch_default}}"
  if [[ "${patch_src}" != "${patch_default}" ]]; then
    [[ -f "${patch_src}" ]] || die "PTOAS_LLVM_PATCH_FILE is missing: ${patch_src}"
    if [[ -e "${patch_default}" && ! -L "${patch_default}" ]]; then
      warn "using the existing ${patch_default}; PTOAS_LLVM_PATCH_FILE is ignored"
    else
      ln -sfn "${patch_src}" "${patch_default}"
      log "linked ${patch_default} -> ${patch_src}"
    fi
  fi
  if [[ -f "${patch_default}" ]]; then
    log "using cached vpto patch: ${patch_default}"
  else
    warn "vpto patch not cached at ${patch_default}; build.sh will download it"
  fi
}

# A previous run that was interrupted leaves a configured-but-incomplete LLVM
# build tree. build.sh treats any tree with LLVM/MLIR CMake packages as
# reusable, so an incremental `cmake --build` is what actually repairs it.
ensure_llvm_complete() {
  [[ ${DO_BUILD} -eq 1 ]] || return 0
  local cache_root="${THIRD_PARTY_DIR}/lib_cache/llvm_${LLVM_SOURCE_VERSION}"
  [[ -d "${cache_root}" ]] || return 0
  local config
  local dir
  local found=0
  while IFS= read -r config; do
    [[ -n "${config}" ]] || continue
    # .../<build>/lib/cmake/mlir/MLIRConfig.cmake -> .../<build>
    dir="${config%/lib/cmake/mlir/MLIRConfig.cmake}"
    [[ -d "${dir}" ]] || continue
    found=1
    log "completing cached LLVM/MLIR build: ${dir}"
    if ! cmake --build "${dir}" -- -j "${JOBS}"; then
      die "failed to complete the cached LLVM/MLIR build at ${dir}"
    fi
  done < <(find "${cache_root}" -maxdepth 5 -type f \
             -path '*/lib/cmake/mlir/MLIRConfig.cmake' 2>/dev/null | sort)
  [[ ${found} -eq 1 ]] || log "no configured LLVM/MLIR cache found; build.sh will create one"
}

# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------
device_health() {
  npu-smi info -t health -i "$1" 2>/dev/null \
    | awk -F: '/Health Status/ {gsub(/[[:space:]]/, "", $2); print $2}'
}

device_is_busy() {
  local out
  # Count only numeric process ids: some npu-smi versions print the string
  # "Process id" in the table header even when the device is idle.
  out="$(npu-smi info -t proc-mem -i "$1" 2>/dev/null \
    | awk -F: '/Process id/ {gsub(/[[:space:]]/, "", $2); if ($2 ~ /^[0-9]+$/) n++} END {print n+0}')"
  [[ "${out:-0}" -gt 0 ]]
}

available_devices() {
  local dev
  for dev in /dev/davinci[0-9]*; do
    [[ -e "${dev}" ]] || continue
    printf '%s\n' "${dev##*[!0-9]}"
  done | sort -n
}

device_ready() {
  local health
  health="$(device_health "$1")"
  [[ "${health}" == "OK" || "${health}" == "Healthy" ]] || return 1
  ! device_is_busy "$1"
}

select_device() {
  local dev
  local waited=0
  if [[ "${DEVICE_ID}" != "auto" ]]; then
    [[ "${DEVICE_ID}" =~ ^[0-9]+$ ]] || die "--device must be a non-negative integer or 'auto'"
    while ! device_ready "${DEVICE_ID}"; do
      if (( waited >= WAIT_IDLE_MIN )); then
        die "requested device ${DEVICE_ID} is not healthy and idle " \
            "(health=$(device_health "${DEVICE_ID}"), waited ${waited} min)"
      fi
      log "device ${DEVICE_ID} not ready yet (health=$(device_health "${DEVICE_ID}")); " \
          "retrying in 60s (${waited}/${WAIT_IDLE_MIN} min)"
      sleep 60
      waited=$((waited + 1))
    done
  else
    while :; do
      while IFS= read -r dev; do
        if device_ready "${dev}"; then
          DEVICE_ID="${dev}"
          break 2
        fi
      done < <(available_devices)
      if (( waited >= WAIT_IDLE_MIN )); then
        die "no healthy idle A5 device found after ${waited} min; refusing to preempt other work"
      fi
      log "no healthy idle A5 device yet; retrying in 60s (${waited}/${WAIT_IDLE_MIN} min)"
      sleep 60
      waited=$((waited + 1))
    done
  fi

  # Prevent two of our own runs from sharing a device even when the scheduler
  # starts them at the same time.
  exec 9>"${CI_ROOT}/locks/npu${DEVICE_ID}.lock"
  flock -n 9 || die "device ${DEVICE_ID} is locked by another PTOAS CI run"
  log "selected device=${DEVICE_ID} health=$(device_health "${DEVICE_ID}") waited=${waited}min"
}

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------
BUILD_DIR=""
INSTALL_DIR=""
MLIR_PYTHON_ROOT=""

build_ptoas() {
  BUILD_DIR="${SRC_DIR}/build"
  INSTALL_DIR="${SRC_DIR}/install"
  local python_dir
  python_dir="$(dirname "${PYTHON_BIN}")"
  log "building PTOAS (jobs=${JOBS})"
  (
    cd "${SRC_DIR}"
    export PATH="${python_dir}:${PATH}"
    export CCACHE_DIR="${CI_ROOT}/cache/ccache"
    mkdir -p "${CCACHE_DIR}"
    bash ./build.sh --build --cann_3rd_lib_path "${THIRD_PARTY_DIR}" -j "${JOBS}"
  )
  [[ -x "${BUILD_DIR}/tools/ptoas/ptoas" ]] || die "ptoas binary not found after build"
  [[ -x "${BUILD_DIR}/tools/ptobc/ptobc" ]] || die "ptobc binary not found after build"
}

load_build_env() {
  local env_file="${BUILD_DIR}/ptoas-test-env.sh"
  if [[ -f "${env_file}" ]]; then
    # shellcheck disable=SC1090
    source "${env_file}"
    log "loaded build environment from ${env_file}"
  else
    # No build.sh env file: locate the LLVM/MLIR tree that build.sh actually
    # used instead of guessing its cache-key suffix.
    local mlir_config
    mlir_config="$(find "${THIRD_PARTY_DIR}/lib_cache" -maxdepth 6 -type f \
                     -path '*/lib/cmake/mlir/MLIRConfig.cmake' 2>/dev/null | head -1)"
    [[ -n "${mlir_config}" ]] \
      || die "no LLVM/MLIR build found under ${THIRD_PARTY_DIR}/lib_cache; drop --no-build"
    LLVM_BUILD_DIR="${mlir_config%/lib/cmake/mlir/MLIRConfig.cmake}"
    MLIR_PYTHON_ROOT="${LLVM_BUILD_DIR}/tools/mlir/python_packages/mlir_core"
    export MLIR_PYTHON_ROOT
    export PTO_INSTALL_DIR="${INSTALL_DIR}"
    export PTO_PYTHON_ROOT="${INSTALL_DIR}"
  fi
  [[ -d "${INSTALL_DIR}" ]] || die "install tree missing: ${INSTALL_DIR}"
  MLIR_PYTHON_ROOT="${MLIR_PYTHON_ROOT:-${LLVM_BUILD_DIR}/tools/mlir/python_packages/mlir_core}"
  [[ -d "${MLIR_PYTHON_ROOT}/mlir" ]] || die "MLIR python packages missing: ${MLIR_PYTHON_ROOT}"
  PYTHONPATH="${INSTALL_DIR}:${MLIR_PYTHON_ROOT}:${PYTHONPATH:-}"
  LD_LIBRARY_PATH="${LLVM_BUILD_DIR}/lib:${INSTALL_DIR}/lib:${LD_LIBRARY_PATH:-}"
  export PYTHONPATH LD_LIBRARY_PATH
}

# ---------------------------------------------------------------------------
# Payload generation
# ---------------------------------------------------------------------------
generate_payload() {
  log "generating A5 sample payload"
  mkdir -p "${PAYLOAD_DIR}/test/samples" \
           "${PAYLOAD_DIR}/test/npu_validation/scripts" \
           "${PAYLOAD_DIR}/test/npu_validation/templates"
  local gen_rc
  set +e
  (
    cd "${SRC_DIR}"
    export PTOAS_BIN="${BUILD_DIR}/tools/ptoas/ptoas"
    export PTOBC_BIN="${BUILD_DIR}/tools/ptobc/ptobc"
    export PYTHON_BIN
    export PTO_PYTHON_ROOT="${INSTALL_DIR}/"
    export PTOAS_OUT_DIR="${PAYLOAD_DIR}/test/samples"
    export PTOAS_FLAGS="--pto-arch a5 --enable-insert-sync"
    # runop.sh keys its A5/A3 decisions on SOC_VERSION: the model-case
    # completeness glob and the A5-only Python samples. Without it the manifest
    # falls back to the A3 model glob and asks the board runner to cover cases
    # an A5 target can never build.
    export SOC_VERSION
    export PTOAS_SAMPLE_JOBS="${PTOAS_RUNOP_JOBS:-1}"
    export PTOAS_SKIP_CASES="${A3_ONLY_CASES},${A5_INCOMPATIBLE_LEVEL3_CASES},${A5_GENERATION_SKIP_CASES}"
    # A sample named after a stdlib module shadows it: test/samples/Random/random.py
    # gets imported as "random" by the compiler's dependency chain when the
    # sample's directory leads sys.path, so it re-executes itself and the ptoas
    # extension registers twice. Keep the sample directory off sys.path for
    # generation only -- the compare scripts later rely on sibling imports.
    export PYTHONSAFEPATH=1
    export PYTHONPATH LD_LIBRARY_PATH
    bash test/samples/runop.sh --enablebc all
  ) 2>&1 | tee "${SAMPLE_BUILD_LOG}"
  gen_rc="${PIPESTATUS[0]}"
  set -e
  if [[ ${gen_rc} -ne 0 ]]; then
    die "sample payload generation failed (rc=${gen_rc}); failing samples are listed in ${SAMPLE_BUILD_LOG}"
  fi

  cp "${SRC_DIR}/test/npu_validation/scripts/generate_testcase.py" \
     "${PAYLOAD_DIR}/test/npu_validation/scripts/"
  cp "${SRC_DIR}/test/npu_validation/scripts/run_remote_npu_validation.sh" \
     "${PAYLOAD_DIR}/test/npu_validation/scripts/"
  cp "${SRC_DIR}/test/npu_validation/templates/"* \
     "${PAYLOAD_DIR}/test/npu_validation/templates/"
  chmod +x "${PAYLOAD_DIR}/test/npu_validation/scripts/run_remote_npu_validation.sh"
  write_not_generated_manifest
}

# A case excluded before generation is absent from the manifest, which makes the
# suite look 1 case smaller instead of flagging a gap. Record it with a reason so
# the summary and the Feishu card can report it.
write_not_generated_manifest() {
  local out="${NOT_GENERATED_TSV}"
  : > "${out}"
  [[ -n "${A5_GENERATION_SKIP_CASES}" ]] || return 0
  local name sample reason
  for name in ${A5_GENERATION_SKIP_CASES//,/ }; do
    sample="$(find "${SRC_DIR}/test/samples" -mindepth 2 -maxdepth 2 \
                -name "${name}.py" -printf '%h\n' 2>/dev/null | head -1)"
    case "${name}" in
      matmul_mx_low_precision)
        reason="主线 67dd41eaf 起 MX 张量视图要求 dim1 能被 16 整除，样例用 110/63，生成阶段跳过，需样例 owner 迁移形状" ;;
      *)
        reason="generation skipped via PTOAS_A5_GENERATION_SKIP_CASES" ;;
    esac
    if [[ -n "${sample}" ]]; then
      printf '%s/%s\t%s\n' "$(basename "${sample}")" "${name}" "${reason}" >> "${out}"
    else
      printf '%s\t%s\n' "${name}" "${reason}（样例文件未找到，按名称记录）" >> "${out}"
    fi
  done
  log "not generated (reported, not silently dropped): $(wc -l < "${out}") case(s)"
}

resolve_pto_isa_commit() {
  [[ -n "${PTO_ISA_COMMIT}" ]] && return 0
  if [[ "${PTO_ISA_TRACK_LATEST}" == "1" ]]; then
    local -a git_opts=(-c credential.helper=)
    if [[ -n "${PTOAS_GIT_CREDENTIALS_FILE:-}" && -f "${PTOAS_GIT_CREDENTIALS_FILE}" ]]; then
      git_opts=(-c credential.helper= -c "credential.helper=store --file=${PTOAS_GIT_CREDENTIALS_FILE}")
    fi
    local latest
    # Bounded: an unreachable mirror must not stall the night before the board.
    latest="$(timeout 60 git "${git_opts[@]}" ls-remote "${PTO_ISA_REPO}" refs/heads/master 2>/dev/null \
                | awk 'NR == 1 { print $1 }')"
    if [[ -n "${latest}" ]]; then
      PTO_ISA_COMMIT="${latest}"
      log "pto-isa: tracking latest ${PTO_ISA_REPO} master at ${latest}"
      return 0
    fi
    log "pto-isa: could not resolve latest master from ${PTO_ISA_REPO}; falling back to the repository pin"
  fi
  local pin
  pin="$(sed -n 's/^PTO_ISA_COMMIT="\${PTO_ISA_COMMIT:-\([0-9a-f]\{40\}\)}"$/\1/p' \
           "${SRC_DIR}/test/npu_validation/scripts/run_remote_npu_validation.sh" | head -1)"
  [[ -n "${pin}" ]] || die "cannot resolve the repository pto-isa pin"
  PTO_ISA_COMMIT="${pin}"
  log "pto-isa: using the repository pin ${pin}"
}

vendor_pto_isa() {
  log "vendoring pto-isa ${PTO_ISA_COMMIT}"
  local cache="${CI_ROOT}/work/pto-isa-source"
  local -a git_opts=(-c credential.helper=)
  if [[ -n "${PTOAS_GIT_CREDENTIALS_FILE:-}" ]]; then
    [[ -f "${PTOAS_GIT_CREDENTIALS_FILE}" ]] \
      || die "PTOAS_GIT_CREDENTIALS_FILE is set but missing: ${PTOAS_GIT_CREDENTIALS_FILE}"
    git_opts=(-c credential.helper= -c "credential.helper=store --file=${PTOAS_GIT_CREDENTIALS_FILE}")
  fi
  if [[ ! -d "${cache}/.git" ]]; then
    rm -rf -- "${cache}"
    git "${git_opts[@]}" clone "${PTO_ISA_REPO}" "${cache}"
  fi
  if git -C "${cache}" remote get-url origin >/dev/null 2>&1; then
    git -C "${cache}" remote set-url origin "${PTO_ISA_REPO}"
  fi
  # The pinned commit must be fetchable even from a shallow mirror.
  if ! git -C "${cache}" cat-file -e "${PTO_ISA_COMMIT}^{commit}" 2>/dev/null; then
    git -C "${cache}" "${git_opts[@]}" fetch --depth 1 origin "${PTO_ISA_COMMIT}"
  fi
  git -C "${cache}" checkout -f "${PTO_ISA_COMMIT}"
  mkdir -p "${PAYLOAD_DIR}/pto-isa"
  cp -a "${cache}/." "${PAYLOAD_DIR}/pto-isa/"
  rm -rf -- "${PAYLOAD_DIR}/pto-isa/.git"
}

# ---------------------------------------------------------------------------
# Board run
# ---------------------------------------------------------------------------
run_board() {
  # --skip-cases is additive: keep the known-bad A5 defaults even when the
  # caller adds triage cases.
  local skip_cases="${A5_DEFAULT_SKIP_CASES}"
  if [[ -n "${EXTRA_SKIP_CASES}" ]]; then
    skip_cases="${skip_cases},${EXTRA_SKIP_CASES}"
  fi
  skip_cases="${skip_cases:+${skip_cases},}${A3_ONLY_CASES}"
  skip_cases="$(printf '%s' "${skip_cases}" | tr '[:space:]' ',' | tr -s ',')"
  skip_cases="${skip_cases#,}"
  skip_cases="${skip_cases%,}"
  log "skip_cases=${skip_cases}"
  log "run_only_cases=${RUN_ONLY_CASES:-<all>}"

  local env_file="${WORK_ROOT}/board-validation.env"
  # %q keeps space-separated lists and paths safe when run_remote sources it.
  {
    printf 'STAGE=%q\n' run
    printf 'RUN_MODE=%q\n' npu
    printf 'SOC_VERSION=%q\n' "${SOC_VERSION}"
    printf 'GOLDEN_MODE=%q\n' npu
    printf 'SKIP_CASES=%q\n' "${skip_cases}"
    printf 'RUN_ONLY_CASES=%q\n' "${RUN_ONLY_CASES}"
    printf 'EXPECTED_CASES_FILTER=%q\n' "${RUN_ONLY_CASES}"
    printf 'PTO_ISA_REPO=%q\n' "${PTO_ISA_REPO}"
    printf 'PTO_ISA_COMMIT=%q\n' "${PTO_ISA_COMMIT}"
    printf 'TMPDIR=%q\n' "${WORK_ROOT}/tmp"
    printf 'TMP=%q\n' "${WORK_ROOT}/tmp"
    printf 'TEMP=%q\n' "${WORK_ROOT}/tmp"
    printf 'OUTPUT_ROOT=%q\n' "${OUTPUT_ROOT}"
    printf 'RESULTS_TSV=%q\n' "${RESULTS_TSV}"
  } > "${env_file}"
  mkdir -p "${WORK_ROOT}/tmp"

  local rc=0
  set +e
  (
    cd "${PAYLOAD_DIR}"
    export DEVICE_ID="${DEVICE_ID}"
    set -a
    # shellcheck disable=SC1090
    source "${env_file}"
    set +a
    bash ./test/npu_validation/scripts/run_remote_npu_validation.sh
  ) 2>&1 | tee "${BOARD_LOG}"
  rc="${PIPESTATUS[0]}"
  set -e
  BOARD_RC="${rc}"
  log "board validation exit code: ${rc}"
}

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
write_summary() {
  local report_python
  if [[ -x "${PYTHON_BIN}" ]]; then
    report_python="${PYTHON_BIN}"
  else
    report_python="$(command -v python3 || true)"
  fi
  local run_url="${PTOAS_CI_RUN_URL:-}"
  local sha="${PTOAS_CI_COMMIT_SHA:-unknown}"
  local -a not_generated_args=()
  # The "generated + not generated = A5 total" line only describes a full sweep;
  # a --run-only triage run would claim a total of (subset + skips).
  if [[ -f "${NOT_GENERATED_TSV}" && -z "${RUN_ONLY_CASES}" ]]; then
    not_generated_args=(--not-generated "${NOT_GENERATED_TSV}")
  fi

  if [[ ${DO_REPORT} -eq 1 && -n "${report_python}" \
        && -f "${SRC_DIR}/.github/scripts/report_a5_board_results.py" ]]; then
    local report_rc=0
    rm -f -- "${FEISHU_STATUS}"
    "${report_python}" "${SRC_DIR}/.github/scripts/report_a5_board_results.py" \
      --results "${RESULTS_TSV}" \
      --log "${BOARD_LOG}" \
      "${not_generated_args[@]}" \
      --conclusion "$([[ ${BOARD_RC} -eq 0 ]] && echo success || echo failure)" \
      --run-url "${run_url}" \
      --run-id "${RUN_ID}" \
      --sha "${sha}" \
      --notify-status "${FEISHU_STATUS}" \
      > "${SUMMARY_MD}" 2>&1 || report_rc=$?
    cat "${SUMMARY_MD}"
    # The report script sends the Feishu card itself. Only a confirmed delivery
    # counts as "reported"; otherwise on_exit sends the short text alert, so a
    # webhook outage cannot leave a failed night silently marked as reported.
    if [[ -f "${FEISHU_STATUS}" && "$(cat "${FEISHU_STATUS}")" == "ok" ]]; then
      REPORTED=1
    elif (( report_rc != 0 )) && [[ ! -s "${SUMMARY_MD}" ]]; then
      log "WARN: the result report failed before writing a summary (rc=${report_rc})"
    fi
  else
    {
      echo "# A5 board validation ${RUN_ID}"
      echo
      echo "- conclusion: $([[ ${BOARD_RC} -eq 0 ]] && echo success || echo failure)"
      echo "- commit: ${sha}"
      echo "- device: ${DEVICE_ID}"
      # Same errexit trap as above: a bare `[[ -f ]] && awk` returns 1 when the
      # TSV is missing, which aborts the run instead of printing a short summary.
      if [[ -f "${RESULTS_TSV}" ]]; then
        awk -F'\t' 'NR>1 {c[$2]++} END {for (k in c) printf "- %s: %d\n", k, c[k]}' "${RESULTS_TSV}"
      fi
    } | tee "${SUMMARY_MD}"
  fi
}

# Each run writes a ~17 GiB npu-validation tree and a ~66 MiB payload; its
# results, summary and logs are ~3 MiB. Keep the heavy trees only for the
# current run (and optionally the N newest earlier ones) so a week of history
# costs megabytes instead of hundreds of GiB. Everything pruned here is
# regenerated by the next run.
prune_heavy_artifacts() {
  local keep="${1:-0}"
  local -a runs=()
  local name dir sub size kept=0
  mapfile -t runs < <(
    find "${CI_ROOT}/work" -mindepth 1 -maxdepth 1 -type d -name '20*-*' \
      -printf '%f\n' 2>/dev/null | sort -r
  )
  for name in ${runs[@]+"${runs[@]}"}; do
    dir="${CI_ROOT}/work/${name}"
    [[ "${dir}" == "${WORK_ROOT}" ]] && continue
    if (( kept < keep )); then
      kept=$((kept + 1))
      continue
    fi
    for sub in npu-validation payload; do
      [[ -n "${dir}" && "${dir}" != "/" ]] || continue
      [[ -d "${dir}/${sub}" ]] || continue
      size="$(du -sh "${dir}/${sub}" 2>/dev/null | cut -f1)"
      log "pruning ${sub} (${size:-?}) from ${dir}"
      rm -rf -- "${dir}/${sub}" || true
    done
  done
}

cleanup_old_runs() {
  prune_heavy_artifacts "${KEEP_HEAVY_RUNS}"
  [[ "${RETENTION_DAYS}" =~ ^[0-9]+$ ]] || return 0
  local dir
  while IFS= read -r dir; do
    [[ "${dir}" == "${WORK_ROOT}" ]] && continue
    log "removing expired run: ${dir}"
    rm -rf -- "${dir}"
  done < <(find "${CI_ROOT}/work" -mindepth 1 -maxdepth 1 -type d -name '20*-*' \
             -mtime "+${RETENTION_DAYS}" 2>/dev/null)
  # The systemd unit appends to nightly-cron.log through a long-lived fd;
  # unlinking it would silently drop everything written afterwards.
  find "${LOG_ROOT}" -mindepth 1 -maxdepth 1 -type f -name '*.log' \
    ! -name 'nightly-cron.log' \
    -mtime "+${RETENTION_DAYS}" -delete 2>/dev/null || true
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
main() {
  if (( NIGHT_WINDOW )); then
    local hour
    hour="$(date +%H)"
    if (( 10#${hour} >= 6 && 10#${hour} < 22 )); then
      die "refusing to start the A5 nightly at ${hour}:00 local time; the nightly window is 22:00-06:00. " \
          "Omit --night-window (or set allow_daytime=true in the workflow) for a manual daytime run."
    fi
  fi
  # Take the lock before touching work/: a second run must not prune the heavy
  # trees of the run that already holds it. The cron wrapper takes the same
  # lock and passes --skip-lock.
  if [[ ${SKIP_LOCK} -eq 0 ]]; then
    exec 8>"${CI_ROOT}/locks/nightly.lock"
    flock -n 8 || die "another A5 nightly run holds ${CI_ROOT}/locks/nightly.lock"
  fi
  # Free last night's heavy trees before this run needs the space. Runs here,
  # not at load time, so the function is defined and a failure is recorded in
  # the state file; pruning is best-effort and must never fail the night.
  prune_heavy_artifacts "${KEEP_HEAVY_RUNS}" \
    || warn "heavy-artifact pruning failed; continuing (disk may run low)"
  if [[ -n "${PTOAS_GIT_CREDENTIALS_FILE:-}" ]]; then
    [[ -f "${PTOAS_GIT_CREDENTIALS_FILE}" ]] \
      || die "PTOAS_GIT_CREDENTIALS_FILE is set but missing: ${PTOAS_GIT_CREDENTIALS_FILE}"
    local mode
    mode="$(stat -c '%a' "${PTOAS_GIT_CREDENTIALS_FILE}" 2>/dev/null || echo 600)"
    if (( 8#${mode} & 077 )); then
      warn "PTOAS_GIT_CREDENTIALS_FILE is group/world accessible (mode ${mode}); chmod 600 it"
    fi
  fi
  PHASE="preflight"
  preflight_commands
  preflight_python
  preflight_cann
  preflight_disk
  prepare_third_party

  if [[ -z "${JOBS}" ]]; then
    JOBS="$(nproc 2>/dev/null || echo 4)"
    if (( JOBS > 64 )); then
      JOBS=64
    fi
  fi

  PHASE="build"
  mirror_source
  ensure_llvm_complete
  if [[ ${DO_BUILD} -eq 1 ]]; then
    build_ptoas
  else
    BUILD_DIR="${SRC_DIR}/build"
    INSTALL_DIR="${SRC_DIR}/install"
  fi
  load_build_env

  if [[ ${PREPARE_ONLY} -eq 1 ]]; then
    log "prepare-only requested; toolchain is ready"
    exit 0
  fi

  PHASE="device"
  select_device
  PHASE="payload"
  generate_payload
  resolve_pto_isa_commit
  vendor_pto_isa
  PHASE="board"
  run_board
  PHASE="summary"
  write_summary
  cleanup_old_runs

  log "results: ${RESULTS_TSV}"
  log "log: ${BOARD_LOG}"
  exit "${BOARD_RC}"
}

main "$@"
