#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
# Run the PyPTO gates that depend on pto-as, against a candidate pto-as build.
#
# By default the candidate is the released pto-as version named in
# `ptoas_release.txt` next to this script: PyPTO consumes pto-as as a released
# wheel, so the nightly must keep tracking the newest release. Bump that file
# when a release lands; `--candidate DIR` still points the run at an arbitrary
# tree (for example a branch build) when debugging.
#
# The selections mirror PyPTO's own CI rather than "all of tests/st": its
# simulator jobs run a fixed list of files and markers, and the rest of tests/st
# is device-only or needs multi-card batching that a host timer cannot provide.
# Running everything here would report failures PyPTO never gates on and bury
# the signal this nightly exists to produce.
#
# A red selection is re-run with the known-good baseline to separate a pto-as
# regression from a PyPTO-side or environment-side failure.
#
# `--stage lib` runs a second downstream consumer instead of PyPTO itself:
# pypto-lib's A5 on-board gate (ci.yml `device (a5)` over examples/ and
# daily_ci.yml `Model tests (a5)` over models/). Those entries are standalone
# scripts rather than pytest node ids, so they get their own runner, but the
# file list still comes from pypto-lib's own detect_changes.py -- a gate that
# runs a different set than the CI job it mirrors only reports noise. The stage
# is deliberately not part of `--stage all`: a pypto-lib-side model failure must
# not move the pto-as release verdict, so it runs as its own nightly and card.
#
# Environment (all optional):
#   PTOAS_PYPTO_ROOT        default /data/ptoas-pypto
#   PTOAS_CANDIDATE_ROOT    override the candidate (default: release pin below)
#   PTOAS_BASELINE_ROOT     default <root>/ptoas/v070 (skip attribution if absent)
#   PTOAS_WAIT_FOR_UNITS    systemd units to wait out before starting (empty by
#                           default; the timer unit sets ptoas-a5-nightly.service)
#   PTOAS_PYPTOLIB_ROOT     default <root>/src/pypto-lib
#   PTOAS_PYPTOLIB_REMOTE   default https://github.com/hw-native-sys/pypto-lib.git
#   PTOAS_PYPTOLIB_REF      default main
#   PTOAS_LIB_DEVICE_ID     card for the lib stage (default: the A5 device)
#   A5_FEISHU_WEBHOOK_URL / A5_FEISHU_SECRET  notification (silent without)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# The systemd units already load this file; manual runs need it too so the
# Feishu card still goes out.
if [[ -r /etc/ptoas-a5-ci.env ]]; then
  set -a
  # shellcheck disable=SC1091
  source /etc/ptoas-a5-ci.env
  set +a
fi

ROOT="${PTOAS_PYPTO_ROOT:-/data/ptoas-pypto}"
ENV="${ROOT}/mamba/envs/pypto"
PY="${ENV}/bin/python"
SRC="${ROOT}/src/pypto"
CAND="${PTOAS_CANDIDATE_ROOT:-}"
BASE="${PTOAS_BASELINE_ROOT:-${ROOT}/ptoas/v070}"
PIN_FILE="${SCRIPT_DIR}/ptoas_release.txt"
RELEASE_VER=""
RELEASE_NOTE=""
RUN_ID="$(date +%Y%m%d-%H%M%S)"
RESULT_NS="results"
OUT=""
STAGE="all"
ONLY=""
BASE_EXPLICIT=0
NO_NOTIFY=0
A5_DEVICE="${PTOAS_A5_DEVICE_ID:-2}"
LIB_DEVICE=""
PREFIX=""
PREFIX_SET=0
LIB="${PTOAS_PYPTOLIB_ROOT:-${ROOT}/src/pypto-lib}"
LIB_REMOTE="${PTOAS_PYPTOLIB_REMOTE:-https://github.com/hw-native-sys/pypto-lib.git}"
LIB_REF="${PTOAS_PYPTOLIB_REF:-main}"
DOWNSTREAM="PyPTO"

# PyPTO's on-board A5 gate: the five files its onboard-tests-a5 job runs.
A5_TESTS=(
  tests/st/runtime/cross_core/test_cross_core.py
  tests/st/runtime/ops/test_matmul.py::test_fp32_matmul_clears_inherited_hf32
  tests/st/runtime/ops/test_matmul_mx.py
  tests/st/runtime/ops/test_quant_mx.py
  tests/st/runtime/ops/test_quantized_matmul_mx.py
)

# PyPTO's system-tests-a5sim selection.
SIM_A5_FILES=(
  tests/st/runtime/ops/test_assemble.py
  tests/st/runtime/ops/test_gather.py
  tests/st/runtime/ops/test_gatherb.py
  tests/st/runtime/ops/test_mscatter.py
  tests/st/runtime/ops/test_prelu.py
  tests/st/runtime/ops/test_random.py
  tests/st/runtime/ops/test_sels.py
  tests/st/runtime/ops/test_tri.py
  tests/st/runtime/framework_and_models/test_qwen3_decode_scope3_mixed.py
  tests/st/runtime/control_flow/test_dyn_orch_shape.py::TestDynOrchShapeOperations::test_dyn_orch_paged_attention
)

SIM_KEYS=(a5sim-core sim-fillpad a5sim-crosscore a2a3sim-cast)
A5_KEYS=(a5-board)
LIB_KEYS=(lib-a5-examples lib-a5-models)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage) STAGE="$2"; shift 2 ;;
    --only) ONLY="$2"; shift 2 ;;
    --candidate) CAND="$2"; shift 2 ;;
    --baseline) BASE="$2"; BASE_EXPLICIT=1; shift 2 ;;
    --device) A5_DEVICE="$2"; shift 2 ;;
    --prefix) PREFIX="$2"; PREFIX_SET=1; shift 2 ;;
    --no-notify) NO_NOTIFY=1; shift ;;
    --help|-h)
      cat <<EOF
usage: $0 [--stage sim|a5|lib|all] [--only KEY[,KEY]] [--candidate DIR]
          [--baseline DIR] [--device ID] [--prefix TEXT] [--no-notify]

candidate: the release in .gitcode/scripts/ptoas_release.txt unless --candidate
           (or PTOAS_CANDIDATE_ROOT) points at another tree.

selections:
  a5sim-core        PyPTO system-tests-a5sim file list (--platform=a5sim)
  sim-fillpad       test_fillpad_inplace.py with --memory-planner=ptoas (both sims)
  a5sim-crosscore   test_cross_core.py (--platform=a5sim)
  a2a3sim-cast      test_cast_saturation.py (--platform=a2a3sim)
  a5-board          PyPTO onboard-tests-a5 subset on device ID (default 2)
  lib-a5-examples   pypto-lib ci.yml device (a5) file list from examples/
  lib-a5-models     pypto-lib daily_ci.yml Model tests (a5) list from models/

--stage all is sim+a5 only; the lib stage runs separately because its verdict is
about pypto-lib, not about pto-as.
EOF
      exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

LIB_DEVICE="${PTOAS_LIB_DEVICE_ID:-${A5_DEVICE}}"
if [[ "${STAGE}" == "lib" ]]; then
  RESULT_NS="results-lib"
  DOWNSTREAM="PyPTO-Lib"
fi
if (( ! PREFIX_SET )); then
  if [[ "${STAGE}" == "lib" ]]; then
    PREFIX="[PTOAS × PyPTO-Lib A5]"
  else
    PREFIX="[PTOAS × PyPTO 兼容]"
  fi
fi
OUT="${ROOT}/${RESULT_NS}/${RUN_ID}"

mkdir -p "${OUT}"
exec > >(tee "${OUT}/run.log") 2>&1

fail() { echo "ERROR: $*" >&2; exit 1; }

release_install_root() { printf '%s/ptoas/v%s' "${ROOT}" "${1//./}"; }

# Ask the release API for the wheel digest; empty when the API is unreachable
# (rate limit, flaky link) so the run can continue and record its own hash.
release_asset_digest() {
  local tag="$1" wheel="$2" json
  command -v python3 >/dev/null 2>&1 || return 0
  json="$(curl -sS --connect-timeout 10 -m 40 \
    "https://api.github.com/repos/hw-native-sys/PTOAS/releases/tags/${tag}" 2>/dev/null)" || return 0
  printf '%s' "${json}" | python3 - "${wheel}" <<'PY'
import json
import sys

try:
    payload = json.load(sys.stdin)
except Exception:
    raise SystemExit(0)
for asset in payload.get("assets", []):
    if asset.get("name") == sys.argv[1] and asset.get("digest"):
        print(asset["digest"].split(":", 1)[1])
        break
PY
}

install_release() {
  local ver="$1" dir="$2"
  local arch wheel tag gh dest url want
  arch="$(uname -m)"
  wheel="ptoas-${ver}-cp310-cp310-manylinux_2_34_${arch}.whl"
  tag="v${ver}"
  gh="https://github.com/hw-native-sys/PTOAS/releases/download/${tag}/${wheel}"
  dest="${ROOT}/ptoas/${wheel}"
  mkdir -p "${ROOT}/ptoas"
  # A wheel dropped in by hand wins over the network: an unpublished (draft)
  # release is invisible to this host, and the artifact is still the artifact.
  if [[ -s "${dest}" ]]; then
    echo "release: using pre-placed ${dest}"
  else
    for url in "${gh}" "https://ghfast.top/${gh}" "https://gh-proxy.com/${gh}" \
               "https://ghproxy.net/${gh}"; do
      echo "release: fetching ${wheel} via ${url}"
      rm -f "${dest}"
      curl -fL --retry 2 --retry-delay 3 --connect-timeout 15 -m 900 -o "${dest}" "${url}" && break
    done
  fi
  if [[ ! -s "${dest}" ]]; then
    echo "release: ${wheel} is not downloadable (tag ${tag}); drop it into ${ROOT}/ptoas/" >&2
    return 1
  fi
  want="$(release_asset_digest "${tag}" "${wheel}")"
  if [[ -n "${want}" ]]; then
    echo "${want}  ${dest}" | sha256sum -c - || { echo "release: sha256 mismatch for ${dest}" >&2; return 1; }
  else
    echo "release: upstream digest unavailable; recording $(sha256sum "${dest}" | cut -d' ' -f1)"
  fi
  "${PY}" -m venv --system-site-packages "${dir}" \
    || { echo "release: cannot create venv ${dir}" >&2; return 1; }
  local pip_rc=0
  "${dir}/bin/pip" install --no-deps --force-reinstall "${dest}" 2>&1 | tail -3 \
    || pip_rc=$?
  # The pipeline status is tail's; keep pip's own status so a failed install
  # falls back to the branch build instead of looking green.
  return "${pip_rc}"
}

# Candidate: an explicit override, else the newest release we know about. PyPTO
# declares what it consumes in toolchain/versions.env, so that wins over the
# pin file when it names something newer (the pin is the fallback for when the
# upstream checkout cannot be read). When the release has no downloadable wheel
# the run falls back to the branch install and says so on the card: a missing
# artifact should not cost the night's signal.
resolve_candidate() {
  [[ -n "${CAND}" ]] && return 0
  local pinned=""
  [[ -f "${PIN_FILE}" ]] && pinned="$(tr -d '[:space:]' < "${PIN_FILE}")"
  RELEASE_VER="${pinned}"
  local declared
  declared="$(sed -n 's/^PTOAS_VERSION=v\{0,1\}\([0-9][0-9.]*\)$/\1/p' \
    "${SRC}/toolchain/versions.env" 2>/dev/null | head -1)"
  if [[ -n "${declared}" ]]; then
    if [[ -z "${pinned}" ]] || \
       [[ "$(printf '%s\n%s\n' "${pinned}" "${declared}" | sort -V | tail -1)" == "${declared}" ]]; then
      RELEASE_VER="${declared}"
    fi
  fi
  [[ -n "${RELEASE_VER}" ]] || fail "no candidate: ${PIN_FILE} is empty and versions.env has no PTOAS_VERSION"
  CAND="$(release_install_root "${RELEASE_VER}")"
  if [[ ! -x "${CAND}/bin/ptoas" ]] && ! install_release "${RELEASE_VER}" "${CAND}"; then
    local fallback="${ROOT}/ptoas/candidate"
    [[ -x "${fallback}/bin/ptoas" ]] \
      || fail "release ${RELEASE_VER} unavailable and no branch candidate at ${fallback}"
    CAND="${fallback}"
    RELEASE_NOTE="release ${RELEASE_VER} unavailable; testing branch build"
    echo "WARN: ${RELEASE_NOTE}"
  fi
}

rebuild_pypto() {
  # Conda activation puts its own gcc 15 first and exports x86 tuning flags;
  # the 238 build is made with the system compiler on purpose.
  (
    export CC=/usr/bin/gcc CXX=/usr/bin/g++
    unset CFLAGS CXXFLAGS CPPFLAGS LDFLAGS
    cd "${SRC}"
    "${PY}" -m pip install -q --no-build-isolation --no-deps --force-reinstall .
  ) > "${OUT}/pypto.rebuild.log" 2>&1 || return 1
  (
    export CC=/usr/bin/gcc CXX=/usr/bin/g++
    unset CFLAGS CXXFLAGS CPPFLAGS LDFLAGS
    cd "${SRC}/runtime"
    "${PY}" -m pip install -q --no-build-isolation --no-deps --force-reinstall .
  ) >> "${OUT}/pypto.rebuild.log" 2>&1 || return 1
  # A reinstall repackages the assets and drops the managed pto-isa checkout; the
  # runtime then hangs trying to clone it when GitHub is unreachable.
  local assets seed
  assets="$("${PY}" -c \
    'import os,simpler_setup;print(os.path.join(os.path.dirname(simpler_setup.__file__),"_assets","build"))' \
    2>/dev/null || true)"
  if [[ -n "${assets}" && ! -d "${assets}/pto-isa" ]]; then
    seed="${ROOT}/pto-isa-cache"
    [[ -d "${SRC}/runtime/build/pto-isa" ]] && seed="${SRC}/runtime/build/pto-isa"
    [[ -d "${seed}" ]] && cp -a "${seed}" "${assets}/pto-isa"
  fi
  return 0
}

# Keep the upstream checkout current. A fetch or rebuild failure is reported on
# the card and the run continues with what is installed: a network problem must
# not silently turn into "no result tonight".
update_pypto() {
  local status="unchanged"
  if git -C "${SRC}" fetch -q origin main 2> "${OUT}/pypto.fetch.log"; then
    local cur new
    cur="$(git -C "${SRC}" rev-parse HEAD)"
    new="$(git -C "${SRC}" rev-parse origin/main 2>/dev/null || true)"
    if [[ -n "${new}" && "${cur}" != "${new}" ]]; then
      if git -C "${SRC}" checkout -q -f "${new}" && \
         git -C "${SRC}" submodule update --init --recursive -q 2>> "${OUT}/pypto.fetch.log"; then
        if rebuild_pypto; then
          status="updated"
        else
          # Keep tests and the installed package on the same revision.
          git -C "${SRC}" checkout -q -f "${cur}" || true
          git -C "${SRC}" submodule update --init --recursive -q 2>> "${OUT}/pypto.fetch.log" || true
          status="rebuild-failed"
        fi
      else
        status="checkout-failed"
      fi
    fi
  else
    status="fetch-failed"
  fi
  printf '%s\n' "${status}" > "${OUT}/pypto.update"
  echo "pypto update: ${status}"
  if [[ "${status}" == "updated" ]]; then
    git -C "${SRC}" rev-parse HEAD > "${OUT}/pypto.commit"
    git -C "${SRC}" log -1 --format='%H %ci' > "${OUT}/pypto.head"
  fi
}

# pypto-lib is a second consumer of PyPTO (and therefore of pto-as). Its CI
# checks it out per job; a host timer keeps one shallow checkout and moves it,
# because the file list must come from the revision under test.
update_pypto_lib() {
  local status="unchanged" cur="" new="" attempt ok=0
  mkdir -p "$(dirname -- "${LIB}")"
  if [[ ! -d "${LIB}/.git" ]]; then
    # github.com:443 from this host is intermittently unreachable; one retry is
    # cheap next to a whole night spent on yesterday's checkout.
    for attempt in 1 2; do
      if git -c credential.helper= clone -q --depth 1 -b "${LIB_REF}" \
           "${LIB_REMOTE}" "${LIB}" > "${OUT}/pypto-lib.update.log" 2>&1; then
        ok=1
        break
      fi
      # Only clear what a failed clone leaves behind (an empty dir or a bare
      # .git); never delete a tree that holds anything else.
      local keep=0 entry
      for entry in "${LIB}"/* "${LIB}"/.[!.]* "${LIB}"/..?*; do
        [[ -e "${entry}" ]] || continue
        if [[ "${entry##*/}" != ".git" ]]; then
          keep=1
          break
        fi
      done
      if (( ! keep )) && [[ -d "${LIB}" ]]; then
        [[ -n "${LIB}" && "${LIB}" != "/" ]] || fail "refusing to remove an empty LIB path"
        rm -rf -- "${LIB}"
      fi
      if (( attempt == 1 )); then sleep 10; fi
    done
    if (( ok )); then
      status="cloned"
    else
      status="clone-failed"
    fi
  else
    cur="$(git -C "${LIB}" rev-parse HEAD 2>/dev/null || true)"
    for attempt in 1 2; do
      if git -C "${LIB}" -c credential.helper= fetch -q --depth 1 origin "${LIB_REF}" \
           >> "${OUT}/pypto-lib.update.log" 2>&1; then
        ok=1
        break
      fi
      if (( attempt == 1 )); then sleep 10; fi
    done
    if (( ok )); then
      new="$(git -C "${LIB}" rev-parse FETCH_HEAD 2>/dev/null || true)"
      if [[ -n "${new}" && "${cur}" != "${new}" ]]; then
        if git -C "${LIB}" checkout -q -f "${new}" >> "${OUT}/pypto-lib.update.log" 2>&1; then
          status="updated"
        else
          # Keep the checkout and what we report on the same revision.
          git -C "${LIB}" checkout -q -f "${cur}" >> "${OUT}/pypto-lib.update.log" 2>&1 || true
          status="checkout-failed"
        fi
      fi
    else
      status="fetch-failed"
    fi
  fi
  printf '%s\n' "${status}" > "${OUT}/pypto-lib.update"
  echo "pypto-lib update: ${status}"
  git -C "${LIB}" log -1 --format='%H %ci' > "${OUT}/pypto-lib.head" 2>/dev/null || true
  return 0
}

# The list of runnable entries pypto-lib's own CI would run for this platform.
lib_selection_files() {
  local root="$1"
  ( cd "${LIB}" && "${PY}" .github/scripts/detect_changes.py --all --root "${root}" --platform a5 )
}

# pypto-lib's A5 entries are standalone scripts (`python <file> -p a5 -d N`),
# not pytest node ids. Emit the same "<case>\t<OK|FAIL|SKIP>" rows the summary
# code already counts, so both consumers share the reporting path.
#
# The CI runs every entry through `task-submit --device auto`; this host has no
# task-submit and reserves one card, so entries that declare more than one
# device are reported as unvalidated (with the reason) rather than dropped --
# a suite that silently shrinks reads as green.
run_lib_selection() {
  local key="$1" ptoas_root="$2" tag="$3"
  local group="${key#lib-a5-}"
  local tsv="${OUT}/${tag}-${key}.tsv"
  local not_run="${OUT}/${tag}-${key}.not-run.tsv"
  local log="${OUT}/${tag}-${key}.log"
  local filter="" rc=0 ndev st start dur f case_log
  local -a files=()

  if [[ "${tag}" == "baseline" && -f "${OUT}/candidate-${key}.tsv" ]]; then
    filter="$(awk -F'\t' '$2=="FAIL" {print $1}' "${OUT}/candidate-${key}.tsv")"
    [[ -n "${filter}" ]] || filter="__no_failures__"
  fi
  : > "${tsv}"
  stage_banner "${ptoas_root}" > "${log}"
  : > "${not_run}"
  read -r -a files <<< "$(lib_selection_files "${group}")"
  if (( ${#files[@]} == 0 )); then
    echo "WARN: ${key}: pypto-lib selected no ${group} entries" | tee -a "${log}"
  fi

  for f in "${files[@]}"; do
    if [[ -n "${filter}" ]]; then
      grep -qxF -- "${f}" <<< "${filter}" || continue
    fi
    ndev="$(grep -oP '#\s*ci:\s*devices=\K[0-9]+' "${LIB}/${f}" 2>/dev/null | head -1 || true)"
    ndev="${ndev:-1}"
    if (( ndev > 1 )); then
      printf '%s\tSKIP\n' "${f}" >> "${tsv}"
      printf '%s\tneeds %s cards, this host reserves device %s\n' \
        "${f}" "${ndev}" "${LIB_DEVICE}" >> "${not_run}"
      echo "[${key}] ${f}: SKIP (needs ${ndev} cards)" | tee -a "${log}"
      continue
    fi
    start="$(date +%s)"
    case_log="${OUT}/${tag}-${key}.$(printf '%s' "${f}" | tr '/' '_').log"
    # Repo-rooted imports (`from models.deepseek_v4_1_flash.config import ...`)
    # need the checkout on PYTHONPATH, exactly as the CI job runs it.
    if ( cd "${LIB}" &&
         PTOAS_ROOT="${ptoas_root}" TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
           PYTHONPATH="${LIB}:$(stage_pythonpath "${ptoas_root}")" \
           timeout 900 "${PY}" "${f}" -p a5 -d "${LIB_DEVICE}" ) > "${case_log}" 2>&1; then
      st="OK"
    elif grep -q "HcclCommInitRootInfo failed" "${case_log}" 2>/dev/null; then
      # The entries under prefill_* allocate a comm domain before any kernel
      # runs and abort in comm_init with Hccl error 19. pto-as is not involved
      # at that point, so this cannot be a pto-as regression: report the entry
      # as unvalidated with the error instead of turning an environment limit
      # into a nightly red (or a fake pypto-lib attribution).
      st="SKIP"
      printf '%s\tcomm domain unavailable in this run (HcclCommInitRootInfo failed: 19)\n' \
        "${f}" >> "${not_run}"
    else
      st="FAIL"
      rc=1
    fi
    dur=$(( $(date +%s) - start ))
    printf '%s\t%s\n' "${f}" "${st}" >> "${tsv}"
    echo "[${key}] ${f}: ${st} ${dur}s" | tee -a "${log}"
  done
  printf '%s\n' "${rc}" > "${OUT}/${tag}-${key}.rc"
  return 0
}

# Attribution baseline: the release PyPTO currently pins. Only switched when
# that version is already installed; otherwise keep the current baseline.
sync_baseline() {
  (( BASE_EXPLICIT )) && return 0
  local ver dir
  ver="$(sed -n 's/^PTOAS_VERSION=v\{0,1\}\([0-9][0-9.]*\)$/\1/p' \
    "${SRC}/toolchain/versions.env" 2>/dev/null | head -1)"
  [[ -n "${ver}" ]] || return 0
  dir="$(release_install_root "${ver}")"
  if [[ -x "${dir}/bin/ptoas" ]]; then
    BASE="${dir}"
  else
    echo "baseline: PyPTO pins ${ver} but ${dir} is not installed; keeping ${BASE}"
  fi
}

# The A5 board sweep owns card 2 for most of the night; wait it out instead of
# fighting it for the device. Cap the wait so a stuck run cannot hold us forever.
wait_for_units() {
  local units="${PTOAS_WAIT_FOR_UNITS:-}"
  local unit waited=0 budget=$((8 * 3600)) state
  [[ -n "${units}" ]] || return 0
  command -v systemctl >/dev/null 2>&1 || return 0
  # shellcheck disable=SC2086  # units is a deliberate whitespace-separated list
  for unit in ${units}; do
    # A oneshot unit reports "activating" while ExecStart runs, and
    # `is-active --quiet` is false for that state; compare the state directly.
    while true; do
      state="$(systemctl show -p ActiveState --value "${unit}" 2>/dev/null || true)"
      [[ "${state}" == "active" || "${state}" == "activating" ]] || break
      if (( waited >= budget )); then
        echo "WARN: ${unit} still ${state} after ${waited}s; proceeding"
        return 0
      fi
      [[ ${waited} -eq 0 ]] && echo "waiting for ${unit} to finish (state=${state})"
      sleep 60
      waited=$((waited + 60))
    done
    if (( waited > 0 )); then
      echo "${unit} finished after ${waited}s wait"
    fi
  done
}
wait_for_units

[[ -x "${PY}" ]] || fail "missing python: ${PY}"
[[ -d "${SRC}/tests/st" ]] || fail "missing pypto checkout: ${SRC}"

# shellcheck disable=SC1090
export MAMBA_ROOT_PREFIX="${ROOT}/mamba"
source <("${ROOT}/bin/micromamba" shell hook -s bash)
micromamba activate "${ENV}"
set +u
# 238's system CANN (9.1.0) predates the HCCL channel-config APIs that current
# pypto needs at build time, so allow a side-by-side SDK: point PTOAS_CANN_ENV
# at its set_env.sh. The default stays the system install, which the A5 board
# nightly keeps using.
CANN_ENV="${PTOAS_CANN_ENV:-/usr/local/Ascend/ascend-toolkit/latest/set_env.sh}"
if [[ -f "${CANN_ENV}" ]]; then
  # shellcheck disable=SC1090
  source "${CANN_ENV}"
fi
set -u
export TORCH_DEVICE_BACKEND_AUTOLOAD=0

update_pypto
sync_baseline
resolve_candidate

# CANN 9.2 ships its own ptoas (0.68) under python/site-packages and puts it on
# PYTHONPATH. pypto also imports that package for part of its toolchain, so the
# SDK copy shadows the release under test. Every stage therefore prepends the
# site-packages of *its own* root. Exporting the candidate's here instead made
# the attribution stage import and compile with the candidate, so a red
# selection came back "baseline also red" and was blamed on the downstream
# project -- see stage_pythonpath below.
stage_pythonpath() {
  local root="$1" site
  site="$(find "${root}/lib" -maxdepth 2 -type d -name site-packages 2>/dev/null | head -1)"
  printf '%s%s' "${site:+${site}:}" "${PYTHONPATH:-}"
}

# Record the compiler a stage actually used. `bin/ptoas` is a console script
# that imports `ptoas`, so the release that compiles is decided by PYTHONPATH,
# not by the shim's directory: probe under the same prefix the stage runs with,
# otherwise this reports the SDK's bundled copy instead.
stage_banner() {
  local root="$1"
  printf 'ptoas-root: %s\n%s\n' "${root}" \
    "$(PYTHONPATH="$(stage_pythonpath "${root}")" "${root}/bin/ptoas" --version 2>&1 | head -1)"
}

[[ -x "${CAND}/bin/ptoas" ]] || fail "missing candidate ptoas: ${CAND}/bin/ptoas"
# The shim imports `ptoas`, so the version it prints is the imported one: probe
# it under the candidate's own prefix or the card reports the SDK's copy.
PYTHONPATH="$(stage_pythonpath "${CAND}")" "${CAND}/bin/ptoas" --version \
  > "${OUT}/ptoas.version" 2>&1 ||
  fail "candidate ptoas does not run: ${CAND}/bin/ptoas"

# The A5 nightly refreshes the install tree; its mtime says which build this run
# actually exercised, which matters when that build failed part-way.
CAND_INSTALL="$(readlink -f "${CAND}/lib/.." 2>/dev/null || true)"
CAND_STAMP=""
if [[ -n "${CAND_INSTALL}" && -d "${CAND_INSTALL}" ]]; then
  CAND_STAMP="$(stat -c '%y' "${CAND_INSTALL}" 2>/dev/null | cut -d. -f1)"
fi

git -C "${SRC}" rev-parse HEAD > "${OUT}/pypto.commit"
git -C "${SRC}" log -1 --format='%H %ci' > "${OUT}/pypto.head"

# One selection run with a given ptoas root; writes junit xml and a log.
run_selection() {
  local key="$1" root="$2" tag="$3"
  shift 3
  local xml="${OUT}/${tag}-${key}.xml"
  local log="${OUT}/${tag}-${key}.log"
  local rc=0
  stage_banner "${root}" > "${log}"
  (
    cd "${SRC}"
    PTOAS_ROOT="${root}" PYTHONPATH="$(stage_pythonpath "${root}")" \
      "${PY}" -m pytest "$@" -q "--junitxml=${xml}"
  ) >>"${log}" 2>&1 || rc=$?
  echo "${rc}" > "${OUT}/${tag}-${key}.rc"
  return 0
}

run_named() {
  local key="$1" root="$2" tag="$3"
  case "${key}" in
    a5sim-core)
      run_selection "${key}" "${root}" "${tag}" "${SIM_A5_FILES[@]}" \
        --platform=a5sim --forked -k "not TestMscatter" ;;
    sim-fillpad)
      run_selection "${key}" "${root}" "${tag}" \
        tests/st/runtime/ops/test_fillpad_inplace.py \
        --memory-planner=ptoas --platform=a2a3sim,a5sim --forked ;;
    a5sim-crosscore)
      run_selection "${key}" "${root}" "${tag}" \
        tests/st/runtime/cross_core/test_cross_core.py \
        --platform=a5sim --forked ;;
    a2a3sim-cast)
      run_selection "${key}" "${root}" "${tag}" \
        tests/st/runtime/ops/test_cast_saturation.py \
        --platform=a2a3sim --forked ;;
    a5-board)
      run_selection "${key}" "${root}" "${tag}" "${A5_TESTS[@]}" \
        --platform=a5 "--device=${A5_DEVICE}" --forked ;;
    lib-a5-examples|lib-a5-models)
      run_lib_selection "${key}" "${root}" "${tag}" ;;
    *) fail "unknown selection: ${key}" ;;
  esac
}

# Convert junit xml to "<nodeid>\t<OK|FAIL|SKIP>" for the summary.
junit_tsv() {
  local xml="$1" out="$2"
  "${PY}" - "${xml}" "${out}" <<'PY'
import sys
import xml.etree.ElementTree as ET

src, dst = sys.argv[1], sys.argv[2]
rows = []
try:
    root = ET.parse(src).getroot()
except Exception:
    open(dst, "w", encoding="utf-8").close()
    raise SystemExit(0)
for tc in root.iter("testcase"):
    bad = tc.find("failure") is not None or tc.find("error") is not None
    skip = tc.find("skipped") is not None
    status = "FAIL" if bad else ("SKIP" if skip else "OK")
    rows.append("{}.{}\t{}".format(tc.get("classname", ""), tc.get("name", ""), status))
with open(dst, "w", encoding="utf-8") as fh:
    fh.write("\n".join(rows) + ("\n" if rows else ""))
PY
}

counts() {
  local tsv="$1"
  awk -F'\t' '{c[$2]++} END {printf "OK=%d FAIL=%d SKIP=%d", c["OK"]+0, c["FAIL"]+0, c["SKIP"]+0}' "${tsv}"
}

fails_of() {
  local tsv="$1"
  awk -F'\t' '$2=="FAIL" {print $1}' "${tsv}"
}

notify() {
  local text="$1"
  local notifier="${SCRIPT_DIR}/a5_notify.sh"
  [[ -f "${notifier}" ]] || return 0
  # shellcheck disable=SC1090
  source "${notifier}"
  a5_notify_text "${text}"
}

# The A3 host cannot be reached inbound, so it pushes a result file here before
# the card is built. A result counts for this card when it was produced after the
# previous card (so the A3 half can run at any convenient time, not only just
# before this one) and is not older than PTOAS_A3_MAX_AGE -- the age cap keeps a
# result that arrived after a card from being presented a day later.
a3_line() {
  local f="${PTOAS_A3_INCOMING:-${ROOT}/incoming/a3}/result.json"
  local marker="${ROOT}/state/a3.consumed"
  local max_age="${PTOAS_A3_MAX_AGE:-50400}"
  [[ -f "${f}" ]] || { echo "未就绪（无结果文件）"; return 0; }
  local age=$(( $(date +%s) - $(stat -c %Y "${f}") ))
  (( age <= max_age )) || { echo "未就绪（结果 ${age} 秒前，已过期）"; return 0; }
  if [[ -f "${marker}" && ! "${f}" -nt "${marker}" ]]; then
    echo "未就绪（结果早于上一张卡片）"
    return 0
  fi
  command -v python3 >/dev/null 2>&1 || { echo "未就绪（无 python3）"; return 0; }
  python3 - "${f}" <<'PY' 2>/dev/null || echo "未就绪（结果无法解析）"
import json
import sys

d = json.load(open(sys.argv[1]))
sel = "; ".join("{} {}".format(k, v) for k, v in d.get("selections", {}).items())
head = (d.get("pypto") or "").split(" ")[0][:8]
note = d.get("ptoas_note", "")
ver = (d.get("ptoas") or "?").split(" ", 1)
if len(ver) == 2 and ver[0] == "ptoas":
    ver = ver[1]
else:
    ver = ver[0]
print("{} {} | ptoas {} | pypto {} | {} | {}".format(
    d.get("verdict", "?"), sel, ver, head,
    d.get("finished_at", ""), note))
PY
}

# ---------------------------------------------------------------------------
# Execute
# ---------------------------------------------------------------------------
declare -A STATUS=()
declare -A COUNTS=()
declare -A TSV=()
RED=0

KEYS=()
if [[ "${STAGE}" == "sim" || "${STAGE}" == "all" ]]; then
  KEYS+=("${SIM_KEYS[@]}")
fi
if [[ "${STAGE}" == "a5" || "${STAGE}" == "all" ]]; then
  KEYS+=("${A5_KEYS[@]}")
fi
if [[ "${STAGE}" == "lib" ]]; then
  KEYS+=("${LIB_KEYS[@]}")
fi
if [[ -n "${ONLY}" ]]; then
  FILTERED=()
  IFS=',' read -r -a wanted <<< "${ONLY}"
  for w in "${wanted[@]}"; do
    found=0
    for k in "${KEYS[@]}"; do
      [[ "${k}" == "${w}" ]] && found=1
    done
    (( found )) || fail "selection ${w} is not enabled by --stage ${STAGE}"
    FILTERED+=("${w}")
  done
  KEYS=("${FILTERED[@]}")
fi
(( ${#KEYS[@]} )) || fail "no selections to run"

if [[ " ${KEYS[*]} " == *" lib-a5-"* ]]; then
  update_pypto_lib
  [[ -f "${LIB}/.github/scripts/detect_changes.py" ]] \
    || fail "pypto-lib checkout is not usable: ${LIB}"
  # An empty selection would report OK=0 as a green night -- exactly the
  # silent-shrink failure this gate exists to avoid. Refuse to run instead.
  for key in "${KEYS[@]}"; do
    [[ "${key}" == lib-a5-* ]] || continue
    [[ -n "$(lib_selection_files "${key#lib-a5-}")" ]] \
      || fail "pypto-lib ${key}: selection is empty; not reporting a green night"
  done
fi

for key in "${KEYS[@]}"; do
  echo "=== candidate ${key} ==="
  run_named "${key}" "${CAND}" candidate
  TSV[$key]="${OUT}/candidate-${key}.tsv"
  # lib selections already write their tsv; only the pytest ones need junit.
  if [[ "${key}" != lib-a5-* ]]; then
    junit_tsv "${OUT}/candidate-${key}.xml" "${TSV[$key]}"
  fi
  COUNTS[$key]="$(counts "${TSV[$key]}")"
  if [[ "$(cat "${OUT}/candidate-${key}.rc")" -ne 0 ]]; then
    STATUS[$key]="FAIL"; RED=1
  else
    STATUS[$key]="OK"
  fi
  echo "${key}: ${COUNTS[$key]} (${STATUS[$key]})"
done

# Attribution: re-run only the red selections with the known-good baseline.
# A selection can be red for two reasons at once -- a real pto-as regression
# next to a case that already failed on the baseline -- and a whole-selection
# verdict then turns one pre-existing failure into a pass for everything beside
# it. Compare the two runs case by case instead.
ATTR_REGRESSED=0
ATTR_PREEXISTING=0
ATTR_UNKNOWN=0
classify_cases() {
  local key="$1" id stat bstat
  local cand="${OUT}/candidate-${key}.tsv" base="${OUT}/baseline-${key}.tsv"
  local regressed="${OUT}/${key}.regressed.txt"
  : > "${regressed}"
  while IFS=$'\t' read -r id stat; do
    [[ "${stat}" == "FAIL" ]] || continue
    bstat=""
    [[ -f "${base}" ]] \
      && bstat="$(awk -F'\t' -v c="${id}" '$1 == c { print $2; exit }' "${base}")"
    case "${bstat}" in
      OK) ATTR_REGRESSED=$((ATTR_REGRESSED + 1)); printf '%s\n' "${id}" >> "${regressed}" ;;
      FAIL) ATTR_PREEXISTING=$((ATTR_PREEXISTING + 1)) ;;
      *) ATTR_UNKNOWN=$((ATTR_UNKNOWN + 1)) ;;
    esac
  done < "${cand}"
  if (( $(grep -c . "${regressed}") > 0 )); then
    echo "[${key}] pto-as 回归 $(grep -c . "${regressed}") 例（candidate 红、baseline 绿）"
  fi
}
VERDICT="GREEN"
ATTRIBUTION="未触发（candidate 全绿）"
if (( RED )); then
  if [[ -x "${BASE}/bin/ptoas" ]]; then
    for key in "${KEYS[@]}"; do
      [[ "${STATUS[$key]}" == "FAIL" ]] || continue
      echo "=== baseline ${key} ==="
      run_named "${key}" "${BASE}" baseline
      if [[ "${key}" != lib-a5-* ]]; then
        junit_tsv "${OUT}/baseline-${key}.xml" "${OUT}/baseline-${key}.tsv"
      fi
      # Compare regardless of the baseline exit code: one pre-existing failure
      # must not hide the regressions next to it.
      classify_cases "${key}"
    done
    if (( ATTR_REGRESSED > 0 )); then
      VERDICT="REGRESSION"
      ATTRIBUTION="candidate 红、baseline 绿 ${ATTR_REGRESSED} 例 → 归因 pto-as"
    elif (( ATTR_PREEXISTING > 0 )); then
      VERDICT="PYTPO_ISSUE"
      ATTRIBUTION="candidate 红、baseline 同样红 ${ATTR_PREEXISTING} 例 → ${DOWNSTREAM} 自身/环境问题"
    else
      VERDICT="UNATTRIBUTED"
      ATTRIBUTION="candidate 红，但基线未能判定（${ATTR_UNKNOWN} 例无法比较）"
    fi
    (( ATTR_PREEXISTING == 0 )) || ATTRIBUTION="${ATTRIBUTION}（另有 ${ATTR_PREEXISTING} 例 baseline 同红）"
    (( ATTR_UNKNOWN == 0 )) || ATTRIBUTION="${ATTRIBUTION}（另有 ${ATTR_UNKNOWN} 例无基线结果）"
  else
    VERDICT="UNATTRIBUTED"
    ATTRIBUTION="candidate 红，但未配置 baseline（${BASE}），无法归因"
  fi
fi

# pypto-lib's own CI deletes each job's build_output at the end; a nightly that
# keeps it would add ~160 MB per run to a shared /data volume. The per-case logs
# already captured what failed.
if [[ " ${KEYS[*]} " == *" lib-a5-"* && -d "${LIB}/build_output" ]]; then
  [[ -n "${LIB}" && "${LIB}" != "/" ]] || fail "refusing to prune an empty LIB path"
  rm -rf -- "${LIB}/build_output" || true
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
# The lib stage is not the release gate (it reports a different repository), so
# its verdict words say "pass/regression", not "ship/block the release".
if [[ "${STAGE}" == "lib" ]]; then
  T_GREEN="通过（candidate 全绿）"
  T_RED="阻塞（pto-as 回归）"
  T_ISSUE="不阻塞（${DOWNSTREAM} 自身/环境问题）"
else
  T_GREEN="可发版（candidate 全绿）"
  T_RED="阻塞发版（pto-as 回归）"
  T_ISSUE="不阻塞（PyPTO 自身/环境问题）"
fi

{
  echo "# ${PREFIX} ${RUN_ID}"
  echo
  case "${VERDICT}" in
    GREEN) echo "- 结论: ${T_GREEN}" ;;
    REGRESSION) echo "- 结论: ${T_RED}" ;;
    PYTPO_ISSUE) echo "- 结论: ${T_ISSUE}" ;;
    *) echo "- 结论: 无结论（未归因）" ;;
  esac
  echo "- candidate: $(tr -d '\n' < "${OUT}/ptoas.version")"
  echo "- release pin: ${RELEASE_VER:-override}${RELEASE_NOTE:+ (${RELEASE_NOTE})}"
  [[ -z "${RELEASE_NOTE}" ]] || echo "- candidate note: ${RELEASE_NOTE}"
  echo "- candidate install: ${CAND_INSTALL:-?} ${CAND_STAMP:-(mtime unavailable)}"
  echo "- baseline: ${BASE}"
  echo "- baseline compiler: $(PYTHONPATH="$(stage_pythonpath "${BASE}")" \
    "${BASE}/bin/ptoas" --version 2>&1 | head -1)"
  echo "- pypto: $(cat "${OUT}/pypto.head") ($(cat "${OUT}/pypto.update" 2>/dev/null || echo ?))"
  if [[ -f "${OUT}/pypto-lib.head" ]]; then
    echo "- pypto-lib: $(cat "${OUT}/pypto-lib.head") ($(cat "${OUT}/pypto-lib.update" 2>/dev/null || echo ?))"
  fi
  echo "- 归因: ${ATTRIBUTION}"
  echo
  for key in "${KEYS[@]}"; do
    echo "- ${key}: ${COUNTS[$key]} (${STATUS[$key]})"
  done
  for key in "${KEYS[@]}"; do
    not_run="${OUT}/candidate-${key}.not-run.tsv"
    if [[ -s "${not_run}" ]]; then
      while IFS=$'\t' read -r case reason; do
        echo "- 未验证(${key}): ${case} — ${reason}"
      done < "${not_run}"
    fi
  done
  if [[ "${STAGE}" != "lib" ]]; then
    echo "- A3(68.6): $(a3_line)"
  fi
  echo
  echo "日志: ${OUT}/"
} > "${OUT}/summary.md"

echo "--- summary ---"
cat "${OUT}/summary.md"

CARD="${PREFIX} ${RUN_ID}
"
case "${VERDICT}" in
  GREEN) CARD+="结论: ✅ ${T_GREEN}" ;;
  REGRESSION) CARD+="结论: ❌ ${T_RED}" ;;
  PYTPO_ISSUE) CARD+="结论: ⚠️ ${T_ISSUE}" ;;
  *) CARD+="结论: ⛔ 无结论（未归因）" ;;
esac
candidate_line="candidate: $(tr -d '\n' < "${OUT}/ptoas.version")"
candidate_line+="${RELEASE_VER:+ (release ${RELEASE_VER})}"
candidate_line+="${RELEASE_NOTE:+ [${RELEASE_NOTE}]}"
candidate_line+=" (${CAND_STAMP:-mtime unknown})"
CARD+="
${candidate_line}
pypto: $(cat "${OUT}/pypto.head") ($(cat "${OUT}/pypto.update" 2>/dev/null || echo ?))
"
if [[ -f "${OUT}/pypto-lib.head" ]]; then
  CARD+="pypto-lib: $(cat "${OUT}/pypto-lib.head") ($(cat "${OUT}/pypto-lib.update" 2>/dev/null || echo ?))
"
fi
CARD+="归因: ${ATTRIBUTION}
"
for key in "${KEYS[@]}"; do
  CARD+="${key}: ${COUNTS[$key]} (${STATUS[$key]})
"
done
for key in "${KEYS[@]}"; do
  not_run="${OUT}/candidate-${key}.not-run.tsv"
  if [[ -s "${not_run}" ]]; then
    while IFS=$'\t' read -r case reason; do
      CARD+="未验证(${key}): ${case} — ${reason}
"
    done < "${not_run}"
  fi
done
if [[ "${STAGE}" != "lib" ]]; then
  CARD+="A3(68.6): $(a3_line)
"
fi
# First few failing cases, one line each. When the baseline could separate them,
# list only the regressions: those are the cases someone has to act on, and the
# pre-existing ones are reported as a count in the attribution line.
for key in "${KEYS[@]}"; do
  [[ "${STATUS[$key]}" == "FAIL" ]] || continue
  if [[ -s "${OUT}/${key}.regressed.txt" ]]; then
    mapfile -t ff < <(head -3 "${OUT}/${key}.regressed.txt")
  else
    mapfile -t ff < <(fails_of "${TSV[$key]}" | head -3)
  fi
  (( ${#ff[@]} )) || continue
  CARD+="失败(${key}):
"
  for f in "${ff[@]}"; do CARD+="${f}
"; done
done
CARD+="日志: ${OUT}/"

printf '%s\n' "${CARD}" > "${OUT}/card.txt"
if (( NO_NOTIFY )); then
  echo "(notification skipped: --no-notify)"
else
  notify "${CARD}"
fi

printf '%s\n' "${OUT}" > "${ROOT}/${RESULT_NS}/latest.path"
ln -sfn "${OUT}" "${ROOT}/${RESULT_NS}/latest"
if [[ "${STAGE}" != "lib" ]]; then
  # Mark the A3 result as consumed: only a newer one may appear on the next card.
  mkdir -p "${ROOT}/state"
  : > "${ROOT}/state/a3.consumed"
fi
echo "RESULT_DIR=${OUT}"
