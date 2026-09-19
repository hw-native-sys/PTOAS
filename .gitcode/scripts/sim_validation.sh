#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set -euo pipefail

# Driver of the SIM_VALIDATION job. Master's workflow still runs its older
# VPTO_SIM / TileLib_ST / PTODSL_ST / PyPTO_SIM jobs against
# .gitcode/scripts/vpto_sim.sh, and their checkout is the pull request's merge
# preview, so that script has to keep behaving exactly as it does today while
# this merge request is in flight. This copy is the one the new job drives; it
# takes the vpto_sim.sh name back once those jobs are gone.
#
# Usage: sim_validation.sh [build|test|all] [workspace]
#   build  resolve the environment and build PTOAS/LLVM, then stop
#   test   run the selected suite against an existing build; never runs build.sh
#   all    build and then run the suite (default; a bare workspace path keeps it)
MODE="${1:-all}"
case "${MODE}" in
  build|test|all) shift ;;
  *) MODE="all" ;;
esac
WORKSPACE="${1:-${WORKSPACE:-$(pwd)}}"
WORKSPACE="$(cd "${WORKSPACE}" && pwd)"
BUILD_ROOT="${BUILD_ROOT:-${WORKSPACE}/.work/gitcode-vpto-sim}"
CANN_HOME="${CANN_HOME:-${ASCEND_HOME_PATH:-}}"
# CI images ship a prebuilt LLVM/MLIR cache alongside the CANN installation; the
# same root is used by .gitcode/scripts/build_and_test.sh and compile.sh. It is
# only accepted when it really contains llvm-19 (see the BUILD_ARGS selection
# below), so an image without the cache keeps building LLVM from source.
ASCEND_3RD_LIB_PATH="${ASCEND_3RD_LIB_PATH:-/home/jenkins/opensource}"
SIM_LIB_DIR="${SIM_LIB_DIR:-}"
CASE_PREFIX="${VPTO_SIM_CASE_PREFIX:-}"
# The runner confines jobs to its own service cgroup, so the limits that matter
# are the ones on the path /proc/self/cgroup reports. Reading the namespace
# root instead (/sys/fs/cgroup/memory.max) always reads "max", which let a
# capped runner size its fan-out for the whole host (observed as
# "194 GiB available" on a runner capped at 120 GiB).
CGROUP_ROOT="${CGROUP_ROOT:-/sys/fs/cgroup}"
SELF_CGROUP_FILE="${SELF_CGROUP_FILE:-/proc/self/cgroup}"

cgroup_rel_path() {
  local line path
  while IFS= read -r line; do
    case "${line}" in
      0::*) path="${line#0::}" ;;
    esac
  done < "${SELF_CGROUP_FILE}"
  printf "%s\n" "${path:-/}"
}

cgroup_file() {
  local candidate
  candidate="${CGROUP_ROOT}$(cgroup_rel_path)/$1"
  [[ -r "${candidate}" ]] && printf "%s\n" "${candidate}"
}

CPU_COUNT="$(nproc 2>/dev/null || printf '4')"
# A runner confined to a cgroup slice must not size its fan-out for the whole
# host: two runners on one machine would otherwise each assume every core.
CPU_MAX_FILE="$(cgroup_file cpu.max)"
if [[ -n "${CPU_MAX_FILE}" ]]; then
  read -r CPU_QUOTA_STR CPU_PERIOD_STR < "${CPU_MAX_FILE}"
  if [[ "${CPU_QUOTA_STR}" =~ ^[0-9]+$ && "${CPU_PERIOD_STR}" =~ ^[0-9]+$ && "${CPU_PERIOD_STR}" -gt 0 ]]; then
    CPU_QUOTA="$(( CPU_QUOTA_STR / CPU_PERIOD_STR ))"
    (( CPU_QUOTA >= 1 && CPU_QUOTA < CPU_COUNT )) && CPU_COUNT="${CPU_QUOTA}"
  fi
elif [[ -r /sys/fs/cgroup/cpu/cpu.cfs_quota_us && -r /sys/fs/cgroup/cpu/cpu.cfs_period_us ]]; then
  read -r CPU_QUOTA_STR < /sys/fs/cgroup/cpu/cpu.cfs_quota_us
  read -r CPU_PERIOD_STR < /sys/fs/cgroup/cpu/cpu.cfs_period_us
  if [[ "${CPU_QUOTA_STR}" =~ ^[0-9]+$ && "${CPU_PERIOD_STR}" =~ ^[0-9]+$ && "${CPU_PERIOD_STR}" -gt 0 ]]; then
    CPU_QUOTA="$(( CPU_QUOTA_STR / CPU_PERIOD_STR ))"
    (( CPU_QUOTA >= 1 && CPU_QUOTA < CPU_COUNT )) && CPU_COUNT="${CPU_QUOTA}"
  fi
fi
BUILD_JOBS="${BUILD_JOBS:-${CPU_COUNT}}"
# Concurrency follows the GitHub simulator job (32 VPTO cases, 32 smoke build
# jobs, 64 TileLang cases) but is additionally bounded by the memory the job may
# use. A single simulator case peaks around 7 GiB, so an unbounded fan-out makes
# the kernel OOM killer take down the cases and, on a shared machine, other
# users' processes as well (runs #438, #430 and the first self-hosted run).
SIM_MEMORY_BUDGET_GIB="${SIM_MEMORY_BUDGET_GIB:-16}"
VPTO_CASE_JOBS="${VPTO_SIM_JOBS:-32}"
TILELIB_BUILD_JOBS="${TILELIB_BUILD_JOBS:-32}"
TILELIB_CASE_JOBS="${TILELIB_CASE_JOBS:-64}"
# Camodel ESL worker threads per simulator instance; empty means "derive it
# from the fan-out" (see prepare_camodel_runtime).
CAMODEL_THREADS="${CAMODEL_THREADS:-}"
SIM_PYTHON_BIN="${SIM_PYTHON_BIN:-${CI_SIM_PYTHON_BIN:-}}"
PYPTO_REF="${PYPTO_REF:-ef6ce7cd8bd33b4c93b58dc34830a20b145736ac}"
PTO_ISA_COMMIT="${PTO_ISA_COMMIT:-927253b784344d4ecbed524fd44737097c2bc3e0}"
# Upstream sources; an environment that cannot reach github can provide a
# local checkout (PTO_ISA_SOURCE_DIR/PYPTO_SOURCE_DIR) or an internal mirror.
PTO_ISA_REPO_URL="${PTO_ISA_REPO_URL:-https://github.com/hw-native-sys/pto-isa.git}"
PYPTO_REPO_URL="${PYPTO_REPO_URL:-https://github.com/hw-native-sys/pypto.git}"
PTO_ISA_SOURCE_DIR="${PTO_ISA_SOURCE_DIR:-}"
PYPTO_SOURCE_DIR="${PYPTO_SOURCE_DIR:-}"
PYPTO_WORKSPACE="${PYPTO_WORKSPACE:-${BUILD_ROOT}/pypto-ci}"
PYPTO_RUN_WORKSPACE="${PYPTO_RUN_WORKSPACE:-${BUILD_ROOT}/pypto-run}"
PTO_ISA_ROOT="${PTO_ISA_ROOT:-${BUILD_ROOT}/pto-isa-ci}"
SUITE_STATUS=0
SIM_SUITE="${SIM_SUITE:-all}"

mkdir -p "${BUILD_ROOT}"
exec > >(tee "${BUILD_ROOT}/vpto-sim.log") 2>&1

# Observation phase: this gate validates nothing yet, and it can never fail the
# job it runs in. A pull request pipeline executes the workflow of the *default*
# branch, so the first real run of this script happens on somebody else's merge
# request - an unfinished runner or a missing simulator must not fail that merge
# request while the setup is still being proven on real pull requests. The
# announcement below lands in the main log through the redirection above; the
# remaining artifacts are the ones the workflow uploads from an always() step,
# and obs-upload is not known to tolerate a missing path, so every one of them
# has to exist even though nothing ran. Delete this block - nothing else, here
# or in the workflow - to turn the simulator gate on.
OBSERVATION_NOTE="Observation phase: the GitCode simulator gate does not enforce yet; no validation was executed."
echo "${OBSERVATION_NOTE}"
mkdir -p "${BUILD_ROOT}/cases"
for placeholder in \
  "${BUILD_ROOT}/tilelib-st.log" \
  "${BUILD_ROOT}/ptodsl-dsl-st.log" \
  "${BUILD_ROOT}/pypto-observation.log" \
  "${BUILD_ROOT}/cases/parallel-runner.log" \
  "${BUILD_ROOT}/cases/parallel-summary.tsv"
do
  printf '%s\n' "${OBSERVATION_NOTE}" > "${placeholder}"
done
echo "Placeholder artifacts written under ${BUILD_ROOT}; nothing was validated."
exit 0

skip_simulator() {
  echo "::warning::Skipping ${SIM_SUITE} simulator validation: $1"
  echo "Simulator environment is unavailable; no validation was executed."
  exit 0
}

# Resolve the CANN installation the way the GitHub simulator job does, so the
# script works both on the shared image (which exports CANN_HOME) and on a
# self-hosted runner whose CANN tree lives elsewhere, typically /usr/local.
resolve_ascend_home() {
  local candidate
  for candidate in \
    "${CANN_HOME}" \
    "${ASCEND_HOME_PATH:-}" \
    /usr/local/Ascend/cann \
    /usr/local/Ascend/cann-* \
    /usr/local/Ascend/ascend-toolkit/latest \
    /usr/local/CANN/cann \
    /usr/local/CANN/cann-* \
    /usr/local/CANN/ascend-toolkit/latest \
    /home/jenkins/Ascend/cann-*
  do
    [[ -n "${candidate}" && -d "${candidate}" ]] || continue
    [[ -f "${candidate}/bin/setenv.bash" || -f "${candidate}/set_env.sh" ]] || continue
    printf '%s\n' "${candidate}"
    return 0
  done
  return 1
}

CANN_HOME="$(resolve_ascend_home)" || {
  skip_simulator "no CANN installation found; set CANN_HOME or ASCEND_HOME_PATH"
}
export CANN_HOME

if [[ -f "${CANN_HOME}/bin/setenv.bash" ]]; then
  # shellcheck disable=SC1091
  source "${CANN_HOME}/bin/setenv.bash"
elif [[ -f "${CANN_HOME}/set_env.sh" ]]; then
  # shellcheck disable=SC1091
  source "${CANN_HOME}/set_env.sh"
fi

ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-${CANN_HOME}}"

[[ -n "${ASCEND_HOME_PATH:-}" && -d "${ASCEND_HOME_PATH}" ]] || {
  skip_simulator "ASCEND_HOME_PATH is unavailable: ${ASCEND_HOME_PATH:-<unset>}"
}

BISHENG_BIN="${BISHENG_BIN:-${ASCEND_HOME_PATH}/bin/bisheng}"
MSPROF_BIN="${MSPROF_BIN:-${ASCEND_HOME_PATH}/bin/msprof}"
command -v "${BISHENG_BIN}" >/dev/null 2>&1 || {
  skip_simulator "bisheng is unavailable: ${BISHENG_BIN}"
}
command -v "${MSPROF_BIN}" >/dev/null 2>&1 || {
  skip_simulator "msprof is unavailable: ${MSPROF_BIN}"
}

if [[ -n "${SIM_LIB_DIR}" ]]; then
  [[ -d "${SIM_LIB_DIR}" ]] || skip_simulator "SIM_LIB_DIR is invalid: ${SIM_LIB_DIR}"
else
  readarray -t SIM_LIB_DIRS < <(
    find "${ASCEND_HOME_PATH}" -type d -path '*/simulator/dav_3510/camodel' 2>/dev/null | sort
  )
  if [[ "${#SIM_LIB_DIRS[@]}" -eq 0 ]]; then
    readarray -t SIM_LIB_DIRS < <(
      find "${ASCEND_HOME_PATH}" -type d -path '*/simulator/dav_3510/lib' 2>/dev/null | sort
    )
  fi
  if [[ "${#SIM_LIB_DIRS[@]}" -eq 0 ]]; then
    skip_simulator "dav_3510 simulator runtime is unavailable under ${ASCEND_HOME_PATH}"
  fi
  SIM_LIB_DIR="${SIM_LIB_DIRS[0]}"
fi

prepare_camodel_runtime() {
  [[ "${CAMODEL_THREADS}" =~ ^[0-9]+$ && "${CAMODEL_THREADS}" -ge 1 ]] || {
    echo "ERROR: CAMODEL_THREADS must be a positive integer, got: ${CAMODEL_THREADS}" >&2
    exit 2
  }
  [[ "${SIM_LIB_DIR}" == */camodel ]] || return 0

  local config="${SIM_LIB_DIR}/camodel_v100.json"

  local overlay_dir="${BUILD_ROOT}/camodel-runtime"
  mkdir -p "${overlay_dir}"
  local entry name
  for entry in "${SIM_LIB_DIR}"/*; do
    name="${entry##*/}"
    [[ "${name}" == "camodel_v100.json" ]] && continue
    if [[ ! -e "${overlay_dir}/${name}" && ! -L "${overlay_dir}/${name}" ]]; then
      ln -s "${entry}" "${overlay_dir}/${name}"
    fi
  done
  # CANN 9.0.0 does not ship camodel_v100.json at all, and the stock runtime
  # then keeps its own thread count (32 per instance on this host). A case that
  # takes seconds with the count capped takes minutes without it, so write the
  # sysconf the simulator expects instead of silently running the slow default.
  if [[ -f "${config}" ]]; then
    sed "s/\"esl_top_thread_cnt\": [0-9][0-9]*/\"esl_top_thread_cnt\": ${CAMODEL_THREADS}/" \
      "${config}" > "${overlay_dir}/camodel_v100.json"
  else
    printf '{\n    "sysconf": {\n        "esl_top_thread_cnt": %s\n    }\n}\n' \
      "${CAMODEL_THREADS}" > "${overlay_dir}/camodel_v100.json"
  fi
  SIM_LIB_DIR="${overlay_dir}"
  echo "Camodel thread count: ${CAMODEL_THREADS}"
}

# The simulator cases import torch and torch_npu, and the LLVM/MLIR python
# bindings only load in the interpreter version they were built for. Pick an
# interpreter that provides torch, wrap it in a CI venv that inherits
# site-packages (mirroring the GitHub simulator job), and use it for the build
# and for the cases alike.
python_has_torch() {
  TORCH_DEVICE_BACKEND_AUTOLOAD=0 "$1" - <<'PY' >/dev/null 2>&1
import importlib.util
missing = [name for name in ("torch", "torch_npu")
           if importlib.util.find_spec(name) is None]
raise SystemExit(1 if missing else 0)
PY
}

# The environment declares the interpreter: SIM_PYTHON_BIN when the runner has a
# dedicated one, otherwise python3 from PATH. The script never searches the host
# for a suitable interpreter; it only verifies the declared one.
resolve_simulator_python() {
  if [[ -n "${SIM_PYTHON_BIN}" ]]; then
    printf '%s\n' "${SIM_PYTHON_BIN}"
    return 0
  fi
  command -v python3 2>/dev/null
}

setup_simulator_python() {
  local base_python
  base_python="$(resolve_simulator_python)" || {
    echo "ERROR: no python3 on PATH; set SIM_PYTHON_BIN to the simulator interpreter." >&2
    exit 2
  }
  python_has_torch "${base_python}" || {
    echo "ERROR: ${base_python} cannot import torch and torch_npu." >&2
    echo "ERROR: point SIM_PYTHON_BIN at the interpreter this environment provides." >&2
    exit 2
  }
  export PYTHON_BIN="${base_python}"
  export PTO_PYTHON_BIN="${PYTHON_BIN}"
  # build.sh selects its interpreter with `command -v python3`, so put the
  # declared one first on PATH and use it unchanged everywhere else.
  local python_dir
  python_dir="$(dirname "${base_python}")"
  [[ -x "${python_dir}/python3" ]] || {
    echo "WARNING: ${python_dir} has no python3; build.sh may pick another interpreter" >&2
  }
  export PATH="${python_dir}:${PATH}"
  PYTHON_TAG="$("${PYTHON_BIN}" -c 'import sys; print("py%d%d" % sys.version_info[:2])')"
  echo "Simulator python: ${PYTHON_BIN} (${PYTHON_TAG})"
}

setup_simulator_python

echo "CANN environment: ${ASCEND_HOME_PATH}"
echo "Bisheng: ${BISHENG_BIN}"
echo "msprof: ${MSPROF_BIN}"
echo "SIM_LIB_DIR: ${SIM_LIB_DIR}"

# Bisheng consumes this environment-provided contract for host-side parsing.
# Derive it from the active gcc installation only when the environment has not
# supplied a value; no distro- or image-specific path is encoded here.
if [[ -z "${BISHENG_FLAGS:-}" ]] && command -v gcc >/dev/null 2>&1; then
  gcc_install_dir="$(gcc -print-search-dirs | sed -n 's/^install: //p')"
  gcc_toolchain="${gcc_install_dir%%/lib/gcc/*}"
  if [[ -n "${gcc_toolchain}" && -d "${gcc_toolchain}" ]]; then
    export BISHENG_FLAGS="--gcc-toolchain=${gcc_toolchain}"
    echo "Bisheng toolchain: ${gcc_toolchain} (derived from active gcc)"
  fi
fi

# Memory available to this job: the cgroup limit inside a container, otherwise
# the host's available memory.
available_memory_kib() {
  local limit file
  file="$(cgroup_file memory.max)"
  [[ -n "${file}" ]] && limit="$(cat "${file}" 2>/dev/null || true)"
  if [[ ! "${limit}" =~ ^[0-9]+$ ]]; then
    limit="$(cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null || true)"
    [[ "${limit}" =~ ^[0-9]+$ ]] && (( limit < 1125899906842624 )) || limit=""
  fi
  if [[ -n "${limit}" ]]; then
    printf '%s\n' "$(( limit / 1024 ))"
    return 0
  fi
  awk '/^MemAvailable:/ { print $2 }' /proc/meminfo 2>/dev/null
}

# Clamp a requested fan-out to what the memory budget allows and report it.
cap_jobs_to_memory() {
  local name="$1" value="$2"
  if (( value > MEMORY_JOBS )); then
    echo "Capping ${name} to ${MEMORY_JOBS} (requested ${value});" \
      "budget ${SIM_MEMORY_BUDGET_GIB} GiB per process, ${MEMORY_AVAIL_GIB} GiB available" >&2
    value="${MEMORY_JOBS}"
  fi
  printf '%s\n' "${value}"
}

[[ "${SIM_MEMORY_BUDGET_GIB}" =~ ^[0-9]+$ && "${SIM_MEMORY_BUDGET_GIB}" -ge 1 ]] || {
  echo "ERROR: SIM_MEMORY_BUDGET_GIB must be a positive integer, got: ${SIM_MEMORY_BUDGET_GIB}" >&2
  exit 2
}
MEMORY_AVAIL_KIB="$(available_memory_kib)"
if [[ "${MEMORY_AVAIL_KIB}" =~ ^[0-9]+$ ]] && (( MEMORY_AVAIL_KIB > 0 )); then
  MEMORY_AVAIL_GIB="$(( MEMORY_AVAIL_KIB / 1048576 ))"
else
  MEMORY_AVAIL_KIB=0
  MEMORY_AVAIL_GIB=0
fi
MEMORY_JOBS="$(( MEMORY_AVAIL_KIB / (SIM_MEMORY_BUDGET_GIB * 1048576) ))"
(( MEMORY_JOBS < 1 )) && MEMORY_JOBS=1

[[ "${BUILD_JOBS}" =~ ^[0-9]+$ && "${BUILD_JOBS}" -ge 1 ]] || {
  echo "ERROR: BUILD_JOBS must be a positive integer, got: ${BUILD_JOBS}" >&2
  exit 2
}
[[ "${VPTO_CASE_JOBS}" =~ ^[0-9]+$ && "${VPTO_CASE_JOBS}" -ge 1 ]] || {
  echo "ERROR: VPTO_SIM_JOBS must be a positive integer, got: ${VPTO_CASE_JOBS}" >&2
  exit 2
}
[[ "${TILELIB_BUILD_JOBS}" =~ ^[0-9]+$ && "${TILELIB_BUILD_JOBS}" -ge 1 ]] || {
  echo "ERROR: TILELIB_BUILD_JOBS must be a positive integer, got: ${TILELIB_BUILD_JOBS}" >&2
  exit 2
}
[[ "${TILELIB_CASE_JOBS}" =~ ^[0-9]+$ && "${TILELIB_CASE_JOBS}" -ge 1 ]] || {
  echo "ERROR: TILELIB_CASE_JOBS must be a positive integer, got: ${TILELIB_CASE_JOBS}" >&2
  exit 2
}
VPTO_CASE_JOBS="$(cap_jobs_to_memory VPTO_SIM_JOBS "${VPTO_CASE_JOBS}")"
TILELIB_BUILD_JOBS="$(cap_jobs_to_memory TILELIB_BUILD_JOBS "${TILELIB_BUILD_JOBS}")"
TILELIB_CASE_JOBS="$(cap_jobs_to_memory TILELIB_CASE_JOBS "${TILELIB_CASE_JOBS}")"
echo "Memory budget: ${MEMORY_AVAIL_GIB} GiB available," \
  "${SIM_MEMORY_BUDGET_GIB} GiB per process -> at most ${MEMORY_JOBS} parallel"
echo "VPTO case concurrency: ${VPTO_CASE_JOBS}"
echo "TileLib concurrency: build=${TILELIB_BUILD_JOBS} cases=${TILELIB_CASE_JOBS}"

# Every simulator instance starts its own ESL worker pool, and the stock config
# asks for 32 of them. A 12 way case fan-out then puts 384 threads on a 64 core
# host: the same suite that needed 30 minutes took 4h46m, because a single case
# only saturates about three cores anyway (20 s CPU for 7.6 s wall) while the
# extra threads fight for the CPU. Give the fan-out half of the cores, which is
# also what the measured baseline run used, unless CAMODEL_THREADS says otherwise.
if [[ -z "${CAMODEL_THREADS}" ]]; then
  WIDEST_FAN_OUT="${VPTO_CASE_JOBS}"
  if [[ "${TILELIB_CASE_JOBS}" -gt "${WIDEST_FAN_OUT}" ]]; then
    WIDEST_FAN_OUT="${TILELIB_CASE_JOBS}"
  fi
  CAMODEL_THREADS="$(( CPU_COUNT / 2 / WIDEST_FAN_OUT ))"
  (( CAMODEL_THREADS < 1 )) && CAMODEL_THREADS=1
  (( CAMODEL_THREADS > 4 )) && CAMODEL_THREADS=4
fi
prepare_camodel_runtime

if [[ "${MODE}" != "test" ]]; then
  BUILD_ARGS=(--build -j "${BUILD_JOBS}")
  if [[ -n "${ASCEND_3RD_LIB_PATH:-}" && -d "${ASCEND_3RD_LIB_PATH}/llvm-19" ]]; then
    echo "Using LLVM cache: ${ASCEND_3RD_LIB_PATH}"
    BUILD_ARGS+=(--cann_3rd_lib_path "${ASCEND_3RD_LIB_PATH}")
  else
    echo "No LLVM cache configured; build.sh will use its default third-party workspace."
  fi
  # The cache root is shared: reading it concurrently is fine, but build.sh
  # deletes and rebuilds the keyed LLVM build directory when it deems the cache
  # unusable, which would pull the ground out from under another job. Serialize
  # only that phase (about a minute in the steady state).
  build_lock="${ASCEND_3RD_LIB_PATH:-}/.ptoas-build.lock"
  if command -v flock >/dev/null 2>&1 && [ -n "${ASCEND_3RD_LIB_PATH:-}" ] &&
    [ -w "${ASCEND_3RD_LIB_PATH}" ]; then
    echo "Serializing the build on the shared cache: ${build_lock}"
    flock -w "${SIM_BUILD_LOCK_TIMEOUT:-3600}" "${build_lock}" \
      bash "${WORKSPACE}/build.sh" "${BUILD_ARGS[@]}"
  else
    bash "${WORKSPACE}/build.sh" "${BUILD_ARGS[@]}"
  fi
else
  echo "Test mode: reusing the PTOAS build in ${WORKSPACE}/build; build.sh is not run."
fi

if [[ -f "${WORKSPACE}/build/ptoas-test-env.sh" ]]; then
  # shellcheck disable=SC1091
  source "${WORKSPACE}/build/ptoas-test-env.sh"
elif [[ "${MODE}" == "test" ]]; then
  echo "ERROR: no PTOAS build in ${WORKSPACE}/build;" \
    "run '$(basename "${BASH_SOURCE[0]}") build ${WORKSPACE}' first." >&2
  exit 2
fi

# The build reuses whatever LLVM/MLIR tree the cache holds, but its python
# bindings only load in the interpreter they were built with. Verify that here so
# a mismatch names itself instead of failing every source-backed case later.
if [[ -n "${LLVM_BUILD_DIR:-}" && -d "${LLVM_BUILD_DIR}/tools/mlir/python_packages/mlir_core" ]]; then
  if ! PYTHONPATH="${LLVM_BUILD_DIR}/tools/mlir/python_packages/mlir_core${PYTHONPATH:+:${PYTHONPATH}}" \
    TORCH_DEVICE_BACKEND_AUTOLOAD=0 "${PYTHON_BIN}" -c 'import mlir.ir' >/dev/null 2>&1; then
    echo "ERROR: the cached LLVM/MLIR tree at ${LLVM_BUILD_DIR} was built for another Python." >&2
    echo "ERROR: point LLVM_BUILD_DIR at a tree built with ${PYTHON_TAG}, or remove that tree." >&2
    exit 2
  fi
  echo "LLVM/MLIR python bindings match ${PYTHON_TAG}"
fi

PTOAS_BIN="${PTOAS_BIN:-${WORKSPACE}/build/tools/ptoas/ptoas}"
[[ -x "${PTOAS_BIN}" ]] || {
  echo "ERROR: built ptoas is unavailable: ${PTOAS_BIN}" >&2
  exit 1
}

export PTOAS_BIN ASCEND_HOME_PATH SIM_LIB_DIR DEVICE=SIM JOBS="${VPTO_CASE_JOBS}"
export WORK_SPACE="${BUILD_ROOT}/cases"
export CASES_ROOT="${WORKSPACE}/test/vpto/cases"
export PATH="$(dirname "${PTOAS_BIN}"):${PATH}"

if [[ -n "${CASE_PREFIX}" ]]; then
  export CASE_PREFIX
fi

run_suite() {
  local suite_name="$1"
  shift
  echo "=== ${suite_name} ==="
  set +e
  "$@"
  local status=$?
  set -e
  if [[ "${status}" -ne 0 ]]; then
    echo "${suite_name}: FAILED (${status})"
    SUITE_STATUS=1
  else
    echo "${suite_name}: PASSED"
  fi
}

run_vpto() {
  bash "${WORKSPACE}/test/vpto/scripts/run_host_vpto_validation_parallel.sh"
}

run_tilelib_st() {
  mkdir -p "${BUILD_ROOT}/tilelib-st-a5"
  [[ -f "${PTO_ISA_ROOT}/include/pto/pto-inst.hpp" ]] || {
    echo "ERROR: pto-isa is not prepared at ${PTO_ISA_ROOT}; run the build phase first." >&2
    return 1
  }
  LLVM_BUILD_DIR="${LLVM_BUILD_DIR:-${WORKSPACE}/build}" \
  ASCEND_HOME_PATH="${ASCEND_HOME_PATH}" \
  PYTHON_BIN="${PYTHON_BIN:-python3}" \
  PTO_ISA_ROOT="${PTO_ISA_ROOT}" \
  PTOAS_BIN="${PTOAS_BIN}" \
    bash "${WORKSPACE}/test/tilelang_st/script/run_ci.sh" -r sim -v a5 \
      --build-jobs "${TILELIB_BUILD_JOBS}" --jobs "${TILELIB_CASE_JOBS}" --smoke \
      2>&1 | tee "${BUILD_ROOT}/tilelib-st.log"
}

prepare_pto_isa() {
  command -v git >/dev/null 2>&1 || return 1
  if [[ ! -f "${PTO_ISA_ROOT}/include/pto/pto-inst.hpp" ]]; then
    materialize_repo "${PTO_ISA_REPO_URL}" "${PTO_ISA_SOURCE_DIR}" "${PTO_ISA_ROOT}" || return 1
    # A copied local source already carries the pin; only reach out over the
    # network when it does not.
    if ! git -C "${PTO_ISA_ROOT}" cat-file -e "${PTO_ISA_COMMIT}^{commit}" 2>/dev/null; then
      git_retry -C "${PTO_ISA_ROOT}" fetch --depth 1 origin "${PTO_ISA_COMMIT}" || return 1
    fi
    git_retry -C "${PTO_ISA_ROOT}" checkout --force "${PTO_ISA_COMMIT}" || return 1
  fi
  [[ -f "${PTO_ISA_ROOT}/include/pto/pto-inst.hpp" ]]
}

run_ptodsl_st() {
  ASCEND_HOME_PATH="${ASCEND_HOME_PATH}" PTOAS_BIN="${PTOAS_BIN}" \
    bash "${WORKSPACE}/scripts/sim_dsl.sh" "${WORKSPACE}/test/dsl-st" \
      2>&1 | tee "${BUILD_ROOT}/ptodsl-dsl-st.log"
}

# The install targets the interpreter, so its marker belongs next to that
# interpreter rather than in the per run workspace: two runners on one host
# share the interpreter, and without this each of them would rebuild the wheel
# and write to the same site-packages on every run. The marker name carries the
# pin, so a new pin installs again.
pypto_marker_path() {
  local prefix
  prefix="$(cd "$(dirname "${PYTHON_BIN:-python3}")/.." 2>/dev/null && pwd)" || prefix=""
  if [[ -n "${prefix}" && -w "${prefix}" ]]; then
    printf "%s\n" "${prefix}/.ptoas-pypto-${PYPTO_REF}"
  else
    printf "%s\n" "${BUILD_ROOT}/pypto-prepared"
  fi
}

# Install the frontend and the runtime into the interpreter, under a lock so
# that concurrent runs cannot interleave writes into site-packages.
install_pypto_packages() {
  local python_bin="$1"
  local -a pip_index=()
  if [[ -n "${SIM_PIP_INDEX_URL:-}" ]]; then
    pip_index=(-i "${SIM_PIP_INDEX_URL}")
  fi
  # These only have to be present, not fresh: an unreachable index must not
  # fail a build whose toolchain is already installed.
  "${python_bin}" -m pip install --no-input "${pip_index[@]}" --upgrade "pip>=22.1" ||
    echo "WARNING: could not refresh pip; keeping the installed version" >&2
  "${python_bin}" -m pip install --no-input "${pip_index[@]}" \
    "scikit-build-core>=0.10.0" \
    "nanobind>=2.0.0" \
    "ninja>=1.11.0" \
    "cmake>=3.15" ||
    echo "WARNING: could not refresh the PyPTO build helpers" >&2

  # A reused workspace still holds the build trees of the previous toolchain.
  rm -rf "${PYPTO_WORKSPACE}/build" "${PYPTO_WORKSPACE}/_skbuild" \
    "${PYPTO_WORKSPACE}/runtime/build"
  "${python_bin}" -m pip install --no-build-isolation --no-deps \
    "${PYPTO_WORKSPACE}" || return 1
  # ci_sim only runs PyPTO simulator tests, so keep the runtime install from
  # auto-detecting onboard platforms from the runner CANN environment.
  env -u ASCEND_HOME_PATH \
    "${python_bin}" -m pip install --no-build-isolation --no-deps \
    "${PYPTO_WORKSPACE}/runtime" || return 1
  verify_pypto_install
}

# Everything PyPTO needs before its tests can run: the checkout with its
# submodules, pto-isa, the compiler its runtime builds kernels with, and the
# installed frontend/runtime packages.
prepare_pypto() {
  command -v git >/dev/null 2>&1 || return 1

  local python_bin="${PYTHON_BIN:-python3}"
  local marker
  marker="$(pypto_marker_path)"
  if [[ -f "${marker}" && -d "${PYPTO_WORKSPACE}/.git" ]] &&
    "${python_bin}" -m pip show pypto >/dev/null 2>&1; then
    echo "PyPTO ${PYPTO_REF} is already prepared; reusing ${PYPTO_WORKSPACE}."
    ensure_pypto_gcc15 || return 1
    prepare_pto_isa || return 1
    mkdir -p "${PYPTO_RUN_WORKSPACE}"
    return 0
  fi

  if [[ ! -d "${PYPTO_WORKSPACE}/.git" ]]; then
    if [[ -n "${PYPTO_SOURCE_DIR}" && -d "${PYPTO_SOURCE_DIR}/.git" ]]; then
      materialize_repo "${PYPTO_REPO_URL}" "${PYPTO_SOURCE_DIR}" "${PYPTO_WORKSPACE}" || return 1
    else
      git_retry clone --depth 1 --recurse-submodules --shallow-submodules \
        "${PYPTO_REPO_URL}" "${PYPTO_WORKSPACE}" || return 1
    fi
  fi
  if ! git -C "${PYPTO_WORKSPACE}" checkout --force "${PYPTO_REF}" 2>/dev/null; then
    git_retry -C "${PYPTO_WORKSPACE}" fetch --depth 1 origin "${PYPTO_REF}" || return 1
    git_retry -C "${PYPTO_WORKSPACE}" checkout --force "${PYPTO_REF}" || return 1
  fi
  # The submodules come from github too; a copied source already has them, so
  # warn instead of failing when they cannot be refreshed.
  git_retry -C "${PYPTO_WORKSPACE}" submodule update --init --recursive --depth 1 ||
    echo "WARNING: could not refresh the PyPTO submodules; using the ones at hand" >&2
  prepare_pto_isa || return 1
  ensure_pypto_gcc15 || return 1

  # Lock only around the install: a subshell keeps the lock scoped and still
  # passes the unexported variables the install needs.
  local lock_file="${marker}.lock"
  if command -v flock >/dev/null 2>&1; then
    ( flock -w "${SIM_PREPARE_LOCK_TIMEOUT:-1800}" 9 &&
      install_pypto_packages "${python_bin}" ) 9>"${lock_file}" || return 1
  else
    install_pypto_packages "${python_bin}" || return 1
  fi
  # The frontend and runtime build trees are around 1.5 GiB and are only needed
  # while installing: the next run reuses the installation through the marker, so
  # drop them instead of leaving them in the runner workspace for good.
  rm -rf "${PYPTO_WORKSPACE}/build" "${PYPTO_WORKSPACE}/_skbuild" \
    "${PYPTO_WORKSPACE}/runtime/build"
  mkdir -p "${PYPTO_RUN_WORKSPACE}"
  printf '%s\n' "${PYPTO_REF}" > "${marker}"
}

# PyPTO's simulator runtime compiles its kernels with GCC/G++ 15, which no
# distribution ships as the default compiler. Use the host pair when present and
# otherwise fall back to the conda cross toolchain; either way gcc-15/g++-15 has
# to be callable by name.
ensure_pypto_gcc15() {
  compiler_major() { "$1" -dumpfullversion -dumpversion | cut -d. -f1; }

  if command -v g++-15 >/dev/null 2>&1; then
    [[ "$(compiler_major "$(command -v g++-15)")" == "15" ]] || {
      echo "ERROR: g++-15 exists but is not GCC 15: $(g++-15 --version | head -1)" >&2
      return 1
    }
    command -v gcc-15 >/dev/null 2>&1 || {
      echo "ERROR: g++-15 exists but gcc-15 is missing." >&2
      return 1
    }
    echo "PyPTO compiler: $(command -v g++-15)"
    return 0
  fi

  if ! command -v x86_64-conda-linux-gnu-g++ >/dev/null 2>&1 ||
    ! command -v x86_64-conda-linux-gnu-gcc >/dev/null 2>&1 ||
    [[ "$(compiler_major "$(command -v x86_64-conda-linux-gnu-g++)")" != "15" ]]; then
    command -v conda >/dev/null 2>&1 || {
      echo "ERROR: the PyPTO simulator suite needs GCC/G++ 15;" \
        "provide gcc-15/g++-15 on PATH or make conda available." >&2
      return 1
    }
    conda install -y -c conda-forge gcc_linux-64=15 gxx_linux-64=15 || return 1
    hash -r
  fi

  mkdir -p "${PYPTO_WORKSPACE}/.work/bin"
  ln -sf "$(command -v x86_64-conda-linux-gnu-g++)" "${PYPTO_WORKSPACE}/.work/bin/g++-15"
  ln -sf "$(command -v x86_64-conda-linux-gnu-gcc)" "${PYPTO_WORKSPACE}/.work/bin/gcc-15"
  activate_pypto_toolchain
  echo "PyPTO compiler: $(command -v g++-15)"
}

# The build phase creates that directory; the test phase runs in a fresh shell
# and has to make it visible again.
activate_pypto_toolchain() {
  [[ -x "${PYPTO_WORKSPACE}/.work/bin/g++-15" ]] || return 0
  export PATH="${PYPTO_WORKSPACE}/.work/bin:${PATH}"
}

# The runtime only works when the compiled headers are on the include path the
# frontend resolves for the simulator platform; check that instead of letting
# every PyPTO test fail on a bad install.
verify_pypto_install() {
  (cd "${PYPTO_WORKSPACE}" && "${PYTHON_BIN:-python3}" - <<'PY'
from simpler_setup.environment import PROJECT_ROOT
from simpler_setup.kernel_compiler import KernelCompiler

required = PROJECT_ROOT / "src" / "common" / "task_interface" / "arg_direction.h"
if not required.is_file():
    raise SystemExit(f"missing required simpler runtime header: {required}")

include_dirs = KernelCompiler(platform="a5sim").get_orchestration_include_dirs(
    "tensormap_and_ringbuffer"
)
if str(required.parent) not in include_dirs:
    raise SystemExit(
        "simpler orchestration include dirs do not contain "
        f"{required.parent}: {include_dirs}"
    )

print(f"simpler PROJECT_ROOT={PROJECT_ROOT}")
print(f"simpler arg_direction.h={required}")
PY
  )
}

run_pypto_tests() {
  # Same marker the preparation writes: it lives next to the interpreter, not in
  # the per run workspace, so a runner reusing that interpreter sees it.
  [[ -f "$(pypto_marker_path)" ]] || {
    echo "ERROR: PyPTO is not prepared for ${PYTHON_BIN:-python3};" \
      "run the build phase first." >&2
    return 1
  }
  local python_bin="${PYTHON_BIN:-python3}"
  # The framework node ids are long enough to overrun the line limit on their
  # own, so the two shared prefixes are spelled once.
  local st_fw="tests/st/runtime/framework_and_models"
  local attn="${st_fw}/test_paged_attention.py::TestPagedAttentionKernels"
  local -a core_tests=(
    tests/st/examples/00_hello_world/test_hello_world.py
    tests/st/examples/02_intermediate/test_softmax.py::TestTileSoftmax::test_tile_softmax
    tests/st/examples/02_intermediate/test_rms_norm.py::TestRMSNormCore::test_rms_norm_core
    tests/st/runtime/ops/test_elementwise.py
    tests/st/runtime/ops/test_assemble.py
    "${st_fw}/test_jit.py::TestJITExecution::test_cache_hit_reuses_compiled_program"
    "${st_fw}/test_jit.py::TestJITDynamicBatch::test_one_artifact_serves_multiple_batches"
    "${st_fw}/test_compiled_program.py::TestManualWorkerExtraction::test_block_dim_override_runs"
  )
  local -a fa_tests=(
    tests/st/runtime/ops/test_cast.py::TestCast::test_tile_cast_col_major_narrow
    "${attn}::test_qk_matmul_ptoas[16-128-128]"
    "${attn}::test_softmax_prepare_ptoas[16-128]"
    "${attn}::test_softmax_prepare_unaligned_ptoas[16-128-100]"
    "${attn}::test_pv_matmul_ptoas[16-128-128]"
    "${attn}::test_online_update_ptoas[16-128-0-1]"
  )
  local -a int8_tests=(tests/st/codegen/dsl/test_batch_matmul_pipeline.py::test_no_mat_to_mat_tmov)
  activate_pypto_toolchain
  run_pypto_pytest() {
    local platform="$1"
    local suite_name="$2"
    shift 2
    (cd "${PYPTO_WORKSPACE}" && \
      PTOAS_ROOT="$(dirname "${PTOAS_BIN}")" \
      PTO_ISA_ROOT="${PTO_ISA_ROOT}" \
      TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
      "${python_bin}" -m pytest "$@" -v --platform="${platform}" \
        --save-kernels \
        --kernels-dir="${PYPTO_RUN_WORKSPACE}/${suite_name}_${platform}") \
      2>&1 | tee "${BUILD_ROOT}/pypto-${suite_name}-${platform}.log"
  }
  local platform
  local -a core
  for platform in a5sim a2a3sim; do
    core=("${core_tests[@]}")
    # This dynamic-orchestration case needs the PyPTO tensormap_and_ringbuffer
    # runtime, which the a5sim environment does not ship, so keep that coverage
    # on a2a3sim rather than failing the gate on an environment gap.
    if [[ "${platform}" != "a5sim" ]]; then
      local dyn_orch="tests/st/runtime/control_flow/test_dyn_orch_shape.py"
      dyn_orch+="::TestDynOrchShapeOperations"
      dyn_orch+="::test_dyn_orch_valid_shape_add[shape0-valid_shape0-${platform}]"
      core+=("${dyn_orch}")
    fi
    run_suite "PyPTO core ${platform}" run_pypto_pytest "${platform}" core_ptoas "${core[@]}"
    run_suite "PyPTO FA ${platform}" run_pypto_pytest "${platform}" fa_ptoas "${fa_tests[@]}"
  done
  run_suite "PyPTO INT8 a2a3sim" run_pypto_pytest a2a3sim int8_ptoas_codegen "${int8_tests[@]}"
}

# Which suites does this invocation cover? Preparation follows the same answer,
# so a VPTO-only run does not clone or install PyPTO.
needs_suite() {
  [[ "${SIM_SUITE}" == "all" || "${SIM_SUITE}" == "$1" ]]
}

# github is not always reachable from CI hosts: run #451 lost pto-isa to a
# connect timeout after 132 s and took the TileLang and PyPTO steps down with
# it. Retry the network operations a few times before giving up.
git_retry() {
  local attempt status
  for attempt in 1 2 3; do
    if git "$@"; then
      return 0
    fi
    status=$?
    echo "WARNING: git $1 failed with ${status} (attempt ${attempt}/3)" >&2
    sleep "$(( attempt * 10 ))"
  done
  return "${status:-1}"
}

# Give a suite its checkout under the workspace: copy a locally provided
# source when the environment has one, otherwise clone the configured URL.
materialize_repo() {
  local repo_url="$1" source_dir="$2" target="$3"
  if [[ -d "${target}/.git" ]]; then
    return 0
  fi
  if [[ -n "${source_dir}" && -d "${source_dir}/.git" ]]; then
    echo "Copying ${source_dir} into ${target}"
    cp -a "${source_dir}" "${target}" || return 1
    # The copy has to stay writable even when the provided source is mounted
    # read-only: the pin checkout and the submodule refresh happen inside the
    # workspace copy.
    chmod -R u+w "${target}" || return 1
    return 0
  fi
  git_retry clone --depth 1 "${repo_url}" "${target}"
}

# Preparation happens in the build phase only; the suite steps consume what the
# build left behind and name whatever is missing. Re-cloning or reinstalling per
# suite would make every suite depend on the network, which is how a transient
# github TLS error turned the PyPTO step into a failure.
if [[ "${MODE}" != "test" ]]; then
  if needs_suite tilelib || needs_suite pypto; then
    echo "Preparing pto-isa for the tilelib/pypto suites"
    prepare_pto_isa || {
      echo "ERROR: failed to prepare pto-isa at ${PTO_ISA_ROOT}" >&2
      exit 1
    }
  fi
  if needs_suite pypto; then
    echo "Preparing PyPTO (checkout, compiler, frontend and runtime)"
    prepare_pypto || {
      echo "ERROR: failed to prepare PyPTO in ${PYPTO_WORKSPACE}" >&2
      exit 1
    }
  fi
fi

if [[ "${MODE}" == "build" ]]; then
  echo "PTOAS build ready: ${PTOAS_BIN}"
  exit 0
fi

case "${SIM_SUITE}" in
  vpto)
    run_suite "VPTO SIM" run_vpto
    ;;
  tilelib)
    run_suite "TileLib ST" run_tilelib_st
    ;;
  ptodsl)
    run_suite "PTODSL/DSL ST" run_ptodsl_st
    ;;
  pypto)
    run_suite "PyPTO simulator suites" run_pypto_tests
    ;;
  all)
    run_suite "VPTO SIM" run_vpto
    run_suite "TileLib ST" run_tilelib_st
    run_suite "PTODSL/DSL ST" run_ptodsl_st
    run_suite "PyPTO simulator suites" run_pypto_tests
    ;;
  *)
    echo "ERROR: unsupported SIM_SUITE=${SIM_SUITE}; expected vpto, tilelib, ptodsl, pypto, or all" >&2
    exit 2
    ;;
esac

echo "=== SIM suite summary ==="
if [[ "${SUITE_STATUS}" -ne 0 ]]; then
  echo "One or more simulator suites failed."
else
  echo "All migrated simulator suites passed."
fi
exit "${SUITE_STATUS}"
