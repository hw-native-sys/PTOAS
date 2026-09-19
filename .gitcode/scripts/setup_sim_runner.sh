#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
# Provision a GitCode runner host for the simulator gate. Run it by hand on the
# runner host, as root:
#
#   .gitcode/scripts/setup_sim_runner.sh shared --shared /opt/ptoas-opensource/shared
#   .gitcode/scripts/setup_sim_runner.sh runner --prefix /opt/ptoas-opensource/runner-a \
#       --python /opt/ptoas-opensource/runner-a/venv/bin/python --unit gc0.service
#
# "shared" provisions the parts every runner on the host can share and that are
# never written again afterwards: the GCC 15 pair PyPTO needs and read-only
# copies of the upstream checkouts. "runner" provisions the writable parts that
# must not be shared between runners: the LLVM cache root and the interpreter
# PyPTO installs into, plus the systemd drop-in that hands them to the job.
#
# Usage: setup_sim_runner.sh <shared|runner> [options]   (see --help)

set -euo pipefail

# This script configures a runner host; it is not part of the gate and must not be
# run by a workflow. GitCode exports ATOMGIT_ACTIONS=true inside a job, so refuse
# to do anything there unless the operator insists.
if [ "${ATOMGIT_ACTIONS:-}" = "true" ] && [ "${RUNNER_SETUP_ALLOW_CI:-}" != "1" ]; then
  printf "[setup] ERROR: this script is meant to be run by hand on the runner host;\n" >&2
  printf "[setup] ERROR: set RUNNER_SETUP_ALLOW_CI=1 to override.\n" >&2
  exit 2
fi

log() { printf "[setup] %s\n" "$*"; }
warn() { printf "[setup] WARNING: %s\n" "$*" >&2; }
die() {
  printf "[setup] ERROR: %s\n" "$*" >&2
  exit 1
}

DRY_RUN=0
run() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf "[dry-run] %s\n" "$*"
    return 0
  fi
  "$@"
}

usage() {
  cat <<EOF
Usage:
  setup_sim_runner.sh shared [options]
  setup_sim_runner.sh runner --prefix <dir> [options]

Shared options:
  --shared <dir>          shared root (default /opt/ptoas-opensource/shared)
  --bin-dir <dir>         where to link gcc-15/g++-15 (default /usr/local/bin)
  --conda-channel <url>   conda channel for the cross compiler (default conda-forge)
  --mamba-url <url>       micromamba download URL
  --git-proxy <prefix>    proxy prefix for github, e.g. https://ghfast.top/ (default none)
  --pto-isa-url <url>     pto-isa clone URL (default https://github.com/hw-native-sys/pto-isa.git)
  --pypto-url <url>       pypto clone URL (default https://github.com/hw-native-sys/pypto.git)

Runner options:
  --prefix <dir>          writable per-runner root, also the runner work dir parent
  --python <path>         interpreter that already provides torch/torch_npu
  --base-python <path>    interpreter used to create the venv (default python3)
  --copy-venv-from <dir>  copy an existing venv instead of creating one (keeps torch)
  --pip-index <url>       pip index URL (default PyPI)
  --cache-root <dir>      LLVM cache root handed to the gate (default <prefix>/cache)
  --runner-root <dir>     the runner installation; prunes the checkouts of older
                          worker versions, which the platform leaves behind
  --copy-cache-from <d>   seed that cache from another runner prefix
  --unit <name>           systemd unit of this runner (writes a drop-in)
  --systemd-dir <dir>     where the unit lives (default /etc/systemd/system)
  --memory-max <value>    MemoryMax for the drop-in (default 80G)
  --cpu-quota <value>     CPUQuota for the drop-in (default 3200%)
  --dry-run               print every command instead of running it
  -h, --help              show this help
EOF
}

SHARED=/opt/ptoas-opensource/shared
BIN_DIR=/usr/local/bin
CONDA_CHANNEL=conda-forge
MAMBA_URL=https://micro.mamba.pm/api/micromamba/linux-64/latest
GIT_PROXY=
PTO_ISA_URL=https://github.com/hw-native-sys/pto-isa.git
PYPTO_URL=https://github.com/hw-native-sys/pypto.git
PREFIX=
PYTHON=
BASE_PYTHON=python3
COPY_VENV_FROM=
PIP_INDEX=
COPY_CACHE_FROM=
CACHE_ROOT=
RUNNER_ROOT=
UNIT=
SYSTEMD_DIR=/etc/systemd/system
MEMORY_MAX=80G
CPU_QUOTA=3200%

MODE="${1:-}"
case "${MODE}" in
  shared|runner) shift ;;
  -h|--help|"") usage; exit 0 ;;
  *) usage >&2; die "unknown mode: ${MODE}" ;;
esac

while [[ $# -gt 0 ]]; do
  case "$1" in
    --shared) SHARED="$2"; shift 2 ;;
    --bin-dir) BIN_DIR="$2"; shift 2 ;;
    --conda-channel) CONDA_CHANNEL="$2"; shift 2 ;;
    --mamba-url) MAMBA_URL="$2"; shift 2 ;;
    --git-proxy) GIT_PROXY="$2"; shift 2 ;;
    --pto-isa-url) PTO_ISA_URL="$2"; shift 2 ;;
    --pypto-url) PYPTO_URL="$2"; shift 2 ;;
    --prefix) PREFIX="$2"; shift 2 ;;
    --python) PYTHON="$2"; shift 2 ;;
    --base-python) BASE_PYTHON="$2"; shift 2 ;;
    --copy-venv-from) COPY_VENV_FROM="$2"; shift 2 ;;
    --pip-index) PIP_INDEX="$2"; shift 2 ;;
    --copy-cache-from) COPY_CACHE_FROM="$2"; shift 2 ;;
    --cache-root) CACHE_ROOT="$2"; shift 2 ;;
    --runner-root) RUNNER_ROOT="$2"; shift 2 ;;
    --unit) UNIT="$2"; shift 2 ;;
    --systemd-dir) SYSTEMD_DIR="$2"; shift 2 ;;
    --memory-max) MEMORY_MAX="$2"; shift 2 ;;
    --cpu-quota) CPU_QUOTA="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; die "unknown option: $1" ;;
  esac
done

PIP_ARGS=()
[[ -z "${PIP_INDEX}" ]] || PIP_ARGS=(-i "${PIP_INDEX}")

retry() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf "[dry-run] %s\n" "$*"
    return 0
  fi
  local attempt
  for attempt in 1 2 3; do
    if "$@"; then
      return 0
    fi
    warn "command failed, retrying (${attempt}/3): $*"
    sleep "$(( attempt * 10 ))"
  done
  return 1
}

clone_repo() {
  local name="$1" url="$2" target="$3"
  if [[ -d "${target}/.git" ]]; then
    log "${name} is already present at ${target}"
    retry git -C "${target}" fetch --all --tags || warn "could not refresh ${name}"
    return 0
  fi
  log "cloning ${name} into ${target}"
  if retry git clone "${url}" "${target}"; then
    return 0
  fi
  if [[ -n "${GIT_PROXY}" ]]; then
    warn "${name}: retrying through ${GIT_PROXY}"
    run rm -rf "${target}"
    retry git clone "${GIT_PROXY}${url}" "${target}"
    return $?
  fi
  return 1
}

clone_submodules() {
  local target="$1"
  local -a cmd=(git -C "${target}")
  if [[ -n "${GIT_PROXY}" ]]; then
    cmd+=(-c "url.${GIT_PROXY}https://github.com/.insteadOf=https://github.com/")
  fi
  cmd+=(submodule update --init --recursive)
  retry "${cmd[@]}"
}
install_micromamba() {
  local mamba_root="${SHARED}/mamba"
  if [[ -x "${mamba_root}/bin/micromamba" ]]; then
    log "micromamba is already available at ${mamba_root}/bin/micromamba"
    return 0
  fi
  log "downloading micromamba from ${MAMBA_URL}"
  run mkdir -p "${mamba_root}/bin"
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf "[dry-run] curl -Ls %s | tar -xvj -C %s bin/micromamba\n" "${MAMBA_URL}" "${mamba_root}"
  else
    curl -Ls --max-time 600 "${MAMBA_URL}" | tar -xvj -C "${mamba_root}" bin/micromamba
  fi
}

install_gcc15() {
  local prefix="${SHARED}/gcc15"
  local mamba="${SHARED}/mamba/bin/micromamba"
  if [[ -x "${prefix}/bin/x86_64-conda-linux-gnu-g++" ]]; then
    log "GCC 15 is already provisioned at ${prefix}"
  else
    log "installing GCC 15 with micromamba (channel ${CONDA_CHANNEL})"
    run env MAMBA_ROOT_PREFIX="${SHARED}/mamba" "${mamba}" create -y -p "${prefix}" \
      -c "${CONDA_CHANNEL}" gcc_linux-64=15 gxx_linux-64=15
  fi
  run ln -sf "${prefix}/bin/x86_64-conda-linux-gnu-gcc" "${BIN_DIR}/gcc-15"
  run ln -sf "${prefix}/bin/x86_64-conda-linux-gnu-g++" "${BIN_DIR}/g++-15"
  if [[ "${DRY_RUN}" != "1" ]]; then
    "${BIN_DIR}/gcc-15" --version | head -1
    "${BIN_DIR}/g++-15" --version | head -1
  fi
}

verify_shared() {
  [[ "${DRY_RUN}" == "1" ]] && return 0
  local incomplete=0
  local tool repo
  for tool in gcc-15 g++-15; do
    if ! command -v "${tool}" >/dev/null 2>&1; then
      warn "${tool} is not on PATH"
      incomplete=1
    fi
  done
  for repo in pto-isa pypto; do
    if [[ ! -d "${SHARED}/sources/${repo}/.git" ]]; then
      warn "missing checkout: ${SHARED}/sources/${repo}"
      incomplete=1
    fi
  done
  [[ "${incomplete}" == "0" ]] || die "shared provisioning is incomplete"
  log "shared provisioning verified"
}

provision_shared() {
  log "shared root: ${SHARED}"
  run mkdir -p "${SHARED}/sources"
  install_micromamba
  install_gcc15
  clone_repo pto-isa "${PTO_ISA_URL}" "${SHARED}/sources/pto-isa"
  clone_repo pypto "${PYPTO_URL}" "${SHARED}/sources/pypto"
  clone_submodules "${SHARED}/sources/pypto"
  verify_shared
}

# The platform installs each worker version under its own directory and gives it
# its own checkout, so every self upgrade leaves a full workspace (checkout plus
# the .work directory, a few GiB) behind. Keep the newest one and drop the
# checkouts of the older versions, but never touch one that was used today.
prune_stale_workspaces() {
  [[ -n "${RUNNER_ROOT}" ]] || return 0
  [[ -d "${RUNNER_ROOT}/runner/workers" ]] || {
    warn "no worker directory under ${RUNNER_ROOT}"
    return 0
  }
  local -a versions=()
  while IFS= read -r line; do versions+=("${line}"); done < <(
    ls -1dt "${RUNNER_ROOT}"/runner/workers/*/ 2>/dev/null | head -20
  )
  local newest="${versions[0]:-}"
  local version
  for version in "${versions[@]:1}"; do
    [[ "${version}" == "${newest}" ]] && continue
    if [[ -n "$(find "${version}worker_dir" -maxdepth 4 -newermt "-1 day" -print -quit 2>/dev/null)" ]]; then
      log "keeping $(basename "${version}") (used within the last day)"
      continue
    fi
    log "pruning the stale checkout under $(basename "${version}")"
    run rm -rf "${version}worker_dir"
  done
}

seed_cache() {
  if [[ -z "${COPY_CACHE_FROM}" ]]; then
    log "no --copy-cache-from given: the first build fills ${CACHE_ROOT} by building LLVM"
    return 0
  fi
  [[ -d "${COPY_CACHE_FROM}/cache" ]] || die "no cache under ${COPY_CACHE_FROM}"
  log "seeding ${CACHE_ROOT} from ${COPY_CACHE_FROM}/cache"
  run mkdir -p "${CACHE_ROOT}"
  run cp -a "${COPY_CACHE_FROM}/cache/." "${CACHE_ROOT}/"
}

ensure_interpreter() {
  if [[ -n "${PYTHON}" ]]; then
    [[ -x "${PYTHON}" ]] || die "interpreter is not executable: ${PYTHON}"
    log "using the provided interpreter ${PYTHON}"
    return 0
  fi
  PYTHON="${PREFIX}/venv/bin/python"
  if [[ -x "${PYTHON}" ]]; then
    log "reusing the interpreter at ${PYTHON}"
    return 0
  fi
  if [[ -n "${COPY_VENV_FROM}" ]]; then
    [[ -x "${COPY_VENV_FROM}/bin/python" ]] || die "no interpreter at ${COPY_VENV_FROM}/bin/python"
    log "copying the venv from ${COPY_VENV_FROM} (keeps torch and torch_npu)"
    run mkdir -p "$(dirname "${PREFIX}/venv")"
    run cp -a "${COPY_VENV_FROM}" "${PREFIX}/venv"
    log "each runner owns this copy: the gate installs PyPTO into it"
    return 0
  fi
  command -v "${BASE_PYTHON}" >/dev/null 2>&1 || die "base interpreter not found: ${BASE_PYTHON}"
  log "creating ${PREFIX}/venv from ${BASE_PYTHON} with system site packages"
  run "${BASE_PYTHON}" -m venv --system-site-packages "${PREFIX}/venv"
  log "install torch and torch_npu into it before pointing a runner here"
}

verify_interpreter() {
  [[ "${DRY_RUN}" == "1" ]] && return 0
  local module
  for module in numpy ml_dtypes cloudpickle pytest xdist pybind11; do
    "${PYTHON}" -c "import ${module}" >/dev/null 2>&1 ||
      die "the interpreter cannot import ${module}: ${PYTHON}"
    log "ok: ${module}"
  done
  # torch_npu loads only with the CANN environment sourced, which the job does.
  if "${PYTHON}" -c "import torch" >/dev/null 2>&1; then
    log "ok: torch"
  else
    warn "torch is not importable here; verify it from a job with CANN sourced"
  fi
}

install_python_deps() {
  log "installing the simulator test dependencies into ${PYTHON}"
  run "${PYTHON}" -m pip install --no-input "${PIP_ARGS[@]}" --upgrade pip
  # pybind11 is not a test dependency: PTOAS configures and compiles its MLIR
  # python bindings through the interpreter the gate runs with, so a fresh host
  # without it fails in CMake with "pybind11 not found" instead of building.
  # Keep it in the interpreter rather than on the host so a copied venv stays
  # self sufficient. The pin is deliberate: 3.1 rejects the keep_alive used by
  # the PTOAS bindings with a static_assert, and 3.0.1 is the validated version.
  run "${PYTHON}" -m pip install --no-input "${PIP_ARGS[@]}" \
    numpy ml_dtypes cloudpickle pytest pytest-xdist "pybind11==3.0.1"
  if ! run "${PYTHON}" -m pip install --no-input "${PIP_ARGS[@]}" en_dtypes; then
    warn "en_dtypes is unavailable on this index; a few mx cases will not run"
  fi
  verify_interpreter
}

emit_dropin() {
  if [[ -z "${UNIT}" ]]; then
    log "no --unit given; pass it to also write the systemd drop-in"
    return 0
  fi
  local dir="${SYSTEMD_DIR}/${UNIT}.d"
  local file="${dir}/10-sim-env.conf"
  log "writing ${file}"
  run mkdir -p "${dir}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf "[dry-run] would write %s with the simulator variables\n" "${file}"
    printf "[dry-run] ASCEND_3RD_LIB_PATH=%s SIM_PYTHON_BIN=%s\n" \
      "${CACHE_ROOT}" "${PYTHON}"
    return 0
  fi
  cat > "${file}" <<CONF
[Service]
Environment=ASCEND_3RD_LIB_PATH=${CACHE_ROOT}
Environment=SIM_PYTHON_BIN=${PYTHON}
Environment=PTO_ISA_SOURCE_DIR=${SHARED}/sources/pto-isa
Environment=PYPTO_SOURCE_DIR=${SHARED}/sources/pypto
MemoryMax=${MEMORY_MAX}
CPUQuota=${CPU_QUOTA}
CONF
  log "apply it with: systemctl daemon-reload && systemctl restart ${UNIT}"
}

print_exports() {
  printf "\nRunner environment (the drop-in sets exactly this):\n"
  printf "  ASCEND_3RD_LIB_PATH=%s   (required)\n" "${CACHE_ROOT}"
  printf "  SIM_PYTHON_BIN=%s   (required)\n" "${PYTHON}"
  printf "  PTO_ISA_SOURCE_DIR=%s   (optional)\n" "${SHARED}/sources/pto-isa"
  printf "  PYPTO_SOURCE_DIR=%s   (optional)\n" "${SHARED}/sources/pypto"
  printf "\nThe gate checks the two required ones and stops when they are missing;\n"
  printf "the optional ones fall back to cloning the upstream checkouts.\n"
  printf "\nKeep the runner work dir on a large filesystem; %s/work is reserved for it.\n" "${PREFIX}"
  printf "Repository variables of the same name still work; the runner environment wins.\n"
}

provision_runner() {
  [[ -n "${PREFIX}" ]] || die "runner mode needs --prefix <dir>"
  # The interpreter must be per runner: the gate installs PyPTO into it. The LLVM
  # cache is read only in the steady state, so --cache-root lets several runners on
  # one host share it.
  CACHE_ROOT="${CACHE_ROOT:-${PREFIX}/cache}"
  log "runner prefix: ${PREFIX}"
  log "llvm cache root: ${CACHE_ROOT}"
  run mkdir -p "${CACHE_ROOT}" "${PREFIX}/work"
  prune_stale_workspaces
  seed_cache
  ensure_interpreter
  install_python_deps
  emit_dropin
  print_exports
}

case "${MODE}" in
  shared) provision_shared ;;
  runner) provision_runner ;;
esac