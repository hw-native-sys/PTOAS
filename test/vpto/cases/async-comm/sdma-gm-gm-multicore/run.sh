#!/bin/bash
# Build and run the multicore GM->GM SDMA case, one channel group per core.
#
# This does not go through run_host_vpto_validation.sh because that script
# hard-codes --cce-aicore-arch=dav-c310 and this case targets c220, and because
# the host program needs the PTOAS include path for AsyncWorkspace.h.
#
# Requires CANN 9.0.0 at run time: the workspace setup calls
# aclnnShmemSdmaStarsQuery, which earlier toolkits do not ship.
#
#   PTOAS_BUILD_DIR=... ARCH=a3 AICORE_ARCH=dav-c220-vec ./run.sh [device]
#   PTOAS_BUILD_DIR=... ARCH=a5 AICORE_ARCH=dav-c310-vec  ./run.sh [device]
set -euo pipefail

CASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${CASE_DIR}/../../../../.." && pwd)"

# The build tree is often outside the repo, so try the usual places and say
# which ones were tried rather than reporting only the default.
resolve_ptoas() {
  if [[ -n "${PTOAS_BIN:-}" ]]; then
    echo "${PTOAS_BIN}"
    return
  fi
  local candidates=()
  [[ -n "${PTOAS_BUILD_DIR:-}" ]] && candidates+=("${PTOAS_BUILD_DIR}/tools/ptoas/ptoas")
  candidates+=("${REPO_ROOT}/build/tools/ptoas/ptoas")
  local onpath
  onpath="$(command -v ptoas 2>/dev/null || true)"
  [[ -n "${onpath}" ]] && candidates+=("${onpath}")
  for c in "${candidates[@]}"; do
    if [[ -x "${c}" ]]; then
      echo "${c}"
      return
    fi
  done
  echo "ptoas not found; set PTOAS_BIN or PTOAS_BUILD_DIR. Tried:" >&2
  printf '  %s\n' "${candidates[@]}" >&2
  exit 1
}

PTOAS_BIN="$(resolve_ptoas)"
ARCH="${ARCH:-a3}"
AICORE_ARCH="${AICORE_ARCH:-dav-c220-vec}"
# Only the vec/cube suffix is dropped for the launch object's arch flag.
# Only the vec/cube suffix is dropped. `%-*` would turn dav-c310 into dav.
AICORE_ARCH_BASE="${AICORE_ARCH%-vec}"
AICORE_ARCH_BASE="${AICORE_ARCH_BASE%-cube}"
OUT_DIR="${OUT_DIR:-${CASE_DIR}/build}"
DEVICE_ID="${1:-0}"

: "${ASCEND_HOME_PATH:?source your CANN set_env.sh first}"
BISHENG_BIN="${BISHENG_BIN:-${ASCEND_HOME_PATH}/bin/bisheng}"

command -v "${BISHENG_BIN}" >/dev/null 2>&1 || { echo "bisheng not found: ${BISHENG_BIN}" >&2; exit 1; }

mkdir -p "${OUT_DIR}"
echo "== arch ${ARCH} / ${AICORE_ARCH}, toolkit ${ASCEND_HOME_PATH}"

echo "== 1/4 kernel fatobj"
"${PTOAS_BIN}" --pto-arch "${ARCH}" --pto-backend=vpto \
  "${CASE_DIR}/kernel.pto" -o "${OUT_DIR}/kernel.fatobj.o"

echo "== 2/4 launch object"
"${BISHENG_BIN}" \
  -c -fPIC -xcce -fenable-matrix --cce-aicore-enable-tl \
  -Xhost-start -Xhost-end \
  -mllvm -cce-aicore-stack-size=0x8000 \
  -mllvm -cce-aicore-function-stack-size=0x8000 \
  -mllvm -cce-aicore-record-overflow=true \
  -mllvm -cce-aicore-addr-transform \
  -mllvm -cce-aicore-dcci-insert-for-scalar=false \
  --cce-aicore-arch="${AICORE_ARCH_BASE}" \
  -DREGISTER_BASE -std=c++17 \
  -Wno-macro-redefined -Wno-ignored-attributes \
  -I "${REPO_ROOT}/include" \
  -I "${REPO_ROOT}/test/comm" \
  -I "${ASCEND_HOME_PATH}/include" \
  -I "${ASCEND_HOME_PATH}/pkg_inc" \
  -I "${ASCEND_HOME_PATH}/pkg_inc/profiling" \
  -I "${ASCEND_HOME_PATH}/pkg_inc/runtime/runtime" \
  "${CASE_DIR}/launch.cpp" \
  -o "${OUT_DIR}/launch.o"

echo "== 3/4 kernel shared library"
"${BISHENG_BIN}" \
  -fPIC -Wl,-z,relro -Wl,-z,now --cce-fatobj-link \
  -shared -Wl,-soname,libsdma_gm_gm_multicore_kernel.so \
  -L "${ASCEND_HOME_PATH}/lib64" \
  -Wl,-rpath,"${ASCEND_HOME_PATH}/lib64" \
  -o "${OUT_DIR}/libsdma_gm_gm_multicore_kernel.so" \
  "${OUT_DIR}/kernel.fatobj.o" \
  "${OUT_DIR}/launch.o" \
  -Wl,--no-as-needed -lruntime

echo "== 4/4 host executable"
"${BISHENG_BIN}" \
  -xc++ -std=c++17 \
  "${CASE_DIR}/main.cpp" \
  -I "${CASE_DIR}" \
  -I "${REPO_ROOT}/include" \
  -I "${REPO_ROOT}/test/comm" \
  -I "${ASCEND_HOME_PATH}/include" \
  -L "${OUT_DIR}" \
  -L "${ASCEND_HOME_PATH}/lib64" \
  -Wl,-rpath,"${OUT_DIR}" \
  -Wl,-rpath,"${ASCEND_HOME_PATH}/lib64" \
  -o "${OUT_DIR}/sdma_gm_gm_multicore" \
  -lsdma_gm_gm_multicore_kernel \
  -Wl,--allow-shlib-undefined -lruntime \
  -lstdc++ -lascendcl -lm -lc_sec -ldl -lnnopbase

# Building is safe anywhere; running needs a card. Splitting the two lets a
# change be checked for compile errors without putting a device at risk.
if [ -n "${BUILD_ONLY:-}" ]; then
  echo "== build only, skipping the run"
  exit 0
fi

echo "== run on device ${DEVICE_ID}"
cd "${OUT_DIR}"
# A kernel that never returns holds the device until the process dies, and
# suspending the shell leaves it held. Kill the run instead, hard if it will
# not go, so a failed case costs a case rather than a card.
set +e
LD_LIBRARY_PATH="${OUT_DIR}:${ASCEND_HOME_PATH}/lib64:${LD_LIBRARY_PATH:-}" \
  timeout --signal=KILL "${RUN_TIMEOUT:-60}" ./sdma_gm_gm_multicore "${DEVICE_ID}"
rc=$?
set -e
if [ "${rc}" -eq 137 ]; then
  echo "== killed after ${RUN_TIMEOUT:-60}s; device ${DEVICE_ID} may need a reset" >&2
fi
exit "${rc}"
