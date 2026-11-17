#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
#
# A3 (a2a3, Ascend910) half of the PTOAS x PyPTO compatibility nightly.
#
# The development host this was written for (68.6) is retired. The A3 half now
# runs on the self-hosted runner through .gitcode/workflows/a3_pypto_smoke.yml;
# keep this driver as the reference for porting the full selection, with every
# host-specific path supplied explicitly through the environment.
#
# Runs on the A3 host, which the board host cannot reach inbound: it pushes a
# result file to the board host instead, and the 04:00 run there folds it into
# the single Feishu card. Candidate is the same release pin as the board host;
# when that release is not installed for aarch64 it falls back to the branch
# build and says so in the result.

set -euo pipefail

ROOT="${PTOAS_PYPTO_ROOT:?set PTOAS_PYPTO_ROOT to the A3 checkout/venv root}"
VENV="${ROOT}/venv/bin/python"
SRC="${ROOT}/pypto"
LLVM="${PTOAS_LLVM_DIR:?set PTOAS_LLVM_DIR to the LLVM build tree on the A3 host}"
PIN="${ROOT}/ptoas_release.txt"
PUBLISH_HOST="${PTOAS_PUBLISH_HOST:-}"
PUBLISH_PORT="${PTOAS_PUBLISH_PORT:-10118}"
PUBLISH_DIR="${PTOAS_PUBLISH_DIR:-/data/ptoas-pypto/incoming/a3}"
PUBLISH_KEY="${PTOAS_PUBLISH_KEY:-$HOME/.ssh/id_ed25519_ptoas238}"
RUN_ID="$(date +%Y%m%d-%H%M%S)"
OUT="${ROOT}/results/${RUN_ID}"
# The host is shared: other tenants hold all cards for hours at a time, so give
# the queue two hours before reporting NO_DEVICE.
DEVICE_WAIT="${PTOAS_DEVICE_WAIT:-7200}"

# PyPTO's system-tests-direct content that runs in-process on one card. Kept
# small: the batched shards need pypto's own task-submit batching, which is a
# separate step.
TESTS=(
  tests/st/runtime/ops/test_tri.py
  tests/st/runtime/ops/test_prelu.py
  tests/st/runtime/ops/test_assemble.py
)

mkdir -p "${OUT}"
exec > >(tee "${OUT}/run.log") 2>&1

fail() { echo "ERROR: $*" >&2; exit 1; }

# A second daily shot exists only to retry when the shared card pool was busy.
# Skip it when the first one already produced a result.
if [[ "${PTOAS_A3_FORCE:-0}" != 1 ]]; then
  last="$(ls -1dt "${ROOT}"/results/*/result.json 2>/dev/null | head -1 || true)"
  if [[ -n "${last}" ]]; then
    last_age=$(( $(date +%s) - $(stat -c %Y "${last}") ))
    if (( last_age < 3 * 3600 )); then
      echo "skip: result produced ${last_age}s ago (${last}); set PTOAS_A3_FORCE=1 to rerun"
      exit 0
    fi
  fi
fi

[[ -x "${VENV}" ]] || fail "missing python: ${VENV}"
[[ -d "${SRC}/tests/st" ]] || fail "missing pypto checkout: ${SRC}"

# Keep the upstream checkout on the same revision the board host runs. The A3
# host has no GitHub route, so it pulls the tree from the board host over the
# same SSH channel it publishes on. A failed sync keeps the current tree: a
# network problem must not turn into "no A3 result".
SYNC_PYPTO="${PTOAS_A3_SYNC_PYPTO:-1}"

remote_ssh() {
  ssh -i "${PUBLISH_KEY}" -p "${PUBLISH_PORT}" \
      -o BatchMode=yes -o ConnectTimeout=15 \
      "${PUBLISH_HOST}" "$@"
}

# Install the fetched tree the way it was installed originally: system gcc, no
# conda flags, no build isolation (the sandbox has no package index), and CANN
# on the environment -- without it the runtime build skips the on-board variants
# and never writes build/lib/pto_isa_build.json, which every on-card test needs.
rebuild_tree() {
  local tree="$1"
  (
    # shellcheck disable=SC1091
    source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh >/dev/null 2>&1 || true
    export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
    export CC=/usr/bin/gcc CXX=/usr/bin/g++
    unset CFLAGS CXXFLAGS CPPFLAGS LDFLAGS
    cd "${tree}" && "${VENV}" -m pip install -q --no-build-isolation --no-deps --force-reinstall .
  ) || return 1
  (
    # shellcheck disable=SC1091
    source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh >/dev/null 2>&1 || true
    export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
    export CC=/usr/bin/gcc CXX=/usr/bin/g++
    unset CFLAGS CXXFLAGS CPPFLAGS LDFLAGS
    cd "${tree}/runtime" && "${VENV}" -m pip install -q --no-build-isolation --no-deps --force-reinstall .
  )
}

# The runtime resolves pto-isa from the packaged assets. On this host GitHub is
# unreachable, so an empty pto-isa directory turns into a git clone that hangs
# until the task queue kills the job. Copy the checkout the source tree carries.
seed_pto_isa() {
  local from="$1" assets dest
  [[ -d "${from}" && -n "$(ls -A "${from}" 2>/dev/null)" ]] || return 0
  assets="$("${VENV}" -c \
    'import os,simpler_setup;print(os.path.join(os.path.dirname(simpler_setup.__file__),"_assets","build"))' \
    2>/dev/null || true)"
  [[ -n "${assets}" ]] || return 0
  dest="${assets}/pto-isa"
  [[ -n "$(ls -A "${dest}" 2>/dev/null)" ]] && return 0
  rm -rf "${dest}"
  cp -a "${from}" "${dest}" && echo "pto-isa: seeded from ${from}"
}

sync_pypto() {
  (( SYNC_PYPTO )) || return 0
  local remote_head local_head staging
  remote_head="$(remote_ssh "git -C /data/ptoas-pypto/src/pypto rev-parse HEAD" 2>/dev/null || true)"
  if [[ -z "${remote_head}" ]]; then
    echo "pypto sync: board head unreadable, keeping current"
    return 0
  fi
  local_head="$(git -C "${SRC}" rev-parse HEAD 2>/dev/null || true)"
  if [[ "${remote_head}" == "${local_head}" ]]; then
    echo "pypto sync: up to date at ${local_head}"
    return 0
  fi
  echo "pypto sync: ${local_head:-?} -> ${remote_head}"
  staging="${ROOT}/pypto.next"
  rm -rf "${staging}" "${ROOT}.next"
  mkdir -p "${ROOT}.next"
  if ! remote_ssh "tar czf - -C /data/ptoas-pypto/src \
        --exclude=pypto/build --exclude=pypto/_skbuild \
        --exclude=pypto/runtime/build/cache --exclude='pypto/runtime/build/cp310*' \
        --exclude=pypto/runtime/build/lib \
        --exclude='__pycache__' --exclude='*.pyc' pypto" \
       2>> "${OUT}/pypto.sync.log" \
       | tar xzf - -C "${ROOT}.next" 2>> "${OUT}/pypto.sync.log"; then
    echo "pypto sync: transfer failed, keeping current"
    rm -rf "${ROOT}.next"
    return 0
  fi
  mv "${ROOT}.next/pypto" "${staging}"
  rm -rf "${ROOT}.next"
  # Bytecode compiled on the board host bakes that host's paths into every
  # traceback and @pl.jit lookup ("cannot retrieve source code").
  find "${staging}" -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
  if ! rebuild_tree "${staging}" > "${OUT}/pypto.rebuild.log" 2>&1; then
    echo "pypto sync: rebuild failed, keeping current (see ${OUT}/pypto.rebuild.log)"
    rm -rf "${staging}"
    # The venv may now hold a half-installed new tree; put the old one back.
    rebuild_tree "${SRC}" >> "${OUT}/pypto.rebuild.log" 2>&1 \
      || echo "pypto sync: WARN cannot restore ${SRC} into the venv"
    return 0
  fi
  # A reinstall drops the managed pto-isa checkout from the packaged assets.
  rm -rf "${SRC}.prev"
  mv "${SRC}" "${SRC}.prev"
  mv "${staging}" "${SRC}"
  seed_pto_isa "${SRC}/runtime/build/pto-isa"
  echo "pypto sync: updated to $(git -C "${SRC}" rev-parse --short HEAD 2>/dev/null)"
}
sync_pypto
# Always purge bytecode in the checkout: it may predate the sync exclusion and
# would make @pl.jit fail to recover source for every test.
find "${SRC}" -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
seed_pto_isa "${SRC}/runtime/build/pto-isa"

# Same "newest release" rule as the board host, with a visible fallback.
RELEASE_VER=""
CAND=""
rel_root=""
if [[ -f "${PIN}" ]]; then
  RELEASE_VER="$(tr -d '[:space:]' < "${PIN}")"
  # Same install-root layout as the board host (`v072`), but accept a dotted
  # directory too so an older hand-made install keeps working.
  rel_root="${ROOT}/ptoas-v${RELEASE_VER//./}"
  [[ -x "${rel_root}/bin/ptoas" ]] || rel_root="${ROOT}/ptoas-v${RELEASE_VER}"
  if [[ -n "${RELEASE_VER}" && -x "${rel_root}/bin/ptoas" ]]; then
    CAND="${rel_root}"
  fi
fi
if [[ -z "${CAND}" ]]; then
  CAND="${ROOT}/pto-as-install"
  CAND_NOTE="release ${RELEASE_VER:-?} not installed for aarch64; testing branch build"
  RELEASE_VER="branch-$(git -C "${ROOT}/pto-as-shallow" rev-parse --short HEAD 2>/dev/null || echo unknown)"
else
  CAND_NOTE="release ${RELEASE_VER}"
fi
[[ -x "${CAND}/bin/ptoas" ]] || fail "missing ptoas: ${CAND}/bin/ptoas"

echo "candidate: ${CAND} (${CAND_NOTE})"
PTOAS_VERSION="$(LD_LIBRARY_PATH="${LLVM}/lib:${CAND}/lib" "${CAND}/bin/ptoas" --version 2>&1 | head -1)"

# Run the selection on a borrowed card. task-submit writes the command into a
# script of its own, so hand it a file instead of a long inlined command; the
# sandbox also does not inherit the CANN environment.
INNER="${OUT}/task-inner.sh"
cat > "${INNER}" <<EOF
#!/usr/bin/env bash
set -euo pipefail
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh >/dev/null 2>&1 || true
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
export PTOAS_ROOT=${CAND}
export LD_LIBRARY_PATH=${LLVM}/lib:${CAND}/lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}
cd ${SRC}
# task-submit exports TASK_DEVICE only when it actually handed out a card; a
# timeout can still run the command with an empty environment.
if [[ -z "\${TASK_DEVICE:-}" ]]; then
  echo "no card assigned (TASK_DEVICE unset)" >&2
  exit 42
fi
exec ${VENV} -m pytest ${TESTS[*]} -q --platform=a2a3 \
  --device="\$TASK_DEVICE" --junitxml=${OUT}/a3.xml
EOF
chmod +x "${INNER}"

RC=0
task-submit --timeout "${DEVICE_WAIT}" --max-time "${TASK_MAX_TIME:-1800}" \
  --device auto --run "bash ${INNER}" \
  > "${OUT}/task.log" 2>&1 || RC=$?
echo "task-submit rc=${RC}"

# Counts and verdict; a missing XML means the run never got a card.
COUNTS="OK=0 FAIL=0 SKIP=0"
VERDICT="FAIL"
if [[ -f "${OUT}/a3.xml" ]]; then
  COUNTS="$("${VENV}" - "${OUT}/a3.xml" <<'PY'
import sys
import xml.etree.ElementTree as ET

ok = bad = skip = 0
for tc in ET.parse(sys.argv[1]).getroot().iter("testcase"):
    if tc.find("failure") is not None or tc.find("error") is not None:
        bad += 1
    elif tc.find("skipped") is not None:
        skip += 1
    else:
        ok += 1
print("OK={} FAIL={} SKIP={}".format(ok, bad, skip))
PY
)"
  [[ "${COUNTS}" == *"FAIL=0"* && "${RC}" -eq 0 ]] && VERDICT="GREEN"
else
  VERDICT="NO_DEVICE"
fi

PYTPO_HEAD="$(git -C "${SRC}" log -1 --format='%H %ci' 2>/dev/null || echo unknown)"

case "${VERDICT}" in
  GREEN) verdict_text='✅ 通过' ;;
  NO_DEVICE) verdict_text='⏳ 未拿到卡' ;;
  *) verdict_text='❌ 失败' ;;
esac
{
  echo "# [PTOAS × PyPTO 兼容] A3 ${RUN_ID}"
  echo
  echo "- 结论: ${verdict_text}"
  echo "- candidate: ${PTOAS_VERSION} (${CAND_NOTE})"
  echo "- pypto: ${PYTPO_HEAD}"
  echo "- a3-direct: ${COUNTS}"
  echo "- 日志: ${OUT}/"
} > "${OUT}/summary.md"

"${VENV}" - "${OUT}" "${RUN_ID}" "${VERDICT}" "${COUNTS}" "${PTOAS_VERSION}" \
  "${PYTPO_HEAD}" "${CAND_NOTE}" <<'PY' > "${OUT}/result.json"
import json
import sys
import time

out, run_id, verdict, counts, version, head, note = sys.argv[1:8]
print(json.dumps({
    "host": "68.6",
    "run_id": run_id,
    # UTC: this host runs on America/Los_Angeles time, the card is read in CST.
    "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "verdict": verdict,
    "ptoas": version,
    "ptoas_note": note,
    "pypto": head,
    "selections": {"a3-direct": counts},
}, ensure_ascii=False))
PY

cat "${OUT}/summary.md"
echo "RESULT_DIR=${OUT}"

# Publish for the board host's 04:00 card. A publish failure is not a test
# failure: the local result is still complete.
if [[ -n "${PUBLISH_HOST}" && -f "${PUBLISH_KEY}" ]]; then
  ssh -i "${PUBLISH_KEY}" -p "${PUBLISH_PORT}" \
      -o BatchMode=yes -o ConnectTimeout=15 "${PUBLISH_HOST}" \
      "mkdir -p ${PUBLISH_DIR}" 2>/dev/null || true
  scp -q -i "${PUBLISH_KEY}" -P "${PUBLISH_PORT}" \
      -o BatchMode=yes -o ConnectTimeout=15 \
      "${OUT}/result.json" "${OUT}/summary.md" "${PUBLISH_HOST}:${PUBLISH_DIR}/" \
    && echo "published to ${PUBLISH_HOST}:${PUBLISH_DIR}" \
    || echo "WARN: publish failed"
else
  echo "WARN: publishing disabled (set PTOAS_PUBLISH_HOST and a key at ${PUBLISH_KEY})"
fi
