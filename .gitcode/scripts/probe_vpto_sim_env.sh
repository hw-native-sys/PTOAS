#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set -euo pipefail

echo "=== VPTO SIM environment probe ==="
echo "CPU parallelism:"
if command -v nproc >/dev/null 2>&1; then
  echo "  nproc: $(nproc)"
else
  echo "  nproc: unavailable"
fi
if command -v getconf >/dev/null 2>&1; then
  echo "  online processors: $(getconf _NPROCESSORS_ONLN 2>/dev/null || echo unavailable)"
else
  echo "  online processors: unavailable"
fi
if [[ -r /sys/fs/cgroup/cpu.max ]]; then
  echo "  cgroup CPU quota: $(tr ' ' '/' < /sys/fs/cgroup/cpu.max)"
elif [[ -r /sys/fs/cgroup/cpu/cpu.cfs_quota_us && -r /sys/fs/cgroup/cpu/cpu.cfs_period_us ]]; then
  echo "  cgroup CPU quota: $(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us)/$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us)"
else
  echo "  cgroup CPU quota: unavailable"
fi
printf 'python: '
command -v python3 || true
python3 --version || true

if [[ -n "${ASCEND_HOME_PATH:-}" && -d "${ASCEND_HOME_PATH}" ]]; then
  echo "ASCEND_HOME_PATH=${ASCEND_HOME_PATH}"
else
  echo "ASCEND_HOME_PATH is not set"
fi
if [[ -d /usr/local/Ascend ]]; then
  echo "/usr/local/Ascend entries:"
  find /usr/local/Ascend -maxdepth 1 -mindepth 1 -printf '  %f\n' 2>/dev/null | sort
else
  echo "/usr/local/Ascend: unavailable"
fi
if [[ -d /home/jenkins/Ascend ]]; then
  echo "/home/jenkins/Ascend entries:"
  find /home/jenkins/Ascend -maxdepth 1 -mindepth 1 -printf '  %f\n' 2>/dev/null | sort
  for tool in bisheng msprof; do
    found_tool="$(find /home/jenkins/Ascend -type f -name "${tool}" -perm -u+x 2>/dev/null | sort | head -n 1)"
    if [[ -n "${found_tool}" ]]; then
      echo "${tool} under /home/jenkins/Ascend: ${found_tool}"
    fi
  done
  sim_path="$(find /home/jenkins/Ascend -type d -path '*/simulator/dav_3510/lib' 2>/dev/null | sort | head -n 1)"
  if [[ -n "${sim_path}" ]]; then
    echo "dav_3510 under /home/jenkins/Ascend: ${sim_path}"
  fi
else
  echo "/home/jenkins/Ascend: unavailable"
fi

for tool in bisheng msprof npu-smi; do
  if command -v "${tool}" >/dev/null 2>&1; then
    echo "${tool}: $(command -v "${tool}")"
  else
    echo "${tool}: unavailable"
  fi
done

readarray -t sim_libs < <(
  if [[ -n "${ASCEND_HOME_PATH:-}" && -d "${ASCEND_HOME_PATH}" ]]; then
    find "${ASCEND_HOME_PATH}" -type d -path '*/simulator/dav_3510/lib' 2>/dev/null | sort
  fi
)
if [[ "${#sim_libs[@]}" -eq 0 ]]; then
  echo "dav_3510 simulator: unavailable"
else
  printf 'dav_3510 simulator: %s\n' "${sim_libs[0]}"
fi

python3 - <<'PY'
import importlib.util

for name in ("torch", "torch_npu"):
    spec = importlib.util.find_spec(name)
    print(f"{name}: {'available' if spec else 'unavailable'}")
    if not spec:
        continue
    try:
        module = __import__(name)
    except Exception as exc:  # Environment probe must never fail the build gate.
        print(f"{name} import failed: {exc}")
        continue
    print(f"{name} version: {getattr(module, '__version__', 'unknown')}")
PY

echo "=== VPTO SIM environment probe complete ==="
