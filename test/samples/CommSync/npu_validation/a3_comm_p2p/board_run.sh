#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set -euo pipefail

kernel_cpp="${1:?pass the PTOAS-generated C++ source}"
output_dir="${2:?pass the output directory}"
isa_root="${3:?pass the PTO-ISA checkout}"
stage="${4:?pass build or run}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
use_nightly_devices=false
if [[ "${stage}" == run && -z "${TASKQUEUE_INSIDE:-}" ]]; then
    # This job already has exclusive devices assigned by the runner. Other
    # entry points on shared boards must obtain their devices from TaskQueue.
    repository="${ATOMGIT_REPOSITORY:-}"
    job="${ATOMGIT_JOB:-}"
    # GitCode runners can expose the job ID or the displayed job name.
    if [[ "${ATOMGIT_ACTIONS:-}" == true && ( "${repository}" == cann/pto-as || "${repository}" == CANN/pto-as ) &&
          "${ATOMGIT_WORKFLOW:-}" == "A3 Nightly Board" &&
          ( "${job}" == a3-board || "${job}" == "A3 nightly board validation" ) ]]; then
        use_nightly_devices=true
    elif ! command -v task-submit >/dev/null 2>&1; then
        echo "A3 P2P requires task-submit to reserve two devices" >&2
        exit 1
    fi
fi

mkdir -p "${output_dir}"
output_dir="$(cd "${output_dir}" && pwd)"
bisheng -std=c++17 -O2 -xcce --cce-aicore-arch=dav-c220-vec -DMEMORY_BASE -Xhost-start -Xhost-end \
    -mllvm -cce-aicore-addr-transform -mllvm -cce-aicore-dcci-insert-for-scalar=false \
    -I"${isa_root}/include" -I"${ASCEND_HOME_PATH}/include" -I"$(dirname "${kernel_cpp}")" \
    -c "${script_dir}/DeviceBuffer.cpp" -o "${output_dir}/DeviceBuffer.o"
bisheng --cce-fatobj-link "${output_dir}/DeviceBuffer.o" -L"${ASCEND_HOME_PATH}/lib64" \
    -lascendcl -lruntime -lstdc++ -o "${output_dir}/a3_comm_p2p"

if [[ "${stage}" == run ]]; then
    if [[ -n "${TASKQUEUE_INSIDE:-}" ]]; then
        exec "${output_dir}/a3_comm_p2p"
    fi
    if [[ "${use_nightly_devices}" == true ]]; then
        echo "A3 P2P using devices assigned to the GitCode A3 Nightly job"
        exec timeout --kill-after=10s 120s "${output_dir}/a3_comm_p2p" --job-devices
    fi
    printf -v command 'cd %q && exec ./a3_comm_p2p' "${output_dir}"
    task_env=(--env LD_LIBRARY_PATH)
    for mask in ASCEND_VISIBLE_DEVICES ASCEND_RT_VISIBLE_DEVICES; do
        if [[ -n "${!mask:-}" ]]; then
            task_env+=(--env "${mask}")
        fi
    done
    task-submit --device auto --device-num 2 --timeout 180 --max-time 120 "${task_env[@]}" --run "${command}" \
        2>&1 | tee "${output_dir}/task.log"
    if ! grep -Fxq "A3 P2P TPUT/TGET verification passed" "${output_dir}/task.log"; then
        echo "A3 P2P validation result missing from task output" >&2
        exit 1
    fi
fi
