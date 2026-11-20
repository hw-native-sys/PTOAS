#!/usr/bin/env bash
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# Exercise the bridge passes in separate processes, then link independent units.
# Paths and tools are supplied by lit; no CANN installation or host JIT is used.
set -euo pipefail

fixture="$1"
output="$2"
python_bin="$3"
optimizer="$4"
linker="$5"
helper="$(dirname -- "${BASH_SOURCE[0]}")/check_bridge_symbol_namespace.py"
directory="${output}.namespace"
mode="${6:-ir}"

if [[ "${mode}" == "ptodsl" ]]; then
    mkdir -p "${directory}"
    "${python_bin}" "${fixture}" > "${directory}/container.pto"
    pipeline='builtin.module(builtin.module(func.func(pto-lower-frontend-pipe-ops)),'
    pipeline+='pto-infer-validate-pipe-init,pto-plan-memory,pto-resolve-reserved-buffers)'
    "${optimizer}" "${directory}/container.pto" \
        "--pass-pipeline=${pipeline}" \
        > "${directory}/container.lowered.pto"
    "${python_bin}" "${helper}" split-ptodsl "${directory}/container.lowered.pto" "${output}"
    variants=(vector cube)
else
    "${python_bin}" "${helper}" prepare "${fixture}" "${output}"
    variants=(original other_unit other_spec other_core other_payload location argument_location)
fi
for variant in "${variants[@]}"; do
    "${optimizer}" "${directory}/${variant}.pto" \
        --pto-lower-bridge-ops --vpto-resolve-bridge-instances > "${directory}/${variant}.resolved"
    "${optimizer}" "${directory}/${variant}.pto" \
        --pto-lower-bridge-ops --vpto-resolve-bridge-instances \
        --vpto-bridge-wrapper-gen > "${directory}/${variant}.wrapper"
    "${optimizer}" "${directory}/${variant}.pto" \
        --pto-lower-bridge-ops --vpto-resolve-bridge-instances \
        --vpto-bridge-call-lowering > "${directory}/${variant}.calls"
done
if [[ "${mode}" == "ir" ]]; then
    "${optimizer}" "${directory}/original.pto" \
        --pto-lower-bridge-ops --vpto-resolve-bridge-instances > "${directory}/repeat.resolved"
    # Bytecode preserves the resolved IR even when textual output omits regions.
    # Decode in a fresh process so comparisons use the normal printing settings.
    for variant in "${variants[@]}"; do
        "${optimizer}" "${directory}/${variant}.pto" \
            --pto-lower-bridge-ops --vpto-resolve-bridge-instances --emit-bytecode \
            --mlir-print-skip-regions --mlir-print-value-users --mlir-print-unique-ssa-ids \
            --mlir-elide-elementsattrs-if-larger=0 --mlir-print-elementsattrs-with-hex-if-larger=0 \
            --mlir-elide-resource-strings-if-larger=0 --mlir-print-debuginfo \
            > "${directory}/${variant}.printing.bc"
        "${optimizer}" "${directory}/${variant}.printing.bc" > "${directory}/${variant}.printing.resolved"
        cmp "${directory}/${variant}.resolved" "${directory}/${variant}.printing.resolved"
    done
fi
"${python_bin}" "${helper}" "check-${mode}" "${fixture}" "${output}"
for unit in 0 1; do
    "${linker}" "${directory}/device${unit}.ll" "${directory}/wrapper${unit}.ll" \
        -o "${directory}/unit${unit}.bc"
done
"${linker}" "${directory}/main.ll" "${directory}/unit0.bc" "${directory}/unit1.bc" \
    -S -o "${directory}/linked.ll"
"${python_bin}" "${helper}" "check-linked-${mode}" "${fixture}" "${output}"
