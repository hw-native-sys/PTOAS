#!/usr/bin/env bash
set -euo pipefail

ptoas_bin="./build/tools/ptoas/ptoas"

if [[ ! -x "${ptoas_bin}" ]]; then
  echo "error: missing ./build/tools/ptoas/ptoas" >&2
  exit 1
fi

legacy_guard_pattern='a5vm\.\(load\|abs\|store\)'
legacy_guard_regex='a5vm\.(load|abs|store)'

rg -n "${legacy_guard_regex}" test/phase1/*.mlir >/dev/null && {
  echo "error: obsolete pseudo-op fixture content detected under test/phase1" >&2
  exit 1
} || true

echo "phase1 check: a5vm_vec_type.mlir"
"${ptoas_bin}" test/phase1/a5vm_vec_type.mlir 2>&1 | FileCheck test/phase1/a5vm_vec_type.mlir

echo "phase1 check: a5vm_copy_gm_to_ubuf_op.mlir"
"${ptoas_bin}" test/phase1/a5vm_copy_gm_to_ubuf_op.mlir -o - | FileCheck test/phase1/a5vm_copy_gm_to_ubuf_op.mlir

echo "phase1 check: a5vm_vabs_kernel_shape.mlir"
"${ptoas_bin}" test/phase1/a5vm_vabs_kernel_shape.mlir -o - | FileCheck test/phase1/a5vm_vabs_kernel_shape.mlir

echo "phase1 check: a5vm_copy_ubuf_to_gm_op.mlir"
"${ptoas_bin}" test/phase1/a5vm_copy_ubuf_to_gm_op.mlir -o - | FileCheck test/phase1/a5vm_copy_ubuf_to_gm_op.mlir

echo "phase1 check: a5vm_backend_switch.mlir"
backend_switch_output="$(mktemp)"
"${ptoas_bin}" --pto-backend=a5vm test/phase1/a5vm_backend_switch.mlir -o - > "${backend_switch_output}"
FileCheck test/phase1/a5vm_backend_switch.mlir < "${backend_switch_output}"
if rg -n "llvm\\.hivm" "${backend_switch_output}" >/dev/null; then
  echo "error: backend switch emitted deferred HIVM text instead of corrected A5VM text" >&2
  rm -f "${backend_switch_output}"
  exit 1
fi
rm -f "${backend_switch_output}"

echo "phase1 check: a5vm_shared_dialects.mlir"
"${ptoas_bin}" --pto-backend=a5vm --a5vm-print-ir test/phase1/a5vm_shared_dialects.mlir -o /dev/null 2>&1 | \
  FileCheck test/phase1/a5vm_shared_dialects.mlir
