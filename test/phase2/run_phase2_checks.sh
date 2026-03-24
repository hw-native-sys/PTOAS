#!/usr/bin/env bash
set -euo pipefail

ptoas_bin="./build/tools/ptoas/ptoas"
a5vm_ops_td="include/PTO/IR/A5VMOps.td"
filecheck_candidates=("FileCheck" "FileCheck-19" "/usr/lib/llvm-19/bin/FileCheck")
filecheck_bin=""

for candidate in "${filecheck_candidates[@]}"; do
  if command -v "${candidate}" >/dev/null 2>&1; then
    filecheck_bin="$(command -v "${candidate}")"
    break
  fi
  if [[ "${candidate}" = /* && -x "${candidate}" ]]; then
    filecheck_bin="${candidate}"
    break
  fi
done

if [[ -z "${filecheck_bin}" ]]; then
  echo "error: missing FileCheck; checked: ${filecheck_candidates[*]}" >&2
  exit 1
fi

if [[ ! -x "${ptoas_bin}" ]]; then
  echo "error: missing ./build/tools/ptoas/ptoas" >&2
  exit 1
fi

for required in CopyGmToUbuf CopyUbufToGm Vlds Vabs Vsts; do
  rg -n "def A5VM_${required}Op" "${a5vm_ops_td}" >/dev/null
done

if rg -n 'a5vm\.(load|store|abs)\b' "${a5vm_ops_td}" >/dev/null; then
  echo "error: legacy pseudo-op names detected in ${a5vm_ops_td}" >&2
  exit 1
fi

if rg -n 'a5vm\.(load|store|abs)\b|tabs_precheck\.mlir' test/phase2/*.mlir >/dev/null; then
  echo "error: obsolete Phase 2 fixture content detected" >&2
  exit 1
fi

if ! rg -n 'cce_aiv_loop_hint|llvm\.loop\.aivector_scope' test/phase2/tabs_abs_loop_shape.mlir >/dev/null; then
  echo "error: tabs_abs_loop_shape.mlir must require explicit AIV carrier strings" >&2
  exit 1
fi

if rg -n '^// CHECK(?:(?:-[A-Z]+)?)?: scf\.for$' test/phase2/tabs_abs_loop_shape.mlir >/dev/null; then
  echo "error: tabs_abs_loop_shape.mlir still checks bare scf.for nesting without vec-scope carrier details" >&2
  exit 1
fi

echo "phase2 check: tload_copy_family_shape.mlir"
"${ptoas_bin}" --pto-backend=a5vm --a5vm-print-ir test/phase2/tload_copy_family_shape.mlir -o /dev/null 2>&1 | \
  "${filecheck_bin}" test/phase2/tload_copy_family_shape.mlir

echo "phase2 check: tabs_abs_loop_shape.mlir"
"${ptoas_bin}" --pto-backend=a5vm --a5vm-print-ir test/phase2/tabs_abs_loop_shape.mlir -o /dev/null 2>&1 | \
  "${filecheck_bin}" test/phase2/tabs_abs_loop_shape.mlir

echo "phase2 check: tabs_precheck_a5.mlir"
"${ptoas_bin}" --pto-backend=a5vm test/phase2/tabs_precheck_a5.mlir -o /dev/null 2>&1 | \
  "${filecheck_bin}" test/phase2/tabs_precheck_a5.mlir

echo "phase2 check: tstore_copy_family_shape.mlir"
"${ptoas_bin}" --pto-backend=a5vm --a5vm-print-ir test/phase2/tstore_copy_family_shape.mlir -o /dev/null 2>&1 | \
  "${filecheck_bin}" test/phase2/tstore_copy_family_shape.mlir

echo "phase2 check: copy_dynamic_transfer_operands.mlir"
"${ptoas_bin}" --pto-backend=a5vm --a5vm-print-ir test/phase2/copy_dynamic_transfer_operands.mlir -o /dev/null 2>&1 | \
  "${filecheck_bin}" test/phase2/copy_dynamic_transfer_operands.mlir

echo "phase2 check: copy_dynamic_transfer_operands.mlir HIVM names"
"${ptoas_bin}" --pto-arch=a5 --pto-backend=a5vm --a5vm-emit-hivm-llvm test/phase2/copy_dynamic_transfer_operands.mlir -o - 2>/dev/null | \
  "${filecheck_bin}" --check-prefix=CHECK-HIVM test/phase2/copy_dynamic_transfer_operands.mlir

echo "phase2 check: tstore_domain_todos.mlir"
"${ptoas_bin}" --pto-backend=a5vm --a5vm-print-ir test/phase2/tstore_domain_todos.mlir -o /dev/null 2>&1 | \
  "${filecheck_bin}" test/phase2/tstore_domain_todos.mlir

echo "phase2 check: pto_backend_a5vm_wiring.mlir"
"${ptoas_bin}" --pto-backend=a5vm --a5vm-print-ir test/phase2/pto_backend_a5vm_wiring.mlir -o /dev/null 2>&1 | \
  "${filecheck_bin}" test/phase2/pto_backend_a5vm_wiring.mlir

echo "phase2 check: ctest"
ctest --test-dir build --output-on-failure
