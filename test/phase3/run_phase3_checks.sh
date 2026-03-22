#!/usr/bin/env bash
set -euo pipefail

ptoas_bin="./build/tools/ptoas/ptoas"
work_dir="$(mktemp -d)"
trap 'rm -rf "${work_dir}"' EXIT

if [[ ! -x "${ptoas_bin}" ]]; then
  echo "error: missing ./build/tools/ptoas/ptoas" >&2
  exit 1
fi

echo "phase3 check: llvm_wrapper_abi_abs.pto"
"${ptoas_bin}" --pto-arch a5 --pto-backend=a5vm --a5vm-emit-hivm-llvm \
  test/phase3/llvm_wrapper_abi_abs.pto -o "${work_dir}/abs.ll" >/dev/null 2>/dev/null
rg -n '^define dso_local void @abs_wrapper_abi\(ptr addrspace\(1\) %arg0, ptr addrspace\(1\) %arg1\)' "${work_dir}/abs.ll" >/dev/null
if rg -n '__ptoas_impl_|insertvalue' "${work_dir}/abs.ll" >/dev/null; then
  echo "error: abs LLVM output still contains internal ABI bridge artifacts" >&2
  exit 1
fi
if rg -n 'addrspacecast ptr addrspace\(1\).*to ptr|addrspacecast ptr .*to ptr addrspace\(1\)' "${work_dir}/abs.ll" >/dev/null; then
  echo "error: abs LLVM output still contains GM addrspace cast round-trips" >&2
  exit 1
fi

echo "phase3 check: llvm_wrapper_abi_dynamic_tabs.pto"
"${ptoas_bin}" --pto-arch a5 --pto-backend=a5vm --a5vm-emit-hivm-llvm \
  test/phase3/llvm_wrapper_abi_dynamic_tabs.pto -o "${work_dir}/dynamic.ll" >/dev/null 2>/dev/null
rg -n '^define dso_local void @dynamic_tabs_wrapper_abi\(ptr addrspace\(1\) %arg0, ptr addrspace\(1\) %arg1, i32 %arg2\)' "${work_dir}/dynamic.ll" >/dev/null
if rg -n '__ptoas_impl_|insertvalue' "${work_dir}/dynamic.ll" >/dev/null; then
  echo "error: dynamic LLVM output still contains internal ABI bridge artifacts" >&2
  exit 1
fi
if rg -n 'addrspacecast ptr addrspace\(1\).*to ptr|addrspacecast ptr .*to ptr addrspace\(1\)' "${work_dir}/dynamic.ll" >/dev/null; then
  echo "error: dynamic LLVM output still contains GM addrspace cast round-trips" >&2
  exit 1
fi
