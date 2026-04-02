Qwen3 scope2 PTO kernels generated from `pypto-lib/examples/models/qwen3/qwen3_32b_decode_scope2.py`.

Scope:
- compile-regression inputs for `ptoas`
- A5-only kernels; `runop.sh` injects `--pto-arch a5` for this directory unless the caller already overrides `PTOAS_FLAGS`

Notes:
- The source PyPTO program lowers to 13 kernel-level `.pto` files plus an orchestration C++ file.
- This sample directory vendors only the kernel `.pto` inputs.
- No custom `golden.py` or `compare.py` is included in this draft because those are tied to the full orchestration flow, not to individual kernel-only `.pto` files.
- The existing `test/npu_validation/scripts/generate_testcase.py` flow can still auto-generate generic validation assets for these kernels when needed.
