Qwen3 tilelet PTO kernels generated from `pypto-lib/examples/models/qwen3/qwen3_32b_decode_tilelet.py`.

Scope:
- compile-regression inputs for `ptoas`
- A5-only kernels; `runop.sh` injects `--pto-arch a5 --pto-level=level3` for this directory unless the caller already overrides `PTOAS_FLAGS`

Notes:
- The source PyPTO program lowers to a full orchestration file plus 5 ptoas-facing mixed-kernel `.pto` inputs:
  `qwen3_decode_layer_incore_1`, `qwen3_decode_layer_incore_2`,
  `qwen3_decode_layer_incore_10`, `qwen3_decode_layer_incore_13`,
  `qwen3_decode_layer_incore_14`.
- This sample directory vendors only those direct `ptoas` regression inputs.
- `test/npu_validation/scripts/generate_testcase.py` now wraps the paired `_aic`/`_aiv` entrypoints into a standalone mixed-kernel launch wrapper for board validation.
- Custom golden assets follow the normal sample convention and live beside the `.pto` files as `<case>_golden.py`.
