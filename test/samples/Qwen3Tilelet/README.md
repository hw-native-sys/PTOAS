Qwen3 tilelet PTO kernels generated from `pypto-lib/examples/models/qwen3/qwen3_32b_decode_tilelet.py`.

Scope:
- compile-regression inputs for `ptoas`
- tilelet kernels that default to `--pto-arch a5 --pto-level=level3` in `runop.sh`, but can also be compiled on A3 when the caller overrides `PTOAS_FLAGS`

Notes:
- The source PyPTO program lowers to 20 `qwen3_decode_layer_incore_*.pto` fragments; this directory vendors the full emitted `.pto` set regenerated from the tilelet source with `BATCH_TILE=16`.
- `test/npu_validation/scripts/generate_testcase.py` wraps the paired `_aic`/`_aiv` entrypoints into a standalone mixed-kernel launch wrapper for board validation when the lowered fragment contains split cube/vector entrypoints.
- Custom golden assets currently exist only for the board-validation cases that need them and live beside the `.pto` files as `<case>_golden.py`.
