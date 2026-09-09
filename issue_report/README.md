# Quant leftover PTOAS issues (0909 pins)

PTOAS `b465f26b`, CANN 9.2.0, TileLang `5038c468`, A5.

TileKernels-vmi coverage now treats a variant as **ported only when every measured shape is numerically correct**. Shape-specific leftovers therefore stay `TODO(impl)` in the top table. Isolated legal-VMI attempts (fused last-wave / even-split / stages=1; compose via `cast_back` TMA npt=1) did not close the new rows.

The original three `#7` `cast_back` dirs are unchanged. Three new dirs record leftovers that the old “any shape pass” rule hid.

Each directory has legal desired VMI (`desired_vmi.ptodsl.py` = production dump), TileLang PTODSL dump, Nightly ASC C++ when available (`asc_reference.cpp`), `run_repro.sh`, and recorded logs.

| Dir | Variant | Recorded |
|---|---|---|
| `cast_back_e4m3_fp32_tma_npt32_h128` | e4m3→fp32 TMA npt=32 H=128 | compile-OK, ACL 507035 |
| `cast_back_e4m3_fp32_tma_npt1` | e4m3→fp32 TMA npt=1 H=128 | compile-OK, ACL 507035 |
| `cast_back_row_npt1` | row npt=1 e2m1/e4m3→bf16/fp32 | compile-OK, numerical mismatch |
| `cast_back_tma_npt1_h2048` | TMA npt=1 e2m1/e4m3→bf16 @ 512×2048 | compile-OK, isolated mismatch (117550 / 124084) |
| `cast_back_e2m1_fp32_tma_npt32_h2048` | e2m1→fp32 TMA npt=32 @ 512×2048 | compile-OK, isolated mismatch (81987 / 3.0) |
| `per_channel_tma_in_large_shape` | e4m3 rescale TMA-col **input** SF, large tiles | fused 512×7168 last-wave mismatch; compose inherits TMA npt=1 |

Maps to [cann/pto-as issue #7](https://gitcode.com/cann/pto-as/issues/7) for the `cast_back` 1×T leftovers. The per_channel TMA-in large-shape case is a Persistent remainder / periodic-wave lowering leftover on the same legal 1×T unpack (not a VMI host permute).

`per_token` still has 12 ASC a5 FP4 TMA rescale rows gated at M=8001 (`num_tokens % 16 == 0`). Those stay “ported w/ shape limits” (alignment, not a measured fail). `per_block_cast` has no measured leftover.
