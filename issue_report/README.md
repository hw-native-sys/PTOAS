# Quant leftover PTOAS issues (0909 pins)

PTOAS `b465f26b`, CANN 9.2.0, TileLang `5038c468`, A5.

TileKernels-vmi coverage treats a variant as **ported only when every ASC-supported config of that variant has a working in-kernel VMI counterpart**. Host-pad is not a port unless ASC also host-pads. Shape-specific leftovers and in-code gates therefore stay `TODO(impl)`. Isolated legal-VMI attempts (fused last-wave / even-split / stages=1; compose via `cast_back` TMA npt=1; in-kernel e4m3 tail `vbrc`/`T.clear`) did not close the new rows.

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
| `per_token_fp4_rescale_m8001` | fused rescale M=8001 (ASC ceildiv, no host pad) | `vbrc(f8e4m3(0))` / `T.clear` e4m3 UB do not lower |
| `per_block_h384` | per_block H=384 128-strip (ASC align-128, no host pad) | TileLang OK, PTOAS `VMI-RESIDUAL-OP` |

Maps to [cann/pto-as issue #7](https://gitcode.com/cann/pto-as/issues/7) for the `cast_back` 1×T leftovers. The per_channel TMA-in large-shape case is a Persistent remainder / periodic-wave lowering leftover on the same legal 1×T unpack (not a VMI host permute).

`per_token` has 12 ASC a5 FP4 TMA rescale rows at M=8001 (`8001 % 16 == 1`). ASC uses in-kernel `ceildiv` (no host pad). VMI cannot lower an e4m3/e2m1 last-tile zero (`vbrc(f8e4m3(0))` / `T.clear` fp8). Those 4 variants are **TODO(impl)** — see `per_token_fp4_rescale_m8001/`.

`per_block` ASC a5 includes H=384. ASC uses 128-aligned `block_k` in-kernel. VMI’s H=128 128-strip hits `VMI-RESIDUAL-OP` at H=384. Those variants are **TODO(impl)** — see `per_block_h384/`. Host-pad is not a port.
