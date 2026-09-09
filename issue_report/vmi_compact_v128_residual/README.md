# B — Compact `V<128×T>` leftover strip

PTOAS `b465f26b` / CANN 9.2.0 / TileLang `5038c468` / A5.

ASC tiles hidden in 128-wide chunks (`hidden % 128 == 0`). VMI design allows compact `V<128×bf16>` (`K_raw = 1`). PTOAS reports `VMI-RESIDUAL-OP` when that strip is used at H=384. Per-token never emits IR (`hidden % 256 == 0` assert). Host-pad is not a port.

## VMI design

[PTO-vmi-design.en.md](../../../../kernel_study/PTO-Gym-vmi-design/docs/PTO-vmi-design.en.md) §1.1: compact/partial vregs smaller than 256 B are legal; physical backing is still one 256 B vreg. `create_mask(128)` is in the legal set `{1,64,128,256}`.

`desired_vmi.ptodsl.py` is a 128-lane load/amax VF, not a copied H=3072 dump.

## ASC reference (working)

`asc_reference.cpp` is Nightly `per_block_cast_asc` at H=384: `block_k` aligned to 128, GM copy `src_stride=768`, dest 256 B. `run_asc.py` launches it (`ASC_LAUNCH_OK`).

`asc_pattern.mi`: `vlds_norm` 128-lane + `vsts_pack_quarter`. Three K tiles for H=384. No host pad.

## High-level configs that hit this

| Config | Recorded | Archive |
|---|---|---|
| per_token H=128/384 (every bf16/fp32 cast layout; 40 rows) | assert `hidden % 256 == 0` | [`archive/per_token_h128_h384/`](../archive/per_token_h128_h384/) |
| per_block H=384 128-strip (8 rows) | TileLang OK, `VMI-RESIDUAL-OP` | [`archive/per_block_h384/`](../archive/per_block_h384/) |

## Reproducer

```bash
./run_repro.sh vmi-compile
./run_repro.sh vmi 0
./run_repro.sh asc 0
```

See `recorded.log` (representative: per_block 512×384, `VMI-RESIDUAL-OP`). H=128 dump appendix: `tilelang_dump.ptodsl.py`.
