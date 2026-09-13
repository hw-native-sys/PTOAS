# D — Unpacked float scale `vload` / `vstore`

PTOAS `b465f26b` (recorded) / still open on `9edc5a0`, TileLang `c1a4276c`, CANN 9.2.0, A5. No kernel workaround after PRs 89–91 (unpacked in-SF still gated).

ASC Nightly L1 stores and loads `V<L×f32>` scales without packing. TMA-col is a compiler address / `dist-mode`, not a host permute of row-major GM. Production VMI asserts packed-only and never emits IR.

## VMI design

[PTO-vmi-Instruction-SPEC.md](../../../../kernel_study/PTO-Gym/docs/PTO-vmi-Instruction-SPEC.md) `vload`/`vstore` `dist_mode` (`continuous` / `brc`). Layout is compiler-held ([design §0.0](../../../../kernel_study/PTO-Gym-vmi-design/docs/PTO-vmi-design.en.md)).

`desired_vmi.ptodsl.py`: one row-major unpacked `vstore`, one TMA-col unpacked `vstore`, one unpacked `vload` for rescale in-SF.

## ASC reference (working pattern)

Nightly ASC already launches these configs (level-1). There is no production VMI dump.

`asc_pattern.mi` is the physical target from Nightly `per_block_cast_asc` / `per_token` / `per_channel` when `use_packed_ue8m0=False`: `vsts ONEPT_B32` (row or transposed TMA-col) and `vld BRC_B32` for rescale in-SF.

`run_asc.py` records the committed pattern. Full C++ is Nightly codegen for the same flags (`use_packed=False`, `use_tma` true or false).

## High-level configs that hit this

| Config | Recorded | Archive |
|---|---|---|
| per_token TMA-col unpacked SF | assert: packed required for TMA adapter | [`archive/per_token_tma_unpacked/`](../archive/per_token_tma_unpacked/) |
| per_token FP4 unpacked SF | assert: packed UE8M0 required for e2m1 | [`archive/per_token_fp4_unpacked/`](../archive/per_token_fp4_unpacked/) |
| per_channel rescale unpacked row-major in-SF (2 rows) | assert: packed input SF required | [`archive/per_channel_rescale_unpacked_in/`](../archive/per_channel_rescale_unpacked_in/) |

## Reproducer

```bash
./run_repro.sh vmi-compile
./run_repro.sh vmi 0
./run_repro.sh asc 0
```

See `recorded.log` (representative: TMA-unpacked assert). `tilelang_dump.ptodsl.py` is a stub — no IR.
