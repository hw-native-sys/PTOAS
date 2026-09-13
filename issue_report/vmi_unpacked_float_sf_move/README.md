# D — Unpacked float SF move / unpacked FP4 store

PTOAS `9edc5a0`, TileLang `c1a4276c`, Nightly `adefcb7`, CANN 9.2.0, A5.

After TileKernels-vmi `quant_missing_impl_0913`, **per_channel unpacked
in-SF** (2 rows) is kernel-closed (`cast_back` compose, bitwise). Two
per_token leftovers remain in this dir (one hole: unpacked `V<L×f32>` /
unpacked FP4 payload that ASC already does):

1. **TMA-col unpacked SF** — production IR **emits** and PTOAS **lowers**.
   Isolated launch is bitwise-wrong on SF (**48 bytes** vs Nightly ASC).
2. **FP4 unpacked** — TileLang frontend rejects packed-FP4 `vstore` at
   **32 lanes** (`lanes one of 1,2,4,8,64,128,256`). No IR.

Host permute of row-major SF to TMA-col is not a port.

## Why this VMI should work

[PTO-vmi-Instruction-SPEC.md](../../../../kernel_study/PTO-Gym/docs/PTO-vmi-Instruction-SPEC.md)
`vload`/`vstore` `dist_mode` (`continuous` / `brc`). Layout is
compiler-held ([design §0.0](../../../../kernel_study/PTO-Gym-vmi-design/docs/PTO-vmi-design.en.md)).

`desired_vmi.ptodsl.py`: one row-major unpacked `vstore`, one TMA-col
unpacked `vstore`, one unpacked `vload` for rescale in-SF. Logical
`V<64×f32>`, not ASC ONEPT/BRC impersonation.

ASC Nightly L1 already stores/loads `V<L×f32>` scales without packing.
TMA-col is a compiler address / dist-mode.

## Working ASC reference

`asc_pattern.mi` is the physical target from Nightly
`per_token` / `per_block` / `per_channel` when `use_packed_ue8m0=False`:
`vsts ONEPT_B32` (row or transposed TMA-col) and `vld BRC_B32` for
rescale in-SF. `run_asc.py` records the committed pattern. Full C++ is
Nightly codegen for the same flags.

## Failing VMI evidence

**TMA-col unpacked** (`512×3072` e4m3, bf16, unpacked+round, TMA):

- `tilelang_dump.ptodsl.py` / `tilelang_dump.pto` — production TileLang
  PTODSL (`x_sf: ptr(f32)`; unpacked `vstore` of scales). Compile OK.
- Isolated remasure: `per_token ASC/VMI scale mismatch: 48 bytes`
  (`recorded.log`). Not a stub assert.

**FP4 unpacked** (`512×3072` e2m1, bf16, unpacked, row):

- `fp4_unpacked.err.txt` — frontend
  `T.vmi.vstore(...) packed FP4 physical lane count requires lanes to be
  one of (1, 2, 4, 8, 64, 128, 256); got 32`.
- No `kernel.ptodsl.py` (never reached codegen). `desired_vmi` is the
  legal target; a 64-lane packed FP4 store would be the production emit.

## High-level configs that still hit this (2 rows)

| Config | Recorded | Archive |
|---|---|---|
| per_token TMA-col unpacked SF (1) | compile OK, 48-byte SF mismatch | [`archive/per_token_tma_unpacked/`](../archive/per_token_tma_unpacked/) |
| per_token FP4 unpacked (1) | 32-lane packed FP4 `vstore` illegal | [`archive/per_token_fp4_unpacked/`](../archive/per_token_fp4_unpacked/) |
| per_channel unpacked in-SF (2) | **withdrawn** — compose bitwise | [`archive/per_channel_rescale_unpacked_in/`](../archive/per_channel_rescale_unpacked_in/) |

Exact leftover IDs: [`row_map.txt`](row_map.txt).

## Reproducer

```bash
./run_repro.sh vmi-compile
./run_repro.sh vmi 0
./run_repro.sh asc 0
```

`run_vmi.py` compile-checks `desired_vmi.ptodsl.py`. Production TMA dump
is `tilelang_dump.ptodsl.py`; FP4 frontend is `fp4_unpacked.err.txt`.
