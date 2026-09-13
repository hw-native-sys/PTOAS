# B — Compact packed leftover strip (`VMI-RESIDUAL-OP`)

PTOAS `9edc5a0`, TileLang `c1a4276c`, Nightly `adefcb7`, CANN 9.2.0, A5.

After TileKernels-vmi `quant_missing_impl_0913`, **unpacked** compact H=128/384
and **per_block H=384** are kernel-closed (dual legal `V<64×T>` strips,
bitwise vs ASC). The remaining hole is **per_token packed** compact
H=128/384: production TileLang emits legal dual-`V<64>` packed IR, and
PTOAS still reports `VMI-RESIDUAL-OP`. Host-pad to 256 is not a port.

Fused compact `(32,32)` TMA at H=128/384 is the same leftover class
(frontend `vand` mask/vector lane mismatch on the packed path).

## Why this VMI should work

[PTO-vmi-design.en.md](../../../../kernel_study/PTO-Gym-vmi-design/docs/PTO-vmi-design.en.md)
§1.1: compact/partial vregs smaller than 256 B are legal; physical backing
is still one 256 B vreg. `create_mask(128)` is in the legal set
`{1,64,128,256}`. `K_raw = 128 * 16 / 2048 = 1`.

`desired_vmi.ptodsl.py` is a 128-lane load/amax VF (design target), not a
copied H=3072 dump and not an ASC PART/PK impersonation.

Production emit (this dir's `tilelang_dump.ptodsl.py`) already avoids
`V<128>`: it uses two `V<64×bf16>` strips + packed UE8M0 `vstore`. That
is still logical VMI. PTOAS must lower it without `VMI-RESIDUAL-OP`.

## Working ASC reference

`asc_reference.cpp` is Nightly `per_block_cast_asc` at H=384: `block_k`
aligned to 128, GM copy `src_stride=768`, dest 256 B. `run_asc.py`
launches it (`ASC_LAUNCH_OK`). The same 128-wide K tiling is what
Nightly `per_token_cast_asc` uses at H=128 (one tile) and H=384 (three).

`asc_pattern.mi`: `vlds_norm` 128-lane + `vsts_pack_quarter`. No host pad.
This is the `pto.mi` lowering target (“no worse than this”).

## Failing VMI evidence

Production per_token packed H=128 row-major (`e4m3`, bf16, npc=32, M=512):

- `tilelang_dump.ptodsl.py` / `tilelang_dump.pto` — TileLang codegen of
  the **production** kernel (dual `V<64>`, packed UE8M0).
- `ptoas_stderr.txt` — `ptoas --pto-arch=a5 --pto-backend=vpto --pto-level=level3`
  → `VMI-RESIDUAL-OP` at `kernel.pto:116`.
- Fused compact TMA sibling: `T.vmi.vand(...) requires mask and vector
  lane counts to match` (same leftover class; see `recorded.log`).

## High-level configs that still hit this (24 rows)

| Config | Recorded | Archive |
|---|---|---|
| per_token packed H=128/384, e4m3/e2m1 × bf16/fp32 × row/TMA (16) | `VMI-RESIDUAL-OP` | [`archive/per_token_h128_h384/`](../archive/per_token_h128_h384/) |
| per_token fused packed `(32,32)` TMA compact H=128/384 (8) | `vand` lane mismatch | same |
| per_block H=384 (8) | **withdrawn** — dual `V<64>` bitwise | [`archive/per_block_h384/`](../archive/per_block_h384/) |
| per_token unpacked compact H=128/384 (~16) | **withdrawn** — dual `V<64>` bitwise | same leftover-class dir |

Exact leftover IDs: [`row_map.txt`](row_map.txt).

## Reproducer

```bash
./run_repro.sh vmi-compile
./run_repro.sh vmi 0
./run_repro.sh asc 0
```

`run_vmi.py` compile-checks `desired_vmi.ptodsl.py`. Production fail is
`ptoas_stderr.txt` / `recorded.log`.
