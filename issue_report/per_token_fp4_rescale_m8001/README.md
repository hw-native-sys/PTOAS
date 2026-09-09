# per_token fused FP4/e4m3 rescale M=8001 (in-kernel tail)

ASC Nightly a5 supports quantized-in TMA packed rescale at `M=8001` (`8001 % 16 == 1`) with in-kernel `T.ceildiv(num_tokens, block_m=16)`. ASC does **not** host-pad. VMI fused rescale only lowers when `M % 16 == 0`.

## Bug pattern

Legal 1×T fused rescale (aligned M) compiles and matches ASC bitwise. The missing counterpart is an in-kernel last-tile tail:

1. `vbrc(f8e4m3(0))` / `vbrc(f4e2m1(0))` — PTODSL eager constructor
   `TypeError: unsupported eager constructor target type f8E4M3FN`
2. `T.clear(e4m3 UB)` — TileLang codegen
   `Fatal: Bad bit-width for float: float8_e4m3fn`

Host-pad to M=8016 is numerically correct but is **not** a port: ASC has no host torch padding.

`tilelang_dump.ptodsl.py` is the working aligned fused kernel (512×3072 e4m3→e4m3 TMA packed). That IR is already legal 1×T. `desired_vmi.ptodsl.py` is the same dump plus the tail ops PTOAS/TileLang must accept so Persistent `ceildiv(8001, 16)` can zero unused e4m3/e2m1 UB rows.

## ASC reference

`asc_reference.cpp` is Nightly `per_token_cast_asc` codegen (`T.ceildiv`, clamped DMA, no host pad). `run_asc.py` launches that kernel at M=8001.

## Reproducer

```bash
./run_repro.sh vmi-compile
./run_repro.sh vmi 1
./run_repro.sh asc 1
```

## Recorded result

- `vbrc(e4m3 0)` isolate: PTODSL compile-only lowering failed (`f8E4M3FN` constructor) — 2026-09-10 device 0
- `T.clear(e4m3 UB)` isolate: `Bad bit-width for float: float8_e4m3fn` — 2026-09-10 device 0
- Host-pad M=8016 (rejected as a port) was bitwise vs ASC; not used in coverage

See `recorded.log`.
