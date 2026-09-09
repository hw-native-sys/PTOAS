# per_block H=384 (128-wide K strip)

ASC Nightly a5 includes `H=384` (`384 % 256 == 128`). ASC uses `block_k = align(gcd(limit, hidden), 128)` and `T.ceildiv(hidden, block_k)` — in-kernel 128-wide chunks, **no host pad**.

VMI per_block uses a 256-lane strip except for `H=128` (special-cased 128-lane strip). Extending that legal 128-strip to `H=384` (`384 % 128 == 0`) compiles TileLang then dies in PTOAS:

```
VMI-RESIDUAL-OP: failed to convert all VMI ops/types to VPTO
```

Host-pad H=384→512 is **not** a port (ASC has no host padding).

## Bug pattern

`tilelang_dump.ptodsl.py` is the working H=128 128-lane strip (legal 1×T, `create_mask` in {1,64,128,256}). That is the desired strip for H=384. PTOAS must lower the same 128-wide VF when `hidden=384` (three K tiles of 128).

## ASC reference

`asc_reference.cpp` is Nightly `per_block_cast_asc` for H=384 (`block_k` aligned to 128). `run_asc.py` launches that kernel.

## Reproducer

```bash
./run_repro.sh vmi-compile
./run_repro.sh vmi 1
./run_repro.sh asc 1
```

## Recorded result

Isolated VMI `512×384` bf16→e4m3 unpacked: TileLang compile-OK, PTOAS `VMI-RESIDUAL-OP` (2026-09-10 device 1, PTOAS `b465f26b`, CANN 9.2.0).

See `recorded.log`.
