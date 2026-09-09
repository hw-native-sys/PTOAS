# cast_back e2m1→fp32 TMA npt=32 H=2048

Production TileLang VMI for this leftover, recorded on PTOAS `b465f26b` / CANN 9.2.0 / A5.

## Bug pattern

Legal 1×T UE8M0 unpack. H=128 of this variant is numerically correct (TODO(perf) ~0.85). Isolated 512×2048 mismatches ASC: count=81987 max_abs=3.0 (jsonl sweep had aclnnInplaceNormal after a prior fault; the isolated process is a real payload mismatch, not a cascade).

`tilelang_dump.ptodsl.py` is the real kernel dump (`wrapper.pto_kernel_source`).
It is already legal logical VMI: `create_mask` widths in {1,64,128,256}, compact
UE8M0 extract as **1×T** (`vload` size=1 → `vshl` → store), no PART/ONEPT/PK
impersonation. That file is also the desired VMI; PTOAS should compile it and
match the ASC reference.

## ASC reference

`asc_reference.cpp` is TileLang `target='ascend'` / Nightly codegen for the same
logical kernel. `run_asc.py` launches that Nightly ASC kernel and prints
`ASC_LAUNCH_OK`.

## Reproducer

```bash
./run_repro.sh vmi-compile
./run_repro.sh vmi 1
./run_repro.sh asc 1
```

Set `REPRO_ROOT` to the pto-as checkout, `CANN_HOME` to CANN 9.2.0,
`PTOAS_BIN` to the built ptoas, and `TILELANG_ROOT` so the dumped PTODSL
can import `tilelang.contrib.ptodsl`.

## Recorded result

PTODSL_COMPILE_OK; isolated 512×2048 mismatch count=81987 max_abs=3.0 (2026-09-10 device 0)

See `recorded.log`.
