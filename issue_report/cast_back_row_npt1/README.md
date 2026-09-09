# cast_back row-major npt=1 (e2m1/e4m3 → bf16/fp32)

Production TileLang VMI for this leftover, recorded on PTOAS `b465f26b` / CANN 9.2.0 / A5.

## Bug pattern

Legal 1×T UE8M0 unpack compiles and launches; payload mismatches ASC (512×2048 isolated): e4m3→bf16 count=326207 max_abs~1e37; e2m1→bf16 count=736363 max_abs=6; e4m3→fp32 count=324427 max_abs~1e37; e2m1→fp32 count=737043 max_abs=6. Same root 1×T scale-apply.

`tilelang_dump.ptodsl.py` is the real kernel dump (`wrapper.pto_kernel_source`).
It is already legal logical VMI: `create_mask` widths in {1,64,128,256}, compact
UE8M0 extract as **1×T** (`vload` size=1 → `vshl` → store), no PART/ONEPT/PK
impersonation. That file is also the desired VMI; PTOAS should compile it and
match the ASC reference.

## ASC reference

`asc_reference.cpp` is TileLang `target='ascend'` / Nightly `cast_back_asc`
codegen for the same logical kernel. `run_asc.py` launches that Nightly ASC
kernel and prints `ASC_LAUNCH_OK`.

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

PTODSL_COMPILE_OK; device launch returns; golden-vs-got bytes mismatch (not 507035 when isolated)

See `recorded.log`.
