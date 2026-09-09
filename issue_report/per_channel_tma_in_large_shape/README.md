# per_channel e4m3 rescale TMA-col input SF (large shape)

Production TileLang VMI for this leftover, recorded on PTOAS `b465f26b` / CANN 9.2.0 / A5.

## Bug pattern

Legal fused 1×T TMA-col SF unpack compiles and launches. Isolated 512×7168 fused vs ASC: payload_mismatch=47725 sf_mismatch=1536, confined to Persistent remainder wave (m-tile 7 × k-tiles 20–27). 8064×2048 fused is a periodic m-tile wave mismatch (tiles 27–35 / 54–62 / 81–89 / 108–116). Compose is not a workaround: it calls cast_back TMA npt=1, which mismatches at H≥2048. stages=1 / 32-core even-split legal reroutes made the fail worse.

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

PTODSL_COMPILE_OK; isolated fused 512×7168 payload=47725 sf=1536 (2026-09-10 device 1). Isolated compose 512×7168 payload=656904 (device 0). Isolated fused 8064×2048 payload=1656192 (device 0).

See `recorded.log`.
