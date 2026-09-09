# per_block sf_only+packed fp32

ASC Nightly L1 supports `Nightly L1 sf_only packed UE8M0, in_dtype=fp32`.

## Bug pattern

fp32 sf_only+packed bitwise vs ASC (bf16 sf_only+packed already matches, including TMA).

`tilelang_dump.ptodsl.py`: bf16 sf_only+packed row/TMA is a working production dump (ported this pass).

## Reproducer

```bash
./run_repro.sh vmi-compile
./run_repro.sh vmi 0
```

## Recorded result

Compile+launch OK; isolated SF mismatch vs ASC (1024 bytes at 512x2048 e4m3 row packed)

See `recorded.log`.
