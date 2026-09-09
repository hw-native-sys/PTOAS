# per_channel e4m3 rescale unpacked row-major input SF

ASC Nightly L1 supports `Nightly L1 in_use_packed=False, in_tma=False`.

## Bug pattern

On-device dequant of unpacked float input SF then per-channel requant (ASC L1). Packed TMA-in fused path exists but fails on large tiles (see per_channel_tma_in_large_shape).

`tilelang_dump.ptodsl.py`: Working sibling is packed input-SF fused rescale (small tiles).

## Reproducer

```bash
./run_repro.sh vmi-compile
./run_repro.sh vmi 0
```

## Recorded result

AssertionError: VMI rescale needs packed UE8M0 input SF

See `recorded.log`.
