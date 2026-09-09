# per_token FP4 unpacked SF

ASC Nightly L1 supports `Nightly L1 fmt=e2m1 use_packed=False`.

## Bug pattern

On-device e2m1 quant with unpacked float SF (ASC L1). Packed UE8M0 e2m1 cast is already ported.

`tilelang_dump.ptodsl.py`: Working sibling is packed UE8M0 e2m1 full-cast.

## Reproducer

```bash
./run_repro.sh vmi-compile
./run_repro.sh vmi 0
```

## Recorded result

AssertionError: VMI per_token FP4 requires packed UE8M0 full-cast

See `recorded.log`.
