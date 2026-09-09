# per_token TMA-col unpacked SF

ASC Nightly L1 supports `Nightly L1 use_tma=True use_packed=False (round=True)`.

## Bug pattern

On-device TMA-col store of unpacked float SF. Host permute of row-major SF is not a port.

`tilelang_dump.ptodsl.py`: Working sibling is TMA-col packed+round on-device path.

## Reproducer

```bash
./run_repro.sh vmi-compile
./run_repro.sh vmi 0
```

## Recorded result

AssertionError at get_per_token_cast_kernel_vmi (packed required for TMA adapter)

See `recorded.log`.
