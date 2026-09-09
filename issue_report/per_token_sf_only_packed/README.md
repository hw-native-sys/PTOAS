# per_token sf_only+packed

ASC Nightly L1 supports `Nightly L1 per_token_cast_with_sf_only with packed UE8M0 (row and TMA)`.

## Bug pattern

On-device packed UE8M0 SF-only path (same VF amax/UE8M0 as full cast, no payload store). ASC tests this for every packed combo.

`tilelang_dump.ptodsl.py`: Working sibling is unpacked sf_only (ported) and packed full-cast (ported).

## Reproducer

```bash
./run_repro.sh vmi-compile
./run_repro.sh vmi 0
```

## Recorded result

AssertionError: sf_only/cast_only require unpacked SF

See `recorded.log`.
