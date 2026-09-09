# per_token rescale non-a5 layout (row-major / compose)

ASC Nightly L1 supports `Nightly L1 e4m3/e2m1 rescale with row-major SF (not TMA-(32,32) packed)`.

## Bug pattern

Fused 1xT rescale for row-major packed in / (1,32) packed out, or a compose that matches ASC fused SF bitwise.

`tilelang_dump.ptodsl.py`: Working sibling is a5 TMA-(32,32) packed fused rescale (aligned M).

## Reproducer

```bash
./run_repro.sh vmi-compile
./run_repro.sh vmi 0
```

## Recorded result

Fused path rejects; compose (cast_back→per_token) is SF off-by-one vs ASC fused

See `recorded.log`.
