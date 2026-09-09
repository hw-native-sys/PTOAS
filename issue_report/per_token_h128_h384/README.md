# per_token H=128/384 (ASC hidden % 128; VMI VL=256)

ASC Nightly L1 supports `Nightly L1 hidden 128 and 384 on every bf16/fp32 cast layout`.

## Bug pattern

Legal 1xT 128-lane strip (create_mask in {1,64,128,256}) so Persistent tiles cover hidden=128 (one tile) and hidden=384 (three tiles), matching ASC hidden % 128. Host-pad is not a port.

`tilelang_dump.ptodsl.py`: Working sibling is H=3072 VL=256 packed TMA (ported). H=128 never reaches TileLang dump.

## Reproducer

```bash
./run_repro.sh vmi-compile
./run_repro.sh vmi 0
```

## Recorded result

AssertionError: VMI per-token cast needs hidden % 256 == 0

See `recorded.log`.
