# F — Persistent last-wave remainder (per_channel TMA-in fused)

PTOAS `b465f26b` / CANN 9.2.0 / TileLang `5038c468` / A5.

Fused e4m3 rescale with TMA-col **input** SF matches ASC on full waves and mismatches on the last / periodic software-pipeline wave. Isolated legal reroutes (stages=1, even-split) made it worse. This is not a host permute.

**Compose** of the same configs calls `cast_back` TMA npt=1 and belongs to issue A.

Confirmed distinct from A: waves that match are bitwise on payload+SF; the fail is confined to remainder groups (512×7168 m-tile 7 × k-tiles 20–27; 8064×2048 period 27).

## VMI design

Same 1×T extract as issue A ([design §1.1](../../../../kernel_study/PTO-Gym-vmi-design/docs/PTO-vmi-design.en.md)). The extra contract: the **last incomplete Persistent group** must run the same VF. `desired_vmi.ptodsl.py` is one remainder wave, not the full 7k-hidden kernel.

## ASC reference (working)

`asc_reference.cpp` is Nightly / TileLang ascend fused per_channel TMA-in. `run_asc.py` launches it (`ASC_LAUNCH_OK`).

`asc_pattern.mi`: `vlds_unpack4` + `vgather2` of TMA-col UE8M0 + `vshl 7` + `vmul`. ASC applies that VF to every wave, including the last.

## High-level configs that hit this

| Config | Recorded | Archive |
|---|---|---|
| per_channel e4m3 rescale TMA-col in-SF, large tiles (7 rows) **fused** | 512×7168 payload=47725 sf=1536 (remainder wave) | [`archive/per_channel_tma_in_large_shape/`](../archive/per_channel_tma_in_large_shape/) (fused half) |
| same configs **compose** | issue A | — |

## Reproducer

```bash
./run_repro.sh vmi-compile
./run_repro.sh vmi 0
./run_repro.sh asc 0
```

See `recorded.log`. Full-kernel dump appendix: `tilelang_dump.ptodsl.py`.
