# A — UE8M0 1×T extract + scale apply

PTOAS `b465f26b` / CANN 9.2.0 / TileLang `5038c468` / A5.

Legal VMI is a compact **1×T** UE8M0 load, a shift into the exponent field, a broadcast, then `vcvt`/`vmul` on the payload. `pto.as` owns `BRC_ELEM` / `UNPK4` / `PART_P0`. Two recorded outcomes of this **same IR family**: ACL 507035, or compile-OK scale/payload mismatch.

This is not five Nightly dtype×npt×H bugs.

## VMI design

[PTO-vmi-design.en.md](../../../../kernel_study/PTO-Gym-vmi-design/docs/PTO-vmi-design.en.md) §1.1 compact vregs and §0.2 Category A/B. [PTO-vmi-Instruction-SPEC.md](../../../../kernel_study/PTO-Gym/docs/PTO-vmi-Instruction-SPEC.md) `vload` slot/group load (`result.L == C` → compact `V<1×T>`).

`desired_vmi.ptodsl.py` is a short VF: `vload(size=1)` + `vshls` + `vbrc` + `vcvt` + `vmul`. No PART/ONEPT/PK impersonation.

## ASC reference (working)

`asc_reference.cpp` is Nightly `cast_back_asc` for e4m3→fp32 TMA npt=32 H=128. `run_asc.py` launches it (`ASC_LAUNCH_OK`).

`asc_pattern.mi` is the physical sequence PTOAS must emit: `vlds_brc_elem` → `vshl`/`vand` → `vlds_unpack4` → `vcvt` → `vmul` → `vsts_norm`.

## High-level configs that hit this (not separate issues)

| Config | Recorded | Archive |
|---|---|---|
| cast_back e4m3→fp32 TMA npt=32 H=128 | compile-OK, ACL 507035 (PR78 remasure still 507035) | [`archive/cast_back_e4m3_fp32_tma_npt32_h128/`](../archive/cast_back_e4m3_fp32_tma_npt32_h128/) |
| cast_back e4m3→fp32 TMA npt=1 | compile-OK mismatch 141800 / 5.25 (was 507035) | [`archive/cast_back_e4m3_fp32_tma_npt1/`](../archive/cast_back_e4m3_fp32_tma_npt1/) |
| cast_back TMA npt=1 H=2048 e2m1/e4m3→bf16 | isolated mismatch 113856 / 125926 | [`archive/cast_back_tma_npt1_h2048/`](../archive/cast_back_tma_npt1_h2048/) |
| cast_back e2m1→fp32 TMA npt=32 H=2048 | isolated mismatch 79781 / 3.0 | [`archive/cast_back_e2m1_fp32_tma_npt32_h2048/`](../archive/cast_back_e2m1_fp32_tma_npt32_h2048/) |
| per_channel TMA-in **compose** (cast_back TMA npt=1) | compose payload mismatch | [`archive/per_channel_tma_in_large_shape/`](../archive/per_channel_tma_in_large_shape/) (compose half) |
| per_token row-major rescale | compose calls `cast_back` → SF off-by-one vs ASC fused | [`archive/per_token_rescale_row_sf/`](../archive/per_token_rescale_row_sf/) |
| per_token fp32 packed TMA cast M=8001 H=16384/65536 | isolated ACL 507035 (fresh lock; same family, large TMA store) | no leftover dir — size manifestation |

cast_back **row** npt=1 512×2048 is kernel-fixed by TileKernels-vmi PR78 and is not listed here. Snapshot: [`archive/cast_back_row_npt1/`](../archive/cast_back_row_npt1/).

Maps to [cann/pto-as issue #7](https://gitcode.com/cann/pto-as/issues/7). Fused per_channel last-wave mismatch is issue F, not this file.

## Reproducer

```bash
./run_repro.sh vmi-compile
./run_repro.sh vmi 0
./run_repro.sh asc 0
```

See `recorded.log` (representative: e4m3→fp32 TMA npt=32 H=128, 507035). Production dump appendix: `tilelang_dump.ptodsl.py`.
