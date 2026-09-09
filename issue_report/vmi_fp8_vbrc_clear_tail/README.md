# C — fp8 `vbrc` / UB clear for a ceildiv tail

PTOAS `b465f26b` / CANN 9.2.0 / TileLang `5038c468` / A5.

ASC fused per-token rescale at M=8001 (`8001 % 16 == 1`) uses in-kernel `ceildiv` and zeros leftover e4m3/e2m1 lanes. VMI `vbrc(f8e4m3(0))` raises `TypeError` (`f8E4M3FN` constructor). `T.clear` on e4m3 UB fails (`Bad bit-width float8_e4m3fn`). Host-pad M to a multiple of 16 is not a port.

## VMI design

[PTO-vmi-design.en.md](../../../../kernel_study/PTO-Gym-vmi-design/docs/PTO-vmi-design.en.md) `vbrc` tables list i8–i32 / f16 / bf16 / f32. **fp8_e4m3 / fp8_e2m1 is the hole.** Spec `V<64×fp8>` is a legal partial vreg.

`desired_vmi.ptodsl.py`: `vbrc(pto.f8e4m3(0), size=64)` + `vstore`. Same as i8 zero-fill.

## ASC reference (working)

`asc_reference.cpp` is Nightly `per_token_cast_asc` fused rescale with ceildiv / clamped DMA. `run_asc.py` launches it (`ASC_LAUNCH_OK`).

`asc_pattern.mi`: `vdup` 0.0 on the dequant (bf16) side and clamp `nburst = min(tile, remain)`. Native fp8 store of the leftover tile is `vdup`/`vsts` on the fp8 carrier — that is the missing VMI lowering.

## High-level configs that hit this

| Config | Recorded | Old dir |
|---|---|---|
| per_token fused rescale M=8001, 4 dtype pairs × 3 aligned H (12 rows) | compile fail `vbrc`/`T.clear` e4m3 | `per_token_fp4_rescale_m8001/` |

The two **fp32 packed TMA cast** 507035 rows at M=8001 H=16384/65536 are issue A (same ACL family, large TMA store), not this tail-zero hole.

## Reproducer

```bash
./run_repro.sh vmi-compile
./run_repro.sh vmi 0
./run_repro.sh asc 0
```

See `recorded.log`. Aligned-M dump appendix: `tilelang_dump.ptodsl.py`.
