# WITHDRAWN C — fp8 `vbrc` / UB clear for a ceildiv tail

**Withdrawn** on TileKernels-vmi `quant_missing_impl_0913`. Production
kernel zeros leftover fp8/fp4 lanes with i8 `vbrc(0)` + `vinterpret`
(no host-pad, no fp8 `vbrc` / `T.clear`). Fused rescale M=8001 is
bitwise at H=3072 and H=16384 (8/12 rows). H=65536×4 never ran VMI:
Nightly ASC host `aten::bitwise_left_shift` AICPU **507018**.

This directory is a snapshot. Not a live PTOAS issue.

---

# C — fp8 `vbrc` / UB clear for a ceildiv tail

PTOAS `b465f26b` (recorded) / still open on `9edc5a0`, TileLang `c1a4276c`, CANN 9.2.0, A5. No kernel workaround after PRs 89–91.

ASC fused per-token rescale at M=8001 (`8001 % 16 == 1`) uses in-kernel `ceildiv` and zeros leftover e4m3/e2m1 lanes. VMI `vbrc(f8e4m3(0))` raises `TypeError` (`f8E4M3FN` constructor). `T.clear` on e4m3 UB fails (`Bad bit-width float8_e4m3fn`). Host-pad M to a multiple of 16 is not a port.

## VMI design

[PTO-vmi-design.en.md](../../../../kernel_study/PTO-Gym-vmi-design/docs/PTO-vmi-design.en.md) `vbrc` tables list i8–i32 / f16 / bf16 / f32. **fp8_e4m3 / fp8_e2m1 is the hole.** Spec `V<64×fp8>` is a legal partial vreg.

`desired_vmi.ptodsl.py`: `vbrc(pto.f8e4m3(0), size=64)` + `vstore`. Same as i8 zero-fill.

## ASC reference (working)

`asc_reference.cpp` is Nightly `per_token_cast_asc` fused rescale with ceildiv / clamped DMA. `run_asc.py` launches it (`ASC_LAUNCH_OK`).

`asc_pattern.mi`: `vdup` 0.0 on the dequant (bf16) side and clamp `nburst = min(tile, remain)`. Native fp8 store of the leftover tile is `vdup`/`vsts` on the fp8 carrier — that is the missing VMI lowering.

## High-level configs that hit this

| Config | Recorded | Archive |
|---|---|---|
| per_token fused rescale M=8001, 4 dtype pairs × 3 aligned H (12 rows) | compile fail `vbrc`/`T.clear` e4m3 | [`archive/per_token_fp4_rescale_m8001/`](../archive/per_token_fp4_rescale_m8001/) |

The two **fp32 packed TMA cast** 507035 rows at M=8001 H=16384/65536 were later kernel-fixed (E4M3 exact-M / on-device TMA, bitwise ≥0.98). They are not this tail-zero hole and not a live PTOAS issue.

## Reproducer

```bash
./run_repro.sh vmi-compile
./run_repro.sh vmi 0
./run_repro.sh asc 0
```

See `recorded.log`. Aligned-M dump appendix: `tilelang_dump.ptodsl.py`.
