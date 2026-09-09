# E — UE8M0 pack from an fp32 amax

PTOAS `b465f26b` / CANN 9.2.0 / TileLang `5038c468` / A5.

Scale-only (or scale+payload) path: amax → exponent → packed UE8M0 `vstore`. **bf16** per-block sf_only+packed is already bitwise versus ASC (row and TMA). **fp32** of that path mismatches 1024 SF bytes. per-token sf_only+packed never emits IR.

One issue: pack scales from an fp32 (or bf16-widened) amax into UE8M0.

## VMI design

[PTO-vmi-Instruction-SPEC.md](../../../../kernel_study/PTO-Gym/docs/PTO-vmi-Instruction-SPEC.md) `vcmax` + `vcvt` + `vstore`. User writes logical amax and a u8 pack; `pto.as` owns `ONEPT_B16` / `vsts_pack_quarter`.

`desired_vmi.ptodsl.py`: `vcmax` + shift + `vcvt` to u8 + `vstore`. No payload store required.

## ASC reference (working pattern)

Nightly `per_block_cast_asc.store_scale_pair` when `use_packed_ue8m0`:

```
exponents = (bitcast_u32(scale) >> 23)
packed    = lo | (hi << 8)
vsts ONEPT_B16
```

`asc_pattern.mi` is that sequence. The bf16 sf_only+packed port is the working sibling (same pack, different amax dtype). `run_asc.py` points at the pattern; Nightly launches the fp32 sf_only+packed config today.

## High-level configs that hit this

| Config | Recorded | Archive |
|---|---|---|
| per_block sf_only+packed **fp32** (e2m1/e4m3 × row/TMA; 4 rows) | launch OK, 1024-byte SF mismatch | [`archive/per_block_sf_only_packed_fp32/`](../archive/per_block_sf_only_packed_fp32/) |
| per_token sf_only+packed (2 rows) | assert: sf_only requires unpacked | [`archive/per_token_sf_only_packed/`](../archive/per_token_sf_only_packed/) |

If the fp32 mismatch is only a rounding/`vcvt` after IR exists, keep it as a second recorded outcome here — not a seventh issue.

## Reproducer

```bash
./run_repro.sh vmi-compile
./run_repro.sh vmi 0
./run_repro.sh asc 0
```

See `recorded.log` (representative: per_block fp32 1024-byte SF mismatch).
