# Quant leftover PTOAS issues (0909 pins)

PTOAS `b465f26b`, CANN 9.2.0, TileLang `5038c468`, A5.

These reports are **low-level PTOAS bugs**, not one directory per Nightly TileLang leftover class. Each issue has:

- a short **desired VMI** (logical `V<L×T>`, no PART/ONEPT/PK impersonation)
- a working **ASC reference** (Nightly C++ that launches, plus `asc_pattern.mi` — the `pto.mi` PTOAS must emit)
- a table of high-level configs that hit the same bug

Historical leftover-class snapshots (recorded logs, old dumps) live under [`archive/`](archive/). They are not issues.

Host-pad and host TMA permute are not ports. ASC-illegal / untested guards (per-channel FP4, Ascend `npt≠32`, packed+`round_sf=False`, output TMA) are **not** issues.

Design: [PTO-vmi-design.en.md](../../../kernel_study/PTO-Gym-vmi-design/docs/PTO-vmi-design.en.md), [PTO-vmi-Instruction-SPEC.md](../../../kernel_study/PTO-Gym/docs/PTO-vmi-Instruction-SPEC.md).

## Six issues

| | Dir | Low-level hole | ASC reference | Recorded |
|---|---|---|---|---|
| **A** | [`vmi_1xT_ue8m0_scale_apply/`](vmi_1xT_ue8m0_scale_apply/) | 1×T UE8M0 extract + scale apply | Nightly `cast_back_asc` launches; `asc_pattern.mi` is `vlds_brc_elem` + `vshl` + `vmul` | 507035 **or** compile-OK mismatch |
| **B** | [`vmi_compact_v128_residual/`](vmi_compact_v128_residual/) | Compact `V<128×T>` strip (`create_mask(128)`) | Nightly per_block H=384 launches (128-aligned `block_k`, no host pad) | `VMI-RESIDUAL-OP` / `hidden % 256` assert |
| **C** | [`vmi_fp8_vbrc_clear_tail/`](vmi_fp8_vbrc_clear_tail/) | `vbrc` / UB clear of fp8 (ceildiv tail) | Nightly fused rescale launches (`ceildiv`, clamped DMA) | `vbrc(f8e4m3(0))` / `T.clear` e4m3 |
| **D** | [`vmi_unpacked_float_sf_move/`](vmi_unpacked_float_sf_move/) | Unpacked `V<L×f32>` SF load/store (row and TMA-col) | Nightly unpacked path (`ONEPT_B32` / `BRC_B32`); pattern in `asc_pattern.mi` | packed-only adapter assert |
| **E** | [`vmi_ue8m0_pack_from_fp32_amax/`](vmi_ue8m0_pack_from_fp32_amax/) | amax → UE8M0 pack (sf_only or with payload) | Nightly `store_scale_pair` packed; bf16 sibling already bitwise | 1024-byte SF mismatch (fp32) / assert (per_token) |
| **F** | [`vmi_persistent_last_wave/`](vmi_persistent_last_wave/) | Last Persistent software-pipeline wave | Nightly fused per_channel TMA-in launches | remainder-wave payload+SF mismatch |

Joins checked in TileKernels-vmi:

- `per_token_rescale_row_sf` compose calls `cast_back` → **A**.
- per_token fp32 packed TMA M=8001 H=16384/65536 isolated 507035 (fresh lock) → **A** (same ACL family, large TMA store).
- per_channel TMA-in **compose** → **A**; **fused** remainder waves stay **F** (matching waves are bitwise).

## Leftover-class → issue (100% four-kernel ASC-parity blockers)

Every in-matrix `TODO(impl)` row on `per_token_cast` / `per_block_cast` / `per_channel_cast` / `cast_back` maps here. Counts from TileKernels-vmi `kernel_coverage.md` on `per_channel_debug_0909` (459 ASC configs, 135 TODO(impl)).

| Leftover class (rows) | Issue |
|---|---|
| per_token hidden 128/384 (40) | B |
| per_token fused rescale M=8001 (12) | C |
| per_token sf_only+packed (2) | E |
| per_token TMA-col unpacked SF (1) | D |
| per_token FP4 unpacked SF (1) | D |
| per_token row-major rescale compose (1) | A |
| per_token fp32 packed TMA M=8001 H=16384/65536 isolated 507035 (2) | A |
| per_channel large TMA-in fused (7) | F |
| per_channel large TMA-in compose | A |
| per_channel unpacked in-SF (2) | D |
| per_block H=384 (8) | B |
| per_block sf_only+packed fp32 (4) | E |
| cast_back npt=1 mismatches (6) + ACL 507035 (48) + e2m1→fp32 TMA launch (1) | A |
| ASC-illegal / untested guards | not an issue |

## What is not a PTOAS issue

- Host-pad of M or H (ASC uses in-kernel `ceildiv` / 128-aligned tiles).
- Host permute of row-major SF to look like TMA-col.
- ASC-illegal combinations (per-channel FP4, Ascend `npt≠32`, packed without round, per-channel **output** TMA).
- SwiGLU / top-k / fused cast+cast-back (outside the four-kernel parity goal).
