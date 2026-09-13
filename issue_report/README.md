# Quant leftover PTOAS issues (0909 pins)

Live reports are **low-level PTOAS holes that still block a Nightly VMI
port**, with no known kernel workaround. Each live issue has:

- a short **desired VMI** (logical `V<L×T>`, no PART/ONEPT/PK impersonation)
- a working **ASC reference** (`asc_pattern.mi` — the `pto.mi` PTOAS must emit)
- a table of high-level configs that still hit the same hole

Recorded originally on PTOAS `b465f26b` / TileLang `5038c468`. **Still
open** on PTOAS `9edc5a0` / TileLang `c1a4276c` / Nightly `adefcb7` /
CANN **9.2.0** after TileKernels-vmi `#88` + sequential remasure of
PRs **89, 91, 90** (`pr_89_90_91_rebase`). Coverage grain: four-kernel
**623** ASC configs → **324** ready / **273** TODO(impl) / **26** TODO(perf).

Historical leftover-class snapshots and **withdrawn** compiler reports
live under [`archive/`](archive/). They are not live issues.

Host-pad and host TMA permute are not ports. ASC-illegal / untested
guards (per-channel FP4, Ascend `npt≠32`, packed+`round_sf=False`,
output TMA) are **not** issues.

Design: [PTO-vmi-design.en.md](../../../kernel_study/PTO-Gym-vmi-design/docs/PTO-vmi-design.en.md),
[PTO-vmi-Instruction-SPEC.md](../../../kernel_study/PTO-Gym/docs/PTO-vmi-Instruction-SPEC.md).

## Live issues (no known kernel workaround)

| | Dir | Low-level hole | Still hits | Recorded |
|---|---|---|---|---|
| **B** | [`vmi_compact_v128_residual/`](vmi_compact_v128_residual/) | Compact `V<128×T>` strip (`create_mask(128)`) | per_token H=128/384 (40); per_block H=384 (8) | `VMI-RESIDUAL-OP` / `hidden % 256` assert |
| **C** | [`vmi_fp8_vbrc_clear_tail/`](vmi_fp8_vbrc_clear_tail/) | `vbrc` / UB clear of fp8 (ceildiv tail) | per_token fused rescale M=8001 (12) | `vbrc(f8e4m3(0))` / `T.clear` e4m3 |
| **D** | [`vmi_unpacked_float_sf_move/`](vmi_unpacked_float_sf_move/) | Unpacked `V<L×f32>` SF load/store (row and TMA-col) | per_token TMA-unpacked (1), FP4 unpacked (1); per_channel unpacked in-SF (2) | packed-only adapter assert |
| **E** | [`vmi_ue8m0_pack_from_fp32_amax/`](vmi_ue8m0_pack_from_fp32_amax/) | amax → UE8M0 pack from an **fp32** amax | per_block sf_only+packed fp32 (4); per_token sf_only+packed (2) | 1024-byte SF mismatch / assert |

Letters A and F are **withdrawn** (kernel workarounds exist). Snapshots:
[`archive/vmi_1xT_ue8m0_scale_apply/`](archive/vmi_1xT_ue8m0_scale_apply/),
[`archive/vmi_persistent_last_wave/`](archive/vmi_persistent_last_wave/).

## Leftover-class → live issue

Every four-kernel `TODO(impl)` class that is still a **PTOAS** blocker
maps here. Kernel-fixed / unisolated / TODO(perf) classes are **not**
PTOAS issues (see below).

| Leftover class (rows on `pr_89_90_91_rebase`) | Issue |
|---|---|
| per_token hidden 128/384 (40) | B |
| per_token fused rescale M=8001 (12) | C |
| per_token sf_only+packed (2) | E |
| per_token TMA-col unpacked SF (1) | D |
| per_token FP4 unpacked SF (1) | D |
| per_channel unpacked in-SF (2) | D |
| per_block H=384 (8) | B |
| per_block sf_only+packed fp32 (4) | E |
| ASC-illegal / untested guards | not an issue |

## What is not a PTOAS issue

- Host-pad of M or H; host permute of row-major SF to TMA-col.
- ASC-illegal combinations (per-channel FP4, Ascend `npt≠32`, packed
  without round, per-channel **output** TMA).
- SwiGLU / top-k / fused `#169` `VMI-RESIDUAL-OP` (outside this
  four-kernel PTOAS set).
- **Withdrawn A:** 1×T UE8M0 extract. Production kernels no longer
  need it: PR90 `(32,1)` `dintlv` + decode-once / TMA occupancy; PR91
  TMA-col `vgather` + in-register `<<7`. Representative
  `cast_back` e4m3→fp32 TMA npt=32 512×128 was ACL 507035; isolated
  remasure is now bitwise **1.139**. Canonical `(1,32)` leftover is
  kernel TODO(perf) (4-lane decode), not a compiler hole. Snapshot:
  [`archive/vmi_1xT_ue8m0_scale_apply/`](archive/vmi_1xT_ue8m0_scale_apply/).
- **Withdrawn F:** Persistent last-wave fused mismatch. The 7
  per_channel TMA-col large-shape payload mismatches are bitwise after
  PR91 gather (1 ready, 6 TODO(perf)). Production evidence for a
  distinct last-wave PTOAS bug is gone. Snapshot:
  [`archive/vmi_persistent_last_wave/`](archive/vmi_persistent_last_wave/).
- PR89 bf16 TMA+packed `sf_only` 512×2048 (0.53 → ≥1.23): kernel
  `token_group=4` store, not PTOAS. Issue E is **fp32** pack only.
- per_token row-major rescale compose (1): kernel compose leftover
  (SF off-by-one vs ASC fused), not a proven PTOAS hole.
- per_token fp32 packed TMA M=8001 H=16384/65536: later kernel-fixed
  (E4M3 exact-M / on-device TMA), bitwise ≥0.98.
- Unisolated ACL 507035 (`cast_back` 201 rows; one per_channel
  8064×768 TMA-col): counted as TODO(impl) until isolated
  one-process-per-row. Not a PTOAS issue on that evidence.
- TODO(perf) (canonical `(1,32)`, some TMA fp32 / h4096, row
  `sf_only` packed, six bitwise-slow per_channel TMA-col rows):
  kernel schedule, not PTOAS.
