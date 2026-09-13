# Quant leftover PTOAS issues (0909 pins)

Live reports are **low-level PTOAS holes that still block a Nightly VMI
port**, with no known kernel workaround. Each live issue has:

- a short **desired VMI** (logical `V<L×T>`, no PART/ONEPT/PK impersonation)
- a working **ASC reference** (`asc_pattern.mi` — the `pto.mi` PTOAS must emit)
- a **production** TileLang dump of the failing emit (or frontend error if
  codegen never runs)
- a table / `row_map.txt` of high-level configs that still hit the same hole

Recorded originally on PTOAS `b465f26b` / TileLang `5038c468`. **Still
open** on PTOAS `9edc5a0` / TileLang `c1a4276c` / Nightly `adefcb7` /
CANN **9.2.0** after TileKernels-vmi `quant_missing_impl_0913` (from
`main` `#92` `f8c46b282`). Four-kernel grain on that branch: **623** ASC
configs; leftover **TODO(impl)** that is still a PTOAS blocker is **26**
rows (all `per_token_cast`).

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
| **B** | [`vmi_compact_v128_residual/`](vmi_compact_v128_residual/) | Packed compact leftover strip (`V<128>` design; production dual `V<64>` packed) | per_token packed H=128/384 (24) | `VMI-RESIDUAL-OP` / fused `vand` lane mismatch |
| **D** | [`vmi_unpacked_float_sf_move/`](vmi_unpacked_float_sf_move/) | Unpacked `V<L×f32>` SF / unpacked FP4 payload | per_token TMA-unpacked (1); FP4 unpacked (1) | 48-byte SF mismatch; 32-lane FP4 `vstore` |

Letters **A, C, E, F** are **withdrawn** (kernel workarounds exist).
Snapshots:

- A: [`archive/vmi_1xT_ue8m0_scale_apply/`](archive/vmi_1xT_ue8m0_scale_apply/)
- C: [`archive/vmi_fp8_vbrc_clear_tail/`](archive/vmi_fp8_vbrc_clear_tail/) — i8 `vbrc(0)` + `vinterpret` to fp8/fp4
- E: [`archive/vmi_ue8m0_pack_from_fp32_amax/`](archive/vmi_ue8m0_pack_from_fp32_amax/) — fp32 pack + per_token `sf_only` packed bitwise
- F: [`archive/vmi_persistent_last_wave/`](archive/vmi_persistent_last_wave/)

## Leftover-class → live issue

Every four-kernel `TODO(impl)` class that is still a **PTOAS** blocker
maps here. Kernel-fixed / unisolated / TODO(perf) classes are **not**
PTOAS issues (see below).

| Leftover class (rows on `main@f8c46b282` / `#92`) | After 0913 |
|---|---|
| per_token hidden 128/384 unpacked (~16 of 40) | kernel-closed (dual `V<64>`) |
| per_token hidden 128/384 packed (24 of 40) | **B** |
| per_token fused rescale M=8001 (12) | **withdrawn C** — 8 remasured bitwise; H=65536×4 ASC AICPU 507018 (VMI never ran) |
| per_token sf_only+packed (2) | **withdrawn E** (bitwise, TODO(perf)) |
| per_token TMA-col unpacked SF (1) | **D** |
| per_token FP4 unpacked SF (1) | **D** |
| per_token row-major compose (1) | kernel-closed |
| per_channel unpacked in-SF (2) | **withdrawn D** (bitwise) |
| per_channel ACL 507035 `8064×768` TMA-col (1) | isolated bitwise (not PTOAS) |
| per_block H=384 (8) | **withdrawn B** (bitwise) |
| per_block sf_only+packed fp32 (4) | **withdrawn E** (bitwise, TODO(perf)) |
| cast_back 201 ACL 507035 | isolated 190/190 bitwise (cascade poison, not PTOAS) |
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
  kernel TODO(perf) (4-lane decode), not a compiler hole.
- **Withdrawn C:** M=8001 fp8 tail. Kernel zeros leftover fp8/fp4 lanes
  via i8 `vbrc(0)` + `vinterpret` (no host-pad, no fp8 `vbrc`).
- **Withdrawn E:** fp32 amax → UE8M0 pack, and per_token `sf_only`+packed.
- **Withdrawn F:** Persistent last-wave fused mismatch.
- Isolated ACL 507035 (`cast_back` + per_channel `8064×768`): 13/13
  layout isolates bitwise; full remasure 190/190 OK. Not a PTOAS issue.
- TODO(perf) on newly bitwise rows (some compact unpacked `sf_only`,
  packed `sf_only`, large TMA): kernel schedule, not PTOAS.
- No `vmi_tma_cast_back_launch_507035/` dir: 507035 was cascade poison.
