# PTODSL Element-wise 1D/2D Template Design

## Status

This document is the implementation inventory and design baseline for adding
deterministic 1D/2D template selection to the A5 PTODSL TileLib element-wise
operations.

The inventory was taken on 2026-07-30 from:

- PTOAS commit `456ebb4b6478382650d66b3ce100671b698a1e34`.
- `cann/pto-isa` commit
  `57bd62714023b18d082456da2014398957e00a81`.

The pinned A5 PTO-ISA revision is the primary reference for instruction and
ordinary 1D/2D traversal legality. The PTOAS implementation remains the source
of truth for PTODSL integration, registered operand forms, compiler metadata,
and existing generated-helper behavior.

The shared ordinary-element-wise legality predicate and complete tile-config
metadata plumbing are now available. Ranked candidate order is also preserved
from Python through the compact candidate attribute and `ExpandTileOp`.

`_elementwise.py` now provides reusable flattened 1D and row-wise 2D traversal
forms for ordinary unary, Tile-Tile, Tile-Scalar, and scalar-fill families.
Its registration helpers accept an explicit traversal form, derive
`loop_depth`, and attach `require_elementwise_1d(...)` to 1D candidates. All
runtime loops use the source-backed Python `for ... in range(...)` syntax and
are lowered by PTODSL's control-flow AST rewrite.

All unary operations now support deterministic 1D/2D selection. The ordinary
operations `tabs`, `texp`, `tneg`, `tnot`, `trelu`, `trsqrt`, and `tsqrt`
register preferred shared 1D candidates while preserving their original shared
2D candidates as fallbacks. `tlog` and `trecip` keep their algorithms local to
their operation modules and use only the shared unary traversal emitters. This
raises the current unary candidate count to 20 and the current scoped candidate
count to 92.

The ordinary Tile-Tile operations `tadd`, `tand`, `tmax`, `tmin`, `tmul`,
`tor`, `tshl`, `tshr`, and `tsub` also register preferred shared 1D candidates
and preserve their ID-0 2D fallbacks. `tmin` now uses the same shared binary
registrar instead of duplicating its row-wise loop. This raises the current
Tile-Tile candidate count to 23 and the current scoped candidate count to 101.

The non-temporary specialized Tile-Tile operations are also migrated. `tdiv`
keeps its default and IEEE high-precision algorithms in `tdiv.py`, while
`tfmod` keeps its dtype-sensitive computation in the remainder family module;
both use shared binary traversal emitters. This raises the current Tile-Tile
candidate count to 25 and the current scoped candidate count to 103.

At that milestone, temporary-operand Tile-Tile operations remained a
subsequent step.

The remaining temporary-operand Tile-Tile operations are now migrated:
`tprelu`, `trem`, and `txor` preserve their existing ID-0 2D candidates and
register preferred ID-1 1D candidates. All four tile operands (`src0`, `src1`,
`tmp`, and `dst`) participate in flattened-traversal legality, including the
ABI temporary when the current generated body does not access it. This raises
the Tile-Tile candidate count to 28 and the current scoped candidate count to
106, completing 1D/2D coverage for all 14 Tile-Tile operations.

The ordinary Tile-Scalar operations `tadds`, `tands`, `tmaxs`, `tmins`,
`tmuls`, `tors`, `tshls`, `tshrs`, and `tsubs` now also preserve their ID-0
2D candidates and register preferred ID-1 1D candidates. Scalar operands do
not participate in memory-contiguity checks; the shared rule is applied to
`src` and `dst`. The shift operations retain their `i16` scalar signatures,
and `tsubs` now uses the shared registrar while preserving its `vbr` plus
`vsub` computation. This raises the Tile-Scalar candidate count to 24 and the
current scoped candidate count to 115. At that milestone, specialized
Tile-Scalar operations remained a subsequent step.

The specialized Tile-Scalar operations are now migrated as well. `tdivs`
keeps both operand orders and its precision-dependent algorithm local, with
preferred 1D IDs 2 and 3 paired with existing 2D IDs 0 and 1. `tfmods` and
`trems` use the traversal-aware scalar remainder registrar, while `tlrelu`
keeps slope coercion local and `txors` retains scalar broadcasting. The
temporary tiles of `trems` and `txors` participate in 1D legality. This raises
the Tile-Scalar candidate count to 30 and the current scoped candidate count
to 121, completing 1D/2D coverage for all 14 Tile-Scalar operations. Compare,
select, conversion, scalar fill, functional execution tests, and performance
measurements remain subsequent steps.

Scalar fill is now migrated. `texpands` preserves its ID-0 2D fallback and
registers a preferred ID-1 1D candidate for all six existing dtype
signatures. Only the destination tile participates in flattened-traversal
legality. This raises the current scoped candidate count to 122. Predicate
compare/select, conversion, functional execution tests, and performance
measurements remain subsequent steps.

Predicate compare is now migrated. `tcmp` and `tcmps` preserve their ID-0
row-wise fallbacks and register preferred ID-1 flattened candidates. Their
shared `require_predicate_compare_1d(...)` rule models one predicate bit per
source element together with the complete 16-byte PK or 32-byte NORM stores
used by A5. A single logical row may flatten when its physical predicate row
has enough capacity. Multiple rows additionally require the source row width
to end on a predicate-store boundary and the destination row stride to equal
the exact packed bytes produced per row. This raises the current scoped
candidate count to 124. Select, conversion, functional execution tests, and
performance measurements remain subsequent steps.

The A5 PTOAS verifier requires `tcmp` source and destination physical shapes
to match. Since an ordinary predicate destination consequently has a wider
row stride than its packed result, practical multi-row `tcmp` cases retain the
2D candidate; eligible single-row cases select 1D. `tcmps` permits a dense
packed destination and therefore selects 1D for block-aligned multi-row cases.
The previous f32/i32 `tcmps` candidate flattened unconditionally despite its
2D metadata. It is now a genuine row-wise fallback, preventing partial rows or
predicate row padding from being crossed speculatively.

Predicate select is now migrated. `tsel` and `tsels` preserve their ID-0
row-wise fallbacks and register preferred ID-1 flattened candidates. Their
shared `require_predicate_select_1d(...)` rule applies the compare packing
units in reverse: ordinary data tiles must describe one contiguous logical
range, while the mask is checked as a byte-addressed packed predicate. Mask
row capacity accounts for the nominal i8/i16/i32 container dtype used by
`tsels`. Multi-row flattening requires complete predicate blocks and an exact
packed mask row stride. This raises the current scoped candidate count to 126.
Conversion, functional execution tests, and performance measurements remain
subsequent steps.

The A5 implementations do not access the `tsel`/`tsels` temporary tile. The
shared rule nevertheless includes it in legality by requiring complete,
supported local tile metadata. Its shape is not required to match the data
range because doing so would reject the small ABI-compatible temporary tiles
used by existing callers. The f32 row-wise fallback also now derives paired
iterations from the rounded vector-repeat count, matching A5 for valid widths
between 65 and 127 instead of treating that entire range as one 64-lane tail.

## Goals

- Account for every A5 PTODSL element-wise TileOp in scope.
- Record existing callable forms, dtypes, attributes, temporary operands, and
  traversal forms before refactoring them.
- Separate straightforward flattening from predicate- and conversion-specific
  legality.
- Define a candidate identity scheme that does not make candidate IDs carry
  ranking semantics.
- Expose the metadata and specialization gaps that must be closed before a 1D
  candidate can be selected safely.

## Non-Goals

- This work does not include reductions, row/column expansion, partial updates,
  load/store, data movement, sorting, random-number generation, or matrix
  operations.
- This baseline does not decide that any unresolved operation is a permanent
  1D exception.
- Existing precision-mode coverage gaps are recorded but are not automatically
  expanded by the traversal refactor.

## Inventory Summary

At the inventory baseline, the scope contained 43 TileOps and 82 registered
PTODSL candidates:

| Family | TileOps | Registered candidates |
|---|---:|---:|
| Unary | 9 | 10 |
| Tile-Tile | 14 | 14 |
| Tile-Scalar | 14 | 15 |
| Compare, select, conversion, scalar fill | 6 | 43 |
| Total | 43 | 82 |

At that baseline, all 82 candidates declared `loop_depth=2`. All nine unary
operations and all 14 Tile-Tile operations have since gained explicit
`loop_depth=1` candidates. The important baseline exception is `tcmps`: its
f32/i32 branch already traverses `valid_rows * valid_cols` as a flat range even
though the containing candidate is marked as two-dimensional. That mixed
candidate must be separated into explicit 1D and 2D forms.

Each operation is loaded from
`ptodsl/ptodsl/tilelib/templates/a5/<op>.py`, except that `texpands` is
registered by `texpand.py`. Generated registrations delegate to
`_elementwise.py` or `_remainder.py`; bespoke registrations and bodies remain
in their operation modules. The tables below name every registered candidate,
so the module and callable contract can be recovered without depending on
Python import order.

The following abbreviations are used in the inventory:

| Abbreviation | Dtypes |
|---|---|
| `F2` | `f16`, `f32` |
| `I6` | `i8`, `i16`, `i32`, `ui8`, `ui16`, `ui32` |
| `N9` | `I6`, `f16`, `bf16`, `f32` |
| `NEG6` | `i8`, `i16`, `i32`, `f16`, `bf16`, `f32` |
| `RELU3` | `i32`, `f16`, `f32` |
| `FILL6` | `i8`, `i16`, `i32`, `f16`, `bf16`, `f32` |

The provisional 1D classifications mean:

- **Shared**: the operation can use the common same-element-range legality rule
  and a standard flat vector traversal.
- **Algorithm-specific**: flattening appears possible, but the operation has
  computation, operand, or mode details that require a dedicated body or an
  extension to the shared rule.
- **Predicate-specific**: legality depends on the packed predicate
  representation as well as the data tiles.
- **Conversion-specific**: legality depends on source/destination widths,
  distribution modes, packing, or a multi-step conversion.
- **Exception**: 1D has been proven illegal and the reviewed reason must be
  documented and tested. No operation is classified as a confirmed exception
  at this inventory stage.

## Unary Inventory

| Op | Existing candidate(s) and operands | Dtypes | Current form | Provisional 1D classification |
|---|---|---|---|---|
| `tabs` | `template_tabs(src, dst)`; `template_tabs_1d(src, dst)` | same `F2` | shared 2D fallback and preferred shared 1D | Shared |
| `texp` | `template_texp(src, dst)`; `template_texp_1d(src, dst)` | same `F2` | shared 2D fallback and preferred shared 1D, default precision only | Shared |
| `tlog` | default `template_tlog`/`template_tlog_1d`; high-precision `template_tlog_high_precision`/`template_tlog_high_precision_1d` | same `F2` | precision-selected 2D fallbacks and preferred 1D candidates | Algorithm-specific |
| `tneg` | `template_tneg(src, dst)`; `template_tneg_1d(src, dst)` | same `NEG6` | shared 2D fallback and preferred shared 1D | Shared |
| `tnot` | `template_tnot(src, dst)`; `template_tnot_1d(src, dst)` | same `I6` | shared 2D fallback and preferred shared 1D | Shared |
| `trecip` | `template_trecip(src, dst)`; `template_trecip_1d(src, dst)` | same `F2` | operation-local `1 / src` computation with shared 2D/1D traversal | Algorithm-specific |
| `trelu` | `template_trelu(src, dst)`; `template_trelu_1d(src, dst)` | same `RELU3` | shared 2D fallback and preferred shared 1D | Shared |
| `trsqrt` | `template_trsqrt(src, dst)`; `template_trsqrt_1d(src, dst)` | same `F2` | shared 2D fallback and preferred shared 1D, default precision only | Shared |
| `tsqrt` | `template_tsqrt(src, dst)`; `template_tsqrt_1d(src, dst)` | same `F2` | shared 2D fallback and preferred shared 1D, default precision only | Shared |

`precisionType` is forwarded for `texp`, `tlog`, `trecip`, `trsqrt`, and
`tsqrt`. In the current PTODSL templates only `tlog` uses it for candidate
selection; the other listed templates explicitly implement default precision
only. The 1D/2D work must preserve that current coverage unless precision
parity is approved as separate work.

## Tile-Tile Inventory

| Op | Existing candidate and operands | Dtypes | Current form | Provisional 1D classification |
|---|---|---|---|---|
| `tadd` | `template_tadd`/`template_tadd_1d(src0, src1, dst)` | same `N9` | shared 2D fallback and preferred shared 1D | Shared |
| `tand` | `template_tand`/`template_tand_1d(src0, src1, dst)` | same `I6` | shared 2D fallback and preferred shared 1D | Shared |
| `tdiv` | `template_tdiv`/`template_tdiv_1d(src0, src1, dst)` | same `F2` | operation-local precision-aware computation with shared 2D/1D traversal | Algorithm-specific |
| `tfmod` | `template_tfmod`/`template_tfmod_1d(src0, src1, dst)` | same `f32`, `f16`, `i16`, or `ui16` | remainder-family 2D fallback and preferred 1D | Algorithm-specific |
| `tmax` | `template_tmax`/`template_tmax_1d(src0, src1, dst)` | same `N9` | shared 2D fallback and preferred shared 1D | Shared |
| `tmin` | `template_tmin`/`template_tmin_1d(src0, src1, dst)` | same `N9` | shared 2D fallback and preferred shared 1D | Shared |
| `tmul` | `template_tmul`/`template_tmul_1d(src0, src1, dst)` | same `N9` | shared 2D fallback and preferred shared 1D | Shared |
| `tor` | `template_tor`/`template_tor_1d(src0, src1, dst)` | same `I6` | shared 2D fallback and preferred shared 1D | Shared |
| `tprelu` | `template_tprelu`/`template_tprelu_1d(src0, src1, tmp, dst)` | data/dst `f16` or `f32`; `tmp` is the data dtype or `i8` | shared binary 2D fallback and preferred 1D; temporary included in legality | Algorithm-specific |
| `trem` | `template_trem`/`template_trem_1d(src0, src1, tmp, dst)` | all `f32`, all `f16`, or all `i32` | remainder-family 2D fallback and preferred 1D; temporary included in legality | Algorithm-specific |
| `tshl` | `template_tshl`/`template_tshl_1d(src0, src1, dst)` | same `I6` | shared 2D fallback and preferred shared 1D | Shared |
| `tshr` | `template_tshr`/`template_tshr_1d(src0, src1, dst)` | same `I6` | shared 2D fallback and preferred shared 1D | Shared |
| `tsub` | `template_tsub`/`template_tsub_1d(src0, src1, dst)` | same `N9` | shared 2D fallback and preferred shared 1D | Shared |
| `txor` | `template_txor`/`template_txor_1d(src0, src1, tmp, dst)` | same `I6` across all operands | shared binary 2D fallback and preferred 1D; temporary included in legality | Algorithm-specific |

`precisionType` is forwarded and consumed by `tdiv`. A 1D form must preserve
both its ordinary `vdiv` path and its high-precision helper path.

Every temporary tile remains part of the 1D legality decision even where the
current PTODSL body does not read it directly. Whether a temporary's logical
range must equal the data range or satisfy a representation-specific relation
is an operation-family decision; it must not be omitted from the predicate.

## Tile-Scalar Inventory

| Op | Existing candidate(s) and operands | Dtypes | Current form | Provisional 1D classification |
|---|---|---|---|---|
| `tadds` | `template_tadds`/`template_tadds_1d(src, scalar, dst)` | same `N9` | shared 2D fallback and preferred shared 1D | Shared |
| `tands` | `template_tands`/`template_tands_1d(src, scalar, dst)` | same `I6` | shared 2D fallback and preferred shared 1D | Shared |
| `tdivs` | `template_tdivs_tile_scalar`/`template_tdivs_tile_scalar_1d(src, scalar, dst)`; `template_tdivs_scalar_tile`/`template_tdivs_scalar_tile_1d(scalar, src, dst)` | same `F2` | operand-order-specific precision-aware 2D fallbacks and preferred 1D candidates | Algorithm-specific |
| `tfmods` | `template_tfmods`/`template_tfmods_1d(src, scalar, dst)` | same `f32`, `f16`, `i32`, or `i16` | scalar remainder 2D fallback and preferred 1D | Algorithm-specific |
| `tlrelu` | `template_tlrelu`/`template_tlrelu_1d(src, slope, dst)` | `(f16,f16,f16)`, `(f16,f32,f16)`, `(f32,f32,f32)` | operation-local slope coercion with shared 2D/1D traversal | Algorithm-specific |
| `tmaxs` | `template_tmaxs`/`template_tmaxs_1d(src, scalar, dst)` | same `N9` | shared 2D fallback and preferred shared 1D | Shared |
| `tmins` | `template_tmins`/`template_tmins_1d(src, scalar, dst)` | same `N9` | shared 2D fallback and preferred shared 1D | Shared |
| `tmuls` | `template_tmuls`/`template_tmuls_1d(src, scalar, dst)` | same `N9` | shared 2D fallback and preferred shared 1D | Shared |
| `tors` | `template_tors`/`template_tors_1d(src, scalar, dst)` | same `I6` | shared 2D fallback and preferred shared 1D | Shared |
| `trems` | `template_trems`/`template_trems_1d(src, scalar, tmp, dst)` | all `f32` or all `f16` | scalar remainder 2D fallback and preferred 1D; temporary included in legality | Algorithm-specific |
| `tshls` | `template_tshls`/`template_tshls_1d(src, scalar, dst)` | data/dst `I6`; scalar `i16` | shared 2D fallback and preferred shared 1D | Shared |
| `tshrs` | `template_tshrs`/`template_tshrs_1d(src, scalar, dst)` | data/dst `I6`; scalar `i16` | shared 2D fallback and preferred shared 1D | Shared |
| `tsubs` | `template_tsubs`/`template_tsubs_1d(src, scalar, dst)` | same `N9` | shared broadcast-scalar 2D fallback and preferred 1D | Shared |
| `txors` | `template_txors`/`template_txors_1d(src, scalar, tmp, dst)` | same `I6` across all operands | shared scalar 2D fallback and preferred 1D; temporary included in 1D legality | Algorithm-specific |

`tdivs` is the only scoped Tile-Scalar operation with both normal and reverse
operand forms currently registered. Both forms must receive distinct 1D
candidates and must retain their operand-kind constraints. Adding reverse forms
to other operations is outside this work unless a missing existing PTO callable
form is demonstrated separately.

## Compare, Select, and Scalar-Fill Inventory

| Op | Existing candidate and operands | Dtypes | Current form | Provisional 1D classification |
|---|---|---|---|---|
| `tcmp` | `template_tcmp(src0, src1, dst)` | data pairs `f32`, `i32`, `f16`, `i16`, `i8`, or `ui8`; predicate destination `i8` | dtype-specific packed predicate stores inside 2D row traversal | Predicate-specific |
| `tcmps` | `template_tcmps(src, scalar, dst)` | source/scalar `f32`, `i32`, `f16`, `i16`, `i8`, or `ui8`; predicate destination `ui8` | mixed: f32/i32 flatten, other dtypes traverse rows; metadata says 2D | Predicate-specific |
| `tsel` | `template_tsel(mask, src0, src1, tmp, dst)` | mask `i8`; data/tmp/dst all `f32`, all `f16`, or all `i8` | dtype-specific predicate loads inside 2D row traversal | Predicate-specific |
| `tsels` | `template_tsels(mask, src, tmp, scalar, dst)` | mask `i8`, `i16`, or `i32`; data/tmp/scalar/dst all one of `i8`, `i16`, `i32`, `f32`, or `f16` | dtype-specific predicate loads inside 2D row traversal | Predicate-specific |
| `texpands` | `template_texpands`/`template_texpands_1d(scalar, dst)` | same `FILL6` | shared scalar-fill 2D fallback and preferred 1D | Shared |

The existing shape rules do not yet express predicate representation:

- `tcmp` currently requires the predicate destination and data inputs to have
  the same valid shape even though the body stores packed predicate bytes.
- `tcmps` does not require source and destination valid shapes to match. Its
  f32/i32 branch flattens the source unconditionally, so a multi-row
  partial-column source can cross a physical row gap.
- `tsel` checks no relationship between the mask range and the data range.
- `tsels` checks `src` and `dst` equality but deliberately ignores `mask` and
  `tmp` shapes.

These are inventory findings, not permission to tighten public legality
silently. The predicate-specific design must establish the intended physical
mask contract and add regression coverage before changing these constraints.

`cmp_mode` is forwarded and consumed by `tcmp` and `tcmps`. It changes the
generated helper body and therefore remains part of specialization.

## Conversion Inventory

`tcvt` has 38 existing candidates. Every candidate has operands `(src, dst)`,
priority 0, a unique ID from 0 through 37, and `loop_depth=2`. Except for the
packed BF16-to-FP4 form, the current legality predicate requires equal physical
and valid shapes with row-major/none-box layouts. `round_mode` is forwarded and
used by conversion bodies that request rounding.

| Op | Existing candidates and operands | Dtypes | Current form | Provisional 1D classification |
|---|---|---|---|---|
| `tcvt` | 38 candidates, each `(src, dst)` | 40 registered signatures across 38 candidates, listed below | conversion-specific 2D row traversal | Conversion-specific |

| Existing ID | Candidate | Dtype signature | Implementation category |
|---:|---|---|---|
| 0 | `template_tcvt_f32_to_i32` | `f32 -> i32` | generic conversion |
| 1 | `template_tcvt_i32_to_f32` | `i32 -> f32` | generic conversion |
| 2 | `template_tcvt_i16_to_f16` | `i16 -> f16` | generic conversion |
| 3 | `template_tcvt_f16_to_i16` | `f16 -> i16` | two-step conversion, unpack/pack |
| 4 | `template_tcvt_bf16_to_f16` | `bf16 -> f16` | generic conversion |
| 5 | `template_tcvt_f32_to_f16` | `f32 -> f16` | packed store |
| 6 | `template_tcvt_f32_to_bf16` | `f32 -> bf16` | packed store |
| 7 | `template_tcvt_f16_to_i32` | `f16 -> i32` | unpacked load |
| 8 | `template_tcvt_f16_to_f32` | `f16 -> f32` | unpacked load |
| 9 | `template_tcvt_bf16_to_i32` | `bf16 -> i32` | unpacked load |
| 10 | `template_tcvt_ui8_to_ui16` | `ui8 -> ui16` | unpacked load |
| 11 | `template_tcvt_f32_to_fp8` | `f32 -> f8e4m3` or `f8e5m2` | low-precision select/reorder |
| 12 | `template_tcvt_f32_to_hif8` | `f32 -> hif8` | low-precision select/reorder |
| 13 | `template_tcvt_f16_to_hif8` | `f16 -> hif8` | packed low-precision store |
| 14 | `template_tcvt_bf16_to_fp4` | `bf16 -> f4e1m2x2` or `f4e2m1x2` | packed 2:1 source/destination columns |
| 15 | `template_tcvt_f32_to_i16` | `f32 -> i16` | two-step conversion, packed store |
| 16 | `template_tcvt_f32_to_i64` | `f32 -> i64` | widening store representation |
| 17 | `template_tcvt_f32_to_f32` | `f32 -> f32` | truncation operation |
| 18 | `template_tcvt_f16_to_ui8` | `f16 -> ui8` | packed store |
| 19 | `template_tcvt_f16_to_si8` | `f16 -> si8` | multi-step conversion, packed store |
| 20 | `template_tcvt_bf16_to_f32` | `bf16 -> f32` | unpacked load |
| 21 | `template_tcvt_i16_to_f32` | `i16 -> f32` | unpacked load |
| 22 | `template_tcvt_i16_to_i32` | `i16 -> i32` | unpacked load |
| 23 | `template_tcvt_i16_to_ui32` | `i16 -> ui32` | unpacked load |
| 24 | `template_tcvt_ui8_to_f16` | `ui8 -> f16` | unpacked load |
| 25 | `template_tcvt_si8_to_f16` | `si8 -> f16` | unpacked load |
| 26 | `template_tcvt_si8_to_si16` | `si8 -> si16` | unpacked load, 16-bit store mode |
| 27 | `template_tcvt_i32_to_i64` | `i32 -> i64` | widening store representation |
| 28 | `template_tcvt_i32_to_i16` | `i32 -> i16` | packed store |
| 29 | `template_tcvt_i32_to_ui16` | `i32 -> ui16` | saturated packed store |
| 30 | `template_tcvt_ui32_to_i16` | `ui32 -> i16` | saturated packed store |
| 31 | `template_tcvt_ui32_to_ui16` | `ui32 -> ui16` | saturated packed store |
| 32 | `template_tcvt_si8_to_i32` | `si8 -> i32` | interleave plus two output stores |
| 33 | `template_tcvt_i32_to_ui8` | `i32 -> ui8` | select/reorder plus byte store |
| 34 | `template_tcvt_ui32_to_ui8` | `ui32 -> ui8` | select/reorder plus byte store |
| 35 | `template_tcvt_i16_to_ui8` | `i16 -> ui8` | packed store |
| 36 | `template_tcvt_i64_to_f32` | `i64 -> f32` | packed 64-to-32 store |
| 37 | `template_tcvt_i64_to_i32` | `i64 -> i32` | packed 64-to-32 store |

All 38 `tcvt` candidates now have a paired preferred 1D form. The A5
implementation provides flattened helpers for the generic, unpacked-load,
packed-store, multi-step, widening, 64-bit, and low-precision categories, so
none of the currently registered PTODSL conversions is a 2D-only exception.
BF16-to-FP4 uses a separate rule that proves its 2:1 source/destination column
relationship over both physical and valid ranges.

## Existing Shared Traversal Structure

The ordinary implementations are concentrated in:

- `ptodsl/ptodsl/tilelib/templates/a5/_elementwise.py`
- `ptodsl/ptodsl/tilelib/templates/a5/_remainder.py`
- `ptodsl/ptodsl/tilelib/templates/a5/_common.py`

`_elementwise.py` registers unary, Tile-Tile, Tile-Scalar, and scalar-fill
templates. Unmigrated operation call sites retain the default 2D form, which
restarts `remained = valid_cols` for every row and then walks columns by vector
lane count. Migrated unary and Tile-Tile operations add an explicit 1D
registration next to that unchanged fallback. Specialized algorithms remain
outside `_elementwise.py` and call its lower-level family traversal emitters.
`_remainder.py` owns remainder/fmod computation and exposes traversal-aware
binary registration without moving that computation into the ordinary module.

The reusable ordinary traversal foundation consists of:

- `emit_elementwise_1d(anchor, emit_chunk)`, which computes
  `valid_rows * valid_cols`, carries one remaining-element count, and invokes
  the chunk emitter from one Python range loop;
- `emit_elementwise_2d(anchor, emit_chunk)`, which retains the outer row loop
  and starts a new remaining-column count for each row;
- unary, Tile-Tile, Tile-Scalar, and scalar-fill wrappers for both forms;
- registration helpers whose `traversal="1d"` form derives `loop_depth=1`,
  adds the named-operand 1D legality predicate, and whose `traversal="2d"`
  form derives `loop_depth=2`.

The 1D family wrappers use a base tile pointer plus the flattened element
offset. They do not reconstruct row and column indices. The 2D wrappers retain
row/column addressing so each physical tile's row stride remains respected.
The two module-level traversal cores opt into `rewrite_jit_function`; this is
required because only a registered template body and its lexically nested
helpers are rewritten automatically.
Candidate IDs and priorities are explicit registrar parameters, allowing a 1D
candidate to rank ahead of its stable-ID 2D fallback without relying on
registration order.

The operation-specific files that cannot be migrated by replacing a shared
registrar alone are:

- `tdiv.py`, `tdivs.py`, `tlog.py`, `trecip.py`, `tlrelu.py`, and `tsubs.py`;
- `tcmp.py`, `tcmps.py`, `tsel.py`, and `tsels.py`;
- `tcvt.py`.

Temporary-operand forms in `tprelu`, `trem`, `trems`, `txor`, and `txors` also
require explicit audit even when their bodies come from a shared registrar.

### Existing constraint patterns

The current registrations use several related but non-identical legality
patterns:

- Ordinary `_elementwise.py` forms accept tile operands in `ub` or `vec`,
  require row-major/none-box layouts, and require the named data/temporary
  operands to have equal valid shapes.
- `_remainder.py` forms require `ub`, row-major/none-box tiles and equal valid
  shapes, including the temporary tile when present.
- `tdiv` constrains block layout and memory space through metadata but does not
  currently require equal valid shapes or `none_box` sub-layout.
- `trecip` requires `ub`, row-major/none-box tiles but does not currently
  require equal source/destination valid shapes.
- `txors` checks equal valid shapes only for `src` and `dst`; its `tmp` tile is
  constrained by location/layout but omitted from the shape relation.
- `tcvt` checks physical shape, valid shape, block layout, and sub-layout in its
  custom predicates, but does not declare a common memory-space restriction.

The new shared 1D predicate must compose with each operation's existing legal
domain. It must not accidentally make the 2D fallback narrower while adding a
more conservative 1D candidate.

## Pinned PTO-ISA Reference

At the pinned PTO-ISA revision:

- `include/pto/npu/a5/TUnaryOp.hpp` provides separate
  `TUnaryOps_1D_*` and `TUnaryOps_2D` implementations.
- `include/pto/npu/a5/TBinOp.hpp` provides separate Tile-Tile 1D and 2D
  implementations.
- `include/pto/npu/a5/TBinSOp.hpp` provides separate Tile-Scalar 1D and 2D
  implementations.

Their compile-time selection admits the 1D path when all participating ordinary
tiles have full valid columns, or when all physical tiles have one row. The
PTO-ISA types also expose `RowStride`, and its 2D forms use those row strides
explicitly.

PTODSL follows this ordinary legality model and makes the implicit tile-type
conditions explicit:

- every named traversal operand is checked, including temporary tiles;
- the tiles must describe the same static logical valid shape;
- row-major/none-box local tiles are gap-free when their compact mode is
  `null` or `normal`;
- every tile must use its full physical column axis, or the logical valid
  region must occupy only the first row;
- unknown information is rejected;
- predicate and conversion representations remain separate family rules.

Pinned references:

- `https://gitcode.com/cann/pto-isa/blob/57bd62714023b18d082456da2014398957e00a81/include/pto/npu/a5/TUnaryOp.hpp`
- `https://gitcode.com/cann/pto-isa/blob/57bd62714023b18d082456da2014398957e00a81/include/pto/npu/a5/TBinOp.hpp`
- `https://gitcode.com/cann/pto-isa/blob/57bd62714023b18d082456da2014398957e00a81/include/pto/npu/a5/TBinSOp.hpp`

## Shared Ordinary 1D Legality Infrastructure

`tilelib.require_elementwise_1d(*operand_names)` implements the reusable
ordinary-tile rule. It is intentionally opt-in: operation registrations must
name every tile that participates in traversal, including ABI temporary tiles.
The predicate accepts only when all of the following are proven:

1. Every named operand is a rank-2 tile with positive static physical and valid
   shapes.
2. Every valid extent is within its physical extent.
3. Every tile is in local `ub`/`vec` memory with `row_major` block layout,
   `none_box` sub-layout, and a gap-free compact mode (`null` or `normal`).
4. All named tiles have the same logical valid shape.
5. Either every tile has `valid_cols == physical_cols`, or the common logical
   valid row count is one.

The logical one-row case is safe even when the physical tile has additional
rows because a TileBuf valid region begins at the first element and never
crosses a row boundary. Multi-row partial-column regions are rejected.

This predicate is not sufficient for predicate tiles or dtype-width-changing
and packed conversions. Predicate compare uses the separate rule below;
predicate select (`tsel` and `tsels`) and conversion (`tcvt`) must establish
their logical-to-physical representation rules before registering a 1D
candidate.

### Packed predicate compare legality

`tilelib.require_predicate_compare_1d(*data_operand_names,
predicate_operand="dst")` implements the compare-specific rule. The data
tiles must satisfy the same rank, locality, layout, compact-mode, bounds, and
common-valid-range requirements as ordinary element-wise operands. The
predicate destination is checked independently because it stores one bit per
comparison instead of one element of the source dtype.

For f32/i32 and f16/i16 comparisons, one complete predicate store represents
128 source elements and occupies 16 bytes. For i8/ui8 comparisons, one store
represents 256 source elements and occupies 32 bytes. A one-row range is legal
when every source physical row can contain the rounded full vector loads and
the destination physical row can contain the rounded number of stores.
For multiple rows, the source valid column count must be a multiple of the
corresponding 128- or 256-element store unit, every data tile must use its full
physical column axis, and the destination physical row stride must equal the
exact packed byte count for that row. The predicate physical row must also
satisfy A5's 32-byte tile-row alignment. Unknown dtype, shape, layout, compact
mode, alignment, or insufficient destination capacity rejects the 1D
candidate.

### Packed predicate select legality

`tilelib.require_predicate_select_1d(predicate_operand,
*data_operand_names, temporary_operand=...)` implements the inverse packed
representation rule for `tsel` and `tsels`. All data operands must have one
common static valid shape and satisfy local row-major, none-box, gap-free
continuity. The mask row byte width is `mask.shape[1] * bytewidth(mask.dtype)`;
the nominal mask dtype is a storage container and does not change the one-bit
predicate interpretation.

The same 128-element/16-byte unit is used for 32- and 16-bit data, and the same
256-element/32-byte unit is used for 8-bit data. A single row requires enough
rounded data and mask capacity. Multiple rows additionally require full data
columns, a valid column count ending on the relevant predicate unit, and a
mask row stride equal to the exact packed bytes per row. The mask physical row
must satisfy 32-byte alignment. The A5-unused temporary is checked for static
local layout and compact-mode metadata but does not share the data range.

The older `require_contiguous()` helper remains available for existing users;
new element-wise 1D candidates must use the named-operand rule.

### Conversion legality

`tilelib.require_conversion_1d()` proves that the source and destination are
two independently contiguous typed streams. Both operands must be static
rank-2 local tiles with row-major, none-box, gap-free compact storage. Ordinary
conversions require equal physical and valid shapes. Multi-row regions must
fill both physical column axes; a single logical row may use partial columns
because neither stream crosses a row boundary.

Changing dtype width does not by itself make flattening illegal. Each source
and destination pointer advances in its own element type, while the existing
unpack/pack distribution mode and mask representation preserve the conversion
body's logical-element mapping. BF16-to-FP4 supplies
`source_elements_per_destination=2`, requiring source columns to be exactly
twice the packed destination columns. Unknown memory, layout, shape, or compact
metadata rejects the 1D candidate.

The current A5 PTODSL candidates have only `(src, dst)` operands. A5's
non-saturating multi-step conversions use register temporaries, so no ABI
temporary tile is omitted from the proof. A future candidate with a tile
temporary must extend the conversion rule to validate that operand before it
can gain a 1D form.

## Metadata and Legality Status

### Current continuity predicate

`tilelib.require_contiguous()` returns true when all tile valid
columns equal their physical columns, or when all physical row counts equal
one. It does not:

- require equal logical valid shapes;
- inspect valid row counts for the single-row case;
- name the exact participating operands;
- represent an explicit tile row stride;
- model predicate or conversion packing;
- distinguish unknown metadata from a proven-contiguous range.

It is therefore not the shared 1D legality rule.

### Tile specialization metadata

The compiler-side operand JSON includes physical shape, valid shape, memory
space, block layout, sub-layout, sub-fractal size, pad value, and compact mode.
The compiler-side `SpecKey` also includes these fields.

Python `TileSpec`, daemon reconstruction, rendered `tile_buf` types, and the
compiler JSON path now preserve `s_fractal_size` and `compact_mode`. The
`ExpandTileOp` specialization key and generated helper name also distinguish
compact mode.

PTO tile layout lowering establishes that an unboxed row-major tile has
`col_stride == 1`. Its row stride equals the physical column count for compact
mode `null` or `normal`; `row_plus_one` adds a stride gap. The shared predicate
therefore accepts the first two modes and rejects `row_plus_one` or unknown
compact metadata. No explicit row-stride field is needed for this restricted
ordinary layout contract.

### Candidate ordering

The Python registry and daemon sort legal candidates by descending priority.
Equal-priority reporting is canonicalized by candidate name, while an equal
top-priority match is rejected as ambiguous.

The daemon metadata response represents candidates as a ranked JSON array.
`InsertTemplateAttributes` validates unique candidate IDs without reordering
the array, stores the same order in the compact candidates attribute, and
`ExpandTileOp` selects candidate zero by name. Candidate IDs are stable identity
fields only and do not participate in ranking.

A compiler regression supplies a preferred candidate with ID 99 and a fallback
with ID 0. It verifies both the compact attribute order and that expansion
requests the ID-99 candidate, preventing an accidental return to ID-based
selection.

### Context attributes

The two compiler stages both forward the scoped body-changing context
attributes:

- `round_mode` for `tcvt`;
- `cmp_mode` for `tcmp` and `tcmps`;
- `precisionType` for the precision-aware unary/division families.

The selection and specialization work must keep the two C++ reconstructions in
lockstep. Candidate filtering may use a context attribute directly, as `tlog`
does, or the rendered body may consume it, as `tdiv`, `tdivs`, `tcmp`, `tcmps`,
and `tcvt` do.

## Proposed Candidate Identity Scheme

Candidate identity and ranking must remain separate.

1. Preserve every existing candidate name and ID for its 2D fallback. This
   minimizes churn in named-render tests and keeps current IDs stable.
2. Add `_1d` to each new flattened candidate name.
3. Give a legal 1D candidate a higher priority than its 2D fallback.
4. Use `loop_depth=1` for flattened candidates.
5. Retain `loop_depth=2` for fallback candidates.
6. Never rely on the numeric ID to select the winner.

Proposed ID allocation:

| Existing form | Preserved 2D IDs | New 1D IDs |
|---|---|---|
| Ordinary one-candidate op | `0` | `1` |
| `tlog` default/high-precision | `0`, `1` | `2`, `3`, paired in the same order |
| `tdivs` tile-scalar/scalar-tile | `0`, `1` | `2`, `3`, paired in the same order |
| `tcvt` semantic variants | `0` through `37` | `38 + existing_id`, giving `38` through `75` |

This allocation deliberately makes the preferred 1D IDs larger than the
fallback IDs, so an accidental ascending-ID sort is exposed by tests rather
than appearing to work.

## Resolved Conversion Decisions

1. The registered A5 PTODSL `tcvt` forms contain only source and destination
   tiles; their multi-step paths use register temporaries.
2. Width-changing forms use equal logical element offsets on independently
   typed pointers. Their existing load/store distributions define the bytes
   consumed and produced per iteration.
3. A5 supplies flat forms for every registered packed and multi-step category;
   no form requires per-row state when both streams are proven contiguous.
4. `pad_value` does not change conversion selection because the 1D body stores
   only the mask-bounded logical range and does not consume padding values as
   operands.
5. Non-static shapes and unknown memory, layout, sub-layout, or compact-mode
   metadata conservatively reject the 1D candidate and retain the 2D fallback.

The implementation follows PTO-ISA revision
`23e31ddf51233835810997ba7cff12fda2808f50`, principally
`include/pto/common/arch/register/tcvt_common.hpp`, which exposes paired 1D and
2D conversion helpers and selects the flat path for full-column or single-row
tiles.

## Coverage Checklist

The inventory accounts for every operation in the issue exactly once:

- Unary: `tabs`, `texp`, `tlog`, `tneg`, `tnot`, `trecip`, `trelu`, `trsqrt`,
  `tsqrt`.
- Tile-Tile: `tadd`, `tand`, `tdiv`, `tfmod`, `tmax`, `tmin`, `tmul`, `tor`,
  `tprelu`, `trem`, `tshl`, `tshr`, `tsub`, `txor`.
- Tile-Scalar: `tadds`, `tands`, `tdivs`, `tfmods`, `tlrelu`, `tmaxs`,
  `tmins`, `tmuls`, `tors`, `trems`, `tshls`, `tshrs`, `tsubs`, `txors`.
- Other scoped element-wise families: `tcmp`, `tcmps`, `tsel`, `tsels`,
  `tcvt`, `texpands`.

Future catalog tests should encode this list as data and require each operation
to provide:

- a legal 2D fallback;
- a preferred legal 1D candidate for eligible metadata; or
- a named, reviewed, and tested exception record.
