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
from Python through the compact candidate attribute and `ExpandTileOp`, but no
existing candidate consumes the new predicate yet. Template registrations,
family implementations, functional tests, and performance measurements remain
subsequent implementation steps.

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

The scope contains 43 TileOps and 82 currently registered PTODSL candidates:

| Family | TileOps | Registered candidates |
|---|---:|---:|
| Unary | 9 | 10 |
| Tile-Tile | 14 | 14 |
| Tile-Scalar | 14 | 15 |
| Compare, select, conversion, scalar fill | 6 | 43 |
| Total | 43 | 82 |

All 82 candidates currently declare `loop_depth=2`. The ordinary shared
element-wise implementations are row-wise. The important exception is
`tcmps`: its f32/i32 branch already traverses `valid_rows * valid_cols` as a
flat range even though the containing candidate is marked as two-dimensional.
That mixed candidate must be separated into explicit 1D and 2D forms.

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
| `tabs` | `template_tabs(src, dst)` | same `F2` | shared 2D | Shared |
| `texp` | `template_texp(src, dst)` | same `F2` | shared 2D, default precision only | Shared |
| `tlog` | `template_tlog(src, dst)`; `template_tlog_high_precision(src, dst)` | same `F2` | two precision-selected 2D candidates | Algorithm-specific |
| `tneg` | `template_tneg(src, dst)` | same `NEG6` | shared 2D | Shared |
| `tnot` | `template_tnot(src, dst)` | same `I6` | shared 2D | Shared |
| `trecip` | `template_trecip(src, dst)` | same `F2` | bespoke 2D using `1 / src`, default precision only | Algorithm-specific |
| `trelu` | `template_trelu(src, dst)` | same `RELU3` | shared 2D | Shared |
| `trsqrt` | `template_trsqrt(src, dst)` | same `F2` | shared 2D, default precision only | Shared |
| `tsqrt` | `template_tsqrt(src, dst)` | same `F2` | shared 2D, default precision only | Shared |

`precisionType` is forwarded for `texp`, `tlog`, `trecip`, `trsqrt`, and
`tsqrt`. In the current PTODSL templates only `tlog` uses it for candidate
selection; the other listed templates explicitly implement default precision
only. The 1D/2D work must preserve that current coverage unless precision
parity is approved as separate work.

## Tile-Tile Inventory

| Op | Existing candidate and operands | Dtypes | Current form | Provisional 1D classification |
|---|---|---|---|---|
| `tadd` | `template_tadd(src0, src1, dst)` | same `N9` | shared 2D | Shared |
| `tand` | `template_tand(src0, src1, dst)` | same `I6` | shared 2D | Shared |
| `tdiv` | `template_tdiv(src0, src1, dst)` | same `F2` | bespoke precision-aware 2D | Algorithm-specific |
| `tfmod` | `template_tfmod(src0, src1, dst)` | same `f32`, `f16`, `i16`, or `ui16` | shared remainder 2D | Algorithm-specific |
| `tmax` | `template_tmax(src0, src1, dst)` | same `N9` | shared 2D | Shared |
| `tmin` | `template_tmin(src0, src1, dst)` | same `N9` | bespoke registration, ordinary 2D compute | Shared |
| `tmul` | `template_tmul(src0, src1, dst)` | same `N9` | shared 2D | Shared |
| `tor` | `template_tor(src0, src1, dst)` | same `I6` | shared 2D | Shared |
| `tprelu` | `template_tprelu(src0, src1, tmp, dst)` | data/dst `f16` or `f32`; `tmp` is the data dtype or `i8` | shared binary 2D with temporary tile | Algorithm-specific |
| `trem` | `template_trem(src0, src1, tmp, dst)` | all `f32`, all `f16`, or all `i32` | shared remainder 2D with temporary tile | Algorithm-specific |
| `tshl` | `template_tshl(src0, src1, dst)` | same `I6` | shared 2D | Shared |
| `tshr` | `template_tshr(src0, src1, dst)` | same `I6` | shared 2D | Shared |
| `tsub` | `template_tsub(src0, src1, dst)` | same `N9` | shared 2D | Shared |
| `txor` | `template_txor(src0, src1, tmp, dst)` | same `I6` across all operands | shared binary 2D with temporary tile | Algorithm-specific |

`precisionType` is forwarded and consumed by `tdiv`. A 1D form must preserve
both its ordinary `vdiv` path and its high-precision helper path.

Every temporary tile remains part of the 1D legality decision even where the
current PTODSL body does not read it directly. Whether a temporary's logical
range must equal the data range or satisfy a representation-specific relation
is an operation-family decision; it must not be omitted from the predicate.

## Tile-Scalar Inventory

| Op | Existing candidate(s) and operands | Dtypes | Current form | Provisional 1D classification |
|---|---|---|---|---|
| `tadds` | `template_tadds(src, scalar, dst)` | same `N9` | shared 2D | Shared |
| `tands` | `template_tands(src, scalar, dst)` | same `I6` | shared 2D | Shared |
| `tdivs` | `template_tdivs_tile_scalar(src, scalar, dst)`; `template_tdivs_scalar_tile(scalar, src, dst)` | same `F2` | two operand-order-specific, precision-aware 2D candidates | Algorithm-specific |
| `tfmods` | `template_tfmods(src, scalar, dst)` | same `f32`, `f16`, `i32`, or `i16` | shared scalar remainder 2D | Algorithm-specific |
| `tlrelu` | `template_tlrelu(src, slope, dst)` | `(f16,f16,f16)`, `(f16,f32,f16)`, `(f32,f32,f32)` | bespoke 2D with scalar coercion | Algorithm-specific |
| `tmaxs` | `template_tmaxs(src, scalar, dst)` | same `N9` | shared 2D | Shared |
| `tmins` | `template_tmins(src, scalar, dst)` | same `N9` | shared 2D | Shared |
| `tmuls` | `template_tmuls(src, scalar, dst)` | same `N9` | shared 2D | Shared |
| `tors` | `template_tors(src, scalar, dst)` | same `I6` | shared 2D | Shared |
| `trems` | `template_trems(src, scalar, tmp, dst)` | all `f32` or all `f16` | shared scalar remainder 2D with temporary tile | Algorithm-specific |
| `tshls` | `template_tshls(src, scalar, dst)` | data/dst `I6`; scalar `i16` | shared 2D | Shared |
| `tshrs` | `template_tshrs(src, scalar, dst)` | data/dst `I6`; scalar `i16` | shared 2D | Shared |
| `tsubs` | `template_tsubs(src, scalar, dst)` | same `N9` | bespoke tile-minus-scalar 2D | Shared |
| `txors` | `template_txors(src, scalar, tmp, dst)` | same `I6` across all operands | shared scalar 2D; current same-shape constraint omits `tmp` | Algorithm-specific |

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
| `texpands` | `template_texpands(scalar, dst)` | same `FILL6` | shared scalar-fill 2D | Shared |

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

No `tcvt` candidate is yet classified as a confirmed 1D exception. The
generic, unpacked-load, packed-store, multi-step, widening, and low-precision
categories each require a family rule proving that a flat source chunk maps to
the corresponding flat destination chunk without crossing a physical gap or
changing tail semantics. BF16-to-FP4 additionally requires the existing 2:1
logical-column relationship to hold over the complete flat ranges.

## Existing Shared Traversal Structure

The ordinary implementations are concentrated in:

- `ptodsl/ptodsl/tilelib/templates/a5/_elementwise.py`
- `ptodsl/ptodsl/tilelib/templates/a5/_remainder.py`
- `ptodsl/ptodsl/tilelib/templates/a5/_common.py`

`_elementwise.py` registers unary, Tile-Tile, Tile-Scalar, and scalar-fill
templates. Each body restarts `remained = valid_cols` for every row and then
walks columns by vector lane count. `_remainder.py` uses the same traversal
around a more complex vector computation.

The operation-specific files that cannot be migrated by replacing a shared
registrar alone are:

- `tdiv.py`, `tdivs.py`, `tlog.py`, `trecip.py`, `tlrelu.py`, `tmin.py`, and
  `tsubs.py`;
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

This predicate is not sufficient for predicate tiles (`tcmp`, `tcmps`, `tsel`,
and `tsels`) or dtype-width-changing and packed conversions (`tcvt`). Those
families must establish their logical-to-physical representation rules before
registering a 1D candidate.

The older `require_contiguous()` helper remains available for existing users;
new element-wise 1D candidates must use the named-operand rule.

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
4. Use `loop_depth=1` and a `1d` tag for flattened candidates.
5. Retain `loop_depth=2` and add a `2d` tag to fallback candidates.
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

## Open Legality Questions for the Next Step

1. What exact logical-to-physical relation does an `i8`/`ui8` predicate tile
   represent for each input dtype in compare and select?
2. Are wider `tsels` mask dtypes representation choices, storage containers, or
   logical predicates with a different packing rule?
3. Which temporary tiles are true traversal participants, and which are ABI
   scratch operands whose required size relation differs from the data range?
   They must all still be checked.
4. For each `tcvt` distribution mode, what source and destination byte ranges
   are touched by a vector iteration and its tail?
5. Does any packed conversion require per-row restart state even when both
   physical buffers are contiguous?
6. Must `pad_value` affect 1D legality, or is it relevant only when an
   instruction can observe elements outside the logical valid range?
7. Which dynamic or unknown tile metadata encodings can reach
   `InsertTemplateAttributes`, and how should each be rejected conservatively?

These questions must be answered with focused legality tests or an explicitly
documented exception before the relevant family gains a 1D candidate.

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
