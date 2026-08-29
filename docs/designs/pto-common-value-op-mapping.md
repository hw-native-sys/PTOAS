# PTO Common Value Operation Mapping

## Purpose

This document is the implementation inventory for making `pto.*` the single
frontend dialect for common scalar and builtin-vector value computation. It
maps the LLVM 21 MLIR Arith dialect to two deliberately separate API layers:

- PTODSL keeps concise Python operators and type-directed helpers;
- PTO IR uses signless integer carriers and records signed or unsigned
  interpretation explicitly in a semantic attribute.

The inventory separates three cases that must not be conflated:

1. the same semantic operation is already represented by a common `pto.*` op;
2. a common operation is missing or exposes only part of the Arith contract;
3. a target-specific operation has explicit semantics that justify a separate
   PTO interface.

The stable operation semantics remain documented in
[Generic Scalar and Builtin Vector Operations](../isa/micro-isa/14-generic-scalar-ops.md).
This file is a design and coverage audit, so it intentionally names the
standard operations used after frontend legalization.

## PR Acceptance Criteria: Scalar/SIMT Interface Consistency

This PR owns scalar and SIMT programming/IR interface unification. Its review
and merge criteria are the following, rather than whether every target control
state has already received an optimizing implementation.

1. **Unified DSL authoring surface.** PTODSL exposes one scalar/SIMT
   programming interface for operations with the same source-level intent. It
   is sensitive to authored type and signedness, and selects the appropriate
   PTO IR operation automatically.
2. **Signless common IR types.** Common PTO IR uses sign-insensitive integer
   carrier types. Each operation whose interpretation depends on signedness
   records `signed` or `unsigned` uniformly as an operation attribute, rather
   than encoding it in the operand/result type.
3. **Operation names separate numeric categories.** Operations whose floating
   and integer forms have distinct contracts use distinct operation names (for
   example, `pto.addf`/`pto.addi`, `pto.cmpf`/`pto.cmpi`, and
   `pto.ftof`/`pto.ftoi`/`pto.itof`). Signedness within an integer contract is
   still represented by the attribute in criterion 2.
4. **Regular IR assembly.** PTO IR assembly is concise and regular across an
   operation family: the same semantic controls appear in the same position and
   with the same spelling, and no equivalent case has an accidental asymmetric
   syntax.

These criteria define frontend-interface completeness for this PR. Hardware
control-state materialization and its reuse/restoration optimization are
separate follow-up work unless needed to preserve a currently promised
observable semantic contract.

## Scope And Decision Rules

The common frontend covers scalar and builtin-vector values. It mirrors the
corresponding Arith semantic attributes and multi-result integer helpers so
frontends do not need to escape to `arith.*` for those contracts. Tensor
arithmetic remains outside this surface.

Use these rules when adding or classifying an interface:

- **Separate authoring from IR semantics:** PTODSL may use one operator or
  helper for several typed cases, but it must emit the PTO IR operation whose
  semantics have already been selected.
- **Keep semantic-neutral operations unified:** operations such as integer
  add, subtract, multiply, left shift, and bitwise logic do not need signed and
  unsigned PTO names because their bit-level result is identical.
- **Encode semantic variants in PTO IR:** division, remainder, right shift,
  integer ordering, extrema, and conversion must not recover signedness from a
  signless carrier. Their common PTO operation carries an explicit `signed` or
  `unsigned` attribute.
- **Keep common integer carriers signless:** common scalar and builtin-vector
  operations accept `i*`, not `si*` or `ui*`. Authored PTODSL `si*` and `ui*`
  values select the attribute at the boundary; legacy `i*` authoring uses the
  established signed convention.
- **Make the `index` authoring policy deterministic:** `index` has no authored
  signedness. PTODSL `//`, remainder, relational comparison, and extrema on
  `index` retain the existing signed/floor convention and attach `signed`.
  Explicit PTO IR may select either signed or unsigned interpretation when the
  corresponding standard operation supports `index`; this includes division,
  remainder, comparison, extrema, extended multiplication, and absolute value.
  Fixed-width-only shifts and numeric conversions continue to reject `index`;
  use `pto.index_cast` when a fixed-width integer representation is required.
- **Keep conversions unified:** source and destination types identify the
  conversion category and width; the `signedness` attribute identifies integer
  interpretation. The same rule applies to `pto.index_cast`.
- **Target-specific controls:** keep a separate interface when rounding,
  saturation, packed ABI, execution state, or another explicit control changes
  the observable contract.
- **No decomposition equivalence:** an operation is covered only by a PTO op
  with the same observable semantics. A sequence of other operations does not
  count as interface coverage.
- **Keep optional contracts explicit:** poison-producing overflow promises,
  fast-math permissions, and rounding modes default to absent/none and appear
  only when explicitly authored.

## Conversion Saturation Decision Record

This scalar-surface unification change standardizes the conversion interface;
it does not introduce general CTRL-state scheduling or control-word
save/restore optimization. This distinction is deliberate and must not be
treated as an incomplete `pto.cast` frontend mapping.

For SIMT conversion operations, `saturation` is consequently retained as a
uniform PTO IR attribute, including type pairs whose current instruction
encoding does not select saturation directly. The current VPTO lowering
normalizes `CTRL[60]` to zero for PTO entry functions, but it does not manage
`CTRL[48]` on behalf of generic conversion operations. Direct micro-instruction
authors remain able to manage CTRL explicitly.

The ISA behavior under that `CTRL[60] = 0` baseline is:

| Conversion form | Current saturation control | Meaning of the uniform PTO `saturation` attribute |
|-----------------|----------------------------|----------------------------------------------------|
| F2F, source dynamic range wider than destination, destination not `f32` | Instruction `#sat` | Effective. |
| F2F, source dynamic range narrower than destination, destination not `f32` | `CTRL[48]`; instruction `#sat` is ignored | Retained for interface symmetry, but currently has no observable effect. |
| F2F `f16 -> f16` or `bf16 -> bf16` | `CTRL[48]`; instruction `#sat` is ignored | Retained for interface symmetry, but currently has no observable effect. |
| F2F with destination `f32` | Fixed non-saturating; both controls are ignored | Retained for interface symmetry, but currently has no observable effect. |
| I2F narrowing | Instruction `#sat` | Effective. |
| I2F widening | Fixed non-saturating | Retained for interface symmetry, but currently has no observable effect. |
| F2I | Fixed saturating ISA behavior | PTO verification requires `sat`; it is not a selectable instruction control. |

A future, separate control-state pass may implement `CTRL[48]` materialization,
state reuse, and restoration. Until then, do not claim that the generic
`saturation` attribute controls the CTRL-governed F2F forms, and do not require
that infrastructure as a prerequisite for frontend-interface completeness.

## Status Definitions

| Status | Meaning |
|--------|---------|
| **Covered** | The target PTO IR explicitly represents the relevant scalar and builtin-vector semantics. |
| **Partial** | The semantics are representable today, but the PTO IR still relies on a generic op, incomplete type/predicate coverage, or an incorrect classification. |
| **Gap** | The target semantic PTO IR operation has no representation yet. |
| **Specialized** | A separate PTO interface is intentional because it exposes target-specific controls or behavior. |
| **Out of scope** | The capability is currently an optimizer/IR facility, tensor form, or uncommon multi-result primitive rather than a common author operation. |

## Arith Mapping Matrix

### Constants And Basic Arithmetic

| Arith capability | Target PTO IR | Status | Contract and action |
|------------------|--------------------|--------|---------------------|
| `arith.constant` | `pto.constant` | **Covered** | Numeric scalar and dense builtin-vector constants are covered. Tensor and non-numeric constants are outside the common value surface. |
| `arith.addi`, `arith.addf` | `pto.addi`, `pto.addf` | **Covered** | Integer and floating-point contracts are separate, following the cuTile category split. |
| `arith.subi`, `arith.subf` | `pto.subi`, `pto.subf` | **Covered** | Integer and floating-point subtraction have distinct operation types. |
| `arith.muli`, `arith.mulf` | `pto.muli`, `pto.mulf` | **Covered** | Integer and floating-point multiplication have distinct operation types. |
| `arith.negf` and integer additive inverse | `pto.negf`, `pto.negi` | **Covered** | Integer negation carries overflow promises; floating negation carries fast-math. |
| Integer overflow flags on add/sub/mul/neg | integer `pto.*i overflow<...>` | **Covered** | `nsw`/`nuw` are explicit poison-producing promises and never appear on floating operations. |
| `arith.addui_extended` | `pto.addui_extended` | **Covered** | Returns the wrapped unsigned sum and the scalar or same-shape builtin-vector `i1` overflow result. |
| `arith.mulsi_extended`, `arith.mului_extended` | `pto.mul_extended signed/unsigned` | **Covered** | One two-result operation carries explicit integer interpretation, including explicit signed or unsigned `index` forms. |

### Division, Remainder, Bitwise, And Shift

| Arith capability | Target PTO IR | Status | Contract and action |
|------------------|--------------------|--------|---------------------|
| `arith.divf` | `pto.divf` | **Covered** | PTODSL `/` on floating values emits the explicit floating operation. |
| `arith.divsi`, `arith.divui` | `pto.divi signed/unsigned` | **Covered** | Signed division rounds toward zero; unsigned division uses unsigned interpretation. Both interpretations accept `index` in explicit PTO IR. |
| `arith.floordivsi`, unsigned floor division | `pto.floordiv signed/unsigned` | **Covered** | Signed floor division rounds toward negative infinity; unsigned floor division is equivalent to unsigned division. Both interpretations accept `index` in explicit PTO IR. |
| `arith.ceildivsi`, `arith.ceildivui` | `pto.ceildiv signed/unsigned` | **Covered** | The attribute selects signed or unsigned ceiling division, including for explicit `index` IR. |
| `arith.remsi`, `arith.remui` | `pto.remi signed/unsigned` | **Covered** | The attribute selects signed or unsigned remainder, including for explicit `index` IR. |
| `arith.remf` | `pto.remf` | **Covered** | Floating-point remainder is explicit. |
| `arith.andi`, `arith.ori`, `arith.xori` | `pto.and`, `pto.or`, `pto.xor` | **Covered** | Integer and `i1` bitwise operations are type-directed. |
| `arith.shli` | `pto.shl` | **Covered** | Fixed-width integer left shift with optional explicit `nsw`/`nuw` promises. |
| `arith.shrsi`, `arith.shrui` | `pto.shr signed/unsigned` | **Covered** | Signed selects arithmetic right shift; unsigned selects logical right shift. |

### Extrema

| Arith capability | Target PTO IR | Status | Contract and action |
|------------------|--------------------|--------|---------------------|
| `arith.maxsi`, `arith.maxui` | `pto.maxi signed/unsigned` | **Covered** | Integer interpretation is a required attribute. |
| `arith.minsi`, `arith.minui` | `pto.mini signed/unsigned` | **Covered** | Integer interpretation is a required attribute. |
| `arith.maxnumf`, `arith.minnumf` | `pto.maxf`, `pto.minf` | **Covered** | Floating extrema use maxNum/minNum behavior without any signedness attribute. |
| `arith.maximumf`, `arith.minimumf` | `pto.maximum`, `pto.minimum` | **Covered** | These propagate NaN and define `-0.0 < +0.0`; they are intentionally distinct from maxNum/minNum. |

`pto.absi` follows the same explicit-interpretation rule. Signed `index`
computes the mathematical absolute value; unsigned `index` is unchanged.

### Numeric Conversion And Bit Reinterpretation

| Arith capability | Target PTO IR | Status | Contract and action |
|------------------|--------------------|--------|---------------------|
| `arith.extsi`, `arith.extui` | `pto.exti signed/unsigned` | **Covered** | Common integer types are signless; the required attribute selects sign or zero extension. |
| `arith.trunci` | `pto.trunci` | **Covered** | Truncation is signedness-independent and carries optional `nsw`/`nuw` promises. |
| `arith.extf`, `arith.truncf` | `pto.ftof` | **Covered** | Floating format conversion supports fast-math; narrowing also supports an explicit rounding mode. |
| `arith.sitofp`, `arith.uitofp` | `pto.itof signed/unsigned` | **Covered** | The required attribute selects integer interpretation. |
| `arith.fptosi`, `arith.fptoui` | `pto.ftoi signed/unsigned` | **Covered** | The required attribute selects the destination integer range; conversion does not saturate. |
| `arith.index_cast`, `arith.index_castui` | `pto.index_cast signed/unsigned` | **Covered** | One operation supports scalar and matching builtin-vector forms. |
| `arith.bitcast` | `pto.bitcast` | **Covered** | Equal-bit-width scalar and same-shape builtin-vector reinterpretation remains distinct from numeric conversion. |

`arith.scaling_extf` and `arith.scaling_truncf` are not part of the LLVM 19
Arith dialect used by PTOAS, so they are outside this baseline mapping.

### Comparison And Selection

| Arith capability | Target PTO IR | Status | Contract and action |
|------------------|--------------------|--------|---------------------|
| `arith.cmpi` equality | `pto.cmpi eq/ne ... signed/unsigned` | **Covered** | Equality is bitwise-independent of interpretation, but the integer interface records signedness uniformly. |
| Signed/unsigned `arith.cmpi` relations | `pto.cmpi lt/le/gt/ge ... signed/unsigned` | **Covered** | Predicate and signedness are separate required attributes. |
| Ordered `arith.cmpf` relations | `pto.cmpf eq/ne/lt/le/gt/ge` | **Covered** | The six short predicates mean `oeq/one/olt/ole/ogt/oge`. |
| Unordered and classification `arith.cmpf` predicates | `pto.cmpf` | **Covered** | Explicit ordered, unordered, classification, `false`, and `true` predicates are available. |
| `arith.select` with scalar condition and scalar values | `pto.select` | **Covered** | Scalar `i1` selects one scalar value. |
| `arith.select` with same-shape vector condition | `pto.select` | **Covered** | Selection is elementwise. |
| `arith.select` with scalar condition and vector values | `pto.select` | **Covered** | A scalar `i1` selects either whole builtin vector. |
| Tensor comparison and selection | none | **Out of scope** | The common PTO frontend is intentionally scalar and builtin-vector based. |

### Cross-Cutting Arith Facilities

| Arith capability | Target PTO IR | Status | Contract and action |
|------------------|--------------------|--------|---------------------|
| Fast-math flags on floating-point operations | corresponding `pto.* fastmath<...>` | **Covered** | PTO mirrors Arith's flag set. The default is `none`; explicit PTODSL helper keywords are required to relax semantics. |
| Tensor elementwise forms | none | **Out of scope** | PTO tile and VMI abstractions own higher-rank computation; builtin-vector coverage does not imply tensor coverage. |
| Poison contracts | overflow attributes on the relevant `pto.*` op | **Covered** | Poison is not a standalone operation: it is the result contract of explicit `nsw`/`nuw` on add/sub/mul/shl and integer truncation. |

## Adjacent PTO Scalar And SIMT Inventory

These operations are not direct Arith inventory rows, but they affect the same
interface-unification decision.

| Existing PTO operation | Classification | Required action |
|------------------------|----------------|-----------------|
| `pto.ftof`, `pto.ftoi`, or `pto.itof` | **Covered** | These are the category-specific PTO IR emitted by the single PTODSL `pto.cast` authoring API. Standard forms legalize to Arith; explicit SIMT controls, packed payloads, or SIMT-only rounding retain the op for backend lowering. |
| `pto.fma` | **Covered** | Fused multiply-add is a common value operation. Ordinary scalar/vector forms use standard `math.fma`; supported packed forms may retain target-specific backend lowering. |
| `pto.pow` | **Covered** | Power is a common value operation like `pto.exp/log/sqrt`; ordinary forms use standard math legalization and supported packed forms may retain target-specific lowering. |
| `pto.round`, `pto.rint` | **Specialized** | Preserve separate interfaces wherever target behavior is part of the contract. Conversion controls remain attributes of `pto.cast`-emitted category operations. |

## Implementation Backlog

### Priority 1: Closed Common Semantic Gaps

1. Added category-specific `pto.exti/trunci/ftof/ftoi/itof` operations and kept
   PTODSL `pto.cast(...)` as the convenience API selecting among them.
2. Added category-specific `pto.remi` and `pto.remf` operations.
3. Extended `pto.cmpf` with the complete floating-point predicate set,
   while retaining the six ordered shorthand predicates.
4. Added `pto.bitcast` for equal-bit-width scalar and same-shape builtin-vector
   reinterpretation.

Each item requires synchronized ODS, verifier, parser/printer where applicable,
standard legalization, Python surface support, documentation, and focused IR
plus PTODSL regression tests.

### Priority 2: Materialized Explicit PTO IR Semantics

Completed in the scalar-surface unification implementation:

1. Kept one `pto.divi/floordiv/ceildiv/remi/shr` operation per integer semantic family
   and added an explicit signedness attribute.
2. Split integer and floating extrema into `pto.maxi/mini` and `pto.maxf/minf`;
   integer forms require signedness while floating forms retain maxNum/minNum
   behavior. `pto.maximum/minimum` remain the NaN-propagating variants.
3. Kept comparison predicates type-independent and attached signedness
   separately for all integer and index comparisons.
4. Unified signed and unsigned index conversion under `pto.index_cast` with a
   required signedness attribute.
5. Tightened all common integer carriers to signless `i*`; category conversion
   operations use signedness attributes instead of signed/unsigned carrier types.
6. Added scalar-condition whole-vector `pto.select`.
7. Reclassified `pto.fma` and `pto.pow` as common value operations while
   retaining target-specific packed lowering only where required by the
   backend ABI.

### Priority 3: Deliberately Separate Interfaces

- tensor forms and other higher-rank standard arithmetic.

## Completion Criteria

The common value interface is considered closed when:

- every LLVM 21 Arith operation is represented by one row in this matrix;
- every **Gap** has either an implemented PTO interface or an explicit
  out-of-scope decision with rationale;
- every **Partial** row has a recorded compatibility and completion decision;
- PTODSL author paths emit common PTO operations with signless integer carriers
  and explicit signedness where required, rather than direct Arith arithmetic;
- specialized SIMT operations remain distinguishable through explicit
  semantic controls rather than execution placement alone;
- tests cover scalar and builtin-vector forms, both signedness attributes, NaN
  and out-of-range conversion behavior, and parse/print round
  trips where relevant.
