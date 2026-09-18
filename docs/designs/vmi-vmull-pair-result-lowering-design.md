# VMI VMULL Pair-Result Lowering Design

## Status

Proposed design for `mouliangyu/PTOAS:feature-vmi`.

This document defines the implementation contract for adding executable
`pto.vmi.vmull` support. The initial pull request containing this document is
design-only. ODS, verifier, lowering, PTODSL, user documentation, and tests are
follow-up implementation work.

## 1. Motivation

The physical VPTO operation already represents a native 32-bit widening
multiply as a pair of 32-bit vector results:

```mlir
%low, %high = pto.vmull %lhs, %rhs, %mask
    : !pto.vreg<64xi32>, !pto.vreg<64xi32>, !pto.mask<b32>
      -> !pto.vreg<64xi32>, !pto.vreg<64xi32>
```

One physical operation processes 64 `i32` or `ui32` lanes. VMI must expose the
same logical operation for 64, 128, and 256 lanes and split a larger logical
operation into the required physical operations automatically.

The current VMI definition instead returns one `Lxi64` value:

```mlir
%result = pto.vmi.vmull %a, %b, %mask
    : !pto.vmi.vreg<Lxi32>, !pto.vmi.vreg<Lxi32>, !pto.vmi.mask<Lxpred>
      -> !pto.vmi.vreg<Lxi64>
```

That form does not match the existing physical op. It would require a separate
virtual width-axis representation to translate one logical `i64` result into
low/high `i32` physical registers. The current VMI type converter has no such
result-axis contract, and there is no `VMIVmullOp` conversion pattern.

The proposed surface therefore represents the widened product as two logical
`Lxi32` results. This makes every physical chunk a direct pair-result lowering
while retaining VMI's logical lane count.

## 2. Goals and non-goals

### 2.1 Goals

- Support logical lane counts `L in {64, 128, 256}`.
- Support signed `i32` and unsigned `ui32` inputs.
- Return logical `%low` and `%high` values, both with the same `LxT` type as
  the inputs.
- Lower a contiguous 64/128/256-lane operation to exactly 1/2/4 physical
  `pto.vmull` operations.
- Support same-layout contiguous and deinterleaved factor-2/factor-4 dense
  values with `lane_stride = 1`.
- Preserve one common layout relation across both inputs, the mask, and both
  results.
- Preserve the low-result and high-result grouping expected by MLIR 1:N type
  conversion.
- Expose the pair result through PTODSL as a Python tuple.
- Reject unsupported shapes and predicate modes before conversion instead of
  leaving residual VMI IR.

### 2.2 Non-goals

- Supporting lane counts other than 64, 128, and 256.
- Reconstructing a logical `Lxi64` value from the low/high result pair.
- Adding a new VPTO op or changing the existing VPTO emitter.
- Adding partial-register or tail-specific VMULL behavior.
- Supporting `block_deinterleaved` VMULL layouts.
- Implementing the code in the design-only pull request.

## 3. Proposed VMI operation contract

### 3.1 Syntax

```mlir
%low, %high = pto.vmi.vmull %a, %b, %mask
    : !pto.vmi.vreg<LxT>, !pto.vmi.vreg<LxT>, !pto.vmi.mask<Lxpred>
      -> !pto.vmi.vreg<LxT>, !pto.vmi.vreg<LxT>
```

Examples:

```mlir
%low64, %high64 = pto.vmi.vmull %a64, %b64, %mask64
    : !pto.vmi.vreg<64xi32>, !pto.vmi.vreg<64xi32>,
      !pto.vmi.mask<64xpred>
      -> !pto.vmi.vreg<64xi32>, !pto.vmi.vreg<64xi32>

%low256, %high256 = pto.vmi.vmull %a256, %b256, %mask256
    : !pto.vmi.vreg<256xui32>, !pto.vmi.vreg<256xui32>,
      !pto.vmi.mask<256xpred>
      -> !pto.vmi.vreg<256xui32>, !pto.vmi.vreg<256xui32>
```

### 3.2 Type constraints

The verifier requires all of the following:

```text
L is exactly 64, 128, or 256
T is exactly i32 or ui32
type(a) == type(b) == type(low) == type(high)
lanes(mask) == L
```

At the surface boundary, none of these types carries an assigned layout or a
concrete mask granularity. After assignment, the data values retain identical
element type, lane count, and layout. The mask has the same logical lane count
and layout and uses `b32` granularity.

Mixed signedness is not legal. In particular, `i32` inputs cannot return
`ui32` results, and `ui32` inputs cannot return `i32` results.

Here `i32` means MLIR's signless 32-bit integer type and `ui32` means its
unsigned 32-bit integer type. Explicitly signed `si32` is not an alias for
`i32` in this contract, and `si32`/`si64` must be rejected. The verifier must
test `isSignless()` or `isUnsigned()` explicitly; treating every type for
which `isUnsigned()` is false as signed would incorrectly admit `si32`.

The old single `Lxi64` result form is removed rather than supported as a second
assembly form. Repository search shows no active VMI VMULL tests or kernel
consumers, so an atomic update of ODS, PTODSL, documentation, and tests is
preferred over maintaining two contracts.

### 3.3 Per-lane semantics

For active lane `i`, the operation computes one 64-bit product and returns its
two 32-bit halves:

```text
if T == i32:
  product = signed_64(a[i]) * signed_64(b[i])
else:
  product = unsigned_64(a[i]) * unsigned_64(b[i])

low[i]  = bits(product, 31, 0)
high[i] = bits(product, 63, 32)
```

The result values use `T` so that signedness continues to select the existing
signed or unsigned physical VMULL form. The low 32 bits have the same bit
pattern for signed and unsigned multiplication; the high 32 bits reflect the
selected signedness.

### 3.4 Predicate mode

The only predicate mode is zeroing:

```text
mask[i] == true:
  low[i], high[i] are the product halves

mask[i] == false:
  low[i] = 0
  high[i] = 0
```

For minimum assembly churn, `OptionalAttr<StrAttr>:$pmode` may remain in ODS,
but the verifier accepts only an omitted attribute or `pmode = "zero"`. An
omitted attribute means zeroing.

## 4. Layout contract

VMULL is an elementwise pair producer. Logical lane `i` in `%a`, `%b`, and
`%mask` maps to logical lane `i` in both `%low` and `%high`.

The assigned layout relation is:

```text
layout(a) == layout(b) == layout(mask) == layout(low) == layout(high)
```

The initial implementation supports exactly these dense layouts:

- contiguous with `lane_stride = 1`;
- deinterleaved factor 2 with `lane_stride = 1`;
- deinterleaved factor 4 with `lane_stride = 1`.

Group-slot layouts, dense lane strides greater than one, and deinterleaved
block layouts are not part of the initial contract. `VMILayoutAttr` represents
the latter as `block_deinterleaved` for other operations, but VMULL preflight
must reject them with an actionable diagnostic.

Contiguous is the default layout and gives the required canonical expansion:

| Logical type | Physical input parts | Physical VMULL count | Physical results |
|---|---:|---:|---:|
| `64xi32` | 1 | 1 | 1 low + 1 high |
| `128xi32` | 2 | 2 | 2 low + 2 high |
| `256xi32` | 4 | 4 | 4 low + 4 high |

For the supported ordinary deinterleaved layouts, the exact physical
arity is:

| Logical lanes | factor 2 | factor 4 |
|---:|---:|---:|
| 64 | 2 | 4 |
| 128 | 2 | 4 |
| 256 | 4 | 4 |

Lowering still obtains this arity from `getVMIPhysicalArity` rather than
hard-coding the table in the conversion pattern. All five logical values must
have the same physical arity. Corresponding parts of the four data values
`%a`, `%b`, `%low`, and `%high` must have the same `!pto.vreg<64xi32/ui32>`
type, while each mask part must be the corresponding `!pto.mask<b32>`.
Restricting the layout kind closes the initial support set explicitly;
`block_deinterleaved=2/4` is rejected even when its physical arity is otherwise
computable.

### 4.1 Assignment and propagation integration

VMULL cannot be implemented only in the final conversion pattern. The earlier
passes must establish its mask and layout contract:

1. `VMIMaskGranularityAssignment` requests `b32` for the VMULL mask because
   the data element type is 32-bit.
2. `VMILayoutAssignment` follows the ordinary elementwise `unite()` path for
   `%a`, `%b`, `%low`, and `%high`. This registers the values with the layout
   solver without placing them in a hard data-layout DSU equivalence class.
3. `VMILayoutPropagation` treats `VMIVmullOp` as a same-layout operation so a
   layout fact on any input, mask, or result propagates to the other ports.
4. Conflicting consumer layouts are handled through the existing explicit
   `ensure_layout` or `ensure_mask_layout` materialization mechanism.

`uniteDataEquivalent` must not be used for VMULL. It is reserved for values
that are genuinely the same SSA value across control-flow, call, and function
boundaries. Using it for an elementwise producer would turn compatible layout
requests into hard natural/preferred-layout conflicts and prevent use-site
materialization.

No VMULL-specific layout transform is needed. The relation is identity across
all ports, unlike `vintlv`/`vdintlv`, whose input and result layouts may differ.

## 5. VMI-to-VPTO lowering

### 5.1 Canonical contiguous expansion

For contiguous `Lxi32`, define:

```text
K = L / 64
```

The type converter produces:

```text
a    -> [a_0,    ..., a_(K-1)]
b    -> [b_0,    ..., b_(K-1)]
mask -> [mask_0, ..., mask_(K-1)]
low  -> [low_0,  ..., low_(K-1)]
high -> [high_0, ..., high_(K-1)]
```

The conversion creates one physical operation for every `p` in `[0, K)`:

```text
low_p, high_p = pto.vmull(a_p, b_p, mask_p)

a_p, b_p, low_p, high_p : !pto.vreg<64xi32> or !pto.vreg<64xui32>
mask_p                  : !pto.mask<b32>
```

### 5.2 Conversion pattern algorithm

`OneToNVMIVmullOpPattern` performs the following checks and rewrite:

1. Read `aParts`, `bParts`, and `maskParts` from the 1:N adaptor.
2. Obtain the converted type lists for result index 0 (`low`) and result index
   1 (`high`).
3. Require all five arities to be equal and non-zero.
4. Require every input and result part to be `!pto.vreg<64xi32>` or
   `!pto.vreg<64xui32>` with matching signedness, and every mask part to be
   `!pto.mask<b32>`.
5. For physical part `p`, create
   `pto.vmull(aParts[p], bParts[p], maskParts[p])`.
6. Replace the two logical results with the flattened physical result list.

The flattened replacement order is critical:

```text
[low_0, low_1, ..., low_(K-1), high_0, high_1, ..., high_(K-1)]
```

It must not be interleaved as `[low_0, high_0, low_1, high_1, ...]`.
`replaceOpWithFlatConvertedValues` partitions the flat list by logical result
index, so the first complete segment belongs to `%low` and the second complete
segment belongs to `%high`.

### 5.3 Preflight validation

`verifySupportedVMIToVPTOOps` gains an explicit VMULL check before dialect
conversion. The check requires:

- assigned, equal, supported dense layouts on all ports;
- contiguous, or deinterleaved factor 2/4;
- `lane_stride = 1` on every port;
- `b32` mask granularity;
- element type exactly signless `i32` or unsigned `ui32`, and a legal lane
  count;
- matching, computable physical arity for inputs, mask, and both results;
- physical part types compatible with `pto.vmull`.

This gives an actionable `VMI-UNSUPPORTED` diagnostic instead of a final
`VMI-RESIDUAL-OP` failure.

## 6. Pipeline integration

The complete implementation crosses the following layers:

| Layer | File | Required change |
|---|---|---|
| ODS | `include/PTO/IR/VMIOps.td` | Change one `Lxi64` result to `(low, high)` `Lxi32` results and update syntax/description |
| Verifier | `lib/PTO/IR/VMI.cpp` | Enforce legal lane counts, exact signless/unsigned types, pair equality, mask shape, and zero-only pmode |
| Mask assignment | `lib/PTO/Transforms/VMIMaskGranularityAssignment.cpp` | Request `b32` for the mask use |
| Layout assignment | `lib/PTO/Transforms/VMI/VMILayoutAssignment.cpp` | Use ordinary elementwise `unite()` bookkeeping for both inputs and both results; do not use `uniteDataEquivalent` |
| Layout propagation | `lib/PTO/Transforms/VMI/VMILayoutPropagation.cpp` | Register VMULL as a same-layout relation |
| Unified bridge | `lib/PTO/Transforms/VMI/VMILowerUnifiedToLegacy.cpp` | Keep VMULL on the direct-to-VPTO path; update comments only if needed |
| Physicalization | `lib/PTO/Transforms/VMIToVPTO.cpp` | Add preflight validation, pair-result 1:N pattern, and pattern registration |
| PTODSL | `ptodsl/ptodsl/_vmi_namespace.py` | Return two results and infer their types from the inputs |
| User docs | VMI ISA and PTODSL guide | Replace the single widened result with the pair-result contract |
| Tests | `test/lit/vmi_new`, `ptodsl/tests` | Add verifier, assignment, lowering, and frontend coverage |

The existing `PTO_VmullOp`, `VmullOp::verify`, LLVM emitters, and physical
VMULL tests remain unchanged.

## 7. PTODSL API

The Python surface follows the existing two-result `vintlv`/`vdintlv` style:

```python
low, high = pto.vmi.vmull(
    a,
    b,
    mask,
    pmode=None,
)
```

Both result types are inferred from the matching input type. The Python
surface does not accept explicit result types; the IR verifier requires both
result types to equal the input type.

The old required `result_type=` keyword is removed atomically with the ODS
change. Because there are no in-repository PTODSL VMULL call sites, a temporary
dual API is not required. The user guide must clearly document that the return
value is a tuple `(low, high)`.

## 8. Test plan

### 8.1 ODS and verifier tests

Add positive parse/verify coverage for:

- `64xi32`;
- `128xi32`;
- `256xi32`;
- `64xui32` or another unsigned shape;
- omitted pmode and explicit `pmode = "zero"`.

Add negative coverage for:

- lane count outside `{64, 128, 256}`;
- input element type other than `i32/ui32`, with explicit `si32` and `si64`
  cases so the current width-only/`isUnsigned()` behavior cannot pass;
- `si32` low/high result types even when both inputs use `si32`;
- input type or signedness mismatch;
- low/high type, signedness, or lane-count mismatch;
- mask lane-count mismatch;
- the old single `Lxi64` result form;
- unknown pmode strings.

### 8.2 Mask and layout tests

- Verify that `!pto.vmi.mask<Lxpred>` becomes `b32` for VMULL.
- Verify that `%a`, `%b`, `%mask`, `%low`, and `%high` receive the same layout.
- Verify that a conflicting consumer layout inserts an explicit
  materialization rather than silently changing one VMULL port.
- Cover ordinary deinterleaved factor-2 and factor-4 same-layout cases. Both
  are mandatory first-version tests and must check the
  exact arities above, corresponding `%a`/`%b`/`%mask` part alignment,
  low/high result grouping, and inactive or padding-lane mask alignment.
- Add preflight rejection tests for `block_deinterleaved=2/4`. These cases
  ensure the implementation does not accept arbitrary layout kinds merely
  because `getVMIPhysicalArity` can compute them.

### 8.3 VMI-to-VPTO tests

For contiguous layouts, check exact physical operation counts:

```text
64xi32  -> CHECK-COUNT-1: pto.vmull
128xi32 -> CHECK-COUNT-2: pto.vmull
256xi32 -> CHECK-COUNT-4: pto.vmull
```

The multi-chunk checks must also capture result grouping, not only operation
count. Uses of the logical `%low` value must receive all low parts, while uses
of `%high` must receive all high parts.

Add both signed and unsigned lowering coverage and verify `!pto.mask<b32>` on
every emitted physical operation.

### 8.4 PTODSL and end-to-end tests

- Verify `pto.vmi.vmull` returns a two-item tuple.
- Verify both result types are inferred for signed and unsigned inputs.
- Verify invalid input types fail with a clear IR verifier diagnostic.
- Compile at least one VMI-authored kernel through the complete VPTO pipeline
  and confirm no VMI op or type remains.

### 8.5 Numerical semantics tests

Compile-only coverage is not sufficient for VMULL. At least one executable
numerical backend must compare both result vectors against a scalar reference
oracle. The PTO ISA CPU simulator is preferred; if it does not support the
required VMULL form, equivalent A5 execution is mandatory for completion.

The reference oracle computes each lane independently:

```text
if mask[i]:
  if T == i32:
    product = signed_64(signed_32(a[i])) * signed_64(signed_32(b[i]))
  else:
    product = unsigned_64(unsigned_32(a[i])) *
              unsigned_64(unsigned_32(b[i]))
  expected_low[i]  = unsigned_32(product)
  expected_high[i] = unsigned_32(product >> 32)
else:
  expected_low[i]  = 0
  expected_high[i] = 0
```

Required signed boundary cases include:

```text
(-1) * 2                    -> low=0xfffffffe, high=0xffffffff
INT32_MIN * (-1)            -> low=0x80000000, high=0x00000000
INT32_MIN * 2               -> low=0x00000000, high=0xffffffff
INT32_MAX * INT32_MAX       -> low=0x00000001, high=0x3fffffff
```

Required unsigned boundary cases include:

```text
0xffffffff * 0xffffffff     -> low=0x00000001, high=0xfffffffe
0x80000000 * 2              -> low=0x00000000, high=0x00000001
```

The numerical suite must also include:

- logical sizes 64, 128, and 256;
- distinct per-chunk values so a chunk-ordering error is observable;
- sparse masks spanning chunk boundaries, including lanes 0, 63, 64, 127,
  128, and 255 when present;
- inactive input lanes containing non-zero sentinel values, with both output
  halves checked to be zero;
- ordinary deinterleaved factor-2 and factor-4 cases, including
  logical lanes that map to different physical parts.

Inactive lanes must remain observable. The test kernel uses a sparse mask for
VMULL, but writes `%low` and `%high` to their output buffers with either an
unmasked store or a separate all-true store mask. It must not reuse the sparse
VMULL mask for either store. The output buffers are initialized with non-zero
sentinels, every logical lane is written and read back, and inactive lanes are
compared with zero. Otherwise a masked store could hide incorrect non-zero
VMULL results in precisely the lanes this test is intended to validate.

These checks detect signed high-half errors, low/high swaps, incorrect flat
result grouping, physical part reordering, and failure to zero inactive lanes.

## 9. Compatibility and rollout

Changing one `Lxi64` result to two `Lxi32` results is an intentional breaking
change to an incomplete, currently non-lowerable VMI operation. All public
layers must change in one implementation pull request:

```text
ODS + verifier
mask/layout assignment
VMIToVPTO preflight + conversion
PTODSL
VMI ISA and PTODSL documentation
focused regression tests
```

Splitting the ODS change from the conversion pattern would temporarily create
another parseable but non-executable VMULL form and is therefore not
recommended.

## 10. Acceptance criteria

The implementation is complete when:

1. The only accepted VMI VMULL surface is pair-result `Lxi32/ui32`, with
   `L in {64, 128, 256}`; explicit `si32` and `si64` forms are rejected.
2. Mask granularity and all five layouts are assigned consistently.
3. Contiguous 64/128/256-lane inputs produce exactly 1/2/4 physical
   `pto.vmull` operations.
4. Low and high result parts are associated with the correct logical result.
5. Signed and unsigned forms select their existing physical backend forms.
6. Deinterleaved factor-2 and factor-4 pass mandatory arity, part-alignment,
   result-grouping, and mask-alignment tests, while `block_deinterleaved`
   receives a preflight diagnostic.
7. CPU simulator or A5 numerical validation confirms signed/unsigned high and
   low halves and zeroing for sparse inactive lanes observed through full-lane
   output stores.
8. The full pipeline contains no residual VMI op or VMI type.
9. ODS, verifier, lowering, PTODSL, user documentation, and tests land
   together in the follow-up implementation pull request.

## 11. Decisions requested from review

This proposal asks reviewers to confirm two API decisions before code is
implemented:

1. Use two logical `Lxi32/ui32` results instead of one logical `Lxi64/ui64`
   result.
2. Define predication as zeroing: inactive lanes produce 0.

Once those decisions are accepted, the implementation path is fully defined
by the contracts above.
