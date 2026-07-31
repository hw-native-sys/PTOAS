# VMI Lane-Stride Layout Generalization Implementation Plan

本文给出 `lane_stride` 泛化的实现路径。设计目标是把 lane-strided dense
layout 作为一等 layout fact 固化、传播、rematerialize 和 lower，而不是在
`vmi-to-vpto` 中识别单个 `64xf16 -> 64xf32` pattern。

## 1. Implementation Principles

1. `lane_stride` is a lane-map field.
2. Dense `lane_stride` does not change the VMI logical element type.
3. Group-slot lane-stride keeps the logical element carrier; any wider
   pack/unpack carrier is local to the lowering instruction.
4. Layout assignment decides layout before `vmi-to-vpto`.
5. `vmi-to-vpto` only lowers explicit assigned layout attrs.

Pre-existing baseline before this design:

```text
dense contiguous/deinterleaved layouts:
  did not carry lane_stride

regular VMI load/store:
  did not support dense lane_stride
  support contiguous and selected deinterleaved lowering/materialization paths

VPTO load/store:
  pto.vlds/pto.vsts have a dist string and the VPTO surface supports several
  distribution families, but there is no generic lane_stride operand

group-slot lane_stride:
  already existed and was used by selected group-store packed-byte lowering
```

Any dense lane-stride load/store support must enter explicitly by mapping a VMI
lane-stride layout to a specific supported VPTO dist family or materialization
sequence.  It must not be inferred in `vmi-to-vpto` from a one-off producer or
consumer pattern.

Current stage status:

| Area | Status | Notes |
|---|---|---|
| Dense layout attrs | Supported | Dense contiguous/deinterleaved and group-slot layouts carry `lane_stride`; all data parts keep the logical element carrier. |
| Direct compact load/store | Supported for selected phase-zero maps | LS=2 b8/b16/b32 through `UNPK_B8/B16/B32` and `PK_B16/B32/B64`; LS=4 b8 through `UNPK4` and `PK4_B32`. |
| Load/store layout folds | Supported with one-load/one-store preservation | `load -> ensure_layout(lane_stride)` rewrites the original load layout when all uses agree; `ensure_layout(lane_stride -> contiguous) -> store` lets the VMI store consume the lane-stride value. |
| Dense widening ext | Supported | `getPreferredCastLayoutFact` chooses the arity-reducing source `lane_stride=W` / result contiguous relation when it beats the natural deinterleaved result; otherwise it keeps the natural relation. |
| Dense narrowing trunc | Supported for dense natural paths | `getPreferredCastLayoutFact` uses the same arity rule in the inverse direction, so trunc keeps the natural deinterleaved-source / contiguous-result relation unless a compact relation actually reduces arity. |
| Masked compact store | Partially supported | Legal only when value and mask have the same lane map and the mask can be compacted for the selected store dist. |
| Masked trunc tail | Not optimized yet | Keep the existing legal path until mask lane-stride assignment/materialization is available. |
| Register fallback | Partially supported | Only same-physical-arity contiguous `<->` lane_stride paths with legal pack/unpack carriers.  Arity-changing fallback is not in scope for this stage. |
| Group broadcast load | Supported only through specific strategies | `group_broadcast_load` remains a VMI semantic; E2B is one strategy with exact shape/layout constraints. |

Remaining design/implementation work from this discussion is intentionally
limited to these areas:

| Area | Work to settle | Required proof before enabling |
|---|---|---|
| Cast assignment | Keep `getPreferredCastLayoutFact` as the single op-local preferred relation helper, but make it shape-aware: compute the natural relation, compute the compact lane-stride relation, and select compact only when physical arity improves. | `64xf16 -> 64xf32` chooses source `lane_stride=2` and result contiguous; `128/256xf16 -> f32` keep natural `deinterleaved=2`; dense trunc keeps the natural relation unless compact arity wins. |
| Masked store | Let `masked_store` request the same lane map for value and mask, or keep the existing legal path when the mask cannot be assigned/rematerialized into that lane map. | No path may lower a lane-stride value with a stale contiguous user mask; lowering must compact the assigned mask into the packed-store predicate. |
| Group broadcast load | Keep `group_broadcast_load` as a VMI logical operation and make E2B only one support/lowering strategy selected by shape, element width, stride, and assigned result layout. | A failed E2B match must mean "this lowering strategy is unavailable", not "the VMI op is invalid" unless no fallback strategy is registered. |

Known support boundaries that are not part of this discussion's remaining-work
queue:

```text
b32 contiguous <-> lane_stride register fallback through generic vpack/vunpack
generic scalar broadcast-load VMI semantic for BRC
dense lane-stride masked_load
arity-changing register fallback
LS=4 b16/b32 direct compact load/store
LS > 4 direct compact load/store
non-zero lane_offset / lane_phase
ordinary load cloning/rematerialization without safe-read proof
global cost search across conflicting consumer layouts
partial-chunk dense lane-stride direct memory beyond the current full-chunk gate
```

### 1.1 VPTO Dist Capability Boundary

VPTO already exposes fixed distribution families that can implement specific
layout-producing or layout-consuming memory operations:

```text
vlds:
  NORM
  BRC_B8/B16/B32
  US_B8/B16
  DS_B8/B16
  UNPK_B8/B16/B32
  BRC_BLK
  E2B_B16/B32
  UNPK4
  SPLT4CHN
  SPLT2CHN_B8/B16

vldsx2:
  BDINTLV
  DINTLV_B8/B16/B32

vsts:
  NORM_B8/B16/B32
  1PT_B8/B16/B32
  PK_B16/B32/B64
  PK4_B32
  MRG4CHN_B8
  MRG2CHN_B8/B16

vstsx2:
  INTLV_B8/B16/B32
```

These are not equivalent to an arbitrary dense `lane_stride` operand:

```text
DINTLV/INTLV:
  two-stream deinterleave/interleave memory operation
  maps naturally to VMI deinterleaved layouts, not to one sparse semantic stream

US/DS:
  fixed 2x upsample/downsample load families for b8/b16
  can serve selected lane-map producers when the semantic mapping matches exactly

UNPK/PK/PK4:
  fixed slot-pack/slot-unpack memory families
  directly express selected dense lane_stride layouts such as b16 LS=2 and
  b8 LS=4, but not arbitrary LS=N

BRC/E2B/BRC_BLK:
  fixed broadcast or group-expansion load families
  useful when logical broadcast plus assigned layout matches the family

MRG/SPLT:
  fixed channel merge/split families
  useful only for matching channel layouts
```

So VPTO has enough surface area to support selected dense lane-stride memory
optimizations, but VMI must model them as explicit support cases:

```text
VMI layout fact + op semantics + element width
  -> exact VPTO dist family
  or materialization/rematerialization sequence
  or unsupported
```

Concrete support matrix for dense phase-zero `lane_stride`:

| Dense lane_stride | Compact stream load -> dense LS | Single-scalar broadcast load -> dense LS | Dense LS -> compact stream store |
|---:|---|---|---|
| 2 | direct for b8/b16/b32 through `vlds UNPK_B8/B16/B32` | target dist exists as `vlds BRC_B8/B16/B32`; needs a separate single-scalar broadcast-load VMI semantic | direct for b8 through `vsts PK_B16`, b16 through `vsts PK_B32`, and b32 through `vsts PK_B64` |
| 4 | direct for b8 through `vlds UNPK4` | target dist exists as `vlds BRC_B8/B16/B32`; needs a separate single-scalar broadcast-load VMI semantic | direct for b8 through `vsts PK4_B32` |

| VMI memory semantic | Element width | VPTO op/dist | VMI result layout | Direct dense `lane_stride` support |
|---|---:|---|---|---|
| load one scalar and every logical lane uses it | b8 | `vlds BRC_B8` | any dense phase-zero lane map | target dist exists; needs a separate VMI scalar broadcast-load semantic |
| load one scalar and every logical lane uses it | b16 | `vlds BRC_B16` | any dense phase-zero lane map | target dist exists; needs a separate VMI scalar broadcast-load semantic |
| load one scalar and every logical lane uses it | b32 | `vlds BRC_B32` | any dense phase-zero lane map | target dist exists; needs a separate VMI scalar broadcast-load semantic |
| load compact stream `x[i]` into semantic lane `2*i` | b8 | `vlds UNPK_B8` | `contiguous, lane_stride=2` | yes |
| load compact stream `x[i]` into semantic lane `2*i` | b16 | `vlds UNPK_B16` | `contiguous, lane_stride=2` | yes |
| load compact stream `x[i]` into semantic lane `2*i` | b32 | `vlds UNPK_B32` | `contiguous, lane_stride=2` | yes |
| load compact stream `x[i]` into semantic lane `4*i` | b8 | `vlds UNPK4` | `contiguous, lane_stride=4` | yes |
| load compact stream `x[i]` into semantic lane `4*i` | b16/b32 | none | `contiguous, lane_stride=4` | no direct VPTO dist |
| load compact stream `x[i]` into semantic lane `K*i`, `K > 4` | any | none | `contiguous, lane_stride=K` | no direct VPTO dist |
| load memory `x[2*i]` into logical lane `i` | b8 | `vlds DS_B8` | contiguous | no; this is memory downsample |
| load memory `x[2*i]` into logical lane `i` | b16 | `vlds DS_B16` | contiguous | no; this is memory downsample |
| load alternating memory stream into even/odd logical streams | b8 | `vldsx2 DINTLV_B8` | two compact streams or deinterleaved=2 | no; not one sparse stream |
| load alternating memory stream into even/odd logical streams | b16 | `vldsx2 DINTLV_B16` | two compact streams or deinterleaved=2 | no; not one sparse stream |
| load alternating memory stream into even/odd logical streams | b32 | `vldsx2 DINTLV_B32` | two compact streams or deinterleaved=2 | no; not one sparse stream |
| store semantic lane `2*i` as compact memory `x[i]` | b8 | `vsts PK_B16` | source `contiguous, lane_stride=2` | yes |
| store semantic lane `2*i` as compact memory `x[i]` | b16 | `vsts PK_B32` | source `contiguous, lane_stride=2` | yes |
| store semantic lane `2*i` as compact memory `x[i]` | b32 | `vsts PK_B64` | source `contiguous, lane_stride=2` | yes |
| store semantic lane `4*i` as compact memory `x[i]` | b8 | `vsts PK4_B32` | source `contiguous, lane_stride=4` | yes |
| store semantic lane `4*i` as compact memory `x[i]` | b16/b32 | none | source `contiguous, lane_stride=4` | no direct VPTO dist |
| store semantic lane `K*i` as compact memory `x[i]`, `K > 4` | any | none | source `contiguous, lane_stride=K` | no direct VPTO dist |
| store two compact streams as alternating memory | b8 | `vstsx2 INTLV_B8` | two compact streams or deinterleaved=2 | no; not one sparse stream |
| store two compact streams as alternating memory | b16 | `vstsx2 INTLV_B16` | two compact streams or deinterleaved=2 | no; not one sparse stream |
| store two compact streams as alternating memory | b32 | `vstsx2 INTLV_B32` | two compact streams or deinterleaved=2 | no; not one sparse stream |

Masked compact stores have an extra legality rule.  The `vmi.masked_store`
predicate is a logical-lane predicate, while a VPTO packed store consumes a
predicate in the compacted store coordinate after the sparse lanes have been
packed.  Therefore a lane-stride value cannot be paired with an unrelated
contiguous mask and lowered directly to `PK`/`PK4`.

The direct masked-store path is legal only when all of these hold:

```text
value source layout == mask source layout
value/mask physical arity matches
mask granularity matches the logical value element width before compaction
target has a predicate compaction path for the packed-store dist
```

For example, an f16 value with `lane_stride=2` places logical lanes in even
physical lanes.  If a user mask remains contiguous, mask bit `i` still denotes
logical lane `i`, not physical lane `2*i`.  Passing that mask directly to
`vsts PK_B32` would gate the wrong compact positions for tail or sparse masks.
The current legal path requires the mask to carry the same lane map as the value
and then compacts it with predicate unpack operations before emitting the
packed store.  Ordinary unmasked `vmi.store` is different: lowering creates the
compact prefix predicate itself, so there is no user mask to reinterpret.

Until masked-store assignment can request and prove the same lane map for value
and mask, assignment must keep masked-tail narrowing on an existing legal path
instead of choosing a lane-stride trunc result solely because the store could
otherwise use `PK`.

Current-stage implementation:

```text
lib/PTO/Transforms/VMILayoutAssignment.cpp

VMIMaskedStoreOp keeps the existing conservative request:
  requestDataUse(value, contiguous)
  requestMaskUse(mask, contiguous, elementGranularity)

trunc assignment does not inspect masked_store users and does not preserve a
special masked-store guard.  It records the source/result relation returned by
getPreferredCastLayoutFact.  If that conflicts with a masked_store contiguous
request, normal assignment conflict handling inserts the required
ensure_layout.
```

Future lane-stride `masked_store` support must be added as an explicit
consumer-owned extension, not as a trunc special case.  The future dataflow must
prove that value and mask share the same lane map before a packed masked store
is legal:

```text
%n = vmi.trunc* %wide
     : source contiguous -> result contiguous, lane_stride = W

%m_ls = vmi.ensure_mask_layout %m
     : mask contiguous -> mask contiguous, lane_stride = W

vmi.masked_store %n, %dst[%off], %m_ls
```

That future extension would need the same local VMI proof before lowering can do
the mechanical predicate compaction:

```text
vmi-layout-fold:
  may fold ensure_layout(value) + ensure_mask_layout(mask) into masked_store
  only through canFoldContiguousMaskedStoreMaterialization

vmi-to-vpto:
  sees valueLayout == maskLayout
  calls createDenseLaneStrideStorePredicate
  emits LOWER punpack on the mask
  emits vsts PK_B16/PK_B32/PK4_B32 as selected by value element width/layout
```

Future negative tests should cover the fallback:

```text
fallback:
  mask cannot be assigned/materialized to the candidate lane_stride
  CHECK masked_store keeps contiguous value/mask request
  CHECK no PK/PK4 masked compact store is emitted with a stale contiguous mask
```

The remaining VPTO dist tokens are fixed non-lane-stride operations:

```text
UNPK_B8/B16/B32:
  compact load into one element per 16/32/64-bit slot, giving lane_stride=2 for
  b8/b16/b32 dense values

UNPK4:
  compact load into one b8 element per 32-bit slot, giving lane_stride=4 for b8

PK_B16/B32/B64 and PK4_B32:
  compact store from one active low element per 16/32/64-bit slot.  PK_B32 is
  exactly the direct compact store for a b16 value with lane_stride=2, and
  PK4_B32 is exactly the direct compact store for a b8 value with lane_stride=4

MRG4CHN_B8 and MRG2CHN_B8/B16:
  fixed channel merge stores, not generic lane_stride stores

SPLT4CHN and SPLT2CHN_B8/B16:
  fixed channel split loads, not generic lane_stride loads

BRC_BLK and E2B_B16/B32:
  usable only after their exact block/group expansion semantic is modeled as a
  VMI broadcast producer; do not count them as generic dense lane_stride load
```

### 1.2 Contiguous/Lane-Stride Fallback Materialization

Direct load/store support is preferred.  When a value already lives in VPTO
registers and a consumer requires the other layout, `ensure_layout` provides the
fallback conversion between contiguous and dense phase-zero `lane_stride`.

For `contiguous -> lane_stride`, use register unpack placement when the VPTO
surface supports the required carrier type:

```text
LS=2:
  use vzunpack/vsunpack-style widening placement
  b8  contiguous -> b16 slots with low b8 semantic
  b16 contiguous -> b32 slots with low b16 semantic
  b32 contiguous -> b64 slots with low b32 semantic

LS=4:
  for b8, apply two LS=2 unpack placements:
  b8 contiguous -> b16 slots -> b32 slots with low b8 semantic
```

For `lane_stride -> contiguous`, use register pack when the VPTO surface
supports the required carrier type:

```text
LS=2:
  use vpack-style narrowing placement
  low b8  from each b16 slot -> b8 contiguous
  low b16 from each b32 slot -> b16 contiguous
  low b32 from each b64 slot -> b32 contiguous

LS=4:
  for b8, apply two LS=2 pack placements:
  low b8 from each b32 slot -> b16 slots -> b8 contiguous
```

This is the register-side counterpart of `UNPK`/`PK` memory distributions.  Do
not use `vintlv`/`vdintlv` as the primary fallback for dense `lane_stride`; those
belong to two-stream interleave/deinterleave layouts.

Current checked-in VPTO coverage:

```text
register pack:
  vpack supports integer 32 -> u16 and integer 16 -> u8
  so b16 LS=2 -> contiguous and b8 LS=2/4 -> contiguous are directly covered
  when the VMI source/result physical arity is the same
  b32 LS=2 -> contiguous needs 64 -> 32 pack support or another materializer

register unpack:
  vsunpack/vzunpack support integer widening by 2x
  so integer b8/b16 contiguous -> LS=2 and b8 contiguous -> LS=4 are covered
  when the VMI source/result physical arity is the same

floating-point lane_stride:
  b8/b16 FloatType values use bit-preserving vbitcast to unsigned integer
  carriers around the same pack/unpack sequence; non-FloatType low precision
  types need a VPTO vbitcast contract before enabling this fallback

arity-changing lane_stride materialization:
  contiguous -> lane_stride can be expressed as multiple unpack parts, and
  lane_stride -> contiguous needs an explicit multi-part merge/pack plan.
  The current stage rejects those helpers instead of guessing a cross
  physical-chunk materialization.
```

This fallback is a materialization cost, not a layout preference.  Assignment may
insert the `ensure_layout`; later folding/rematerialization should remove it when
the producer or consumer has direct support:

```text
load -> ensure_layout(lane_stride)
  fold into a VMI load whose result has the requested lane_stride; vmi-to-vpto
  later lowers that load to UNPK when the element width and stride match.
  BRC remains the target dist for a separate scalar broadcast-load VMI semantic.

ensure_layout(lane_stride) -> store
  fold into a VMI store that directly consumes the lane_stride value; vmi-to-vpto
  later lowers that store to PK/PK4 when the element width and stride match

ordinary producer -> ensure_layout(contiguous <-> lane_stride)
  lower to register pack/unpack materialization when the element width is
  supported
```

### 1.3 Pass Responsibilities

Dense `lane_stride` should use the existing helper-driven layout pipeline.  Do
not add a separate global candidate solver for the current stage.

```text
pto-validate-vmi-ir:
  verify surface syntax before assignment
  reject malformed dense lane_stride attrs once the parser accepts them
  keep lane_offset unavailable in the public attr

vmi-layout-assignment:
  assign explicit dense layouts, including lane_stride, on VMI value types
  use op support queries to choose local cast relations:
    widening compares natural deinterleaved result arity with compact
      contiguous result arity; when compact wins, request source lane_stride=W
      and set result contiguous
    narrowing supports the inverse relation; when arity or a supported consumer
      request chooses a strided result, request source contiguous and set result
      lane_stride=W
  keep unsupported or conflicting uses legal by inserting ensure_layout
  serialize all decisions as type attrs or helper ops
  do not clone producers, fold memory ops, or solve a global cost problem

canonicalize/cse:
  remove dead helpers and merge identical rematerialized values when normal MLIR
  canonicalization can prove equivalence
  no lane_stride-specific decision logic

vmi-layout-rematerialize:
  consume producer -> ensure_layout shapes
  clone/rematerialize cheap producers directly in the requested lane_stride
  layout when the producer support query says it can create that layout
  examples: scalar broadcast, splat constants, iota, layout-transparent chains,
  widening ext, and supported mask producers
  do not rematerialize ordinary loads unless the load form has an explicit
  safe-read proof and direct UNPK lowering support

vmi-layout-fold:
  consume helper-adjacent producer/consumer shapes
  fold ensure_layout(lane_stride) feeding store into a VMI store that directly
  consumes the lane_stride value when the support table has a direct compact
  store lowering; this is still a VMI store, not a VPTO PK op
  fold load -> ensure_layout when the load can directly produce the requested
  lane map with UNPK and the rewrite preserves one load at the original
  program point
  fold identity lane-map conversions
  leave unsupported conversions as explicit ensure_layout for validation or
  vmi-to-vpto materialization

vmi-layout-sink-materialization:
  move ensure_layout across pure layout-transparent ops when all operands/results
  can keep one identical dense lane map
  reduce duplicated contiguous <-> lane_stride materializations
  do not sink through cast, load, store, reduce, group_broadcast, or control flow

pto-validate-vmi-layout-ir:
  verify every dense value has a supported layout attr
  verify ensure_layout has a supported materialization path:
    identity
    contiguous <-> lane_stride through register pack/unpack when supported
    existing contiguous <-> deinterleaved relations
  verify direct layout-aware load/store choices:
    LS=2 b8/b16/b32 through UNPK/PK
    LS=4 b8 through UNPK4/PK4
    BRC only after a scalar broadcast-load VMI semantic is modeled
  reject unsupported direct cases such as LS=4 b16/b32 compact load/store

vmi-to-vpto:
  lower only from assigned type attrs, helper ops, and op attributes
  emit direct vlds/vsts dist for UNPK/PK-supported memory cases
  lower surviving contiguous <-> lane_stride ensure_layout through register
  pack/unpack materialization when the VPTO verifier supports the carrier path
  lower widening/narrowing casts according to the assigned source/result
  lane_stride relation and concrete vcvt part
  emit diagnostics instead of inventing hidden layout conversions
```

Implementation impact by pass/component:

| Component or pass | Lane-stride implementation work |
|---|---|
| `VMILayoutAttr` ODS/C++ helpers | Yes. Add dense `laneStride` storage, parse/print, verifier, equality, lane-map helpers, and keep it separate from group-slot carrier packing. |
| VMI type physicalization helpers | Yes. Compute dense physical arity from `laneStride`; expose carrier-slot lowering helpers for pack/unpack paths without changing the VMI logical element type. |
| `VMILayoutSupport` / target capability helpers | Yes. Add support queries for dense `lane_stride` layouts, cast layout relations, direct UNPK/PK memory support, and contiguous `<->` lane-stride materialization support. BRC remains target capability for a separate scalar broadcast-load semantic. |
| `pto-validate-vmi-ir` | No lane-stride-specific pass algorithm. It relies on attr/op verifier updates; keep the existing surface-IR validation role. |
| `vmi-layout-assignment` | Yes. Assign dense lane-stride layouts when support queries choose them; insert `ensure_layout` for incompatible uses; serialize all decisions in types/helpers. |
| `canonicalize/cse` between VMI passes | No implementation. It remains ordinary cleanup for dead helpers and identical rematerialized producers. |
| `vmi-layout-rematerialize` | Yes. Teach producer rematerialization to create requested dense lane-stride layouts for cheap/safe producers. Do not add ordinary load remat without safe-read proof. |
| `vmi-layout-fold` | Yes. Fold `ensure_layout` into layout-aware VMI consumers, especially stores that can consume lane_stride and later lower to `PK/PK4`; fold `load -> ensure_layout` into a direct layout-aware load when it can preserve one load at the original program point; fold identity lane-map conversions. |
| `vmi-layout-sink-materialization` | Minimal generic update. It should compare dense layout keys including `laneStride` and reuse existing layout-transparent sinking; do not add cast/load/store/reduce-specific lane-stride patterns here. |
| `vmi-legalize-arith-select` | No implementation. Lane stride does not change scalar-condition select legalization. |
| `pto-validate-vmi-layout-ir` | Yes. Reject unsupported assigned layouts/helpers before lowering, including unsupported LS=4 b16/b32 compact load/store and unsupported register pack/unpack materializations. |
| `vmi-to-vpto` | Yes. Lower assigned dense lane-stride layouts, direct `UNPK/PK` memory cases, register pack/unpack `ensure_layout`, and lane-stride-aware ext/trunc lowering. |
| VPTO op verifier/emitter | Only if needed by the selected support matrix. Existing `vlds/vsts` dist tokens are already present; extending register fallback to b32 or floating-point carriers requires verifier/emitter support for the corresponding pack/unpack or bitcast form. |
| Lower VPTO/backend passes after `vmi-to-vpto` | No lane-stride-specific implementation. They see ordinary VPTO ops and existing dist tokens. |

Any pass not listed above should not implement lane-stride-specific logic in the
current stage.  New behavior must enter through the explicit layout attr,
support queries, helper ops, validation, or `vmi-to-vpto` lowering.

Current-stage component checklist:

This checklist records the components that participate in the current-stage
lane-stride implementation.  It is not the remaining-work queue; remaining work
is limited to the masked-store and group-broadcast-load items above.

```text
include/PTO/IR/VMIAttrs.td
lib/PTO/IR/VMI.cpp
  add laneStride storage for dense contiguous/deinterleaved layouts
  keep group-slot laneStride parse/print compatibility
  add getContiguous(ctx, laneStride) and getDeinterleaved(..., laneStride)
  split helpers into isDenseLaneStrided(), isGroupSlotLaneStrided(),
  getLaneStride(), and exact dense lane-map equality helpers
  update attr verifier so laneStride > 0 and lane_offset is not accepted

lib/PTO/IR/VMI.cpp
lib/PTO/Transforms/VMIToVPTO.cpp
  replace the current "hasLaneStride implies unsigned carrier widening" helper
  with logical-element physicalization shared by dense and group-slot values;
  keep wider carrier types local to selected pack/unpack and memory
  instructions

include/PTO/Transforms/VMILayoutSupport.h
lib/PTO/Transforms/VMILayoutSupport.cpp
  extend dense store layout facts with lane_stride=2/4 cases
  extend VMILayoutMaterializationSupportKind with:
    ContiguousToLaneStrideViaUnpack
    LaneStrideToContiguousViaPack
    LaneStrideToLaneStrideViaContiguous, only if needed
  update getPreferredCastLayoutFact:
    keep an internal baseline natural relation for dense widening/narrowing
    compute the compact lane-stride relation from the same conversion ratio
    select compact only when source/result physical arities match and the
    relevant arity is strictly smaller than the baseline relation
    use the returned source/result layouts for both ext and trunc assignment
  update getWidenSourceLayoutForResultLayout for dense lane_stride result/source
  update getStoreLayoutFact and canFoldContiguousStoreMaterialization for
    LS=2 b8/b16/b32 -> PK_B16/B32/B64
    LS=4 b8 -> PK4_B32
  update canMaterializeDataLayout for contiguous <-> dense lane_stride through
  register pack/unpack when the element/carrier path is supported

lib/PTO/Transforms/VMILayoutAssignment.cpp
  teach natural/preferred layout collection to accept dense lane_stride facts
  from VMILayoutSupport
  keep conflict handling unchanged: insert ensure_layout at mismatched uses
  do not add producer cloning, memory folding, or global cost selection here

lib/PTO/Transforms/VMILayoutRematerialize.cpp
  allow cheap producers to be cloned with dense lane_stride result types when
  VMILayoutSupport says the producer can directly create that lane map
  keep ordinary load/group_load/masked_load cloning blocked until a safe-read
  proof is added for the specific rematerialized memory operation

lib/PTO/Transforms/VMILayoutFold.cpp
  add producer-side fold for load -> ensure_layout:
    replace the load result layout with the ensure target layout when the load
    has no other incompatible uses and VMILayoutSupport has direct UNPK
    support
    erase the ensure_layout and keep a single load at the original program point
    do not clone ordinary loads in this fold
  add fold for ensure_layout(lane_stride -> contiguous) feeding pto.vmi.store or
  pto.vmi.masked_store into a VMI store that consumes the lane_stride source
  directly; this pass does not emit or model VPTO PK. VMIToVPTO later selects
  the corresponding PK/PK4 store lowering from the assigned VMI store contract
  masked_store direct fold additionally requires the mask to carry the same
  dense lane_stride layout and a compactable element-width granularity:
    LS=2 b8/b16 and LS=4 b8 are supported through LOWER punpack mask compaction
    LS=2 b32 is left as explicit materialization until b32 lane-stride mask
    compaction is specified and implemented
    a contiguous user mask is not enough, even if the value layout can be
    compact-stored; assignment/rematerialization must first derive the same
    lane map for the mask
  fold exact dense lane-map identity helpers
  do not fold unsupported LS=4 b16/b32 cases

lib/PTO/Transforms/VMILayoutSinkMaterialization.cpp
  include laneStride in dense layout equality/support checks
  reuse existing layout-transparent sinking logic
  do not add lane-stride-specific sinking through casts or memory ops

lib/PTO/Transforms/PTOValidateVMIIR.cpp
  no new lane-stride algorithm
  validation changes should come from attr/op verifier and VMILayoutSupport
  diagnostics at the layout gate

lib/PTO/Transforms/VMIToVPTO.cpp
  update OneToN physical type conversion for dense laneStride and carrier slots
  lower direct compact loads:
    LS=2 b8/b16/b32 -> vlds UNPK_B8/B16/B32
    LS=4 b8 -> vlds UNPK4
  lower direct compact stores:
    LS=2 b8/b16/b32 -> vsts PK_B16/B32/B64
    LS=4 b8 -> vsts PK4_B32
  lower direct compact masked_stores:
    LS=2 b8/b16 -> LOWER punpack mask compaction + vsts PK_B16/B32
    LS=4 b8 -> two LOWER punpack steps + vsts PK4_B32
    LS=2 b32 -> no direct masked compact store until b32 lane-stride mask
    compaction is specified and implemented
  lower surviving ensure_layout contiguous <-> lane_stride through vpack and
  vsunpack/vzunpack when the carrier path is legal
  lower lane-stride-aware ext by selecting the concrete vcvt part from
  the assigned source/result relation
  lower lane-stride-aware trunc by selecting the concrete vcvt part from
  the assigned source/result relation

lib/PTO/IR/VPTO.cpp
lib/PTO/Transforms/VPTOLLVMEmitter.cpp
lib/PTO/Transforms/VPTOCANN900LLVMEmitter.cpp
  no change for existing vlds/vsts dist tokens
  extend vpack/vsunpack/vzunpack verifier/emitter only if the first implemented
  fallback needs currently unsupported b64->b32 or floating-point carrier paths

test/lit/vmi_new
  add parser/verifier tests for dense laneStride attrs
  add assignment tests for ext lane-stride facts
  add fold/remat/sink tests for helper-driven rewrites
  add vmi-to-vpto checks for UNPK/PK and vpack/unpack fallback
  add negative tests for unsupported LS=4 b16/b32 compact load/store
```

Load/`ensure_layout` fold algorithm:

```text
input shape:
  %x0 = pto.vmi.load ... : !vmi.vreg<NxT, contiguous>
  %x1 = pto.vmi.ensure_layout %x0
      : !vmi.vreg<NxT, contiguous>
     -> !vmi.vreg<NxT, contiguous, lane_stride = LS>

preconditions:
  the load result has no other use that requires the old layout
  the load semantics are a compact logical stream
  VMILayoutSupport says the target layout has a direct load lowering:
    compact stream:
      LS=2 b8/b16/b32 -> UNPK_B8/B16/B32
      LS=4 b8 -> UNPK4
  masks/passthroughs, if present, already have compatible assigned layouts

rewrite:
  replace the original load op in place, or create the replacement load at the
  same insertion point and erase the old load
  the replacement load result type is the ensure target type
  all ensure users use the replacement load result
  erase the ensure_layout

output shape:
  %x = pto.vmi.load ... : !vmi.vreg<NxT, contiguous, lane_stride = LS>

lowering:
  vmi-to-vpto emits the corresponding vlds UNPK dist
```

This fold changes the assigned result layout of the existing load; it does not
clone the load at the helper use-site.  If the original load has both contiguous
and lane-stride consumers, the fold must leave the helper in place unless a
separate rematerialization step has a safe-read proof to clone the load.

### 1.4 Scenario Ownership

Each optimization scenario has exactly one owning pass.  Other passes may verify
or lower the resulting explicit IR, but should not solve the same rewrite again.

| Scenario | Example shape | Owning pass | Non-owners |
|---|---|---|---|
| Assign a layout request | `ext -> store` where store wants contiguous | `vmi-layout-assignment` inserts explicit layouts/helpers | Assignment does not clone, fold, or lower |
| Direct load produces requested lane map | `load(contiguous) -> ensure_layout(lane_stride=2)` | `vmi-layout-fold` rewrites the original load result layout when UNPK support exists | Remat must not clone this load without safe-read proof |
| Direct store consumes lane map | `ensure_layout(lane_stride -> contiguous) -> store` | `vmi-layout-fold` rewrites the VMI store to consume the lane_stride source directly when direct compact-store support exists | `vmi-to-vpto` emits the actual `vsts PK/PK4` |
| Cheap producer can produce target layout | `broadcast -> ensure_layout(lane_stride=2)` | `vmi-layout-rematerialize` rebuilds broadcast with lane-stride result | Fold does not rebuild arbitrary producers |
| Cast chooses arity-reducing relation | `64xf16 -> 64xf32` or a supported narrowing with smaller strided result | `vmi-layout-assignment` chooses the cast source/result layout relation | Remat only handles later use-site requests; fold only handles adjacent load/store helpers |
| Cast can move materialization to cheap source | `ext/trunc -> ensure_layout(requested layout)` with source broadcast/load-fold case | `vmi-layout-rematerialize` rebuilds the cast with the requested relation | Assignment may already choose the self-preferred relation; fold only handles the load/store subcase |
| Layout-transparent op has ensured operands | `ensure(a), ensure(b) -> add` | `vmi-layout-sink-materialization` sinks matching helpers to the result | Remat handles the opposite shape `add -> ensure` |
| Surviving supported helper | `ensure_layout(contiguous <-> lane_stride)` after optimizations | `vmi-to-vpto` lowers to register pack/unpack | Earlier passes are allowed to leave it explicit |
| Unsupported helper or layout | `lane_stride=4 b16 compact store` | `pto-validate-vmi-layout-ir` rejects before lowering | `vmi-to-vpto` should not invent a repair |
| Multi-consumer value with incompatible layouts | one load feeds contiguous user and lane-stride user | baseline keeps helper; optional remat only with safe-read proof | Fold must not silently duplicate memory effects |

Examples:

```text
load fold, owned by vmi-layout-fold:
  before:
    %x0 = pto.vmi.load ... : contiguous
    %x1 = pto.vmi.ensure_layout %x0 : contiguous -> lane_stride=2
  after:
    %x1 = pto.vmi.load ... : lane_stride=2
  vmi-to-vpto:
    %x1 = pto.vlds ... {dist = "UNPK_B8/B16/B32 or UNPK4"}

store fold, owned by vmi-layout-fold:
  before:
    %x_c = pto.vmi.ensure_layout %x_ls : lane_stride=2 -> contiguous
    pto.vmi.store %x_c, %dst
  after:
    pto.vmi.store %x_ls, %dst   // VMI store consumes lane_stride source
  vmi-to-vpto:
    pto.vsts %x_ls, %dst {dist = "PK_B16/B32/B64 or PK4_B32"}

broadcast remat, owned by vmi-layout-rematerialize:
  before:
    %b0 = pto.vmi.broadcast %s : contiguous
    %b1 = pto.vmi.ensure_layout %b0 : contiguous -> lane_stride=2
  after:
    %b1 = pto.vmi.broadcast %s : lane_stride=2

elementwise sink, owned by vmi-layout-sink-materialization:
  before:
    %a1 = ensure_layout %a0 -> lane_stride=2
    %b1 = ensure_layout %b0 -> lane_stride=2
    %c1 = pto.vmi.addf %a1, %b1
  after:
    %c0 = pto.vmi.addf %a0, %b0
    %c1 = ensure_layout %c0 -> lane_stride=2
```

## 2. IR Attribute Changes

### 2.1 Extend `VMILayoutAttr`

The storage keeps a legacy `blockElems` field internally, but public layout
semantics split lane stride and block distribution into separate fields/kinds:

```c++
kind
factor
blockElems
slots
laneStride
```

Meaning by kind:

```text
contiguous:
  factor = 1
  blockElems = 1
  slots = 0
  laneStride >= 1

deinterleaved:
  factor = F
  blockElems = 1
  slots = 0
  laneStride >= 1

block_deinterleaved:
  factor = F
  blockElems = 1 as an internal storage invariant
  effective block width = 32B / sizeof(carrier element)
  slots = 0
  laneStride = 1

num_groups:
  factor = G
  blockElems = 1
  slots = K
  laneStride >= 1
```

Do not add a public `laneOffset` field in the current stage.  The targeted
optimization only needs phase-zero strided dense layouts.  A future
phase field is justified only when there is a concrete VMI value whose logical
lane map is intentionally non-zero-phase, for example a zero-copy
deinterleave/extract view that keeps the odd lanes in place or a narrowing
conversion whose consumer explicitly requires an odd-lane result.

Recommended helpers:

```c++
bool isDense() const;
bool hasDenseLaneStride() const;
bool hasGroupSlotLaneStride() const;
int64_t getLaneStride() const;
VMILayoutAttr withLaneStride(int64_t stride) const;
```

Constructor surface:

```c++
getContiguous(ctx)
getDeinterleaved(ctx, factor, laneStride = 1)
getBlockDeinterleaved(ctx, factor)
getGroupSlots(ctx, numGroups, slots = 0, laneStride = 1)
```

### 2.2 Parser And Printer

Accepted dense spellings:

```text
#pto.vmi.layout<contiguous>
#pto.vmi.layout<contiguous, lane_stride = 2>

#pto.vmi.layout<deinterleaved = 2>
#pto.vmi.layout<deinterleaved = 2, lane_stride = 2>
#pto.vmi.layout<block_deinterleaved = 2>
```

Existing group-slot spellings remain valid:

```text
#pto.vmi.layout<num_groups = 8, slots = 8>
#pto.vmi.layout<num_groups = 8, slots = 8, lane_stride = 4>
```

Printing omits defaults:

```text
lane_stride = 1 is omitted
```

### 2.3 Verifier

Verifier rules:

```text
all layouts:
  laneStride > 0

contiguous:
  factor == 1
  blockElems == 1
  slots == 0

deinterleaved:
  factor in supported dense factors
  blockElems == 1
  slots == 0

block_deinterleaved:
  factor in supported dense factors
  blockElems == 1 as an internal storage invariant
  slots == 0
  laneStride == 1

num_groups:
  factor > 0
  slots >= 0
  blockElems == 1
```

The verifier should not require every strided layout to fit one VPTO register.
Fit depends on the VMI type shape and element type, so it belongs in type
physicalization and op support checks.

## 3. Physicalization Helpers

### 3.1 Separate Element Carrier From Lane Map

Replace the current shared helper shape:

```c++
getVMIPhysicalElementType(type)
```

with two concepts:

```c++
getVMILogicalStorageElementType(type)
getVMIPhysicalCarrierElementType(type, loweringKind)
```

Dense lane-strided values keep the VMI logical element type.  The lowering may
represent the same lane map either as logical-element lanes or as wider carrier
slots when the selected VPTO instruction is a pack/unpack family.

Logical-element lane representation:

```text
!vmi.vreg<Nxf16, contiguous, lane_stride=2>
  -> !pto.vreg<128xf16> physical register
```

Carrier-slot representation for pack/unpack lowering:

```text
!vmi.vreg<Nxui16, contiguous, lane_stride=2>
  -> low ui16 in each ui32 slot for vpack/PK_B32-style lowering

!vmi.vreg<Nxui8, contiguous, lane_stride=4>
  -> low ui8 in each ui32 slot for PK4_B32-style lowering
```

Group-slot packed stores also request a wider carrier in the specific lowering
path:

```text
!vmi.vreg<Gxui8, num_groups=G, slots=8, lane_stride=4>
  group_store -> b32 carrier + PK4_B32
```

Do not let dense `hasLaneStride()` imply unsigned-integer carrier widening
globally.  Carrier widening is a property of a selected materialization or
load/store lowering, not of the VMI logical type itself.

### 3.2 Physical Arity

Add a dense lane-map helper:

```c++
struct DenseLaneMap {
  bool blockDeinterleaved;
  int64_t deinterleaveFactor;
  int64_t blockElems; // 1, or a 32B width derived from the carrier type.
  int64_t laneStride;
};

int64_t getPhysicalLaneForDenseLogicalLane(DenseLaneMap map,
                                           int64_t logicalLane);
```

For a VMI vreg type:

```text
lanesPerVPTO = getVPTOPhysicalLanes(elementType)
lanesInDensePart = ceil(N / F) with block-aware distribution
requiredLanes = O + (lanesInDensePart - 1) * LS + 1
registersPerDensePart = ceil(requiredLanes / lanesPerVPTO)
physicalArity = F * registersPerDensePart
```

For the current stage, require full block divisibility for dense
deinterleaved strided layouts, matching existing direct lowering restrictions:

```text
N % (F * B) == 0
```

Relaxing tail handling is outside the current stage and should be enabled only
with an explicit materialization/lowering proof.

## 4. Layout Support Interface

Extend support queries to include dense strided layouts:

```text
supportsResultLayout(op, resultIndex, layout)
supportsOperandLayout(op, operandIndex, layout)
supportsLayoutRelation(op, operandLayouts, resultLayouts)
```

The important change is relation support.  Some ops are not independently
described by "operand supports layout X" and "result supports layout Y"; they
support specific pairs.

Examples:

```text
elementwise:
  all dense operands/results must use identical dense layout key

extf/extui/extsi:
  source/result layouts must satisfy a widening relation.  Assignment chooses
  between the natural deinterleaved relation and the compact-result
  lane-stride-source relation by comparing physical arity, not by matching a
  concrete lane count such as 64.

truncf/trunci:
  source/result layouts must satisfy a narrowing relation.  Assignment uses the
  inverse relation conservatively: keep the natural deinterleaved-source to
  contiguous-result relation unless arity or a supported consumer request
  selects a strided result relation.  Masked-store consumers may only use the
  strided result relation when the value and mask can be assigned/materialized
  to the same lane map.

broadcast/group_broadcast:
  result may use a dense layout only when the materialization lowering has an
  explicit support case for that lane map

load:
  default result contiguous
  producer rematerialization may create selected strided layouts if a direct
  load/mask sequence can produce that lane map

store:
  memory effect is contiguous unless the op is an explicit logical interleave
  store; a strided source requires store lowering support or ensure_layout
```

Assignment should still insert `ensure_layout` for incompatible use-local
requests. Rematerialization/fold can later remove it.

### 4.1 Cast Relation Helper Shape

Keep `getPreferredCastLayoutFact` as the assignment entry point for dense
widening and narrowing casts, but make the helper return the actual preferred
source/result relation for the current shape.  Internally it first builds the
natural relation:

```text
widen:
  source contiguous
  result deinterleaved=W

narrow:
  source deinterleaved=W
  result contiguous
```

Then it computes the compact relation:

```text
widen:
  source contiguous, lane_stride=W
  result contiguous

narrow:
  source contiguous
  result contiguous, lane_stride=W
```

The compact relation is selected only when its source/result physical arities
match and it strictly reduces the relevant baseline arity:

```text
widen:
  physical_arity(compact result) < physical_arity(natural result)

narrow:
  physical_arity(compact source) < physical_arity(natural source)
```

If the compact relation does not win, the helper returns the natural relation.
`vmi-layout-assignment` calls this helper for `extf/extui/extsi` and
`truncf/trunci`, requests the returned source layout, and records the returned
result layout.

The support query must validate the returned pair before assignment commits it:

```text
supportsExtRelation(sourceTypeWithLayout, resultTypeWithLayout)
supportsTruncRelation(sourceTypeWithLayout, resultTypeWithLayout)
```

The validation step is a legality check, not a second optimizer.

### 4.2 Current Framework Fit

The existing assignment pass already has use-site requests. For example,
`pto.vmi.store` requests a contiguous source operand, and assignment can insert
`ensure_layout` when the stored value is assigned another layout.

The dense-stride `ext` optimization should keep the same model: the cast op is
the layout-entry point and stores one preferred source/result relation.  The old
preferred relation was:

```text
extf:
  request source contiguous
  set result deinterleaved=W
```

The current stage keeps the existing single-preference framework and lets
`ext` choose one fact for the current op:

```text
baseline fact:
  source contiguous
  result deinterleaved=W

lane-stride fact:
  source lane_stride=W
  result contiguous
```

The `ext` support query chooses between these facts from op-local information:

```text
conversion ratio W
target support for one selected hardware conversion part
physical arity of the natural result layout
physical arity of the compact contiguous result layout
requested result layout when a consumer materialization/remat path provides one
```

It does not inspect the defining source producer.  If compact result arity is
strictly smaller than natural result arity and the target supports the
single-part relation, it selects the lane-stride fact.  If it selects the
lane-stride fact and the source is not already in that layout, assignment
inserts an explicit source `ensure_layout`.  Later passes either discharge that
helper by rematerializing/folding a concrete producer, lower it with a
registered pack/unpack materializer, or let validation reject the unsupported
relation.

## 5. Widening Conversion Lowering

Let:

```text
W = result element storage bits / source element storage bits
```

For a dense source layout:

```text
source lane_stride = LS
```

Single-part lowering is legal when:

```text
LS % W == 0
```

Then:

```text
hardware part        = 0
result lane_stride   = LS / W
```

The current stage only emits the zero-phase single-part conversion.
`vcvt ODD` remains necessary for full packed contiguous conversion, but that is
handled by the existing multi-part relation:

```text
source contiguous, lane_stride=1
result deinterleaved=W
```

Do not add a phase field merely to name that existing ODD instruction.  Add a
phase field only when an assigned VMI layout needs to represent a concrete
zero-copy value/view already resident in odd/non-zero-phase lanes.

The support query for the conversion should accept the pair only when the
requested result layout equals this computed result lane map, including
deinterleave/block fields.

The support query should expose helpers for both legal facts, but assignment
chooses one immediately:

```text
baseline fact:
  source contiguous
  result deinterleaved=W
  natural result arity = physical_arity(result deinterleaved=W)

lane-stride fact:
  result contiguous
  source same dense shape with lane_stride = W
  compact result arity = physical_arity(result contiguous)
```

Assignment uses this deterministic rule:

```text
if compact result arity < natural result arity
   and the lane-stride fact is supported:
  choose lane-stride fact
else:
  choose baseline fact
```

For example, for `f16 -> f32`, the `extf` op chooses
`source lane_stride=2 -> result contiguous` for `64xf32`, because the compact
result has one physical chunk while the natural `deinterleaved=2` result has two
physical chunks.  For `128xf32` and `256xf32`, both layouts have the same result
arity, so assignment chooses the natural `deinterleaved=2` result.  The source
producer is handled by the explicit source `ensure_layout` and later
fold/rematerialization; it is not part of the cast support query.

Current contiguous widening remains a separate legal relation:

```text
source dense contiguous, lane_stride=1
result deinterleaved=W, lane_stride=1
```

Implementation steps:

1. Factor conversion ratio calculation by storage bit width.
2. Add helper that computes the natural result layout and its physical arity.
3. Add helper that computes the compact result layout, required source
   lane-stride layout, and compact result physical arity.
4. Teach `VMIToVPTO` conversion lowering to emit only the selected hardware
   part when the relation is single-part.
5. Keep existing multi-part lowering for contiguous-to-deinterleaved cases.
6. Add diagnostics when an assigned conversion layout pair has no lowering.

Hardware part names should be abstracted:

```text
W=2:
  part 0 -> EVEN
  part 1 -> ODD

W=4:
  part 0..3 -> target-specific conversion part names or the existing sequence
```

Do not special-case f16/f32 in the matcher. The type only determines `W` and
the concrete VPTO conversion opcode.

## 6. Narrowing Conversion Lowering

Let:

```text
W = source element storage bits / result element storage bits
```

For a single-part narrowing relation:

```text
result lane_stride = source lane_stride * W
hardwarePart = 0 for the current stage
```

The narrowing assignment relation is the inverse of widening, but it must not
blindly choose a lane-stride result.  Build two facts:

```text
baseline fact:
  source deinterleaved=W
  result contiguous
  natural result arity = physical_arity(result contiguous)

lane-stride fact:
  source contiguous
  result contiguous, lane_stride=W
  strided result arity = physical_arity(result lane_stride=W)
```

Then choose a strided result only when it is justified:

```text
if strided result arity < natural result arity
   and the lane-stride fact is supported:
  choose lane-stride fact
else if a consumer/requested result layout is the strided result
        and the lane-stride fact is supported:
  choose or rematerialize lane-stride fact
else:
  choose baseline fact
```

This keeps trunc symmetric with ext while avoiding the earlier mistake of
producing lane_stride solely because the operation is a narrowing cast.  A
consumer may still request or preserve a strided result.  For example, an
ordinary store with direct `PK` support can consume a supported lane-stride
result, and rematerialization/fold may keep that relation.  A masked store may
do so only when the mask can be assigned/materialized to the same lane map.

Implementation steps:

1. Share ratio, dense-factor, lane-map, and physical-arity helpers with
   widening.
2. Add helper that computes the natural source/result relation and result
   arity.
3. Add helper that computes the strided-result relation and result arity.
4. Add support query for valid narrowing layout pairs.
5. Teach assignment/rematerialization to select the strided fact for explicit
   result requests, direct compact-store consumers, or true arity reductions.
6. Lower single-part narrowing directly when the target has a part-selecting
   narrow instruction.
7. Preserve existing deinterleaved-to-contiguous narrowing for the packed full
   result case.

This is the same family as the recently discussed `d4 -> c -> d2 -> vcvt -> c`
optimization: if a cast op has a direct source/result layout relation,
assignment/rematerialization should expose that relation before lowering.

## 7. Ensure-Layout And Rematerialization

### 7.1 `ensure_layout`

`ensure_layout` remains the explicit use-site materialization op.

Verifier/lowering policy:

```text
same source and target dense lane map:
  fold away

known dense relation:
  lower contiguous <-> lane_stride through register pack/unpack when supported
  lower contiguous/deinterleaved relations through existing intlv/dintlv paths

producer can rematerialize target layout:
  rematerialization should replace ensure_layout(producer)

unknown relation:
  reject before vmi-to-vpto
```

Avoid adding a generic "any dense layout to any dense layout" promise unless the
target really has a lowering for it.

### 7.2 Rematerialization

The current checked-in `vmi-layout-rematerialize` cheap producers are:

```text
data:
  VMIExtFOp / VMIExtSIOp / VMIExtUIOp when the source layout can be
  materialized for the requested result relation
  VMIFmaOp
  binary layout-transparent ops:
    addf/addi/subf/subi/mulf/muli/divf/minf/maxf/andi/ori/xori/shli/shrui/shrsi
  unary layout-transparent ops:
    negf/absf/absi/sqrt/exp/ln/relu/not
  VMIConstantOp only when the DenseElementsAttr is a splat
  VMIBroadcastOp
  VMIIotaOp

mask:
  VMICreateMaskOp
  VMICreateGroupMaskOp
  VMIConstantMaskOp

special rewrite:
  selected VMITruncFOp / VMITruncIOp through source/result ensure_layout when
  the cast relation is a supported narrowing relation
```

Not included as cheap producers in the current pass:

```text
load / masked_load / group_load / group_slot_load / group_broadcast /
group_broadcast_load / store / reduce / control-flow ops
```

Loads need a separate policy. `load -> ensure_layout` should be folded in
`vmi-layout-fold` when one original load can directly produce the requested
layout. A normal load should not be cloned/rematerialized unless a later safe-read
proof explicitly permits that clone.

Relationship between cheap producers and dense `lane_stride`:

```text
assignment:
  creates the target layout request explicitly, usually as ensure_layout(... ->
  lane_stride) or as a cast source/result relation.  For casts, assignment may
  itself choose the arity-reducing lane-stride relation; remat only reacts to
  later use-site layout requests.

rematerialize:
  does not choose lane_stride as a preference
  only consumes the explicit helper/request and rebuilds the producer with the
  requested lane_stride result type when the producer is cheap and locally legal
```

Required rematerialize changes for dense `lane_stride`:

```text
materializeDataLayout:
  no special producer logic, but canMaterializeDataLayout must understand
  contiguous <-> lane_stride through register pack/unpack

splat constant / broadcast / iota:
  rebuild the op with the requested lane_stride result type
  lowering later materializes that layout directly or through ensure_layout

layout-transparent unary/binary/fma:
  rebuild the op with the requested lane_stride result type
  materialize each operand to the same lane_stride layout before rebuilding
  this relies on canMaterializeDataLayout for operand conversions

widening ext:
  update getWidenSourceLayoutForResultLayout so a requested result layout derives
  the required source lane_stride:
    result contiguous, W=2 -> source lane_stride=2
    result lane_stride=R, W=2 -> source lane_stride=2*R
  remat then inserts/uses source ensure_layout and rebuilds ext with the
  requested result layout

narrowing trunc:
  add getNarrowSourceLayoutForResultLayout or an equivalent relation helper.
  For a requested result lane_stride=R and narrowing ratio W, derive the source
  layout that can produce that result with a selected hardware part:
    result lane_stride=W, W=2 -> source contiguous
    result lane_stride=R, W=2 -> source lane_stride=R/W when divisible
  remat then inserts/uses source ensure_layout and rebuilds trunc with the
  requested result layout

trunc source-ensure rewrite:
  extend the existing source-ensure rewrite to recognize lane_stride narrowing
  relations for VMITruncFOp and VMITruncIOp, not only deinterleaved narrowing
  relations

mask producers:
  only participate after mask layout support defines the corresponding
  lane-stride or predicate-granularity relation; otherwise unchanged
```

Example:

```text
before remat:
  %b0 = pto.vmi.broadcast %s : !vmi.vreg<64xf16, contiguous>
  %b1 = pto.vmi.ensure_layout %b0
      : contiguous -> contiguous, lane_stride=2
  %y  = pto.vmi.extf %b1 : f16 -> f32

after remat:
  %b1 = pto.vmi.broadcast %s
      : !vmi.vreg<64xf16, contiguous, lane_stride=2>
  %y  = pto.vmi.extf %b1 : f16 -> f32
```

This removes a register layout materialization and lets `vmi-to-vpto` lower the
ext as the single selected conversion part.  It is still driven by the explicit
layout request; remat does not inspect sibling consumers or choose lane_stride by
itself.

Do lane-stride cast rematerialization only in these cases:

```text
required shape:
  cast result is followed by ensure_layout to a requested dense result layout
  widening or narrowing ratio W > 1
  the requested source/result layout pair is accepted by the cast relation
  helper
  the cast with that source/result layout can lower as one selected conversion
  part or the existing multi-part relation

acceptance/safety gate:
  the source-side lane_stride request must be discharged by a concrete local
  rewrite, not merely moved from result side to source side
  accepted cases:
    the source already has the required lane_stride
    the source producer is in the checked-in cheap producer list and can be
    rebuilt with the required lane_stride
    the source is load -> ensure_layout and vmi-layout-fold can replace it with
    a single original-position layout-aware VMI load
    a layout-transparent chain can be sunk/rematerialized until one of the above
    concrete producer cases is reached

do not apply:
  result consumer already accepts the natural cast layout
  requested cast layout relation is unsupported
  source is an ordinary load with other incompatible consumers and no safe-read
  proof to clone it
  the rewrite only moves an expensive materialization from result side to source
  side without exposing a direct lowering
```

Typical accepted shapes:

```text
broadcast -> ext -> ensure_layout(contiguous) -> store
  remat broadcast as lane_stride=W
  ext lowers with one conversion part
  no source-side ensure_layout remains

load -> ensure_layout(lane_stride=W) -> ext -> store
  fold load into a layout-aware VMI load
  vmi-to-vpto later emits the matching UNPK dist
  ext lowers with one conversion part
  no extra load is cloned

elementwise cheap chain -> ext -> ensure_layout(contiguous)
  remat/sink the chain to lane_stride=W only when the chain reaches a concrete
  cheap producer or direct load-fold case

trunc -> ensure_layout(lane_stride=W) -> compact store
  remat/rebuild trunc with the requested lane_stride result when the source
  layout relation is supported
  store fold may then consume the lane_stride result directly

trunc -> ensure_layout(lane_stride=W) -> masked_store
  only accepted after mask layout assignment can provide the same lane map for
  the predicate; otherwise keep the conservative contiguous masked-store path
```

## 8. Broadcast And E2B Interaction

Do not encode E2B in `lane_stride`, and do not define
`vmi.group_broadcast_load` in terms of E2B.  The VMI operation is a logical
fused memory operation:

```text
for each logical group g:
  scalar = source[offset + g * source_group_stride]
  for each lane i in group g:
    result[i] = scalar
```

The result layout is assigned separately.  It may be contiguous,
deinterleaved, or dense lane-strided if the consumer asks for that lane map and
the target support table accepts it.  E2B is only one VPTO lowering strategy for
a restricted subset of this logical operation.

The layering should be:

```text
logical group broadcast load
  -> assigned dense layout, possibly lane-strided
  -> support query chooses a lowering strategy
  -> selected VPTO dist, if any
```

For the current E2B strategy, the support query checks:

```text
source is direct memory
source_group_stride is constant 1
num_groups is a multiple of 8
element storage width
logical group size derived from num_groups
assigned result layout:
  contiguous for the direct packet size
  or deinterleaved=2 for the split packet size
```

Then it may choose an E2B packet:

```text
b16 contiguous:              direct 1 -> 16 packet
b16 deinterleaved=2:         two logical halves / 1 -> 32 reuse
b16 dense lane_stride=2:     direct phase-zero strided consumer packet
b32 contiguous or strided:   target-specific packet size
```

If those conditions do not hold, the operation is still a valid VMI semantic if
some other lowering exists, such as `group_slot_load + group_broadcast`, scalar
loads plus broadcast, or future target-specific broadcast-load support.  The
failure is only "this E2B lowering strategy is not applicable", not "the VMI
operation means E2B".

Concrete implementation plan for lane-stride `group_broadcast_load`:

```text
include/PTO/Transforms/VMILayoutSupport.h
lib/PTO/Transforms/VMILayoutSupport.cpp

1. Split semantic support from E2B strategy checks:

   getGroupBroadcastLoadSupport(capabilities, op)
     try getE2BGroupBroadcastLoadSupport(capabilities, op)
     if success:
       return {kind = E2BVlds}
     return failure("no registered group_broadcast_load lowering strategy; "
                    "E2B rejected because ...")

   getE2BGroupBroadcastLoadSupport(capabilities, op)
     contains the current E2B constraints:
       source is !pto.ptr direct memory
       element width is b16 or b32
       source_group_stride is constant 1
       num_groups is a multiple of 8
       group size matches direct or split E2B packet size
       result layout is contiguous or deinterleaved=2
       result has full physical chunks

2. Keep VMIGroupBroadcastLoadSupportKind strategy-specific:
     E2BVlds means "lower this VMI semantic using E2B"
   It must not be used as the definition of the VMI op.
```

```text
lib/PTO/Transforms/VMILayoutAssignment.cpp

3. Rename strategy helpers so the direction is clear:

   isE2BGroupBroadcastLoadCandidate
     -> isE2BGroupBroadcastLoadStrategyApplicable

   getPreferredGroupBroadcastLoadLayout
     -> getPreferredE2BGroupBroadcastLoadLayout

4. Fusion from group_slot_load + group_broadcast to group_broadcast_load remains
   guarded by E2B applicability.  If E2B is not applicable, do not create a
   fused group_broadcast_load merely because the VMI semantic would be valid.
   That avoids producing an op with no registered lowering strategy.

5. Layout assignment for an explicit group_broadcast_load uses the support
   query:
     if E2B strategy applies:
       assign the E2B-preferred result layout
     else:
       leave the op to validation unless a fallback strategy is added
```

```text
lib/PTO/Transforms/VMIToVPTO.cpp

6. Replace duplicated local E2B legality checks with:
     support = getGroupBroadcastLoadSupport(capabilities, op)
     switch support.kind:
       E2BVlds:
         emit the existing E2B packet sequence

   The E2B lowering code may still assert/recheck structural invariants needed
   for indexing, but user-facing diagnostics should come from the support query.

7. Diagnostics must name the strategy:
     good: "group_broadcast_load has no registered lowering strategy; E2B
            rejected because source_group_stride is not constant 1"
     bad:  "group_broadcast_load requires constant unit source_group_stride"

   The second form is only valid inside an E2B-specific diagnostic.
```

Required group-broadcast-load tests:

```text
E2B positive:
  explicit group_broadcast_load with b16/b32, stride=1, matching group size,
  and assigned contiguous/deinterleaved result layout
  CHECK vmi-to-vpto emits E2B_B16/E2B_B32

E2B strategy rejection:
  source_group_stride != 1, wrong group size, or unsupported element width
  CHECK validation/lowering diagnostic says no registered lowering strategy and
  reports E2B as the rejected strategy

fusion guard:
  group_slot_load + group_broadcast shape that is not E2B-applicable
  CHECK assignment does not fuse it into group_broadcast_load

semantic boundary:
  explicit group_broadcast_load that is not E2B-applicable
  CHECK failure wording does not redefine the op as E2B and does not imply the
  logical VMI semantic itself is E2B
```

This keeps broadcast optimization generic across type width and layout, instead
of hardcoding one `ComputeY1ToFP8` scale pattern.

## 9. Tests

Use the following as the coverage matrix for current-stage support plus the
masked-store and group-broadcast-load follow-up items.  It is not a separate
list of all remaining implementation work.

Parser/verifier:

```text
parse/print contiguous lane_stride
parse/print deinterleaved + lane_stride and block_deinterleaved
```

Physicalization:

```text
64xf16 contiguous lane_stride=2 has one physical 128xf16 part
ui16 contiguous lane_stride=2 may lower through low ui16 in ui32 carrier slots
when the selected materialization is vpack/PK_B32
ui8 contiguous lane_stride=4 may lower through low ui8 in ui32 carrier slots
when the selected materialization is PK4_B32
65xf16 contiguous lane_stride=2 is rejected by direct full-chunk-only paths, or
covered only by an arity-changing materialization test outside this discussion
group-slot ui8 lane_stride=4 has a physical 256xui8 part; any b32 carrier used
by PK4_B32 lowering is instruction-local
```

Conversion lowering:

```text
f16 lane_stride=2 -> f32 contiguous emits one EVEN conversion
bf16 lane_stride=2 -> f32 contiguous follows the same relation
ui8 lane_stride=2 -> ui16 contiguous follows W=2
ui8 lane_stride=4 -> ui32 contiguous follows W=4 when target supports it
contiguous f16 -> deinterleaved=2 f32 still emits EVEN + ODD
f32 contiguous -> f16 lane_stride=2 emits the selected narrowing part when the
assigned relation is supported
f32 deinterleaved=2 -> f16 contiguous keeps the existing packed full-result
narrowing relation
ui16 lane_stride=2 -> contiguous can materialize with vpack 32->16 carrier path
ui8 lane_stride=4 -> contiguous can materialize with two vpack stages
```

Assignment/rematerialization:

```text
extf records a strided dense source relation when compact result arity is
smaller than natural result arity
extf 64xf16 -> 64xf32 chooses source lane_stride=2, result contiguous
extf 128xf16 -> 128xf32 chooses result deinterleaved=2
extf 256xf16 -> 256xf32 chooses result deinterleaved=2
truncf records a strided result relation only when the conservative
self-preference/support rule or a supported consumer request selects it; it
does not choose lane_stride solely because the op narrows
layout-transparent op propagates the same strided layout through operands/result
ensure_layout is folded when source and target lane maps match
rematerialization clones a cheap broadcast for two different dense layouts
```

End-to-end assignment cases:

```text
contiguous load -> ext -> contiguous store:
  uses lane_stride only when the source ensure_layout can be folded to the
  original load, rematerialized from a cheap producer, or lowered by a supported
  register materializer

cheap broadcast -> ext -> contiguous store:
  rematerializes broadcast as lane_stride=2 and lowers ext with one EVEN part

producer -> ext -> deinterleaved reduce:
  keeps source contiguous and result deinterleaved=2

cheap producer -> ext feeding both store and reduce:
  keeps shared deinterleaved path for reduce and rematerializes a contiguous
  result path for store only through the checked cheap-producer remat path

group_broadcast_load -> ext -> contiguous consumer:
  chooses lane_stride only if group_broadcast_load supports that lane map
```

Negative tests:

```text
assigned ext layout pair where LS % W != 0 and no multi-part relation exists
assigned trunc layout pair where result lane_stride is not compatible with the
narrowing ratio
ordinary dense op with mismatched lane_stride operands
store consuming strided dense layout without a supported store/materialization
masked_store consuming lane_stride value with a stale contiguous user mask is
rejected or kept on the conservative contiguous path
```

## 10. Suggested Patch Order

1. Add attr fields, parser/printer, verifier, and round-trip tests.
2. Unify dense and group-slot data physicalization on the logical element type,
   while keeping pack/unpack carrier selection inside lowering helpers.
3. Update physical arity/unpack helpers for dense lane stride.
4. Extend support queries and assignment layout keys.
5. Implement widening arity-driven self-preference, single-part relation, and
   tests.
6. Implement narrowing inverse relation support, consumer-request handling, and
   tests.
7. Teach rematerialization/fold about exact dense lane-map equality.
8. Add broadcast/E2B recognition improvements that consume assigned lane maps.

Each step should keep existing group-slot `lane_stride` tests passing. The first
functional optimization can be the `f16/bf16 lane_stride=2 -> f32 contiguous`
single-part conversion, but the IR and helper changes should already be generic
over type width and lane-map fields.
