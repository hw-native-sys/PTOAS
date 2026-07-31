# VMI Lane-Stride Layout Generalization Design

本文定义 `lane_stride` 从 group-slot 专用属性泛化为 VMI layout 的通用
物理 lane 映射轴。目标不是只优化 `64xf16 -> 64xf32`，而是给 dense
value、group-slot value、类型转换、broadcast materialization 和 load/store
rematerialization 提供统一表达。

## 1. Problem

当前文档对 `lane_stride` 的语义是：

```text
logical lane-sized physical slot 之间有固定间距
```

但实现只允许它出现在：

```text
#pto.vmi.layout<num_groups = G, slots = K, lane_stride = LS>
```

并且现有 helper 会把 `ui8 lane_stride=4` 这类 group-slot lowering 映射为
b32 carrier。这导致两个问题：

1. dense value 无法表达“64 个 f16 logical lanes 放在一个 128xf16 物理向量
   的偶数 lane 上”。
2. `lane_stride` 的 layout 语义和 group-slot carrier lowering 被混在一起。

泛化后必须保持以下边界：

```text
lane_stride:
  layout lane map, does not change logical element type

carrier packing:
  one lowering strategy for selected group-slot integer stores
```

## 2. Semantic Model

### 2.1 Dense Layout

Dense layout 仍然表示每个 logical lane 都有语义值。第一阶段只增加
`lane_stride` 一个新轴：

```text
layout kind: contiguous or deinterleaved
deinterleave factor F when applicable
lane stride LS
```

建议 surface spelling：

```text
#pto.vmi.layout<contiguous>
#pto.vmi.layout<contiguous, lane_stride = LS>

#pto.vmi.layout<deinterleaved = F>
#pto.vmi.layout<deinterleaved = F, lane_stride = LS>
#pto.vmi.layout<block_deinterleaved = F>
```

Defaults:

```text
F = 1 for contiguous
LS = 1
```

Ordinary dense lane map:

```text
logical lane i
part p = i % F
physical lane = (i / F) * LS
```

`block_deinterleaved` remains a separate fixed-32B layout kind and does not
accept `lane_stride`; its block element count is derived from the carrier type.

The current stage intentionally describes only phase-zero strided dense layouts.
For `lane_stride = 2`, that means semantic lanes occupy even physical lanes.

An optional future `lane_offset` or `lane_phase` field is useful only after the
IR has a concrete zero-copy view or producer whose logical lane `i` is
intentionally represented at physical lane `2 * i + 1` or another non-zero
phase.  The current stage has no such producer.  The field should
not be added just because the target has a `vcvt ODD` instruction.

`vcvt ODD` is needed in two different situations:

```text
1. Full conversion of a packed contiguous source.
   Example: contiguous f16 -> deinterleaved=2 f32 uses EVEN and ODD.
   This is not an odd-phase dense source layout; it is the normal multi-part
   lowering of a packed source.

2. Single-part conversion of a future zero-copy odd-lane view.
   Example: if a logical deinterleave/extract result were represented as
   f16 lane_stride=2, lane_offset=1 instead of being compacted, then converting
   that view to f32 contiguous would use ODD.  This requires an explicit VMI
  producer or consumer contract; current-stage dense stride does not
  create such values.
```

The current design implements case 1 with existing conversion lowering and case
2 only as a non-goal extension.  The useful dense-stride optimization in this
stage uses phase-zero layout and therefore selects `EVEN` for `W=2`.

### 2.2 Deinterleaved vs Lane Stride

Use `deinterleaved` when multiple semantic residue classes or physical parts of
the same dense logical value are all present.

Use dense `lane_stride` when one semantic stream is stored sparsely inside each
physical part and the skipped lanes have no semantic value for this VMI value.

Decision rule:

```text
all residue classes are semantic:
  use deinterleaved

only one phase-zero residue class is semantic:
  use lane_stride

multiple parts are semantic and each part is internally strided:
  use deinterleaved + lane_stride
```

Examples:

```text
contiguous f16 -> f32 full dense widen:
  source lanes 0,1,2,3,... are all semantic
  result naturally has even/odd conversion parts
  use result deinterleaved=2

64xf16 -> 64xf32 where the f32 consumer wants contiguous:
  the vcvt layout support may request source lane_stride=2
  if the source producer/rematerialization can satisfy that request, source
  lanes 0,2,4,... become semantic and lanes 1,3,5,... are holes for this value
  extf result can then be contiguous through one EVEN conversion

group-reduce or dense consumer that needs two/four logical fragments:
  the fragments are semantic parts of the same dense value
  use deinterleaved=2/4, not lane_stride
```

Do not use `lane_stride` to describe a full packed value that happens to need an
ODD conversion part.  Do not use `deinterleaved` to describe holes inside one
physical part.

Important distinction:

```text
one hardware vcvt output:
  always one contiguous VPTO output register

VMI ext result layout:
  describes how one or more hardware output registers map back to logical lane
  order
```

For `W=2`, with logical f16 lanes named by their logical indices:

```text
source contiguous:
  physical lanes: 0, 1, 2, 3, 4, 5, ...
  vcvt EVEN output carries logical lanes 0, 2, 4, ...
  vcvt ODD  output carries logical lanes 1, 3, 5, ...
  VMI result layout is deinterleaved=2 unless another materialization
  interleaves the two outputs.

source lane_stride=2:
  physical lanes: 0, _, 1, _, 2, _, ...
  vcvt EVEN output carries logical lanes 0, 1, 2, ...
  VMI result layout is contiguous.
```

So "vcvt output is contiguous" does not by itself mean the VMI `extf` result is
contiguous.  The result layout depends on the logical lane mapping of the source
layout and the selected conversion parts.

### 2.3 Group-Slot Layout

Group-slot layout remains non-dense. Only `G` group result slots have semantic
values:

```text
#pto.vmi.layout<num_groups = G, slots = K>
#pto.vmi.layout<num_groups = G, slots = K, lane_stride = LS>
```

Existing mapping is preserved:

```text
slot_block(g) = g / K
slot_lane(g)  = (g % K) * LS
```

This remains a group-slot placement property. It does not make non-slot lanes
semantic. Existing `ui8 lane_stride=4` to b32 carrier lowering is still legal,
but it is not the definition of `lane_stride`.

Group-slot `lane_offset` is not needed in the current stage. It should remain
out of scope unless a real group-slot producer needs non-zero phase.

## 3. Physical Capacity

`lane_stride` increases the number of physical lane slots needed by a dense
part, but it does not change the VMI logical element type.

For one dense physical part in the current stage:

```text
logical lanes in this part = M
required physical lanes    = (M - 1) * LS + 1
```

The number of VPTO physical registers for each part is:

```text
ceil(required physical lanes / lanes_per_vpto_register(T))
```

Total physical arity:

```text
deinterleave factor F * registers per part
```

Example:

```text
!vmi.vreg<64xf16, contiguous, lane_stride=2>

lanes_per_vpto_register(f16) = 128
required physical lanes      = 63 * 2 + 1 = 127
physical arity               = 1
```

The 64 logical f16 lanes occupy physical f16 lanes `0, 2, 4, ... 126` of one
`!pto.vreg<128xf16>`.  The other lanes are undefined unless another layout
value gives them semantics.

VMI physical parts keep the logical element type for both dense and packed
group-slot lane-stride layouts.  For example, a b16 value with
`lane_stride=2` is represented as a b16 vector whose semantic values occupy
lanes `0, 2, 4, ...`.  `UNPK_B16`, `PK_B32`, and register pack/unpack lowering
may use b32 values internally, but those wider values are instruction-local
carriers and do not become the physical type of the VMI SSA value.

## 4. Type And Operation Generalization

The design is element-type agnostic.  Dense `lane_stride` applies to any VMI
element type whose physical VPTO lane count is known:

```text
f8, f16, bf16, f32
i8, ui8, i16, ui16, i32, ui32
pred masks at an explicit predicate granularity
```

An op may support a strided dense layout only when its VPTO lowering can
preserve the lane map. Unsupported combinations are rejected by layout support
queries, not silently repaired in `vmi-to-vpto`.

### 4.1 VPTO Pack/Unpack Support Boundary

Dense `lane_stride` is not a generic VPTO load/store operand.  It is supported
only when the lane map matches a concrete VPTO distribution or register
materializer.

Direct compact memory support:

| Dense lane_stride | compact load | compact store |
|---:|---|---|
| 2, b8 | `vlds UNPK_B8` | `vsts PK_B16` |
| 2, b16 | `vlds UNPK_B16` | `vsts PK_B32` |
| 2, b32 | `vlds UNPK_B32` | `vsts PK_B64` |
| 4, b8 | `vlds UNPK4` | `vsts PK4_B32` |
| 4, b16/b32 | no direct dist | no direct dist |

Direct scalar broadcast load target capability:

```text
lane_stride=2/4, b8/b16/b32:
  vlds BRC_B8/B16/B32
```

The current stage does not add a VMI scalar broadcast-load op.  BRC is therefore
a target capability for a separate scalar broadcast-load semantic, not part of
the current `vmi.load -> ensure_layout` compact-stream fold.

Register fallback between contiguous and dense `lane_stride` should use the
register-side counterpart of these distributions:

```text
contiguous -> lane_stride:
  vsunpack/vzunpack-style placement into wider slots

lane_stride -> contiguous:
  vpack-style extraction from wider slots
```

`vintlv`/`vdintlv` remain the materializers for two-stream
interleave/deinterleave layouts; they are not the primary fallback for dense
`lane_stride`.

### 4.2 Layout-Transparent Dense Ops

Layout-transparent dense ops include ordinary elementwise arithmetic and
select-like ops when every dense data operand/result has the same layout:

```text
add/mul/fma/min/max/select:
  operands and result require identical dense layout key
  key includes F, B, and LS
```

No physical shuffle is implied by these ops.

### 4.3 Widening Conversion

Let a widening conversion increase element storage width by ratio `W`:

```text
f16 -> f32: W = 2
bf16 -> f32: W = 2
i16 -> i32: W = 2
ui16 -> ui32: W = 2
f8 -> f32: W = 4
i8 -> i32: W = 4
ui8 -> ui32: W = 4
ui8 -> ui16: W = 2
```

For a phase-zero source dense layout with `lane_stride = LS`, a single hardware
conversion part is sufficient when:

```text
LS % W == 0
```

The selected hardware part in the current stage is:

```text
part = 0
```

The result layout after conversion is:

```text
result lane_stride = LS / W
```

For a future phase-aware layout with `lane_offset = O`, the generic relation is:

```text
part               = O % W
result lane_stride = LS / W
result lane_offset = (O - part) / W
```

That future relation should be enabled only when a real odd/non-zero-phase VMI
producer or consumer exists.

Examples:

```text
f16 source: contiguous, lane_stride=2
extf to f32:
  use vcvt EVEN
  result contiguous

f16 source: contiguous, lane_stride=4
extf to f32:
  use vcvt EVEN
  result contiguous, lane_stride=2
```

If `LS < W` or `LS % W != 0`, the conversion may need multiple hardware parts
and may naturally produce a deinterleaved result. The current contiguous source
case is the common example:

```text
f16 source: contiguous, lane_stride=1
extf to f32:
  use vcvt EVEN and vcvt ODD
  result deinterleaved=2
```

Assignment chooses one preferred fact for the op before lowering.  Consumer
requests are handled by the existing use-site materialization path after the
op's assigned result layout is fixed.

The preferred direction for this optimization is not "notice the input is
already strided".  The conversion op can be the layout-entry point and compute a
single preferred layout fact for the current op instance.  The choice must be
arity-driven, not special-cased by a spelling such as `64xf32`.

For source/result logical lane count `N`, let:

```text
natural result layout:
  source dense factor F, lane_stride 1
  result dense factor F * W, lane_stride 1

compact result layout:
  result keeps source dense factor F and uses lane_stride 1
  source uses lane_stride W inside the same dense factor F
```

The self-preferred widening rule is:

```text
if physical_arity(compact result) < physical_arity(natural result)
   and target supports the required source lane_stride relation:
  choose compact result and request source lane_stride=W
else:
  choose natural result deinterleaved by W
```

For ordinary contiguous `f16 -> f32` this gives:

```text
64xf32:
  compact arity = 1
  natural deinterleaved=2 arity = 2
  choose source lane_stride=2, result contiguous

128xf32:
  compact arity = 2
  natural deinterleaved=2 arity = 2
  choose natural result deinterleaved=2

256xf32:
  compact arity = 4
  natural deinterleaved=2 arity = 4
  choose natural result deinterleaved=2
```

If the source is already deinterleaved by `F`, the natural result factor is
`F * W`.  For example, `deinterleaved=2 f16 -> f32` naturally produces
`deinterleaved=4 f32`.

The same arity rule applies to other widening ratios and types.  For example,
`ui8 -> ui32` has `W=4`; a lane-stride source is preferred only when the
contiguous result has fewer physical chunks than the natural
`deinterleaved=4` result and the target supports the `lane_stride=4` relation.

The two layout facts are therefore:

```text
baseline fact:
  source contiguous
  result deinterleaved=W
  arity: physical_arity(result deinterleaved=W)

lane-stride fact:
  source lane_stride=W
  result contiguous
  arity: physical_arity(result contiguous)
  source layout request: explicit
```

In the current single-preference framework, `ext` should publish one preferred
fact.  The lane-stride fact is an op-local preference: assignment records the
required source/result relation in the IR and inserts `ensure_layout` at the
source use if the producer is not already in that layout.  Later
rematerialization or fold passes may remove that helper when a concrete producer
rewrite exists; otherwise the helper is either lowered by a registered
contiguous/lane-stride materializer or rejected before `vmi-to-vpto`.

This keeps the optimization in layout assignment/rematerialization, not in a
late `vmi-to-vpto` peephole, and stays within the existing single-preference
assignment model.

### 4.4 Narrowing Conversion

Narrowing uses the same arity-driven idea in the opposite direction.  If source
element width is `W` times the result element width, a single hardware narrowing
part can produce a phase-zero strided result when:

```text
result lane_stride = source lane_stride * W
part = 0
```

This covers more than f32-to-f16. The same relation applies to:

```text
f32 -> f16/bf16
i32 -> i16/i8
ui32 -> ui16/ui8
ui16 -> ui8
```

The natural narrowing relation is the inverse of natural widening:

```text
source dense factor F * W, lane_stride 1
result dense factor F, lane_stride 1
```

The compact-store-oriented relation is:

```text
source keeps dense factor F and lane_stride 1
result keeps dense factor F and uses lane_stride W
```

Narrowing has the same candidate family as widening.  The arity comparison is
made on the source side, because the compact relation keeps the source
contiguous while the natural relation may require a deinterleaved source.

The self-preferred narrowing rule is:

```text
if physical_arity(compact contiguous source)
     < physical_arity(natural deinterleaved source)
   and physical_arity(compact source) == physical_arity(strided result)
   and target supports the source-contiguous/result-lane_stride relation:
  choose source contiguous, result lane_stride=W
else:
  choose natural deinterleaved-source to contiguous-result relation
```

Use-site requests may still select the strided relation when a later consumer
can directly consume it:

```text
if a consumer requests result lane_stride=W
   and target supports source-contiguous/result-lane_stride narrowing:
  request source contiguous
  set or rematerialize result lane_stride=W
```

For ordinary `f32 -> f16`:

```text
64xf32 -> 64xf16:
  natural source deinterleaved=2 arity = 2
  compact source contiguous arity = 1
  choose source contiguous, result lane_stride=2

128xf32 -> 128xf16:
  natural source deinterleaved=2 arity = 2
  compact source contiguous arity = 2
  choose natural source deinterleaved=2, result contiguous
```

So trunc should not blindly create a lane-stride result for every narrowing.
It should apply the same arity/support checks as ext.  A consumer may still
request a strided result when that layout is useful, such as an unmasked compact
store lowered with `PK`/`PK4`.  For masked stores, the value and mask must share
the same lane map before a direct packed masked store is legal.

The exact supported parts are target-op dependent. The layout assignment layer
should ask the op support interface whether a given source/result layout pair is
legal, rather than encoding type-specific shortcuts.

### 4.5 Broadcast Materialization

Broadcast remains a logical operation.  `lane_stride` only describes the chosen
materialized layout.

Scalar or group broadcast can materialize to a dense layout only when the
broadcast lowering or rematerialization support query accepts that lane map:

```text
logical broadcast:
  lane i gets value group(i)

materialized layout:
  lane i is stored at physical lane map(i)
```

This keeps E2B-style optimizations in the layout/rematerialization layer.  A
group broadcast load may choose a dense strided layout when that layout directly
matches a consumer or a target instruction.  If another consumer needs a
different layout, rematerialization may clone the broadcast or insert
`ensure_layout`.

`group_broadcast_load` is also a VMI semantic, not an E2B semantic.  It means:

```text
for each logical group g:
  load one scalar from source[offset + g * source_group_stride]
  broadcast that scalar to all lanes in group g
```

E2B is a target lowering choice for the subset where that logical memory pattern,
the group size, the element width, and the assigned result layout match the E2B
packet semantics.  Other lowering strategies may implement the same VMI
operation, so support queries should report "E2B is applicable" instead of
rewriting the VMI meaning to "this op is E2B".

### 4.6 Masked Lane-Stride Stores

Masks are logical predicates.  A `masked_store` mask bit denotes whether a
logical element participates in the store; it is not automatically a predicate
for the physical lane slot that happens to carry that element after layout
assignment.

For dense `lane_stride`, this distinction matters.  With `lane_stride=2`,
logical lane `i` is carried in physical lane `2*i`.  A packed store then
compacts those even physical lanes into a contiguous memory stream.  A user mask
that is still contiguous cannot be passed directly to that packed store, because
the packed-store predicate is interpreted after the value lanes have been
compacted.

A direct masked compact store is therefore legal only when the compiler has
assigned the value and mask the same lane map.  That may happen because the mask
producer can directly produce the requested lane map, because assignment inserts
a mask `ensure_layout`, or because rematerialization rebuilds the mask producer
for that lane map.  Without that compiler-derived proof, assignment should keep
a layout that the existing masked-store path can lower, even if the corresponding
unmasked store could use a dense lane-stride `PK` instruction.

## 5. Assignment And Optimization Boundary

The assignment pipeline should keep the existing responsibility split:

```text
layout assignment:
  collect consumer requests
  ask producer/op support
  assign explicit layout attrs
  insert ensure_layout for use-local conflicts

rematerialization:
  clone cheap producers for incompatible use-site layouts
  replace ensure_layout(producer) when producer can directly create target layout

layout fold:
  erase or fuse materialization helpers when the producer already has the
  requested lane map

vmi-to-vpto:
  lower explicit assigned layouts only
  no hidden layout selection policy
```

Dense `lane_stride` is therefore an assigned layout fact, not a lowering-side
pattern.  An entry op such as `extf` may prefer it from the conversion ratio
alone; producer-specific rewrites are handled later by fold/rematerialization
passes over explicit helpers.  The selected layout is fixed before
`vmi-to-vpto`, and `vmi-to-vpto` does not rediscover the preference.

## 6. End-To-End Case Walkthroughs

These cases are the intended test for the design.  They show when dense
`lane_stride` is useful and when it should lose to the existing deinterleaved
plan.

The logical programs in this section are pre-assignment VMI and do not carry
concrete layouts.  Layouts shown under "baseline plan" or "lane-stride plan" are
possible assignment results, not layouts written in the input program.

### 6.1 Contiguous Load, Ext, Contiguous Store

Logical program:

```text
%x16 = vmi.load %in      : 64xf16
%x32 = vmi.extf %x16     : 64xf16 -> 64xf32
vmi.store %x32, %out     : dense contiguous memory effect
```

Baseline plan:

```text
load result:
  contiguous f16

ext relation:
  source contiguous f16
  result deinterleaved=2 f32
  lower: vcvt EVEN + vcvt ODD

store:
  needs contiguous f32
  requires result materialization deinterleaved=2 -> contiguous
```

Lane-stride plan:

```text
load result:
  lane_stride=2 f16

ext relation:
  source lane_stride=2 f16
  result contiguous f32
  lower: vcvt EVEN

store:
  consumes contiguous f32 directly
```

Assignment chooses the lane-stride plan for this shape because the contiguous
`64xf32` result uses one physical chunk while the natural deinterleaved result
uses two physical chunks.  This decision is made by the cast arity rule, not by
a pattern that names `64xf32` directly.

The load side then has two concrete outcomes:

```text
accepted direct load fold:
  the original load has only the lane-stride use
  compact load semantics match a supported UNPK dist
  vmi-layout-fold changes the VMI load result layout in place

no direct load fold:
  keep the explicit source ensure_layout
  lower it through register pack/unpack if that materialization is supported
  otherwise validation rejects the unsupported assigned relation
```

This case proves that `extf` can be the layout-entry point, while `load` support
is still decided by the load/ensure fold or by the explicit materialization
helper.

### 6.2 Broadcast, Ext, Contiguous Store

Logical program:

```text
%b16 = vmi.broadcast %s  : 1xf16 -> 64xf16
%b32 = vmi.extf %b16     : 64xf16 -> 64xf32
vmi.store %b32, %out
```

Baseline plan:

```text
broadcast materializes contiguous f16
ext produces deinterleaved=2 f32 through EVEN + ODD
store materializes deinterleaved=2 -> contiguous
```

Lane-stride plan:

```text
broadcast rematerializes directly as lane_stride=2 f16
ext produces contiguous f32 through one EVEN
store consumes contiguous f32
```

Here the lane-stride plan is accepted because broadcast is a rematerializable
producer: it can be rebuilt with the requested physical lane map instead of
requiring a register layout conversion.  This is the kind of producer where
`vcvt` should drive a source `lane_stride=2` request.

### 6.3 Ext Feeding A Deinterleaved Consumer

Logical program:

```text
%x16 = producer          : 128xf16
%x32 = vmi.extf %x16     : 128xf16 -> 128xf32
%r   = vmi.group_reduce %x32  // requests deinterleaved=2
```

Baseline plan:

```text
source contiguous f16
result deinterleaved=2 f32
consumer consumes result directly
```

Lane-stride plan:

```text
source lane_stride=2 f16
result contiguous f32
consumer then needs contiguous -> deinterleaved=2 materialization
```

The baseline plan should win.  A lane-stride fact is not useful when it creates a
layout the consumer does not want.  The cast arity rule also does not prefer
lane_stride here: `128xf32` contiguous and `128xf32 deinterleaved=2` both use
two physical chunks.

### 6.4 One Ext Result Feeding Store And Reduce

Logical program:

```text
%x16 = cheap_or_expensive_producer : 128xf16
%x32 = vmi.extf %x16               : 128xf16 -> 128xf32
vmi.store %x32, %out               // requests contiguous
vmi.group_reduce %x32              // requests deinterleaved=2
```

If `%x16` is not cheap to rematerialize:

```text
assign ext result deinterleaved=2 for the reduce
insert ensure_layout at the store use
```

If `%x16` and `extf` are cheap to rematerialize:

```text
shared path:
  source contiguous -> ext result deinterleaved=2 -> reduce

store-only remat path:
  rematerialized source lane_stride=2 -> ext result contiguous -> store
```

This is a rematerialization decision, not a local `vcvt` peephole.

### 6.5 Group Broadcast Load Feeding Ext

Logical program:

```text
%g16 = vmi.group_broadcast_load %scale : logical dense 64xf16
%g32 = vmi.extf %g16                   : 64xf16 -> 64xf32
consumer requests contiguous %g32
```

The lane-stride plan is accepted only if the group broadcast load lowering can
emit the requested lane map directly:

```text
group broadcast load result lane_stride=2 f16
ext result contiguous f32
```

If the broadcast load can only produce contiguous or deinterleaved packets for
the target element width, assignment should keep those layouts and let later
materialization/rematerialization handle the consumer conflict.  Dense
`lane_stride` is a requestable layout, not a guarantee that every producer can
create it.

## 7. Compatibility Rules

Two dense layouts are identical only if all lane-map fields match:

```text
F, B, LS
```

Two dense layouts may be related by an explicit materialization only if a
registered relation can lower the map conversion. Examples:

```text
contiguous <-> deinterleaved=2
deinterleaved=2 <-> deinterleaved=4 when supported by existing intlv/dintlv
contiguous <-> contiguous, lane_stride=2 when pack/unpack materialization or
producer rematerialization supports it
```

The baseline assignment must not assume an arbitrary dense-to-dense
`ensure_layout` is free or legal. Unsupported materializations should fail in
verification or remain unselected by support queries.

## 8. Non-Goals

This design does not:

1. Turn memory layout into strided memory semantics. Dense VMI `lane_stride`
   describes register materialization, not GM/UB address stride.
2. Make non-slot lanes of group-slot layouts semantic.
3. Require every VPTO op to support every strided layout.
4. Encode `64xf16 -> 64xf32` as a one-off `vcvt EVEN` peephole.

## 9. First Useful Optimization

The motivating case becomes one instance of the generic rule:

```text
source:
  requested as !vmi.vreg<64xf16, contiguous, lane_stride=2>

op:
  extf f16 -> f32, W=2

result:
  !vmi.vreg<64xf32, contiguous>

lowering:
  one vcvt EVEN
```

The same mechanism also covers:

```text
bf16 -> f32 with phase-zero lane_stride=2
ui8 -> ui16 with lane_stride=2
ui8 -> ui32 with lane_stride=4
f8 -> f32 with lane_stride=4
narrowing conversions that intentionally produce phase-zero strided results
broadcast materialization into a consumer-required strided dense layout
```
