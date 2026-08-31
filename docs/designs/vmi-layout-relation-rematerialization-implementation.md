# VMI Layout Relation-Aware Rematerialization Implementation Plan

本文是 `vmi-layout-relation-rematerialization-design.md` 的实现计划。目标是
扩展现有 `vmi-layout-rematerialize` / `vmi-layout-fold` 优化，让 assignment
保持 legal baseline，并从显式 `ensure_layout` 中恢复更好的 producer layout。

## 1. Current Baseline

当前 pipeline 中相关 pass：

```text
vmi-layout-assignment:
  chooses concrete layouts
  inserts ensure_layout / ensure_mask_layout / ensure_mask_granularity

vmi-layout-fold:
  folds selected ensure_layout helpers into layout-aware producers/consumers
  current coverage includes store-side fold, load -> ensure_layout producer
  fold, and inverse nested ensure_layout fold

vmi-layout-rematerialize:
  replaces ensure_* around cheap construction producers
  current data coverage: splat constant, broadcast, iota
  current mask coverage: create_mask, create_group_mask, constant_mask

vmi-layout-sink-materialization:
  sinks matching operand-side helpers through pure elementwise ops
  it does not currently rewrite result-side ensure_layout(op(...), L)
```

ComputeY1-like IR currently remains suboptimal because assignment emits:

```text
ensure_layout(vmul(ext(...), ext(...)), deinterleaved=4)
```

but remat does not yet:

```text
1. hoist result-side ensure_layout through vmul
2. rematerialize ext under a requested result layout
3. expose foldable load/group_broadcast_load helpers
```

## 2. Support APIs

Add support-layer helpers in `VMILayoutSupport`.

### 2.1 Widen Relation Query

```cpp
FailureOr<VMILayoutAttr> getWidenSourceLayoutForResultLayout(
    VMIVRegType sourceType,
    VMIVRegType resultType,
    VMILayoutAttr requestedResultLayout,
    std::string *reason = nullptr) const;
```

Semantics:

```text
1. source/result lane count must match.
2. result element width must be an integer multiple of source element width.
3. first implementation supports widenFactor 2 and 4.
4. requestedResultLayout must be contiguous or ordinary deinterleaved.
5. requested result factor F must be divisible by widenFactor K.
6. derived source factor is F / K.
7. derived source factor 1 means contiguous.
8. derived source/result layout pair must be accepted by ext support gates.
```

Examples:

```text
f16 -> f32, requested result deinterleaved=4
  => source deinterleaved=2

f16 -> f32, requested result deinterleaved=2
  => source contiguous

f8 -> f32, requested result deinterleaved=4
  => source contiguous
```

### 2.2 Ext Support Gates

Update `getExtFSupport`, `getExtSISupport`, and `getExtUISupport` so they accept
relation-rematerialized local shapes:

```text
source layout:
  contiguous or deinterleaved(S)

result layout:
  deinterleaved(S * widenFactor)
```

Keep group_slots integer extension behavior unchanged.

Reject:

```text
1. result layout that is not deinterleaved for dense ext.
2. block_deinterleaved in this first implementation.
3. source/result arity that does not satisfy resultArity = factor * sourceArity.
4. unsupported element width relation.
```

`vmi-to-vpto` ext lowering already works from physical source/result arity. If
support admits `source deinterleaved=2 -> result deinterleaved=4`, lowering must
be covered by tests.

## 3. Rematerialize Pass Changes

Extend `VMILayoutRematerialize.cpp` around `VMIEnsureLayoutOp`.

Recommended ordering for one helper:

```text
try relation-aware ext remat
try result-side layout-transparent producer remat
try existing cheap construction remat
```

The pass should use a helper worklist. When one rewrite creates new
`ensure_layout` helpers, enqueue them so the same pass can continue locally.

### 3.1 Ext Remat Pattern

Match:

```text
%wanted = pto.vmi.ensure_layout %old
%old = pto.vmi.extf %src
```

where `%wanted` has `requestedResultType`.

Rewrite:

```text
derivedSourceLayout =
  support.getWidenSourceLayoutForResultLayout(srcType, requestedResultType,
                                              requestedResultLayout)

%src2 = materialize source to derivedSourceLayout
%new = pto.vmi.extf %src2 : derivedSourceType -> requestedResultType
replace %wanted with %new
```

Equivalent patterns are needed for:

```text
pto.vmi.extf
pto.vmi.extsi
pto.vmi.extui
```

The source materialization step should:

```text
1. reuse %src if it already has derivedSourceLayout.
2. create pto.vmi.ensure_layout otherwise.
3. enqueue the new helper for further remat/fold opportunities.
```

### 3.2 Layout-Transparent Result Helper Remat

Match:

```text
%wanted = pto.vmi.ensure_layout %old
%old = pto.vmi.vmul %lhs, %rhs
```

Rewrite:

```text
%lhs2 = ensure_layout %lhs : lhsLayout -> requestedLayout
%rhs2 = ensure_layout %rhs : rhsLayout -> requestedLayout
%new  = pto.vmi.vmul %lhs2, %rhs2 : requestedLayout
replace %wanted with %new
```

Initial op coverage:

```text
vadd/vsub/vmul/vdiv/vmin/vmax
vand/vor/vxor/vshl/vshr
vneg/vabs/vsqrt/vexp/vln/vrelu/vnot
vmula
```

Optional later coverage:

```text
cmpf/cmpi:
  result is mask, so this belongs with ensure_mask_layout.

select:
  requires coordinated data and mask layout/granularity handling.
```

The pass must preserve op attributes exactly.

### 3.3 Existing Cheap Producer Remat

Keep current behavior for:

```text
splat pto.vmi.constant
pto.vmi.broadcast
pto.vmi.iota
pto.vmi.create_mask
pto.vmi.create_group_mask
pto.vmi.constant_mask
```

These remain direct remat cases and do not require relation queries.

## 4. Fold Pass Interaction

Relation remat may create producer-side helpers:

```text
ensure_layout(load(...), deinterleaved=2)
ensure_layout(group_broadcast_load(...), deinterleaved=2)
```

`vmi-layout-fold` should absorb these when the producer can directly materialize
the requested layout.

Existing load fold should use producer capability, not helper materialization
capability. A load may directly produce a requested contiguous or
ordinary deinterleaved=2/4 result layout even when the helper conversion
from the old load layout to the requested layout would not be a legal register
materialization.

Add fold coverage if missing for:

```text
group_broadcast_load result layout requested as deinterleaved=2
group_slot_broadcast_load result layout requested as deinterleaved=2
```

The fold pass must still be local:

```text
load/group_broadcast_load + ensure_layout
  => cloned/retyped producer with requested result layout
```

It must not inspect downstream `ext` or `trunc`.

## 5. Pipeline

Use a pipeline with fold after remat:

```text
vmi-layout-assignment
  -> canonicalize/cse
  -> vmi-layout-rematerialize
  -> canonicalize/cse
  -> vmi-layout-fold
  -> canonicalize/cse
  -> vmi-layout-sink-materialization
  -> canonicalize/cse
  -> pto-validate-vmi-layout-ir
  -> vmi-to-vpto
```

The first fold handles helpers already emitted by assignment. The second fold
handles helpers exposed by relation-aware remat.

If later result-side remat and operand-side sink need to alternate for longer
chains, the driver may repeat:

```text
vmi-layout-rematerialize
canonicalize/cse
vmi-layout-fold
canonicalize/cse
```

Keep the first implementation single-pass unless tests prove a fixed point is
needed.

## 6. Tests

Add focused lit tests.

### 6.1 Direct Ext Remat

Input shape:

```text
load f16
extf f16 -> f32
ensure_layout ext result deinterleaved=2 -> deinterleaved=4
truncf f32 -> f8
```

Check after:

```text
vmi-layout-rematerialize
```

```text
extf source is deinterleaved=2
extf result is deinterleaved=4
old ensure_layout is gone
```

Check after:

```text
vmi-layout-rematerialize -vmi-layout-fold -vmi-to-vpto
```

```text
load uses deinterleaved load lowering when fold is available
extf lowers from local source/result arity
```

### 6.2 Elementwise Result Helper Remat

Input shape:

```text
extf lhs -> deinterleaved=2
extf rhs -> deinterleaved=2
vmul lhs, rhs -> deinterleaved=2
ensure_layout vmul result -> deinterleaved=4
truncf
```

Check:

```text
vmul is cloned/rebuilt with deinterleaved=4 operands/results
each ext is rematerialized as source deinterleaved=2 -> result deinterleaved=4
no ensure_layout remains between vmul and truncf
```

### 6.3 Multi-Consumer Conflict

Input shape:

```text
ext result deinterleaved=2
consumer A uses deinterleaved=2
consumer B has ensure_layout to deinterleaved=4
```

Check:

```text
original ext remains for consumer A
new cloned ext feeds consumer B
no global layout selection is required
```

### 6.4 ComputeY1

Run:

```text
pto-test-opt compute_y1_to_fp8_fp16_vmi.pto \
  -vmi-layout-assignment \
  -vmi-layout-rematerialize \
  -vmi-layout-fold \
  -vmi-to-vpto
```

Expected:

```text
x load can become deinterleaved=2 and lower through deinterleaved load support
scale path can keep the E2B-compatible deinterleaved layout
vmul/truncf path has no deinterleaved=2 -> deinterleaved=4 helper immediately
before truncf
```

## 7. Non-Goals And Follow-Ups

Do not implement in this change:

```text
1. assignment relation propagation.
2. global layout cost model.
3. trunc/narrow relation remat.
4. cloning memory loads in remat without going through explicit fold support.
5. context-sensitive vmi-to-vpto lowering.
```

Follow-ups:

```text
1. Add narrow relation remat for selected trunc patterns after widen is stable.
2. Add select/cmp mask-aware result helper remat.
3. Consider a fixed-point layout optimization pipeline if long chains need it.
4. Move repeated op-family cloning utilities into a shared helper if the pass
   grows beyond the first ext/elementwise implementation.
```
