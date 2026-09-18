# VMI Layout Request Propagation

This document describes VMI layout request propagation for layout assignment
and local layout rewrites.  It is intentionally independent from any single
optimization case such as `group_reduce -> truncf -> group_broadcast`.

The propagator answers this question:

```text
Given one or more requested layouts for VMI values, can the surrounding IR be
rewritten so those semantic values are available in those layouts?
```

It must not work by clearing existing layout attributes.  Existing layout
attributes are the current IR state.  A pass expresses desired changes by adding
value-layout requests.

## Legacy Propagation Model

Before the shared propagator, VMI layout propagation was split between layout
assignment and post-assignment rematerialization.

`vmi-layout-assignment` used an equivalence-class model.  The pass collected
values whose layouts must be identical, then assigned one layout to each class.
This worked for layout-transparent relations:

```text
source/result of elementwise ops
bitcast-like ops
control-flow forwarded values
mask/data same-layout relations
```

In that model, assignment propagation means "same layout flows through this
relation".  It does not cross relations where the connected values naturally
have different layouts.  Casts, reductions, channel transforms, broadcasts, and
other width- or shape-changing operations cannot be represented as a simple
equivalence-class union.  When the requested layout on one side did not match
the layout chosen for the other side, assignment had to connect the two sides
with `ensure_layout` materialization.

`vmi-layout-rematerialize` then performed a second, optimization-oriented
propagation after assignment.  It started from explicit `ensure_layout` ops,
looked at the producer feeding the `ensure_layout`, and decided whether the
layout conversion could be pushed through or removed by recomputing the
producer.  This was intentionally local and peephole-like:

```text
ensure_layout result requested as layout B
  inspect producer in layout A
  ask whether this producer can be recomputed for B cheaply
  clone/recompute the producer or one of its operands
  replace this one use with the recomputed value
  continue recursively while the local cost model allows it
```

This pass therefore acted like an instcombine-style rematerialization pass over
layout helper IR.  It did not build a graph-level layout solution first.  The
propagation happened as a sequence of one-by-one IR rewrites and replacements,
driven by the `ensure_layout` currently being optimized.

## Refactor Motivation

The split model caused the same layout knowledge to appear in two places.
Assignment needed rules for natural/preferred layouts and same-layout
constraints.  Rematerialization also needed to know which non-equivalence
operations could be crossed, and what layout should appear on the other side
after crossing.  As more cast and reduction cases were added, the duplicated
parts became the risky part:

```text
vcvt/ext/trunc layout facts
group-value cast layout facts
reduce input/result layout facts
broadcast/group-slot layout support checks
```

Adding another rematerialization case would have required copying more of the
assignment-side rule set into the peephole optimizer.  That makes the result
order-sensitive and hard to reason about: assignment may choose one layout,
rematerialization may rediscover a related layout later, and the two passes can
silently disagree about whether the same op relation is legal.

The refactor centralizes the reusable part as op-local transfer relations and a
value-layout propagator:

```text
propagate(value, layout)
  record the requested layout for the semantic SSA value
  run transfer relations through defining ops and users
  derive uniquely implied layouts on connected values
  record conflicts when a second layout is needed
  materialize unresolved conflicts with ensure_layout during apply
```

Assignment remains responsible for policy: it chooses the initial seeds from
true layout decision points such as ext/trunc/reduce results, loads, stores, and
target-specific boundary requirements.  Same-layout ops are relations, not
decision roots.  Consumer operand requests are not producer facts unless a
support-checked direct producer can really adopt that layout.

Rematerialization should then consume the same transfer/support helpers instead
of owning a second copy of cast/reduce layout logic.  Its job becomes local IR
cleanup after a consistent layout table exists: folding helper IR, cloning cheap
producers when that is profitable, or removing redundant `ensure_layout` chains.
It should not be the only place where non-equivalence layout propagation is
defined.

## Core Assignments

The propagator is a pass-local object.  It does not rewrite IR while requests
are being added.  Its core assignment table is value-centric:

```text
assignments:
  Value -> VMIValueLayoutAssignment

worklist:
  Primary Value/layout facts that were just added and still need to be
  propagated through defining and user op transfers.  Conflict layouts do not
  enter this worklist unless a later rematerialization/fork planner explicitly
  creates a new producer instance for that alternate layout.
```

The assignment uses short field names:

```text
VMIValueLayoutAssignment:
  layout:
    The layout that apply must make available for this SSA value.

  conflicts:
    Extra layouts required for this SSA value.  There are two forms:
    def-side, meaning the SSA value itself must also be available in another
    layout; and use-side, meaning one operand use must see the value through
    another layout.

VMILayoutConflict:
  operand:
    Empty for a def-side conflict.  Otherwise, the OpOperand to rewrite during
    apply if the conflict remains material.  Do not create a fake operand for a
    def-side conflict.

  layout:
    The extra layout required for this SSA value or operand use.
```

The implementation can represent this directly as:

```text
VMILayoutConflict:
  OpOperand *operand  // nullptr means def-side
  VMILayoutAttr layout
```

Conflict uniqueness is checked by conflict form:

```text
def-side:
  key = layout
  duplicate layout is a no-op

use-side:
  key = operand
  duplicate operand with the same layout is a no-op
  duplicate operand with a different layout is a hard conflict
```

`assignment.layout` is not a promise that the defining op can directly produce
that layout.  It is the layout that apply must make available for the semantic
value.  Apply implements it by rewriting the source value's VMI type when the
value is type-rewriteable inside the current rewrite scope, or by creating a
primary `ensure_layout` value at a boundary when the source value's type cannot
be rewritten.

A conflict does not overwrite `assignment.layout`.  It records an additional
fork requirement:

```text
def-side conflict:
  if assignment.layout differs from conflict.layout:
    materialize ensure_layout assignment.layout -> conflict.layout at the
    def/boundary

use-side conflict:
  if assignment.layout differs from conflict.layout:
    insert ensure_layout assignment.layout -> conflict.layout before
    conflict.operand
    replace only conflict.operand
  else:
    the conflict is an identity fork and is dropped
```

This keeps the `assignment.layout` model value-bound while still representing
multiple layout requirements.  Operand layout requirements are not stored in a
separate global table and are not encoded inside `VMILayoutAttr`.

The propagator does not rank competing layouts for a value.  The caller decides
the initial request order.  The first layout accepted for a value becomes
`assignment.layout`; subsequent different layouts become conflicts.  Only
`assignment.layout` is a producer-transfer fact.  A conflict means an alternate
layout must be materialized as a fork from the primary value; it is not a claim
that the original defining op result has that alternate layout.  Cost-based
layout choice is outside this first implementation and would change only the
merge policy, not transfer relations.

Do not pre-build separate tables for current layouts, requests,
materializations, rewrites, or conflicts:

```text
current layout:
  Read from value.getType() on demand.  If the type has no layout, the current
  layout is unknown, not contiguous.

request:
  Immediately merge into assignments.  If `assignment.layout` is newly added,
  enqueue the value/layout fact.  Conflict layouts are recorded in
  `assignment.conflicts` but are not propagated through producers by this
  utility.  The worklist de-duplicates by `(Value, VMILayoutAttr)` so the same
  primary fact is propagated once.  The first implementation does not keep a
  separate request log.

materialization:
  Derive during apply by comparing each value assignment's layout with its
  conflicts and with the current IR type.

rewrite:
  Derive during apply from assignments by finding defining ops whose result
  types need to change.

hard conflict:
  Detect while merging one operand's required layout.  Def-side conflicting
  value requests are represented as conflicts; same-operand conflicting
  requests still fail.
```

## API Shape

The basic API is:

```text
request(value, layout):
  request that the SSA value be available in layout.  If this differs from the
  existing `assignment.layout`, record a def-side conflict.

request(operand, layout):
  request that operand.get() be available in layout for this operand only.  This
  does not create the primary assignment for the source value.  If the source
  value's `assignment.layout` is absent or different, record a use-side
  requirement on that operand.

run():
  propagate until the worklist is empty

apply():
  rewrite in-scope VMI value types to their assigned layouts
  materialize boundary values and conflicts with ensure_layout forks
```

The merge operation is the only place that mutates the core table:

```text
request(value, layout):
  if assignments[value].layout exists and differs from layout:
    add assignments[value].conflicts[{def, layout}]
  else if assignments[value].layout is absent:
    assignments[value].layout = layout
    push (value, layout) to worklist

request(operand, layout):
  value = operand.get()
  if assignments[value].layout exists and differs from layout:
    if operand already has a different conflict layout:
      report hard conflict
    add or update assignments[value].conflicts[operand] = layout
  else if assignments[value].layout exists and equals layout:
    no-op
  else:
    create assignments[value] if needed
    add assignments[value].conflicts[operand] = layout
```

`request(operand, layout)` must not delegate to `request(value, layout)`.  A
consumer operand requirement is not a producer fact.  It becomes a local
materialization request unless a separate value request or transfer relation
chooses the same primary layout for the source value.

Transfer relations call the API that matches the derived fact.  For
layout-transparent ops, an operand value fact can derive the op result primary
layout, while the other operands receive operand-local layout requirements.
A result value fact similarly derives operand-local requirements.  Cast inverse
relations also call `request(operand, layout)` because a result layout
determines the layout needed by that cast operand, not necessarily the
producer's global primary layout.  The first implementation may promote an
operand-local request into a producer seed in two narrow cases:

- the source value is defined by a direct layout producer, such as a supported
  `load`, splat-like `constant`, `broadcast`, `iota`, or `group_broadcast`
  whose source group-slot layout and requested result layout pass the
  target-support query;
- the source value is defined by a single-use layout-transparent op, every data
  operand can directly produce the requested layout, and neither the value nor
  any data operand already has a different primary assignment or current layout.

The second case is a local rematerialization-style choice for layout-free IR.
It does not chase arbitrary producer chains, and it must not override an
existing primary assignment, an assignment seed, or an explicit current layout.
Assignment must add producer value seeds before consumer operand requests so a
consumer request cannot steal the primary layout from an ext/trunc/reduce value
that already has a preferred layout.  If an alternate layout is cheap but
requires cloning or sinking an existing producer chain, a later
rematerialization/fold pass may remove the inserted `ensure_layout` by cloning
or folding the producer at the use site.

If a subsequent request asks for a different extra layout, that request is
recorded as a def-side or use-side
conflict depending on which overload created it.

Existing explicit type layouts are part of the current IR state, not the
request input.  A pass should not first write an explicit layout and then ask
the propagator to rediscover it.  Instead, the pass requests the desired
layout directly.  If an explicit layout already exists, the propagator reads it
on demand as the current state when deciding whether materialization is needed.

An existing explicit layout is not a lock.  If a request asks for a layout that
differs from the current IR type or `assignment.layout`, the propagator
records the requested value/layout fact and continues propagation.  During
apply, the propagator decides whether the defining op can be rewritten to that
layout by ordinary in-scope type rewrite because sourceValue is
type-rewriteable, or whether a primary `ensure_layout` fork is needed at a
boundary.

## Implementation Shape

The first implementation should be a small utility, not a replacement for every
layout pass at once:

```text
include/PTO/Transforms/VMILayoutPropagation.h
lib/PTO/Transforms/VMI/VMILayoutPropagation.cpp
```

The public type should expose value layout requests, operand-local value layout
requests, fixed-point propagation, and final IR application:

```text
class VMILayoutPropagator {
  LogicalResult request(Value value, VMILayoutAttr layout);
  LogicalResult request(OpOperand &operand, VMILayoutAttr layout);
  LogicalResult run();
  LogicalResult apply(RewriterBase &rewriter);

  VMILayoutAttr getRequestedOrCurrentLayout(Value value) const;
  VMILayoutAttr getRequestedLayout(Value value) const;
};
```

`getRequestedLayout` returns only `assignment.layout`, not conflict layouts.
`getRequestedOrCurrentLayout` reads `assignment.layout` first, then the current
VMI type layout.  It returns an empty layout when neither exists.  It must not
default to contiguous during propagation.  Passes that need to materialize or
inspect extra layouts should use the value assignment during apply instead of
overloading this singular accessor.

The current assignment pass already has pieces that map directly onto this
utility:

```text
LayoutSolver::setNaturalLayout
  becomes request(value, layout).

LayoutSolver::requestDataUse
  becomes request(operand, layout).  If the request conflicts with the source
  value's `assignment.layout`, the propagator records a use-side conflict on that
  value.

LayoutSolver::getExplicitDataLayout
  becomes a current-layout read from the value type or assigned equivalence.

LayoutSolver::getDataLayout
  currently defaults unknown to contiguous.  The propagator must not do that
  until finalization/apply.

LayoutSolver::applyConsumerDrivenDataLayouts
  is removed.  Consumer operand requirements are represented as use-side
  conflicts and do not become natural layouts.

LayoutSolver::rewriteDataTypes and insertDataUseMaterializations
  provide the first implementation material for VMILayoutPropagator::apply in
  vmi-layout-assignment.
```

This means the first change can be staged:

```text
1. Add VMILayoutPropagator with assignments, per-value def/use conflicts,
   value-layout fact worklist, and strict same-operand conflict checking.

2. Move layout relation helpers into it.  Run mask granularity assignment/split
   before layout propagation, and keep the layout propagator focused on layout
   only.  Do not move all materialization logic in the first step.

3. Let vmi-layout-assignment call the propagator for the relations that need
   order-independent propagation.

4. Factor existing assignment finalization into VMILayoutPropagator::apply
   once the assignment table is authoritative.
```

## Propagation

The propagator's loop is deterministic:

```text
while worklist is not empty:
  pop changed value/layout fact
  inspect the defining op transfer relation, if the value is an op result
  inspect each user op transfer relation
    operand/source value layout known -> derive result value layouts
    result value layout known -> derive operand source value layouts when
    inverse is legal
```

The propagator should work with generic relation records, not with hard-coded
patterns:

```text
same-layout relation:
  elementwise ops, bitcast, select-compatible values

cast width relation:
  source layout known -> derive result
  result layout known -> derive source when inverse is legal

channel split/merge relation:
  source/input side layout known -> derive result side layout, and vice versa
  when the inverse is unique

control-flow relation:
  CFG branch/yield/region layout consistency
  call/return only when function/call boundary rewrite is enabled
```

Each relation should use the same helper that support checks use.  For example,
a cast width relation should know only about source/result layout relations of
cast ops; it must not know about `truncf -> group_broadcast` as a pattern.

### Transfer Relation Details

The core asset is the static transfer relation:

```text
known value + known layout + op-local rule
  -> zero or more uniquely derived layouts for connected source/result values
```

This is different from assignment-specific policy such as "natural layout" or
"consumer request".  Those policies decide where the first request comes from.
The transfer relation only answers whether a known layout on one connected
value uniquely determines layouts on other connected values.

The first useful transfer relation set is:

```text
same-layout relation:
  For layout-transparent ops, if one operand/result gets a requested layout,
  require the same layout on the connected values.  Existing assignment
  `unite` logic is a source of the supported op list.

cast width relation:
  Use VMILayoutSupport cast fact helpers.  Dense and group-value casts should
  both generate VMICastLayoutFact pairs.  Group-value casts need a support
  helper that derives:
    narrow: result.LS = source.LS * width_ratio
    widen:  source.LS = result.LS * width_ratio

channel split/merge relation:
  Split and merge have fixed source/result layout equations once the channel
  count is known.
```

The relation providers should call `VMILayoutSupport` for legality instead of
duplicating target checks.  Missing support helpers should be added there, not
open-coded in the propagator.

### Existing Assignment Transfer Assets By Op

`vmi-layout-assignment` already contains several transfer relations.  They
should be extracted by op or op family.  Each op family may expose more than
one transfer rule, but every rule must still be local to that op.

```text
VMIAddF/AddI/SubF/SubI/MulF/MulI/DivF/MinF/MaxF/AndI/OrI/XOrI:
  Existing code:
    constrainElementwiseBinary(...)

  Transfer rules:
    lhs layout -> rhs layout and result layout, same layout
    rhs layout -> lhs layout and result layout, same layout
    result layout -> lhs layout and rhs layout, same layout

  Notes:
    existing code has fallback handling for unsupported group_broadcast result
    layouts; that fallback is assignment policy, not the transfer rule.

VMIFma:
  Existing code:
    unite(lhs, rhs), unite(lhs, acc), unite(lhs, result)

  Transfer rules:
    any of lhs/rhs/acc/result layout
      -> all of lhs/rhs/acc/result use the same layout

VMINegF/AbsF/AbsI/Sqrt/Exp/Ln/Relu/Not:
  Existing code:
    unite(source, result)

  Transfer rules:
    source layout -> result layout, same layout
    result layout -> source layout, same layout

VMIFPToSI/VMISIToFP:
  Existing code:
    unite(source, result)

  Transfer rules:
    source layout -> result layout, same layout
    result layout -> source layout, same layout

VMICmpF/VMICmpI:
  Existing code:
    unite(lhs, rhs)

  Transfer rules:
    lhs layout -> rhs layout, same layout
    rhs layout -> lhs layout, same layout
    lhs/rhs layout -> mask result layout, same layout
    mask result layout -> lhs/rhs layout, same layout when unique

VMISelect:
  Existing code:
    unite(trueValue, falseValue), unite(trueValue, result)

  Transfer rules:
    any of trueValue/falseValue/result layout
      -> all of trueValue/falseValue/result use the same layout
    selected value/result layout -> mask operand layout, same layout
    mask operand layout -> selected value/result layout, same layout when
      unique

VMIBitcast:
  Existing code:
    unite(source, result)

  Transfer rules:
    source layout -> result layout, same layout
    result layout -> source layout, same layout

Widen cast transfer (VMIExtF/ExtSI/ExtUI):
  Existing code:
    ExtF uses getPreferredCastLayoutFact(...) for dense widening.
    ExtSI/ExtUI have a source group_slots slots=8 branch.
    getPreferredCastLayoutFact(...)

  Transfer rules:
    These three ops are the same widening transfer relation at the layout
    level.  Float/integer signedness affects element-type legality and lowering,
    not the layout algebra.
    dense source layout -> dense result layout when the cast fact is
    unique and supported
    dense result layout -> dense source layout by matching the same cast facts
    group-value source layout -> group-value result layout for supported
    widening
    group-value result layout -> group-value source layout when the matching
    cast fact is unique

  Extraction work:
    ExtF does not currently have the group-value branch that ExtSI/ExtUI have.
    This is a legacy assignment-framework artifact.  Add one widening-cast fact
    generator used by all three ops and let VMILayoutSupport decide which
    element types are legal.

VMITruncF:
  Existing code:
    VMILayoutSupport::getPreferredCastLayoutFact(...)
    TruncF source group_slots slots=1 case

  Transfer rules:
    dense source layout -> dense result layout when the cast fact is
    unique and supported
    dense result layout -> dense source layout when the matching cast fact is
    unique and supported
    group-value source layout -> group-value result layout for supported
    narrowing
    group-value result layout -> group-value source layout when the matching
    cast fact is unique

  Extraction work:
    current group-value branch supports only slots=1 and is source-driven.
    Add dense trunc inverse generation to the shared cast fact helper.

VMITruncI:
  Existing code:
    VMILayoutSupport::getPreferredCastLayoutFact(...)
    TruncI source group_slots slots=1/8 branch

  Transfer rules:
    dense source layout -> dense result layout when the cast fact is
    unique and supported
    dense result layout -> dense source layout when the matching cast fact is
    unique and supported
    group-value source layout -> group-value result layout for supported
    narrowing
    group-value result layout -> group-value source layout when the matching
    cast fact is unique

  Extraction work:
    current group-value branch is source-driven.  The narrow4 slots=8 case
    already computes lane_stride=4; move that equation into the shared cast
    fact helper.

VMIChannelSplit:
  Existing code:
    VMIChannelSplitOp case in addConstraints()

  Transfer rules:
    source deinterleaved=channel_count -> each result contiguous
    result contiguous on every result -> source deinterleaved=channel_count

VMIChannelMerge:
  Existing code:
    VMIChannelMergeOp case in addConstraints()

  Transfer rules:
    every input contiguous -> result deinterleaved=channel_count
    result deinterleaved=channel_count -> every input contiguous

control-flow ops:
  Existing code:
    addIfConstraints, addYieldConstraints, addExecuteRegionConstraints,
    addIndexSwitchConstraints, addWhileConstraints, addForConstraints,
    addBranchConstraints, addReturnConstraints, addCallConstraints

  Transfer rules:
    any equivalent incoming/yield/result/call value layout
      -> same layout on every value in that equivalence group

mask ops:
  Existing code:
    uniteMask(...)

  Transfer rules:
    same-layout mask propagation mirrors data same-layout propagation
```

Mask layout propagation is not a separate assignment flow.  Mask values
participate in the same propagator/worklist as VMI data values.  Data-producing
or data-consuming ops drive their mask operands/results through same-layout
relations:

```text
data layout L -> mask layout L
mask layout L -> data layout L when the relation is unique
```

Mask granularity assignment is separate from layout propagation and runs before
layout assignment.  Different granularities represent different mask values, so
the granularity pass should split or materialize mask values before the layout
propagator sees them.  After that split, each mask SSA value has one fixed
granularity, and the layout propagator only assigns its layout.

Granularity must not become a second independent request dimension in the
layout worklist, and different granularity requirements must not be represented
as layout conflicts on one mask value.

Some assignment logic is not a transfer relation and should not be moved into
the propagator as if it were one:

```text
producer-only layout choice:
  group_reduce/group_load/group_slot_load/group_broadcast_load choosing an
  initial result layout is assignment policy.  It can seed assignments, but it
  is not derived from another operand layout.

store-only requirements:
  store/group_store/masked_store have no result value to infer.  They seed an
  operand layout request for their value operand when the store form requires a
  concrete input layout.  They are not bidirectional transfer relations.

group_broadcast pair support:
  VMILayoutSupport can validate a source/result pair, but source layout alone
  does not always uniquely choose a dense result layout.  Treat it as a support
  check and source requirement unless another value request makes the result
  layout concrete.
```

### Relation Mechanics

Each op does not own a persistent layout table.  The only persistent table is
the propagator's global `assignments`.

An op relation is implemented by a transfer object:

```text
class VMILayoutTransfer:
  propagate(op, changedValue, changedLayout, propagator)
```

`propagate` is an op-local fact propagation method.  It does not rewrite IR.
It requests layouts for connected source and result values.  It should be a
thin wrapper around the op family's pure relation evaluator:

```text
derive(op, changedPort, changedLayout, assignmentView)
  -> zero or more derived port/layout facts

propagate(op, changedValue, changedLayout, propagator):
  inspect op operands/results, attrs, element types, and VMILayoutSupport
  facts = derive(op, changedPort, changedLayout, propagator.assignments)
  for each result fact:
    call request(resultValue, derivedLayout)
  for each operand fact:
    call request(operand, derivedLayout)
```

The evaluator is query-like in the ordinary sense: it is pure, it does not
mutate `assignments`, it does not enqueue work, and it does not rewrite IR.  It
is not a public propagator `query` API because the propagator API that changes
state is still `request`.  This keeps the mutation point explicit while letting
propagation and apply-time validation consume the same op relation.

When the connected value is reached through an operand, `propagate` calls the
operand overload so a merge conflict can become a use-side conflict on the
source value.

This is the mechanism that propagates layout information.  When one connected
value receives a layout, the op relation may infer layouts for other connected
values:

```text
same-layout op:
  any operand source/result layout -> all connected source/result values get
  the same layout

cast op:
  source value layout -> result value layout using width ratio
  result value layout -> source value layout when the inverse relation is legal

channel split/merge:
  channel count plus one side layout -> the other side layout when the inverse
  is unique
```

Not every relation is symmetric, and not every input layout determines every
other operand.  If the relation cannot derive a unique supported layout, it
emits nothing.  If a requested layout differs from the source value's
`assignment.layout`, the request is recorded as a conflict on that value.  The
value overload records a def-side conflict; the operand overload records a
use-side conflict.  If the same operand records two different conflict layouts,
strict propagation fails.

Block arguments are also worklist values.  They are not `OpResult`s and do not
have `getDefiningOp()`, but the propagator can still process them through a
boundary transfer:

```text
process(value, layout):
  if value is an OpResult:
    process the defining op result port
  if value is a BlockArgument:
    process the block/function boundary port
  process all uses of value
```

The block/function boundary transfer is separate from ordinary op-result
transfer:

```text
ordinary op result:
  defining op result <-> defining op operands/results

block argument:
  block argument <-> predecessor terminator successor operands

function argument:
  function argument <-> function signature / call operands, when the pass owns
  signature or interprocedural rewrite
```

CFG block arguments require a same-transfer.  A block argument and each
predecessor terminator successor operand represent the same semantic stream, so
layout requests must propagate in both directions:

```text
block argument layout L -> request predecessor successor operand layout L
predecessor successor operand layout L -> request block argument layout L
```

If a CFG edge cannot satisfy the same layout as the block argument, the
terminator operand request becomes a use-side conflict on the predecessor
source value and apply materializes that edge operand before the terminator.

Function signatures and call sites are a separate boundary.  The first
implementation can leave function/call boundary transfer out if it only rewrites
inside one function and does not update function signatures or call sites.  In
that mode, function arguments are boundary source values: they propagate to
their users, and primary boundary materialization is handled by apply.

### Cast Layout Facts

Width-changing casts should keep source/result layout information paired.  Do
not encode individual concrete vector cases directly in `propagate`, such as
`64xf16 -> 64xf32`.  Generate layout facts from the source/result element widths
and the known anchor layout:

```text
VMICastLayoutFact:
  sourceLayout
  resultLayout
```

`propagate` uses those facts mechanically:

```text
if changedValue is the cast source:
  for each fact whose sourceLayout == changedLayout:
    request result value layout = fact.resultLayout

if changedValue is the cast result:
  for each fact whose resultLayout == changedLayout:
    call request(sourceOperand, fact.sourceLayout)
```

The support layer should provide a fact generator shaped like:

```text
getCastLayoutFacts(sourceType, resultType, anchorSide, anchorLayout)
  -> zero or more VMICastLayoutFact
```

The `anchorSide` is source or result.  Together, `anchorSide` and
`anchorLayout` limit generation to the small set of facts that can match the
currently propagated layout.

For width-changing dense casts, legality and preference are separate tables.
The legal table records only storage element widths and paired source/result
layouts.  It is `AnyN`: vector element count is not part of dense cast
legality.  `N` belongs only in the preferred table when one legal relation is
chosen over another for a concrete shape.

Storage widths are represented by an `ElementBitsPattern`, parallel to
`LayoutPattern`.  A row matches when the concrete source/result storage widths
match the row's bit patterns and the concrete source/result layouts match the
row's layout patterns.  Do not introduce a separate width-class enum; write the
supported bit set directly in the row.

Legal dense rows are written as paired relations, for example:

```text
T16 -> T32:
  contiguous      -> deinterleaved=2
  lane_stride=2  -> contiguous
  deinterleaved=2 -> deinterleaved=4

T32 -> T16:
  deinterleaved=2 -> contiguous
  contiguous      -> lane_stride=2
  deinterleaved=4 -> deinterleaved=2
```

`T8 <-> T32` follows the same idea with factor 4:

```text
T8 -> T32:
  contiguous      -> deinterleaved=4
  lane_stride=2  -> deinterleaved=2
  lane_stride=4  -> contiguous

T32 -> T8:
  deinterleaved=4 -> contiguous
  deinterleaved=2 -> lane_stride=2
  contiguous      -> lane_stride=4
```

Layout legality depends on storage width and physical layout, not on whether
the op is floating-point or integer.  The op support layer still checks whether
a particular VMI op and element type are valid.

Group-slot cast rows live in the same legal table.  They are written as
parameterized layout patterns; `num_groups = G` is inherited from the anchor
layout used for the query.  Packed narrowing records the selected sub-lane
stride on the result, and widening uses the inverse relation:

```text
T8 -> T16:
  group_slots(G, slots=1) -> group_slots(G, slots=1)
  group_slots(G, slots=8, lane_stride=2) -> group_slots(G, slots=8)

T16 -> T32:
  group_slots(G, slots=1) -> group_slots(G, slots=1)
  group_slots(G, slots=8, lane_stride=2) -> group_slots(G, slots=8)

T8 -> T32:
  group_slots(G, slots=1) -> group_slots(G, slots=1)
  group_slots(G, slots=8, lane_stride=4) -> group_slots(G, slots=8)

T16 -> T8:
  group_slots(G, slots=1) -> group_slots(G, slots=1)
  group_slots(G, slots=8) -> group_slots(G, slots=8, lane_stride=2)

T32 -> T16:
  group_slots(G, slots=1) -> group_slots(G, slots=1)
  group_slots(G, slots=8) -> group_slots(G, slots=8, lane_stride=2)

T32 -> T8:
  group_slots(G, slots=1) -> group_slots(G, slots=1)
  group_slots(G, slots=8) -> group_slots(G, slots=8, lane_stride=4)
```

There is no separate group-slot branch in the fact query.  Dense and group-slot
casts are both produced by matching the same legal table against the source or
result anchor layout.

The preferred table is a subset of the legal table.  Exact `N` rows override
the default row for the same width pair:

```text
T16 -> T32, N=64:
  lane_stride=2 -> contiguous

T16 -> T32, default:
  contiguous -> deinterleaved=2
```

When a preferred row is selected, the support helper must validate it through
the same legal fact query.  A preferred row that is not legal is a bug in the
table, not a fallback opportunity.

Examples of legal dense facts:

```text
f32 -> f8, R=4:
  source 256xf32 deinterleaved=4
    -> result 256xf8 contiguous

f32 -> f16, R=2:
  source 256xf32 deinterleaved=4
    -> result 256xf16 deinterleaved=2
```

The inverse direction matches the same facts:

```text
f8 result contiguous
  -> f32 source deinterleaved=4

f16 result deinterleaved=2
  -> f32 source deinterleaved=4
```

The propagator should accept an inverse only when fact generation returns a
single supported matching fact for the anchor side.  If several facts could
satisfy the same side, the relation must not guess; it should emit nothing
unless another request makes the choice concrete.

### Reduce Layout Facts

Plain vector reductions currently have a fixed layout relation:

```text
reduce_*:
  source contiguous
  init   contiguous
  mask   same(source)
  result contiguous
```

`group_reduce_*` has a richer relation and should be expressed as a table.  The
table is parameterized by:

```text
G = num_groups
group_size = source_element_count / G
VcgBlockElems = elements in one 32B VCG block
```

The query first classifies `group_size` against `VcgBlockElems`.  The class is
an enum, not a numeric encoding:

```text
QuarterBlock      group_size == VcgBlockElems / 4
HalfBlock         group_size == VcgBlockElems / 2
OneBlock          group_size == VcgBlockElems
TwoBlock          group_size == 2 * VcgBlockElems
FourBlock         group_size == 4 * VcgBlockElems
FullPartMultiple  group_size >= 8 * VcgBlockElems &&
                  group_size % (8 * VcgBlockElems) == 0
```

Other values are unsupported.  Preferred group block rows are written with a
`gb` pattern.  `gb(1, 4)` means one quarter of a 32B VCG block, and `gb(4)`
means four 32B VCG blocks.  `group_reduce_*` is one consumer of this shared
block classification:

```text
gb(1, 4):
  source ls(4)
  mask   same(source)
  result gs(8)

gb(1, 2):
  source ls(2)
  mask   same(source)
  result gs(8)

gb(1):
  source c()
  mask   same(source)
  result gs(8)

gb(2):
  source d(2)
  mask   same(source)
  result gs(8)

gb(4):
  source d(4)
  mask   same(source)
  result gs(8)

gbFull():
  source c()
  mask   same(source)
  result gs(1)
```

The preferred table is not the whole legal relation.  A concrete fact query
also exists for post-assignment validation:

```text
getGroupReduceLayoutFactForLayouts(source, mask, result, num_groups)
```

It matches the assigned source/mask/result layouts against legal rows for the
classified group block.  Legal rows include additional source/mask alternatives
for the same semantic row, such as ordinary `d(2|4)` and block-based `bd(2|4)`
for the two-block and four-block cases. Those alternatives are part of the
layout relation, not ad-hoc support relaxations.

The concrete query is the single source of truth for layout-driven shape
legality.  It may compute physical arity from the concrete VMI types, but only
to validate the selected relation row; arity is not an independent support
policy.  For example, the two-block row implies two source/mask physical parts
per result part, while the four-block row implies four.

Checks that are intrinsic to the VMI op contract belong in the op verifier, not
in layout support:

```text
floating-point reassociation requirement
source/result element type equality
result element count equals num_groups
integer group reduction accumulator type
mask/data compatibility
num_groups divisibility
```

The concrete lowering plan is derived from the returned fact.  The lowering may
still defensively check the OneToN-converted `sourceParts`, `maskParts`, and
`resultTypes`, but it must not re-encode a separate group-size or layout support
table.

### Memory Layout Facts

Load/store layout support follows the same split:

```text
VMILayoutSupport:
  layout relation facts only

VMIToVPTO:
  target/memory/stride/lowering preconditions
  defensive checks on the actual OneToN value/result ranges
```

Do not encode VPTO instruction names in layout support.  Names such as
`Vstsx2`, `Vsldb`, `PK4_B32`, or `Slots1PointVsts` are lowering choices, not
layout facts.

Dense `vmi.load` / `vmi.store` facts are keyed by element bits and assigned
value layout:

```text
load:
  bits(8,16,32), contiguous
  bits(8,16,32), lane_stride=2
  bits(8),       lane_stride=4
  bits(8,16,32), deinterleaved=2/4

store:
  bits(8,16,32), contiguous
  bits(8,16,32), lane_stride=2
  bits(8),       lane_stride=4
  bits(8,16,32), deinterleaved=2/4
```

The fact query records only the dense memory layout pattern and element-bit
pattern.  It may reject impossible layout/width combinations such as
`lane_stride=4` for non-b8 elements.  It must not decide whether a particular
memref is UB-backed, whether an offset is aligned, or whether the current
lowering will materialize a fallback path.

Two-way memory ops are separate VMI semantics:

```text
deinterleave_load:
  low  contiguous
  high contiguous

interleave_store:
  low  contiguous
  high contiguous
```

They are not represented as dense load/store `deinterleaved` facts.  Their
current VPTO lowering can still require `!pto.ptr`, direct UB memory, full
chunks, and `vldsx2/vstsx2` element support in `VMIToVPTO`.

Group memory facts are also table relations:

```text
group_load:
  bits(32), gb(2) -> result bd(2)
  bits(32), gb(4) -> result bd(4)

group_slot_load:
  result group_slots(num_groups=G, slots=1)
  result group_slots(num_groups=G, slots=8)
  result group_slots(num_groups=G, slots=8, lane_stride=2)
  result group_slots(num_groups=G, slots=8, lane_stride=4)

group_store:
  value group_slots(num_groups=G, slots=1)
  value group_slots(num_groups=G, slots=8)
  value group_slots(num_groups=G, slots=8, lane_stride=2)
  value group_slots(num_groups=G, slots=8, lane_stride=4)

group_broadcast_load:
  bits(8,16,32), memContiguous()    -> source group_slots(G, slots=8)
  bits(8,16,32), memBlockAligned()  -> source group_slots(G, slots=1)

masked_store compact lane-stride:
  bits(8),  value ls(2), mask same(value) -> packed predicate b16
  bits(16), value ls(2), mask same(value) -> packed predicate b32
  bits(8),  value ls(4), mask same(value) -> packed predicate b32
```

The `num_groups` equality is part of the layout relation.  Current lowering
requirements such as `group_load` row stride being a constant positive multiple
of 8, `group_slot_load slots=8` using unit source-group stride, or
`group_store slots=8` using unit row stride remain in `VMIToVPTO`.
`group_broadcast_load` support is expressed as the equivalent
`group_slot_load + group_broadcast` relation.  `VMIToVPTO` may still choose an
E2B VPTO lowering when the matched layout/shape is the E2B-friendly case, but
E2B is not a separate layout support fact.  `masked_store` still materializes
the final predicate in lowering; the support table only records which
layout/mask-shape relations are legal.

The lowering may check actual converted arity before indexing:

```text
if resultTypes.size() != expected arity:
  notifyMatchFailure(...)
```

This is a crash guard for the concrete rewrite.  It must not become another
support policy in `VMILayoutSupport`.

Group-value casts use a different static relation:

```text
narrow by R:
  result.LS = source.LS * R

widen by R:
  source.LS = result.LS * R
```

These fact generators belong in shared support helpers so assignment,
rematerialization, validation, and lowering agree on the same relation.

## Apply

Apply derives concrete IR actions from `assignments` and the current IR.
These actions do not need to be stored in separate propagator tables before
apply:

Apply must not run a second propagation-style relation query over results or
operands.  By the time apply starts, `run()` has already reached a fixed point:

```text
assignment.layout records the layout to make available for each value
assignment.conflicts records def-side and use-side alternate layouts
every def/user relation has already propagated its required layouts
every use that cannot consume assignment.layout has already become a conflict
```

Therefore apply does not ask the op relation again.  It writes the layouts
already recorded in `assignments` into IR:

```text
apply must not create new layout requests
apply must not discover new layout conflicts
apply only consumes assignment.layout and assignment.conflicts
```

```text
sourceValue:
  the original SSA value before apply

currentLayout:
  the layout carried by sourceValue's current VMI type

assignedLayout:
  assignment.layout for sourceValue

assignedValue:
  the SSA value that carries assignedLayout after apply
  if sourceValue is type-rewriteable, assignedValue is sourceValue after its
  VMI type is rewritten to assignedLayout
  otherwise assignedValue is ensure_layout sourceValue :
    currentLayout -> assignedLayout when currentLayout != assignedLayout

def-side conflict:
  materialize an extra SSA value from assignedValue to conflict.layout near
  the def or boundary

use-side conflict:
  materialize an extra SSA value from assignedValue to conflict.layout before
  that use
```

Primary type rewrite is not a producer-specific optimization.  It is the normal
way assignment becomes explicit in IR.  It does not choose a layout and does not
ask whether the producer relation supports the layout; propagation already did
that.  It only updates the VMI type to `assignment.layout`.

Primary type rewrite is available for op results whose defining op is inside
the rewrite scope and whose result type can be changed without rewriting an
external ABI boundary.  Multi-result ops should be rewritten as one op update
using the final assignments for all assigned results.  Block arguments,
function arguments, values defined outside the rewrite scope, and ABI boundary
values use `ensure_layout` materialization instead.

In this document, `type-rewriteable` means exactly:

```text
the value is an OpResult
the defining op is inside the current rewrite scope
changing the result VMI type does not rewrite an external ABI boundary
```

It does not mean the defining op was queried again for a preferred layout.

Primary materialization is not conflict-driven.  A value can have no conflicts
and still need `ensure_layout` when its current IR type cannot be rewritten to
`assignment.layout`:

```text
function argument current layout A
only in-scope use requests layout B

assignment.layout = B
assignment.conflicts = empty

if the function signature is not rewritten:
  arg_B = ensure_layout arg : A -> B
  use(arg_B)
```

Conflicts only describe extra layouts besides `assignment.layout`.  They do not
replace the primary action that makes `assignment.layout` available.

Producer-specific improvements, such as folding a fallback
`load -> ensure_layout` into a load with the requested layout or rematerializing
a cast across an `ensure_layout`, are not part of apply.  They should run as
ordinary layout-fold/rematerialization over explicit helper IR.

```text
1. Make assignment.layout available.
   Let sourceValue be the original SSA value, currentLayout be the layout on
   sourceValue's current VMI type, and assignedLayout be assignment.layout.
   If sourceValue is type-rewriteable, rewrite its VMI type to assignedLayout
   and use sourceValue as assignedValue.  Otherwise, if
   currentLayout == assignedLayout, assignedValue is sourceValue.  Otherwise
   insert ensure_layout sourceValue : currentLayout -> assignedLayout and use
   its result as assignedValue.  If that ensure_layout is not supported by
   VMILayoutSupport, apply fails.

2. Materialize def-side conflicts.
   For each def-side conflict layout, if assignedValue already has that layout,
   reuse assignedValue.  Otherwise insert ensure_layout near the value's
   definition or boundary.  If that ensure_layout is not supported by
   VMILayoutSupport, apply fails.  The ensure_layout result is the materialized
   SSA value for that layout; no persistent container is needed to represent it.

3. Rewrite non-conflicting uses.
   Uses in the rewrite scope are redirected to assignedValue unless a use-side
   conflict records a different layout for that operand.  Do not implement this
   as an unconditional replace-all-uses.  Iterate the original uses and skip
   operands recorded in use-side conflicts.  Do not query the user op relation
   here; non-conflicting uses were already accepted during propagation.

4. Materialize use-side conflicts.
   For each use-side conflict, insert ensure_layout from assignedValue to
   conflict.layout before conflict.operand and replace only that operand.
   If that ensure_layout is not supported by VMILayoutSupport, apply fails.  Do
   not special-case the old producer layout here.  Redundant chains such as
   l1 -> l2 -> l1 are folded by a separate layout-fold/rematerialization pass.
```

Concrete insertion points:

```text
op result:
  normally rewrite the result VMI type in place.  If the result is outside the
  rewrite scope or crosses an ABI boundary, insert the fallback primary
  ensure_layout immediately after the defining op.

block argument:
  insert the fallback primary ensure_layout at the first legal insertion point
  of the owning block.

function argument:
  insert the fallback primary ensure_layout at the first legal insertion point
  of the entry block.

def-side conflict:
  insert ensure_layout after assignedValue is available, using the same
  def/boundary placement as the primary materialization.

use-side conflict:
  insert ensure_layout immediately before conflict.operand.getOwner() and
  replace only conflict.operand.
```

Apply should keep a local materialization map keyed by `(Value, Layout,
placement)` so the same required layout at the same placement is not emitted
twice.  Different use-side conflicts may still materialize separately when a
single def-side value would not dominate all uses.

The source for conflict materialization is always `assignedValue`, not the
operand's old value:

```text
def-side conflict layout C:
  c = ensure_layout assignedValue : assignment.layout -> C

use-side conflict operand op.i requiring layout C:
  c = ensure_layout assignedValue : assignment.layout -> C
  op.i = c
```

If `C == assignment.layout`, the conflict is an identity and no
`ensure_layout` is inserted.

A def-side conflict by itself does not replace arbitrary uses of the original
SSA value.  It only materializes another layout view at the definition or
boundary because a value-level request asked for that layout.  Use-side
conflicts are still materialized locally from assignedValue.

For example, a block argument with current layout `A` and requested
`assignment.layout` `B` keeps its original type unless the rewrite scope allows
changing the boundary.  Apply inserts `ensure_layout A -> B` near the boundary
and rewires in-scope uses to the materialized value, except for operands that
have explicit use-side conflicts.

For a normal defining op inside the rewrite scope, apply rewrites the result
type in place:

```text
before:
  a = producer() : layout A
  use(a)

after assignment.layout = B:
  a = producer() : layout B
  use(a)
```

For a value that cannot be rewritten in place, apply materializes the assigned
layout with `ensure_layout`:

```text
before:
  a0 = boundary_value : layout A
  use(a0)

after assignment.layout = B:
  a1 = boundary_value : layout A
  a  = ensure_layout a1 : A -> B
  use(a)
```

If one use still requires layout `A`, apply emits the local materialization
from the assigned value:

```text
after assignment.layout = B, with one use-side conflict requiring A:
  a = producer() : layout B
  c  = ensure_layout a : B -> A
  use(c)
```

For the non-rewrite fallback, the same conflict may produce an `A -> B -> A`
chain.  That chain is not a special case in apply.  A separate
layout-fold/rematerialization pass may fold it back to the original boundary
value when legal.

For `vmi-layout-assignment`, the existing apply path is already usable:

```text
rewriteDataTypes:
  Sets VMI value types to `assignment.layout`.

insertDataUseMaterializations:
  Inserts pto.vmi.ensure_layout before operand uses whose requested layout does
  not match the source value type.

rewriteMaskTypes / insertMaskUseMaterializations:
  Reused after first-phase mask granularity assignment/split and mask layout
  propagation.
```

For post-assignment optimization passes, apply must be more conservative:

```text
rewritable value:
  The value is an op result inside the rewrite scope and changing the VMI type
  does not cross an external ABI boundary.

non-rewritable value:
  Function/block arguments, external boundaries, or values outside the pass
  rewrite scope keep their original type.  If they have a requested
  assignment.layout different from the current explicit layout, apply inserts
  a def-side ensure_layout inside the rewrite scope and rewires in-scope uses
  to the materialized value.
```

Whether a value is rewritable is derived from the IR and the caller's rewrite
scope.  It is not stored in `assignments`.

## Conflicts

The propagator distinguishes representable conflicts from hard conflicts:

```text
def-side conflict:
  A source value's assignment.layout differs from another value-level request.
  Record VMILayoutConflict{def, layout}.  Apply will materialize it as an
  ensure_layout fork near the value definition or boundary if it remains
  different.

use-side conflict:
  A source value's assignment.layout differs from one operand's required layout.
  Record VMILayoutConflict{operand, layout}.  Apply will materialize it as an
  ensure_layout fork if it remains different.

hard operand conflict:
  The same operand is requested as two different layouts.  Do not create two
  forks for one operand.  The first implementation fails the current
  propagation request.
```

The initial conflict policy should be strict inside the propagator:

```text
same value requested as two different layouts:
  keep the first layout as assignment.layout, record subsequent layouts as
  def-side conflicts.  Propagate only the primary assignment.layout fact.

same operand requested as two different layouts:
  fail the propagation request with a diagnostic at the requesting operation.
```

Value-level and operand-level layout differences are not hard conflicts when
they can be represented by an unambiguous fork.  They are recorded in the
source value's `conflicts` list and are materialized by
VMILayoutPropagator::apply.

## Assignment Shape

`vmi-layout-assignment` can use the propagator as:

```text
collect op layout constraints and relations
request natural layouts for producers that choose concrete layouts
request layouts required by consumers
propagate the value-layout table through op relations
apply ensure_layout / ensure_mask_layout materialization
validate assigned VMI IR
```

Later layout optimization passes should follow the same model:

```text
request a new layout for one or more anchor values
propagate the value-layout table through registered op relations
materialize mismatches at values or uses
run layout-fold/rematerialization to remove redundant helper IR
```

Manual layout clearing is unsafe because it loses boundary contracts and can
turn an already validated assigned IR back into an ambiguous pre-assignment IR.

## First Implementation Boundary

The first implementation is deliberately limited:

```text
included:
  data value layout propagation
  value-level requests
  def-side and operand-level conflicts stored inside each value assignment
  strict same-operand conflict diagnostics
  same-layout data op transfer
  mask granularity assignment/split before layout propagation
  mask layout propagation
  CFG/block-argument same-transfer for control-flow values
  cast width relations needed by group-value cast/broadcast
  reuse of existing assignment type rewrite and data ensure_layout insertion

not included:
  function signature and call-site interprocedural rewrite
  a separate operand-request table outside value assignments
  cost-based layout choice
  best-effort request dropping
  global replacement of fold/rematerialize/sink passes
```

This boundary makes the design implementable without forcing all existing VMI
layout passes to move at once.

Acceptance requirement:

```text
Existing VMI lit and simulator regression outcomes must not regress after the
refactor.  Any test that passed before the propagator refactor must still pass
after it.  If a test's expected IR shape changes because the new propagation is
more canonical, update the expectation only with an explicit before/after
reason in the change description.
```

## Anti-Specialization Rules

The propagator must not grow optimization-pattern-specific boundary checks.
These forms are not acceptable:

```text
if producer is group_reduce and op is truncf and user is group_broadcast:
  choose layout X

if value is function argument and consumer is some specific op:
  insert special materialization Y
```

Boundary handling should be expressed through generic materialization support,
not producer-specific pattern checks inside the propagator:

```text
canMaterializeLayout(sourceType, resultType):
  delegate to VMILayoutSupport::getDataLayoutMaterializationSupport
```

Op-specific logic is still necessary, but it must be local to one op and one
role.  Do not combine several ops into one pattern:

```text
cast transfer:
  source/result width relation only

channel split/merge transfer:
  channel-count layout equation only

group_broadcast support/request:
  source operand group-value requirement and source/result support only

group_reduce seed:
  initial group-value result layout choice only

store request/support:
  required operand layout and store support only
```

With this structure, `group_reduce -> truncf -> group_broadcast` works because
independent op-local rules compose through `assignments`, not because the
propagator recognizes that whole chain.

## Example: Group-Value Cast

For a group-value cast relation:

```text
if the source value is requested/propagated as group-value:
  derive result group-value layout
  request the result value layout

if the result value is requested/propagated as group-value:
  derive inverse source value layout
  request the source value layout through the source operand overload
```

If the derived source layout conflicts with the source value's existing
`assignment.layout`, the operand overload records a use-side conflict in the
source value's `assignment.conflicts`.  Apply materializes it with
`ensure_layout` if the layouts still differ.  That conflict is not propagated
back through the producer.  Propagating an alternate layout through the producer
would mean rematerializing or cloning that producer for the alternate layout,
which is a separate optimization and must create a real forked value.

## Pre-Refactor Regression Baseline

Baseline captured on 2026-07-02 before introducing the shared VMI layout
propagator implementation.

```text
git HEAD:
  4a2a100f

working tree notes:
  unrelated pre-existing changes were present in 3rdparty/PTO-Gym and
  docs/designs/vmi-layout-lowering-cases.md.
  untracked local investigation files were also present and are not part of
  this baseline.
```

VMI lit baseline:

```bash
export PATH="/home/mouliangyu/projects/github.com/vpto-dev/llvm-project/build-shared/bin:$PATH"
python3 /home/mouliangyu/projects/github.com/vpto-dev/llvm-project/llvm/utils/lit/lit.py \
  -v -j16 build/test/lit/vmi_new
```

Result:

```text
Total Discovered Tests: 379
Passed: 379
Failed: 0
```

VMI simulator baseline:

```bash
WORK_SPACE=/tmp/ptoas-vmi-baseline-latest/sim \
CASE_PREFIX='vmi_new/' \
JOBS=16 \
test/vpto/scripts/run_host_vpto_validation_parallel.sh
```

Result:

```text
Total cases: 85
PASS: 81
FAIL: 4
```

Existing simulator failures in this baseline:

```text
vmi/group-reduce-s16-truncf-broadcast-store
vmi/group-reduce-s64-slot-add-store
vmi/group-reduce-s64-broadcast-reduce-store
vmi/group-reduce-s64-truncf-store
```

The refactor acceptance point is equality-or-better against this baseline:
all 379 VMI lit tests must keep passing, and the VMI simulator run must not add
new failing cases or turn any of the 81 passing cases into failures.
