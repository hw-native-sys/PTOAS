# VMI Layout Assignment Implementation Plan

本文是 `vmi-layout-assignment` 和 `vmi-to-vpto` 的实现计划。它配套
`vmi-layout-assignment-lowering-design.md`，并以
`vmi-layout-lowering-cases.md` 为测试和验收来源。

不使用早期 VMI 草稿作为设计输入。

## 1. Pipeline

Recommended pass pipeline:

```text
pto-validate-vmi-ir
  -> vmi-layout-assignment                  // hard legalization baseline
  -> canonicalize/cse
  -> vmi-layout-rematerialize               // optional optimization
  -> canonicalize/cse
  -> vmi-layout-fold              // optional optimization over remat-exposed helpers
  -> canonicalize/cse
  -> vmi-layout-sink-materialization        // optional optimization
  -> canonicalize/cse
  -> vmi-legalize-arith-select
  -> pto-validate-vmi-layout-ir
  -> vmi-to-vpto
  -> canonicalize/cse
  -> existing VPTO lowering/codegen
```

Only `vmi-layout-assignment` is required for the first legal implementation.
The optimization passes may be introduced one by one.  Their contract is that
they consume legal layout-assigned VMI IR and produce legal layout-assigned VMI
IR; they never move a hidden decision into `vmi-to-vpto`.

Pass responsibilities:

```text
pto-validate-vmi-ir:
  verify surface VMI has no physical VPTO layout dependency
  reject public/external VMI ABI unless explicitly enabled

vmi-layout-assignment:
  solve hard value layout constraints
  choose explicit layouts visible in IR
  insert ensure_layout / ensure_mask_layout / ensure_mask_granularity helpers
  make internal function boundary layouts explicit
  rewrite VMI types with layout attrs

canonicalize/cse:
  remove dead helpers and merge identical cloned producers where MLIR legality
  permits

vmi-layout-fold:
  fold use-site materialization into consumers that can directly consume the
  source layout while preserving the same logical effect
  example: ensure_layout(deinterleaved=2 -> contiguous) feeding store may become
  a store of deinterleaved=2 when the store has a layout-aware vstsx2 INTLV
  lowering
  current implementation: pto.vmi.store and the value operand of
  pto.vmi.masked_store when the existing mask arity matches, fed by
  ensure_layout from deinterleaved=2/4 to contiguous.  factor=2
  uses the store's vstsx2 INTLV lowering; factor=4 is still store-local, but it
  materializes through physical interleave before vsts.

vmi-layout-rematerialize:
  replace explicit ensure_* helpers with cloned cheap layout-polymorphic
  producers when the clone directly creates the requested result type
  current implementation: splat pto.vmi.constant, pto.vmi.broadcast,
  pto.vmi.iota, selected layout-transparent data ops, widening
  pto.vmi.ext{f,si,ui}, pto.vmi.create_mask, pto.vmi.create_group_mask, and
  pto.vmi.constant_mask.  Relation-aware remat rewrites result-side
  ensure_layout through layout-transparent producers and widening ext
  producers, leaving any newly exposed producer-side helpers for the following
  vmi-layout-fold.
  not included in the first implementation: load, group_load, masked_load,
  group_slot_load, and group_broadcast; those require separate memory,
  execution-count, or source-layout proof before they can be rematerialized

vmi-layout-sink-materialization:
  move ensure_layout across pure layout-transparent elementwise chains when the
  rewritten IR reduces materialization overhead and keeps every op locally legal
  current implementation: sink two identical operand ensure_layout helpers
  across binary add/sub/mul/div/min/max/and/or/xor/shl/shru VMI ops, three
  identical operand ensure_layout helpers across fma, or one source
  ensure_layout across unary neg/abs/sqrt/exp/ln/relu/not VMI ops, producing
  one result ensure_layout.  It also sinks compare data helpers to one result
  ensure_mask_layout, and sinks select only when both selected values and the
  mask carry matching explicit helpers.  Matching ensure_mask_layout or
  ensure_mask_granularity helpers are sunk across mask_and/mask_or/mask_xor/
  mask_not, producing one result mask helper.  It does not sink through cast,
  load, store, reduce, group_broadcast, or control-flow ops.

vmi-legalize-arith-select:
  restore scalar-condition arith.select with VMI result type back to scf.if
  after canonicalize; canonicalize may fold simple scf.if into arith.select,
  but VMI values must not cross non-VMI semantic ops before vmi-to-vpto

pto-validate-vmi-layout-ir:
  verify every VMI data/mask value has layout
  verify every VMI value has an assigned layout and every non-local lowering
  choice has been serialized explicitly
  verify helper ops have supported materialization paths.  Current
  implementation checks `ensure_layout`, `ensure_mask_layout`, and
  `ensure_mask_granularity` at the layout gate, so unsupported helper
  materializations fail before `vmi-to-vpto`.  It also checks the first
  semantic local lowering families, non-contiguous
  `pto.vmi.store`, block8
  `pto.vmi.group_load`, `pto.vmi.group_slot_load`, group_slots
  `pto.vmi.group_store`, group_slots `pto.vmi.group_reduce_add{f|i}`,
  explicit-slots `pto.vmi.group_broadcast`, `pto.vmi.truncf`,
  `pto.vmi.extf`, `pto.vmi.bitcast`, and histogram family ops at the layout gate.

vmi-to-vpto:
  use OneToN type conversion
  lower only from current-op attrs/operands, operand/result layouts, and helper
  ops
  emit VPTO or precise unsupported diagnostic
```

### 1.1 Hard Constraints Versus Optimizations

Hard legalization answers "can this program be lowered correctly?"  It is
allowed to be conservative:

```text
%w = pto.vmi.extf %a                 // natural layout deinterleaved=2
%t1 = pto.vmi.mulf %w, %k1           // layout-transparent, stays deinterleaved=2
%t1_c = pto.vmi.ensure_layout %t1    // hard store contract wants contiguous
pto.vmi.store %t1_c, %OUT1
%w_c = pto.vmi.ensure_layout %w
pto.vmi.store %w_c, %OUT2
```

This is a correct legal shape.  The contiguous action is explicit at each store
use, and `vmi-to-vpto` lowers the helper with register materialization such as
`vintlv` before ordinary `vsts`.

Optimization answers "can the same external effect be cheaper?"  A fold pass
may rewrite the two store uses to consume the deinterleaved values directly:

```text
pto.vmi.store %t1, %OUT1   // value type still says deinterleaved=2
pto.vmi.store %w,  %OUT2
```

This optimized shape is legal only because `pto.vmi.store` has enough local
information to lower a `deinterleaved=2` f32 value to row-major memory, for
example with `vstsx2 INTLV_B32`.  The optimization does not require
`vmi-to-vpto` to inspect `%w`'s producer or the sibling store.

The split gives later passes room to improve layout choices:

```text
hard pass:
  guarantee legality with explicit ensure_* helpers

optimization passes:
  remove, fold, clone, or sink helpers when the optimized IR is still locally
  deterministic

vmi-to-vpto:
  physicalize exactly the IR it sees, with no global planning
```

## 2. Files To Add Or Update

Expected implementation files:

```text
include/PTO/IR/VMITypes.td
include/PTO/IR/VMIOps.td
include/PTO/IR/VMIAttrs.td
lib/PTO/IR/VMI.cpp

include/PTO/Transforms/Passes.td
lib/PTO/Transforms/PTOValidateVMIIR.cpp
lib/PTO/Transforms/VMILayoutAssignment.cpp
lib/PTO/Transforms/VMIToVPTO.cpp
small layout fact/materialization helpers under lib/PTO/Transforms

test/lit/vmi_new/vmi_layout_assignment_*.pto
test/lit/vmi_new/vmi_to_vpto_*.pto
test/vpto/cases/vmi_new/*/
```

Exact names may follow project conventions, but the layering should remain:

```text
IR definitions
  -> validation
  -> assignment
  -> OneToN lowering
  -> lit and sim tests
```

## 3. IR Types And Attributes

### 3.1 Layout Attribute

Represent layout as a closed attribute family:

```text
#pto.vmi.layout<contiguous>
#pto.vmi.layout<deinterleaved = F>
#pto.vmi.layout<block_deinterleaved = F>
#pto.vmi.layout<num_groups = G, slots = K>
#pto.vmi.layout<num_groups = G, slots = K, lane_stride = LS>
```

C++ form:

```c++
enum class VMILayoutKind {
  Contiguous,
  Deinterleaved,
  BlockDeinterleaved,
  GroupSlots,
};

struct VMILayoutKey {
  VMILayoutKind kind;
  int64_t deinterleaveFactor = 1;
  int64_t numGroups = 0;
  int64_t slots = 0;
  int64_t laneStride = 1;
};
```

Verifier rules:

```text
contiguous:
  no extra parameters

deinterleaved:
  F > 1
  direct full-chunk lowerings require N % F == 0

block_deinterleaved:
  F > 1
  block size is fixed at 32B; element count is derived from the carrier type
  direct full-chunk lowerings require N % (F * blockElems(type)) == 0

group_slots:
  G > 0
  K > 0
  G % K == 0
  K fits in one physical vreg for element type
  LS > 0
```

Parser compatibility during migration:

```text
#pto.vmi.layout<num_groups = G, slots = K>
```

is the lowering contract for group-slot values.  The parser still accepts
`#pto.vmi.layout<num_groups = G>` as a legacy spelling for the pre-design
implicit group layout, but `vmi-to-vpto` support queries require explicit slots.
New `vmi-layout-assignment` output must print one of:

```text
#pto.vmi.layout<num_groups = G, slots = 8>
#pto.vmi.layout<num_groups = G, slots = 1>
#pto.vmi.layout<num_groups = G, slots = 8, lane_stride = 4>
```

so `vmi-to-vpto` can lower from the assigned type without reconstructing group
slot placement from producer or consumer context.

`lane_stride` is counted in logical element-sized physical slots and records a
regular gap between stored group slots.  The group-slot VMI value keeps its
logical element carrier; a packed store such as `PK4_B32` may use a wider
instruction-local carrier at the memory boundary.

The dense generalization is tracked separately in
`vmi-lane-stride-generalization-implementation.md`.  Dense and group-slot
lane-stride layouts now share logical-element physicalization, while retaining
their distinct lane maps and supported lowering strategies.

### 3.2 VMI Types

Surface:

```text
!pto.vmi.vreg<NxT>
!pto.vmi.mask<Nxpred>
```

Layout-assigned:

```text
!pto.vmi.vreg<NxT, #pto.vmi.layout<...>>
!pto.vmi.mask<Nxpred, #pto.vmi.layout<...>>
```

Surface VMI types are legal before assignment.  Layout-assigned VMI types are
required after assignment.

### 3.3 Explicit Lowering Carriers

Lowering decisions are carried by the current op and its types, not by a
separate lowering-plan string.  The allowed carriers are:

```text
op attrs and operands
operand/result VMI layouts
mask granularity and mask layouts
helper ops such as ensure_layout / ensure_mask_layout
cloned or rematerialized producers
diagnostics for unsupported shapes
```

If assignment made a non-local choice by inspecting producers, users, sibling
users, control flow, callees, or memory context, it must rewrite the IR so that
the final choice is visible through those carriers before `vmi-to-vpto`.

Local-decision table for the current implementation:

```text
op                         local decision inputs
group_load                 result layout, num_groups, row_stride, source type
group_slot_load            result group_slots layout and source_group_stride
group_reduce_add{f|i}      source/mask/result layouts, num_groups, typed reduce semantics
group_broadcast            source/result layouts and num_groups
truncf                     source/result layouts and element widths
dhist/chist                acc/source/mask/result layouts and target capability
ensure_layout              always carries source/result layouts
ensure_mask_layout         always carries source/result layouts
ensure_mask_granularity    always carries source/result granularities
```

Layout/attr-only decisions today:

```text
load                       result layout plus full chunk or shaped memref proof
group_store                source group_slots layout plus explicit output stride
masked_load                explicit passthrough, mask layout, and memory proof
masked_store/select        operand/result layouts plus mask granularity
dense extf/truncf          source/result layouts and element widths
```

Implementation rule:

```text
validate-assigned-vmi validates assigned layouts, mask granularity, boundaries,
and helper placement.
vmi-to-vpto emits VMI-LAYOUT-CONTRACT for missing local proof.
If a layout/attr-only op later gains a second legal lowering that cannot be
distinguished from current-op information, that lowering must be represented by a
new attr, helper op, or rematerialized op before vmi-to-vpto can emit it.
Unsupported shapes that have no explicit materialization/lowering path still
diagnose through their specific capability check rather than failing with a generic
missing-lowering
error.
```

Examples of forbidden recovery in `vmi-to-vpto`:

```text
group_reduce_add{f|i} cannot walk to a load/group_load producer to choose
  two-vlane parity versus block8.
group_store cannot inspect the group_reduce producer; it consumes only the
  assigned source layout and explicit stride.
group_broadcast cannot inspect sibling users to decide whether to rematerialize.
masked_load cannot inspect the mask producer to prove memory safety.
func.call cannot inspect the callee body to decide physical function layout.
```

## 4. VMI Surface Ops Required By Cases

Initial op set from the case catalog:

```text
load
group_load
group_slot_load
store
masked_store

create_mask
create_group_mask

extf
truncf
extsi
extui
trunci
addf
addi
mulf
select
broadcast

group_reduce_addf
group_reduce_addi
group_broadcast
group_store
dhist
chist

ensure_layout                 // internal
ensure_mask_layout            // internal
ensure_mask_granularity       // internal
```

Type policy before lowering:

```text
storage / memory boundary:
  f8-like, i8, f16, i16, f32, i32 may appear as load/store element types when
  the target memory instruction supports the physical width.

cast boundary:
  f8-like may appear as extf/truncf source or destination.
  i8 may appear as extsi/extui/trunci source or destination. Signedness is an
  op semantic, not a VMI type spelling.
  Current VPTO lowering supports 32-bit integer narrowing to unsigned i8
  storage, matching the available VCVTII s32/u32 -> u8 forms; signed i8
  narrowing needs a separate target lowering.

compute / accumulator:
  floating compute baseline: f16/f32, with reassoc required for reductions
  that lower through pair-wise VPTO reductions.
  integer compute baseline: i32 for grouped reduction; i8/i16 storage must
  first cast to i32 because integer reduction instructions widen narrow inputs.
  f8/i8 are not baseline accumulator/compute types. Supporting direct 8-bit
  compute requires a target capability entry and a separate lowering family.
```

Important semantic split:

```text
load:
  pointer sources must load full physical chunks directly.  Partial logical
  loads require a shaped memref proof or a future guarded/scratch fallback.

group_load:
  loads group_size data elements per group

group_slot_load:
  loads one scalar per group and produces group_slots
```

## 5. Layout Fact Helpers And Ensure-Based Optimization Hooks

Do not implement a target-aware lowering-plan registry shared by assignment and
lowering.  The shared contract is the IR itself: assigned VMI layouts, explicit
`ensure_layout` / `ensure_mask_layout` / `ensure_mask_granularity` helpers,
semantic op attrs/operands, and target capability diagnostics.

Small pure helpers are still useful when they remove duplicated layout math.
They must return semantic layout facts, not VPTO instruction plans, costs,
clone decisions, or multi-user plans.

Keep the support layer small.  A query belongs in `VMILayoutSupport` only when
at least two stages need the same fact and a mismatch would create an
assignment-vs-lowering bug.  Current valid shared facts are:

```text
cast layout fact:
  shared by layout assignment, layout validation, and vmi-to-vpto.
  Example: f32->f8 must see deinterleaved=4 source and contiguous result in
  every stage.

group_reduce layout fact:
  shared by layout assignment, layout validation, and vmi-to-vpto.
  Example: S=2*VLaneElems means deinterleaved=2 source/mask and
  group_slots(G, slots=8) result in every stage.

histogram layout fact:
  shared by layout assignment, layout validation, and vmi-to-vpto.
  Example: dhist requires contiguous Nxui8 source, contiguous b8 mask, and
  contiguous 256xui16 acc/result. chist uses the same layout fact but also
  requires a target capability that classifies CHISTv2 cumulative range
  semantics.

layout materialization support:
  shared by layout validation, vmi-to-vpto, and helper-based optimizations.
  Example: ensure_layout from deinterleaved=2 f32 to contiguous f32 is the same
  materialization whether it survives to lowering or is folded into a store.

contiguous store support:
  shared by fold-consumers and vmi-to-vpto because both must preserve the same
  row-major memory effect when consuming a non-contiguous value.
```

Do not add a support query for a single private branch such as "this exact op
uses this exact VPTO mnemonic".  Keep that branch in the lowering pattern until
another stage needs the same semantic fact.  This prevents `VMILayoutSupport`
from becoming a second copy of the lowering pass.

```c++
struct VMICastLayoutFact {
  VMICastLayoutKind kind;
  VMILayoutAttr sourceLayout;
  VMILayoutAttr resultLayout;
  int64_t factor;
};

struct VMIGroupReduceLayoutFact {
  VMILayoutAttr sourceLayout;
  VMILayoutAttr maskLayout;
  VMILayoutAttr resultLayout;
  int64_t groupSize;
  int64_t vlaneElems;
};

FailureOr<VMICastLayoutFact>
getPreferredCastLayoutFact(VMIVRegType sourceType, VMIVRegType resultType);

FailureOr<VMIGroupReduceLayoutFact>
getPreferredGroupReduceLayoutFact(VMIVRegType sourceType, int64_t numGroups);

LogicalResult canMaterializeDataLayout(VMIVRegType sourceType,
                                       VMIVRegType resultType,
                                       std::string *reason);
```

Baseline assignment uses these helpers only to produce assigned layouts and
use-site helpers.  It does not clone producers, rematerialize cheap ops, choose
memory-fused layouts by cost, or specialize private function signatures for
performance.

Optimization passes are deliberately helper-driven:

```text
fold-consumers:
  input shape: ensure_layout feeding a layout-aware consumer.
  support query: can this consumer preserve the same logical memory effect from
  the source layout?
  output shape: the consumer directly uses the source value.

rematerialize:
  input shape: cheap producer feeding ensure_layout / ensure_mask_layout.
  support query: can the cloned producer directly create the requested type?
  output shape: a cloned producer at the use.

sink-materialization:
  input shape: pure elementwise op whose operands are matching ensure_* helpers.
  support query: can the result helper be materialized if it remains?
  output shape: the op runs in the source layout and one helper remains on the
  result.
```

These passes may improve multi-consumer cases without asking assignment to solve
a global cost problem.  Assignment guarantees a legal baseline with helpers;
optimization removes or moves those helpers locally when the rewritten IR still
contains enough information for `vmi-to-vpto`.

Implementation-relevant layout facts:

```text
dense store:
  requests contiguous source.  If the value is assigned deinterleaved,
  assignment inserts ensure_layout at the store use.  A later optimization may
  fold ensure_layout + store into a layout-aware VMI store.  `vmi-to-vpto`
  later lowers that explicit store contract.

data/mask helper materialization:
  identity conversions are always legal.
  contiguous <-> deinterleaved=2/4 is legal only when source/result physical
  arity and physical chunk shapes make the same logical value materializable.
  unsupported conversions remain explicit diagnostics.

group_slot_load:
  assigned result layout is group_slots(G, slots=8) for packed slots or
  group_slots(G, slots=1) for row-local slots.  Because the result type is
  `GxT`, assignment does not derive this choice from result lane count.  A
  constant unit `source_group_stride` selects slots=8; non-unit or dynamic
  stride selects slots=1 first, then the support query rejects dynamic or
  unaligned row-local lowering when the target cannot materialize it.

block group_load:
  assigned result layout is block_deinterleaved=2/4 only when the op carries
  the required constant stride and memory-safety proof.  Its block is always
  32B, with the element count derived from the result type.

group_store:
  consumes group_slots(G,K).  Explicit output stride attrs/operands decide
  whether slots=8 packed or slots=1 row-local stores are legal.

group_reduce_add{f|i}:
  define E = sizeof(accumulator T), VLaneElems = 32B / E, L = 256B / E,
  S = N / G.  T is the accumulator/reduce element type after any required
  storage cast.
  S=VLaneElems uses contiguous source/mask and group_slots(G, slots=8).
  S=2*VLaneElems uses deinterleaved=2 source/mask and group_slots(G, slots=8).
  S=4*VLaneElems uses deinterleaved=4 source/mask and group_slots(G, slots=8).
  S>=L && S%L==0 uses contiguous source/mask and group_slots(G, slots=1).

group_broadcast:
  consumes group_slots(G,K) and produces one assigned dense layout.  If another
  consumer wants a different dense layout, assignment inserts ensure_layout.
  Optimization may clone/rematerialize group_broadcast per use.

extf/truncf:
  contiguous f16/bf16 -> deinterleaved=2 f32
  contiguous f8-like -> deinterleaved=4 f32
  deinterleaved=2 f32 -> contiguous f16
  deinterleaved=4 f32 -> contiguous f8-like
  group_slots(G, slots=1) f32 -> f16 remains a slot-preserving transform.

extsi/extui/trunci:
  contiguous i8/i16 -> deinterleaved i32 according to widening factor.
  deinterleaved i32 -> contiguous i8/i16 according to narrowing factor.
  packed group_slots integer width-changing cast is unsupported until a
  slot-wise transform is represented explicitly.

bitcast:
  per-part vbitcast is valid when source/result layouts match, physical arity
  matches, and every physical chunk carries the same logical bit footprint.
  This includes contiguous, deinterleaved, and identical group_slots layouts.
```

`vmi-layout-fold`, rematerialization, sink/hoist, and private
function specialization passes consume explicit helper IR.  They may replace
helpers with cheaper equivalent IR, but they must not introduce hidden lowering
plans that `vmi-to-vpto` has to rediscover from producer/user context.

## 6. Layout Assignment Data Model

### 6.1 Solver State

```c++
struct ValueLayoutState {
  Value value;
  Type logicalType;
  std::optional<VMILayoutKey> chosen;
  std::optional<VMILayoutKey> naturalLayout;
  SmallVector<UseRequest> useRequests;
};

struct UseRequest {
  OpOperand *operand;
  VMILayoutKey requestedLayout;
  Operation *requestingOp;
  bool hard;
};
```

### 6.2 Collection Phase

Walk the module and collect:

```text
1. every VMI value
2. every VMI block argument
3. every VMI function argument/result
4. every VMI op with natural producer layouts or use-site layout requests
5. every branch/yield/call/return edge carrying VMI
```

Build SCCs over:

```text
dataflow uses
region yields
loop iter_args
function call graph for private/internal functions
```

Public/external VMI function boundaries are rejected unless
`enablePublicVMIABI` is explicitly supported.

Block arguments are first-class layout variables.  Assignment must write the
chosen layout into the block argument type or specialized function signature.
`vmi-to-vpto` must never recover a block argument layout by walking to an
incoming branch, yield, or call operand.

### 6.3 Constraint Generation

Examples:

```text
truncf f32->f16:
  source request deinterleaved=2
  result contiguous

group_reduce S=16:
  source request deinterleaved=2
  result group_slots(G, slots=8)

group_reduce S=32:
  source request deinterleaved=4
  result group_slots(G, slots=8)

group_reduce S=64:
  source request contiguous
  result group_slots(G, slots=1)

group_broadcast:
  source request group_slots(G,K)
  result receives one assigned dense layout
  incompatible dense uses are represented by ensure_layout

ordinary dense add/mul/select:
  operands/results same dense layout

group-slot add/mul:
  operands/results same group_slots(G,K)

ordinary store:
  dense source required
  group_slots source is illegal

group_store:
  source request group_slots(G,K)

dhist:
  acc/result request contiguous 256xui16
  source request contiguous Nxui8
  mask request contiguous b8

chist:
  same layout requests as dhist
  diagnostic unless CHISTv2 cumulative range semantics are classified
```

Baseline assignment does not perform consumer-driven adoption for performance.
It records natural producer layouts and hard use-site requests.  If a request
does not match the assigned layout, the pass inserts an explicit helper at that
use.

```text
natural layout producer:
  extf/truncf, group_reduce, group_slot_load, group_load, dhist/chist when the
  op itself carries a layout-producing contract

layout equality producer:
  dense add/mul/select and CFG-carried values tie operands/results but do not
  pick a cheaper layout by cost
```

Memory legality constraints:

```text
S=32 tail fast load:
  requires full_footprint_readable
  otherwise require gather fallback or diagnose

compact S=12 logical S=16:
  requires compact-row gather materialization
  diagnose if gather fallback is disabled/missing
```

### 6.3.1 Request Builders

Implement request generation as small per-op builders.  The builders produce
natural layouts, use-site requests, equality constraints, and diagnostics; they
do not choose optimization plans.

```text
buildStoreRequests:
  ordinary store -> dense contiguous request
  group_store -> group_slots(G,K) request plus stride/alignment capability
  checks

buildCastRequests:
  extf f16->f32 -> source contiguous, result deinterleaved=2
  extf f8->f32  -> source contiguous, result deinterleaved=4
  truncf f32->f16 -> source deinterleaved=2, result contiguous
  truncf f32->f8  -> source deinterleaved=4, result contiguous
  group_slots slots=1 f32->f16 -> explicit slot-preserving transform
  group_slots slots=8 width-changing cast -> diagnostic unless a packed
  transform is explicitly represented

buildGroupReduceRequests:
  derive E = sizeof(accumulator type), VLaneElems = 32B / E,
  L = 256B / E, and S = logical_lanes / num_groups
  S=VLaneElems -> contiguous source, group_slots(G,8) result
  S=2*VLaneElems -> deinterleaved=2 source,
                    group_slots(G,8) result
  S=4*VLaneElems -> deinterleaved=4 source,
                    group_slots(G,8) result
  S>=L && S%L==0 -> contiguous source, group_slots(G,1) result
  8-bit storage must be cast to an accumulator type before this request builder
  other S -> diagnostic unless an explicit fallback op/helper is enabled

buildGroupMemoryRequests:
  group_load S=16/S=32 with aligned constant stride -> natural
  block_deinterleaved=2/4
  group_load row-local full chunks -> natural contiguous
  group_slot_load unit stride -> group_slots(G,8)
  group_slot_load aligned row-local stride -> group_slots(G,1)
  unsupported dynamic/unaligned grouped memory -> diagnostic

buildElementwiseRequests:
  dense add/mul/fma/min/max/select -> all dense operands/results share one
  dense layout
  group-slot add/mul/select -> all operands/results share one group_slots(G,K)
  dense/group_slots mixing -> diagnostic unless an explicit group_broadcast or
  group_store boundary exists

buildMaskRequests:
  mask layout follows each consuming data layout
  predicate granularity follows each consuming element type
  create_mask/create_group_mask produce one assigned mask layout and use
  ensure_mask_layout / ensure_mask_granularity for incompatible uses
  masked_store requests source layout, mask layout, and store predicate
  granularity explicitly

buildHistogramRequests:
  dhist -> acc/result contiguous 256xui16, source contiguous Nxui8,
           mask contiguous b8
  chist -> same layout requests, plus target capability diagnostic until
           CHISTv2 high-range semantics are classified
  do not create group_slots or group_reduce requests; histogram result bins are
  selected by source values, not by lane/group position

buildControlFlowRequests:
  region yields, branch operands, loop iter_args, call operands, and returns
  create equality requests on the carried VMI layout variable

buildFunctionBoundaryRequests:
  private/internal function argument/result layouts are materialized with
  callee-entry/return-site helpers in baseline assignment; signature
  specialization is an optimization pass
  public/external VMI arguments/results diagnose unless enablePublicVMIABI has
  a real ABI contract
```

Request builders must record the requesting op.  Diagnostics and inserted
helpers are use-site operations, so the user can see which consumer forced a
layout.

### 6.3.2 Optimization Producer Classes

Baseline assignment does not use producer classes to solve conflicts.  It
inserts helpers.  Later optimization passes may classify producers to replace
helpers with cheaper equivalent IR.

```text
cheap rematerializable producers:
  load when address operands dominate the clone site, no intervening may-alias
  write exists, and any shaped memory proof is preserved
  broadcast
  create_mask
  create_group_mask
  group_broadcast
  group_slot_load when the same address/no-alias/proof conditions as load hold
  and the memory access remains legal at the clone site

layout-transparent producers:
  add/sub/mul/fma/min/max/neg/abs
  select
  bitcast
  integer bitwise and shift ops

fixed-layout producers:
  extf/truncf physical conversion layouts
  group_load block-fragment layouts
  group_reduce result group_slots
  dhist/chist result contiguous 256xui16 and source/mask contiguous b8 contract
  masked_load when the physical memory-safety proof fixes a full-read lowering
```

Optimization conflict policy:

```text
cheap producer:
  clone for each incompatible request when cloning does not duplicate a
  side-effect, cross an aliasing write, or duplicate an illegal memory read

layout-transparent producer:
  merge into the consumer-requested equivalence class; insert materialization
  only at incompatible uses

fixed-layout producer:
  use explicit helper materialization only; otherwise diagnose
```

These classes are not assignment constraints.  They are rewrite preconditions
for passes that consume `ensure_layout` and decide whether the helper can be
folded, sunk, hoisted, or replaced by rematerialization.

### 6.4 Solving And Rewriting

Algorithm:

```text
1. Collect natural layouts, use-site requests, equality constraints, and
   memory-safety proofs.
2. Propagate equality constraints through SCCs.
3. Choose one deterministic assigned layout per value/equivalence class:
   explicit user layout, then unique producer natural layout, then hard
   non-contiguous layout, then contiguous.
4. For conflicting uses, insert ensure_layout / ensure_mask_layout /
   ensure_mask_granularity at the use.
5. Emit diagnostics for unsupported semantic constraints or missing explicit
   materialization/memory-safety proof.
6. Rewrite VMI result/block/function types with chosen layouts.
7. Insert helper ops with source/result layout attrs.
```

Rewrite invariants:

```text
No VMI data/mask value after assignment has a null layout.
Any non-local choice is represented by op attrs, operand/result layouts, a
helper op, or an explicit diagnostic.  Cloned/rematerialized producers may
appear only after later layout optimization passes.
Every ensure_* helper has an explicit supported materialization path or a
diagnostic.
Every function/call boundary carrying VMI is materialized, kept in an explicit
ABI contract, or diagnosed.
```

### 6.5 Rewrite Artifacts

Assignment rewrites the IR so that later lowering has no hidden choices.

```text
type rewrite:
  every VMI data/mask result and block argument receives a layout attr

ensure rewrite:
  mismatched uses get pto.vmi.ensure_layout or ensure_mask_layout at the use
  site, with source and target layouts visible in the types

granularity rewrite:
  one semantic mask used by f32 and f16 consumers gets
  ensure_mask_granularity at the use site

control-flow rewrite:
  scf.if/scf.for yields and block arguments are rewritten to one agreed layout;
  materialization is inserted before yield when branches differ

function rewrite:
  baseline private VMI functions get callee-entry/return-site ensure_layout;
  signature specialization is an optimization pass
  public/external VMI functions are diagnosed
```

Canonical assigned IR shape for a conflicting load:

```text
%x = pto.vmi.load ...
  : ... -> !pto.vmi.vreg<256xf32, #pto.vmi.layout<deinterleaved = 4>>

%x_dense = pto.vmi.ensure_layout %x
  : !pto.vmi.vreg<256xf32, #pto.vmi.layout<deinterleaved = 4>>
 -> !pto.vmi.vreg<256xf32, #pto.vmi.layout<contiguous>>

pto.vmi.store %x_dense, ...
```

Optional future optimized IR shape for a cloned load with an explicit
safe-read/execution proof:

```text
%x_s16 = pto.vmi.load ...
  : ... -> !pto.vmi.vreg<256xf32, #pto.vmi.layout<deinterleaved = 2>>

%x_s32 = pto.vmi.load ...
  : ... -> !pto.vmi.vreg<256xf32, #pto.vmi.layout<deinterleaved = 4>>
```

Canonical assigned IR shape for `group_broadcast` multi-use:

```text
%b = pto.vmi.group_broadcast %slots
  : !pto.vmi.vreg<8xf32, #pto.vmi.layout<num_groups = 8, slots = 8>>
 -> !pto.vmi.vreg<256xf32, #pto.vmi.layout<deinterleaved = 4>>

%b_c = pto.vmi.ensure_layout %b
  : !pto.vmi.vreg<256xf32, #pto.vmi.layout<deinterleaved = 4>>
 -> !pto.vmi.vreg<256xf32, #pto.vmi.layout<contiguous>>
```

If the assigned IR does not have one of these explicit shapes, `vmi-to-vpto`
must reject it instead of attempting to recover the missing decision.

### 6.6 Case-To-Implementation Closure Matrix

The current case catalog is sufficient for the first implementation.  No new
layout kind is justified by the supported endpoints.  The implementation work
should instead close the following finite matrix.  Each row names the request
builder that owns the decision, the assignment artifact that must appear in IR,
and the `vmi-to-vpto` contract.

```text
case family                     builder / owner             assignment artifact
3.1, 3.2, 3.3 dense casts       buildCastRequests           dense layout on each cast result
3.29 mask width split           buildMaskRequests           per-use mask granularity helper
3.31, 3.32 dense fanout         conflict resolver           cloned load or ensure_layout

vmi-to-vpto contract:
  consume only the assigned dense layouts.  It may emit VCVT and dense
  materialization, but it must not choose deinterleaved=2/4 by inspecting a
  later truncf, store, or group_reduce user.
```

```text
case family                     builder / owner             assignment artifact
3.4 32-bit S=8 reduce           buildGroupReduceRequests    one_vlane contiguous lowering
3.5 32-bit S=16 reduce          buildGroupReduceRequests    two_vlane parity/block8 layout
3.6 32-bit S=32 reduce          buildGroupReduceRequests    four_vlane dintlv4/block8 layout
3.7 32-bit S=64 reduce          buildGroupReduceRequests    full_chunk row_local lowering
3.11.1 S=64 active-row tail     buildMaskRequests           active-row store/reduce masks
3.19.1 S=16 split-kind choice   buildGroupReduceRequests    explicit dense layout kind
3.38 multi-tile S=32 reduce     buildGroupReduceRequests    multiple group_slots chunks
3.26 grouped tail               buildMaskRequests           split grouped masks
3.44, 3.45 grouped S=32 masks   buildMaskRequests           explicit deinterleaved mask values

vmi-to-vpto contract:
  lower each reduce from the current op's attrs, source/mask layout, result
  group_slots layout.  It must not walk to the load/group_load producer to
  decide parity versus block8, row-local versus packed slots, or static versus
  dynamic mask generation.
```

```text
case family                     builder / owner             assignment artifact
3.56 full distribution hist     buildHistogramRequests      contiguous src/mask/acc/result
3.57 cumulative hist boundary   buildHistogramRequests      capability diagnostic or classified path

vmi-to-vpto contract:
  lower dhist from the current op and assigned layouts by carrying two physical
  accumulator parts for bins 0..127 and 128..255.  It must not expose the VPTO
  #bin range selector on the VMI surface and must not model histogram as
  group_reduce.  chist remains rejected until the target records whether the
  high-range cumulative result is global or range-local and, for range-local
  behavior, until low-total materialization is explicit.
```

```text
case family                     builder / owner             assignment artifact
3.15.1 S=16 row stride 16       buildGroupMemoryRequests    block_deinterleaved group_load
3.15.2 S=16 row stride > 16     buildGroupMemoryRequests    strided block layout plan
3.16.1 group_slot_load slots=8  buildGroupMemoryRequests    unit-stride packed slots plan
3.16.2 group_slot_load slots=1  buildGroupMemoryRequests    row-local aligned slots plan
3.27 strided group_load         buildGroupMemoryRequests    block_deinterleaved plan
3.28 slots=1 non-unit load      buildGroupMemoryRequests    row-local group_slot_load layout
3.37 slots=1 strided store      buildStoreRequests          group_store stride/alignment proof
3.39 strided load fanout        conflict resolver           preserving layout or materialization

vmi-to-vpto contract:
  consume only explicit memory stride/alignment attrs, current op operands,
  and layouts.  It must not infer safe read/write placement from neighboring
  compute ops.  Unsupported dynamic, unaligned, or compact-row gather shapes
  stay diagnostics until a gather fallback is explicit in the current op.
```

```text
case family                     builder / owner             assignment artifact
3.8 reduce->truncf->broadcast   conflict resolver           slot cast plus dense materialization
3.10 non-load S=32 producer     buildElementwiseRequests    transparent deinterleaved chain
3.17 broadcast deint consumer   conflict resolver           use-site group_broadcast layout
3.18 dense + reduce users       conflict resolver           ensure_layout; optional remat/fold
3.23 broadcast multi-user       conflict resolver           per-op group_broadcast layout
3.33 S=16 + S=32 users          conflict resolver           use-site materialization; optional cloned load
3.34 S=64 slots=1 cast          buildCastRequests           group_slot_cast layout
3.35 slots fanout               buildElementwiseRequests    same group_slots layout on users
3.36 scalar slots=8/slots=1     conflict resolver           explicit slots=8/slots=1 producers
3.40 scalar dense + grouped     conflict resolver           ensure_layout; optional broadcast remat
3.41 incompatible fixed value   conflict resolver           diagnostic or ensure_layout

vmi-to-vpto contract:
  each op instance is already single-plan.  The lowering pass never scans
  sibling users to decide whether to clone, pack, broadcast, or materialize.
```

```text
case family                     builder / owner             assignment artifact
3.21 S=32 rounded tail mask      buildMaskRequests           rounded vector plus mask
3.24 mask/select/store          buildMaskRequests           explicit mask layout/granularity
3.12 scf.if before reduce       buildControlFlowRequests    common yielded layout
3.20 group_slots scf.if         buildControlFlowRequests    common group_slots layout
3.22 scf.for carried value      buildControlFlowRequests    fixed-point iter_arg layout
3.25 function boundary          buildFunctionBoundary       specialized/internal boundary
3.42 loop accumulator           buildControlFlowRequests    loop-carried group_slots layout
3.43 call argument materialize  buildFunctionBoundary       callee-entry/return helper

vmi-to-vpto contract:
  block argument, region result, call operand, and function result layouts are
  visible in types or helper ops.  It must not inspect branch bodies, loop
  bodies, callers, or callees to discover a layout.
```

```text
diagnostic family               builder / owner             required failure
3.7.4 slots=1 unit-stride store buildStoreRequests          no aligned row-local store path
3.9 dense store of group slots  buildStoreRequests          use group_store/group_broadcast
3.11.2 S=32 unsafe tail         buildMaskRequests           missing full_footprint_readable/gather
3.13 slots=8 width cast         buildCastRequests           no packed slot cast transform
3.14 unsupported group size     buildGroupReduceRequests    no supported reduce layout/lowering
3.15.3 compact S=12            buildGroupMemoryRequests    no compact gather plan
3.16.1 slots=8 non-unit load    buildGroupMemoryRequests    no packed strided slot load path
3.16.2 slots=1 bad stride       buildGroupMemoryRequests    no dynamic/unaligned row-local plan
3.19.2 incompatible split kind  conflict resolver           no preserving materialization
3.25.2 public/external ABI      buildFunctionBoundary       no stable public VMI ABI
3.27 unaligned group_load       buildGroupMemoryRequests    no gather/block fallback path
3.30 masked_load unsafe tail    buildMaskRequests           no padding/gather fallback

vmi-to-vpto contract:
  these cases must fail before or at the layout contract boundary with the
  requesting op named.  They must not be accepted by falling back to a generic
  dense load, dense store, or producer/user inspection.
```

Additional cases are needed only when the scope changes:

```text
stable gather fallback enabled:
  add compact S=12 positive lowering and masked_load unsafe-tail positive
  lowering before accepting either path.

pack-to-slots=8 or unaligned row-local stores enabled:
  add positive S=64 unit-stride group_store and reduce->pack->dense store cases.

public VMI ABI enabled:
  add public call/return ABI cases before removing the public-boundary
  diagnostic.

packed group-slot width cast enabled:
  add slots=8 f32->f16 cast and downstream group_store/broadcast cases.
```

## 7. OneToN Type Conversion

`vmi-to-vpto` should use OneToN conversion for VMI values.

Conversion rules:

```text
contiguous:
  ceil(N / lanesPerVReg(T)) physical vregs

deinterleaved=F:
  F * ceil((N / F) / lanesPerVReg(T)) physical vregs
  ordering: part-major, then chunk

group_slots(G,K):
  ceil(G / K) physical vregs
  each vreg has logical slot lanes 0..K-1 live
```

Mask conversion:

```text
mask layout follows data layout
mask granularity is selected from consumer element width:
  f32/i32 -> b32
  f16/i16 -> b16
  f8/i8   -> b8
```

If one logical mask is used by multiple widths, assignment inserts
`ensure_mask_granularity` or rematerializes the mask producer.

## 8. VMI-to-VPTO Pattern Rules

Each pattern uses:

```text
op
op attrs and operand values
operand/result layouts
adaptor physical values
```

Each pattern rejects:

```text
missing current-op proof for an otherwise unsafe memory lowering
missing target capability
unexpected group_slots dense consumer
```

Target local lowering matrix:

```text
load, lowering=dense_load_norm:
  result layout contiguous
  emits pto.vlds / pto.vsts NORM paths
  covers dense store users and full-chunk row-local reduce input

load, lowering=load_dintlv2:
  result layout deinterleaved=2
  emits vldsx2 DINTLV_B32 or normal load + vdintlv materialization
  covers f32->f16, S=16 parity reduce, f16->f32 widened values

load, lowering=load_dintlv4:
  result layout deinterleaved=4
  emits two vldsx2 DINTLV_B32 plus vdintlv
  covers f32->f8, S=32 dintlv4 reduce

group_load, lowering=s16_group_load_block8_unit_stride:
  result layout block_deinterleaved=2
  emits vldsx2/BDINTLV for 8 rows of 16xf32
  covers compact logical S=16 when source_group_stride == 16

group_load, lowering=s16_group_load_block8_stride:
  result layout block_deinterleaved=2
  emits two vsldb strided 32B block loads
  requires source_group_stride % 8 == 0

group_load, lowering=s32_group_load_block8_stride:
  result layout block_deinterleaved=4
  emits four vsldb strided 32B block loads
  requires source_group_stride % 8 == 0

group_load, lowering=group_load_contiguous_chunks:
  result layout contiguous
  emits one vlds per physical group chunk using row_stride address arithmetic
  covers the currently implemented full-chunk row-local group_load path

group_reduce_add{f|i}, lowering=one_vlane_reduce_contiguous:
  consumes contiguous accumulator type T with group size VLaneElems(T)
  produces group_slots(G, slots=8)
  emits one vcgadd

group_reduce_add{f|i}, lowering=two_vlane_reduce_deinterleaved:
  consumes deinterleaved=2
  produces group_slots(G, slots=8)
  emits two vcgadd operations and one vadd

group_reduce_add{f|i}, lowering=two_vlane_reduce_block8:
  consumes block_deinterleaved=2
  produces group_slots(G, slots=8)
  emits two vcgadd operations and one vadd

group_reduce_add{f|i}, lowering=four_vlane_reduce_dintlv4:
  consumes deinterleaved=4
  produces group_slots(G, slots=8)
  emits four vcgadd operations and a vadd tree

group_reduce_add{f|i}, lowering=four_vlane_reduce_block8_stride:
  consumes block_deinterleaved=4
  produces group_slots(G, slots=8)
  emits four vcgadd operations and a vadd tree

group_reduce_add{f|i}, lowering=full_chunk_reduce_row_local:
  consumes contiguous accumulator type T with group size that is a multiple of
  one physical chunk L(T)
  produces group_slots(G, slots=1)
  target lowering emits per-row vcgadd plus vcadd; the current prototype uses
  the existing row-local VCADD/VADD/VSEL sequence while preserving the same
  group_slots(G, slots=1) value contract

dhist, lowering=full_256bin_histogram:
  consumes contiguous Nxui8 source and contiguous b8 mask
  consumes/produces contiguous 256xui16 accumulator/result
  physical result parts are [bins 0..127, bins 128..255]
  emits one low-range and one high-range histogram update for each 256-lane
  source chunk
  final partial source chunks require an explicit valid-lane b8 mask

chist, lowering=capability_gated_cumulative_histogram:
  uses the same layout shape as dhist
  rejects until target capability classifies CHISTv2 high-range cumulative
  semantics and any required low-total correction materialization is explicit

group_slot_load, lowering=group_slot_load_slots8_unit_stride:
  result group_slots(G, slots=8)
  requires source_group_stride == 1
  emits one packed vsldb load

group_slot_load, lowering=group_slot_load_slots1_row_local:
  result group_slots(G, slots=1)
  supports aligned non-unit source_group_stride
  requires constant positive source_group_stride divisible by 256 / elementBits
  emits one lane-0 vsldb per group

group_broadcast, lowering=group_broadcast_slots8_vselr:
  source group_slots(G, slots=8)
  result dense layout selected per use
  emits vselr using assigned result layout

group_broadcast, lowering=group_broadcast_slots1_vselr:
  source group_slots(G, slots=1)
  result dense layout selected per use
  emits vdup/vselr row-local materialization

truncf, lowering=group_slot_cast_slots1_f32_to_f16:
  source/result group_slots(G, slots=1)
  emits one lane-0 vcvt per group slot block
  rejects packed slots=8 unless slot-preserving cast support exists
```

The target matrix is the implementation contract.  The staged status below
records how much of that contract the current prototype has already enforced.

Current staged implementation status:

```text
group_slot_load:
  vmi-to-vpto lowers from #pto.vmi.layout<num_groups = G, slots = 8/1>
  and source_group_stride.

group_reduce_addf:
  explicit slots=8 VCGADD lowering is selected from contiguous source/mask
  layout, slots=8 result layout, num_groups, and reassoc.
  S=16 block8 assignment emits source/mask
  #pto.vmi.layout<block_deinterleaved=2>, result
  #pto.vmi.layout<num_groups = G, slots = 8>; vmi-to-vpto lowers through two
  VCGADDs plus a PAT_VL8 VADD per packed result block.
  S=32 block8 assignment emits source/mask
  #pto.vmi.layout<block_deinterleaved=4>, result
  #pto.vmi.layout<num_groups = G, slots = 8>; vmi-to-vpto lowers through four
  VCGADDs plus a PAT_VL8 VADD tree per packed result block.
  Full-chunk row-local assignment, including S=64 and S=256 f32 cases, uses
  #pto.vmi.layout<num_groups = G, slots = 1> and has focused
  layout-assignment/vmi-to-vpto lit coverage; the explicit slots=1 generic
  VCADD row-local lowering is selected locally from the current op attrs and
  assigned layouts.
  group_reduce_addi is implemented for i8/i16/i32 values over the registered
  high-performance group-block classes. VCGADD paths preserve the logical
  element type. Full-chunk row-local paths use widening VCADD intermediates
  internally and bitcast the final low bits back to the declared VMI result
  type; widening is not part of the VMI contract.

group_broadcast:
  explicit slots=8/1 source layouts select
  packed or row-local VSELR lowerings locally. Deinterleaved block-fragment
  results derive the 32B block width from the result type as the local vselr
  selection group, so
  `block_deinterleaved=4` broadcasts one group slot across each
  32B row fragment. VSELR index vectors are materialized per physical result
  chunk.  For small-group results, layout assignment has already fixed the
  result layout, and vmi-to-vpto computes:
  `firstGroup = first logical group covered by this result chunk`,
  `sourceChunk = firstGroup / slots`, and
  `baseGroupSlot = firstGroup % slots`.  The generated index vector selects
  `baseGroupSlot .. baseGroupSlot + groupsPerResultChunk - 1`; it must not be
  reused across result chunks.

group_load:
  contiguous full-chunk path is selected from a contiguous result layout.
  S=16/S=32 block-aligned strided loads are selected from
  #pto.vmi.layout<block_deinterleaved=2/4>, and lower to one
  vsldb per 32B row fragment and physical chunk.  The explicit block8 support
  is checked by pto-validate-vmi-layout-ir before vmi-to-vpto.
  The dedicated S=16 unit-stride vldsx2/BDINTLV lowering remains a local
  peephole target.
  S=16/S=32 group_load with a non-constant, non-positive, or non-8-f32-aligned
  row_stride is rejected by vmi-layout-assignment because the stable gather
  fallback is not implemented.

truncf group-slot cast:
  layout assignment and vmi-to-vpto support group_slots(G, slots=1)
  f32 -> f16 from source/result layouts and element widths. The reduce->truncf
  -> group_store slots=1 flow has focused lit coverage and no longer relies on
  vmi-to-vpto inspecting the truncf producer.

group_store:
  row-local group_slots(G, slots=1) lowering is implemented as one lane-0
  vsts per group for packed unit-stride output, or as one 1PT store per group
  for non-unit row strides. The packed path is covered by the
  reduce->truncf->group_store lit case, while the point-store path is covered
  by `test/lit/vmi_new/vmi_to_vpto_group_store_slots1_1pt.pto`.
  Packed group_slots(G, slots=8) group_store is implemented only when
  num_groups is a multiple of 8 and row_stride is constant 1; it emits one
  PAT_VL8 store per packed slot block. Non-unit packed group stores remain a
  design target unless a strided packed-lane store lowering is made explicit.
```

Current implementation contract for type-generic grouped reduction:

```text
ODS/verifiers:
  pto.vmi.group_reduce_addi is the integer counterpart to group_reduce_addf.
  group_reduce_addi, group_reduce_maxi, and group_reduce_mini accept
  i8/i16/i32 element types when the group shape matches a registered layout
  table row.
  extsi/extui/trunci carry integer signedness across storage/accumulator
  boundaries when an algorithm explicitly wants a wider accumulator.

Layout assignment:
  compute VLaneElems and L from the accumulator/reduce element type:
    VLaneElems = 32B / sizeof(accumulator T)
    L          = 256B / sizeof(accumulator T)
  use the same S formula for f16/f32/i8/i16/i32 once the typed reduce op and target
  capability say the type is legal.
  route f8 storage through extf to f32 before group_reduce_addf.
  keep direct i8/i16 integer reductions in their declared logical type;
  extsi/extui remains available for explicitly widened algorithms.
  route integer narrowing to i8 through trunci; direct i8 compute remains
  illegal unless target capability and explicit op semantics define that
  lowering.
  diagnose direct f8/i8 compute use with a message that points at the offending
  op and suggests inserting the explicit cast when the op is meant to consume
  storage data.

Layout fact helpers:
  replace f32-shaped checks with width-parametric group-reduce classifiers:
    one_vlane_reduce layout fact
    two_vlane_reduce_deinterleaved layout fact
    four_vlane_reduce_deinterleaved layout fact
    full_chunk_row_local_reduce layout fact
  key legality on accumulator byte width, source/mask layout, result
  group_slots layout, num_groups, and target instruction capability.

VMI-to-VPTO:
  lower group_reduce_addi through the same VCGADD/VADD skeleton used for
  floating-point where the target supports the integer accumulator type.
  for full-chunk i8/i16 rows, use the widening VCADD result only as an internal
  partial type, combine partials at that width, then bitcast back to the
  declared slots=1 VMI result type.
  keep VPTO lowering local: it consumes assigned layouts and current-op
  attrs/operands, but does not invent a new global layout plan.

Tests:
  cover direct i8/i16/i32 grouped reductions and explicitly widened variants.
  add i32 S=8/S=16/S=32/S=64 group-reduce cases.
  add f8 storage -> extf -> f32 group_reduce_addf cases.
  add i8/i16 full-chunk VCADD plus bitcast cases.
  retain invalid f8 and unsupported group-shape diagnostics.
```

Examples:

```text
group_reduce_add{f|i}, lowering=two_vlane_reduce_deinterleaved:
  consume deinterleaved=2
  emit two VCGADDs and one VADD

group_reduce_add{f|i}, lowering=two_vlane_reduce_block8:
  consume block_deinterleaved=2
  emit two VCGADDs and one VADD

group_reduce_add{f|i}, lowering=four_vlane_reduce_dintlv4:
  consume deinterleaved=4
  emit four VCGADDs and reduction tree

group_broadcast:
  consume group_slots
  emit VSELR or VDUP depending slots and target dense layout

group_slot_load slots=8:
  emit one packed block load for unit stride

group_slot_load slots=1:
  emit row-local lane-0 loads for constant positive 32B-aligned strides
```

## 9. Validation Passes

### 9.1 Surface Validation

Before assignment:

```text
VMI types may omit layout.
VPTO physical op must not consume VMI values.
Public/external VMI function ABI rejected unless enabled.
Unsupported vector-to-scalar extract rejected.
```

### 9.2 Layout Validation

After assignment:

```text
Every VMI value has layout.
Every VMI mask has layout and granularity plan.
Every lowering choice is locally deterministic or explicit in attrs/layouts.
Every ensure_* helper has a materialization path.
Every control-flow edge has matching VMI layouts.
```

### 9.3 `vmi-to-vpto` Context Read Audit

`vmi-to-vpto` may still read defining ops in narrowly scoped cases that do not
select a layout or plan:

```text
allowed:
  arith.constant for the current op's scalar operands
  create_mask/create_group_mask internals when lowering that mask op itself
  ensure_mask_layout / ensure_mask_granularity stripping for static mask facts
  memref.subview only to improve an already-failed non-identity memref
  diagnostic

not allowed:
  walking from a consumer to a producer to decide a lowering
  walking from a consumer to a mask producer to decide whether a lowering is legal
  inspecting users to choose a result layout or materialization
  recovering full_footprint_readable from surrounding MTE/caller context
```

Current audit result:

```text
3.44 partial S=32 create_group_mask:
  assignment writes explicit contiguous and deinterleaved mask values.  When
  lowering the deinterleaved create_group_mask itself, vmi-to-vpto first
  materializes contiguous grouped predicate chunks and then applies predicate
  pdintlv in the same tree shape as the data vdintlv.  It still does not walk
  from group_reduce_addf to the mask defining op to choose or reject lowering.
  The dynamic active_elems_per_group form is also op-local: vmi-to-vpto lowers
  contiguous chunks with vci/vshrs/vshls/vsub/vcmps, then uses the same
  predicate pdintlv tree for S=32 deinterleaved masks.

masked_load:
  direct lowering is load + vsel.  It does not inspect the mask producer to
  choose a different load form; memory safety is provided by full physical
  chunks or shaped memref proof.

memref.subview:
  mentioned only after identity lane-to-address planning fails.  It is not used
  to recover a hidden base/stride lowering.
```

## 10. Diagnostics

Implement diagnostics with stable prefixes:

```text
VMI-LAYOUT-CONTRACT
VMI-UNSUPPORTED-PLAN
VMI-MISSING-CAPABILITY
VMI-PUBLIC-ABI
VMI-MASK-GRANULARITY
VMI-CONTROL-FLOW-LAYOUT
```

Minimum diagnostic payload:

```text
op name
logical type
actual layout
requested layout
selected/missing support path
recommended rewrite or option
```

Example:

```text
VMI-LAYOUT-CONTRACT:
  pto.vmi.truncf requires
  #pto.vmi.layout<deinterleaved=2>, but the source value is
  fixed to #pto.vmi.layout<block_deinterleaved=2> by the selected
  strided group_load layout. Register a rematerialization or preserving
  materialization path, or avoid consuming this block-loaded value with truncf.
```

## 11. Test And Simulator Acceptance

Each numbered endpoint in `vmi-layout-lowering-cases.md` should become:

```text
1. a layout-assignment lit test
2. a vmi-to-vpto lit test
3. a simulator case when the VPTO sequence is supported by the current backend
4. a diagnostic lit test when the case is explicitly unsupported
```

Repository locations:

```text
test/lit/vmi_new/
test/vpto/cases/vmi_new/
```

The current repository uses descriptive flat lit names rather than
case-numbered subdirectories.  New tests should follow the existing prefixes:

```text
vmi_layout_assignment_<case>.pto
vmi_to_vpto_<case>.pto
<runtime-case-name>/kernel.pto
```

The case number should still be recoverable from the coverage table in this
document and from the corresponding section in `vmi-layout-lowering-cases.md`.

### 11.1 Layout Assignment Checks

Each positive layout-assignment test must check:

```text
assigned data layouts
assigned mask layouts
assigned op attrs
direct vmi-to-vpto local lowering
inserted ensure_layout/rematerialized producers
control-flow/function signature specialization
```

Negative tests check diagnostic text.

### 11.2 VMI-to-VPTO Checks

Each positive vmi-to-vpto test must check:

```text
no pto.vmi ops remain
VPTO op sequence matches the case lowering
physical value arity and ordering are correct
mask granularity is correct
stores preserve observable logical memory order
```

### 11.3 Simulator Checks

Simulator cases should compare final memory against the memory result written in
the case catalog.

Current broad runtime sweep:

```text
WORK_SPACE=$PWD/.tmp/vmi-runtime-batch-final CASE_PREFIX='vmi_new/' JOBS=4 \
  test/vpto/scripts/run_host_vpto_validation_parallel.sh

TOTAL_CASES=47
PASS=47 FAIL=0
summary: .tmp/vmi-runtime-batch-final/parallel-summary.tsv
result: all summary entries are PASS
```

The `find: Permission denied` messages printed while discovering CANN simulator
paths are environment noise and are not treated as simulator failures.

Required groups:

```text
dense conversion:
  3.1, 3.2, 3.3, 3.31, 3.32

group reduce:
  3.4, 3.5.1, 3.5.2, 3.5.3
  3.6.1, 3.6.2, 3.6.3
  3.7.1, 3.7.2, 3.7.3
  3.7.4 diagnostic

layout/rematerialization:
  3.8, 3.10, 3.17, 3.18, 3.19.1, 3.22, 3.23, 3.31,
  3.32, 3.33, 3.34, 3.35, 3.36, 3.38, 3.40, 3.41

mask/tail:
  3.11.1, 3.15.1, 3.15.2, 3.21, 3.24, 3.26, 3.29,
  3.30, 3.44, 3.45

strided/group-slot memory:
  3.27, 3.28, 3.37, 3.39

function/control-flow:
  3.12, 3.20, 3.22, 3.25.1, 3.42, 3.43

histogram:
  3.56 positive dhist layout/lowering and simulator case when backend support
  is enabled
  3.57 diagnostic chist case until CHISTv2 range semantics are classified
```

Aggregate catalog headings are covered through their endpoint subcases:

```text
3.11 partial tail groups:
  3.11.1 positive S=64 active-row tail
  3.11.2 diagnostic S=32 tail without full_footprint_readable

3.15 compact S=12 written as logical S=16:
  3.15.1 positive source row stride 16
  3.15.2 positive source row stride greater than 16
  3.15.3 diagnostic compact source row stride 12

3.16 group_slot_load layout contract:
  3.16.1 packed slots=8 positive and non-unit-stride diagnostic
  3.16.2 row-local slots=1 positive plus dynamic/unaligned diagnostics

3.25 function boundary layout specialization:
  3.25.1 private/internal boundary lit and runtime coverage
  3.25.2 public/external boundary diagnostics
```

Current coverage audit result:

```text
SIM-backed positive endpoints:
  3.1, 3.2, 3.3, 3.4, 3.5.1, 3.5.2, 3.5.3,
  3.6.1, 3.6.2, 3.6.3, 3.7.1, 3.7.2, 3.7.3,
  3.8, 3.10, 3.11.1, 3.12, 3.15.1, 3.15.2,
  3.16.1 positive, 3.16.2 positive, 3.17, 3.18,
  3.19.1, 3.20, 3.21, 3.22, 3.23, 3.24, 3.25.1, 3.26,
  3.27 positive, 3.28 positive, 3.29, 3.31, 3.32,
  3.33, 3.34, 3.35, 3.36, 3.37, 3.38, 3.39,
  3.40, 3.41, 3.42, 3.43, 3.44, 3.45

diagnostic endpoints:
  3.7.4, 3.9, 3.11.2, 3.13, 3.14, 3.15.3,
  3.16.1 non-unit slots=8 source stride,
  3.16.2 dynamic/unaligned slots=1 source stride,
  3.19.2, 3.25.2, 3.27 unaligned source_group_stride,
  3.30 unsafe masked_load tail

repository evidence:
  all concrete lit/runtime paths listed below exist
  all 47 runtime case directories contain kernel.pto, launch.cpp, main.cpp,
  golden.py, and compare.py
  latest broad VMI runtime sweep passed: PASS=47 FAIL=0
  latest full VMI lit sweep passed: 350/350
  this historical sweep predates 3.56/3.57; histogram endpoints require new
  lit/SIM or diagnostic tests before they can be counted as implemented
```

Current checked-in coverage for 3.3 dense f8->f32->compute->f8:

```text
lit:
  test/lit/vmi_new/vmi_layout_assignment_f8_compute_f8.pto

runtime SIM:
  test/vpto/cases/vmi_new/f8-compute-f8
```

Current checked-in coverage for 3.1/3.2 dense f16/f32 conversion stores:

```text
lit:
  test/lit/vmi_new/vmi_layout_assignment_dense_f16_f32_store.pto

runtime SIM:
  test/vpto/cases/vmi_new/widen-f16-to-f32-store-reduce
  test/vpto/cases/vmi_new/quant-f32-to-f16-tail
```

Current checked-in coverage for basic packed group_reduce -> group_store paths
for 3.4, 3.5.1, and 3.6.1:

```text
lit:
  test/lit/vmi_new/vmi_layout_assignment_group_reduce_slots8_store.pto
  test/lit/vmi_new/vmi_layout_assignment_group_reduce_s16_store.pto
  test/lit/vmi_new/vmi_layout_assignment_group_reduce_s32_store.pto

runtime SIM:
  test/vpto/cases/vmi_new/group-reduce-basic-store
```

Current checked-in coverage for S=16 group broadcast continuation:

```text
lit:
  test/lit/vmi_new/vmi_layout_assignment_group_slots_fanout.pto

runtime SIM:
  test/vpto/cases/vmi_new/group-reduce-s16-broadcast-reduce-store
```

Current checked-in coverage for 3.35 group_slots fanout to direct group_store
and group_broadcast:

```text
lit:
  test/lit/vmi_new/vmi_layout_assignment_group_slots_fanout.pto

runtime SIM:
  test/vpto/cases/vmi_new/group-slots-fanout-store-broadcast
```

Current checked-in coverage for 3.8 `group_reduce -> group_broadcast ->
truncf -> dense store` and 3.17 `group_broadcast` feeding a
deinterleaved consumer:

```text
lit:
  test/lit/vmi_new/vmi_layout_assignment_group_reduce_s16_truncf_broadcast_store.pto

runtime SIM:
  test/vpto/cases/vmi_new/group-reduce-s16-truncf-broadcast-store
```

Current checked-in coverage for 3.18 one dense value with dense and
group-reduce consumers:

```text
lit:
  test/lit/vmi_new/vmi_layout_assignment_dense_group_reduce_multi_consumer.pto

runtime SIM:
  test/vpto/cases/vmi_new/dense-group-reduce-multi-consumer
```

Current checked-in coverage for 3.10 non-load producer feeding S=32
`group_reduce`:

```text
lit:
  test/lit/vmi_new/vmi_layout_assignment_non_load_s32_reduce.pto

runtime SIM:
  test/vpto/cases/vmi_new/group-reduce-s32-add-bias-store
```

Current checked-in coverage for 3.23 group_broadcast with multiple dense
consumers:

```text
lit:
  test/lit/vmi_new/vmi_layout_assignment_group_broadcast_multi_consumer.pto

runtime SIM:
  test/vpto/cases/vmi_new/group-broadcast-multi-consumer
```

Current checked-in coverage for S=32 contiguous group broadcast continuation:

```text
lit:
  test/lit/vmi_new/vmi_layout_assignment_group_reduce_s32_broadcast_reduce.pto

runtime SIM:
  test/vpto/cases/vmi_new/group-reduce-s32-broadcast-reduce-store
```

Current checked-in coverage for 3.21 S=32 tail with a statically safe
full-read source:

```text
lit:
  test/lit/vmi_new/vmi_layout_assignment_group_reduce_s32_tail_full_tile.pto

runtime SIM:
  test/vpto/cases/vmi_new/group-reduce-s32-tail-full-tile-store
  The VPTO backend always runs layout assignment before VPTO/LLVM emission for
  the partial pointer load.
```

Current checked-in coverage for 3.44 masked_load grouped tail feeding S=32
reduce:

```text
lit:
  test/lit/vmi_new/vmi_layout_assignment_masked_load_group_tail_s32.pto

runtime SIM:
  test/vpto/cases/vmi_new/masked-load-group-tail-s32-reduce-store
```

Current checked-in coverage for 3.45 dynamic S=32 `create_group_mask`:

```text
lit:
  test/lit/vmi_new/vmi_layout_assignment_create_group_mask_s32_dynamic.pto

runtime SIM:
  test/vpto/cases/vmi_new/dynamic-create-group-mask-s32-reduce-store

runtime scalar source:
  active_cols is passed as a kernel i32 scalar argument and cast to index inside
  vecscope before pto.vmi.create_group_mask. This is an explicit scalar ABI,
  not a value recovered by vmi-to-vpto from producer/consumer context.
```

Current checked-in runtime coverage for 3.12 control-flow join before S=32
`group_reduce`:

```text
lit:
  test/lit/vmi_new/vmi_layout_assignment_cf_branch.pto
  test/lit/vmi_new/vmi_to_vpto_cf_branch.pto

runtime SIM:
  test/vpto/cases/vmi_new/group-reduce-s32-cf-join-store
```

Current checked-in runtime coverage for 3.20 `group_slots` control-flow join:

```text
lit:
  test/lit/vmi_new/vmi_layout_assignment_group_slots_cf_join.pto

runtime SIM:
  test/vpto/cases/vmi_new/group-slots-cf-join-store
```

Current checked-in runtime coverage for 3.22 `scf.for` loop-carried VMI layout:

```text
lit:
  test/lit/vmi_new/vmi_layout_assignment_scf_for.pto
  test/lit/vmi_new/vmi_to_vpto_scf_for.pto

runtime SIM:
  test/vpto/cases/vmi_new/scf-for-loop-carried-store
```

Current checked-in runtime coverage for 3.42 `group_slots` `scf.for`
loop-carried accumulator:

```text
lit:
  test/lit/vmi_new/vmi_layout_assignment_group_slots_scf_for.pto

runtime SIM:
  test/vpto/cases/vmi_new/group-slots-scf-for-store
```

Current checked-in coverage for 3.25.1 private function result boundary:

```text
lit:
  test/lit/vmi_new/vmi_ptoas_private_call_inline.pto

runtime SIM:
  test/vpto/cases/vmi_new/private-call-inline-store

implementation note:
  after vmi-to-vpto physicalizes the private helper, ptoas inlines private
  single-block helpers whose signatures contain !pto.vreg or !pto.mask. This
  happens before VPTO vecscope/backend emission, so physical vector values do
  not escape through a function return.
```

Current checked-in coverage for 3.43 internal function argument boundary
materialization:

```text
lit:
  test/lit/vmi_new/vmi_layout_assignment_call_argument_boundary.pto
  test/lit/vmi_new/vmi_ptoas_call_boundary_vecscope.pto

runtime SIM:
  test/vpto/cases/vmi_new/private-call-argument-boundary-store

implementation note:
  private physical helper inlining also covers void helper calls with physical
  VMI arguments, so the backend no longer sees a physical VPTO vector function
  ABI for this internal boundary.
```

Current checked-in coverage for packed group-slot RHS elementwise continuations
for 3.5.3 and 3.6.2:

```text
lit:
  test/lit/vmi_new/vmi_layout_assignment_group_slot_load_dual_layout.pto

runtime SIM:
  test/vpto/cases/vmi_new/group-reduce-slot-add-store
```

Current checked-in coverage for S=64 row-local group broadcast continuation
with aligned row_stride:

```text
lit:
  test/lit/vmi_new/vmi_layout_assignment_group_reduce_s64_broadcast_reduce.pto

runtime SIM:
  test/vpto/cases/vmi_new/group-reduce-s64-broadcast-reduce-store
```

Current checked-in coverage for S=64 active-row tail with aligned row_stride:

```text
lit:
  test/lit/vmi_new/vmi_layout_assignment_group_reduce_s64_tail_store.pto

runtime SIM:
  test/vpto/cases/vmi_new/group-reduce-s64-tail-store
```

The companion lit case for non-unit slots=1 point-store lowering is:

```text
test/lit/vmi_new/vmi_to_vpto_group_store_slots1_1pt.pto
```

Current checked-in coverage for S=64 row-local group-slot RHS elementwise
continuation with aligned source_group_stride and aligned output row_stride:

```text
lit:
  test/lit/vmi_new/vmi_layout_assignment_group_slot_load_dual_layout.pto

runtime SIM:
  test/vpto/cases/vmi_new/group-reduce-s64-slot-add-store
```

Current checked-in coverage for 3.34 S=64 `slots = 1` group-slot f32->f16 cast:

```text
lit:
  test/lit/vmi_new/vmi_layout_assignment_group_reduce_s64_truncf.pto

runtime SIM:
  test/vpto/cases/vmi_new/group-reduce-s64-truncf-store
```

The companion negative lit cases for dynamic or unaligned `%c2` slots=1
group_slot_load, and non-unit `slots = 8` group_slot_load, are:

```text
test/lit/vmi_new/vmi_to_vpto_group_slot_load_nonunit_slots8_invalid.pto
test/lit/vmi_new/vmi_layout_assignment_group_slot_load_slots1_dynamic_stride_invalid.pto
test/lit/vmi_new/vmi_layout_assignment_group_slot_load_slots1_unaligned_stride_invalid.pto
```

Current checked-in coverage for the strided block-load cases:

```text
lit:
  test/lit/vmi_new/vmi_layout_assignment_group_load_s16_stride_store.pto
  test/lit/vmi_new/vmi_layout_assignment_group_load_s16_unaligned_stride_invalid.pto
  test/lit/vmi_new/vmi_layout_assignment_group_load_s32_stride_store.pto
  test/lit/vmi_new/vmi_layout_assignment_group_load_s32_stride_broadcast_reduce.pto
  test/lit/vmi_new/vmi_layout_assignment_group_load_s32_unaligned_stride_invalid.pto

runtime SIM:
  test/vpto/cases/vmi_new/group-load-s16-stride-store
  test/vpto/cases/vmi_new/group-load-s32-stride-store
  test/vpto/cases/vmi_new/group-load-s32-stride-broadcast-reduce
```

Current checked-in coverage for grouped mask S=16 tail:

```text
lit:
  test/lit/vmi_new/vmi_layout_assignment_create_group_mask_s16.pto
  test/lit/vmi_new/vmi_create_group_mask_invalid.pto

runtime SIM:
  test/vpto/cases/vmi_new/group-reduce-s16-group-mask-tail-store
  test/vpto/cases/vmi_new/group-reduce-s16-stride-group-mask-tail-store
  test/vpto/cases/vmi_new/group-reduce-s16-group-mask-broadcast-reduce-store
```

Current checked-in coverage for 3.24 mask/select/masked-store semantics:

```text
lit:
  test/lit/vmi_new/vmi_layout_assignment_mask_select_store.pto

runtime SIM:
  test/vpto/cases/vmi_new/mask-select-store
```

Current checked-in coverage for 3.29 one semantic mask with f32 and f16
consumers:

```text
lit:
  test/lit/vmi_new/vmi_layout_assignment_mask_granularity_f32_f16_store.pto

runtime SIM:
  test/vpto/cases/vmi_new/mask-granularity-f32-f16-store
```

Current checked-in coverage for 3.31 f16->f32 feeding dense store and S=16
reduce:

```text
lit:
  test/lit/vmi_new/vmi_layout_assignment_widen_f16_store_reduce.pto

runtime SIM:
  test/vpto/cases/vmi_new/widen-f16-to-f32-store-reduce
```

Current checked-in lit coverage for the first `vmi-layout-fold`
optimization is:

```text
test/lit/vmi_new/vmi_layout_fold_store.pto
test/lit/vmi_new/vmi_layout_fold_masked_store.pto
test/lit/vmi_new/vmi_layout_fold_deint4.pto
```

Current checked-in lit coverage for the first `vmi-layout-rematerialize`
optimization is:

```text
test/lit/vmi_new/vmi_layout_rematerialize_data.pto
test/lit/vmi_new/vmi_layout_rematerialize_mask.pto
```

Current checked-in lit coverage for the first
`vmi-layout-sink-materialization` optimization is:

```text
test/lit/vmi_new/vmi_layout_sink_materialization_binary.pto  // unary, binary, fma, cmp, and select data ops
test/lit/vmi_new/vmi_layout_sink_materialization_mask.pto
```

Current checked-in lit coverage for canonicalized VMI control-flow restoration is:

```text
test/lit/vmi_new/vmi_legalize_arith_select.pto
test/lit/vmi_new/vmi_ptoas_cli_control_flow.pto
```

Current checked-in lit coverage for the first semantic local-lowering layout gate
is:

```text
test/lit/vmi_new/vmi_layout_gate_group_slot_load_support_invalid.pto
test/lit/vmi_new/vmi_layout_gate_group_load_support_invalid.pto
test/lit/vmi_new/vmi_layout_gate_group_store_support_invalid.pto
test/lit/vmi_new/vmi_layout_gate_group_slots_unsupported_slots_invalid.pto
test/lit/vmi_new/vmi_layout_gate_store_support_invalid.pto
test/lit/vmi_new/vmi_layout_gate_helper_materialization_shape_invalid.pto
test/lit/vmi_new/vmi_layout_gate_group_reduce_support_invalid.pto
test/lit/vmi_new/vmi_layout_gate_group_reduce_slots1_support_invalid.pto
test/lit/vmi_new/vmi_layout_gate_group_broadcast_support_invalid.pto
test/lit/vmi_new/vmi_layout_gate_truncf_support_invalid.pto
test/lit/vmi_new/vmi_layout_gate_extf_support_invalid.pto
test/lit/vmi_new/vmi_layout_gate_bitcast_support_invalid.pto
test/lit/vmi_new/vmi_layout_gate_bitcast_group_slots.pto
```

Current checked-in direct `vmi-to-vpto` preflight coverage for bitcast local
lowering is:

```text
test/lit/vmi_new/vmi_to_vpto_bitcast_footprint_invalid.pto
test/lit/vmi_new/vmi_to_vpto_bitcast_group_slots.pto
```

Current checked-in coverage for 3.32 f32 feeding f8 store and S=32 reduce:

```text
lit:
  test/lit/vmi_new/vmi_layout_assignment_f32_f8_store_reduce.pto

runtime SIM:
  test/vpto/cases/vmi_new/f32-to-f8-store-reduce
```

Current checked-in coverage for multi-tile group-slot arity:

```text
lit:
  test/lit/vmi_new/vmi_layout_assignment_group_reduce_s32_multitile_store.pto

runtime SIM:
  test/vpto/cases/vmi_new/group-reduce-s32-multitile-store
```

Current checked-in coverage for 3.40 scalar broadcast feeding dense and grouped
users:

```text
lit:
  test/lit/vmi_new/vmi_layout_assignment_broadcast_dense_group_users.pto

runtime SIM:
  test/vpto/cases/vmi_new/broadcast-dense-group-users
```

Current checked-in coverage for 3.41 non-rematerializable `masked_load` feeding
dense and grouped users:

```text
lit:
  test/lit/vmi_new/vmi_layout_assignment_masked_load_dense_group_users.pto

runtime SIM:
  test/vpto/cases/vmi_new/masked-load-dense-group-users
```

Diagnostic-only cases:

```text
3.9 dense store of group slots
3.11.2 S=32 tail without full_footprint_readable
3.7.4 S=64 slots=1 group_store with unit output stride
3.13 packed group-slot f32 -> f16 cast
3.14 unsupported group size
3.15.3 compact source row stride 12
3.16.1 group_slot_load slots=8 non-unit stride
3.16.2 group_slot_load slots=1 dynamic or unaligned stride
3.27 S=32 source_group_stride not divisible by 8 f32 elements
3.19.2 block_deinterleaved value consumed by truncf without materialization path
3.25.2 public/external VMI boundary
```

Current checked-in diagnostic coverage for 3.9/3.13/3.14:

```text
lit:
  test/lit/vmi_new/vmi_layout_assignment_dense_store_group_slots_invalid.pto
  test/lit/vmi_new/vmi_layout_assignment_packed_group_slots_truncf_invalid.pto
  test/lit/vmi_new/vmi_layout_assignment_group_reduce_s12_invalid.pto
```

Current checked-in diagnostic coverage for the remaining non-SIM diagnostic
entries:

```text
lit:
  test/lit/vmi_new/vmi_layout_gate_helper_support_invalid.pto
  test/lit/vmi_new/vmi_layout_gate_helper_materialization_shape_invalid.pto
  test/lit/vmi_new/vmi_layout_assignment_group_reduce_s32_tail_no_full_tile_invalid.pto
  test/lit/vmi_new/vmi_layout_assignment_group_load_s16_compact_stride12_invalid.pto
  test/lit/vmi_new/vmi_to_vpto_group_slot_load_nonunit_slots8_invalid.pto
  test/lit/vmi_new/vmi_layout_assignment_group_slot_load_slots1_dynamic_stride_invalid.pto
  test/lit/vmi_new/vmi_layout_assignment_group_slot_load_slots1_unaligned_stride_invalid.pto
  test/lit/vmi_new/vmi_layout_assignment_group_load_block8_truncf_invalid.pto
  test/lit/vmi_new/vmi_to_vpto_group_store_slots1_1pt.pto
  test/lit/vmi_new/vmi_layout_assignment_group_load_s16_unaligned_stride_invalid.pto
  test/lit/vmi_new/vmi_layout_assignment_group_load_s32_unaligned_stride_invalid.pto
  test/lit/vmi_new/vmi_ptoas_public_abi_invalid.pto
  test/lit/vmi_new/vmi_ptoas_public_result_abi_invalid.pto
  test/lit/vmi_new/vmi_layout_assignment_external_call_invalid.pto
  test/lit/vmi_new/vmi_layout_assignment_external_decl_invalid.pto
```

Capability boundaries and runtime evidence notes:

```text
private physical function ABI:
  3.25.1 and 3.43 runtime coverage is closed for private/internal single-block
  helpers by inlining private physical VMI helpers after vmi-to-vpto and before
  VPTO vecscope/backend emission. Public/external VMI boundaries are still
  rejected until a stable VMI ABI is defined.

memory-proof runtime coverage:
  3.21 S=32 rounded tail-mask coverage is provided by a runtime case that loads
  a full 256xf32 UB pointer vector and uses a 192-lane mask to define the active
  logical rows. No surrounding MTE, caller/body context, or producer/user scan is
  inspected to justify partial pointer reads.
```

## 12. Implementation Slices

### Slice 1: IR Skeleton And Verifiers

```text
layout attrs
vmi.vreg/vmi.mask types
surface op definitions
surface/layout validators
```

### Slice 2: Straight-Line Dense Assignment/Lowering

```text
3.1 f16->f32->store
3.2 f32->f16->store
3.3 f8->f32->compute->f8
```

### Slice 3: Group Slots And Reductions

```text
3.4 S=8
3.5 S=16 parity/block8
3.6 S=32
3.7 S=64
group_slot_load
group_broadcast
group_store
```

### Slice 4: Layout Conflicts And Materialization

```text
3.8 cast commute through group_broadcast
3.18 dense/group-reduce multi-consumer
3.19 deinterleaved versus block_deinterleaved selection
3.23 group_broadcast multi-consumer
3.32 f32 feeding f8 store and S=32 reduce
3.33 S=16/S=32 reduce multi-consumer rematerialization
3.34 slots=1 group-slot f32->f16 cast
3.35 group_slots fanout to group_store and group_broadcast
3.36 group_slot_load expressed as explicit slots=8/slots=1 producers
3.38 multi-tile group_slots arity
3.40 scalar broadcast materialized for dense/grouped users
3.41 non-rematerializable value with ensure_layout
```

### Slice 5: Masks, Tail, And Memory Legality

```text
create_mask
create_group_mask
masked_store
safe full-read proof
compact/gather diagnostics
mask granularity per use
group_load stride greater than group size
group_slot_load slots=1 aligned non-unit stride plus dynamic/unaligned diagnostic
group_store slots=1 non-unit output stride
strided group_load feeding broadcast and a second reduce
masked_load grouped tail feeding S=32 reduce
```

### Slice 6: Control Flow And Functions

```text
scf.if
scf.for
group_slots across control flow
group_slots loop-carried accumulator
internal function specialization
internal function argument boundary materialization
public ABI diagnostic
```

### Slice 7: Histogram

```text
3.56 full 256-bin dhist logical op
3.57 chist semantic capability diagnostic
```

## 13. Completion Checklist

Current evidence for the case-catalog objective:

```text
1. every pre-histogram catalog endpoint is mapped in section 6.6 to an
   assignment owner, assignment artifact, and vmi-to-vpto contract
2. every pre-histogram SIM-backed positive endpoint is listed in section 11.3
   and has a checked-in runtime case directory
3. every existing runtime case directory contains kernel.pto, launch.cpp,
   main.cpp, golden.py, and compare.py
4. the latest historical broad VMI runtime sweep passed: PASS=47 FAIL=0
5. the latest historical full VMI lit sweep passed: 350/350
6. every pre-histogram unsupported endpoint listed in section 11.3 has a
   diagnostic lit test
7. vmi-to-vpto decisions are represented by current-op attrs/operands,
   assigned layouts, helper ops, rematerialization, or diagnostics
8. no separate lowering-plan string attr is emitted or consumed
9. release docs remain untouched; this is still a design/implementation plan
   under docs/designs
10. new histogram endpoints 3.56/3.57 are mapped in section 6.6, but their
    implementation evidence is intentionally pending new lit/SIM or diagnostic
    tests
```
