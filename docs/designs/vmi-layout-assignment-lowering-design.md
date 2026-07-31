# VMI Layout Assignment And Lowering Design

本文是新的 VMI layout assignment / lowering 设计文档。它只以
`docs/designs/vmi-layout-lowering-cases.md` 为 source of truth，不继承早期
VMI 草稿的 layout 设计，以避免旧上下文污染。

目标：

```text
VMI surface IR
  -> pto-validate-vmi-ir
  -> vmi-layout-assignment          // hard legalization baseline
  -> canonicalize/cse
  -> vmi-layout-rematerialize       // optional optimization
  -> canonicalize/cse
  -> vmi-layout-fold      // optional optimization over remat-exposed helpers
  -> canonicalize/cse
  -> vmi-layout-sink-materialization // optional optimization
  -> canonicalize/cse
  -> vmi-legalize-arith-select
  -> pto-validate-vmi-layout-ir
  -> layout-assigned and optimized VMI IR
  -> vmi-to-vpto
  -> VPTO IR
```

核心验收约束：

```text
vmi-to-vpto 不允许通过上下文猜 lowering。

任何需要 producer/consumer/control-flow/memory/mask 上下文才能决定的事，
必须在 vmi-layout-assignment 或后续 VMI layout optimization 阶段变成显式 IR：

1. vmi.vreg/vmi.mask 的 layout
2. current-op attrs/operands that make the local lowering deterministic
3. use-site ensure_layout / ensure_mask_layout / ensure_mask_granularity
4. rematerialized or cloned producer
5. target capability diagnostic
```

## 0. Hard Legalization And Optimization Boundary

Layout assignment is a stage, not necessarily one monolithic pass.  The design
separates correctness from optimization:

```text
hard legalization:
  produces legal layout-assigned VMI IR for all supported semantics
  inserts conservative ensure_* helpers at incompatible uses
  may choose a simple canonical layout even when a fused consumer lowering exists
  must diagnose unsupported semantics before vmi-to-vpto has to guess

layout optimization:
  rewrites already legal VMI IR into cheaper but equivalent VMI IR
  may fold ensure_layout into a layout-aware consumer
  may clone/rematerialize cheap producers for different use-site layouts
  may sink or hoist layout materialization through pure elementwise chains
  may specialize private VMI function signatures
```

The driver currently runs MLIR's normal `canonicalize` and `cse` between these
VMI-specific passes.  They are allowed to clean up trivially unused helpers,
merge identical rematerialized producers, and expose simpler use-def shapes.
They are not a source of hidden lowering information; after every optimization,
the IR must still carry enough local information for `vmi-to-vpto`.

The baseline hard pass may emit:

```text
%x_c = pto.vmi.ensure_layout %x : deinterleaved=2 -> contiguous
pto.vmi.store %x_c
```

A later optimization may replace that use with:

```text
pto.vmi.store %x : deinterleaved=2
```

only if the store op itself has a local deterministic lowering for preserving the
same row-major memory effect, such as a layout-aware `vstsx2 INTLV` lowering.
Both forms are semantically complete.  The second form is an optimization, not
a hard requirement for correctness.

## 1. Source Case Coverage

设计必须覆盖 case catalog 中的端到端场景：

```text
dense cast:
  f16 -> f32 -> store
  f32 -> f16 -> store
  f8  -> f32 -> compute -> f8
  f8  -> f32 accumulator -> group_reduce_addf
  i8/i16 -> signed/unsigned integer cast to i32 accumulator
          -> group_reduce_addi
  f8/i8 appear as cast source or cast destination at compute boundaries
  integer narrowing back to i8 is an explicit cast, not implicit arithmetic
  f16 -> f32 shared by dense store and S=16 reduce
  f32 shared by f8 store and S=32 reduce

group reduce:
  32-bit accumulator: S=8, S=16, S=32, S=64
  16-bit accumulator: S=16, S=32, S=64, S=128
  8-bit storage reduces only through an explicit accumulator cast
  reduce -> group_store
  reduce -> group_slot_load/elemwise -> group_store
  reduce -> group_broadcast -> elemwise -> reduce -> store
  one group_slots result fanning out to group_store and group_broadcast
  grouped tail -> broadcast -> reduce -> store

layout conflict:
  one value with dense and group-reduce consumers
  one value with S=16 and S=32 group-reduce consumers
  one scalar broadcast materialized for dense and grouped users, with optional rematerialization
  one non-rematerializable value materialized with use-site ensure_layout
  one scalar group-slot source expressed as explicit slots=8 and slots=1 producers
  S=16 element-deinterleaved/block-deinterleaved layout selection
  dense consumer of group_slots diagnostic
  packed group-slot width-changing cast diagnostic
  S=64 slots=1 group-slot width-changing cast

control flow:
  scf.if before group_reduce
  group_slots across scf.if
  scf.for loop-carried layout fixed point
  group_slots as scf.for loop-carried accumulator
  internal function boundary specialization
  internal function argument boundary materialization
  public/external VMI ABI diagnostic

mask and tail:
  prefix mask
  group-periodic mask
  dynamic group-periodic mask
  masked_load tail with explicit passthrough instead of padding
  masked_load grouped tail feeding group_reduce
  masked select/store
  one semantic mask used by multiple predicate granularities
  S=32 tail with and without full_footprint_readable
  compact S=12 diagnostic

strided memory:
  group_load source stride greater than logical group size
  strided group_load feeding broadcast and a second group_reduce
  group_slot_load slots=1 with non-unit source stride
  group_store slots=1 with non-unit output stride

value-indexed accumulation:
  full 256-bin distribution histogram over Nxui8 source lanes
  VPTO low/high bin range split hidden behind one logical 256xui16 VMI result
  cumulative histogram is a semantic boundary until CHISTv2 range semantics are verified
```

### 1.1 Case-Set Sufficiency

The current case set is sufficient to define the first implementation of layout
assignment and lowering.  It covers every decision axis that has changed the
design so far:

```text
physical dense layout:
  contiguous, deinterleaved=2/4, block_deinterleaved=2/4

group-slot result layout:
  group_slots(G, slots=8) for packed VCG results
  group_slots(G, slots=1) for row-local S=64 results

producer-driven layout:
  load, group_load, group_slot_load, broadcast, create_mask,
  create_group_mask

consumer-driven pressure:
  dense store, group_reduce, group_store, group_broadcast, truncf,
  elementwise/select, masked_load/masked_store

conflict resolution:
  explicit ensure_layout, explicit ensure_mask_layout, explicit diagnostics
  optimization passes may later replace the helpers with rematerialization or
  layout-aware consumers

control-flow propagation:
  scf.if, scf.for iter_args/results, internal/private function boundaries,
  public ABI rejection

memory legality:
  full_footprint_readable proof, grouped masks, predicate granularity, aligned
  strided group memory, stable gather diagnostic

value-indexed accumulation:
  histogram source/result shape, b8 source mask, and fixed low/high VPTO bin
  split for a logical 256-bin result
```

No extra layout kind should be added unless a new case proves that the existing
layouts and explicit helper contracts cannot express the logical behavior.  The remaining open
items are not missing layout semantics:

```text
dynamic active_elems_per_group runtime source:
  create_group_mask layout lowering is defined and has both lit and SIM
  coverage. The supported runtime source is a kernel scalar argument cast to
  index inside vecscope; vmi-to-vpto does not recover this value from GM/UB
  scalar loads or surrounding context.

private vector function runtime:
  private/internal single-block helpers are runtime-covered by ptoas inlining
  private physical VMI helpers after vmi-to-vpto and before VPTO vecscope/backend
  emission. This is a post-physicalization backend hygiene step; vmi-to-vpto
  still lowers only from assigned layouts and helper ops.

diagnostic-only cases:
  compact S=12 gather fallback, packed slots=8 width-changing cast, public VMI
  ABI, unsafe masked_load tail, and unaligned/dynamic group memory remain
  explicit capability boundaries.
```

## 2. Layout Domain

Layout is a property of a layout-assigned VMI value, not a property inferred by
the final lowering pattern.

Type policy:

```text
storage boundary:
  f8-like/i8/f16/i16/f32/i32 may appear in load/store values when the target
  memory instruction supports the physical width.

cast boundary:
  f8-like participates through extf/truncf.
  i8 participates through extsi/extui/trunci. Signedness is carried by the
  cast op semantics, not by a separate layout.
  On the current VPTO target, 32-bit to 8-bit integer narrowing is only a
  baseline lowering for unsigned i8 results because the available VCVTII forms
  are s32/u32 -> u8.

compute boundary:
  baseline floating compute uses f16/f32.
  baseline integer grouped reduction compute uses i32 accumulators.  i8/i16
  storage must be widened first because integer reduction instructions widen
  narrow inputs.
  f8/i8 are not baseline accumulator/compute element types.

value-indexed accumulation boundary:
  pto.vmi.vdhist consumes ui8 source lanes and produces a logical 256xui16
  accumulator/result.  It is not a group_reduce family member because result
  bins are selected by source values rather than by source lane/group position.
  pto.vmi.vchist uses the same surface shape only after the target CHISTv2
  range semantics are verified.
```

### 2.1 Dense Layouts

```text
#pto.vmi.layout<contiguous>
#pto.vmi.layout<deinterleaved = F>
#pto.vmi.layout<block_deinterleaved = F>
```

Dense layouts preserve one semantic value for every logical lane.  The two
split layouts are distinct kinds:

```text
deinterleaved=F:
  part p        = i % F
  physical lane = i / F

block_deinterleaved=F:
  B = 32B / sizeof(element)
  block q         = i / B
  in-block lane r = i % B
  part p           = q % F
  part block t      = q / F
  physical lane     = t * B + r
```

The 32B block width is derived from the carrier type rather than stored in the
attribute:

```text
f32/b32 -> B=8
f16/b16 -> B=16
f8/b8   -> B=32
```

Important consequence:

```text
deinterleaved=2
block_deinterleaved=2
```

are different layouts.  They cannot be treated as compatible merely because
`F` is the same.

See `vmi-lane-stride-generalization-design.md` for the planned extension that
allows dense layouts to carry `lane_stride` as an additional lane-map field.
That extension keeps dense lane stride separate from the existing group-slot
carrier lowering use case.  Non-zero lane phase is left as a future extension
and is not required for the first dense-stride optimization.

### 2.2 Group-Slot Layouts

```text
#pto.vmi.layout<num_groups = G, slots = K>
#pto.vmi.layout<num_groups = G, slots = K, lane_stride = LS>
```

Only `G` lanes have semantic values:

```text
slot_block(g) = g / K
slot_lane(g)  = (g % K) * LS
```

All non-slot lanes are undefined and may only be read by group-aware operations.
Ordinary dense `add/mul/store/truncf` cannot consume `group_slots`.

`LS` defaults to 1 and is measured in logical element-sized physical slots.  It
is not a new group semantic; it records regular physical spacing for each stored
group slot.  For example, `ui8 lane_stride=4` maps slot values to byte lanes
0, 4, 8, ... in a ui8 physical vector and lets `group_store` use a
`PK4_B32` store.  Any b16/b32 values used by pack/unpack or store instructions
are instruction-local carriers; the group-slot SSA value keeps its logical
element type, matching ordinary dense lane-stride values.

`K` is selected by the assigned producer/result contract:

```text
S=8/16/32 packed VCG result -> slots=8
S=64 row-local result       -> slots=1
```

Histogram does not add a layout family.  A full logical histogram result uses:

```text
!pto.vmi.vreg<256xui16, #pto.vmi.layout<contiguous>>
```

and physicalizes to two ordered VPTO parts:

```text
part0 = logical bins   0..127
part1 = logical bins 128..255
```

The VPTO `#bin` selector is therefore an op-local lowering detail, not a VMI
layout attribute and not a user-visible operand on `pto.vmi.vdhist`.

## 3. Lowering Context Must Become Explicit IR Output

`vmi-to-vpto` may inspect only:

```text
1. op name and explicit op attrs
2. converted operand/result types with layout
3. helper/materialization ops written by layout assignment
4. inserted helper ops
5. target capability registry
```

It must not:

```text
1. walk to defining op to infer layout
2. inspect all users to choose a lowering path
3. infer memory legality from a later mask
4. decide S=16 deinterleaved vs block_deinterleaved locally
5. decide whether group_broadcast should be materialized for one or many users
6. specialize function signatures during vmi-to-vpto
```

Any of those decisions belongs to the layout stage before `vmi-to-vpto`.

## 4. Explicit Assignment Products

After `vmi-layout-assignment`, every VMI data and mask value must be in one of
these states:

```text
layout-assigned type:
  !pto.vmi.vreg<NxT, #pto.vmi.layout<...>>
  !pto.vmi.mask<Nxpred, #pto.vmi.layout<...>>

or explicit helper:
  pto.vmi.ensure_layout
  pto.vmi.ensure_mask_layout
  pto.vmi.ensure_mask_granularity
```

`vmi-to-vpto` is allowed to choose a deterministic lowering from local
information on the current op:

```text
current op name
current op attrs
operand/result types and layouts
current op operand values such as stride and offset
target capability and pass options
```

This is not context inference.  What remains forbidden is walking to producers,
users, sibling users, branch/loop bodies, callees/callers, or nearby memory/MTE
ops to recover a lowering decision or a memory-safety proof.

If a decision cannot be made from that local information, layout assignment
must rewrite the IR until the decision is explicit in attrs, operand/result
layouts, helper ops, or diagnostics.  Later optimization passes may replace
helpers with cloned/rematerialized producers, but `vmi-to-vpto` must not
consume a separate string lowering-plan attr.

### 4.1 Local Lowering Contract

The lowering path is derived from op + assigned operand/result layouts +
explicit attrs/operands.  If two legal lowerings cannot be distinguished from
that local information, the IR is missing a semantic carrier and must be
extended before that lowering is implemented.

The shared abstraction is a layout fact classifier, not a central lowering-plan
registry.  A classifier may answer questions such as:

```text
cast layout fact:
  f16/i16 -> f32/i32 requires contiguous source and deinterleaved=2 result
  f8/i8  -> f32/i32 requires contiguous source and deinterleaved=4 result
  f32/i32 -> f16/i16 requires deinterleaved=2 source and contiguous result
  f32/i32 -> f8/i8  requires deinterleaved=4 source and contiguous result

group_reduce layout fact:
  define E = sizeof(accumulator T), VLaneElems = 32B / E,
  L = 256B / E, S = N / G.
  S == VLaneElems      requires contiguous source/mask and
                       group_slots(G, slots=8) result.
  S == 2 * VLaneElems  requires deinterleaved=2 source/mask and
                       group_slots(G, slots=8) result.
  S == 4 * VLaneElems  requires deinterleaved=4 source/mask and
                       group_slots(G, slots=8) result.
  S >= L && S % L == 0 requires contiguous source/mask and
                       group_slots(G, slots=1) result.

memory safety fact:
  full physical chunks are legal for pointer sources. Partial logical loads
  need a shaped safe-tail memref proof or an explicit fallback option.
```

These helpers return semantic layout requirements and capability diagnostics.
They do not return VPTO instruction names, cost decisions, clone decisions, or
multi-user plans.

The useful shared fact is the part that would otherwise be recomputed by two or
more stages and must stay identical for correctness:

```text
cast width ratio:
  assignment uses it to request source/result layouts and insert ensure_layout.
  validation uses it to reject unsupported assigned cast shapes.
  lowering uses it to check the local op shape before emitting VPTO.

group_reduce lane partition:
  assignment uses N/G and accumulator element width to request source/mask and
  result layouts.
  validation uses the same math to reject legacy or incomplete group_slots.
  lowering uses the already assigned layouts to select the local VPTO sequence.

layout materialization shape:
  assignment may insert ensure_layout without proving every physical sequence.
  validation and lowering use one support query to decide whether that explicit
  helper is materializable on the target.
  optimization uses the same query only when it wants to fold/sink/remove an
  explicit helper.
```

The helper is not useful when it only renames one local pattern.  A single
`if (is this op with this attr)` that is not shared by assignment, validation,
lowering, or an optimization should stay local to that pass.  The support layer
exists to prevent divergent layout math, not to move every branch into a table.

Forbidden non-local lowering recovery:

```text
No pattern may recover a lowering decision or memory proof by:
  - walking from group_reduce to the load/group_load producer
  - walking from store/broadcast/truncf to the group_reduce producer
  - scanning sibling users of a group_slots value
  - inspecting branch bodies or loop bodies from a control-flow boundary
  - inspecting private callee bodies while lowering a call
```

If the current op lacks enough local information, `vmi-to-vpto` emits
`VMI-LAYOUT-CONTRACT` at the current op and prints the op name, logical type,
assigned layouts, and the missing decision class.

## 5. Layout Requests, Helpers, And Optimization

The compiler must not carry a target-aware lowering-plan registry as the shared
contract between assignment, optimization, validation, and lowering.  The
shared contract is:

```text
1. assigned layouts on VMI types
2. explicit use-site helpers: ensure_layout, ensure_mask_layout,
   ensure_mask_granularity
3. explicit op attrs/operands that are part of the semantic op
4. small layout fact classifiers shared only where they remove duplicated
   layout math
5. target capability diagnostics
```

This split makes optimization simpler only when optimization is phrased as
rewriting explicit helper IR:

```text
baseline:
  %x_d2 = pto.vmi.extf %x_f16
  %a    = pto.vmi.addf %x_d2, %k_d2
  %a_c  = pto.vmi.ensure_layout %a : deinterleaved=2 -> contiguous
  pto.vmi.store %a_c, %out0
  %x_c  = pto.vmi.ensure_layout %x_d2 : deinterleaved=2 -> contiguous
  pto.vmi.store %x_c, %out1

fold-consumers:
  checks only each local ensure_layout + store use.
  If VMILayoutSupport says the store can preserve row-major memory from the
  source layout, rewrite that use to store the source directly.
  It does not inspect sibling users of %x_d2 and does not recompute the layout
  assignment.

rematerialize:
  checks only cheap producer + ensure_layout.
  If the producer can directly create the requested layout, clone/rematerialize
  that producer for the use.
  Memory producers such as group_slot_load are excluded until a separate proof
  says cloning is semantically and economically valid.

sink-materialization:
  checks only explicit ensure_* operands of a layout-transparent op.
  If every operand helper is compatible, rebuild the op in the source layout and
  leave one ensure_* on the result.
```

If an optimization needs a global cost decision, it should produce a new
explicit IR shape and then rely on canonicalize/CSE.  It must not communicate a
private decision to `vmi-to-vpto`.

### 5.1 Baseline Dense Layout Requests

```text
f16 -> f32:
  source contiguous f16
  result deinterleaved=2

f8 -> f32:
  source contiguous f8
  result deinterleaved=4

f32 -> f16:
  source deinterleaved=2
  result contiguous f16

f32 -> f8:
  source deinterleaved=4
  result contiguous f8

elementwise dense:
  all dense operands/results share the same layout

dense store:
  requests contiguous source
  if the stored value is assigned deinterleaved, baseline assignment inserts
  ensure_layout at the store use

two-way interleaved memory ops:
  `pto.vmi.deinterleave_load` produces two dense logical streams and requests
  contiguous layouts for both results
  `pto.vmi.interleave_store` consumes two dense logical streams and requests
  contiguous layouts for both inputs
  the deinterleave/interleave memory pattern is op semantics, not a VMI layout
```

### 5.2 Baseline Group Layout Requests

```text
group_reduce_add{f|i}:
  uses the group_reduce layout fact in section 4.1.
  The source and mask operands request the computed dense layout.
  The result is assigned group_slots(G, slots=8) or group_slots(G, slots=1).
  Floating-point `group_reduce_addf` carries `reassoc`; integer
  `group_reduce_addi` does not.

group_slot_load:
  result group_slots(G, slots=8) for packed slots
  result group_slots(G, slots=1) for row-local slots

group_broadcast:
  source requests group_slots(G,K)
  result requests one dense layout
  incompatible dense consumers are represented by ensure_layout after the
  broadcast result; a later optimization may clone/rematerialize the broadcast

group_store:
  source requests group_slots(G,K)
  explicit output stride attrs/operands decide store legality

group_slot_cast f32 -> f16:
  slots=1 row-local source/result is legal
  slots=8 packed source is illegal unless a future explicit helper or semantic
  op defines the packed slot-preserving transform
```

### 5.3 Tail And Memory Safety

Mask semantics and memory legality are separate:

```text
mask:
  decides which logical lanes participate in compute/store semantics

full_footprint_readable:
  decides whether a rounded-up physical load is allowed to read inactive lanes
```

The full-tile-readable proof must be explicit.  It may be carried by a
statically shaped memref source. Pointer-source runtime kernels should load a
rounded physical vector and use a mask to express logical active lanes.
`vmi-to-vpto` consumes only the op/type-local proof carrier; it does not inspect
surrounding MTE copies, producer bodies, callers, or later consumers to decide
whether inactive physical lanes are safe to read.

Example:

```text
S=32 tail num_groups=6:
  without full_footprint_readable:
    fast DINTLV_B32 full-tile load is illegal

  with full_footprint_readable:
    full 8-row physical tile may be loaded
    compute mask is PAT_VL48 per physical part
    group store mask is PAT_VL6

S=16 grouped tail active_elems_per_group=12:
  low 8-lane row half uses PAT_ALL
  high 8-lane row half uses lane_mod_8 < 4
  the same split applies before and after group_broadcast

one mask used by f32 and f16 consumers:
  f32 use materializes a b32 predicate
  f16 use materializes a b16 predicate
  vmi-to-vpto consumes the assigned per-use mask materialization
```

### 5.4 Case-Driven Request Matrix

The first implementation should build requests from the following finite table.
This table is deliberately case-derived; adding a new request kind requires a
new catalog case or a proof that it is equivalent to one listed here.

```text
dense store:
  requests dense contiguous source
  if source is deinterleaved, baseline assignment inserts ensure_layout at the
  store use.  A later optimization may fold that helper into a layout-aware
  store lowering such as vstsx2.

truncf f32 -> f16:
  requests source deinterleaved=2
  requests result contiguous f16

truncf f32 -> f8:
  requests source deinterleaved=4
  requests result contiguous f8

group_reduce_add{f|i}:
  computes E = sizeof(accumulator type), VLaneElems = 32B / E,
  L = 256B / E, and S = logical_lanes / num_groups
  S=VLaneElems requests source contiguous and result group_slots(G, slots=8)
  S=2*VLaneElems requests source deinterleaved=2 and result
  group_slots(G, slots=8)
  S=4*VLaneElems requests source deinterleaved=4 and result
  group_slots(G, slots=8)
  S>=L && S%L==0 requests source contiguous and result
  group_slots(G, slots=1)
  8-bit storage reaches this request only after an explicit cast to the
  accumulator type

group_broadcast:
  requests source group_slots(num_groups, slots=K)
  produces one assigned dense result layout
  incompatible dense consumers are represented by ensure_layout uses; a later
  optimization may clone/rematerialize the group_broadcast per consumer

group_store:
  requests source group_slots(num_groups, slots=K)
  explicit output stride attrs/operands decide store legality

dense elementwise add/mul/fma/min/max/select:
  requests all dense data operands and results use one dense layout
  mask operands request the same data layout and the consumer element
  granularity

group-slot elementwise add/mul/select:
  requests all group-slot operands and results use the same
  group_slots(num_groups, slots=K)
  rejects mixing dense and group_slots without explicit group_broadcast or
  group_store

group_slot_load:
  requests result group_slots(num_groups, slots=8) for packed unit-stride slots
  requests result group_slots(num_groups, slots=1) for row-local aligned slots

group_load:
  requests result block_deinterleaved=2/4 for S=16/S=32 block
  fragments, or contiguous for row-local full chunks

masked_load:
  requests result layout from its consumers
  requests mask layout matching the result
  requires explicit passthrough; padding is not synthesized

masked_store:
  requests dense source layout required by the store op
  requests mask layout matching the source layout and store element granularity
  does not choose memory safety for an earlier load

create_mask/create_group_mask:
  produces one assigned mask layout and granularity
  incompatible mask consumers are represented by ensure_mask_layout or
  ensure_mask_granularity; optimization may clone/rematerialize the mask op

dhist:
  requests acc/result contiguous !pto.vmi.vreg<256xui16>
  requests source contiguous !pto.vmi.vreg<Nxui8>
  requests mask contiguous with b8 granularity
  lowers each 256-lane source chunk by carrying two accumulator parts:
  bins 0..127 use VPTO histogram #bin=0, bins 128..255 use #bin=1
  final partial source chunks are represented by AND-ing the user mask with a
  valid-lane prefix mask before the VPTO histogram op

chist:
  same layout requests as dhist
  baseline lowering is disabled until target capability records whether the
  high-range VPTO cumulative result is global or range-local

scf.if/scf.for/call/return:
  requests equality across carried VMI values, yielded values, call operands,
  callee arguments, and function results
  baseline private/internal functions materialize at boundaries; optimization
  may specialize signatures
  public/external VMI boundaries are diagnostics until an ABI is defined
```

Important negative requests:

```text
ordinary dense add/mul/store/truncf cannot request group_slots
packed group_slots(slots=8) cannot request width-changing cast unless a packed
slot-preserving cast transform is explicitly represented
slots=1 group_store cannot request unit-stride row-major output until a pack or
unaligned-store transform is explicitly represented
```

### 5.5 Optimization Hooks

Baseline assignment resolves incompatible use-site requests by keeping one
assigned layout on the value and inserting explicit helpers at the use sites
that need another layout.  It does not clone producers, rematerialize cheap
ops, choose memory-fused layouts by cost, or specialize private function
signatures for performance.

Those choices belong to later VMI layout optimization passes.  They consume
the explicit helper IR and may rewrite it when the rewrite preserves the same
logical value and externally visible memory effect:

```text
ensure_layout + store:
  fold into a layout-aware store if the store can directly consume the source
  layout and still write row-major memory

producer + ensure_layout:
  clone/rematerialize the producer for that use only when the producer is cheap
  or has an explicit safe-read proof

elementwise chain + ensure_layout:
  sink or hoist materialization through pure layout-transparent ops

group_broadcast + incompatible dense consumers:
  type each group_broadcast op for its consumer layout; do not force one result
  layout across independent group_broadcast users

create_mask/create_group_mask + incompatible mask consumers:
  clone/rematerialize the mask producer per layout or predicate granularity

private function boundary:
  specialize function signatures only in an optimization pass; baseline
  assignment materializes at boundary uses
```

If no helper materialization or optimization rewrite is legal, the diagnostic
must name the value's assigned layout, the use-site requested layout, and the
op that requested it.

## 6. Layout Assignment Algorithm

`vmi-layout-assignment` is module-level.  It must see function/call/control-flow
connections before choosing layouts.

### 6.1 Variables

Create a layout variable for:

```text
1. every VMI OpResult
2. every VMI BlockArgument
3. every function argument/result that is allowed to carry VMI
4. every VMI mask value
```

Create a use-site request for:

```text
1. every operand use that requires a specific layout
2. every control-flow yield/branch/call/return edge
3. every memory operation that requires an explicit memory legality proof
```

### 6.2 Constraints

Hard constraints:

```text
group_slots cannot feed ordinary dense consumers
direct group-slot width-changing cast requires an explicit slot-preserving transform
public/external VMI function boundary requires a stable ABI or diagnostic
S=32 fast tail load requires full_footprint_readable or gather fallback
```

`slots = 1` row-local cast may satisfy the slot-preserving transform requirement.
Packed `slots = 8` f32->f16 remains a diagnostic unless a separate packed cast
or unpack/materialization transform is represented explicitly.

Equivalence constraints:

```text
dense add/mul/select:
  operands/results use same dense layout unless an explicit materialization is
  inserted at a use site

scf.if/scf.for:
  region yield operands and block arguments must have the same assigned layout
  as the region result/iter_arg
```

Canonical baseline constraints:

```text
S=16 group_reduce:
  request deinterleaved=2 unless the producer result already carries
  block_deinterleaved=2 as an explicit layout

one dense value feeding S=16 and S=32 group_reduce:
  keep the value's assigned layout and insert ensure_layout at both use sites
  that need deinterleaved=2 or deinterleaved=4

load/group_load:
  use the op's assigned result layout and explicit memory-safety attrs only

group_broadcast:
  keep one assigned dense result layout and communicate other dense use layouts
  through ensure_layout
```

### 6.3 Solving

Recommended solving order:

```text
1. Build function/control-flow SCCs.
2. Collect natural producer layouts and hard use-site layout requests.
3. Propagate equality constraints through dense elementwise ops and CFG edges.
4. Choose one deterministic assigned layout for each value or equivalence
   class.
5. Insert ensure_layout / ensure_mask_layout / ensure_mask_granularity at uses
   whose requested layout differs from the assigned layout.
6. Emit diagnostics for unsupported semantic constraints or missing explicit
   memory-safety proofs.
7. Rewrite VMI types and insert explicit helper ops.
```

Tie-breaking must be deterministic and deliberately simple.  Suggested priority:

```text
1. Preserve an explicit user-provided layout attr.
2. Preserve a unique producer natural layout when present.
3. Preserve an equality-class non-contiguous layout when required by a hard op.
4. Otherwise choose contiguous.
```

## 7. Control Flow And Functions

### 7.1 `scf.if`

All branch yields for one result must agree on one assigned layout.  If they do
not, assignment inserts materialization before `scf.yield` where possible.
The `scf.if` result type after assignment carries that layout, so
`vmi-to-vpto` does not need to inspect either branch body.

### 7.2 `scf.for`

Loop-carried VMI values are fixed-point variables:

```text
initial iter_arg layout
body block argument layout
yield operand layout
loop result layout
```

must converge to one layout.  If a body consumer needs another layout, it is a
use-site request inside the loop body.
The loop body block argument has no defining op.  Its layout is therefore part
of the block argument type after assignment, not information reconstructed from
the initial value or previous iteration during lowering.

### 7.3 Calls

Internal/private VMI function boundaries must make layout choices explicit in
the assigned IR.  The baseline implementation keeps function arguments in a
contiguous VMI ABI and inserts callee-entry `ensure_layout` helpers when the
callee body needs another layout.  Private helpers are then physicalized by
`vmi-to-vpto` and inlined before VPTO vecscope/backend emission so physical
`!pto.vreg`/`!pto.mask` values do not become a backend function ABI.  A later
private-function optimization may specialize signatures directly:

```text
func @producer() -> !vmi.vreg<256xf32, deinterleaved=4>
```

then physicalized by `vmi-to-vpto` into multiple VPTO function results.

Public/external VMI function boundaries are rejected until a stable VMI ABI is
defined.

## 8. vmi-to-vpto Contract

`vmi-to-vpto` receives layout-assigned VMI.  It performs no global reasoning.

For each op, the pattern:

```text
1. reads operand/result layouts
2. reads current op attrs and operand values
3. asks TypeConverter for ordered physical values
4. emits the locally implied VPTO lowering
5. fails if target capability or required local proof is absent
```

The pattern must not:

```text
1. inspect all users to decide result layout
2. inspect defining ops to decide source layout
3. choose between S=16 deinterleaved and block_deinterleaved
4. decide whether a load is full_footprint_readable
5. decide function signature specialization
```

Allowed local reads are deliberately narrower:

```text
arith.constant defining op:
  allowed only to materialize an operand of the current op, such as
  create_mask active_lanes or a constant memory offset

current VMI op body/attrs:
  allowed for op-local semantics, such as create_group_mask
  active_elems_per_group when lowering the create_group_mask op itself

helper materialization chain:
  allowed only to strip ensure_mask_layout / ensure_mask_granularity for
  static predicate analysis that does not choose a different layout or lowering

diagnostic embellishment:
  allowed only to improve an already-failed capability message, such as naming
  memref.subview after identity lane-to-address planning has failed
```

Anything else is a layout-assignment responsibility.  In particular, an
unsupported producer/consumer combination must be rejected before assignment
emits layout-assigned IR.  Section 3.44 is the model for supported partial S=32
grouped masks: assignment emits explicit contiguous and deinterleaved mask
values, and `vmi-to-vpto` lowers the deinterleaved mask op itself through
contiguous grouped-mask materialization followed by predicate deinterleave.  It
does not walk from `group_reduce_addf` to the mask producer to choose or reject
the lowering.  Dynamic `active_elems_per_group` follows the same rule: the
`create_group_mask` op lowers its own SSA scalar with vci/vshrs/vshls/vsub/vcmps
for contiguous chunks before any predicate deinterleave.

## 9. Physical Value Ordering

The OneToN lowering order is fixed.

```text
contiguous:
  chunk0, chunk1, ...

deinterleaved=F:
  part0_chunk0, part0_chunk1, ...,
  part1_chunk0, part1_chunk1, ...,
  ...
  part(F-1)_chunk0, ...

group_slots(G,K):
  slot_block0, slot_block1, ...
```

Two physical bundle entries may alias the same VPTO SSA value when the current
op semantics prove they have the same contents, such as group_broadcast feeding both
parts of a `deinterleaved=2` broadcast result.  Arity still follows the layout;
aliasing is not a different layout.

## 10. Diagnostics

Diagnostics are part of the design.  They must name:

```text
1. the VMI op
2. source logical type
3. assigned source layout
4. requested layout
5. missing local proof or disabled fallback
6. suggested rewrite when available
```

Examples:

```text
dense store of group_slots:
  use group_store, group_broadcast, or explicit group-pack

packed group-slot f32->f16:
  group_broadcast before truncf, or keep group_store as f32

S=32 tail without full_footprint_readable:
  mark source full_footprint_readable or enable stable gather fallback

S=32 group_load with unaligned source_group_stride:
  choose a stride divisible by 8 f32 elements or enable stable gather fallback

public VMI function boundary:
  make function internal, inline before assignment, or define ABI layout
```

## 11. Implementation Migration Checks

The design is useful only if the implementation removes duplicated decision
points instead of renaming them.  The migration target is:

```text
assignment:
  computes assigned layouts, records use-site requests, inserts ensure_* helpers,
  and diagnoses unsupported semantics
  does not clone/rematerialize producers
  does not choose memory-fused layouts by cost
  does not inspect sibling users to optimize a value

layout optimization:
  consumes explicit ensure_* helpers
  may fold ensure_layout into layout-aware consumers
  may clone/rematerialize cheap producers
  may sink/hoist materialization through pure elementwise chains
  may specialize private function signatures

vmi-to-vpto:
  consumes current op attrs/operands, assigned operand/result layouts, and
  explicit helper ops
  performs local physical shape and target-capability checks
  does not recover layout plans from producers, sibling users, CFG regions, or
  callees/callers
```

Concrete implementation debt to remove:

```text
1. Move assignment-side data/mask rematerialization into
   vmi-layout-rematerialize.  Baseline assignment should insert ensure_* for
   mismatched uses.
2. Keep `VMILayoutSupport` as target capability and layout-shape queries, not
   as a shared plan table.  Group-reduce layout math now lives in
   `getPreferredGroupReduceLayoutFact`.  Dense cast layout shape now lives in
   `getPreferredCastLayoutFact`.  Helper materialization gates use
   `canMaterializeDataLayout`, `canMaterializeMaskLayout`, and
   `canMaterializeMaskGranularity`.
3. Assignment, validation, and lowering may call layout fact helpers, but must
   not each independently derive VLaneElems/groupSize/factor/slots rules.
4. Keep store-fold, rematerialization, and sink/hoist as local rewrites over
   explicit ensure_* IR.  They must not walk sibling users to rediscover why the
   helper exists.
5. Update pass descriptions, diagnostics, and tests so "assignment only" output
   is legal with helpers, and optimized output is a separate, equivalent IR
   form.
```

Regression tests should prove the boundary:

```text
assignment only:
  multi-consumer values keep one assigned layout and use ensure_* at mismatched
  uses

fold-consumers:
  ensure_layout + store becomes a layout-aware store only when the consumer can
  preserve the same row-major memory effect

rematerialize:
  cheap producer + ensure_layout becomes a cloned/rematerialized producer; with
  the pass disabled, the ensure_layout form remains legal

vmi-to-vpto:
  rejects any residual need for producer/user context with VMI-LAYOUT-CONTRACT
```

## 12. Design Completion Criteria

The design is complete only when:

```text
1. every case in vmi-layout-lowering-cases.md maps to assignment requests,
   explicit helpers, or a precise diagnostic
2. every VMI-to-VPTO lowering can be emitted without looking at producer/user
   context
3. every unsupported case has a precise capability diagnostic
4. every control-flow/function boundary materializes, specializes in an
   optimization pass, or diagnoses
5. every mask has explicit data layout and predicate granularity
6. every positive case has end-to-end lit coverage
7. every simulator-supported positive case has simulator validation
```
