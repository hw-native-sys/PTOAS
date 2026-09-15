# VMI E2B Scale Broadcast Optimization Study

本文推演 VMI 是否能把 block quant 中的 scale broadcast 自动优化成
`E2B_B16` load。结论是：

```text
group_slot_load + group_broadcast 足以表达逻辑语义。

它不足以单独触发 E2B，因为 E2B 是某个 physical chunk layout 下的
materialization，不是 dense logical broadcast 的直接 lowering。

如果后续 layout 已经由 consumer requirement 或 target-specific layout
optimization 选成 E2B-compatible 形态，vmi-to-vpto 可以把对应 chunk lower
成 E2B。

如果想从普通 dense quant IR 自动得到 CCE 的 DINTLV/E2B 形态，需要一个
target-specific layout optimization/cost selection 阶段整体选择这套计划。
```

## 1. Logical Quant Semantics

`ComputeY1ToFP8` 的 surface VMI 语义应保持 dense quant：

```text
for i in 0..255:
  y[i] = fp8(x[i] * scale[i / 32])
```

也就是 8 个 scale，每个覆盖 32 个 dense logical lanes：

```text
s0 x32, s1 x32, ..., s7 x32
```

对应 VMI 形态是：

```text
%x = pto.vmi.load %x_base[%x_off]
  : !pto.ptr<f16, ub> -> !pto.vmi.vreg<256xf16>

%scale_slots = pto.vmi.group_slot_load %scale_base[%scale_off], %c1
    {num_groups = 8}
  : !pto.ptr<bf16, ub> -> !pto.vmi.vreg<8xbf16>

%scale = pto.vmi.group_broadcast %scale_slots {num_groups = 8}
  : !pto.vmi.vreg<8xbf16> -> !pto.vmi.vreg<256xbf16>
```

This form is the canonical logical IR.  The source scale should be BF16 payload,
not FP16, because the CCE implementation loads `uint16_t` values and later
reinterprets them as `vector_bf16`.

`num_groups = 16` would express a different algorithm:

```text
16 scale values, each covering 16 dense lanes
```

That is not equivalent unless the input memory redundantly stores
`s0, s0, s1, s1, ...`, which is not what the CCE kernel does.

## 2. E2B_B16 Semantics

`E2B_B16` is a VPTO load distribution mode.  For a b16 destination register it
loads 8 source elements and expands each one to 16 consecutive destination
lanes:

```text
dst[j] = src[floor(j / 16)]   for j = 0..127
```

The result is:

```text
s0 x16, s1 x16, ..., s7 x16
```

So `E2B_B16` does not directly materialize the dense VMI broadcast
`8 -> 256`.  It materializes one 128-lane physical view that becomes useful only
after the x data and later f32 computation have been split into compatible
physical chunks.

## 3. Why CCE Can Use E2B

The CCE FP16 path uses a physical implementation shape like:

```text
vlds(x0F16, x1F16, xHalf, stride, DINTLV_B16, POST_UPDATE)
vlds(scaleForMulFP16, scale_base, 0, E2B_B16)

vcvt(x0_even_f32, x0F16, PART_EVEN)
vcvt(x0_odd_f32,  x0F16, PART_ODD)
vcvt(x1_even_f32, x1F16, PART_EVEN)
vcvt(x1_odd_f32,  x1F16, PART_ODD)

vcvt(scale_f32, (vector_bf16 &)scaleForMulFP16, PART_EVEN)
```

`DINTLV_B16` splits the dense 256-element row into two 128-lane physical streams.
After each stream is converted from f16 to f32, the computation is effectively
four 64-lane f32 chunks:

```text
x0 even part
x0 odd part
x1 even part
x1 odd part
```

For every one of those chunks, the needed scale pattern is:

```text
s0 x8, s1 x8, ..., s7 x8
```

`E2B_B16` produces:

```text
s0 x16, s1 x16, ..., s7 x16
```

Then `vcvt PART_EVEN` produces:

```text
s0 x8, s1 x8, ..., s7 x8
```

Because every scale value is duplicated in adjacent even/odd b16 positions,
`PART_EVEN` and `PART_ODD` would produce the same f32 scale chunk.  The CCE code
computes the scale chunk once and reuses it for all four x chunks.

## 4. What Is A Legal Automatic Optimization?

The following rewrite is not legal as a standalone local rule:

```text
group_slot_load + group_broadcast(8 -> 256)  =>  E2B_B16
```

It is invalid because the left side is a dense 256-lane logical value, while
`E2B_B16` produces a 128-lane physical value with a different lane repetition
count.

A legal E2B lowering must be conditional on the assigned physical layout:

```text
if the broadcasted scale value is required in physical chunks where each chunk
needs s0 x16 ... s7 x16 at b16 width, or s0 x8 ... s7 x8 after bf16->f32,
then that chunk may be materialized with E2B_B16.
```

In other words:

```text
group_slot_load + group_broadcast
  is the logical source pattern

consumer-required or target-selected layout
  determines whether any physical chunk is E2B-compatible

vmi-to-vpto
  lowers only those compatible chunks to E2B
```

`group_slot_load` alone cannot lower to E2B.  A group-slot value has only group
slots as semantic lanes.  `E2B_B16` already produces broadcasted physical lanes.
The `group_broadcast` use is required to justify reading those lanes.

## 5. Layout Selection Boundary

Deinterleaved layout must not be inferred only because E2B would be cheaper.
The selected layout must be explicit before `vmi-to-vpto`.  That layout can come
from either side:

```text
consumer requirement:
  a later op requires a particular layout.

producer natural layout:
  the producing op has a declared, deterministic natural layout that is legal
  for all of its uses.
```

`group_broadcast` is a materialization op, so it may define or participate in an
E2B-friendly natural layout when that layout is part of the declared layout
contract.  That is still a layout-assignment decision, not a hidden
`vmi-to-vpto` peephole.  Do not reuse `block_deinterleaved` as an ad-hoc
broadcast split knob; it specifically describes fixed-32B block distribution
and has existing producer/consumer meanings.

Baseline layout assignment may still choose conservative contiguous layouts even
when a target-specific fused implementation exists.

Therefore this optimization has two valid implementation levels.

### 5.1 Compatible-Layout Lowering Shortcut

If some earlier layout pass has already assigned an E2B-compatible physical
layout, `vmi-to-vpto` may lower the scale chunk with `E2B_B16`.

This is a local deterministic lowering.  It does not discover the CCE plan by
itself.  It only avoids a generic `vsldb + vselr` materialization when the
assigned layout has already made the required physical chunk shape explicit.

### 5.2 Producer Natural Layout

For simple broadcasts, the producer itself may choose an E2B-friendly natural
layout when that layout satisfies every use.

Example for b16, using an existing DINTLV-like element-parity layout:

```text
logical 1 -> 32:
  s0 x32

layout:
  deinterleaved=2

physical part 0:
  s0 x16

physical part 1:
  s0 x16
```

The two physical parts can share one E2B materialization or use two identical
E2B materializations.  This is a general layout choice for the broadcast result,
not a quant-specific graph rewrite.

For a uniform `1 -> 32` or per-group `x32` broadcast, `deinterleaved = 2`
yields 16 lanes of the same group per physical part and is
closer to an even/odd `DINTLV_B16` data layout.

For the MX quant scale:

```text
logical 8 -> 256:
  s0 x32, s1 x32, ..., s7 x32

layout:
  deinterleaved=2

physical part 0:
  s0 x16, s1 x16, ..., s7 x16

physical part 1:
  s0 x16, s1 x16, ..., s7 x16
```

Each physical part is directly `E2B_B16`-compatible.
The implementation should run the E2B compatibility query over the assigned
lane mapping. It should not infer a new meaning for `block_deinterleaved`.

### 5.3 Target-Specific Layout Optimization

To automatically discover the complete CCE plan from canonical dense quant IR,
add an optional target-specific layout optimization before `vmi-to-vpto`.

That pass may select a cheaper equivalent implementation for the whole quant
subgraph:

```text
dense x load
f16/bf16 -> f32 conversion
scale group_slot_load + group_broadcast 8 -> 256
scale bf16 -> f32 conversion
mul
fp32 -> fp8 conversion/store
```

The pass must rewrite or annotate the VMI layout-assigned IR so that
`vmi-to-vpto` no longer has to infer the plan from context.

Expected selected physical plan:

```text
x load:
  vlds DINTLV_B16 into two b16 streams

scale load:
  vlds E2B_B16 into one b16 stream

scale conversion:
  vcvt PART_EVEN into one 64-lane f32 stream

mul:
  reuse that scale f32 stream for the four x f32 chunks
```

This is an optimization, not a correctness requirement.  If the optimizer does
not fire, the canonical dense VMI program still has a valid generic lowering.

## 6. Candidate Match Preconditions

A target-specific optimization may match the CCE-style scale pattern only under
strict conditions:

```text
scale_slots:
  pto.vmi.group_slot_load
  num_groups = 8
  source_group_stride = 1
  source element width = 16 bits
  semantic type is bf16 or a bitcastable ui16 payload later interpreted as bf16

scale broadcast:
  pto.vmi.group_broadcast
  same num_groups = 8
  dense logical result has 256 b16 lanes for this case

scale conversion:
  bf16 -> f32
  conversion has no rounding/exception behavior that distinguishes duplicated
  even and odd source lanes

x path:
  dense logical row has 256 f16 or bf16 lanes
  the target plan can legally compute the row as four 64-lane f32 chunks

uses:
  no user observes the intermediate dense scale layout in a way that prevents
  rematerialization or chunk reuse
```

The optimization should reject or skip the pattern if any of these conditions are
not proven.

## 7. Correctness Sketch

Let the logical dense lane be `i`.

The canonical VMI scale value is:

```text
scale_dense[i] = s[floor(i / 32)]
```

The CCE physical decomposition maps each dense lane into one of four f32 chunks.
For a chunk-local f32 lane `k`:

```text
dense lane = 4 * k + delta
delta in {0, 1, 2, 3}
```

Then:

```text
floor((4 * k + delta) / 32) = floor(k / 8)
```

So every f32 chunk needs:

```text
scale_chunk[k] = s[floor(k / 8)]
```

`E2B_B16` plus `vcvt PART_EVEN` gives:

```text
e2b_b16[j] = s[floor(j / 16)]        for j = 0..127
scale_f32[k] = e2b_b16[2 * k]
             = s[floor((2 * k) / 16)]
             = s[floor(k / 8)]
```

That matches the required `scale_chunk[k]` for all four f32 chunks.

## 8. Recommendation

Prefer adding a target-agnostic VMI `group_broadcast_load` logical memory op if
we want to make this optimization robust and local.  The op should mean:

```text
load one source value per logical group, then broadcast that value to every lane
in the group.
```

It must not mean `E2B`.  `E2B_B16` is only one possible lowering when the
assigned layout is compatible.

The unfused logical IR remains valid:

```text
group_slot_load + group_broadcast
```

but a canonicalization/layout-prep pass may fuse it to:

```text
group_broadcast_load
```

when the group-slot value has no separate semantic users.

Then implement E2B support in phases:

```text
1. Ensure the example PTO uses the correct logical semantics:
   bf16 scale, num_groups = 8, dense 8 -> 256 broadcast.

2. Add group_broadcast_load as a logical VMI memory op, plus canonicalization
   from group_slot_load + group_broadcast when legal.

3. Add a compatible-layout lowering shortcut:
   when layout assignment already exposes an E2B-compatible chunk, lower the
   group_broadcast_load chunk with vlds E2B_B16.

4. Add an optional target-specific quant layout optimization:
   recognize the whole dense quant subgraph and select the DINTLV/E2B plan when
   it is legal and profitable.
```

This keeps VMI logical semantics independent from physical layout, while still
leaving a clear path to recover the CCE optimization automatically.

## 9. Generalized E2B Broadcast Optimization

The scale case above is one instance of a broader rule: E2B is a physical
materialization primitive for a packet of repeated group slots.  It is not tied
to MX quant, but its legality depends on the physical chunk layout and on the
load distribution's carrier element width.

### 9.1 E2B As A Packet Primitive

For the verified `B16` case:

```text
E2B_B16 packet:
  source slots per packet = 8
  destination lanes per packet = 128 b16 lanes
  repeat per source slot = 16 b16 lanes

dst[lane] = src[base_slot + floor(lane / 16)]
```

This can materialize a physical chunk that needs:

```text
s0 x16, s1 x16, ..., s7 x16
```

The optimization should reason in terms of physical chunks:

```text
logical group_broadcast
  source group slot for logical lane i = floor(i / logical_group_size)

assigned physical layout
  maps physical chunk lane l to logical lane i(l)

E2B-compatible chunk
  floor(i(l) / logical_group_size) = base_slot + floor(l / 16)
```

If this equality holds for a b16 physical chunk, the chunk can be loaded with
`E2B_B16` instead of materializing the broadcast with `vselr`.

### 9.2 Direct 1 -> 16

A logical `1 -> 16` b16 broadcast is directly compatible with one E2B group:

```text
s0 x16
```

However, `E2B_B16` is naturally an 8-group packet:

```text
s0 x16, s1 x16, ..., s7 x16
```

So a single `1 -> 16` use may lower to E2B only under one of these conditions:

```text
packed case:
  the compiler can pack eight independent 1 -> 16 broadcasts into one E2B load.

partial-live case:
  only one 16-lane group is live, and the target semantics prove inactive E2B
  groups do not require valid source memory or can be safely over-read.

full-packet case:
  the logical IR actually contains eight adjacent groups, even if the current
  consumer observes only one group through a layout/mask.
```

If these conditions are not proven, `BRC_B16`, `vdup`, or the existing generic
broadcast lowering is safer than E2B.  In particular, do not introduce an E2B
load that reads seven extra source values unless the memory safety rule is
explicit.

### 9.3 1 -> 32 Via Deinterleaved Reuse

A dense logical `1 -> 32` b16 broadcast does not fit one E2B group in a single
contiguous physical chunk:

```text
logical: s0 x32
E2B group: s0 x16
```

It becomes E2B-compatible when the assigned physical layout splits those 32
logical lanes into two 16-lane physical uses:

```text
physical use A: s0 x16
physical use B: s0 x16
```

This split can use the existing DINTLV-like element-parity layout:

```text
#pto.vmi.layout<deinterleaved=2>
```

For logical lanes `0..31`, this maps:

```text
even lanes 0,2,...,30  -> physical part 0 lanes 0..15
odd lanes 1,3,...,31   -> physical part 1 lanes 0..15
```

Because all 32 logical lanes carry the same `s0`, each part still sees
`s0 x16`.  The lowering rule should check the resulting group index function,
not invent a new layout spelling.

Then the compiler has two valid strategies:

```text
reuse:
  materialize one E2B group/chunk and map both physical uses to the same value.

duplicate:
  materialize the same E2B group twice if reuse would violate scheduling,
  lifetime, or destructive-update constraints.
```

This is the mechanism behind the MX quant scale case:

```text
dense logical scale: 8 groups, each x32
physical f16/bf16 streams: each group appears as x16 per stream
```

The optimization is legal only if the 32 logical lanes are split by layout.  It
is not legal as a direct E2B chunk load for a contiguous physical chunk that
genuinely needs `s0 x32` inside one chunk; that would require a separate
duplicate/interleave/concat materialization.

### 9.4 N -> N * 16 And N -> N * 32

For b16 group broadcasts with consecutive slots and unit source stride:

```text
N -> N * 16
```

can be lowered by E2B in packets of 8 groups when the physical chunk sees the
groups in E2B order:

```text
for base_slot in 0, 8, 16, ...
  load src[base_slot : base_slot + 8] with E2B_B16
```

Tail packets require either a proven safe masked/partial E2B form or a generic
fallback.

For:

```text
N -> N * 32
```

E2B is profitable when the assigned layout decomposes each 32-lane logical group
into two 16-lane physical uses.  That assigned layout may be the
`group_broadcast` producer's natural layout, or it may be required by a
downstream consumer.  The lowering then reuses or duplicates the corresponding
E2B materialization for those two uses.  This rule extends to:

```text
N -> N * (16 * F)
```

when the layout decomposes each logical group into `F` physical 16-lane uses.

### 9.5 Type Generalization

E2B is a carrier-width load distribution.  For `E2B_B16`, the load itself is
valid for 16-bit carriers:

```text
bf16
f16
ui16 / si16 payloads
other 16-bit bit patterns whose consumers preserve the intended interpretation
```

The optimization must keep type interpretation outside the load:

```text
bf16 scale + extf to f32:
  E2B_B16 may feed vcvt bf16 -> f32.

f16 broadcast:
  E2B_B16 may materialize repeated f16 lanes if the consumer expects f16.

ui16 payload later bitcast to bf16:
  E2B_B16 may load the ui16 carrier, but the later bitcast/interpretation must
  remain explicit in VMI or in the selected lowering plan.
```

Do not infer a floating-point type from E2B itself.  `E2B_B16` only says how UB
bytes are placed into b16 lanes.

`E2B_B32` is the b32 member of the same distribution family.  The VPTO verifier
accepts `E2B_B32`, the ISA docs list E2B for `b16` and `b32`, and CCE quant code
uses `E2B_B32` in FP32 paths.  It follows the same 8-source-slot packet rule:

```text
E2B_B16: 8 source slots * 16 lanes/slot = 128 b16 lanes
E2B_B32: 8 source slots *  8 lanes/slot =  64 b32 lanes
```

The implemented E2B broadcast optimization therefore supports:

```text
b16 contiguous:      logical 1 -> 16
b16 deinterleaved=2: logical 1 -> 32
b32 contiguous:      logical 1 -> 8
b32 deinterleaved=2: logical 1 -> 16
```

There is no `E2B_B8` in the documented load distribution family, so b8
broadcasts should use other distributions or generic materialization.

### 9.6 Broadcast Generalization

E2B can optimize `pto.vmi.group_broadcast` when all of these are true:

```text
source:
  group slots come from consecutive memory slots
  source_group_stride = 1
  slot type matches the E2B carrier width

broadcast:
  each physical chunk needs a run-length pattern compatible with the E2B repeat
  count for that carrier width

layout:
  the run-length pattern is visible in the assigned layout before vmi-to-vpto

uses:
  rematerializing or reusing the E2B packet does not change observable memory or
  arithmetic semantics
```

E2B is generally not the right primitive for ordinary scalar `pto.vmi.broadcast`
unless the scalar value is already in memory as an E2B packet or the compiler can
pack several independent scalar broadcasts into one E2B load.  For a scalar
stored once in memory and needed in every lane, `BRC_B16/B32`, `BRC_BLK`, or a
register `vdup` is usually the more direct representation.

### 9.7 Implementation Shape

The recommended implementation order is:

```text
1. Keep VMI semantics canonical:
   group_slot_load + group_broadcast is the desugared meaning.

2. Optionally canonicalize to group_broadcast_load:
   this keeps memory source and broadcast semantics in one local op.

3. Add an E2B compatibility query over assigned physical chunks:
   given source slots, result layout, carrier width, and live lanes, answer
   whether a chunk's group-index function is E2B-shaped.

4. Lower compatible chunks to E2B packets:
   generate one E2B load per needed packet, or reuse an existing packet when
   multiple physical uses require identical contents.

5. Add a later target-specific layout optimizer:
   it may choose layouts that expose E2B-compatible chunks, but only by
   rewriting/annotating layout-assigned VMI before vmi-to-vpto.
```

The compatibility query should return a reason when it rejects a candidate:

```text
non-unit source stride
non-consecutive group slots
unsupported carrier width
tail packet lacks safe partial E2B semantics
physical lane mapping is not E2B-shaped
extra source memory read would be unsafe
consumer observes a different dense layout
```

This keeps the optimization auditable and prevents E2B from becoming an implicit
layout-changing peephole.

## 10. Recognition, Solidification, Propagation, Lowering

This section describes how an implementation should carry the optimization from
canonical VMI to VPTO without making `vmi-to-vpto` rediscover hidden context.

### 10.1 Recognize Information

Run recognition after hard layout assignment, when every relevant value already
has an explicit layout.

Recognize the source shape:

```text
%slots = pto.vmi.group_slot_load %base[%off], %stride {num_groups = G}
%bcast = pto.vmi.group_broadcast %slots {num_groups = G}
```

or the already-fused form:

```text
%bcast = pto.vmi.group_broadcast_load %base[%off], %stride {num_groups = G}
```

Collect candidate facts:

```text
source memory:
  base pointer
  offset
  source_group_stride
  element carrier width
  memory element type

logical broadcast:
  num_groups = G
  logical lanes = N
  logical group size S = N / G

assigned result layout:
  physical arity
  physical lanes per chunk
  logical lane mapped to each physical lane

uses:
  whether the broadcast feeds elementwise ops, extf/truncf, stores, or multiple
  independent consumers
```

Then compute an E2B packet plan per physical chunk.  For `E2B_B16`, a physical
chunk is compatible when:

```text
group_index_for_physical_lane(l) = base_slot + floor(l / 16)
```

for all live lanes in that chunk.

Reject the candidate if:

```text
source_group_stride != 1
source slots are not consecutive
carrier width is unsupported
the assigned layout does not produce E2B-shaped chunks
tail/partial packet would read memory that is not proven valid
the group_slot_load has other non-rematerializable users
```

This recognition is an analysis step.  It must not silently change layouts.

### 10.2 Solidify Information

`vmi-to-vpto` should not have to look at an arbitrary
`group_slot_load -> group_broadcast` use-def chain and decide to suppress one
load while replacing another op with E2B.  The optimization pass must solidify
the decision in the layout-assigned IR.

The preferred solidification is a target-agnostic logical memory op:

```text
%bcast = pto.vmi.group_broadcast_load %base[%off], %stride {num_groups = G}
  : !pto.ptr<T, ub> -> !pto.vmi.vreg<NxT, layout = ...>
```

Semantic definition:

```text
group_size = N / G
for logical lane i:
  group = floor(i / group_size)
  result[i] = base[off + group * stride]
```

This op is not target-specific and does not promise E2B.  It is exactly the
fused logical form of:

```text
%slots = pto.vmi.group_slot_load %base[%off], %stride {num_groups = G}
%bcast = pto.vmi.group_broadcast %slots {num_groups = G}
```

The fused op makes lowering local because the memory source, stride, group count,
result type, and assigned layout are all available on one op.  A generic lowering
can still materialize it with `vsldb + vselr`; an optimized lowering may choose
`E2B_B16` for compatible physical chunks.

The current implementation is intentionally narrower: because
`group_broadcast_load` does not yet have a generic `vsldb + vselr` lowering,
layout assignment fuses `group_slot_load + group_broadcast` only when the fused
op is already an E2B-compatible b16 candidate.  Non-E2B shapes stay in the
unfused form and continue to use the existing `group_slot_load` plus
`group_broadcast` lowering path.

Canonicalization rules:

```text
group_slot_load + group_broadcast -> group_broadcast_load
  when the group_slot_load has exactly that broadcast use, or when cloning the
  load is legal and profitable for that use.

group_broadcast_load -> group_slot_load + group_broadcast
  remains a valid conceptual expansion for verification, documentation, and
  generic fallback reasoning.
```

Solidification must preserve semantics for multi-use values:

```text
if all uses consume only the broadcasted value:
  replace with one shared group_broadcast_load.

if only one use can benefit from the fused form:
  clone/rematerialize that use-site load as group_broadcast_load and keep the
  original group_slot_load for other users.

if the group_slot_load itself has semantic group-slot users:
  do not delete it; add a separate group_broadcast_load only if the extra memory
  read is legal or if load cloning is otherwise proven safe.
```

### 10.3 Propagate Information

After solidification, propagation should use ordinary VMI layout rules whenever
possible:

```text
elementwise ops:
  preserve the assigned layout when operands agree.

ensure_layout:
  makes layout transitions explicit when one use needs E2B-compatible chunks and
  another use needs a different layout.

rematerialization:
  may clone group_broadcast_load per use-site instead of forcing a single layout
  for all consumers.
```

For casts, propagation may need a targeted rule.  The important MX quant case is:

```text
E2B_B16 gives:
  s0 x16, s1 x16, ..., s7 x16

bf16 -> f32 PART_EVEN gives:
  s0 x8, s1 x8, ..., s7 x8
```

If multiple f32 physical chunks require that same `s0 x8 ... s7 x8` pattern,
the post-assignment plan may mark them as the same rematerialized value.  The
lowerer can then generate one `vcvt PART_EVEN` and map several logical physical
chunks to the same VPTO value.

This reuse fact must be derived from the assigned lane mapping and the E2B packet
plan.  It must not rely on a later CSE pass accidentally proving the duplicate.

### 10.4 Implement Lowering

`vmi-to-vpto` should lower `group_broadcast_load` locally.  It may choose E2B
only when the op's assigned layout and source facts produce an explicit
E2B-compatible packet plan.

For each E2B packet:

```text
1. compute the source pointer:
   base + packet_base_slot

2. emit:
   pto.vlds {dist = "E2B_B16"}

3. map the emitted VPTO value to the physical result chunk(s) recorded in the
   group_broadcast_load packet plan.
```

For `1 -> 32` under `deinterleaved=2`:

```text
logical group:
  s0 x32

physical part 0:
  s0 x16

physical part 1:
  s0 x16

lowering:
  emit one E2B packet if reuse is legal, or two identical E2B packets if
  scheduling/lifetime constraints require duplication.
```

For MX quant scale after bf16->f32:

```text
1. emit E2B_B16 for the b16 scale packet.
2. emit vcvt PART_EVEN to produce the f32 packet.
3. map that f32 packet to every physical f32 chunk whose lane mapping requires
   s0 x8, s1 x8, ..., s7 x8.
4. lower vmul normally using the assigned physical chunks.
```

### 10.5 Where Layout Choices Happen

There are three levels of optimization:

```text
level 0: no E2B
  canonical group_broadcast lowers through generic vselr materialization.

level 1: E2B for already-compatible layouts
  recognition sees the assigned layout is E2B-shaped and solidifies an E2B
  materialization.

level 2: choose E2B-compatible layouts
  an optional layout optimization changes/rematerializes layouts before
  recognition, for example selecting deinterleaved=2 for a
  broadcast use when all consumers can accept that layout.
```

The full CCE-like optimization for `ComputeY1ToFP8` is level 2:

```text
x path:
  select DINTLV-compatible layout for the dense x load/cast path.

scale path:
  select an E2B-compatible broadcast materialization.

compute path:
  keep mul/trunc/store in the selected physical chunk layout or insert explicit
  layout materialization where required.
```

### 10.6 Test Plan

Add focused tests in phases:

```text
positive:
  bf16 group_slot_load stride=1 + group_broadcast 8->256 assigned to
  deinterleaved=2 lowers scale chunks with E2B_B16.

positive:
  f16 1->16 or packed 8*(1->16) lowers to E2B only when source memory safety is
  proven by full packet or supported partial semantics.

positive:
  1->32 assigned to deinterleaved=2 maps two physical uses to one
  E2B packet or to two explicit duplicate packets.

positive:
  f32 1->8 lowers to E2B_B32, and f32 1->16 under deinterleaved=2
  maps two physical uses to one E2B_B32 packet.

negative:
  source_group_stride != 1 falls back or diagnoses the E2B optimization.

negative:
  non-E2B-shaped assigned layout falls back to generic group_broadcast lowering.

negative:
  partial packet without proven safe memory read does not emit E2B.

deferred:
  E2B_B32 remains disabled until simulator/spec tests confirm the exact lane
  mapping.
```
