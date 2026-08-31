# VMI Layout Lowering Cases

本文是 VMI layout/lowering 的典型 case catalog，不是完整设计总文档。它只回答一个问题：
一个 VMI logical vector 在某个场景下选择某种 layout 后，`vmi-to-vpto` 必须生成什么
VPTO 结果。这里不写动机式描述；每个场景都给出 layout assignment 和 lowering result。

## 1. Layout Families

### 1.1 Dense Layout

Dense layout 的每个 logical lane 都有语义值。

```text
#pto.vmi.layout<contiguous>
```

Physical ordering:

```text
chunk c, lane l -> logical lane c * L + l
```

`L` is the physical lanes per 256B VPTO vector register for the element type.

```text
#pto.vmi.layout<deinterleaved = F>
#pto.vmi.layout<block_deinterleaved = F>
```

The two split layouts are independent kinds:

```text
deinterleaved=F distributes individual elements by logical lane modulo F.
block_deinterleaved=F distributes fixed 32B blocks modulo F; the element count
per block is derived from the carrier type.
```

Logical-to-physical mapping:

```text
deinterleaved:
  part p = i % F
  physical lane = i / F

block_deinterleaved:
  B = 32B / sizeof(carrier element)
  block q = i / B
  in_block lane r = i % B
  part p = q % F
  part_block t = q / F
  physical lane = t * B + r
```

Required invariants:

```text
F is 2 or 4
N must fill the physical chunks required by each direct lowering in this document
```

### 1.2 Group-Slot Layout

Group-slot layout is not dense. Only `G` lanes have semantic values.

```text
#pto.vmi.layout<num_groups = G, slots = K>
#pto.vmi.layout<num_groups = G, slots = K, lane_stride = LS>
```

Physical slot mapping:

```text
N = logical lane count
S = N / G                 // logical lanes per source group

slot_block(g) = g / K
slot_lane(g)  = (g % K) * LS
```

Required invariants:

```text
G > 0
K > 0
G % K == 0
K must fit in the physical vreg element count
LS > 0
```

`LS` defaults to 1 and is counted in logical element-sized physical slots.  It
is used when the group result value is intentionally stored with a regular lane
gap.  For example, `ui8 lane_stride=4` places group slots in byte positions 0,
4, 8, ... and can be lowered to a b32 carrier plus `PK4_B32` store.

`K` is selected by the producer/consumer layout support rule. It is not always 8. For
`VCGADD`-packed results, `K = 8` matches the eight 32B block results written to
the low lanes of one destination vreg. For row-local reductions where each
logical group already occupies one full 256B vreg, `K = 1` keeps each group's
scalar result in lane 0 of its own physical vreg and avoids an unsupported
cross-vreg scalar pack.

Only these lanes are semantic:

```text
physical slot block slot_block(g), lane slot_lane(g)
```

All other lanes are undefined for ordinary VMI consumers. They may only be read
by group-aware ops that define how to interpret group slots.

## 2. Layout Support Selection Rules

VMI cast ops must not hard-code one physical `vcvt` lowering as their semantic
layout rule.  Layout assignment records the required value layout; target
support queries only answer whether that layout can be materialized or lowered.

```text
dense cast:
  source/result are dense layouts.
  lowering may require ordinary deinterleaved(F) around VCVT.

group-slot cast:
  source/result are both group_slots(G,K).
  lowering preserves slot_block(g) and slot_lane(g). Width-changing casts are
  legal only when slot-preserving VPTO lowering support exists, or when the cast
  can be commuted through a later group-aware consumer such as group_broadcast.
```

Illegal consumer mix:

```text
group_slots value -> ordinary dense store/add/mul
```

This must fail unless an explicit semantic op converts the group-slot value:

```text
group_broadcast
group_store
future explicit group-pack op
```

Contiguous memory loads may produce a non-contiguous physical value directly
when the requested result layout is a dense deinterleaved layout.  This is a
lowering choice, not a separate layout family.

```text
pto.vmi.load -> #pto.vmi.layout<contiguous>
  lower as:
    vlds NORM for each physical chunk

pto.vmi.load -> #pto.vmi.layout<deinterleaved=2>
  lower as:
    vldsx2 DINTLV_B* for each pair of physical chunks

pto.vmi.load -> #pto.vmi.layout<deinterleaved=4>
  lower as:
    two vldsx2 DINTLV_B* operations for each four-chunk group
    followed by two vdintlv operations to split mod4 parts

pto.vmi.load -> #pto.vmi.layout<block_deinterleaved = F>
  lower using the producer-specific path or fall back to explicit
  materialization.  Do not treat DINTLV_B* as a block-fragment layout.
```

The `deinterleaved = 4` result order remains the normal VMI physical part
order:

```text
results = [part0 chunks..., part1 chunks..., part2 chunks..., part3 chunks...]
```

For one full `256xf32` tile:

```text
%even0, %odd0 = pto.vldsx2 %base[%off0],   "DINTLV_B32"
%even1, %odd1 = pto.vldsx2 %base[%off128], "DINTLV_B32"

%part0, %part2 = pto.vdintlv %even0, %even1
%part1, %part3 = pto.vdintlv %odd0,  %odd1

replace pto.vmi.load with [%part0, %part1, %part2, %part3]
```

This optimization is legal only for full physical chunks and supported
`DINTLV_B8/B16/B32` element widths.  Tail and masked loads keep their explicit
safe lowering until a masked or guarded `vldsx2` strategy is designed.

Two-way logical interleaved memory access is represented by dedicated VMI ops,
not by exposing assigned layouts in surface IR:

```mlir
%x, %y = pto.vmi.deinterleave_load %src[%off]
    : !pto.ptr<f32, ub> -> !pto.vmi.vreg<64xf32>, !pto.vmi.vreg<64xf32>

pto.vmi.interleave_store %x, %y, %dst[%off]
    : !pto.vmi.vreg<64xf32>, !pto.vmi.vreg<64xf32>, !pto.ptr<f32, ub>
```

Each VMI value is an ordinary dense logical vector.  Layout assignment requests
contiguous layouts for both streams.  Lowering maps full-chunk 8/16/32-bit cases
to `vldsx2 DINTLV_B*` and `vstsx2 INTLV_B*`.

## 3. Lowering Results

The following examples use symbolic VPTO names. `PAT_ALL_B*` means an all-true
predicate with the element granularity required by the instruction. `PAT_VLk`
means a prefix predicate for the first `k` lanes.

Completeness rule for this section: every numbered endpoint below must contain
VMI input, assigned layouts, VPTO lowering result, and either a memory result or
an explicit diagnostic.  Non-endpoint layout notes may appear only as setup for
the immediately following complete endpoints.

```text
3.1 f16 -> f32 -> store                                  complete
3.2 f32 -> f16 -> store                                  complete
3.3 f8 -> f32 -> compute -> f8                           complete
3.4 group_reduce S=8 -> group_store                      complete
3.5.1 group_reduce S=16 -> group_store                   complete
3.5.2 group_reduce S=16 -> broadcast -> compute -> reduce -> store
                                                            complete
3.5.3 group_reduce S=16 -> elemwise(rhs) -> group_store  complete
3.6.1 group_reduce S=32 -> group_store                   complete
3.6.2 group_reduce S=32 -> elemwise(rhs) -> group_store  complete
3.6.3 group_reduce S=32 -> broadcast -> compute -> reduce -> store
                                                            complete
3.7.1 group_reduce S=64 -> aligned group_store           complete
3.7.2 group_reduce S=64 -> elemwise(rhs) -> aligned group_store
                                                            complete
3.7.3 group_reduce S=64 -> broadcast -> compute -> reduce -> store
                                                            complete
3.7.4 group_reduce S=64 -> unit-stride group_store       illegal diagnostic
3.8 group_reduce -> truncf -> broadcast -> dense store   complete
3.9 dense store of group slots                           illegal diagnostic
3.10 non-load producer feeding S=32 group_reduce         complete
3.11 partial tail groups                                 complete/diagnostic
3.12 control-flow join before group_reduce               complete
3.13 packed group-slot f32 -> f16 cast                   illegal diagnostic
3.14 unsupported group size                              illegal diagnostic
3.15 compact S=12 written as logical S=16                complete/diagnostic
3.16 group_slot_load layout contract                     complete
3.17 group_broadcast feeding deinterleaved consumer      complete
3.18 one value with dense and group-reduce consumers     complete/materialization
3.19 S=16 split-layout support selection                 complete/diagnostic
3.20 group_slots control-flow join                       complete
3.21 S=32 tail with full-tile-readable source            complete
3.22 scf.for loop-carried layout                         complete
3.23 group_broadcast with multiple dense consumers       complete
3.24 mask with elementwise/select/store                  complete
3.25 function boundary layout specialization             complete
3.26 S=16 grouped tail through broadcast/reduce/store    complete
3.27 S=32 group_load with stride greater than group size complete
3.28 group_slot_load slots=1 aligned non-unit stride     complete
3.29 one semantic mask with f32 and f16 consumers        complete
3.30 masked_load tail without padding                    complete/diagnostic
3.31 f16->f32 feeding dense store and S=16 reduce        complete
3.32 f32 feeding f8 store and S=32 reduce                complete
3.33 one dense value feeding S=16 and S=32 reduces       complete/materialization
3.34 S=64 group-slot result f32->f16 cast                complete
3.35 group_slots fanout to group_store and broadcast     complete
3.36 same scalar source materialized as slots=8/slots=1  complete/materialization
3.37 S=64 group_store with non-unit output stride        complete
3.38 multi-tile S=32 group_reduce                        complete
3.39 strided S=32 group_load through broadcast/reduce    complete
3.40 scalar broadcast feeding dense and grouped users    complete/materialization
3.41 non-rematerializable value with incompatible users  complete/materialization
3.42 group_slots scf.for loop-carried accumulator        complete
3.43 internal function argument boundary materialization complete
3.44 masked_load grouped tail feeding S=32 reduce        complete
3.45 dynamic S=32 create_group_mask                      complete
3.46 extf value and derived elemwise value both stored   complete/optimization
3.47-3.55 typed group-reduce generalization              complete/diagnostic
3.56 full 256-bin distribution histogram                 complete
3.57 full 256-bin cumulative histogram                   design boundary
```

### 3.1 `f16 -> f32 -> store`

VMI input:

```text
%x16 = pto.vmi.load %base[%off]
  : memref<128xf16> -> !pto.vmi.vreg<128xf16>
%x32 = pto.vmi.extf %x16
  : !pto.vmi.vreg<128xf16> -> !pto.vmi.vreg<128xf32>
pto.vmi.store %x32, %out[%off]
```

Assigned layouts:

```text
%x16 : !pto.vmi.vreg<128xf16, #pto.vmi.layout<contiguous>>
%x32 : !pto.vmi.vreg<128xf32, #pto.vmi.layout<deinterleaved = 2>>
```

VPTO lowering result:

```text
%x16_0 = pto.vlds %base[%off] {dist = "NORM"}
  : !pto.ptr<f16, ub> -> !pto.vreg<128xf16>

%x32_p0 = pto.vcvt %x16_0, PAT_ALL_B16 {part = "EVEN"}
  : !pto.vreg<128xf16>, !pto.mask<b16> -> !pto.vreg<64xf32>
%x32_p1 = pto.vcvt %x16_0, PAT_ALL_B16 {part = "ODD"}
  : !pto.vreg<128xf16>, !pto.mask<b16> -> !pto.vreg<64xf32>

pto.vstsx2 %x32_p0, %x32_p1, %out[%off], "INTLV_B32", PAT_ALL_B32
  : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.ptr<f32, ub>, index,
    !pto.mask<b32>
```

Alternative complete VPTO lowering result if `vstsx2 INTLV_B32` is unavailable:

```text
%x16_0 = pto.vlds %base[%off] {dist = "NORM"}
  : !pto.ptr<f16, ub> -> !pto.vreg<128xf16>

%x32_p0 = pto.vcvt %x16_0, PAT_ALL_B16 {part = "EVEN"}
  : !pto.vreg<128xf16>, !pto.mask<b16> -> !pto.vreg<64xf32>
%x32_p1 = pto.vcvt %x16_0, PAT_ALL_B16 {part = "ODD"}
  : !pto.vreg<128xf16>, !pto.mask<b16> -> !pto.vreg<64xf32>

%x32_d0, %x32_d1 = pto.vintlv %x32_p0, %x32_p1
  : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>, !pto.vreg<64xf32>

pto.vsts %x32_d0, %out[%off], PAT_ALL_B32 {dist = "NORM_B32"}
  : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
pto.vsts %x32_d1, %out[%off_plus_64], PAT_ALL_B32 {dist = "NORM_B32"}
  : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
```

Memory result:

```text
for i = 0..127:
  out[off + i] = extf(base[off + i])
```

### 3.2 Dense `f32 -> f16 -> store`

VMI input:

```text
%x32 = pto.vmi.load %base[%off]
  : memref<128xf32> -> !pto.vmi.vreg<128xf32>
%x16 = pto.vmi.truncf %x32
  : !pto.vmi.vreg<128xf32> -> !pto.vmi.vreg<128xf16>
pto.vmi.store %x16, %out[%off]
```

Assigned layouts:

```text
%x32 : !pto.vmi.vreg<128xf32, #pto.vmi.layout<deinterleaved = 2>>
%x16 : !pto.vmi.vreg<128xf16, #pto.vmi.layout<contiguous>>
```

VPTO lowering result:

```text
%x32_p0, %x32_p1 = pto.vldsx2 %base[%off], "DINTLV_B32"
  : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>, !pto.vreg<64xf32>

%part0 = pto.vcvt %x32_p0, PAT_ALL_B32
  {part = "EVEN", rnd = "R", sat = "SAT"}
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<128xf16>

%part1 = pto.vcvt %x32_p1, PAT_ALL_B32
  {part = "ODD", rnd = "R", sat = "SAT"}
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<128xf16>

%x16_0 = pto.vor %part0, %part1, PAT_ALL_B16
  : !pto.vreg<128xf16>

pto.vsts %x16_0, %out[%off], PAT_ALL_B16 {dist = "NORM_B16"}
  : !pto.vreg<128xf16>, !pto.ptr<f16, ub>, !pto.mask<b16>
```

Alternative complete VPTO lowering result if the source has already been loaded
as two contiguous f32 chunks and must be materialized to `deinterleaved=2` before
the conversion:

```text
%x32_d0 = pto.vlds %base[%off] {dist = "NORM"}
  : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
%x32_d1 = pto.vlds %base[%off_plus_64] {dist = "NORM"}
  : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
%x32_p0, %x32_p1 = pto.vdintlv %x32_d0, %x32_d1
  : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>, !pto.vreg<64xf32>

%part0 = pto.vcvt %x32_p0, PAT_ALL_B32
  {part = "EVEN", rnd = "R", sat = "SAT"}
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<128xf16>

%part1 = pto.vcvt %x32_p1, PAT_ALL_B32
  {part = "ODD", rnd = "R", sat = "SAT"}
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<128xf16>

%x16_0 = pto.vor %part0, %part1, PAT_ALL_B16
  : !pto.vreg<128xf16>

pto.vsts %x16_0, %out[%off], PAT_ALL_B16 {dist = "NORM_B16"}
  : !pto.vreg<128xf16>, !pto.ptr<f16, ub>, !pto.mask<b16>
```

Memory result:

```text
for i = 0..127:
  out[off + i] = truncf(base[off + i])
```

### 3.3 Dense `f8 -> f32 -> compute -> f8`

VMI input:

```text
%x8  = pto.vmi.load %base[%off]
%x32 = pto.vmi.extf %x8
%scale = pto.vmi.broadcast %scale_s : f32 -> !pto.vmi.vreg<256xf32>
%y32 = pto.vmi.vmul %x32, %scale
%y8  = pto.vmi.truncf %y32
pto.vmi.store %y8, %out[%off]
```

Assigned layouts:

```text
%x8  : !pto.vmi.vreg<256xf8,  #pto.vmi.layout<contiguous>>
%x32 : !pto.vmi.vreg<256xf32, #pto.vmi.layout<deinterleaved = 4>>
%scale : !pto.vmi.vreg<256xf32, #pto.vmi.layout<deinterleaved = 4>>
%y32 : !pto.vmi.vreg<256xf32, #pto.vmi.layout<deinterleaved = 4>>
%y8  : !pto.vmi.vreg<256xf8,  #pto.vmi.layout<contiguous>>
```

VPTO lowering result:

```text
%x8_0 = pto.vlds %base[%off] {dist = "NORM"}
  : !pto.ptr<f8, ub> -> !pto.vreg<256xf8>

%x32_p0 = pto.vcvt %x8_0, PAT_ALL_B8 {part = "P0"}
  : !pto.vreg<256xf8>, !pto.mask<b8> -> !pto.vreg<64xf32>
%x32_p1 = pto.vcvt %x8_0, PAT_ALL_B8 {part = "P1"}
  : !pto.vreg<256xf8>, !pto.mask<b8> -> !pto.vreg<64xf32>
%x32_p2 = pto.vcvt %x8_0, PAT_ALL_B8 {part = "P2"}
  : !pto.vreg<256xf8>, !pto.mask<b8> -> !pto.vreg<64xf32>
%x32_p3 = pto.vcvt %x8_0, PAT_ALL_B8 {part = "P3"}
  : !pto.vreg<256xf8>, !pto.mask<b8> -> !pto.vreg<64xf32>

%scale_p0 = pto.vdup %scale_s, PAT_ALL_B32
  : f32, !pto.mask<b32> -> !pto.vreg<64xf32>
%scale_p1 = pto.vdup %scale_s, PAT_ALL_B32
  : f32, !pto.mask<b32> -> !pto.vreg<64xf32>
%scale_p2 = pto.vdup %scale_s, PAT_ALL_B32
  : f32, !pto.mask<b32> -> !pto.vreg<64xf32>
%scale_p3 = pto.vdup %scale_s, PAT_ALL_B32
  : f32, !pto.mask<b32> -> !pto.vreg<64xf32>

%y32_p0 = pto.vmul %x32_p0, %scale_p0, PAT_ALL_B32
%y32_p1 = pto.vmul %x32_p1, %scale_p1, PAT_ALL_B32
%y32_p2 = pto.vmul %x32_p2, %scale_p2, PAT_ALL_B32
%y32_p3 = pto.vmul %x32_p3, %scale_p3, PAT_ALL_B32

%y8_p0 = pto.vcvt %y32_p0, PAT_ALL_B32
  {part = "P0", rnd = "R", sat = "SAT"} -> !pto.vreg<256xf8>
%y8_p1 = pto.vcvt %y32_p1, PAT_ALL_B32
  {part = "P1", rnd = "R", sat = "SAT"} -> !pto.vreg<256xf8>
%y8_p2 = pto.vcvt %y32_p2, PAT_ALL_B32
  {part = "P2", rnd = "R", sat = "SAT"} -> !pto.vreg<256xf8>
%y8_p3 = pto.vcvt %y32_p3, PAT_ALL_B32
  {part = "P3", rnd = "R", sat = "SAT"} -> !pto.vreg<256xf8>

%y8_01 = pto.vor %y8_p0, %y8_p1, PAT_ALL_B8
%y8_23 = pto.vor %y8_p2, %y8_p3, PAT_ALL_B8
%y8_0  = pto.vor %y8_01, %y8_23, PAT_ALL_B8

pto.vsts %y8_0, %out[%off], PAT_ALL_B8 {dist = "NORM_B8"}
  : !pto.vreg<256xf8>, !pto.ptr<f8, ub>, !pto.mask<b8>
```

Memory result:

```text
for i = 0..255:
  out[off + i] = truncf(extf(base[off + i]) * scale_s)
```

### 3.4 `group_reduce` S=8 f32

VMI input:

```text
%x = pto.vmi.load %base[%off]
  : memref<64xf32> -> !pto.vmi.vreg<64xf32>
%mask = pto.vmi.create_mask %c64 : index -> !pto.vmi.mask<64xpred>
%sum = pto.vmi.group_reduce_addf %x, %mask {num_groups = 8}
  : !pto.vmi.vreg<64xf32>, !pto.vmi.mask<64xpred>
 -> !pto.vmi.vreg<64xf32>
pto.vmi.group_store %sum, %sum_out[%group_off], %c1 {num_groups = 8}
```

Assigned layouts:

```text
%x    : !pto.vmi.vreg<64xf32, #pto.vmi.layout<contiguous>>
%mask : !pto.vmi.mask<64xpred, #pto.vmi.layout<contiguous>>
%sum  : !pto.vmi.vreg<64xf32,
          #pto.vmi.layout<num_groups = 8, slots = 8>>
```

VPTO lowering result for one full 8-row tile:

```text
%mask_chunk = pto.pge_b32 "PAT_ALL"

%x_chunk = pto.vlds %base[%tile_off] {dist = "NORM"}
  : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>

%sum_block = pto.vcgadd %x_chunk, %mask_chunk
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>

%store8 = pto.pge_b32 "PAT_VL8"
pto.vsts %sum_block, %sum_out[%group_tile_off], %store8 {dist = "NORM_B32"}
  : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
```

Lowering result for one chunk, per the `visa.txt` VCGADD contract:

```text
%sum_block lane 0 = reduce %x lanes 0..7
%sum_block lane 1 = reduce %x lanes 8..15
...
%sum_block lane 7 = reduce %x lanes 56..63
all non-slot lanes are non-semantic
```

Layout result:

```text
G = N / 8
K = 8

slot_block(g) = g / 8
slot_lane(g)  = g % 8
```

Memory result:

```text
for r = 0..7:
  sum_out[group_tile_off + r] = reduce(row_r[0..7])
```

### 3.5 `group_reduce` S=16 f32, load-fused split

The facts used by this lowering are checked against the current repo:

```text
pto.vldsx2 supports "BDINTLV".
pto.vstsx2 supports only "INTLV_B8" / "INTLV_B16" / "INTLV_B32".
visa.txt says VCGADD writes one 32B-block result continuously to destination
LSBs; the current repository golden tests follow lanes 0..7 for f32.
```

There are three complete consumers for this layout today:

```text
load -> group_reduce -> group_store(sum)
load -> group_reduce -> elementwise compute on group-slot values
     -> group_store
load -> group_reduce -> group_broadcast -> elementwise compute
     -> group_reduce -> group_store
```

#### 3.5.1 Reduce And Store Group Sums

VMI input:

```text
%x = pto.vmi.load %base[%off]
  : memref<Nxf32> -> !pto.vmi.vreg<Nxf32>
%sum = pto.vmi.group_reduce_addf %x, %mask {num_groups = N / 16}
pto.vmi.group_store %sum, %sum_out[%group_off], %c1 {num_groups = N / 16}
```

Assigned layouts:

```text
%x : !pto.vmi.vreg<Nxf32,
       #pto.vmi.layout<block_deinterleaved=2>>

%sum : !pto.vmi.vreg<Nxf32,
         #pto.vmi.layout<num_groups = N / 16, slots = 8>>
```

For each 8-row tile:

```text
row r = 16xf32 = row_r.lo8, row_r.hi8
```

VPTO lowering result for one full 8-row tile:

```text
%all_b32 = pto.pge_b32 "PAT_ALL"
%sum_mask = pto.pge_b32 "PAT_VL8"

%lo, %hi = pto.vldsx2 %base[%tile_off], "BDINTLV"
  : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>, !pto.vreg<64xf32>

%lo lanes 0..7   = row0.lo8
%lo lanes 8..15  = row1.lo8
...
%lo lanes 56..63 = row7.lo8

%hi lanes 0..7   = row0.hi8
%hi lanes 8..15  = row1.hi8
...
%hi lanes 56..63 = row7.hi8

%lo_sum = pto.vcgadd %lo, %all_b32
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%hi_sum = pto.vcgadd %hi, %all_b32
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>

%sum_block = pto.vadd %lo_sum, %hi_sum, %sum_mask
  : !pto.vreg<64xf32>

%store8 = pto.pge_b32 "PAT_VL8"
pto.vsts %sum_block, %sum_out[%group_tile_off], %store8 {dist = "NORM_B32"}
  : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
```

`BDINTLV` here denotes the ISA `#bdintlv` block-based interleaving load mode:
it loads `2 * VL` bytes and sends even 32B blocks to the first destination
register and odd 32B blocks to the second destination register. For f32,
one 32B block is `8xf32`; f16 and f8 carriers instead derive 16 and 32
elements per block.

Tail tiles use the same dataflow with `%all_b32` replaced by masks derived from
the VMI mask for the low and high 8-lane halves of each row.

Layout result:

```text
G = N / 16
K = 8

slot_block(g) = g / 8
slot_lane(g)  = g % 8

%sum_block lane 0 = reduce row0 lanes 0..15
%sum_block lane 1 = reduce row1 lanes 0..15
...
%sum_block lane 7 = reduce row7 lanes 0..15
```

No VMI value exposes `%lo_sum` or `%hi_sum`. They are internal VPTO values.

Memory result:

```text
sum_out[group_tile_off + 0] = reduce row0 lanes 0..15
sum_out[group_tile_off + 1] = reduce row1 lanes 0..15
...
sum_out[group_tile_off + 7] = reduce row7 lanes 0..15
```

This endpoint is fully specified: the only group-slot value is `%sum`; `group_store`
stores the low 8 slot lanes with an ordinary prefix store.

#### 3.5.2 Reduce, Broadcast, Elementwise, Reduce, Store

VMI input:

```text
%x = pto.vmi.load %base[%off]
  : memref<Nxf32> -> !pto.vmi.vreg<Nxf32>
%sum = pto.vmi.group_reduce_addf %x, %mask {num_groups = N / 16}
%b = pto.vmi.group_broadcast %sum {num_groups = N / 16}
%y = pto.vmi.vmul %x, %b
%ysum = pto.vmi.group_reduce_addf %y, %mask {num_groups = N / 16}
pto.vmi.group_store %ysum, %out[%group_off], %c1 {num_groups = N / 16}
```

Assigned layouts:

```text
%x   : !pto.vmi.vreg<Nxf32,
         #pto.vmi.layout<block_deinterleaved=2>>
%sum : !pto.vmi.vreg<Nxf32,
         #pto.vmi.layout<num_groups = N / 16, slots = 8>>
%b   : !pto.vmi.vreg<Nxf32,
         #pto.vmi.layout<block_deinterleaved=2>>
%y   : !pto.vmi.vreg<Nxf32,
         #pto.vmi.layout<block_deinterleaved=2>>
%ysum : !pto.vmi.vreg<Nxf32,
          #pto.vmi.layout<num_groups = N / 16, slots = 8>>
```

VPTO lowering result for one full 8-row tile:

```text
%all_b32 = pto.pge_b32 "PAT_ALL"
%sum_mask = pto.pge_b32 "PAT_VL8"

%x_lo, %x_hi = pto.vldsx2 %base[%tile_off], "BDINTLV"
  : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>, !pto.vreg<64xf32>

%x_lo_sum = pto.vcgadd %x_lo, %all_b32
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%x_hi_sum = pto.vcgadd %x_hi, %all_b32
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>

%sum_block = pto.vadd %x_lo_sum, %x_hi_sum, %sum_mask
  : !pto.vreg<64xf32>

%lane_id = pto.vci %c0_i32 : i32 -> !pto.vreg<64xi32>
%broadcast_idx = pto.vshrs %lane_id, %c3_i16, %all_b32
  : !pto.vreg<64xi32>, i16, !pto.mask<b32> -> !pto.vreg<64xi32>

// This is the materialization of pto.vmi.group_broadcast.  The group sums are
// in %sum_block lanes 0..7; vselr expands each sum to the 8 lanes of the
// corresponding row half.  The following vmul/vcgadd consume an ordinary dense
// physical vector.
%b_rows = pto.vselr %sum_block, %broadcast_idx
  : !pto.vreg<64xf32>, !pto.vreg<64xi32> -> !pto.vreg<64xf32>

%y_lo = pto.vmul %x_lo, %b_rows, %all_b32
  : !pto.vreg<64xf32>
%y_hi = pto.vmul %x_hi, %b_rows, %all_b32
  : !pto.vreg<64xf32>

%y_lo_sum = pto.vcgadd %y_lo, %all_b32
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%y_hi_sum = pto.vcgadd %y_hi, %all_b32
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>

// Final per-row reduction and store.
%ysum_block = pto.vadd %y_lo_sum, %y_hi_sum, %sum_mask
  : !pto.vreg<64xf32>

%store8 = pto.pge_b32 "PAT_VL8"
pto.vsts %ysum_block, %out[%group_tile_off], %store8 {dist = "NORM_B32"}
  : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
```

This trace processes 8 logical rows at once. `num_groups = N / 16` means each
logical group is one `16xf32` row, and one full f32 VPTO tile covers 8 such
groups:

```text
64 f32 lanes per physical part = 8 rows * 8 f32 lanes per half-row
```

Tail tiles use the same dataflow with `%all_b32` replaced by masks derived from
the VMI mask for the low and high 8-lane halves of each row.

Physical lane result for the tile:

```text
%x_lo lanes 0..7   = row0[0..7]
%x_lo lanes 8..15  = row1[0..7]
...
%x_lo lanes 56..63 = row7[0..7]

%x_hi lanes 0..7   = row0[8..15]
%x_hi lanes 8..15  = row1[8..15]
...
%x_hi lanes 56..63 = row7[8..15]

%sum_block lanes 0..7 =
  reduce(row0[0..15]), reduce(row1[0..15]), ..., reduce(row7[0..15])

%b_rows lanes 0..7   = reduce(row0[0..15])
%b_rows lanes 8..15  = reduce(row1[0..15])
...
%b_rows lanes 56..63 = reduce(row7[0..15])

For each row `r` in this 8-row tile:

%y_lo lanes r*8 .. r*8+7 =
  row_r[0..7] * reduce(row_r[0..15])

%y_hi lanes r*8 .. r*8+7 =
  row_r[8..15] * reduce(row_r[0..15])

Concretely:
%y_lo lanes 0..7   = row0[0..7] * reduce(row0[0..15])
%y_lo lanes 8..15  = row1[0..7] * reduce(row1[0..15])
...
%y_lo lanes 56..63 = row7[0..7] * reduce(row7[0..15])

%y_hi lanes 0..7   = row0[8..15] * reduce(row0[0..15])
%y_hi lanes 8..15  = row1[8..15] * reduce(row1[0..15])
...
%y_hi lanes 56..63 = row7[8..15] * reduce(row7[0..15])

%ysum_block lanes 0..7 =
  reduce(%y row0), reduce(%y row1), ..., reduce(%y row7)
```

Memory result:

```text
out[group_tile_off + r] =
    reduce_i((row_r[i] * reduce_j(row_r[j])) for i in 0..15)
  = reduce(row_r[0..15]) * reduce(row_r[0..15])
for r = 0..7
```

If a later consumer requires row-major contiguous order, `vmi-to-vpto` must
materialize:

```text
block_deinterleaved=2 -> contiguous
```

This materialization cannot be implemented with `vstsx2 INTLV_B32`, because
that instruction interleaves individual b32 elements, not 32B row halves. Until
a concrete block-interleave register materialization or store op is selected,
row-major store of this layout must be rejected with:

```text
VMI-LAYOUT-CONTRACT:
  pto.vmi.store requires materializing
  #pto.vmi.layout<block_deinterleaved=2> to contiguous, but no
  VPTO block-interleave materialization/store support exists.
```

#### 3.5.3 Reduce Result, Elementwise, Store

This case computes a per-row reduction, applies an elementwise operation to the
reduced values themselves, and stores one result per group.  There is no
`group_broadcast` in this flow because the elementwise op is not applied to the
original `8x16xf32` matrix elements.

VMI input:

```text
%x = pto.vmi.load %base[%off]
  : memref<128xf32> -> !pto.vmi.vreg<128xf32>
%rhs = pto.vmi.group_slot_load %rhs_base[%rhs_off], %c1 {num_groups = 8}
  : !pto.ptr<f32, ub> -> !pto.vmi.vreg<128xf32>
%sum = pto.vmi.group_reduce_addf %x, %mask {num_groups = 8}
%outv = pto.vmi.vadd %sum, %rhs
pto.vmi.group_store %outv, %out[%group_off], %c1 {num_groups = 8}
```

Assigned layouts:

```text
%x for reduce:
  !pto.vmi.vreg<128xf32,
    #pto.vmi.layout<block_deinterleaved=2>>

%sum:
  !pto.vmi.vreg<128xf32, #pto.vmi.layout<num_groups = 8, slots = 8>>

%rhs:
  !pto.vmi.vreg<128xf32, #pto.vmi.layout<num_groups = 8, slots = 8>>

%outv:
  !pto.vmi.vreg<128xf32, #pto.vmi.layout<num_groups = 8, slots = 8>>
```

For this endpoint, the RHS is a packed per-group vector:

```text
rhs_base[rhs_off + r] = rhs(row r), for r = 0..7
```

Layout assignment must treat `group_slot_load` as a group-slot producer: one
f32 value per group is placed in the live slot lanes.  It must not use
`group_load`, which loads `group_size` data elements per group instead of one
per-group scalar.

The elementwise op runs only on the live group-slot lanes:

```text
%sum lanes 0..7 =
  reduce(row0[0..15]), reduce(row1[0..15]), ..., reduce(row7[0..15])

%rhs lanes 0..7 =
  rhs(row0), rhs(row1), ..., rhs(row7)

%outv lanes 0..7 =
  %sum lanes 0..7 + %rhs lanes 0..7

lanes 8..63 remain dead/zero and are masked off by PAT_VL8.
```

VPTO lowering result for one full 8-row tile:

```text
%all_b32 = pto.pge_b32 "PAT_ALL"
%sum_mask = pto.pge_b32 "PAT_VL8"
%one_block = pto.pge_b32 "PAT_VL1"

// Reduction path: use BDINTLV to feed two VCG reductions.
%x_lo, %x_hi = pto.vldsx2 %base[%tile_off], "BDINTLV"
  : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>, !pto.vreg<64xf32>

%x_lo_sum = pto.vcgadd %x_lo, %all_b32
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%x_hi_sum = pto.vcgadd %x_hi, %all_b32
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>

%sum_block = pto.vadd %x_lo_sum, %x_hi_sum, %sum_mask
  : !pto.vreg<64xf32>

// Packed RHS group-slot load.  %rhs_tile_base points to rhs_base[rhs_off].
// One 32B block contains 8 f32 RHS values and materializes lanes 0..7; all
// other lanes are dead/zero.
%rhs_block = pto.vsldb %rhs_tile_base, %c0_i16, %c0_i16, %one_block
  : !pto.ptr<f32, ub>, i16, i16, !pto.mask<b32> -> !pto.vreg<64xf32>

// Elementwise compute on group-slot values.  Only lanes 0..7 are live.
%outv_block = pto.vadd %sum_block, %rhs_block, %sum_mask
  : !pto.vreg<64xf32>

pto.vsts %outv_block, %out[%group_tile_off], %sum_mask {dist = "NORM_B32"}
  : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
```

Memory result:

```text
for r = 0..7:
  s = reduce(row_r[0..15])
  out[group_tile_off + r] = s + rhs[r]
```

### 3.6 `group_reduce` S=32 f32, 4-way split

This case covers one `8x32xf32` tile.  Each logical row is 128B, so it must be
split into four 32B partial rows before `vcgadd` can reduce it efficiently.

The canonical layout for the input is:

```text
%x : !pto.vmi.vreg<256xf32, #pto.vmi.layout<deinterleaved = 4>>
```

With `deinterleaved = 4`, physical part `p` contains columns whose logical
column index is `p mod 4`:

```text
%x_p0 lanes r*8 .. r*8+7 =
  row_r[0], row_r[4], row_r[8],  ..., row_r[28]

%x_p1 lanes r*8 .. r*8+7 =
  row_r[1], row_r[5], row_r[9],  ..., row_r[29]

%x_p2 lanes r*8 .. r*8+7 =
  row_r[2], row_r[6], row_r[10], ..., row_r[30]

%x_p3 lanes r*8 .. r*8+7 =
  row_r[3], row_r[7], row_r[11], ..., row_r[31]
```

Each physical part now has exactly 8 f32 values per row, so one `vcgadd` per
part computes one partial sum per row.  The four partial sums are then added
under `PAT_VL8`.

The full contiguous-to-4-way materialization for one tile should fuse the first
deinterleave level into the load.  `vldsx2 DINTLV_B32` loads `2 * VL` bytes and
splits even/odd f32 elements into two physical vectors.  Two such loads cover
the `8x32xf32` tile, and a second register `vdintlv` level splits even columns
into `mod4 = 0/2` and odd columns into `mod4 = 1/3`.

This setup documentation is repeated inside every complete 32-wide endpoint
below.

```text
%x_even_0, %x_odd_0 = pto.vldsx2 %base[%tile_off_0], "DINTLV_B32"
  : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>, !pto.vreg<64xf32>
%x_even_1, %x_odd_1 = pto.vldsx2 %base[%tile_off_1], "DINTLV_B32"
  : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>, !pto.vreg<64xf32>

%x_p0, %x_p2 = pto.vdintlv %x_even_0, %x_even_1
  : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>, !pto.vreg<64xf32>
%x_p1, %x_p3 = pto.vdintlv %x_odd_0, %x_odd_1
  : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>, !pto.vreg<64xf32>
```

Each endpoint below inlines this materialization before the first consumer of
`%x_p0..%x_p3`.

#### 3.6.1 Reduce And Store Group Sums

VMI input:

```text
%x = pto.vmi.load %base[%off]
  : memref<256xf32> -> !pto.vmi.vreg<256xf32>
%sum = pto.vmi.group_reduce_addf %x, %mask {num_groups = 8}
pto.vmi.group_store %sum, %sum_out[%group_off], %c1 {num_groups = 8}
```

Assigned layouts:

```text
%x:
  !pto.vmi.vreg<256xf32, #pto.vmi.layout<deinterleaved = 4>>

%sum:
  !pto.vmi.vreg<256xf32, #pto.vmi.layout<num_groups = 8, slots = 8>>
```

VPTO lowering result for one full 8-row tile:

```text
%x_even_0, %x_odd_0 = pto.vldsx2 %base[%tile_off_0], "DINTLV_B32"
  : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>, !pto.vreg<64xf32>
%x_even_1, %x_odd_1 = pto.vldsx2 %base[%tile_off_1], "DINTLV_B32"
  : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>, !pto.vreg<64xf32>

%x_p0, %x_p2 = pto.vdintlv %x_even_0, %x_even_1
  : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>, !pto.vreg<64xf32>
%x_p1, %x_p3 = pto.vdintlv %x_odd_0, %x_odd_1
  : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>, !pto.vreg<64xf32>

%all_b32 = pto.pge_b32 "PAT_ALL"
%sum_mask = pto.pge_b32 "PAT_VL8"

%s0 = pto.vcgadd %x_p0, %all_b32
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%s1 = pto.vcgadd %x_p1, %all_b32
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%s2 = pto.vcgadd %x_p2, %all_b32
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%s3 = pto.vcgadd %x_p3, %all_b32
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>

%s01 = pto.vadd %s0, %s1, %sum_mask : !pto.vreg<64xf32>
%s23 = pto.vadd %s2, %s3, %sum_mask : !pto.vreg<64xf32>
%sum_block = pto.vadd %s01, %s23, %sum_mask : !pto.vreg<64xf32>

pto.vsts %sum_block, %sum_out[%group_tile_off], %sum_mask {dist = "NORM_B32"}
  : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
```

Memory result:

```text
for r = 0..7:
  sum_out[group_tile_off + r] = reduce(row_r[0..31])
```

#### 3.6.2 Reduce Result, Elementwise, Store

VMI input:

```text
%x = pto.vmi.load %base[%off]
  : memref<256xf32> -> !pto.vmi.vreg<256xf32>
%rhs = pto.vmi.group_slot_load %rhs_base[%rhs_off], %c1 {num_groups = 8}
  : !pto.ptr<f32, ub> -> !pto.vmi.vreg<256xf32>
%sum = pto.vmi.group_reduce_addf %x, %mask {num_groups = 8}
%outv = pto.vmi.vadd %sum, %rhs
pto.vmi.group_store %outv, %out[%group_off], %c1 {num_groups = 8}
```

Assigned layouts:

```text
%x:
  !pto.vmi.vreg<256xf32, #pto.vmi.layout<deinterleaved = 4>>

%sum, %rhs, %outv:
  !pto.vmi.vreg<256xf32, #pto.vmi.layout<num_groups = 8, slots = 8>>
```

VPTO lowering result for one full 8-row tile:

```text
%x_even_0, %x_odd_0 = pto.vldsx2 %base[%tile_off_0], "DINTLV_B32"
  : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>, !pto.vreg<64xf32>
%x_even_1, %x_odd_1 = pto.vldsx2 %base[%tile_off_1], "DINTLV_B32"
  : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>, !pto.vreg<64xf32>

%x_p0, %x_p2 = pto.vdintlv %x_even_0, %x_even_1
  : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>, !pto.vreg<64xf32>
%x_p1, %x_p3 = pto.vdintlv %x_odd_0, %x_odd_1
  : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>, !pto.vreg<64xf32>

%all_b32 = pto.pge_b32 "PAT_ALL"
%sum_mask = pto.pge_b32 "PAT_VL8"
%one_block = pto.pge_b32 "PAT_VL1"

%s0 = pto.vcgadd %x_p0, %all_b32
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%s1 = pto.vcgadd %x_p1, %all_b32
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%s2 = pto.vcgadd %x_p2, %all_b32
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%s3 = pto.vcgadd %x_p3, %all_b32
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>

%s01 = pto.vadd %s0, %s1, %sum_mask : !pto.vreg<64xf32>
%s23 = pto.vadd %s2, %s3, %sum_mask : !pto.vreg<64xf32>
%sum_block = pto.vadd %s01, %s23, %sum_mask : !pto.vreg<64xf32>

// Packed RHS group-slot load.  %rhs_tile_base points to rhs_base[rhs_off].
%rhs_block = pto.vsldb %rhs_tile_base, %c0_i16, %c0_i16, %one_block
  : !pto.ptr<f32, ub>, i16, i16, !pto.mask<b32> -> !pto.vreg<64xf32>

%outv_block = pto.vadd %sum_block, %rhs_block, %sum_mask
  : !pto.vreg<64xf32>

pto.vsts %outv_block, %out[%group_tile_off], %sum_mask {dist = "NORM_B32"}
  : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
```

Memory result:

```text
for r = 0..7:
  out[group_tile_off + r] = reduce(row_r[0..31]) + rhs[r]
```

#### 3.6.3 Reduce, Broadcast, Elementwise, Reduce, Store

VMI input:

```text
%x = pto.vmi.load %base[%off]
  : memref<256xf32> -> !pto.vmi.vreg<256xf32>
%sum = pto.vmi.group_reduce_addf %x, %mask {num_groups = 8}
%b = pto.vmi.group_broadcast %sum {num_groups = 8}
%y = pto.vmi.vmul %x, %b
%ysum = pto.vmi.group_reduce_addf %y, %mask {num_groups = 8}
pto.vmi.group_store %ysum, %out[%group_off], %c1 {num_groups = 8}
```

Assigned layouts:

```text
%x, %b, %y:
  !pto.vmi.vreg<256xf32, #pto.vmi.layout<deinterleaved = 4>>

%sum, %ysum:
  !pto.vmi.vreg<256xf32, #pto.vmi.layout<num_groups = 8, slots = 8>>
```

VPTO lowering result for one full 8-row tile:

```text
%x_even_0, %x_odd_0 = pto.vldsx2 %base[%tile_off_0], "DINTLV_B32"
  : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>, !pto.vreg<64xf32>
%x_even_1, %x_odd_1 = pto.vldsx2 %base[%tile_off_1], "DINTLV_B32"
  : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>, !pto.vreg<64xf32>

%x_p0, %x_p2 = pto.vdintlv %x_even_0, %x_even_1
  : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>, !pto.vreg<64xf32>
%x_p1, %x_p3 = pto.vdintlv %x_odd_0, %x_odd_1
  : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>, !pto.vreg<64xf32>

%all_b32 = pto.pge_b32 "PAT_ALL"
%sum_mask = pto.pge_b32 "PAT_VL8"

%s0 = pto.vcgadd %x_p0, %all_b32
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%s1 = pto.vcgadd %x_p1, %all_b32
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%s2 = pto.vcgadd %x_p2, %all_b32
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%s3 = pto.vcgadd %x_p3, %all_b32
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>

%s01 = pto.vadd %s0, %s1, %sum_mask : !pto.vreg<64xf32>
%s23 = pto.vadd %s2, %s3, %sum_mask : !pto.vreg<64xf32>
%sum_block = pto.vadd %s01, %s23, %sum_mask : !pto.vreg<64xf32>

%lane_id = pto.vci %c0_i32 : i32 -> !pto.vreg<64xi32>
%broadcast_idx = pto.vshrs %lane_id, %c3_i16, %all_b32
  : !pto.vreg<64xi32>, i16, !pto.mask<b32> -> !pto.vreg<64xi32>

// group_broadcast materialized for each deinterleaved=4 physical part.
%b_p0 = pto.vselr %sum_block, %broadcast_idx
  : !pto.vreg<64xf32>, !pto.vreg<64xi32> -> !pto.vreg<64xf32>
%b_p1 = pto.vselr %sum_block, %broadcast_idx
  : !pto.vreg<64xf32>, !pto.vreg<64xi32> -> !pto.vreg<64xf32>
%b_p2 = pto.vselr %sum_block, %broadcast_idx
  : !pto.vreg<64xf32>, !pto.vreg<64xi32> -> !pto.vreg<64xf32>
%b_p3 = pto.vselr %sum_block, %broadcast_idx
  : !pto.vreg<64xf32>, !pto.vreg<64xi32> -> !pto.vreg<64xf32>

%y_p0 = pto.vmul %x_p0, %b_p0, %all_b32 : !pto.vreg<64xf32>
%y_p1 = pto.vmul %x_p1, %b_p1, %all_b32 : !pto.vreg<64xf32>
%y_p2 = pto.vmul %x_p2, %b_p2, %all_b32 : !pto.vreg<64xf32>
%y_p3 = pto.vmul %x_p3, %b_p3, %all_b32 : !pto.vreg<64xf32>

%ys0 = pto.vcgadd %y_p0, %all_b32
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%ys1 = pto.vcgadd %y_p1, %all_b32
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%ys2 = pto.vcgadd %y_p2, %all_b32
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%ys3 = pto.vcgadd %y_p3, %all_b32
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>

%ys01 = pto.vadd %ys0, %ys1, %sum_mask : !pto.vreg<64xf32>
%ys23 = pto.vadd %ys2, %ys3, %sum_mask : !pto.vreg<64xf32>
%ysum_block = pto.vadd %ys01, %ys23, %sum_mask : !pto.vreg<64xf32>

pto.vsts %ysum_block, %out[%group_tile_off], %sum_mask {dist = "NORM_B32"}
  : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
```

Memory result:

```text
for r = 0..7:
  s = reduce(row_r[0..31])
  out[group_tile_off + r] =
      reduce_i(row_r[i] * s for i = 0..31)
    = s * s
```

### 3.7 `group_reduce` S=64 f32, row-local reduction

This case covers one `8x64xf32` tile. Each logical row is exactly 256B, so the
input does not need a deinterleaved layout:

```text
row r = 64xf32 = one !pto.vreg<64xf32>
```

The reduction is two-stage but row-local:

```text
vcgadd(row_r)       -> 8 partial sums in lanes 0..7
vcadd(PAT_VL8)     -> one row sum in lane 0
```

The result layout is therefore not `slots = 8`. It is:

```text
#pto.vmi.layout<num_groups = 8, slots = 1>
```

Physical slot mapping for this tile:

```text
slot_block(r) = r
slot_lane(r)  = 0

%sum0 lane 0 = reduce row0 lanes 0..63
%sum1 lane 0 = reduce row1 lanes 0..63
...
%sum7 lane 0 = reduce row7 lanes 0..63
```

Trying to canonicalize this result to `slots = 8` would require packing lane 0
from eight different physical vregs into lanes 0..7 of one vreg. This document
does not use that packing transform. `slots = 1` is the canonical layout for
S=64 row-local group reductions.

#### 3.7.1 Reduce And Store Group Sums

VMI input:

```text
%x = pto.vmi.load %base[%off]
  : memref<512xf32> -> !pto.vmi.vreg<512xf32>
%sum = pto.vmi.group_reduce_addf %x, %mask {num_groups = 8}
%c8 = arith.constant 8 : index
pto.vmi.group_store %sum, %sum_out[%group_off], %c8 {num_groups = 8}
```

Assigned layouts:

```text
%x:
  !pto.vmi.vreg<512xf32, #pto.vmi.layout<contiguous>>

%sum:
  !pto.vmi.vreg<512xf32, #pto.vmi.layout<num_groups = 8, slots = 1>>
```

VPTO lowering result for one full 8-row tile:

```text
%all_b32 = pto.pge_b32 "PAT_ALL"
%block8 = pto.pge_b32 "PAT_VL8"
%one_b32 = pto.pge_b32 "PAT_VL1"

%x0 = pto.vlds %base[%row_off_0] {dist = "NORM"}
  : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
%x1 = pto.vlds %base[%row_off_1] {dist = "NORM"}
  : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
%x2 = pto.vlds %base[%row_off_2] {dist = "NORM"}
  : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
%x3 = pto.vlds %base[%row_off_3] {dist = "NORM"}
  : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
%x4 = pto.vlds %base[%row_off_4] {dist = "NORM"}
  : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
%x5 = pto.vlds %base[%row_off_5] {dist = "NORM"}
  : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
%x6 = pto.vlds %base[%row_off_6] {dist = "NORM"}
  : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
%x7 = pto.vlds %base[%row_off_7] {dist = "NORM"}
  : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>

%p0 = pto.vcgadd %x0, %all_b32 : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%p1 = pto.vcgadd %x1, %all_b32 : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%p2 = pto.vcgadd %x2, %all_b32 : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%p3 = pto.vcgadd %x3, %all_b32 : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%p4 = pto.vcgadd %x4, %all_b32 : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%p5 = pto.vcgadd %x5, %all_b32 : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%p6 = pto.vcgadd %x6, %all_b32 : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%p7 = pto.vcgadd %x7, %all_b32 : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>

%sum0 = pto.vcadd %p0, %block8 : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%sum1 = pto.vcadd %p1, %block8 : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%sum2 = pto.vcadd %p2, %block8 : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%sum3 = pto.vcadd %p3, %block8 : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%sum4 = pto.vcadd %p4, %block8 : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%sum5 = pto.vcadd %p5, %block8 : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%sum6 = pto.vcadd %p6, %block8 : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%sum7 = pto.vcadd %p7, %block8 : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>

pto.vsts %sum0, %sum_out[%group_tile_off_0], %one_b32 {dist = "NORM_B32"}
  : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
pto.vsts %sum1, %sum_out[%group_tile_off_1], %one_b32 {dist = "NORM_B32"}
  : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
pto.vsts %sum2, %sum_out[%group_tile_off_2], %one_b32 {dist = "NORM_B32"}
  : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
pto.vsts %sum3, %sum_out[%group_tile_off_3], %one_b32 {dist = "NORM_B32"}
  : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
pto.vsts %sum4, %sum_out[%group_tile_off_4], %one_b32 {dist = "NORM_B32"}
  : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
pto.vsts %sum5, %sum_out[%group_tile_off_5], %one_b32 {dist = "NORM_B32"}
  : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
pto.vsts %sum6, %sum_out[%group_tile_off_6], %one_b32 {dist = "NORM_B32"}
  : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
pto.vsts %sum7, %sum_out[%group_tile_off_7], %one_b32 {dist = "NORM_B32"}
  : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
```

Memory result:

```text
for r = 0..7:
  sum_out[group_tile_off + r * 8] = reduce(row_r[0..63])
```

#### 3.7.2 Reduce Result, Elementwise, Store

VMI input:

```text
%x = pto.vmi.load %base[%off]
  : memref<512xf32> -> !pto.vmi.vreg<512xf32>
%rhs = pto.vmi.group_slot_load %rhs_base[%rhs_off], %c1 {num_groups = 8}
  : !pto.ptr<f32, ub> -> !pto.vmi.vreg<512xf32>
%sum = pto.vmi.group_reduce_addf %x, %mask {num_groups = 8}
%outv = pto.vmi.vadd %sum, %rhs
%c8 = arith.constant 8 : index
pto.vmi.group_store %outv, %out[%group_off], %c8 {num_groups = 8}
```

Assigned layouts:

```text
%x:
  !pto.vmi.vreg<512xf32, #pto.vmi.layout<contiguous>>

%sum, %rhs, %outv:
  !pto.vmi.vreg<512xf32, #pto.vmi.layout<num_groups = 8, slots = 1>>
```

VPTO lowering result for one full 8-row tile:

```text
%all_b32 = pto.pge_b32 "PAT_ALL"
%block8 = pto.pge_b32 "PAT_VL8"
%one_b32 = pto.pge_b32 "PAT_VL1"

%x0 = pto.vlds %base[%row_off_0] {dist = "NORM"} : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
%x1 = pto.vlds %base[%row_off_1] {dist = "NORM"} : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
%x2 = pto.vlds %base[%row_off_2] {dist = "NORM"} : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
%x3 = pto.vlds %base[%row_off_3] {dist = "NORM"} : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
%x4 = pto.vlds %base[%row_off_4] {dist = "NORM"} : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
%x5 = pto.vlds %base[%row_off_5] {dist = "NORM"} : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
%x6 = pto.vlds %base[%row_off_6] {dist = "NORM"} : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
%x7 = pto.vlds %base[%row_off_7] {dist = "NORM"} : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>

%p0 = pto.vcgadd %x0, %all_b32 : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%p1 = pto.vcgadd %x1, %all_b32 : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%p2 = pto.vcgadd %x2, %all_b32 : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%p3 = pto.vcgadd %x3, %all_b32 : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%p4 = pto.vcgadd %x4, %all_b32 : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%p5 = pto.vcgadd %x5, %all_b32 : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%p6 = pto.vcgadd %x6, %all_b32 : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%p7 = pto.vcgadd %x7, %all_b32 : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>

%sum0 = pto.vcadd %p0, %block8 : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%sum1 = pto.vcadd %p1, %block8 : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%sum2 = pto.vcadd %p2, %block8 : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%sum3 = pto.vcadd %p3, %block8 : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%sum4 = pto.vcadd %p4, %block8 : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%sum5 = pto.vcadd %p5, %block8 : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%sum6 = pto.vcadd %p6, %block8 : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%sum7 = pto.vcadd %p7, %block8 : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>

%rhs0 = pto.vsldb %rhs_ptr_0, %c0_i16, %c0_i16, %one_b32
  : !pto.ptr<f32, ub>, i16, i16, !pto.mask<b32> -> !pto.vreg<64xf32>
%rhs1 = pto.vsldb %rhs_ptr_1, %c0_i16, %c0_i16, %one_b32
  : !pto.ptr<f32, ub>, i16, i16, !pto.mask<b32> -> !pto.vreg<64xf32>
%rhs2 = pto.vsldb %rhs_ptr_2, %c0_i16, %c0_i16, %one_b32
  : !pto.ptr<f32, ub>, i16, i16, !pto.mask<b32> -> !pto.vreg<64xf32>
%rhs3 = pto.vsldb %rhs_ptr_3, %c0_i16, %c0_i16, %one_b32
  : !pto.ptr<f32, ub>, i16, i16, !pto.mask<b32> -> !pto.vreg<64xf32>
%rhs4 = pto.vsldb %rhs_ptr_4, %c0_i16, %c0_i16, %one_b32
  : !pto.ptr<f32, ub>, i16, i16, !pto.mask<b32> -> !pto.vreg<64xf32>
%rhs5 = pto.vsldb %rhs_ptr_5, %c0_i16, %c0_i16, %one_b32
  : !pto.ptr<f32, ub>, i16, i16, !pto.mask<b32> -> !pto.vreg<64xf32>
%rhs6 = pto.vsldb %rhs_ptr_6, %c0_i16, %c0_i16, %one_b32
  : !pto.ptr<f32, ub>, i16, i16, !pto.mask<b32> -> !pto.vreg<64xf32>
%rhs7 = pto.vsldb %rhs_ptr_7, %c0_i16, %c0_i16, %one_b32
  : !pto.ptr<f32, ub>, i16, i16, !pto.mask<b32> -> !pto.vreg<64xf32>

%out0 = pto.vadd %sum0, %rhs0, %one_b32 : !pto.vreg<64xf32>
%out1 = pto.vadd %sum1, %rhs1, %one_b32 : !pto.vreg<64xf32>
%out2 = pto.vadd %sum2, %rhs2, %one_b32 : !pto.vreg<64xf32>
%out3 = pto.vadd %sum3, %rhs3, %one_b32 : !pto.vreg<64xf32>
%out4 = pto.vadd %sum4, %rhs4, %one_b32 : !pto.vreg<64xf32>
%out5 = pto.vadd %sum5, %rhs5, %one_b32 : !pto.vreg<64xf32>
%out6 = pto.vadd %sum6, %rhs6, %one_b32 : !pto.vreg<64xf32>
%out7 = pto.vadd %sum7, %rhs7, %one_b32 : !pto.vreg<64xf32>

pto.vsts %out0, %out[%group_tile_off_0], %one_b32 {dist = "NORM_B32"} : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
pto.vsts %out1, %out[%group_tile_off_1], %one_b32 {dist = "NORM_B32"} : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
pto.vsts %out2, %out[%group_tile_off_2], %one_b32 {dist = "NORM_B32"} : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
pto.vsts %out3, %out[%group_tile_off_3], %one_b32 {dist = "NORM_B32"} : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
pto.vsts %out4, %out[%group_tile_off_4], %one_b32 {dist = "NORM_B32"} : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
pto.vsts %out5, %out[%group_tile_off_5], %one_b32 {dist = "NORM_B32"} : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
pto.vsts %out6, %out[%group_tile_off_6], %one_b32 {dist = "NORM_B32"} : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
pto.vsts %out7, %out[%group_tile_off_7], %one_b32 {dist = "NORM_B32"} : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
```

Memory result:

```text
for r = 0..7:
  out[group_tile_off + r * 8] = reduce(row_r[0..63]) + rhs[r]
```

#### 3.7.3 Reduce, Broadcast, Elementwise, Reduce, Store

VMI input:

```text
%x = pto.vmi.load %base[%off]
  : memref<512xf32> -> !pto.vmi.vreg<512xf32>
%sum = pto.vmi.group_reduce_addf %x, %mask {num_groups = 8}
%b = pto.vmi.group_broadcast %sum {num_groups = 8}
%y = pto.vmi.vmul %x, %b
%ysum = pto.vmi.group_reduce_addf %y, %mask {num_groups = 8}
%c8 = arith.constant 8 : index
pto.vmi.group_store %ysum, %out[%group_off], %c8 {num_groups = 8}
```

Assigned layouts:

```text
%x, %b, %y:
  !pto.vmi.vreg<512xf32, #pto.vmi.layout<contiguous>>

%sum, %ysum:
  !pto.vmi.vreg<512xf32, #pto.vmi.layout<num_groups = 8, slots = 1>>
```

VPTO lowering result for one full 8-row tile:

```text
%all_b32 = pto.pge_b32 "PAT_ALL"
%block8 = pto.pge_b32 "PAT_VL8"
%one_b32 = pto.pge_b32 "PAT_VL1"

// The compiler emits this row-local block once for each r in 0..7.
%x_r = pto.vlds %base[%row_off_r] {dist = "NORM"}
  : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>

%p_r = pto.vcgadd %x_r, %all_b32
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%sum_r = pto.vcadd %p_r, %block8
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>

// This vdup is the lowering of pto.vmi.group_broadcast for slots=1.
%b_r = pto.vdup %sum_r, %all_b32 {position = "LOWEST"}
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>

%y_r = pto.vmul %x_r, %b_r, %all_b32 : !pto.vreg<64xf32>

%yp_r = pto.vcgadd %y_r, %all_b32
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%ysum_r = pto.vcadd %yp_r, %block8
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>

pto.vsts %ysum_r, %out[%group_tile_off_r], %one_b32 {dist = "NORM_B32"}
  : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
```

The row-local block above is not a runtime loop requirement. It is the repeated
VPTO shape for row offsets `%row_off_0` through `%row_off_7` and store offsets
`%group_tile_off_0` through `%group_tile_off_7`.

Memory result:

```text
for r = 0..7:
  s = reduce(row_r[0..63])
  out[group_tile_off + r * 8] =
      reduce_i(row_r[i] * s for i = 0..63)
    = s * s
```

#### 3.7.4 Slots=1 Store Lowers To Packed Or Point Stores

The row-local S=64 result uses one physical vreg per group with the semantic
value in lane 0:

```text
%sum_r lane 0 = reduce(row_r[0..63])
```

The current VPTO lowering for `slots = 1` group_store has two paths.

For unit-stride output where all groups fit in one physical vector, the
lowering packs the lane-0 values into one dense vector and stores that vector
with a normal `vsts`.

For non-unit row strides, each group stores its lane-0 scalar with a point
store. That emits `vsts` with `dist = "1PT_B32"` for f32 and only requires the
natural 4B alignment of the scalar element.

VMI input:

```text
%c1 = arith.constant 1 : index
%sum = pto.vmi.group_reduce_addf %x, %mask {num_groups = 8}
pto.vmi.group_store %sum, %out[%group_off], %c1 {num_groups = 8}
```

Current checked-in coverage for the point-store path is:

```text
test/lit/vmi_new/vmi_to_vpto_group_store_slots1_1pt.pto
```

### 3.8 `group_reduce -> truncf -> group_broadcast -> store`

This case keeps the source op order by representing the `f32 -> f16` cast result
as lane-strided group slots.  The source reduction is a packed f32 group-slot
value in b32 lanes 0..7.  After `truncf`, the f16 values occupy even b16 lanes
0, 2, 4, ..., 14, so the result layout is `slots = 8, lane_stride = 2`.

VMI input:

```text
%x = pto.vmi.load %base[%off]
  : memref<128xf32> -> !pto.vmi.vreg<128xf32>
%sum32 = pto.vmi.group_reduce_addf %x, %mask {num_groups = 8}
%sum16 = pto.vmi.truncf %sum32
%b16   = pto.vmi.group_broadcast %sum16 {num_groups = 8}
pto.vmi.store %b16, %out[%off]
```

Final assigned IR:

```text
%mask = pto.vmi.create_mask %active
  : index -> !pto.vmi.mask<128xb32,
       #pto.vmi.layout<contiguous>>

%x = pto.vmi.load %base[%off]
  : memref<128xf32> -> !pto.vmi.vreg<128xf32,
       #pto.vmi.layout<block_deinterleaved=2>>

%mask_d2 = pto.vmi.ensure_mask_layout %mask
  : !pto.vmi.mask<128xb32,
       #pto.vmi.layout<contiguous>>
 -> !pto.vmi.mask<128xb32,
       #pto.vmi.layout<block_deinterleaved=2>>

%sum32 = pto.vmi.group_reduce_addf %x, %mask_d2 {num_groups = 8}
  : !pto.vmi.vreg<128xf32,
       #pto.vmi.layout<block_deinterleaved=2>>,
    !pto.vmi.mask<128xb32,
       #pto.vmi.layout<block_deinterleaved=2>>
 -> !pto.vmi.vreg<8xf32,
       #pto.vmi.layout<num_groups = 8, slots = 8>>

%sum16 = pto.vmi.truncf %sum32
  : !pto.vmi.vreg<8xf32,
       #pto.vmi.layout<num_groups = 8, slots = 8>>
 -> !pto.vmi.vreg<8xf16,
       #pto.vmi.layout<num_groups = 8, slots = 8, lane_stride = 2>>

%b16 = pto.vmi.group_broadcast %sum16 {num_groups = 8}
  : !pto.vmi.vreg<8xf16,
       #pto.vmi.layout<num_groups = 8, slots = 8, lane_stride = 2>>
 -> !pto.vmi.vreg<128xf16,
       #pto.vmi.layout<contiguous>>

pto.vmi.store %b16, %out[%off]
  : !pto.vmi.vreg<128xf16,
       #pto.vmi.layout<contiguous>>, !pto.ptr<f16, ub>
```

The layout without `lane_stride = 2` is illegal: it would claim that f16 group
results are packed in lanes 0..7, but `vcvt` produces them in lanes
0, 2, 4, ..., 14.  No lane compaction is performed in this case.

VPTO lowering result for one full 8-row tile:

```text
%all_b32 = pto.pge_b32 "PAT_ALL"
%sum_mask = pto.pge_b32 "PAT_VL8"

%x_lo, %x_hi = pto.vldsx2 %base[%tile_off], "BDINTLV"
  : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>, !pto.vreg<64xf32>

%x_lo_sum = pto.vcgadd %x_lo, %all_b32
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%x_hi_sum = pto.vcgadd %x_hi, %all_b32
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>

%sum32_block = pto.vadd %x_lo_sum, %x_hi_sum, %sum_mask
  : !pto.vreg<64xf32>

%sum16_block = pto.vcvt %sum32_block, %sum_mask {part = "EVEN", rnd = "R", sat = "SAT"}
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<128xf16>

%broadcast_idx = compute index vector [0 repeated 16, 2 repeated 16,
                                       4 repeated 16, 6 repeated 16,
                                       8 repeated 16, 10 repeated 16,
                                       12 repeated 16, 14 repeated 16]
  : !pto.vreg<128xi16>

// The index vector uses the lane_stride=2 group-slot source layout.
%b16 = pto.vselr %sum16_block, %broadcast_idx
  : !pto.vreg<128xf16>, !pto.vreg<128xi16> -> !pto.vreg<128xf16>

%all_b16 = pto.pge_b16 "PAT_ALL"
pto.vsts %b16, %out[%off], %all_b16 {dist = "NORM_B16"}
  : !pto.vreg<128xf16>, !pto.ptr<f16, ub>, !pto.mask<b16>
```

Memory result:

```text
for r = 0..7:
  s32 = reduce(row_r[0..15])
  s16 = truncf(s32)
  out[r * 16 + 0 .. r * 16 + 15] = splat(s16)
```

Required assignment rule:

```text
Packed `slots = 8` group-slot `truncf` may be assigned only when the narrowing
result layout records the sub-lane gap:

  f32 group_slots(G, slots=8, lane_stride=1)
    -> f16 group_slots(G, slots=8, lane_stride=2)

Consumers of lane-strided group slots, including `group_broadcast`, must select
source lanes with `(group % slots) * lane_stride`.  They must not treat
`slots = 8` as lanes 0..7 after width-changing casts.
```

### 3.9 Illegal Dense Consumer Of Group Slots

VMI input:

```text
%sum32 = pto.vmi.group_reduce_addf %x, %mask {num_groups = G}
pto.vmi.store %sum32, %out[%off]
```

Assigned layouts before the illegal consumer:

```text
%sum32 : group_slots(G,K)
```

Required diagnostic:

```text
VMI-LAYOUT-CONTRACT:
  pto.vmi.store cannot consume #pto.vmi.layout<num_groups = G, slots = K>
  as a dense vector. Use pto.vmi.group_store, pto.vmi.group_broadcast, or an
  explicit group-pack op.
```

It must not be diagnosed as:

```text
dense store materializes group slots implicitly
```

That behavior would silently reinterpret a group-slot value as a dense
vector.

### 3.10 Non-Load Producer Feeding S=32 `group_reduce`

This case proves that layout assignment is consumer-driven. The producer of the
S=32 input is an elementwise op, not a load. The S=32 `group_reduce` still
requires the elementwise result to be `deinterleaved = 4`, and that requirement
must propagate backward through the elementwise op to both operands.

VMI input:

```text
%a = pto.vmi.load %base[%off]
  : memref<256xf32> -> !pto.vmi.vreg<256xf32>
%bias = pto.vmi.broadcast %bias_s
  : f32 -> !pto.vmi.vreg<256xf32>
%x = pto.vmi.vadd %a, %bias
%sum = pto.vmi.group_reduce_addf %x, %mask {num_groups = 8}
pto.vmi.group_store %sum, %out[%group_off], %c1 {num_groups = 8}
```

Assigned layouts:

```text
%a, %bias, %x:
  !pto.vmi.vreg<256xf32, #pto.vmi.layout<block_deinterleaved=4>>

%sum:
  !pto.vmi.vreg<256xf32, #pto.vmi.layout<num_groups = 8, slots = 8>>
```

VPTO lowering result for one full `8x32xf32` tile:

```text
%all_b32 = pto.pge_b32 "PAT_ALL"
%sum_mask = pto.pge_b32 "PAT_VL8"

%a_even_0, %a_odd_0 = pto.vldsx2 %base[%tile_off_0], "DINTLV_B32"
  : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>, !pto.vreg<64xf32>
%a_even_1, %a_odd_1 = pto.vldsx2 %base[%tile_off_1], "DINTLV_B32"
  : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>, !pto.vreg<64xf32>

%a_p0, %a_p2 = pto.vdintlv %a_even_0, %a_even_1
  : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>, !pto.vreg<64xf32>
%a_p1, %a_p3 = pto.vdintlv %a_odd_0, %a_odd_1
  : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>, !pto.vreg<64xf32>

%bias_p0 = pto.vdup %bias_s, %all_b32 : f32, !pto.mask<b32> -> !pto.vreg<64xf32>
%bias_p1 = pto.vdup %bias_s, %all_b32 : f32, !pto.mask<b32> -> !pto.vreg<64xf32>
%bias_p2 = pto.vdup %bias_s, %all_b32 : f32, !pto.mask<b32> -> !pto.vreg<64xf32>
%bias_p3 = pto.vdup %bias_s, %all_b32 : f32, !pto.mask<b32> -> !pto.vreg<64xf32>

%x_p0 = pto.vadd %a_p0, %bias_p0, %all_b32 : !pto.vreg<64xf32>
%x_p1 = pto.vadd %a_p1, %bias_p1, %all_b32 : !pto.vreg<64xf32>
%x_p2 = pto.vadd %a_p2, %bias_p2, %all_b32 : !pto.vreg<64xf32>
%x_p3 = pto.vadd %a_p3, %bias_p3, %all_b32 : !pto.vreg<64xf32>

%s0 = pto.vcgadd %x_p0, %all_b32 : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%s1 = pto.vcgadd %x_p1, %all_b32 : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%s2 = pto.vcgadd %x_p2, %all_b32 : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%s3 = pto.vcgadd %x_p3, %all_b32 : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>

%s01 = pto.vadd %s0, %s1, %sum_mask : !pto.vreg<64xf32>
%s23 = pto.vadd %s2, %s3, %sum_mask : !pto.vreg<64xf32>
%sum_block = pto.vadd %s01, %s23, %sum_mask : !pto.vreg<64xf32>

pto.vsts %sum_block, %out[%group_tile_off], %sum_mask {dist = "NORM_B32"}
  : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
```

Memory result:

```text
for r = 0..7:
  out[group_tile_off + r] =
    reduce_i(base[row_r, i] + bias_s for i = 0..31)
```

### 3.11 Partial Tail Groups

Tail handling must be separated by the physical input layout. Row-local S=64
can avoid inactive rows entirely. Load-fused S=16/S=32 cannot safely do that
with the current `vldsx2` materialization unless the source is known to be
full-tile readable.

#### 3.11.1 S=64 Active Row Tail

VMI input:

```text
%x = pto.vmi.load %base[%off]
  : memref<384xf32> -> !pto.vmi.vreg<384xf32>
%sum = pto.vmi.group_reduce_addf %x, %mask {num_groups = 6}
%c8 = arith.constant 8 : index
pto.vmi.group_store %sum, %out[%group_off], %c8 {num_groups = 6}
```

Assigned layouts:

```text
%x:
  !pto.vmi.vreg<384xf32, #pto.vmi.layout<contiguous>>

%sum:
  !pto.vmi.vreg<384xf32, #pto.vmi.layout<num_groups = 6, slots = 1>>
```

VPTO lowering result:

```text
%all_b32 = pto.pge_b32 "PAT_ALL"
%block8 = pto.pge_b32 "PAT_VL8"
%one_b32 = pto.pge_b32 "PAT_VL1"

// Emit this row-local block for r = 0..5 only. No load or store is emitted for
// rows 6 and 7.
%x_r = pto.vlds %base[%row_off_r] {dist = "NORM"}
  : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
%p_r = pto.vcgadd %x_r, %all_b32
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%sum_r = pto.vcadd %p_r, %block8
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
pto.vsts %sum_r, %out[%group_tile_off_r], %one_b32 {dist = "NORM_B32"}
  : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
```

Memory result:

```text
for r = 0..5:
  out[group_tile_off + r * 8] = reduce(row_r[0..63])
```

#### 3.11.2 S=32 Tail Without Full-Tile Read Contract

VMI input:

```text
%x = pto.vmi.load %base[%off]
  : memref<192xf32> -> !pto.vmi.vreg<192xf32>
%sum = pto.vmi.group_reduce_addf %x, %mask {num_groups = 6}
pto.vmi.group_store %sum, %out[%group_off], %c1 {num_groups = 6}
```

Assigned layout requested by the consumer:

```text
%x:
  !pto.vmi.vreg<192xf32, #pto.vmi.layout<deinterleaved = 4>>

%sum:
  !pto.vmi.vreg<192xf32, #pto.vmi.layout<num_groups = 6, slots = 8>>
```

Required diagnostic when the source does not carry a full-tile-readable
contract:

```text
VMI-LAYOUT-CONTRACT:
  pto.vmi.group_reduce_addf with group size 32 and num_groups tail 6 requires
  materializing #pto.vmi.layout<deinterleaved = 4>. The fast lowering support
  uses vldsx2 DINTLV_B32 over a full 8-row tile. This source is not marked
  full-tile-readable, and the stable gather tail fallback is not implemented.
```

If a future option enables the stable gather tail fallback, the same VMI input
may lower by gathering only the active lanes. Until that support exists, the
converter must not silently issue the full-tile `vldsx2` loads.

### 3.12 Control-Flow Join Before `group_reduce`

The layout carried by a value must survive block arguments. In MLIR converter
terms, the logical VMI value lowered through control flow becomes a tuple of
physical VPTO values with one tuple type per assigned layout.

VMI input:

```text
%x = scf.if %cond -> !pto.vmi.vreg<256xf32> {
  %a = pto.vmi.load %a_base[%a_off]
    : memref<256xf32> -> !pto.vmi.vreg<256xf32>
  scf.yield %a : !pto.vmi.vreg<256xf32>
} else {
  %b = pto.vmi.load %b_base[%b_off]
    : memref<256xf32> -> !pto.vmi.vreg<256xf32>
  scf.yield %b : !pto.vmi.vreg<256xf32>
}
%sum = pto.vmi.group_reduce_addf %x, %mask {num_groups = 8}
pto.vmi.group_store %sum, %out[%group_off], %c1 {num_groups = 8}
```

Assigned layouts:

```text
%a, %b, %x:
  !pto.vmi.vreg<256xf32, #pto.vmi.layout<deinterleaved = 4>>

%sum:
  !pto.vmi.vreg<256xf32, #pto.vmi.layout<num_groups = 8, slots = 8>>
```

VPTO lowering result for the join:

```text
%x_p0, %x_p1, %x_p2, %x_p3 =
  scf.if %cond
    -> (!pto.vreg<64xf32>, !pto.vreg<64xf32>,
        !pto.vreg<64xf32>, !pto.vreg<64xf32>) {
    %a_even_0, %a_odd_0 = pto.vldsx2 %a_base[%a_tile_off_0], "DINTLV_B32"
      : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>, !pto.vreg<64xf32>
    %a_even_1, %a_odd_1 = pto.vldsx2 %a_base[%a_tile_off_1], "DINTLV_B32"
      : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>, !pto.vreg<64xf32>
    %a_p0, %a_p2 = pto.vdintlv %a_even_0, %a_even_1
      : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>, !pto.vreg<64xf32>
    %a_p1, %a_p3 = pto.vdintlv %a_odd_0, %a_odd_1
      : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>, !pto.vreg<64xf32>
    scf.yield %a_p0, %a_p1, %a_p2, %a_p3
      : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.vreg<64xf32>
  } else {
    %b_even_0, %b_odd_0 = pto.vldsx2 %b_base[%b_tile_off_0], "DINTLV_B32"
      : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>, !pto.vreg<64xf32>
    %b_even_1, %b_odd_1 = pto.vldsx2 %b_base[%b_tile_off_1], "DINTLV_B32"
      : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>, !pto.vreg<64xf32>
    %b_p0, %b_p2 = pto.vdintlv %b_even_0, %b_even_1
      : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>, !pto.vreg<64xf32>
    %b_p1, %b_p3 = pto.vdintlv %b_odd_0, %b_odd_1
      : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>, !pto.vreg<64xf32>
    scf.yield %b_p0, %b_p1, %b_p2, %b_p3
      : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.vreg<64xf32>
  }
```

The consumer after the join uses the same S=32 reduction lowering support as
section 3.6:

```text
%all_b32 = pto.pge_b32 "PAT_ALL"
%sum_mask = pto.pge_b32 "PAT_VL8"

%s0 = pto.vcgadd %x_p0, %all_b32 : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%s1 = pto.vcgadd %x_p1, %all_b32 : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%s2 = pto.vcgadd %x_p2, %all_b32 : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%s3 = pto.vcgadd %x_p3, %all_b32 : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>

%s01 = pto.vadd %s0, %s1, %sum_mask : !pto.vreg<64xf32>
%s23 = pto.vadd %s2, %s3, %sum_mask : !pto.vreg<64xf32>
%sum_block = pto.vadd %s01, %s23, %sum_mask : !pto.vreg<64xf32>

pto.vsts %sum_block, %out[%group_tile_off], %sum_mask {dist = "NORM_B32"}
  : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
```

Memory result:

```text
for r = 0..7:
  selected_row = cond ? a_row_r : b_row_r
  out[group_tile_off + r] = reduce(selected_row[0..31])
```

If the two branches cannot be assigned the same layout and no materialization
support exists before `scf.yield`, the required diagnostic is:

```text
VMI-LAYOUT-CONTRACT:
  scf.yield joins incompatible VMI layouts for !pto.vmi.vreg<256xf32>.
  Expected #pto.vmi.layout<deinterleaved = 4> on every incoming value.
```

### 3.13 Packed Group-Slot `f32 -> f16` Cast

This case is intentionally illegal for the current S=16/S=32 packed
group-slot layout. It prevents the compiler from treating a width-changing
`vcvt` as if it preserved low-lane group slots.

VMI input:

```text
%x = pto.vmi.load %base[%off]
  : memref<128xf32> -> !pto.vmi.vreg<128xf32>
%sum32 = pto.vmi.group_reduce_addf %x, %mask {num_groups = 8}
%sum16 = pto.vmi.truncf %sum32
pto.vmi.group_store %sum16, %out[%group_off], %c1 {num_groups = 8}
```

Assigned layouts before the illegal cast:

```text
%x:
  !pto.vmi.vreg<128xf32,
    #pto.vmi.layout<block_deinterleaved=2>>

%sum32:
  !pto.vmi.vreg<128xf32, #pto.vmi.layout<num_groups = 8, slots = 8>>
```

Required diagnostic:

```text
VMI-LAYOUT-CONTRACT:
  pto.vmi.truncf cannot lower from
  #pto.vmi.layout<num_groups = 8, slots = 8> f32 to f16 because no
  slot-preserving width-changing VPTO support exists. f32->f16 vcvt writes
  even/odd sub-lanes, not lanes 0..7. Use group_broadcast before truncf, or
  keep the group_store element type as f32.
```

This does not contradict section 3.8. Section 3.8 is legal because the cast is
commuted after `group_broadcast`, where the value is dense again.

### 3.14 Unsupported Group Size

VMI input:

```text
%x = pto.vmi.load %base[%off]
  : memref<96xf32> -> !pto.vmi.vreg<96xf32>
%sum = pto.vmi.group_reduce_addf %x, %mask {num_groups = 8}
pto.vmi.group_store %sum, %out[%group_off], %c1 {num_groups = 8}
```

Here `S = 96 / 8 = 12` f32 elements per group. The current VCG-based lowering
support uses 32B groups, i.e. 8 f32 elements per row fragment:

```text
S = 8   -> one VCGADD block per group
S = 16  -> two 8-lane row fragments, add partial sums
S = 32  -> four 8-lane row fragments, add partial sums
S = 64  -> one full 256B row, VCGADD then VCADD
```

Required diagnostic:

```text
VMI-LAYOUT-CONTRACT:
  pto.vmi.group_reduce_addf with f32 group size 12 has no supported VPTO
  layout/lowering path. Supported VCG-based f32 group sizes are 8, 16, 32, and 64.
  A scalar/gather fallback or a rewrite to logical group size 16 with an
  explicit per-group mask is required.
```

### 3.15 Compact S=12 Written As Logical S=16

If the program wants to use the S=16 lowering for data with 12 semantic f32
elements per group, the IR must distinguish two sizes:

```text
logical group size used by VMI ops: 16
active elements per group:          12
```

The mask is not a prefix mask over the whole vector. It is a per-group mask:

```text
mask lane i is active iff (i % 16) < 12
```

The group load surface carries the physical source stride as an SSA operand:

```text
%x = pto.vmi.group_load %base[%off], %source_group_stride
  {num_groups = G, group_size = S}
  : !pto.ptr<T, ub>, index -> !pto.vmi.vreg<NxT>
```

`source_group_stride` is in elements, not bytes. It is an operand because it may
come from a dynamic leading dimension, a subview, or a runtime tile descriptor.
Static strides use a constant index operand and can be canonicalized later.
`group_size` remains an attribute in this design because it selects the logical
load layout. `active_elems_per_group` belongs to the mask producer, not to the
load.

Grouped masks use a paired `pto.vmi.create_group_mask` op. It is intentionally
separate from ordinary prefix `pto.vmi.create_mask` so the IR makes group
semantics explicit next to `pto.vmi.group_load` / `pto.vmi.group_reduce_*`:

```text
%mask = pto.vmi.create_group_mask %active_elems_per_group
  {num_groups = G, group_size = S}
  : index -> !pto.vmi.mask<(G*S)xpred>
```

Semantics:

```text
lane i is active iff (i % S) < active_elems_per_group
```

Current lowering support covers constant `active_elems_per_group`. Dynamic
grouped masks require a runtime lane-index predicate materializer and remain a
separate implementation item.

Ordinary `pto.vmi.create_mask %active_lanes` keeps the prefix-mask meaning:

```text
lane i is active iff i < active_lanes
```

#### 3.15.1 Existing Design Works If Source Row Stride Is 16

If memory already has a 16-f32 row stride, the user can write a logical S=16
tile and mask off the last four lanes of every group.

VMI input:

```text
%stride16 = arith.constant 16 : index
%x = pto.vmi.group_load %base[%off], %stride16
  {num_groups = 8, group_size = 16}
  : !pto.ptr<f32, ub>, index -> !pto.vmi.vreg<128xf32>
%c12 = arith.constant 12 : index
%mask = pto.vmi.create_group_mask %c12 {num_groups = 8, group_size = 16}
  : index -> !pto.vmi.mask<128xpred>
%sum = pto.vmi.group_reduce_addf %x, %mask {num_groups = 8}
pto.vmi.group_store %sum, %out[%group_off], %c1 {num_groups = 8}
```

Assigned layouts:

```text
%x:
  !pto.vmi.vreg<128xf32,
    #pto.vmi.layout<block_deinterleaved=2>>

%mask:
  !pto.vmi.mask<128xpred,
    #pto.vmi.layout<block_deinterleaved=2>>

%sum:
  !pto.vmi.vreg<128xf32, #pto.vmi.layout<num_groups = 8, slots = 8>>

%x32_for_store:
  pto.vmi.ensure_layout %x32
    : #pto.vmi.layout<deinterleaved = 2> -> #pto.vmi.layout<contiguous>
```

VPTO lowering result for one `8x16xf32` tile:

```text
%lo_mask = pto.pge_b32 "PAT_ALL"
%sum_mask = pto.pge_b32 "PAT_VL8"

%lane = pto.vci %c0_i32 : i32 -> !pto.vreg<64xi32>
%row = pto.vshrs %lane, %c3_i16, %lo_mask
  : !pto.vreg<64xi32>, i16, !pto.mask<b32> -> !pto.vreg<64xi32>
%row8 = pto.vshls %row, %c3_i16, %lo_mask
  : !pto.vreg<64xi32>, i16, !pto.mask<b32> -> !pto.vreg<64xi32>
%col = pto.vsub %lane, %row8, %lo_mask
  : !pto.vreg<64xi32>
%hi4_mask = pto.vcmps %col, %c4_i32, %lo_mask, "lt"
  : !pto.vreg<64xi32>, i32, !pto.mask<b32> -> !pto.mask<b32>

%lo, %hi = pto.vldsx2 %base[%tile_off], "BDINTLV"
  : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>, !pto.vreg<64xf32>

%lo lanes r*8 .. r*8+7 = row_r[0..7]
%hi lanes r*8 .. r*8+3 = row_r[8..11]
%hi lanes r*8+4 .. r*8+7 = row_r[12..15]  // inactive by mask

%lo_sum = pto.vcgadd %lo, %lo_mask
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%hi_sum = pto.vcgadd %hi, %hi4_mask
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>

%sum_block = pto.vadd %lo_sum, %hi_sum, %sum_mask
  : !pto.vreg<64xf32>

pto.vsts %sum_block, %out[%group_tile_off], %sum_mask {dist = "NORM_B32"}
  : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
```

Memory result:

```text
for r = 0..7:
  out[group_tile_off + r] = reduce(row_r[0..11])
```

Design requirement added by this case: VMI mask lowering must support
group-periodic masks by generating the predicate from lane indices. It must not
rewrite this mask to `PAT_M4`: VISA defines `M4` as multiples of 4, not the
first four lanes of each 8-lane block.

```text
lane = vci(0)
row  = lane >> 3
col  = lane - (row << 3)
mask = col < 4
```

#### 3.15.2 Source Row Stride Greater Than 16

For now, support the non-compact case where each physical row has at least 16
f32 slots and the row stride is greater than 16. The fast strided-block path
requires the row stride to be a multiple of one 32B block:

```text
source_group_stride % 8 == 0
```

The example below uses `source_group_stride = 24`. Each row has 12 semantic
values, 4 masked-but-readable slots, and 8 extra skipped slots:

```text
row_r[0..11]   semantic
row_r[12..15]  readable but inactive for the S=16 logical group
row_r[16..23]  outside the logical group
```

VMI input:

```text
%stride24 = arith.constant 24 : index
%x = pto.vmi.group_load %base[%off], %stride24
  {num_groups = 8, group_size = 16}
  : !pto.ptr<f32, ub>, index -> !pto.vmi.vreg<128xf32>
%c12 = arith.constant 12 : index
%mask = pto.vmi.create_group_mask %c12 {num_groups = 8, group_size = 16}
  : index -> !pto.vmi.mask<128xpred>
%sum = pto.vmi.group_reduce_addf %x, %mask {num_groups = 8}
pto.vmi.group_store %sum, %out[%group_off], %c1 {num_groups = 8}
```

Assigned layouts are the same as section 3.15.1:

```text
%x, %mask:
  #pto.vmi.layout<block_deinterleaved=2>
%sum:
  #pto.vmi.layout<num_groups = 8, slots = 8>
```

VPTO lowering result:

```text
%lo_mask = pto.pge_b32 "PAT_ALL"
%sum_mask = pto.pge_b32 "PAT_VL8"

%lane = pto.vci %c0_i32 : i32 -> !pto.vreg<64xi32>
%row = pto.vshrs %lane, %c3_i16, %lo_mask
  : !pto.vreg<64xi32>, i16, !pto.mask<b32> -> !pto.vreg<64xi32>
%row8 = pto.vshls %row, %c3_i16, %lo_mask
  : !pto.vreg<64xi32>, i16, !pto.mask<b32> -> !pto.vreg<64xi32>
%col = pto.vsub %lane, %row8, %lo_mask
  : !pto.vreg<64xi32>
%hi4_mask = pto.vcmps %col, %c4_i32, %lo_mask, "lt"
  : !pto.vreg<64xi32>, i32, !pto.mask<b32> -> !pto.mask<b32>

// source_group_stride = 24 f32 = 3 * 32B blocks.
%stride_blocks = %c3_i16

%base_lo = %base + tile_off
%base_hi = %base + tile_off + 8

%lo = pto.vsldb %base_lo, %stride_blocks, %c0_i16, %lo_mask
  : !pto.ptr<f32, ub>, i16, i16, !pto.mask<b32> -> !pto.vreg<64xf32>
%hi = pto.vsldb %base_hi, %stride_blocks, %c0_i16, %lo_mask
  : !pto.ptr<f32, ub>, i16, i16, !pto.mask<b32> -> !pto.vreg<64xf32>

%lo lanes r*8 .. r*8+7 = row_r[0..7]
%hi lanes r*8 .. r*8+7 = row_r[8..15]

%lo_sum = pto.vcgadd %lo, %lo_mask
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%hi_sum = pto.vcgadd %hi, %hi4_mask
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>

%sum_block = pto.vadd %lo_sum, %hi_sum, %sum_mask
  : !pto.vreg<64xf32>

pto.vsts %sum_block, %out[%group_tile_off], %sum_mask {dist = "NORM_B32"}
  : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
```

Memory result:

```text
for r = 0..7:
  out[group_tile_off + r] =
    reduce(base[tile_off + r * 24 + 0 .. tile_off + r * 24 + 11])
```

If `source_group_stride > 16` but is not a multiple of 8 f32 elements, this
strided-block path is not legal because `vsldb` block addresses are 32B based.
That case remains unsupported until a gather materialization is selected.

#### 3.15.3 Compact Source Row Stride 12

Compact storage is explicitly out of scope for the first implementation:

```text
row0[0..11], row1[0..11], row2[0..11], ...
```

Required diagnostic:

```text
VMI-LAYOUT-CONTRACT:
  logical group size 16 with active_elems_per_group 12 and
  source_group_stride 12 requires compact-row gather materialization. This
  plan is not part of the initial VMI layout lowering.
```

### 3.16 `group_slot_load` Layout Contract

`group_slot_load` is separate from `group_load`.

```text
group_load:
  loads group_size data elements per group and produces dense grouped data.

group_slot_load:
  loads one scalar value per group and produces group slots.
```

Surface form:

```text
%v = pto.vmi.group_slot_load %base[%off], %source_group_stride
  {num_groups = G}
  : !pto.ptr<T, ub>, index -> !pto.vmi.vreg<NxT>
```

Semantics:

```text
semantic group slot g = base[off + g * source_group_stride]
```

The result logical lane count `N` remains the surrounding VMI value shape. Only
the `G` group slots are semantic. Layout assignment chooses the group-slot physical
placement requested by the consumer:

```text
#pto.vmi.layout<num_groups = G, slots = 8>
#pto.vmi.layout<num_groups = G, slots = 1>
```

#### 3.16.1 Packed `group_slot_load`, `slots = 8`

VMI input:

```text
%rhs = pto.vmi.group_slot_load %rhs_base[%rhs_off], %c1 {num_groups = 8}
  : !pto.ptr<f32, ub>, index -> !pto.vmi.vreg<128xf32>
pto.vmi.group_store %rhs, %out[%group_off], %c1 {num_groups = 8}
```

Assigned layout:

```text
%rhs:
  !pto.vmi.vreg<128xf32, #pto.vmi.layout<num_groups = 8, slots = 8>>
```

VPTO lowering result:

```text
%slot_mask = pto.pge_b32 "PAT_VL8"
%one_block = pto.pge_b32 "PAT_VL1"

// source_group_stride = 1, so one 32B block contains all 8 scalar group slots.
%rhs_block = pto.vsldb %rhs_base[%rhs_off], %c0_i16, %c0_i16, %one_block
  : !pto.ptr<f32, ub>, i16, i16, !pto.mask<b32> -> !pto.vreg<64xf32>

pto.vsts %rhs_block, %out[%group_off], %slot_mask {dist = "NORM_B32"}
  : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
```

Memory result:

```text
for g = 0..7:
  out[group_off + g] = rhs_base[rhs_off + g]
```

The single-group unit-stride case is special. A `group_slot_load` with
`num_groups = 1` has only one semantic scalar slot, so it lowers through
`pto.vlds {dist = "BRC_B*"}` rather than `pto.vsldb`. `vsldb` requires its
source base operand to be 32B aligned, while the scalar effective address only
needs the natural alignment of its element type. The broadcast load preserves
the physical vreg shape; only lane 0 is semantically live for the one-slot
layout.

If `source_group_stride != 1`, this packed `slots = 8` layout requires a
strided/gather group-slot load materializer. Until that support exists,
`group_slot_load` with `slots = 8` and non-unit stride must diagnose instead of
silently using full-group `group_load`.

#### 3.16.2 Row-Local `group_slot_load`, `slots = 1`

VMI input:

```text
%c8 = arith.constant 8 : index
%rhs = pto.vmi.group_slot_load %rhs_base[%rhs_off], %c8 {num_groups = 8}
  : !pto.ptr<f32, ub>, index -> !pto.vmi.vreg<512xf32>
pto.vmi.group_store %rhs, %out[%group_off], %c8 {num_groups = 8}
```

Assigned layout:

```text
%rhs:
  !pto.vmi.vreg<512xf32, #pto.vmi.layout<num_groups = 8, slots = 1>>
```

VPTO lowering result:

```text
%one_b32 = pto.pge_b32 "PAT_VL1"

// Emit this shape for r = 0..7.  Each result value carries one semantic slot
// in lane 0, matching the S=64 row-local group_reduce result layout.
// For f32, source_group_stride = 8 elements = 32B, so every lane-0 vsldb is
// aligned.
%rhs_r = pto.vsldb %rhs_base[%rhs_off_plus_r], %c0_i16, %c0_i16, %one_b32
  : !pto.ptr<f32, ub>, i16, i16, !pto.mask<b32> -> !pto.vreg<64xf32>

pto.vsts %rhs_r, %out[%group_off_plus_r], %one_b32 {dist = "NORM_B32"}
  : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
```

Memory result:

```text
for r = 0..7:
  out[group_off + r * 8] = rhs_base[rhs_off + r * 8]
```

Current lowering rule:

```text
slots = 1 group_slot_load uses one lane-0 vsldb per semantic group slot.
For f32, source_group_stride must be a positive constant divisible by 8
elements.  For f16 it must be divisible by 16 elements, and for f8 it must be
divisible by 32 elements.
```

### 3.17 `group_broadcast` Feeding A Deinterleaved Consumer

This case fixes a lowering invariant: `group_broadcast` itself does not infer a
consumer-specific deinterleaved result.  It produces the layout selected by
layout assignment.  If a later consumer requires another layout, assignment must
insert an explicit `ensure_layout`.

The current endpoint is:

```text
group_reduce -> group_broadcast(contiguous f32)
             -> ensure_layout(deinterleaved = 2)
             -> truncf(contiguous f16)
```

VMI input:

```text
%x = pto.vmi.load %base[%off]
  : memref<128xf32> -> !pto.vmi.vreg<128xf32>
%sum = pto.vmi.group_reduce_addf %x, %mask {num_groups = 8}
%b = pto.vmi.group_broadcast %sum {num_groups = 8}
%h = pto.vmi.truncf %b
pto.vmi.store %h, %out[%off]
```

Assigned layouts:

```text
%x:
  !pto.vmi.vreg<128xf32,
    #pto.vmi.layout<block_deinterleaved=2>>

%sum:
  !pto.vmi.vreg<128xf32, #pto.vmi.layout<num_groups = 8, slots = 8>>

%b_dense:
  !pto.vmi.vreg<128xf32, #pto.vmi.layout<contiguous>>

%b_split = pto.vmi.ensure_layout %b_dense:
  #pto.vmi.layout<contiguous>
    -> #pto.vmi.layout<deinterleaved = 2>

%h:
  !pto.vmi.vreg<128xf16, #pto.vmi.layout<contiguous>>
```

VPTO lowering result:

```text
%all_b32 = pto.pge_b32 "PAT_ALL"
%sum_mask = pto.pge_b32 "PAT_VL8"

%x_lo, %x_hi = pto.vldsx2 %base[%tile_off], "BDINTLV"
  : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>, !pto.vreg<64xf32>

%lo_sum = pto.vcgadd %x_lo, %all_b32
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%hi_sum = pto.vcgadd %x_hi, %all_b32
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%sum_block = pto.vadd %lo_sum, %hi_sum, %sum_mask
  : !pto.vreg<64xf32>

// group_broadcast lowers to two contiguous f32 chunks.
%idx_lo = materialize indices [0 repeated 16, 1 repeated 16,
                               2 repeated 16, 3 repeated 16]
  : !pto.vreg<64xi32>
%idx_hi = materialize indices [4 repeated 16, 5 repeated 16,
                               6 repeated 16, 7 repeated 16]
  : !pto.vreg<64xi32>

%b_lo = pto.vselr %sum_block, %idx_lo
  : !pto.vreg<64xf32>, !pto.vreg<64xi32> -> !pto.vreg<64xf32>
%b_hi = pto.vselr %sum_block, %idx_hi
  : !pto.vreg<64xf32>, !pto.vreg<64xi32> -> !pto.vreg<64xf32>

// ensure_layout contiguous -> deinterleaved=2 is explicit in assigned VMI.
%b_even_input, %b_odd_input = pto.vdintlv %b_lo, %b_hi
  : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>, !pto.vreg<64xf32>

%h_even = pto.vcvt %b_even_input, %all_b32 {part = "EVEN", rnd = "R", sat = "SAT"}
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<128xf16>
%h_odd = pto.vcvt %b_odd_input, %all_b32 {part = "ODD", rnd = "R", sat = "SAT"}
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<128xf16>

%all_b16 = pto.pge_b16 "PAT_ALL"
%h0 = pto.vor %h_even, %h_odd, %all_b16
  : !pto.vreg<128xf16>

pto.vsts %h0, %out[%off], %all_b16 {dist = "NORM_B16"}
  : !pto.vreg<128xf16>, !pto.ptr<f16, ub>, !pto.mask<b16>
```

Memory result:

```text
for r = 0..7:
  s = reduce(row_r[0..15])
  out[r * 16 + 0 .. r * 16 + 15] = truncf(s)
```

Required assignment rule:

```text
`group_broadcast` layout is chosen before `vmi-to-vpto`.  A width-changing
consumer such as `truncf` may require a deinterleaved f32 source, but that
requirement must be represented by `ensure_layout`; `truncf` lowering must not
look through the defining `group_broadcast` and choose a hidden broadcast shape.
```

### 3.18 One Value With Dense And Group-Reduce Consumers

This case forces layout assignment to handle a solvable use-site conflict.  One
consumer requires an S=32 group-reduce layout; another consumer requires dense
row-major store.  This is not semantically illegal.  It must be solved by
explicit use-site materialization. A later optimization pass may fold the
materialization into a store or rematerialize a cheap producer when the required
support exists.

VMI input:

```text
%x = pto.vmi.load %base[%off]
  : memref<256xf32> -> !pto.vmi.vreg<256xf32>
%sum = pto.vmi.group_reduce_addf %x, %mask {num_groups = 8}
pto.vmi.group_store %sum, %sum_out[%group_off], %c1 {num_groups = 8}
pto.vmi.store %x, %copy_out[%off]
```

Assigned layouts:

```text
%x for group_reduce:
  !pto.vmi.vreg<256xf32, #pto.vmi.layout<deinterleaved = 4>>

%sum:
  !pto.vmi.vreg<256xf32, #pto.vmi.layout<num_groups = 8, slots = 8>>

%x for dense store:
  requires #pto.vmi.layout<contiguous>
```

Baseline layout assignment keeps `%x` in the group-reduce layout and inserts
`ensure_layout` before the dense store use.  A later rematerialization pass may
clone the load for the dense store if that is profitable.  A later fold-consumer
pass may also fold `ensure_layout + store` into a layout-aware store lowering.

VPTO lowering result:

```text
%x_even_0, %x_odd_0 = pto.vldsx2 %base[%tile_off_0], "DINTLV_B32"
  : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>, !pto.vreg<64xf32>
%x_even_1, %x_odd_1 = pto.vldsx2 %base[%tile_off_1], "DINTLV_B32"
  : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>, !pto.vreg<64xf32>

%x_p0, %x_p2 = pto.vdintlv %x_even_0, %x_even_1
  : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>, !pto.vreg<64xf32>
%x_p1, %x_p3 = pto.vdintlv %x_odd_0, %x_odd_1
  : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>, !pto.vreg<64xf32>

%all_b32 = pto.pge_b32 "PAT_ALL"
%sum_mask = pto.pge_b32 "PAT_VL8"

%s0 = pto.vcgadd %x_p0, %all_b32
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%s1 = pto.vcgadd %x_p1, %all_b32
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%s2 = pto.vcgadd %x_p2, %all_b32
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%s3 = pto.vcgadd %x_p3, %all_b32
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>

%s01 = pto.vadd %s0, %s1, %sum_mask : !pto.vreg<64xf32>
%s23 = pto.vadd %s2, %s3, %sum_mask : !pto.vreg<64xf32>
%sum_block = pto.vadd %s01, %s23, %sum_mask : !pto.vreg<64xf32>

pto.vsts %sum_block, %sum_out[%group_off], %sum_mask {dist = "NORM_B32"}
  : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>

// Dense store materialization for the second consumer.
%even0, %even1 = pto.vintlv %x_p0, %x_p2
  : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>, !pto.vreg<64xf32>
%odd0, %odd1 = pto.vintlv %x_p1, %x_p3
  : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>, !pto.vreg<64xf32>

%d0, %d1 = pto.vintlv %even0, %odd0
  : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>, !pto.vreg<64xf32>
%d2, %d3 = pto.vintlv %even1, %odd1
  : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>, !pto.vreg<64xf32>

pto.vsts %d0, %copy_out[%off_0], %all_b32 {dist = "NORM_B32"}
pto.vsts %d1, %copy_out[%off_64], %all_b32 {dist = "NORM_B32"}
pto.vsts %d2, %copy_out[%off_128], %all_b32 {dist = "NORM_B32"}
pto.vsts %d3, %copy_out[%off_192], %all_b32 {dist = "NORM_B32"}
```

Memory result:

```text
for r = 0..7:
  sum_out[group_off + r] = reduce(row_r[0..31])

for i = 0..255:
  copy_out[off + i] = base[off + i]
```

If `deinterleaved = 4 -> contiguous` materialization support does not exist, the
required diagnostic is:

```text
VMI-LAYOUT-CONTRACT:
  value %x is required as #pto.vmi.layout<deinterleaved = 4> by
  pto.vmi.group_reduce_addf and as #pto.vmi.layout<contiguous> by
  pto.vmi.store, but no materialization support exists at the store use site.
```

### 3.19 S=16 Reduce Split-Layout Support Selection

S=16 f32 group reduction has two legal dense input layouts:

```text
#pto.vmi.layout<deinterleaved=2>
#pto.vmi.layout<block_deinterleaved=2>
```

Ordinary `deinterleaved=2` is the element-parity layout required by f32->f16
`truncf`.
It is also a valid S=16 reduction layout: each physical part contains eight
values per row, so `VCGADD` can reduce each part and `VADD` can combine the two
partial sums.

`block_deinterleaved=2` is still useful when the producer is a block load shape
such as `BDINTLV` or `vsldb` over 32B row fragments. Baseline layout assignment must
express any mismatch with an explicit `ensure_layout`; producer rematerialization
or consumer folding can choose the cheaper equivalent form later.  Assignment
must not hard-code S=16 reduce to `block_deinterleaved=2`.

#### 3.19.1 Continuous S=16 Reduce And Truncf

VMI input:

```text
%x = pto.vmi.load %base[%off]
  : memref<128xf32> -> !pto.vmi.vreg<128xf32>
%sum = pto.vmi.group_reduce_addf %x, %mask {num_groups = 8}
pto.vmi.group_store %sum, %sum_out[%group_off], %c1 {num_groups = 8}
%h = pto.vmi.truncf %x
  : !pto.vmi.vreg<128xf32> -> !pto.vmi.vreg<128xf16>
pto.vmi.store %h, %out[%off]
```

Assigned layouts:

```text
%x:
  !pto.vmi.vreg<128xf32, #pto.vmi.layout<deinterleaved = 2>>

%sum:
  !pto.vmi.vreg<128xf32, #pto.vmi.layout<num_groups = 8, slots = 8>>

%h:
  !pto.vmi.vreg<128xf16, #pto.vmi.layout<contiguous>>
```

Physical lane map:

```text
%x_p0 lanes r*8 .. r*8+7 =
  row_r[0], row_r[2], row_r[4], ..., row_r[14]

%x_p1 lanes r*8 .. r*8+7 =
  row_r[1], row_r[3], row_r[5], ..., row_r[15]
```

VPTO lowering result:

```text
%all_b32 = pto.pge_b32 "PAT_ALL"
%sum_mask = pto.pge_b32 "PAT_VL8"

%x_p0, %x_p1 = pto.vldsx2 %base[%tile_off], "DINTLV_B32"
  : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>, !pto.vreg<64xf32>

%s0 = pto.vcgadd %x_p0, %all_b32
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%s1 = pto.vcgadd %x_p1, %all_b32
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%sum_block = pto.vadd %s0, %s1, %sum_mask
  : !pto.vreg<64xf32>

pto.vsts %sum_block, %sum_out[%group_off], %sum_mask {dist = "NORM_B32"}
  : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>

%h_even = pto.vcvt %x_p0, %all_b32 {part = "EVEN", rnd = "R", sat = "SAT"}
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<128xf16>
%h_odd = pto.vcvt %x_p1, %all_b32 {part = "ODD", rnd = "R", sat = "SAT"}
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<128xf16>

%all_b16 = pto.pge_b16 "PAT_ALL"
%h0 = pto.vor %h_even, %h_odd, %all_b16
  : !pto.vreg<128xf16>
pto.vsts %h0, %out[%off], %all_b16 {dist = "NORM_B16"}
  : !pto.vreg<128xf16>, !pto.ptr<f16, ub>, !pto.mask<b16>
```

Memory result:

```text
for r = 0..7:
  sum_out[group_off + r] = reduce(row_r[0..15])

for i = 0..127:
  out[off + i] = truncf(base[off + i])
```

#### 3.19.2 Block-Load Producer Fixed To `block_deinterleaved=2`

This is the real conflict case. The value is fixed to `block_deinterleaved=2`
because the producer uses block-load support. A later `truncf` requires
ordinary element-parity `deinterleaved=2`.

VMI input:

```text
%stride24 = arith.constant 24 : index
%x = pto.vmi.group_load %base[%off], %stride24
  {num_groups = 8, group_size = 16}
  : !pto.ptr<f32, ub>, index -> !pto.vmi.vreg<128xf32>
%sum = pto.vmi.group_reduce_addf %x, %mask {num_groups = 8}
pto.vmi.group_store %sum, %sum_out[%group_off], %c1 {num_groups = 8}
%h = pto.vmi.truncf %x
  : !pto.vmi.vreg<128xf32> -> !pto.vmi.vreg<128xf16>
pto.vmi.store %h, %out[%off]
```

Assigned layouts before the conflicting `truncf` use:

```text
%x from strided block group_load:
  !pto.vmi.vreg<128xf32,
    #pto.vmi.layout<block_deinterleaved=2>>

%sum:
  !pto.vmi.vreg<128xf32, #pto.vmi.layout<num_groups = 8, slots = 8>>
```

The reduction path is legal and uses the same `vsldb` block-load shape as
section 3.15.2.  The `truncf` path is legal only if one of these transforms
exists:

```text
1. rematerialize the original memory producer as deinterleaved=2
2. materialize block_deinterleaved=2 -> deinterleaved=2 in registers
3. use an explicitly enabled scratch/reload fallback
```

If no such transform exists, the required diagnostic is:

```text
VMI-LAYOUT-CONTRACT:
  pto.vmi.truncf requires
  #pto.vmi.layout<deinterleaved=2>, but the source value is
  fixed to #pto.vmi.layout<block_deinterleaved=2> by the strided
  group_load. Add rematerialization or preserving materialization support, or
  avoid consuming this block-loaded value with truncf.
```

### 3.20 `group_slots` Control-Flow Join

`group_slots` values must be allowed to cross control flow.  The join type is a
group-slot physical tuple, not a dense vector.

VMI input:

```text
%sum = scf.if %cond -> !pto.vmi.vreg<128xf32> {
  %x = pto.vmi.load %base[%off]
    : memref<128xf32> -> !pto.vmi.vreg<128xf32>
  %a = pto.vmi.group_reduce_addf %x, %mask {num_groups = 8}
  scf.yield %a : !pto.vmi.vreg<128xf32>
} else {
  %b = pto.vmi.group_slot_load %rhs_base[%rhs_off], %c1 {num_groups = 8}
    : !pto.ptr<f32, ub>, index -> !pto.vmi.vreg<128xf32>
  scf.yield %b : !pto.vmi.vreg<128xf32>
}
%bias = pto.vmi.group_slot_load %bias_base[%bias_off], %c1 {num_groups = 8}
  : !pto.ptr<f32, ub>, index -> !pto.vmi.vreg<128xf32>
%outv = pto.vmi.vadd %sum, %bias
pto.vmi.group_store %outv, %out[%group_off], %c1 {num_groups = 8}
```

Assigned layouts:

```text
%x:
  !pto.vmi.vreg<128xf32,
    #pto.vmi.layout<block_deinterleaved=2>>

%a, %b, %sum, %bias, %outv:
  !pto.vmi.vreg<128xf32, #pto.vmi.layout<num_groups = 8, slots = 8>>
```

VPTO lowering result for the join:

```text
%sum_block = scf.if %cond -> !pto.vreg<64xf32> {
  %all_b32 = pto.pge_b32 "PAT_ALL"
  %sum_mask = pto.pge_b32 "PAT_VL8"

  %x_lo, %x_hi = pto.vldsx2 %base[%tile_off], "BDINTLV"
    : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>, !pto.vreg<64xf32>
  %lo_sum = pto.vcgadd %x_lo, %all_b32
    : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
  %hi_sum = pto.vcgadd %x_hi, %all_b32
    : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
  %a_block = pto.vadd %lo_sum, %hi_sum, %sum_mask
    : !pto.vreg<64xf32>
  scf.yield %a_block : !pto.vreg<64xf32>
} else {
  %one_block = pto.pge_b32 "PAT_VL1"
  %b_block = pto.vsldb %rhs_base[%rhs_off], %c0_i16, %c0_i16, %one_block
    : !pto.ptr<f32, ub>, i16, i16, !pto.mask<b32> -> !pto.vreg<64xf32>
  scf.yield %b_block : !pto.vreg<64xf32>
}

%one_block = pto.pge_b32 "PAT_VL1"
%slot_mask = pto.pge_b32 "PAT_VL8"
%bias_block = pto.vsldb %bias_base[%bias_off], %c0_i16, %c0_i16, %one_block
  : !pto.ptr<f32, ub>, i16, i16, !pto.mask<b32> -> !pto.vreg<64xf32>
%out_block = pto.vadd %sum_block, %bias_block, %slot_mask
  : !pto.vreg<64xf32>

pto.vsts %out_block, %out[%group_off], %slot_mask {dist = "NORM_B32"}
  : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
```

Memory result:

```text
for r = 0..7:
  lhs = cond ? reduce(row_r[0..15]) : rhs_base[rhs_off + r]
  out[group_off + r] = lhs + bias_base[bias_off + r]
```

### 3.21 S=32 Tail With Full-Tile-Readable Source

This is the positive counterpart to section 3.11.2.  Tail participation is
still expressed by masks, but the source must provide a static proof that
reading the rounded-up 8-row physical tile is memory-safe.  That proof is
explicit for partial logical loads: it can come from a statically shaped memref
source.  Pointer-source runtime kernels should instead load the rounded physical
vector and use a mask to express active logical lanes; this is not inferred from
surrounding MTE copies or caller context.

VMI input:

```text
%x = pto.vmi.load %base[%off]
  : memref<256xf32> -> !pto.vmi.vreg<192xf32>
%mask = pto.vmi.create_mask %c192 : index -> !pto.vmi.mask<192xpred>
%sum = pto.vmi.group_reduce_addf %x, %mask {num_groups = 6}
pto.vmi.group_store %sum, %out[%group_off], %c1 {num_groups = 6}
```

Equivalent pointer-source VMI input for runtime kernels:

```text
%x = pto.vmi.load %base[%off]
  : !pto.ptr<f32, ub> -> !pto.vmi.vreg<256xf32>
%mask = pto.vmi.create_mask %c192 : index -> !pto.vmi.mask<256xpred>
```

Assigned layouts:

```text
%x:
  !pto.vmi.vreg<192xf32, #pto.vmi.layout<block_deinterleaved=4>>

%mask:
  !pto.vmi.mask<192xpred,
    #pto.vmi.layout<block_deinterleaved=4>>

%sum:
  !pto.vmi.vreg<192xf32, #pto.vmi.layout<num_groups = 6, slots = 8>>
```

VPTO lowering result:

```text
// A statically safe full-read proof allows the load plan to read the
// rounded-up 8-row tile.  Only rows 0..5 are semantically active.
%x_c0 = pto.vlds %base[%tile_off_0]
  : memref<256xf32> -> !pto.vreg<64xf32>
%x_c1 = pto.vlds %base[%tile_off_1]
  : memref<256xf32> -> !pto.vreg<64xf32>
%x_c2 = pto.vlds %base[%tile_off_2]
  : memref<256xf32> -> !pto.vreg<64xf32>
%x_c3 = pto.vlds %base[%tile_off_3]
  : memref<256xf32> -> !pto.vreg<64xf32>

%x_lo01, %x_hi01 = pto.vdintlv %x_c0, %x_c1
  : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>, !pto.vreg<64xf32>
%x_lo23, %x_hi23 = pto.vdintlv %x_c2, %x_c3
  : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>, !pto.vreg<64xf32>
%x_p0, %x_p2 = pto.vdintlv %x_lo01, %x_lo23
  : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>, !pto.vreg<64xf32>
%x_p1, %x_p3 = pto.vdintlv %x_hi01, %x_hi23
  : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>, !pto.vreg<64xf32>

%data_mask0, %_ = pto.plt_b32 %c48_i32
  : i32 -> !pto.mask<b32>, i32
%data_mask1, %_ = pto.plt_b32 %c48_i32
  : i32 -> !pto.mask<b32>, i32
%data_mask2, %_ = pto.plt_b32 %c48_i32
  : i32 -> !pto.mask<b32>, i32
%data_mask3, %_ = pto.plt_b32 %c48_i32
  : i32 -> !pto.mask<b32>, i32
%sum_mask, %_ = pto.plt_b32 %c6_i32
  : i32 -> !pto.mask<b32>, i32

%s0 = pto.vcgadd %x_p0, %data_mask0
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%s1 = pto.vcgadd %x_p1, %data_mask1
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%s2 = pto.vcgadd %x_p2, %data_mask2
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%s3 = pto.vcgadd %x_p3, %data_mask3
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>

%s01 = pto.vadd %s0, %s1, %sum_mask : !pto.vreg<64xf32>
%s23 = pto.vadd %s2, %s3, %sum_mask : !pto.vreg<64xf32>
%sum_block = pto.vadd %s01, %s23, %sum_mask : !pto.vreg<64xf32>

pto.vsts %sum_block, %out[%group_off], %sum_mask {dist = "NORM_B32"}
  : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
```

Memory result:

```text
for r = 0..5:
  out[group_off + r] = reduce(row_r[0..31])
```

Rows 6 and 7 may be physically loaded because of the safe full-read proof, but
their lanes are not active in `%data_mask*`, and their group slots are not
stored because `%sum_mask` is produced by `plt_b32 %c6_i32`.

### 3.22 `scf.for` Loop-Carried Layout

Loop-carried VMI values require a layout fixed point.  The iter_arg, body block
argument, yield operand, loop result, and later consumer must all agree on one
layout, or `vmi-layout-assignment` must insert a materialization at a legal
dominating use site.

VMI input:

```text
%init = pto.vmi.load %base[%off]
  : memref<256xf32> -> !pto.vmi.vreg<256xf32>
%acc = scf.for %i = %c0 to %steps step %c1
    iter_args(%arg = %init) -> !pto.vmi.vreg<256xf32> {
  %bias = pto.vmi.broadcast %bias_s
    : f32 -> !pto.vmi.vreg<256xf32>
  %next = pto.vmi.vadd %arg, %bias
  scf.yield %next : !pto.vmi.vreg<256xf32>
}
%sum = pto.vmi.group_reduce_addf %acc, %mask {num_groups = 8}
pto.vmi.group_store %sum, %out[%group_off], %c1 {num_groups = 8}
```

Assigned layouts:

```text
%init, %arg, %bias, %next, %acc:
  !pto.vmi.vreg<256xf32, #pto.vmi.layout<deinterleaved = 4>>

%sum:
  !pto.vmi.vreg<256xf32, #pto.vmi.layout<num_groups = 8, slots = 8>>
```

VPTO lowering result:

```text
%init_even_0, %init_odd_0 = pto.vldsx2 %base[%tile_off_0], "DINTLV_B32"
  : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>, !pto.vreg<64xf32>
%init_even_1, %init_odd_1 = pto.vldsx2 %base[%tile_off_1], "DINTLV_B32"
  : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>, !pto.vreg<64xf32>

%init_p0, %init_p2 = pto.vdintlv %init_even_0, %init_even_1
  : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>, !pto.vreg<64xf32>
%init_p1, %init_p3 = pto.vdintlv %init_odd_0, %init_odd_1
  : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>, !pto.vreg<64xf32>

%acc_p0, %acc_p1, %acc_p2, %acc_p3 =
  scf.for %i = %c0 to %steps step %c1
      iter_args(%arg_p0 = %init_p0, %arg_p1 = %init_p1,
                %arg_p2 = %init_p2, %arg_p3 = %init_p3)
      -> (!pto.vreg<64xf32>, !pto.vreg<64xf32>,
          !pto.vreg<64xf32>, !pto.vreg<64xf32>) {
    %all_b32 = pto.pge_b32 "PAT_ALL"
    %bias_p0 = pto.vdup %bias_s, %all_b32
      : f32, !pto.mask<b32> -> !pto.vreg<64xf32>
    %bias_p1 = pto.vdup %bias_s, %all_b32
      : f32, !pto.mask<b32> -> !pto.vreg<64xf32>
    %bias_p2 = pto.vdup %bias_s, %all_b32
      : f32, !pto.mask<b32> -> !pto.vreg<64xf32>
    %bias_p3 = pto.vdup %bias_s, %all_b32
      : f32, !pto.mask<b32> -> !pto.vreg<64xf32>

    %next_p0 = pto.vadd %arg_p0, %bias_p0, %all_b32 : !pto.vreg<64xf32>
    %next_p1 = pto.vadd %arg_p1, %bias_p1, %all_b32 : !pto.vreg<64xf32>
    %next_p2 = pto.vadd %arg_p2, %bias_p2, %all_b32 : !pto.vreg<64xf32>
    %next_p3 = pto.vadd %arg_p3, %bias_p3, %all_b32 : !pto.vreg<64xf32>
    scf.yield %next_p0, %next_p1, %next_p2, %next_p3
      : !pto.vreg<64xf32>, !pto.vreg<64xf32>,
        !pto.vreg<64xf32>, !pto.vreg<64xf32>
  }

%all_b32 = pto.pge_b32 "PAT_ALL"
%sum_mask = pto.pge_b32 "PAT_VL8"
%s0 = pto.vcgadd %acc_p0, %all_b32 : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%s1 = pto.vcgadd %acc_p1, %all_b32 : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%s2 = pto.vcgadd %acc_p2, %all_b32 : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%s3 = pto.vcgadd %acc_p3, %all_b32 : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%s01 = pto.vadd %s0, %s1, %sum_mask : !pto.vreg<64xf32>
%s23 = pto.vadd %s2, %s3, %sum_mask : !pto.vreg<64xf32>
%sum_block = pto.vadd %s01, %s23, %sum_mask : !pto.vreg<64xf32>
pto.vsts %sum_block, %out[%group_off], %sum_mask {dist = "NORM_B32"}
  : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
```

Memory result:

```text
for r = 0..7:
  for c = 0..31:
    acc[row_r, c] = base[row_r, c] + steps * bias_s
  out[group_off + r] = reduce(acc[row_r, 0..31])
```

### 3.23 `group_broadcast` With Multiple Dense Consumers

One `group_slots` value may feed multiple `group_broadcast` uses with different
dense result layout requirements.  Each `group_broadcast` op has its own result
layout, so layout assignment should type each op at its use site instead of
forcing one result layout onto all consumers.

VMI input:

```text
%x = pto.vmi.load %base[%off]
  : memref<128xf32> -> !pto.vmi.vreg<128xf32>
%sum = pto.vmi.group_reduce_addf %x, %mask {num_groups = 8}

%b_for_mul = pto.vmi.group_broadcast %sum {num_groups = 8}
%y = pto.vmi.vmul %x, %b_for_mul
%ysum = pto.vmi.group_reduce_addf %y, %mask {num_groups = 8}
pto.vmi.group_store %ysum, %sum_out[%group_off], %c1 {num_groups = 8}

%b_for_cast = pto.vmi.group_broadcast %sum {num_groups = 8}
%h = pto.vmi.truncf %b_for_cast
pto.vmi.store %h, %dense_out[%off]
```

Assigned layouts in the current implementation:

```text
%x:
  !pto.vmi.vreg<128xf32, #pto.vmi.layout<contiguous>>

%x_for_reduce:
  !pto.vmi.vreg<128xf32,
    #pto.vmi.layout<block_deinterleaved=2>>

%sum, %ysum:
  !pto.vmi.vreg<128xf32, #pto.vmi.layout<num_groups = 8, slots = 8>>

%b_for_mul, %y:
  !pto.vmi.vreg<128xf32, #pto.vmi.layout<contiguous>>

%y_for_reduce:
  !pto.vmi.vreg<128xf32,
    #pto.vmi.layout<block_deinterleaved=2>>

%b_for_cast:
  !pto.vmi.vreg<128xf32, #pto.vmi.layout<contiguous>>

%b_for_cast_split:
  !pto.vmi.vreg<128xf32, #pto.vmi.layout<deinterleaved = 2>>

%h:
  !pto.vmi.vreg<128xf16, #pto.vmi.layout<contiguous>>
```

The important invariant is not that both dense consumers choose the same dense
layout.  It is that each use has an explicit layout boundary:

```text
%x_for_reduce = pto.vmi.ensure_layout %x
%y_for_reduce = pto.vmi.ensure_layout %y
%b_for_cast_split = pto.vmi.ensure_layout %b_for_cast
```

If a future direct `group_broadcast -> deinterleaved` support path is added, layout
assignment may assign `%b_for_mul` or `%b_for_cast` directly to that layout, but
the choice must still be visible in the assigned IR.

VPTO lowering result:

```text
%all_b32 = pto.pge_b32 "PAT_ALL"
%sum_mask = pto.pge_b32 "PAT_VL8"

%x_lo, %x_hi = pto.vldsx2 %base[%tile_off], "BDINTLV"
  : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>, !pto.vreg<64xf32>
%x_lo_sum = pto.vcgadd %x_lo, %all_b32
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%x_hi_sum = pto.vcgadd %x_hi, %all_b32
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%sum_block = pto.vadd %x_lo_sum, %x_hi_sum, %sum_mask
  : !pto.vreg<64xf32>

%lane_id = pto.vci %c0_i32 : i32 -> !pto.vreg<64xi32>
%broadcast_idx = pto.vshrs %lane_id, %c3_i16, %all_b32
  : !pto.vreg<64xi32>, i16, !pto.mask<b32> -> !pto.vreg<64xi32>

// Use 1: broadcast for the multiply path.  Current lowering materializes two
// contiguous f32 chunks, multiplies them with the original contiguous chunks,
// then deinterleaves the product for the second group_reduce.
%b_rows_for_mul_0 = pto.vselr %sum_block, %broadcast_idx_0
  : !pto.vreg<64xf32>, !pto.vreg<64xi32> -> !pto.vreg<64xf32>
%b_rows_for_mul_1 = pto.vselr %sum_block, %broadcast_idx_1
  : !pto.vreg<64xf32>, !pto.vreg<64xi32> -> !pto.vreg<64xf32>
%y0 = pto.vmul %x0, %b_rows_for_mul_0, %all_b32 : !pto.vreg<64xf32>
%y1 = pto.vmul %x1, %b_rows_for_mul_1, %all_b32 : !pto.vreg<64xf32>
%y_lo, %y_hi = pto.vdintlv %y0, %y1
  : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>, !pto.vreg<64xf32>
%y_lo_sum = pto.vcgadd %y_lo, %all_b32
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%y_hi_sum = pto.vcgadd %y_hi, %all_b32
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%ysum_block = pto.vadd %y_lo_sum, %y_hi_sum, %sum_mask
  : !pto.vreg<64xf32>
pto.vsts %ysum_block, %sum_out[%group_off], %sum_mask {dist = "NORM_B32"}
  : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>

// Use 2: rematerialize broadcast for the f32->f16 parity cast path.
%b_rows_for_cast_0 = pto.vselr %sum_block, %broadcast_idx_0
  : !pto.vreg<64xf32>, !pto.vreg<64xi32> -> !pto.vreg<64xf32>
%b_rows_for_cast_1 = pto.vselr %sum_block, %broadcast_idx_1
  : !pto.vreg<64xf32>, !pto.vreg<64xi32> -> !pto.vreg<64xf32>
%cast_lo, %cast_hi = pto.vdintlv %b_rows_for_cast_0, %b_rows_for_cast_1
  : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>, !pto.vreg<64xf32>
%h_even = pto.vcvt %cast_lo, %all_b32
  {part = "EVEN", rnd = "R", sat = "SAT"}
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<128xf16>
%h_odd = pto.vcvt %cast_hi, %all_b32
  {part = "ODD", rnd = "R", sat = "SAT"}
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<128xf16>
%all_b16 = pto.pge_b16 "PAT_ALL"
%h0 = pto.vor %h_even, %h_odd, %all_b16 : !pto.vreg<128xf16>
pto.vsts %h0, %dense_out[%off], %all_b16 {dist = "NORM_B16"}
  : !pto.vreg<128xf16>, !pto.ptr<f16, ub>, !pto.mask<b16>
```

Memory result:

```text
for r = 0..7:
  s = reduce(row_r[0..15])
  sum_out[group_off + r] = reduce_i(row_r[i] * s for i = 0..15)
  dense_out[r * 16 + 0 .. r * 16 + 15] = truncf(s)
```

### 3.24 Mask With Elementwise, Select, And Store

This case separates compute masking from memory effects.  A masked elementwise
operation with passthrough semantics can be represented as ordinary compute
plus `select`; a masked store uses the mask only on the store effect.

VMI input:

```text
%x = pto.vmi.load %base[%off]
  : memref<64xf32> -> !pto.vmi.vreg<64xf32>
%rhs = pto.vmi.load %rhs_base[%off]
  : memref<64xf32> -> !pto.vmi.vreg<64xf32>
%mask = pto.vmi.create_mask %c48
  : index -> !pto.vmi.mask<64xpred>
%sum = pto.vmi.vadd %x, %rhs
%passthrough = pto.vmi.select %mask, %sum, %x
pto.vmi.store %passthrough, %dense_out[%off]
pto.vmi.masked_store %sum, %masked_out[%off], %mask
```

Assigned layouts:

```text
%x, %rhs, %sum, %passthrough:
  !pto.vmi.vreg<64xf32, #pto.vmi.layout<contiguous>>

%mask:
  !pto.vmi.mask<64xpred, #pto.vmi.layout<contiguous>>
```

VPTO lowering result:

```text
%all_b32 = pto.pge_b32 "PAT_ALL"
%m, %_ = pto.plt_b32 %c48_i32 : i32 -> !pto.mask<b32>, i32

%x0 = pto.vlds %base[%off] {dist = "NORM"}
  : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
%rhs0 = pto.vlds %rhs_base[%off] {dist = "NORM"}
  : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
%sum0 = pto.vadd %x0, %rhs0, %all_b32 : !pto.vreg<64xf32>

%pass0 = pto.vsel %sum0, %x0, %m
  : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
pto.vsts %pass0, %dense_out[%off], %all_b32 {dist = "NORM_B32"}
  : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>

pto.vsts %sum0, %masked_out[%off], %m {dist = "NORM_B32"}
  : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
```

Memory result:

```text
for i = 0..63:
  if i < 48:
    dense_out[off + i] = base[off + i] + rhs_base[off + i]
    masked_out[off + i] = base[off + i] + rhs_base[off + i]
  else:
    dense_out[off + i] = base[off + i]
    masked_out[off + i] is unchanged
```

### 3.25 Function Boundary Layout Specialization

Function boundaries cannot rely on hidden layout side tables.  Either the
function is internal and layout-specialized by `vmi-layout-assignment`, or a
public/external VMI boundary must diagnose until a stable VMI ABI is defined.

#### 3.25.1 Internal Function Specialized To Consumer Layout

VMI input:

```text
func.func private @producer(%base: !pto.ptr<f32, ub>, %off: index)
    -> !pto.vmi.vreg<256xf32> {
  %x = pto.vmi.load %base[%off]
    : memref<256xf32> -> !pto.vmi.vreg<256xf32>
  return %x : !pto.vmi.vreg<256xf32>
}

func.func @caller(%base: !pto.ptr<f32, ub>, %off: index, %out: !pto.ptr<f32, ub>) {
  %x = call @producer(%base, %off)
    : (!pto.ptr<f32, ub>, index) -> !pto.vmi.vreg<256xf32>
  %sum = pto.vmi.group_reduce_addf %x, %mask {num_groups = 8}
  pto.vmi.group_store %sum, %out[%off], %c1 {num_groups = 8}
  return
}
```

Assigned layouts:

```text
@producer result:
  !pto.vmi.vreg<256xf32, #pto.vmi.layout<deinterleaved = 4>>

%x in @caller:
  !pto.vmi.vreg<256xf32, #pto.vmi.layout<deinterleaved = 4>>

%sum:
  !pto.vmi.vreg<256xf32, #pto.vmi.layout<num_groups = 8, slots = 8>>
```

VPTO lowering result for the function boundary:

```text
func.func private @producer(...)
    -> (!pto.vreg<64xf32>, !pto.vreg<64xf32>,
        !pto.vreg<64xf32>, !pto.vreg<64xf32>) {
  %x_even_0, %x_odd_0 = pto.vldsx2 %base[%tile_off_0], "DINTLV_B32"
    : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>, !pto.vreg<64xf32>
  %x_even_1, %x_odd_1 = pto.vldsx2 %base[%tile_off_1], "DINTLV_B32"
    : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>, !pto.vreg<64xf32>
  %x_p0, %x_p2 = pto.vdintlv %x_even_0, %x_even_1
    : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>, !pto.vreg<64xf32>
  %x_p1, %x_p3 = pto.vdintlv %x_odd_0, %x_odd_1
    : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>, !pto.vreg<64xf32>
  return %x_p0, %x_p1, %x_p2, %x_p3
    : !pto.vreg<64xf32>, !pto.vreg<64xf32>,
      !pto.vreg<64xf32>, !pto.vreg<64xf32>
}

func.func @caller(...) {
  %x_p0, %x_p1, %x_p2, %x_p3 = call @producer(...)
    : (...) -> (!pto.vreg<64xf32>, !pto.vreg<64xf32>,
                !pto.vreg<64xf32>, !pto.vreg<64xf32>)

  %all_b32 = pto.pge_b32 "PAT_ALL"
  %sum_mask = pto.pge_b32 "PAT_VL8"
  %s0 = pto.vcgadd %x_p0, %all_b32
    : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
  %s1 = pto.vcgadd %x_p1, %all_b32
    : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
  %s2 = pto.vcgadd %x_p2, %all_b32
    : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
  %s3 = pto.vcgadd %x_p3, %all_b32
    : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
  %s01 = pto.vadd %s0, %s1, %sum_mask : !pto.vreg<64xf32>
  %s23 = pto.vadd %s2, %s3, %sum_mask : !pto.vreg<64xf32>
  %sum_block = pto.vadd %s01, %s23, %sum_mask : !pto.vreg<64xf32>
  pto.vsts %sum_block, %out[%group_off], %sum_mask {dist = "NORM_B32"}
    : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
}
```

Memory result:

```text
for r = 0..7:
  out[off + r] = reduce(row_r[0..31])
```

Runtime closure:

```text
lit:
  test/lit/vmi_new/vmi_ptoas_private_call_inline.pto

runtime SIM:
  test/vpto/cases/vmi_new/private-call-inline-store

ptoas pipeline:
  vmi-layout-assignment makes the private result layout explicit
  vmi-to-vpto physicalizes the private helper result into !pto.vreg values
  ptoas then inlines private physical VMI helpers before VPTO vecscope/backend
  emission, so physical vector values do not escape through a function return
```

#### 3.25.2 Public Or External VMI Boundary

VMI input:

```text
func.func @public_producer(%base: !pto.ptr<f32, ub>, %off: index)
    -> !pto.vmi.vreg<256xf32> attributes {public} {
  %x = pto.vmi.load %base[%off]
    : memref<256xf32> -> !pto.vmi.vreg<256xf32>
  return %x : !pto.vmi.vreg<256xf32>
}
```

Required diagnostic for the initial design:

```text
VMI-LAYOUT-CONTRACT:
  public or external function boundary returns !pto.vmi.vreg<256xf32> without a
  stable VMI layout ABI. Mark the function internal for layout specialization,
  inline it before vmi-layout-assignment, or define an explicit ABI layout.
```

### 3.26 S=16 Grouped Tail Through Broadcast, Reduce, Store

This case extends section 3.15.1 from `reduce -> group_store` to the full
grouped compute path.  It is needed because `create_group_mask` must remain a
group-periodic mask after a `group_broadcast`; it cannot collapse to a prefix
mask or an all-true mask.

VMI input:

```text
%stride16 = arith.constant 16 : index
%x = pto.vmi.group_load %base[%off], %stride16
  {num_groups = 8, group_size = 16}
  : !pto.ptr<f32, ub>, index -> !pto.vmi.vreg<128xf32>
%c12 = arith.constant 12 : index
%mask = pto.vmi.create_group_mask %c12 {num_groups = 8, group_size = 16}
  : index -> !pto.vmi.mask<128xpred>
%sum = pto.vmi.group_reduce_addf %x, %mask {num_groups = 8}
%b = pto.vmi.group_broadcast %sum {num_groups = 8}
%y = pto.vmi.vmul %x, %b
%ysum = pto.vmi.group_reduce_addf %y, %mask {num_groups = 8}
pto.vmi.group_store %ysum, %out[%group_off], %c1 {num_groups = 8}
```

Assigned layouts:

```text
%x, %b, %y:
  !pto.vmi.vreg<128xf32,
    #pto.vmi.layout<block_deinterleaved=2>>

%mask:
  !pto.vmi.mask<128xpred,
    #pto.vmi.layout<block_deinterleaved=2>>

%sum, %ysum:
  !pto.vmi.vreg<128xf32, #pto.vmi.layout<num_groups = 8, slots = 8>>
```

VPTO lowering result for one `8x16xf32` tile:

```text
%all_b32 = pto.pge_b32 "PAT_ALL"
%sum_mask = pto.pge_b32 "PAT_VL8"

%lane = pto.vci %c0_i32 : i32 -> !pto.vreg<64xi32>
%row = pto.vshrs %lane, %c3_i16, %all_b32
  : !pto.vreg<64xi32>, i16, !pto.mask<b32> -> !pto.vreg<64xi32>
%row8 = pto.vshls %row, %c3_i16, %all_b32
  : !pto.vreg<64xi32>, i16, !pto.mask<b32> -> !pto.vreg<64xi32>
%col = pto.vsub %lane, %row8, %all_b32
  : !pto.vreg<64xi32>
%hi4_mask = pto.vcmps %col, %c4_i32, %all_b32, "lt"
  : !pto.vreg<64xi32>, i32, !pto.mask<b32> -> !pto.mask<b32>

%x_lo, %x_hi = pto.vldsx2 %base[%tile_off], "BDINTLV"
  : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>, !pto.vreg<64xf32>

%x_lo_sum = pto.vcgadd %x_lo, %all_b32
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%x_hi_sum = pto.vcgadd %x_hi, %hi4_mask
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%sum_block = pto.vadd %x_lo_sum, %x_hi_sum, %sum_mask
  : !pto.vreg<64xf32>

%broadcast_idx = pto.vshrs %lane, %c3_i16, %all_b32
  : !pto.vreg<64xi32>, i16, !pto.mask<b32> -> !pto.vreg<64xi32>
%b_rows = pto.vselr %sum_block, %broadcast_idx
  : !pto.vreg<64xf32>, !pto.vreg<64xi32> -> !pto.vreg<64xf32>

%y_lo = pto.vmul %x_lo, %b_rows, %all_b32 : !pto.vreg<64xf32>
%y_hi = pto.vmul %x_hi, %b_rows, %hi4_mask : !pto.vreg<64xf32>

%y_lo_sum = pto.vcgadd %y_lo, %all_b32
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%y_hi_sum = pto.vcgadd %y_hi, %hi4_mask
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%ysum_block = pto.vadd %y_lo_sum, %y_hi_sum, %sum_mask
  : !pto.vreg<64xf32>

pto.vsts %ysum_block, %out[%group_tile_off], %sum_mask {dist = "NORM_B32"}
  : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
```

Memory result:

```text
for r = 0..7:
  s = reduce(row_r[0..11])
  out[group_tile_off + r] =
      reduce_i(row_r[i] * s for i = 0..11)
    = s * s
```

Required assignment rule:

```text
%mask is a grouped mask with S=16 and active_elems_per_group=12.
For the low half, the physical predicate is PAT_ALL.
For the high half, the physical predicate is lane_mod_8 < 4.
The same split must be reused for both group_reduce operations.
```

### 3.27 S=32 `group_load` With Stride Greater Than Group Size

This case is the S=32 counterpart to section 3.15.2.  The logical group is
`32xf32`, but rows in memory have a larger stride.  The fast plan is legal only
when the stride is a multiple of one 32B f32 block.

VMI input:

```text
%stride40 = arith.constant 40 : index
%x = pto.vmi.group_load %base[%off], %stride40
  {num_groups = 8, group_size = 32}
  : !pto.ptr<f32, ub>, index -> !pto.vmi.vreg<256xf32>
%mask = pto.vmi.create_group_mask %c32 {num_groups = 8, group_size = 32}
  : index -> !pto.vmi.mask<256xpred>
%sum = pto.vmi.group_reduce_addf %x, %mask {num_groups = 8}
pto.vmi.group_store %sum, %out[%group_off], %c1 {num_groups = 8}
```

Assigned layouts:

```text
%x:
  !pto.vmi.vreg<256xf32,
    #pto.vmi.layout<block_deinterleaved=4>>

%mask:
  !pto.vmi.mask<256xpred,
    #pto.vmi.layout<block_deinterleaved=4>>

%sum:
  !pto.vmi.vreg<256xf32, #pto.vmi.layout<num_groups = 8, slots = 8>>
```

VPTO lowering result:

```text
%all_b32 = pto.pge_b32 "PAT_ALL"
%sum_mask = pto.pge_b32 "PAT_VL8"

// source_group_stride = 40 f32 = 5 * 32B blocks.
%stride_blocks = %c5_i16

%frag0 = pto.vsldb %base_frag0, %stride_blocks, %c0_i16, %all_b32
  : !pto.ptr<f32, ub>, i16, i16, !pto.mask<b32> -> !pto.vreg<64xf32>
%frag1 = pto.vsldb %base_frag1, %stride_blocks, %c0_i16, %all_b32
  : !pto.ptr<f32, ub>, i16, i16, !pto.mask<b32> -> !pto.vreg<64xf32>
%frag2 = pto.vsldb %base_frag2, %stride_blocks, %c0_i16, %all_b32
  : !pto.ptr<f32, ub>, i16, i16, !pto.mask<b32> -> !pto.vreg<64xf32>
%frag3 = pto.vsldb %base_frag3, %stride_blocks, %c0_i16, %all_b32
  : !pto.ptr<f32, ub>, i16, i16, !pto.mask<b32> -> !pto.vreg<64xf32>

%frag0 lanes r*8 .. r*8+7 = row_r[0..7]
%frag1 lanes r*8 .. r*8+7 = row_r[8..15]
%frag2 lanes r*8 .. r*8+7 = row_r[16..23]
%frag3 lanes r*8 .. r*8+7 = row_r[24..31]

%s0 = pto.vcgadd %frag0, %all_b32
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%s1 = pto.vcgadd %frag1, %all_b32
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%s2 = pto.vcgadd %frag2, %all_b32
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%s3 = pto.vcgadd %frag3, %all_b32
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>

%s01 = pto.vadd %s0, %s1, %sum_mask : !pto.vreg<64xf32>
%s23 = pto.vadd %s2, %s3, %sum_mask : !pto.vreg<64xf32>
%sum_block = pto.vadd %s01, %s23, %sum_mask : !pto.vreg<64xf32>

pto.vsts %sum_block, %out[%group_tile_off], %sum_mask {dist = "NORM_B32"}
  : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
```

Memory result:

```text
for r = 0..7:
  out[group_tile_off + r] =
    reduce(base[tile_off + r * 40 + 0 .. tile_off + r * 40 + 31])
```

Required diagnostic when the stride is not block-aligned:

```text
VMI-LAYOUT-CONTRACT:
  pto.vmi.group_load group_size 32 with source_group_stride not divisible by
  8 f32 elements cannot use the vsldb strided-block lowering support. Enable a
  stable gather fallback or choose a block-aligned source_group_stride.
```

Required assignment rule:

```text
This producer requires the S=32 block-fragment layout:
  #pto.vmi.layout<block_deinterleaved=4>

It must not be unified with the contiguous-load S=32 plan from section 3.6:
  #pto.vmi.layout<deinterleaved=4>

Both layouts are legal inputs to group_reduce_addf S=32, but they require
different producer materialization/lowering support.
```

### 3.28 `group_slot_load` `slots = 1` With Aligned Non-Unit Stride

Section 3.16.1 diagnoses non-unit stride for the packed `slots = 8` plan.  The
row-local `slots = 1` plan supports non-unit stride only when each one-lane
load can be issued as an aligned `vsldb`.  In the current lowering this means
the stride is a positive compile-time constant and is divisible by the 32B
alignment expressed in source elements.

VMI input:

```text
%c8 = arith.constant 8 : index
%rhs = pto.vmi.group_slot_load %rhs_base[%rhs_off], %c8 {num_groups = 8}
  : !pto.ptr<f32, ub>, index -> !pto.vmi.vreg<512xf32>
pto.vmi.group_store %rhs, %out[%group_off], %c8 {num_groups = 8}
```

Assigned layout:

```text
%rhs:
  !pto.vmi.vreg<512xf32, #pto.vmi.layout<num_groups = 8, slots = 1>>
```

VPTO lowering result:

```text
%one_b32 = pto.pge_b32 "PAT_VL1"

// Emit this shape for r = 0..7.  The address expression is scalar/index
// arithmetic outside the vector register layout.  For f32, %c8 is 32B.
%addr_r = %rhs_base + %rhs_off + r * 8
%rhs_r = pto.vsldb %addr_r, %c0_i16, %c0_i16, %one_b32
  : !pto.ptr<f32, ub>, i16, i16, !pto.mask<b32> -> !pto.vreg<64xf32>

pto.vsts %rhs_r, %out[%group_tile_off_r], %one_b32 {dist = "NORM_B32"}
  : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
```

Memory result:

```text
for r = 0..7:
  out[group_off + r * 8] = rhs_base[rhs_off + r * 8]
```

Required assignment rule:

```text
If a non-unit-stride group_slot_load has only slots=1 consumers and its stride
is a positive constant divisible by the element count of 32B, select
group_slot_load_slots1_row_local.  Do not diagnose it using the slots=8
unit-stride restriction.
```

Required diagnostic:

```text
%c2 = arith.constant 2 : index
%bad = pto.vmi.group_slot_load %rhs_base[%rhs_off], %c2 {num_groups = 8}
  : !pto.ptr<f32, ub> -> !pto.vmi.vreg<512xf32>

VMI-UNSUPPORTED: pto.vmi.group_slot_load
  slots=1 group_slot_load currently lowers as one lane-0 vsldb per group and
  requires constant positive source_group_stride divisible by 8 elements for
  32B load alignment; packed or unaligned scalar load lowering is not
  implemented.
```

Dynamic stride has the same status until a stable gather or scalarized packed
load plan is designed:

```text
%bad = pto.vmi.group_slot_load %rhs_base[%rhs_off], %runtime_stride
  {num_groups = 8}
  : !pto.ptr<f32, ub> -> !pto.vmi.vreg<512xf32>

VMI-UNSUPPORTED: pto.vmi.group_slot_load
  requires constant positive source_group_stride divisible by 8 elements.
```

### 3.29 One Semantic Mask With f32 And f16 Consumers

One VMI mask may feed consumers with different physical predicate
granularities.  Layout assignment must keep the semantic mask value single, but
materialize per-use physical masks after element type is known.

VMI input:

```text
%x = pto.vmi.load %base[%off]
  : memref<128xf32> -> !pto.vmi.vreg<128xf32>
%mask = pto.vmi.create_mask %c96
  : index -> !pto.vmi.mask<128xpred>
pto.vmi.masked_store %x, %out32[%off], %mask
%h = pto.vmi.truncf %x
  : !pto.vmi.vreg<128xf32> -> !pto.vmi.vreg<128xf16>
pto.vmi.masked_store %h, %out16[%off], %mask
```

Assigned layouts:

```text
%x:
  !pto.vmi.vreg<128xf32, #pto.vmi.layout<contiguous>>

%mask:
  !pto.vmi.mask<128xb32, #pto.vmi.layout<contiguous>>

%x_for_cast:
  pto.vmi.ensure_layout %x
    : #pto.vmi.layout<contiguous> -> #pto.vmi.layout<deinterleaved = 2>

%mask_for_h_store:
  pto.vmi.create_mask %c96
    : index -> !pto.vmi.mask<128xb16, #pto.vmi.layout<contiguous>>

%h:
  !pto.vmi.vreg<128xf16, #pto.vmi.layout<contiguous>>
```

Physical mask materialization:

```text
use at masked_store %x:
  predicate granularity b32, PAT_VL96, layout contiguous

use at vcvt %x -> %h:
  predicate granularity b32, PAT_ALL.  The cast may compute inactive lanes
  because the following masked_store controls the external memory effect.

use at masked_store %h:
  predicate granularity b16, PAT_VL96, layout contiguous
```

VPTO lowering result:

```text
%all_b32 = pto.pge_b32 "PAT_ALL"
%mask32_0 = pto.pge_b32 "PAT_ALL"
%mask32_1 = pto.pge_b32 "PAT_VL32"

%x0 = pto.vlds %base[%off]
  : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>
%x1 = pto.vlds %base[%off_plus_64]
  : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>

pto.vsts %x0, %out32[%off], %mask32_0 {dist = "NORM_B32"}
  : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
pto.vsts %x1, %out32[%off_plus_64], %mask32_1 {dist = "NORM_B32"}
  : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>

%x_p0, %x_p1 = pto.vdintlv %x0, %x1
  : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>, !pto.vreg<64xf32>

%h_even = pto.vcvt %x_p0, %all_b32 {part = "EVEN", rnd = "R", sat = "SAT"}
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<128xf16>
%h_odd = pto.vcvt %x_p1, %all_b32 {part = "ODD", rnd = "R", sat = "SAT"}
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<128xf16>

%all_b16 = pto.pset_b16 "PAT_ALL"
%h0 = pto.vor %h_even, %h_odd, %all_b16
  : !pto.vreg<128xf16>
%mask_b16, %scalar_out = pto.plt_b16 %c96_i32
  : i32 -> !pto.mask<b16>, i32
pto.vsts %h0, %out16[%off], %mask_b16 {dist = "NORM_B16"}
  : !pto.vreg<128xf16>, !pto.ptr<f16, ub>, !pto.mask<b16>
```

Memory result:

```text
for i = 0..95:
  out32[off + i] = base[off + i]
  out16[off + i] = truncf(base[off + i])

for i = 96..127:
  out32[off + i] is unchanged
  out16[off + i] is unchanged
```

Required assignment rule:

```text
`vmi-to-vpto` must not decide mask granularity by inspecting users.  It consumes
the per-use typed mask materialization inserted by vmi-layout-assignment.  For
a rematerializable `create_mask`, assignment may clone it as b32/b16 masks.  For
a non-rematerializable mask producer, assignment must insert
`ensure_mask_granularity` or diagnose if no materialization support exists.
```

### 3.30 `masked_load` Tail Without Padding

This case is the replacement for `vector.transfer_read` padding semantics in the
initial VMI surface.  Tail lanes are expressed by a mask and a passthrough value;
there is no implicit padding constant in the load.  The direct lowering is legal
only when every physical chunk read by `vlds` is memory-safe.

VMI input:

```text
%c100 = arith.constant 100 : index
%mask = pto.vmi.create_mask %c100 : index -> !pto.vmi.mask<100xpred>
%zero = pto.vmi.broadcast %c0_f32 : f32 -> !pto.vmi.vreg<100xf32>
%x = pto.vmi.masked_load %base[%c0], %mask, %zero
  : memref<128xf32>, !pto.vmi.mask<100xpred>, !pto.vmi.vreg<100xf32>
  -> !pto.vmi.vreg<100xf32>
pto.vmi.store %x, %out[%c0]
```

Assigned layouts:

```text
%mask:
  !pto.vmi.mask<100xb32, #pto.vmi.layout<contiguous>>

%zero, %x:
  !pto.vmi.vreg<100xf32, #pto.vmi.layout<contiguous>>
```

VPTO lowering result:

```text
%m0 = pto.pge_b32 "PAT_ALL"
%m1 = pto.pge_b32 "PAT_VL36"

%zero0 = pto.vdup %c0_f32, %m0
  : f32, !pto.mask<b32> -> !pto.vreg<64xf32>
%zero1 = pto.vdup %c0_f32, %m0
  : f32, !pto.mask<b32> -> !pto.vreg<64xf32>

%l0 = pto.vlds %base[%c0]
  : memref<128xf32> -> !pto.vreg<64xf32>
%x0 = pto.vsel %l0, %zero0, %m0
  : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>

%l1 = pto.vlds %base[%c64]
  : memref<128xf32> -> !pto.vreg<64xf32>
%x1 = pto.vsel %l1, %zero1, %m1
  : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>

pto.vsts %x0, %out[%c0], %m0 {dist = "NORM_B32"}
  : !pto.vreg<64xf32>, memref<128xf32>, !pto.mask<b32>
pto.vsts %x1, %out[%c64], %m1 {dist = "NORM_B32"}
  : !pto.vreg<64xf32>, memref<128xf32>, !pto.mask<b32>
```

Memory result:

```text
for i = 0..99:
  out[i] = base[i]

for i = 100..127:
  out[i] is unchanged
```

Required diagnostic when the source cannot prove a safe full-read footprint:

```text
VMI-UNSUPPORTED:
  pto.vmi.masked_load direct lowering requires a supported memory source,
  contiguous result/passthru/mask layouts, and either full physical chunks or a
  statically safe full-read footprint. Use a memref with enough static extent,
  enable the future stable masked/gather load plan, or make the logical vector a
  full physical chunk.
```

Required assignment rule:

```text
`masked_load` requests contiguous result, passthru, and mask layouts.  Padding
is not a layout decision; it is the explicit passthrough operand selected by the
user.
```

### 3.31 `f16 -> f32` Feeding Dense Store And S=16 Reduce

This case proves that the `deinterleaved = 2` layout produced by widening
`f16 -> f32` is not just a store layout.  It must also be a legal S=16 grouped
reduction input.  Layout assignment must not force the reduce consumer to
`block_deinterleaved=2` and then rematerialize the widened value.

VMI input:

```text
%x16 = pto.vmi.load %base[%off]
  : memref<128xf16> -> !pto.vmi.vreg<128xf16>
%x32 = pto.vmi.extf %x16
  : !pto.vmi.vreg<128xf16> -> !pto.vmi.vreg<128xf32>
%mask = pto.vmi.create_mask %c128 : index -> !pto.vmi.mask<128xpred>
%sum = pto.vmi.group_reduce_addf %x32, %mask {num_groups = 8}
pto.vmi.group_store %sum, %sum_out[%group_off], %c1 {num_groups = 8}
pto.vmi.store %x32, %dense_out[%off]
```

Assigned layouts:

```text
%x16:
  !pto.vmi.vreg<128xf16, #pto.vmi.layout<contiguous>>

%x32:
  !pto.vmi.vreg<128xf32, #pto.vmi.layout<deinterleaved = 2>>

%mask:
  !pto.vmi.mask<128xb32, #pto.vmi.layout<deinterleaved = 2>>

%sum:
  !pto.vmi.vreg<128xf32, #pto.vmi.layout<num_groups = 8, slots = 8>>
```

VPTO lowering result:

```text
%all_b16 = pto.pge_b16 "PAT_ALL"
%all_b32 = pto.pge_b32 "PAT_ALL"
%sum_mask = pto.pge_b32 "PAT_VL8"

%x16_0 = pto.vlds %base[%off]
  : memref<128xf16> -> !pto.vreg<128xf16>
%x32_p0 = pto.vcvt %x16_0, %all_b16 {part = "EVEN"}
  : !pto.vreg<128xf16>, !pto.mask<b16> -> !pto.vreg<64xf32>
%x32_p1 = pto.vcvt %x16_0, %all_b16 {part = "ODD"}
  : !pto.vreg<128xf16>, !pto.mask<b16> -> !pto.vreg<64xf32>

%s0 = pto.vcgadd %x32_p0, %all_b32
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%s1 = pto.vcgadd %x32_p1, %all_b32
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%sum_block = pto.vadd %s0, %s1, %sum_mask
  : !pto.vreg<64xf32>

pto.vsts %sum_block, %sum_out[%group_off], %sum_mask {dist = "NORM_B32"}
  : !pto.vreg<64xf32>, memref<8xf32>, !pto.mask<b32>

%dense0, %dense1 = pto.vintlv %x32_p0, %x32_p1
  : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>, !pto.vreg<64xf32>

pto.vsts %dense0, %dense_out[%off], %all_b32 {dist = "NORM_B32"}
  : !pto.vreg<64xf32>, memref<128xf32>, !pto.mask<b32>
pto.vsts %dense1, %dense_out[%off_plus_64], %all_b32 {dist = "NORM_B32"}
  : !pto.vreg<64xf32>, memref<128xf32>, !pto.mask<b32>
```

Memory result:

```text
for r = 0..7:
  sum_out[group_off + r] =
    reduce(extf(base[off + r * 16 + 0 .. off + r * 16 + 15]))

for i = 0..127:
  dense_out[off + i] = extf(base[off + i])
```

Required assignment rule:

```text
When S=16 group_reduce consumes an existing `deinterleaved = 2` dense value,
the reduce plan must accept ordinary `deinterleaved=2`.
`block_deinterleaved=2` is only a producer-driven fast plan for block-fragment
loads, not the semantic requirement of S=16 reduction.
```

### 3.32 `f32` Feeding f8 Store And S=32 Reduce

This is the `f32 -> f8` counterpart to section 3.31.  A 256-lane f32 value can
serve both `truncf -> f8` and S=32 group reduction with the same
`deinterleaved=4` layout.  The value must not be forced to a
`block_deinterleaved=4` layout unless its producer requires that plan.

VMI input:

```text
%x32 = pto.vmi.load %base[%off]
  : memref<256xf32> -> !pto.vmi.vreg<256xf32>
%mask = pto.vmi.create_mask %c256 : index -> !pto.vmi.mask<256xpred>
%sum = pto.vmi.group_reduce_addf %x32, %mask {num_groups = 8}
pto.vmi.group_store %sum, %sum_out[%group_off], %c1 {num_groups = 8}
%x8 = pto.vmi.truncf %x32
  : !pto.vmi.vreg<256xf32> -> !pto.vmi.vreg<256xf8>
pto.vmi.store %x8, %out8[%off]
```

Assigned layouts:

```text
%x32:
  !pto.vmi.vreg<256xf32, #pto.vmi.layout<deinterleaved = 4>>

%mask:
  !pto.vmi.mask<256xb32, #pto.vmi.layout<deinterleaved = 4>>

%sum:
  !pto.vmi.vreg<256xf32, #pto.vmi.layout<num_groups = 8, slots = 8>>

%x8:
  !pto.vmi.vreg<256xf8, #pto.vmi.layout<contiguous>>
```

VPTO lowering result:

```text
%all_b32 = pto.pge_b32 "PAT_ALL"
%sum_mask = pto.pge_b32 "PAT_VL8"

%x_even_0, %x_odd_0 = pto.vldsx2 %base[%off], "DINTLV_B32"
  : memref<256xf32>, index -> !pto.vreg<64xf32>, !pto.vreg<64xf32>
%x_even_1, %x_odd_1 = pto.vldsx2 %base[%off_plus_128], "DINTLV_B32"
  : memref<256xf32>, index -> !pto.vreg<64xf32>, !pto.vreg<64xf32>

%x_p0, %x_p2 = pto.vdintlv %x_even_0, %x_even_1
  : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>, !pto.vreg<64xf32>
%x_p1, %x_p3 = pto.vdintlv %x_odd_0, %x_odd_1
  : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>, !pto.vreg<64xf32>

%s0 = pto.vcgadd %x_p0, %all_b32 : !pto.vreg<64xf32>
%s1 = pto.vcgadd %x_p1, %all_b32 : !pto.vreg<64xf32>
%s2 = pto.vcgadd %x_p2, %all_b32 : !pto.vreg<64xf32>
%s3 = pto.vcgadd %x_p3, %all_b32 : !pto.vreg<64xf32>
%s01 = pto.vadd %s0, %s1, %sum_mask : !pto.vreg<64xf32>
%s23 = pto.vadd %s2, %s3, %sum_mask : !pto.vreg<64xf32>
%sum_block = pto.vadd %s01, %s23, %sum_mask : !pto.vreg<64xf32>

pto.vsts %sum_block, %sum_out[%group_off], %sum_mask {dist = "NORM_B32"}
  : !pto.vreg<64xf32>, memref<8xf32>, !pto.mask<b32>

%x8_p0 = pto.vcvt %x_p0, %all_b32 {part = "P0", rnd = "R", sat = "SAT"}
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<256xf8>
%x8_p1 = pto.vcvt %x_p1, %all_b32 {part = "P1", rnd = "R", sat = "SAT"}
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<256xf8>
%x8_p2 = pto.vcvt %x_p2, %all_b32 {part = "P2", rnd = "R", sat = "SAT"}
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<256xf8>
%x8_p3 = pto.vcvt %x_p3, %all_b32 {part = "P3", rnd = "R", sat = "SAT"}
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<256xf8>

%x8_01 = pto.vor %x8_p0, %x8_p1, PAT_ALL_B8
  : !pto.vreg<256xf8>
%x8_23 = pto.vor %x8_p2, %x8_p3, PAT_ALL_B8
  : !pto.vreg<256xf8>
%x8_0 = pto.vor %x8_01, %x8_23, PAT_ALL_B8
  : !pto.vreg<256xf8>

pto.vsts %x8_0, %out8[%off], PAT_ALL_B8 {dist = "NORM_B8"}
  : !pto.vreg<256xf8>, memref<256xf8>, !pto.mask<b8>
```

Memory result:

```text
for r = 0..7:
  sum_out[group_off + r] =
    reduce(base[off + r * 32 + 0 .. off + r * 32 + 31])

for i = 0..255:
  out8[off + i] = truncf(base[off + i])
```

Required assignment rule:

```text
The common layout selected for `%x32` is
`#pto.vmi.layout<deinterleaved=4>`.  This satisfies both
`truncf f32 -> f8` and S=32 `group_reduce_addf`.  A later strided block-load
producer may introduce `block_deinterleaved=4`, but that is a different case and
requires an explicit materialization/rematerialization decision.

When `%x32` is produced by a full contiguous `pto.vmi.load`, `vmi-to-vpto`
should not first materialize four contiguous f32 chunks and then run a full
four-op `vdintlv` tree.  The load lowering should fold the first deinterleave
level into two `vldsx2 DINTLV_B32` operations and then run only the second
`vdintlv` level, as shown above.  The layout remains just
`deinterleaved=4`; it does not encode the fact that `vldsx2`
was used.
```

### 3.33 One Dense Value Feeding S=16 And S=32 Reduces

This case is a pure layout-assignment conflict.  The same logical
`256xf32` value is consumed by two legal reductions, but their efficient input
layouts are different:

```text
S=16 reduce over 16 groups:
  #pto.vmi.layout<deinterleaved = 2>

S=32 reduce over 8 groups:
  #pto.vmi.layout<deinterleaved = 4>
```

The program is semantically legal.  Baseline layout assignment solves it by
inserting an explicit use-site `ensure_layout`. A later optimization pass may
clone or rematerialize the cheap load for one use. `vmi-to-vpto` must not
inspect both users and choose one locally.

VMI input:

```text
%x = pto.vmi.load %base[%off]
  : memref<256xf32> -> !pto.vmi.vreg<256xf32>

%mask16 = pto.vmi.create_group_mask %c16 {num_groups = 16, group_size = 16}
  : index -> !pto.vmi.mask<256xpred>
%sum16 = pto.vmi.group_reduce_addf %x, %mask16 {num_groups = 16}
pto.vmi.group_store %sum16, %out16[%group_off16], %c1 {num_groups = 16}

%mask32 = pto.vmi.create_group_mask %c32 {num_groups = 8, group_size = 32}
  : index -> !pto.vmi.mask<256xpred>
%sum32 = pto.vmi.group_reduce_addf %x, %mask32 {num_groups = 8}
pto.vmi.group_store %sum32, %out32[%group_off32], %c1 {num_groups = 8}
```

Assigned layouts after rematerializing the load:

```text
%x_s16:
  !pto.vmi.vreg<256xf32, #pto.vmi.layout<deinterleaved = 2>>

%mask16:
  !pto.vmi.mask<256xpred, #pto.vmi.layout<deinterleaved = 2>>

%sum16:
  !pto.vmi.vreg<256xf32, #pto.vmi.layout<num_groups = 16, slots = 8>>

%x_s32:
  !pto.vmi.vreg<256xf32, #pto.vmi.layout<deinterleaved = 4>>

%mask32:
  !pto.vmi.mask<256xpred, #pto.vmi.layout<deinterleaved = 4>>

%sum32:
  !pto.vmi.vreg<256xf32, #pto.vmi.layout<num_groups = 8, slots = 8>>
```

VPTO lowering result:

```text
%all_b32 = pto.pge_b32 "PAT_ALL"
%sum8_mask = pto.pge_b32 "PAT_VL8"

// Rematerialized S=16 use.  The first vldsx2 covers rows 0..7, the second
// covers rows 8..15.  Each pair is deinterleaved by element parity.
%s16_p0, %s16_p1 = pto.vldsx2 %base[%tile_off_0], "DINTLV_B32"
  : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>, !pto.vreg<64xf32>
%s16_p2, %s16_p3 = pto.vldsx2 %base[%tile_off_1], "DINTLV_B32"
  : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>, !pto.vreg<64xf32>

%s16_0 = pto.vcgadd %s16_p0, %all_b32 : !pto.vreg<64xf32>
%s16_1 = pto.vcgadd %s16_p1, %all_b32 : !pto.vreg<64xf32>
%s16_2 = pto.vcgadd %s16_p2, %all_b32 : !pto.vreg<64xf32>
%s16_3 = pto.vcgadd %s16_p3, %all_b32 : !pto.vreg<64xf32>

%sum16_lo = pto.vadd %s16_0, %s16_1, %sum8_mask
  : !pto.vreg<64xf32>
%sum16_hi = pto.vadd %s16_2, %s16_3, %sum8_mask
  : !pto.vreg<64xf32>

pto.vsts %sum16_lo, %out16[%group_off16], %sum8_mask {dist = "NORM_B32"}
  : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
pto.vsts %sum16_hi, %out16[%group_off16_plus_8], %sum8_mask {dist = "NORM_B32"}
  : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>

// Rematerialized S=32 use.  Two DINTLV loads plus one register deinterleave
// level produce mod-4 columns for rows 0..7.
%x_even_0, %x_odd_0 = pto.vldsx2 %base[%tile_off_0], "DINTLV_B32"
  : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>, !pto.vreg<64xf32>
%x_even_1, %x_odd_1 = pto.vldsx2 %base[%tile_off_1], "DINTLV_B32"
  : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>, !pto.vreg<64xf32>

%x_p0, %x_p2 = pto.vdintlv %x_even_0, %x_even_1
  : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>, !pto.vreg<64xf32>
%x_p1, %x_p3 = pto.vdintlv %x_odd_0, %x_odd_1
  : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>, !pto.vreg<64xf32>

%s32_0 = pto.vcgadd %x_p0, %all_b32 : !pto.vreg<64xf32>
%s32_1 = pto.vcgadd %x_p1, %all_b32 : !pto.vreg<64xf32>
%s32_2 = pto.vcgadd %x_p2, %all_b32 : !pto.vreg<64xf32>
%s32_3 = pto.vcgadd %x_p3, %all_b32 : !pto.vreg<64xf32>

%s32_01 = pto.vadd %s32_0, %s32_1, %sum8_mask : !pto.vreg<64xf32>
%s32_23 = pto.vadd %s32_2, %s32_3, %sum8_mask : !pto.vreg<64xf32>
%sum32_block = pto.vadd %s32_01, %s32_23, %sum8_mask : !pto.vreg<64xf32>

pto.vsts %sum32_block, %out32[%group_off32], %sum8_mask {dist = "NORM_B32"}
  : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
```

Memory result:

```text
for r = 0..15:
  out16[group_off16 + r] =
    reduce(base[off + r * 16 + 0 .. off + r * 16 + 15])

for r = 0..7:
  out32[group_off32 + r] =
    reduce(base[off + r * 32 + 0 .. off + r * 32 + 31])
```

Required assignment rule:

```text
Baseline assignment inserts `ensure_layout` at the mismatched use. A later
rematerialization pass may clone a cheap producer such as load and assign each
clone independently. If no deinterleaved=2 <-> deinterleaved=4 materialization
support exists, emit a layout-contract diagnostic naming both consumers and
both required layouts.
```

### 3.34 S=64 Group-Slot Result `f32 -> f16` Cast

Section 3.13 rejects direct width-changing cast for packed `slots = 8`
group-slot values.  This case is the positive counterpart for row-local
`slots = 1`: each group result is already lane 0 of its own physical vreg, so a
slot-preserving cast can lower one row-local result at a time.

VMI input:

```text
%x = pto.vmi.load %base[%off]
  : memref<512xf32> -> !pto.vmi.vreg<512xf32>
%mask = pto.vmi.create_group_mask %c64 {num_groups = 8, group_size = 64}
  : index -> !pto.vmi.mask<512xpred>
%sum32 = pto.vmi.group_reduce_addf %x, %mask {num_groups = 8}
%sum16 = pto.vmi.truncf %sum32
  : !pto.vmi.vreg<512xf32> -> !pto.vmi.vreg<512xf16>
pto.vmi.group_store %sum16, %out[%group_off], %c1 {num_groups = 8}
```

Assigned layouts:

```text
%x:
  !pto.vmi.vreg<512xf32, #pto.vmi.layout<contiguous>>

%sum32:
  !pto.vmi.vreg<512xf32, #pto.vmi.layout<num_groups = 8, slots = 1>>

%sum16:
  !pto.vmi.vreg<512xf16, #pto.vmi.layout<num_groups = 8, slots = 1>>
```

VPTO lowering result for one full 8-row tile:

```text
%all_b32 = pto.pge_b32 "PAT_ALL"
%block8 = pto.pge_b32 "PAT_VL8"
%one_b32 = pto.pge_b32 "PAT_VL1"
%one_b16 = pto.pge_b16 "PAT_VL1"

// The compiler emits this row-local sequence for r = 0..7.
%x_r = pto.vlds %base[%row_off_r] {dist = "NORM"}
  : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
%p_r = pto.vcgadd %x_r, %all_b32
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%sum32_r = pto.vcadd %p_r, %block8
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>

// Only lane 0 is semantic.  EVEN keeps f32 lane 0 in f16 lane 0; all other
// lanes are non-semantic for group_slots(num_groups=8, slots=1).
%sum16_r = pto.vcvt %sum32_r, %one_b32 {part = "EVEN", rnd = "R", sat = "SAT"}
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<128xf16>

pto.vsts %sum16_r, %out[%group_tile_off_r], %one_b16 {dist = "NORM_B16"}
  : !pto.vreg<128xf16>, !pto.ptr<f16, ub>, !pto.mask<b16>
```

Memory result:

```text
for r = 0..7:
  out[group_off + r] =
    truncf(reduce(base[off + r * 64 + 0 .. off + r * 64 + 63]))
```

Required assignment rule:

```text
Group-slot casts are layout-specific.  `slots = 1` may use a slot-preserving
row-local cast because each semantic scalar is lane 0 of its own physical vreg.
This does not legalize packed `slots = 8` casts from section 3.13.
```

### 3.35 `group_slots` Fanout To `group_store` And `group_broadcast`

This case fixes the fanout rule for group-slot values.  A `group_slots` value may
feed multiple group-aware consumers directly.  Layout assignment must not
materialize it as dense just because one later use broadcasts it.

VMI input:

```text
%x = pto.vmi.load %base[%off]
  : memref<128xf32> -> !pto.vmi.vreg<128xf32>
%mask = pto.vmi.create_group_mask %c16 {num_groups = 8, group_size = 16}
  : index -> !pto.vmi.mask<128xpred>
%sum = pto.vmi.group_reduce_addf %x, %mask {num_groups = 8}
pto.vmi.group_store %sum, %sum_out[%group_off], %c1 {num_groups = 8}

%b = pto.vmi.group_broadcast %sum {num_groups = 8}
%y = pto.vmi.vmul %x, %b
%ysum = pto.vmi.group_reduce_addf %y, %mask {num_groups = 8}
pto.vmi.group_store %ysum, %out[%group_off], %c1 {num_groups = 8}
```

Assigned layouts:

```text
%x:
  !pto.vmi.vreg<128xf32, #pto.vmi.layout<contiguous>>

%x_for_reduce:
  !pto.vmi.vreg<128xf32,
    #pto.vmi.layout<block_deinterleaved=2>>

%mask_for_reduce:
  !pto.vmi.mask<128xb32,
    #pto.vmi.layout<block_deinterleaved=2>>

%sum, %ysum:
  !pto.vmi.vreg<128xf32, #pto.vmi.layout<num_groups = 8, slots = 8>>

%b, %y:
  !pto.vmi.vreg<128xf32, #pto.vmi.layout<contiguous>>

%y_for_reduce:
  !pto.vmi.vreg<128xf32,
    #pto.vmi.layout<block_deinterleaved=2>>
```

VPTO lowering result for one full 8-row tile:

```text
%all_b32 = pto.pge_b32 "PAT_ALL"
%slot8 = pto.pge_b32 "PAT_VL8"

%x0 = pto.vlds %base[%tile_off]
  : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>
%x1 = pto.vlds %base[%tile_off_plus_64]
  : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>

// ensure_layout for the first group_reduce.
%x_lo, %x_hi = pto.vdintlv %x0, %x1
  : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>, !pto.vreg<64xf32>

%lo_sum = pto.vcgadd %x_lo, %all_b32
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%hi_sum = pto.vcgadd %x_hi, %all_b32
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%sum_block = pto.vadd %lo_sum, %hi_sum, %slot8 : !pto.vreg<64xf32>

// First group-slot consumer: store the group slots without changing layout.
pto.vsts %sum_block, %sum_out[%group_off], %slot8 {dist = "NORM_B32"}
  : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>

// Second group-slot consumer: materialize only this use as dense grouped data.
%broadcast_idx0 = compute index vector [0 repeated 16, 1 repeated 16,
                                        2 repeated 16, 3 repeated 16]
  : !pto.vreg<64xi32>
%broadcast_idx1 = compute index vector [4 repeated 16, 5 repeated 16,
                                        6 repeated 16, 7 repeated 16]
  : !pto.vreg<64xi32>
%b0 = pto.vselr %sum_block, %broadcast_idx0
  : !pto.vreg<64xf32>, !pto.vreg<64xi32> -> !pto.vreg<64xf32>
%b1 = pto.vselr %sum_block, %broadcast_idx1
  : !pto.vreg<64xf32>, !pto.vreg<64xi32> -> !pto.vreg<64xf32>

%y0 = pto.vmul %x0, %b0, %all_b32 : !pto.vreg<64xf32>
%y1 = pto.vmul %x1, %b1, %all_b32 : !pto.vreg<64xf32>

// ensure_layout for the second group_reduce.
%y_lo, %y_hi = pto.vdintlv %y0, %y1
  : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>, !pto.vreg<64xf32>

%y_lo_sum = pto.vcgadd %y_lo, %all_b32
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%y_hi_sum = pto.vcgadd %y_hi, %all_b32
  : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%ysum_block = pto.vadd %y_lo_sum, %y_hi_sum, %slot8 : !pto.vreg<64xf32>

pto.vsts %ysum_block, %out[%group_off], %slot8 {dist = "NORM_B32"}
  : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
```

Memory result:

```text
for r = 0..7:
  s = reduce(row_r[0..15])
  sum_out[group_off + r] = s
  out[group_off + r] = reduce_i(row_r[i] * s for i = 0..15)
```

Required assignment rule:

```text
`%sum` keeps one assigned layout:
  #pto.vmi.layout<num_groups = 8, slots = 8>

`group_store` consumes that group-slot layout directly.
`group_broadcast` is a use-site materialization to a dense layout.  It must not
rewrite the defining `group_reduce` result or the sibling `group_store` use.
```

### 3.36 Same Scalar Source Materialized As `slots = 8` And `slots = 1`

The same memory scalar stream may be used by both packed S=16 group-slot
compute and row-local S=64 group-slot compute.  The two uses require different
logical vector shapes and different group-slot layouts, so the source must be
rematerialized as two VMI values.  There is no single `group_slots` layout that
serves both uses.

VMI input:

```text
%rhs16 = pto.vmi.group_slot_load %rhs_base[%rhs_off], %c1 {num_groups = 8}
  : !pto.ptr<f32, ub>, index -> !pto.vmi.vreg<128xf32>
%x16 = pto.vmi.load %base16[%off16]
  : memref<128xf32> -> !pto.vmi.vreg<128xf32>
%sum16 = pto.vmi.group_reduce_addf %x16, %mask16 {num_groups = 8}
%out16v = pto.vmi.vadd %sum16, %rhs16
pto.vmi.group_store %out16v, %out16[%group_off16], %c1 {num_groups = 8}

%rhs64 = pto.vmi.group_slot_load %rhs_base[%rhs_off], %c1 {num_groups = 8}
  : !pto.ptr<f32, ub>, index -> !pto.vmi.vreg<512xf32>
%x64 = pto.vmi.load %base64[%off64]
  : memref<512xf32> -> !pto.vmi.vreg<512xf32>
%sum64 = pto.vmi.group_reduce_addf %x64, %mask64 {num_groups = 8}
%out64v = pto.vmi.vadd %sum64, %rhs64
pto.vmi.group_store %out64v, %out64[%group_off64], %c1 {num_groups = 8}
```

Assigned layouts:

```text
%rhs16, %sum16, %out16v:
  !pto.vmi.vreg<128xf32, #pto.vmi.layout<num_groups = 8, slots = 8>>

%x16, %mask16:
  #pto.vmi.layout<block_deinterleaved=2>

%rhs64, %sum64, %out64v:
  !pto.vmi.vreg<512xf32, #pto.vmi.layout<num_groups = 8, slots = 1>>

%x64, %mask64:
  #pto.vmi.layout<contiguous>
```

VPTO lowering result:

```text
// Packed S=16 RHS: one 32B scalar block in lanes 0..7.
%slot8 = pto.pge_b32 "PAT_VL8"
%one_b32 = pto.pge_b32 "PAT_VL1"
%rhs16_block = pto.vsldb %rhs_base[%rhs_off], %c0_i16, %c0_i16, %one_b32
  : !pto.ptr<f32, ub>, i16, i16, !pto.mask<b32> -> !pto.vreg<64xf32>

// S=16 reduction is the section 3.5.1 shape.
%x16_lo, %x16_hi = pto.vldsx2 %base16[%tile_off16], "BDINTLV"
  : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>, !pto.vreg<64xf32>
%s16_lo = pto.vcgadd %x16_lo, PAT_ALL_B32 : !pto.vreg<64xf32>
%s16_hi = pto.vcgadd %x16_hi, PAT_ALL_B32 : !pto.vreg<64xf32>
%sum16_block = pto.vadd %s16_lo, %s16_hi, %slot8 : !pto.vreg<64xf32>
%out16_block = pto.vadd %sum16_block, %rhs16_block, %slot8
  : !pto.vreg<64xf32>
pto.vsts %out16_block, %out16[%group_off16], %slot8 {dist = "NORM_B32"}
  : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>

// Row-local S=64 RHS: a separate group_slot_load op produces one lane-0
// value per physical row-local result.
%rhs64_r = pto.vsldb %rhs_base[%rhs_off_plus_r], %c0_i16, %c0_i16, %one_b32
  : !pto.ptr<f32, ub>, i16, i16, !pto.mask<b32> -> !pto.vreg<64xf32>

// Emit this row-local reduction/add/store shape for r = 0..7.
%x64_r = pto.vlds %base64[%row_off64_r] {dist = "NORM"}
  : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
%p64_r = pto.vcgadd %x64_r, PAT_ALL_B32 : !pto.vreg<64xf32>
%sum64_r = pto.vcadd %p64_r, PAT_VL8_B32 : !pto.vreg<64xf32>
%out64_r = pto.vadd %sum64_r, %rhs64_r, %one_b32 : !pto.vreg<64xf32>
pto.vsts %out64_r, %out64[%group_off64_plus_r], %one_b32 {dist = "NORM_B32"}
  : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
```

Memory result:

```text
for r = 0..7:
  out16[group_off16 + r] = reduce(base16[row_r, 0..15]) + rhs_base[rhs_off + r]
  out64[group_off64 + r] = reduce(base64[row_r, 0..63]) + rhs_base[rhs_off + r]
```

Required assignment rule:

```text
`group_slot_load` is a memory op, so the baseline rematerialization pass must
not clone it as a generic cheap producer. If two use sites need different
`group_slots` layouts, the legal first-stage shape is to write two explicit
`group_slot_load` ops, as above, or to introduce a future load-cloning
optimization with an explicit memory-safety proof. Do not invent a common
layout or make `vmi-to-vpto` inspect both users.
```

### 3.37 S=64 `group_store` With Non-Unit Output Stride

Packed `slots = 8` stores currently require unit output stride.  Row-local
`slots = 1` does not have that restriction because each group scalar is stored
by a separate lane-0 store.

VMI input:

```text
%row_stride = arith.index_cast %ld : i64 to index
%x = pto.vmi.load %base[%off]
  : memref<512xf32> -> !pto.vmi.vreg<512xf32>
%sum = pto.vmi.group_reduce_addf %x, %mask {num_groups = 8}
pto.vmi.group_store %sum, %out[%group_off], %row_stride {num_groups = 8}
```

Assigned layouts:

```text
%x:
  !pto.vmi.vreg<512xf32, #pto.vmi.layout<contiguous>>

%sum:
  !pto.vmi.vreg<512xf32, #pto.vmi.layout<num_groups = 8, slots = 1>>
```

VPTO lowering result:

```text
%all_b32 = pto.pge_b32 "PAT_ALL"
%block8 = pto.pge_b32 "PAT_VL8"
%one_b32 = pto.pge_b32 "PAT_VL1"

// Emit this row-local sequence for r = 0..7.
%x_r = pto.vlds %base[%row_off_r] {dist = "NORM"}
  : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
%p_r = pto.vcgadd %x_r, %all_b32 : !pto.vreg<64xf32>
%sum_r = pto.vcadd %p_r, %block8 : !pto.vreg<64xf32>

%dst_r = %out + %group_off + r * %row_stride
pto.vsts %sum_r, %dst_r, %one_b32 {dist = "NORM_B32"}
  : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
```

Memory result:

```text
for r = 0..7:
  out[group_off + r * row_stride] = reduce(row_r[0..63])
```

Required assignment rule:

```text
If `group_store` has non-unit row_stride and the source can legally use
`slots = 1`, assignment may select `slots = 1` to keep the store legal.  If the
source is fixed to `slots = 8`, current target support must diagnose unless a
strided packed store materializer exists.
```

### 3.38 Multi-Tile S=32 `group_reduce`

The S=32 plan is not only a one-tile special case.  For more than eight groups,
layout assignment keeps the same layout and `vmi-to-vpto` emits the same
8-row tile lowering sequence for each physical tile.

VMI input:

```text
%x = pto.vmi.load %base[%off]
  : memref<512xf32> -> !pto.vmi.vreg<512xf32>
%mask = pto.vmi.create_group_mask %c32 {num_groups = 16, group_size = 32}
  : index -> !pto.vmi.mask<512xpred>
%sum = pto.vmi.group_reduce_addf %x, %mask {num_groups = 16}
pto.vmi.group_store %sum, %out[%group_off], %c1 {num_groups = 16}
```

Assigned layouts:

```text
%x, %mask:
  !pto.vmi.vreg<512xf32,
    #pto.vmi.layout<deinterleaved=4>>

%sum:
  !pto.vmi.vreg<512xf32, #pto.vmi.layout<num_groups = 16, slots = 8>>
```

VPTO lowering result:

```text
// Emit this shape for tile t = 0 and tile t = 1.
// Each tile covers eight 32-f32 rows.
%tile_base_t = %base + %off + t * 256
%tile_out_t = %out + %group_off + t * 8

%x_even_0_t, %x_odd_0_t = pto.vldsx2 %tile_base_t[%c0], "DINTLV_B32"
  : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>, !pto.vreg<64xf32>
%x_even_1_t, %x_odd_1_t = pto.vldsx2 %tile_base_t[%c128], "DINTLV_B32"
  : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>, !pto.vreg<64xf32>

%x_p0_t, %x_p2_t = pto.vdintlv %x_even_0_t, %x_even_1_t
  : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>, !pto.vreg<64xf32>
%x_p1_t, %x_p3_t = pto.vdintlv %x_odd_0_t, %x_odd_1_t
  : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>, !pto.vreg<64xf32>

%s0_t = pto.vcgadd %x_p0_t, PAT_ALL_B32 : !pto.vreg<64xf32>
%s1_t = pto.vcgadd %x_p1_t, PAT_ALL_B32 : !pto.vreg<64xf32>
%s2_t = pto.vcgadd %x_p2_t, PAT_ALL_B32 : !pto.vreg<64xf32>
%s3_t = pto.vcgadd %x_p3_t, PAT_ALL_B32 : !pto.vreg<64xf32>
%s01_t = pto.vadd %s0_t, %s1_t, PAT_VL8_B32 : !pto.vreg<64xf32>
%s23_t = pto.vadd %s2_t, %s3_t, PAT_VL8_B32 : !pto.vreg<64xf32>
%sum_block_t = pto.vadd %s01_t, %s23_t, PAT_VL8_B32
  : !pto.vreg<64xf32>

pto.vsts %sum_block_t, %tile_out_t, PAT_VL8_B32 {dist = "NORM_B32"}
  : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
```

Memory result:

```text
for r = 0..15:
  out[group_off + r] =
    reduce(base[off + r * 32 + 0 .. off + r * 32 + 31])
```

Required assignment rule:

```text
For `group_slots(num_groups = 16, slots = 8)`, the physical arity is
`num_groups / slots = 2`.  The type conversion must expose two packed result
blocks in group order.  `group_store` stores both blocks with offsets
`group_off + 0` and `group_off + 8`.
```

### 3.39 Strided S=32 `group_load` Through Broadcast And Second Reduce

Section 3.27 covers strided S=32 `group_load -> group_reduce -> group_store`.
This case adds the missing dense continuation.  The important layout fact is
that a strided block load naturally produces
`block_deinterleaved=4`; `group_broadcast` must materialize the
broadcast into that same block-fragment layout when the broadcast feeds
elementwise compute and another S=32 group reduction.

VMI input:

```text
%stride40 = arith.constant 40 : index
%x = pto.vmi.group_load %base[%off], %stride40
  {num_groups = 8, group_size = 32}
  : !pto.ptr<f32, ub>, index -> !pto.vmi.vreg<256xf32>
%mask = pto.vmi.create_group_mask %c32 {num_groups = 8, group_size = 32}
  : index -> !pto.vmi.mask<256xpred>
%sum = pto.vmi.group_reduce_addf %x, %mask {num_groups = 8}
%b = pto.vmi.group_broadcast %sum {num_groups = 8}
%y = pto.vmi.vmul %x, %b
%ysum = pto.vmi.group_reduce_addf %y, %mask {num_groups = 8}
pto.vmi.group_store %ysum, %out[%group_off], %c1 {num_groups = 8}
```

Assigned layouts:

```text
%x, %mask, %b, %y:
  !pto.vmi.vreg<256xf32,
    #pto.vmi.layout<block_deinterleaved=4>>

%sum, %ysum:
  !pto.vmi.vreg<256xf32, #pto.vmi.layout<num_groups = 8, slots = 8>>
```

VPTO lowering result:

```text
%all_b32 = pto.pge_b32 "PAT_ALL"
%slot8 = pto.pge_b32 "PAT_VL8"
%stride_blocks = %c5_i16  // 40 f32 = 5 * 32B blocks.

%x_p0 = pto.vsldb %base_frag0, %stride_blocks, %c0_i16, %all_b32
  : !pto.ptr<f32, ub>, i16, i16, !pto.mask<b32> -> !pto.vreg<64xf32>
%x_p1 = pto.vsldb %base_frag1, %stride_blocks, %c0_i16, %all_b32
  : !pto.ptr<f32, ub>, i16, i16, !pto.mask<b32> -> !pto.vreg<64xf32>
%x_p2 = pto.vsldb %base_frag2, %stride_blocks, %c0_i16, %all_b32
  : !pto.ptr<f32, ub>, i16, i16, !pto.mask<b32> -> !pto.vreg<64xf32>
%x_p3 = pto.vsldb %base_frag3, %stride_blocks, %c0_i16, %all_b32
  : !pto.ptr<f32, ub>, i16, i16, !pto.mask<b32> -> !pto.vreg<64xf32>

%s0 = pto.vcgadd %x_p0, %all_b32 : !pto.vreg<64xf32>
%s1 = pto.vcgadd %x_p1, %all_b32 : !pto.vreg<64xf32>
%s2 = pto.vcgadd %x_p2, %all_b32 : !pto.vreg<64xf32>
%s3 = pto.vcgadd %x_p3, %all_b32 : !pto.vreg<64xf32>
%s01 = pto.vadd %s0, %s1, %slot8 : !pto.vreg<64xf32>
%s23 = pto.vadd %s2, %s3, %slot8 : !pto.vreg<64xf32>
%sum_block = pto.vadd %s01, %s23, %slot8 : !pto.vreg<64xf32>

%lane_id = pto.vci %c0_i32 : i32 -> !pto.vreg<64xi32>
%broadcast_idx = pto.vshrs %lane_id, %c3_i16, %all_b32
  : !pto.vreg<64xi32>, i16, !pto.mask<b32> -> !pto.vreg<64xi32>

// Materialize the same per-row scalar into every 32B row fragment.  The four
// bundle entries have the same lane contents, but the result layout remains
// block_deinterleaved=4 because the consumer `%y = vmul %x, %b`
// operates on the block-fragment layout.
%b_p0 = pto.vselr %sum_block, %broadcast_idx
  : !pto.vreg<64xf32>, !pto.vreg<64xi32> -> !pto.vreg<64xf32>
%b_p1 = pto.vselr %sum_block, %broadcast_idx
  : !pto.vreg<64xf32>, !pto.vreg<64xi32> -> !pto.vreg<64xf32>
%b_p2 = pto.vselr %sum_block, %broadcast_idx
  : !pto.vreg<64xf32>, !pto.vreg<64xi32> -> !pto.vreg<64xf32>
%b_p3 = pto.vselr %sum_block, %broadcast_idx
  : !pto.vreg<64xf32>, !pto.vreg<64xi32> -> !pto.vreg<64xf32>

%y_p0 = pto.vmul %x_p0, %b_p0, %all_b32 : !pto.vreg<64xf32>
%y_p1 = pto.vmul %x_p1, %b_p1, %all_b32 : !pto.vreg<64xf32>
%y_p2 = pto.vmul %x_p2, %b_p2, %all_b32 : !pto.vreg<64xf32>
%y_p3 = pto.vmul %x_p3, %b_p3, %all_b32 : !pto.vreg<64xf32>

%ys0 = pto.vcgadd %y_p0, %all_b32 : !pto.vreg<64xf32>
%ys1 = pto.vcgadd %y_p1, %all_b32 : !pto.vreg<64xf32>
%ys2 = pto.vcgadd %y_p2, %all_b32 : !pto.vreg<64xf32>
%ys3 = pto.vcgadd %y_p3, %all_b32 : !pto.vreg<64xf32>
%ys01 = pto.vadd %ys0, %ys1, %slot8 : !pto.vreg<64xf32>
%ys23 = pto.vadd %ys2, %ys3, %slot8 : !pto.vreg<64xf32>
%ysum_block = pto.vadd %ys01, %ys23, %slot8 : !pto.vreg<64xf32>

pto.vsts %ysum_block, %out[%group_off], %slot8 {dist = "NORM_B32"}
  : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
```

Memory result:

```text
for r = 0..7:
  s = reduce(base[off + r * 40 + 0 .. off + r * 40 + 31])
  out[group_off + r] =
    reduce_i(base[off + r * 40 + i] * s for i = 0..31)
```

Required assignment rule:

```text
The split layout kind is part of dense layout compatibility. A broadcast result feeding
an elementwise op with `%x : block_deinterleaved=4` must also be
assigned `block_deinterleaved=4`.  Reusing a
`deinterleaved=4` broadcast would be a layout mismatch even
though both have four physical parts.
```

### 3.40 Scalar Broadcast Feeding Dense And Grouped Users

This case fixes the rule for ordinary scalar broadcasts.  A scalar broadcast is
not born with a physical layout.  Baseline layout assignment assigns the
transfer-equivalent producer chain to the non-contiguous layout requested by the
grouped consumer and inserts an explicit materialization at the dense store use.
The later `vmi-layout-rematerialize` pass may replace that helper with a cloned
broadcast when profitable.

VMI input:

```text
%scale = pto.vmi.broadcast %scale_s
  : f32 -> !pto.vmi.vreg<256xf32>
%x = pto.vmi.load %base[%off]
  : memref<256xf32> -> !pto.vmi.vreg<256xf32>

%copy = pto.vmi.vadd %x, %scale
pto.vmi.store %copy, %copy_out[%off]

%mask = pto.vmi.create_group_mask %c32 {num_groups = 8, group_size = 32}
  : index -> !pto.vmi.mask<256xpred>
%prod = pto.vmi.vmul %x, %scale
%sum = pto.vmi.group_reduce_addf %prod, %mask {num_groups = 8}
pto.vmi.group_store %sum, %sum_out[%group_off], %c1 {num_groups = 8}
```

Assigned layouts:

```text
%x, %scale, %copy, %prod:
  !pto.vmi.vreg<256xf32,
    #pto.vmi.layout<block_deinterleaved=4>>

%copy_dense = pto.vmi.ensure_layout %copy:
  #pto.vmi.layout<block_deinterleaved=4>
    -> #pto.vmi.layout<contiguous>

%mask:
  !pto.vmi.mask<256xpred,
    #pto.vmi.layout<block_deinterleaved=4>>

%sum:
  !pto.vmi.vreg<256xf32, #pto.vmi.layout<num_groups = 8, slots = 8>>
```

VPTO lowering result:

```text
%all_b32 = pto.pge_b32 "PAT_ALL"
%slot8 = pto.pge_b32 "PAT_VL8"

// The shared load is assigned block_deinterleaved=4 because the
// grouped consumer dominates the useful compute layout.
%x0 = pto.vlds %base[%off] : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>
%x1 = pto.vlds %base[%off_plus_64] : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>
%x2 = pto.vlds %base[%off_plus_128] : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>
%x3 = pto.vlds %base[%off_plus_192] : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>

%x01_lo, %x01_hi = pto.vdintlv %x0, %x1
  : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>, !pto.vreg<64xf32>
%x23_lo, %x23_hi = pto.vdintlv %x2, %x3
  : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>, !pto.vreg<64xf32>
%x_p0, %x_p2 = pto.vdintlv %x01_lo, %x23_lo
  : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>, !pto.vreg<64xf32>
%x_p1, %x_p3 = pto.vdintlv %x01_hi, %x23_hi
  : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>, !pto.vreg<64xf32>

%scale_p0 = pto.vdup %scale_s, %all_b32 : f32, !pto.mask<b32> -> !pto.vreg<64xf32>
%scale_p1 = pto.vdup %scale_s, %all_b32 : f32, !pto.mask<b32> -> !pto.vreg<64xf32>
%scale_p2 = pto.vdup %scale_s, %all_b32 : f32, !pto.mask<b32> -> !pto.vreg<64xf32>
%scale_p3 = pto.vdup %scale_s, %all_b32 : f32, !pto.mask<b32> -> !pto.vreg<64xf32>

// Dense store use: compute in deinterleaved=4, then ensure_layout materializes
// the contiguous memory order for the external effect.
%copy_p0 = pto.vadd %x_p0, %scale_p0, %all_b32 : !pto.vreg<64xf32>
%copy_p1 = pto.vadd %x_p1, %scale_p1, %all_b32 : !pto.vreg<64xf32>
%copy_p2 = pto.vadd %x_p2, %scale_p2, %all_b32 : !pto.vreg<64xf32>
%copy_p3 = pto.vadd %x_p3, %scale_p3, %all_b32 : !pto.vreg<64xf32>

%c01_lo, %c01_hi = pto.vintlv %copy_p0, %copy_p2
  : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>, !pto.vreg<64xf32>
%c23_lo, %c23_hi = pto.vintlv %copy_p1, %copy_p3
  : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>, !pto.vreg<64xf32>
%copy0, %copy1 = pto.vintlv %c01_lo, %c23_lo
  : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>, !pto.vreg<64xf32>
%copy2, %copy3 = pto.vintlv %c01_hi, %c23_hi
  : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>, !pto.vreg<64xf32>

pto.vsts %copy0, %copy_out[%off], %all_b32 {dist = "NORM_B32"}
pto.vsts %copy1, %copy_out[%off_plus_64], %all_b32 {dist = "NORM_B32"}
pto.vsts %copy2, %copy_out[%off_plus_128], %all_b32 {dist = "NORM_B32"}
pto.vsts %copy3, %copy_out[%off_plus_192], %all_b32 {dist = "NORM_B32"}

// Grouped use: reuse the same deinterleaved operands directly.
%prod_p0 = pto.vmul %x_p0, %scale_p0, %all_b32 : !pto.vreg<64xf32>
%prod_p1 = pto.vmul %x_p1, %scale_p1, %all_b32 : !pto.vreg<64xf32>
%prod_p2 = pto.vmul %x_p2, %scale_p2, %all_b32 : !pto.vreg<64xf32>
%prod_p3 = pto.vmul %x_p3, %scale_p3, %all_b32 : !pto.vreg<64xf32>

%s0 = pto.vcgadd %prod_p0, %all_b32 : !pto.vreg<64xf32>
%s1 = pto.vcgadd %prod_p1, %all_b32 : !pto.vreg<64xf32>
%s2 = pto.vcgadd %prod_p2, %all_b32 : !pto.vreg<64xf32>
%s3 = pto.vcgadd %prod_p3, %all_b32 : !pto.vreg<64xf32>
%s01 = pto.vadd %s0, %s1, %slot8 : !pto.vreg<64xf32>
%s23 = pto.vadd %s2, %s3, %slot8 : !pto.vreg<64xf32>
%sum_block = pto.vadd %s01, %s23, %slot8 : !pto.vreg<64xf32>

pto.vsts %sum_block, %sum_out[%group_off], %slot8 {dist = "NORM_B32"}
  : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
```

Memory result:

```text
for i = 0..255:
  copy_out[off + i] = base[off + i] + scale_s

for r = 0..7:
  sum_out[group_off + r] =
    reduce_i(base[off + r * 32 + i] * scale_s for i = 0..31)
```

Required assignment rule:

```text
`broadcast` is layout-transparent and cheaply rematerializable by the optional
`vmi-layout-rematerialize` pass, but baseline assignment does not have to force
a separate contiguous broadcast just because a dense store exists.  It may
choose a common deinterleaved compute layout for transfer-equivalent elementwise
ops and insert `ensure_layout` at the dense store.  The required invariant is
that this choice is explicit in the assigned IR; `vmi-to-vpto` must not infer it
by inspecting both users.
```

### 3.41 Non-Rematerializable Value With Incompatible Users

This is the non-cheap counterpart to section 3.18.  A `masked_load` has explicit
mask and passthrough semantics, so layout assignment should not clone it as a
normal cheap load unless a dedicated rematerialization rule proves that clone
legal.  The conflict is solved by inserting `ensure_layout` at one use site.

VMI input:

```text
%mask = pto.vmi.create_mask %c256 : index -> !pto.vmi.mask<256xpred>
%zero = pto.vmi.broadcast %c0_f32 : f32 -> !pto.vmi.vreg<256xf32>
%x = pto.vmi.masked_load %base[%off], %mask, %zero
  : memref<256xf32>, !pto.vmi.mask<256xpred>, !pto.vmi.vreg<256xf32>
  -> !pto.vmi.vreg<256xf32>

pto.vmi.store %x, %copy_out[%off]

%sum = pto.vmi.group_reduce_addf %x, %mask {num_groups = 8}
pto.vmi.group_store %sum, %sum_out[%group_off], %c1 {num_groups = 8}
```

Assigned layouts:

```text
%x, %zero for masked_load/store:
  !pto.vmi.vreg<256xf32, #pto.vmi.layout<contiguous>>

%mask for masked_load/store:
  !pto.vmi.mask<256xpred, #pto.vmi.layout<contiguous>>

%x_for_reduce = pto.vmi.ensure_layout %x
  : #pto.vmi.layout<contiguous>
 -> #pto.vmi.layout<deinterleaved = 4>

%mask_for_reduce = pto.vmi.ensure_mask_layout %mask
  : #pto.vmi.layout<contiguous>
 -> #pto.vmi.layout<deinterleaved = 4>

%sum:
  !pto.vmi.vreg<256xf32, #pto.vmi.layout<num_groups = 8, slots = 8>>
```

VPTO lowering result:

```text
%all_b32 = pto.pge_b32 "PAT_ALL"
%slot8 = pto.pge_b32 "PAT_VL8"

%zero0 = pto.vdup %c0_f32, %all_b32 : f32, !pto.mask<b32> -> !pto.vreg<64xf32>
%zero1 = pto.vdup %c0_f32, %all_b32 : f32, !pto.mask<b32> -> !pto.vreg<64xf32>
%zero2 = pto.vdup %c0_f32, %all_b32 : f32, !pto.mask<b32> -> !pto.vreg<64xf32>
%zero3 = pto.vdup %c0_f32, %all_b32 : f32, !pto.mask<b32> -> !pto.vreg<64xf32>

%l0 = pto.vlds %base[%off] : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>
%l1 = pto.vlds %base[%off_plus_64] : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>
%l2 = pto.vlds %base[%off_plus_128] : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>
%l3 = pto.vlds %base[%off_plus_192] : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>

%x0 = pto.vsel %l0, %zero0, %all_b32 : !pto.vreg<64xf32>
%x1 = pto.vsel %l1, %zero1, %all_b32 : !pto.vreg<64xf32>
%x2 = pto.vsel %l2, %zero2, %all_b32 : !pto.vreg<64xf32>
%x3 = pto.vsel %l3, %zero3, %all_b32 : !pto.vreg<64xf32>

pto.vsts %x0, %copy_out[%off], %all_b32 {dist = "NORM_B32"}
pto.vsts %x1, %copy_out[%off_plus_64], %all_b32 {dist = "NORM_B32"}
pto.vsts %x2, %copy_out[%off_plus_128], %all_b32 {dist = "NORM_B32"}
pto.vsts %x3, %copy_out[%off_plus_192], %all_b32 {dist = "NORM_B32"}

// ensure_layout contiguous -> deinterleaved=4 at the reduce use.
%x01_lo, %x01_hi = pto.vdintlv %x0, %x1
  : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>, !pto.vreg<64xf32>
%x23_lo, %x23_hi = pto.vdintlv %x2, %x3
  : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>, !pto.vreg<64xf32>
%x_p0, %x_p2 = pto.vdintlv %x01_lo, %x23_lo
  : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>, !pto.vreg<64xf32>
%x_p1, %x_p3 = pto.vdintlv %x01_hi, %x23_hi
  : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>, !pto.vreg<64xf32>

%s0 = pto.vcgadd %x_p0, %all_b32 : !pto.vreg<64xf32>
%s1 = pto.vcgadd %x_p1, %all_b32 : !pto.vreg<64xf32>
%s2 = pto.vcgadd %x_p2, %all_b32 : !pto.vreg<64xf32>
%s3 = pto.vcgadd %x_p3, %all_b32 : !pto.vreg<64xf32>
%s01 = pto.vadd %s0, %s1, %slot8 : !pto.vreg<64xf32>
%s23 = pto.vadd %s2, %s3, %slot8 : !pto.vreg<64xf32>
%sum_block = pto.vadd %s01, %s23, %slot8 : !pto.vreg<64xf32>

pto.vsts %sum_block, %sum_out[%group_off], %slot8 {dist = "NORM_B32"}
  : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
```

Memory result:

```text
for i = 0..255:
  copy_out[off + i] = base[off + i]

for r = 0..7:
  sum_out[group_off + r] =
    reduce(base[off + r * 32 + 0 .. off + r * 32 + 31])
```

Required assignment rule:

```text
For non-rematerializable producers, assignment must insert an explicit use-site
materialization helper, such as contiguous -> deinterleaved=4. If that helper
has no supported materialization, the layout gate must diagnose before
vmi-to-vpto. `vmi-to-vpto` must not clone the masked_load or choose a
materialization after seeing both users.
```

### 3.42 `group_slots` `scf.for` Loop-Carried Accumulator

Section 3.22 covers dense loop-carried values.  Group-slot values need a
separate case because the loop-carried block argument has no dense lane
semantics outside the live group slots.

VMI input:

```text
%acc0 = pto.vmi.group_slot_load %init[%group_off], %c1 {num_groups = 8}
  : !pto.ptr<f32, ub>, index -> !pto.vmi.vreg<128xf32>

%acc = scf.for %k = %c0 to %steps step %c1
    iter_args(%arg = %acc0) -> !pto.vmi.vreg<128xf32> {
  %x = pto.vmi.group_load %base[%tile_off_k], %c16
    {num_groups = 8, group_size = 16}
    : !pto.ptr<f32, ub>, index -> !pto.vmi.vreg<128xf32>
  %mask = pto.vmi.create_group_mask %c16 {num_groups = 8, group_size = 16}
    : index -> !pto.vmi.mask<128xpred>
  %sum = pto.vmi.group_reduce_addf %x, %mask {num_groups = 8}
  %next = pto.vmi.vadd %arg, %sum
  scf.yield %next : !pto.vmi.vreg<128xf32>
}

pto.vmi.group_store %acc, %out[%group_off], %c1 {num_groups = 8}
```

Assigned layouts:

```text
%acc0, %arg, %sum, %next, %acc:
  !pto.vmi.vreg<128xf32, #pto.vmi.layout<num_groups = 8, slots = 8>>

%x:
  !pto.vmi.vreg<128xf32,
    #pto.vmi.layout<block_deinterleaved=2>>

%mask:
  !pto.vmi.mask<128xpred,
    #pto.vmi.layout<block_deinterleaved=2>>
```

VPTO lowering result:

```text
%all_b32 = pto.pge_b32 "PAT_ALL"
%slot8 = pto.pge_b32 "PAT_VL8"
%one_b32 = pto.pge_b32 "PAT_VL1"

%acc0_block = pto.vsldb %init[%group_off], %c0_i16, %c0_i16, %one_b32
  : !pto.ptr<f32, ub>, i16, i16, !pto.mask<b32> -> !pto.vreg<64xf32>

%acc_block = scf.for %k = %c0 to %steps step %c1
    iter_args(%arg_block = %acc0_block) -> !pto.vreg<64xf32> {
  %lo, %hi = pto.vldsx2 %base[%tile_off_k], "BDINTLV"
    : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>, !pto.vreg<64xf32>
  %lo_sum = pto.vcgadd %lo, %all_b32 : !pto.vreg<64xf32>
  %hi_sum = pto.vcgadd %hi, %all_b32 : !pto.vreg<64xf32>
  %sum_block = pto.vadd %lo_sum, %hi_sum, %slot8 : !pto.vreg<64xf32>
  %next_block = pto.vadd %arg_block, %sum_block, %slot8 : !pto.vreg<64xf32>
  scf.yield %next_block : !pto.vreg<64xf32>
}

pto.vsts %acc_block, %out[%group_off], %slot8 {dist = "NORM_B32"}
  : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
```

Memory result:

```text
for r = 0..7:
  out[group_off + r] =
      init[group_off + r]
    + sum_k reduce(base[tile_k, row_r, 0..15])
```

Required assignment rule:

```text
Loop-carried `group_slots` values are valid.  The iter_arg, body block
argument, yield operand, loop result, and final group_store operand all carry
the same `group_slots(num_groups=8, slots=8)` layout.  Ordinary dense consumers
inside the loop still require an explicit `group_broadcast` or diagnostic.
```

### 3.43 Internal Function Argument Boundary Materialization

Section 3.25 covers a private function returning a VMI value.  A callee argument
is the other direction of the same ABI problem: the callee body may require a
layout that is different from the layout naturally produced at a call site.

The current implementation keeps the internal function VMI signature
contiguous and makes the callee-entry materialization explicit with
`ensure_layout` / `ensure_mask_layout`.  This is less aggressive than
specializing the VMI function signature to `deinterleaved = 4`, but it preserves
the same invariant: after layout assignment, `vmi-to-vpto` lowers only from
explicit type and helper information and does not inspect the callee body while
lowering a call.

VMI input:

```text
func.func private @consume(%x: !pto.vmi.vreg<256xf32>,
                           %mask: !pto.vmi.mask<256xpred>,
                           %out: !pto.ptr<f32, ub>, %group_off: index) {
  %sum = pto.vmi.group_reduce_addf %x, %mask {num_groups = 8}
  pto.vmi.group_store %sum, %out[%group_off], %c1 {num_groups = 8}
  return
}

func.func @caller(%base: !pto.ptr<f32, ub>, %off: index,
                  %out: !pto.ptr<f32, ub>, %group_off: index) {
  %x = pto.vmi.load %base[%off]
    : !pto.ptr<f32, ub>, index -> !pto.vmi.vreg<256xf32>
  %mask = pto.vmi.create_group_mask %c32 {num_groups = 8, group_size = 32}
    : index -> !pto.vmi.mask<256xpred>
  call @consume(%x, %mask, %out, %group_off)
    : (!pto.vmi.vreg<256xf32>, !pto.vmi.mask<256xpred>,
       !pto.ptr<f32, ub>, index) -> ()
  return
}
```

Assigned layouts:

```text
@consume argument %x:
  !pto.vmi.vreg<256xf32, #pto.vmi.layout<contiguous>>

@consume argument %mask:
  !pto.vmi.mask<256xpred, #pto.vmi.layout<contiguous>>

inside @consume:
  %x_split = pto.vmi.ensure_layout %x
    : #pto.vmi.layout<contiguous>
   -> #pto.vmi.layout<block_deinterleaved=4>

  %mask_split = pto.vmi.ensure_mask_layout %mask
    : #pto.vmi.layout<contiguous>
   -> #pto.vmi.layout<block_deinterleaved=4>

@caller %x and %mask:
  !pto.vmi.vreg<256xf32, #pto.vmi.layout<contiguous>>
  !pto.vmi.mask<256xpred, #pto.vmi.layout<contiguous>>

%sum:
  !pto.vmi.vreg<256xf32, #pto.vmi.layout<num_groups = 8, slots = 8>>
```

VPTO lowering result for the function boundary:

```text
func.func private @consume(%x_p0: !pto.vreg<64xf32>,
                           %x_p1: !pto.vreg<64xf32>,
                           %x_p2: !pto.vreg<64xf32>,
                           %x_p3: !pto.vreg<64xf32>,
                           %m0: !pto.mask<b32>,
                           %m1: !pto.mask<b32>,
                           %m2: !pto.mask<b32>,
                           %m3: !pto.mask<b32>,
                           %out: !pto.ptr<f32, ub>,
                           %group_off: index) {
  // Callee-entry lowering of ensure_layout contiguous -> block_deinterleaved=4.
  %x01_lo, %x01_hi = pto.vdintlv %x_p0, %x_p1
    : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>, !pto.vreg<64xf32>
  %x23_lo, %x23_hi = pto.vdintlv %x_p2, %x_p3
    : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>, !pto.vreg<64xf32>
  %x_d0, %x_d2 = pto.vdintlv %x01_lo, %x23_lo
    : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>, !pto.vreg<64xf32>
  %x_d1, %x_d3 = pto.vdintlv %x01_hi, %x23_hi
    : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>, !pto.vreg<64xf32>

  %m01_lo, %m01_hi = pto.pdintlv_b32 %m0, %m1
    : !pto.mask<b32>, !pto.mask<b32> -> !pto.mask<b32>, !pto.mask<b32>
  %m23_lo, %m23_hi = pto.pdintlv_b32 %m2, %m3
    : !pto.mask<b32>, !pto.mask<b32> -> !pto.mask<b32>, !pto.mask<b32>
  %m_d0, %m_d2 = pto.pdintlv_b32 %m01_lo, %m23_lo
    : !pto.mask<b32>, !pto.mask<b32> -> !pto.mask<b32>, !pto.mask<b32>
  %m_d1, %m_d3 = pto.pdintlv_b32 %m01_hi, %m23_hi
    : !pto.mask<b32>, !pto.mask<b32> -> !pto.mask<b32>, !pto.mask<b32>

  %slot8 = pto.pge_b32 "PAT_VL8"
  %s0 = pto.vcgadd %x_d0, %m_d0 : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
  %s1 = pto.vcgadd %x_d1, %m_d1 : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
  %s2 = pto.vcgadd %x_d2, %m_d2 : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
  %s3 = pto.vcgadd %x_d3, %m_d3 : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
  %s01 = pto.vadd %s0, %s1, %slot8 : !pto.vreg<64xf32>
  %s23 = pto.vadd %s2, %s3, %slot8 : !pto.vreg<64xf32>
  %sum_block = pto.vadd %s01, %s23, %slot8 : !pto.vreg<64xf32>
  pto.vsts %sum_block, %out[%group_off], %slot8 {dist = "NORM_B32"}
    : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
  return
}

func.func @caller(...) {
  // Caller keeps the load and group mask in the contiguous function ABI layout.
  %x0 = pto.vlds %base[%off] : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
  %x1 = pto.vlds %base[%off_plus_64] : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
  %x2 = pto.vlds %base[%off_plus_128] : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
  %x3 = pto.vlds %base[%off_plus_192] : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>

  %m0 = pto.pset_b32 "PAT_ALL" : !pto.mask<b32>
  %m1 = pto.pset_b32 "PAT_ALL" : !pto.mask<b32>
  %m2 = pto.pset_b32 "PAT_ALL" : !pto.mask<b32>
  %m3 = pto.pset_b32 "PAT_ALL" : !pto.mask<b32>

  call @consume(%x0, %x1, %x2, %x3, %m0, %m1, %m2, %m3, %out, %group_off)
    : (!pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.vreg<64xf32>,
       !pto.vreg<64xf32>, !pto.mask<b32>, !pto.mask<b32>,
       !pto.mask<b32>, !pto.mask<b32>, !pto.ptr<f32, ub>, index) -> ()
  return
}
```

Memory result:

```text
for r = 0..7:
  out[group_off + r] =
    reduce(base[off + r * 32 + 0 .. off + r * 32 + 31])
```

Required assignment rule:

```text
Private function boundary layout is explicit in the assigned function type and
callee-entry helpers.  The current endpoint chooses a contiguous VMI function
ABI and inserts callee-entry materialization for the grouped body requirement.
`vmi-to-vpto` does not inspect the callee body while lowering the call and does
not inspect callers while lowering the callee block argument.

Future optimization may specialize private VMI function signatures directly to
`block_deinterleaved=4` when all call sites agree.  That
optimization must still be expressed in the assigned VMI function type before
`vmi-to-vpto` runs.
```

Runtime closure:

```text
lit:
  test/lit/vmi_new/vmi_layout_assignment_call_argument_boundary.pto
  test/lit/vmi_new/vmi_ptoas_call_boundary_vecscope.pto

runtime SIM:
  test/vpto/cases/vmi_new/private-call-argument-boundary-store

ptoas pipeline:
  vmi-layout-assignment inserts explicit callee-entry materialization
  vmi-to-vpto physicalizes the call operands and callee body
  ptoas then inlines the private physical helper before VPTO vecscope/backend
  emission, so the backend never needs a physical VPTO vector function ABI
```

### 3.44 `masked_load` Grouped Tail Feeding S=32 Reduce

This case connects the explicit `masked_load` tail model from section 3.30 with
grouped reduction.  The load has no padding constant hidden in the op; inactive
lanes are provided by the passthrough value and excluded from the reduction by
the same grouped mask.

VMI input:

```text
%c25 = arith.constant 25 : index
%mask = pto.vmi.create_group_mask %c25 {num_groups = 8, group_size = 32}
  : index -> !pto.vmi.mask<256xpred>
%zero = pto.vmi.broadcast %c0_f32 : f32 -> !pto.vmi.vreg<256xf32>
%x = pto.vmi.masked_load %base[%off], %mask, %zero
  : memref<256xf32>, !pto.vmi.mask<256xpred>, !pto.vmi.vreg<256xf32>
  -> !pto.vmi.vreg<256xf32>
%sum = pto.vmi.group_reduce_addf %x, %mask {num_groups = 8}
pto.vmi.group_store %sum, %out[%group_off], %c1 {num_groups = 8}
```

Assigned layouts:

```text
%mask for masked_load:
  !pto.vmi.mask<256xpred, #pto.vmi.layout<contiguous>>

%zero, %x for masked_load:
  !pto.vmi.vreg<256xf32, #pto.vmi.layout<contiguous>>

%x_for_reduce = pto.vmi.ensure_layout %x:
  #pto.vmi.layout<contiguous>
    -> #pto.vmi.layout<block_deinterleaved=4>

%mask_for_reduce:
  pto.vmi.create_group_mask %c25 {num_groups = 8, group_size = 32}
    -> !pto.vmi.mask<256xpred,
         #pto.vmi.layout<block_deinterleaved=4>>

%sum:
  !pto.vmi.vreg<256xf32, #pto.vmi.layout<num_groups = 8, slots = 8>>
```

Lowering:

```text
%all_b32 = pto.pge_b32 "PAT_ALL"
%slot8 = pto.pge_b32 "PAT_VL8"

// masked_load direct lowering stays contiguous.
%m0, %m1, %m2, %m3 = materialize contiguous create_group_mask(c25, S=32)
%z0, %z1, %z2, %z3 = vdup zero
%l0 = pto.vlds %base[%off]
%l1 = pto.vlds %base[%off_plus_64]
%l2 = pto.vlds %base[%off_plus_128]
%l3 = pto.vlds %base[%off_plus_192]
%x0 = pto.vsel %l0, %z0, %m0 : !pto.vreg<64xf32>
%x1 = pto.vsel %l1, %z1, %m1 : !pto.vreg<64xf32>
%x2 = pto.vsel %l2, %z2, %m2 : !pto.vreg<64xf32>
%x3 = pto.vsel %l3, %z3, %m3 : !pto.vreg<64xf32>

// ensure_layout contiguous -> block_deinterleaved=4.
%x01_lo, %x01_hi = pto.vdintlv %x0, %x1
%x23_lo, %x23_hi = pto.vdintlv %x2, %x3
%x_p0, %x_p2 = pto.vdintlv %x01_lo, %x23_lo
%x_p1, %x_p3 = pto.vdintlv %x01_hi, %x23_hi

// The reduce-side grouped mask is not built by guessing the final group-slot
// predicate image.  It is first materialized as the same contiguous grouped
// mask used by masked_load, then converted to the reduce layout with predicate
// deinterleave.  This keeps predicate reordering identical to the data
// reordering above.
%rm0, %rm1, %rm2, %rm3 = materialize contiguous create_group_mask(c25, S=32)
%rm01_lo, %rm01_hi = pto.pdintlv_b32 %rm0, %rm1
%rm23_lo, %rm23_hi = pto.pdintlv_b32 %rm2, %rm3
%mask_p0, %mask_p2 = pto.pdintlv_b32 %rm01_lo, %rm23_lo
%mask_p1, %mask_p3 = pto.pdintlv_b32 %rm01_hi, %rm23_hi

%s0 = pto.vcgadd %x_p0, %mask_p0 : !pto.vreg<64xf32>
%s1 = pto.vcgadd %x_p1, %mask_p1 : !pto.vreg<64xf32>
%s2 = pto.vcgadd %x_p2, %mask_p2 : !pto.vreg<64xf32>
%s3 = pto.vcgadd %x_p3, %mask_p3 : !pto.vreg<64xf32>
%s01 = pto.vadd %s0, %s1, %slot8 : !pto.vreg<64xf32>
%s23 = pto.vadd %s2, %s3, %slot8 : !pto.vreg<64xf32>
%sum_block = pto.vadd %s01, %s23, %slot8 : !pto.vreg<64xf32>

pto.vsts %sum_block, %out[%group_off], %slot8 {dist = "NORM_B32"}
  : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
```

Memory result:

```text
for r = 0..7:
  out[group_off + r] =
    reduce(base[off + r * 32 + 0 .. off + r * 32 + 24])
```

Required assignment rule:

`masked_load` and `group_reduce` must share the same grouped mask layout.  The
passthrough value defines inactive loaded lanes, while the reduce mask defines
participation.  Assignment materializes two explicit mask values when needed:
one contiguous value for `masked_load`, and one deinterleaved value for
`group_reduce_addf`.  `vmi-to-vpto` lowers the deinterleaved
`create_group_mask` by materializing the contiguous grouped predicate chunks
and then applying `pdintlv_b32` in the same tree shape as the data
`vdintlv`.  It does not walk from `group_reduce_addf` to the mask producer to
choose or reject the support path.

Assignment may select a deinterleaved S=32 load layout only when the rounded
physical reads are memory-safe; otherwise it must diagnose or use a future
stable gather fallback.

Runtime coverage:

```text
test/vpto/cases/vmi_new/masked-load-group-tail-s32-reduce-store
```

### 3.45 Dynamic S=32 `create_group_mask`

This is the dynamic-shape form of section 3.44.  The active column count is an
SSA `index`, not a constant.  The semantic mask is still grouped:

```text
lane i active iff (i % 32) < active_cols
```

VMI input:

```text
%mask = pto.vmi.create_group_mask %active_cols
  {num_groups = 8, group_size = 32}
  : index -> !pto.vmi.mask<256xpred>
```

Assigned layouts:

```text
%mask for masked_load:
  !pto.vmi.mask<256xb32, #pto.vmi.layout<contiguous>>

%mask for S=32 group_reduce:
  !pto.vmi.mask<256xb32,
    #pto.vmi.layout<block_deinterleaved=4>>
```

Contiguous VPTO lowering for one b32 physical chunk:

```text
%active_i32 = arith.index_cast %active_cols : index to i32
%active_nonneg = arith.maxsi %active_i32, %c0_i32 : i32
%active_clamped = arith.minui %active_nonneg, %c32_i32 : i32

%all = pto.pset_b32 "PAT_ALL" : !pto.mask<b32>
%lane = pto.vci %c0_i32 : i32 -> !pto.vreg<64xi32>
%row = pto.vshrs %lane, %c5_i16, %all
  : !pto.vreg<64xi32>, i16, !pto.mask<b32> -> !pto.vreg<64xi32>
%row_base = pto.vshls %row, %c5_i16, %all
  : !pto.vreg<64xi32>, i16, !pto.mask<b32> -> !pto.vreg<64xi32>
%col = pto.vsub %lane, %row_base, %all
  : !pto.vreg<64xi32>, !pto.vreg<64xi32>, !pto.mask<b32>
  -> !pto.vreg<64xi32>
%m = pto.vcmps %col, %active_clamped, %all, "lt"
  : !pto.vreg<64xi32>, i32, !pto.mask<b32> -> !pto.mask<b32>
```

For `block_deinterleaved=4`, lowering first emits four contiguous
chunks with the sequence above, then applies the same predicate deinterleave
tree used by section 3.44:

```text
%rm0, %rm1, %rm2, %rm3 = dynamic contiguous grouped masks
%rm01_lo, %rm01_hi = pto.pdintlv_b32 %rm0, %rm1
%rm23_lo, %rm23_hi = pto.pdintlv_b32 %rm2, %rm3
%mask_p0, %mask_p2 = pto.pdintlv_b32 %rm01_lo, %rm23_lo
%mask_p1, %mask_p3 = pto.pdintlv_b32 %rm01_hi, %rm23_hi
```

Current coverage validates both IR lowering and runtime behavior:

```text
test/lit/vmi_new/vmi_layout_assignment_create_group_mask_s32_dynamic.pto
test/vpto/cases/vmi_new/dynamic-create-group-mask-s32-reduce-store
```

The runtime case passes `active_cols` as a kernel scalar argument and casts it
to `index` inside `pto.vecscope`.  This keeps scalar materialization outside
`vmi-to-vpto`; the lowering pass only consumes the current
`create_group_mask` operand.

### 3.46 `extf` Value And Derived Elementwise Value Both Stored

This case fixes where contiguous materialization belongs when one widened value
is used directly by a store and also by a layout-transparent elementwise chain
that is stored.

VMI input:

```text
%a = pto.vmi.load %in[%off]
  : memref<128xf16> -> !pto.vmi.vreg<128xf16>
%k = pto.vmi.broadcast %k1
  : f32 -> !pto.vmi.vreg<128xf32>

%w = pto.vmi.extf %a
  : !pto.vmi.vreg<128xf16> -> !pto.vmi.vreg<128xf32>
%t1 = pto.vmi.vmul %w, %k
  : !pto.vmi.vreg<128xf32>, !pto.vmi.vreg<128xf32>
    -> !pto.vmi.vreg<128xf32>

pto.vmi.store %t1, %out1[%off]
pto.vmi.store %w,  %out2[%off]
```

Hard-legalized assigned layouts:

```text
%a:
  !pto.vmi.vreg<128xf16, #pto.vmi.layout<contiguous>>

%w, %k, %t1:
  !pto.vmi.vreg<128xf32, #pto.vmi.layout<deinterleaved = 2>>

%t1_c = pto.vmi.ensure_layout %t1:
  #pto.vmi.layout<deinterleaved = 2> -> #pto.vmi.layout<contiguous>
pto.vmi.store %t1_c, %out1[%off]

%w_c = pto.vmi.ensure_layout %w:
  #pto.vmi.layout<deinterleaved = 2> -> #pto.vmi.layout<contiguous>
pto.vmi.store %w_c, %out2[%off]
```

Baseline VPTO lowering result:

```text
%a0 = pto.vlds %in[%off] {dist = "NORM"}
  : !pto.ptr<f16, ub>, index -> !pto.vreg<128xf16>

%w_p0 = pto.vcvt %a0, PAT_ALL_B16 {part = "EVEN"}
  : !pto.vreg<128xf16>, !pto.mask<b16> -> !pto.vreg<64xf32>
%w_p1 = pto.vcvt %a0, PAT_ALL_B16 {part = "ODD"}
  : !pto.vreg<128xf16>, !pto.mask<b16> -> !pto.vreg<64xf32>

%k_p0 = pto.vdup %k1, PAT_ALL_B32 : f32, !pto.mask<b32> -> !pto.vreg<64xf32>
%k_p1 = pto.vdup %k1, PAT_ALL_B32 : f32, !pto.mask<b32> -> !pto.vreg<64xf32>

%t1_p0 = pto.vmul %w_p0, %k_p0, PAT_ALL_B32 : !pto.vreg<64xf32>
%t1_p1 = pto.vmul %w_p1, %k_p1, PAT_ALL_B32 : !pto.vreg<64xf32>

// ensure_layout for the first store.
%t1_0, %t1_1 = pto.vintlv %t1_p0, %t1_p1
  : !pto.vreg<64xf32>, !pto.vreg<64xf32>
    -> !pto.vreg<64xf32>, !pto.vreg<64xf32>
pto.vsts %t1_0, %out1[%off], %all_b32 {dist = "NORM_B32"}
pto.vsts %t1_1, %out1[%off_plus_64], %all_b32 {dist = "NORM_B32"}

// ensure_layout for the second store.
%w_0, %w_1 = pto.vintlv %w_p0, %w_p1
  : !pto.vreg<64xf32>, !pto.vreg<64xf32>
    -> !pto.vreg<64xf32>, !pto.vreg<64xf32>
pto.vsts %w_0, %out2[%off], %all_b32 {dist = "NORM_B32"}
pto.vsts %w_1, %out2[%off_plus_64], %all_b32 {dist = "NORM_B32"}
```

Memory result:

```text
for i = 0..127:
  out1[off + i] = f32(in[off + i]) * k1
  out2[off + i] = f32(in[off + i])
```

Optimization pass result:

```text
// vmi-layout-fold may remove both ensure_layout ops if the target
// supports store lowering that consumes deinterleaved=2 and writes contiguous
// row-major memory.
pto.vmi.store %t1, %out1[%off]
pto.vmi.store %w,  %out2[%off]
```

Optimized VPTO lowering result:

```text
pto.vstsx2 %t1_p0, %t1_p1, %out1[%off], "INTLV_B32", PAT_ALL_B32
  : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.ptr<f32, ub>, index,
    !pto.mask<b32>

pto.vstsx2 %w_p0, %w_p1, %out2[%off], "INTLV_B32", PAT_ALL_B32
  : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.ptr<f32, ub>, index,
    !pto.mask<b32>
```

Required assignment and optimization rule:

```text
Hard legalization may always preserve `%w` and `%t1` in deinterleaved=2 and
insert use-site ensure_layout before ordinary stores.  This is correct because
the layout change is explicit at the store use.

Consumer folding is optional.  It may remove the ensure_layout only when the
store itself can locally prove the same contiguous memory effect from the
source layout.  vmi-to-vpto must not scan the `%w` producer or both store users
to decide this.
```

### 3.47 Type-Parametric Group Reduce Rule

The group-reduce layout rule is parameterized by the element width, not by f32
case names.

```text
E          = sizeof(T)
VLaneElems = 32B / E
L          = 256B / E
S          = logical_lane_count / num_groups
```

The canonical grouped-reduce layouts are:

```text
Packed group-slot rule:
  K is the physical slot capacity of one packed group-result chunk.
  For VCG-style packed reductions, K = 8.
  G does not have to be divisible by K; the final chunk may be partial.
  active_groups(chunk c) = min(K, G - c * K).

S == VLaneElems:
  source/mask layout = contiguous
  result layout      = group_slots(num_groups=G, slots=8)

S == 2 * VLaneElems:
  source/mask layout = deinterleaved=2
  result layout      = group_slots(num_groups=G, slots=8)

S == 4 * VLaneElems:
  source/mask layout = deinterleaved=4
  result layout      = group_slots(num_groups=G, slots=8)

S >= L && S % L == 0:
  source/mask layout = contiguous
  result layout      = group_slots(num_groups=G, slots=1)
```

Concrete shape table:

```text
T     VLaneElems  L    packed cases              row-local cases
f32   8           64   S=8, S=16, S=32           S=64, S=128, ...
i32   8           64   S=8, S=16, S=32           S=64, S=128, ...
f16   16          128  S=16, S=32, S=64          S=128, S=256, ...
i16   16          128  S=16, S=32, S=64          S=128, S=256, ...
f8    32          256  cast to f32 before grouped reduce
i8    32          256  S=32, S=64, S=128          S=256, S=512, ...
```

These non-f32 cases are part of the type-generic layout/lowering design.  If a
typed reduce op admits the element type and the target capability registry
accepts it, assignment must use the same `VLaneElems/L/S` formula instead of
adding per-type shape special cases.  Any f32-only behavior in the current
implementation is staged implementation status, not the intended design limit.
For the current baseline, `f8` remains a storage and cast-boundary type for
group reduction. Integer `i8/i16/i32` grouped reductions are direct VMI
operations when their group shape matches a registered table row.

### 3.48 16-bit Typed Group Reduce, `S = VLaneElems = 16`

This case covers both `f16` and `i16`. The element width is the same, so the
layout and VPTO instruction skeleton are identical. The VMI op name carries the
semantic difference:

```text
f16: pto.vmi.group_reduce_addf ... {reassoc}
i16: pto.vmi.group_reduce_addi ...
```

VMI-shaped input:

```text
// Floating form.
%xf = pto.vmi.load %base_f16[%off]
  : memref<128xf16> -> !pto.vmi.vreg<128xf16>
%mf = pto.vmi.create_group_mask %c16 {num_groups = 8, group_size = 16}
  : index -> !pto.vmi.mask<128xpred>
%sumf = pto.vmi.group_reduce_addf %xf, %mf {num_groups = 8, reassoc}
pto.vmi.group_store %sumf, %out_f16[%group_off], %c1 {num_groups = 8}

// Integer form.
%xi = pto.vmi.load %base_i16[%off]
  : memref<128xi16> -> !pto.vmi.vreg<128xi16>
%mi = pto.vmi.create_group_mask %c16 {num_groups = 8, group_size = 16}
  : index -> !pto.vmi.mask<128xpred>
%sumi = pto.vmi.group_reduce_addi %xi, %mi {num_groups = 8}
pto.vmi.group_store %sumi, %out_i16[%group_off], %c1 {num_groups = 8}
```

Assigned layouts:

```text
%xf, %mf, %xi, %mi:
  #pto.vmi.layout<contiguous>

%sumf:
  !pto.vmi.vreg<128xf16, #pto.vmi.layout<num_groups = 8, slots = 8>>

%sumi:
  !pto.vmi.vreg<128xi16, #pto.vmi.layout<num_groups = 8, slots = 8>>
```

VPTO lowering shape:

```text
%x0 = pto.vlds %base[%off] {dist = "NORM"}
  : !pto.ptr<T16, ub> -> !pto.vreg<128xT16>

%all_b16 = pto.pge_b16 "PAT_ALL"
%slot8_b16 = pto.pge_b16 "PAT_VL8"

%sum0 = pto.vcgadd %x0, %all_b16
  : !pto.vreg<128xT16>, !pto.mask<b16> -> !pto.vreg<128xT16>

pto.vsts %sum0, %out[%group_off], %slot8_b16 {dist = "NORM_B16"}
  : !pto.vreg<128xT16>, !pto.ptr<T16, ub>, !pto.mask<b16>
```

Memory result:

```text
for r = 0..7:
  out[group_off + r] = reduce_T16(base[off + r * 16 + 0 .. 15])
```

### 3.49 16-bit Typed Group Reduce, `S = 2 * VLaneElems = 32`

This case covers both `f16` and `i16`. Each logical row is 64B and must be
split into two 32B VLane fragments before `vcgadd`.

VMI-shaped input:

```text
%x = pto.vmi.load %base[%off]
  : memref<256xT16> -> !pto.vmi.vreg<256xT16>
%mask = pto.vmi.create_group_mask %c32 {num_groups = 8, group_size = 32}
  : index -> !pto.vmi.mask<256xpred>
%sum = pto.vmi.group_reduce_add{f|i} %x, %mask {num_groups = 8}
pto.vmi.group_store %sum, %out[%group_off], %c1 {num_groups = 8}
```

Assigned layouts:

```text
%x, %mask:
  #pto.vmi.layout<deinterleaved = 2>

%sum:
  !pto.vmi.vreg<256xT16, #pto.vmi.layout<num_groups = 8, slots = 8>>
```

VPTO lowering shape:

```text
%x_p0, %x_p1 = pto.vldsx2 %base[%off], "DINTLV_B16"
  : !pto.ptr<T16, ub>, index -> !pto.vreg<128xT16>, !pto.vreg<128xT16>

%all_b16 = pto.pge_b16 "PAT_ALL"
%slot8_b16 = pto.pge_b16 "PAT_VL8"

%s0 = pto.vcgadd %x_p0, %all_b16
  : !pto.vreg<128xT16>, !pto.mask<b16> -> !pto.vreg<128xT16>
%s1 = pto.vcgadd %x_p1, %all_b16
  : !pto.vreg<128xT16>, !pto.mask<b16> -> !pto.vreg<128xT16>
%sum0 = pto.vadd %s0, %s1, %slot8_b16
  : !pto.vreg<128xT16>, !pto.vreg<128xT16>, !pto.mask<b16>
    -> !pto.vreg<128xT16>

pto.vsts %sum0, %out[%group_off], %slot8_b16 {dist = "NORM_B16"}
  : !pto.vreg<128xT16>, !pto.ptr<T16, ub>, !pto.mask<b16>
```

Memory result:

```text
for r = 0..7:
  out[group_off + r] = reduce_T16(base[off + r * 32 + 0 .. 31])
```

### 3.50 16-bit Typed Group Reduce, `S = 4 * VLaneElems = 64`

This is the four-fragment packed case for both `f16` and `i16`.

Assigned layouts:

```text
%x, %mask:
  #pto.vmi.layout<deinterleaved = 4>

%sum:
  !pto.vmi.vreg<512xT16, #pto.vmi.layout<num_groups = 8, slots = 8>>
```

VPTO lowering shape:

```text
%x_p0, %x_p1, %x_p2, %x_p3 = materialize deinterleaved=4 input
  : four !pto.vreg<128xT16>

%all_b16 = pto.pge_b16 "PAT_ALL"
%slot8_b16 = pto.pge_b16 "PAT_VL8"

%s0 = pto.vcgadd %x_p0, %all_b16 : !pto.vreg<128xT16>
%s1 = pto.vcgadd %x_p1, %all_b16 : !pto.vreg<128xT16>
%s2 = pto.vcgadd %x_p2, %all_b16 : !pto.vreg<128xT16>
%s3 = pto.vcgadd %x_p3, %all_b16 : !pto.vreg<128xT16>

%s01 = pto.vadd %s0, %s1, %slot8_b16 : !pto.vreg<128xT16>
%s23 = pto.vadd %s2, %s3, %slot8_b16 : !pto.vreg<128xT16>
%sum0 = pto.vadd %s01, %s23, %slot8_b16 : !pto.vreg<128xT16>

pto.vsts %sum0, %out[%group_off], %slot8_b16 {dist = "NORM_B16"}
  : !pto.vreg<128xT16>, !pto.ptr<T16, ub>, !pto.mask<b16>
```

Memory result:

```text
for r = 0..7:
  out[group_off + r] = reduce_T16(base[off + r * 64 + 0 .. 63])
```

#### 3.50.1 Partial Packed `S = 64` Reductions

This is the same `S = 4 * VLaneElems` lowering family as section 3.50, but it
covers `G` values that do not fill every packed group-result chunk.  The key
point is that `slots = 8` is a physical capacity, not a promise that every
chunk contains eight valid group results.

The result layout remains:

```text
!pto.vmi.vreg<(G * 64)xf16, #pto.vmi.layout<num_groups = G, slots = 8>>
```

The lowering computes per result chunk:

```text
K = 8
chunk c active groups A(c) = min(K, G - c * K)

source active lanes per deinterleaved part for chunk c:
  A(c) * VLaneElems = A(c) * 16 f16 lanes

reduce input mask:
  PAT_VL(A(c) * 16)

combine/store mask:
  PAT_VL(A(c))
```

For full chunks, `A(c) = 8`, so the reduce input mask is `PAT_ALL` for f16
and the combine/store mask is `PAT_VL8`.  For partial chunks, masks are
required for correctness.  The semantic source mask produced by
`pto.vmi.create_group_mask` must also materialize only the valid source lanes;
the reduce lowering should not treat padding lanes as active data.

##### `G = 4`: `256xf16, num_groups = 4`

VMI-shaped input:

```text
%x = pto.vmi.load %base[%off]
  : memref<256xf16> -> !pto.vmi.vreg<256xf16>
%mask = pto.vmi.create_group_mask %c64 {num_groups = 4, group_size = 64}
  : index -> !pto.vmi.mask<256xpred>
%sum = pto.vmi.group_reduce_addf %x, %mask {num_groups = 4, reassoc}
pto.vmi.group_store %sum, %out[%group_off], %c1 {num_groups = 4}
```

Assigned layouts:

```text
%x, %mask:
  #pto.vmi.layout<block_deinterleaved=4>

%sum:
  !pto.vmi.vreg<256xf16, #pto.vmi.layout<num_groups = 4, slots = 8>>
```

VPTO lowering shape for the only result chunk:

```text
%x_p0, %x_p1, %x_p2, %x_p3 = materialize block_deinterleaved=4 input
  : four !pto.vreg<128xf16>

%lane64_b16 = pto.pge_b16 "PAT_VL64"  // A * 16 = 4 * 16
%slot4_b16 = pto.pge_b16 "PAT_VL4"

%s0 = pto.vcgadd %x_p0, %lane64_b16 : !pto.vreg<128xf16>
%s1 = pto.vcgadd %x_p1, %lane64_b16 : !pto.vreg<128xf16>
%s2 = pto.vcgadd %x_p2, %lane64_b16 : !pto.vreg<128xf16>
%s3 = pto.vcgadd %x_p3, %lane64_b16 : !pto.vreg<128xf16>

%s01 = pto.vadd %s0, %s1, %slot4_b16 : !pto.vreg<128xf16>
%s23 = pto.vadd %s2, %s3, %slot4_b16 : !pto.vreg<128xf16>
%sum0 = pto.vadd %s01, %s23, %slot4_b16 : !pto.vreg<128xf16>

pto.vsts %sum0, %out[%group_off], %slot4_b16 {dist = "NORM_B16"}
  : !pto.vreg<128xf16>, !pto.ptr<f16, ub>, !pto.mask<b16>
```

Memory result:

```text
for r = 0..3:
  out[group_off + r] = reduce_f16(base[off + r * 64 + 0 .. 63])

sum0 lanes 4..127 are not semantic for this VMI result.
```

##### `G = 8`: full packed chunk

This is section 3.50.  There is one result chunk with `A = 8`:

```text
source mask       = PAT_ALL        // 8 * 16 = 128 f16 lanes
combine/store     = PAT_VL8
result layout     = group_slots(num_groups=8, slots=8)
```

##### `G = 12`: full chunk plus partial chunk

This case needs two packed result chunks:

```text
result layout = group_slots(num_groups=12, slots=8)
result arity  = ceil(12 / 8) = 2
```

Chunk 0 handles groups `0..7`:

```text
A(0) = 8
source mask   = PAT_ALL
combine/store = PAT_VL8
```

Chunk 1 handles groups `8..11`:

```text
A(1) = 4
source mask   = PAT_VL64
combine/store = PAT_VL4
```

Implementation checklist for this family:

```text
layout attr:
  slots=8 should be legal even when num_groups is not divisible by 8.
  slot_block(g) = g / 8 and slot_lane(g) = g % 8 are still well-defined.

layout assignment:
  packed VCG-style group_reduce results keep slots=8.

mask materialization:
  create_group_mask must not activate padding lanes in partial chunks.
  For chunk c, source active lanes are A(c) * VLaneElems.

vmi-to-vpto group_reduce:
  use A(c) from result layout slots and num_groups.
  combine masks use PAT_VL(A(c)).
  input vcgadd consumes the physical mask parts, which must already encode
  PAT_VL(A(c) * VLaneElems) for all-true grouped masks.

vmi-to-vpto group_store:
  use A(c) to build the store predicate.
  output group offset for chunk c is c * slots.
```

### 3.51 16-bit Typed Group Reduce, `S = L = 128`

This is the first row-local full-physical-chunk case for both `f16` and `i16`.
The canonical result is row-local `slots = 1`, not packed `slots = 8`.

VMI-shaped input:

```text
%x = pto.vmi.load %base[%off]
  : memref<1024xT16> -> !pto.vmi.vreg<1024xT16>
%mask = pto.vmi.create_group_mask %c128 {num_groups = 8, group_size = 128}
  : index -> !pto.vmi.mask<1024xpred>
%sum = pto.vmi.group_reduce_add{f|i} %x, %mask {num_groups = 8}
pto.vmi.group_store %sum, %out[%group_off], %c1 {num_groups = 8}
```

Assigned layouts:

```text
%x, %mask:
  #pto.vmi.layout<contiguous>

%sum:
  !pto.vmi.vreg<1024xT16, #pto.vmi.layout<num_groups = 8, slots = 1>>
```

VPTO lowering shape:

```text
%all_b16 = pto.pge_b16 "PAT_ALL"
%slot1_b16 = pto.pge_b16 "PAT_VL1"

// Repeated for r = 0..7.
%x_r = pto.vlds %base[%row_off_r] {dist = "NORM"}
  : !pto.ptr<T16, ub> -> !pto.vreg<128xT16>

// Floating-point keeps the same physical result type.
%sumf_r = pto.vcadd %x_r, %all_b16
  : !pto.vreg<128xf16>, !pto.mask<b16> -> !pto.vreg<128xf16>

// Integer VCADD widens internally; VMI restores the declared i16 type.
%wide_r = pto.vcadd %x_r, %all_b16
  : !pto.vreg<128xi16>, !pto.mask<b16> -> !pto.vreg<64xi32>
%sumi_r = pto.vbitcast %wide_r
  : !pto.vreg<64xi32> -> !pto.vreg<128xi16>

pto.vsts %sum{f|i}_r, %out[%group_off_plus_r], %slot1_b16 {dist = "NORM_B16"}
  : !pto.vreg<128xT16>, !pto.ptr<T16, ub>, !pto.mask<b16>
```

Memory result:

```text
for r = 0..7:
  out[group_off + r] = reduce_T16(base[off + r * 128 + 0 .. 127])
```

### 3.52 32-bit Typed Group Reduce

This case covers both `f32` and `i32`. The element width is the same, so
`VLaneElems = 8` and `L = 64` for both. Floating-point uses
`group_reduce_addf` with `reassoc`; integer uses `group_reduce_addi`.

Example for `S = 2 * VLaneElems = 16`:

```text
%x:
  !pto.vmi.vreg<128xT32, #pto.vmi.layout<deinterleaved = 2>>

%sum:
  !pto.vmi.vreg<128xT32, #pto.vmi.layout<num_groups = 8, slots = 8>>
```

VPTO lowering shape:

```text
%x_p0, %x_p1 = pto.vldsx2 %base[%off], "DINTLV_B32"
  : !pto.ptr<T32, ub>, index -> !pto.vreg<64xT32>, !pto.vreg<64xT32>

%all_b32 = pto.pge_b32 "PAT_ALL"
%slot8_b32 = pto.pge_b32 "PAT_VL8"

%s0 = pto.vcgadd %x_p0, %all_b32
  : !pto.vreg<64xT32>, !pto.mask<b32> -> !pto.vreg<64xT32>
%s1 = pto.vcgadd %x_p1, %all_b32
  : !pto.vreg<64xT32>, !pto.mask<b32> -> !pto.vreg<64xT32>
%sum0 = pto.vadd %s0, %s1, %slot8_b32
  : !pto.vreg<64xT32>, !pto.vreg<64xT32>, !pto.mask<b32>
    -> !pto.vreg<64xT32>

pto.vsts %sum0, %out[%group_off], %slot8_b32 {dist = "NORM_B32"}
  : !pto.vreg<64xT32>, !pto.ptr<T32, ub>, !pto.mask<b32>
```

The same formula gives:

```text
S=8:
  contiguous, slots=8, one vcgadd.

S=32:
  deinterleaved=4, slots=8, four vcgadd plus vadd tree.

S=64:
  contiguous, slots=1, row-local vcgadd plus vcadd.

S=128:
  contiguous, slots=1, row-local multi-chunk accumulation.
```

### 3.53 Integer Semantics And Invalid Typed Reductions

Integer group reduction is not a variant of `group_reduce_addf`; it requires a
typed integer op:

```text
%sum = pto.vmi.group_reduce_addi %x, %mask {num_groups = G}
```

Required semantics:

```text
inactive lanes contribute integer zero
addition uses the target's normal integer add behavior
wrap/saturating variants must be represented by distinct ops if both are needed
signedness does not affect add, but does affect future max/min integer reduces
```

Required invalid cases:

```text
pto.vmi.group_reduce_addf with integer element type -> verifier error
pto.vmi.group_reduce_addi with floating-point element type -> verifier error
pto.vmi.group_reduce_addi with an integer width other than i8/i16/i32
  -> verifier error
S not in {VLaneElems, 2*VLaneElems, 4*VLaneElems} and not a full-chunk multiple
  -> layout-contract diagnostic
```

### 3.54 8-bit Floating Group Reduce

There is no direct f8 `vcgadd` grouped reduction in the current target model,
but f8 supports cast to an accumulator type. The semantic path is:

```text
f8 storage -> cast/extf to f32 accumulator -> group_reduce_addf on f32
```

Here `f8` is only the cast source and the memory element type.  The reduction
itself is a f32 accumulator operation.

The group size remains a logical-lane property.  For example, reducing eight
rows of 32 f8 elements produces the same logical result as reducing eight rows
of 32 f32 accumulator elements after extension.

VMI-shaped input:

```text
%x8 = pto.vmi.load %base_f8[%off]
  : memref<256xf8> -> !pto.vmi.vreg<256xf8>
%x32 = pto.vmi.extf %x8
  : !pto.vmi.vreg<256xf8> -> !pto.vmi.vreg<256xf32>
%mask = pto.vmi.create_group_mask %c32 {num_groups = 8, group_size = 32}
  : index -> !pto.vmi.mask<256xpred>
%sum = pto.vmi.group_reduce_addf %x32, %mask {num_groups = 8, reassoc}
pto.vmi.group_store %sum, %out_f32[%group_off], %c1 {num_groups = 8}
```

Assigned layouts:

```text
%x8:
  !pto.vmi.vreg<256xf8, #pto.vmi.layout<contiguous>>

%x32, %mask:
  !pto.vmi.vreg<256xf32, #pto.vmi.layout<deinterleaved = 4>>
  !pto.vmi.mask<256xb32, #pto.vmi.layout<deinterleaved = 4>>

%sum:
  !pto.vmi.vreg<256xf32, #pto.vmi.layout<num_groups = 8, slots = 8>>
```

VPTO lowering shape:

```text
%x8_packed = pto.vlds %base_f8[%off] {dist = "NORM"}
  : !pto.ptr<f8, ub> -> !pto.vreg<256xf8>

%all_b8 = pto.pge_b8 "PAT_ALL"
%all_b32 = pto.pge_b32 "PAT_ALL"
%slot8_b32 = pto.pge_b32 "PAT_VL8"

%x32_p0 = pto.vcvt %x8_packed, %all_b8 {part = "P0"}
  : !pto.vreg<256xf8>, !pto.mask<b8> -> !pto.vreg<64xf32>
%x32_p1 = pto.vcvt %x8_packed, %all_b8 {part = "P1"}
  : !pto.vreg<256xf8>, !pto.mask<b8> -> !pto.vreg<64xf32>
%x32_p2 = pto.vcvt %x8_packed, %all_b8 {part = "P2"}
  : !pto.vreg<256xf8>, !pto.mask<b8> -> !pto.vreg<64xf32>
%x32_p3 = pto.vcvt %x8_packed, %all_b8 {part = "P3"}
  : !pto.vreg<256xf8>, !pto.mask<b8> -> !pto.vreg<64xf32>

%s0 = pto.vcgadd %x32_p0, %all_b32 : !pto.vreg<64xf32>
%s1 = pto.vcgadd %x32_p1, %all_b32 : !pto.vreg<64xf32>
%s2 = pto.vcgadd %x32_p2, %all_b32 : !pto.vreg<64xf32>
%s3 = pto.vcgadd %x32_p3, %all_b32 : !pto.vreg<64xf32>
%s01 = pto.vadd %s0, %s1, %slot8_b32 : !pto.vreg<64xf32>
%s23 = pto.vadd %s2, %s3, %slot8_b32 : !pto.vreg<64xf32>
%sum0 = pto.vadd %s01, %s23, %slot8_b32 : !pto.vreg<64xf32>

pto.vsts %sum0, %out_f32[%group_off], %slot8_b32 {dist = "NORM_B32"}
  : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.mask<b32>
```

Memory result:

```text
for r = 0..7:
  out_f32[group_off + r] =
    reduce_f32(f32(base_f8[off + r * 32 + 0 .. 31]))
```

Direct f8 grouped reduction is invalid:

```text
pto.vmi.group_reduce_addf %x8, %mask
  : !pto.vmi.vreg<256xf8>, !pto.vmi.mask<256xpred>
  -> verifier or layout-contract diagnostic
```

### 3.55 8-bit Integer Group Reduce

The target exposes same-type i8 `vcgadd` for 32B-block group classes and a
widening i8-to-i16 `vcadd` for full-row reduction. VMI keeps a same-type i8
contract in both cases:

```text
i8 source -> group_reduce_addi -> i8 group-slot result
```

Packed 32B-block example:

```text
%x8 = pto.vmi.load %base_i8[%off]
  : memref<256xi8> -> !pto.vmi.vreg<256xi8>
%mask = pto.vmi.create_group_mask %c32 {num_groups = 8, group_size = 32}
  : index -> !pto.vmi.mask<256xpred>
%sum = pto.vmi.group_reduce_addi %x8, %mask {num_groups = 8}
pto.vmi.group_store %sum, %out_i8[%group_off], %c1 {num_groups = 8}
```

Assigned layouts:

```text
%x8, %mask:
  #pto.vmi.layout<contiguous>

%sum:
  !pto.vmi.vreg<8xi8, #pto.vmi.layout<num_groups = 8, slots = 8>>
```

VPTO lowering shape:

```text
%all_b8 = pto.pge_b8 "PAT_ALL"
%slot8_b8 = pto.pge_b8 "PAT_VL8"
%sum0 = pto.vcgadd %x8, %all_b8
  : !pto.vreg<256xi8>, !pto.mask<b8> -> !pto.vreg<256xi8>
pto.vsts %sum0, %out_i8[%group_off], %slot8_b8 {dist = "NORM_B8"}
```

For an aligned full row (`S = 256`), lowering uses widening only internally:

```text
%wide = pto.vcadd %x8, %all_b8
  : !pto.vreg<256xi8>, !pto.mask<b8> -> !pto.vreg<128xi16>
%sum_i8 = pto.vbitcast %wide
  : !pto.vreg<128xi16> -> !pto.vreg<256xi8>
```

The low i8 lane contains the same-type wraparound result. Explicit
`extsi`/`extui` before reduction remains available when the algorithm itself
requires a wider accumulator, but widening is not required by the direct i8
group-reduce contract.

### 3.56 Full 256-Bin Distribution Histogram

Histogram is not modeled as `group_reduce`.  A group reduce maps source lanes to
result slots by lane/group position.  A histogram maps each active source lane
to a result bin by the source value itself.

VMI-shaped input:

```text
%src = pto.vmi.load %src_base[%src_off]
  : memref<Nxui8> -> !pto.vmi.vreg<Nxui8>
%mask = pto.vmi.create_mask %active_lanes
  : index -> !pto.vmi.mask<Nxpred>
%acc = pto.vmi.load %acc_base[%acc_off]
  : memref<256xui16> -> !pto.vmi.vreg<256xui16>
%hist = pto.vmi.vdhist %acc, %src, %mask
  : !pto.vmi.vreg<256xui16>, !pto.vmi.vreg<Nxui8>,
    !pto.vmi.mask<Nxpred> -> !pto.vmi.vreg<256xui16>
pto.vmi.store %hist, %out[%out_off]
```

Logical semantics:

```text
for b = 0..255:
  hist[b] = acc[b]

for i = 0..N-1:
  if mask[i]:
    hist[src[i]] += 1
```

Assigned layouts:

```text
%src:
  !pto.vmi.vreg<Nxui8, #pto.vmi.layout<contiguous>>

%mask:
  !pto.vmi.mask<Nxb8, #pto.vmi.layout<contiguous>>

%acc, %hist:
  !pto.vmi.vreg<256xui16, #pto.vmi.layout<contiguous>>
```

The `256xui16` accumulator/result is one logical VMI value but two physical
VPTO vector registers:

```text
physical result part0 = logical bins   0..127
physical result part1 = logical bins 128..255
```

For `N = 256`, VPTO lowering shape:

```text
%src0 = pto.vlds %src_base[%src_off] {dist = "NORM"}
  : !pto.ptr<ui8, ub> -> !pto.vreg<256xui8>

%acc_lo = pto.vlds %acc_base[%acc_off + 0] {dist = "NORM"}
  : !pto.ptr<ui16, ub> -> !pto.vreg<128xui16>
%acc_hi = pto.vlds %acc_base[%acc_off + 128] {dist = "NORM"}
  : !pto.ptr<ui16, ub> -> !pto.vreg<128xui16>

%hist_lo = pto.dhistv2 %acc_lo, %src0, %mask0, %bin0
  : !pto.vreg<128xui16>, !pto.vreg<256xui8>, !pto.mask<b8>, i32
    -> !pto.vreg<128xui16>
%hist_hi = pto.dhistv2 %acc_hi, %src0, %mask0, %bin1
  : !pto.vreg<128xui16>, !pto.vreg<256xui8>, !pto.mask<b8>, i32
    -> !pto.vreg<128xui16>

pto.vsts %hist_lo, %out[%out_off + 0],   %all_b16 {dist = "NORM_B16"}
pto.vsts %hist_hi, %out[%out_off + 128], %all_b16 {dist = "NORM_B16"}
```

Memory result:

```text
for b = 0..127:
  out[out_off + b] = acc_base[acc_off + b] +
    count(i where mask[i] && src_base[src_off + i] == b)

for b = 128..255:
  out[out_off + b] = acc_base[acc_off + b] +
    count(i where mask[i] && src_base[src_off + i] == b)
```

For `N > 256`, the source is processed in contiguous 256-lane chunks.  The two
histogram accumulator parts are carried through all chunks:

```text
%lo = %acc_lo
%hi = %acc_hi

for source chunk c in logical order:
  %chunk_mask = mask chunk c
  if c is the final partial chunk:
    %chunk_mask = %chunk_mask & valid-lane-prefix-for-this-chunk

  %lo = pto.dhistv2 %lo, %src_c, %chunk_mask, %bin0
  %hi = pto.dhistv2 %hi, %src_c, %chunk_mask, %bin1

result physical parts = [%lo, %hi]
```

Tail source lanes are expressed only through the b8 mask.  Padding lanes in the
last physical source chunk must be masked off before `pto.dhistv2`; they are
not padding values.

The VMI op does not expose `#bin`.  `#bin` is a VPTO range selector forced by
the physical result width:

```text
ui8 value domain      = 256 bins
complete histogram    = 256 x ui16 = 512B
one VPTO vreg result  = 128 x ui16 = 256B
```

Therefore VMI represents one logical `256xui16` result and `vmi-to-vpto`
locally emits the low-range and high-range VPTO histogram updates.

### 3.57 Full 256-Bin Cumulative Histogram

The desired VMI surface shape mirrors `dhist`:

```text
%hist = pto.vmi.vchist %acc, %src, %mask
  : !pto.vmi.vreg<256xui16>, !pto.vmi.vreg<Nxui8>,
    !pto.vmi.mask<Nxpred> -> !pto.vmi.vreg<256xui16>
```

The intended logical semantics is a full cumulative histogram:

```text
dist[b] = count(i where mask[i] && src[i] == b)

hist[0] = acc[0] + dist[0]
for b = 1..255:
  hist[b] = acc[b] + dist[0] + dist[1] + ... + dist[b]
```

The current VPTO/VISA documentation only states that `CHISTv2` computes a
`uint16 Cumulative histogram` over the selected bin range.  It does not state
whether the high-range call with `#bin = 1` returns:

```text
global cumulative:
  result[j] = count(src <= 128 + j)

or range-local cumulative:
  result[j] = count(128 <= src <= 128 + j)
```

These two interpretations have different VMI lowerings.  If the hardware result
is global cumulative, the full VMI lowering is the same low/high split as
`dhist`, replacing `pto.dhistv2` with `pto.chistv2`.  If the hardware result is
range-local cumulative, the high half also needs the total low-half count added
to every high-half bin:

```text
%lo = pto.chistv2 %acc_lo, %src0, %mask0, %bin0
%hi_local = pto.chistv2 %acc_hi, %src0, %mask0, %bin1

%low_total = materialize count(src <= 127) from the low-half result
%low_total_vec = broadcast %low_total to every high-half bin
%hi = pto.vadd %hi_local, %low_total_vec, %all_b16
```

That correction path also requires a designed way to materialize and broadcast
the low-half total.  Since baseline VMI does not support arbitrary vector
extract, the range-local CHISTv2 interpretation remains unsupported until that
materialization path is explicit.

The baseline design therefore treats `pto.vmi.vchist` as a semantic op whose
exact lowering is gated by a target semantic capability:

```text
if target documents or validation proves CHISTv2 high range is global:
  lower as two pto.chistv2 calls
elif target documents or validation proves CHISTv2 high range is range-local:
  lower as pto.chistv2 low/high plus explicit high-half correction only after
  low-total materialization support is designed
else:
  VMI-UNSUPPORTED: pto.vmi.vchist requires a verified CHISTv2 range semantics contract
```

This boundary is deliberate.  `pto.vmi.vdhist` is fully defined because
distribution bins are independent across the low/high split.  `pto.vmi.vchist`
has cross-range prefix semantics, so VMI must not guess the high-half behavior
from the VPTO op name alone.
