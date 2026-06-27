# VPTO vector-address op support design

## 1. Goal

This document defines the VPTO IR surface for VISA `VAG` and vector-address
load/store forms. The goal is to support the LLVM-path `vector_address`
intrinsic families without reusing overly generic op names such as `vld`, `vst`,
`pld`, or `pst`.

Naming rule:

- `va` means vector-addressed form, i.e. the memory offset operand is a
  `vector_address`/address-register offset.
- `vald`, `vast`, `pald`, and `past` are the vector-addressed counterparts of
  `vld`, `vst`, `pld`, and `pst`.
- Stateful unaligned forms return updated SSA values instead of relying on
  source-level argument mutation.

This document describes the VPTO vector-address support contract for this
change.

## 2. Type model

### 2.1 `!pto.vaddr`

```mlir
!pto.vaddr<b8>
!pto.vaddr<b16>
!pto.vaddr<b32>
```

`!pto.vaddr<G>` represents a CCE `vector_address` value. It is an offset token,
not a complete pointer. A consumer computes the effective UB address as:

```text
effective_address = base + vaddr
```

The granularity parameter records the element-width family used to create the
address. It must be one of `b8`, `b16`, or `b32`.

The value must not be treated as `!pto.ptr`. Pointer provenance and memory space
come from the explicit base operand of each memory op.

The compiler ABI represents `vector_address` as `uint32_t
__attribute__((ext_vector_type(1)))`, so the VPTO LLVM conversion type is
`<1 x i32>`.

### 2.2 Vector LLVM ABI suffixes

Vector-address memory intrinsics are typed LLVM intrinsic families. The textual
LLVM IR name carries the selected vector ABI suffix:

| VPTO element family | LLVM vector ABI | Intrinsic suffix |
| --- | --- | --- |
| `i8`/`ui8` | `<256 x i8>` | `v256i8` |
| `i16`/`ui16` | `<128 x i16>` | `v128i16` |
| `f16` | `<128 x half>` | `v128f16` |
| `bf16` | `<128 x bfloat>` | `v128bf16` |
| `i32`/`ui32` | `<64 x i32>` | `v64i32` |
| `f32` | `<64 x float>` | `v64f32` |
| `i64`/`ui64` | `<32 x i64>` | `v32i64` |

For example, a `!pto.vreg<64xf32>` vector-address load lowers to
`@llvm.hivm.vldx1.v64f32`, not to a suffix-less `@llvm.hivm.vldx1`.

### 2.3 Offset operands

`pto.vag` operands are 32-bit unsigned byte strides. CCE source wrappers such as
`vag_b16` and `vag_b32` accept element strides and multiply them by the element
byte width before calling the compiler builtin. VPTO lowers directly to LLVM IR,
so its op boundary uses byte strides and does not repeat that source-level
scaling.

### 2.4 Evidence model for LLVM lowering

Do not infer an LLVM intrinsic signature from `strings` output. `strings
$ASCEND_HOME_PATH/bin/bisheng` only proves that an intrinsic name is present in
the installed compiler binary; it does not provide argument types, result type,
or operand order.

This document uses three evidence levels:

- Installed CANN Clang headers define source wrapper signatures, builtin names,
  operand order, units, and fixed control constants such as `0 /* #loop */`.
- Generated LLVM IR or recovered AICore bitcode defines the final LLVM function
  type for a concrete installed compiler.
- `strings bisheng` is only a name inventory used to check that the intrinsic
  spelling exists in the installed compiler.

The repo lit coverage prints VPTO-generated LLVM IR and checks lowering
signatures for the implemented forms. VAG is special: the VPTO direct LLVM input
uses the same no-IV builtin shape exposed by the CCE source wrapper, and the
call must remain inside a loop so later CCE middle-end passes can bind it to
loop state.

### 2.5 Canonical signatures

The table below is the implementation contract for the A5 VPTO LLVM path. `V`
means the LLVM vector ABI selected from Section 2.2, `S` means the matching
typed intrinsic suffix, and all pointers are UB pointers lowered to
`ptr addrspace(6)`.

The VPTO column is intentionally written in the same assembly order as the ODS
`assemblyFormat`. Attributes shown inside `{...}` are named attributes.
Quoted operands such as `"D"` and `"POST_UPDATE"` are required string
attributes printed positionally by that op.

| VPTO op signature | LLVM lowering signature |
| --- | --- |
| `%addr = pto.vag %s... : i32 -> !pto.vaddr<G>` | `declare <1 x i32> @llvm.hivm.vag.32(i32, i32, i32, i32)` |
| `%v = pto.vald %base[%addr] {dist = "D"} : !pto.ptr<T, ub>, !pto.vaddr<G> -> !pto.vreg<NxT>` | `declare V @llvm.hivm.vldx1.S(ptr addrspace(6) nocapture readonly, <1 x i32>, i32, i32)` |
| `%lo, %hi = pto.valdx2 %base[%addr], "D" : !pto.ptr<T, ub>, !pto.vaddr<G> -> !pto.vreg<NxT>, !pto.vreg<NxT>` | `declare { V, V } @llvm.hivm.vldx2.S(ptr addrspace(6) nocapture readonly, <1 x i32>, i32, i32)` |
| `pto.vast %v, %base[%addr], %mask {dist = "D"} : !pto.vreg<NxT>, !pto.ptr<T, ub>, !pto.vaddr<G>, !pto.mask<M>` | `declare void @llvm.hivm.vstx1.S(V, ptr addrspace(6) nocapture writeonly, <1 x i32>, i32, i32, <256 x i1>)` |
| `pto.vastx2 %lo, %hi, %base[%addr], "D", %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.ptr<T, ub>, !pto.vaddr<G>, !pto.mask<M>` | `declare void @llvm.hivm.vstx2.S(V, V, ptr addrspace(6) nocapture writeonly, <1 x i32>, i32, i32, <256 x i1>)` |
| `%mask = pto.pald %base[%addr], "D" : !pto.ptr<i32, ub>, !pto.vaddr<G> -> !pto.mask<M>` | `declare <256 x i1> @llvm.hivm.pld.b8(ptr addrspace(6) nocapture readonly, <1 x i32>, i32, i32)` |
| `pto.past %mask, %base[%addr], "D" : !pto.mask<M>, !pto.ptr<i32, ub>, !pto.vaddr<G>` | `declare void @llvm.hivm.pst.b8(<256 x i1>, ptr addrspace(6) nocapture writeonly, <1 x i32>, i32, i32)` |
| `%align = pto.valda %base[%addr] : !pto.ptr<T, ub>, !pto.vaddr<G> -> !pto.align` | `declare <32 x i8> @llvm.hivm.vlda(ptr addrspace(6), <1 x i32>, i32)` |
| `%v, %align1, %addr1 = pto.valdu %base[%addr0], %align0, %inc : !pto.ptr<T, ub>, !pto.vaddr<G>, !pto.align, i32 -> !pto.vreg<NxT>, !pto.align, !pto.vaddr<G>` | `declare { V, <32 x i8>, <1 x i32> } @llvm.hivm.vldu.v300.S(ptr addrspace(6) nocapture readonly, <1 x i32>, <32 x i8>, i32, i32)` |
| `pto.vasta %align, %base[%addr] : !pto.align, !pto.ptr<T, ub>, !pto.vaddr<G>` | `declare void @llvm.hivm.vsta(<32 x i8>, ptr addrspace(6) nocapture writeonly, <1 x i32>, i32)` |
| `%align1, %addr1 = pto.vastu %align0, %addr0, %v, %base, "POST_UPDATE" : !pto.align, !pto.vaddr<G>, !pto.vreg<NxT>, !pto.ptr<T, ub> -> !pto.align, !pto.vaddr<G>` | `declare { <32 x i8>, <1 x i32> } @llvm.hivm.vstu.S(V, ptr addrspace(6) nocapture writeonly, <1 x i32>, <32 x i8>, i32, i32)` |

All LLVM lowering definitions below use the same declaration shapes. Aggregate
returns such as `{ V, V }` may appear as `!llvm.struct<(...)>` in MLIR LLVM
dialect dumps; that is the same final LLVM IR aggregate-return declaration. The
final control operand named loop mode is currently emitted as `i32 0`.

### 2.6 Immediate tokens used by these signatures

The current A5 lowering accepts the following string tokens and emits these
integer immediates. These are part of the lowering contract for this change.

| VPTO attribute | Legal tokens and emitted `i32` values |
| --- | --- |
| `pto.vald` `dist` | omitted/`"NORM"` -> `0`; `"BRC_B8"` -> `1`; `"BRC_B16"` -> `2`; `"BRC_B32"` -> `3`; `"US_B8"` -> `6`; `"US_B16"` -> `7`; `"DS_B8"` -> `8`; `"DS_B16"` -> `9`; `"UNPK_B8"` -> `13`; `"UNPK_B16"` -> `14`; `"UNPK_B32"` -> `18`; `"BRC_BLK"` -> `15`; `"E2B_B16"` -> `16`; `"E2B_B32"` -> `17`; `"UNPK4"` -> `20` for b8 element vectors only; `"SPLT4CHN"` -> `21` for b8 element vectors only; `"SPLT2CHN_B8"` -> `22`; `"SPLT2CHN_B16"` -> `23` |
| `pto.valdx2` `"D"` | `"BDINTLV"` -> `10`; `"DINTLV_B8"` -> `11`; `"DINTLV_B16"` -> `12`; `"DINTLV_B32"` -> `19` |
| `pto.vast` `dist` | omitted -> element-width default (`b8` -> `0`, `b16` -> `1`, `b32` -> `2`); `"NORM_B8"` -> `0`; `"NORM_B16"` -> `1`; `"NORM_B32"` -> `2`; `"1PT_B8"` -> `3`; `"1PT_B16"` -> `4`; `"1PT_B32"` -> `5`; `"PK_B16"` -> `6`; `"PK_B32"` -> `7`; `"PK_B64"` -> `10`; `"PK4_B32"` -> `12`; `"MRG4CHN_B8"` -> `13`; `"MRG2CHN_B8"` -> `14`; `"MRG2CHN_B16"` -> `15` |
| `pto.vastx2` `"D"` | `"INTLV_B8"` -> `8`; `"INTLV_B16"` -> `9`; `"INTLV_B32"` -> `11` |
| `pto.pald` `"D"` | `"NORM"` -> `0`; `"US"` -> `1`; `"DS"` -> `2` |
| `pto.past` `"D"` | `"NORM"` -> `0`; `"PK"` -> `1` |
| `pto.vastu` `"MODE"` | `"POST_UPDATE"` -> `1` |

### 2.7 Implementation signature ledger

This section is the implementation checklist for the current VPTO lowering.
Every operand is listed in VPTO assembly order first, then in the LLVM call
order emitted by `VPTOCANN900LLVMEmitter`. The LLVM snippets in this section
use aliases for compactness instead of literal copy-paste LLVM IR. `V` means
the payload LLVM vector ABI from Section 2.2, `A` means `<1 x i32>`
vector-address ABI, `P` means `ptr addrspace(6)`, `M` means `<256 x i1>`, and
`L` means `<32 x i8>`.

#### 2.7.1 `pto.vag`

VPTO:

```text
%addr = pto.vag %s0 : i32 -> !pto.vaddr<G>
```

Current A5 LLVM lowering:

```text
%addr = call A @llvm.hivm.vag.32(i32 %s0, i32 0, i32 0, i32 0)
```

Declaration:

```llvm
declare <1 x i32> @llvm.hivm.vag.32(i32, i32, i32, i32)
```

Inactive dimensions are emitted as `i32 0`. `pto.vag` must be nested under an
`i16` `scf.for`; VPTO does not synthesize that loop.

#### 2.7.2 `pto.vald`

VPTO:

```text
%value = pto.vald %base[%addr] {dist = "DIST"}
  : !pto.ptr<T, ub>, !pto.vaddr<G> -> !pto.vreg<NxT>
```

LLVM:

```text
declare V @llvm.hivm.vldx1.S(P readonly, A, i32 /*dist*/, i32 /*loop*/)
```

Emitted call operands are `%base, %addr, dist_code, i32 0`.

#### 2.7.3 `pto.valdx2`

VPTO:

```text
%lo, %hi = pto.valdx2 %base[%addr], "DIST"
  : !pto.ptr<T, ub>, !pto.vaddr<G> -> !pto.vreg<NxT>, !pto.vreg<NxT>
```

LLVM:

```text
declare { V, V } @llvm.hivm.vldx2.S(
  P readonly, A, i32 /*dist*/, i32 /*loop*/)
```

Emitted call operands are `%base, %addr, dist_code, i32 0`. Aggregate result
index 0 maps to `%lo`; index 1 maps to `%hi`.

#### 2.7.4 `pto.vast`

VPTO:

```text
pto.vast %value, %base[%addr], %mask {dist = "DIST"}
  : !pto.vreg<NxT>, !pto.ptr<T, ub>, !pto.vaddr<G>, !pto.mask<M>
```

LLVM:

```text
declare void @llvm.hivm.vstx1.S(
  V, P writeonly, A, i32 /*dist*/, i32 /*loop*/, M)
```

Emitted call operands are `%value, %base, %addr, dist_code, i32 0, %mask`.

#### 2.7.5 `pto.vastx2`

VPTO:

```text
pto.vastx2 %lo, %hi, %base[%addr], "DIST", %mask
  : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.ptr<T, ub>,
    !pto.vaddr<G>, !pto.mask<M>
```

LLVM:

```text
declare void @llvm.hivm.vstx2.S(
  V, V, P writeonly, A, i32 /*dist*/, i32 /*loop*/, M)
```

Emitted call operands are `%lo, %hi, %base, %addr, dist_code, i32 0, %mask`.

#### 2.7.6 `pto.pald`

VPTO:

```text
%mask = pto.pald %base[%addr], "DIST"
  : !pto.ptr<i32, ub>, !pto.vaddr<G> -> !pto.mask<M>
```

LLVM:

```text
declare M @llvm.hivm.pld.b8(P readonly, A, i32 /*dist*/, i32 /*loop*/)
```

Emitted call operands are `%base, %addr, dist_code, i32 0`.

#### 2.7.7 `pto.past`

VPTO:

```text
pto.past %mask, %base[%addr], "DIST"
  : !pto.mask<M>, !pto.ptr<i32, ub>, !pto.vaddr<G>
```

LLVM:

```text
declare void @llvm.hivm.pst.b8(M, P writeonly, A, i32 /*dist*/, i32 /*loop*/)
```

Emitted call operands are `%mask, %base, %addr, dist_code, i32 0`.

#### 2.7.8 `pto.valda`

VPTO:

```text
%align = pto.valda %base[%addr]
  : !pto.ptr<T, ub>, !pto.vaddr<G> -> !pto.align
```

LLVM:

```text
declare L @llvm.hivm.vlda(P, A, i32 /*loop*/)
```

Emitted call operands are `%base, %addr, i32 0`.

#### 2.7.9 `pto.valdu`

VPTO:

```text
%value, %align_out, %addr_out =
  pto.valdu %base[%addr_in], %align_in, %inc
  : !pto.ptr<T, ub>, !pto.vaddr<G>, !pto.align, i32
    -> !pto.vreg<NxT>, !pto.align, !pto.vaddr<G>
```

LLVM:

```text
declare { V, L, A } @llvm.hivm.vldu.v300.S(
  P readonly, A, L, i32 /*inc*/, i32 /*loop*/)
```

Emitted call operands are `%base, %addr_in, %align_in, %inc, i32 0`.
Aggregate result indexes map to `%value`, `%align_out`, `%addr_out` in that
order.

#### 2.7.10 `pto.vasta`

VPTO:

```text
pto.vasta %align, %base[%addr]
  : !pto.align, !pto.ptr<T, ub>, !pto.vaddr<G>
```

LLVM:

```text
declare void @llvm.hivm.vsta(L, P writeonly, A, i32 /*loop*/)
```

Emitted call operands are `%align, %base, %addr, i32 0`.

#### 2.7.11 `pto.vastu`

VPTO:

```text
%align_out, %addr_out =
  pto.vastu %align_in, %addr_in, %value, %base, "POST_UPDATE"
  : !pto.align, !pto.vaddr<G>, !pto.vreg<NxT>, !pto.ptr<T, ub>
    -> !pto.align, !pto.vaddr<G>
```

LLVM:

```text
declare { L, A } @llvm.hivm.vstu.S(
  V, P writeonly, A, L, i32 /*mode*/, i32 /*loop*/)
```

Emitted call operands are `%value, %base, %addr_in, %align_in, i32 1, i32 0`.
Aggregate result index 0 maps to `%align_out`; index 1 maps to `%addr_out`.

## 3. Address generation

### 3.1 `pto.vag`

```mlir
%addr = pto.vag %s1 : i32 -> !pto.vaddr<G>
```

Semantics:

```text
addr = s1 * i1
```

The stride is in bytes. VISA supports up to four VAG stride registers for
nested loop layers, but this VPTO implementation currently exposes only the
one-stride form until nested vector-loop IVs are represented in lowering state.

Verifier constraints:

- Operand count must be exactly 1.
- The operand must be `i32`.
- Result type must be `!pto.vaddr<b8>`, `!pto.vaddr<b16>`, or
  `!pto.vaddr<b32>`.
- VPTO does not currently enforce a source-position restriction. The intended
  use is to define the address pattern once for the vector scope and not mutate
  it differently across dynamic vector-loop iterations.

LLVM lowering:

| Result type | LLVM intrinsic family |
| --- | --- |
| `!pto.vaddr<b8>` | `llvm.hivm.vag.32` on A5 32-bit VAG ABI targets |
| `!pto.vaddr<b16>` | `llvm.hivm.vag.32` on A5 32-bit VAG ABI targets |
| `!pto.vaddr<b32>` | `llvm.hivm.vag.32` on A5 32-bit VAG ABI targets |

The installed CANN 9.0.0 source wrapper accepts element strides and calls the
compiler builtin in reverse byte-stride order:

```llvm
%addr = call <1 x i32> @llvm.hivm.vag.32(
    i32 %s4, i32 %s3, i32 %s2, i32 %s1)

declare <1 x i32> @llvm.hivm.vag.32(i32, i32, i32, i32)
```

The VPTO LLVM path emits this no-IV form directly. The call must be nested under
an `i16` `scf.for` in VPTO IR so CCE middle-end VAG lowering can associate it
with loop state before object generation. In MLIR assembly, non-`index`
`scf.for` loops require an explicit loop type marker, for example:

```mlir
%c0_i16 = arith.constant 0 : i16
%c1_i16 = arith.constant 1 : i16
%c2_i16 = arith.constant 2 : i16
scf.for %i = %c0_i16 to %c2_i16 step %c1_i16 : i16 {
  %addr = pto.vag %stride : i32 -> !pto.vaddr<b32>
}
```

Normal vector-address load/store operations may share one `!pto.vaddr` value.
Update forms that return a next vector address, currently `pto.valdu` and
`pto.vastu`, are different: one `!pto.vaddr` value must not seed multiple
update chains. Generate a separate `pto.vag` for independent update chains.

Lowering operands:

| LLVM operand | VPTO source | Value |
| --- | --- | --- |
| arg0 | `%s1` | byte stride for the innermost active loop layer |
| arg1 | `%s2` | byte stride for the next loop layer, or `i32 0` |
| arg2 | `%s3` | byte stride for the next loop layer, or `i32 0` |
| arg3 | `%s4` | byte stride for the next loop layer, or `i32 0` |

Examples:

```llvm
; pto.vag %s1
call <1 x i32> @llvm.hivm.vag.32(i32 %s1, i32 0, i32 0, i32 0)

```

The installed CANN 9.0.0 `bisheng` name inventory contains these VAG intrinsic
spellings:

```text
llvm.hivm.vag.16
llvm.hivm.vag.32
llvm.hivm.vag.iv.16
llvm.hivm.vag.iv.16.se
llvm.hivm.vag.iv.32
llvm.hivm.vag.iv.32.se
llvm.hivm.vag.v210
```

For non-A5 targets that still use 16-bit or v210 VAG ABI, the target lowering
may select a different builtin family, but the VPTO op contract remains
byte-stride based. Do not lower `pto.vag` through source-level wrapper emission.

## 4. Aligned vector load/store

### 4.1 `pto.vald`

```mlir
%result = pto.vald %base[%addr] {dist = "DIST"}
  : !pto.ptr<T, ub>, !pto.vaddr<G> -> !pto.vreg<NxT>
```

Semantics:

```text
result = load_vector(base + addr, dist)
```

Verifier constraints:

- `%base` must be a UB pointer or UB memref lowered to a VPTO pointer.
- `%addr` must be `!pto.vaddr<G>`.
- `G` must match the element or distribution granularity required by `DIST`.
- `DIST` follows the existing `pto.vlds` distribution vocabulary.

LLVM lowering:

```llvm
%result = call <64 x float> @llvm.hivm.vldx1.v64f32(
    ptr addrspace(6) %base, <1 x i32> %addr, i32 %dist, i32 0)

declare <64 x float> @llvm.hivm.vldx1.v64f32(
    ptr addrspace(6) nocapture readonly, <1 x i32>, i32, i32)
```

Select the intrinsic suffix from the result vector ABI in Section 2.2. The
installed CANN 9.0.0 LLVM evidence also includes `v32i64`, `v64i32`,
`v128bf16`, `v128f16`, `v128i16`, and `v256i8` overloads with the same operand
list.

Lowering operands:

| LLVM operand | VPTO source | Value |
| --- | --- | --- |
| arg0 | `%base` | UB pointer converted to `ptr addrspace(6)` |
| arg1 | `%addr` | `!pto.vaddr<G>` converted to `<1 x i32>` |
| arg2 | `DIST` | existing `vlds` distribution enum code, passed as `i32` |
| arg3 | implicit loop mode | `i32 0`, matching `0 /* #loop */` in the CANN wrapper |

### 4.2 `pto.valdx2`

```mlir
%low, %high = pto.valdx2 %base[%addr], "DIST"
  : !pto.ptr<T, ub>, !pto.vaddr<G> -> !pto.vreg<NxT>, !pto.vreg<NxT>
```

Semantics:

```text
(low, high) = load_vector_pair(base + addr, dist)
```

Verifier constraints:

- Same base/address constraints as `pto.vald`.
- `DIST` must be a distribution that produces two destination vector registers,
  such as de-interleave/block-deinterleave forms already accepted by
  `pto.vldsx2`.

LLVM lowering:

```llvm
%pair = call { <64 x float>, <64 x float> } @llvm.hivm.vldx2.v64f32(
    ptr addrspace(6) %base, <1 x i32> %addr, i32 %dist, i32 0)

declare { <64 x float>, <64 x float> } @llvm.hivm.vldx2.v64f32(
    ptr addrspace(6) nocapture readonly, <1 x i32>, i32, i32)
```

Select the intrinsic suffix from the result vector ABI in Section 2.2. The
`v64f32` declaration above matches recovered Bisheng LLVM IR as
`@llvm.hivm.vldx2.v64f32`.

Lowering operands:

| LLVM operand | VPTO source | Value |
| --- | --- | --- |
| arg0 | `%base` | UB pointer converted to `ptr addrspace(6)` |
| arg1 | `%addr` | `!pto.vaddr<G>` converted to `<1 x i32>` |
| arg2 | `DIST` | existing pair-load distribution enum code, passed as `i32` |
| arg3 | implicit loop mode | `i32 0`, matching `0 /* #loop */` in the CANN wrapper |

Return extraction:

| Aggregate index | VPTO result | Value |
| --- | --- | --- |
| 0 | `%low` | first loaded vector |
| 1 | `%high` | second loaded vector |

The installed CANN wrapper stores the builtin result in `vector_<type>x2_t` and
then assigns `dst0 = ret.val[0]`, `dst1 = ret.val[1]`; the VPTO lowering
preserves that order.

### 4.3 `pto.vast`

```mlir
pto.vast %value, %base[%addr], %mask {dist = "DIST"}
  : !pto.vreg<NxT>, !pto.ptr<T, ub>, !pto.vaddr<G>, !pto.mask<M>
```

Semantics:

```text
store_vector(base + addr, value, mask, dist)
```

Verifier constraints:

- `%base` must be a UB pointer or UB memref lowered to a VPTO pointer.
- `%addr` must be `!pto.vaddr<G>`.
- `%mask` granularity must be legal for `%value` and `DIST`.
- `DIST` follows the existing `pto.vsts` distribution vocabulary.

LLVM lowering:

```llvm
call void @llvm.hivm.vstx1.v64f32(
    <64 x float> %value, ptr addrspace(6) %base, <1 x i32> %addr,
    i32 %dist, i32 0, <256 x i1> %mask)

declare void @llvm.hivm.vstx1.v64f32(
    <64 x float>, ptr addrspace(6) nocapture writeonly,
    <1 x i32>, i32, i32, <256 x i1>)
```

Select the intrinsic suffix from the source vector ABI in Section 2.2. The
installed CANN 9.0.0 LLVM evidence also includes `v32i64`, `v64i32`,
`v128bf16`, `v128f16`, `v128i16`, and `v256i8` overloads with the same operand
list.

Lowering operands:

| LLVM operand | VPTO source | Value |
| --- | --- | --- |
| arg0 | `%value` | vector value converted to the selected `<N x T>` LLVM vector ABI |
| arg1 | `%base` | UB pointer converted to `ptr addrspace(6)` |
| arg2 | `%addr` | `!pto.vaddr<G>` converted to `<1 x i32>` |
| arg3 | `DIST` | existing `vsts` distribution enum code, passed as `i32` |
| arg4 | implicit loop mode | `i32 0`, matching `0 /* #loop */` in the CANN wrapper |
| arg5 | `%mask` | predicate converted to `<256 x i1>` |

### 4.4 `pto.vastx2`

```mlir
pto.vastx2 %low, %high, %base[%addr], "DIST", %mask
  : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.ptr<T, ub>,
    !pto.vaddr<G>, !pto.mask<M>
```

Semantics:

```text
store_vector_pair(base + addr, low, high, mask, dist)
```

Verifier constraints:

- Same base/address/mask constraints as `pto.vast`.
- `DIST` must be a distribution that consumes two source vector registers,
  matching the existing `pto.vstsx2` distribution set.

LLVM lowering:

```llvm
call void @llvm.hivm.vstx2.v64i32(
    <64 x i32> %low, <64 x i32> %high, ptr addrspace(6) %base,
    <1 x i32> %addr, i32 %dist, i32 0, <256 x i1> %mask)

declare void @llvm.hivm.vstx2.v64i32(
    <64 x i32>, <64 x i32>, ptr addrspace(6) nocapture writeonly,
    <1 x i32>, i32, i32, <256 x i1>)
```

Select the intrinsic suffix from the source vector ABI in Section 2.2, but only
for overloads accepted by the installed compiler. CANN 9.0.0 LLVM evidence
captures `v64i32`, `v128bf16`, `v128f16`, `v128i16`, and `v256i8` `vstx2`
overloads; it does not expose a `v64f32` store-pair overload even though the
source wrapper inventory contains a `vldx2_v64f32` load-pair builtin.

Lowering operands:

| LLVM operand | VPTO source | Value |
| --- | --- | --- |
| arg0 | `%low` | first source vector converted to the selected `<N x T>` LLVM vector ABI |
| arg1 | `%high` | second source vector converted to the selected `<N x T>` LLVM vector ABI |
| arg2 | `%base` | UB pointer converted to `ptr addrspace(6)` |
| arg3 | `%addr` | `!pto.vaddr<G>` converted to `<1 x i32>` |
| arg4 | `DIST` | existing pair-store distribution enum code, passed as `i32` |
| arg5 | implicit loop mode | `i32 0`, matching `0 /* #loop */` in the CANN wrapper |
| arg6 | `%mask` | predicate converted to `<256 x i1>` |

## 5. Predicate load/store

### 5.1 `pto.pald`

```mlir
%mask = pto.pald %base[%addr], "DIST"
  : !pto.ptr<i32, ub>, !pto.vaddr<G> -> !pto.mask<M>
```

Semantics:

```text
mask = load_predicate(base + addr, dist)
```

Verifier constraints:

- `%base` must be a UB pointer. The CCE wrapper uses `__ubuf__ uint32_t *`.
- `%addr` must be `!pto.vaddr<G>`.
- `DIST` must be one of the predicate load distributions supported by the
  existing predicate load surface.

LLVM lowering:

```llvm
%mask = call <256 x i1> @llvm.hivm.pld.b8(
    ptr addrspace(6) %base, <1 x i32> %addr, i32 %dist, i32 0)

declare <256 x i1> @llvm.hivm.pld.b8(
    ptr addrspace(6) nocapture readonly, <1 x i32>, i32, i32)
```

Lowering operands:

| LLVM operand | VPTO source | Value |
| --- | --- | --- |
| arg0 | `%base` | UB `uint32_t` pointer converted to `ptr addrspace(6)` |
| arg1 | `%addr` | `!pto.vaddr<G>` converted to `<1 x i32>` |
| arg2 | `DIST` | predicate-load distribution enum code, passed as `i32` |
| arg3 | implicit loop mode | `i32 0`, matching `0 /* #loop */` in the CANN wrapper |

The declaration above is captured in the local CANN 9.0.0 LLVM evidence.

### 5.2 `pto.past`

```mlir
pto.past %mask, %base[%addr], "DIST"
  : !pto.mask<M>, !pto.ptr<i32, ub>, !pto.vaddr<G>
```

Semantics:

```text
store_predicate(base + addr, mask, dist)
```

Verifier constraints:

- `%base` must be a UB pointer. The CCE wrapper uses `__ubuf__ uint32_t *`.
- `%addr` must be `!pto.vaddr<G>`.
- `DIST` must be one of the predicate store distributions supported by the
  existing predicate store surface.

LLVM lowering:

```llvm
call void @llvm.hivm.pst.b8(
    <256 x i1> %mask, ptr addrspace(6) %base, <1 x i32> %addr,
    i32 %dist, i32 0)

declare void @llvm.hivm.pst.b8(
    <256 x i1>, ptr addrspace(6) nocapture writeonly,
    <1 x i32>, i32, i32)
```

Lowering operands:

| LLVM operand | VPTO source | Value |
| --- | --- | --- |
| arg0 | `%mask` | predicate converted to `<256 x i1>` |
| arg1 | `%base` | UB `uint32_t` pointer converted to `ptr addrspace(6)` |
| arg2 | `%addr` | `!pto.vaddr<G>` converted to `<1 x i32>` |
| arg3 | `DIST` | predicate-store distribution enum code, passed as `i32` |
| arg4 | implicit loop mode | `i32 0`, matching `0 /* #loop */` in the CANN wrapper |

The installed CANN 9.0.0 Clang header maps the source-level vector-address form
to `__builtin_cce_pst_b8(src, base, offset, dist, 0 /* #loop */)`. The
declaration above follows the same argument order and UB base memory effect.

## 6. Unaligned vector load/store

### 6.1 `pto.valda`

```mlir
%align = pto.valda %base[%addr]
  : !pto.ptr<T, ub>, !pto.vaddr<G> -> !pto.align
```

Semantics:

```text
align = init_load_alignment(base + addr)
```

LLVM lowering:

```llvm
%align = call <32 x i8> @llvm.hivm.vlda(
    ptr addrspace(6) %base, <1 x i32> %addr, i32 0)

declare <32 x i8> @llvm.hivm.vlda(ptr addrspace(6), <1 x i32>, i32)
```

Lowering operands:

| LLVM operand | VPTO source | Value |
| --- | --- | --- |
| arg0 | `%base` | UB pointer converted to `ptr addrspace(6)` |
| arg1 | `%addr` | `!pto.vaddr<G>` converted to `<1 x i32>` |
| arg2 | implicit loop mode | `i32 0`, matching `0 /* #loop */` in the CANN wrapper |

Observed LLVM intrinsic family:

```text
llvm.hivm.vlda
```

The declaration above is captured in the local CANN 9.0.0 LLVM evidence.

### 6.2 `pto.valdu`

```mlir
%value, %align_out, %addr_out = pto.valdu %base[%addr_in],
    %align_in, %inc
  : !pto.ptr<T, ub>, !pto.vaddr<G>, !pto.align, i32
    -> !pto.vreg<NxT>, !pto.align, !pto.vaddr<G>
```

Semantics:

```text
(value, align_out, addr_out) =
    unaligned_load(base, addr_in, align_in, inc)
```

`inc` is a byte increment. `addr_out` represents the post-updated
`vector_address` value.

Verifier constraints:

- `%base` must be a UB pointer.
- `%addr_in` and `%addr_out` must have the same `!pto.vaddr<G>` type.
- `%align_in` must be `!pto.align`.
- `%inc` must be `i32`.

LLVM lowering:

```llvm
%triple = call { <64 x float>, <32 x i8>, <1 x i32> }
    @llvm.hivm.vldu.v300.v64f32(
        ptr addrspace(6) %base, <1 x i32> %addr_in,
        <32 x i8> %align_in, i32 %inc, i32 0)

declare { <64 x float>, <32 x i8>, <1 x i32> }
    @llvm.hivm.vldu.v300.v64f32(
        ptr addrspace(6) nocapture readonly, <1 x i32>,
        <32 x i8>, i32, i32)
```

Select the intrinsic suffix from the loaded vector ABI in Section 2.2. The
installed CANN 9.0.0 LLVM evidence also includes `v32i64`, `v64i32`,
`v128bf16`, `v128f16`, `v128i16`, and `v256i8` overloads with the same operand
list and aggregate return layout.

The lowering extracts the loaded vector, updated alignment value, and updated
vector-address value from the returned aggregate.

Lowering operands:

| LLVM operand | VPTO source | Value |
| --- | --- | --- |
| arg0 | `%base` | UB pointer converted to `ptr addrspace(6)` |
| arg1 | `%addr_in` | input `!pto.vaddr<G>` converted to `<1 x i32>` |
| arg2 | `%align_in` | input `!pto.align` converted to `<32 x i8>` |
| arg3 | `%inc` | byte increment, passed as `i32` |
| arg4 | implicit loop mode | `i32 0`, matching `0 /* #loop */` in the CANN wrapper |

Return extraction:

| Aggregate index | VPTO result | Value |
| --- | --- | --- |
| 0 | `%value` | loaded vector |
| 1 | `%align_out` | updated load-alignment state |
| 2 | `%addr_out` | post-updated vector-address offset token |

### 6.3 `pto.vasta`

```mlir
pto.vasta %align, %base[%addr]
  : !pto.align, !pto.ptr<T, ub>, !pto.vaddr<G>
```

Semantics:

```text
flush_store_alignment(base + addr, align)
```

LLVM lowering:

```llvm
call void @llvm.hivm.vsta(
    <32 x i8> %align, ptr addrspace(6) %base,
    <1 x i32> %addr, i32 0)

declare void @llvm.hivm.vsta(
    <32 x i8>, ptr addrspace(6) nocapture writeonly, <1 x i32>, i32)
```

Lowering operands:

| LLVM operand | VPTO source | Value |
| --- | --- | --- |
| arg0 | `%align` | store-alignment state converted to `<32 x i8>` |
| arg1 | `%base` | UB pointer converted to `ptr addrspace(6)` |
| arg2 | `%addr` | `!pto.vaddr<G>` converted to `<1 x i32>` |
| arg3 | implicit loop mode | `i32 0`, matching `0 /* #loop */` in the CANN wrapper |

### 6.4 `pto.vastu`

```mlir
%align_out, %addr_out = pto.vastu %align_in, %addr_in, %value,
    %base, "POST_UPDATE"
  : !pto.align, !pto.vaddr<G>, !pto.vreg<NxT>, !pto.ptr<T, ub>
    -> !pto.align, !pto.vaddr<G>
```

Semantics:

```text
(align_out, addr_out) =
    unaligned_store_post_update(base, addr_in, align_in, value)
```

`pto.vastu` is the vector-address stateful unaligned store. Its address state
is the `!pto.vaddr<G>` offset token, not a post-updated base pointer. The A5
CANN 9.0.0 wrapper accepts only `POST_UPDATE` for this vector-address form, so
the VPTO op exposes only `"POST_UPDATE"` rather than an arbitrary integer post
amount. A no-post scalar-offset store uses the scalar-offset stateful store
family instead.

Verifier constraints:

- `%base` must be a UB pointer.
- `%addr_in` and `%addr_out` must have the same `!pto.vaddr<G>` type.
- `%align_in` and `%align_out` must be `!pto.align`.
- The mode attribute must be `"POST_UPDATE"` on A5.

LLVM lowering:

```llvm
%pair = call { <32 x i8>, <1 x i32> } @llvm.hivm.vstu.v64f32(
    <64 x float> %value, ptr addrspace(6) %base, <1 x i32> %addr_in,
    <32 x i8> %align_in, i32 1, i32 0)
%align_out = extractvalue { <32 x i8>, <1 x i32> } %pair, 0
%addr_out = extractvalue { <32 x i8>, <1 x i32> } %pair, 1

declare { <32 x i8>, <1 x i32> } @llvm.hivm.vstu.v64f32(
    <64 x float>, ptr addrspace(6) nocapture writeonly,
    <1 x i32>, <32 x i8>, i32, i32)
```

Select the intrinsic suffix from the source vector ABI in Section 2.2. The
installed compiler canonicalizes a handwritten suffix-less `@llvm.hivm.vstu`
declaration to the typed textual IR name `@llvm.hivm.vstu.v64f32` for the f32
case.

Lowering operands:

| LLVM operand | VPTO source | Value |
| --- | --- | --- |
| arg0 | `%value` | source vector converted to the selected `<N x T>` LLVM vector ABI |
| arg1 | `%base` | UB pointer converted to `ptr addrspace(6)` |
| arg2 | `%addr_in` | input `!pto.vaddr<G>` converted to `<1 x i32>` |
| arg3 | `%align_in` | input `!pto.align` converted to `<32 x i8>` |
| arg4 | `"POST_UPDATE"` | `i32 1`, matching `1 /*post update mode*/` in the CANN wrapper |
| arg5 | implicit loop mode | `i32 0`, matching `0 /*loop*/` in the CANN wrapper |

Return extraction:

| Aggregate index | VPTO result | Value |
| --- | --- | --- |
| 0 | `%align_out` | updated store-alignment state |
| 1 | `%addr_out` | post-updated vector-address offset token |

The operand order and result aggregate are taken from the installed CANN 9.0.0
Clang wrapper for `vstu`: it creates a return object with `alignData` followed
by `offset`, then calls `__builtin_cce_vstu_<type>(&ret, src, base, offset,
alignData, 1 /*post update mode*/, 0 /*loop*/)`. The declaration above follows
the same aggregate extraction order.

The local `pto-isa` copy under
`/home/mouliangyu/projects/github.com/hw-native-sys/pypto/build_output/_deps/pto-isa`
documents the public `pto.vstu` surface as an align-plus-offset state update:
`%align_out, %offset_out = pto.vstu %align_in, %offset_in, %value, %base,
"MODE"`. This matches the VPTO `!pto.vaddr<G>` result contract and confirms
that `vastu` threads offset state, not a full pointer.

## 7. Excluded scalar-offset store forms

### 7.1 `VSSTB`

`VSSTB` is not included in this vector-address support set. VISA lists it as a
deprecated scalar-addressing instruction:

```text
VSSTB.type Vd, [Sn], Sm, Pg, #p
```

The installed CANN 9.0.0 Clang wrapper also takes a scalar offset, not a
`vector_address` value:

```cpp
void vsstb(vector_ST data, __ubuf__ LT *base, int32_t offset,
           vector_bool mask)
```

The compiler-facing LLVM family follows the same scalar-offset contract:

```llvm
declare void @llvm.hivm.vsstb.S(V, ptr addrspace(6), i32, i32, <256 x i1>)
```

An attempted `pto.vasst %base[%vaddr]` wrapper would have to extract a scalar
lane from the `<1 x i32>` vector-address ABI value inside the SIMD-VF loop.
Bisheng rejects that shape during AIV object generation with "Unsupported
scalar instruction in AIV loop". Use the existing scalar-offset `pto.vsstb`
operation for this family instead of adding a vector-address alias.

No matching A5 `vsld` vector-address wrapper was found in the local CANN 9.0.0
inventory, so a `pto.vasld` op is intentionally not included either.

## 8. Lowering ownership

The VPTO implementation lowers these ops in the vector LLVM emitter. The
normative target is direct LLVM IR, not source-level wrapper emission:

- Use the installed CANN headers as the source of truth for wrapper semantics,
  operand order, units, and fixed constants.
- Use generated LLVM IR from the current toolchain as the source of truth for
  exact `llvm.hivm.*` function types. Treat `strings bisheng` only as an
  intrinsic-name inventory.
- Reuse the existing VPTO pointer, vector, mask, and align type conversion
  helpers where possible.
- Add a dedicated conversion for `!pto.vaddr<G>` to the compiler ABI used for
  `vector_address`; local probes show the vector-address load/store families
  accepting a `<1 x i32>` operand.
- Do not lower `!pto.vaddr` through pointer arithmetic or `pto.addptr`.

Current implementation status:

- `vald`, `valdx2`, `vast`, `vastx2`, `pald`, `past`, `valda`, `valdu`,
  `vasta`, and `vastu` lower directly to the typed LLVM intrinsic signatures
  listed in Section 2.5.
- `vag` lowers to the no-IV `@llvm.hivm.vag.32(i32, i32, i32, i32)` builtin
  form. It must be written under an `i16` `scf.for` so CCE middle-end VAG
  lowering can associate the builtin with loop state.
- A `!pto.vaddr` value may be shared by normal non-update vector-address
  operations, but must not be used as the `addr_in` seed of multiple update
  chains such as `valdu` and `vastu`.
- The current VPTO lowering supports one to four byte-stride operands and pads
  inactive dimensions with `i32 0`.

## 9. Initial implementation checklist

1. Add `VAddrType` with `b8`/`b16`/`b32` granularity.
2. Add ODS definitions for:
   - `pto.vag`
   - `pto.vald`
   - `pto.valdx2`
   - `pto.vast`
   - `pto.vastx2`
   - `pto.pald`
   - `pto.past`
   - `pto.valda`
   - `pto.valdu`
   - `pto.vasta`
   - `pto.vastu`
3. Add verifiers for vaddr granularity, UB pointer spaces, distribution
   legality, mask compatibility, and stateful result type equality.
4. Add lowering patterns and targeted lit tests that check emitted intrinsic
   calls or wrapper-equivalent LLVM IR.
5. Add end-to-end micro-op coverage as runtime-safe cases become available.
   The initial simulator cases cover `vag`, aligned `vald`/`vast`,
   pair `valdx2`/`vastx2`, predicate `pald`/`past`, and stateful unaligned
   `valda`/`valdu`/`vastu`/`vasta`.
