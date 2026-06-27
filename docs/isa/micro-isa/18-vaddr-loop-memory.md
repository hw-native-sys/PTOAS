# 18. Vector-Address Loop Memory

> **Category:** loop-derived UB ↔ Vector/Predicate data movement
> **Pipeline:** PIPE_V (Vector Core)

Vector-address memory ops use a loop-derived address token together with an
explicit UB base pointer. They are useful when a vector loop needs address
progression that is tied to the loop counter rather than to scalar pointer
arithmetic.

This chapter documents the PTO surface contract for `!pto.vaddr` and the
`pto.vag` / `pto.vald` / `pto.vast` / `pto.pald` / `pto.past` families.

---

## Common Operand Model

- `%base` is the explicit UB base pointer. It MUST have type `!pto.ptr<T, ub>`.
- `%addr` is a `!pto.vaddr<G>` vector-address offset token. It is not a pointer.
- The effective UB address is logically `base + addr`.
- `%mask` is a predicate mask used by predicated vector stores.
- `!pto.align` is the explicit SSA carrier for unaligned load/store state.

`!pto.vaddr<G>` supports these granularities:

```mlir
!pto.vaddr<b8>
!pto.vaddr<b16>
!pto.vaddr<b32>
```

The granularity records the address family used by the vector-address producer.
It does not replace the element type of the explicit `%base` pointer.

---

## Loop Form

`pto.vag` MUST be nested under an `i16` `scf.for` loop. The loop type is part of
the valid vector-address form.

Use the explicit non-`index` loop type marker:

```mlir
%c0_i16 = arith.constant 0 : i16
%c1_i16 = arith.constant 1 : i16
%c2_i16 = arith.constant 2 : i16

pto.vecscope {
  scf.for %i = %c0_i16 to %c2_i16 step %c1_i16 : i16 {
    %addr = pto.vag %stride : i32 -> !pto.vaddr<b32>
    // vector-address memory ops using %addr
  }
}
```

An `index` loop is not valid for `pto.vag`:

```mlir
%c0 = arith.constant 0 : index
%c1 = arith.constant 1 : index
%c2 = arith.constant 2 : index

// Invalid for pto.vag.
scf.for %i = %c0 to %c2 step %c1 {
  %addr = pto.vag %stride : i32 -> !pto.vaddr<b32>
}
```

`pto.vag` does not create or outline a loop. The surrounding loop must already
exist in the PTO IR.

For nested loops, the active loop stack is determined at the lexical position of
the `pto.vag` op. The immediately enclosing `scf.for` is the innermost active
loop for that `pto.vag`; each enclosing `scf.for` outside it is the next outer
active loop.

---

## Address Generation

### `pto.vag`

- **syntax:** `%addr = pto.vag %s0 : i32 -> !pto.vaddr<G>`
- **syntax:** `%addr = pto.vag %s0, %s1 : i32, i32 -> !pto.vaddr<G>`
- **syntax:** `%addr = pto.vag %s0, %s1, %s2 : i32, i32, i32 -> !pto.vaddr<G>`
- **syntax:** `%addr = pto.vag %s0, %s1, %s2, %s3 : i32, i32, i32, i32 -> !pto.vaddr<G>`
- **semantics:** Create a vector-address offset value for the surrounding loop.
- **inputs:**
  `%s0` ... `%s3` are byte strides for active loop dimensions. The operands are
  ordered from inner loop to outer loop:
  `%s0` applies to the immediately enclosing loop, `%s1` applies to the next
  outer loop, `%s2` applies to the next outer loop after that, and `%s3` applies
  to the fourth active loop.
- **outputs:**
  `%addr` is a `!pto.vaddr<G>` offset token.
- **constraints and limitations:**
  `pto.vag` takes one to four `i32` byte-stride operands. It MUST be nested in
  an `i16` `scf.for`. The result granularity MUST be `b8`, `b16`, or `b32`.

For a `pto.vag` nested under active loop counters `i0, i1, i2, i3`, listed from
inner to outer at the `pto.vag` location, the logical offset is:

```text
addr = i0 * s0
     + i1 * s1
     + i2 * s2
     + i3 * s3
```

If fewer than four stride operands are present, only the corresponding innermost
active loop counters participate in the address. For example, `pto.vag %s0,
%s1` uses the immediately enclosing loop and its next outer loop.

**Example:**

```mlir
%stride = arith.constant 4 : i32
%addr = pto.vag %stride : i32 -> !pto.vaddr<b32>
```

Four nested loops:

```mlir
scf.for %k = %c0_i16 to %k_bound step %c1_i16 : i16 {
  scf.for %l = %c0_i16 to %l_bound step %c1_i16 : i16 {
    scf.for %m = %c0_i16 to %m_bound step %c1_i16 : i16 {
      scf.for %n = %c0_i16 to %n_bound step %c1_i16 : i16 {
        %addr = pto.vag %n_stride, %m_stride, %l_stride, %k_stride
            : i32, i32, i32, i32 -> !pto.vaddr<b32>
      }
    }
  }
}
```

---

## Vector-Address Vector Loads

### `pto.vald`

- **syntax:** `%result = pto.vald %source[%addr] {dist = "DIST"} : !pto.ptr<T, ub>, !pto.vaddr<G> -> !pto.vreg<NxT>`
- **semantics:** Load a vector register from UB using a vector-address offset.
- **inputs:**
  `%source` is the UB base pointer, `%addr` is the vector-address offset, and
  `DIST` selects the load distribution mode.
- **outputs:**
  `%result` is the loaded vector register.
- **constraints and limitations:**
  `%source` MUST be UB-backed. `%addr` MUST be `!pto.vaddr<...>`. The result
  element type MUST match the source pointer element type.

Supported `DIST` values:

| Family | Notes |
|--------|-------|
| `NORM` | normal vector-address load |
| `BRC_B8` / `BRC_B16` / `BRC_B32` | broadcast family |
| `US_B8` / `US_B16` | upsample family |
| `DS_B8` / `DS_B16` | downsample family |
| `UNPK_B8` / `UNPK_B16` / `UNPK_B32` | unpack family |
| `BRC_BLK` | block broadcast path |
| `E2B_B16` / `E2B_B32` | element-to-byte expansion family |
| `UNPK4` | `b8` unpack-4 family |
| `SPLT4CHN` | `b8` split-channel family |
| `SPLT2CHN_B8` / `SPLT2CHN_B16` | 2-channel split family |

**Example:**

```mlir
%v = pto.vald %ub[%addr] {dist = "NORM"}
    : !pto.ptr<f32, ub>, !pto.vaddr<b32> -> !pto.vreg<64xf32>
```

### `pto.valdx2`

- **syntax:** `%lo, %hi = pto.valdx2 %source[%addr], "DIST" : !pto.ptr<T, ub>, !pto.vaddr<G> -> !pto.vreg<NxT>, !pto.vreg<NxT>`
- **semantics:** Load two vector registers from UB using a vector-address offset
  and a deinterleave-style distribution.
- **inputs:** `%source` is the UB base pointer, `%addr` is the vector-address
  offset, and `DIST` selects the x2 load distribution mode.
- **outputs:** `%lo` and `%hi` are the two loaded vector registers.
- **constraints and limitations:**
  `%lo` and `%hi` MUST have the same vector type. `%source` MUST be UB-backed.

Supported `DIST` values:

```text
BDINTLV
DINTLV_B8, DINTLV_B16, DINTLV_B32
```

---

## Vector-Address Vector Stores

### `pto.vast`

- **syntax:** `pto.vast %value, %dest[%addr], %mask {dist = "DIST"} : !pto.vreg<NxT>, !pto.ptr<T, ub>, !pto.vaddr<G>, !pto.mask<M>`
- **semantics:** Store a vector register to UB using a vector-address offset and
  a predicate mask.
- **inputs:**
  `%value` is the vector to store, `%dest` is the UB base pointer, `%addr` is
  the vector-address offset, `%mask` controls active lanes, and `DIST` selects
  the store distribution mode.
- **constraints and limitations:**
  `%dest` MUST be UB-backed. `%addr` MUST be `!pto.vaddr<...>`.

Supported `DIST` values:

```text
NORM_B8, NORM_B16, NORM_B32
1PT_B8, 1PT_B16, 1PT_B32
PK_B16, PK_B32, PK_B64
PK4_B32
MRG4CHN_B8
MRG2CHN_B8, MRG2CHN_B16
INTLV_B8, INTLV_B16, INTLV_B32
```

### `pto.vastx2`

- **syntax:** `pto.vastx2 %lo, %hi, %dest[%addr], "DIST", %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.ptr<T, ub>, !pto.vaddr<G>, !pto.mask<M>`
- **semantics:** Store two vector registers to UB using a vector-address offset
  and an x2 store distribution.
- **inputs:** `%lo` and `%hi` are the vectors to store, `%dest` is the UB base,
  `%addr` is the vector-address offset, and `%mask` controls active lanes.
- **constraints and limitations:**
  `%lo` and `%hi` MUST have the same vector type. `%dest` MUST be UB-backed.

Supported `DIST` values:

```text
INTLV_B8, INTLV_B16, INTLV_B32
```

---

## Vector-Address Predicate Loads And Stores

### `pto.pald`

- **syntax:** `%mask = pto.pald %source[%addr], "DIST" : !pto.ptr<i32, ub>, !pto.vaddr<G> -> !pto.mask<M>`
- **semantics:** Load a predicate mask from UB using a vector-address offset.
- **DIST:** mandatory token, one of `NORM`, `US`, or `DS`.
- **constraints and limitations:**
  `%source` MUST be UB-backed. `%addr` MUST be `!pto.vaddr<...>`.

### `pto.past`

- **syntax:** `pto.past %mask, %dest[%addr], "DIST" : !pto.mask<M>, !pto.ptr<i32, ub>, !pto.vaddr<G>`
- **semantics:** Store a predicate mask to UB using a vector-address offset.
- **DIST:** mandatory token, one of `NORM` or `PK`.
- **constraints and limitations:**
  `%dest` MUST be UB-backed. `%addr` MUST be `!pto.vaddr<...>`.

---

## Vector-Address Unaligned Update Chains

Unaligned vector-address ops carry explicit alignment state. Some forms return
an updated vector address, which makes the address value part of an update
chain.

The update-chain seed rule is:

> One `!pto.vaddr` SSA value MUST NOT be used as the `addr_in` seed of multiple
> update chains.

Normal non-update vector-address operations may share one `!pto.vaddr`. The
restriction applies to `pto.valdu` and `pto.vastu` `addr_in` operands.

### `pto.valda`

- **syntax:** `%align = pto.valda %source[%addr] : !pto.ptr<T, ub>, !pto.vaddr<G> -> !pto.align`
- **semantics:** Prime load-side alignment state for a later vector-address
  unaligned load.
- **constraints and limitations:**
  `%source` MUST be UB-backed. `%addr` MUST be `!pto.vaddr<...>`.

### `pto.valdu`

- **syntax:** `%value, %align_out, %addr_out = pto.valdu %source[%addr_in], %align_in, %inc : !pto.ptr<T, ub>, !pto.vaddr<G>, !pto.align, i32 -> !pto.vreg<NxT>, !pto.align, !pto.vaddr<G>`
- **semantics:** Unaligned vector-address load using incoming alignment state.
- **inputs:**
  `%source` is the UB base pointer, `%addr_in` is the input vector address,
  `%align_in` is the input alignment state, and `%inc` is a byte increment used
  to compute `%addr_out`.
- **outputs:**
  `%value` is the loaded vector, `%align_out` is the updated alignment state,
  and `%addr_out` is the updated vector address.
- **constraints and limitations:**
  `%addr_in` and `%addr_out` MUST have the same `!pto.vaddr` type. `%inc` MUST
  be `i32`. The same `%addr_in` value MUST NOT seed another update chain.

### `pto.init_align`

- **syntax:** `%align = pto.init_align : !pto.align`
- **semantics:** Create a fresh store-side alignment state.

### `pto.vastu`

- **syntax:** `%align_out, %addr_out = pto.vastu %align_in, %addr_in, %value, %base, "POST_UPDATE" : !pto.align, !pto.vaddr<G>, !pto.vreg<NxT>, !pto.ptr<T, ub> -> !pto.align, !pto.vaddr<G>`
- **semantics:** Unaligned vector-address store using incoming alignment and
  address state.
- **inputs:**
  `%align_in` is the incoming store alignment state, `%addr_in` is the input
  vector address, `%value` is the vector to store, and `%base` is the UB base
  pointer.
- **outputs:**
  `%align_out` and `%addr_out` are the updated store state.
- **constraints and limitations:**
  The mode MUST be `"POST_UPDATE"`. `%base` MUST be UB-backed. `%addr_in` and
  `%addr_out` MUST have the same `!pto.vaddr` type. The same `%addr_in` value
  MUST NOT seed another update chain.

### `pto.vasta`

- **syntax:** `pto.vasta %align, %base[%addr] : !pto.align, !pto.ptr<T, ub>, !pto.vaddr<G>`
- **semantics:** Complete a vector-address unaligned store chain by consuming
  the final alignment and address state.
- **constraints and limitations:**
  `%base` MUST be UB-backed. `%align` should be produced by `pto.vastu` or a
  compatible store-align producer.

---

## Sharing Patterns

### Valid normal sharing

```mlir
%addr = pto.vag %stride : i32 -> !pto.vaddr<b32>
%mask = pto.pset_b32 "PAT_ALL" : !pto.mask<b32>
%v = pto.vald %src[%addr] {dist = "NORM"}
    : !pto.ptr<f32, ub>, !pto.vaddr<b32> -> !pto.vreg<64xf32>
pto.vast %v, %dst[%addr], %mask {dist = "NORM_B32"}
    : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.vaddr<b32>, !pto.mask<b32>
```

### Invalid update-chain fanout

```mlir
%addr = pto.vag %stride : i32 -> !pto.vaddr<b32>
%load_align = pto.valda %src[%addr]
    : !pto.ptr<f32, ub>, !pto.vaddr<b32> -> !pto.align
%v, %next_load_align, %next_load_addr =
    pto.valdu %src[%addr], %load_align, %inc
    : !pto.ptr<f32, ub>, !pto.vaddr<b32>, !pto.align, i32
      -> !pto.vreg<64xf32>, !pto.align, !pto.vaddr<b32>
%store_align = pto.init_align : !pto.align

// Invalid: %addr also seeds a store update chain.
%next_store_align, %next_store_addr =
    pto.vastu %store_align, %addr, %v, %dst, "POST_UPDATE"
    : !pto.align, !pto.vaddr<b32>, !pto.vreg<64xf32>, !pto.ptr<f32, ub>
      -> !pto.align, !pto.vaddr<b32>
```

### Valid independent update chains

```mlir
%load_addr = pto.vag %stride : i32 -> !pto.vaddr<b32>
%store_addr = pto.vag %stride : i32 -> !pto.vaddr<b32>

%load_align = pto.valda %src[%load_addr]
    : !pto.ptr<f32, ub>, !pto.vaddr<b32> -> !pto.align
%v, %next_load_align, %next_load_addr =
    pto.valdu %src[%load_addr], %load_align, %inc
    : !pto.ptr<f32, ub>, !pto.vaddr<b32>, !pto.align, i32
      -> !pto.vreg<64xf32>, !pto.align, !pto.vaddr<b32>

%store_align = pto.init_align : !pto.align
%next_store_align, %next_store_addr =
    pto.vastu %store_align, %store_addr, %v, %dst, "POST_UPDATE"
    : !pto.align, !pto.vaddr<b32>, !pto.vreg<64xf32>, !pto.ptr<f32, ub>
      -> !pto.align, !pto.vaddr<b32>
pto.vasta %next_store_align, %dst[%next_store_addr]
    : !pto.align, !pto.ptr<f32, ub>, !pto.vaddr<b32>
```

---

## Complete Example

```mlir
module attributes {pto.target_arch = "a5", pto.kernel_kind = #pto.kernel_kind<vector>} {
  func.func @vaddr_loop_copy(%src: !pto.ptr<f32, ub>,
                             %dst: !pto.ptr<f32, ub>) attributes {pto.kernel} {
    %c0_i16 = arith.constant 0 : i16
    %c1_i16 = arith.constant 1 : i16
    %c2_i16 = arith.constant 2 : i16
    %stride = arith.constant 4 : i32

    pto.vecscope {
      scf.for %i = %c0_i16 to %c2_i16 step %c1_i16 : i16 {
        %addr = pto.vag %stride : i32 -> !pto.vaddr<b32>
        %mask = pto.pset_b32 "PAT_ALL" : !pto.mask<b32>
        %value = pto.vald %src[%addr] {dist = "NORM"}
            : !pto.ptr<f32, ub>, !pto.vaddr<b32> -> !pto.vreg<64xf32>
        pto.vast %value, %dst[%addr], %mask {dist = "NORM_B32"}
            : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, !pto.vaddr<b32>, !pto.mask<b32>
      }
    }

    return
  }
}
```
