# 13. DSA/SFU Ops

> **Category:** Domain-specific accelerator and special function unit operations
> **Pipeline:** PIPE_V (Vector Core) / SFU

Fused operations, special functions, and UB-to-UB operations that leverage hardware acceleration.

## Common Operand Model

- `%input`, `%lhs`, `%rhs`, `%acc`, and `%alpha` are source SSA values whose
  roles are called out per instruction.
- `%mask` is the predicate operand `Pg` when present.
- `%result` is the destination SSA value.
- This page mixes three different backend shapes: pure `vreg -> vreg` ops,
  conversion/fusion ops, and UB-to-UB helpers. Each instruction section calls
  out which storage model it uses.

---

## Fused Activation Ops (vreg→vreg)

### `pto.vlrelu`

- **syntax:** `%result = pto.vlrelu %input, %alpha, %mask : !pto.vreg<NxT>, T, !pto.mask<G> -> !pto.vreg<NxT>`
- **A5 types:** f16, f32
- **semantics:** Leaky ReLU with scalar alpha.

```c
for (int i = 0; i < N; i++)
    dst[i] = (src[i] >= 0) ? src[i] : alpha * src[i];
```

- **inputs:** `%input` is the activation vector, `%alpha` is the scalar slope,
  and `%mask` selects active lanes.
- **outputs:** `%result` is the leaky-ReLU vector.
- **constraints and limitations:** Only `f16` and `f32` forms are currently
  documented for `pto.vlrelu`.

---

### `pto.vprelu`

- **syntax:** `%result = pto.vprelu %input, %alpha, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask<bW> -> !pto.vreg<NxT>`
- **A5 types:** f16, f32
- **semantics:** Parametric ReLU with per-element alpha vector.

```c
for (int i = 0; i < N; i++)
    dst[i] = (src[i] >= 0) ? src[i] : alpha[i] * src[i];
```

- **inputs:** `%input` is the activation vector, `%alpha` is the per-element
  slope vector, and `%mask` selects active lanes.
- **outputs:** `%result` is the parametric-ReLU vector.
- **constraints and limitations:** Floating-point element types only on the
  current A5 surface.

---

### `pto.vexpdif`

- **syntax:** `%result = pto.vexpdif %input, %max, %mask, "EVEN|ODD" : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask<bW> -> !pto.vreg<Mxf32>`
- **A5 types:** input `f16` or `f32`, output `f32`
- **semantics:** Fused exp(x - max) for numerically stable softmax.

```c
for (int i = 0; i < N; i++)
    dst[i] = expf(src[i] - max[i]);
```

**Use case:** Softmax numerator computation with numerical stability.

- **inputs:** `%input` is the source vector, `%max` is the broadcasted
  subtraction term, `%mask` selects active source lanes, and `%part` selects
  `EVEN` or `ODD` for the underlying hardware contract.
- **outputs:** `%result` is the fused `exp(input - max)` vector with `f32`
  elements.
- **constraints and limitations:** Source vectors must be `f16` or `f32`, the
  result vector must be `f32`, the mask granularity must match the input
  vector element width, and source/result storage width must match.

---

## Fused Compute+Convert Ops

### `pto.vaxpy`

- **syntax:** `%result = pto.vaxpy %src0, %src1, %alpha, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, T, !pto.mask<G> -> !pto.vreg<NxT>`
- **A5 types:** f16, f32
- **semantics:** AXPY — scalar-vector multiply-add.

```c
for (int i = 0; i < N; i++)
    dst[i] = alpha * src0[i] + src1[i];
```

- **inputs:** `%src0` is the scaled vector, `%src1` is the addend vector,
  `%alpha` is the scalar multiplier, and `%mask` selects active lanes.
- **outputs:** `%result` is the fused AXPY result.
- **constraints and limitations:** Floating-point element types only on the
  current documented surface.

---

### `pto.vmulscvt`

- **syntax:** `%result = pto.vmulscvt %input, %scalar, %mask, %rnd, %part : !pto.vreg<NxT0>, T0, !pto.mask<G> -> !pto.vreg<MxT1>`
- **A5 types:** input `f32`, output `f16` (primary documented pair)
- **semantics:** Fused multiply-by-scalar and type conversion. Each active lane
  is multiplied by `%scalar` and then converted from the source element type to
  the destination element type in a single hardware step using the authored
  round mode.

```c
for (int i = 0; i < N; i++)
    if (mask[i])
        dst[i] = convert_type<T1>(src[i] * scalar, rnd);
```

**Use case:** Softmax scale-and-downcast: apply the reciprocal scale factor and
narrow from `f32` (accumulator precision) to `f16` (storage precision) before a
block-strided store.

- **inputs:** `%input` is the source vector (wider type), `%scalar` is the
  uniform scale factor, `%mask` selects active lanes, `%rnd` selects the cast
  round mode, and `%part` selects `EVEN` or `ODD` for half-width output
  placement.
- **outputs:** `%result` is the scaled and converted vector with the narrower
  destination element type.
- **constraints and limitations:** The source/destination type pair must be a
  legal hardware narrowing conversion (e.g., `f32 -> f16`). Illegal pairs are
  rejected. `%scalar` must match the source element type. The mask granularity
  must match the source vector element width.

**Example** — softmax scale and downcast:
```mlir
// Apply scale=1.0 and narrow f32 -> f16, writing into even half of the
// destination packing layout
%f16_even = pto.vmulscvt %f32_exp, %one, %mask, "A", "EVEN"
    : !pto.vreg<64xf32>, f32, !pto.mask<b32> -> !pto.vreg<128xf16>
%f16_odd  = pto.vmulscvt %f32_exp2, %one, %mask, "A", "ODD"
    : !pto.vreg<64xf32>, f32, !pto.mask<b32> -> !pto.vreg<128xf16>
```

---

## Extended Arithmetic

### `pto.vmull`

- **syntax:** `%low, %high = pto.vmull %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>, !pto.vreg<NxT>`
- **A5 types:** i32/ui32 (native 32×32→64 widening multiply)
- **semantics:** Widening multiply with high/low results.

```c
for (int i = 0; i < 64; i++) {
    int64_t r = (int64_t)src0_i32[i] * (int64_t)src1_i32[i];
    dst_lo[i] = (int32_t)(r & 0xFFFFFFFF);
    dst_hi[i] = (int32_t)(r >> 32);
}
```

- **inputs:** `%lhs` and `%rhs` are the source vectors and `%mask` selects
  active lanes.
- **outputs:** `%low` and `%high` expose the widened-product low/high parts.
- **constraints and limitations:** The current documented A5 form is the native
  widening 32x32->64 integer multiply family.

---

### `pto.vmula`

- **syntax:** `%result = pto.vmula %acc, %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>`
- **semantics:** Multiply-accumulate.

```c
for (int i = 0; i < N; i++)
    if (mask[i])
        dst[i] = acc[i] + lhs[i] * rhs[i];
```

- **inputs:** `%acc` is the accumulator input, `%lhs` and `%rhs` are the
  multiplicands, and `%mask` selects active lanes.
- **outputs:** `%result` is the multiply-accumulate result.
- **constraints and limitations:** `pto.vmula` is a fused multiply-accumulate
  operation and is not always interchangeable with separate `vmul` plus `vadd`.

---

## Index Generation

### `pto.vci`

- **syntax:** `%result = pto.vci %index {order = "ASC|DESC"} : T -> !pto.vreg<NxT>`
- **semantics:** Generate a lane index vector from a scalar base value.

```c
for (int i = 0; i < N; i++)
    dst[i] = (order == ASC) ? (base_index + i) : (base_index - i);
```

**Use case:** Generate indices for gather/scatter, argsort, etc.

- **inputs:** `%index` is the scalar base value. Supported scalar types are
  `i8/i16/i32`, `f16`, and `f32`.
- **outputs:** `%result` is the generated index vector.
- **constraints and limitations:** `%result` element type determines both the
  generated element type and the lane count. Supported result types are
  `!pto.vreg<256xsi8>`, `!pto.vreg<128xsi16>`, `!pto.vreg<64xsi32>`,
  `!pto.vreg<128xf16>`, and `!pto.vreg<64xf32>`. `%index` must use the
  matching scalar type for `f16`/`f32`; for integer results, `%index` must use
  the same bit width and may be signless or signed.

---

## UB-to-UB Operations

### `pto.vtranspose`

- **syntax:** `pto.vtranspose %dest, %src : !pto.ptr<T, ub>, !pto.ptr<T, ub>`
- **semantics:** Interpret the 512-byte region beginning at `%src` as a
  row-major 16×16 matrix of 16-bit elements, and write its transpose to the
  512-byte region beginning at `%dest`. For `0 ≤ i, j < 16`, the observable
  result is:

  ```text
  dst[i][j] = src[j][i]
  ```

  The operation returns no SSA value.
- **inputs:** `%src` and `%dest` are UB-backed pointers with the same element
  type. The element type is a signed, signless, or unsigned 16-bit integer
  (`si16`, `i16`, or `ui16`); signed and signless values use the same bitwise
  transpose behavior. Each pointer must designate a
  contiguous region of at least 512 bytes, and each address must satisfy the
  hardware UB alignment requirement of 32 bytes.
- **outputs:** `%dest` receives all 256 transposed elements in row-major order;
  no bytes outside that 512-byte destination region are part of the result.
- **constraints and limitations:** Destination and source element types must
  match, and both pointers must address UB memory. The source and destination
  regions must not overlap. If either region is shorter than 512 bytes, an
  access crosses the end of its buffer, an address is misaligned, or the two
  regions overlap, behavior is undefined (the implementation is not required
  to diagnose the condition or produce a partial result). For a transpose of
  an arbitrary shape, use the Tile-layer `pto.ttrans` operation instead.

---

## Sorting Operations

### `pto.vbitsort`

- **syntax:** `pto.vbitsort %dest, %src, %indices, %repeat_times : !pto.ptr<...>, !pto.ptr<...>, !pto.ptr<...>, index`
- **semantics:** Sort 32 region proposals by score and materialize sorted
  proposal records into `%dest`.
- **inputs:** `%dest` is the UB destination buffer. `%src` is the UB score
  buffer. `%indices` is the UB index buffer. `%repeat_times` is the repeat
  count; each repeat processes the next adjacent group of 32 scores and 32
  indices.
- **outputs:** This op writes UB memory and returns no SSA value. Each output
  record occupies 8 bytes: the upper 4 bytes hold the index and the lower
  4 bytes hold the score. For `f16` score forms, the score uses the lower
  2 bytes of that 4-byte score field and the upper 2 bytes are reserved.
- **constraints and limitations:** `%dest`, `%src`, and `%indices` MUST be
  UB-backed pointers and SHOULD satisfy the backend alignment contract expected
  by the A5 `VBS32` instruction. Scores are sorted in descending order, so the
  highest score is written to the lowest destination address. Equal-score ties
  preserve the earlier input proposal first. This is a UB helper, not a pure
  `vreg -> vreg` op.

---

### `pto.vmrgsort4`

- **syntax:** `pto.vmrgsort4 %dest, %src0, %src1, %src2, %src3, %count, %config : !pto.ptr<T, ub>, !pto.ptr<T, ub>, !pto.ptr<T, ub>, !pto.ptr<T, ub>, !pto.ptr<T, ub>, i64, i64`
- **semantics:** Merge-sort 4 pre-sorted input vectors.
- **inputs:** `%dest` is the UB destination, `%src0..%src3` are the four
  pre-sorted UB inputs, `%count` is the number of valid elements, and `%config`
  is the operation control word.
- **outputs:** This op writes UB memory and returns no SSA value.
- **constraints and limitations:** Inputs MUST already be sorted according to
  the sort order encoded by `%config`.

---

### `pto.get_vms4_sr`

- **syntax:** `%list0, %list1, %list2, %list3 = pto.get_vms4_sr : i16, i16, i16, i16`
- **semantics:** Read `VMS4_SR` and return the finished counts for source
  lists 0, 1, 2, and 3. After exhausted `pto.vmrgsort4`, the four results map
  to `VMS4_SR[15:0]`, `VMS4_SR[31:16]`, `VMS4_SR[47:32]`, and
  `VMS4_SR[63:48]`.

---

## Current Implementation Surface Summary

- `pto.vmull %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>, !pto.vreg<NxT>`
- `pto.vmula %acc, %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>`
- `pto.vmulscvt %input, %scalar, %mask, %rnd, %part : !pto.vreg<NxT0>, T0, !pto.mask<G> -> !pto.vreg<MxT1>`
- `pto.vci %index {order = "ASC|DESC"} : T -> !pto.vreg<NxT>`
- `pto.vtranspose %dest, %src : !pto.ptr<T, ub>, !pto.ptr<T, ub>`
- `pto.vbitsort %dest, %src, %indices, %repeat_times : !pto.ptr<...>, !pto.ptr<...>, !pto.ptr<...>, index`
- `pto.vmrgsort4 %dest, %src0, %src1, %src2, %src3, %count, %config : !pto.ptr<...>, !pto.ptr<...>, !pto.ptr<...>, !pto.ptr<...>, !pto.ptr<...>, i64, i64`
- `pto.get_vms4_sr : i16, i16, i16, i16`

---

## Typical Usage

```mlir
// Softmax with fused expdiff
%max_broadcast = pto.vlds %ub_max[%c0] {dist = "BRC_B32"} : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
%exp_stable = pto.vexpdif %logits, %max_broadcast, %mask, "ODD" : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>

// Leaky ReLU activation
%activated = pto.vlrelu %linear_out, %alpha_scalar, %mask : !pto.vreg<64xf32>, f32, !pto.mask<G> -> !pto.vreg<64xf32>

// Generate ascending si32 indices for argsort
%indices = pto.vci %c0 {order = "ASC"} : i32 -> !pto.vreg<64xsi32>
```
