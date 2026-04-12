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

- **syntax:** `%result = pto.vprelu %input, %alpha : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<NxT>`
- **A5 types:** f16, f32
- **semantics:** Parametric ReLU with per-element alpha vector.

```c
for (int i = 0; i < N; i++)
    dst[i] = (src[i] >= 0) ? src[i] : alpha[i] * src[i];
```

- **inputs:** `%input` is the activation vector and `%alpha` is the per-element
  slope vector.
- **outputs:** `%result` is the parametric-ReLU vector.
- **constraints and limitations:** Floating-point element types only on the
  current A5 surface.

---

### `pto.vexpdiff`

- **syntax:** `%result = pto.vexpdiff %input, %max, "EVEN|ODD" : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<Mxf32>`
- **A5 types:** input `f16` or `f32`, output `f32`
- **semantics:** Fused exp(x - max) for numerically stable softmax.

```c
for (int i = 0; i < N; i++)
    dst[i] = expf(src[i] - max[i]);
```

**Use case:** Softmax numerator computation with numerical stability.

- **inputs:** `%input` is the source vector and `%max` is the broadcasted
  subtraction term. `%part` selects `EVEN` or `ODD` for the
  underlying hardware contract.
- **outputs:** `%result` is the fused `exp(input - max)` vector with `f32`
  elements.
- **constraints and limitations:** Source vectors must be `f16` or `f32`, the
  result vector must be `f32`, and source/result storage width must match.

---

## Fused Compute+Convert Ops

### `pto.vaxpy`

- **syntax:** `%result = pto.vaxpy %src0, %src1, %alpha : !pto.vreg<NxT>, !pto.vreg<NxT>, T -> !pto.vreg<NxT>`
- **A5 types:** f16, f32
- **semantics:** AXPY — scalar-vector multiply-add.

```c
for (int i = 0; i < N; i++)
    dst[i] = alpha * src0[i] + src1[i];
```

- **inputs:** `%src0` is the scaled vector, `%src1` is the addend vector, and
  `%alpha` is the scalar multiplier.
- **outputs:** `%result` is the fused AXPY result.
- **constraints and limitations:** Floating-point element types only on the
  current documented surface.

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

- **syntax:** `%result = pto.vci %index {order = "ORDER"} : integer -> !pto.vreg<NxT>`
- **semantics:** Generate lane index vector.

```c
for (int i = 0; i < N; i++)
    dst[i] = base_index + i;
```

**Use case:** Generate indices for gather/scatter, argsort, etc.

- **inputs:** `%index` is the scalar seed/base index.
- **outputs:** `%result` is the generated index vector.
- **constraints and limitations:** This page documents the arithmetic/indexing
  use of the family; the conversion page also records the same opcode for
  completeness.

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

## Current Implementation Surface Summary

- `pto.vmull %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>, !pto.vreg<NxT>`
- `pto.vmula %acc, %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>`
- `pto.vci %index {order = "ORDER"} : integer -> !pto.vreg<NxT>`
- `pto.vbitsort %dest, %src, %indices, %repeat_times : !pto.ptr<...>, !pto.ptr<...>, !pto.ptr<...>, index`
- `pto.vmrgsort4 %dest, %src0, %src1, %src2, %src3, %count, %config : !pto.ptr<...>, !pto.ptr<...>, !pto.ptr<...>, !pto.ptr<...>, !pto.ptr<...>, i64, i64`

---

## Typical Usage

```mlir
// Softmax with fused expdiff
%max_broadcast = pto.vlds %ub_max[%c0] {dist = "BRC"} : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
%exp_stable = pto.vexpdiff %logits, %max_broadcast : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>

// Leaky ReLU activation
%activated = pto.vlrelu %linear_out, %alpha_scalar, %mask : !pto.vreg<64xf32>, f32, !pto.mask<G> -> !pto.vreg<64xf32>

// Generate indices for argsort
%indices = pto.vci %c0 {order = "ASC"} : i32 -> !pto.vreg<64xi32>
```
