# 13. DSA/SFU Ops

> **Category:** Domain-specific accelerator and special function unit operations
> **Pipeline:** PIPE_V (Vector Core) / SFU

Fused operations, special functions, and UB-to-UB operations that leverage hardware acceleration.

---

## Fused Activation Ops (vreg→vreg)

### `pto.vlrelu`

- **syntax:** `%result = pto.vlrelu %input, %alpha, %mask : !pto.vreg<NxT>, T, !pto.mask -> !pto.vreg<NxT>`
- **A5 types:** f16, f32
- **semantics:** Leaky ReLU with scalar alpha.

```c
for (int i = 0; i < N; i++)
    dst[i] = (src[i] >= 0) ? src[i] : alpha * src[i];
```

---

### `pto.vprelu`

- **syntax:** `%result = pto.vprelu %input, %alpha : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<NxT>`
- **A5 types:** f16, f32
- **semantics:** Parametric ReLU with per-element alpha vector.

```c
for (int i = 0; i < N; i++)
    dst[i] = (src[i] >= 0) ? src[i] : alpha[i] * src[i];
```

---

### `pto.vexpdiff`

- **syntax:** `%result = pto.vexpdiff %input, %max : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<NxT>`
- **A5 types:** f16, f32
- **semantics:** Fused exp(x - max) for numerically stable softmax.

```c
for (int i = 0; i < N; i++)
    dst[i] = expf(src[i] - max[i]);
```

**Use case:** Softmax numerator computation with numerical stability.

---

## Fused Compute+Convert Ops

### `pto.vaddrelu`

- **syntax:** `%result = pto.vaddrelu %lhs, %rhs : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<NxT>`
- **A5 types:** f16, f32
- **semantics:** Fused add + ReLU.

```c
for (int i = 0; i < N; i++)
    dst[i] = max(src0[i] + src1[i], 0);
```

---

### `pto.vsubrelu`

- **syntax:** `%result = pto.vsubrelu %lhs, %rhs : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<NxT>`
- **A5 types:** f16, f32
- **semantics:** Fused sub + ReLU.

```c
for (int i = 0; i < N; i++)
    dst[i] = max(src0[i] - src1[i], 0);
```

---

### `pto.vaxpy`

- **syntax:** `%result = pto.vaxpy %src0, %src1, %alpha : !pto.vreg<NxT>, !pto.vreg<NxT>, T -> !pto.vreg<NxT>`
- **A5 types:** f16, f32
- **semantics:** AXPY — scalar-vector multiply-add.

```c
for (int i = 0; i < N; i++)
    dst[i] = alpha * src0[i] + src1[i];
```

---

### `pto.vaddreluconv`

- **syntax:** `%result = pto.vaddreluconv %lhs, %rhs : !pto.vreg<NxT0>, !pto.vreg<NxT0> -> !pto.vreg<MxT1>`
- **semantics:** Fused add + ReLU + type conversion (HW fusion).

```c
// f32→f16 variant:
for (int i = 0; i < 64; i++)
    dst_f16[i] = f32_to_f16(max(src0_f32[i] + src1_f32[i], 0));

// f16→i8 variant:
for (int i = 0; i < 128; i++)
    dst_i8[i] = f16_to_i8(max(src0_f16[i] + src1_f16[i], 0));
```

---

### `pto.vmulconv`

- **syntax:** `%result = pto.vmulconv %lhs, %rhs : !pto.vreg<NxT0>, !pto.vreg<NxT0> -> !pto.vreg<MxT1>`
- **semantics:** Fused mul + type conversion (HW fusion).

```c
// f16→i8 variant:
for (int i = 0; i < 128; i++)
    dst_i8[i] = f16_to_i8(src0_f16[i] * src1_f16[i]);
```

---

## Extended Arithmetic

### `pto.vmull`

- **syntax:** `%low, %high = pto.vmull %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>, !pto.vreg<NxT>`
- **A5 types:** i32/u32 (native 32×32→64 widening multiply)
- **semantics:** Widening multiply with high/low results.

```c
for (int i = 0; i < 64; i++) {
    int64_t r = (int64_t)src0_i32[i] * (int64_t)src1_i32[i];
    dst_lo[i] = (int32_t)(r & 0xFFFFFFFF);
    dst_hi[i] = (int32_t)(r >> 32);
}
```

---

### `pto.vmula`

- **syntax:** `%result = pto.vmula %acc, %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>`
- **semantics:** Multiply-accumulate.

```c
for (int i = 0; i < N; i++)
    if (mask[i])
        dst[i] = acc[i] + lhs[i] * rhs[i];
```

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

---

## UB-to-UB Operations

### `pto.vtranspose`

- **syntax:** `pto.vtranspose %dest, %src, %config : !pto.ptr<T, ub>, !pto.ptr<T, ub>, i64`
- **semantics:** UB-to-UB transpose operation (not vreg-to-vreg).

**Note:** This operates on UB memory directly, not on vector registers.

---

## Sorting Operations

### `pto.vsort32`

- **syntax:** `pto.vsort32 %dest, %src, %config : !pto.ptr<T, ub>, !pto.ptr<T, ub>, i64`
- **semantics:** Sort 32 elements in UB.

---

### `pto.vmrgsort`

- **syntax:** `pto.vmrgsort4 %dest, %src0, %src1, %src2, %src3, %count, %config : !pto.ptr<T, ub>, !pto.ptr<T, ub> x4, i64, i64`
- **semantics:** Merge-sort 4 pre-sorted input vectors.

---

## Current Implementation Surface Summary

- `pto.vmull %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>, !pto.vreg<NxT>`
- `pto.vmula %acc, %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>`
- `pto.vci %index {order = "ORDER"} : integer -> !pto.vreg<NxT>`
- `pto.vbitsort %dest, %src, %indices, %repeat_times : !pto.ptr<...>, !pto.ptr<...>, !pto.ptr<...>, index`
- `pto.vmrgsort4 %dest, %src0, %src1, %src2, %src3, %count, %config : !pto.ptr<...>, !pto.ptr<...>, !pto.ptr<...>, !pto.ptr<...>, !pto.ptr<...>, i64, i64`

---

## Typical Usage

```mlir
// Softmax with fused expdiff
%max_broadcast = pto.vlds %ub_max[%c0] {dist = "BRC_B32"} : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
%exp_stable = pto.vexpdiff %logits, %max_broadcast : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>

// Leaky ReLU activation
%activated = pto.vlrelu %linear_out, %alpha_scalar, %mask : !pto.vreg<64xf32>, f32, !pto.mask -> !pto.vreg<64xf32>

// Fused residual add + ReLU
%residual = pto.vaddrelu %conv_out, %skip_connection : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>

// Generate indices for argsort
%indices = pto.vci %c0 {order = "ASC"} : i32 -> !pto.vreg<64xi32>
```
