# 8. Vec-Scalar Ops

> **Category:** Vector-scalar operations
> **Pipeline:** PIPE_V (Vector Core)

Operations that combine a vector with a scalar value, applying the scalar to every lane.

---

## Arithmetic

### `pto.vadds`

- **syntax:** `%result = pto.vadds %input, %scalar, %mask : !pto.vreg<NxT>, T, !pto.mask -> !pto.vreg<NxT>`

```c
for (int i = 0; i < N; i++)
    dst[i] = src[i] + scalar;
```

---

### `pto.vsubs`

- **syntax:** `%result = pto.vsubs %input, %scalar, %mask : !pto.vreg<NxT>, T, !pto.mask -> !pto.vreg<NxT>`

```c
for (int i = 0; i < N; i++)
    dst[i] = src[i] - scalar;
```

---

### `pto.vmuls`

- **syntax:** `%result = pto.vmuls %input, %scalar, %mask : !pto.vreg<NxT>, T, !pto.mask -> !pto.vreg<NxT>`

```c
for (int i = 0; i < N; i++)
    dst[i] = src[i] * scalar;
```

---

### `pto.vmaxs`

- **syntax:** `%result = pto.vmaxs %input, %scalar, %mask : !pto.vreg<NxT>, T, !pto.mask -> !pto.vreg<NxT>`

```c
for (int i = 0; i < N; i++)
    dst[i] = (src[i] > scalar) ? src[i] : scalar;
```

---

### `pto.vmins`

- **syntax:** `%result = pto.vmins %input, %scalar, %mask : !pto.vreg<NxT>, T, !pto.mask -> !pto.vreg<NxT>`

```c
for (int i = 0; i < N; i++)
    dst[i] = (src[i] < scalar) ? src[i] : scalar;
```

---

## Bitwise

### `pto.vands`

- **syntax:** `%result = pto.vands %input, %scalar, %mask : !pto.vreg<NxT>, T, !pto.mask -> !pto.vreg<NxT>`

```c
for (int i = 0; i < N; i++)
    dst[i] = src[i] & scalar;
```

---

### `pto.vors`

- **syntax:** `%result = pto.vors %input, %scalar, %mask : !pto.vreg<NxT>, T, !pto.mask -> !pto.vreg<NxT>`

```c
for (int i = 0; i < N; i++)
    dst[i] = src[i] | scalar;
```

---

### `pto.vxors`

- **syntax:** `%result = pto.vxors %input, %scalar, %mask : !pto.vreg<NxT>, T, !pto.mask -> !pto.vreg<NxT>`

```c
for (int i = 0; i < N; i++)
    dst[i] = src[i] ^ scalar;
```

---

## Shift

### `pto.vshls`

- **syntax:** `%result = pto.vshls %input, %scalar, %mask : !pto.vreg<NxT>, T, !pto.mask -> !pto.vreg<NxT>`

```c
for (int i = 0; i < N; i++)
    dst[i] = src[i] << scalar;
```

---

### `pto.vshrs`

- **syntax:** `%result = pto.vshrs %input, %scalar, %mask : !pto.vreg<NxT>, T, !pto.mask -> !pto.vreg<NxT>`

```c
for (int i = 0; i < N; i++)
    dst[i] = src[i] >> scalar;
```

---

### `pto.vlrelu`

- **syntax:** `%result = pto.vlrelu %input, %scalar, %mask : !pto.vreg<NxT>, T, !pto.mask -> !pto.vreg<NxT>`

```c
for (int i = 0; i < N; i++)
    dst[i] = (src[i] >= 0) ? src[i] : scalar * src[i];
```

---

## Carry Operations

### `pto.vaddcs`

- **syntax:** `%result, %carry = pto.vaddcs %lhs, %rhs, %carry_in, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask, !pto.mask -> !pto.vreg<NxT>, !pto.mask`
- **semantics:** Add with carry-in and carry-out.

```c
for (int i = 0; i < N; i++) {
    uint64_t r = (uint64_t)src0[i] + src1[i] + carry_in[i];
    dst[i] = (T)r;
    carry_out[i] = (r >> bitwidth);
}
```

---

### `pto.vsubcs`

- **syntax:** `%result, %borrow = pto.vsubcs %lhs, %rhs, %borrow_in, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask, !pto.mask -> !pto.vreg<NxT>, !pto.mask`
- **semantics:** Subtract with borrow-in and borrow-out.

```c
for (int i = 0; i < N; i++) {
    dst[i] = src0[i] - src1[i] - borrow_in[i];
    borrow_out[i] = (src0[i] < src1[i] + borrow_in[i]);
}
```

---

## Typical Usage

```mlir
// Add bias to all elements
%biased = pto.vadds %activation, %bias_scalar, %mask : !pto.vreg<64xf32>, f32, !pto.mask -> !pto.vreg<64xf32>

// Scale by constant
%scaled = pto.vmuls %input, %scale, %mask : !pto.vreg<64xf32>, f32, !pto.mask -> !pto.vreg<64xf32>

// Clamp to [0, 255] for uint8 quantization
%clamped_low = pto.vmaxs %input, %c0, %mask : !pto.vreg<64xf32>, f32, !pto.mask -> !pto.vreg<64xf32>
%clamped = pto.vmins %clamped_low, %c255, %mask : !pto.vreg<64xf32>, f32, !pto.mask -> !pto.vreg<64xf32>

// Shift right by fixed amount
%shifted = pto.vshrs %data, %c4, %mask : !pto.vreg<64xi32>, i32, !pto.mask -> !pto.vreg<64xi32>
```
