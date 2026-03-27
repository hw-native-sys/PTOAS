# 7. Binary Vector Ops

> **Category:** Two-input vector operations
> **Pipeline:** PIPE_V (Vector Core)

Element-wise operations that take two vector inputs and produce one vector output.

---

## Arithmetic

### `pto.vadd`

- **syntax:** `%result = pto.vadd %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>`
- **A5 types:** i8-i64, f16, bf16, f32

```c
for (int i = 0; i < N; i++)
    dst[i] = src0[i] + src1[i];
```

---

### `pto.vsub`

- **syntax:** `%result = pto.vsub %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>`
- **A5 types:** i8-i64, f16, bf16, f32

```c
for (int i = 0; i < N; i++)
    dst[i] = src0[i] - src1[i];
```

---

### `pto.vmul`

- **syntax:** `%result = pto.vmul %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>`
- **A5 types:** i16-i32, f16, bf16, f32 (**NOT** i8/u8)

```c
for (int i = 0; i < N; i++)
    dst[i] = src0[i] * src1[i];
```

---

### `pto.vdiv`

- **syntax:** `%result = pto.vdiv %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>`
- **A5 types:** f16, f32 only (no integer division)

```c
for (int i = 0; i < N; i++)
    dst[i] = src0[i] / src1[i];
```

---

### `pto.vmax`

- **syntax:** `%result = pto.vmax %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>`
- **A5 types:** i8-i32, f16, bf16, f32

```c
for (int i = 0; i < N; i++)
    dst[i] = (src0[i] > src1[i]) ? src0[i] : src1[i];
```

---

### `pto.vmin`

- **syntax:** `%result = pto.vmin %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>`
- **A5 types:** i8-i32, f16, bf16, f32

```c
for (int i = 0; i < N; i++)
    dst[i] = (src0[i] < src1[i]) ? src0[i] : src1[i];
```

---

## Bitwise

### `pto.vand`

- **syntax:** `%result = pto.vand %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>`
- **A5 types:** all integer types

```c
for (int i = 0; i < N; i++)
    dst[i] = src0[i] & src1[i];
```

---

### `pto.vor`

- **syntax:** `%result = pto.vor %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>`
- **A5 types:** all integer types

```c
for (int i = 0; i < N; i++)
    dst[i] = src0[i] | src1[i];
```

---

### `pto.vxor`

- **syntax:** `%result = pto.vxor %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>`
- **A5 types:** all integer types

```c
for (int i = 0; i < N; i++)
    dst[i] = src0[i] ^ src1[i];
```

---

## Shift

### `pto.vshl`

- **syntax:** `%result = pto.vshl %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>`
- **A5 types:** all integer types

```c
for (int i = 0; i < N; i++)
    dst[i] = src0[i] << src1[i];
```

---

### `pto.vshr`

- **syntax:** `%result = pto.vshr %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>`
- **A5 types:** all integer types

```c
for (int i = 0; i < N; i++)
    dst[i] = src0[i] >> src1[i];  // arithmetic for signed, logical for unsigned
```

---

## Carry Operations

### `pto.vaddc`

- **syntax:** `%result, %carry = pto.vaddc %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>, !pto.mask`
- **semantics:** Add with carry output.

```c
for (int i = 0; i < N; i++) {
    uint64_t r = (uint64_t)src0[i] + src1[i];
    dst[i] = (T)r;
    carry[i] = (r >> bitwidth);
}
```

---

### `pto.vsubc`

- **syntax:** `%result, %borrow = pto.vsubc %lhs, %rhs, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>, !pto.mask`
- **semantics:** Subtract with borrow output.

```c
for (int i = 0; i < N; i++) {
    dst[i] = src0[i] - src1[i];
    borrow[i] = (src0[i] < src1[i]);
}
```

---

## Typical Usage

```mlir
// Vector addition
%sum = pto.vadd %a, %b, %mask : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>

// Element-wise multiply
%prod = pto.vmul %x, %y, %mask : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>

// Clamp to range [min, max]
%clamped_low = pto.vmax %input, %min_vec, %mask : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>
%clamped = pto.vmin %clamped_low, %max_vec, %mask : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>

// Bit manipulation
%masked = pto.vand %data, %bitmask, %mask : !pto.vreg<64xi32>, !pto.vreg<64xi32>, !pto.mask -> !pto.vreg<64xi32>
```
