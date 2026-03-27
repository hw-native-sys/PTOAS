# 6. Unary Vector Ops

> **Category:** Single-input vector operations
> **Pipeline:** PIPE_V (Vector Core)

Element-wise operations that take one vector input and produce one vector output.

---

## Arithmetic

### `pto.vabs`

- **syntax:** `%result = pto.vabs %input, %mask : !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>`
- **A5 types:** i8-i32, f16, f32

```c
for (int i = 0; i < N; i++)
    dst[i] = (src[i] < 0) ? -src[i] : src[i];
```

---

### `pto.vneg`

- **syntax:** `%result = pto.vneg %input, %mask : !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>`
- **A5 types:** i8-i32, f16, f32

```c
for (int i = 0; i < N; i++)
    dst[i] = -src[i];
```

---

## Transcendental

### `pto.vexp`

- **syntax:** `%result = pto.vexp %input, %mask : !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>`
- **A5 types:** f16, f32

```c
for (int i = 0; i < N; i++)
    dst[i] = expf(src[i]);
```

---

### `pto.vln`

- **syntax:** `%result = pto.vln %input, %mask : !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>`
- **A5 types:** f16, f32

```c
for (int i = 0; i < N; i++)
    dst[i] = logf(src[i]);
```

---

### `pto.vsqrt`

- **syntax:** `%result = pto.vsqrt %input, %mask : !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>`
- **A5 types:** f16, f32

```c
for (int i = 0; i < N; i++)
    dst[i] = sqrtf(src[i]);
```

---

### `pto.vrsqrt`

- **syntax:** `%result = pto.vrsqrt %input, %mask : !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>`
- **A5 types:** f16, f32

```c
for (int i = 0; i < N; i++)
    dst[i] = 1.0f / sqrtf(src[i]);
```

---

### `pto.vrec`

- **syntax:** `%result = pto.vrec %input, %mask : !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>`
- **A5 types:** f16, f32

```c
for (int i = 0; i < N; i++)
    dst[i] = 1.0f / src[i];
```

---

## Activation

### `pto.vrelu`

- **syntax:** `%result = pto.vrelu %input, %mask : !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>`
- **A5 types:** f16, f32

```c
for (int i = 0; i < N; i++)
    dst[i] = (src[i] > 0) ? src[i] : 0;
```

---

## Bitwise

### `pto.vnot`

- **syntax:** `%result = pto.vnot %input, %mask : !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>`
- **A5 types:** all integer types

```c
for (int i = 0; i < N; i++)
    dst[i] = ~src[i];
```

---

### `pto.vbcnt`

- **syntax:** `%result = pto.vbcnt %input, %mask : !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>`
- **A5 types:** all integer types

```c
for (int i = 0; i < N; i++)
    dst[i] = __builtin_popcount(src[i]);
```

---

### `pto.vcls`

- **syntax:** `%result = pto.vcls %input, %mask : !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>`
- **A5 types:** all integer types

```c
for (int i = 0; i < N; i++)
    dst[i] = count_leading_sign_bits(src[i]);
```

---

## Movement

### `pto.vmov`

- **syntax:** `%result = pto.vmov %input, %mask : !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>`
- **semantics:** Vector register copy.

```c
for (int i = 0; i < N; i++)
    dst[i] = src[i];
```

---

## Typical Usage

```mlir
// Softmax numerator: exp(x - max)
%sub = pto.vsub %x, %max_broadcast, %mask : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>
%exp = pto.vexp %sub, %mask : !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>

// Reciprocal for division
%sum_rcp = pto.vrec %sum, %mask : !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>

// ReLU activation
%activated = pto.vrelu %linear_out, %mask : !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>
```
