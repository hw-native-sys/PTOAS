# 6. Unary Vector Ops

> **Category:** Single-input vector operations
> **Pipeline:** PIPE_V (Vector Core)

Element-wise operations that take one vector input and produce one vector output.

## Common Operand Model

- `%input` is the source vector register value.
- `%mask` is the predicate operand. For this family, inactive lanes follow the
  predication behavior of the selected instruction form: zeroing forms
  zero-fill inactive lanes, while merging forms preserve the destination value.
- `%result` is the destination vector register value. Unless stated otherwise,
  `%result` has the same lane count and element type as `%input`.

## CA latency (A5, Ascend910_9599 CA)

Cycle-accurate simulator **popped→retire** latency (cycles). **fp16** values use **aclFloat16** in traces where measured. **bf16:** no simple-tile ST coverage on this surface; treat as **—**.

| PTO op | RV (CA) | fp32 | fp16 | bf16 |
|--------|---------|------|------|------|
| `pto.vabs` | `RV_VABS_FP` | **5** | **5** | — |
| `pto.vneg` | `RV_VMULS` | **8** | **8** | — |
| `pto.vexp` | `RV_VEXP` | **16** | **21** | — |
| `pto.vln` | `RV_VLN` | **18** | **23** | — |
| `pto.vsqrt` | `RV_VSQRT` | **17** | **22** | — |
| `pto.vrelu` | `RV_VRELU` | **5** | **5** | — |
| `pto.vnot` | `RV_VNOT` | — | int-only paths | — |
| `pto.vmov` | `RV_VLD` proxy | **9** | **9** | — |

---

## Arithmetic

### `pto.vabs`

- **syntax:** `%result = pto.vabs %input, %mask : !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>`
- **A5 types:** i8-i32, f16, f32

```c
for (int i = 0; i < N; i++)
    dst[i] = (src[i] < 0) ? -src[i] : src[i];
```

- **inputs:** `%input` supplies the source lanes and `%mask` selects which lanes
  participate.
- **outputs:** `%result` receives the lane-wise absolute values.
- **constraints and limitations:** Source and result types MUST match. On A5,
  integer overflow follows the ISA default truncation behavior for this family;
  `pto.vabs` is not an explicit saturating op.

---

### `pto.vneg`

- **syntax:** `%result = pto.vneg %input, %mask : !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>`
- **A5 types:** i8-i32, f16, f32

```c
for (int i = 0; i < N; i++)
    dst[i] = -src[i];
```

- **inputs:** `%input` is the source vector and `%mask` selects active lanes.
- **outputs:** `%result` is the lane-wise arithmetic negation.
- **constraints and limitations:** Source and result types MUST match.

---

## Transcendental

### `pto.vexp`

- **syntax:** `%result = pto.vexp %input, %mask : !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>`
- **A5 types:** f16, f32

```c
for (int i = 0; i < N; i++)
    dst[i] = expf(src[i]);
```

- **inputs:** `%input` is the source vector and `%mask` selects active lanes.
- **outputs:** `%result` holds `exp(input[i])` per active lane.
- **constraints and limitations:** Only floating-point element types are legal.

---

### `pto.vln`

- **syntax:** `%result = pto.vln %input, %mask : !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>`
- **A5 types:** f16, f32

```c
for (int i = 0; i < N; i++)
    dst[i] = logf(src[i]);
```

- **inputs:** `%input` is the source vector and `%mask` selects active lanes.
- **outputs:** `%result` holds the natural logarithm per active lane.
- **constraints and limitations:** Only floating-point element types are legal.
  For real-number semantics, active inputs SHOULD be strictly positive; non-
  positive inputs follow the target's exception/NaN rules.

---

### `pto.vsqrt`

- **syntax:** `%result = pto.vsqrt %input, %mask : !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>`
- **A5 types:** f16, f32

```c
for (int i = 0; i < N; i++)
    dst[i] = sqrtf(src[i]);
```

- **inputs:** `%input` is the source vector and `%mask` selects active lanes.
- **outputs:** `%result` holds the square root per active lane.
- **constraints and limitations:** Only floating-point element types are legal.
  Negative active inputs follow the target's exception/NaN rules.

---

## Activation

### `pto.vrelu`

- **syntax:** `%result = pto.vrelu %input, %mask : !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>`
- **A5 types:** f16, f32

```c
for (int i = 0; i < N; i++)
    dst[i] = (src[i] > 0) ? src[i] : 0;
```

- **inputs:** `%input` is the source vector and `%mask` selects active lanes.
- **outputs:** `%result` holds `max(input[i], 0)` per active lane.
- **constraints and limitations:** Only floating-point element types are legal
  on the current A5 surface described here.

---

## Bitwise

### `pto.vnot`

- **syntax:** `%result = pto.vnot %input, %mask : !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>`
- **A5 types:** all integer types

```c
for (int i = 0; i < N; i++)
    dst[i] = ~src[i];
```

- **inputs:** `%input` is the source vector and `%mask` selects active lanes.
- **outputs:** `%result` holds the lane-wise bitwise inversion.
- **constraints and limitations:** Integer element types only.

---

## Movement

## Typical Usage

```mlir
// Softmax numerator: exp(x - max)
%sub = pto.vsub %x, %max_broadcast, %mask : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<G> -> !pto.vreg<64xf32>
%exp = pto.vexp %sub, %mask : !pto.vreg<64xf32>, !pto.mask<G> -> !pto.vreg<64xf32>

// ReLU activation
%activated = pto.vrelu %linear_out, %mask : !pto.vreg<64xf32>, !pto.mask<G> -> !pto.vreg<64xf32>
```
