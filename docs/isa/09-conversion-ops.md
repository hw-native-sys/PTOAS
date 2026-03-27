# 9. Conversion Ops

> **Category:** Type conversion operations
> **Pipeline:** PIPE_V (Vector Core)

Operations that convert between data types (float/int, narrowing/widening).

---

## `pto.vci`

- **syntax:** `%result = pto.vci %index {order = "ORDER"} : integer -> !pto.vreg<NxT>`
- **semantics:** Generate a lane-index vector from a scalar seed/index value.

---

## `pto.vcvt`

- **syntax:** `%result = pto.vcvt %input {round_mode = "ROUND_MODE", sat = "SAT_MODE", part = "PART_MODE"} : !pto.vreg<NxT0> -> !pto.vreg<MxT1>`
- **semantics:** Type conversion between float/int types with rounding control.

```c
for (int i = 0; i < min(N, M); i++)
    dst[i] = convert(src[i], T0, T1, round_mode);
```

---

### Rounding Modes

| Mode | Description |
|------|-------------|
| `ROUND_R` | Round to nearest, ties to even (default) |
| `ROUND_A` | Round away from zero |
| `ROUND_F` | Round toward negative infinity (floor) |
| `ROUND_C` | Round toward positive infinity (ceil) |
| `ROUND_Z` | Round toward zero (truncate) |
| `ROUND_O` | Round to odd |

---

### Saturation Modes

| Mode | Description |
|------|-------------|
| `RS_ENABLE` | Saturate on overflow |
| `RS_DISABLE` | No saturation (wrap/undefined on overflow) |

---

### Part Modes (for width-changing conversions)

| Mode | Description |
|------|-------------|
| `PART_EVEN` | Output to even-indexed lanes |
| `PART_ODD` | Output to odd-indexed lanes |

---

### A5 Supported Conversions

**Float-Float (vcvtff):**
- f32 ↔ f16
- f32 ↔ bf16
- f16 ↔ bf16

**Float-Int (vcvtfi):**
- f16 → i16, f16 → i32
- f32 → i16, f32 → i32
- bf16 → i32

**Int-Float (vcvtif):**
- i16 → f16
- i32 → f32

---

### Width-Changing Conversion Pattern

For conversions that change width (e.g., f32→f16), use even/odd parts and combine:

```mlir
// Convert two f32 vectors to one f16 vector
%even = pto.vcvt %in0 {round_mode = "ROUND_R", sat = "RS_ENABLE", part = "PART_EVEN"}
    : !pto.vreg<64xf32> -> !pto.vreg<128xf16>
%odd  = pto.vcvt %in1 {round_mode = "ROUND_R", sat = "RS_ENABLE", part = "PART_ODD"}
    : !pto.vreg<64xf32> -> !pto.vreg<128xf16>
%result = pto.vor %even, %odd, %mask : !pto.vreg<128xf16>, !pto.vreg<128xf16>, !pto.mask -> !pto.vreg<128xf16>
```

---

## `pto.vtrc`

- **syntax:** `%result = pto.vtrc %input, "ROUND_MODE" : !pto.vreg<NxT> -> !pto.vreg<NxT>`
- **semantics:** Truncate/round float to integer-valued float (stays in float type).

```c
for (int i = 0; i < N; i++)
    dst[i] = round_to_int_valued_float(src[i], round_mode);
```

**Example:**
```mlir
// Round to nearest integer, keep as float
%rounded = pto.vtrc %input, "ROUND_R" : !pto.vreg<64xf32> -> !pto.vreg<64xf32>
// input:  [1.4, 2.6, -1.5, 3.0]
// output: [1.0, 3.0, -2.0, 3.0]
```

---

## Typical Usage

```mlir
// Quantization: f32 → i8 with saturation
%scaled = pto.vmuls %input, %scale, %mask : !pto.vreg<64xf32>, f32, !pto.mask -> !pto.vreg<64xf32>
%quantized = pto.vcvt %scaled {round_mode = "ROUND_R", sat = "RS_ENABLE"}
    : !pto.vreg<64xf32> -> !pto.vreg<64xi32>
// Then narrow i32 → i8 via pack ops

// Mixed precision: bf16 → f32 for accumulation
%f32_vec = pto.vcvt %bf16_input {round_mode = "ROUND_R"}
    : !pto.vreg<128xbf16> -> !pto.vreg<64xf32>

// Floor for integer division
%floored = pto.vtrc %ratio, "ROUND_F" : !pto.vreg<64xf32> -> !pto.vreg<64xf32>
%int_div = pto.vcvt %floored {round_mode = "ROUND_Z"}
    : !pto.vreg<64xf32> -> !pto.vreg<64xi32>
```
