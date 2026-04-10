# 9. Conversion Ops

> **Category:** Type conversion operations
> **Pipeline:** PIPE_V (Vector Core)

Operations that convert between data types (float/int, narrowing/widening).

## Common Operand Model

- `%input` is the source vector register value.
- `%mask` is the predicate mask that selects active conversion lanes.
- `%result` is the destination vector register value.
- `rnd`, `sat`, and `part` are optional attributes that refine
  conversion behavior when the selected source/destination type pair needs
  rounding, saturation, or lane placement control.
- The single `pto.vcvt` surface covers float-int, float-float, int-float, and
  int-int conversion families.

## CA latency (A5, Ascend910_9599 CA)

Cycle-accurate simulator **popped→retire** latency (cycles). Only representative traces below; other `pto.vcvt` conversion pairs depend on the RV lowering in the trace.

| PTO op | RV (CA) | Note | Latency |
|--------|---------|------|---------|
| `pto.vcvt` | `RV_VCVT_F2F` | f32→f16 | **7** |
| `pto.vci` | — | no vector `RV_*` in sampled `veccore0` trace | — |

---

## `pto.vci`

- **syntax:** `%result = pto.vci %index {order = "ORDER"} : integer -> !pto.vreg<NxT>`
- **semantics:** Generate a lane-index vector from a scalar seed/index value.
- **inputs:**
  `%index` is the scalar seed or base index.
- **outputs:**
  `%result` is the generated index vector.
- **constraints and limitations:**
  This is an index-generation family, not a numeric conversion. `ORDER` and the
  result element type together determine how indices are generated. `%result`
  uses an integer element type, and the scalar `%index` type matches that
  result element type.

---

## `pto.vcvt`

- **syntax:** `%result = pto.vcvt %input, %mask {rnd = "RND", sat = "SAT", part = "PART"} : !pto.vreg<NxT0>, !pto.mask<G> -> !pto.vreg<MxT1>`
- **semantics:** Type conversion between float/int types with rounding control.

```c
for (int i = 0; i < min(N, M); i++)
    if (mask[i])
        dst[i] = convert(src[i], T0, T1, rnd);
```

- **inputs:**
  `%input` is the source vector, `%mask` selects active lanes, and attributes
  select rounding, saturation, and output placement when the conversion changes
  width or packs into sub-lane positions.
- **outputs:**
  `%result` is the converted vector.
- **constraints and limitations:**
  Only documented source/destination type pairs are legal. All three
  attributes are optional at the surface level, but only the subset meaningful
  to the selected conversion kind should be provided. The execution mask must
  use the typed-mask granularity that matches the source vector family on the
  current surface; there is no `!pto.mask<b64>` form in VPTO.

---

### Rounding Modes

| Mode | Description |
|------|-------------|
| `R` | Round to nearest, ties to even (default) |
| `A` | Round away from zero |
| `F` | Round toward negative infinity (floor) |
| `C` | Round toward positive infinity (ceil) |
| `Z` | Round toward zero (truncate) |
| `O` | Round to odd |

---

### Saturation Modes

| Mode | Description |
|------|-------------|
| `SAT` | Saturate on overflow |
| `NOSAT` | No saturation (wrap/undefined on overflow) |

---

### Part Modes

Use `part` when a width-changing conversion writes only one half of each wider
destination lane group. This is typically used in even/odd placement forms such
as `32 -> 16` or `16 -> 32` style conversions.

| Mode | Description |
|------|-------------|
| `EVEN` | Output to even-indexed lanes |
| `ODD` | Output to odd-indexed lanes |

---

### Attribute Guidance

- `rnd`
  - Use when the conversion needs an explicit rounding rule, especially for
    float-to-int, float-to-float narrowing, or integer-to-float forms that do
    not map exactly.
- `mask`
  - Use to select which source lanes participate in the conversion. In
    width-changing conversions, `mask` works together with `part` / `pp` to
    determine which logical lane positions are produced.
- `sat`
  - Use when the conversion may overflow the destination range and hardware
    exposes a saturating form.
- `part`
  - Use for width-changing conversions that select the even or odd half of the
    destination packing layout.

#### Float To Int

- `%dst = pto.vcvt %src, %mask {rnd, sat, part} : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<32xsi64>`
- `%dst = pto.vcvt %src, %mask {rnd, sat} : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xsi32>`
- `%dst = pto.vcvt %src, %mask {rnd, sat, part} : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<128xsi16>`
- `%dst = pto.vcvt %src, %mask {rnd, part} : !pto.vreg<128xf16>, !pto.mask<b16> -> !pto.vreg<64xsi32>`
- `%dst = pto.vcvt %src, %mask {rnd, sat} : !pto.vreg<128xf16>, !pto.mask<b16> -> !pto.vreg<128xsi16>`
- `%dst = pto.vcvt %src, %mask {rnd, sat, part} : !pto.vreg<128xf16>, !pto.mask<b16> -> !pto.vreg<256xsi8>`
- `%dst = pto.vcvt %src, %mask {rnd, sat, part} : !pto.vreg<128xf16>, !pto.mask<b16> -> !pto.vreg<256xui8>`
- `%dst = pto.vcvt %src, %mask {rnd, sat, part} : !pto.vreg<128xbf16>, !pto.mask<b16> -> !pto.vreg<64xsi32>`

#### Float To Float

- `%dst = pto.vcvt %src, %mask {rnd, sat, part} : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<128xf16>`
- `%dst = pto.vcvt %src, %mask {rnd, sat, part} : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<128xbf16>`
- `%dst = pto.vcvt %src, %mask {part} : !pto.vreg<128xf16>, !pto.mask<b16> -> !pto.vreg<64xf32>`
- `%dst = pto.vcvt %src, %mask {part} : !pto.vreg<128xbf16>, !pto.mask<b16> -> !pto.vreg<64xf32>`

#### Int To Float

- `%dst = pto.vcvt %src, %mask {part} : !pto.vreg<256xui8>, !pto.mask<b8> -> !pto.vreg<128xf16>`
- `%dst = pto.vcvt %src, %mask {part} : !pto.vreg<256xsi8>, !pto.mask<b8> -> !pto.vreg<128xf16>`
- `%dst = pto.vcvt %src, %mask {rnd} : !pto.vreg<128xsi16>, !pto.mask<b16> -> !pto.vreg<128xf16>`
- `%dst = pto.vcvt %src, %mask {part} : !pto.vreg<128xsi16>, !pto.mask<b16> -> !pto.vreg<64xf32>`
- `%dst = pto.vcvt %src, %mask {rnd} : !pto.vreg<64xsi32>, !pto.mask<b32> -> !pto.vreg<64xf32>`

#### Int To Int

- `%dst = pto.vcvt %src, %mask {part} : !pto.vreg<256xui8>, !pto.mask<b8> -> !pto.vreg<128xui16>`
- `%dst = pto.vcvt %src, %mask {part} : !pto.vreg<256xui8>, !pto.mask<b8> -> !pto.vreg<64xui32>`
- `%dst = pto.vcvt %src, %mask {part} : !pto.vreg<256xsi8>, !pto.mask<b8> -> !pto.vreg<128xsi16>`
- `%dst = pto.vcvt %src, %mask {part} : !pto.vreg<256xsi8>, !pto.mask<b8> -> !pto.vreg<64xsi32>`
- `%dst = pto.vcvt %src, %mask {sat, part} : !pto.vreg<128xui16>, !pto.mask<b16> -> !pto.vreg<256xui8>`
- `%dst = pto.vcvt %src, %mask {part} : !pto.vreg<128xui16>, !pto.mask<b16> -> !pto.vreg<64xui32>`
- `%dst = pto.vcvt %src, %mask {sat, part} : !pto.vreg<128xsi16>, !pto.mask<b16> -> !pto.vreg<256xui8>`
- `%dst = pto.vcvt %src, %mask {part} : !pto.vreg<128xsi16>, !pto.mask<b16> -> !pto.vreg<64xui32>`
- `%dst = pto.vcvt %src, %mask {part} : !pto.vreg<128xsi16>, !pto.mask<b16> -> !pto.vreg<64xsi32>`
- `%dst = pto.vcvt %src, %mask {sat, part} : !pto.vreg<64xui32>, !pto.mask<b32> -> !pto.vreg<256xui8>`
- `%dst = pto.vcvt %src, %mask {sat, part} : !pto.vreg<64xui32>, !pto.mask<b32> -> !pto.vreg<128xui16>`
- `%dst = pto.vcvt %src, %mask {sat, part} : !pto.vreg<64xui32>, !pto.mask<b32> -> !pto.vreg<128xsi16>`
- `%dst = pto.vcvt %src, %mask {sat, part} : !pto.vreg<64xsi32>, !pto.mask<b32> -> !pto.vreg<256xui8>`
- `%dst = pto.vcvt %src, %mask {sat, part} : !pto.vreg<64xsi32>, !pto.mask<b32> -> !pto.vreg<128xui16>`
- `%dst = pto.vcvt %src, %mask {sat, part} : !pto.vreg<64xsi32>, !pto.mask<b32> -> !pto.vreg<128xsi16>`
- `%dst = pto.vcvt %src, %mask {part} : !pto.vreg<64xsi32>, !pto.mask<b32> -> !pto.vreg<32xsi64>`

### A5 Supported Type Matrix

The table below is only a summary. For exact attribute combinations, use the
per-form entries above as the source of truth.

| `src \ dst` | `ui8` | `si8` | `ui16` | `si16` | `ui32` | `si32` | `si64` | `f16` | `f32` | `bf16` |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `ui8` |  |  | Y |  | Y |  |  | Y |  |  |
| `si8` |  |  |  | Y |  | Y |  | Y |  |  |
| `ui16` | Y |  |  |  | Y |  |  |  |  |  |
| `si16` | Y |  |  |  | Y | Y |  | Y | Y |  |
| `ui32` | Y |  | Y | Y |  |  |  |  |  |  |
| `si32` | Y |  | Y | Y |  |  | Y |  | Y |  |
| `si64` |  |  |  |  |  |  |  |  |  |  |
| `f16` | Y | Y |  | Y |  | Y |  |  | Y |  |
| `f32` |  |  |  | Y |  | Y | Y | Y |  | Y |
| `bf16` |  |  |  |  |  | Y |  |  | Y |  |

---

### Width-Changing Conversion Pattern

For conversions that change width (e.g., f32→f16), use even/odd parts and combine:

```mlir
// Convert two f32 vectors to one f16 vector
%even = pto.vcvt %in0, %mask {rnd = "R", sat = "SAT", part = "EVEN"}
    : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<128xf16>
%odd  = pto.vcvt %in1, %mask {rnd = "R", sat = "SAT", part = "ODD"}
    : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<128xf16>
%result = pto.vor %even, %odd, %mask : !pto.vreg<128xf16>, !pto.vreg<128xf16>, !pto.mask<b16> -> !pto.vreg<128xf16>
```

---

## `pto.vtrc`

- **syntax:** `%result = pto.vtrc %input, "RND" : !pto.vreg<NxT> -> !pto.vreg<NxT>`
- **semantics:** Truncate/round float to integer-valued float (stays in float type).

```c
for (int i = 0; i < N; i++)
    dst[i] = round_to_int_valued_float(src[i], rnd);
```

- **inputs:**
  `%input` is the floating-point source vector and `RND` selects the
  truncation/rounding rule.
- **outputs:**
  `%result` is still a floating-point vector, but each active lane now carries
  an integer-valued floating-point result.
- **constraints and limitations:**
  This op does not change the element type. `O` is supported for avoiding
  double-rounding errors during staged conversions.

**Example:**
```mlir
// Round to nearest integer, keep as float
%rounded = pto.vtrc %input, "R" : !pto.vreg<64xf32> -> !pto.vreg<64xf32>
// input:  [1.4, 2.6, -1.5, 3.0]
// output: [1.0, 3.0, -2.0, 3.0]
```

---

## Typical Usage

```mlir
// Quantization: f32 → i8 with saturation
%scaled = pto.vmuls %input, %scale, %mask : !pto.vreg<64xf32>, f32, !pto.mask<b32> -> !pto.vreg<64xf32>
%quantized = pto.vcvt %scaled, %mask {rnd = "R", sat = "SAT"}
    : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xi32>
// Then narrow i32 → i8 via pack ops

// Mixed precision: bf16 → f32 for accumulation
%f32_vec = pto.vcvt %bf16_input, %mask {part = "EVEN"}
    : !pto.vreg<128xbf16>, !pto.mask<b16> -> !pto.vreg<64xf32>

// Floor for integer division
%floored = pto.vtrc %ratio, "F" : !pto.vreg<64xf32> -> !pto.vreg<64xf32>
%int_div = pto.vcvt %floored, %mask {rnd = "Z"}
    : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xi32>
```
