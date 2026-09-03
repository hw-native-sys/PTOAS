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

- **syntax:** `%result = pto.vci %index {order = "ASC|DESC"} : T -> !pto.vreg<NxT>`
- **semantics:** Generate a lane-index vector from a scalar base value.
- **inputs:**
  `%index` is the scalar base value. Supported scalar types are `i8/i16/i32`,
  `f16`, and `f32`.
- **outputs:**
  `%result` is the generated index vector.
- **constraints and limitations:**
  This is an index-generation family, not a numeric conversion. `order` and
  the result element type together determine whether lanes are generated as
  `base + lane_id` or `base - lane_id`. Supported result types are
  `!pto.vreg<256xsi8>`, `!pto.vreg<128xsi16>`, `!pto.vreg<64xsi32>`,
  `!pto.vreg<128xf16>`, and `!pto.vreg<64xf32>`. `%index` must use the
  matching scalar type for `f16`/`f32`; for integer results, `%index` must use
  the same bit width and may be signless or signed.

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
destination lane group.

- `Part` (`PART_EVEN`, `PART_ODD`)
  - Used by ordinary width-changing conversions.
  - Typical cases include `32 -> 16`, `16 -> 32`, and other even/odd packing
    or unpacking forms.
- `Part_T` (`PART_P0`, `PART_P1`, `PART_P2`, `PART_P3`)
  - Used by lower-level packed placement forms.
  - Typical cases include `32 -> 8`, packed fp8/fp4 conversion paths, and
    other flows where the result is written into one of four sub-parts before a
    later merge or compact step.

| Mode | Description |
|------|-------------|
| `EVEN` | Output to even-indexed lanes |
| `ODD` | Output to odd-indexed lanes |
| `P0` | Output to sub-part 0 in 4-way packed placement forms |
| `P1` | Output to sub-part 1 in 4-way packed placement forms |
| `P2` | Output to sub-part 2 in 4-way packed placement forms |
| `P3` | Output to sub-part 3 in 4-way packed placement forms |

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
- `%dst = pto.vcvt %src, %mask {rnd, sat} : !pto.vreg<128xbf16>, !pto.mask<b16> -> !pto.vreg<128xf16>`
- `%dst = pto.vcvt %src, %mask {rnd} : !pto.vreg<128xf16>, !pto.mask<b16> -> !pto.vreg<128xbf16>`
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
| `f16` | Y | Y |  | Y |  | Y |  |  | Y | Y |
| `f32` |  |  |  | Y |  | Y | Y | Y |  | Y |
| `bf16` |  |  |  |  |  | Y |  | Y | Y |  |

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

- **syntax:** `%result = pto.vtrc %input, %mask, "RND" : !pto.vreg<NxT>, !pto.mask<BW> -> !pto.vreg<NxT>`
- **semantics:** Truncate/round float to integer-valued float (stays in float type).

```c
for (int i = 0; i < N; i++)
    dst[i] = round_to_int_valued_float(src[i], rnd);
```

- **inputs:**
  `%input` is the floating-point source vector, `%mask` selects active lanes,
  and `RND` selects the truncation/rounding rule.
- **outputs:**
  `%result` is still a floating-point vector, but each active lane now carries
  an integer-valued floating-point result.
- **constraints and limitations:**
  This op does not change the element type. `T` must be `f16`, `f32`, or
  `bf16`. `RND` must be one of `R`, `A`, `F`, `C`, or `Z`. `BW` must match the
  element width: `b16` for `f16`/`bf16`, `b32` for `f32`.

**Example:**
```mlir
// Round to nearest integer, keep as float
%rounded = pto.vtrc %input, %mask, "R" : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
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
%floored = pto.vtrc %ratio, %mask, "F" : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
%int_div = pto.vcvt %floored, %mask {rnd = "Z"}
    : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xi32>
```

---

## `pto.vbitcast`

- **syntax:** `%result = pto.vbitcast %input : !pto.vreg<NxT0> -> !pto.vreg<MxT1>`
- **semantics:** Bitwise reinterpretation of a vreg vector without changing the underlying bit pattern. This operation performs a pure type cast that preserves the exact bits of each element, changing only their interpretation (e.g., from floating-point to integer).

- **inputs:**
  `%input` is the source vector register value.
- **outputs:**
  `%result` is the reinterpreted vector register value.
- **constraints and limitations:**
  1. Both source and result must be `!pto.vreg<...>` types.
  2. Source and result vectors must have the same total bit width (currently 2048 bits).
  3. Only integer and floating-point element types are supported.

**Element bit-width equality examples:**
- `f32<64>` → `i32<64>`  (both 32-bit elements, total 2048 bits)
- `f16<128>` → `i16<128>` (both 16-bit elements, total 2048 bits)
- `bf16<128>` → `ui16<128>` (both 16-bit elements, total 2048 bits)
- `si32<64>` → `ui32<64>` (both 32-bit elements, total 2048 bits)
- `f32<64>` → `i16<128>` (32-bit/16-bit elements, total 2048 bits)

**Verification:** The operation verifies that:
1. Both input and result are `!pto.vreg<...>` types.
2. Total bit width equals 2048 (the fixed vreg size).

**Comparison with `pto.vcvt`:**
- `pto.vcvt` performs value conversion with rounding, saturation, and lane placement control.
- `pto.vbitcast` performs bitwise reinterpretation without changing the underlying bit pattern.

**Example: Reinterpreting float as integer for bit manipulation**
```mlir
// Prepare a vector of float values
%fvec = pto.vlds %ub[%lane] : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>

// Reinterpret as integer for bitwise operations
%ivec = pto.vbitcast %fvec : !pto.vreg<64xf32> -> !pto.vreg<64xi32>

// Extract sign bit (bit 31)
%sign_bits = pto.vand %ivec, %sign_mask, %mask : !pto.vreg<64xi32>, !pto.vreg<64xi32>, !pto.mask<b32> -> !pto.vreg<64xi32>

// Reinterpret back to float
%fvec_without_sign = pto.vbitcast %sign_bits : !pto.vreg<64xi32> -> !pto.vreg<64xf32>
```

**Example: Type punning between signed and unsigned integer**
```mlir
// Convert signed to unsigned without changing bits
%signed = pto.vlds %ub[%lane] : !pto.ptr<si32, ub> -> !pto.vreg<64xsi32>
%unsigned = pto.vbitcast %signed : !pto.vreg<64xsi32> -> !pto.vreg<64xui32>
// Bits are identical; interpretation changes from signed to unsigned
```

## `pto.pbitcast`

- **syntax:** `%result = pto.pbitcast %input : !pto.mask<G0> -> !pto.mask<G1>`
- **semantics:** Bitwise reinterpretation of a predicate register without
  changing the underlying predicate-register image. This op makes mask-family
  reinterpretation explicit in VPTO IR when a producer and consumer expect
  different `!pto.mask<...>` views of the same hardware predicate state.

- **inputs:**
  `%input` is the source predicate register value.
- **outputs:**
  `%result` is the reinterpreted predicate register value.
- **constraints and limitations:**
  1. Both source and result must be `!pto.mask<...>` types.
  2. `pto.pbitcast` does not materialize or normalize predicate contents; it
     only changes which mask granularity the surrounding VPTO IR uses to
     interpret the same predicate bits.

**Example: Reinterpret a b16 predicate as b32 before a consumer**
```mlir
%m16 = pto.pintlv_b16 %lhs, %rhs : !pto.mask<b16>, !pto.mask<b16> -> !pto.mask<b16>, !pto.mask<b16>
%m32 = pto.pbitcast %m16#0 : !pto.mask<b16> -> !pto.mask<b32>
%result = pto.vsel %a, %b, %m32 : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
```
