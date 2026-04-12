# 5. Materialization & Predicate Ops

> **Category:** Scalar broadcast, predicate generation and manipulation
> **Pipeline:** PIPE_V (Vector Core)

These ops create vectors from scalar values and manipulate predicate registers.

## Common Operand Model

- `%value` is the scalar source value in SSA form.
- `%input` is either a source scalar or a source vector depending on the op.
- `%result` is the destination vector register value.
- For 32-bit scalar inputs, the scalar source MUST satisfy the backend's legal
  scalar-source constraints for this family.

---

## Scalar Materialization

### `pto.vbr`

- **syntax:** `%result = pto.vbr %value : T -> !pto.vreg<NxT>`
- **semantics:** Broadcast scalar to all vector lanes.
- **inputs:**
  `%value` is the scalar source.
- **outputs:**
  `%result` is a vector whose active lanes all carry `%value`.
- **constraints and limitations:**
  Supported forms are `b8`, `b16`, and `b32`. For `b8`, only the low 8 bits of
  the scalar source are consumed.

```c
for (int i = 0; i < N; i++)
    dst[i] = value;
```

**Example:**
```mlir
%one = pto.vbr %c1_f32 : f32 -> !pto.vreg<64xf32>
```

---

### `pto.vdup`

- **syntax:** `%result = pto.vdup %input, %mask {position = "LOWEST|HIGHEST"} : T|!pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>`
- **semantics:** Duplicate scalar or vector element to all lanes.
- **inputs:**
  `%input` supplies the scalar or source-lane value selected by `position`,
  and `%mask` controls the active lanes.
- **outputs:**
  `%result` is the duplicated vector.
- **constraints and limitations:**
  `position` selects which source vector element is duplicated and is only valid
  for vector input. `position` defaults to `LOWEST`.

```c
for (int i = 0; i < N; i++)
    dst[i] = mask[i] ? input_scalar_or_element : 0;
```

---

## Predicate Generation

### `pto.pset_b8` / `pto.pset_b16` / `pto.pset_b32`

- **syntax:** `%result = pto.pset_b8 "PATTERN" : !pto.mask<b8>`
- **syntax:** `%result = pto.pset_b16 "PATTERN" : !pto.mask<b16>`
- **syntax:** `%result = pto.pset_b32 "PATTERN" : !pto.mask<b32>`
- **semantics:** Materialize a predicate register from a named pattern token.

**Supported pattern tokens:**

| Pattern | Description |
|---------|-------------|
| `PAT_ALL` | All lanes active |
| `PAT_ALLF` | All lanes inactive |
| `PAT_H` | High half active |
| `PAT_Q` | Upper quarter active |
| `PAT_VL1`...`PAT_VL128` | First N logical lanes active |
| `PAT_M3`, `PAT_M4` | Modular patterns |

`PAT_ALL` is the PTO spelling of the VISA-style all-true predicate pattern.
The other tokens listed above are also concrete installed-toolchain pattern
objects, not PTO-only aliases.

**Example — All 64 f32 lanes active:**
```mlir
%all_active = pto.pset_b32 "PAT_ALL" : !pto.mask<b32>
```

**Example — First 16 lanes active:**
```mlir
%first_16 = pto.pset_b32 "PAT_VL16" : !pto.mask<b32>
```

---

### `pto.pge_b8` / `pto.pge_b16` / `pto.pge_b32`

- **syntax:** `%result = pto.pge_b8 "PATTERN" : !pto.mask<b8>`
- **syntax:** `%result = pto.pge_b16 "PATTERN" : !pto.mask<b16>`
- **syntax:** `%result = pto.pge_b32 "PATTERN" : !pto.mask<b32>`
- **semantics:** Generate a predicate from a lane-count pattern token. In the
  common tail-mask form, `PAT_VL<N>` marks the first `N` logical lanes active.
- **supported pattern tokens:** `PAT_ALL`, `PAT_ALLF`, `PAT_H`, `PAT_Q`,
  `PAT_VL1`, `PAT_VL2`, `PAT_VL3`, `PAT_VL4`, `PAT_VL8`, `PAT_VL16`,
  `PAT_VL32`, `PAT_VL64`, `PAT_VL128`, `PAT_M3`, `PAT_M4`

```c
for (int i = 0; i < TOTAL_LANES; i++)
    mask[i] = (i < len);
```

**Example — Tail mask for remainder loop:**
```mlir
%tail_mask = pto.pge_b32 "PAT_VL8" : !pto.mask<b32>
```

---

### `pto.plt_b8` / `pto.plt_b16` / `pto.plt_b32`

- **syntax:** `%mask, %scalar_out = pto.plt_b8 %scalar : i32 -> !pto.mask<b8>, i32`
- **syntax:** `%mask, %scalar_out = pto.plt_b16 %scalar : i32 -> !pto.mask<b16>, i32`
- **syntax:** `%mask, %scalar_out = pto.plt_b32 %scalar : i32 -> !pto.mask<b32>, i32`
- **semantics:** Generate a tail-style predicate from an SSA lane-count value.
  On A5/V300-style toolchains, this family is exposed as a post-update wrapper:
  the predicate result becomes `%mask`, and the wrapper's carry-out scalar state
  is surfaced as `%scalar_out`.
- **inputs:**
  `%scalar` is the incoming lane-count / remaining-count state.
- **outputs:**
  `%mask` is the generated predicate.
  `%scalar_out` is the post-update scalar carry-out from the same `plt` call
  and can be threaded into a subsequent `pto.plt_b*` call in the same chain.

```c
for (int i = 0; i < VL_t; ++i)
    mask[i] = (i < scalar_in);

scalar_out = (scalar_in < VL_t) ? 0 : (scalar_in - VL_t);
```

Where `VL_t` is the logical lane count of the concrete op variant:

- `pto.plt_b8`: `VL_t = 256`
- `pto.plt_b16`: `VL_t = 128`
- `pto.plt_b32`: `VL_t = 64`

---

## Predicate Pack/Unpack

### `pto.ppack`

- **syntax:** `%result = pto.ppack %input, "PART" : !pto.mask<G> -> !pto.mask<G>`
- **semantics:** Narrowing pack of predicate register.
- **part tokens:**
  - `LOWER`: pack into the lower half of `%result`; the upper half is zeroed.
  - `HIGHER`: pack into the higher half of `%result`; the lower half is zeroed.

Conceptually, `pto.ppack` keeps one bit out of each adjacent 2-bit group from
`%input`, packs those kept bits into the selected half of `%result`, and fills
the other half with zeros.

```c
// Let VL be the logical lane count of the destination predicate.
// LOWER
for (int i = 0; i < VL / 2; ++i)
    result[i] = input[2 * i];
for (int i = VL / 2; i < VL; ++i)
    result[i] = 0;

// HIGHER
for (int i = 0; i < VL / 2; ++i)
    result[VL / 2 + i] = input[2 * i];
for (int i = 0; i < VL / 2; ++i)
    result[i] = 0;
```

---

### `pto.punpack`

- **syntax:** `%result = pto.punpack %input, "PART" : !pto.mask<G> -> !pto.mask<G>`
- **semantics:** Widening unpack of predicate register.
- **part tokens:**
  - `LOWER`: unpack from the lower half of `%input`.
  - `HIGHER`: unpack from the higher half of `%input`.

Conceptually, `pto.punpack` reads the selected half of `%input`, zero-extends
each 1-bit predicate element into a 2-bit group in `%result`, and leaves the
expanded image in the full destination predicate register.

```c
// Let VL be the logical lane count of the destination predicate.
// LOWER
for (int i = 0; i < VL / 2; ++i) {
    result[2 * i] = input[i];
    result[2 * i + 1] = 0;
}

// HIGHER
for (int i = 0; i < VL / 2; ++i) {
    result[2 * i] = input[VL / 2 + i];
    result[2 * i + 1] = 0;
}
```

---

## Predicate Logical Ops

### `pto.pand`

- **syntax:** `%result = pto.pand %src0, %src1, %mask : !pto.mask<G>, !pto.mask<G>, !pto.mask<G> -> !pto.mask<G>`
- **semantics:** Predicate bitwise AND gated by a governing predicate.

Inactive lanes selected out by `%mask` are zeroed.

```c
for (int i = 0; i < N; i++)
    dst[i] = mask[i] ? (src0[i] & src1[i]) : 0;
```

---

### `pto.por`

- **syntax:** `%result = pto.por %src0, %src1, %mask : !pto.mask<G>, !pto.mask<G>, !pto.mask<G> -> !pto.mask<G>`
- **semantics:** Predicate bitwise OR gated by a governing predicate.

Inactive lanes selected out by `%mask` are zeroed.

```c
for (int i = 0; i < N; i++)
    dst[i] = mask[i] ? (src0[i] | src1[i]) : 0;
```

---

### `pto.pxor`

- **syntax:** `%result = pto.pxor %src0, %src1, %mask : !pto.mask<G>, !pto.mask<G>, !pto.mask<G> -> !pto.mask<G>`
- **semantics:** Predicate bitwise XOR gated by a governing predicate.

Inactive lanes selected by `%mask` are zeroed.

```c
for (int i = 0; i < N; i++)
    dst[i] = mask[i] ? (src0[i] ^ src1[i]) : 0;
```

---

### `pto.pnot`

- **syntax:** `%result = pto.pnot %input, %mask : !pto.mask<G>, !pto.mask<G> -> !pto.mask<G>`
- **semantics:** Predicate bitwise NOT gated by a governing predicate.

Inactive lanes selected by `%mask` are zeroed.

```c
for (int i = 0; i < N; i++)
    dst[i] = mask[i] ? (~src[i]) : 0;
```

---

### `pto.psel`

- **syntax:** `%result = pto.psel %src0, %src1, %sel : !pto.mask<G>, !pto.mask<G>, !pto.mask<G> -> !pto.mask<G>`
- **semantics:** Predicate select (mux). `%sel` is the governing predicate that
  chooses lanes from `%src0` or `%src1`.

```c
for (int i = 0; i < N; i++)
    dst[i] = sel[i] ? src0[i] : src1[i];
```

---

### `pto.pdintlv_b8` / `pto.pdintlv_b16` / `pto.pdintlv_b32`

- **syntax:** `%low, %high = pto.pdintlv_b8 %src0, %src1 : !pto.mask<b8>, !pto.mask<b8> -> !pto.mask<b8>, !pto.mask<b8>`
- **syntax:** `%low, %high = pto.pdintlv_b16 %src0, %src1 : !pto.mask<b16>, !pto.mask<b16> -> !pto.mask<b16>, !pto.mask<b16>`
- **syntax:** `%low, %high = pto.pdintlv_b32 %src0, %src1 : !pto.mask<b32>, !pto.mask<b32> -> !pto.mask<b32>, !pto.mask<b32>`
- **semantics:** De-interleave two predicate sources and return the two
  de-interleaved predicate images in the same predicate element family.

---

### `pto.pintlv_b8` / `pto.pintlv_b16` / `pto.pintlv_b32`

- **syntax:** `%low, %high = pto.pintlv_b8 %src0, %src1 : !pto.mask<b8>, !pto.mask<b8> -> !pto.mask<b8>, !pto.mask<b8>`
- **syntax:** `%low, %high = pto.pintlv_b16 %src0, %src1 : !pto.mask<b16>, !pto.mask<b16> -> !pto.mask<b16>, !pto.mask<b16>`
- **syntax:** `%low, %high = pto.pintlv_b32 %src0, %src1 : !pto.mask<b32>, !pto.mask<b32> -> !pto.mask<b32>, !pto.mask<b32>`
- **semantics:** Interleave two predicate sources and return the two
  resulting predicate images in the same predicate element family.

---

## Typical Usage

```mlir
// Generate all-active mask for f32 (64 lanes)
%all = pto.pset_b32 "PAT_ALL" : !pto.mask<b32>

// Generate tail mask for remainder (last 12 elements)
%tail = pto.pge_b32 "PAT_VL12" : !pto.mask<b32>

// Compare and generate mask
%cmp_mask = pto.vcmp %a, %b, %all, "lt" : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.mask<b32>

// Combine masks: only process tail elements that passed comparison
%combined = pto.pand %cmp_mask, %tail, %all : !pto.mask<b32>, !pto.mask<b32>, !pto.mask<b32> -> !pto.mask<b32>

// Use for predicated operation
%result = pto.vsel %true_vals, %false_vals, %combined : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
```
