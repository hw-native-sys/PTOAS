# 4. Predicate Load/Store

> **Category:** UB ↔ Predicate Register data movement
> **Pipeline:** PIPE_V (Vector Core)

Predicate registers (`!pto.mask<G>`) are 256-bit registers that enable per-lane conditional execution. These ops move predicate values between UB and predicate registers.

In concrete examples, `G` should be chosen to match the consumer family. The
examples below use `b32` when the loaded/stored mask is used with `f32`
vector compares or selects.

The predicate load/store ops documented on this page always use explicit
`base[offset]` addressing. The immediate forms (`pldi`, `psti`) and dynamic
forms (`plds`, `psts`) differ only in how `%offset` is supplied.

---

## Predicate Loads

### `pto.plds`

- **syntax:** `%result = pto.plds %source[%offset], "DIST" : !pto.ptr<T, ub>, index -> !pto.mask<G>`
- **semantics:** Load predicate register with runtime offset. This is the
  dynamic-offset form of `pto.pldi`: the predicate payload interpretation is
  the same, but `%offset` is supplied as an SSA `index` instead of a constant
  `index` immediate.
- **DIST:** mandatory string token, one of `NORM`, `US`, `DS`.
  - `NORM`: load a normal packed predicate payload of size `VL/8`.
  - `US`: load a packed predicate payload of size `VL/16`, then duplicate each
    loaded bit once.
  - `DS`: load a packed predicate payload of size `2 * VL/8`, then keep one
    bit out of every two bits.

The loaded payload is a packed predicate image in UB. Consumer ops interpret
the resulting `!pto.mask<G>` according to the mask granularity `G`.
`pto.plds` only
models the explicit `base[offset]` form.

**Example:**
```mlir
%mask = pto.plds %ub[%c0], "NORM" : !pto.ptr<T, ub>, index -> !pto.mask<G>
```

---

### `pto.pldi`

- **syntax:** `%result = pto.pldi %source[%offset], "DIST" : !pto.ptr<T, ub>, index -> !pto.mask<G>`
- **offset:** must be a constant `index` immediate in PTO surface form.
- **semantics:** Load predicate register with immediate offset.
- **DIST:** mandatory string token, one of `NORM`, `US`, `DS`.
  - `NORM`: load a normal packed predicate payload of size `VL/8`.
  - `US`: load a packed predicate payload of size `VL/16`, then duplicate each
    loaded bit once.
  - `DS`: load a packed predicate payload of size `2 * VL/8`, then keep one
    bit out of every two bits.

Like `pto.plds`, this op reads a packed predicate payload from UB and
materializes it as `!pto.mask<G>`.

---

## Predicate Stores

### `pto.psts`

- **syntax:** `pto.psts %value, %dest[%offset], "DIST" : !pto.mask<G>, !pto.ptr<T, ub>, index`
- **semantics:** Store predicate register with runtime offset. This is the
  dynamic-offset form of `pto.psti`: the predicate payload interpretation is
  the same, but `%offset` is supplied as an SSA `index` instead of a constant
  `index` immediate.
- **DIST:** mandatory string token, one of `NORM`, `PK`.
  - `NORM`: store the packed predicate payload into a normal destination space
    of size `VL/8`.
  - `PK`: store the packed predicate payload into a destination space of size
    `VL/16`, keeping one bit out of every two bits.

`pto.psts` stores the packed predicate payload represented by `!pto.mask<G>`.
It only models the explicit `base[offset]` form.

**Example:**
```mlir
pto.psts %mask, %ub[%c0], "NORM" : !pto.mask<G>, !pto.ptr<T, ub>, index
```

---

### `pto.psti`

- **syntax:** `pto.psti %value, %dest[%offset], "DIST" : !pto.mask<G>, !pto.ptr<T, ub>, index`
- **offset:** must be a constant `index` immediate in PTO surface form.
- **semantics:** Store predicate register with immediate offset.
- **DIST:** mandatory string token, one of `NORM`, `PK`.
  - `NORM`: store the packed predicate payload into a normal destination space
    of size `VL/8`.
  - `PK`: store the packed predicate payload into a destination space of size
    `VL/16`, keeping one bit out of every two bits.

`pto.psti` and `pto.psts` store the packed predicate payload represented by
`!pto.mask<G>`. The surface distinction is only immediate-offset versus
dynamic-offset.

---

### `pto.pstu`

- **syntax:** `%align_out, %base_out = pto.pstu %align_in, %value, %base : !pto.align, !pto.mask<b16>, !pto.ptr<ui16, ub> -> !pto.align, !pto.ptr<ui16, ub>`
- **syntax:** `%align_out, %base_out = pto.pstu %align_in, %value, %base : !pto.align, !pto.mask<b32>, !pto.ptr<ui32, ub> -> !pto.align, !pto.ptr<ui32, ub>`
- **semantics:** Predicate unaligned store with align/base state update. The base type is fixed by mask granularity: `b16 <-> ui16`, `b32 <-> ui32`.
- **outputs:**
  `%align_out` and `%base_out` are the updated unaligned-store state and are
  intended to be used by a later `pto.pstu` call.
- **constraints and limitations:**
  The first `%align_in` in a predicate unaligned-store stream should come from
  `pto.init_align`.

---

## Typical Usage Pattern

```mlir
// Generate comparison mask
%mask = pto.vcmp %v0, %v1, %seed, "lt" : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.mask<b32>

// Store mask to UB for later use
pto.psts %mask, %ub_mask[%c0], "NORM" : !pto.mask<b32>, !pto.ptr<T, ub>, index

// ... later in another kernel ...

// Load mask from UB
%saved_mask = pto.plds %ub_mask[%c0], "NORM" : !pto.ptr<T, ub>, index -> !pto.mask<b32>

// Use for predicated select
%result = pto.vsel %v_true, %v_false, %saved_mask : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
```
