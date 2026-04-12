# 12. Data Rearrangement

> **Category:** In-register data movement and permutation
> **Pipeline:** PIPE_V (Vector Core)

Operations that rearrange data within or between vector registers without memory access.

## Common Operand Model

- `%lhs` / `%rhs` are source vector register values.
- `%src` is a single source vector register value.
- `%result` is the destination vector register value unless an op explicitly
  returns multiple vectors.
- These families do not access UB directly; they only rearrange register
  contents.

---

## Interleave / Deinterleave

### `pto.vintlv`

- **syntax:** `%low, %high = pto.vintlv %lhs, %rhs : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<NxT>, !pto.vreg<NxT>`
- **semantics:** Interleave elements from two sources.

```c
// Interleave: merge even/odd elements from two sources
// low  = {src0[0], src1[0], src0[1], src1[1], ...}
// high = {src0[N/2], src1[N/2], src0[N/2+1], src1[N/2+1], ...}
```

- **inputs:** `%lhs` and `%rhs` are the two source vectors.
- **outputs:** `%low` and `%high` are the two destination vectors.
- **constraints and limitations:** The two outputs form a paired interleave
  result. The PTO micro Instruction representation exposes that pair as two SSA results, and the pair ordering MUST
  be preserved.

---

### `pto.vdintlv`

- **syntax:** `%low, %high = pto.vdintlv %lhs, %rhs : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<NxT>, !pto.vreg<NxT>`
- **semantics:** Deinterleave elements into even/odd.

```c
// Deinterleave: separate even/odd elements
// low  = {src0[0], src0[2], src0[4], ...}  // even
// high = {src0[1], src0[3], src0[5], ...}  // odd
```

- **inputs:** `%lhs` and `%rhs` represent the interleaved source stream in the
  current PTO micro Instruction representation.
- **outputs:** `%low` and `%high` are the separated destination vectors.
- **constraints and limitations:** The two outputs form the even/odd
  deinterleave result pair, and their ordering MUST be preserved.

---

## Compress / Expand

### `pto.vsqz`

- **syntax:** `%result = pto.vsqz %src, %mask : !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>`
- **semantics:** Compress — pack active lanes to front.

```c
int j = 0;
for (int i = 0; i < N; i++)
    if (mask[i]) dst[j++] = src[i];
while (j < N) dst[j++] = 0;
```

**Use case:** Sparse data compaction, filtering.

- **inputs:** `%src` is the source vector and `%mask` selects which elements are
  kept.
- **outputs:** `%result` is the compacted vector.
- **constraints and limitations:** This is a reduction-style compaction family.
  Preserved element order MUST match source lane order.

---

### `pto.vusqz`

- **syntax:** `%result = pto.vusqz %src, %mask : !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>`
- **semantics:** Generate per-lane prefix counts from the governing predicate.

```c
dst[0] = 0;
for (int i = 1; i < N; i++)
    dst[i] = mask[i - 1] ? (dst[i - 1] + 1) : dst[i - 1];
```

- **inputs:** `%mask` is the governing predicate. The current PTO surface keeps
  `%src` in the operand list for interface compatibility, but the observable
  result semantics are determined by `%mask`.
- **outputs:** `%result[i]` equals the number of active lanes in `%mask[0:i)`,
  with `%result[0] = 0`.
- **constraints and limitations:** `T` is currently limited to `si8`, `si16`,
  or `si32`. This operation is a predicate-derived counting/rearrangement
  primitive rather than a value-placement primitive. The final predicate lane
  does not contribute to a later output lane because there is no `dst[N]`.

---

---

### `pto.vselr`

- **syntax:** `%result = pto.vselr %src, %idx : !pto.vreg<NxT>, !pto.vreg<Nxi<width>> -> !pto.vreg<NxT>`
- **semantics:** Register lane-select with an explicit index vector.

```c
for (int i = 0; i < N; i++)
    dst[i] = src[idx[i]];
```

- **inputs:** `%src` is the source vector. `%idx` is the lane-index vector.
- **outputs:** `%result` is the reordered vector.
- **constraints and limitations:** This page records the rearrangement use of
  the family; the compare/select page documents the same name from the predicate
  selection perspective.

---

## Pack / Unpack

### `pto.vpack`

- **syntax:** `%result = pto.vpack %src, "PART" : !pto.vreg<NxT_wide> -> !pto.vreg<2NxT_narrow>`
- **semantics:** Narrow one wide vector and place the narrowed payload into the
  selected half of the result. The other half is filled with zero.

```c
// e.g., vreg<64xi32> → vreg<128xui16>
for (int i = 0; i < N; i++)
    dst[i] = 0;

if (part == LOWER) {
    for (int i = 0; i < N; i++)
        dst[i] = truncate(src[i]);
} else { // HIGHER
    for (int i = 0; i < N; i++)
        dst[N + i] = truncate(src[i]);
}
```

- **inputs:** `%src` is the wide source vector. `"LOWER"` and `"HIGHER"`
  select whether the narrowed payload lands in the lower or upper half.
- **outputs:** `%result` is the packed narrow vector.
- **constraints and limitations:** Packing is a narrowing conversion with
  truncation semantics. Current VPTO surface supports `i32/ui32 -> ui16` and
  `i16/ui16 -> ui8`.

---

### `pto.vsunpack`

- **syntax:** `%result = pto.vsunpack %src, %part : !pto.vreg<NxT_narrow>, index -> !pto.vreg<N/2xT_wide>`
- **semantics:** Sign-extending unpack — narrow to wide (half).

```c
// e.g., vreg<128xi16> → vreg<64xi32> (one half)
for (int i = 0; i < N/2; i++)
    dst[i] = sign_extend(src[part_offset + i]);
```

- **inputs:** `%src` is the packed narrow vector and `%part` selects which half
  is unpacked.
- **outputs:** `%result` is the widened vector.
- **constraints and limitations:** This is the sign-extending unpack family.

---

### `pto.vzunpack`

- **syntax:** `%result = pto.vzunpack %src, %part : !pto.vreg<NxT_narrow>, index -> !pto.vreg<N/2xT_wide>`
- **semantics:** Zero-extending unpack — narrow to wide (half).

```c
for (int i = 0; i < N/2; i++)
    dst[i] = zero_extend(src[part_offset + i]);
```

- **inputs:** `%src` is the packed narrow vector and `%part` selects which half
  is unpacked.
- **outputs:** `%result` is the widened vector.
- **constraints and limitations:** This is the zero-extending unpack family.

---

## Typical Usage

```mlir
// AoS → SoA conversion using deinterleave
%even, %odd = pto.vdintlv %interleaved0, %interleaved1
    : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>, !pto.vreg<64xf32>

// Filter: keep only elements passing condition
%pass_mask = pto.vcmps %values, %threshold, %all, "gt"
    : !pto.vreg<64xf32>, f32, !pto.mask<G> -> !pto.mask<G>
%compacted = pto.vsqz %values, %pass_mask
    : !pto.vreg<64xf32>, !pto.mask<G> -> !pto.vreg<64xf32>

// Type narrowing via pack
%packed_i16 = pto.vpack %wide_i32, "LOWER"
  : !pto.vreg<64xi32> -> !pto.vreg<128xui16>
```

---

## V2 Interleave Forms

### `pto.vintlvv2`

- **syntax:** `%result = pto.vintlvv2 %lhs, %rhs, "PART" : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<NxT>`
- **inputs:** `%lhs` and `%rhs` are source vectors and `PART` selects the
  returned half of the V2 interleave result.
- **outputs:** `%result` is the selected interleave half.
- **constraints and limitations:** This op exposes only one half of the V2
  result in SSA form.

### `pto.vdintlvv2`

- **syntax:** `%result = pto.vdintlvv2 %lhs, %rhs, "PART" : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<NxT>`
- **inputs:** `%lhs` and `%rhs` are source vectors and `PART` selects the
  returned half of the V2 deinterleave result.
- **outputs:** `%result` is the selected deinterleave half.
- **constraints and limitations:** This op exposes only one half of the V2
  result in SSA form.
