# 10. Reduction Ops

> **Category:** Vector reduction operations
> **Pipeline:** PIPE_V (Vector Core)

Operations that reduce a vector to a scalar or per-group result.

## Common Operand Model

- `%input` is the source vector register value.
- `%mask` is the predicate operand `Pg`; inactive lanes do not participate.
- `%result` is the destination vector register value.
- Reduction results are written into the low-significance portion of the
  destination vector and the remaining destination bits are zero-filled.
- A5 has no native 8-bit `vcadd`, `vcmax`, `vcmin`, `vcgadd`, `vcgmax`, or
  `vcgmin` form. Supported 8-bit VMI reductions extend their inputs before
  using these physical operations.

---

## Full Vector Reductions

### `pto.vcadd`

- **syntax:** `%result = pto.vcadd %input, %mask : !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<MxU>`
- **A5 types:** i16/i32 (signed, unsigned, or signless), f16, f32
- **semantics:** Sum all elements. Result in lane 0, others zeroed.

```c
U sum = 0;  // U is the result element type, widened for 16-bit integers.
for (int i = 0; i < N; i++)
    if (mask[i]) sum += (U)src[i];
dst[0] = sum;
for (int i = 1; i < M; i++)
    dst[i] = 0;
```

- **inputs:** `%input` is the source vector and `%mask` selects participating
  lanes.
- **outputs:** `%result` contains the reduction result in its low element(s).
- **constraints and limitations:** On A5, 16-bit integer inputs produce
  widened 32-bit integer results with the same signedness and half as many
  lanes (`M = N / 2`). For `i32/u32/f16/f32` inputs, `U = T` and `M = N`.
  If all predicate bits are zero, the result is zero.
- **VMI compact use:** `!pto.vreg<128xui16>` reduces to
  `!pto.vreg<64xui32>` with one sum in lane zero. The compact lowering uses
  this widened type, combines partial sums before narrowing, and assembles
  each logical group's low bits separately. It does not require the
  eight-result normalization used for 16-bit integer `vcgadd`.

---

### `pto.vcmax`

- **syntax:** `%result = pto.vcmax %input, %mask : !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>`
- **A5 types:** i16/i32 (signed, unsigned, or signless), f16, f32
- **semantics:** Find max element with argmax. The lowest destination element
  stores the maximum value, the second-lowest destination element stores the
  index of the first maximum, and all remaining elements are zero-filled.

```c
T mx = -INF; int idx = 0;
for (int i = 0; i < N; i++)
    if (src[i] > mx) { mx = src[i]; idx = i; }
dst[0] = mx;
dst[1] = idx;
for (int i = 2; i < N; i++)
    dst[i] = 0;
```

- **inputs:** `%input` is the source vector and `%mask` selects participating
  lanes.
- **outputs:** `%result[0]` holds the extremum value and `%result[1]` holds the
  index. Other destination elements are zero-filled.
- **constraints and limitations:** If there are multiple maxima, the minimum
  index is written. For floating-point types, inactive lanes are treated as
  `-INF`; if all lanes are inactive, `%result[0]` becomes `-INF`. For integer
  types, inactive lanes are treated as the literal minimum value; if all lanes
  are inactive, `%result[0]` becomes that literal minimum value. The index is
  written into the second destination element slot of the same destination
  vector register.

---

### `pto.vcmin`

- **syntax:** `%result = pto.vcmin %input, %mask : !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>`
- **A5 types:** i16/i32 (signed, unsigned, or signless), f16, f32
- **semantics:** Find min element with argmin. The lowest destination element
  stores the minimum value, the second-lowest destination element stores the
  index of the first minimum, and all remaining elements are zero-filled.

```c
T mn = INF; int idx = 0;
for (int i = 0; i < N; i++)
    if (src[i] < mn) { mn = src[i]; idx = i; }
dst[0] = mn;
dst[1] = idx;
for (int i = 2; i < N; i++)
    dst[i] = 0;
```

- **inputs:** `%input` is the source vector and `%mask` selects participating
  lanes.
- **outputs:** `%result[0]` holds the extremum value and `%result[1]` holds the
  index. Other destination elements are zero-filled.
- **constraints and limitations:** If there are multiple minima, the minimum
  index is written. For floating-point types, inactive lanes are treated as
  `+INF`; if all lanes are inactive, `%result[0]` becomes `+INF`. For integer
  types, inactive lanes are treated as the literal maximum value; if all lanes
  are inactive, `%result[0]` becomes that literal maximum value. The index is
  written into the second destination element slot of the same destination
  vector register.

---

### `pto.vcbmax`

- **syntax:** `%value, %predicate = pto.vcbmax %input, %mask : !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>, !pto.mask<G>`
- **A5 types:** i8-i32, f16, f32
- **semantics:** Find the maximum value and produce a predicate marking every
  participating lane whose value matches that maximum.

```c
T mx = max_active(src, mask);
for (int i = 0; i < N; i++) {
    value[i] = (i == 0) ? mx : 0;
    predicate[i] = mask[i] && matches_max(src[i], mx);
}
```

- **inputs:** `%input` is the source vector and `%mask` selects participating
  lanes.
- **outputs:** `%value[0]` holds the maximum and remaining value elements are
  zero-filled. `%predicate` marks all active lanes matching the maximum.
- **constraints and limitations:** If all lanes are inactive, `%predicate` is
  all zero. Floating-point inactive lanes are treated as `-INF`; integer
  inactive lanes are treated as the literal minimum value. For floating-point
  `+0/-0`, the value result follows the target maximum rule while predicate
  matching marks both zero signs as matching locations.

---

### `pto.vcbmin`

- **syntax:** `%value, %predicate = pto.vcbmin %input, %mask : !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>, !pto.mask<G>`
- **A5 types:** i8-i32, f16, f32
- **semantics:** Find the minimum value and produce a predicate marking every
  participating lane whose value matches that minimum.

```c
T mn = min_active(src, mask);
for (int i = 0; i < N; i++) {
    value[i] = (i == 0) ? mn : 0;
    predicate[i] = mask[i] && matches_min(src[i], mn);
}
```

- **inputs:** `%input` is the source vector and `%mask` selects participating
  lanes.
- **outputs:** `%value[0]` holds the minimum and remaining value elements are
  zero-filled. `%predicate` marks all active lanes matching the minimum.
- **constraints and limitations:** If all lanes are inactive, `%predicate` is
  all zero. Floating-point inactive lanes are treated as `+INF`; integer
  inactive lanes are treated as the literal maximum value. Floating-point NaN
  handling follows the target instruction semantics.

---

## Histogram Reductions

### `pto.chistv2`

- **syntax:** `%result = pto.chistv2 %acc, %source, %mask, %bin : !pto.vreg<128xui16>, !pto.vreg<256xui8>, !pto.mask<b8>, i32 -> !pto.vreg<128xui16>`
- **semantics:** Cumulative histogram update over unsigned 8-bit source lanes.
  `%acc` provides the incoming 16-bit bin accumulators and `%result` contains
  the updated accumulators.
- **inputs:** `%source` provides 256 unsigned 8-bit samples, `%mask` selects
  active source lanes, and `%bin` is the target bin/control operand passed to
  the A5 histogram instruction.
- **constraints and limitations:** `%acc` and `%result` are fixed to
  `!pto.vreg<128xui16>`, `%source` is fixed to `!pto.vreg<256xui8>`, and the
  mask granularity is fixed to `b8`.

---

### `pto.dhistv2`

- **syntax:** `%result = pto.dhistv2 %acc, %source, %mask, %bin : !pto.vreg<128xui16>, !pto.vreg<256xui8>, !pto.mask<b8>, i32 -> !pto.vreg<128xui16>`
- **semantics:** Distribution histogram update over unsigned 8-bit source
  lanes. `%acc` provides the incoming 16-bit bin accumulators and `%result`
  contains the updated accumulators.
- **inputs:** `%source` provides 256 unsigned 8-bit samples, `%mask` selects
  active source lanes, and `%bin` is the target bin/control operand passed to
  the A5 histogram instruction.
- **constraints and limitations:** `%acc` and `%result` are fixed to
  `!pto.vreg<128xui16>`, `%source` is fixed to `!pto.vreg<256xui8>`, and the
  mask granularity is fixed to `b8`.

---

## Per-VLane (Group) Reductions

The vector register is organized as **8 VLanes** of 32 bytes each. Group
reductions operate within each VLane independently and produce one result per
VLane. The eight results occupy the low portion of the destination register;
the remaining bits are zero. Their element width depends on the operation:
16-bit integer `vcgadd` writes 32-bit sums, whereas `vcgmax` and `vcgmin`
preserve the input element width.

```
vreg layout (f32 example, 64 elements total):
VLane 0: [0..7]   VLane 1: [8..15]  VLane 2: [16..23] VLane 3: [24..31]
VLane 4: [32..39] VLane 5: [40..47] VLane 6: [48..55] VLane 7: [56..63]
```

### `pto.vcgadd`

- **syntax:** `%result = pto.vcgadd %input, %mask : !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>`
- **A5 types:** i16/i32 (signed, unsigned, or signless), f16, f32
- **semantics:** Sum active elements within each 32-byte VLane. The 8 VLane
  sums are consecutive in a view of the hardware sum type `U`: 32-bit integers
  with the input signedness for 16-bit integer `T`, otherwise `U = T`. The
  current VPTO signature retains the input-shaped register type, so this view
  must be distinguished from indexing the declared `T` elements directly.

```c
int groups = 8;
int K = 32 / sizeof(T);  // elements per 32-byte VLane
int M = 256 / sizeof(U);
// sum_view aliases the destination register as M elements of type U.
for (int g = 0; g < 8; g++) {
    U sum = 0;
    for (int i = 0; i < K; i++)
        if (mask[g*K + i])
            sum += (U)src[g*K + i];
    sum_view[g] = sum;
}
for (int i = groups; i < M; i++)
    sum_view[i] = 0;
```

- **inputs:** `%input` is the source vector and `%mask` selects participating
  lanes.
- **outputs:** `%result` contains one sum per 32-byte VLane group. For 16-bit
  integers, its first sixteen declared elements hold alternating low/high
  halves of eight 32-bit sums. A `vbitcast` to 32-bit integers exposes the sum
  view without moving bits. The VMI lowering then uses unsigned `vpack LOWER`
  to select each sum's low 16 bits and restores the logical signedness,
  producing consecutive group slots with modulo, not saturating, narrowing.
- **constraints and limitations:** This is a per-32-byte VLane-group reduction.
  Inactive lanes are treated as zero. If all lanes in a VLane are inactive, the
  corresponding result element is `0` (`+0` for floating-point types).

---

### `pto.vcgmax`

- **syntax:** `%result = pto.vcgmax %input, %mask : !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>`
- **A5 types:** i16/i32 (signed, unsigned, or signless), f16, f32
- **semantics:** Find the maximum active element within each 32-byte VLane. The
  8 VLane maxima are written to result elements `0..7`; all other result
  elements are zero.

```c
int groups = 8;
int K = 32 / sizeof(T);
for (int g = 0; g < 8; g++) {
    T mx = max_identity_for_T;  // -INF for float, minimum value for integer
    for (int i = 0; i < K; i++)
        if (mask[g*K + i])
            mx = max(mx, src[g*K + i]);
    dst[g] = mx;
}
for (int i = groups; i < N; i++)
    dst[i] = 0;
```

- **inputs:** `%input` is the source vector and `%mask` selects participating
  lanes.
- **outputs:** `%result` contains one maximum per 32-byte VLane group, written
  contiguously to the low elements of the result vector. In particular,
  16-bit integer extrema remain 16-bit values; the `vcgadd` sum-narrowing pack
  does not apply. Unlike row `vcmax`, this operation does not return indices.
- **constraints and limitations:** Grouping is by hardware 32-byte VLane, not by
  arbitrary software subvector. Inactive floating-point lanes are treated as
  `-INF`; inactive integer lanes are treated as the element type's minimum
  value. If all lanes in a VLane are inactive, that neutral value is written for
  the corresponding VLane result. For floating-point values, `max(+0, -0)`
  returns `+0`.

---

### `pto.vcgmin`

- **syntax:** `%result = pto.vcgmin %input, %mask : !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>`
- **A5 types:** i16/i32 (signed, unsigned, or signless), f16, f32
- **semantics:** Find the minimum active element within each 32-byte VLane. The
  8 VLane minima are written to result elements `0..7`; all other result
  elements are zero.

```c
int groups = 8;
int K = 32 / sizeof(T);
for (int g = 0; g < 8; g++) {
    T mn = min_identity_for_T;  // +INF for float, maximum value for integer
    for (int i = 0; i < K; i++)
        if (mask[g*K + i])
            mn = min(mn, src[g*K + i]);
    dst[g] = mn;
}
for (int i = groups; i < N; i++)
    dst[i] = 0;
```

- **inputs:** `%input` is the source vector and `%mask` selects participating
  lanes.
- **outputs:** `%result` contains one minimum per 32-byte VLane group, written
  contiguously to the low elements of the result vector. In particular,
  16-bit integer extrema remain 16-bit values; the `vcgadd` sum-narrowing pack
  does not apply. Unlike row `vcmin`, this operation does not return indices.
- **constraints and limitations:** Grouping is by hardware 32-byte VLane, not by
  arbitrary software subvector. Inactive floating-point lanes are treated as
  `+INF`; inactive integer lanes are treated as the element type's maximum
  value. If all lanes in a VLane are inactive, that neutral value is written for
  the corresponding VLane result. For floating-point values, `min(-0, +0)`
  returns `-0`.

---

## Prefix Operations

### `pto.vcpadd`

- **syntax:** `%result = pto.vcpadd %input, %mask : !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>`
- **A5 types:** f16, f32
- **semantics:** Inclusive prefix sum (scan).

```c
dst[0] = src[0];
for (int i = 1; i < N; i++)
    dst[i] = dst[i-1] + src[i];
```

**Example:**
```c
// input:  [1, 2, 3, 4, 5, ...]
// output: [1, 3, 6, 10, 15, ...]
```

- **inputs:** `%input` is the source vector and `%mask` selects participating
  lanes.
- **outputs:** `%result` is the inclusive prefix-sum vector.
- **constraints and limitations:** Only floating-point element types are
  documented on the current A5 surface here.

---

## Typical Usage

```mlir
// Softmax: find max for numerical stability
%max_vec = pto.vcmax %logits, %mask : !pto.vreg<64xf32>, !pto.mask<G> -> !pto.vreg<64xf32>
// max is in lane 0, broadcast it
%max_broadcast = pto.vlds %ub_tmp[%c0] {dist = "BRC_B32"} : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>

// Per-VLane sums using vcgadd
%row_sums = pto.vcgadd %tile, %mask : !pto.vreg<64xf32>, !pto.mask<G> -> !pto.vreg<64xf32>
// Results at indices 0..7; remaining elements are zero

// Full vector sum for normalization
%total = pto.vcadd %values, %mask : !pto.vreg<64xf32>, !pto.mask<G> -> !pto.vreg<64xf32>
// total[0] contains the sum

// Prefix sum for cumulative distribution
%cdf = pto.vcpadd %pdf, %mask : !pto.vreg<64xf32>, !pto.mask<G> -> !pto.vreg<64xf32>
```
