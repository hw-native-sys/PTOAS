# 11. Compare & Select

> **Category:** Comparison and conditional selection operations
> **Pipeline:** PIPE_V (Vector Core)

Operations that compare vectors and conditionally select elements.

## Common Operand Model

- `%src0` and `%src1` are source vector operands.
- `%scalar` is the scalar operand for scalar-comparison families.
- `%seed` is the incoming predicate that limits which lanes participate in the
  compare.
- `%result` is either a predicate mask (`vcmp`, `vcmps`) or a vector register
  (`vsel`, `vselr`, `vselrv2`).

---

## Comparison Operations

### `pto.vcmp`

- **syntax:** `%result = pto.vcmp %src0, %src1, %seed, "CMP_MODE" : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask<G> -> !pto.mask<G>`
- **semantics:** Element-wise comparison, output predicate mask.

```c
for (int i = 0; i < N; i++)
    if (seed[i])
        dst[i] = (src0[i] CMP src1[i]) ? 1 : 0;
```

**Compare modes:**

| Mode | Operation |
|------|-----------|
| `eq` | Equal (==) |
| `ne` | Not equal (!=) |
| `lt` | Less than (<) |
| `le` | Less than or equal (<=) |
| `gt` | Greater than (>) |
| `ge` | Greater than or equal (>=) |

**Example:**
```mlir
%all_active = pto.pset_b32 "PAT_ALL" : !pto.mask<b32>
%lt_mask = pto.vcmp %a, %b, %all_active, "lt" : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.mask<b32>
// lt_mask[i] = 1 if a[i] < b[i]
```

- **inputs:** `%src0`, `%src1`, and `%seed`; `CMP_MODE` selects the comparison
  predicate.
- **outputs:** `%result` is the generated predicate mask.
- **constraints and limitations:** Only lanes enabled by `%seed` participate.
  Integer and floating-point comparisons follow their own element-type-specific
  comparison rules. `%seed` and `%result` keep the typed-mask granularity that
  matches `%src0` / `%src1`.

---

### `pto.vcmps`

- **syntax:** `%result = pto.vcmps %src, %scalar, %seed, "CMP_MODE" : !pto.vreg<NxT>, T, !pto.mask<G> -> !pto.mask<G>`
- **semantics:** Compare vector against scalar.

```c
for (int i = 0; i < N; i++)
    if (seed[i])
        dst[i] = (src[i] CMP scalar) ? 1 : 0;
```

**Example:**
```mlir
%positive_mask = pto.vcmps %values, %c0_f32, %all_active, "gt"
    : !pto.vreg<64xf32>, f32, !pto.mask<b32> -> !pto.mask<b32>
// positive_mask[i] = 1 if values[i] > 0
```

- **inputs:** `%src` is the vector source, `%scalar` is the scalar comparison
  value, and `%seed` is the incoming predicate.
- **outputs:** `%result` is the generated predicate mask.
- **constraints and limitations:** For 32-bit scalar forms, the scalar source
  MUST satisfy the backend's legal scalar-source constraints for this family.
  `%seed` and `%result` keep the typed-mask granularity that matches `%src`.

---

## Selection Operations

### `pto.vsel`

- **syntax:** `%result = pto.vsel %src0, %src1, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask<G> -> !pto.vreg<NxT>`
- **semantics:** Per-lane select based on mask.

```c
for (int i = 0; i < N; i++)
    dst[i] = mask[i] ? src0[i] : src1[i];
```

**Example — Conditional assignment:**
```mlir
// dst = mask ? true_vals : false_vals
%result = pto.vsel %true_vals, %false_vals, %condition
    : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
```

- **inputs:** `%src0` is the true-path vector, `%src1` is the false-path vector,
  and `%mask` selects between them.
- **outputs:** `%result` is the selected vector.
- **constraints and limitations:** Source vectors and result MUST have matching
  vector shapes and element types. `%mask` keeps the typed-mask granularity
  that matches the selected vector family.

---

### `pto.vselr`

- **syntax:** `%result = pto.vselr %src, %idx : !pto.vreg<NxT>, !pto.vreg<Nxi<width>> -> !pto.vreg<NxT>`
- **semantics:** Lane-select by index vector.

```c
for (int i = 0; i < N; i++)
    dst[i] = src[idx[i]];
```

- **inputs:** `%src` is the source vector. `%idx` is the lane-index vector.
- **outputs:** `%result` is the reordered vector.
- **constraints and limitations:** `%idx` must use integer elements. `%idx`
  must have the same lane count as `%src`, and its integer element width must
  match the bit width of `%src` element type.

---

### `pto.vselrv2`

- **syntax:** `%result = pto.vselrv2 %src0, %src1 : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<NxT>`
- **semantics:** Variant select form with the same current two-vector operand shape.
- **inputs:** `%src0` and `%src1` are the source vectors.
- **outputs:** `%result` is the selected vector.
- **constraints and limitations:** This page records the surface shape only.
  Lowering MUST preserve the exact A5 variant semantics selected for this form.

---

## Typical Usage

```mlir
// Clamp negative values to zero (manual ReLU)
%all = pto.pset_b32 "PAT_ALL" : !pto.mask<b32>
%zero = pto.vbr %c0_f32 : f32 -> !pto.vreg<64xf32>
%neg_mask = pto.vcmps %input, %c0_f32, %all, "lt" : !pto.vreg<64xf32>, f32, !pto.mask<b32> -> !pto.mask<b32>
%clamped = pto.vsel %zero, %input, %neg_mask : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>

// Element-wise max via compare+select
%gt_mask = pto.vcmp %a, %b, %all, "gt" : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.mask<b32>
%max_ab = pto.vsel %a, %b, %gt_mask : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>

// Threshold filter
%above_thresh = pto.vcmps %scores, %threshold, %all, "ge" : !pto.vreg<64xf32>, f32, !pto.mask<b32> -> !pto.mask<b32>
%filtered = pto.vsel %scores, %zero, %above_thresh : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
```

---

## Compare + Select Pattern

```mlir
// Softmax safe exp: exp(x - max) where x < max returns exp of negative
// but we want to clamp to avoid underflow

%all = pto.pset_b32 "PAT_ALL" : !pto.mask<b32>

// 1. Compare against threshold
%too_small = pto.vcmps %x_minus_max, %min_exp_arg, %all, "lt"
    : !pto.vreg<64xf32>, f32, !pto.mask<b32> -> !pto.mask<b32>

// 2. Clamp values below threshold
%clamped = pto.vsel %min_exp_arg_vec, %x_minus_max, %too_small
    : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>

// 3. Safe exp
%exp_result = pto.vexp %clamped, %all : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
```
