# 11. Compare & Select

> **Category:** Comparison and conditional selection operations
> **Pipeline:** PIPE_V (Vector Core)

Operations that compare vectors and conditionally select elements.

---

## Comparison Operations

### `pto.vcmp`

- **syntax:** `%result = pto.vcmp %src0, %src1, %seed, "CMP_MODE" : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask -> !pto.mask`
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
%all_active = pto.pset_b32 "PAT_ALL" : !pto.mask
%lt_mask = pto.vcmp %a, %b, %all_active, "lt" : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask -> !pto.mask
// lt_mask[i] = 1 if a[i] < b[i]
```

---

### `pto.vcmps`

- **syntax:** `%result = pto.vcmps %src, %scalar, %seed, "CMP_MODE" : !pto.vreg<NxT>, T, !pto.mask -> !pto.mask`
- **semantics:** Compare vector against scalar.

```c
for (int i = 0; i < N; i++)
    if (seed[i])
        dst[i] = (src[i] CMP scalar) ? 1 : 0;
```

**Example:**
```mlir
%positive_mask = pto.vcmps %values, %c0_f32, %all_active, "gt"
    : !pto.vreg<64xf32>, f32, !pto.mask -> !pto.mask
// positive_mask[i] = 1 if values[i] > 0
```

---

## Selection Operations

### `pto.vsel`

- **syntax:** `%result = pto.vsel %src0, %src1, %mask : !pto.vreg<NxT>, !pto.vreg<NxT>, !pto.mask -> !pto.vreg<NxT>`
- **semantics:** Per-lane select based on mask.

```c
for (int i = 0; i < N; i++)
    dst[i] = mask[i] ? src0[i] : src1[i];
```

**Example — Conditional assignment:**
```mlir
// dst = mask ? true_vals : false_vals
%result = pto.vsel %true_vals, %false_vals, %condition
    : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>
```

---

### `pto.vselr`

- **syntax:** `%result = pto.vselr %src0, %src1 : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<NxT>`
- **semantics:** Select with reversed mask semantics.

```c
for (int i = 0; i < N; i++)
    dst[i] = mask[i] ? src1[i] : src0[i];  // reversed from vsel
```

---

### `pto.vselrv2`

- **syntax:** `%result = pto.vselrv2 %src0, %src1 : !pto.vreg<NxT>, !pto.vreg<NxT> -> !pto.vreg<NxT>`
- **semantics:** Variant select form with the same current two-vector operand shape.

---

## Typical Usage

```mlir
// Clamp negative values to zero (manual ReLU)
%all = pto.pset_b32 "PAT_ALL" : !pto.mask
%zero = pto.vbr %c0_f32 : f32 -> !pto.vreg<64xf32>
%neg_mask = pto.vcmps %input, %c0_f32, %all, "lt" : !pto.vreg<64xf32>, f32, !pto.mask -> !pto.mask
%clamped = pto.vsel %zero, %input, %neg_mask : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>

// Element-wise max via compare+select
%gt_mask = pto.vcmp %a, %b, %all, "gt" : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask -> !pto.mask
%max_ab = pto.vsel %a, %b, %gt_mask : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>

// Threshold filter
%above_thresh = pto.vcmps %scores, %threshold, %all, "ge" : !pto.vreg<64xf32>, f32, !pto.mask -> !pto.mask
%filtered = pto.vsel %scores, %zero, %above_thresh : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>
```

---

## Compare + Select Pattern

```mlir
// Softmax safe exp: exp(x - max) where x < max returns exp of negative
// but we want to clamp to avoid underflow

%all = pto.pset_b32 "PAT_ALL" : !pto.mask

// 1. Compare against threshold
%too_small = pto.vcmps %x_minus_max, %min_exp_arg, %all, "lt"
    : !pto.vreg<64xf32>, f32, !pto.mask -> !pto.mask

// 2. Clamp values below threshold
%clamped = pto.vsel %min_exp_arg_vec, %x_minus_max, %too_small
    : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>

// 3. Safe exp
%exp_result = pto.vexp %clamped, %all : !pto.vreg<64xf32>, !pto.mask -> !pto.vreg<64xf32>
```
