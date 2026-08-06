# 16. Cube Matrix Multiply

> **Category:** Cube unit ops — staged load/store, matrix multiply, and
> FIXPIPE MTE writeback

This chapter documents the high-level Cube VPTO surface. It describes logical
data objects, operand units, layout contracts, numeric behavior, and writeback
effects from the user's point of view.

---

## Common Cube Operand Model

Cube ops use typed PTO pointers to name logical storage domains. The canonical
`!pto.ptr` address-space names are the hardware-domain names below. The legacy
names are accepted only as parser aliases and are printed back as canonical
names.

| Canonical address space | Legacy alias | Logical role |
|-------------------------|--------------|--------------|
| `gm` | - | Global memory |
| `l1` | `mat` | L1 matrix staging buffer |
| `l0a` | `left` | Left matrix operand tile for Cube compute |
| `l0b` | `right` | Right matrix operand tile for Cube compute |
| `l0c` | `acc` | Accumulator/result tile produced by Cube compute |
| `bt` | `bias` | Bias vector payload consumed by bias matmul forms |
| `fb` | `scaling` | FIXPIPE parameter payloads consumed by vector quant/ReLU clauses |
| `ub` | `vec` | Unified Buffer destination/source for vector-side use |

Unless an op says otherwise:

- Shape operands such as `%m`, `%n`, `%k`, `shape(%n, %d)` are logical element
  counts, not byte counts.
- Length operands named `%len_burst` in byte-copy surfaces are byte counts
  unless the op explicitly states a different unit.
- Strides named `src_stride` or `dst_stride` are start-to-start distances in
  the unit stated by the op. Do not infer byte units from the name alone.
- Pointer operands select the base address of the logical object. Sub-tile
  selection is expressed by computing a different base pointer before calling
  the op, unless the op exposes an explicit start or group operand.
- Cache/session hint operands may affect the memory path but do not change the
  mathematical value written or read.

---

## Cube Compute Ops

The `pto.mad*` family computes logical matrix multiplication over tiles already
prepared in `l0a` and `l0b`:

```text
lhs: M x K
rhs: K x N
dst: M x N
```

The matrix element types are inferred from `%lhs`, `%rhs`, and `%dst` pointer
element types. There is no separate type selector. Unsupported type
combinations are invalid programs.

The current VPTO surface enforces the Cube storage roles through pointer
address spaces: `%lhs` is `l0a`, `%rhs` is `l0b`, and `%dst` is `l0c`.
Bias forms additionally require `%bias` in the `bt` address space with the
same element type as `%dst`. MX forms require MX element types on both `%lhs`
and `%rhs`; the current target-profile MX data type is `f8E4M3FN`.

### MAD Common Clauses

| Clause | Values | Effect |
|--------|--------|--------|
| `unit_flag(...)` | `check_only`, `check_and_set` | Participates in producer-side tile synchronization. `check_only` checks that the producer slot can be used. `check_and_set` also publishes the produced `%dst` tile for later consumers. Omit the clause when the schedule does not use unit flags for this tile. |
| `disable_gemv` | flag | Applies only when `%m = 1`. Omitted means GEMV A-vector consumption: `%lhs` must contain the logical `1 x K` row in the target GEMV left-tile organization. Present means normal matmul left-tile organization. The mathematical result is still `lhs @ rhs`; only the required `%lhs` organization changes. For `%m != 1`, normal matmul organization is used. |
| `sat` / `nosat` | flags | Floating exceptional-value mode for floating and MX MAD forms. With `sat`, exceptional multiply inputs are normalized before arithmetic (`+/-inf` to finite type extrema, `nan` to 0) and finite overflow saturates to the finite type range. With `nosat`, exceptional inputs are preserved and overflow may produce exceptional outputs. Omit both to use the execution mode selected outside this op. Integer MAD forms do not accept these flags. |
| `tf32_mode(...)` | `round_even`, `round_away` | Valid only for non-MX `f32 x f32 -> f32`. FP32 inputs are rounded to TF32 precision before multiplication; accumulation and output remain FP32. |
| `n_dir` | flag | Requests N-direction result production order for schedules that combine compute with unit flags and later layout movement. It does not change `dst[m, n]`. |

Reference semantics for non-MX forms:

```text
product[m, n] = sum k in 0 .. K-1:
                  numeric_lhs(lhs[m, k]) * numeric_rhs(rhs[k, n])

pto.mad:      dst[m, n] = product[m, n]
pto.mad_acc:  dst[m, n] = dst[m, n] + product[m, n]
pto.mad_bias: dst[m, n] = product[m, n] + bias[n]
```

For integer forms, the op multiplies the typed values already present in
`l0a` and `l0b`. Per-input offset correction for quantized integer
algorithms is not an operand of `pto.mad*`; apply such correction before
loading the Cube operands when the algorithm needs it.

### MX Matmul Model

`pto.mad_mx*` additionally applies microscaling. The scale payloads are loaded
with `pto.mte_l1_l0a_mx` / `pto.mte_l1_l0b_mx` and are associated with the
selected `%lhs` / `%rhs` tiles; they are not direct operands of `pto.mad_mx*`.

The K dimension is partitioned into 32-element groups:

```text
k_group = floor(k / 32)

mx_product[m, n] =
  sum k in 0 .. K-1:
    (lhs[m, k] * lhs_scale[m, k_group]) *
    (rhs[k, n] * rhs_scale[k_group, n])
```

Current target-profile MX data tiles use `f8E4M3FN`. `%k` must be compatible
with MX grouping. On the current target profile, MX matmul consumes K in
64-element multiples, which contain two 32-element scale groups.

### `pto.mad`

- **syntax:**
```mlir
pto.mad %lhs, %rhs, %dst, %m, %n, %k
  unit_flag(check_only | check_and_set)?
  disable_gemv?
  (sat | nosat)?
  tf32_mode(round_even | round_away)?
  n_dir?
  : !pto.ptr<A, l0a>, !pto.ptr<B, l0b>, !pto.ptr<C, l0c>, i64, i64, i64
```
- **semantics:** Zero-init matrix multiply, `dst[m, n] = sum_k(lhs[m, k] * rhs[k, n])`.

**Parameter Table:**

| Parameter | Width | Description |
|-----------|-------|-------------|
| `%lhs` | ptr | Left operand tile in `l0a`, interpreted as logical `M x K` |
| `%rhs` | ptr | Right operand tile in `l0b`, interpreted as logical `K x N` |
| `%dst` | ptr | Accumulator destination tile in `l0c`, interpreted as logical `M x N` |
| `%m` | i64 | Logical M element count |
| `%n` | i64 | Logical N element count |
| `%k` | i64 | Logical K element count |
| optional clauses | - | See [MAD Common Clauses](#mad-common-clauses) |

**Constraints:**

- `%lhs`, `%rhs`, and `%dst` must be in `l0a`, `l0b`, and `l0c`.
- `%m`, `%n`, and `%k` must be positive and satisfy the target shape limits
  for the selected element-type combination.
- `tf32_mode(...)` requires `f32` lhs, rhs, and dst element types.
- `sat` / `nosat` requires a floating element-type combination.
- Packed 4-bit integer data requires `%k` to select an even number of K
  elements.

**Example:**

```mlir
pto.mad %l0a, %l0b, %l0c, %c16_i64, %c16_i64, %c32_i64
  : !pto.ptr<f16, l0a>, !pto.ptr<f16, l0b>, !pto.ptr<f32, l0c>, i64, i64, i64
```

---

### `pto.mad_acc`

- **syntax:**
```mlir
pto.mad_acc %lhs, %rhs, %dst, %m, %n, %k
  unit_flag(check_only | check_and_set)?
  disable_gemv?
  (sat | nosat)?
  tf32_mode(round_even | round_away)?
  n_dir?
  : !pto.ptr<A, l0a>, !pto.ptr<B, l0b>, !pto.ptr<C, l0c>, i64, i64, i64
```
- **semantics:** Accumulating matrix multiply,
  `dst[m, n] = dst[m, n] + sum_k(lhs[m, k] * rhs[k, n])`.

**Parameter Table:** same as `pto.mad`.

**Constraints:** same as `pto.mad`.

**Example:**

```mlir
pto.mad_acc %l0a, %l0b, %l0c, %c16_i64, %c16_i64, %c32_i64 unit_flag(check_only)
  : !pto.ptr<f16, l0a>, !pto.ptr<f16, l0b>, !pto.ptr<f32, l0c>, i64, i64, i64
```

---

### `pto.mad_bias`

- **syntax:**
```mlir
pto.mad_bias %lhs, %rhs, %dst, %bias, %m, %n, %k
  unit_flag(check_only | check_and_set)?
  disable_gemv?
  (sat | nosat)?
  tf32_mode(round_even | round_away)?
  n_dir?
  : !pto.ptr<A, l0a>, !pto.ptr<B, l0b>, !pto.ptr<C, l0c>, !pto.ptr<C, bt>, i64, i64, i64
```
- **semantics:** Bias-init matrix multiply,
  `dst[m, n] = sum_k(lhs[m, k] * rhs[k, n]) + bias[n]`.

**Parameter Table:**

| Parameter | Width | Description |
|-----------|-------|-------------|
| `%lhs`, `%rhs`, `%dst`, `%m`, `%n`, `%k` | - | Same as `pto.mad` |
| `%bias` | ptr | Bias vector in `bt`, interpreted as `N` values and broadcast across M |
| optional clauses | - | See [MAD Common Clauses](#mad-common-clauses) |

**Constraints:**

- `%bias` must be in `bt` address space.
- `%bias` element type must match `%dst` element type.
- Only `N` bias values are consumed; `%bias` is not an `M x N` matrix.
- Other constraints match `pto.mad`.

**Example:**

```mlir
pto.mad_bias %l0a, %l0b, %l0c, %bt, %c16_i64, %c16_i64, %c32_i64
  : !pto.ptr<f16, l0a>, !pto.ptr<f16, l0b>, !pto.ptr<f32, l0c>, !pto.ptr<f32, bt>, i64, i64, i64
```

---

### `pto.mad_mx`

- **syntax:**
```mlir
pto.mad_mx %lhs, %rhs, %dst, %m, %n, %k
  unit_flag(check_only | check_and_set)?
  disable_gemv?
  (sat | nosat)?
  n_dir?
  : !pto.ptr<A, l0a>, !pto.ptr<B, l0b>, !pto.ptr<C, l0c>, i64, i64, i64
```
- **semantics:** Zero-init MX matrix multiply, `dst[m, n] = mx_product[m, n]`.

**Parameter Table:** same as `pto.mad`; `%lhs` and `%rhs` must have matching
MX scale payloads prepared by the MX load ops.

**Constraints:**

- Operands must use a target-supported MX dtype combination.
- Matching left and right MX scale payloads must be loaded before this op.
- `%k` must satisfy the MX grouping rule described in [MX Matmul Model](#mx-matmul-model).
- `tf32_mode(...)` is not a clause of MX MAD.

**Example:**

```mlir
pto.mad_mx %l0a, %l0b, %l0c, %c16_i64, %c16_i64, %c64_i64
  : !pto.ptr<f8E4M3FN, l0a>, !pto.ptr<f8E4M3FN, l0b>, !pto.ptr<f32, l0c>, i64, i64, i64
```

---

### `pto.mad_mx_acc`

- **syntax:**
```mlir
pto.mad_mx_acc %lhs, %rhs, %dst, %m, %n, %k
  unit_flag(check_only | check_and_set)?
  disable_gemv?
  (sat | nosat)?
  n_dir?
  : !pto.ptr<A, l0a>, !pto.ptr<B, l0b>, !pto.ptr<C, l0c>, i64, i64, i64
```
- **semantics:** Accumulating MX matrix multiply,
  `dst[m, n] = dst[m, n] + mx_product[m, n]`.

**Parameter Table:** same as `pto.mad_mx`.

**Constraints:** same as `pto.mad_mx`.

**Example:**

```mlir
pto.mad_mx_acc %l0a, %l0b, %l0c, %c16_i64, %c16_i64, %c64_i64
  : !pto.ptr<f8E4M3FN, l0a>, !pto.ptr<f8E4M3FN, l0b>, !pto.ptr<f32, l0c>, i64, i64, i64
```

---

### `pto.mad_mx_bias`

- **syntax:**
```mlir
pto.mad_mx_bias %lhs, %rhs, %dst, %bias, %m, %n, %k
  unit_flag(check_only | check_and_set)?
  disable_gemv?
  (sat | nosat)?
  n_dir?
  : !pto.ptr<A, l0a>, !pto.ptr<B, l0b>, !pto.ptr<C, l0c>, !pto.ptr<C, bt>, i64, i64, i64
```
- **semantics:** Bias-init MX matrix multiply,
  `dst[m, n] = mx_product[m, n] + bias[n]`.

**Parameter Table:** same as `pto.mad_bias`, with MX `%lhs` / `%rhs` scale
payload requirements from `pto.mad_mx`.

**Constraints:** same as `pto.mad_mx` plus `pto.mad_bias` bias constraints.

**Example:**

```mlir
pto.mad_mx_bias %l0a, %l0b, %l0c, %bt, %c16_i64, %c16_i64, %c64_i64
  : !pto.ptr<f8E4M3FN, l0a>, !pto.ptr<f8E4M3FN, l0b>, !pto.ptr<f32, l0c>, !pto.ptr<f32, bt>, i64, i64, i64
```

---

## Cube Data Movement Ops

### Cube Burst / Loop Addressing Model

`pto.mte_gm_l1` and `pto.mte_l1_ub` use the same grouped transfer model:

```text
burst(row) = len_burst contiguous bytes
nburst     = innermost repeated burst group
loop       = optional outer repetition group
```

For each `nburst` row, the source and destination start addresses advance by
`src_stride` and `dst_stride` after a burst row. Optional `loop(...)` groups
wrap the full inner transfer pattern and advance by their own source and
destination strides between repetitions. All lengths and strides in this model
are bytes.

### `pto.mte_gm_l1`

- **syntax:**
```mlir
pto.mte_gm_l1 %src, %dst, %len_burst
  nburst(%count, %src_stride, %dst_stride)
  [loop(%count_i, %src_stride_i, %dst_stride_i)]*
  : !pto.ptr<T, gm>, !pto.ptr<T, l1>, i64, i64, i64, i64
```
- **semantics:** Structured GM-to-L1 copy. The op copies grouped byte ranges
  from `%src` in `gm` to `%dst` in `l1`.

**Parameter Table:**

| Parameter | Width | Description |
|-----------|-------|-------------|
| `%src` | ptr | GM source base pointer |
| `%dst` | ptr | L1 matrix destination base pointer in `l1` |
| `%len_burst` | i64 | Bytes copied per burst row |
| `nburst(%count, %src_stride, %dst_stride)` | i64 triple | Innermost burst count and byte strides between row starts |
| `loop(%count_i, %src_stride_i, %dst_stride_i)` | i64 triple | Optional outer repetition; strides are byte advances between enclosed patterns |

**Constraints:**

- `nburst(...)` is required.
- Each `loop(...)` group must provide all three operands.
- For a contiguous 16-element f16 vector, use `%len_burst = 32`.

**Example:**

```mlir
pto.mte_gm_l1 %bias_gm, %l1_bias, %c32_i64
  nburst(%c4_i64, %c64_i64, %c32_i64)
  : !pto.ptr<f16, gm>, !pto.ptr<f16, l1>, i64, i64, i64, i64
```

---

### `pto.mte_l1_ub`

- **syntax:**
```mlir
pto.mte_l1_ub %src, %dst, %len_burst
  nburst(%count, %src_stride, %dst_stride)
  [loop(%count_i, %src_stride_i, %dst_stride_i)]*
  : !pto.ptr<T, l1>, !pto.ptr<T, ub>, i64, i64, i64, i64
```
- **semantics:** Structured L1-to-UB copy. The grouped byte ranges are read
  from `%src` in `l1` and written to `%dst` in `ub`.

**Parameter Table:** same grouped byte model as `pto.mte_gm_l1`, with source
and destination address spaces reversed to `l1 -> ub`.

**Constraints:**

- `%src` must be in `l1`, `%dst` must be in `ub`.
- `nburst(...)` is required.
- Each `loop(...)` group must provide all three operands.

**Example:**

```mlir
pto.mte_l1_ub %l1_src, %ub_dst, %c64_i64
  nburst(%c2_i64, %c128_i64, %c64_i64)
  : !pto.ptr<f16, l1>, !pto.ptr<f16, ub>, i64, i64, i64, i64
```

---

### `pto.mte_gm_l1_frac`

- **syntax:**
```mlir
pto.mte_gm_l1_frac %src, %dst, nd2nz|dn2nz,
  shape(%n_value, %d_value),
  src_layout(%src_inner_stride[, %src_outer_stride]),
  dst_group(%group_count, %dst_loop2_stride, %dst_loop3_stride, %dst_loop4_stride),
  ctrl(%l2_cache_ctrl, %smallc0_en)
  : !pto.ptr<T, gm>, !pto.ptr<T, l1>, ...
```
- **semantics:** Load a logical 2-D GM region and write one or more L1 NZ
  matrix groups. `nd2nz` reads a logical `src[n, d]` matrix. `dn2nz` reads a
  logical `src[d, n]` matrix and writes the same logical `N x D` result into
  NZ layout.

**Parameter Table:**

| Parameter | Width | Description |
|-----------|-------|-------------|
| `%src` | ptr | GM source base pointer |
| `%dst` | ptr | L1 NZ destination base pointer in `l1` |
| `nd2nz` / `dn2nz` | keyword | Source logical layout mode |
| `shape(%n_value, %d_value)` | i64 pair | Logical output shape before NZ packing |
| `src_layout(%src_inner_stride[, %src_outer_stride])` | i64 / optional i64 | Source row/matrix byte strides |
| `dst_group(...)` | i64 tuple | Destination group count and placement strides in C0-size units |
| `ctrl(%l2_cache_ctrl, %smallc0_en)` | i64, i1 | Cache hint and small-C0 packing enable |

`src_layout(%src_inner_stride)` describes one logical source matrix. For
`nd2nz`, `%src_inner_stride` is the byte distance from `src[n, 0]` to
`src[n + 1, 0]`. For `dn2nz`, it is the byte distance from `src[d, 0]` to
`src[d + 1, 0]`. When `%src_outer_stride` is present, it is the byte distance
between adjacent source matrices. When omitted, the outer source stride is 0.

`dst_group(%group_count, %dst_loop2_stride, %dst_loop3_stride,
%dst_loop4_stride)` writes `%group_count` logical matrices. Destination strides
are measured in C0-size units; one C0-size unit is 32 bytes. These strides
place generated NZ blocks relative to `%dst`. They do not select a separate
memory block.

Reference addressing:

```text
for g in 0 .. group_count-1:
  src_g = src + g * src_outer_stride
  dst_g = dst + g * dst_loop4_stride * 32

  for n in 0 .. n_value-1:
    for d in 0 .. d_value-1:
      if mode == nd2nz:
        value = load(src_g + n * src_inner_stride + d * sizeof(T))
      else:
        value = load(src_g + d * src_inner_stride + n * sizeof(T))
      store value into NZ position for logical [n, d] under dst_g

  invalid lanes in the final C0 group are written as zero
```

**Constraints:**

- Source strides are bytes. For row-major `16 x 16` f16 input,
  `src_layout(32)` describes consecutive rows.
- Destination strides are C0-size units, not bytes and not elements.
- `smallc0_en = true` is valid only for target-supported small-C0 cases. The
  current contract rejects `d_value > 4` in small-C0 mode.
- In normal C0 mode, each destination C0 burst is padded to 32 bytes. In
  small-C0 mode, each destination burst is padded to 4 logical channels, and
  the generated inner-N and C0 destination placement is fixed by that
  small-C0 packing rule. `%dst_loop4_stride` still places adjacent matrix
  groups.
- In small-C0 mode, missing logical `N` rows and invalid `D` lanes are written
  as zero, and the tail of a generated NZ matrix is padded to the 32-byte C0
  boundary.
- Destination regions selected by `%dst` and `dst_group(...)` must not overlap.
  If two generated writes target the same bytes, the final value is not a
  stable program result.

**Example:**

```mlir
pto.mte_gm_l1_frac %src, %dst, nd2nz,
  shape(%c32_i64, %c16_i64),
  src_layout(%c32_i64, %c1024_i64),
  dst_group(%c2_i64, %c1_i64, %c16_i64, %c64_i64),
  ctrl(%c0_i64, %false)
  : !pto.ptr<f16, gm>, !pto.ptr<f16, l1>, nd2nz, shape i64, i64,
    src_layout(i64, i64), dst_group i64, i64, i64, i64, ctrl i64, i1
```

---

### `pto.mte_l1_bt`

- **syntax:**
```mlir
pto.mte_l1_bt %src, %dst, %len_burst
  nburst(%count, %src_gap, %dst_gap)
  : !pto.ptr<T, l1>, !pto.ptr<U, bt>, i64, i64, i64, i64
```
- **semantics:** Load an L1 bias payload into the `bt` address space for
  later `pto.mad_bias` / `pto.mad_mx_bias` consumption. The consumer interprets
  the result as an `N`-element bias vector `bias[n]`.

**Parameter Table:**

| Parameter | Width | Description |
|-----------|-------|-------------|
| `%src` | ptr | L1 source pointer in `l1` |
| `%dst` | ptr | Bias destination pointer in `bt` |
| `%len_burst` | i64 | Number of bias-load units per burst |
| `%count` | i64 | Burst count |
| `%src_gap` | i64 | Source gap between bursts, in bias-load units |
| `%dst_gap` | i64 | Destination gap between bursts, in bias-load units |

One burst loads `%len_burst` units from `%src` and writes the corresponding
bias values to `%dst`. After each burst except the last, source and destination
advance by the burst length plus the corresponding gap.

**Constraints:**

- Supported type pairs: `f32->f32`, `i32->i32`, `f16->f32`, `bf16->f32`.
- For `bf16->f32`, compact bf16 source values are always widened to f32 bias
  values. For `f16->f32`, compact f16 source values are widened when the load
  is used as an f32 bias payload; otherwise the f16 payload is stored in the
  32-bit bias slot with unused high bits.
- Load exactly the channel bias values needed by the consumer tile; the bias
  payload is not result-shaped.

**Example:**

```mlir
pto.mte_l1_bt %l1_bias, %bt, %c1_i64 nburst(%c4_i64, %c0_i64, %c0_i64)
  : !pto.ptr<f16, l1>, !pto.ptr<f32, bt>, i64, i64, i64, i64
```

---

### `pto.mte_l1_fb`

- **syntax:**
```mlir
pto.mte_l1_fb %src, %dst, %len_burst
  nburst(%count, %src_gap, %dst_gap)
  : !pto.ptr<T, l1>, !pto.ptr<U, fb>, i64, i64, i64, i64
```
- **semantics:** Load FIXPIPE parameter payloads from L1 into `fb`.
  Vector `pre_quant(...)` and `pre_relu(...)` clauses in `pto.mte_l0c_l1*`
  later consume these payloads through `fb` pointers.

**Parameter Table:**

| Parameter | Width | Description |
|-----------|-------|-------------|
| `%src` | ptr | L1 source pointer in `l1` |
| `%dst` | ptr | Scaling destination pointer in `fb` |
| `%len_burst` | i64 | Number of parameter-load units per burst |
| `%count` | i64 | Burst count |
| `%src_gap` | i64 | Source gap between bursts, in parameter-load units |
| `%dst_gap` | i64 | Destination gap between bursts, in parameter-load units |

The copy unit of `pto.mte_l1_fb` is the parameter-load unit of this op. It is
separate from the row size consumed by `mte_l0c_*` vector payloads.
`%len_burst` and the `nburst(...)` gaps are counted in these load units, not
in bytes and not in destination elements. After `pto.mte_l1_fb` materializes the
payload in `fb`, vector pre-ReLU consumers read it as 64B parameter rows
and vector pre-quant consumers read it as 128B parameter rows. The payload
pointer passed to `mte_l0c_*` must point at the first row for the logical
output tile, and rows must follow the same channel/NZ order consumed by that
store.

**Constraints:**

- `%src` must be in `l1`, `%dst` must be in `fb`.
- Vector `pre_quant` and `pre_relu` consumers require parameter data prepared
  in the row order documented by [FIXPIPE MTE Ops](#fixpipe-mte-ops).

**Example:**

```mlir
pto.mte_l1_fb %l1_fp, %fb_fp, %c2_i64 nburst(%c4_i64, %c0_i64, %c0_i64)
  : !pto.ptr<f32, l1>, !pto.ptr<f32, fb>, i64, i64, i64, i64
```

---

### Left / Right Tile Load Model

`pto.mte_l1_l0a` and `pto.mte_l1_l0b` move L1 cube-fractal tiles into the
compute operand domains. `%src` must already point to an L1 cube-fractal tile;
these ops do not convert arbitrary row-major matrices. Use
`pto.mte_gm_l1_frac` first when the original data is plain ND/DN layout.

If `transpose = true`, the selected logical source tile is transposed before it
is placed in the destination operand domain. Omitting the attribute means
`transpose = false`.

The `%start_row` and `%start_col` operands select the row and column offset of
the source tile extraction start position. Frontends that expose these as
optional user arguments must materialize `0` for both operands when omitted.

### `pto.mte_l1_l0a`

- **syntax:**
```mlir
pto.mte_l1_l0a %src, %dst, %m, %k, %start_row, %start_col
  : !pto.ptr<T, l1>, !pto.ptr<T, l0a>, i64, i64, i64, i64
```
- **semantics:** Load a logical `%m x %k` left tile from L1 `l1` into `l0a`.

**Parameter Table:**

| Parameter | Width | Description |
|-----------|-------|-------------|
| `%src` | ptr | L1 cube-fractal source tile in `l1` |
| `%dst` | ptr | Left operand destination in `l0a` |
| `%m` | i64 | Logical M extent |
| `%k` | i64 | Logical K extent |
| `%start_row` | i64 | Source row offset |
| `%start_col` | i64 | Source column offset |
| `transpose` | attr | Optional boolean source-tile transpose before destination placement |

**Constraints:**

- `%src` must be in `l1`, `%dst` must be in `l0a`.
- `%src` and `%dst` must satisfy the target alignment for Cube tile loads.
- `transpose = true` requires a tile shape supported by the element-type
  transpose granularity.

**Example:**

```mlir
pto.mte_l1_l0a %l1_a, %l0a, %c16_i64, %c32_i64, %c0_i64, %c0_i64
  : !pto.ptr<f16, l1>, !pto.ptr<f16, l0a>, i64, i64, i64, i64
```

---

### `pto.mte_l1_l0b`

- **syntax:**
```mlir
pto.mte_l1_l0b %src, %dst, %k, %n, %start_row, %start_col
  : !pto.ptr<T, l1>, !pto.ptr<T, l0b>, i64, i64, i64, i64
```
- **semantics:** Load a logical `%k x %n` right tile from L1 `l1` into
  `l0b`.

**Parameter Table:**

| Parameter | Width | Description |
|-----------|-------|-------------|
| `%src` | ptr | L1 cube-fractal source tile in `l1` |
| `%dst` | ptr | Right operand destination in `l0b` |
| `%k` | i64 | Logical K extent |
| `%n` | i64 | Logical N extent |
| `%start_row` | i64 | Source row offset |
| `%start_col` | i64 | Source column offset |
| `transpose` | attr | Optional boolean source-tile transpose before destination placement |

**Constraints:**

- `%src` must be in `l1`, `%dst` must be in `l0b`.
- `%src` and `%dst` must satisfy the target alignment for Cube tile loads.
- `transpose = true` requires a tile shape supported by the element-type
  transpose granularity.

**Example:**

```mlir
pto.mte_l1_l0b %l1_b, %l0b, %c32_i64, %c16_i64, %c0_i64, %c0_i64
  : !pto.ptr<f16, l1>, !pto.ptr<f16, l0b>, i64, i64, i64, i64
```

---

### MX Scale Load Model

MX scale loads prepare the scale payloads consumed by `pto.mad_mx*`. Each scale
entry applies to one 32-element K group.

- Left scale logical shape: `[M, ceil(K / 32)]`.
- Right scale logical shape: `[ceil(K / 32), N]`.
- L1 source data is organized as 32B scale fragments in the same logical order
  as the associated data tile.

### MX Scale Load Operands

`pto.mte_l1_l0a_mx` and `pto.mte_l1_l0b_mx` store the following operands as
independently optional fields, but a valid operation supplies exactly one
complete operand group:

- **Shape-derived group:** `m`, `k`, `start_row`, `start_col` for L0A, or
  `k`, `n`, `start_row`, `start_col` for L0B. PTOAS derives the MX traversal
  from the matrix shape and element type.
- **Full MX group:** `x_start`, `y_start`, `x_step`, `y_step`, `src_stride`,
  `dst_stride`. Use this group when the scale source has an explicit physical
  layout that cannot be derived from the logical matrix shape.

The two groups are mutually exclusive. A partial group is invalid. `x_start`
and `y_start` are MX scale-fragment grid coordinates; `x_step` and `y_step`
are grid traversal extents; `src_stride` and `dst_stride` are grid-row strides.
The full MX values are not logical element counts or byte offsets.

The named field spelling is available when an incomplete or mixed operation
must be diagnosed. Complete groups print in the compact positional spelling
used by the examples below.

```mlir
pto.mte_l1_l0a_mx %src, %dst,
  x_start(%x_start), y_start(%y_start), x_step(%x_step), y_step(%y_step),
  src_stride(%src_stride), dst_stride(%dst_stride)
  : !pto.ptr<T, l1>, !pto.ptr<T, l0a>, i64, i64, i64, i64, i64, i64
```

`%src` must be in `l1` and address a 32-byte-aligned MX scale fragment.
`%dst` must be in the matching `l0a` or `l0b` space. It is always the real
staged L0 byte pointer and must be 16-byte aligned.

### `pto.mte_l1_l0a_mx`

- **syntax:**
```mlir
pto.mte_l1_l0a_mx %src, %dst, %m, %k, %start_row, %start_col
  : !pto.ptr<T, l1>, !pto.ptr<T, l0a>, i64, i64, i64, i64
```
- **semantics:** Load left-side MX scale fragments for a logical `%m x %k`
  left data tile.

**Parameter Table:**

| Parameter | Width | Description |
|-----------|-------|-------------|
| `%src` | ptr | L1 MX scale source in `l1` |
| `%dst` | ptr | Left-side MX payload destination associated with `l0a` |
| `%m` | i64 | M extent of the associated left data tile |
| `%k` | i64 | K extent; scale grouping is by 32 K elements |
| `%start_row` | i64 | Source MX-fractal row offset in the packed L1 big matrix |
| `%start_col` | i64 | Source MX-fractal column offset in the packed L1 big matrix |

**Constraints:**

- `%src` must be in `l1`, `%dst` must be in `l0a`.
- `%src` must be 32-byte aligned to an MX scale fragment; `%dst` must be
  16-byte aligned for the MX destination address unit.
- Use either the complete shape-derived group or the complete full MX group
  from [MX Scale Load Operands](#mx-scale-load-operands).

**Example:**

```mlir
pto.mte_l1_l0a_mx %l1_a_scale, %l0a_scale, %c16_i64, %c64_i64, %c0_i64, %c0_i64
  : !pto.ptr<f8E4M3FN, l1>, !pto.ptr<f8E4M3FN, l0a>, i64, i64, i64, i64
```

---

### `pto.mte_l1_l0b_mx`

- **syntax:**
```mlir
pto.mte_l1_l0b_mx %src, %dst, %k, %n, %start_row, %start_col
  : !pto.ptr<T, l1>, !pto.ptr<T, l0b>, i64, i64, i64, i64
```
- **semantics:** Load right-side MX scale fragments for a logical `%k x %n`
  right data tile.

**Parameter Table:**

| Parameter | Width | Description |
|-----------|-------|-------------|
| `%src` | ptr | L1 MX scale source in `l1` |
| `%dst` | ptr | Right-side MX payload destination associated with `l0b` |
| `%k` | i64 | K extent; scale grouping is by 32 K elements |
| `%n` | i64 | N extent of the associated right data tile |
| `%start_row` | i64 | Source MX-fractal row offset in the packed L1 big matrix |
| `%start_col` | i64 | Source MX-fractal column offset in the packed L1 big matrix |

**Constraints:**

- `%src` must be in `l1`, `%dst` must be in `l0b`.
- `%src` must be 32-byte aligned to an MX scale fragment; `%dst` must be
  16-byte aligned for the MX destination address unit.
- Use either the complete shape-derived group or the complete full MX group
  from [MX Scale Load Operands](#mx-scale-load-operands).

**Example:**

```mlir
pto.mte_l1_l0b_mx %l1_b_scale, %l0b_scale, %c64_i64, %c16_i64, %c0_i64, %c0_i64
  : !pto.ptr<f8E4M3FN, l1>, !pto.ptr<f8E4M3FN, l0b>, i64, i64, i64, i64
```

---

## FIXPIPE MTE Ops

`pto.mte_l0c_l1*` writes logical accumulator results from `l0c` to `l1`, `gm`,
or `ub`. The family shares this pipeline order:

```text
1. Read logical acc[m, n] from %src using the selected layout mode.
2. Optionally participate in consumer-side unit-flag synchronization.
3. Optionally apply pre_quant(payload, mode).
4. Optionally apply pre_relu(payload, mode), then optional clip.
5. Convert to the destination element type using sat/nosat behavior.
6. Write to the selected destination layout and address space.
7. Apply store-target effects such as GM atomic or UB dual destination.
```

Only the clauses documented here affect `pto.mte_l0c_l1*`. Other transforms
must be represented by separate PTO ops before producing `l0c` or after the
writeback destination is materialized.

### FIXPIPE Common Clauses

| Clause | Values | Effect |
|--------|--------|--------|
| `unit_flag(...)` | `check_only`, `check_and_clear` | Checks that the accumulator tile is ready for consumption. `check_and_clear` also clears the consumed tile state for later reuse. Omit when the schedule does not use unit flags. |
| `pre_quant(%payload, mode = ...)` | see below | Applies the selected pre-quantization or conversion before ReLU/clip and final store. |
| `pre_relu([%payload, ]mode = ...[, clip = %clip])` | `no_relu`, `normal_relu`, `scalar_relu`, `vector_relu` | Applies ReLU-family activation before final destination conversion. `clip` is part of this clause and applies after the selected ReLU mode. |
| `nz2nd` / `nz2dn(...)` / `nz2nz(...)` | layout modes | Selects how logical `acc[m, n]` is written to the destination layout. |
| `loop3(%count, %src_stride3, %dst_stride3)` | i64 triple | Repeats the whole selected `m x n` writeback pattern. |
| `sat` / `sat(preserve_nan)` / `nosat` | flags | Selects final conversion behavior for floating exceptional values and finite overflow where the destination type is affected. |

`pre_quant` legal modes:

```text
f32_f16,
qf322hif8_pre_vec, qf322hif8_pre_scalar,
qf322hif8_pre_hybrid_vec, qf322hif8_pre_hybrid_scalar,
deqs32_int_vec, deqs32_int_scalar,
req8_vec, req8_scalar,
deqf16_vec, deqf16_scalar,
qf322fp8_pre_vec, qf322fp8_pre_scalar,
qf322f32_pre_vec, qf322f32_pre_scalar,
f32_bf16,
qf162b8_pre_vec, qf162b8_pre_scalar,
qf162s4_pre_vec, qf162s4_pre_scalar,
req4_vec, req4_scalar,
qf322b8_pre_vec, qf322b8_pre_scalar,
qf322s4_pre_vec, qf322s4_pre_scalar,
deqs16_vec, deqs16_scalar,
qf162s16_pre_vec, qf162s16_pre_scalar,
qf322f16_pre_vec, qf322f16_pre_scalar,
qf322bf16_pre_vec, qf322bf16_pre_scalar,
qs322bf16_pre_vec, qs322bf16_pre_scalar
```

`_scalar` modes take one floating scalar payload (`f16`, `bf16`, or `f32`)
broadcast to the whole logical output tile. `f16` and `bf16` scalar payloads
are first interpreted as numeric values and widened to `f32`; `f32` payloads
are used directly. `_vec` modes take a `!pto.ptr<f16|bf16|f32, fb>`
payload pointer. The pointer element type is the logical parameter element
type, not a packed transport carrier. The pointer names the first parameter
row for this store; later rows
advance in the same channel/NZ order as the logical accumulator elements
consumed by the selected layout mode. Each vector pre-quant row is a 128B
parameter row prepared by `pto.mte_l1_fb`; each row supplies the per-channel
scale and any mode-specific offset/sign controls used by the selected
quantization family. Vector pre-ReLU rows are 64B parameter rows and supply
the per-channel alpha values consumed by `vector_relu`.

`pre_quant` mode families:

| Family | Acc source | Result meaning | Payload |
|--------|------------|----------------|---------|
| `f32_f16`, `f32_bf16` | `f32` | Convert f32 accumulator values to f16 or bf16; rounding is nearest, ties to even | Scalar payload is required by syntax but does not select per-channel scaling |
| `qf322hif8_pre_*`, `qf322fp8_pre_*` | `f32` | Scale and quantize f32 to hif8/fp8-style destination payloads | Scalar scale or vector scale rows; hybrid modes use the target hybrid rule |
| `qf322f32_pre_*` | `f32` | Apply quant scaling while keeping f32 destination values | Scalar scale or vector scale rows |
| `qf322f16_pre_*`, `qf322bf16_pre_*` | `f32` | Scale f32, then convert to f16 or bf16 destination values | Scalar scale or vector scale rows |
| `qf322b8_pre_*`, `qf322s4_pre_*` | `f32` | Scale, offset, round, and narrow f32 to 8-bit or signed 4-bit integer payloads | Scalar or vector scale/offset parameter set |
| `qf162b8_pre_*`, `qf162s4_pre_*` | `f32` | Convert through an f16-domain pre-stage, then scale/narrow to integer payloads | Scalar or vector scale/offset parameter set |
| `qf162s16_pre_*` | `i32` | Convert through an f16-domain pre-stage, then scale/narrow to signed 16-bit payloads | Scalar or vector scale/offset parameter set |
| `deqs32_int_*`, `deqs16_*` | `i32` | Rescale integer accumulator values in an integer destination family | Scalar or vector multiplier/offset parameter set |
| `req8_*`, `req4_*` | `i32` | Requantize i32 accumulator values to 8-bit or 4-bit integer payloads | Scalar or vector multiplier/offset/sign parameter set |
| `deqf16_*` | `i32` | Dequantize i32 accumulator values to f16 destination values | Scalar or vector multiplier/offset parameter set |
| `qs322bf16_pre_*` | `i32` | Scale i32 accumulator values and convert to bf16 destination values | Scalar or vector multiplier/offset parameter set |

The mode name determines the accepted accumulator source family. `f32_f16`,
`f32_bf16`, `qf322hif8_pre_*`, `qf322fp8_pre_*`, `qf322f32_pre_*`,
`qf322f16_pre_*`, `qf322bf16_pre_*`, `qf322b8_pre_*`,
`qf322s4_pre_*`, `qf162b8_pre_*`, and `qf162s4_pre_*` consume `f32`
accumulator values. `deqs32_int_*`, `deqs16_*`, `req8_*`, `req4_*`,
`deqf16_*`, `qf162s16_pre_*`, and `qs322bf16_pre_*` consume `i32`
accumulator values. The final destination element type must match the result
family implied by the mode name; for example, `qf322f16_pre_*` writes an
f16-family result, while `req8_*` writes an 8-bit integer-family result.

Integer quantization families with `b8` in the name can produce either signed
8-bit or unsigned 8-bit results according to the sign control carried by the
scalar or vector parameter set. Families with `s4` or `s16` produce signed
4-bit or signed 16-bit results. Offset fields are added after scaling and
before the final narrow/saturate step. When a family has no offset/sign in its
payload, the payload scale alone controls the conversion.

`pre_relu` semantics:

```text
no_relu:      y = x
normal_relu:  y = max(x, 0)
scalar_relu:  y = x >= 0 ? x : alpha * x
vector_relu:  y = x >= 0 ? x : alpha[channel] * x
```

`scalar_relu` takes a floating scalar payload (`f16`, `bf16`, or `f32`) and
broadcasts it to all negative values in the logical tile. `vector_relu` takes
a `!pto.ptr<f16|bf16|f32, fb>` pointer whose elements are per-channel
alpha values and whose 64B rows follow the same channel/NZ order as the store.
`no_relu` and `normal_relu` do not take a payload. If
`clip = %clip` is present:

```text
y = min(y, clip)
```

`sat`, `sat(preserve_nan)`, and `nosat` control final conversion to destination
element types affected by FIXPIPE saturation:

- `sat`: finite overflow clamps to the destination finite range; `+/-inf`
  clamps to finite extrema; `nan` writes as 0.
- `sat(preserve_nan)`: same finite overflow and infinity handling as `sat`,
  but NaN writes as NaN when the destination format can represent NaN. This is
  intended for fp8 and hif8 destination families; for formats without a NaN
  encoding it is equivalent to `sat`.
- `nosat`: finite overflow may produce destination exceptional values;
  exceptional input values are preserved where the destination format supports
  them.
- For fp8 and hif8 destination families, `nosat` preserves NaN; overflow
  becomes the destination exceptional value when the destination encoding
  supports it.
- For integer destination families, `sat`/`nosat` is not the integer overflow
  policy; integer narrowing and clipping are determined by the selected
  pre-quant mode, its payload, and any `clip` clause.
- For `f32` destinations, floating exceptional values are preserved; `sat`
  does not force f32 `inf`/`nan` into finite values.

### FIXPIPE Layout Model

`%src` points to the base accumulator tile. `%m` and `%n` select the logical
result rectangle to write. If the physical accumulator tile contains dummy rows
or lanes outside that rectangle, they are not written to the destination.

Layout modes:

| Mode | Destination layout | Extra operand |
|------|--------------------|---------------|
| omitted | Normal target-profile writeback layout | none |
| `nz2nd` | Logical ND order | none |
| `nz2dn(%loop0_src_stride)` | Logical D/N-swapped order | `%loop0_src_stride` in C0-size units |
| `nz2nz(%split)` | NZ-style destination | `%split`, destination split point |

`%src_stride` is measured in C0-size units and advances the accumulator source
between adjacent source groups selected by the layout mode. `%dst_stride` is
measured in destination elements and advances the destination row/group
selected by the layout mode. In `loop3`, `%src_stride3` is in C0-size units and
`%dst_stride3` is in destination elements.

Reference semantics:

```text
repeat_count = loop3.count if loop3 is present else 1

for r in 0 .. repeat_count-1:
  src_r = src + r * loop3.src_stride * 32
  dst_r = dst + r * loop3.dst_stride * sizeof(dst_element)

  for m in 0 .. M-1:
    for n in 0 .. N-1:
      x = read_acc_logical(src_r, m, n, src_stride, layout_mode)

      if pre_quant:
        x = apply_pre_quant(x, payload, mode)

      if pre_relu:
        x = apply_pre_relu(x, payload, mode)
        if clip:
          x = min(x, clip)

      y = convert_to_destination_type(x, sat_or_nosat)
      write_destination(dst_r, y, m, n, dst_stride, layout_mode)
```

When no layout clause is present, the store uses the target-profile normal
writeback layout for the destination address space. This mode performs no
explicit ND/DN/NZ layout transform; `%dst_stride` is still the destination
start-to-start stride in destination elements for the normal writeback rows or
groups.

For `nz2nd`, `write_destination` stores logical `y[m, n]` in ND order. For
`nz2dn`, it stores the same logical result with the D/N dimensions swapped; the
extra `%loop0_src_stride` selects how the swapped source walk advances through
the accumulator tile. For `nz2nz`, it preserves NZ-style destination packing
and uses `%split` as the destination split point.

### `pto.mte_l0c_l1`

- **syntax:**
```mlir
pto.mte_l0c_l1 %src, %dst, %m, %n, %src_stride, %dst_stride
    [, unit_flag(check_only | check_and_clear)]?
    [, pre_quant(%payload, mode = <quant_pre_mode>)]?
    [, pre_relu([%payload, ]mode = <relu_pre_mode> [, clip = %clip])]?
    [, nz2nd | nz2dn(%loop0_src_stride) | nz2nz(%split)?]
    [, loop3(%count, %src_stride3, %dst_stride3)]?
    [, sat | sat(preserve_nan) | nosat]?
  : ...
```
- **semantics:** FIXPIPE writeback from `l0c` to L1 `l1`.

**Parameter Table:**

| Parameter | Width | Description |
|-----------|-------|-------------|
| `%src` | buffer-like | Accumulator source in `l0c` |
| `%dst` | buffer-like | L1 destination in `l1` |
| `%m` | i64 | Logical M element count |
| `%n` | i64 | Logical N element count |
| `%src_stride` | i64 | Source stride in C0-size units |
| `%dst_stride` | i64 | Destination stride in destination elements |
| optional clauses | - | See [FIXPIPE Common Clauses](#fixpipe-common-clauses) and [FIXPIPE Layout Model](#fixpipe-layout-model) |

**Constraints:**

- Clauses must appear in canonical order:
  `unit_flag` -> `pre_quant` -> `pre_relu` -> layout -> `loop3` -> `sat`/`nosat`.
- `pre_quant` requires payload and mode together.
- Vector `pre_quant` modes require a `fb` pointer with `f16`, `bf16`, or
  `f32` element type.
- Scalar `pre_quant` modes require an `f16`, `bf16`, or `f32` scalar payload.
- `pre_quant` source element type must be `f32` or `i32`, and the selected
  mode must be compatible with the source and destination element types.
- `no_relu` and `normal_relu` do not accept a payload.
- `scalar_relu` requires an `f16`, `bf16`, or `f32` scalar payload.
- `vector_relu` requires a `fb` pointer with `f16`, `bf16`, or `f32`
  element type.
- `clip` can appear only inside `pre_relu(...)`.
- `clip` is supported for destination `f16`, `ui8`, and signed/signless
  4/8/16-bit integer destinations. The clip payload must match the destination
  family: `f16` for f16, 16-bit unsigned/signless payload for `ui8`, and
  signed/signless `i4/i8/i16` for signed integer destinations.
- `nz2dn` requires `%loop0_src_stride`; `nz2nd` and `nz2nz` do not accept it.
- `unit_flag` must be omitted when `nz2dn(%loop0_src_stride)` uses a value
  other than 1.
- `nz2nz` requires `f32` destination element type and does not accept `loop3`.
- `sat`, `sat(preserve_nan)`, and `nosat` are mutually exclusive.

**Example:**

```mlir
pto.mte_l0c_l1 %l0c, %l1_out, %c16_i64, %c32_i64, %c16_i64, %c32_i64,
  pre_quant(%c1_f32, mode = qf322f16_pre_scalar),
  pre_relu(%c025_f32, mode = scalar_relu),
  nz2nd,
  sat
  : !pto.ptr<f32, l0c>, !pto.ptr<f16, l1>, i64, i64, i64, i64, f32, f32
```

---

### `pto.mte_l0c_gm`

- **syntax:**
```mlir
pto.mte_l0c_gm %src, %dst, %m, %n, %src_stride, %dst_stride, %sid, %l2_cache_ctrl
    [, unit_flag(check_only | check_and_clear)]?
    [, pre_quant(%payload, mode = <quant_pre_mode>)]?
    [, pre_relu([%payload, ]mode = <relu_pre_mode> [, clip = %clip])]?
    [, nz2nd | nz2dn(%loop0_src_stride) | nz2nz(%split)?]
    [, loop3(%count, %src_stride3, %dst_stride3)]?
    [, sat | sat(preserve_nan) | nosat]?
    [, atomic(type = <atomic_type>, op = <atomic_op>)]?
  : ...
```
- **semantics:** FIXPIPE writeback from `l0c` to GM. The data transform clauses
  match `pto.mte_l0c_l1`; GM-specific operands select the GM write path and
  optional atomic update behavior.

**Parameter Table:**

| Parameter | Width | Description |
|-----------|-------|-------------|
| `%src`, `%m`, `%n`, `%src_stride` | - | Same as `pto.mte_l0c_l1` |
| `%dst` | buffer-like | GM destination |
| `%dst_stride` | i64 | GM destination stride in destination elements |
| `%sid` | i64 | GM stream/session hint for the OUT/GM path; does not change written values |
| `%l2_cache_ctrl` | i64 | GM store cache hint; does not change written values |
| `atomic(type = ..., op = ...)` | clause | Optional GM read-modify-write |
| other optional clauses | - | Same as `pto.mte_l0c_l1` |

`%sid` and `%l2_cache_ctrl` affect the memory path only. They do not change
the logical result, destination layout, numeric conversion, or atomic
operation. For target-profile GM writeback, constant `%sid` values must be in
`[0, 3]`; use `0` unless the surrounding memory system deliberately assigns a
different stream/session hint. Constant `%l2_cache_ctrl` values must fit in the
target cache-control hint range `[0, 15]`.

`atomic(type = T, op = add|max|min)` performs an atomic read-modify-write at
each GM destination element. `add` accumulates the converted value into the
existing GM value. `max` and `min` compare using `T` and write the selected
value. Supported atomic types are `f32`, `f16`, `bf16`, `s32`, `s16`, and `s8`.

**Constraints:**

- `atomic(...)` is valid only on `pto.mte_l0c_gm`.
- `atomic` requires both `type` and `op`.
- Atomic op values are `add`, `max`, and `min`.
- If `%sid` or `%l2_cache_ctrl` is a constant, it must be in the target range
  described above.
- Other constraints match `pto.mte_l0c_l1`.

**Example:**

```mlir
pto.mte_l0c_gm %l0c, %out, %c16_i64, %c32_i64, %c16_i64, %c32_i64,
  %c0_i64, %c0_i64,
  pre_quant(%c1_f32, mode = qf322f16_pre_scalar),
  nz2nd,
  atomic(type = f16, op = add)
  : !pto.ptr<f32, l0c>, !pto.ptr<f16, gm>, i64, i64, i64, i64, i64, i64, f32
```

---

### `pto.mte_l0c_ub`

- **syntax:**
```mlir
pto.mte_l0c_ub %src, %dst, %m, %n, %src_stride, %dst_stride,
    dst_mode(%sub_blockid | split_m | split_n)
    [, unit_flag(check_only | check_and_clear)]?
    [, pre_quant(%payload, mode = <quant_pre_mode>)]?
    [, pre_relu([%payload, ]mode = <relu_pre_mode> [, clip = %clip])]?
    [, nz2nd | nz2dn(%loop0_src_stride) | nz2nz(%split)?]
    [, loop3(%count, %src_stride3, %dst_stride3)]?
    [, sat | sat(preserve_nan) | nosat]?
  : ...
```
- **semantics:** FIXPIPE writeback from `l0c` to UB. The data transform clauses
  match `pto.mte_l0c_l1`; UB-specific operands select single or dual destination
  behavior.

**Parameter Table:**

| Parameter | Width | Description |
|-----------|-------|-------------|
| `%src`, `%m`, `%n`, `%src_stride` | - | Same as `pto.mte_l0c_l1` |
| `%dst` | buffer-like | UB destination |
| `%dst_stride` | i64 | UB destination stride in destination elements |
| `dst_mode(%sub_blockid)` | i64 operand | Single-destination mode. `%sub_blockid` selects UB sub-block `0` or `1`; the value may be dynamic. |
| `dst_mode(split_m)` | keyword | Dual-destination mode that splits the logical tile along M. |
| `dst_mode(split_n)` | keyword | Dual-destination mode that splits the logical tile along N. |
| optional clauses | - | Same as `pto.mte_l0c_l1`; `atomic(...)` is not supported |

In `dst_mode(%sub_blockid)`, the whole logical result tile is written to the
selected UB sub-block using the selected layout mode and `%dst` as that
sub-block's base destination pointer.

In `dst_mode(split_m)`, the logical tile is split into two M ranges:
`[0, m/2)` and `[m/2, m)`. The first range is written to UB sub-block 0 and the
second range is written to UB sub-block 1. Each sub-block sees its own
destination origin at `%dst`; within each sub-block, the written logical tile
has shape `(m / 2) x n`.

In `dst_mode(split_n)`, the logical tile is split into two N ranges:
`[0, n/2)` and `[n/2, n)`. The first range is written to UB sub-block 0 and the
second range is written to UB sub-block 1. Each sub-block sees its own
destination origin at `%dst`; within each sub-block, the written logical tile
has shape `m x (n / 2)`.

**Constraints:**

- `atomic(...)` is not supported.
- `dst_mode(%sub_blockid)` writes the whole logical tile to one UB sub-block.
  Runtime `%sub_blockid` values must be `0` or `1`; constant values are checked
  statically when available.
- `dst_mode(split_m)` splits the logical tile along M into two equal-height
  sub-block regions. `%m` must be even; each sub-block receives an
  `(m / 2) x n` tile.
- `dst_mode(split_n)` splits the logical tile along N into two equal-width
  sub-block regions. `%n` must be a multiple of 32; each sub-block receives an
  `m x (n / 2)` tile.
- Dual-destination split modes are valid only for target-supported normal or
  `nz2nd` writeback cases with pre-quant, pre-ReLU/clip, and other transform
  clauses omitted.
- Other constraints match `pto.mte_l0c_l1`.

**Example:**

```mlir
pto.mte_l0c_ub %l0c, %ub_out, %c16_i64, %c32_i64, %c16_i64, %c32_i64,
  dst_mode(%c1_i64),
  nz2nd
  : !pto.ptr<f32, l0c>, !pto.ptr<f32, ub>, i64, i64, i64, i64, i64
```

---

## Typical Usage / Patterns

A common Cube matmul flow is:

```text
GM row/column-major data
  -> pto.mte_gm_l1_frac or pto.mte_gm_l1 into L1 `l1`
  -> pto.mte_l1_l0a / pto.mte_l1_l0b into `l0a`/`l0b` tiles
  -> pto.mad* produces `l0c` tile
  -> pto.mte_l0c_l1* writes L1, GM, or UB with optional FIXPIPE transforms
```

For MX matmul, load the data tiles and the matching MX scale payloads before
calling `pto.mad_mx*`:

```text
left data tile + left scale payload
right data tile + right scale payload
  -> pto.mad_mx*
```

For bias matmul, prepare the bias vector in `bt` with `pto.mte_l1_bt` before the
`pto.mad_bias` / `pto.mad_mx_bias` consumer.
