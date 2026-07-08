# Cube Matrix Multiply Operations

Cube operations target the AIC (Cube) hardware unit for matrix multiplication and
staged data movement. They are only available inside `@pto.ckernel` function
bodies. All Cube operands use `pto.ptr<T, addr_space>` raw pointers — no
`vecscope` execution scope is used.

## Address Spaces

Cube operations use typed PTO pointers to describe logical storage domains. This
guide uses only the canonical address-space names:
`gm`/`l1`/`l0a`/`l0b`/`l0c`/`bt`/`fb`/`ub`.

| Canonical address space | Canonical IR Type | DSL enum value | Logical role |
|-------------------------|-------------------|----------------|--------------|
| `gm` | `!pto.ptr<T, gm>` | `MemorySpace.GM` | Global memory |
| `l1` | `!pto.ptr<T, l1>` | `MemorySpace.MAT` | L1 matrix staging buffer |
| `l0a` | `!pto.ptr<T, l0a>` | `MemorySpace.LEFT` | Left matrix operand tile for Cube compute |
| `l0b` | `!pto.ptr<T, l0b>` | `MemorySpace.RIGHT` | Right matrix operand tile for Cube compute |
| `l0c` | `!pto.ptr<T, l0c>` | `MemorySpace.ACC` | Accumulator/result tile produced by Cube compute |
| `bt` | `!pto.ptr<T, bt>` | `MemorySpace.BIAS` | Bias vector payload consumed by bias matmul forms |
| `fb` | `!pto.ptr<T, fb>` | `MemorySpace.SCALING` | FIXPIPE and MX scaling/parameter payloads |
| `ub` | `!pto.ptr<T, ub>` | `MemorySpace.UB` | Unified Buffer source/destination for vector-side use |

## Shared Infrastructure

Cube operations reuse general tile and pointer facilities documented elsewhere:

| Facility | Description | Reference |
|----------|-------------|-----------|
| `pto.Tile` | Allocate a tile buffer with address space | [Type System — Tile Type Definition](05-type-system.md#tile-type-definition) |
| `.as_ptr()` | Get raw pointer from Tile / TensorView | [Frontend Operations — Pointer Construction](07-frontend-operations.md#pointer-construction-advanced-tier) |
| `pto.addptr` | Element-offset a pointer | [Frontend Operations — Pointer Construction](07-frontend-operations.md#pointer-construction-advanced-tier) |

---

## Matrix Compute Operations

### Common Cube Operand Model

Unless an op says otherwise:

- Shape operands such as `m`, `n`, `k`, and `shape=(n_value, d_value)` are
  logical element counts, not byte counts.
- Burst lengths and strides in `pto.mte_gm_l1` / `pto.mte_l1_ub` are byte counts.
- Pointer operands select the base address of the logical object. Sub-tile
  selection is expressed by forming a different pointer before calling the op,
  unless the op exposes an explicit layout or group operand.
- Cache/session controls may affect the transfer path but do not change the
  mathematical value read or written.

The `pto.mad*` family computes logical matrix multiplication over tiles already
prepared in `l0a` and `l0b`:

```text
lhs: M x K
rhs: K x N
dst: M x N
```

Element types are inferred from `lhs`, `rhs`, `dst`, and `bias` pointer types.
There is no separate dtype selector on the matmul ops.

### MAD Common Clauses

All six matmul variants share the same logical clause model:

| Clause | Values | Effect |
|--------|--------|--------|
| `unit_flag` | `"check_only"`, `"check_and_set"`, or `None` | Producer-side tile synchronization. `"check_only"` verifies the destination slot can be used. `"check_and_set"` additionally publishes the produced `dst` tile for later consumers. |
| `disable_gemv` | `True` / `False` | Applies only when `m = 1`. `False` means GEMV A-vector consumption layout. `True` forces normal matmul left-tile layout. The mathematical result is unchanged. |
| `sat` | `"sat"`, `"nosat"`, or `None` | Floating/MX exceptional-value mode. `"sat"` normalizes exceptional inputs before arithmetic and saturates finite overflow to the finite type range. `"nosat"` preserves exceptional inputs and allows exceptional outputs. `None` inherits the surrounding execution mode. |
| `tf32_mode` | `"round_even"`, `"round_away"`, or `None` | Valid only for non-MX `f32 x f32 -> f32`. Inputs are rounded to TF32 precision before multiplication; accumulation and output remain `f32`. |
| `n_dir` | `True` / `False` | Requests N-direction result production order for schedules that combine compute with later layout movement. It does not change `dst[m, n]`. |

Reference semantics for non-MX forms:

```text
product[m, n] = sum k in 0 .. K-1:
                  numeric_lhs(lhs[m, k]) * numeric_rhs(rhs[k, n])

pto.mad:      dst[m, n] = product[m, n]
pto.mad_acc:  dst[m, n] = dst[m, n] + product[m, n]
pto.mad_bias: dst[m, n] = product[m, n] + bias[n]
```

For integer forms, per-input offset correction is not an operand of `pto.mad*`.
If a quantized algorithm requires offset correction, apply it before the Cube
operand tiles are loaded.

### MX Matmul Model

`pto.mad_mx*` additionally applies microscaling. The scale payloads are loaded
with the MX left/right tile load ops and are associated with the selected
`lhs` / `rhs` tiles; they are not direct operands of `pto.mad_mx*`.

The K dimension is partitioned into 32-element groups:

```text
k_group = floor(k / 32)

mx_product[m, n] =
  sum k in 0 .. K-1:
    (lhs[m, k] * lhs_scale[m, k_group]) *
    (rhs[k, n] * rhs_scale[k_group, n])
```

Current MX tiles use `f8E4M3FN` data. `k` must satisfy the target MX grouping
rule. On the current target profile, MX matmul consumes K in 64-element
multiples, which contain two 32-element scale groups.

### `pto.mad` — zero-init matmul

#### `pto.mad(lhs: PtrType, rhs: PtrType, dst: PtrType, m: int, n: int, k: int, *, unit_flag: Literal["check_only", "check_and_set"] | None = None, disable_gemv: bool = False, sat: Literal["sat", "nosat"] | None = None, tf32_mode: Literal["round_even", "round_away"] | None = None, n_dir: bool = False) -> None`

**Description**: Zero-init cube matrix multiply. Clears the accumulator and
computes `dst[m, n] = sum_k(lhs[m, k] * rhs[k, n])`.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `lhs` | `pto.ptr<T, l0a>` | Left operand tile in `l0a`, interpreted as logical `M x K` |
| `rhs` | `pto.ptr<T, l0b>` | Right operand tile in `l0b`, interpreted as logical `K x N` |
| `dst` | `pto.ptr<U, l0c>` | Accumulator destination tile in `l0c`, interpreted as logical `M x N` |
| `m` | `int` | Logical M element count |
| `n` | `int` | Logical N element count |
| `k` | `int` | Logical K element count |
| `unit_flag`, `disable_gemv`, `sat`, `tf32_mode`, `n_dir` | see above | See [MAD Common Clauses](#mad-common-clauses) |

**Constraints**:
- `lhs`, `rhs`, and `dst` must be in `l0a`, `l0b`, and `l0c`.
- `m`, `n`, and `k` must be positive and satisfy the target shape limits for
  the selected element-type combination.
- `tf32_mode` requires `f32` `lhs`, `rhs`, and `dst`.
- `sat` is valid only for floating element-type combinations.
- Packed 4-bit integer data requires `k` to select an even number of K
  elements.

**Example**:
```python
pto.mad(
    l0a, l0b, l0c, 16, 16, 64,
    unit_flag="check_only",
    sat="sat",
)
```

---

### `pto.mad_acc` — accumulating matmul

#### `pto.mad_acc(lhs: PtrType, rhs: PtrType, dst: PtrType, m: int, n: int, k: int, *, unit_flag: Literal["check_only", "check_and_set"] | None = None, disable_gemv: bool = False, sat: Literal["sat", "nosat"] | None = None, tf32_mode: Literal["round_even", "round_away"] | None = None, n_dir: bool = False) -> None`

**Description**: Accumulating cube matrix multiply. Computes
`dst[m, n] = dst[m, n] + sum_k(lhs[m, k] * rhs[k, n])`.

**Parameters**: Same as `pto.mad`.

**Example**:
```python
pto.mad_acc(
    l0a, l0b, l0c, 16, 16, 64,
    unit_flag="check_and_set",
    tf32_mode="round_even",
)
```

---

### `pto.mad_bias` — bias-init matmul

#### `pto.mad_bias(lhs: PtrType, rhs: PtrType, dst: PtrType, bias: PtrType, m: int, n: int, k: int, *, unit_flag: Literal["check_only", "check_and_set"] | None = None, disable_gemv: bool = False, sat: Literal["sat", "nosat"] | None = None, tf32_mode: Literal["round_even", "round_away"] | None = None, n_dir: bool = False) -> None`

**Description**: Bias-init cube matrix multiply. Computes
`dst[m, n] = sum_k(lhs[m, k] * rhs[k, n]) + bias[n]`.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `bias` | `pto.ptr<U, bt>` | Bias vector in `bt`, interpreted as `N` values and broadcast across `M` |

Other parameters are the same as `pto.mad`.

**Constraints**:
- `bias` must be in `bt` address space.
- `bias` element type must match `dst` element type.
- Only `N` bias values are consumed; `bias` is not an `M x N` matrix.

**Example**:
```python
pto.mad_bias(
    l0a, l0b, l0c, bt, 16, 16, 64,
    n_dir=True,
)
```

---

### `pto.mad_mx` — zero-init MX matmul

#### `pto.mad_mx(lhs: PtrType, rhs: PtrType, dst: PtrType, m: int, n: int, k: int, *, unit_flag: Literal["check_only", "check_and_set"] | None = None, disable_gemv: bool = False, sat: Literal["sat", "nosat"] | None = None, n_dir: bool = False) -> None`

**Description**: Zero-init MX (micro-scaling) cube matrix multiply. Computes
`dst[m, n] = mx_product[m, n]`.

**Parameters**: Same as `pto.mad`, with `lhs` / `rhs` carrying matching MX
scale payloads prepared by the MX left/right tile load ops.

**Constraints**:
- Operands must use a target-supported MX dtype combination.
- Matching left and right MX scale payloads must be loaded before this op.
- `k` must satisfy the MX grouping rule described in
  [MX Matmul Model](#mx-matmul-model).
- `tf32_mode` is not a clause of MX matmul.

**Example**:
```python
pto.mad_mx(
    l0a, l0b, l0c, 16, 16, 64,
    unit_flag="check_only",
    sat="sat",
)
```

---

### `pto.mad_mx_acc` — accumulating MX matmul

#### `pto.mad_mx_acc(lhs: PtrType, rhs: PtrType, dst: PtrType, m: int, n: int, k: int, *, unit_flag: Literal["check_only", "check_and_set"] | None = None, disable_gemv: bool = False, sat: Literal["sat", "nosat"] | None = None, n_dir: bool = False) -> None`

**Description**: Accumulating MX cube matrix multiply. Computes
`dst[m, n] = dst[m, n] + mx_product[m, n]`.

**Parameters**: Same as `pto.mad`.

---

### `pto.mad_mx_bias` — MX bias-init matmul

#### `pto.mad_mx_bias(lhs: PtrType, rhs: PtrType, dst: PtrType, bias: PtrType, m: int, n: int, k: int, *, unit_flag: Literal["check_only", "check_and_set"] | None = None, disable_gemv: bool = False, sat: Literal["sat", "nosat"] | None = None, n_dir: bool = False) -> None`

**Description**: MX bias-init cube matrix multiply. Computes
`dst[m, n] = mx_product[m, n] + bias[n]`.

**Parameters**: Same as `pto.mad_bias`, with MX scale payload requirements from
`pto.mad_mx`.

---

## Data Movement Operations

### `pto.mte_gm_l1` — GM → L1 (cbuf)

#### `pto.mte_gm_l1(src: PtrType, dst: PtrType, len_burst: int, *, nburst: tuple[int, int, int], loops: list[tuple[int, int, int]] | None = None) -> None`

**Description**: Structured GM-to-L1 (`cbuf` / `l1`) data movement wrapper.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `src` | `pto.ptr<T, gm>` | Global memory source pointer |
| `dst` | `pto.ptr<T, l1>` | L1 (cbuf) destination pointer |
| `len_burst` | `int` | Burst length in bytes |
| `nburst` | `tuple[int, int, int]` | `(count, src_stride, dst_stride)` |
| `loops` | `list[tuple[int, int, int]]` or `None` | Optional nested loop params, each `(count_i, src_stride_i, dst_stride_i)` |

**Constraints**:
- `src` must be in `gm` address space.
- `dst` must be in `l1` address space.

**Example**:
```python
pto.mte_gm_l1(a_ptr, l1_a.as_ptr(), 16, nburst=(1, 0, 0))
```

---

### `pto.mte_l1_ub` — L1 (cbuf) → UB

#### `pto.mte_l1_ub(src: PtrType, dst: PtrType, len_burst: int, *, nburst: tuple[int, int, int], loops: list[tuple[int, int, int]] | None = None) -> None`

**Description**: Structured L1 (`cbuf`) to UB data movement wrapper.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `src` | `pto.ptr<T, l1>` | L1 source pointer |
| `dst` | `pto.ptr<T, ub>` | UB destination pointer |
| `len_burst` | `int` | Burst length in bytes |
| `nburst` | `tuple[int, int, int]` | `(count, src_stride, dst_stride)` |
| `loops` | `list[tuple[int, int, int]]` or `None` | Optional nested loop params |

**Example**:
```python
pto.mte_l1_ub(l1_src.as_ptr(), ub_dst.as_ptr(), 16, nburst=(1, 0, 0))
```

---

### `pto.mte_gm_l1_frac` — fractal load

#### `pto.mte_gm_l1_frac(src: PtrType, dst: PtrType, mode: pto.FractalMode, *, shape: tuple[int, int], src_layout: tuple[int] | tuple[int, int], dst_group: tuple[int, int, int, int], ctrl: tuple[int, bool]) -> None`

**Description**: Structured fractal-load wrapper for `nd2nz` and `dn2nz` modes.
It loads a logical 2-D GM region and writes one or more L1 NZ matrix groups.
`nd2nz` reads a logical `src[n, d]` matrix. `dn2nz` reads a logical
`src[d, n]` matrix and writes the same logical `N x D` result into NZ layout.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `src` | `pto.ptr<T, gm>` | Global memory source pointer |
| `dst` | `pto.ptr<T, l1>` | L1 destination pointer |
| `mode` | `pto.FractalMode` | `pto.FractalMode.ND2NZ` or `pto.FractalMode.DN2NZ` |
| `shape` | `tuple[int, int]` | `(n_value, d_value)` |
| `src_layout` | `tuple[int]` or `tuple[int, int]` | `(src_inner_stride,)` or `(src_inner_stride, src_outer_stride)` in bytes |
| `dst_group` | `tuple[int, int, int, int]` | `(group_count, dst_loop2_stride, dst_loop3_stride, dst_loop4_stride)` in C0-size units |
| `ctrl` | `tuple[int, bool]` | `(l2_cache_ctrl, smallc0_en)` |

**Constraints**:
- `src` must be in `gm` address space.
- `dst` must be in `l1` address space.
- Destination strides are C0-size units, not bytes and not elements. One
  C0-size unit is 32 bytes.
- `smallc0_en=True` is valid only for target-supported small-C0 cases.

**Example**:
```python
pto.mte_gm_l1_frac(a_ptr, l1_a.as_ptr(), pto.FractalMode.ND2NZ,
                   shape=(16, 16), src_layout=(32, 1024),
                   dst_group=(1, 0, 0, 0), ctrl=(0, False))
```

---

### `pto.mte_l1_bt` — L1 (cbuf) → bias table

#### `pto.mte_l1_bt(src: PtrType, dst: PtrType, len_burst: int, *, nburst: tuple[int, int, int]) -> None`

**Description**: Structured L1 (`cbuf`) to bias-table load wrapper for later
`pto.mad_bias` / `pto.mad_mx_bias` consumption. The consumer interprets the
result as an `N`-element bias vector `bias[n]`.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `src` | `pto.ptr<T, l1>` | L1 source pointer |
| `dst` | `pto.ptr<U, bt>` | Bias table destination pointer |
| `len_burst` | `int` | Number of bias-load units per burst |
| `nburst` | `tuple[int, int, int]` | `(count, src_gap, dst_gap)` in bias-load units |

**Constraints**:
- Supported source/destination type pairs: `f32→f32`, `i32→i32`, `f16→f32`, `bf16→f32`.
- Load exactly the bias values needed by the consumer tile; the payload is not
  result-shaped.

**Example**:
```python
pto.mte_l1_bt(l1_bias.as_ptr(), bt.as_ptr(), 16, nburst=(1, 0, 0))
```

---

### `pto.mte_l1_fb` — L1 (cbuf) → FIXPIPE / scaling payloads

#### `pto.mte_l1_fb(src: PtrType, dst: PtrType, len_burst: int, *, nburst: tuple[int, int, int]) -> None`

**Description**: Load FIXPIPE parameter payloads from `l1` into `fb`. Later
`pre_quant(...)` and `pre_relu(...)` clauses on `pto.mte_l0c_*` consume these
payloads through `fb` pointers.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `src` | `pto.ptr<T, l1>` | L1 source pointer |
| `dst` | `pto.ptr<U, fb>` | FIXPIPE/scaling destination pointer |
| `len_burst` | `int` | Number of parameter-load units per burst |
| `nburst` | `tuple[int, int, int]` | `(count, src_gap, dst_gap)` in parameter-load units |

**Constraints**:
- `src` must be in `l1`, `dst` must be in `fb`.
- Vector `pre_quant` consumers read 128B parameter rows from the payload
  prepared by this op.
- Vector `pre_relu` consumers read 64B parameter rows from the payload prepared
  by this op.

**Example**:
```python
pto.mte_l1_fb(l1_fp.as_ptr(), fb_fp.as_ptr(), 2, nburst=(4, 0, 0))
```

---

### `pto.mte_l1_l0a` — L1 (cbuf) → L0A

#### `pto.mte_l1_l0a(src: PtrType, dst: PtrType, m: int, k: int, *, start_row: int, start_col: int, transpose: bool = False) -> None`

**Description**: Load a logical `m x k` left tile from L1 into `l0a`. `src`
must already point to an L1 cube-fractal tile; this op does not convert an
arbitrary row-major matrix. Use `pto.mte_gm_l1_frac(...)` first when the
original data is plain ND/DN layout.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `src` | `pto.ptr<T, l1>` | L1 source pointer |
| `dst` | `pto.ptr<T, l0a>` | L0A destination pointer |
| `m` | `int` | M dimension size |
| `k` | `int` | K dimension size |
| `start_row` | `int` | Source tile row offset for the extraction start position; the DSL materializes `0` when omitted |
| `start_col` | `int` | Source tile column offset for the extraction start position; the DSL materializes `0` when omitted |
| `transpose` | `bool` | Whether to transpose the selected logical source tile before destination placement |

**Constraints**:
- `src` must be in `l1` address space.
- `dst` must be in `l0a` address space.
- `src` and `dst` must satisfy the target alignment for Cube tile loads.

**Example**:
```python
pto.mte_l1_l0a(l1_a.as_ptr(), l0a.as_ptr(), 16, 64)
```

---

### `pto.mte_l1_l0b` — L1 (cbuf) → L0B

#### `pto.mte_l1_l0b(src: PtrType, dst: PtrType, k: int, n: int, *, start_row: int, start_col: int, transpose: bool = False) -> None`

**Description**: Load a logical `k x n` right tile from L1 into `l0b`. `src`
must already point to an L1 cube-fractal tile; this op does not convert an
arbitrary row-major matrix. Use `pto.mte_gm_l1_frac(...)` first when the
original data is plain ND/DN layout.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `src` | `pto.ptr<T, l1>` | L1 source pointer |
| `dst` | `pto.ptr<T, l0b>` | L0B destination pointer |
| `k` | `int` | K dimension size |
| `n` | `int` | N dimension size |
| `start_row` | `int` | Source tile row offset for the extraction start position; the DSL materializes `0` when omitted |
| `start_col` | `int` | Source tile column offset for the extraction start position; the DSL materializes `0` when omitted |
| `transpose` | `bool` | Whether to transpose the selected logical source tile before destination placement |

**Constraints**:
- `src` must be in `l1` address space.
- `dst` must be in `l0b` address space.
- `src` and `dst` must satisfy the target alignment for Cube tile loads.

**Example**:
```python
pto.mte_l1_l0b(l1_b.as_ptr(), l0b.as_ptr(), 64, 16)
```

---

### `pto.mte_l1_l0a_mx` — MX L1 → L0A

#### `pto.mte_l1_l0a_mx(src: PtrType, dst: PtrType, m: int, k: int, *, start_row: int = 0, start_col: int = 0) -> None`

**Description**: MX-mode L1-to-L0A wrapper. It prepares both the logical left
operand tile and the associated MX scale payload that later `pto.mad_mx*`
consumes.

**Parameters**: Same as `pto.mte_l1_l0a`, except MX stage loads do not expose
`transpose`.

**Constraints**:
- The source tile must use a target-supported MX dtype such as `f8E4M3FN`.
- `k` must satisfy the MX grouping rule required by the later MX matmul.

---

### `pto.mte_l1_l0b_mx` — MX L1 → L0B

#### `pto.mte_l1_l0b_mx(src: PtrType, dst: PtrType, k: int, n: int, *, start_row: int = 0, start_col: int = 0) -> None`

**Description**: MX-mode L1-to-L0B wrapper. It prepares both the logical right
operand tile and the associated MX scale payload that later `pto.mad_mx*`
consumes.

**Parameters**: Same as `pto.mte_l1_l0b`, except MX stage loads do not expose
`transpose`.

**Constraints**:
- The source tile must use a target-supported MX dtype such as `f8E4M3FN`.
- `k` must satisfy the MX grouping rule required by the later MX matmul.

---

## Result Writeback Operations

`pto.mte_l0c_*` writes logical accumulator results from `l0c` to `l1`, `gm`,
or `ub`. The family shares this pipeline order:

```text
1. Read logical acc[m, n] from src using the selected layout mode.
2. Optionally participate in consumer-side unit-flag synchronization.
3. Optionally apply pre_quant(payload, mode).
4. Optionally apply pre_relu(payload, mode), then optional clip.
5. Convert to the destination element type using sat/nosat behavior.
6. Write to the selected destination layout and address space.
7. Apply store-target effects such as GM atomic or UB dual destination.
```

### FIXPIPE Common Clauses

| Clause | Values | Effect |
|--------|--------|--------|
| `unit_flag` | `"check_only"`, `"check_and_clear"`, or `None` | Checks that the accumulator tile is ready for consumption. `"check_and_clear"` also clears the consumed tile state for later reuse. |
| `pre_quant(payload, mode=...)` | see below | Applies the selected pre-quantization or conversion before ReLU/clip and final store. |
| `pre_relu([payload, ]mode=...[, clip=...])` | `"no_relu"`, `"normal_relu"`, `"scalar_relu"`, `"vector_relu"` | Applies ReLU-family activation before final destination conversion. `clip` applies after the selected ReLU mode. |
| `layout` | `"nz2nd"`, `("nz2dn", loop0_src_stride)`, `("nz2nz", split)`, or `None` | Selects how logical `acc[m, n]` is written to the destination layout. |
| `loop3` | `(count, src_stride3, dst_stride3)` or `None` | Repeats the whole selected `m x n` writeback pattern. |
| `sat` | `"sat"`, `"sat(preserve_nan)"`, `"nosat"`, or `None` | Selects final conversion behavior for floating exceptional values and finite overflow where the destination type is affected. |

### `pre_quant` Modes

Accepted `pre_quant` modes:

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

Payload rules:

- `_scalar` modes take one floating scalar payload (`f16`, `bf16`, or `f32`)
  broadcast to the whole logical output tile.
- `_vec` modes take a `!pto.ptr<f16|bf16|f32, fb>` pointer naming the first
  parameter row for this store.
- Vector `pre_quant` rows are 128B parameter rows prepared by `pto.mte_l1_fb`.
- Vector `pre_relu` rows are 64B parameter rows prepared by `pto.mte_l1_fb`.

Mode families:

| Family | Acc source | Result meaning | Payload |
|--------|------------|----------------|---------|
| `f32_f16`, `f32_bf16` | `f32` | Convert `f32` accumulator values to `f16` or `bf16`; rounding is nearest, ties to even | Scalar payload is required by syntax but does not select per-channel scaling |
| `qf322hif8_pre_*`, `qf322fp8_pre_*` | `f32` | Scale and quantize `f32` to hif8/fp8-style destination payloads | Scalar scale or vector scale rows; hybrid modes use the target hybrid rule |
| `qf322f32_pre_*` | `f32` | Apply quant scaling while keeping `f32` destination values | Scalar scale or vector scale rows |
| `qf322f16_pre_*`, `qf322bf16_pre_*` | `f32` | Scale `f32`, then convert to `f16` or `bf16` destination values | Scalar scale or vector scale rows |
| `qf322b8_pre_*`, `qf322s4_pre_*` | `f32` | Scale, offset, round, and narrow `f32` to 8-bit or signed 4-bit integer payloads | Scalar or vector scale/offset parameter set |
| `qf162b8_pre_*`, `qf162s4_pre_*` | `f32` | Convert through an `f16`-domain pre-stage, then scale/narrow to integer payloads | Scalar or vector scale/offset parameter set |
| `qf162s16_pre_*` | `i32` | Convert through an `f16`-domain pre-stage, then scale/narrow to signed 16-bit payloads | Scalar or vector scale/offset parameter set |
| `deqs32_int_*`, `deqs16_*` | `i32` | Rescale integer accumulator values in an integer destination family | Scalar or vector multiplier/offset parameter set |
| `req8_*`, `req4_*` | `i32` | Requantize `i32` accumulator values to 8-bit or 4-bit integer payloads | Scalar or vector multiplier/offset/sign parameter set |
| `deqf16_*` | `i32` | Dequantize `i32` accumulator values to `f16` destination values | Scalar or vector multiplier/offset parameter set |
| `qs322bf16_pre_*` | `i32` | Scale `i32` accumulator values and convert to `bf16` destination values | Scalar or vector multiplier/offset parameter set |

Additional quant semantics:

- The mode name determines the accepted accumulator source family and implied
  destination family.
- Integer quantization families with `b8` in the name can produce signed or
  unsigned 8-bit results according to the payload sign control.
- Families with `s4` or `s16` produce signed 4-bit or signed 16-bit results.
- Offset fields are added after scaling and before the final narrow/saturate
  step.

### `pre_relu` and `sat` Semantics

`pre_relu` modes:

```text
no_relu:      y = x
normal_relu:  y = max(x, 0)
scalar_relu:  y = x >= 0 ? x : alpha * x
vector_relu:  y = x >= 0 ? x : alpha[channel] * x
```

- `scalar_relu` takes an `f16`, `bf16`, or `f32` scalar payload.
- `vector_relu` takes a `!pto.ptr<f16|bf16|f32, fb>` pointer with per-channel
  alpha rows.
- `no_relu` and `normal_relu` do not take a payload.
- If `clip` is present, the post-ReLU result is `min(y, clip)`.

Final conversion behavior:

- `"sat"` clamps finite overflow to the destination finite range; `+/-inf`
  clamps to finite extrema; `nan` writes as `0`.
- `"sat(preserve_nan)"` matches `"sat"` for finite overflow and infinities, but
  preserves NaN when the destination format can represent NaN.
- `"nosat"` preserves exceptional values where the destination format supports
  them.
- For integer destination families, narrowing and clipping are governed mainly
  by the selected `pre_quant` mode, its payload, and any `clip` clause.
- For `f32` destinations, exceptional values are preserved; saturation does not
  force `inf` or `nan` to finite values.

### FIXPIPE Layout Model

`src` points to the base accumulator tile. `m` and `n` select the logical
result rectangle to write.

| Mode | Destination layout | Extra operand |
|------|--------------------|---------------|
| omitted | Normal target-profile writeback layout | none |
| `nz2nd` | Logical ND order | none |
| `nz2dn` | Logical D/N-swapped order | `loop0_src_stride` in C0-size units |
| `nz2nz` | NZ-style destination | `split`, destination split point |

- `src_stride` is measured in C0-size units.
- `dst_stride` is measured in destination elements.
- In `loop3`, `src_stride3` is in C0-size units and `dst_stride3` is in
  destination elements.

### `pto.mte_l0c_l1` — L0C → L1 (FIXPIPE writeback)

#### `pto.mte_l0c_l1(src: PtrType, dst: PtrType, m: int, n: int, src_stride: int, dst_stride: int, *, unit_flag: Literal["check_only", "check_and_clear"] | None = None, pre_quant: tuple[object, str] | None = None, pre_relu: tuple[str, object | None, object | None] | None = None, layout: object | None = None, loop3: tuple[int, int, int] | None = None, sat: Literal["sat", "sat(preserve_nan)", "nosat"] | None = None) -> None`

**Description**: FIXPIPE writeback from `l0c` to `l1`.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `src` | `pto.ptr<T, l0c>` | Accumulator source in `l0c` |
| `dst` | `pto.ptr<U, l1>` | L1 destination in `l1` |
| `m` | `int` | Logical M element count |
| `n` | `int` | Logical N element count |
| `src_stride` | `int` | Source stride in C0-size units |
| `dst_stride` | `int` | Destination stride in destination elements |
| optional clauses | see above | See FIXPIPE common clauses and layout model |

**Constraints**:
- Clause order is canonical: `unit_flag -> pre_quant -> pre_relu -> layout -> loop3 -> sat`.
- `pre_quant` requires payload and mode together.
- Vector `pre_quant` modes require an `fb` pointer. The payload may use
  `f16`, `bf16`, or `f32` elements, or `ui64` when it is already packed as
  8-byte Quant_PRE parameter entries.
- Scalar `pre_quant` modes require an `f16`, `bf16`, or `f32` scalar payload.
- `pre_quant` source element type must be `f32` or `i32`, and the selected mode
  must be compatible with source and destination element types.
- `scalar_relu` requires an `f16`, `bf16`, or `f32` scalar payload.
- `vector_relu` requires an `fb` pointer with `f16`, `bf16`, or `f32` element
  type.
- `clip` can appear only inside `pre_relu(...)`.
- `clip` is supported for destination `f16`, `ui8`, and signed/signless
  4/8/16-bit integer destinations.
- `nz2dn` requires `loop0_src_stride`; `nz2nd` and `nz2nz` do not accept it.
- `unit_flag` must be omitted when `nz2dn(loop0_src_stride)` uses a value other
  than `1`.
- `nz2nz` requires `f32` destination element type and does not accept `loop3`.
- `sat`, `sat(preserve_nan)`, and `nosat` are mutually exclusive.

**Example**:
```python
pto.mte_l0c_l1(
    l0c.as_ptr(), l1_out.as_ptr(), 16, 32, 16, 32,
    pre_quant=(pto.f32(1.0), "qf322f16_pre_scalar"),
    pre_relu=("scalar_relu", pto.f32(0.25), None),
    layout="nz2nd",
    sat="sat",
)
```

---

### `pto.mte_l0c_gm` — L0C → GM (FIXPIPE writeback)

#### `pto.mte_l0c_gm(src: PtrType, dst: PtrType, m: int, n: int, src_stride: int, dst_stride: int, sid: int, l2_cache_ctrl: int, *, unit_flag: Literal["check_only", "check_and_clear"] | None = None, pre_quant: tuple[object, str] | None = None, pre_relu: tuple[str, object | None, object | None] | None = None, layout: object | None = None, loop3: tuple[int, int, int] | None = None, sat: Literal["sat", "sat(preserve_nan)", "nosat"] | None = None, atomic: tuple[str, str] | None = None) -> None`

**Description**: FIXPIPE writeback from `l0c` to `gm`. The transform clauses
match `pto.mte_l0c_l1`; GM-specific operands select the GM write path and
optional atomic update behavior.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `src`, `m`, `n`, `src_stride` | - | Same as `pto.mte_l0c_l1` |
| `dst` | `pto.ptr<U, gm>` | GM destination |
| `dst_stride` | `int` | GM destination stride in destination elements |
| `sid` | `int` | GM stream/session hint; does not change written values |
| `l2_cache_ctrl` | `int` | GM store cache hint; does not change written values |
| `atomic` | `tuple[str, str]` or `None` | Optional atomic `(type, op)` clause |

`atomic(type=T, op=add|max|min)` performs an atomic read-modify-write at each
GM destination element.

**Constraints**:
- `atomic` is valid only on `pto.mte_l0c_gm`.
- `atomic` requires both `type` and `op`.
- Supported atomic ops are `add`, `max`, and `min`.
- Supported atomic types are `f32`, `f16`, `bf16`, `s32`, `s16`, and `s8`.
- Other constraints match `pto.mte_l0c_l1`.

**Example**:
```python
pto.mte_l0c_gm(
    l0c.as_ptr(), out.as_ptr(), 16, 32, 16, 32, 0, 0,
    pre_quant=(pto.f32(1.0), "qf322f16_pre_scalar"),
    layout="nz2nd",
    atomic=("f16", "add"),
)
```

---

### `pto.mte_l0c_ub` — L0C → UB (FIXPIPE writeback)

#### `pto.mte_l0c_ub(src: PtrType, dst: PtrType, m: int, n: int, src_stride: int, dst_stride: int, dst_mode: object, *, unit_flag: Literal["check_only", "check_and_clear"] | None = None, pre_quant: tuple[object, str] | None = None, pre_relu: tuple[str, object | None, object | None] | None = None, layout: object | None = None, loop3: tuple[int, int, int] | None = None, sat: Literal["sat", "sat(preserve_nan)", "nosat"] | None = None) -> None`

**Description**: FIXPIPE writeback from `l0c` to `ub`. The transform clauses
match `pto.mte_l0c_l1`; UB-specific operands select single or dual destination
behavior.

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `src`, `m`, `n`, `src_stride` | - | Same as `pto.mte_l0c_l1` |
| `dst` | `pto.ptr<U, ub>` | UB destination |
| `dst_stride` | `int` | UB destination stride in destination elements |
| `dst_mode` | `int` or `str` | `sub_blockid`, `split_m`, or `split_n` destination mode |

`dst_mode` forms:

- `dst_mode=sub_blockid` writes the whole logical tile to UB sub-block `0` or
  `1`.
- `dst_mode="split_m"` splits the logical tile into two equal M ranges and
  writes them to UB sub-blocks 0 and 1.
- `dst_mode="split_n"` splits the logical tile into two equal N ranges and
  writes them to UB sub-blocks 0 and 1.

**Constraints**:
- `atomic` is not supported.
- `split_m` requires `m` to be even.
- `split_n` requires `n` to be a multiple of `32`.
- Dual-destination split modes are valid only for target-supported normal or
  `nz2nd` writeback cases with pre-quant, pre-ReLU/clip, and other transform
  clauses omitted.
- Other constraints match `pto.mte_l0c_l1`.

**Example**:
```python
pto.mte_l0c_ub(
    l0c.as_ptr(), ub_out.as_ptr(), 16, 32, 16, 32, 1,
    layout="nz2nd",
)
```

---

## Quick Reference

### By Data Flow

| Data Flow | Operation | Src Space | Dst Space |
|-----------|-----------|-----------|-----------|
| GM → L1 | `pto.mte_gm_l1` | gm | l1 |
| GM → L1 (fractal) | `pto.mte_gm_l1_frac` | gm | l1 |
| L1 → UB | `pto.mte_l1_ub` | l1 | ub |
| L1 → L0A | `pto.mte_l1_l0a` | l1 | l0a |
| L1 → L0B | `pto.mte_l1_l0b` | l1 | l0b |
| L1 → L0A (MX) | `pto.mte_l1_l0a_mx` | l1 | l0a |
| L1 → L0B (MX) | `pto.mte_l1_l0b_mx` | l1 | l0b |
| L1 → Bias | `pto.mte_l1_bt` | l1 | bt |
| L0A×L0B → L0C | `pto.mad` | l0a, l0b | l0c |
| L0A×L0B → L0C (acc) | `pto.mad_acc` | l0a, l0b | l0c |
| L0A×L0B+Bias → L0C | `pto.mad_bias` | l0a, l0b, bt | l0c |
| L1 → FB | `pto.mte_l1_fb` | l1 | fb |
| L0C → L1 | `pto.mte_l0c_l1` | l0c | l1 |
| L0C → GM | `pto.mte_l0c_gm` | l0c | gm |
| L0C → UB | `pto.mte_l0c_ub` | l0c | ub |

### MX Variants

| Base Op | MX Variant | Description |
|---------|------------|-------------|
| `pto.mad` | `pto.mad_mx` | Zero-init MX matmul |
| `pto.mad_acc` | `pto.mad_mx_acc` | Accumulating MX matmul |
| `pto.mad_bias` | `pto.mad_mx_bias` | Bias-init MX matmul |

---

## Template Slot Support

Cube operations support `pto.tpl()` template-slot dispatch, consistent with the
Vector DSL mechanism. See [Template Kernels](04-template-kernels.md) for general
`pto.tpl()` usage.

**Constraints**: Variants within the same slot must have identical parameter
signatures. For example, `mad` and `mad_acc` can share a slot, but `mad_bias`
(which adds a `bias` parameter) requires a separate slot.

---

## See Also

- [Kernel Declaration](03-kernel-declaration.md) — `@pto.ckernel` decorator specification
- [Examples](13-examples.md) — full Cube kernel code examples
- [Design doc](../../../docs/designs/tilelang-cube-dsl-design.md) — Cube DSL design details
