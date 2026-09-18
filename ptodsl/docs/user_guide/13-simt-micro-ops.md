# 13. SIMT Micro-ops

Chapter 3 introduces `@pto.simt` helpers and launch syntax. This chapter covers
the SIMT micro-op API surface used inside those helpers. These wrappers map to
VPTO SIMT operations and operate on PTO scalar values, typed pointers, and
scalar values loaded from tiles.

## 13.1 Launch descriptor

#### `pto.store_vfsimt_info(dim_z, dim_y, dim_x) -> None`

**Description**: Emits the low-level VPTO launch descriptor operation. Most
code should use `body[dim_x, dim_y, dim_z](...)`, `pto.simt_launch(...)`, or
the inline form `with pto.simt(dim_x, dim_y, dim_z):` instead.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `dim_z` | `i32`-compatible value | Launch dimension in Z |
| `dim_y` | `i32`-compatible value | Launch dimension in Y |
| `dim_x` | `i32`-compatible value | Launch dimension in X |

**Returns**: None.

**Example**:

<!-- ptodsl-doc-test: {"mode":"compile","symbol":"simt_ops_store_info_probe","compile":{}} -->
```python
@pto.jit(target="a5")
def simt_ops_store_info_probe():
    dim_z = pto.const(1, dtype=pto.i32)
    dim_y = pto.const(1, dtype=pto.i32)
    dim_x = pto.const(32, dtype=pto.i32)
    pto.store_vfsimt_info(dim_z, dim_y, dim_x)
```

## 13.2 Query ops

#### `pto.get_tid() -> tuple[pto.i32, pto.i32, pto.i32]`
#### `pto.get_tid_x() -> pto.i32`
#### `pto.get_tid_y() -> pto.i32`
#### `pto.get_tid_z() -> pto.i32`

**Description**: Returns the current SIMT work-item coordinate. The grouped
form returns `(x, y, z)` and lowers through the three axis-specific micro-ops.

**Parameters**: None.

**Returns**:

| Return Value | Type | Description |
|--------------|------|-------------|
| `x`, `y`, `z` | `pto.i32` | Work-item coordinates |

---

#### `pto.get_block_dim() -> tuple[pto.i32, pto.i32, pto.i32]`
#### `pto.get_block_dim_x() -> pto.i32`
#### `pto.get_block_dim_y() -> pto.i32`
#### `pto.get_block_dim_z() -> pto.i32`

**Description**: Returns SIMT block dimensions. The grouped form returns
`(x, y, z)`.

**Parameters**: None.

**Returns**: `pto.i32` for axis-specific forms, or a tuple of three `pto.i32`
values for `pto.get_block_dim()`.

---

#### `pto.get_grid_dim() -> tuple[pto.i32, pto.i32, pto.i32]`
#### `pto.get_grid_dim_x() -> pto.i32`
#### `pto.get_grid_dim_y() -> pto.i32`
#### `pto.get_grid_dim_z() -> pto.i32`

**Description**: Returns SIMT grid dimensions. The grouped form returns
`(x, y, z)`.

**Parameters**: None.

**Returns**: `pto.i32` for axis-specific forms, or a tuple of three `pto.i32`
values for `pto.get_grid_dim()`.

---

#### `pto.get_block_idx_x() -> pto.i32`
#### `pto.get_block_idx_y() -> pto.i32`
#### `pto.get_block_idx_z() -> pto.i32`

**Description**: Returns the current SIMT block index in the selected axis.

**Parameters**: None.

**Returns**: `pto.i32`.

---

#### `pto.get_veccoreid() -> pto.i32`
#### `pto.get_clock32() -> pto.i32`
#### `pto.get_clock64() -> pto.i64`
#### `pto.get_laneid() -> pto.i32`
#### `pto.get_lanemask_eq() -> pto.i32`
#### `pto.get_lanemask_le() -> pto.i32`
#### `pto.get_lanemask_lt() -> pto.i32`
#### `pto.get_lanemask_ge() -> pto.i32`
#### `pto.get_lanemask_gt() -> pto.i32`

**Description**: Returns SIMT execution state: vector-core id, clock samples,
lane id, or lane masks derived from the current lane id.

**Parameters**: None.

**Returns**: `pto.get_clock64()` returns `pto.i64`; the other query ops return
`pto.i32`.

**Example**:

<!-- ptodsl-doc-test: {"mode":"compile","symbol":"simt_ops_query_probe","compile":{}} -->
```python
@pto.simt
def capture_query_state(dst: pto.ptr(pto.i32, "gm")):
    tid_x, tid_y, tid_z = pto.get_tid()
    block_x, block_y, block_z = pto.get_block_dim()
    grid_x, grid_y, grid_z = pto.get_grid_dim()
    pto.get_block_idx_x()
    pto.get_block_idx_y()
    pto.get_block_idx_z()
    pto.get_veccoreid()
    pto.get_clock32()
    pto.get_clock64()
    lane = pto.get_laneid()
    pto.get_lanemask_eq()
    pto.get_lanemask_le()
    pto.get_lanemask_lt()
    pto.get_lanemask_ge()
    pto.get_lanemask_gt()
    value = (
        tid_x + tid_y + tid_z +
        block_x + block_y + block_z +
        grid_x + grid_y + grid_z
    )
    pto.stg(value, dst, lane)


@pto.jit(target="a5")
def simt_ops_query_probe(dst: pto.ptr(pto.i32, "gm")):
    capture_query_state[32, 1, 1](dst)
```

## 13.3 Lane collective ops

#### `pto.vote_all(pred: pto.i1) -> pto.i1`
#### `pto.vote_any(pred: pto.i1) -> pto.i1`
#### `pto.vote_uni(pred: pto.i1) -> pto.i1`
#### `pto.vote_ballot(pred: pto.i1) -> pto.i32`

**Description**: Performs a SIMT lane vote over an `i1` predicate.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `pred` | `pto.i1` | Per-lane predicate |

**Returns**: `pto.i1` for `vote_all`, `vote_any`, and `vote_uni`; `pto.i32`
for `vote_ballot`.

**Example**:

<!-- ptodsl-doc-test: {"mode":"compile","symbol":"simt_ops_vote_probe","compile":{}} -->
```python
@pto.simt
def vote_probe(dst: pto.ptr(pto.i32, "gm")):
    lane = pto.get_laneid()
    pred = lane < pto.const(16, dtype=pto.i32)
    ballot = pto.vote_ballot(pred)
    all_pred = pto.vote_all(pred)
    any_pred = pto.vote_any(pred)
    uni_pred = pto.vote_uni(pred)
    value = ballot + all_pred + any_pred + uni_pred
    pto.stg(value, dst, lane)


@pto.jit(target="a5")
def simt_ops_vote_probe(dst: pto.ptr(pto.i32, "gm")):
    vote_probe[32, 1, 1](dst)
```

---

#### `pto.shuffle_idx(value: ScalarType, index: Index, *, width: int = 32) -> ScalarType`
#### `pto.shuffle_up(value: ScalarType, offset: Index, *, width: int = 32) -> ScalarType`
#### `pto.shuffle_down(value: ScalarType, offset: Index, *, width: int = 32) -> ScalarType`
#### `pto.shuffle_bfly(value: ScalarType, mask: Index, *, width: int = 32) -> ScalarType`

**Description**: Reads a scalar payload from another lane. `shuffle_idx` uses an
absolute lane index, `shuffle_up` and `shuffle_down` use relative offsets, and
`shuffle_bfly` uses a butterfly mask.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `value` | PTO scalar | Payload to shuffle |
| `index` / `offset` / `mask` | `i32`-compatible value | Lane selector |
| `width` | Python `int` | Subgroup width, either `16` or `32` |

**Returns**: PTO scalar with the same type as `value`.

**Example**:

<!-- ptodsl-doc-test: {"mode":"compile","symbol":"simt_ops_shuffle_probe","compile":{}} -->
```python
@pto.simt
def shuffle_probe(dst: pto.ptr(pto.i32, "gm")):
    lane = pto.get_laneid()
    shuffled = pto.shuffle_idx(lane, lane, width=32)
    shifted_up = pto.shuffle_up(lane, 1, width=32)
    shifted_down = pto.shuffle_down(lane, 1, width=32)
    butterfly = pto.shuffle_bfly(lane, 1, width=32)
    value = shuffled + shifted_up + shifted_down + butterfly
    pto.stg(value, dst, lane)


@pto.jit(target="a5")
def simt_ops_shuffle_probe(dst: pto.ptr(pto.i32, "gm")):
    shuffle_probe[32, 1, 1](dst)
```

---

#### `pto.redux_add(value: ScalarType) -> ScalarType`
#### `pto.redux_max(value: ScalarType, *, signedness: str | None = None) -> ScalarType`
#### `pto.redux_min(value: ScalarType, *, signedness: str | None = None) -> ScalarType`

**Description**: Reduces a scalar value across SIMT lanes. Integer
`redux_max` and `redux_min` require `signedness="signed"` or
`signedness="unsigned"`. Integer addition is bitwise-identical for signed and
unsigned values, so `redux_add` has no signedness control. Floating-point
reductions do not accept `signedness`.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `value` | PTO scalar | Payload to reduce |
| `signedness` | `"signed"`, `"unsigned"`, or `None` | Required for integer `redux_max` and `redux_min`; omitted for floating point |

**Returns**: PTO scalar with the same type as `value`.

**Example**:

<!-- ptodsl-doc-test: {"mode":"compile","symbol":"simt_ops_collective_probe","compile":{}} -->
```python
@pto.simt
def reduce_lane_value(dst: pto.ptr(pto.i32, "gm")):
    pred = pto.const(1, dtype=pto.i1)
    lane = pto.get_laneid()

    pto.vote_all(pred)
    pto.vote_any(pred)
    pto.vote_uni(pred)
    pto.vote_ballot(pred)

    pto.shuffle_idx(lane, lane, width=32)
    pto.shuffle_up(lane, 1, width=32)
    pto.shuffle_down(lane, 1, width=32)
    value = pto.shuffle_bfly(lane, 1, width=32)
    total = pto.redux_add(value)
    maximum = pto.redux_max(total, signedness="signed")
    minimum = pto.redux_min(maximum, signedness="signed")
    pto.stg(minimum, dst, lane)


@pto.jit(target="a5")
def simt_ops_collective_probe(dst: pto.ptr(pto.i32, "gm")):
    reduce_lane_value[32, 1, 1](dst)
```

## 13.4 Scalar GM memory and atomic ops

#### `pto.ldg(ptr: PtrType, offset: Index = 0, *, l1cache: str = "cache", l2cache: str = "nmfv") -> ScalarType | VectorType`
#### `pto.stg(value: ScalarType | VectorType, ptr: PtrType, offset: Index = 0, *, l1cache: str = "cache", l2cache: str = "nmfv") -> None`

**Description**: Loads or stores one element through a typed pointer with
cache controls.  Supports both scalar and packed vector types.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `ptr` | `pto.ptr(dtype, "gm")` | GM pointer (scalar or vector element type) |
| `value` | PTO scalar or vector | Store payload for `pto.stg` |
| `offset` | index-like value | Element offset (not byte offset) |
| `l1cache` | `"cache"` or `"uncache"` | L1 cache policy |
| `l2cache` | cache token string | L2 cache policy accepted by VPTO |

**Returns**: `pto.ldg` returns the pointer element type. `pto.stg` returns None.

Packed vector pointer element descriptors include `pto.f32x2`, `pto.f16x2`,
`pto.bf16x2`, `pto.f8e4m3x2`, `pto.f8e4m3x4`, `pto.f8e4m3x8`,
`pto.f8e5m2x2`, `pto.f8e5m2x4`, `pto.f8e5m2x8`, `pto.hif8x2`,
`pto.i8x2`, `pto.i16x2`, and `pto.i32x2`.

**Offset contract**: The offset is an element-level ``index``, not a byte
offset. For ``pto.ptr(pto.f32x2, "gm")``, offset 1 advances by 8 bytes
(2 × sizeof(f32)). Dynamic integer offsets can be passed directly; the memory
operation adapts them to the required index type. Use
``pto.cast(value, pto.index)`` only when an explicit reusable index value
is needed.

**Example**:

<!-- ptodsl-doc-test: {"mode":"compile","symbol":"simt_ops_ldg_stg_probe","compile":{}} -->
```python
@pto.simt
def ldg_stg_probe(src: pto.ptr(pto.i32, "gm"), dst: pto.ptr(pto.i32, "gm")):
    lane = pto.get_tid_x()
    idx = lane
    value = pto.ldg(src, idx, l1cache="cache", l2cache="nmfv")
    pto.stg(value, dst, idx, l1cache="uncache", l2cache="wtsred")


@pto.jit(target="a5")
def simt_ops_ldg_stg_probe(
    src: pto.ptr(pto.i32, "gm"),
    dst: pto.ptr(pto.i32, "gm"),
):
    ldg_stg_probe[32, 1, 1](src, dst)
```

**Vector example** (``f32x2`` with dynamic ``i64`` offset):

```python
@pto.simt
def gm_vector_ldst(
    src: pto.ptr(pto.f32x2, "gm"),
    dst: pto.ptr(pto.f32x2, "gm"),
    offset: pto.i64,
):
    idx = offset
    value = pto.ldg(src, idx)
    pto.stg(value, dst, idx)


@pto.jit(target="a5")
def gm_vector_ldst_entry(
    src: pto.ptr(pto.f32x2, "gm"),
    dst: pto.ptr(pto.f32x2, "gm"),
    offset: pto.i64,
):
    gm_vector_ldst[32, 1, 1](src, dst, offset)
```

---

#### `pto.atomic_exch(ptr: PtrType, value: ScalarType, *, l2cache: str = "nmfv") -> ScalarType`
#### `pto.atomic_add(ptr: PtrType, value: ScalarType, *, l2cache: str = "nmfv") -> ScalarType`
#### `pto.atomic_sub(ptr: PtrType, value: ScalarType, *, l2cache: str = "nmfv") -> ScalarType`
#### `pto.atomic_min(ptr: PtrType, value: ScalarType, *, l2cache: str = "nmfv", signedness: str | None = None) -> ScalarType`
#### `pto.atomic_max(ptr: PtrType, value: ScalarType, *, l2cache: str = "nmfv", signedness: str | None = None) -> ScalarType`
#### `pto.atomic_and(ptr: PtrType, value: ScalarType, *, l2cache: str = "nmfv") -> ScalarType`
#### `pto.atomic_or(ptr: PtrType, value: ScalarType, *, l2cache: str = "nmfv") -> ScalarType`
#### `pto.atomic_xor(ptr: PtrType, value: ScalarType, *, l2cache: str = "nmfv") -> ScalarType`
#### `pto.atomic_cas(ptr: PtrType, compare: ScalarType, value: ScalarType, *, l2cache: str = "nmfv") -> ScalarType`

**Description**: Performs a scalar atomic operation and returns the old value.
Only integer `atomic_min` and `atomic_max` use signedness because their ordering
depends on the integer interpretation. Those two operations require
`signedness="signed"` or `signedness="unsigned"` for integer values and omit it
for floating-point values. The other atomic operations have no signedness
parameter.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `ptr` | typed pointer | Atomic target |
| `value` | PTO scalar | Atomic payload |
| `compare` | PTO scalar | Compare value for `atomic_cas` |
| `l2cache` | cache token string | L2 cache policy accepted by VPTO |
| `signedness` | `"signed"`, `"unsigned"`, or `None` | Required only for integer `atomic_min` and `atomic_max` |

**Returns**: Old value loaded from `ptr`.

**Example**:

<!-- ptodsl-doc-test: {"mode":"compile","symbol":"simt_ops_memory_atomic_probe","compile":{}} -->
```python
@pto.simt
def update_counter(counter: pto.ptr(pto.i32, "gm")):
    tid = pto.get_tid_x()
    idx = tid
    value = pto.ldg(counter, idx, l1cache="cache", l2cache="nmfv")
    old = pto.atomic_add(counter, value, l2cache="nmfv")
    pto.atomic_exch(counter, value)
    pto.atomic_sub(counter, value)
    pto.atomic_min(counter, value, signedness="signed")
    pto.atomic_max(counter, value, signedness="signed")
    pto.atomic_and(counter, value)
    pto.atomic_or(counter, value)
    pto.atomic_xor(counter, value)
    pto.atomic_cas(counter, old, value)
    pto.stg(old, counter, idx, l1cache="uncache", l2cache="wtsred")


@pto.jit(target="a5")
def simt_ops_memory_atomic_probe(counter: pto.ptr(pto.i32, "gm")):
    update_counter[32, 1, 1](counter)
```

## 13.5 Scalar math and conversion ops

#### `pto.prmt(lhs: ScalarType, rhs: ScalarType, selector: Index) -> ScalarType`
#### `pto.mulhi(lhs: ScalarType, rhs: ScalarType, *, signedness: str) -> ScalarType`
#### `pto.mul_i32toi64(lhs: ScalarType, rhs: ScalarType, *, signedness: str) -> pto.i64`

**Description**: Performs integer byte permutation or multiplication helper
operations.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `lhs`, `rhs` | integer PTO scalar | Source operands |
| `selector` | `i32`-compatible value | Byte selector for `prmt` |
| `signedness` | `"signed"` or `"unsigned"` | Integer signedness control |

**Returns**: `pto.prmt` and `pto.mulhi` return the source integer type.
`pto.mul_i32toi64` returns `pto.i64`.

**Example**:

<!-- ptodsl-doc-test: {"mode":"compile","symbol":"simt_ops_integer_math_probe","compile":{}} -->
```python
@pto.simt
def integer_math_probe(dst: pto.ptr(pto.i32, "gm")):
    lane = pto.get_laneid()
    permuted = pto.prmt(lane, lane, lane)
    high = pto.mulhi(permuted, lane, signedness="unsigned")
    wide = pto.mul_i32toi64(lane, lane, signedness="unsigned")
    _ = wide
    pto.stg(high, dst, lane)


@pto.jit(target="a5")
def simt_ops_integer_math_probe(dst: pto.ptr(pto.i32, "gm")):
    integer_math_probe[32, 1, 1](dst)
```

---

#### `pto.absf(value: ScalarType) -> ScalarType`

This helper emits the common floating-point `pto.absf` operation documented in
the shared arithmetic chapter; it is not a SIMT-only semantic operation.
#### `pto.sqrt(value: ScalarType) -> ScalarType`
#### `pto.exp(value: ScalarType) -> ScalarType`
#### `pto.log(value: ScalarType) -> ScalarType`
#### `pto.ceil(value: ScalarType) -> ScalarType`
#### `pto.floor(value: ScalarType) -> ScalarType`
#### `pto.rint(value: ScalarType) -> ScalarType`
#### `pto.round(value: ScalarType) -> ScalarType`

**Description**: Performs floating-point value operations available to SIMT
code. `pto.sqrt`, `pto.exp`, and `pto.log` use the generic scalar contract for
ordinary scalar inputs and the packed PTO contract for supported packed values.
`pto.pow` and `pto.fma` are also common value helpers documented in the scalar
chapter; they may be used inside SIMT code but do not imply SIMT execution.
Prefer `pto.abs`, `pto.min`, and `pto.max` for common value operations;
`pto.min` and `pto.max` also support the packed builtin-vector forms. `ceil`,
`floor`, `rint`, and `round` retain their target-specific contracts. Numeric
conversion is authored with `pto.cast`, which emits the category-specific
`pto.ftof`, `pto.ftoi`, `pto.itof`, `pto.exti`, or `pto.trunci` operation.

The common `pto.min` and `pto.max` helpers cover ordinary floating-point
minimum/maximum semantics, including supported packed vector forms. Use
`pto.minimum`/`pto.maximum` when NaN propagation is required; the old
`pto.fmin`/`pto.fmax` names are not part of the unified surface.

### Migration from the pre-unification API

| Former API | Unified API |
|------------|-------------|
| `scalar.load` / `scalar.store` | `pto.load` / `pto.store` |
| `scalar.cast` / `scalar.index_cast` | `pto.cast` |
| `pto.convert(...)` | `pto.cast(..., rounding=..., saturation=...)` |
| `pto.fmin` / `pto.fmax` | `pto.min` / `pto.max` |

**Parameters**: PTO floating-point scalar or packed operands supported by the
selected operation.

**Returns**: PTO scalar with the same type as the input value.

**Example**:

<!-- ptodsl-doc-test: {"mode":"compile","symbol":"simt_ops_float_math_probe","compile":{}} -->
```python
@pto.simt
def float_math_probe(dst: pto.ptr(pto.f32, "gm")):
    lane = pto.get_laneid()
    value = pto.cast(lane, pto.f32, rounding="to_nearest_even")
    root = pto.sqrt(pto.absf(value))
    powered = pto.pow(root, root)
    rounded = pto.round(pto.rint(pto.floor(pto.ceil(powered))))
    bounded = pto.min(pto.max(value, root), rounded)
    accum = pto.fma(bounded, pto.exp(value), pto.log(pto.max(value, root)))
    pto.stg(accum, dst, lane)


@pto.jit(target="a5")
def simt_ops_float_math_probe(dst: pto.ptr(pto.f32, "gm")):
    float_math_probe[32, 1, 1](dst)
```

---

## 13.6 Sync and state ops

#### `pto.syncthreads() -> None`
#### `pto.threadfence() -> None`
#### `pto.threadfence_block() -> None`

**Description**: Emits SIMT synchronization or memory fence operations.

**Parameters**: None.

**Returns**: None.

**Example**:

<!-- ptodsl-doc-test: {"mode":"compile","symbol":"simt_ops_sync_probe","compile":{}} -->
```python
@pto.simt
def sync_probe(dst: pto.ptr(pto.i32, "gm")):
    lane = pto.get_laneid()
    pto.syncthreads()
    pto.threadfence()
    pto.threadfence_block()
    pto.stg(lane, dst, lane)


@pto.jit(target="a5")
def simt_ops_sync_probe(dst: pto.ptr(pto.i32, "gm")):
    sync_probe[32, 1, 1](dst)
```

---

#### `pto.keep(payload: ScalarType, *, slot: int) -> None`
#### `pto.resume(result_type: Type, *, slot: int) -> ScalarType`

**Description**: Preserves and restores a SIMT scalar payload through an
explicit slot. Placement constraints are enforced by the VPTO verifier.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `payload` | PTO scalar | Value to preserve |
| `result_type` | PTO type | Type restored by `resume` |
| `slot` | non-negative Python `int` | State slot |

**Returns**: `pto.keep` returns None. `pto.resume` returns the restored scalar.

**Example**:

<!-- ptodsl-doc-test: {"mode":"compile","symbol":"simt_ops_sync_state_probe","compile":{}} -->
```python
@pto.simt
def save_lane_state():
    pto.keep(pto.get_tid_x(), slot=0)


@pto.simt
def use_lane_state(dst: pto.ptr(pto.i32, "gm")):
    lane = pto.resume(pto.i32, slot=0)
    pto.syncthreads()
    pto.threadfence()
    pto.threadfence_block()
    pto.stg(lane, dst, lane)


@pto.jit(target="a5")
def simt_ops_sync_state_probe(dst: pto.ptr(pto.i32, "gm")):
    save_lane_state[32, 1, 1]()
    use_lane_state[32, 1, 1](dst)
```

## 13.5 SIMT allreduce

`pto.simt_allreduce_sum`, `pto.simt_allreduce_max`, and `pto.simt_allreduce_min` reduce a scalar value across SIMT work-items.

These are Python-level SIMT collective helpers. They are expanded inline into existing PTO IR such as `pto.redux_*`, `pto.shuffle_bfly`, `pto.load`, `pto.store`, `pto.syncthreads`, and structured control flow; they do not lower to dedicated `pto.simt_allreduce_*` ops.

#### `pto.simt_allreduce_sum(value, *, threads, scale=1, thread_offset=0, scratch=None) -> ScalarType`
#### `pto.simt_allreduce_max(value, *, threads, scale=1, thread_offset=0, scratch=None) -> ScalarType`
#### `pto.simt_allreduce_min(value, *, threads, scale=1, thread_offset=0, scratch=None) -> ScalarType`

**Description**: Reduce `value` across `threads` work-items. The result is replicated to every participating work-item.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `value` | PTO scalar | Per-lane value to reduce |
| `threads` | Python `int` | Number of work-items participating in the reduction |
| `scale` | Python `int` | Number of consecutive work-items that share the same reduced result; must divide `threads` |
| `thread_offset` | Python `int` | Offset subtracted from `get_tid_x()` for lane-index computation |
| `scratch` | UB pointer or `None` | Required unless `threads ≤ scale` or `(threads ≤ 32 ∧ pow2(threads) ∧ pow2(scale))` |

**Returns**: PTO scalar with the same type as `value`.
