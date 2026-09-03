# 6. Scalar and Pointer Operations

Chapter 5 established the rule: Python constructs are resolved at trace time, PTO constructs produce device-side behavior. This chapter applies that distinction to scalars and pointers — when to use a plain Python number, when to use a top-level `scalar.*` helper, and how to work with typed pointers.

## 6.1 Python scalars vs PTO scalars

A **Python scalar** is any value computed by Python during tracing: a literal (`3.14159`), a constexpr parameter (`BLOCK`), or an arithmetic expression built only from compile-time-known values (`1.0 / sqrt(128)`). These are evaluated at trace time and their results are baked into the device code as constants.

A **PTO scalar** is a value that lives on the device at runtime. It comes from a `scalar.load` read, a device-side computation (`scalar.max`, `scalar.exp`), a runtime query (`pto.get_block_idx()`), or `@pto.jit` tensor metadata such as `A.shape[0]` / `A.strides[1]`. PTO scalars flow through the recorded program and are not resolved until the kernel executes. The helper functions that operate on them live in the top-level `scalar` namespace, not under `pto.*`.

### The mixed expression

In practice, a single expression can mix both kinds:

```python
alpha * o_prev + beta * pv_val
# ^ Python float (trace-time constant, e.g. 1.0 / sqrt(dim))
#        ^ PTO scalar (loaded from tile at runtime)
#                  ^ PTO scalar (loaded from tile at runtime)
```

`alpha` is a Python float computed from compile-time information — it becomes an immediate constant in the device code. `o_prev` and `pv_val` are PTO scalars read from tiles at runtime. The `*` and `+` operators are recorded as device-side multiply-add instructions. The tracer sees the whole expression and produces the appropriate device instructions, embedding the constant operand where possible.

### Rule of thumb

| If the value... | Use... | Example |
|-----------------|--------|---------|
| Is known at compile time | Python scalar | `BLOCK`, `1.0 / sqrt(128)` |
| Comes from device memory | PTO scalar | `scalar.load(tile[r, c])` |
| Depends on a runtime value | PTO scalar | `scalar.max(m_prev, row_max)` |
| Comes from tensor metadata at the `@pto.jit` boundary | PTO scalar | `A.shape[0]`, `Q.strides[2]` |
| Is a block/subblock index | PTO scalar | `pto.get_block_idx()` |

When in doubt, ask: *can this value change between launches of the same compiled kernel?* If yes, it must be a PTO scalar.

## 6.2 Scalar access: load and store

`scalar.load` reads one scalar element from a typed pointer or tile location.
`scalar.store` writes one scalar element back. These are the canonical scalar
memory ops for SIMT authoring. Offsets are counted in elements, not bytes.

#### `scalar.load(ptr: PtrType, offset: Index) -> ScalarType`

**Description**: Loads one scalar element from a typed pointer at the given element offset.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `ptr` | `PtrType` | Typed pointer (`pto.ptr<T, space>`) or the result of `tile.as_ptr()` |
| `offset` | `Index` | Element displacement from `ptr` |

**Returns**:

| Return Value | Type | Description |
|--------------|------|-------------|
| `value` | `ScalarType` | The loaded scalar, matching the pointer's element type |

**Tile-index form** — the preferred syntax when loading from a tile:

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"scalar_ops.tile_access","symbol":"scalar_ops_tile_access_probe","compile":{}} -->
```python
val = scalar.load(tile[row, col])
```

`tile[row, col]` selects one element. Row and column indices are PTO scalars (or Python integers that the tracer promotes). This form is equivalent to computing the pointer and offset from the tile's base address and layout.

**Pointer forms**:

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"scalar_ops.tile_access","symbol":"scalar_ops_tile_access_probe","compile":{}} -->
```python
val = scalar.load(ptr, offset)       # explicit offset
val = scalar.load(ptr + offset)      # pointer arithmetic shorthand
```

---

#### `scalar.store(value: ScalarType, ptr: PtrType, offset: Index) -> None`

**Description**: Stores one scalar element to a typed pointer at the given element offset.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `value` | `ScalarType` | Scalar value to write |
| `ptr` | `PtrType` | Typed destination pointer |
| `offset` | `Index` | Element displacement from `ptr` |

**Returns**: None (side-effect operation).

**Tile-index form**:

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"scalar_ops.tile_access","symbol":"scalar_ops_tile_access_probe","compile":{}} -->
```python
scalar.store(value, tile[row, col])
```

**Pointer forms**:

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"scalar_ops.tile_access","symbol":"scalar_ops_tile_access_probe","compile":{}} -->
```python
scalar.store(value, ptr, offset)
```

### Contiguous vector access

Pass `contiguous=N` to read or write `N` adjacent elements as a single
vector value. `N` must be a positive integer greater than `1`.

#### `scalar.load(ptr: PtrType, offset: Index, *, contiguous: int) -> VecValue`

**Description**: Loads `contiguous` adjacent elements from a typed pointer.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `ptr` | `PtrType` | Typed source pointer |
| `offset` | `Index` | First element to load |
| `contiguous` | Positive Python `int` greater than `1` | Number of adjacent elements to load |

**Returns**:

| Return Value | Type | Description |
|--------------|------|-------------|
| `value` | `pto.Vec(T, size=N)` | Vector value with `N == contiguous` and element type `T` |

**Example**:

```python
x4 = scalar.load(ptr, offset, contiguous=4)
```

---

#### `scalar.store(value: VecValue, ptr: PtrType, offset: Index, *, contiguous: int | None = None) -> None`

**Description**: Stores a vector value to adjacent elements of a typed pointer.
The store width is taken from the vector size. If `contiguous` is
provided, it must match that size.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `value` | `pto.Vec(T, size=N)` | Vector value to write |
| `ptr` | `PtrType` | Typed destination pointer |
| `offset` | `Index` | First element to store |
| `contiguous` | `int` or `None` | Optional width check; when provided, it must equal `N` |

**Example**:

```python
scalar.store(x4, ptr, offset)
scalar.store(x4, ptr, offset, contiguous=4)  # optional width check
```

`scalar.store(scalar_value, ptr, offset, contiguous=N)` is rejected because
scalar values are not implicitly broadcast for vector stores. To build an
explicit broadcast vector, use `pto.Vec(...)`; see Section 4.9.

To write distinct runtime scalars as adjacent elements in one aligned vector
store instead of several scalar stores, pack them with `pto.Vec(...,
init=(...))` before storing; for example:

```python
pair = pto.Vec(pto.f32, 2, init=(v0, v1))
scalar.store(pair, ptr, offset)
```

Each `pto.Vec(..., init=sequence)` entry is coerced to the destination element
type and the whole vector is emitted as a single LLVM store; see Section 4.9.

A vector store accepts a vector whose element type exactly matches the
destination element type, or any integer element type of the same bit width
(signless / signed / unsigned store identical bits). This matches the scalar
coercion rules, so a signless `vector<2xi32>` packed from `si32`/`ui32` scalars
can be stored to a `si32` or `ui32` destination without loss.

### Scalar value adaptation

`scalar.store` adapts the authored `value` to the destination element type.
Use this for normal scalar stores instead of manually materializing constants
with a particular MLIR type.

The adaptation rules are intentionally narrow:

| Destination element type | Accepted values |
|--------------------------|-----------------|
| `index` | Python `int`, runtime `index`, runtime integer |
| Integer types | Python `int`, runtime integer, runtime `index` |
| Floating-point types | Python `int`/`float`, runtime float of the same format or a different width |

Integer and `index` values are converted with `index_cast` where needed.
Integer width changes use the destination type's signedness. Floating-point
width changes use `extf` or `truncf`.

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"scalar_ops.value_adaptation","symbol":"scalar_ops_value_adaptation_probe","compile":{}} -->
```python
int_ptr = int_tile.as_ptr()
row = pto.const(0, dtype=pto.index)
wide_count = pto.const(4, dtype=pto.i64)

scalar.store(row, int_ptr + 0)          # runtime index -> i32 destination
scalar.store(wide_count, int_ptr + 1)   # i64 -> i32 destination
scalar.store(3, int_ptr + 2)            # Python int -> i32 destination

half_value = scalar.load(f16_tile[0, 0])
scalar.store(1.0, f32_tile[0, 0])       # Python float -> f32 destination
scalar.store(half_value, f32_tile[0, 1]) # f16 -> f32 destination
```

The following conversions are not implicit:

- Python `bool` is not accepted as a normal integer or index value.
- A Python `float` literal is rejected for `index` and integer destinations.
- Runtime floating-point values are rejected for `index` and integer
  destinations.
- Runtime `index` and integer values are rejected for floating-point
  destinations.
- `f16` and `bf16` are different formats even though both are 16-bit; PTODSL
  does not silently reinterpret one as the other.

Use an explicit conversion operation when you need a semantic numeric
conversion, or a bitcast operation when you need bit reinterpretation.

---

### Typical SIMT usage

`scalar.load` and `scalar.store` are the primary data access pattern inside `@pto.simt` kernels. Each `load`/`store` operates on one element per work-item, but the SIMT unit executes the same instruction across many work-items in parallel:

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"flash_attention.simt_blend","symbol":"flash_attention_simt_blend_probe","compile":{"BLOCK":8}} -->
```python
@pto.simt
def blend_output_rows(
    o_prev_tile: pto.Tile, pv_tile: pto.Tile,
    alpha_tile: pto.Tile, beta_tile: pto.Tile,
    o_next_tile: pto.Tile,
    row_start: pto.i32, row_stop: pto.i32, valid_dim: pto.i32,
):
    for row in range(row_start, row_stop, 1):
        alpha = scalar.load(alpha_tile[row, 0])
        beta = scalar.load(beta_tile[row, 0])
        for col in range(0, valid_dim, 1):
            o_prev = scalar.load(o_prev_tile[row, col])
            pv_val = scalar.load(pv_tile[row, col])
            o_next = alpha * o_prev + beta * pv_val
            scalar.store(o_next, o_next_tile[row, col])
```

When writing to a raw pointer (e.g., a small metadata buffer obtained via `as_ptr()`), use the pointer-plus-offset form. The following self-contained kernel is the smallest compilable pointer-offset example:

<!-- ptodsl-doc-test: {"mode":"compile","symbol":"scalar_pointer_offset_probe","compile":{}} -->
```python
from ptodsl import pto, scalar


@pto.jit(target="a5")
def scalar_pointer_offset_probe():
    meta_tile = pto.alloc_tile(shape=[1, 8], dtype=pto.i32, valid_shape=[1, 3])
    meta_ptr = meta_tile.as_ptr()

    scalar.store(0, meta_ptr, 0)
    scalar.store(1, meta_ptr, 1)
    scalar.store(2, meta_ptr + 2)

    row_start = scalar.load(meta_ptr, 0)
    row_stop = scalar.load(meta_ptr, 1)
    valid_cols = scalar.load(meta_ptr + 2)

    _ = row_start
    _ = row_stop
    _ = valid_cols
```

## 6.3 Scalar arithmetic and comparisons

### Python operators for basic arithmetic

Addition, subtraction, multiplication, and division of PTO scalars use standard Python syntax. The tracer records the corresponding device-side instructions automatically:

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"scalar_ops.math","symbol":"scalar_ops_math_probe","compile":{}} -->
```python
o_next = alpha * o_prev + beta * pv_val      # multiply-add
l_scaled = l_prev * scalar.exp(m_prev - m_next)  # subtraction inside exp
step = (N + BLOCK - 1) // BLOCK               # Python int arithmetic (trace-time)
```

When both operands are PTO scalars (loaded from device memory or produced by another device-side op), `+`, `-`, `*`, `/` produce device-side arithmetic instructions. When one operand is a Python scalar (trace-time constant), the tracer embeds it as an immediate.

Runtime scalar binary operators materialize Python literals against the other
operand's type. `index` mixed with an integer runtime scalar stays in the
`index` domain. Integer mixed with integer uses the wider integer type. Float
operators require floating-point operands; Python float literals are not
accepted in runtime `index` or integer expressions.

### Bitwise operators

PTO integer scalars support Python bitwise operators `&`, `|`, and `^`. Runtime `index` values, such as loop induction variables produced by `pto.for_` or by AST-rewritten `for range(...)` loops, also support these operators for low-bit masks and parity checks.

The common use case is double-buffering or flag-slot selection:

- `i & 1` selects an alternating slot from a runtime loop index.
- The result of an `index` bitwise expression remains index-like, so it can be passed to APIs that accept runtime index values, such as dynamic synchronization `event_id`.

For fixed-width bit manipulation where the exact integer width matters, cast to an explicit integer type first and keep the expression in that integer domain.

### Math functions: `scalar.*`

Non-trivial scalar math functions live under the top-level `scalar` namespace (imported as `from ptodsl import scalar`). They are intentionally separate from the `pto.*` namespace:

Use the `scalar.*` helpers for device-side runtime math. Python built-ins such
as `max(...)`, `min(...)`, and `abs(...)` run at trace time and are only
correct for plain Python values. When the operands are PTO runtime scalars,
write `scalar.max(a, b)`, `scalar.min(a, b)`, and `scalar.abs(x)` explicitly.

#### `scalar.max(a: ScalarType, b: ScalarType) -> ScalarType`

**Description**: Returns the maximum of two scalars.

#### `scalar.min(a: ScalarType, b: ScalarType) -> ScalarType`

**Description**: Returns the minimum of two scalars.

#### `scalar.exp(x: ScalarType) -> ScalarType`

**Description**: Exponential, e^x.

#### `scalar.log(x: ScalarType) -> ScalarType`

**Description**: Natural logarithm.

#### `scalar.sqrt(x: ScalarType) -> ScalarType`

**Description**: Square root.

#### `scalar.abs(x: ScalarType) -> ScalarType`

**Description**: Absolute value.

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"scalar_ops.math","symbol":"scalar_ops_math_probe","compile":{}} -->
```python
lo = scalar.min(m_prev, row_max)
mag = scalar.abs(m_prev - row_max)
ln = scalar.log(threshold + 1.0)
root = scalar.sqrt(threshold + 4.0)
```

### Comparisons

**Description**: PTO scalars use Python's native comparison operators. The tracer records the corresponding device-side comparison instruction and returns a `pto.i1` result.

| Operator | Predicate (signed) | Predicate (unsigned) | Predicate (float) |
|----------|---------------------|-----------------------|--------------------|
| `>` | `sgt` | `ugt` | `ogt` |
| `<` | `slt` | `ult` | `olt` |
| `==` | `eq` | `eq` | `oeq` |
| `!=` | `ne` | `ne` | `one` |
| `>=` | `sge` | `uge` | `oge` |
| `<=` | `sle` | `ule` | `ole` |

**Example**:

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"scalar_ops.math","symbol":"scalar_ops_math_probe","compile":{}} -->
```python
m_next = scalar.max(m_prev, row_max)
l_scaled = l_prev * scalar.exp(m_prev - m_next)
need_scale = val > threshold       # pto.i1 result
is_zero_mask = val == threshold
in_range = (val >= threshold) & (val <= row_max)
```

For readability in files with many scalar operations, use the top-level `scalar` namespace directly:

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"scalar_ops.math","symbol":"scalar_ops_math_probe","compile":{}} -->
```python
m_next = scalar.max(m_prev, row_max)
l_scaled = l_prev * scalar.exp(m_prev - m_next)
```

These are the scalar-path counterparts of the vector math operations covered in Chapter 8. Use them inside `@pto.simt` kernels and in explicit-mode orchestration code where you need to compute a loop bound or a scalar coefficient from runtime data.

## 6.4 Pointer operations

Typed pointers (Section 4.4) carry both an element type and a memory space. This section covers the operations that create and manipulate them.

### Obtaining pointers: as_ptr()

Tiles and tensor views expose their base address via `as_ptr()`:

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"scalar_ops.pointer_sources","symbol":"scalar_ops_pointer_sources_probe","compile":{"BLOCK":8}} -->
```python
gm_ptr = partition.as_ptr()    # GM pointer from a PartitionTensorView
ub_ptr = tile.as_ptr()         # UB pointer from a Tile
```

`as_ptr()` is the preferred way to get a typed pointer from a high-level descriptor. The result carries the correct element type and memory space from the source.

---

#### `pto.addptr(ptr: PtrType, offset: Index) -> PtrType`

**Description**: Advances a pointer by a number of elements (not bytes).

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `ptr` | `PtrType` | Source pointer |
| `offset` | `Index` | Number of elements to advance |

**Returns**:

| Return Value | Type | Description |
|--------------|------|-------------|
| `new_ptr` | `PtrType` | Pointer advanced by `offset` elements |

**Example**:

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"scalar_ops.pointer_manip","symbol":"scalar_ops_pointer_manip_probe","compile":{}} -->
```python
ptr = pto.addptr(base_ptr, 1024)
```

The `+` shorthand on pointers also counts in elements, not bytes.

Pointer offsets are index-like. They accept Python `int`, runtime `index`, and
runtime integer scalar values. Runtime integer offsets are converted to
`index` before pointer arithmetic. Python `bool`, Python `float`, and runtime
floating-point values are rejected.

---

#### `pto.castptr(address: Index, ptr_type: Type) -> PtrType`

**Description**: Creates a typed pointer from an integer address or reinterprets a pointer as a different type.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `address` | `Index` | Integer address or existing pointer value |
| `ptr_type` | `Type` | Target pointer type, e.g. `pto.ptr(pto.f32, pto.MemorySpace.UB)` |

**Returns**:

| Return Value | Type | Description |
|--------------|------|-------------|
| `ptr` | `PtrType` | Typed pointer value |

This is an advanced operation. Prefer `as_ptr()` when the source already carries type information.

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"scalar_ops.pointer_manip","symbol":"scalar_ops_pointer_manip_probe","compile":{}} -->
```python
ptr = pto.castptr(addr, pto.ptr(pto.i32, pto.MemorySpace.UB))
```

## 6.5 Compile-time queries

These functions return values that are known at trace time from type information or hardware constants.

#### `pto.bytewidth(dtype: Type) -> int`

**Description**: Returns the size in bytes of a single element of the given data type. The result is a Python `int` evaluated at trace time.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `dtype` | `Type` | Data type, e.g. `pto.f32`, `pto.f16`, `pto.i8` |

**Returns**:

| Return Value | Type | Description |
|--------------|------|-------------|
| `size` | `int` | Element size in bytes |

**Example**:

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"scalar_ops.helper_queries","symbol":"scalar_ops_helper_queries_probe","compile":{}} -->
```python
bw = pto.bytewidth(pto.f32)   # 4
bw = pto.bytewidth(pto.f16)   # 2
bw = pto.bytewidth(pto.i8)    # 1
```

---

#### `pto.elements_per_vreg(dtype: Type) -> int`

**Description**: Returns how many elements of `dtype` fit in one 256-byte vector register. The result is a Python `int` evaluated at trace time.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `dtype` | `Type` | Data type, e.g. `pto.f32`, `pto.f16`, `pto.i8` |

**Returns**:

| Return Value | Type | Description |
|--------------|------|-------------|
| `elems` | `int` | Number of elements per vector register |

**Example**:

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"scalar_ops.helper_queries","symbol":"scalar_ops_helper_queries_probe","compile":{}} -->
```python
vec = pto.elements_per_vreg(pto.f32)   # 64
vec = pto.elements_per_vreg(pto.f16)   # 128
vec = pto.elements_per_vreg(pto.i8)    # 256
```

This is the standard stride for chunking column loops in SIMD kernels:

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"scalar_ops.chunk_loop","symbol":"scalar_ops_chunk_loop_probe","compile":{"BLOCK":128}} -->
```python
VEC = pto.elements_per_vreg(pto.f32)
for c in range(0, cols, VEC):
    ...
```

## 6.6 Per-element tile traversal in @pto.simt

`@pto.simt` kernels are the natural home for per-element scalar work. A typical pattern uses nested Python `for range(...)` loops to walk over a tile row by row, column by column; the default AST rewrite lowers them to runtime loops:

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"scalar_ops.simt_scale","symbol":"scalar_ops_simt_scale_probe","compile":{"BLOCK":8}} -->
```python
@pto.simt
def elementwise_scale(
    src_tile: pto.Tile,
    dst_tile: pto.Tile,
    scale: pto.f32,
    rows: pto.i32,
    cols: pto.i32,
):
    for r in range(0, rows, 1):
        for c in range(0, cols, 1):
            val = scalar.load(src_tile[r, c])
            scaled = val * scale
            scalar.store(scaled, dst_tile[r, c])
```

This reads each element from `src_tile`, multiplies by `scale`, and writes to `dst_tile`. The SIMT unit executes the body in parallel across work-items, so this scalar-looking code achieves high throughput — each work-item handles a different `(r, c)` pair.

For operations that need per-row metadata alongside per-element computation, lift the row-level scalar out of the inner loop:

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"scalar_ops.simt_row_coeffs","symbol":"scalar_ops_simt_row_coeffs_probe","compile":{"BLOCK":8}} -->
```python
@pto.simt
def blend_with_per_row_coeffs(
    o_prev_tile: pto.Tile,
    pv_tile: pto.Tile,
    alpha_tile: pto.Tile,    # [rows, 1] — one coefficient per row
    beta_tile: pto.Tile,     # [rows, 1]
    o_next_tile: pto.Tile,
    rows: pto.i32,
    cols: pto.i32,
):
    for r in range(0, rows, 1):
        alpha = scalar.load(alpha_tile[r, 0])   # read once per row
        beta = scalar.load(beta_tile[r, 0])     # read once per row
        for c in range(0, cols, 1):
            o_prev = scalar.load(o_prev_tile[r, c])
            pv_val = scalar.load(pv_tile[r, c])
            o_next = alpha * o_prev + beta * pv_val
            scalar.store(o_next, o_next_tile[r, c])
```

This hoists `alpha` and `beta` out of the inner loop — the row coefficients are loaded once and broadcast across all columns in that row.
