# 4. Type System and Buffer Management

This chapter covers every type you can use in a PTODSL kernel, plus the operations for allocating, managing, and reinterpreting buffers in global memory (GM) and on-chip Unified Buffer (UB).

## 4.1 Scalar types

### Numeric scalar types

| DSL Type | Description | Bit Width |
|----------|-------------|-----------|
| `pto.i1` | Boolean | 1 |
| `pto.i8` | 8-bit signless integer | 8 |
| `pto.si8` | 8-bit signed integer | 8 |
| `pto.ui8` | 8-bit unsigned integer | 8 |
| `pto.i16` | 16-bit signless integer | 16 |
| `pto.si16` | 16-bit signed integer | 16 |
| `pto.ui16` | 16-bit unsigned integer | 16 |
| `pto.i32` | 32-bit signless integer | 32 |
| `pto.si32` | 32-bit signed integer | 32 |
| `pto.ui32` | 32-bit unsigned integer | 32 |
| `pto.i64` | 64-bit signless integer | 64 |
| `pto.si64` | 64-bit signed integer | 64 |
| `pto.ui64` | 64-bit unsigned integer | 64 |
| `pto.f16` | Half-precision float | 16 |
| `pto.bf16` | Brain float 16 | 16 |
| `pto.f32` | Single-precision float | 32 |

Python literals are typed by the tracer in contexts that accept them: `bool` → `pto.i1`, `int` → context-dependent (typically `pto.i32`, `pto.i64`, or `index`), `float` → a floating-point type. This does not mean every literal is accepted everywhere; for example, float literals are rejected in index-like and integer-only contexts.

For explicit typing, use type constructors:

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"type_system.scalar_expr","symbol":"type_system_scalar_expr_probe","compile":{}} -->
```python
x = pto.i32(1024)
y = pto.ui16(7)
z: pto.i32 = 1024
```

### Low-precision types

The following low-precision types may appear as element types for device storage and vector memory movement: `Tile`, `TensorView`, `PartitionTensorView`, `pto.ptr(...)`, and `pto.vreg_type(...)`. Use them to reduce memory bandwidth; convert to a compute-capable type before arithmetic unless the operation explicitly supports that low-precision format.

Low-precision `VRegType` values are valid intermediate payloads, but they are not generic vector-arithmetic types. In practice, use them on explicit memory/conversion paths such as `vlds`, `vsts`, `vcvt`, `vmulscvt`, and `vpack`, then convert to a compute-capable type before feeding the value to generic vector compute ops.

| DSL Type | Description |
|----------|-------------|
| `pto.hif8` | HiFloat8 format |
| `pto.f4e1m2x2` | 4-bit float (E1M2, 2-wide packed) |
| `pto.f4e2m1x2` | 4-bit float (E2M1, 2-wide packed) |
| `pto.f8e4m3` | 8-bit float (E4M3) |
| `pto.f8e5m2` | 8-bit float (E5M2) |
| `pto.f8e8m0` | 8-bit exponent-only MX scale format |

These types can be used when constructing on-chip tiles, view descriptors, UB pointers, and vector register types:

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"type_system.low_precision_types","symbol":"type_system_low_precision_types_probe","compile":{}} -->
```python
lp_tile = pto.alloc_tile(shape=[128, 64], dtype=pto.f8e4m3)
fp4_tile = pto.alloc_tile(shape=[64, 32], dtype=pto.f4e2m1x2)
lp_ptr = pto.castptr(pto.const(0, dtype=pto.ui64), pto.ptr(pto.f8e4m3, "ub"))
lp_vreg_ty = pto.vreg_type(256, pto.f8e4m3)
```

Constructing scalar eager values or host tensor ABI contracts with a low-precision type is **not supported** — `pto.f8e4m3(1.0)` and `pto.tensor_spec(rank=2, dtype=pto.f8e4m3)` will raise an error.

`pto.f8e8m0` is storage-only. Use it for MX scale storage, including an L1 source pointer passed with the full MX field group to `pto.mte_l1_l0a_mx` or `pto.mte_l1_l0b_mx`; it is not a scalar arithmetic type.

### Integer literal guidance

Prefer plain integer literals. Hex string literals are reserved for explicit bit-pattern authoring:

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"type_system.scalar_expr","symbol":"type_system_scalar_expr_probe","compile":{}} -->
```python
count = pto.i32(1024)
delta = pto.i16(-12)
hi_bit = pto.i32("0x80000000")   # bit-pattern: -2147483648
```

### Floating-point literal forms

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"type_system.scalar_expr","symbol":"type_system_scalar_expr_probe","compile":{}} -->
```python
a = pto.f16(-1.5)
b = pto.f32("inf")
c = pto.f32("-inf")
d = pto.f32("nan")
# Bit-pattern hex
f16_neg_inf = pto.f16("0xFC00")
```

### Stack-local structs

`pto.struct_type` describes a heterogeneous stack-local aggregate. Declare it inside a traced function, then access scalar fields with a static positional path:

```text
pto.struct_type(*field_types) -> StructTypeDescriptor
pto.declare_struct(struct_type) -> Value
pto.struct_get(struct, path) -> ScalarValue
pto.struct_set(struct, path, value) -> None
```

Fields may be `i8` / `i16` / `i32` / `i64` (including signed and unsigned variants), `f16` / `bf16` / `f32`, or another `struct_type`. `path` is an `int`, `tuple[int, ...]`, or `list[int]`; it must be non-empty, static, and end at a scalar field. For example, `(1, 0)` selects field 1, a nested struct, then its field 0. Structs are local handles to stack storage: `struct_set` mutates in place and returns `None`.

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"type_system.struct","symbol":"type_system_struct_probe","compile":{}} -->
```python
state_ty = pto.struct_type(pto.i32, pto.struct_type(pto.f32, pto.i16))
state = pto.declare_struct(state_ty)
pto.struct_set(state, 0, 1)
pto.struct_set(state, (1, 0), 3.5)
count = pto.struct_get(state, [0])
value = pto.struct_get(state, (1, 0))
```

Python `int` literals can initialize integer and floating-point fields; Python `float` literals only initialize floating-point fields and are rounded to the destination type. Existing SSA values must exactly match the destination field type, so `i32 -> i16`, `index -> i32`, and `f32 -> f16` do not insert implicit casts. `bool`, strings, dynamic path values, low-precision storage types, pointer/view/tile handles, and `i1` are rejected.

Structs cannot be used as `@pto.jit`, `@pto.tileop`, or `@pto.simt` parameters, or as `pto.for_(...).carry(...)` state. A struct declared outside an ordinary `pto.for_` may be read and mutated inside that loop. Struct values also cannot be returned, yielded, or passed as function arguments from their declaring scope.

## 4.2 Vector register type

Vector registers hold a fixed 256-byte payload. `pto.vreg(dtype)` infers the element count automatically:

| `dtype` | Result | Elements |
|---------|--------|----------|
| `pto.f32` / `pto.i32` / ... | `vreg<64xT>` | 64 |
| `pto.f16` / `pto.bf16` / `pto.i16` / ... | `vreg<128xT>` | 128 |
| `pto.i8` / `pto.si8` / `pto.ui8` | `vreg<256xT>` | 256 |

Constraint: `element_count × bitwidth(dtype) = 2048`.

Use `pto.elements_per_vreg(dtype)` to query the element count:

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"type_system.scalar_expr","symbol":"type_system_scalar_expr_probe","compile":{}} -->
```python
lanes = pto.elements_per_vreg(pto.f32)  # 64
```

### vbitcast

Reinterpret the bits of a vector register as a different element type:

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"type_system.vreg_bitcast_ptr","symbol":"type_system_vreg_bitcast_ptr_probe","compile":{"BLOCK":128}} -->
```python
fvec = pto.vlds(ptr, offset)            # !pto.vreg<64xf32>
ivec = pto.vbitcast(fvec, pto.i32)      # !pto.vreg<64xi32>
f16_vec = pto.vbitcast(fvec, pto.f16)   # !pto.vreg<128xf16>
```

`vbitcast` preserves the exact bit pattern (type punning). Use `vcvt` for numeric value conversion.

## 4.3 Mask (predicate) types

Masks are typed by bit granularity and must match the vector element width:

| DSL Type | Granularity | Used with |
|----------|-------------|-----------|
| `pto.mask_b8` | 8-bit | `i8`, `si8`, `ui8` |
| `pto.mask_b16` | 16-bit | `f16`, `bf16`, `i16`, `si16`, `ui16` |
| `pto.mask_b32` | 32-bit | `f32`, `i32`, `si32`, `ui32` |

### Constructing masks

Use `make_mask` to generate a mask from a pattern or scalar — it automatically selects the correct bit width from the element dtype:

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"type_system.make_mask","symbol":"type_system_make_mask_probe","compile":{}} -->
```python
active       = pto.make_mask(pto.f16, pto.MaskPattern.ALL)  # pattern-based full mask
tail_mask, _ = pto.make_mask(pto.f32, tail_count) # load mask from tail count scalar
```

The bit-width-specific `pset_b32` and `plt_b32` forms are also available:

```python
active      = pto.pset_b32("PAT_ALL")
one_mask, _ = pto.plt_b32(c1_i32)
```

### Reinterpreting masks

`pbitcast` reinterprets a mask register at a different granularity:

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"type_system.mask_bitcast","symbol":"type_system_mask_bitcast_probe","compile":{}} -->
```python
mask_b16 = pto.pbitcast(mask_b8, pto.mask_b16)
```

## 4.4 Pointer types

Pointers combine an element type and a memory space:

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"type_system.scalar_expr","symbol":"type_system_scalar_expr_probe","compile":{}} -->
```python
ptr_gm  = pto.ptr(pto.f32, pto.MemorySpace.GM)
ptr_ub  = pto.ptr(pto.f16, pto.MemorySpace.UB)
```

### MemorySpace enum

| Enum Value | Description |
|------------|-------------|
| `MemorySpace.GM` | Global Memory (off-chip HBM) |
| `MemorySpace.UB` | Unified Buffer (on-chip scratchpad) |
| `MemorySpace.MAT` | Cube L1 / cbuf staging buffer |
| `MemorySpace.LEFT` | Cube L0A left-operand buffer |
| `MemorySpace.RIGHT` | Cube L0B right-operand buffer |
| `MemorySpace.ACC` | Cube L0C accumulator buffer |
| `MemorySpace.BIAS` | Cube bias table buffer |

## 4.5 TensorView

`TensorView` is a descriptor for a tensor in Global Memory. Create one inside a `@pto.jit` body with `make_tensor_view`:

<!-- ptodsl-doc-test: {"mode":"compile","symbol":"kernel","compile":{"BLOCK":128}} -->
```python
@pto.jit(target="a5")
def kernel(
    A_ptr: pto.ptr(pto.f32, "gm"),
    rows: pto.i32,
    cols: pto.i32,
    *,
    BLOCK: pto.const_expr = 128,
):
    tv = pto.make_tensor_view(A_ptr, shape=[rows, cols], strides=[cols, 1])
    return
```

`make_tensor_view` wraps an explicit GM pointer plus authored metadata. You provide the logical shape and the stride of each dimension in **elements** (not bytes). The resulting `TensorView` can be partitioned for `tile.load`/`tile.store`.

### TensorView attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `shape` | `tuple[int, ...]` | Logical dimensions (up to 5D) |
| `element_type` | `Type` | Element dtype (e.g., `pto.f32`) |
| `strides` | `tuple[int, ...]` | Stride of each dimension, in elements |

Strides support non-contiguous tensors. Pass `strides=A.strides` from the source tensor for the default row-major layout, or supply explicit strides for sub-views. Use `tv.as_ptr()` to obtain a typed GM pointer for use with MTE Ops in explicit-mode orchestration.

## 4.6 PartitionTensorView

`partition_view` creates a sub-view of a TensorView at a given offset and size. It describes *which part* of the GM tensor a `tile.load` or `tile.store` should operate on:

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"type_system.partition_view","symbol":"type_system_partition_view_probe","compile":{"BLOCK":128}} -->
```python
part = pto.partition_view(tv, offsets=[row_offset, 0], sizes=[BLOCK, dim])
```

The result is a `PartitionTensorView` — a lightweight descriptor, not a data buffer. It carries the partition's shape, strides, and element type (inherited from the source TensorView). Use `part.as_ptr()` to obtain a typed GM pointer for MTE Ops in explicit-mode orchestration.

## 4.7 Tile

A `Tile` is an on-chip buffer allocated in UB or cube-local memory. Allocate tiles with `alloc_tile`:

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"type_system.tile_alloc","symbol":"type_system_tile_alloc_probe","compile":{"BLOCK":128,"Br":16,"Bc":16,"dim":16}} -->
```python
# UB tile
a_tile = pto.alloc_tile(shape=[BLOCK, dim], dtype=pto.f32)

# Logical column tile
m_tile = pto.alloc_tile(shape=[Br, 1], dtype=pto.f32, blayout="ColMajor")

# Cube-local scratch with explicit memory space
q_l0a = pto.alloc_tile(shape=[Br, dim], dtype=pto.f16, memory_space=pto.MemorySpace.LEFT)
s_acc = pto.alloc_tile(shape=[Br, Bc], dtype=pto.f32, memory_space=pto.MemorySpace.ACC)
```

`alloc_tile` returns a `Tile` object. The `shape` must be a compile-time constant. The default memory space is UB.
For narrow logical column tiles such as `[Br, 1]`, author them with
`blayout="ColMajor"`. Row-major none-box tiles are validated against a 32-byte
physical row-alignment rule.

For packed types (`pto.f4e1m2x2`, `pto.f4e2m1x2`), `shape` dimensions refer to the number of **packed** elements, each containing 2 f4 values. For example, `alloc_tile(shape=[128, 64], dtype=pto.f4e1m2x2)` allocates a 128×64 tile of packed elements, holding 128×64×2 individual 4-bit floats. The same applies to TensorView shapes when the tensor spec uses a packed dtype.

### Tile attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `shape` | `tuple[int, ...]` | Physical tile dimensions (compile-time constant) |
| `element_type` | `Type` | Element dtype |
| `memory_space` | `MemorySpace` | Where the tile lives (UB, LEFT, RIGHT, ACC, BIAS) |
| `valid_shape` | `tuple[int, ...]` | Logical data region, ≤ `shape` in each dimension |

`valid_shape` is mutable. PTODSL uses this to describe runtime tails without
changing the physical tile allocation. A common pattern is to allocate one
full-size tile per block shape, then update `tile.valid_shape = [...]` before
each sub-kernel or Tile Op call so the live region matches the current tail.

### Tile methods

| Method | Description |
|--------|-------------|
| `tile.fill(value)` | Fill the entire tile with a scalar value |
| `tile.as_ptr()` | Obtain a typed pointer to the tile's base address |

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"type_system.tile_methods","symbol":"type_system_tile_methods_probe","compile":{"Br":16,"Bc":16,"dim":16}} -->
```python
m_prev_tile.fill(float("-inf"))
l_prev_tile.fill(0.0)

rows = q_tile.valid_shape[0]
cols = k_tile.valid_shape[1]
meta_tile.valid_shape = [pto.const(1), pto.const(2)]
tail_tile.valid_shape = [rows]

meta_ptr = meta_tile.as_ptr()
```

When the live region is compile-time known, direct Python integers and
`pto.const(...)` both work. When the live region depends on runtime metadata,
assign PTO scalar values directly with `tile.valid_shape = [rows, cols]`.

## 4.8 Tile Reinterpretation

`pto.tile.reshape` reinterprets a tile buffer with a new shape or layout without any data movement. It is a zero-cost buffer reinterpretation on the Unified Buffer: the underlying bytes are unchanged and PTODSL returns a new tile handle with the requested result type metadata. No destination tile allocation is needed; the result is returned directly.

#### `pto.tile.reshape(src: Tile, *, shape: tuple[int, ...], dtype: DType | None = None, blayout: str | None = None) -> Tile`

**Description**: Returns a reinterpreted tile handle for `src` with the given `shape`, element type, and buffer layout. The total byte size of the new shape must equal the total byte size of `src`. This is a buffer reinterpretation only — no data is copied or moved.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `src` | `Tile` | Source tile to reinterpret |
| `shape` | `tuple[int, ...]` | Target shape. `numel(shape) × elem_size` must equal the total bytes of `src` |
| `dtype` | `DType` or `None` | Target element type (default: `None` — keep the source dtype) |
| `blayout` | `str` or `None` | Target buffer layout, e.g. `"RowMajor"` or `"ColMajor"` (default: `None` — PTODSL authors a row-major result layout) |

**Returns**:

| Return Value | Type | Description |
|--------------|------|-------------|
| `dst` | `Tile` | A tile that reinterprets `src`'s buffer with the new shape, dtype, and layout |

**Constraints**:

- **Total byte size must match**: `numel(shape) × sizeof(dst_elem) == numel(src.shape) × sizeof(src_elem)`.
- **No boxed/non-boxed conversion**: cannot reshape between non-boxed layouts and boxed layouts. The layout family must stay the same.
- **Memory space is preserved**: the returned tile shares the same memory space as `src` (typically UB).
- **Result valid-shape metadata is not remapped from `src`**: the current PTODSL surface does not preserve or infer a reshaped logical `valid_shape`. If later code depends on `tile.valid_shape`, treat the reshape result as carrying only the result type's own valid-dim metadata.
- **Hardware mapping**: executes on the **Vector pipeline** (`PIPE_V`).

**Example** — reshape a 2D tile into 1D to avoid layout constraints during element-wise processing:

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"type_system.tile_reshape","symbol":"type_system_tile_reshape_probe","compile":{"BR":8,"BC":64}} -->
```python
# Allocate source tile (2D, row-major)
tile_2d = pto.alloc_tile(shape=[BR, BC], dtype=pto.f32)

# Reinterpret 2D → 1D (zero-cost, no data movement)
tile_1d = pto.tile.reshape(tile_2d, shape=[BR * BC])
```

**Example** — reinterpret a row-reduced tile `[BR, 1]` as a column-major tile for broadcast loads:

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"type_system.tile_reshape","symbol":"type_system_tile_reshape_probe","compile":{"BR":8,"BC":64}} -->
```python
# Source: row-reduced tile authored with a column-major physical layout
reduce_tile = pto.alloc_tile(shape=[BR, 1], dtype=pto.f32, valid_shape=[BR, 1], blayout="ColMajor")

# Reinterpret as ColMajor layout (same shape, different layout)
reduce_col = pto.tile.reshape(reduce_tile, shape=[BR, 1], blayout="ColMajor")
```

## 4.9 Builtin Vector

An MLIR `vector<size x dtype>` is a rank-1 builtin vector type with `size` elements of `dtype`. It is distinct from a PTO SIMD `vreg`.

PTODSL uses builtin vector values in SIMT scalar code, including contiguous `scalar.load` / `scalar.store` paths and elementwise vector arithmetic. Create a builtin vector type descriptor or initialized vector value with `pto.Vec`:

#### `pto.Vec(dtype, size, *, init=None)`

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `dtype` | PTO dtype | Element type, such as `pto.f32` |
| `size` | Positive Python `int` | Number of elements in the builtin vector value |
| `init` | Scalar value, sequence of scalar values, builtin vector value, or `None` | Optional initializer. A scalar broadcasts to every element; a sequence (tuple/list of scalars) packs one distinct element per entry in input order, with length equal to `size`; a builtin vector value passes through. Each scalar entry is coerced to the element type (Python scalar literal, SSA scalar value, or dynamic runtime scalar value) |

**Returns**:

| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | Builtin vector type descriptor or `pto.Vec(dtype, size)` value | Without `init`, returns a vector type descriptor; with `init`, returns a broadcast or per-element vector value (a scalar initializer broadcasts, a sequence initializer packs elements in input order) |

**Example**:

<!-- ptodsl-doc-pending: {"reason":"illustrative fragment; covered by test_jit_compile scalar contiguous vector probes"} -->
```python
x4 = scalar.load(ptr, offset, contiguous=4)
rstd4 = pto.Vec(pto.f32, 4, init=rstd)
y4 = x4 * rstd4
scalar.store(y4, ptr, offset)
```

Pass `init` a sequence to pack distinct runtime scalars, in order, into one
vector for a single contiguous store:

<!-- ptodsl-doc-pending: {"reason":"illustrative fragment; covered by test_jit_compile per-element Vec probes"} -->
```python
v0 = scalar.load(ptr, index)
v1 = scalar.load(ptr, index + 1)
pair = pto.Vec(pto.f32, 2, init=(v0, v1))
scalar.store(pair, ptr, index)
```

Vectors built with `pto.Vec(..., init=sequence)` or `pto.Vec(..., init=scalar)`
always expose LLVM-compatible element types: signed / unsigned integer dtypes
(`siN` / `uiN`) are built as same-width signless `iN` vectors, keeping the bit
pattern (`pto.Vec(pto.ui32, 2, init=(a, b))` is a `vector<2xi32>`). In contrast,
`scalar.load(..., contiguous=N)` keeps the declared pointer element type (a
`ui32` pointer yields a `vector<2xui32>`). Both store bit-compatibly to
`siN`/`ui32` destinations, but elementwise `VecValue` arithmetic requires equal
element types, so avoid mixing the two origins in one expression.

## 4.10 Explicit scratch buffers

A scratch buffer is a local temporary buffer inside an explicit kernel or helper body. Allocate scratch buffers with `pto.alloc_buffer`. A buffer allocated in the top-level explicit entry body can be used by an inline SIMT scope in the same lexical body, including both `with pto.simt():` and dimensioned `with pto.simt(x, y, z):` scopes. It cannot be used by other SIMT forms. SIMT helpers should allocate their own local scratch with `pto.alloc_buffer`, or receive ordinary typed pointers authored with `pto.castptr` / `pto.addptr`.

```text
pto.alloc_buffer(shape, dtype)
```

<!-- ptodsl-doc-pending: {"reason":"illustrative fragment; covered by test_jit_compile alloc_buffer probes"} -->
```python
scratch = pto.alloc_buffer((32,), pto.f32)
```

| Parameter | Description |
|-----------|-------------|
| `shape` | Static positive integer shape. Pass an `int`, `tuple[int, ...]`, or `list[int]`. |
| `dtype` | Element type of the returned buffer, such as `pto.f32` or `pto.i32`. |

The returned value wraps the buffer address together with its allocation metadata: shape, dtype, element type, element count, and byte size.
