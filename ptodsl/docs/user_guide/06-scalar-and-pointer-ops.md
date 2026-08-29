# 6. Scalar and Pointer Operations

Chapter 5 established the rule: Python constructs are resolved at trace time, PTO constructs produce device-side behavior. This chapter applies that distinction to scalars and pointers — when to use a plain Python number, when to use a `pto.*` scalar helper, and how to work with typed pointers and stack-local buffers.

## 6.1 Python scalars vs PTO scalars

A **Python scalar** is any value computed by Python during tracing: a literal (`3.14159`), a constexpr parameter (`BLOCK`), or an arithmetic expression built only from compile-time-known values (`1.0 / sqrt(128)`). These are evaluated at trace time and their results are baked into the device code as constants.

A **PTO scalar** is a value that lives on the device at runtime. It comes from a `pto.load` read, a device-side computation (`pto.max`, `pto.exp`), a runtime query (`pto.get_block_idx()`), or `@pto.jit` tensor metadata such as `A.shape[0]` / `A.strides[1]`. PTO scalars flow through the recorded program and are not resolved until the kernel executes. All public scalar helpers live in the same `pto.*` namespace as the rest of PTODSL.

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
| Comes from device memory | PTO scalar | `pto.load(tile[r, c])` |
| Depends on a runtime value | PTO scalar | `pto.max(m_prev, row_max)` |
| Comes from tensor metadata at the `@pto.jit` boundary | PTO scalar | `A.shape[0]`, `Q.strides[2]` |
| Is a block/subblock index | PTO scalar | `pto.get_block_idx()` |

When in doubt, ask: *can this value change between launches of the same compiled kernel?* If yes, it must be a PTO scalar.

### Public scalar surface inventory

The public namespace and the frontend seam IR are both organized around PTO
semantics. Common `pto.*` scalar helpers first produce PTO dialect operations;
PTOAS later legalizes their standard semantics to `arith`/`math` after scope
validation.

| Family | Public surface | Detailed section |
|--------|----------------|------------------|
| Constants and scalar types | `pto.const`, `pto.i*`, `pto.si*`, `pto.ui*`, `pto.f*`, `pto.index` | Chapter 4 and this section |
| Arithmetic and comparison | Python `+ - * / // %`, `& | ^`, and comparisons | Section 6.3 |
| Generic scalar helpers | `pto.max`, `pto.min`, `pto.exp`, `pto.log`, `pto.sqrt`, `pto.abs`, `pto.select` | Section 6.3 |
| Conversion | `pto.cast`, `pto.bitcast`, `pto.index_cast` | Section 6.3 |
| Ordinary memory | `pto.load`, `pto.store`, `pto.addptr`, `pto.castptr` | Sections 6.2 and 6.4 |
| Stack-local memory | `pto.alloc_buffer` plus `pto.load/store` | Section 6.2 |
| Target-specific numeric semantics | `pto.ceil`, `pto.floor`, `pto.rint`, `pto.round`, `pto.fma`, packed-value math | Chapter 13 |
| SIMT-dependent scalar operations | queries, vote, shuffle, redux, `ldg/stg`, atomics, permutation, synchronization, state | Chapter 13 |

The historical `scalar` namespace is no longer public. New code and all
examples in this manual use `pto.*`.

## 6.2 Scalar access: load and store

`pto.load` reads one scalar element from a typed pointer, tile location, or
stack-local buffer. `pto.store` writes one scalar element back. These are the
canonical scalar memory operations. Offsets are counted in elements, not bytes.

#### `pto.load(ptr_or_buffer, offset: Index = 0) -> ScalarType`

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
val = pto.load(tile[row, col])
```

`tile[row, col]` selects one element. Row and column indices are PTO scalars (or Python integers that the tracer promotes). This form is equivalent to computing the pointer and offset from the tile's base address and layout.

**Pointer forms**:

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"scalar_ops.tile_access","symbol":"scalar_ops_tile_access_probe","compile":{}} -->
```python
val = pto.load(ptr, offset)       # explicit offset
val = pto.load(ptr + offset)      # pointer arithmetic shorthand
```

---

#### `pto.store(value: ScalarType, ptr_or_buffer, offset: Index = 0) -> None`

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
pto.store(value, tile[row, col])
```

**Pointer forms**:

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"scalar_ops.tile_access","symbol":"scalar_ops_tile_access_probe","compile":{}} -->
```python
pto.store(value, ptr, offset)
```

### Contiguous vector access

Pass `contiguous=N` to read or write `N` adjacent elements as a single
vector value. `N` must be a positive integer greater than `1`.

#### `pto.load(ptr_or_buffer, offset: Index = 0, *, contiguous: int) -> VecValue`

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
x4 = pto.load(ptr, offset, contiguous=4)
```

---

#### `pto.store(value: VecValue, ptr_or_buffer, offset: Index = 0, *, contiguous: int | None = None) -> None`

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
pto.store(x4, ptr, offset)
pto.store(x4, ptr, offset, contiguous=4)  # optional width check
```

`pto.store(scalar_value, ptr, offset, contiguous=N)` is rejected because
scalar values are not implicitly broadcast for vector stores. To build an
explicit broadcast vector, use `pto.Vec(...)`; see Section 4.9.

To write distinct runtime scalars as adjacent elements in one aligned vector
store instead of several scalar stores, pack them with `pto.Vec(...,
init=(...))` before storing; for example:

```python
pair = pto.Vec(pto.f32, 2, init=(v0, v1))
pto.store(pair, ptr, offset)
```

Each `pto.Vec(..., init=sequence)` entry is coerced to the destination element
type and the whole vector is emitted as a single LLVM store; see Section 4.9.

A vector store accepts a vector whose element type exactly matches the
destination element type, or any integer element type of the same bit width
(signless / signed / unsigned store identical bits). This matches the scalar
coercion rules, so a signless `vector<2xi32>` packed from `si32`/`ui32` scalars
can be stored to a `si32` or `ui32` destination without loss.
PTO pointers and local allocated buffers intentionally retain different IR
pointer domains. GM/UB accesses use vector-typed `pto.load/store` so
`!pto.ptr<T, space>` remains available to PTOAS; the authored offset still
names the first scalar element. Local buffers created by `pto.alloc_buffer`
use `llvm.load/store` because their storage is represented by `!llvm.ptr`.
Both paths ultimately lower to LLVM load/store operations.

### Scalar value adaptation

`pto.store` adapts the authored `value` to the destination element type.
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

pto.store(row, int_ptr + 0)          # runtime index -> i32 destination
pto.store(wide_count, int_ptr + 1)   # i64 -> i32 destination
pto.store(3, int_ptr + 2)            # Python int -> i32 destination

half_value = pto.load(f16_tile[0, 0])
pto.store(1.0, f32_tile[0, 0])       # Python float -> f32 destination
pto.store(half_value, f32_tile[0, 1]) # f16 -> f32 destination
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

### Stack-local scalar storage

`pto.alloc_buffer(shape, dtype)` allocates fixed-size storage in the current
explicit kernel or helper body. Every dimension must be a positive Python
integer known while tracing. The result is an address-like value accepted by
the same `pto.load` and `pto.store` APIs used for PTO pointers; no separate
LLVM-facing API is exposed.

#### `pto.alloc_buffer(shape: int | tuple[int, ...], dtype: Type) -> LocalBuffer`

| Parameter | Meaning |
|-----------|---------|
| `shape` | One or more static element extents. Empty, dynamic, boolean, zero, and negative dimensions are invalid. |
| `dtype` | Element type used for allocation size, element addressing, load result type, and store adaptation. |

The buffer is local to the body invocation that creates it. Pointer arithmetic
and explicit offsets are measured in elements. `contiguous=N` reads or writes a
rank-1 builtin vector of `N` adjacent elements.

<!-- ptodsl-doc-test: {"mode":"compile","symbol":"scalar_stack_buffer_probe","compile":{}} -->
```python
from ptodsl import pto


@pto.simt
def scalar_stack_buffer_body():
    scratch = pto.alloc_buffer((8,), pto.f32)
    pto.store(1.5, scratch, 0)
    pto.store(2.5, scratch + 1)
    pair = pto.load(scratch, 0, contiguous=2)
    pto.store(pair, scratch, 4)


@pto.jit(target="a5", mode="explicit")
def scalar_stack_buffer_probe():
    scalar_stack_buffer_body()
```

The generated IR contract is deliberately address-driven:

| Authored address | Observable memory operation |
|------------------|-----------------------------|
| `!pto.ptr<T, space>` or tile element | `pto.load` / `pto.store` in the PTO pointer domain |
| `pto.alloc_buffer(...)` result | `llvm.load` / `llvm.store` over the local stack allocation |

This distinction is internal to generated IR. User code always writes
`pto.load` and `pto.store`.

---

### Typical SIMT usage

`pto.load` and `pto.store` are the primary data access pattern inside `@pto.simt` kernels. Each `load`/`store` operates on one element per work-item, but the SIMT unit executes the same instruction across many work-items in parallel:

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
        alpha = pto.load(alpha_tile[row, 0])
        beta = pto.load(beta_tile[row, 0])
        for col in range(0, valid_dim, 1):
            o_prev = pto.load(o_prev_tile[row, col])
            pv_val = pto.load(pv_tile[row, col])
            o_next = alpha * o_prev + beta * pv_val
            pto.store(o_next, o_next_tile[row, col])
```

When writing to a raw pointer (e.g., a small metadata buffer obtained via `as_ptr()`), use the pointer-plus-offset form. The following self-contained kernel is the smallest compilable pointer-offset example:

<!-- ptodsl-doc-test: {"mode":"compile","symbol":"scalar_pointer_offset_probe","compile":{}} -->
```python
from ptodsl import pto


@pto.jit(target="a5")
def scalar_pointer_offset_probe():
    meta_tile = pto.alloc_tile(shape=[1, 8], dtype=pto.i32, valid_shape=[1, 3])
    meta_ptr = meta_tile.as_ptr()

    pto.store(0, meta_ptr, 0)
    pto.store(1, meta_ptr, 1)
    pto.store(2, meta_ptr + 2)

    row_start = pto.load(meta_ptr, 0)
    row_stop = pto.load(meta_ptr, 1)
    valid_cols = pto.load(meta_ptr + 2)

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
l_scaled = l_prev * pto.exp(m_prev - m_next)  # subtraction inside exp
step = (N + BLOCK - 1) // BLOCK               # Python int arithmetic (trace-time)
```

When both operands are PTO scalars (loaded from device memory or produced by another device-side op), `+`, `-`, `*`, `/` produce device-side arithmetic instructions. When one operand is a Python scalar (trace-time constant), the tracer embeds it as an immediate.

Runtime scalar binary operators materialize Python literals against the other
operand's type. `index` mixed with an integer runtime scalar stays in the
`index` domain. Integer mixed with integer uses the wider integer type. Float
operators require floating-point operands; Python float literals are not
accepted in runtime `index` or integer expressions.

| Surface expression | Floating point | Signed integer / `index` | Unsigned integer |
|--------------------|----------------|---------------------------|------------------|
| `a + b`, `a - b`, `a * b` | add/sub/multiply | add/sub/multiply | add/sub/multiply |
| `a / b` | division | division rounded toward zero | division rounded toward zero |
| `a // b` | not accepted | signed floor division | unsigned division |
| `a % b` | floating-point remainder | signed remainder | unsigned remainder |
| `a & b`, `a | b`, `a ^ b` | not accepted | bitwise operation | bitwise operation |
| `-a` | negation | negation | modular negation |
| `a << b` | not accepted | left shift | left shift |
| `a >> b` | not accepted | arithmetic right shift | logical right shift |

These expressions generate common `pto.*` value operations. They do not imply
a SIMT launch and may be used wherever their operands are valid runtime values.
The same operators apply elementwise to compatible builtin vectors.

### Bitwise operators

PTO integer scalars support Python bitwise operators `&`, `|`, and `^`. Runtime `index` values, such as loop induction variables produced by `pto.for_` or by AST-rewritten `for range(...)` loops, also support these operators for low-bit masks and parity checks.

The common use case is double-buffering or flag-slot selection:

- `i & 1` selects an alternating slot from a runtime loop index.
- The result of an `index` bitwise expression remains index-like, so it can be passed to APIs that accept runtime index values, such as dynamic synchronization `event_id`.

For fixed-width bit manipulation where the exact integer width matters, cast to an explicit integer type first and keep the expression in that integer domain.

### Scalar helper functions: `pto.*`

Non-trivial scalar functions use the same `pto.*` namespace as memory,
pointer, vector, and SIMT operations.

Use the `pto.*` helpers for device-side runtime math. Python built-ins such
as `max(...)`, `min(...)`, and `abs(...)` run at trace time and are only
correct for plain Python values. When the operands are PTO runtime scalars,
write `pto.max(a, b)`, `pto.min(a, b)`, and `pto.abs(x)` explicitly.

#### `pto.add(a, b, *, overflow=None, fastmath=None)`
#### `pto.sub(a, b, *, overflow=None, fastmath=None)`
#### `pto.mul(a, b, *, overflow=None, fastmath=None)`
#### `pto.div(a, b, *, fastmath=None)`
#### `pto.floordiv(a, b)`
#### `pto.ceildiv(a, b)`
#### `pto.rem(a, b, *, fastmath=None)`

**Description**: These are the generic Python arithmetic helpers. PTODSL
selects the category-specific PTO IR operation from the operand type: integer
and index values use the `*i` family, while floating-point values use the `*f`
family. PTODSL derives integer signedness from the authored type. Integer
overflow controls are only valid for integer operations; `fastmath` is only
valid for floating-point operations. `floordiv` and `ceildiv` accept integer
and index operands only.

#### `pto.max(a: ScalarType, b: ScalarType) -> ScalarType`

**Description**: Returns the maximum of two scalars. Floating-point operands
use `maxNum` semantics: when exactly one operand is NaN, the non-NaN
operand is returned. Integer forms emit `pto.maxi` with signedness derived from
the authored type; index operands use signed ordering.

#### `pto.min(a: ScalarType, b: ScalarType) -> ScalarType`

**Description**: Returns the minimum of two scalars. Floating-point operands
use `minNum` semantics, with the same NaN rule as `pto.max`. Integer and index
forms emit `pto.mini` with explicit signed or unsigned semantics. Floating
forms emit `pto.maxf` and `pto.minf`.

#### `pto.exp(x: ScalarType) -> ScalarType`

**Description**: Exponential, e^x.

#### `pto.log(x: ScalarType) -> ScalarType`

**Description**: Natural logarithm.

#### `pto.sqrt(x: ScalarType) -> ScalarType`

**Description**: Square root.

#### `pto.pow(lhs, rhs) -> ScalarType | VecValue`

**Description**: Floating-point power with matching scalar or builtin-vector
operands.

#### `pto.fma(lhs, rhs, acc) -> ScalarType | VecValue`

**Description**: Fused floating-point `lhs * rhs + acc` with one final
rounding. All operands have the same scalar or builtin-vector type.

#### `pto.abs(x: ScalarType) -> ScalarType`

**Description**: Absolute value. Unsigned integer and `index` inputs are
returned unchanged.

#### `pto.ceildiv(a, b) -> ScalarType | VecValue`

**Description**: Integer division rounded toward positive infinity. PTODSL
emits `pto.ceildiv` with `signed` for `si*`/`i*`/`index` and `unsigned` for
`ui*` values.
Fixed-width integer builtin vectors apply the operation elementwise.

#### `pto.neg(x, *, overflow=None, fastmath=None)`, `pto.shl(a, b, *, overflow=None)`, `pto.shr(a, b)`

**Description**: `pto.neg` computes numeric negation and emits `pto.negi` or
`pto.negf` according to the operand category. `pto.shl` shifts a
fixed-width integer left. `pto.shr` records signed arithmetic or unsigned
logical right-shift semantics in an attribute.
Python `-x`, `a << b`, and `a >> b` use the same contracts.

#### `pto.select(cond: i1 | Vec[i1], true_value: T, false_value: T) -> T`

**Description**: Returns `true_value` when `cond` is true, otherwise
`false_value`. Both values must have the same runtime type. When `T` is a
builtin vector, `cond` may be a scalar `i1` selecting the whole vector or a
same-shape builtin vector selecting elementwise.

#### `pto.index_cast(value, *, signedness=None) -> index`
#### `pto.index_cast(dtype, value, *, signedness=None) -> ScalarType`

**Description**: Converts between `index` and fixed-width integer types.
PTODSL derives signedness from the authored integer type and defaults signless
`i*` to signed semantics. Pass `signedness="signed"` or
`signedness="unsigned"` when the signless carrier needs an explicit
interpretation. Matching builtin-vector shapes are supported. The one-argument
form produces `index`; the two-argument form uses the explicit destination
type.

#### `pto.cast(value, dtype, *, rounding=None, saturation=None, overflow=None, fastmath=None) -> ScalarType | VecValue`

**Description**: Performs the ordinary numeric conversion supported by the
source and destination types. Integer width changes preserve the authored
signedness; floating width changes extend or truncate; integer/floating-point
cross-category conversions use the integer source or destination signedness.
The convenience API emits category-specific PTO IR: `pto.exti`, `pto.trunci`,
`pto.ftof`, `pto.ftoi`, or `pto.itof`.
Common PTO IR uses signless integer carriers. PTODSL derives signedness from
authored `si*`/`ui*` values and uses signed semantics for authored `i*` values.
A builtin vector keeps its lane count when `dtype` is a scalar element type.
Floating truncation accepts the standard `rounding` modes. In a SIMT execution
scope, `rounding` additionally accepts `to_odd` (`o`) only for `f32 -> f16`,
and `hybrid` (`h`) only for conversion to `hif8x2`; other SIMT type pairs use
the ISA-supported subset of the standard modes. Integer truncation accepts
explicit `nsw`/`nuw` overflow promises, and floating width conversion accepts
`fastmath`. `saturation="sat"` selects SIMT saturation for conversion forms
that expose a selectable saturation mode, including narrowing float-to-float
and narrowing integer-to-floating conversion. Ordinary generic float-to-integer
conversion follows the corresponding `arith.fptosi/fptoui` non-saturating
semantics; SIMT hardware float-to-integer conversion uses the explicit
`sat`/`nosat` control required by its instruction contract. Some widening or
same-format float conversions, and all float conversions with `f32` destination,
have no selectable saturation mode;
the attribute is accepted for a uniform conversion interface but has no
observable effect for those forms. Packed conversion and explicit SIMT controls
are rejected outside a SIMT execution scope.

#### `pto.bitcast(value, dtype) -> ScalarType | VecValue`

**Description**: Reinterprets a traced numeric value without changing its bit
pattern. Source and destination element widths must match. Builtin vectors also
keep the same shape. Use `pto.cast` when the numeric value should be converted.

#### `pto.cmp(lhs, rhs, predicate, *, fastmath=None) -> i1 | VecValue`

**Description**: Performs a comparison with an explicit predicate. Integer
values use `eq`, `ne`, `lt`, `le`, `gt`, or `ge`; PTODSL derives the required
signedness attribute from the authored integer type. Floating-point values
emit `pto.cmpf` and
use the six ordered short predicates and additionally support `false`, `oeq`,
`ogt`, `oge`, `olt`, `ole`, `one`,
`ord`, `ueq`, `ugt`, `uge`, `ult`, `ule`, `une`, `uno`, and `true`. Python
comparison operators use the authored integer type to select the signedness
attribute and use the ordered short predicates for floating-point values.

#### `pto.max/min(lhs, rhs, *, fastmath=None)`
#### `pto.maximum/minimum(lhs, rhs, *, fastmath=None)`

**Description**: Floating `max/min` use maxNum/minNum semantics and return the
numeric input when exactly one input is NaN. `maximum/minimum` propagate NaN
and order `-0.0` below `+0.0`. Integer `max/min` attach explicit signed or
unsigned semantics to the unified PTO operation.

#### `pto.addui_extended`, `pto.mul_extended`

**Description**: Return two values: sum plus overflow for unsigned addition,
or low plus high halves for full-width multiplication. `pto.mul_extended`
infers or accepts signedness.

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"scalar_ops.math","symbol":"scalar_ops_math_probe","compile":{}} -->
```python
lo = pto.min(m_prev, row_max)
mag = pto.abs(m_prev - row_max)
ln = pto.log(threshold + 1.0)
root = pto.sqrt(threshold + 4.0)
chosen = pto.select(lo < root, lo, root)
widened = pto.cast(chosen, pto.f32)
unordered = pto.cmp(lo, root, "uno")
bits = pto.bitcast(widened, pto.ui32)
```

### Verifiable scalar IR contract

The following table is the expected generated-IR contract for the generic
scalar surface. It is useful for regression tests and does not change how user
code is authored.

| Public surface | Frontend seam IR | Standard legalization |
|----------------|------------------|----------------------|
| Python scalar or builtin-vector arithmetic, division, shifts, bitwise, and comparisons | category-specific `pto.addi/addf`, `subi/subf`, `muli/mulf`, `negi/negf`, `cmpi/cmpf`; plus division, shift, and bitwise families | type-directed standard value operations |
| Scalar and builtin-vector literals | `pto.constant` | standard constants |
| `pto.max/min` | `pto.maxi/mini` or `pto.maxf/minf` | float maxNum/minNum; signed/unsigned integer extrema; index compare/select |
| `pto.exp/log/sqrt` | `pto.exp/log/sqrt` | scalar `math.exp/log/sqrt`; packed forms remain PTO-specific |
| `pto.abs` | `pto.absi` or `pto.absf` | category-specific integer, index, or floating absolute value; index uses signed interpretation |
| `pto.select`, `pto.index_cast`, ordinary `pto.cast`, `pto.bitcast` | corresponding `pto.*` op | standard selection, numeric conversion, and bit reinterpretation |
| `pto.load/store` on a PTO pointer | `pto.load/store` | backend LLVM load/store |
| `pto.load/store` on `pto.alloc_buffer` | `llvm.load/store` | unchanged |

This two-stage contract is intentional: PTODSL does not expose `arith`/`math`
as a second user-facing value dialect, while PTOAS can still use standard
operations for optimization and LLVM lowering.

Operations with target-specific contracts remain explicit PTO operations:
`pto.round`, `pto.rint`, packed-value forms, SIMT collectives, atomics,
queries, and synchronization. Chapter 13 documents those interfaces.

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
m_next = pto.max(m_prev, row_max)
l_scaled = l_prev * pto.exp(m_prev - m_next)
need_scale = val > threshold       # pto.i1 result
is_zero_mask = val == threshold
in_range = (val >= threshold) & (val <= row_max)
```

The scalar helpers remain explicit even in files with many scalar operations:

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"scalar_ops.math","symbol":"scalar_ops_math_probe","compile":{}} -->
```python
m_next = pto.max(m_prev, row_max)
l_scaled = l_prev * pto.exp(m_prev - m_next)
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
            val = pto.load(src_tile[r, c])
            scaled = val * scale
            pto.store(scaled, dst_tile[r, c])
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
        alpha = pto.load(alpha_tile[r, 0])   # read once per row
        beta = pto.load(beta_tile[r, 0])     # read once per row
        for c in range(0, cols, 1):
            o_prev = pto.load(o_prev_tile[r, c])
            pv_val = pto.load(pv_tile[r, c])
            o_next = alpha * o_prev + beta * pv_val
            pto.store(o_next, o_next_tile[r, c])
```

This hoists `alpha` and `beta` out of the inner loop — the row coefficients are loaded once and broadcast across all columns in that row.
