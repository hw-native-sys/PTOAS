# 14. The Virtual Micro-instruction Set (VMI)

The Virtual Micro-instruction Set (VMI) is a logical SIMD vector instruction
set exposed through the `pto.vmi` namespace. It provides a complete set of
virtual micro-instructions for writing vectorized kernels directly against a
hardware-abstracted instruction set — load, store, compute, compare, reduce,
convert, rearrange, and predicate control.

Use `pto.vmi` when you want to:

- author kernels against a stable, logical vector instruction set
- write SIMD code that mirrors the formal VMI specification directly
- carry explicit logical vector and mask types in your authored code
- bypass the top-level PTODSL vector helpers and work one level closer to the
  hardware abstraction

VMI is not a replacement for the existing top-level vector helpers
(`pto.vadd`, `pto.vlds`, etc.). The two surfaces coexist: the top-level helpers
remain the established PTODSL vector programming surface, while `pto.vmi` is
the explicit, instruction-set-oriented alternative.

## 14.1 VMI logical types

VMI introduces two logical type constructors. They describe a logical vector
register and a logical predicate mask at the PTODSL level — the physical
register mapping is handled by the backend.

### `pto.vmi.vreg(lanes, dtype) -> TypeDescriptor`

**Description**: Creates a logical VMI vector register type descriptor.
`lanes` is the logical lane count (not a physical register count). `dtype` is
a PTODSL element type token such as `pto.f32`, `pto.f16`, `pto.i32`, etc.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `lanes` | `int` | Logical lane count. Must be a multiple of 64. See Constraints below |
| `dtype` | `DType` | Element type token (`pto.f32`, `pto.f16`, `pto.i32`, etc.) |

**Returns**:

| Return Value | Type | Description |
|--------------|------|-------------|
| `type_descriptor` | VMI vreg type | Logical VMI vector type descriptor |

**Constraints**:

- `lanes` must be a multiple of 64.
- `lanes · bitwidth(dtype)` determines the physical register count
  `K = ⌈ lanes · bitwidth(dtype) / 2048 ⌉`. Each physical register is
  256 B (2048 bits).
- Common legal combinations:

  | dtype | bitwidth | lanes per physical reg | example `lanes` |
  |-------|----------|------------------------|-----------------|
  | `f32`, `i32`, `ui32`, `si32` | 32 | 64 | 64, 128, 256 |
  | `f16`, `bf16`, `i16`, `ui16`, `si16` | 16 | 128 | 64, 128, 256 |
  | `i8`, `ui8`, `si8`, `fp8_e4m3`, `fp8_e5m2` | 8 | 256 | 64, 128, 256 |

- Compact/partial vectors (`K < 1` in the formula above, e.g. `vreg(64, pto.f16)`
  = 128 B) still occupy one physical register; lanes outside the logical value
  are undefined and must be masked out.

**Example**:

```python
vec_f32 = pto.vmi.vreg(64, pto.f32)    # 1 physical reg (64 × 32b = 256B)
vec_i32 = pto.vmi.vreg(64, pto.i32)    # 1 physical reg
vec_f16 = pto.vmi.vreg(128, pto.f16)   # 1 physical reg (128 × 16b = 256B)
vec_f32_x2 = pto.vmi.vreg(128, pto.f32) # 2 physical regs (128 × 32b = 512B)
```

---

### `pto.vmi.mask(lanes) -> TypeDescriptor`

**Description**: Creates a logical VMI mask type descriptor. The predicate
granularity is always per-lane: one mask bit governs one vector lane.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `lanes` | `int` | Logical lane count. Must match the gated vector's lanes. See Constraints below |

**Returns**:

| Return Value | Type | Description |
|--------------|------|-------------|
| `type_descriptor` | VMI mask type | Logical VMI mask type descriptor |

**Constraints**:

- `lanes` must match the lane count of the vector being gated. A `mask(64)`
  gates a `vreg(64, ...)`; a `mask(128)` gates a `vreg(128, ...)`.

**Example**:

```python
mask64 = pto.vmi.mask(64)    # gates a vreg(64, ...)
mask128 = pto.vmi.mask(128)  # gates a vreg(128, ...)
```

**About VMI layouts.** A VMI logical vector value may span `K` physical vector
registers (256 B each). PTOAS tracks the logical lane layout internally during
lowering; PTODSL does not expose layout selection on `pto.vmi.vreg(...)` or
`pto.vmi.mask(...)`. Two common internal layouts are:

| Layout | Description |
|--------|-------------|
| `contiguous` (default) | Stride-1 mapping: lane `i` sits at position `i mod (2048/bitwidth(T))` within physical register `⌊i / (2048/bitwidth(T))⌋` |
| `deinterleaved` | Parity split: EVEN lanes occupy the first `K/2` physical registers, ODD lanes occupy the second `K/2`. This is the natural output layout of a widening `vcvt` (e.g., `f16 → f32`) or a `vload` with `dist_mode="dintlv"` |

Layouts are a lowering concern managed by `ptoas`. The system
propagates layouts automatically through Category A ops (most elementwise
compute) and inserts materialization at Category C boundaries. In day-to-day
authoring you do not spell layouts explicitly; just declare the logical lane
count and element type, and let PTOAS infer the internal layout.

VMI types are mainly used as type annotations. They
are not Python callables that produce values — use `pto.vmi.vload`,
`pto.vmi.vci`, etc. to produce actual VMI vector values, and
`pto.vmi.create_mask` to produce actual VMI mask
values.

---

## 14.2 Load and store

The load/store family moves data between UB memory and VMI logical vector
registers. These are the primary entry and exit points for VMI vector data.

PTODSL groups the `vload` / `vstore` surface into three mutually exclusive
mode families:

- `dist_mode`: the regular logical memory surface. This covers the default
  contiguous case plus other access patterns selected by `dist_mode`.
- `group`: grouped row-strided load/store.
- `block_stride`: block-strided load/store using paired
  `block_stride` / `repeat_stride` operands.

Pick exactly one family per call. Do not mix `dist_mode`, `group`, and
`block_stride` parameters in the same `vload` / `vstore`.

### `vload`

### `pto.vmi.vload(source, offset, *, size, dist_mode=None, to_dtype=None) -> VRegType`
### `pto.vmi.vload(source, offset, *, size, dist_mode="dintlv") -> (VRegType, VRegType)`
### `pto.vmi.vload(source, offset, *, size, group, stride) -> VRegType`
### `pto.vmi.vload(source, offset, *, size, group, stride, dist_mode="brc") -> VRegType`
### `pto.vmi.vload(source, offset, *, size, block_stride, repeat_stride) -> VRegType`

**Description**: Loads a logical VMI vector from a UB pointer. The element
type is derived from the source pointer; `size` determines the logical lane
count. Which memory access pattern is used depends on the selected mode family.

**Common parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `source` | `PtrType` (ub) | UB source pointer |
| `offset` | `IndexLike` | Element offset into the source buffer |
| `size` | `int` | Logical result lane count |

**About `mask` on `vload`.**

- `pto.vmi.vload(...)` does not take an explicit `mask` parameter. The load
  surface describes how data is read from UB into a logical VMI value, not
  which lanes of a later computation are active.
- Tail handling and partial-lane participation are expressed on the consumer
  side, typically by passing a mask to a later compute op such as
  `pto.vmi.vadd(...)`, or to the final `pto.vmi.vstore(...)`.
- In practice, if you need "load only the active lanes" behavior in authored
  DSL code, write a normal `vload`, then apply your mask on the first consumer
  or on the eventual store.

**Mode 1: `dist_mode`**

Use this family for the normal logical load surface. `dist_mode=None` means the
default contiguous load.

| Parameter | Type | Description |
|-----------|------|-------------|
| `dist_mode` | `str` or `None` | One of `None`, `"continuous"`, `"dintlv"`, `"unpack"`, or `"brc"` |
| `to_dtype` | `DType` or `None` | Required only when `dist_mode="unpack"` |

Dist-mode behavior:

- `None` or `"continuous"`: contiguous stride-1 load, returning one VMI vector.
- `"dintlv"`: deinterleaved load, returning an `(even, odd)` pair.
- `"unpack"`: widening unpack load, returning one widened VMI vector.
- `"brc"`: broadcast load from one source element, returning one VMI vector.

**Mode 2: `group`**

Use this family for grouped row-strided accesses.

| Parameter | Type | Description |
|-----------|------|-------------|
| `group` | `int` | Number of groups |
| `stride` | `IndexLike` | Element stride between groups |

Grouped load behavior:

- `size > group`: full-group load. Each group contributes `size / group`
  elements.
- `size == group`: slot load. Each group contributes one scalar slot.
- `dist_mode="brc"` with `group` switches grouped load into grouped broadcast:
  one source scalar is loaded per group and broadcast within that group.

**Mode 3: `block_stride`**

Use this family for block-strided accesses.

| Parameter | Type | Description |
|-----------|------|-------------|
| `block_stride` | `int` | 16-bit block stride operand |
| `repeat_stride` | `int` | 16-bit repeat stride operand |

**Returns**:

| Return Value | Type | Description |
|--------------|------|-------------|
| `vec` | `VRegType` | Single vector result for continuous, unpack, brc, group, and block-stride modes |
| `(even, odd)` | `(VRegType, VRegType)` | Two-vector result for `dist_mode="dintlv"` |

**Examples**:

Continuous load:

```python
lhs = pto.vmi.vload(src_ptr, offset, size=64)
rhs = pto.vmi.vload(other_ptr, offset, size=64)
```

Dist-mode unpack load:

```python
wide = pto.vmi.vload(
    src_ptr,
    offset,
    size=128,
    dist_mode="unpack",
    to_dtype=pto.i16,
)
```

Broadcast load:

```python
bias = pto.vmi.vload(
    src_ptr,
    offset,
    size=64,
    dist_mode="brc",
)
```

Group-mode load:

```python
tile = pto.vmi.vload(
    src_ptr,
    offset,
    size=64,
    group=8,
    stride=row_stride,
)
```

Grouped broadcast load:

```python
grouped = pto.vmi.vload(
    src_ptr,
    offset,
    size=64,
    group=8,
    stride=row_stride,
    dist_mode="brc",
)
```

Block-stride load:

```python
blocks = pto.vmi.vload(
    src_ptr,
    offset,
    size=64,
    block_stride=pto.i16(8),
    repeat_stride=pto.i16(0),
)
```

**Constraints**:

- `size` is required for every `vload` form.
- `block_stride` mode is mutually exclusive with both `dist_mode` and `group`.
- `group` and `dist_mode` are mutually exclusive except for grouped broadcast,
  spelled as `group=...`, `stride=...`, `dist_mode="brc"`.
- `to_dtype` is only accepted when `dist_mode="unpack"`.
- `stride` is only accepted when `group` is provided.
- `block_stride` and `repeat_stride` must be provided together.
- The unpack form widens by exactly one adjacent bit-width step.

---

### `vstore`

### `pto.vmi.vstore(values, destination, offset, mask=None, *, dist_mode=None, pmode=None) -> None`
### `pto.vmi.vstore((even, odd), destination, offset, mask=None, *, dist_mode="dintlv", pmode=None) -> None`
### `pto.vmi.vstore(values, destination, offset, *, group, stride, pmode=None) -> None`
### `pto.vmi.vstore(values, destination, offset, mask=None, *, block_stride, repeat_stride, pmode=None) -> None`

**Description**: Writes one logical VMI vector, or a deinterleaved pair, back
to a UB pointer. As with `vload`, the PTODSL surface is organized into the same
three mutually exclusive mode families.

**Common parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `values` | `VRegType` or `(VRegType, VRegType)` | One VMI vector for normal forms, or an `(even, odd)` pair for `dist_mode="dintlv"` |
| `destination` | `PtrType` (ub) | UB destination pointer |
| `offset` | `IndexLike` | Element offset into the destination buffer |
| `pmode` | `str` or `None` | Optional inactive-lane mode: `"zero"` stores 0 to masked-off lanes; `"merge"` skips the write for masked-off lanes |

**About `pmode` on `vstore`.**

- `pmode="zero"` is the default store behavior. When a `mask` is present,
  inactive lanes are written as zero.
- `pmode="merge"` preserves destination contents on inactive lanes by skipping
  those writes.
- `pmode` only matters on store forms that actually use a `mask`. Group-mode
  store does not take a mask operand, so there are no inactive lanes to define
  there.

**Mode 1: `dist_mode`**

| Parameter | Type | Description |
|-----------|------|-------------|
| `mask` | VMI mask or `None` | Optional predicate mask gating which lanes are written |
| `dist_mode` | `str` or `None` | One of `None`, `"continuous"`, or `"dintlv"` |

Dist-mode behavior:

- `None` or `"continuous"`: contiguous store of one VMI vector.
- `"dintlv"`: interleaved store of an `(even, odd)` pair.

**Mode 2: `group`**

| Parameter | Type | Description |
|-----------|------|-------------|
| `group` | `int` | Number of groups |
| `stride` | `IndexLike` | Element stride between groups |

Group store writes one grouped logical stream. On the current VMI contract,
group-mode store does not take a mask operand.

**Mode 3: `block_stride`**

| Parameter | Type | Description |
|-----------|------|-------------|
| `mask` | VMI mask or `None` | Optional store mask; omitting it means all lanes are active |
| `block_stride` | `int` | 16-bit block stride operand |
| `repeat_stride` | `int` | 16-bit repeat stride operand |

**Returns**: None (side-effect operation).

**Examples**:

Continuous store:

```python
pto.vmi.vstore(vec, dst_ptr, offset, mask)
```

Group-mode store:

```python
pto.vmi.vstore(
    tile,
    dst_ptr,
    offset,
    group=8,
    stride=row_stride,
)
```

Block-stride store:

```python
pto.vmi.vstore(
    vec,
    dst_ptr,
    offset,
    mask,
    block_stride=pto.i16(8),
    repeat_stride=pto.i16(0),
)
```

**Constraints**:

- `dist_mode`, `group`, and `block_stride` mode selection are mutually
  exclusive.
- `dist_mode="dintlv"` requires `values` to be an `(even, odd)` pair.
- Group mode requires `group` and `stride`, and does not accept `mask`.
- `block_stride` and `repeat_stride` must be provided together.

---

## 14.3 Index generation and broadcast

These instructions produce a new logical vector from a scalar seed — either as
a lane-wise ramp or a uniform broadcast.

### `pto.vmi.vci(base, *, size, order=None) -> VRegType`

**Description**: Builds a logical lane-wise index ramp starting from a scalar
base value. Use it when you need an index vector for lane addressing,
gather/scatter offsets, or dynamic lane selection.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `base` | `ScalarType` | Typed scalar starting value for the ramp |
| `size` | `int` | Logical lane count of the result vector |
| `order` | `str` or `None` | Ramp order: `"ASC"` for ascending (default if omitted), or `"DESC"` for descending |

**Returns**:

| Return Value | Type | Description |
|--------------|------|-------------|
| `idx` | `VRegType` | Lane-wise index vector |

**Example**:

```python
idx = pto.vmi.vci(pto.i32(0), size=64, order="ASC")
out = pto.vmi.vselr(src, idx)
```

**Constraints**:
- `base` must already carry a scalar dtype. A plain Python literal like `0`
  is ambiguous, so use `pto.i32(0)`, `pto.i16(0)`, `pto.f16(0.0)`, or
  `pto.f32(0.0)`.
- `size` determines the logical lane count of the result vector.

---

### `pto.vmi.vbrc(value, *, size) -> VRegType`
### `pto.vmi.vbrc(value, *, size, group) -> VRegType`

**Description**: Broadcasts a scalar value (or a compact group-shaped input)
across all lanes of a logical vector.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `value` | `ScalarType` or `VRegType` | Scalar to broadcast, or a compact VMI vector for grouped broadcast |
| `size` | `int` | Logical lane count of the expanded result |
| `group` | `int` or `None` | Group count for grouped broadcast. When provided, `value` is treated as a compact group-shaped input and expanded accordingly |

**Returns**:

| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Broadcast vector |

**Example** — scalar broadcast:

```python
bias = pto.vmi.vbrc(pto.f32(0.0), size=64)
```

**Example** — grouped broadcast:

```python
expanded = pto.vmi.vbrc(compact, size=256, group=8)
```

**Constraints**:
- `value` must already carry a scalar dtype or be a VMI vector. A plain Python
  literal like `0.0` is ambiguous, so use `pto.f32(0.0)` or `pto.i32(23)`.
- When `group` is provided, `value` must be a VMI vector whose lane count
  matches the group count.

---

## 14.4 Elementwise compute

Elementwise instructions operate lane-by-lane on one or two VMI vector
operands. They form the arithmetic core of VMI SIMD kernels.

### 14.4.1 Binary vector-vector

#### `pto.vmi.vadd(lhs, rhs, mask=None, *, pmode=None) -> VRegType`
#### `pto.vmi.vsub(lhs, rhs, mask=None, *, pmode=None) -> VRegType`
#### `pto.vmi.vmul(lhs, rhs, mask=None, *, pmode=None) -> VRegType`
#### `pto.vmi.vdiv(lhs, rhs, mask=None, *, pmode=None) -> VRegType`
#### `pto.vmi.vmax(lhs, rhs, mask=None, *, pmode=None) -> VRegType`
#### `pto.vmi.vmin(lhs, rhs, mask=None, *, pmode=None) -> VRegType`
#### `pto.vmi.vand(lhs, rhs, mask=None, *, pmode=None) -> VRegType`
#### `pto.vmi.vor(lhs, rhs, mask=None, *, pmode=None) -> VRegType`
#### `pto.vmi.vxor(lhs, rhs, mask=None, *, pmode=None) -> VRegType`
#### `pto.vmi.vshl(lhs, rhs, mask=None, *, pmode=None) -> VRegType`
#### `pto.vmi.vshr(lhs, rhs, mask=None, *, pmode=None) -> VRegType`

**Description**: Element-wise binary operation: `result[i] = lhs[i] <op> rhs[i]`
for lanes where `mask[i]` is true (or all lanes when `mask` is omitted).

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `lhs` | `VRegType` | First operand vector |
| `rhs` | `VRegType` | Second operand vector |
| `mask` | VMI mask or `None` | Optional predicate mask gating lane participation |
| `pmode` | `str` or `None` | Optional predicate mode: `"merge"` keeps predicate-inactive lanes at their prior value; `"zero"` writes 0 |

**Returns**:

| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Result vector (same shape and element type as `lhs`) |

**Example**:

```python
out = pto.vmi.vadd(lhs, rhs, mask)
out = pto.vmi.vmul(scale, data, full_mask)
```

**Constraints**:
- `lhs` and `rhs` must have compatible shapes and element types.
- The result type is inferred from `lhs`.
- For bitwise ops (`vand`, `vor`, `vxor`, `vshl`, `vshr`), integer element
  types are expected. Floating-point usage is rejected.
- `vshr` performs logical right shift for explicit unsigned element types and
  arithmetic right shift for signed or signless element types.

---

### 14.4.2 Unary vector

#### `pto.vmi.vabs(source, mask=None, *, pmode=None) -> VRegType`
#### `pto.vmi.vneg(source, mask=None, *, pmode=None) -> VRegType`
#### `pto.vmi.vrelu(source, mask=None, *, pmode=None) -> VRegType`
#### `pto.vmi.vexp(source, mask=None, *, pmode=None) -> VRegType`
#### `pto.vmi.vln(source, mask=None, *, pmode=None) -> VRegType`
#### `pto.vmi.vsqrt(source, mask=None, *, pmode=None) -> VRegType`
#### `pto.vmi.vnot(source, mask=None, *, pmode=None) -> VRegType`

**Description**: Element-wise unary operation: `result[i] = op(source[i])` for
active lanes. `vrelu` = `max(0, x)`, `vnot` = bitwise NOT (integer types
only).

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `source` | `VRegType` | Input vector |
| `mask` | VMI mask or `None` | Optional predicate mask gating lane participation |
| `pmode` | `str` or `None` | Optional predicate mode: `"merge"` keeps predicate-inactive lanes at their prior value; `"zero"` writes 0 |

**Returns**:

| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Result vector (same shape and element type as `source`) |

**Example**:

```python
activated = pto.vmi.vrelu(src, mask)
inverted = pto.vmi.vnot(int_vec)
```

---

### 14.4.3 Vector-scalar

Formal `pto.vmi` vector-scalar ops in VMI v0.1:

#### `pto.vmi.vadds(source, scalar, mask, *, pmode=None) -> VRegType`
#### `pto.vmi.vmuls(source, scalar, mask, *, pmode=None) -> VRegType`
#### `pto.vmi.vmaxs(source, scalar, mask, *, pmode=None) -> VRegType`
#### `pto.vmi.vmins(source, scalar, mask, *, pmode=None) -> VRegType`
#### `pto.vmi.vshls(source, scalar, mask, *, pmode=None) -> VRegType`
#### `pto.vmi.vshrs(source, scalar, mask, *, pmode=None) -> VRegType`

The following are **PTODSL syntax sugar** — convenience wrappers provided by the
PTODSL authoring layer. They have **no corresponding VMI instruction**; PTODSL lowers
each to an equivalent `pto.vmi.*` form (e.g., `pto.vsubs` lowers to `pto.vmi.vadds` with a
negated scalar). Users may freely use these spellings in PTODSL programs, but tooling and
the VMI v0.1 spec only recognize the formal `pto.vmi.*` ops listed above.

#### `pto.vsubs(source, scalar, mask) -> VRegType`
#### `pto.vands(source, scalar, mask) -> VRegType`
#### `pto.vors(source, scalar, mask) -> VRegType`
#### `pto.vxors(source, scalar, mask) -> VRegType`

**Description**: Element-wise operation with a uniform scalar second operand:
`result[i] = source[i] <op> scalar`. The scalar is broadcast to all active
lanes.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `source` | `VRegType` | Input vector |
| `scalar` | `ScalarType` | Scalar operand (Python number or PTODSL scalar). For `vshls` and `vshrs`, this is a signless `i16` shift amount; other operations coerce it to the vector element type |
| `mask` | VMI mask | **Required.** Predicate mask gating lane participation |
| `pmode` | `str` or `None` | Optional predicate mode: `"merge"` keeps predicate-inactive lanes at their prior value; `"zero"` writes 0 |

**Returns**:

| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Result vector (same shape and element type as `source`) |

**Example**:

```python
scaled = pto.vmi.vmuls(data, 0.5, full_mask)
shifted = pto.vsubs(scores, row_max, col_mask)
```

**Constraints**:
- `mask` is always required for vector-scalar ops — unlike binary and unary
  ops, there is no mask-optional form.
- `vshls` and `vshrs` coerce `scalar` to signless `i16`; its type is
  independent of the source element type. Other vector-scalar ops coerce
  `scalar` to match the element type of `source`.
- `vsubs`, `vands`, `vors`, and `vxors` are PTODSL convenience spellings, not
  dedicated formal `pto.vmi` instructions in VMI v0.1.
- `vdivs` is listed for symmetry with `vdiv`, but it is not currently surfaced
  as a PTODSL API and is not part of the formal VMI v0.1 instruction set.

---

## 14.5 Compare and select

Compare instructions produce logical VMI masks from vector data. Select
instructions consume masks to pick between values lane by lane.

### `pto.vmi.vcmp(lhs, rhs, seed, cmp, *, pmode=None) -> MaskType`

**Description**: Element-wise vector-vector comparison producing a VMI mask:
`result[i] = seed[i] ? (lhs[i] <cmp> rhs[i]) : 0`.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `lhs` | `VRegType` | First operand vector |
| `rhs` | `VRegType` | Second operand vector |
| `seed` | VMI mask | Seed mask gating which lanes participate |
| `cmp` | `str` | Comparison predicate. VMI accepts bare predicates `"eq"`, `"ne"`, `"lt"`, `"le"`, `"gt"`, `"ge"`. Floating-point compares also accept ordered forms `"oeq"`, `"one"`, `"olt"`, `"ole"`, `"ogt"`, `"oge"` |
| `pmode` | `str` or `None` | Optional predicate mode: `"merge"` keeps predicate-inactive lanes at their prior value; `"zero"` writes 0 |
**Returns**:

| Return Value | Type | Description |
|--------------|------|-------------|
| `pred` | VMI mask | Result predicate mask (same granularity and lane count as `seed`) |

---

### `pto.vmi.vcmps(source, scalar, seed, cmp, *, pmode=None) -> MaskType`

**Description**: Vector-scalar comparison: `result[i] = seed[i] ? (source[i] <cmp> scalar) : 0`.
The scalar is broadcast to all lanes.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `source` | `VRegType` | Input vector |
| `scalar` | `ScalarType` | Scalar operand (coerced to the vector element type) |
| `seed` | VMI mask | Seed mask gating lane participation |
| `cmp` | `str` | Comparison predicate. Same accepted spellings as `vcmp` |
| `pmode` | `str` or `None` | Optional predicate mode: `"merge"` keeps predicate-inactive lanes at their prior value; `"zero"` writes 0 |
**Returns**:

| Return Value | Type | Description |
|--------------|------|-------------|
| `pred` | VMI mask | Result predicate mask |

**Example**:

```python
pred = pto.vmi.vcmp(lhs, rhs, seed_mask, "gt")
pred2 = pto.vmi.vcmps(src, 0.0, seed_mask, "ge")
```

---

### `pto.vmi.vsel(mask, true_value, false_value, *, pmode=None) -> VRegType`

**Description**: Per-lane ternary select: `result[i] = mask[i] ? true_value[i] : false_value[i]`.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `mask` | VMI mask | Selection predicate |
| `true_value` | `VRegType` | Value taken when mask is true |
| `false_value` | `VRegType` | Value taken when mask is false |
| `pmode` | `str` or `None` | Optional predicate mode: `"merge"` keeps predicate-inactive lanes at their prior value; `"zero"` writes 0 |
**Returns**:

| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Selected vector |

---

### `pto.vmi.vselr(source, index) -> VRegType`

**Description**: Dynamic per-lane selection from a source vector using an
index vector: `result[i] = source[index[i]]`. This is a gather-style select
within a single vector register.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `source` | `VRegType` | Source vector to select from |
| `index` | `VRegType` | Integer index vector (per-lane source lane indices) |
**Returns**:

| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Selected vector |

**Example**:

```python
out = pto.vmi.vselr(src, idx)
```

**Constraints**:
- The result type is inferred directly from `source`.
- `index` must be an integer-typed VMI vector.

---

## 14.6 Reduction

Reduction instructions collapse a logical vector along its lane dimension,
producing a smaller logical result.

### `pto.vmi.vcadd(source, mask, *, group=None, reassoc, pmode=None) -> VRegType`
### `pto.vmi.vcmax(source, mask, *, group=None, pmode=None) -> VRegType`
### `pto.vmi.vcmin(source, mask, *, group=None, pmode=None) -> VRegType`

**Description**: Full-vector or grouped reduction. `vcadd` computes the sum,
`vcmax` / `vcmin` compute the maximum / minimum with their lane index. When
`group` is omitted (or `None`), the reduction is across the full vector and
the result lane count is 1. When `group` is provided, the vector is
partitioned into that many equal-sized groups and a separate reduction is
performed per group.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `source` | `VRegType` | Input vector |
| `mask` | VMI mask | **Required.** Predicate mask gating lane participation |
| `group` | `int` or `None` | Number of groups for per-group reduction. `None` means full-vector reduction |
| `reassoc` | `bool` | For `vcadd` on floating-point data only: PTODSL requires this keyword to be written explicitly as `True` or `False` |
| `pmode` | `str` or `None` | Optional predicate mode: `"merge"` keeps predicate-inactive lanes at their prior value; `"zero"` writes 0 |

**Returns**:

| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Reduced vector |

**Example** — full-vector reduction:

```python
total = pto.vmi.vcadd(src, mask, reassoc=True)
peak = pto.vmi.vcmax(src, mask)
```

**Example** — grouped reduction:

```python
group_max = pto.vmi.vcmax(
    src,
    mask,
    group=8,
)
```

**Constraints**:
- `mask` is always required.
- PTODSL infers the result type automatically: full-vector reduction returns
  `!pto.vmi.vreg<1xT>`, and grouped reduction returns `!pto.vmi.vreg<GxT>`,
  where `T` is the source element type and `G` is `group`.
- `reassoc` is only meaningful for `vcadd` on floating-point data.
- Floating-point `vcadd` must spell `reassoc` explicitly at the PTODSL surface.
- `reassoc=None` is rejected by PTODSL; use `reassoc=True` or `reassoc=False`.
- The current VMI op encoding remains presence-based, so `reassoc=False`
  lowers to the same no-attribute form as legacy callers.

---

## 14.7 Conversion and reinterpretation

### `pto.vmi.vcvt(source, to_dtype, *, rounding=None, saturate=None, pmode=None) -> VRegType`

**Description**: Numeric type conversion between VMI vector element types.
Converts the element type of `source` to the target element type. PTODSL
infers the result vector type from the source lane count/layout and `to_dtype`.

For int→int widening, the source element type must carry signedness
(e.g. `si8`/`ui8`/`si16`/`ui16`); signless integers are rejected.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `source` | `VRegType` | Input vector (source element type) |
| `to_dtype` | `DType` | Target element type. PTODSL derives the result vector type from the source lane count/layout and this dtype |
| `rounding` | rounding mode or `None` | Optional rounding mode token |
| `saturate` | saturate mode or `None` | Optional saturation mode token |
| `pmode` | `str` or `None` | Optional predicate mode: `"merge"` keeps predicate-inactive lanes at their prior value; `"zero"` writes 0 |

**Returns**:

| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Converted vector |

**Example**:

```python
wide = pto.vmi.vcvt(src_f16, pto.f32)
narrow = pto.vmi.vcvt(src_f32, pto.f16)
```

**Constraints**:
- The masked form of `vcvt` is not currently supported on this surface.
- The source and target dtype pair must be legal for the target backend.
- For `f32 -> f8e4m3/f8e5m2`, PTODSL accepts `rounding="R"`, `"A"`, `"H"`,
  and `"Z"`; other low-level rounding tokens remain rejected on the VMI
  surface.

---

### `pto.vmi.vinterpret_cast(source, to_dtype) -> VRegType`

**Description**: Bitwise reinterpretation of a VMI vector under a different
element type. The logical bit pattern is unchanged; only the element type
annotation changes. This is a reinterpretation, not a numeric conversion.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `source` | `VRegType` | Input vector |
| `to_dtype` | `DType` | **Required.** Target element type. PTODSL preserves the vector's total bit footprint and derives the target lane count |

**Returns**:

| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Reinterpreted vector |

**Example**:

```python
as_i32 = pto.vmi.vinterpret_cast(src, pto.i32)
```

**Constraints**:
- `to_dtype` is always required — PTODSL must not guess a reinterpretation
  target element type.
- The target lane count is `source_lanes * source_element_bits /
  target_element_bits`; this value must be integral.

---

## 14.8 SFU, fused, and indexed memory instructions

This family covers special-function-unit ops, fused multiply-accumulate forms,
and indexed memory access (gather, scatter, histogram). They go beyond simple
elementwise arithmetic.

### `pto.vmi.vexpdif(x, max_value, mask, *, pmode=None) -> VRegType`

**Description**: Computes `exp(x[i] - max_value[i])` for active lanes — the
stable softmax numerator.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | `VRegType` | Input vector |
| `max_value` | `VRegType` | Maximum value vector to subtract before exponentiation |
| `mask` | VMI mask | **Required.** Predicate mask gating lane participation |
| `pmode` | `str` or `None` | Optional predicate mode: `"merge"` keeps predicate-inactive lanes at their prior value; `"zero"` writes 0 |
**Returns**:

| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | `exp(x - max_value)` |

---

### `pto.vmi.vaxpy(x, acc, alpha, mask, *, pmode=None) -> VRegType`

**Description**: Fused multiply-add: `result[i] = alpha * x[i] + acc[i]`.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | `VRegType` | Input vector |
| `acc` | `VRegType` | Accumulator vector |
| `alpha` | `ScalarType` | Scalar multiplier (coerced to the element type of `x`) |
| `mask` | VMI mask | **Required.** Predicate mask |
| `pmode` | `str` or `None` | Optional predicate mode: `"merge"` keeps predicate-inactive lanes at their prior value; `"zero"` writes 0 |
**Returns**:

| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | `alpha * x + acc` |

---

### `pto.vmi.vlrelu(x, slope, mask, *, pmode=None) -> VRegType`

**Description**: Leaky ReLU: `result[i] = x[i] >= 0 ? x[i] : slope * x[i]`.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | `VRegType` | Input vector |
| `slope` | `ScalarType` | Negative-slope multiplier (coerced to the element type of `x`) |
| `mask` | VMI mask | **Required.** Predicate mask |
| `pmode` | `str` or `None` | Optional predicate mode: `"merge"` keeps predicate-inactive lanes at their prior value; `"zero"` writes 0 |
**Returns**:

| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Leaky ReLU result |

---

### `pto.vmi.vprelu(x, alpha, mask, *, pmode=None) -> VRegType`

**Description**: Parametric ReLU with a per-lane vector alpha:
`result[i] = x[i] >= 0 ? x[i] : alpha[i] * x[i]`.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `x` | `VRegType` | Input vector |
| `alpha` | `VRegType` | Per-lane slope vector |
| `mask` | VMI mask | **Required.** Predicate mask |
| `pmode` | `str` or `None` | Optional predicate mode: `"merge"` keeps predicate-inactive lanes at their prior value; `"zero"` writes 0 |
**Returns**:

| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Parametric ReLU result |

---

### `pto.vmi.vmull(a, b, mask, *, pmode=None) -> (VRegType, VRegType)`

**Description**: Widening multiply for 32-bit integer vectors. PTODSL returns
a `(low, high)` pair of 32-bit VMI vectors; `low` carries the lower 32 bits
and `high` carries the upper 32 bits. Signedness follows the inputs.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `a` | `VRegType` | First `i32` or `ui32` operand vector |
| `b` | `VRegType` | Matching second operand vector |
| `mask` | VMI mask | **Required.** Predicate mask |
| `pmode` | `str` or `None` | Optional predicate mode: `"merge"` keeps predicate-inactive lanes at their prior value; `"zero"` writes 0 |

**Returns**:

| Return Value | Type | Description |
|--------------|------|-------------|
| `low` | `VRegType` | Lower 32 bits of the widened product |
| `high` | `VRegType` | Upper 32 bits of the widened product |

**Example**:

```python
low, high = pto.vmi.vmull(a32, b32, mask)
```

**Constraints**:
- `a` and `b` must be identical 32-bit integer VMI vectors.
- `low` and `high` each have the same lane count and signedness as `a`.

### `pto.vmi.vmula(acc, lhs, rhs, mask, *, pmode=None) -> VRegType`

**Description**: Fused multiply-accumulate: `result = acc + (lhs * rhs)`.
The accumulator is both the input and the result value.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `acc` | `VRegType` | Accumulator vector |
| `lhs` | `VRegType` | First multiply operand |
| `rhs` | `VRegType` | Second multiply operand |
| `mask` | VMI mask | **Required.** Predicate mask |
| `pmode` | `str` or `None` | Optional predicate mode: `"merge"` keeps predicate-inactive lanes at their prior value; `"zero"` writes 0 |
**Returns**:

| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Accumulator after the multiply-add |

**Constraints**:
- `acc`, `lhs`, `rhs`, and `result` must have identical VMI vreg types.
- Supported element types are `i8`–`i32`, `f16`, `bf16`, and `f32`.

---

### `pto.vmi.vdhist(acc, source, mask) -> VRegType`

**Description**: Distribution histogram (dhistv2). Counts per-bin
occurrences of `source` values and adds them to the accumulator `acc`,
producing a 256-bin unsigned 16-bit result.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `acc` | `VRegType` | Accumulator vector (256×ui16) |
| `source` | `VRegType` | Source values (N×ui8) |
| `mask` | VMI mask | **Required.** Predicate mask (b8 granularity) |

**Returns**:

| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Distribution histogram counts per bin |

**Constraints**:
- PTODSL infers the result vector type from `acc`; it must be the matching
  256×ui16 histogram accumulator layout.

---

### `pto.vmi.vchist(acc, source, mask) -> VRegType`

**Description**: Cumulative histogram (chistv2 half-axis). Same signature
as `vdhist` but each bin accumulates the sum of counts for all bins ≤ its index
(cumulative / prefix-sum semantics).

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `acc` | `VRegType` | Accumulator vector (256×ui16) |
| `source` | `VRegType` | Source values (N×ui8) |
| `mask` | VMI mask | **Required.** Predicate mask (b8 granularity) |

**Returns**:

| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Cumulative histogram counts per bin |

**Constraints**:
- PTODSL infers the result vector type from `acc`; it must be the matching
  256×ui16 histogram accumulator layout.

---

### `pto.vmi.vgather(source, offsets, mask, *, pmode=None) -> VRegType`

**Description**: Indexed gather from a UB pointer using per-lane element
offsets. Only masked-on lanes participate; masked-off lanes produce an
unspecified value.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `source` | `PtrType` (ub) | UB source pointer |
| `offsets` | `VRegType` | Per-lane element offsets (integer VMI vector) |
| `mask` | VMI mask | **Required.** Predicate mask gating lane participation |
| `pmode` | `str` or `None` | Optional predicate mode: `"merge"` keeps predicate-inactive lanes at their prior value; `"zero"` writes 0 |

**Returns**:

| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Gathered vector |

**Constraints**:
- PTODSL infers the result lane count from `offsets` and the element type
  from `source`.

---

### `pto.vmi.vgatherb(source, offsets, mask, *, pmode=None) -> VRegType`

**Description**: Block gather from a UB pointer. Each participating lane
gathers one 32-byte block using byte-level offsets.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `source` | `PtrType` (ub) | UB source pointer |
| `offsets` | `VRegType` | Per-lane byte offsets (integer VMI vector) |
| `mask` | VMI mask | **Required.** Predicate mask |
| `pmode` | `str` or `None` | Optional predicate mode: `"merge"` keeps predicate-inactive lanes at their prior value; `"zero"` writes 0 |
**Returns**:

| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `VRegType` | Block-gathered vector |

**Constraints**:
- PTODSL infers the result element type from `source`.
- PTODSL infers the result lane count, mask granularity, and layout match from
  `mask`.

---

### `pto.vmi.vscatter(value, destination, offsets, mask, *, pmode=None) -> None`

**Description**: Indexed scatter to a UB pointer. Writes vector lanes to
irregular memory locations using per-lane element offsets.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `value` | `VRegType` | Source vector to scatter |
| `destination` | `PtrType` (ub) | UB destination pointer |
| `offsets` | `VRegType` | Per-lane element offsets (integer VMI vector) |
| `mask` | VMI mask | **Required.** Predicate mask gating lane participation |
| `pmode` | `str` or `None` | Optional predicate mode: `"merge"` keeps predicate-inactive lanes at their prior value; `"zero"` writes 0 |

**Returns**: None (side-effect operation).

**Example**:

```python
g = pto.vmi.vgather(src_ptr, offsets, mask)
pto.vmi.vscatter(value, dst_ptr, offsets, mask)
```

---

## 14.9 Predicate construction

VMI provides one surface API for creating predicate masks:
`create_mask(...)`. It covers both whole-vector prefix masks and grouped
prefix masks.

### `pto.vmi.create_mask(active_lanes, *, size) -> MaskType`
### `pto.vmi.create_mask(active_lanes, *, size, group) -> MaskType`

**Description**: Creates a prefix-style VMI mask where the first
`active_lanes` lanes are active and all remaining lanes are inactive. This is
the primary mask constructor for tail handling and partial-vector scenarios.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `active_lanes` | `IndexLike` | Number of active lanes in the prefix |
| `size` | `int` | Total logical lane count |
| `group` | `int` or `None` | When provided, creates a grouped prefix mask instead of a whole-vector prefix mask. The group size is inferred as `size / group` |

**Returns**:

| Return Value | Type | Description |
|--------------|------|-------------|
| `mask` | VMI mask | Prefix mask with `active_lanes` active lanes, either across the whole logical vector or within each group |

**Example**:

```python
full_mask = pto.vmi.create_mask(64, size=64)
tail_mask = pto.vmi.create_mask(remained, size=64)
group_mask = pto.vmi.create_mask(
    active_per_group,
    size=128,
    group=8,
)
```

**Constraints**:
- When `group` is present, `size` must be divisible by `group`.
- In grouped form, `group_size` is inferred as `size / group`.
- In grouped form, `active_lanes` must be ≤ inferred `group_size`.

---

## 14.10 Data rearrangement

Rearrangement instructions reorganize data between VMI vector registers
without touching memory. They are used to switch between interleaved and
deinterleaved data layouts.

### `pto.vmi.vintlv(lhs, rhs, mask, *, pmode=None) -> (VRegType, VRegType)`

**Description**: Interleave two logical vectors lane-by-lane and return the
result as a pair: `low` contains the interleaved lower half, `high` contains
the upper half.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `lhs` | `VRegType` | First source vector |
| `rhs` | `VRegType` | Second source vector |
| `mask` | VMI mask | **Required.** Predicate mask gating lane participation |
| `pmode` | `str` or `None` | Optional predicate mode: `"merge"` keeps predicate-inactive lanes at their prior value; `"zero"` writes 0 |
**Returns**:

| Return Value | Type | Description |
|--------------|------|-------------|
| `low` | `VRegType` | Interleaved lower half |
| `high` | `VRegType` | Interleaved upper half |

---

### `pto.vmi.vdintlv(lhs, rhs, mask, *, pmode=None) -> (VRegType, VRegType)`

**Description**: Deinterleave a previously interleaved vector pair. This is
the inverse of `vintlv`: it separates even-positioned and odd-positioned lanes
of the logical input stream into two output vectors.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `lhs` | `VRegType` | Lower half of the interleaved input |
| `rhs` | `VRegType` | Upper half of the interleaved input |
| `mask` | VMI mask | **Required.** Predicate mask gating lane participation |
| `pmode` | `str` or `None` | Optional predicate mode: `"merge"` keeps predicate-inactive lanes at their prior value; `"zero"` writes 0 |
**Returns**:

| Return Value | Type | Description |
|--------------|------|-------------|
| `even` | `VRegType` | Lanes from even interleaved positions |
| `odd` | `VRegType` | Lanes from odd interleaved positions |

**Example**:

```python
lo, hi = pto.vmi.vintlv(src, src, mask)
even, odd = pto.vmi.vdintlv(lo, hi, mask)
```

**Constraints**:
- `lhs` and `rhs` must have the same type.
- The two returned vectors form one logical interleaved pair. Preserve their
  order when passing them to subsequent ops.

---

## 14.11 Result typing rules

VMI infers result types when the output shape is unambiguous from the inputs.
Some ops still need a small hint such as `size` or `to_dtype`, but the
surface no longer asks you to spell the full result type manually.

**Ops that infer their result type automatically**:

- Same-shape elementwise binary and unary ops: inferred from the first vector
  operand.
- Vector-scalar ops: inferred from the source vector.
- `vci` / `vbrc`: inferred from the typed scalar or vector input plus `size`
  (and `group` for grouped broadcast).
- `vcmp` / `vcmps`: inferred from the `seed` mask.
- `vsel`: inferred from `true_value`.
- `vcadd`, `vcmax`, `vcmin`: inferred from the source vector and `group`.
- `vmull`: inferred from `a`, returning a `(low, high)` pair with the same 32-bit integer type.
- `vmula`: inferred from `acc`, preserving the accumulator type.
- `vdhist`, `vchist`: inferred from `acc`.
- `vgather`: inferred from the source element type and the `offsets` lanes.
- `vcvt` (when `to_dtype` is provided): inferred from the source lane count
  and target element type.
- `vload` (when `size` is provided): inferred from the source pointer element
  type and `size`.
- `vstore`: no result — side-effect only.
- `vscatter`: no result — side-effect only.
- `vintlv` / `vdintlv`: inferred from the input vector types.
- `vmull`: both low/high result types are inferred from the two matching input
  vector types.

**Ops that infer when given the right hint**:

- `vload` with `dist_mode="unpack"` requires `to_dtype` to derive the widened
  element type.
- `vcvt` requires `to_dtype`.
- `vinterpret_cast` requires `to_dtype`; its lane count is derived by
  preserving the source vector's total bit footprint.
- `vgatherb` is inferred from the source pointer element type and the mask.

---

## 14.12 Relationship to the top-level vector surface

`pto.vmi` and the top-level PTODSL vector helpers (`pto.vadd`, `pto.vlds`,
`pto.vcvt`, etc.) are two distinct authoring surfaces that coexist in PTODSL.

| Aspect | Top-level helpers | `pto.vmi` |
|--------|-------------------|-----------|
| Type system | `VRegType` / `MaskType` | `pto.vmi.vreg(...)` / `pto.vmi.mask(...)` |
| Naming | `pto.vadd`, `pto.vlds` | `pto.vmi.vadd`, `pto.vmi.vload` |
| Style | PTODSL vector programming model | Formal VMI instruction set |
| Predicate creation | `pto.make_mask`, `pto.pset_b32` | `pto.vmi.create_mask` |
| Return model | Varies by op | Consistent: returns new value or `(values, ...)` tuple |

Key differences in practice:

- `pto.vmi.vreg(...)` is distinct from `pto.vreg_type(...)`.
- `pto.vmi.mask(...)` is distinct from `pto.mask_type(...)`.
- `pto.vmi.vadd(...)` is a formal VMI call, not a synonym for `pto.vadd(...)`.
- VMI operations return values rather than writing to destination buffers.

Choose one surface intentionally inside a given sub-kernel or helper region,
and keep the authored style consistent. Mixing both surfaces in the same
region is possible but makes the IR intent harder to follow.

---

## 14.13 Full example: elementwise vector pipeline

The following example shows a complete VMI pipeline: load, compute under mask,
and store back.

```python
from ptodsl import pto

@pto.jit(
    name="vmi_elementwise",
    target="a5",
    backend="vpto",
    mode="explicit",
    kernel_kind="vector",
    insert_sync=False,
)
def vmi_elementwise(
    src_ptr: pto.ptr(pto.f32, "ub"),
    dst_ptr: pto.ptr(pto.f32, "ub"),
    count: pto.i32,
    scale: pto.f32,
):
    full_mask = pto.vmi.create_mask(64, size=64)

    lhs = pto.vmi.vload(src_ptr, 0, size=64)
    rhs = pto.vmi.vload(src_ptr, 64, size=64)

    summed = pto.vmi.vadd(lhs, rhs, full_mask)
    scaled = pto.vmi.vmuls(summed, scale, full_mask)
    activated = pto.vmi.vrelu(scaled, full_mask)

    pto.vmi.vstore(activated, dst_ptr, 0, full_mask)
```

---

## 14.14 VMI instruction quick reference

| Category | Instructions |
|----------|-------------|
| Types | `vreg`, `mask` |
| Load / Store | `vload`, `vstore` |
| Index / Broadcast | `vci`, `vbrc` |
| Binary vector-vector | `vadd`, `vsub`, `vmul`, `vdiv`, `vmax`, `vmin`, `vand`, `vor`, `vxor`, `vshl`, `vshr` |
| Unary vector | `vabs`, `vneg`, `vrelu`, `vexp`, `vln`, `vsqrt`, `vnot` |
| Vector-scalar | formal `pto.vmi`: `vadds`, `vmuls`, `vmaxs`, `vmins`, `vshls`, `vshrs`; DSL convenience: `vsubs`, `vands`, `vors`, `vxors` |
| Compare / Select | `vcmp`, `vcmps`, `vsel`, `vselr` |
| Reduction | `vcadd`, `vcmax`, `vcmin` |
| Conversion | `vcvt`, `vinterpret_cast` |
| SFU / Fused | `vexpdif`, `vaxpy`, `vlrelu`, `vprelu`, `vmull`, `vmula` |
| Histogram | `vchist`, `vdhist` |
| Indexed memory | `vgather`, `vgatherb`, `vscatter` |
| Predicate construction | `create_mask` |
| Data rearrangement | `vintlv`, `vdintlv` |

All formally listed VMI instructions above are members of the `pto.vmi`
namespace. Entries explicitly labeled as DSL convenience spellings are PTODSL
authoring helpers and are not part of the formal `pto.vmi` v0.1 inventory.
