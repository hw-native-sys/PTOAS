# 9. Predicate and Mask Operations

Vector operations on the SIMD unit execute across many lanes in parallel — but not all lanes always hold valid data. The last chunk of a row may be shorter than the hardware vector width; a row-wise reduction may need to skip padding elements. **Predicate masks** are the mechanism that gates which lanes participate in an operation.

This chapter covers mask types, mask creation, logical manipulation, reorganization, and load/store. Comparison operations that *produce* masks from vector data (`vcmp`, `vcmps`) are also covered here, since masks are their primary output.

## 9.1 Mask types

The hardware predicate register is a 256-bit register. PTODSL exposes three typed views of it, differing in how many elements each bit represents:

| Mask type | ALU width | Lanes | Used with vector types |
|-----------|-----------|-------|----------------------|
| `pto.mask_b8` | 8-bit | 256 | `i8` vectors |
| `pto.mask_b16` | 16-bit | 128 | `f16`, `bf16`, `i16` vectors |
| `pto.mask_b32` | 32-bit | 64 | `f32`, `i32` vectors |

A mask and the vector it gates must share the same granularity: a `mask_b32` gates an `f32` vector (64 lanes), not an `f16` vector (128 lanes).

**Zeroing predication**: when a lane is masked off, the operation produces zero in that lane. This is the gating model for all vector compute ops in Chapter 8.

## 9.2 Mask creation: `pto.make_mask`

The recommended front door for creating masks is `pto.make_mask`. It dispatches to the right underlying op based on its arguments.

#### `pto.make_mask(dtype: Type, value: int-like | MaskPattern) -> MaskType | (MaskType, int-like)`

**Description**: Creates a predicate mask of the granularity matching `dtype`. When `value` is an integer-like scalar (typically a remaining-element count in a chunked loop), returns a tuple `(mask, remaining)`. When `value` is a `MaskPattern`, returns just the mask.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `dtype` | `Type` | Element type to infer mask granularity from (e.g., `pto.f32` → `mask_b32`, `pto.f16` → `mask_b16`) |
| `value` | `int-like` or `MaskPattern` | Either a remaining-element count or a pattern token |

**Returns**:

| Return Value | Type | Description |
|--------------|------|-------------|
| `mask` | `MaskType` | The created mask |
| `remained` | `int-like` | Updated remaining count (only when `value` is an integer-like scalar); its scalar kind is preserved, so an `index` remainder stays an `index` |

**Example** — chunked SIMD loop with tail handling:

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"tail.chunked_inner_loop","symbol":"tail_chunked_inner_loop_probe","compile":{"BLOCK":128}} -->
```python
VEC = pto.elements_per_vreg(pto.f32)
remained = cols
for c in range(0, cols, VEC):
    mask, remained = pto.make_mask(pto.f32, remained)
    vec = pto.vlds(tile[r, c:])
    # ... operate under mask ...
    pto.vsts(vec, out_tile[r, c:], mask)
```

`make_mask` generates a tail mask from the remaining count: the first `min(remained, VL)` lanes are active, and `remained` is decremented by `VL` for the next iteration. On the final partial chunk, fewer than `VL` lanes are active. PTODSL handles the hardware `i32` tail-mask operand internally, so loop-carried `index` metadata can flow through `make_mask` without manual casts.

---

When the mask pattern is known at compile time, pass a `MaskPattern` instead:

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"mask_ops.creation","symbol":"mask_ops_creation_probe","compile":{}} -->
```python
full_mask = pto.make_mask(pto.f32, pto.MaskPattern.ALL)
```

This is equivalent to calling the granularity-specific ops described below.

---

## 9.3 Granularity-specific creation ops

When you need explicit control over the mask granularity, use these ops directly.

### 9.3.1 Pattern-based: `pset_b*` and `pge_b*`

`pset` generates a mask from a named pattern. `pge` generates a tail mask where the first N lanes are active (N encoded in the pattern).

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"mask_ops.creation","symbol":"mask_ops_creation_probe","compile":{}} -->
```python
full_mask = pto.pset_b32(pto.MaskPattern.ALL)
```

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"mask_ops.creation","symbol":"mask_ops_creation_probe","compile":{}} -->
```python
mask8 = pto.pset_b8(pto.MaskPattern.ALL)
mask16 = pto.pset_b16(pto.MaskPattern.ALL)
```

#### `pto.pset_b8(pattern: MaskPattern) -> pto.mask_b8`
#### `pto.pset_b16(pattern: MaskPattern) -> pto.mask_b16`
#### `pto.pset_b32(pattern: MaskPattern) -> pto.mask_b32`

**Description**: Creates a mask from a pattern token. `PAT_ALL` sets all lanes active; other patterns set a subset.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `pattern` | `MaskPattern` | Pattern token: `ALL`, `ALLF`, `H`, `Q`, `VL1`–`VL128`, `M3`, `M4` |

**Returns**:

| Return Value | Type | Description |
|--------------|------|-------------|
| `mask` | `MaskType` | Mask with lanes set per the pattern |

---

#### `pto.pge_b8(pattern: MaskPattern) -> pto.mask_b8`
#### `pto.pge_b16(pattern: MaskPattern) -> pto.mask_b16`
#### `pto.pge_b32(pattern: MaskPattern) -> pto.mask_b32`

**Description**: Tail mask — `mask[i] = (i < N) ? 1 : 0`, where N is encoded in the pattern. Typically uses `VL*` patterns.

---

### 9.3.2 Scalar-driven: `plt_b*`

`plt` generates a tail mask from a live `i32` scalar — the idiomatic choice for dynamic tail handling when not using `make_mask`.

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"mask_ops.creation","symbol":"mask_ops_creation_probe","compile":{}} -->
```python
mask, remained = pto.plt_b32(remained)
```

#### `pto.plt_b8(scalar: pto.i32) -> (pto.mask_b8, pto.i32)`
#### `pto.plt_b16(scalar: pto.i32) -> (pto.mask_b16, pto.i32)`
#### `pto.plt_b32(scalar: pto.i32) -> (pto.mask_b32, pto.i32)`

**Description**: Generates a tail mask where the first `min(scalar, VL)` lanes are active, and returns `scalar - min(scalar, VL)` as the updated remaining count.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `scalar` | `pto.i32` | Remaining element count |

**Returns**:

| Return Value | Type | Description |
|--------------|------|-------------|
| `mask` | `MaskType` | Tail mask (first N lanes active) |
| `scalar_out` | `pto.i32` | Updated remaining = `max(0, scalar - VL)` |

`VL` is 256 for `b8`, 128 for `b16`, and 64 for `b32`.

---

## 9.4 Mask logical operations

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"mask_ops.logical","symbol":"mask_ops_logical_probe","compile":{}} -->
```python
merged = pto.pand(src0, src1, gate)
```

Once created, masks can be combined with bitwise logical ops. All take a gating mask that selects which lanes participate; inactive lanes are zeroed in the result.

#### `pto.pand(src0: MaskType, src1: MaskType, mask: MaskType) -> MaskType`
#### `pto.por(src0: MaskType, src1: MaskType, mask: MaskType) -> MaskType`
#### `pto.pxor(src0: MaskType, src1: MaskType, mask: MaskType) -> MaskType`

**Description**: Bitwise AND / OR / XOR of two masks, gated by a third mask: `dst[i] = mask[i] ? (src0[i] <op> src1[i]) : 0`. All three masks must share the same granularity.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `src0` | `MaskType` | First source mask |
| `src1` | `MaskType` | Second source mask |
| `mask` | `MaskType` | Gating mask (lanes where false produce 0) |

**Returns**:

| Return Value | Type | Description |
|--------------|------|-------------|
| `result` | `MaskType` | Combined mask |

---

#### `pto.pnot(src: MaskType, mask: MaskType) -> MaskType`

**Description**: Bitwise NOT under gate: `dst[i] = mask[i] ? (~src[i]) : 0`.

---

#### `pto.psel(src0: MaskType, src1: MaskType, sel: MaskType) -> MaskType`

**Description**: Per-lane mask select: `dst[i] = sel[i] ? src0[i] : src1[i]`. All lanes participate directly — there is no additional gating beyond `sel` itself.

---

## 9.5 Mask reorganization

These ops reshape masks between granularities and layouts without changing the underlying 256-bit register image (except pack/unpack, which remap bits).

#### `pto.pbitcast(mask: MaskType, to_type: MaskType) -> MaskType`

**Description**: Bitwise reinterpretation of a mask at a different granularity. The 256-bit predicate register image is unchanged; only the lane count and element-width interpretation change.

**Example**:

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"type_system.mask_bitcast","symbol":"type_system_mask_bitcast_probe","compile":{}} -->
```python
# Reinterpret a b16 mask as b32
mask32 = pto.pbitcast(mask16, pto.mask_b32)
```

---

#### `pto.ppack(mask: MaskType, part: PredicatePart) -> MaskType`

**Description**: Narrowing pack — keeps one bit out of each adjacent 2-bit group from the source, packing them into the selected half (`LOWER` or `HIGHER`) of the result. The other half is zero-filled.

#### `pto.punpack(mask: MaskType, part: PredicatePart) -> MaskType`

**Description**: Widening unpack — reads the selected half of the source, zero-extends each 1-bit element into a 2-bit group in the result.

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"mask_ops.reorg","symbol":"mask_ops_reorg_probe","compile":{}} -->
```python
packed_hi = pto.ppack(mask32, pto.PredicatePart.HIGHER)
unpacked_hi = pto.punpack(packed_hi, pto.PredicatePart.HIGHER)
```

---

#### `pto.pintlv_b8(src0: pto.mask_b8, src1: pto.mask_b8) -> (pto.mask_b8, pto.mask_b8)`
#### `pto.pintlv_b16(src0: pto.mask_b16, src1: pto.mask_b16) -> (pto.mask_b16, pto.mask_b16)`
#### `pto.pintlv_b32(src0: pto.mask_b32, src1: pto.mask_b32) -> (pto.mask_b32, pto.mask_b32)`

**Description**: Interleave two masks element-wise. Returns `(low, high)` where `low[i] = src0[i]` and `high[i] = src1[i]` at each interleaved position.

#### `pto.pdintlv_b8(src0: pto.mask_b8, src1: pto.mask_b8) -> (pto.mask_b8, pto.mask_b8)`
#### `pto.pdintlv_b16(src0: pto.mask_b16, src1: pto.mask_b16) -> (pto.mask_b16, pto.mask_b16)`
#### `pto.pdintlv_b32(src0: pto.mask_b32, src1: pto.mask_b32) -> (pto.mask_b32, pto.mask_b32)`

**Description**: Deinterleave — the inverse of `pintlv`. Takes interleaved data in two masks and separates even/odd elements.

---

## 9.6 Comparisons: producing masks from vectors

Vector comparisons produce predicate masks from vector data. The result can feed into mask logical ops, `vsel`, or gated stores.

#### `pto.vcmp(v0: VRegType, v1: VRegType, seed_mask: MaskType, cmp_mode: CmpMode) -> MaskType`

**Description**: Element-wise vector-vector comparison: `dst[i] = seed_mask[i] ? (v0[i] <cmp> v1[i]) : 0`.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `v0` | `VRegType` | First operand vector |
| `v1` | `VRegType` | Second operand vector |
| `seed_mask` | `MaskType` | Seed mask gating which lanes participate |
| `cmp_mode` | `CmpMode` | `EQ`, `NE`, `LT`, `LE`, `GT`, `GE` |

**Returns**:

| Return Value | Type | Description |
|--------------|------|-------------|
| `pred` | `MaskType` | Result predicate mask (inherits granularity from operands) |

---

#### `pto.vcmps(vec: VRegType, scalar: ScalarType, seed_mask: MaskType, cmp_mode: CmpMode) -> MaskType`

**Description**: Vector-scalar comparison: `dst[i] = seed_mask[i] ? (vec[i] <cmp> scalar) : 0`. The scalar is broadcast to all lanes.

**Example** — threshold a vector:

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"mask_ops.compare","symbol":"mask_ops_compare_probe","compile":{}} -->
```python
big = pto.vcmps(scores, threshold, seed, pto.CmpMode.GT)
# big[i] = 1 where scores[i] > threshold
```

---

**Tile-level comparisons** (`pto.tile.cmp`, `pto.tile.cmps`) compare two tiles and write packed predicate bytes into an `i8` destination tile. They are used when the comparison result needs to be stored to UB for later selection (`tile.sel`) or cross-kernel communication.

---

## 9.7 Mask load and store

Masks can be persisted to and loaded from UB memory, enabling cross-stage predicate communication.

### 9.7.1 Predicate loads

#### `pto.plds(buf: PtrType, offset: Index, *, dist: PredicateDist = PredicateDist.NORM) -> MaskType`

**Description**: Load a predicate mask from UB memory at the given byte offset. The mask granularity is determined by the pointer element type of `buf` (`ui8`/`ui16`/`ui32` -> `mask_b8`/`mask_b16`/`mask_b32`).

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `buf` | `PtrType` (UB) | Source buffer |
| `offset` | `Index` | Byte offset |
| `dist` | `PredicateDist` | `NORM` (load VL/8 packed bytes), `US` (upsample), `DS` (downsample) |

**Returns**:

| Return Value | Type | Description |
|--------------|------|-------------|
| `mask` | `MaskType` | Loaded predicate mask |

---

### 9.7.2 Predicate stores

#### `pto.psts(mask: MaskType, buf: PtrType, offset: Index, *, dist: PredicateDist = PredicateDist.NORM) -> None`

**Description**: Store a predicate mask to UB memory.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `mask` | `MaskType` | Predicate mask to store |
| `buf` | `PtrType` (UB) | Destination buffer |
| `offset` | `Index` | Byte offset |
| `dist` | `PredicateDist` | `NORM` (store VL/8 packed bytes) or `PK` (pack to VL/16 bytes) |

**Returns**: None.

---

### 9.7.3 Unaligned predicate store

#### `pto.pstu(align_in: AlignType, mask: MaskType, buf: PtrType) -> (AlignType, PtrType)`

**Description**: Unaligned predicate store with alignment state threading. Threads the `align` state through a stream of stores, ensuring tail bytes are correctly buffered. This op currently supports only `mask_b16` and `mask_b32`; the base pointer type is determined by the mask granularity (`ui16` for `b16`, `ui32` for `b32`).

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `align_in` | `AlignType` | Incoming alignment state (from `init_align` or previous `pstu`) |
| `mask` | `MaskType` | Predicate mask to store (`mask_b16` or `mask_b32` only) |
| `buf` | `PtrType` (UB) | Destination buffer |

**Returns**:

| Return Value | Type | Description |
|--------------|------|-------------|
| `align_out` | `AlignType` | Updated alignment state |
| `base_out` | `PtrType` | Post-update base pointer |


## 9.8 How masks gate vector operations

Every vector compute op in Chapter 8 takes a mask as its last operand. The contract is consistent:

- For **unary ops** (`vexp`, `vabs`, etc.): `dst[i] = mask[i] ? f(src[i]) : 0`
- For **binary ops** (`vadd`, `vmul`, etc.): `dst[i] = mask[i] ? (lhs[i] <op> rhs[i]) : 0`
- For **vector stores** (`vsts`): `dst[i] = mask[i] ? src[i]` — masked-off lanes are not written
- For **reductions** (`vcadd`, `vcgmax`, etc.): only lanes where `mask[i]` is true contribute to the result

The mask granularity must match the vector element type. Using a `mask_b16` with an `f32` vector (or vice versa) is an error.

**Typical pattern** — tail-safe vector processing:

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"tail.vector_pattern","symbol":"tail_vector_pattern_probe","compile":{"BLOCK":128}} -->
```python
VEC = pto.elements_per_vreg(pto.f32)
for r in range(0, rows, 1):
    remained = cols
    for c in range(0, cols, VEC):
        mask, remained = pto.make_mask(pto.f32, remained)

        vec = pto.vlds(tile[r, c:])
        vec = pto.vexp(vec, mask)
        pto.vsts(vec, out_tile[r, c:], mask)
```

The `mask` gates the `vexp` (masked-off lanes produce 0) and the `vsts` (masked-off lanes are not written). The `remained` assignment makes the remaining count loop-carried after AST rewrite, so the final partial chunk correctly masks only the valid tail elements.

---

## 9.9 Tile-level mask operations

When working at the tile level (L1, `@pto.jit`), masks are carried in `i8` tile buffers holding packed predicate bytes. The key consumer of tile-level masks is `tile.sel`.

#### `pto.tile.sel(mask: Tile, src0: Tile, src1: Tile, dst: Tile, *, tmp: Tile | None = None) -> None`

**Description**: Element-wise ternary select: `dst[i,j] = mask[i,j] ? src0[i,j] : src1[i,j]`. `mask` is an integer tile (typically `i8`) where zero means false. `tmp` is an optional scratch tile override; when omitted, PTODSL synthesizes any architecture-specific scratch tile automatically.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `mask` | `Tile` | Integer mask tile (zero = false) |
| `src0` | `Tile` | True-branch source tile |
| `src1` | `Tile` | False-branch source tile |
| `tmp` | `Tile \| None` | Optional scratch tile override |
| `dst` | `Tile` | Destination tile |

**Returns**: None.

---

#### `pto.tile.sels(mask: Tile, src: Tile, scalar: ScalarType, dst: Tile, *, tmp: Tile | None = None) -> None`

**Description**: Element-wise select with scalar fallback: `dst[i,j] = mask[i,j] ? src[i,j] : scalar`. As with `tile.sel`, `tmp` is optional and PTODSL synthesizes any required scratch tile automatically when it is omitted.

---

## 9.10 Enum reference

| Enum | Values | Used with |
|------|--------|-----------|
| `MaskPattern` | `ALL`, `ALLF`, `H`, `Q`, `VL1`–`VL128`, `M3`, `M4` | `pset_b*`, `pge_b*`, `make_mask` |
| `CmpMode` | `EQ`, `NE`, `LT`, `LE`, `GT`, `GE` | `vcmp`, `vcmps` |
| `PredicateDist` (load) | `NORM`, `US`, `DS` | `plds` |
| `PredicateDist` (store) | `NORM`, `PK` | `psts` |
| `PredicatePart` | `LOWER`, `HIGHER` | `ppack`, `punpack` |
| `PostUpdate` | `OFF`, `ON` | `vstur`, `vlds`, `vsstb` |
| `VPackPart` | `LOWER`, `HIGHER` | `vpack`, `vunpack`, `vsunpack`, `vzunpack` |
| `PartMode` | `EVEN`, `ODD` | `vmulscvt`, `vexpdif` |
