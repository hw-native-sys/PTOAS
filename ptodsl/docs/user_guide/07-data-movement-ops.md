# 7. Data Movement Operations

This chapter covers every operation that moves data between memory spaces in PTODSL — tile-level transfers, DMA micro-instructions, vector loads and stores, and cube data movement. Operations are organized by abstraction level: tile ops for auto mode, DMA orchestration for explicit mode, vector memory ops on the SIMD unit, and cube memory ops on the Cube unit.

## 7.1 Tile-level movement: tile.load and tile.store

Section 7.6 on pipe communication is an advanced topic. Most kernels can skip
it on a first read and come back only when they need explicit cube/vector FIFO
coordination.

Tile ops move entire blocks between Global Memory and the Unified Buffer in a single call. They are the primary data movement interface inside `@pto.jit`.

#### `pto.tile.load(partition: PartitionTensorView, tile: Tile) -> None`

#### `pto.tile.load(tensor: TensorView, tile: Tile, *, offsets: tuple[IndexLike, ...] | None = None, sizes: tuple[IndexLike, ...] | None = None) -> None`

**Description**: Copies data from a GM partition into a UB tile. The transfer size is determined by the partition's `sizes` and the tile's shape — they must be compatible.

The `TensorView` overload builds the `partition_view` internally. When `offsets` is omitted, it defaults to zero offsets for every tensor dimension. When `sizes` is omitted, it is inferred from `tile.valid_shape`; for rank-changing layouts, pass `sizes=` explicitly or build a `PartitionTensorView` yourself.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `partition` | `PartitionTensorView` | Source region in GM |
| `tensor` | `TensorView` | Source GM tensor view; used by the overload that builds a partition internally |
| `tile` | `Tile` | Destination buffer in UB |
| `offsets` | `tuple[IndexLike, ...]` or `None` | Optional tensor offsets for the internal partition; defaults to all zeros |
| `sizes` | `tuple[IndexLike, ...]` or `None` | Optional partition sizes; defaults to `tile.valid_shape` when ranks match |

**Returns**: None (side-effect operation).

**Example**:

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"data_movement.tload","symbol":"data_movement_tload_probe","compile":{"BLOCK":128}} -->
```python
a_part = pto.partition_view(a_view, offsets=[offset, 0], sizes=[1, cols])
a_tile = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32)
pto.tile.load(a_part, a_tile)
pto.tile.load(a_view, a_tile, offsets=[offset, 0], sizes=[1, cols])
```

---

#### `pto.tile.store(tile: Tile, partition: PartitionTensorView) -> None`

#### `pto.tile.store(tile: Tile, tensor: TensorView, *, offsets: tuple[IndexLike, ...] | None = None, sizes: tuple[IndexLike, ...] | None = None) -> None`

**Description**: Copies data from a UB tile back to a GM partition. The tile's `valid_shape` determines how many elements are written; elements outside `valid_shape` are not stored.

The `TensorView` overload builds the `partition_view` internally. When `offsets` is omitted, it defaults to zero offsets for every tensor dimension. When `sizes` is omitted, it is inferred from `tile.valid_shape`; for rank-changing layouts, pass `sizes=` explicitly or build a `PartitionTensorView` yourself.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `tile` | `Tile` | Source buffer in UB |
| `partition` | `PartitionTensorView` | Destination region in GM |
| `tensor` | `TensorView` | Destination GM tensor view; used by the overload that builds a partition internally |
| `offsets` | `tuple[IndexLike, ...]` or `None` | Optional tensor offsets for the internal partition; defaults to all zeros |
| `sizes` | `tuple[IndexLike, ...]` or `None` | Optional partition sizes; defaults to `tile.valid_shape` when ranks match |

**Returns**: None (side-effect operation).

**Example**:

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"quick_start.tile_io","symbol":"quick_start_tile_io_probe","compile":{"BLOCK":128}} -->
```python
pto.tile.store(o_tile, o_part)
pto.tile.store(o_tile, o_view, offsets=[0, 0], sizes=[rows, cols])
```

---

Both `tile.load` and `tile.store` operate at **tile granularity** — they are the idiomatic choice inside `@pto.jit` loops. When you need finer control over DMA scheduling, switch to
`mode="explicit"` and use the DMA micro-instructions covered in the next section.

## 7.2 DMA micro-instructions (explicit mode)

Inside explicit-mode orchestration, data movement between memory spaces is expressed with grouped DMA instructions on typed pointers. There are four operations covering the four data-movement directions:

| Operation | Direction | Stride unit | Padding |
|-----------|-----------|-------------|---------|
| `pto.mte_gm_ub` | GM → UB | bytes | Supported |
| `pto.mte_ub_gm` | UB → GM | bytes | — (de-padded on read) |
| `pto.mte_ub_ub` | UB → UB | 32B units | — |
| `pto.mte_ub_l1` | UB → L1 | 32B units | — |

All four share a common structure: a required innermost `nburst(...)` group that defines the repeated burst transfer, plus optional outer `loop(...)` groups for multi-level repetition. `pto.mte_gm_ub` additionally supports `pad(...)` for UB row padding.

For day-to-day explicit-mode authoring, PTODSL also exposes the shorthand
wrappers `pto.mte_load(...)` and `pto.mte_store(...)`. They are ptr-based
aliases for the canonical `pto.mte_gm_ub(...)` and `pto.mte_ub_gm(...)`
surfaces and accept the same grouped-DMA shape arguments. Later walkthroughs
use the shorthand names because they read more naturally in orchestration code;
this chapter documents the underlying canonical operations.

### 7.2.1 GM → UB: `pto.mte_gm_ub`

#### `pto.mte_gm_ub(gm_src: PtrType, ub_dst: PtrType, l2_cache_ctl: int, len_burst: int, *, nburst: tuple[int, int, int], loops: list[tuple[int, int, int]] | None = None, pad: tuple[ScalarType, int, int] | tuple[ScalarType] | None = None) -> None`

**Description**: Grouped DMA transfer from Global Memory to Unified Buffer. `nburst(...)` defines the innermost repeated burst (count, source stride in bytes, destination stride in bytes). Optional `loop(...)` groups add outer repetition levels. Optional `pad(...)` fills the gap between `len_burst` and `dst_stride` up to the 32B-aligned boundary.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `gm_src` | `PtrType` (gm) | GM source pointer |
| `ub_dst` | `PtrType` (ub) | UB destination pointer (must be 32B-aligned) |
| `l2_cache_ctl` | `int` | L2 cache allocate control (2 bits). PTODSL forwards the integer verbatim; most kernels should pass `0` unless they are matching a backend-specific cache policy |
| `len_burst` | `int` | Contiguous bytes transferred per burst row |
| `nburst` | `tuple[int, int, int]` | `(n_burst, src_stride, dst_stride)` — innermost burst group (required) |
| `loops` | `list[tuple[int, int, int]]` or `None` | Optional outer loop groups, each `(count, src_stride, dst_stride)`. Ordered inner to outer |
| `pad` | `tuple[ScalarType, int, int]` or `tuple[ScalarType]` or `None` | Optional padding: `(pad_value, left_count, right_count)` or `(pad_value,)`. Omitted counts default to 0 |

**Returns**: None (side-effect operation).

**Constraints**:
- `nburst` is always required.
- `loop` groups are ordered from inner (wrapping `nburst`) to outer.
- If `pad` specifies either left or right count, both must be provided.

**Example** — load a 32×32 f32 tile from contiguous GM into contiguous UB:

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"data_movement.grouped_dma_ptrs","symbol":"data_movement_grouped_dma_ptrs_probe","compile":{}} -->
```python
pto.mte_gm_ub(gm_src, ub_dst, 0, 128,
              nburst=(32, 128, 128))
# 32 rows, 128 bytes per row, contiguous in both GM and UB
```

**Example** — load a 64×128 f16 tile from a larger GM matrix (1024×512) into UB:

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"data_movement.grouped_dma_ptrs","symbol":"data_movement_grouped_dma_ptrs_probe","compile":{}} -->
```python
pto.mte_gm_ub(gm_src, ub_dst, 0, 256,
              nburst=(64, 1024, 256))
# 64 rows of 256 bytes each.
# GM: each row is 1024 bytes apart (full matrix row stride).
# UB: rows are packed contiguously (256-byte stride).
```

**Example** — load with padding (100 valid f16 columns into a 128-wide UB tile):

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"data_movement.grouped_dma_ptrs","symbol":"data_movement_grouped_dma_ptrs_probe","compile":{}} -->
```python
pto.mte_gm_ub(gm_src, ub_dst, 0, 200,
              nburst=(64, 200, 256),
              pad=(0.0, 0, 0))
# 64 rows, 200 valid bytes per row, 256-byte UB stride.
# Gap (56 bytes) between len_burst and dst_stride is zero-padded.
```

**Example** — multi-level loop: load 4 batches of 8×128 f16 tiles:

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"data_movement.grouped_dma_ptrs","symbol":"data_movement_grouped_dma_ptrs_probe","compile":{}} -->
```python
pto.mte_gm_ub(gm_src, ub_dst, 0, 256,
              nburst=(8, 256, 256),
              loops=[(4, 2048, 2048)])
# Innermost: 8 rows × 256B (one tile).
# Outer loop: 4 iterations, each advancing 2048 bytes in both GM and UB.
```

---

### 7.2.2 UB → GM: `pto.mte_ub_gm`

#### `pto.mte_ub_gm(ub_src: PtrType, gm_dst: PtrType, len_burst: int, *, nburst: tuple[int, int, int], loops: list[tuple[int, int, int]] | None = None) -> None`

**Description**: Grouped DMA transfer from Unified Buffer to Global Memory. The MTE reads `len_burst` bytes from each UB row (skipping any padding), writing only valid data to GM.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `ub_src` | `PtrType` (ub) | UB source pointer (must be 32B-aligned) |
| `gm_dst` | `PtrType` (gm) | GM destination pointer |
| `len_burst` | `int` | Contiguous bytes transferred per burst row |
| `nburst` | `tuple[int, int, int]` | `(n_burst, src_stride, dst_stride)` — innermost burst group (required) |
| `loops` | `list[tuple[int, int, int]]` or `None` | Optional outer loop groups, ordered inner to outer |

**Returns**: None (side-effect operation).

**Example** — store a 32×32 f32 tile from UB to GM:

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"data_movement.grouped_dma_ptrs","symbol":"data_movement_grouped_dma_ptrs_probe","compile":{}} -->
```python
pto.mte_ub_gm(ub_src_f32, gm_dst_f32, 128,
              nburst=(32, 128, 128))
```

**Example** — store a 64×128 f16 tile back to a larger GM matrix:

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"data_movement.grouped_dma_ptrs","symbol":"data_movement_grouped_dma_ptrs_probe","compile":{}} -->
```python
pto.mte_ub_gm(ub_src, gm_dst, 256,
              nburst=(64, 256, 1024))
# UB: contiguous rows (256-byte stride).
# GM: rows spaced at 1024-byte intervals (full matrix width).
```

---

### 7.2.3 UB → UB: `pto.mte_ub_ub`

#### `pto.mte_ub_ub(ub_src: PtrType, ub_dst: PtrType, len_burst: int, *, nburst: tuple[int, int, int]) -> None`

**Description**: Grouped UB-to-UB copy. Stride and gap values are in units of 32 bytes.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `ub_src` | `PtrType` (ub) | UB source pointer (must be 32B-aligned) |
| `ub_dst` | `PtrType` (ub) | UB destination pointer (must be 32B-aligned) |
| `len_burst` | `int` | Burst length in units of 32 bytes |
| `nburst` | `tuple[int, int, int]` | `(n_burst, src_gap, dst_gap)` — count, source gap, destination gap (all in 32B units) |

**Returns**: None (side-effect operation).

Each burst copies `len_burst * 32` bytes. The next burst starts at `src + (len_burst + src_gap) * 32` and `dst + (len_burst + dst_gap) * 32`.

**Example**:

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"data_movement.grouped_dma_ptrs","symbol":"data_movement_grouped_dma_ptrs_probe","compile":{}} -->
```python
pto.mte_ub_ub(ub_src, ub_dst, 8,
              nburst=(16, 0, 4))
# 16 bursts, each copying 8×32=256 bytes.
# Source: contiguous (src_gap=0).
# Destination: 4×32=128-byte gap between bursts.
```

---

### 7.2.4 UB → L1: `pto.mte_ub_l1`

#### `pto.mte_ub_l1(ub_src: PtrType, l1_dst: PtrType, len_burst: int, *, nburst: tuple[int, int, int]) -> None`

**Description**: Grouped UB-to-L1 (CBUF) copy. Identical structure to `mte_ub_ub` but the destination is L1 cube buffer space.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `ub_src` | `PtrType` (ub) | UB source pointer (must be 32B-aligned) |
| `l1_dst` | `PtrType` (l1) | L1 destination pointer (must be 32B-aligned) |
| `len_burst` | `int` | Burst length in units of 32 bytes |
| `nburst` | `tuple[int, int, int]` | `(n_burst, src_gap, dst_gap)` — all in 32B units |

**Returns**: None (side-effect operation).

---

### 7.2.5 The nburst / loop / pad model

All grouped DMA operations follow a nested-loop execution model. `nburst` is the innermost group; each `loop` wraps the previous group as an outer iteration level.

For `mte_gm_ub` and `mte_ub_gm`, strides are **byte distances** from the start of one burst row to the start of the next:

```
GM → UB (nburst only):

  for r in range(n_burst):
      memcpy(ub_dst + r * dst_stride,
             gm_src + r * src_stride,
             len_burst)
      if pad enabled:
          memset(ub_dst + r * dst_stride + len_burst,
                 pad_value,
                 dst_stride_aligned - len_burst)
```

Each additional `loop(count, src_stride, dst_stride)` adds one outer `for` level that advances both base pointers by the corresponding strides.

For `mte_ub_ub` and `mte_ub_l1`, the parameters are in **32-byte units**. Each burst copies `len_burst * 32` bytes, and the next burst starts at `src + (len_burst + src_gap) * 32` / `dst + (len_burst + dst_gap) * 32`.

**UB address alignment**: For all four operations, every UB address (source and destination) must be 32-byte aligned. The `pad(...)` on `mte_gm_ub` ensures each UB row is padded to the 32B-aligned boundary of `dst_stride`, so subsequent rows stay aligned.

### 7.2.6 Typical explicit-mode DMA pattern

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"data_movement.explicit_dma","symbol":"data_movement_explicit_dma_probe","compile":{"ROWS":8,"COLS":16}} -->
```python
# Inside a @pto.jit(mode="explicit") body:
def process_block(k_part, v_part, k_tile, v_tile, o_tile, o_part,
                  rows: pto.i32, cols: pto.i32):
    # Stage K and V blocks from GM to UB
    pto.mte_gm_ub(k_part.as_ptr(), k_tile.as_ptr(), 0,
                  cols * pto.bytewidth(pto.f16),
                  nburst=(rows, cols * pto.bytewidth(pto.f16),
                          cols * pto.bytewidth(pto.f16)))
    pto.mte_gm_ub(v_part.as_ptr(), v_tile.as_ptr(), 0,
                  cols * pto.bytewidth(pto.f16),
                  nburst=(rows, cols * pto.bytewidth(pto.f16),
                          cols * pto.bytewidth(pto.f16)))
    pto.pipe_barrier(pto.Pipe.ALL)

    # ... compute on tiles ...

    pto.pipe_barrier(pto.Pipe.ALL)
    pto.mte_ub_gm(o_tile.as_ptr(), o_part.as_ptr(),
                  cols * pto.bytewidth(pto.f32),
                  nburst=(rows, cols * pto.bytewidth(pto.f32),
                          cols * pto.bytewidth(pto.f32)))
```

## 7.3 Vector loads (simd)

Inside `@pto.tileop`, data moves between UB tiles and vector registers (`vreg`). Vector loads read a contiguous chunk of a tile row into a `vreg`; the chunk size equals the hardware vector width for the element type (e.g., 64 elements for `f32`, 128 for `f16`).

### Tile-index syntax

All vector load and store operations support the element-indexing syntax, which eliminates manual byte-offset calculation:

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"data_movement.tile_slice_2d","symbol":"data_movement_tile_slice_2d_probe","compile":{"BLOCK":128}} -->
```python
vec = pto.vlds(tile[row, col:])       # load from row, starting at column col
```

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"data_movement.tile_slice_1d","symbol":"data_movement_tile_slice_1d_probe","compile":{"BLOCK":128}} -->
```python
vec = pto.vlds(tile[start:])          # 1D tile, starting at element start
```

The compiler automatically computes the byte offset from the tile's shape, element type, and layout. The `:` indicates a full vector-width range — the number of elements loaded is `elements_per_vreg(dtype)`.

---

#### `pto.vlds(tile[row, col:], *, dist: VLoadDist | None = None) -> VRegType`
#### `pto.vlds(tile[start:], *, dist: VLoadDist | None = None) -> VRegType`
#### `pto.vlds(buf: PtrType, offset: Index, *, dist: VLoadDist | None = None, post_update: PostUpdate = PostUpdate.OFF) -> VRegType | (VRegType, PtrType)`

**Description**: Stateless vector load from UB. Reads one vector-width slice.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `tile[row, col:]` | Tile index | 2D tile row with starting column (vector-width range) |
| `tile[start:]` | Tile index | 1D tile with starting element (vector-width range) |
| `buf` | `PtrType` (UB) | Pointer to buffer in UB (pointer form) |
| `offset` | `Index` | Element offset (pointer form) |
| `dist` | `VLoadDist` or `None` | Optional load distribution: `NORM` (default), `UNPK_B8`/`UNPK_B16`/`UNPK_B32`, `BRC_B8`/`BRC_B16`/`BRC_B32` |
| `post_update` | `PostUpdate` | Pointer form only. `OFF` (default) — stateless load. `ON` — returns `(vec, updated_buf)` where `updated_buf` is the buffer pointer advanced past the loaded elements |

**Returns**:

| Return Value | Type | Description |
|--------------|------|-------------|
| `vec` | `VRegType` | Loaded vector register (when `post_update=OFF`) |
| `(vec, updated_buf)` | `(VRegType, PtrType)` | Loaded vector and advanced pointer (when `post_update=ON`, pointer form only) |

Low-precision element types use the same pointer/tile forms. Use `b8` masks for 8-bit storage formats, including packed FP4 storage types:

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"data_movement.low_precision_vector_memory","symbol":"data_movement_low_precision_vector_memory_probe","compile":{}} -->
```python
mask_b8 = pto.pset_b8(pto.MaskPattern.ALL)
vec_f8 = pto.vlds(f8_src, pto.const(0))
pto.vsts(vec_f8, f8_dst, pto.const(0), mask_b8)
low, high = pto.vldsx2(fp4_src, pto.const(0), pto.DeinterleaveDist.DINTLV_B8)
pto.vstsx2(low, high, fp4_dst, pto.const(0), pto.InterleaveDist.INTLV_B8, mask_b8)
```


#### `pto.vldsx2(tile[row, col:], dist: DeinterleaveDist) -> (VRegType, VRegType)`
#### `pto.vldsx2(tile[start:], dist: DeinterleaveDist) -> (VRegType, VRegType)`
#### `pto.vldsx2(buf: PtrType, offset: Index, dist: DeinterleaveDist) -> (VRegType, VRegType)`

**Description**: Dual vector load with deinterleave (AoS → SoA). Loads interleaved data and deinterleaves into two vectors.

PTODSL accepts both pointer-based forms and tile-slice forms. The tile-slice
spellings are PTODSL surface sugar; the pointer form `buf[offset] + dist` is
the canonical form.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `tile[row, col:]` | Tile index | 2D tile row with starting column (vector-width range) |
| `tile[start:]` | Tile index | 1D tile with starting element (vector-width range) |
| `buf` | `PtrType` (UB) | Pointer to buffer in UB (pointer form) |
| `offset` | `Index` | Element offset (pointer form) |
| `dist` | `DeinterleaveDist` | `DINTLV_B8` / `DINTLV_B16` / `DINTLV_B32` (alternating elements) or `BDINTLV` (block deinterleave) |

**Returns**:

| Return Value | Type | Description |
|--------------|------|-------------|
| `low` | `VRegType` | Even-indexed elements |
| `high` | `VRegType` | Odd-indexed elements |

---

#### `pto.vldas(tile[row, col:]) -> AlignType`
#### `pto.vldas(tile[start:]) -> AlignType`
#### `pto.vldas(buf: PtrType) -> AlignType`

**Description**: Primes the alignment buffer for a subsequent unaligned load stream. Returns alignment state consumed by `vldus`.

PTODSL accepts both pointer-based forms and tile-slice forms. The tile-slice
spellings are PTODSL surface sugar; the pointer form is the canonical form.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `tile[row, col:]` | Tile index | 2D tile row with starting column |
| `tile[start:]` | Tile index | 1D tile with starting element |
| `buf` | `PtrType` | Pointer to buffer in UB |

**Returns**:

| Return Value | Type | Description |
|--------------|------|-------------|
| `align` | `AlignType` | Alignment state for use with `vldus` |

---

#### `pto.vldus(tile[row, col:], align: AlignType) -> (VRegType, AlignType)`
#### `pto.vldus(tile[start:], align: AlignType) -> (VRegType, AlignType)`
#### `pto.vldus(buf: PtrType, align: AlignType) -> (VRegType, AlignType)`

**Description**: Unaligned load with alignment state threading. Requires alignment state from `vldas` or a previous `vldus`.

PTODSL accepts both pointer-based forms and tile-slice forms. The tile-slice
spellings are PTODSL surface sugar; the pointer form is the canonical form.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `tile[row, col:]` | Tile index | 2D tile row with starting column (vector-width range) |
| `tile[start:]` | Tile index | 1D tile with starting element (vector-width range) |
| `buf` | `PtrType` (UB) | Pointer to buffer in UB (pointer form) |
| `align` | `AlignType` | Alignment state from `vldas` or previous `vldus` |

**Returns**:

| Return Value | Type | Description |
|--------------|------|-------------|
| `vec` | `VRegType` | Assembled vector |
| `align_out` | `AlignType` | Updated alignment state for next load |
**Example**:

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"data_movement.tile_slice_2d","symbol":"data_movement_tile_slice_2d_probe","compile":{"BLOCK":128}} -->
```python
align = pto.vldas(tile[row, col:])
vec, align = pto.vldus(tile[row, col:], align)
```

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"data_movement.tile_slice_1d","symbol":"data_movement_tile_slice_1d_probe","compile":{"BLOCK":128}} -->
```python
align = pto.vldas(tile[start:])
vec, align = pto.vldus(tile[start:], align)
```

---

#### `pto.vsld(tile[row, col], stride: StrideMode) -> VRegType`
#### `pto.vsld(tile[pos], stride: StrideMode) -> VRegType`
#### `pto.vsld(buf: PtrType, offset: Index, stride: StrideMode) -> VRegType`

**Description**: Strided scalar load with broadcast. Loads a single element using a strided access pattern and broadcasts to all vector lanes.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `tile[row, col]` | Tile index | 2D single-element index |
| `tile[pos]` | Tile index | 1D single-element index |
| `stride` | `StrideMode` | `S3_B16`, `S4_B64`, `S8_B32`, or `S2_B64` |

**Returns**:

| Return Value | Type | Description |
|--------------|------|-------------|
| `vec` | `VRegType` | Broadcast vector |

---

#### `pto.vgather2(buf: PtrType, offsets: Index, mask: MaskType) -> VRegType`

**Description**: Indexed gather from UB using per-lane offsets. Only masked-on
lanes participate.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `buf` | `PtrType` (UB) | Source buffer |
| `offsets` | `Index` | Per-lane element offsets (vector register) |
| `mask` | `MaskType` | Predicate mask gating lane participation |

**Returns**:

| Return Value | Type | Description |
|--------------|------|-------------|
| `vec` | `VRegType` | Gathered vector |

---

#### `pto.vgather2_bc(buf: PtrType, offsets: Index, mask: MaskType) -> VRegType`

**Description**: Indexed gather with mask. Masked-off lanes are zero-filled.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `buf` | `PtrType` (UB) | Source buffer |
| `offsets` | `Index` | Per-lane element offsets (vector register) |
| `mask` | `MaskType` | Mask gating lane participation |

**Returns**:

| Return Value | Type | Description |
|--------------|------|-------------|
| `vec` | `VRegType` | Gathered vector |

---

#### `pto.vgatherb(buf: PtrType, offsets: Index, mask: MaskType) -> VRegType`

**Description**: Block gather load. Participating lanes gather 32-byte blocks
from UB using byte offsets.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `buf` | `PtrType` (UB) | Source buffer |
| `offsets` | `Index` | Per-block byte offsets |
| `mask` | `MaskType` | `b32` predicate controlling which blocks participate |

**Returns**:

| Return Value | Type | Description |
|--------------|------|-------------|
| `vec` | `VRegType` | Gathered vector |

---

#### `pto.vsldb(tile[row, col], block_stride: Index, repeat_stride: Index, mask: MaskType) -> VRegType`
#### `pto.vsldb(tile[pos], block_stride: Index, repeat_stride: Index, mask: MaskType) -> VRegType`
#### `pto.vsldb(buf: PtrType, block_stride: Index, repeat_stride: Index, mask: MaskType) -> VRegType`

**Description**: Block-strided load. The source is interpreted as a sequence of
32-byte blocks addressed by `repeat_stride + blk * block_stride`. Masked-off
blocks are zero-filled.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `block_stride` | `Index` | 16-bit block stride field |
| `repeat_stride` | `Index` | 16-bit repeat stride field |
| `mask` | `MaskType` | Mask controlling which blocks participate |

**Returns**:

| Return Value | Type | Description |
|--------------|------|-------------|
| `vec` | `VRegType` | Block-strided vector |

## 7.4 Vector stores (simd)

Vector stores write `vreg` contents back to UB tiles. Like loads, they support tile-index syntax.

#### `pto.vsts(vec: VRegType, tile[row, col:], mask: MaskType, dist: VStoreDist | None = None, *, post_update: PostUpdate = PostUpdate.OFF) -> None`
#### `pto.vsts(vec: VRegType, tile[start:], mask: MaskType, dist: VStoreDist | None = None, *, post_update: PostUpdate = PostUpdate.OFF) -> None`
#### `pto.vsts(vec: VRegType, buf: PtrType, offset: Index, mask: MaskType, dist: VStoreDist | None = None, *, post_update: PostUpdate = PostUpdate.OFF) -> None | PtrType`

**Description**: Vector store to UB. The mask gates writes for the distributions
that use predicate masking. When `post_update=PostUpdate.ON`, the pointer form
returns the updated destination pointer. Tile-index forms remain side-effect
only and currently support `post_update=PostUpdate.OFF` only.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Vector to store |
| `tile[row, col:]` | Tile index | 2D destination (vector-width range) |
| `tile[start:]` | Tile index | 1D destination (vector-width range) |
| `buf` | `PtrType` (UB) | Destination buffer (pointer form) |
| `offset` | `Index` | Element offset (pointer form) |
| `post_update` | `PostUpdate` | Pointer-form stateful store mode. `ON` returns the updated destination pointer. |
| `mask` | `MaskType` | Predicate mask gating writes |
| `dist` | `VStoreDist` or `None` | Store distribution token. When omitted, PTODSL defaults to `NORM_B32` on the current surface. |

**Returns**: `None` for tile-index forms and pointer-form stores with `post_update=PostUpdate.OFF`; the updated destination `PtrType` for pointer-form stores with `post_update=PostUpdate.ON`.

**Distribution families**:

| Family | Notes |
|--------|-------|
| `NORM_B8` / `NORM_B16` / `NORM_B32` | Contiguous vector store |
| `1PT_B8` / `1PT_B16` / `1PT_B32` | First-element-only store; predicate is ignored |
| `PK_B16` / `PK_B32` / `PK_B64` | Packed store families |
| `PK4_B32` | 4-way packed store |
| `MRG4CHN_B8` | 4-channel merge store |
| `MRG2CHN_B8` / `MRG2CHN_B16` | 2-channel merge store |

---

#### `pto.psts(mask: MaskType, buf: PtrType, offset: Index, *, dist: PredicateDist = PredicateDist.NORM) -> None`

**Description**: Predicate store. Writes the packed predicate payload of `mask`
to UB memory.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `mask` | `MaskType` | Predicate payload to store |
| `buf` | `PtrType` (UB) | Destination buffer |
| `offset` | `Index` | Byte offset |
| `dist` | `PredicateDist` | Predicate payload layout. PTODSL defaults to `NORM` on the current surface. |

**Returns**: None (side-effect operation).

---

#### `pto.vstsx2(low: VRegType, high: VRegType, tile[row, col:], dist: InterleaveDist, mask: MaskType) -> None`
#### `pto.vstsx2(low: VRegType, high: VRegType, tile[start:], dist: InterleaveDist, mask: MaskType) -> None`
#### `pto.vstsx2(low: VRegType, high: VRegType, buf: PtrType, offset: Index, dist: InterleaveDist, mask: MaskType) -> None`

**Description**: Dual interleaving store (SoA → AoS). Interleaves two vectors
into one destination.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `low` | `VRegType` | First vector (even elements) |
| `high` | `VRegType` | Second vector (odd elements) |
| `tile[row, col:]` | Tile index | 2D destination (vector-width range) |
| `tile[start:]` | Tile index | 1D destination (vector-width range) |
| `buf` | `PtrType` (UB) | Destination buffer (pointer form) |
| `offset` | `Index` | Element offset (pointer form) |
| `dist` | `InterleaveDist` | `INTLV_B8` / `INTLV_B16` / `INTLV_B32` |
| `mask` | `MaskType` | Parameter retained for call-shape regularity; for the `INTLV_B*` family it does not affect the stored result |

**Returns**: None (side-effect operation).

---

#### `pto.vsstb(tile[row, col], block_stride: Index, repeat_stride: Index, mask: MaskType, *, post_update: PostUpdate = PostUpdate.OFF) -> None | PtrType`
#### `pto.vsstb(tile[pos], block_stride: Index, repeat_stride: Index, mask: MaskType, *, post_update: PostUpdate = PostUpdate.OFF) -> None | PtrType`
#### `pto.vsstb(buf: PtrType, block_stride: Index, repeat_stride: Index, mask: MaskType, *, post_update: PostUpdate = PostUpdate.OFF) -> None | PtrType`

**Description**: Block-strided store. Stores 32-byte source blocks to a
block-strided UB destination. Masked-off blocks do not write memory.
When `post_update=PostUpdate.ON`, the pointer form (and the pointer underlying
tile-index forms) returns the updated destination pointer advanced by the
repeat-stride distance, enabling stateful store streams.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `tile[row, col]` | Tile index | 2D starting element |
| `tile[pos]` | Tile index | 1D starting element |
| `buf` | `PtrType` (UB) | Destination buffer (pointer form) |
| `block_stride` | `Index` | 16-bit block stride field |
| `repeat_stride` | `Index` | 16-bit repeat stride field |
| `mask` | `MaskType` | Mask controlling which blocks participate |
| `post_update` | `PostUpdate` | `OFF` (default) — stateless store. `ON` — returns the destination pointer advanced by the repeat stride |

**Returns**:

| Return Value | Type | Description |
|--------------|------|-------------|
| *(none)* | — | When `post_update=OFF` (default) |
| `updated_buf` | `PtrType` | Post-update destination pointer (when `post_update=ON`)

---

#### `pto.vstar(align: AlignType, tile[row, col:]) -> None`
#### `pto.vstar(align: AlignType, tile[start:]) -> None`
#### `pto.vstar(align: AlignType, buf: PtrType) -> None`

**Description**: Flush alignment state to memory. Commits buffered tail bytes from an unaligned store stream. Consumes the alignment state.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `align` | `AlignType` | Pending store-alignment state |
| `tile[row, col:]` | Tile index | 2D destination (vector-width range) |
| `tile[start:]` | Tile index | 1D destination (vector-width range) |
| `buf` | `PtrType` (UB) | Destination buffer (pointer form) |

**Returns**: None (side-effect operation).

---

#### `pto.vstas(align: AlignType, tile[row, col:], offset: Index) -> None`
#### `pto.vstas(align: AlignType, tile[start:], offset: Index) -> None`
#### `pto.vstas(align: AlignType, buf: PtrType, offset: Index) -> None`

**Description**: Scalar-register-offset form of alignment-state flush. Same buffered-tail semantics as `vstar` with an explicit scalar offset.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `align` | `AlignType` | Pending store-alignment state |
| `tile[row, col:]` | Tile index | 2D destination (vector-width range) |
| `tile[start:]` | Tile index | 1D destination (vector-width range) |
| `buf` | `PtrType` (UB) | Destination buffer (pointer form) |
| `offset` | `Index` | Element offset (all forms) |

**Returns**: None (side-effect operation).

---

#### `pto.vscatter(vec: VRegType, buf: PtrType, offsets: Index, mask: MaskType) -> None`

**Description**: Indexed scatter to UB. Stores vector lanes to irregular locations using per-lane offsets.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `vec` | `VRegType` | Source vector to scatter |
| `buf` | `PtrType` (UB) | Destination buffer |
| `offsets` | `Index` | Per-lane element offsets (vector register) |
| `mask` | `MaskType` | Predicate mask gating lane participation |

**Returns**: None (side-effect operation).

---

### Stateful store family

For streaming unaligned stores with explicit alignment threading:

#### `pto.vstus(align_in: AlignType, offset: Index, vec: VRegType, buf: PtrType) -> AlignType`

**Description**: Scalar-offset unaligned store. Returns updated alignment state for the next store in the stream.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `align_in` | `AlignType` | Incoming store-alignment state |
| `offset` | `Index` | Scalar displacement |
| `vec` | `VRegType` | Vector to store |
| `buf` | `PtrType` (UB) | Destination buffer |

**Returns**:

| Return Value | Type | Description |
|--------------|------|-------------|
| `align_out` | `AlignType` | Updated buffered-tail state |

---

#### `pto.vstur(align_in: AlignType, vec: VRegType, buf: PtrType, mode: PostUpdate = PostUpdate.OFF) -> AlignType`

**Description**: Register-update unaligned store. Updates only residual alignment state without base pointer update.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `align_in` | `AlignType` | Incoming store-alignment state |
| `vec` | `VRegType` | Vector to store |
| `buf` | `PtrType` (UB) | Destination buffer |
| `mode` | `PostUpdate` | `PostUpdate.OFF` (default) or `PostUpdate.ON` |

**Returns**:

| Return Value | Type | Description |
|--------------|------|-------------|
| `align_out` | `AlignType` | Updated buffered-tail state |

---

#### `pto.pstu(align_in: AlignType, mask: MaskType, buf: PtrType) -> (AlignType, PtrType)`

**Description**: Predicate unaligned store with alignment state threading.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `align_in` | `AlignType` | Incoming store-alignment state |
| `mask` | `MaskType` | Predicate mask to store |
| `buf` | `PtrType` (UB) | Destination buffer |

**Returns**:

| Return Value | Type | Description |
|--------------|------|-------------|
| `align_out` | `AlignType` | Updated alignment state |
| `base_out` | `PtrType` | Post-update base pointer |

---

**Unaligned store stream pattern** — prime, thread, flush:

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"data_movement.grouped_dma_ptrs","symbol":"data_movement_grouped_dma_ptrs_probe","compile":{}} -->
```python
align = pto.init_align()
vec0 = pto.vlds(ub_src_f32, pto.const(0))
align = pto.vstur(align, vec0, ub_dst_f32, pto.PostUpdate.OFF)
align = pto.vstus(align, pto.const(32), vec0, ub_dst_f32)
pto.vstas(align, ub_dst_f32, pto.const(64))
```

### Distribution enums reference

| Enum | Values | Used with |
|------|--------|-----------|
| `VLoadDist` | `NORM`, `UNPK_B8`, `UNPK_B16`, `UNPK_B32`, `BRC_B8`, `BRC_B16`, `BRC_B32`, `US_B8`, `US_B16`, `DS_B8`, `DS_B16` | `vlds` |
| `VStoreDist` | `NORM_B8`, `NORM_B16`, `NORM_B32`, `1PT_B8`, `1PT_B16`, `1PT_B32`, `PK_B16`, `PK_B32`, `PK_B64`, `PK4_B32`, `MRG4CHN_B8`, `MRG2CHN_B8`, `MRG2CHN_B16` | `vsts` |
| `DeinterleaveDist` | `DINTLV_B8`, `DINTLV_B16`, `DINTLV_B32`, `BDINTLV` | `vldsx2` |
| `InterleaveDist` | `INTLV_B8`, `INTLV_B16`, `INTLV_B32` | `vstsx2` |
| `StrideMode` | `S3_B16`, `S4_B64`, `S8_B32`, `S2_B64` | `vsld` |
| `PostUpdate` | `OFF`, `ON` | `vstur`, `vsstb`, `vlds` (pointer form) |

## 7.5 Cube data movement (cube)

Inside a cube-style `@pto.tileop` helper, data flows through a hierarchy of
private buffers: GM → L1 (cbuf) → L0A/L0B (operand buffers) → L0C
(accumulator) → UB or back to GM.

### Staging: GM → L1 and L1 → UB

#### `pto.mte_gm_l1(src: PtrType, dst: PtrType, len_burst: int, *, nburst: tuple[int, int, int], loops: list[tuple[int, int, int]] | None = None) -> None`

**Description**: Structured GM-to-L1 (cbuf) data movement.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `src` | `PtrType` (GM) | Global Memory source pointer |
| `dst` | `PtrType` (L1) | L1 (cbuf) destination pointer |
| `len_burst` | `int` | Burst length in bytes |
| `nburst` | `tuple[int, int, int]` | `(count, src_stride, dst_stride)` |
| `loops` | `list[tuple[int, int, int]]` or `None` | Optional nested loop parameters |

**Returns**: None (side-effect operation).

---

#### `pto.mte_gm_l1_frac(src: PtrType, dst: PtrType, mode: pto.FractalMode, *, shape: tuple[int, int], src_layout: tuple[int] | tuple[int, int], dst_group: tuple[int, int, int, int], ctrl: tuple[int, bool]) -> None`

**Description**: Fractal GM-to-L1 load for specialized layouts (`ND2NZ`, `DN2NZ`).

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `src` | `PtrType` (GM) | Global Memory source pointer |
| `dst` | `PtrType` (L1) | L1 destination pointer |
| `mode` | `pto.FractalMode` | `pto.FractalMode.ND2NZ` or `pto.FractalMode.DN2NZ` |
| `shape` | `tuple[int, int]` | `(n_value, d_value)` |
| `src_layout` | `tuple[int]` or `tuple[int, int]` | `(inner_stride,)` or `(inner_stride, outer_stride)` |
| `dst_group` | `tuple[int, int, int, int]` | `(group_count, loop2_stride, loop3_stride, loop4_stride)` |
| `ctrl` | `tuple[int, bool]` | `(l2_cache_ctrl, smallc0_en)` |

**Returns**: None (side-effect operation).

---

#### `pto.mte_l1_ub(src: PtrType, dst: PtrType, len_burst: int, *, nburst: tuple[int, int, int], loops: list[tuple[int, int, int]] | None = None) -> None`

**Description**: Structured L1 (cbuf) to UB data movement.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `src` | `PtrType` (L1) | L1 source pointer |
| `dst` | `PtrType` (UB) | UB destination pointer |
| `len_burst` | `int` | Burst length in bytes |
| `nburst` | `tuple[int, int, int]` | `(count, src_stride, dst_stride)` |
| `loops` | `list[tuple[int, int, int]]` or `None` | Optional nested loop parameters |

**Returns**: None (side-effect operation).

---

### Operand loading: L1 → L0A / L0B

#### `pto.mte_l1_l0a(src: PtrType, dst: PtrType, m: int, k: int, *, start_row: int, start_col: int, transpose: bool = False) -> None`

**Description**: Structured L1-to-L0A (left-operand buffer) load.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `src` | `PtrType` (L1) | L1 source pointer |
| `dst` | `PtrType` (L0A) | L0A destination pointer |
| `m` | `int` | M dimension size |
| `k` | `int` | K dimension size |
| `start_row` | `int` | Source tile row offset for the extraction start position; the DSL materializes `0` when omitted |
| `start_col` | `int` | Source tile column offset for the extraction start position; the DSL materializes `0` when omitted |
| `transpose` | `bool` | Whether to load in transposed order |

**Returns**: None (side-effect operation).

---

#### `pto.mte_l1_l0b(src: PtrType, dst: PtrType, k: int, n: int, *, start_row: int, start_col: int, transpose: bool = False) -> None`

**Description**: Structured L1-to-L0B (right-operand buffer) load.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `src` | `PtrType` (L1) | L1 source pointer |
| `dst` | `PtrType` (L0B) | L0B destination pointer |
| `k` | `int` | K dimension size |
| `n` | `int` | N dimension size |
| `start_row` | `int` | Source tile row offset for the extraction start position; the DSL materializes `0` when omitted |
| `start_col` | `int` | Source tile column offset for the extraction start position; the DSL materializes `0` when omitted |
| `transpose` | `bool` | Whether to load in transposed order |

**Returns**: None (side-effect operation).

---

#### `pto.mte_l1_l0a_mx(src: PtrType, dst: PtrType, m: int, k: int, *, start_row: int = 0, start_col: int = 0) -> None`
#### `pto.mte_l1_l0b_mx(src: PtrType, dst: PtrType, k: int, n: int, *, start_row: int = 0, start_col: int = 0) -> None`

**Description**: MX-mode variants of `mte_l1_l0a` and `mte_l1_l0b` for MX-capable dtypes. Parameters match their non-MX counterparts.

---

### Bias and factor loading

#### `pto.mte_l1_bt(src: PtrType, dst: PtrType, len_burst: int, *, nburst: tuple[int, int, int]) -> None`
#### `pto.mte_l1_fb(src: PtrType, dst: PtrType, len_burst: int, *, nburst: tuple[int, int, int]) -> None`

**Description**: Structured L1 (cbuf) to bias table (`BT`) or factor/scaling buffer (`FB`) load.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `src` | `PtrType` (L1) | L1 source pointer |
| `dst` | `PtrType` (BIAS/SCALING) | BT or FB destination pointer |
| `len_burst` | `int` | Burst length in bytes |
| `nburst` | `tuple[int, int, int]` | `(count, src_gap, dst_gap)` |

**Returns**: None (side-effect operation).

---

### Accumulator writeback: L0C → L1 / GM / UB

#### `pto.mte_l0c_l1(src: PtrType, dst: PtrType, m: int, n: int, src_stride: int, dst_stride: int, *, unit_flag: pto.AccStoreUnitFlagCtrl | None = None, pre_quant: tuple[object, str] | None = None, pre_relu: tuple[str, object | None, object | None] | None = None, layout: object | None = None, loop3: tuple[int, int, int] | None = None, sat: pto.SatMode | None = None) -> None`

**Description**: Structured L0C (acc) to L1 (cbuf) writeback.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `src` | `PtrType` (L0C) | L0C accumulator source pointer |
| `dst` | `PtrType` (L1) | L1 destination pointer |
| `m` | `int` | M dimension size |
| `n` | `int` | N dimension size |
| `src_stride` | `int` | Source stride |
| `dst_stride` | `int` | Destination stride |
| `unit_flag` | `pto.AccStoreUnitFlagCtrl` or `None` | `CHECK_ONLY` or `CHECK_AND_CLEAR` |
| `pre_quant` | `tuple[object, str]` or `None` | Optional pre-quantization payload and mode |
| `pre_relu` | `tuple[str, object | None, object | None]` or `None` | Optional ReLU mode, payload, and clip payload |
| `layout` | `object` or `None` | `None`, `"nz2nd"`, or a layout tuple such as `("nz2dn", loop0_src_stride)` / `("nz2nz", split)` |
| `loop3` | `tuple[int, int, int]` or `None` | Optional loop-3 group `(count, src_stride, dst_stride)` |
| `sat` | `pto.SatMode` or `None` | `ON`, `OFF`, or `PRESERVE_NAN` |

**Returns**: None (side-effect operation).

---

#### `pto.mte_l0c_gm(src: PtrType, dst: PtrType, m: int, n: int, src_stride: int, dst_stride: int, sid: int, l2_cache_ctrl: int, *, unit_flag: pto.AccStoreUnitFlagCtrl | None = None, pre_quant: tuple[object, str] | None = None, pre_relu: tuple[str, object | None, object | None] | None = None, layout: object | None = None, loop3: tuple[int, int, int] | None = None, sat: pto.SatMode | None = None, atomic: tuple[str, str] | None = None) -> None`

**Description**: Structured L0C (acc) to GM writeback.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `src` | `PtrType` (L0C) | L0C accumulator source pointer |
| `dst` | `PtrType` (gm) | GM destination pointer |
| `m` | `int` | M dimension size |
| `n` | `int` | N dimension size |
| `src_stride` | `int` | Source stride |
| `dst_stride` | `int` | Destination stride |
| `sid` | `int` | Stream ID |
| `l2_cache_ctrl` | `int` | L2 cache control |
| `unit_flag` | `pto.AccStoreUnitFlagCtrl` or `None` | `CHECK_ONLY` or `CHECK_AND_CLEAR` |
| `pre_quant` | `tuple[object, str]` or `None` | Optional pre-quantization payload and mode |
| `pre_relu` | `tuple[str, object | None, object | None]` or `None` | Optional ReLU mode, payload, and clip payload |
| `layout` | `object` or `None` | `None`, `"nz2nd"`, or a layout tuple such as `("nz2dn", loop0_src_stride)` / `("nz2nz", split)` |
| `loop3` | `tuple[int, int, int]` or `None` | Optional loop-3 group `(count, src_stride, dst_stride)` |
| `sat` | `pto.SatMode` or `None` | `ON`, `OFF`, or `PRESERVE_NAN` |
| `atomic` | `tuple[str, str]` or `None` | Optional GM-only atomic `(type, op)`, e.g. `("f32", "add")` |

**Returns**: None (side-effect operation).

---

#### `pto.mte_l0c_ub(src: PtrType, dst: PtrType, m: int, n: int, src_stride: int, dst_stride: int, sub_blockid: int = 0, *, split: pto.SplitMode | None = None, unit_flag: pto.AccStoreUnitFlagCtrl | None = None, pre_quant: tuple[object, str] | None = None, pre_relu: tuple[str, object | None, object | None] | None = None, layout: object | None = None, loop3: tuple[int, int, int] | None = None, sat: pto.SatMode | None = None) -> None`

**Description**: Structured L0C (acc) directly to UB. This is the most common writeback path for cube kernels that feed results into subsequent processing.

**Parameters**:

| Parameter | Type | Description |
|-----------|------|-------------|
| `src` | `PtrType` (L0C) | L0C accumulator source pointer |
| `dst` | `PtrType` (ub) | UB destination pointer |
| `m` | `int` | M dimension size |
| `n` | `int` | N dimension size |
| `src_stride` | `int` | Source stride |
| `dst_stride` | `int` | Destination stride |
| `sub_blockid` | `int` | `0` or `1` for single-destination sub-block writeback |
| `split` | `pto.SplitMode` or `None` | `M` or `N` for dual-destination split; cannot be combined with non-default `sub_blockid` |
| `unit_flag` | `pto.AccStoreUnitFlagCtrl` or `None` | `CHECK_ONLY` or `CHECK_AND_CLEAR` |
| `pre_quant` | `tuple[object, str]` or `None` | Optional pre-quantization payload and mode |
| `pre_relu` | `tuple[str, object | None, object | None]` or `None` | Optional ReLU mode, payload, and clip payload |
| `layout` | `object` or `None` | `None`, `"nz2nd"`, or a layout tuple such as `("nz2dn", loop0_src_stride)` / `("nz2nz", split)` |
| `loop3` | `tuple[int, int, int]` or `None` | Optional loop-3 group `(count, src_stride, dst_stride)` |
| `sat` | `pto.SatMode` or `None` | `ON`, `OFF`, or `PRESERVE_NAN` |

**Returns**: None (side-effect operation).

`sub_blockid` / `split` forms:

```python
pto.mte_l0c_ub(acc, ub, 16, 32, 16, 32)                         # sub-block 0
pto.mte_l0c_ub(acc, ub, 16, 32, 16, 32, sub_blockid=1)           # sub-block 1
pto.mte_l0c_ub(acc, ub, 16, 32, 16, 32, split=pto.SplitMode.M)   # split M
pto.mte_l0c_ub(acc, ub, 16, 32, 16, 32, split=pto.SplitMode.N)   # split N
```

`atomic` is not supported on `mte_l0c_l1` or `mte_l0c_ub`; use `mte_l0c_gm(..., atomic=(type, op))` for GM atomic writeback.

---

### Cube data movement quick reference

| Data Flow | Operation | Src Space | Dst Space |
|-----------|-----------|-----------|-----------|
| GM → L1 | `mte_gm_l1` | gm | l1 |
| GM → L1 (fractal) | `mte_gm_l1_frac` | gm | l1 |
| L1 → UB | `mte_l1_ub` | l1 | ub |
| L1 → L0A | `mte_l1_l0a` | l1 | l0a |
| L1 → L0B | `mte_l1_l0b` | l1 | l0b |
| L1 → L0A (MX) | `mte_l1_l0a_mx` | l1 | l0a |
| L1 → L0B (MX) | `mte_l1_l0b_mx` | l1 | l0b |
| L1 → Bias table | `mte_l1_bt` | l1 | bt |
| L1 → Factor buffer | `mte_l1_fb` | l1 | fb |
| L0C → L1 | `mte_l0c_l1` | l0c | l1 |
| L0C → GM | `mte_l0c_gm` | l0c | gm |
| L0C → UB | `mte_l0c_ub` | l0c | ub |

### Typical cube dataflow in a matmul

A full cube matmul (a cube-style `@pto.tileop` helper) follows this dataflow
pattern:

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"data_movement.cube_helper","symbol":"data_movement_cube_helper_probe","compile":{"BLOCK_M":16,"BLOCK_K":16,"BLOCK_N":16}} -->
```python
@pto.tileop
def qk_matmul(
    q_tile: pto.Tile,
    k_tile: pto.Tile,
    q_l0a: pto.Tile,
    k_l0b: pto.Tile,
    s_acc: pto.Tile,
    s_tile: pto.Tile,
):
    m = q_tile.valid_shape[0]
    k = q_tile.valid_shape[1]
    n = k_tile.valid_shape[0]

    pto.mte_l1_l0a(q_tile.as_ptr(), q_l0a.as_ptr(), m, k)          # L1 tile → L0A
    pto.mte_l1_l0b(k_tile.as_ptr(), k_l0b.as_ptr(), k, n, transpose=True)  # L1 tile → L0B
    pto.mad(q_l0a.as_ptr(), k_l0b.as_ptr(), s_acc.as_ptr(), m, n, k)        # L0A × L0B → L0C
    pto.mte_l0c_ub(s_acc.as_ptr(), s_tile.as_ptr(), m, n, n, n, 0)          # L0C → UB tile
```

At the cube micro-op boundary, PTODSL currently uses explicit typed pointers. `tile.as_ptr()` materializes the pointer view for UB and cube-local scratch buffers, while the surrounding sub-kernel surface still uses `Tile` values for metadata such as `valid_shape`.

## 7.6 Pipe Communication (Cube ↔ Vector FIFO)

Pipe communication is the mechanism for Cube and Vector sub-kernels to exchange
data through hardware FIFO channels. PTODSL provides a high-level `pto.pipe`
API that presents pipes as logical declarations plus direction-aware
transactions.

### 7.6.1 Pipe Constructors

#### `pto.pipe.c2v(*, id, slot_size=None, consumer_buf=None, gm_slot_buffer=None, gm_slot_tensor=None, local_slot_num=None, nosplit=None)`

Creates a logical Cube-to-Vector pipe.

The constructor does not expose separate global/local names. A local tile-entry
pipe is selected by passing `slot_size` and `consumer_buf`. An A2/A3
global-entry L2G2L pipe is selected by passing `gm_slot_tensor`; the tensor view
describes the single FIFO slot entry. Global-entry pipes do not use a
consumer-side local FIFO buffer and do not take `consumer_buf` or
`gm_slot_buffer`. On A5, use the local tile-entry form and omit
`gm_slot_tensor`.

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `consumer_buf` | varies | Required for local tile-entry pipes. Consumer-owned FIFO buffer. The consumer side reserves it with `pto.reserve_buffer`; the producer side imports it with `pto.import_reserved_buffer`. Omit for global-entry pipes. |
| `id` | `int` | Required. Stable pipe identifier shared by the producer and consumer sides. |
| `slot_size` | `int` | Required for local tile-entry pipes. For global-entry pipes, omit it only when `nosplit=True` and `gm_slot_tensor` already describes one full slot; otherwise pass the full-slot byte size explicitly. |
| `gm_slot_buffer` | `PtrType` | Optional GM FIFO storage pointer for A2/A3 local tile-entry L2G2L lowering. Do not use with `gm_slot_tensor`. |
| `gm_slot_tensor` | `TensorView` | Optional. When provided, the pipe uses GlobalTensor-like entries and `alloc/pop` infer the entry descriptor type. |
| `local_slot_num` | `int` | Optional. Local FIFO slot count override for local tile-entry pipes. |
| `nosplit` | `bool` | Optional. Override-only metadata; not required in the common path. |

The returned pipe object exposes C2V-producer methods (`init_cube`, `alloc`,
`push`) on the Cube side and C2V-consumer methods (`init_simd`, `pop`, `free`)
on the Vector side.

#### `pto.pipe.v2c(*, id, slot_size=None, consumer_buf=None, gm_slot_buffer=None, gm_slot_tensor=None, local_slot_num=None, nosplit=None)`

Creates a logical Vector-to-Cube pipe. Same contract as `c2v`, but
reversed direction: the Vector side is the producer and the Cube side is the
consumer.

#### `pto.pipe.bidirectional(*, slot_size, c2v_consumer_buf, v2c_consumer_buf, id, gm_slot_buffer=None, local_slot_num=None, nosplit=None)`

Creates a bidirectional local tile-entry pipe. Accepts both `c2v_consumer_buf`
and `v2c_consumer_buf` since the pipe carries traffic in both directions. Use
the root pipe for `init_cube()` / `init_simd()`, and use directional endpoints
for transactions:


```python
# Cube side
bidi.init_cube()
bidi.c2v.push(cube_tile)
cube_tile = bidi.v2c.pop(result_type=cube_tile_type)
bidi.v2c.free()

# SIMD side
bidi.init_simd()
vec_tile = bidi.c2v.pop(result_type=vec_tile_type)
bidi.c2v.free()
bidi.v2c.push(vec_tile)
```

### 7.6.2 Pipe Methods

Every logical pipe object exposes only the methods that make sense for its
direction.

**Producer-side methods** (Cube side for C2V, Vector side for V2C):

| Method | Description |
|--------|-------------|
| `init_cube()` | Initialise the pipe on the Cube side. |
| `init_simd()` | Initialise the pipe on the Vector (SIMD) side. |
| `alloc(split=0)` | Allocate the next FIFO slot. Global-entry pipes only. Returns an entry descriptor. |
| `push(entry_or_tile, split=0)` | Push a filled GlobalTensor entry or local tile to the consumer. Notifies the consumer side. |

**Consumer-side methods** (Vector side for C2V, Cube side for V2C):

| Method | Description |
|--------|-------------|
| `init_cube()` | Initialise the pipe on the Cube side. |
| `init_simd()` | Initialise the pipe on the Vector (SIMD) side. |
| `pop(split=0, result_type=None, valid_shape=None)` | Pop the next entry from the producer. Global-entry pipes return a GM slot descriptor; local/tile-entry pipes return a tile. |
| `free(entry=None, split=0)` | Release the consumed slot back to the producer. |

**Read-only properties:**

| Property | Type | Description |
|----------|------|-------------|
| `entry_type` | type | The entry descriptor type for this pipe. |
| `id` | `int` | The compile-time stable pipe identifier. |
| `slot_size` | `int` | The logical slot size in bytes. |

Rules:

- `split` is a compile-time integer: `0` = no split, `1` = up/down split,
  `2` = left/right split.
- Pipe transactions are associated with their pipe by the stable `id`,
  direction, and the enclosing kernel function. Python variable names are not part of the IR. Kernels
  with multiple pipes must use distinct stable ids.
- For global-entry pipes, `push` and `pop` do not implicitly perform
  `tstore`/`tload`; callers must move data explicitly before `push` or after
  `pop`.
- For global-entry pipes, `result_type` on `pop()` defaults to the pipe's
  `entry_type`, and `pop()` returns a `TensorView` descriptor for the current GM
  FIFO slot. Callers can derive a sub-view from that descriptor and then load
  data explicitly.
- For local/tile-entry pipes, there is no `alloc()`. The producer pushes an
  existing tile directly. The consumer pops into a newly declared tile of
  `result_type`.
- For local/tile-entry pipes, `result_type` may be either a tile type or a tile
  value whose type should be reused. `valid_shape=[rows, cols]` can be supplied
  when the popped tile needs runtime valid-shape metadata.
- `entry` on `free()` may be omitted for tile-entry pipes. For global-entry
  pipes it must carry the entry descriptor returned by the matching `pop()`.

### 7.6.3 Global-Entry C2V Pipe

Declaration (shared between Cube and Vector sides):

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"pipe_communication.c2v_global_declaration","symbol":"pipe_communication_c2v_global_declaration_probe","compile":{}} -->
```python
c2v = pto.pipe.c2v(
    gm_slot_tensor=gm_slots,
    id=0,
    nosplit=True,
)
```

Cube (producer) side:

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"pipe_communication.c2v_global_producer","symbol":"pipe_communication_c2v_global_producer_probe","compile":{}} -->
```python
@pto.tileop
def producer(src_tile: pto.Tile):
    c2v.init_cube()
    entry = c2v.alloc(split=0)
    entry_part = pto.partition_view(entry, offsets=[0, 0], sizes=[16, 16])
    pto.tile.store(src_tile, entry_part)
    c2v.push(entry, split=0)
```

Vector (consumer) side:

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"pipe_communication.c2v_global_consumer","symbol":"pipe_communication_c2v_global_consumer_probe","compile":{}} -->
```python
@pto.tileop
def consumer(dst_tile: pto.Tile):
    c2v.init_simd()
    entry = c2v.pop(split=0)
    entry_part = pto.partition_view(entry, offsets=[0, 0], sizes=[16, 16])
    pto.tile.load(entry_part, dst_tile)
    c2v.free(entry, split=0)
```

The Cube side initialises the pipe, allocates a GM FIFO slot, stores the tile
data into that slot via `tile.store`, then pushes the entry to notify the
consumer. The Vector side initialises the pipe, pops the next ready entry, loads
the data into a local tile via `tile.load`, then frees the slot.

### 7.6.4 Global-Entry V2C Pipe

Declaration:

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"pipe_communication.v2c_global_declaration","symbol":"pipe_communication_v2c_global_declaration_probe","compile":{}} -->
```python
v2c = pto.pipe.v2c(
    gm_slot_tensor=gm_slots,
    id=0,
    nosplit=True,
)
```

Vector (producer) side:

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"pipe_communication.v2c_global_producer","symbol":"pipe_communication_v2c_global_producer_probe","compile":{}} -->
```python
@pto.tileop
def producer(src_tile: pto.Tile):
    v2c.init_simd()
    entry = v2c.alloc(split=0)
    pto.tile.store(src_tile, pto.partition_view(entry, offsets=[0, 0], sizes=[16, 16]))
    v2c.push(entry, split=0)
```

Cube (consumer) side:

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"pipe_communication.v2c_global_consumer","symbol":"pipe_communication_v2c_global_consumer_probe","compile":{}} -->
```python
@pto.tileop
def consumer(dst_tile: pto.Tile):
    v2c.init_cube()
    entry = v2c.pop(split=0)
    pto.tile.load(pto.partition_view(entry, offsets=[0, 0], sizes=[16, 16]), dst_tile)
    v2c.free(entry, split=0)
```

### 7.6.5 Local FIFO C2V Pipe

Vector (consumer) side reserves the local buffer:

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"pipe_communication.c2v_local_declaration","symbol":"pipe_communication_c2v_local_declaration_probe","compile":{}} -->
```python
c2v_buf = pto.reserve_buffer("c2v_fifo", size=8192, location="vec")
c2v = pto.pipe.c2v(
    slot_size=1024,
    consumer_buf=c2v_buf,
    id=0,
)
```

Cube (producer) side imports the peer buffer:

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"pipe_communication.c2v_local_import","symbol":"pipe_communication_c2v_local_import_probe","compile":{}} -->
```python
c2v_buf = pto.import_reserved_buffer("c2v_fifo", peer_func="vector_kernel")
c2v_peer = pto.pipe.c2v(
    slot_size=1024,
    consumer_buf=c2v_buf,
    id=0,
)
```

Cube (producer) transaction:

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"pipe_communication.c2v_local_producer","symbol":"pipe_communication_c2v_local_producer_probe","compile":{}} -->
```python
@pto.tileop
def producer(src_tile: pto.Tile):
    c2v_peer.init_cube()
    c2v_peer.push(src_tile, split=0)
```

Vector (consumer) transaction:

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"pipe_communication.c2v_local_consumer","symbol":"pipe_communication_c2v_local_consumer_probe","compile":{}} -->
```python
@pto.tileop
def consumer(dst_tile: pto.Tile):
    c2v.init_simd()
    tile = c2v.pop(result_type=dst_tile, split=0)
    pto.tile.load(tile, dst_tile)
    c2v.free(split=0)
```

The local form is the A5-facing form used when Cube and Vector exchange UB/MAT
tiles through a local FIFO. `push(tile)` emits a tile-entry `tpush`; `pop()`
emits a tile-entry `tpop` into the vector-side local tile; `free()` can omit
the entry because no GM FIFO slot descriptor was allocated by the frontend.

### 7.6.6 Bidirectional Local Pipe

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"pipe_communication.bidirectional_local_declaration","symbol":"pipe_communication_bidirectional_local_declaration_probe","compile":{}} -->
```python
bidi = pto.pipe.bidirectional(
    slot_size=1024,
    c2v_consumer_buf=c2v_buf,
    v2c_consumer_buf=v2c_buf,
    id=0,
)
```

For bidirectional local pipes, the C2V consumer buffer lives in `VEC` and the V2C consumer buffer lives in `MAT`.

The two-buffer shape only appears where the pipe is genuinely bidirectional.

### 7.6.7 Complete Example: C2V Global-Entry Pipe

This is a complete two-kernel C2V pipe example with `@pto.jit` entry points
and `gm_slots` expressed via `pto.make_tensor_view`:

<!-- ptodsl-doc-test: {"mode":"compile_fragment","fixture":"pipe_communication.c2v_global","symbol":"pipe_communication_c2v_global_probe","compile":{"BLOCK":128}} -->
```python
@pto.jit(target="a3")
def cube_producer(
    gm_slot_buffer: pto.gm_ptr(pto.f32),
    src: pto.gm_ptr(pto.f32),
    *,
    BLOCK: pto.const_expr = 128,
):
    gm_view = pto.make_tensor_view(gm_slot_buffer, shape=[16, 16], strides=[16, 1])
    c2v = pto.pipe.c2v(
        gm_slot_tensor=gm_view,
        id=0,
        nosplit=True,
    )

    a_part = pto.partition_view(
        pto.make_tensor_view(src, shape=[16, 16], strides=[16, 1]),
        offsets=[0, 0], sizes=[16, 16])
    a_tile = pto.alloc_tile(shape=[16, 16], dtype=pto.f32)

    with pto.tileop():
        pto.tile.load(a_part, a_tile)
        c2v.init_cube()
        entry = c2v.alloc(split=0)
        entry_part = pto.partition_view(entry, offsets=[0, 0], sizes=[16, 16])
        pto.tile.store(a_tile, entry_part)
        c2v.push(entry, split=0)

@pto.jit(target="a3")
def vector_consumer(
    gm_slot_buffer: pto.gm_ptr(pto.f32),
    dst: pto.gm_ptr(pto.f32),
    *,
    BLOCK: pto.const_expr = 128,
):
    gm_view = pto.make_tensor_view(gm_slot_buffer, shape=[16, 16], strides=[16, 1])
    c2v = pto.pipe.c2v(
        gm_slot_tensor=gm_view,
        id=0,
        nosplit=True,
    )

    b_tile = pto.alloc_tile(shape=[16, 16], dtype=pto.f32)
    b_part = pto.partition_view(
        pto.make_tensor_view(dst, shape=[16, 16], strides=[16, 1]),
        offsets=[0, 0], sizes=[16, 16])

    with pto.tileop():
        c2v.init_simd()
        entry = c2v.pop(split=0)
        entry_part = pto.partition_view(entry, offsets=[0, 0], sizes=[16, 16])
        pto.tile.load(entry_part, b_tile)
        c2v.free(entry, split=0)
        pto.tile.store(b_tile, b_part)
```
