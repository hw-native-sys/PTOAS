# 3. Pointer & View Operations

> **Category:** Address arithmetic, tensor-view construction, tile-buffer allocation
> **Pipeline:** None (all ops are metadata / view construction; no HW side effect)

These instructions build the address, view, and tile-buffer metadata that later DMA and compute instructions consume. None of them moves data.

---

## `pto.addptr`

- **syntax:**
```mlir
%result = pto.addptr %base, %offset : !pto.ptr<T> -> !pto.ptr<T>
```
- **semantics:** `result = ptr + offset`, with `offset` counted in **elements** (not bytes).

**Parameter Table:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `%base` | `!pto.ptr<T>` | Base pointer. |
| `%offset` | `index` | Element offset. |

**Constraints:**

- Result type must match the input pointer type.
- The op is pure (no side effects).

**Example:**

```mlir
%ptr_off = pto.addptr %base, %offset : !pto.ptr<f32> -> !pto.ptr<f32>
```

---

## `pto.castptr`

- **syntax:**
```mlir
%p_ptr  = pto.castptr %addr : i64 -> !pto.ptr<T, space>
%p_ptr2 = pto.castptr %p_ptr : !pto.ptr<T, space> -> !pto.ptr<T2, space>
%addr2  = pto.castptr %p_ptr : !pto.ptr<T, space> -> i64
```
- **semantics:** Explicit cast between integer addresses and `!pto.ptr`, or between two `!pto.ptr` types.

**Parameter Table:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `input` | integer \| `!pto.ptr<...>` | Source value. |

**Constraints:**

- Integer-to-integer casts are rejected; use normal integer cast ops.
- Descriptor types (`!pto.tensor_view<...>`, `!pto.partition_tensor_view<...>`) are not legal direct inputs; extract an address first.
- Pointer-to-pointer casts are only legal when source and destination stay in the same PTO memory space (`gm` or `vec`).
- The op is pure.

**Example:**

```mlir
%p0 = pto.castptr %addr : i64 -> !pto.ptr<f32, vec>
%p1 = pto.castptr %p0   : !pto.ptr<f32, vec> -> !pto.ptr<i8, vec>
%a2 = pto.castptr %p1   : !pto.ptr<i8, vec>  -> i64
```

---

## `pto.make_tensor_view`

- **syntax:**
```mlir
%tv = pto.make_tensor_view %ptr, shape = [%m, %n], strides = [%s0, %s1]
    : !pto.tensor_view<?x?xT>
```
- **semantics:** Construct a global tensor view from a pointer, declaring the physical base and strides. No allocation, no data movement.

**Parameter Table:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `%ptr` | `AnyType` | Source pointer (must be `!pto.ptr<T>` with element type matching the result). |
| `shape` | `Variadic<Index>` | Dynamic shape dimensions. |
| `strides` | `Variadic<Index>` | Dynamic strides. |
| `layout` | `LayoutAttr` (optional) | `nd`, `dn`, `nz`, or an MX layout from §2.8. |

**Constraints:**

- `ptr` element type must match the result element type.
- `shape` and `strides` operand counts must match the tensor_view rank.
- If `layout` is provided with static shapes/strides, it must be consistent with the inferred layout.
- MX layouts are available only on A5 and require a rank-2 view of `!pto.f8E8M0` scale elements. Shapes and strides are measured in scale elements, not quantized data elements.
- MX ND layouts require contiguous row-major strides `[cols, 1]`; MX DN layouts require contiguous column-major strides `[1, rows]`. MX ZZ/NN views use `[cols, 1]` for their logical shape.
- MX A layouts require an even column count; MX B layouts require an even row count. Additionally, `mx_a_zz` requires rows divisible by 16, and `mx_b_nn` requires columns divisible by 16. Dynamic dimensions must satisfy the same constraints at runtime.
- Runtime dimensions, element counts (`rows * cols`), and packed strides (including stride times two) must fit the signed index range and the backing global-memory allocation.
- A partition inherits the source layout; an explicitly different result layout is invalid. Dimension queries on MX views use logical dimension indices 0 and 1.

**Example:**

```mlir
%tv = pto.make_tensor_view %ptr, shape = [%m, %n], strides = [%s0, %s1]
    : !pto.tensor_view<?x?xf32>
```

---

## `pto.get_tensor_view_dim`

- **syntax:**
```mlir
%dim = pto.get_tensor_view_dim %tv, %idx : !pto.tensor_view<...> -> index
```
- **semantics:** Return the runtime size of dimension `%idx` from a `tensor_view`.

**Parameter Table:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `%tv` | `!pto.tensor_view<...>` | Logical tensor view. |
| `%idx` | `index` | Dimension index (0-based). |

**Example:**

```mlir
%h = pto.get_tensor_view_dim %tv, %c0 : !pto.tensor_view<?x?xf32> -> index
```

---

## `pto.get_tensor_view_stride`

- **syntax:**
```mlir
%stride = pto.get_tensor_view_stride %tv, %idx : !pto.tensor_view<...> -> index
```
- **semantics:** Return the logical stride of dimension `%idx`, measured in **elements** (not bytes).

**Parameter Table:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `%tv` | `!pto.tensor_view<...>` or memref form | Tensor view or its lowered memory-reference form. |
| `%idx` | `index` | Dimension index (0-based). |

**Example:**

```mlir
%s0 = pto.get_tensor_view_stride %tv, %c0 : !pto.tensor_view<?x?xf32> -> index
```

---

## `pto.tensor_view_addr`

- **syntax:**
```mlir
%result = pto.tensor_view_addr %src : !pto.tensor_view<...> -> memref<...>
%result = pto.tensor_view_addr %src : !pto.tensor_view<...> -> !pto.ptr<T, gm>
```
- **semantics:** Extract the underlying address view from a `tensor_view` or `partition_tensor_view`.

**Parameter Table:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `%src` | `!pto.tensor_view<...>` or `!pto.partition_tensor_view<...>` | Source view descriptor. |

**Constraints:**

- The result type must be either the lowered memref view or a GM pointer `!pto.ptr<T, gm>` to the same underlying storage.
- The op is pure and does not move data.

**Example:**

```mlir
%base = pto.tensor_view_addr %tv : !pto.tensor_view<?x?xf32> -> !pto.ptr<f32, gm>
```

`pto.tensor_view_addr` exposes the underlying address represented by the view descriptor. When the result type is a memref, it exposes the lowered view directly. When the result type is `!pto.ptr<..., gm>`, it exposes the same address in pointer form. During compiler-internal lowering, the operand may already be rewritten to a memref form; in that case this op is folded away or rewritten to an equivalent memref-to-ptr cast.

---

## `pto.partition_view`

- **syntax:**
```mlir
%pv = pto.partition_view %tv, offsets = [%o0, %o1], sizes = [%s0, %s1]
    : !pto.tensor_view<...> -> !pto.partition_tensor_view<...>
```
- **semantics:** `result = source[offsets, sizes]` — a logical window on a `tensor_view`. Captures both static and dynamic shapes; does not move data.

**Parameter Table:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `%tv` | `TensorViewType` | Input tensor view. |
| `offsets` | `Variadic<Index>` | Dynamic offsets. |
| `sizes` | `Variadic<Index>` | Dynamic sizes. |

**Constraints:**

- `offsets`/`sizes` counts must match the rank of `source`.

**Example:**

```mlir
%pv = pto.partition_view %tv, offsets = [%off0, %off1], sizes = [%s0, %s1]
    : !pto.tensor_view<1024x512xf16> -> !pto.partition_tensor_view<16x16xf16>
```

---

## `pto.alloc_tile`

- **syntax:**
```mlir
%tb  = pto.alloc_tile : !pto.tile_buf<...>
%tb2 = pto.alloc_tile valid_row = %vr valid_col = %vc : !pto.tile_buf<vec, RxCxT, valid=?x?>
%tb3 = pto.alloc_tile addr = %ad : !pto.tile_buf<...>
```
- **semantics:** Declare the lifetime of a tile buffer. Each call produces an **independent** tile-buffer instance.

**Parameter Table:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `addr` | `Optional<i64>` | Optional start address. If omitted, assigned by the implementation. |
| `valid_row` | `Optional<index>` | Dynamic valid-row count (required when result `v_row = ?`). |
| `valid_col` | `Optional<index>` | Dynamic valid-col count (required when result `v_col = ?`). |

**Constraints:**

- If result `v_row`/`v_col` are dynamic (`?`), the corresponding operands must be present.
- If result `v_row`/`v_col` are static, the corresponding operands must be absent.

**Example:**

```mlir
%tb = pto.alloc_tile : !pto.tile_buf<vec, 16x16xf16>
```

---

## `pto.subset`

- **syntax:**
```mlir
%sub = pto.subset %src[%i, %j] sizes [rows, cols] : !pto.tile_buf<...>
```
- **semantics:** `result = source[offsets]` with static `sizes`. Creates a strided view of a parent tile.

**Parameter Table:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `%src` | `pto.tile_buf` | Parent tile buffer. |
| `offsets` | `Variadic<Index>` | Runtime offsets `[i, j]`. |
| `sizes` | `I64ArrayAttr` | Static shape `[rows, cols]`. |

**Constraints:**

- Boxed-vs-non-boxed behavior is derived from the source's tile config (`blayout`, `slayout`, `fractal`) and element type.
- For non-boxed layouts (`slayout=none_box`), no additional subset-specific structural checks are enforced.
- For boxed layouts:
  - `sizes` must have length 2 and both subset sizes must be positive.
  - Subset sizes must be multiples of the inferred inner boxed shape.
  - `offsets` must have length 2; constant offsets must be non-negative and multiples of the inferred inner boxed shape.
  - Source tile shape must be statically known.
  - For boxed row-major tiles: subset must keep the full source column extent, and the column offset must be the constant `0`.
  - For boxed col-major tiles: subset must keep the full source row extent, and the row offset must be the constant `0`.
- The inferred result reuses the source's element type, address space, and tile config. `valid_shape` is derived from the parent valid shape and constant offsets, or dynamic when offsets are dynamic.

**Example:**

```mlir
%sub = pto.subset %src[%i, %j] sizes [32, 32]
     : !pto.tile_buf<vec, 64x64xf16>
```

---

## `pto.set_validshape`

- **syntax:**
```mlir
pto.set_validshape %src, %valid_row, %valid_col : !pto.tile_buf<vec, RxCxT, valid=?x?>
```
- **semantics:** Update the runtime `v_row`/`v_col` metadata on an existing **dynamic** tile buffer.

**Parameter Table:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `%src` | `pto.tile_buf` | Dynamic rank-2 tile buffer. |
| `%valid_row` | `index` | Runtime valid row count. |
| `%valid_col` | `index` | Runtime valid column count. |

**Constraints:**

- `%src` must be rank-2 and use `v_row = ?` and `v_col = ?` on both dimensions.
- `%src` must be a locally bound tile: an `alloc_tile`/`declare_tile`/frontend `tpop` result, or a `bitcast`/`treshape` view of one. Function arguments and other non-local sources are rejected.
- Tile programs use `pto.tile_buf`; memref forms are a lowering artifact and are not part of this surface.
- Constant `valid_row`/`valid_col` must be non-negative and `<=` the tile's static shape bounds.

**Example:**

```mlir
%src = pto.alloc_tile : !pto.tile_buf<vec, 32x32xf16, valid=?x?>
pto.set_validshape %src, %vr, %vc : !pto.tile_buf<vec, 32x32xf16, valid=?x?>
```

---

## `pto.tile_buf_addr`

- **syntax:**
```mlir
%ub_ptr = pto.tile_buf_addr %tile : !pto.tile_buf<...> -> !pto.ptr<T, vec>
%ub_ref = pto.tile_buf_addr %tile : !pto.tile_buf<...> -> memref<...>
```
- **semantics:** Extract the address of a `pto.tile_buf`'s data region. Returns either a typed PTO pointer (`!pto.ptr<T, space>`) or a memref view, depending on the requested result type. Pure op: no data movement, no pipeline activity.

This op is the **boundary between tile-buffer instructions and pointer-based vector instructions**. Inside a `pto.vecscope` body, use `pto.tile_buf_addr` to materialize a vec-space pointer from a tile handle allocated outside the scope; vector load/store ops such as `pto.vlds` and `pto.vsts` then consume that pointer.

**Parameter Table:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `%tile` | `pto.tile_buf` or tile-bound memref | Tile handle whose data-region address is taken. |

**Results:** `!pto.ptr<T, space>` or `memref<...>`. Memref results use the tile's static shape and address space; pointer results use the tile's element type and memory space (e.g. `vec`).

**Constraints:**

- Result must be either a typed PTO pointer or a memref view; no other result types are accepted.
- When a memref result is requested, the lowered form uses the tile's static shape and address space.
- `pto.tile_buf_addr` is **only legal inside `pto.vecscope` / `pto.strict_vecscope`**. Outside a vector scope, tile handles must be consumed by tile-level ops (`pto.tload`, `pto.tstore`, `pto.tadd`, …) rather than by address extraction. Conversely, tile-level ops must **not** appear inside `pto.vecscope`.

**Example (inside `pto.vecscope`):**

```mlir
%tile = pto.alloc_tile addr = %c0_i64 valid_row = %r
  : !pto.tile_buf<vec, 8x128xf32, valid=?x?>

pto.vecscope {
  %ub = pto.tile_buf_addr %tile
    : !pto.tile_buf<vec, 8x128xf32, valid=?x?> -> !pto.ptr<f32, vec>
  // ... vector-scope loads/stores on %ub ...
}
```

See [`03-vector-load-store.md`](../micro-isa/03-vector-load-store.md) for the pointer-based
vector load/store side of this handoff.
