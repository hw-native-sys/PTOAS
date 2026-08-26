# 4. DMA Data Movement

> **Category:** GM↔on-chip DMA for tile buffers
> **Pipelines:** PIPE_MTE2 (GM→UB), PIPE_MTE3 (UB→GM), PIPE_FIX (when source is `loc=acc`)

This chapter documents the public tile DMA instructions `pto.tload` and `pto.tstore`. Other raw scalar load/store helpers are outside the current tile-instruction subset and are not covered here.

---

## `pto.tload`

- **syntax:**
```mlir
pto.tload ins(%src : !pto.partition_tensor_view<...>)
          outs(%dst : !pto.tile_buf<...>)
          {cache_policy = #pto.load_cache_policy<l2_bypass>}
```
- **semantics:** Physical DMA transfer from a global partition view into a local tile buffer. For each element `(i, j)` in the destination valid region: `dst[i, j] = src[i, j]`.

**Parameter Table:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `src` | `PartitionTensorViewType` | Source partition view. |
| `dst` | `pto.tile_buf` | Destination tile buffer. |
| `cache_policy` | `LoadCachePolicyAttr` (optional) | `default` when absent; `l2_bypass` requests a non-allocating L2 path for this load. |

**Constraints:**

- Tile element type ∈ `{i8, i16, i32, i64, f16, bf16, f32}`.
- Destination tile must use `loc=vec`.
- Destination tile element type and source partition element type must have the same bitwidth.
- Runtime: source partition extents and destination valid region must be positive.
- `l2_bypass` is currently supported on A2/A3 and rejected on A5.
- PTO-ISA owns the target-specific bypass mechanism; PTOAS emits no address-alias constant.

**Pipeline:** `PIPE_MTE2`.

**Example:**

```mlir
pto.tload ins(%pv : !pto.partition_tensor_view<16x16xf16>)
          outs(%tb : !pto.tile_buf<vec, 16x16xf16>)
```

When `cache_policy` is absent or `default`, PTOAS emits the legacy
`TLOAD(dst, src)` call. `l2_bypass` emits
`TLOAD<pto::LoadCachePolicy::L2Bypass>(dst, src)`.

---

## `pto.tstore`

- **syntax:**
```mlir
pto.tstore ins(%src : !pto.tile_buf<...>)
           outs(%dst : !pto.partition_tensor_view<...>)
```
- **semantics:** Store a 2-D tile buffer back to a 2-D partition view. For each element `(i, j)` in the source valid region: `dst[i, j] = src[i, j]`.

**Parameter Table:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `src` | `pto.tile_buf` | Source tile buffer. |
| `dst` | `PartitionTensorViewType` | Destination partition view. |

**Constraints:**

- `src` must be `!pto.tile_buf`, `dst` must be `!pto.partition_tensor_view`.
- Static dst shape dims and static src valid-shape dims must be positive.
- `src.loc ∈ {vec, mat, acc}`.
- For `loc=vec` / `loc=mat`: src element type ∈ `{i8, i16, i32, i64, f16, bf16, f32}`; src/dst element bitwidth must match.
- For `loc=acc`:
  - src element type must be `i32` or `f32`.
  - dst element type ∈ `{i32, f32, f16, bf16}`.

**Pipeline:**

- `src.loc=acc` uses **PIPE_FIX**.
- `src.loc=vec` / `src.loc=mat` uses **PIPE_MTE3**.

**Example:**

```mlir
pto.tstore ins(%tb : !pto.tile_buf<vec, 16x16xf16>)
           outs(%pv : !pto.partition_tensor_view<16x16xf16>)
```
