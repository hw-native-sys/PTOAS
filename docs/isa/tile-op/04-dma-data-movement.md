# 4. DMA Data Movement

> **Category:** GM↔on-chip DMA for tile buffers
> **Pipelines:** PIPE_MTE2 (GM→UB), PIPE_MTE3 (UB→GM), PIPE_FIX (when source is `loc=acc`)

This chapter documents the public tile DMA instructions `pto.tload`, `pto.tstore`, and the L1-to-L0A feature-map transfer `pto.timg2col`. Other raw scalar load/store helpers are outside the current tile-instruction subset and are not covered here.

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
- `l2_bypass` is supported on A2/A3 and A5.
- The target implementation owns the architecture-specific bypass mechanism.

**Pipeline:** `PIPE_MTE2`.

**Example:**

```mlir
pto.tload ins(%pv : !pto.partition_tensor_view<16x16xf16>)
          outs(%tb : !pto.tile_buf<vec, 16x16xf16>)
```

When `cache_policy` is absent or `default`, the target uses its ordinary cache
allocation behavior. `l2_bypass` requests no L2 allocation for this transfer
without changing the logical source address.

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


## `pto.timg2col`

A2/A3 `TIMG2COL` unfolds a feature map from Mat (L1) to Left (L0A), on
`PIPE_MTE1`. Both operands are rank-2 tile buffers with the same element type
(`f16`, `bf16`, `f32`, or `i8`). The source is a fully valid NZ tile `[H*W, C]`;
its packed bytes are `NC1HWC0` when `H*W` is divisible by 16 and C is divisible
by `C0=32/sizeof(dtype)`. Both tiles use 512-byte fractals. The destination
uses `blayout=row_major, slayout=row_major`.

```mlir
pto.timg2col ins(%src, %pos_m, %pos_k : !pto.tile_buf<mat, 64x32xf16,
    blayout=col_major, slayout=row_major>, index, index)
  outs(%dst : !pto.tile_buf<left, 16x32xf16,
    blayout=row_major, slayout=row_major>)
  {fmap_h=8 : i64, fmap_w=8 : i64, kernel_h=3 : i64, kernel_w=3 : i64,
   pad_top=1 : i64, pad_bottom=1 : i64, pad_left=1 : i64, pad_right=1 : i64}
```

`fmap_h/w` and `kernel_h/w` are required. `stride_h/w` and `dilation_h/w`
default to 1; `pad_top/bottom/left/right` default to 0. Padding supplies zero.
The lowering creates a ConvTile descriptor aliasing the existing L1 buffer,
sets its geometry, and invokes `TIMG2COL` with `FMATRIX_A_AUTO`, configuring
FMATRIX, repeat and padding registers on every call.

For destination `[M,K]`, the unfolded row axis enumerates output H/W and the
column axis enumerates C1/kernel H/kernel W/C0. Matmul weights must use that
same K order. M is divisible by 16 and K by C0. `pos_m` and `pos_k` select the
window; both must fit uint16 and keep the whole destination window in range.
`pos_k` must be C0-aligned. Runtime source valid dimensions must equal its
physical dimensions. Static violations are rejected by the verifier.

Image H/W and C fit uint16; kernel H/W are in [1,511]; stride and dilation
are in [1,255]; each padding value is in [0,255]. Destination M/K fit uint16.
This primitive handles spatial convolution. A temporal kernel for causal
3-D convolution needs an outer loop and accumulation over temporal slices.
