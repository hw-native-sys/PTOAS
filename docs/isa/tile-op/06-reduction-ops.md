# 6. Reduction Operations

> **Category:** Tile-local VEC reductions
> **Pipeline:** PIPE_V

This chapter documents the TileLib reduction families. These ops reduce one or more source dimensions into smaller destination tiles and are organized into row-reduction and column-reduction groups.

---

## 6.1 Row Reductions

Row reductions reduce each row of `%src` into one element stored at `%dst[row, 0]`. The op shape carries a scratch tile operand `%tmp` to keep the operand list aligned with the A2/A3 PTO instruction interface (see [§1.7](01-tile-overview.md#17-scratch-operands-and-a2a3-compatibility)).

### Common Syntax

```mlir
pto.<op> ins(%src, %tmp : !pto.tile_buf<...>, !pto.tile_buf<...>)
         outs(%dst : !pto.tile_buf<...>)
```

| Op | Semantics |
|----|-----------|
| `pto.trowsum` | `dst[i, 0] = sum_j src[i, j]` |
| `pto.trowprod` | `dst[i, 0] = prod_j src[i, j]` |
| `pto.trowmax` | `dst[i, 0] = max_j src[i, j]` |
| `pto.trowmin` | `dst[i, 0] = min_j src[i, j]` |
| `pto.trowargmax` | `dst[i, 0] = argmax_j src[i, j]` |
| `pto.trowargmin` | `dst[i, 0] = argmin_j src[i, j]` |

**Parameter Table:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `src` | `pto.tile_buf` | Source tile buffer. |
| `tmp` | `pto.tile_buf` | Scratch tile for A2/A3 interface compatibility. |
| `dst` | `pto.tile_buf` | Destination tile storing one result per source row. |

**Constraints:**

- `dst.v_row` should match `src.v_row`.
- `dst.v_col` should be `1`.
- For `pto.trowsum`, `pto.trowprod`, `pto.trowmax`, and `pto.trowmin`, PTO IR keeps `%tmp` for ABI/workspace plumbing but does not impose extra `%tmp` element-type, shape, or layout constraints beyond requiring a `loc=vec` tile.
- `pto.trowargmax` and `pto.trowargmin` require an integer destination element type for the row-local index result.
- Numeric widening / narrowing inside the reduction is target-defined by the selected template (e.g. `trowsum` may widen `i16` accumulation internally before storing to `dst`).
- On A5, `pto.trowmax` and `pto.trowmin` additionally accept 8-bit integer sources (`i8`/`si8`/`ui8`); the A5 TileLib template widens each 8-bit row losslessly to 16 bits for the vector reduction, so the stored `dst` result is exact for min/max. `pto.trowsum` and `pto.trowprod` still require 16/32-bit integers, `f16`, or `f32` sources.

**Example:**

```mlir
pto.trowsum ins(%src, %tmp : !pto.tile_buf<vec, 16x32xf32>, !pto.tile_buf<vec, 16x32xf32>)
            outs(%dst : !pto.tile_buf<vec, 16x1xf32>)
```

---

## 6.2 Column Reductions

Column reductions reduce each column of `%src` into one element stored at `%dst[0, col]`.

### Common Syntax

```mlir
pto.<op> ins(%src : !pto.tile_buf<...>)
         outs(%dst : !pto.tile_buf<...>)
```

| Op | Semantics |
|----|-----------|
| `pto.tcolsum` | `dst[0, j] = sum_i src[i, j]` |
| `pto.tcolprod` | `dst[0, j] = prod_i src[i, j]` |
| `pto.tcolmax` | `dst[0, j] = max_i src[i, j]` |
| `pto.tcolmin` | `dst[0, j] = min_i src[i, j]` |

**Constraints:**

- `dst.v_row` should be `1`.
- `dst.v_col` should match `src.v_col`.
- Templates assume prefix-aligned valid regions and row-major VEC tiles.

**Example:**

```mlir
pto.tcolsum ins(%src : !pto.tile_buf<vec, 16x16xf32>)
            outs(%dst : !pto.tile_buf<vec, 1x16xf32>)
```
