# 11. Selection Operations

> **Category:** Tile-local VEC selection compute
> **Pipeline:** PIPE_V

This chapter documents the TileLib selection families. These ops select between data sources under control of a packed predicate-mask tile.

The mask tile carries packed predicate bytes in UB. Templates load predicate bits directly with predicate-load helpers such as `plds`, then use `vsel` to choose the data path.

`pto.tsel` and `pto.tsels` carry an extra `%tmp` operand for A2/A3 interface compatibility (see [§1.7](01-tile-overview.md#17-scratch-operands-and-a2a3-compatibility)).

---

## 11.1 `pto.tsel`

- **syntax:**
```mlir
pto.tsel ins(%mask, %src0, %src1, %tmp :
             !pto.tile_buf<...>, !pto.tile_buf<...>,
             !pto.tile_buf<...>, !pto.tile_buf<...>)
         outs(%dst : !pto.tile_buf<...>)
```
- **semantics:** `dst[i, j] = mask[i, j] ? src0[i, j] : src1[i, j]`.

**Parameter Table:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `mask` | `pto.tile_buf` | VEC (UB) tile buffer of 8-bit integers carrying packed predicate bits. |
| `src0` | `pto.tile_buf` | Value selected when the predicate bit is true. |
| `src1` | `pto.tile_buf` | Value selected when the predicate bit is false. |
| `tmp` | `pto.tile_buf` | Scratch tile for A2/A3 interface compatibility. |
| `dst` | `pto.tile_buf` | Destination tile buffer. |

**Constraints:**

- `src0`, `src1`, and `dst` must have the same shape and valid region.
- The IR verifier accepts `i8`, `si8`, or `ui8` mask elements. Each mask byte holds eight selection bits.
- The A5 TileLib template specializes on the `i8` mask element type.

**Example:**

```mlir
pto.tsel ins(%mask, %a, %b, %tmp :
             !pto.tile_buf<vec, 16x32xi8, valid=16x2>, !pto.tile_buf<vec, 16x16xf16>,
             !pto.tile_buf<vec, 16x16xf16>, !pto.tile_buf<vec, 1x8xi32>)
         outs(%dst : !pto.tile_buf<vec, 16x16xf16>)
```

---

## 11.2 `pto.tsels`

- **syntax:**
```mlir
pto.tsels ins(%mask, %src, %tmp, %scalar :
              !pto.tile_buf<...>, !pto.tile_buf<...>,
              !pto.tile_buf<...>, <scalar_type>)
          outs(%dst : !pto.tile_buf<...>)
```
- **semantics:** `dst[i, j] = mask[i, j] ? src[i, j] : scalar`.

**Parameter Table:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `mask` | `pto.tile_buf` | Packed predicate-mask carrier. |
| `src` | `pto.tile_buf` | Source tile selected when the predicate bit is true. |
| `tmp` | `pto.tile_buf` | Scratch tile for A2/A3 interface compatibility. |
| `scalar` | signless integer / floating-point scalar | Scalar selected when the predicate bit is false. |
| `dst` | `pto.tile_buf` | Destination tile buffer. |

**Constraints:**

- `src` and `dst` must have the same shape and valid region.
- `tsels` accepts packed-mask carrier tiles with `i8`, `i16`, or `i32` element types.

**Example:**

```mlir
pto.tsels ins(%mask, %src, %tmp, %scalar :
              !pto.tile_buf<vec, 16x16xi8>, !pto.tile_buf<vec, 16x16xf16>,
              !pto.tile_buf<vec, 16x16xf16>, f16)
          outs(%dst : !pto.tile_buf<vec, 16x16xf16>)
```
