# **资源绑定操作**

本章介绍 PTO ISA 中的资源绑定操作。这些操作负责 Tile 缓冲区的逻辑分配、地址绑定以及 FIFO 管线槽位的分配与释放。

---

## 目录

- [`pto.tassign` — Tile 地址重绑定](#ptotassign--tile-地址重绑定)
- [`pto.alloc_tile` — 逻辑 Tile 缓冲区分配](#ptoalloc_tile--逻辑-tile-缓冲区分配)

---

## 操作详解

### `pto.tassign` — Tile 地址重绑定

```
pto.tassign <tile>, <addr> : <tile_type> -> <result_type>
```

**语义：**

```
result = rebind(tile, addr)
// tile handle 保持原有的形状和类型元数据，仅更新底层地址为 addr
```

**参数:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `tile` | `!pto.tile_buf<...>` 或其他 PTO DPS 类型 | 待重绑定的 tile handle |
| `addr` | `i64` | 新的运行时地址 |

**返回值:** 与 `tile` 操作数相同类型的 tile handle，绑定到新地址。

**约束：**

- **实现检查 (A2A3/A5)**
  - 结果类型必须与 `tile` 操作数类型完全匹配。
  - 纯操作（`Pure` trait），无副作用。

**示例:**

```mlir
%tile = pto.declare_tile -> !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
                                          v_row=16, v_col=16, blayout=row_major,
                                          slayout=none_box, fractal=512, pad=0>
%addr = arith.constant 4096 : i64
%rebound = pto.tassign %tile, %addr
    : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
                    v_row=16, v_col=16, blayout=row_major,
                    slayout=none_box, fractal=512, pad=0>
    -> !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
                     v_row=16, v_col=16, blayout=row_major,
                     slayout=none_box, fractal=512, pad=0>
```

---

### `pto.alloc_tile` — 逻辑 Tile 缓冲区分配

```
pto.alloc_tile (addr = <addr>)? (valid_row = <vr>)? (valid_col = <vc>)?
               : <result_type>
```

**语义：**

```
result = allocate_tile_buffer(addr?, valid_row?, valid_col?)
// 创建一个逻辑 tile buffer handle，可选指定固定地址和动态有效维度
```

**参数:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `addr` | `i64`（可选） | 固定的缓冲区起始地址 |
| `valid_row` | `index`（可选） | 运行时有效行数，当结果类型的 `v_row` 为动态（`?`）时必须提供 |
| `valid_col` | `index`（可选） | 运行时有效列数，当结果类型的 `v_col` 为动态（`?`）时必须提供 |

**返回值:** `!pto.tile_buf<...>` — 分配得到的 tile buffer handle。

**约束：**

- **实现检查 (A2A3/A5)**
  - 结果 tile_buf 必须具有 rank-2 的 validShape。
  - 当结果类型的 `v_row` 为动态（`?`）时，必须提供 `valid_row` 操作数；当 `v_row` 为静态时，不得提供。
  - 当结果类型的 `v_col` 为动态（`?`）时，必须提供 `valid_col` 操作数；当 `v_col` 为静态时，不得提供。
  - 结果 tile_buf 的 layout 约束必须合法（通过 `verifyTileBufLayoutConstraints` 验证）。

**示例:**

```mlir
// 静态 valid shape，无需 valid_row/valid_col
%t0 = pto.alloc_tile addr = %c0_i64
    : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=32,
                    v_row=16, v_col=32, blayout=row_major,
                    slayout=none_box, fractal=512, pad=0>

// 动态 valid shape，需提供 valid_row 和 valid_col
%t1 = pto.alloc_tile addr = %c0_i64
                     valid_row = %c16 valid_col = %c256
    : !pto.tile_buf<loc=acc, dtype=f32, rows=16, cols=256,
                    v_row=?, v_col=?, blayout=col_major,
                    slayout=row_major, fractal=1024, pad=0>
```