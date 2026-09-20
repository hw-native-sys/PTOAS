# Tile Buffer 类型

## 概述

`!pto.tile_buf<...>` 是当前 `ptoas` 中最核心的局部存储类型。它直接把局部 tile 计算所需的关键元信息编码进类型本身。

## 语法

```mlir
!pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>
!pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
```

现有样例中也可能看到兼容简写：

```mlir
!pto.tile_buf<vec, 1x64xi32>
```

新文档和新样例更推荐显式 key-value 形式。

## 参数

| 参数 | 类型 | 说明 |
| --- | --- | --- |
| `loc` | 关键字 | 局部位置，如 `vec`、`mat`、`left`、`right`、`acc`、`bias`、`scaling` |
| `dtype` | 元素类型 | tile 中元素的数据类型 |
| `rows` | `int64` | 物理行数 |
| `cols` | `int64` | 物理列数 |
| `v_row` | `int64` 或 `?` | 有效行数；`?` 表示省略有效区域，默认取物理 shape |
| `v_col` | `int64` 或 `?` | 有效列数；`?` 表示省略有效区域，默认取物理 shape |
| `blayout` | 布局助记符 | 基础布局：`row_major` 或 `col_major` |
| `slayout` | 布局助记符 | 次级布局：`none_box`、`row_major` 或 `col_major` |
| `fractal` | `int32` | 标称分形块大小（字节，按 f32 计）；仅在 `slayout` 非 `none_box`（boxed）时生效，`none_box` 下忽略；合法值为 `32`、`512`、`1024`，具体操作可进一步限制 |
| `pad` | 助记符或整数 | padding 策略：`null`（0）、`zero`（1）、`max`（2）、`min`（3）；整数编码不是任意填充值 |

字段省略时的默认值：

| 参数 | 默认值 |
| --- | --- |
| `v_row` / `v_col` | `?`（有效区域 = 物理 shape） |
| `blayout` | `row_major` |
| `slayout` | `none_box` |
| `fractal` | `512` |
| `pad` | `0`（`null`） |

## 类型承载的信息

`tile_buf` 同时表达：

- tile 位于哪一类本地存储位置
- tile 的元素类型
- tile 的物理尺寸
- tile 的有效区域
- tile 的布局和 padding 语义

这使很多位置、布局和有效区域相关检查能够更早在类型层面完成。

## 常见构造路径

用户程序通过 `pto.alloc_tile` 创建 `!pto.tile_buf` 值。`pto.declare_tile`
用于地址稍后分配的低层场景，不是常规 TileOp 编程入口。

## 特殊说明

对于 `dtype=!pto.f4E1M2x2` 和 `dtype=!pto.f4E2M1x2`：

- `rows` / `cols` 描述的是物理打包 extent
- `v_row` / `v_col` 描述的也是物理有效 extent
- 它们不是逻辑标量 FP4 元素个数

## Constraints

- `loc`、`dtype`、布局和尺寸组合必须满足后端支持边界
- `v_row` / `v_col` 不应超过对应物理尺寸
- `fractal` 仅在 boxed（`slayout` 为 `row_major` 或 `col_major`）时生效
- boxed 时内块行列数由 `fractal` 与元素字节数决定：`1024` → 16×16；`32` → 16×2；`512` 且 `row_major` → 16×(32 / 元素字节数)，`col_major` 行列互换；`fractal=512` 还要求元素字节数能整除 32
- boxed 时 `rows` / `cols` 须为内块行列数的整数倍（例外：`loc=vec`、`fractal=32` 或单行 tile 时行数可不对齐）
- 具体操作还会进一步限制输入输出 `tile_buf` 的位置和布局组合

## Example

```mlir
%tile = pto.alloc_tile
  : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
```
