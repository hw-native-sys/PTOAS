# 4.5 Tile Buffer 类型

## 1. 概述

`!pto.tile_buf<...>` 是当前 `ptoas` 中最核心的局部存储类型。它直接把局部 tile 计算所需的关键元信息编码进类型本身。

## 2. 语法

```mlir
!pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>
!pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
```

现有样例中也可能看到兼容简写：

```mlir
!pto.tile_buf<vec, 1x64xi32>
```

新文档和新样例更推荐显式 key-value 形式。

## 3. 参数

| 参数 | 类型 | 说明 |
| --- | --- | --- |
| `loc` | 关键字 | 局部位置，如 `vec`、`mat`、`left`、`right`、`acc`、`bias` |
| `dtype` | 元素类型 | tile 中元素的数据类型 |
| `rows` | `int64` | 物理行数 |
| `cols` | `int64` | 物理列数 |
| `v_row` | `int64` 或 `?` | 有效行数 |
| `v_col` | `int64` 或 `?` | 有效列数 |
| `blayout` | 布局助记符 | 基础布局 |
| `slayout` | 布局助记符 | 次级布局 |
| `fractal` | `int32` | 分形相关参数 |
| `pad` | 助记符或整数 | padding 策略或值 |

## 4. 类型承载的信息

`tile_buf` 同时表达：

- tile 位于哪一类本地存储位置
- tile 的元素类型
- tile 的物理尺寸
- tile 的有效区域
- tile 的布局和 padding 语义

这使很多位置、布局和有效区域相关检查能够更早在类型层面完成。

## 5. 常见构造路径

- `pto.alloc_tile`
- `pto.bind_tile`
- `pto.materialize_tile`
- `pto.declare_tile`

其中最常见的是 `pto.alloc_tile`。

## 6. 特殊说明

对于 `dtype=!pto.f4E1M2x2` 和 `dtype=!pto.f4E2M1x2`：

- `rows` / `cols` 描述的是物理打包 extent
- `v_row` / `v_col` 描述的也是物理有效 extent
- 它们不是逻辑标量 FP4 元素个数

## 7. Constraints

- `loc`、`dtype`、布局和尺寸组合必须满足后端支持边界
- `v_row` / `v_col` 不应超过对应物理尺寸
- 具体操作还会进一步限制输入输出 `tile_buf` 的位置和布局组合

## 8. Example

```mlir
%tile = pto.alloc_tile
  : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
```
