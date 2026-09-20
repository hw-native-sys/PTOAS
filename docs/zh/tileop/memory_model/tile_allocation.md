# Tile 分配

## 概述

在当前 `ptoas` 的 构建层级下，tile 不是纯值语义对象，而是带显式生命周期的本地缓冲对象。tile 分配模型关注的是：什么时候产生一个 tile、该 tile 是否参与地址规划、以及它能否与其他 tile 复用同一块本地内存。

## 基本对象

tile 分配模型主要围绕 `!pto.tile_buf<...>` 展开。一个 tile buffer 至少带有：

- 地址空间
- 元素类型
- 物理尺寸
- 有效尺寸
- 布局 / pad 配置

因此，tile 分配并不是只决定“有没有一个 buffer”，还隐含决定了这个 buffer 属于哪类本地资源。

## 典型生命周期

常见路径如下：

1. `pto.alloc_tile`
2. 参与 `tload`、`tmov`、`tmatmul`、`tgemv`、`tadd` 等操作
3. 根据需要参与 `tstore`、`tpush`、`tpop` 或后续别名关系传播
4. 生命周期结束，或与其他 buffer 参与复用 / inplace 合并

在文本层面，这常表现为：

```mlir
%tile = pto.alloc_tile
  : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32,
                  blayout=row_major, slayout=none_box, fractal=512, pad=0>
```

## 分配模型中的几类关系

### 独立分配

最直接的情况是每个 `alloc_tile` 对应独立的本地缓冲需求，由地址规划为它分配独立偏移。

### 别名 / 重解释关系

某些操作表达的不是“新分配一块新内存”，而是已有 tile 的重解释或子视图关系，例如：

- `treshape`
- 与子区域相关的 tile alias 语义

这类关系会影响后续内存规划与冲突分析。

### inplace / 复用关系

`pto-plan-memory` 不只做简单顺序分配，还会分析生命周期和语义冲突，决定哪些 buffer 可以复用同一块存储。

因此，tile 分配模型本质上是“生命周期分析 + 地址空间分组 + 复用约束”的组合。其中涉及的具体实现细节不属于稳定接口，用户程序不应依赖。

## 自动分配与显式控制的边界

当前 `ptoas` 中，前端或上游 IR 可以显式给出 tile 的类型、位置、布局和 alias 关系；而真正的本地地址分配、偏移选择和多数复用决策通常由编译阶段完成。

也就是说：

- 语义对象由 IR 显式声明
- 具体地址和复用关系由编译器在后续地址规划阶段决定

## Constraints

- tile 的地址空间、布局和元素类型必须满足后续操作约束
- 允许 alias 或 reshape 的 tile 关系不能破坏语义一致性和内存规划前提
- 动态有效区域会影响 buffer 的使用边界，但不改变其所属地址空间
- 某些场景下，为了满足目标相关限制，编译器会收紧 inplace 或复用策略
