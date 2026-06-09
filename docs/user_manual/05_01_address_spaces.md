# 5.1 地址空间

## 1. 概述

地址空间用于描述 PTO 对象位于哪一类存储域中。它既是类型系统的一部分，也是同步分析、内存规划和代码生成的重要输入。

在 `ptoas` 中，地址空间通过 `#pto.address_space<...>` 枚举属性建模，并常以内嵌字段的形式出现在 `tile_buf`、`memref` 或相关对象中。

## 2. 地址空间枚举

当前常见的主要地址空间包括：

- `zero`
- `gm`
- `mat`
- `left`
- `right`
- `acc`
- `vec`
- `bias`
- `scaling`

其中：

- `gm` 表示全局内存
- `vec` 常对应向量侧本地缓冲
- `mat`、`left`、`right`、`acc`、`bias` 常对应矩阵或矩阵计算相关本地存储域
- `scaling` 用于某些缩放或量化辅助路径

## 3. 地址空间出现的位置

### 3.1 `tile_buf` 类型

地址空间最常见的文本呈现方式，是 `tile_buf` 内部的 `loc` 字段：

```mlir
!pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>
!pto.tile_buf<loc=acc, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>
```

### 3.2 视图或降级后的内存对象

在编译过程中，地址空间也可能出现在 `memref` 的 memory space 或专用属性中，用于保留对象所在的物理域。

### 3.3 保留缓冲区与本地地址规划

`pto.reserve_buffer` 显式带有 `location` 字段，表示保留区域属于哪一个本地地址空间：

```mlir
pto.reserve_buffer {name = "pipe0", size = 1024, location = #pto.address_space<vec>, auto = true} -> i32
```

## 4. 地址空间的作用

### 4.1 约束操作合法性

很多 PTO 操作不是只看元素类型和 shape，还要求对象位于正确的地址空间。例如：

- 某些逐元素操作要求输入输出都在 `vec`
- 某些矩阵计算路径要求操作数分别位于 `left` / `right` / `acc`
- 某些数据搬运路径要求源和目的地址空间组合满足后端支持边界

### 4.2 驱动同步分析

同步分析不会把所有内存对象一视同仁。地址空间是依赖分析中的核心维度之一：

- `GM` 依赖通常与全局访存顺序相关
- 本地地址空间更直接影响本地 pipeline 之间的 set / wait / barrier 决策

### 4.3 驱动内存规划

本地地址规划不是对所有 buffer 统一分配一个大地址池，而是按地址空间分别规划：

- `vec` 的 buffer 在 `vec` 容量内规划
- `acc` 的 buffer 在 `acc` 容量内规划
- `reserve_buffer` 也只会在其声明的目标地址空间内找空洞

## 5. Constraints

- 地址空间不是装饰性信息；它会直接参与目标相关检查
- 同一个操作中，输入输出地址空间组合必须满足该操作的实现约束
- `reserve_buffer` 的 `location` 必须是受支持的本地地址空间，否则 `pto-plan-memory` 会报错
- 某些目标相关编译阶段会针对特定地址空间组合做调整或规避
