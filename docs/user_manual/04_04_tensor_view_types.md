# 4.4 Tensor View 类型

## 1. 概述

PTO 通过两种视图类型描述全局内存中的张量对象：

- `!pto.tensor_view<...>`：完整张量视图
- `!pto.partition_tensor_view<...>`：切片后的分区视图

它们都不拥有数据，而是描述“如何解释某段全局内存”。

## 2. `!pto.tensor_view<...>`

### 语法

```mlir
!pto.tensor_view<64xi32>
!pto.tensor_view<32x32xf32>
!pto.tensor_view<?x?xf16>
```

### 角色

- 表示全局张量对象的逻辑 shape
- 作为后续分区、加载和存储操作的上游描述符

### 常见构造路径

- `pto.make_tensor_view`

## 3. `!pto.partition_tensor_view<...>`

### 语法

```mlir
!pto.partition_tensor_view<64xi32>
!pto.partition_tensor_view<32x32xf32>
!pto.partition_tensor_view<?x16xf16>
```

### 角色

- 表示从 tensor view 中切出的逻辑子区域
- 作为 `tload`、`tprefetch`、`tstore` 等操作的直接输入或输出描述符

### 常见构造路径

- `pto.partition_view`
- `pto.subview`

## 4. 视图类型与指针、Tile 的关系

- 指针只表达地址入口
- `tensor_view` 在指针之上补充 shape / stride 级语义
- `partition_tensor_view` 进一步把全局张量收缩到一个局部可操作区域
- 真正进入局部计算域时，通常再通过 `tload` 转成 `tile_buf`

## 5. Constraints

- 视图类型本身不拥有底层存储
- 视图类型的元素类型必须与其来源地址或目标 tile 语义兼容
- 分区视图的 shape、offset 和 size 必须满足源 view 的边界约束

## 6. Example

```mlir
%tv = pto.make_tensor_view %arg0,
  shape = [%c32, %c32],
  strides = [%c32, %c1]
  : !pto.tensor_view<?x?xf32>

%pv = pto.partition_view %tv,
  offsets = [%c0, %c0],
  sizes = [%c32, %c32]
  : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<32x32xf32>
```
