# 4.2 元素类型

## 1. 范围

本页描述 PTO 类型系统中的元素类型类别、文本写法和使用约束，不重复指针、视图和 tile buffer 的完整定义。

## 2. 概述

元素类型描述存储对象中每个标量单元的数值解释方式。它们通常作为 `!pto.ptr<...>`、`!pto.tensor_view<...>`、`!pto.partition_tensor_view<...>` 和 `!pto.tile_buf<...>` 的参数出现，而不是单独承担复杂对象语义。

## 3. 常见类别

### 3.1 通用整数类型

- `i1`
- `i8`
- `i16`
- `i32`
- `i64`

### 3.2 通用浮点类型

- `f16`
- `f32`
- `bf16`

### 3.3 低精度扩展类型

- `f8E4M3FN`
- `f8E5M2`
- `!pto.hif8`
- `!pto.f4E1M2x2`
- `!pto.f4E2M1x2`

## 4. PTO 自定义低精度类型

### `!pto.hif8`

- 每个元素 1 byte
- 作为元素类型嵌入更高层 PTO 类型中使用

### `!pto.f4E1M2x2`

- 每个打包对 1 byte
- 适合作为 `tile_buf.dtype` 的低精度元素类型

### `!pto.f4E2M1x2`

- 每个打包对 1 byte
- 与 `!pto.f4E1M2x2` 类似，也是打包 FP4 对类型

对于打包 FP4 类型，`tile_buf` 中的尺寸描述物理打包后的 extent，而不是逻辑标量元素个数。

## 5. 常见出现位置

### 5.1 作为指针元素类型

```mlir
!pto.ptr<f16>
!pto.ptr<!pto.hif8>
```

### 5.2 作为视图元素类型

```mlir
!pto.tensor_view<128x128xf16>
!pto.partition_tensor_view<32x32x!pto.hif8>
```

### 5.3 作为 tile buffer 的 `dtype`

```mlir
!pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>
!pto.tile_buf<loc=vec, dtype=!pto.f4E1M2x2, rows=16, cols=32, v_row=16, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
```

## 6. Constraints

- 元素类型被 parser 接受，不等于所有 PTO 操作都支持它
- 逐元素、规约、转换、位操作等对元素类型有各自额外限制
- 布局、地址空间和位置不会改变元素类型本身的数值语义

## 7. Example

```mlir
!pto.ptr<!pto.hif8>
!pto.tensor_view<128x128x!pto.hif8>
!pto.tile_buf<loc=vec, dtype=!pto.f4E2M1x2, rows=16, cols=32, v_row=16, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
```
