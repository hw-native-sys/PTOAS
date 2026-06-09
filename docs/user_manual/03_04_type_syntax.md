# 3.4 类型语法

## 1. 范围

本页只说明 PTO 类型在文本中的出现位置和书写模式，不重复类型语义、参数表和构造方式的完整定义。

## 2. PTO 类型的文本前缀

PTO 自定义类型使用标准 MLIR 方言类型语法，统一以前缀 `!pto.` 开始，例如：

- `!pto.ptr<f16>`
- `!pto.tensor_view<1024x512xf16>`
- `!pto.partition_tensor_view<16x16xf16>`
- `!pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16, ...>`

## 3. 类型出现的位置

### 3.1 函数签名

```mlir
func.func @kernel(%src: !pto.ptr<f16>, %dst: !pto.ptr<f16>) {
  return
}
```

### 3.2 SSA 结果类型

```mlir
%tv = pto.make_tensor_view %arg0,
  shape = [%c32, %c32],
  strides = [%c32, %c1]
  : !pto.tensor_view<?x?xf16>
```

### 3.3 源类型到结果类型

```mlir
%pv = pto.partition_view %tv,
  offsets = [%c0, %c0],
  sizes = [%c16, %c16]
  : !pto.tensor_view<?x?xf16> -> !pto.partition_tensor_view<16x16xf16>
```

### 3.4 `ins(...)` / `outs(...)`

```mlir
pto.tload ins(%pv : !pto.partition_tensor_view<16x16xf16>)
          outs(%tile : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16, v_row=16, v_col=16,
                                     blayout=row_major, slayout=none_box, fractal=512, pad=0>)
```

## 4. 常见书写模式

### 4.1 形状类类型

- `!pto.tensor_view<d0x...xdnxtype>`
- `!pto.partition_tensor_view<d0x...xdnxtype>`

其中：

- `?` 表示动态维度
- 末尾元素类型使用标准 MLIR 标量类型或 PTO 自定义低精度类型

### 4.2 `tile_buf` 键值形式

`tile_buf` 采用键值对文本格式，而不是简单的位置参数列表。例如：

```mlir
!pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16, v_row=16, v_col=16,
              blayout=row_major, slayout=none_box, fractal=512, pad=0>
```

这种形式用于显式承载：

- 位置
- 元素类型
- 物理尺寸
- 有效尺寸
- 布局和填充信息

## 5. Constraints

- 类型必须满足本手册描述的文本格式要求
- 使用在同一操作签名中的类型必须满足该操作的语义约束
- 本页只描述“怎么写”，不替代第四章对“是什么”和“如何构造”的定义
