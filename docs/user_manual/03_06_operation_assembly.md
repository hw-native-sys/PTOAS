# 3.6 Operation 汇编格式

## 1. 范围

本页描述 PTO 操作在文本中的常见 assembly 形式，重点是“怎么读、怎么写”。各具体操作的参数语义和约束以第六章对应页面为准。

## 2. 通用构成

一个 PTO 操作的文本通常由以下部分组成：

1. 可选的 SSA 结果绑定
2. 操作名
3. 操作数
4. 可选属性
5. 类型标注

最基础的形式如下：

```mlir
%r = pto.some_op %arg0, %arg1 : type0, type1 -> type2
```

## 3. 常见 PTO 自定义 assembly 模式

### 3.1 直接操作数 + 类型

```mlir
%p1 = pto.addptr %p0, %off : !pto.ptr<f16>, index -> !pto.ptr<f16>
```

适用于指针、标量和部分轻量级辅助操作。

### 3.2 View 构造模式

```mlir
%tv = pto.make_tensor_view %arg0,
  shape = [%c32, %c32],
  strides = [%c32, %c1]
  : !pto.tensor_view<?x?xf16>
```

该模式通常显式列出 shape、stride、offset 等结构化参数。

### 3.3 源类型到结果类型

```mlir
%pv = pto.partition_view %tv,
  offsets = [%c0, %c0],
  sizes = [%c16, %c16]
  : !pto.tensor_view<?x?xf16> -> !pto.partition_tensor_view<16x16xf16>
```

该模式适合表达“从一种视图描述符派生出另一种视图描述符”的操作。

### 3.4 `ins(...)` / `outs(...)` 模式

```mlir
pto.tload ins(%pv : !pto.partition_tensor_view<16x16xf16>)
          outs(%tile : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16, v_row=16, v_col=16,
                                     blayout=row_major, slayout=none_box, fractal=512, pad=0>)
```

```mlir
pto.tadd ins(%lhs, %rhs
          : !pto.tile_buf<...>, !pto.tile_buf<...>)
         outs(%dst : !pto.tile_buf<...>)
```

这是 PTO 最常见的 tile 级操作书写模式，强调输入对象与目的缓冲区的分离。

### 3.5 带枚举属性的模式

```mlir
pto.syncall() mode = #pto.sync_all_mode<hard>, core_type = #pto.sync_core_type<mix>
```

适用于同步、运行时控制和部分策略型操作。

## 4. 阅读要点

- 是否有 SSA 结果，决定该操作是“产出新值”还是“写入既有对象”
- `ins(...)` / `outs(...)` 中的类型往往同时编码了位置、布局和有效区域约束
- `source -> result` 形式通常表示描述符转换，而非实际数据搬运
- 自定义 assembly 只是一层文本外观，真实合法性仍由语义约束决定

## 5. Constraints

- 文本形式必须满足本手册描述的操作书写规则
- 操作数个数、类型和属性必须与具体 op 定义一致
- 某些操作支持多种 assembly 变体时，应在同一模块内保持写法一致
