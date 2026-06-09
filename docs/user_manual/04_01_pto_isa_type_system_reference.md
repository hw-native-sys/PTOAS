# 4.1 PTO ISA 类型系统总览

## 1. 范围

本文档给出 `ptoas` 中 `PTO ISA` 类型系统的总体说明，覆盖：

- 类型家族划分
- 各类 PTO 类型在 IR 中的职责
- 常见构造路径
- 类型之间的关系
- 典型使用链路

本页是第四章的总入口。各具体类型的细分说明参见 4.2 到 4.6。

## 2. 文档索引

- [4.2 元素类型](./04_02_element_types.md)
- [4.3 指针类型](./04_03_pointer_types.md)
- [4.4 Tensor View 类型](./04_04_tensor_view_types.md)
- [4.5 Tile Buffer 类型](./04_05_tile_buffer_types.md)
- 4.6 Pipe 与 Event 类型

## 3. 类型系统组成

PTO ISA 的类型系统由两部分组成：

1. MLIR 内建标量类型
2. PTO 自定义类型

其中 PTO 自定义类型又可分为以下几类：

- 元素类型扩展
- 指针类型
- 视图类型
- Tile 类型与 Tile Buffer 类型
- Pipe / Event 句柄类型

## 4. 类型家族划分

### 4.1 元素类型

元素类型描述存储对象中每个标量单元的数值解释方式，常见于：

- `i1`、`i8`、`i16`、`i32`、`i64`
- `f16`、`f32`、`bf16`
- `f8E4M3FN`、`f8E5M2`
- `!pto.hif8`
- `!pto.f4E1M2x2`
- `!pto.f4E2M1x2`

详见：[4.2 元素类型](./04_02_element_types.md)

### 4.2 指针类型

`!pto.ptr<T>` 表示全局内存入口，是很多 PTO 程序的外部输入边界。

详见：[4.3 指针类型](./04_03_pointer_types.md)

### 4.3 视图类型

视图类型描述“如何看待一段全局内存”：

- `!pto.tensor_view<...>` 表示全局张量视图
- `!pto.partition_tensor_view<...>` 表示分区视图

详见：[4.4 Tensor View 类型](./04_04_tensor_view_types.md)

### 4.4 Tile 与 Tile Buffer 类型

- `!pto.tile<...>` 表示更抽象的 tile 形状对象
- `!pto.tile_buf<...>` 表示当前 `ptoas` 中更核心的局部 tile buffer 对象

其中 `tile_buf` 直接编码位置、布局、物理尺寸、有效尺寸和 padding 等信息。

详见：[4.5 Tile Buffer 类型](./04_05_tile_buffer_types.md)

### 4.5 Pipe / Event 句柄类型

这组类型用于表达同步、流水线和事件相关对象，例如：

- `!pto.pipe`
- `!pto.eventid_array<N>`

详见：4.6 Pipe 与 Event 类型。

## 5. 常见构造路径

### 5.1 从函数边界引入

函数参数最常引入：

- `!pto.ptr<T>`
- 标量整数 / 浮点类型

```mlir
func.func @kernel(%arg0: !pto.ptr<f32>, %arg1: i32) {
  return
}
```

### 5.2 从 PTO 操作结果引入

常见路径包括：

- `pto.make_tensor_view` 生成 `!pto.tensor_view<...>`
- `pto.partition_view` 生成 `!pto.partition_tensor_view<...>`
- `pto.alloc_tile` 生成 `!pto.tile_buf<...>`
- declare / initialize 类操作生成本地辅助对象或句柄类型

### 5.3 作为其他类型的嵌套参数出现

低精度元素类型通常不会单独作为复杂对象存在，而是嵌入到以下类型中：

- `!pto.ptr<!pto.hif8>`
- `!pto.tensor_view<...x!pto.hif8>`
- `!pto.tile_buf<..., dtype=!pto.f4E1M2x2, ...>`

## 6. 类型关系

PTO 程序中常见的一条类型关系链如下：

1. 用 `!pto.ptr<T>` 接收全局内存入口
2. 通过 `pto.make_tensor_view` 构造 `!pto.tensor_view<...>`
3. 通过 `pto.partition_view` 构造 `!pto.partition_tensor_view<...>`
4. 通过 `pto.alloc_tile` 构造 `!pto.tile_buf<...>`
5. 用 `pto.tload` / `pto.tstore` 在 view 与 tile buffer 之间移动数据

这条链路把“全局地址空间中的对象”和“局部 tile 计算对象”区分开来，是 PTO 类型系统的核心使用方式。

## 7. 典型示例

```mlir
module {
  func.func @example(%arg0: !pto.ptr<f32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index

    %tv = pto.make_tensor_view %arg0,
      shape = [%c32, %c32],
      strides = [%c32, %c1]
      : !pto.tensor_view<?x?xf32>

    %pv = pto.partition_view %tv,
      offsets = [%c0, %c0],
      sizes = [%c32, %c32]
      : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<32x32xf32>

    %tile = pto.alloc_tile
      : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32,
                      blayout=row_major, slayout=none_box, fractal=512, pad=0>

    pto.tload ins(%pv : !pto.partition_tensor_view<32x32xf32>)
              outs(%tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32,
                                         blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    return
  }
}
```

## 8. 全局约束

- 类型的文本格式必须满足 `pto` 方言 parser / printer 要求
- 某个类型被定义出来，不等于所有操作都接受它
- 元素类型、位置、布局、有效尺寸和句柄类别的真正合法性由具体操作语义决定
- 某些类型主要用于特定使用场景，公开使用时应以本手册约束和示例为准
