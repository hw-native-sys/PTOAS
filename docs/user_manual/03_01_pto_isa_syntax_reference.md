# 3.1 PTO ISA 语法总览

## 1. 范围

本文档给出 `ptoas` 中 `PTO ISA` 文本形式的总览性语法说明，覆盖：

- 顶层 `module` 与 `func.func`
- SSA 值、block、region
- PTO 类型、属性与操作在文本中的出现位置
- PTO 常见自定义 assembly 形式
- 一个最小可读的完整 PTO 程序骨架

本页只描述第三章的总体组织和全局规则。类型本体的定义、参数和构造方式参见第四章；具体操作语法参见第六章。

## 2. 文本结构总览

PTO ISA 建立在标准 MLIR 文本格式之上。一个 PTO 程序通常按如下层次组织：

1. `module`
2. `func.func`
3. block / region
4. SSA 值
5. PTO 类型、属性与操作

最小结构如下：

```mlir
module {
  func.func @empty() {
    return
  }
}
```

其中：

- `module` 是顶层编译单元
- `func.func` 是函数符号定义
- `{ ... }` 是 region
- `return` 是函数终结操作

## 3. 语法对象在 PTO 程序中的分工

### 3.1 模块与函数

- `module` 承载符号表、顶层属性和函数集合
- `func.func` 承载一个可验证、可调度、可变换的 PTO 过程体
- 函数参数通常承载全局指针、标量控制参数或运行时辅助对象

详见：[3.2 模块与函数语法](./03_02_module_and_function.md)

### 3.2 SSA 与 Region

- PTO 操作之间通过 SSA 值连接
- block 参数表示控制流边界上的值传递
- `scf`、`cf`、`arith` 等标准 dialect 可与 PTO 操作混用

详见：[3.3 SSA 值与 Region](./03_03_ssa_and_region.md)

### 3.3 类型

- PTO 自定义类型直接写在函数签名、SSA 值、`ins(...)` / `outs(...)` 和结果类型标注中
- `!pto.ptr<...>`、`!pto.tensor_view<...>`、`!pto.partition_tensor_view<...>`、`!pto.tile_buf<...>` 是最常见的文本类型

详见：[3.4 类型语法](./03_04_type_syntax.md) 和 [PTO ISA 类型系统总览](./04_01_pto_isa_type_system_reference.md)

### 3.4 属性

- 属性既可以作为独立 op attribute 出现，也可以内嵌在某些自定义语法中
- PTO 方言枚举属性通常使用 `#pto.*<...>` 形式

详见：[3.5 属性语法](./03_05_attribute_syntax.md)

### 3.5 Operation Assembly

- PTO 既使用通用 MLIR 语法，也广泛使用自定义 assembly format
- 数据搬运、矩阵计算、同步等操作通常采用 `ins(...)` / `outs(...)` 风格

详见：[3.6 Operation 汇编格式](./03_06_operation_assembly.md)

## 4. 常见文本模式

### 4.1 函数参数中的 PTO 类型

```mlir
func.func @kernel(%src: !pto.ptr<f16>, %dst: !pto.ptr<f16>, %n: i32) {
  return
}
```

### 4.2 SSA 结果绑定

```mlir
%c0 = arith.constant 0 : index
%tile = pto.alloc_tile
  : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16, v_row=16, v_col=16,
                  blayout=row_major, slayout=none_box, fractal=512, pad=0>
```

### 4.3 View 到 Tile 的数据流

```mlir
%tv = pto.make_tensor_view %arg0,
  shape = [%c16, %c16],
  strides = [%c16, %c1]
  : !pto.tensor_view<?x?xf16>

%pv = pto.partition_view %tv,
  offsets = [%c0, %c0],
  sizes = [%c16, %c16]
  : !pto.tensor_view<?x?xf16> -> !pto.partition_tensor_view<16x16xf16>

pto.tload ins(%pv : !pto.partition_tensor_view<16x16xf16>)
          outs(%tile : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16, v_row=16, v_col=16,
                                     blayout=row_major, slayout=none_box, fractal=512, pad=0>)
```

## 5. 全局约束

- PTO 文本必须满足标准 MLIR 的符号、SSA 和 region 规则
- PTO 自定义类型和属性必须满足本手册定义的文本写法
- 自定义 assembly 只是文本形式；最终合法性仍由语义约束决定
- 第六章中的很多操作对位置、布局、有效区域和元素类型还有额外约束

## 6. 最小完整示例

```mlir
module {
  func.func @vec_add_kernel(%arg0: !pto.ptr<f32>,
                            %arg1: !pto.ptr<f32>,
                            %arg2: !pto.ptr<f32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index

    %src0 = pto.make_tensor_view %arg0,
      shape = [%c32, %c32],
      strides = [%c32, %c1]
      : !pto.tensor_view<?x?xf32>

    %src1 = pto.make_tensor_view %arg1,
      shape = [%c32, %c32],
      strides = [%c32, %c1]
      : !pto.tensor_view<?x?xf32>

    %dst = pto.make_tensor_view %arg2,
      shape = [%c32, %c32],
      strides = [%c32, %c1]
      : !pto.tensor_view<?x?xf32>

    %pv0 = pto.partition_view %src0,
      offsets = [%c0, %c0],
      sizes = [%c32, %c32]
      : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<32x32xf32>

    %pv1 = pto.partition_view %src1,
      offsets = [%c0, %c0],
      sizes = [%c32, %c32]
      : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<32x32xf32>

    %pv2 = pto.partition_view %dst,
      offsets = [%c0, %c0],
      sizes = [%c32, %c32]
      : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<32x32xf32>

    %t0 = pto.alloc_tile
      : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32,
                      blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %t1 = pto.alloc_tile
      : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32,
                      blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %t2 = pto.alloc_tile
      : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32,
                      blayout=row_major, slayout=none_box, fractal=512, pad=0>

    pto.tload ins(%pv0 : !pto.partition_tensor_view<32x32xf32>)
              outs(%t0 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32,
                                       blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.tload ins(%pv1 : !pto.partition_tensor_view<32x32xf32>)
              outs(%t1 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32,
                                       blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.tadd ins(%t0, %t1
              : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32,
                              blayout=row_major, slayout=none_box, fractal=512, pad=0>,
                !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32,
                              blayout=row_major, slayout=none_box, fractal=512, pad=0>)
             outs(%t2 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32,
                                      blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.tstore ins(%t2 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32,
                                       blayout=row_major, slayout=none_box, fractal=512, pad=0>)
               outs(%pv2 : !pto.partition_tensor_view<32x32xf32>)
    return
  }
}
```
