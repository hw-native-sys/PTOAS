# 3.5 属性语法

## 1. 范围

本页描述 PTO 文本中属性的主要写法、出现位置和约束，不重复类型中已经内嵌承载的键值字段定义。

## 2. 属性的两种主要承载位置

### 2.1 作为操作语法中的属性

属性可以直接写在操作语法内部，用于表达模式、策略或枚举语义，例如：

```mlir
pto.tcvt ins(%src {rmode = #pto<round_mode FLOOR>, satmode = #pto<saturation_mode ON>}
         : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16, v_row=16, v_col=16,
                         blayout=row_major, slayout=none_box, fractal=512, pad=0>)
         outs(%dst : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16, v_row=16, v_col=16,
                                   blayout=row_major, slayout=none_box, fractal=512, pad=0>)
```

### 2.2 作为类型内部字段

有些 PTO 语义并不以独立 attribute 形式出现，而是直接写入类型文本，例如 `tile_buf` 内部的 `loc`、`blayout`、`slayout`、`pad` 等键值。

例如：

```mlir
%tile = pto.alloc_tile
  : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16, v_row=16, v_col=16,
                  blayout=row_major, slayout=none_box, fractal=512, pad=0>
```

## 3. 常见文本形式

### 3.1 PTO 枚举属性

```mlir
#pto<round_mode FLOOR>
#pto<saturation_mode ON>
```

### 3.2 内建属性

PTO 操作也可能使用 MLIR 内建属性作为参数承载形式，例如整数、布尔、字符串、数组等。

例如：

```mlir
%buf = pto.reserve_buffer {
  name = "pipe0",
  size = 1024,
  auto = true
} -> i32
```

其中：

- `name = "pipe0"` 是字符串属性
- `size = 1024` 是整数属性
- `auto = true` 是布尔属性

## 4. 语法位置

属性在 PTO 程序中通常出现在以下位置：

- 自定义操作的尾部命名字段
- 通用 op attribute 字典
- `module` 等顶层容器的 attribute 字典
- 某些类型的内部键值配置

## 5. Constraints

- 属性名、属性个数和属性类型由具体操作定义决定
- PTO 枚举属性的可选 token 集合由方言属性定义固定，不可任意扩展文本取值
- 文本写法正确后，仍需满足属性组合本身的语义约束

## 6. Example

```mlir
pto.tcvt ins(%src {rmode = #pto<round_mode FLOOR>, satmode = #pto<saturation_mode ON>}
         : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16, v_row=16, v_col=16,
                         blayout=row_major, slayout=none_box, fractal=512, pad=0>)
         outs(%dst : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16, v_row=16, v_col=16,
                                   blayout=row_major, slayout=none_box, fractal=512, pad=0>)
```
