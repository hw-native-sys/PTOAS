# **轴规约与广播操作**

本节描述了 PTO ISA 中沿行或沿列进行规约（reduction）和广播（broadcast）的全部操作。所有操作均作用于本地缓冲区（`tile_buf`，位于 `loc=vec` 空间），采用"目标传递风格"（Destination-Passing Style, DPS）：操作本身不产生 SSA 返回值，而是直接将结果写入预先分配好的目标 `tile_buf`。全部操作执行在 **Vector 流水线**（`PIPE_V`）上。

这一类操作通常具有如下装配形式：

```mlir
pto.op ins(%src : !pto.tile_buf<...>)
       outs(%dst : !pto.tile_buf<...>)
```

通用约束通常包括：

- 所有 tile 必须位于 `loc=vec`（VEC/UB 存储空间）
- 输入 tile 使用 ND-style 布局（`blayout=row_major`, `slayout=none_box`）
- 输入与输出元素类型一致（`argmax`/`argmin` 除外，其输出为整数索引类型）

---

## 目录

- [`pto.tcolexpand` — 列广播](#ptotcolexpand--列广播)
- [`pto.tcolmax` — 列最大值规约](#ptotcolmax--列最大值规约)
- [`pto.tcolargmax` — 列最大值索引规约](#ptotcolargmax--列最大值索引规约)
- [`pto.tcolmin` — 列最小值规约](#ptotcolmin--列最小值规约)
- [`pto.tcolargmin` — 列最小值索引规约](#ptotcolargmin--列最小值索引规约)
- [`pto.tcolsum` — 列求和规约](#ptotcolsum--列求和规约)
- [`pto.trowexpand` — 行广播](#ptotrowexpand--行广播)
- [`pto.trowmax` — 行最大值规约](#ptotrowmax--行最大值规约)
- [`pto.trowargmax` — 行最大值索引规约](#ptotrowargmax--行最大值索引规约)
- [`pto.trowmin` — 行最小值规约](#ptotrowmin--行最小值规约)
- [`pto.trowargmin` — 行最小值索引规约](#ptotrowargmin--行最小值索引规约)
- [`pto.trowsum` — 行求和规约](#ptotrowsum--行求和规约)
- [`pto.tcolprod` — 列乘积规约](#ptotcolprod--列乘积规约)
- [`pto.trowprod` — 行乘积规约](#ptotrowprod--行乘积规约)
- [`pto.trowexpandsub` — 行广播减法](#ptotrowexpandsub--行广播减法)
- [`pto.trowexpandmul` — 行广播乘法](#ptotrowexpandmul--行广播乘法)
- [`pto.trowexpanddiv` — 行广播除法](#ptotrowexpanddiv--行广播除法)
- [`pto.tcolexpandmax` — 列广播取最大值](#ptotcolexpandmax--列广播取最大值)
- [`pto.tcolexpandmin` — 列广播取最小值](#ptotcolexpandmin--列广播取最小值)
- [`pto.tcolexpandmul` — 列广播乘法](#ptotcolexpandmul--列广播乘法)
- [`pto.tcolexpandadd` — 列广播加法](#ptotcolexpandadd--列广播加法)
- [`pto.tcolexpanddiv` — 列广播除法](#ptotcolexpanddiv--列广播除法)
- [`pto.tcolexpandexpdif` — 列广播指数差](#ptotcolexpandexpdif--列广播指数差)
- [`pto.tcolexpandsub` — 列广播减法](#ptotcolexpandsub--列广播减法)
- [`pto.trowexpandadd` — 行广播加法](#ptotrowexpandadd--行广播加法)
- [`pto.trowexpandexpdif` — 行广播指数差](#ptotrowexpandexpdif--行广播指数差)
- [`pto.trowexpandmax` — 行广播取最大值](#ptotrowexpandmax--行广播取最大值)
- [`pto.trowexpandmin` — 行广播取最小值](#ptotrowexpandmin--行广播取最小值)

---

## 操作详解

### `pto.tcolexpand` — 列广播

```
pto.tcolexpand ins(<src> : <src_type>) outs(<dst> : <dst_type>)
```

**语义：**

```
For each element (i, j):
    dst[i, j] = src[0, j]
```

**参数:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | 源 tile，行向量，每列携带一个逻辑标量 |
| `dst` | `pto.tile_buf` | 目标 tile |

**返回值:** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查 (A2A3/A5)**
  - `src` 和 `dst` 必须使用 `loc=vec`
  - `src` 和 `dst` 必须使用 ND-style 布局（`blayout=row_major`, `slayout=none_box`）
  - 元素类型一致：`dst_type == src_type`
  - 数据类型：`i8`、`i16`、`i32`、`f16`、`bf16`、`f32`
  - `src valid column == dst valid column`

**示例:**

```mlir
pto.tcolexpand ins(%src : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=16,
                   v_row=1, v_col=16, blayout=row_major, slayout=none_box,
                   fractal=512, pad=0>)
               outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                   v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                   fractal=512, pad=0>)
```

---

### `pto.tcolmax` — 列最大值规约

```
pto.tcolmax ins(<src> : <src_type>) outs(<dst> : <dst_type>)
```

**语义：**

```
For each column j:
    dst[0, j] = max over i of src[i, j]
```

**参数:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | 源 tile |
| `dst` | `pto.tile_buf` | 目标 tile，行向量，存储每列的最大值 |

**返回值:** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查 (A2A3)**
  - `src` 和 `dst` 必须使用 `loc=vec`
  - 所有 tile 使用 ND-style 布局（`blayout=row_major`, `slayout=none_box`）
  - 数据类型：`f16`、`f32`、`i16`、`i32`
  - 元素类型一致：`dst_type == src_type`
  - `src valid column == dst valid column`

- **实现检查 (A5)**
  - `src` 和 `dst` 必须使用 `loc=vec`
  - 所有 tile 使用 ND-style 布局（`blayout=row_major`, `slayout=none_box`）
  - 数据类型：`i8`、`i16`、`i32`、`f16`、`bf16`、`f32`
  - 元素类型一致：`dst_type == src_type`
  - `src valid row` 和 `src valid column` 必须非零
  - `src valid column == dst valid column`

**示例:**

```mlir
pto.tcolmax ins(%src : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
                v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                fractal=512, pad=0>)
            outs(%dst : !pto.tile_buf<loc=vec, dtype=f16, rows=1, cols=16,
                v_row=1, v_col=16, blayout=row_major, slayout=none_box,
                fractal=512, pad=0>)
```

---

### `pto.tcolargmax` — 列最大值索引规约

```
pto.tcolargmax ins(<src>, <tmp> : <src_type>, <tmp_type>)
               outs(<dst> : <dst_type>)
```

**语义：**

```
For each column j:
    dst[0, j] = argmax over i of src[i, j]
```

**参数:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | 源 tile |
| `tmp` | `pto.tile_buf` | 临时缓冲区，与 `src` 同 shape 和元素类型 |
| `dst` | `pto.tile_buf` | 目标 tile，存储每列最大值的行索引 |

**返回值:** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查 (A2A3/A5)**
  - `src`、`tmp`、`dst` 必须使用 `loc=vec`
  - 所有 tile 使用 ND-style 布局（`blayout=row_major`, `slayout=none_box`）
  - `tmp` 必须与 `src` 具有相同的 shape、valid shape 和元素类型
  - `src` 元素类型必须为 `f16` 或 `f32`
  - `dst` 元素类型必须为 `i32` 或 `ui32`
  - `src valid row != 0` 且 `src valid column != 0`
  - `dst valid row == 1`
  - `src valid column == dst valid column`

**示例:**

```mlir
pto.tcolargmax ins(%src, %tmp : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=32,
                   v_row=16, v_col=32, blayout=row_major, slayout=none_box,
                   fractal=512, pad=0>,
                   !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=32,
                   v_row=16, v_col=32, blayout=row_major, slayout=none_box,
                   fractal=512, pad=0>)
               outs(%dst : !pto.tile_buf<loc=vec, dtype=ui32, rows=1, cols=32,
                   v_row=1, v_col=32, blayout=row_major, slayout=none_box,
                   fractal=512, pad=0>)
```

---

### `pto.tcolmin` — 列最小值规约

```
pto.tcolmin ins(<src> : <src_type>) outs(<dst> : <dst_type>)
```

**语义：**

```
For each column j:
    dst[0, j] = min over i of src[i, j]
```

**参数:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | 源 tile |
| `dst` | `pto.tile_buf` | 目标 tile，行向量，存储每列的最小值 |

**返回值:** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查 (A2A3)**
  - `src` 和 `dst` 必须使用 `loc=vec`
  - 所有 tile 使用 ND-style 布局（`blayout=row_major`, `slayout=none_box`）
  - 数据类型：`f16`、`f32`、`i16`、`i32`
  - 元素类型一致：`dst_type == src_type`
  - `src valid column == dst valid column`

- **实现检查 (A5)**
  - `src` 和 `dst` 必须使用 `loc=vec`
  - 所有 tile 使用 ND-style 布局（`blayout=row_major`, `slayout=none_box`）
  - 数据类型：`i8`、`i16`、`i32`、`f16`、`bf16`、`f32`
  - 元素类型一致：`dst_type == src_type`
  - `src valid row` 和 `src valid column` 必须非零
  - `src valid column == dst valid column`

**示例:**

```mlir
pto.tcolmin ins(%src : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
                v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                fractal=512, pad=0>)
            outs(%dst : !pto.tile_buf<loc=vec, dtype=f16, rows=1, cols=16,
                v_row=1, v_col=16, blayout=row_major, slayout=none_box,
                fractal=512, pad=0>)
```

---

### `pto.tcolargmin` — 列最小值索引规约

```
pto.tcolargmin ins(<src>, <tmp> : <src_type>, <tmp_type>)
               outs(<dst> : <dst_type>)
```

**语义：**

```
For each column j:
    dst[0, j] = argmin over i of src[i, j]
```

**参数:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | 源 tile |
| `tmp` | `pto.tile_buf` | 临时缓冲区，与 `src` 同 shape 和元素类型 |
| `dst` | `pto.tile_buf` | 目标 tile，存储每列最小值的行索引 |

**返回值:** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查 (A2A3/A5)**
  - `src`、`tmp`、`dst` 必须使用 `loc=vec`
  - 所有 tile 使用 ND-style 布局（`blayout=row_major`, `slayout=none_box`）
  - `tmp` 必须与 `src` 具有相同的 shape、valid shape 和元素类型
  - `src` 元素类型必须为 `f16` 或 `f32`
  - `dst` 元素类型必须为 `i32` 或 `ui32`
  - `src valid row != 0` 且 `src valid column != 0`
  - `dst valid row == 1`
  - `src valid column == dst valid column`

**示例:**

```mlir
pto.tcolargmin ins(%src, %tmp : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=32,
                   v_row=16, v_col=32, blayout=row_major, slayout=none_box,
                   fractal=512, pad=0>,
                   !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=32,
                   v_row=16, v_col=32, blayout=row_major, slayout=none_box,
                   fractal=512, pad=0>)
               outs(%dst : !pto.tile_buf<loc=vec, dtype=i32, rows=1, cols=32,
                   v_row=1, v_col=32, blayout=row_major, slayout=none_box,
                   fractal=512, pad=0>)
```

---

### `pto.tcolsum` — 列求和规约

```
pto.tcolsum ins(<src>, <tmp> : <src_type>, <tmp_type>)
            outs(<dst> : <dst_type>) isBinary = false
```

**语义：**

```
For each column j:
    dst[0, j] = sum over i of src[i, j]
```

**参数:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | 源 tile |
| `tmp` | `pto.tile_buf` | 临时缓冲区，用于中间计算 |
| `dst` | `pto.tile_buf` | 目标 tile，行向量，存储每列的求和结果 |

**返回值:** 无。以 DPS 的形式写入 `dst`。

**属性:**

- `isBinary` — 是否使用二叉规约树。默认值为 `false`。
  - `true` — 使用二叉规约树
  - `false` — 使用默认规约方式

**约束：**

- **实现检查 (A2A3)**
  - `src`、`tmp`、`dst` 必须使用 `loc=vec`
  - 所有 tile 使用 ND-style 布局（`blayout=row_major`, `slayout=none_box`）
  - 数据类型：`f16`、`f32`、`i16`、`i32`
  - 元素类型一致：`dst_type == tmp_type == src_type`
  - `src valid column == dst valid column`

- **实现检查 (A5)**
  - `src`、`tmp`、`dst` 必须使用 `loc=vec`
  - 所有 tile 使用 ND-style 布局（`blayout=row_major`, `slayout=none_box`）
  - 数据类型：`i8`、`i16`、`i32`、`f16`、`bf16`、`f32`
  - 元素类型一致：`dst_type == tmp_type == src_type`
  - `src valid row` 和 `src valid column` 必须非零
  - `src valid column == dst valid column`

**示例:**

```mlir
pto.tcolsum ins(%src, %tmp : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                fractal=512, pad=0>,
                !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                fractal=512, pad=0>)
            outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=16,
                v_row=1, v_col=16, blayout=row_major, slayout=none_box,
                fractal=512, pad=0>) isBinary = false
```

---

### `pto.trowexpand` — 行广播

```
pto.trowexpand ins(<src> : <src_type>) outs(<dst> : <dst_type>)
```

**语义：**

```
For each element (i, j):
    dst[i, j] = src[i, 0]
```

**参数:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | 源 tile，列向量，每行携带一个逻辑标量 |
| `dst` | `pto.tile_buf` | 目标 tile |

**返回值:** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查 (A2A3/A5)**
  - `src` 和 `dst` 必须使用 `loc=vec`
  - `src` 必须使用 `slayout=none_box`
  - `dst` 必须使用 ND-style 布局（`blayout=row_major`, `slayout=none_box`）
  - 元素类型一致：`dst_type == src_type`
  - 数据类型：`i8`、`i16`、`i32`、`f16`、`bf16`、`f32`
  - `src valid row == dst valid row`
  - `src valid row != 0` 且 `src valid column != 0` 且 `dst valid row != 0` 且 `dst valid column != 0`

**示例:**

```mlir
pto.trowexpand ins(%src : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=1,
                   v_row=16, v_col=1, blayout=row_major, slayout=none_box,
                   fractal=512, pad=0>)
               outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                   v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                   fractal=512, pad=0>)
```

---

### `pto.trowmax` — 行最大值规约

```
pto.trowmax ins(<src> : <src_type>) outs(<dst> : <dst_type>)
```

**语义：**

```
For each row i:
    dst[i, 0] = max over j of src[i, j]
```

**参数:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | 源 tile |
| `dst` | `pto.tile_buf` | 目标 tile，列向量，存储每行的最大值 |

**返回值:** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查 (A2A3/A5)**
  - `src` 和 `dst` 必须使用 `loc=vec`
  - `src` 使用 ND-style 布局（`blayout=row_major`, `slayout=none_box`）
  - `dst` 布局：推荐使用 DN-style 1D 列向量（`cols=1`, `blayout=col_major`）；也兼容 ND-style 2D tile（`valid column == 1`）
  - 数据类型：`i16`、`i32`、`f16`、`f32`
  - 元素类型一致：`src_type == dst_type`
  - `src valid column != 0` 且 `src valid row != 0`
  - `src valid row == dst valid row`

**示例:**

```mlir
pto.trowmax ins(%src : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
                v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                fractal=512, pad=0>)
            outs(%dst : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=1,
                v_row=16, v_col=1, blayout=row_major, slayout=none_box,
                fractal=512, pad=0>)
```

---

### `pto.trowargmax` — 行最大值索引规约

```
pto.trowargmax ins(<src>, <tmp> : <src_type>, <tmp_type>)
               outs(<dst> : <dst_type>)
```

**语义：**

```
For each row i:
    dst[i, 0] = argmax over j of src[i, j]
```

**参数:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | 源 tile |
| `tmp` | `pto.tile_buf` | 临时缓冲区，与 `src` 同 shape 和元素类型 |
| `dst` | `pto.tile_buf` | 目标 tile，存储每行最大值的列索引 |

**返回值:** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查 (A2A3/A5)**
  - `src`、`tmp`、`dst` 必须使用 `loc=vec`
  - `src` 使用 ND-style 布局（`blayout=row_major`, `slayout=none_box`）
  - `tmp` 必须与 `src` 具有相同的 shape、valid shape 和元素类型
  - `dst` 使用 `slayout=none_box`，且为 DN-style 列向量（`blayout=col_major`, `cols=1`）或 ND-style tile（`valid column == 1`）
  - `src` 元素类型：`i16`、`i32`、`f16`、`f32`
  - `dst` 元素类型：`i32` 或 `ui32`
  - `src valid row != 0` 且 `src valid column != 0`
  - `src valid row == dst valid row`
  - `dst valid column == 1`

**示例:**

```mlir
pto.trowargmax ins(%src, %tmp : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=32,
                   v_row=16, v_col=32, blayout=row_major, slayout=none_box,
                   fractal=512, pad=0>,
                   !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=32,
                   v_row=16, v_col=32, blayout=row_major, slayout=none_box,
                   fractal=512, pad=0>)
               outs(%dst : !pto.tile_buf<loc=vec, dtype=ui32, rows=16, cols=1,
                   v_row=16, v_col=1, blayout=col_major, slayout=none_box,
                   fractal=512, pad=0>)
```

---

### `pto.trowmin` — 行最小值规约

```
pto.trowmin ins(<src>, <tmp> : <src_type>, <tmp_type>)
            outs(<dst> : <dst_type>)
```

**语义：**

```
For each row i:
    dst[i, 0] = min over j of src[i, j]
```

**参数:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | 源 tile |
| `tmp` | `pto.tile_buf` | 临时缓冲区，用于中间计算 |
| `dst` | `pto.tile_buf` | 目标 tile，列向量，存储每行的最小值 |

**返回值:** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查 (A2A3/A5)**
  - `src` 和 `dst` 必须使用 `loc=vec`
  - `src` 使用 ND-style 布局（`blayout=row_major`, `slayout=none_box`）
  - `dst` 布局：推荐使用 DN-style 1D 列向量（`cols=1`, `blayout=col_major`）；也兼容 ND-style 2D tile（`valid column == 1`）
  - 数据类型：`i16`、`i32`、`f16`、`f32`
  - 元素类型一致：`src_type == dst_type`
  - `src valid column != 0` 且 `src valid row != 0`
  - `src valid row == dst valid row`

**示例:**

```mlir
pto.trowmin ins(%src, %tmp : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
                v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                fractal=512, pad=0>,
                !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
                v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                fractal=512, pad=0>)
            outs(%dst : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=1,
                v_row=16, v_col=1, blayout=row_major, slayout=none_box,
                fractal=512, pad=0>)
```

---

### `pto.trowargmin` — 行最小值索引规约

```
pto.trowargmin ins(<src>, <tmp> : <src_type>, <tmp_type>)
               outs(<dst> : <dst_type>)
```

**语义：**

```
For each row i:
    dst[i, 0] = argmin over j of src[i, j]
```

**参数:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | 源 tile |
| `tmp` | `pto.tile_buf` | 临时缓冲区，与 `src` 同 shape 和元素类型 |
| `dst` | `pto.tile_buf` | 目标 tile，存储每行最小值的列索引 |

**返回值:** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查 (A2A3/A5)**
  - `src`、`tmp`、`dst` 必须使用 `loc=vec`
  - `src` 使用 ND-style 布局（`blayout=row_major`, `slayout=none_box`）
  - `tmp` 必须与 `src` 具有相同的 shape、valid shape 和元素类型
  - `dst` 使用 `slayout=none_box`，且为 DN-style 列向量（`blayout=col_major`, `cols=1`）或 ND-style tile（`valid column == 1`）
  - `src` 元素类型：`i16`、`i32`、`f16`、`f32`
  - `dst` 元素类型：`i32` 或 `ui32`
  - `src valid row != 0` 且 `src valid column != 0`
  - `src valid row == dst valid row`
  - `dst valid column == 1`

**示例:**

```mlir
pto.trowargmin ins(%src, %tmp : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=32,
                   v_row=16, v_col=32, blayout=row_major, slayout=none_box,
                   fractal=512, pad=0>,
                   !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=32,
                   v_row=16, v_col=32, blayout=row_major, slayout=none_box,
                   fractal=512, pad=0>)
               outs(%dst : !pto.tile_buf<loc=vec, dtype=ui32, rows=16, cols=1,
                   v_row=16, v_col=1, blayout=col_major, slayout=none_box,
                   fractal=512, pad=0>)
```

---

### `pto.trowsum` — 行求和规约

```
pto.trowsum ins(<src> : <src_type>) outs(<dst> : <dst_type>)
```

**语义：**

```
For each row i:
    dst[i, 0] = sum over j of src[i, j]
```

**参数:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | 源 tile |
| `dst` | `pto.tile_buf` | 目标 tile，列向量，存储每行的求和结果 |

**返回值:** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查 (A2A3/A5)**
  - `src` 和 `dst` 必须使用 `loc=vec`
  - `src` 使用 ND-style 布局（`blayout=row_major`, `slayout=none_box`）
  - `dst` 布局：推荐使用 DN-style 1D 列向量（`cols=1`, `blayout=col_major`）；也兼容 ND-style 2D tile（`valid column == 1`）
  - 数据类型：`i16`、`i32`、`f16`、`f32`
  - 元素类型一致：`src_type == dst_type`
  - `src valid column != 0` 且 `src valid row != 0`
  - `src valid row == dst valid row`

**示例:**

```mlir
pto.trowsum ins(%src : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
                v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                fractal=512, pad=0>)
            outs(%dst : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=1,
                v_row=16, v_col=1, blayout=row_major, slayout=none_box,
                fractal=512, pad=0>)
```

---

### `pto.tcolprod` — 列乘积规约

```
pto.tcolprod ins(<src> : <src_type>) outs(<dst> : <dst_type>)
```

**语义：**

```
For each column j:
    dst[0, j] = product over i of src[i, j]
```

**参数:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | 源 tile |
| `dst` | `pto.tile_buf` | 目标 tile，行向量，存储每列的乘积结果 |

**返回值:** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查 (A2A3)**
  - `src` 和 `dst` 必须使用 `loc=vec`
  - 所有 tile 使用 ND-style 布局（`blayout=row_major`, `slayout=none_box`）
  - 数据类型：`f16`、`f32`、`i16`、`i32`
  - 元素类��一致：`dst_type == src_type`
  - `src valid column == dst valid column`

- **实现检查 (A5)**
  - `src` 和 `dst` 必须使用 `loc=vec`
  - 所有 tile 使用 ND-style 布局（`blayout=row_major`, `slayout=none_box`）
  - 数据类型：`i16`、`ui16`、`i32`、`ui32`、`f16`、`bf16`、`f32`
  - 元素类型一致：`dst_type == src_type`
  - `src valid column == dst valid column`

**示例:**

```mlir
pto.tcolprod ins(%src : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                 v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                 fractal=512, pad=0>)
             outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=16,
                 v_row=1, v_col=16, blayout=row_major, slayout=none_box,
                 fractal=512, pad=0>)
```

---

### `pto.trowprod` — 行乘积规约

```
pto.trowprod ins(<src>, <tmp> : <src_type>, <tmp_type>)
             outs(<dst> : <dst_type>)
```

**语义：**

```
For each row i:
    dst[i, 0] = product over j of src[i, j]
```

**参数:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | 源 tile |
| `tmp` | `pto.tile_buf` | 临时缓冲区，与 `src` 同 shape 和元素类型 |
| `dst` | `pto.tile_buf` | 目标 tile，列向量，存储每行的乘积结果 |

**返回值:** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查 (A2A3/A5)**
  - `src` 和 `dst` 必须使用 `loc=vec`
  - `src` 使用 ND-style 布局（`blayout=row_major`, `slayout=none_box`）
  - `tmp` 必须与 `src` 具有相同的 shape 和元素类型
  - `dst` 布局：推荐使用 DN-style 1D 列向量（`cols=1`, `blayout=col_major`）；也兼容 ND-style 2D tile（`valid column == 1`）
  - 数据类型：`i16`、`i32`、`f16`、`f32`
  - 元素类型一致：`src_type == dst_type`
  - `src valid column != 0` 且 `src valid row != 0`
  - `src valid row == dst valid row`

**示例:**

```mlir
pto.trowprod ins(%src, %tmp : !pto.tile_buf<loc=vec, dtype=i16, rows=16, cols=16,
                 v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                 fractal=512, pad=0>,
                 !pto.tile_buf<loc=vec, dtype=i16, rows=16, cols=16,
                 v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                 fractal=512, pad=0>)
             outs(%dst : !pto.tile_buf<loc=vec, dtype=i16, rows=16, cols=1,
                 v_row=16, v_col=1, blayout=row_major, slayout=none_box,
                 fractal=512, pad=0>)
```

---

### `pto.trowexpandsub` — 行广播减法

```
pto.trowexpandsub ins(<src0>, <src1>[, <tmp>] : <src0_type>, <src1_type>[, <tmp_type>])
                  outs(<dst> : <dst_type>)
```

**语义：**

```
For each element (i, j):
    dst[i, j] = src0[i, j] - src1[i, 0]
```

**参数:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | 主源 tile |
| `src1` | `pto.tile_buf` | 每行标量载体（被减数广播源） |
| `dst` | `pto.tile_buf` | 目标 tile |

**返回值:** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查 (A2A3)**
  - `src0`、`src1`、`dst` 元素类型一致
  - 数据类型：`i16`、`i32`、`f16`、`f32`
  - `dst` 使用 `blayout=row_major`
  - `src0` 与 `dst` 具有相同的 shape 和 valid_shape
  - 可选 `tmp` 操作数：用于 pto-isa 中需要 tmp 的重载

- **实现检查 (A5)**
  - `src0`、`src1`、`dst` 元素类型一致
  - 数据类型：`i8`、`i16`、`i32`、`f16`、`f32`
  - `dst` 使用 `blayout=row_major`
  - `src0` 与 `dst` 具有相同的 shape 和 valid_shape
  - 可选 `tmp` 操作数：用于 pto-isa 中需要 tmp 的重载

**示例:**

```mlir
pto.trowexpandsub ins(%src0, %src1, %tmp : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>,
                      !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=1,
                      v_row=16, v_col=1, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>,
                      !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
                  outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
```

---

### `pto.trowexpandmul` — 行广播乘法

```
pto.trowexpandmul ins(<src0>, <src1>[, <tmp>] : <src0_type>, <src1_type>[, <tmp_type>])
                  outs(<dst> : <dst_type>)
```

**语义：**

```
For each element (i, j):
    dst[i, j] = src0[i, j] * src1[i, 0]
```

**参数:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | 主源 tile |
| `src1` | `pto.tile_buf` | 每行标量载体（乘数广播源） |
| `dst` | `pto.tile_buf` | 目标 tile |

**返回值:** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查 (A2A3/A5)**
  - `src0`、`src1`、`dst` 元素类型一致
  - 数据类型：`f16`、`f32`
  - `dst` 使用 `blayout=row_major`
  - 可选 `tmp` 操作数：用于 pto-isa 中需要 tmp 的重载

**示例:**

```mlir
pto.trowexpandmul ins(%src0, %src1, %tmp : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>,
                      !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=1,
                      v_row=16, v_col=1, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>,
                      !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
                  outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
```

---

### `pto.trowexpanddiv` — 行广播���法

```
// 默认精度
pto.trowexpanddiv ins(<src0>, <src1> : <src0_type>, <src1_type>)
                  outs(<dst> : <dst_type>)

// 高精度（需要 tmp）
pto.trowexpanddiv ins(<src0>, <src1>, <tmp> : <src0_type>, <src1_type>, <tmp_type>)
                  outs(<dst> : <dst_type>)
                  {precisionType = #pto<div_precision high_precision>}
```

**语义：**

```
For each element (i, j):
    dst[i, j] = src0[i, j] / src1[i, 0]
```

**参数:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | 主源 tile（被除数） |
| `src1` | `pto.tile_buf` | 每行标量载体（除数广播源） |
| `dst` | `pto.tile_buf` | 目标 tile |

**返回值:** 无。以 DPS 的形式写入 `dst`。

**属性:**

- `precisionType` — 除法精度模式。默认值为 `#pto<div_precision default>`。
  - `#pto<div_precision default>` — 标准精度除法
  - `#pto<div_precision high_precision>` — 高精度除法，需要浮点元素类型和额外的 `tmp` 操作数

**约束：**

- **实现检查 (A2A3/A5)**
  - `src0`、`src1`、`dst` 元素类型一致
  - 仅支持浮点类型：`f16`、`f32`
  - `dst` 使用 `blayout=row_major`
  - 高精度模式下 `tmp` 操作数必须提供

**示例:**

```mlir
// 默认精度
pto.trowexpanddiv ins(%src0, %src1 : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>,
                      !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=1,
                      v_row=16, v_col=1, blayout=col_major, slayout=none_box,
                      fractal=512, pad=0>)
                  outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)

// 高精���
pto.trowexpanddiv ins(%src0, %src1, %tmp : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>,
                      !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=1,
                      v_row=16, v_col=1, blayout=col_major, slayout=none_box,
                      fractal=512, pad=0>,
                      !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
                  outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
                  {precisionType = #pto<div_precision high_precision>}
```

---

### `pto.tcolexpandmax` — 列广播取最大值

```
pto.tcolexpandmax ins(<src0>, <src1> : <src0_type>, <src1_type>)
                  outs(<dst> : <dst_type>)
```

**语义：**

```
For each element (i, j):
    dst[i, j] = max(src0[i, j], src1[0, j])
```

**参数:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | 主源 tile |
| `src1` | `pto.tile_buf` | 每列标量载体 |
| `dst` | `pto.tile_buf` | 目标 tile |

**返回值:** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查 (A2A3)**
  - `src0`、`src1`、`dst` 元素类型一致
  - 数据类型：`i16`、`i32`、`f16`、`f32`
  - `src0` 与 `dst` 具有相同的 shape 和 valid_shape
  - `src0`、`src1`、`dst` 使用 `blayout=row_major`
  - `src1 valid_shape[1] == dst valid_shape[1]`

- **实现检查 (A5)**
  - `src0`、`src1`、`dst` 元素类型一致
  - 数据类型：`i8`、`i16`、`i32`、`f16`、`f32`
  - `src0` 与 `dst` 具有相同的 shape 和 valid_shape
  - `src0`、`src1`、`dst` 使用 `blayout=row_major`
  - `src1 valid_shape[1] == dst valid_shape[1]`

**示例:**

```mlir
pto.tcolexpandmax ins(%src0, %src1 : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>,
                      !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=16,
                      v_row=1, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
                  outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
```

---

### `pto.tcolexpandmin` — 列广播取最小值

```
pto.tcolexpandmin ins(<src0>, <src1> : <src0_type>, <src1_type>)
                  outs(<dst> : <dst_type>)
```

**语义：**

```
For each element (i, j):
    dst[i, j] = min(src0[i, j], src1[0, j])
```

**参数:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | 主源 tile |
| `src1` | `pto.tile_buf` | 每列标量载体 |
| `dst` | `pto.tile_buf` | 目标 tile |

**返回值:** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查 (A2A3)**
  - `src0`、`src1`、`dst` 元素类型一致
  - 数据类型：`i16`、`i32`、`f16`、`f32`
  - `src0` 与 `dst` 具有相同的 shape 和 valid_shape
  - `src0`、`src1`、`dst` 使用 `blayout=row_major`
  - `src1 valid_shape[1] == dst valid_shape[1]`

- **实现检查 (A5)**
  - `src0`、`src1`、`dst` 元素类型一致
  - 数据类型：`i8`、`i16`、`i32`、`f16`、`f32`
  - `src0` 与 `dst` 具有相同的 shape 和 valid_shape
  - `src0`、`src1`、`dst` 使用 `blayout=row_major`
  - `src1 valid_shape[1] == dst valid_shape[1]`

**示例:**

```mlir
pto.tcolexpandmin ins(%src0, %src1 : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>,
                      !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=16,
                      v_row=1, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
                  outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
```

---

### `pto.tcolexpandmul` — 列广播乘法

```
pto.tcolexpandmul ins(<src0>, <src1> : <src0_type>, <src1_type>)
                  outs(<dst> : <dst_type>)
```

**语义：**

```
For each element (i, j):
    dst[i, j] = src0[i, j] * src1[0, j]
```

**参数:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | 主源 tile |
| `src1` | `pto.tile_buf` | 每列标量载体 |
| `dst` | `pto.tile_buf` | 目标 tile |

**返回值:** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查 (A2A3)**
  - `src0`、`src1`、`dst` 元素类型一致
  - 数据类型：`i16`、`i32`、`f16`、`f32`
  - `src0` 与 `dst` 具有相同的 shape 和 valid_shape
  - `src0`、`src1`、`dst` 使用 `blayout=row_major`
  - `src1 valid_shape[1] == dst valid_shape[1]`

- **实现检查 (A5)**
  - `src0`、`src1`、`dst` 元素类型一致
  - 数据类型：`i8`、`i16`、`i32`、`f16`、`f32`
  - `src0` 与 `dst` 具有相同的 shape 和 valid_shape
  - `src0`、`src1`、`dst` 使用 `blayout=row_major`
  - `src1 valid_shape[1] == dst valid_shape[1]`

**示例:**

```mlir
pto.tcolexpandmul ins(%src0, %src1 : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>,
                      !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=16,
                      v_row=1, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
                  outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
```

---

### `pto.tcolexpandadd` — 列广播加法

```
pto.tcolexpandadd ins(<src0>, <src1> : <src0_type>, <src1_type>)
                  outs(<dst> : <dst_type>)
```

**语义：**

```
For each element (i, j):
    dst[i, j] = src0[i, j] + src1[0, j]
```

**参数:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | 主源 tile |
| `src1` | `pto.tile_buf` | 每列标量载体 |
| `dst` | `pto.tile_buf` | 目标 tile |

**返回值:** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查 (A2A3)**
  - `src0`、`src1`、`dst` 元素类型一致
  - 数据类型：`i16`、`i32`、`f16`、`f32`
  - `src0` 与 `dst` 具有相同的 shape 和 valid_shape
  - `src0`、`src1`、`dst` 使用 `blayout=row_major`
  - `src1 valid_shape[1] == dst valid_shape[1]`

- **实现检查 (A5)**
  - `src0`、`src1`、`dst` 元素类型一致
  - 数据类型：`i8`、`i16`、`i32`、`f16`、`f32`
  - `src0` 与 `dst` 具有相同的 shape 和 valid_shape
  - `src0`、`src1`、`dst` 使用 `blayout=row_major`
  - `src1 valid_shape[1] == dst valid_shape[1]`

**示例:**

```mlir
pto.tcolexpandadd ins(%src0, %src1 : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>,
                      !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=16,
                      v_row=1, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
                  outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
```

---

### `pto.tcolexpanddiv` — 列广播除法

```
pto.tcolexpanddiv ins(<src0>, <src1> : <src0_type>, <src1_type>)
                  outs(<dst> : <dst_type>)
                  {precisionType = <precision>}
```

**语义：**

```
For each element (i, j):
    dst[i, j] = src0[i, j] / src1[0, j]
```

**参数:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | 主源 tile（被除数） |
| `src1` | `pto.tile_buf` | 每列标量载体（除数） |
| `dst` | `pto.tile_buf` | 目标 tile |

**返回值:** 无。以 DPS 的形式写入 `dst`。

**属性:**

- `precisionType` — 除法精度模式。默认值为 `#pto<div_precision default>`。
  - `#pto<div_precision default>` — 标准精度除法
  - `#pto<div_precision high_precision>` — 高精度除法，仅当元素类型为 `f16` 或 `f32` 时合法

**约束：**

- **实现检查 (A2A3/A5)**
  - `src0`、`src1`、`dst` 元素类型一致
  - 仅支持浮点类型：`f16`、`f32`
  - `src0` 与 `dst` 具有相同的 shape 和 valid_shape
  - `src0`、`src1`、`dst` 使用 `blayout=row_major`
  - `src1 valid_shape[1] == dst valid_shape[1]`

**示例:**

```mlir
pto.tcolexpanddiv ins(%src0, %src1 : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>,
                      !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=16,
                      v_row=1, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
                  outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
```

---

### `pto.tcolexpandexpdif` — 列广播指数差

```
pto.tcolexpandexpdif ins(<src0>, <src1> : <src0_type>, <src1_type>)
                     outs(<dst> : <dst_type>)
```

**语义：**

```
For each element (i, j):
    dst[i, j] = exp(src0[i, j] - src1[0, j])
```

**参数:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | 主源 tile |
| `src1` | `pto.tile_buf` | 每列标量载体（指数差中的减数） |
| `dst` | `pto.tile_buf` | 目标 tile |

**返回值:** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查 (A2A3/A5)**
  - `src0`、`src1`、`dst` 元素类型一致
  - 仅支持浮点类型：`f16`、`f32`
  - `src0` 与 `dst` 具有相同的 shape 和 valid_shape
  - `src0`、`src1`、`dst` 使用 `blayout=row_major`
  - `src1 valid_shape[1] == dst valid_shape[1]`

**示例:**

```mlir
pto.tcolexpandexpdif ins(%src0, %src1 : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                         v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                         fractal=512, pad=0>,
                         !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=16,
                         v_row=1, v_col=16, blayout=row_major, slayout=none_box,
                         fractal=512, pad=0>)
                     outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                         v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                         fractal=512, pad=0>)
```

---

### `pto.tcolexpandsub` — 列广播减法

```
pto.tcolexpandsub ins(<src0>, <src1> : <src0_type>, <src1_type>)
                  outs(<dst> : <dst_type>)
```

**语义：**

```
For each element (i, j):
    dst[i, j] = src0[i, j] - src1[0, j]
```

**参数:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | 主源 tile |
| `src1` | `pto.tile_buf` | 每列标量载体（被减数广播源） |
| `dst` | `pto.tile_buf` | 目标 tile |

**返回值:** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查 (A2A3)**
  - `src0`、`src1`、`dst` 元素类型一致
  - 数据类型：`i16`、`i32`、`f16`、`f32`
  - `src0` 与 `dst` 具有相同的 shape 和 valid_shape
  - `src0`、`src1`、`dst` 使用 `blayout=row_major`
  - `src1 valid_shape[1] == dst valid_shape[1]`

- **实现检查 (A5)**
  - `src0`、`src1`、`dst` 元素类型一致
  - 数据类型：`i8`、`i16`、`i32`、`f16`、`f32`
  - `src0` 与 `dst` 具有相同的 shape 和 valid_shape
  - `src0`、`src1`、`dst` 使用 `blayout=row_major`
  - `src1 valid_shape[1] == dst valid_shape[1]`

**示例:**

```mlir
pto.tcolexpandsub ins(%src0, %src1 : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>,
                      !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=16,
                      v_row=1, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
                  outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
```

---

### `pto.trowexpandadd` — 行广播加法

```
pto.trowexpandadd ins(<src0>, <src1> : <src0_type>, <src1_type>)
                  outs(<dst> : <dst_type>)
```

**语义：**

```
For each element (i, j):
    dst[i, j] = src0[i, j] + src1[i, 0]
```

**参数:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | 主源 tile |
| `src1` | `pto.tile_buf` | 每行标量载体 |
| `dst` | `pto.tile_buf` | 目标 tile |

**返回值:** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查 (A2A3)**
  - `src0`、`src1`、`dst` 元素类型一致
  - 数据类型：`i16`、`i32`、`f16`、`f32`
  - `src0` 与 `dst` 具有相同的 shape 和 valid_shape
  - `src0`、`dst` 使用 `blayout=row_major`
  - `src1 valid_shape[0] == dst valid_shape[0]`
  - `src1` 为 row_major 时：`src1 valid_shape[1] == 32 / sizeof(dtype)`；否则：`src1 valid_shape[1] == 1`

- **实现检查 (A5)**
  - `src0`、`src1`、`dst` 元素类型一致
  - 数据类型：`i8`、`i16`、`i32`、`f16`、`f32`
  - `src0` 与 `dst` 具有相同的 shape 和 valid_shape
  - `src0`、`dst` 使用 `blayout=row_major`
  - `src1 valid_shape[0] == dst valid_shape[0]`
  - `src1` 为 row_major 时：`src1 valid_shape[1] == 32 / sizeof(dtype)`；否则：`src1 valid_shape[1] == 1`

**示例:**

```mlir
pto.trowexpandadd ins(%src0, %src1 : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>,
                      !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=1,
                      v_row=16, v_col=1, blayout=col_major, slayout=none_box,
                      fractal=512, pad=0>)
                  outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
```

---

### `pto.trowexpandexpdif` — 行广播指数差

```
pto.trowexpandexpdif ins(<src0>, <src1>[, <tmp>] : <src0_type>, <src1_type>[, <tmp_type>])
                     outs(<dst> : <dst_type>)
```

**语义：**

```
For each element (i, j):
    dst[i, j] = exp(src0[i, j] - src1[i, 0])
```

**参数:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | 主源 tile |
| `src1` | `pto.tile_buf` | 每行标量载体（指数差中的减数） |
| `dst` | `pto.tile_buf` | 目标 tile |

**返回值:** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查 (A2A3/A5)**
  - `src0`、`src1`、`dst` 元素类型一致
  - 仅支持浮点类型：`f16`、`f32`
  - `dst` 使用 `blayout=row_major`
  - 可选 `tmp` 操作数：用于 pto-isa 中需要 tmp 的重载

**示例:**

```mlir
pto.trowexpandexpdif ins(%src0, %src1 : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                         v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                         fractal=512, pad=0>,
                         !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=1,
                         v_row=16, v_col=1, blayout=col_major, slayout=none_box,
                         fractal=512, pad=0>)
                     outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                         v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                         fractal=512, pad=0>)
```

---

### `pto.trowexpandmax` — 行广播取最大值

```
pto.trowexpandmax ins(<src0>, <src1>[, <tmp>] : <src0_type>, <src1_type>[, <tmp_type>])
                  outs(<dst> : <dst_type>)
```

**语义：**

```
For each element (i, j):
    dst[i, j] = max(src0[i, j], src1[i, 0])
```

**参数:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | 主源 tile |
| `src1` | `pto.tile_buf` | 每行标量载体 |
| `dst` | `pto.tile_buf` | 目标 tile |

**返回值:** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查 (A2A3)**
  - `src0`、`src1`、`dst` 元素类型一致
  - 数据类型：`i16`、`i32`、`f16`、`f32`
  - `dst` 使用 `blayout=row_major`
  - 可选 `tmp` 操作数：用于 pto-isa 中需要 tmp 的重载

- **实现检查 (A5)**
  - `src0`、`src1`、`dst` 元素类型一致
  - 数据类型：`i8`、`i16`、`i32`、`f16`、`bf16`、`f32`
  - `dst` 使用 `blayout=row_major`
  - 可选 `tmp` 操作数：用于 pto-isa 中需要 tmp 的重载

**示例:**

```mlir
pto.trowexpandmax ins(%src0, %src1 : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>,
                      !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=1,
                      v_row=16, v_col=1, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
                  outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
```

---

### `pto.trowexpandmin` — 行广播取最小值

```
pto.trowexpandmin ins(<src0>, <src1>[, <tmp>] : <src0_type>, <src1_type>[, <tmp_type>])
                  outs(<dst> : <dst_type>)
```

**语义：**

```
For each element (i, j):
    dst[i, j] = min(src0[i, j], src1[i, 0])
```

**参数:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | 主源 tile |
| `src1` | `pto.tile_buf` | 每行标量载体 |
| `dst` | `pto.tile_buf` | 目标 tile |

**返回值:** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查 (A2A3)**
  - `src0`、`src1`、`dst` 元素类型一致
  - 数据类型：`i16`、`i32`、`f16`、`f32`
  - `dst` 使用 `blayout=row_major`
  - 可选 `tmp` 操作数：用于 pto-isa 中需要 tmp 的重载

- **实现检查 (A5)**
  - `src0`、`src1`、`dst` 元素类型一致
  - 数据类型：`i8`、`i16`、`i32`、`f16`、`bf16`、`f32`
  - `dst` 使用 `blayout=row_major`
  - 可选 `tmp` 操作数：用于 pto-isa 中需要 tmp 的重载

**示例:**

```mlir
pto.trowexpandmin ins(%src0, %src1 : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>,
                      !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=1,
                      v_row=16, v_col=1, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
                  outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
```
