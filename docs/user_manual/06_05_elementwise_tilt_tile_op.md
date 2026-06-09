# **逐元素操作（Tile-Tile）**

本节描述了 PTO ISA 全部逐元素操作的指令名称、签名和语义。所有操作均作用于本地缓冲区（`tile_buf`，位于 `loc=vec` 空间），采用"目标传递风格"（Destination-Passing Style, DPS）：操作本身不产生 SSA 返回值，而是直接将结果写入预先分配好的目标 `tile_buf`。

这一类操作通常具有如下装配形式：

```mlir
 pto.op ins(%lhs, %rhs : !pto.tile_buf<...>, !pto.tile_buf<...>)
        outs(%dst : !pto.tile_buf<...>)
```

通用约束通常包括：

- 输入 tile 的元素类型兼容
- 输入 tile 的 shape 和 valid-shape 兼容
- 输出 tile 的类型与目标语义匹配

---

## 目录

- [`pto.tadd` — 逐元素加法](#ptotadd--逐元素加法)
- [`pto.tabs` — 逐元素绝对值](#ptotabs--逐元素绝对值)
- [`pto.tand` — 逐元素按位与](#ptotand--逐元素按位与)
- [`pto.tor` — 逐元素按位或](#ptotor--逐元素按位或)
- [`pto.tsub` — 逐元素减法](#ptotsub--逐元素减法)
- [`pto.tmul` — 逐元素乘法](#ptotmul--逐元素乘法)
- [`pto.tmin` — 逐元素取最小值](#ptotmin--逐元素取最小值)
- [`pto.tmax` — 逐元素取最大值](#ptotmax--逐元素取最大值)
- [`pto.tcmp` — 逐元素比较](#ptotcmp--逐元素比较)
- [`pto.tdiv` — 逐元素除法](#ptotdiv--逐元素除法)
- [`pto.tshl` — 逐元素左移](#ptotshl--逐元素左移)
- [`pto.tshr` — 逐元素右移](#ptotshr--逐元素右移)
- [`pto.txor` — 逐元素按位异或](#ptotxor--逐元素按位异或)
- [`pto.tlog` — 逐元素自然对数](#ptotlog--逐元素自然对数)
- [`pto.trecip` — 逐元素倒数](#ptotrecip--逐元素倒数)
- [`pto.tprelu` — 参数化 ReLU](#ptotprelu--参数化-relu)
- [`pto.taddc` — 三元逐元素加法](#ptotaddc--三元逐元素加法)
- [`pto.tsubc` — 三元逐元素减加](#ptotsubc--三元逐元素减加)
- [`pto.tcvt` — 逐元素类型转换](#ptotcvt--逐元素类型转换)
- [`pto.tsel` — 掩码选择](#ptotsel--掩码选择)
- [`pto.trsqrt` — 逐元素倒数平方根](#ptotrsqrt--逐元素倒数平方根)
- [`pto.tsqrt` — 逐元素平方根](#ptotsqrt--逐元素平方根)
- [`pto.texp` — 逐元素指数函数](#ptotexp--逐元素指数函数)
- [`pto.tnot` — 逐元素按位取反](#ptotnot--逐元素按位取反)
- [`pto.trelu` — ReLU 激活](#ptotrelu--relu-激活)
- [`pto.tneg` — 逐元素取负](#ptotneg--逐元素取负)
- [`pto.trem` — 逐元素取余（带临时 tile）](#ptotrem--逐元素取余带临时-tile)
- [`pto.tfmod` — 逐元素取余（无需临时 tile）](#ptotfmod--逐元素取余无需临时-tile)

---

## 操作详解

### `pto.tadd` — 逐元素加法

```
pto.tadd ins(<src0>, <src1> : <src0_type>, <src1_type>)
         outs(<dst> : <dst_type>)
```

**语义：**

```
For each element (i, j):
    dst[i, j] = src0[i, j] + src1[i, j]
```

**参数:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | 第一个源 tile buffer |
| `src1` | `pto.tile_buf` | 第二个源 tile buffer |
| `dst` | `pto.tile_buf` | 目标 tile buffer |

**返回值:** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查 (A2A3)**
  - tile 元素类型必须为以下之一：`i32`、`i16`、`f16` 或 `f32`。
  - tile 必须使用行优先布局 (`blayout=row_major`)。

- **实现检查 (A5)**
  - tile 元素类型必须为以下之一：`i32`、`f32`、`i16`、`f16`、`bf16` 或 `i8`。
  - tile 必须使用行优先布局 (`blayout=row_major`)。

**示例:**

```mlir
pto.tadd ins(%a, %b : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>,
             !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
         outs(%c : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

### `pto.tabs` — 逐元素绝对值

```
pto.tabs ins(<src> : <src_type>)
         outs(<dst> : <dst_type>)
```

**语义：**

```
For each element (i, j):
    dst[i, j] = abs(src[i, j])
```

**参数:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | 源 tile buffer |
| `dst` | `pto.tile_buf` | 目标 tile buffer |

**返回值:** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **NPU 约束**
  - tile 元素类型必须为 `f32` 或 `f16`。
  - `src` 和 `dst` 必须使用 `loc=vec`。
  - 有效区域必须在静态 tile 形状范围内。
  - `src` 和 `dst` 必须具有相同的有效区域。
  - tile 必须使用行优先布局 (`blayout=row_major`)。

**示例:**

```mlir
pto.tabs ins(%a : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
         outs(%c : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

### `pto.tand` — 逐元素按位与

```
pto.tand ins(<src0>, <src1> : <src0_type>, <src1_type>)
         outs(<dst> : <dst_type>)
```

**语义：**

```
For each element (i, j):
    dst[i, j] = src0[i, j] & src1[i, j]
```

**参数:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | 第一个源 tile buffer |
| `src1` | `pto.tile_buf` | 第二个源 tile buffer |
| `dst` | `pto.tile_buf` | 目标 tile buffer |

**返回值:** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查 (A2A3)**
  - `src0`、`src1` 和 `dst` 元素类型必须一致。
  - 共享元素类型必须为 `i8` 或 `i16`。
  - 三个 tile 必须使用行优先布局 (`blayout=row_major`)。
  - `src0`、`src1` 必须分别与 `dst` 具有相同有效区域。

- **实现检查 (A5)**
  - `src0`、`src1` 和 `dst` 元素类型必须一致。
  - 共享元素类型必须为 `i8`、`i16` 或 `i32`。
  - 三个 tile 必须使用行优先布局 (`blayout=row_major`)。
  - `src0`、`src1` 必须分别与 `dst` 具有相同有效区域。

**示例:**

```mlir
pto.tand ins(%a, %b : !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>,
             !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
         outs(%c : !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

### `pto.tor` — 逐元素按位或

```
pto.tor ins(<src0>, <src1> : <src0_type>, <src1_type>)
        outs(<dst> : <dst_type>)
```

**语义：**

```
For each element (i, j):
    dst[i, j] = src0[i, j] | src1[i, j]
```

**参数:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | 第一个源 tile buffer |
| `src1` | `pto.tile_buf` | 第二个源 tile buffer |
| `dst` | `pto.tile_buf` | 目标 tile buffer |

**返回值:** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查 (A2A3)**
  - `src0`、`src1` 和 `dst` 元素类型必须一致。
  - 共享元素类型必须为 `i8` 或 `i16`。
  - 三个 tile 必须使用行优先布局 (`blayout=row_major`)。
  - `src0`、`src1` 必须分别与 `dst` 具有相同有效区域。

- **实现检查 (A5)**
  - `src0`、`src1` 和 `dst` 元素类型必须一致。
  - 共享元素类型必须为 `i8`、`i16` 或 `i32`。
  - 三个 tile 必须使用行优先布局 (`blayout=row_major`)。
  - `src0`、`src1` 必须分别与 `dst` 具有相同有效区域。

**示例:**

```mlir
pto.tor ins(%a, %b : !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
            v_row=16, v_col=16, blayout=row_major, slayout=none_box,
            fractal=512, pad=0>,
            !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
            v_row=16, v_col=16, blayout=row_major, slayout=none_box,
            fractal=512, pad=0>)
        outs(%c : !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
            v_row=16, v_col=16, blayout=row_major, slayout=none_box,
            fractal=512, pad=0>)
```

---

### `pto.tsub` — 逐元素减法

```
pto.tsub ins(<src0>, <src1> : <src0_type>, <src1_type>)
         outs(<dst> : <dst_type>)
```

**语义：**

```
For each element (i, j):
    dst[i, j] = src0[i, j] - src1[i, j]
```

**参数:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | 被减数 tile buffer |
| `src1` | `pto.tile_buf` | 减数 tile buffer |
| `dst` | `pto.tile_buf` | 目标 tile buffer |

**返回值:** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查 (A2A3)**
  - tile 元素类型必须为 `i32`、`i16`、`f16` 或 `f32`。
  - tile 必须使用行优先布局 (`blayout=row_major`)。
  - 有效区域必须在静态 tile 形状范围内。
  - `src0`、`src1` 和 `dst` 应具有相同的 `validRow/validCol`。

- **实现检查 (A5)**
  - tile 元素类型必须为 `i32`、`i16`、`i8`、`f32` 或 `f16`。
  - tile 必须使用行优先布局 (`blayout=row_major`)。
  - 有效区域必须在静态 tile 形状范围内。
  - `src0`、`src1` 和 `dst` 应具有相同的 `validRow/validCol`。

**示例:**

```mlir
pto.tsub ins(%a, %b : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>,
             !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
         outs(%c : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

### `pto.tmul` — 逐元素乘法

```
pto.tmul ins(<src0>, <src1> : <src0_type>, <src1_type>)
         outs(<dst> : <dst_type>)
```

**语义：**

```
For each element (i, j):
    dst[i, j] = src0[i, j] * src1[i, j]
```

**参数:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | 第一个源 tile buffer |
| `src1` | `pto.tile_buf` | 第二个源 tile buffer |
| `dst` | `pto.tile_buf` | 目标 tile buffer |

**返回值:** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查 (A2A3)**
  - tile 元素类型必须为 `i32`、`i16`、`f16` 或 `f32`。
  - tile 必须使用 `loc=vec` 和行优先布局 (`blayout=row_major`)。
  - 有效区域必须在静态 tile 形状范围内。
  - `src0`、`src1` 和 `dst` 应具有相同的 `validRow/validCol`。

- **实现检查 (A5)**
  - tile 元素类型必须为 `i32`、`f32`、`i16` 或 `f16`。
  - tile 必须使用 `loc=vec` 和行优先布局 (`blayout=row_major`)。
  - 有效区域必须在静态 tile 形状范围内。
  - `src0`、`src1` 和 `dst` 应具有相同的 `validRow/validCol`。

**示例:**

```mlir
pto.tmul ins(%a, %b : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>,
             !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
         outs(%c : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

### `pto.tmin` — 逐元素取最小值

```
pto.tmin ins(<src0>, <src1> : <src0_type>, <src1_type>)
         outs(<dst> : <dst_type>)
```

**语义：**

```
For each element (i, j):
    dst[i, j] = min(src0[i, j], src1[i, j])
```

**参数:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | 第一个源 tile buffer |
| `src1` | `pto.tile_buf` | 第二个源 tile buffer |
| `dst` | `pto.tile_buf` | 目标 tile buffer |

**返回值:** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查 (A2A3)**
  - tile 元素类型必须为 `i32`、`i16`、`f16` 或 `f32`。
  - tile 必须使用行优先布局 (`blayout=row_major`)。
  - 有效区域必须在静态 tile 形状范围内。
  - `src0`、`src1` 和 `dst` 应具有相同的 `validRow/validCol`。

- **实现检查 (A5)**
  - tile 元素类型必须为 `i32`、`i16`、`i8`、`f32` 或 `f16`。
  - tile 必须使用行优先布局 (`blayout=row_major`)。
  - 有效区域必须在静态 tile 形状范围内。
  - `src0`、`src1` 和 `dst` 应具有相同的 `validRow/validCol`。

**示例:**

```mlir
pto.tmin ins(%a, %b : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>,
             !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
         outs(%c : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

### `pto.tmax` — 逐元素取最大值

```
pto.tmax ins(<src0>, <src1> : <src0_type>, <src1_type>)
         outs(<dst> : <dst_type>)
```

**语义：**

```
For each element (i, j):
    dst[i, j] = max(src0[i, j], src1[i, j])
```

**参数:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | 第一个源 tile buffer |
| `src1` | `pto.tile_buf` | 第二个源 tile buffer |
| `dst` | `pto.tile_buf` | 目标 tile buffer |

**返回值:** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查 (A2A3)**
  - tile 元素类型必须为 `i32`、`i16`、`f16` 或 `f32`。
  - tile 必须使用行优先布局 (`blayout=row_major`)。
  - 有效区域必须在静态 tile 形状范围内。
  - `src0`、`src1` 和 `dst` 应具有相同的 `validRow/validCol`。

- **实现检查 (A5)**
  - tile 元素类型必须为 `i32`、`i16`、`i8`、`f32` 或 `f16`。
  - tile 必须使用行优先布局 (`blayout=row_major`)。
  - 有效区域必须在静态 tile 形状范围内。
  - `src0`、`src1` 和 `dst` 应具有相同的 `validRow/validCol`。

**示例:**

```mlir
pto.tmax ins(%a, %b : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>,
             !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
         outs(%c : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

### `pto.tcmp` — 逐元素比较

```
pto.tcmp ins(<src0>, <src1> {cmpMode = <mode>} : <src0_type>, <src1_type>)
         outs(<dst> : <dst_type>)
```

**语义：**

```
For each element (i, j):
    dst[i, j] = (src0[i, j] <cmpMode> src1[i, j]) ? 1 : 0
```

**参数:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | 第一个比较输入 |
| `src1` | `pto.tile_buf` | 第二个比较输入 |
| `dst` | `pto.tile_buf` | 目标 mask tile buffer |

**返回值:** 无。以 DPS 的形式写入 `dst`。

**属性：**

- `cmpMode` — 比较模式。默认值为 `eq`。
  - `#pto<cmp eq>` — 等于
  - `#pto<cmp ne>` — 不等于
  - `#pto<cmp lt>` — 小于
  - `#pto<cmp le>` — 小于等于
  - `#pto<cmp gt>` — 大于
  - `#pto<cmp ge>` — 大于等于

**约束：**

- **实现检查 (A2A3)**
  - 输入元素类型必须为 `i32`、`f16` 或 `f32`。
  - 输出 mask 元素类型必须为 `i8`。
  - `src0`、`src1` 和 `dst` 必须使用 `loc=vec`。
  - 有效区域必须在静态 tile 形状范围内，且 `src0/src1/dst` 有效区域一致。

- **实现检查 (A5)**
  - 输入元素类型必须为 `i32`、`i16`、`i8`、`f32`、`f16` 或 `bf16`。

**示例:**

```mlir
pto.tcmp ins(%a, %b {cmpMode = #pto<cmp lt>} :
             !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>,
             !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
         outs(%mask : !pto.tile_buf<loc=vec, dtype=i8, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

### `pto.tdiv` — 逐元素除法

```
pto.tdiv ins(<src0>, <src1> : <src0_type>, <src1_type>)
         outs(<dst> : <dst_type>)
         {precisionType = <precision>}
```

**语义：**

```
For each element (i, j):
    dst[i, j] = src0[i, j] / src1[i, j]
```

**参数:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | 被除数 tile buffer |
| `src1` | `pto.tile_buf` | 除数 tile buffer |
| `dst` | `pto.tile_buf` | 目标 tile buffer |

**返回值:** 无。以 DPS 的形式写入 `dst`。

**属性：**

- `precisionType` — 除法精度模式。默认值为 `default`。
  - `#pto<div_precision default>` — 默认精度
  - `#pto<div_precision high_precision>` — 高精度

**约束：**

- **实现检查 (A2A3)**
  - tile 元素类型必须为 `f16` 或 `f32`。
  - `src0`、`src1` 和 `dst` 必须元素类型一致，并使用行优先布局 (`blayout=row_major`)。
  - 有效区域必须在静态 tile 形状范围内，且三个 tile 有效区域一致。

- **实现检查 (A5)**
  - tile 元素类型必须为 `i32`、`i16`、`f16` 或 `f32`。
  - `src0`、`src1` 和 `dst` 必须元素类型一致，并使用行优先布局 (`blayout=row_major`)。
  - 有效区域必须在静态 tile 形状范围内，且三个 tile 有效区域一致。

- **除零行为**
  - 除零行为由目标实现定义。

**示例:**

```mlir
pto.tdiv ins(%a, %b : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>,
             !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
         outs(%c : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

### `pto.tshl` — 逐元素左移

```
pto.tshl ins(<src0>, <src1> : <src0_type>, <src1_type>)
         outs(<dst> : <dst_type>)
```

**语义：**

```
For each element (i, j):
    dst[i, j] = src0[i, j] << src1[i, j]
```

**参数:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | 待左移的源 tile buffer |
| `src1` | `pto.tile_buf` | 左移位数 tile buffer |
| `dst` | `pto.tile_buf` | 目标 tile buffer |

**返回值:** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查 (A2A3/A5)**
  - `src0` 和 `src1` 元素类型必须一致。
  - 共享元素类型必须为 `i8`、`i16` 或 `i32`。
  - `src0`、`src1` 和 `dst` 必须使用行优先布局 (`blayout=row_major`)。
  - `src0`、`src1` 必须分别与 `dst` 具有相同有效区域。

**示例:**

```mlir
pto.tshl ins(%a, %b : !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>,
             !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
         outs(%c : !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

### `pto.tshr` — 逐元素右移

```
pto.tshr ins(<src0>, <src1> : <src0_type>, <src1_type>)
         outs(<dst> : <dst_type>)
```

**语义：**

```
For each element (i, j):
    dst[i, j] = src0[i, j] >> src1[i, j]
```

**参数:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | 待右移的源 tile buffer |
| `src1` | `pto.tile_buf` | 右移位数 tile buffer |
| `dst` | `pto.tile_buf` | 目标 tile buffer |

**返回值:** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查 (A2A3/A5)**
  - `src0` 和 `src1` 元素类型必须一致。
  - 共享元素类型必须为 `i8`、`i16` 或 `i32`。
  - `src0`、`src1` 和 `dst` 必须使用行优先布局 (`blayout=row_major`)。
  - `src0`、`src1` 必须分别与 `dst` 具有相同有效区域。

**示例:**

```mlir
pto.tshr ins(%a, %b : !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>,
             !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
         outs(%c : !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

### `pto.txor` — 逐元素按位异或

```
pto.txor ins(<src0>, <src1>, <tmp> : <src0_type>, <src1_type>, <tmp_type>)
         outs(<dst> : <dst_type>)
```

**语义：**

```
For each element (i, j):
    dst[i, j] = src0[i, j] ^ src1[i, j]
```

**参数:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | 第一个源 tile buffer |
| `src1` | `pto.tile_buf` | 第二个源 tile buffer |
| `tmp` | `pto.tile_buf` | 临时 tile buffer；A5 中可复用占位 |
| `dst` | `pto.tile_buf` | 目标 tile buffer |

**返回值:** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查 (A2A3)**
  - `src0`、`src1`、`tmp` 和 `dst` 元素类型必须一致。
  - 共享元素类型必须为 `i8` 或 `i16`。
  - 四个 tile 必须使用行优先布局 (`blayout=row_major`)。
  - `src0`、`src1`、`tmp` 必须分别与 `dst` 具有相同有效区域。

- **实现检查 (A5)**
  - `src0`、`src1` 和 `dst` 元素类型必须一致。
  - 共享元素类型必须为 `i8`、`i16` 或 `i32`。
  - `src0`、`src1` 和 `dst` 必须使用行优先布局 (`blayout=row_major`)。
  - `src0`、`src1` 必须分别与 `dst` 具有相同有效区域。
  - `tmp` 在 A5 路径中仅作为占位参数。

**示例:**

```mlir
// A2/A3：需要独立的 tmp tile
pto.txor ins(%src0, %src1, %tmp :
             !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>,
             !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>,
             !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
         outs(%dst : !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)

// A5：tmp 可复用 dst 作为占位
pto.txor ins(%src0, %src1, %dst :
             !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>,
             !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>,
             !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
         outs(%dst : !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

### `pto.tlog` — 逐元素自然对数

```
pto.tlog ins(<src> : <src_type>)
         outs(<dst> : <dst_type>)
         {precisionType = <precision>}
```

**语义：**

```
For each element (i, j):
    dst[i, j] = ln(src[i, j])
```

**参数:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | 源 tile buffer |
| `dst` | `pto.tile_buf` | 目标 tile buffer |

**返回值:** 无。以 DPS 的形式写入 `dst`。

**属性：**

- `precisionType` — 对数精度模式。默认值为 `default`。
  - `#pto<log_precision default>` — 默认精度
  - `#pto<log_precision high_precision>` — 高精度

**约束：**

- **NPU 约束**
  - tile 元素类型必须为 `f32` 或 `f16`。
  - `src` 和 `dst` 必须使用 `loc=vec`。
  - 有效区域必须在静态 tile 形状范围内，且 `src` 与 `dst` 有效区域一致。
  - tile 必须使用行优先布局 (`blayout=row_major`)。

- **定义域行为**
  - 对 `src <= 0` 等输入的行为由目标实现定义。

**示例:**

```mlir
pto.tlog ins(%a : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
         outs(%c : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

### `pto.trecip` — 逐元素倒数

```
pto.trecip ins(<src> : <src_type>)
           outs(<dst> : <dst_type>)
           {precisionType = <precision>}
```

**语义：**

```
For each element (i, j):
    dst[i, j] = 1.0 / src[i, j]
```

**参数:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | 源 tile buffer |
| `dst` | `pto.tile_buf` | 目标 tile buffer |

**返回值:** 无。以 DPS 的形式写入 `dst`。

**属性：**

- `precisionType` — 倒数精度模式。默认值为 `default`。
  - `#pto<recip_precision default>` — 默认精度
  - `#pto<recip_precision high_precision>` — 高精度

**约束：**

- **NPU 约束**
  - tile 元素类型必须为 `f32` 或 `f16`。
  - tile 必须使用 `loc=vec` 和行优先布局 (`blayout=row_major`)。
  - 有效区域必须在静态 tile 形状范围内。
  - `src` 和 `dst` 必须具有相同的有效区域。
  - A3 的 `TRECIP` 指令不支持源 tile 与目标 tile 使用同一段内存。

- **除零行为**
  - 除零行为由目标实现定义；CPU simulator 的 debug 构建可能 assert。

**示例:**

```mlir
pto.trecip ins(%a : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
               v_row=16, v_col=16, blayout=row_major, slayout=none_box,
               fractal=512, pad=0>)
           outs(%c : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
               v_row=16, v_col=16, blayout=row_major, slayout=none_box,
               fractal=512, pad=0>)
```

---

### `pto.tprelu` — 参数化 ReLU

```
pto.tprelu ins(<src0>, <src1>, <tmp> : <src0_type>, <src1_type>, <tmp_type>)
           outs(<dst> : <dst_type>)
```

**语义：**

```
For each element (i, j):
    dst[i, j] = src0[i, j] > 0 ? src0[i, j] : src1[i, j] * src0[i, j]
```

**参数:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | 输入激活 tile buffer |
| `src1` | `pto.tile_buf` | 逐元素负半轴斜率 tile buffer |
| `tmp` | `pto.tile_buf` | 临时 tile buffer；A5 中可复用占位 |
| `dst` | `pto.tile_buf` | 目标 tile buffer |

**返回值:** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查 (A2A3)**
  - `dst/src0/src1` 元素类型必须一致，且必须为 `f16` 或 `f32`。
  - `tmp` 元素类型必须为 `u8`。
  - 所有相关 tile 必须使用行优先布局 (`blayout=row_major`)。
  - `src0`、`src1` 必须分别与 `dst` 具有相同有效区域。
  - A3 中两个源 tile、目标 tile、临时空间不得内存重叠。

- **实现检查 (A5)**
  - `dst/src0/src1` 元素类型必须一致，且必须为 `f16` 或 `f32`。
  - 所有相关 tile 必须使用行优先布局 (`blayout=row_major`)。
  - `src0`、`src1` 必须分别与 `dst` 具有相同有效区域。
  - `tmp` 在 A5 路径中可作为占位参数。

**示例:**

```mlir
// A2/A3：需要独立的 tmp tile（元素类型为 u8）
pto.tprelu ins(%a, %slopes, %tmp :
               !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
               v_row=16, v_col=16, blayout=row_major, slayout=none_box,
               fractal=512, pad=0>,
               !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
               v_row=16, v_col=16, blayout=row_major, slayout=none_box,
               fractal=512, pad=0>,
               !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
               v_row=16, v_col=16, blayout=row_major, slayout=none_box,
               fractal=512, pad=0>)
           outs(%c : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
               v_row=16, v_col=16, blayout=row_major, slayout=none_box,
               fractal=512, pad=0>)

// A5：tmp 可复用 dst 作为占位
pto.tprelu ins(%a, %slopes, %c :
               !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
               v_row=16, v_col=16, blayout=row_major, slayout=none_box,
               fractal=512, pad=0>,
               !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
               v_row=16, v_col=16, blayout=row_major, slayout=none_box,
               fractal=512, pad=0>,
               !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
               v_row=16, v_col=16, blayout=row_major, slayout=none_box,
               fractal=512, pad=0>)
           outs(%c : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
               v_row=16, v_col=16, blayout=row_major, slayout=none_box,
               fractal=512, pad=0>)
```

---

### `pto.taddc` — 三元逐元素加法

```
pto.taddc ins(<src0>, <src1>, <src2> : <src0_type>, <src1_type>, <src2_type>)
          outs(<dst> : <dst_type>)
```

**语义：**

```
For each element (i, j):
    dst[i, j] = src0[i, j] + src1[i, j] + src2[i, j]
```

**参数:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | 第一个源 tile buffer |
| `src1` | `pto.tile_buf` | 第二个源 tile buffer |
| `src2` | `pto.tile_buf` | 第三个源 tile buffer |
| `dst` | `pto.tile_buf` | 目标 tile buffer |

**返回值:** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现约束**
  - 实现以 `dst` 的有效行/列作为迭代域。
  - `src0`、`src1`、`src2` 与 `dst` 的具体类型和有效区域约束以 verifier 为准。

**示例:**

```mlir
pto.taddc ins(%a, %b, %c :
              !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>,
              !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>,
              !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>)
          outs(%d : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>)
```

---

### `pto.tsubc` — 三元逐元素减加

```
pto.tsubc ins(<src0>, <src1>, <src2> : <src0_type>, <src1_type>, <src2_type>)
          outs(<dst> : <dst_type>)
```

**语义：**

```
For each element (i, j):
    dst[i, j] = src0[i, j] - src1[i, j] + src2[i, j]
```

**参数:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | 第一个源 tile buffer |
| `src1` | `pto.tile_buf` | 减数 tile buffer |
| `src2` | `pto.tile_buf` | 加数 tile buffer |
| `dst` | `pto.tile_buf` | 目标 tile buffer |

**返回值:** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现约束**
  - 实现以 `dst` 的有效行/列作为迭代域。
  - `src0`、`src1`、`src2` 与 `dst` 的具体类型和有效区域约束以 verifier 为准。

**示例:**

```mlir
pto.tsubc ins(%a, %b, %c :
              !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>,
              !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>,
              !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>)
          outs(%d : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>)
```

---

### `pto.tcvt` — 逐元素类型转换

```
pto.tcvt ins(<src> {rmode = <round_mode>, satmode = <saturation_mode>} : <src_type>)
         outs(<dst> : <dst_type>)
```

**语义：**

```
For each element (i, j):
    dst[i, j] = saturate(cast(src[i, j], rmode), sat_mode)
```

**参数:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | 源 tile buffer |
| `dst` | `pto.tile_buf` | 目标 tile buffer，元素类型可不同于 `src` |

**返回值:** 无。以 DPS 的形式写入 `dst`。

**属性：**

- `rmode` — 舍入模式。默认值为 `CAST_RINT`。
  - `#pto<round_mode NONE>` — 无舍入
  - `#pto<round_mode RINT>` — 四舍五入到最近偶数
  - `#pto<round_mode ROUND>` — 四舍五入
  - `#pto<round_mode FLOOR>` — 向下取整
  - `#pto<round_mode CEIL>` — 向上取整
  - `#pto<round_mode TRUNC>` — 截断
  - `#pto<round_mode ODD>` — 向最近奇数舍入
  - `#pto<round_mode CAST_RINT>` — 类型转换默认舍入
- `satmode` — 饱和模式，控制舍入后是否按目标类型范围 clamp。默认值为 `OFF`。
  - `#pto<saturation_mode ON>` — 启用饱和
  - `#pto<saturation_mode OFF>` — 关闭饱和

**约束：**

- **通用检查**
  - `src` 和 `dst` 必须是兼容的 tile buffer。
  - `src` 与 `dst` 的逻辑范围和有效区域必须兼容。

- **A2/A3 与 A5 低精度限制**
  - A2/A3 不支持低精度 `tcvt` 操作数。
  - A5 仅接受实现中列出的低精度转换对，例如 `f32 -> f8E4M3*`、`f32 -> f8E5M2*`、`f32 -> !pto.hif8`、`f16 -> !pto.hif8`、`bf16 <-> !pto.f4E1M2x2`、`bf16 <-> !pto.f4E2M1x2`、`f8E4M3* -> f32`、`f8E5M2* -> f32`、`!pto.hif8 -> f32`。
  - 非低精度类型对沿用目标定义的转换行为。

**硬件:**

- PIPE_V

**示例:**

```mlir
pto.tcvt ins(%src {rmode = #pto<round_mode FLOOR>, satmode = #pto<saturation_mode ON>} :
             !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
         outs(%dst : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

### `pto.tsel` — 掩码选择

```
pto.tsel ins(<mask>, <src0>, <src1>, <tmp> : <mask_type>, <src0_type>, <src1_type>, <tmp_type>)
         outs(<dst> : <dst_type>)
```

**语义：**

```
For each element (i, j):
    dst[i, j] = mask[i, j] ? src0[i, j] : src1[i, j]
```

**参数:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `mask` | `pto.tile_buf` | 谓词 mask tile buffer |
| `src0` | `pto.tile_buf` | mask 为真时选择的源 tile buffer |
| `src1` | `pto.tile_buf` | mask 为假时选择的源 tile buffer |
| `tmp` | `pto.tile_buf` | 临时 scratch tile buffer |
| `dst` | `pto.tile_buf` | 目标 tile buffer |

**返回值:** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查 (A2A3)**
  - `src0`、`src1` 和 `dst` 元素类型必须一致。
  - 共享元素类型必须为 `i16`、`i32`、`f16`、`bf16` 或 `f32`。
  - `src0`、`src1` 和 `dst` 必须使用行优先布局 (`blayout=row_major`)。

- **实现检查 (A5)**
  - `src0`、`src1` 和 `dst` 元素类型必须一致。
  - 共享元素类型必须为 `i8`、`i16`、`i32`、`f16`、`bf16` 或 `f32`。
  - `src0`、`src1` 和 `dst` 必须使用行优先布局 (`blayout=row_major`)。

- **临时 tile**
  - `tmp` 是当前 DPS/ISA 形式要求的临时 scratch tile。

**示例:**

```mlir
pto.tsel ins(%mask, %a, %b, %tmp :
             !pto.tile_buf<loc=vec, dtype=i8, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>,
             !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>,
             !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>,
             !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
         outs(%dst : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

### `pto.trsqrt` — 逐元素倒数平方根

```
// 默认精度（无 tmp）
pto.trsqrt ins(<src> : <src_type>)
           outs(<dst> : <dst_type>)

// 高精度（需提供 tmp）
pto.trsqrt ins(<src>, <tmp> : <src_type>, <tmp_type>)
           outs(<dst> : <dst_type>)
           {precisionType = #pto<rsqrt_precision high_precision>}
```

**语义：**

```
For each element (i, j):
    dst[i, j] = 1.0 / sqrt(src[i, j])
```

**参数:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | 源 tile buffer |
| `tmp` | `pto.tile_buf`（可选） | 临时 tile buffer；仅 `HighPrecision` 模式下必须提供，至少 32 字节 |
| `dst` | `pto.tile_buf` | 目标 tile buffer |

**返回值:** 无。以 DPS 的形式写入 `dst`。

**属性：**

- `precisionType` — 倒数平方根精度模式。默认值为 `default`。
  - `#pto<rsqrt_precision default>` — 默认精度
  - `#pto<rsqrt_precision high_precision>` — 高精度，需提供 `tmp` 操作数

**约束：**

- **可选 tmp 操作数**
  - `tmp` 是可选的临时 tile buffer（`Optional<PTODpsType>`），仅在 `precisionType = HighPrecision` 时必须提供。
  - 当 `precisionType` 为默认值 `Default` 时，`tmp` 可省略。
  - `tmp` 必须位于 `loc=vec`，且至少提供 32 字节的存储空间。

- **NPU 约束**
  - tile 元素类型必须为 `f32` 或 `f16`。
  - `src` 和 `dst` 必须使用 `loc=vec`。
  - 有效区域必须在静态 tile 形状范围内，且 `src` 与 `dst` 有效区域一致。
  - tile 必须使用行优先布局 (`blayout=row_major`)。

- **定义域行为**
  - 对 `src == 0` 或负数等输入的行为由目标实现定义。

**示例:**

```mlir
// 默认精度，无 tmp
pto.trsqrt ins(%a : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
               v_row=16, v_col=16, blayout=row_major, slayout=none_box,
               fractal=512, pad=0>)
           outs(%c : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
               v_row=16, v_col=16, blayout=row_major, slayout=none_box,
               fractal=512, pad=0>)

// 高精度，带 tmp
pto.trsqrt ins(%a, %tmp : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
               v_row=16, v_col=16, blayout=row_major, slayout=none_box,
               fractal=512, pad=0>,
               !pto.tile_buf<loc=vec, dtype=f16, rows=1, cols=16,
               v_row=1, v_col=16, blayout=row_major, slayout=none_box,
               fractal=512, pad=0>)
           outs(%c : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
               v_row=16, v_col=16, blayout=row_major, slayout=none_box,
               fractal=512, pad=0>)
           {precisionType = #pto<rsqrt_precision high_precision>}
```

---

### `pto.tsqrt` — 逐元素平方根

```
pto.tsqrt ins(<src> : <src_type>)
          outs(<dst> : <dst_type>)
          {precisionType = <precision>}
```

**语义：**

```
For each element (i, j):
    dst[i, j] = sqrt(src[i, j])
```

**参数:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | 源 tile buffer |
| `dst` | `pto.tile_buf` | 目标 tile buffer |

**返回值:** 无。以 DPS 的形式写入 `dst`。

**属性：**

- `precisionType` — 平方根精度模式。默认值为 `default`。所有可选值：`#pto<sqrt_precision default>`（默认精度）、`#pto<sqrt_precision high_precision>`（高精度）。

**约束：**

- **NPU 约束**
  - tile 元素类型必须为 `f32` 或 `f16`。
  - `src` 和 `dst` 必须使用 `loc=vec`。
  - 有效区域必须在静态 tile 形状范围内，且 `src` 与 `dst` 有效区域一致。
  - tile 必须使用行优先布局 (`blayout=row_major`)。

- **定义域行为**
  - 对负数输入等情况的行为由目标实现定义。

**示例:**

```mlir
pto.tsqrt ins(%a : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>)
          outs(%c : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>)
```

---

### `pto.texp` — 逐元素指数函数

```
pto.texp ins(<src> : <src_type>)
         outs(<dst> : <dst_type>)
         {precisionType = <precision>}
```

**语义：**

```
For each element (i, j):
    dst[i, j] = exp(src[i, j])
```

**参数:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | 源 tile buffer |
| `dst` | `pto.tile_buf` | 目标 tile buffer |

**返回值:** 无。以 DPS 的形式写入 `dst`。

**属性：**

- `precisionType` — 指数精度模式。默认值为 `default`。所有可选值：`#pto<exp_precision default>`（默认精度）、`#pto<exp_precision high_precision>`（高精度）。

**约束：**

- **NPU 约束**
  - tile 元素类型必须为 `f32` 或 `f16`。
  - `src` 和 `dst` 必须使用 `loc=vec`。
  - 有效区域必须在静态 tile 形状范围内，且 `src` 与 `dst` 有效区域一致。
  - tile 必须使用行优先布局 (`blayout=row_major`)。

**示例:**

```mlir
pto.texp ins(%a : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
         outs(%c : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

### `pto.tnot` — 逐元素按位取反

```
pto.tnot ins(<src> : <src_type>)
         outs(<dst> : <dst_type>)
```

**语义：**

```
For each element (i, j):
    dst[i, j] = ~src[i, j]
```

**参数:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | 源 tile buffer |
| `dst` | `pto.tile_buf` | 目标 tile buffer |

**返回值:** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查 (A2A3)**
  - tile 元素类型必须为 `i16`。
  - `src` 和 `dst` 元素类型必须一致。
  - tile 必须使用 `loc=vec` 和行优先布局 (`blayout=row_major`)。
  - 有效区域必须在静态 tile 形状范围内，且 `src` 与 `dst` 有效区域一致。

- **实现检查 (A5)**
  - tile 元素类型必须为 `i32`、`i16` 或 `i8`。
  - `src` 和 `dst` 元素类型必须一致。
  - tile 必须使用 `loc=vec` 和行优先布局 (`blayout=row_major`)。
  - 有效区域必须在静态 tile 形状范围内，且 `src` 与 `dst` 有效区域一致。

**示例:**

```mlir
pto.tnot ins(%a : !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
         outs(%c : !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

### `pto.trelu` — ReLU 激活

```
pto.trelu ins(<src> : <src_type>)
          outs(<dst> : <dst_type>)
```

**语义：**

```
For each element (i, j):
    dst[i, j] = max(0, src[i, j])
```

**参数:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | 源 tile buffer |
| `dst` | `pto.tile_buf` | 目标 tile buffer |

**返回值:** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查 (A2A3/A5)**
  - tile 元素类型必须为 `f16`、`f32` 或 `i32`。
  - tile 必须使用 `loc=vec` 和行优先布局 (`blayout=row_major`)。
  - 有效区域必须在静态 tile 形状范围内。
  - `src` 和 `dst` 应具有相同的 `validRow/validCol`。

**示例:**

```mlir
pto.trelu ins(%a : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>)
          outs(%c : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>)
```

---

### `pto.tneg` — 逐元素取负

```
pto.tneg ins(<src> : <src_type>)
         outs(<dst> : <dst_type>)
```

**语义：**

```
For each element (i, j):
    dst[i, j] = -src[i, j]
```

**参数:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | 源 tile buffer |
| `dst` | `pto.tile_buf` | 目标 tile buffer |

**返回值:** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查 (A2A3)**
  - tile 元素类型必须为 `i32`、`i16`、`f16` 或 `f32`。
  - tile 必须使用 `loc=vec`。
  - 有效区域必须在静态 tile 形状范围内。
  - `src` 和 `dst` 必须具有相同有效区域。

- **实现检查 (A5)**
  - tile 元素类型必须为 `i8`、`i16`、`i32`、`f16`、`f32` 或 `bf16`。
  - tile 必须使用 `loc=vec`。
  - 有效区域必须在静态 tile 形状范围内。
  - `src` 和 `dst` 至少必须具有相同有效列，具体以 verifier 为准。

**示例:**

```mlir
pto.tneg ins(%a : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
         outs(%c : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

### `pto.trem` — 逐元素取余（带临时 tile）

```
pto.trem ins(<src0>, <src1>, <tmp> : <src0_type>, <src1_type>, <tmp_type>)
         outs(<dst> : <dst_type>)
         {precisionType = <precision>}
```

**语义：**

```
For each element (i, j):
    dst[i, j] = fmod(src0[i, j], src1[i, j])
```

**参数:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | 被取余 tile buffer |
| `src1` | `pto.tile_buf` | 除数 tile buffer |
| `tmp` | `pto.tile_buf` | 临时 tile buffer |
| `dst` | `pto.tile_buf` | 目标 tile buffer |

**返回值:** 无。以 DPS 的形式写入 `dst`。

**属性：**

- `precisionType` — 取余精度模式。默认值为 `default`。所有可选值：`#pto<rem_precision default>`（默认精度）、`#pto<rem_precision high_precision>`（高精度）。

**约束：**

- **通用检查**
  - 实现使用 `dst` 的有效行/列作为迭代域。
  - `src0`、`src1` 和 `dst` 元素类型必须一致。
  - `tmp` 元素类型必须与 `dst` 一致。
  - `src0`、`src1`、`tmp` 和 `dst` 必须使用行优先布局 (`blayout=row_major`)。
  - `src0`、`src1` 和 `dst` 必须具有相同有效区域。
  - `tmp` 至少提供 1 个有效行，且 `tmp.validCol >= dst.validCol`。

- **实现检查 (A2A3)**
  - `src0/src1/dst` 元素类型必须为 `i32` 或 `f32`。

- **实现检查 (A5)**
  - `src0/src1/dst` 元素类型必须为 `i32`、`i16`、`f16` 或 `f32`。

**示例:**

```mlir
pto.trem ins(%a, %b, %tmp :
             !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>,
             !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>,
             !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
         outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
             v_row=16, v_col=16, blayout=row_major, slayout=none_box,
             fractal=512, pad=0>)
```

---

### `pto.tfmod` — 逐元素取余（无需临时 tile）

```
pto.tfmod ins(<src0>, <src1> : <src0_type>, <src1_type>)
          outs(<dst> : <dst_type>)
          {precisionType = <precision>}
```

**语义：**

```
For each element (i, j):
    dst[i, j] = fmod(src0[i, j], src1[i, j])
```

**参数:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | 被取余 tile buffer |
| `src1` | `pto.tile_buf` | 除数 tile buffer |
| `dst` | `pto.tile_buf` | 目标 tile buffer |

**返回值:** 无。以 DPS 的形式写入 `dst`。

**属性：**

- `precisionType` — fmod 精度模式。默认值为 `default`。所有可选值：`#pto<fmod_precision default>`（默认精度）、`#pto<fmod_precision high_precision>`（高精度）。

**约束：**

- **实现检查 (A2A3/A5)**
  - `src0`、`src1` 和 `dst` 元素类型必须一致。
  - tile 元素类型必须为 `i32`、`i16`、`f16` 或 `f32`。
  - 三个 tile 必须满足二元 tile 操作的形状/有效区域一致性检查。
  - tile 必须使用行优先布局 (`blayout=row_major`)。

**示例:**

```mlir
pto.tfmod ins(%a, %b : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>,
              !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>)
          outs(%c : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
              v_row=16, v_col=16, blayout=row_major, slayout=none_box,
              fractal=512, pad=0>)
```
