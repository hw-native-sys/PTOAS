# **复杂操作（Complex Operations）**

本节描述了 PTO ISA 中的复杂操作，包括连续整数序列生成、聚集/散射数据重排、部分有效区域逐元素运算、排序和填充等操作。

通用装配形式：

```
pto.op ins(<src>, ... : <src_type>, ...) outs(<dst> : <dst_type>)
```

---

## 目录

- [`pto.tci` — 连续整数序列生成](#ptotci--连续整数序列生成)
- [`pto.tgatherb` — 按字节偏移聚集](#ptotgatherb--按字节偏移聚集)
- [`pto.tgather` — 聚集/选择元素](#ptotgather--聚集选择元素)
- [`pto.tmrgsort` — 归并排序](#ptotmrgsort--归并排序)
- [`pto.tpartadd` — 部分逐元素加法](#ptotpartadd--部分逐元素加法)
- [`pto.tpartmul` — 部分逐元素乘法](#ptotpartmul--部分逐元素乘法)
- [`pto.tpartmax` — 部分逐元素取最大值](#ptotpartmax--部分逐元素取最大值)
- [`pto.tpartmin` — 部分逐元素取最小值](#ptotpartmin--部分逐元素取最小值)
- [`pto.tscatter` — 散射元素](#ptotscatter--散射元素)
- [`pto.tsort32` — 32 元素块排序](#ptotsort32--32-元素块排序)
- [`pto.tpartargmax` — 部分逐元素取最大值及索引](#ptotpartargmax--部分逐元素取最大值及索引)
- [`pto.tpartargmin` — 部分逐元素取最小值及索引](#ptotpartargmin--部分逐元素取最小值及索引)
- [`pto.thistogram` — 逐行直方图累加](#ptothistogram--逐行直方图累加)
- [`pto.trandom` — 随机数生成](#ptotrandom--随机数生成)
- [`pto.ttri` — 三角掩码生成](#ptottri--三角掩码生成)

---

## 操作详解

### `pto.tci` — 连续整数序列生成

```
pto.tci ins(<S> {descending = <bool>} : <int_type>)
        outs(<dst> : <dst_type>)
```

**语义：**

```
For each element at linear_index:
    if descending == false:
        dst[linear_index] = S + linear_index
    else:
        dst[linear_index] = S + (total_elements - 1 - linear_index)
```

**参数:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `S` | `AnyInteger` | 起始整数值 |
| `dst` | `pto.tile_buf` | 目标 tile 缓冲区 |

**返回值:** 无。以 DPS 的形式写入 `dst`。

**属性:**

- `descending` — 是否生成降序序列。默认值为 `false`。
  - `false` — 生成升序序列（S, S+1, S+2, ...）
  - `true` — 生成降序序列

**约束：**

- **实现检查 (A2A3/A5)**
  - `dst` 元素类型必须为整数类型，且仅支持 `i16` 或 `i32`。
  - `S` 的类型必须与 `dst` 元素类型完全一致。
  - `dst` 必须为 rank-2。
  - `dst.cols` 不能为 1。

**示例:**

```mlir
// 生成 i16 升序序列
pto.tci ins(%c0_i16 : i16)
        outs(%tile : !pto.tile_buf<loc=vec, dtype=i16, rows=1, cols=16,
            v_row=1, v_col=16, blayout=row_major, slayout=none_box,
            fractal=512, pad=1>)
```

---

### `pto.tgatherb` — 按字节偏移聚集

```
pto.tgatherb ins(<src>, <offsets> : <src_type>, <offsets_type>)
             outs(<dst> : <dst_type>)
```

**语义：**

```
For each element (i, j):
    dst[i, j] = src[byte_offset = offsets[i, j]]
```

**参数:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | 源 tile 缓冲区 |
| `offsets` | `pto.tile_buf` | 字节偏移 tile，每个元素表示从 `src` 起始地址的字节偏移量 |
| `dst` | `pto.tile_buf` | 目标 tile 缓冲区 |

**返回值:** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查 (A2A3)**
  - `dst` 必须使用行主序布局（`blayout=row_major`）。
  - `dst` 元素大小必须为 1、2 或 4 字节。

- **实现检查 (A5)**
  - `dst` 元素大小必须为 1、2 或 4 字节。
  - 无行主序布局要求。

**示例:**

```mlir
pto.tgatherb ins(%src, %offsets :
                 !pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=32,
                     v_row=8, v_col=32, blayout=row_major, slayout=none_box,
                     fractal=512, pad=0>,
                 !pto.tile_buf<loc=vec, dtype=i32, rows=8, cols=32,
                     v_row=8, v_col=32, blayout=row_major, slayout=none_box,
                     fractal=512, pad=0>)
             outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=32,
                     v_row=8, v_col=32, blayout=row_major, slayout=none_box,
                     fractal=512, pad=0>)
```

---

### `pto.tgather` — 聚集/选择元素

`pto.tgather` 有三种使用形式：索引形式、比较形式和掩码形式。

```
// 索引形式
pto.tgather ins(<src>, <indices>, <tmp> : <src_type>, <indices_type>, <tmp_type>)
            outs(<dst> : <dst_type>)

// 比较形式
pto.tgather ins(<src>, <kValue>, <tmp> : <src_type>, <scalar_type>, <tmp_type>)
            outs(<dst>, <cdst> : <dst_type>, <cdst_type>)
            {cmpMode = #pto<cmp <mode>>, offset = <i32>}

// 掩码形式
pto.tgather ins(<src>, {maskPattern = #pto.mask_pattern<<pattern>>} : <src_type>)
            outs(<dst> : <dst_type>)
```

**语义：**

```
索引形式：
    For each element (i, j):
        dst[i, j] = src[indices[i, j]]

比较形式：
    dst 存储满足标量比较条件的索引
    cdst 存储每行选中的元素个数

掩码形式：
    For each element (i, j):
        dst[i, j] = src[...] 按掩码模式选取
```

**参数:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | 源 tile 缓冲区 |
| `dst` | `pto.tile_buf` | 主目标 tile 缓冲区 |
| `cdst` | `pto.tile_buf` | 比较形式中的辅助目标 tile（仅比较形式） |
| `indices` | `pto.tile_buf` | 索引形式中的索引 tile（仅索引形式） |
| `tmp` | `pto.tile_buf` | 索引形式和比较形式中的临时 tile（仅索引/比较形式） |
| `kValue` | 标量类型 | 比较形式中的标量比较值（仅比较形式） |

**返回值:** 无。以 DPS 的形式写入 `dst`（及比较形式下的 `cdst`）。

**属性:**

- `maskPattern` — 掩码模式，仅用于掩码形式。
  - `#pto.mask_pattern<P0101>` — 按 0101 模式选取
  - `#pto.mask_pattern<P1010>` — 按 1010 模式选取
  - `#pto.mask_pattern<P0001>` — 按 0001 模式选取
  - `#pto.mask_pattern<P0010>` — 按 0010 模式选取
  - `#pto.mask_pattern<P0100>` — 按 0100 模式选取
  - `#pto.mask_pattern<P1000>` — 按 1000 模式选取
  - `#pto.mask_pattern<P1111>` — 按 1111 模式选取（全选）

- `cmpMode` — 比较模式，仅用于比较形式。默认值为 `eq`。
  - `#pto<cmp eq>` — 相等比较
  - `#pto<cmp gt>` — 大于比较

- `offset` — 比较形式中的聚集基索引偏移。默认值为 `0`。

**约束：**

- **实现检查 (A2A3)**
  - 索引形式：`src` 和 `dst` 元素类型必须一致，且为 `i16`、`i32`、`f16` 或 `f32` 之一。`indices` 元素类型必须为 `i32`。`tmp` 元素类型必须与 `indices` 一致。`dst` 的 `valid_shape[1]` 必须等于 `dst.cols`。
  - 比较形式：`dst` 和 `cdst` 元素类型必须为 `i32`。`src` 元素类型必须为 `f16`、`f32`，或当 `cmpMode=eq` 时可为 `i32`。`kValue` 类型必须与 `src` 元素类型一致。`cmpMode` 必须为 `eq` 或 `gt`。`src`、`dst`、`cdst`、`tmp` 必须为 `loc=vec`。
  - 掩码形式：`src` 元素大小必须为 2 或 4 字节。`src` 和 `dst` 必须使用 `loc=vec` 和 `blayout=row_major`。`src` 和 `dst` 元素大小必须一致。`dst` 的 `valid_shape[1]` 必须等于 `dst.cols`。

- **实现检查 (A5)**
  - 索引形式：`src` 和 `dst` 元素类型必须一致，且为 `i8`、`i16`、`i32`、`f16` 或 `f32` 之一。`indices` 元素类型可为 `i16` 或 `i32`。`dst` 的 `valid_shape[1]` 必须等于 `dst.cols`。
  - 比较形式：`dst` 和 `cdst` 元素类型必须为 `i32`。`src` 元素类型必须为 `i16`、`i32`、`f16` 或 `f32` 之一。`kValue` 类型必须与 `src` 元素类型一致。`cmpMode` 必须为 `eq` 或 `gt`。`src`、`dst`、`cdst`、`tmp` 必须为 `loc=vec`。
  - 掩码形式：`src` 元素大小必须为 1、2 或 4 字节。`src` 和 `dst` 必须使用 `loc=vec` 和 `blayout=row_major`。`src`/`dst` 元素类型必须为 `i8`、`i16`、`i32`、`f16`、`bf16`、`f32` 或 fp8 类支持类型之一。`src` 和 `dst` 元素大小必须一致。`dst` 的 `valid_shape[1]` 必须等于 `dst.cols`。

**示例:**

```mlir
// 索引形式
pto.tgather ins(%src, %indices, %index_tmp :
                !pto.tile_buf<loc=vec, dtype=f16, rows=1, cols=32,
                    v_row=1, v_col=32, blayout=row_major, slayout=none_box,
                    fractal=512, pad=0>,
                !pto.tile_buf<loc=vec, dtype=i32, rows=1, cols=32,
                    v_row=1, v_col=32, blayout=row_major, slayout=none_box,
                    fractal=512, pad=0>,
                !pto.tile_buf<loc=vec, dtype=i32, rows=1, cols=32,
                    v_row=1, v_col=32, blayout=row_major, slayout=none_box,
                    fractal=512, pad=0>)
            outs(%index_dst : !pto.tile_buf<loc=vec, dtype=f16, rows=1, cols=32,
                    v_row=1, v_col=32, blayout=row_major, slayout=none_box,
                    fractal=512, pad=0>)
```

---

### `pto.tmrgsort` — 归并排序

`pto.tmrgsort` 有两种格式：单列表归并排序（format1）和多列表归并排序（format2）。

```
// format1：单列表归并排序
pto.tmrgsort ins(<src>, <blockLen> : <src_type>, <int_type>)
             outs(<dst> : <dst_type>)

// format2：多列表归并排序（2~4 路）
pto.tmrgsort ins(<src0>, <src1>, ... , <tmp> {exhausted = <bool>} :
                 <src_type>, <src_type>, ... , <tmp_type>)
             outs(<dst>, <excuted> : <dst_type>, vector<4xi16>)
```

**语义：**

```
format1：
    dst = merge_sort(src, blockLen)
    // 对 src 中每 blockLen*4 个元素为一组，按 blockLen 长度的有序子块进行归并排序

format2：
    dst = merge(src0, src1, ...)
    // 将 2~4 个已排序的输入列表归并为单个有序输出
    excuted = 每路消耗的元素计数
```

**参数:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` / `src0..src3` | `pto.tile_buf` | 输入 tile，format2 支持 2~4 个源 |
| `blockLen` | `AnyInteger` | format1 中的块长度 |
| `dst` | `pto.tile_buf` | 输出 tile |
| `tmp` | `pto.tile_buf` | format2 中的临时 tile（仅 format2） |
| `excuted` | `vector<4xi16>` | format2 中输出的每路消耗计数向量（仅 format2） |

**返回值:** 无。以 DPS 的形式写入 `dst`。

**属性:**

- `exhausted` — 是否使用耗尽模式（format2 中使用）。默认值为 `false`。
  - `false` — 非耗尽模式
  - `true` — 耗尽模式，可接受额外的源操作数

**约束：**

- **实现检查 (A2A3/A5)**
  - format1：元素类型必须为 `f16` 或 `f32`，且 `src` 和 `dst` 元素类型必须一致。`src` 和 `dst` 必须为 rank-2，且 `rows == 1`（数据存储在单行中）。`src` 和 `dst` 的 `cols` 必须一致。`blockLen` 必须大于 0 且为 64 的整数倍。`src` 有效列数必须为 `blockLen * 4` 的整数倍。`repeatTimes = src 有效列数 / (blockLen * 4)` 必须在 `[1, 255]` 范围内。
  - format2：接受 2 路、3 路或 4 路归并。`dst` 和 `tmp` 元素类型和 shape 必须一致。所有 `src` 的元素类型必须与 `dst`/`tmp` 一致，且为 `f16` 或 `f32`。所有 tile 必须为 rank-2，且 `rows == 1`。`tmp.cols >= dst.cols`。`excuted` 必须为 `vector<4xi16>` 类型。

**示例:**

```mlir
// format2：2 路归并排序
pto.tmrgsort ins(%src0, %src1, %tmp2 {exhausted = false} :
                 !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=128,
                     v_row=1, v_col=128, blayout=row_major, slayout=none_box,
                     fractal=512, pad=0>,
                 !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=128,
                     v_row=1, v_col=128, blayout=row_major, slayout=none_box,
                     fractal=512, pad=0>,
                 !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=256,
                     v_row=1, v_col=256, blayout=row_major, slayout=none_box,
                     fractal=512, pad=0>)
             outs(%dst2, %ex :
                 !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=256,
                     v_row=1, v_col=256, blayout=row_major, slayout=none_box,
                     fractal=512, pad=0>,
                 vector<4xi16>)
```

---

### `pto.tpartadd` — 部分逐元素加法

```
pto.tpartadd ins(<src0>, <src1> : <src0_type>, <src1_type>)
             outs(<dst> : <dst_type>)
```

**语义：**

```
For each element (i, j) in the valid region:
    dst[i, j] = src0[i, j] + src1[i, j]

有效区域为各 tile 通过 `v_row`/`v_col` 定义的有效矩形的交集；
当 src0 和 src1 有效区域不同时，非重叠区域的行为由实现定义。
```

**参数:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | 第一个源 tile 缓冲区 |
| `src1` | `pto.tile_buf` | 第二个源 tile 缓冲区 |
| `dst` | `pto.tile_buf` | 目标 tile 缓冲区 |

**返回值:** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查 (A2A3)**
  - `dst`/`src0`/`src1` 元素类型必须一致，且为 `i32`、`i16`、`f16` 或 `f32` 之一。
  - 三个 tile 必须为 rank-2。
  - 要求至少一个输入的有效区域与 `dst` 的有效区域一致，另一个输入的有效区域不超过 `dst` 的有效区域。

- **实现检查 (A5)**
  - `dst`/`src0`/`src1` 元素类型必须一致，且为 `i8`、`i16`、`i32`、`f16`、`bf16` 或 `f32` 之一。
  - 三个 tile 必须为 rank-2。
  - 仅支持特定的部分有效区域模式（例如一个源等于 `dst`，另一个源在 valid-rows 或 valid-cols 上小于 `dst`）。

**示例:**

```mlir
pto.tpartadd ins(%a, %b :
                 !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64,
                     v_row=16, v_col=64, blayout=row_major, slayout=none_box,
                     fractal=512, pad=0>,
                 !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64,
                     v_row=16, v_col=64, blayout=row_major, slayout=none_box,
                     fractal=512, pad=0>)
             outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64,
                     v_row=16, v_col=64, blayout=row_major, slayout=none_box,
                     fractal=512, pad=0>)
```

---

### `pto.tpartmul` — 部分逐元素乘法

```
pto.tpartmul ins(<src0>, <src1> : <src0_type>, <src1_type>)
             outs(<dst> : <dst_type>)
```

**语义：**

```
For each element (i, j) in the valid region:
    dst[i, j] = src0[i, j] * src1[i, j]

有效区域为各 tile 通过 `v_row`/`v_col` 定义的有效矩形的交集；
当 src0 和 src1 有效区域不同时，非重叠区域的行为由实现定义。
```

**参数:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | 第一个源 tile 缓冲区 |
| `src1` | `pto.tile_buf` | 第二个源 tile 缓冲区 |
| `dst` | `pto.tile_buf` | 目标 tile 缓冲区 |

**返回值:** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查 (A2A3)**
  - `dst`/`src0`/`src1` 元素类型必须一致，且为 `i32`、`i16`、`f16` 或 `f32` 之一。
  - 三个 tile 必须为 rank-2。
  - 要求至少一个输入的有效区域与 `dst` 的有效区域一致，另一个输入的有效区域不超过 `dst` 的有效区域。

- **实现检查 (A5)**
  - `dst`/`src0`/`src1` 元素类型必须一致，且为 `i8`、`i16`、`i32`、`f16`、`bf16` 或 `f32` 之一。
  - 三个 tile 必须为 rank-2。
  - 要求 `src0` 和 `src1` 的有效区域在两个维度上均不超过 `dst` 的有效区域。

**示例:**

```mlir
pto.tpartmul ins(%src0, %src1 :
                 !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32,
                     v_row=32, v_col=32, blayout=row_major, slayout=none_box,
                     fractal=512, pad=0>,
                 !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32,
                     v_row=32, v_col=32, blayout=row_major, slayout=none_box,
                     fractal=512, pad=0>)
             outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32,
                     v_row=32, v_col=32, blayout=row_major, slayout=none_box,
                     fractal=512, pad=0>)
```

---

### `pto.tpartmax` — 部分逐元素取最大值

```
pto.tpartmax ins(<src0>, <src1> : <src0_type>, <src1_type>)
             outs(<dst> : <dst_type>)
```

**语义：**

```
For each element (i, j) in the valid region:
    dst[i, j] = max(src0[i, j], src1[i, j])

有效区域为各 tile 通过 `v_row`/`v_col` 定义的有效矩形的交集；
当 src0 和 src1 有效区域不同时，非重叠区域的行为由实现定义。
```

**参数:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | 第一个源 tile 缓冲区 |
| `src1` | `pto.tile_buf` | 第二个源 tile 缓冲区 |
| `dst` | `pto.tile_buf` | 目标 tile 缓冲区 |

**返回值:** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查 (A2A3)**
  - `dst`/`src0`/`src1` 元素类型必须一致，且为 `i32`、`i16`、`f16` 或 `f32` 之一。
  - 三个 tile 必须为 rank-2，且 shape 一致。
  - 要求至少一个输入的有效区域与 `dst` 的有效区域一致，另一个输入的有效区域不超过 `dst` 的有效区域。

- **实现检查 (A5)**
  - `dst`/`src0`/`src1` 元素类型必须一致，且为 `i8`、`i16`、`i32`、`f16`、`bf16` 或 `f32` 之一。
  - 三个 tile 必须为 rank-2，且 shape 一致。
  - 要求 `src0` 和 `src1` 的有效区域在两个维度上均不超过 `dst` 的有效区域。

**示例:**

```mlir
pto.tpartmax ins(%a, %b :
                 !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64,
                     v_row=16, v_col=64, blayout=row_major, slayout=none_box,
                     fractal=512, pad=0>,
                 !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64,
                     v_row=16, v_col=64, blayout=row_major, slayout=none_box,
                     fractal=512, pad=0>)
             outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64,
                     v_row=16, v_col=64, blayout=row_major, slayout=none_box,
                     fractal=512, pad=0>)
```

---

### `pto.tpartmin` — 部分逐元素取最小值

```
pto.tpartmin ins(<src0>, <src1> : <src0_type>, <src1_type>)
             outs(<dst> : <dst_type>)
```

**语义：**

```
For each element (i, j) in the valid region:
    dst[i, j] = min(src0[i, j], src1[i, j])

有效区域为各 tile 通过 `v_row`/`v_col` 定义的有效矩形的交集；
当 src0 和 src1 有效区域不同时，非重叠区域的行为由实现定义。
```

**参数:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | 第一个源 tile 缓冲区 |
| `src1` | `pto.tile_buf` | 第二个源 tile 缓冲区 |
| `dst` | `pto.tile_buf` | 目标 tile 缓冲区 |

**返回值:** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查 (A2A3)**
  - `dst`/`src0`/`src1` 元素类型必须一致，且为 `i32`、`i16`、`f16` 或 `f32` 之一。
  - 三个 tile 必须为 rank-2，且 shape 一致。
  - 要求至少一个输入的有效区域与 `dst` 的有效区域一致，另一个输入的有效区域不超过 `dst` 的有效区域。

- **实现检查 (A5)**
  - `dst`/`src0`/`src1` 元素类型必须一致，且为 `i8`、`i16`、`i32`、`f16`、`bf16` 或 `f32` 之一。
  - 三个 tile 必须为 rank-2，且 shape 一致。
  - 要求 `src0` 和 `src1` 的有效区域在两个维度上均不超过 `dst` 的有效区域。

**示例:**

```mlir
pto.tpartmin ins(%a, %b :
                 !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64,
                     v_row=16, v_col=64, blayout=row_major, slayout=none_box,
                     fractal=512, pad=0>,
                 !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64,
                     v_row=16, v_col=64, blayout=row_major, slayout=none_box,
                     fractal=512, pad=0>)
             outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64,
                     v_row=16, v_col=64, blayout=row_major, slayout=none_box,
                     fractal=512, pad=0>)
```

---

### `pto.tscatter` — 散射元素

`pto.tscatter` 有两种使用形式：索引形式和掩码形式。

```
// 索引形式
pto.tscatter ins(<src>, <indexes> : <src_type>, <indexes_type>)
             outs(<dst> : <dst_type>)

// 掩码形式
pto.tscatter ins(<src>, {maskPattern = #pto.mask_pattern<<pattern>>} : <src_type>)
             outs(<dst> : <dst_type>)
```

**语义：**

```
索引形式：
    For each element (i, j):
        dst[indexes[i], j] = src[i, j]

掩码形式：
    按掩码模式将 src 元素分散写入 dst
```

**参数:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | 源 tile 缓冲区 |
| `indexes` | `pto.tile_buf` | 索引形式中的行索引 tile（仅索引形式） |
| `dst` | `pto.tile_buf` | 目标 tile 缓冲区 |

**返回值:** 无。以 DPS 的形式写入 `dst`。

**属性:**

- `maskPattern` — 掩码模式，仅用于掩码形式。
  - `#pto.mask_pattern<P0101>` — 按 0101 模式散射（扩展因子 2）
  - `#pto.mask_pattern<P1010>` — 按 1010 模式散射（扩展因子 2）
  - `#pto.mask_pattern<P0001>` — 按 0001 模式散射（扩展因子 4）
  - `#pto.mask_pattern<P0010>` — 按 0010 模式散射（扩展因子 4）
  - `#pto.mask_pattern<P0100>` — 按 0100 模式散射（扩展因子 4）
  - `#pto.mask_pattern<P1000>` — 按 1000 模式散射（扩展因子 4）
  - `#pto.mask_pattern<P1111>` — 按 1111 模式散射（扩展因子 1，即全量复制）

**约束：**

- **实现检查 (A2A3)**
  - 索引形式：`src`、`dst` 和 `indexes` 必须为 `loc=vec`。`src`/`dst` 元素类型必须一致，且为 `i8`、`i16`、`i32`、`f16`、`bf16` 或 `f32` 之一。`indexes` 元素类型必须为 `i16` 或 `i32`。当 `dst` 元素大小为 4 字节时，`indexes` 元素大小也必须为 4 字节；2 字节时也必须为 2 字节；1 字节时 `indexes` 必须为 2 字节。不对 `indexes` 中的值进行越界检查。索引形式在 A2/A3 上降低到标量 UB 循环（`PIPE_S`）。
  - 掩码形式：`src` 和 `dst` 必须为 `loc=vec` 且 `blayout=row_major`。`src` 和 `dst` 元素类型必须一致，且为 `i8`、`i16`、`i32`、`f16`、`bf16` 或 `f32` 之一。`src` 和 `dst` 的有效行数必须一致。`src` 的有效列数必须等于 `dst` 有效列数乘以掩码扩展因子。

- **实现检查 (A5)**
  - 索引形式：约束与 A2A3 索引形式相同。索引形式在 A5 上使用向量散射（`PIPE_V`）。
  - 掩码形式：A5 不支持掩码形式。

**示例:**

```mlir
// 掩码形式（A2A3）
pto.tscatter ins(%src, {maskPattern = #pto.mask_pattern<P0101>} :
                 !pto.tile_buf<loc=vec, dtype=f16, rows=1, cols=64,
                     v_row=1, v_col=64, blayout=row_major, slayout=none_box,
                     fractal=512, pad=0>)
             outs(%dst : !pto.tile_buf<loc=vec, dtype=f16, rows=1, cols=32,
                     v_row=1, v_col=32, blayout=row_major, slayout=none_box,
                     fractal=512, pad=0>)
```

---

### `pto.tsort32` — 32 元素块排序

```
// 基本形式
pto.tsort32 ins(<src>, <idx> : <src_type>, <idx_type>)
            outs(<dst> : <dst_type>)

// 带临时 tile 的形式
pto.tsort32 ins(<src>, <idx>, <tmp> : <src_type>, <idx_type>, <tmp_type>)
            outs(<dst> : <dst_type>)
```

**语义：**

```
dst = sort(src, idx)
// 对 src 中固定 32 元素块进行排序
// idx 为索引 tile，与 src 的值一起按排序结果进行排列
```

**参数:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | 输入值 tile |
| `idx` | `pto.tile_buf` | 输入索引 tile，与 `src` 一起排列 |
| `tmp` | `pto.tile_buf` | 临时 scratch tile（可选） |
| `dst` | `pto.tile_buf` | 输出 tile，存储排序后的值-索引对 |

**返回值:** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查 (A2A3/A5)**
  - `src` 和 `dst` 元素类型必须一致，且为 `f16` 或 `f32`。
  - `idx` 元素类型必须为 32 位无符号整数（MLIR 中表示为 `ui32` 或 `i32`）。
  - `src`、`dst` 和 `idx` 必须为 `loc=vec` 和 `blayout=row_major`。

**示例:**

```mlir
pto.tsort32 ins(%src, %idx :
                !pto.tile_buf<loc=vec, dtype=f16, rows=1, cols=32,
                    v_row=1, v_col=32, blayout=row_major, slayout=none_box,
                    fractal=512, pad=0>,
                !pto.tile_buf<loc=vec, dtype=ui32, rows=1, cols=32,
                    v_row=1, v_col=32, blayout=row_major, slayout=none_box,
                    fractal=512, pad=0>)
            outs(%dst0 : !pto.tile_buf<loc=vec, dtype=f16, rows=1, cols=64,
                    v_row=1, v_col=64, blayout=row_major, slayout=none_box,
                    fractal=512, pad=0>)
```

---

### `pto.tpartargmax` — 部分逐元素取最大值及索引

```
pto.tpartargmax ins(<src0>, <src1>, <src0Idx>, <src1Idx>
                    : <src0_type>, <src1_type>, <idx0_type>, <idx1_type>)
                outs(<dst>, <dstIdx> : <dst_type>, <dstIdx_type>)
```

**语义：**

```
For each element (i, j):
    if src0[i, j] >= src1[i, j]:
        dst[i, j] = src0[i, j]
        dstIdx[i, j] = src0Idx[i, j]
    else:
        dst[i, j] = src1[i, j]
        dstIdx[i, j] = src1Idx[i, j]
```

**参数:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | 第一组值 tile |
| `src1` | `pto.tile_buf` | 第二组值 tile |
| `src0Idx` | `pto.tile_buf` | 第一组索引 tile（`ui32`） |
| `src1Idx` | `pto.tile_buf` | 第二组索引 tile（`ui32`） |
| `dst` | `pto.tile_buf` | 输出值 tile |
| `dstIdx` | `pto.tile_buf` | 输出索引 tile（`ui32`） |

**返回值:** 无。以 DPS 的形式写入 `dst` 和 `dstIdx`。

**约束：**

- **实现检查 (A2A3/A5)**
  - 所有值 tile（src0、src1、dst）的元素类型必须一致，为 `f16` 或 `f32`。
  - 所有索引 tile（src0Idx、src1Idx、dstIdx）的元素类型必须为 `ui32`。
  - 所有 tile 必须使用 `loc=vec`。

**示例:**

```mlir
pto.tpartargmax
    ins(%src0, %src1, %src0_idx, %src1_idx :
        !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=32,
                      v_row=16, v_col=32, blayout=row_major,
                      slayout=none_box, fractal=512, pad=0>,
        !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=32,
                      v_row=16, v_col=32, blayout=row_major,
                      slayout=none_box, fractal=512, pad=0>,
        !pto.tile_buf<loc=vec, dtype=ui32, rows=16, cols=32,
                      v_row=16, v_col=32, blayout=row_major,
                      slayout=none_box, fractal=512, pad=0>,
        !pto.tile_buf<loc=vec, dtype=ui32, rows=16, cols=32,
                      v_row=16, v_col=32, blayout=row_major,
                      slayout=none_box, fractal=512, pad=0>)
    outs(%dst, %dst_idx :
        !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=32,
                      v_row=16, v_col=32, blayout=row_major,
                      slayout=none_box, fractal=512, pad=0>,
        !pto.tile_buf<loc=vec, dtype=ui32, rows=16, cols=32,
                      v_row=16, v_col=32, blayout=row_major,
                      slayout=none_box, fractal=512, pad=0>)
```

---

### `pto.tpartargmin` — 部分逐元素取最小值及索引

```
pto.tpartargmin ins(<src0>, <src1>, <src0Idx>, <src1Idx>
                    : <src0_type>, <src1_type>, <idx0_type>, <idx1_type>)
                outs(<dst>, <dstIdx> : <dst_type>, <dstIdx_type>)
```

**语义：**

```
For each element (i, j):
    if src0[i, j] <= src1[i, j]:
        dst[i, j] = src0[i, j]
        dstIdx[i, j] = src0Idx[i, j]
    else:
        dst[i, j] = src1[i, j]
        dstIdx[i, j] = src1Idx[i, j]
```

**参数:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | 第一组值 tile |
| `src1` | `pto.tile_buf` | 第二组值 tile |
| `src0Idx` | `pto.tile_buf` | 第一组索引 tile（`ui32`） |
| `src1Idx` | `pto.tile_buf` | 第二组索引 tile（`ui32`） |
| `dst` | `pto.tile_buf` | 输出值 tile |
| `dstIdx` | `pto.tile_buf` | 输出索引 tile（`ui32`） |

**返回值:** 无。以 DPS 的形式写入 `dst` 和 `dstIdx`。

**约束：**

- **实现检查 (A2A3/A5)**
  - 所有值 tile（src0、src1、dst）的元素类型必须一致，为 `f16` 或 `f32`。
  - 所有索引 tile（src0Idx、src1Idx、dstIdx）的元素类型必须为 `ui32`。
  - 所有 tile 必须使用 `loc=vec`。

**示例:**

```mlir
pto.tpartargmin
    ins(%src0, %src1, %src0_idx, %src1_idx :
        !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=32,
                      v_row=16, v_col=32, blayout=row_major,
                      slayout=none_box, fractal=512, pad=0>,
        !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=32,
                      v_row=16, v_col=32, blayout=row_major,
                      slayout=none_box, fractal=512, pad=0>,
        !pto.tile_buf<loc=vec, dtype=ui32, rows=16, cols=32,
                      v_row=16, v_col=32, blayout=row_major,
                      slayout=none_box, fractal=512, pad=0>,
        !pto.tile_buf<loc=vec, dtype=ui32, rows=16, cols=32,
                      v_row=16, v_col=32, blayout=row_major,
                      slayout=none_box, fractal=512, pad=0>)
    outs(%dst, %dst_idx :
        !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=32,
                      v_row=16, v_col=32, blayout=row_major,
                      slayout=none_box, fractal=512, pad=0>,
        !pto.tile_buf<loc=vec, dtype=ui32, rows=16, cols=32,
                      v_row=16, v_col=32, blayout=row_major,
                      slayout=none_box, fractal=512, pad=0>)
```

---

### `pto.thistogram` — 逐行直方图累加

```
pto.thistogram ins(<src>, <idx> : <src_type>, <idx_type>)
               outs(<dst> : <dst_type>)
```

**语义：**

```
For each row i:
    bin = select_bin(src[i, :], idx[i, 0], isMSB)
    dst[i, bin] += 1
// 逐行对源 tile 按索引确定的位段进行 256-bin 直方图累加
```

**参数:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | 源数据 tile（`ui16`） |
| `idx` | `pto.tile_buf` | 索引 tile，指定位选择（`ui8`，单列） |
| `dst` | `pto.tile_buf` | 目标直方图 tile（`ui32`，列数为 256） |

**返回值:** 无。以 DPS 的形式写入 `dst`。

**属性:**

- `isMSB` — 是否选择高位字节。默认值为 `true`。
  - `true` — 选择高 8 位
  - `false` — 选择低 8 位

**约束：**

- **实现检查 (A2A3)**
  - 不支持，仅 A5 可用。

- **实现检查 (A5)**
  - `src` 元素类型必须为 `ui16`。
  - `idx` 元素类型必须为 `ui8`，且为单列（cols=1）。
  - `dst` 元素类型必须为 `ui32`，且列数为 256。
  - 所有 tile 必须使用 `loc=vec`。

**示例:**

```mlir
pto.thistogram
    ins(%src, %idx :
        !pto.tile_buf<loc=vec, dtype=ui16, rows=32, cols=32,
                      v_row=8, v_col=32, blayout=row_major,
                      slayout=none_box, fractal=512, pad=0>,
        !pto.tile_buf<loc=vec, dtype=ui8, rows=32, cols=1,
                      v_row=8, v_col=1, blayout=col_major,
                      slayout=none_box, fractal=512, pad=0>)
    outs(%dst : !pto.tile_buf<loc=vec, dtype=ui32, rows=32, cols=256,
                              v_row=8, v_col=256, blayout=row_major,
                              slayout=none_box, fractal=512, pad=0>)
```

---

### `pto.trandom` — 随机数生成

```
pto.trandom ins(<key0>, <key1>, <counter0>, <counter1>, <counter2>, <counter3>
                : i32, i32, i32, i32, i32, i32)
            outs(<dst> : <dst_type>)
```

**语义：**

```
dst = philox_random(key0, key1, counter0..counter3, rounds)
// 使用 Philox 算法通过 key/counter 对生成伪随机数填充 dst tile
```

**参数:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `key0` | `i32` | 密钥字 0 |
| `key1` | `i32` | 密钥字 1 |
| `counter0` | `i32` | 计数器字 0 |
| `counter1` | `i32` | 计数器字 1 |
| `counter2` | `i32` | 计数器字 2 |
| `counter3` | `i32` | 计数器字 3 |
| `dst` | `pto.tile_buf` | 目标 tile buffer（`i32`/`ui32`） |

**返回值:** 无。以 DPS 的形式写入 `dst`。

**属性:**

- `rounds` — Philox 迭代轮次。默认值为 `10`。
  - `7` — 7 轮（较快，随机性稍弱）
  - `10` — 10 轮（标准）

**约束：**

- **实现检查 (A2A3)**
  - 不支持，仅 A5 可用。

- **实现检查 (A5)**
  - 所有 key/counter 操作数必须为 `i32`/`ui32`。
  - `dst` 元素类型必须为 `i32` 或 `ui32`。
  - `dst` 必须使用 `blayout=row_major`。
  - `rounds` 必须为 7 或 10。

**示例:**

```mlir
pto.trandom
    ins(%k0, %k1, %c0, %c1, %c2, %c3 : i32, i32, i32, i32, i32, i32)
    outs(%dst : !pto.tile_buf<loc=vec, dtype=i32, rows=4, cols=256,
                              v_row=4, v_col=256, blayout=row_major,
                              slayout=none_box, fractal=512, pad=0>)

// 使用 7 轮
pto.trandom
    ins(%k0, %k1, %c0, %c1, %c2, %c3 {rounds = 7 : i32}
        : i32, i32, i32, i32, i32, i32)
    outs(%dst : !pto.tile_buf<loc=vec, dtype=ui32, rows=2, cols=256,
                              v_row=2, v_col=256, blayout=row_major,
                              slayout=none_box, fractal=512, pad=0>)
```

---

### `pto.ttri` — 三角掩码生成

```
pto.ttri ins(<diagonal> : <integer_type>)
         outs(<dst> : <dst_type>)
```

**语义：**

```
For each element (i, j):
    if upperOrLower == 0:  // lower triangular
        dst[i, j] = (j <= i + diagonal) ? max_value : 0
    else:                  // upper triangular
        dst[i, j] = (j >= i + diagonal) ? max_value : 0
```

**参数:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `diagonal` | 整数类型（`i32`） | 对角线偏移 |
| `dst` | `pto.tile_buf` | 目标掩码 tile |

**返回值:** 无。以 DPS 的形式写入 `dst`。

**属性:**

- `upperOrLower` — 三角类型。默认值为 `0`。
  - `0` — 下三角掩码
  - `1` — 上三角掩码

**约束：**

- **实现检查 (A2A3)**
  - `dst` 元素类型必须为 `f16`、`f32`、`i16`、`i32`、`u16` 或 `u32`。
  - `dst` 必须使用 `loc=vec`。
  - `upperOrLower` 必须为 0 或 1。

- **实现检查 (A5)**
  - `dst` 元素类型必须为 `f16`、`f32`、`bf16`、`i8`、`i16`、`i32`、`u8`、`u16` 或 `u32`。
  - `dst` 必须使用 `loc=vec`。
  - `upperOrLower` 必须为 0 或 1。

**示例:**

```mlir
// 下三角掩码
pto.ttri
    ins(%diag : i32)
    outs(%lower : !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32,
                                v_row=32, v_col=32, blayout=row_major,
                                slayout=none_box, fractal=512, pad=0>)

// 上三角掩码
pto.ttri
    ins(%diag {upperOrLower = 1 : i32} : i32)
    outs(%upper : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
                                v_row=16, v_col=16, blayout=row_major,
                                slayout=none_box, fractal=512, pad=0>)
```
