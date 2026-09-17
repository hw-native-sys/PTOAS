# 重排与转换操作

本节描述了 PTO ISA 中用于 Tile 形状重解释、拼接、子区域提取与插入、类型转换、量化与反量化、以及填充处理的操作指令族。所有操作均作用于本地缓冲区（`tile_buf`，位于 `loc=vec` 或 `loc=mat` 空间），采用“目标传递风格”（Destination-Passing Style, DPS）：操作本身不产生 SSA 返回值，而是直接将结果写入预先分配好的目标 `tile_buf`。

这一类操作通常具有如下装配形式：

```mlir
pto.op ins(...) outs(%dst : !pto.tile_buf<...>)
```

通用约束通常包括：

- 操作数与结果 tile 使用相同或兼容的元素类型
- 所有 tile 使用 `loc=vec` 或 `loc=mat`（具体见各操作说明）
- 大小、layout、有效区域等必须满足语义要求

---

## 目录

- [`pto.treshape` — Tile 形状重解释](#ptotreshape--tile-形状重解释)
- [`pto.tconcat` — 列方向 Tile 拼接](#ptotconcat--列方向-tile-拼接)
- [`pto.textract` — 子 Tile 提取](#ptotextract--子-tile-提取)
- [`pto.tinsert` — 子 Tile 插入](#ptotinsert--子-tile-插入)
- [`pto.tquant` — Tile 量化](#ptotquant--tile-量化)
- [`pto.tdequant` — Tile 反量化](#ptotdequant--tile-反量化)
- [`pto.tfillpad` — 填充 Padding 区域](#ptotfillpad--填充-padding-区域)
- [`pto.tfillpad_expand` — 扩展填充 Padding 区域](#ptotfillpad_expand--扩展填充-padding-区域)
- [`pto.tfillpad_inplace` — 原地填充 Padding 区域](#ptotfillpad_inplace--原地填充-padding-区域)
- [`pto.tconcatidx` — 索引控制列拼接](#ptotconcatidx--索引控制列拼接)
- [`pto.textract` 的 `fp` 形式](#ptotextract-的-fp-形式)
- [`pto.tinsert` 的 `fp` 形式](#ptotinsert-的-fp-形式)

---

## 操作详解

### `pto.treshape` — Tile 形状重解释

```mlir
pto.treshape ins(<src> : !pto.tile_buf)
             outs(<dst> : !pto.tile_buf)
```

**语义：**

```text
dst = reinterpret(src)
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | 源 tile buffer |
| `dst` | `pto.tile_buf` | 目标 tile buffer（新的形状和 layout） |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查（A2A3/A5）**
  - 源和目标必须使用相同的存储位置：`src.loc == dst.loc`
  - 源和目标的总字节大小必须相等
  - 不支持有装箱（boxed）与无装箱（non-boxed）layout 之间的转换

**示例：**

```mlir
pto.treshape
    ins(%src : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=32,
                             v_row=16, v_col=32, blayout=row_major,
                             slayout=none_box, fractal=512, pad=0>)
    outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=16,
                              v_row=32, v_col=16, blayout=row_major,
                              slayout=none_box, fractal=512, pad=0>)
```

---

### `pto.tconcat` — 列方向 Tile 拼接

```mlir
pto.tconcat ins(<src0>, <src1> : !pto.tile_buf, !pto.tile_buf)
            outs(<dst> : !pto.tile_buf)
```

**语义：**

```text
For each row i:
    dst[i, 0:C0) = src0[i, 0:C0)
    dst[i, C0:C0+C1) = src1[i, 0:C1)
```

其中 C0 为 src0 的列数，C1 为 src1 的列数。

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | 第一个源 tile buffer（左侧） |
| `src1` | `pto.tile_buf` | 第二个源 tile buffer（右侧） |
| `dst` | `pto.tile_buf` | 目标 tile buffer（拼接结果） |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查（A2A3）**
  - src0、src1 和 dst 必须使用相同的元素类型，且为以下之一：`i8`、`i16`、`i32`、`f16`、`f32`、`bf16`
  - 所有 tile 必须使用 `loc=vec`
  - 三个 tile 必须为 rank-2，且 src0、src1 的有效行数必须与 dst 的有效行数相同
  - src0 的有效列数 + src1 的有效列数 <= dst 的列数
  - 拼接会改变列方向长度，因此三者的物理 static shape 不要求完全相同

- **实现检查（A5）**
  - 同 A2A3 要求，额外要求所有 tile 必须使用 `blayout=row_major`

**示例：**

```mlir
pto.tconcat
    ins(%a, %b : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=16,
                               v_row=32, v_col=16, blayout=row_major,
                               slayout=none_box, fractal=512, pad=0>,
                 !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=16,
                               v_row=32, v_col=16, blayout=row_major,
                               slayout=none_box, fractal=512, pad=0>)
    outs(%c : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32,
                            v_row=32, v_col=32, blayout=row_major,
                            slayout=none_box, fractal=512, pad=0>)
```

---

### `pto.textract` — 子 Tile 提取

```mlir
pto.textract ins(<src> [<indexRow>, <indexCol>] : !pto.tile_buf, index, index)
             outs(<dst> : !pto.tile_buf)
```

**语义：**

```text
For each element (i, j):
    dst[i, j] = src[i + indexRow, j + indexCol]
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | 源 tile buffer |
| `indexRow` | `index` | 提取起始行偏移 |
| `indexCol` | `index` | 提取起始列偏移 |
| `dst` | `pto.tile_buf` | 目标 tile buffer（子区域） |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查（A2A3）**
  - dst 的元素类型必须与 src 相同，且为以下之一：`i8`、`f16`、`bf16`、`f32`
  - 支持 Vec->Vec 转换
  - src 的 layout/fractal 必须与 dst 支持的组合兼容
  - 运行时约束：`indexRow + dst.rows <= src.rows` 且 `indexCol + dst.cols <= src.cols`
  - dst 必须使用 `loc=left` 或 `loc=right`

- **实现检查（A5）**
  - dst 的元素类型必须与 src 相同（int8/fp8/fp16/bf16/f32 family）
  - 支持 Mat->Left/Right/Scaling、Vec->Mat、Acc->Mat/Vec，以及
    ND 布局的 Vec->Vec；ND 指 `blayout=row_major, slayout=none_box`
  - 运行时约束：`indexRow + dst.rows <= src.rows` 且 `indexCol + dst.cols <= src.cols`

**示例：**

```mlir
pto.textract
    ins(%src [%row, %col] :
        !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32,
                      v_row=32, v_col=32, blayout=row_major,
                      slayout=none_box, fractal=512, pad=0>,
        index, index)
    outs(%dst : !pto.tile_buf<loc=left, dtype=f32, rows=16, cols=16,
                              v_row=16, v_col=16, blayout=row_major,
                              slayout=none_box, fractal=256, pad=0>)
```

---

### `pto.tinsert` — 子 Tile 插入

```mlir
pto.tinsert ins(<src>, <indexRow>, <indexCol> : !pto.tile_buf, index, index)
            outs(<dst> : !pto.tile_buf)
```

**语义：**

```text
For each element (i, j):
    dst[i + indexRow, j + indexCol] = src[i, j]
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | 源 tile buffer（待插入） |
| `indexRow` | `index` | 插入起始行偏移 |
| `indexCol` | `index` | 插入起始列偏移 |
| `dst` | `pto.tile_buf` | 目标 tile buffer |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查（A2A3）**
  - `src`、`dst` 必须为 rank-2 `tile_buf`，`indexRow`、`indexCol` 必须非负，且插入区域不能越过 `dst` 静态边界
  - 支持 Vec->Vec（同元素类型，`i8`/`f16`/`bf16`/`f32`）和 Acc->Mat
  - Acc->Mat 要求两端为 NZ 布局（`col_major` + `row_major`），目标 `fractal=512`
  - `fp` 与 `preQuantScalar` 互斥，且仅适用于 `src.loc=acc`；`fp` 必须使用 `loc=scaling`

- **实现检查（A5）**
  - 支持 Acc->Mat/Vec、Vec->Mat 和 Vec->Vec
  - Vec->Vec 两端布局必须同为 ND（`row_major` + `none_box`）或同为 NZ（`col_major` + `row_major`）
  - Vec->Mat 的目标必须为 NZ；源可为 ND 或 NZ，且源、目标元素类型相同
  - `accToVecMode` 仅适用于 Acc->Vec；`tinsertMode` 仅适用于 NZ 的 Vec->Mat
  - `fp`、`preQuantScalar` 和 ReLU 形式要求 `src.loc=acc`；`fp` 必须使用 `loc=scaling`

**示例：**

```mlir
pto.tinsert
    ins(%src, %row, %col :
        !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                      v_row=16, v_col=16, blayout=row_major,
                      slayout=none_box, fractal=256, pad=0>,
        index, index)
    outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32,
                              v_row=32, v_col=32, blayout=row_major,
                              slayout=none_box, fractal=512, pad=0>)
```

---

### `pto.tquant` — Tile 量化

```mlir
pto.tquant ins(<src>, <fp> : !pto.tile_buf, !pto.tile_buf)
           outs(<dst> : !pto.tile_buf) {quant_type = <quant_type>}
```

**语义：**

```text
For each element (i, j):
    dst[i, j] = Quantize(src[i, j]; fp, quant_type)
```

其中 `fp` 为缩放因子 tile（通常为单列或单行）。

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | 源 tile buffer（f32 类型） |
| `fp` | `pto.tile_buf` | 缩放因子 tile buffer |
| `dst` | `pto.tile_buf` | 目标 tile buffer（整数类型） |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**属性：**

- `quant_type` — 量化类型。
  - `#pto<quant_type INT8_SYM>` — 对称量化，dst 为 `i8`
  - `#pto<quant_type INT8_ASYM>` — 非对称量化，dst 为 `ui8`

**约束：**

- **实现检查（A2A3/A5）**
  - src 必须为 `f32` 类型
  - 可选 `offset` 的元素类型必须为 `f32`
  - src 与 dst 的有效 shape 必须一致
  - A2/A3: src 和 dst 必须使用 `blayout=row_major`

**示例：**

```mlir
pto.tquant
    ins(%src, %fp :
        !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32,
                      v_row=32, v_col=32, blayout=row_major,
                      slayout=none_box, fractal=512, pad=0>,
        !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=1,
                      v_row=32, v_col=1, blayout=row_major,
                      slayout=none_box, fractal=512, pad=0>)
    outs(%dst : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32,
                              v_row=32, v_col=32, blayout=row_major,
                              slayout=none_box, fractal=512, pad=0>)
    {quant_type = #pto<quant_type INT8_SYM>}
```

---

### `pto.tdequant` — Tile 反量化

```mlir
pto.tdequant ins(<src>, <scale>, <offset> : !pto.tile_buf, !pto.tile_buf, !pto.tile_buf)
             outs(<dst> : !pto.tile_buf)
```

**语义：**

```text
For each row i:
    For each column j:
        dst[i][j] = (float(src[i][j]) - offset[i][0]) * scale[i][0]
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | 源 tile buffer（整数类型，i8 或 i16） |
| `scale` | `pto.tile_buf` | 缩放因子 tile buffer（通常为单列） |
| `offset` | `pto.tile_buf` | 偏移 tile buffer（通常为单列） |
| `dst` | `pto.tile_buf` | 目标 tile buffer（f32 类型） |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查（A2A3/A5）**
  - `src` 元素类型必须为 8 位或 16 位整数
  - `scale`、`offset` 和 `dst` 的元素类型必须为 `f32`
  - `src`、`scale`、`offset` 和 `dst` 都必须是合法的 rank-2 `tile_buf`
  - A2/A3 额外要求 `src`、`dst` 使用 row-major 布局；A5 无此附加布局限制

**示例：**

```mlir
pto.tdequant
    ins(%src, %scale, %offset :
        !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32,
                      v_row=32, v_col=32, blayout=row_major,
                      slayout=none_box, fractal=512, pad=0>,
        !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=1,
                      v_row=32, v_col=1, blayout=row_major,
                      slayout=none_box, fractal=512, pad=0>,
        !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=1,
                      v_row=32, v_col=1, blayout=row_major,
                      slayout=none_box, fractal=512, pad=0>)
    outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32,
                              v_row=32, v_col=32, blayout=row_major,
                              slayout=none_box, fractal=512, pad=0>)
```

---

### `pto.tfillpad` — 填充 Padding 区域

```mlir
pto.tfillpad ins(<src> : !pto.tile_buf)
             outs(<dst> : !pto.tile_buf)
```

**语义：**

```text
For each element in valid region:
    dst = src
For each element in padded region:
    dst = PadVal(dst)
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | 源 tile buffer |
| `dst` | `pto.tile_buf` | 目标 tile buffer（包含 pad 属性） |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查（A2A3/A5）**
  - `src` 和 `dst` 必须为 rank-2。
  - `dst` 的 `pad` 值不能为 `Null`。
  - `src` 和 `dst` 元素大小必须一致，且为 1、2 或 4 字节。
  - `src` 和 `dst` 必须具有相同的静态 shape（`rows`/`cols` 一致）。

- **特殊行为（loc=mat）**
  - 当 `loc=mat` 时，`src` 和 `dst` 必须可降低到同一个 `TFILLPAD` tile 特化，即 `validShape` 和 `pad` 必须一致。
  - 异构 `TFILLPAD` 重载仅在 `loc=vec` 下可用。

**示例：**

```mlir
pto.tfillpad
    ins(%src : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32,
                             v_row=32, v_col=32, blayout=row_major,
                             slayout=none_box, fractal=512, pad=1>)
    outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32,
                              v_row=32, v_col=32, blayout=row_major,
                              slayout=none_box, fractal=512, pad=1>)
```

---

### `pto.tfillpad_expand` — 扩展填充 Padding 区域

```mlir
pto.tfillpad_expand ins(<src> : !pto.tile_buf)
                    outs(<dst> : !pto.tile_buf)
```

**语义：**

```text
For each element in src valid region:
    dst[i, j] = src[i, j]
For each element in dst padded region:
    dst[i, j] = PadVal(dst)

Constraint: dst.rows >= src.rows and dst.cols >= src.cols
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | 源 tile buffer |
| `dst` | `pto.tile_buf` | 目标 tile buffer（可能更大） |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查（A2A3/A5）**
  - `src` 和 `dst` 必须为 rank-2。
  - `src` 和 `dst` 元素大小必须一致，且为 1、2 或 4 字节。
  - `dst` 的 `pad` 值不能为 `Null`。
  - `dst.rows >= src.rows` 且 `dst.cols >= src.cols`（允许 `dst` 的 shape 大于 `src`）。

**示例：**

```mlir
pto.tfillpad_expand
    ins(%src : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                             v_row=16, v_col=16, blayout=row_major,
                             slayout=none_box, fractal=256, pad=1>)
    outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32,
                              v_row=32, v_col=32, blayout=row_major,
                              slayout=none_box, fractal=512, pad=1>)
```

---

### `pto.tfillpad_inplace` — 原地填充 Padding 区域

```mlir
pto.tfillpad_inplace ins(<src> : !pto.tile_buf)
                     outs(<dst> : !pto.tile_buf)
```

**语义：**

```text
For each element inside valid_shape:
    keep existing value
For each element in padded region:
    dst = PadVal(dst)

Note: src and dst often refer to the same SSA value
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | 源 tile buffer（通常与 dst 相同） |
| `dst` | `pto.tile_buf` | 目标 tile buffer（原地修改） |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查（A2A3/A5）**
  - `src` 和 `dst` 必须为 rank-2。
  - `src` 和 `dst` 元素大小必须一致，且为 1、2 或 4 字节。
  - `dst` 的 `pad` 值不能为 `Null`。
  - `src` 和 `dst` 必须具有相同的静态 shape（`rows`/`cols` 一致）。
  - 不允许 `dst` 的 shape 大于 `src`（与 `tfillpad_expand` 不同）。

**底层指令：**

- 降低为 `TFILLPAD_INPLACE(dst, src)`

**示例：**

```mlir
pto.tfillpad_inplace
    ins(%tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32,
                              v_row=32, v_col=32, blayout=row_major,
                              slayout=none_box, fractal=512, pad=1>)
    outs(%tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32,
                               v_row=32, v_col=32, blayout=row_major,
                               slayout=none_box, fractal=512, pad=1>)
```

---

### `pto.tconcatidx` — 索引控制列拼接

```mlir
pto.tconcatidx ins(<src0>, <src1>, <src0Idx>, <src1Idx>
                   : <src0_type>, <src1_type>, <idx0_type>, <idx1_type>)
               outs(<dst> : <dst_type>)
```

**语义：**

```text
For each row i:
    idx0_num = src0Idx[i, 0]
    idx1_num = src1Idx[i, 0]
    copy from src0: min(idx0_num, src0_valid_col, dst_valid_col) columns
    copy from src1: min(idx1_num, src1_valid_col, dst_valid_col - copied_from_src0) columns
```

逐行按索引控制从两个源 tile 拼接到目标 tile 的列方向操作。与 `pto.tconcat` 不同，每行的拼接列数由索引 tile 动态控制。

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src0` | `pto.tile_buf` | 第一个源 tile buffer |
| `src1` | `pto.tile_buf` | 第二个源 tile buffer |
| `src0Idx` | `pto.tile_buf` | src0 的逐行索引 tile（每行指定拷贝列数） |
| `src1Idx` | `pto.tile_buf` | src1 的逐行索引 tile（每行指定拷贝列数） |
| `dst` | `pto.tile_buf` | 目标 tile buffer |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查（A2A3）**
  - 所有操作数必须使用 `loc=vec`。
  - 数据 tile（src0、src1、dst）的元素类型必须一致，且为以下之一：`i8`、`i16`、`i32`、`f16`、`f32`、`bf16`。
  - 索引 tile（src0Idx、src1Idx）的元素类型必须相同，允许 signless `i8`、`i16` 或 `i32`。

- **实现检查（A5）**
  - 同 A2A3 约束。
  - 额外要求所有操作数必须使用 `blayout=row_major`。

**示例：**

```mlir
pto.tconcatidx
    ins(%src0, %src1, %idx0, %idx1 :
        !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=32,
                      v_row=16, v_col=32, blayout=row_major,
                      slayout=none_box, fractal=512, pad=0>,
        !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=32,
                      v_row=16, v_col=32, blayout=row_major,
                      slayout=none_box, fractal=512, pad=0>,
        !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=1,
                      v_row=16, v_col=1, blayout=row_major,
                      slayout=none_box, fractal=512, pad=0>,
        !pto.tile_buf<loc=vec, dtype=i32, rows=16, cols=1,
                      v_row=16, v_col=1, blayout=row_major,
                      slayout=none_box, fractal=512, pad=0>)
    outs(%dst : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=64,
                              v_row=16, v_col=64, blayout=row_major,
                              slayout=none_box, fractal=512, pad=0>)
```

---

### `pto.textract` 的 `fp` 形式

```mlir
pto.textract ins(<src>, <indexRow>, <indexCol> : <src_type>, index, index
                fp <fp> : <fp_type>)
             outs(<dst> : <dst_type>)
```

**语义：**

```text
For each element (i, j):
    dst[i, j] = dequant_extract(src[i + indexRow, j + indexCol], fp)
// 从累加器 tile 中提取子窗口，同时通过缩放因子进行反量化/类型转换
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | 源 tile buffer，必须为 `loc=acc` |
| `fp` | `pto.tile_buf` | 缩放因子 tile buffer，必须为 `loc=scaling` |
| `indexRow` | `index` | 提取起始行偏移 |
| `indexCol` | `index` | 提取起始列偏移 |
| `dst` | `pto.tile_buf` | 目标 tile buffer |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查（A2A3）**
  - 位置必须为 `src=acc`、`fp=scaling`、`dst=mat`。
  - 支持的类型对：`(src=f32, dst=i8)` 或 `(src=i32, dst=i8/f16/i16)`。
  - `dst` 的 fractal 必须为 512。

- **实现检查（A5）**
  - 位置必须为 `src=acc`、`fp=scaling`，`dst` 可为 `mat` 或 `vec`。
  - 支持的类型对：`(src=f32, dst=i8/fp8/f16/bf16/f32)` 或 `(src=i32, dst=i8/f16/bf16)`。
  - 无 fractal 512 限制。

**示例：**

```mlir
pto.textract ins(%src, %row, %col : !pto.tile_buf<loc=acc, dtype=f32, rows=32, cols=32,
                 v_row=32, v_col=32, blayout=col_major, slayout=row_major,
                 fractal=1024, pad=0>, index, index
                 fp %fp : !pto.tile_buf<loc=scaling, dtype=f32, rows=32, cols=32,
                 v_row=32, v_col=32, blayout=row_major, slayout=row_major,
                 fractal=512, pad=0>)
             outs(%dst : !pto.tile_buf<loc=mat, dtype=i8, rows=32, cols=32,
                 v_row=32, v_col=32, blayout=col_major, slayout=row_major,
                 fractal=512, pad=0>)
```

---

### `pto.tinsert` 的 `fp` 形式

```mlir
pto.tinsert ins(<src>, <indexRow>, <indexCol> : <src_type>, index, index
               fp <fp> : <fp_type>)
            outs(<dst> : <dst_type>)
```

**语义：**

```text
For each element (i, j):
    dst[i + indexRow, j + indexCol] = quant_insert(src[i, j], fp)
// 将 vector tile 通过缩放因子进行量化后插入到累加器 tile 的指定子窗口
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | 源 tile buffer，必须为 `loc=acc` |
| `fp` | `pto.tile_buf` | 缩放因子 tile buffer，必须为 `loc=scaling` |
| `indexRow` | `index` | 插入起始行偏移 |
| `indexCol` | `index` | 插入起始列偏移 |
| `dst` | `pto.tile_buf` | 目标 tile buffer |

**返回值：** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查（A2A3）**
  - 位置必须为 `src=acc`、`fp=scaling`、`dst=mat`。
  - 支持的类型对：`(src=f32, dst=i8)` 或 `(src=i32, dst=i8/f16/i16)`。
  - `dst` 的 fractal 必须为 512。

- **实现检查（A5）**
  - 位置必须为 `src=acc`、`fp=scaling`，`dst` 可为 `mat` 或 `vec`。
  - 支持的类型对：`(src=f32, dst=i8/fp8/f16/bf16/f32)` 或 `(src=i32, dst=i8/f16/bf16)`。
  - 无 fractal 512 限制。

**示例：**

```mlir
pto.tinsert ins(%src, %row, %col : !pto.tile_buf<loc=acc, dtype=f32, rows=32, cols=32,
                v_row=32, v_col=32, blayout=col_major, slayout=row_major,
                fractal=1024, pad=0>, index, index
                fp %fp : !pto.tile_buf<loc=scaling, dtype=f32, rows=32, cols=32,
                v_row=32, v_col=32, blayout=row_major, slayout=row_major,
                fractal=512, pad=0>)
            outs(%dst : !pto.tile_buf<loc=mat, dtype=i8, rows=32, cols=32,
                v_row=32, v_col=32, blayout=col_major, slayout=row_major,
                fractal=512, pad=0>)
```
