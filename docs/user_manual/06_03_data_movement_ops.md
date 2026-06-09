# **数据搬运操作**

本节描述了 PTO ISA 中用于数据搬运的指令族，包括从全局内存到本地缓冲区的数据转移、本地内存域之间的数据移动，以及标量元素的读写操作。这些操作采用"目标传递风格"（Destination-Passing Style, DPS）：操作本身不产生 SSA 返回值，而是直接将结果写入预先分配好的目标缓冲区或指针位置。

数据搬运操作通常涉及以下场景：

- **GM 到本地 Tile 缓冲区转移**：通过 `pto.tload` 和 `pto.tprefetch` 将全局内存分区视图加载到本地 tile buffer
- **异步预取**：使用 `pto.tprefetch_async` 启动 SDMA 驱动的异步预取，并返回同步事件
- **本地缓冲区存储回全局内存**：通过 `pto.tstore` 将 tile buffer 写回全局内存分区，支持原子操作和量化转换
- **带缩放因子的累加器存储**：通过 `pto.tstore_fp` 使用缩放 tile 对累加器数据进行向量量化后存储到全局内存
- **聚集/散射操作**：通过 `pto.mgather` 和 `pto.mscatter` 基于索引 tile 在全局内存与本地 tile 之间进行非连续数据搬运（仅 A5）
- **本地内存域间数据移动**：使用 `pto.tmov` 在 `mat`、`vec`、`acc`、`bias` 等本地存储域之间转移数据
- **Tile 转置**：通过 `pto.ttrans` 对 tile buffer 进行矩阵转置
- **标量读写**：通过 `pto.load_scalar` 和 `pto.store_scalar` 对全局内存中的单个标量元素进行读写操作

---

## 目录

- [`pto.tload` — 物理 DMA 加载](#ptotload--物理-dma-加载)
- [`pto.tprefetch` — 预取加载](#ptotprefetch--预取加载)
- [`pto.tprefetch_async` — 异步预取](#ptotprefetch_async--异步预取)
- [`pto.tstore` — Tile 缓冲区存储](#ptotstore--tile-缓冲区存储)
- [`pto.tstore_fp` — 带缩放因子的累加器存储](#ptotstore_fp--带缩放因子的累加器存储)
- [`pto.mgather` — 聚集加载](#ptomgather--聚集加载)
- [`pto.mscatter` — 散射存储](#ptomscatter--散射存储)
- [`pto.load_scalar` — 标量加载](#ptoload_scalar--标量加载)
- [`pto.store_scalar` — 标量存储](#ptostore_scalar--标量存储)
- [`pto.tmov` — 本地内存域间数据移动](#ptotmov--本地内存域间数据移动)
- [`pto.ttrans` — Tile 转置](#ptottrans--tile-转置)
- [`pto.tmov.fp` — 带缩放因子的累加器移动](#ptotmovfp--带缩放因子的累加器移动)

---

## 操作详解

### `pto.tload` — 物理 DMA 加载

```
pto.tload ins(<src> : <src_type>)
          outs(<dst> : <dst_type>)
```

**语义：**

```
For each element (i, j) in tile valid region:
    dst[i, j] = src[i, j]
```

**参数:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `!pto.partition_tensor_view<...>` | 源全局内存分区视图 |
| `dst` | `!pto.tile_buf` | 目标本地 tile buffer |

**返回值:** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查 (A2A3)**
  - Tile 元素类型必须为以下之一：`i8`、`i16`、`i32`、`i64`、`f16`、`bf16`、`f32`。
  - `dst` 必须使用 `loc=vec` 或 `loc=mat`。
  - `dst` 和 `src` 的元素类型必须具有相同的位宽。
  - 运行时约束：所有源分区的 extent 必须为正；`dst` 的有效区域必须非负。

- **实现检查 (A5)**
  - `src` 和 `dst` 元素类型必须为以下之一：`i8`、`i16`、`i32`、`i64`、`f16`、`bf16`、`f32`、`f8E4M3*`、`f8E5M2*`、`!pto.hif8`、`!pto.f4E1M2x2`、`!pto.f4E2M1x2`。
  - `dst` 元素大小必须为 1、2、4 或 8 字节，并与 `src` 匹配。
  - 对于 `i64` 类型，`dst` 的 padding 必须为 null 或零。

**硬件管道:** PIPE_MTE2（GM 至 UB 的 DMA 转移）

**示例:**

```mlir
pto.tload ins(%pv : !pto.partition_tensor_view<16x16xf16>)
          outs(%tb : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
                 v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                 fractal=512, pad=0>)
```

---

### `pto.tprefetch` — 预取加载

```
pto.tprefetch ins(<src> : <src_type>)
              outs(<dst> : <dst_type>)
```

**语义：**

```
TPREFETCH(dst, src)
For each element (i, j) in tile valid region:
    dst[i, j] = src[i, j]
```

**参数:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `!pto.partition_tensor_view<...>` | 源全局内存视图 |
| `dst` | `!pto.tile_buf` | 目标本地缓冲区 |

**返回值:** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查 (A2A3/A5)**
  - `dst` 必须使用 `loc=vec` 或 `loc=mat`。
  - 静态源 extent 必须为正；`dst` 的有效 extent 必须非负。
  - `src` 和 `dst` 的元素类型必须具有相同的元素大小。
  - 低精度类型（`f8*`、`!pto.hif8`、`!pto.f4*`）仅在 A5 上受支持。

**硬件管道:** PIPE_MTE2（GM 到本地 tile 的预取）

**示例:**

```mlir
pto.tprefetch ins(%pv : !pto.partition_tensor_view<16x16xf16>)
              outs(%tb : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
                     v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                     fractal=512, pad=0>)
```

---

### `pto.tprefetch_async` — 异步预取

```
%event = pto.tprefetch_async ins(<src>, <ctx> : <src_type>, !pto.prefetch_async_context)
                             -> !pto.async_event
```

**语义：**

```
%event = pto.tprefetch_async(%src, %ctx)
启动基于 SDMA 的异步预取，从全局内存到缓存中，并返回异步事件。
```

**参数:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `!pto.tensor_view` / `!pto.partition_tensor_view` | 源全局内存视图，必须是平坦的连续逻辑 1D 视图 |
| `ctx` | `!pto.prefetch_async_context` | 预取异步上下文，必须有效 |

**返回值:** `!pto.async_event` — 用于同步异步操作的事件句柄。

**约束：**

- **实现检查 (A2A3/A5)**
  - `src` 必须是扁平连续的逻辑 1D GM 视图。
  - `ctx` 必须是有效的 `prefetch_async_context`。

**硬件管道:** PIPE_SDMA（异步 DMA 预取）

**示例:**

```mlir
%ctx = pto.make_prefetch_async_context(%workspace : !pto.ptr<i8>) -> !pto.prefetch_async_context
%event = pto.tprefetch_async ins(%src, %ctx : !pto.partition_tensor_view<128xf32>, !pto.prefetch_async_context)
                             -> !pto.async_event
```

---

### `pto.tstore` — Tile 缓冲区存储

```
pto.tstore ins(<src> : <src_type>)
           outs(<dst> : <dst_type>)
           {attributes}

// 支持可选的 preQuantScalar 参数：
pto.tstore ins(<src>, <preQuantScalar> : <src_type>, i64)
           outs(<dst> : <dst_type>)
           {attributes}
```

**语义：**

```
For each element (i, j) in tile valid region:
    dst[i, j] = src[i, j]

（支持可选的原子操作、ReLU 前置处理和量化转换）
```

**参数:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `!pto.tile_buf` | 源 tile buffer，位置为 `vec`、`mat` 或 `acc` |
| `dst` | `!pto.partition_tensor_view<...>` | 目标全局内存分区视图 |
| `preQuantScalar` | `i64` （可选） | 量化前的标量值（仅在特定类型组合下使用） |

**返回值:** 无。以 DPS 的形式写入 `dst`。

**属性:**

- `stPhase` — 存储阶段标记。默认值为 `unspecified`。
  - `#pto<st_phase unspecified>` — 未指定的存储阶段
  - `#pto<st_phase partial>` — 部分存储（累加中间值）
  - `#pto<st_phase final>` — 最终存储（完成累加）

- `atomicType` — 原子操作类型。默认值为 `atomic_none`。
  - `#pto<atomic_type atomic_none>` — 无原子操作
  - `#pto<atomic_type atomic_add>` — 原子加法（dst += src）

- `reluPreMode` — ReLU 前置处理模式。默认值为 `no_relu`。
  - `#pto<relu_pre_mode no_relu>` — 不进行 ReLU 处理
  - `#pto<relu_pre_mode normal_relu>` — 应用标准 ReLU（max(0, x)）

**约束：**

- **实现检查 (A2A3)**
  - `src.loc` 必须为 `vec`、`mat` 或 `acc`。
  - 对于 `loc=vec` 或 `loc=mat`：不允许 `preQuantScalar`；`src` 元素类型必须为 `i8`、`i16`、`i32`、`i64`、`f16`、`bf16`、`f32`；位宽必须匹配。
  - 对于 `loc=acc`：`src` 必须为 `i32` 或 `f32`。
  - 不带 `preQuantScalar` 时：`dst` 为 `i32`、`f32`、`f16` 或 `bf16`。
  - 带 `preQuantScalar` 时：`src=i32` → `dst=i8` 或 `f16`；`src=f32` → `dst=i8`。
  - 静态列数 `1 <= cols <= 4095`。

- **实现检查 (A5)**
  - `src.loc` 必须为 `vec` 或 `acc`。
  - 对于 `loc=vec`：不允许 `preQuantScalar`；`src` 元素类型必须为 `i8`、`i16`、`i32`、`i64`、`f16`、`bf16`、`f32`、`f8*`、`!pto.hif8`、`!pto.f4*`。
  - 对于 `loc=acc`：`src` 必须为 `i32` 或 `f32`。
  - 不带 `preQuantScalar` 时：`dst` 为 `i32`、`f32`、`f16` 或 `bf16`。
  - 带 `preQuantScalar` 时：`src=i32` → `dst=i8`、`f16` 或 `bf16`；`src=f32` → `dst=i8`、`f16`、`bf16`、`f32` 或 `!pto.hif8` 或 `f8E4M3*`。

**硬件管道:**

- `loc=acc` 使用 PIPE_FIX（浮点修复）
- `loc=vec` / `loc=mat` 使用 PIPE_MTE3（MTE3 存储）

**示例:**

```mlir
// 基本存储
pto.tstore ins(%tb : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
                   v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                   fractal=512, pad=0>)
           outs(%pv : !pto.partition_tensor_view<16x16xf16>)

// 带最终阶段标记的存储
pto.tstore ins(%tb : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
                   v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                   fractal=512, pad=0>)
           outs(%pv : !pto.partition_tensor_view<16x16xf16>)
           {stPhase = #pto<st_phase final>}

// 带原子加法的存储
pto.tstore ins(%tb : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
                   v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                   fractal=512, pad=0>)
           outs(%pv : !pto.partition_tensor_view<16x16xf16>)
           {atomicType = #pto<atomic_type atomic_add>}

// Accumulator 到 mat 的存储，带原子操作和 ReLU
pto.tstore ins(%acc : !pto.tile_buf<loc=acc, dtype=i32, rows=32, cols=32,
                   v_row=32, v_col=32, blayout=col_major, slayout=row_major,
                   fractal=1024, pad=0>)
           outs(%pv2 : !pto.partition_tensor_view<32x32xf16>)
           {atomicType = #pto<atomic_type atomic_add>, reluPreMode = #pto<relu_pre_mode normal_relu>}
```

---

### `pto.tstore_fp` — 带缩放因子的累加器存储

```
pto.tstore_fp ins(<src>, <fp> : <src_type>, <fp_type>)
              outs(<dst> : <dst_type>)
```

**语义：**

```
For each element (i, j):
    dst[i, j] = Convert(src[i, j]; fp)
```

将累加器 tile 中的数据使用缩放（`fp`）tile 进行向量量化转换后存储到全局内存。

**参数:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `!pto.tile_buf<...>` | 源累加器 tile |
| `fp` | `!pto.tile_buf<...>` | 缩放因子 tile，用于配置缩放/FPC 状态 |
| `dst` | `!pto.partition_tensor_view<...>` | 目标全局内存 |

**返回值:** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查 (A2A3)**
  - `src` 的地址空间必须为 `loc=acc`。
  - `src` 元素类型必须为 `i32` 或 `f32`。
  - 列数约束：`1 <= cols <= 4095`。
  - 运行时：`1 <= src valid column <= 4095`。
  - `fp` 用于配置缩放/FPC 状态；对其形状不强制额外的 PTO 层面静态约束。

- **实现检查 (A5)**
  - `src` 的地址空间必须为 `loc=acc`。
  - `fp` 用于配置缩放/FPC 状态；对其形状不强制额外的 PTO 层面静态约束。

- **硬件管道**
  - 在 DMA 管道上执行（`PIPE_MTE3`）。

**示例:**

```mlir
pto.tstore_fp ins(%acc : !pto.tile_buf<loc=acc, dtype=f32, rows=16, cols=32,
                      v_row=16, v_col=32, blayout=col_major, slayout=row_major,
                      fractal=1024, pad=0>,
                  %fp : !pto.tile_buf<loc=vec, dtype=ui64, rows=1, cols=16,
                      v_row=1, v_col=16, blayout=row_major, slayout=none_box,
                      fractal=512, pad=0>)
              outs(%dst : !pto.partition_tensor_view<16x32xf16>)
```

---

### `pto.mgather` — 聚集加载

```
pto.mgather ins(<mem>, <idx> : <mem_type>, <idx_type>)
            outs(<dst> : <dst_type>)

// 带 OOB 模式（仅 A5）
pto.mgather ins(<mem>, <idx> : <mem_type>, <idx_type>)
            outs(<dst> : <dst_type>)
            {gatherOob = <mode>}
```

**语义：**

```
Row mode (default):
    For each element (r, j):
        dst[r, j] = mem[idx[r], j]

Element mode:
    For each element (i, j):
        dst[i, j] = mem[idx[i, j]]
```

使用逐元素索引从全局内存表中聚集加载数据到 VEC tile。

**参数:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `mem` | `!pto.partition_tensor_view<...>` | 全局源数据表 |
| `idx` | `!pto.tile_buf<...>` | 索引 tile |
| `dst` | `!pto.tile_buf<...>` | 目标 VEC tile |

**返回值:** 无。以 DPS 的形式写入 `dst`。

**属性:**

- `gatherOob` — 越界处理模式（仅 A5）。默认值为 `undefined`。
  - `#pto<gather_oob undefined>` — 未定义行为
  - `#pto<gather_oob clamp>` — 钳位到有效范围
  - `#pto<gather_oob wrap>` — 环绕取模
  - `#pto<gather_oob zero>` — 越界元素置零

**约束：**

- **仅 A5 支持**
  - `pto.mgather` 仅在 A5 目标上受支持。

- **类型约束（数据和索引）**
  - `mem` 和 `dst` 的元素类型必须相同。支持的类型：`i8`/`i16`/`i32`/`f16`/`bf16`/`f32`。A5 额外支持 `float8_e4m3`/`float8_e5m2` 系列。
  - `idx` 的元素类型必须为无符号 `i32`。

- **Tile / 内存角色**
  - `dst` 必须为 `loc=vec`、`blayout=row_major`、`slayout=none_box`。
  - `idx` 必须为 `loc=vec`、`slayout=none_box`。行模式下 `row_major` 和 `col_major` 均可接受。
  - `mem` 必须为 GM 内存中的 GlobalTensor。
  - `mem` 在可推断布局时必须使用 `ND` 布局。

- **形状约束**
  - 元素模式：`idx valid_shape == dst valid_shape`。
  - 行模式：`idx valid_shape` 可为 `[1, dst.valid_row]` 或 `[dst.valid_row, 1]`。
  - `[1, R]` 行模式变体使用 `row_major`；`[R, 1]` 行模式变体使用 `col_major`。
  - 若 `mem` 为 rank-5 静态 `!pto.partition_tensor_view`，必须满足 `<1, 1, 1, Rows, RowWidth>` 形式。

- **越界模式**
  - 默认 `gatherOob = undefined` 降低为默认 `MGATHER(dst, mem, idx)` 重载。
  - 非默认 `gatherOob` 值仅在 **A5** 上支持，降低为 `MGATHER<GatherOOB::...>(dst, mem, idx)`。

**硬件管道:** PIPE_MTE2（DMA 聚集加载）

**示例:**

```mlir
// 基本聚集加载
pto.mgather ins(%mem : !pto.partition_tensor_view<1024x32xi32>,
                %idx : !pto.tile_buf<loc=vec, dtype=i32, rows=1, cols=32,
                    v_row=1, v_col=32, blayout=row_major, slayout=none_box,
                    fractal=512, pad=0>)
            outs(%dst : !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32,
                    v_row=32, v_col=32, blayout=row_major, slayout=none_box,
                    fractal=512, pad=0>)

// A5：带越界置零模式
pto.mgather ins(%mem : !pto.partition_tensor_view<1024x32xi32>,
                %idx : !pto.tile_buf<loc=vec, dtype=i32, rows=1, cols=32,
                    v_row=1, v_col=32, blayout=row_major, slayout=none_box,
                    fractal=512, pad=0>)
            outs(%dst : !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32,
                    v_row=32, v_col=32, blayout=row_major, slayout=none_box,
                    fractal=512, pad=0>)
            {gatherOob = #pto<gather_oob zero>}
```

---

### `pto.mscatter` — 散射存储

```
pto.mscatter ins(<src>, <idx> : <src_type>, <idx_type>)
             outs(<mem> : <mem_type>)

// 带原子和 OOB 模式（仅 A5）
pto.mscatter ins(<src>, <idx> : <src_type>, <idx_type>)
             outs(<mem> : <mem_type>)
             {scatterAtomicOp = <atomic>, scatterOob = <oob>}
```

**语义：**

```
Row mode (default):
    For each element (r, j):
        mem[idx[r], j] = src[r, j]

Element mode:
    For each element (i, j):
        mem[idx[i, j]] = src[i, j]
```

使用逐元素索引将 VEC tile 中的数据散射存储到全局内存表。

**参数:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `!pto.tile_buf<...>` | 源 VEC tile |
| `idx` | `!pto.tile_buf<...>` | 索引 tile |
| `mem` | `!pto.partition_tensor_view<...>` | 全局目标数据表 |

**返回值:** 无。以 DPS 的形式写入 `mem`。

**属性:**

- `scatterAtomicOp` — 原子操作模式（仅 A5）。默认值为 `none`。
  - `#pto<scatter_atomic_op none>` — 无原子操作
  - `#pto<scatter_atomic_op add>` — 原子加（要求 `i32`/`f16`/`f32`）
  - `#pto<scatter_atomic_op max>` — 原子取最大值（要求 `i32` 或 `f32`）
  - `#pto<scatter_atomic_op min>` — 原子取最小值（要求 `i32` 或 `f32`）

- `scatterOob` — 越界处理模式（仅 A5）。默认值为 `undefined`。
  - `#pto<scatter_oob undefined>` — 未定义行为
  - `#pto<scatter_oob skip>` — 跳过越界元素
  - `#pto<scatter_oob clamp>` — 钳位到有效范围
  - `#pto<scatter_oob wrap>` — 环绕取模

**约束：**

- **仅 A5 支持**
  - `pto.mscatter` 仅在 A5 目标上受支持。

- **类型约束（数据和索引）**
  - `src` 和 `mem` 的元素类型必须相同。支持的类型：`i8`/`i16`/`i32`/`f16`/`bf16`/`f32`。A5 额外支持 `float8_e4m3`/`float8_e5m2` 系列。
  - `idx` 的元素类型必须为无符号 `i32`。

- **Tile / 内存角色**
  - `src` 必须为 `loc=vec`、`blayout=row_major`、`slayout=none_box`。
  - `idx` 必须为 `loc=vec`、`slayout=none_box`。行模式下 `row_major` 和 `col_major` 均可接受。
  - `mem` 必须为 GM 内存中的 GlobalTensor。
  - `mem` 在可推断布局时必须使用 `ND` 布局。

- **形状约束**
  - 元素模式：`idx valid_shape == src valid_shape`。
  - 行模式：`idx valid_shape` 可为 `[1, src.valid_row]` 或 `[src.valid_row, 1]`。
  - `[1, R]` 行模式变体使用 `row_major`；`[R, 1]` 行模式变体使用 `col_major`。
  - 若 `mem` 为 rank-5 静态 `!pto.partition_tensor_view`，必须满足 `<1, 1, 1, Rows, RowWidth>` 形式。

- **原子模式**
  - 默认 `scatterAtomicOp = none` 降低为默认 `MSCATTER(mem, src, idx)` 重载。
  - 非默认 `scatterAtomicOp` 值仅在 **A5** 上支持。
  - `add` 要求元素类型为 `i32`/`f16`/`f32`。
  - `max`/`min` 要求元素类型为无符号 `i32` 或 `f32`。

- **越界模式**
  - 默认 `scatterOob = undefined` 在仅指定 atomic 时降低为 `MSCATTER<Atomic>(mem, src, idx)` 形式，两个属性均为默认时降低为默认重载。
  - 非默认 `scatterOob` 值仅在 **A5** 上支持，降低为 `MSCATTER<ScatterAtomicOp::..., ScatterOOB::...>(mem, src, idx)`。

**硬件管道:** PIPE_MTE3（DMA 散射存储）

**示例:**

```mlir
// 基本散射存储
pto.mscatter ins(%src : !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32,
                     v_row=32, v_col=32, blayout=row_major, slayout=none_box,
                     fractal=512, pad=0>,
                 %idx : !pto.tile_buf<loc=vec, dtype=i32, rows=1, cols=32,
                     v_row=1, v_col=32, blayout=row_major, slayout=none_box,
                     fractal=512, pad=0>)
             outs(%mem : !pto.partition_tensor_view<1024x32xi32>)

// A5：带原子加
pto.mscatter ins(%src : !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32,
                     v_row=32, v_col=32, blayout=row_major, slayout=none_box,
                     fractal=512, pad=0>,
                 %idx : !pto.tile_buf<loc=vec, dtype=i32, rows=1, cols=32,
                     v_row=1, v_col=32, blayout=row_major, slayout=none_box,
                     fractal=512, pad=0>)
             outs(%mem : !pto.partition_tensor_view<1024x32xi32>)
             {scatterAtomicOp = #pto<scatter_atomic_op add>}

// A5：带原子加和越界跳过
pto.mscatter ins(%src : !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32,
                     v_row=32, v_col=32, blayout=row_major, slayout=none_box,
                     fractal=512, pad=0>,
                 %idx : !pto.tile_buf<loc=vec, dtype=i32, rows=1, cols=32,
                     v_row=1, v_col=32, blayout=row_major, slayout=none_box,
                     fractal=512, pad=0>)
             outs(%mem : !pto.partition_tensor_view<1024x32xi32>)
             {scatterAtomicOp = #pto<scatter_atomic_op add>,
              scatterOob = #pto<scatter_oob skip>}
```

---

### `pto.load_scalar` — 标量加载

```
%val = pto.load_scalar %ptr[%offset] : !pto.ptr<type> -> type
```

**语义：**

```
value = ptr[offset]
```

**参数:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `ptr` | `!pto.ptr<...>` | 指向目标元素的指针 |
| `offset` | `index` | 相对于指针的字节偏移量 |

**返回值:** AnyType — 与指针元素类型相同的标量值。

**约束：**

- **实现检查 (A2A3/A5)**
  - 指针的元素类型必须与返回值类型匹配。

**硬件操作:** 从全局内存加载单个标量元素

**示例:**

```mlir
%val = pto.load_scalar %ptr[%offset] : !pto.ptr<f32> -> f32
```

---

### `pto.store_scalar` — 标量存储

```
pto.store_scalar %val, %ptr[%offset] : type, !pto.ptr<type>
```

**语义：**

```
ptr[offset] = value
```

**参数:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `value` | AnyType | 要存储的标量值 |
| `ptr` | `!pto.ptr<...>` | 指向目标位置的指针 |
| `offset` | `index` | 相对于指针的字节偏移量 |

**返回值:** 无。

**约束：**

- **实现检查 (A2A3/A5)**
  - `value` 的类型必须与指针的元素类型匹配。

**硬件操作:** 向全局内存存储单个标量元素

**示例:**

```mlir
pto.store_scalar %val, %ptr[%offset] : f32, !pto.ptr<f32>
```

---

### `pto.tmov` — 本地内存域间数据移动

```
// 基本形式
pto.tmov ins(<src> : <src_type>)
         outs(<dst> : <dst_type>)

// 带 fp（scaling 缓冲区）和属性的形式
pto.tmov ins(<src>, <fp> : <src_type>, !pto.tile_buf<loc=scaling, ...>)
         outs(<dst> : <dst_type>)
         {accToVecMode = ..., reluPreMode = ...}

// 带 preQuantScalar 的形式
pto.tmov ins(<src>, <preQuantScalar> : <src_type>, i64)
         outs(<dst> : <dst_type>)
         {attributes}
```

**语义：**

```
For each element (i, j):
    dst[i, j] = src[i, j]

（支持可选的精度转换、ReLU 前置处理和量化）
```

**参数:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `!pto.tile_buf` | 源 tile buffer，位置为 `mat`、`vec` 或 `acc` |
| `dst` | `!pto.tile_buf` | 目标 tile buffer，位置为 `left`、`right`、`bias`、`scaling` 等 |
| `fp` | `!pto.tile_buf<loc=scaling>` （可选） | 浮点精度缓冲区，仅在特定转换中使用 |
| `preQuantScalar` | `i64` （可选） | 量化前的标量值 |

**返回值:** 无。以 DPS 的形式写入 `dst`。

**属性:**

- `accToVecMode` — Accumulator 到 vec 的转换模式。仅在 `src.loc=acc` 且 `dst.loc=vec` 时使用。
  - `#pto<acc_to_vec_mode cast>` — 类型转换模式
  - `#pto<acc_to_vec_mode round>` — 舍入转换模式
  - `#pto<acc_to_vec_mode saturate>` — 饱和转换模式
  - 其他支持的转换模式（具体值参照 ODS 定义）

- `reluPreMode` — ReLU 前置处理模式。默认值为 `no_relu`。仅当 `src.loc=acc` 时支持。
  - `#pto<relu_pre_mode no_relu>` — 不进行 ReLU 处理
  - `#pto<relu_pre_mode normal_relu>` — 应用标准 ReLU（max(0, x)）

**约束：**

- **实现检查 (A2A3)**
  - 静态 shape 必须匹配。
  - 支持的位置对：`mat` → `left`/`right`/`bias`/`scaling`；`vec` → `vec`；`acc` → `mat`；`acc` → `vec`。
  - `accToVecMode` 仅用于 `acc` → `vec` 转换。
  - `reluPreMode`、`fp`、`preQuantScalar` 仅在 `src.loc=acc` 时支持。

- **实现检查 (A5)**
  - 类似于 A2A3，但支持的位置对为：`mat` → `left`/`right`/`bias`/`scaling`/`scale`；`vec` → `vec` 和 `vec` → `mat`；`acc` → `vec` 和 `acc` → `mat`。

**硬件管道:**

- `vec` → `vec` 使用 PIPE_V（向量管道）
- `mat` → `left`/`right`/`bias`/`scaling` 使用 PIPE_MTE1（MTE1 转移）
- `acc` → `mat`/`vec` 使用 PIPE_FIX（浮点修复）

**示例:**

```mlir
// 基本 acc 到 vec 转换
pto.tmov ins(%src : !pto.tile_buf<loc=acc, dtype=f16, rows=16, cols=16,
                  v_row=16, v_col=16, blayout=col_major, slayout=row_major,
                  fractal=1024, pad=0>)
         outs(%dst : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
                  v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                  fractal=512, pad=0>)

// 带 ReLU 的 acc 到 vec 转换
pto.tmov ins(%src : !pto.tile_buf<loc=acc, dtype=f32, rows=32, cols=32,
                  v_row=32, v_col=32, blayout=col_major, slayout=row_major,
                  fractal=1024, pad=0>)
         outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32,
                  v_row=32, v_col=32, blayout=row_major, slayout=none_box,
                  fractal=512, pad=0>)
         {reluPreMode = #pto<relu_pre_mode normal_relu>}

// mat 到 scaling 的移动
pto.tmov ins(%src : !pto.tile_buf<loc=mat, dtype=f16, rows=16, cols=16,
                  v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                  fractal=512, pad=0>)
         outs(%dst : !pto.tile_buf<loc=scaling, dtype=f16, rows=16, cols=16,
                  v_row=16, v_col=16, blayout=row_major, slayout=none_box,
                  fractal=512, pad=0>)
```

---

### `pto.ttrans` — Tile 转置

```
pto.ttrans ins(<src>, <tmp> : <src_type>, <tmp_type>)
           outs(<dst> : <dst_type>)
```

**语义：**

```
For each element (i, j):
    dst[i, j] = src[j, i]

使用临时缓冲区 tmp 完成转置操作。
```

**参数:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `!pto.tile_buf` | 源 tile buffer，必须使用 `blayout=row_major` |
| `tmp` | `!pto.tile_buf` | 临时工作缓冲区 |
| `dst` | `!pto.tile_buf` | 目标 tile buffer |

**返回值:** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查 (A2A3)**
  - `src` 和 `dst` 的元素类型必须匹配。
  - `src` 必须使用 `blayout=row_major`。
  - 元素大小必须为 1、2 或 4 字节。
  - 支持的类型：`i32`/`f32`（4 字节）、`i16`/`f16`/`bf16`（2 字节）、`i8`（1 字节）。
  - 转置在 `src` 的有效区域上进行。

- **实现检查 (A5)**
  - `src` 和 `dst` 的元素大小必须匹配。
  - 主维度上需要 32 字节对齐。
  - `src` 和 `dst` 必须具有相同的元素类型。
  - 转置在静态 tile shape 上操作。

**硬件管道:** PIPE_V（向量管道）

**示例:**

```mlir
pto.ttrans ins(%src, %tmp : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
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

### `pto.tmov.fp` — 带缩放因子的累加器移动

```
pto.tmov.fp ins(<src>, <fp> : <src_type>, <fp_type>)
            outs(<dst> : <dst_type>)
```

**语义：**

```
For each element (i, j):
    dst[i, j] = dequant_move(src[i, j], fp)
// 将累加器（loc=acc）中的数据通过缩放因子 tile（fp）进行反量化移动到矩阵缓冲区（loc=mat）
// 本质为 ACC → MAT 数据搬运，附带 vector 量化参数
```

**参数:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `pto.tile_buf` | 源 tile buffer（通常 `loc=acc`） |
| `fp` | `pto.tile_buf` | 缩放因子 tile buffer（通常 `loc=vec`，SCALING 区域） |
| `dst` | `pto.tile_buf` | 目标 tile buffer（通常 `loc=mat`） |

**返回值:** 无。以 DPS 的形式写入 `dst`。

**约束：**

- **实现检查 (A2A3)**
  - `src` 通常位于 `loc=acc`，`dst` 位于 `loc=mat`。
  - 通过 `verifyTMovFpCommon` 和 `verifyTMovFpA2A3` 进行类型兼容性验证。

- **实现检查 (A5)**
  - 同上，但通过 `verifyTMovFpA5` 验证，支持更多类型组合。

**示例:**

```mlir
pto.tmov.fp
    ins(%src, %fp :
        !pto.tile_buf<loc=acc, dtype=f32, rows=16, cols=256,
                      v_row=16, v_col=256, blayout=col_major,
                      slayout=row_major, fractal=1024, pad=0>,
        !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=1,
                      v_row=16, v_col=1, blayout=row_major,
                      slayout=none_box, fractal=512, pad=0>)
    outs(%dst : !pto.tile_buf<loc=mat, dtype=bf16, rows=16, cols=256,
                              v_row=16, v_col=256, blayout=col_major,
                              slayout=row_major, fractal=512, pad=0>)
```