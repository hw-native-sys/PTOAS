# **指针与视图操作**

本章介绍 PTO ISA 中的指针与视图操作。这些操作涵盖指针算术、类型转换、张量视图构造、平铺子视图创建、Tile 类型重解释、有效维度读写，以及缓冲区同步令牌管理。大多数是纯元数据操作，不涉及数据移动，也不占用硬件管道。这些操作在动态形状处理、内存重映射和多级存储访问中至关重要。

---

## 目录

- [`pto.addptr` — 指针加法](#ptoaddptr--指针加法)
- [`pto.ptrtoint` — 指针转整数](#ptoptrtoint--指针转整数)
- [`pto.inttoptr` — 整数转指针](#ptointtoptr--整数转指针)
- [`pto.make_tensor_view` — 构造张量视图](#ptomake_tensor_view--构造张量视图)
- [`pto.get_tensor_view_dim` — 获取张量视图维度](#ptoget_tensor_view_dim--获取张量视图维度)
- [`pto.partition_view` — 分割视图](#ptopartition_view--分割视图)
- [`pto.subview` — 创建平铺子视图](#ptosubview--创建平铺子视图)
- [`pto.bitcast` — Tile 元素类型重解释](#ptobitcast--tile-元素类型重解释)
- [`pto.get_validshape` — 读取运行时有效维度](#ptoget_validshape--读取运行时有效维度)
- [`pto.set_validshape` — 设置运行时有效维度](#ptoset_validshape--设置运行时有效维度)
- [`pto.get_buf` — 获取缓冲区同步令牌](#ptoget_buf--获取缓冲区同步令牌)
- [`pto.rls_buf` — 释放缓冲区同步令牌](#ptorls_buf--释放缓冲区同步令牌)
- [`pto.tget_scale_addr` — 绑定缩放 Tile 视图](#ptotget_scale_addr--绑定缩放-tile-视图)

---

## 操作详解

### `pto.addptr` — 指针加法

```
pto.addptr <ptr>, <offset> : !pto.ptr<elementType> -> !pto.ptr<elementType>
```

**语义：**

```
result = ptr + offset  // offset is in elements, not bytes
```

**参数:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `ptr` | `!pto.ptr<elementType>` | 基指针 |
| `offset` | `index` | 元素偏移量（非字节偏移） |

**返回值:** `!pto.ptr<elementType>` — 与输入指针类型相同的指针。

**约束：**

- **实现检查 (A2A3/A5)**
  - 结果类型必须与输入指针类型匹配。
  - 纯操作（无副作用）。

**示例:**

```mlir
%c2 = arith.constant 2 : index
%ptr_off = pto.addptr %base, %c2 : !pto.ptr<f32> -> !pto.ptr<f32>
```

---

### `pto.ptrtoint` — 指针转整数

```
pto.ptrtoint <ptr> : !pto.ptr<elementType> -> i64
```

**语义：**

```
result = reinterpret_cast<i64>(ptr)
// 若源指针由 pto.addptr 生成：
// pto.ptrtoint(pto.addptr %p, %idx) == pto.ptrtoint(%p) + idx * sizeof(elementType)
```

**参数:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `ptr` | `!pto.ptr<elementType>` | 源全局指针 |

**返回值:** `i64` — 字节地址。

**约束：**

- **实现检查 (A2A3/A5)**
  - 纯操作（无副作用）。

**示例:**

```mlir
%addr = pto.ptrtoint %ptr : !pto.ptr<f32> -> i64
```

---

### `pto.inttoptr` — 整数转指针

```
pto.inttoptr <addr> : i64 -> !pto.ptr<resultElementType>
```

**语义：**

```
result = reinterpret_cast<result-element-type *>(addr)
```

**参数:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `addr` | `i64` | 源字节地址 |

**返回值:** `!pto.ptr<resultElementType>` — 指定元素类型的指针。

**约束：**

- **实现检查 (A2A3/A5)**
  - 结果被限制为仅标量内存访问：其所有直接使用必须是 `pto.load_scalar` 或 `pto.store_scalar` 的指针操作数。不能馈入 `pto.addptr`、`pto.make_tensor_view`、返回值或其他通用指针用户。
  - 结果元素类型必须为 EmitC 标量指针可表示的类型：`f16`、`bf16`、`f32`、`f64`、8/16/32/64 位整数类型、PTO 低精度类型。非标量类型（如 `index`）被拒绝。

**示例:**

```mlir
%addr = arith.constant 1024 : i64
%ptr = pto.inttoptr %addr : i64 -> !pto.ptr<f32>
pto.store_scalar %ptr, %value : !pto.ptr<f32>, f32
```

---

### `pto.make_tensor_view` — 构造张量视图

```
pto.make_tensor_view <ptr>, shape = [<d0>, <d1>, ...], strides = [<s0>, <s1>, ...]
                     : !pto.ptr<elementType> -> !pto.tensor_view<...>
```

**语义：**

```
result = tensor_view(ptr, shape, strides, layout)
// 从指针构造全局张量视图，不进行分配或数据移动
```

**参数:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `ptr` | `!pto.ptr<elementType>` | 源指针 |
| `shape` | 可变 `index` | 动态形状维度 |
| `strides` | 可变 `index` | 动态步长 |

**返回值:** `!pto.tensor_view<...>` — 构造的张量视图。

**属性:**

- `layout` — 可选布局属性（ND、DN 或 NZ）。

**约束：**

- **实现检查 (A2A3/A5)**
  - `ptr` 必须为 `!pto.ptr<...>`，其元素类型与结果匹配。
  - `shape` 和 `strides` 操作数计数必须与张量视图秩匹配。
  - `pto.inttoptr` 的结果不能馈入 `pto.make_tensor_view`。

**示例:**

```mlir
%m = arith.constant 128 : index
%n = arith.constant 256 : index
%s0 = arith.constant 256 : index
%s1 = arith.constant 1 : index
%tv = pto.make_tensor_view %ptr, shape = [%m, %n], strides = [%s0, %s1]
    : !pto.ptr<f32> -> !pto.tensor_view<?x?xf32>
```

---

### `pto.get_tensor_view_dim` — 获取张量视图维度

```
pto.get_tensor_view_dim <tensor_view>, <dim_index>
                        : !pto.tensor_view<...> -> index
```

**语义：**

```
result = get_dim(tensor_view, dim_index)
```

**参数:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `tensor_view` | `!pto.tensor_view<...>` | 源张量视图 |
| `dim_index` | `index` | 0-based 维度索引 |

**返回值:** `index` — 指定维度的大小。

**约束：**

- **实现检查 (A2A3/A5)**
  - 纯操作（无副作用）。

**示例:**

```mlir
%h = pto.get_tensor_view_dim %tv, %c0 : !pto.tensor_view<?x?xf32> -> index
%w = pto.get_tensor_view_dim %tv, %c1 : !pto.tensor_view<?x?xf32> -> index
```

---

### `pto.partition_view` — 分割视图

```
pto.partition_view <source>, offsets = [<o0>, <o1>, ...], sizes = [<s0>, <s1>, ...]
                   : !pto.tensor_view<...> -> !pto.partition_tensor_view<...>
```

**语义：**

```
result = source[offsets : offsets + sizes]
// 在张量视图上创建逻辑窗口
```

**参数:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `source` | `!pto.tensor_view<...>` | 源张量视图 |
| `offsets` | 可变 `index` | 窗口起始偏移 |
| `sizes` | 可变 `index` | 窗口大小 |

**返回值:** `!pto.partition_tensor_view<...>` — 分割的张量视图。

**约束：**

- **实现检查 (A2A3/A5)**
  - `offsets` 和 `sizes` 的计数必须与`source` 匹配。
  - 纯操作（无副作用）。

**示例:**

```mlir
%pv = pto.partition_view %tv, offsets = [%off0, %off1], sizes = [%s0, %s1]
    : !pto.tensor_view<1024x512xf16> -> !pto.partition_tensor_view<16x16xf16>
```

---

### `pto.subview` — 创建平铺子视图

```
pto.subview <source>[<i>, <j>] sizes [<rows>, <cols>] (valid [<vr>, <vc>])?
            : <source_type> -> <result_type>
```

**语义：**

```
result = source[offsets]
result.shape = sizes
result.valid = clip(explicit_valid_or_sizes, sizes)
```

**参数:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `source` | `!pto.tile_buf<...>` | 源平铺缓冲 |
| `offsets` | 可变 `index` | 运行时动态偏移 [i, j] |
| `valid_row` | `index`（可选） | 有效行数 |
| `valid_col` | `index`（可选） | 有效列数 |

**返回值:** `!pto.tile_buf<...>` — 子视图平铺缓冲。

**属性:**

- `sizes` — 静态形状数组（长度 2）。

**约束：**

- **实现检查 (A2A3/A5)**
  - 对于 boxed 布局（`slayout != none_box`）：`sizes` 必须是内部 boxed 形状的倍数；`offsets` 当为常数时必须是内部 boxed 形状的倍数。
  - 对于 boxed 行主序：子视图保持完整源列范围，列偏移必须为 0。
  - 对于 boxed 列主序：子视图保持完整源行范围，行偏移必须为 0。
  - `valid_row`/`valid_col` 必须同时存在或同时缺失。

**示例:**

```mlir
%i = arith.constant 0 : index
%j = arith.constant 0 : index
%sub = pto.subview %src[%i, %j] sizes [32, 32]
    : !pto.tile_buf<loc=vec, dtype=f16, rows=64, cols=64,
                    v_row=64, v_col=64, blayout=row_major,
                    slayout=none_box, fractal=512, pad=0>
    -> !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=32,
                     v_row=32, v_col=32, blayout=row_major,
                     slayout=none_box, fractal=512, pad=0>

// 带有效维度
%sub2 = pto.subview %src[%i, %j] sizes [32, 32] valid [%vr, %vc]
    : !pto.tile_buf<loc=vec, dtype=f16, rows=64, cols=64,
                    v_row=64, v_col=64, blayout=row_major,
                    slayout=none_box, fractal=512, pad=0>
    -> !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=32,
                     v_row=?, v_col=?, blayout=row_major,
                     slayout=none_box, fractal=512, pad=0>
```

---

### `pto.bitcast` — Tile 元素类型重解释

```
pto.bitcast <src> : <src_type> -> <result_type>
```

**语义：**

```
result = reinterpret_dtype(src)
// 返回与 src 共享相同底层存储的 tile buffer 视图，但具有不同的元素类型
// 仅元数据/config 重写，不涉及数据移动
```

**参数:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `!pto.tile_buf<...>` | 源 tile buffer |

**返回值:** `!pto.tile_buf<...>` — 元素类型不同的 tile buffer 视图，与源共享存储。

**约束：**

- **实现检查 (A2A3/A5)**
  - 源和结果必须具有不同的元素类型（如需仅改变 shape/config，使用 `pto.treshape`）。
  - 源和结果必须具有相同的 shape、validShape、memory space 和 tile config。
  - shape 必须为静态已知。
  - 结果所需的总字节数不得超过源的总字节数。
  - 纯操作（`Pure` + `ViewLikeOpInterface`），不分配存储。

**示例:**

```mlir
%casted = pto.bitcast %src
    : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16,
                    v_row=16, v_col=16, blayout=row_major,
                    slayout=none_box, fractal=512, pad=0>
    -> !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16,
                     v_row=16, v_col=16, blayout=row_major,
                     slayout=none_box, fractal=512, pad=0>
```

---

### `pto.get_validshape` — 读取运行时有效维度

```
pto.get_validshape <source> : <source_type>
```

**语义：**

```
(valid_row, valid_col) = read_valid_metadata(source)
// 读取 tile_buf handle 上当前的运行时 valid_row/valid_col 元数据
```

**参数:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `source` | `!pto.tile_buf<...>` | 源 tile handle |

**返回值:** `(index, index)` — 当前的有效行数和有效列数。

**约束：**

- **实现检查 (A2A3/A5)**
  - 源必须为 rank-2 的 tile_buf。
  - 源的 validShape 必须为 rank-2。

**示例:**

```mlir
%vr, %vc = pto.get_validshape %tile
    : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=32,
                    v_row=?, v_col=?, blayout=row_major,
                    slayout=none_box, fractal=512, pad=0>
```

---

### `pto.set_validshape` — 设置运行时有效维度

```
pto.set_validshape <source>, <valid_row>, <valid_col> : <source_type>
```

**语义：**

```
source.valid_row = valid_row
source.valid_col = valid_col
// 原地更新 tile_buf handle 上的运行时有效维度元数据
```

**参数:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `source` | `!pto.tile_buf<...>` | 目标 tile handle |
| `valid_row` | `index` | 新的有效行数 |
| `valid_col` | `index` | 新的有效列数 |

**返回值:** 无。原地修改 `source` 的元数据。

**约束：**

- **实现检查 (A2A3/A5)**
  - 源必须为 rank-2 的 tile_buf。
  - 源 tile_buf 的 valid_row 和 valid_col 必须均为动态（`?`）。
  - 源必须是本地绑定的 tile handle（如 `alloc_tile` 或由其派生的视图）；函数参数/返回值不受支持。
  - `valid_row` 不得为负且不得超过 shape 维度的静态大小。
  - `valid_col` 不得为负且不得超过 shape 维度的静态大小。

**示例:**

```mlir
pto.set_validshape %tile, %new_vr, %new_vc
    : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=32,
                    v_row=?, v_col=?, blayout=row_major,
                    slayout=none_box, fractal=512, pad=0>
```

---

### `pto.get_buf` — 获取缓冲区同步令牌

```
pto.get_buf [<op_type>, <buf_id>]
```

**语义：**

```
acquire_buffer_token(op_type, buf_id, mode)
// 获取指定管线操作类型和缓冲区 ID 的同步令牌。
// 映射到相同 pipe 且受相同 buf_id 保护的操作将按程序顺序执行。
```

**参数:** 无操作数。

**返回值:** 无。

**属性:**

- `op_type` — 管线事件类型（`pipe_event_type` 或 `sync_op_type`），映射到具体硬件管线。
- `buf_id` — 缓冲区 ID（`i32`），范围 [0, 31]。
- `mode` — 模式（`i32`），默认值为 `0`。

**约束：**

- **实现检查 (A5)**
  - `op_type` 必须映射到具体管线（不能是 `PIPE_ALL` 或 `PIPE_UNASSIGNED`）。
  - `buf_id` 必须在 [0, 31] 范围内。
  - `mode` 必须为非负整数。

**示例:**

```mlir
pto.get_buf[#pto.pipe_event_type<TLOAD>, 0]
pto.get_buf[#pto.pipe_event_type<TVEC>, 1]
```

---

### `pto.rls_buf` — 释放缓冲区同步令牌

```
pto.rls_buf [<op_type>, <buf_id>]
```

**语义：**

```
release_buffer_token(op_type, buf_id, mode)
// 释放之前通过 pto.get_buf 获取的缓冲区同步令牌
```

**参数:** 无操作数。

**返回值:** 无。

**属性:**

- `op_type` — 管线事件类型（`pipe_event_type` 或 `sync_op_type`），映射到具体硬件管线。
- `buf_id` — 缓冲区 ID（`i32`），范围 [0, 31]。
- `mode` — 模式（`i32`），默认值为 `0`。

**约束：**

- **实现检查 (A5)**
  - `op_type` 必须映射到具体管线（不能是 `PIPE_ALL` 或 `PIPE_UNASSIGNED`）。
  - `buf_id` 必须在 [0, 31] 范围内。
  - `mode` 必须为非负整数。

**示例:**

```mlir
pto.rls_buf[#pto.pipe_event_type<TLOAD>, 0]
pto.rls_buf[#pto.pipe_event_type<TVEC>, 1]
```

---

### `pto.tget_scale_addr` — 绑定缩放 Tile 视图

```
pto.tget_scale_addr ins(<src> : <src_type>)
                    outs(<dst> : <dst_type>)
```

**语义：**

```
dst = scale_view_of(src)
// 将缩放 tile 视图绑定到源 tile 的缩放地址，用于 MX（可变精度）浮点支持
// 纯地址绑定操作，无数据移动
```

**参数:**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src` | `!pto.tile_buf<...>` | 源 tile 缓冲区 |
| `dst` | `!pto.tile_buf<loc=scaling, ...>` | 目标缩放 tile 视图 |

**返回值:** 无。以 DPS 的形式绑定 `dst` 到 `src` 的缩放地址。

**约束：**

- **实现检查 (A2A3)**
  - 不支持此操作。

- **实现检查 (A5)**
  - `src` 必须为有效的 tile_buf。`dst` 必须使用 `loc=scaling`。
  - `src` 和 `dst` 必须具有相同的秩、shape 和有效 shape。
  - `src` 的元素类型必须支持 MX 缩放（如 `f8E4M3FN`）。

**示例:**

```mlir
// A5：绑定 f8 源到其相关的缩放 tile
pto.tget_scale_addr
    ins(%src : !pto.tile_buf<loc=left, dtype=f8E4M3FN, rows=1, cols=128,
                             v_row=1, v_col=128, blayout=col_major,
                             slayout=row_major, fractal=512, pad=0>)
    outs(%scale : !pto.tile_buf<loc=scaling, dtype=f16, rows=1, cols=128,
                                v_row=1, v_col=128, blayout=row_major,
                                slayout=row_major, fractal=512, pad=0>)
```
