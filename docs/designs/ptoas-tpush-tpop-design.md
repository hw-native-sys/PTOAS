# TALLOC/TPUSH/TPOP 前端接口与 PTOAS 实现设计

## 1. 文档范围

本文定义PTOAS TALLOC/TPUSH/TPOP 前端IR接口，以及其在 PTOAS 内部的 lowering、地址传播、flag 分配和 EmitC 映射规则。

本文覆盖两层接口：

- 前端接口
  - `pto.aic_initialize_pipe`
  - `pto.aiv_initialize_pipe`
  - `pto.talloc_to_aiv`
  - `pto.talloc_to_aic`
  - `pto.tpush_to_aiv`
  - `pto.tpush_to_aic`
  - `pto.tpop_from_aic`
  - `pto.tpop_from_aiv`
  - `pto.tfree_from_aic`
  - `pto.tfree_from_aiv`
  - `pto.reserve_buffer`
  - `pto.import_reserved_buffer`
- PTOAS 内部统一接口
  - `pto.initialize_l2g2l_pipe`
  - `pto.initialize_l2l_pipe`
  - `pto.talloc`
  - `pto.tpush`
  - `pto.declare_tile`
  - `pto.declare_global`
  - `pto.tpop`
  - `pto.tfree`

本文只描述接口契约与编译流程，不展开具体 C++ 模板实现细节。

## 2. 设计目标

本设计的目标如下：

- 对前端提供\*\_initialize_pipe/talloc_to_\*/tpush_to_\*/tpop_from_\*/tfree_from_\*IR接口。
- 在 PTOAS 内部统一为 pipe/talloc/tpush/tpop/tfree 指令，便于复用已有 pass。
- 支持 A2/A3 与 A5 两个平台使用同一套前端接口。
- 定义consumer slot buffer的分配地址与producer之间的匹配关系，并传播。
- 支持两类 pipe entry：
  - `tile` entry：现有 `!pto.tile_buf` local tile 传输语义。
  - `global` entry：对应 pto-isa `GlobalTensor` 形式的 GM FIFO entry，只管理 FIFO 同步与 GM slot 地址赋值，实际数据搬运由显式 `TSTORE` / `TLOAD` 对应的 PTO IR 完成。

## 3. 前端 IR 接口定义

### 3.1 `pto.aic_initialize_pipe`

#### 语义

由 Cube kernel 在函数启动时调用，初始化该函数涉及的通信 pipe。

#### 语法

```mlir
pto.aic_initialize_pipe {id = 0, dir_mask = 3, slot_size = 1024, local_slot_num = 1}
  (gm_slot_buffer = %gm_buf : !pto.ptr<f32>,
   c2v_consumer_buf = %c2v_consumer_buf : i32,
   v2c_consumer_buf = %v2c_consumer_buf : i32)
```

若同一 `id` 绑定的 pipe entry 全部是 `global` entry，则该 pipe 是 global-only GM FIFO。此时初始化只需要描述单个 FIFO slot entry 的 `tensor_view`，不需要 consumer 侧 local FIFO buffer：

```mlir
%gm_slots = pto.make_tensor_view %gm_slot_buffer,
  shape = [%c16, %c16], strides = [%c16, %c1]
  : !pto.tensor_view<16x16xf32>
pto.aic_initialize_pipe {id = 0, dir_mask = 1, slot_size = 1024}
  (gm_slot_tensor = %gm_slots : !pto.tensor_view<16x16xf32>)
```

#### 参数

| 参数 | 类型 | 说明 |
|---|---|---|
| `ID` | 编译期整数常量 | 前端逻辑 pipe 标识，要求 `>= 0` |
| `DIR_MASK` | 编译期整数常量 | `1`、`2` 或 `3` |
| `SLOT_SIZE` | 编译期整数常量 | 单 slot 字节数，定义为切分前完整 pipe entry 字节数 |
| `LOCAL_SLOT_NUM` | 编译期整数常量或空值 | 可选，仅在使用 consumer 侧 local FIFO buffer 的 tile-entry 路径影响槽数；global-only GM FIFO 省略 |
| `GM_SLOT_BUFFER` | `!pto.ptr<T>` 或空值 | 使用 local consumer FIFO buffer 的 A2/A3 GM 路径使用的 GM slot buffer 指针，A5 路径为空 |
| `GM_SLOT_TENSOR` | `!pto.tensor_view<...>` 或空值 | global-only GM FIFO 使用的单 slot entry descriptor；shape/stride/layout 必须与 `talloc` / `tpop` / `tpush` / `tfree` 使用的 GlobalTensor entry 对齐 |
| `C2V_CONSUMER_BUF` | `i32` 或空值 | C2V 方向 consumer 的 local slot buffer 基址；global-only GM FIFO 省略 |
| `V2C_CONSUMER_BUF` | `i32` 或空值 | V2C 方向 consumer 的 local slot buffer 基址；global-only GM FIFO 省略 |

### 3.2 `pto.aiv_initialize_pipe`

#### 语义

由 Vector kernel 在函数启动时调用，初始化该函数涉及的通信 pipe。

#### 语法

```mlir
pto.aiv_initialize_pipe {id = 0, dir_mask = 3, slot_size = 1024, local_slot_num = 1}
  (gm_slot_buffer = %gm_buf : !pto.ptr<f32>,
   c2v_consumer_buf = %c2v_consumer_buf : i32,
   v2c_consumer_buf = %v2c_consumer_buf : i32)
```

global-only GM FIFO 形式同样只传 `gm_slot_tensor`：

```mlir
%gm_slots = pto.make_tensor_view %gm_slot_buffer,
  shape = [%c16, %c16], strides = [%c16, %c1]
  : !pto.tensor_view<16x16xf32>
pto.aiv_initialize_pipe {id = 0, dir_mask = 1, slot_size = 1024}
  (gm_slot_tensor = %gm_slots : !pto.tensor_view<16x16xf32>)
```

参数语义与 `pto.aic_initialize_pipe` 相同。

### 3.3 前端数据传输接口

前端数据传输接口统一面向 pipe entry。当前支持两类 entry：

- `tile` entry：`!pto.tile_buf<...>` 或 lowering 后等价的 local memref。该路径保持既有 `TPUSH(pipe, tile)` / `TPOP(pipe, tile)` 语义。
- `global` entry：`!pto.tensor_view<...>` 或 lowering 后等价的 GM descriptor。该路径映射到 pto-isa `GlobalTensor` 形式的 `TALLOC` / `TPUSH` / `TPOP` / `TFREE`，只计算并传递 FIFO GM slot 地址，不隐式执行 `TSTORE` 或 `TLOAD`。若要读写 entry 的子区域，先用 `pto.partition_view` 从 `tensor_view` 派生 `!pto.partition_tensor_view<...>`，再交给 `pto.tload` / `pto.tstore`。

`global` entry 当前仅用于 A2/A3 的 GM FIFO 路径，即 lower 到 `pto.initialize_l2g2l_pipe` 的 pipe；A5 `initialize_l2l_pipe` 路径没有 GM slot 地址可赋给 `GlobalTensor`。

当某条 frontend logical pipe 的数据传输 op 全部使用 `global` entry 时，该 pipe 不需要 consumer 侧 local FIFO buffer。对应的 `pto.aic_initialize_pipe` / `pto.aiv_initialize_pipe` 只携带 `gm_slot_tensor`，不携带 `c2v_consumer_buf` / `v2c_consumer_buf`，也不生成或引用 `pto.reserve_buffer` / `pto.import_reserved_buffer`。

`global` entry 的 GM FIFO 单 slot 信息由 initialize op 的 `gm_slot_tensor` 描述；该 `tensor_view` 本身就是单个 GlobalTensor slot descriptor，不包含 FIFO slot 数这一外层维度。`talloc_to_*` / `tpop_from_*` 返回的 `tensor_view` 描述当前分配或弹出的完整单个 slot，dtype、rank、静态 shape 和字节大小必须与 `gm_slot_tensor` 一致。若未显式提供 stride/layout，则按 row-major contiguous GlobalTensor 处理。

#### `pto.talloc_to_aiv`

```mlir
%entry = pto.talloc_to_aiv {id = 0, split = 0}
  -> !pto.tensor_view<128x512xf32>
```

- 仅出现在 Cube kernel 中
- 表示 C2V 方向 producer 为一个 `global` entry 分配当前 FIFO GM slot
- 单个 slot 的 dtype、静态 shape 与 stride/layout 来自同 `id` 的 initialize op 所携带的 `gm_slot_tensor`；`talloc_to_aiv` 的返回类型必须与该 slot descriptor 匹配
- lower 到内部 `pto.talloc`
- 不写数据、不通知 consumer；后续由用户显式 `pto.tstore` 写入 `%entry` 所描述的 GM slot，再调用 `pto.tpush_to_aiv(%entry)`

#### `pto.talloc_to_aic`

```mlir
%entry = pto.talloc_to_aic {id = 0, split = 0}
  -> !pto.tensor_view<128x512xf32>
```

- 仅出现在 Vector kernel 中
- 表示 V2C 方向 producer 为一个 `global` entry 分配当前 FIFO GM slot
- 单个 slot 的 dtype、静态 shape 与 stride/layout 来自同 `id` 的 initialize op 所携带的 `gm_slot_tensor`；`talloc_to_aic` 的返回类型必须与该 slot descriptor 匹配
- lower 到内部 `pto.talloc`
- 不写数据、不通知 consumer；后续由用户显式 `pto.tstore` 写入 `%entry` 所描述的 GM slot，再调用 `pto.tpush_to_aic(%entry)`

#### `pto.tpush_to_aiv`

```mlir
pto.tpush_to_aiv(%entry : !pto.tile_buf<...>) {id = 0, split = 0}
pto.tpush_to_aiv(%entry : !pto.tensor_view<...>) {id = 0, split = 0}
```

- 仅出现在 Cube kernel 中
- 表示 C2V 方向 producer push
- 对 `tile` entry，语义保持现有 tile push
- 对 `global` entry，表示 producer commit；它只通知 consumer 当前 FIFO GM slot 已就绪，不执行数据写入

#### `pto.tpush_to_aic`

```mlir
pto.tpush_to_aic(%entry : !pto.tile_buf<...>) {id = 0, split = 0}
pto.tpush_to_aic(%entry : !pto.tensor_view<...>) {id = 0, split = 0}
```

- 仅出现在 Vector kernel 中
- 表示 V2C 方向 producer push
- 对 `tile` entry，语义保持现有 tile push
- 对 `global` entry，表示 producer commit；它只通知 consumer 当前 FIFO GM slot 已就绪，不执行数据写入

#### `pto.tpop_from_aic`

```mlir
%tile = pto.tpop_from_aic {id = 0, split = 0} -> !pto.tile_buf<...>
%entry = pto.tpop_from_aic {id = 0, split = 0}
  -> !pto.tensor_view<128x512xf32>
```

- 仅出现在 Vector kernel 中
- 表示 C2V 方向 consumer pop
- 返回 `tile` entry 时保持现有 tile pop 语义
- 返回 `global` entry 时，单个 slot 的 dtype、静态 shape 与 stride/layout 来自同 `id` 的 initialize op 所携带的 `gm_slot_tensor`
- 返回 `global` entry 时只等待 producer ready 并把当前 FIFO GM slot 地址赋给返回的 GlobalTensor-like view；后续由用户显式 `pto.tload` 或基于该 view 派生子 view 后加载

#### `pto.tpop_from_aiv`

```mlir
%tile = pto.tpop_from_aiv {id = 0, split = 0} -> !pto.tile_buf<...>
%entry = pto.tpop_from_aiv {id = 0, split = 0}
  -> !pto.tensor_view<128x512xf32>
```

- 仅出现在 Cube kernel 中
- 表示 V2C 方向 consumer pop
- 返回 `tile` entry 时保持现有 tile pop 语义
- 返回 `global` entry 时，单个 slot 的 dtype、静态 shape 与 stride/layout 来自同 `id` 的 initialize op 所携带的 `gm_slot_tensor`
- 返回 `global` entry 时只等待 producer ready 并把当前 FIFO GM slot 地址赋给返回的 GlobalTensor-like view；后续由用户显式 `pto.tload` 或基于该 view 派生子 view 后加载

#### `pto.tfree_from_aic`

```mlir
pto.tfree_from_aic {id = 0, split = 0}
pto.tfree_from_aic(%entry : !pto.tensor_view<...>) {id = 0, split = 0}
```

- 仅出现在 Vector kernel 中
- 表示 C2V 方向 consumer free
- `tile` entry 路径不带 operand，保持现有语义
- `global` entry 路径携带与 `tpop_from_aic` 返回值匹配的 entry descriptor，lower 到 `TFREE(pipe, gmTensor)` 形式

#### `pto.tfree_from_aiv`

```mlir
pto.tfree_from_aiv {id = 0, split = 0}
pto.tfree_from_aiv(%entry : !pto.tensor_view<...>) {id = 0, split = 0}
```

- 仅出现在 Cube kernel 中
- 表示 V2C 方向 consumer free
- `tile` entry 路径不带 operand，保持现有语义
- `global` entry 路径携带与 `tpop_from_aiv` 返回值匹配的 entry descriptor，lower 到 `TFREE(pipe, gmTensor)` 形式

以上前端数据传输接口中的 `id` 和 `split` 均为编译期常量属性，不是运行时 SSA operand。

- 取值使用 `TileSplitAxis` 枚举语义：`0/1/2/3/4` 分别对应 `TILE_NO_SPLIT`、`TILE_UP_DOWN`、`TILE_LEFT_RIGHT`、`TILE_UP_DOWN_ODD`、`TILE_LEFT_RIGHT_ODD`
- lowering 到 PTOAS 内部 IR 时，`split` 继续以属性形式保留
- `global` entry 的 result type 和 matched initialize op 的 `gm_slot_tensor` metadata 是底层 `GlobalData` 模板实参的 IR 描述；其 element type、静态 shape 与 stride/layout 必须描述完整 FIFO slot。若 consumer 只加载 slot 的子区域，应先 pop 完整 slot descriptor，再由该 descriptor 派生更窄的 GM view。

#### GlobalTensor pipe entry 使用示例

下面是 C2V 方向的 global-only GM FIFO 示例。两个 kernel 都只把单 slot entry descriptor `gm_slot_tensor` 传给初始化 op；因为不使用 consumer 侧 local FIFO buffer，IR 中没有对应的 `pto.reserve_buffer` 或 `pto.import_reserved_buffer`。

```mlir
func.func @cube_kernel(%gm_slot_buffer : !pto.ptr<f32>,
                       %src : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=1024, pad=0>)
    attributes {pto.kernel_kind = #pto.kernel_kind<cube>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %gm_slots = pto.make_tensor_view %gm_slot_buffer,
    shape = [%c16, %c16], strides = [%c16, %c1]
    : !pto.tensor_view<16x16xf32>
  pto.aic_initialize_pipe {id = 0, dir_mask = 1, slot_size = 1024}
    (gm_slot_tensor = %gm_slots : !pto.tensor_view<16x16xf32>)

  %entry = pto.talloc_to_aiv {id = 0, split = 0}
    -> !pto.tensor_view<16x16xf32>
  %entry_partition = pto.partition_view %entry,
    offsets = [%c0, %c0], sizes = [%c16, %c16]
    : !pto.tensor_view<16x16xf32> -> !pto.partition_tensor_view<16x16xf32>
  pto.tstore ins(%src : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=1024, pad=0>)
             outs(%entry_partition : !pto.partition_tensor_view<16x16xf32>)
  pto.tpush_to_aiv(%entry : !pto.tensor_view<16x16xf32>) {id = 0, split = 0}
  func.return
}

func.func @vector_kernel(%gm_slot_buffer : !pto.ptr<f32>,
                         %dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=1024, pad=0>)
    attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  %gm_slots = pto.make_tensor_view %gm_slot_buffer,
    shape = [%c16, %c16], strides = [%c16, %c1]
    : !pto.tensor_view<16x16xf32>
  pto.aiv_initialize_pipe {id = 0, dir_mask = 1, slot_size = 1024}
    (gm_slot_tensor = %gm_slots : !pto.tensor_view<16x16xf32>)

  %entry = pto.tpop_from_aic {id = 0, split = 0}
    -> !pto.tensor_view<16x16xf32>
  %entry_partition = pto.partition_view %entry,
    offsets = [%c0, %c0], sizes = [%c16, %c16]
    : !pto.tensor_view<16x16xf32> -> !pto.partition_tensor_view<16x16xf32>
  pto.tload ins(%entry_partition : !pto.partition_tensor_view<16x16xf32>)
            outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=1024, pad=0>)
  pto.tfree_from_aic(%entry : !pto.tensor_view<16x16xf32>) {id = 0, split = 0}
  func.return
}
```

### 3.4 地址提示接口

#### `pto.reserve_buffer`

用于在当前函数内声明一块 consumer slot buffer 预留空间，仅服务于需要 consumer 侧 local FIFO buffer 的 tile-entry 路径。global-only GM FIFO 不需要该 op。其合法写法由
当前编译流程是否启用 local address planning 决定。

```mlir
%buf = pto.reserve_buffer {
    name = "c2v_slot_buffer",
    size = 2048,
    location = #pto.address_space<vec>,
    auto = true
} -> i32
```

或使用显式地址：

```mlir
%buf = pto.reserve_buffer {
    name = "c2v_slot_buffer",
    size = 2048,
    location = #pto.address_space<vec>,
    auto = false,
    base = 4096
} -> i32
```

#### 参数

| 参数 | 类型 | 说明 |
|---|---|---|
| `name` | 字符串属性 | 本函数内唯一的预留段名字 |
| `size` | 整数属性 | 预留字节数 |
| `location` | 地址空间属性 | 预留空间所在 local 地址空间 |
| `auto` | `bool` 属性 | 地址解析路径标志；`true` 表示地址由 PTOAS 地址规划路径分配，`false` 表示地址已在输入 IR 中显式给定 |
| `base` | 可选整数属性 | 显式起始地址；仅 manual 路径使用 |

#### 结果

- 结果类型为 `i32`
- 结果值表示该 buffer 当前可用的基址
- 当前可用基址可来自显式 `base`，也可来自 plan memory 回填后的解析地址
- 单函数允许存在多条 `reserve_buffer`，但 `name` 必须唯一
- 编译路径与 `auto` 的合法组合只有两种：
  - 启用 local address planning：`auto = true`，且不带 `base`
  - 跳过 local address planning：`auto = false`，且显式提供 `base`

#### `pto.import_reserved_buffer`

用于引用 peer function 中已经定义的 `reserve_buffer` 结果，仅服务于需要 consumer 侧 local FIFO buffer 的 tile-entry 路径。global-only GM FIFO 不需要该 op。

```mlir
%buf = pto.import_reserved_buffer {
    name = "c2v_slot_buffer",
    peer_func = @vector_kernel
} -> i32
```

#### 参数

| 参数 | 类型 | 说明 |
|---|---|---|
| `name` | 字符串属性 | peer 侧 `reserve_buffer` 的名字 |
| `peer_func` | symbol ref | peer 函数符号 |

#### 结果

- 结果类型为 `i32`
- 结果值表示从 peer `reserve_buffer` 导入的已解析基址
- 单函数允许存在多条 `import_reserved_buffer`，但 `(name, peer_func)` 必须唯一

### 3.5 前端层约束

> **约束来源说明**：以下约束是当前软件方案的设计选择，而非硬件限制。
> 当前前端方案通过 `id` 在 push/pop/free 与 initialize 之间建立绑定关系，
> 因此单函数可以包含多条前端 initialize 语句，包括同方向多条逻辑 pipe。
> 每条 initialize 的 `id` 必须唯一，且数据传输 op 只能引用同函数内一条匹配的 initialize。
> 若同一函数内存在多条同方向 pipe，它们需要通过不同的 consumer FIFO
> 标识区分，即绑定到不同的 `reserve_buffer` / `import_reserved_buffer`
>（或显式不同的 local address）。global-only GM FIFO 不使用 consumer local address，
> 直接由 `id + direction` 区分前端 logical pipe。
> 硬件 flag 资源总量仍有限制：单向 pipe 占 2 个 hardware flag id，`dir_mask = 3`
> 的双向 pipe 占 4 个 hardware flag id，因此单函数内所有 lowered pipe 的总占用
> 必须落在 16 个 hardware flag id 以内。

前端 IR 需满足以下约束：

- 前端 initialize 的 `id` 在函数内必须唯一
- `talloc/tpush/tpop/tfree` 的 `id` 必须在同函数内匹配且仅匹配一条 frontend initialize
- C2V 方向数据 op（`talloc_to_aiv`/`tpush_to_aiv`/`tpop_from_aic`/`tfree_from_aic`）要求匹配的 init `dir_mask` 为 `1` 或 `3`
- V2C 方向数据 op（`talloc_to_aic`/`tpush_to_aic`/`tpop_from_aiv`/`tfree_from_aiv`）要求匹配的 init `dir_mask` 为 `2` 或 `3`
- 单函数内 lowered pipe 的 hardware flag id 总占用必须不超过 16
- 单函数允许多条 `reserve_buffer`
- 单函数允许多条 `import_reserved_buffer`
- `DIR_MASK` 只允许 `1`、`2`、`3`
- `SLOT_SIZE > 0`
- 使用 consumer 侧 local FIFO buffer 时，`reserve_buffer.size` 表示该
  consumer FIFO 实际预留的本地字节数。A2/A3 GM FIFO 路径要求
  `reserve_buffer.size == SLOT_SIZE * EFFECTIVE_LOCAL_SLOT_NUM`，其中
  `EFFECTIVE_LOCAL_SLOT_NUM` 为显式 `local_slot_num`，缺省时为有效
  `slot_num`。A5 L2L 路径不支持 `local_slot_num`，要求
  `reserve_buffer.size == SLOT_SIZE * EFFECTIVE_SLOT_NUM`。这里的
  `EFFECTIVE_SLOT_NUM` 为显式 `slot_num`，缺省时 `DIR_MASK=1/2` 为 `8`、
  `DIR_MASK=3` 为 `4`
- 使用 consumer 侧 local FIFO buffer 时，C2V consumer 的 `reserve_buffer.location` 必须是 `VEC`
- 使用 consumer 侧 local FIFO buffer 时，V2C consumer 的 `reserve_buffer.location` 必须是 `MAT`
- `reserve_buffer.name` 在本函数内必须唯一
- `import_reserved_buffer` 的 `(name, peer_func)` 在本函数内必须唯一
- op 级约束：`reserve_buffer.auto = false` 时必须提供 `base`
- op 级约束：`reserve_buffer.auto = true` 时必须不提供 `base`
- 启用 local address planning 的编译流程：`reserve_buffer` 只允许 `auto = true`
- 跳过 local address planning 的编译流程：`reserve_buffer` 只允许 `auto = false` 且显式提供 `base`
- `import_reserved_buffer` 必须能在 `peer_func` 中找到同名 `reserve_buffer`
- global-only GM FIFO 的 initialize 只提供 `gm_slot_tensor`（可附带 `slot_num`），不提供 `gm_slot_buffer`、`local_slot_num`、`c2v_consumer_buf`、`v2c_consumer_buf`，且不要求成对的 `reserve_buffer` / `import_reserved_buffer`

## 4. 核心约定

### 4.1 逻辑 pipe

本文中的”逻辑 pipe”指一条通信通道。

- C2V：Cube producer -> Vector consumer
- V2C：Vector producer -> Cube consumer

`DIR_MASK=3` 表示前端一个同时包含 C2V 和 V2C 的初始化请求。在 PTOAS lowering
后，生成单条 `dir_mask = 3` 的 DIR_BOTH 内部 pipe，同时承载 C2V 和 V2C 双向
通信。该 pipe 携带两个地址操作数：`local_addr`（C2V consumer buf）和
`peer_local_addr`（V2C consumer buf）。下游 TALLOC/TPUSH/TPOP/TFREE 共享同一 pipe
handle。

### 4.2 `split` 的角色

`split` 使用 `TileSplitAxis` 枚举表达：

- `TILE_NO_SPLIT`
- `TILE_UP_DOWN`
- `TILE_LEFT_RIGHT`
- `TILE_UP_DOWN_ODD`
- `TILE_LEFT_RIGHT_ODD`

当前 odd 模式允许 A2/A3/A5 GM-backed tile pipe 的 C2V 方向，以及 A2/A3 GM-backed
tile pipe 的 V2C 方向；A2/A3 双向 pipe 中具有对应 consumer buffer 的 V2C 方向也支持。
initialize 必须提供 GM slot backing，并为 C2V 提供 `c2v_consumer_buf`、为 V2C 提供
`v2c_consumer_buf`；双向 pipe 必须同时提供两个 consumer buffer。A5 odd V2C、local-only
pipe 与 GlobalTensor entry 仍不允许 odd 模式。

在 PTOAS 设计中，`split` 的角色定义为：

- `split` 是 `talloc/tpush/tpop/tfree` 的逐指令执行模式
- `split` 在 IR 中表示为编译期常量属性，不是运行时 SSA operand
- `split` 不参与pipe 初始化
- `split` 不参与 plan memory、地址传播、flag 分配
- PTOAS 将 `split` 作为透明的编译期参数向 EmitC 和底层 pto-isa 透传

因此：

- 同一条逻辑 pipe 上可以出现不同 `split` 的 `talloc/tpush/tpop/tfree`
- PTOAS 不要求同一逻辑 pipe 内所有指令使用同一个 `split`
- `split` 相关的语义正确性由前端生成逻辑或前端 verifier 保证；PTOAS 仅校验 `split` 枚举合法并向下透传

### 4.3 `SLOT_SIZE` 的定义

`SLOT_SIZE` 的定义固定为：

- 切分前完整 pipe entry 的字节数

即使 `split` 为 `TILE_UP_DOWN`、`TILE_LEFT_RIGHT`、`TILE_UP_DOWN_ODD` 或 `TILE_LEFT_RIGHT_ODD`，`SLOT_SIZE` 仍然表示未切分前的逻辑 pipe entry 总字节数。

`split` 只影响底层 `TALLOC/TPUSH/TPOP/TFREE` 的执行方式，不影响 `SLOT_SIZE` 的含义。

对 `global` entry，`split` 还会影响底层 pto-isa 对 GM FIFO slot 地址的计算方式：

- `TILE_NO_SPLIT`：不增加 sub-core offset
- `TILE_UP_DOWN`：sub-core offset 为 `get_subblockid() * rows * cols * sizeof(dtype)`
- `TILE_LEFT_RIGHT`：sub-core offset 为 `get_subblockid() * cols * sizeof(dtype)`
- `TILE_UP_DOWN_ODD`：奇数有效行由 AIV0/AIV1 按 `ceil(rows / 2)` 与 `floor(rows / 2)` 切分
- `TILE_LEFT_RIGHT_ODD`：奇数有效列由 AIV0/AIV1 按 `ceil(cols / 2)` 与 `floor(cols / 2)` 切分

GM tile 路径中的偶数和奇数切分模式都使用运行时 valid shape 计算 offset 与
row stride。odd 模式不会自动推导两个子核的 valid shape：前端必须根据
`pto.get_subblock_idx` 为 AIV0 计算 `ceil` 半块、为 AIV1 计算 `floor` 半块。C2V
方向把这两个 valid shape 作为 `tpop_from_aic` 的 `valid_row` / `valid_col` operand；
V2C 方向则在 `tpush_to_aic` 前把它们设置到两个 AIV producer tile 上，Cube 侧
`tpop_from_aiv` 使用切分前的完整 valid shape。描述 GM slot 的 metadata 仍必须覆盖
切分前的完整 FIFO slot。

### 4.4 `SLOT_NUM` 规则

`SLOT_NUM` 由 `DIR_MASK` 固定决定：

- `DIR_MASK = 1` 或 `2`：`SLOT_NUM = 8`
- `DIR_MASK = 3`（DIR_BOTH）：单条 pipe，`SLOT_NUM = 4`（每方向 4 slot，总缓冲 2 × 4 × SLOT_SIZE）

`SLOT_NUM` 不由 `split` 决定。

## 5. PTOAS 内部 IR 接口定义

### 5.1 `!pto.pipe`

本文设计的内部 `!pto.pipe` 为不透明 handle。

`!pto.pipe` 的协议信息由其定义 op 上的属性承载，而不是由 type 参数承载。

底层 `pto-isa` 若对 `TALLOC/TPUSH/TPOP/TFREE` 的模板形态继续演进，不反向约束 `!pto.pipe` 的 type 设计；内部 `!pto.pipe` 仍保持 opaque handle。

### 5.2 `pto.initialize_l2g2l_pipe`

用于 A2/A3 路径。

单向示例：

```mlir
%pipe = pto.initialize_l2g2l_pipe {
    dir_mask = 1,
    slot_size = 512,
    slot_num = 8,
    local_slot_num = 8
}(%gm_addr : i32, %local_addr : i32) -> !pto.pipe
```

global-only GM FIFO 示例：

```mlir
%pipe = pto.initialize_l2g2l_pipe {
    dir_mask = 1,
    slot_size = 1024,
    slot_num = 8
}(%gm_slot_tensor : !pto.tensor_view<16x16xf32>) -> !pto.pipe
```

DIR_BOTH 示例：

```mlir
%pipe = pto.initialize_l2g2l_pipe {
    dir_mask = 3,
    slot_size = 512,
    slot_num = 4,
    local_slot_num = 4
}(%gm_addr : i32, %c2v_addr : i32, %v2c_addr : i32) -> !pto.pipe
```

#### 必需属性

- `dir_mask`
- `slot_size`
- `slot_num`

#### 可选属性

- `local_slot_num`
  - 可直接由 `initialize_l2g2l_pipe` 承载，也可由 legacy 前端
    `pto.aic_initialize_pipe` / `pto.aiv_initialize_pipe` 提供并在 A2/A3 lowering 时转发
  - 表示 GM 路径下 consumer 侧 local slot buffer 的槽数，仅在存在 local FIFO buffer 的 tile-entry 路径有意义
  - 仅在通过 GM 传递时对底层 `TPipe` 模板参数有意义，不改变 GM FIFO 的 `slot_num`
  - A2/A3 consumer 侧 `reserve_buffer.size` 应按
    `slot_size * effective_local_slot_num` 预留
  - 存在 local FIFO buffer 且缺省时，默认值等于该内部 pipe 的 `slot_num`
  - 因此前端未显式指定 `slot_num` 时：
    - `DIR_MASK=1/2` 直接 lowering 时，`local_slot_num = 8`
    - `DIR_MASK=3` 单条 DIR_BOTH pipe，`local_slot_num = 4`
  - global-only GM FIFO 不携带 `local_slot_num`
- `flag_base`
  - 由 PTOAS flag 分配阶段填写
  - frontend lowering 阶段可以缺省
  - EmitC 前必须已经解析为显式常量

#### 操作数

- `gm_addr`：使用 local consumer FIFO buffer 的 A2/A3 GM 路径使用
- `gm_slot_tensor`：global-only GM FIFO 使用，描述单个 slot entry
- `local_addr`（可选）：C2V consumer buf（或单向时唯一方向的 consumer buf），仅存在 consumer 侧 local FIFO buffer 时出现
- `peer_local_addr`（可选）：V2C consumer buf，仅 `dir_mask = 3` 且存在 V2C consumer 侧 local FIFO buffer 时出现

global-only GM FIFO 只携带 `gm_slot_tensor`；`local_addr` / `peer_local_addr` 均省略。

### 5.3 `pto.initialize_l2l_pipe`

用于 A5 路径。

单向示例：

```mlir
%pipe = pto.initialize_l2l_pipe {
    dir_mask = 1,
    slot_size = 512,
    slot_num = 8
}(%local_addr : i32) -> !pto.pipe
```

双向（DIR_BOTH）示例：

```mlir
%pipe = pto.initialize_l2l_pipe {
    dir_mask = 3,
    slot_size = 1024,
    slot_num = 4
}(%c2v_addr : i32, %v2c_addr : i32) -> !pto.pipe
```

#### 必需属性

- `dir_mask`：合法值 `1`（C2V）、`2`（V2C）、`3`（DIR_BOTH）
- `slot_size`
- `slot_num`

#### 可选属性

- `flag_base`
  - 由 PTOAS flag 分配阶段填写
  - frontend lowering 阶段可以缺省
  - EmitC 前必须已经解析为显式常量

#### 操作数

- `local_addr`：C2V consumer buf（或单向时唯一方向的 consumer buf）
- `peer_local_addr`（可选）：V2C consumer buf，仅 `dir_mask = 3` 时出现

#### 操作数

- `local_addr`

### 5.4 pipe entry type

内部 `talloc` / `tpush` / `tpop` / `tfree` 统一使用 pipe entry operand。entry 可以是：

- `tile` entry：`!pto.tile_buf<...>` 或 lowering 后等价 local memref。
- `global` entry：`!pto.tensor_view<...>` 或 lowering 后等价 GM descriptor；`!pto.partition_tensor_view<...>` 只作为从 entry 派生出的 load/store window。

`tile` entry 表示数据本身由底层 `TPUSH/TPOP` 搬运；`global` entry 表示一个 GM FIFO slot descriptor，底层 pipe op 只对该 descriptor 赋 GM 地址或进行同步提交/释放。

### 5.5 `pto.talloc`

```mlir
%entry = pto.declare_global -> !pto.tensor_view<128x512xf32>
pto.talloc(%entry, %pipe : !pto.tensor_view<128x512xf32>, !pto.pipe) {split = 0}
```

`pto.talloc` 只支持 `global` entry，用于 producer 侧开始一次 GM FIFO transaction。它等待并分配 producer 侧空闲 slot，计算 FIFO GM slot 地址，并把该地址绑定到 `%entry`。

`pto.talloc` 不写数据，也不通知 consumer。用户必须在 `pto.talloc` 和对应 `pto.tpush` 之间显式写入该 GM slot。

### 5.6 `pto.tpush`

```mlir
pto.tpush(%tile, %pipe) { split = 0 }
pto.tpush(%entry, %pipe : !pto.tensor_view<128x512xf32>, !pto.pipe) {split = 0}
```

`pto.tpush` 支持 `tile` entry 和 `global` entry：

- `tile` entry：保持现有语义，映射到底层 `TPUSH(pipe, tile)`。
- `global` entry：提交已由 `pto.talloc` 分配并由用户写入的 GM FIFO slot，映射到底层 `TPUSH(pipe, gmTensor)`；不执行 `TSTORE`。

### 5.7 `pto.declare_tile`

```mlir
%tile = pto.declare_tile -> !pto.tile_buf<...>
```

用于声明一个地址稍后由 `pto.tpop` 绑定的 tile entry。

### 5.8 `pto.declare_global`

```mlir
%entry = pto.declare_global -> !pto.tensor_view<128x512xf32>
```

用于声明一个地址稍后由 `pto.talloc` 或 `pto.tpop` 绑定的 GlobalTensor-like entry。该 op 不分配 GM 内存，只提供带静态 dtype、shape、stride/layout 信息的 descriptor。

### 5.9 `pto.tpop`

```mlir
pto.tpop(%tile, %pipe : !pto.tile_buf<...>, !pto.pipe) {split = 0}
pto.tpop(%entry, %pipe : !pto.tensor_view<128x512xf32>, !pto.pipe) {split = 0}
```

`pto.tpop` 支持 `tile` entry 和 `global` entry：

- `tile` entry：保持现有语义，等待 producer ready 并把 consumer local slot 地址绑定到 tile。
- `global` entry：等待 producer ready，计算 FIFO GM slot 或 subslot 地址，并把该地址绑定到 GlobalTensor-like descriptor；不执行 `TLOAD`。

### 5.10 `pto.tfree`

```mlir
pto.tfree(%pipe : !pto.pipe) {split = 0}
pto.tfree(%entry, %pipe : !pto.tensor_view<128x512xf32>, !pto.pipe) {split = 0}
```

`tile` entry 路径不携带 entry operand，保持现有 `TFREE(pipe)` 语义。`global` entry 路径携带与 `pto.tpop` 绑定的 entry descriptor，并映射到底层 `TFREE(pipe, gmTensor)`。

`split` 在内部 IR 中必须以编译期常量属性形式保留，不能在 lowering 时擦除或降为运行时 operand。

## 6. 前端到内部 IR 的 lowering 规则

### 6.1 初始化接口 lowering

#### A2/A3

- `pto.aic_initialize_pipe` 和 `pto.aiv_initialize_pipe` lower 为 `pto.initialize_l2g2l_pipe`
- 若前端 init 只提供 `gm_slot_tensor`（可附带 `slot_num`），则 lower 为只携带 `gm_slot_tensor` 的 global-only GM FIFO；不补 `local_slot_num`，不生成 local consumer address operand，也不依赖 `reserve_buffer` / `import_reserved_buffer`
- 若前端提供了 consumer 侧 local FIFO buffer，且提供了 `local_slot_num`，则直接转发到 lowered
  `pto.initialize_l2g2l_pipe`
- 若前端提供了 consumer 侧 local FIFO buffer 但未提供 `local_slot_num`，lowering 默认补上 `local_slot_num = slot_num`

#### A5

- `pto.aic_initialize_pipe` 和 `pto.aiv_initialize_pipe` lower 为 `pto.initialize_l2l_pipe`
- A5 不支持 `local_slot_num`；前端 init 若显式携带该属性，verifier 会报错
- A5 的 consumer 侧 `reserve_buffer.size` 不由 `local_slot_num` 决定；A5
  L2L pipe 本地 FIFO 地址按 `slot_num` 取模，按
  `slot_size * effective_slot_num` 预留本地 FIFO buffer

### 6.2 `DIR_MASK=1/2`

- 只生成一条内部 pipe
- `slot_num` 缺省为 `8`，也可由前端显式指定
- 对带 consumer 侧 local FIFO buffer 的 `initialize_l2g2l_pipe`，默认 `local_slot_num = slot_num`
- 若前端显式提供 `local_slot_num`，则使用显式值
- global-only GM FIFO 不携带 `local_slot_num`，地址/descriptor 操作数只有 `gm_slot_tensor`

### 6.3 `DIR_MASK=3`

前端一个 init op 生成**单条** DIR_BOTH 内部 pipe：

- `%pipe`：`dir_mask = 3`，`slot_num` 缺省为 `4`，也可由前端显式指定
- 若 lowering 为带 consumer 侧 local FIFO buffer 的 `initialize_l2g2l_pipe`，默认 `local_slot_num = slot_num`
- 若前端显式提供 `local_slot_num`，则使用显式值

地址选择规则：

- 若存在 consumer 侧 local FIFO buffer，`local_addr` = `C2V_CONSUMER_BUF`
- 若存在 consumer 侧 local FIFO buffer，`peer_local_addr` = `V2C_CONSUMER_BUF`
- global-only GM FIFO 只设置 `gm_slot_tensor` = `GM_SLOT_TENSOR`

`FrontendPipeHandles` 中 `c2vPipe` 和 `v2cPipe` 指向同一个 pipe Value。

### 6.4 前端数据传输 op 与内部 pipe 的绑定

绑定规则固定如下：

| 前端 op | 所在函数 | 方向 | 使用的内部 pipe |
|---|---|---|---|
| `talloc_to_aiv` | Cube | C2V | `c2vPipe` |
| `tpush_to_aiv` | Cube | C2V | `c2vPipe` |
| `tpop_from_aic` | Vector | C2V | `c2vPipe` |
| `tfree_from_aic` | Vector | C2V | `c2vPipe` |
| `talloc_to_aic` | Vector | V2C | `v2cPipe` |
| `tpush_to_aic` | Vector | V2C | `v2cPipe` |
| `tpop_from_aiv` | Cube | V2C | `v2cPipe` |
| `tfree_from_aiv` | Cube | V2C | `v2cPipe` |

当 `DIR_MASK=3` 时，`c2vPipe` 和 `v2cPipe` 指向同一个 DIR_BOTH pipe，下游 TALLOC/TPUSH/TPOP/TFREE 只关心 pipe handle 是否存在，不关心是否是同一个。

### 6.5 数据传输 op lowering

#### `talloc_to_aiv` / `talloc_to_aic`

`global` entry producer allocation lower 为：

```mlir
%decl = pto.declare_global -> !pto.tensor_view<...>
pto.talloc(%decl, %pipe : !pto.tensor_view<...>, !pto.pipe) {split = 0}
```

即：

- 前端 `pto.talloc_to_aiv` / `pto.talloc_to_aic` 不接收 GM tensor descriptor operand；它们通过同 `id` 的 initialize op 找到 `gm_slot_tensor`，并返回绑定了当前 FIFO GM slot 地址的 `global` entry
- PTOAS 内部 `pto.talloc` 是 destination-style 形式，显式接收一个 `pto.declare_global` 结果作为入参；`pto.declare_global` 的 dtype、shape、stride/layout 来自 matched initialize op 的 `gm_slot_tensor` 的单 slot descriptor
- `pto.talloc` 之后、对应 `pto.tpush` 之前，用户 IR 可以通过普通 `pto.tstore` 写入该 entry 指向的 GM FIFO slot

#### `tpush_to_aiv` / `tpush_to_aic`

lower 为：

```mlir
pto.tpush(%entry, %pipe : !pto.tensor_view<...>, !pto.pipe) {split = 0}
```

其中 `%entry` 可以是 `tile` entry 或 `global` entry。`global` entry 路径要求同一 producer transaction 中存在支配该 `tpush` 的对应 `pto.talloc`。

#### `tpop_from_aic` / `tpop_from_aiv`

lower 为：

```mlir
%decl = pto.declare_tile -> !pto.tile_buf<...>
pto.tpop(%decl, %pipe : !pto.tile_buf<...>, !pto.pipe) {split = 0}

%gdecl = pto.declare_global -> !pto.tensor_view<...>
pto.tpop(%gdecl, %pipe : !pto.tensor_view<...>, !pto.pipe) {split = 0}
```

即：

- 前端 `pto.tpop_from_aic` / `pto.tpop_from_aiv` 是返回 pipe entry 结果值的接口
- 对 `global` entry，前端 `pto.tpop_from_aic` / `pto.tpop_from_aiv` 不接收 GM tensor descriptor operand；它们通过同 `id` 的 initialize op 找到 `gm_slot_tensor`，并返回绑定了当前 FIFO GM slot 地址的 `global` entry
- PTOAS 内部 `pto.tpop` 才是 destination-style 形式，显式接收一个 `pto.declare_tile` 或 `pto.declare_global` 结果作为入参；`pto.declare_global` 的 dtype、shape、stride/layout 来自 matched initialize op 的 `gm_slot_tensor` 的单 slot descriptor
- `global` entry 路径下，`pto.tpop` 之后、对应 `pto.tfree` 之前，用户 IR 可以直接使用该 entry 或从该 entry 派生子 view，再通过普通 `pto.tload` 读取 GM FIFO slot

#### `tfree_from_aic` / `tfree_from_aiv`

lower 为：

```mlir
pto.tfree(%pipe : !pto.pipe) {split = 0}
pto.tfree(%entry, %pipe : !pto.tensor_view<...>, !pto.pipe) {split = 0}
```

其中无 entry operand 的形式用于 `tile` entry 路径；带 entry operand 的形式用于 `global` entry 路径，且 `%entry` 必须来自对应 consumer transaction 的 `pto.tpop` 结果。

## 7. `reserve_buffer` 与地址传播

### 7.1 设计原则

- `reserve_buffer` 只表示本函数 consumer slot buffer 的本地预留
- `import_reserved_buffer` 只表示对 peer 预留段地址的引用
- `reserve_buffer` 用属性描述“如何得到地址”，用结果值统一承载“当前可用地址”
- 当前编译流程是否启用 local address planning 与 `reserve_buffer.auto` 共同决定地址处理路径
- 启用 local address planning：`reserve_buffer` 必须使用 `auto = true`，由 `PlanMemory` 分配地址
- 跳过 local address planning：`reserve_buffer` 必须使用 `auto = false` 且显式提供 `base`，不再进入 `PlanMemory` 分配路径
- PTOAS 复用现有 `PlanMemory` pass 实现 `reserve_buffer` 地址确定，不额外增加独立的预分配 pass
- PTOAS 新增独立地址传播 pass，专门处理 `import_reserved_buffer` 常量替换与 peer pipe 的 `flag_base` 对齐
- 地址传播 pass 在 EmitC 之前运行；启用规划时位于 plan memory 之后，跳过规划时直接消费前端已给定地址

### 7.2 使用规则

#### C2V

- consumer 是 Vector
- Vector function 需要 `reserve_buffer(location = VEC)`
- Cube function 需要 `import_reserved_buffer(peer_func = @vector_kernel)`

#### V2C

- consumer 是 Cube
- Cube function 需要 `reserve_buffer(location = MAT)`
- Vector function 需要 `import_reserved_buffer(peer_func = @cube_kernel)`

### 7.3 编译路径与地址处理路径

对包含 `reserve_buffer` 的函数，PTOAS 按当前编译流程是否启用 local address planning 以及 `auto` 的组合选择地址处理路径：

- 启用 local address planning + `auto = true`
  - 进入 auto 路径
  - 由 `PlanMemory` 为 `reserve_buffer` 分配 `base`
  - 随后由 `pto-resolve-reserved-buffers` 传播地址并完成 peer `flag_base` 对齐
- 跳过 local address planning + `auto = false` + 显式 `base`
  - 进入 manual 路径
  - 跳过 `PlanMemory`
  - 由 `pto-resolve-reserved-buffers` 直接传播已给定地址并完成 peer `flag_base` 对齐

以下组合均非法：

- 启用 local address planning + `auto = false`
- 跳过 local address planning + `auto = true`

若函数内不存在 `reserve_buffer`，则保持现有编译流程对 `PlanMemory` 的原始控制行为，不引入额外语义。

### 7.4 启用 local address planning 的 auto 路径

在启用 local address planning 的编译流程中，`reserve_buffer` 必须使用 `auto = true`，并由 plan memory 负责地址分配。

若函数中存在 `reserve_buffer`，则对其 `location` 对应的地址空间执行：

1. 先按现有逻辑完成普通 local buffer 的 `MemPlan`
2. 再收集该地址空间内已经分配完成的 local 区间
3. 对该地址空间内每条 `reserve_buffer`，按稳定顺序在剩余空洞中寻找一段满足大小与对齐要求的连续区间
4. 将找到的区间起始地址分别回填为对应 `reserve_buffer` 的 `base`

即：

- 普通 `memref.alloc` / tile buffer 等 local 内存仍先由既有 `MemPlan` 按原逻辑分配
- `reserve_buffer` 不参与普通 local buffer 的 inplace / reuse 规划
- `reserve_buffer` 在普通 local buffer 分配完成后，再作为独立的一段连续 local 区间进行 hole 分配；多条 `reserve_buffer` 会逐条占用不重叠区间
- `reserve_buffer` 不保证位于地址空间起始地址，也不保证形成预留前缀；其语义仅为“在该地址空间中为 consumer slot buffer 找到一段对齐且连续的可用地址”
- 若整体容量足够但 `MemPlan` 结果将空间打散，导致不存在满足大小和对齐要求的连续空洞，则 `reserve_buffer` 分配失败并报错

### 7.5 跳过 local address planning 的 manual 路径

在跳过 local address planning 的编译流程中：

- 每个 `reserve_buffer` 必须显式提供 `base`
- PTOAS 只校验 `base` 的基本合法性
- `PlanMemory` 不参与该函数的 local 地址分配
- 因此该函数中其他 local buffer 地址也必须已由前端或更前阶段整体确定
- 地址传播 pass 不做地址分配，只将显式 `base` 传播到 `import_reserved_buffer`

该 manual 路径的目标是：

- 保持前端或外部地址规划结果不被 PTOAS 改写
- 避免 `reserve_buffer` 显式地址与 PTOAS 自动规划结果相互覆盖

### 7.6 `import_reserved_buffer` 规则

- 不做地址分配

### 7.7 地址传播 pass 规则

对每个 `import_reserved_buffer`：

1. 通过 `peer_func` 找到 peer 函数
2. 在 peer 函数内查找同名 `reserve_buffer`
3. 读取对方已经解析出的 `base` 或其等价结果值
4. 用该常量地址替换 `import_reserved_buffer` 的结果

地址传播完成后：

- producer 与 consumer 对同一逻辑 pipe 使用同一个 local buffer 地址
- EmitC 只处理解析后的常量地址，不处理 `import_reserved_buffer`

#### 7.7.1 pass 落点

- PTOAS 增加独立 `ModulePass`：`pto-resolve-reserved-buffers`
- 该 pass 固定运行在 EmitC lowering 之前
- 启用规划时：运行在 `pto-plan-memory` 之后
- 跳过规划时：不经过 `pto-plan-memory`，但该 pass 仍会运行
- 该 pass 不负责地址分配，只消费前一阶段已经确定的 `reserve_buffer.base`

#### 7.7.2 输入假设

- 启用规划时，`reserve_buffer.auto = true`，其 `base` 已由 `PlanMemory` 回填
- 跳过规划时，`reserve_buffer.auto = false`，其 `base` 已由前端显式给定
- `import_reserved_buffer.peer_func` 已能解析到合法 peer function
- `import_reserved_buffer.name` 已能在 peer function 中找到唯一匹配的 `reserve_buffer`

#### 7.7.3 实现流程

pass 在模块级按两步执行：

1. 先建立 peer 对应关系
2. 再将 `reserve_buffer` / `import_reserved_buffer` 物化为显式常量地址

其中第一步的实现方式是：

- 遍历模块内所有 `pto.initialize_l2l_pipe` / `pto.initialize_l2g2l_pipe`
- 对每条 init op 的每个 local consumer 地址操作数，以”函数 + reserve 名字 + 方向”构建 PipePeerKey 并归入逻辑 pipe 分组：
  - `dir_mask = 1/2`：只有 `local_addr`，方向即 `dir_mask`
  - `dir_mask = 3`（DIR_BOTH）：一条 pipe 携带两个地址，分别归入两个逻辑方向——`local_addr` 归入 C2V（方向 1），`peer_local_addr` 归入 V2C（方向 2）
- global-only GM FIFO init 没有 local consumer 地址操作数，不参与 `reserve_buffer` / `import_reserved_buffer` 地址物化；它的 peer pipe 与 `flag_base` 对齐由 frontend lowering 保留的 logical pipe 绑定关系（`id + direction`）建立
- 将 peer 两侧引用到同一逻辑 pipe 的内部 init op 归并到同一组
- 若某条带 local consumer 地址操作数的 init 未显式提供 `flag_base`，则其 `local_addr` 必须来自 `reserve_buffer` 或 `import_reserved_buffer`
- 以这些逻辑 pipe 分组为边建立 peer component；每个 component 必须恰好包含两条、且分别来自 peer 两侧函数的兼容 init op，否则直接报错
- “兼容”指两侧 init 的 `dir_mask`、`slot_size`、`slot_num` 以及 `local_slot_num` 一致；因此 `DIR_BOTH` 与拆分后的单向 pipe 不能在同一条 peer 通道上混用
- 在同一 component 内，若任一侧已显式提供 `flag_base`，则该值作为该 component 最终值；若两侧显式值冲突则报错
- 若 component 内两侧都未显式提供 `flag_base`，则由 flag 分配器为该 component 选择一段与同函数其它 pipe component 不重叠的 flag 区间
- 完成分配后，将最终 `flag_base` 回填到该 component 内所有尚未显式填写的 init op，保证 peer 两侧一致

第二步的实现方式是：

- 对每个 `reserve_buffer`，读取其已解析 `base`
- 在该 op 位置插入 `arith.constant`
- 用该常量替换 `reserve_buffer` 结果值的全部 uses
- 对每个 `import_reserved_buffer`，通过 `peer_func + name` 找到 peer `reserve_buffer`
- 读取对方已解析 `base`
- 在当前 op 位置插入同值 `arith.constant`
- 用该常量替换 `import_reserved_buffer` 结果值的全部 uses
- 常量替换完成后，删除 `reserve_buffer` / `import_reserved_buffer`

#### 7.7.4 结果 IR 形态

地址传播 pass 之后：

- IR 中不再保留 `reserve_buffer` / `import_reserved_buffer`
- 内部 pipe init op 的 `local_addr` 只再引用普通 SSA 常量地址；global-only GM FIFO init 没有 `local_addr`
- 因而后续 EmitC 无需理解 frontend 预留地址语义，只需透传解析后的地址值

#### 7.7.5 失败条件

若出现以下情况，pass 直接报错：

- `reserve_buffer.base` 在 pass 运行时仍未解析
- 启用规划的编译流程却出现 `reserve_buffer.auto = false`
- 跳过规划的编译流程却出现 `reserve_buffer.auto = true`
- `peer_func` 无法解析到函数
- 在 peer function 中找不到同名 `reserve_buffer`
- 某条未显式提供 `flag_base` 且带 local consumer 地址操作数的内部 init，其 `local_addr` 不来自 `reserve_buffer` / `import_reserved_buffer`
- 基于 `reserve_buffer` / `import_reserved_buffer` 建立的某个 peer component，未形成完整兼容的 peer init pair
- peer `flag_base` 已显式给定但两侧取值冲突
- 同一函数内两个不同 pipe component 的 flag 区间重叠

## 8. flag 分配规则

### 8.1 总原则

- `flag_base` 由 PTOAS flag 分配阶段在内部 init op 上填写
- 在 flag 分配完成前，内部 init op 可以暂时不携带 `flag_base`
- peer 两侧同一逻辑 pipe 必须使用同一个 `flag_base`
- 同一函数内不同 pipe component 的 flag 区间必须互不重叠
- 硬件只提供 16 个 flag id，因此单函数内所有 pipe component 的 flag 区间总占用必须落在 `[0, 16)` 内

### 8.2 单向场景

当前规划中，单向 pipe component 每条占用一对逻辑 flag：

- 若该函数内没有更早分配的 pipe component，则首条单向 pipe 的 `flag_base = 0`
- 后续单向 pipe 继续按偶数递增分配，例如 `0`、`2`、`4` ...
- 每条单向 pipe component 占用逻辑 flag 对：`flag_base` 和 `flag_base + 1`
- 因此若函数内全是单向 pipe，最多可容纳 8 条

### 8.3 双向场景

当前规划中，当 `DIR_MASK = 3`（DIR_BOTH）时，单条物理 pipe component 固定占用两组逻辑 flag：

- 若该 component 的 `flag_base = B`，则它占用 `B/B+1` 与 `B+2/B+3`
- 因而 `DIR_BOTH` component 的 flag 宽度等价于两个单向 component
- 因此若函数内全是 `DIR_BOTH` pipe，最多可容纳 4 条
- 若单向与双向混用，则总规则仍是所有 component 的 flag 宽度之和不超过 16

对于单条 DIR_BOTH pipe，最终写回到内部 init op 的仍是该 component 的起始 `flag_base`，底层 pto-isa `TPipe<flagBase, Direction::DIR_BOTH, ...>` 会自动管理两个方向的 flag 对。

### 8.4 与地址传播的关系

地址传播 pass 在识别出 `import_reserved_buffer` 与 `reserve_buffer` 的 peer 对应关系后，同时可以完成 peer pipe 的 `flag_base` 对齐。

即：

- 基于同一 FIFO 通信的两条 peer init op，必须拿到相同的 `flag_base`

## 9. verifier 规则

### 9.1 前端 verifier

前端 IR 需满足以下约束：

- 每个函数 init op 数量是否合法
- 每个函数 `reserve_buffer` / `import_reserved_buffer` 数量是否合法
- `DIR_MASK` 取值是否合法
- `SLOT_SIZE > 0`
- 使用 consumer 侧 local FIFO buffer 时，`reserve_buffer.size` 必须匹配对应
  pipe 的本地 FIFO 字节数：A2/A3 GM FIFO 路径为
  `SLOT_SIZE * EFFECTIVE_LOCAL_SLOT_NUM`，A5 L2L 路径为
  `SLOT_SIZE * EFFECTIVE_SLOT_NUM`
- 使用 consumer 侧 local FIFO buffer 时，`reserve_buffer.location` 与 consumer 函数类型匹配
- `reserve_buffer.name` 在函数内唯一
- `import_reserved_buffer` 的 `(name, peer_func)` 在函数内唯一
- `reserve_buffer.auto = false` 时必须带 `base`
- `reserve_buffer.auto = true` 时必须不带 `base`
- driver / pipeline 级约束：启用规划的编译流程只接受 `auto = true`
- driver / pipeline 级约束：跳过规划的编译流程只接受 `auto = false` 且显式 `base`
- `import_reserved_buffer` 能在 `peer_func` 中找到同名 `reserve_buffer`
- 方向相关 op 只能出现在合法 kernel 中
- 前端数据传输 op 的 `split` 必须是合法的编译期常量属性
- `global` entry 形式的 `talloc_to_*` / `tpush_to_*` / `tpop_from_*` / `tfree_from_*` 只能绑定到 GM FIFO pipe（A2/A3 `initialize_l2g2l_pipe` 路径）
- 绑定到 global-only GM FIFO 的 initialize 只允许携带 `gm_slot_tensor`（可附带 `slot_num`），不得携带 `gm_slot_buffer`、`local_slot_num`、`c2v_consumer_buf`、`v2c_consumer_buf`；该路径不要求 `reserve_buffer` / `import_reserved_buffer`
- `gm_slot_tensor` 本身描述单个 slot entry；其字节数必须匹配 `slot_size`
- `talloc_to_*` / `tpop_from_*` 返回的 `tensor_view` 类型必须匹配 `gm_slot_tensor`
- `global` entry 的 dtype、shape 与 stride/layout 必须足以生成底层 `GlobalTensor<RawDType, Shape, Stride, Layout>` 类型
- `global` entry transaction 中，producer 侧 `tpush_to_*` 必须有同 pipe、同 entry 的支配性 `talloc_to_*`
- `global` entry transaction 中，consumer 侧 `tfree_from_*` 必须携带对应 `tpop_from_*` 返回的 entry；`tile` entry 路径保持无 operand `tfree_from_*`
- 同一次 logical transaction 不允许混用 `tile` entry 和 `global` entry

### 9.2 内部 IR verifier

内部 verifier 负责检查：

- `slot_size > 0`
- `slot_num >= 1`
- legacy 前端 `pto.aic_initialize_pipe` / `pto.aiv_initialize_pipe` 可显式提供
  `slot_num`；缺省时 `DIR_MASK=1/2` 使用 `8`，`DIR_MASK=3` 使用 `4`
- `local_slot_num` 若出现，可出现在 `pto.initialize_l2g2l_pipe` 或 legacy 前端
  `pto.aic_initialize_pipe` / `pto.aiv_initialize_pipe` 上，且必须大于 `0`
  且不大于其有效 `slot_num`；A5 和 global-only GM FIFO 不携带 `local_slot_num`
- `flag_base` 若出现，必须满足基本合法性；是否已填写以及具体分配值由 flag 分配保证
- `pto.initialize_l2g2l_pipe` 必须提供 `gm_addr` 或 `gm_slot_tensor`；只有存在 consumer 侧 local FIFO buffer 时才提供 `local_addr` / `peer_local_addr`
- `pto.initialize_l2l_pipe` 必须提供 `local_addr`
- `dir_mask = 1` 的 pipe 只能被 C2V 方向 lowering 使用
- `dir_mask = 2` 的 pipe 只能被 V2C 方向 lowering 使用
- `talloc/tpush/tpop/tfree` 的 `split` 必须是合法的编译期常量属性
- `pto.talloc` 只接受 `global` entry，且只能使用 `initialize_l2g2l_pipe` 产生的 pipe
- `pto.tpush` / `pto.tpop` 接受 `tile` entry 或 `global` entry；`global` entry 只能使用 `initialize_l2g2l_pipe` 产生的 pipe
- `pto.tfree(%pipe : !pto.pipe)` 仅用于 `tile` entry 路径；`pto.tfree(%entry, %pipe : !pto.tensor_view<...>, !pto.pipe)` 仅用于 `global` entry 路径
- `global` entry 的 `pto.tpush` 必须匹配同 pipe、同 entry 的 producer-side `pto.talloc`
- `global` entry 的 `pto.tfree` 必须匹配同 pipe、同 entry 的 consumer-side `pto.tpop`

### 9.3 关于 `split` 的校验边界

PTOAS 对 `split` 的处理边界如下：

- PTOAS 验证 `split` 是合法枚举值
- PTOAS 要求 `split` 以编译期常量属性形式出现
- PTOAS 不验证同一逻辑 pipe 上多个 `talloc/tpush/tpop/tfree` 的 `split` 是否一致
- PTOAS 不根据 `split` 改变地址分配、flag 分配或 pipe 配对

因此：

- `split` 混用是否语义正确，不是 PTOAS 静态保证项
- `split` 相关的语义正确性由前端生成逻辑或前端 verifier 保证
- PTOAS 只负责校验 `split` 枚举值合法，并将其透传到底层

## 10. EmitC 与 pto-isa 映射

### 10.1 初始化 op

在进入 EmitC 前：

- 前端 `pto.aic_initialize_pipe` / `pto.aiv_initialize_pipe`
- 前端 `pto.talloc_to_aiv` / `pto.talloc_to_aic`
- 前端 `pto.tpush_to_aiv` / `pto.tpush_to_aic`
- 前端 `pto.tpop_from_aic` / `pto.tpop_from_aiv`
- 前端 `pto.tfree_from_aic` / `pto.tfree_from_aiv`
- `pto.reserve_buffer` / `pto.import_reserved_buffer`

都必须已经被前序 pass 消除。

EmitC 只处理 PTOAS 内部统一 IR，不直接理解前端 pipe 接口或地址提示接口。

EmitC 将以下内部 init op 映射到底层 `TPipe`：

- `pto.initialize_l2l_pipe`
- `pto.initialize_l2g2l_pipe`

映射时需要使用以下信息：

- `dir_mask`
- `slot_size`
- `slot_num`
- `local_slot_num`
- `flag_base`
- `gm_addr` 或 `gm_slot_tensor`（仅 `initialize_l2g2l_pipe`）
- `local_addr`
- `peer_local_addr`（仅 `dir_mask = 3` 时）

`dir_mask` 到 `Direction` 枚举的映射：

| `dir_mask` | `Direction` 令牌 |
|---|---|
| 1 | `Direction::DIR_C2V` |
| 2 | `Direction::DIR_V2C` |
| 3 | `Direction::DIR_BOTH` |

当 `initialize_l2g2l_pipe` 来自 global-only GM FIFO 时，IR 中保留完整 `gm_slot_tensor` / tensor-like `gm_addr` 类型用于校验 slot descriptor；EmitC 构造底层 `TPipe` 时只传 GM FIFO 的起始地址。如果 `gm_addr` 仍是 `tensor_view` 形式，EmitC 先取 `PTOAS__GLOBAL_TENSOR_DATA(gm_addr)`；如果前端函数入参是 `!pto.ptr` 并通过 `pto.make_tensor_view` 描述形状，则最终仍直接把该 ptr 起始地址传给 `TPipe`。

当 `dir_mask = 3` 时，EmitC 将 `local_addr` 作为 C2V consumer buf、`peer_local_addr` 作为 V2C consumer buf 传入 `TPipe` 构造函数。

其中：

- 若 `flag_base` 尚未在 EmitC 前完成填写，PTOAS 应报错。

### 10.2 数据传输 op

EmitC 将以下内部数据传输 op 映射到底层：

- `pto.talloc` -> `TALLOC`
- `pto.tpush` -> `TPUSH`
- `pto.tpop` -> `TPOP`
- `pto.tfree` -> `TFREE`

映射时需要使用以下信息：

- pipe entry（`tile` 或 `global`）
- `split`
- `pipe`

其中：

- `split` 不在 PTOAS 内部解释
- `split` 作为底层 `TALLOC/TPUSH/TPOP/TFREE` 的编译期模板实参透传
- `tile` entry 映射到 `TPUSH<Pipe, Tile, Split>` / `TPOP<Pipe, Tile, Split>` / `TFREE<Pipe, Split>`
- `global` entry 映射到 `TALLOC<Pipe, GlobalData, Split>` / `TPUSH<Pipe, GlobalData, Split>` / `TPOP<Pipe, GlobalData, Split>` / `TFREE<Pipe, GlobalData, Split>`
- `GlobalData` 的类型来自单个 slot 的 `tensor_view` entry，必须与 pipe init 的 `gm_slot_tensor` descriptor 匹配；`TPipe` 构造函数本身只接收 GM FIFO 起始地址和两个 consumer buffer 地址
- `global` entry 的 `TPUSH` / `TFREE` 不执行 `TSTORE` / `TLOAD`；它们只使用 entry descriptor 作为底层 transaction 描述符

### 10.3 InsertSync

`split` 不影响 PTOAS 中的 pipeline derivation 与 InsertSync 规则。

InsertSync 只依赖：

- op 种类
- init op 形态
- `dir_mask`
- 目标架构

而不依赖 `split`。

## 11. 编译流程总览

完整流程如下：

```text
前端 IR 接口
  -> lowering pass
  -> PTOAS 内部统一 IR
  -> plan memory
  -> 地址传播 pass
  -> EmitC
  -> pto-isa C++ 代码
```

其中：

- lowering pass 负责拆分 `DIR_MASK=3`、绑定方向与 pipe
- 启用规划的编译流程中，plan memory 先按既有逻辑规划普通 local buffer，再为 `reserve_buffer` 在目标地址空间中分配 hole
- 跳过规划的编译流程中，不运行 plan memory；`reserve_buffer.base` 必须已由前端给定
- 地址传播 pass 负责 `import_reserved_buffer` 常量替换与 peer pipe 的 `flag_base` 对齐
- EmitC 只负责将内部 `initialize_l2l_pipe` / `initialize_l2g2l_pipe` / `talloc` / `tpush` / `tpop` / `tfree` 及其属性透传到底层
