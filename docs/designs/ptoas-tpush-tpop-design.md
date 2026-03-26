# TPUSH/TPOP 前端接口与 PTOAS 实现设计

## 1. 文档范围

本文定义PTOAS TPUSH/TPOP 前端IR接口，以及其在 PTOAS 内部的 lowering、地址传播、flag 分配和 EmitC 映射规则。

本文覆盖两层接口：

- 前端接口
  - `pto.aic_initialize_pipe`
  - `pto.aiv_initialize_pipe`
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
  - `pto.tpush`
  - `pto.declare_tile`
  - `pto.tpop`
  - `pto.tfree`

本文只描述接口契约与编译流程，不展开具体 C++ 模板实现细节。

## 2. 设计目标

本设计的目标如下：

- 对前端提供\*\_initialize_pipe/tpush_to_\*/tpop_from_\*/tfree_from_\*IR接口。
- 在 PTOAS 内部统一为 pipe/tpush/tpop/tfree 指令，便于复用已有 pass。
- 支持 A2/A3 与 A5 两个平台使用同一套前端接口。
- 定义consumer slot buffer的分配地址与producer之间的匹配关系，并传播。

## 3. 前端 IR 接口定义

### 3.1 `pto.aic_initialize_pipe`

#### 语义

由 Cube kernel 在函数启动时调用，初始化该函数涉及的通信 pipe。

#### 语法

```mlir
pto.aic_initialize_pipe(
    DIR_MASK,
    SLOT_SIZE,
    GM_SLOT_BUFFER,
    C2V_CONSUMER_BUF,
    V2C_CONSUMER_BUF)
```

#### 参数

| 参数 | 类型 | 说明 |
|---|---|---|
| `DIR_MASK` | 编译期整数常量 | `1`、`2` 或 `3` |
| `SLOT_SIZE` | 编译期整数常量 | 单 slot 字节数，定义为切分前完整 tile 字节数 |
| `GM_SLOT_BUFFER` | GM 地址或空值 | A2/A3 路径使用，A5 路径为空 |
| `C2V_CONSUMER_BUF` | `i32` | C2V 方向 consumer 的 local slot buffer 基址 |
| `V2C_CONSUMER_BUF` | `i32` | V2C 方向 consumer 的 local slot buffer 基址 |

### 3.2 `pto.aiv_initialize_pipe`

#### 语义

由 Vector kernel 在函数启动时调用，初始化该函数涉及的通信 pipe。

#### 语法

```mlir
pto.aiv_initialize_pipe(
    DIR_MASK,
    SLOT_SIZE,
    GM_SLOT_BUFFER,
    C2V_CONSUMER_BUF,
    V2C_CONSUMER_BUF)
```

参数语义与 `pto.aic_initialize_pipe` 相同。

### 3.3 前端数据传输接口

#### `pto.tpush_to_aiv`

```mlir
pto.tpush_to_aiv(%tile) { split = 0 }
```

- 仅出现在 Cube kernel 中
- 表示 C2V 方向 producer push

#### `pto.tpush_to_aic`

```mlir
pto.tpush_to_aic(%tile) { split = 0 }
```

- 仅出现在 Vector kernel 中
- 表示 V2C 方向 producer push

#### `pto.tpop_from_aic`

```mlir
%tile = pto.tpop_from_aic { split = 0 } -> !pto.tile_buf<...>
```

- 仅出现在 Vector kernel 中
- 表示 C2V 方向 consumer pop

#### `pto.tpop_from_aiv`

```mlir
%tile = pto.tpop_from_aiv { split = 0 } -> !pto.tile_buf<...>
```

- 仅出现在 Cube kernel 中
- 表示 V2C 方向 consumer pop

#### `pto.tfree_from_aic`

```mlir
pto.tfree_from_aic { split = 0 }
```

- 仅出现在 Vector kernel 中
- 表示 C2V 方向 consumer free

#### `pto.tfree_from_aiv`

```mlir
pto.tfree_from_aiv { split = 0 }
```

- 仅出现在 Cube kernel 中
- 表示 V2C 方向 consumer free

以上前端数据传输接口中的 `split` 均为编译期常量属性，不是运行时 SSA operand。

- 取值使用 `TileSplitAxis` 枚举语义：`0/1/2` 分别对应 `TILE_NO_SPLIT`、`TILE_UP_DOWN`、`TILE_LEFT_RIGHT`
- lowering 到 PTOAS 内部 IR 时，`split` 继续以属性形式保留

### 3.4 地址提示接口

#### `pto.reserve_buffer`

用于在当前函数内声明一块 consumer slot buffer 预留空间。其合法写法由
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
- 在当前约束下，每个函数最多一条 `reserve_buffer`
- 编译路径与 `auto` 的合法组合只有两种：
  - 启用 local address planning：`auto = true`，且不带 `base`
  - 跳过 local address planning：`auto = false`，且显式提供 `base`

#### `pto.import_reserved_buffer`

用于引用 peer function 中已经定义的 `reserve_buffer` 结果。

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

### 3.5 前端层约束

> **约束来源说明**：以下约束是当前软件方案的设计选择，而非硬件限制。
> 当前 `tpush_to_aic` / `tpush_to_aiv` / `tpop_from_aic` / `tpop_from_aiv` /
> `tfree_from_aic` / `tfree_from_aiv` 等前端接口上不携带 pipe 参数，因此在
> lowering 时只能将函数内的所有 push/pop/free 绑定到同一条 pipe。这决定了每个
> 函数最多只能存在一条初始化语句（即一条 pipe）。如果后续需要支持多条 pipe 并行
> 通信，前端接口方案需要重新设计（例如在 push/pop/free 上显式引用 pipe handle）。

前端 IR 需满足以下约束：

- 每个 Cube function 最多一条 `pto.aic_initialize_pipe`
- 每个 Vector function 最多一条 `pto.aiv_initialize_pipe`
- 每个函数内最多一条 C2V 逻辑 pipe 和一条 V2C 逻辑 pipe
- 每个函数最多一条 `reserve_buffer`
- 每个函数最多一条 `import_reserved_buffer`
- `DIR_MASK` 只允许 `1`、`2`、`3`
- `SLOT_SIZE > 0`
- `reserve_buffer.size == SLOT_SIZE * SLOT_NUM`
- C2V consumer 的 `reserve_buffer.location` 必须是 `VEC`
- V2C consumer 的 `reserve_buffer.location` 必须是 `MAT`
- `reserve_buffer.name` 在本函数内必须唯一
- op 级约束：`reserve_buffer.auto = false` 时必须提供 `base`
- op 级约束：`reserve_buffer.auto = true` 时必须不提供 `base`
- 启用 local address planning 的编译流程：`reserve_buffer` 只允许 `auto = true`
- 跳过 local address planning 的编译流程：`reserve_buffer` 只允许 `auto = false` 且显式提供 `base`
- `import_reserved_buffer` 必须能在 `peer_func` 中找到同名 `reserve_buffer`

## 4. 核心约定

### 4.1 逻辑 pipe

本文中的”逻辑 pipe”指一条通信通道。

- C2V：Cube producer -> Vector consumer
- V2C：Vector producer -> Cube consumer

`DIR_MASK=3` 表示前端一个同时包含 C2V 和 V2C 的初始化请求。在 PTOAS lowering
后，生成单条 `dir_mask = 3` 的 DIR_BOTH 内部 pipe，同时承载 C2V 和 V2C 双向
通信。该 pipe 携带两个地址操作数：`local_addr`（C2V consumer buf）和
`peer_local_addr`（V2C consumer buf）。下游 TPUSH/TPOP/TFREE 共享同一 pipe
handle。

### 4.2 `split` 的角色

`split` 使用 `TileSplitAxis` 枚举表达：

- `TILE_NO_SPLIT`
- `TILE_UP_DOWN`
- `TILE_LEFT_RIGHT`

在 PTOAS 设计中，`split` 的角色定义为：

- `split` 是 `tpush/tpop/tfree` 的逐指令执行模式
- `split` 在 IR 中表示为编译期常量属性，不是运行时 SSA operand
- `split` 不参与pipe 初始化
- `split` 不参与 plan memory、地址传播、flag 分配
- PTOAS 将 `split` 作为透明的编译期参数向 EmitC 和底层 pto-isa 透传

因此：

- 同一条逻辑 pipe 上可以出现不同 `split` 的 `tpush/tpop/tfree`
- PTOAS 不要求同一逻辑 pipe 内所有指令使用同一个 `split`
- `split` 相关的语义正确性由前端生成逻辑或前端 verifier 保证；PTOAS 仅校验 `split` 枚举合法并向下透传

### 4.3 `SLOT_SIZE` 的定义

`SLOT_SIZE` 的定义固定为：

- 切分前完整 tile 的字节数

即使 `split` 为 `TILE_UP_DOWN` 或 `TILE_LEFT_RIGHT`，`SLOT_SIZE` 仍然表示未切分前的逻辑 tile 总字节数。

`split` 只影响底层 `TPUSH/TPOP/TFREE` 的执行方式，不影响 `SLOT_SIZE` 的含义。

### 4.4 `SLOT_NUM` 规则

`SLOT_NUM` 由 `DIR_MASK` 固定决定：

- `DIR_MASK = 1` 或 `2`：`SLOT_NUM = 8`
- `DIR_MASK = 3`（DIR_BOTH）：单条 pipe，`SLOT_NUM = 4`（每方向 4 slot，总缓冲 2 × 4 × SLOT_SIZE）

`SLOT_NUM` 不由 `split` 决定。

## 5. PTOAS 内部 IR 接口定义

### 5.1 `!pto.pipe`

本文设计的内部 `!pto.pipe` 为不透明 handle。

`!pto.pipe` 的协议信息由其定义 op 上的属性承载，而不是由 type 参数承载。

底层 `pto-isa` 若对 `TPUSH/TPOP` 的模板形态继续演进，不反向约束 `!pto.pipe` 的 type 设计；内部 `!pto.pipe` 仍保持 opaque handle。

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
  - 仅 `initialize_l2g2l_pipe` 承载
  - 表示 GM 路径下 consumer 侧 local slot buffer 的槽数
  - 仅在通过 GM 传递时对底层 `TPipe` 模板参数有意义，不改变 GM FIFO 的 `slot_num`
  - 缺省值等于该内部 pipe 的 `slot_num`
  - 因此当前固定规则下：
    - `DIR_MASK=1/2` 直接 lowering 时，`local_slot_num = 8`
    - `DIR_MASK=3` 单条 DIR_BOTH pipe，`local_slot_num = 4`
- `flag_base`
  - 由 PTOAS flag 分配阶段填写
  - frontend lowering 阶段可以缺省
  - EmitC 前必须已经解析为显式常量

#### 操作数

- `gm_addr`
- `local_addr`：C2V consumer buf（或单向时唯一方向的 consumer buf）
- `peer_local_addr`（可选）：V2C consumer buf，仅 `dir_mask = 3` 时出现

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

### 5.4 `pto.tpush`

```mlir
pto.tpush(%tile, %pipe) { split = 0 }
```

### 5.5 `pto.declare_tile`

```mlir
%tile = pto.declare_tile -> !pto.tile_buf<...>
```

### 5.6 `pto.tpop`

```mlir
pto.tpop(%tile, %pipe) { split = 0 }
```

### 5.7 `pto.tfree`

```mlir
pto.tfree(%pipe) { split = 0 }
```

`split` 在内部 IR 中必须以编译期常量属性形式保留，不能在 lowering 时擦除或降为运行时 operand。

## 6. 前端到内部 IR 的 lowering 规则

### 6.1 初始化接口 lowering

#### A2/A3

- `pto.aic_initialize_pipe` 和 `pto.aiv_initialize_pipe` lower 为 `pto.initialize_l2g2l_pipe`
- 若前端未提供更具体信息，lowering 默认补上 `local_slot_num = slot_num`

#### A5

- `pto.aic_initialize_pipe` 和 `pto.aiv_initialize_pipe` lower 为 `pto.initialize_l2l_pipe`

### 6.2 `DIR_MASK=1/2`

- 只生成一条内部 pipe
- `slot_num = 8`
- 对 `initialize_l2g2l_pipe`，`local_slot_num = 8`

### 6.3 `DIR_MASK=3`

前端一个 init op 生成**单条** DIR_BOTH 内部 pipe：

- `%pipe`：`dir_mask = 3`，`slot_num = 4`
- 若 lowering 为 `initialize_l2g2l_pipe`，`local_slot_num = 4`

地址选择规则：

- `local_addr` = `C2V_CONSUMER_BUF`
- `peer_local_addr` = `V2C_CONSUMER_BUF`

`FrontendPipeHandles` 中 `c2vPipe` 和 `v2cPipe` 指向同一个 pipe Value。

### 6.4 前端数据传输 op 与内部 pipe 的绑定

绑定规则固定如下：

| 前端 op | 所在函数 | 方向 | 使用的内部 pipe |
|---|---|---|---|
| `tpush_to_aiv` | Cube | C2V | `c2vPipe` |
| `tpop_from_aic` | Vector | C2V | `c2vPipe` |
| `tfree_from_aic` | Vector | C2V | `c2vPipe` |
| `tpush_to_aic` | Vector | V2C | `v2cPipe` |

当 `DIR_MASK=3` 时，`c2vPipe` 和 `v2cPipe` 指向同一个 DIR_BOTH pipe，下游 TPUSH/TPOP/TFREE 只关心 pipe handle 是否存在，不关心是否是同一个。
| `tpop_from_aiv` | Cube | V2C | `dir_mask = 2` |
| `tfree_from_aiv` | Cube | V2C | `dir_mask = 2` |

### 6.5 数据传输 op lowering

#### `tpush_to_aiv` / `tpush_to_aic`

lower 为：

```mlir
pto.tpush(%tile, %pipe) { split = 0 }
```

#### `tpop_from_aic` / `tpop_from_aiv`

lower 为：

```mlir
%decl = pto.declare_tile -> !pto.tile_buf<...>
pto.tpop(%decl, %pipe) { split = 0 }
```

即：

- 前端 `pto.tpop_from_aic` / `pto.tpop_from_aiv` 是返回 tile 结果值的接口
- PTOAS 内部 `pto.tpop` 才是 destination-style 形式，显式接收一个 `pto.declare_tile` 结果作为入参

#### `tfree_from_aic` / `tfree_from_aiv`

lower 为：

```mlir
pto.tfree(%pipe) { split = 0 }
```

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
3. 在剩余空洞中按地址空间对齐要求寻找一段可容纳 `reserve_buffer.size` 的连续区间
4. 将该区间起始地址回填为这条唯一 `reserve_buffer` 的 `base`

即：

- 普通 `memref.alloc` / tile buffer 等 local 内存仍先由既有 `MemPlan` 按原逻辑分配
- `reserve_buffer` 不参与普通 local buffer 的 inplace / reuse 规划
- `reserve_buffer` 在普通 local buffer 分配完成后，再作为独立的一段连续 local 区间进行 hole 分配
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
- 对每条 init op 的每个地址操作数，以”函数 + reserve 名字 + 方向”构建 PipePeerKey 并归入逻辑 pipe 分组：
  - `dir_mask = 1/2`：只有 `local_addr`，方向即 `dir_mask`
  - `dir_mask = 3`（DIR_BOTH）：一条 pipe 携带两个地址，分别归入两个逻辑方向——`local_addr` 归入 C2V（方向 1），`peer_local_addr` 归入 V2C（方向 2）
- 将 peer 两侧引用到同一逻辑 pipe 的内部 init op 归并到同一组
- 若某条 init 未显式提供 `flag_base`，则其 `local_addr` 必须来自 `reserve_buffer` 或 `import_reserved_buffer`
- 对每个逻辑 pipe 分组，要求必须形成完整 peer init pair：恰好两条 init，且分别来自 peer 两侧函数；若 peer 信息不完整则直接报错
- 在同一组内，若任一侧已显式提供 `flag_base`，则该值作为该组最终值；若两侧显式值冲突则报错
- 若同组两侧都未显式提供 `flag_base`，则按默认规则回填：
  - 单向场景：`flag_base = 0`
  - 双向场景：C2V 组 `flag_base = 0`，V2C 组 `flag_base = 2`
- “双向场景”指同一对 peer 函数之间同时存在 C2V 和 V2C 两个逻辑 pipe 分组；DIR_BOTH 的一条物理 pipe 天然产生这两个分组
- 完成分组决策后，将最终 `flag_base` 回填到该组内所有尚未显式填写的 init op，保证 peer 两侧一致

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
- 内部 pipe init op 的 `local_addr` 只再引用普通 SSA 常量地址
- 因而后续 EmitC 无需理解 frontend 预留地址语义，只需透传解析后的地址值

#### 7.7.5 失败条件

若出现以下情况，pass 直接报错：

- `reserve_buffer.base` 在 pass 运行时仍未解析
- 启用规划的编译流程却出现 `reserve_buffer.auto = false`
- 跳过规划的编译流程却出现 `reserve_buffer.auto = true`
- `peer_func` 无法解析到函数
- 在 peer function 中找不到同名 `reserve_buffer`
- 某条未显式提供 `flag_base` 的内部 init，其 `local_addr` 不来自 `reserve_buffer` / `import_reserved_buffer`
- 基于 `reserve_buffer` / `import_reserved_buffer` 建立的某个逻辑 pipe 分组，未形成完整 peer init pair
- peer `flag_base` 已显式给定但两侧取值冲突

## 8. flag 分配规则

### 8.1 总原则

- `flag_base` 由 PTOAS flag 分配阶段在内部 init op 上填写
- 在 flag 分配完成前，内部 init op 可以暂时不携带 `flag_base`
- peer 两侧同一逻辑 pipe 必须使用同一个 `flag_base`

### 8.2 单向场景

当前规划中，当 `DIR_MASK = 1` 或 `2` 且函数内仅有该唯一逻辑 pipe 时，可采用：

- 该方向唯一逻辑 pipe 的 `flag_base = 0`
- 该 pipe 占用逻辑 flag 对：`0` 和 `1`

### 8.3 双向场景

当前规划中，当 `DIR_MASK = 3`（DIR_BOTH）时，虽然物理上只有一条 pipe，但 resolve pass 将其拆为两个逻辑方向分别分配 `flag_base`：

- C2V 方向：`flag_base = 0`
- V2C 方向：`flag_base = 2`

因此双向固定占用两组逻辑 flag：

- C2V：`0` / `1`
- V2C：`2` / `3`

对于单条 DIR_BOTH pipe，最终 `flag_base` 取 C2V 方向的值（`0`），底层 pto-isa `TPipe<flagBase, Direction::DIR_BOTH, ...>` 会自动管理两个方向的 flag 对。

### 8.4 与地址传播的关系

地址传播 pass 在识别出 `import_reserved_buffer` 与 `reserve_buffer` 的 peer 对应关系后，同时可以完成 peer pipe 的 `flag_base` 对齐。

即：

- 基于同一 FIFO 通信的两条 peer init op，必须拿到相同的 `flag_base`

## 9. verifier 规则

### 9.1 前端 verifier

前端 verifier 负责检查：

- 每个函数 init op 数量是否合法
- 每个函数 `reserve_buffer` / `import_reserved_buffer` 数量是否合法
- `DIR_MASK` 取值是否合法
- `SLOT_SIZE > 0`
- `reserve_buffer.size == SLOT_SIZE * SLOT_NUM`
- `reserve_buffer.location` 与 consumer 函数类型匹配
- `reserve_buffer.name` 在函数内唯一
- `reserve_buffer.auto = false` 时必须带 `base`
- `reserve_buffer.auto = true` 时必须不带 `base`
- driver / pipeline 级约束：启用规划的编译流程只接受 `auto = true`
- driver / pipeline 级约束：跳过规划的编译流程只接受 `auto = false` 且显式 `base`
- `import_reserved_buffer` 能在 `peer_func` 中找到同名 `reserve_buffer`
- 方向相关 op 只能出现在合法 kernel 中
- 前端数据传输 op 的 `split` 必须是合法的编译期常量属性

### 9.2 内部 IR verifier

内部 verifier 负责检查：

- `slot_size > 0`
- `slot_num` 只允许 `8` 或 `4`
- `DIR_MASK=1/2` 时，`slot_num` 必须与单向/双向 lowering 规则一致
- `local_slot_num` 若出现，只允许出现在 `pto.initialize_l2g2l_pipe` 上，且必须大于 `0` 且不大于 `slot_num`
- `flag_base` 若出现，必须满足基本合法性；是否已填写以及具体分配值由 flag 分配保证
- `pto.initialize_l2g2l_pipe` 必须提供 `gm_addr` 和 `local_addr`
- `pto.initialize_l2l_pipe` 必须提供 `local_addr`
- `dir_mask = 1` 的 pipe 只能被 C2V 方向 lowering 使用
- `dir_mask = 2` 的 pipe 只能被 V2C 方向 lowering 使用
- `tpush/tpop/tfree` 的 `split` 必须是合法的编译期常量属性

### 9.3 关于 `split` 的校验边界

PTOAS 对 `split` 的处理边界如下：

- PTOAS 验证 `split` 是合法枚举值
- PTOAS 要求 `split` 以编译期常量属性形式出现
- PTOAS 不验证同一逻辑 pipe 上多个 `tpush/tpop/tfree` 的 `split` 是否一致
- PTOAS 不根据 `split` 改变地址分配、flag 分配或 pipe 配对

因此：

- `split` 混用是否语义正确，不是 PTOAS 静态保证项
- `split` 相关的语义正确性由前端生成逻辑或前端 verifier 保证
- PTOAS 只负责校验 `split` 枚举值合法，并将其透传到底层

## 10. EmitC 与 pto-isa 映射

### 10.1 初始化 op

在进入 EmitC 前：

- 前端 `pto.aic_initialize_pipe` / `pto.aiv_initialize_pipe`
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
- `local_slot_num`（仅 `initialize_l2g2l_pipe`）
- `flag_base`
- `gm_addr`（仅 `initialize_l2g2l_pipe`）
- `local_addr`
- `peer_local_addr`（仅 `dir_mask = 3` 时）

`dir_mask` 到 `Direction` 枚举的映射：

| `dir_mask` | `Direction` 令牌 |
|---|---|
| 1 | `Direction::DIR_C2V` |
| 2 | `Direction::DIR_V2C` |
| 3 | `Direction::DIR_BOTH` |

当 `dir_mask = 3` 时，EmitC 将 `local_addr` 作为 C2V consumer buf、`peer_local_addr` 作为 V2C consumer buf 传入 `TPipe` 构造函数。

其中：

- 若 `flag_base` 尚未在 EmitC 前完成填写，PTOAS 应报错。

### 10.2 数据传输 op

EmitC 将以下内部数据传输 op 映射到底层：

- `pto.tpush` -> `TPUSH`
- `pto.tpop` -> `TPOP`
- `pto.tfree` -> `TFREE`

映射时需要使用以下信息：

- `tile`
- `split`
- `pipe`

其中：

- `split` 不在 PTOAS 内部解释
- `split` 作为底层 `TPUSH/TPOP/TFREE` 的编译期模板实参透传

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
- EmitC 只负责将内部 `initialize_l2l_pipe` / `initialize_l2g2l_pipe` / `tpush` / `tpop` / `tfree` 及其属性透传到底层
