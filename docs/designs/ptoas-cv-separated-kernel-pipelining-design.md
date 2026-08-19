# PTOAS C/V 分离式 Kernel 可配置 Preload Pipeline 设计

## 1. 文档范围

本文定义 PTOAS 对 Cube/Vector（下文简称 C/V）分离式 kernel 进行跨核流水调度的算法。
设计以 Flash Attention 为主要输入形态，同时把调度规则抽象为可复用的 stage graph，覆盖：

- 在已经完成 C/V 分离表达的两个 `func.func` 之间匹配通信 pipe；
- 保持外层 Q/S0 tile 与内层 K/V 的 S1 tile 两层数据遍历不变；
- 以可配置的 `preload_count = P` 改写 S1 内层循环；
- preload 一个完整的 QK/softmax 前缀，而不是只提前加载 K；
- 使用现有 `tpush` / `tpop` / `tfree` 的阻塞与 FIFO 语义完成跨核衔接，不引入 ACK；
- 用 multi-buffer 保存跨越 pipeline cut 的 Vector 本地状态；
- 对 A3 与 A5 使用同一调度算法，同时允许目标相关 lowering 选择不同 transport；
- 使调度算法不依赖 pipe 使用 `tile` entry 还是 `global`/`gm_slot_tensor` entry。

本文描述的是编译器设计，不定义新的硬件 pipe 指令，也不改变
[`ptoas-tpush-tpop-design.md`](ptoas-tpush-tpop-design.md) 中已有的 entry 和 transport 语义。

## 2. 设计结论

本设计采用以下统一方案：

1. 在 ModuleOp 层同时分析一对 C/V 函数，构建跨函数的逻辑 stage graph。
2. 对每个 S1 tile `i`，识别 `C_QK(i) -> V_P(i) -> C_PV(i) -> V_O(i)` 四个完整 stage。
3. 把前两个 stage 作为 pipeline prefix，把后两个 stage 作为 pipeline suffix。
4. 在 Cube 和 Vector 各自的内层循环中，让 prefix 相对 suffix 领先 `P` 个逻辑 tile。
5. 通信事务整体移动；调度器只识别 acquire/commit/release 边界，不检查 entry 的具体表示。
6. A5 tile entry lower 为本地 L2L pipe；A2/A3 tile entry lower 为 raw GM buffer 的 L2G2L
   pipe；A2/A3 global entry 可 lower 为 `gm_slot_tensor` 的 global-only GM FIFO。
7. entry/transport 合法性由已有 verifier 和目标 lowering 决定，pipeline scheduler 不为不同平台复制算法。

因此，同一份 pipeline 结果可以服务以下部署策略：

- A5：只允许 tile entry，使用 local buffer TPipe，不出现 GM tensor entry；
- A2/A3：只允许 tile entry，使用 `gm_slot_buffer + consumer local buffer`；
- A2/A3：允许 global entry，使用 `gm_slot_tensor`；
- 后续平台：增加新的 transport adapter，而不修改 preload 调度。

## 3. 目标与非目标

### 3.1 目标

- `P` 可由编译选项或函数属性显式指定。
- `P = 0` 保留原始逐 tile 串行顺序，便于 A/B 对比与回退。
- `P > 0` 时，QK matmul、softmax 和对应的 push 都属于 preload 范围。
- K 与 V 仍按相同逻辑 tile `i` 配对，不改变注意力计算语义。
- 调度结果不依赖 A3/A5，不依赖 entry 是 tile 还是 global。
- 不引入跨核 phase ACK、全局 barrier 或额外的 stage handshake。
- 对容量不足、pipe 配对失败、循环不匹配和状态无法 multi-buffer 的情况给出确定性诊断。

### 3.2 非目标

- 本阶段不自动搜索最优 `P`。
- 本阶段不改变 Q/S0 的外层切分策略。
- 本阶段不融合 C/V 函数，也不把两个核上的 op 放入同一个函数。
- 本阶段不为任意不可规约 CFG 做软件流水；首版只处理规范化的单入口 S1 循环。
- 本阶段不承诺“所有 P 个 prefix 已在两个核上同时完成后才启动 suffix”的全局相位语义。
  `P` 表示程序顺序上的提前距离；实际并行度由两个核的执行速度与 FIFO backpressure 决定。

## 4. 术语与基本模型

### 4.1 循环层次

Flash Attention 输入通常有两层 tile 遍历：

```text
for q in S0 tiles:          // 外层：Q tile
  initialize row state
  for i in S1 tiles:        // 内层：与当前 Q tile 配对的 K/V tile
    process K[i], V[i]
  finalize O[q]
```

pipeline 只改写 S1 内层循环。它不会新增第三个数据维度，也不会把 Q、K、V 分别变成三层循环。
文中出现的 prologue、steady-state、epilogue 是同一个 S1 循环的三个调度区间，而不是三层嵌套循环。

### 4.2 `preload_count`

设当前 Q tile 对应的 S1 tile 数为 `N`，用户指定值为 `P`：

```text
Pe = min(P, N)
```

`Pe` 是本次外层迭代的有效 preload 距离。

- `P = 0`：prefix 与 suffix 在同一个逻辑迭代中依次执行；
- `0 < P < N`：suffix `i` 前先调度 prefix `i + P`；
- `P >= N`：先调度当前 Q tile 的全部 prefix，再排空全部 suffix。

`P` 是逻辑调度距离，不等于任意时刻 FIFO 中必须存在的 entry 数。阻塞式 `tpush`/`tpop`
可以让实际 outstanding 数小于该距离。

### 4.3 Stage、事务与 transport

- **stage**：一个不能跨 pipeline cut 随意拆开的计算与通信单元。
- **pipe transaction**：从 entry acquire/allocate 到 commit，或从 acquire/pop 到 release/free 的有序操作组。
- **entry kind**：`tile` 或 `global`。
- **transport kind**：A5 L2L、A2/A3 L2G2L 等目标相关实现。

调度器处理 stage 与逻辑事务；entry kind 和 transport kind 只作为合法性及成本元数据。

## 5. Flash Attention 的规范 Stage Graph

对每个 S1 tile `i`，C/V 分离后的计算归一为四个 stage：

```text
C_QK(i) -> pipe_qk -> V_P(i) -> pipe_p -> C_PV(i) -> pipe_pv -> V_O(i)
```

### 5.1 `C_QK(i)`：Cube prefix stage

必须包含当前 tile 的完整 QK producer 路径：

1. 加载或取得 `K[i]`；
2. 完成 K/Q 所需的 layout、搬运和 Cube 输入准备；
3. 执行 `Q @ K[i]^T`；
4. 完成结果 entry 的 producer 写入；
5. `tpush` QK entry 到 Vector。

只把 K 的 `tload` 提到前面、而把 QK matmul 与 `tpush` 留在原位置，不构成本文定义的 preload。
这样只能隐藏部分 MTE 延迟，不能形成 C/V stage 之间的稳定距离。

### 5.2 `V_P(i)`：Vector prefix stage

必须包含当前 tile 的完整 softmax/probability 路径：

1. `tpop` QK entry；
2. mask、scale、row max、exp、row sum 等 Vector 计算；
3. 产生 suffix 所需的逐 tile normalization 状态；
4. 完成 P entry 的 producer 写入并 `tpush` 到 Cube；
5. 在 QK entry 最后一次读取后 `tfree`。

### 5.3 `C_PV(i)`：Cube suffix stage

必须包含：

1. `tpop` P entry；
2. 加载或取得同一个逻辑 tile 的 `V[i]`；
3. 执行 `P[i] @ V[i]`；
4. 完成 PV entry 的 producer 写入并 `tpush` 到 Vector；
5. 在 P entry 最后一次读取后 `tfree`。

V 侧代码必须随 QK preload 一起调整。若只提前 `C_QK` 而不把 `V_P` 作为 prefix，Cube 仍会逐 tile
等待 P，无法形成期望的 C/V 重叠。

### 5.4 `V_O(i)`：Vector suffix stage

必须包含：

1. `tpop` PV entry；
2. 使用 `V_P(i)` 保存的 normalization 状态更新 O accumulator；
3. 在 PV entry 最后一次读取后 `tfree`。

若 tile 0 使用初始化公式、后续 tile 使用累加公式，stage extractor 把它记录为
`V_P_init`/`V_P_update` 和 `V_O_init`/`V_O_update` 两种 emitter，而不是建立新的数据循环。

### 5.5 Pipeline cut

统一 cut 定义为：

```text
prefix = { C_QK, V_P }
suffix = { C_PV, V_O }
```

此 cut 的直接通信边是 `pipe_p`。所有从 `V_P(i)` 活到 `V_O(i)` 的 Vector 本地 SSA 值或内存状态
也是 cut edge，必须进行 multi-buffer 扩展。

## 6. Entry 无关的逻辑 Pipe Transaction

### 6.1 为什么调度不能直接匹配某一种 op 形态

tile entry 与 global entry 的数据搬运位置不同：

- tile entry 的 `tpush`/`tpop` 自身携带 local tile 语义；
- global entry 的 `talloc`/`tpop` 只获得 GM FIFO slot descriptor，数据写入和读取由显式
  `tstore`/`tload` 完成。

若 pipeline pass 只移动单条 `tpush`，会把 global entry 的 `talloc`、`tstore` 和 commit 拆开；
若只识别 `gm_slot_tensor`，又无法服务 A5 local-buffer pipe。因此 pass 必须首先把具体 op 归纳为逻辑事务。

### 6.2 Producer transaction

| Entry kind | 必须保持顺序的 producer group |
|---|---|
| tile | producer compute/write -> `tpush(tile)` |
| global | `talloc(entry)` -> 所有 `tstore(entry/subview)` -> `tpush(entry)` |

整个 group 作为 stage 的 commit 尾部移动。任何写 entry 的 op 都不能越过 `tpush`。

### 6.3 Consumer transaction

| Entry kind | 必须保持顺序的 consumer group |
|---|---|
| tile | `tpop -> tile` -> 所有读取/计算 -> `tfree` |
| global | `tpop -> entry` -> 所有 `tload(entry/subview)` -> 本地读取完成 -> `tfree(entry)` |

`tfree` 必须位于 entry 最后一次读取之后。global entry 的本地计算可以在 `tload` 后继续，但调度器不得把
`tfree(entry)` 移到尚未完成的 GM 读取之前。

### 6.4 建议的内部抽象

```text
LogicalPipeTransaction {
  pipe_key              // peer pair + id + direction + split
  logical_iteration     // S1 tile induction expression
  role                  // producer or consumer
  entry_kind            // tile or global
  transport_kind        // l2l, l2g2l, ...; scheduler 不分支
  acquire_ops[]         // talloc or tpop and address/view derivation
  payload_ops[]         // tstore/tload or tile uses
  commit_op             // tpush, producer only
  release_op            // tfree, consumer only
  capacity              // effective slot_num
}
```

调度 legality 使用 `pipe_key`、program order、SSA/memory dependence 和 capacity；`entry_kind` 只用于构造
事务边界，`transport_kind` 只交给目标 verifier/lowering。

## 7. A3/A5 统一与 Transport Adapter

### 7.1 统一层次

编译流程分为三层：

```text
CV pipeline scheduler
  -> entry transaction adapter
    -> target transport lowering
```

- scheduler 决定第几个逻辑 tile 的哪个 stage 先执行；
- entry adapter 保证 allocate/load/store/push/pop/free 不被拆散；
- transport lowering 决定数据通过 A5 local buffer、A2/A3 raw GM buffer 或 GM tensor entry 传输。

### 7.2 当前平台矩阵

| 平台 | Entry | Transport | Pipeline scheduler |
|---|---|---|---|
| A5 | tile | `initialize_l2l_pipe`，consumer local reserved buffer | 统一算法 |
| A2/A3 | tile | `initialize_l2g2l_pipe`，`gm_slot_buffer` + consumer local buffer | 统一算法 |
| A2/A3 | global | `initialize_l2g2l_pipe`，global-only `gm_slot_tensor` | 统一算法 |
| A5 | global | 当前不合法；A5 L2L 没有可赋给 GlobalTensor 的 GM slot | scheduler 可表达，目标 verifier 拒绝 |

这意味着“设计是否支持 pipeline”与“pipe 是否使用 `gm_slot_tensor` entry”无关。具体编译目标仍可通过
deployment policy 限制 `allowed_entry_kinds = {tile}`，从而生成完全不使用 GM tensor entry 的 A3/A5 kernel。

### 7.3 A5 local-buffer 配置

A5 使用 tile entry 时：

- frontend init 不提供 `gm_slot_buffer` 或 `gm_slot_tensor`；
- lower 为 `initialize_l2l_pipe`；
- consumer FIFO 由 `reserve_buffer`/`import_reserved_buffer` 配对；
- reserve 大小按 `slot_size * effective_slot_num` 计算；
- 不使用 `local_slot_num`。

### 7.4 A2/A3 非 GM tensor 配置

A2/A3 使用 tile entry 时：

- frontend init 提供 raw `gm_slot_buffer`，但不提供 `gm_slot_tensor`；
- lower 为带 GM 地址和 consumer local 地址的 `initialize_l2g2l_pipe`；
- `slot_num` 是 GM FIFO 深度；
- `local_slot_num` 只描述 consumer 本地 staging buffer 槽数，不改变 GM FIFO 深度；
- `local_slot_num` 可以小于 `slot_num`，由 `tpop`/`tfree` backpressure 复用本地槽。

因此，A3 与 A5 只在 transport 配置和容量资源计算上不同，stage 划分与 preload 算法完全一致。

## 8. Kernel Pair 与循环匹配

### 8.1 函数配对

Module pass 通过以下信息建立 C/V pair：

- `pto.kernel_kind = #pto.kernel_kind<cube|vector>`；
- `reserve_buffer`/`import_reserved_buffer` 的 `peer_func`；
- frontend pipe 的 `id`、direction 和 `split`；
- 相反方向的 push/pop/free 契约；
- 必要时由调用端或模块属性提供的显式 pair 标识。

所有 pipe 必须唯一配对。仅凭函数名或源码顺序猜测 peer 不合法。

### 8.2 循环配对

对每个外层 Q tile，C/V 两个函数中的 S1 循环必须满足：

- 具有相同的逻辑 trip count `N`，或能证明上下界/步长等价；
- 使用相同的 tile index 映射访问 `K[i]` 与 `V[i]`；
- pipe transaction 的生产和消费都是每逻辑迭代一次；
- 不存在提前退出、非结构化跳转或条件性遗漏 push/pop；
- loop-carried state 可以被分类为 prefix-only、suffix-only 或 crossing-cut。

首版要求规范化的 `scf.for`。动态 `N` 可以在后续阶段支持，但两侧必须共享等价的运行时表达式。

### 8.3 tile 0 特例

Flash Attention 常对 `i = 0` 使用初始化公式。pass 可选择：

1. 保留 `if i == 0`，让单循环调度继续工作；或
2. 静态 trip count 下 peel tile 0，再对剩余区间调度。

两种方式都必须确保 tile 0 只执行一次，且不把初始化状态错误地复制到 preload 的每个 tile。

## 9. 可配置 Preload 调度算法

### 9.1 单循环规范形式

为了保持“外层 Q + 内层 S1”两层结构，推荐以一个扩展的内层 schedule loop 表达 pipeline：

```text
Pe = min(P, N)

for t in [0, N + Pe):
  if t < N:
    Prefix(t)
  if t >= Pe:
    Suffix(t - Pe)
```

同一 `t` 中必须保持 `Prefix(t)` 在 `Suffix(t - Pe)` 之前。这一顺序允许 producer 先尝试填充下一项，
再消费较早的 suffix；容量达到上限时，已有 `tpush` backpressure 会自然限速。

### 9.2 Cube 侧改写

```text
Pe = min(P, N)

for t in [0, N + Pe):
  if t < N:
    C_QK(t)
  if t >= Pe:
    C_PV(t - Pe)
```

`C_QK(t)` 包含 K load、QK matmul 和 QK push；`C_PV(t - Pe)` 包含 P pop、对应 V load、PV
matmul 和 PV push。

### 9.3 Vector 侧改写

```text
Pe = min(P, N)

for t in [0, N + Pe):
  if t < N:
    V_P(t)
  if t >= Pe:
    V_O(t - Pe)
```

`V_P(t)` 包含 QK pop、softmax、P push 和 QK free；`V_O(t - Pe)` 包含 PV pop、O 更新和 PV free。

### 9.4 等价三段形式

单循环形式可证明等价于：

```text
for i in [0, Pe):
  Prefix(i)                       // prologue

for i in [0, N - Pe):
  Prefix(i + Pe)
  Suffix(i)                       // steady-state

for i in [N - Pe, N):
  Suffix(i)                       // epilogue
```

这里是三个顺序区间，不是三层嵌套循环。后端可以在静态 `N/P` 时把 guarded 单循环 peel 成三段以消除
分支，也可以直接保留单循环。算法正确性以单循环规范形式定义。

### 9.5 边界行为

| 条件 | 调度结果 |
|---|---|
| `P = 0` | 每个 `t` 执行 `Prefix(t); Suffix(t)`，等价于串行基线 |
| `N = 0` | 循环不执行，不发生 pipe transaction |
| `0 < N < P` | `Pe = N`，先执行全部 prefix，再执行全部 suffix |
| `P = 1` | 经典相邻 tile double-buffer 风格 |
| `P > 1` | crossing-cut 状态通常需要 `P + 1` 个版本；静态 `N <= P` 时可按 `N` 收缩 |

## 10. 不使用 ACK 的跨核协同

本设计不生成 ACK、phase counter 或“preload 完成”专用 pipe。

原因是算法只要求以下局部顺序：

- producer 在当前 entry 完整写入后 `tpush`；
- consumer 在使用前 `tpop`；
- consumer 最后一次读取后 `tfree`；
- 每个 kernel 内的 prefix/suffix 按改写后的程序顺序执行。

当一侧过快时，`tpush` 在 FIFO 满时阻塞，`tpop` 在 FIFO 空时阻塞。已有通信依赖即可把两个核连接成
一个有界数据流网络。

需要特别区分：不使用 ACK 时，Cube 可能在 Vector 完成第 `Pe - 1` 个 `V_P` 前就开始等待
`C_PV(0)`；这不违反算法。`C_PV(0)` 的 `tpop` 会等待 `V_P(0)`，而不是等待整个全局 preload
相位。如果未来要求“两个核都完成 P 个 prefix 后才允许任何 suffix”，那是额外的全局 barrier 语义，
不属于本文的性能 pipeline，也不能仅由现有单 entry FIFO 依赖表达。

## 11. Crossing-Cut 状态与 Multi-Buffer

### 11.1 为什么通常需要 `P + 1`

Vector steady-state 的顺序是：

```text
V_P(i + Pe)
V_O(i)
```

当 `Pe < N` 时，在第一次执行 `V_O(0)` 前，`V_P(0 ... Pe)` 的结果可能同时存活，因此从
`V_P(i)` 活到 `V_O(i)` 的每个本地值需要 `Pe + 1` 个版本。当 `Pe = N` 时不再执行同时间步的
新 prefix，最多只有 `N` 个版本。静态 shape 下精确的最大 live version 数为：

```text
B = 1                         if Pe = 0 and N > 0
B = min(N, Pe + 1)           if Pe > 0
```

动态 `N` 或不做 shape specialization 的首版实现可以保守取 `B = P + 1`。`P = 0` 时原始单 tile
已经足够，不生成 `alloc_multi_tile<count=1>`，因为 multi-buffer 容器只用于两个及以上物理槽。

典型 crossing-cut 状态包括：

- O rescale 使用的 `alpha[i]`；
- 当前 tile 的 normalization factor；
- `V_O(i)` 仍会读取的 row sum、max 派生值或 mask 派生值；
- 任何由 `V_P(i)` 写入、由 `V_O(i)` 读取的 local tile。

不能只为一个名为 `alpha` 的已知变量做特判。pass 应通过 SSA use-def 与 memory dependence 计算
完整 live-out 集合。

### 11.2 显式 slot 选择

对每个 crossing-cut local tile `state`，使用
[`ptoas-multi-buffer-explicit-design.md`](ptoas-multi-buffer-explicit-design.md) 定义的显式表达：

```text
state_mb = alloc_multi_tile<count = B>
producer_slot(i) = multi_tile_get state_mb[i mod B]
consumer_slot(i) = multi_tile_get state_mb[i mod B]
```

producer 与 consumer 必须使用相同的逻辑 `i` 计算 slot。slot 选择由 pipeline pass 显式生成，不能依赖
后续 pass 猜测 induction variable。

### 11.3 Loop-carried reduction 状态

running max、running sum 等 `V_P` 内部的递推状态仍按 `i = 0..N-1` 的顺序更新；pipeline 不并行执行
两个 `V_P`，因此这类 prefix-only loop-carried state 不需要复制。

O accumulator 仍按 `V_O(0)..V_O(N-1)` 的顺序更新；suffix-only accumulator 也不需要复制。
只有跨越 `V_P -> V_O` cut 的逐 tile 值需要 multi-buffer。

### 11.4 资源上限

当前 multi-buffer 设计的最大槽数为 16，因此要求：

```text
B <= 16
```

保守分配 `B = P + 1` 的实现等价于要求 `P + 1 <= 16`。超过上限时显式配置必须报错，不允许静默
减小 `P`。未来 auto-tuning 模式可以基于资源预算选择更小值。

## 12. Pipe 容量与资源约束

### 12.1 Hard constraint

`pipe_p` 位于 prefix/suffix cut 上。Vector prologue 在 Cube 必须消费 `P(0)` 前可能先提交
`P(0 ... Pe - 1)`，因此必须满足：

```text
effective_slot_num(pipe_p) >= max(Pe, 1)
```

静态 `N` 时可以使用 `Pe` 做精确检查；动态 `N` 或不做 shape specialization 时必须按最坏情况检查
`effective_slot_num(pipe_p) >= max(P, 1)`。若有效 preload 大于该容量，两个核可能分别阻塞在 QK
producer 和 P producer 上，Cube 又尚未进入 `C_PV` 消费 P，形成有界 FIFO 死锁。pass 必须在改写前
拒绝该配置。

### 12.2 其他 pipe

`pipe_qk` 和 `pipe_pv` 不直接跨 prefix/suffix cut，正确性最低容量为 1。为减少 steady-state 中 producer
立即阻塞，建议所有相关 pipe 满足：

```text
effective_slot_num >= min(P + 1, platform_limit)
```

这是性能建议，不是所有 pipe 的 correctness hard constraint。首版实现可以选择更保守的策略：要求三个
pipe 都至少为 `P`，以简化容量证明和诊断，但文档与诊断必须明确这是实现限制。

### 12.3 A5 资源计算

A5 tile-entry L2L pipe 的 `slot_num` 同时决定本地 FIFO 深度与 reserved buffer 大小：

```text
reserve_bytes = slot_size * effective_slot_num
```

pipeline pass 验证已有 init/reserve 合同，不在未授权情况下扩大本地内存。

### 12.4 A2/A3 资源计算

A2/A3 tile-entry L2G2L pipe 中：

- `slot_num` 决定 GM FIFO 深度，必须满足 pipeline capacity；
- `local_slot_num` 决定 consumer local staging 深度；
- 只要每次 `tpop -> use -> tfree` 事务完整，`local_slot_num` 可以小于 `P`；
- reserve 大小按 `slot_size * effective_local_slot_num` 计算。

A2/A3 global-only GM FIFO 中只检查 `slot_num`，不要求 local reserve/import contract。

## 13. 正确性与无死锁条件

### 13.1 迭代覆盖

单循环规范形式中：

- `Prefix(i)` 对且只对 `i in [0, N)` 执行一次；
- `Suffix(i)` 对且只对 `i in [0, N)` 执行一次；
- `Suffix(i)` 的 schedule time 是 `i + Pe`，所以不会早于同 kernel 的 `Prefix(i)`。

### 13.2 跨核 tile 配对

每条 logical pipe 保持 FIFO，且每个 stage 每逻辑迭代只 commit/acquire 一次。因此：

- `V_P(i)` pop 的是 `C_QK(i)` push 的 entry；
- `C_PV(i)` pop 的是 `V_P(i)` push 的 entry；
- `V_O(i)` pop 的是 `C_PV(i)` push 的 entry。

entry kind 改变事务内部形式，不改变该映射。

### 13.3 状态顺序

- `V_P` 的 loop-carried state 按递增 `i` 更新；
- `V_O` 的 O accumulator 按递增 `i` 更新；
- crossing-cut state 使用同一个 `i mod B` 版本；
- `K[i]` 与 `V[i]` 的索引不因 schedule time `t` 改变。

因此 pipeline 只改变不同 stage 间的重叠，不改变每条递推链的顺序。

### 13.4 无死锁前提

在以下条件成立时，改写不会引入新的 cycle：

1. 原始串行 C/V graph 可执行；
2. 每个 consumer transaction 最终执行匹配的 `tfree`；
3. `pipe_p.slot_num` 已覆盖所有运行时可能出现的 `max(Pe, 1)`；
4. 每个其他 pipe 至少有一个 slot；
5. stage 内不存在与 graph 方向相反的隐藏跨核同步；
6. C/V 两侧 trip count 与每迭代 transaction 次数匹配。

直观证明如下：

- Cube 的首个 `C_QK` 不依赖任何本轮 pop，可以启动数据流；
- `V_P` 由 QK push 解锁，并在 cut pipe 容量内完成 prologue；
- cut pipe 容量保证 Cube 到达首个 `C_PV` 前 Vector 不因 prologue commit 永久阻塞；
- steady-state 中任一满 FIFO 都由另一侧后续的 pop/free 释放，且该 consumer 不被同一个满 FIFO 之前的
  未完成 producer 永久挡住；
- 当 prefix 全部完成后，epilogue 不再产生新 entry，只排空 suffix。

实现应把该证明转化为结构化 legality checks，而不是依赖运行时超时发现死锁。

## 14. 编译器 Pass 设计

### 14.1 Pass 位置

需要新增 ModuleOp pass，例如：

```text
pto-cv-pipeline
```

推荐主流程顺序：

```text
PTOAssignDefaultFrontendPipeId
PTOCVPipeline                         // 新增，ModuleOp
PTOLowerFrontendPipeOps
PTOInferValidatePipeInit
LoweringSyncToPipe
InferPTOLayout
...
```

现有 `SerialFrontendPipeLoweringPass` 把 default-id materialization 与 frontend pipe lowering 放在同一个
串行 wrapper 中。实现本设计时应把 wrapper 拆成三个串行步骤，或在其中插入 ModuleOp 调度步骤，使
pipeline pass：

- 已能看到显式 pipe id；
- 仍能看到方向明确的 frontend push/pop/free 与 entry type；
- 运行在 layout、view-to-memref、memory planning 和 sync insertion 之前；
- 能同时查看 peer C/V 函数。

### 14.2 分析数据结构

```text
CVKernelPair {
  cube_func
  vector_func
  outer_loop_pairs[]
  logical_pipes[]
}

CVLoopPipelinePlan {
  cube_inner_loop
  vector_inner_loop
  trip_count
  requested_preload
  effective_preload
  stages { C_QK, V_P, C_PV, V_O }
  crossing_values[]
  transactions[]
  capacity_checks[]
}
```

分析与改写必须分离。只有完整 plan 通过全部 legality check 后，才能同时修改两个函数，避免只改写一侧。

### 14.3 Stage extraction

建议按以下顺序提取：

1. 以 `tpush`/`tpop`/`tfree` 和 pipe key 建立跨函数通信边；
2. 从每个 transaction 向前/向后闭包收集其必须伴随的 SSA producer/consumer；
3. 加入 memory dependence、layout/move 与 loop-carried dependence；
4. 验证四个 stage 对当前逻辑迭代各出现一次；
5. 验证 stage 间没有违反 cut 的反向 dependence；
6. 计算 `V_P -> V_O` crossing-cut live set；
7. 计算 pipe capacity 和 local memory 预算。

### 14.4 改写

对每个匹配的 inner-loop pair：

1. 物化 `Pe = min(P, N)`；静态 `N/P` 时常量折叠；
2. 计算 live version 数 `B`，并在 `B >= 2` 时为 crossing-cut local tile 生成
   `alloc_multi_tile<count=B>`；
3. 克隆一个 schedule loop，范围为 `[0, N + Pe)`；
4. 在 `t < N` 分支克隆 prefix stage，用 `i = t` 替换原 induction variable；
5. 在 `t >= Pe` 分支克隆 suffix stage，用 `i = t - Pe` 替换原 induction variable；
6. 将 crossing-cut use 改为相同逻辑 `i` 的 `multi_tile_get`；
7. 删除原 inner loop；
8. 对 Cube 和 Vector 的 plan 同时 commit；任一侧失败则不修改 IR。

### 14.5 Canonicalization

后续 canonicalization 可以：

- 在 `P = 0` 时消除恒真/恒假的 guard 和多余算术；
- 在静态 `N/P` 时 peel 为 branch-free prologue/steady/epilogue；
- 合并 `i mod 1`；
- 删除没有 crossing value 时的 multi-buffer scaffolding。

canonicalization 只能改变表达形式，不能改变逻辑 stage 距离。

## 15. 配置接口与诊断

### 15.1 编译选项

建议增加：

```text
--enable-cv-pipelining
--cv-preload-count=<non-negative integer>
```

默认值建议为：

- 未启用 `--enable-cv-pipelining`：不运行改写；
- 已启用但未指定 count：`P = 1`；
- 显式 `P = 0`：运行 legality 分析，但生成串行等价调度，便于测试。

### 15.2 IR 属性

为单独控制某个 kernel pair，可允许在 Cube/Vector 两个函数上同时设置：

```mlir
attributes {pto.cv_preload_count = 2 : i64}
```

规则：

- pair 两侧都出现时值必须相同；
- 只在一侧出现时，通过已验证的 peer pair 传播到另一侧；
- 函数属性优先于命令行默认值；
- 显式非法值报错，不静默 clamp；只有运行时 `N < P` 时使用 `Pe = min(P, N)`。

### 15.3 关键诊断

至少覆盖：

- 找不到唯一 C/V peer；
- 两侧 S1 trip count 不等价；
- 每迭代 pipe transaction 数不匹配；
- transaction 被条件分支拆分；
- `pipe_p.slot_num` 小于静态 `Pe` 或动态最坏情况 `P`；
- crossing-cut live version 数 `B > 16`；
- crossing-cut memory 无法证明 alias/last-use；
- stage 间存在反向 dependence；
- 目标不支持当前 entry/transport 组合。

最后一类错误来自 target legality，不应被误报成 pipeline scheduler 不支持该 entry。

## 16. 算法伪代码

```text
run(module, requestedP):
  materializeDefaultFrontendPipeIds(module)

  for pair in matchCVKernelPairs(module):
    plans = []

    for loopPair in matchS1Loops(pair):
      graph = buildCrossFunctionStageGraph(loopPair)
      stages = extractFAQKPVStages(graph)
      transactions = groupLogicalPipeTransactions(stages)
      crossing = computeCrossingCutState(stages.V_P, stages.V_O)

      P = resolveRequestedPreload(pair, requestedP)
      checkNonNegative(P)
      checkEquivalentTripCounts(loopPair)
      checkTransactionBalance(transactions)
      checkCutCapacity(graph.pipe_p, maxEffectivePreload(P, tripCountKnowledge))
      B = computeCrossingLiveVersions(P, tripCountKnowledge)
      checkMultiBufferLimit(crossing, B)
      checkNoReverseDependence(graph)

      plans.push(buildPlan(loopPair, stages, crossing, P))

    if every plan is legal:
      rewriteCubeAndVectorAtomically(plans)
    else:
      emitDiagnosticAndLeavePairUnchanged()
```

`groupLogicalPipeTransactions` 是 entry-specific adapter 的唯一入口；`buildPlan` 和
`rewriteCubeAndVectorAtomically` 不根据 tile/global 或 A3/A5 分支。

## 17. 示例调度

设 `N = 6`、`P = 2`，单循环时间步为 `t = 0..7`：

| `t` | Cube | Vector |
|---:|---|---|
| 0 | `C_QK(0)` | `V_P(0)`（等待 QK0） |
| 1 | `C_QK(1)` | `V_P(1)` |
| 2 | `C_QK(2); C_PV(0)` | `V_P(2); V_O(0)` |
| 3 | `C_QK(3); C_PV(1)` | `V_P(3); V_O(1)` |
| 4 | `C_QK(4); C_PV(2)` | `V_P(4); V_O(2)` |
| 5 | `C_QK(5); C_PV(3)` | `V_P(5); V_O(3)` |
| 6 | `C_PV(4)` | `V_O(4)` |
| 7 | `C_PV(5)` | `V_O(5)` |

表中同一行不表示两个核必须锁步，也不表示存在 phase barrier。每个核只遵守自己的程序顺序，跨核
ready/backpressure 完全由三条 pipe 的 push/pop/free 决定。

## 18. 验证计划

### 18.1 IR 回归测试

至少增加以下 lit 用例：

- `P = 0/1/2/4`；
- `N = 0/1`、`N < P`、`N = P`、`N > P`；
- tile 0 初始化分支；
- A5 tile-entry frontend IR；
- A2/A3 tile-entry raw GM buffer frontend IR；
- A2/A3 global-entry `gm_slot_tensor` frontend IR；
- entry 不同但 schedule skeleton 相同的对比检查；
- crossing-cut 值生成正确的 `B` 槽 multi-buffer 与 `i mod B`；
- 多 pipe id、DIR_BOTH 和显式 split；
- capacity、trip count、transaction balance 和 reverse-dependence 负例。

### 18.2 Lowering 检查

- A5 tile-entry 输出只含 L2L/local-buffer pipe，不含 GM tensor entry；
- A2/A3 tile-entry 输出含 raw GM slot buffer 与 local consumer buffer，不含 global entry；
- A2/A3 global-entry 保持 `talloc/tstore/tpush` 与 `tpop/tload/tfree` 事务顺序；
- 生成的 C++/VPTO 可由目标工具链编译；
- `P = 0` 与串行基线 IR/结果等价。

### 18.3 功能与性能验证

- 使用小 shape golden 覆盖 mask、非整 tile、不同 dtype；
- 比较 `P = 0` 与 `P > 0` 的数值结果；
- 在 CA model/NPU 上确认无死锁并统计 pipe stall；
- 分别测量 `P = 1..capacity`，验证最优值随 shape 与平台变化；
- 检查 A5 local memory、A3 GM FIFO 与 Vector multi-buffer 资源没有超预算。

## 19. 分阶段实现建议

### 阶段 1：Flash Attention 精确模式

- 只识别规范的四 stage FA graph；
- 只支持静态 `P` 和静态 S1 trip count；
- 只支持 tile entry；
- A5 走 L2L local buffer，A2/A3 走 raw GM buffer L2G2L；
- 使用单 schedule loop；
- 加入完整 capacity/multi-buffer 诊断。

### 阶段 2：Entry 无关

- 接入 `LogicalPipeTransaction` adapter；
- 支持 A2/A3 global entry；
- 用同一组 schedule FileCheck 对比 tile/global 两种输入。

### 阶段 3：通用 CV pipeline

- 从固定四 stage 识别扩展为合法 stage graph cut；
- 支持动态 `N` 与运行时 `Pe`；
- 支持多个独立 inner-loop pair；
- 加入基于资源和延迟模型的 auto `P`。

阶段划分允许首个实现满足“不使用 GM tensor entry 的 A3/A5 统一 kernel”，同时保证算法接口不会把后续
global entry 支持锁死。

## 20. 设计不变量汇总

实现和 review 必须保持以下不变量：

1. 数据遍历仍只有外层 Q/S0 tile 与内层 K/V S1 tile。
2. preload stage 包含 K load、QK matmul、QK push、softmax 和 P push，不是单独 K load。
3. V load 与 PV matmul 属于 suffix，并使用与 P 相同的逻辑 tile index。
4. 不生成 ACK；跨核只使用已有 FIFO 事务。
5. `P` 是调度距离，不是全局 phase barrier。
6. entry transaction 不能被拆分或跨 commit/release 重排。
7. scheduler 不根据 `gm_slot_tensor`、A3 或 A5 选择不同算法。
8. `pipe_p.slot_num` 覆盖最大有效 preload，且 crossing-cut local state 使用精确或保守的 `B` 个版本。
9. 两个函数必须在完整 plan 合法后原子式改写。
10. target verifier 可以拒绝某个 entry/transport 组合，但不能改变已定义的 schedule 语义。
