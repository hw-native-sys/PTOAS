# PTOAS C/V 分离式 Kernel 可配置 Preload Pipeline 设计

## 1. 文档范围

本文定义 PTOAS 对 Cube/Vector（下文简称 C/V）分离式 kernel 进行跨核流水调度的算法。
设计以 Flash Attention 为主要输入形态，同时把调度规则抽象为可复用的 stage graph，覆盖：

- 在已经完成 C/V 分离表达的两个 `func.func` 之间匹配通信 pipe；
- 保持外层 Q/S0 tile 与内层 K/V 的 S1 tile 两层数据遍历不变；
- 以可配置的 `preload_count = P` 改写 S1 内层循环；
- preload 一个完整的 QK/softmax 前缀，而不是只提前加载 K；
- 使用现有 `tpush` / `tpop` / `tfree` 的阻塞与 FIFO 语义完成跨核衔接，不引入 ACK；
- 复用 tpipe 已有 FIFO slot 抽象保存跨 C/V 传输的 payload，不对 pipe entry 再做一次
  `alloc_multi_tile` 转换；
- 对没有进入 tpipe、但因调度而跨迭代存活的 Cube/Vector 核内 tile 做 multi-buffer；
- 区分 Cube 的 L1 `mat`、L0A/L0B `left/right` 和 L0C `acc`，根据实际重叠目标分别决定是否多份化；
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
6. 跨核 payload 的多槽语义完全由 tpipe 的 `slot_num`、reserved buffer 和
   `tpush`/`tpop`/`tfree` 表达；CV pass 不为 QK/P/PV payload 生成 local multi-buffer。
7. 对核内 buffer 分成两类：给定 `P` 的 correctness-required buffer 和只增加 engine overlap 的
   performance-optional buffer。前者不可静默关闭，后者可以独立回退而不取消 C/V pipeline。
8. CV/local pipeline pass 在 PlanMemory 前只生成逻辑 slot 表达或稳定标注；PlanMemory 检查容量并分配
   物理槽，Sync 保留 slot identity，最后由 `PTOResolveBufferSelect` 物化地址选择。
9. A5 tile entry lower 为本地 L2L pipe；A2/A3 tile entry lower 为 raw GM buffer 的 L2G2L
   pipe；A2/A3 与 A5 的 global entry 都 lower 为 `gm_slot_tensor` 的 global-only L2G2L GM FIFO。
10. entry/transport 合法性由已有 verifier 和目标 lowering 决定，pipeline scheduler 不为不同平台复制算法。

因此，同一份 pipeline 结果可以服务以下部署策略：

- A5：只允许 tile entry，使用 local buffer TPipe，不出现 GM tensor entry；
- A5：允许 global entry，使用 `gm_slot_tensor`；单向 pipe 输出 `DIR_C2V_GM`/`DIR_V2C_GM`；
- A2/A3：只允许 tile entry，使用 `gm_slot_buffer + consumer local buffer`；
- A2/A3：允许 global entry，使用 `gm_slot_tensor`；
- 后续平台：增加新的 transport adapter，而不修改 preload 调度。

## 3. 目标与非目标

### 3.1 目标

- `P` 可由编译选项或函数属性显式指定。
- `P = 0` 保留原始逐 tile 串行顺序，便于 A/B 对比与回退。
- `P > 0` 时，QK matmul、softmax 和对应的 push 都属于 preload 范围。
- K 与 V 仍按相同逻辑 tile `i` 配对，不改变注意力计算语义。
- 将 C/V stage preload 与 Cube/Vector 核内 operand software pipeline 解耦；两者可以分别开关和降级。
- 覆盖 Cube 的 GM->L1、L1->L0A/L0B、Cube->L0C/FIX 数据路径，而不是只核算 L1。
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
- 本阶段不承诺仅提高 `P` 就必然得到 K/V MTE2、MTE1 与 Cube 指令重叠；该收益还依赖核内
  operand buffer、L0 生命周期与同步是否合法。

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

此 cut 的直接通信边是 `pipe_p`。QK、P、PV 等跨 C/V payload 的多槽存储已经由各自 tpipe 的 FIFO
抽象负责，不再转换成 `alloc_multi_tile`。

所有从 `V_P(i)` 活到 `V_O(i)`、且没有随 pipe payload 传输的 Vector 本地 SSA 值或内存状态，仍是
Vector 核内的 cut edge。若 `P > 0` 使多个逻辑 tile 的这些状态同时存活，则必须进行 multi-buffer
扩展或选择等价的传输/重算方案。

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
- transport lowering 决定 A5 tile entry 使用 L2L local buffer、A2/A3 tile entry 使用 raw GM buffer，
  或 A2/A3/A5 global entry 使用 `gm_slot_tensor` 的 L2G2L GM FIFO。

### 7.2 当前平台矩阵

| 平台 | Entry | Transport | Pipeline scheduler |
|---|---|---|---|
| A5 | tile | `initialize_l2l_pipe`，consumer local reserved buffer | 统一算法 |
| A2/A3 | tile | `initialize_l2g2l_pipe`，`gm_slot_buffer` + consumer local buffer | 统一算法 |
| A2/A3 | global | `initialize_l2g2l_pipe`，global-only `gm_slot_tensor` | 统一算法 |
| A5 | global | `initialize_l2g2l_pipe`，global-only `gm_slot_tensor`；单向 EmitC 使用 `DIR_C2V_GM`/`DIR_V2C_GM` | 统一算法 |

frontend init 存在 `gm_slot_tensor` 时，lowering 优先选择 global-only L2G2L 路径，不会进入 A5 tile-entry
的 L2L 分支。因此，“A5 只允许 tile entry”只能是 deployment policy，而不是 target capability。

这意味着“设计是否支持 pipeline”与“pipe 是否使用 `gm_slot_tensor` entry”无关。具体部署仍可通过
`allowed_entry_kinds = {tile}` 生成完全不使用 GM tensor entry 的 A3/A5 kernel。

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
| `P > 1` | Vector 核内 crossing-cut 状态通常需要 `P + 1` 个版本；静态 `N <= P` 时可按 `N` 收缩 |

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

## 11. 核内 Multi-Buffer 与 Operand Software Pipeline

### 11.1 与 tpipe FIFO 的职责边界

multi-buffer 必须先按所有权分为两类：

1. **跨 C/V payload**：QK、P、PV 等通过 tpipe 发送的数据。它们的并发 entry 数已经由
   `slot_num`/`local_slot_num`、reserved buffer 与 `tpush`/`tpop`/`tfree` 表达，CV pass 不再为其生成
   `alloc_multi_tile`。
2. **核内 local tile**：没有进入 tpipe、但因跨 stage 或异步 engine 重叠而同时存活的 Cube/Vector
   本地值。这些值才使用 tile-native multi-buffer 表达。

因此，§12 的 pipe FIFO 槽和本节的 local tile 槽属于两套资源。即使它们恰好具有相同深度，也不能
互相替代或重复计数。

### 11.2 Vector 核内 crossing-cut 状态

Vector steady-state 的顺序是：

```text
V_P(i + Pe)
V_O(i)
```

当 `Pe < N` 时，在第一次执行 `V_O(0)` 前，`V_P(0 ... Pe)` 的结果可能同时存活。因此，从
`V_P(i)` 活到 `V_O(i)`、且没有放入 pipe payload 的每个本地值需要 `Pe + 1` 个版本。当 `Pe = N`
时不再执行同时间步的新 prefix，最多只有 `N` 个版本。静态 shape 下精确的最大 live version 数为：

```text
Bcv = 1                         if Pe = 0 and N > 0
Bcv = min(N, Pe + 1)           if Pe > 0
```

动态 `N` 或不做 shape specialization 的首版实现可以保守取 `Bcv = P + 1`。典型状态包括：

- O rescale 使用的 `alpha[i]`；
- 当前 tile 的 normalization factor；
- `V_O(i)` 仍会读取的 row sum、max 派生值或 mask 派生值；
- 任何由 `V_P(i)` 写入、由 `V_O(i)` 读取的 local tile。

这是给定 `P` 的 correctness-required multi-buffer。不能只为名为 `alpha` 的变量做特判，pass 应通过
SSA use-def 与 memory dependence 计算完整 live-out 集合。running max/sum 等 prefix-only 递推状态仍按
递增 `i` 更新，O accumulator 仍按 `V_O(0)..V_O(N-1)` 更新；不跨 cut 的递推状态不因 `P` 自动复制。

### 11.3 Cube 本地存储层次

Cube operand 路径不只有 L1。PTO tile 地址空间与典型 FA 数据对应如下：

| 数据层次 | PTO 地址空间 | 典型数据 |
|---|---|---|
| L1 | `mat` | Q、K、V，以及必要时从 pipe entry 复制出的 P tile |
| L0A | `left` | QK 的 Q、PV 的 P |
| L0B | `right` | QK 的 K、PV 的 V |
| L0C | `acc` | QK/PV accumulator |
| UB | `vec` | softmax、归一化和 O 更新的 Vector tile |

local pipeline analysis 必须在 layout 和 tile op 展开后识别真实的数据搬运链，不能把
`GM -> L1 -> L0 -> Cube -> L0C/FIX` 简化成只有一份 K/V L1 buffer。

地址空间不能单独决定 buffer 所有权。A5 tile-entry 或 A2/A3 tile-entry L2G2L 中，`tpop` 返回的 P
可能物理位于 Cube L1，QK/PV consumer entry 可能物理位于 Vector UB，但这些区域仍是 tpipe 的 local
consumer slot/reserved buffer。除非数据在 `tfree` 前被复制到另一个 local tile，否则不能再次把它们
转换成核内 `alloc_multi_tile`。

### 11.4 不同 overlap 对 buffer 的要求

是否复制某级 buffer 由目标 overlap 决定：

| 目标 overlap | 需要检查或多份化的 local buffer |
|---|---|
| `GM->L1 K(i+1) || QK(i)` | K 的 `mat` |
| `L1->L0B K(i+1) || QK(i)` | K 的 `right`，以及相应 MTE1/Cube 事件依赖 |
| `tpop P(i+1) || PV(i)` | P 的 consumer local slot，由 tpipe `local_slot_num`/reserved buffer 管理 |
| `GM->L1 V(i+1) || PV(i)` | V 的 `mat` |
| `L1->L0A/L0B P/V(i+1) || PV(i)` | P 的 `left`、V 的 `right` |
| `FIX(i) || Cube(i+1)` | 必要时复制或轮换 `acc`，并证明上一结果已被 drain |
| Vector load/compute/store overlap | 不属于 pipe entry backing 的 `vec` scratch/output tile |

不要求把表中所有 buffer 无条件改为双缓冲。例如，只要求隐藏 MTE2 延迟时，L1 ping-pong 可能已经
足够；只有把下一次 MTE1 也提前到当前 Cube 计算期间时，才需要额外解决 L0A/L0B 的复用冲突。

参考手写
[`fa_performance_kernel.cpp`](https://github.com/hw-native-sys/pto-isa/blob/main/kernels/manual/common/flash_atten/fa_performance_kernel.cpp)，
`qkPreloadNum` 负责跨 C/V 的 stage distance，而 K/P/V 使用两份 L1 `TileType::Mat`，Vector 输入、exp
和输出也使用 ping-pong，QK/PV accumulator 则轮换两个 L0C 地址。
`pMatTile[0]` 同时作为 P pipe 的 consumer local base，`qkVecTile[0]`/`pvVecTile[0]` 也分别作为
QK/PV pipe 的 Vector consumer local base；这些数组中的 pipe backing 部分在 PTOAS 中应归入 tpipe
容量和 reserved-buffer 规划，而不是再生成第二套 local multi-buffer。
其中 L1->L0 搬运封装在 `pto_macro_matmul` 内；PTOAS 若把它展开为显式 `tmov + tmatmul`，必须得到
等价的 L0 生命周期和事件依赖，而不是只复制 L1 后假设 overlap 自动成立。

### 11.5 逻辑 slot 表达与物理地址选择

对已选中的核内 local tile `state`，使用
[`ptoas-multi-buffer-explicit-design.md`](ptoas-multi-buffer-explicit-design.md) 定义的 tile-native 表达：

```text
state_mb = alloc_multi_tile<count = B>
producer_slot(i) = multi_tile_get state_mb[i mod B]
consumer_slot(i) = multi_tile_get state_mb[i mod B]
```

producer 与 consumer 必须使用同一逻辑迭代计算 slot。这里的 `multi_tile_get` 是逻辑槽选择，不是最终
地址 select：

1. pipeline/bufferize pass 在 PlanMemory 前生成 `alloc_multi_tile`、`multi_tile_get` 或等价的稳定标注；
2. PlanMemory 识别 slot 数和地址空间，检查容量并写入 `pto.multi_buffer_addrs`；
3. Sync pass 在 `multi_tile_get` 仍存在时分析每次 slot 访问；
4. `PTOResolveBufferSelect` 最后才把动态 slot 物化为地址 `arith.select` 和 addressed `alloc_tile`。

CV/local pipeline pass 不直接生成地址 select，PlanMemory 也不负责生成 select。

### 11.6 Hard requirement 与 performance option

对给定 `P`，buffer requirement 分为：

| Requirement | 性质 | 分配失败时的行为 |
|---|---|---|
| `pipe_p` 等 correctness 所需 FIFO 容量 | hard | 降低 `P` 或拒绝 CV pipeline |
| Vector `V_P(i)->V_O(i)` 核内 retained state | hard | 降低 `P`、改为传输/重算，或拒绝 CV pipeline |
| Cube K/P/V L1/L0 operand ping-pong | soft | 关闭对应核内 overlap，C/V pipeline 可保留 |
| Vector scratch ping-pong | soft | 关闭对应 Vector 核内 overlap，C/V pipeline 可保留 |
| L0C/FIX ping-pong | soft，除非调度已依赖它 | 回退到串行 drain；若已提交重排则必须回滚该重排 |

`alloc_multi_tile<count=N>` 表示一个已经选定的硬槽数；当前 PlanMemory 只负责分配或报告 overflow，不能
把 N 静默降成 1。若要支持资源自适应，必须在最终调度提交前比较候选配置，或在原始 ModuleOp 的 clone
上执行“改写 + PlanMemory”试跑后提交成功候选。不能先生成依赖 N 个槽的 schedule，再在 PlanMemory
之后单独关闭 multi-buffer。

首版显式 `P` 模式建议保持严格语义：hard requirement 分配失败时报错；soft local pipeline 可以独立
关闭。后续 auto/fallback 模式可以按“先减少 L0、再减少 L1/UB scratch、最后减少 P”的策略搜索，但每个
候选都必须重新完成 schedule、liveness 和 PlanMemory 验证。

### 11.7 资源上限

当前 tile-native multi-buffer 的最大槽数为 16，因此每个核内 local multi-buffer 都要求：

```text
2 <= B <= 16
```

`B = 1` 使用普通 `alloc_tile`。对 Vector hard crossing state，保守分配 `Bcv = P + 1` 等价于要求
`P + 1 <= 16`；超过上限时，严格模式必须报错，不能在保持相同 schedule 的同时静默减小 `P`。

## 12. tpipe 容量与 Reserved Buffer 约束

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

pipeline pass 验证已有 init/reserve 合同，并把给定 `P` 所需的最小容量记录为 PlanMemory 可见的
requirement，不在未授权情况下扩大本地内存。PlanMemory/`PTOResolveReservedBuffers` 负责确认实际
reserved range 可分配；该 range 是 tpipe FIFO 存储，不是 K/V/P 的 L1/L0 operand buffer。

### 12.4 A2/A3 资源计算

A2/A3 tile-entry L2G2L pipe 中：

- `slot_num` 决定 GM FIFO 深度，必须满足 pipeline capacity；
- `local_slot_num` 决定 consumer local staging 深度；
- 只要每次 `tpop -> use -> tfree` 事务完整，`local_slot_num` 可以小于 `P`；
- reserve 大小按 `slot_size * effective_local_slot_num` 计算。

A2/A3 和 A5 的 global-only GM FIFO 中只检查 `slot_num`，不要求 local reserve/import contract。

### 12.5 与核内 multi-buffer 的联合预算

PlanMemory 必须在同一目标资源模型中同时看到：

- tpipe consumer reserved buffer；
- Vector `vec` hard crossing state；
- Cube `mat`、`left`、`right`、`acc` local buffer；
- Vector optional scratch/output ping-pong。

但它们保持不同的逻辑所有权。PlanMemory 可以让生命周期不重叠的 local buffer 复用地址，不能把 tpipe
slot 当作任意 operand tile 的一个版本，也不能通过减小 hard slot 数来消除 overflow。PlanMemory 的输出
是容量成功/失败与物理地址列表，不包含运行时 slot select 代码。

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
- 没有进入 tpipe 的 Vector crossing-cut state 使用同一个 `i mod Bcv` 版本；
- QK/P/PV payload 只使用 tpipe entry 生命周期，不额外绑定 local multi-buffer 版本；
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

### 13.5 核内 engine overlap 正确性

核内 software pipeline 还必须独立证明：

1. 下一次 MTE2 写入的 L1 slot 不再被上一轮 MTE1 读取；
2. 下一次 MTE1 写入的 L0A/L0B slot 不再被上一轮 Cube 指令读取；
3. 下一次 Cube 写入的 L0C slot 已完成上一轮 FIX/pipe producer drain；
4. Vector 下一轮写入的 UB slot 已完成上一轮 Vector/MTE3 读取；
5. 动态 `i mod B` 无法静态判定互异时，Sync 采用保守依赖而不是假定无 alias。

这些条件由 local pipeline 与 Sync pass 联合保证，不属于 tpipe backpressure。tpipe 只能保护 entry 的
生产消费，不能保护任意 L1/L0/UB 地址的提前复用。

## 14. 编译器 Pass 设计

### 14.1 Pass 位置

建议把跨核调度、核内调度和物理地址选择拆成三个层次：

```text
PTOCVPipeline          // ModuleOp：C/V stage schedule 与 hard state 标注
PTOLocalPipeline       // FuncOp：Cube/Vector 核内 operand software pipeline
PTOPipelineBufferize   // 逻辑 local slot 表达；不生成物理 select
```

推荐主流程顺序：

```text
PTOAssignDefaultFrontendPipeId
PTOLowerFrontendPipeOps
PTOInferValidatePipeInit
PTOCVPipeline                         // ModuleOp，跨函数调度
LoweringSyncToPipe
InferPTOLayout
FusionPlan / OpScheduling
PTORematerializeFixpipeVectorQuant
PTOLocalPipeline                      // 最后一个可能重排 MTE/Cube/Vector 的 pass
PTOPipelineBufferize                  // alloc_multi_tile/multi_tile_get
PTOPlanMemory                         // 分配地址，写 multi_buffer_addrs
PTOResolveReservedBuffers
PTORemoveIdentityTMov
PTOInsertSync / PTOGraphSyncSolver
PTOResolveBufferSelect                // 最后物化地址 select
```

`PTOCVPipeline` 放在 frontend pipe lowering 和 init validation 之后，原因是此时已具有统一的
`talloc`/`tpush`/`tpop`/`tfree` 与已解析 capacity，同时 lowering 必须保留 frontend id、direction、split
和 entry kind 等配对元数据。该 pass 仍需运行在 layout、memory planning 和 sync insertion 之前，并能
同时查看 peer C/V 函数。

`PTOLocalPipeline` 必须位于最后一个可能重排 load/move/matmul/vector op 的 pass 之后，否则后续 scheduling
可能破坏其 look-ahead 距离。它仍必须位于 PlanMemory 前，因为 PlanMemory 需要看到最终 live range、
地址空间和逻辑 slot 数。

Sync 必须位于 `PTOResolveBufferSelect` 之前，使依赖分析仍能追踪 `multi_tile_get` 的 slot identity。把动态
slot 过早 lowering 成 N 路 `arith.select` 会丢失精确 alias 信息并产生不必要同步。

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
  vector_crossing_values[]
  transactions[]
  capacity_checks[]
  hard_buffer_requirements[]
}

LocalPipelinePlan {
  function
  engine_stages[]              // MTE2, MTE1, Cube, FIX, Vector, MTE3
  buffer_requirements[]
  schedule_rewrites[]
}

LocalBufferRequirement {
  role                         // vector_crossing, k_l1, k_l0b, p_l0a, ...
  memory_space                 // vec, mat, left, right, acc
  min_count
  preferred_count
  slot_expr                    // logical i -> slot
  requirement_kind             // hard or soft
  coupled_schedule_id
}
```

分析与改写必须分离。只有 C/V pair 的完整 plan 通过全部 legality check 后，才能同时修改两个函数，避免
只改写一侧。每个 local buffer requirement 必须绑定产生它的 schedule；关闭一个 soft requirement 时，
必须同时撤销对应的 look-ahead 重排，不能只把槽数改成 1。

### 14.3 Stage extraction

建议按以下顺序提取：

1. 以 `tpush`/`tpop`/`tfree` 和 pipe key 建立跨函数通信边；
2. 从每个 transaction 向前/向后闭包收集其必须伴随的 SSA producer/consumer；
3. 加入 memory dependence、layout/move 与 loop-carried dependence；
4. 验证四个 stage 对当前逻辑迭代各出现一次；
5. 验证 stage 间没有违反 cut 的反向 dependence；
6. 计算没有进入 tpipe 的 `V_P -> V_O` Vector crossing-cut live set；
7. 计算 pipe capacity requirement；
8. 为 hard crossing state 记录 `Bcv`，但不为 QK/P/PV pipe payload 生成 local multi-buffer；
9. 给 stage 与逻辑迭代附加稳定 metadata，供后续 local pipeline 在 layout/scheduling 后恢复其来源。

### 14.4 C/V 调度改写

对每个匹配的 inner-loop pair：

1. 物化 `Pe = min(P, N)`；静态 `N/P` 时常量折叠；
2. 计算 Vector hard live version 数 `Bcv`，为这些值记录 hard local buffer requirement；
3. 克隆一个 schedule loop，范围为 `[0, N + Pe)`；
4. 在 `t < N` 分支克隆 prefix stage，用 `i = t` 替换原 induction variable；
5. 在 `t >= Pe` 分支克隆 suffix stage，用 `i = t - Pe` 替换原 induction variable；
6. 保持 QK/P/PV 的 tpipe transaction 完整，不为 entry 额外创建 multi-buffer；
7. 删除原 inner loop；
8. 对 Cube 和 Vector 的 plan 同时 commit；任一侧失败则不修改 IR。

### 14.5 核内 software pipeline 与 bufferize

`PTOLocalPipeline` 在最终 tile layout 和 op schedule 上分析每条异步 engine 链：

1. 将 Cube stage 细分为 GM->L1 load、L1->L0 move、matmul、L0C/FIX drain；
2. 将 Vector stage 细分为 load、Vector compute、store/pipe drain；
3. 只在存在合法 look-ahead 时生成对应 local buffer requirement；
4. 为 `K/P/V` L1、必要的 L0A/L0B、L0C 和 Vector scratch 选择独立 depth；
5. 插入下一逻辑迭代的 load/move，并保持同一 operand 的 tile index 不变；
6. 为每项重排记录其 hard/soft 属性和 schedule coupling。

`PTOPipelineBufferize` 消费 C/V hard state 与核内 local requirement：

```text
selected_count == 1:
  保持 alloc_tile

selected_count >= 2:
  alloc_tile -> alloc_multi_tile<count=selected_count>
  each use  -> multi_tile_get[logical_iteration mod selected_count]
```

该 pass 只生成逻辑 slot，不生成物理地址 select。PlanMemory 必须原生识别这些 op，在 `mat`、`left`、
`right`、`acc`、`vec` 各自容量内分配槽并附加 `pto.multi_buffer_addrs`。

### 14.6 资源可行性、回退与 PlanMemory

PlanMemory 是最终物理容量权威，但不能在已提交的 schedule 下静默关闭 multi-buffer：

- hard requirement 失败意味着当前 `P` 不合法；严格模式报错；
- soft requirement 失败只允许在同时撤销对应核内 schedule 后回退；
- `PTOResolveBufferSelect` 只消费已成功规划的地址，不参与容量决策。

首版可以要求用户显式选择 local buffer depth，并让 PlanMemory 对固定配置给出确定性 overflow 诊断。若实现
自动回退，推荐使用候选事务：

```text
for candidate in orderedCandidates(requestedP, localDepths):
  trial = clone(originalModule)
  rewriteCVAndLocalPipeline(trial, candidate)
  materializeLogicalSlots(trial, candidate)
  if PlanMemory(trial) succeeds:
    commit(trial)
    break
```

候选顺序通常先撤销 optional L0 look-ahead，再减少 optional L1/UB scratch，最后才降低 `P`。每次 retry
都从未改写 IR 开始，避免复用上一候选的 schedule 或 liveness。成功候选已经具有最终 schedule 和逻辑
槽；随后才运行正式 Sync 与 `PTOResolveBufferSelect`。

另一种实现是在 PlanMemory 中加入只读 feasibility API，让 pipeline pass 在提交前查询候选。无论采用
哪种实现，不能在 PlanMemory 后再新增会改变 live range 的 pipeline schedule；否则必须重新运行完整
PlanMemory。

### 14.7 Canonicalization

后续 canonicalization 可以：

- 在 `P = 0` 时消除恒真/恒假的 guard 和多余算术；
- 在静态 `N/P` 时 peel 为 branch-free prologue/steady/epilogue；
- 合并 `i mod 1`；
- 删除没有核内 crossing value 或没有选中 local overlap 时的 multi-buffer scaffolding。

canonicalization 只能改变表达形式，不能改变逻辑 stage 距离。

## 15. 配置接口与诊断

### 15.1 编译选项

建议增加：

```text
--enable-cv-pipelining
--cv-preload-count=<non-negative integer>
--local-pipeline-mode=<off|explicit|auto>
--cube-l1-buffer-count=<1..16>
--cube-l0-buffer-count=<1..16>
--cube-acc-buffer-count=<1..16>
--vector-scratch-buffer-count=<1..16>
--cv-pipeline-resource-policy=<strict|fallback>
```

默认值建议为：

- 未启用 `--enable-cv-pipelining`：不运行改写；
- 已启用但未指定 count：`P = 1`；
- 显式 `P = 0`：运行 legality 分析，但生成串行等价调度，便于测试。
- `local-pipeline-mode=off`：仍允许 C/V pipeline，只关闭核内 operand/scratch look-ahead；
- `resource-policy=strict`：显式 `P` 的 hard requirement 分配失败时报错；
- `resource-policy=fallback`：允许基于完整候选 retry 降低 local depth，必要时再降低 `P`。

`P` 与 local buffer depth 是独立参数。典型性能配置可以是 `P > 1`，而 K/P/V operand 仍只使用两份
ping-pong；不能把 `Bcv=P+1` 无条件套到所有 Cube L1/L0 buffer。

### 15.2 IR 属性

为单独控制某个 kernel pair，可允许在 Cube/Vector 两个函数上同时设置：

```mlir
attributes {pto.cv_preload_count = 2 : i64}
```

可选的核内策略属性示例：

```mlir
attributes {
  pto.local_pipeline = "explicit",
  pto.cube_l1_buffer_count = 2 : i64,
  pto.cube_l0_buffer_count = 1 : i64,
  pto.cube_acc_buffer_count = 2 : i64,
  pto.vector_scratch_buffer_count = 2 : i64
}
```

规则：

- pair 两侧都出现时值必须相同；
- 只在一侧出现时，通过已验证的 peer pair 传播到另一侧；
- 函数属性优先于命令行默认值；
- strict 策略下显式非法值报错，不静默 clamp；只有运行时 `N < P` 时使用 `Pe = min(P, N)`；
- fallback 策略必须报告最终选中的 `effective_preload` 和各 local depth，不能静默改变生成配置。

### 15.3 关键诊断

至少覆盖：

- 找不到唯一 C/V peer；
- 两侧 S1 trip count 不等价；
- 每迭代 pipe transaction 数不匹配；
- transaction 被条件分支拆分；
- `pipe_p.slot_num` 小于静态 `Pe` 或动态最坏情况 `P`；
- Vector hard crossing-cut live version 数 `Bcv > 16`；
- crossing-cut memory 无法证明 alias/last-use；
- `mat`、`left`、`right`、`acc` 或 `vec` 的 multi-buffer 容量 overflow；
- local buffer requirement 已被降低，但与其绑定的 look-ahead schedule 没有回滚；
- PlanMemory 后仍有 pass 尝试改变 local buffer live range 或 slot count；
- stage 间存在反向 dependence；
- 目标不支持当前 entry/transport 组合。

最后一类错误来自 target legality，不应被误报成 pipeline scheduler 不支持该 entry。
A5 global entry 是已有 lowering 和 EmitC 支持的合法组合，不能作为该诊断的触发案例；该诊断只覆盖
实际 verifier/lowering 明确拒绝的 entry、direction、shape 或 transport 组合。

## 16. 算法伪代码

```text
analyzeCV(module, requestedP):
  plans = []
  for pair in matchCVKernelPairs(module):
    for loopPair in matchS1Loops(pair):
      graph = buildCrossFunctionStageGraph(loopPair)
      stages = extractFAQKPVStages(graph)
      transactions = groupLogicalPipeTransactions(stages)
      vectorState = computeLocalCrossingState(stages.V_P, stages.V_O)

      P = resolveRequestedPreload(pair, requestedP)
      checkNonNegative(P)
      checkEquivalentTripCounts(loopPair)
      checkTransactionBalance(transactions)
      checkCutCapacity(graph.pipe_p, maxEffectivePreload(P, tripCountKnowledge))
      Bcv = computeCrossingLiveVersions(P, tripCountKnowledge)
      checkMultiBufferLimit(vectorState, Bcv)
      checkNoReverseDependence(graph)

      addHardRequirements(vectorState, Bcv)
      addPipeCapacityRequirements(transactions, P)
      plans.push(buildCVPlan(loopPair, stages, transactions, P))
  return plans

buildCandidate(originalModule, cvPlans, localConfig):
  trial = clone(originalModule)
  rewriteCubeAndVectorAtomically(trial, cvPlans)
  localPlan = analyzeFinalEngineSchedule(trial, localConfig)
  rewriteLocalLookAhead(trial, localPlan)
  materializeLogicalLocalSlots(trial, collectHardRequirements(cvPlans), localPlan)
  return trial

run(module, requestedP, resourcePolicy):
  lowerAndValidateFrontendPipes(module)
  cvPlans = analyzeCV(module, requestedP)

  for config in enumerateCandidates(cvPlans, resourcePolicy):
    trial = buildCandidate(module, cvPlans, config)
    if planMemory(trial) succeeds:
      replaceModule(module, trial)
      runSyncWithLogicalSlotIdentity(module)
      resolveBufferSelect(module)
      return success

  emitResourceDiagnostic()
  return failure
```

`groupLogicalPipeTransactions` 是 entry-specific adapter 的唯一入口；`buildCVPlan` 和
`rewriteCubeAndVectorAtomically` 不根据 tile/global 或 A3/A5 分支。`materializeLogicalLocalSlots` 只处理
Cube/Vector 核内 tile，不处理 tpipe entry；`resolveBufferSelect` 只在 PlanMemory 和 Sync 成功后运行。

## 17. 示例调度

### 17.1 跨 C/V stage 调度

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

### 17.2 Cube 核内 operand 调度

假设 K/V 核内 L1 depth 为 2、P tpipe consumer local depth 为 2，且 L0 lowering 能证明相邻 operand
slot 安全，则 steady-state 还可以形成：

```text
MTE2 K(i + 1)    || MTE1/Cube QK(i)
TPop P(i + 1) + MTE2 V(i + 1) || MTE1/Cube PV(i)
FIX result(i)    || Cube compute(i + 1)    // 仅当 acc ping-pong 已启用
```

若 PlanMemory 只能支持单份 K/V L1 或单个 P consumer local slot，则撤销对应 look-ahead 后，§17.1 的
跨 C/V stage 调度仍然合法；
它只是不能隐藏同样多的 Cube 核内 MTE 延迟。反之，若 Vector hard crossing state 或 `pipe_p` 容量无法
支持 `P=2`，则不能保留该跨 C/V 调度。

## 18. 验证计划

### 18.1 IR 回归测试

至少增加以下 lit 用例：

- `P = 0/1/2/4`；
- `N = 0/1`、`N < P`、`N = P`、`N > P`；
- tile 0 初始化分支；
- A5 tile-entry frontend IR；
- A2/A3 tile-entry raw GM buffer frontend IR；
- A2/A3 global-entry `gm_slot_tensor` frontend IR；
- A5 global-entry `gm_slot_tensor` frontend IR；
- entry 不同但 schedule skeleton 相同的对比检查；
- Vector 核内 crossing-cut 值生成正确的 `Bcv` 槽 multi-buffer 与 `i mod Bcv`；
- QK/P/PV tpipe payload 不生成额外 `alloc_multi_tile`；
- K/V `mat` 双缓冲、P tpipe consumer local slots 与 `left/right` 单缓冲、双缓冲的独立组合；
- QK/PV `acc` ping-pong 与 FIX drain 回退；
- 多 pipe id、DIR_BOTH 和显式 split；
- capacity、trip count、transaction balance 和 reverse-dependence 负例。

### 18.2 Lowering 检查

- A5 tile-entry 输出只含 L2L/local-buffer pipe，不含 GM tensor entry；
- A2/A3 tile-entry 输出含 raw GM slot buffer 与 local consumer buffer，不含 global entry；
- A2/A3 与 A5 global-entry 保持 `talloc/tstore/tpush` 与 `tpop/tload/tfree` 事务顺序；
- A5 单向 global-entry lower 为 L2G2L，并输出 `DIR_C2V_GM`/`DIR_V2C_GM`；
- PlanMemory 前保留 `alloc_multi_tile/multi_tile_get`，并生成各地址空间的 `multi_buffer_addrs`；
- Sync 在 slot resolve 前运行，最终 IR 中才出现动态地址 `arith.select`；
- soft local buffer overflow 能撤销对应 look-ahead，但不改变 strict 模式下的 `P`；
- hard Vector state 或 pipe capacity overflow 在 strict 模式下给出确定性诊断；
- 生成的 C++/VPTO 可由目标工具链编译；
- `P = 0` 与串行基线 IR/结果等价。

### 18.3 功能与性能验证

- 使用小 shape golden 覆盖 mask、非整 tile、不同 dtype；
- 比较 `P = 0` 与 `P > 0` 的数值结果；
- 在 CA model/NPU 上确认无死锁并统计 pipe stall；
- 分别测量 `P = 1..capacity`，验证最优值随 shape 与平台变化；
- 检查 A5 local memory、A3 GM FIFO 与 Cube/Vector 核内 multi-buffer 资源没有超预算；
- 从指令 trace 验证 `MTE2(i+1) || MTE1/Cube(i)`、必要的 L0/FIX overlap 和 Vector ping-pong；
- 与手写 FA performance kernel 比较 stage distance、buffer 轮换和事件依赖；相同逻辑流水不等价于保证
  完全相同的周期数，最终性能仍以 CA model/NPU profile 为准。

## 19. 分阶段实现建议

### 阶段 1：Flash Attention 精确模式

- 只识别规范的四 stage FA graph；
- 只支持静态 `P` 和静态 S1 trip count；
- 只支持 tile entry；
- A5 走 L2L local buffer，A2/A3 走 raw GM buffer L2G2L；
- 使用单 schedule loop；
- tpipe payload 不生成 local multi-buffer；
- 为 Vector hard crossing state 生成 tile-native multi-buffer；
- 使用 strict resource policy，加入 pipe capacity 和 PlanMemory 诊断。

### 阶段 2：核内 Operand Software Pipeline

- 识别 Cube 的 MTE2->MTE1->Cube->FIX 链；
- 支持 K/V L1 ping-pong、P tpipe consumer local slots、必要的 L0A/L0B 和 L0C ping-pong；
- 支持 Vector scratch/output ping-pong；
- 在 PlanMemory 前生成逻辑 local slot，在 Sync 后生成地址 select；
- local overlap 失败时允许独立回退，不取消可行的 C/V pipeline。

### 阶段 3：Entry 无关

- 接入 `LogicalPipeTransaction` adapter；
- 支持 A2/A3 与 A5 global entry；
- 用同一组 schedule FileCheck 对比 tile/global 两种输入。

### 阶段 4：通用 CV pipeline 与资源搜索

- 从固定四 stage 识别扩展为合法 stage graph cut；
- 支持动态 `N` 与运行时 `Pe`；
- 支持多个独立 inner-loop pair；
- 加入候选 clone/feasibility 或 planner query；
- 基于资源和延迟模型联合选择 `P`、L1/L0 depth 与 Vector scratch depth。

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
8. QK/P/PV payload 的多槽存储只由 tpipe 表达，不生成重复的 local `alloc_multi_tile`。
9. `pipe_p.slot_num` 覆盖最大有效 preload，且 Vector 核内 crossing state 使用精确或保守的 `Bcv`
   个版本。
10. Cube local pipeline 分别分析 L1 `mat`、L0A/L0B `left/right` 和 L0C `acc`，不能以 L1 双缓冲
    代替全部 L0 生命周期证明。
11. hard local buffer 失败必须降低 `P` 或拒绝调度；soft local buffer 失败可以撤销对应核内 overlap，
    不能只把已调度代码的槽数改为 1。
12. PlanMemory 前保留逻辑 slot，PlanMemory 只分配地址，Sync 在 slot resolve 前运行，最终 select 由
    `PTOResolveBufferSelect` 生成。
13. 两个函数必须在完整 plan 合法后原子式改写；资源 retry 必须从未改写 IR 开始。
14. target verifier 可以拒绝某个 entry/transport 组合，但不能改变已定义的 schedule 语义。
