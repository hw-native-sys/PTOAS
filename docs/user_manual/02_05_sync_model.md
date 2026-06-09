# 2.5 PTO 同步模型

## 1. 概述

PTO 不是默认全局串行的执行模型。不同 pipe、不同执行域和不同本地存储对象可以在依赖允许时并行推进，因此程序必须通过数据依赖、显式同步或可被编译器分析出的内存依赖来表达必要顺序。

同步模型需要同时区分两件事：

- 编译期如何发现某个生产者和消费者之间需要顺序约束
- 运行时同步指令最终到底约束了哪些 pipe 和 event 资源

这两个层次相关，但不是同一个概念。

## 2. 程序顺序、数据依赖与同步顺序

PTO 程序中常见三类顺序：

- 文本顺序：IR 中 op 的书写先后
- 数据依赖顺序：SSA use-def、buffer read/write、view/tile alias 等关系
- 同步顺序：`record_event` / `wait_event`、`set_flag` / `wait_flag`、barrier、`syncall` 等显式同步建立的顺序

文本相邻不等价于硬件执行上天然满足所需同步。特别是同一个 pipe 上的多个异步或队列化操作，也可能需要 pipe-local barrier 或其它显式机制来建立架构要求的可观察顺序。

## 3. 同步抽象层次

### 3.1 高层 endpoint 同步

高层同步使用“生产者类型/消费者类型”描述依赖，不直接写死具体 pipe：

- `pto.record_event`
- `pto.wait_event`
- `pto.barrier_sync`

`pto-lowering-sync-to-pipe` 会根据 op type 到 pipe 的映射，把这些高层同步 lowering 成更底层的 pipe 同步。例如 `barrier_sync` 接收的是 `SyncOpType`，lowering 后才变成具体 `pto.barrier`。

### 3.2 低层 pipe/event 同步

低层同步直接携带 pipe、event 或 barrier 信息：

- `pto.set_flag`
- `pto.wait_flag`
- `pto.set_flag_dyn`
- `pto.wait_flag_dyn`
- `pto.barrier`
- `pto.tsync`
- `pto.syncall`
- `pto.sync.set`
- `pto.sync.wait`

`set_flag` / `wait_flag` 的运行时语义是 pipe pair 加 event id。它们不携带“保护哪个 buffer”的信息；buffer、alias、生命周期等只是在编译期用来判断是否需要插入这组同步。

### 3.3 A5 buffer-id 同步

A5 还存在 buffer-id 形式的同步：

- `pto.get_buf`
- `pto.rls_buf`

这类同步以 buffer id token 建立顺序，和 A3/A5 上的 `set_flag` / `wait_flag` event-id 模型不同。它们同样属于同步语义的一部分，不能和普通数据 op 混为一谈。

## 4. 自动同步如何工作

自动同步不是凭空恢复硬件顺序，而是依赖 IR 中已经存在的语义：

- op 的 pipe 归属
- MemoryEffects 中的 read/write 信息
- tile/view/memref 的 alias 关系
- 地址空间和本地/全局存储属性
- 控制流、loop、branch 的 must-path 信息
- 显式事件对象和已有同步
- 目标架构相关限制

同步分析会根据这些信息判断某个 producer/consumer 之间是否需要同步；最终生成的同步指令通常只携带 pipe pair、event id 或 barrier 作用域。

## 5. 当前同步模式

### 5.1 `pto-insert-sync`

基于依赖分析的自动同步插入路径。它从 PTO ISA 翻译出同步分析 IR，结合 pipe、alias、MemoryEffects 和控制流信息插入 `set_flag` / `wait_flag` 或 pipe barrier，并做冗余同步删除和 event-id 分配。

### 5.2 `pto-inject-barrier-all-sync`

保守兜底路径。当前实现是在有 MemoryEffects read/write 的 PTO pipe op 前插入 `pto.barrier <PIPE_ALL>`。它不复用 `set_flag` / `wait_flag` 依赖分析，也不替代 `tpush` / `tpop` / `tfree` 等复杂 op 的内部同步协议。

两种模式在 driver 中互斥，不应同时启用。

## 6. Event ID 与动态 event id

`pto.set_flag` / `pto.wait_flag` 使用静态 `#pto.event<EVENT_ID*>` 属性。`pto.set_flag_dyn` / `pto.wait_flag_dyn` 使用运行时 index 值，常和 `!pto.eventid_array<N>`、`pto.eventid_array_get` / `set` 配合。

event id 是有限硬件资源，event-id 分配阶段需要保证生命周期不重叠。更多 event lane 不一定表示更强同步；在 multibuffer、loop-carried 或 ping-pong 场景中，多个 event id 往往用于区分不同 phase，以保留并行度。

## 7. 冗余同步判断的关键边界

因为运行时 `set_flag` / `wait_flag` 不携带 buffer 信息，所以冗余删除不能把 dep root buffer 当作运行时语义本身。判断一组外层同步是否被内部同步覆盖时，核心是执行区间、pipe pair、event-id 能力和控制流路径是否能保证同样的顺序。

同时，loop back-edge、zero-trip、分支空路径和 macro op 内部协议都可能改变同步是否可删。遇到这些场景时，分析必须保持保守，避免把只用于 loop 回边或内部协议的同步误当成普通数据依赖同步。

## 8. 编程模型上的约束

### 8.1 数据依赖要可见

如果程序真实依赖某个 tile、view 或 buffer 的生产结果，应让这种关系在 SSA、buffer 使用、MemoryEffects 或事件对象中可见。

### 8.2 不要依赖隐式顺序

两个 op 文本上相邻，不代表它们天然满足架构需要的同步边界。同 pipe 操作也可能需要 pipe-local barrier。

### 8.3 复杂控制流需要清晰语义

循环、分支、双缓冲和多 pipe 交织会显著提高自动同步难度。此类程序应尽量保留清晰的 tile/view/pipe/event 语义，必要时使用显式同步或稳定的结构化 pattern。
