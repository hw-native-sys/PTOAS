# PTODSL TileOp 多 Phase 设计

## 1. 背景和目标

PTODSL 需要一种可复用的 tile 级 helper：调用者把已经规划好的片上
`Tile` 传入，helper 用 MTE、SIMD、Cube 和必要时的 SIMT 微指令完成一段
计算。一个真实的 helper 很少只属于一条 pipe。例如，一个 block 可能依次执行：

```text
MTE load -> SIMD normalize -> MTE move -> Cube matmul -> SIMD post-process
```

旧的 `simd` / `cube` subkernel 模型把一次 helper call 概括为单一角色。这样无法
表达 helper 内不同阶段分别读取、写入了哪个边界 Tile，也无法正确处理多段
Vector/Cube 计算、MTE 与计算域的归属，以及直接写在 helper 中的 SIMT 微指令。

本设计以 `@pto.tileop` 作为唯一的 tile 级 helper surface。它的边界简单且稳定：
只传递 `Tile` 和 PTO scalar。helper body 则可以包含多个不同 pipe 的微指令阶段。

本文描述的是目标设计。当前实现仍保留 TensorView ABI、单一 `primary_domain` 和
单个主计算 section 等过渡性行为；第 10 节列出差异和迁移顺序。

## 2. 基础概念

### 2.1 Kernel、helper、Tile 与 scalar

- **kernel** 是 `@pto.jit` 定义的可执行入口，负责 tile 分配、整体调度和调用
  helper。
- **tileop helper** 是由 `@pto.tileop` 定义、可被 kernel 多次调用的命名函数。它
  不单独规划 Tile 生命周期，也不是设备执行入口。
- **Tile** 是调用者已经拥有并交给 helper 使用的片上数据缓冲区。它是 tileop
  唯一的复合数据边界类型。
- **PTO scalar** 是 `i32`、`f32` 等设备标量，可作为大小、步长、标志或计算参数。
  scalar 不代表一块可由同步系统追踪的 Tile 数据。

`@pto.simt` 保留为独立的 launched-SIMT public surface，可使用既有的 ptr ABI 和
launch 语法。它不成为 tileop 的 ABI，也不能从 tileop 中被调用。本设计额外允许
用户在 tileop body 中直接写 **SIMT 微指令**；这两件事不同。

| surface | public ABI | 使用场景 |
|---|---|---|
| `@pto.tileop` | `Tile`、PTO scalar | 复用多 pipe 的 tile 级计算；body 可直接写 SIMT 微指令 |
| `@pto.simt` | `ptr`、PTO scalar | 保留的 launched-SIMT helper；需要将 ptr 作为函数边界时使用 |

因此，“tileop 允许 SIMT”只表示其 body 可以包含和自动 outline SIMT 微指令，
不表示 `@pto.tileop` 接受 ptr，也不替换 `@pto.simt`。

### 2.2 Pipe、phase 与 section

- **pipe** 是一类执行资源，例如 MTE、Vector、Cube、SIMT 或 Scalar。
- **phase** 是 helper 内一段有确定执行 pipe、边界 Tile 读写和顺序关系的操作。
  一个 helper 可以有任意多个 phase，同一 pipe 也可以多次出现。
- **section** 是 PTOAS 为 VPTO 分模块和 lowering 物化的 IR 容器：
  `pto.section.vector` 或 `pto.section.cube`。它不是用户选择的 helper role。

phase 描述“这段操作做了什么以及依赖什么”；section 描述“这段 Vector/Cube 操作
在后续 VPTO 流程中属于哪个计算模块”。MTE、SIMT、同步和控制流可构成 phase，
但不自动变成 Vector/Cube section。

## 3. `@pto.tileop` 的公共契约

### 3.1 ABI

`@pto.tileop` 的位置参数只允许：

| 类型 | 用途 |
|---|---|
| `pto.Tile` | 输入、输出或 scratch Tile |
| PTO scalar | 标量参数和标量结果 |

不允许 `TensorView`、`PartitionTensorView`、`ptr`、`memref`、host tensor、
`TensorSpec`、vreg、mask 或 pipe handle 跨 helper 边界。ptr 继续只属于独立
`@pto.simt` 的 public ABI；这项 `@pto.simt` 契约不在本次 redesign 的收紧范围内。

Tile 输出通过可写的 Tile 参数表达，不通过 function result 返回。function result
仅允许 PTO scalar。这样 caller 始终拥有 Tile 的分配、别名和生命周期，PTOAS 也能
将 helper 的读写明确映射到 call 的 Tile operands。

```python
# 设计示意：所有复合数据都通过 Tile 边界传递。
@pto.tileop
def fused_block(src: pto.Tile, weight: pto.Tile, dst: pto.Tile,
                scratch: pto.Tile, rows: pto.i32, cols: pto.i32):
    # body 见下一节。
    ...

@pto.jit
def kernel(src, weight, dst):
    # kernel 负责创建和规划 src_tile、weight_tile、dst_tile、scratch_tile。
    fused_block(src_tile, weight_tile, dst_tile, scratch_tile, rows, cols)
```

### 3.2 Body 允许和禁止的内容

tileop body 允许下列内容：

- MTE、SIMD、Cube 和 SIMT 微指令；
- 标量运算、地址派生、cast、`arith` 和 `scf` 等结构性操作；
- 显式 pipe 同步和显式 SIMT 线程同步。

tileop body 不允许下列内容：

- 高层 TileOps，例如由自动调度或 Tile API 负责语义的 load/store/compute 操作；
- `alloc_tile`、`reserve_buffer`、`TAlloc` 等需要为 callee 单独规划生命周期的
  Tile 分配；scratch 和输出 Tile 必须由 caller 传入；
- 对另一个 tileop helper 或 `@pto.simt` helper 的调用；
- 将内部 vreg、mask、pointer 或 pipe handle 作为参数或结果泄漏到 helper 外。

这里的限制不禁止在 body 内从 Tile 派生地址，也不禁止 SIMT 微指令使用这些内部
地址。限制的是 public ABI：派生 pointer 的作用域不能超出当前 helper 或其自动
生成的内部 SIMT entry。

## 4. 用户可见的多 Phase 模型

用户不为 helper 指定 `vector`、`cube` 或 `simt` role。用户按算法需要写微指令和
显式同步；PTOAS 从 body 推导阶段。下面是概念流程，不表示可直接编译的 Python
代码：

```text
Tile src, weight, dst, scratch
       |
       +-- MTE phase:      src -> scratch
       +-- SIMD phase:     normalize(scratch)
       +-- MTE phase:      scratch -> Cube input buffer
       +-- Cube phase:     matmul(weight, scratch)
       +-- SIMD phase:     post-process -> dst
       +-- SIMT phase:     per-element fixup(dst)
```

同一 helper 可以有多个 Vector section、多个 Cube section，以及二者交错的 phase。
例如 `Vector -> MTE -> Vector` 必须产生两个 Vector 计算 span，而不是为了保持
“单一 primary span”把中间 MTE 包进 section。控制流也不改变这一原则：PTOAS 递归
分析 `scf.for` 和 `scf.if` 的 region，在具体 region 内物化计算 span；不得因为顶层
只有一个 `scf.for` 就跳过或包裹整个 loop。

## 5. PTODSL tracing 与 PTOAS phase graph

### 5.1 PTODSL tracing 的职责

PTODSL tracing 是执行 Python helper 定义并记录 PTO IR 的过程。新设计中，它只做
两件与 tileop 相关的事：

1. 验证 Tile/Scalar public ABI，并记录带 `pto.tileop.helper` marker 的命名
   `func.func`；
2. 原样保留 body 中的微指令、结构化控制流和用户写出的同步。

tracing 不预先选择 Vector 或 Cube，不写 `primary_domain`，不把 body 包进
`pto.section.vector/cube`，也不预先填写 phase 或 operand effect 属性。inline
`with pto.tileop():` 与命名 `@pto.tileop` 使用同一套 body 规则；inline 路径不因
位置不同而硬编码一个 section kind。

```text
PTODSL tracing
  -> func.func {pto.tileop.helper}，Tile/Scalar ABI，原始微指令 body
  -> PTOAS 推导 phase graph 并验证
```

### 5.2 Phase graph

PTOAS 的 phase 推导 pass 扫描 helper body，读每个微指令自身声明的 `getPipe()`，
并生成有序 phase graph。每个 phase 至少记录：

- 执行 pipe；
- 在源顺序和结构化控制流中的位置；
- 读取和写入的 Tile 参数编号；
- 与前后 phase 的数据依赖和控制依赖；
- 用户显式同步形成的不可删除顺序边；
- 对 MTE phase 而言，最终归属的 Vector 或 Cube 物理模块。

读写摘要必须追溯到 helper 参数。例如 `tile[row, col:]`、地址计算、cast、
`memref.subview`、`scf` 的 iter_arg/result 等会产生派生 SSA value；真正被微指令
读写的可能是派生值而不是函数参数。PTOAS 沿这些透明派生关系追溯，最后记录
“第几个 Tile 参数被读或写”。这样摘要既不把内部 pointer 误当作 ABI，也不会漏掉
tile slice 的边界 effect。

目标 IR 属性形态如下，字段名仅说明语义：

```text
func.func @fused_block(%src, %weight, %dst, %scratch, %rows, %cols)
    {pto.tileop.helper,
     pto.tileop.phases = [
       {pipe = MTE2, tile_uses = [0], tile_defs = [3]},
       {pipe = V,    tile_uses = [3], tile_defs = [3]},
       {pipe = MTE1, tile_uses = [3], tile_defs = [3], owner = CUBE},
       {pipe = CUBE, tile_uses = [1, 3], tile_defs = [2]},
       {pipe = V,    tile_uses = [2], tile_defs = [2]},
       {pipe = SIMT, tile_uses = [2], tile_defs = [2]}
     ],
     pto.tileop.operand_effects = [read, read, readwrite, readwrite, read, read]} {
  ...
}
```

`operand_effects` 是所有 phase 的 Tile effect 的并集，用于快速检查和 callsite
建模；真正的顺序和逐阶段同步依据 `phases`。不再存在 `pto.tileop.primary_domain`。
scalar 可以出现在参数列表和 phase 内，但不作为 Tile memory hazard 节点。

## 6. 自动同步和用户显式同步

PTOAS 负责普通数据 hazard：某个 phase 写入 Tile 后，后续不同 pipe 的 phase 或
另一次 helper call 读取/写入同一 Tile 时，`InsertSync` 根据 phase graph 插入所需的
pipe 同步。它既分析 helper body 内的常规生产者/消费者关系，也在 callsite 将摘要中
的 Tile 参数编号映射回本次 `func.call` 的实际 Tile operand。

```text
helper A 的 phase 2 写 %scratch
        +-----------------------+
                                v
helper B 的 phase 0 读 %scratch

InsertSync 根据写/读的 pipe 和顺序插入必要同步。
```

以下同步不能靠普通 Tile 读写关系可靠推导，必须由用户显式表达：

- 算法规定的阶段边界、流水重叠和事件编号；
- 对同一 Tile 的刻意并发访问或别名协议；
- SIMT work-item 之间的 `syncthreads`、thread fence 等线程级同步。

显式 pipe barrier、flag wait/set 和 SIMT 线程同步会成为 phase graph 的固定边。
PTOAS 不删除它们；自动同步只补尚未覆盖的普通数据 hazard，且不能再插入等价的
重复同步。

## 7. TileOp 内的 SIMT 微指令

tileop 允许直接包含 SIMT 微指令 phase，但这不等于允许调用 `@pto.simt` helper。
PTOAS 会将可 outline 的连续 SIMT 微指令 span 变成内部实现细节：

```text
tileop 原始 body
  -> pto.store_vfsimt_info(dim_z, dim_y, dim_x)
  -> SIMT 微指令 span

PTOAS SIMT outline
  -> 内部 func.func {pto.simt_entry}，capture 仅为 Tile/Scalar
  -> pto.simt_launch(dim_x, dim_y, dim_z, captures...)
```

每个 SIMT span 前必须显式出现并支配该 span 的
`pto.store_vfsimt_info(dim_z, dim_y, dim_x)`。该配置只消费给紧随其后的一个
SIMT span；缺失、被多个 span 竞争、维度不合法或无法确定配置关联时均报错。显式
维度避免 PTOAS 从 Tile shape 或循环结构猜测线程布局。

outline 后的 `pto.simt_entry` 只捕获 Tile 和 scalar。SIMT body 内需要的 pointer
从捕获 Tile 派生，不能被提升为 tileop 的函数参数。用户写出的 `syncthreads` 和
其他 SIMT 同步原样保留在 entry 中。若一个 SIMT region 无法在保持结构化控制流和
capture 规则的前提下 outline，PTOAS 必须诊断，而不能退化为普通 Vector 或忽略它。

## 8. Section 物化、MTE 归属和 VPTO split

### 8.1 Section materialization

在同步摘要已经可用后，PTOAS 将每个最大 SIMD phase range 物化为
`pto.section.vector`，将每个最大 Cube phase range 物化为 `pto.section.cube`。
同一函数可以生成多个同类或异类 section。MTE、Scalar、SIMT 和同步保持在 section
外，以便它们保留自己的 pipe 语义。

```text
MTE -> [section.vector] -> MTE -> [section.cube] -> [section.vector] -> SIMT
```

materialize pass 必须在嵌套 region 中处理 span，不能把混有 MTE、sync 或另一个
计算域的整个 `scf.for` / `scf.if` 容器套成一个 Vector 或 Cube section。inline 后
若出现同域 section 嵌套，由 section normalization 展开；跨域嵌套或无法确定语义的
布局必须拒绝。

### 8.2 MTE 归属

MTE 本身不等于 Vector 或 Cube。PTOAS 根据 Tile 数据流把 MTE phase 归属到其唯一
服务的 Vector 或 Cube 计算模块：例如 MTE 将数据搬到只被 Cube phase 消费的 Tile，
该 MTE 归 Cube；计算结果经 MTE 写回且唯一来自 Vector phase，则归 Vector。

一个 MTE phase 若同时服务两个计算域、数据流不足以确定归属，或其 Tile alias 使
归属不唯一，必须报错。用户需要将数据移动拆开、使用独立 Tile，或写出能够消除
歧义的算法结构；PTOAS 不能把同一条 MTE 指令复制到两个模块。

### 8.3 `VPTOSplitCVModule` 与 `kernel_kind`

`VPTOSplitCVModule` 以 section 和 MTE 归属为输入生成 Vector/Cube 模块。它必须：

- 支持一个函数中的多个 Vector 和 Cube section；
- 在 Vector clone 中保留全部 Vector section、归属 Vector 的 MTE，以及生成的
  SIMT entry/launch；
- 在 Cube clone 中保留全部 Cube section 和归属 Cube 的 MTE；
- 只在 split clone 中删除另一域的内容，不能在用户显式要求单一 `kernel_kind`
  的模块中静默删除相反域的 phase。

`pto.kernel_kind` 是 PTOAS 在模块分拆和 VPTO lowering 时使用的模块语义标记。
它不是 `@pto.tileop` 的源级参数，也不能代替 phase 分析。显式单 kind 模块若仍含
不属于该 kind 的 phase，应当诊断；否则错误会被隐藏到更晚的 lowering。

## 9. 验证规则和预期诊断

PTOAS 的 tileop contract verifier 至少应拒绝以下情况：

- 非 Tile/Scalar ABI，或非 scalar function result；
- 高层 TileOps、callee-local Tile allocation、tileop/helper 调用；
- SIMT span 前没有唯一且有效的 `store_vfsimt_info`；
- SIMT outline 需要捕获 ptr、vreg、mask 或 pipe handle；
- 无法追溯到 Tile 参数的边界 memory effect；
- 无法唯一归属到 Vector/Cube 的 MTE phase；
- 无法物化的跨域/混合 section 结构；
- 显式单 kind 模块中存在相反计算域的 phase。

诊断应指出 helper、phase 和相关 Tile 参数，而不是只在最终 VPTO 或 LLVM lowering
阶段报告“非法 section”。

## 10. 当前实现差异和迁移计划

| 项目 | 当前实现 | 目标设计 |
|---|---|---|
| tileop ABI | Tile、TensorView、PartitionTensorView、scalar | 仅 Tile、scalar |
| 计算域 | 单一 `primary_domain` | 多 pipe、多 phase、无 primary domain |
| SIMD/Cube | 只允许一个主计算域和主 span | 可有多个 Vector/Cube section 并交错 |
| SIMT | tileop body 禁止 SIMT-only op | 允许直接写 SIMT 微指令并自动 outline |
| 摘要 | phase + effect，服务单主域模型 | phase graph + Tile effect + MTE owner |
| section | 只物化单个主计算 span | 物化所有最大 SIMD/Cube span |
| split | 依赖单域化 section 形态 | 处理多 section、MTE owner 和 SIMT entry |
| 同步 | 已开始按 phase 建 callsite 节点 | 覆盖多 phase body、跨 call Tile hazard 与显式边 |

独立 `@pto.simt` 的 ptr ABI、launch syntax 和使用场景保持不变；本 redesign 仅修改
`@pto.tileop` 的 ABI 及其 body 中的 SIMT 微指令处理方式。

迁移按以下顺序进行：

1. 收紧 PTODSL `@pto.tileop` ABI 到 Tile/Scalar，并补齐前端和后端负例；
2. 让 tracing 只产生 raw tileop body，不写 `primary_domain` 或预套 section；
3. 将 phase summary 扩展为无主域的多 phase graph，并完成 Tile 派生值追溯；
4. 让 `InsertSync` 消费该 graph，先保证普通 Tile data hazard 和显式同步共存；
5. 实现 tileop 内 SIMT span 的 launch 配置验证与 outline；
6. 改造 section materialization、MTE ownership 和 `VPTOSplitCVModule`；
7. 完成 VPTO/LLVM lowering 回归后，删除旧的单主域 tileop 路径和兼容属性。

## 11. 回归测试

目标实现至少需要以下覆盖：

- Tile/Scalar ABI 正例，以及 TensorView、ptr、memref、vreg、mask 和非 scalar
  result 的负例；
- MTE+SIMD、MTE+Cube+SIMD、重复同域 section、控制流内 section；
- Tile slice、address cast 和 `scf` iter_arg 派生值的 effect 追溯；
- helper 内和跨 helper call 的自动 data-hazard sync，用户显式 barrier 保留且不重复；
- tileop 内 SIMT 微指令、合法 launch 配置、线程同步、缺失配置和非法 capture；
- MTE owner 推导成功、歧义 owner 诊断；
- 多 section Vector/Cube split、SIMT 位于 Vector 侧、显式单 kind 冲突诊断；
- 最终 VPTO 和 LLVM lowering 的端到端编译回归。
