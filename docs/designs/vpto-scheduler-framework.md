# VPTO 调度分析框架

本文描述当前分支中 [PTOAS issue #1143](https://github.com/hw-native-sys/PTOAS/issues/1143) 阶段一框架的实际实现。当前 pass 构建 region、依赖 DAG、目标资源模型和寄存器压力报告，但不选择新顺序，也不修改 IR。

## 1. 集成位置、作用范围与模式

调度 pass 位于 `prepareVPTOForEmission` 的最终 `CSE` 之后、`PTOValidateVPTOEmissionIR` 之前：

```text
sync lowering / SIMT lowering
  -> wrapper expansion / vecscope inference / optional soft-postupdate
  -> LICM / loop-counter narrowing / canonicalize / CSE
  -> VPTOSchedulerPass
  -> PTOValidateVPTOEmissionIR
```

此时调度器看到 emission-ready VPTO SSA，且仍处于下游硬件寄存器分配之前。pass 调度范围严格限制为模块内 `pto.vecscope` 和`pto.strict_vecscope` 的 body。本文后续使用“vecscope”统称 `pto.vecscope` 和 `pto.strict_vecscope`。每个 vecscope body 是独立调度范围；pass 不跨 vecscope 移动或联合分析操作。body 中若存在其他嵌套 region，其 block 仍属于该 vecscope，并按深度优先顺序分析。

`ptoas` 选项为 `--vpto-scheduler=<off|analyze|on>`：

| 模式 | 当前行为 |
| --- | --- |
| `off` | 默认值；driver 不加入 pass。独立运行 pass 时也立即返回。 |
| `analyze` | 构建分析并向标准错误输出报告；IR 不变。 |
| `on` | 当前与 `analyze` 完全相同，仅报告中的 `mode` 字段不同；IR 不变。 |

driver 只允许在 A5 上显式启用 `analyze` 或 `on`。pass 本身要求当前模块或某个祖先模块声明 `pto.target_arch = "a5"`；属性缺失或有效值不是 A5 时均失败。

独立运行方式：

```bash
pto-test-opt input.pto '-pto-vpto-scheduler=mode=analyze'
```

## 2. 组件和数据流

```text
VPTOSchedulerPass
  -> collect pto.vecscope / pto.strict_vecscope
  -> VPTOSchedulingOpInterface / operation classifier
  -> VPTOSchedRegionBuilder
  -> VPTOSchedDAGBuilder
       -> SSA dependencies
       -> memory dependencies
       -> implicit-state and synchronization dependencies
       -> unknown-model fallback edges
       -> critical-path depth/height
  -> VPTOSchedModel
  -> Top/Bottom VPTOSchedBoundary
       -> direction-local ResourceTracker
       -> direction-local RegPressureTracker
       -> direction-local HazardRecognizer
  -> original-order simulation through Top Boundary trackers
  -> deterministic per-module text report
```

当前没有 `VPTOScheduler`、candidate、strategy、schedule result、schedule verifier或 IR apply 组件。`Boundary` 和 tracker 已提供基础接口，但 pass 只用它们分析原始顺序。

### 2.1 核心数据对象

| 对象 | 定义 |
| --- | --- |
| `VPTOSchedRegion` | block 内可独立分析的一段连续 operation；记录 operation 原始顺序及前后边界。 |
| `VPTOSUnit` | scheduling unit，也是 DAG node；与 region 内一个 operation 一一对应。 |
| `VPTOSchedEdge` | 两个 `VPTOSUnit` 之间的有向依赖；记录依赖类型、强度、latency 和原因。 |
| `VPTOSchedDAG` | 持有 region、`VPTOSUnit`、edge、live-in 和 live-out，并维护 operation 到 `VPTOSUnit` 的映射。 |

RegionBuilder 先产生 `VPTOSchedRegion`，DAGBuilder 再按 region 中的原始顺序为每个 operation 创建一个 `VPTOSUnit` 并建立 `VPTOSchedEdge`。本文后续使用“SUnit”简称 `VPTOSUnit`。

## 3. 操作调度语义

### 3.1 接口契约

`VPTOSchedulingOpInterface` 描述单个 operation 的局部调度语义，只提供一个查询：

| 查询 | 含义 |
| --- | --- |
| `getVPTOSchedulingSemantics()` | 返回完整、规范化的 `VPTOSchedulingSemantics`。 |

`VPTOSchedulingSemantics` 是 operation-local 语义的固定数据结构：

| 字段 | 含义 |
| --- | --- |
| `schedulingClass` | `Schedulable`、`Structural`、`SchedulingBoundary` 或 `Unsupported`。 |
| `effects` | 隐式状态、vecscope 内 memory barrier 和 post-update 等局部 effect。 |
| `memoryBehavior` | 普通内存语义的完整性：`None` 表示确定无访问，`Explicit` 表示由访问列表完整描述，`Unknown` 表示缺少完整声明。 |
| `memoryAccesses` | 规范化的读写、地址空间、范围和顺序属性。 |

`effects` 中的每个 `VPTOSchedulingEffect` 包含：

| 字段 | 含义 |
| --- | --- |
| `kind` | effect 类型。 |
| `resource` | effect 的逻辑域或动作名，例如 SPR 名、`ctrl`、`signal`。 |
| `value` | 可选的 SSA identity；当前用于 post-update 地址结果。 |

已定义的 effect kind 为 `ImplicitRead`、`ImplicitWrite`、`Barrier`、`PostUpdate`、`VolatileMemory`、`AtomicMemory` 和 `Unknown`。`PostUpdate` 只标注 SSA 地址结果；`VolatileMemory` 和 `AtomicMemory` 只改变 memory ordering。这三类 effect 不直接生成同名 DAG edge。

`memoryAccesses` 中的每个 `VPTOMemoryAccess` 包含：

| 字段 | 含义 |
| --- | --- |
| `address` | 被访问的 pointer 或 memref SSA value；无法确定时为空。 |
| `addressSpace` | 从 `address` 类型取得的地址空间；无法确定时为空。 |
| `byteOffset`、`byteSize` | 可选静态字节区间。 |
| `reads`、`writes` | 是否读、写普通内存。 |
| `ordered` | 是否必须保留内存顺序。 |
| `unknown` | 地址或读写类型是否无法可靠确定。 |

接口只报告当前 operation 的事实，不判断两个 operation 是否存在依赖。SSA def-use、alias root、may-alias 和依赖 edge 均由 DAGBuilder 统一计算。DAGBuilder 不应根据具体 operation 名称或 operand 位置重新推导已经进入 `VPTOSchedulingSemantics` 的局部语义。

除 scope 外同步边界外，`PTO_MicroOp` 实现该接口。set/wait flag、get/release buffer、pipe barrier、`dsb`、`dcci` 和 SIMT barrier/fence 等同步操作不为了 scheduler 实现该接口。vecscope verifier 保证它们不会进入函数级调度 DAG。

### 3.2 默认分类策略

这里的分类决定 operation 是否进入调度 region，不是 TargetModel 中的硬件资源分类。RegionBuilder 按下表从上到下匹配；先匹配的规则优先：

| 条件 | 分类 | RegionBuilder 行为 |
| --- | --- | --- |
| operation 为空、是 terminator 或包含 region | `SchedulingBoundary` | 结束当前片段，operation 不进入 region。 |
| operation 实现 `VPTOSchedulingOpInterface` | 使用接口返回的分类 | 当前默认接口实现在能确定 execution pipe 或存在任一调度 effect 时返回 `Schedulable`；接口也可显式返回 `SchedulingBoundary` 或 `Unsupported`。 |
| operation 未实现调度接口，且 `isMemoryEffectFree` | `Structural` | 与相邻 `Schedulable` operation 一同进入 region，用于保留 SSA 计算；纯 structural 片段不生成 region。 |
| 未命中以上规则 | `SchedulingBoundary`，`classificationKnown=false` | 作为保守边界结束当前片段，并计入 unclassified coverage。 |

`SchedulingBoundary` 和 `Unsupported` 都会切断 region，但含义不同。前者表示接口明确声明的调度边界；后者表示接口明确声明 scheduler 当前不支持该 operation。`Unsupported` 不作为 dialect 或接口缺失的默认值。任何缺少显式分类的 operation 都使用安全的 boundary 行为，并由 `classificationKnown=false` 保留“尚未分类”的事实。

默认接口实现解析 execution pipe 只是为了确认该 operation 具有已知的执行类别。解析优先级为：

1. `OpPipeInterface::getPipe()` 提供的精确 pipe；
2. Vector family fallback 为 `PIPE_V`；
3. Cube family fallback 为 `PIPE_M`；
4. SIMT family fallback 为 `PIPE_S`；
5. MTE family fallback 为 `PIPE_ALL`。

这组 fallback 由所有实现调度接口的 PTO micro-op family 共用，范围大于 scheduler pass 实际分析的 vecscope。当前 pass 只消费 vecscope，规范化流程不应在 vecscope 中产生 Cube operation；`Cube -> PIPE_M` 只保证通用接口查询能够描述 Cube family，不表示 vecscope scheduler 支持调度 Cube operation。当前 vecscope verifier 没有按 operation family 建立完整白名单，因此该 fallback 也为手写或非规范 IR 提供保守分类。

MTE 的 `PIPE_ALL` 只表示尚无精确 execution pipe，不能解释为 operation 占用所有硬件 pipe。进入 TargetModel 后，MTE family 仍映射到 MTE sched class；资源映射见 8.2 节。

### 3.3 当前显式 effect

| 操作或属性 | effect |
| --- | --- |
| `mem_bar` | `Barrier("memory-order")` |
| atomic CAS/exchange/add/sub/min/max/and/or/xor | `AtomicMemory` |
| 带 `volatile` 或 `is_volatile` 属性的接口 op | `VolatileMemory` |
| 支持可选 `updated_base` 的已登记 memory op | 结果存在时产生 `PostUpdate` |
| `sprclr` | 以具体 SPR 名为域的 `ImplicitWrite` |
| `sprsti`、`sprsts` | 以具体 SPR 名为域的 `ImplicitRead` |
| `get_ctrl`、`set_ctrl` | `ctrl` 域的 `ImplicitRead` / `ImplicitWrite` |

登记 post-update 的操作为 `vlds`、`vldsx2`、`sprsti`、`sprsts`、`vldus`、`plds`、`pldi`、`psti`、`vsts`、`psts`、`vsldb`、`vsstb` 和 `vstas`。

## 4. RegionBuilder 与覆盖率

### 4.1 Region 形成

`VPTOSchedRegionBuilder` 只接收 vecscope 内的 block，并对单个 block 线性扫描。`SchedulingBoundary` 和 `Unsupported` 结束当前片段，且自身不进入 region；`Schedulable` 和 `Structural` 保留在片段中。只有至少包含一个 `Schedulable` 操作的片段才生成 region，因此纯 structural 片段不会进入 DAG。

每个 region 保存：

- 所属 block；
- block 内从 0 开始的 region index；
- 按原始顺序排列的 operation 指针；
- 前后 boundary operation；
- 前后 boundary reason。

reason 的固定形式为：

| 场景 | reason |
| --- | --- |
| block 起点/终点 | `block-start` / `block-end` |
| terminator | `terminator` |
| 包含 region 的操作 | `contains-regions` |
| 其他边界 | `<class>:<operation-name>` |

包含 region 的操作在父 block 中是边界，但 pass 会继续递归分析其内部 block。嵌套的专用 vecscope 除外：它不通过外层 vecscope 递归分析，而是作为独立调度范围处理。

### 4.2 Coverage

coverage 按函数统计所有遍历到的 vecscope 内操作，不限于已生成的 region：

- 四类操作总数；
- unclassified 操作总数；
- 每种 `Unsupported` 操作的数量；
- 每种 unclassified 操作的数量。

unclassified 特指：未命中显式分类规则，因而以 `SchedulingBoundary` 和 `classificationKnown=false` 保守处理的操作。`Unsupported` 则必须由调度接口显式返回，并携带 `classificationKnown=true`。unsupported 和 unclassified 名称在输出前按字典序排序。vecscope verifier 会在调度前拒绝 pipeline、system 和 cache 同步操作，因此这些操作不参与函数级 coverage。

## 5. DAG 数据结构

### 5.1 Node 与 edge

每个 region operation 对应一个 `VPTOSUnit`，包括 structural operation。构造 `VPTOSUnit` 时查询一次 `getVPTOSchedulingSemantics()` 并保存返回值；后续 DAG 构建只读取该快照。当前 `id` 和 `originalIndex` 都等于 operation 在 region 中的原始下标。SUnit 保存：

- operation 和 `VPTOSchedulingSemantics` 快照；
- predecessor/successor edge 列表；
- Must predecessor/successor 剩余计数；
- critical-path `depth` 和 `height`。

`VPTOSchedEdge` 保存 predecessor、successor、kind、strength、latency 和 reason。kind 枚举包含 `Data`、`Anti`、`Output`、`Memory`、`Control`、`Sync`、`Artificial`、`Cluster`；strength 包含 `Must` 和 `Weak`。

当前 builder 只生成 Must edge；`Control`、`Cluster` 和 Weak edge 尚未使用。builder 不合并重复边，因此同一对节点可以同时存在 SSA、memory 和 sync edge，也可能因多个保守规则产生多条同类边。报告中的 edge 数量是实际保存的 edge 数量。

### 5.2 构建顺序

DAG 按固定顺序构建：

```text
SSA edges
  -> memory edges
  -> implicit/synchronization edges
  -> unknown sched-class fallback edges
  -> critical paths
  -> Must dependency counters
```

该顺序也决定 analyze 报告中的 edge 输出顺序。

## 6. 依赖构建策略

### 6.1 SSA、live-in 与 live-out

对每个 operand：

- producer 在同一 DAG 时建立 `Data/Must` edge；
- producer 不在 DAG 时把 value 加入去重后的 live-in 集合。

SSA edge latency 取 producer sched class 的 `writeLatency`。operand 是已登记的 post-update 地址结果时，reason 为 `post-update address operand #N`；其余为 `ssa operand #N`。post-update 不改变依赖类型或 latency。

某个 result 只要有任一 user 不在当前 DAG，就加入 live-out；无 user 的 result 不是 live-out。

### 6.2 Memory semantic 与 alias 解析

操作语义层把每个内存 effect 规范化为 `VPTOMemoryAccess`：

```text
address / addressSpace
optional byteOffset / byteSize
reads / writes / ordered / unknown
```

`memoryBehavior` 的含义如下：

- `None`：已确认没有普通内存访问，`memoryAccesses` 为空；
- `Explicit`：普通内存访问已由 `memoryAccesses` 完整描述；
- `Unknown`：缺少完整声明，`memoryAccesses` 包含保守的 unknown、ordered write。

访问语义优先来自 `MemoryEffectOpInterface`。effect value 不是 `!pto.ptr` 或 memref 时忽略。可证明 memory-effect-free 的操作归为 `None`。`mem_bar`、`sprclr` 和 ctrl register 操作已由调度语义完整描述，也归为 `None`，但不添加 `Pure` trait。未实现内存接口且不属于上述两类的操作归为 `Unknown`。名称为 `pto.store`、`pto.stg`、`pto.st_dev`，或以 `pto.vst`、`pto.pst` 开头的操作会在语义归一化时额外标记为 write。上述操作特判均位于语义层，DAGBuilder 不识别具体 memory op。

DAGBuilder 读取 SUnit 的 `memoryAccesses` 后，为每个 address 解析 alias root。alias root 通过已有 alias 信息沿 defining-op 链向上查找，并用 visited 集合防止循环。不同 SSA root 即使位于同一 address space，也保守视为可能别名，因为内存规划可能让它们复用同一物理区间。

### 6.3 静态区间证明

当前只为以下标量 indexed op 计算静态字节区间：

```text
pto.load / pto.store / pto.ldg / pto.stg / pto.ld_dev / pto.st_dev
```

语义层在以下条件满足时记录候选区间：

1. access address 正是操作的 `ptr`；
2. offset 是可表示为有符号 64 位数的整数常量；
3. pointer element type 是固定长度整数、浮点或固定 vector，且位宽为整字节；
4. offset 和 size 计算不溢出。

DAGBuilder 解析 alias 时，如果 address 不是 alias root，则丢弃该候选区间。比较两个区间时若端点加法溢出，也退化为保守 may-alias。

offset 按 element 计数，字节区间为：

```text
[elementOffset * elementByteSize,
 elementOffset * elementByteSize + elementByteSize)
```

仅当两个访问具有同一 alias root 且都具备静态区间时，才可用区间不重叠证明 no-alias。不同且确定的 address space 直接视为 no-alias；其他情况均保守 may-alias。

### 6.4 Memory edge 规则

对于原始顺序中的每一对操作，只要任意访问对满足以下条件，就从较早操作向较晚操作建立一条 `Memory/Must` edge：

1. 两个访问 may-alias；并且
2. 任一访问为 ordered/unknown，或任一访问写内存。

普通 read/read 不建边。atomic 和 volatile effect 把所属访问标记为 ordered；若操作没有可识别访问，则创建 unknown read+write 访问。不同确定 address space 仍不会因 ordered 标记而建立边。

### 6.5 隐式状态依赖

隐式状态按 `resource` 字符串独立跟踪。每个域维护最后一次 write 和该 write 后的 read 集合：

- write -> read：`Data/Must`，latency 1；
- write -> next write：`Output/Must`，latency 0；
- reads since write -> next write：`Anti/Must`，latency 0。

因此不同 SPR 名互不约束，SPR 与 `ctrl` 也互不约束。首次 read 没有前置 write 时不建 Data edge，但会记录下来，使后续 write 获得 Anti edge。

### 6.6 完整 Barrier

`mem_bar` 是 vecscope 内唯一合法的同步操作，并被视为完整调度屏障：

- region 中所有较早节点 -> barrier：`Sync/Must`，latency 0；
- 最近完整 barrier -> 每个较晚节点：`Sync/Must`，latency 0。

`pto.barrier`、set/wait flag、get/release buffer、`dsb`、`dcci`、CMO/fence、SIMT barrier/fence 和跨核同步均必须位于 vecscope 外。自动 vecscope 推断把它们作为边界，显式 vecscope 的 verifier 会拒绝嵌套使用。因此阶段一调度器不为这些 scope 外操作构造 event、pipe 或 buffer-id effect/edge。

### 6.7 Unknown sched class 保序

语义上允许进入 region、但目标模型返回 unknown sched class 的节点，会获得：

- 原始前一节点 -> unknown 节点；
- unknown 节点 -> 原始后一节点。

两者均为 `Artificial/Must`、latency 0。该策略固定 unknown 节点与相邻节点的原始相对顺序。语义未分类的操作在更早的 RegionBuilder 阶段已经成为 boundary。

## 7. Critical path 与 ready 计数

critical path 只考虑 Must edge。builder 用拓扑遍历计算：

```text
depth(successor) = max(depth(successor), depth(node) + edge.latency)
height(node) = max(height(node), height(successor) + edge.latency)
```

拓扑遍历未覆盖全部节点表示 Must-edge 环，DAG 构建失败。Weak edge 不参与拓扑、depth、height 或 ready 计数。

构建结束后，每个 SUnit 的 remaining predecessor/successor 分别初始化为 Must 前驱/后继数量，供 Top/Bottom boundary 使用。

## 8. TargetModel

### 8.1 稳定接口

`VPTOSchedModel` 是只读查询接口，提供 machine model、resource 列表、pressure-set 列表、operation 的 sched class 和 value 的 pressure contribution。数据目前由静态 C++ 构造，scheduler 不依赖其存储方式。

数据结构已经预留以下字段：

- machine：target、version、issue width、micro-op buffer size；
- resource：units、buffer size、group members；
- sched class：micro-ops、write latency、resource reservations、read advance；
- pressure set：limit、weight、spill cost。

预留字段不等于当前模型已实现对应硬件策略。

### 8.2 `generic-a5-v1`

machine model 固定为：

```text
target=a5
version=generic-a5-v1
issueWidth=1
microOpBufferSize=0
completeness=minimal
```

资源为 scalar、vector、mte、cube、control、unknown；每类各 1 unit，buffer size 均为 0，无 resource group。sched class 如下：

| class | known | micro-ops | write latency | reservation |
| --- | --- | ---: | ---: | --- |
| structural | true | 0 | 0 | 无 |
| scalar | true | 1 | 1 | scalar，1 cycle |
| vector | true | 1 | 1 | vector，1 cycle |
| mte | true | 1 | 2 | mte，1 cycle |
| cube | true | 1 | 4 | cube，1 cycle |
| control | true | 1 | 1 | control，1 cycle |
| unknown | false | 1 | 1 | unknown，1 cycle |

所有 reservation 都从 issue cycle 开始，使用 1 unit。所有 read-advance 列表为空。

operation 到 sched class 的解析优先级为：

1. structural；
2. `OpPipeInterface`：S -> scalar，V/V2 -> vector，M -> cube，MTE1-5/FIX/ 两个 virtual MTE2 pipe -> mte，ALL -> control；
3. Vector、MTE、Cube、SIMT family marker；
4. 具有非空调度 effect -> control；
5. unknown。

`PIPE_NUM` 和 `PIPE_UNASSIGNED` 不直接映射 class，会继续尝试后续规则。

### 8.3 Pressure sets

模型定义 vector、predicate、scalar、address、align、special 六个集合。所有 limit 为空，weight 和 spill cost 均为 1。每个 value 当前只贡献 1 unit：

| value type | pressure set |
| --- | --- |
| `!pto.vreg` | vector |
| `!pto.mask` | predicate |
| `!pto.align` | align |
| `!pto.ptr`、memref | address |
| integer、index、float | scalar |
| 其他类型 | special |

## 9. Boundary

`VPTOSchedBoundary` 支持 Top 和 Bottom 两个方向。构造时重置 DAG 的 Must 依赖计数：

- Top：remaining predecessors 为 0 的节点进入 available；
- Bottom：remaining successors 为 0 的节点进入 available。

available 始终按 `originalIndex` 升序排列。每个 Boundary 聚合该方向的全部可变调度状态：current cycle、available、pending、scheduled、ResourceTracker、RegPressureTracker 和 HazardRecognizer。Top 与 Bottom 共享 DAG，但不共享 tracker 状态；依赖计数分别保存在 SUnit 的 predecessor/successor counter 中。构造函数接受目标模型，并默认创建 Null HazardRecognizer；也允许注入目标专用 recognizer。

三个状态操作为：

- `defer(unit, readyCycle)`：只接受当前 available 且 `readyCycle > currentCycle` 的节点，移入按 cycle、originalIndex 排序的 pending；
- `advanceToNextPendingCycle()`：推进到最早 pending cycle，并释放所有到期节点；
- `commit(unit)`：要求节点当前 available 且未提交；减少对应方向相邻节点的 Must 依赖计数，计数归零时按原始顺序加入 available。

当前 pass 分别构造 Top/Bottom boundary。Top Boundary 持有的三个 tracker 用于原始顺序模拟；Bottom Boundary 当前只提供初始 ready 数量，其 tracker 已按 Bottom 方向初始化但未提交 candidate。pass 不调用 Boundary 的 defer、advance 或 commit。

## 10. ResourceTracker 与 HazardRecognizer

### 10.1 ResourceTracker

ResourceTracker 保存逐周期 issue occupancy，以及按 resource id 分组的逐周期 occupancy。`evaluate(unit, requestedCycle)` 从请求周期开始线性搜索最早合法周期：

1. sched class 的 micro-op 数不能超过 machine issue width；
2. 当前周期 issue occupancy 加 micro-op 数不能超过 issue width；
3. 每项 reservation 引用的 resource 必须存在；
4. 请求 unit 数不能超过 resource units；
5. `[cycle + acquireAt, cycle + acquireAt + duration)` 内每周期都不能超量。

普通占用冲突使搜索继续到下一周期；非法模型立即返回 reason。搜索最多尝试 `2^20` 个周期，超出后返回 `resource search budget exceeded`。成功结果包含最早周期、该周期已有 issue 数形成的 slot，以及相对请求周期的 stall。

`commit` 会重新验证指定周期必须正好可发射，然后写入 issue/resource timeline。当前实现只消费直接 resource reservation，不解释 resource group、resource buffer 或 machine micro-op buffer。

### 10.2 HazardRecognizer

接口预留 `check(unit, direction, cycle)` 和 `commit(...)`，用于资源表难以表达的 pair、spacing 或 issue 限制。当前唯一实现是 `VPTONullHazardRecognizer`：所有 candidate 合法，最早周期不变，commit 无状态。

正确性依赖必须由 DAG Must edge 或 region boundary 表达，不能依赖当前空 hazard 实现。

## 11. RegPressureTracker

Tracker 为每个 pressure set 维护 current 和 peak，并返回 candidate 的 delta、projected、projected excess、weighted delta。模型 limit 为空时，对应 excess 为 0；当前报告只打印 delta、current 和 peak。

### 11.1 Top 方向

初始化时：

- 所有去重后的 live-in 进入 live set 并计入 current；
- 统计 DAG 内每个 operand 的剩余使用次数，重复 operand 按出现次数计数；
- peak 初始化为 live-in pressure。

评估节点时：

- 某个 live operand 的全部剩余 use 都位于该节点，且它不是 live-out，则减去其 pressure；
- 某个 result 尚未 live，且它是 live-out 或存在 DAG 内 user，则增加其 pressure；
- projected 为 current + delta；weighted delta 使用各 pressure-set weight。

commit 拒绝任一 projected 小于 0 的状态，然后更新 current/peak、递减 operand 剩余 use、移除最后使用且非 live-out 的 operand，并把需要存活的 result 加入 live set。live-out 在 Top 模拟中不会因区域内最后一次使用而移除。

### 11.2 Bottom 方向

初始化时只加入 live-out。反向评估节点时：

- 当前 live 的 result 被定义点消除，减去 pressure；
- 尚未 live 的去重 operand 成为反向 live value，增加 pressure。

commit 更新 current/peak，移除 results，并加入 operands。Bottom Boundary 构造时会实例化该方向的 RegPressureTracker，但当前 pass 不对它评估或提交 candidate。

## 12. Analyze 驱动与报告

### 12.1 函数和 block 遍历

pass 对模块中的函数执行 IR-order walk，收集函数内的 `pto.vecscope` 和 `pto.strict_vecscope`。没有 vecscope 的函数直接跳过。每个有 vecscope 的函数拥有独立 coverage 和从 0 开始的 block index；block index 按 vecscope 出现顺序以及其内部嵌套 region 的深度优先顺序递增。region index 在每个 block 内重新从 0 开始。vecscope 容器本身、函数入口 block 和 vecscope 外 block 均不分类、不计数。

每个模块的报告先写入字符串，再在互斥锁保护下整体写入标准错误，避免并行的嵌套模块 pass 发生字符级交错。单个模块内的内容顺序稳定；多个并行 sibling 模块之间的整体先后顺序不作保证。

### 12.2 Region 报告

每个 region 依次输出：

1. boundary reason、node/edge/live-in/live-out 数量；
2. Top/Bottom 初始 ready 数；
3. known/unknown sched-class 数；
4. node 的原始下标、操作名、语义类、sched class、depth、height；
5. edge 的端点、kind、strength、latency、reason；
6. 原始顺序的 issue/pressure 模拟；
7. 逐周期 issue/resource timeline。

### 12.3 原始顺序模拟

模拟严格按 `dag.getUnits()` 的原始顺序处理，不从 Boundary available 集合选点，也不调用 Boundary commit。流程为：

```text
top.resource.evaluate(unit, requestedCycle)
top.hazard.check(unit, Top, resource.earliestCycle)
top.pressure.evaluate(unit)
top.resource.commit(unit, max(resourceCycle, hazardCycle))
top.pressure.commit(unit)
top.hazard.commit(unit, Top, cycle)
```

`requestedCycle` 初始为 0，每次更新为上一节点的实际 issue cycle，而不是 `cycle + 1`。因此 structural class 可以与下一条有效指令同周期；issue width 或 resource 冲突会把下一条指令推迟。

该 timeline 只反映原始顺序下的 issue/resource reservation 和 Top pressure 变化。它不消费 DAG ready 状态，也不把 edge latency 转换为 candidate ready cycle；depth/height 仅单独报告。因此 timeline 不是依赖感知的调度结果，也不是硬件周期预测。

若 resource/hazard evaluate 拒绝节点，输出 `fallback=tracker-rejected`；若 commit 失败，输出 `fallback=tracker-commit-failed`，并终止该 region 的模拟。DAG 出现环时输出 `fallback=dag-cycle` 并跳过该 region。以上情况不会修改 IR，也不会使 pass 整体失败。非法 mode、缺失 target 或有效 target 不是 A5 会使 pass 失败。

### 12.4 Coverage 报告

每个函数最后输出四类 operation 总数和 unclassified 总数，以及排序后的 `unsupported-op` 和 `unclassified-op` 明细。unclassified 是分类完整性元数据，其 operation 已计入 `SchedulingBoundary`，因此不构成第五种 scheduling class。unknown sched class 不在该明细中，而是在 region 汇总的 `unknown-classes` 和节点 `known=false` 中体现；其保序原因由 artificial edge 输出。

## 13. 当前保证与限制

当前实现保证：

- `off`、`analyze`、`on` 均不改变 IR；
- 只分析 `pto.vecscope` 和 `pto.strict_vecscope` 内部，不跨 vecscope；
- region 不跨 block，也不跨 scheduling boundary；
- DAG 中所有当前 correctness edge 都是 Must edge；
- unknown 语义通过 boundary 隔离，unknown model class 通过相邻 artificial edge 保序；
- A5 Vector 模块报告可稳定复现到模块粒度。

当前没有实现：

- candidate 构造、优先级策略、top-down 或 bidirectional list scheduling；
- schedule result、permutation/semantic verifier、model replay、rollback 和 IR apply；
- 依赖 ready cycle 与资源 timeline 的联合推进；
- 非空 hazard 规则、bundle/pair、bank conflict、NOP 或 software pipelining；
- resource group/buffer、read advance/bypass、pressure limit 的有效模型数据；
- 跨 block 调度和 Cube kernel 调度。

## 14. 回归覆盖

当前 lit 覆盖：

- `off`、`analyze`、`on` 都保持 IR 不变，以及非法 mode；
- 只报告 `pto.vecscope`/`pto.strict_vecscope` 内操作，忽略作用域外操作和无 vecscope 函数；
- CLI 默认 off 与显式 off 等价，真实 A5 VPTO emission coverage 无 unsupported/ unclassified op；
- SSA、memory、静态区间、volatile、post-update 地址依赖；
- SPR/CTRL 的 Data、Anti、Output edge；
- vecscope 内 `mem_bar` 的完整 barrier 顺序；
- pipeline、buffer-id、system、cache 和 SIMT 同步操作的 vecscope verifier 与自动推断边界；
- 未显式分类的 operation 采用安全 boundary，并由函数 coverage 报告为 unclassified；
- CV 分裂时只报告 Vector 子模块。
