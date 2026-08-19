# ADR-0002: VMI VF Fusion 采用分层资源控制与后端反馈

- 日期: 2026-08-09
- 状态: Proposed
- 关联: ADR-0001 VMI VF Fusion canonical pipeline

## Context

VMI VF Fusion 已经能够在统一 PTODSL backend 中完成 candidate 选择、
FusionRegion 生成、row-loop fusion、UB load/store forwarding、layout assignment 和
VMIToVPTO。现有资源控制只在 `PTOVmiLoopFusion` 合并多个主循环前估算 VMI SSA
值的峰值 physical-vector chunk 数。

A5 验证表明，仅限制 loop fusion 不能解决 vector-function stack overflow：

| 验证项 | 结果 |
|---|---:|
| 历史 stack-overflow 用例 | 14 |
| `emit-vpto` 成功 | 14/14 |
| 最终 VPTO 残留 VMI | 0/14 |
| A5 device object 成功 | 0/14 |
| 后端 stack object size | 8480B 至 37152B |
| 后端 Vector Slots | 33 至 145 |

两个诊断用例在关闭 PTOAS VMI loop fusion 和 load/store elision 后仍然超限：

| 用例类型 | PTOAS 形态 | Stack | Vector Slots |
|---|---|---:|---:|
| QK/PV | VMI candidate only | 8480B | 33 |
| Quant/Convert | VMI candidate only | 24864B | 97 |

相同输入使用 ordinary PTODSL/VPTO candidate 时能够生成 A5 object。因此主要问题
不是输入、同步配置或 A5 工具链失效，而是单个 VMI candidate、wide logical vreg
物化、vecscope 调度及后续后端 lowering 已经产生过高压力。即使将 loop-fusion
预算降到 1，结果也不会改变。

`6144B / 256B = 24` 不能解释为 A5 只有 24 个物理向量寄存器，也不能直接作为
VMI fusion 的统一阈值。Bisheng 报告的 stack object 包含寄存器分配后的 spill
对象；VMI 层必须区分 logical wide-vreg、physical chunks、寄存器类别、live range
和后端保留资源。

## Requirements

1. 单个 VMI candidate 必须先满足资源可行性，才能进入 FusionPlan。
2. 无法证明资源可行的 candidate 必须稳定回退到 ordinary PTODSL lowering；资源
   回退不能导致整个 TileOp lowering 失败。
3. Fusion planner 必须支持按资源预算分段，不能只有“全部融合”和“全部不融合”。
4. Load/store forwarding 必须考虑延长 live range 的代价，允许只消除部分 UB
   往返。
5. wide logical vreg 语义继续保留；降低压力应在后期采用 physical-chunk scheduling，
   不能让 TileLib candidate 重新生成运行时 physical-chunk 内层循环。
6. unknown layout、mask、动态范围、无法规范化的地址或不精确资源估计必须保守
   fallback，不能猜测为安全。
7. 资源优化失败必须保持正确的 unfused/ordinary IR，不能造成新的 verifier、object
   或 runtime failure。
8. 默认 pipeline 不依赖在线调用 Bisheng；后端 compile probe 只用于离线校准和
   可复用缓存。

## Decision

### 1. 使用三层资源控制

```text
Candidate resource guard
  -> Fusion and forwarding resource planning
  -> Backend feedback and model calibration
```

第一层在 candidate 锁定前判断单实现是否可行；第二层控制多 candidate 融合及
forwarding；第三层使用真实 Bisheng 数据校准前两层模型。

统一使用以下分析结果，不再让不同 pass 各自定义压力含义：

```text
VMIResourceEstimate {
  peak_vector_chunks
  persistent_vector_chunks
  temporary_vector_chunks
  loop_carried_vector_chunks
  peak_vector_values
  estimate_exact
  rejection_reason
}
```

估算采用 SSA live interval，并按 element type、logical lanes 和 layout 映射到
physical chunks。思想参考 LLVM `RegPressureTracker`，但 VMI 分析不假设 LLVM
virtual register 与 A5 physical vector register 一一对应。

### 2. Candidate resource guard 位于 FusionPlan 前

每个 VMI TileLib candidate 提供 shape-dependent resource contract：

```text
logical vector inputs and outputs
temporary vector values
persistent accumulators
physical chunk arity
chunk-streaming capability
estimate confidence
```

`SelectTemplateCandidate` 同时保留 ordinary fallback。VMI candidate 超预算或估计
不精确时，选择 ordinary candidate，并记录：

```text
pto.vmi.fusion.boundary = "local"
pto.vmi.fusion.rejection_reason = "resource_pressure"
```

ordinary fallback 不进入 VMI loop fusion，但仍在统一
`--tile-lib-backend=ptodsl` 中完成 lowering。

### 3. Fusion 使用确定性的资源分段

Fusion planner 按 DFG 拓扑顺序增量加入 candidate。每加入一个 candidate，模拟：

```text
loop fusion
  -> same-iteration store/load forwarding
  -> canonicalization
  -> live-range pressure
```

超过预算时结束当前 segment，并从当前 candidate 创建下一个 segment。首版使用
确定性的 greedy partition，不引入多个性能 candidate 竞争或 autotuning。

hard boundary、local boundary、未知 alias、同步、mask 不兼容和跨迭代依赖仍然具有
更高优先级，资源模型不能放宽任何 correctness legality。

### 4. Mem2Reg 改为 pressure-aware forwarding

每个 store-load forwarding 单独评估：

```text
合法且 forwarding 后压力不超预算 -> 转为 SSA
合法但延长 live range 后超预算    -> 保留 UB store/load
不合法                             -> 拒绝 forwarding
```

允许通过少量 UB reload 缩短 live range。region 外可观察 store、mask obligation、
byte-range alias 和 iteration-domain 规则保持不变。

### 5. 在后期执行 physical-chunk scheduling

VMI IR 保持一个 logical row 对应一次主循环 iteration。LayoutAssignment/VMIToVPTO
阶段将 elementwise、broadcast 和 convert 链优先改为 chunk-major 静态调度：

```text
load chunk 0 -> compute chain -> store chunk 0
load chunk 1 -> compute chain -> store chunk 1
...
```

避免先物化一个 wide value 的所有 chunks，再物化下一 wide value。physical chunks
可以静态展开，但不能生成改变 TileLib 主循环协议的运行时 chunk loop。

Reduce 需要单独维护 partial accumulator 和最终归约 phase；不能直接套用普通
elementwise chunk streaming。

### 6. 使用离线 Bisheng feedback 校准

对代表性 `(candidate, shape, dtype, layout, fusion signature, schedule)` 编译 device
object，并采集：

```text
Vector Slots
Total Spilled Byte Size
stack object size
object success/failure
```

结果按 compiler SHA 和完整 code-shape key 缓存。该机制用于回归测试、模型校准和
阈值选择，不进入默认用户编译关键路径。工程思路参考 Triton 的编译后资源反馈和
XLA GPU 的 fused/unfused 成本比较。

## Pipeline

目标流水线调整为：

```text
InsertTemplateAttributes
  -> SelectTemplateCandidate + CandidateResourceGuard
  -> FusionPlan / OpScheduling / FusionRegionGen
  -> ExpandTileOp / InlineTileLib
  -> VMIResourceAnalysis
  -> PressureAwareVMILoopFusion
  -> PressureAwareVMILoadStoreElision
  -> VMIPhysicalChunkScheduling
  -> VMI LayoutAssignment
  -> VMIToVPTO
```

不新增 TileLib backend。VMI VF Fusion 仍由现有 `--enable-vmi` 与
`--enable-op-fusion` 控制。资源预算相关选项在模型校准期间保持诊断用途，不能在
没有 A5 数据支撑时改变生产默认值。

## Delivery Plan

### PR1: Observability and safe candidate fallback

- 建立统一 `VMIResourceEstimate` 和稳定 remarks。
- 添加 candidate resource contract。
- 超预算 candidate 自动 ordinary fallback。
- 使历史 14 个用例全部通过 A5 object gate。

### PR2: Physical-chunk scheduling

- 支持 elementwise、broadcast 和 convert 的 chunk-major lowering。
- 减少 ordinary fallback，恢复 VMI candidate 覆盖率。
- 为 reduce 建立独立资源模型，不在本 PR 强行流式化。

### PR3: Pressure-aware fusion partition

- 用增量 greedy 算法划分 fusion segments。
- 保持 alias、boundary、mask 和跨迭代 legality。
- 输出稳定的接受/拒绝原因。

### PR4: Pressure-aware forwarding

- 对每个 store-load forwarding 计算压力增量。
- 支持部分 forwarding 和必要 reload。
- 保证无新增 stack overflow 或可观察 store 删除。

### PR5: A5 calibration and acceptance

- 固化 Bisheng compile-probe cache 与报告生成。
- 完成 DSv4 120 用例 compile gate。
- 完成关键 VF 子图的固定输入正确性和串行性能采样。

## Acceptance Criteria

基础 gate：

- PTOAS lit、PTODSL Python tests 和 VMI template tests 全部通过。
- DSv4 120/120 `emit-vpto`，最终 VPTO 残留 VMI 为 0。
- 历史 14 个 stack-overflow 用例 14/14 生成 A5 device object。
- 不支持或资源不确定的 VMI candidate 稳定 ordinary fallback。

正确性 gate：

- Fusion 失败保持可执行的 unfused IR。
- local/hard/resource boundary 不被穿透。
- mask、tail、dynamic valid shape 和未知 alias 继续保守处理。
- A5 固定输入下 ordinary 与 VMI 输出满足既有数值容差。

性能 gate：

- Softmax、RoPE、RMSNorm、Decode/Prefill Sinkhorn 分别报告 ordinary、
  candidate-only、loop-fused 和 forwarding-enabled 四种形态。
- object 中不得出现 stack overflow；spill 增加必须明确报告，不能只看时延。
- 同时报告 candidate 覆盖、FusionRegion 数、loop 数、VLD/VST、Vector Slots 和
  stack bytes，避免将 candidate code shape 收益误报为 loop fusion 收益。

## Alternatives

### A. 只给 loop fusion 设置固定 24-chunk 阈值

不采用。A5 数据证明 candidate-only 已可能达到 33 至 145 Vector Slots，且
`6144B / 256B` 不是物理寄存器数量。

### B. 所有高压力 VMI candidate 永久回退 ordinary

只作为第一阶段 correctness fallback。它能恢复 object 可编译性，但会失去 VMI
覆盖和融合机会，不能作为最终性能方案。

### C. 在 TileLib candidate 内生成 physical-chunk 运行时循环

不采用。它会破坏 logical-row 主循环协议并增加后续 loop fusion 的复杂度。chunk
scheduling 应位于较晚 lowering 阶段，并优先静态展开。

### D. 每次用户编译都调用 Bisheng 探测最优 fusion

不采用。编译开销和环境依赖不可接受；backend feedback 只用于离线校准、缓存和
持续集成。

## Consequences

### Pros

- 单 candidate、融合和 forwarding 三类压力来源可以独立归因。
- correctness fallback 与性能优化解耦，无法优化时仍能稳定 lowering。
- 保留 wide logical vreg 的可分析语义，同时允许后期降低物理寄存器压力。
- A5 后端数据能够持续校准模型，而不是依赖未经验证的固定阈值。

### Costs and Risks

- Candidate metadata 与实际 template 必须保持一致，需要协议测试。
- 模拟 forwarding 的 live range 比当前 loop-only 估算复杂。
- Chunk-major scheduling 必须正确处理 layout、mask、post-update 和 reduce phase。
- 资源回退可能暂时降低 VMI 覆盖率，必须在报告中区分 correctness fallback 与性能
  回退。

## Open-source References

- LLVM `RegPressureTracker` and `MachineScheduler`: live-range pressure and
  pressure-aware scheduling.
- Triton compiler: backend resource reporting and configuration feedback.
- XLA GPU performance model: fused versus unfused cost comparison.
- Halide autoschedulers: storage, working-set and recomputation tradeoffs.

这些实现只作为算法和工程流程参考，不引入新的运行时依赖。
