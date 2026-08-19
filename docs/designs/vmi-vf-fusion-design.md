# VMI VF Fusion RFC 最小实现设计

## 1. 文档状态

- 状态: Proposed
- 日期: 2026-07-15
- 决策记录: [ADR-0001](adr/0001-vmi-vf-fusion-rfc-minimal-pipeline.md)
- 参考设计: [上游 VMI VF Fusion RFC](https://github.com/WenboCodes/PTOAS/blob/new-vf-fusion-design/docs/new-vf-fusion-design/RFC-vf-fusion-on-vmi.md)

本文定义 PTOAS `feature-vmi` 分支上第一阶段 VMI VF Fusion 的编译器边界和 pass
协议。第一阶段只建立 RFC 的正确性闭环，不引入多 candidate、算法专项 schedule、
cost model 或 autotuning。

## 2. 目标与非目标

### 2.1 目标

1. `PIPE_V` TileOp 通过 PTODSL VMI TileLib 展开为独立正确的 canonical VMI 实现。
2. Expand 和 Inline 后，从真实 VMI IR 中识别由 TileLib 产生的独立 fusion unit。
3. 保守合并结构兼容的相邻 VMI 循环。
4. 在融合后通过 VMI mem2reg 消除可证明安全的中间 UB store-load。
5. 利用 1VL inner contract，把同一 row 的 Reduce -> Broadcast -> Elementwise ->
   Reduce/Convert phases 合并进一个 row runtime loop，支撑 FA/Online Softmax 深融合。
6. VMI 模板和 Fusion 保持 scope-free；`VMIToVPTO` 后由现有
   `PTOInferVPTOVecScope` 统一推断物理 VPTO vecscope。
7. 任何分析失败均保持原独立实现，不能改变程序语义。

### 2.2 非目标

- 不优化任意用户手写 VMI 或任意用户控制流。
- 不为一个 TileOp 注册多个 schedule candidate。
- 不做 candidate selection、candidate locking 或 region-aware specialization。
- 不做多 VL inner schedule、RowMax/ColMax 选择、代价模型或自动调优。
- 不要求首期支持 physical inner 非 1VL 或动态 tail 的 Reduce/Broadcast 深融合。
- 不替换 VMI layout assignment、物理 vreg 分配或 VMI-to-VPTO lowering。

## 3. 当前基础与缺口

当前已有链路为：

```text
TileOp
  -> ExpandTileOp(--tile-lib-backend=ptodsl-vmi)
  -> func.call @__pto_ptodsl_vmi_...
  -> PTOInlineLibCall
  -> 多个独立 scf.for + pto.vmi.*
  -> FoldTileBufIntrinsics
  -> VMI semantic/layout pipeline
  -> VPTO
```

现有 canonical VMI TileLib 已能让静态 Softmax compute harness 中的一组基础 TileOp
独立 lower，并已打通静态 `[64,64]xf32` FlashAttention 多项式样例。该样例不包含标准
Softmax 的 RowReduce 归一化，尚不代表完整 FA/Online Softmax 已覆盖。Expand + Inline
后每个 TileOp 仍有自己的循环。例如 `tadd -> texp` 会得到：

```mlir
scf.for %i = %c0 to %blocks step %c1 {
  %a = pto.vmi.vload %src0[%off_i] ...
  %b = pto.vmi.vload %src1[%off_i] ...
  %x = pto.vmi.vadd %a, %b, %mask ...
  pto.vmi.vstore %x, %tmp[%off_i], %mask ...
}
scf.for %j = %c0 to %blocks step %c1 {
  %x = pto.vmi.vload %tmp[%off_j] ...
  %y = pto.vmi.vexp %x, %mask ...
  pto.vmi.vstore %y, %dst[%off_j], %mask ...
}
```

缺口不是 VMI op lowering，而是：

- Inline 后缺少稳定的 TileLib fusion-unit provenance。
- 没有 VMI 循环兼容性、依赖和 alias 分析。
- 没有 VMI loop merge。
- 没有 fusion-after mem2reg。

## 4. Canonical VMI TileLib 协议

### 4.1 唯一实现

RFC 模式下，每个 `(target, op)` 必须恰好注册一个 canonical VMI `TileTemplate`：

```text
(a5, tadd) -> exactly one canonical VMI implementation
(a5, texp) -> exactly one canonical VMI implementation
```

同一个模板可以根据静态 dtype/shape 做合法特化，但 provider 不在多个模板之间进行
性能选择。零个实现是 coverage error；多个实现是 provider contract error。

provider module 必须暴露独立的 `VMI_TILELIB_REGISTRY`，canonical 模板通过
`@canonical_vmi_template` 显式注册。helper 只查询该 registry，不通过扫描 module
全局变量发现模板，也不与普通 PTODSL TileLib 的 VPTO/MI registry 混用。

### 4.2 独立正确性

每个 canonical 实现必须包含完整 load/compute/store，在没有 fusion pass 时也能正确
lower 到 VPTO。Fusion 只能删除已证明冗余的控制和访存，不能成为 TileOp 正确执行的
前置条件。

### 4.3 单主循环与 1VL 调度

一个 canonical fusion unit 应具有一个主 `scf.for`。用户切分的 dense vector Tile
必须满足物理 inner extent 等于 candidate iteration contract 的一个 VL；f32 主链即
`cols=64`。每个 row 只有一个 logical block，因此模板只生成 row 主循环，不生成
`for col`，也不把多 VL inner tile 平坦化为更多 logical blocks：

```text
physicalInner = logicalLanes
blocksPerRow = 1
logicalBlockCount = rows
```

物理 inner 固定为 1VL 不等于所有 lane 必须有效。后续动态支持允许
`0 < validInner <= physicalInner`，并在同一个 block 内使用 tail mask；当前实现仍只
支持静态 `validInner == physicalInner`。`cols < VL` 和 `cols > VL` 都不属于首期
canonical candidate，前端应按 1VL 重新切 Tile。

Reduce 的 `[rows,1]` compact 结果是辅助 Tile，不是 dense iteration Tile；Convert
沿用源 iteration contract，例如 `64xf32 -> 64xf16` 仍是同一个 logical block，不能
按目标 dtype 的物理容量重新拆分。后续性能优化只能 unroll outer rows，不能恢复多 VL
inner schedule。若 BR 小于 VL，compact state reshape 后的 `[1,BR]` 必须使用显式
compact-domain candidate，或 pad 到 1VL 并携带 valid mask；通用 dense candidate
不能因此放宽为接受任意 `cols<VL`。

### 4.4 不暴露物理 layout

canonical 模板只能表达 surface VMI 类型和语义 op。以下内容不能成为模板选择维度：

- contiguous / deinterleaved physical layout
- physical vreg 编号和数量
- MI post-update 地址模式
- interleave/pack/materialization 指令序列

这些信息继续由 VMI semantic/layout pipeline 决定。

## 5. 只处理 TileLib 代码的 provenance 协议

当前 ExpandTileOp 已给实例化函数添加：

```mlir
attributes {pto.tileop.instance = "ptodsl-vmi"}
```

设计实现时，ExpandTileOp 还应给实例化函数或 call 添加显式 TileOp 名称，不能依赖
解析私有函数名恢复来源。函数 inline 后 function/call attribute 会消失，因此
`PTOInlineLibCall` 需要在 inline PTODSL VMI provider function 时，把来源信息转移到
canonical unit 的主 `scf.for`：

```mlir
scf.for ... attributes {
  pto.vmi.fusion.source = "tilelib",
  pto.vmi.fusion.tileop = "tadd",
  pto.vmi.fusion.unit_id = 7 : i64
}
```

这些属性是编译器内部 provenance，不是 candidate lock，也不承诺一定融合。
`VMIIdentifyFusionUnits` 必须同时验证 marker 和结构协议；只有 marker 没有 canonical
结构，或只有结构没有 marker，都不能进入首期融合。

## 6. Fusion unit 与宽松分组

### 6.1 Fusion unit

一个首期 fusion unit 是满足以下条件的 TileLib 代码段：

- 带合法 provenance。
- 包含一个主 `scf.for`。
- 不包含显式 `pto.vecscope`；物理 vecscope 由 emission boundary 的现有 pass 推断。
- 主循环 step 可证明一致，首期要求正的常量 step。
- 循环体中的 VMI memory effect 可枚举。
- unit 外 setup 仅包含常量、mask、pointer/address 计算等可分析操作。
- 不包含未知 call、barrier、sync、DMA、Cube 或未知副作用。

### 6.2 宽松分组

`VMIPlanLoopFusion` 可先收集同一 block 内、两个硬边界之间的一段连续 fusion units，
形成宽松分组。宽松分组只表示“可以共同分析”，不表示组内所有循环都必须合并。

随后按相邻循环逐对检查兼容性，得到一个或多个真正的 fusion groups：

```text
[unit0, unit1, unit2, unit3]
      loose analysis span

unit0 ~= unit1    -> fusion group A
unit2 incompatible -> standalone
unit3             -> standalone or later group
```

这样可以支持部分融合，同时不需要在 Tile 层预先构造 `pto.fusion_region`。

## 7. Pass 设计

### 7.1 `VMIIdentifyFusionUnits`

类型：analysis / validation pass，建议作用于 `func::FuncOp`。

职责：

- 查找 TileLib provenance marker。
- 验证 canonical unit 结构。
- 记录主循环、setup、storage accesses、mask uses 和硬边界。
- 为诊断输出稳定的拒绝原因。

它不做 group selection，不修改循环，也不重新选择 TileOp 实现。跨 pass 的分析结果
必须通过 AnalysisManager 缓存或显式、可打印的 IR metadata 共享，不能使用隐藏全局
状态；若采用 AnalysisManager，`VMIPlanLoopFusion` 应直接请求同一个
`VMIFusionUnitAnalysis`，而不是依赖前一个 pass 的进程内副作用。

### 7.2 `VMIPlanLoopFusion`

类型：analysis + metadata pass，建议作用于 `func::FuncOp`。

职责：

- 在 block 内构建宽松分组。
- 对相邻 unit 做兼容性检查。
- 生成确定性的 group id/order 或 analysis result。
- 保持原程序顺序，不做算法级重排。

可复用现有 TileFusion 的 block-local DFG、liveness、iteration-domain 代码思路；若直接
复用代码，应抽取与具体 TileOp 类型无关的 utility，而不是让 VMI pass 消费
`FusionPlan` 的 TileOp metadata。

### 7.3 `VMIFuseCompatibleLoops`

类型：transform pass，建议作用于 `func::FuncOp`。

职责：

- 为 fusion group 创建一个共享 `scf.for`。
- 将后续循环 IV 映射到第一个循环 IV。
- 按原程序顺序克隆/移动循环体。
- 保留每个 op 的 mask、属性和内存顺序。
- 删除被合并的旧循环。

首期建议只支持 resultless `scf.for`。带 `iter_args`、跨迭代 accumulator 或复杂 region
branch 的循环先保守拒绝，作为 Reduce 深融合阶段扩展。

### 7.4 `VMIMem2Reg`

类型：transform pass，建议作用于 `func::FuncOp`。

必须在 loop fusion 后运行。首期处理同一融合循环、同一迭代中的：

```mlir
pto.vmi.vstore %x, %tmp[%off], %mask
...
%reload = pto.vmi.vload %tmp[%off]
```

当 location、值 shape、访问覆盖范围和 mask obligation 均可证明兼容时：

```mlir
// %reload users 改用 %x
// 删除冗余 vstore/vload
```

如果中间 Tile 仍有 fusion group 外用户，或 store 可能被其他访问覆盖，则不能删除
对外可观察的 store。跨迭代 promotion 到 `scf.for iter_args` 属于后续扩展。

### 7.5 VecScope 插入边界

不新增 VMI vecscope coalescing pass。canonical VMI 模板和 Fusion transform 都不显式
生成 `pto.vecscope`，只维护合法的 SCF、VMI SSA 和内存顺序。现有
`PTOInferVPTOVecScope` 在 `VMIToVPTO` 之后、LLVM emission 之前基于物理 VPTO 操作自动
划分 resultless vecscope；DMA、sync、barrier 和无法安全移动的操作继续作为其边界。

## 8. 兼容性与安全判据

两个 unit 只有全部满足以下条件才可合并。

### 8.1 控制边界

- 位于同一 block，且保持原顺序。
- 中间没有 call、sync、barrier、DMA、Cube、未知副作用或 region boundary。
- 首期不跨 `scf.if`、外层 `scf.for` 边界移动 unit。

### 8.2 迭代域

- lower、upper、step 相同 SSA，或可由简单 affine/canonical expression 证明等价。
- 动态 shape 可以支持，但两个循环必须共享同一动态 bound SSA 或可证明等价。
- trip count 不同则不融合。当前 `trowmax` 的 row loop 与 elementwise 的 flattened
  block loop 因此通常保持独立。

### 8.3 Alias 与依赖

VMI load/store 的首期 location key 定义为：

```text
LocationKey = (
  storage root,
  normalized linear offset,
  per-iteration accessed span,
  VMI value shape and element type,
  dist/group/block-stride mode
)
```

`storage root` 需要穿透 `tile_buf_addr`、合法 cast 和可规范化 addptr；offset 只处理
常量、IV 和简单 affine arithmetic。规则为：

- 可证明 NoAlias：允许保持顺序后融合。
- 精确 RAW：允许融合，mem2reg 可进一步判断是否提升。
- WAW/WAR：只有保持顺序且可证明每迭代访问关系安全时允许。
- MayAlias 或无法规范化：保守拒绝。
- 不允许把同一迭代依赖误判成跨迭代依赖，反之亦然。

现有 VMI 使用线性 offset；后续若引入 shaped pointer / multidimensional index，可替换
LocationKey 的构造方式，不改变 pass 顺序和保守原则。

### 8.4 Mask

- 融合不能丢失任何 consumer mask。
- A5 `vload` 不可谓词化，promotion 后 consumer 的 mask obligation 仍存在。
- store 的 mask/pmode 与 load 后所有 consumer 的有效 lane 关系无法证明时，不做
  mem2reg。
- 动态 tail mask 只要由同一 bound/remaining SSA 推导且逐 use 保留，可以参与融合；
  首期 provider 尚未生成动态 valid-shape tail，因此先完成静态 mask 用例。

## 9. Pipeline 顺序

VMI provider 的目标顺序为：

```text
ExpandTileOp(--tile-lib-backend=ptodsl-vmi)
  -> PTOInlineLibCall
  -> FoldTileBufIntrinsics(shape-only)
  -> VMIIdentifyFusionUnits
  -> VMIPlanLoopFusion
  -> VMIFuseCompatibleLoops
  -> canonicalize / CSE
  -> VMIMem2Reg
  -> canonicalize / CSE
  -> FoldTileBufIntrinsics(addr-only)
  -> existing VMI semantic/layout pipeline
  -> VMIToVPTO
  -> existing PTOInferVPTOVecScope at the VPTO emission boundary
```

关键顺序约束：

- Expand + Inline 之前看不到真实 VMI 循环，不能做 VMI loop compatibility 分析。
- shape-only folding 先暴露静态/动态 loop bound。
- mem2reg 必须在 fusion 之后，才能看到原本位于不同循环体的 store-load。
- layout assignment 必须在 fusion/mem2reg 之后，避免物理 layout 细节污染判据。
- addr-only folding 放在分析之后，以保留 Tile handle/storage provenance；分析需要能
  追踪 `tile_buf_addr` 的 root。
- vecscope inference 必须在 VMI 物理化之后统一运行，VMI Fusion 不分析或合并 scope。

## 10. 与现有 Fusion pipeline 的关系

### 10.1 不直接复用的 pass

- `FusionPlan` / `FusionRegionGen`：输入是 Tile-native PTO IR。
- `PTOLowLevelLoopFusion`：输入是已经展开到 VPTO/MI 的低层循环。
- `PTOFusionLoadStoreElision`：不是 fusion-after 的 VMI SSA promotion。

### 10.2 可以复用的能力

- block-local DFG 构建框架。
- value liveness、external user、write-instance escape 的建模思路。
- iteration-domain equivalence 的部分 solver/utility。
- 确定性的 group id/order、打印和测试方式。

### 10.3 CLI 路由

当前 `--tile-lib-backend=ptodsl-vmi --enable-op-fusion` 会报错，防止误入 legacy
VPTO fusion。新 passes
完成后，`--enable-op-fusion` 应按 provider 路由：

```text
--tile-lib-backend=tilelang   -> existing Tile/VPTO fusion lifecycle
--tile-lib-backend=ptodsl-vmi -> new VMI fusion lifecycle
```

在 VMI pipeline 可用前应保留当前拒绝逻辑。

## 11. 失败与 fallback

- 缺少 VMI implementation：ExpandTileOp 明确报 coverage error，不静默回退到 MI。
- 同一 `(target, op)` 存在多个 VMI implementation：provider contract error。
- unit 不符合 canonical 结构：保持独立，输出可诊断拒绝原因。
- loop/domain/alias/mask 无法证明：不融合。
- mem2reg 无法证明：保留原 store/load。
- 任一 unit 独立 lowering 必须始终有效。

部分融合示例：

```text
tadd(loop=rows) + texp(loop=rows) + trowmax(loop=rows, reduce phase)

首期结果：
  依赖和 mask 兼容时，[tadd + texp + trowmax] 合并到同一个 row loop
```

这不是错误，而是 RFC 保守闭环的预期行为。

## 12. 验证计划

### 12.1 正向 lit tests

- 两个相邻 elementwise canonical loops 合并为一个 `scf.for`。
- 三个 elementwise loops 连续合并且保持 op 顺序。
- 同 location、同 offset 的中间 `vstore -> vload` 被 mem2reg 消除。
- 1VL RowMax -> Broadcast -> Exp -> RowSum/Convert 在同一个 row loop 内完成。
- 动态 upper bound 使用同一 SSA 时可融合。
- VMI Fusion 输出不包含显式 `pto.vecscope`，最终 VPTO emission 自动推断合法 scope。
- 最终 VMI-to-VPTO 编译通过且不残留 `pto.vmi.*`。

### 12.2 负向 lit tests

- 用户手写、无 provenance 的 VMI loop 不处理。
- trip count、step 或 offset mapping 不一致时不融合。
- 中间存在 sync/call/unknown side effect 时不融合。
- MayAlias、WAW/WAR 无法证明时不融合。
- mask/pmode 不兼容时不做 mem2reg。
- 中间 Tile 有 group 外用户时保留必要 store。
- dense Tile 的 physical inner 小于或大于 1VL 时拒绝 canonical candidate。

### 12.3 端到端基线

- 现有 PTODSL VMI TileTemplate Python test 保持通过。
- composite provider 和 no-vector-fallback lit tests 保持通过。
- 静态 Softmax compute-op coverage harness 完成 Expand、Inline、VMI-to-VPTO。
- 静态 `[64,64]xf32` FlashAttention 多项式样例完成 Expand、Inline、VMI-to-VPTO。
- 固定 1VL inner 的 FA/Online Softmax 用例补齐 compact-state candidate 后纳入 M3 验收。
- M3 要求 RowMax -> Broadcast -> Exp -> RowSum/Convert 关键链在同一 row loop 内深融合；
  动态 tail 和任意 Shape 泛化不作为门槛。

## 13. 后续迭代

基本闭环稳定后，再分别设计和评审：

1. 超出 canonical 1VL row 模式的 `iter_args` Reduce/accumulator loop fusion。
2. 非 canonical Shape 的 Reduce -> Broadcast -> Elementwise 泛化。
3. 动态 valid shape 和 tail mask 完整覆盖。
4. 多 canonical schedule candidate 与 region-aware selection。
5. outer-row unroll、重读/保活选择和 physical vreg pressure cost。
6. FA/Softmax 专项 schedule、cost model 和性能验收。
