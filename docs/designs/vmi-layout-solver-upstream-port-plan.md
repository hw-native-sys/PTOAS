<!--
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
-->

# 升级方案：把带代价的 VMI layout solver 迁移到当前上游底座

目标：**我们的 solver**（带代价的 `VMILayoutPlan` planner + conflict solver + cost model）继续作为
layout 决策引擎，整体搬到 `origin/master` 上（比我们的分叉点领先 269 个 commit）。
上游自己的决策链路不迁移，只复用它的**事实表**和**纯分析**。

## 1. 现状基线

| 项目 | 数值 |
|---|---|
| 我们的分支 | `feature/vmi-layout-decision-layers` @ `e5bd4511`（比分叉点领先 4 个 commit） |
| 分叉点 | `ca7ccb409664` |
| 上游 | `origin/master`，**领先 269 个 commit**，与我们 diff 2399 个文件 / +254269 / -158438 |
| 我们独有的新文件 | `VMILayoutPlanner.{h,cpp}`、`VMILayoutCostModel.{h,cpp}`、`VMILayoutConflictSolver.{h,cpp}`、`VPTOPack4StoreMaskNormalize.cpp`、2 篇设计文档、约 42 个 lit 测试 |
| 上游 layout 结构 | pass 已迁到 `lib/PTO/Transforms/VMI/`；layout 事实拆成 9 个 `.inc` 表 + pattern DSL；新增 `VMILayoutSpineAnalysis.cpp`（无 pass 注册、生产侧唯一消费者是 `VMILayoutAssignment.cpp`，可整体删除）；`VMIToVPTO.cpp` 拆成约 20 个编译单元 |
| 分叉点之后动过 `VMIAttrs.td` 的 commit | 只有 2 个，都是方言层收敛：`7eda9ab55`（vstore dist mode `dintlv` 改名 `intlv`、去掉 unpack vload）、`8a0a5689b`（pmode 收敛为 zero-only） |
| 动手前必须先做 | **vexpdif layout 层还没提交**（VMILayoutSupport.{h,cpp}、VMILayoutPlanner.cpp、VMIMaskGranularityAssignment.cpp）——先提交它 |

## 2. 已确认的设计决策

1. **事实与偏好分离。** 候选集合（表行）属于事实，跟随上游；preference penalty 仍然是我们决策链里的**靠后 tie-break**（`pen` 排在
   lane-stride 两层之后），所以两棵树在偏好行上的差异不值得对齐。我们的提交
   `00890b827 Prefer E2B layouts` 因此是**删除候选**，而不是合并对象。
2. **Spine 分析整体不要：不保留 `VMILayoutSpineAnalysis`，但保留它背后的那些新 layout 行。**
   上游那三个 collector（`collectDirectionSpineLegs`、`collectSpineScopedCasts`、`collectNarrowSideCompute`）
   存在的唯一理由，是上游没有代价函数：它必须先用 IR 形状把 `<up,down,up,down>` chain 认出来，才敢走
   scope 化的 cast 行。**我们的 solver 有代价函数，这个 scope 就是冗余的。** 上游注释把这批行的价值写得很清楚
   （`VMILayoutSupport.h:323-336`）：它们是 16<->32、32<->8 的复合行（`deinterleaved = 4 -> deinterleaved = 4`
   并带显式 lane stride），让窄侧值仍然摊在宽侧的四个物理 part 上，于是窄化与加宽**按 chunk 1:1**、
   不需要 `pto.vor` 拼装。而“要不要 `pto.vor`”正是我们第 2 层（数据耦合次数）在数的东西。所以：
   * **删掉** `include/PTO/Transforms/VMILayoutSpineAnalysis.h`、
     `lib/PTO/Transforms/VMI/VMILayoutSpineAnalysis.cpp`、它的 CMake 行，以及测这个分析本身的 lit。
     它**没有 pass 注册**（`Passes.td`/`Passes.h`/`ptoas` 里都查不到），生产侧唯一消费者就是
     `VMILayoutAssignment.cpp:20`——也就是我们要替换掉的 seed/peephole 那部分，所以删除是外科手术式的。
   * 传播器里 `setSpineScopedCastOps` / `isSpineScopedCast`（`VMILayoutPropagation.h:72-83,126`）一并删掉，
     不再往 propagator 里塞 scope 集合。
   * **两张表照旧枚举，不做 scope 过滤**：把 `kSpineScopedCastLayoutPatterns` /
     `kSpineScopedLegalCastLayoutPatterns` 接成我们 relation provider 的普通候选来源（等价于无条件走
     `...ForLayout` 版本），由第 2 层数据耦合 + 第 3 层 compute-side ls 决定要不要用。这样“新 layout 不能丢”
     得到满足，决策权回到我们的代价链，同时消掉了 scope 与 compute-side 层的重复计数风险。
   * 代价是候选空间变宽（原来被 scope 挡掉的行现在也进候选）。这正好是一次**对代价模型的检验**：
     如果某个 case 在有代价的情况下反而选了会多出 `pto.vor` 的行，那是我们分层的 bug，
     而不是“上游 scope 更聪明”。
3. **过时的表行删掉。** 任何建立在“支持集比上游窄”这个前提上的我们自己的行
   （尤其是为 full-width store 加的 dense store 行、`ls(2) <-> c()` 的 ensure-layout 行），都按上游的
   泛化谓词重新核对，上游已覆盖的删掉。`trunci` 硬失败也正是在这一步随升级消失。

## 3. 升级步骤

| # | 步骤 | 涉及位置 | 规模 | 通过判据 |
|---|---|---|---|---|
| 0 | 提交 vexpdif layout 层；`VMIDIAG` 埋点不进提交 | `VMILayoutSupport.{h,cpp}`、`VMILayoutPlanner.cpp`、`VMIMaskGranularityAssignment.cpp` | 0.5 天 | `lit/vmi_new` 不变（12 failed） |
| 1 | 从 `origin/master` 拉 `feature/vmi-layout-solver-upstream` 到独立 worktree；先确认上游自己能构建 | - | 0.5 天 | 上游 `pto-test-opt` + `lit/vmi_new` 全绿 |
| 2 | 搬运**真正无侵入**的部分：3 个头文件 → `include/PTO/Transforms/`，`VMILayoutConflictSolver.cpp` → `lib/PTO/Transforms/VMI/`，补 CMake | `lib/PTO/Transforms/CMakeLists.txt`、header CMake | 0.5 天 | `VMILayoutConflictSolver.cpp` 零改动编译通过（**已实测验证**） |
| 2b | `VMILayoutCostModel.cpp` 是**半侵入**的：实测只缺 6 个符号——3 个事实字段（`VMIEnsureLayoutFact::forwardsPhysicalParts`、`VMIEnsureMaskLayoutFact::forwardsPhysicalParts`、`VMIGroupStoreLayoutFact::stagingLayout`）+ 3 个查询（`getGeneratedMaskLayoutFact`、`getInterleaveStoreSupport`、`getSameLayoutRelationSupport`）。注意 `forwardsPhysicalParts` 不能只补字段：必须把上游 ensure-layout 事实构造处真正计算它的逻辑一起搬，否则代价模型永远读到 false（那就成了假规则） | `VMI/VMILayoutSupport.{h,cpp}` + `.inc` | 1-1.5 天 | 代价模型编译通过 + ensure-layout 相关一致性用例输出与主树一致 |
| 2c | `VMILayoutPlanner.cpp`（2688 行）与一致性工具**不是**无侵入：planner 依赖 `VMILayoutPropagation.h`（我们的 exact-mode 协议）与 44 处 `VMILayoutSupport::*` | - | - | 只能随步骤 3/4 一起进 |
| 3 | **支持层合并（关键路径）**：采用上游表/DSL；把我们的 9 个枚举查询写成对 `...ForLayout` 的循环；两张 spine-scoped 表按普通候选源**无条件**枚举（不迁 `VMILayoutSpineAnalysis`，删掉它的 .h/.cpp/CMake 行/lit）；只保留我们真正新增的表（`kVexpdifLayoutPatterns` 5 行、`kGeneratedMaskStagingPatterns` 4 行）和我们自己的事实字段（`intrinsicRearrangementCost`、`stagingLayout`、`forwardsPhysicalParts`、`VMIReduceLayoutFact`、`VMIGeneratedMaskLayoutFact`、`VMIInterleaveStoreSupport`、`VMIVexpdifLayoutFact`） | `VMI/VMILayoutSupport.cpp` + `.inc` 表 + `VMILayoutSupport.h` | 4-6 天 | **用上游自己的决策链路跑上游测试仍然通过**（这是“合并行为中性”的证明） |
| 4 | 传播器合并：保留上游传播器的分析入口（**不含** spine scope 集合，`setSpineScopedCastOps/isSpineScopedCast` 删除），加上我们的 `requestExact/installPlanned/endExactRequests/getRequestedLayout(OpOperand&)`；消掉唯一重名符号 `isVMISameLayoutOp`（采用上游的算子集合，因为分类现在来自上游分析） | `VMI/VMILayoutPropagation.cpp` 及头文件 | 1.5-2 天 | 链接干净；class-edge 相关 lit 全绿 |
| 5 | 替换决策：在上游 `applyLayouts()` 里插入我们的 `selectLayoutPlan()` + `commitVMILayoutPlan()`（以及我们的结构化 seeding），删掉那 4 处 seed 调用；保留 `collect/materializeCallBoundaries/insertDataUseMaterializations/rewriteFunctionType/validateVMILayoutAssignedIR` | `VMI/VMILayoutAssignment.cpp`（约 500 行改动） | 2-3 天 | 我们的 32 个一致性测试通过 |
| 6 | 注册：CMake、`Passes.td`/`Passes.h` 里加 `vpto-normalize-packed-store-mask`，在 `tools/ptoas/ptoas_pipeline.cpp` 的 `appendVMISemanticPipeline()` 之后、`prepareVPTOForEmission()` 之前挂上 | 5 个文件 | 0.5-1 天 | `ptoas --emit-vpto` 输出 b8 谓词形式 |
| 7 | 在约 20 个上游单元里按 pattern family 逐个重放 VMIToVPTO 的 delta（单项最大，与步骤 3 并行） | `lib/PTO/Transforms/VMIToVPTO/*` | 4-6 天 | 步骤前后做 pattern class 级别的符号 diff |
| 8 | 测试：搬入我们新增的 42 个测试；手工合并 45 个被修改的（其中 23 个上游也改过）；上游表/下沉变化导致期望变化的重打 | `test/lit/vmi_new/` | 3-4 天 | `lit/vmi_new` 每个失败都有逐条解释 |
| 9 | 验证：全量 `lit/vmi_new + vpto + pto + tile_fusion`，再跑 A5 NPU 验证套件，然后重测放大后的端到端用例（`gbmc-amp-dep`、`truncf-amp2`）以复核决策分层的性能结论 | - | 3-5 天 | 无新增失败；性能结论可复现 |

单人总计 19-28 人日；两人并行（步骤 3 与 7 并行）约 2 周。

## 4. 无法机械搬运、必须重写/重判的部分

1. `VMILayoutRelationProvider::enumerateRelations`（约 1400 行）——按上游的 per-layout API 和 scoped 入口
   重新推导；上游 `...ForLayout` 的语义是唯一事实来源，函数体不能照抄。
2. 缺失的 16 个 `VMILayoutSupport` 方法 + 8 个我们独有的事实类型/字段 + 我们多出的 3 张表——
   要针对上游 DSL 重新设计接口。另外 `getSpineScopedCastLayoutFact*` 三个入口要**降级**成普通
   `...ForLayout` 循环（去掉 scope 参数），这不是“搬运”，而是接口语义的改变。
3. E2B group-broadcast-load 偏好（我们的 `00890b827`）——与上游 `0d7da8454` 冲突，需重新决策并重打
   `vmi_layout_assignment_group_slot_broadcast_load_e2b_b16.pto`、`vmi_extf_8bit_factor_contract.pto`
   （步骤 3 里保留 4 条 “E2B b8 + factor 4” 行是刻意的：那两条测试通过，说明这 4 行忠实反映了当前行为）。
4. 算子类归属（`isVMISameLayoutOp` / `isVMILayoutCastOp`）——要按上游 `isSameLayoutOp` +
   `isVMIClassTransparentOp` 的划分重写。
5. 传播器的 exact-mode 协议——要与上游的 `const` 设计及其分析输入对齐。
6. 两套决策机制不能共存：我们已死的 seed 机制和上游在用的 seed phase——必须去掉一套（去上游的），
   否则同一个 pass 会决策两次。
7. VMIToVPTO 的 physical-action absorption delta——按 pattern family 重放，不能当 patch 打。

## 5. 主要风险与提前发现方式

| 风险 | 如何提前发现 |
|---|---|
| E2B / group-broadcast 策略冲突 | 第 1 天：装上我们的 planner 后跑上游的 `vmi_layout_assignment_group_slot_broadcast_load_e2b_b16.pto` 和 `vmi_extf_8bit_factor_contract.pto` |
| `isVMISameLayoutOp` 重名且算子集不一致 | 链接报错，或消解之后的 class-edge lit |
| 枚举查询按 `...ForLayout` 重写后值域不全 | 用 planner debug 跑 32 个一致性测试：期望看到“无完整合法 plan”的硬失败，而不是错误代码 |
| 去掉 spine scope 后候选变宽，代价模型选错（反而选中会多出 `pto.vor` 的行） | 上游 spine-scoped 相关的 lit（chain 形状那几个）**一条不改**地在我们的 solver 下跑：期望仍选中复合行且 `merges + interleaves = 0`；否则是第 2 层的定价问题 |
| 合并时漏掉上游独有的表行 | CI 门禁：逐表行数对比 66/93 基线，每一行上游独有行都要显式给出归类 |
| VMIToVPTO 重放时丢 pattern family | 步骤 7 前后对 20 个单元做 pattern class 级符号 diff |

## 6. 下一步马上要做的

1. **第 1 天在上游 worktree 上摸底**（对我们的树只读）：
   * 确认上游 `0967a848b` 是否修掉我们的 `trunci` 硬失败（`vmi_prefer_lane_stride_narrowing` 以及
     trunci/truncf 家族）——这是本次升级的主要收益点；
   * 把上游 spine-scoped 两张表的行抄出来，确认它们只用到上游已存在的 `VMILayoutAttr` 形状、
     能被我们现有候选枚举吞下；这决定“删分析、留表”是否真的成立；
   * diff `isVMISameLayoutOp` 的算子集合（我们的 vs 上游的）；
   * 产出事实表对齐的第一版（我们的 66 行 vs 上游的 93 行），确认“我们 66 行里 41 行与上游同表达，
     25 行上游用另一种表达，0 行上游根本没有”。
2. **提交 vexpdif 层**，避免切分支时丢失。
\n\n
## 7. 摸底结果（步骤 1）

### 7.1 「`0967a848b` 的泛化 dense lane-stride 要跟上游对齐」这条作废

`git merge-base --is-ancestor 0967a848b ca7ccb409` 为真：该 commit 距我们的分叉点 **958 个 commit**，
**早就在我们的树里**。我们的 `include/PTO/IR/VMIAttrs.td` 现在就有 `int64_t laneStride` 参数、
`getDeinterleaved(ctx, factor, laneStride = 1)`、`hasDenseLaneStride()` /
`hasGroupSlotLaneStride()` / `isDense()`。因此 `d(4)`、`dls(4, 2)`、`dls(4, 4)` 这些复合形状
**今天就能表达**，spine-scoped 两张表不需要任何属性层工作，§2 里这一项删掉。

分叉点之后只有 2 个 commit 动过 `VMIAttrs.td`，都是方言层收敛，不是新能力，要在步骤 3/7 里当**机械改名**处理：
`7eda9ab55`（vstore dist mode `dintlv` 改名 `intlv`、去掉 unpack vload）、
`8a0a5689b`（pmode 收敛为 zero-only）。

### 7.2 真正的收益在下面这串 post-fork commit 里

| commit | 标题 | 与我们现状的关系 |
|---|---|---|
| `dce6afea0` | feat(vmi): bridge lane-strided dense values to group-slot views | 打开 dense lane-stride 与 group-slot 视图之间的桥（我们的候选集里没有） |
| `9ae084d38` | feat(vmi): keep closed cast round trips in the deinterleaved family | 闭合成对 cast 留在 deinterleaved 家族 |
| `9721a4faa` | fix(vmi): constrain cast layouts by type class | 就是 spine 表里 `CastTypeClass::Integer`/`Float` 的来源 |
| `48e67a2c5` | fix(vmi): split the bitcast layout mechanism and keep compute-carrying narrow sides contiguous | equal-width bitcast 拆分 + 窄侧承载计算保持 contiguous，也就是 spine scope 的来源 |
| `6d744afb5` | fix(vmi): preserve native i16 group sum layout | group reduce i16 |
| `65ea22142` | fix(vmi): correct lane-strided iota lowering for float and integer types | iota group deint |
| `f2c0b9785` | fix(vmi): lower lane-strided single-group broadcasts and 8-bit widening | **和我们现存失败最对得上**：group broadcast multi-consumer、8-bit widening |
| `067da4864` | Support composed dense layout materialization | 复合 dense layout 的 materialization |
| `6b1ba8d99` | fix(vmi): align grouped reductions with the slots=1 dense fallback | group reduce 的 slots=1 回退 |

所以升级能否修掉 trunci 硬失败，已被 7.6 节的探针反证（上游同样失败）；更准确的说法是
post-fork 行为与我们的 solver 决策共同作用**。`0967a848b` 早已在树上却仍然失败，
说明它是**决策问题**（我们的代价链选了 lowering 不支持的那条路），不是能力缺失。

### 7.3 算子分类（probe 2）

上游现在的划分（`VMI/VMILayoutPropagation.cpp:213/234/1152/1155`）：

* `isSameLayoutOp(op)`：仍是 `isa<>` 列表，但集合与我们不同；
* `isEqualWidthBitcastOp(op)`：只认 legacy `pto.vmi.bitcast` 且源/结果 storage 位宽相同；
* `isVMISameLayoutOp = isSameLayoutOp`；`isVMIClassTransparentOp = isSameLayoutOp || isEqualWidthBitcastOp`。

集合差异就是步骤 4 的实际清单：

* **上游有我们没有**：`VMIVsubcOp`、`VMIVsubcsOp`、`VMIActivePrefixIndexOp`、`VMICompressOp`、
  `VMIExpandLoadOp`、`VMIVUnzipOp`、`VMIVZipOp`，以及 `VMIFPToSIOp`/`VMISIToFPOp`——
  后两个我们归进 cast 集合，上游归进 same-layout（等宽 cast 不换 layout）。
* **我们有上游根本没有**：`VMIAddFOp`、`VMIMulFOp`、`VMIMinFOp` … 等 legacy 名字在上游
  `include/PTO/IR` + `lib/PTO/IR` 中 **0 命中**，即 **legacy 算术方言已被上游删除**，这些条目随合并消失。
* 我们 `VMILayoutPlanner.cpp:1239` 的 `sameWidthNumericCast` 已经把等宽 FP<->int cast 变成
  单一 identity relation，**语义与上游一致**，差的只是归类：步骤 4 直接采用上游的
  `isVMIClassTransparentOp` 划分，不需要重写关系。

### 7.4 spine-scoped 两张表的实际内容（probe 3）

`VMILayoutSupportSpineTables.inc`：preferred 6 行、legal 4 行。

    {32,16,0,d(4),     dls(4,2), Integer}   // 整数 16<->32：与通用表的 dense 往返重复
    {16,32,0,d(2),     d(4),     Integer}
    {32,16,0,d(4),     dls(4,2), Float}     // 浮点 16<->32：复合一步到位
    {16,32,0,dls(4,2), d(4),     Float}
    {32,8, 0,d(4),     dls(4,4)}            // 8-bit
    {8,32, 0,dls(4,4), d(4)}

上游注释把价值写得很清楚：窄侧是「宽侧第 i 个 part 的转换结果落在第 i 个物理 part」，
四个 part 一一对应，**不需要 `pto.vor` 把互斥的 part 结果拼成 dense 再拆开**。
而「要不要 `pto.vor`」正是我们第 2 层在数的东西——这反过来验证「删分析、留表」是对的：
代价模型能自己认出这个场景，scope 只是上游没有代价函数时的替代品。
两条硬约束写进了表本身，枚举时不能丢：复合 16<->32 形式**只对浮点成立**
（整数窄化 lowering 没有 one-part-per-chunk 路线，整数只能走 dense `d(4) -> d(2)` 往返），
且两类 type class 不相交。

### 7.5 顺带发现：上游 `master` 提交了工作区元数据

`origin/master` 的根目录**跟踪了 `.ptoas-workspace.json` 和 `env.sh`**（workspace manager 生成的文件）。
后果：workspace manager 会以
"managed path already exists in the source tree" 拒绝建工作区，步骤 1 只能用等价的手工步骤
（`git worktree add` + `python -m venv --system-site-packages` + `quick_install.sh`）。
建议单独清理这两个文件（不在本计划范围内）。


### 7.6 决定性的探针：把 12 个失败用例喂给上游（/tmp/probe_upstream.sh）

上游底座已构建完成（worktree 分支 `feature/vmi-layout-solver-upstream` @ `0a3e01731`，
build `.work/upstream-port/builds/vmi-layout-solver-upstream`，LLVM 19.1.7）。
把我们的 12 个失败用例的 RUN 行原样、逐条用**上游的 `ptoas` / `pto-test-opt`** 执行：

| 用例 | 上游结果 | 失败性质 |
|---|---|---|
| `vmi_to_vpto_compress_tail_invalid` | RUN1 **PASS** | 升级自带修好 |
| `vmi_to_vpto_group_store_compact_small` | RUN1 PASS / RUN2 FAIL | 部分修好，剩决策差异 |
| `opt/fused_quant_dequant_vmi_opt` | FAIL（parse） | **方言漂移**：`pto.castptr` 现在要求 signless i64 |
| `vmi_layout_assignment_store_prefer_lane_stride` | FAIL | 新硬校验：`masked_store` 需要可证明的 store 对齐 |
| `vmi_prefer_lane_stride_narrowing` | FAIL | 同上（`lane_stride = 4` 的 masked_store 被拒） |
| `vmi_to_vpto_quant_dequant` | FAIL | 同上（`deinterleaved = 2` 的 masked_store 被拒） |
| `vmi_to_vpto_group_reduce_partial_slots8` | FAIL | 新硬校验：no executable A5 grouped-reduction layout |
| `vmi_layout_assignment_group_broadcast_multi_consumer` | FAIL | 决策差异（`X = 0`）+ load 安全 remark |
| `vmi_layout_assignment_group_slot_load_dual_layout` | FAIL | 决策差异（`SUM16 = 3`） |
| `vmi_layout_assignment_group_slot_load` | FAIL | 决策差异（lowering 输出不同） |
| `vmi_to_vpto_sub_mul` | FAIL | 决策差异（`SUB0 = 7`） |
| `opt/per_block_bf16_group8_quant_vmi_opt` | FAIL | 决策差异（`vintlv` 组合不同） |

**结论（对原计划的修正）**：

1. **升级不会自动修掉我们的失败**。12 个里只有 2 个 RUN 直接转绿，其余分三类：
   (a) 我们的测试输入按上游新语法已经过时（`castptr` signless i64）；
   (b) 上游把「我们目前会静默产出的错误代码」升级成了**硬报错**（`masked_store` 对齐证明、
   grouped-reduction 可执行性）——这是**安全护栏**，不是通过率礼物；
   (c) 真正的决策差异，正是我们的 solver 要重新决定的那些（也正是第 2/3/4 层要比的东西）。
2. 因此 7.2 节里「升级能修掉 trunci 硬失败」被**反证**：上游在我们这个输入上同样失败，
   只是失败形式更严格（VMI-UNSUPPORTED 对 我们的残余 op）。升级收益应重新表述为
   **拿到护栏 + 新 layout 能力**，失败要由我们的决策层解决。
3. 步骤 8（测试合并）比原先估计更重：45 个被改用例里，凡期望涉及 (a)(c) 两类，
   都要按上游新行为**逐条重判**，不是简单重打。
4. 探针脚手架留在 `/tmp/probe_upstream.sh`（结果 `/tmp/probe_all.txt`），
   步骤 3/5 每完成一段都可以重跑同一批用例，看失败性质是否从 (b)/(c) 收敛。

### 7.7 步骤 2 已落地（上游 worktree 内）

* 已搬运并**零改动编译通过**：`VMILayoutPlanner.h`、`VMILayoutCostModel.h`、`VMILayoutConflictSolver.h`
  → `include/PTO/Transforms/`，`VMILayoutConflictSolver.cpp` → `lib/PTO/Transforms/VMI/`，
  并在 `lib/PTO/Transforms/CMakeLists.txt` 加行；提交 `81b78d21c`（上游 worktree 分支
  `feature/vmi-layout-solver-upstream`）。`libPTOTransforms.a` 与 `pto-test-opt` 链接通过。
* `VMILayoutCostModel.cpp` 暂不入库、CMake 行也先不加（保持基线可链接），它缺的 6 个符号见步骤 2b。
* **上游基线已记录**：`llvm-lit -j8 lit/vmi_new` = 608 个用例，606 通过，2 个失败——
  `vmi_integer_reductions_i8_invalid.pto` 与 `vmi_ptodsl_vunzip_vzip_validation.pto`，
  都是上游自带的失败，与本步骤新增代码无关（新增符号没有任何 pass 引用，未进入流水线）。
  这 2 个即步骤 3 的**通过率基线**，后续合并不得让它变差。

### 7.8 附带清理 PR

[hw-native-sys/PTOAS#1576](https://github.com/hw-native-sys/PTOAS/pull/1576)（draft）：
删掉 `origin/master` 上误提交的 `.ptoas-workspace.json` 与 `env.sh`，并在 `.gitignore`
根锚定地忽略它们；分支 `chore/drop-workspace-metadata` @ `4925e9ed8`，3 个文件 +2/-22。

### 7.9 全量漂移盘点：我们 567 个用例喂给上游（/tmp/upstream_drift.sh）

把 `test/lit/vmi_new` 全部 567 个用例的第一条 RUN 行喂给上游二进制，分类结果：

| 结果 | 数量 | 含义 |
|---|---|---|
| PASS | 383 | 上游原样接受（我们 2/3 的套件不受影响） |
| FileCheck 不匹配 | ~156 | 输入相同、**产出 IR 文本不同**（上游表事实/决策/lowering 的差异） |
| 硬报错（stdin 为空） | 24 | 上游**拒绝**这条流水线（`VMI-UNSUPPORTED` 一类新护栏） |
| 解析失败 | 1 | `opt/fused_quant_dequant_vmi_opt.pto`：`pto.castptr` 现在要求 signless `i64`（我们写的是 `ui64`，共 7 个用例有此写法，但只有这 1 个被新校验挡住） |
| 其他 | 2 | — |

**对步骤 8 的修正**：

1. 真正必须处理的硬约束只有 **24 个硬报错 + 1 个解析**；156 个 FileCheck 差异**不要现在去对齐**——
   它们是我们按自己的 solver 刷过的期望，而步骤 5 之后决策权回到我们的 solver，
   这批用例的期望应当**在步骤 5 完成后重新测量**再定，提前对齐等于把上游的决策抄进我们的测试。
2. 7 个 `pto.castptr ... : ui64` 的用例无论如何都要改成 signless `i64`（纯方言更新，与决策无关）。
3. 这 24 个硬报错就是上游给的**新约束清单**，步骤 3/5 的候选枚举必须保证不再选出这些被拒的布局，
   它们同时也是我们 solver 决策正确性的天然测试。

## 8. 施工清单（来自两次后台盘点，步骤 3/5 的可执行版本）

### 8.1 支持层缺口（步骤 3 的施工面）

solver 一共调用 **46 个** \`VMILayoutSupport\` 方法（48 个签名；planner 64 处、cost model 6 处、conflict solver 2 处、一致性工具 0 处）：

* **同签名存在：25 个**（直接用）。
* **存在但签名/语义不同：4 个**——\`getGroupReduceLayoutFactsForLayout\` 与 \`getPreferredGroupReduceLayoutFact\`（上游多一个 \`VMIGroupReduceKind kind\`，且 preferred 行取决于 \`integer16AddResultLayout\`）；\`getGroupSlotLoadLayoutFact\`（上游没有 \`Value sourceGroupStride\`，改走 \`isSupportedGroupSlotMemoryLayout\`）；\`getPreferredCastLayoutFact\`（上游多 \`bool allowLaneStridePreference = true\`，默认值让调用兼容）。
* **完全没有：17 个**——3 个已由步骤 2b 补（\`getGeneratedMaskLayoutFact\`、\`getInterleaveStoreSupport\`、\`getSameLayoutRelationSupport\`），其余 14 个：\`getLoadLayoutFacts\`、\`getStoreLayoutFacts\`、\`getCastLayoutFacts\`、\`getSameWidthCastLayoutFact\`、\`validateCastOperationRelation\`、\`getReduceLayoutFactForLayouts\`、\`getVintlvLayoutFacts\`、\`getVdintlvLayoutFacts\`、\`getGroupBroadcastLoadLayoutFacts\`、\`getPreferredVdhistLayoutFact\`、\`getPreferredVchistLayoutFact\`、\`getVexpdifLayoutFactsForLayout\`、\`getPreferredVexpdifLayoutFact\`、\`getGroupIotaLayoutFacts\`。
* **重建基座**：\`getCastLayoutFacts\` → 上游 \`getCastLayoutFactsForLayout\` 按 Source 端口枚举；\`getVintlv/VdintlvLayoutFacts\` → 对应 \`...ForLayout\`；\`getLoadLayoutFacts\`/\`getStoreLayoutFacts\` → 上游单数版本 + 我们多出的 3 行 store。**上游没有任何对应物、必须整表搬的**：group iota、非 group reduce（vcadd/vcmax/vcmin）、vdhist/vchist 的 preferred、vexpdif 全族、\`validateCastOperationRelation\`。
* 枚举重建的危险（已写进风险表）：值域收窄不会报错，只会变成 \`no complete legal VMI layout plan\` 的硬失败。

### 8.2 事实结构

* **还缺 3 个字段**：\`VMIGroupStoreLayoutFact::stagingLayout\`、\`VMICastLayoutFact::intrinsicRearrangementCost\`、\`VMIMaskGranularityCastLayoutFact::intrinsicRearrangementCost\`。前两个 cost 字段必须从表的 \`intrinsicCost\` **真正填值**（fork 在 \`VMILayoutSupport.cpp:2190/2320\`），\`stagingLayout\` 必须在两个构造点（\`VMILayoutSupport.cpp:2090\`、\`VMILayoutSupportQueryHelpers.inc:127\`，现在是 \`VMIGroupStoreLayoutFact{layout}\` 聚合初始化）**填上**，否则代价模型读到空、staging 规则变死代码。
* **我们独有的结构 6 个**：\`VMIGroupIotaLayoutFact\`、\`VMIGeneratedMaskLayoutFact\`(已)、\`VMIReduceLayoutFact\`、\`VMIInterleaveStoreSupport\`(已)、\`VMIVexpdifLayoutFact\`、\`VMIVexpdifLayoutPort\`(enum)。
* **上游独有、不能丢**：\`VMIGroupBlockClass::Compact\`、\`VMIGroupReduceKind\` + \`getVMIGroupReduceKind\`、\`isVMISingleCarrierGroupSlots\`/\`...WithStride\`/\`isVMISingleCarrierGroupSlotAlias\`、\`needsVMIDenseLaneStrideGroupSlotBridge\`、\`getPreferredGroupBroadcastLayoutFact\`/\`...ResultLayout\`、\`getGroup{Reduce,Broadcast,Operation}ShapeSupport\`、\`getVZipSupport\`/\`getVUnzipSupport\`、三个 \`getSpineScopedCastLayoutFact*\`。

### 8.3 表与 pattern DSL

* **我们独有的表族**：\`kGeneratedMaskStagingPatterns\`(4 行，2b 已搬)、\`kVexpdifLayoutPatterns\`(6 行，**必须整族搬，否则 vexpdif 层白做**)。
* **我们独有的行（按上游 DSL 重写）**：ensure 6 + ensure-mask 2 + dense-store 3 + group-broadcast-load 1 + group-broadcast-load-direct 1 + legal-mask-granularity-cast 6 + vintlv 3 + vdintlv 3 + group-broadcast 1。
* **上游独有的行**：spine-scoped 两表（6+4，按§2 决策**无条件**接成候选）、ensure +20、ensure-mask +6、group-load +3、broadcast-load/direct 各 +5、group-block-class +1、legal-cast +2、preferred-cast +1、mask-granularity +1、group-broadcast +15。
* **DSL 差异（会改变行的含义）**：上游新增 \`SingleCarrierGroupSlots\`、\`dls(factor, laneStride)\`、\`gsFit()/gsFitStride(S)\`、\`gbCompact()\`、\`CastTypeClass\`、\`GroupBlockPatternKind::Compact\`；\`matchesLayoutPattern\` 多一个 \`lanesPerPart\` 参数；**上游 \`materializeLayoutPattern\` 会转发 \`laneStride\`，我们的不会**——我们现有 DSL 表达不了 \`dls\` 行，这一步必须换成上游的。
* 行序敏感：上游 \`matches*\` 取**首个命中**，我们独有的行是插在前面还是追加，要逐条确认。

### 8.4 步骤 5 的编辑清单（上游 \`VMI/VMILayoutAssignment.cpp\`，2333 行）

* \`applyLayouts()\`（UP 2189-2203）是**唯一**决策点；seed 机制是完整一块（UP 2033-2187，155 行）且只有一个调用点。替换量：删 ~135 行、插 ~256 行（fork \`applyLayouts\` 2310-2503 + \`mergePlan\` + \`selectLayoutPlan\` + \`createPropagator\`）+ 1 个 \`getExplicitLayout\` helper + 2 个 include ≈ **400 行**，结论是**不需要重构**。
* \`collect()\`、约束遍历（UP 860-2001）、各 fixup、pass/factory 一律不动；\`rewriteDataTypes\` 与 \`insertDataUseMaterializations\` 在两棵树里**都是死代码**，不要“顺手恢复”。
* 传播器要加 4 个方法：\`installPlanned(Value)\`、\`installPlanned(OpOperand&)\`、\`getRequestedLayout(OpOperand&) const\`、\`endExactRequests()\`。\`requestExact\` 在整棵树**零调用**，不需要 exactMode；上游 \`addUseConflict\` 已是 \`const\` 且总成功，恰好等于我们非 exact 路径，无需改动。
* 可选但影响 IR 文本：我们的 \`materializeSharedUseConflict\` + \`apply()\` 里的 shared-use 去重（不搬则决策不变、发射的 \`ensure_layout\` 数量不同 → 相关期望会红）。
* **链接阻塞**：\`isVMISameLayoutOp\` 在我们 planner 与上游 propagator 各有一份外部链接定义 → 删 planner 的，改用上游 `isVMISameLayoutOp`/`isVMIClassTransparentOp`，并按上游划分重推 \`isVMILayoutCastOp\`。

### 8.5 边界条件（来自步骤 5 配方，按严重度）

| 风险 | 现象 | 处理 |
|---|---|---|
| R3 字段只加不填 | 编译链接都过，代价链把 cast 自身的重排代价算成 0，静默选错关系（没有测试直接断言该字段） | 3 个字段配 3 处填值逻辑，一起进 |
| R5 上游约束遍历的硬错误面 | 我们的 planner 不调 \`run()\`，\`equivalentValues\` 变惰性；但 \`setNaturalLayout/setPreferredLayout\` 的“冲突”硬错误与 \`validateUnconstrainedOperation\` 仍在 → 会因为**已死的偏好**失败 | 在步骤 3/5 中显式决定是删这些 seed 写入点，还是让它们只做校验 |
| R6 spine/narrow 分析**不能**在步骤 5 单独删 | 它们被 \`addConstraints\` 里的 seed 路径读取，单独删会改错误面与记录相位 | 与步骤 3/4 的 spine 表处理一起删，保证移除本身行为中性 |
| R7 \`rewriteFunctionType\` 两边不一致 | 上游按调用点推导结果布局，我们保留声明布局（还有 mask 粒度归一化） | 步骤 5 保留上游版本，但要用 `validateVMILayoutAssignedIR` 盯住 return 与函数类型不一致 |
| R9 planner 全有或全无 | 任一 component 无解 → 整个 module 硬失败（上游有 contiguous 兜底） | 步骤 5 后用一致性测试覆盖；这是“更多硬失败”而非“错误代码”，可接受但不许变成新常态 |

> 一条更正：后台盘点里曾写“上游已经有 \`forwardsPhysicalParts\`，步骤 2b 的说明过时”。
> 实测 \`git show 0a3e01731:include/PTO/Transforms/VMILayoutSupport.h | grep -c forwardsPhysicalParts\` = **0**，
> 该结论是读到了 2b 正在编辑中的工作区。原始判断成立，字段确实缺。

### 8.6 步骤 8 的方言清单（本轮实测，独立于 solver 移植）

把 567 个用例按“上游二进制”分类后，**需要改方言的只有这几处**：

| 类型 | 数量 | 说明 |
|---|---|---|
| \`pto.vmi.vstore\` 的 \`dist = "dintlv"\` 被拒 | 2 个用例（\`vmi_interleaved_memory_ops.pto\`、\`vmi_to_vpto_memory_x2_widths.pto\`） | 上游新增 \`Intlv\`，vstore 不再接受 \`dintlv\` 拼写；改 \`dintlv\` → \`intlv\` |
| \`pto.castptr ... : ui64\` 被拒 | 1 个用例（\`opt/fused_quant_dequant_vmi_opt.pto\`） | 上游要求 signless \`i64\`（全树只有这 1 个用例用 \`ui64\`） |
| "应当失败"的用例期望变了 | 1 个（\`vmi_interleaved_memory_ops_invalid.pto\`） | 报错文本/位置变化，需按上游新文案重打 |
| **上游真的拒绝这条流水线** | **3 个**：\`opt/compute_mrope_f16_vmi_opt.pto\`、\`vmi_to_vpto_ensure_mask_layout.pto\`、\`vmi_to_vpto_ensure_layout_deint4.pto\` | 都是 \`VMI-UNSUPPORTED\`——这才是步骤 3/5 必须尊重的**真实约束**（此前把 24 个“stdin 为空”误当成硬约束，实际那 24 个全是我们自己的 \`vmi_layout_cost_conformance_*\` 用例，缺的只是还没搬过去的 \`-test-vmi-layout-cost-conformance\` pass） |

其余 \`dintlv\` 相关用例（48 个里有 15 PASS、26 只差 CHECK 文本）不需要改输入，等步骤 5 决策权回到我们的 solver 之后一起重测。

## 9. 步骤 2b 完成（已独立复核）

提交 \`ee461ffa4 vmi: port the layout cost model's support surface onto the upstream base\`
（上游 worktree 分支 \`feature/vmi-layout-solver-upstream\`，6 个文件 +2595 行）：

* 新增 API：\`VMIEnsureLayoutFact::forwardsPhysicalParts\`、\`VMIEnsureMaskLayoutFact::forwardsPhysicalParts\`、
  \`VMIGroupStoreLayoutFact::stagingLayout\`、\`VMIGeneratedMaskLayoutFact\`、\`VMIInterleaveStoreSupport\`，
  以及 \`getGeneratedMaskLayoutFact\` / \`getInterleaveStoreSupport\` / \`getSameLayoutRelationSupport\`。
* 组织方式跟随上游：三个查询落进新的 \`VMILayoutSupportRelationQueries.inc\`（\`VMILayoutSupport.cpp\` 的文本包含单元），
  4 行 \`kGeneratedMaskStagingPatterns\` 落在 \`VMILayoutSupportTables.inc\`，两个 ensure 事实构造与 \`getGroupStoreLayoutFact\` 落在 \`VMILayoutSupport.cpp\`。

**复核（我做的，不是转述）**：

1. \`forwardsPhysicalParts\` 是**真的算出来的**，不是恒 false 的字段：实测 \`VMILayoutSupport.cpp:1955-1962\`（mask 版）
   与 \`~1917\`（vreg 版）都在唯一构造点计算——恒等布局对，或 fork 的“单元素 dense ↔ 单 group 单 slot 载体”对；
   mask 版另加 "contiguous ↔ block-deinterleaved\" 视为谓词寄存器直通。\`grep\` 确认全树无第二处构造点。
2. \`stagingLayout\` 同样实算：\`getGroupStoreLayoutFact\` 的 group-slots 分支用 fork 的 \`compactSmallStore\` 谓词
   （slots==8、laneStride 2 或 4、elementCount 4 或 8、numGroups==elementCount、payloadBits 在 (0,256) 且 %32==0、rowStride 常量 1）
   置 \`stagingLayout = getGroupSlots(ctx, numGroups, slots)\`，其余构造点与 fork 一致留空。
3. 编译链接：\`-Werror\` 下 \`VMILayoutCostModel.cpp.o\` 通过；\`ninja pto-test-opt\` 链接通过；
   归档里有 \`getGeneratedMaskLayoutFact\` / \`getInterleaveStoreSupport\` / \`getSameLayoutRelationSupport\` 与 \`evaluateVMILayoutPlanCost\`。
4. **上游基线未退化**：我自己跑 \`llvm-lit -j8 lit/vmi_new\` = **608 用例 / 606 通过 / 2 失败**，
   两个失败仍是 \`vmi_integer_reductions_i8_invalid.pto\` 与 \`vmi_ptodsl_vunzip_vzip_validation.pto\`。

**已声明的偏差（都记录在案，不影响当前行为）**：

* 上游 ensure 表是我们的严格超集，对那些**只有上游承认**的布局对（非恒等对），fork 没有对应的计算可搬，
  该标志保持 false —— 与其编一条规则，不如留空并记录（代价模型尚未接线，无实际影响）。
* \`getGeneratedMaskLayoutFact\` 里 fork 直接用 \`op->getResult(0)\` 无元数检查，这里加了一道 \`getNumResults() != 1 -> failure\` 防护。
* 未做全仓 clang-format（仓库本身不是 format-clean，且 pre-commit 钩子全局排除了该检查），保持周围手写风格。

**下一步（已启动）**：一个后台任务按阶段推进 3a→3d + 传播器接口 + 步骤 2c（搬 planner 与一致性工具），
每阶段都跑 \`ninja pto-test-opt\` 与上游 608/606/2 门禁；最终门禁是把 fork 的 **32 个 \`vmi_layout_cost_conformance_*\` 用例**
搬进上游树运行——这是移植后的支持层第一次拿到真正的功能证据。

### 9.1 一致性套件基线（fork 树实测）

\`llvm-lit -j8 --filter cost_conformance lit/vmi_new\` 在 fork 树（\`.work/build-llvm19\`）= **32 个用例全过**（567 发现 / 535 排除 / 32 通过）。
因此移植后的门禁是 **32/32**：支持层每补一块，都可以用这 32 个用例的通过数来判断差距，而不是靠“看起来搬完了”。
（此前文档里写的 26 是错的，实际 32 个文件：\`test/lit/vmi_new/vmi_layout_cost_conformance_*.pto\`。）

### 9.2 更强的门禁：一致性 dump 的差分对比（本轮建立）

\`-test-vmi-layout-cost-conformance\` 的输出是**确定性、机器可读**的，每行一件：

    vmi-layout-cost-conformance <case> <op> relation=N cost=M operand0=<layout> result0=<layout>
    vmi-layout-cost-conformance-summary <case> <op> relations=K

所以门禁不应该只看 FileCheck 是否命中——那只能覆盖作者写进 CHECK 的那几行。正确的做法是：
**把同一批用例分别在 fork 树与上游树跑，逐字节 diff 两个 dump**。这能抓出候选集差异
（例如某条查询少枚举了一种 layout，但恰好没被 CHECK 覆盖），而候选集收窄正是本移植最大的静默风险。

* fork 侧参考 dump 已生成：\`.work/upstream-port/probes/fork_conformance_dump.txt\`
  （32 个文件、**334 个 case**、4705 行），生成命令：

      for f in test/lit/vmi_new/vmi_layout_cost_conformance_*.pto; do
        echo "===== $(basename $f)" >> $OUT
        .work/build-llvm19/tools/pto-test-opt/pto-test-opt $f -test-vmi-layout-cost-conformance >> $OUT 2>&1
      done

* 上游侧用同样脚本、把二进制换成 \`.work/upstream-port/builds/.../pto-test-opt\` 即可；
  期望是 **diff 为空**。任何一行差异都要给出解释（是移植漏了候选，还是上游 DSL 语义本就不同）。
* 注意每个用例的第二条 RUN 用 \`-test-vmi-layout-lowering-conformance\`，该 pass 上游同样没有；
  它与 cost 版**在同一个文件里**（\`tools/pto-test-opt/pto-test-vmi-layout-cost-conformance.cpp\`，526 行），
  所以 2c 只要搬这一个文件就两个 pass 都有。

### 9.3 端到端性能复现环境已冻结

完成判据里要求「\`gbmc-amp-dep\`、\`truncf-amp2\` 的性能结论可复现」，所以把产生这些结论的整套环境
冻进了 \`.work/upstream-port/perf/\`（241 MB，含 \`sim-runs/\` 原始 profile）：

* \`cases/\`：9 个自包含用例（\`gbmc-amp-c6/c18/dep\`、\`gbmc-multilaunch\`、
  \`group-broadcast-multi-consumer-amplified\`、\`truncf-amp\`、\`truncf-amp2\`、\`eltwise-k1/k2\`），
  每个含 \`kernel.pto\` + \`ptoas.flags\` + \`main.cpp\`/\`launch.cpp\` + \`golden.py\`/\`compare.py\`。
* 脚本：\`capture5.sh\`（逐 RUN 捕获 lowering 产物，先做 IR 层对比再花仿真机时）、
  \`cmp5.sh\`/\`cmp_dep.py\`/\`truncf_cmp.py\`（两变体产物对比）、\`loop_period.py\`（从 trace 提稳态循环周期）、
  \`nomat_list.txt\`。
* 记录基线（fork 树、移植前）：packed-truncf 32 迭代 MERGE 3011 ticks vs SPLIT 3277（修 absorption 前）/3076；
  \`gbmc-amp-dep\` MERGE 0.927 µs vs SPLIT 0.960 µs（+3.4%）；\`eltwise-k1/k2\` 都是 809.0 ns（store 数中性）；
  计算幅度差 +1.00/+0.44/+1.375 ns/iter。
* 判据写法：只要求**方向与量级**复现；同参数重复跑会有 ±3 ticks 抖动，差异小于约 1% 视为噪声。

（\`/tmp/cases_root\` 与 \`~/msprof-op-simulator-runs\` 仍在，但 /tmp 会被清理，所以必须有这份冻结副本。）

### 8.7 对 §8.3 的一处更正：vintlv/vdintlv 那 6 行不是我们独有的行

移植任务在通读阶段发现并报回：fork 的 \`kVintlvLayoutPatterns\` / \`kVdintlvLayoutPatterns\` 里那 3+3 行，
与上游对应的行**是同一条物理关系**，只是元素位宽掩码更窄（我们 \`bits<8,16,32>\`，上游 \`bits<8,16,32,64>\`）。
把它们插到上游行之前会有两个后果：**静默丢掉 64-bit 支持**，以及 \`getAllInterleaveLayoutFactsImpl\`
不做事后去重、于是返回重复事实。

结论（已批准）：**这 6 行不注入**，保留上游更宽的行，并在提交信息里记录「fork 的窄行被上游的宽行取代」。
因此真正需要整族/整行搬过来的只剩：\`kVexpdifLayoutPatterns\`（6 行整族）、
\`kGeneratedMaskStagingPatterns\`（4 行整族，2b 已搬），以及其余按上游 DSL 表达仍成立的独有行。

这条也定了一条工作方式：**证据不足就报「无法忠实表达」，而不是注入一条会静默改变语义的行**——
与本移植一贯的判准一致（宁可留下一条显式声明，也不接受假规则）。

### 8.8 步骤 6 的落地配方（本轮把锚点钉死，可直接机械执行）

fork 侧 \`VPTOPack4StoreMaskNormalize\`（160 行，\`lib/PTO/Transforms/VPTOPack4StoreMaskNormalize.cpp\`）
在上游的落点如下，**五处改动**：

| # | 位置 | 做什么 |
|---|---|---|
| 1 | \`lib/PTO/Transforms/VPTO/VPTOPack4StoreMaskNormalize.cpp\` | 整文件搬入（上游 VPTO 级 pass 就住这个目录）；它只依赖 \`PTO/IR/PTO.h\`、\`PTO/IR/PTOTypeUtils.h\`、\`PTO/Transforms/Passes.h\` 与 \`GEN_PASS_DEF_...\`，自包含 |
| 2 | \`include/PTO/Transforms/Passes.td\` | 加 fork 的 pass 定义（照抄 \`VPTOPack4StoreMaskNormalize\` 那段：\`Pass<\"vpto-normalize-packed-store-mask\", \"ModuleOp\">\`、summary/description、\`constructor\`、\`dependentDialects = [\"mlir::pto::PTODialect\"]\`）；上游同类可参照 \`VPTONormalizeContainer\`（Passes.td:438） |
| 3 | \`include/PTO/Transforms/Passes.h\` | 加 \`std::unique_ptr<Pass> createVPTOPack4StoreMaskNormalizePass();\` |
| 4 | \`lib/PTO/Transforms/CMakeLists.txt\` | 加一行 \`  VPTO/VPTOPack4StoreMaskNormalize.cpp\` |
| 5 | \`tools/ptoas/ptoas_pipeline.cpp\` | **唯一需要翻译、不能照抄的一处**：fork 是在 \`tools/ptoas/ptoas.cpp:3297\` 直接 \`addPass\`，而上游管线已搬进 \`ptoas_pipeline.cpp\`。插到 \`appendVMISemanticPipeline\`（953 行起）里、\`pm.addPass(pto::createVMIToVPTOPass());\`（994 行）**之后**、\`pm.addPass(pto::createVPTOStatefulStreamFusionPass());\`（995 行）**之前**——与该 pass 自己的说明「Run this immediately after VMI-to-VPTO lowering」一致 |

验证：\`ptoas --pto-arch=a5 --pto-backend=vpto --emit-vpto\` 的输出里，PK4_B32 的 store 谓词应变成字节粒度形式
（\`PAT_VL64 -> PAT_ALL\`、\`PAT_VL32 -> PAT_VL128\` … 并带 \`pto.pbitcast\` 回到 b32），
而运行时谓词、无合法字节拼写的形状（如 \`PAT_VL3\`）、非 packed 元素类型、非 PK4_B32 的 store 都保持原样。

注意第 4 步与正在进行的支持层任务**共用同一个 CMakeLists.txt**，所以步骤 6 要等那一批提交落地后再动，避免行冲突。
