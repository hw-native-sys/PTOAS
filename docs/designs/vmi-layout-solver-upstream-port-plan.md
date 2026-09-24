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

### 8.9 步骤 8 的真实规模（数据化，替换计划里估的 45 个）

\`git diff --name-only ca7ccb409..HEAD -- test/lit/vmi_new\` = **132 个**被我们改过的用例，
逐个与「上游二进制下的表现」交叉分类（明细表 \`.work/upstream-port/probes/step8_plan.tsv\`）：

| 分类 | 数量 | 处理方式 |
|---|---|---|
| CHECKONLY（只差期望文本） | **96** | **步骤 5 之前不要动**：期望是照我们 solver 刷的，决策权回到我们的 solver 后要重新测量再定 |
| PASS（上游原样接受） | 19 | 直接搬 |
| TOOLERR（上游直接拒绝或硬失败） | **16** | 唯一必须逐个处理的集合，见下 |
| UNKNOWN | 1 | 单独查 |

这 16 个的成因很清楚，也基本就是上游新护栏清单：

* 方言：\`opt/fused_quant_dequant_vmi_opt.pto\`（\`castptr\` 需 signless \`i64\`）。
* 上游不再支持的形式：\`vmi_group_reduce_addi_i8\`（\`pto.vmi.vcadd\`）、\`vmi_group_reduce_maxi_i8\`（\`pto.vmi.vcmax\`）。
* 上游布局契约/降级拒绝：\`vmi_layout_assignment_ensure_mask_layout\`（\`VMI-LAYOUT-CONTRACT\`）、
  \`vmi_layout_assignment_group_broadcast_load_e2b_b16\`（conversion pattern 应用失败）、
  \`vmi_prefer_lane_stride_narrowing\`、\`vmi_to_vpto_ensure_layout_deint4\`、\`vmi_to_vpto_ensure_mask_layout\`、
  \`vmi_to_vpto_masked_store\`、\`vmi_to_vpto_masked_store_deint_tail\`（均 \`VMI-UNSUPPORTED\`，就是 masked-store 对齐证明那一类）。
* “应当失败”的用例报错文本变了：\`vmi_layout_assignment_group_load_block8_truncf\`、\`vmi_layout_assignment_group_reduce_s12_invalid\`。

**结论**：步骤 8 的真实工作量在 16 个用例 + 96 个期望的**重新测量**，而不是 132 个逐个手改；
而且 96 个里凡是步骤 5 之后我们的 solver 重新选定同一布局的，期望应当**回到我们原来的文本**，
不是改成上游的。这一条决定了步骤 8 必须排在步骤 5 之后，不能提前做。

### 9.4 门禁的覆盖面检查（本轮实测）

从冻结的 fork dump 里统计 32 个一致性用例覆盖到的算子（按 case 数）：

* 厚覆盖：\`ensure_mask_granularity\`(37)、\`ensure_layout\`(28)、\`extui\`(23)、\`group_broadcast\`(19)、
  \`ensure_mask_layout\`(18)、\`store\`(15)、\`group_store\`(13)、\`masked_store\`(12)、
  \`group_broadcast_load\`(11)、\`group_reduce_addf\`(9)、\`group_load\`(8)、\`trunci\`(7)、\`vselr\`(6)；
* 我们新增/独有的族也有覆盖：\`vintlv\`(5)、\`vdintlv\`(5)、\`truncf\`(5)、\`create_group_mask\`(4)、
  \`bitcast\`(4)、\`group_slot_load\`(4)、\`shuffle\`(2)、\`channel_split/merge\`(2)；
* **薄覆盖（只有 1 个 case）**：\`vexpdif\`、\`vdhist\`、\`vchist\`、\`vcadd\`、\`vcmax\`、\`vcmin\`。

结论与风险边界：

1. 门禁**没有整族漏空**——每个 fork 独有的族至少有一个 case，所以「整条查询被漏搬」会在 diff 里现形。
2. 真正的薄点是**单 case 族**：如果某个族的候选集在**别的形状**下会收窄，而那个形状没有 case，diff 抓不到。
   \`vexpdif\` 尤其值得注意，因为它的 6 行表是我们这次新加的（vexpdif 层），而只有 1 个 case 覆盖它。
3. 因应措施：步骤 5 落地后、步骤 9 之前，给 \`vexpdif\`（以及 \`vdhist\`/\`vchist\`）各补 1-2 个一致性用例
   （照现有 \`vmi_layout_cost_conformance_*.pto\` 的 MARK+CHECK 格式），把这些族的候选集在多种形状下钉住，
   再跑差分门禁。这一步属于步骤 8/9 之间的小任务，不阻塞主线。

### 9.5 门禁加厚：vexpdif 五个形状已钉住（基线更新为 33/33）

按 §9.4 的结论动手补了薄覆盖族。新增 \`test/lit/vmi_new/vmi_layout_cost_conformance_vexpdif_shapes.pto\`
（98 行，提交 \`c8b2ffa81\`），覆盖 \`vexpdif\` 的五个 f32 形状：contiguous、deinterleaved=2、deinterleaved=4、
block_deinterleaved=2、block_deinterleaved=4，每个形状恰好 1 条关系、cost=0。

* 一致性套件基线 **32 → 33**（\`--filter cost_conformance\` = 568 发现 / 535 排除 / **33 通过**）。
* 参考 dump 已重新生成：**33 个文件 / 339 个 case / 4769 行**
  （\`.work/upstream-port/probes/fork_conformance_dump.txt\`）。以后所有差分门禁都以这份为准。

**同时查出一个待办（已记录、未擅自结论）**：表中第 6 行是 f16 源、f32 结果、\`c() -> d(2)\`，
也是 f16 输入时的 **preferred** 行；但我把该形状写成用例后，**一致性工具对它一条关系都不报**：
\`vexpdif\` 的 f16 用例在 dump 里完全缺席（其余 5 个形状各出 2 行）。两种可能——
要么一致性框架只枚举同宽关系、这条行本就要走 preferred 路径才可见；要么该行在当前实现里实际不可达：
fork 侧 \`getVexpdifLayoutFactsForLayout\` / \`matchesVexpdifPhysicalShape\`（\`VMILayoutSupport.cpp:1326\`）
是判据所在。**处理方式：不猜也不钉死**，作为待查项列在文档里；等支持层移植到步骤 3b/3d 时，
用同一份 dump 差分顺带回答它（若上游侧同样不报，则属于框架限制而非移植缺陷）。

## 10. 步骤 7 的真实规模（本轮盘点：比原估小一个数量级，而且不能打 patch）

### 10.1 关键数字

* fork 的 \`lib/PTO/Transforms/VMIToVPTO.cpp\` 改动**只来自一个提交**：\`7e2207d65 Add costed VMI layout solver\`（3832+/2389-）。
  \`e5bd4511b\`（absorbed physical actions）与 \`589213988\`（vexpdif）**根本没碰这个文件**。
* 原始 diff 是 6221 行 / 1139 hunk，但其中 **831 个 hunk 只是加花括号与换行的风格改动**
  （fork 上跑过一次 brace 风格 pass，例如 \`containsVMIType\`）。把每个 hunk 的非花括号、去空白文本合并后，
  真实 delta = **42 hunk / +434 / -347 ≈ 780 行**。
* 结论：步骤 7 是 **~780 行 / 42 hunk 的重放**，不是 6221 行；**不能打 patch 或 cherry-pick**——
  既会把风格 pass 一起带进来，也必然冲突（这点 lit 抓不到，门禁只能是 diff 本身）。
* 原始 diff 已留存：\`.work/upstream-port/probes/forkvmi.diff\`。

上游 \`lib/PTO/Transforms/VMIToVPTO/\` 的 18 个文件**是同一个编译单元的文本包含片段**
（\`#include\` 写在 \`namespace {\` 内，顺序刻意非数字：…Pattern0,1,2,3,4,5,**10**,6,UnifiedPattern,7,8,9），
因此 delta 只能按**区域**映射到单元，单元边界本身在语义上是任意的。

### 10.2 十个族与处理结论

| 族 | fork 锚点 | 规模 | 结论 |
|---|---|---|---|
| F1 mask 物理粒度 helper | 349、5352 | -38/+0、-18/+0 | **(a) 丢弃**：上游已在方言层提供 \`getVMIMaskPhysicalGranularity\`（\`lib/PTO/IR/VMI/VMIHelpers.cpp:639\`，声明 \`VMIInternal.h:42\`）；\`getVMIMaskPhysicalCarrierLayout\` 上游也有 |
| F2 \`validateCastOperationRelation\` | 13956/14007/14019/14047/14245/14816-14906 | ~-40/+60 | **(c) 新增**：上游 0 命中；需步骤 3 的查询 + \`PatternInternals7/8\` 里加检查 |
| F3 形状检查改走事实查询 | 1643/1852/1924/1982 | ~-45/+33 | **(b) 合并**。**其中 \`checkSupportedGroupBroadcastLoadShape\` 是 -18/+0：我们在删上游逻辑**，盲重放会放宽接受面、产出上游会拒绝的 IR |
| F4 deint4<->2 与 lane-stride 桥 | 4530、4576 | -2/+114 | **(b) 合并**：上游 \`067da4864\`/\`dce6afea0\` 已建同一函数，我们的 +114 **不是超集** |
| F5 mask layout/粒度转换 + ensure-mask pattern | 4687/4747/4768-5075/5177-5289/5726 | ~-30/+124 | **(b) 合并** |
| F6 create_group_mask factor-4 内联块 | 6587 | -66/+2 | **(a) 丢弃**：上游用 \`lowerFactor4Block\` 替代（\`PatternInternals0.cpp\`） |
| F7 group store staging | 8295-8299、8361-8364 | -6/+5 | **(b) 合并**：**这就是 absorption 在 lowering 里的落点** |
| F8 group reduce 泛化 | 10882/10909/11137-11246 | ~-76/+68 | **(b) 合并**：上游 \`6d744afb5\`（i16 原生求和）/\`6b1ba8d99\`（slots=1 回退）重塑过同一段，且引入了 \`VMIGroupReduceKind\` 分类 |
| F9 pattern 注册表 | 13833 | -16/+15 | **(b) 必须与 F2 同落**，否则出现注册了但缺实现的 pattern |
| F10 \`verifySupportedVMIToVPTOOps\` | 14816-14906 | -9/+12 | **(b) 随 F2/F3 自动跟进** |

**零 delta 的族（因此不属于步骤 7）**：vexpdif（我们的工作全在 layout 层，上游 lowering 里 \`OneToNVMIVexpdifOpPattern\` 本来就有）、
quant/dequant（是方言 \`castptr\` 与决策问题）、packed-store mask（是步骤 6 的独立 pass，\`Pack4\` 在 VMIToVPTO 里 0 命中）、
deinterleave store / group slot / 整数 cast（纯粹是格式差异，\`getDenseLaneStrideLoad/StoreDistToken\`、\`packToPreviousCarrier\`、
\`getVcaddResultType\`、\`OneToNVMITruncI/F\` 等函数体在归一化后没有任何行为 delta）。

### 10.3 absorption 到底要改哪里（精确定位）

\`absorb\` 在两个树的 \`VMIToVPTO/*\` 里 **0 命中**——上游 lowering 没有这个概念，它只在 cost model 与 conflict solver 里。
lowering 一侧的落点是**未命名但真实存在**的两处：

1. \`VMIToVPTOMemoryInternals.cpp\`：把 group store 形状检查从 \`isCompactSmallGroupStore\` 换成 \`getGroupStoreLayoutFact\` 得到的 \`stagingLayout\`（fork 锚点 1982-1984）。
2. \`VMIToVPTOPatternInternals3.cpp\`：\`GroupStoreLayoutKind\`/compact 判定与 packed staging 类型构造同样改用该事实（fork 锚点 8295-8299、8361-8364）。

dist-mode 实现（\`getDenseLaneStride{Load,Store}DistToken\`）与 cast-part 转发（\`viewVcvtResult\`）在两棵树里都只是格式差异，**步骤 7 无需改动**。
结论：**absorption 不是一个 pass，而是「cost model 定价」与「这两处 lowering 路径」之间的约定**；只改一边就会静默走错（代价模型给 staging 路径定价、lowering 却不走）。

### 10.4 批次与门禁

| 批次 | 内容 | 规模 | 依赖 |
|---|---|---|---|
| A 丢弃 | F1 + F6（**只是不要再加回来**） | ~130 行 | 无 |
| B cast 关系校验 | F2 | ~110 行 | 需步骤 3 的 \`validateCastOperationRelation\` |
| C 形状检查 + group store staging | F3 + F7（**必须同落**，共用 \`getGroupStoreLayoutFact\`/\`isCompactSmallGroupStore\`/\`GroupStoreLayoutKind\`） | ~90 行 | 步骤 3 的事实 |
| D mask layout 转换 | F5 | ~150 行 | 无 |
| E data-layout 转换 | F4 | ~120 行 | 排在 D 之后 |
| F group reduce | F8 | ~145 行 | 与 B/D+E 互不依赖 |
| G 注册与校验 | F9 + F10 | ~50 行 | 必须最后 |

总计 ~780 行，单人约 4-6 天；B/F/D+E 可并行（对应计划里「步骤 7 与步骤 3 并行」）。
**批次门禁用符号级 pattern-class diff**（\`populateVMIConversionPatterns\` 成员 + 每族 \`grep -c\`），
因为步骤 5 之前 lit 还不是主门禁（决策权仍在我们的 solver 之外）。

### 10.5 风险（机械搬运会错的地方）

| 风险 | 现象/探测 |
|---|---|
| 831 个花括号 hunk | 打 patch 会冲突或静默引入全仓风格 pass，**lit 看不见**；门禁只能是归一化 diff 本身（断言 42 hunk / +434/-347） |
| F3 的 -18/+0 | 闸门被删 → 接受面变宽 → 产出上游会拒绝的 IR（\`vmi_interleaved_memory_ops_invalid\`、\`vmi_to_vpto_ensure_mask_layout\`、\`vmi_to_vpto_ensure_layout_deint4\` 可观察） |
| F7 事实/谓词只改一边 | 定价与 lowering 不一致 → 静默选错布局；探测：\`vmi_to_vpto_group_store_compact_small.pto\` 两条 RUN + 一致性 dump 差分 |
| F8 与上游分类混用 | 保留我们的 factor 循环但用上游 \`VMIGroupReduceKind\` 分类 → part 顺序错；探测：\`vmi_to_vpto_group_reduce_partial_slots8.pto\` 等 |
| F4/F5 与上游已覆盖部分重叠 | 重复覆盖 → 多出一对 \`vintlv\`/\`vdintlv\`；探测：\`vmi_to_vpto_memory_x2_widths.pto\`、\`vmi_interleaved_memory_ops.pto\`；**不要**在步骤 5 前去对齐那 ~156 个 FileCheck 差异 |
| F6 丢弃是否安全 | 上游 \`lowerFactor4Block\` 需覆盖被删块里的**动态** active-elems 路径，未验证；探测：\`create_group_mask\` 相关 lit |

> 盘点自身声明的不确定项（保留原样，不当作结论）：F1 的「丢弃」判定基于名字与位置、**未比对函数体**；
> 上游多数对应物只到单元级、未到行级；三个后加的单元（\`UnifiedMaskInternals\`、\`PatternInternals10\`、\`UnifiedPatternInternals\`）
> 的原始归属是按 include 顺序推断的；F2 究竟为了正确性还是只为诊断未确认。

### 10.6 更正 §10.5 的一条风险：F3 不是「删上游逻辑」，而是内联了上游后来抽出的 helper

§10.5 里那条「\`checkSupportedGroupBroadcastLoadShape\` 是 -18/+0，我们在删上游逻辑、盲重放会放宽接受面」
经逐行核对是**表述错误**。两边的实际形态：

上游（\`VMIToVPTOMemoryInternals.cpp:954\`）：

    checkSupportedGroupBroadcastLoadShape(op, reason):
      supports.getGroupBroadcastLoadSupport(op, reason)      // 同一个查询
      checkSupportedGroupBroadcastLoadMemory(op, resultType, reason)
    checkSupportedGroupBroadcastLoadMemory(op, resultType, reason):
      buildReadAccessPlan(source, op.getOffset(), resultType, VMIMemoryCoverageKind::Dense)
      if (!isa<PtrType>(...)) fail(...)

fork（\`VMIToVPTO.cpp:1899-1923\`）：

    checkSupportedGroupBroadcastLoadShape(op, reason):
      supports.getGroupBroadcastLoadSupport(op, reason)      // 同一个查询
      buildReadAccessPlan(source, sourceType, resultType,
                          getConstantIndexValue(op.getOffset()), VMIMemoryValidMaskKind::AllTrue)
      if (!isa<PtrType>(...)) fail(...)

* 上游**也已经**委托给同一个查询 \`getGroupBroadcastLoadSupport\`，所以「我们改成查询、上游还在用字面谓词」不成立；
  真正的差异是：fork 把上游后来抽成独立函数的那段**内联**在形状检查里（fork 侧 \`checkSupportedGroupBroadcastLoadMemory\` **0 命中**，
  上游 2 命中），即 fork 写在这段重构**之前**。
* 于是 \`buildReadAccessPlan\` 的**实参**不同：上游 \`op.getOffset()\` + \`VMIMemoryCoverageKind::Dense\`，
  fork \`getConstantIndexValue(op.getOffset())\` + \`VMIMemoryValidMaskKind::AllTrue\`。
  接受面的差异只可能来自这里，而不是「少了一个检查」。
* **对批次 C 的指令随之更正**：这个 hunk 按 **(a) 保留上游版本、不重放我们这一版** 处理；
  若重放，风险方向恰好相反——会把上游的 **Dense 覆盖要求**换成旧的 AllTrue 形式，属于**放宽**而非收紧。

### 10.7 盘点结论的置信度分级（据此安排验证）

这次核对说明：盘点的**分类与风险描述是假设，必须落到代码上验**。因此按置信度排优先级：

1. **已独立核对**：F3 的这一处（本节的更正）；\`absorb\` 在两个树的 \`VMIToVPTO/*\` 中 0 命中；
   \`validateCastOperationRelation\` 上游 0 命中；上游 \`checkSupportedGroupBroadcastLoadMemory\` 存在且被形状检查调用。
2. **待验证（会改变动作）**：F1 的「丢弃」判定——上游 \`lib/PTO/IR/VMI/VMIHelpers.cpp:639\` 的
   \`getVMIMaskPhysicalGranularity\` 与 fork 侧同名/同责函数**是否语义等价**（盘点只比了名字与所在位置）；
   F6 的「丢弃」是否安全（上游 \`lowerFactor4Block\` 是否覆盖被删块的**动态** active-elems 路径，已追加求证任务）。
3. **仅为推断**：上游多数对应物只定位到单元级（未到行级）；三个后加单元的原始归属按 include 顺序推断；
   F2 究竟服务正确性还是仅服务诊断。

> 结论：步骤 7 每批落地前，先把该批涉及的「丢弃/保留」判定按第 1 级标准核一遍（读代码或跑用例），
> 不允许按盘点结论直接照做。

### 10.8 F1 的「丢弃」判定已核对：成立

§10.7 第 2 级里「F1 只比了名字与位置、未比函数体」这条**已查清并可以结案**：

* fork 的 \`VMIToVPTO.cpp\` 现在有 **8 处**调用 \`getVMIMaskPhysicalGranularity\`（376/908/2992/3066/3202/5342/5356/6410），
  也就是说它删掉的是**本文件内的本地副本**，改用了方言层的共享实现：\`lib/PTO/IR/VMI.cpp:746\`
（薄封装，转 \`getVMIMaskPhysicalGranularityImpl\`，定义在 \`:304\`）。
* 上游的同一函数在 \`lib/PTO/IR/VMI/VMIHelpers.cpp:639\`，实现就是把 \`granularity bit width × laneStride\`
  映射成物理粒度字符串，与 fork 的 \`...Impl\` 职责一致。
* \`getVMIMaskPhysicalCarrierLayout\` 同理：fork 侧定义在 \`lib/PTO/IR/VMI.cpp:750\`（同样转 \`...Impl\`），上游在 data-layout 单元里有对应实现。

**结论**：两棵树都已经改用方言层共享 helper，方向一致；重放 F1 只会往 \`VMIToVPTO.cpp\` 里**加回一个已废弃的本地副本**。
因此 F1 按 \`(a) 丢弃\` 处理是**正确的**，不需要在批次 A 做任何事（只要不回搬）。
§10.7 第 2 级现在只剩 **F6** 一个待验证项（上游 \`lowerFactor4Block\` 是否覆盖动态 active-elems 路径，已追加求证）。

### 10.9 F2 是正确性依赖，不是诊断（已核对，改变批次 B 的定位）

盘点把 F2（\`validateCastOperationRelation\`）标为「上游 0 命中 → 新增」，但把「服务正确性还是仅服务诊断」
留成了未决项。现在查清：**它是正确性依赖，而且我们的 solver 直接依赖它**。

实现只有 20 行（\`lib/PTO/Transforms/VMILayoutSupport.cpp:2042-2063\`），规则是：
**当 source 与 result 都是 group-slot 载体、且算子属于 \`VMIExtFOp\`/\`VMIFPToSIOp\`/\`VMIFPToUIOp\`/\`VMISIToFPOp\` 时直接失败**——
注释写明理由是「这些族没有 group-slot 物理配方，在这里拒绝可以让 Support 与 VMIToVPTO 保持同步」。

三类消费者（这就是为什么它不能当诊断看待）：

1. **planner**：\`lib/PTO/Transforms/VMILayoutPlanner.cpp:1272\` —— 关系枚举时 **失败的候选直接跳过**，
   所以它参与决定 solver 的**候选集**；不搬它，solver 就会选出 lowering 根本实现不了的 group-slot 转换，
   失败形态正是我们最怕的那种（\`no complete legal plan\` 或残余 op），而不是一条清晰的报错。
2. lowering 形状检查：\`VMIToVPTO.cpp:13958/14009/14020/14059\`（FPToSI/FPToUI/SIToFP/Compress 等）。
3. 校验 pass：\`PTOValidateVMIIR.cpp:813/826/839\`。

**对批次 B 的定位修正**：F2 不是「补一条诊断」，而是「把 solver 的候选约束与 lowering 的能力重新对齐」的必需项；
它必须与步骤 3 的 cast 查询、以及 F9 的 pattern 注册一起落（否则会出现注册了 pattern 但候选集更宽的组合）。

**待办**：移植时确认上游新增的 \`CastTypeClass\` 约束是否已经挡住了同样的组合——若已挡住，F2 变成冗余但无害；
若没挡住，它就是唯一防线。这条判断在批次 B 落地时用一致性 dump 差分直接验证（候选集里是否出现 group-slot 对）。

## 11. 一次真实回归与门禁陷阱（本轮抓到，已回退给实现方）

### 11.1 事实经过

支持层任务提交了 Stage 1（\`2f62e9ef9 vmi: add the layout solver's cast facts and charge their intrinsic cost\`，
4 文件 +287/-19），并报告「上游基线未退化」。我在它之后**先重建再跑**门禁（\`ninja -C <build> pto-test-opt\` →
\`exit 0\`、二进制重新链接），结果：

    Total Discovered Tests: 608
      Passed: 598 (98.36%)
      Failed:  10 (1.64%)

而基线是 **608 / 606 / 2**。多出来的 8 个失败：

* \`vmi_deinterleave_load_layout_propagation.pto\`
* \`vmi_layout_assignment_group_broadcast_deint4_consumer.pto\`
* \`vmi_layout_assignment_group_reduce_partial_slots8.pto\`
* \`vmi_layout_assignment_group_slot_broadcast_load_e2b_b16.pto\`
* \`vmi_layout_assignment_group_store_truncf_contiguous.pto\`
* \`vmi_layout_assignment_mask_use_ensure.pto\`
* \`vmi_layout_assignment_reduce_minmaxf.pto\`
* \`vmi_to_vpto_ensure_mask_granularity.pto\`

### 11.2 成因（从它自己的提交说明里就能看出）

该提交把 \`makeMaskGranularityCastLayoutFact\` 改成接收 mask 类型、返回 \`FailureOr\`，让调用方
**跳过「没有可物化 carrier 路径」的行**，并让 \`collectMatchingCastFacts\`（每个来自 legal 表的 cast 事实的**唯一漏斗**）
**丢弃「代价算不出来」的行**。这两处改的是**上游自己查询的接受面**，于是上游的布局决策跟着变——
正是支持层合并**最不允许**出现的那种变化（判据是「用上游自己的决策链路跑上游测试仍然通过」）。
把 \`getMaskGranularityBits\` 移到 pattern DSL 没问题，**改接受面才是问题**。

### 11.3 门禁陷阱（比这次 bug 更值得记住）

**「跑过 lit」不等于「用新代码跑过 lit」。** 两个人都踩了同一个坑：前一次 \`ninja | tail -3\` 的输出被截在
\`[15/498]\` 这样的中间行（看起来像构建中途断掉），于是「构建成功」这个前提没有被真正验证，
lit 就是拿**旧二进制**跑的——于是 606/2「通过」。

因此定成硬规矩，写进每批的仪式：

1. 先跑 \`ninja … ; echo exit=$?\`，**必须看到 exit=0**，且日志里 \`grep -ci error\` 为 0；
2. 再核对**二进制 mtime 晚于**源码 mtime（\`ls -l --time-style=+%H:%M\`）；
3. 然后才跑 lit，并把**完整的** \`Total Discovered / Passed / Failed\` 三行原样贴出来；
4. 断言失败集合等于基线集合，而不是只看数字（\`vmi_integer_reductions_i8_invalid.pto\` 与
   \`vmi_ptodsl_vunzip_vzip_validation.pto\` 是上游自带的两个）。

### 11.4 已下达的修复要求

* 先在**独立 build 目录**里构建父提交 \`ee461ffa4\` 跑同一过滤，证明归因（父提交应为 606/2）；
* Stage 1 必须对上游决策路径**行为中性**：两个 \`intrinsicRearrangementCost\` 仍要从表的 \`intrinsicCost\` 真正填值
  （这条要求不变），但**不许因为代价接线而丢行或改变任何查询的接受面**；fork 的「丢弃该行」语义若 solver 侧需要，
  必须挂在显式参数后面、只对我们自己的枚举生效；
* 用**重建后**的二进制重跑门禁，贴出 ninja 退出码与完整 lit 三行；
* 修复另起一个提交（历史保持诚实），并给出「父提交 lit 结果 / 修复后 lit 结果 / 删掉接受面改动的那段 diff」。

> 结论：这次回归**不是白跑**——它证明「先重建、再跑、断言失败集合」这条仪式确实能抓到别人（和我自己）漏掉的问题；
> 也说明为什么我一直不肯拿汇报当通过证据。

### 10.12 批次 C 的逐 hunk 指令（已按代码核对，替换原来笼统的「MERGE」）

F3 的 4 个 hunk + F7 的 2 个 hunk 就是 absorption 批次。逐个核对后的指令：

**hunk 1（interleave store，fork +1643，-15/+14）→ 可以重放，但只换谓词块。**
上游原本是内联字面检查：low/high 都必须 contiguous、元素个数与类型必须一致、
\`getX2MemoryDistToken(..., \"INTLV\")\`（限 8/16/32 位），随后单独调 \`checkFullDataPhysicalChunks\`。
我们 2b 已经搬进来的 \`getInterleaveStoreSupport\`（worktree \`VMILayoutSupportRelationQueries.inc:80-107\`）
实现了**同样四项**（contiguous、元素个数与类型一致、8/16/32 位、整物理块），与 fork 原文（\`VMILayoutSupport.cpp:3511-3539\`）逐行一致。
所以：**只把内联谓词块换成查询调用**，后面的 access-plan / maskable / full-chunk 尾巴保持上游原样，接受面不变。

**hunk 2（group slot load，fork +1852，-5/+10）→ 被签名差异挡住，需先决策。**
fork 的 \`getGroupSlotLoadLayoutFact(VMIVRegType resultType, Value sourceGroupStride, int64_t numGroups, reason)\`
比上游多一个 \`sourceGroupStride\`（上游 worktree 现状 \`VMILayoutSupport.h:497-499\` 只有 \`(resultType, numGroups, reason)\`）。
两种做法都可行，但必须显式选一个并记录：**(甲)** 给上游查询加上 stride 参数、采用 fork 语义（与
\`vmi_layout_assignment_group_slot_load_slots1_dynamic_stride_invalid\` 这类「动态 stride」用例一致）；
**(乙)** 保留上游签名、放弃 stride 感知的行匹配。
建议选 **(甲)**：上游本来就承认动态 stride（有对应 invalid 用例），而且这是我们 4 个签名差异之一，迟早要处理；
选 (乙) 会让候选集收窄，属于最危险的静默失败方向。受影响用例：
\`vmi_layout_assignment_group_slot_load_dual_layout\`、\`..._group_slot_load\`、\`vmi_to_vpto_group_slot_load\`。

**hunk 3（group broadcast load，fork +1924，-18/+0）→ 保留上游版本，不重放**（§10.6 已核实：
上游同样委托同一个查询，fork 只是内联了上游后来抽出的 memory 检查；重放会把 Dense 覆盖要求换成 AllTrue）。

**hunk 4（group store，fork +1982，-7/+9）＋ F7（fork 8295-8299、8361-8364）→ 与 F7 必须同落。**
上游是 \`isCompactSmallGroupStore(layout, valueType, numGroups, rowStride)\` 字面谓词，
fork 改成读 \`getGroupStoreLayoutFact(...)->stagingLayout\`。我们 2b 搬进来的 \`stagingLayout\` 计算
**就是同一个谓词**（slots==8、laneStride ∈ {1,2,4}、elementCount ∈ {4,8}、numGroups==elementCount、
payload 位宽 ∈ (0,256) 且 %32==0、rowStride==1），且只在 \`getGroupStoreLayoutFact\` 的 group-slots 分支置位、
其余构造点保持空——与 fork 一致。所以这组替换是**等价替换**，前提是
（i）两处一起改（形状检查 + \`GroupStoreLayoutKind\`/packed staging 类型构造），
（ii）不要把 \`stagingLayout\` 也塞进 \`getPreferredGroupStoreLayoutFact\` 等其它构造点。

**批次 C 的门禁**：\`vmi_to_vpto_group_store_compact_small.pto\` 两条 RUN、\`vmi_layout_assignment_group_slot_load*\`、
\`vmi_interleaved_memory_ops.pto\`，加上「重建后」的上游 608/606/2 三条断言（见 §11.3 的四步仪式）。

### 10.13 步骤 5 的子决策：上游 seed 写入点的硬错误必须去掉，但写入要留

§8.5 的 R5 说「上游约束遍历的硬错误面仍由已死的偏好驱动」。本轮读代码把处置方式定下来。

\`setNaturalLayout\`（upstream \`VMILayoutAssignment.cpp:418-436\`）与 \`setPreferredLayout\`（438-456）各做三件事：
（1）写 \`dataNodes[root].naturalLayout/preferredLayout\`；（2）push 一条 \`DataLayoutSeed\`；
（3）**当同一个 root 已经有不同值时直接 \`emitError\`**：\`\"conflicting natural layouts\"\` / \`\"conflicting preferred layouts\"\`。
调用点约 **17 处 \`setNaturalLayout\` + 10 处 \`setPreferredLayout\`**，全部在我们要保留的约束遍历 \`addConstraints\` 里。

步骤 5 之后的状态：我们的 \`applyLayouts\` 从不调用 \`propagator.run()\`，
\`dataLayoutSeeds\` 的**唯一读者**是 \`2046/2057\`（就在被删掉的 seed 机制里），所以 seed 向量与 root 布局**变成惰性**。
但**那条硬错误是活的**——它由 \`addConstraints\` 触发，而 \`addConstraints\` 我们要保留。

后果：步骤 5 之后，同一个值有两个消费者、各自偏好不同布局时，pass 会**直接以布局契约错误中止**，
而我们的代价求解器本可以合法地插入一次转换来解决——这是 fork 侧**根本不存在**的新失败模式。

**处置（建议，写入步骤 5 的实现清单）**：

* **去掉这两处 \`emitError\`**（改为不报错、后写覆盖或忽略），但**保留写入**。
  理由是 \`getDataLayout(value)\` 仍被保留代码读取（\`670\` 在约束遍历里、\`1887\` 在 \`rewriteDataTypes\` 里虽然已是死代码），
  一旦把写入也删掉，\`getDataLayout\` 的结果会变，进而可能改变 \`addConstraints\` 建立的等价关系——那是**另一处静默行为变化**。
  留写入、只去错误，是 6 行左右的改动，行为上刚好满足要求。
* 更彻底的方案（把这两个函数连同 27 个调用点、\`dataLayoutSeeds\`、\`DataLayoutSeedPhase\` 全删）不必在本轮做：
  收益是去掉死代码，风险是在 976 行的约束遍历里动 27 处。留到步骤 5 之后、用例全绿时再作为清理单独提交。
* **探测方式**：步骤 5 之后跑上游 \`lit/vmi_new\` 与我们的用例，凡出现
  \`kVMIDiagLayoutContractPrefix\` + \`\"conflicting natural layouts\"\` / \`\"conflicting preferred layouts\"\`，
  按定义就是**步骤 5 之后不应存在**的错误（同一前缀仍被我们 planner 的失败诊断使用，所以不要误删前缀本身）。

### 11.5 修复已完成并验证（由我直接做，不再等实现方）

实现方在收到回归报告后迟迟没有产出（也没有建父提交的独立 build 目录做归因），所以我中断了它并自己动手。
修复提交：**\`8bc4c428e vmi: keep every legal cast relation when attaching its intrinsic cost\`**（上游 worktree）。

改法（只动两处，都是「恢复上游接受面」，代价字段仍按能力尽力填值）：

1. \`collectMatchingCastFacts\`：把「代价算不出来就 \`continue\`（丢行）」改成
   **尽力填值、无论如何都 push 该行**，并写明理由：这个漏斗定义的是**合法性**，
   「代价无法推导」不等于「关系非法」；在这里丢行会收窄**所有调用方**（包括上游自己的布局决策）的候选集。
2. \`makeMaskGranularityCastLayoutFact\`：返回类型从 \`FailureOr<...>\` 改回值类型（不再因代价失败），
   调用方恢复无条件 push。
   「丢弃未定价行」属于**fork 侧、由 solver 驱动的枚举**（步骤 3 落地时在那里恢复），不属于共享漏斗。

**验证（按 §11.3 的四步仪式）**：

* \`ninja -C <build> pto-test-opt\` → **exit=0**，日志 \`grep -ci error\` = 0（\`-Werror\` 下无告警）；
* 二进制重新链接（relink 行可见），源码 mtime 早于二进制；
* \`llvm-lit -j8 lit/vmi_new\` → **608 discovered / 606 passed / 2 failed**，
  两个失败正是上游自带的 \`vmi_integer_reductions_i8_invalid.pto\` 与 \`vmi_ptodsl_vunzip_vzip_validation.pto\`；
* 失败集合与基线**逐条相同**，不是只看数字。

**结论**：Stage 1 的实质内容（事实结构、两个 cost 字段、\`VMILayoutSupportSolverCosts.inc\`）+ 修复提交
= 既有我们需要的字段，又对上游决策路径行为中性。步骤 3 可以在这个状态上继续。

> 过程教训已固化为规矩：**门禁必须「先重建、再跑、断言失败集合」**（§11.3）。
> 这条规矩在本轮直接救回了一次 8 个用例的静默回归。

## 12. 步骤 4 的传播器接口已落地（我方实现并验证）

提交 **\`2d9d45b9b vmi: add the propagator surface the costed planner needs\`**（上游 worktree）。
新增 4 个入口，全部按 fork 原文搬运（fork \`VMILayoutPropagation.cpp:1295-1334\`、\`1138-1149\`）：

| 入口 | 作用 |
|---|---|
| \`getRequestedLayout(OpOperand&) const\` | 取该操作数上**已被记录的 use conflict** 布局，优先于值自身的 assignment；planner 用它确认已提交的 use 关系是否仍然有效 |
| \`installPlanned(Value, layout)\` | 记录单条计划赋值，**不入队**；与既有 assignment 冲突则失败 |
| \`installPlanned(OpOperand&, layout)\` | 同上，但冲突时按 fork 语义记成 use conflict；且保留「无类型结构传输值（如 SCF region 结果）以计划布局为主种子、已有显式类型布局的值保留自身布局」的规则 |
| \`endExactRequests()\` | **故意是空实现**：fork 的两个 \`requestExact\` 重载全树零调用，本移植不需要 exact-mode；它存在只为让 planner 的提交序列保持不变 |

**验证**：\`ninja pto-test-opt\` exit=0、0 告警；\`llvm-lit -j8 lit/vmi_new\` = **608 / 606 / 2**，
失败集合**逐条**等于基线（\`vmi_integer_reductions_i8_invalid.pto\`、\`vmi_ptodsl_vunzip_vzip_validation.pto\`）。

**步骤 4 还剩两项，都刻意推迟到相关批次**：

1. \`isVMISameLayoutOp\` 的重复定义要等 planner 真正进树时再消（现在 planner 还没编译进上游，不构成链接冲突）；
2. 删除 \`setSpineScopedCastOps\`/\`isSpineScopedCast\` 必须**与 spine 分析的移除同时进行**（§8.5 R6：
   \`spineScopedCasts\` 被 \`addConstraints\` 里的 seed 路径读取，单独删会改变错误面与记录相位）——
   所以它属于支持层批次（Stage 2/3 的 spine 行处理），不属于本轮。

## 13. 步骤 6 的两次受挫与本轮教训（诚实记录）

### 13.1 步骤 6 现在的真实阻塞点：完整运行时链接依赖 planner

我按 §8.8 把 packed-store-mask pass 五处接线做完（复制 160 行源文件、CMake 行、Passes.td 定义、
Passes.h 声明、\`ptoas_pipeline.cpp\` 里插到 \`createVMIToVPTOPass()\` 之后），然后构建：

* **\`ninja pto-test-opt\` 成功**（pass 与注册都没问题）；
* **\`ninja … ptoas_runtime_deps\` 失败**，链接错误是：

      undefined reference to `mlir::pto::isVMILayoutCastOp(mlir::Operation*)'
      VMILayoutCostModel.cpp:(…PlanGraphBuilder13buildRelation…+0x45f7)

原因链：\`ptoas_runtime_deps\` 会把归档里**被引用到的**对象全链进来；\`VMILayoutConflictSolver.cpp:1189\`
引用了代价模型，代价模型又调用 \`isVMILayoutCastOp\`，而**该函数定义在我们的 planner 里**（\`VMILayoutPlanner.cpp:1173\`），
planner 尚未进树。之前没暴露，是因为没有任何东西引用 conflict solver，那些对象一直没被拉进可执行文件。

**结论**：步骤 6 不能早于步骤 2c。正确顺序是 *stage 3 的查询 → planner 进树（顺便消掉 \`isVMILayoutCastOp\` 与
\`isVMISameLayoutOp\` 的重定义）→ 再做步骤 6*。我已把自己那五处未提交改动**原样撤回**，把工作树还给 Stage 2。

### 13.2 新的过程陷阱：**不要在别人有未提交改动时构建**

撤回之后我跑了一次 \`ninja pto-test-opt\` + \`lit\`，得到 **608 / 541 / 67**——看起来像一次大回归，其实不是：
那次的 \`git status\` 显示 Stage 2 正在改 \`VMILayoutSupport.h/.cpp\`、\`Materialization.inc\`、\`Tables.inc\`
并新增 \`VMILayoutSupportVexpdifQueries.inc\`，我这一构建**把它半成品状态编进了二进制**，于是 67 个失败。

因此 §11.3 的四步仪式要再加一条前置条件：

**0. 构建/测量前先 \`git status\` 确认工作树干净**（或确认在测的就是某个已提交状态）。
并发写者 + 共享 build 目录 = 测出来的是「混合物」，既不能当通过也不能当回归。
这个坑和 §11.3 的「旧二进制」是同一个问题的两面：**被测对象必须是明确的那一个状态**。

### 13.3 顺带查出：packed-store-mask pass **没有测试**

为步骤 6 找验证基准时发现：\`vpto-normalize-packed-store-mask\`（160 行，提交 \`e5bd4511b\` 引入）
在 fork 里**没有任何 lit 测试**——\`grep -rln \"normalize-packed-store-mask\" test/\` 为空，
\`PAT_VL\`/\`pbitcast\` 也没出现在 \`test/lit/vmi_new/\` 的期望里。

也就是说：步骤 6 原来的判据「\`ptoas --emit-vpto\` 输出字节粒度谓词」**在 fork 侧就没有基线**。
注意这**不是**这次移植引入的问题（fork 当初就是这么提交的），但它意味着：

1. 步骤 6 一旦落地，我们**无法判断它是否真的生效**（没有期望可比）；
2. 更重要的是，这个 pass 的实际行为从未被验证过，而它改写的是**谓词语义**（谓词覆盖的字节必须完全一致），
   属于「错了会静默写错内存」的那一类。

**处置**：已另起一个任务在 fork 树里补这个 lit 测试（不影响上游 worktree，也不与 Stage 2 冲突），
要求同时覆盖正例（静态 \`pset_b32\` 前缀 → 字节粒度 + \`pbitcast\` 回 b32）与反例
（运行时 \`plt_b32\`、无合法字节拼写的形状必须原样保留），并要求它跑完整套件确认没扰动；
若测试暴露出 pass 与自身描述不符，**报告而不是顺手改**。

### 13.4 步骤 8 的方言改动**不能**在 fork 里做（已核实）

§8.6 把「\`dist = \"dintlv\"\` → \`intlv\`」列为步骤 8 的一项，但没有写明**在哪棵树里做**。本轮核清：

* fork 的 \`include/PTO/IR/VMIAttrs.td\` 里 \`VMIDistMode\` 只有四个取值：
  \`Continuous\`/\`Unpack\`/\`Dintlv\`/\`Brc\` —— **没有 \`Intlv\`**。所以在我们这棵树上写 \`intlv\` 会被自己的方言直接拒绝；
* 上游同时有 \`Dintlv\` 与 \`Intlv\`（\`VMIAttrs.td:67-68\`），但**删掉了 \`Unpack\`**（上游 0 命中，fork 里仍在）。

**结论（写进步骤 8 的执行细则）**：

1. \`dintlv → intlv\` 的改名必须在**移植到上游树之后**、对着上游方言做（作用对象是我们搬过去的测试文件），
   不能在 fork 里预改——否则 fork 自己就红了；
2. 反过来，凡 fork 期望里出现 **unpack vload** 的地方，上游已无此 dist mode，属于**必须重写**（不是改名）；
   这类用例要在步骤 8 里单独挑出来，不能按批量替换处理。
3. 因此步骤 8 的前置条件不仅是「步骤 5 之后」，还包括「目标树是上游树」——这两条共同决定了它的位置。

### 13.5 步骤 8 方言改动的**精确规模**（把 §13.4 的担心收敛掉）

按「方言拼写」而不是「子串」重新量了一遍，结论比之前清爽：

| 查询 | 结果 | 含义 |
|---|---|---|
| fork 测试里被引号引起来的 \`\"unpack\"\` | **0 文件 / 0 处** | 上游删掉 \`Unpack\` dist mode 这件事，在我们的用例里**没有实际出现**；§13.4 里担心的「必须重写而非改名」的那一类，至少在这批测试里是空的（步骤 8 时再复核一次即可） |
| fork 测试里被引号引起来的 \`\"dintlv\"\` | **6 文件 / 11 处** | 这才是真正要改拼写的范围 |

之前那个「48 文件 / 113 处」是**子串**统计（把 \`pto.vdintlv\`、\`pto.vdintlvx2\` 这类**算子名**以及 CHECK 文本里的同名片段都算进去了），
与「改 dist-mode 拼写」不是一回事——算子名不需要改，改了反而错。

上游枚举现状（\`VMIAttrs.td:66-73\`）：\`continuous(0)\`、\`dintlv(2)\`、\`intlv(3)\`、\`brc(4)\`，
**\`unpack(1)\` 已删除**；注意 \`Dintlv\` 仍在枚举里（其它算子还在用），
但 \`pto.vmi.vstore\` 的合法集合里没有它——这正是那 2 个 TOOLERR 用例（\`vmi_interleaved_memory_ops.pto\`、
\`vmi_to_vpto_memory_x2_widths.pto\`）报 \`invalid dist-mode\` 的原因。

**结论**：步骤 8 的方言改动是「**6 个文件里 11 处引号拼写**」这个量级，可以逐条看清楚；
不需要担心被算子名的字面重合误导，也不需要为 unpack 做批量重写。

### 13.6 **更正 §9.5**：那个「待查项」是我自己的错误，行 6 没问题

§9.5 里我写过「f16 源、f32 结果、\`c() -> d(2)\` 这一行一致性工具一条关系都不报，可能是行不可达」。
本轮查清了：**这个结论是错的，而且错在我身上**——当时那次实验里 f16 用例**根本没有真的写进文件**
（生成脚本的两次写入互相覆盖了，提交进去的文件只有 5 个 f32 函数；我在报告里当成 6 个来读，
于是把「文件里没有这个用例」误读成了「工具不为它产生关系」）。

这次把它真正写进去之后，得到的是**真实且有用的结论**：

1. 第一次尝试（mask 用结果的 \`deinterleaved = 2\`）被 op 校验直接拒绝：

       error: 'pto.vmi.vexpdif' op requires mask layout to match data layout

2. 把 mask 改成与**源数据布局**一致（\`contiguous\`）后，\`rc=0\` 且该行**恰好枚举出 1 条关系、cost=0**。

所以真正的约束是：**\`pto.vmi.vexpdif\` 要求 mask 布局与数据布局一致**（对本例即跟随源布局），
而这正是移植时要尊重的规则——vexpdif 的候选布局变更**不能把 mask 与源布局拆开**。
行 6（f16 → f32、\`c() -> d(2)\`）本身是可达的，之前那条「可能不可达」的说法**作废**。

**已落地**：f16 用例写入 \`vmi_layout_cost_conformance_vexpdif_shapes.pto\`（提交 \`998d45ae9\`），
该文件现在 6 个形状、12 条 COST 断言；一致性过滤套件仍 **33 全过**；
参考 dump 刷新为 **33 文件 / 340 case / 4781 行**。

> 复盘：这次教训与 §11.3、§13.2 是同一族——**先说结论、后核对证据**就会出错。
> 我把它记在这里而不是悄悄删掉，因为它比那条被作废的结论更有用：门禁里的每一条「发现」都要能被**重放**。

### 13.7 步骤 5 的依赖清单已核完：**只剩 planner 一个门槛**

把预抽出的 \`applyLayouts\`（194 行）用到的东西逐个核了一遍，这是它的完整依赖：

| 依赖 | 状态 |
|---|---|
| \`VMILayoutAttr\` / \`VMILayoutPlan\` / \`VMILayoutPropagator\` | ✓ 头文件已在上游树（\`81b78d21c\`） |
| \`installPlanned\`（6 处） | ✓ 步骤 4 已加（\`2d9d45b9b\`） |
| \`matchEdge\`（6 处）、\`getExplicitLayout\`（1 处） | ✓ 都在预抽出的片段里（\`step5/\`） |
| \`VMIControlFlowSupport::addWhileConstraints\` | ✓ 上游本来就有（\`VMIControlFlowSupport.h:38\`，其 mask 粒度 pass 在用） |
| \`commitVMILayoutPlan\` / \`selectCostedVMILayoutPlans\` | ✗ **来自 planner，尚未进树** |

**结论**：步骤 5 的唯一硬依赖是 planner（stage 2c）；而 planner 又依赖 stage 3 的支持层查询。
于是关键路径被压缩成一条直线，可以据此排期：

    stage 3 查询（14 个） → planner 进树 → 同时解锁 步骤 5（决策替换）与 步骤 6（pass 接线）

而步骤 6 之所以也卡在 planner，是 §13.1 那条：\`ptoas_runtime_deps\` 的完整链接需要
\`isVMILayoutCastOp\` 的定义，它就在 planner 里（\`VMILayoutPlanner.cpp:1173\`）。
换句话说 **planner 是这两个步骤共同的、也是唯一的阻塞点**——不是两件独立的事，而是一件事。

### 13.8 终局门禁脚本已就位（含「脏树拒绝」）

\`.work/upstream-port/validate_port.sh\`（59 行）把这段时间积累的门禁收成一条命令，按顺序做四件事并给出单一裁决：

1. **第 0 步先查工作树**：非 \`--allow-dirty\` 且 \`git status\` 非空就**直接退出（exit 2）**，
   理由写在输出里：「脏树上测出来的都是混合物」（§13.2）。这一步正是本轮实测过的——
   当前上游工作树有 Stage 2 的 5 个未提交文件，脚本立刻拒绝，不进入构建；
2. 构建 \`pto-test-opt\`，打印 **ninja 退出码**与错误行数（不是「看起来跑完了」）；
3. 跑上游 \`lit/vmi_new\`，并把**失败集合排序后与基线集合逐条比对**，不一致就判失败；
4. 跑 fork 的 33 个一致性用例，再做 §9.2 的 **dump 差分**（对 §9.5 刷新后的 33 文件 / 340 case 参考），期望 \`IDENTICAL\`。

最后打印 \`ALL GATES PASS\` / \`GATES FAILED\`。用法：\`validate_port.sh [--allow-dirty]\`。

设计取舍：**不做「宽容模式」**。之前的两次假信号（旧二进制 606/2、脏树 67 failed）都是因为门禁允许自己
在状态不明的情况下给出结论；这个脚本宁可拒绝执行，也不产出无法归因的数字。

### 13.9 批次 F（F8 group reduce）的风险前提已核实，指令随之收紧

盘点把 F8 标为 MERGE 并警告「机械重放会把我们的 factor 循环与上游的 \`VMIGroupReduceKind\` 分类混用、导致 part 顺序错」。
本轮读了上游实现，**前提成立**：\`VMIToVPTOPatternInternals4.cpp:1882-1912\` 是

    enum class GroupReduceLoweringPlan { CompactMaskedRows, OneBlockVcgadd,
      TwoBlockDeinterleaved2VcgaddVadd, FourBlockDeinterleaved4VcgaddTree,
      FullDeinterleaved2VcaddRows, ContiguousVcaddRows };
    classifyGroupReduceLoweringPlan(VMIGroupReduceKind kind, source, mask, result, numGroups, reason)
      -> supports.getGroupReduceLayoutFactForLayouts(kind, …) 然后 switch (fact->blockClass)

即上游是 **kind 驱动 + blockClass 驱动的显式计划集合**（6 种），而我们的 F8 是把 \`deinterleaved=2\` 的
两段拆分**泛化成按 layout factor 的循环**，并把 row-reduce 与 combine 折成一趟 \`zip_equal(groupSources, groupMasks)\`。
两者是**两种不同做法**，不是同一段代码的两个版本——所以：

* **批次 F 不许整体重放**。正确做法是逐 hunk 先回答「上游的 kind/blockClass 分类是否已经覆盖这条路径」：
  覆盖的（很可能包括 2-block/4-block 的拆分与排序）按 (a) 保留上游；
  只有上游确实没有的（例如把两趟合成一趟的 \`zip_equal\` 结构）才作为 (b) 合并过去。
* **门禁**：\`vmi_to_vpto_group_reduce_partial_slots8.pto\`、\`vmi_layout_assignment_group_reduce_*\`、
  \`vmi_integer_reductions_i8_invalid.pto\`（注意后者是上游自带失败，只能当「不变差」的参照），
  外加「重建后」的 608/606/2 与失败集合逐条比对。
* 规模修正：9 hunk / -76/+68 里，预计真正需要移植的**远少于**表面行数——与 F3 同一模式（先分类再动手）。

### 13.10 批次 E（F4）的分类：两处是新的，一处要先验证是否已被覆盖

F4 只有 2 个 hunk，但内容上其实是三类改动：

1. **dense lane-stride 桥**：\`deint2 ↔ laneStride\`、\`laneStride2 ↔ laneStride4\`、\`laneStride → deint2\`，
   实现方式是**绕 dense 中间态**（先调 \`materializeDataLayoutConversion(op, *dense, …)\` 再转回来）；
2. **\`deinterleaved = 4 → 2\`**：用 \`vintlv\` 取 even/odd 两个 part；
3. **\`deinterleaved = 2 → 4\`**：同样的逆操作。

上游侧核对结果（\`lib/PTO/Transforms/VMIToVPTO/VMIToVPTODataLayoutInternals.cpp\`）：

* 该文件里 **只有 factor 2 的处理**（\`:356\` 的 \`isDeinterleaved() && getFactor() == 2\` 判据，\`:971/:974\` 的
  \`getFactor() == 2 && getLaneStride() == 1\`），**没有任何 \`getFactor() == 4\` 的转换代码**；
* \`VintlvOp\`/\`VdintlvOp\` 在这个文件里出现 10 次（上游确实用它们做交织/解交织），但与 factor 4 无关；
* \`materializeLaneStrideToContiguous\`（\`:19-38\`）是上游自己的 lane-stride → contiguous 路径。

**批次 E 的指令因此分两类**：

* 上面第 2/3 条（deint4↔2 的 vintlv/vdintlv 构造）**上游没有**，属于真正要移植的部分；
* 第 1 条（lane-stride 桥）要**先确认上游的组合路径是否已经等价**（它已经有 \`materializeLaneStrideToContiguous\`，
  以及 \`dce6afea0\`/\`067da4864\` 引入的组合式 dense 物化）；若等价则按 (a) 保留上游，避免重复覆盖后多出一对
  \`vintlv\`/\`vdintlv\`——这正是盘点给的探测手段（\`vmi_to_vpto_memory_x2_widths.pto\`、\`vmi_interleaved_memory_ops.pto\`）。
* 门禁仍按 §10.4：本批次以**符号级 pattern-class diff** 为主，lit 在步骤 5 之前只能当辅助证据。

### 13.11 packed-store-mask 的测试已出现，并且我做了**反证检查**

任务产出了 \`test/lit/vpto/vpto_normalize_packed_store_mask.pto\`（183 行，两条 RUN）：

* 管线级：\`ptoas --pto-arch=a5 --pto-backend=vpto --emit-vpto %s -o - | FileCheck --check-prefix=VPTO\`；
* pass 级：\`pto-test-opt %s -vpto-normalize-packed-store-mask | FileCheck --check-prefix=PASS\`。

用例里出现 \`pto.vsts\`(31)、\`pto.pset_b32\`(18)、\`pto.pbitcast\`(10)、\`pto.pset_b8\`(9)、\`pto.pge_b32\`(5)，
从分布看同时覆盖了改写侧（\`pset_b32\` → \`pset_b8\` + \`pbitcast\`）与应当保持原样的运行时谓词侧（\`pge_b32\`）。

**独立验证（两步，都做了）**：

1. 直接跑：\`llvm-lit lit/vpto/vpto_normalize_packed_store_mask.pto\` → **PASS**（用 fork 现有二进制，未重新构建）；
2. **反证**：把副本里第二条 RUN 的 \`-vpto-normalize-packed-store-mask\` 去掉，再跑 → **FAIL**。
   说明这些 CHECK **确实依赖该 pass 生效**，不是「怎么跑都能过」的空转测试。

第二条尤其重要：这正是前面抓过的两类假信号（旧二进制、脏树）之外的另一类——**测试自己没在测东西**。
凡新增 pass/行为的测试，都应当做一次这样的反证（去掉被测对象后必须变红），否则它只会给人虚假的信心。

测试文件目前仍未被作者提交（\`git status\` 里是未跟踪状态），我不越俎代庖，等它按流程提交。

## 14. 第三类假信号：缺失的 .so（附两次提交）

### 14.1 事实经过与更正

我在 round 22 用 \`ptoas_runtime_deps\` 试步骤 6 时链接失败（\`isVMILayoutCastOp\` 未定义）。ninja 在命令失败时会
**删除该目标的输出文件**，于是 \`libPTOASCompiler.so\` 被删掉了。之后所有**走 python 包装器**的用例都会在 import 阶段失败：

    ImportError: libPTOASCompiler.so: cannot open shared object file: No such file or directory
    FileCheck error: '<stdin>' is empty.

结果就是 \`lit/vmi_new\` 报 **608 / 541 / 67**。我先后**两次**误判了这个数字：

* 第一次归因为「脏树构建」（§13.2）——不准确；
* 第二次归因为 Stage 2 的未提交改动，**据此回退了它 155 行的工作**（已存档到 \`.work/upstream-port/step3-wip/\`）——也错。

真正的教训：**门禁不仅要断言失败集合，还要看失败的「原因」**。我这两次都只看了数量与集合，没看日志内容；
只要点开一条失败就会看到 \`ImportError\`，而不是任何与布局决策有关的报错。

### 14.2 修复与恢复（两次提交）

1. \`5e9493f37 vmi: define the layout cast-op classification the cost model links against\`：
   新增 \`lib/PTO/Transforms/VMI/VMILayoutOpClasses.cpp\`，给出 \`isVMILayoutCastOp\`（fork 原文，8 个算子类上游都存在）。
   这既补齐了链接缺口，也顺带完成 §8.4 里「planner 不应重复定义该符号」的前置：日后 planner 进树时直接用这份定义。
   修完 \`pto-test-opt\` 与 python 模块都能链接（exit 0、0 告警），基线回到 **608 / 606 / 2**。
2. \`1c1bba8cc vmi: port the vexpdif layout table family and its rows\`：把回退掉的 Stage 2 工作按存档补回并**重新测量**——
   \`ninja\` exit 0、\`lit/vmi_new\` **608 / 606 / 2**，失败集合仍是那两个上游自带用例。
   也就是说那次回退其实是**基于假信号的误伤**；现在它已被正常提交，\`.work/upstream-port/step3-wip/\` 保留为凭证。

### 14.3 顺带更正 §13.11 与基线口径（来自测试任务的复核）

* fork 侧 \`lit/vmi_new\` 的实际规模是 **568 发现 / 556 通过 / 12 失败**——我之前写的「33」是
  \`--filter cost_conformance\` 的**子集**口径，不是整套；两处口径以后必须写清楚。
* fork 全量套件是 **1872 发现 / 1858 通过 / 13 失败**，其中 1 个（\`vpto/vmi_f4x2_to_bf16x2_vcvt_llvm.pto\`）
  是**既有失败**，与本次改动无关（测试任务把新文件移出树后它仍然失败，已证）；所以「12 个已知失败」要带 +1 说明。
* §13.11 里对测试文件的算子计数（\`pge_b32\`(5) 等）来自**早期草稿**，与提交进去的版本不一致——
  以最终提交 \`f35bdcf66\` 的内容为准；那两条被 pass 描述遗漏的行为（共享 mask 分支、\`PAT_ALL→PAT_ALL\`）已被测试钉住。

## 15. planner 缺口的**活清单**（可反复运行）

\`.work/upstream-port/planner_gap.sh\`：把 fork 的 \`VMILayoutPlanner.cpp\` 用**当前上游树**做 \`-fsyntax-only\` 编译
（不修改任何树、不链接），把「还差哪些符号」直接打出来。做法是从 ninja 的 \`compile_commands\` 里取一条同类编译命令，
把 \`-c\` 指向 planner、去掉 \`-o/-MF/-MT\`，再加 \`-fsyntax-only\`；**脚本自己会校验改写后的命令确实在编译 planner**，
否则拒绝运行。

当前裁决（上游 HEAD \`1c1bba8cc\`）：**73 个错误**，成因分三类：

1. **legacy 算子类已不存在**（\`VMIAddFOp\`/\`VMIAddIOp\`/\`VMISubFOp\`/\`VMISubIOp\` …）——
   这是 planner 里 \`isVMISameLayoutOp\` 那份手写算子表；按 §8.4，它**应当整份删掉**，改用上游的
   \`isSameLayoutOp\` / \`isVMIClassTransparentOp\` 划分，而不是逐个补名字。这是目前最大的一类错误。
2. **group-reduce 的签名差异**：\`getGroupReduceLayoutFactsForLayout\` 与 \`getPreferredGroupReduceLayoutFact\`
   在上游多一个 \`VMIGroupReduceKind\` 参数（§8.1 的 (b) 类第 1、2 项）。
3. **其余缺失查询**（§8.1 的 (c) 类）——正是 Stage 3 正在补的部分。

完整输出已归档：\`.work/upstream-port/probes/planner_gap.txt\`。
**用法**：Stage 3 每落一批就重跑一次；错误数下降到 0 即代表「planner 可以进树」。
这比等某个 agent 说「搬完了」可靠——它是一个可复现的判据。

### 15.1 又一次同类教训：**先验证量具，再看读数**

这个脚本的第一版给出 \`rc=0 / 0 errors\`，看起来像「planner 已经能编译」。实际是 \`sed\` 替换错了位置：
命令里的 \`-c\` 仍指向 \`VMILayoutConflictSolver.cpp\`，等于什么都没检查。
我发现它的方式不是「觉得可疑」，而是**把改写后的命令打印出来核对**（末尾 260 字符 + 是否含 planner 路径），
于是看到 \`-c .../VMILayoutConflictSolver.cpp\` 还在。修正后 \`rc=1 / 73 errors\`。

这已经是同一族的第四次：旧二进制、脏树、缺 .so、**量具本身没在量东西**。共同点只有一个——
**结论必须能追溯到「被测对象确实是那个东西」**。因此规约再收紧一条：
**任何自定义检查脚本都要自带一次自证**（例如「被编译的文件确实是目标文件」「被测二进制新于源码」），
否则它的绿灯不比没有更可靠。

### 15.2 planner 已预处理，缺口从 73 → **25**，且剩下一份精确清单

在**树外**（\`.work/upstream-port/step2c/VMILayoutPlanner.cpp\`，2671 行）把 planner 里那两份算子分类定义**删掉**：

* \`isVMISameLayoutOp\`：上游 \`VMILayoutPropagation.cpp\` 已是唯一来源；
* \`isVMILayoutCastOp\`：由我们新加的 \`VMILayoutOpClasses.cpp\`（\`5e9493f37\`）提供；
* 并在原处留注释说明「planner 故意两个都不定义，保证整条链接里每个符号只有一份定义」，
  这正是 §8.4 里那条链接阻塞的正解。

效果（用同一个 \`planner_gap2.sh\` 量）：**73 → 25 个错误**，而且剩下的全是 Stage 3 该补的东西——

| 类型 | 具体 |
|---|---|
| 缺失查询（12） | \`getCastLayoutFacts\`、\`getStoreLayoutFacts\`、\`getLoadLayoutFacts\`、\`getVintlvLayoutFacts\`、\`getVdintlvLayoutFacts\`、\`getGroupBroadcastLoadLayoutFacts\`、\`getGroupIotaLayoutFacts\`、\`getReduceLayoutFactForLayouts\`、\`getSameWidthCastLayoutFact\`、\`getPreferredVdhistLayoutFact\`、\`getPreferredVchistLayoutFact\`、\`validateCastOperationRelation\` |
| 签名不一致（2） | \`getGroupReduceLayoutFactsForLayout\`、\`getPreferredGroupReduceLayoutFact\`（上游多 \`VMIGroupReduceKind\`） |
| 另有一处类型转换错误 | 与 \`getGroupSlotLoadLayoutFact\` 的 \`Value sourceGroupStride\` 差异相关（上游签名无该参数） |

于是 §8.1 里「17 个缺失」的口径也被校正为：**12 个查询 + 2 处签名 + 1 处参数差异**，
其中 \`getGeneratedMaskLayoutFact\`/\`getInterleaveStoreSupport\`/\`getSameLayoutRelationSupport\` 已在 2b 完成，
vexpdif 两族已在 Stage 2 完成。

**这份预处理文件就是步骤 2c 的输入**：等 25 个错误清零，把 \`.work/upstream-port/step2c/VMILayoutPlanner.cpp\`
拷进上游树、加 CMake 行即可；一致性工具同理（它依赖 planner，属于同一批）。

#### 15.2.1 那 25 个错误已逐条定位，没有意外项

用原始编译输出（带 \`文件:行:列\`）核对后，25 个错误全部落进已计划的三类，没有新东西：

* \`VMILayoutPlanner.cpp:921\`、\`:1813\` —— \`cannot convert TypedValue<IndexType> to int64_t\`，
  正是 **group-slot-load 的参数差异**：fork 在这两处传 \`op.getSourceGroupStride()\`（一个 \`Value\`），
  而上游第二参数是 \`int64_t numGroups\`。所以它的表现是**类型不匹配**而不是「无此成员」，
  与 §8.1 第 3 项一致，且这就是 §10.12 里建议选 (甲)（给上游查询加 stride 参数）的那个决策点。
* \`:1237\` \`getSameWidthCastLayoutFact\` 缺失，\`:1245\` 的 \`could not convert '{<expression error>, …}'\`
  是它的**级联错误**（同一个表达式里两个失败导致的事实构造失败），不是独立问题。
* \`:2017\`、\`:2091\` 等为其余缺失查询（\`getStoreLayoutFacts\`、\`getGroupBroadcastLoadLayoutFacts\` …）。

因此 Stage 3 的完成判据可以写得很硬：**\`planner_gap2.sh\` 的错误数归零**（外加 2 处 group-reduce 与 1 处 group-slot-load 的签名决策落地）。

## 16. 当前状态快照（给下一个接手的人/agent，随时更新）

### 16.1 树与分支

| 项 | 值 |
|---|---|
| 上游 worktree | \`.work/upstream-port/workspaces/vmi-layout-solver-upstream\`，分支 \`feature/vmi-layout-solver-upstream\`，HEAD **\`1c1bba8cc\`** |
| 上游构建 | \`.work/upstream-port/builds/vmi-layout-solver-upstream\`（Ninja，LLVM 19.1.7）|
| fork 树 | 主检出 \`feature/vmi-layout-decision-layers\`，测试与文档均已提交 |
| 已经落地的上游提交 | \`81b78d21c\`（头文件+conflict solver）、\`ee461ffa4\`（代价模型支持面）、\`2f62e9ef9\`+\`8bc4c428e\`（cast 事实与成本；后者修掉前者引入的接受面回归）、\`2d9d45b9b\`（传播器接口）、\`5e9493f37\`（cast-op 分类）、\`1c1bba8cc\`（vexpdif 行族）|

### 16.2 门禁（三条，全部可复跑）

| 门禁 | 命令 | 期望 |
|---|---|---|
| 上游基线 | \`cd <build>/test && llvm-lit -j 8 lit/vmi_new\` | **608 发现 / 606 通过 / 2 失败**，且失败集合恰为 \`vmi_integer_reductions_i8_invalid.pto\`、\`vmi_ptodsl_vunzip_vzip_validation.pto\` |
| planner 缺口 | \`.work/upstream-port/planner_gap2.sh [源码]\` | 当前 **25 个错误**（阶段 3 完成时应为 0）；脚本自带「确实在编译目标文件」的自证 |
| 全门禁（终局） | \`.work/upstream-port/validate_port.sh\` | 脏树直接拒绝；否则跑构建+基线+33 一致性用例+dump 差分 |
| 差分工具 | \`probes/run_conformance_dump.sh\` + \`probes/conformance_diff.sh\` | 对 \`probes/fork_conformance_dump.txt\`（33 文件 / 340 case）期望 \`IDENTICAL\` |

### 16.3 还缺什么（按依赖排序）

1. **Stage 3（进行中）**：12 个缺失查询 + 2 处 group-reduce 签名 + 1 处 group-slot-load 参数；完成判据 = \`planner_gap2.sh\` 归零。
2. **步骤 2c**：把 \`.work/upstream-port/step2c/VMILayoutPlanner.cpp\`（已删掉两份算子分类定义，2671 行）拷进上游树 + CMake 行；一致性工具 \`tools/pto-test-opt/pto-test-vmi-layout-cost-conformance.cpp\`（526 行，含 cost 与 lowering 两个 pass）随同。
3. **步骤 5**：按 §8.4/§10.13 替换 \`applyLayouts\`（片段已在 \`step5/\`），同时去掉上游 seed 写入点的硬错误、保留写入。
4. **步骤 6**：把 \`VPTOPack4StoreMaskNormalize.cpp\` 接线（配方 §8.8；必须在 planner 之后，否则完整链接缺 \`isVMILayoutCastOp\`——现在该符号已由 \`5e9493f37\` 提供，门槛已消除）。
5. **步骤 7**：批次 A–G，逐族 patch 在 \`.work/upstream-port/step7/\`（\`normalized.diff\` 42 hunk / +434 / −347 为证）；F1/F6 明确「不要移植」。
6. **步骤 8/9**：16 个用例要处理、96 个期望在步骤 5 后重测；方言改名只在上游树做（§13.5）；性能环境已冻结在 \`.work/upstream-port/perf/\`。

### 16.4 协同约定（血的教训）

* **构建/测量前先看 \`git status\`**：脏树上得到的是混合物（§13.2）；\`validate_port.sh\` 已把这条做成硬拒绝。
* **测完再断言失败集合**，并**先看失败原因**：\`ImportError: libPTOASCompiler.so\` 表示缺构建产物而不是回归（§14）。
* **自定义脚本要自证**（「确实编译了目标文件」「二进制新于源码」），否则绿灯无意义（§15.1）。
* **别把未提交的改动留在工作树里过夜**：别人无法验证、构建会被污染；每批提交（§13 起反复出现的教训）。

### 16.5 fork 侧基线（我自己复核过，不是转述）

| 套件 | 结果 | 说明 |
|---|---|---|
| \`lit/vmi_new\`（fork） | **568 发现 / 556 通过 / 12 失败** | 12 个是已知失败集合（见 §7.9 的 12 项）|
| \`lit/vpto\`（fork） | **548 发现 / 547 通过 / 1 失败** | 唯一失败 \`vpto/vmi_f4x2_to_bf16x2_vcvt_llvm.pto\` 是**既有失败**，与本次新增测试无关 |
| \`lit/vmi_new\`（上游 worktree） | **608 发现 / 606 通过 / 2 失败** | 两个上游自带失败；移植期间不得变差 |

这三个数字是后续一切「没退化」判断的参照，均已用各自的构建目录实测。
注意 fork 与上游的 **发现数不同**（568 vs 608）：上游套件更大，所以
**不要拿同一个失败数字在两侧比较**，只能各自与自己的基线比集合。

### 16.6 Stage 3 第一批已落地，缺口开始下降

* 提交 **\`1ea00d2c9 vmi: add the load layout fact enumeration the costed solver drives\`**（3 文件 +83/-1）：
  新增查询单元 \`VMILayoutSupportSolverQueries.inc\`（60 行）并声明 \`getLoadLayoutFacts\`；
  header +20、\`.cpp\` +4（把它纳入文本包含）。
* **planner 缺口 25 → 23**（\`planner_gap2.sh\` 实测），说明这条「批次完成 → 缺口下降」的反馈回路是按预期在工作的；
  剩下的 23 就是 §15.2.1 里那三类的余额。
* 提交粒度符合要求：**一批一提交**，工作树里只留当前正在做的那一批。

**复核方式说明**：缺口脚本编译的是**工作树当前状态**，所以它反映的是「HEAD + 在飞改动」；
作为趋势指标够用，但**不能当作某个提交的干净测量**。等 Stage 3 告一段落（工作树干净时）再跑
「重建 + 上游基线 + 失败集合」的正式门禁——这正是 §13.2 那条规矩的用法。

### 16.7 已备好的离线产物（都在 \`.work/\` 下，未进版本库）与待决策项

| 产物 | 路径 | 用途 |
|---|---|---|
| planner 预处理 + 一致性工具 + 应用说明 | \`.work/upstream-port/step2c/\` | 步骤 2c 的输入（planner 已删掉两份算子分类定义）|
| 步骤 5 插入片段 | \`.work/upstream-port/step5/\` | 5 个片段共 257 行（applyLayouts / mergePlan+selectLayoutPlan / createPropagator / addEquivalentValues / getExplicitLayout）|
| 步骤 6 接线脚本与说明 | \`.work/upstream-port/step6/\` | \`wire_step6.py\`（幂等、脏树拒绝）+ README |
| 步骤 7 逐族 patch | \`.work/upstream-port/step7/\` | \`normalized.diff\`（42 hunk / +434 / −347）+ F0–F10 + MANIFEST |
| 门禁工具 | \`.work/upstream-port/planner_gap2.sh\`、\`validate_port.sh\`、\`probes/\` | 缺口清单、终局门禁、dump 差分与参考 dump、各类探针脚本 |
| 性能复现环境 | \`.work/upstream-port/perf/\` | 9 个用例 + 19 组仿真结果 + 确切命令（§本轮 README）|
| Stage 3 存档（误伤回退时的凭证） | \`.work/upstream-port/step3-wip/\` | 已正常提交，仅留档 |

**待决策项（写明「要选一个」，不要留给下一个人猜）**：

1. **\`getGroupSlotLoadLayoutFact\` 的 \`sourceGroupStride\`**：选 (甲) 给上游查询加 stride 参数（建议）还是 (乙) 保留上游签名。见 §10.12。
2. **\`getGroupReduceLayoutFactsForLayout\` / \`getPreferredGroupReduceLayoutFact\` 的 \`VMIGroupReduceKind\`**：采用上游签名，并在 fork 侧调用处补出 kind（§8.1）。
3. **\`00890b827 Prefer E2B layouts\` 是否删除**：它与上游 \`0d7da8454\` 冲突；§2.1 的立场是「偏好属于靠后 tie-break，不值得对齐」，因此倾向删除，但需在步骤 5 之后用用例确认。
4. **vintlv/vdintlv 那 6 行窄掩码行**：已决定**不注入**（§8.7），保留上游更宽的行；若后续发现上游行覆盖不足再回头讨论。
5. **步骤 5 的 seed 硬错误**：已决定「去掉 emitError、保留写入」（§10.13）。
6. **步骤 4 遗留**：\`setSpineScopedCastOps\`/\`isSpineScopedCast\` 的删除必须与 spine 分析一起做（§8.5 R6），不要单独删。

### 16.8 E2B 那件事的证据更新：**行都在，我们的改动只是「谁优先」**

§16.7 第 3 项原来只写了「倾向删除 \`00890b827\`」。本轮查了两棵树的事实：

* **上游已经有同一组 E2B 直接行**：\`VMILayoutSupportTables.inc:621/623/625\`
  （\`E2B, G<8>(), gb(1|2|4), bits<16,32>\`），fork 侧对应 \`VMILayoutSupport.cpp:822/824/826\`——
  三行的 kind/组数/载体/元素位宽一致。
* 也就是说 E2B 这个能力**不是我们独有的 delta**；\`00890b827\`（7 文件 +155/-35）
  影响的是**在这些行之间谁先被选中**（偏好/优先级），外加它对 4 个测试期望的影响。
* 上游那个「同测试冲突」的提交 \`0d7da8454\` 是 **FP4 布局与物理 part 收窄**，不是 E2B 偏好策略；
  两者只是都改了 \`vmi_layout_assignment_group_slot_broadcast_load_e2b_b16.pto\` 的期望，
  所以冲突是**局部的、只在那几条期望上**。

这与 §2.1 的立场（偏好是决策链靠后的 tie-break，不值得对齐）合起来给出结论：
**\`00890b827\` 应按「删除候选」处理**，而不是「必须合并」；删掉之后由我们的代价链自己决定选哪一行。

**仍未验证的部分（写明，不假装已证）**：我没有逐列比对三行的全部字段（只比了 kind/组数/载体/元素位宽），
也没有证明 \`00890b827\` 的改动**纯粹**是排序（它还动了 propagation 与 support 的若干处）。
因此落地时的判据是行为性的：步骤 5 之后跑
\`vmi_layout_assignment_group_slot_broadcast_load_e2b_b16.pto\`、\`group_broadcast_load_e2b_layout_opt.pto\`、
\`group_broadcast_load_contiguous_fallback.pto\`、\`vmi_extf_8bit_factor_contract.pto\` 四个用例，
看我们的 solver 在没有该偏好的情况下是否仍选中 E2B/复合行。

### 16.9 Stage 3 前两批已过基线门禁；同时**第五次**撞上测量上下文问题

**好消息（在干净状态上测得的）**：Stage 3 前两批（\`1ea00d2c9\` load 枚举、\`8bcf023dd\` store 枚举）落地后，
\`ninja pto-test-opt\` + python 模块 exit 0、0 告警，\`llvm-lit -j8 lit/vmi_new\` 的**失败集合恰为**
\`vmi_integer_reductions_i8_invalid.pto\` 与 \`vmi_ptodsl_vunzip_vzip_validation.pto\`——
即支持层的新增查询对上游决策路径仍是行为中性的。

**坏消息（同一轮里发生的）**：我在同一条命令里跑了两次 lit，第一次（干净状态）是那 2 个失败，
第二次却报 **563 / 45**。原因不是回归，而是**两次之间 Stage 3 又改了文件并重建**——
即 §13.2 那条陷阱**在一次命令内部**又发生了一次。事后 \`git status\` 确认树又变脏（2 个文件）。

**因此门禁脚本本轮加了一项**：\`validate_port.sh\` 在打印裁决前多打一行
\`PROVENANCE  HEAD <sha>  dirty <n>\`，并要求它与第 0 步打印的状态**一致**。
这样任何一次结果都自带出处；两次状态不一致就说明测的是混合物，结论作废。

> 这已是同族问题的第五次（旧二进制、脏树、缺 .so、量具未自证、**测量期间被改写**）。
> 它们的共同点始终是：**结论必须能追溯到「测的到底是哪个状态」**。
> 规矩已经从「先看 git status」升级为「**测量前打印、测量后再打印，两者必须一致**」。

### 16.10 复核 Stage 3 已提交批次：**纯增量，未碰共享接受面**

对前两批逐个查了「有没有改到上游已经在用的代码」：

* \`1ea00d2c9\`：\`VMILayoutSupport.cpp\` 的改动**只有**一行注释更新 + 新增 \`#include \"VMILayoutSupportSolverQueries.inc\"\`，
  唯一的那处删除就是被替换掉的旧注释；查询实现 60 行全在新单元里。
* \`8bcf023dd\`：根本没有改 \`VMILayoutSupport.cpp\`（header +12、新单元 +36）。

也就是说新查询一律进**新单元**，旧漏斗与旧查询的行为一字未动——这正是 \`8bc4c428e\` 修补过的那条规则，
现在被稳定执行。缺口计数同步从 25 → 22，方向与幅度都符合预期。

### 16.11 步骤 5 也已脚本化：\`step5/wire_step5.py\`（默认 dry-run，按文本锚点而非行号）

把 §8.4 的编辑做成锚点式脚本：**默认只 dry-run 并报告它做了什么 + 括号平衡**，\`--apply\` 才写文件。
当前 dry-run 结果（对 HEAD \`8bcf023dd\` 的上游 \`VMILayoutAssignment.cpp\`，2333 行 → 2415 行，括号平衡 0）：

* 加 \`VMILayoutPlanner.h\` 包含；插入 \`getExplicitLayout\` helper；
* **按 §10.13 的决定**：\`setNaturalLayout\`/\`setPreferredLayout\` 的两处「冲突布局」硬错误被移除，**写入保留**；
* 删除 11 个 seed 辅助函数：\`hasRequestedLayout\`、\`hasLayoutAssignment\`、\`requestDataLayoutSeeds\`、\`applySeedRequest\`、
  \`requestDataUseSeeds\`、\`requestMaskUseSeeds\`、\`runSeedPhase\`、\`requestExplicitLayouts\`、\`runLayoutSeedPhases\`、
  \`requestLateLayouts\`、\`requestFallbackLayouts\`；
* 用我们的 \`addEquivalentValues\` + \`createPropagator\` 替换 \`addEquivalentLayoutValues\`；
* 用我们的 \`applyLayouts\`（194 行）替换上游版本；插入 \`mergePlan\` + \`selectLayoutPlan\`。

**顺带一个好处**：锚点式编辑第一次跑就报了 3 处不匹配——\`hasRequestedLayout\`/\`hasLayoutAssignment\` 实际返回 \`bool\`
（不是 \`LogicalResult\`）、helper 的插入锚点是 \`bool containsVMIType(Type type) {\`（不带 \`static\`）。
这类漂移若用行号编辑会**静默改错地方**；锚点让它当场报错。修正后 18 项全部命中。

**应用前置**：仍要等步骤 2c（planner 进树）——\`applyLayouts\` 依赖 \`selectCostedVMILayoutPlans\`/\`commitVMILayoutPlan\`。
应用后跑 \`validate_port.sh\`（含 PROVENANCE 行）复核。

### 16.12 批次 D（F5）再分类：更像**重构**而不是新增能力

把 F5 的 12 个 hunk 里真正“只有我们有”的符号抽出来看，只有两个名字：
\`materializeStagingDeintToContiguousMaskLayout\` 与 \`materializeStagingContiguousToDeintMaskLayout\`（各 3 处调用）。
但两棵树的命中情况是：**上游 2 个文件里有、fork 1 个文件里有**——也就是说它们上游也有。
而 F5 另外涉及的 \`createPredicateIntlv\`（上游 \`VMIToVPTODataLayoutInternals.cpp\`）、
\`materializeAdjacentMaskGranularityConversion\`（上游 \`VMIToVPTOPatternInternals4.cpp\`）同样上游都在。

结论：**F5 的主体是对既有函数的改写/搬位，而不是新增能力**。因此批次 D 的正确做法与批次 F 一样——
**逐 hunk 先问「上游这一版是否已经等价」**，等价的按 (a) 保留上游；预计真正需要移植的远少于 12 个 hunk。
这与 §13.9（F8）同型：**hunk 数不代表工作量**，先分类再动手。

> 待办：批次 D 落地时，对每个 hunk 记录「上游已有 / 需合并 / 需新增」三选一，并给出 file:line 证据——
> 与批次 C（§10.12）、批次 F（§13.9）保持同一标准。

### 16.13 Stage 3 状态与两项决策（采纳实现方提议）

Stage 3 已提交 **8 批**（缺口 25 → 13 → 后续继续下降），全部**纯增量**：\`1ea00d2c9\` load、\`8bcf023dd\` store、
\`93df6702e\` cast、\`65e8b9ab6\` 等宽 cast、\`70034b64f\` cast 关系守卫（F2，正确性关键）、\`443a0ae55\` 非分组 reduce、
\`8da4a1823\` vintlv/vdintlv 枚举。剩余顺序：group broadcast load 枚举 → group iota → vdhist/vchist preferred → 3 处签名差异。

**决策 1（采纳）**：\`getGroupSlotLoadLayoutFact\` 的 stride **不改上游签名**，而是在旁边**加一个重载**
（fork 签名 + \`Value sourceGroupStride\`）；传空 Value 时行为与上游完全一致，传 stride 时是 fork 语义。
这比我原先建议的「给上游查询加参数」更严格地满足「不碰上游接受面」，且 2c 的调用点无需改写。

**决策 2（采纳）**：group-reduce 的两处**不做无 kind 的垫片**——那会静默丢掉 \`group_reduce.addi\` 的 i16 原生求和行（假规则）。
保留上游 kind 感知签名，由 2c 在 5 个调用点（\`VMILayoutPlanner.cpp:810/2418/2444/2497/2522\`）补 \`getVMIGroupReduceKind(op)\`；
\`vcadd/vcmax/vcmin\` 该 helper 返回 \`Other\`，与上游自身处理一致。

**已预登记的差异（不是缺陷）**：fork 的三条「全宽 dense store 偏好行」**故意不注入**共享表（否则会放宽上游的
\`getPreferredStoreLayoutFact\`）。后果是 \`VMILayoutPlanner.cpp:2038\` 处我们拿不到偏好行，
\`preferencePenalty\` 为 0 而非 1——**只影响靠后的 tie-break，绝不影响合法性**。
这是步骤 5 之后 **dump 差分第一处「允许出现」的差异**，修法是在 planner 驱动的 fork 专属包装里补，而不是改共享表。

### 16.14 第四类同族陷阱：**宿主机负载**

本机同时在跑 PTOAS-4 的全量 lit（64 核上负载 69），实现方有两次 lit 出现 15–17 个无关失败，
单独重跑或干净重跑即通过。它加了 \`probes/gate.sh\`：保留两份日志、宣告回归前**重测一次**，且**从未在受污染的运行上提交**。

这与前几类（旧二进制、脏树、缺 .so、量具未自证、测量中被改写）同族：**结论必须能追溯到「测的到底是哪个状态」**。
新增规约：报失败前先看负载并重测一次；受污染运行不得作为提交依据。

## 17. 步骤 2c 已落地；差分门禁首跑把差异分成三类

提交 **\`e46bb4110 vmi: bring the costed layout planner and its conformance tool onto the upstream base\`**：
planner（已删两份算子分类定义、5 处 group-reduce 调用点补 \`getVMIGroupReduceKind(op)\` 并把 kind 透传给
不带 \`op\` 的 \`rememberGroupReduceRelationLayouts\`）、一致性工具（两个 pass）、注册调用与两处 CMake。
**验证**：\`ninja\` exit 0 / 0 警告；\`--help\` 里能看到新 pass；\`lit/vmi_new\` = **608 / 606 / 2**（上游基线未退化）。

**差分门禁首跑**（上游二进制 vs fork 参考 dump）：157 条 conformance 行 vs 368 条，21 条错误。逐类归因：

| 类别 | 数量/样例 | 归属 |
|---|---|---|
| 我们的 pass 报「marker has no exposed relation」 | **11 条** | **真实枚举缺口**——需逐例定位是哪个算子/形状在上游侧枚举不出关系 |
| pmode 方言漂移 | 4 条：\`vmula\`/\`vexpdif\`/\`vcmp\`/\`vadds\` 报 \`invalid pmode \"merge\"\`/\`\"zeroing\"\`，上游只收 \`zero\` | **步骤 8 的输入重写**（我们的 conformance 输入用了旧 pmode 拼写） |
| legacy 算子已删除 | \`custom op 'pto.vmi.addf' is unknown\` | **步骤 8 的输入重写**（上游删掉 legacy 方言；dump 只跑 cost pass，未走 unified→legacy 降级） |
| 上游硬约束 | \`group_reduce_addi\` 8-bit integer reduction VMI-UNSUPPORTED | 上游护栏，属已知约束 |

另有**纯噪声**一类：\`loc(\"...\")\` 里的路径不同（我把用例拷到 \`/tmp/conf_tests\` 跑），
会让 33/33 文件都“显示不同”。**下次差分前必须先规范化路径**，否则真实差异会被噪声淹没。

> 门禁的价值在这里体现出来了：它没有给「差不多」的模糊结论，而是把 134 vs 339 的巨大差异**拆成可行动的三类**——
> 其中两类是已知的步骤 8 方言工作，一类（11 条）是真实的枚举缺口。

## 18. 步骤 5 落地（决策权已交回我们的 solver）与其真实后果

**提交 \`657590cd7 vmi: hand layout decisions to the costed planner\`**（上游 worktree，树已干净）。
\`applyLayouts\` 不再跑上游的 seed 阶段机制：建传播器 → 向代价 planner 要整份计划 → 提交 →
由 fork 版的结构化 seeding 与边匹配兜住 planner 未赋值的传输值。11 个 seed 辅助函数、4 处 \`propagator.run()\`
与四个 phase 入口全部删除；约束遍历保留（现在只提供它建立的等价关系）。按 §10.13，两处 seed 写入保留、硬错误移除。

### 18.1 立刻测量到的效果（这是关键路径上最重要的一次读数）

| 指标 | 值 |
|---|---|
| \`lit/vmi_new\` | **608 发现 / 500 通过 / 108 失败**（基线 606/2）|
| FileCheck 期望不匹配 | ~130 处（\`no match found\` 74、\`ASSIGN-SAME\` 27、\`CHECK\` 22、\`ASSIGN\` 16、\`CHECK-DAG\` 4）|
| **\`VMI-LAYOUT-CONTRACT: no complete legal VMI layout plan\`** | **25**（硬失败）|
| **\`VMI-UNSUPPORTED: no legal VMI layout relation\`** | **22**（硬失败）|
| **\`type of return operand 0 (…)\`** | **11**（形似 §8.5 R7）|
| \`group_load requires …\` | 3 |

**判读**：期望差异是**决策引擎更换的必然产物**（正是 §8.9 预判要重测的那 96 个用例），
属于步骤 8 的工作量而非缺陷；**约 50 个硬失败才是真问题**。

### 18.2 硬失败最可能的成因（有明确指向，不是猜测）

Stage 3 收尾清单第 4 条写明：**若干 fork 独有行被刻意不注入共享表**（6 条 ensure、2 条 ensure-mask、
6 条 legal-mask-granularity、3 条 dense-store 偏好、1 条 group-broadcast-load 及其 direct 行），
理由是注入会放宽**上游**的接受面。但那些行对我们的 solver **恰恰是候选集**——
上游决策链不需要它们，我们的 solver 需要。于是「no complete legal plan / no legal relation」与此高度吻合。

**下一步的正确解法**（也是 Stage 3 自己点出的方向）：把候选来源从「共享表」改为
**「共享表 + fork 独有行，且后者只对 solver 路径可见」**——即 fork-only wrapper。
这既不放宽上游接受面（不违反那条铁律），又让 solver 拿到完整候选集。

### 18.3 教训（又一次同族）

步骤 5 的编译期暴露了 3 个只有编译器能发现的问题（1 处残留 \`template <typename RequestTy>\` 导致重载解析失败、
2 处删掉硬错误后遗留的未使用变量），**dry-run 只能保证括号平衡与锚点命中，编译才能暴露签名与未用变量**。
所以「脚本化改造」之后的下一动作必须是编译，而不是直接提交。

### 18.4 差分门禁的**路径规范化**后果：21/33 文件不同，且主因是**输入方言过时**（门禁前提被修正）

我把两份 dump 里 \`loc(\"…/vmi_layout_cost_conformance_X.pto\")\` 的目录部分规范化后再差分（这是 §17 记下的必须步骤）：

* 参考 dump 4781 行 vs 上游 dump **1607 行**；**21 of 33** 文件不同（不是 33/33——之前那 12 个“不同”纯属路径噪声）。
* 差异的主因不是 solver，而是**我们的 conformance 输入在上游方言下根本过不去**：
  \`custom op 'pto.vmi.addf' is unknown\`（legacy 算子已删）、
  \`'pto.vmi.vcmp' op invalid pmode \"merge\"; expected \"zero\"\`（pmode 拼写变了）。
  于是整份文件的用例在上游侧全部消失（例：elementwise 文件 A-only 243 行 vs B-only 3 行）。
* 只有一部分差异是真实的关系枚举差别（例：\`ensure_layout\` 文件 A-only 217 行）。

**对计划的修正**：**在把 conformance 输入做方言更新之前，dump 差分不能当验收门禁**——
否则它无法区分「solver 候选集有缺口」与「输入本身解析不了」。正确顺序是：

1. 先把 33 个一致性输入的方言更新到上游拼写（legacy → unified、pmode 归一），
2. **重新生成 fork 侧参考 dump**（同一批输入、fork 二进制），
3. 再跑差分——此时的差异才可信地指向 solver/枚举问题。

> 这条也解释了此前 134 vs 339 的“巨大缺口”里有多少是假象：主因是输入漂移，而非候选集收窄。
> 这正是“量具必须先自证”的又一例：**门禁本身要先在两边都能正确解析输入，其读数才有意义**。

### 18.5 一致性输入的方言漂移**已量化**（步骤 8 的第一批具体工作）

对 fork 的 33 个一致性输入做只读统计，漂移集中在很小的范围：

* **pmode 拼写**：5 个文件出现 \`pmode = \"merge\"\` 或 \`\"zeroing\"\`（上游只收 \`\"zero\"\`）——
  按文件/次数：\`cmp_merge_invalid\`(1 merge)、\`same_layout_invalid\`(1 merge)、\`unified_merge_invalid\`(1 merge)、
  \`vexpdif_invalid\`(1 merge)、\`unified\`(3 zeroing)；其余文件已是 \`\"zero\"\`（合法）。
* **legacy 算子拼写**：\`pto.vmi.addf/addi/absf/absi/cmpf/cmpi/exp/ln/maxf/maxi/minf/mini…\` 各约 2 次，
  集中在 \`elementwise\`（正是差分里 A-only 243 行 / B-only 3 行那份）与 \`unified\`（含 \`legacy_generic\` 用例）。

**结论**：差分门禁的「134 vs 339」缺口，其可解释部分就落在这 5–6 个文件的方言更新上；
这是**有界、可核对**的工作量，而不是大范围重写。做法（按 §18.4 的顺序）：先更新输入 → 用同一批输入重生成 fork 侧参考 dump → 再差分。

#### 18.5.1 搬运脚本的 legacy 检测 bug 已修并**自证**

上一轮扩展 \`step8/copy_tests.sh\` 时，legacy 算子检测报 0——与「\`elementwise.pto\` 明确含 \`pto.vmi.addf\`」的事实矛盾，
说明正则（多层引号/转义嵌套）没生效，会**静默漏掉**需要真正重写的文件，而它们正是差分缺口的主因。

已改为不含反斜杠转义的写法 \`pto[.]vmi[.](addf|addi|…)[^a-z]\`，并用**已知含 legacy 算子**的文件做自证：

    copy_tests.sh --dry-run --only vmi_layout_cost_conformance_elementwise.pto
      LEGACY-OPS vmi_layout_cost_conformance_elementwise.pto : needs a real rewrite, not a rename
      copied=1 … legacy_reports=1  (dry=1)

规矩再次生效：**量具必须用它本该命中的正例去验一次**，否则 0 报告与「真的没有」无法区分。

### 18.6 换引擎后的 108 个失败：**逐例分类**完成（42 真缺陷 / 49 期望差异 / 17 待补分类）

新增 \`probes/classify_failures.sh\`：对每个失败用例取**第一条 \`pto-test-opt\` RUN** 实跑，按「流水线是否自己报错」而不是按文本关键词分类。
结果：**PIPE-ERROR 42 / CHECK-DIFF 49 / MISSING+NOPTOOPT 17**（后者首条 RUN 走 \`ptoas\` 或位于 \`opt/\`，需另一条路径）。

42 个真缺陷的族分布：

| 族 | 数量 | 初判 |
|---|---|---|
| \`VMI-LAYOUT-CONTRACT: no complete legal VMI layout plan\` | **16** | solver 枚举不出合法 plan——与「未注入的 fork 独有候选行」假设一致 |
| \`type of return operand 0 (…)\` | **14** | 函数返回类型校验失败——疑似 §8.5 **R7**（上游 \`rewriteFunctionType\` 与我们 ABI 边界处理不一致）|
| \`VMI-UNSUPPORTED: no legal VMI layout relation is registered\` | **10** | solver 选中的关系在 lowering 侧没注册——与 Stage 3 点出的「capability-locked」同类 |
| 其他 | 2 | 一条 \`failed to apply conversion patterns\`、一条消息为空 |

**这两组数字合起来说明**：换引擎引入的**真实缺陷是 42 个**，且**三个族各有明确嫌疑成因**，
不是一堆散乱失败；而 49 个期望差异属步骤 8 重测（既定做法：步骤 5 之后重新测量）。

### 18.7 「no complete legal plan」族的第一处归因：**ensure_mask_granularity**

逐个跑 16 个该族用例并读诊断，第一个（\`vmi_group_reduce_addi_i16.pto\`）就给出了明确指向：

    error: VMI-LAYOUT-CONTRACT: no complete legal VMI layout plan exists for this component
    note: see current operation: %0 = \"pto.vmi.ensure_mask_granularity\"(%arg1)
          : (!pto.vmi.mask<128xb32>) -> !pto.vmi.mask<128xb16>

也就是说：**枚举不出来的是 mask 粒度转换（b32 → b16）那条关系**。
而 Stage 3 明确没有注入的 fork 独有行里，恰好包括
**6 条 \`kLegalMaskGranularityCastLayoutPatterns\`** 与 **2 条 \`kEnsureMaskLayoutPatterns\`**。
两者对上——这是对「未注入的 fork 独有候选行导致 solver 无解」这一假设的**第一份直接证据**
（此前只是“高度可疑”，现在是“诊断里点名的就是那批行覆盖的关系”）。

**修法**（与既定共识一致）：fork 专属包装——共享表 + fork 独有行，**只对 solver 可见**；
不注入共享表，因此上游接受面不变。分诊任务已收到该族 16 个用例的目标清单与这一归因。

### 18.8 no-plan 族的逐 component 量化（分诊任务的 VMI_LAYOUT_DIAG 输出）

分诊任务加了 \`VMI_LAYOUT_DIAG\` 环境变量诊断（会打印失败 component 内每个算子的**可用关系与端口布局**），
其 \`/tmp/diag_all.txt\` 给出了此前只能用推断的量化证据。以 \`vmi_group_reduce_addi_i16.pto\` 为例：

    solver failed: component ops=3
      pto.vmi.ensure_mask_granularity relations=1
         ports: contiguous -> contiguous, lane_stride = 2   direct=0
      pto.vmi.group_reduce_addi relations=1
         ports: contiguous, contiguous -> num_groups = 8, slots = 8, lane_stride = 2
      func.return relations=1
      fixed contiguous

**每个算子只贡献 1 条关系**，链因此无解；其它用例可见 \`relations=2\`（group_reduce_addf）、\`relations=3\`（ensure_mask_granularity），
普遍偏薄。这与「未注入的 fork 独有行（6 条 legal-mask-granularity + 2 条 ensure-mask 等）使 solver 候选集小于 fork 侧」一致：
上游决策链靠 seed 直接定布局、不需要这些行，而我们的 solver 靠**候选集搜索**——缺行即无解。

因此「**fork 专属包装**（共享表 + fork 独有行、仅 solver 可见）」这条修法有了逐 component 的支撑，
而不再只是假设；但 16 个用例仍需逐个过完，因为**成因未必同一**（也可能是结构性等价边缺失，修法不同）。

### 18.9 no-plan 族的**竞争假设**：ABI 边界被当成固定赋值也可能是主因

把 \`VMI_LAYOUT_DIAG\` 输出按用例汇总后，看到一个与「缺候选行」并列的成因信号：

    group_reduce_s64.pto    ops=2  group_reduce_addf=1, func.return=1
    group_reduce_slots8.pto ops=2  group_reduce_addf=1, func.return=1
    group_reduce_s256.pto   ops=2  group_reduce_addf=3, func.return=1
    group4_broadcast_shape_matrix.pto ops=3 broadcast=3, group_broadcast=1, func.return=1
    group_reduce_addi_i16.pto ops=3 ensure_mask_granularity=1, group_reduce_addi=1, func.return=1, **fixed contiguous**

**几乎每个失败 component 都含 \`func.return\`**，其中一例明确打印 \`fixed contiguous\`。而 \`func.return\` 在我们这边是被当作
**固定赋值**处理的（\`collectFixedAssignments\` / \`getABIBoundaryLayout\`）。若边界被钉死为 contiguous，
而 reduce/broadcast 唯一可行的关系产出的是 group-slots 布局，那么该 component **无论加多少候选行都无解**——
这时正确的修法就不是「补行」，而是「允许 ABI 边界发生转换」。

两种成因（候选集过薄 vs 边界固定赋值）**在诊断里同时存在**，因此要求分诊任务先用小用例（s64 / slots8）
验证「放宽边界后是否可解」，再决定修哪一边；混合的情况要逐例说明，不许一概而论。

### 18.10 **竞争假设被证实**：无注解 ABI 边界 + 固定赋值 = 无解（与缺行无关）

读最小用例即可确认。\`vmi_layout_assignment_group_reduce_s64.pto\`：

    func.func @…(%source: vreg<512xf32>, %mask: mask<512xpred>) -> vreg<8xf32> {
      %out = pto.vmi.vcadd %source, %mask {group = 8, reassoc} : … -> vreg<8xf32>
      return %out : vreg<8xf32>
    }

**签名里没有任何布局注解**，因此 ABI 边界按 \`getABIBoundaryLayout\`（显式布局否则 contiguous）解析为 \`contiguous\`，
\`func.return\` 也就被钉在 contiguous；而 \`vcadd {group = 8}\` 只有 **1 条关系**，产出的是 group-slots 布局——
对它是**不可达**的。于是该 component **无论加多少候选行都无解**。

真正的缺陷在**边界的处理方式**：我们的 \`collectFixedAssignments\` / \`commitVMILayoutPlan\` 把 ABI 边界当成
**固定赋值且没有转换边**；而上游原来的 seed 路径允许传播器在 use 处物化一次转换
（\`requestFallbackLayouts\` + use-conflict 机制）。修法是**允许边界发生转换**（或从可达关系反推边界布局），
而不是补行。**最快的路径**是拿 fork 自己的 planner 对照这些用例（fork 的这些 reduce 用例是通过的），
看 \`getABIBoundaryLayout\` / \`collectFixedAssignments\` 在「结果被无注解函数直接 return」时的行为差在哪。

### 18.11 定论：不是「fork 的做法」，而是上游的**边界转换**行为在计划式流程里丢了

按 §18.10 的指引去对照 fork 的同名用例，**发现它根本不是同一个测试**：

* **fork 版**：\`vcadd → pto.vmi.vstore %out, %dst[%off]\`（把归约结果写进内存，**没有 vreg 函数返回值**）；
* **上游版**：\`vcadd → return %out : vreg<8xf32>\`（归约结果直接作为函数返回值）。

因此 **fork 的 planner 从未走过「vreg 作为 ABI 边界」这条路径**——所以「照 fork 的做法」不是答案。
真正的事实是：**上游自己的 solver 能过这个用例**，因为它把边界保持为 contiguous、
**在 use 处物化了一次转换**（\`requestFallbackLayouts\` + use-conflict 机制）。
而我们的计划式流程把 ABI 边界当成**固定赋值、没有转换边**，于是只要算子的唯一关系产不出边界布局就无解。

**正确修法**：**边界布局保持不变**（函数结果布局不能改，调用方依赖它），
让计划在**生产者的布局与边界之间允许一条转换边**——即上游 use-conflict 产生的那个「return 前 ensure_layout」。
在 planner 里表述为：边界是**被钉住的值 + 允许转换**，而不是「任何关系都不得与之不同的赋值」。

验证用最小的四个用例（\`group_reduce_s64\`、\`slots8\`、\`s256\`、\`group4_broadcast_shape_matrix\`），
它们的 component 里都含 \`func.return\`，改动前后比 \`no-plan\` 计数即可。

### 18.12 移植分支已推送到 fork（23 个提交，工作不再只存在于本机）

\`git push fork feature/vmi-layout-solver-upstream\` 成功：
\`github.com:mouliangyu/PTOAS.git\` 上现有分支 **\`feature/vmi-layout-solver-upstream\`，含 \`origin/master..HEAD\` 的 23 个提交**
（Stage 1/2/2b/2c/3 的 14 批 + 步骤 4/5 + 施工与门禁相关提交）。

动机很直接：这套移植的成果此前**只存在于本机 worktree**——一旦会话或磁盘出问题就全丢。推送后，
任何人（包括下一个 agent）都能从远端检出到「决策权已交给我们的 solver」这一状态，
并继续未完成的 42 个真缺陷与步骤 6/7/8/9。
参考命令（恢复工作环境）：

    git -C <repo> worktree add <dir> -b feature/vmi-layout-solver-upstream fork/feature/vmi-layout-solver-upstream

注意：**未提交的诊断埋点不会被推送**（它必须留在本地并在提交前剥掉，见 §18.13 的提醒）。

### 18.13 换引擎后的**第一个量化改善**：108 → 89 失败

分诊任务实现「无注解 ABI 边界不再钉住布局」（§18.11 定论）后，其 \`lit/vmi_new\` 运行（\`/tmp/hf_lit_fix1.log\`）：

    Total Discovered Tests: 608
      Passed: 519 (85.36%)
      Failed:  89 (14.64%)

对照修复前 **500 / 108**，即**净减 19 个**——这同时证明该修改**不是**把 no-plan 失败换成 return-operand 失败的假持平，
而是真实解决了一批（no-plan 族原 16 个及其连带）。

**出处与限制（必须标注）**：
1. 这是**分诊任务的测量**（我读其日志），非我独立复跑；且在工作树**未提交**状态下取得，属 WIP 读数；
2. 该日志是**非 verbose** 运行，故按族的关键字统计在其中为 0、不具信息量；按族增量需用 \`classify_failures.sh\`。

**因此尚未结案**，仍需：提交 → 我独立复跑按族分类 → **单独确认上游配置（planner 不驱动）下失败集合仍为那两个既有用例**，
确认「修好我们的 solver」没有悄悄改坏上游路径。

### 18.14 更正：没有回归，两族根因独立且不相交

上一轮我把 hf_iso_stride.log 的 2/17 当成回归并报警——错了。那是分诊任务有意做的对照实验：临时撤掉边界修复，单独测 stride 修复。

| 修订 | 全量 | 相对 108 |
|---|---|---|
| A 两处修复 | 608 / 519 / 89 | 108 到 89（修 19，回归 0）|
| B 仅 stride | 608 / 502 / 106 | 108 到 106（修 2，回归 0）|
| B 的 19 用例子集 | 2/19 通过 | 正是那 2 个 stride 用例 |

两族加法式且不相交：stride 修 vmi_vsstb 与 vmi_to_vpto_stride_store（2 个），边界修另外 17 个，集合不重叠。

* stride 根因：上游 VMIStrideStoreOp 是 5 个操作数（value, dst, offset, block_stride, mask），fork 是 6 个（mask 前多一个 repeat_stride）。移植后的 planner 仍按 fork 编号取 operandPort(5)，越界，该算子唯一的关系被拒，报 no complete legal plan。
* 边界根因：planner 把无注解函数结果钉在 contiguous ABI 边界；上游路径从不这么做（rewriteFunctionType 采纳返回值布局）。而且不存在从 gs(8, slots=1) 到 contiguous 的 ensure_layout 行，所以 §18.11 提的“在边界物化一次转换”对这些形状不可实现，该建议作废，正确做法是边界采纳返回值布局。

批准的落地顺序：剥掉诊断，先提交 stride（108 到 106），再提交边界（106 到 89），各带数字。随后进攻统一算子（vload/vstore/vcvt 在 -vmi-lower-unified-to-legacy 之前就被布局赋值，planner 没有对应关系分支）与 group_store/group_broadcast 的候选行——后者是“fork 专属包装是否必要”这个悬案终于由数据回答的地方。

### 18.15 修复的**按族效果**：type of return operand 族归零

从分诊任务修复后的细节日志 hf_detail_after.log（5190 行）统计各族关键字：

| 族 | 修复后 | 对照修复前 |
|---|---|---|
| type of return operand | 0 | 14，全部消除 |
| VMI-LAYOUT-CONTRACT | 16 | 原 16（含 FileCheck 侧同类文本）|
| VMI-UNSUPPORTED | 28 | 原 10（口径可能不同，见下）|

type of return operand 归零说明：无注解边界不再钉住布局这处修改不只是减少总数，而是确实解决了那一整族（函数结果类型与返回操作数不一致），与总失败 108 到 89 的读数相互印证。

仍未定论：这是分诊任务的日志，且 VMI-UNSUPPORTED 的 28 与修复前的 10 口径不同（可能含子类或不同 flag 组合），不能直接当作变多。要它提交后由我独立跑 classify_failures.sh 才能给按族定论。

### 18.16 两处修复已提交，且由**我独立复核**通过（108 到 89 复现）

提交：

* 914368bec stride 修复，VMILayoutPlanner.cpp +10/-1
* 453ed8e3b 边界修复，VMILayoutPlanner.cpp +19/-3

独立复核（pre: HEAD 453ed8e3b dirty 0，即测的就是这两个提交；post 显示它在测量结束后才开始下一处改动，故读数可归因）：

| 检查 | 结果 |
|---|---|
| 改动范围 | 两个提交都只动 VMILayoutPlanner.cpp，不可能触碰上游接受面或支持层 |
| 构建 | ninja pto-test-opt 与 python 模块 exit 0 |
| 全量 | 608 发现 / 519 通过 / 89 失败 |

因此 108 到 89（净修 19、无回归）由我自己的运行复现，而非转述实现方的日志。
剩余：89 个失败中约 23 个仍是真缺陷（VMI-LAYOUT-CONTRACT 16 与 VMI-UNSUPPORTED 28 需去重与分子类），其余为期望差异（步骤 8 重测）。

### 18.17 统一算子族修复的读数：89 到 80

分诊任务对统一算子族（vload/vstore/vcvt，在 -vmi-lower-unified-to-legacy 之前就被布局赋值）的修复，其全量测量：

    Total Discovered Tests: 608
      Passed: 528 (86.84%)
      Failed:  80 (13.16%)

即累计 108 到 89 到 80（先 stride + 边界，后统一算子族），改动限于 VMILayoutPlanner.cpp 与 VMILayoutAssignment.cpp 两个文件，尚未提交。
依据（取其注释大意）：上游自己的 collect() 对这些统一拼写同样不声明关系，布局由 lowering 从交到它手上的值决定，因此关系提供者缺少该拼写分支不应使 component 失败。

待其提交后由我在干净状态独立复核：改动范围、构建退出码、失败集合断言、按族增量；并特别核对 VMILayoutAssignment.cpp 部分是否结构必需且未放宽判定。

### 18.18 三处修复**由我独立复核**：108 到 80（净修 28，无回归）

在干净状态（pre: HEAD 5cc2fe723 dirty 0）跑构建与全量：

| 提交 | 作用域 | 效果 |
|---|---|---|
| 914368bec | 1 文件 +10/-1 | 108 到 106（stride_store 的 mask 操作数）|
| 453ed8e3b | 1 文件 +19/-3 | 106 到 89（无注解函数结果不再钉住）|
| 5cc2fe723 | 2 文件 +89 | 89 到 80（pre-lowering 统一拼写）|

复核读数：ninja exit 0；lit/vmi_new = 608 发现 / 528 通过 / 80 失败 —— 与实现方汇报一致，由我自己的运行复现。
三个提交的失败集合都是前一个的**严格子集**（0 回归）。

**对我那处建议的进一步更正（已被证明而非论证）**：pass 不变式（PTOValidateVMIIR.cpp：layout-assigned VMI IR requires
vreg with layout）意味着统一算子的结果**不可能保持无布局**；上游流程里 pto.vmi.vload/vstore/vcvt 归 lowering 所有，
其结果取 pass 给无类型传输实参的稳定 dense 主布局——这才是那 9 个统一算子用例通过的真正原因。
我在 §18.11 提的「在边界物化转换」因此既不必要也不成立。

**剩余（实现方列的进攻顺序）**：group_store 族（7）、vmi_compact_group_broadcast（1）、vmi_layout_assignment_group_slot_load 暴露的 extf 缺口、
以及其余约 11 个 no-plan component（都含一个关系集为空或存在真实结构冲突的非 return 算子）。

### 18.19 group_store 族的尝试：测量无改善 → 回退（不留无收益改动）

实现方对 group_store 族（7 个用例）的改动做了完整测量，读数仍是 608 / 528 / 80，即**没有净收益**；随后工作树被回退：
HEAD 仍为 5cc2fe723、dirty=0、提交数未增加。

这体现了一条我们反复强调的纪律：**没有净收益的改动不留**——改了多少行、思路多漂亮都不算进展，判据是数字。
当前树干净地停在最后一个已验证的收益点：608 / 528 / 80（硬失败已修 28）。

group_store 族因此需要**另找根因**（初次的修法不成立），而不是继续在同一个方向上打磨。

### 18.20 状态快照（本节为准，替代 §16 的旧读数）

| 项 | 值 |
|---|---|
| 上游 worktree HEAD | 5cc2fe723，工作树干净（唯一“改动”是长期存在的 .codex/CLAUDE.md 噪声）|
| 决策引擎 | 我们的代价 solver（步骤 5，提交 657590cd7）|
| 全量读数 | 608 发现 / **528 通过 / 80 失败**（换引擎时是 500/108）|
| 已修硬失败 | 28（stride 2 + 无注解边界 17 + pre-lowering 统一拼写 9），我独立复核过 |
| 剩余硬失败 | 约 14：group_store 族 7（尝试未果已回退，需新根因）、vmi_compact_group_broadcast 1、extf 缺口、约 11 个 no-plan component 中含“关系集为空或结构冲突”的非 return 算子 |
| 期望差异 | 约 52，属步骤 8 的重测范围（步骤 5 后测量，不得把上游决定抄回测试）|
| 远端备份 | fork 的 feature/vmi-layout-solver-upstream（含上述全部提交）与 feature/vmi-layout-decision-layers（含 1700 行计划文档）|
| 待做 | 剩余硬失败（真实数字见 §19.5：23 个「真」+4 个负例诊断文本位移，不是 16 也不是 60）→ 步骤 6（pass 接线，脚本已备）→ 步骤 7（A–G 批次，逐族 patch 已备）→ 步骤 8（测试搬运与重测）→ 步骤 9（全门禁 + A5 + 性能复测）|

判据不变：硬失败归零、上游基线配置下失败集合不变、一致性 33/33、差分在修正后的量具（路径规范化 + 输入方言更新）下可比、端到端性能方向复现。

## 19. 硬失败分诊完成（16 个已修，假设被证伪）

### 19.1 数字（每个提交都断言失败集合是前一个的严格子集，0 回归）

| 修订 | 全量 | 相对上一状态 |
|---|---|---|
| 657590cd7（步骤 5）| 608 / 500 / 108 | — |
| 914368bec（stride_store 的 mask 操作数）| 608 / 502 / 106 | +2 |
| 453ed8e3b（无注解结果不再钉住）| 608 / 519 / 89 | +17 |
| 5cc2fe723（pre-lowering 统一拼写）| 608 / 528 / 80 | +9 |

族分布变化：硬失败 60 → **16**；FileCheck-only 46 → 62；上游自带 2 → 2。
改动范围：只动 VMILayoutPlanner.cpp 与 VMILayoutAssignment.cpp，**0 个测试文件被触碰**，无共享表行注入，无 fork 专属包装。

### 19.2 逐族归因（四条，全部修好）

1. **ABI 边界缺陷（5 个）**：\`group_reduce_s64/_slots8/_s256\`、\`group4_broadcast_shape_matrix\`、\`group_reduce_addi_i16\`——
   reduce/broadcast 的**唯一**关系产出 group-slots 结果（\`gs(8,1)\`、\`gs(8,8)\`、\`gs(4,8)ls2\`），而 \`func.return\` 被钉在 contiguous；
   而**没有 ensure 行覆盖 \`gs(8,1)\`**（\`gsFit()\` 要求 num_groups ≤ slots），故无解。**这证实是边界缺陷，不是候选行缺口。**
2. **方言元数缺陷（2 个）**：\`vmi_vsstb\`、\`vmi_to_vpto_stride_store\`——stride_store 的关系问了 \`operandPort(5)\`，
   而上游该算子只有 **5 个操作数**（fork 是 6 个，mask 前多 \`repeat_stride\`），越界被当作硬错误直接中止搜索。
3. **流水线顺序不匹配（9 个统一算子用例）**：上游管线在 \`-vmi-lower-unified-to-legacy\` **之前**就跑布局赋值，
   而关系提供者（来自 fork）没有统一拼写的分支，fork 的管线又总是先降级——归因类别是「完全是另一回事」。
4. **\`type of return operand\`（15 个）**：同一个 ABI 钉住问题——钉住迫使 return 处做转换，
   而 \`rewriteFunctionType\` 读到的是赋值**之前**捕获的值类型，于是操作数类型与函数类型不一致。**同一个提交修好。**

### 19.3 结论：我的假设被证伪（记在案）

**Stage 3 那批「未注入的 fork 独有行」不是主因**：已修用例里**没有一个是缺候选行**，全部是移植后的 planner 自身假设有误
（一个方言元数、一个上游从未有过的 ABI 钉住、一个流水线顺序不匹配）。因此我先前主张的「fork 专属包装」**没有必要**——
分诊任务没有注入任何共享表行，也没有做包装。候选行缺口**确实存在，但只涉及 6 个用例**，不是 50 个。

**我 §18.11 的建议被明确否证**：对 \`gs(8,slots=1)\` 这些形状，从 group-slots 到 contiguous **不存在 ensure_layout 行**，
而且**上游测试本身**就把返回类型 CHECK 成 \`gs(8,1)\`；上游的 \`rewriteFunctionType\` 采纳返回值布局——所以正确做法是「边界不钉住」。
这是我在此问题上第二次被实现方纠正（第一次是 §18.14），两次都已记录。

### 19.4 剩余硬失败的旧清单（**已被 §19.5 修正**，保留以记录这次分诊误差）

* **no-plan（5）**：\`opt/fused_quant_dequant_vmi_opt\` 167:17、\`vmi_compact_group_reduce\` 18:10、\`vmi_group_execution_paths\` 17:10、
  \`vmi_explicit_integer_cast_reduction_paths\` 17:7、\`vmi_to_vpto_block_mask_granularity\` 34:10——
  都是**多算子 component**，每个算子都有关系但 solver 找不到完整赋值；逐算子原因**未及查明**（诚实标注）。
* **no-relation（9）**：\`pto.vmi.group_store\` ×6、\`pto.vmi.group_broadcast\` ×1、\`pto.vmi.extf\` ×1（新暴露）、
  以及一个「负例测试」其诊断文本变了。**group_store 族已精确定位**：对 \`vreg<1xi8>/gs=1\`、\`vreg<4xf32>/gs=4\` 等形状，
  **preferred 行本身是合法的**（实测 \`getPreferredGroupStoreLayoutFact\` 成功返回 \`gs(numGroups,8)\` 等），
  只是该分支**仅在候选列表为空时才去问 preferred**；把 preferred 加进候选列表后实测 **80 → 80 且失败集合完全相同**
  （只是把 \`no relation\` 变成 \`no plan\`），因此按「不许放宽」的规矩**回退了**——
  并把它作为下一条线索：**行是可达的，真正的阻塞与 no-plan 族同源（solver 侧）**。
* **其他（2）**：\`vmi_to_vpto_gs1_consumer_matrix\`（VMI-RESIDUAL-OP）、\`vmi_to_vpto_vselr_invalid\`（负例，诊断文本变了）。
* 上游自带的 2 个失败不变。

分诊任务还记录了修复后的失败集合到 \`probes/planned_fail_set.txt\`，并指出 \`gate.sh\` 的基线集合断言在 planner 驱动下
自然成为当前集合的**子集**（符合预期）。

## 19.5 分诊修正：真实规模是 23 个「真」硬失败（+4 个负例诊断文本位移）

### 19.5.1 为什么 §19.4 的「16」是错的

§19.4 数的是**粗看像硬失败的用例名**；这里数的是**工具侧真的报错**的用例，判据不同，数出来的东西当然不同。
在 80 个失败里，只有 **27** 个带工具侧错误输出（planner 诊断）；其余 53 个是 FileCheck-only 差异（属步骤 8 范围）。
27 = **23 个真失败** + **4 个负例测试**——它们仍然失败，但只是**诊断文本位移**，工具行为本身是对的。
（先前的 16、24、26、48 这类数字都是我在不同判据下报出来的，别再引用；以本节为准。）

### 19.5.2 清单（按错误类别）

| 类别 | 数量 | 用例 |
|---|---|---|
| no-plan | 12 | \`opt/fused_quant_dequant_vmi_opt\`、\`opt/per_block_bf16_group8_quant_vmi_opt\`、\`vmi_compact_group_reduce\`、\`vmi_group_execution_paths\`、\`vmi_group_reduce_addi_i16\`、\`vmi_layout_assignment_group_reduce_partial_slots8\`、\`vmi_layout_assignment_reduce_minmaxf\`、\`vmi_to_vpto_reduce_extended\`、\`vmi_to_vpto_gather_granularity_conflict\`、\`vmi_to_vpto_block_mask_granularity\`（**已修，见 19.5.4**）、\`vmi_compact_group_broadcast\`（关系缺失）、\`vmi_layout_assignment_group_slot_broadcast_partial_packet\` |
| group_store 无关系 | 7 | \`short_vector_cross_width_bitcast\`、\`compact_load_store_group_alias\`、\`to_vpto_group_store_dense_alias\`、\`ptoas_cli_integer_vneg_layout\`、\`short_vector_narrow_store_l1\`、\`short_vector_narrow_store\`、\`short_vector_extend_store\` |
| group_broadcast 无关系 | 1 | \`compact_group_broadcast\` |
| extf 无关系 | 1 | \`vmi_layout_assignment_group_slot_load\` |
| VMI-RESIDUAL-OP | 2 | \`vmi_to_vpto_extf_f4x2_to_bf16x2_variants\`、\`vmi_to_vpto_gs1_consumer_matrix\` |
| 负例、诊断文本位移（**不得改测试**）| 4 | \`vmi_layout_gate_gs1_dense_join_invalid\`、\`vmi_layout_assignment_group_reduce_s12_invalid\`、\`vmi_layout_assignment_group_load_block8_truncf\`、\`vmi_to_vpto_vselr_invalid\` |

### 19.5.3 四条**互相独立**的根因（同一个量具的产物，不是猜的）

量具：env 门控的 planner dump + solver 失败点 instrumentation（提交前已全部剥离，提交里没有残留）。

* **(a) 关系漏掉/错索引了一个「带布局」的端口**，planner 的 \`hasValidInput\` 于是拒掉**整个 component**，不是拒掉一条关系。
  两个实例：\`stride_load\` 的 mask 在上游是第 4 个操作数（fork 是第 5 个），索引越界；
  \`pto.vmi.broadcast\` 取 vreg 作源时，关系只声明了 \`resultPort(0)\`——而上游「不带 group 的 vbrc」正好降级成这个形态。
* **(b) 传播把域抽空**：\`ensure_mask_granularity\` ×4、\`group_reduce_addi\` ×1、\`gather\` ×1、\`vmul\` ×1。
* **(c) 代价前沿被清空**：两个 \`opt/\` 用例（\`fused_quant_dequant\`、\`per_block_bf16_group8_quant\`）。
* **(d) group_store 的 preferred 行**：候选池里从来没有那唯一可用的行，而该分支**只在池子为空时**才去问 preferred。
  新的线索是后继失败：**生产端（一个普通 load）到不了那个布局**——所以它和前三条一样是 solver 侧缺口，不是「行不存在」。

### 19.5.4 已落地的提交（每个都断言失败集合是前一状态的严格子集）

| 修订 | 事实（为什么这是方言正确性，而不是放宽）| 全量 | 修好的用例 |
|---|---|---|---|
| \`71978e4fd\` | 上游 \`pto.vmi.stride_load\` 只有 4 个操作数（source, offset, block_stride, mask），fork 是 5 个；关系问 \`operandPort(4)\` 越界，唯一的关系被拒 → 整个 component 无解。改为按**本方言**的操作数表取 mask 下标，并加类型自检。与 \`914368bec\`（stride_store）同族。| 608 / 529 / 79 | \`vmi_to_vpto_block_mask_granularity\` |

### 19.5.5 纪律（重申并写进文档，免得下一轮又忘）

1. **一个提交一个事实**。提交信息必须写出：所依据的方言/solver 事实、改动前后的三元组、被修好的用例名、以及「失败集合是严格子集」。
   只报计数不算证据。
2. **门禁 ritual**：\`probes/gate.sh <label>\`（构建 + 产物比源码新 + lit，含污染自动复测一次）；\`.codex/CLAUDE.md\` 是已知的换行符噪声，
   两个门禁脚本都已用 \`grep -v CLAUDE.md\` 排除，不要「修」它，也不要让它挡住门禁。
3. **不许为了让测试过而放宽**：不往共享表里注入 fork 专属行、不放宽 verifier、不改测试。四个「诊断文本位移」的负例，
   只有在**新文本是同一个错误的更准确表述**时才算等价；不是的话那是 solver 的 bug，不是测试的 bug。
4. **类 (c) 最危险**。前沿为空若是因为代价模型**不会给合法行打分**，正确答案是**把这些合法行枚举出来并打分**
   （用户原话：「这些新 layout 我们的确是要枚举出来的，不能丢掉」），**不是**降阈值、也不是把某行标成免费；
   反过来，合法但代价不可导的行必须**按 best-effort 代价照旧提供**——这就是 \`8bc4c428e\` 定下的规矩（永不丢合法行）。
5. **类 (a) 必须对着本方言的 ODS 操作数顺序核对**每个碰到的算子，像 stride_load 这样，并在提交信息里写下这个顺序；
   不许沿用 fork 的顺序。
6. **不属于修复的代码不许活着进提交**（dump/instrumentation 一律 env 门控并在提交前剥离）。

分工：**步骤 6、步骤 7 不由分诊任务做**——避免两个写者同时在同一个 worktree 里；等硬失败归零后由主线依次落地。

## 19.6 第二次分诊回报，以及一处**必须做决定**的缺口（已定案）

### 19.6.1 已落地提交（累计 2 个，每个都断言失败集合是前一状态的严格子集）

| 修订 | 事实（方言正确性 / solver 缺口，不是放宽）| 全量 | 修好的用例 |
|---|---|---|---|
| \`71978e4fd\` | 见 §19.5.4（stride_load 的 mask 端口）| 608 / 529 / 79 | \`vmi_to_vpto_block_mask_granularity\` |
| \`ea1c1e6bb\` | \`pto.vmi.broadcast\` 只有「1 条 vreg」这一形态带布局，而 \`-vmi-lower-unified-to-legacy\` 会给每个不带 group 的 vbrc 产出这一形态；关系只声明了结果端口，\`hasValidInput\` 于是拒掉整个 component。三处**互相耦合、缺一不可**的改动：①关系提供者增列 source 的使用布局；②代价模型给这条双端口关系打分（原来硬要求 \`ports.size()==1\`，实测 \`dropPhys=12/12\`、前沿被抽空）；③\`group_store\` 在候选池为空时提供 preferred 行（实测 \`1xf32\`/num_groups=1 的池 = {contiguous, ls2, d2, d4} 皆不可用，preferred = \`gs(1,8)\`）。| 608 / 530 / 78 | \`vmi_compact_group_reduce\` |

### 19.6.2 待定问题：\`contiguous -> group-slots\` 这个转换在代价模型里算多少钱

分诊方给的**实测**（不是猜）：剩下 6 个 \`group_store\` 用例的唯一合法行是 preferred 的 group-slots 行；值来自普通 \`load\`，而 load 的候选行是 dense-load 表；把 load 直接改成 gs 会被 VPTO 下降拒绝（已实测并回退）；**手写**
\`load@contiguous + ensure_layout(contiguous->gs(1,8)) + group_store\` 在 \`-vmi-lower-unified-to-legacy -vmi-mask-granularity-assignment -vmi-to-vpto\` 下产出与测试期望**完全一致**的三条指令
（\`vlds dist=BRC_B32\` / \`pset_b32 PAT_VL1\` / \`vsts dist=1PT_B32\`），即这个转换**不额外消耗指令**。
但我们的代价模型在 \`materialize()\` 里对「任一端是 contiguous」的配对直接返回失败（fork \`VMILayoutCostModel.cpp:1958-1963\` 的守卫），合法行因此被丢掉。

### 19.6.3 定案：走「真实动作序列」，**不走**「带 contiguous 端就一律免费」

1. 把「带 contiguous 端的配对一律按被吸收处理」当**规则**，等于对**所有**这类配对做断言，而我们只有一个位置的实测。代价模型是全局的：layer 1（布局转换代价）会因此不再度量它该度量的东西——这正是用户禁止的「为了通过率接受假规则」的隐蔽形式。
2. 上游自己两个测试文件把这条路的形态写清楚了：
   * \`vmi_to_vpto_group_store_dense_alias.pto:66-74\`：别名化稠密形态必须降成 \`pto.vsts {dist = "PK4_B32"}\`、跨行形态降成 \`pto.vsstb\`——布局由 store 的寻址/dist 模式承载。这才是「被吸收」的真实来源：它是**动作的属性**，不是「配对免费」的许可。
   * \`vmi_to_vpto_short_vector_narrow_store.pto:12-16\`（上游自己的注释原文）："the dense lane stride is normalized **through the contiguous form** first … **Before that bridge existed the layout contract rejected the pair.**"——上游的答案是**经过 contiguous 的桥**，而我们的守卫正好把这桥切断。
3. 做法：**保留**守卫注释里写明的意图（避免经 contiguous 中间态无限重入），但把被它吞掉的**端点配对**，按紧邻其上已经显式处理的 \`contiguousToLaneStride\` / \`laneStrideToContiguous\` / \`deint2->ls2\` 一样显式给出动作；复用 fork 已有的动作种类，不引入新的代价种类。
4. **不许为这个配对特判返回 0**：若确实被邻近访存吸收，0 必须由既有的 absorbed 记账给出；合法配对永不返回 failure（\`8bc4c428e\` 的规矩）。

### 19.6.4 两条被排除的路（记下来，免得再走一遍）

1. **让 load 采用消费者请求的布局**——已实测被下降拒绝（「result layout does not match a supported dense load table row」）。dense-load 表是下降的权威，回退正确。
2. **\`VMIGroupStoreLayoutFact::stagingLayout\` 不是稠密操作数的通道**。上游 pre-port 的 \`getGroupStoreLayoutFact\`
   （\`1c1bba8cc:lib/PTO/Transforms/VMI/VMILayoutSupport.cpp:2084-2126\`）**先**要求 \`isSupportedGroupSlotMemoryLayout(layout, numGroups)\`，
   之后才为**短 group-slots 包**（slots==8、lane stride 2 或 4、4 或 8 元素、payload<256bit 且 32bit 对齐、row stride 1）计算 staging——
   它是在 group slots **内部**打包，并不接受 contiguous。

## 19.7 第三次回报：提交 3（group_store 族大面积关闭）与 mask-granularity 族的一处**事实更正**

### 19.7.1 提交 3（按 §19.6.3 的定案 (B) 落地）

| 修订 | 事实 | 全量 | 修好的用例（6） |
|---|---|---|---|
| \`daaa86b22\` | 实现方式与既有 \`contiguousToLaneStride\` / \`laneStrideToContiguous\` / \`d2<->ls2\` 同款：每个物理 part 一条 Pack（dense→packet）/ Unpack（packet→dense）动作；**守卫原样保留**（arity 不同仍然失败），**没有特判 0**，合法配对永不 failure。改动只有 \`VMILayoutCostModel.cpp\`（+35），0 个测试文件被触碰，0 行共享表注入。| 608 / 536 / 72 | \`vmi_compact_load_store_group_alias\`、\`vmi_layout_assignment_group_slot_broadcast_partial_packet\`、\`vmi_short_vector_cross_width_bitcast\`、\`vmi_to_vpto_short_vector_extend_store\`、\`vmi_to_vpto_short_vector_narrow_store_l1\`、\`vmi_to_vpto_short_vector_narrow_store\` |

§19.6.3 要求的证据 (i) 是**字面达成**的：测试自身的自动 LOWER 运行现在为 \`@compact_1\` 产出与手写实测完全一致的三条指令
（\`pto.vlds ... {dist = "BRC_B32"}\` / \`pto.pset_b32 "PAT_VL1"\` / \`pto.vsts ... {dist = "1PT_B32"}\`），测试文件未改。

### 19.7.2 事实更正：\`mb32 -> mb16\` 的合法行（我核对了表本体）

分诊方一度把「合法对」报成 \`c->d2, c->bd2, d2->c, bd2->c\`，据此认为要注入 fork 行。**这个前提是错的。** 共享表
\`lib/PTO/Transforms/VMI/VMILayoutSupportTables.inc:374-380\`（「2x narrowing」块）的原文是：

    {mb16(), mb8(),  d(2), c()},
    {mb16(), mb8(),  c(),  ls(2)},
    {mb16(), mb8(),  d(4), d(2)},
    {mb32(), mb16(), d(2), c()},
    {mb32(), mb16(), c(),  ls(2)},
    {mb32(), mb16(), d(4), d(2)},

即：\`c() -> ls(2)\` **是合法行**（正是关系挑中的那条，dump 里的 \`out0=contiguous lane_stride=2\`），而**能交出 plain contiguous 结果的是** \`d(2) -> c()\`。
误读来源大概率是紧邻其上的 **vreg** 转换表（同文件 ~318-324 行，用 \`bits<32>()\`/\`bits<16>()\` 而不是 \`mb32()\`/\`mb16()\`）——我核对时也差点踩进去，已记在案。
分诊方那次「上游也不合法」的实验写的是 \`c -> c\`，那不是本表任何一行，不能用来给 \`d(2) -> c\` 下结论。

**由此：需要的计划是** \`arg@d(2) -> ensure_mask_granularity -> contiguous(b16) -> group_reduce_addi\`，**不需要任何新表行**；
而且下降侧接受它：两处下降点（\`VMIToVPTO/VMIToVPTOPatternInternals8.cpp:552\` 与 \`VMIToVPTO/VMIToVPTOPatternInternals0.cpp:586\`）走的都是同一个
\`getMaskGranularityCastLayoutFactForLayouts\`，内部遍历同一张 \`kLegalMaskGranularityCastLayoutPatterns\`。

### 19.7.3 方向：关系提供者只问了表的一个方向

我们的 provider（\`VMILayoutPlanner.cpp:1434\`）只用 \`VMICastLayoutPort::Source\` 问表：对每个 source 候选收集 result 布局。
而上游自己的 transfer 是**双向**问的（\`lib/PTO/Transforms/VMI/VMILayoutPropagation.cpp:475-509\`）：\`changedValue == ensure.getSource()\` 按 Source 问，
\`changedValue == ensure.getResult()\` 按 Result 问。**被消费者钉住 result 的场景需要的正是第二种方向，而我们移植时把它丢了。**
把它补回来是**移植上游自己的查询**，不是放宽。

补回后若仍不成，下一个问题按顺序是：\`d(2)\` 这个 source 是否**可达**——mb32/N=128 有没有 \`ensure_mask_layout\` 从 contiguous 桥到 \`d(2)\`，
以及 \`d(2)\` 是否在调用方交给 provider 的 domain 里（\`planner:1419-1430\`：source 无显式布局时，候选 = {contiguous} ∪ 调用方的 \`polymorphicLayouts\`）。
domain 缺项是 provider/domain 缺陷（属移植），仍在代码里修，不注入表行。
**若实测 \`d(2)\` 对一个被钉在 contiguous 的入参确实不可达，则停下来报 dump** —— 那属于步骤 7 批次 D（F5 mask layout conversions）的范畴，由主线排期，不许就地强解。

### 19.7.4 一条警戒：消费者侧的 mask 行来自**上游的表**，不是我们的枚举

\`group_reduce\` 关系里带的 mask 布局来自 support fact 本身：\`planner:831-834\` 用 \`fact.sourceLayout/maskLayout/resultLayout\` 去
\`rememberGroupReduceRelationLayouts\`。所以「消费者只给了一个 mask 布局」是**上游表对该形状的答案**，不是我们枚举漏了。
动它之前必须先证明那行**是我们的**而不是上游的（\`1c1bba8cc\` 与我们各提交的 blame 对比）；比表上写的多枚举一条，就是我们不做的那种放宽。

## 19.8 分诊阶段收官（80 → 69）与两条**改变打法**的核对结论

### 19.8.1 四个提交（每个都在当时的干净树上实测、失败集合严格子集、0 回归、0 测试改动、0 行共享表注入）

| 修订 | 事实 | 全量 | 修好 |
|---|---|---|---|
| \`5cc2fe723\` | 起点（步骤 5 之后）| 608 / 528 / 80 | — |
| \`71978e4fd\` | stride_load 的 mask 端口（上游 4 操作数，fork 5）| 608 / 529 / 79 | \`vmi_to_vpto_block_mask_granularity\` |
| \`ea1c1e6bb\` | broadcast 的 source 布局 + 双端口关系打分 + preferred 行 | 608 / 530 / 78 | \`vmi_compact_group_reduce\` |
| \`daaa86b22\` | dense carrier ↔ group packet（Pack/Unpack 动作，守卫不动）| 608 / 536 / 72 | 6 个（见 §19.7.1）|
| \`bccd44a29\` | mask-granularity 表**双向问**（补回上游 transfer 的 Result 方向）| 608 / 539 / 69 | \`vmi_group_reduce_addi_i16\`、\`vmi_layout_assignment_reduce_minmaxf\`、\`vmi_to_vpto_gather_granularity_conflict\` |

我独立复核：HEAD \`bccd44a29\`、工作树干净（仅 \`.codex/CLAUDE.md\` 换行符噪声）、两个门禁产物新鲜、
planner / 代价模型 / assignment 三个文件里 \`VMI_LAYOUT_DIAG|vmiLayoutDiagEnabled|dumpDiagValue\` 均为 **0 命中**（instrumentation 已剥离）。

### 19.8.2 更正一：\`kLegalCastLayoutPatterns\` **有** gs 行（分诊方报成「一行都没有」）

\`VMILayoutSupportTables.inc:306-359\` 里 \`bits<16>() -> bits<32>()\` 的 gs 行共 5 条：

    {bits<16>(), bits<32>(), gs(1), gs(1)},
    {bits<16>(), bits<32>(), gs(8, 2), gs(8)},
    {bits<16>(), bits<32>(), gs(2), gs(2), CastTypeClass::Integer},
    {bits<16>(), bits<32>(), gs(4), gs(4), CastTypeClass::Integer},
    {bits<16>(), bits<32>(), gs(8), gs(8), CastTypeClass::Integer},

而 \`matchesCastTypeClass\`（\`VMILayoutSupportPatternDSL.inc:144-169\`）的语义是：\`Any\` 恒真；\`Integer\` 要求两端都是 \`IntegerType\`；\`Float\` 要求两端都是浮点类。
所以 \`vmi_layout_assignment_group_slot_load_extf\` 钉住的 \`vreg<8xbf16, gs(8,8)> -> vreg<8xf32, gs(8,8)>\`（浮点对）**没有可用的 Any 行**
（Any 的两条是 \`gs(1)->gs(1)\` 与 \`gs(8,2)->gs(8)\`），能匹配的那条 \`gs(8)->gs(8)\` 是 **Integer 专属**。准确表述是「行在、类把它排除了」，不是「一行都没有」。

### 19.8.3 更正二（这一条决定后面的打法）：**表不是我们能怪的对象**

* \`git show 1c1bba8cc:lib/PTO/Transforms/VMI/VMILayoutSupportTables.inc\` 与 HEAD 在那些行上**逐字节相同**；
* \`git log --oneline 1c1bba8cc..HEAD -- lib/PTO/Transforms/VMI/VMILayoutSupportTables.inc\` **为空**——移植以来**没有任何提交碰过这张表**。

再叠加已记录并反复重建断言过的基线（移植起点 608 发现 / 606 通过 / 2 失败），结论只剩一个：
**剩下这 10 个硬失败，每一个都是"上游自己的 assignment/propagation 能过、我们接手后过不了"的用例。**

由此定下后半程的判定规则（写死，免得再绕）：

1. **一个都不允许靠加/改表行来关**（本来就禁止，现在也证明不需要）；
2. 问题永远不是"这是不是表的缺口"，而是"**上游有哪条路被我们弄丢了**"；
3. 对 \`group_broadcast\` / \`vreg<1xi8>\` / \`num_groups=1\` 与 extf-on-group-slots 这两个"查不到行"的用例，答案**不可能是**"这个形状没有合法 lowering"——上游用**同一张表**解开了它们。这两处 IR 都把布局**显式写在类型上**，所以下一步是去读 \`1c1bba8cc\` 的
   \`VMILayoutAssignment.cpp\`（seed 入口）与 \`VMILayoutPropagation.cpp\`（\`getExplicitLayout\`/已注解类型的 seed），验证假设：**上游把"已被 IR 钉住的配对"当作既成事实，而我们的 solver 坚持要为它查到一条关系行**。若成立，修的是"移植后对显式注解端口的处理"，仍旧是搬上游行为，不是立新规。

### 19.8.4 剩余 10 个硬失败的分类与顺序

* **solver 侧（我们自己的机器）**：两个 \`opt/\` 用例——前沿在 \`extend()\` 里被 frontier-group/Pareto 剪枝清空，**所有 drop 计数器都是 0**，即不是关系失败。要求：先用具名 witness 表记录"哪一条把哪一条剪掉、按哪个 key"，并按顺序排除两件事——(i) key 退化（某分量未设 ⇒ 一切支配一切），(ii) 比较违反既定层次顺序（**偏好必须严格排在 ls 层之下**）。
* **上游有路由（优先做，属同一个调查）**：\`vmi_compact_group_broadcast\`（\`group_broadcast\` 无行）、\`vmi_layout_assignment_group_slot_load\`（extf 浮点 gs 对无 Any 行）、\`vmi_group_execution_paths\`（reduce 结果 \`gs(1,1)\` 对 broadcast 源 \`gs(1,8)\`，无 ensure 行）——都要先查上游 pre-port 的**显式钉住**路径，而不是断言缺 ensure 行。
* **其余**：\`vmi_layout_assignment_group_reduce_partial_slots8\`、\`vmi_explicit_integer_cast_reduction_paths\`（no complete plan）、两个 VMI-RESIDUAL-OP（\`vmi_to_vpto_extf_f4x2_to_bf16x2_variants\`、\`vmi_to_vpto_gs1_consumer_matrix\`）——后两个带"下降后仍有 VMI 残留"的味道，很可能落在**步骤 7** 的范围（F5/F8/F9/F10 的 lowering 模式），由主线在步骤 7 之后复测，不要求分诊阶段强解。
* **4 个负例诊断文本位移**：按纪律先比较新旧文本、引用双方原文，再判"新文本是不是同一个错误的更准确表述"。

### 19.8.5 交接状态

分诊方（\`7edccc40\`）在被重新开启后按 §19.8.3/§19.8.4 继续；**步骤 6、步骤 7 仍归主线**，在硬失败归零后的安静树上依次落地。
量具入口：\`.work/upstream-port/probes/measure.sh\`；当前失败集合：\`probes/planned_fail_set_post69.txt\`（旧的 \`planned_fail_set.txt\` 是 80 用例分诊记录，保留作历史）。

## 19.9 记录文件落地、一次"零收益即回退"的实验，以及 extf 那条堵点的**归属判决**

### 19.9.1 分类记录文件（分诊阶段的可复核产物）

\`probes/hard_fail_classification_post69.txt\`：69 行 TSV，每行 \`<用例名>\t<类别>\t<失败算子>\t<诊断首行原文>\`，
由 \`llvm-lit -v\` 跑完 69 个失败用例、逐个解析其自有 RUN 行得出（负例是否触发按行为区分，不靠命名猜）。
**合计：真硬失败 9 / 负例诊断文本位移 4 / FileCheck-only 56 = 69。**

采纳我更正后的数字：真硬失败是 **9** 不是 10（\`vmi_to_vpto_reduce_extended\` 现在是 FileCheck-only）。9 个用例：
\`opt/per_block_bf16_group8_quant_vmi_opt\`、\`opt/fused_quant_dequant_vmi_opt\`、\`vmi_layout_assignment_group_reduce_partial_slots8\`、
\`vmi_explicit_integer_cast_reduction_paths\`、\`vmi_group_execution_paths\`、\`vmi_compact_group_broadcast\`、
\`vmi_layout_assignment_group_slot_load\`、\`vmi_to_vpto_extf_f4x2_to_bf16x2_variants\`、\`vmi_to_vpto_gs1_consumer_matrix\`。

### 19.9.2 一次"零收益即回退"的实验（正面记录）

分诊方把"显式钉住端口"那条路的两半都移植了：①按 Result 侧枚举 cast fact（对齐上游自己的双向 transfer）；②两端都已注解且表里查不到时，把注解的元组本身作为关系。
结果：**关系出现了、planner 也成功了，但 applier 拒绝**（见 19.9.3）。全量实测 **608 / 539 / 69，失败集合与改动前完全相同**。
按"无实测收益即回退"的规矩**回退**，树回到 \`bccd44a29\`。这条纪律执行得对，记下来作为正例。

### 19.9.3 extf 堵点的归属：堵它的那道闸**是我们自己的**

我查了 \`validateCastOperationRelation\` 的来历：

* **pre-port（\`1c1bba8cc\`）：该符号在 \`lib/\` 与 \`include/\` 里根本不存在**；
* HEAD：定义在 \`lib/PTO/Transforms/VMI/VMILayoutSupportSolverQueries.inc:212\`、声明在 \`include/PTO/Transforms/VMILayoutSupport.h:774\`、**调用点在 \`lib/PTO/Transforms/VMI/VMILayoutPlanner.cpp:1329\`**；
* \`git log --oneline -S validateCastOperationRelation 1c1bba8cc..HEAD\` → \`65e8b9ab6\`（等宽 cast 关系）、\`70034b64f\`（"reject cast relations no lowering implements, for the solver"）、\`e46bb4110\`（planner 移植）——**三个都是我们的提交**。

也就是说：拒绝这个配对的不是上游的校验器，而是**fork 为了阻止 solver 挑中"表合法但没有 lowering 的配对"而新加的一道闸**。
而上游对**IR 已经写明**的配对，规矩是相反的：\`DataLayoutSeed{..., Explicit}\`（\`VMILayoutAssignment.cpp:320-323\`）+
\`addShuffleConstraint\` 在任一端已带布局时直接跳过约束生成（\`:1540-1543\`）。

**由此补齐那条路缺的第三块**：我们这道闸**不该跑在"两端布局都来自 IR 显式注解"的配对上**——它存在的意义是拒绝 **solver 自己挑的**配对，不是拒绝 **IR 已经写死的**配对。
已授权为**有界实验**，三个条件都是硬条件：①四个负例/门禁用例必须仍按各自原因失败，若其中有任何一个依赖"闸拒绝一个已注解配对"，立刻停下报我（那说明两条规则真冲突，冲突由我裁决）；②全量实测，若无用例被修好则回退；③若落地，提交信息必须写明恢复的是**assignment 阶段**的接受行为（该用例 RUN 止于 \`-vmi-layout-assignment\`），**不得**声称 lowering 支持该配对，并需单独报告 \`-vmi-to-vpto\` 对产物 IR 的行为（若在下降处失败，那属于步骤 7 的 lowering 工作，由我排期）。

### 19.9.4 group_broadcast：丢的是"把 preferred fact 变成请求"的那段管线（并修正我自己的一处记录）

* 我们的 planner **从不调用** preferred 路线，只枚举 group-broadcast 的表 fact（\`planner:1033/2409/2513\` 三处都是 \`getGroupBroadcastLayoutFactsForLayout\`）。
* 上游 pre-port 的路线仍在文件里，但**已不在决策路径上**：\`addGroupBroadcastConstraint\`（\`VMILayoutAssignment.cpp:1216\`，被 :1243 调用）
  用 \`getPreferredGroupBroadcastSourceLayout\`（$:659）+ \`getPreferredGroupBroadcastResultLayout\`（$:689 → \`VMILayoutSupport::getPreferredGroupBroadcastResultLayout\`，\`VMILayoutSupportGroupCapabilities.inc:70\`）；
  而现在的活路径是 \`selectLayoutPlan()\`（$:2094）。
* **修正我自己的一处记录**：§8（本文档 1313-1315 行）说的"删除 11 个 seed 辅助函数"指的是**请求/播种管线**
  （\`hasRequestedLayout\`…\`requestFallbackLayouts\`），这一点没错；但 \`addGroupBroadcastConstraint\`/\`getPreferredGroupBroadcast*\` 这些**是保留下来却离开了决策路径**，不是被删。
  所以真正丢掉的是"**把 preferred fact 翻译成请求**"的那段管线。修法因此明确：把这条 preferred 规则**重新表达为 solver 的候选**（与已被接受的
  group_store preferred 行 \`ea1c1e6bb\` 第三块同形），**不要去重新激活那个 walker**（未经我同意不许动）。

### 19.9.5 顺序

group_broadcast 的 preferred 路线 → extf 的有界实验 → 两个 \`opt/\` 用例的剪枝 witness 表 → 4 个负例文本判定。
纪律不变：一提交一事实、失败集合严格子集、门禁 ritual、无收益即回退、不注入表行、不改测试。

## 19.10 提交 5（annotated pair 的归属规则）与**步骤 6 落地**，附一次双写者事故与两次隔离实测

### 19.10.1 提交 5 = \`e936a9876\`（只动 planner，+70/-0）

把 §19.9.3 的判决实现为**权限划分**而不是"关闸"：

1. 两端都带注解、且共享表里**有**这一对的行（\`getCastLayoutFactForLayouts\` 成功）时，该注解元组**单独**成为该算子的关系；
2. 从**已注解的 result** 反查出来的行视为 IR 自身的推论，不走 solver 的闸（这一块才够到 extf 用例要的那一行）；
3. 其余配对**照原样**过 \`validateCastOperationRelation\`。无代价改动、无表行、无 verifier 改动、无测试改动。

证据（五条，均由分诊方实测）：(i) \`@..._extf\` 与 CHECK 逐行一致（\`ensure_layout -> gs(8,8, lane_stride=2)\` 然后 \`extf -> gs(8,8)\`）；
(ii) \`@..._extui\` 恢复 \`%1 = pto.vmi.extui %0\`、不再插入 ensure；(iii) 四个门禁/负例仍 4/4 各自失败；(iv) 608/539/69 → **608/540/68**；
(v) 失败集合严格子集：只删掉 \`vmi_layout_assignment_group_slot_load\`、无新增。

**我方独立隔离（重要）**：分诊方主动指出它的 68 是"混入我未提交的 step-6 文件"的产物，我没有接受这个混合态下的归属，而是
**把我那 5 个路径 stash 掉 → 重建 → 全量实测**：\`e936a9876\` 单独 = **608 / 540 / 68**，与 69 集合之差**恰好是删掉那一个用例、无新增**。
于是这个提交的功劳由"推断"变成"实测"，并且它也同时证明我的 step-6 文件在那个窗口里既没帮忙也没添乱。**它的这条协调提醒比那个提交本身更有价值。**

### 19.10.2 步骤 6 落地 = \`92d08341f\`（+189，5 个文件）

内容：把 fork 的 \`VPTOPack4StoreMaskNormalize\` **原样**搬入（含 OAT.3 许可头），并完成四处接线——
\`include/PTO/Transforms/Passes.td\` 的 pass 定义、\`include/PTO/Transforms/Passes.h\` 的声明、\`lib/PTO/Transforms/CMakeLists.txt\` 的目标源行、
以及 \`tools/ptoas/ptoas_pipeline.cpp\` 里紧跟 \`createVMIToVPTOPass\`、在 \`createVPTOStatefulStreamFusionPass\` 之前的那一次调用。

**双向隔离实测**（这一步的两个套件都要，因为 \`lit/vpto\` 在本 base 上没有基线）：

| 套件 | 含本改动 | 不含本改动（stash→重建）| 结论 |
|---|---|---|---|
| \`lit/vmi_new\` | 608 / 540 / 68 | 608 / 540 / 68 | 失败集合**完全相同** → 中性 |
| \`lit/vpto\` | 620 / 619 / 1 | 620 / 619 / 1 | 唯一失败 \`vpto/vmi_f4x2_to_bf16x2_vcvt_llvm.pto\` **在本 base 上是既有失败**（新记录基线）|

另有：\`ninja\` exit 0、0 error、两个产物都比它们用到的源码新（门禁的 freshness 断言）；
合规检查器（\`.agents/skills/enforce-ptoas-code-compliance/scripts/check_changed_code.py --repo . --base HEAD\`）对 6 个改动路径 **0 error / 0 warning**。
已推 fork：\`5cc2fe723..92d08341f\`。

### 19.10.3 双写者事故与教训（诚实记录，同类假信号第三例）

* 我在 **20:03:55** 写入 step-6 的 5 个路径且尚未提交；分诊方在 **20:05:10** 构建，于是它那次 68 的实测**混入了我的文件**。
* 我随后的第一次 step-6 全量实测则报 **608/608 全部失败、用时 1.04 秒**，而同一批用例单独跑（含 4 个随机抽取的用例）**全部通过** ——
  这是它的**并发重建在我的 lit 运行期间改写产物**造成的假信号，与已记录的两类（旧二进制、脏树）同类，属第三例。
* 处置：没有接受任何混合态结论；两次隔离（stash→重建→实测→恢复→重建）把归属钉死，结论写在 19.10.1 与 19.10.2 两张表里。
* **规则（写死）**：谁拿树谁宣布；另一方在其交回前**不许**构建或测量。

### 19.10.4 当前状态与下一步

* HEAD \`92d08341f\`（已推 fork），工作树干净（仅 \`.codex/CLAUDE.md\` 换行符噪声）。
* 累计 6 个提交，\`lit/vmi_new\`：**80 → 68**（608 / 540 / 68）；新基线 \`lit/vpto\`：620 / 619 / 1。
* 真硬失败 **8**（§19.9.1 的 9 个减去 \`vmi_layout_assignment_group_slot_load\`）；负例文本位移 4；FileCheck-only 56。
* 分诊方接续：刷新记录文件为 post68 → \`group_broadcast\` 的 1→8 形状（\`@compact_broadcast_i8_8_1\`）→ 两个 \`opt/\` 的 witness 表 → 4 个负例文本判定。
* **步骤 7 归主线**，等这一轮结束、树交回后开始（批次 A 是 F1/F6 纯删除，可先做；G 依赖 B/C/D/E）。

## 19.11 group_broadcast 链路的判决、步骤 7 批次 A 的验证结论与批次 B 的勘定

### 19.11.1 判决：这条链路交回步骤 7（不再在 solver 里逐族打补丁）

分诊方实测出的"每修一族、阻塞点就前移到下一族"链条：

| 补丁 | 实测位移 | 说明 |
|---|---|---|
| (a) preferred 路线（walk 自己的规则；注意 1→8 形状下它的 preferred **result 为空**，存活的关系来自 fact 表 \`in0=gs(1,8) -> out0=contiguous lane_stride=4\`）| \`16:10 -> 81:10\` | 修好 \`@compact_broadcast_i8_1_1\` |
| (b) lane-stride→packet 的显式两步（先 ls>1 归到 plain contiguous，再用 \`daaa86b22\` 已加的 dense/packet 一步），依据是共享 ensure 表**自己的注释** | \`81:10 -> 378:10\` | 修好 \`@compact_broadcast_i8_8_1\` |

两个补丁都上了之后全量仍是 **608/540/68、失败集合完全相同**，故按规矩**全部回退**；第三个阻塞点是 \`@compact_broadcast_i16_4_1\`（line 378），同样的 component 形状但是 i16。

**判决 (ii)：停止在 solver 里继续挖这条链，把它交给步骤 7。** 三条理由：
1. 该用例在 pre-port base 上是**通过**的，说明它的**每一族**上游都有路；一条链需要"每族一块新补丁"，只能说明我们找到的是某个**更一般路线**的特例，而不是那条路线本身。
2. 缺的路很可能在 **lowering** 侧：步骤 7 的 F4（deint 4↔2 + lane-stride bridge）与 F5（mask layout/granularity conversions）正是这一族需要的转换；我们的 solver 一直在**回避**那些 lowering 还降不了的布局，(b) 恰恰是"给不存在的 lowering 打补丁"的形态。
3. 分诊方自己的 i16 观察同向：若共享 ensure 表的 \`ls(4)\` 行真是 \`bits<8>\` 专属，那这个 workaround 本来就够不着 i16，诚实的修法在 lowering/表那一层。

两个补丁**保留但不上**：已让分诊方把它们写成 \`.work/upstream-port/staged/\` 下的两个可机械重放的 patch（含 README 记录所用文件、函数、实测位移与三元组）。**不许**把"部分修好"当进度提交（规矩：失败集合是单位）。

### 19.11.2 步骤 7 批次 A：验证结论 = **无需改动**

批次 A 是 F1/F6 两个 DROP 族，判据是"上游树本身就应当是上游的形态"：

* **F1 探针（本地 static 副本）**：\`getVMIMaskPhysicalGranularity\` 在单元里的定义在 \`VMIToVPTO/VMIToVPTOConversionInternals.cpp:638\`，
  使用点分布在 \`VMIToVPTOMaskInternals.cpp:564/616/918\`、\`VMIToVPTOPatternInternals8.cpp:26/73\`、\`VMIToVPTODataLayoutInternals.cpp:1807/1843\` 等；
  \`getVMIMaskPhysicalCarrierLayout\` 在 \`VMIToVPTODataLayoutInternals.cpp:1818\`。这些**是上游自己的代码**（我们的提交从未碰过 \`VMIToVPTO/\`），所以 F1 的 DROP 成立，**无需改动**。
* **F6 探针（factor-4 三处必须与上游一致）**：\`buildFactor4ContiguousParts:1862\`、\`lowerFactor4Block:1891\`，调用点 \`:1901\`、\`:1931/:1934/:1935\` —— 与上游一致，**无需改动**。
* 因此**批次 A 不产生提交**（这次是"验证后确认无事可做"，不是跳过）。

**门禁基线已落盘**（\`step7/\`）：\`G1.patterns.before\`（0 项）、\`G2.symcount.before\`（18 个单元）。
⚠ 顺带发现 **G1 探针已过期**：它按 \`populateVMIConversionPatterns\` + \`OneToN[A-Za-z0-9_]*\` 抽取，在当前拆分后的树上抽到 **0 项**——
在批次 G 之前必须重新推导 G1（否则它是个永远"通过"的空门禁）。已记。

### 19.11.3 批次 B（F2）勘定：前置条件齐备，锚点已定位

* 两个 support API 都在我们树里：\`include/PTO/Transforms/VMILayoutSupport.h:760 getSameWidthCastLayoutFact\` 与 \`:774 validateCastOperationRelation\`（来自 fork 侧提交 \`65e8b9ab6\`/\`70034b64f\`）。
* 四个 cast 形状检查在 \`lib/PTO/Transforms/VMIToVPTO/VMIToVPTOPatternInternals7.cpp\`：\`checkSupportedFPToSIShape:950\`、\`checkSupportedFPToUIShape:960\`、\`checkSupportedSIToFPShape:970\`、\`checkSupportedCompressShape:1410\`；
  注册/校验调用点在 \`VMIToVPTOPatternInternals8.cpp:811/817/824/1188\`。
* 上游树是**完全拆分**的（没有 fork 的 \`VMIToVPTO.cpp\` 单体文件），所以 F2 规格里的行号必须**映射**、不能按行号套用——这也是 MANIFEST 第 1 节写明"这是规格不是 patch"的原因。

## 19.12 暂存补丁、四个负例的判定结论，以及一处**修正我自己判决**的新证据

### 19.12.1 暂存补丁（".work/upstream-port/staged/"）

\`group_broadcast_preferred_route.patch\`（planner，\`VMIGroupBroadcastOp\` 分支，+33/-0）与
\`lane_stride_packet_normalization.patch\`（代价模型，\`PlanGraphBuilder::materialize\`，+25/-0），加 \`README.md\`：
每块记录所改文件/函数、作用、实测位移（$16:10→81:10、81:10→378:10）、实测所用套件与三元组，以及"暂存未落地"的原因与重放步骤。
两块都是**由 \`92d08341f\` 自己的 blob 生成的真 diff**（生成脚本只读 \`git show\`、只写 \`/tmp\`），并已用 \`patch -p1 --dry-run\` 对 pristine 副本验证通过；
**不含**任何诊断打印、表行或测试改动。

### 19.12.2 四个"诊断文本位移"的判定（两真两假，其中两条是 solver 缺陷）

| 用例 | 判定 |
|---|---|
| \`vmi_layout_assignment_group_reduce_s12_invalid\` | **solver 缺陷（诊断管线）**：同一条错误，但我们的文本**丢掉了 support 模型给出的原因**（期望文本是"…has no registered group_slots layout support: group_reduce layout table has no row for this group size"），而 \`reportMissingRelation(op, reason)\` 本来就是为携带这个原因而存在的 —— group_reduce 分支把 query 的 reason 丢了。 |
| \`vmi_layout_gate_gs1_dense_join_invalid\` | **solver 侧诊断回退**：同一条错误被提前一个阶段报出，且**不点名算子与配对**（我们的是"整个 component 无完整解"的全或无消息），期望文本点名了 \`pto.vmi.vmul\` 的操作数 #1 与两个布局。无需行为改动。 |
| \`vmi_to_vpto_vselr_invalid\` | **等价内容、不同报出点**：两句都是**上游自己的原文**（pre-port → HEAD 未变），我们报的是其中一句，来自 planner 的关系查询而非下降/verifier 点 → 按规矩属**步骤 8 的文本搬运**，信息未丢。 |
| \`vmi_layout_assignment_group_load_block8_truncf\` | **不是同一个错误**：实际失败的是 \`group_load\`（不是 \`truncf\`）→ **疑似 solver 缺陷**：我们的枚举把一个 value 布局交给了 \`group_load\`，而下游上游自带的 \`validateGroupLoadLayoutPlan\` 随后拒绝它。需带树确认。 |

### 19.12.3 修正判决的新证据：i16 那一族很可能是 **solver 侧**，不是 lowering 侧

* **C(1)（表本身，已逐字核对）**：能"把 lane-strided carrier 归到 group packet"的 \`kEnsureLayoutPatterns\` 行是
  \`{bits<8, 16>(), anyN(), ls(2), gsFit()}\` / \`{... gsFit(), ls(2)}\` 与 \`{bits<8>(), anyN(), ls(4), gsFit()}\` / \`{... gsFit(), ls(4)}\`，
  外加同 bit 模式上的 \`gsFitStride(2)/gsFitStride(4)\` 变体。
  **结论：\`ls(2)\`/packet 族是 \`bits<8,16>\`，而 \`ls(4)\`/packet 族确实只有 \`bits<8>\`** —— 对 \`@compact_broadcast_i16_4_1\`（i16、ls4 carrier）**根本没有这一行**，这解释了为什么暂存补丁 (b) 在那里够不着。
* **C(2)（路由，已读 transfer）**：\`VMIGroupBroadcastTransfer::query\`（\`VMILayoutPropagation.cpp\`，\`1c1bba8cc\` 与 HEAD **逐字节相同**）按**哪一侧发生变化**选择查询端口：
  \`changedValue == broadcast.getSource()\` → \`Source\`，否则 \`broadcast.getResult()\` → \`Result\`，然后调 \`getGroupBroadcastLayoutFactsForLayout(...)\`。
  而**我们的 provider 只问了 \`Source\`**（planner 的 \`VMIGroupBroadcastOp\` 分支）—— **与 \`bccd44a29\` 修掉的 mask-granularity 缺陷同类**。
* **因此修正 §19.11.1 的判决**：这条链剩下的那一块**最可能是 solver 侧的"双向查询"**，而不是缺表行；缺 \`ls(4)\`/16-bit 行是**兜底解释**。
  已按此重排指令（先关 \`vmi_compact_group_broadcast\`：暂存 (a) + Result 方向查询，必要时再叠 (b)，最小组合成一个提交；若仍不关，则退回"表行缺失"并交由步骤 7 批次 D/E）。

### 19.12.4 修正后的执行顺序

1. 关 \`vmi_compact_group_broadcast\`（一条最小组合提交）→ 2. 两条**诊断**提交（\`group_reduce_s12_invalid\`、\`gs1_dense_join_invalid\`，不改验收、只改报什么）
→ 3. 查 \`block8_truncf\`（枚举给了 \`group_load\` 一个下游拒绝的布局）→ 4. 两个 \`opt/\` 的剪枝 witness 表。
\`vmi_to_vpto_vselr_invalid\` 不动（属步骤 8）。步骤 7 批次 B–G 仍归我，等这一轮结束、树交回后继续。

## 19.13 优先级 1 的实测结论：链路**挂起为步骤 7 的依赖**（并记下它的回归前提）

三块补丁（暂存 (a) + Result 方向查询 + 暂存 (b)）逐级实测：

| 步骤 | 实测结果 |
|---|---|
| 暂存 (a)（preferred 路线）| 干净应用；阻塞点离开 \`16:10\` |
| + Result 方向查询（对齐 \`VMIGroupBroadcastTransfer::query\`）| 文件仍失败（1→8 族）|
| + 暂存 (b)（lane-stride 归一）| **该文件所有硬失败消失**，管线输出干净 VPTO |

但仍不通过，且原因变成**另一种**：测试期望 \`vselr\` 路径，而我们的计划降成
\`pto.vlds {dist = "BRC_B8"} ; pset_b8 PAT_ALL ; pset_b8 PAT_VL1 ; pto.vdup %result, %0 {position = "LOWEST"} ; pto.vsts {dist = "1PT_B8"}\`，
首个不匹配在测试第 22 行（\`@compact_broadcast_i8_1_1\`），其后 33、55 —— 这是**下降形态的期望差异**，不是布局赋值失败。

**决定性数据（全量）**：三块都上 = **608 / 539 / 69**，比 68 基线**多一个**，且集合差是**新增**：

    > vmi_group_sparse_compare.pto

即 **Result 方向查询按现状对 \`group_broadcast\` 不安全**（与安全落地的 \`bccd44a29\` 的 mask-granularity 不同）：它也会为别的形状承认结果侧关系，把 \`vmi_group_sparse_compare\` 的计划带偏。已全部回退并复核：\`vmi_group_sparse_compare\` 恢复通过，\`vmi_compact_group_broadcast\` 恢复原硬诊断，树干净在 \`92d08341f\`。

**判决（修正 §19.12.3 的"先关它"）**：该链路**挂起为步骤 7 的依赖**，不在 solver 里继续追。依据：
1. 三块齐上后**该文件的硬失败全部消失**——这是"路线判断对了"的强证据，也是该文件第一次有合法计划；
2. 剩下的差距是**下降形态**（\`vdup\` vs \`vselr\`），属 VMIToVPTO delta（步骤 7，尤其 group-broadcast 下降族），不属布局赋值；
3. 即便形态问题解决，Result 方向查询也必须先**收窄作用域**（现在它会动 \`vmi_group_sparse_compare\` 的计划），否则不可落地。

**处置**：Result 方向片段作为**第三块暂存补丁**（含 README 行：实测效果 + 那句回归 + 前提条件"步骤 7 先改形态、且 Result 方向先收窄"）。
正确顺序：**步骤 7 落地下降形态 → 再一起重放三块并重测**（\`vmi_group_sparse_compare\` 必须保持通过）。

其余两条要保留的事实：16-bit 的 \`ls(4)\`/packet ensure 行**确实不存在**（C(1) 逐字核对，现为兜底解释）；目标文件的剩余差距在步骤 7/8 而非布局赋值。

### 19.13.1 紧接着的顺序

分诊方按优先级 2–4 继续：两条**诊断**提交（\`group_reduce\` 的 reason 透传、component 消息点名算子与两个布局；预期各自**删掉**一个失败项）→ \`group_load\`/\`validateGroupLoadLayoutPlan\` 调查 → 两个 \`opt/\` 的 witness 表。
随后主线接树做步骤 7 批次 B–G。

## 19.14 优先级 2(i)：主线自己落地（提交 7 = \`b58fa790f\`）

分诊方的 2(i) 已经把事实查清（不需要再挖），所以这一条由主线直接实现：

* **事实**：期望文本两半都是**上游原文**——前半由 \`PTOValidateVMIIR\` 报出，后半由 \`buildGroupReduceLayoutKey\` 提供，
  且 \`git log -S "has no registered group_slots layout support" 1c1bba8cc..HEAD\` 为空（无移植提交碰过这句）。
  我们的 provider 在**同一形状上更早**失败，而 group-reduce 分支**丢掉了 family query 已经产出的 reason**，只报通用句。
* **改动**：分支保留该 reason；\`reportMissingRelation\` 对 legacy group-reduce 家族按**形态检查的措辞**报出。
  **仅诊断**：同阶段、同失败、无验收/代价/表行/测试改动。
* **实测**：\`608 / 540 / 68\` → **\`608 / 541 / 67\`**；失败集合**严格子集**（只删 \`vmi_layout_assignment_group_reduce_s12_invalid\`，无新增）；
  \`ninja\` exit 0 / 0 error；合规检查器 **0 error / 0 warning**。
* **实际报出的原文**（与测试 CHECK 逐字一致）：
  \`loc(...):19:12: error: VMI-LAYOUT-CONTRACT: pto.vmi.group_reduce_addf has no registered group_slots layout support: group_reduce layout table has no row for this group size\`
* **一条代码形状的说明（诚实记录）**：条件被提升为具名 bool。合规预过滤器的 \`G.FMT.11-CPP\` 正则会把 \`if (call() && ...) {\` 这种内联条件误判为"无花括号体"——这是该正则的已知假阳性；
  提升后既可读又让检查器安静，且行为完全不变。

**新基线**：HEAD \`b58fa790f\`，\`lit/vmi_new\` = **608/541/67**（真硬失败 8 + 负例文本位移 3 + FileCheck-only 56）。
下一条（2(ii)）已按"一次只做一件事"派给分诊方：若在该失败点无法在不引入新状态的前提下恢复"算子 + 操作数序号 + 两个布局"，就直接停下来报我。

## 19.15 2(ii) 的否定答案：这**不是**文本位移，而是一处**行为分歧**；并把它升格为具名假设

分诊方按"是停就停"的要求给出了否定答案，理由不是"信息拿不到"，而是**阶段与措辞所主张的东西**：

* 测试文件**自己的注释**（RUN 行上方，逐字）说明了该失败应当发生在哪一阶段：
  "…so the two are not a carrier identity and **no ensure_layout row joins them**. **Layout assignment still inserts the bridge, and the VPTO conversion is where it finds no materialization.**"
* 实测我们的移植正好相反：只跑到 \`-vmi-layout-assignment\`（不带 \`-vmi-to-vpto\`）就已经失败：
  \`vmi_layout_gate_gs1_dense_join_invalid.pto:24:13: error: VMI-LAYOUT-CONTRACT: no complete legal VMI layout plan exists for this component\`
* 而这个拒绝**按表是正确的**：该配对的所有 ensure 行都是 \`gsFit()\`/\`gsFitStride(S)\` 形态，而 \`gsFit()\` 要求 \`num_groups <= slots\`，
  所以 \`contiguous <-> num_groups = 4, slots = 1\` **根本没有行**（\`VMILayoutSupportTables.inc:32-33, 76-79\`）——测试注释里那句"no ensure_layout row joins them"说的就是这件事。

**因此**：要让管线走到测试描述的状态，planner 必须**接受一个没有 ensure 行的转换**——那是**验收行为改变**，不是诊断改变；
而从 planner 报出那句话则是**编造消息**（把 lowering 的失败安在一个还没走到的阶段上，还点名一个计划里根本不含的转换）。它停下来是对的。

**归类修正**：该用例**不是**"负例文本位移"，而是**赋值阶段的一处真实行为分歧**——上游的 assignment 接受 \`gs(4,1) <-> contiguous\` 这一对、把失败推迟给转换；我们直接拒绝了这一对。
于是负例文本位移从 3 减到 2，另立一类"行为分歧（未解决）1"。

**我的决定**：**暂不 re-baseline**（选项 (b) 作为兜底保留），把选项 (a) 升格为**具名假设**：
> **H-a**：我们的 planner 是"全或无"的——找不到**可物化**的计划就拒绝整个 component；上游的 assignment 接受"关系自洽但无 ensure 行"的布局，让转换阶段去报失败。

**为什么值得单独立项**：剩下 8 个真硬失败里有 **5 个**是"no complete legal VMI layout plan"（\`opt/fused_quant_dequant_vmi_opt\`、\`opt/per_block_bf16_group8_quant_vmi_opt\`、\`vmi_layout_assignment_group_reduce_partial_slots8\`、\`vmi_explicit_integer_cast_reduction_paths\`、\`vmi_group_execution_paths\`）。
如果拒绝它们的就是这条"全或无"规则，那么**一个兜底可以同时处理这 5 个加上本用例**。**若现在 re-baseline，就等于把一个也许能消掉的分歧固化成基线。**

**下一步（已派、只读）**：定位拒绝点（\`applyLayouts\`/\`selectLayoutPlan\` 失败路径）与该处可见状态；说明兜底若要复现上游行为需要做什么（把关系自洽的布局交出去，物化交给 \`VMILayoutSinkMaterialization\`/\`VMIExpandImplicitEnsureLayouts\`/\`VMILayoutRematerialize\`）；
把步骤 5 删掉的 **11 个 seed/request 辅助函数**（上游这套行为的原型）逐条说明到"可作为部分复活的基础"的程度、或明确说不可以；
逐个判定那 5 个用例的拒绝是否**同类**（无可物化计划）还是**异类**（关系本身矛盾）；并给出**风险清单**（哪些现在通过的用例最可能因此换计划），让"严格子集"断言有意义而不是靠运气。
之后由主线按纪律做该实验（两个方向都测、无收益即回退）；若判定为"新发明且爆炸半径大"，则取 (b) 并把 H-a 记为否决。

## 19.16 H-a 的勘探结果：**收益被大幅下调**，判决为"暂缓"（带触发条件）

分诊方按我给的五个问题做了只读勘探，结果改变了优先级判断：

### 19.16.1 拒绝点与兜底的形状（回答 1–3）

* **全或无的根**：\`VMILayoutPlanner.cpp:2942-2947\`（\`selectCostedVMILayoutPlans\`）——某个 component 求解失败时**整个 module** 返回 failure，
  并丢弃已解出的 \`result.plans\`；每 component 的消息在 \`:1165\`（\`solveComponent\`）发出。
* **挂接点**：\`VMILayoutAssignment.cpp\` 的 \`selectLayoutPlan():2080-2094\` → \`selectCostedVMILayoutPlans\` → 每个计划的 \`mergePlan\`，
  由 \`applyLayouts():2092\` 调用；后者在失败时**尚未** \`commitVMILayoutPlan\`，所以什么都没装上、pass 直接中止。
  该处**可见状态**：IR 全文；**新建的 propagator**（API 完整：\`request/addUseConflict/run/apply/verifyMaterializationPlan/canMaterializeLayout/materializePrimary\`）；
  以及**仍在被填充**的 LayoutSolver walk 状态（\`dataNodes\`、\`dataLayoutSeeds\`、\`dataUseRequests\`、\`maskUseRequests\`）。
  可见性**之外**：失败 component 的算子列表（局部于该函数）与任何部分计划。
* **兜底要做什么**：①先去掉全或无（失败记为 deferred 并继续，返回"计划 + deferred components"，只是一个小结果结构改动，无新算法）；
  ②对每个 deferred component：把"关系自洽的请求集"（显式 seed + 关系的端口布局）装进 propagator，\`run()\`，再 commit，
  并且**按 pre-port 老路**用 \`propagator.apply(rewriter)\` 物化 use conflict，或交给 \`VMILayoutSinkMaterialization\`/\`VMIExpandImplicitEnsureLayouts\`/\`VMILayoutRematerialize\`。
  这些 pass 需要的是**完整赋值过的 IR**（不变式：\`vreg\` 必须带布局），而且冲突使用的 use 布局必须记录或插入 ensure_layout，
  所以"只赋 value 布局"的兜底会违反不变式并在后面炸掉；\`insertDataUseMaterializations(:2003)\` 虽然还在，但 §8.4 记为**无调用者的死代码**。
* **删掉的管线可用作基础**：四个 installer 循环（\`requestExplicitLayouts\`、\`runLayoutSeedPhases\`、\`requestLateLayouts\`、\`requestFallbackLayouts\`）都是薄循环；
  真正难的部分（propagator 本身、phase 枚举与 seed 上的 phase 标记、产出 seed 的约束 walk）**都完好**；
  而且与 §8.5 的记录相反，**那 11 个 seed 辅助函数并没有被删除，只是离开了决策路径**。所以这是"部分复活"，不是新发明。

### 19.16.2 五个 "no complete legal plan" 的分类：**收益从 5+1 降到最多 1+1**（回答 4）

| 用例 | 判定 |
|---|---|
| \`vmi_group_execution_paths\` | **同类**（H-a）：\`group_reduce_addi\` 的唯一关系产出 \`gs(1,1)\`，消费者 \`group_broadcast\` 的关系要求源 \`gs(1,8)\`，而无 \`gs<->gs\` ensure 行 → 计划关系自洽、只缺转换。与 gs1 用例同型。 |
| \`opt/fused_quant_dequant_vmi_opt\` | **异类**：记录是"frontier empty after \`create_mask\`，**所有 drop 计数器为 0**"——物化失败会留下 \`dropPhys\`/\`dropMat\`，所以这是**我们自己的前沿/Pareto 剪枝**问题，兜底大概够不到。 |
| \`opt/per_block_bf16_group8_quant_vmi_opt\` | **异类**，同一签名。 |
| \`vmi_layout_assignment_group_reduce_partial_slots8\` | **无法判定（记录已过期）**：记录是"propagate emptied domain of \`ensure_mask_granularity\`"——那正是 \`bccd44a29\` 修掉的族，所以它现在的拒绝原因与记录不同。**需要带树重测。** |
| \`vmi_explicit_integer_cast_reduction_paths\` | **无法判定（记录不足）**：只记到"ablation[no-fixed]=unsolved"，没有 solve-fail 行。**需要带树重测。** |

### 19.16.3 风险清单（回答 5）与**判决**

* **结构性边界**：兜底只在"求解器拒绝"处触发，而今天被拒绝的 component 不可能是通过用例（拒绝会中止 pass）——所以**只要严格限定在 deferred component 上，就不可能有现在通过的用例走到兜底**；
  若改成"对所有东西重跑 propagator"，这个边界就消失、"严格子集"也就变成运气。
* **预期会动**：8 个真硬失败（拒绝 → 有计划 → 或通过、或在更后阶段失败）与 4 个负例文本用例；其中 \`vmi_layout_gate_gs1_dense_join_invalid\` 应报出它期望的转换阶段文本——**这才是该实验真正的假设检验**。
* **二阶风险**（粗心实现会泄漏到通过用例）：①\`mergePlan\` 遇到"deferred component 与已计划 component 共享同一个 value"（同一个 value 两个布局必须仍是真冲突）；
  ②\`planWasCommitted\`/\`verifySelectedRelations\` 假定每个 solver 算子都有被选中的关系，deferred 的没有，必须显式跳过；③\`rewriteFunctionType\` 会看到兜底赋的 ABI 类型——正是 \`453ed8e3b\` 处理过的地方；④deferred component 的 ensure_layout 数量变化（只能影响今天失败的用例，但要**验证而不是假定**）；⑤两个 \`opt/\` 用例如果也翻了，那是**反对**上述分类的证据，值得知道。

**判决：暂缓（deferred），不是否决。** 理由：收益已从"5+1"缩到"最多 1+1"，而代价是一个接受策略层面的改动外加上面五条二阶风险——其中"共享 value 的 mergePlan"正是会产出**静默错误计划**的那一类。
同时，**必做的步骤 7 里有两个真硬失败（VMI-RESIDUAL-OP"failed to apply conversion patterns"）属于 lowering 工作**，而 \`vmi_compact_group_broadcast\` 已挂起为步骤 7 的依赖——所以先做步骤 7 更划算。
**触发条件（写死）**：步骤 7 落地并重测后，带树重判那两个"无法判定"的用例；若它们与 \`vmi_group_execution_paths\` 同类，收益回升，则按 §19.16.3 的边界与二阶风险清单**立项做 H-a**。

### 19.16.4 它下一步的只读工作

我接树做步骤 7 批次 B；同时它做**只读的锚点映射**（把 F2/F3/F7 的每个 hunk 映射到拆分后单元里的当前行号、待替换原文与替换文本），因为 F 文件的行号是 fork-vs-merge-base 的**单体坐标**，而当前树是**完全拆分**的——这正是批次 B 落地前缺的那一步。

## 19.17 锚点映射结果：步骤 7 的落地方式必须改（两条**安全发现**）

### 19.17.1 总况：F 文件的上下文基本不可用

F 文件的导引上下文是**单体坐标**，在拆分子树里**大多不存在**。两种系统性差异反复出现：

* fork 单体用 \`if (srcBits == dstBits)\` 与 \`return fail(...)\`，而拆分子树用**显式位宽对分支**（\`srcBits == kValue32 && dstBits == kValue32\` 等）与 \`emitLogicalFailure(reason, ...)\`
  —— 所以每个 hunk 都要**按分支重新表达**，不能按上下文套用；
* fork 删掉了上游**保留且仍在使用**的辅助函数，因此**至少有一个 hunk 若照字面重放会直接编译失败**（见 19.17.2）。

11 个 hunk 的分类：**3 个纯插入可照字面落地**（F2 h3 若核实、F3 h2 去掉 stride 实参、F7 h1 改名后）、**6 个需按拆分子树惯用法重表达**（F2 h1/h2/h4/h5、F3 h1/h4）、**1 个绝对不能落地**（F3 h3）、**1 个需在三处候选里选定落点**（F7 h2）。
落地方式因此改为：**以"函数名 + 语义改动"驱动**，把 fork 的上下文当作**改动说明**而不是 patch 文本。

### 19.17.2 两条安全发现（若不记录，下一次就可能踩进去）

1. **F3 hunk 3 = DO NOT APPLY。** 它删除 \`isCompactSmallGroupStore\`，而拆分子树里该函数**定义在** \`VMIToVPTOMemoryInternals.cpp:964\`、**被上游自己的代码使用在** \`:1142\`。
   fork 能删是因为它把逻辑搬进了 layout fact，上游从未搬。**照字面重放会破坏构建，并且会放宽验收**——F-spec 自己也警告过这一 hunk。
2. **F3 hunk 2 需要改形**：该 hunk 给 \`getGroupSlotLoadLayoutFact\` 传了新的 stride 实参，而上游的查询**没有 stride 形参**（§8.1 记录的正是这个签名差异）。
   要么落上游的三实参形式，要么把 stride-aware 查询作为 support 层 shim 一起落——**不能照 hunk 落**。

### 19.17.3 一处**语义决定**（由主线下判，不许静默）

F2 的依赖是"step-3 的 \`validateCastOperationRelation\` 查询存在"——它确实存在（\`VMILayoutSupportSolverQueries.inc:212\`，我们 \`70034b64f\` 的代码）。
但该查询**当前是 solver 的闸**；把它用到 lowering 里是**语义决定，不是移植动作**。

**判决：可以复用，但定位为"共享的 support 谓词"，而不是"把 solver 的闸借给 lowering"。** 它问的内容就是"support 模型是否认识这一对"，而这正是 lowering 的形状检查要问的；复用让两个阶段**由构造保持一致**（solver 的闸镜像 lowering 的拒绝，这正是两边拒绝能对齐的原因）。
提交信息必须写明这一点，并由**实测**决定：若 lowering 的行为变化超出预期的检查之外，就回退。

### 19.17.4 下一步

已派它做**只读的逐 hunk 精确文本**（把 19.17.1 里标记为"未核实"的条目全部落实：F2 的 1/2/3/4/5、F3 的 1/2/4、F7 的 1/2，逐条给出拆分单元里的**当前原文**与**替换文本**，F7 还需在三处 \`compactLayout\` 构造点中指明落点）。
拿到后我按"函数名 + 语义改动"机械落地批次 B（F2）→ 与 F3/F7 同批（二者必须一起落）。

## 19.18 逐 hunk 精确文本到手：**F2 的三个 hunk 其实是一处改动**，以及我的落地判决

### 19.18.1 最重要的一条发现

拆分树里 \`checkSupportedFPToSIShape(:950)\` 与 \`checkSupportedFPToUIShape(:960)\` 都是**薄包装**，二者都走共享模板 \`checkSupportedFPToIntShape(:881-921)\`；
\`checkSupportedSIToFPShape\` 走同一个 \`checkSameWidthConversionArity\`。所以上下文为 \`if (srcBits == dstBits) { … }\` 的那个 hunk **只有一个落点**：
模板里的 \`:902-907\`。**一处改动覆盖 F2 h1/h2/h4 三个 hunk**；两条路线（改模板 / 逐个改调用点）是**互斥的，绝不能都做**（否则重复校验）。

### 19.18.2 可移植性总表（逐条已核实到原文）

| 条目 | 判定 |
|---|---|
| F2 h3（\`+4\` 插入）| **可照字面落地**（\`:910-918\`，只把命名换成 \`layouts->…\`）|
| F2 h5（compress 的 contiguous 检查）| **半精确**：\`:1422-1426\` 换成 \`getReduceLayoutFactForLayouts\`；fork 的 \`checkFullDataPhysicalChunks\` 尾巴在拆分子树里**没有对应**（拆分用 \`buildCompressPhysicalShapePlan\`）——那部分**不要动** |
| F3 h1（interleave store）| \`:447-462\` 精确可换（布尔守卫 → \`getInterleaveStoreSupport\`，保留 \`buildWriteAccessPlan\`）；**但它会去掉 \`getX2MemoryDistToken(…, "INTLV")\`，验收问题**（见 19.18.3）|
| F3 h2（group_slot_load）| **按原文不可移植**：stride 实参在上游查询里不存在（需 §8.1 的签名 shim）→ **记为具名缺口，暂不落地** |
| F3 h3 | **绝不落地**（\`isCompactSmallGroupStore\` 定义 \`:964\`、使用 \`:1142\`）|
| F3 h4 | F-spec 的锚点是**包装函数** \`:1161\`，真正落点是**助手**里的 \`:1142-1143\` |
| F7 h1 | 落点 \`:1076\`（变量名是 \`compact\` 而非 \`compactSmallGroupStore\`，且在 const 方法里）|
| F7 h2 | 落点是 **\`:710\`**，且与 \`:1076\` **不是同一个函数** —— \`groupStoreFact\` **不在作用域内**，必须在该处重新查询；\`:626\` 是多分片 stream 路径，不是这一处 |

### 19.18.3 两条**前置校验**（决定我怎么落，已派只读核对）

1. **F3 h1 的验收问题**：\`getInterleaveStoreSupport\` 是否**覆盖**被删掉的 \`getX2MemoryDistToken(…, "INTLV")\`？
   等价 → 删除不是放宽，照写落地；**更宽 → 保留该检查、只换守卫**；更严 → 说明。
2. **F7 h2 / F3 h4 的谓词等价性**：\`succeeded(getGroupStoreLayoutFact(op, valueVMIType)) && fact->stagingLayout\` 是否与
   被替换的 \`isCompactSmallGroupStore(layout, valueVMIType, numGroups, getConstantIndexValue(op.getRowStride()))\` **逐例等价**？
   这决定 \`:710\` 能否不加 \`succeeded()\` 守卫直接解引用 \`stagingLayout\`，也决定 F3 h4 会不会改变"哪些 store 被归为 compact"。

### 19.18.4 我的落地判决（已记录并发出）

* **F2**：只走**共享模板**路线（\`:902-907\`）+ F2 h3 照字面 + F2 h5 只换 contiguity 块（拆分树的分片检查不动）；
* **F3 h3 永不落地**；**F3 h2 暂缓**（缺 §8.1 shim，记为具名缺口，不许就地发明）；
* **F3 h4 + F7 h1 + F7 h2 落地**（F7 h2 视前置校验 2 决定是否加守卫）；
* **F3 h1 条件落地**（视前置校验 1）；两项校验的结果出来之前不落 F3 h1。

### 19.18.5 两条前置校验的答案（已核实到源码）

* **校验 1：F3 h1 可以照写落地（不是放宽）。** 被删的 \`getX2MemoryDistToken(elemType, "INTLV")\` 唯一条件就是 \`elementBits in {8,16,32}\`（\`VMIToVPTOMaskInternals.cpp:1290-1299\`）；
  而 \`getInterleaveStoreSupport\`（\`VMILayoutSupportRelationQueries.inc:80-108\`）的类型集**完全相同**、失败消息**逐字相同**、并且**额外要求 full physical chunks** —— 所以**任何被 token 拒绝的都不会因此被接受**（等价或更严）。
  → **F3 h1 照写落地**（换守卫 + 换查询，删掉两个 \`emitLogicalFailure\` 点与 token 检查）。
* **校验 2：F7 h1/h2 可落（带 \`succeeded()\` 守卫），但 F3 h4 是行为变更不是改名。**
  两者在**唯一一轴**上不一致——**lane stride**：旧谓词接受 \`laneStride == 1\`，而 staging 的计算要求 \`laneStride != 1\`（即 2 或 4）。
  于是存在一整类"旧为真、新为假"的形状：\`gs(numGroups, slots=8, laneStride=1)\` 且 \`elementCount == numGroups in {4,8}\`、unit row stride、payload 32 的倍数且 <256。
  后果分开看：
  * \`:710\`（F7 h2）：该处位于 \`alreadyCompact = laneStride == 1\` 的**else 分支**之后，所以那一类**不可能出现**，谓词在该分支内一致；但查询仍可能因其他原因失败，所以**仍需 \`succeeded()\` 守卫**，并保留原有的"计算布局"作为 else 路径——**该处不是语义变更**。
  * \`:1142\`（F3 h4）：**不是纯重构**。上述那一类 store 将**不再被判为 Compact**，而是落到 \`:1147\` 的 \`checkSupportedGroupSlotsStoreShape\`，即**换了一个校验器**，可能改变通过/失败/诊断。**必须作为行为变更来实测**（分诊方判断新行为"很可能是对的"：laneStride=1 的包本来就是 packed 形态，不需要 staging），但**由实测决定**。
  → **判决**：F3 h4 允许作为**行为变更**落地，附三条硬条件（严格子集；任何现在通过的用例若改变结局必须作为新增上报，回归或未解释的计划变化即回退；提交信息须写明语义与"由实测决定"）。

## 19.19 批次 B（F2）实测：**回归两个用例，已回退**；并**推翻我自己的一处判决**

### 19.19.1 实测

三处改动按约定落地（共享模板 \`:902-907\`、F2 h3 照字面插 \`:910\`、F2 h5 只换 \`:1422-1426\`），门禁结果：

* \`ninja\` exit 0、0 error；**G2 逐单元符号数前后完全一致**；
* \`lit/vpto\` **620/619/1 不变**（仍是既有的 \`vpto/vmi_f4x2_to_bf16x2_vcvt_llvm.pto\`）；
* \`lit/vmi_new\` = **608 / 539 / 69** —— 比基线 \`608/541/67\` **多两个失败**，**严格子集条件不成立**。

失败集合差（对 68 期集合）：

    < vmi_layout_assignment_group_reduce_s12_invalid.pto      （预期，来自 b58fa790f）
    > vmi_to_vpto_compress_tail_invalid.pto                   （新增，回归）
    > vmi_to_vpto_group_slot_widen.pto                        （新增，回归）

两条新增**干净地各自对应一个半批**：

* \`vmi_to_vpto_compress_tail_invalid\` → **F2 h5 半**：用 \`getReduceLayoutFactForLayouts\` 替换 contiguity 守卫，改变了"哪些非法 compress 形状被拒"以及拒绝的消息/阶段；
* \`vmi_to_vpto_group_slot_widen\` → **共享同宽半**：\`validateCastOperationRelation\` 严格强于旧的"布局相等 + arity 相等"，于是**把 lowering 变得比上游更严**。

已按要求**整批回退**（没有"丢掉一半试图抢救"），并复核：文件已 \`git checkout\`、树干净、\`ninja\` exit 0、\`lit/vmi_new\` 回到 **608/541/67**（与基线逐字节一致）。

### 19.19.2 推翻我自己的判决（第 N 次，如实记录）

我在 §19.17.3 判"可以复用 \`validateCastOperationRelation\` 作为共享 support 谓词，让两个阶段由构造保持一致"。**实测把它推翻了**：
\`vmi_to_vpto_group_slot_widen\` 在 **pre-port 是通过的**，所以那个组合在**本 base 上确实可下降** —— 一个拒绝它的谓词对 lowering 而言就是**过宽**，无论它在 planner 里有多正确。
于是新的判决：**lowering 里只加 support 的 fact 查询（\`getSameWidthCastLayoutFact\`），保留原来的"布局相等 + 物理 arity 相等"检查，不引入 solver 的拒绝闸**。这仍然落地了 F2 的意图（形状检查去问共享 support 模型），却不会让 lowering 比上游更严。

### 19.19.3 拆成两个子批次（已下达）

* **B1 = F2 h5 单独**：若 \`compress_tail_invalid\` 翻转，问题是"新拒绝是否与旧拒绝同因"；该用例是**负例**，其期望文本就是契约 —— 若消息或阶段变了，则 **contiguity 守卫必须保留**，fact 查询作为**附加检查**而不是替代。
* **B2 = 共享同宽改动 + F2 h3，但去掉 \`validateCastOperationRelation\`**：按 19.19.2 的新判决落地并实测；若 \`group_slot_widen\` 保持通过且失败集合是 \`608/541/67\` 的严格子集，则提交。

每个提交信息必须写明它属于哪个半批、改动前后数字、以及失败集合差。

### 19.19.5 B1 与批次 C 的最终结果（本轮收尾）

* **B1（F2 h5 单独）**：即使按最保守方向（**保留** contiguity 守卫、把 fact 查询作为**附加检查**跟在后面）仍然不稳：\`lit/vmi_new\` = **608/540/68**，翻转的正是目标负例 \`vmi_to_vpto_compress_tail_invalid\`（**单独双向验证**）。
  结论比规则预期的更强：对该输入，fact 查询**不是附加的**，而是**第二个、措辞不同的拒绝** —— "守卫prior 就不会改变结局"这条推理**不成立**。
  **判决 (b)**：F2 h5 记为**缺口 #2**，不在步骤 8 之前 re-baseline（负例文本是契约，只有"同因且更准确"才允许移动，而这一点未被核实）。
* **批次 C**：四个 hunk 全落（含一个计划外但必需的处理：两个调用点替换后 \`isCompactSmallGroupStore\` 无调用者，\`-Werror=unused-function\` 会**编译失败**，而 F3 h3 删除被禁止），实测 **608/540/68**，新增 \`vmi_to_vpto_group_store_compact_small.pto\` —— 即 **F3 h4 的分类是承重的**，读代码时的"应该走 packed 校验器"被**实测推翻**。整批回退。
* **我授权 fallback 变体只试一次**（旧谓词**优先**并 OR 上 fact 路由，使旧谓词判为 compact 的形状**由构造保持**），**结果中性**：
  \`f759c8256\` 落地 —— \`ninja\` 0/0、**G2 与基线一致**、\`lit/vmi_new\` **608/541/67**（失败集合不再新增）、\`lit/vpto\` **620/619/1**。
  落地形态里谓词仍有调用者，所以**不需要** \`[[maybe_unused]]\`，那个属性位置陷阱没有出现。**缺口 #3 因此被解决**，而不是记为最终。

### 19.19.6 一条**门禁标准**的澄清（我自己设错过，已纠正）

* **严格子集**这条标准属于 **solver 侧改动**（目的是修失败；无收益即尝试失败、必须回退）；
* **步骤 7 的批次是移植批次**，目的是对 fork delta 的**忠实度**；主门禁是**批次局部门禁**（G1/G2、逐族探针、构建干净、无死代码/未用代码），**lit 次级**；移植批次的通过条件是"**预期改动落地 + 两个套件都不回归 + 任何结局变化都被解释**"。
  B2（\`4fad83fdb\`）就是按这条标准落地的：0/0、无结局变化，但**结构性目标达成**。

### 19.19.7 三个具名缺口（都在 `.work/upstream-port/step7/PORTING_GAPS.md`，归步骤 8 预算）

1. **F3 h2**（group_slot_load）：需 §8.1 的 **stride 签名 shim**；
2. **F2 h5**（compress）：contiguity 守卫 vs 共享 reduce-layout fact 查询的**措辞/阶段冲突**，重访判据是"读该形状的查询消息、判定是否同因且更准确"；
3. **F3 h4**：**已解决**（fallback 变体中性落地），并记下"曾因字面替换回归 \`group_store_compact_small\`"作为保留旧谓词的理由。
另记：批次 C 那次 G2 的 \`MemoryInternals.cpp\` 49 → 48 是 \`[[maybe_unused]]\` 放在 \`static\` 之前导致正则失配的**人为产物**，不是定义丢失。

## 19.20 缺口 #1 是假缺口：stride 是我们自己加的，且早已在上游树里（F3 h2 已落地）

### 19.20.1 结论与证据

用户的关键一问（传 stride 是我们加的还是主线演进丢掉了）答案是：**是我们加的，而且早就加进了这棵树**。

* 上游原本只有三形参版本（上游自己的 \`929a1fe30\` 引入），我们**没有删它**，而是**并排加了重载**：
  \`include/PTO/Transforms/VMILayoutSupport.h:874\` 的 stride 版签名，形参依次是 resultType、sourceGroupStride、numGroups、reason。
* 来历：\`5fe30a018\`（add the stride-aware group_slot_load fact the solver drives）；
  \`git merge-base --is-ancestor 5fe30a018 1c1bba8cc\` **为假**，即它在**移植开始之后**由我们加入（支撑层补齐的一部分）。
* 头注释（同文件 862-873 行）写明：上游三形参声明**原样保留**，新增检查**都以 stride 操作数存在为前提**，
  因此**上游调用者验收不变**，只有 solver（总是带该操作数）看到更严的 fork 规则。
* F3 h2 想要的调用与我们重载的**实参逐一对应**，故**可照原文落地**。

**为什么之前记成缺口**：它抄了 \`MANIFEST.md 8.1\`，而那份写在**支撑层尚未补齐**的时候，之后没人回头对树复核 —— 与**过期的 G1 门禁**同型。
**本条写入文件作为规则**：缺口是对树的一个断言，动手前必须对着 HEAD 复核；\`MANIFEST.md 8.1\` 及其余旧结论一并列入步骤 7 的固定复核动作。

### 19.20.2 落地（提交 8）

\`648898a01\` 把 group_slot_load 形状检查的调用改为传 sourceGroupStride（cosmetic 的 accessPlan 与花括号改动未做，也不需要）。按**移植批次**门禁实测：

| 门禁 | 结果 |
|---|---|
| \`ninja\` | exit 0 / 0 error，两个产物重链 |
| G2 逐单元定义数 | 与 step-7 基线**完全一致** |
| \`lit/vmi_new\` | **608 / 541 / 67**，无新增（与 step-6 期记录只差 \`b58fa790f\` 修掉的那一个）|
| \`lit/vpto\` | **620 / 619 / 1**（既有 f4x2 转 bf16x2 vcvt 用例）|

值得记一句：slots=1 与 slots=8 的**专用助手本来就已经各自校验 stride**，所以这次改动在 lowering 里属**重复校验**；
fork 的意图是先把共享 support 模型问清楚，而实测说明它**不改变任何用例结局**。

### 19.20.3 目标已延长并重新武装

按用户要求：edit 把 maxGoalRounds 由 256 提到 **512**（objective 未改），再 resume 使 phase 由 blocked 转 active、activation 由 disarmed 转 armed（revision 6）。自动接力恢复。

### 19.20.4 一个新识别出的**系统性**模式（后续批次的筛选准则）

到目前为止，凡是把 lowering 的形状检查改去问 fork 共享 support 查询的改动，**两次里两次**都撞到负例测试（F2 h5 的 compress、批次 C 尝试里的 group_slot_widen）。
原因一致：**共享查询的拒绝措辞与触发阶段和上游内联检查不同，而上游的负例测试断言的正是旧措辞**。
F3 h2 之所以没撞上，只因它查的是**同一批**已被专用助手校验过的东西（重复校验，无措辞变化）。
**因此**：这类 hunk 不该逐个硬拱，而应**成组地**放进步骤 8，连同负例 re-baseline 一起处置；步骤 7 只落**不改变拒绝措辞与阶段**的部分。

## 19.21 item 3 的实测答案、由它暴露的一个**候选枚举缺口**，以及 D/E 批次的落地计划

### 19.21.1 item 3（\`vmi_layout_assignment_group_load_block8_truncf\`）：拒绝者不是 assignment

* **阶段实测**：只跑到 \`-vmi-layout-assignment\` **成功**（模块正常打印、无错）—— 所以**我们的 planner 不拒绝这个 component**；
  之前那句"我们的枚举把下游拒绝的布局交给 group_load"在**那个形态下被证伪**。
* **真正的拒绝点**：加上 \`-vmi-to-vpto\` 后报出的是 **lowering 的形状检查**（文本在 \`VMIToVPTOMemoryInternals.cpp:806\`）：

      :21:10: error: VMI-UNSUPPORTED: pto.vmi.group_load requires a supported UB source, a contiguous or
      block_deinterleaved f32 result layout, and a group/row-stride shape the block plans can address ...
        %4 = "pto.vmi.group_load"(...){num_groups = 8} : ... -> !pto.vmi.vreg<128xf32, #pto.vmi.layout<contiguous>>

* **端口与配对**：端口是 **group_load 的 result**，被赋成 \`contiguous\`；决定结局的是它与**访存形状**（stride 24、num_groups 8、128×f32：每 group 16 元素=64B，stride 24 元素=96B）的配对，由**上游自己的 strided-plan 判定**裁决。
* **哪个诊断才是诚实的**：测试钉的是 **truncf** 那条，而那条文本里写明 vcvt 的**输入**布局是 \`block_deinterleaved = 2\` —— 那**正是 group_load 的 result**。
  也就是说：**pre-port 的赋值给了该 result \`block_deinterleaved = 2\`，我们给了 \`contiguous\`**；两个拒绝各自诚实地描述了同一输入的不同缺陷，**谁先报由赋值选择决定**。

### 19.21.2 由它暴露的**候选枚举缺口**（比"消息"问题更根本）

我读了 planner 的 group_load 分支（\`VMILayoutPlanner.cpp:2108-2143\`），它与紧随其后的 group_store 分支（\`2145+\`）同形：

1. 候选 = **result 类型上的显式布局**（若有）+ **调用方交给 provider 的 domain**（\`polymorphicLayouts\`）；
2. 逐个用 \`getGroupLoadLayoutFact(typed, rowStride, numGroups)\` **校验**，通过才产出关系；
3. 只有在**候选为空**时才去问 preferred 行（2122-2127）。

**问题**：这个分支**从不枚举 group_load 自己的合法布局表**，只"校验调用方给的那几个"。而 load 家族的 polymorphic 集是 \`contiguous/ls2/d2/d4\` 一类，
**不含 \`block_deinterleaved = 2\`** —— 于是 planner **根本拿不到 bd2 这个选项**，只能给 contiguous，接着被 lowering 的 strided-plan 判定拒掉。
对照上游 pre-port：它有按 support 表 seeding 的约束路径（正是步骤 5 那批被删掉/离开决策路径的 seed 管线），所以**它能选 bd2**。

**结论**：这是**与 \`ea1c1e6bb\` 第三块同族**的缺口（"候选池没有唯一可用行时，去问 preferred/表"），只是这一次缺的是 **group_load 的合法结果布局**。
而用户的既有约束正好适用：**这些新 layout 我们的确是要枚举出来的，不能丢掉**。

**待验修复（有界实验）**：在 group_load 分支里，除显式布局与调用方 domain 外，**按 support 表枚举该形状的合法结果布局**（或在池中无合法行时回退到表/preferred 行），
然后实测：预期 \`vmi_layout_assignment_group_load_block8_truncf\` **转为通过**（它期望的 truncf 诊断会因选择 bd2 而出现，从而是**上游行为**的恢复，不是改消息）；
接受条件是**失败集合严格子集**，且任何其他用例的计划变化都必须被解释，否则回退并把本条并入步骤 8。

### 19.21.3 步骤 7 批次 D/E 的落地计划（据 F5/F4 规格）

* **F5（批次 D，12 hunks，-34/+132）**目标单元：\`VMIToVPTODataLayoutInternals.cpp\`（\`materializeMaskLayoutConversion\`、\`materializeAdjacentMaskGranularityConversion\`、\`createPredicateIntlv\`）与 \`VMIToVPTOPatternInternals0.cpp\`（\`OneToNVMIEnsureMaskLayoutOpPattern\`）。
  内容分三类：①**纯新增能力**（ensure 的 \`forwardsPhysicalParts\` 快路径、contiguous↔block-deinterleaved 的恒等转发）；②**新增 staging 助手**（\`materializeStagingDeintToContiguousMaskLayout\` / \`materializeStagingContiguousToDeintMaskLayout\` 的原型在 hunk 2，定义在后几个 hunk，被 hunk 4/5 调用）；③**把"拒绝"改成"委托给 staging 助手"**（hunk 4/5）与一处 unpack 顺序修正（hunk 6）。
  **批次内互相依赖 ⇒ 必须整批落地**；其中第 ③ 类**改变接受范围**，按 §19.20.4 的准则要预期可能的负例翻转。
* **F4（批次 E，2 hunks，-2/+114）**目标单元：\`VMIToVPTODataLayoutInternals.cpp\` 的 \`materializeDataLayoutConversion\`；
  规格明确：上游 post-fork 的两个提交（\`067da4864\` 组合式 dense 物化、\`dce6afea0\` dense lane-stride ↔ group-slot 桥）**建的是同一个函数，我们不是超集** ⇒ **必须手工合并**，且排在 F5 之后（F4 要穿过 F5 的路径）。
* **门禁准则（写死）**：整批落地后若失败集合是**严格子集或相等**则提交；若出现负例翻转，**回退**并把翻转清单记为**步骤 8 re-baseline 候选**（不在部分 delta 上动负例文本）。

### 19.21.4 那个有界实验已做：中性，已回退，并把下一步的读点定死

实现：把 group_load 分支的候选从"仅调用方 domain"改为"族 zoo"——复用 group reduce 的辅助函数并把它的名字统一为 \`getGroupFamilyQueryLayouts\`（语义 = seeds + contiguous + d{2,4} + bd{2,4}，**其中就含我们缺的 bd2**）。
实测：**608 / 541 / 67**，与基线完全相同、无新增，即 \`vmi_layout_assignment_group_load_block8_truncf\` **没有转绿**。按 solver 侧"无实测收益即回退"的规矩回退，树回到 \`648898a01\`。

这一步把问题缩小到**两个候选解释之一**（下一次只需**读**，不必再建）：

1. **表本身不为该访存形状承认 bd2**：那么 pre-port 能选到 bd2 就说明它**不走这个 fact 查询**，而走 seeding（support 表的 preferred/seed 路径）—— 那就要去看 group-load 表在 stride 24 / 8 groups / 128×f32 下到底有哪些行；
2. **表承认但代价模型偏好 contiguous**：那是 cost 比序问题（与 §19.16 的 frontier/Pareto 讨论同族），修点在代价模型而不是枚举。

读法：查 group-load 表在该形状下的行 + planner 里 group_load 分支候选择序的代价；两条都能在**无构建**的情况下判定。

### 19.21.5 读完了：**上游在 fork 之后给 strided group_load 加了 memAny 行**，我们的 solver 于是选了不可下降的 contiguous

**表本体（三处逐一核过）：**

* **fork 点**（\`ca7ccb409\`，即我从 fork 分叉时的上游内容）：group-load 表**只有**这些行——

      {bits<8,16,32>(), gb(1,4), memContiguous(), c()},   {bits<8,16,32>(), gb(1,2), memContiguous(), c()},
      {bits<8,16,32>(), gb(1),   memContiguous(), c()},   {bits<8,16,32>(), gb(2),   memContiguous(), c()},
      {bits<8,16,32>(), gb(4),   memContiguous(), c()},   {bits<8,16,32>(), gbFull(), memAny(), c()},
      {bits<32>(),      gb(2),   memBlockAligned(), bd(2)}, {bits<32>(), gb(4), memBlockAligned(), bd(4)},

* **上游后来加了三条 \`memAny()\` 行**（注释写着"strided source 需要自己的一行"）：

      {bits<8,16,32>(), gb(1), memAny(), c()},   {bits<8,16,32>(), gb(2), memAny(), c()},   {bits<8,16,32>(), gb(4), memAny(), c()},

* 我们的 fork **保留的是窄表**（无这三条）——这就是 fork 里从没这个问题、移植后才出现的原因。

**于是因果链闭合**：移植后我们的 solver 看到的是**上游的宽表**，对 "f32 / 64B group / stride 96B（跨 3 个 32B 块）" 这个形状，
窄的 \`memAny()\` 行给出了一条**免转换的 \`contiguous\`** 结果布局；我们的 solver 依代价选了它（无 ensure、最省），
而 lowering 对 strided 源只认两条路（块跨步 vsldb 计划要求 **block-deinterleaved** 结果；或整 chunk 计划要求 group 达一个物理 part），于是报出那条 generic 拒绝。
**上游 pre-port 的流程选的是 \`bd2\`**（测试期望里 vcvt 的输入就是 \`block_deinterleaved = 2\`）—— 也就是**上游的赋值并不只是"取表里最省的那一行"**。

**这不是表 bug、也不是测试 bug，而是"我们的 solver 与上游 lowering 的合法性口径不一致"**：solver 认为可采纳的（\`memAny()+c()\`），lowering 不认。
这与本项目里已解决的两次同型（\`ea1c1e6bb\` 的 preferred 行、\`bccd44a29\` 的双向查询）是一类：**让 solver 的口径与 support/lowering 的真相一致**。

**待验修复（下一步，solver 侧、非表、非测试）**：group_load 分支里，当**源 row stride 是 strided**（≠ group size）且形状为 f32 整块组时，
不要把 \`memAny()\` 那条 \`c()\` 当作合法候选/不要让代价模型偏好它，而要求落到 \`memBlockAligned()\` 的 \`bd\` 行（镜像 lowering 的 \`isSupportedBlockStrideF32GroupLoad\` 前提）。
预期：该用例**转绿**（因为回到了上游的 bd2 选择，truncf 诊断随之出现），且失败集合**严格子集**；任何其他用例的计划变化都必须被解释，否则回退并把本条并入步骤 8。

### 19.21.6 那条修复方向**被算术证伪**，item 3 的真相另有其形（本轮定案）

执行方按\`停就停\`的要求先做算术，结论是**我给的修复方向不成立**：

* \`isSupportedBlockStrideF32GroupLoad\`（\`VMIToVPTOMemoryInternals.cpp:551-570\`）要求 \`groupSize == lanesPerPart/8\` —— f32 的 \`lanesPerPart = 64\` ⇒ **fragmentElems = 8**；
  而本用例是 128×f32 / \`num_groups = 8\` ⇒ **groupSize = 16 ≠ 8** ⇒ 该谓词为 **false**。
  我又核了 \`getGroupSizeFromNumGroups\`（\`:475\`）——它**只按 elementCount / numGroups 算、与布局无关** ⇒ 改成 \`bd2\` **也不会**让该形状变得可下降。
  ⇒ 我原先那句"要求落到 bd 行（镜像 \`isSupportedBlockStrideF32GroupLoad\`）"是**错的**；这个形状在本 base 上**两条计划都不覆盖**（整块 f32 组 = 8 元素、一个物理 part = 64 元素，16 两者都不是）。
* 因此该编辑即使落地也只会产出**无收益变更**，严格子集条件不满足 ⇒ 执行方**停手、不写树**，处置正确。

**真相（读完测试的 RUN 后闭合）**：该用例的 RUN 是 \`not pto-test-opt ... -vmi-layout-assignment -vmi-to-vpto\`，
而它钉的两条消息（\`pto.vmi.truncf operand #0 ... ensure_layout cannot materialize this conversion\` 与 \`source/result layouts do not match a supported ensure_layout table row\`）
**属于赋值阶段的"物化规划"失败**——也就是说 **pre-port 的赋值选了 \`bd2\` 并插入 \`bd2 → contiguous\` 的 ensure，然后因该 ensure 无法物化而失败**；
**我们的 planner 反过来绕开了这个不可能转换**（直接给 \`contiguous\`、不插 ensure），于是赋值**成功**，失败被推迟到 \`-vmi-to-vpto\` 的 strided group_load 检查。

**所以这是一条负例的合同问题，属步骤 8**，并且有一条对**我方诊断有利**的论证要一并记下：
我们的消息（"这个 strided 形状没有可用计划"）**比测试钉的那条更准确**地指出了真实障碍——测试那条把责任归给了一个我们**根本不需要的** ensure 转换；
换句话说，pre-port 之所以报那条，是因为它先选了 bd2 才自找了一个无法物化的 ensure。这符合我方的既有规矩（只有在"新文本是同一错误的更准确表述"时才允许移动负例文本），因此**re-baseline 到我们的消息是可辩护的**，但要在步骤 8 与其它负例一起成组处理。

**更深一层的缺陷（记入 backlog）**：表里的 \`memAny()\` 行（上游 fork 后新增，见 §19.21.5）**承认**了一个 lowering 不接受的形状，
而我们的代价模型偏好"无需转换"的计划，于是没有任何环节在赋值期拦住它——pre-port 是靠那个"多余 ensure 无法物化"**间接**拦下的。
这属于\`solver 合法性口径 vs lowering 真相\`的同一族（\`ea1c1e6bb\`、\`bccd44a29\`），但**修点在合法性模型**（要不要让 planner 直接拒绝该 component），需单独立项与实测，不在本轮硬拱。

## 19.22 步骤 9 的前置核查（本轮只读完成）

**性能量具在位且是冻结版**（\`.work/upstream-port/perf/\`）：9 个自包含用例（含决定性的 \`gbmc-amp-dep\` 与 \`truncf-amp2\`，各带 \`kernel.pto\` / \`ptoas.flags\` / \`main.cpp\` / \`launch.cpp\` / \`golden.py\` / \`compare.py\`）、
\`capture5.sh\`（按 RUN 抓取全部用例的下降 MLIR，用于在花仿真机时之前先比 IR）、\`cmp5.sh\`（两份抓取的逐用例行数差）、\`loop_period.py\`（从 msprof 轨迹提稳态循环周期），
以及 **19 个已记录的 \`sim-runs/\`**（base/split 成对：dep_base/dep_split、truncf_merge/truncf_split、ew_k1/ew_k2 …）—— 也就是说复现性能结论这件事**有参考基线**。

**环境可用性（实测）**：\`msprof\` 在 \`/usr/local/Ascend/cann-9.0.0/bin/msprof\`，\`ASCEND_HOME_PATH\` = \`/usr/local/Ascend/cann-9.0.0\`，
\`/usr/local/Ascend/ascend-toolkit/latest\` 存在，\`ninja\` 在 PATH（\`msprof.py\` 不在，但我们用的是 \`msprof\` 本体）。
**\`llvm-lit\` 不在 PATH**，但一直是按绝对路径调用的（\`/home/mouliangyu/projects/github.com/vpto-dev/llvm-project/build-shared/bin/llvm-lit\`）—— 这不是缺口，只是记下来免得下次误判。

⇒ **步骤 9 在本机可跑**：全门禁（\`validate_port.sh\` + 两个套件 + 差分）、A5 侧先用 \`capture5.sh\` 做 IR 级比较再花仿真机时、性能结论按 \`sim-runs/\` 的成对基线复现。

## 19.23 步骤 8 的侦察（本轮实测）：56 个 FileCheck-only 差异**主要是"我们的 solver 选了不同（通常同样合法）的布局"**

做法：在干净树上跑 \`llvm-lit -v lit/vmi_new\`（67 failures，与 608/541/67 基线一致），然后对日志归类。

**错误类别计数**（按日志里的 error 行）：\`no match found\` 83、\`ASSIGN-SAME\` 37、\`CHECK\` 22、\`ASSIGN\` 16、\`match on wrong line\` 7、\`CHECK-DAG\` 4、\`CHECK-COUNT\` 4 —— 即 **FileCheck 类占绝大多数**；
另有 9 处 \`'<stdin>' is empty.\`（工具无输出 ⇒ 属于那 8-9 个硬失败类），以及 \`VMI-LAYOUT-CONTRACT\` 8、\`VMI-UNSUPPORTED\` 6、\`VMI-RESIDUAL-OP\` 2 的工具侧报错。

**首个不匹配期望的分桶**（每个 RUN 取第一处）：**含布局期望的 42 条 / 不含的 42 条**（样本 84 条，多于 67 个用例是因为一个用例多条 RUN）。
含布局的那半直接写着 \`#pto.vmi.layout<contiguous>\` / \`deinterleaved = 2|4\` / \`contiguous, lane_stride = 2|4\` / \`num_groups = 8, slots = 8\` 这类**结果布局**；
不含布局的那半多是布局差异的**下游后果**（\`call @callee\` 的多结果形态、\`vadds\`、\`extf\`、\`mask_and\` 等）。

**两个具体样例**（日志原文）：

* \`vmi_layout_assignment_group_slots_scf_for.pto:53\`：期望 group_load 结果 \`contiguous\`，我们给 \`block_deinterleaved = 2\`（并让 mask 侧也走 bd2）；
* \`vmi_layout_assignment_group_reduce_s32_store.pto:32\`：期望先有一处 \`pto.vmi.ensure_layout\` 把 source 拆开，我们的计划**省掉了那次转换**（更省）。
  注意它与 item 3（§19.21.6）方向**相反**：那里上游选 bd2、我们选 contiguous；这里上游期望 contiguous、我们选 bd2 —— 说明差异是**逐形状的决策差异**，不是单向偏好 bug。

**结论与步骤 8 的裁决规则（写死）**：这些不是拼写差异，而是**不同决策引擎的合法决策差异**（上游的 CHECK 行编码的是**上游 LayoutSolver 的决策**）。因此步骤 8 的做法只能是**逐例裁决**，且必须给出理由：

1. 若我们的决策**合法且不劣**（更少的 ensure、更少的 ls 数量、层次顺序一致）⇒ **更新该上游用例的期望**（这属于移植的一部分，_不是_ 往我们自己的用例里抄上游决定）；
2. 若我们的决策**更差或明显不合理** ⇒ 记为该用例对应的**solver 缺陷**并修 solver（这才是"我们的 solver 是决策引擎"的应有状态）；
3. 若差异来自 **lowering 侧**（调用边界、多结果 call、缺 op）⇒ 归**步骤 7**；
4. 拼写/属性名类（如 \`dintlv→intlv\`、\`merge/zeroing→zero\`）⇒ 直接改名，无裁决成本。

⇒ 步骤 8 的工作量由此**从"56 个未知差异"变成"逐例裁决 + 四类处理"**，并且第 1 类必须成批处理（同一决策原因往往命中多个用例）。

## 19.24 批次 D（F5）阶段 1 映射：**12 个 hunk 里有 5 个上游早已实现**，整批落地不是正确形状

执行方按只读阶段做完逐 hunk 映射，关键发现是**规格与拆分树的结构已经分叉**：F5 写的是 fork 单体的一个巨型 \`materializeMaskLayoutConversion\`，
而上游把这整片**重构成了具名助手**（\`VMIToVPTODataLayoutInternals.cpp\` 的 \`materializeDataLayoutConversion\` / \`materializeDeinterleaved2MaskLayout\` / \`materializeMaskLaneStrideUnpack\` / \`materializeBlockLayoutForwarding\` / \`tryMaskLayoutMaterializers\` / \`buildMaskGranularityConversionPlan\` 等）。

**逐 hunk 判定结果（12 个）：**

| 类别 | hunk | 说明 |
|---|---|---|
| **上游早已实现（no-op）** | h2、h3、h9、h11 | h2 的两条 staging 声明**逐字节相同**；h3 的 contiguous↔block 恒等转发上游做得**更丰富**（\`materializeBlockLayoutForwarding\`，含 unrealized-cast 路径）；h9/h11 的 \`resultArity\` 已提在 driver 里 |
| **真实缺口（纯新增能力）** | h1、h6、h7、h8、h12 | h1：ensure 取出 \`forwardsPhysicalParts\` 却**没有**用它做恒等短路（该字段在全树里**只被代价模型消费**，\`VMILayoutCostModel.cpp:465/:1865\`，**从不被任何物化器使用**）；h6/h7/h8：laneStride==4 时**首次** unpack/pack 应该用 \`b16\`；h12：\`OneToNVMIEnsureMaskLayoutOpPattern\` 只把 fact 当守卫 |
| **拒绝措辞变化（须单独筛查）** | h4、h5 | 把 \`deinterleaved={2,4} … requires {2,4}*N parts\` 的不等 arity 情形**改成委托给 staging** —— 正是我的移植批次准则里要筛的那一类 |
| **需实测才能判** | h10 | groupSlots↔groupSlots 同 arity 恒等；上游的 chunked 路径**可能已产出等价 IR**，但它会改变 staging |

**适配冲突（照抄编译不过）**：h1/h12（fact 的 \`&supportReason\` 形参、且上游用 \`replacePhysicalResults\` 而非 \`replaceOpWithFlatConvertedValues\`）；h6/h7/h8（fork 的单个局部 \`pairType\` 需要穿过 \`MaskLaneStridePackContext::packPair\` 的签名**以及** unpack 循环 —— 三处而非两处）。**缺失符号：无**。

**判决：整批落地不是正确形状**（5/12 是 no-op、3 个要手工适配进不同结构、h4/h5 会动拒绝文本）。改为两个提交：

* **提交 A（纯新增能力，不碰任何拒绝文本）**：**h1 + h6/h7/h8 + h12** —— 这也是最可能修掉两个 VMI-RESIDUAL-OP 硬失败（"failed to apply conversion patterns"）的一组；
* **提交 B（单独筛查）**：**h4/h5**（负例风险）与 **h10**（IR 形状），按 §19.20.4 的准则处理。
  
另有一个**必须先回答的开问题**：staging 的**定义体**在 \`VMIToVPTOPatternInternals0.cpp:45\` 与 \`:232\`（声明在 \`DataLayoutInternals.cpp:1303-1308\`），F5 的 staging 语义必须先与这两个既有实现体对照，确认不是同一份代码再谈移植。

## 19.25 步骤 8 的第一半已落地：我们的 33 个 conformance 用例此前根本不在上游树里（fidelity 现在可测：12/33）

**怎么发现的**：本轮在干净树上跑 \`validate_port.sh\`，第 3 节 \`fork conformance suite (33 files)\` **输出为空**，第 4 节差分直接报 \`no conformance tests under .../test/lit/vmi_new\`，并把 33 个用例列为 \`only in A\`。
也就是说：目标里的「我们的 vmi_new 用例通过」这一条，**此前根本无法测量** —— 参考 dump 里有 33 个用例，树里一个都没有。

**处置**：把 fork 的 33 个 \`vmi_layout_cost_conformance_*.pto\` 拷入上游树，只做上游方言需要的规范化（\`pmode = "merge"|"zeroing" -> "zero"\`，命中 5 个文件），**不改任何期望**——拷贝的目的是**拿 fork 的决策来量移植**，不是调期望。
先 dry-run 了 \`step8/copy_tests.sh\`，确认这 33 个不需要 \`dintlv\` / \`castptr\` 那两项改名。提交 \`2b783862c\`，已推 fork。

**实测（拷贝到位后的同一次测量）**：

* 全局 **641 discovered / 553 passed / 88 failed**；
* 其中 **21 个是 conformance 文件本身**，另外 **67 个与拷贝前完全一致** ⇒ **零回归**；
* 于是 **fidelity 指标可测了，当前值 = 12 / 33（36%）** —— 这既是「我们的 vmi_new 用例通过」的真实进度，也是步骤 8 的直接抓手。

**两条边界（写死）**：

1. \`step8/copy_tests.sh\` 的**整批模式会拷 568 个文件**（fork 的整个 vmi_new），那会**覆盖上游自己的用例版本**、可能把上游期望换成 fork 期望 —— 与「不把上游决策抄回我们自己用例」相反。**没有跑整批**；后续必须**逐文件**决定（本轮这 33 个之所以安全，是因为上游树里**根本不存在**同名文件）。
2. 新的全局基线是 **641 / 553 / 88**（= 原 67 + conformance 21）；后续测量都要按这个基线断言，别再拿 608/541/67 当全局数（那是 conformance 未入树前的状态）。

**下一步**：把这 21 个 conformance 失败**按原因分类**（方言/属性差异 vs 真正的决策差异 vs 缺 lowering 能力），再按 §19.23 的四类规则裁决。它们比上游那 56 个 FileCheck 差异**更适合先做**：量的是**我们自己 solver 的忠实度**，且有明确的 33 个基线可比。
