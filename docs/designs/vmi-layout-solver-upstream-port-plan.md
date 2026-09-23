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
| 动手前必须先做 | **vexpdif layout 层还没提交**（VMILayoutSupport.{h,cpp}、VMILayoutPlanner.cpp、VMIMaskGranularityAssignment.cpp）——先提交它 |

## 2. 已确认的设计决策

1. **事实与偏好分离。** 候选集合（表行）属于事实，跟随上游（包括 `0967a848b` 的泛化 dense
   lane-stride 行）；preference penalty 仍然是我们决策链里的**靠后 tie-break**（`pen` 排在
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
| 2 | 搬运“无侵入”文件：3 个头文件 + `VMILayoutPlanner/CostModel/ConflictSolver.cpp` → `lib/PTO/Transforms/VMI/`，加上一致性工具 `tools/pto-test-opt/pto-test-vmi-layout-cost-conformance.cpp`；补 CMake | `lib/PTO/Transforms/CMakeLists.txt`、工具 CMake | 0.5 天 | 编译通过且行为不变 |
| 3 | **支持层合并（关键路径）**：采用上游表/DSL；把我们的 9 个枚举查询写成对 `...ForLayout` 的循环；两张 spine-scoped 表按普通候选源**无条件**枚举（不迁 `VMILayoutSpineAnalysis`，删掉它的 .h/.cpp/CMake 行/lit）；只保留我们真正新增的表（`kVexpdifLayoutPatterns` 5 行、`kGeneratedMaskStagingPatterns` 4 行）和我们自己的事实字段（`intrinsicRearrangementCost`、`stagingLayout`、`forwardsPhysicalParts`、`VMIReduceLayoutFact`、`VMIGeneratedMaskLayoutFact`、`VMIInterleaveStoreSupport`、`VMIVexpdifLayoutFact`） | `VMI/VMILayoutSupport.cpp` + `.inc` 表 + `VMILayoutSupport.h` | 4-6 天 | **用上游自己的决策链路跑上游测试仍然通过**（这是“合并行为中性”的证明） |
| 4 | 传播器合并：保留上游传播器的分析入口（**不含** spine scope 集合，`setSpineScopedCastOps/isSpineScopedCast` 删除），加上我们的 `requestExact/installPlanned/endExactRequests/getRequestedLayout(OpOperand&)`；消掉唯一重名符号 `isVMISameLayoutOp`（采用上游的算子集合，因为分类现在来自上游分析） | `VMI/VMILayoutPropagation.cpp` 及头文件 | 1.5-2 天 | 链接干净；class-edge 相关 lit 全绿 |
| 5 | 替换决策：在上游 `applyLayouts()` 里插入我们的 `selectLayoutPlan()` + `commitVMILayoutPlan()`（以及我们的结构化 seeding），删掉那 4 处 seed 调用；保留 `collect/materializeCallBoundaries/insertDataUseMaterializations/rewriteFunctionType/validateVMILayoutAssignedIR` | `VMI/VMILayoutAssignment.cpp`（约 500 行改动） | 2-3 天 | 我们的 26 个一致性测试通过 |
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
| 枚举查询按 `...ForLayout` 重写后值域不全 | 用 planner debug 跑 26 个一致性测试：期望看到“无完整合法 plan”的硬失败，而不是错误代码 |
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
