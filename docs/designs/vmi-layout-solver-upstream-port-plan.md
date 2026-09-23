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
