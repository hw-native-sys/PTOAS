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
