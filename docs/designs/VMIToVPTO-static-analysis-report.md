# VMIToVPTO 静态分析报告记录

## 报告来源与范围

本文记录 2026-09-05 从内部静态分析报告导入的结果，目标文件为
`lib/PTO/Transforms/VMIToVPTO.cpp`。报告不是本仓库
`check_changed_code.py` 的输出，而是对完整文件进行的质量扫描；因此它包含历史代码问题，
不等价于当前 MR 的增量检查结果。

报告原始阈值包括：方法规模超过 50 行、圈复杂度超过 20，以及工具定义的数据泥团、
默认捕获、匿名 namespace、warning suppression 等规则。原始报告中的行号随主线变化，
后续复核时应以扫描时提交版本为准。

## 分类汇总

| 类别 | 报告中出现次数 | 含义 |
| --- | ---: | --- |
| `huge_method` | 46 | 方法超过工具配置的 50 行阈值 |
| `huge_cyclomatic_complexity` | 33 | 方法圈复杂度超过 20 |
| `Data Clumps` | 11 | 多个方法重复携带相同参数组 |
| `warning_suppression` | 1 | 使用了 warning suppression |
| 其他规则 | 报告未给出明确计数 | 包括 `G.FUN.01-CPP`、`G.RES.06-CPP`、`G.INC.12-CPP` 等需结合完整扫描上下文确认的规则 |

同一个方法可能同时命中多个类别，因此分类数量不能相加作为问题总数。

## 重点问题清单

### 方法规模与圈复杂度

报告指出以下高风险集中区域：

- `verifySupportedVMIToVPTOOps`：约 896 行，圈复杂度约 166；
- `materializeDataLayoutConversion`：约 383 行；
- `materializeMaskLayoutConversion`：约 278 行，圈复杂度约 71；
- 多个 `matchAndRewrite`：约 52–397 行，圈复杂度约 22–116；
- `lowerGroupSlotLoadParts`：约 134 行，圈复杂度约 29；
- `createSubVLGroupPeriodicChunk`：约 104 行，圈复杂度约 23；
- `computeGroupMaskMaterializationForType`：约 67 行，圈复杂度约 22；
- `checkSupportedGroupStoreShape`、`checkSupportedScatterShape`、`checkSupportedVmullShape`、
  `checkSupportedGroupBroadcastShape` 等 shape 检查函数也同时命中规模或复杂度规则。

完整方法、数值和扫描行号见随报告导入的原始文本；由于原始内容为单行导出，不能把其中的
行号直接当作当前工作树行号。

### 数据泥团（Data Clumps）

报告重点指出以下重复参数组：

- `createIotaContiguousChunk`、`createSubVLGroupPeriodicChunk`、`createIotaDeinterleavedChunk`；
- `materializeContiguousToLaneStride`、`materializeLaneStrideToContiguous`、
  `materializeGroupSlotLaneStride`；
- `materializeDataLayoutConversion`、`materializeMaskLayoutConversion` 及 staging mask layout
  转换函数；
- 多个 mask granularity/layout conversion helper 之间重复的 `op/sourceType/resultType/
  sourceParts/rewriter/message` 参数组。

这类问题不应通过随意增加全局状态解决。可行方向是按语义引入小型上下文对象或
`llvm::ArrayRef`/结构化参数，但需要保持调用点的类型约束和错误路径清晰。

### 其他规则

- `G.FMT.11-CPP`：选择和循环语句使用花括号。当前 MR 已修复该 MR 新增/修改行中的两处，
  不能据此宣称整个文件的历史问题已清零。
- `G.FUN.01-CPP`：函数职责需要单一。主要与超长的 verifier、materialization 和
  conversion pattern 函数重叠，应通过按职责拆分来处理。
- `G.RES.06-CPP`：避免 lambda 默认捕获。需要逐个确认 lambda 的实际捕获集合，不能机械地
  通过 warning suppression 绕过。
- `G.INC.12-CPP`：对于不需要导出的变量、常量或函数使用匿名 namespace。需结合链接可见性
  和跨翻译单元使用情况判断，不能对所有文件级符号一律包裹。
- `warning_suppression`：报告记录到一处 warning suppression。应定位其具体编译器告警和
  作用域，优先修正代码或使用最窄的、有理由的抑制；禁止新增 blanket suppression。

## 与仓库内置检查器的关系

`check_changed_code.py` 是增量预检查器，只检查相对目标分支的变更行，主要覆盖文本级格式、
安全和脚本规则。它不计算 AST 级圈复杂度、方法规模、数据泥团，也不能发现整文件历史债务。

因此出现“内置检查为 0，但完整报告有很多问题”是预期行为，不是同一工具漏报。完整复核应
组合使用具备编译数据库的 `clang-tidy`、复杂度工具（如 `lizard`）和 include 分析工具，
并以内部 EChecker 的规则口径作为最终门禁标准。

## 处理建议

1. 先确认报告对应的提交版本，并保留原始扫描结果作为基线。
2. 将 `verifySupportedVMIToVPTOOps`、`materializeDataLayoutConversion`、
   `materializeMaskLayoutConversion` 等超大函数作为结构化重构候选，拆分后逐次运行 lit 回归。
3. 对重复参数组设计语义明确的上下文类型，避免隐藏状态和无约束的大型参数对象。
4. 单独处理 lambda 捕获、文件级可见性和 warning suppression；这些规则不能通过简单格式化解决。
5. 每个重构 MR 保持小范围，分别记录功能回归、静态指标变化和未处理的历史问题。

## 本轮整改记录（2026-09-05）

`OneToNVMIGroupStoreOpPattern::matchAndRewrite` 中的 `group_slots(slots=1)`
路径已抽取为 `lowerSlots1` 私有辅助方法（提交 `e16d9868f`）。该方法集中负责
slots=1 的物理 arity/元素宽度校验、unit-stride 打包到连续 `vsts` 或非对齐
stateful stream 的选择，以及非 unit-stride 的 1PT fallback；主 `matchAndRewrite`
仅保留操作数归一化和布局分派。拆分保持原有诊断顺序与 lowering 语义，避免通过
无意义切割或全局状态规避复杂度指标。

本轮复核：

```text
python3 .agents/skills/enforce-ptoas-code-compliance/scripts/check_changed_code.py \
  --repo . --base origin/master --fail-on none
checked_files=1 errors=0 warnings=0
git diff --check  # passed
```

增量构建仍受工作区既有 CMake 外部依赖配置阻断：构建系统尝试创建
`/cann-cmake` 并因权限不足失败；该错误未产生 C++ 编译诊断，需在修复构建环境后
补跑完整编译及 lit 回归。

随后将 `OneToNVMIGroupBroadcastLoadOpPattern` 的直接 BRC 结果构造抽取为
`lowerDirectBRC`（提交 `8f7456fc7`）。该 helper 只负责 BRC 的物理 arity、指针类型、
结果 vreg 类型和逐 group offset 构造；E2B 以及 group-slot fallback 仍由原分派函数
处理。使用现有 `build/tools/pto-test-opt/pto-test-opt` 对
`vmi_layout_assignment_group_broadcast_load_e2b_b16.pto` 的 lowering 运行成功，输出仍
包含预期的 `vlds {dist = "E2B_B32"}`/BRC 路径。

本轮又将 `OneToNVMIGroupStoreOpPattern` 中已证明 32-byte 对齐的 packed-byte
快路径抽取为 `lowerAlignedPackedByteStore`（提交 `dcc8387bc`）。该 helper 只构造
`vpack` 链、有效 lane mask 和 `NORM_B8` store；未对齐路径仍进入原有 PK4 或
stateful stream 逻辑。抽取后增量合规检查仍为 `errors=0 warnings=0`，GitCode push
hook 通过。

随后将同一 packed-byte 分支的非对齐 stateful stream 地址物化与发射抽取为
`emitPackedByteStoreStream`（提交 `19289e5b0`）。该 helper 统一处理 destination
指针转换、offset 合成和 stream 发射，调用方仅准备 packed values 与 lane advances；
`PK4_B32` 直写仍保持独立路径。增量合规检查继续通过。

回归抽样使用现有 `build/tools/pto-test-opt/pto-test-opt` 运行：

```text
vmi_to_vpto_group_store_slots8_packed_byte.pto                         exit=0
  输出包含 PK4_B32、vstus stream 以及 NORM_B8 路径
vmi_to_vpto_group_store_slots1_unit_stride_alignment.pto               exit=0
  输出包含对齐 vsts 与非对齐 vstus 路径
vmi_layout_assignment_group_store_slots1_unit_stride.pto               exit=0
```

这组 case 覆盖了本轮抽取涉及的三种后端选择，未观察到 lowering 失败。

另外将 `checkSupportedGroupStoreShape` 的 `group_slots` 支持判定抽取为
`checkSupportedGroupSlotsStoreShape`（提交 `f330e7ecb`），集中处理布局事实、dense
memory access proof、slots=1 的 1PT 约束和 slots=8 的 unit-stride 约束；通用
group-store（one-block/deinterleaved）检查流程保持原顺序。

本轮再将 `checkSupportedGroupLoadShape` 的 block-deinterleaved f32 专用校验抽取为
`checkSupportedBlockDeinterleavedGroupLoadShape`（提交 `acf22e584`）。该 helper
封装 layout fact、dense read proof、指针/组数/row-stride 约束和 full physical chunk
检查；contiguous group-load 路径保持原有判断顺序。相关新增代码合规检查通过。

同时将 `checkSupportedGroupSlotLoadShape` 的 `slots=1` 元素宽度和对齐步长约束
抽取为 `checkSupportedSlots1GroupSlotLoadShape`（提交 `dc240eceb`），使 slots=8
unit-stride 判定与 slots=1 1PT 对齐判定具有独立职责；原有 memory proof、指针类型
和 layout fact 检查顺序保持不变。

本轮将 `checkSupportedScatterShape` 的物理承载检查抽取为
`checkSupportedScatterPhysicalShape`（提交 `0c463c20e`），把 value/indices/mask
的 physical arity 一致性及 full-chunk 证明与 scatter 的布局、元素宽度和索引契约
分离。调用顺序和诊断保持不变，增量合规检查通过。
随后将 `checkSupportedGatherShape` 的 physical arity、四寄存器上限和 full-chunk
证明抽取为 `checkSupportedGatherPhysicalShape`（提交 `1cb382b06`），由 gather 主
检查只负责布局、元素宽度和索引契约。`b16` 单物理 part 的 partial-chunk 例外仍由
原有条件传入，保持诊断与支持范围不变。

本轮还将 `checkSupportedStrideStoreShape` 与 `checkSupportedStrideLoadShape` 共享的
单 physical chunk 校验抽取为 `checkSinglePhysicalStrideAccess`（提交 `1454cdbba`），
通过调用方传入的方向性诊断区分 store/load；同时修正了该批新增代码触及的单行控制
语句大括号。合规检查通过。

针对 `expand_load`，将静态 all-active mask 下的多 chunk 普通 `vlds` lowering 抽取为
`OneToNVMIExpandLoadOpPattern::lowerStaticExpandLoad`（提交 `0f4eee56c`）。runtime-mask
路径仍保留 prefix-index、`vgather2_bc` 和 `vsel` 的 expand 语义，两条路径没有被
错误合并；增量合规检查通过。
另外将 verifier 中六个 vector-scalar 算术操作共享的 pmode 检查和 maskable
physical-vreg 检查抽取为模板 helper `verifySupportedVecScalarOp`（提交
`9f997d659`）。它保留每个操作原有 op name 和 `pmode=merge` 诊断，降低主 verifier
walk lambda 的重复分支；新增代码合规检查通过。
本轮继续将普通 maskable 向量操作的结果类型提取为模板 helper
`verifySupportedMaskableOp`（提交 `eb6a4f802`），覆盖 add/sub/mul、div/min/max、
abs、sqrt、exp、ln 等重复 dispatch；`pmode` 特殊处理仍仅位于 vector-scalar helper。
同时修复了本批新增代码触及的控制语句大括号，合规检查通过。
本轮又将位运算、移位、一元 not 和 select 的 verifier dispatch 全部改用
`verifySupportedMaskableOp`（提交 `ad0189d3f`），进一步消除同构分支；随后按
`G.FMT.11-CPP` 补齐了 `getMaskGranularityRank` 的控制语句大括号（提交
`ab7407a2f`）。两次增量合规检查均通过。
本轮将普通 reduce（addi/addf/max/min，整数与浮点）的重复验证与诊断流程抽取为
`verifySupportedReduceOp`（提交 `03fc8ef6f`），保留 `requiresReassoc` 和每个
操作原有提示文本。另将 expand-load 路径中被此次重构触及的结果类型失败分支统一
为带大括号形式（提交 `c0d2922d9`）。增量合规检查通过。
随后将六个 group-reduce verifier 分支统一抽取为 `verifySupportedGroupReduceOp`
（提交 `c7dea762c`）。helper 只封装 support check、reason 传播和 WalkResult，调用方
保留每个 group-reduce 的后端能力描述；新增代码合规检查通过。

本轮再将 group-broadcast、`vdhist` 和 `vchist` 的同构 shape-check + 诊断流程抽取
为模板 helper `verifySupportedShapeOp`（提交 `013a971e5`）。helper 只负责 reason
传播和 WalkResult 转换，具体 support checker 与每个操作的能力描述仍在调用点明确
指定；增量合规检查通过。

本轮进一步复用 `verifySupportedShapeOp` 处理 `fptosi`、`fptoui`、`sitofp` 和
`bitcast` 的 conversion shape 验证（提交 `5dff6114a`），仅保留各操作独立的能力
描述与 checker，删除重复的 reason/WalkResult 样板代码；增量合规检查通过。

本轮继续将 `extsi`、`extui`、`trunci` 的 conversion shape 分支改用
`verifySupportedShapeOp`（提交 `b13d8f42f`），各自保留详细布局/宽度支持描述；同时
修复 expand-load 操作数失败路径的大括号（提交 `02a3be5d7`）。合规检查通过。
本轮将 `channel_split` 与 `channel_merge` 的 channel 数量判定、shape checker
调用和失败诊断抽取为 `verifySupportedChannelOp`（提交 `b3b17704e`），保留 2/4
channel 的专用能力提示和各自 layout 约束；合规检查通过。

本轮将 `constant_mask` 的 materialization checker 与统一错误诊断抽取为
`verifySupportedConstantMaskOp`（提交 `3abab5964`），使 verifier walk 只负责操作
分派；同时修复了本轮相邻修改路径中的两个单行控制语句大括号。增量合规检查通过。

随后将常量 mask 与 group mask 的物理 chunk 遍历统一收敛到模板辅助函数
`materializeMaskChunks`（提交 `de5e92069`）。调用方分别提供 dense mask 值谓词和
group active-lane 谓词；helper 统一处理 physical part/chunk/lane 遍历、padding、
physical-to-logical lane 映射及 `ConstantMaskChunkMaterialization` 构造，保持两类
mask 的语义差异只存在于 active-lane 判定。补齐该批代码及 scalar group-store 路径
触及的控制语句大括号后，增量合规检查结果为 `checked_files=1 errors=0 warnings=0`，
`git diff --check` 通过。

本轮将 `OneToNVMIDeinterleaveLoadOpPattern` 的非对齐两轮 `vldus` 与 `vdintlv` 组合
抽取为 `lowerUnaligned`。helper 显式维护 `streamBase/streamAlign` 的更新链，负责
ptr 物化、offset 合成、每轮增量和 low/high 结果重排；direct `vldsx2` 路径保持独立。
`vmi_interleaved_memory_ops.pto` 回归通过，增量合规检查与 `git diff --check` 通过。

本轮将 `OneToNVMIGroupStoreOpPattern` 的 scalar `group_store(num_groups=1)` 路径
抽取为 `lowerScalarGroupStore`。helper 单独负责物理 value arity、vreg/mask 类型、
1PT dist token 和 `PAT_VL1` mask 的校验及 `vsts` 发射，主分派函数不再混入 scalar
语义与 slots=8/packed-byte 路径。`vmi_to_vpto_group_store_slots1_1pt.pto` 回归通过，
增量合规检查和 `git diff --check` 通过。

本轮将 `createSubVLGroupPeriodicChunk` 的非 power-of-two residual lane 拼接抽取为
`createResidualSubVLGroupPeriodicChunk`。该 helper 负责按 ASC/DESC 公式生成每个
sub-group 的调整向量、构造 lane-range mask 并完成 `vsel` 合并；原函数保留 vreg、
sub-VL、all-mask、group-size 校验及 power-of-two 快路径。`iota_group_subvl`、
`iota_group_vl_half` 和 `iota_group2` 三个 case 均通过完整 lowering，增量合规检查和
`git diff --check` 通过。

本轮将 `OneToNVMILoadOpPattern` 的 dense lane-stride direct load 路径抽取为
`lowerLaneStride`。helper 负责逐 physical chunk 的 `vlds`、semantic offset 累加、
active-lane 计算和结果替换；主函数保留 dist 合法性判断及 contiguous/deinterleaved
fallback，未改变 load 路径选择顺序。`vmi_to_vpto_load_store_contiguous.pto` 回归
通过，增量合规检查和 `git diff --check` 通过。

本轮又将 slots=8 普通/lane-stride 两条非对齐 store stream 的 destination pointer
物化、offset 合成和 `emitStatefulStoreStream` 调用统一抽取为
`emitGroupStoreStream`。该 helper 只封装地址与 stream 生命周期，保留 packed-byte
专用 stream 和对齐 `vsts` 路径的独立选择；packed-byte 与 scalar 代表性 group-store
case 均回归通过，增量合规检查和 `git diff --check` 通过。

本轮继续整理 `materializeMaskLaneStrideLayout`：将 contiguous lane-stride unpack
分支前置为独立方向路径，集中处理结果 arity、`punpack` 层级和 mask 类型校验，pack
路径只保留 `ppack`/`por` 合并逻辑。这样降低了同一函数中双向控制流的嵌套，同时保持
lane_stride=2/4 的物理 part 顺序与诊断文本不变；相关 mask granularity 和 group-store
回归通过，增量合规检查与 `git diff --check` 通过。

本轮将 `materializeMaskGranularityCastLayoutConversion` 的 layout、staging 和
contiguous fallback 分派抽取为 `materializeMaskGranularityCastLayoutFallback`。主函数
保留 layout 存在性与 identity forwarding 检查，并统一处理 fallback 成功、失败和最终
诊断；helper 保持原有 layout→staging→dense-split contiguous 的尝试顺序，未改变转换
语义。相关 mask granularity case 回归通过，增量合规检查与 `git diff --check` 通过。


另外将 `computeShuffleVselrPlans` 的单个 result physical chunk 规划抽取为
`computeShuffleVselrPlanForChunk`。外层函数现在只负责布局因子和 chunk 枚举，辅助函数
集中处理 padding、logical-to-physical lane 映射、单 source chunk 约束及升序/降序 affine
索引判定；失败原因仍通过原有 `reason` 通道传播。为满足本文件的控制语句规范，复合
`FailureOr` 条件拆成具名状态并保持原短路语义；增量合规检查和 `git diff --check` 均通过。

本轮将相邻 mask granularity 转换按方向拆分为
`materializeWideningMaskGranularityPart` 与 `materializeNarrowingMaskGranularityPart`
两个职责明确的 helper，分别封装 `punpack` 展开和 `ppack`/`por` 合并；外层转换只负责
每个 layout part 的 chunk 计数、方向选择及最终 arity 校验，保持多步转换的顺序和错误
诊断不变。`vmi_to_vpto_ensure_mask_granularity.pto` 回归通过；多步/identity 相关 case
仍受测试自身的 pack/unpack 顺序 invariant 约束。增量合规检查与 `git diff --check`
均通过。

本轮将 factor=4 的 deinterleaved→contiguous mask staging 组合抽取为
`materializeFactor4DeintToContiguousGroup`，集中封装两级 `predicate intlv` 及四个
结果的顺序化构造；外层函数继续负责 source group 遍历和结果 arity 截断，factor=2
路径保持原有直接转换。增量合规检查和 `git diff --check` 通过。

本轮又将 mask granularity cast 的 staging layout 分派抽取为
`materializeMaskGranularityCastStagingForFactor`。该 helper 统一处理 contiguous 与
element-deinterleaved factor=2/4 的方向判断及失败传播，外层只遍历受支持的 factor，
消除了四组重复的 layout 条件和调用样板；转换优先级及结果保持不变。
`vmi_to_vpto_ensure_mask_granularity.pto` 继续通过完整 lowering，增量合规检查与
`git diff --check` 通过。

本轮进一步将 direct deinterleaved load 的 factor=2/4 物理构造分别抽取为
`lowerDeinterleaved2` 与 `lowerDeinterleaved4`。两个 helper 分别封装 `vldsx2` 的
chunk offset、结果类型一致性、factor=4 的两级 `vdintlv` 重排及结果替换；主 load
pattern 仅负责 footprint、地址 dist 合法性和路径选择。普通 load/store 代表性 case
回归通过，涉及 pack/unpack 的多 chunk 测试仍受既有 pipeline invariant 约束；增量
合规检查与 `git diff --check` 通过。

本轮将 `OneToNVMIDeinterleaveLoadOpPattern` 之外的 group-load
block-deinterleaved f32 物理 `vsldb` 构造抽取为 `lowerBlockDeinterleaved`，集中处理
block stride、part/chunk offset、结果 vreg/mask 校验和结果替换；主 pattern 保留布局
契约、arity 和 row-stride 检查。新增代码的增量合规检查和 `git diff --check` 通过，
相关 layout-assignment pipeline 未产生该 direct lowering 路径，未将其误报为功能回归。

本轮将 `OneToNVMIGroupBroadcastLoadOpPattern` 的 E2B lowering 抽取为
`lowerDirectE2B`。helper 集中负责 E2B layout/element/stride/arity 契约、packet
`vlds` 生成和跨 dense-split part 的结果复用；主 `matchAndRewrite` 仅保留 BRC、E2B
与 group-slot fallback 的能力分派，原有优先级和诊断保持不变。
`vmi_layout_assignment_group_broadcast_load_e2b_b16.pto` 回归通过，增量合规检查和
`git diff --check` 通过。

本轮将 verifier 中重复的普通算术、向量标量算术、位运算、比较选择等 maskable
操作分派收敛到 `verifySupportedVMIArithmeticOp`，并将统一的物理 vreg/mask 能力
检查提取为 `emitMaskableUnsupported`。该 helper 只负责按操作类别选择既有
`verifySupportedMaskableOp`/`verifySupportedVecScalarOp`，保留每个操作的诊断名称、
`pmode=merge` 检查以及原 verifier 的 memory/layout/compare 优先顺序；`vaddc`、
`vmull`、`relu` 等具有独立 shape 契约的操作仍保持专用分支。相关源码增量检查结果为
`checked_files=1 errors=0 warnings=0`，`git diff --check` 通过。

本轮回归了 `vmi_interleaved_memory_ops.pto`、
`vmi_to_vpto_load_store_contiguous.pto` 和
`vmi_layout_assignment_group_broadcast_load_e2b_b16.pto`，均以 `pto-test-opt` 完整
lowering 成功。尝试增量编译时仍被工作区既有 CMake 外部依赖配置阻断：构建系统尝试
创建 `/cann-cmake` 而权限不足，未产生 C++ 编译诊断。

随后将该 pattern 的 group-slot fallback 抽取为 `lowerGroupSlotFallback`，集中负责
根据 unit-stride 选择 slots=8/1、构造物理 source 类型、生成 source part 类型以及调用
group-slot load 与广播物化。主分派函数只保留 direct BRC/E2B 能力判断和 fallback 选择，
保持原有 lowering 顺序与诊断；相关 E2B 代表性 case 和增量合规检查均通过。

随后对称地将 factor=4 contiguous→deinterleaved staging 的两级 `predicate dintlv`
组合抽取为 `materializeFactor4ContiguousToDeintGroup`。该 helper 负责四个 source 的
分组校验、低/高半部交织以及最终四路结果顺序；外层函数只负责补齐缺失 source、维护
part 分组和结果 arity。转换语义与原有 factor=2 路径保持不变，增量合规检查、
`git diff --check` 及 mask granularity 回归均通过。

本轮将 `materializeDataLayoutConversion` 中多组经 contiguous 中间布局的 fallback
判定和递归转换抽取为 `materializeDataLayoutViaContiguous`。该 helper 负责 lane-stride
与 deinterleaved 组合的识别、中间 part 类型数量计算及两阶段转换；主函数保留 simple、
专用 deinterleave、lane-stride 和最终失败路径，fallback 优先级与输出语义不变。相关
layout/store/shuffle 代表性 case 回归通过，增量合规检查和 `git diff --check` 通过。

本轮进一步将动态 `create_group_mask` 的单个物理 chunk 生成抽取为
`materializeDynamicGroupMaskChunk`。辅助函数负责 index/块内 lane 推导、active-lane
比较、padding mask 合并和结果类型检查；外层 `materializeDynamicGroupMaskForType`
仅负责全局 layout/arity 校验、active-lane 限幅以及 part/chunk 枚举，避免把不同层次的
契约混在一个循环中。`vmi_to_vpto_create_group_mask_block8_dynamic.pto` 与
`vmi_layout_assignment_create_group_mask_s32_dynamic.pto` 均通过完整 lowering 回归，
增量合规检查和 `git diff --check` 通过。

本轮继续将整数/浮点转换与 bitcast 的 verifier 分派抽取为
`verifySupportedVMIConversionOp`。该 helper 仅聚合共享的 shape-check 与诊断模板，
每个转换操作仍显式绑定原 checker 和完整支持说明，主 walk 保留 memory、layout、
compare、算术及专用 shape 检查的先后顺序。源码增量合规检查结果为
`checked_files=1 errors=0 warnings=0`，`git diff --check` 通过；连续代表性
`pto-test-opt` lowering case 通过。`vmi_to_vpto_integer_casts.pto` 在测试自身的
`unpack` pipeline invariant 处提前失败，未进入本轮 verifier/lowering 变化路径。

本轮将 `vaddc`、`vaddcs`、`vmull`、`relu` 和 `vselr` 等具有独立 shape 契约的
专用 verifier 分派抽取为 `verifySupportedVMISpecialOp`。helper 只聚合既有 checker
与错误诊断，未改变这些操作的能力范围、诊断内容或相对于 memory/layout/compare/
arithmetic 检查的顺序。源码增量合规检查及 `git diff --check` 通过；连续的 load/store、
interleave memory 和 group-broadcast lowering 回归均通过。

本轮将 `materializeMaskLaneStrideLayout` 按方向拆分为
`materializeMaskLaneStrideUnpack` 与 `materializeMaskLaneStridePack`。前者只负责
`punpack` 展开，后者只负责 `ppack`/`por` 合并及结果 arity 检查；外层仅保留布局
方向、lane stride 支持范围和 helper 分派。删除了原函数中已由 helper 取代的重复
控制流，未改变 mask 物理 part 顺序或转换语义。源码增量合规检查、`git diff --check`
及连续 load/store lowering 回归通过。

本轮将普通 reduce 与 group-reduce 的 verifier 分派抽取为
`verifySupportedVMIReductionOp`。helper 保留 `reduce_addf` 的 `reassoc` 要求、各
整数/浮点 reduce 的具体诊断，以及 group-reduce 的独立支持说明；主 walk 只负责按
memory/layout/compare、算术、专用 shape、归约的顺序调用分类 helper。源码增量合规
检查和 `git diff --check` 通过；`vmi_group_reduce_addi_i16.pto` lowering 通过。
两个普通 reduce 样例仍在测试自身 `unpack` pipeline invariant 处提前失败，未进入本轮
verifier/lowering 变化路径。

本轮将 `fma`、`extf` 和 `truncf` 的浮点 shape verifier 分派抽取为
`verifySupportedVMIFloatOp`，保留各自的支持描述和 checker 绑定，主 walk 不再混合
浮点转换与其它操作类别。源码增量合规检查和 `git diff --check` 通过；普通连续
load/store 回归通过。现有三个独立浮点样例均在测试自身 `unpack` pipeline invariant
处提前失败，未进入本轮 verifier/lowering 变化路径。

本轮将 `materializeDeinterleaved4Layout` 的两个方向拆分为
`materializeDeinterleaved4ToContiguous` 与 `materializeContiguousToDeinterleaved4`。
前者独立负责四路 `vintlv` 重排，后者独立负责两级 `vdintlv` 及结果 part 分配；外层
只负责布局方向判定、空输入诊断和结果转发，避免一个函数同时维护两套相反的数据布局
算法。源码增量合规检查和 `git diff --check` 通过；连续 load/store 与 group-store
回归通过。factor=4 专项样例在既有 `unpack` pipeline invariant 处提前失败，未进入
本轮转换逻辑。

本轮将 `lowerGroupBroadcastParts` 中 slots=1 且结果 chunk 覆盖多个 group 的 lane
映射、splat 和 `vsel` 合并逻辑抽取为 `materializeSlots1GroupBroadcastChunk`。
该 helper 单独负责跨物理 source chunk 的 lane-mask 构造，主 lowering 继续负责布局
selector 选择、常量路径和 `vselr` 路径；未改变 group broadcast 的结果顺序或错误诊断。
源码增量合规检查、`git diff --check` 及 group-broadcast lowering 回归均通过。

本轮继续拆分 `lowerGroupBroadcastParts`：将普通结果 chunk 的 lane 映射验证、常量
`vdup` 路径和 `vselr` 结果生成抽取为 `materializeGroupBroadcastChunk`，并用具名的
`GroupBroadcastSelectorKind` 表达 selector 计划。slots=1 跨 source 的特殊合并仍由
`materializeSlots1GroupBroadcastChunk` 处理，避免两种语义重新耦合；布局检查、结果
顺序和失败诊断保持不变。源码增量合规检查、`git diff --check` 及 group-broadcast
和连续 load/store lowering 回归均通过。

本轮将 memory verifier 中重复的 load shape 诊断闭包抽取为具名 helper
`emitMemoryUnsupported`，使 `verifySupportedVMIMemoryOp` 只负责操作分类和能力
检查；诊断内容及 stable masked-load 选项语义保持不变。源码增量合规检查、
`git diff --check` 和连续 load/store lowering 回归通过。

本轮将 memory verifier 的写路径分派抽取为 `verifySupportedVMIMemoryStoreOp`，
统一承载 store、interleave_store、group_store、masked_store、stride_store 和
scatter 的 shape checker 与诊断；load、expand-load 及 stable masked-load 选项仍由
原 helper 按原顺序处理。该拆分只改变职责边界，不改变 lowering 能力或错误文本。
源码增量合规检查、`git diff --check` 通过；连续 load/store、interleave memory 和
group-store lowering 回归均通过。

本轮将 layout verifier 按职责拆分为 `verifySupportedVMIEnsureLayoutOp`、
`verifySupportedVMIMaskLayoutOp` 和 `verifySupportedVMIMaskGranularityOp`，外层
`verifySupportedVMILayoutOp` 仅按原顺序串联三类检查。数据 layout、mask layout 和
granularity cast 的诊断及支持关系保持不变。源码增量合规检查、`git diff --check`，
连续 load/store、mask-granularity 和 group-store lowering 回归均通过。

本轮进一步将 masked-load、gather 和 expand-load 的高级读路径抽取为
`verifySupportedVMIMemoryAdvancedLoadOp`，使 `verifySupportedVMIMemoryLoadOp` 只
负责基础 load/group-load 分类与检查。stable masked-load 的保留策略、各 fallback
支持描述和诊断优先级均保持不变；源码增量合规检查、`git diff --check` 及连续
load/store、interleave memory、group-reduce lowering 回归均通过。

本轮继续将 `verifySupportedVMIMemoryOp` 按读写方向拆分为
`verifySupportedVMIMemoryLoadOp` 与 `verifySupportedVMIMemoryStoreOp`，顶层 helper
只负责保持 load→store 的检查顺序。load 中的 stable masked-load 选项仍在原位置
优先诊断，store 的 shape checker 与诊断保持不变。源码增量合规检查、`git diff --check`
及连续 load/store、interleave、group-store lowering 回归均通过；masked-load 专项样例
按其既有测试约束失败，未显示本轮读写分派变化。

本轮将 `materializeDeinterleaved2Layout` 的两个方向拆分为
`materializeDeinterleaved2ToContiguous` 与 `materializeContiguousToDeinterleaved2`。
前者只负责 `vintlv`，后者只负责 `vdintlv` 和结果 part 分配；外层仅负责布局方向
判定及结果转发。同步补齐本轮触及的 group-load 控制语句大括号。源码增量合规检查、
`git diff --check` 及连续 load/store、interleave memory lowering 回归通过。

本轮将 `materializeMaskGranularityConversion` 的多级 granularity 递进循环抽取为
`materializeMaskGranularitySteps`；顶层函数保留支持性校验、相邻转换快路径和最终
方向选择。每一步仍复用 `materializeAdjacentMaskGranularityConversion`，保持
b8/b16/b32 的递进顺序、布局属性和失败诊断不变。源码增量合规检查、`git diff --check`
及 `vmi_to_vpto_ensure_mask_granularity.pto` 回归通过。

本轮将 `verifyNoResidualVMIIR` 的 create-mask、constant 和 residual VMI 检查拆分为
`verifyNoResidualCreateMask`、`verifyNoResidualConstant` 及显式的 residual 判定，
使最终 IR 检查的各类失败职责独立。错误文本、遍历顺序和 pass failure 行为保持不变。
源码增量合规检查、`git diff --check` 及连续 load/store lowering 回归通过。

本轮继续拆分 `lowerGroupBroadcastParts` 的 selector 状态管理。新增
`GroupBroadcastSelectorContext`，集中表达 selector 类型、源 lane stride、shift、共享
ramp 与按 base index 的缓存；`getGroupBroadcastSelector` 只负责 selector 的缓存查找、
constant `vdup` 生成、共享 `vci` ramp 初始化及 base offset 添加。主 lowering 仍负责
layout fact 选择、物理 lane 校验和结果 chunk 分派，selector 的生成顺序与缓存语义保持
不变；同时修正该区域残留的枚举名称引用。增量合规检查为 `errors=0 warnings=0`，
`git diff --check` 通过；group-broadcast 与连续 load/store lowering 回归通过。

本轮将 `verifySupportedVMIToVPTOOps` 中常量、broadcast、group-broadcast、直方图、
active-prefix 和 compress 相关的杂项 shape 检查抽取为 `verifySupportedVMIMiscOp`。该
helper 只负责这些操作各自的支持性验证与原有诊断，主 verifier 保留 memory、layout、
compare、misc、arithmetic 等既有检查顺序，避免将不相关的操作契约混在同一巨大函数中。
增量合规检查为 `errors=0 warnings=0`，`git diff --check` 通过；group-broadcast、
连续 load/store 和 interleave memory lowering 回归通过。

本轮再将 channel split/merge、shuffle 及 constant-mask 的检查抽取为
`verifySupportedVMIChannelShuffleOp`。其中 shuffle 的 forwarding、lane0 splat 和
vselr 三种候选路径及组合诊断保持原顺序，主 verifier 仅负责分类分派。增量合规检查为
`errors=0 warnings=0`，`git diff --check` 通过；连续 load/store 与 interleave memory
lowering 回归通过。

本轮将 memory、layout、compare、misc、arithmetic、special、reduction、float 和
conversion 的标准检查顺序封装为 `verifySupportedVMIStandardOp`。`verifySupportedVMIToVPTOOps`
仅保留标准分类、channel/shuffle 收尾分类和 walk 结果处理，既有检查优先级与诊断路径
不变。增量合规检查为 `errors=0 warnings=0`，`git diff --check` 通过；连续 load/store
与 interleave memory lowering 回归通过。

本轮抽取 `verifyGroupBroadcastChunkMapping`，统一 group-broadcast 普通 chunk 与
slots=1 特殊 chunk 共用的 padding/lane 映射、selector 预期 group 及 source chunk
校验。`lowerGroupBroadcastParts` 不再重复维护该契约，selector 生成和结果分派语义不变。
增量合规检查为 `errors=0 warnings=0`，`git diff --check` 通过；group-broadcast 与
连续 load/store lowering 回归通过。

本轮继续拆分 `OneToNVMIGroupStoreOpPattern::lowerSlots1`：将 unit-stride 的多源
value 拼接及 aligned/unaligned store 选择抽取为 `lowerSlots1PackedUnitStride`，将
逐 group 的 1PT fallback 抽取为 `lowerSlots1PointStores`。两个 helper 分别承担 packed
stream 与 point-store 语义，原有对齐判定、mask 和诊断保持不变。增量合规检查为
`errors=0 warnings=0`，`git diff --check` 通过；group-store 与连续 load/store
lowering 回归通过。

本轮将模板化 `OneToNVMIIotaOpPattern::matchAndRewrite` 的 grouped-iota 分支抽取为
`lowerGroupedIota`。该 helper 独立负责 group size 合法性、contiguous layout、物理
arity、共享 chunk 缓存及 sub-VL/VCI 选择；普通 contiguous 与 deinterleaved iota
路径保持在主 pattern 中，诊断与结果顺序不变。增量合规检查为 `errors=0 warnings=0`，
`git diff --check` 通过；group2、group-subvl、group-size1 iota lowering 回归通过。

本轮进一步将普通 contiguous 与 deinterleaved iota 物化分别抽取为
`lowerContiguousIota` 与 `lowerDeinterleavedIota`，主 pattern 仅保留输入检查、布局
分派和结果替换。物理 chunk 顺序、factor 校验和诊断文本保持不变；增量合规检查为
`errors=0 warnings=0`，`git diff --check` 通过；group2、group-subvl、group-deint
iota lowering 回归通过。

本轮将模板化 interleave pattern 的 zero-copy `vintlv/vdintlv` chunk 重排与结果类型
校验抽取为 `materializeZeroCopyResults`。该 helper 只负责两类 zero-copy 布局的 chunk
顺序及 arity/type 契约，lane-stride carrier 与单 chunk direct path 保持原实现。增量
合规检查为 `errors=0 warnings=0`，`git diff --check` 通过；interleave memory 与连续
load/store lowering 回归通过。

本轮将 `OneToNVMIMaskedStoreOpPattern` 的普通 contiguous mask/value materialization
与逐 chunk `vsts` 路径抽取为 `lowerContiguous`。helper 独立负责 data/mask layout
转换、physical arity、active-lane 与 predicate 构造、地址证明和 store 发射；主 pattern
继续保留 lane-stride dist 快路径。该拆分保持 masked-store 的 predicate、chunk offset
和对齐约束语义不变。增量合规检查结果为 `checked_files=1 errors=0 warnings=0`，
`git diff --check` 通过；连续与 interleave memory 代表性 lowering 均成功。

本轮将 `OneToNVMIGroupStoreOpPattern` 的 `slots=8 + lane_stride` 路径抽取为
`lowerSlots8LaneStride`。helper 独立负责 lane-stride dist/mask granularity、每个 slot
block 的地址合法性、未对齐 compact + stateful stream fallback，以及对齐逐块 `vsts`
和 active-lane mask；主 pattern 仅做布局分派。该拆分保持 dist token、stream advances、
mask 和 offset 语义不变。增量合规检查结果为 `checked_files=1 errors=0 warnings=0`，
`git diff --check` 通过；group-store 与连续 load/store 代表性 lowering 均成功。

本轮将 `OneToNVMITruncIOpPattern` 的 factor narrowing（多个 source physical chunks
转换后合并到单个 result chunk）抽取为 `lowerFactorTrunc`。helper 负责 source/result
mask、按 factor 选择 part、`VcvtOp`/`VorOp` 合并以及 alias 结果收尾；主 pattern 保留
width、arity 和布局判定。该拆分保持 part 顺序、物理 arity 与 signed alias bitcast 语义
不变。增量合规检查结果为 `checked_files=1 errors=0 warnings=0`，`git diff --check`
通过；连续 load/store 与 group-store 代表性 lowering 均成功。

本轮将 `OneToNVMIGroupLoadOpPattern` 的 block-deinterleaved f32 参数校验与 chunk
一致性检查抽取为 `lowerBlockF32`。helper 负责 group size/factor、num_groups、row
stride、pointer、result arity、block elements 以及各 part chunk uniformity 校验，随后
调用既有 `lowerBlockDeinterleaved` 发射 `vsldb`；主 pattern 只保留布局识别和操作数归一化。
该拆分不改变 block load 的布局约束、stride 语义或结果顺序。增量合规检查结果为
`checked_files=1 errors=0 warnings=0`，`git diff --check` 通过；连续 load/store 与
group-store 代表性 lowering 均成功。

随后复核 `OneToNVMIStoreOpPattern` 时发现其两个输入校验分支缺少闭合大括号，导致
后续 store lowering 语句在源码结构上落入错误作用域。已补齐 `lanesPerPart` 与地址
operand 校验分支的大括号；这是格式/控制流正确性修复，不改变 store lowering 语义。
增量合规检查结果为 `checked_files=1 errors=0 warnings=0`，`git diff --check` 通过；
连续与 interleave memory lowering 均成功。

本轮将 `fptosi`/`fptoui` 的 widen（Even/Odd）物理发射路径统一抽取为
`lowerWidenFpToInt`。helper 负责 source/result physical arity、all-true mask、part
索引和逐 chunk `VcvtOp`/结果替换；两个 pattern 保留各自的 conversion contract 与
诊断文本。该拆分保持 signed/unsigned widen 支持矩阵和结果顺序不变。增量合规检查结果
为 `checked_files=1 errors=0 warnings=0`，`git diff --check` 通过；连续 load/store 与
group-store 代表性 lowering 均成功。

本轮将 `OneToNVMISIToFPOpPattern` 的 same-width 与 widening 物理转换发射抽取为
`lowerConversion`。helper 统一负责物理 arity、`VcvtOp` part 选择及结果替换；主 pattern
保留 source/result element contract 与 mask 构造。该拆分保持 `si32→f32` 与 `si8→f16`
的支持矩阵、rounding/part 语义和诊断文本不变。增量合规检查结果为
`checked_files=1 errors=0 warnings=0`，`git diff --check` 通过；连续 load/store 与
group-store 代表性 lowering 均成功。

本轮将 `OneToNVMITruncFOpPattern` 的 factor narrowing（按 result lane stride 选择
source factor，逐 chunk 执行 `VcvtOp` 并用 `VorOp` 合并）抽取为 `lowerNarrow`。helper
统一负责 source/result mask、BF16x2 source view、part 索引和结果替换；主 pattern 仅
保留 width/factor 推导及其它布局路径。该拆分保持 truncf 的 rounding/saturate contract、
part 顺序和物理 arity 语义不变。增量合规检查结果为 `checked_files=1 errors=0 warnings=0`，
`git diff --check` 通过；连续 group-store 与 load/store 代表性 lowering 均成功。

本轮将 `OneToNVMITruncFOpPattern` 的 contiguous same-width `VcvtOp` 路径抽取为
`lowerSameWidth`。helper 负责 source mask、rounding mode、逐 physical chunk 转换和
结果替换；主 pattern 保留 packed BF16x2 source view、布局判定以及其它 narrowing
路径。该拆分不改变 same-width fp-to-fp 的 round/saturate contract 或结果顺序。增量
合规检查结果为 `checked_files=1 errors=0 warnings=0`，`git diff --check` 通过；连续
group-store 与 load/store 代表性 lowering 均成功。

本轮进一步将 `lowerGroupBroadcastParts` 的源布局、selector plan、shift 合法性及
index mask 初始化抽取为 `createGroupBroadcastLoweringContext`，并以
`GroupBroadcastLoweringContext` 传递真实的 lowering 状态。主函数仅保留结果 chunk
枚举与分派，selector 生成顺序、缓存和错误诊断保持不变。增量合规检查为
`errors=0 warnings=0`，`git diff --check` 通过；group-broadcast lowering 回归通过。

本轮将 `materializeMaskLaneStridePack` 的单个结果 chunk 组装逻辑抽取为局部
`materializeChunk`：该逻辑统一处理低/高 half 的 `ppack`、lane-stride=4 的二级 pack
及 `por` 合并；外层仅负责结果类型、source/result arity 和 chunk 枚举。mask 语义与
pack 顺序保持不变。增量合规检查为 `errors=0 warnings=0`，`git diff --check` 通过；
mask granularity lowering 回归通过。

本轮继续拆分模板化 interleave pattern，保留 zero-copy 重排 helper，并将 lane-stride
carrier 与 direct contiguous 路径维持为独立分支。当前改动只调整前置检查的具名条件和
zero-copy helper 的职责边界，不改变 `vintlv/vdintlv` 的 layout fact、结果顺序或
目标指令生成。增量合规检查为 `errors=0 warnings=0`，`git diff --check` 通过；
interleave memory 与连续 load/store lowering 回归通过。

本轮回归检查额外发现 `OneToNVMIConstantMaskOpPattern` 的结果 arity 分支缺少闭合
大括号，导致后续 physical replacement 语句落入条件块；已补齐该控制流边界并单独提交。
这是正确性修复，不改变 constant-mask 的物化算法。增量合规检查为
`errors=0 warnings=0`，`git diff --check` 通过；受测 constant-mask 样例在既有
`unpack` pipeline invariant 处提前失败，未进入该 pattern。

本轮将 `OneToNVMICreateMaskOpPattern` 的 dynamic 与 constant active-lanes 物化分别
抽取为 `lowerDynamicMask` 与 `lowerConstantMask`。前者负责 runtime prefix mask 和
partition remaining，后者负责 padding/lane 映射、prefix pattern 与 PLT fallback；主
pattern 仅负责输入归一化和路径选择。mask part 顺序、active-lanes 限幅及错误诊断保持
不变。增量合规检查为 `errors=0 warnings=0`，`git diff --check` 通过；create-mask
样例在既有 `unpack` pipeline invariant 处提前失败，未进入该 pattern。

本轮将 `materializeSimpleDataLayoutConversion` 中重复的 identity part forwarding
抽取为 `forwardIdentityLayoutParts`，统一处理 identity、单 lane group 和 block-
deinterleaved forwarding 的共同契约。layout fallback 选择、unrealized cast 优化及
错误诊断保持不变。增量合规检查为 `errors=0 warnings=0`，`git diff --check` 通过；
连续 load/store lowering 回归通过。

本轮进一步将 block-deinterleaved 与 contiguous 转换中的 unrealized-cast 输入直通
识别抽取为 `forwardBlockLayoutCastInputs`，主函数只负责布局关系选择和普通 identity
校验。cast 输入数量及类型匹配条件保持不变。增量合规检查为 `errors=0 warnings=0`，
`git diff --check` 通过；连续 load/store lowering 回归通过。

本轮将 `materializeDeinterleaved2MaskLayout` 的正向与反向 mask 重排分别抽取为
`materializeDeinterleaved2MaskToContiguous` 和
`materializeContiguousToDeinterleaved2Mask`。外层仅负责方向识别、2*N arity 与
identity forwarding 校验；`pintlv/pdintlv` 的 part 顺序和错误诊断保持不变。增量
合规检查为 `errors=0 warnings=0`，`git diff --check` 通过；mask granularity
lowering 回归通过。

本轮将 dense lane-stride pack/unpack 共用的 arity、元素宽度和 stride 合法性检查
抽取为 `validateDenseLaneStrideShape`。两个方向的 materialization 函数分别保留
carrier 变换和结果枚举职责，统一校验不改变原有诊断与支持范围。增量合规检查为
`errors=0 warnings=0`，`git diff --check` 通过；连续 load/store lowering 回归通过。

本轮继续整理 `materializeStagingDeintToContiguousMaskLayout`，将 factor=2 的单 group
`pintlv` 物化与结果容量处理抽取为局部 `appendFactor2Group`，factor=4 仍通过既有
专用 helper 处理。staging 分组顺序、尾部容量语义和失败诊断保持不变。增量合规检查为
`errors=0 warnings=0`，`git diff --check` 通过；mask granularity lowering 回归通过。

本轮同样将 `materializeStagingContiguousToDeintMaskLayout` 的 factor=2 单 group
`pdintlv` 物化抽取为局部 `appendFactor2Group`，factor=4 继续复用既有四路 helper；
source 补零、part 聚合和结果 arity 检查保持不变。增量合规检查为
`errors=0 warnings=0`，`git diff --check` 通过；mask granularity lowering 回归通过。

本轮将模板化 interleave pattern 的 lane-stride carrier 路径抽取为
`lowerLaneStrideInterleave`，集中处理 carrier 宽度校验、输入/输出 bitcast 和目标
interleave 指令生成；主 pattern 继续负责 layout fact 查询及 contiguous/zero-copy
路径分派。增量合规检查为 `errors=0 warnings=0`，`git diff --check` 通过；interleave
memory lowering 回归通过。

本轮将 `materializeDynamicGroupMaskChunk` 的 padding lane 分类与有效 mask 合并抽取
为 `applyGroupMaskPadding`，chunk 主体只负责 index/ramp 计算和 active-lane 比较；
padding mask 的构造、`pand` 合并及失败诊断保持不变。增量合规检查为
`errors=0 warnings=0`，`git diff --check` 通过；dynamic create-group-mask lowering
回归通过。

本轮将 `materializeDynamicGroupMaskForType` 的 part/chunk 枚举与单 chunk helper 调用
抽取为 `materializeDynamicGroupMaskChunks`，顶层函数仅保留 layout、group size、物理
arity 和 power-of-two block 校验。chunk 顺序、active-lanes 限幅和失败传播保持不变。
增量合规检查为 `errors=0 warnings=0`，`git diff --check` 通过；dynamic
create-group-mask lowering 回归通过。

本轮将 `materializeMaskLaneStridePack` 的 `ppack`/`por` 状态封装为
`MaskLaneStridePackContext`，集中管理 all-true mask 缓存、低/高 half pack 及 mask 合并；
外层仅负责 source/result arity 和 chunk 枚举。pack 顺序与 lane_stride=2/4 语义保持不变。
增量合规检查为 `errors=0 warnings=0`，`git diff --check` 通过；mask granularity
lowering 回归通过。

本轮将 `getContiguousMaterializationPartCount` 的 VMI layout 提取与 dense-split
part/chunk 一致性检查抽取为 `getMaterializationLayout` 和
`verifyMaterializationPartCounts`。主函数仅负责 contiguous 类型构造及最终 physical
arity 计算，错误文本和 layout 支持范围保持不变。增量合规检查为
`errors=0 warnings=0`，`git diff --check` 通过；连续 load/store lowering 回归通过。

本轮将 `lowerGroupBroadcastParts` 的单个结果 chunk 定位、slots=1 特殊合并、selector
映射验证和普通 selector 物化抽取为 `lowerGroupBroadcastChunk`。外层仅负责 layout
fact、context 初始化及 part/chunk 枚举；source chunk 边界、结果顺序和 selector cache
语义保持不变。增量合规检查为 `errors=0 warnings=0`，`git diff --check` 通过；
group-broadcast lowering 回归通过。

本轮将 `materializeLaneStrideToContiguous` 的单个 contiguous result part 物化抽取为
`materializeLaneStrideResultPart`。辅助函数集中负责 source carrier 收集、多级
`vpack`/`vor` 合并以及最终 bitcast；外层函数只保留 dense lane-stride shape 校验、
carrier 类型计算和结果 part 枚举。该拆分对应 source-part 拼接与整体转换编排的真实
职责边界，未改变 carrier 位宽递进、尾部 part 处理或失败路径。增量合规检查结果为
`checked_files=1 errors=0 warnings=0`，`git diff --check` 通过；
`vmi_to_vpto_load_store_contiguous.pto`、`vmi_interleaved_memory_ops.pto` 和
`vmi_layout_assignment_group_store_slots1_unit_stride.pto` 均完成 lowering 回归。

本轮将 `checkSupportedExpandLoadShape` 的 runtime-mask 路径契约抽取为
`checkSupportedExpandLoadRuntimePath`。辅助函数集中负责 `!pto.ptr`、b32 结果/mask、
单 physical chunk 以及 result/passthru/mask full-chunk 校验；主函数保留 access-plan、
静态 all-active 快路径和 runtime-mask/all-active 诊断拼接。该拆分保持 expand 的
prefix/gather 语义和原有失败顺序不变。增量合规检查结果为
`checked_files=1 errors=0 warnings=0`，`git diff --check` 通过；连续 load/store
lowering 回归通过。

本轮将模板化 group-reduce lowering 的 one-block `vcgadd` 分支抽取为
`lowerOneBlock`。辅助函数独立负责 source/mask/result physical arity、统一物理类型
校验和逐 part 的 group-reduce 指令构造；主 `matchAndRewrite` 继续负责支持表查询、
lowering plan 分类以及其它 two/four-block 与 row-reduction 路径。该拆分不改变
`GroupReduceLoweringPlan::OneBlockVcgadd` 的诊断和结果替换语义。增量合规检查结果为
`checked_files=1 errors=0 warnings=0`，`git diff --check` 通过；
`vmi_group_reduce_addi_i16.pto` 完成完整 lowering 回归。

本轮将 `OneToNVMICreateGroupMaskOpPattern::matchAndRewrite` 的 dynamic 与 constant
active-lanes 路径分别抽取为 `lowerDynamicMask` 和 `lowerConstantMask`。dynamic helper
统一负责单值 active 参数、deinterleaved 到 contiguous 的中间物化及最终 layout 转换；
constant helper 统一负责 materialization 结果、mask 类型和 physical arity 收尾。主
pattern 现在仅负责结果类型转换、factor-4 block 特例和路径分派，保持原有 layout 转换
顺序、结果顺序与诊断语义不变。增量合规检查结果为 `checked_files=1 errors=0 warnings=0`，
`git diff --check` 通过；`vmi_to_vpto_create_group_mask_block8_dynamic.pto`
lowering exit=0。

本轮清理 `group_slot_load` 与 `masked_load` pattern 的入口控制流：为 layout、operand、
physical arity 和 chunk 校验补齐大括号，并将多条件失败判断命名化。该调整保持 slots=1/8
的 `vsldb` 选择、mask/passthru 语义、结果顺序和原有诊断不变。增量合规检查结果为
`checked_files=1 errors=0 warnings=0`，`git diff --check` 通过；group-slot SCF 与
连续 load/store lowering exit=0。

本轮将 `OneToNVMIMaskedLoadOpPattern` 的逐 physical chunk 发射抽取为
`lowerPhysicalParts`，集中处理 mask/passthru/result arity、物理类型、chunk offset、
`vlds` 和 `vsel`；主 pattern 仅负责 source/offset、read footprint 和结果类型准备。
保持 masked-load 的 passthru 语义、结果顺序和指令选择不变。增量合规检查结果为
`checked_files=1 errors=0 warnings=0`，`git diff --check` 通过；连续 load/store
lowering exit=0。

本轮将 `OneToNVMIGatherOpPattern` 的逐 physical chunk 发射抽取为
`lowerPhysicalParts`，集中处理 arity/type 校验、`Vgather2/Vgather2Bc` 选择以及静态
all-active 时跳过 `VselOp` 的优化；主 pattern 只保留 source/result 准备和策略判定。
保持 gather 的 passthru 语义、结果顺序和 16-bit/其它位宽指令选择不变。增量合规检查
结果为 `checked_files=1 errors=0 warnings=0`，`git diff --check` 通过；连续 load/store
lowering exit=0。

本轮为 `OneToNVMIExtFOpPattern` 引入 `ExtFPhysicalPlan`/`buildPhysicalPlan`，统一负责
source physical part 一致性、result vreg 类型和 BF16x2/F32 结果契约；主 pattern 仅
负责 packed view 规划及 lane-stride/factor 发射分派。同步将 vmull verifier 的复合物理
形状条件改为具名布尔变量。保持 extf/vmull 的支持矩阵、Vcvt part、结果 arity 和诊断
语义不变。增量合规检查结果为 `checked_files=1 errors=0 warnings=0`，`git diff --check`
通过；连续 load/store lowering exit=0。

本轮清理控制流转换 pattern 中遗留的 `G.FMT.11-CPP`：为 branch destination 缓存、
cond-branch/switch selector、execute-region/index-switch result 变化检查补齐大括号，
并将多行 arity/变化条件命名化。该调整不改变 block 映射、one-to-N operand 展开或
region 内联语义。增量合规检查结果为 `checked_files=1 errors=0 warnings=0`，
`git diff --check` 通过。

本轮继续整理 channel split/merge 的布局判定：将 source/result layout 的复合条件拆成
具名布尔变量，并为 split/merge 的 channel layout 显式标注 `VMILayoutAttr` 类型；同时
补齐这两条 pattern 的控制流大括号。该调整仅改善 verifier/lowering 的职责可读性，保持
channel 数量支持矩阵、layout 转换方向和结果顺序不变。增量合规检查结果为
`checked_files=1 errors=0 warnings=0`，`git diff --check` 通过；interleave/channel
相关 lowering exit=0。

本轮将 `OneToNVMIShuffleOpPattern` 的 lane0 splat 路径抽取为 `lowerLane0Splat`，集中
处理 source part 范围、physical vreg 类型、全真 mask 和 `VdupOp` 结果替换；主 pattern
继续按 forwarding → lane0 splat → vselr 的优先级分派。该拆分保持 shuffle 的结果顺序、
mask 语义、诊断文本和其它路径不变。增量合规检查结果为
`checked_files=1 errors=0 warnings=0`，`git diff --check` 通过；interleave/shuffle
相关 lowering exit=0。

本轮进一步将 shuffle 的零拷贝 forwarding 路径抽取为 `lowerForwarding`，集中处理源
physical part 范围检查、identity forwarding 校验和结果替换；主 pattern 继续保持
forwarding → lane0 splat → vselr 的策略优先级。该拆分减少重复控制流并保持 shuffle
语义与诊断不变。增量合规检查结果为 `checked_files=1 errors=0 warnings=0`，
`git diff --check` 通过；interleave/shuffle lowering exit=0。

本轮将 shuffle 的 `vselr` 路径抽取为 `lowerVselr`，集中处理 plan arity、source part
范围、source/result 类型、索引位宽、`VciOp` 索引向量和 `VselrOp` 发射；主 pattern 现在
只负责 forwarding、lane0 splat、vselr 三种策略的识别与分派。该拆分保持索引方向、结果
顺序、诊断文本和原有优先级不变。增量合规检查结果为 `checked_files=1 errors=0 warnings=0`，
`git diff --check` 通过；interleave/shuffle lowering exit=0。

本轮继续清理 `VMIExtF`/`VMITruncF` conversion pattern 中遗留的 `G.FMT.11-CPP` 问题：
为 result/source physical type、mask、位宽及 group-slot layout 分支补齐控制流大括号，
并将多行条件整理为具名布尔变量。该调整不改变 BF16x2 view、Vcvt part、mask、arity
及诊断语义。增量合规检查结果为 `checked_files=1 errors=0 warnings=0`，
`git diff --check` 通过。

复核 interleave 模板的 contiguous lowering 时发现 `invalidSingleChunkTypes` 条件缺少
闭合大括号，导致直接 `TargetOp` 构造语句错误地落入失败分支。已补齐控制流边界，恢复
contiguous interleave 的正常 lowering；该修复不改变支持矩阵或结果语义。增量合规检查
结果为 `checked_files=1 errors=0 warnings=0`，`git diff --check` 通过；
`vmi_interleaved_memory_ops.pto` lowering exit=0。

本轮将 `OneToNVMIInterleaveOpPattern` 的 contiguous 单 chunk 物化抽取为
`lowerContiguous`，统一处理单物理 part、mask 类型、carrier 类型和结果替换；主模板仅
负责 layout fact 判定及 lane-stride/zero-copy 路径分派。同时修复该区域以及 `extf`
入口中发现的缺失大括号控制语句。新增代码通过增量合规检查，`git diff --check` 通过，
`vmi_interleaved_memory_ops.pto` lowering exit=0。

本轮复核 `OneToNVMILoadOpPattern::matchAndRewrite` 时，将 physical chunk 安全校验
提前到 lane-stride dist 快路径之前，确保所有 load 路径都先获得有效的
`lanesPerPart`；这修正了快路径对后置变量的依赖，并保持 dist/对齐判断及 fallback
顺序不变。同时补齐 `extf` 入口剩余的控制语句大括号。增量合规检查结果为
`checked_files=1 errors=0 warnings=0`，`git diff --check` 通过；
`vmi_to_vpto_load_store_contiguous.pto` lowering exit=0。

本轮将 `OneToNVMIExtFOpPattern` 的物理发射拆分为 `lowerLaneStride` 与 `lowerFactor`：
前者负责 contiguous lane-stride 的单 part `VcvtOp`，后者负责 factor=2/4 的
`EVEN/ODD` 或 `P0..P3` 展开；主 pattern 继续负责输入/结果 physical type 契约、packed
BF16x2 view 规划和支持矩阵选择。同时补齐整数 extension 入口的控制语句大括号。该重构
保持 `Vbitcast` 结果 view、part 顺序和 arity 语义不变。增量合规检查结果为
`checked_files=1 errors=0 warnings=0`，`git diff --check` 通过；连续 load/store
lowering exit=0。

本轮继续将 two-block `deinterleaved=2` 的 `vcgadd`/combine 路径抽取为
`lowerTwoBlock`。该 helper 独立维护双物理块 arity、source/mask/result 类型一致性、
尾部 active-group mask 和两路 group-reduce 后的 combine；主 pattern 仅负责 plan 分派。
保留原有 two-block 诊断文本、结果顺序和 physical replacement 语义。增量合规检查结果
为 `checked_files=1 errors=0 warnings=0`，`git diff --check` 通过；
`vmi_group_reduce_addi_i16.pto` 完成 lowering 回归。

本轮将 four-block `deinterleaved=4` 的 `vcgadd` 树形归约抽取为
`lowerFourBlock`。辅助函数集中负责四路 physical arity/type 校验、四个 partial
`GroupReduceOpTy`、两级 `CombineOpTy` tree 以及尾部 active-group mask；主 pattern
继续只做 lowering plan 分派。原有四路 source 顺序、诊断文本和结果替换语义保持不变。
增量合规检查结果为 `checked_files=1 errors=0 warnings=0`，`git diff --check` 通过；
`vmi_group_reduce_addi_i16.pto` 完成 lowering 回归。

本轮将 full `deinterleaved=2` row-reduction 路径抽取为
`lowerFullDeinterleaved2`。辅助函数集中负责 slots=1 结果契约、物理 lane/group 与
arity 推导、row reduction 类型/mask 构造、双路 chunk 合并及每组结果回填；主 pattern
仅保留 plan 分派。原有 chunk 顺序、uniform physical type 诊断和结果替换语义保持不变。
增量合规检查结果为 `checked_files=1 errors=0 warnings=0`，`git diff --check` 通过；
`vmi_group_reduce_addi_i16.pto` 完成 lowering 回归。

本轮将 contiguous `Vcadd` rows 路径抽取为 `lowerContiguousRows`。辅助函数独立负责
contiguous group chunk 计算、结果 arity/layout 契约、row-reduction 类型与 mask 构造、
每组累积以及 slots=1/多 chunk 结果回填；主 pattern 现在仅执行 plan 分派。保持原有
结果广播、chunk 顺序和失败诊断语义。增量合规检查结果为
`checked_files=1 errors=0 warnings=0`，`git diff --check` 通过；
`vmi_group_reduce_addi_i16.pto` 完成 lowering 回归。

本轮将模板化 histogram lowering 的单 physical chunk 处理抽取为
`lowerHistogramChunk`。辅助函数集中负责 b8 mask 校验、尾部 active-lane mask 修正、
以及各 histogram half 的 `Dhistv2Op`/`Chistv2Op` 发射；外层函数保留 accumulator
契约、source lane 计算和最终结果替换。该拆分不改变 bin 常量、chunk 顺序或 tail mask
语义。增量合规检查结果为 `checked_files=1 errors=0 warnings=0`，
`git diff --check` 通过；`vmi_layout_assignment_f32_f8_store_reduce.pto` 完成
lowering 回归。

本轮将 `checkSupportedGatherShape` 的结果/索引元素宽度与 mask granularity 契约抽取
为 `checkGatherElementContract`。该 helper 只负责 `ui8/ui16/i8/i16/f16/bf16` 与
`b16/b32` 的组合判定及统一诊断；主函数保留 layout、pointer 和 physical arity/full
chunk 检查。这样避免把 gather 的类型能力矩阵与物理承载检查耦合，支持范围和失败
顺序保持不变。增量合规检查结果为 `checked_files=1 errors=0 warnings=0`，
`git diff --check` 通过；gather 与 group-store 代表性 lowering 命令可正常执行。

本轮将模板化 `OneToNVMIReduceMinMaxOpPattern` 的实际归约流程抽取为
`lowerReduction`。辅助函数负责等价 masked-part 合并快路径、逐 physical chunk 的
`ChunkReduceOp`/`CombineOp` 累积以及结果替换；主 pattern 保留 source/mask/result
物理 arity、vreg/mask 类型和统一性校验。这样将输入契约与归约算法分离，min/max
支持范围和诊断语义不变。增量合规检查结果为 `checked_files=1 errors=0 warnings=0`，
`git diff --check` 通过；`vmi_group_reduce_addi_i16.pto` 与
`vmi_layout_assignment_reduce_addf.pto` 均完成 lowering 回归。

本轮将 `OneToNVMIExtFOpPattern` 中 BF16x2 结果视图与 `VcvtOp` 发射逻辑抽取为
具名 helper `createVcvtResult`，消除默认捕获 lambda，并集中处理 native BF16 转换后
的物理 `VbitcastOp`。主 pattern 保留源/结果 physical layout、factor=2/4 分派和结果
顺序，未改变 extf 支持范围。增量合规检查结果为
`checked_files=1 errors=0 warnings=0`，`git diff --check` 通过；连续 load/store
与 reduction 代表性 lowering 均通过。

本轮对称地将 `OneToNVMITruncFOpPattern` 的 BF16x2 source view 逻辑抽取为
`makeVcvtSourceView`，去除默认捕获 lambda。helper 负责识别可直接复用的 BF16
pairing bitcast，并在必要时构造 native BF16 `VbitcastOp`；truncf 的 group-slot、
same-width 和 factor 分派保持不变。增量合规检查结果为
`checked_files=1 errors=0 warnings=0`，`git diff --check` 通过；连续 load/store 与
reduction 代表性 lowering 均通过。

本轮将 `OneToNVMICreateGroupMaskOpPattern` 的 block-deinterleaved factor=4
“先 contiguous 物化、再布局转换”路径抽取为 `lowerFactor4Block`。helper 独立负责
常量/动态 active-lane 物化、contiguous physical arity 校验和最终 mask layout
conversion；主 pattern 保留普通 constant/dynamic 路径分派。该拆分保持 mask part
顺序、失败诊断和 factor=4 语义不变。增量合规检查结果为
`checked_files=1 errors=0 warnings=0`，`git diff --check` 通过；
`vmi_layout_assignment_create_group_mask_s32_dynamic.pto` 与连续 load/store
代表性 lowering 均通过。

本轮将 `materializeMaskLaneStridePack` 的单个结果 chunk 组装抽取为
`materializeMaskLaneStridePackChunk`，去除局部默认捕获 lambda。helper 只负责
lane_stride=2/4 下 source mask 的低/高 half `ppack` 与 `por` 合并；外层继续负责
source/result arity、mask 类型和 chunk 枚举。增量合规检查结果为
`checked_files=1 errors=0 warnings=0`，`git diff --check` 通过；mask granularity 与
group-store 代表性 lowering 均通过。

本轮将 `OneToNVMILoadOpPattern` 的 contiguous 物化阶段抽取为 `lowerContiguous`。
helper 独立负责 NORM 对齐 `vlds` 与非对齐 `vldas`/多轮 `vldus` 的选择、align/base
状态链更新、contiguous part 收集及后续 data-layout conversion；主 pattern 保留
lane-stride/deinterleaved 快路径和地址安全证明。该拆分保持非对齐访存的 align 寄存器
逐轮传递语义及结果顺序不变。增量合规检查结果为 `checked_files=1 errors=0 warnings=0`，
`git diff --check` 通过；连续 load/store 与 interleave memory lowering 回归通过。

本轮将 load lowering 中 deinterleaved factor=2/4 的 direct `vldsx2` 能力判断抽取为
`lowerDirectDeinterleaved`。helper 统一处理布局 factor、DINTLV dist token、地址合法性
和 physical arity，再分别调用 factor=2/4 物化 helper；主 pattern 保留 lane-stride
优先级、full/safe-read 证明和 contiguous fallback。非对齐路径选择及结果顺序不变。
增量合规检查结果为 `checked_files=1 errors=0 warnings=0`，`git diff --check` 通过；
连续 load/store 与 interleave memory lowering 回归通过。

本轮将 load 的 dense lane-stride physical part 构造抽取为
`materializeLaneStrideParts`。helper 负责逐 part 的 semantic offset 累加、active-lane
计算和 `vlds` 发射；`lowerLaneStride` 只负责结果替换。该拆分保持 dist token、尾部
active lane 和地址步长语义不变。增量合规检查结果为
`checked_files=1 errors=0 warnings=0`，`git diff --check` 通过；
`vmi_to_vpto_load_store_contiguous.pto` 完成 lowering 回归。

本轮将 `OneToNVMIGroupLoadOpPattern` 的 contiguous unit-stride direct `vlds` 路径
抽取为 `lowerContiguousUnitStride`。helper 负责 physical lane 计算、逐 chunk offset
和 vlds 发射及结果替换；主 pattern 保留 contiguous 布局与 row-stride 识别以及其它
group-load fallback。该拆分不改变 group size/row stride 约束或结果顺序。增量合规检查
结果为 `checked_files=1 errors=0 warnings=0`，`git diff --check` 通过；group-store
unit-stride 与连续 load/store 代表性 lowering 均通过。

本轮将 `OneToNVMITruncIOpPattern` 中 s32→s8 alias 恢复与结果替换的默认捕获 lambda
抽取为具名 `finalizeResults` helper。helper 集中负责 alias 场景的结果 `VbitcastOp`
恢复及统一 physical replacement；trunci 的 group-slot、dense lane-stride 和 factor
2/4 合并路径保持不变。增量合规检查结果为 `checked_files=1 errors=0 warnings=0`，
`git diff --check` 通过；连续 load/store 与 group-store 代表性 lowering 均通过。

本轮将 `fptosi`/`fptoui` 的 narrow physical conversion（按 Even/Odd 或 packed part
执行 `VcvtOp`，再以 `VorOp` 合并）抽取为共享 helper `lowerNarrowFpToInt`。helper
统一维护 source/result mask、source-factor 与 lane-stride 对应的 part 选择、物理 chunk
索引和最终结果替换；两个 pattern 仅保留各自的类型 contract、factor 推导和诊断文本。
同时把 interleave 中重复的 contiguous/deinterleaved factor 判断提升为具名函数
`getElementDeinterleaveFactor`，避免局部 lambda 重复定义。两处重构均保持原有 part 顺序、
结果 arity 和 signed/unsigned 支持矩阵不变。增量合规检查结果为
`checked_files=1 errors=0 warnings=0`，`git diff --check` 通过；本地 `ninja -C build
pto-test-opt` 受现有 CMake 外部依赖尝试创建 `/cann-cmake` 的权限错误阻断，未报告源码
编译错误。

随后补充了该共享布局 factor helper 的前置声明，确保其在 interleave pattern 使用前
满足 C++ 声明顺序要求；不改变任何 lowering 行为。增量合规检查仍为
`checked_files=1 errors=0 warnings=0`，`git diff --check` 通过。

本轮将 `OneToNVMIGroupStoreOpPattern` 的 deinterleaved=2 `vstsx2` 发射路径抽取为
`lowerDeinterleaved2GroupStore`。helper 独立负责 chunk-shape、INTLV dist、双路物理
arity/type、all-true mask 和 group/chunk offset 计算；主 `matchAndRewrite` 只保留布局
事实判定及路径分派，普通 contiguous store 逻辑不变。同时将本轮触及的 lane-stride
条件整理为具名布尔变量以满足格式规则。增量合规检查结果为
`checked_files=1 errors=0 warnings=0`，`git diff --check` 通过；
`vmi_layout_assignment_group_store_slots1_unit_stride.pto` lowering exit=0，输出仍含
预期 `vsts`。

本轮将 `materializeAdjacentMaskGranularityConversion` 的单 layout-part 转换抽取为
`materializeAdjacentMaskGranularityPart`。该 helper 负责计算 source/result chunks 并
选择 widening 或 narrowing 的 chunk materializer；外层函数只负责 physical factor 的
part 遍历、source offset 累加和最终 arity 校验。这样直接降低了相邻 mask granularity
转换的职责复杂度，保持 ppack/por 与 part 顺序不变。增量合规检查结果为
`checked_files=1 errors=0 warnings=0`，`git diff --check` 通过；
`vmi_to_vpto_ensure_mask_granularity.pto` lowering exit=0，输出仍包含预期的
`pdintlv/ppack/pintlv` 序列。

本轮将 `materializeStagingDeintToContiguousMaskLayout` 中 factor=2 的内嵌 lambda
抽取为 `materializeFactor2DeintToContiguousGroup`，使 factor=2 与已有 factor=4
都通过独立的 group materializer 负责 predicate interleave 和 result type 选择；外层
函数只负责分组迭代、结果容量和 arity 校验。该拆分保持 mask part 顺序与尾部结果裁剪
语义不变。增量合规检查结果为 `checked_files=1 errors=0 warnings=0`，
`git diff --check` 通过；`vmi_to_vpto_ensure_mask_granularity.pto` lowering exit=0。

本轮将 `OneToNVMIGroupStoreOpPattern` 的普通 contiguous full-chunk `vsts` 发射路径
抽取为 `lowerContiguousGroupStore`。helper 独立负责 contiguous chunk shape、physical
arity、all-true mask 及 group/chunk offset；主 pattern 保留 deinterleaved=2 与
contiguous 路径的互斥判定和分派。该拆分不改变 chunk 顺序、offset 计算或 store mask
语义。增量合规检查结果为 `checked_files=1 errors=0 warnings=0`，`git diff --check`
通过；`vmi_layout_assignment_group_store_slots1_unit_stride.pto` 与
`vmi_group_reduce_addi_i16.pto` lowering 均 exit=0。

本轮将 `OneToNVMIGroupStoreOpPattern` 的 one-block `vsstb` 路径抽取为
`lowerOneBlockGroupStore`。helper 独立负责 one-block plan、physical arity、block/repeat
stride、contiguous mask、part offset 和 `vsstb` 发射；主 pattern 仅保留 block class
分派及 deinterleaved/contiguous fallback。该拆分保持计划参数、结果顺序和诊断语义不变。
增量合规检查结果为 `checked_files=1 errors=0 warnings=0`，`git diff --check` 通过；
group-store 与 reduction 代表性 lowering 均成功。

本轮将 `OneToNVMIGroupLoadOpPattern` 的普通 contiguous full-chunk `vlds` fallback
抽取为 `lowerContiguousChunks`。helper 独立负责 group size、full physical chunk、
result arity、group/chunk offset 和逐 chunk `vlds`；主 pattern 继续保留 unit-stride
快路径与 block-deinterleaved 专用路径。该拆分保持 load 顺序、offset 语义和物理结果
替换不变。增量合规检查结果为 `checked_files=1 errors=0 warnings=0`，`git diff --check`
通过；连续 load/store 与 group-store 代表性 lowering 均成功。

本轮将 `slots=8` 普通 contiguous group-store（非 packed-byte、非 lane-stride）路径
抽取为 `lowerSlots8Contiguous`。helper 独立处理每个 slot block 的地址合法性判定、
非对齐 stateful stream fallback、active-lane mask 和对齐 `vsts` 发射；主 pattern 继续
负责 packed-byte/lane-stride 路径分派。同步将本轮触及的 stride-load 控制条件整理为
具名布尔变量，保持格式规则一致。增量合规检查结果为
`checked_files=1 errors=0 warnings=0`，`git diff --check` 通过；
`vmi_layout_assignment_group_store_slots1_unit_stride.pto` lowering exit=0。

本轮将 `OneToNVMITruncIOpPattern` 的 dense contiguous lane-stride narrowing 路径抽取为
`lowerDenseLaneStrideTrunc`。helper 负责 source mask、`EVEN/P0` part 选择、逐 physical
chunk 的 `VcvtOp` 以及 signed alias 结果收尾；主 pattern 保留布局识别、NOSAT carrier
快路径和后续多 part 合并。该拆分保持 SAT/NOSAT 语义、结果顺序与 alias bitcast 行为不变。
增量合规检查结果为 `checked_files=1 errors=0 warnings=0`，`git diff --check` 通过；
连续 group-store 与 load/store 代表性 lowering 均成功。

本轮将 `OneToNVMITruncIOpPattern` 的 group-slots 专用 lowering 抽取为
`lowerGroupSlotTrunc`。helper 统一处理 slots=1/8 的支持矩阵、active-slot mask、
packed carrier、lane-stride carrier、physical type 校验及 `VcvtOp` 发射；主 pattern
继续负责非 group-slots 的 dense lane-stride、alias 和 factor 合并路径。该拆分保持
group-slot 的 part 选择、结果 carrier 和诊断语义不变。增量合规检查结果为
`checked_files=1 errors=0 warnings=0`，`git diff --check` 通过；连续 group-store 与
load/store lowering 回归通过。

本轮将 `OneToNVMIExtIOpPattern` 的非 group-slot 整数 extension 物理发射抽取为
`lowerPhysicalExtension`。该 helper 统一处理 contiguous lane-stride 的单 part 快路径、
按 2/4 倍 factor 的 `EVEN/ODD` 或 `P0..P3` 展开、all-true mask 构造和结果替换；
group-slot 的 `Vsunpack/Vzunpack` 与 `Vcvt` 路径仍由主模式独立负责。该拆分保持物理
结果顺序、mask 语义和诊断文本不变。增量合规检查结果为
`checked_files=1 errors=0 warnings=0`，`git diff --check` 通过；完整构建和完整静态
报告仍未宣称清零。

本轮进一步将 `materializeSimpleDataLayoutConversion` 中两个独立的转换类别拆开：
`materializeGroupSlotLaneStrideLayout` 负责 slots=8 的 group-slot lane-stride 物化，
`materializeBlockLayoutForwarding` 负责 block-deinterleaved 与 contiguous 之间的
cast-input/identity forwarding。simple conversion 主函数现在只执行类别分派，未改变
转换优先级、physical part arity 或失败诊断。增量合规检查结果为
`checked_files=1 errors=0 warnings=0`，`git diff --check` 通过；mask/layout 相关
`pto-test-opt` lowering exit=0。

本轮将 `materializeDataLayoutViaContiguous` 的中间布局识别与中间 part 数量计算抽取
为 `getDataLayoutIntermediatePlan`，使用 `DataLayoutIntermediatePlan` 显式表达 contiguous
或 deinterleaved 中间态及其 arity。实际递归物化仍由原 conversion dispatcher 执行，
因此不改变转换优先级、padding/part 顺序或中间类型。增量合规检查结果为
`checked_files=1 errors=0 warnings=0`，`git diff --check` 通过。

本轮将 `materializeMaskLayoutConversion` 的 identity forwarding 分支抽取为
`materializeIdentityMaskLayout`。该 helper 只负责相同 layout 的 physical part 合法性
校验和无变换转发；`deinterleaved=2` 及 lane-stride 的实际 mask 物化仍由各自 helper
负责。增量合规检查结果为 `checked_files=1 errors=0 warnings=0`，`git diff --check`
通过；`vmi_to_vpto_ensure_mask_granularity.pto` lowering exit=0，输出仍包含预期的
`pdintlv/ppack/pintlv` 序列。

本轮将 `lowerGroupSlotLoadParts` 的布局、指针、结果 arity 和 slots 支持矩阵校验抽取
为 `getGroupSlotLoadSlots`。该 helper 只建立合法的 slots plan，slots=1/8 的具体地址
物化与 `vsldb` 发射仍由原有专用 helper 负责；失败诊断和分派顺序保持不变。增量合规
检查结果为 `checked_files=1 errors=0 warnings=0`，`git diff --check` 通过。完整构建仍
需在修复 `/cann-cmake` 外部依赖权限后补跑。
本轮将 `materializeAdjacentMaskGranularityConversion` 的方向、物理 arity、layout factor
和结果 mask 类型计算抽取为 `MaskGranularityConversionPlan` 与
`buildMaskGranularityConversionPlan`。转换函数现在只负责按 layout part 调用 widening/
narrowing materializer 及最终 arity 校验；相邻粒度的语义和错误诊断保持不变。增量合规
检查结果为 `checked_files=1 errors=0 warnings=0`，`git diff --check` 通过；
`vmi_to_vpto_ensure_mask_granularity.pto` lowering exit=0，仍生成预期的
`pdintlv/ppack/pintlv` 序列。
本轮将 `materializeStagingContiguousToDeintMaskLayout` 中 factor=2 的内嵌 lambda
抽取为 `materializeFactor2ContiguousToDeintGroup`，与 factor=4 的现有 helper 形成对称
的 group 物化边界。外层函数继续负责 source padding、part 收集和最终 arity 校验，
保持 `dintlv` 操作顺序与 all-false 补齐语义不变。增量合规检查结果为
`checked_files=1 errors=0 warnings=0`，`git diff --check` 通过；
`vmi_to_vpto_ensure_mask_granularity.pto` lowering exit=0。
本轮将 `materializeMaskGranularityCastConversion` 的 physical carrier 解析和基础契约
校验抽取为 `MaskGranularityCastPlan`/`buildMaskGranularityCastPlan`。执行函数现在只
负责 identity、同 layout 粒度转换，以及“先统一粒度、再转换 layout”的顺序；没有改变
mask carrier、物理 arity 或 fallback 语义。增量合规检查结果为
`checked_files=1 errors=0 warnings=0`，`git diff --check` 通过；
`vmi_to_vpto_ensure_mask_granularity.pto` lowering exit=0。
本轮将 `checkSupportedGroupBroadcastShape` 的基础契约校验抽取为
`GroupBroadcastShapePlan`/`buildGroupBroadcastShapePlan`。该 plan 集中保存 source/result
layout、num_groups、physical lanes、derived group size 和 result factor；主校验函数现在
只负责 full-chunk 约束及 block/deinterleaved 小组形状判定。BRC/E2B/group-slot 支持
矩阵、诊断文本和成功条件保持不变。增量合规检查结果为
`checked_files=1 errors=0 warnings=0`，`git diff --check` 通过；
`vmi_layout_assignment_group_broadcast_load_e2b_b16.pto` lowering exit=0。
本轮将 `checkSupportedVmullShape` 的公共输入契约抽取为 `VmullShapePlan` 与
`buildVmullShapePlan`，集中验证四个数据端口、mask layout/granularity、逻辑 lane 数和
physical arity；主函数仅保留 vmull 特有的 64xi32/ui32 carrier 与 b32 mask 校验。
该拆分保持原有支持矩阵及诊断语义不变。增量合规检查结果为
`checked_files=1 errors=0 warnings=0`，`git diff --check` 通过；
`vmi_layout_assignment_group_broadcast_load_e2b_b16.pto` lowering exit=0。
本轮将 `checkSupportedCompressShape` 与 `checkSupportedCompressStoreShape` 的公共
contiguous value/mask、full physical chunk 和单 chunk arity 契约抽取为
`CompressPhysicalShapePlan`/`buildCompressPhysicalShapePlan`。两个 verifier 只保留各自
的 result 或 destination 专属约束；支持矩阵和诊断语义保持不变。增量合规检查结果为
`checked_files=1 errors=0 warnings=0`，`git diff --check` 通过；已有 mask/layout
lowering 回归 exit=0。
本轮将动态 `create_group_mask` 的全局契约校验抽取为
`DynamicGroupMaskPlan`/`buildDynamicGroupMaskPlan`，集中表达 layout、factor、block
元素数、physical lanes 和 arity。`materializeDynamicGroupMaskForType` 现在只负责
active lane clamp 与按 plan 调用 chunk materializer；单 chunk 的 index/比较/padding
逻辑保持不变。增量合规检查结果为 `checked_files=1 errors=0 warnings=0`，
`git diff --check` 通过；`vmi_to_vpto_create_group_mask_block8_dynamic.pto` 完整
lowering 成功。
本轮将普通 reduce verifier 的公共 source/mask/result layout、full-chunk 和 physical
arity 契约抽取为 `ReducePhysicalShapePlan`/`buildReducePhysicalShapePlan`。模板入口
现在仅处理 `reassoc` 专属约束并调用公共 plan，保持整数/浮点 reduce 的支持矩阵、结果
arity 和诊断语义不变。增量合规检查结果为 `checked_files=1 errors=0 warnings=0`，
`git diff --check` 通过。
本轮将 `checkSupportedActivePrefixIndexShape` 的 shape 契约抽取为
`ActivePrefixIndexShapePlan`/`buildActivePrefixIndexShapePlan`，集中保存 mask/result
类型并完成 contiguous layout、full physical chunks 及 single-chunk carry 约束；公开
checker 仅负责 plan 构建结果转换。该拆分保持 active-prefix 的跨 chunk 限制与诊断
语义不变。增量合规检查结果为 `checked_files=1 errors=0 warnings=0`，
`git diff --check` 通过。
本轮将 `checkSupportedFPToSIShape` 与 `checkSupportedFPToUIShape` 的重复 verifier
逻辑抽取为模板 helper `checkSupportedFPToIntShape`。调用点仅提供 signed/unsigned
contract lookup 和 conversion 名称；同宽 layout/arity 及非同宽 cast-layout framework
检查统一实现，保留原有诊断前缀和支持矩阵。增量合规检查结果为
`checked_files=1 errors=0 warnings=0`，`git diff --check` 通过。
整数 cast lit 在测试自身的 VMI pack/unpack pipeline invariant 处提前失败，未进入
本轮 verifier/lowering 路径。
本轮将 FP→整数 verifier 的同宽 layout/physical arity 检查进一步抽取为
`checkSameWidthConversionArity`，并由共享的 `checkSupportedFPToIntShape` 调用；
signed/unsigned contract 与非同宽 cast-layout 检查保持原有独立语义。这样消除了
conversion verifier 中重复的同宽契约代码。增量合规检查结果为
`checked_files=1 errors=0 warnings=0`，`git diff --check` 通过；mask/layout lowering
代表性 case exit=0。
本轮将 `checkSupportedSIToFPShape` 的同宽 layout/physical arity 校验改为复用
`checkSameWidthConversionArity`，整数→浮点 verifier 仍保留 `si32→f32` 与 `si8→f16`
的元素类型/宽度契约以及非同宽 cast-layout 检查。增量合规检查结果为
`checked_files=1 errors=0 warnings=0`，`git diff --check` 通过。
尝试使用 `vmi_vcvt_s8_to_f16_lower.pto` 回归时，输入在既有 VMI verifier 处因
`sitofp` 的 `si8` source element contract 提前失败，未进入本轮共享 helper 路径，
因此不将其记为 lowering 通过。
本轮将 `channel_split` 与 `channel_merge` 的共同 channel 数量和 expected
deinterleaved layout 判定抽取为 `ChannelShapePlan`/`buildChannelShapePlan`；两条
checker 继续各自负责输入/输出 layout、physical arity 汇总及方向性诊断。2/4 channel
支持范围和结果 arity 语义保持不变。增量合规检查结果为
`checked_files=1 errors=0 warnings=0`，`git diff --check` 通过。
本轮继续整理 channel verifier：新增 `getContiguousChannelInputArity`，统一
`channel_merge` 输入 vreg、contiguous layout 和 physical arity 汇总；channel split/merge
仍分别负责结果/源布局及方向性约束。该拆分保持 channel 数量、expected layout 和
source/result arity 语义不变。增量合规检查结果为 `checked_files=1 errors=0 warnings=0`，
`git diff --check` 通过。
本轮移除了文件级 `-Woverloaded-virtual` warning suppression（原先从第一个 OneToN
pattern 覆盖至 pattern 注册结束）。这些 pattern 已通过显式的
`using OneToNOpConversionPattern<...>::OneToNOpConversionPattern` 构造函数继承和
`override` 实现，当前增量合规检查未报告新的 suppression 或格式错误；实际编译器
warning 仍需在 CMake 外部依赖权限修复后通过完整构建确认。

本轮复核 `OneToNVMITruncIOpPattern::matchAndRewrite` 时发现 source/result physical
part 校验和宽度/布局分支中存在多处缺失闭合大括号，已补齐这些控制流边界；该修复
避免后续 narrowing 逻辑错误地落入前置失败条件，保持原有 group-slot、lane-stride、
NOSAT alias 及 factor=2/4 lowering 语义不变。增量合规检查结果为
`checked_files=1 errors=0 warnings=0`，`git diff --check` 通过。

本轮继续复核 truncation 与 verifier 代码时，补齐 `trunci` physical type 检查遗漏的
控制流边界，并将 bitcast verifier 的元素类型比较改为具名布尔条件，避免静态规则将
多行条件误判为无大括号控制语句。该修复不改变支持矩阵、诊断或 lowering 结果。增量
合规检查结果为 `checked_files=1 errors=0 warnings=0`，`git diff --check` 通过。

本轮补齐 `OneToNVMIFPToSIOpPattern` 与 `OneToNVMIFPToUIOpPattern` 入口的 conversion
contract、source/result physical part 校验大括号，覆盖 `fptosi/fptoui` 的同宽、拓宽和
窄化分支。该调整仅修复控制流边界，保持 `VcvtOp` part 选择、mask、arity 和诊断语义
不变。增量合规检查结果为 `checked_files=1 errors=0 warnings=0`，`git diff --check`
通过。

本轮将 `computeSafeStatefulReadProof` 中的常量/循环 offset 范围推导抽取为
`getStatefulOffsetRange`，使用 `VMIStatefulOffsetRange` 显式传递最小/最大 offset；
主体继续负责元素宽度、地址余数、footprint 和 byte envelope 安全证明。该拆分保留
有限循环范围分析、溢出检查和失败原因，不改变 dist/stateful read 的安全判定。
增量合规检查结果为 `checked_files=1 errors=0 warnings=0`，`git diff --check` 通过；
连续与交错 memory lowering case 均 exit=0。

本轮继续收口 FP→整数窄化/拓宽分支：为 `fptosi/fptoui` 的 mask、lane-stride、source
arity 和 widen arity 检查补齐大括号，并将多行条件整理为具名布尔变量；同时将
`exti` 的 group-slot layout 判断命名化，避免静态检查器误报。所有调整仅影响控制流
表达和可读性，不改变 conversion 的 part、mask、arity 或诊断语义。增量合规检查结果为
`checked_files=1 errors=0 warnings=0`，`git diff --check` 通过。

本轮继续清理 bitcast/channel/shuffle 及 verifier 相关的 `G.FMT.11-CPP` 问题：为
physical arity、channel 数量、layout、转换结果和错误回调中的控制语句补齐大括号，
并将多行条件改为具名布尔变量。调整不改变 bitcast、channel split/merge、shuffle 或
compress 的支持矩阵与结果语义。增量合规检查结果为 `checked_files=1 errors=0 warnings=0`，
`git diff --check` 通过。

本轮为 `OneToNVMILoadOpPattern` 引入 `LoadPhysicalPlan`，由 `buildPhysicalPlan` 统一
完成 source/offset 归一化、result/contiguous physical type 计算、read footprint 校验
和 `lanesPerPart` 推导。主 lowering 仅负责 lane-stride dist、direct deinterleaved 与
contiguous fallback 的路径选择，避免重复准备逻辑并保持对齐/非对齐访存语义不变。同步
将该 helper 的多条件失败判断命名化。增量合规检查结果为
`checked_files=1 errors=0 warnings=0`，`git diff --check` 通过；
`vmi_to_vpto_load_store_contiguous.pto` lowering exit=0。

本轮在 `OneToNVMIGroupLoadOpPattern` 中新增 `getResultTypes`，统一 group-load 三条
路径（block-deinterleaved、unit-stride contiguous、普通 contiguous）对 one-to-N result
physical types 的转换与失败处理。主分派不再重复调用 type converter，保持各路径的
layout 判定、地址计算、结果顺序和指令选择不变。增量合规检查结果为
`checked_files=1 errors=0 warnings=0`，`git diff --check` 通过；连续 load/store
lowering exit=0。

本轮进一步将 group-load 的 contiguous 路径抽取为 `lowerContiguousPath`，集中完成
group size/row stride 判定、result physical type 准备，以及 unit-stride 与普通 chunk
lowering 的选择；主 pattern 只保留 block-deinterleaved 特例和 contiguous fallback
分派。保持地址计算、对齐语义、结果顺序和指令选择不变。增量合规检查结果为
`checked_files=1 errors=0 warnings=0`，`git diff --check` 通过；连续 load/store
lowering exit=0。

本轮将 `OneToNVMIDeinterleaveLoadOpPattern` 的对齐 `vldsx2` 发射抽取为 `lowerDirect`，
集中负责 low/high physical type、chunk offset、结果收集和替换；非对齐路径仍由
`lowerUnaligned` 独立管理 `vldas`/`vldus` 的 align/base 状态。主 pattern 只负责
operands、lane 数、dist 合法性及对齐与非对齐路径选择，保持 stream 与 dist 语义不变。
增量合规检查结果为 `checked_files=1 errors=0 warnings=0`，`git diff --check` 通过；
interleave memory lowering exit=0。

本轮为 `OneToNVMIStoreOpPattern` 新增 `getContiguousStoreTypes`，统一 contiguous
fallback 的 physical type/footprint 计算，主 store lowering 继续独立处理 lane-stride
dist、deinterleaved `vstsx2`、对齐 `vsts` 与非对齐 stateful stream。同步将 packed
float truncation 的复合条件命名化并补齐大括号。保持 store 地址、mask、stream state
和 dist 语义不变。增量合规检查结果为 `checked_files=1 errors=0 warnings=0`，
`git diff --check` 通过；连续 load/store lowering exit=0。

本轮继续拆分 `OneToNVMIStoreOpPattern` 的 contiguous fallback：新增
`emitAlignedContiguousStoreParts` 与 `collectUnalignedStoreValues`，分别负责对齐
`vsts`（含尾 chunk mask）和非对齐 stateful stream 的 physical part 收集；外层
`lowerContiguousStoreParts` 仅负责地址合法性分派、stream base 物化及最终
`emitStatefulStoreStream` 调用。非对齐路径仍以单一 `align/base` 状态贯穿全部
`vstus`，最后发射一个 `vstas`，没有退化为独立单轮访存。增量合规检查结果为
`checked_files=1 errors=0 warnings=0`，`git diff --check` 通过；连续、交错和
group-slot 相关 lowering case 均 exit=0。

本轮将 `OneToNVMIGroupStoreOpPattern` 的 slots=8 packed-byte 分支抽取为
`lowerPackedByteSlots8`，集中处理 uniform physical part 校验、slot selector、每
32 个 group 的向量合并，以及 `PK4_B32` 对齐直写或 packed-byte stateful stream
选择。顶层 pattern 只保留 slots=8 的 arity/row-stride 校验和后端分派；没有改变
`vselr/vsel` 拼接顺序、尾部 mask、PK4 条件或 stream 的 align/base 生命周期。
增量合规检查结果为 `checked_files=1 errors=0 warnings=0`，`git diff --check` 通过；
packed-byte、slots=1 和 group-store lowering case 均 exit=0。

本轮将 `lowerSlots8Contiguous` 的对齐 physical-part 发射抽取为
`emitAlignedSlots8Contiguous`。前者继续负责所有 chunk 的 direct-address 判定和
非对齐 stateful stream 的 advances 计算，后者只负责尾部 active-group mask 与
逐块 `vsts`；因此不会把 stream 拆成独立访存，也不改变 chunk 顺序。增量合规检查
结果为 `checked_files=1 errors=0 warnings=0`，`git diff --check` 通过；slots=1
对齐相关 lowering case exit=0。

本轮将 `OneToNVMIGroupStoreOpPattern` 的 compact-small 分支抽取为
`lowerCompactSmallGroupStore`，把 compact layout 物化、对齐 `NORM` store 和非对齐
单 stream 发射封装为独立职责；主 pattern 仅保留 scalar/compact/slots/普通布局的
顶层分派。对齐判断、prefix mask、`align/base` 状态和诊断文本保持不变。增量合规
检查结果为 `checked_files=1 errors=0 warnings=0`，`git diff --check` 通过；packed-byte、
slots=1 及 group-store lowering case 均 exit=0。
本轮为 `computeGroupMaskMaterializationForType` 引入
`GroupMaskMaterializationPlan` 与 `buildGroupMaskMaterializationPlan`，集中表达
active constant、mask layout/granularity、physical lanes、group shape 和 clamp 后的
active 元素数；原函数仅调用 plan 并生成逻辑 lane predicate。该拆分不改变 mask
chunk 顺序、padding 处理或失败诊断。增量合规检查结果为 `checked_files=1 errors=0
warnings=0`，`git diff --check` 通过；动态 group-mask lowering case exit=0。

本轮将 `verifySupportedVMIMemoryLoadOp` 中的 deinterleave、stride、group、group-slot
和 group-broadcast shape verifier 分派抽取为 `verifySupportedVMIStructuredLoadOp`。
基础 `load` 与 masked/gather/expand advanced load 的处理顺序保持不变；各结构化
load 的 verifier、诊断文本和返回状态均原样保留。增量合规检查结果为
`checked_files=1 errors=0 warnings=0`，`git diff --check` 通过；连续、交错和
group-slot lowering case 均 exit=0。

本轮将 `verifySupportedVMIMemoryStoreOp` 中的 interleave、group、masked、stride 和
scatter shape 校验抽取为 `verifySupportedVMIStructuredStoreOp`；普通 `store` 保留在
入口处理，随后统一调用结构化 store 分派。该调整只改变 verifier 的职责边界，保持
所有支持条件、诊断文本和返回状态不变。增量合规检查结果为
`checked_files=1 errors=0 warnings=0`，`git diff --check` 通过；interleave memory
lowering case exit=0。
本轮将 `verifySupportedVMIReductionOp` 按语义拆为
`verifySupportedVMINormalReductionOp` 与 `verifySupportedVMIGroupReductionOp`，分别
处理普通 reduce 和 group-reduce 的 shape contract/诊断，外层仅做类别分派。原有
reassoc 要求、操作支持矩阵和错误文本保持不变。增量合规检查结果为
`checked_files=1 errors=0 warnings=0`，`git diff --check` 通过；group-reduce
lowering case exit=0。两个普通 reduce case 在既有 VMI pack/unpack pipeline invariant
处提前失败，未进入本轮 verifier 路径。

本轮将 `verifySupportedVMISpecialOp` 按算子契约拆为
`verifySupportedVMIAddCarryOp`、`verifySupportedVMIMultiplyLongOp` 和
`verifySupportedVMISpecialUnaryOp`，分别负责 addc/addcs、vmull 以及 relu/vselr；
外层只进行三类分派。各 shape checker、支持矩阵和诊断文本保持不变。增量合规检查
结果为 `checked_files=1 errors=0 warnings=0`，`git diff --check` 通过；vmull contiguous/
deinterleaved 和 vselr lowering case 均 exit=0。
本轮将 `verifySupportedVMIMiscOp` 按职责拆分为基础 scalar misc、histogram/group
broadcast 和 compression/active-prefix 三组 verifier helper，外层只负责顺序分派。
同时修复本次触及的三个 conversion-shape helper 控制流大括号。支持矩阵、诊断和
检查顺序保持不变。增量合规检查结果为 `checked_files=1 errors=0 warnings=0`，
`git diff --check` 通过；compress/active-prefix 样例在既有 VMI pack/unpack pipeline
invariant 处提前失败，未进入本轮 verifier 路径。

本轮将 `checkSupportedGroupStoreShape` 的 compact-small 支持判定抽取为
`checkSupportedCompactSmallGroupStoreShape`，使 compact layout 的 pointer/memory
proof 与 group-slots、one-block、deinterleaved 检查分离。同步补齐本轮触及位置的
控制流大括号，并将复杂条件命名化以避免静态检查误报。支持矩阵、诊断和判断顺序
保持不变。增量合规检查结果为 `checked_files=1 errors=0 warnings=0`，`git diff --check`
通过。
# 本轮 layout staging 整改

本轮将 `materializeStagingContiguousToDeintMaskLayout` 的 source group 构造、缺失
part 的 all-false 补齐以及 factor=2/4 单组物化抽取为
`materializeContiguousToDeintMaskGroup`。外层函数只负责分组循环、part 收集和最终
arity 校验；`dintlv` 发射顺序与 mask padding 语义保持不变。增量合规检查结果为
`checked_files=1 errors=0 warnings=0`，`git diff --check` 通过；
`vmi_to_vpto_ensure_mask_granularity.pto` lowering exit=0。
# mask staging 方向对称整改

本轮将 `materializeStagingDeintToContiguousMaskLayout` 的单组 source 收集和
factor=2/4 `intlv` 物化抽取为 `materializeDeintToContiguousMaskGroup`。外层函数
继续负责结果容量截断和最终 arity 校验，保持 deinterleaved part 顺序、尾部结果
截断和失败诊断不变。增量合规检查结果为 `checked_files=1 errors=0 warnings=0`，
`git diff --check` 通过；`vmi_to_vpto_ensure_mask_granularity.pto` lowering exit=0。
# group_broadcast lowering 整改

本轮将 `lowerGroupBroadcastParts` 的 source physical 类型/元素宽度校验抽取为
`validateGroupBroadcastSources`，将结果 layout factor 枚举、chunk lowering 和结果
arity 校验抽取为 `lowerGroupBroadcastResultChunks`。主函数现在只负责 layout fact、
selector context 和两阶段分派；selector cache、结果顺序、BRC/E2B/fallback 选择及
诊断保持不变。增量合规检查结果为 `checked_files=1 errors=0 warnings=0`，
`git diff --check` 通过；`vmi_layout_assignment_group_broadcast_load_e2b_b16.pto`
lowering exit=0。
本轮将 `OneToNVMIGroupReduceOpPattern::lowerContiguousRows` 的每 group/chunk
row-reduce 与 combine 循环抽取为 `buildContiguousGroupReduceResults`。外层现在只
负责 contiguous shape/类型契约、row result 类型准备及 slots=1/多 chunk 结果布局
写回；累积顺序和 `PAT_VL1` combine mask 保持不变。增量合规检查结果为
`checked_files=1 errors=0 warnings=0`，`git diff --check` 通过；group_reduce partial
slots=8 case exit=0。其他两个 group-reduce case 在既有 VMI pack/unpack pipeline
invariant 处提前失败，未进入本轮 lowering 路径。
# deinterleaved group reduction 整改

本轮将 `OneToNVMIGroupReduceOpPattern::lowerFullDeinterleaved2` 中按 group/chunk
执行 low/high row reduction、pair combine 和 accumulator 的循环抽取为
`buildDeinterleaved2GroupResults`。外层函数继续负责 slots=1、group/chunk arity、
row 类型/mask 准备和最终结果 bitcast；low/high 配对顺序、`PAT_VL1` mask 及诊断
保持不变。增量合规检查结果为 `checked_files=1 errors=0 warnings=0`，`git diff --check`
通过；group-reduce partial slots=8 case exit=0。
# packed-byte group store 二次拆分

本轮继续拆分 `lowerPackedByteSlots8`：新增 `buildPackedByteStoreBlock`，集中负责
每 32 个 group 的 `vselr/vsel` 合并、尾部 store mask 和 group offset 构造；外层只
负责 physical selector 初始化、`PK4_B32` 直写或 packed-byte stream 的后端选择。
保持向量拼接顺序、active group mask、offset 和 stream 生命周期不变。增量合规检查
结果为 `checked_files=1 errors=0 warnings=0`，`git diff --check` 通过；
`vmi_to_vpto_group_store_slots8_packed_byte.pto` lowering exit=0。
