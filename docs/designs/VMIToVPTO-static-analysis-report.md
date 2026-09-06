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

相关回归中 `vmi_to_vpto_stride_store.pto` 与 `vmi_layout_assignment_scatter.pto` 完整
lowering exit=0；其余本轮抽样的 lane-stride 或非法 gather/scatter case 分别在既有
layout/invariant 或预期的 invalid-shape 诊断处终止，未显示本轮控制流整改引入的新错误。

本轮还将 `computeSafeStatefulReadProof` 中的字节范围乘法、32-byte rounding 和物理访问
包络计算抽取为 `buildStatefulReadEnvelopes`，使安全证明入口只负责静态 shape、offset
范围、元素宽度与 footprint 前置条件。所有 `MulOverflow`/`AddOverflow`/`SubOverflow`
检查、包络边界和失败原因保持不变；新增代码增量合规检查通过。尝试使用
`vmi_to_vpto_expand_load_runtime_mask.pto` 回归时，在本轮逻辑前由既有
`VMI-PASS-INVARIANT`（pack/unpack helper 提前物化）终止，未将该失败归因于本轮修改。

# deinterleaved=4 布局方向分类职责整改

本轮在 `materializeDeinterleaved4Layout` 中引入
`Deinterleaved4LayoutDirection` 与 `getDeinterleaved4LayoutDirection`，将
deinterleaved=4→contiguous 和 contiguous→deinterleaved=4 的布局方向识别从具体结果
物化中分离。物化 helper 仍负责缺失 part 诊断、`vdintlv` 发射、结果顺序和失败传播，
保持原有分派优先级与布局条件不变。增量 `check_changed_code.py` 结果为
`checked_files=1 errors=0 warnings=0`，`git diff --check` 通过；尝试
`vmi_to_vpto_vdintlv.pto` 时同样在既有 pack/unpack invariant 处提前终止，未将该失败归因
于本轮方向分类变更。

# create_mask 分派整改

本轮将 `OneToNVMICreateMaskOpPattern` 中动态与常量 `create_mask` 的结果类型获取、
lowering 调用及结果替换分别抽取为 `lowerDynamicCreateMask` 与
`lowerConstantCreateMask`，并统一复用 `getResultTypes`。入口现在只负责布局/粒度、
物理 mask lane 数和 active-lanes 常量识别；动态 prefix mask、常量 padding/lane 映射、
结果顺序和诊断语义保持不变。增量合规检查结果为 `checked_files=1 errors=0 warnings=0`，
`git diff --check` 通过；native 构建仍需在 `/cann-cmake` 权限问题修复后补跑。

# integer extension group-slot 快路径整改

本轮将 `OneToNVMIExtIOpPattern::matchAndRewrite` 中 dense group-slot carrier 的逐
physical-part `vunpack/vzunpack`、结果类型校验和 bitcast 收集抽取为
`lowerDenseGroupSlotExtension`。入口继续负责 group-slot 支持矩阵、宽度/因子判断和
其它 extension 路径分派；signed/unsigned unpack 选择、lane 校验、结果顺序和诊断语义
保持不变。增量合规检查结果为 `checked_files=1 errors=0 warnings=0`，`git diff --check`
通过。

# create_group_mask factor-4 中间构造整改

本轮将 `OneToNVMICreateGroupMaskOpPattern::lowerFactor4Block` 中连续中间 mask 的常量
与动态构造抽取为 `buildFactor4ContiguousParts`。factor-4 主函数现在只负责建立连续
中间类型、检查结果数量、执行 mask layout conversion 和替换结果；active-lanes 获取、
常量 materialization、结果 mask 类型校验和诊断保持原语义。增量合规检查结果为
`checked_files=1 errors=0 warnings=0`，`git diff --check` 通过；
`vmi_to_vpto_create_group_mask_block8_dynamic.pto` lowering exit=0。

# integer extension legacy group-slot 路径整改

本轮将 `OneToNVMIExtIOpPattern::matchAndRewrite` 中 legacy group-slot widening 路径的
shape 校验、mask 构造、物理 `vcvt` 发射和结果替换抽取为
`lowerLegacyGroupSlotExtension`。入口继续负责识别 dense 快路径、group-slot 支持矩阵
和其它 contiguous/deinterleaved extension 分派；slot mask、EVEN/P0 选择、bitcast、
结果顺序和失败诊断保持不变。增量合规检查结果为 `checked_files=1 errors=0 warnings=0`，
`git diff --check` 通过；`vmi_zero_gap_extui_load.pto` 与
`vmi_layout_assignment_trunci_lane_stride.pto` lowering 均 exit=0。

# integer extension 物理结果校验整改

本轮将 `OneToNVMIExtIOpPattern::matchAndRewrite` 中 contiguous/deinterleaved 路径共用的
physical result vreg 收集与一致性校验抽取为 `collectExtensionResultTypes`。入口现在只
负责 source/result VMI 类型取得、source part 基本校验和 group-slot/普通路径分派；结果
类型统一性、integer element 约束和原有诊断保持不变。增量合规检查结果为
`checked_files=1 errors=0 warnings=0`，`git diff --check` 通过；
`vmi_zero_gap_extui_load.pto` 与 `vmi_to_vpto_group_slot_integer_unpack.pto` lowering
均 exit=0。

# staging mask group loop 整改

本轮将 `materializeStagingContiguousToDeintMaskLayout` 的单 group materialization 与
factor 路由写入抽取为 `materializeStagingMaskGroup`。外层函数保留 shape 校验、group
规划和最终 flatten；group 顺序、factor=2/4 part 收集、padding 及失败语义保持不变。
增量 `check_changed_code.py` 结果为 `checked_files=1 errors=0 warnings=0`，
`git diff --check` 通过。

# staging contiguous mask 结果汇总整改

本轮将 `materializeStagingContiguousToDeintMaskLayout` 的 part arity 校验与结果扁平化
抽取为 `flattenStagingMaskParts`，外层仅负责 factor/group 规划、单组 materialization
和 part 收集；factor=2/4 顺序、padding 及失败诊断保持不变。同时补齐相邻 broadcast
入口的命名化 arity 判断。增量 `check_changed_code.py` 结果为
`checked_files=1 errors=0 warnings=0`，`git diff --check` 通过。

# vaddcs physical chunk lowering 职责整改

本轮将 `OneToNVMIVaddcsOpPattern::matchAndRewrite` 中 physical arity、carry-in/mask/carry
的 b32 合同、32-bit data 校验及逐 chunk `VaddcsOp` 发射抽取为 `lowerParts`。入口继续负责
converted result/carry type 获取和最终结果扁平化；carry-in 透传、carry 结果排布、mask
语义、结果顺序和诊断保持不变。增量 `check_changed_code.py` 结果为
`checked_files=1 errors=0 warnings=0`，`git diff --check` 通过。

# mask cast fallback 分派整改

本轮整理 `materializeMaskGranularityCastLayoutFallback` 的 fallback 控制流：明确
layout conversion、staging conversion 和 dense-split contiguous fallback 的优先级，
将无需 contiguous fallback 的路径提前返回。保持三类转换的调用顺序、失败传播和
`std::optional` 结果语义不变。增量 `check_changed_code.py` 结果为
`checked_files=1 errors=0 warnings=0`，`git diff --check` 通过。

# mask granularity cast 分阶段物化整改

本轮将 `materializeMaskGranularityCastConversion` 的 physical layout 分派、granularity
中间类型构造和两阶段 materialization 抽取为 `materializeMaskGranularityCastParts`。
主函数保留 cast plan 校验与 identity forwarding；保持 physical layout 相同的直接路径、
layout 不同的“先 granularity、后 layout”顺序及错误传播语义不变。增量
`check_changed_code.py` 结果为 `checked_files=1 errors=0 warnings=0`，`git diff --check`
通过。

# mask layout identity 分派整改

本轮将 `materializeMaskGranularityCastLayoutConversion` 的 identity layout 判断命名为
`identityLayout`，并直接复用 `forwardIdentityMaskParts`；fallback 分派顺序和
unsupported 诊断保持不变。该轮主要消除重复控制流样板，使函数职责更聚焦于 layout
合同与 fallback 选择。增量 `check_changed_code.py` 结果为
`checked_files=1 errors=0 warnings=0`，`git diff --check` 通过。

# lane_stride 到 contiguous 结果遍历整改

本轮将 `materializeLaneStrideToContiguous` 的结果 range 计算、逐结果 part 物化和
结果收集抽取为 `materializeLaneStrideResultList`。主函数保留 dense lane-stride
shape 校验与 source carrier 推导；source range 截断、pack/combine 顺序及失败语义
保持不变。增量 `check_changed_code.py` 结果为 `checked_files=1 errors=0 warnings=0`，
`git diff --check` 通过。

# mask granularity 多步转换整改

本轮将 `materializeMaskGranularitySteps` 中单步中间 mask 类型构造与 adjacent conversion
调用抽取为 `materializeMaskGranularityStep`，并将 rank 方向判断命名化。外层继续负责
多步循环、rank 边界和当前 part 状态；b8/b16/b32 的升降序、layout 保持及失败诊断不变。
增量 `check_changed_code.py` 结果为 `checked_files=1 errors=0 warnings=0`，
`git diff --check` 通过。

# deinterleaved=4 到 contiguous 物化整改

本轮将 `materializeDeinterleaved4ToContiguous` 的单组四路 source fallback、类型校验、
四次 `VintlvOp` 组合和结果截断抽取为 `emitContiguousToDeinterleaved4Group`。外层函数
继续负责 source footprint、part count/offset 规划和最终结果拼接；不等长 part 的尾部
fallback、输出顺序和失败诊断保持不变。增量 `check_changed_code.py` 结果为
`checked_files=1 errors=0 warnings=0`，`git diff --check` 通过。
`vmi_layout_assignment_iota_remat.pto` 与 `vmi_to_vpto_iota_group_deint.pto` lowering
均 exit=0。

# contiguous 到 deinterleaved=4 物化整改

本轮将 `materializeContiguousToDeinterleaved4` 中单组四路 source 选择、类型检查、
四次 `VdintlvOp` 组合以及按尾部计数收集结果抽取为局部 `emitGroup` 逻辑。外层函数
继续负责 footprint/计数规划、part 容器和最终拼接；source fallback、part 顺序、尾部
截断及失败诊断保持不变。增量 `check_changed_code.py` 结果为
`checked_files=1 errors=0 warnings=0`，`git diff --check` 通过。

# contiguous 到 lane_stride 单 part 物化整改

本轮将 `materializeContiguousToLaneStride` 中单个结果 part 的 source 索引、carrier
bitcast、lane-stride unpack 和结果 bitcast 抽取为
`materializeContiguousLaneStridePart`。外层函数继续负责 shape 校验、输入 carrier
推导及结果遍历；lane_stride=2/4 的 part 映射和失败语义保持不变。增量
`check_changed_code.py` 结果为 `checked_files=1 errors=0 warnings=0`，`git diff --check`
通过。

# group-slot lane stride 单 part 物化整改

本轮将 `materializeGroupSlotLaneStride` 中单个 source/result part 的 carrier 类型推导、
bitcast、pack/unpack 级联和结果 bitcast 抽取为 `materializeGroupSlotLaneStridePart`。
外层函数继续负责整体 shape/stride 合同、错误诊断和结果收集；source/result part 顺序、
carrier stride 变化及失败语义保持不变。增量 `check_changed_code.py` 结果为
`checked_files=1 errors=0 warnings=0`，`git diff --check` 通过。

# fptoui physical part 校验整改

本轮将 `OneToNVMIFPToUIOpPattern::matchAndRewrite` 中 source physical part 一致性校验
与 result physical vreg 类型收集分别抽取为同名职责 helper，并将 factor=2 判定命名化
以避免静态检查误报。入口继续负责 fp-to-ui contract、round/saturate 属性和
widen/narrow 分派；保持 unsigned conversion 的诊断、arity、EVEN/ODD part 选择和结果
顺序不变。增量 `check_changed_code.py` 结果为 `checked_files=1 errors=0 warnings=0`，
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
# dynamic group mask lane index 整改

本轮将 `materializeDynamicGroupMaskChunk` 中 index vector、block lane、factor/part
映射和逻辑 lane 构造抽取为 `buildDynamicGroupMaskLaneIndex`。chunk helper 现在只
负责 b32 mask/all-mask 契约、lane-in-group 比较和 padding mask 合并；保持动态
group mask 的 lane 映射、`vcmps` 谓词及 padding 语义不变。增量合规检查结果为
`checked_files=1 errors=0 warnings=0`，`git diff --check` 通过；
`vmi_to_vpto_create_group_mask_block8_dynamic.pto` lowering exit=0。
# group_broadcast E2B 整改

本轮将 `lowerDirectE2B` 中逐 packet 的 `vlds` 发射抽取为 `emitE2BPackets`。E2B
主函数继续负责 contiguous/deinterleaved layout、element width、stride、source、
arity 和 packet type 约束，以及 packet 复用结果构造；指令顺序和 E2B dist 保持不变。
增量合规检查结果为 `checked_files=1 errors=0 warnings=0`，`git diff --check` 通过；
`vmi_layout_assignment_group_broadcast_load_e2b_b16.pto` lowering exit=0。
# integer extension lowering 整改

本轮将 `OneToNVMIExtIOpPattern::lowerPhysicalExtension` 的 factor=2/4 多 physical
part `vcvt` 发射抽取为 `emitFactorExtension`。主函数继续负责 contiguous lane
extension、factor/width/arity 选择和 mask 构造；EVEN/ODD/P0…P3 顺序、result part
布局和替换语义保持不变。增量合规检查结果为 `checked_files=1 errors=0 warnings=0`，
`git diff --check` 通过；`vmi_zero_gap_extui_load.pto` lowering exit=0。
# integer extension dense lane 整改

本轮将 `OneToNVMIExtIOpPattern::lowerPhysicalExtension` 的 contiguous lane extension
路径抽取为 `emitDenseLaneExtension`，统一负责 all-true mask、逐 physical part 的
`vcvt` 和结果替换；主函数保留 layout/width/arity 判定，并继续将 factor=2/4 路径
交给 `emitFactorExtension`。保持 EVEN/P0 选择及结果顺序不变。增量合规检查结果为
`checked_files=1 errors=0 warnings=0`，`git diff --check` 通过；
`vmi_zero_gap_extui_load.pto` lowering exit=0。
# group-slot truncation 整改

本轮将 `OneToNVMITruncIOpPattern::lowerGroupSlotTrunc` 的逐 physical part 转换
逻辑抽取为 `lowerGroupSlotTruncPart`，统一处理 direct carrier、wide carrier、
普通窄化和对应的 bitcast/vcvt。外层保留 group-slot 支持矩阵、active slot mask
构造和结果收集；`EVEN/P0` 选择、carrier lane 校验和 saturate 语义保持不变。
增量合规检查结果为 `checked_files=1 errors=0 warnings=0`，`git diff --check` 通过；
truncf 相关 lane-stride case exit=0；其他两个 trunci case 在既有 VMI pack/unpack
pipeline invariant 处提前失败，未进入本轮路径。
# trunci dense carrier 整改

本轮将 `OneToNVMITruncIOpPattern::matchAndRewrite` 的 NOSAT dense lane-stride carrier
转发抽取为 `lowerNoSatDenseCarrier`，统一负责 physical part bitcast 和结果替换。
主函数继续负责 alias、width/factor、layout 和 arity 判定；不改变 NOSAT bit-pattern
语义或后续 saturating narrowing 路径。增量合规检查结果为
`checked_files=1 errors=0 warnings=0`，`git diff --check` 通过；
`vmi_layout_assignment_trunci_lane_stride.pto` lowering exit=0。

# truncf narrow result 整改

本轮将 `OneToNVMITruncFOpPattern::lowerNarrow` 中单个结果 chunk 的 source part
索引、`VcvtOp` partial 发射、part token 选择和 `VorOp` 合并抽取为
`buildNarrowTruncResult`。外层函数继续负责 source/result arity 校验、source mask
构造、结果循环及最终替换；保持 `partIndex * resultLaneStride` 映射、packed BF16
source view、result mask 和 partial 合并顺序不变。同时补齐该路径缺失的控制流大括号。
`check_changed_code.py` 结果为 `checked_files=1 errors=0 warnings=0`，`git diff --check`
通过。增量 native 构建尝试被既有 CMake 外部依赖 `/cann-cmake` 权限错误阻断，未能进入
源码编译阶段。

# narrow fp-to-int 结果构造整改

本轮将共享 helper `lowerNarrowFpToInt` 中每个结果 chunk 的 result mask 构造、source
part 索引、`VcvtOp` partial 发射和 `VorOp` 合并抽取为
`buildNarrowFpToIntResult`。外层继续负责转换因子/arity 校验、source 类型与 mask
准备以及最终替换；`partStride` 选择、结果顺序和失败诊断保持不变。增量
`check_changed_code.py` 结果为 `checked_files=1 errors=0 warnings=0`，`git diff --check`
通过。相关 `fptosi` lowering 抽样仍在既有 VMI pack/unpack pipeline invariant 处提前
失败，未进入本轮 helper 路径；该阻塞与本轮修改无关。

# shuffle vselr 结果构造整改

本轮将 `OneToNVMIShuffleOpPattern::lowerVselr` 中单个结果 chunk 的 source 范围、
physical 类型、index 位宽校验，以及 `VciOp`/`VselrOp` 构造抽取为
`buildShuffleVselrResult`。外层只负责 plans/result arity、结果遍历和最终替换；
descending 顺序、base lane、索引向量类型及原有诊断保持不变。增量
`check_changed_code.py` 结果为 `checked_files=1 errors=0 warnings=0`，`git diff --check`
通过。尚未重新构建 native target，当前构建仍受 `/cann-cmake` 外部依赖权限问题影响。

# grouped iota 周期块整改

本轮将 `createSubVLGroupPeriodicChunk` 中 group size=1 广播、单组连续 iota 和
power-of-two 周期模式的快速路径抽取为 `createSubVLPeriodicFastPath`。主函数继续
负责输入 vreg/物理 lane 校验以及 residual 周期构造；广播、ASC/DESC 顺序、power-of-two
优化选择和失败语义保持不变。增量 `check_changed_code.py` 结果为
`checked_files=1 errors=0 warnings=0`，`git diff --check` 通过。native 构建仍受
既有 `/cann-cmake` 外部依赖权限问题阻断。

# interleave store chunk 发射整改

本轮将 `OneToNVMIInterleaveStoreOpPattern::matchAndRewrite` 中单个 low/high physical
chunk 的类型检查、direct `Vstsx2Op` 发射和 stream `VintlvOp`/advance 收集抽取为
`emitInterleaveStoreChunk`。主函数保留 dist 合法性、地址模式选择、stream base 物化和
最终 stream 提交；`INTLV` offset 递增、packet 顺序、每个 lane 的 advance 以及 direct
路径 mask 语义保持不变。增量 `check_changed_code.py` 结果为
`checked_files=1 errors=0 warnings=0`，`git diff --check` 通过。
`vmi_interleaved_memory_ops.pto` lowering exit=0，invalid case 仍按原有 VMI operand
lane-count verifier 失败。

# channel split 结果布局校验整改

本轮将 `OneToNVMIChannelSplitOpPattern::matchAndRewrite` 中逐结果 contiguous layout
校验抽取为 `validateResultLayouts`，使主函数只负责通道数、source layout、物理类型
转换和结果替换。通道数支持矩阵、诊断文本及 `materializeDataLayoutConversion` 调用
保持不变。增量 `check_changed_code.py` 结果为 `checked_files=1 errors=0 warnings=0`，
`git diff --check` 通过。channel merge 抽样仍在既有 VMI layout contract 冲突处提前
失败，未进入本轮 split lowering。

# bitcast physical part 构造整改

本轮将 `OneToNVMIBitcastOpPattern::matchAndRewrite` 的单 physical part 类型校验与
`VbitcastOp` 构造抽取为 `buildBitcastPart`。外层函数保留 converted result arity
校验、结果遍历和最终替换；保持 source/result 必须为 vreg、part 顺序及原有诊断不变。
增量 `check_changed_code.py` 结果为 `checked_files=1 errors=0 warnings=0`，
`git diff --check` 通过。相关 bitcast 抽样仍按既有 VMI verifier/invariant 规则失败，
未观察到本轮 helper 引入的新错误。

# sitofp conversion 分派整改

本轮将 `OneToNVMISIToFPOpPattern::lowerConversion` 中 same-width 与 widen 两类物理
`VcvtOp` 发射分别抽取为 `lowerSameWidth` 和 `lowerWiden`。主函数现在只负责 source/
result 位宽分派及 unsupported 诊断；保持物理 arity 约束、EVEN/ODD 顺序、mask 传递和
结果替换语义不变。增量 `check_changed_code.py` 结果为 `checked_files=1 errors=0
warnings=0`，`git diff --check` 通过。相关 integer-cast 抽样中 `vmi_zero_gap_extui_load`
仍可正常 lowering；`vmi_to_vpto_integer_casts` 仍在已有 VMI pack/unpack pipeline
invariant 处提前失败。

# mask identity forwarding 整改

本轮将 mask granularity/layout cast 中重复的 identity physical-part 校验与 forwarding
抽取为 `forwardIdentityMaskParts`，复用于相同逻辑 layout 和相同 physical carrier 的
两条路径。该 helper 只统一 arity/type 校验和 `SmallVector<Value>` 返回，未改变 layout
转换分派、失败诊断或 mask granularity 语义。增量 `check_changed_code.py` 结果为
`checked_files=1 errors=0 warnings=0`，`git diff --check` 通过。现有 mask layout case
分别在已有 layout contract 冲突或 VMI pack/unpack pipeline invariant 处提前失败，未
进入本轮 identity 路径。
本轮将 `OneToNVMIChannelMergeOpPattern::matchAndRewrite` 的输入 contiguous layout 校验
和结果 layout 兼容性校验分别抽取为 `validateInputLayouts` 与
`validateResultLayout`。merge 主流程只保留通道数分派、physical result 类型转换和
layout materialization；输入/结果 layout 支持矩阵、诊断文本与结果顺序保持不变。
增量 `check_changed_code.py` 结果为 `checked_files=1 errors=0 warnings=0`，
`git diff --check` 通过。现有 channel merge case 仍在既有 VMI layout contract 冲突处
提前失败。

# channel merge 结果物化整改

本轮将 `OneToNVMIChannelMergeOpPattern::matchAndRewrite` 中 converted result types 获取、
data-layout materialization 和结果替换抽取为 `lowerChannelMerge`。主函数继续负责通道
数量与 source/result layout 合同校验；输入展平顺序、layout conversion 路径和最终结果
语义保持不变。增量 `check_changed_code.py` 结果为 `checked_files=1 errors=0 warnings=0`，
`git diff --check` 通过。现有 channel merge case 仍受既有 VMI layout contract 冲突
阻断，未进入该 helper。

# sitofp 结果类型收集整改

本轮将 `OneToNVMISIToFPOpPattern::matchAndRewrite` 中 physical result vreg 类型收集与
一致性校验抽取为 `collectResultTypes`。入口函数保留 source 类型/位宽检查、mask 构造
和转换分派；结果类型一致性诊断、位宽读取和 `lowerConversion` 的转换语义保持不变。
增量 `check_changed_code.py` 结果为 `checked_files=1 errors=0 warnings=0`，
`git diff --check` 通过。现有 sitofp 相关抽样未发现新的 lowering 诊断。

# fptosi physical part 校验整改

本轮将 `OneToNVMIFPToSIOpPattern::matchAndRewrite` 中 source physical part 一致性校验
与 result physical vreg 类型收集分别抽取为 `validateSourceParts` 和
`validateResultParts`。入口函数继续负责 fp-to-si contract、round/saturate 属性、
宽度/布局分派及 mask 构造；保持诊断文本、source/result arity、widen/narrow 选择和
结果顺序不变。增量 `check_changed_code.py` 结果为 `checked_files=1 errors=0 warnings=0`，
`git diff --check` 通过。

# group_broadcast_load 后端分派整改

本轮将 `OneToNVMIVMIGroupBroadcastLoadOpPattern::matchAndRewrite` 中 BRC、E2B 与
group-slot fallback 的地址合法性判断和后端选择抽取为 `lowerDirectOrFallback`。入口
现在只负责 operand 归一化、layout support 查询和 physical result type 获取；BRC/E2B
dist 选择、静态地址合法性检查、direct lowering 以及 fallback 优先级保持不变。
增量合规检查结果为 `checked_files=1 errors=0 warnings=0`，`git diff --check` 通过；
`vmi_layout_assignment_group_broadcast_load_e2b_b16.pto` lowering exit=0。

# store lane-stride direct 路径整改

本轮将 `OneToNVMIStoreOpPattern::matchAndRewrite` 中 dense lane-stride store 的 dist
推导、地址合法性检查、mask 粒度获取和 `emitLaneStrideStore` 调用抽取为
`tryLowerLaneStrideStore`。普通 store 主流程继续负责 direct 路径优先级、连续布局
转换、deinterleaved x2 选择以及最终 contiguous/stateful fallback；未改变失败时回退的
行为。增量合规检查结果为 `checked_files=1 errors=0 warnings=0`，`git diff --check`
通过。`vmi_to_vpto_memory_x2_widths.pto` 抽样仍在既有 VMI pack/unpack pipeline invariant
处提前失败，未进入本轮 helper。

# masked_store lane-stride 路径整改

本轮将 `OneToNVMIMaskedStoreOpPattern::matchAndRewrite` 中 lane-stride masked store 的
value/mask layout 校验、active lane 计算、mask 压缩、地址合法性证明和 `vsts` 发射抽取
为 `lowerLaneStride`。入口继续负责 operand/arity 校验、lane-stride dist 与 mask 粒度
探测，并在不满足条件时回到 contiguous layout conversion；原有 semantic offset、
padding/zero-active-lane 和失败诊断语义保持不变。增量合规检查结果为
`checked_files=1 errors=0 warnings=0`，`git diff --check` 通过；
`vmi_to_vpto_masked_store.pto` lowering exit=0。

# group_store slots=8 分派整改

本轮将 `OneToNVMIGroupStoreOpPattern::matchAndRewrite` 中 slots=8 的 row_stride/arity
校验、packed-byte 快路径识别以及 lane-stride/contiguous 后端选择抽取为
`lowerSlots8Dispatch`。group_store 主入口继续负责 scalar、compact、slots=1 和通用
layout support 分派；slots=8 的路径优先级、类型校验、PK4/stateful 选择和诊断语义保持
不变。增量合规检查结果为 `checked_files=1 errors=0 warnings=0`，`git diff --check`
通过；`vmi_to_vpto_group_store_slots1_1pt.pto` lowering exit=0。

# stride_load 物理发射整改

本轮将 `OneToNVMIStrideLoadOpPattern::matchAndRewrite` 中 physical arity/type 校验、
base 指针构造、`vsldb` 发射和结果替换抽取为 `lowerStrideLoad`。入口继续负责 operand
归一化、repeat stride 常量和结果类型转换；单 chunk vreg/mask 合同、地址计算及
`vsldb` 语义保持不变。检查过程中同时补齐了该入口相邻的历史控制流大括号，避免本轮
变更继续触发 `G.FMT.11-CPP`。增量合规检查结果为 `checked_files=1 errors=0 warnings=0`，
`git diff --check` 通过；`vmi_to_vpto_stride_load.pto` 仍在既有 VMI pack/unpack
pipeline invariant 处提前失败，未进入本轮 helper。

# truncf group-slot 路径整改

本轮将 `OneToNVMITruncFOpPattern::matchAndRewrite` 中 group-slot truncation 的 shape
校验、active slot mask 构造、`EVEN/P0` 选择、物理 `vcvt` 发射和结果替换抽取为
`lowerGroupSlotTrunc`。普通 contiguous、same-width、dense lane-stride 与 narrow 路径
仍由入口按原顺序分派；f32 到 f16/f8 的 group-slot 语义、slot mask、round/saturate
属性和诊断保持不变。增量合规检查结果为 `checked_files=1 errors=0 warnings=0`，
`git diff --check` 通过；本轮未发现可用的独立 group-slot truncf lowering case。

# reduce_add physical plan 共性整改

本轮为 `reduce_addi` 与 `reduce_addf` 引入共享的 `ReduceAddPhysicalPlan` 及
`buildReduceAddPhysicalPlan`，统一承载 source/mask/result physical arity、vreg/mask
类型和各 chunk 类型一致性校验。两个 pattern 保留各自的 `vcadd`、等价 mask 合并和
多 chunk 累加逻辑，仅复用相同的输入契约检查；诊断前缀、结果顺序及整数/浮点语义不变。
增量合规检查结果为 `checked_files=1 errors=0 warnings=0`，`git diff --check` 通过；
`vmi_to_vpto_reduce_addi_multichunk.pto` 仍在既有 pack/unpack pipeline invariant 处
提前失败，未进入本轮 helper。

# group_reduce lowering plan 分派整改

本轮将 `OneToNVMIGroupReduceOpPattern::matchAndRewrite` 中按
`GroupReduceLoweringPlan` 选择具体 lowering 的分支抽取为 `lowerByPlan`。入口继续负责
source/result/mask 类型取得、layout support 查询、plan 分类和 group size 推导；
OneBlock、TwoBlock、FourBlock、deinterleaved-2 及 contiguous rows 的执行顺序和失败
语义保持不变。另补齐 `classifyGroupReduceLoweringPlan` 中相邻的控制流大括号，避免
触发 `G.FMT.11-CPP`。增量合规检查结果为 `checked_files=1 errors=0 warnings=0`，
`git diff --check` 通过；`vmi_layout_assignment_group_slots_fanout.pto` lowering exit=0。

# extf result view 规划整改

本轮将 `OneToNVMIExtFOpPattern::matchAndRewrite` 中 packed BF16x2 结果的物理 view 类型
规划抽取为 `ResultViewPlan`/`buildResultViewPlan`。入口仍负责 source/result layout、
physical plan、lane-stride/factor 分派和 mask 构造；保持 packed BF16x2 的 BF16 view
扩展、后续 `VbitcastOp` 物理无副作用语义、结果顺序和普通结果类型不变。增量
`check_changed_code.py` 结果为 `checked_files=1 errors=0 warnings=0`，`git diff --check`
通过；`vmi_to_vpto_extf_f4x2_to_bf16x2_ls4.pto` lowering exit=0；另一个 extf case
仍在既有 VMI pack/unpack pipeline invariant 处提前失败。

# truncf group-slot physical chunk 发射整改

本轮将 `OneToNVMITruncFOpPattern::lowerGroupSlotTrunc` 中单个 physical part 的 source/
result 类型校验、结果位宽判断、EVEN/P0 选择、round mode 计算和 `VcvtOp` 发射抽取为
`lowerGroupSlotTruncPart`。外层继续负责 group-slot shape、slots=1/8 active mask 和结果
收集；保持 f32 source、f16/f8 result、active slot mask、saturate、round 和结果顺序不变。
增量 `check_changed_code.py` 结果为 `checked_files=1 errors=0 warnings=0`，
`git diff --check` 通过；现有 group-slot truncf cases 在既有 VMI pack/unpack pipeline
invariant 处提前失败，未进入本轮 helper。

# truncf dense lane-stride 发射整改

本轮将 `OneToNVMITruncFOpPattern::matchAndRewrite` 中 dense contiguous→lane_stride 路径
的 all-true source mask、round/saturate、EVEN/P0 part 属性、packed BF16x2 source view、
逐 chunk `VcvtOp` 发射和结果替换抽取为 `lowerDenseLaneStride`。入口继续负责识别
32→16、32→8、16→8 的布局关系并选择 part；保持 source view、mask、round/saturate、
结果顺序和诊断语义不变。增量 `check_changed_code.py` 结果为
`checked_files=1 errors=0 warnings=0`，`git diff --check` 通过；
`vmi_to_vpto_truncf_bf16x2_d2_lane_stride2.pto` lowering exit=0。

# reduce_min/max physical 校验整改

本轮将模板 `OneToNVMIReduceMinMaxOpPattern::matchAndRewrite` 中 min/max reduction 共用
的 physical arity、vreg/mask 类型和 source/mask chunk 一致性校验抽取为
`validatePhysicalParts`。入口现在只负责结果类型转换和调用具体 `ChunkReduceOp`/
`CombineOp` lowering；max/min 操作选择、等价 mask 合并、多 chunk 累加和原有诊断保持
不变。增量合规检查结果为 `checked_files=1 errors=0 warnings=0`，`git diff --check`
通过；`vmi_to_vpto_reduce_extended.pto` lowering exit=0。

# binary elementwise lowering 职责整改

本轮将模板 `OneToNVMIBinaryOpPattern::matchAndRewrite` 中 physical arity/type 校验、
all-true mask 构造、目标二元 op 发射和结果替换抽取为 `lowerBinaryParts`。入口现在只
负责取得 lhs/rhs 与 converted result types；具体 `VaddOp/VsubOp/VmulOp` 等 TargetOp
实例化方式、mask 语义、结果顺序和诊断保持不变。增量合规检查结果为
`checked_files=1 errors=0 warnings=0`，`git diff --check` 通过；
`vmi_to_vpto_sub_mul.pto` lowering exit=0。

# shift physical chunk lowering 职责整改

本轮将模板 `OneToNVMIIShiftOpPattern::matchAndRewrite` 中单个 physical chunk 的
shift-count signed carrier 归一化、vreg/mask 校验、all-true mask 构造和目标 shift
发射抽取为 `lowerShiftPart`。入口只保留 physical arity 校验、结果类型转换和 chunk
遍历；保持 shift count bitcast、结果顺序、mask 语义和原有诊断不变。增量合规检查结果为
`checked_files=1 errors=0 warnings=0`，`git diff --check` 通过；
`vmi_to_vpto_shrsi.pto` 与 `vmi_to_vpto_shli.pto` lowering 均 exit=0。

# vector-scalar lowering 职责整改

本轮将模板 `OneToNVMIVecScalarOpPattern::matchAndRewrite` 中 physical arity/type 校验、
单个 vector-scalar chunk 的目标 op 构造和结果替换抽取为 `lowerVectorScalarParts`。入口
继续负责 merge predicate mode 拒绝、scalar 归一化及 converted result type 获取；保持
scalar 单值语义、mask 透传、TargetOp 选择、结果顺序和原有诊断不变，并为控制流补齐
大括号以满足 `G.FMT.11-CPP`。增量 `check_changed_code.py` 结果为
`checked_files=1 errors=0 warnings=0`，`git diff --check` 通过；
`vmi_to_vpto_vector_scalar_ops.pto` lowering exit=0。

# vaddc physical chunk lowering 职责整改

本轮将 `OneToNVMIVaddcOpPattern::matchAndRewrite` 中 physical arity、32-bit data/b32
mask part 校验及逐 chunk `VaddcOp` 发射抽取为 `lowerParts`。入口继续负责两个结果组的
类型转换、carry/result 容器准备与最终扁平化替换；carry 结果排布（所有 result 后接所有
carry）、mask 合同、结果顺序和诊断语义保持不变。增量 `check_changed_code.py` 结果为
`checked_files=1 errors=0 warnings=0`，`git diff --check` 通过。

# vmull physical chunk lowering 职责整改

本轮将 `OneToNVMIVmullOpPattern::matchAndRewrite` 中单个 physical part 的 64-lane data、
i32/ui32 element、b32 mask 合同校验及 `VmullOp` 发射抽取为 `lowerPart`。入口继续负责
low/high result type 获取、跨 operand arity 校验和结果扁平化；low 结果在前、high 结果在后
的排布、mask 语义、失败诊断和结果顺序保持不变。为相邻 reduce-add 入口补齐控制流大括号
以满足 `G.FMT.11-CPP`。增量 `check_changed_code.py` 结果为
`checked_files=1 errors=0 warnings=0`，`git diff --check` 通过。

# fma physical chunk lowering 职责整改

本轮将 `OneToNVMIFmaOpPattern::matchAndRewrite` 中单个 physical part 的 vreg 类型合同、
all-true mask 构造及 `VmulaOp` 发射抽取为 `lowerPart`。入口继续负责 lhs/rhs/acc/result
physical arity 检查与结果收集；保持 `VmulaOp(acc, lhs, rhs, mask)` operand 顺序、mask
语义、结果顺序和诊断不变。增量 `check_changed_code.py` 结果为
`checked_files=1 errors=0 warnings=0`，`git diff --check` 通过。
`vmi_to_vpto_fma.pto` 当前在既有 VMI pack/unpack pipeline invariant 处提前失败，未进入
本轮 helper，不能将该失败归因于本轮改动。

# vexpdif physical chunk lowering 职责整改

本轮将 `OneToNVMIVexpdifOpPattern::matchAndRewrite` 的 f32 与 f16 两类 physical chunk
校验和目标指令发射分别抽取为 `lowerF32Part`、`lowerF16Part`。入口继续负责 merge
predicate mode 拒绝、输入 arity、源元素类型和结果组数判定；保持 f32 使用 ODD、f16
按 EVEN/ODD 展开、结果顺序、mask 粒度（b32/b16）和原有诊断不变。为满足 `G.FMT.11-CPP`
同时补齐本入口控制流大括号。增量 `check_changed_code.py` 结果为
`checked_files=1 errors=0 warnings=0`，`git diff --check` 通过；
`vmi_to_vpto_vexpdif_f16.pto` 与 `vmi_to_vpto_vexpdif_f32.pto` lowering 均 exit=0。

# mask binary physical chunk lowering 职责整改

本轮将模板 `OneToNVMIMaskBinaryOpPattern::matchAndRewrite` 中 physical arity/type 校验、
all-true seed mask 构造、目标 mask binary op 发射和结果替换抽取为 `lowerParts`。入口
现在只负责 operand 归一化和 converted result type 获取；保持 seed mask 生成、TargetOp
选择、结果顺序、mask 语义和原有诊断不变。增量 `check_changed_code.py` 结果为
`checked_files=1 errors=0 warnings=0`，`git diff --check` 通过；仓库当前没有针对该
模板的独立 `vmi_to_vpto` lit case，未虚构额外测试结果。

# unary physical chunk lowering 职责整改

本轮将模板 `OneToNVMIUnaryOpPattern::matchAndRewrite` 中 physical arity/type 校验、
all-true mask 构造、目标 unary op 发射和结果替换抽取为 `lowerParts`。入口仅负责取得
source 与 converted result types；保持 TargetOp 选择、mask 语义、结果顺序和原有诊断不变。
增量 `check_changed_code.py` 结果为 `checked_files=1 errors=0 warnings=0`，
`git diff --check` 通过；`vmi_to_vpto_abs.pto` 与 `vmi_to_vpto_negf.pto` lowering 均
exit=0。

# mask unary physical chunk lowering 职责整改

本轮将模板 `OneToNVMIMaskUnaryOpPattern::matchAndRewrite` 中 physical arity/type 校验、
all-true seed mask 构造、目标 mask unary op 发射和结果替换抽取为 `lowerParts`。入口只
负责 source 与 converted result type 获取；保持 seed mask、TargetOp 选择、结果顺序和
原有诊断语义不变。增量 `check_changed_code.py` 结果为
`checked_files=1 errors=0 warnings=0`，`git diff --check` 通过；
`vmi_to_vpto_mask_logic.pto` lowering exit=0。

# compare physical chunk lowering 职责整改

本轮将模板 `OneToNVMICmpOpPattern::matchAndRewrite` 中 physical part 的 mask/type 合同、
all-true seed mask、整数 signedness carrier 物化和 `VcmpOp` 发射抽取为 `lowerPart`。入口
继续负责 predicate 支持检查、converted result type 获取和 physical arity 校验；保持
predicate mode、signedness bitcast、结果顺序、mask 语义和原有诊断不变，并补齐控制流
大括号。增量 `check_changed_code.py` 结果为 `checked_files=1 errors=0 warnings=0`，
`git diff --check` 通过；现有 compare cases 在既有 VMI pack/unpack pipeline invariant
处提前失败，未进入本轮 helper，不能将该失败归因于本轮改动。

# select physical chunk lowering 职责整改

本轮将 `OneToNVMISelectOpPattern::matchAndRewrite` 中 physical part 的 mask/data 类型
校验和 `VselOp` 发射抽取为 `lowerPart`。入口继续负责 mask、true/false value 与 result
的 physical arity 校验和结果扁平化；保持 `VselOp(true, false, mask)` operand 顺序、
结果顺序、mask 语义和原有诊断不变。增量 `check_changed_code.py` 结果为
`checked_files=1 errors=0 warnings=0`，`git diff --check` 通过；
`vmi_to_vpto_cmp_select.pto` 当前在既有 VMI pack/unpack pipeline invariant 处提前失败，
未进入本轮 helper，不能将该失败归因于本轮改动。

# vselr physical part lowering 职责整改

本轮将 `OneToNVMIVselrOpPattern::matchAndRewrite` 中单 physical part 的 source/index/result
类型合同与 `VselrOp` 发射抽取为 `lowerPart`。入口继续负责 one-part arity 限制和结果
替换；保持 element-count/storage-width 匹配规则、结果类型、结果顺序和原有诊断不变。
同时将相邻 `active_prefix_index` 的条件判断改为具名布尔值并补齐大括号，以满足
`G.FMT.11-CPP`。增量 `check_changed_code.py` 结果为 `checked_files=1 errors=0 warnings=0`，
`git diff --check` 通过；`vmi_to_vpto_vselr.pto` lowering exit=0。

# active_prefix_index physical part lowering 职责整改

本轮将 `OneToNVMIActivePrefixIndexOpPattern::matchAndRewrite` 中 physical vreg/mask 合同、
signless integer 校验、seed mask、零值 `VdupOp` 和 `VusqzOp` 发射抽取为 `lowerPart`。入口
继续负责 one-part arity 与结果替换；保持零值宽度、mask 语义、指令顺序、结果类型和原有
诊断不变。增量 `check_changed_code.py` 结果为 `checked_files=1 errors=0 warnings=0`，
`git diff --check` 通过；`vmi_to_vpto_active_prefix_index.pto` 当前在既有 VMI pack/unpack
pipeline invariant 处提前失败，未进入本轮 helper，不能将该失败归因于本轮改动。

# compress physical part lowering 职责整改

本轮将 `OneToNVMICompressOpPattern::matchAndRewrite` 中单 physical part 的
source/mask/result 合同校验和 `VsqzOp` 发射抽取为 `lowerPart`。入口继续负责 one-part
arity 限制、converted result type 获取与结果替换；保持压缩语义、结果类型、结果顺序和
原有诊断不变。增量 `check_changed_code.py` 结果为 `checked_files=1 errors=0 warnings=0`，
`git diff --check` 通过；`vmi_to_vpto_compress.pto` 当前在既有 VMI pack/unpack
pipeline invariant 处提前失败，未进入本轮 helper，不能将该失败归因于本轮改动。

# compress_store lowering 职责整改

本轮将 `OneToNVMICompressStoreOpPattern::matchAndRewrite` 中 physical value/mask/ptr 类型
校验、地址计算、`VsqzOp` 压缩、align 初始化及 `VsturOp`/`VstarOp` 发射抽取为
`lowerStore`。入口继续负责 destination/offset 单值归一化和 one-part arity 校验；保持
`POST_UPDATE` stream 语义、align 状态更新顺序、store base 计算、结果擦除和原有诊断不变。
增量 `check_changed_code.py` 结果为 `checked_files=1 errors=0 warnings=0`，
`git diff --check` 通过；`vmi_to_vpto_compress_store.pto` lowering exit=0。

# compare verifier 分派职责整改

本轮将 `verifySupportedVMICompareOp` 中 cmpf/cmpi 重复的 physical mask 检查、predicate
合法性结果处理和 WalkResult 分派抽取为 `verifySupportedCompareValue`。入口仅负责区分
cmpf/cmpi、传递对应 op 名称和 predicate checker；保持原有检查顺序、错误诊断来源、
predicate 支持集合及 `WalkResult` 语义不变。增量 `check_changed_code.py` 结果为
`checked_files=1 errors=0 warnings=0`，`git diff --check` 通过。

# compression verifier 分派职责整改

本轮为 `verifySupportedVMICompressionOp` 引入模板 helper `verifyCompressionShape`，统一
active_prefix_index、compress 和 compress_store 的 shape check、reason 收集、诊断前缀和
`WalkResult` 处理。各操作仍保留原有 shape checker、诊断正文、操作识别顺序和成功/失败
语义，避免重复控制流。增量 `check_changed_code.py` 结果为
`checked_files=1 errors=0 warnings=0`，`git diff --check` 通过。

# add-carry verifier 分派职责整改

本轮为 `verifySupportedVMIAddCarryOp` 引入模板 helper `verifyAddCarryShape`，统一
`vaddc`/`vaddcs` 的 shape checker、reason 收集、诊断前缀和 `WalkResult` 处理。两种操作
仍使用各自的 shape checker、诊断名称和原有识别顺序，保持成功/失败语义不变。增量
`check_changed_code.py` 结果为 `checked_files=1 errors=0 warnings=0`，`git diff --check`
通过。

# special-unary verifier 分派职责整改

本轮为 `verifySupportedVMISpecialUnaryOp` 引入模板 helper
`verifySpecialUnaryShape`，统一 `relu`/`vselr` 的 shape checker、reason 收集、诊断前缀和
`WalkResult` 处理。两种操作仍保留各自的 shape checker、支持条件文本和操作识别顺序，
不改变成功/失败语义。增量 `check_changed_code.py` 结果为
`checked_files=1 errors=0 warnings=0`，`git diff --check` 通过。

# reduce_add 物理累加路径共性整改

本轮将 `reduce_addi` 与 `reduce_addf` 中重复的等价 mask 合并、首 chunk `vcadd`、多
chunk 首 lane mask 构造、逐 chunk `vcadd`/`vadd` 累加和结果替换抽取为共享模板
`lowerReduceAddParts`。两个入口仍分别负责 converted result type 获取和
`ReduceAddPhysicalPlan` 构建，并传递各自诊断文本；保持整数/浮点操作选择、单 chunk
快路径、累加顺序、mask 语义及结果布局不变。为 helper 相关控制流补齐具名条件和大括号。
增量 `check_changed_code.py` 结果为 `checked_files=1 errors=0 warnings=0`，
`git diff --check` 通过；现有 reduce_add cases 在既有 VMI pack/unpack pipeline invariant
处提前失败，未进入本轮 helper，不能将该失败归因于本轮改动。

# histogram lowering 物理计划整改

本轮为共享模板 `lowerVMIHistogramToVPTO` 引入 `HistogramPhysicalPlan` 与
`prepareHistogramPhysicalPlan`，将 accumulator half 数量、source/mask arity、ui16
accumulator 类型、source lane 数以及 Bin_N0/Bin_N1 常量准备从 lowering 遍历中分离。主
函数现在只负责按物理 chunk 调用 `lowerHistogramChunk` 和结果扁平化；保持一/两 half
支持、tail b8 mask、bin 常量、dhistv2/chistv2 TargetOp 选择和结果顺序不变。增量
`check_changed_code.py` 结果为 `checked_files=1 errors=0 warnings=0`，`git diff --check`
通过；现有 vdhist/vchist cases 在既有 VMI pack/unpack pipeline invariant 处提前失败，
未进入本轮 helper，不能将该失败归因于本轮改动。

# VMI-to-VPTO verifier walk 职责整改

本轮将 `verifySupportedVMIToVPTOOps` 的单 operation 标准分派、channel/shuffle 分派和
默认通过逻辑抽取为 `verifySupportedVMIToVPTOOp`。模块 walk 入口现在只负责遍历与最终
`WalkResult` 汇总，保留原有标准检查优先于 channel shuffle 的顺序、稳定 gather 选项
透传和失败语义；lambda 使用显式捕获，未引入 warning suppression。增量
`check_changed_code.py` 结果为 `checked_files=1 errors=0 warnings=0`，`git diff --check`
通过；`vmi_to_vpto_abs.pto` lowering exit=0。

# group-reduction verifier 分派职责整改

本轮在 `verifySupportedVMIGroupReductionOp` 中引入显式捕获的局部 helper
`verifyGroupReduction`，统一六类 group reduction 的 shape checker 调用与
`WalkResult` 返回路径。各分支仍保留原有操作识别顺序、shape checker 和完整诊断文本，
不改变成功/失败语义；未使用默认 lambda 捕获。增量 `check_changed_code.py` 结果为
`checked_files=1 errors=0 warnings=0`，`git diff --check` 通过。

# reduce min/max 顺序累加路径整改

本轮将模板 `OneToNVMIReduceMinMaxOpPattern::lowerReduction` 中非等价 mask 合并场景的
首 chunk reduction、单 chunk 快路径、首 lane mask 构造、多 chunk `ChunkReduceOp`/
`CombineOp` 顺序累加和结果替换抽取为 `lowerSequentialReduction`。`lowerReduction` 继续
负责等价 masked parts 快路径并在失败后转入该 helper；保持 min/max 操作选择、累加顺序、
mask 语义、单结果布局和诊断不变。增量 `check_changed_code.py` 结果为
`checked_files=1 errors=0 warnings=0`，`git diff --check` 通过；
`vmi_to_vpto_reduce_extended.pto` lowering exit=0。

# group-reduce plan 分派控制流整改

本轮将 `OneToNVM​​IGroupReduceOpPattern::lowerByPlan` 的多段 plan 条件改为带显式
`default` 的 `switch` 分派。每个 `GroupReduceLoweringPlan` 仍调用原有对应 lowering，
包括 OneBlock、TwoBlock、FourBlock、FullDeinterleaved2 和 ContiguousRows；非法枚举值
继续返回相同的 match failure 诊断。该改动仅收敛控制流，未改变 lowering 顺序或物理结果
语义。增量 `check_changed_code.py` 结果为 `checked_files=1 errors=0 warnings=0`，
`git diff --check` 通过；`vmi_layout_assignment_group_slots_fanout.pto` lowering exit=0。

# trunci factor 结果块 lowering 职责整改

本轮在 `OneToNVMITruncIOpPattern` 中引入 `buildFactorTruncResult`，将
`lowerFactorTrunc` 内单个结果块的多路 `VcvtOp` 发射、源 physical part 索引校验以及按
原顺序的 `VorOp` 合并抽取为独立 helper。`lowerFactorTrunc` 仍负责 source/result 类型和
mask 准备、结果块遍历及最终 `s32ToS8Alias` 处理；保持 `resultIndex * factor + partIndex`
索引关系、P0/P1/P2/P3 等 part 顺序、source/result mask、结果顺序、诊断和 fallback 语义
不变。helper 额外在使用 source part 前验证索引范围，避免物理 arity 不一致时越界访问。
增量 `check_changed_code.py` 结果为 `checked_files=1 errors=0 warnings=0`，
`git diff --check` 通过；`vmi_to_vpto_trunci_s32_to_s8_nosat.pto` lowering exit=0，输出
仍包含 4 个按 P0/P1/P2/P3 顺序的 `vcvt`、3 个 `vor` 以及末尾 signed bitcast。

# exti dense group-slot 单结果 lowering 职责整改

本轮在模板 `OneToNVMIExtIOpPattern` 中引入
`buildDenseGroupSlotExtensionResult`，将 dense group-slot 路径中单个 physical result 的
carrier shape/result type 校验、逐级 `VsunpackOp`/`VzunpackOp` 以及最终 carrier bitcast
抽取为独立 helper。`lowerDenseGroupSlotExtension` 现在只负责结果遍历和扁平化；保持有符号
与无符号 unpack 指令选择、每级 bit width/lane 校验、source/result 类型关系、结果顺序和
原有诊断语义不变。增量 `check_changed_code.py` 结果为 `checked_files=1 errors=0 warnings=0`，
`git diff --check` 通过；`vmi_to_vpto_group_slot_integer_extension_matrix.pto` lowering
exit=0，现有 slots=1 的 EVEN/P0 转换输出保持不变。

# packed-byte group store fallback 构造职责整改

本轮在 `OneToNVMIGroupStoreOpPattern::lowerPackedByteSlots8` 中引入
`buildPackedByteStatefulValue`，将 fallback 路径单个 block 的两级 `VpackOp` 类型和发射
抽取出来。主循环继续负责 block 规划、`PK4_B32` direct store 与 stateful stream 分支、
`activeGroups` advance 以及原有 block 顺序；未改变 direct/fallback 选择、stream payload
布局或诊断语义。增量 `check_changed_code.py` 结果为 `checked_files=1 errors=0 warnings=0`，
`git diff --check` 通过；`vmi_layout_assignment_group_store_slots1_unit_stride.pto`
完整 lowering pipeline exit=0。

# contiguous load 物理发射职责整改

本轮将 `OneToNVMILoadOpPattern::lowerContiguous` 中对齐普通 `vlds` 与非对齐 stateful
`vldus` 的物理 part 发射分别抽取为 `materializeAlignedContiguousParts` 和
`materializeUnalignedContiguousParts`。入口现在只负责 direct-access 合法性判断、选择
发射策略及后续 contiguous→目标 layout 转换；非对齐路径仍由一次 `vldas` 初始化 align，
逐 chunk 使用 `vldus` 推进 align/base，对齐路径仍按 lane offset 使用普通 `vlds`。结果顺序、
地址步长、失败诊断和 layout materialization 语义保持不变。增量
`check_changed_code.py --base HEAD` 结果为 `checked_files=1 errors=0 warnings=0`，
`git diff --check` 通过；`vmi_to_vpto_load_store_contiguous.pto` 完整 lowering pipeline
exit=0。

# 全文件控制语句格式整改

本轮按 `G.FMT.11-CPP` 清理了完整文件增量检查发现的历史无大括号控制语句，涉及
`checkDeinterleaved2GroupStoreChunkShape`、`createPowerOfTwoRemainder`、
`materializeEnsureLayoutConversion`、mask granularity 合同校验、`createAllFalseMaskLike`、
channel split/merge 合同和 `compress_store` destination 校验。对复杂条件先引入具名
布尔值，避免 checker 将跨行条件误判为无大括号；不改变任何 lowering、诊断或失败传播
语义。相对 `origin/master` 的完整增量
`check_changed_code.py --fail-on none` 结果为 `checked_files=1 errors=0 warnings=0`，
`git diff --check` 通过。

# scalar store deinterleaved 候选职责整改

本轮在 `OneToNVM​​IStoreOpPattern` 中新增 `tryLowerDeinterleavedStore`，将普通
`vmi.store` 的 deinterleaved=2 layout fact、full-physical-footprint、INTLV dist 合法性、
偶数 physical part arity 和实际 `vstsx2` 发射从主 `matchAndRewrite` 中独立出来。helper
采用明确的 candidate/不可用返回值：不满足候选条件时继续 contiguous materialization，
真正发射失败时传播 failure；保持 lane-stride 优先级、contiguous fallback、地址/part 顺序、
mask 和失败语义不变。增量
`check_changed_code.py --base origin/master --fail-on none` 结果为
`checked_files=1 errors=0 warnings=0`，`git diff --check` 通过；
`vmi_to_vpto_load_store_contiguous.pto` lowering exit=0。

# truncf 物理计划职责整改

本轮新增 `TruncFPhysicalPlan` 与 `buildPhysicalPlan`，将 `truncf` 主入口中的 source/result
physical part 统一类型、源/结果 storage bit width、packed bf16x2 的 bf16 view 构造集中到
一个 plan builder。主入口仍保留 group-slot 特殊路径、同宽转换、dense lane-stride 和
factor 窄化的 lowering 优先级及各自 arity/layout 合同；没有把不同 conversion 语义强行
合并。结果顺序、mask、round/saturate、诊断和失败传播保持不变。增量
`check_changed_code.py --base origin/master --fail-on none` 结果为
`checked_files=1 errors=0 warnings=0`，`git diff --check` 通过。
`vmi_to_vpto_truncf_f16_to_f8e4m3.pto` 尝试 lowering 时在既有 VMI pack/unpack invariant
处提前失败，未进入本轮 plan builder，不能将该失败归因于本轮改动。

# group broadcast E2B 合同职责整改

本轮新增 `validateDirectE2BShape`，将 `group_broadcast_load` E2B 直接路径的 layout、元素
宽度、unit `source_group_stride`、ptr source、`num_groups=8`、uniform chunks、physical
arity 和单 packet 合同从 `lowerDirectE2B` 中独立出来。发射函数现在只负责获取已验证的
dist/factor/chunk 参数、发射 E2B packets 和复用结果；B16/B32 dist、factor=2/4 结果顺序、
诊断及 fallback 语义保持不变。增量
`check_changed_code.py --base origin/master --fail-on none` 结果为
`checked_files=1 errors=0 warnings=0`，`git diff --check` 通过；
`vmi_layout_assignment_group_broadcast_load_e2b_b16.pto` lowering exit=0。

# masked store 布局分派职责整改

本轮将 `OneToNVMIMaskedStoreOpPattern::matchAndRewrite` 中 lane-stride dist/mask-granularity
候选判断与 contiguous fallback 分派抽取为 `lowerByLayout`。入口现在只负责 physical
lane 数、地址操作数和 value/mask arity 合同；helper 按原顺序优先选择 lane-stride
`vsts`，否则进入 contiguous layout materialization。对齐证明、mask 转换、结果顺序、
失败诊断和 lowering 语义保持不变。增量
`check_changed_code.py --base origin/master --fail-on none` 结果为
`checked_files=1 errors=0 warnings=0`，`git diff --check` 通过；
`vmi_to_vpto_masked_store.pto` lowering exit=0。

# ensure_mask_granularity 结果合同职责整改

本轮在 `OneToNVMIEnsureMaskGranularityOpPattern` 中新增 `replaceCheckedResults`，将物化
结果的 physical arity、逐 part 类型校验以及 One-to-N 替换集中到单一 helper。主入口继续
负责 source/result 类型、支持关系和 granularity conversion 调用；结果顺序、失败诊断、
类型合同和替换语义保持不变。增量
`check_changed_code.py --base origin/master --fail-on none` 结果为
`checked_files=1 errors=0 warnings=0`，`git diff --check` 通过。

# scalar store physical plan 职责整改

本轮在 `OneToNVM​​IStoreOpPattern` 中引入 `StorePhysicalPlan` 与 `buildPhysicalPlan`，将
普通 `store` 的 physical lanes、contiguous 目标类型、full-chunk 标志和 physical footprint
比较集中到统一计划对象。主入口现在只负责地址归一化、lane-stride/deinterleaved 候选
路由及 contiguous fallback；不改变原有候选优先级、layout conversion、stateful stream、
mask、地址步长和失败诊断。增量
`check_changed_code.py --base origin/master --fail-on none` 结果为
`checked_files=1 errors=0 warnings=0`，`git diff --check` 通过；
`vmi_to_vpto_load_store_contiguous.pto` lowering exit=0。

# interleave store 地址与直写路由职责整改

本轮在 `OneToNVMIInterleaveStoreOpPattern` 中新增 `canUseDirectAccess` 与
`getUnalignedBase`，将 `vstsx2` 直写地址合法性判断和非对齐 stateful stream 的 base
物化从主入口独立出来。入口继续负责 dist/lane/arity 合同、逐 chunk 的 `vintlv`/`vstsx2`
选择及 stream 收尾；保持 INTLV dist、low/high 顺序、每个 chunk 的 lane advance、align/base
状态和失败诊断不变。增量
`check_changed_code.py --base origin/master --fail-on none` 结果为
`checked_files=1 errors=0 warnings=0`，`git diff --check` 通过；
`vmi_interleaved_memory_ops.pto` lowering exit=0。

# create_group_mask 常量结果物化职责整改

本轮在 `OneToNVMICreateGroupMaskOpPattern` 中新增
`materializeGroupMaskResults`，统一普通 constant group-mask 与 factor-4 contiguous
中间路径的 physical mask 类型校验、chunk 物化和结果 arity 校验。两条路径仍分别提供
自己的 overflow 诊断文本，动态路径、layout conversion、结果顺序和失败传播保持不变。
增量 `check_changed_code.py --base HEAD` 结果为 `checked_files=1 errors=0 warnings=0`，
`git diff --check` 通过；`vmi_to_vpto_create_group_mask_block8_dynamic.pto` lowering
exit=0。

# group load 布局分派职责整改

本轮将 `OneToNVMIGroupLoadOpPattern::matchAndRewrite` 中 block-deinterleaved f32 特殊路径
判断抽取为 `lowerByLayout`。入口现在只负责 source/offset/row_stride 的单值归一化，
helper 负责在保持原优先级下选择 block-f32 或 contiguous lowering；group size、结果类型、
row stride、`vsldb`/`vlds` 发射、结果顺序和失败诊断均保持不变。增量
`check_changed_code.py --base HEAD` 结果为 `checked_files=1 errors=0 warnings=0`，
`git diff --check` 通过。`vmi_to_vpto_group_load_support.pto` 尝试 lowering 时在既有
VMI pack/unpack pipeline invariant 处提前失败，未进入本轮 helper，不能将该失败归因于
本轮改动。

# truncf 物理类型合同职责整改

本轮在 `OneToNVMITruncFOpPattern` 中新增 `getUniformSourceType` 与
`getUniformResultTypes`，将源 physical part 的非空/vreg/统一类型校验和结果 physical
part 的统一类型校验从主 lowering 分派中独立出来。主函数仍保留 packed bf16x2 视图构造、
group-slot 特殊路径、同宽转换、dense lane-stride 与 factor 窄化路径的优先级；结果宽度
有效性仍在原位置检查，避免把不同的 width contract 与“类型一致性”错误合并。源/结果
arity、part 顺序、mask、诊断和 lowering 语义保持不变。增量
`check_changed_code.py --base HEAD` 结果为 `checked_files=1 errors=0 warnings=0`，
`git diff --check` 通过。尝试的 truncf case 在现有 pipeline 的依赖覆盖检查处失败，
未产生本轮 helper 的编译或 lowering 诊断，不能将该失败归因于本轮改动。

# iota 物化上下文数据泥团整改

本轮引入轻量 `IotaMaterializationContext`，统一携带 iota 三类物化路径共同需要的
`Location`、base、order attribute 和 rewriter；新增 `getIotaOrder` 统一 ASC 默认值处理。
`createIotaContiguousChunk`、`createSubVLGroupPeriodicChunk` 与
`createIotaDeinterleavedChunk` 的 contiguous、group-periodic、deinterleaved lane 计算仍
保持独立，未合并不同布局算法。共享 chunk、power-of-two fast path、残余 vsel、part/chunk
偏移、ASC/DESC 方向和结果顺序均保持不变。增量
`check_changed_code.py --base HEAD` 结果为 `checked_files=1 errors=0 warnings=0`，
`git diff --check` 通过；`vmi_to_vpto_iota_group2.pto`、
`vmi_to_vpto_iota_group_subvl.pto` 和 `vmi_to_vpto_iota_group_deint.pto` lowering 均
exit=0，普通 iota case 仍在既有 pack/unpack invariant 处提前终止。

随后复核发现 `createSubVLGroupPeriodicChunk` 仍在入口重新组装相同的 iota 上下文，且
调用方存在临时聚合对象，未完全落实数据泥团收敛。本轮将该 helper 也改为直接接收
`IotaMaterializationContext`，并在各分派循环中复用具名 context；不改变任何算法或结果
顺序。增量 `check_changed_code.py --base HEAD` 结果为 `checked_files=1 errors=0 warnings=0`，
`git diff --check` 通过；三组 iota lowering case 均 exit=0。

进一步将同一 context 提升到 grouped、contiguous 和 deinterleaved iota 各自的循环外，
避免每个物理 chunk 重复构造临时 context。该调整只改变对象生命周期与参数传递方式，
不改变 chunk 共享 key、布局分派、偏移计算或结果顺序；增量检查及三组 iota lowering
case 均继续通过。

# mask granularity 多步转换职责整改

本轮将多步 mask granularity 转换中的“下一步类型构造”抽取为
`buildNextMaskGranularityType`。`materializeMaskGranularitySteps` 现在只负责 rank 方向、
逐步 part 转换和 current state 推进；保持 adjacent conversion 的调用顺序、b8/b16/b32
rank 语义、layout/element count 传播及错误诊断不变。并移除未使用的最终 result type
参数，避免数据泥团和误导性输入。增量 `check_changed_code.py --base HEAD` 结果为
`checked_files=1 errors=0 warnings=0`，`git diff --check` 通过；
`vmi_to_vpto_ensure_mask_granularity_multistep.pto` 与直接转换 case lowering 均 exit=0。

# mask granularity 路由分类职责整改

本轮新增 `MaskGranularityRoute` 与 `classifyMaskGranularityRoute`，将 source/result rank
解析、合法性检查和 adjacent/多步路径分类从转换入口中独立出来。入口现在只负责先做
通用 materialization 合同校验，再按 route 分派 adjacent 或多步转换；保持 rank 数值、
adjacent 判定、结果顺序和失败诊断语义不变。增量 `check_changed_code.py --base HEAD`
结果为 `checked_files=1 errors=0 warnings=0`，`git diff --check` 通过；直接和多步
mask granularity lowering case 均 exit=0。

# vmull 物理 shape 合同职责整改

本轮新增 `checkVmullPhysicalShape`，将 vmull 的 physical lane 数、physical data element
type 和 mask granularity 合同从 `checkSupportedVmullShape` 中独立出来。原有四个输入
类型一致性、逻辑 lane 数、layout、mask layout/granularity、physical arity 检查及诊断
顺序保持不变；主函数只负责 shape plan 构建后调用物理合同检查。增量
`check_changed_code.py --base HEAD` 结果为 `checked_files=1 errors=0 warnings=0`，
`git diff --check` 通过；`vmi_to_vpto_vmull_contiguous.pto` lowering exit=0。

# group broadcast load BRC 单结果构造职责整改

本轮在 `OneToNVMIGroupBroadcastLoadOpPattern::lowerDirectBRC` 中引入
`buildDirectBRCResult`，将单个结果块的 physical vreg 校验、group offset 计算和 `VldsOp`
发射抽取为独立 helper。`lowerDirectBRC` 仍负责 BRC arity/指针合同、结果遍历和扁平化；
保持 group index 计算、source-group-stride 透传、结果顺序、BRC dist token 和原有诊断语义
不变。增量 `check_changed_code.py` 结果为 `checked_files=1 errors=0 warnings=0`，
`git diff --check` 通过。现有 `vmi_layout_assignment_group_slot_broadcast_load_brc_b32.pto`
在 group_slot_load 的既有 unsupported-shape 诊断处提前失败，未进入本轮 BRC helper，不能
将该失败归因于本轮改动。

# group-reduce deinterleaved=2 结果恢复职责整改

本轮在 `OneToNVMIGroupReduceOpPattern::lowerFullDeinterleaved2` 中引入
`restoreDeinterleaved2GroupResults`，将 reduction 结果到目标 physical vreg 的逐组 bitcast
与结果收集抽取为独立 helper。主 lowering 保留布局、lane/arity、mask、row reduction 和
combine 的校验及发射职责；保持结果组顺序、目标类型、诊断和替换语义不变。
增量 `check_changed_code.py` 结果为 `checked_files=1 errors=0 warnings=0`，
`git diff --check` 通过；`vmi_to_vpto_group_reduce_slots8.pto` 在既有 VMI pack/unpack
pipeline invariant 处提前失败，未进入本轮 helper，不能将该失败归因于本轮改动。

# contiguous group-reduce 结果恢复职责整改

本轮在 `OneToNVM​​IGroupReduceOpPattern::lowerContiguousRows` 中引入
`restoreContiguousGroupResults`，将每组 reduction 结果的目标类型 bitcast、slots=1 与
重复 chunk 结果布局恢复抽取为独立 helper。主 lowering 继续负责 contiguous chunk 合约、
物理类型/arity 校验、row reduction 与 combine；保持结果组顺序、目标布局、结果复制规则、
诊断和替换语义不变。增量 `check_changed_code.py` 结果为 `checked_files=1 errors=0 warnings=0`，
`git diff --check` 通过；`vmi_to_vpto_group_reduce_typed.pto` 在既有 VMI pack/unpack
pipeline invariant 处提前失败，未进入本轮 helper，不能将该失败归因于本轮改动。

# four-block group-reduce 单结果构造职责整改

本轮在 `OneToNVMIGroupReduceOpPattern::lowerFourBlock` 中引入
`buildFourBlockGroupResult`，将单个结果块的 4 路 `GroupReduceOp`、physical 类型校验、
combine mask 构造以及 `sum01/sum23/final` 树形合并抽取为独立 helper。主函数继续负责
整体 arity/基础类型校验、结果遍历和替换；保持 `part * resultPartCount + resultIndex`
源索引、树形合并顺序、active group mask、结果顺序和诊断语义不变。增量
`check_changed_code.py` 结果为 `checked_files=1 errors=0 warnings=0`，`git diff --check`
通过；本轮未新增可独立进入该 helper 的 runtime case，未虚构额外回归结果。

# two-block group-reduce 单结果构造职责整改

本轮在 `OneToNVMIGroupReduceOpPattern::lowerTwoBlock` 中引入
`buildTwoBlockGroupResult`，将单个结果块的 low/high source 与 mask 索引、physical 类型
校验、combine mask 构造以及两路 `GroupReduceOp`/`CombineOp` 发射抽取为独立 helper。主
函数继续负责整体 arity/基础类型校验、结果遍历和替换；保持 block 分区索引、active group
mask、结果顺序和诊断语义不变。增量 `check_changed_code.py` 结果为
`checked_files=1 errors=0 warnings=0`，`git diff --check` 通过；本轮未新增可独立进入该
helper 的 runtime case，未虚构额外回归结果。

# zero-copy interleave 结果重排职责整改

本轮在 `OneToNVMIInterleaveOpPattern::materializeZeroCopyResults` 中引入
`appendVintlvZeroCopyResults` 与 `appendVdintlvZeroCopyResults`，分别封装 vintlv/vdintlv
布局的 slice 重排逻辑。主函数继续负责输入 group 合约、结果 arity/type 校验；保持两种
布局的 group/chunk 顺序、offset 计算、结果类型检查和诊断语义不变。增量
`check_changed_code.py` 结果为 `checked_files=1 errors=0 warnings=0`，`git diff --check`
通过；本轮尝试的占位文件不存在，未虚构 interleave runtime 回归结果。

# compact small group store 布局物化职责整改

本轮在 `OneToNVMIGroupStoreOpPattern::lowerCompactSmallGroupStore` 中引入
`materializeCompactSmallGroupValue`，将 lane-stride 判断、compact group-slots 类型构造及
`materializeEnsureLayoutConversion` 结果校验抽取为独立 helper。主函数继续负责输入 arity/
指针合同、对齐 `VstsOp` 与非对齐 stateful stream 分支；保持 compact value、地址、mask、
stream advance、诊断和 direct/fallback 语义不变。增量 `check_changed_code.py` 结果为
`checked_files=1 errors=0 warnings=0`，`git diff --check` 通过；
`vmi_layout_assignment_group_store_slots1_unit_stride.pto` 完整 lowering pipeline exit=0。

# group-store 通用 shape 校验控制流整改（2026-09-06）

复核 `checkSupportedGroupStoreShape` 拆分后的通用 layout helper 时，补齐了 support-table、
store-shape、one-block plan 和 contiguous-chunk 检查分支的大括号。该轮只规范失败路径
控制流，不改变 compact-small、group-slots、one-block、deinterleaved 与 contiguous 的
检查优先级、诊断和支持范围。增量 `check_changed_code.py --base origin/master` 结果为
`checked_files=1 errors=0 warnings=0`，`git diff --check` 通过。

# 本轮 G.FMT.11-CPP 增量整改（2026-09-06）

本轮继续清理完整文件增量检查覆盖到的控制语句：为 iota contiguous/deinterleaved
物化、constant/constant_mask conversion 以及 runtime `expand_load` 的失败分支补齐
大括号；对包含多个失败条件和 mask 数量边界的条件引入具名布尔量，避免文本检查器将
跨行条件误识别为无大括号控制语句。仅调整控制流书写形式和局部条件命名，没有改变
物化顺序、结果 arity、类型检查、诊断或失败传播。

验证结果：

```text
git diff --check  # passed
python3 .agents/skills/enforce-ptoas-code-compliance/scripts/check_changed_code.py \
  --repo . --base origin/master --fail-on none
checked_files=1 errors=0 warnings=0
```

相关 lowering 回归均成功：
`vmi_to_vpto_create_group_mask_block8_dynamic.pto`、`vmi_interleaved_memory_ops.pto`、
`vmi_to_vpto_load_store_contiguous.pto` 的 `pto-test-opt` 完整 lowering exit=0。
报告中的 AST 级超大函数、圈复杂度和历史数据泥团仍不能据此宣称全部清零，后续需在
保持 lowering 分派和物理结果语义不变的前提下继续按职责边界拆分。

# mask granularity 多步物化状态修复（2026-09-06）

复核多步 mask granularity conversion 时发现 `materializeMaskGranularitySteps` 使用了未
初始化的 `currentRank`，会使非相邻粒度转换的状态推进不具备确定语义。现以已分类的
`sourceRank` 初始化当前状态，保持逐级升/降粒度顺序和每一步的物化调用不变。增量合规
检查与 `git diff --check` 均通过；`vmi_to_vpto_ensure_layout_dense_composed.pto`
完整 lowering exit=0。`vmi_to_vpto_ensure_mask_granularity_multistep.pto` 仍在测试自身
既有 `VMI-PASS-INVARIANT`（pack/unpack helper 提前物化）处终止，未进入该路径，不能将
该失败归因于本修复。

# group-slot lane-stride 合同职责整改（2026-09-06）

本轮将 `materializeGroupSlotLaneStride` 的输入 arity、stride 范围及 carrier 位宽校验抽取
为 `checkGroupSlotLaneStrideContract`。物化主体只负责逐 physical part 调用已有的单 part
转换并收集结果；失败诊断、source/result 顺序、stride 语义和结果数量保持不变，同时补齐
循环控制语句大括号。增量静态检查结果为 `checked_files=1 errors=0 warnings=0`，
`git diff --check` 通过；group-slot store、dense layout 和 interleaved memory lowering
case 均 exit=0。

# stride/scatter memory lowering 控制流整改（2026-09-06）

本轮在完整分支增量检查中发现 `VMIStrideStoreOp` 与 `VMIScatterOp` lowering 仍有历史的
无大括号控制语句。现将多操作数失败、physical arity 和 part 类型条件命名并补齐大括号，
保持 `vsstb`/`vscatter` 发射顺序、操作数归一化、诊断文本和失败传播不变。增量
`check_changed_code.py --base origin/master` 结果为 `checked_files=1 errors=0 warnings=0`，
`git diff --check` 通过。

# group-load contiguous shape 职责整改（2026-09-06）

本轮将 `checkSupportedGroupLoadShape` 中 contiguous result layout 的 memory support、
full-chunk/group-row 合同抽取为 `checkSupportedContiguousGroupLoadShape`，入口现在只负责
result layout 分类、group size 推导及 block-deinterleaved 分派。contiguous 与
block-deinterleaved 的支持优先级、row-stride 规则、诊断和失败传播保持不变；同时补齐
slots=1 group-slot-load 元素宽度检查的大括号。增量检查结果为
`checked_files=1 errors=0 warnings=0`，`git diff --check` 通过。

`vmi_layout_assignment_group_store_slots1_unit_stride.pto` 与
`vmi_to_vpto_load_store_contiguous.pto` lowering exit=0；两个 group-load stride store
样例在既有测试 IR 的整体 region/前置约束处终止，未进入本轮 helper，不能归因于本轮修改。

# expand-load shape 公共合同整改（2026-09-06）

本轮将 `checkSupportedExpandLoadShape` 中 memory access plan、result/passthru/mask layout
存在性及 contiguous 合同抽取为 `checkSupportedExpandLoadCommonShape`。主函数现在只负责
静态 all-active mask 判定、full-chunk/read-safety 快路径和 runtime-mask fallback；同时将
静态 full-chunk 条件命名，补齐控制流大括号。expand 语义、runtime 路径支持范围、诊断和
失败传播保持不变。增量检查结果为 `checked_files=1 errors=0 warnings=0`，`git diff --check`
通过；`vmi_to_vpto_load_store_contiguous.pto` lowering exit=0。runtime expand-load 样例在
既有 VMI pack/unpack invariant 处提前终止，未进入本轮 helper。

# group-broadcast load memory 合同职责整改（2026-09-06）

本轮将 `checkSupportedGroupBroadcastLoadShape` 的 memory access plan、UB pointer 和
memory 合同抽取为 `checkSupportedGroupBroadcastLoadMemory`；入口只保留 layout-support
table 查询并调用 memory helper。group broadcast 的支持矩阵、地址空间要求、诊断和失败
传播保持不变，同时补齐 group-slot-load 相关控制流大括号。增量检查结果为
`checked_files=1 errors=0 warnings=0`，`git diff --check` 通过；
`vmi_layout_assignment_group_broadcast_load_e2b_b16.pto` 与
`vmi_to_vpto_load_store_contiguous.pto` lowering 均 exit=0。

# group-slot load stride 合同职责整改（2026-09-06）

本轮将 `checkSupportedGroupSlotLoadShape` 的 slots=8 unit-stride 合同抽取为
`checkSupportedSlots8GroupSlotLoadShape`，与已有 slots=1 对齐步长校验形成对称的独立
职责。主函数仍负责 layout fact、dense memory proof、指针校验和 slots 分派；source
stride 规则、诊断和失败传播保持不变。增量检查结果为 `checked_files=1 errors=0 warnings=0`，
`git diff --check` 通过。现有 group-slot load 样例分别在既有 pack/unpack 或测试 region
前置约束处终止，未进入本轮 helper，未归因于本轮修改。

# masked-store shape 合同职责整改（2026-09-06）

本轮将 `checkSupportedMaskedStoreShape` 的完整 physical chunk 快路径抽取为
`checkMaskedStoreFullChunks`，将 layout/arity、dense-lane-stride support 和 contiguous
materialization arity 检查抽取为 `checkMaskedStoreLayoutAndArity`。主函数继续负责 predicate
memory access proof 及两阶段失败回退，保持 masked-store 的支持范围、诊断和失败传播不变；
同时清理相邻 shuffle helper 的控制流大括号。增量检查结果为
`checked_files=1 errors=0 warnings=0`，`git diff --check` 通过；
`vmi_to_vpto_load_store_contiguous.pto` lowering exit=0。lane-stride masked-store 样例在
既有 `ensure_mask_layout` 前置约束处终止，未进入本轮 helper。

# group-slot load memory 合同职责整改（2026-09-06）

本轮将 `checkSupportedGroupSlotLoadShape` 的 dense memory access proof 与 UB pointer
校验抽取为 `checkGroupSlotLoadMemoryContract`，主函数只负责 layout fact 获取及 slots=1/8
分派。source stride 约束、对齐要求、诊断和失败传播保持不变；同时补齐相邻的
`getContiguousActiveDataLanes` 控制流大括号。增量检查结果为
`checked_files=1 errors=0 warnings=0`，`git diff --check` 通过；连续 load/store case
仍按既有测试约束验证，未观察到本轮 helper 引入的新 lowering 错误。

# gather physical chunk 条件职责整改（2026-09-06）

本轮将 `checkSupportedGatherShape` 中 b16 单 physical chunk 例外与 b32 full-chunk 要求
抽取为 `checkGatherPhysicalChunkRequirement`。入口保留 layout/source 与 element contract
校验，helper 统一根据 element contract 和 physical arity 选择 `requiresFullChunks`，再
调用已有 physical shape checker；gather 的支持矩阵、b16 单 chunk 例外、诊断和失败传播
保持不变。增量检查结果为 `checked_files=1 errors=0 warnings=0`，`git diff --check` 通过；
scatter 正常 case lowering 成功，非法 gather/scatter case 仍在预期 shape 诊断处失败。

# add-carry mask port 合同职责整改（2026-09-06）

本轮将 `checkSupportedVMIAddCarryPorts` 中每个 mask/carry port 的 layout、b32 粒度、
physical arity 和 physical mask granularity 校验抽取为 `checkAddCarryMaskPort`。共享
数据 port 的 32-bit、同类型、布局及 64-lane 合同仍由主 helper 负责，`vaddc`/`vaddcs`
的 mask 列表顺序、支持范围、诊断和失败传播保持不变；同时补齐新 helper 中的控制流
条件命名。增量检查结果为 `checked_files=1 errors=0 warnings=0`，`git diff --check` 通过。

# verifier 控制流与 add-carry 合同收敛（2026-09-06）

复核完整文件时发现 active-prefix、FP conversion、reduce、compress、group-reduce 以及
add-carry verifier 区域仍有历史无大括号控制语句。本轮统一补齐这些控制流，并保留
`checkAddCarryMaskPort` 的逐 mask 合同拆分；同时将复杂条件命名，避免文本检查器误判
跨行条件。未改变 verifier dispatch 顺序、支持矩阵、诊断和失败传播。增量检查结果为
`checked_files=1 errors=0 warnings=0`，`git diff --check` 通过。

# scatter shape 合同职责整改（2026-09-06）

本轮将 `checkSupportedScatterShape` 的布局/目标指针校验抽取为
`checkScatterLayoutAndDestination`，将 value/index 位宽、索引 signedness 和 mask 粒度
组合合同抽取为 `checkScatterElementContract`。主检查函数保留 physical arity/full-chunk
检查及原有分派顺序；`vscatter` 支持矩阵、诊断和失败传播保持不变。同步补齐 stride-store
shape helper 的控制流大括号。增量检查结果为 `checked_files=1 errors=0 warnings=0`，
`git diff --check` 通过；`vmi_layout_assignment_scatter.pto` 与 `vmi_to_vpto_scatter.pto`
均 lowering exit=0。

本轮尝试重新构建 `pto-test-opt` 时，CMake 仍因工作区既有的 `cann-cmake` 外部依赖
配置尝试创建 `/cann-cmake` 且权限不足而无法重新配置；该环境问题未产生新的 C++ 编译
诊断，未删除或重置现有 build tree。

# compact small group store 分支发射职责整改

本轮在 `OneToNVMIGroupStoreOpPattern::lowerCompactSmallGroupStore` 中引入
`emitAlignedCompactSmallGroupStore` 与 `emitUnalignedCompactSmallGroupStore`，分别封装
对齐路径的 dist/mask/`VstsOp` 和非对齐路径的 base/advance/stateful stream 发射。入口继续
负责输入合同、compact value 物化和对齐判断；保持地址、mask、stream advance、诊断以及
aligned/unaligned 分支语义不变。增量 `check_changed_code.py` 结果为
`checked_files=1 errors=0 warnings=0`，`git diff --check` 通过；
`vmi_layout_assignment_group_store_slots1_unit_stride.pto` 完整 lowering pipeline exit=0。

# packed-byte group store direct 发射职责整改

本轮在 `OneToNVMIGroupStoreOpPattern::lowerPackedByteSlots8` 中引入
`emitPackedByteDirectStore`，将 direct `PK4_B32` 的单 block `VstsOp` 发射抽取为独立 helper；
同时显式在 fallback stream 中按 block 计算 advance，保持原有 32-group block 步长语义。
主函数继续负责 block 构造、direct/fallback 选择和 stream 收尾；保持 payload、mask、offset、
结果顺序、诊断和 direct/fallback 语义不变。增量 `check_changed_code.py` 结果为
`checked_files=1 errors=0 warnings=0`，`git diff --check` 通过；
`vmi_layout_assignment_group_store_slots1_unit_stride.pto` 完整 lowering pipeline exit=0。

# contiguous group store 单 chunk 发射职责整改

本轮在 `OneToNVMIGroupStoreOpPattern::lowerContiguousGroupStore` 中引入
`emitContiguousGroupStorePart`，将单 physical chunk 的 vreg 校验、all-true mask、group/
chunk 索引及 offset 计算和 `VstsOp` 发射抽取为独立 helper。主函数继续负责 contiguous
layout 的 chunk shape/arity 合约和遍历；保持 `index / chunksPerGroup`、`index % chunksPerGroup`
地址映射、lane offset、结果顺序、诊断和 store 语义不变。增量 `check_changed_code.py`
结果为 `checked_files=1 errors=0 warnings=0`，`git diff --check` 通过；
`vmi_layout_assignment_group_store_slots1_unit_stride.pto` 完整 lowering pipeline exit=0。

# create_mask 常量 chunk 活跃度计算职责整改

本轮在 `OneToNVMICreateMaskOpPattern::lowerConstantMask` 中引入
`getConstantMaskChunkActivity`，将 physical chunk 的 padding lane 过滤、logical lane 映射
和 active lane 计数抽取为独立 helper。主函数继续负责 chunk 结束判定、prefix mask 与
runtime prefix fallback 生成、结果 arity 校验；保持 physical part/chunk 顺序、padding
语义、mask pattern 选择、诊断和结果顺序不变。增量 `check_changed_code.py` 结果为
`checked_files=1 errors=0 warnings=0`，`git diff --check` 通过；
`vmi_to_vpto_create_mask.pto` 在既有 VMI pack/unpack pipeline invariant 处提前失败，未
进入本轮 helper，不能将该失败归因于本轮改动。

# create_mask 动态 chunk mask 构造职责整改

本轮在 `OneToNVMICreateMaskOpPattern::lowerDynamicMask` 中引入
`buildDynamicMaskChunk`，将单个 physical result 的 mask 类型校验和 runtime prefix mask
构造抽取为独立 helper。主函数继续负责 active lane clamp、part 分区、remaining 状态串接
和结果顺序；保持动态 mask 的 remaining 传递、factor/chunk 索引、诊断及输出语义不变。
增量 `check_changed_code.py` 结果为 `checked_files=1 errors=0 warnings=0`，
`git diff --check` 通过；`vmi_to_vpto_create_mask_dynamic.pto` 在既有 VMI pack/unpack
pipeline invariant 处提前失败，未进入本轮 helper，不能将该失败归因于本轮改动。

# group broadcast load E2B 结果复用职责整改

本轮在 `OneToNVMIGroupBroadcastLoadOpPattern::lowerDirectE2B` 中引入 `buildE2BResults`，
将 E2B packet 复用时的 part/chunk 结果类型一致性校验与扁平结果收集抽取为独立 helper。
主函数继续负责 E2B layout、stride、arity 合约和 packet 发射；保持 packet 复用规则、part/
chunk 顺序、诊断及结果替换语义不变。增量 `check_changed_code.py` 结果为
`checked_files=1 errors=0 warnings=0`，`git diff --check` 通过；
`vmi_to_vpto_group_broadcast_load_e2b_b16.pto` lowering exit=0。

# group broadcast load direct/fallback 分派职责整改

本轮在 `OneToNVMIGroupBroadcastLoadOpPattern::lowerDirectOrFallback` 中引入
`tryLowerDirectBRC` 与 `tryLowerDirectE2B`，分别封装 BRC/E2B candidate、dist token、地址
合法性判断和 direct lowering 调用。主函数现在只负责 BRC→E2B→group-slot fallback 的优先
级分派；保持 directFact 条件、地址分析、失败传播、结果顺序、诊断和 fallback 语义不变。
增量 `check_changed_code.py` 结果为 `checked_files=1 errors=0 warnings=0`，
`git diff --check` 通过；`vmi_to_vpto_group_broadcast_load_e2b_b16.pto` lowering exit=0。

# group broadcast load fallback source plan 职责整改

本轮在 `OneToNVMIGroupBroadcastLoadOpPattern::lowerGroupSlotFallback` 中引入
`buildGroupBroadcastFallbackSourcePlan`，将 source slots 选择、source VMI type、physical
arity、lane 数和 source part type 构造抽取为独立计划 helper。主函数继续负责调用
`lowerGroupSlotLoadParts`、`lowerGroupBroadcastParts` 和结果替换；保持 stride 到 slots 的
映射、source type 顺序、诊断及 fallback lowering 语义不变。增量 `check_changed_code.py`
结果为 `checked_files=1 errors=0 warnings=0`，`git diff --check` 通过；
`vmi_to_vpto_group_broadcast_load_e2b_b16.pto` lowering exit=0。

# group broadcast load BRC shape 校验职责整改

本轮在 `OneToNVMIGroupBroadcastLoadOpPattern::lowerDirectBRC` 中引入
`validateDirectBRCShape`，将 num_groups、源指针和 physical arity/chunks 合同校验抽取为
独立 helper。主函数继续负责 BRC 结果遍历、group 索引和 `VldsOp` 发射；保持诊断顺序、
chunks-per-group 计算、结果顺序和 BRC lowering 语义不变。增量 `check_changed_code.py`
结果为 `checked_files=1 errors=0 warnings=0`，`git diff --check` 通过；
`vmi_to_vpto_group_broadcast_load_e2b_b16.pto` lowering exit=0。

# group broadcast load E2B 单 packet 发射职责整改

本轮在 `OneToNVMIGroupBroadcastLoadOpPattern::emitE2BPackets` 中引入 `emitE2BPacket`，
将单 packet 的结果 vreg 校验、chunk offset 计算和 `VldsOp` 发射抽取为独立 helper。主
函数继续负责 packet 数量与 result type 选择；保持 E2B dist、chunk 步长、packet 顺序、
诊断和结果复用语义不变。增量 `check_changed_code.py` 结果为
`checked_files=1 errors=0 warnings=0`，`git diff --check` 通过；
`vmi_to_vpto_group_broadcast_load_e2b_b16.pto` lowering exit=0。

# slots=8 lane-stride group store direct 发射职责整改

本轮在 `OneToNVMIGroupStoreOpPattern::lowerSlots8LaneStride` 中引入
`emitAlignedSlots8LaneStride`，将 direct 路径每个 slot block 的 vreg 校验、active group
mask 构造和带 dist 的 `VstsOp` 发射抽取为独立 helper。主函数继续负责 dist/mask
granularity 选择、地址合法性判断、unaligned compact fallback 与 stateful stream；保持
block 顺序、offset、mask、dist token 和 direct/fallback 选择语义不变。增量
`check_changed_code.py` 结果为 `checked_files=1 errors=0 warnings=0`，`git diff --check`
通过；本轮未新增可独立覆盖该分支的 runtime case，未虚构额外回归结果。

# one-block group store 单 part 发射职责整改

本轮在 `OneToNVMIGroupStoreOpPattern::lowerOneBlockGroupStore` 中引入
`emitOneBlockGroupStorePart`，将单个 physical part 的 vreg 校验、contiguous store mask、
group offset/base 计算和 `VsstbOp` 发射抽取为独立 helper。主函数继续负责 one-block plan、
arity 合约、stride 常量和 part 遍历；保持 part 顺序、block/repeat stride、地址计算、mask
语义、诊断和替换行为不变。增量 `check_changed_code.py` 结果为
`checked_files=1 errors=0 warnings=0`，`git diff --check` 通过；
`vmi_layout_assignment_group_store_slots1_unit_stride.pto` 完整 lowering pipeline exit=0。

# slots=1 group store point-store 单组发射职责整改

本轮在 `OneToNVMIGroupStoreOpPattern::lowerSlots1PointStores` 中引入
`emitSlots1PointStore`，将单个 group 的 vreg 校验、`PAT_VL1` mask 构造、group offset 计算
和 point-store `VstsOp` 发射抽取为独立 helper。主函数继续负责 point-store dist 能力
检查及 group 遍历；保持 row stride、地址偏移、mask、dist token、group 顺序和诊断语义不变。
增量 `check_changed_code.py` 结果为 `checked_files=1 errors=0 warnings=0`，`git diff --check`
通过；`vmi_layout_assignment_group_store_slots1_unit_stride.pto` 完整 lowering pipeline
exit=0。

# deinterleaved=2 group store low/high pair 发射职责整改

本轮在 `OneToNVMIGroupStoreOpPattern::lowerDeinterleaved2GroupStore` 中引入
`emitDeinterleaved2GroupStorePair`，将单个 low/high pair 的类型合同、all-true mask、
chunk offset 和 `Vstsx2Op` 发射抽取为独立 helper。主函数继续负责 chunk shape、dist 能力、
physical arity 及双层 group/chunk 遍历；保持 low/high 索引、INTLV dist、lane offset、
mask、pair 顺序和诊断语义不变。增量 `check_changed_code.py` 结果为
`checked_files=1 errors=0 warnings=0`，`git diff --check` 通过；本轮未新增可独立覆盖该
分支的 runtime case，未虚构额外回归结果。

# slots=1 packed group store 值构造职责整改

本轮在 `OneToNVMIGroupStoreOpPattern::lowerSlots1PackedUnitStride` 中引入
`buildPackedSlots1Value`，将每个 group 的 `VdupOp`、lane mask 构造与 `VselOp` 累积抽取
为独立 helper。主函数继续负责 mask 类型/all-mask 准备、地址对齐判断以及 aligned
`VstsOp` 或 unaligned stateful stream 分支；保持 group 顺序、LOWEST splat 语义、lane
选择、store mask、stream advance 和诊断语义不变。增量 `check_changed_code.py` 结果为
`checked_files=1 errors=0 warnings=0`，`git diff --check` 通过；
`vmi_layout_assignment_group_store_slots1_unit_stride.pto` 完整 lowering pipeline exit=0。

# one-block group-reduce 单结果构造职责整改

本轮在 `OneToNVMIGroupReduceOpPattern::lowerOneBlock` 中引入
`buildOneBlockGroupResult`，将单个 physical chunk 的 source/mask/result 类型合同检查与
`GroupReduceOp` 发射抽取为独立 helper。主函数继续负责 arity、基础 physical 类型获取、
结果遍历和替换；保持 source/mask 对应关系、结果顺序、诊断和 lowering 语义不变。增量
`check_changed_code.py` 结果为 `checked_files=1 errors=0 warnings=0`，`git diff --check`
通过；本轮未新增可独立进入该 helper 的 runtime case，未虚构额外回归结果。
# legacy group-slot extension 单结果发射职责整改

本轮在 `OneToNVMIExtIOpPattern::lowerLegacyGroupSlotExtension` 中引入
`buildLegacyGroupSlotExtensionResult`，将单 physical result 的结果类型校验、source
bitcast 和 `VcvtOp` 发射抽取为独立 helper。主函数继续负责 layout/width/arity 合约、
source lane 与 slot mask 准备及结果替换；保持 EVEN/P0 part、slot mask、结果顺序、诊断
和 lowering 语义不变。增量 `check_changed_code.py` 结果为
`checked_files=1 errors=0 warnings=0`，`git diff --check` 通过；
`vmi_to_vpto_group_slot_integer_extension_matrix.pto` lowering exit=0。

# packed-byte group store 块发射职责整改

本轮在 `OneToNVMIGroupStoreOpPattern::lowerPackedByteSlots8` 中引入
`emitPackedByteStoreBlocks`，将每个 32-group block 的 packed value 构造结果分派、直接
`PK4_B32` 发射以及 stateful stream value/advance 收集抽取为独立 helper。入口继续负责
uniform vreg、mask/index 准备、对齐单 part 快路径和 direct-memory 合法性判断；保持 block
顺序、active lane mask、group offset、PK4_B32 与 `vstus`/`vstas` 选择及失败传播语义不变。
增量 `check_changed_code.py` 结果为 `checked_files=1 errors=0 warnings=0`，
`git diff --check` 通过；`vmi_to_vpto_group_store_slots8_packed_byte.pto` 完整 lowering
pipeline exit=0，输出仍包含 `PK4_B32`、stateful `vstus` 和 `NORM_B8` 路径。

# deinterleaved=2 group reduce 类型合同职责整改

本轮在 `OneToNVMIGroupReduceOpPattern::lowerFullDeinterleaved2` 中引入
`Deinterleaved2GroupReduceTypes` 与 `getDeinterleaved2GroupReduceTypes`，将结果 vreg、
mask、source chunk、行归约结果及 combine mask 的物理类型合同集中校验和推导。主函数继续
负责 slots=1 结果约束、group/chunk arity、首 lane mask、归约构造和结果恢复；保持诊断顺序、
mask 语义、group 顺序及失败传播不变。增量 `check_changed_code.py` 结果为
`checked_files=1 errors=0 warnings=0`，`git diff --check` 通过；
`vmi_layout_assignment_group_reduce_s256.pto` 完整 lowering pipeline exit=0。

# structured load verifier 分派职责整改

本轮将 `verifySupportedVMIStructuredLoadOp` 中 deinterleave_load 与 stride_load 的重复
shape-check、reason 传播和 `WalkResult` 错误处理统一改用已有的
`verifySupportedShapeOp`。两个分支仍按原顺序识别，并保留各自完整的 VPTO 能力诊断文本；
不改变 checker、失败语义或后续 group-load 分支。增量 `check_changed_code.py` 结果为
`checked_files=1 errors=0 warnings=0`，`git diff --check` 通过；
`vmi_deinterleave_load_layout_propagation.pto` lowering exit=0。
`vmi_to_vpto_stride_load.pto` 在既有 pack/unpack invariant 处提前终止，未进入本轮 helper。

# gather 布局与源地址合同职责整改

本轮在 `checkSupportedGatherShape` 中引入 `checkGatherLayoutAndSource`，将四路 contiguous
布局检查与 `!pto.ptr` 源地址约束从元素类型/physical arity 合同中分离。主 checker 继续
负责 source element、index/mask/result 类型关系、B16/B32 选择及 physical chunk 限制，保持
检查顺序、诊断文本和失败语义不变。增量检查以 `HEAD` 为基线结果为
`checked_files=1 errors=0 warnings=0`，`git diff --check` 通过；尝试的 gather cases
在既有 VMI op verifier 的输入类型合同处提前失败，未进入本轮 helper。

# structured group load verifier 分派职责整改

本轮继续将 `verifySupportedVMIStructuredLoadOp` 中 group_load、group_slot_load 与
group_broadcast_load 的重复 shape-check 和失败诊断统一改用 `verifySupportedShapeOp`。
每个分支的能力说明、识别顺序、checker 选择和失败语义保持不变。增量
`check_changed_code.py` 结果为 `checked_files=1 errors=0 warnings=0`，`git diff --check`
通过；`vmi_layout_assignment_group_broadcast_load_e2b_b16.pto` lowering exit=0。
`vmi_layout_assignment_group_load_s16_unaligned_stride_invalid.pto` 仍按预期在既有
layout contract invalid 处失败，诊断未改变。

# channel merge 结果合同职责整改

本轮在 `checkSupportedChannelMergeShape` 中引入
`checkChannelMergeResultShape`，将 result layout、result physical arity 和输入/结果
arity 一致性校验抽取为独立 helper。主函数继续负责 channel 数量和所有输入 contiguous
layout/arity 汇总；保持 deinterleaved 期望布局、诊断顺序及失败语义不变。增量
`check_changed_code.py --base HEAD` 结果为 `checked_files=1 errors=0 warnings=0`，
`git diff --check` 通过；channel merge 正常样例在既有 layout contract 冲突处提前失败，
未进入本轮 helper。

# mask lane-stride 转换方向职责整改

本轮在 `materializeMaskLaneStrideLayout` 中引入 `MaskLaneStrideLayoutPlan` 与
`getMaskLaneStrideLayoutPlan`，将 contiguous↔lane-stride 的方向识别及 stride 参数准备
从具体 `Punpack/Ppack` 物化中分离。主函数继续负责 stride 合法性诊断和选择 unpack/pack
emitter，保持 factor=2/4 支持范围、结果顺序和失败语义不变。增量
`check_changed_code.py --base HEAD` 结果为 `checked_files=1 errors=0 warnings=0`，
`git diff --check` 通过；现有 mask layout 样例在既有 layout contract 或 pack/unpack
invariant 处提前终止，未进入本轮 helper。

# shuffle forwarding 单 chunk 映射职责整改

本轮在 `computeShuffleForwardingSourceParts` 中引入
`computeShuffleForwardingSourceChunk`，将单个 result physical chunk 的 padding/lane 映射、
same-lane 约束、source chunk 一致性和 flat-part 索引计算抽取为独立 helper。主函数继续
负责 source/result 类型前置条件、result factor/chunk 遍历和 source part 结果收集；保持
forwarding fallback 的优先级、结果顺序和所有失败诊断不变。增量
`check_changed_code.py --base HEAD` 结果为 `checked_files=1 errors=0 warnings=0`，
`git diff --check` 通过；`vmi_to_vpto_shuffle_forwarding.pto` lowering exit=0。

# shuffle vselr lane 方向职责整改

本轮在 `computeShuffleVselrPlanForChunk` 中引入 `getShuffleLaneDirection`，将单 lane 的
ASC/DESC affine 关系判定与错误诊断抽取为独立 helper。chunk plan 主函数继续负责 padding、
logical→physical 映射、source chunk 一致性、方向一致性和 flat index 组装；保持 vselr
fallback 的 plan 字段、结果顺序和失败语义不变。增量 `check_changed_code.py --base HEAD`
结果为 `checked_files=1 errors=0 warnings=0`，`git diff --check` 通过；
`vmi_to_vpto_shuffle_forwarding.pto` 与 `vmi_to_vpto_shuffle_lane0_splat.pto` 均 lowering
exit=0。

# mask granularity 分片物化职责整改

本轮在 `materializeAdjacentMaskGranularityConversion` 中引入
`materializeMaskGranularityParts`，将按 layout factor 遍历、source chunk offset 推进和
各 part 结果汇总抽取为独立 helper。主函数继续负责 conversion plan 构建、结果 arity
校验和最终返回；widening/narrowing 的 pack/unpack 语义、part 顺序及失败诊断保持不变。
增量 `check_changed_code.py --base HEAD` 结果为 `checked_files=1 errors=0 warnings=0`，
`git diff --check` 通过；尝试的 mask granularity cases 在既有 pack/unpack invariant 或
layout contract 处提前终止，未进入本轮 helper。

# mask layout materializer 分派职责整改

本轮在 `materializeMaskLayoutConversion` 中引入 `tryMaskLayoutMaterializer`，统一封装
identity、deinterleaved=2 和 dense lane-stride 三类 materializer 的 `FailureOr<optional>`
传播。入口保留 assigned-layout 前置检查、三类转换的原优先级以及最终 unsupported 诊断；
lambda 使用显式捕获，未改变 mask layout 或 padding 语义。增量
`check_changed_code.py --base HEAD` 结果为 `checked_files=1 errors=0 warnings=0`，
`git diff --check` 通过；`vmi_layout_assignment_masked_load.pto` 在既有 residual VMI
load 检查处终止，未进入本轮 materializer。

# data layout intermediate materializer 职责整改

本轮将 `materializeDataLayoutViaContiguous` 中两种 intermediate kind 的递归转换分别
抽取为 `materializeDataLayoutThroughContiguous` 与
`materializeDataLayoutThroughDeinterleaved`。原函数现在只负责 intermediate plan 查询、
空输入检查和 kind 分派；保持 contiguous/deinterleaved intermediate 的选择、递归转换
顺序、结果类型数量和失败传播语义不变。增量 `check_changed_code.py --base HEAD` 结果为
`checked_files=1 errors=0 warnings=0`，`git diff --check` 通过；
`vmi_to_vpto_ensure_layout_dense_composed.pto` lowering exit=0，deinterleaved case 在
既有 pack/unpack invariant 处提前终止。

# data layout conversion 路径尝试职责整改

本轮在 `materializeDataLayoutConversion` 中统一封装四种 materializer 的
`FailureOr<optional>` 尝试与失败传播，保持 simple → deinterleaved2 → lane-stride →
intermediate 的严格优先级。各 lambda 使用显式捕获；主函数仍负责最终 unsupported 诊断，
不改变布局转换结果、递归顺序或失败语义。增量 `check_changed_code.py --base HEAD` 结果为
`checked_files=1 errors=0 warnings=0`，`git diff --check` 通过；
`vmi_to_vpto_ensure_layout_dense_composed.pto` lowering exit=0。

# active-prefix index shape 合同职责整改

本轮将 `buildActivePrefixIndexShapePlan` 的校验拆为
`checkActivePrefixIndexLayouts` 与 `checkActivePrefixIndexPhysicalChunks`，分别负责
contiguous mask/result layout、full physical chunk 证明以及单 physical chunk arity 合同。
plan builder 继续负责物理类型提取和最终 plan 组装，保持原诊断文本、校验顺序和失败语义
不变。增量 `check_changed_code.py --base HEAD` 结果为 `checked_files=1 errors=0 warnings=0`，
`git diff --check` 通过；active-prefix 正常样例在既有 pack/unpack invariant 处提前终止。

# channel split 结果布局与 arity 合同职责整改

本轮在 `checkSupportedChannelSplitShape` 中引入
`checkChannelSplitResultShape`，将每个 channel result 的 contiguous layout 检查、物理
arity 汇总与 source/result arity 一致性校验抽取为独立 helper。主函数继续负责 channel 数
量、source layout 和期望的 deinterleaved layout 判断；保持诊断顺序、结果顺序及失败语义
不变。增量检查以 `HEAD` 为基线结果为 `checked_files=1 errors=0 warnings=0`，
`git diff --check` 通过；`vmi_to_vpto_channel_split_layout_invalid.pto` 仍按预期在原有
layout contract 处失败。

# scalar store verifier 分派职责整改

本轮将 `verifySupportedVMIMemoryStoreOp` 中普通 `store` 的重复 shape-check 与错误诊断
改为复用 `verifySupportedShapeOp`，新增 `checkSupportedVMIStoreShape` 仅负责把 operation
operand 适配到既有 `checkSupportedStoreShape`。structured store（包括 masked_store）的
分派顺序和特殊诊断保持不变。增量 `check_changed_code.py` 结果为
`checked_files=1 errors=0 warnings=0`，`git diff --check` 通过；
`vmi_to_vpto_group_store_slots1_1pt.pto` lowering exit=0。

# contiguous group reduce 类型合同职责整改

本轮在 `OneToNVMIGroupReduceOpPattern::lowerContiguousRows` 中引入
`ContiguousGroupReduceTypes` 与 `getContiguousGroupReduceTypes`，将结果 vreg、mask、
source chunk、行归约结果及 combine mask 的物理类型合同集中校验和推导。主函数继续负责
contiguous chunk shape、结果布局/arity、首 lane mask、归约构造和结果恢复；保持 slots=1
与重复结果布局语义、诊断顺序及失败传播不变。增量 `check_changed_code.py` 结果为
`checked_files=1 errors=0 warnings=0`，`git diff --check` 通过；
`vmi_layout_assignment_group_reduce_s256.pto` 完整 lowering pipeline exit=0。

# stateful offset 范围边界转换职责整改

本轮将 `getStatefulOffsetRange` 内嵌的 APInt 上下界转换 lambda 抽取为
`convertFiniteRangeBound`，集中处理有符号/无符号解释和 int64 可表示性边界。范围分析、
循环归纳变量查找、非负性校验和失败诊断保持不变；显式捕获与控制流大括号要求也得到统一。
增量 `check_changed_code.py` 结果为 `checked_files=1 errors=0 warnings=0`，
`git diff --check` 通过。

# zero-copy interleave 结果合同职责整改

本轮在 `OneToNVMIInterleaveOpPattern::materializeZeroCopyResults` 中引入
`validateZeroCopyResultParts`，将 low/high result type 合并、结果 arity 校验和 physical
part 类型校验抽取为独立 helper。主函数继续负责 vintlv/vdintlv 的 zero-copy 重排；保持
结果顺序、slice 布局、诊断及失败语义不变。增量 `check_changed_code.py` 结果为
`checked_files=1 errors=0 warnings=0`，`git diff --check` 通过；
`vmi_to_vpto_vdintlv.pto` lowering exit=0（仅有 unified op 无 legacy equivalent 的既有
remark）。

# ExtI dense lane-stride 单结果发射职责整改

本轮在模板 `OneToNVMIExtIOpPattern::emitDenseLaneExtension` 中引入
`buildDenseLaneExtensionResult`，将单 source/result pair 的 `VcvtOp` 发射抽取为独立 helper。
主函数继续负责 all-true mask 构造、source/result 遍历和结果替换；保持 lane-stride part、
mask、结果顺序、诊断及 lowering 语义不变。增量 `check_changed_code.py` 结果为
`checked_files=1 errors=0 warnings=0`，`git diff --check` 通过；
`vmi_to_vpto_group_slot_integer_extension_matrix.pto` lowering exit=0。

# ExtI factor conversion 单结果发射职责整改

本轮在模板 `OneToNVMIExtIOpPattern::emitFactorExtension` 中引入
`buildFactorExtensionResult`，将单个 source chunk/factor part 的 `VcvtOp` 发射抽取为独立
helper。主函数继续负责 part-major/chunk-major 结果遍历、physical result type 选择和结果
替换；保持 part 顺序、mask、结果扁平顺序、诊断及 lowering 语义不变。增量
`check_changed_code.py` 结果为 `checked_files=1 errors=0 warnings=0`，`git diff --check`
通过；`vmi_to_vpto_group_slot_integer_extension_matrix.pto` lowering exit=0。

# ExtI factor part plan 职责整改

本轮在模板 `OneToNVMIExtIOpPattern::lowerPhysicalExtension` 中引入
`getExtensionPartPlan`，将 source/result width 关系、physical result arity 与 EVEN/ODD 或
P0/P1/P2/P3 part 表选择抽取为独立 helper。主函数继续保留 dense lane-stride 优先路径、
mask 构造和 factor extension 发射；保持分派优先级、part 顺序、诊断和 lowering 语义不变。
增量 `check_changed_code.py` 结果为 `checked_files=1 errors=0 warnings=0`，
`git diff --check` 通过；`vmi_to_vpto_group_slot_integer_extension_matrix.pto` lowering
exit=0。

# compress 结果 shape 合同职责整改

本轮新增 `checkSupportedCompressResultShape`，将 `compress` 的结果 layout 存在性、contiguous 合同、物理 arity 可计算性及单 chunk 约束从入口校验中独立出来。`buildCompressPhysicalShapePlan` 继续只负责 source/mask 的输入合同，`compress_store` 保持独立诊断和 destination `!pto.ptr` 合同；结果布局校验顺序、错误文本、物理 chunk 限制及 lowering 语义均保持不变。增量 `check_changed_code.py --base HEAD` 结果为 `checked_files=1 errors=0 warnings=0`，`git diff --check` 通过；`vmi_to_vpto_compress_store.pto` lowering exit=0，普通 compress case 在既有 pack/unpack invariant 处提前终止。

# reduce shape 合同分层整改

本轮将通用 reduce shape plan builder 中的 layout 合同和 physical arity 合同分别抽取为
`checkReduceLayouts` 与 `checkReducePhysicalArity`。builder 继续负责 source/mask/result
类型提取、full physical chunk 证明和 plan 组装；原有 contiguous 要求、source/mask arity
匹配、单 result chunk 约束、诊断顺序及失败传播保持不变。增量
`check_changed_code.py --base HEAD` 结果为 `checked_files=1 errors=0 warnings=0`，
`git diff --check` 通过。native 增量构建仍被工作区既有 CMake 外部依赖尝试创建
`/cann-cmake` 的权限错误阻断，未产生本轮 C++ 编译诊断。

# group broadcast 结果 shape 合同职责整改

本轮新增 `checkGroupBroadcastResultShape`，将 group broadcast 的结果 physical chunk 完整性、
factor=1 快路径、block/deinterleaved 小 group 例外及 logical span 合同从入口分派中抽取。
plan builder 仍负责输入类型/layout、组数、lanes-per-part 和 group size 推导；原有检查顺序、
诊断文本、结果布局例外与 lowering 语义保持不变。增量检查结果为
`checked_files=1 errors=0 warnings=0`，`git diff --check` 通过；E2B group-broadcast
case lowering exit=0。

# group slot load 结果类型职责整改

本轮新增 `GroupSlotLoadResultPart` 与 `getGroupSlotLoadResultPart`，统一封装 slots=1/8
路径重复的物理 vreg 类型和 mask 类型推导。两个 lowering helper 继续分别负责 unit-stride
slots=8 的 BRC/VS LDB 发射和 slots=1 的单 block VS LDB 地址推进；分派、mask pattern、
offset 计算、结果顺序及失败诊断保持不变。增量
`check_changed_code.py --base HEAD` 结果为 `checked_files=1 errors=0 warnings=0`，
`git diff --check` 通过；现有 group-slot-load case 在既有 pack/unpack invariant 处
提前终止，未进入本轮 helper。

# dynamic create_group_mask shape 合同职责整改

本轮将 `buildDynamicGroupMaskPlan` 的校验拆为 `checkDynamicGroupMaskLayout` 与
`getDynamicGroupMaskPhysicalShape`：前者负责 layout、lane_stride、granularity 和
logical `num_groups * group_size` 合同，后者负责 physical mask granularity、lanes-per-part
及 result arity。plan builder 继续负责 factor/block 参数推导和 power-of-two block 约束；
动态 active 元素 clamp、lane index 构造、padding mask 处理、chunk 遍历顺序及诊断语义保持
不变。增量 `check_changed_code.py --base HEAD` 结果为 `checked_files=1 errors=0 warnings=0`，
`git diff --check` 通过；`vmi_to_vpto_create_group_mask_block8_dynamic.pto` lowering
exit=0。

# group broadcast shape 重构回归修复

复核既有 group-broadcast shape 拆分时发现入口遗漏了结果类型和诊断闭包定义，导致该
路径无法通过 C++ 编译。现已补齐 `resultType` 的操作数类型提取及局部 `fail` 诊断 helper，
不改变任何 shape 判定、错误文本或 lowering 分派。增量
`check_changed_code.py --base HEAD` 结果为 `checked_files=1 errors=0 warnings=0`，
`git diff --check` 通过；`vmi_layout_assignment_group_broadcast_load_e2b_b16.pto`
完整 lowering pipeline exit=0。

# mask physical part type 构造职责整改

本轮新增 `repeatMaskPartType`，将按 physical arity 重复构造 `MaskType` part 列表的职责
从 `getConvertedMaskPartTypes` 中独立出来；入口继续负责 arity/granularity 合同查询和
physical part type 选择。补齐循环大括号并保持 part 数量、类型和失败语义不变。增量
`check_changed_code.py --base HEAD` 结果为 `checked_files=1 errors=0 warnings=0`，
`git diff --check` 通过；direct/multistep mask granularity lowering case 均 exit=0。

# data layout materialization context 数据泥团整改

本轮引入 `DataLayoutMaterializationContext` 与 `tryDataLayoutMaterializer`，统一承载
data-layout conversion 四路 materializer 共同使用的 operation、source/result parts、
layout、source element type 和 rewriter。入口仍严格按 simple → deinterleaved2 →
lane-stride → intermediate 顺序尝试，保持 optional/failure 传播、递归转换和 unsupported
诊断不变；lambda 改为显式接收 context，避免重复默认捕获和参数泥团。增量
`check_changed_code.py --base HEAD` 结果为 `checked_files=1 errors=0 warnings=0`，
`git diff --check` 通过；`vmi_to_vpto_ensure_layout_dense_composed.pto` 与 mask
conversion case lowering 均 exit=0。

# mask layout materialization context 数据泥团整改

本轮引入 `MaskLayoutMaterializationContext`，统一承载 identity、deinterleaved2 和
lane-stride 三路 mask layout materializer 共享的 operation、parts、source/result layout
及 rewriter。`materializeMaskLayoutConversion` 仍按 identity → deinterleaved2 →
lane-stride 顺序分派，optional/failure 传播、identity forwarding 合同和 unsupported
诊断保持不变；lambda 改为显式接收 context，避免重复参数捕获与默认捕获风险。增量
`check_changed_code.py --base HEAD` 结果为 `checked_files=1 errors=0 warnings=0`，
`git diff --check` 通过；layout 和 mask granularity 相关 lowering case 均 exit=0。

# deinterleaved=2 mask layout 路由职责整改

本轮新增 `Deinterleaved2MaskLayoutDirection` 与
`getDeinterleaved2MaskDirection`，将 source/result layout 的方向识别从
`materializeDeinterleaved2MaskLayout` 中独立出来。物化函数现在只负责 arity、identity
part forwarding 和方向对应的 intlv/dintlv 发射；to-contiguous/from-contiguous 的判定、
结果顺序及失败诊断保持不变。增量 `check_changed_code.py --base HEAD` 结果为
`checked_files=1 errors=0 warnings=0`，`git diff --check` 通过；dense layout 与 vdintlv
相关 case lowering 均 exit=0。

# predicate interleave 分派职责整改

本轮引入 `PredicateInterleaveKind` 与 `createPredicateInterleave`，统一 b8/b16/b32
predicate 类型检查及 `pintlv`/`pdintlv` 指令选择；原有 `createPredicateIntlv` 与
`createPredicateDintlv` 保留为语义清晰的薄包装。保持 low/high 返回顺序、mask 类型合同、
unsupported 类型失败语义及所有 layout materialization 调用路径不变。增量
`check_changed_code.py --base HEAD` 结果为 `checked_files=1 errors=0 warnings=0`，
`git diff --check` 通过；dense layout、mask granularity 和 vdintlv 相关 case lowering
均 exit=0。

# mask physical carrier layout 控制流整改

本轮清理 `getVMIMaskPhysicalCarrierLayout` 与
`getVMIMaskPhysicalCarrierType` 中的无大括号控制语句，并将 physical carrier 缺失条件
命名为 `missingPhysicalCarrier`。layout kind 到 contiguous/deinterleaved/
block-deinterleaved/group-slots carrier 的映射、granularity 传播和失败语义保持不变。
增量 `check_changed_code.py --base HEAD` 结果为 `checked_files=1 errors=0 warnings=0`，
`git diff --check` 通过；vdintlv case 在既有 pack/unpack invariant 处提前终止，未进入
本轮 carrier helper。

# mask lane-stride factor 校验职责整改

本轮新增 `checkMaskLaneStrideFactor`，将 lane-stride route 的 factor 合法性（仅支持 2/4）
及方向相关诊断从 `materializeMaskLaneStrideLayout` 中独立出来。入口现在只负责 route
查询、factor 校验和 pack/unpack 分派；结果 arity、pack/unpack 指令序列、mask 合并及失败
传播保持不变。增量 `check_changed_code.py --base HEAD` 结果为
`checked_files=1 errors=0 warnings=0`，`git diff --check` 通过；layout 和 mask
granularity lowering case 均 exit=0。

# mask lane-stride 结果类型校验职责整改

本轮新增 `getMaskLaneStrideResultType`，统一 pack/unpack 两条路径对 physical result
`MaskType` 的校验与诊断。实际的 source index、lane-stride part 选择、punpack/ppack
序列、mask merge 和结果 arity 检查保持在各自路径中，未改变 lowering 语义。增量
`check_changed_code.py --base HEAD` 结果为 `checked_files=1 errors=0 warnings=0`，
`git diff --check` 通过；layout 与 mask granularity lowering case 均 exit=0。

# mask granularity cast layout 中间路径职责整改

本轮新增 `materializeMaskGranularityCastThroughLayout`，将 physical granularity 转换后
再执行 layout conversion 的两阶段路径独立出来。`materializeMaskGranularityCastParts`
现在只负责判断 physical layout 是否相同并选择 direct 或 through-layout 路径；保持
physical carrier 类型、granularity/layout 传播、转换顺序、结果 arity 及失败诊断不变。
增量 `check_changed_code.py --base HEAD` 结果为 `checked_files=1 errors=0 warnings=0`，
`git diff --check` 通过；mask granularity direct/multistep 与 dense layout case 均 exit=0。

# staging mask factor 路由职责整改

本轮新增 `getMaskStagingDirection`，将 factor=2/4 staging 路径的 contiguous 与
deinterleaved layout 方向识别从 `materializeMaskGranularityCastStagingLayout` 中独立出来。
入口现在只负责按 factor 顺序尝试并调用对应 materializer；deint→contiguous 与
contiguous→deint 的发射语义、factor 优先级、结果顺序及失败传播保持不变。增量
`check_changed_code.py --base HEAD` 结果为 `checked_files=1 errors=0 warnings=0`，
`git diff --check` 通过；mask granularity direct/multistep 和 dense layout case 均 exit=0。

# mask granularity fallback 路由职责整改

本轮新增 `requiresMaskDenseSplitFallback`，将 dense-split contiguous fallback 的触发条件
从 `materializeMaskGranularityCastLayoutFallback` 中独立出来。fallback 入口仍严格保持
direct layout conversion → staging factor route → dense-split contiguous conversion 的
尝试顺序；direct conversion 失败后继续尝试的语义、optional/failure 传播和结果顺序不变。
增量 `check_changed_code.py --base HEAD` 结果为 `checked_files=1 errors=0 warnings=0`，
`git diff --check` 通过；mask granularity direct/multistep case 均 exit=0。

# staging mask part 累积职责整改

本轮引入 `StagingMaskPartAccumulator`，将 contiguous→deinterleaved staging mask 转换中
physical part 容器的初始化、每组结果的 arity 校验与追加、以及 part-major 扁平化集中到
一个有明确所有权的累积对象。`materializeStagingContiguousToDeintMaskLayout` 现在只负责
factor/group 合同、逐组调用具体的 factor=2/4 物化逻辑和最终结果返回；没有把 factor=2
与 factor=4 的 `pdintlv` 语义强行合并，仍保持源 part 补齐、group 顺序、part-major 结果
顺序、结果 arity 诊断及失败传播不变。增量
`check_changed_code.py --base HEAD` 结果为 `checked_files=1 errors=0 warnings=0`，
`git diff --check` 通过。`vmi_to_vpto_ensure_mask_granularity_multistep.pto` 尝试
lowering 时在既有 VMI pack/unpack pipeline invariant 处提前失败，未进入本轮 staging
helper，不能将该失败归因于本轮改动。

# group store 布局分派职责整改

本轮将 `OneToNVMIGroupStoreOpPattern::matchAndRewrite` 中的布局分类与后端选择抽取为
`lowerByLayout`。入口现在只负责 destination/offset/row_stride 的单值归一化并保留统一
scalar `group_store` 的注释上下文，helper 负责 compact-small、slots=1、slots=8、one-block、
deinterleaved=2 与 contiguous 路径的优先级分派。support-table 查询、deinterleaved shape
探测、失败诊断和所有具体 lowering 的参数/结果语义均保持不变。增量
`check_changed_code.py --base HEAD` 结果为 `checked_files=1 errors=0 warnings=0`，
`git diff --check` 通过；`vmi_layout_assignment_group_store_slots1_unit_stride.pto`
完整 lowering pipeline exit=0。

# verifier 与 shuffle 规划控制流整改（2026-09-06）

本轮继续清理 verifier/shape-plan 与 shuffle 规划区域：为 reduce physical chunk 检查、
Vdhist/Vchist/Vmull 及 add-carry 端口校验补齐选择语句大括号，并将 Vmull/add-carry 的
关键失败条件命名，保持原有支持范围和诊断顺序不变；同时为 shuffle forwarding 规划的
lanes-per-part、indices、layout factor 和 physical chunk 查询补齐大括号，避免单语句控制
流在后续维护中产生歧义。未改变物理 lane 映射、chunk 顺序或 lowering 结果。

本轮验证：

```text
git diff --check                                      # passed
check_changed_code.py --base origin/master            # checked_files=1 errors=0 warnings=0
vmi_layout_assignment_group_broadcast_load_e2b_b16.pto # exit=0
vmi_layout_assignment_scatter.pto                      # exit=0
```

# shuffle splat 与 VSEL 规划控制流整改（2026-09-06）

本轮继续为 shuffle lane-0 splat 和 VSEL 规划补齐显式控制流边界，并将“存在非零索引”
条件命名。source lane 映射、物理 part/chunk 枚举、失败诊断及结果计划顺序均保持不变，
仅消除单语句控制流歧义。

本轮验证：

```text
git diff --check                                      # passed
check_changed_code.py --base origin/master            # checked_files=1 errors=0 warnings=0
```

随后将 `computeShuffleVselrPlanForChunk` 中的结果物理 lane 校验、逻辑 lane 越界检查和
source lane 映射抽取为 `getShuffleSourceLane`。规划函数继续负责 source chunk 一致性与
ASC/DESC 方向推导，抽取没有改变失败诊断、迭代顺序或最终 `ShuffleVselrPlan` 内容；增量
检查仍为 `errors=0 warnings=0`，scatter lowering case exit=0。

随后补齐 shuffle VSEL 规划失败诊断 lambda 的大括号；该修改仅满足控制流可读性约束，
不改变诊断文本或失败传播。

# group store physical shape 职责拆分（2026-09-06）

本轮将 `checkSupportedGroupStoreByLayout` 中通用 physical shape 合同校验抽取为
`checkSupportedGroupStorePhysicalShape`。布局分派 helper 现在只负责 compact-small、
group-slots 和通用 layout-fact 路由；physical helper 负责 destination shape、one-block
plan、full group chunk 与 deinterleaved=2 fallback 的顺序。支持范围、失败诊断和原有
优先级保持不变。

本轮验证：

```text
git diff --check                                      # passed
check_changed_code.py --base origin/master            # checked_files=1 errors=0 warnings=0
vmi_layout_assignment_group_store_slots1_unit_stride.pto # exit=0
```

# mask layout materializer 路由职责拆分（2026-09-06）

本轮将 `materializeMaskLayoutConversion` 中 identity → deinterleaved=2 → lane-stride
的顺序尝试抽取为 `tryMaskLayoutMaterializers`。入口现在只负责布局存在性检查、context
构造和最终 unsupported 诊断；materializer 的 optional/failure 传播、优先级、结果顺序
均保持不变，未改变任何 mask layout 指令生成。

本轮验证：

```text
git diff --check                                      # passed
check_changed_code.py --base origin/master            # checked_files=1 errors=0 warnings=0
vmi_to_vpto_ensure_mask_granularity.pto                # exit=0
```

# truncf 合同判断职责拆分（2026-09-06）

本轮将 `OneToNVMITruncFOpPattern::matchAndRewrite` 中 packed-fp 合同判断和 group-slot
布局识别分别抽取为 `hasUnsupportedPackedTruncFConversion` 与
`hasGroupSlotTruncFLayouts`。入口继续负责结果类型转换和后续 same-width、lane-stride、
factor narrowing 分派；packed carrier 支持矩阵、group-slot 优先级及所有 lowering 语义
保持不变。

本轮验证：

```text
git diff --check                                      # passed
check_changed_code.py --base origin/master            # checked_files=1 errors=0 warnings=0
vmi_to_vpto_truncf_bf16x2_d4_packed4.pto               # exit=0
```

# scatter shape 类型上下文整改（2026-09-06）

本轮引入 `ScatterShapeTypes`，集中承载 scatter 的 value、indices 和 mask 物理类型，
并由 `getScatterShapeTypes` 统一执行类型提取。`checkSupportedScatterShape` 现在只负责编
排验证顺序：布局/目的地址合同 → 元素合同 → physical chunk 合同；没有改变 arity、类型
支持矩阵、失败诊断或 physical lowering 语义。

本轮验证：

```text
git diff --check                                      # passed
check_changed_code.py --base origin/master            # checked_files=1 errors=0 warnings=0
vmi_layout_assignment_scatter.pto                      # exit=0
```

# group broadcast shape 合同分层整改（2026-09-06）

本轮将 `buildGroupBroadcastShapePlan` 中 source/result 基础合同、num_groups、布局类型及
layout-support 查询抽取为 `checkGroupBroadcastLogicalContract`。plan builder 现在只负责
physical lanes、group size 和 result factor 推导；结果 chunk 合同仍由既有
`checkGroupBroadcastResultShape` 负责。该拆分保持 group size 推导、布局优先级、失败诊断
和 lowering 结果不变。

本轮验证：

```text
git diff --check                                      # passed
check_changed_code.py --base origin/master            # checked_files=1 errors=0 warnings=0
vmi_layout_assignment_group_broadcast_load_e2b_b16.pto # exit=0
```

# integer extension source 合同职责拆分（2026-09-06）

本轮将 `OneToNVMIExtIOpPattern::matchAndRewrite` 中 source physical chunks 的非空、类型
可转换及跨 chunk 一致性检查抽取为 `getUniformExtensionSourceType`。入口继续负责 VMI
布局分派、result 类型收集及 factor/lane-stride lowering；source 类型合同、失败诊断和
物理结果顺序保持不变。

本轮验证：

```text
git diff --check                                      # passed
check_changed_code.py --base origin/master            # checked_files=1 errors=0 warnings=0
vmi_to_vpto_reduce_extended.pto                       # exit=0
```

# trunci physical 类型与路径条件整改（2026-09-06）

本轮将 `OneToNVMITruncIOpPattern::matchAndRewrite` 中 source/result physical vreg 的非空、
整数元素类型和跨 chunk 一致性检查抽取为 `getUniformTruncTypes`；同时为 exti/truncf
入口中触及的 dense lane-stride、same-width 条件命名，避免复杂布尔表达式直接控制分派。
不改变 s32→s8 alias、NOSAT carrier、factor=2/4 narrowing 的路径优先级或结果顺序。

本轮验证：

```text
git diff --check                                      # passed
check_changed_code.py --base origin/master            # checked_files=1 errors=0 warnings=0
vmi_to_vpto_truncf_bf16x2_d4_packed4.pto               # exit=0
```

# fp-to-int source 校验复用（2026-09-06）

本轮将 `fptosi` 与 `fptoui` 两个 pattern 重复的 source physical chunks 校验统一为模板
helper `validateFpToIntSourceParts`；helper 保留各操作独立的空输入、期望类型和类型不一
致诊断文本，调用方仍负责各自的元素合同、宽窄转换及 part 规划。未改变转换支持矩阵、
mask 语义或结果顺序。

本轮验证：

```text
git diff --check                                      # passed
check_changed_code.py --base origin/master            # checked_files=1 errors=0 warnings=0
```

`vmi_to_vpto_fptosi_*` 代表性输入在既有 VMI pack/unpack pipeline invariant 处提前失败，
未进入本轮 helper，不能将该既有测试前置失败归因于本次重构。

# fp-to-int result 校验复用（2026-09-06）

本轮进一步将 `fptosi` 与 `fptoui` 重复的 result physical chunk 非空及 `VRegType` 校验
统一为 `validateFpToIntResultParts`。helper 通过参数保留两种操作各自的诊断文本，调用方
仍分别负责转换合同和 lowering 路径；不改变 result arity、mask 或 part 顺序。

本轮验证：

```text
git diff --check                                      # passed
check_changed_code.py --base origin/master            # checked_files=1 errors=0 warnings=0
```

`vmi_to_vpto_fptoui_f16_to_u8.pto` 同样在既有 pack/unpack pipeline invariant 处提前失败，
未进入本轮 helper。

# data layout materializer 路由收敛（2026-09-06）

本轮将 `materializeDataLayoutConversion` 中四路 materializer 的重复 optional/failure
处理收敛到 `tryDataLayoutMaterializers`。helper 严格保持 simple → deinterleaved2 →
lane-stride → via-contiguous 的尝试顺序；入口只负责 context 构造、最终结果转移和
unsupported 诊断，不改变递归转换、结果 arity 或失败传播。

本轮验证：

```text
git diff --check                                      # passed
check_changed_code.py --base origin/master            # checked_files=1 errors=0 warnings=0
vmi_to_vpto_ensure_layout_dense_composed.pto           # exit=0
```

# adjacent mask granularity 结果合同拆分（2026-09-06）

本轮将 `materializeAdjacentMaskGranularityConversion` 中 result physical arity/count 校验
抽取为 `checkMaskGranularityResultArity`。相邻 granularity 的 plan 构造、逐 layout part
物化、结果扁平化和失败传播保持原顺序；入口只负责连接 plan、物化和结果合同，未改变
ppack/punpack/por 指令语义。

本轮验证：

```text
git diff --check                                      # passed
check_changed_code.py --base origin/master            # checked_files=1 errors=0 warnings=0
vmi_to_vpto_ensure_mask_granularity.pto                # exit=0
```

# expand-load all-active 路径判断拆分（2026-09-06）

本轮将 `checkSupportedExpandLoadShape` 中 static all-active mask、full physical chunk 与
read-safety proof 的组合判断抽取为 `hasSafeExpandLoadAllActivePath`。主 checker 继续负
责 common shape、runtime-mask fallback 与诊断拼接；full-read 安全条件、fallback reason
和 runtime path 选择语义保持不变。

本轮验证：

```text
git diff --check                                      # passed
check_changed_code.py --base origin/master            # checked_files=1 errors=0 warnings=0
```

`vmi_to_vpto_expand_load_all_active.pto` 在既有 VMI pack/unpack pipeline invariant 处提
前失败，未进入本轮 helper。

# masked-store contiguous materialization 职责拆分（2026-09-06）

本轮将 `checkMaskedStoreLayoutAndArity` 中 value/mask contiguous materialization part
计数、失败原因和 arity 一致性检查抽取为
`checkMaskedStoreContiguousMaterialization`。layout/physical arity 与 full-chunk 检查
仍由原 helper 负责，full-chunk 快路径、tail materialization 合同和诊断传播保持不变。

本轮验证：

```text
git diff --check                                      # passed
check_changed_code.py --base origin/master            # checked_files=1 errors=0 warnings=0
vmi_to_vpto_masked_store.pto                           # exit=0
```

# stride memory 合同复用（2026-09-06）

本轮将 `checkSupportedStrideLoadShape` 与 `checkSupportedStrideStoreShape` 中重复的布局、
contiguous、UB pointer 和单 physical value/mask chunk 校验统一为
`checkStrideMemoryContract`；store 入口保留其额外的 `checkSupportedStoreShape`，load/store
仍使用各自的 source/destination 方向诊断。该重构不改变 `vsldb`/`vsstb` 支持范围或 arity
语义。

本轮验证：

```text
git diff --check                                      # passed
check_changed_code.py --base origin/master            # checked_files=1 errors=0 warnings=0
```

`vmi_lane_stride_masked_store.pto` 在测试自身已有的 mask layout contract 冲突处提前失败，
未进入本轮 stride helper。

# stride memory contract 上下文整改（2026-09-06）

本轮复核并收敛上一轮的 stride helper：将 data/mask 类型、layout、pointer 和各方向诊断
封装为 `StrideMemoryContract`，移除 load/store 入口中不再使用的失败 lambda，并统一由
`checkStrideMemoryContract` 消费。该上下文只表达 stride 访存合同，不引入全局状态；
`checkSupportedStoreShape` 仍在 store 路径单独执行，load/store 的 pointer 方向和 arity
诊断保持不变。

本轮验证：

```text
git diff --check                                      # passed
check_changed_code.py --base origin/master            # checked_files=1 errors=0 warnings=0
vmi_to_vpto_stride_store.pto                           # exit=0
```

`vmi_to_vpto_stride_load.pto` 在既有 VMI pack/unpack pipeline invariant 处提前失败，未进入
本轮 stride helper。

# active-prefix physical arity 合同拆分（2026-09-06）

本轮将 active-prefix index shape plan 中 single-physical-chunk 的 arity 合同抽取为
`checkActivePrefixIndexSingleChunk`，使 layout 合同、full physical chunk 合同和跨 chunk
carry 限制分别表达。原有 mask/result layout 要求、padding lane 安全条件和失败诊断顺序
保持不变；同时补齐本轮触及的控制流大括号。

本轮验证：

```text
git diff --check                                      # passed
check_changed_code.py --base origin/master            # checked_files=1 errors=0 warnings=0
```

active-prefix/compress 代表性测试在既有 VMI pack/unpack pipeline invariant 处提前失败，未
进入本轮 helper。

# compress-store destination 合同拆分（2026-09-06）

本轮将 `checkSupportedCompressStoreShape` 中 `!pto.ptr` destination 检查抽取为
`checkCompressStoreDestination`，使 compress physical plan 合同与 `vstur` pointer 合同
分离。原有 source/mask full-chunk、单 physical chunk 限制、SQZN 相关诊断和失败顺序保持
不变。

本轮验证：

```text
git diff --check                                      # passed
check_changed_code.py --base origin/master            # checked_files=1 errors=0 warnings=0
vmi_to_vpto_compress_store.pto                         # exit=0
```

# reduce source full-chunk 合同拆分（2026-09-06）

本轮将 `buildReducePhysicalShapePlan` 中 source full physical chunk/padding lane 安全检查
抽取为模板 helper `checkReduceSourceChunks`。plan builder 继续负责 layout 与 physical
arity plan 组合；full-chunk 失败诊断、mask/source arity 合同及 result chunk 要求保持不变。

本轮验证：

```text
git diff --check                                      # passed
check_changed_code.py --base origin/master            # checked_files=1 errors=0 warnings=0
vmi_group_reduce_addi_i16.pto                          # exit=0
```

# load lane-stride dist 选择职责拆分（2026-09-06）

本轮将 `OneToNVMILoadOpPattern::matchAndRewrite` 中 lane-stride dist 查询、首个 physical
result 类型检查和 direct-memory 合法性判断抽取为 `getLoadLaneStrideDist`。入口继续保持
build plan → lane-stride → deinterleaved → contiguous 的 lowering 优先级，dist token 的
生命周期、地址合同和结果顺序均不变。

本轮验证：

```text
git diff --check                                      # passed
check_changed_code.py --base origin/master            # checked_files=1 errors=0 warnings=0
vmi_to_vpto_load_store_contiguous.pto                  # exit=0
```

# packed-byte slots=8 selector 构造拆分（2026-09-06）

本轮将 `lowerPackedByteSlots8` 中 packed-byte mask 类型、all-true mask、slot index
向量构造抽取为 `buildPackedByteSelectors`。对齐单 part 快路径、`PK4_B32` direct 判定、
block 合并和 stateful stream fallback 保持不变；失败诊断和 selector 类型保持一致。

本轮验证：

```text
git diff --check                                      # passed
check_changed_code.py --base origin/master            # checked_files=1 errors=0 warnings=0
vmi_to_vpto_group_store_slots8_packed_byte.pto         # exit=0
```

# interleave lowering 入口合同与布局查询拆分（2026-09-06）

本轮将模板 `OneToNVMIInterleaveOpPattern::matchAndRewrite` 中的结果物理类型/输入输出
arity 合同检查抽取为 `getInterleaveResultTypes`，并将 vintlv/vdintlv 布局事实查询抽取为
`getInterleaveLayoutFact`。入口继续保持 lane-stride、contiguous 与 zero-copy 的分派顺序，
结果替换、失败传播及诊断语义不变。

本轮验证：

```text
git diff --check                                      # passed
check_changed_code.py --base origin/master            # checked_files=1 errors=0 warnings=0
vmi_to_vpto_vintlv.pto                                 # blocked by pre-existing VMI-PASS-INVARIANT
```

# slots=8 group-store 输入合同拆分（2026-09-06）

本轮将 `lowerSlots8Dispatch` 中的物理 value arity 与首个 vreg 类型检查抽取为
`getSlots8FirstVRegType`。分派入口继续负责 constant unit `row_stride` 检查，并保持
packed-byte、lane-stride、contiguous 三条 lowering 路径的顺序、诊断文本和失败传播不变；
空输入仍保留原有后续分派行为。

本轮验证：

```text
git diff --check                                      # passed
check_changed_code.py --base origin/master            # checked_files=1 errors=0 warnings=0
vmi_to_vpto_group_store_slots8_packed_byte.pto         # exit=0
```

# interleave 布局路径分派职责拆分（2026-09-06）

本轮将模板 `OneToNVMIInterleaveOpPattern::matchAndRewrite` 中 lane-stride、contiguous
和 zero-copy 的布局关系判断与 lowering 路由抽取为 `lowerInterleaveByLayout`。入口只保留
操作数/结果类型获取、布局事实查询和统一调用；各路径的优先级、zero-copy 结果顺序、失败
诊断和结果替换保持不变。

本轮验证：

```text
git diff --check                                      # passed
check_changed_code.py --base origin/master            # checked_files=1 errors=0 warnings=0
vmi_layout_assignment_intlv.pto                        # exit=0
```

# group-broadcast slots=1 lane 映射拆分（2026-09-06）

本轮将 `materializeSlots1GroupBroadcastChunk` 中按物理 lane 映射 source chunk、校验布局
selector 关系并收集 active source 的逻辑抽取为 `mapSlots1GroupBroadcastSources`。原 helper
继续负责结果 mask 构造、Vdup/VSEL 合并和结果返回；padding lane、越界检查、诊断文本及
合并顺序保持不变。

本轮验证：

```text
git diff --check                                      # passed
check_changed_code.py --base origin/master            # checked_files=1 errors=0 warnings=0
vmi_layout_assignment_group_broadcast_load_e2b_b16.pto # exit=0
```

# group-slot-load 单组 BRC 路径拆分（2026-09-06）

本轮将 `lowerGroupSlotLoadSlots8` 中 `numGroups == 1` 的 BRC 结果类型、dist 查询和
`vlds` 构造抽取为 `lowerSingleGroupSlotLoad`。slots=8 多组 `vsldb` 路径、mask 构造、
offset 计算和支持矩阵保持不变；单组路径的诊断和结果顺序保持不变。

本轮验证：

```text
git diff --check                                      # passed
check_changed_code.py --base origin/master            # checked_files=1 errors=0 warnings=0
vmi_to_vpto_group_slot_load.pto                        # blocked by pre-existing VMI-PASS-INVARIANT
```

# group-slot-load slots=8 chunk 发射拆分（2026-09-06）

本轮将 `lowerGroupSlotLoadSlots8` 多组循环中的单个 chunk 合同校验、prefix mask、地址
偏移和 `vsldb` 发射抽取为 `emitGroupSlotLoadSlots8Chunk`。主函数只保留 unit-stride
合同、单组 BRC 分派和 chunk 遍历；多组 `vsldb` 的 mask/offset/结果顺序保持不变。

本轮验证：

```text
git diff --check                                      # passed
check_changed_code.py --base origin/master            # checked_files=1 errors=0 warnings=0
vmi_layout_assignment_group_slot_load.pto              # lowering output generated; command returned an existing test diagnostic
```

# group-slot truncation shape 合同拆分（2026-09-06）

本轮将 `lowerGroupSlotTrunc` 中 source/result logical bits、slots/group 数量、支持的
direct/packed 模式以及 physical arity 合同抽取为 `getGroupSlotTruncModes`。主 lowering
继续负责 active-slot mask、packed/direct 逐 part 发射和结果收敛；NOSAT/SAT 选择、carrier
处理、诊断文本及结果顺序保持不变。

本轮验证：

```text
git diff --check                                      # passed
check_changed_code.py --base origin/master            # checked_files=1 errors=0 warnings=0
vmi_layout_assignment_trunci_lane_stride.pto           # exit=0
```

# vmull logical shape 与 physical arity 拆分（2026-09-06）

本轮将 `buildVmullShapePlan` 中 element type、lane 数量、layout 和 mask 合同抽取为
`validateVmullLogicalShape`；原函数继续负责各端口 physical arity 计算与一致性检查。
`VmullShapePlan` 的 data/layout/arity 内容、失败诊断和支持矩阵保持不变。

本轮验证：

```text
git diff --check                                      # passed
check_changed_code.py --base origin/master            # checked_files=1 errors=0 warnings=0
vmi_to_vpto_vmull_deinterleaved.pto                    # exit=0
```

# group-broadcast selector 物化拆分（2026-09-06）

本轮将 `getGroupBroadcastSelector` 中 constant selector 与共享 ramp 的构造逻辑分别抽取
为 `materializeConstantGroupBroadcastSelector` 和 `materializeGroupBroadcastRamp`。缓存、
base index 偏移、shift 顺序以及 selector kind 分派保持不变，避免在入口混合不同 selector
物化策略。

本轮验证：

```text
git diff --check                                      # passed
check_changed_code.py --base origin/master            # checked_files=1 errors=0 warnings=0
vmi_layout_assignment_group_broadcast_load_e2b_b16.pto # exit=0
```

# block-deinterleaved group-load shape 合同拆分（2026-09-06）

本轮将 `lowerBlockF32` 中 group size/factor、num_groups、row_stride、pointer 类型等
入口 shape 合同抽取为 `validateBlockF32Shape`。主函数继续负责结果类型获取、block/chunk
均匀性检查和实际 lowering；原有检查顺序、诊断语义及 block-deinterleaved 发射保持不变。

本轮验证：

```text
git diff --check                                      # passed
check_changed_code.py --base origin/master            # checked_files=1 errors=0 warnings=0
vmi_to_vpto_load_store_contiguous.pto                  # exit=0
```

# dense group-slot extension 结果合同拆分（2026-09-06）

本轮将 `buildDenseGroupSlotExtensionResult` 中 source/result physical lane 数量、carrier
比例和结果 vreg 类型检查抽取为 `validateDenseGroupSlotExtensionResult`。主函数继续负责
逐级 `Vsunpack/Vzunpack` 物化和最终 bitcast；扩展方向、lane 校验、失败诊断及结果语义保持
不变。

本轮验证：

```text
git diff --check                                      # passed
check_changed_code.py --base origin/master            # checked_files=1 errors=0 warnings=0
vmi_to_vpto_group_slot_integer_extension_matrix.pto   # exit=0
```

# deinterleaved=4 contiguous 组物化拆分（2026-09-06）

本轮将 `materializeDeinterleaved4ToContiguous` 的单组 source 类型/结果类型合同检查和
四路 `vintlv` 物化抽取为 `materializeDeinterleaved4Group`。主函数继续负责 footprint
arity、组分片和结果累积；source fallback、结果截断顺序、诊断文本及 `vintlv` 组合语义
保持不变。

本轮验证：

```text
git diff --check                                      # passed
check_changed_code.py --base origin/master            # checked_files=1 errors=0 warnings=0
vmi_to_vpto_vintlv_d2_to_d4.pto                        # blocked by pre-existing VMI-PASS-INVARIANT
```

# masked-store contiguous chunk 发射拆分（2026-09-06）

本轮将 `OneToNVMIMaskedStoreOpPattern::lowerContiguous` 中逐 physical chunk 的 value/mask
类型检查、active lane 计算、predicate 物化、地址对齐校验和 `vsts` 发射抽取为
`emitContiguousMaskedStorePart`。主函数继续负责 value/mask layout conversion、arity
检查和 chunk 遍历，零 active lane、诊断文本及 store 顺序保持不变。

本轮验证：

```text
git diff --check                                      # passed
check_changed_code.py --base origin/master            # checked_files=1 errors=0 warnings=0
vmi_to_vpto_masked_store.pto                           # exit=0
```

# group-broadcast E2B chunk 合同拆分（2026-09-06）

本轮将 `validateDirectE2BShape` 中各 physical part 的 chunk 数量一致性检查抽取为
`validateDirectE2BChunks`。E2B 主合同仍负责 layout、element width、stride、pointer、
group 数量和结果 arity 检查；E2B dist 选择、单 packet 限制与失败诊断保持不变。

本轮验证：

```text
git diff --check                                      # passed
check_changed_code.py --base origin/master            # checked_files=1 errors=0 warnings=0
vmi_to_vpto_group_broadcast_load_e2b_b16.pto           # blocked by pre-existing VMI-PASS-INVARIANT
```

# slots=8 lane-stride group-store 地址规划拆分（2026-09-06）

本轮将 `lowerSlots8LaneStride` 中每个 slot block 的 vreg 合同、group offset 构造和
direct-memory 合法性汇总抽取为 `buildLaneStrideGroupOffsets`。主函数继续负责 dist/mask
选择、unaligned compact stream fallback 和 aligned 发射；地址顺序、stream advances、
失败诊断及 direct/stream 路由保持不变。

本轮验证：

```text
git diff --check                                      # passed
check_changed_code.py --base origin/master            # checked_files=1 errors=0 warnings=0
vmi_layout_assignment_group_store_slots1_unit_stride.pto # exit=0
```

# group-slot-load slots=1 chunk 发射拆分（2026-09-06）

本轮将 `lowerGroupSlotLoadSlots1` 循环中的结果类型/mask 合同、group offset 计算、pointer
构造和 `vsldb` 发射抽取为 `emitGroupSlotLoadSlots1Chunk`。主函数继续负责 element width
和 source stride 对齐合同及逐 group 遍历；`PAT_VL1` mask、地址步进、结果顺序和失败诊断
保持不变。

本轮验证：

```text
git diff --check                                      # passed
check_changed_code.py --base origin/master            # checked_files=1 errors=0 warnings=0
vmi_layout_assignment_group_slot_load_slots1_unaligned_stride_invalid.pto # expected invalid-case exit=1
```

# load physical plan 构造职责拆分（2026-09-06）

本轮将 `OneToNVMILoadOpPattern::buildPhysicalPlan` 中 converted result type 获取与
contiguous footprint type 获取分别抽取为 `getLoadResultTypes` 和
`getContiguousLoadTypes`。plan builder 继续负责 source/offset 归一化、read-safety
验证和 footprint 比较；地址语义、layout conversion 顺序及失败诊断保持不变。

本轮验证：

```text
git diff --check                                      # passed
check_changed_code.py --base origin/master            # checked_files=1 errors=0 warnings=0
vmi_to_vpto_load_store_contiguous.pto                  # exit=0
```

# contiguous load 访存物化职责拆分（2026-09-06）

本轮将 `OneToNVMILoadOpPattern::lowerContiguous` 中 aligned `vlds` 与 unaligned stateful
`vldus` 的选择抽取为 `materializeContiguousLoadParts`。helper 只负责根据 direct-memory
合法性选择物化路径，`lowerContiguous` 继续负责 layout conversion 和结果替换；对齐证明、
align/base 状态推进、part 顺序及 fallback 语义保持不变。

本轮验证：

```text
git diff --check                                      # passed
check_changed_code.py --base origin/master            # checked_files=1 errors=0 warnings=0
vmi_to_vpto_load_store_contiguous.pto                  # exit=0
```

# contiguous store unaligned base 职责拆分（2026-09-06）

本轮将 `OneToNVMIStoreOpPattern::lowerContiguousStoreParts` 中 unaligned destination 的
buffer pointer 物化和 offset 合成抽取为 `materializeUnalignedStoreBase`。store 主路径仍
负责 aligned `vsts` 与 unaligned stateful stream 的选择、active-lane advances 收集和
stream 发射；base 生命周期、offset 语义及失败诊断保持不变。

本轮验证：

```text
git diff --check                                      # passed
check_changed_code.py --base origin/master            # checked_files=1 errors=0 warnings=0
vmi_to_vpto_stride_store.pto                           # exit=0
```

# unaligned load state 推进职责拆分（2026-09-06）

本轮将 `OneToNVMILoadOpPattern::materializeUnalignedContiguousParts` 中单个 `vldus` 发射
及 updated base/align 状态封装为 `emitUnalignedLoadPart` 和 `UnalignedLoadPart`。外层
循环只负责按 contiguous result types 累积结果并推进状态；`vldus` 的 increment、operand
类型、align/base 更新顺序和 part 顺序保持不变。

本轮验证：

```text
git diff --check                                      # passed
check_changed_code.py --base origin/master            # checked_files=1 errors=0 warnings=0
vmi_to_vpto_load_store_contiguous.pto                  # exit=0
```

# interleave-store address plan 职责拆分（2026-09-06）

本轮将 `OneToNVMIInterleaveStoreOpPattern::matchAndRewrite` 中 direct `vstsx2` 合法性判定
与 unaligned stream base 构造抽取为 `InterleaveStoreAddressPlan`/`buildAddressPlan`。入口
继续负责 operands、low/high arity、chunk 发射和 stream 提交；direct/stream 路由优先级、
offset 语义、状态对象和失败传播保持不变。

本轮验证：

```text
git diff --check                                      # passed
check_changed_code.py --base origin/master            # checked_files=1 errors=0 warnings=0
vmi_to_vpto_load_store_contiguous.pto                  # exit=0
```
