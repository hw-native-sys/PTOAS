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
