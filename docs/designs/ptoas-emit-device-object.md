# `ptoas emit device object`（VPTO 可重定位 AI Core device object 输出）

对应 issue #1726。为 VPTO 后端新增一种显式输出模式：跳过 host stub 编译与
fatobj 打包，把合并后的**可重定位** AI Core device ELF 直接写入 `-o`。默认的
fatobj 输出不变。设计上与 [`ptoas-emit-fatobj.md`](ptoas-emit-fatobj.md) 平行；
该文档描述 fatobj 发射链路，本文只描述其中被"截短"的分支。

## 背景与动机

PyPTO persistent runtime 自带 kernel entry 包装和最终 device 链接。当前它要消费
PTOAS 编译产物，只能先得到 fatobj，再从 host 容器中抽取 `__aicore_rel_binary`，
补 logical-name 别名后交给 runtime 链接（hw-native-sys/pypto#3239 验证过的变通）。
这条链路存在两个问题：

- 多一次打包/抽取往返，且抽取逻辑耦合 host 容器布局与符号约定；
- PyPTO 需要的其实是发射管线中段就存在的中间产物——合并后的可重定位
  device object——只是 fatobj 路径随后把它嵌进容器并丢弃原文件。

因此本设计的核心判断是：**不新增任何编译逻辑，把既有中间产物升格为一等输出**。

## 目标 / 非目标

目标：

- 输出可重定位 AI Core ELF（`ld.lld -r` 合并），符号与重定位原样保留；
- 复用 fatobj 路径的全部前置阶段（bridge wrapper 链接、按 target 的 Bisheng
  `--cce-aicore-only` 编译、VFSIMT size 校验修复、调度策略），行为与 fatobj
  路径完全一致；
- 显式文档化导出符号命名契约（logical 名与 native 名的映射规则），客户端
  无需推断 CANN 后缀规则；
- 对不支持的输入给出明确诊断，绝不静默只输出第一个对象。

非目标：

- 不改变默认 fatobj 输出（一个字节都不动）；
- 不支持 mixed `pto.backend` 装配（issue 明确 full mixed-backend support
  follows separately，本设计给出显式拒绝诊断）；
- 不做 final-linked（无 `-r`）直出模式——那是面向"直接加载 kernel 镜像"
  的另一形态，与本设计正交，另行工作提供；
- 不在本特性内暴露 EmitC 侧的同类输出（`emitCppVectorDeviceObject` /
  `emitCppCubeDeviceObject` 原语已存在，留作后续接口决策）。

## CLI 接口

```bash
ptoas kernel.pto --pto-arch=a5 --pto-backend=vpto \
  --emit-vpto-device-object -o kernel.device.o
```

`--emit-vpto-device-object` 命名与既有 `--emit-vpto` / `--emit-vpto-llvm-ir`
输出模式家族一致。校验规则（`buildBackendInfo` 中早期触发，全部在工具链
初始化之前，因此诊断不依赖 CANN 环境）：

| 约束 | 诊断 |
|---|---|
| 需要 VPTO 后端（`--pto-backend=vpto` 或模块 backend 属性解析为 vpto） | `Error: --emit-vpto-device-object requires the VPTO backend; pass --pto-backend=vpto (or mark the module with the vpto backend attribute).` |
| 不与 IR 输出 flag 组合（`--emit-pto-ir` / `--emit-vpto` / `--emit-vpto-llvm-ir`） | `Error: --emit-vpto-device-object cannot be combined with the IR output flags ...` |
| 不支持 mixed `pto.backend` 装配 | `Error: --emit-vpto-device-object does not support mixed pto.backend assemblies; split the input or drop the flag to get the default fatobj output.` |
| 显式 `-o` 文件路径（`-o -` 不支持）。本模式在 `buildBackendInfo` 中早期校验，早于工具链初始化与 lowering；作业侧既有检查保留为兜底 | `Error: object output requires an explicit file path passed with -o.` |

LLVM-inspection 工作流（`--emit-vpto-llvm-ir` 等，PyPTO 用于同步准入检查）不受
影响：这些模式与 device-object 模式互斥，各自独立可用。

## 发射管线

fatobj 路径与 device-object 路径共享同一套 `VPTOFatobjArtifacts` 步骤，区别只在
终点：

```text
                       ┌──────────────── fatobj 路径（默认）────────────────┐
compilePTOASModule ───►│ initCommandLogs                                   │
(emitHostStub=false)   │ compileBridgeWrappers（bitcode 链入 device IR）    │
                       │ emitCubeObject   （ABI 重命名 + --cce-aicore-only）│
                       │ emitVectorObject （同上 + VFSIMT 校验/修复）        │
                       │ mergeDeviceObjects（ld.lld -r，cube+vector 合并） │
                       └──────────────┬──────────────────────┬──────────────┘
                                      │                      │
                              copy 到 -o（本特性）     compileHostStubToFatobj
                              （emitVPTODeviceObjectsLLVM）  （host stub + 容器打包）
```

关键点：

- **stub 源不再收集**：device-object 模式下 `VPTOBackendJob::run` 以
  `emitHostStub = false` 调用 `compilePTOASModule`。fatobj 路径中 stub 源只被
  host 打包步骤消费，本模式没有该步骤，收集是纯浪费。
- **bridge wrapper 属于 device 侧内容**：wrapper TU 编译为 bitcode 后经
  `llvm-link` 链入 device IR，不是 host 步骤，本模式保留。
- **多 device 模块**：cube + vector 都存在时输出单个合并 `.o`（`-r`），与
  fatobj 路径喂给 host 打包的产物相同；仅一个模块时同样走 `ld.lld -r`，
  产物形态统一。不存在"静默只输出第一个对象"的分支。
- **与 final-linked 直出模式在同一分叉点分岔**：该模式（另行工作）消费
  同一套 device object，但走不带 `-r` 的最终链接。两者共享管线复用结构与
  校验框架；其合入时应一并落地两个 flag 的互斥诊断，避免歧义产物。
- **改动范围与挂点**：flag 定义于 `tools/ptoas/ptoas.cpp`（声明在
  `ptoas.h`）；发射入口位于 `tools/ptoas/ObjectEmission.cpp`，复用
  `VPTOFatobjArtifacts` 的步骤到 `mergeDeviceObjects` 为止后直接写出；
  后端选择与校验挂在 `driver.cpp` 的 `buildBackendInfo` 与 VPTO 作业的
  发射分支；使用说明与符号契约写入 `README.md`；诊断回归用例放在
  `test/lit/pto/`。核心约束：`lib/PTO`（pass/lowering）零改动——本特性
  只触碰 `tools/ptoas` 与文档。

## 产物契约

- ELF64、`ET_REL`（可重定位）、AI Core 机器类型（`llvm-readelf -h` 显示
  `Machine: 1029`），符号与重定位保留；
- 无 host 段、无 `__aicore_rel_binary` / `__aicore_rel_rec` 容器段——裸
  device ELF；
- 面向"自带 kernel entry 包装 + 最终 device 链接"的 runtime（PyPTO
  persistent）。
- 输出写入采用**同目录临时文件 + `rename` 原子替换**：写入失败（如磁盘
  满）不会截断或破坏 `-o` 路径上已有的有效对象，命令以失败退出且原文件
  保持不变。

## 导出符号命名契约

这是 issue #1726 中 "clients need not infer CANN suffix rules" 的落地（方案：
文档化后缀规则；manifest 可作后续增强）。规则与 fatobj 内嵌产物完全一致，
因为 ABI 重命名发生在共享的前置阶段：

| 类别 | 命名 | 来源 |
|---|---|---|
| kernel 入口 | `<logical>_mix_aiv`（vector）/ `<logical>_mix_aic`（cube） | VPTO LLVM emitter（`kVectorSuffix` / `kCubeSuffix`），ABI 重命名 pass 跳过已带该后缀的符号 |
| 其余外部可见函数 | 逻辑名 + target ABI 后缀：CANN ≥ 9.0.0-beta.2 用 `.vector` / `.cube`，更早版本用 `_mix_aiv` / `_mix_aic` | `applyVPTOLLVMABINames` + `CANNToolchain::vptoPublicABISuffix` |
| logical 名还原 | 去掉 target 后缀（host stub 侧 `getLogicalKernelName` 已是该规则的实现） | |

符号形态示例（新 ABI，Abs 内核为例）：

```text
$ llvm-readelf --symbols kernel.device.o
  18: ... FUNC GLOBAL DEFAULT 1 abs_kernel_2d_mix_aiv          ← 入口
  3:  ... FUNC LOCAL  DEFAULT 1 abs_kernel_2d_mix_aiv.vector.thread
  ...OBJECT  LOCAL  .AIV_Kernel_Type_abs_kernel_2d_mix_aiv ...
```

客户端从产物符号去后缀即可得到 logical 名并注册别名。后缀随 CANN 版本
切换的两条规则都已文档化（README）；后缀版本由工具链探测
（`discoverCANNVersion`，读 `ASCEND_HOME_PATH` 下的 version.info）决定，
`--cann-output-version` 仅作用于 VPTO lowering，**不影响**符号后缀。

## 验证设计

三层：

1. **诊断回归（无工具链，任意 CI 机可跑）**：
   `test/lit/pto/` 下的 `not ptoas` 用例——backend 约束、IR flag 互斥、
   显式 `-o` 路径（缺失与 `-o -`）、mixed 拒绝，全部在工具链初始化与
   lowering 之前触发。
2. **真实发射验证（需 Bisheng/CANN，`ASCEND_HOME_PATH`）**：同一输入分别走
   默认 fatobj 与 `--emit-vpto-device-object`，`llvm-readelf` 断言：
   默认产物为 x86-64 host 容器且含 `__aicore_rel_binary` 段（回归不变）；
   新产物为 `ET_REL` + Machine 1029、无容器段、入口与后缀符合上表。
3. **全量回归**：`test/lit/pto` + `test/lit/vpto` 全量 lit。

验收的后续部分（与现有 EmitC caller + runtime wrapper 链接、focused A5
Worker 执行）需要硬件与 PyPTO 侧配合，在 PyPTO 集成时进行，不在本特性
的本地验收范围内。

## 边界与已知约束

- `pto.kernel_kind` 必须是**模块级**属性且模块只含函数体——这是 VPTO 对象
  发射的既有前置（`VPTONormalizeContainer`），IR 输出模式无此要求。PyPTO /
  PTODSL 前端生成供本模式消费的输入时需携带该属性。
- 后缀规则随 CANN 版本切换是既定行为，本设计只做契约文档化，不冻结版本
  行为；客户端应按上表规则而非硬编码某一版本的后缀。
- `-r` 合并对 cube/vector 符号无碰撞风险（target 后缀天然隔离）；
  `--allow-multiple-definition` 语义与 fatobj 内部 merge 一致，最终链接
  由调用方完成。

## 相关工作

- [`ptoas-emit-fatobj.md`](ptoas-emit-fatobj.md)：fatobj 发射链路设计（本
  设计的共享前置）；
- [`vpto-vfsimt-size-object-patch-plan.md`](vpto-vfsimt-size-object-patch-plan.md)：
  VFSIMT size 校验/修复，device-object 路径原样复用；
- final-linked（无 `-r`）直出模式为正交扩展，另行设计提供；本设计与其
  共享管线复用结构与校验框架，两者将互斥（互斥诊断随其合入落地），先后
  合入时后者基于同一结构增量实现，避免重复。
