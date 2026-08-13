# VPTO TPush/TPop 桥接方案汇报

## 一、概述

**目标**：让 VPTO 后端（LLVM IR 后端）支持 PTO-ISA 的 FIFO `TPush` / `TPop` / `TFree` 协议，打通 Cube 到 Vector 的 tile 数据通路。

**核心思路**：VPTO **不重新实现 FIFO 协议**，而是通过固定 C ABI 的 wrapper 调用 **Bisheng 已实例化好的 PTO-ISA C++ 模板**——wrapper 编译为 device bitcode，用 CANN `llvm-link` 与 VPTO 生成的 LLVM module 合并，再统一编译为 device object。

**当前状态**：固定 specialization（A5 本地 C2V、f32、16x16→8x16、split=1）已跑通完整链路并在 CA 模拟器上端到端验证通过。

---

## 二、背景与问题

### 2.1 `TPush` / `TPop` 不是普通函数

PTO-ISA 的 FIFO 操作由 C++ 模板实现。一次 `TPUSH` 的真实行为包括：producer slot 等待、FIFO slot 地址计算、Acc 到 Vec 的 fixpipe 分片、producer index 更新、ready/free flag 同步，以及 split 轴和 Tile 类型的编译期分派；`TPOP` 还要把 consumer Tile 绑定到运行时取得的 FIFO slot 地址。它们依赖完整的 `TPipe<...>` / `Tile<...>` 类型——这不是可以替换成单条 LLVM intrinsic 的函数：

```cpp
TPUSH<Pipe, AccTile, TileSplitAxis::TILE_UP_DOWN>(pipe, tile);
```

### 2.2 VPTO 与 EmitC 的根本差异

| 后端 | 产物 | 能否复用模板 |
|---|---|---|
| EmitC | 生成 C++ 源码 | ✅ 模板参数可直接出现在输出文本中，交给 Bisheng 实例化 |
| VPTO | 生成 LLVM IR | ❌ LLVM IR 里只剩整数地址、指针和函数调用，模板类型信息已不存在 |

**EmitC 怎么做（参照）**：EmitC 从 internal op 读出 pipe 配置与 Tile 类型，直接生成 `TPipe<...>` / `Tile<...>` token 和 `TPUSH<...>(pipe, tile)` 调用文本，Bisheng 在编译时完成模板实例化、对象布局和指令选择。EmitC 不需要手工计算 `TPipe` 大小、不需要 placement new、不需要外部 wrapper 或 bitcode 链接——`TPOP` 直接修改 C++ Tile 对象，C++ 作用域负责 pipe 的构造与析构。这正是 VPTO 要借鉴的语义，但 VPTO 没有"输出文本"这个自由度，只能走函数调用。

**VPTO 侧的两条出路**：

1. **自己实现 FIFO 协议语义**（flag 同步、分片、轮转）——需要重复实现一套复杂协议，且与 PTO-ISA 实现存在语义漂移风险，维护成本高；
2. **调用 Bisheng 实例化好的模板**（本方案）——协议语义由 PTO-ISA 单一来源保证，VPTO 只负责"桥接"。

方案 2 显然更优，需要解决的核心问题是：**VPTO 生成的 LLVM IR 如何调用到 C++ 模板的实例化代码**。

---

## 三、方案设计

### 3.1 总体架构

```text
kernel.pto（公开 PTO IR，测试用例书写）
  ↓ ① 输入即 internal pipe IR（main 已有能力，本分支复用）
initialize_l2l_pipe / tpush / tpop / tfree
  ↓ ② VPTOSplitCVModule【本分支改动】
按 pto.kernel_kind 拆成 Cube 子模块 + Vector 子模块
  ↓ ③ FoldTileBufIntrinsics【本分支改动】
跳过被 TPOP 重绑定的 declare_tile，不把动态地址折叠成占位值
  ↓ ④ CANN900 VPTO LLVM emitter【本分支改动，核心】
internal ops → 固定 wrapper ABI 调用（pto_vpto_pipe_*）
  ↓ ⑤ 测试脚本【本分支改动】
用 Bisheng 把 vpto_bridge.cpp 按 cube/vec 两个 target 编成两份 bitcode
  ↓ ⑥ ObjectEmission【本分支改动】
读环境变量，用 CANN llvm-link 把 bridge bitcode 合入 VPTO device LLVM module
  ↓ ⑦ Bisheng 编译 device object → fatobj【已有流程】
  ↓ ⑧ CA 模拟器运行 + host 侧 golden 比较【测试用例】
```

### 3.2 关键设计决策

| # | 决策 | 理由 |
|---|---|---|
| 1 | **bitcode 层合并**（`llvm-link`），而非分别编译 device object 再做 relocatable 链接 | 避免重定位、符号可见性、双 device object 的 target 冲突；合并后走原有 Bisheng 编译流程，改动最小 |
| 2 | **wrapper ABI 固定为 C 函数**（`pto_vpto_pipe_*`），tile 地址用 **i64 承载** | 模板类型信息不出 wrapper，跨 ABI 干净；i64 不携带 CCE 地址空间信息，回避地址空间指针跨 ABI 的问题 |
| 3 | **TPipe 生命周期由 emitter 管理**：storage 分配在 kernel 栈帧上（大小运行时取自 wrapper 导出的 `pto_vpto_pipe_size()`），init 在函数入口调用，**不插入 finish** | TPipe 是有状态对象，storage 必须从初始化存活到函数退出（kernel 栈帧满足该约束）；析构握手（finish）经模拟器实验证明在固定配置下非必需——consumer 侧 TPipe 构造时 freed flags 已预置，握手不会阻塞，单次 launch 的 FIFO 数据正确性不依赖它，故 bridge 不导出 finish；多轮 push / 跨 launch 复用不在本阶段范围。storage 大小不硬编码，避免 TPipe 布局漂移 |
| 4 | **TPop 的"运行时重绑定"用 SSA 映射实现**：emitter 内 `popTileAddresses`（tile 值 → pop 返回的 slot 地址）+ `replaceAllUsesWith` 注入 | LLVM IR 是 SSA，无法"改地址"；把"重绑定"变成"替换后续使用" |
| 5 | **按函数级 `pto.kernel_kind` 拆分子模块** | Cube/Vector 是两种 device object，对应 `dav-c310-cube` / `dav-c310-vec` target，必须分开编译 |
| 6 | **wrapper 编译属于测试/工具链资产**（脚本 + 环境变量注入），编译器只提供合并通道 | bridge 编译依赖具体用例的 wrapper 源码，不属于编译器内置行为；解耦后该通道可复用 |
| 7 | **固定配置硬校验**：不满足支持范围（dir_mask/slot_size/slot_num/flag_base/nosplit/acc_push_epilogue）报明确诊断 | 第一阶段只做固定 specialization，其他配置不能静默生成近似实现 |

### 3.3 为什么第一阶段限定固定 specialization

TPush/TPop 模板的实例化参数空间很大（方向、元素类型、形状、split 方式、TPipe 配置组合）。固定配置（`TPipe<0, DIR_C2V, 1024, 8, 2, false>`）先把**整条链路跑通并验证**——包括最不确定的部分（TPop 返回地址参与后续 tile 运算、fixpipe 分片语义）——再谈通用化。研究文档已给出通用化的工程化路径（见第六节）。

---

## 四、实现内容


### 4.1 核心：emitter 把 pipe op 降级为 wrapper 调用（`VPTOCANN900LLVMEmitter.cpp`）

VPTO 不实现 FIFO 协议，emitter 只做三件事：

1. **类型降级**：`TileBufType` → i64（tile 句柄降为整数地址），`PipeType` → 不透明指针；
2. **桥接调用**：四个 internal pipe op 降级为固定 C ABI 调用，LLVM module 里只含 `pto_vpto_pipe_*` 的 declare，实现由 bridge bitcode 提供；
3. **TPop 地址重绑定**：pop 返回的 FIFO slot 地址经 `popTileAddresses` 映射替换后续 tile 使用。

各 op 的降级规则（不满足支持范围一律报明确诊断，不静默近似）：

| internal op | 降级为 |
|---|---|
| `initialize_l2l_pipe` | 校验固定配置 → kernel 栈帧上分配 storage（大小运行时取自 wrapper 导出的 `pto_vpto_pipe_size()`，不硬编码）→ `pipe_init(storage, consumerBuf)` |
| `tpush` | 要求 split=1、tile 来自有规划地址的 `alloc_tile` → `pipe_push(storage, tileAddr)` |
| `tpop` | 要求 split=1 → `pipe_pop(storage)` 返回 i64 slot 地址 → 重绑定到后续 tile 使用 |
| `tfree` | 要求 tile-entry、split=1 → `pipe_free(storage)` |

两个值得说明的设计点：

- **配置来源只有 internal op**：方向、slot 尺寸、flag base、peer 配对（Cube/Vector 两侧拿到同一个 i32 local buffer 地址）都由前端 lowering 补齐并校验过，emitter 不做任何默认值推断；
- **SSA 值 = storage 指针**：`!pto.pipe` 是 SSA 值，TPipe 是有状态对象——以"SSA 值 = storage 指针"化解：数据流在 SSA 层显式连通（init 结果流入 push/pop/free），可变状态在载体内核，转换后没有隐式全局 pipe 状态。storage 生命周期与 kernel 一致；不插入 finish（析构握手非必需，见决策 #3）。

### 4.2 wrapper ABI：五个 `pto_vpto_pipe_*` 函数

emitter 与 wrapper 之间是固定 C ABI（全部 `extern "C" [aicore]`）。参数只有"哪个 pipe（storage）+ 哪块数据（地址）"，没有 pipe 配置参数——wrapper 是固定 specialization，模板参数已在 bridge 编译期固化：

| 函数 | 签名 | 语义 |
|---|---|---|
| `pto_vpto_pipe_size` | `size_t ()` | 返回 `sizeof(TPipe)`，emitter 据此分配 storage |
| `pto_vpto_pipe_init` | `void (void *storage, uint32_t consumerBuf)` | 在 storage 上 placement new 构造 TPipe |
| `pto_vpto_pipe_push` | `void (void *storage, uint64_t tileAddr)` | Cube 侧：把 Acc tile 地址绑成 `Tile<>` 后 `TPUSH` |
| `pto_vpto_pipe_pop` | `uint64_t (void *storage)` | Vector 侧：`TPOP` 后把 FIFO slot 地址作为 i64 返回 |
| `pto_vpto_pipe_free` | `void (void *storage)` | `TFREE`：推进 consumer 生命周期 |


隐含契约：

- **storage 所有权归 emitter**：分配、生命周期、回收都在 emitter 侧，bridge 只借用——这也是"不插入 finish"的基础：storage 是随 kernel 生死的内存，没有析构依赖；
- **固定函数名、无 specialization 后缀**：bridge 按 `__DAV_CUBE__` / `__DAV_VEC__` 裁剪，Cube bitcode 只实现 push、Vector bitcode 只实现 pop/free（init/size 两边都有），各合各的模块。

### 4.3 按 `kernel_kind` 拆分 Cube/Vector（`VPTOSplitCVModule.cpp`）

函数带 `pto.kernel_kind` 时，把 module 拆成 Cube / Vector 两个子模块，各自按对应 target（`dav-c310-cube` / `dav-c310-vec`）发射、编译——两者是独立的 device object，必须分开编译。

### 4.4 保护 TPOP 动态地址（`FoldTileBufIntrinsics.cpp`）

该 pass 会把 tile 句柄折叠成静态地址；而 declare_tile 本身没有地址（emitter 里是占位 0），真实地址由 TPOP 运行时返回。跳过"被 TPOP 重绑定"的折叠，否则动态地址会在进入 emitter 前被错误折叠成占位值，FIFO 数据流就断了。

### 4.5 bitcode 合并通道（`ObjectEmission.cpp`）

bridge bitcode 由测试脚本按 cube/vec 两个 target 分别编译，经环境变量（`PTOAS_VPTO_CUBE_BRIDGE_BITCODE` / `PTOAS_VPTO_VECTOR_BRIDGE_BITCODE`）注入；`linkDeviceLLVMBitcode` 用 CANN 自带 `llvm-link` 把 bridge bitcode 与 VPTO device LLVM module 合并，再走原有编译流程。这是"把外部 PTO-ISA C++ 模板实例化代码塞进 VPTO 产物"的**通用通道**——不限于 TPush/TPop，以后任何 PTO-ISA C++ 模板都可复用同一机制。



## 五、验证情况

| 验证项 | 结果 | 说明 |
|---|---|---|
| CA 模拟器端到端 | ✅ 通过 | AIC/AIV 双核正常起停，`compare passed`（2026-08-13） |
| 数值正确性 | ✅ | 128 个 f32 全量比较通过；A=I、B=arange 使 golden 精确可判 |
| FIFO 多轮轮转 | ✅ | 模拟器 12 轮，超过 `slot_num=8`（research 阶段验证） |
| 数据真实性 | ✅ | 无 Cube-to-GM 旁路，输出只可能来自 FIFO tile |
| 固定配置诊断 | ✅ | 不满足支持范围时报明确错误（编译期代码审查确认） |

---

## 六、当前边界与后续计划

### 6.1 当前边界（第一阶段）

- 固定 specialization：A5 本地 C2V、f32、16x16→8x16、`split=1`、`TPipe<0, C2V, 1024, 8, 2, false>`；
- 不包含：global-entry、A2/A3、V2C、双向 pipe、量化、odd split；
- TPOP 地址传播验证范围为直线控制流（loop/branch/block argument/alias 链未覆盖）；
- bridge bitcode 由测试脚本编译、环境变量传入，wrapper 生成尚未自动化。

### 6.2 后续工程化路径（研究文档已规划）

```text
PTO frontend pipe ops
  -> frontend-to-internal pipe lowering
  -> 收集实际使用的 pipe 与 Tile specialization
  -> 生成 target-specific C wrapper
  -> Bisheng 编译 wrapper device bitcode
  -> ObjectEmission 自动合入对应 VPTO LLVM module
  -> Bisheng 编译 device object
```

即：**ptoas 自动收集 specialization 并生成 wrapper**（含 PTO-ISA 头文件路径的自动发现），替代当前"测试脚本编译 + 环境变量注入"的手工流程；同时把环境变量机制演进为 pass option / 配置驱动的正式通道。

### 6.3 方法论沉淀：如何把 PTO-ISA 模板接入 VPTO（不只适用于 FIFO）

1. **先找出模板的 compile-time contract**：明确模板真正需要的参数（pipe 配置、Tile 类型、shape/layout、split/mode、target arch），不要先设计 wrapper ABI；
2. **以 internal op 为唯一配置来源**：不从 `.pto` 文本、脚本正则或 host 参数恢复模板参数——前端 lowering 后默认值、peer 关系和 verifier 结果都已确定；
3. **把 semantic handle 与 ABI carrier 分开**：`PipeType`→指针、`TileBufType`→i64 只是载体，必须另保留一份 specialization 描述，不能让 i64 抹掉 Tile 的真实类型信息；
4. **设计持久化 context ABI**：有状态模板对象按 `init → storage`、操作复用同一 storage、退出时析构组织；size/alignment/lifetime 必须成为后端 contract（本方案大小已由 `pto_vpto_pipe_size()` 提供，析构握手在固定配置下经实验证明可省略）；
5. **wrapper 只做 ABI 适配**：不重新实现模板协议——只负责 placement new、把 i64 地址绑成 Tile、调用模板、把返回值转回 LLVM ABI carrier；协议、同步、layout 转换仍由 PTO-ISA 模板负责；
6. **按 target 分编译，再 bitcode 合并**：Cube-only / Vector-only 模板代码不在同一个 wrapper TU 中编译，按 `dav-c310-cube` / `dav-c310-vec` 分别编译后合并；
7. **用稳定描述传递 specialization**：短期可用可读 ABI 后缀（`__f0_s1024_n8_...`），长期建议稳定 hash + debug side table，ObjectEmission 依据同一份描述生成 wrapper，避免依赖函数名字符串的脆弱解析。

---

## 七、请评审关注的问题

以下问题希望评审重点把关（按不确定程度排序）：

1. **wrapper ABI 设计**：tile 地址用 i64 承载、storage 由 emitter 分配（大小由 wrapper 的 `pto_vpto_pipe_size()` 提供）、init 在函数入口调用且不插入 finish——这套 ABI 是否合理？是否有更优的形态（例如直接暴露地址空间指针）？
2. **TPop 重绑定的 SSA 注入方式**：`popTileAddresses` 映射 + `replaceAllUsesWith` 只验证了直线控制流；扩展到 loop/branch/多 subblock 场景应该采用什么机制（例如改由 IR 层携带 pop 结果、或引入显式重绑定 op）？
3. **固定配置的校验策略**：当前对不支持配置一律报错。进入第二阶段后，是继续"白名单 + 报错"，还是对部分参数（如 slot_size/slot_num）做自动适配？
4. **bridge 注入的工程化通道**：环境变量注入作为 PoC 是否可接受？正式通道是让 ptoas 自动编译 wrapper，还是由构建系统负责、ptoas 只接受"已编译 bitcode 路径"？
5. **TPipe 生命周期约束**：storage 分配在 kernel 栈帧上、init 在函数入口、无 finish（析构握手经实验证明非必需，依赖 consumer 侧 freed flags 预置）——在多 subkernel、多次 launch、异步并发场景下，无 finish 的假设是否继续成立？跨 launch 复用是否需要重新引入 finish？
6. **验证充分性**：当前单用例单配置。在投入第二阶段前，还需要补哪些测试（负面用例、多轮、多配置矩阵）才能证明桥接机制本身的可靠性？

---

## 附：改动清单（相对 main）

```
lib/PTO/Transforms/VPTOCANN900LLVMEmitter.cpp      ~180 行（核心：pipe op 降级）
lib/PTO/Transforms/VPTOSplitCVModule.cpp            ~70 行（函数级 kind 拆分）
lib/PTO/Transforms/FoldTileBufIntrinsics.cpp        ~28 行（declare_tile 地址保护）
tools/ptoas/ObjectEmission.cpp                      ~50 行（llvm-link 合并）
lib/PTO/IR/PTO.cpp                                  ~7 行（module 级 kind 判定）
include/PTO/Transforms/TileOpExpansionUtils.h       ~3 行（internal pipe op 排除 TileLib）
test/vpto/cases/kernels/fifo-tile-data-consume/     新用例（kernel.pto + vpto_bridge.cpp + host/golden/compare）
test/vpto/scripts/run_host_vpto_validation.sh       ~35 行（bridge bitcode 编译与注入）
docs/designs/                                       方案研究 + 实现说明两篇
```
