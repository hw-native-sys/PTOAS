# VPTO C++ 接口桥接泛化方案（初稿）

## 1. 背景与目标

`feature/vpto-tpush-tpop-bridge` 上的 PoC 已证明桥接机制的可行性：把 Bisheng 实例化的
PTO-ISA C++ 模板 wrapper 编译为 device bitcode，经 `llvm-link` 与 VPTO 生成的 LLVM IR
在 bitcode 层合并，端到端跑通了 `TPush → TPOP` FIFO 通路（`fifo-tile-data-consume`
用例，CA 模拟器全量 compare 通过）。

PoC 的局限（详见 `vpto-tpush-tpop-bridge-report.md`）：

- **固定 specialization 硬编码**：emitter 里写死 `dir_mask=1, slot_size=1024,
  slot_num=8, flag_base=0, nosplit=false`（`VPTOCANN900LLVMEmitter.cpp:428`），
  wrapper 写死 `TPipe<0, C2V, 1024, 8, 2, false>` 与 16x16/8x16 tile；
- **桥接逻辑与 TPipe 语义耦合**：`LowerPipeBridgeOpPattern` 一个 pattern 里既有
  "TPipe 参数如何转换"（config 校验、storage、重绑定），又有"如何发 wrapper 调用"
  （decl、call、i64/指针 ABI）——两者混在一起，新接口族（如 MATMUL）无法复用。

本方案目标：

1. **参数泛化**：TPipe 配置、tile 形状/dtype 等参数不再写死，由 op 属性 + 白名单驱动；
2. **接口泛化**：桥接机制与 TPipe 解耦，成为所有 PTO-ISA C++ 接口可复用的通用通道，
   第一个验证对象是 **CUBE 侧 MATMUL**（接口面与 pipe 完全不同的第二家族）；
3. **工程化**：wrapper 自动生成、bitcode 自动合入，替代脚本 + 环境变量的手工流程。

**实现原则**：PoC 的验证结论（bitcode 层桥接可行、ABI 形态可用）是本方案的前提，
但 PoC 代码**仅作功能参照，不套用**——通用化实现从头新写（见 §3.4）。

## 2. 核心决策（四条）

1. **IR 白名单文本**：新增一个文本文件描述"哪些 IR op 走 C++ 接口桥接"，pass 运行时读取；
2. **白名单解决两个问题**：
   - **路由**：pass 不再需要静态分析"哪些接口生成 C++ 调用、哪些直接 emit LLVM IR"——
     白名单即路由表；
   - **类型映射**：IR 参数到目标 C++ 模板类型的映射关系在白名单中声明；
3. **类型映射参照 EmitC 逻辑（不改动 EmitC 实现）**：以 `PTOToEmitC.cpp` 里现有的
   "IR op 属性 → C++ 模板 token" 构建逻辑为参照，在**桥接侧自行适配实现**对应逻辑，
   EmitC 侧代码保持不动（见 §3.3）；
4. **分层（从头实现）**：把 TPush/TPop 的参数 API 转换做成 **TPipe 相关 pass**，
   把 "C++ 混编" 做成独立的**通用 pass**——通用 pass 不感知 TPipe 的参数接口变化，
   桥接能力对所有操作公用。实现上**不套用 PoC 代码**：PoC 仅作功能参照（§3.4），
   新实现全部新写。

## 3. 架构设计

### 3.1 分层

```text
[家族专用 pass 层]（每个 PTO-ISA 接口家族一个）
  示例：TPipe 专用 pass（本阶段）、MATMUL 专用 pass（下一阶段）
  职责：理解本家族 op 的语义与参数
  输入：pto.initialize_l2l_pipe / tpush / tpop / tfree（未来：mad/matmul）
  输出：pto.bridge_call 内部 op（callee + operand/result）；
        家族级语义（如 TPOP 地址重绑定）在家族 pass 内经 SSA result 完成
  ↓
[通用 C++ 混编 pass 层]（只实现一次，所有家族公用）
  职责：读白名单文本；把 pto.bridge_call 机械转换为
        LLVM 侧 func.call + 外部声明（plannedDecls）；ABI 校验（i64/ptr 约定）
  不感知：TPipe 参数、MATMUL 参数、任何家族的语义
  ↓
[wrapper 生成与 bitcode 合入]
  按白名单 + 家族 pass 收集到的 specialization 生成 wrapper 源码
  → Bisheng 按 target 编译 bitcode → ObjectEmission 合入（现有通道泛化）
```

关键约束：**通用混编 pass 不因新增接口家族而改动**。新增家族 = 新增一个家族专用 pass
+ 白名单条目 + wrapper 模板，通用层与 ObjectEmission 通道保持不变。

### 3.2 白名单文本设计（草案）

文本条目（每条一个 op）暂定字段：

| 字段 | 含义 | 用途 |
|---|---|---|
| `op` | IR op 名称（如 `pto.tpush`） | 路由：命中即桥接候选 |
| `entry` | wrapper 入口名（如 `pto_vpto_pipe_push`） | 通用 pass 发 call 的 callee |
| `abi` | ABI 参数描述（参数个数、i64/ptr 约定） | 调用侧校验 + decl 生成 |
| `tmpl-map` | IR operand/attr → C++ 模板类型参数的映射规则 | wrapper 生成侧（结合家族 pass 的收集结果） |

白名单解决的两个问题分别落到这两组字段：

- **路由**（`op`/`entry`）：不在白名单中的 op 走原有 VPTO 发射路径；白名单中但未被
  家族 pass 转换的 op（例如家族 pass 漏了）→ 报明确诊断，而不是静默走 LLVM IR；
- **类型映射**（`tmpl-map`）：IR 侧只有 `slot_size=1024` 这类整数属性与 tile 类型，
  目标 C++ 侧需要 `TPipe<0, C2V, 1024, 8, 2, false>` 这样的模板实参——映射规则描述
  两者如何对应（复用 EmitC 的 token 构建逻辑，见 §3.3）。

开放问题：白名单格式选型（TD / YAML / TOML，是否与 ODS 联动），见 §5。

### 3.3 类型映射参照 EmitC 逻辑（不改动其实现）

EmitC 已经实现了"IR op 属性 → C++ 模板 token"的完整逻辑，正是白名单 `tmpl-map`
需要的参照素材（`lib/PTO/Transforms/PTOToEmitC.cpp`，**仅作参照**）：

| EmitC 现有函数 | 能力 | 桥接侧用途 |
|---|---|---|
| `getTPipeDirectionToken`（:1142） | dirMask/arch → `DIR_C2V` 等方向 token | TPipe 模板实参 |
| `buildTPipeToken` / `buildTPipeTokenFromInitOp`（:1158/:1170） | op 属性 → `TPipe<flagBase, dir, slotSize, slotNum, ...>` | 同上 |
| `getPipeDataTypeToken`（:8523 调用处） | tile 类型 → `Tile<...>` token | TPUSH/TPOP 的 Tile 实参 |
| `getTileSplitToken` | split → `TileSplitAxis::...` token | split 实参 |

复用方式（**EmitC 实现不动**）：桥接侧新增自己的 utility（如 `BridgeTokenUtils`），
**参照** EmitC 的 token 构建逻辑自行实现，两者互不引用、互不修改：

- EmitC：保持现状，token 拼进输出文本 `TPUSH<pipeTok, tileTok, splitTok>(...)`；
- 桥接：参照同样的构建规则，产出 wrapper 生成器的模板实参清单 + ABI 签名描述。

代价与风险：token 构建逻辑在两个后端各有一份，存在后续漂移的可能——如果将来需要
统一共享，另起变更评估（提取共用 utility 会动到 EmitC 侧调用点），**不在本方案
范围内**；且 EmitC 侧 tile token 依赖**已转换的 OpaqueType 携带 tile 布局**，桥接侧
在 VPTO 管线中取到 tile 类型信息的时点不同（当前 PoC 用 `TileBufType` 的 config
属性），参照移植时需要先对齐"类型信息来源"（见 §5）。

### 3.4 PoC 功能对照（仅参考，不套用实现）

PoC 是**功能参照**：它证明了哪些能力必须存在。新实现不修改、不搬用 PoC 代码，
按下列对照重新实现；验收通过后**删除** PoC 在 emitter 内的两个 pattern
（`LowerPipeBridgeOpPattern` / `LowerPipeTileHandlePattern`）及其固定配置校验。

| PoC 验证过的功能 | 新实现如何覆盖（全新代码） |
|---|---|
| 固定配置校验（emitter :428，错误信息明确） | TPipe 专用 pass；校验依据改为白名单声明 + wrapper 生成能力，不再写死 |
| 固定 C ABI 发 call + `plannedDecls` 声明发射 | 通用混编 pass：`pto.bridge_call` → `func.call` + decl，ABI 按白名单 `abi` 字段校验 |
| `alloc_tile`→规划地址、`declare_tile`→占位 0、`tile_buf_addr`→地址查询、TPOP 重绑定注入 | 家族 pass 负责语义（重绑定经 `bridge_call` 的 SSA result 完成）；tile 句柄 → i64/指针的机械转换进通用层 |
| `ObjectEmission::linkDeviceLLVMBitcode` 合并通道 | 已验证的管线能力，沿用；接口泛化为 bitcode 列表/配置驱动 |
| `VPTOSplitCVModule` 函数级 kind 拆分、`FoldTileBufIntrinsics` 保护、`TileOpExpansionUtils` 排除项 | 已验证的通用管线能力，非桥接机制本体，本次保持不动 |

**分支基址**：本分支基于 `feature/vpto-tpush-tpop-bridge`（保留已验证的管线件与
`fifo-tile-data-consume` 测试资产作为验收目标）；新实现全部为新文件/新 pass。
若后续希望完全基于 main 重做管线件，另议。

### 3.5 通用混编 pass 不感知 TPipe 的实现要点

- 家族 pass 的产物是 `pto.bridge_call`（callee + operand/result），
  不含 `!pto.pipe` / tile 类型等家族类型；家族语义（如 TPOP 重绑定）经
  `bridge_call` 的 SSA result 在家族 pass 内完成；
- 通用 pass 只做三件事：读白名单确认 entry 合法 → 把 `bridge_call` 降为
  `func::CallOp` + decl → 按 `abi` 字段校验参数（i64 地址载体、不透明指针 storage）；
- TPipe 后续任何参数接口变化（新增配置、换 storage 策略）只改 TPipe 专用 pass 与
  wrapper，通用 pass 零改动。

## 4. 改造步骤（建议顺序）

**Phase 0 —— 从零搭建分层骨架（前置）**
不重构 PoC 代码，全部新写：`pto.bridge_call` op、TPipe 专用 pass、通用混编 pass、
`BridgeTokenUtils`（token 构建）。以 PoC 的 `fifo-tile-data-consume` 为验收目标
（相同外部行为、全新内部实现）；验收通过后删除 PoC 的 emitter 内嵌 pattern。
产出：分层骨架，为 Phase 1-3 提供改造点。

**Phase 1 —— 白名单化**
引入白名单文本与读取逻辑；路由决策（桥接 vs LLVM IR 发射）由白名单驱动；
白名单命中但未转换的 op 报诊断。通用混编 pass 改为读白名单而不是内嵌 callee 名。

**Phase 2 —— 参数泛化**
TPipe 专用 pass 从 op 属性读取全部配置（dir_mask/slot_size/slot_num/flag_base/
nosplit/dtype/tile 形状），wrapper 按收集到的 specialization 自动生成（参照 §3.3
在桥接侧实现的 token 逻辑）并编译注入，替换固定的 `TPipe<0, C2V, 1024, 8, 2, false>`。
补配置矩阵测试（不同 slot_size/slot_num、不同 tile 形状）。

**Phase 3 —— 第二接口族：CUBE 侧 MATMUL**
新增 MATMUL 专用 pass + 白名单条目 + MATMUL wrapper。**验收点：通用混编 pass 与
ObjectEmission 通道零改动**。MATMUL 与 pipe 的差异在接口形态而非复杂度：tile 类型
信息分散在 3 个 operand 的类型上（无 storage 生命周期、无地址重绑定），正好检验
白名单 `tmpl-map` 的多来源映射描述力与分层假设。
参考现有文档 `mad-lowering-contract-design.md` / `mad-semantic-op-design.md`。

已核实的真实接口面（`~/pto-isa/include/pto/npu/a5/TMatmul.hpp`）：
主入口调用点模板实参只有 `AccPhase` 一个显式参数，3 个 tile 类型由实参推导：

```cpp
template <AccPhase Phase = AccPhase::Unspecified,
          typename TileRes, typename TileLeft, typename TileRight>
PTO_INTERNAL void TMATMUL_IMPL(TileRes&, TileLeft&, TileRight&);   // :170
```

但家族内存在多个入口变体（quant 双 scale、bias、acc 原地累加等，:103/:185/:207），
白名单条目与 wrapper 生成需要以"家族 + 变体选择"组织，而不是单个函数名——
这正是"家族专用 pass"存在的理由：变体选择（如 quant/bias 形态）属于家族语义。

**Phase 4 —— 工程化收尾**
wrapper 生成进入 ptoas 正式通道（配置驱动，取代环境变量注入）；docs/测试同步。
含 PTO-ISA 头文件路径的自动发现：本机 PTO-ISA 实际位于 `~/pto-isa`（`~/pto-isa/
include/pto/npu/a5/*.hpp`），而 PoC 脚本默认解析到 `llvm-workspace/pto-isa/include`
（不存在，当前依赖用户显式导出 `PTO_ISA_INCLUDE_DIR`）——正式通道需支持多候选路径
探测 + 显式配置。

## 5. 决策记录（原开放问题，已定案）

1. **白名单格式：YAML**。
   - 理由：本工作区 LLVM 无 `llvm::toml`（`Toml.h` 不在树），而 `llvm::yaml`
     （YAMLParser/YAMLTraits）在树且属于 LLVMSupport、无额外链接依赖；
     结构化 + 注释，pass 与 wrapper 生成器同为 C++ 侧，消费同一文件；
   - **不与 ODS 联动**：白名单是"桥接路由与映射"的工具链策略配置，不属于 IR 定义；
     与 ODS 联动会把桥接策略绑进 dialect，违背"通用通道"定位；
   - Python 绑定侧复用暂不作为约束（需要时另议）。
   - 条目 schema 草案（`llvm::yaml` 映射 IO 实现）：
     ```yaml
     # vpto-bridge-whitelist.yaml
     bridge_ops:
       - op: pto.tpush                # 路由：命中即桥接候选
         family: pipe                  # 家族（决定家族 pass 与 wrapper 模板）
         entry: pto_vpto_pipe_push     # wrapper 入口名
         abi:
           - arg: storage
             type: ptr                 # ptr | i64 | i32 ...
           - arg: tile
             type: i64
         tmpl_map:
           - source: pipe.init         # 模板实参来源（op 属性 / operand 类型）
             field: slot_size          # 抽取字段
             target: Pipe.slotSize     # 目标模板实参
     ```
     完整字段集在 Phase 1 落地时细化。
2. **中间形态：新内部 op `pto.bridge_call`**（StringAttr `callee` + 可变
   operand/result），不用属性标注的 `func.call`。
   - 理由：显式 op 可被 legalize 校验——白名单命中但家族 pass 未转换的 op 直接
     illegal 报错，诊断清晰；属性标注易被通用变换丢失/忽略；
   - **家族语义经 SSA result 完成**：如 TPOP 重绑定，由家族 pass 把后续 tile 使用
     替换为 `bridge_call` 的 result（i64 slot 地址）——重绑定在家族 pass 内完成，
     通用 pass 只做"op → call + decl"的机械降级，彻底不感知家族语义；
   - 未来家族可携带结构化动作属性（如 subblock 绑定），不依赖 func.call 语义。
3. **EmitC 参照边界（EmitC 侧零改动）**：
   - **照搬**：与上下文无关的纯映射逻辑（dirMask→方向 token、split→
     `TileSplitAxis` token、`TPipe<...>` 字段拼接顺序）；
   - **桥接侧自建**：tile token 构建（来源 = `TileBufType` config 属性，
     不复用 EmitC 对已转换 OpaqueType 的依赖）；
   - **漂移兜底**：新增 lit 比对测试——同一组 op 输入下，EmitC 输出文本中的模板
     token 与 `BridgeTokenUtils` 产出逐项一致。
4. **MATMUL 多来源映射：不回退**。`tmpl-map` 以"声明 + 家族 pass 组装"为设计：
   白名单条目声明哪些 operand 的哪些字段进入哪个模板槽位；具体收集与跨 operand
   一致性校验（如 lhs.cols == rhs.rows）由家族 pass + `BridgeTokenUtils` 实现，
   校验失败报家族级诊断。原"家族 pass 直接产出 wrapper 生成描述"的形态即常规路径
   （wrapper 生成描述本就是家族 pass 的输出物），白名单负责其中的声明性部分，
   不再作为回退分支存在。
5. ~~遗留问题（控制流下 TPOP 重绑定、split≠1、无 finish 假设）~~ **本阶段不考虑**：
   PoC 的遗留边界不随本次泛化带入。重绑定在新设计中经 `bridge_call` 的 SSA result
   实现（见决策 2），控制流覆盖等语义问题到 Phase 2/3 的测试中按需暴露、按需处理。

---

## 附：Phase 0 实施记录（2026-08-21）

Phase 0 已完成并验收通过。与计划的偏差与实施要点：

1. **构建环境切换 LLVM 19**：桥接分支原要求 LLVM 21，本机仅有 LLVM 19.1.7 构建。
   将 main 的 `eb79e5f9 build: downgrade PTOAS to LLVM 19` cherry-pick 到本分支；
   该提交引用的 VPTOScheduler（main 上另一个提交的产物，未随 cherry-pick 带入）
   从 ptoas.cpp 与 lit 工具依赖中移除，lit 恢复 `pto-vfsimt-size-patcher-test`/`yaml2obj`。
2. **通用 pass 的真实插入点**：A5 测试路径走 CANN900 的 `runPipeline`
   （VPTOCANN900LLVMEmitter.cpp），而非 Beta1 的 runPipeline——两处都已插入
   `pto::createVPTOBridgeLoweringPass()`。
3. **ConversionTarget 陷阱**：MLIR 转换目标默认对未知 op 是"不合法"的——pattern
   创建的 `func.call`/`llvm.alloca`/私有声明会被驱动拒绝并整体回滚 pattern。
   修复：`target.markUnknownOpDynamicallyLegal([](Operation *) { return true; })`。
4. **依赖方言必须 override**：手动 `createXPass()` 不会带出 Passes.td 的
   `dependentDialects` 声明，需在 pass 类上 `getDependentDialects` override
   LLVM/func 方言（否则在未加载 LLVM 方言的 context 中运行会 abort）。
5. **重绑定实现形态**：家族 pass 中 storage 经 `bridge_call` 的 **SSA result**
   流转（init 的 result RAUW 给 push/pop/free 的 pipe operand），TPOP 的 i64
   result 经 `bridge_inttoptr` 物化为指针并 RAUW 给 `tile_buf_addr` 结果——
   比 PoC 的 `popTileAddresses` 映射更显式，且全部语义留在家族 pass 内。
6. **家族 pass 转换范围与 PoC 一致**：转换函数内**所有** `alloc_tile`/
   `declare_tile`/`tile_buf_addr`（不限 pipe 相关）——与 PoC emitter 行为对齐；
   Phase 1 值得收窄为"仅 pipe 家族相关的 tile 句柄"。
7. **验收**：COMPILE_ONLY device 编译通过；CA 模拟器端到端 compare passed
   （128 f32 与 golden 全量一致）；`tilelib_passes_skip_frontend_pipe_ops` lit
   通过；PoC 的 `LowerPipeBridgeOpPattern`/`LowerPipeTileHandlePattern` 及
   `popTileAddresses` 已从 emitter 删除。
8. **未随 Phase 0 落地**：`BridgeTokenUtils`（token 构建）与 wrapper 自动生成
   属 Phase 2，与参数泛化一起实现；白名单 `tmpl_map` 字段亦留待 Phase 2 消费。

## 附：Phase 1 实施记录（2026-08-23）

Phase 1（白名单化）已完成，Phase 0 记录的遗留项“收窄 tile 句柄转换范围”一并解决：

1. **白名单驱动路由**：家族 pass 新增 `whitelist-path` option（与通用 pass 相同的
   `PTOAS_VPTO_BRIDGE_WHITELIST` env 回退），按 IR op 名（`findOp`）查白名单取
   wrapper entry；删除了全部硬编码的 `pto_vpto_pipe_*` 常量。pipe op 无白名单路由
   时报明确诊断（pipe op 没有非桥接的 VPTO 降级路径，不做静默回退）。
2. **schema 扩展**：init entry 新增 `storage_size_entry` 字段显式关联 size 查询
   entry（取代家族 pass 硬编码）；`op: internal` 定型为 wrapper 内部辅助条目标记
   （`BridgeWhitelist::kInternalOp`），不参与路由。
3. **残留诊断**：通用 pass 降级前按白名单路由表走查 IR——命中白名单但未被家族
   pass 转换的 op 报明确错误（指出 op 名、wrapper entry 与白名单路径），不再依赖
   emitter illegal 列表的通用报错。
4. **转换范围收窄**：函数内无 pipe 家族 op 时家族 pass 直接返回，tile 句柄保持走
   常规 `FoldTileBufIntrinsics` 路径。修复了 Phase 0 的隐患：无 pipe 的
   internal-IR 输入此前会误产 bridge op 并强制要求白名单。已核实 main 的 VPTO
   emitter 本就不处理裸 tile 句柄 op，收窄不丢失任何既有能力。
5. **共用与校验**：`resolveBridgeWhitelistPath`（option → env → 空）提取到
   `VPTOBridgeWhitelist`，两个 pass 共用；白名单解析新增空字段、重复 `op` 名、
   `storage_size_entry` 悬空引用校验。
6. **lit 测试**：新增 4 个测试（`test/lit/vpto/vpto_bridge_*.pto` + fixture
   `test/lit/vpto/Inputs/vpto-bridge-whitelist.yaml`）：两层 pass 链式降级输出、
   无 pipe 函数跳过、白名单残留诊断、无白名单诊断。
7. **验收**：新增 lit 全过；全量 lit 回归 1762 通过（4 个失败为分支既有，经
   stash 对照确认与桥接无关）；ptoas CLI 三路手工验证——fifo 带白名单走到
   emission（本机无 CANN 工具链，与 Phase 0 同口径）、fifo 无白名单报家族 pass
   诊断、非 pipe 内核无白名单不受影响。**CA 模拟器端到端 compare 需板端环境，
   建议在板端重跑一次 `fifo-tile-data-consume` 用例收尾**。
8. **Phase 2 入口**：`BridgeTokenUtils`、wrapper 自动生成、白名单 `tmpl_map`
   消费与配置矩阵测试。

## 附：Phase 2 实施记录（2026-08-24）

Phase 2（参数泛化 + wrapper 自动生成，完全内置到 ptoas）已完成并验收通过：

1. **配置校验移除**：家族 pass 不再拒绝非默认的 `slot_size`/`slot_num`/
   `flag_base`/`nosplit`，pipe 属性原样流入收集到的桥接特化；
   `BridgeTokenUtils`（`VPTOBridgeTokens.h/.cpp`）把特化渲染为
   `pto::TPipe<...>`/`pto::Tile<...>`/`pto::TileSplitAxis::*` C++ 模板 token
   （含元素类型映射，如 f16→half；NoneBox tile 省略 SLayout/SFractalSize
   模板参数）。
2. **并发竞态与修复**：家族 pass 是 func 级 nested pass，MLIR PassManager
   会多线程并发执行各函数实例——最初在家族 pass 内直接 read-modify-write
   模块 spec 属性导致字段随机丢失（lit 表现为 `producer_tile` 缺失）。
   修复为两级：家族 pass 只写自己函数上的 `pto.vpto.bridge.func_spec`
   DictionaryAttr；模块级 wrapper 生成 pass 单线程做
   `mergeFuncSpecsIntoModule` 确定性合并，字段冲突（同一 key 不同值）报
   "conflicting pipe bridge specialization" 诊断。
3. **wrapper 生成 pass**：新增模块 pass `pto-emit-vpto-bridge-wrapper`，
   合并后渲染 `pto.vpto.bridge.wrapper_source` StringAttr——TPipe/Tile
   typedef + `__DAV_CUBE__`/`__DAV_VEC__` 守卫的 init/push/pop/free/size
   入口；producer/consumer 角色按 pipe 方向（C2V/V2C）自动对调。
4. **emission 内置编译**：ObjectEmission 读取 wrapper_source，cube/vec 两路
   fatobj 编译各自注入。手写 `vpto_bridge.cpp` 与
   `PTOAS_VPTO_{CUBE,VEC}_BRIDGE_BITCODE` env 通道已删除，
   `run_host_vpto_validation.sh` 仅保留 `PTOAS_VPTO_BRIDGE_WHITELIST` 注入。
5. **白名单 `tmpl_map`**：schema 定型为 `{source, field, target}` 行，
   pipe 家族 source 仅允许 `pipe.init`/`tile`，缺键由 YAML 层报
   `missing required key`，未知 source 由校验器报明确诊断；本阶段行内容
   仅供校验消费，不改变收集字段。
6. **lit 测试**：新增 5 个用例 + 3 个 fixture——配置矩阵（变体
   slot_size=2048/slot_num=4/flag_base=8/nosplit=true、f16、acc 32×16 /
   vec 4×16，双前缀查 func_spec 与 wrapper token）、C2V/V2C wrapper 源
   FileCheck、tmpl_map 缺键/未知 source 诊断、跨函数 spec 冲突诊断。
   泛型 op 打印的操作数类型（`: !pto.pipe, i64`）与单键字典裸键名打印
   都需在 CHECK 模式里对齐。
7. **验收**：生成 wrapper 与手写 `vpto_bridge.cpp` diff 结构等价（差异仅
   typedef 命名/参数名/格式）；新增 lit 9/9 过（4 个 Phase 1 + 5 个新增），
   全量 lit 回归 1767/1772（4 个失败为分支既有，RUN 行均不含桥接 pass）；
   **CA 模拟器（dav_3510）端到端**：fifo 原样回归 compare passed；变体配置
   （slot_num=4、flag_base=8、fifo 容量 4096）compare passed，fatobj 内
   5 个 wrapper 入口符号确认注入。变体用例为一次性验证副本
   （`build/vpto-variant-cases/`），未转正为常驻用例。
8. **遗留**：`split != 1` 仍报诊断（对齐设计 §5）；`tmpl_map` 行目前仅校验
   不消费，真正驱动模板参数替换留待后续阶段。

## 附：Phase 3 实施记录（2026-08-24）

Phase 3（第二接口族：CUBE 侧 MATMUL）已完成并验收通过，**通用混编 pass
（`VPTOBridgeLowering.cpp`）与 ObjectEmission 通道零改动**：

1. **家族 pass**：新增 `PTOLowerMatmulFamilyOps`，将 `pto.tmatmul` /
   `pto.tmatmul.acc` 降为 `pto.bridge_call`，ABI 值为规划地址 i64 序列
   （tmatmul：dst/lhs/rhs 3 个；acc 变体：dst/accIn/lhs/rhs 4 个）。
   与 pipe 的差异符合设计预期：无 storage 生命周期、无地址重绑定、
   无 `storage_size_entry`；dst 必须来自带规划地址的 `alloc_tile`，
   否则报 "the result tile must come from an alloc_tile with a planned
   address" 诊断。
2. **wrapper 生成扩展**：`VPTOBridgeWrapperGen` 渲染 matmul 家族入口
   （`__DAV_CUBE__` 守卫）：`pto::TMATMUL(dst, lhs, rhs)` /
   `pto::TMATMUL_ACC(dst, acc, lhs, rhs)`；`accPhase` 枚举属性
   （`{accPhase = #pto<acc_phase final>}`）作为模板实参渲染。
   注意 `acc_phase` 与其他 spec 字段一样经 `mergeFuncSpecsIntoModule`
   模块共享——任一函数带 Final phase 会使全模块入口统一渲染
   `TMATMUL<pto::AccPhase::Final>`；单内核共享 phase 策略，判定为合理。
3. **白名单扩展**：`pto.tmatmul` → `pto_vpto_matmul`（3×i64）、
   `pto.tmatmul.acc` → `pto_vpto_matmul_acc`（4×i64）；`tmpl_map` source
   校验扩展 matmul 家族来源集：`left_tile`/`right_tile`/`result_tile`/
   `acc_in_tile`，未知 source 报明确诊断（含文件路径与支持集）。
4. **lit 测试**：新增 3 用例 + 2 fixture——家族 lowering 的 spec/wrapper
   源双前缀检查、无地址 dst 诊断、白名单 matmul 未知 source 诊断；
   9 个既有 bridge 用例全过。
5. **验收**：全量 lit 回归 1775 测试、1770 过，仅 4 个分支既有失败
   （与 Phase 2 基线一致，RUN 行均不含桥接 pass），证实分层假设——
   家族差异全部封装在家族 pass + wrapper 渲染内。
6. **模拟器端到端**：新增常驻用例 `test/vpto/cases/kernels/cube-matmul-bridge`
   （16×16×16 f16，A=单位阵、B=0.25×arange，golden 为 f32 矩阵乘），
   内核走 mte_gm_l1 → mte_l1_l0a/l0b{transpose} → tmatmul（经桥接 wrapper）
   → mte_l0c_gm(nz2nd) → `barrier PIPE_ALL`；DEVICE=SIM compare passed。
   要点：显式 `alloc_tile addr =` 仅 `--pto-level=level3` 可用，本 harness
   默认 level 下用普通 `pto.alloc_tile`（由共享管线中先行运行的
   PlanMemory 赋地址）+ `pto.tile_buf_addr` 取 l0a/l0b/l0c 指针。
7. **遗留**：TMatmul.hpp 的 quant 双 scale/bias 等入口变体未接入
   （家族 + 变体选择结构已就位）；`tmpl_map` 行仍仅校验不消费，
   与 Phase 2 同口径，留待 Phase 4。

## 附：Phase 4 实施记录（2026-08-24）

Phase 4（工程化收尾：白名单正式通道 + tmpl_map 消费）已完成并验收通过：

1. **白名单正式通道**：新增 `loadBridgeWhitelist` 三级解析链（pass
   `whitelist-path` option → `PTOAS_VPTO_BRIDGE_WHITELIST` env → 内置默认
   白名单 `kDefaultBridgeWhitelistYaml`，pipe + matmul 全条目）；
   `parseBridgeWhitelistFromBuffer` 从文件解析中提取，诊断以
   `<built-in vpto bridge whitelist>` 标记来源。pipe 家族 pass 与通用
   lowering 的“无白名单配置”诊断路径删除——`ptoas --pto-backend=vpto`
   开箱即用；matmul 家族 pass 不再跳过未配置场景，默认改走桥接，
   需要 mad 展开的内核用显式空路由白名单退出（见 5）。
2. **tmpl_map 消费**：wrapper 生成 pass 新增 `whitelist-path` option 与
   `validateTmplMapCoverage`——模块实际用到的每个白名单条目，其
   tmpl_map 声明的每个模板槽位必须有家族 pass 收集到的 spec token
   覆盖（source → spec key 映射经 `tmplMapSourceSpecKeys`），未覆盖报
   明确诊断而非静默丢弃。字段级模板实参构建仍以 `VPTOBridgeTokens`
   为权威（决策 4），本阶段“消费”落在覆盖校验形态；tmpl_map 行
   驱动渲染替换收集 token 留待引入新变体/字段时再做。
3. **脚本与用例清理**：`run_host_vpto_validation.sh` 删除 per-case
   白名单探测与 env 注入（用户导出的 env 自然继承）；删除
   `fifo-tile-data-consume` / `cube-matmul-bridge` 两个与内置默认
   功能等价的 `vpto-bridge-whitelist.yaml`，端到端用例直接依赖内置默认。
4. **PTO-ISA 头文件自动发现**：核实已由 ObjectEmission 的
   `discoverCppIncludeDirs` 覆盖（main 2026-05 driver 提交引入：
   `PTO_ISA_PATH`/`PTO_ISA_ROOT` env → `~/pto-isa` →
   `~/llvm-workspace/pto-isa` 探测）；原计划提到的
   `PTO_ISA_INCLUDE_DIR` 随 Phase 2 的手写通道一并删除，已不存在于
   代码，本次未新增代码，仅在此更正原 Phase 4 描述。
5. **lit 测试**：`vpto_bridge_no_whitelist_diag` 更名为
   `vpto_bridge_default_whitelist_lowering` 并改为内置默认下 pipe op
   正向降级的验证（`env -u` 排除机器 env 干扰）；新增
   `vpto_bridge_whitelist_tmpl_coverage_diag` 与 2 个 fixture
   （matmul-uncovered-tmpl / no-routing）；
   `expand_tile_op_tilelang_tmatmul` 改用空路由白名单退出默认桥接路由。
   全量回归 1776 用例、1771 过，4 个失败为分支既有（与 Phase 3 基线
   一致，RUN 行均不含桥接 pass）。
6. **模拟器端到端（dav_3510）**：零 env 注入
   （`env -u PTOAS_VPTO_BRIDGE_WHITELIST`）下
   `fifo-tile-data-consume` 与 `cube-matmul-bridge` 均 compare passed，
   wrapper 编译的 PTO-ISA 头文件经 `~/pto-isa` 候选自动命中。
7. **遗留**：TMatmul.hpp 的 quant 双 scale/bias 等入口变体未接入
   （家族 + 变体选择结构已就位，新增时需同步扩展内置默认白名单）；
   tmpl_map 行仍处于校验形态（见 2）。
