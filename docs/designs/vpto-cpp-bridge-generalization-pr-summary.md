# VPTO C++ 接口桥接泛化——技术汇报


## 1. 背景

VPTO 后端直接发射 LLVM IR，无法实例化 PTO-ISA 中以 C++ 模板实现的硬件接口
（TPush/TPop/TFree FIFO 协议、TMATMUL 等）。前期 PoC 已验证了一条可行路径：
把 Bisheng 实例化的 C++ 模板 wrapper 编译为 device bitcode，在 bitcode 层与
VPTO 生成的 LLVM IR 合并，端到端跑通 FIFO 通路。但 PoC 有两个必须解决的局限：

1. **参数硬编码**：emitter 内写死 `slot_size=1024, slot_num=8, flag_base=0,
   nosplit=false` 等配置，wrapper 写死单一 `TPipe` 特化与固定 tile 形状；
2. **机制与 TPipe 语义耦合**：参数转换与 C ABI 调用发射混在同一个 pattern，
   第二个接口族（如 MATMUL）无法复用。

本工作将桥接构建为**所有 PTO-ISA C++ 接口公用的编译器内置通道**：

- **参数由 IR 属性驱动**——任意合法配置组合均可从 op 属性流入，无硬编码；
- **路由由白名单驱动**——哪些 op 走桥接、映射到哪个 wrapper 入口，全部声明化；
- **wrapper 自动生成**——编译器按收集到的特化信息渲染 wrapper 源码并编译合入，
  全流程无外部脚本、无环境变量依赖；
- **扩展接口家族时通用层零改动**——机械映射型 op 经白名单声明即接入声明式
  通道，无需专用 pass 且无需额外标注（声明式是缺省通道）；仅含真家族语义
  （地址重绑定、storage 生命周期）的 op 以 `lowering: custom` 显式退出、
  走家族通道，差异封装在白名单条目与 wrapper 模板内。

## 2. 整体架构

三层分离，核心约束是**通用混编 pass 不因新增接口家族而改动**：

```text
[op 路由层]                  PTOLowerDeclarativeBridgeOps（声明式，缺省） / PTOLowerPipeFamilyOps（家族）
  声明式通道（白名单条目缺省归属）：白名单驱动的机械降级（abi 行的
  operand 位置绑定 → 规划地址、role → tile token、attr 行 → 枚举属性
  token）；
  家族通道（条目需显式 `lowering: custom` 退出声明式）处理真家族语义
  （TPOP 地址重绑定经 bridge_call 的 SSA result 在本层内完成、storage
  生命周期）
        ↓ 产出 pto.bridge_call（callee + operand/result，不携带家族类型）
[通用混编 pass 层]        VPTOBridgeLowering
  读白名单确认 entry 合法 → bridge_call 机械降级为 func.call + 外部声明
  按白名单 abi 字段校验参数（i64 地址载体 / 不透明指针 storage），
  不感知任何家族语义
        ↓
[wrapper 生成与 bitcode 合入]
  VPTOBridgeWrapperGen（模块级）：合并各函数收集的 spec →
  按白名单 tmpl_map 渲染 wrapper C++ 源码（模块属性 wrapper_source）→
  ObjectEmission 将 wrapper 编译为 cube/vec 两路 bitcode 注入各自 fatobj
```

关键设计决策及理由：

| 决策 | 理由 |
|---|---|
| 中间形态用显式内部 op `pto.bridge_call`，不用属性标注的 `func.call` | 可被 legalize 校验：白名单命中但未被家族 pass 转换的 op 直接报明确诊断，不静默回退 |
| YAML 白名单（不与 ODS 联动） | 本工作区 LLVM 无 `llvm::toml`，`llvm::yaml` 在树无额外依赖；桥接属工具链策略配置而非 IR 定义 |
| token 映射规则共享 `PTOCppTokens`、组装层桥接侧自建（`VPTOBridgeTokens`） | 两侧的 tile 信息来源与产物消费形态不同，组装层无法直接共享；lit 比对测试防漂移（详见 §3.2） |
| 家族语义经 SSA result 完成 | 通用 pass 只做机械降级，TPipe 参数接口后续任何变化只改家族 pass 与 wrapper |
| 声明式/自定义两档路由（白名单 `lowering` 字段） | 分类轴是变换复杂度而非家族：机械映射型 op 零 pass 代码接入，真家族语义封装在专用 pass 内 |
| `lowering` 缺省为声明式，专用 pass 需显式 `lowering: custom` | 仪式成本落在例外一侧：新增机械映射家族零 pass 代码且零标注。且误用可自检——漏标的条目缺少 `operand`/`arg`/`role` 绑定，解析期即报错并点名缺失字段，而非降级后才在残留检查处指向一个并不存在的家族 pass |
| 内置默认白名单编译进 ptoas | `ptoas --pto-backend=vpto` 零环境变量开箱即用；特殊场景可用 option/env 覆盖，或以空路由白名单退出桥接 |

## 3. 核心机制设计

### 3.1 白名单：路由与类型映射声明

白名单同时承担**路由**与**类型映射声明**两个职责：

```yaml
wrappers:                        # 声明式 wrapper 的生成声明（每个纯声明式 wrapper 一条）
  - name: matmul
    includes: [pto/npu/a5/TMatmul.hpp]   # wrapper 源码的 include 行
    core: cube                   # cube | vec | both（决定 __DAV_*__ 守卫）
bridge_ops:
  - op: pto.tmatmul              # 路由键：IR op 名，命中即桥接候选
    wrapper: matmul              # 渲染进哪份 wrapper 源码（生成单位）
                                 # lowering 省略 = 声明式（缺省通道）
    call: pto::TMATMUL           # 调用拼写（声明式条目必填）
    tmpl_args: [acc_phase]       # 可选模板实参：本条目 attr 行 field 或含 :: 字面量
    abi:                         # operand 位置绑定 + 角色（调用侧校验 + decl 生成）
      - {operand: 2, arg: dst, role: result_tile}   # type 缺省 i64
      - {operand: 0, arg: lhs, role: left_tile}
      - {operand: 1, arg: rhs, role: right_tile}
    tmpl_map:                    # 声明式条目只收 attr 行（tile typedef 由 role 推导）
      - source: attr
        field: acc_phase
        target: AccPhase
        enum_type: pto::AccPhase
        omit_value: Unspecified
  - op: pto.tpush
    wrapper: pipe                # 同一 wrapper 的条目共享一份翻译单元
    lowering: custom             # 例外：家族语义需专用 pass，必须显式标注
    entry: pto_vpto_pipe_tpush   # custom 条目必填
    abi:                         # custom 条目无需 operand/role 绑定
      - {type: ptr}              # storage，由桥接降级合成
      - {type: i64}              # producer tile 地址
    tmpl_map:
      - {source: tile, field: tile, target: ProducerTile}
```

- **两档通道，缺省为声明式**：`lowering` 省略即走声明式通道——机械映射型
  op 经白名单声明即接入，无需任何 pass 代码。只有携带真家族语义（storage
  生命周期、地址重绑定）的条目才显式 `lowering: custom` 退出该通道。缺省
  方向使得误用可自检：漏标 `custom` 的条目因缺少 `operand`/`arg`/`role`
  绑定在解析期即被拒绝，诊断直接点名该标注；
- **内置默认白名单**共 12 条：`wrapper: pipe` 4 条路由条目（均 `custom`）+
  1 条 `op: internal` 的 size 查询辅助条目（不参与路由），`wrapper: matmul`
  6 条路由条目（`tmatmul` 与 `.acc`/`.bias`/`.mx`/`.mx.acc`/`.mx.bias`
  变体）与 `wrapper: vec_elem` 1 条（`tadd`），均走缺省声明式通道。
  声明式 wrapper 在顶层 `wrappers` 段声明 includes 与 core 守卫；storage 等
  不透明指针参数以 `{type: ptr}` 行声明；init 条目经 `storage_size_entry`
  字段显式关联 size 查询入口；
- **声明式条目的字段约束**：`call` 必填（调用拼写）；`tmpl_args` 每项必须
  是本条目 `tmpl_map` attr field 或含 `::` 的字面量；`tmpl_map` 只收
  `source: attr` 行（tile typedef 唯一来源是 abi role，解析期拒绝 tile 行）；
  `entry` 可省略，缺省从 op 名推导（`pto.tadd` → `pto_vpto_add`），
  custom 条目必填且禁用 `call`/`tmpl_args`；
- **声明式条目的结构化校验**：abi 行必须带 `operand`/`arg`/`role`，operand
  索引不重复；`attr` source 必须声明 `enum_type`。`op: internal` 条目从不被
  路由，豁免上述通道校验；
- **三级解析链**：pass `whitelist-path` option → `PTOAS_VPTO_BRIDGE_WHITELIST`
  环境变量 → 内置默认白名单；
- **解析期校验**：空字段、未知 `lowering` 值、重复 `op` 名、
  `storage_size_entry` 悬空引用、`tmpl_map` 缺键与未知 source 均报诊断
  （含文件路径与支持集）。

> `wrapper` 是**生成单位**，也是 wrapper 生成器的分派键：同名条目共享一份
> C++ 翻译单元与一套 typedef，每个 module 最多一个。分派只看 wrapper 内
> 条目通道：纯声明式条目走**通用渲染器**（从 `wrappers` 段取 includes 与
> core 守卫，机械渲染所有使用中条目，新增机械映射家族零 C++ 改动）；
> 含 `lowering: custom` 条目走**专用渲染器**（现仅 pipe）——自定义
> wrapper 无专用渲染器时诊断列出可用集合，声明式 wrapper 缺 `wrappers`
> 段声明报错。家族 pass 的选择仍由 pass 流水线决定——那与 wrapper 无关，
> 是 `lowering: custom` 的职责。

### 3.2 参数传递机制：IR 配置 → C++ 模板实参

**要解决的问题**：IR 侧的参数是平铺的整数属性与 tile 类型，而 PTO-ISA
接口是 C++ 模板，需要完整的模板实参拼写。例如一条带 `slot_size=1024,
slot_num=8, flag_base=0, nosplit=false` 属性、方向 C2V 的 pipe init，
桥接最终要能为 wrapper 产出：

```cpp
pto::TPipe<0, pto::Direction::DIR_C2V, 1024, 8, 2, false>
```

这类字符串（下称 **token**）。参数传递机制的职责就是完成这一步翻译，
并保证翻译结果与 EmitC 后端一致。

**参照了 EmitC 的什么**：EmitC 路径（`PTOToEmitC.cpp`）早已解决过同类
问题——它把 PTO op 翻译为 C++ 源码文本时，同样需要把 IR 事实逐项翻译
成 PTO-ISA 的 C++ 拼写，并已沉淀出一套经生产验证的映射规则：

| IR 事实 | 映射产出的 C++ token |
|---|---|
| MLIR 元素类型 | 元素类型拼写（`float` / `half` / `int8_t` …） |
| pipe dir_mask（1=C2V / 2=V2C / 3=BOTH） | `Direction::DIR_*` |
| split 值（0..4） | `TileSplitAxis::TILE_*` |
| tile 地址空间 | `TileType::*` |
| BLayout / SLayout 枚举 | `BLayout::*` / `SLayout::*` |
| flag_base/slot_size/slot_num/… 属性组 | `TPipe<...>` 完整拼写 |

桥接没有重写这些规则：这套纯映射已抽取为共享层 `PTOCppTokens`，
EmitC 与桥接两侧都调用它（限定符参数区分拼写形态：桥接传 `pto::`
全限定前缀，EmitC 传空前缀，依赖输出上下文的命名空间）。

**为什么组装层不能直接复用**：共享的只是"IR 事实 → 单个 token"的映射；
把 token **组装**成最终产物的环节，两侧约束不同：

| 环节 | EmitC 路径 | 桥接路径 |
|---|---|---|
| tile 信息从哪来 | 读已转换的 OpaqueType 携带的 tile 布局 | 桥接时点直接读 `TileBufType` 的 config 属性 |
| 产物去哪 | 拼进输出文本，如 `TPUSH<pipeTok, tileTok, ...>(...)` | 写入 spec，由 wrapper 生成器渲染 typedef 与模板实参 |


因此组装层桥接侧自建：`VPTOBridgeTokens`（`buildBridgePipeToken` /
`buildBridgeTileToken` / `buildBridgeTileSplitToken` /
`buildBridgeElementTypeToken`）产出可直接替换进 wrapper 源码的 token。



**参数完整流转链路**（收集 → 合并 → 渲染）：

- **收集**：声明式/家族 pass 从 op 属性/operand 类型抽取配置，经
  `VPTOBridgeTokens` 解析为 token，写入函数级
  `pto.vpto.bridge.func_spec` 属性（func 级 nested pass，并发安全）；
- **合并**：模块级 wrapper 生成 pass 单线程确定性合并各函数 spec，
  同 key 异值报 "conflicting … bridge specialization" 诊断；
- **渲染**：声明式条目的 tile typedef 由 abi role 直接推导（`left_tile` →
  `using LeftTile = <spec token>;`），按 target 名去重后字母序输出；
  `tmpl_map` 对声明式条目只保留 `attr` 行，其 token 喂给条目的 `tmpl_args`，
  spec 缺 token 时整个模板实参列表省略（attr token 是枚举值而非类型，
  从不渲染 typedef）；pipe 的 typedef 段由 pipe 专用渲染器自建
  （Pipe/ProducerTile/ConsumerTile）。渲染前做必需槽位兜底校验，未覆盖
  报明确诊断而非静默丢弃。

### 3.3 wrapper 自动生成与 bitcode 合入

```text
声明式/家族 pass（func 级，可并发）  收集 spec → 函数属性
        ↓
mergeFuncSpecsIntoModule（模块级，单线程）   确定性合并 + 冲突诊断
        ↓
VPTOBridgeWrapperGen             按条目通道分派渲染 wrapper_source：
                                 纯声明式 wrapper 走通用渲染器（wrappers 段
                                 includes/core 守卫 + role typedef + extern "C"
                                 入口 + TASSIGN + call）；custom wrapper 走
                                 专用渲染器（pipe init/push/pop/free/size 入口，
                                 producer/consumer 角色按 pipe 方向自动对调）
        ↓
ObjectEmission                   读 wrapper_source，cube/vec 两路各自编译
                                 bitcode（PTO-ISA 头文件经多候选路径自动发现）
                                 → linkDeviceLLVMBitcode 合入 device 模块
```

全流程内置于 ptoas：`ptoas --pto-backend=vpto` 一条命令完成桥接降级、
wrapper 渲染、编译与合入，无外部脚本、无环境变量依赖。

## 4. 已验证的能力矩阵

### 4.1 pipe wrapper（TPipe/TPUSH/TPOP/TFREE）

| 维度 | 支持情况 |
|---|---|
| 配置参数 | `slot_size` / `slot_num` / `flag_base` / `nosplit` 任意组合，由 op 属性流入，无硬编码校验 |
| 方向 | C2V / V2C，producer/consumer 角色自动对调 |
| 数据类型 | 常规 dtype 含 f16（`half` 映射） |
| tile 形状 | 任意（含 NoneBox：省略 SLayout/SFractalSize 模板参数） |
| split | `split≠1` 支持（split=2 → `TileSplitAxis::TILE_LEFT_RIGHT`）；函数内共享单一 split token，异值报诊断 |
| 控制流 | 循环内 push/pop/消费/free 验证正确（单 pop op 每迭代绑定当次 fifo slot） |
| 生命周期 | init（storage 经内置 size 查询分配）→ push/pop → free；TPOP 地址重绑定经 SSA result |

### 4.2 matmul wrapper（`pto.tmatmul` 及 5 个变体）

| 项 | 支持情况 |
|---|---|
| ABI | 3×i64（dst/lhs/rhs），经白名单 abi 行位置绑定与校验 |
| dst 校验 | 必须来自带规划地址的 `alloc_tile`，否则报明确诊断 |
| 模板实参 | tile typedef 由 abi role 推导；`tmpl_args` 声明驱动（`acc_phase` → `AccPhase::Final`，spec 缺 token 则省略整个列表） |
| 端到端 | 16×16×16 f16 矩阵乘（A=单位阵），mte 链路→tmatmul→mte 出，compare passed（声明式重构后复跑通过） |

### 4.3 vec_elem wrapper（`pto.tadd`，纯白名单接入）

| 项 | 支持情况 |
|---|---|
| 接入方式 | 零 C++ 改动：仅白名单注册（`wrappers` 段声明 TAdd include 与 vec 核守卫 + 一条 op/wrapper/call/abi 条目），`pto::TADD(dst, src0, src1)` 调用由通用渲染器渲染 |
| ABI | 3×i64（dst/src0/src1），无模板实参（取 a5 推导友好入口，ElementsPerRepeat/validRows 内部推导） |
| 渲染验证 | lit 覆盖 spec 收集（entry 推导 `pto_vpto_add`）与 wrapper 源码（`__DAV_VEC__` 守卫 + TADD 调用体） |

### 4.4 机制验证结论

- 接口面完全不同的第二家族（MATMUL）接入时，通用混编 pass 与
  ObjectEmission 通道零改动，通用层与家族语义解耦的分层设计成立；
- matmul 以声明式通道接入，无专用 pass 代码，机械映射型 op 的接入成本
  仅为一条白名单声明——且声明式为缺省通道，该声明连 `lowering` 标注都
  不需要写；
- matmul 原硬编码渲染器已迁移为通用声明式渲染器（迁移 litmus：渲染输出
  逐字节一致），并删除 matmul 专用渲染器、spec 结构与 entry key 常量；
  tadd/vec_elem 随即**纯白名单接入**（零 C++ 改动），验证了机械映射型
  op "白名单注册即桥接" 的目标：通用渲染器对 cube/vec 核守卫、有无模板
  实参的调用形态均统一覆盖。

## 5. 测试验证

lit 侧新增/调整：

| 用例 | 覆盖点 |
|---|---|
| `vpto_bridge_whitelist_default_channel_diag.pto` | 缺省通道的两条边界：漏标 `lowering: custom` 的条目在解析期即被拒（诊断点名该标注）；旧拼写 `lowering: family` 不被静默重解释 |
| `vpto_bridge_declarative_binding_diag.pto` | 改用 `-split-input-file`：四个场景此前共处一个 module，而声明式 pass 是 FuncOp 嵌套 pass，MLIR 异步 adaptor 在首个失败函数处 break 出该 worker 的分片，导致实际产出的诊断条数在 1~4 之间随机（属既有缺陷，非本次引入） |
| `vpto_bridge_declarative_wrapper_source.pto` | 内置白名单 matmul 走通用渲染器的迁移 litmus：wrapper_source includes/typedef/调用体与迁移前硬编码渲染输出一致 |
| `vpto_bridge_whitelist_render_schema_diag.pto` | 新 schema 边界诊断：缺 `call` / `tmpl_args` 无对应 attr 行 / `core` 非法值 / 声明式 wrapper 缺 `wrappers` 段 |
| `vpto_bridge_tadd_declarative_lowering.pto` | tadd 零 env 声明式降级：func_spec 三个 vec tile token + `bridge_call "pto_vpto_add"` |
| `vpto_bridge_tadd_wrapper_source.pto` | vec_elem wrapper 渲染：`__DAV_VEC__` 守卫 + role typedef + `pto::TADD` 调用体 |

模拟器侧 DEVICE=SIM，内置默认白名单，零 env 注入：

| 用例 | 内容 | 结果 |
|---|---|---|
| `fifo-tile-data-consume` | TPush→TPOP FIFO 通路，128 f32 全量比对 | compare passed |
| `cube-matmul-bridge` | 16×16×16 f16 矩阵乘（A=单位阵），mte 链路→tmatmul→mte 出 | compare passed（声明式重构后复跑通过） |

变体配置（slot_num/flag_base/容量变化）的端到端行为由 lit 侧
`spec_config_matrix` 配置矩阵覆盖，未单独设置模拟器用例。


## 附：变更面速览

| 层 | 文件 |
|---|---|
| 声明式 pass | `lib/PTO/Transforms/PTOLowerDeclarativeBridgeOps.cpp` |
| 家族 pass | `lib/PTO/Transforms/PTOLowerPipeFamilyOps.cpp`（仅 pipe） |
| 通用 pass | `lib/PTO/Transforms/VPTOBridgeLowering.cpp` |
| 白名单 | `VPTOBridgeWhitelist.{h,cpp}`（解析/校验/内置默认） |
| token 构建 | `VPTOBridgeTokens.{h,cpp}`，共享映射层 `PTOCppTokens.{h,cpp}` |
| spec 收集 | `include/PTO/Transforms/VPTOBridgeSpecCollector.h`（函数级 spec 收集器，声明式/家族共用） |
| wrapper 生成 | `VPTOBridgeWrapperGen.cpp` |
| 合入通道 | `tools/ptoas/ObjectEmission.cpp`（wrapper 编译 + bitcode 链接） |
| IR | `pto.bridge_call` / `pto.bridge_inttoptr` 内部 op |

