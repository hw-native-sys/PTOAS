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

本工作的目标是把桥接从"TPipe 专用 PoC"泛化为**所有 PTO-ISA C++ 接口公用的
编译器内置通道**：

- **参数由 IR 属性驱动**——配置不再写死，任意合法组合均可从 op 属性流入；
- **路由由白名单驱动**——哪些 op 走桥接、映射到哪个 wrapper 入口，全部声明化；
- **wrapper 自动生成**——编译器按收集到的特化信息渲染 wrapper 源码并编译合入，
  全流程无外部脚本、无环境变量依赖；
- **新增接口家族时通用层零改动**——机械映射型 op 经白名单声明直接走
  声明式通道；仅含真家族语义（地址重绑定、storage 生命周期）的复杂家族
  需要家族 pass，差异封装在白名单与 wrapper 模板内。

## 2. 整体架构

三层分离，核心约束是**通用混编 pass 不因新增接口家族而改动**：

```text
[op 路由层]                  PTOLowerDeclarativeBridgeOps（声明式） / PTOLowerPipeFamilyOps（家族）
  声明式通道：仅由白名单驱动的机械降级（abi 行的 operand 位置绑定 →
  规划地址、role → tile token、attr 行 → 枚举属性 token），不感知任何家族语义；
  家族通道仅保留给有真家族语义的 op（如 TPOP 地址重绑定经 bridge_call
  的 SSA result 在本层内完成、storage 生命周期）
        ↓ 产出 pto.bridge_call（callee + operand/result，不携带家族类型）
[通用混编 pass 层]        VPTOBridgeLowering
  读白名单确认 entry 合法 → bridge_call 机械降级为 func.call + 外部声明
  按白名单 abi 字段校验参数（i64 地址载体 / 不透明指针 storage）
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
| token 构建参照 EmitC 理念、桥接侧独立实现（`VPTOBridgeTokens`） | 两后端获取 tile 类型信息的时点与消费形态不同，无法直接共享；用 lit 比对测试防漂移（详见 §3.2） |
| 家族语义经 SSA result 完成 | 通用 pass 只做机械降级，TPipe 参数接口后续任何变化只改家族 pass 与 wrapper |
| 声明式/家族两档分类（白名单 `lowering` 字段） | 分类轴是变换复杂度而非家族：机械映射型 op（如 matmul 全家族）零 pass 代码接入，复杂家族（pipe）保留专用 pass |
| 内置默认白名单编译进 ptoas | `ptoas --pto-backend=vpto` 零环境变量开箱即用；特殊场景可用 option/env 覆盖，或以空路由白名单退出桥接 |

## 3. 核心机制设计

### 3.1 白名单：路由与类型映射声明

白名单同时承担**路由**与**类型映射声明**两个职责：

```yaml
bridge_ops:
  - op: pto.tpush                 # 路由键：IR op 名，命中即桥接候选
    family: pipe                  # 家族（决定 wrapper 模板；lowering=family 时亦决定家族 pass）
    entry: pto_vpto_pipe_push     # wrapper 入口名（通用 pass 的 callee）
    abi:                          # 调用侧校验 + decl 生成
      - {arg: storage, type: ptr}
      - {arg: tile,    type: i64}
    tmpl_map:                     # 模板实参来源声明（见 3.2）
      - {source: pipe.init, field: slot_size, target: Pipe.slotSize}
  - op: pto.tmatmul
    family: matmul
    lowering: declarative         # 声明式通道（缺省 family）
    entry: pto_vpto_matmul
    abi:                          # operand 位置绑定 + 角色
      - {operand: 2, arg: dst, type: i64, role: result_tile}
      - {operand: 0, arg: lhs, type: i64, role: left_tile}
      - {operand: 1, arg: rhs, type: i64, role: right_tile}
    tmpl_map:
      - {source: left_tile, field: tile, target: LeftTile}
      - source: attr              # 枚举属性型 source
        field: acc_phase
        target: AccPhase
        enum_type: pto::AccPhase
        omit_value: Unspecified
```

- init 条目另有 `storage_size_entry` 字段显式关联 size 查询入口；
  `op: internal` 为 wrapper 内部辅助条目标记，不参与路由；
- 声明式条目的结构化校验：abi 行必须带 `operand`/`arg`/`role`，operand
  索引不重复；`tmpl_map` 的 tile source 必须出现在 abi role 集，`attr`
  source 必须声明 `enum_type`；
- 三级解析链：pass `whitelist-path` option → `PTOAS_VPTO_BRIDGE_WHITELIST`
  环境变量 → 内置默认白名单（pipe + matmul 全条目）；
- 解析期校验：空字段、重复 `op` 名、`storage_size_entry` 悬空引用、
  `tmpl_map` 缺键与未知 source 均报诊断（含文件路径与支持集）。

### 3.2 参数传递机制：参照 EmitC 理念，桥接侧定制实现（映射层部分复用）

桥接的核心难题是参数形态的鸿沟：IR 侧只有整数属性（`slot_size=1024`）与
tile 类型，C++ 侧需要 `TPipe<0, C2V, 1024, 8, 2, false>` 这类模板实参。
这一机制的设计参照了 EmitC 后端已有的"IR op 属性 → C++ 模板 token"构建
理念，但**并非照搬其实现，而是针对桥接场景独立定制**：

**为什么参照 EmitC**：EmitC 路径（`PTOToEmitC.cpp`）已沉淀了一套完整的
"IR 属性 → 方向/pipe/tile/split token"构建规则，且经生产验证，是桥接侧
最直接的正确性参照。

**为什么不整体复用**：两条路径的约束不同，强行共享组装层会同时绑架两个后端——

| 差异点 | EmitC 路径 | 桥接路径 |
|---|---|---|
| tile 类型信息来源 | 依赖已转换的 OpaqueType 携带 tile 布局 | VPTO 管线内桥接时点直接取 `TileBufType` 的 config 属性 |
| token 消费形态 | 拼进输出文本 `TPUSH<pipeTok, tileTok, ...>(...)` | 喂给 wrapper 生成器：按 `tmpl_map` 声明驱动 typedef 渲染，并配合 ABI 签名校验 |
| 限定形式 | 沿用输出上下文的命名约定 | wrapper 是独立编译单元，token 统一输出 `pto::` 全限定拼写；NoneBox tile 省略尾部默认实参，与 wrapper 特化形态对齐 |

**定制实现 + 部分复用**：桥接侧自建 `VPTOBridgeTokens`（`buildBridgePipeToken` /
`buildBridgeTileToken` / `buildBridgeTileSplitToken` /
`buildBridgeElementTypeToken`），产出可直接替换进 wrapper 源码的全限定
C++ 拼写；组装骨架各自保留，但**纯映射层（IR 事实 → C++ 拼写）已
抽取为共享的 `PTOCppTokens`，两侧改为薄包装调用**（见下）。

**漂移防护**：两侧共享纯映射层 `PTOCppTokens`（IR 事实 → C++ 拼写，
限定符参数区分 `pto::` 全限定与空限定），组装逻辑（NoneBox 省略、
`_GM` 变体等设计性差异）留在各后端；专项 lit（`vpto_bridge_emitc_token_parity`）
比对双路径输出，组装级差异以外的任何不一致即失败。

**参数完整流转链路**（收集 → 合并 → 渲染）：

- **收集**：声明式/家族 pass 从 op 属性/operand 类型抽取配置，经
  `VPTOBridgeTokens` 解析为 token，写入函数级
  `pto.vpto.bridge.func_spec` 属性（func 级 nested pass，并发安全）；
- **合并**：模块级 wrapper 生成 pass 单线程确定性合并各函数 spec，
  同 key 异值报 "conflicting … bridge specialization" 诊断；
- **渲染**：wrapper 的 typedef 段由白名单已用条目的 `tmpl_map` 行按声明
  顺序驱动（source → spec key → token，target first-wins 去重）；渲染前
  按入口条件做必需槽位兜底校验（matmul：Left/Right/ResultTile 必有，
  AccIn/Bias/AScale/BScale 按入口条件；pipe：Pipe/ProducerTile/
  ConsumerTile），未覆盖报明确诊断而非静默丢弃。

### 3.3 wrapper 自动生成与 bitcode 合入

```text
声明式/家族 pass（func 级，可并发）  收集 spec → 函数属性
        ↓
mergeFuncSpecsIntoModule（模块级，单线程）   确定性合并 + 冲突诊断
        ↓
VPTOBridgeWrapperGen             tmpl_map 驱动渲染 wrapper_source：
                                 TPipe/Tile typedef + __DAV_CUBE__/__DAV_VEC__
                                 守卫的 init/push/pop/free/size 与 matmul 各入口；
                                 producer/consumer 角色按 pipe 方向自动对调
        ↓
ObjectEmission                   读 wrapper_source，cube/vec 两路各自编译
                                 bitcode（PTO-ISA 头文件经多候选路径自动发现）
                                 → linkDeviceLLVMBitcode 合入 device 模块
```

全流程内置于 ptoas：`ptoas --pto-backend=vpto` 一条命令完成桥接降级、
wrapper 渲染、编译与合入，无外部脚本、无环境变量依赖。

## 4. 已验证的能力矩阵

### 4.1 pipe 家族（TPipe/TPUSH/TPOP/TFREE）

| 维度 | 支持情况 |
|---|---|
| 配置参数 | `slot_size` / `slot_num` / `flag_base` / `nosplit` 任意组合，由 op 属性流入，无硬编码校验 |
| 方向 | C2V / V2C，producer/consumer 角色自动对调 |
| 数据类型 | 常规 dtype 含 f16（`half` 映射） |
| tile 形状 | 任意（含 NoneBox：省略 SLayout/SFractalSize 模板参数） |
| split | `split≠1` 支持（split=2 → `TileSplitAxis::TILE_LEFT_RIGHT`）；函数内共享单一 split token，异值报诊断 |
| 控制流 | 循环内 push/pop/消费/free 验证正确（单 pop op 每迭代绑定当次 fifo slot） |
| 生命周期 | init（storage 经内置 size 查询分配）→ push/pop → free；TPOP 地址重绑定经 SSA result |

### 4.2 matmul 家族（CUBE 侧，6 个入口）

| IR op / 变体 | wrapper 入口 | ABI | 备注 |
|---|---|---|---|
| `pto.tmatmul` | `TMATMUL<Phase>` | 3×i64（dst/lhs/rhs） | dst 必须来自带规划地址的 `alloc_tile` |
| `pto.tmatmul.acc` | `TMATMUL<Phase>`（accIn 重载） | 4×i64 | `accPhase` 枚举属性渲染为模板实参 |
| bias | `TMATMUL_BIAS<Phase>` | 4×i64 | |
| mx | `TMATMUL_MX<Phase>` | 5×i64（含 aScale/bScale） | |
| mx.acc | `TMATMUL_MX<Phase>` | 6×i64 | |
| mx.bias | `TMATMUL_MX` | 6×i64 | IR op 无 accPhase，不渲染 Phase 实参 |

配套 token 构建覆盖 float8/float4 标量（e4m3/e5m2/e8m0/hifloat8/e1m2x2/e2m1x2）。

### 4.3 分层验证结论

MATMUL 作为接口面与 pipe 完全不同的第二家族接入时，**通用混编 pass 与
ObjectEmission 通道零改动**，证实分层假设成立；随后 matmul 全家族进一步
迁移到声明式通道（`PTOLowerDeclarativeBridgeOps`），家族 pass 被删除，
wrapper 渲染与 spec key 零改动——确认分类轴应为变换复杂度而非家族。

## 5. 测试验证

### 5.1 lit 回归

全量回归 1840 用例、1839 过（1 个 unsupported，零失败）。桥接相关
共 22 个用例 + 9 个 YAML fixture，均为验证桥接功能的核心用例，
按职责分组如下：

| 类别 | 核心用例 | 验证点 |
|---|---|---|
| 降级正向 | `pipe_family_lowering`、`matmul_family_lowering`、`matmul_variants_lowering`（4 变体双前缀）、`default_whitelist_lowering`、`pipe_split_left_right`、`pipe_loop_consume` | 声明式/家族 pass → `bridge_call` → `func.call` + decl 的完整降级形态；内置默认白名单零 env 生效；split≠1 与循环消费形态；matmul 用例 CHECK 零改动验证声明式通道逐字节等价 |
| 声明式绑定 | `declarative_binding_diag`、`declarative_unrouted_passthrough` | operand 越界/非 tile_buf/带 result/非枚举属性四场景诊断；未路由与 family 条目放行（保留 mad 展开回退） |
| wrapper 源码钉住 | `wrapper_source_c2v` / `wrapper_source_v2c` | FileCheck 钉住生成的 wrapper C++ 源码（typedef 与入口体） |
| 配置矩阵 | `spec_config_matrix` | slot_size/slot_num/flag_base/nosplit 变体组合 + f16 + 多形状的 spec 收集与渲染 |
| EmitC 漂移防护 | `emitc_token_parity` | 同一模块走 EmitC 与桥接双路径，模板 token 逐项一致（§3.2） |
| 诊断 | `whitelist_residual_diag`、`whitelist_tmpl_map_diag`、`whitelist_matmul_tmpl_diag`、`whitelist_tmpl_coverage_diag`、`spec_conflict_diag`、`matmul_spec_conflict_diag`、`pop_rebind_diag`、`pipe_split_mismatch_diag`、`matmul_unsupported_diag`、`pipe_family_skip_pipeless` | 白名单残留、tmpl_map 缺键/未知 source、声明式 role 集校验（未覆盖声明在解析期被拒）、跨函数与 matmul spec 冲突、同 tile 多 pop、split 异值、无规划地址 operand、无 pipe 函数跳过 |

### 5.2 模拟器端到端

DEVICE=SIM，内置默认白名单，零 env 注入：

| 用例 | 内容 | 结果 |
|---|---|---|
| `fifo-tile-data-consume` | TPush→TPOP FIFO 通路，128 f32 全量比对 | compare passed |
| `cube-matmul-bridge` | 16×16×16 f16 矩阵乘（A=单位阵），mte 链路→tmatmul→mte 出 | compare passed |

变体配置（slot_num/flag_base/容量变化）的端到端行为已由 lit 侧
`spec_config_matrix` 配置矩阵覆盖，不再单独保留模拟器用例。


## 附：变更面速览

| 层 | 文件 |
|---|---|
| 声明式 pass | `lib/PTO/Transforms/PTOLowerDeclarativeBridgeOps.cpp` |
| 家族 pass | `lib/PTO/Transforms/PTOLowerPipeFamilyOps.cpp`（仅 pipe） |
| 通用 pass | `lib/PTO/Transforms/VPTOBridgeLowering.cpp` |
| 白名单 | `VPTOBridgeWhitelist.{h,cpp}`（解析/校验/内置默认） |
| token 构建 | `VPTOBridgeTokens.{h,cpp}`，共享映射层 `PTOCppTokens.{h,cpp}` |
| wrapper 生成 | `VPTOBridgeWrapperGen.cpp` |
| 合入通道 | `tools/ptoas/ObjectEmission.cpp`（wrapper 编译 + bitcode 链接） |
| IR | `pto.bridge_call` / `pto.bridge_inttoptr` 内部 op |

