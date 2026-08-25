# VPTO C++ 接口桥接泛化——设计总结

**概述**：本分支把 VPTO 后端与 PTO-ISA C++ 模板硬件接口之间的桥接构建为
编译器内置的通用通道。核心交付：

- **白名单驱动的双通道路由**：机械映射型 op 缺省走声明式通道（白名单
  注册即接入，零 pass 代码）；真家族语义（storage 生命周期、地址重绑定）
  以 `lowering: custom` 显式进入专用通道（现仅 pipe 家族）；
- **通用声明式 wrapper 渲染器**：wrapper 的 includes、核守卫、entry 签名、
  typedef 与调用体全部由白名单声明驱动渲染，新增机械映射家族零 C++ 改动；
- **三个家族端到端验证**：pipe（FIFO 通路）、matmul（cube 矩阵乘）、
  tadd（vec 逐元素加，纯白名单接入）均在模拟器 DEVICE=SIM 下
  compare passed。

## 1. 背景与目标

VPTO 后端直接发射 LLVM IR，无法实例化 PTO-ISA 中以 C++ 模板实现的硬件接口
（TPush/TPop/TFree FIFO 协议、TMATMUL 等），需要一层桥接。本工作将桥接定位
为**所有 PTO-ISA C++ 接口公用的编译器内置通道**，设计目标：

- **参数由 IR 属性驱动**——任意合法配置组合均可从 op 属性流入，无硬编码；
- **路由声明化**——哪些 op 走桥接、映射到哪个 wrapper 入口，全部白名单声明；
- **wrapper 自动生成**——编译器按收集到的特化信息渲染 wrapper 源码并编译
  合入，全流程无外部脚本、无环境变量依赖；
- **扩展接口家族时通用层零改动**——机械映射型 op 零 pass 代码且零标注接入
  （声明式是缺省通道）；家族语义封装在专用 pass 与 wrapper 模板内。

## 2. 整体架构

三层分离，核心约束是**通用混编 pass 不因新增接口家族而改动**：

```text
[op 路由层]          声明式通道（条目缺省归属）：白名单驱动的机械降级
                     （abi 行 operand 位置绑定 → 规划地址、role → tile token）
                     家族通道（条目显式 `lowering: custom`）：真家族语义
                     （地址重绑定、storage 生命周期，现仅 pipe）
        ↓ 产出 pto.bridge_call（callee + operand/result，不携带家族类型）
[通用混编 pass 层]   VPTOBridgeLowering
        读白名单确认 entry 合法 → bridge_call 机械降级为 func.call + 外部声明，
        按白名单 abi 字段校验参数（i64 地址载体 / 不透明指针 storage），
        不感知任何家族语义
        ↓
[wrapper 生成与 bitcode 合入]
        VPTOBridgeWrapperGen：合并各函数 spec → 按条目通道渲染 wrapper C++ 源码
        ObjectEmission：wrapper 编译为 cube/vec 两路 bitcode 注入各自 fatobj
```

关键设计决策：

| 决策 | 理由 |
|---|---|
| 中间形态用显式内部 op `pto.bridge_call` | 可被 legalize 校验：白名单命中但未被降级的残留 op 直接报明确诊断，不静默回退 |
| YAML 白名单（不与 ODS 联动） | 桥接属工具链策略配置而非 IR 定义，`llvm::yaml` 在树无额外依赖 |
| 两档路由，`lowering` 缺省为声明式 | 仪式成本落在例外一侧：机械映射型家族零标注接入；漏标条目解析期即被拒绝并点名缺失字段，误用可自检 |
| 家族语义经 SSA result 完成 | 通用 pass 只做机械降级，TPipe 参数接口后续任何变化只改家族 pass 与 wrapper |
| 内置默认白名单编译进 ptoas | `ptoas --pto-backend=vpto` 零环境变量开箱即用；option/env 可覆盖，空路由白名单可退出桥接 |

## 3. 核心机制

### 3.1 白名单：最小接入形态

示例只展示**必填字段**——接入一个机械映射型 op 的最小注册内容（以
`pto.tmatmul` 为例）；可选字段与缺省规则见文末《附录：白名单 schema
速查》：

```yaml
wrappers:                        # 每个声明式 wrapper 一条
  - name: matmul                 # 必填：wrapper 名（生成单位与分派键）
    includes: [pto/npu/a5/TMatmul.hpp]   # 必填：wrapper 源码的 include 行
bridge_ops:
  - op: pto.tmatmul              # 必填：路由键，IR op 名，命中即桥接候选
    wrapper: matmul              # 必填：渲染进哪份 wrapper 源码
    abi:                         # 必填：operand 位置绑定 + 角色
      - {operand: 2, role: result_tile}
      - {operand: 0, role: left_tile}
      - {operand: 1, role: right_tile}
```

未写的字段全部走缺省：`entry`（本例 `pto_vpto_matmul`）、`call`
（`pto::TMATMUL`）、形参名（role 的 lowerCamelCase，如 `resultTile`）、
参数类型（i64）、核守卫（`__DAV_CUBE__`，从条目 tile 种类推导）均由约定
推导；推导不符接口约定时可显式覆盖（缺省规则与失败方式见文末附录
《可推导字段与失败点》表）。其余设计要点：

- **两档通道，缺省为声明式**：仅携带真家族语义的条目显式 `lowering: custom`
  退出声明式通道；
- **内置默认白名单**：pipe 4 条路由条目（均 `custom`）+ 1 条
  `op: internal` 辅助条目，matmul 与 vec_elem 家族均走缺省声明式通道；
- **结构化校验**：必填项缺失、拼写或取值非法均在解析期拒绝并点名条目与文件；
- **三级解析链**：pass `whitelist-path` option →
  `PTOAS_VPTO_BRIDGE_WHITELIST` 环境变量 → 内置默认白名单。

> `wrapper` 是**生成单位**与分派键：同名条目共享一份 C++ 翻译单元与一套
> typedef，每个 module 至多一个。纯声明式 wrapper 走**通用渲染器**（新增
> 机械映射家族零 C++ 改动）；含 `lowering: custom` 条目的 wrapper 走
> **专用渲染器**（现仅 pipe），缺失时诊断列出可用集合。

### 3.2 参数传递机制：IR 配置 → C++ 模板实参

职责是把 IR 事实（平铺属性、tile 类型）翻译为 PTO-ISA 的 C++ 模板拼写——
例如为带 `slot_size=1024, slot_num=8` 属性、方向 C2V 的 pipe init 产出
`pto::TPipe<0, pto::Direction::DIR_C2V, 1024, 8, 2, false>` 这类 **token**。
设计结果：

- **映射规则与 EmitC 同源**：IR 事实 → token 的映射统一实现在
  `PTOCppTokens`，两侧共用，lit 对拍用例保证同一配置下两路渲染逐字段一致；
- **桥接侧组装层** `VPTOBridgeTokens` 产出可直接替换进 wrapper 源码的
  token，写入函数级 spec；
- **三级流转**：收集（路由层 pass 从 op 属性/operand 类型提取配置 → token
  → 函数级 spec，并发安全）→ 合并（模块级单线程确定性合并，同键异值报
  冲突诊断）→ 渲染（必需槽位覆盖校验，未覆盖报明确诊断而非静默丢弃）。

### 3.3 wrapper 自动生成与 bitcode 合入

```text
VPTOBridgeWrapperGen（模块级）   按条目通道渲染 wrapper_source：
                                 声明式 wrapper 走通用渲染器（includes/core
                                 守卫 + role typedef + extern "C" 入口 +
                                 TASSIGN + call）；custom wrapper 走专用
                                 渲染器（现仅 pipe，producer/consumer 角色
                                 按 pipe 方向自动对调）
        ↓
ObjectEmission                   读 wrapper_source，cube/vec 两路各自编译
                                 bitcode（PTO-ISA 头文件多候选路径自动发现）
                                 → 合入 device 模块
```

全流程内置于 ptoas：`ptoas --pto-backend=vpto` 一条命令完成桥接降级、
wrapper 渲染、编译与合入，无外部脚本、无环境变量依赖。

### 3.4 使用方式：接入一个机械映射型 op

接入只需白名单注册（注册内容即 §3.1 示例），零 C++ 改动。注册完成后，
内核中的 `pto.tmatmul` 自动经声明式通道降级为
`pto.bridge_call "pto_vpto_matmul"`（entry 名、调用拼写、形参名与核守卫
均为缺省推导产物），通用渲染器产出如下 wrapper 源码并编译合入：

```cpp
// 渲染产物（模块属性 pto.vpto.bridge.wrapper_source，节选；
// typedef 从 IR tile 类型收集，entry 名、形参名与核守卫均缺省推导）
#include <pto/npu/a5/TMatmul.hpp>
using LeftTile   = pto::Tile<pto::TileType::Left, half, 16, 16, pto::BLayout::ColMajor, 16, 16, pto::SLayout::RowMajor, 512>;
using ResultTile = pto::Tile<pto::TileType::Acc, float, 16, 16, pto::BLayout::ColMajor, 16, 16, pto::SLayout::RowMajor, 1024>;
using RightTile  = pto::Tile<pto::TileType::Right, half, 16, 16, pto::BLayout::RowMajor, 16, 16, pto::SLayout::ColMajor, 512>;
#ifdef __DAV_CUBE__
extern "C" [aicore] void pto_vpto_matmul(uint64_t resultTileAddress, uint64_t leftTileAddress, uint64_t rightTileAddress)
{
  ResultTile resultTile; LeftTile leftTile; RightTile rightTile;
  pto::TASSIGN_IMPL(resultTile, resultTileAddress);
  pto::TASSIGN_IMPL(leftTile, leftTileAddress);
  pto::TASSIGN_IMPL(rightTile, rightTileAddress);
  pto::TMATMUL(resultTile, leftTile, rightTile);
}
#endif
```

宿主侧零配置编译：

```bash
ptoas --pto-arch a5 --pto-backend=vpto kernel.pto -o kernel.fatobj.o
```

验证链路：lit 钉住 spec 收集与 wrapper_source 两段产物；白名单拼写错误在
解析期即被拒绝；推导出的 `call` 不符接口约定时在 wrapper 源码编译期报错
并点名符号。携带真家族语义的 op 不在此列：需显式 `lowering: custom`、
必填 `entry`，由专用 pass 与专用渲染器承接（现仅 pipe 家族）。

## 4. 已验证的能力矩阵

### 4.1 pipe wrapper（TPipe/TPUSH/TPOP/TFREE）

| 维度 | 支持情况 |
|---|---|
| 配置参数 | `slot_size` / `slot_num` / `flag_base` / `nosplit` 任意组合，由 op 属性流入，无硬编码 |
| 方向 | C2V / V2C，producer/consumer 角色自动对调 |
| 数据类型 | 常规 dtype 含 f16（`half` 映射） |
| tile 形状 | 任意（含 NoneBox：省略 SLayout/SFractalSize 模板参数） |
| split | `split≠1` 支持（split=2 → `TileSplitAxis::TILE_LEFT_RIGHT`）；函数内共享单一 split token，异值报诊断 |
| 控制流 | 循环内 push/pop/消费/free 验证正确 |
| 生命周期 | init（storage 经内置 size 查询分配）→ push/pop → free；TPOP 地址重绑定经 SSA result |

### 4.2 matmul wrapper（`pto.tmatmul`）

| 项 | 支持情况 |
|---|---|
| ABI | 3×i64（dst/lhs/rhs），经白名单 abi 行位置绑定与校验 |
| dst 校验 | 必须来自带规划地址的 `alloc_tile`，否则报明确诊断 |
| 端到端 | 16×16×16 f16 矩阵乘（A=单位阵），mte 链路→tmatmul→mte 出，compare passed |

### 4.3 vec_elem wrapper（`pto.tadd`，纯白名单接入）

| 项 | 支持情况 |
|---|---|
| 接入方式 | 零 C++ 改动：仅白名单注册（wrappers 段声明 name 与 includes，条目仅 op/wrapper/abi(operand+role)）；调用拼写、形参名、entry 名与 vec 核守卫均按约定推导，调用体由通用渲染器渲染 |
| ABI | 3×i64（dst/src0/src1），无模板实参 |
| 渲染验证 | lit 覆盖 spec 收集（entry 推导 `pto_vpto_add`）与 wrapper 源码（`__DAV_VEC__` 守卫 + TADD 调用体） |
| 端到端 | 8×16 f32 逐元素加（模拟器 DEVICE=SIM），mte 进 UB → 桥接 TADD → mte 出，compare passed |

### 4.4 机制验证结论

- 接口面完全不同的两个声明式家族（cube MATMUL 与 vec TADD）均由同一个
  通用渲染器产出 wrapper：核守卫、有无模板实参的调用形态统一覆盖，通用层
  不感知家族差异；
- 机械映射型 op 的接入成本仅为白名单注册：无专用 pass、无生成器改动，
  连 `lowering` 标注都不需要写；
- 新增接口家族时通用混编 pass 与 ObjectEmission 合入通道保持不变，通用层
  与家族语义解耦的分层设计成立。

## 5. 测试验证

lit 侧桥接用例共 29 个，按维度分组（标**新增**者为本轮 schema 精简配套
新增）：

**路由与通道**：

| 用例 | 覆盖点 |
|---|---|
| `vpto_bridge_default_whitelist_lowering.pto` | 零配置开箱即用：无 option/env 时回退内置默认白名单完成桥接 |
| `vpto_bridge_declarative_unrouted_passthrough.pto` | 未路由 op 保持常规 tile-op 展开路径，custom 条目留给家族 pass，均不过度降级 |
| `vpto_bridge_whitelist_default_channel_diag.pto` | 漏标 `lowering: custom` 的条目解析期即被拒；旧拼写 `lowering: family` 不被静默重解释 |
| `vpto_bridge_declarative_binding_diag.pto` | 声明式绑定四类诊断（operand 越界、非 tile operand、带结果 op、非枚举 attr）逐场景独立验证 |
| `vpto_bridge_whitelist_residual_diag.pto` | 白名单命中但未被降级的残留 op 报诊断而非静默发射 |

**pipe 家族**：

| 用例 | 覆盖点 |
|---|---|
| `vpto_bridge_pipe_family_lowering.pto` | 家族 pass init/push/pop/free 降级与 storage_size 关联 |
| `vpto_bridge_pipe_family_skip_pipeless.pto` | 无 pipe 的函数不被家族 pass 触碰 |
| `vpto_bridge_pipe_loop_consume.pto` | 循环内 FIFO 消费：单 pop op 每迭代经 SSA result 重绑定当次 slot 地址 |
| `vpto_bridge_pipe_split_left_right.pto` | `split=2` → `TILE_LEFT_RIGHT`，函数共享单一 split token 渲染进全部入口 |
| `vpto_bridge_pipe_split_mismatch_diag.pto` | 函数内 split 异值报诊断而非静默渲染错误轴向 |
| `vpto_bridge_pop_rebind_diag.pto` | 同一 declared tile 的两次 TPOP 报明确诊断（顺序重绑定边界） |
| `vpto_bridge_wrapper_source_c2v.pto` / `_v2c.pto` | 两方向 wrapper 渲染：角色对调、守卫归属与 token 均取自 IR 配置 |

**matmul / tadd 声明式家族**：

| 用例 | 覆盖点 |
|---|---|
| `vpto_bridge_matmul_family_lowering.pto` / `_variants_lowering.pto` | tmatmul 声明式降级：role token 收集、typedef 渲染与入口体 |
| `vpto_bridge_matmul_unsupported_diag.pto` | 无规划地址的 operand tile 拒绝桥接，诊断点名 abi 绑定 |
| `vpto_bridge_matmul_spec_conflict_diag.pto` | 同函数两条 matmul 的 role token 异值报冲突诊断 |
| `vpto_bridge_declarative_wrapper_source.pto` | 内置白名单 matmul 走通用渲染器：wrapper_source 逐行钉住 |
| `vpto_bridge_tadd_declarative_lowering.pto` | tadd 零 env 声明式降级：func_spec 三个 vec tile token + `bridge_call "pto_vpto_add"` |
| `vpto_bridge_tadd_wrapper_source.pto` | vec_elem wrapper 渲染：调用拼写、形参名与核守卫均缺省推导，逐行钉住 |

**schema 与 spec 校验**：

| 用例 | 覆盖点 |
|---|---|
| `vpto_bridge_whitelist_minimal_defaults.pto` **新增** | 全缺省推导路径：最小 schema 下 entry 名、调用拼写、形参名与核守卫全部推导成功 |
| `vpto_bridge_whitelist_render_schema_diag.pto` | 声明式 schema 边界：`tmpl_args` 无对应 attr 行 / `core` 非法值 / 缺 `wrappers` 段声明 |
| `vpto_bridge_whitelist_matmul_tmpl_diag.pto` | 声明式条目 tmpl_map 的 tile 行解析期即被拒，只收 `attr` 行 |
| `vpto_bridge_whitelist_tmpl_map_diag.pto` | tmpl_map 缺键与未知 source 诊断 |
| `vpto_bridge_whitelist_tmpl_coverage_diag.pto` | 未覆盖的 tmpl_map 声明报诊断而非静默丢弃 |
| `vpto_bridge_whitelist_unknown_wrapper_diag.pto` | 无专用渲染器的 custom wrapper 报诊断并列出可用集合 |
| `vpto_bridge_spec_config_matrix.pto` | 非缺省 pipe 配置（slot_size/slot_num/flag_base/nosplit/tile 形状）逐字流入 spec |
| `vpto_bridge_spec_conflict_diag.pto` | 跨函数 spec 合并：同配置去重，异配置报冲突诊断 |
| `vpto_bridge_emitc_token_parity.pto` | token 一致性守卫：同一配置下桥接与 EmitC 两路渲染逐字段一致 |

模拟器侧 DEVICE=SIM，内置默认白名单，零 env 注入：

| 用例 | 内容 | 结果 |
|---|---|---|
| `fifo-tile-data-consume` | TPush→TPOP FIFO 通路，128 f32 全量比对 | compare passed |
| `cube-matmul-bridge` | 16×16×16 f16 矩阵乘（A=单位阵），mte 链路→tmatmul→mte 出 | compare passed |
| `vec-add-bridge` **新增** | 8×16 f32 逐元素加，纯白名单接入的 tadd 经桥接 wrapper（`pto_vpto_add`）计算，mte 进 UB→TADD→mte 出 | compare passed |

变体配置（slot_num/flag_base/容量变化）的端到端行为由 lit 侧
`spec_config_matrix` 配置矩阵覆盖。

## 附录 A：变更面速览

| 层 | 文件 |
|---|---|
| 声明式 pass | `lib/PTO/Transforms/PTOLowerDeclarativeBridgeOps.cpp` |
| 家族 pass | `lib/PTO/Transforms/PTOLowerPipeFamilyOps.cpp`（仅 pipe） |
| 通用 pass | `lib/PTO/Transforms/VPTOBridgeLowering.cpp` |
| 白名单 | `VPTOBridgeWhitelist.{h,cpp}`（解析/校验/内置默认） |
| token 构建 | `VPTOBridgeTokens.{h,cpp}`，共享映射层 `PTOCppTokens.{h,cpp}` |
| spec 收集 | `include/PTO/Transforms/VPTOBridgeSpecCollector.h` |
| wrapper 生成 | `VPTOBridgeWrapperGen.cpp` |
| 合入通道 | `tools/ptoas/ObjectEmission.cpp`（wrapper 编译 + bitcode 链接） |
| IR | `pto.bridge_call` / `pto.bridge_inttoptr` 内部 op |

## 附录 B：白名单 schema 速查

**结论：当前 schema 已是最精简形态**——所有可推导字段均已缺省化，剩余
必填字段全部是 IR 中不存在、编译器无法静态分析的地基信息，不可再删。

**可推导字段与失败点**（均可显式覆盖）：

| 字段 | 缺省规则 | 推导错误时的失败方式 |
|---|---|---|
| `lowering` | 缺省 `declarative` | 漏标 `custom` 的条目缺 operand/role 绑定，解析期拒绝并点名缺失字段 |
| `entry` | 从 op 名推导：`pto.tmatmul` → `pto_vpto_matmul` | 设备链接报缺失符号 |
| `call` | 从 op 名推导：`pto.tmatmul` → `pto::TMATMUL` | wrapper 源码编译期（ObjectEmission）报错并点名符号 |
| `abi.arg` | role 的 lowerCamelCase：`left_tile` → `leftTile` | 纯命名，无对错 |
| `abi.type` | 缺省 `i64`（tile 地址载体） | lowering 参数类型校验不符 |
| `core` | 从路由 tile 地址空间推导：VEC → `vec`，cube 家族 → `cube` | 推错 → 设备链接缺符号；无 tile 可推导 → 渲染期响亮报错要求显式声明 |

**必填字段**（地基信息，均不可省）：

| 字段 | 为什么不能省 |
|---|---|
| `op` | 路由键本身 |
| `wrapper` / `wrappers.name` | N 个条目共享一份翻译单元的分组身份，与头文件名无对应关系 |
| `wrappers.includes` | SDK 头文件布局是外部环境信息，IR 中不存在，编译器不扫描文件系统 |
| `abi.operand` | MLIR 无通用 operand 名反射，位置绑定是唯一事实；结果位置跨接口不固定 |
| `abi.role` | 哪个 operand 是 result/left/right 属 op 语义，无法从 tile 类型区分 |

**可选进阶字段**：`tmpl_args` / `tmpl_map`（枚举模板实参映射，如 matmul 的
`acc_phase` → `AccPhase::Final`；声明式条目只收 `source: attr` 行）、
`storage_size_entry`（stateful 条目关联 size 查询入口，pipe 家族）。逐字段
校验规则详见 `include/PTO/Transforms/VPTOBridgeWhitelist.h` 注释。
