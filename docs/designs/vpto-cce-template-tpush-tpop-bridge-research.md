<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# VPTO 通过 C Wrapper 复用 PTO-ISA TPush/TPop 模板

## 1. 文档目的与范围

本文记录 VPTO 后端通过 C ABI wrapper 调用 Bisheng 实例化的 PTO-ISA C++ 模板，以支持 TPush、TPop 和 TFree 相关操作的方案、PoC 结果和后续工程化计划。

第一阶段范围限定为 A5 本地 C2V tile-entry 路径：

- Cube producer：`loc=acc`、`f32`、`16x16`
- Vector consumer：`loc=vec`、`f32`、`8x16`
- `split = 1`，即 `TileSplitAxis::TILE_UP_DOWN`
- `InitializeL2LPipeOp`、`TPushOp`、`TPopOp`、`TFreeOp`
- 不包含 global-entry、A2/A3、V2C、双向 pipe、量化或 odd split

当前分支已在公开 PTO IR 输入上跑通固定 specialization 的完整链路：前端 pipe lowering、CANN900 VPTO LLVM emitter、PTO-ISA 模板实例化、CANN `llvm-link`、fatobj 生成和 CA simulator。`TPop` 返回地址已经接入 Tile SSA 使用链，并通过 FIFO Tile 数值写回验证；通用 specialization 生成仍未完成。

主要参考：

- `docs/designs/ptoas-tpush-tpop-design.md`
- `pto-isa/include/pto/npu/a5/TPush.hpp`
- `pto-isa/include/pto/npu/a5/TPop.hpp`
- `pto-isa/include/pto/npu/a5/TFree.hpp`

## 2. 当前结论与完成度

VPTO 可以复用 PTO-ISA 现有的 A5 `TPipe`、`TPUSH`、`TPOP` 和 `TFREE` 模板。可行的核心路径是将 CCE wrapper 编译为 device bitcode，再与 VPTO LLVM module 合并，而不是分别生成 device object 后进行 relocatable object 链接。

当前完成度如下：

| 层次 | 状态 | 说明 |
|---|---|---|
| 前端 pipe op 解析及 internal lowering | 已有能力 | 继续复用现有 pipe lowering |
| internal pipe op 到 wrapper call | 已完成固定配置 | 已接入 CANN900 VPTO emitter |
| PTO-ISA 模板实例化 | 已验证 | Cube/Vector 分目标编译成功 |
| wrapper 与 VPTO bitcode 合并 | 已自动化测试流程 | ObjectEmission 调用 CANN LLVM 15 `llvm-link` |
| device object 生成 | 已验证 | 使用 PTOAS 现有 Bisheng 编译参数 |
| ObjectEmission 链接集成 | 已完成 PoC | 测试脚本编译 wrapper bitcode，通过环境变量交给 ObjectEmission |
| TPop 返回地址参与后续 Tile 运算 | 已完成固定路径 | `declare_tile -> TPop -> tile_buf_addr -> MTE3` 已在 CA simulator 验证 |
| 通用 specialization 生成 | 未完成 | 当前固定为第一阶段配置 |
| FIFO 轮转及设备运行 | 已验证 | CA simulator 完成 12 轮，超过 `slot_num=8` |

当前已验证路径为：

```text
PTO frontend pipe ops
  -> frontend-to-internal pipe lowering
  -> 固定配置的 wrapper ABI call lowering
  -> 测试脚本编译 Cube/Vector wrapper bitcode
  -> ObjectEmission 调用 CANN llvm-link 合入对应 VPTO LLVM module
  -> Bisheng 编译 device object、fatobj
  -> CA simulator 运行并比较协议完成标记或 FIFO Tile 数值结果
```

目标工程化路径为：

```text
PTO frontend pipe ops
  -> frontend-to-internal pipe lowering
  -> 收集实际使用的 pipe 与 Tile specialization
  -> 生成 target-specific C wrapper
  -> Bisheng 编译 wrapper device bitcode
  -> ObjectEmission 自动合入对应 VPTO LLVM module
  -> Bisheng 编译 device object
```

该方案有两个不可省略的语义约束：

1. `TPipe` 是有状态对象，必须从 pipe 初始化一直存活到 subkernel 退出，不能在每次 push、pop 或 free 调用中临时构造。
2. 本地 tile-entry 的 `TPOP` 通过 `TASSIGN_IMPL` 将 consumer Tile 重绑定到当前 FIFO slot 地址。wrapper 必须返回该地址，并将其显式注入 VPTO Tile 数据流。

## 3. 第一阶段通信语义

第一阶段建立一条 A5 本地 Cube-to-Vector FIFO：

```text
Cube Acc 16x16xf32
  -> TPUSH<TILE_UP_DOWN>
  -> Acc-to-Vec fixpipe 分片
  -> AIV0/AIV1 各获得 Vec 8x16xf32
  -> TPOP 等待并绑定当前 FIFO slot
  -> Vector 使用 Tile
  -> TFREE 释放 consumer slot
```

对应配置为：

- `dir_mask = 1`：C2V
- `slot_size = 1024`
- 单向 pipe 的 `slot_num = 8`
- consumer buffer 由 Vector 侧 local buffer 提供
- Cube 侧使用同一个 local buffer 基址
- `split = 1`：沿行方向将 `16x16` producer Tile 分成两个 `8x16` consumer Tile

其他配置必须给出明确的 unsupported 诊断，不能静默生成近似实现。

## 4. VPTO 接入点

VPTO 不需要为 `pto.tpush_to_aiv`、`pto.tpop_from_aic` 和 `pto.tfree_from_aic` 重复实现一套前端语义。现有流程已经将其统一为 internal pipe IR：

```text
Cube:
  pto.initialize_l2l_pipe
  pto.tpush

Vector:
  pto.initialize_l2l_pipe
  pto.declare_tile
  pto.tpop
  pto.tfree
```

以 internal ops 为接入点可以继续复用：

- frontend pipe id 匹配和默认 id 分配
- C2V/V2C 方向解析
- local buffer 的 peer 配对和地址规划
- peer-aware flag base 分配
- split、Tile 类型和 pipe contract verifier
- scheduling、MemoryEffects 和 `OpPipeInterface` 信息

第一阶段需要由 VPTO emitter 处理：

- `InitializeL2LPipeOp`
- `TPushOp`
- `TPopOp`
- `TFreeOp`

internal `TAlloc`、`TPush`、`TPop` 和 `TFree` 不能被送入 TileLib 模板候选查询；它们应由专用 pipe lowering 处理。

## 5. PTO-ISA 模板行为

### 5.1 `TPipe` specialization 与持久状态

第一阶段对应的类型概念上为：

```cpp
using Pipe = pto::TPipe<
    FlagBase,
    pto::Direction::DIR_C2V,
    1024,  // SlotSize
    8,     // SlotNum
    2,     // LocalSlotNum
    false  // IsNoSplit
>;
```

`TPipe` 内部保存 FIFO 地址、producer/consumer `tileIndex`、entry offset 和同步开关，并通过 `shouldWaitFree`、`shouldNotifyFree` 实现基于 `SlotNum` 和 `SyncPeriod` 的稀疏同步。构造函数初始化 FIFO credits，析构函数执行收尾等待，因此对象生命周期本身是通信协议的一部分。

正式实现必须从 internal pipe init op 和匹配的 Tile 类型推导模板参数，不能固定写死。

### 5.2 `TPUSH`

概念调用为：

```cpp
pto::TPUSH<Pipe, ProducerTile, pto::TileSplitAxis::TILE_UP_DOWN>(
    pipe, accTile);
```

模板完成 producer slot 等待、FIFO slot 地址计算、consumer Tile 构造、Acc-to-Vec fixpipe 分片、producer index 更新，以及对两个 Vector subblock 的 ready 通知。

### 5.3 `TPOP`

概念调用为：

```cpp
pto::TPOP<Pipe, ConsumerTile, pto::TileSplitAxis::TILE_UP_DOWN>(
    pipe, fifoTile);
```

本地 C2V tile-entry 路径会等待 producer ready flag，计算当前 consumer slot 地址，通过 `TASSIGN_IMPL` 重绑定 `fifoTile`，然后更新 consumer index。该路径通常不复制数据，后续 VPTO 指令必须直接使用重绑定后的地址。

### 5.4 `TFREE`

概念调用为：

```cpp
pto::TFREE<Pipe, pto::TileSplitAxis::TILE_UP_DOWN>(pipe);
```

模板根据 consumer index 和 `shouldNotifyFree` 决定是否通知 Cube producer。它必须与此前的 `TPOP` 使用同一个 `TPipe` context。

## 6. Wrapper ABI 与生命周期

### 6.1 目标工程化 ABI

每个实际使用的 pipe specialization 应生成独立符号：

```cpp
extern "C" [aicore] uint64_t pto_pipe_ctx_size_<hash>();
extern "C" [aicore] uint64_t pto_pipe_ctx_align_<hash>();
extern "C" [aicore] void pto_pipe_init_<hash>(
    void *storage,
    __gm__ void *gmSlotBuffer,
    uint32_t c2vConsumerBuffer,
    uint32_t v2cConsumerBuffer);
extern "C" [aicore] void pto_pipe_finish_<hash>(void *storage);

extern "C" [aicore] void pto_pipe_push_<hash>(
    void *storage,
    uint64_t producerTileAddress,
    int32_t validRows,
    int32_t validCols);
extern "C" [aicore] uint64_t pto_pipe_pop_<hash>(
    void *storage,
    int32_t validRows,
    int32_t validCols);
extern "C" [aicore] void pto_pipe_free_<hash>(void *storage);
```

`init` 在 VPTO 提供的存储中构造 `TPipe`，`finish` 显式调用析构函数。context 必须位于实际 Cube/Vector subkernel 的栈帧中。所有调用均具有内存和同步副作用，不得标记为 `readnone`、`readonly` 或允许跨调用重排的纯函数。

第一版使用 `uint64_t` 传递 Tile 地址，以免在跨前端 ABI 中暴露 CCE 专用 address-space pointer。后续应根据 Bisheng bitcode 的类型表现决定是否改用明确的地址空间指针。

### 6.2 当前 PoC ABI

当前 wrapper 使用固定的 context size、init、finish、push、pop 和 free 符号。这些符号只服务于 A5 C2V 的固定 specialization，不构成公开接口。迁移到目标 ABI 仍需实现配置收集、稳定 hash、alignment、多 pipe 和多 specialization 支持。

## 7. TPop 地址传播

`TPopOp` 当前通过传入的 declared Tile 表达地址重绑定，而 LLVM IR 使用 SSA。VPTO 必须显式表达地址变化。

### 7.1 当前实现

wrapper 返回 `i64` 类型的 FIFO slot 地址。CANN900 emitter 记录 declared Tile 与 pop 地址的对应关系，并在 lowering `tile_buf_addr` 时将该地址转换为目标地址空间指针。`FoldTileBufIntrinsics` 保留由 `TPop` 重绑定的 `declare_tile` 地址查询，避免在进入 emitter 前把动态地址错误折叠为分配地址或占位值。

`test/vpto/cases/kernels/fifo-tile-data-consume/` 已验证该地址进入后续 MTE3 写回路径：Cube 计算 `16x16` 矩阵乘结果，经 `TPush<TILE_UP_DOWN>` 写入 FIFO；AIV0 从 `TPop` 返回的 `8x16` Vec Tile 地址写回 GM，host 对全部 128 个 `f32` 元素与 golden 比较。

### 7.2 推荐正式方案

在进入 VPTO LLVM emitter 前，将隐式重绑定规范化为显式 SSA，例如：

```mlir
%addr = pto.vpto_pipe_pop_addr ...
%tile = pto.bind_tile %addr ...
```

也可以引入返回 Tile/address 的 VPTO-only internal op。显式 SSA 能自然覆盖 CFG merge、loop、block argument、重复 pop 和 view alias，优于在 emitter 内维护隐式的 Tile 地址映射。

## 8. Wrapper 生成与 bitcode 合并

模板 specialization 至少依赖：

- flag base 和 direction
- slot size、slot number、local slot number
- `IsNoSplit`
- producer/consumer Tile location、dtype、shape、valid shape 和 layout
- split axis
- fixpipe 或量化配置

PTOAS 应只收集当前 module 实际使用的配置并生成临时 wrapper translation unit。符号名使用配置的稳定 hash，避免多个 pipe 或 specialization 冲突。

Cube 和 Vector wrapper 必须按目标裁剪并分别编译：

```text
dav-c310-cube:
  init/finish + TPUSH + Acc/fixpipe specialization

dav-c310-vec:
  init/finish + TPOP + TFREE + Vec specialization
```

推荐的对象生成流程为：

```text
VPTO LLVM module
  + Bisheng 编译的 target-specific wrapper bitcode
  -> 使用同版本 llvm-link 合并
  -> Bisheng 编译单个 device object
```

ObjectEmission 最终应提供通用的 CCE device wrapper/library 机制，而不是增加 TPush/TPop 或 Print 专用链接分支，以便其他 PTO-ISA C++ 模板复用同一链路。

## 9. 同步与调度边界

PTO-ISA wrapper 负责 FIFO 内部协议，包括 slot credit 等待、fixpipe、producer ready、consumer wait、consumer free 和 split 模式下的双 AIV flag。

VPTO 不应再次展开同一套 FIFO flag 协议，但 wrapper 外部仍需保持以下依赖：

```text
producer 计算完成 -> TPUSH 读取 Acc Tile
TPOP 完成 -> Vector op 读取 FIFO Tile
所有 FIFO Tile 使用完成 -> TFREE
```

pipe op 的 MemoryEffects 和 `OpPipeInterface` 必须保留到 scheduling 和依赖分析完成后，再 lowering 为 wrapper call。还需要核对自动同步 pass，避免对 opaque wrapper 重复插入冲突的 flag 或 barrier。

数值验证还暴露了一个跨 bitcode 调度边界：PTO-ISA `TPOP` 内部会发出 C2V ready wait，但当后续 MTE3 位于另一个 VPTO LLVM 函数体中时，Bisheng 可能在 wait 退休前调度该独立指令。当前固定 wrapper 在 Vector 侧将 ready wait 显式保留在 bridge 边界，关闭 `TPOP` 的重复 wait，并在返回 Tile 地址前执行 pipe barrier。这样只消费一次 ready flag，同时确保 wrapper 返回时 FIFO Tile 已可供 VPTO 后续指令读取。正式实现应把这项依赖建模为 bridge ABI/调度 contract，而不是依赖测试 IR 手写同步。

## 10. PoC 实现与实验结果

当前分支已完成以下实现：

- 新增固定 PoC wrapper 源文件，实例化第一阶段 A5 C2V 配置。
- 使用 `__DAV_CUBE__` 和 `__DAV_VEC__` 分别裁剪 Cube/Vector wrapper。Vector translation unit 不实例化 Cube-only fixpipe 路径。
- VPTO LLVM emitter 将 `InitializeL2LPipeOp`、`TPushOp`、`TPopOp` 和 `TFreeOp` lowering 为固定 wrapper ABI 调用。
- `PipeType` 使用 opaque LLVM pointer 表示，PoC Tile handle 使用 `i64` 表示。
- 在函数退出前生成 `pipe_finish` 调用。
- 排除 internal pipe op 的 TileLib 模板查询。

在 A5 CANN 9.0、Bisheng 15.0.5 环境已验证：

1. Cube wrapper 可按 `dav-c310-cube` 编译为 bitcode，包含 push wrapper、Acc-to-Vec fixpipe、WAIT 和 SET intrinsic。
2. Vector wrapper 可按 `dav-c310-vec` 编译为 bitcode，包含 pop/free wrapper及 WAIT、SET intrinsic。
3. 自构造的 Cube/Vector A5 local C2V IR 均可由 VPTO 导出 LLVM IR，生成 push、pop、free 和 pipe finish 调用。
4. Cube/Vector wrapper bitcode 可分别与真实 VPTO LLVM module 通过 `llvm-link` 合并。
5. 合并后的单一 bitcode 可使用 PTOAS 现有 Bisheng 参数生成对应 device object。
6. 模板实例化和 module 合并均发生在 LLVM bitcode 层，没有生成独立 wrapper device object 再链接。

实验还确认：

- Cube 和 Vector 必须分别实例化所需 wrapper，否则 Vector 编译会遇到 Cube-only target feature。
- Bisheng device 模式缺少可直接使用的 placement new；当前 wrapper 提供 device placement-new 定义作为 PoC 解决方案。
- 使用与 PTOAS device compilation 相同的 target 和 Bisheng 参数，可以避免仅用 `-x ir` 编译时的参数缺失问题。

### 10.1 公开 PTO IR 端到端 CA simulator 验证（2026-08-11）

> 说明：本节的 12 轮协议验证用例保留在研究分支 `feature/vpto-emitc-template-bridge-research` 的 `test/vpto/cases/kernels/fifo-tpush-tpop/` 中；本分支只保留 §10.2 的数值消费用例。

`test/vpto/cases/kernels/fifo-tpush-tpop/kernel.pto` 使用用户可见接口：`pto.entry`、`pto.reserve_buffer` / `pto.import_reserved_buffer`、`pto.aic_initialize_pipe` / `pto.aiv_initialize_pipe`、`pto.tpush_to_aiv`、`pto.tpop_from_aic` 和 `pto.tfree_from_aic`。测试未直接书写 internal pipe op。

完整验证路径为：

```text
public kernel.pto
  -> frontend pipe lowering
  -> Cube/Vector module splitting
  -> CANN900 VPTO LLVM emitter
  -> target-specific vpto_bridge.cpp bitcode
  -> CANN 9.0 LLVM 15 llvm-link
  -> Bisheng device object / fatobj
  -> AIC_MIX CA simulator
  -> host completion-marker comparison
```

测试环境使用 CANN 9.0 的 `Ascend950PR_9599/lib` 模拟器包、dav-c310 混合模式内核和 ACL host runner。

关键运行前提（踩坑记录）：

- 必须链接并加载 `tools/simulator/Ascend950PR_9599/lib` 下的完整模拟器包（含 `libnpu_drv.so` 的 `Ascend910_9599` 设备识别和 `libmodel_top.so` 的 `Ascend950pr_9599_sim` 模型配置）。若使用不完整的 `dav_3102/lib`，模型会回退加载 `Ascend610Lite_config.toml`，kernel task 不会下发，输出全零。
- host runner 中 kernel launch 模板必须只声明并使用 `extern template`；具体实现由 `launch.cpp` 提供。

测试将 12 轮 push/pop/free 静态展开。12 大于 `slot_num=8`，因此覆盖 FIFO slot 回绕。Vector 侧完成全部轮次后写入 `12.0f` 完成标记。

| case | 轮次 | 运行模式 | 结果 |
|---|---:|---|---|
| 1 | 12 | `AIC_MIX` | PASS，完成标记为 `12.0f` |
| 2 | 12 | `AIC_MIX` | PASS，完成标记为 `12.0f` |

结论：

- 模型日志出现 `block_start: AIC_MIX`，Cube、AIV0 和 AIV1 均正常结束。
- 两次独立运行都完成 12 轮，证明当前固定 specialization 的 CCE 模板、同步、context 生命周期和 slot 回绕能够通过 VPTO 后端执行。
- 当前比较只检查协议完成标记。`TPop` 返回的 FIFO slot 地址尚未传播到后续 Tile SSA 使用，因此本测试不构成 Tile 数值内容正确性证明。

验证 case 位于 `test/vpto/cases/kernels/fifo-tpush-tpop/`。目录只保留端到端所需文件：`kernel.pto`、`vpto_bridge.cpp`、`launch.cpp`、`main.cpp`、`golden.py` 和 `compare.py`。运行方式：

```bash
SIM_LIB_DIR=$ASCEND_HOME_PATH/tools/simulator/Ascend950PR_9599/lib \
WORK_SPACE=<workspace> PTOAS_BIN=<ptoas> \
CASE_NAME=kernels/fifo-tpush-tpop \
DEVICE=SIM \
./test/vpto/scripts/run_host_vpto_validation.sh
```

注意：A5 模拟器包必须显式通过 `SIM_LIB_DIR` 指定（脚本默认只自动查找 A3 的 dav_3510 包）。

### 10.2 TPop Tile 数值消费验证（2026-08-13）

`test/vpto/cases/kernels/fifo-tile-data-consume/` 验证输出没有 Cube-to-GM 旁路：GM 输出只可能来自 Vector 侧由 `TPop` 返回地址绑定的 FIFO Tile。

验证数据为 `A = I(16x16)`、`B = arange(16x16)`。Cube 计算 `A @ B`，将 Acc `16x16xf32` Tile 经 C2V FIFO 按上下方向拆分；AIV0 将取得的 `8x16xf32` Tile 写回 GM。`compare.py` 对完整 128 元素使用 `atol=1e-2, rtol=1e-2` 比较，结果为：

```text
[INFO] compare passed
All 1 VPTO case(s) passed
```

CA simulator 日志确认 Cube fixpipe 写入和 Vector MTE3 读取使用同一 UB slot 地址；最终输出与矩阵乘 golden 一致。因此该测试同时证明：

- `TPop` 返回地址没有被占位值覆盖；
- Tile 地址空间转换保持了 UB slot 地址；
- consumer ready wait 在跨 wrapper/VPTO bitcode 边界后仍先于 Tile 消费完成；
- 输出数值确实经过 `TPush -> TPop` FIFO 数据通路。

运行方式：

```bash
SIM_LIB_DIR=$ASCEND_HOME_PATH/tools/simulator/Ascend950PR_9599/lib \
WORK_SPACE=<workspace> PTOAS_BIN=<ptoas> \
CASE_NAME=kernels/fifo-tile-data-consume DEVICE=SIM \
./test/vpto/scripts/run_host_vpto_validation.sh
```

## 11. 当前限制

- specialization 固定为第一阶段的 A5 C2V、`f32`、`16x16 -> 8x16` 配置。
- `TPop` 返回地址已覆盖直线控制流中的 `declare_tile -> tile_buf_addr`；loop、branch、block argument 和 alias 链尚未覆盖。
- ObjectEmission 已能合并外部 Cube/Vector wrapper bitcode，但尚未自动发现 specialization 或生成 wrapper；当前由测试脚本编译并通过 `PTOAS_VPTO_CUBE_BRIDGE_BITCODE` / `PTOAS_VPTO_VECTOR_BRIDGE_BITCODE` 传入。
- pipe lowering 当前直接实现在 CANN900 emitter 中，尚未抽取为可复用的通用 lowering。
- context 生命周期只验证了单 pipe 和简单控制流；复杂 CFG、loop 和多 pipe 尚未覆盖。
- 已通过 CA simulator 验证超过 FIFO 深度后的 slot 轮转和稀疏同步（见 10.1）；尚未在 NPU 实机上验证。
- CA simulator 已验证协议完成、slot 回绕和单 Tile 数值消费；尚未在 NPU 实机验证。
- 自动同步与 wrapper 内同步的职责边界尚未完成系统验证。

## 12. 下一阶段计划

按优先级推进：

1. 将当前 emitter 内的 TPop 地址映射规范化为显式 SSA，并覆盖 loop、branch、block argument 和 alias 链。
2. 在 ObjectEmission 中接入 wrapper 配置收集、源码生成和 target-specific bitcode 编译；bitcode 合并能力已经具备。
3. 用稳定 specialization hash 替换固定 PoC ABI，并支持 context size、alignment 和多 pipe。
4. 将 CANN900 emitter 中的固定 pipe lowering 抽取为可复用组件。
5. 增加单次 push/pop/free 的正式 regression tests。
6. 将当前静态展开的 12 轮测试扩展为循环/CFG 测试，继续验证 context 持久性和析构插入。
7. 核对 scheduling、MemoryEffects、自动同步和 wrapper 内同步的组合行为。
8. 将已通过的 `TPop -> tile_buf_addr -> GM writeback` 数值验证扩展到 Vector compute 后再写回；NPU 实机验证待环境可用后补做。
9. 在第一阶段稳定后，再扩展 V2C、global-entry、其他类型/shape、量化和更多架构。

## 13. 风险与待验证项

### 13.1 已有 PoC 缓解方案

- placement new：当前通过 wrapper 内的 device placement-new 定义解决，但仍需确认长期工具链兼容性。
- bitcode 兼容性：固定环境中已验证 VPTO module、wrapper bitcode 和 device compilation 兼容；跨 CANN/PTO-ISA 版本仍需持续验证。
- Cube/Vector 目标差异：当前通过目标宏裁剪并分别编译解决。

### 13.2 尚未解决

- 更复杂 Tile view/alias 地址在 Acc/Vec 地址空间中的无损传递及类型安全转换。
- `TPOP` 动态地址在 loop、branch、block argument 和 alias 链中的 SSA 表达。
- 多 specialization context 的栈大小、对齐和生命周期管理。
- Cube/Vector 两侧配置和稳定 hash 的一致性保证。
- 自动同步与 wrapper 内 FIFO 同步是否发生重复或冲突。
- CANN/PTO-ISA 版本变化对模板签名、对象布局和行为的兼容性。
