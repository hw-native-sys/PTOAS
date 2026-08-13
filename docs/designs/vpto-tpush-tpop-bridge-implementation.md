# VPTO TPush/TPop 桥接实现说明


## 1. 背景与问题

PTO-ISA 的 `TPush` / `TPop` / `TFree` 不是普通函数，而是 **C++ 模板**：FIFO slot 等待、fixpipe 分片、producer/consumer flag 同步等协议逻辑全部在 Bisheng 编译 C++ 源码时由模板实例化产生。

- **EmitC 后端**生成 C++ 源码，模板参数可以直接出现在输出文本中交给 Bisheng 实例化；
- **VPTO 后端**生成的是 LLVM IR，模板类型信息已经不存在，无法自己"重新实现"这套协议。

本分支的方案：**VPTO 不重新实现 FIFO 协议，而是调用 Bisheng 实例化好的 PTO-ISA 模板**——把 C++ wrapper 编译成 device bitcode，用 `llvm-link` 与 VPTO 生成的 LLVM module 合并，再统一编译为 device object。

## 2. 端到端流程总览

```text
kernel.pto（公开 PTO IR，测试用例书写）
  ↓ ① 输入即 internal pipe IR【main 已有能力，本分支复用】
initialize_l2l_pipe / tpush / tpop / tfree（用例直接书写 internal 形式；
main 自带的 PTOLowerFrontendPipeOpsPass 也可把公开 pipe op 降到这里）
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

## 3. 各文件改动：改了什么、为什么

### 3.1 `lib/PTO/Transforms/VPTOCANN900LLVMEmitter.cpp`（核心，约 180 行）

**改动内容**：

| 位置 | 改动 |
|---|---|
| `convertVPTOType` | `TileBufType` → **i64**（tile 句柄降为整数地址）；`PipeType`（连同 `StructType`）→ **不透明 LLVM pointer** |
| `hasVPTOConvertibleType` | 加入 `PipeType` / `TileBufType`，使这两类值可被降级体系接受 |
| `LoweringState` | 新增 `popTileAddresses` 映射（tile SSA 值 → TPOP 返回的 FIFO slot 地址） |
| 新增 `LowerPipeTileHandlePattern`（benefit=101） | `AllocTileOp` → 规划好的地址（无地址报错）；`DeclareTileOp` → 常量 0（占位）；`TileBufAddrOp` → 查 `popTileAddresses`，命中则用 pop 地址（必要时 `IntToPtrOp` 转地址空间），未命中退回转换后的 operand |
| 新增 `LowerPipeBridgeOpPattern`（benefit=100） | 4 个 pipe op 的降级（见下表） |
| `populateVPTOOpLoweringPatterns` | 注册上述两个 pattern（tile 句柄 pattern 优先） |
| `configureVPTOOpLoweringTarget` | `AllocTileOp` / `DeclareTileOp` / `InitializeL2LPipeOp` / `TPushOp` / `TPopOp` / `TFreeOp` / `TileBufAddrOp` 声明为 illegal，强制走上述 pattern |

四个 pipe op 的降级规则：

| internal op | 降成什么 |
|---|---|
| `InitializeL2LPipeOp` | 校验固定配置（`dir_mask=1`、`slot_size=1024`、`slot_num=8`、`flag_base=0`、`nosplit=false`、无 `acc_push_epilogue`，不满足报明确诊断）；`alloca` storage，大小运行时取自 wrapper 导出的 `pto_vpto_pipe_size()`（align 静态 8）→ 调 `pto_vpto_pipe_init(storage, consumerBuf)` → **在每个 `func.return` 前插入 `pto_vpto_pipe_finish(storage)`**；op 的 SSA 结果替换为 storage 指针 |
| `TPushOp` | 要求 `split=1`、tile 来自带规划地址的 `alloc_tile`；调 `pto_vpto_pipe_push(storage, tileAddr)` |
| `TPopOp` | 要求 `split=1`；调 `pto_vpto_pipe_pop(storage)` 得到 **i64 的 FIFO slot 地址**；写入 `popTileAddresses[op.getTile()]` 并 `replaceAllUsesWith`，使后续 `tile_buf_addr` 拿到真实地址 |
| `TFreeOp` | 调 `pto_vpto_pipe_free(storage)` |

**为什么这么做**：
- wrapper ABI 不暴露 CCE 地址空间指针，Tile 地址用 i64 承载，跨 ABI 干净；
- `TPipe` 是有状态对象，必须活在整个 kernel 生命周期内——storage 分配在 kernel 栈帧上，析构（finish）是协议的一部分，必须保证函数退出前执行；
- `TPop` 会改变 tile 的地址（重绑定到 FIFO slot），而 LLVM IR 是 SSA——emitter 内用显式映射把"重绑定"变成"替换后续使用"；
- wrapper 函数声明通过 `plannedDecls` 统一发射成 LLVM module 里的外部声明，VPTO 产出的 `.ll` 只有 `declare` 没有实现——实现由 bridge bitcode 提供。

### 3.2 `lib/PTO/Transforms/VPTOSplitCVModule.cpp`（约 70 行）

**改动内容**：
- 新增 `getFunctionKernelKind` / `hasFunctionKernelKind`：检测 module 里的函数是否带 `pto.kernel_kind = #pto.kernel_kind<cube|vector>` 属性；
- 新增 `pruneFunctionsForKind`：克隆 module 后，把**非本 kind 的函数及其调用全部删掉**（先用 `StringSet` 记符号名，先删 call 再删 func，避免 use-after-erase）；
- `cloneModuleForKind` 增加 `functionKindInput` 参数：输入本身带函数级 kind 时，跳过原来的 section-kind 裁剪逻辑，并把本 kind 属性补到子模块每个函数上；
- `splitCVModule` 新增路径：检测到函数级 kernel_kind 时，创建外层 module，分别克隆出 Cube / Vector 两个子模块。

**为什么这么做**：Cube 和 Vector 是两种不同的 device object，分别对应 `dav-c310-cube` / `dav-c310-vec` target。一个 kernel 里 Cube 函数和 Vector 函数共存时，必须先按函数 kind 拆成两个子模块，后续 emitter 才能各自发射、各自编译。

### 3.3 `lib/PTO/IR/PTO.cpp`（7 行）

`isInsideSectionOrAttributedKernel` 增加一条判断：**module 级带 `pto.kernel_kind` 属性时也返回 true**。

**为什么**：拆分后子模块里的 op 依赖这个函数判定"是否在 kernel 内"（影响 verifier 与某些 pass 行为）。原实现只认 section 和函数级属性；拆出来的子模块带着 module 级 kind 属性，必须同样被认作 kernel 内部，否则 verifier 会误报。

### 3.4 `lib/PTO/Transforms/FoldTileBufIntrinsics.cpp`（约 28 行）

**改动内容**：
- `resolveTileHandle` 支持 `DeclareTileOp`：只有当该 declare_tile **被 `TPopOp` 重绑定**时才返回合法 handle（否则报错），且返回的 handle 不带静态地址（地址运行时才知道）；
- 主 pass 里，`tile_buf_addr` 的来源是被 TPOP 重绑定的 declare_tile 时**跳过折叠**。

**为什么**：这个 pass 会把 tile 句柄折叠成静态地址。declare_tile 本身没有地址（emitter 里是占位 0），它的真实地址由 TPOP 在运行时返回。如果不跳过，动态地址会在进入 emitter 之前被错误折叠成占位值，FIFO 数据流就断了。

### 3.5 `tools/ptoas/ObjectEmission.cpp`（约 50 行）

**改动内容**：
- 新增 `linkDeviceLLVMBitcode`：调用 CANN toolchain 自带的 `llvm-link`（Bisheng bin 目录下）把 bridge bitcode 和 VPTO 刚写出的 device LLVM IR 合并成一个 `.linked.bc`；
- `emitVPTOVectorDeviceObject` / `emitVPTOCubeDeviceObject`：若设置了环境变量 `PTOAS_VPTO_VECTOR_BRIDGE_BITCODE` / `PTOAS_VPTO_CUBE_BRIDGE_BITCODE`，先合并再走原有 `compileDeviceLLVMToObject` 流程。

**为什么**：这是"把外部 C++ 模板实例化代码塞进 VPTO 产物"的**通用通道**——不限于 TPush/TPop，以后任何 PTO-ISA C++ 模板都可复用同一机制。合并在 bitcode 层完成，不生成独立的 wrapper device object 再链接。

### 3.6 `include/PTO/Transforms/TileOpExpansionUtils.h`（3 行）

`isTileLibExpandableOp` 把 internal 的 `TAllocOp` / `TPushOp` / `TPopOp` / `TFreeOp` 排除在 TileLib 可展开 op 之外。

**为什么**：internal pipe op 必须走专用的 pipe lowering（§3.1），不能进 TileLib 模板候选查询，否则会被误当成普通 tile op 处理。

### 3.7 `test/vpto/scripts/run_host_vpto_validation.sh`（约 35 行）

**改动内容**：
- 新增 `PTO_ISA_INCLUDE_DIR` 环境变量（默认解析到 PTOAS 工作区上层的 `pto-isa/include`）；
- 新增 `build_vpto_bridge_object`：用 Bisheng `-c -emit-llvm -xcce --cce-aicore-only` 按 `dav-c310-cube` / `dav-c310-vec` 两个 target 分别编译用例目录下的 `vpto_bridge.cpp` 为两份 bitcode；
- `build_one_impl`：用例目录含 `vpto_bridge.cpp` 时，先编译 bridge bitcode（step 0/4），再通过 `PTOAS_VPTO_CUBE_BRIDGE_BITCODE` / `PTOAS_VPTO_VECTOR_BRIDGE_BITCODE` 环境变量传给 ptoas。

**为什么**：bridge bitcode 的编译依赖具体用例的 wrapper 源码，属于测试资产而不是编译器内置行为，因此放在用例构建流程里、通过环境变量喂给 ObjectEmission。按 target 分编译是硬要求：Vector 编译撞上 Cube-only target feature 会失败。

### 3.8 `test/vpto/cases/kernels/fifo-tile-data-consume/`（新用例）

| 文件 | 内容 |
|---|---|
| `kernel.pto` | 直接书写 internal pipe IR：entry 调 cube/vector 两个函数（各带 `pto.kernel_kind` 属性）。Cube：`import_reserved_buffer` + `initialize_l2l_pipe`（显式给出 dir_mask/slot_size/slot_num/flag_base/nosplit）+ `alloc_tile` + `mte_*` 搬数 + `mad` 算 16x16 矩阵乘 + `tpush`（split=1）。Vector：`reserve_buffer`（C2V 8192 字节）+ `initialize_l2l_pipe` + `declare_tile` + `tpop` + `tile_buf_addr` + `scf.if`（仅 subblock 0）+ `mte_ub_gm` 写回 + `tfree` + `barrier` |
| `vpto_bridge.cpp` | 固定实例化 `TPipe<0, DIR_C2V, 1024, 8, 2, false>`；`pto_vpto_pipe_init` 用 placement new 在 VPTO 给的 storage 上构造 TPipe；`pto_vpto_pipe_size` 导出 `sizeof(Pipe)` 供 emitter 决定 storage 大小；`push` 用 `TASSIGN_IMPL` 绑地址后调 `TPUSH<TILE_UP_DOWN>`；`pop` 调 `TPOP` 后把 `tile.data()` 作为 i64 返回；`free`/`finish` 对应释放与析构。用 `__DAV_CUBE__` / `__DAV_VEC__` 宏裁剪：Cube 只编 push，Vector 只编 pop/free |
| `launch.cpp` | 提供 `Launch<...>` kernel 启动 stub 的实现 |
| `main.cpp` | host 侧：ACL 分配/搬运 `A=I(16x16)`、`B=arange(16x16)`，单次 launch |
| `golden.py` | 生成 `A@B` 的前 8 行（AIV0 拿到的 8x16 结果）作为 golden |
| `compare.py` | 完整比较 128 个 f32 元素（`atol=1e-2, rtol=1e-2`） |

**为什么这么设计**：输出**没有 Cube-to-GM 旁路**——GM 输出只可能来自 Vector 侧由 TPOP 返回地址绑定的 FIFO tile，所以 compare 通过即证明数据真的走过了 `TPush → TPOP` FIFO 通路（而不只是协议跑通）；`scf.if` + `get_subblock_idx` 保证只有 AIV0 写回，避免两个 subblock 重复写。

## 4. 改动之间的联系

各改动不是孤立的，而是一条**缺一不可的链**：

```text
TileOpExpansionUtils ──┐
（internal pipe op 不走 TileLib，保住 op 原样到达 emitter）
                        │
PTO.cpp ────────────────┤
（拆出的子模块仍被认作 kernel 内部，verifier 不误报）
                        │
                        ↓
        VPTOSplitCVModule（按 kernel_kind 拆 Cube/Vector 子模块）
                        ↓
   FoldTileBufIntrinsics（保住 declare_tile 的动态地址不被折叠）
                        ↓
     VPTOCANN900LLVMEmitter（internal ops → pto_vpto_pipe_* 调用，
     tile 句柄 → i64/指针，pop 地址进 SSA，plannedDecls 发射外部声明）
                        ↓
   run_host_vpto_validation.sh（编 cube/vec 两份 bridge bitcode）
                        ↓
        ObjectEmission（llvm-link 合并 VPTO IR + bridge bitcode）
                        ↓
              Bisheng 编译 device object → fatobj
                        ↓
               CA 模拟器运行 + compare 验证
```

具体依赖关系：

- **TileOpExpansionUtils → emitter**：如果 internal pipe op 被 TileLib 展开，emitter 就看不到 `initialize_l2l_pipe` / `tpush` / `tpop` / `tfree`，后续全链条无从谈起。
- **VPTOSplitCVModule → PTO.cpp**：拆分把函数级 kind 提升为 module 级属性；PTO.cpp 的改动正是为了让这种 module 级属性被"kernel 内"判定认可——两个改动互相配合，缺了 PTO.cpp 的改动拆分后的模块会 verifier 报错。
- **VPTOSplitCVModule → emitter → ObjectEmission**：拆出的 Cube/Vector 子模块分别发射 LLVM module，ObjectEmission 分别读对应的 `PTOAS_VPTO_{CUBE,VECTOR}_BRIDGE_BITCODE` 合并——Cube 侧合并 push 实现，Vector 侧合并 pop/free 实现。
- **FoldTileBufIntrinsics → emitter 的 `popTileAddresses`**：前者保证 declare_tile 的 `tile_buf_addr` 带着原样到达 emitter；后者在 emitter 内把 TPOP 返回地址注入该使用点。两者共同实现"运行时重绑定"的语义。
- **脚本 → ObjectEmission**：脚本产出 bridge bitcode 并通过环境变量传入，ObjectEmission 只负责合并——bridge 的编译属于测试/工具链资产，与编译器行为解耦。
- **测试用例 → 全部改动**：`fifo-tile-data-consume` 用例同时触发了以上所有路径（函数级 kernel_kind、declare_tile+tpop、桥接 bitcode 编译与合并），是整条链的端到端回归。

## 5. TPush/TPop 功能最终是如何实现的（按执行顺序）

以 `fifo-tile-data-consume` 用例的一次完整编译运行为例：

**第 1 步：输入即 internal pipe IR（main 已有能力，本分支未改）**
`fifo-tile-data-consume` 的 `kernel.pto` **直接书写 internal pipe op**（`pto.initialize_l2l_pipe` / `pto.tpush` / `pto.tpop` / `pto.tfree`），配置在 op 属性里显式给出（`dir_mask=1`、`slot_size=1024`、`slot_num=8`、`flag_base=0`、`nosplit=false`）。这些 op、`reserve_buffer` / `import_reserved_buffer` 的 peer 配对解析和内存规划都是 main 上已有的能力：main 自带的 `PTOLowerFrontendPipeOpsPass` 还能把公开 pipe op（`pto.tpush_to_aiv` 等）降成同一套 internal IR——本用例直接写了 internal 形式，因此没有经过这一步。emitter 从 op 属性读配置，不需要自己猜任何默认值。

**第 2 步：模块拆分（§3.2）**
`VPTOSplitCVModule` 看到两个函数分别带 `pto.kernel_kind = cube` / `vector`，克隆出两个子模块，各自剪掉对方的函数和调用，并打上 module 级 kind 属性（§3.3 保证后续 verifier 认可）。

**第 3 步：地址折叠保护（§3.4）**
`FoldTileBufIntrinsics` 处理 Vector 侧 `declare_tile → tpop → tile_buf_addr`：被 tpop 重绑定的 declare_tile 的地址查询被跳过，不折叠成占位值。

**第 4 步：emitter 降级（§3.1，核心）**
Cube 子模块里：
- `alloc_tile` → 规划地址（i64）；
- `initialize_l2l_pipe` → `alloca`（大小取 `pto_vpto_pipe_size()` 返回值）+ `pto_vpto_pipe_init(storage, fifo_addr)`，函数 return 前插 `pto_vpto_pipe_finish(storage)`；
- `tpush` → `pto_vpto_pipe_push(storage, acc地址)`。

Vector 子模块里：
- `initialize_l2l_pipe` → 同上（用 `reserve_buffer` 的地址）；
- `declare_tile` → 占位 0；
- `tpop` → `pto_vpto_pipe_pop(storage)`，返回的 i64 slot 地址写入 `popTileAddresses` 并替换 tile 的所有使用；
- `tile_buf_addr` → 查映射得 slot 地址，转成 UB 地址空间指针；
- `tfree` → `pto_vpto_pipe_free(storage)`。

发射出的 LLVM module 里只有这些 wrapper 函数的外部声明，没有实现。

**第 5 步：bridge 编译（§3.7）**
测试脚本用 Bisheng 把 `vpto_bridge.cpp` 按 `dav-c310-cube` / `dav-c310-vec` 编译成两份 bitcode。Cube 版包含 `TPipe` 构造/析构 + `TPUSH` 模板实例化（含 Acc-to-Vec fixpipe 分片）；Vector 版包含 `TPOP` / `TFREE` 实例化。`TPOP` 内部等待 C2V ready 后把 Vec tile 重绑定到当前 FIFO slot，wrapper 把 `tile.data()` 地址作为 i64 返回。

**第 6 步：bitcode 合并（§3.5）**
ObjectEmission 读到环境变量后，用 CANN 的 `llvm-link` 把"只有 declare 的 VPTO module"与"有实现的 bridge bitcode"合并为单一 bitcode，再走原有 Bisheng 编译流程生成 device object、进 fatobj。

**第 7 步：运行与验证（§3.8）**
host 侧 launch，CA 模拟器上 Cube 执行 `mad`（16x16 矩阵乘）后 `TPUSH` 把 Acc tile 经 fixpipe 分片写入 FIFO；Vector 侧 AIV0 的 `TPOP` 等到 ready、拿到 slot 地址，`mte_ub_gm` 从该地址把 8x16 结果写回 GM。compare 对全部 128 个 f32 元素与 `A@B` 前 8 行 golden 比较——通过即证明计算结果真的经过了 FIFO 数据通路。

## 6. 当前边界（简述）

- 只支持固定 specialization：A5 C2V、`f32`、`16x16 → 8x16`、`split=1`、`TPipe<0, C2V, 1024, 8, 2, false>`；其他配置报明确诊断；
- TPOP 地址传播已验证直线控制流；loop / branch / block argument / alias 链未覆盖；
- bridge bitcode 由测试脚本编译、环境变量传入，尚未在 ObjectEmission 内自动收集配置并生成 wrapper；

