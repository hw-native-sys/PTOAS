<!--
Copyright (c) 2026 Huawei Technologies Co., Ltd.
This program is free software, you can redistribute it and/or modify it under the terms and conditions of
CANN Open Software License Agreement Version 2.0 (the "License").
Please refer to the License for details. You may not use this file except in compliance with the License.
THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
See LICENSE in the root of the software repository for the full text of the License.
-->

# TEXTRACT ND to 2xNZ 双输出接口设计

## 1. 范围与基线

本文定义 PTOAS 对 PTO-ISA `TEXTRACT` 双输出 ND-to-2xNZ overload 的完整承载方案，
覆盖 PTO IR、verifier、EmitC、DPS、内存效应、同步、内存规划、TileLib、
文档和测试。

本文是设计契约，不包含功能实现。后续实现 PR 必须与这里描述的 verifier、driver helper、
lowering 和兼容层保持一致；本设计 PR 只用于评审方案和实现边界。

PTO-ISA A2/A3、A5 的公开双输出接口是本设计的语义依据；本 PR 不修改外部依赖配置，
也不引入新的 backend 发现或能力声明机制。

当前 PTOAS 基线中，`TExtractOp` 只有一个 `$dst`，且没有 ND-to-2xNZ 的 ODS、
lowering、verifier、TileLib template 或回归测试；后续实现拟在同一个 `TExtractOp` 上补齐该缺口。

## 2. 最终决策

1. 不新增 MLIR operation 或 dotted mnemonic。扩展现有 `TExtractOp`，文本名继续为
   `pto.textract`；由 index 段、DPS destination 段和 tile layout 推断单输出或 ND-to-2xNZ
   overload，不增加 `kind`/`mode` 属性。
2. 现有 form 固定为一个 source、两项 index 和一个 DPS destination；新增 form 固定为
   一个 ND source、四项 index 和两个 NZ DPS destination。两个 destination 可以有不同
   physical/valid shape，但 element type 必须与 source 相同。classifier 必须先验证完整
   `[src, indices, dsts, fp, preQuantScalar]` segment schema，特别是 `src == 1` 和 optional segment
   为 `0/1`；不能只按 index/destination arity 推断。
3. `TExtractOp` 继续实现 `PTO_DpsInitOpInterface` 且不增加 SSA result；单输出 form 返回一个
   DPS init，双输出 form 返回两个连续的 DPS init。
4. 双输出 form 的 pipe 固定为 `PIPE_V`；内存效应为 `Read(src)`、`Write(dst0)`、
   `Write(dst1)`。单输出 form 保持现有 pipe/effects 分派。
5. `src`、`dst0`、`dst1` 三者必须两两不重叠。legacy 和 modern PlanMemory 都执行
   双输出 form 的 no-alias 契约；规划后的静态地址/no-alias 由 driver 普通 helper 统一复核，
   不新增 address validation pass。
6. 首版不支持 runtime-bound tile provenance。`DeclareTileOp`、`TAssignOp`、`TPopOp`/
   `TPopFromAicOp`/`TPopFromAivOp` 绑定或产生的 tile，以及它们的 view/subview/cast 派生值，
   在所有 level 都必须在 PlanMemory 之前被拒绝；只有 planner-owned `alloc_tile` 或已物化的
   plain-NZ `alloc_multi_tile` slot 才能作为该 form 的 local tile provenance。RowPlusOne
   destination 必须追溯到单 `AllocTileOp`；`AllocMultiTileOp`、`MultiTileGetOp` 及其
   view/subview/cast chain 在本设计中永久 unsupported，即使 RowPlusOne 的物理布局支持条件满足也拒绝。
   level1/2 的地址由 planner 产生，level3 则要求调用方提供可静态证明的地址。这样 no-alias
   契约不会依赖 planner 对 runtime handle 的猜测，也不会复用当前把 slot stride 与 access size
   合并为一个字段的 multi-buffer 路径。
7. EmitC 精确生成七参数公开 API：
   `TEXTRACT(dst0, dst1, src, indexRow0, indexCol0, indexRow1, indexCol1)`。
   不直接生成内部名字 `TEXTRACT_ND2XNZ_IMPL`。
8. A2/A3 与 A5 共用 IR 形态，verifier 按 target arch 分派 dtype 和 compact mode 规则；
   只有通过目标 backend compile 和数值测试的组合才进入正向支持集合。
9. A5 TileLib 增加独立 template。A2/A3 EmitC 继续走 tile-level PTO-ISA 调用；A2/A3 VPTO
   则在现有 `LowerPTOToUBufOps` 中把双输出 form 展开成已有 scalar pointer op 和 `scf.for`，
   不能让 pointer-form `pto.textract` 残留到 VPTO LLVM emission，也不为此新增另一套 TileLib
   目录或 backend op。
10. 实现 PR 只在 PTOAS 已配置的目标 backend 上生成本 overload；依赖版本、外部头文件和
   backend 是否可实例化不在本设计中新增 driver/API 约束。目标 backend 缺少该 overload 时，
    由既有构建配置和编译测试报告失败，不引入新的依赖或 driver 机制。
11. partial-valid/odd/`1x1` 是定义明确的 TEXTRACT/UB-only 支持，不自动获得 NZ TSTORE
    支持；driver 的 post-planning safety helper 按静态、带 address space 的 physical range 而不是 SSA use chain，
    在同一 direct-call graph component 内拒绝 partial destination 的所有 alias TSTORE，不能因
    producer 和 TSTORE 位于不同函数而漏检。helper 固定运行在 resolve-buffer-select 后、
    backend-helper inlining 前；普通 codegen 与 `--emit-pto-ir` 共用该切点。任一 partial producer
    存在时，整个 compile unit 的 call surface 还必须闭合；opaque/external call 不能以位于另一个
    direct component 为由绕过检查。首版没有 test-only escape hatch。
12. CPU-sim、cost-model 和其他 optional backend 不属于本设计的新增支持范围；其编译能力
    由各自原有构建与测试流程决定。
13. 既有 `pto.textract` 文本、Python `TExtractOp`/`pto.textract` 调用、legacy
    `.indexRow`/`.indexCol`/`.dst` properties 和 PTOBC v0 单输出 wire schema 保持兼容；
    新双输出 form 使用同一 op class 的新 builder，并以新的 PTOBC generic record 承载，不能复用
    已发布的四/五 operand fixed-width opcode。
14. backend-partitioned outer module 在 child 拆分前建图；local declaration 必须按 exact final-link
    symbol 穿透到唯一 sibling public definition。outer module 含任一 partial producer 时，首版禁止
    所有跨 child direct call，并在 reserved-buffer resolution 前禁止任意 `peer_func` import（无论
    同 child、跨 child 或与 partial component 是否连通），因此 child clone 不会留下需要闭包例外的
    sibling declaration，也不会在 helper 建图后才暴露 peer reserve 同址关系；declaration 的零/多
    sibling 匹配不能成为调用图终点。只要模块中出现 ND-to-2xNZ form
    且存在 descendant `ModuleOp`，首版就执行 fixed-depth structure guard：根 module body 只能
    包含 immediate backend child，所有 `func.func` 必须直接属于 immediate child，child 内不得再
    嵌套 `ModuleOp`。
15. A5 `RowPlusOne` 首版保持关闭。不能只把 emitted `Tile::Rows` 改成
    `physicalRows + 1`；subview 地址、legacy/modern PlanMemory、semantic range、InsertSync、
    EmitC/TSTORE 必须先共用同一个 checked physical-layout/access helper，并通过
    `ColMajor NZ f16 16x32` 的 272-element 第二 block offset、1024-byte payload、1056-byte
    access-end 和 1088-byte allocation-reservation 回归。1088 bytes 是允许的完整矩形 reservation，
    不是 TSTORE 的 access range。
16. 本设计不开放 RowPlusOne `AllocMultiTileOp`。基线 verifier 已因 slot stride 仍按
    `product(shape)` 而拒绝该组合；这个拒绝在 RowPlusOne 物理布局支持条件满足后仍保留。
    后续支持必须另起设计，在 ODS、slot address materialization 和 InsertSync 中分别
    表示 slot reservation stride 与 per-slot access end，不能由本功能的 shared tile helper
    静默改变既有 multi-buffer ABI。

## 3. PTO-ISA 真实契约

### 3.1 公开 API

最新版 `include/pto/common/pto_instr.hpp` 中的接口为：

```cpp
template <
    typename Dst0TileData, typename Dst1TileData, typename SrcTileData,
    typename... WaitEvents,
    std::enable_if_t<
        is_tile_data_v<SrcTileData> && all_events_v<WaitEvents...>, int> = 0>
PTO_INST RecordEvent TEXTRACT(
    Dst0TileData& dst0, Dst1TileData& dst1, SrcTileData& src,
    uint16_t indexRow0 = 0, uint16_t indexCol0 = 0,
    uint16_t indexRow1 = 0, uint16_t indexCol1 = 0,
    WaitEvents&... events) {
  TSYNC(events...);
  MAP_INSTR_IMPL(TEXTRACT_ND2XNZ, dst0, dst1, src,
                 indexRow0, indexCol0, indexRow1, indexCol1);
  return {};
}
```

因此 PTOAS 必须保留以下事实：

- 两个 window 的 index 相互独立；
- 两个 destination 的类型是独立 C++ template parameter，shape 不要求相同；
- 两个 destination 与 source 属于同一次公开 `TEXTRACT` 调用；
- wait event 不进入 PTO IR operand，仍由 PTOAS 的同步 pass 管理。

### 3.2 数值语义

对 `k in {0, 1}`，逻辑 window 为：

```text
window_k[r, c] = src[indexRow_k + r, indexCol_k + c]
  0 <= r < dst_k.validRows
  0 <= c < dst_k.validCols
```

`window_k` 写入 `dst_k` 的 NZ 排布。令 `c0 = 32 / sizeof(T)`，plain NZ 的核心线性
偏移为：

```text
dstOffset = floor(c / c0) * dstPhysicalRows * c0
          + r * c0
          + (c % c0)
```

这里的 `dstPhysicalRows` 是 destination type 的 physical row extent，不能替换成
`align16(validRows)`。A2/A3 和 VPTO 的 block stride 都必须从该 physical extent 计算。
A5 首版还没有经过 PTO-ISA/TileLib 验证的任意 partial-valid plain-NZ stride 语义，因此增加一个
backend 物理 stride 限制：对 A5 partial-valid plain NZ，必须满足
`physicalRows == align16(validRows)`；不满足时在 A5 backend-boundary verifier 阶段拒绝，不能
让 TileLib 静默使用 `align16(validRows)` 生成另一种布局。满足该限制后，plain NZ 的 block
stride 为 `physicalRows`。NZ+1/`RowPlusOne` 另受第 5.4 节更严格的全链路物理布局约束；
该限制解除前，full-valid 和 partial-valid 双输出 form 都拒绝 RowPlusOne。

A5 `RowPlusOne` 在相邻 NZ column block 之间增加一行 bank-conflict padding；padding
不是逻辑输出，`valid_shape` 不随之增大。destination 的 valid 区域之外不定义新值，
调用方不能依赖未写 padding 的内容。

两个 window 可以重叠读取同一 source 区域；这不影响语义。两个 destination 不能互相
重叠，也不能与 source 重叠，因为原生实现按 window 顺序读写，alias 会破坏尚未读取的
source 或使两个可观察输出互相覆盖。

### 3.2.1 partial-valid 与 TSTORE 边界

`TEXTRACT` 的 partial-valid 语义只承诺上面的 logical window。它不承诺对
`valid_shape` 以外的物理元素、NZ block 尾部或 `RowPlusOne` gap 写入任何值。实现和
测试不得把“目的 tile 之前恰好被清零”推导成 TEXTRACT 的输出保证。

因此实现 PR 将测试和 codegen 明确分成两个模式：

1. **full-store eligible**：对每个 destination 都有
   `validRows == physicalRows` 且 `validCols == physicalCols`（对 A5
   `RowPlusOne` 还必须先满足第 5.4 节完整 physical-layout 支持条件）。该模式才生成
   `TLOAD -> TEXTRACT -> TSTORE`。plain NZ 的二维 destination 使用确定的 canonical
   GlobalTensor NZ shape：

   ```text
   [1, physicalCols / c0, physicalRows / 16, 16, c0]
   ```

   于是 PTO-ISA 的 NZ TSTORE 断言
   `validRow == shape[2] * shape[3]`、
   `validCol == shape[0] * shape[1] * shape[4]` 在 debug 和 release 都成立；GlobalTensor
   strides 再按真实 GM view 填入，不能用 valid shape 代替 physical shape。

2. **partial-valid / UB-only**：允许 `valid != physical`、odd `validCol` 和 `1x1`，但不得把
   partial descriptor 或任何与其同 address space physical range 相交的 alias 作为 generic
   `TSTORE` source。
   simulator 可直接观测 UB；NPU 必须使用生产 PTO IR 之外的 backend-native raw-UB dump
   harness。golden 只比较两个 destination 各自的 valid logical region，未定义区不参与比较。
   A5 partial-valid UB-only codegen仍受第 3.4 节 physical-stride 限制；例如
   `physicalRows=32, validRows=13` 不能因为不经过 TSTORE 就绕过 gate。

当前 PTOAS 没有一个能把静态 partial tile 安全地改成 full-valid tile 的现有操作：
`pto.set_validshape` 仅接受 `v_row=?/v_col=?` 的本地动态 tile，并且只修改运行时元数据。
因此首版不把同一个 partial descriptor 伪装成 full-valid，也不在 generic `TSTORE` 中放宽契约。
driver 的 `validateTExtractNd2xNzPostPlanningSafety()` 在内存规划和
`PTOResolveBufferSelect` 完成后、`PTOInlineBackendHelpersPass` 前按带 address space 的 physical
range 检查同址但
不同 SSA root 的 alias；直接使用 partial descriptor、经 view 派生后使用，或者另建同址
full-valid `alloc_tile` 后 TSTORE，都必须在 PTOAS 阶段报错：

```text
pto.tstore source physical range aliases a partial-valid ND-to-2xNZ destination
in the same address space and call component; undefined NZ padding cannot be stored
```

首版故意不提供 CMake test hook、hidden CLI flag 或输入 IR attribute escape。post-planning
helper 对已闭合的 direct-call graph 采用分量级 range 规则：只要同一 weakly connected component
内任一 `TStoreOp.src` 与 partial destination 在同一 address space 中的 physical range 相交就拒绝，
不同 address space 即使数字地址相同也不构成 alias；不考虑函数内文本
顺序、caller/callee 方向、控制流支配关系或中间的 full overwrite。闭包检查本身更保守：只要
compile unit 中存在任一 partial producer，就扫描所有函数并拒绝任意位置的 `func.call_indirect`、
external/unresolved direct callee 或其他无法解析为 internal definition 的 `CallOpInterface`；这些
调用被视为可能连接任意 direct component，不能因 opaque site 与 partial producer 暂时位于不同
component 而放行。闭包成功后，互不连通的独立 kernel/function component 仍可复用相同数值的
UB 地址。这会过拒绝少量理论上可证明安全的程序，但避免为首版引入任何新的 interprocedural
CFG/dataflow validation pass、indirect-target analysis、call-summary fixed point 和测试专用语义。

NPU partial-valid 覆盖若需要导出 UB，必须使用独立 `test/npu_validation` raw-buffer harness，
通过 backend-native UB-to-GM byte copy 导出 physical allocation extent 和紧贴其前后的 redzone；该
harness 不把 full-valid alias `TSTORE` 重新送入 PTOAS production pipeline。在 raw harness
落地前，对应 odd-valid/`1x1` 只能计 compile-only 或 simulator coverage。

后续若要支持 partial-valid 的完整写回，必须先增加一个经过 backend 验证的
full-valid materialization（动态 valid tile、物理 extent 和 `SetValidShape` 的顺序均需
有 IR/EmitC/VPTO 语义），或者让 PTO-ISA 为 partial NZ TSTORE 提供明确的 padding 语义并
移除上述 debug assertion。两者落地前，`1x1` 和 odd-valid case 只能宣称 TEXTRACT/UB
覆盖，不能计入完整 TSTORE coverage。

### 3.3 共同结构约束

PTO-ISA A2/A3 与 A5 的本 overload 都直接检查：

- source 和两个 destination 均为 `TileType::Vec`，即 PTOAS `loc=vec`；
- source 是 ND：`BLayout::RowMajor` + `SLayout::NoneBox`；
- destination 是 NZ：`BLayout::ColMajor` + `SLayout::RowMajor`；
- source/destination element type 相同；
- source row stride 的 byte 数为 32B 对齐；
- destination physical cols 是 `c0` 的整数倍；
- 每个 window 分别满足：

```text
indexRow_k + dst_k.validRows <= src.physicalRows
indexCol_k + dst_k.validCols <= src.physicalCols
indexRow_k + dst_k.validRows <= src.validRows
indexCol_k + dst_k.validCols <= src.validCols
```

前两项防止读出 source allocation，后两项防止在 allocation 内读取 source 的未定义
padding。四项都按 window 独立检查；不能只复用当前单输出 helper 中的
`dst.physicalShape`/`src.physicalShape` 检查。首版要求 source 的 valid shape 也是静态、非零且
可归一化的；如果调用方没有可证明的 source valid extent，必须改用显式
`valid == physical` 的 full-valid source 或在 verifier 阶段拒绝，不能把 physical extent
默认为 valid extent。

destination 的 `validRows`、`validCols` 不要求等于 physical shape，也不要求都是
fractal 倍数。plain NZ 的 PTOAS physical rows 仍须按 16 rows 对齐，使类型能被 NZ
GlobalTensor/TSTORE 链路承载；这是完整存储链路约束，不是上述 overload 自己的
`CheckTExtractNdToNz` static assert。PTO-ISA 已覆盖 `1x1`、非对齐 index，以及 A2/A3
`int8` odd validCol；PTOAS verifier 不得添加这些上游不存在的限制。

上述“不要求相等”是 IR/common contract；A5 的 partial-valid backend 物理 stride 限制是额外的
可执行约束：`physicalRows != align16(validRows)` 的形态在 A2/A3/VPTO 可以按 physical stride
表达，但 A5 首版必须拒绝。这样 `physicalRows=32, validRows=13, validCols >= 2*c0` 不会
在 A5 产生第二个 block 写入 `16*c0`、而 A2/A3 写入 `32*c0` 的跨后端分歧。A5 要支持该形态，
必须先完成并验证对应 PTO-ISA/TileLib 的物理布局实现，再移除该限制。

### 3.4 架构差异

| 约束 | A2/A3 | A5 |
|---|---|---|
| header dtype 集合 | `i8`, `i32`, `f16`, `bf16`, `f32` | A2/A3 集合，加 `hif8`, `f8E4M3`, `f8E5M2`, `f8E8M0`, `f4E2M1x2`, `f4E1M2x2` |
| plain NZ | 支持 | 支持 |
| NZ+1 / `RowPlusOne` | 不支持 | PTO-ISA 支持；PTOAS 首版关闭，满足第 5.4 节后才开放 |
| partial-valid NZ block stride | physical rows；VPTO 同样 | plain 首版要求 `physicalRows == align16(validRows)` 并使用 physical rows；NZ+1 首版关闭 |
| 非 32B source base | scalar fallback | SIMD unaligned path |
| `1x1` | scalar path | scalar path |
| `i8` odd validCol | f16 widen/reshape/narrow | 原生 byte SIMD |

A2/A3 的 `indexCol * sizeof(T)` 不满足 32B 对齐时会走 scalar fallback，而不是非法输入；
A5 的 sub-c0 `indexCol` 由 `vldas`/`vldus` 路径处理。因此 verifier 只验证 bounds，不验证
index 对齐。

### 3.5 FP4 维度域

PTOAS 的 `!pto.f4E*M*x2` 是一个 byte 存两个 FP4 的 packed type。EmitC 生成 Tile type时，
`renderTileTemplateDim` 会把 packed dimension 放大 2 倍：

- RowMajor ND 的 packed dimension 是 column；
- ColMajor NZ 的 packed dimension 是 row。

所以 FP4 的 alignment 和 bounds 必须在“最终生成的 PTO-ISA Tile dimension”上校验，
不能直接拿 raw PTO IR shape 与普通 byte dtype 共用公式。实现应把
`renderTileTemplateDim` 的维度归一化逻辑提取到共享 PTO type utility，并区分 physical
与 valid dimension，例如：

```cpp
int64_t getPTOIsaPhysicalTileDim(TileBufType tile, unsigned dim);
int64_t getPTOIsaValidTileDim(TileBufType tile, unsigned dim);
```

verifier 和 EmitC 都调用同一 helper，防止一边校验 raw dim、另一边输出 doubled dim。

但 header 支持列表不等于已经验证：8 月 14 日新增的 A5 ST 没有实例化两种 FP4。当前
A5 implementation 又固定对 `validCol/indexCol` 除 2，而 PTOAS 对 ColMajor NZ 的 packed
dimension 是 row；这条轴向和 source row-stride 必须先用最小生成 C++ 与设备 golden
确认。第一版 verifier 在验证完成前拒绝 FP4，诊断为“当前 ND-to-2xNZ
FP4 path 尚未验证”；验证通过后再把两种 FP4 加入正向集合。不能只靠 static assert
成功或 EmitC 文本正确就放行。

### 3.6 后端范围

本设计只定义 PTOAS 对现有 `pto.textract` 双输出 form 的 IR 形态和 lowering 契约，不新增
backend 专用接口。A2/A3、A5、VPTO、CPU-sim
和 cost-model 是否能实例化该 overload，继续由各自现有依赖和构建流程决定；实现 PR 只在
目标 backend 已具备对应 API 的路径上增加 compile/lit 回归，不把 backend 探测或依赖升级扩展
为本功能的 driver 设计。本文只把现有 backend 支持矩阵作为实现前提，不重新设计它。

PTODSL micro-op surface 不是当前缺口。PTOAS 基线的 `ptodsl/ptodsl/pto.py` 已公开导出
`vldas`、`vldus` 和 `vsstb`，`ptodsl/ptodsl/_ops.py` 已实现三者 builder，且
`ptodsl/tests/test_jit_compile.py` 覆盖普通 `vsstb` 和 post-update 形态。个别 DSL ST 中
“`vsstb.post` 尚未暴露”的注释已经落后于当前源码，不能据此要求新增另一套 surface。

## 4. PTO IR 设计

### 4.1 ODS

不定义 `TExtractNd2xNzOp`。现有 `TExtractOp` 把坐标和 destination 改为分段 range；示意 ODS
如下，具体 builder 声明按仓库生成绑定的方式落地：

```tablegen
def TExtractOp : PTO_TOp<"textract", [
  AttrSizedOperandSegments,
  PTO_DpsInitOpInterface,
  OpPipeInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> {
  let summary = "Extract source windows into DPS destinations";

  let arguments = (ins
    PTODpsType:$src,
    Variadic<Index>:$indices,
    Variadic<PTODpsType>:$dsts,
    Optional<PTODpsType>:$fp,
    Optional<I64>:$preQuantScalar,
    OptionalAttr<PTO_AccToVecModeAttr>:$accToVecMode,
    DefaultValuedAttr<PTO_ReluPreModeAttr,
      "::mlir::pto::ReluPreMode::NoRelu">:$reluPreMode
  );

  let results = (outs);
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    enum class Form { Invalid, SingleOutput, NdTo2xNz };
    Form classifyForm();
    bool isSingleOutputForm();
    bool isNdTo2xNzForm();

    // Legacy convenience accessors keep their exact generated return types.
    // They may only be used after isSingleOutputForm(); range-aware code uses
    // getIndices/getDsts.
    ::mlir::TypedValue<::mlir::IndexType> getIndexRow();
    ::mlir::TypedValue<::mlir::IndexType> getIndexCol();
    ::mlir::TypedValue<::mlir::Type> getDst();
    ::mlir::OpOperand &getIndexRowMutable();
    ::mlir::OpOperand &getIndexColMutable();
    ::mlir::OpOperand &getDstMutable();

    ::mlir::MutableOperandRange getDpsInitsMutable();
    ::mlir::pto::PIPE getPipe();
    void print(::mlir::OpAsmPrinter &p);
    static ::mlir::ParseResult parse(
        ::mlir::OpAsmParser &parser, ::mlir::OperationState &result);
  }];
}
```

flattened operand 顺序固定为 `src, indices..., dsts..., fp?, preQuantScalar?`。对旧 form，这仍是
`src, indexRow, indexCol, dst, fp?, preQuantScalar?`，因此既有 lowering 中可观察的 operand
顺序不变；双输出 form 则是 `src, row0, col0, row1, col1, dst0, dst1`。两个 DPS init 连续，
符合 `PTO_DpsInitOpInterface` 的单一 `MutableOperandRange` 契约。

现有 declarative `assemblyFormat` 无法同时稳定表达 legacy optional operand 和两个 variadic
range，改为 custom parser/printer。printer 对单输出 form 必须逐字符保持现有 canonical 语法；
双输出 form 只增加第二组坐标和第二个 `outs` operand，不引入 suffix mnemonic。

### 4.2 form 推断与非法组合

`classifyForm()` 返回 `SingleOutput`、`NdTo2xNz` 或 `Invalid`。它不能先调用任何依赖 segment
offset 的 generated operand accessor，包括 `getSrc()`、`getIndices()`、`getDsts()`、`getFp()`。
LLVM/MLIR 19 把 `operandSegmentSizes` 存储在 inherent properties 中，`Operation::getRawDictionaryAttrs()`
明确不包含它。本 op 的 schema helper 固定从 typed property 读取：

```cpp
const std::array<int32_t, 5> &segments =
    op.getProperties().operandSegmentSizes;
```

不得从 `getRawDictionaryAttrs()` 查找该值。若某个 operation-generic 工具确实不能使用 typed
property，只能调用会经过 `getInherentAttr()` 的
`op->getAttrOfType<DenseI32ArrayAttr>("operandSegmentSizes")`。两条路径不能各自实现一套分类规则。

新 ODS 的 property 类型是固定长度 `std::array<int32_t, 5>`，因此长度不是五项的文本在
property conversion/parser 阶段失败，尚未形成可交给 classifier 的 `TExtractOp`；文本中省略
property 则得到默认的全零数组，不存在可由 classifier 区分的“missing attribute”状态。对已构造
operation，schema helper 一次性验证：

1. 五项顺序为 `[src, indices, dsts, fp, preQuantScalar]`；每项非负，五项之和等于 raw operand
   数量。
2. `src == 1`，`fp` 和 `preQuantScalar` 各自只能为 `0` 或 `1`。
3. 只有以下两个完整 schema 可以分类，不能只看 `indices`/`dsts` 两段：

   | form | 完整 `operandSegmentSizes` |
   |---|---|
   | `SingleOutput` | `[1, 2, 1, fp, preQuantScalar]`，其中两个 optional size 各为 `0/1` |
   | `NdTo2xNz` | `[1, 4, 2, 0, 0]` |

其他 schema 一律为 `Invalid`。例如 `[0, 2, 1, 1, 0]` 不能因为 index/dst arity 看似是
单输出就调用 `getSrc()`；`[2, 2, 1, 0, 0]` 也不能把第二个 source 留在 effects 之外。
custom parser 和所有 typed builder 只生成上述 canonical schema。对于已成功完成 property
conversion 且 generated invariants 也通过的非法五段值，custom verifier 给出包含实际五段值和
期望 schema 的诊断；property cardinality/conversion 错误和先被 generated invariant 捕获的错误
保留对应阶段的 MLIR 诊断，不承诺统一进入 classifier。

上述逻辑只实现一份内部 property-schema helper；`classifyForm()` 把 helper failure 映射为 `Invalid`，
`verify()` 使用同一 failure detail 发诊断，`getDpsInitsMutable()`、`getPipe()` 和 MemoryEffects
复用同一结果。不能分别重写五段判断，也不能依赖 `AttrSizedOperandSegments` trait 代替该 helper，
因为 trait 的总数检查不证明固定/optional segment 的本 op 语义。

双输出 form 还必须由类型二次确认：source 是 `loc=vec` 的 ND，两个 destination 都是
`loc=vec` 的 NZ；否则不能仅因 arity 相同就选择七参数 PTO-ISA overload。该 form 禁止 `fp`、
`preQuantScalar`、非默认 `reluPreMode` 和 `accToVecMode`。单输出 form 继续走现有 MAT/ACC/VEC、
FP、pre-quant、relu 和 acc-to-vec verifier 分支，其合法集合不因本功能扩大。

所有 interface 都必须对 `Invalid` fail-safe：`getPipe()` 在 verifier 报错前返回 `PIPE_V`；
`getDpsInitsMutable()` 用同一 property-schema helper 计算 offset，schema 非法时返回空 range，不能
调用 generated `getDstsMutable()`；MemoryEffects 按第 6.2 节保守处理 raw operands。任何 legacy
convenience accessor 在 debug build 断言 `SingleOutput`。generic assembly 即使构造畸形
segments，classifier 和这些 custom interface 也不能越界访问或产生未建模的 tile operand。

这里推断的是 **TEXTRACT overload/form**，不是凭 source 自动创建 destination type。DPS
destination 已由 allocation 决定，所以两个 NZ destination 的 physical shape、valid shape 和
compact mode 仍显式存在于 operand type 中；这正是允许两路 shape 不同所必需的。

### 4.3 汇编示例

现有单输出文本不变：

```mlir
pto.textract
  ins(%src, %r0, %c0 : !pto.tile_buf<vec, 64x128xf16,
                           blayout=row_major, slayout=none_box>, index, index)
  outs(%dst : !pto.tile_buf<vec, 32x64xf16,
                            blayout=row_major, slayout=none_box>)
```

新增双输出文本为：

```mlir
%src = pto.alloc_tile
  : !pto.tile_buf<vec, 64x128xf16,
                  blayout=row_major, slayout=none_box>
%dst0 = pto.alloc_tile
  : !pto.tile_buf<vec, 32x64xf16, valid=32x64,
                  blayout=col_major, slayout=row_major>
%dst1 = pto.alloc_tile
  : !pto.tile_buf<vec, 16x32xf16, valid=13x29,
                  blayout=col_major, slayout=row_major>

pto.textract
  ins(%src, %r0, %c0, %r1, %c1 :
      !pto.tile_buf<vec, 64x128xf16,
                    blayout=row_major, slayout=none_box>,
      index, index, index, index)
  outs(%dst0, %dst1 :
       !pto.tile_buf<vec, 32x64xf16, valid=32x64,
                     blayout=col_major, slayout=row_major>,
       !pto.tile_buf<vec, 16x32xf16, valid=13x29,
                     blayout=col_major, slayout=row_major>)
```

该例刻意使用不同的 destination shape，以固定“两路不是同型数组”的接口语义。

### 4.4 builder、accessor 与 PTOBC 兼容

ODS range 化会改变自动生成的 builder/accessor，不能把“文本兼容”误写成“生成 API 自动
兼容”。实现 PR 必须显式提供以下兼容层：

- C++ operand API 保留当前生成头文件的精确返回类型。`getSrc()`、`getFp()`、
  `getPreQuantScalar()` 及对应 mutable accessor 由未改名的字段继续生成；range 化后消失的三个
  getter 和 mutable accessor 由 legacy wrapper 补回。完整兼容集合为：

  ```cpp
  ::mlir::TypedValue<::mlir::Type> getSrc();
  ::mlir::TypedValue<::mlir::IndexType> getIndexRow();
  ::mlir::TypedValue<::mlir::IndexType> getIndexCol();
  ::mlir::TypedValue<::mlir::Type> getDst();
  ::mlir::TypedValue<::mlir::Type> getFp();
  ::mlir::TypedValue<::mlir::IntegerType> getPreQuantScalar();

  ::mlir::OpOperand &getSrcMutable();
  ::mlir::OpOperand &getIndexRowMutable();
  ::mlir::OpOperand &getIndexColMutable();
  ::mlir::OpOperand &getDstMutable();
  ::mlir::MutableOperandRange getFpMutable();
  ::mlir::MutableOperandRange getPreQuantScalarMutable();
  ::mlir::MutableOperandRange getDpsInitsMutable();
  ```

  `getIndexRow()`、`getIndexCol()`、`getDst()` 及其 mutable wrapper 只能在
  `SingleOutput` form 中调用；它们不能退化为返回 `mlir::Value`，否则显式接收
  `TypedValue<IndexType>`/`TypedValue<Type>` 的既有源码会编译失败。
- C++ 保留当前全部 typed `build`/`create` overload。以下签名中的参数顺序、类型、
  `TypeRange` 变体和 `ReluPreMode::NoRelu` 默认值都是兼容契约；实现内部把
  `indexRow,indexCol` 组装为 `indices={...}`，把 `dst` 组装为 `dsts={...}`：

  ```cpp
  static void build(
      ::mlir::OpBuilder &, ::mlir::OperationState &, ::mlir::Value src,
      ::mlir::Value indexRow, ::mlir::Value indexCol, ::mlir::Value dst,
      ::mlir::Value fp, ::mlir::Value preQuantScalar,
      ::mlir::pto::AccToVecModeAttr accToVecMode,
      ::mlir::pto::ReluPreModeAttr reluPreMode);
  static TExtractOp create(
      ::mlir::OpBuilder &, ::mlir::Location, ::mlir::Value src,
      ::mlir::Value indexRow, ::mlir::Value indexCol, ::mlir::Value dst,
      ::mlir::Value fp, ::mlir::Value preQuantScalar,
      ::mlir::pto::AccToVecModeAttr accToVecMode,
      ::mlir::pto::ReluPreModeAttr reluPreMode);
  static TExtractOp create(
      ::mlir::ImplicitLocOpBuilder &, ::mlir::Value src,
      ::mlir::Value indexRow, ::mlir::Value indexCol, ::mlir::Value dst,
      ::mlir::Value fp, ::mlir::Value preQuantScalar,
      ::mlir::pto::AccToVecModeAttr accToVecMode,
      ::mlir::pto::ReluPreModeAttr reluPreMode);

  static void build(
      ::mlir::OpBuilder &, ::mlir::OperationState &, ::mlir::TypeRange resultTypes,
      ::mlir::Value src, ::mlir::Value indexRow, ::mlir::Value indexCol,
      ::mlir::Value dst, ::mlir::Value fp, ::mlir::Value preQuantScalar,
      ::mlir::pto::AccToVecModeAttr accToVecMode,
      ::mlir::pto::ReluPreModeAttr reluPreMode);
  static TExtractOp create(
      ::mlir::OpBuilder &, ::mlir::Location, ::mlir::TypeRange resultTypes,
      ::mlir::Value src, ::mlir::Value indexRow, ::mlir::Value indexCol,
      ::mlir::Value dst, ::mlir::Value fp, ::mlir::Value preQuantScalar,
      ::mlir::pto::AccToVecModeAttr accToVecMode,
      ::mlir::pto::ReluPreModeAttr reluPreMode);
  static TExtractOp create(
      ::mlir::ImplicitLocOpBuilder &, ::mlir::TypeRange resultTypes,
      ::mlir::Value src, ::mlir::Value indexRow, ::mlir::Value indexCol,
      ::mlir::Value dst, ::mlir::Value fp, ::mlir::Value preQuantScalar,
      ::mlir::pto::AccToVecModeAttr accToVecMode,
      ::mlir::pto::ReluPreModeAttr reluPreMode);

  static void build(
      ::mlir::OpBuilder &, ::mlir::OperationState &, ::mlir::Value src,
      ::mlir::Value indexRow, ::mlir::Value indexCol, ::mlir::Value dst,
      ::mlir::Value fp, ::mlir::Value preQuantScalar,
      ::mlir::pto::AccToVecModeAttr accToVecMode,
      ::mlir::pto::ReluPreMode reluPreMode =
          ::mlir::pto::ReluPreMode::NoRelu);
  static TExtractOp create(
      ::mlir::OpBuilder &, ::mlir::Location, ::mlir::Value src,
      ::mlir::Value indexRow, ::mlir::Value indexCol, ::mlir::Value dst,
      ::mlir::Value fp, ::mlir::Value preQuantScalar,
      ::mlir::pto::AccToVecModeAttr accToVecMode,
      ::mlir::pto::ReluPreMode reluPreMode =
          ::mlir::pto::ReluPreMode::NoRelu);
  static TExtractOp create(
      ::mlir::ImplicitLocOpBuilder &, ::mlir::Value src,
      ::mlir::Value indexRow, ::mlir::Value indexCol, ::mlir::Value dst,
      ::mlir::Value fp, ::mlir::Value preQuantScalar,
      ::mlir::pto::AccToVecModeAttr accToVecMode,
      ::mlir::pto::ReluPreMode reluPreMode =
          ::mlir::pto::ReluPreMode::NoRelu);

  static void build(
      ::mlir::OpBuilder &, ::mlir::OperationState &, ::mlir::TypeRange resultTypes,
      ::mlir::Value src, ::mlir::Value indexRow, ::mlir::Value indexCol,
      ::mlir::Value dst, ::mlir::Value fp, ::mlir::Value preQuantScalar,
      ::mlir::pto::AccToVecModeAttr accToVecMode,
      ::mlir::pto::ReluPreMode reluPreMode =
          ::mlir::pto::ReluPreMode::NoRelu);
  static TExtractOp create(
      ::mlir::OpBuilder &, ::mlir::Location, ::mlir::TypeRange resultTypes,
      ::mlir::Value src, ::mlir::Value indexRow, ::mlir::Value indexCol,
      ::mlir::Value dst, ::mlir::Value fp, ::mlir::Value preQuantScalar,
      ::mlir::pto::AccToVecModeAttr accToVecMode,
      ::mlir::pto::ReluPreMode reluPreMode =
          ::mlir::pto::ReluPreMode::NoRelu);
  static TExtractOp create(
      ::mlir::ImplicitLocOpBuilder &, ::mlir::TypeRange resultTypes,
      ::mlir::Value src, ::mlir::Value indexRow, ::mlir::Value indexCol,
      ::mlir::Value dst, ::mlir::Value fp, ::mlir::Value preQuantScalar,
      ::mlir::pto::AccToVecModeAttr accToVecMode,
      ::mlir::pto::ReluPreMode reluPreMode =
          ::mlir::pto::ReluPreMode::NoRelu);
  ```
- 当前两组 generic overload 也保留：

  ```cpp
  static void build(
      ::mlir::OpBuilder &, ::mlir::OperationState &, ::mlir::TypeRange,
      ::mlir::ValueRange, ::llvm::ArrayRef<::mlir::NamedAttribute> = {});
  static TExtractOp create(
      ::mlir::OpBuilder &, ::mlir::Location, ::mlir::TypeRange,
      ::mlir::ValueRange, ::llvm::ArrayRef<::mlir::NamedAttribute> = {});
  static TExtractOp create(
      ::mlir::ImplicitLocOpBuilder &, ::mlir::TypeRange,
      ::mlir::ValueRange, ::llvm::ArrayRef<::mlir::NamedAttribute> = {});

  static void build(
      ::mlir::OpBuilder &, ::mlir::OperationState &, ::mlir::TypeRange,
      ::mlir::ValueRange, const Properties &,
      ::llvm::ArrayRef<::mlir::NamedAttribute> = {});
  static TExtractOp create(
      ::mlir::OpBuilder &, ::mlir::Location, ::mlir::TypeRange,
      ::mlir::ValueRange, const Properties &,
      ::llvm::ArrayRef<::mlir::NamedAttribute> = {});
  static TExtractOp create(
      ::mlir::ImplicitLocOpBuilder &, ::mlir::TypeRange,
      ::mlir::ValueRange, const Properties &,
      ::llvm::ArrayRef<::mlir::NamedAttribute> = {});
  ```

  这里的函数 surface 保持不变不等于 generated internals 完全源码兼容：
  `Properties::operandSegmentSizes` 从 `std::array<int32_t, 6>` 变为
  `std::array<int32_t, 5>`，直接初始化/访问该字段的源码必须迁移；`getODSOperands()`、index
  constants 和 adaptor 的 fixed-field accessor 也不属于兼容承诺。`ReleaseNotes.md` 必须明确这条
  边界以及 generic assembly 的五段 schema。
- C++ 双输出调用同一 class 的 generated range builder，把四项 index 和两个 destination 分别
  作为 `ValueRange` 传入；Python 提供命名 facade builder
  `build_nd_to_2xnz(src, row0, col0, row1, col1, dst0, dst1)`。二者都不是新 op。
  range-aware verifier、effects、planning 和 lowering 使用 `getIndices()`/`getDsts()`，不得只取
  `.front()`。
- Python 兼容层必须覆盖 generated free-function builder 和 generated OpView properties，不能只
  测试 `TExtractOp` 构造器。`python/pto/dialects/pto.py` 在
  `_export_generated_symbols()` 之后保存 `_GeneratedTExtractOp`，再导出同名 facade 和
  legacy free-function wrapper：

  ```python
  class TExtractOp(_GeneratedTExtractOp):
      # Accept the old positional/keyword constructor and delegate to the
      # generated range constructor with indices=[row, col], dsts=[dst].
      ...

      @property
      def indexRow(self):
          return self.indices[0]

      @property
      def indexCol(self):
          return self.indices[1]

      @property
      def dst(self):
          return self.dsts[0]

      @classmethod
      def build_nd_to_2xnz(cls, src, row0, col0, row1, col1, dst0, dst1,
                           *, loc=None, ip=None):
          return cls(
              src=src, indices=[row0, col0, row1, col1], dsts=[dst0, dst1],
              loc=loc, ip=ip)

  def textract(src, index_row, index_col, dst, *, fp=None,
               pre_quant_scalar=None, acc_to_vec_mode=None,
               relu_pre_mode=None, loc=None, ip=None):
      return TExtractOp(
          src, index_row, index_col, dst, fp=fp,
          preQuantScalar=pre_quant_scalar, accToVecMode=acc_to_vec_mode,
          reluPreMode=relu_pre_mode, loc=loc, ip=ip)
  ```

  省略号部分必须按现有 `PartitionViewOp`/`TScatterOp` facade 模式实现：facade constructor
  同时接受旧的四个 positional/keyword operands 和供 `build_nd_to_2xnz` 使用的
  `indices`/`dsts` ranges；不能把生成 class 或生成的 range-form `textract(src, indices, dsts, ...)`
  直接重新导出。旧的
  `pto.textract(src, row, col, dst, ...)` 位置参数和 `pto.textract(src=..., index_row=...,
  index_col=..., dst=..., ...)` 关键字参数都必须保留；`TExtractOp.indexRow`、`.indexCol`、
  `.dst` 继续分别映射到 `indices[0]`、`indices[1]`、`dsts[0]`。新双输出只通过同一
  `TExtractOp` facade 的 `build_nd_to_2xnz` factory 构造，不能暴露 `TExtractNd2xNzOp` class。

`AttrSizedOperandSegments` 的 segment schema 会从六个 fixed/optional 字段变成
`[src, indices, dsts, fp, preQuantScalar]`。普通文本由 custom parser 重建新 schema；MLIR
generic form 测试必须使用新 schema。PTOBC v0 已发布的 `pto.textract` fixed-width wire opcode
必须保持如下兼容策略：

- 旧四 operand 单输出和旧五 operand FP record 的 opcode、operand 顺序与解码结果不变；decoder
  为解出的单输出 op 生成新的 segment sizes。
- 双输出七 operand form 在 `shouldEncodeViaGenericV0CompatibilityShim()` 中强制走 generic v0
  record，不能复用四/五 operand opcode，也不能改变旧 opcode 的 operand count。
- 增加旧 `.ptobc` fixture decode、单输出/FP encode-decode 和双输出 generic round-trip；未证明
  这些测试通过前，不能宣称 bytecode 兼容。

### 4.5 为什么复用 `TExtractOp`

PTO-ISA 对外提供的是同名 `TEXTRACT` overload，现有 PTOAS `TExtractOp` 也已经通过 operand、
attribute 和 layout 承载 base、FP、preQuant、relu、acc-to-vec 等多种形态。ND-to-2xNZ 的
`2 indices + 1 dst` 与 `4 indices + 2 dsts` 组合可唯一分类，再由 ND/NZ layout 唯一确认；
增加 dotted mnemonic 只会制造一套与 ISA 不一致的 public surface。

复用的代价是 custom parser/printer、兼容 builder/accessor 和 bytecode shim，但这些成本都能以
明确测试封闭。verifier 和 lowering 必须先调用统一 form classifier，再进入现有单输出或新增
双输出 helper，避免把新增规则散落进旧分支。被拒绝的方案包括 `pto.textract.nd2xnz` 和
`pto.textract2`；二者都不提供额外语义信息，也会迫使 TileLib、manual 和 Python binding 暴露
不必要的新 op 名。

## 5. Verifier 设计

### 5.1 公共校验顺序

不能承诺 `TExtractOp::verify()` 在所有 generated accessor 之前执行。LLVM/MLIR 19 的真实验证顺序是：

1. parser/`setPropertiesFromParsedAttr()`/`setPropertiesFromAttr()` 把文本 property 转换为固定长度
   `Properties::operandSegmentSizes`；长度错误在此失败。
2. `OpDefinition` 先执行 traits，其中 `OpInvariants` 调用 TableGen 生成的
   `verifyInvariantsImpl()`；该函数已经通过 `getODSOperands()` 做各段类型检查。
3. traits 成功后才调用用户定义的 `TExtractOp::verify()`。

本设计不为统一错误文案新增一个排在 `OpInvariants` 前的 structural trait。因而 malformed generic
IR 可能先得到 property conversion、`AttrSizedOperandSegments` 或 ODS segment/type diagnostic；
这些都属于稳定失败。实现承诺的是解析/验证不崩溃、classifier 和 custom interfaces 对
`Invalid` fail-safe，以及通过 generated invariants 的非法五段 schema 由 custom verifier 报出
actual/expected segments，而不是所有非法输入共享同一诊断顺序和文案。

`TExtractOp::verify()` 自身的第一步仍调用第 4.2 节的 property-schema validator。在它返回
`SingleOutput` 或 `NdTo2xNz` 前，custom verifier 不得调用依赖 segment offset 的 generated
single-value accessor。单输出随后调用语义不变的现有 verifier helper；双输出调用新的
`verifyNdTo2xNzForm()`，负责以下结构和硬件共同契约中的 1-10 项，诊断中必须带 `src`、
`dst0` 或 `dst1` 名称。frontend lowering 完成后，backend-boundary validation 再按同一顺序
执行 11-13 项；这些项不能被误解为依赖 planner 的 late check：

1. 对三个 tile 调用 `verifyTileBufCommon`；A2/A3 禁止 low precision，A5 允许。
2. 要求三个 operand 都是 rank-2 `!pto.tile_buf`。
3. 要求四个 index 都是 `index` type 和可折叠常量；拒绝负数和大于 `UINT16_MAX` 的值。
4. 把 raw physical/valid shape 归一化为 PTO-ISA Tile dimension；FP4 使用第 3.5 节规则。
   RowPlusOne 物理布局支持条件未满足前直接拒绝，不能只计算 emitted dimension 后继续。
5. 要求 `src loc=vec`、ND layout；两个 dst 都是 `loc=vec`、NZ layout，fractal size 为
   512 bits。
6. 要求 `srcElem == dst0Elem == dst1Elem`。
7. 按 target arch 校验 dtype 和 compact mode；A5 partial-valid plain NZ 额外要求
   `physicalRows == align16(validRows)`，block stride 固定为 `physicalRows`；不满足时在
   backend-boundary gate 失败。A5 RowPlusOne 无论 valid/physical 是否相等，都必须先通过第 5.4 节
   physical-layout 支持条件；首版明确拒绝。
8. 校验 source row-stride bytes 32B 对齐、每个 dst 的 plain-NZ logical physical rows
   16 对齐、emitted physical cols c0 对齐。
9. 首版要求 source 和两个 dst 的 valid shape 静态且非零；确保归一化后的 valid extent 不超过
   `UINT16_MAX`，并检查每个 valid extent 不大于对应 physical extent。source 没有可证明的
   valid shape 时，只接受明确 `valid == physical` 的 full-valid source，不能默认为 physical
   shape。
10. 对每个 window 独立执行 constant bounds 校验，同时检查 source physical 和 source valid
    两组上界；valid 上界失败时必须报告“window reads source undefined padding”，即使 physical
    上界仍然通过。
11. generic `mlir::verify` 成功后，driver 在 planning/sync pass manager 运行前直接调用
    `validateTExtractNd2xNzInputProvenance(module)`；不新增 provenance pass。helper 对 `src`、
    `dst0`、`dst1` 递归穿过 `subview`、`bitcast`、`treshape` 和单输入
    `unrealized_conversion_cast`。首版只接受最终 root 为 `AllocTileOp`，或 non-RowPlusOne
    operand 确实来自 `AllocMultiTileOp` 的 `MultiTileGetOp`。任一 RowPlusOne destination 的 root
    为 `AllocMultiTileOp`/`MultiTileGetOp` 时按第 5.4 节稳定拒绝；`DeclareTileOp`、`TAssignOp`、
    frontend pop、block argument 和未知 root 都按 runtime-bound/unknown provenance 拒绝。
12. runtime-bound provenance 的固定诊断为：

    ```text
    pto.textract ND-to-2xNZ form does not support runtime-bound tile provenance for
    src|dst0|dst1; use alloc_tile with planner-owned or statically known level3 address
    ```

    诊断必须指出 operand 名称和命中的 root op 名。通过 helper 的 level1/level2 operand 才进入
    现有 planner；level3 还必须满足第 7.1 节的规划后静态 range 检查。
13. legacy/modern PlanMemory 保持现有 semantic conflict；planning/sync pipeline 已运行到
    `PTOResolveBufferSelect`、但尚未进入 `PTOInlineBackendHelpersPass` 时，driver 由统一
    post-planning helper 要求三个 operand 均可解析为静态 absolute byte range并复核三组 pair。
    未知 range 直接拒绝，不把“未知”当成“不重叠”。

### 5.2 dtype helper

不要复用当前 `isA2A3ExtractElemType`，因为它没有 `i32`，而本 overload 的 A2/A3
PTO-ISA 明确支持 `int32_t`。新增窄范围 helper：

```text
isA2A3Nd2xNzElemType          = i8 | i32 | f16 | bf16 | f32
isA5LowpCandidateNd2xNzElemType = A2A3 set | hif8 | f8 variants
isA5AllCandidateNd2xNzElemType  = lowp candidate set | packed f4 variants
```

helper 只服务该 overload，不静默扩大单输出 `TExtractOp` 的合法集合。candidate 集合只用于
诊断和测试路由；verifier 的 enabled 集合由已验证的目标路径决定。hif8/fp8 在上游 ST 中只覆盖
aligned window，因此实现 PR 还必须补至少一个 1-byte low-precision sub-c0 window 的设备
golden，才能无条件放行动态/非对齐 index。FP4 未通过第 3.5 节门槛前不能进入正向路径。

### 5.3 shape 与动态值

- physical shape 必须能生成合法的静态 PTO-ISA Tile template；动态 physical shape 直接拒绝。
- 首版要求 destination valid shape 静态。PTO-ISA 会把 `GetValidRow/GetValidCol()` 窄化为
  `uint16_t`，A5 vector path 还会直接计算 `validRow - 1`；其现有 runtime bounds assert 不拒绝
  0。仅依靠 physical upper bound 无法证明动态值非零，因此不能无保护地沿用 PTOAS dynamic
  valid 机制。后续若要放开，必须先在调用前生成 `0 < valid <= physical` 的 runtime guard，
  并增加动态 0、上界和越界测试。
- 首版要求 index 可折叠为静态常量，并在 PTO IR 阶段完成范围和 bounds 检查。PTO-ISA 的
  C++ 形参是 `uint16_t`，而 A5 TileLib/VPTO 路径会绕过 EmitC；仅在 EmitC 插 guard 会造成
  后端语义分叉。后续只有在两个 backend 共用的 runtime-check 机制落地后才开放动态 index，
  且必须在窄化前检查 `[0, UINT16_MAX]` 和 window bounds。
- `validRows` 和 `validCols` 必须大于 0；空 window 不是已定义的 no-op。
- A5 的 aligned TileLib path 还要证明 plain `block_stride`、`repeat_stride` 落在 16-bit
  hardware control field 内。RowPlusOne 变体在首版 template 中直接拒绝；物理布局支持完成后才能从
  shared helper 取得 `physicalRows + 1`，不能把 `align16(validRows) + 1` 当成独立 stride 来源。
- static bounds 的加法使用 checked arithmetic，避免超大常量溢出后误判为合法。

首版的完整 TSTORE eligibility 不由 `TExtractOp::verify()` 假定，也不新增 StoreUse pass。

#### 5.3.1 post-planning helper 执行切点

当前 `compilePTOASModule()` 把 `PTOResolveBufferSelectPass` 和
`PTOInlineBackendHelpersPass` 追加到同一个 `PassManager`；在 `--emit-pto-ir`、VPTO 和 EmitC
三条执行路径上各自只有一次 `pm.run()`，因此不存在
“main pass manager 完成后、但 call inlining 前”的可执行时点。实现必须把当前 main pipeline
拆成两个实际运行的 pass manager，而不是在单次 `pm.run()` 后再调用 helper：

1. `preInlinePlanningPM` 保留现有 generic verification 之后的 planning、sync 和地址物化顺序，
   运行到 `PTOResolveBufferSelectPass`；EmitC 当前紧随其后的第一次
   `NarrowUnusedMultiResultProvenancePass` 也可留在该 PM，因为它不改写 call graph。
2. `preInlinePlanningPM.run(module)` 成功后，driver 立即调用普通函数
   `validateTExtractNd2xNzPostPlanningSafety(module)`。此时 planner 地址和 multi-buffer slot 已经
   materialize，所有尚未 inline 的 `func.call` definition edge 仍可用于构造 component。
3. `--emit-pto-ir` 在同一个校验点通过后打印当前 IR 并返回；它不能走一条跳过 helper、提前打印，
   或使用不同 component 边界的专用路径。
4. 普通 EmitC/VPTO codegen 只有通过 helper 后才运行 `postValidationPM`。该 PM 包含当前后半段的
   CSE、`PTOInlineBackendHelpersPass`、inline 后的 provenance narrowing、canonicalizer 和 CSE，
   然后再进入最终 backend lowering。
5. 两个 PM 都启用 verifier，并分别调用 `applyConfiguredPassManagerCLOptions`，使用可区分的 pipeline
   label；现有 pass-manager CLI 的打印、统计和验证语义必须保留。不得通过 CLI pipeline 配置或
   `emitMlirIR` 分支绕过位于两次 `run()` 之间的普通 driver helper。

该切点同时满足“post-planning”与“pre-inlining”。不采用 planning 前保存 call graph 的替代方案，
因为旧 graph 还必须在地址物化后重新关联 physical range，容易与实际被检查的 IR 脱节。helper
执行以下检查：

1. level1/level2 使用 planner 物化地址，level3 使用调用方显式地址；multi-buffer slot 已经由
   `PTOResolveBufferSelect` materialize。range 使用第 5.4 节带 address space 的 physical access
   envelope，不能使用 valid shape，也不能把 allocation reservation、实际 payload byte count 和
   access end offset 当成同一个量。
2. helper 要求 `src`、`dst0`、`dst1` 都能解析到 non-negative static absolute range 和已知
   address space，并复核三组 pairwise no-alias；常量 cast 和纯常量 `arith.addi` 可折叠，动态
   root、未知 address space 或无法唯一确定 range 一律拒绝。
3. helper 必须在任何 `func.call` lowering/inlining 之前解析同一 symbol table 中的 direct internal
   `func.call`，以函数为节点构造 weakly connected components。对每个含 partial-valid
   ND-to-2xNZ destination 的 component，先汇总所有 producer 的 physical ranges，再扫描该
   component 内所有函数的 `TStoreOp.src`；不能只在 producer 所属函数内找 TSTORE，也不能因
   call site 没有传递 tile operand 就跳过 caller/callee edge。
4. 任一 TSTORE source 与任一 partial destination 在同一 address space 中的 static range 相交即
   拒绝；不同 address space 即使数字地址相同也不构成该 alias。TSTORE source 是 block
   argument/call operand 派生值、address space 未知或无法解析为唯一 static absolute range 时也
   保守拒绝；首版不为此建立 argument-effect/range summary fixed point。
5. helper 先判断 compile unit 是否含任一 partial producer。若有，则 call-surface closure 是
   compile-unit-wide 条件：扫描所有函数中的 call-like op，`func.call` 必须唯一解析到本 compile
   unit 中带 body 的 internal definition；任意 `func.call_indirect`、callee declaration/external
   function、无法解析的 direct callee symbol，或除已解析 internal `func.call` 之外的
   `CallOpInterface` 都拒绝。opaque call 等价于一条可能连接调用者与任意 function component 的
   hyper-edge；首版不做 function-pointer target/signature/address-taken 分析，因此不能只检查
   opaque site 当前所属的 direct component。诊断必须同时指出至少一个 partial producer function、
   opaque call site/callee、二者当前 direct component（可相同或不同）及无法证明 closed 的原因。
6. 只有第 5 项的 module-wide closure 成功后，component 之间才不比较 physical ranges；两个没有
   调用关系的独立 entry/kernel 可以合法复用相同数值的 UB 地址。component 内则与操作文本顺序、
   caller/callee 方向、CFG、dominance 和 full overwrite 无关：即使 TSTORE 位于调用前、producer
   位于 callee、TSTORE 位于 caller，或中间存在完整覆盖，只要 static range 相交仍拒绝。compile
   unit 没有 partial producer 时，本 helper 不因 opaque call 单独失败。
7. alias 诊断必须同时指出 producer function、TSTORE function、address space 和两个 half-open
   physical ranges：

   ```text
   pto.tstore source physical range aliases a partial-valid ND-to-2xNZ destination
   in the same address space and call component; undefined NZ padding cannot be stored
   ```

8. 不提供 test-only flag、module attribute 或其他 escape hatch。

这样 partial-valid op 本身仍可用于不含 generic TSTORE 的 UB-only 测试，不会在 debug PTO-ISA
上晚期触发 TSTORE assertion，也不能通过复制一个同址 full-valid allocation 绕过限制。

#### 5.3.2 backend-partitioned module boundary

`post-planning safety helper` 只在 child compile unit 内运行仍然不充分。mixed-backend driver
会先把 outer module 拆成独立 child compile unit，再把跨 child `func.call` 的 callee 克隆成
declaration；此后每个 child 分别调用 `compilePTOASModule()`。这不仅会使 producer 在 child A、
caller/TSTORE 在 child B 时丢失完整 partial/TSTORE component，也会使 child A 中独立的 partial
producer 与另一个 full-valid `caller -> callee` 跨 child 调用同时存在时，把原本已由 outer 解析的
callee 再次表现为 child 内无 body declaration。child helper 因而不能把该 declaration 当成已经
可信闭合的 internal call。

首版选择在拆分前增加一个普通 driver precheck，而不是传递跨 child range/effect summary：

1. 在任何 backend/output 路由、`isBackendPartitionedContainer()` 判断、single-child
   normalization、`collectSharedPipelineFunctions()` 或 user-visible IR output 之前，driver 先递归
   检测 module 是否包含 ND-to-2xNZ form 以及 descendant `ModuleOp`。只要两者同时存在，就执行
   fixed-depth structure guard；这也覆盖根 module 同时含一个 backend child 和顶层 `func.func` 的
   形状，即使基线 `isBackendPartitionedContainer()` 会因顶层非 ModuleOp 而返回 false。合法形状是
   根 module body 只含 immediate backend child `ModuleOp`，每个 child 下不得再出现 descendant
   `ModuleOp`，且每个 `func.func` 的直接 parent 必须就是该 child。违反任一条件都拒绝，诊断指出
   offending op、最近的 immediate child 和“backend-partitioned ND-to-2xNZ validation does not
   support nested module/function scope”。首版不递归编译、提升或猜测 nested symbol ownership；
   没有 ND-to-2xNZ form 的既有 nested 输入不受该 feature guard 影响。递归 walk 只用于发现并拒绝
   invalid nesting，不得把 nested function 纳入 owner、symbol resolution、call graph 或 compile
   coverage。
2. 通过 structure guard 后，在 `collectChildJobs()`/`buildBackendChildCompileUnit()` 之前，仅收集
   immediate child 直接拥有的 function definitions 建立 direct `func.call` graph；每个 definition
   的 owner 就是其直接 parent child，不能递归收集。也不能用
   `SymbolTable::lookupNearestSymbolFrom()` 的普通结果直接建图：caller child 中同名的 private
   declaration 会先被命中。
3. 每个 direct call 按最终链接符号规则解析。local lookup 命中带 body 的 definition 时连接该
   definition；local lookup 缺失或只命中 declaration 时，declaration 不是 graph node 或 terminal，
   precheck 必须扫描 immediate sibling children 的 direct functions，忽略 private symbol 和无 body
   declaration，并按 exact callee symbol name 查找唯一 public/non-private definition。当前 direct
   `func.call` child assembly 不使用 logical-name fallback；该解析应与 `findSiblingSourceFunction(...,
   allowLogicalNameMatch=false)` 抽取或共享同一 helper，避免 precheck 与最终 child assembly 漂移。
   后续若 driver 改变最终链接名规则，两处必须同步修改。
4. sibling definition 恰有一个时，call edge 直接连接到该 definition，即使 caller child 已有同名
   private declaration、后续 child assembly 因 declaration 已存在而不再 clone。零个匹配记录为
   unresolved；多个 exact public definitions 记录为 ambiguous。完整 outer module 含任一 partial
   producer 时，第 5.3.1 节的 call-surface closure 同样作用于所有 immediate child functions：任一
   child 中的 unresolved/ambiguous direct call、`func.call_indirect` 或其他 opaque
   `CallOpInterface` 都在拆分前拒绝，即使 opaque site 与 producer 位于不同 direct component 或
   不同 child。诊断列出 producer、caller child/function、local declaration（如有）、callee symbol、
   所有 sibling candidate 及 unresolved/ambiguous/opaque 原因。
5. `pto.import_reserved_buffer` 不实现 `CallOpInterface`，而且会在 post-planning helper 之前由
   `PTOResolveReservedBuffersPass` 替换为 peer reserve 的同一静态地址并删除。因此只要完整 outer
   module 或 single compile unit 含任一 partial producer，就必须在 planning/reserved-buffer
   resolution 前拒绝所有 `ImportReservedBufferOp`，无论 import 与 producer 位于同一 child、不同
   child 或不同 direct-call component。对每个 immediate child 直接拥有的 `func.func` 内的 import，按
   `findSiblingSourceFunction(..., allowLogicalNameMatch=true, referenceKind="peer_func reference")`
   的 exact-symbol 优先、logical-name 唯一回退规则解析 `peer_func`；private、无 body、零匹配或多匹配
   都是 unresolved/ambiguous；解析只用于稳定诊断和证明与现有 driver peer lookup 一致，不能成为
   partial 场景的放行证据。outer/single compile unit 不含 partial producer 时，peer link 保持现有
   driver 解析、clone 和 address materialization 语义。
6. 用解析后的 direct-call definition edge 收集 weakly connected components。只要完整 outer module 含任一
   partial-valid ND-to-2xNZ producer，任何 direct edge 的 caller/callee definition owner 不同就
   直接拒绝整个 module；不以 edge 所在 component 是否含 partial producer 为条件。因此 child A
   有独立 partial producer、child B 的 full-valid caller 调用 child C 的 full-valid callee 也拒绝；
   任意 child 中的 peer import 已由第 5 项更早、独立地拒绝。
   此规则同样适用于相同 backend 但被拆成不同 child 的情况。它比只拒绝跨 child partial component
   更保守，但保证所有被允许进入 child compilation 的 call 都有当前 compile unit 内带 body 的
   definition，且不会在 child 中留下跨 child declaration 或已删除的 peer import 供 post-planning
   helper 误判为 opaque/disconnected；child-level call-surface closure 无需信任 clone 前的外层结论
   或特别处理 declaration。
7. precheck 必须在任何 child clone、callee declaration materialization、peer clone 或 child-level
   `compilePTOASModule()` 之前失败，并报告至少一个 partial producer、caller/callee function 及其
   child/backend、link site 和“backend-partitioned module with partial-valid ND-to-2xNZ does not
   permit cross-child direct calls or peer links”。无论 link 是否传递 tile、是否与 producer 不连通、
   是否 full-valid，都不能由最终链接阶段、child declaration 或 reserved-buffer address materialization
   补救。
8. outer module 不含 partial producer 时，本 feature precheck 不限制既有跨 child direct call 或 peer
   link；其 clone/declaration/address-materialization 行为保持 driver 基线语义。outer module 含
   partial producer 且通过第 4 至 6 项时，
   每个 direct-call component 都完全位于一个 child，仍由该 child 的 post-planning helper 执行
   第 5.3 节的 component-wide range 检查；互不连通 child 可以复用相同数值地址，但仍须通过
   第 4 项的 outer call-surface closure 和第 5 项的 peer-link closure。
9. 该 precheck 是 driver 普通函数，不注册 MLIR pass，也不改变 `compilePTOASModule()` 的
   单 child 契约。后续若要在 outer module 含 partial producer 时支持任何跨 child call，必须先
   定义由 outer final-link resolution 产生、且由 child helper 消费和校验的可信 call-closure
   summary；若跨 child component 还可能传递 partial destination，则还需要 producer/TSTORE physical
   range、call argument provenance、backend ownership 和 opaque-call effects 的稳定 summary 协议。
   后续若要支持 nested child，还必须统一升级 `isBackendPartitionedContainer`、sibling symbol
   resolution、`collectChildJobs`、object compile 的 `collectSharedPipelineFunctions` 和递归 codegen，
   再删除本首版拒绝规则。

### 5.4 compact mode

- A2/A3：拒绝 `CompactMode::RowPlusOne`；`Null`/`Normal` 都按 plain NZ 处理。
- A5 PTO-ISA header/implementation 宣称允许 `Null`/`Normal` 和 `RowPlusOne`，但 PTOAS 首版对
  ND-to-2xNZ 的两个 destination 都拒绝 `RowPlusOne`。只有本节的全部物理布局条件同时满足后，才允许
  一边 plain NZ、另一边 NZ+1，或两边都使用 NZ+1；其中每个 RowPlusOne destination 都必须由
  单 `AllocTileOp` backing。

RowPlusOne `AllocMultiTileOp` 不属于本设计的解锁范围。当前 `AllocMultiTileOp::verify()` 已明确
拒绝 `row_plus_one` slot，因为 `PTOResolveBufferSelect` 和 InsertSync 使用同一个
`slotBytes`/`allocateSize` 同时表达 slot address stride 与 per-slot conflict/access range。对
`f16 16x32`，按旧 `product(shape)=1024` 放置第二个 slot 会覆盖第一个 slot 的
`[1024, 1056)` access tail；把该字段改成 1056 又不能表达本设计选择的 1088-byte reservation。
因此即使单 allocation 的 RowPlusOne 物理布局支持条件已经满足，任何 RowPlusOne `AllocMultiTileOp`、
`MultiTileGetOp` 或其 view chain 仍在 generic verification/input-provenance 阶段失败，不能进入
planner、sync 或 buffer-select materialization。

当前问题不是 `PTOResolveBufferSelect` 的 subview offset 错误。基线对 ColMajor NZ
`RowPlusOne` 已使用 `shape[0] + 1` 的 `colStride`；因此 A5 `f16 16x32` 的第二个 block
当前就应为 `17 * 16 = 272` elements。这个既有结果必须保留，shared helper 接管该路径时
不得再次加一。真正未统一的是 legacy/modern PlanMemory 和 InsertSync 仍把
二维 shape 的每个 column 当成 major slice，使用
`(major - 1) * (minor + 1) + minor` 得到 1086 bytes。ColMajor NZ 的 padding 位于相邻
`c0` column block 之间，不是每个 logical column 之间；1086 既不是合法 allocation extent，
也不是任何 PTO-ISA access envelope。

实现必须增加一个共享、checked 的 tile physical-layout/access helper。它从 `TileBufType` 的
logical physical shape、element storage width、B/S layout、fractal 和 compact mode 一次性返回：

- emitted physical rows/cols；
- subview 的 row/column element stride；
- NZ column-block count、stride 和每 block payload（elements）；
- exact payload intervals、`touchedPayloadElements/Bytes` 和
  `accessEndOffsetElements/Bytes`；
- `allocationElements/Bytes` 以及计算失败原因。首版对 RowPlusOne 使用完整矩形作为 allocation
  reservation；地址空间对 allocation 起点或下一 slot 的额外 alignment 由 planner 单独处理。

所有可用于 alias、hazard 或同步的 range 都必须携带地址空间，统一表示为：

```text
PhysicalRange = (addressSpace, [baseByte, endByte))
```

其中 `baseByte`/`endByte` 是同一 address space 内的绝对、半开 byte range。只有 address
space 相同且区间相交时才构成 alias 或同步 hazard；相同数字地址落在 `VEC`、`MAT`、
`ACC` 等不同空间时不冲突。地址空间未知、range 未解析或两者无法证明属于同一空间时，production
校验必须保守失败，不能把 unknown 当作“不重叠”。该字段是 range 的语义组成部分，不是调用方
在比较前可以丢弃的附加诊断信息。

对本 op 唯一待开放的 A5 ColMajor NZ `RowPlusOne`，helper 固定使用：

```text
storageRows          = physicalRows + 1
blockCount           = physicalCols / c0
nzBlockStrideElems   = storageRows * c0
blockPayloadElems    = physicalRows * c0
touchedPayloadElems  = blockCount * blockPayloadElems
accessEndOffsetElems = blockCount == 0
                         ? 0
                         : (blockCount - 1) * nzBlockStrideElems + blockPayloadElems
allocationElems      = storageRows * physicalCols
```

所有 element 值再乘 `storageElemBytes` 得到 byte 值。`touchedPayloadBytes` 是各 block 实际
payload 长度之和，不含 block 间 gap；`accessEndOffsetBytes` 是相对 base 的 exclusive end，
包含地址包络中的 gap。两者不能互换。完整矩形
`storageRows * physicalCols * storageElemBytes` 只可作为诊断中的
`rectangularEnvelopeBytes`，首版把它作为 allocation reservation；不得把它当作 semantic
range、sync size 或 TSTORE access size，因为最后一个 NZ block 后没有 trailing virtual row。
这与 A5 `TStoreVecNZ` 的实际参数一致：`nBurst=blockCount`、每 burst 从 UB 读取
`physicalRows * C0_SIZE_BYTE`，相邻 burst source start 相差
`storageRows * C0_SIZE_BYTE`；上游 RowPlusOne ST 也用
`(blockCount - 1) * blockStrideBytes + blockPayloadBytes` 作为 UB extent。

对该布局的 subview 物化，NZ column-block 使用 helper 返回的
`colStride = storageRows`；第二个 block 的 offset 是 `colStride * c0`。其他 view 维度的
`rowStride`/inner stride 也必须来自同一 layout record，不能继续从 `innerRows` 或 logical
shape 局部推导。对 `f16 16x32`，该记录必须保留基线已有的 `colStride=17`；`16` 只作为
回归检测值，不能被 helper 或调用方重新引入。

valid rows 仍是 logical rows，不因 padding 增大。所有乘加都用 checked arithmetic；dynamic、负数、
不合法 fractal/layout 或溢出返回 failure，不能回退到 product(shape) 或旧 implicit-gap 公式。

以下消费者必须直接调用同一 helper，不得复制公式或再次 `+1`；对已有正确的 subview stride，
helper 应返回并复用 `PTOResolveBufferSelect` 当前的 `shape[0] + 1` 结果，而不是在调用方
再次叠加 padding。以下映射只适用于单 `AllocTileOp` backing 的 RowPlusOne tile；不得把
`allocationBytes`/`accessEndOffsetBytes` 填回现有 multi-buffer 的单一 `slotBytes` 字段来绕过
上述 verifier 限制：

- `PTOResolveBufferSelect` 的 subview 地址物化使用 row/column stride 和 block start；
- legacy/modern PlanMemory 使用 `allocationBytes` reservation；
- semantic range 与 post-planning alias helper 使用带 `addressSpace` 的 exact intervals；现有单区间
  API 不能表达 block 间 gap 时，保守使用
  `(addressSpace, [base, base + accessEndOffsetBytes))`，不得扩大到完整矩形；
- InsertSync translator 使用 access envelope，即
  同一 `addressSpace` 中的 `accessEndOffsetBytes`/bits，而不是 allocation alignment 或 payload
  byte count；不同 address space 不生成跨空间 hazard；
- EmitC/TileLib 使用 emitted dimensions、block stride 和 payload，TSTORE access metadata 使用
  exact intervals 或 `accessEndOffsetBytes`。

固定反例为 A5 ColMajor NZ `f16 16x32`、`c0=16`：`storageRows=17`，现有 subview 的第二个
NZ block offset 必须保持 `17 * 16 = 272` elements（544 bytes），不能回退到 256，也不能因
helper 接管而重复加一。每个 block payload 是 `16 * 16 = 256` elements；exact byte intervals
是 `[0, 512)` 和 `[544, 1056)`，payload 总量为 1024 bytes，exclusive access end 是
`528 * 2 = 1056` bytes。PlanMemory 的 allocation reservation 是完整矩形
`17 * 32 * 2 = 1088` bytes；下一 allocation 不得早于 `base + 1088`（再应用既有 alignment）。
单区间 semantic range 是 `(addressSpace, [base, base + 1056))`，InsertSync 记录 8448 bits。1088
不能作为 TSTORE access range；1086 bytes 则连 block 边界都没有正确建模。

只有上述所有消费者、精确回归、VPTO/EmitC compile 和 A5 `TEXTRACT -> TSTORE` device golden
同时通过，后续实现才可以解除 ND-to-2xNZ `RowPlusOne` 的首版拒绝。
此前 verifier/backend-boundary gate 必须稳定拒绝，不得因为 `CompactMode::RowPlusOne` token 已输出
或单独修正 Tile type 就开放。该支持条件只控制单 allocation form，不覆盖
RowPlusOne multi-buffer；后者没有例外。

### 5.5 不增加的限制

以下输入在 PTO-ISA 有明确路径，PTOAS 不得拒绝：

- 两路 window 在 source 中互相重叠；
- 两个 destination 的 shape/valid shape 不相等；
- `indexCol` 不是 c0 或 32B 对齐；
- `validCol` 不是 c0 倍数；
- `validRows/validCols == 1`；
- A2/A3 `i8` odd validCol。

## 6. DPS、内存效应与同步

### 6.1 DPS

`getDpsInitsMutable()` 对单输出返回一个 destination、对双输出返回 `dst0` 和 `dst1`。现有
以下消费者已经按 range 迭代，设计上无需按 op 名新增特判，但必须增加双输出回归：

- legacy `PTOPlanMemory`；
- `PTOPlanMemoryModern`；
- `PTONormalizeUncoveredTileSections`；
- TileFusion liveness/region generation；
- `PTOMarkLastUse`。

TileFusion 当前只把白名单中的 elementwise/reduction op 视为可融合 compute，`pto.textract`
本身不是白名单成员。双输出 form 延续这个 hard boundary，不在本功能中引入 multi-output
fusion 策略，也不能因为 op 名与单输出相同而被错误加入单输出 fusion 路径。

### 6.2 MemoryEffects

```cpp
void TExtractOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>& effects) {
  Form form = classifyForm(); // Typed operandSegmentSizes property only.
  if (form == Form::Invalid) {
    for (OpOperand &operand : getOperation()->getOpOperands()) {
      if (!isPTODpsType(operand.get().getType()))
        continue;
      addEffect(effects, &operand, MemoryEffects::Read::get());
      addEffect(effects, &operand, MemoryEffects::Write::get());
    }
    return;
  }

  addEffect(effects, &getOperation()->getOpOperand(0), MemoryEffects::Read::get());
  if (auto fp = getFpMutable(); !fp.empty())
    addEffect(effects, &*fp.begin(), MemoryEffects::Read::get());
  for (OpOperand &dst : getDpsInitsMutable())
    addEffect(effects, &dst, MemoryEffects::Write::get());
}
```

`isPTODpsType` 在上面是共享 type predicate 的示意名，实际实现复用 PTO type utility。Invalid
fallback 必须覆盖所有 raw memory-carrying operand 的 Read+Write，不能悄悄忽略多出来的 source；
正常 schema 仍是 source Read、optional FP Read 和全部 DPS destinations Write。source 不声明
Write。A2/A3 EmitC 调用的 PTO-ISA odd-i8 vector 路径使用固定 tmp UB scratch，但不修改 source；
该内部 scratch 由 PTO-ISA 保留区管理，不作为 PTO IR operand。第 9.3 节的 A2/A3 VPTO scalar
lowering 不使用这块 scratch。既有单输出 FP read effect 必须保留。

### 6.3 pipe 与自动同步

`TExtractOp::getPipe()` 先分类 form：双输出 overload 的外部执行类别固定为 `PIPE_V`，单输出
继续按现有 source/destination address space 返回 `PIPE_MTE1`、`PIPE_FIX` 或 `PIPE_V`。双输出
固定 `PIPE_V` 的依据是：

- A5 主路径完全是 vector frontend；
- A2/A3 EmitC 调用的 PTO-ISA vector 路径使用 `vcopy`/`vconv`；其 scalar fallback 的
  `PIPE_V <-> PIPE_S` flag/wait 必须通过第 6.3.1 节的 `SyncMacroModel`/hidden-event reservation
  建模，不能作为同步规划外的隐藏副作用；
- A2/A3 VPTO 首版使用第 9.3 节的 scalar correctness lowering，并显式生成同样的
  `PIPE_V -> PIPE_S` 与 `PIPE_S -> PIPE_V` flag/wait。公共调用前后的依赖仍以 `PIPE_V`
  建模，因此 planning/sync 在 lowering 前看到的 pipe 与展开后的可观察边界一致。

InsertSync 必须从 effects 得到一个 read 和两个 write。至少覆盖：

```text
TLOAD(src) [PIPE_MTE2]
  -> TEXTRACT ND2XNZ [PIPE_V]
    -> TSTORE(dst0) [PIPE_MTE3]
    -> TSTORE(dst1) [PIPE_MTE3]
```

full-valid 测试既要证明 MTE2-to-V 依赖存在，也要证明两个 destination 的 V-to-MTE3 消费
都被看到；partial-valid UB-only 测试只验证 TEXTRACT 的 V-side producer 和两个 destination
的 liveness，不把 partial descriptor 直接传给 NZ TSTORE。两类测试都不能只检查第一个 DPS init。

#### 6.3.1 A2/A3 scalar 与 A5 `1x1` 的内部事件

A2/A3 的 scalar correctness lowering，以及 A5 的 `1x1` scalar path，都会在公开的 `PIPE_V`
边界内执行 `PIPE_S` load/store。它们使用的 V-to-S、S-to-V 事件不能作为 lowering 后临时插入的
裸 `EVENT_ID0`：如果事件没有进入同步规划和 event-id allocation，就可能与 compiler-generated
event 复用同一个 ID，导致错误 wait、数据竞争或死锁。

实现必须复用现有 `SyncMacroModel` 机制，在 event-id allocation 之前为双输出
`TExtractOp` 返回一个 macro model：

- model 至少包含一个对外的 `PIPE_V` phase 和一个实际 scalar implementation 的 `PIPE_S`
  phase；phase 的 def/use values 必须分别覆盖 source read、两个 destination write，且与
  `MemoryEffects` 的 read/write 集合一致；
- model 声明 `PIPE_V -> PIPE_S` 与 `PIPE_S -> PIPE_V` 两个 bidirectional hidden-event pair。
  PTO-ISA 若要求固定内部 event（当前示例为 `EVENT_ID0`），该 ID 只能作为 hidden event
  reservation 记录在 model 中，由现有 `SyncEventIdAllocation::SeedHiddenMacroEventIds`
  在分配前登记生命周期；不能在 lowering 中绕过 allocator 直接写入一个未登记的 literal；
- `LowerPTOToUBufOps` 只消费该 op 已登记的 hidden-event reservation（或 allocator 返回的
  per-op event mapping），再生成对应 set/wait。实现不得在 event allocation 完成后新造一组
  allocator 不知道的 V/S event。当前 pipeline 可保持“InsertSync/event allocation 先于 A2/A3
  scalar expansion”，也可以把 expansion 提前到 allocation 前，但二者必须由同一个 model
  覆盖，不能依赖 pass 的偶然执行顺序。

A5 非 `1x1` vector template 没有这组 scalar hidden event；它仍按普通 `PIPE_V` 依赖建模。该
 例外必须按 compact mode/shape 明确选择，不能只按 op 名为所有双输出 form 注入 V/S 事件。

当前 legacy planner、modern planner、semantic range、InsertSync 和 subview
materialization 各自维护相近但不等价的 tile layout/footprint 公式。实现必须把它们迁移到第 5.4 节
shared checked physical-layout/access helper；planner 读取 `allocationBytes` reservation，post-planning safety
helper 读取 exact intervals 或 `accessEndOffsetBytes` 构造 half-open access envelope，InsertSync
仅把 access end 转为 bits。各路径都要对 dynamic/negative shape、非法 layout 和算术
溢出返回 failure，不能为 plain/RowPlusOne 保留彼此漂移的局部公式，也不能把
`rectangularEnvelopeBytes` 误当实际 access size。这里的 RowPlusOne consumer 仅指单
`AllocTileOp`；`AllocMultiTileOp` 不迁移到该双尺寸模型。

未来若要支持 RowPlusOne multi-buffer，必须另行定义至少 `slotStrideBytes`/
`slotAllocationBytes` 与 `slotAccessEndBytes`（或 exact per-slot intervals）两套独立数据，并同步
升级 `PTOOps.td`/verifier、level3 contiguous-slot contract、legacy/modern PlanMemory、
`PTOResolveBufferSelect`、InsertSync `BaseMemInfo` 及 bytecode/文本兼容
策略，再删除拒绝规则。本设计不预留隐式推导或 fallback。

## 7. No-alias 与内存规划

当 classifier 判定 `TExtractOp` 为双输出 form 时，`getSemanticNoAliasPairs()` 返回：

```text
(src, dst0)
(src, dst1)
(dst0, dst1)
```

这三组 pair 同时进入：

- legacy planner 的 `RecordSemanticConflict`；
- modern planner 的 `addForbidAliasBetweenRoots`；
- `verifySemanticNoAliasRanges` 的显式/规划后 byte-range 校验。

不能只比较 SSA Value 是否相同。`subview`、`bitcast`、`treshape`、multi-buffer slot 和显式
地址可能以不同 Value 指向重叠范围，必须复用现有 semantic range 解析。

这里有三种目的不同的 range 消费：semantic no-alias verifier 证明同一次 TEXTRACT 的三个
operand 不重叠（比较键包含 address space）；InsertSync 为不同 pipe 的读写建立 hazard；driver
post-planning safety helper 防止 partial destination 的未定义 padding 被 alias TSTORE 导出。
相同数字地址但属于不同 address space 的 range 不互相冲突；unknown address space/range 则按
无法证明安全处理。前一项通过不代表后两项自动成立。

runtime-bound gate 是 no-alias 契约的前置条件，而不是 range resolver 的可选优化。当前
legacy planner 会跳过 `DeclareTileOp`，InsertSync 也只能把 declared tile 自身作为没有绝对
地址的 symbolic root；因此不能声称 planner 或 semantic range 已经证明三个 runtime-bound
tile 两两不重叠。首版对这类输入统一拒绝，避免 level1/2 静默跳过约束，也避免 level3 只看
`alloc_tile.addr` 而漏掉 declared/tpop provenance。

### 7.1 level3 显式地址规则

level1/level2 由 legacy/modern planner 产生地址并在规划后检查三组 semantic range；
level3 跳过 planner，`pto.alloc_tile.addr` 由调用方提供。当前 `SemanticRange` 对不同
allocation root 只有在双方都有 absolute address 时才比较，因此两个 allocation 使用同一
动态 `%base`（或由 `%base` 派生的同一动态地址）会被错误地视为“无法证明重叠”并放行。

首版选择保守、可执行的规则；它只接收已经通过 driver input provenance helper 的
allocation-backed operand：

- `preInlinePlanningPM` 运行到 `PTOResolveBufferSelect` 后、`postValidationPM` 的
  `PTOInlineBackendHelpersPass` 前，driver 直接调用 post-planning safety helper；对 level1/level2
  使用 planner 结果，对 level3 使用显式地址，不注册独立 address pass。
- 当 module 含 ND-to-2xNZ form 时，`src`、`dst0`、`dst1` 的最终 physical range 必须具有已知
  address space 和 non-negative static absolute begin。可折叠的常量 `arith.addi`/index/integer
  cast 表达式可以接受，含 block argument、函数参数或动态 `%base` 的地址一律拒绝；不能仅因
  数字 base 相同就忽略 address space。
- 诊断固定为：

  ```text
  pto.textract ND-to-2xNZ form requires statically known level3 addresses for
  semantic no-alias verification (src|dst0|dst1)
  ```

  并附 operand 名称及其 `alloc_tile` 定义位置。这样三种动态同址 pair
  (`src=dst0`、`src=dst1`、`dst0=dst1`) 以及 `%base + constant` 形成的同址 pair 都在
  PTOAS 阶段失败，而不是依赖 C++ 或 NPU 行为。
- 不在本 PR 中扩大通用 `SemanticRange`。后续若要支持 level3 动态地址，必须将 range
  扩展为“symbolic address root + constant offset”，解析 `arith.addi` 等保持同一 root，
  并对无法证明的不同 root 采用保守拒绝；通过 dedicated range tests 后才能删除本 gate。

因此 `DeclareTileOp`、`TAssignOp`、`TPopOp` 及其 subview 即使带有看似常量的 `addr` 也不能
绕过 gate；只有从 `AllocTileOp` 或 non-RowPlusOne materialized multi-tile slot 回溯出的常量地址
才能进入 semantic range overlap 检查。RowPlusOne multi-tile root 在此前已失败，不能靠
materialization 转成表面上的 `AllocTileOp` 后绕过 provenance gate。

post-planning helper 位于 `PTOResolveBufferSelect` 之后。non-RowPlusOne level3
`AllocMultiTileOp` 若在 materialization 后仍产生动态地址 select，按同一规则拒绝；RowPlusOne
multi-buffer 必须在 generic verification/input provenance 阶段更早失败。只有最终三个允许的
operand 都能解析到静态 non-negative absolute range 时才进入 overlap 校验。

两个 dst 的 liveness 从同一 op 开始，planner 必须分别保留到各自最后一次消费。测试使用
不同大小和不同最后消费点，固定不能因只读取 `getDpsInits().front()` 而提前复用第二路内存。

现有 legacy/modern planner 和 sync translator 的 `RowPlusOne` footprint 公式没有
按 NZ block 建模，不能作为本 op 的既有正确基础；但
subview materialization 的 ColMajor NZ `colStride = shape[0] + 1` 与 272-element 第二 block
offset 已是正确基线。实现必须按第 5.4 节抽取并切换到同一个 shared physical-layout/access helper，
同时保留该 subview 结果且禁止调用方重复 `+1`；semantic range 和 post-planning alias helper
必须使用 access intervals/envelope，planner 则使用 allocation extent。该迁移只覆盖单
`AllocTileOp`；multi-buffer 拒绝保持不变。迁移需要有既有 RowPlusOne 非本 op 回归，防止修正
ND-to-2xNZ 时静默改变其他操作的布局语义。

增加 `dst0=plain`、`dst1=RowPlusOne` 的 planning、subview、semantic range、InsertSync、
EmitC 与 full-valid TSTORE 端到端测试，精确证明两路分别使用自己的 stride、payload intervals、
access end 和相邻 allocation 边界。所有链路完成前，该混合 case 保持 negative；partial-valid
RowPlusOne 同样保持 unsupported，不能计入 UB-only 正向覆盖。

## 8. EmitC lowering

不新增 conversion pattern。ODS 改为 `indices`/`dsts` 后，generated `OpAdaptor` 只提供
`getIndices()`/`getDsts()`，不会继承 `TExtractOp` 在 `extraClassDeclaration` 中添加的 legacy
wrapper。因此现有 `PTOExtractToEmitC` 的两条分支都必须从 adaptor ranges 取坐标和 destination；
保持不变的是单输出的生成语义，不是旧 accessor 调用：

```cpp
LogicalResult PTOExtractToEmitC::matchAndRewrite(
    pto::TExtractOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto form = op.classifyForm(); // Validates the typed property before getSrc().
  if (form == pto::TExtractOp::Form::Invalid)
    return rewriter.notifyMatchFailure(op, "malformed TEXTRACT operand segments");

  auto indices = adaptor.getIndices();
  auto dsts = adaptor.getDsts();
  Value src = adaptor.getSrc();

  if (form == pto::TExtractOp::Form::NdTo2xNz) {
    SmallVector<Value, 7> operands{
        dsts[0], dsts[1], src, indices[0], indices[1],
        indices[2], indices[3]};
    rewriter.replaceOpWithNewOp<emitc::CallOpaqueOp>(
        op, TypeRange{}, "TEXTRACT", nullptr, nullptr, operands);
    return success();
  }

  // The existing body is factored to accept range-derived core operands.
  return lowerSingleOutputTExtractForm(
      op, adaptor, src, dsts[0], indices[0], indices[1], rewriter);
}
```

`lowerSingleOutputTExtractForm` 可以继续从 adaptor 读取仍然存在的 optional
`getFp()`/`getPreQuantScalar()`，但其 `src/dst/indexRow/indexCol` 必须使用显式参数。实现中不能出现
`adaptor.getDst()`、`adaptor.getIndexRow()` 或 `adaptor.getIndexCol()`；这些方法在新 ODS 下不会生成。

具体 builder 参数按仓库当前 MLIR EmitC API 调整，但最终调用必须固定为：

```cpp
TEXTRACT(dst0, dst1, src, indexRow0, indexCol0, indexRow1, indexCol1);
```

双输出分支不生成 template argument，不生成 `TEXTRACT_ND2XNZ`，也不拆成两次单输出
`TEXTRACT`。
拆分会选择普通 Vec-to-Vec path，既不能表达 ND-to-NZ layout conversion，也不能保证与
双输出 overload 的 backend dispatch 一致。

该 pattern 继续使用主 conversion pattern set。EmitC 测试分别使用 A3/A5，并把两个 dst 的
类型和四个静态 index 设为可区分值，避免只检查 `TEXTRACT(` 而漏掉参数交换；同时保留
legacy 单输出、FP、pre-quant 和 acc-to-vec 的原有 FileCheck。

## 9. TileLib / VPTO 设计

### 9.1 注册

新增 `lib/TileOps/a5/textract_nd2xnz.py`，但挂到现有 `pto.textract` registry，不新增 op 名：

```text
("a5", "pto.textract") -> (
    ".a5.textract", ".a5.textract_fp", ".a5.textract_nd2xnz")
```

TileLib 选择器先按现有 `pto.textract` op 名加载全部模块，再按 operand arity/layout constraint
选择双输出 template。template 参数顺序与双输出 range 顺序一致：

```python
def template_textract_nd2xnz(
    src, index_row0, index_col0, index_row1, index_col1, dst0, dst1):
    ...
```

`dtypes` schema 包含一个 source、四个 `i32` scalar 和两个 destination。constraint 再校验
三 tile 的 loc/layout/dtype，以及两个 destination 各自的 compact mode。

### 9.2 A5 展开算法

当前 PTODSL 可直接使用以下公开签名，不需要先扩展 Python surface：

```python
align = pto.vldas(source)
value, align = pto.vldus(source, align)
pto.vsstb(value, destination, block_stride, repeat_stride, mask)
```

`vsstb` wrapper 会把两个 stride coercion 到 signless `i16`。template 只能传入 verifier 已证明
可编码的 bit pattern；不得依赖 Python 整数截断。普通形态不产生 updated destination，窗口
helper 应显式维护 destination offset。post-update surface 虽然存在，但只有其指针推进语义与
ND-to-NZ loop 经 VPTO 和设备测试证明一致后才可采用，不能因接口存在就默认等价。

template 复用一个 compile-time specialized window helper，两次调用分别处理 dst0/dst1，
不能假设两路 shape 相同：

1. 根据 dtype 计算 storage width、c0、vector lanes；FP4 使用 packed logical dimension。
2. 计算 source window base。
3. `1x1` 走 scalar load/store。
4. c0-aligned source base 使用 `vlds` + `vsstb`。
5. sub-c0 base 使用 `vldas` + `vldus` + exact masked store。
6. destination block stride 必须直接取第 5.4 节 shared physical-layout/access helper；plain 为
   `physicalRows`，RowPlusOne 只有在物理布局支持条件满足后才为 `physicalRows + 1`。禁止从
   `validRows` 重新计算 stride。A5 partial-valid plain template 先执行第 3.4 节 gate，只有
   `physicalRows == align16(validRows)` 才能展开；A5 RowPlusOne 在首版 template 中稳定拒绝。
7. 尾列 predicate 只允许写 valid element，不能污染下一个 NZ block 或另一 destination。
8. 所有 `vsstb` control field 在构造 `i16` 前完成静态 range proof；失败时 template 不匹配，
   由 tile op verifier 给出面向 shape 的诊断。

TileLib template 的目标不是逐行复刻 PTO-ISA C++，而是保证同一 logical mapping、tail 和
NZ+1 stride。aligned、unaligned、FP4、1x1 分支都要经过 VPTO-to-LLVM intrinsic 检查和
设备 golden，未验证的 dtype 不能被宽泛 `NUMERIC_DTYPES` 提前注册。

固定反例必须保留在 template/verifier 回归中：`physicalRows=32`、`validRows=13`、
`validCols >= 2*c0` 时，A2/A3 和非 A5 VPTO 的第二个 block offset 必须为 `32*c0`；A5
  partial-valid 必须在模板匹配前以稳定 gate diagnostic 拒绝，而不是产生 `16*c0`。对应的
A5 positive control 使用 `physicalRows=16, validRows=13`，验证 plain stride 为 `16*c0`；
  NZ+1 只在 shared physical-layout/access helper 完成后验证。另加 A5 ColMajor NZ `f16 16x32`
  full-valid layout 回归，检查第二 block subview offset=272 elements、payload=1024 bytes、
  access end=1056 bytes、allocation reservation=1088 bytes，以及 adjacent allocation 不重叠；
  回归还必须证明 1088-byte reservation 未被当作 access size。在该回归通过前，A5 RowPlusOne 双输出
  仍必须 negative。

### 9.3 A2/A3 EmitC 与 VPTO

仓库当前只有 `lib/TileOps/a5`，没有 A2/A3 TileLib template tree。本功能不借机新增整套
A2/A3 TileLib 基础设施。A2/A3 EmitC 的 ND-to-2xNZ `pto.textract` 仍按第 8 节生成一次
七参数 PTO-ISA `TEXTRACT` 调用。

A2/A3 VPTO 不能复用这条 EmitC 路径。当前 `lowerPTOToVPTOBackend()` 在
`LowerPTOToUBufOps` 后对 A2/A3 直接返回，不运行 A5 的 `ExpandTileOp`；同时
`LowerPTOToUBufOps` 会先把 `AllocTileOp` materialize 成 `!pto.ptr<..., ub>`。如果不增加专门
lowering，`pto.textract` 会带 pointer operand 残留并在 VPTO LLVM conversion 失败。

首版在现有 `LowerPTOToUBufOps` 中增加 correctness-first 的双输出 lowering，不新增 MLIR op：

1. 在任何 `AllocTileOp` replacement 之前先收集双输出 `TExtractOp`，调用 form classifier 并建立
   `PendingNd2xNzLowering` 快照：记录五段 schema、三个原始 tile type、四个折叠后的 index、
   source ND row stride、两个 destination 的 physical/valid shape、element storage bytes 和
   compact mode。lowering 只接受已经通过 verifier 的 plain-NZ、静态 metadata；缺少任一字段时
   必须在该 pass 报错，不能把 op 留给后续 conversion。
2. `AllocTileOp` materialize 后，op operands 已可能变成 `!pto.ptr<..., ub>`，此时不得再调用要求
   tile type 的 generated accessor。实现从已验证快照中的 segment offset 按 raw operand position
   取得 materialized `src`、`dst0`、`dst1` pointer。令
   `c0 = 32 / storageElemBytes`，每个 window `k` 使用与第 3.2 节相同的 element offset：

   ```text
   srcOff = (indexRow_k + r) * srcRowStrideElements + indexCol_k + c
   dstOff = floor(c / c0) * dstPhysicalRows_k * c0
          + r * c0
          + (c % c0)
   ```

   该 expansion block 位于 allocation replacement 之后、现有 TLOAD/TSTORE conversion 和 dead-view
   cleanup 之前，确保原 op 在 pass verifier 重新运行前已被消除。

3. 为两路 window 分别生成 `0 <= r < validRows_k`、`0 <= c < validCols_k` 的 `scf.for`，循环体
   用现有 `pto.load_scalar` 从 `src[srcOff]` 读取，再用现有 `pto.store_scalar` 写入
   `dst_k[dstOff]`。所有 offset 计算使用 checked/static metadata 加 `index` 算术；循环只写 valid
   logical coordinates，不初始化或读取 destination padding。
4. 因为 `load_scalar`/`store_scalar` 在 `PIPE_S` 执行，而双输出 op 对外仍建模为 `PIPE_V`，整个
   两窗口展开用第 6.3.1 节已登记的 scalar macro model 包围：进入循环前生成已分配的
   `PIPE_V -> PIPE_S` set/wait，两个窗口完成后生成已分配的 `PIPE_S -> PIPE_V` set/wait。
   `EVENT_ID0` 只有在 PTO-ISA 把它作为该 hidden event 的固定 ID、且已由 event allocator 预留时
   才能出现在最终文本中；不能把它作为 lowering 的独立硬编码。外部 InsertSync 已建立的
   MTE2-to-V 和 V-to-MTE3 依赖保持有效；不能只生成 scalar loop 而遗漏这两组内部同步。
5. 展开完成后必须 erase 原 `TExtractOp`。`LowerPTOToUBufOps` 在 A2/A3 路径返回前扫描并拒绝
   任何残留的 ND-to-2xNZ `TExtractOp`；其 operands 不得以非法的 `!pto.ptr` 形态进入 pass 后
   verifier 或 VPTOLLVMEmitter。该检查属于现有 pass，不增加新的 validation pass。

该 scalar lowering 是首版 VPTO 的正确性基线，覆盖 `i8/i32/f16/bf16/f32`、非对齐 index、
odd `validCol` 和 `1x1`，不宣称复刻 PTO-ISA 的 `vcopy`/`vconv` 性能路径。后续可在同一 pass
中把满足对齐条件的循环替换为已有 VPTO vector micro-op，但必须保持上述 offset、tail、内部同步
和数值回归；不能通过新增第二个公开 TEXTRACT op 或让 A2/A3 进入 A5 TileLib registry 来优化。

## 10. Python builder 与文档接口

继续使用现有 Python op class 和 free-function builder。旧调用保持不变，新调用使用同一 class
上的命名 factory：

```python
from ptoas.mlir.dialects import arith, func, pto

# Existing form remains source-compatible.
pto.TExtractOp(src, index_row, index_col, dst, fp=fp)
pto.textract(src, index_row, index_col, dst, fp=fp)

# The keyword spelling remains source-compatible as well.
pto.textract(src=src, index_row=index_row, index_col=index_col, dst=dst,
             pre_quant_scalar=pre_quant_scalar)

# New form; this still constructs an operation named "pto.textract".
pto.TExtractOp.build_nd_to_2xnz(
    src, index_row0, index_col0, index_row1, index_col1, dst0, dst1)
```

这里的 `pto` 明确指 PTOAS ODS-generated low-level dialect binding；`pto.py` facade 必须覆盖
`_pto_ops_gen.py` 生成的 `textract(src, indices, dsts, ...)`，不能让它取代旧的
`textract(src, index_row, index_col, dst, ...)`。实现 PR 增加以下 Python smoke：

1. 位置参数 `pto.textract(src, row, col, dst)` 和关键字参数
   `pto.textract(src=src, index_row=row, index_col=col, dst=dst, fp=fp)` 都构造
   `pto.textract`，并断言返回 op 的 `.indexRow`、`.indexCol`、`.dst` 与传入 SSA value 相同。
2. `pto.TExtractOp(src, row, col, dst, ...)` 和对应关键字构造都通过同一个 facade；双输出
   `build_nd_to_2xnz` 仍返回同一 `TExtractOp` class，且 `.indices`/`.dsts` 保留四项/两项 range。
3. smoke 必须检查 `pto.textract.__signature__`（或等价的 `inspect.signature`）仍接受旧的
   `index_row/index_col/dst` 参数，不能只检查生成 module 中的 private range-form builder。
4. 单独覆盖 parser 路径：完成 facade 的
   `register_operation(_Dialect, replace=True)` 注册后，用 canonical 文本解析一个包含
   `pto.textract` 的 module；从解析得到的 `Operation` 读取 `op.opview`，断言它是
   `pto.TExtractOp`（而不是 `_pto_ops_gen.TExtractOp`），并分别访问
   `op.opview.indexRow`、`op.opview.indexCol`、`op.opview.dst`，确认它们与文本中的三个
   legacy operand 一致。该测试不能由直接调用 `pto.TExtractOp(...)` 替代，因为后者不会
   覆盖 parser 在 operation registration 替换前后选择 opview class 的回归。

这里的 `ptodsl` 是 PTODSL micro-op surface，不能用来构造该 IR op。实现 PR
证明 legacy constructor、legacy free function 和新 factory 都打印为 `pto.textract`、
argument 顺序和文本汇编一致，且 binding 中不存在 `TExtractNd2xNzOp`。`.indexRow`、
`.indexCol`、`.dst` 的 wrapper properties 也必须纳入 smoke，不能把它们视为 generated
range API 的内部细节。

需要同步更新：

- `docs/PTO_IR_manual.md`：语义、汇编、shape/layout/dtype/compact 表；
- `docs/release/PTO-tile-Instruction-SPEC-v0.4.md`：新增双输出形态；
- `ReleaseNotes.md`：记录 `pto.textract` 新增 ND-to-2xNZ 双输出 form、架构差异，以及 public
  C++/Python getter、builder、free-function 和 property facade 兼容，但
  `Properties::operandSegmentSizes`、`getODSOperands()` 和旧 adaptor fixed-field accessor
  不兼容的边界；
- TileLib template 列表或生成文档（若实现 PR 修改对应索引）。

## 11. 测试方案

### 11.1 ODS、parser 与 verifier lit

正向（FP4 与 RowPlusOne 仅在对应物理布局/实现条件已满足时启用）：

- 既有单输出、FP、preQuant、relu、acc-to-vec 文本 parse-print-parse 不变；
- 两路相同 shape；
- 两路不同 physical/valid shape 和不同 index；
- A2/A3 `i8/i32/f16/bf16/f32`；
- A5 已验证的 low-precision 集合；
- A5 plain + plain；plain + RowPlusOne、RowPlusOne + RowPlusOne 在首版均为 negative，
  仅在第 5.4 节物理布局专项回归通过后转为正向；
- `1x1`、非 c0 index、非 c0 validCol、A2/A3 odd-i8 validCol；
- source valid extent 覆盖两个 window、但 source physical extent 更大（证明 bounds 同时使用
  valid/physical 两层）；
- 双输出 parse-print-parse 保持两个 destination 与 index 配对，打印名称始终是 `pto.textract`；
- generic form 的 canonical `[src, indices, dsts, fp, preQuantScalar]` segment sizes round-trip；
- A5 partial-valid plain NZ 只有在 `physicalRows == align16(validRows)` 时进入正向集合并使用
  `physicalRows` stride；RowPlusOne 在第 5.4 节物理布局回归通过前始终是负向。

负向：

- 非 tile、非 rank-2、非 index；
- `(indices,dsts)` 为 `(2,2)`、`(4,1)`、`(3,1)`、`(4,3)` 等未定义 arity；
- generic assembly 的 source segment 分别为 0 和 2：至少固定覆盖
  `[0, 2, 1, 1, 0]` 与 `[2, 2, 1, 0, 0]`；二者通过 property conversion 和 generated type
  invariants 后，由 custom verifier 给出实际五段值与期望 schema，且不能 crash；
- `operandSegmentSizes` property 不是五项时，断言 parser/property conversion diagnostic，不要求
  进入 classifier；文本省略该 property 时断言默认全零 property 最终验证失败，但允许
  `AttrSizedOperandSegments`/`OpInvariants` 先诊断；
- 五段总和与 operand 数不符，以及 `fp`/`preQuantScalar` segment 为 2；按真实验证阶段分别匹配
  generated trait/invariant 或 custom classifier diagnostic，不统一要求 actual/expected 文案；
- 双输出 form 携带 `fp`、`preQuantScalar`、非默认 relu 或 `accToVecMode`；
- dynamic valid shape 或 dynamic index（首版）；静态 0 valid row/col；
- source/destination loc 错误；
- ND/NZ layout 或 fractal size 错误；
- 三者 dtype 不一致或架构不支持 dtype；
- source row stride 非 32B 对齐；
- destination plain-NZ logical physical rows 或 emitted physical cols 不满足 NZ；
- A5 aligned path 的 `vsstb` block/repeat stride 超出 16-bit field；RowPlusOne 在首版无论
  virtual rows 是否可编码都拒绝；
- window0/window1 各自的负 index、超出 source physical 的 row/col，以及 physical 仍足够但
  超出 source valid extent 的 row/col；后者必须稳定报告读取 undefined padding；
- source valid shape 缺失、动态、为零，或小于任一 window 而 source physical shape 仍覆盖该
  window；不得仅因 physical bounds 通过而放行；
- A2/A3 任一路使用 RowPlusOne；
- 三种显式 alias pair 各一例；
- `DeclareTileOp` 作为 src、dst0、dst1 的各一例；`TAssignOp` result、`TPopOp`/
  `TPopFromAicOp`/`TPopFromAivOp` 派生值以及它们的 subview/cast 各一例；所有 level 都必须
  在 planner 前得到 runtime-bound provenance 诊断；
- RowPlusOne `AllocMultiTileOp` 在 level1/2 planner-owned 和 level3 explicit-base 各一例，均由
  既有 verifier 稳定拒绝；另覆盖 `MultiTileGetOp` 后接 subview/cast 再作为 dst0/dst1，证明
  单 allocation 的 RowPlusOne 物理布局支持条件满足后也不能绕过 multi-buffer 限制。固定双 slot
  反例使用 `f16 16x32`：旧 product-shape slot1=`base+1024` 会与 slot0 access tail
  `[1024,1056)` 重叠；测试不得进入 `PTOResolveBufferSelect` 后才失败；
- level3 下任一 operand 的 `alloc_tile.addr` 含动态 root；
- 目标 backend 不具备现有 PTO-ISA 双输出 API 时，保留既有编译失败诊断；不新增环境参数或
  输入 IR 属性来绕过该失败；
- 物理布局支持条件未满足时使用 FP4 或 RowPlusOne；
- A5 ColMajor NZ `f16 16x32 RowPlusOne` 在 shared helper 尚未覆盖全部消费者时稳定拒绝；专项回归
  必须保留 subview second-block offset=272 elements，并拒绝任何回退到 256 的实现，
  同时拒绝 1086-byte 旧公式和把 1088-byte reservation 当成 access footprint；精确匹配
  payload=1024 bytes、access end=1056 bytes、allocation reservation=1088 bytes；
- FP4 支持条件完成后，覆盖 raw dimension 合法但 emitted dimension 非法，以及反向边界；
- `physicalRows=32, validRows=13, validCols >= 2*c0`：A2/A3 与非 A5 VPTO 正向检查第二个
  block offset 为 `32*c0`；A5 partial-valid 必须在 template/verifier 阶段稳定拒绝，不能生成
  `16*c0`；`physicalRows=16, validRows=13` 作为 A5 positive control；
- generated Python range-form free function 不得覆盖 legacy facade；位置/关键字
  `pto.textract`、位置/关键字 `pto.TExtractOp` 以及 `.indexRow/.indexCol/.dst` 各有正向 smoke；
  parser 在 facade registration 后得到的 `op.opview` 也必须是 `pto.TExtractOp`，并通过三个
  legacy properties 回归。
- driver post-planning safety helper 分别拒绝 partial descriptor 的直接/view TSTORE、同址不同
  SSA full-valid `alloc_tile` alias TSTORE 和部分重叠 alias；即使 TSTORE 文本上位于 TEXTRACT 前
  或中间存在 overwrite，首版 component 级保守规则仍拒绝；
- 增加跨函数负向：callee 产生 partial、caller 用同址 allocation TSTORE；caller 产生 partial、
  direct callee TSTORE；`entry -> helper -> leaf` 三级调用中 producer/TSTORE 分居两端；tile block
  argument 使 TSTORE source range unresolved；compile unit 存在 partial producer 时，任意 direct
  component 中存在 indirect、external、unresolved direct call 或其他 opaque `CallOpInterface`；
  这些 case 都必须在 post-planning helper 稳定失败；
- 固定覆盖 opaque 跨 component 反例：`@entry` 通过两个 `func.call_indirect` 分别潜在调用
  partial producer function 和同址 TSTORE function，三者在 direct-call graph 中没有 edge；以及
  opaque/external call 位于与 partial producer 完全 disconnected 的第四个 function。两例都必须由
  compile-unit-wide closure 拒绝。另加无 partial producer 但存在 opaque call 的非回归，证明本
  feature helper 不无条件禁止既有 indirect-call IR；
- 对同一跨函数负向分别运行普通 codegen 和 `--emit-pto-ir`，两者必须在
  `PTOInlineBackendHelpersPass` 前从相同、尚未 inline 的 function component 得到同一类别诊断；
  另加一个会被该 pass inline 的 backend-helper call case，证明检查不是在 main pipeline 整体运行后
  才观察已经扁平化的 IR；
- 增加 mixed-backend partition 负向：child A 的 level3 `alloc_tile addr=0` 产生 partial
  ND-to-2xNZ，child B 通过 `func.call @A` 后以同址 `alloc_tile addr=0` 执行 `TSTORE`；必须在
  child clone/declaration 和任一 child compile 之前由 outer-module precheck 稳定拒绝。反向
  caller/callee、三级跨 child component、同 backend 但分 child 的 case 也覆盖。另固定覆盖本规则的
  closure 反例：child A 有与其他函数不相连的 partial producer，child B 的 full-valid `@caller`
  唯一解析并调用 child C 的 full-valid `@callee`；即使该 edge 不传递 tile，也必须在 clone 前拒绝，
  不能让 child B 内的 declaration 被 post-planning helper 误判为 opaque；
- declaration-shadowed 负向必须复用真实 driver 形态：caller child 同时包含
  `func.func private @A(...)` declaration 和 `func.call @A(...)`，sibling child 包含唯一
  `func.func public @A(...)` full-valid definition，第三个 child 包含不相连的 partial producer；
  precheck 必须跨过 local declaration 连接 sibling definition，并按跨 child call 拒绝。另覆盖完整
  outer module 含 partial producer 时，任一 child 的 sibling definition 为零个和多个的
  unresolved/ambiguous 诊断，不能 crash 或静默截断调用图；
- 增加 peer-reserve 负向：partial producer 所在 child 既包含同 child `pto.import_reserved_buffer`，
  也包含跨 child `peer_func` import；两者都必须在 child clone 和 `PTOResolveReservedBuffersPass`
  删除/materialize import 之前拒绝，即使 import 与 partial producer 不在同一 direct component、也
  没有显式 `func.call`。该回归必须证明 `ImportReservedBufferOp` 虽然不是 `CallOpInterface`，仍按
  `peer_func` exact-symbol/logical-name 解析用于稳定诊断；还覆盖 zero/ambiguous/private peer lookup，
  以及无 partial producer 时保留既有 peer clone/address-materialization 行为；
- 增加 mixed-backend global-closure 负向：partial producer 在 child A，opaque/unresolved call 在
  direct graph 与其不相连的 child B；必须在拆分前拒绝。增加 structure-guard 负向：immediate
  backend child 下再嵌套 `ModuleOp`（无论 nested module 是否带 backend attr），以及
  `func.func` 不是 immediate child 的 direct op；普通 object codegen 与 user-visible IR output 都
  必须在 child clone/pipeline 前得到 fixed-depth diagnostic，不能静默漏掉 nested producer/TSTORE。
  另覆盖“一个 backend child + 顶层 `func.func`”的混合根结构，证明 object 路径不会因
  `isBackendPartitionedContainer()` 返回 false 而跳过 guard，且 `--emit-pto-ir` 与 object 路径得到
  同一诊断。fixed-depth direct-function case 正向通过；不含 ND-to-2xNZ form 的既有 nested input
  不得新增该 feature diagnostic；
- 增加 component scope 正向：两个互不连通的 entry/kernel function 使用相同 UB 数值地址，只有
  一个 component 含 partial producer、另一个含 TSTORE，且整个 compile unit 不含 opaque/external/
  unresolved call；不得因 range 检查退化为 module-wide 地址并集而误拒绝；
- 增加 partition scope 正向：partial producer/TSTORE component 完全位于 child A，child B 是
  独立 non-partial component（即使复用地址）且不存在跨 child direct call 时，不得被 precheck
  误拒绝；outer module 不含 partial producer 时，full-valid component 通过 local declaration
  唯一解析到 sibling public definition 并跨 child，仍保持既有 driver 语义；
- production/test build 使用同一规则；不存在 hidden option、test-only module attribute 或
  canonical fixture escape。

### 11.2 EmitC 与 C++ compile

- 增加一个只包含/编译旧 `TExtractOp` C++ 调用面的 compile-only translation unit，逐项覆盖第
  4.4 节列出的 typed getter、mutable accessor、`ReluPreModeAttr`/`ReluPreMode`、带/不带
  `TypeRange`、`OpBuilder`/`ImplicitLocOpBuilder` 和两组 generic `build`/`create` overload；尤其
  用显式 `TypedValue<IndexType>`/`TypedValue<Type>` 接收 legacy getter，防止返回类型退化；
- 同一 compile-only 回归明确不把 `Properties::operandSegmentSizes` 的旧六项数组布局、
  `getODSOperands()` 或旧 adaptor accessor 当作兼容 API；ReleaseNotes 检查固定记录这些迁移边界；
- A3/A5 FileCheck 精确匹配七参数顺序；
- 同一 `PTOExtractToEmitC` pattern 的 legacy 单输出/FP/preQuant/relu/acc-to-vec FileCheck 全部保留；
  编译回归必须证明 core operands 来自 `adaptor.getIndices()/getDsts()`，源码中不再引用不存在的
  `adaptor.getDst()/getIndexRow()/getIndexCol()`；
- dynamic index 的 verifier 诊断，确保 EmitC 与 VPTO 不出现后端分叉；
- 两个 dst 使用不同 opaque Tile type，确保 lowering 没有误用 dst0 type；
- A5 RowPlusOne 在支持条件未满足时不生成 C++；专项回归才检查 destination Tile type 包含正确
  `CompactMode::RowPlusOne`、`Tile::Rows=17`，并与 subview/TSTORE stride=272 elements、
  TSTORE access end=1056 bytes、planner allocation reservation=1088 bytes 一致；
- FP4 检查 doubled packed dimension；
- A2/A3/A5 对目标 backend 的最小双输出调用做 compile-only；目标 backend 尚未具备该 API 时，
  该 backend 只记录为未覆盖，不修改 PTOAS driver 的参数或输入 IR 状态；
- `--emit-pto-ir`、EmitC 和 VPTO 对同一 `pto.textract` IR 共享 form/verifier 结果，不能通过
  输出模式差异绕过非法双输出 schema、layout、provenance 或 alias 检查；

PTOBC v0 兼容测试单列，不并入普通 MLIR bytecode 假设：

- 已发布四 operand 单输出和五 operand FP fixture 继续 decode 为单输出 `pto.textract`；
- 单输出/FP 重新 encode 时仍使用原 fixed-width opcode 和 operand count；
- 双输出强制使用 generic v0 record，encode-decode 后保留四项 index、两个 destination 及类型；
- 更新后的 v0 decoder 按 generic record 规则处理双输出，不能把七 operand payload 误认成
  原四 operand schema；不承诺旧 PTOAS binary 向前识别新增 form。

### 11.3 effects、sync 与 PlanMemory

- effects 测试看到一个 Read、两个 Write；
- 对 `[2, 2, 1, 0, 0]` 等 Invalid schema 直接查询 MemoryEffects 不崩溃，并把两个 raw source
  tile 及其他 memory-carrying operands 保守建模为 Read+Write；
- full-valid case 的 `TLOAD -> ND2XNZ -> 2xTSTORE` 自动同步覆盖两路；
- A2/A3 scalar form 和 A5 `1x1` 的 `SyncMacroModel` 在 event-id allocation 前声明 V/S phases
  及双向 hidden event；回归检查 lowering 生成的 set/wait 使用已登记事件，不能出现 allocator
  未观察到的裸 `EVENT_ID0`，并证明 compiler-generated event 不与内部事件复用；A5 普通 vector
  form 不额外注入 V/S hidden event；
- 两个 dst 的 consumer 位于不同 block/loop 时 liveness 都正确；
- legacy/modern planner 都为两个 live destination 分配不重叠范围；
- source/dst0/dst1 的 subview overlap 被拒绝；
- driver input provenance helper 位于 generic verification 之后、`preInlinePlanningPM` 之前；
  declared tile、tassign、tpop 及其 view chain 在两个 planner 运行前均被拒绝；静态
  alloc-backed 正向 case 继续进入对应 planner；RowPlusOne multi-buffer 与其 view chain 在
  generic verifier/input provenance 阶段拒绝，不进入 planner；
- level3 三组 pair 分别使用同一动态 `%base` 时拒绝，`%base + constant` 的动态派生地址也
  拒绝；三个静态非重叠常量地址通过，静态重叠地址继续由 range verifier 拒绝；
- post-planning helper 覆盖 direct/view/same-address allocation/partial-overlap TSTORE；同一
  direct-call graph component 内任意相交都拒绝，不做 branch/loop/interprocedural summary
  dataflow，也不因 full overwrite 清除；相同数字地址但不同 `VEC`/`MAT`/`ACC` address space
  不判 alias，unknown address space/range 同样保守拒绝；
- shared physical-layout/access helper 回归对 A5 ColMajor NZ `f16 16x32 RowPlusOne` 同时检查：
  `PTOResolveBufferSelect` 的 `colStride=17`、第二 block subview offset=272 elements；exact payload
  intervals `[0, 512)`/`[544, 1056)` bytes、payload total=1024 bytes；legacy/modern PlanMemory
  allocation reservation=1088 bytes；semantic range 和 InsertSync access envelope=1056 bytes/8448 bits；
  EmitC `Tile::Rows=17`，TSTORE block stride=272 elements、access end=1056 bytes。subview 产生 256、
  任一路径产生 1086，或把 1088-byte rectangle 当作 access size 都是回归，并保持 RowPlusOne
  首版拒绝；
- multi-buffer 回归锁定本设计的 unsupported 边界：plain-NZ 两 slot 的既有
  `AllocMultiTileOp -> MultiTileGetOp -> resolve/sync` 正向行为保持不变；RowPlusOne 两 slot 在
  level1/2 和 level3 均稳定失败，且 diagnostics 明确要求单 `pto.alloc_tile`。不得新增
  `slot_stride_bytes`/`slot_access_end_bytes` 属性、不得修改 multi-buffer textual/PTOBC schema；
- call-surface closure 回归覆盖：存在 partial producer 时，opaque/indirect/external/unresolved call
  位于相同或任意 disconnected direct component 都拒绝；尤其覆盖一个 caller 中两次
  `func.call_indirect` 潜在连接 producer/TSTORE 的反例。无 opaque call 的 disconnected
  same-address components 继续通过，无 partial producer 的既有 opaque-call module 不因本 helper
  失败；
- pipeline-order 回归固定 `preInlinePlanningPM -> post-planning helper -> postValidationPM`；普通
  codegen 和 `--emit-pto-ir` 对相同输入共享检查点，且 diagnostic/IR dump 证明 helper 运行时
  `PTOInlineBackendHelpersPass` 尚未执行；
- 跨函数 `_post_planning_safety` 回归覆盖 callee-producer/caller-store、caller-producer/callee-store、
  三级 transitive call、block-argument unresolved source，以及 compile-unit-wide
  indirect/external/unresolved callee 拒绝；另有 call-surface-closed 的 disconnected entry
  components 同址正向，锁定 range 检查不能退化为 module-wide 地址并集；
- mixed-backend `_pre_partition_safety` 回归在 `collectChildJobs()` 之前覆盖跨 child partial
  producer/TSTORE、反向调用、三级跨 child component 和同 backend 分 child 拒绝；outer module 有
  partial producer 时，另覆盖与 partial component 不相连的 full-valid cross-child direct call 拒绝，
  确保 child clone 的 declaration 不会被 post-planning helper 当作 opaque；caller child 已有 private
  declaration 时，必须按 exact final-link symbol 唯一解析到 sibling public definition；任一 child 的
  零/多匹配或 opaque call 在 outer module 含 partial producer 时稳定失败；nested `ModuleOp` 和非
  direct-child `func.func` 在 backend/output 路由前稳定失败；`ImportReservedBufferOp` 的跨 child
  任意 peer import 也必须在 resolve 前拒绝；partial component 单 child 且没有 cross-child direct
  call 或 peer import、outer module 无 partial producer 的 full-valid cross-child 和
  call-surface-closed 的 disconnected same-address child 正向通过；
- plain 与 RowPlusOne 混合 case 在 shared-helper 完成后使用各自 allocation/access record；
  helper 未完成时稳定拒绝；
- 已有单输出 `textract` sync/plan-memory tests 全部保持不变。

### 11.4 TileLib / VPTO

A2/A3 `LowerPTOToUBufOps` 回归：

- pass-only FileCheck 在 planned `alloc_tile` 已变成 `!pto.ptr<..., ub>` 后运行，要求双输出
  `pto.textract` 完全消失，并分别出现 dst0/dst1 的两个 `scf.for` loop nest；
- 两路使用不同 shape/index，精确检查 `srcOff` 和 plain-NZ `dstOff` 算式，特别是
  `physicalRows=32, validRows=13, validCols >= 2*c0` 的第二 block 从 `32*c0` 开始；
- 展开前后分别存在由 `SyncMacroModel`/event allocation 登记的 V-to-S、S-to-V
  `set_flag`/`wait_flag`，循环体只含已有 `pto.load_scalar`/`pto.store_scalar` pointer op；不得
  新增 raw TEXTRACT op，也不得出现未登记的硬编码 `EVENT_ID0`；
- `i8/i32/f16/bf16/f32`、unaligned index、odd-i8 `validCol`、`1x1` 和两路不同 valid shape
  都能完成 VPTO-to-LLVM，LLVM verifier 通过且没有残留 `pto.textract`；
- 缺失 tile metadata、动态 shape/index 或非 plain-NZ input 在 verifier/pass boundary 给出稳定
  诊断，不得以 pointer operand type mismatch 作为晚期失败；
- byte-exact VPTO simulator golden 对两个 window 分别比较 logical NZ payload，并保留既有
  A2/A3 EmitC 七参数调用测试，证明两个 backend 使用同一 form 与 offset 语义。

A5 TileLib / VPTO 回归：

- aligned f16：展开为 load + block-stride store，不残留 tile op；
- unaligned f32：出现 unaligned load path；
- public surface probe 精确生成 `vldas`、`vldus` 和普通 `vsstb`，不新增私有 builder；
- 两路不同 shape/index；
- tail validCol mask；
- `1x1`；
- A5 NZ+1 在首版保持 negative；支持条件完成后的专项回归使用 full-valid ColMajor NZ `f16 16x32`，同时检查
  `physicalRows + 1`、第二 block offset=272 elements、两个 512-byte payload interval、
  access end=1056 bytes，并断言 1088-byte rectangle 不是 access size。`32/13` 与 `16/13`
  只覆盖 partial plain-NZ gate 的 negative/positive control；
- hif8/fp8 检查正确 vreg/intrinsic type，并至少覆盖一个 1-byte low-precision sub-c0 window；
  FP4 支持条件完成后再增加 FP4 检查；
- `vsstb` control field 最大合法值和首个非法值；
- VPTO LLVM verifier 和现有 CANN output version 组合通过。

### 11.5 NPU ST

PTOAS 新增独立 testcase，不复用 PTO-ISA ST 二进制。golden 对两个 window 分别切片并转换
ND-to-NZ，两个输出独立比较。测试按第 3.2.1 节分成两组：

- full-store group：两个 destination 都是 full-valid；plain NZ 使用 canonical physical NZ
  GlobalTensor shape，经过 `TLOAD -> TEXTRACT -> two TSTORE`，debug build 必须保持
  assertion enabled，并比较两块完整 physical GM output。A5 `RowPlusOne` 只有第 5.4 节所有
  physical-layout consumers 的精确回归和下述 payload/gap-skip device tests 通过后才允许进入；
  golden 不得为 UB gap 定义或比较具体值，否则会把未定义 padding 错当成 TEXTRACT 语义。
- partial-valid group：不得通过 full-valid alias generic `TSTORE` 绕过 production rule。simulator
  直接读取 UB；NPU 使用独立 backend-native raw-buffer harness 导出每个 destination 的 physical
  allocation extent 和 32B pre/post UB sentinel。golden 只比较 valid logical coordinates，未定义 padding
  不比较；四个 UB sentinel 和 physical GM output 两侧 host guard 必须逐 byte 比较。

RowPlusOne 的 device coverage 分成三个互补断言：TEXTRACT raw-UB dump 只比较 exact payload
intervals，忽略 block 间 gap；独立 TSTORE gap-skip testcase 用受控 raw UB buffer 给 payload 与 gap
填入可区分数据，证明 TSTORE 只把 payload 映射到 GM 且不读取/导出 gap；端到端
`TLOAD -> TEXTRACT -> TSTORE` 只比较完整 logical GM payload 和 GM guards。三者都不得检查
TEXTRACT 后 gap 的具体字节，也不得把 gap sentinel 保持不变当作通过条件。

若某架构的公开 `1x32xi8` tile TLOAD/TSTORE 编译或设备测试失败，必须使用独立
raw-buffer NPU harness 通过 backend-native UB-to-GM byte copy 导出相同 pre/post redzone；raw
harness 落地前，该架构的 odd-valid/`1x1` case 只能计 compile-only/simulator coverage，不能计
NPU ST。GM guard 只能证明 GM dump 没有越界，不能代替 UB allocation redzone 观测。

A2/A3 最小集合：

- f16 aligned full-valid（完整 TSTORE）；
- f16 或 i8 unaligned index；
- i8 odd validCol；
- i32；
- `1x1`；
- 两路不同 valid shape。
- A3/VPTO partial-valid `physicalRows=32, validRows=13, validCols >= 2*c0` 的 physical-stride
  golden；A5 同形状的 negative，以及 `physicalRows=16, validRows=13` positive control。

A5 必选最小集合：

- f16 aligned full-valid（完整 TSTORE）；
- f32 sub-c0 unaligned；
- hif8 和至少一种 fp8 的 sub-c0 unaligned byte-exact case；其余 1-byte low-precision dtype
  至少覆盖 aligned byte-exact case；
- `1x1`；
- 两路不同 valid shape。
- partial-valid `physicalRows=32, validRows=13` 必须是 negative；
  `physicalRows=16, validRows=13` 只作为 plain stride 的 positive control；RowPlusOne 仍是
  negative。

A5 物理布局专项集合：

- FP4 packed dimension：必须同时覆盖 RowMajor ND source 和 ColMajor NZ destination 的
  packed axis、row stride 与 byte-exact golden；
- plain + RowPlusOne：先用 full-valid case 经过 `TLOAD -> TEXTRACT -> two TSTORE`，证明
  shared helper、第二 block subview offset、planner allocation、semantic/sync access
  envelope、virtual rows 和 TSTORE stride 一致；`f16 16x32` 必须精确得到 272-element stride、
  1024-byte payload total 和 1056-byte exclusive access end。该 full-valid case 与设备 golden
  通过后才解除 full-valid RowPlusOne 的首版拒绝；partial-valid RowPlusOne 仍保持 unsupported。

full-store group 必须经过完整链路，partial-valid group 必须保留明确的 UB-only 标记；测试
汇总分别报告 `TEXTRACT numerical coverage` 和 `full TSTORE coverage`，不能把后者的数量
扩大到 odd-valid/`1x1` case。

### 11.6 回归门槛

- `test/lit/pto/textract_*` 全部通过；
- `test/lit/vpto/*textract*` 全部通过；
- A2/A3 pass-only 回归证明 `LowerPTOToUBufOps` 消除 pointer-form 双输出 op、生成两路 scalar
  loop 与由 `SyncMacroModel` 预留事件驱动的完整 V/S flag-wait；A2/A3 final VPTO-to-LLVM 和
  LLVM verifier 通过且无残留
  `pto.textract`；
- A3/A5/VPTO stride counterexample matrix 通过：A3/VPTO physical stride 正向、A5 `32/13`
  plain negative、A5 `16/13` plain positive；RowPlusOne 在 shared-helper 完成前保持 negative；
- PTOBC v0 legacy TEXTRACT fixture 与双输出 generic round-trip 全部通过；
- PTOAS unit/lit 全量通过；
- A3/A5 compile-only；
- post-planning safety helper 的 direct/view/same-address/partial-overlap/unresolved-range 和
  interprocedural call-component 正负向回归通过；普通 codegen 与 `--emit-pto-ir` 共用
  `PTOResolveBufferSelect` 后、`PTOInlineBackendHelpersPass` 前的检查点；declaration-shadowed
  mixed-backend call、sibling zero/unique/ambiguous resolution、compile-unit-wide opaque closure 和
  fixed-depth nested-child rejection 回归通过；production/test build 没有行为差异；source valid/physical 双 bounds、
  same-numeric-address/different-address-space no-alias 和 unknown-space conservative failure 回归通过；
- A3/A5 至少执行必选 NPU ST；FP4、RowPlusOne 只有第 5.4 节所有 consumer 精确回归和对应
  device/ST 证据通过后才可解除对应 verifier negative gate。

## 12. 实现拆分

建议按以下顺序提交，保证每一步都可单独 review：

| 阶段 | 内容 | 完成标准 | 实现 PR 粗略规模 |
|---|---|---|---|
| 0 | 确认目标 backend 的双输出 API 与构建前提 | A2/A3/A5 目标路径的双输出 API 可被最小调用验证；既有构建流程可编译目标路径 | 约 80-150 行检查/构建调整及回归 |
| 1 | 扩展 `TExtractOp` ODS ranges、inherent-property schema validator/form classifier、custom assembly、精确兼容 builder/accessor、DPS、pipe、effects、PTOBC shim | property conversion、generated invariants、custom verifier 各阶段负向测试不崩溃；src=0/2 等可到达 classifier 的 schema 稳定失败且 effects 保守；RowPlusOne `AllocMultiTileOp` 的既有 verifier rejection 保留；旧 C++ API compile-only、legacy/new parse-print、binding、v0 bytecode 兼容和 range-based adaptor 编译测试通过，且没有新增 op 名或 multi-buffer stride/access 属性 | 约 600-1000 行 ODS/C++/Python/bytecode 兼容代码及 300-500 行回归 |
| 2 | shared checked physical-layout/access helper、A5 partial-valid plain stride gate、IR verifier，以及 driver input-provenance/post-planning-safety helper | helper 分离 emitted dimension、subview/block stride、payload intervals、access end 和 allocation reservation；仅单 `AllocTileOp` 的 legacy/modern planner、ResolveBufferSelect、semantic range、InsertSync 切换完成；所有 physical range 保留 address space；`16x32xf16 RowPlusOne` 精确得到 272-element stride、1024-byte payload、1056-byte access end、1088-byte allocation reservation，且 1088-byte reservation 不进入 access consumers；RowPlusOne multi-buffer/view chain 在 planner 前失败，plain multi-buffer 正向不变；架构矩阵包含 A3/A5/VPTO `32/13` counterexample；main pipeline、安全 closure、partition/provenance lit 通过，且没有新增 validation pass/test escape | 约 900-1500 行 helper/规划/driver 代码及 500-800 行回归 |
| 3 | no-alias、`TExtractOp` `SyncMacroModel` 与 planner 回归 | 三组 alias 被拒绝，declared/tpop provenance 在 planner 前失败，动态 level3 地址失败，双输出 liveness 正确；A2/A3 scalar 与 A5 `1x1` 的 V/S hidden event 在 event-id allocation 前登记且不与 compiler event 复用；RowPlusOne two-slot negative 与 plain multi-buffer positive 通过 | 约 300-500 行 planner/sync 改动及 200-350 行回归 |
| 4 | EmitC pattern | A3/A5 精确文本与目标 API compile-only 通过 | 约 150-300 行 EmitC 代码及 100-200 行回归 |
| 5 | A2/A3 `LowerPTOToUBufOps` scalar correctness lowering、A5 TileLib/VPTO template 与 Python facade | A2/A3 pointer-form 双输出 op 展开为两路 scalar loop 和由 `SyncMacroModel`/event allocation 登记驱动的完整 V/S flag-wait，final VPTO LLVM 无残留 `pto.textract`；A5 physical-stride、aligned/unaligned/tail/enabled-lowp 展开通过；legacy free function/property/constructor smoke 通过；NZ+1 compile/IR 回归消费 shared layout；FP4 随独立物理布局条件开启 | 约 700-1200 行 lowering/TileLib/VPTO/Python 代码及 450-750 行回归 |
| 6 | A3/A5 NPU ST、UB sentinel/raw-buffer harness | 必选组合两路 byte-exact；partial case 实际导出 UB redzone；RowPlusOne full-store device golden 与所有精确 layout 回归通过后解除首版拒绝 | 约 300-600 行 harness/fixture/脚本，不含设备侧生成物 |
| 7 | manual、SPEC、ReleaseNotes | 文档与实际 verifier/EmitC 一致 | 约 100-200 行文档与发布说明 |

上表规模是实现 PR 的粗略 review 预算，按新增/实质修改的源码和回归估算，不把生成文件、构建
产物或设备输出计入行数；实际拆分可以因仓库既有 helper 复用而下浮，但不应把阶段 2/3 的安全
检查压缩为未覆盖的隐式逻辑。

## 13. 兼容性与完成条件

现有 `pto.textract` 的 op 名、单输出 canonical 文本、语义和生成 C++ 不变，但同一个 ODS class
内部从 fixed fields 改成 `indices`/`dsts` ranges。自动生成的 storage accessor/builder 形态会变化，
必须由第 4.4 节的 wrapper 和 overload 保持已列出的 public C++ source surface，不能声称 ODS
API 天然零变化。`Properties::operandSegmentSizes` 布局、`getODSOperands()` 和 adaptor fixed-field
accessor 等 generated internals 明确不在兼容范围内。没有新 op，也不提供 deprecated alias。

PTOBC v0 不迁移已发布的单输出 wire schema：旧 fixed-width record 继续可解码，新双输出 form
走 generic record。MLIR generic assembly 中手写的旧六项 `operandSegmentSizes` 不是 canonical
public syntax；实现只承诺 canonical `pto.textract ins(...) outs(...)` 文本和上述 PTOBC fixture
兼容。若仓库存在直接持久化 generic assembly 的用户，ReleaseNotes 必须给出新五段 schema。

首版有三项明确的可用性取舍，不能在实现 PR 中被误解成未完成的校验细节：

- **opaque-call closure 是 compile-unit-wide 的保守规则。** 只要一个 compile unit 存在
  partial-valid producer，任意 `func.call_indirect`、external/unresolved call 或其他无法闭合的
  call-like op 都会拒绝，即使它在数据流上理论上与 producer 无关。首版不做 function-pointer
  target、address-taken 或跨函数 range/effect summary 分析，因此不能根据“真实 kernel 里这种
  组合可能少见”而放宽规则；无 partial producer 的 module 仍保持 opaque-call 非回归。后续若要
  降低过拒绝，必须先提供经过验证的 target-set 与 producer/TSTORE summary，并保留 disconnected
  component、mixed-backend child 和 indirect-target 回归。
- **runtime-bound tile provenance 会排除现有 runtime/queue-fed pattern。** `DeclareTileOp`、
  `TAssignOp`、frontend `TPop*` 及其 view chain 常用于由 tpop、队列或调用方在运行时绑定 tile；
  这些 pattern 在首版不能作为 ND-to-2xNZ 的 source/destination，必须改用 planner-owned
  `alloc_tile`（level3 还要提供静态可证明地址），或等待后续 provenance contract。该限制不改变
  这些 op 对其他 PTOAS 操作的既有可用性。
- **RowPlusOne multi-buffer 需要单独跟踪。** 本设计刻意永久拒绝该组合，因为现有
  `AllocMultiTileOp` 的单一 slot 字段不能同时表达 slot reservation stride 和 per-slot access end。
  合入实现 PR 时应同时创建 follow-up issue，标题至少包含“RowPlusOne multi-buffer slot stride /
  access-end split”，并把 issue 链接写入 release note；后续设计必须覆盖 ODS/verifier、slot 地址
  物化、两套 planner、InsertSync、文本/PTOBC 兼容和双 slot overlap 回归后，才能
  删除本首版拒绝规则。

实现合入必须同时满足：

- PTO IR 能表达两个不同 shape 的 NZ destination 和两组 index；
- `TExtractOp` classifier 从 typed inherent property 而非 raw dictionary 读取完整五段 schema，要求
  `src == 1`、optional segment 为 `0/1`，再对 `(2 indices, 1 dst)` 与
  `(4 indices, 2 dsts)` 唯一分派；
- property conversion、generated traits/OpInvariants 和 custom verifier 的真实顺序有分层回归；
  src=0/2、其他组合和双输出附带 legacy optional operand 均稳定失败且不崩溃，但不承诺所有
  malformed input 共享 actual/expected schema 诊断；
- 第 4.4 节列出的 typed getter/mutable/build/create public C++ surface 有 compile-only 回归；
  ReleaseNotes 明确 `Properties` 五段布局和其他 generated internal API 不兼容；
- verifier 的合法集合不宽于目标 PTO-ISA，且在对应实现条件已通过时不误拒绝其
  unaligned/odd/1x1 路径；A5 `physicalRows != align16(validRows)` 的 partial-valid 形态属于
  首版明确拒绝的 backend stride gate，不是隐式的 dtype/layout 限制；
- 两个 DPS init 在 effects、sync、fusion boundary 和两套 PlanMemory 中都不丢失；
- Invalid segment schema 的 interfaces fail-safe，MemoryEffects 对所有 raw memory-carrying operands
  保守给出 Read+Write，不存在额外 source 绕过依赖建模的路径；
- 三 tile 两两 no-alias；
- `DeclareTileOp`、`TAssignOp`、`TPopOp`/frontend pop 绑定及其 view chain 在两个 planner 前均被
  runtime-bound provenance gate 拒绝；正向 operand 必须来自 planner-owned allocation；
- level3 中该双输出 form 的三个 local allocation 都有可静态证明的地址；
- EmitC 只生成一次、参数顺序精确的公开 `TEXTRACT`；
- EmitC 的单/双输出分支都从 adaptor `indices`/`dsts` ranges 取 core operands，不依赖 ODS
  range 化后不存在的 legacy adaptor accessor；
- A2/A3 VPTO 在现有 `LowerPTOToUBufOps` 内把双输出 form 完整展开为两路
  `scf.for + load_scalar/store_scalar`，使用 physical source row stride 和各自 destination physical
  rows 计算 offset，并生成由 `SyncMacroModel`/event allocation 登记驱动的 V-to-S/S-to-V 内部
  flag-wait；pass 返回后和 final LLVM emission 均无残留 pointer-form `pto.textract`，且不新增
  backend TEXTRACT op；
- Python `pto.py` 在 generated symbol export 后重新提供 legacy `TExtractOp` facade 和
  `textract(src, index_row, index_col, dst, ...)` wrapper；位置/关键字调用、
  `.indexRow/.indexCol/.dst` properties 和 `build_nd_to_2xnz` 均有 smoke；
- A5 partial-valid 的 `physicalRows != align16(validRows)` 在 backend-boundary/template gate
  稳定拒绝；A2/A3/VPTO 的 block stride 始终使用 physical rows；A3/A5/VPTO counterexample
  matrix 和 `16/13` positive control 通过；
- 每个 source window 同时通过 source physical 和 source valid bounds；physical allocation 覆盖但
  valid extent 不覆盖的 window 稳定拒绝并报告 undefined-padding read；
- A5 ND-to-2xNZ RowPlusOne 首版默认拒绝；解除前，shared checked
  physical-layout/access helper 必须成为 `PTOResolveBufferSelect`、legacy/modern PlanMemory、semantic
  range/post-planning alias、InsertSync、EmitC/TileLib/TSTORE 的唯一布局来源。ColMajor
  NZ `f16 16x32` 必须在所有路径一致得到 `Tile::Rows=17`、subview `colStride=17`、
  第二 block offset=272 elements、payload intervals `[0, 512)`/`[544, 1056)` bytes、
  payload total=1024 bytes、access end=1056 bytes、allocation reservation=1088 bytes、sync size=8448 bits 和单区间
  range `[base, base + 1056)`。1088 bytes 只用于 rectangular allocation reservation；任何
  256-element offset、1086-byte 旧公式、把 1088 当作 access size 或局部重复 `+1` 都阻止
  支持条件解除；
- RowPlusOne 支持条件满足后仍只支持单 `AllocTileOp` backing。任一 RowPlusOne destination
  来自 `AllocMultiTileOp`、`MultiTileGetOp` 或其 view chain 时，在 level1/2/3 均稳定拒绝；固定
  `f16 16x32 count=2` 回归证明旧 slot1=`base+1024` 的重叠路径不可达。plain-NZ multi-buffer
  正向行为保持不变，本设计不修改 `PTOOps.td`/multi-buffer textual/PTOBC schema，也不引入
  `slotStrideBytes`/`slotAccessEndBytes`；
- 所有宣称支持的实际编译 target，其现有 PTO-ISA/API 配置同时包含公开 overload 和对应
  backend implementation，并有目标路径的 compile/ST 证据；
- driver post-planning safety helper 能拒绝 direct、view、同址/重叠 allocation alias 和
  unresolved source range/address-space 的 partial TSTORE；检查覆盖同一 direct-call graph component 的
  caller/callee/transitive helper；compile unit 含 partial producer 时，任意 component 中的
  indirect/external/unresolved/opaque call 都拒绝，不能通过两个 disconnected indirect targets 隐藏
  producer 与 TSTORE；call surface 闭合后才允许互不连通 component 复用地址。helper 固定运行在
  `PTOResolveBufferSelect` 后、`PTOInlineBackendHelpersPass` 前，普通 codegen 与 `--emit-pto-ir`
  共用该切点；这组保守规则没有 test-only escape；
- driver 在任何 backend/output 路由前递归发现 ND-to-2xNZ form 和 descendant `ModuleOp`；二者
  同时存在时，fixed-depth guard 不依赖 `isBackendPartitionedContainer()`，根 body 只接受 immediate
  backend child/direct function 的形态，nested `ModuleOp`/function、或一个 child 与顶层 function
  混合的根结构在所有 output mode 下稳定拒绝。通过 guard 的 backend-partitioned outer module 在
  child 拆分前完成 component precheck；local declaration 必须按 exact final-link symbol 穿透到唯一
  sibling public definition，`peer_func` 必须按 driver 的 exact/logical peer resolution 解析。outer
  module 含 partial producer 时，任意 child 中零/多匹配、opaque call、任何解析后的 cross-child
  direct call，或任意 `ImportReservedBufferOp` 都稳定失败，即使 link full-valid 且与 partial
  component 不连通；不能依赖 child declaration、peer clone、reserved-buffer address materialization、
  outer 预解析结果或最终链接阶段补救。outer module 无 partial producer 时，既有 full-valid cross-child
  direct call/peer link 行为保持不变；完全位于单 child 且没有跨 child direct call 或 peer import 的 partial component，以及
  call-surface-closed 的 disconnected child component 保持各自既定规则；
- A3/A5 至少各有一条 full-valid 端到端双输出数值链路；partial-valid/odd/`1x1` 只计入
  通过 simulator UB dump 或独立 raw-buffer harness 观测的 UB-only TEXTRACT
  coverage，不能直接进入 generic NZ TSTORE；NPU partial coverage 必须实际导出并逐 byte 比较
  紧贴 physical allocation extent 的 pre/post UB redzone，只有 GM guard 或没有可编译观测 helper 的设备
  只能计 compile-only/simulator coverage；A5 的 FP4/NZ+1 只有通过各自物理布局/设备验证后才进入
  verifier 正向集合；
- A5 RowPlusOne device golden 只定义和比较 logical payload。TEXTRACT raw-UB dump 忽略 gap，
  独立受控 TSTORE testcase 证明 gap 不被读取/导出，端到端链路比较 GM payload 与 guards；任何
  对 TEXTRACT 后 gap 具体值或 sentinel 保持状态的要求都不属于完成条件；
- 既有单输出 `pto.textract` canonical 文本、Python/C++ 调用、PTOBC v0 fixture、verifier、pipe、
  effects 和 EmitC 行为通过兼容回归；Python/ODS surface 中不存在新 op class 或 mnemonic。
