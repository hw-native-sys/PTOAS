# VPTO Soft Post-Update 优化 Pass 设计文档

> 独立的地址递推规范化、signed/unsigned i16 no-wrap 证明证书和共享候选描述见 [VPTO 地址递推 i16 规范化设计](vpto-address-recurrence-normalization-design-zh.md)。CLI pipeline 始终在本 pass 之前运行 `VPTONormalizeAddressRecurrences`，但两者没有 commit/rollback 合同：normalizer 的 canonical i16 recurrence 是永久 IR，即使本 pass 被关闭或拒绝某个候选也会保留。

## 1. 概述

本文档设计 PTOAS 的 `VPTOSoftPostUpdate` pass，在 **MLIR 层**（LLVM lowering 之前）将非 Post-Update 形式的 VPTO 访存操作转换为 Post-Update 形式。候选集合由 `VPTOPostUpdateUtils` 的共享表统一定义，包含第 2 节列出的 Mechanism A 与已接入的 stateful op，而不是 pass 内硬编码少数指令。pass 覆盖两种场景：`scf.for` 循环中能够符号化求出逐轮地址推进量的访存模式（循环路径，推进量可以是每轮可用的动态值），以及同一 block 单次执行中的 `SequentialRun`（顺序路径）。`SequentialRun` 指同一候选桶中按程序序连续、相邻候选的有效地址差均为同一个非零 `step` 的候选序列。

循环路径示例：

```mlir
// 变换前：偏移由归纳变量计算
scf.for %iv = %c0 to %N step %c64 iter_args(...) {
  %vec = pto.vlds %base[%iv] : !pto.ptr<f32, ub> -> !pto.vreg<64xf32>
}

// 变换后：指针通过 iter_args 循环携带
scf.for %iv = %c0 to %N step %c64 iter_args(..., %ptr = %base) {
  %vec, %new_ptr = pto.vlds %ptr[%c64] : !pto.ptr<f32, ub>
      -> !pto.vreg<64xf32>, !pto.ptr<f32, ub>
  scf.yield ..., %new_ptr
}
```

## 2. Post-Update 指令全景

下表首先列出当前 `getPostUpdateOpTable()` 注册的完整 PTOAS 候选集合；这 14 个 op 是 normalizer 和 soft-postupdate 实际共同处理的范围。Bisheng 的 Auto/Soft 分支只作为算法来源背景，不作为 PTOAS 候选集合的事实源。

bisheng 内部将候选指令分为两个处理分支：

- **Auto 分支**：指令有独立的 base 和 offset 操作数。pass 通过 SCEV 分析 `base + offset` 的地址递推，构造嵌套 PHI 来模拟多层循环的 base 递增。
- **Soft 分支**：指令无独立 offset 操作数（如 `vldus`），步长只能从 base 自身的 SCEV 演化中反推。

### 2.1 已实现 `updated_base` 的指令（Mechanism A）

这些指令的 `updated_base` 是 `Optional` 结果——有则为 Post-Update 形式，无则为普通形式。

| 分支 | PTOAS Op | 非 Post intrinsic | Post intrinsic |
|------|----------|-------------------|----------------|
| Auto | `pto.vlds` | `llvm.hivm.vldsx1.v{N}{ty}` | `llvm.hivm.vldsx1.post.v{N}{ty}` |
| Auto | `pto.vldsx2` | `llvm.hivm.vldsx2.v{N}{llvmTy}` | `llvm.hivm.vldsx2.post.v{N}{llvmTy}` |
| Auto | `pto.plds` | `llvm.hivm.plds.b8` | `llvm.hivm.plds.post.b8` |
| Auto | `pto.pldi` | `llvm.hivm.pldi.b8` | `llvm.hivm.pldi.post.b8` |
| Auto | `pto.vsts` | `llvm.hivm.vstsx1.v{N}{ty}` | `llvm.hivm.vstsx1.post.v{N}{ty}` |
| Auto | `pto.vsstb` | `llvm.hivm.vsstb.v{N}{llvmTy}` | `llvm.hivm.vsstb.post.v{N}{llvmTy}` |
| Auto | `pto.psts` | `llvm.hivm.psts.b8` | `llvm.hivm.psts.post.b8` |
| Auto | `pto.psti` | `llvm.hivm.psti.b8` | `llvm.hivm.psti.post.b8` |
| Auto | `pto.sprsts` | `llvm.hivm.sprsts` | `llvm.hivm.sprsts.post` |
| Auto | `pto.sprsti` | `llvm.hivm.sprsti` | `llvm.hivm.sprsti.post` |
| Auto | `pto.vstas` | `llvm.hivm.vstas` | `llvm.hivm.vstas.post` |
| Auto | `pto.vsldb` | `llvm.hivm.vsldb.v{N}{llvmTy}` | `llvm.hivm.vsldb.post.v{N}{llvmTy}` |

LLVM lowering 时根据 op 是否有 `updated_base` 结果来选择生成 post 或非 post intrinsic。

### 2.2 Stateful Post-Update 指令（Mechanism B：align 状态穿针）

这些指令始终返回更新后的 align 状态，同时通过可选的 pointer 结果区分普通形式和 Post-Update 形式。普通形式没有 pointer 结果；Post-Update 形式追加 `updated_base`/`base_out`，并选择对应的 post intrinsic。

| 分支 | PTOAS Op | 普通 intrinsic | Post intrinsic | 始终输出的状态 | Post-Update 增量 |
|------|----------|----------------|----------------|------------------|-----------------|
| Soft | `pto.vldus` | `llvm.hivm.vldus.v{N}{ty}` | `llvm.hivm.vldus.post.v{N}{ty}` | `updated_align` | 普通形式无显式 increment；Post 形式追加 index increment 和 `updated_base`。普通 intrinsic 的隐藏 base ptr 不映射为普通 VPTO 结果 |
| Auto/stateful | `pto.vstus` | `llvm.hivm.vstus.v{N}{ty}` | `llvm.hivm.vstus.post.v{N}{ty}` | `align_out` | 现有 i32 offset 保持为状态推进量且不参与当前访问地址；Post 形式追加 `base_out` |

### 2.3 Bisheng 调研中存在、但当前 PTOAS 未注册的指令

下列指令不在 `getPostUpdateOpTable()` 中，当前两个 PTOAS pass 都不会处理：

| 分支 | 指令 | 说明 |
|------|------|------|
| Auto | vldix1 | 向量 interleaved 加载 x1 |
| Auto | vldix2 | 向量 interleaved 加载 x2 |
| Auto | vstai | align 存储（interleaved） |
| Soft | vldui | 非对齐 interleaved 加载 |

## 3. 上移到 PTOAS 的收益分析

毕昇编译器（bisheng）已在 LLVM IR 层实现了 `hiipu-vf-soft-postupdate` pass。将此优化上移至 MLIR 层有以下具体优势。

### 3.1 相比 bisheng LLVM IR 层实现的优势

bisheng 的 pass 工作在 LLVM IR 上，此时高级循环结构和 PTO 类型信息已被擦除，导致以下问题：

**信息丢失：**

- **`scf.for` 结构消失。** 结构化循环变成 CFG 中的 phi 节点。bisheng 必须通过 `LoopInfo` 重建循环结构，并使用 `ScalarEvolution`（SCEV）分析地址递推模式。当 SCEV 无法将地址表达式展开为仿射 `AddRecExpr` 时，Auto 分支直接放弃变换（遗漏优化），Soft 分支则用 `VecLen`（向量长度）作为步长兜底（可能产生错误代码）。在 MLIR 中，`scf.for` 直接暴露归纳变量、上下界、步长和 `iter_args`，无需任何重建。

- **PTO 指针语义丢失。** LLVM lowering 后，`!pto.ptr<f32, ub>` 变成 `ptr addrspace(6)`，元素偏移被预乘为字节偏移（`%offset * 4`）。bisheng 必须从字节算术中逆向推导元素步长。在 VPTO 层面，偏移以元素为单位，元素类型是显式的。

**已知脆弱性（按分支）：**

- **Soft 分支**（影响 `vldus` 等 Mechanism B 指令）：PHI incoming 顺序敏感，pass 会交换 header PHI 的 incoming 值作为全局副作用；回边修复存在已知的 latch-write bug（当循环不被 `LoopInfo` 识别时，静态下标写到错误的 incoming 槽）；SCEV 失败时的 `VecLen` 兜底可能静默产生错误代码。

- **Auto 分支**（影响 `vlds`/`vsts`/`vsstb` 等首期目标指令）：显式排除 AIV 软件循环，使这些模式未被优化；为多层循环构造嵌套 PHI 链，代码复杂且依赖 incoming 的固定布局；SCEV 分析失败时直接放弃变换——不会出错，但遗漏优化。

**MLIR 层面的优势：** `scf.for` 保证了良好的循环结构，`iter_args` 提供显式的 SSA 语义。不需要 SCEV 分析，不需要构造 PHI，不存在 incoming 顺序问题，AIV 软件循环也使用同样的 `scf.for` 表示，从根源上消除了上述问题。

### 3.2 能覆盖 bisheng 遗漏的场景

| 模式 | bisheng | PTOAS（本方案） |
|------|---------|----------------|
| 简单 `scf.for` + IV 偏移 | ✓（通过 SCEV） | ✓（直接模式匹配） |
| 嵌套 `scf.for` 循环 | 部分支持（脆弱的 PHI 嵌套） | ✓（内层到外层处理，各层独立建立 `iter_args`） |
| 带 mask 的 `pto.vsts` | ✓ | ✓ |
| `pto.vsstb` 块步长存储 | 部分支持（独立开关） | ✓（统一框架） |
| AIV 软件循环 | ✗（排除，因为 AIV 软件循环不被 `LoopInfo` 识别为真实硬件循环，SCEV 无法分析） | ✓（`scf.for` 统一表示，无此限制） |
| 非循环顺序访问 | ✗（不支持） | ✓（顺序路径，见 4.3） |
| 循环体内单次迭代的展开访问 | ✗（不支持） | ✓（循环路径未命中的剩余 op 由顺序路径处理） |
| 显式 `arith.addi` 步长模式 | 可能 SCEV 失败 | ✓（直接匹配 `arith.addi`） |

## 4. 算法设计

### 4.1 整体流程

pass 的驱动分为两个阶段。两个阶段都通过 `VPTOPostUpdateUtils` 中的共享 `PostUpdateOpTable` 识别候选；前置地址规范化 pass 也消费同一张表，因此候选集合和地址语义只有一个事实源。normalizer 只用该表定位需要证明的 loop-varying 地址 leaf 及其 signed/unsigned 域，不计算完整 post stride；base/offset 合并、地址单位和最终 stride 约束只在 soft-postupdate 中分析一次。soft-postupdate 内部继续共享 `StrideExpr`、合法性检查和 post-update op 构造逻辑：

```
阶段一 · 循环路径：候选 op 直接位于某个 scf.for 的循环体内，并且在每次迭代中更新。
    以 ForOp 为单位批量处理（同循环内多 op 合并 iter_arg）：
      · 纯符号分析：直接读取 normalizer 已永久接入 operand 的 canonical recurrence；先做累加器分析，未命中再做 delta 分析。
      · 最终判定与 plan 准备：合并 base/stride delta，完成单位、类型和最终 stride 检查；通用 combine 路径同时拒绝零 stride，vstus 则检查 base advancement 与原 offset 等价。为可改写候选准备 stride 和 init_ptr。
      · 选择提交：只用成功 plan 建立 pointer chain；失败候选保持普通 op，但继续使用 canonical i16 recurrence。
        每个 scf.for 独立建立和清理自己的 pointer iter_arg；不把内层 updated_base 自动传播为外层 iter_arg。

阶段二 · 顺序路径：循环路径完成后，重新收集 vecscope 内的全部 block，
    包括 scf.for body、vecscope 体和 scf.if 分支体。只扫描仍为非 Post-Update
    形式的剩余 op；以 block 为单位分桶，桶内按程序序寻找互不重叠的
    最大 `SequentialRun`，并链式改写。
```

循环路径按内层到外层处理全部 `scf.for`。由于循环改写会 erase 并重建 ForOp，顺序路径必须在全部循环改写结束后重新收集 block 和候选 op，不能跨阶段保存 `Block *` 或 `Operation *`。循环路径已改写的 op 会因已有 `updated_base` 被自然跳过；循环路径未命中、最终约束失败以及原本位于 `scf.if` 等嵌套区域中的普通 op，都以当前 canonical 地址表达式在单次 block 执行范围内参与顺序路径。

### 4.2 循环路径

循环路径对 base 和 strideOperand **各自独立**分析：每个操作数先试 **累加器分析**（优先），未命中再退到 **delta 分析**（兜底），两者的结果最后按 4.2.1 的公式合并为 `stride_new`。前者处理该操作数已通过 `iter_args` 显式累加的场景（stride 可以是任意已计算的值），后者处理从 IV 全新计算、无累加器的场景（stride 须为循环不变量）。因此同一条指令的 base 走累加器、strideOperand 走 delta 是允许的组合。这三类分支没有被前置 pass 替代：normalizer 只把其中可能跨窄整数类型域的直接 IV 或固定步长 iter_arg leaf 证明并映射为 canonical i16，完整递推表达式仍在这里分析。

各类内存指令的分析和改写通过统一的地址描述符抽象，共享同一套分析流程。

无论走哪条路径，递推分解、delta、单位缩放、类型和最终 stride 检查都先只产出**符号表达式**。候选通过这些检查后才物化 stride 和初始指针；若后续初始指针物化仍失败，vecscope 级 dead-pure-op 清理会删除已产生但无用户的常量、cast、clone 或 `pto.addptr`（见 4.2.2）。

#### 4.2.1 地址描述符

每个候选 op 由 `PostUpdateOpInfo` 描述：base 与可选 strideOperand 的操作数下标、stride 是否参与本次访问地址、stride 的**单位**、stride 的 signed/unsigned 地址数值域、普通/post 形式的结果数边界，以及最终 post stride 约束。

```
enum class PostUpdateAddressUnit { Element, Block, Alignment, Byte };
enum class PostUpdateAddressDomain { Signed, Unsigned };
enum class PostUpdateStrideConstraint { Dynamic, Constant, SignedI8 };
enum class PostUpdateElementTypeSource { Base, Operand, Result };
struct PostUpdateOpInfo {
  unsigned baseOperandIdx;
  optional<unsigned> strideOperandIdx;
  bool strideParticipatesInCurrentAddress;
  PostUpdateAddressUnit addressUnit;
  unsigned minResultsForPost;
  PostUpdateAddressDomain strideDomain;
  PostUpdateStrideConstraint strideConstraint;
  PostUpdateElementTypeSource elementTypeSource;
  unsigned elementTypeIndex;
};
```

当前共享表中只有 `pldi`、`psti` 要求最终 stride 为编译期常量，`sprsti` 要求最终常量落在 signed i8 范围；其余候选使用 `Dynamic`，允许满足目标 operand 类型的动态 stride。

Element、Byte、Alignment 类 stride 使用 signed 域；Block 类 `vsldb/vsstb` stride 在地址计算中按 unsigned i16 位模式解释。base 的 `pto.addptr` offset 始终使用 signed index 域。

`delta(base)` 与 `strideOperand` 的单位不同，合并前必须先统一到字节。引入两个字节量：

- **`elemBytes`** = base 指针一个 `pto.addptr` 单位的字节数（由 `addPtrUnitBytes(base)` 按 PTO storage element byte size 求得，与 lowering 规约后的 GEP 元素宽度一致；无法映射为整字节存储单位的类型直接放弃候选）。
- **`unitBytes`** = strideOperand 一个单位的字节数。对于 Element 单位，地址描述符还明确指定元素类型来自 base、某个 operand 或某个 result；其余单位直接由 Block、Byte 或 Alignment 规则决定。该信息必须与 op lowering 缩放 offset 时采用的类型一致：

| 指令 | base | strideOperand | strideUnit | unitBytes | 有效地址 |
|------|------|---------------|-----------|-----------|---------|
| vlds | source | offset (Index) | Element | result 元素字节数 | base + offset |
| vldsx2 | source | offset (Index) | Element | low/high result 元素字节数 | base + offset |
| vsts | destination | offset (Index) | Element | destination 元素字节数 | base + offset |
| vsstb/vsldb | destination/source | repeat_stride (I16) | Block | 32 | dest + (32/elemBytes)·repeat_stride |
| plds/psts | source/destination | offset (Index) | Byte | 1 | base + offset/elemBytes |
| pldi | source | offset (Index) | Alignment | NORM: VL/8；US: VL/16；DS: min(32, VL/4) | base + (unitBytes/elemBytes)·offset |
| psti | destination | offset (Index) | Alignment | NORM: VL/8；PK: VL/16 | base + (unitBytes/elemBytes)·offset |
| sprsts | destination | offset (I32) | Byte | 1 | dest + offset/elemBytes |
| sprsti | destination | offset (I32) | Alignment | AR: 4 | dest + (4/elemBytes)·offset |
| vstas | destination | offset (I32) | Element | elemBytes | dest + offset |
| vldus | source | 无 | Element | result 元素字节数 | base；increment 由 base advancement 得出 |
| vstus | base | offset (I32) | Element | value 元素字节数 | base（offset 只推进返回 base，不参与本次访问） |

> intrinsic 参数原样透传不能单独证明硬件地址单位；Step 4 的 immediate/scalar 差异以 CANN 9.1 SIM 的实际更新地址为准。
> 上表中的 VL 以字节计；A5 的 VL 为 256 bytes，因此 pldi 的 NORM/US/DS 分别为 32/16/32 bytes，psti 的 NORM/PK 分别为 32/16 bytes。其他目标必须提供自己的查询结果，否则该候选不改写。

分析和改写的核心公式统一以字节表达：

```
Δbytes     = elemBytes·delta(base) + unitBytes·delta(strideOperand)
stride_new = Δbytes / unitBytes = (elemBytes/unitBytes)·delta(base) + delta(strideOperand)
init_ptr   = pto.addptr(base_0,  (unitBytes/elemBytes)·strideOperand_0)   // 偏移以元素计
```

该通式适用于 `strideParticipatesInCurrentAddress = true` 的普通候选。`vldus` 没有显式 stride，直接以 base advancement 作为 `stride_new`；`vstus` 的 offset 不计入 `init_ptr`，且只有在它与 base advancement 相等时才建立 post base chain。

展开后可见**只有 `delta(base)` 被缩放,`delta(strideOperand)` 恒等透传**——这保证了 stride 保持符号形式,循环变化的增量对所有指令类别都仍受支持;当 `elemBytes == unitBytes`(Element 类)缩放因子为 1,不发射任何 IR,vlds/vsts 的行为与未引入缩放前逐字相同。

约束（由 4.2.5 检查 3 统一裁定）：`(elemBytes/unitBytes)·delta(base)` 须为精确的整数缩放。`unitBytes % elemBytes == 0` 时，将 `delta(base)` 规范化为“常量 + Σ(系数 × SSA 叶子)”，要求常量和每个系数都能被 `unitBytes/elemBytes` 整除；`elemBytes % unitBytes == 0` 时乘 `elemBytes/unitBytes` 恒精确；互不整除则放弃。`init_ptr` 的 `(unitBytes/elemBytes)·strideOperand_0` 对称：Block 类恒精确（elemBytes 整除 32），Byte 类要求 `elemBytes | strideOperand_0`。

Element 类 index offset 在构造 `init_ptr` 时还会显式复现原访存 lowering 的 index→i32 截断，再 signed-cast 回 index 后交给 `pto.addptr`。这保证 post-update 首地址与普通形式的既有 ABI 行为一致，而不是无意中把首地址计算扩大为 64-bit index 语义。

#### 4.2.2 分析与物化分离

循环路径先进行符号分析，再进行物化。`decomposeLinear`、`getIterArgIncrement`、`computeDelta` 及 stride 的单位、类型和约束检查均为纯分析，返回符号表达式 `StrideExpr`。候选通过这些检查后，由 `materialize` 发射 stride，再由 `computeInitialPtr` 物化首轮地址。初始指针计算仍可能因定义链无法移到 loop entry 或单位换算无法精确表达而失败；这种晚期失败留下的无用户 pure 定义会在循环阶段结束后的 `eraseDeadPureOps(vecscope)` 中删除。

```
StrideExpr := Const(int64)                  // 类型在物化时按上下文决定
            | Leaf(Value)                   // IR 中已存在的叶子值
            | Add(e, e) | Sub(e, e) | Mul(e, e)
            | Cast(op, e)                   // index_cast/index_castui/extsi/extui，op 为克隆模板
```

`StrideExpr` 在构造时即做常量折叠（`foldConst`），因此 `delta(base)` 缩放是否精确、`stride_new` 是否恒为零，全部在符号层判定，判定不依赖任何已生成的 IR。

该划分是正确性要求，而非代码组织风格。以下三条性质依赖于它：

1. **支配性由构造保证。** 物化自底向上进行，每个新建 op 的操作数要么是刚发射的子表达式，要么是已通过可用性检查的叶子，因此结果天然支配插入点。

   可用性检查（4.2.5 检查 6）本身也依赖分析不建 IR。分析期间考察的每个 `Value` 都先于本次变换存在；对这样的值，"定义点早于插入点"可**传递地**推出"其操作数也早于插入点"，于是只需检查表达式的叶子。而对于变换过程中新建的 op，这条蕴含关系并不成立——它自身位置合法，不代表它的操作数在该位置可用。把 IR 生成推迟到分析之后，正是为了让被检查的值始终落在前一种情形里。

2. **递归可 memoize。** 纯函数的结果只取决于入参，`decomposeLinear` 与 `computeDelta` 按 `Value` 缓存，共享子表达式只分解一次，分解代价与定义链 DAG 的规模成线性关系。

3. **被拒候选不在 pass 输出中留残留。** 大部分拒绝发生在纯分析期；初始指针计算造成的晚期拒绝可能暂时留下 pure IR，但 vecscope 级清理会在顺序路径重新收集 block 之前删除这些无用户定义。`elemBytes/unitBytes` 的 stride 缩放仍只存在于符号层，不会为已拒候选单独落地。

#### 4.2.3 累加器分析（优先）

对 base 和 strideOperand 分别检查是否为 `scf.for` 的 block argument（来自 `iter_args`），且对应的 `scf.yield` 值可分解为 `blockArg + increment`。

分解通过递归线性分解实现：沿 `arith.addi`/`arith.subi`/`arith.muli`/`arith.index_cast`/`pto.addptr` 定义链递归，将 yield 表达式分离为 `blockArg * coeff + increment`。要求 `coeff == 1`（保证等差递推），`increment` 不要求是常量或循环不变量。

```
decomposeLinear(Value v, BlockArgument blockArg) -> {coeff, StrideExpr increment}:
  v == blockArg           → {1, Const(0)}
  v 是循环不变量或其他 block arg → {0, Leaf(v)}
  v = addi(a, b)         → {ca + cb, Add(ia, ib)}
  v = subi(a, b)         → {ca - cb, Sub(ia, ib)}
  v = muli(a, b)（一侧不含 blockArg 且为常量 k）→ {c * k, Mul(i, Const(k))}
  v = addptr(ptr, offset) → {c_ptr, Add(i_ptr, Leaf(offset))}
  v = index_cast(a)       → cast 可安全穿过时 {ca, Cast(op, ia)}，否则 unknown
  其他                    → unknown（放弃）
  结果按 v 缓存

getIterArgIncrement(Value v, ForOp forOp) -> {status, StrideExpr}:
  沿 v 的定义链回溯，穿过可安全传递 delta 的 index_cast（记录类型转换）、
  addi/subi/addptr
  与循环不变量的组合（跳过偏移），直到找到 iter_arg BlockArgument。
  对该 iter_arg 的 yield 操作数调用 decomposeLinear：
    coeff == 1            → {Ok, 路径上收集的类型转换套用于 increment}
    分解失败或 coeff != 1 → {Failed, -}
  未回溯到 iter_arg       → {NotIterArg, -}
```

三态返回是必要的：`NotIterArg` 表示该操作数与累加器无关，应回退 delta 路径；`Failed` 表示确实是 iter_arg 但增量无法分解，此时必须整体放弃——把未知增量当作 0 会静默算错地址。

累加器路径和 delta 路径对 cast 使用同一条正确性条件：必须能够证明 `delta(cast(x)) == cast(delta(x))`。循环不变量的 cast 的 delta 恒为零，可以直接保留；纯 index 域内且没有窄整数来源的递推继续使用本 pass 原有分析。loop-varying 窄整数和跨域 cast 只通过独立 normalizer 产生的永久 canonical contract 接入：

- Signed canonical recurrence 是 i16 iter_arg、signed extension（`arith.index_cast`/`arith.extsi`）和 `arith.addi/subi ... overflow<nsw>` 的组合；unsigned canonical recurrence 使用 `arith.index_castui`/`arith.extui` 和 `overflow<nuw>`。Block 类 stride 期望 unsigned，其余共享描述中的窄地址 stride 期望 signed。
- normalizer 直接把通过证明的 canonical 地址接到访存或 `pto.addptr` operand，并用 loop-aware liveness 删除不再需要的宽递推。soft-postupdate 不解包任何 marker。
- soft-postupdate 只检查 canonical 结构并读取常量 backedge step，不再计算 trip count、初值、递推端点或最终 backedge 范围；这些事实已经由 overflow flag 作为 IR 语义承诺。它仍负责合并 base 与 stride delta、单位换算以及 `Dynamic`、`Constant`、`SignedI8` 等最终约束。
- 原生 i16 iter_arg 若本身具有域匹配的 `nsw`/`nuw` constant-step backedge，也满足同一个 canonical 结构。raw i32 iter_arg、signed/unsigned 域不匹配的 cast、没有相应 overflow flag 的 i16 backedge、动态步长和复杂窄整数递推均返回 `Failed`。候选失败时不撤销 normalizer 的改写。

上述规则覆盖 `arith.index_cast`、`arith.index_castui`、`arith.extsi` 和 `arith.extui`；`arith.trunci` 等其他整数转换不属于 canonical 语法，遇到时保守放弃。

```
baseIncr = getIterArgIncrement(desc.base, forOp)      // NotIterArg → 回退 delta
soIncr   = getIterArgIncrement(desc.strideOperand, forOp)

任一为 Failed → 放弃该候选
否则 → stride_new = (elemBytes/unitBytes)·delta(base) + delta(strideOperand)   // 见 4.2.1，符号相加
```

不要求 stride 是常量或循环不变量；增量也不要求定义在候选 op 之前——不满足时由 4.2.5 的可用性检查决定克隆或放弃。

**示例（vlds，Element 类，elemBytes == unitBytes）：**

```mlir
%base = pto.castptr %c0_i64 : i64 -> !pto.ptr<f32, ub>
scf.for %iv = %c0 to %c16 step %c1
    iter_args(%off = %c0) -> index {
  %s = arith.addi %iv, %c1 : index
  %vec = pto.vlds %base[%off] : ...
  %next_off = arith.addi %off, %s : index
  scf.yield %next_off
}
```
`baseIncr` = `NotIterArg`（回退 delta，得 `Const(0)`），`soIncr` = `{Ok, Leaf(%s)}`。elemBytes == unitBytes，`stride_new = (elemBytes/unitBytes)·0 + %s = %s`。

注意 `%s` 在本例中定义于 `pto.vlds` 之前，但这不是前提：若 `%s` 定义在 `pto.vlds` 之后，4.2.5 会把它的定义链克隆到候选 op 之前，结果不变。

#### 4.2.4 delta 分析（累加器未命中时）

对 base 或 strideOperand 中任一个未追溯到 iter_arg、即 accumulator 分析返回 `NotIterArg` 的操作数，独立回退到 delta 分析。因此同一候选可以是 base 使用 accumulator、strideOperand 使用 delta，也可以反过来；只有返回 `Failed` 的操作数会使整个候选停止。

定义 `delta(v)` 为值 `v` 在 `scf.for` 每次迭代间的变化量：

| `v` 的类型 | `delta(v)` 的值 |
|---------|-------|
| `v` 是 IV | `step`（`scf.for` 的步长） |
| `v` 是常量或循环不变量 | `0` |
| `v = arith.addi(a, b)` | `delta(a) + delta(b)` |
| `v = arith.subi(a, b)` | `delta(a) - delta(b)` |
| `v = arith.muli(a, b)`，其中一个循环不变 | `invariant * delta(other)` |
| `v = arith.index_castui(a)`、`arith.index_cast(a)`、`arith.extui(a)` 或 `arith.extsi(a)` | 循环不变量为零；loop-varying 扩展仅在来源为域匹配且带 `nuw/nsw` backedge 证书的 canonical i16 recurrence 时为 `cast(delta(a))`；loop-varying narrowing 以及其他来源为 `unknown` |
| 其他 | `unknown`（放弃） |

```
stride_new = (elemBytes/unitBytes)·delta(base) + delta(strideOperand)   // 见 4.2.1
```

`stride_new` 须为循环不变量，`(elemBytes/unitBytes)·delta(base)` 须为精确整数缩放（见 4.2.1 约束）。

**正确性：** delta 表中的操作构成仿射函数的封闭运算集合。定义链仅由这些操作构成时，delta 计算不会遗漏。遇到表外操作时保守放弃。窄整数 cast 必须满足 4.2.3 的 canonical contract。

delta 分析同样是纯符号的：表中每一行返回 `StrideExpr`，结果按 `Value` 缓存。

#### 4.2.5 合法性检查

无论由累加器分析还是 delta 分析产出 stride，都须满足以下条件。这些 stride 合法性检查在 post-update op 改写之前完成，任一不满足即放弃该候选。放弃只表示保留普通访存形式，不会撤销 normalizer 已经完成的 i16 canonicalization。候选通过 stride 检查后，初始指针计算仍可能晚期失败，此时由 vecscope 级 dead-pure-op 清理保证最终不留下本 pass 自己物化的无用 IR。

1. **op 尚未处于 Post-Update 形式。** 通过共享描述中的 `minResultsForPost` 判断当前结果数是否已经包含 optional pointer 结果。

2. **op 直接位于 `scf.for` 循环体内**（不嵌套在循环内的 `scf.if` 或其他控制流中），避免部分迭代问题。

3. **`(elemBytes/unitBytes)·delta(base)` 为精确整数缩放。** Element 类（elemBytes == unitBytes）自动满足；Block 类（`unitBytes % elemBytes == 0`，如 unitBytes=32）要求规范化 `delta(base)` 的常量项和每个 SSA 叶子系数都能被 `unitBytes/elemBytes` 整除；Byte 类（`elemBytes % unitBytes == 0`）乘 `elemBytes/unitBytes` 恒精确；互不整除则放弃。

   关键在于**只有 `delta(base)` 参与缩放**：`delta(strideOperand)` 恒等透传、不要求整除也不要求是常量，因此循环变化的 strideOperand 增量对各类指令都受支持（如 vsstb 的 `repeat_stride` 为非常量 iter_arg 累加）。仿射规范化使 `8*k` 这类非常量 base delta 也能证明可被 8 精确缩放。

4. **通用 combine 路径的 stride_new 为零时跳过。** 对 stride 参与当前地址的候选以及无显式 stride 的 `vldus`，地址不前进时 post-update 无收益。常量折叠使各项相消、合成结果恒为零的情形（如 base 每轮 +8、strideOperand 每轮 −8）同样能被识别。`vstus` 不经过 `combineStride` 的零值检查，而是要求缩放后的 base delta 与保留的显式 advancement offset 仿射等价。

5. **类型一致性与目标类型精确性。** stride 表达式各子项须归结为同一类型，否则放弃，以免构造出 `arith.addi(index, i32)` 这类非法 op。非常量表达式的结果类型还必须与 op 的 strideOperand 类型一致；不能通过消除 cast 或改用窄类型算术来强行匹配。`Const` 项不参与子项类型约束，其类型在物化时直接采用 strideOperand 类型。

   此外，各 `Const` 项的数值须能被目标类型表示。`stride_new` 对块步长指令是窄整数（i16），超出范围的 stride 会构造出非法常量，因此在物化前一并检查并放弃。动态宽结果若没有完整值域证明，同样保守放弃。

6. **操作数可用性（支配性）。** stride 表达式的每个叶子须在候选 op 处可用：循环不变量、block argument、或定义点早于候选 op。若叶子定义在候选 op **之后**，仅当其定义链全部为 pure op 时克隆到候选 op 之前（克隆保留原结果序号，多结果 op 亦正确）；否则放弃。该检查以只读方式先行完成，克隆发生在物化阶段。

#### 4.2.6 改写

循环内候选逐个完成符号分析并准备可行 plan；全部候选收集完成后执行循环改写：

1. **准备成功 plan。** 叶子全部循环不变时将 stride 发射到循环外，否则发射到候选 op 之前。对 stride 参与当前地址的候选，计算 `init_ptr = pto.addptr(base_0, (unitBytes/elemBytes)·strideOperand_0)`；若无显式 stride、stride 不参与当前地址或初始偏移为零，则直接使用 `base_0`。传给 `pto.addptr` 前将最终偏移规范为 index；Block 单位保持无符号扩展，其他单位使用有符号扩展，Element 类 index offset 还执行 4.2.1 所述的 i32 截断。常量按 `(值, 类型)` 在同一循环内复用同一个 SSA 值——4.2.7 的分组按 stride 的 **Value 同一性** 判定，重复创建等值常量会把本可共享 `iter_arg` 的 op 拆成多组。
2. 新增指针类型的 `iter_arg`，初始值为 plan 中的 `init_ptr`。
3. 创建 Post-Update op：base 替换为 iter_arg 的 block argument。对于 `strideParticipatesInCurrentAddress = true` 的候选，将 `strideOperand` 替换为 `stride_new`；`vldus` 原本没有显式 stride，创建 Post 形式时追加 increment；`vstus` 则保留原 i32 offset，只追加 `base_out`。其余操作数（block_stride、mask、dist 等）不变。
4. 将 `updated_base` 通过 `scf.yield` 传出。
5. 对改写后的循环做 loop-aware liveness，删除已经被 Post-Update 指针链完全取代的地址 accumulator，包括不再使用的 canonical i16 recurrence、cast 和 backedge。没有候选成功时不重建循环，normalizer 的 canonical recurrence 保留。

最后一步不能只依赖普通 DCE。旧 accumulator 即使没有真实用户，仍会形成
`block argument → pure update → scf.yield → block argument` 的循环使用链，局部
DCE 无法从这个环中找到 `use_empty()` 的起点。改写因此从以下根节点反向计算
liveness：

- 有副作用 op 的操作数；
- 在循环外仍有用户的 `scf.for` result；
- 保守保留 region op 自身的 operands，以及嵌套 region 捕获的值。

当活跃值追溯到某个 `iter_arg` block argument 时，对应的 yield 值也加入
liveness，直到跨 backedge 达到不动点。随后重建 `scf.for`，只复制活跃的
`iter_arg` 及其纯定义链。若 accumulator 的最终 loop result 或循环内其他语义
仍可观察，则该 accumulator 保留；只有完全被 Post-Update 地址链替代的递推
才会删除。

**vsstb/vsldb 的硬件语义补充：**

非 Post-Update：`dest + 32*repeat_stride + blk*32*block_stride`
Post-Update：`dest_p + blk*32*block_stride`（repeat_stride 不参与存储地址），返回 `dest_p + 32*repeat_stride_p`

Post-Update 模式下 `repeat_stride` 从地址偏移变为指针前进量，因此初始指针需吸收原始偏移 `32*rs_0` 字节，即 `pto.addptr` 偏移 `(32/elemBytes)*rs_0` 个元素（见 4.2.1）。

#### 4.2.7 同一循环中的多个 Op

两个 op 能共享同一个 `iter_arg`，当且仅当它们走**同一条地址序列**——起点 `init_ptr`（由 `base_0`、`strideOperand_0` 和 `unitBytes` 决定，见 4.2.1）相同，且以字节计的步长相同。

理想的分组键是 `(init_ptr, byte_stride)`。但 `init_ptr` 不适合直接入键：分组按 **Value 同一性** 比较，而 `computeInitialPtr` 可能为每个候选各自物化一个 `pto.addptr`，起点数值相同也未必是同一个 SSA 值。因此改用决定地址序列的 SSA 量：分组键实际取 `(base, strideOperand, stride_new, unitBytes)`。normalizer 已经把 operand 直接规范到 canonical SSA 值，不再需要解开 marker。加入 `unitBytes` 可防止 f32 上数值相同的 Element 与 Byte stride 被误合并；该键可能把本可合并的组拆开，但不会合并字节递推不同的组。

同组的 op 共享一个 `iter_arg`，所有 op 使用同一个 pre-update 指针（block argument），不链式传递 `updated_base`。原因：同一迭代内同组 op 访问相同地址，链式传递会使后续 op 的地址偏移一个 stride。每组只需 yield 一个 `updated_base`。

因为键按 Value 同一性比较，stride 的物化必须保证等值常量复用同一 SSA 值（见 4.2.6 步骤 1）；否则语义相同的 op 会被拆成多组，退化为各自持有一个 `iter_arg`。

#### 4.2.8 嵌套循环

嵌套 `scf.for` 按内层到外层处理，这是因为重建外层循环会使预先收集的内层 op handle 失效。每一层只分析直接位于自身 body 的候选，并为该层成功候选独立添加 pointer `iter_arg`；内层首地址可以引用外层当前迭代中可用的值，但实现不把内层 `updated_base` 自动向外层传播。

### 4.3 顺序路径

顺序路径处理同一 block 单次执行中的等步长访问，包括非循环代码以及 `scf.for` body 中循环路径未命中的剩余 op。本节将同一 `(op 类型, rootBase)` 候选桶中按程序序连续、相邻候选的有效地址差均为同一个非零 `step` 的候选序列称为 `SequentialRun`。步长只需在本次 block 执行期间保持不变，可以是编译期常量，也可以是符号表达式。

顺序路径复用循环路径的 `StrideExpr`、`scaleBaseDelta` / `combineStride`、类型与可用性检查、pure 定义链克隆、初始指针计算和 post-update op 构造。新增逻辑仅包括 base 归一化、以 SSA leaves 为变量的仿射规范化以及 block 内 `SequentialRun` 检测。

#### 4.3.1 有效地址分析

为比较 base 不同的候选，先沿 `pto.addptr` 链把每个 base 归一为：

```
base_i = pto.addptr(rootBase, baseOffset_i)   // baseOffset_i 以元素计
```

- 若 base 不是 `pto.addptr` 的结果，则 `rootBase = base`、`baseOffset = 0`。
- 若 base 来自 `pto.addptr`，则从它开始反向剥离连续的 `pto.addptr` 链：每经过一层就把该层 offset 累加到 `baseOffset`，并把该层输入指针作为新的待检查 root；直到 root 不再由 `pto.addptr` 定义，或下一层 `pto.addptr` 的元素单位与当前 base 不同。停止时的指针即 `rootBase`。因此只归一化元素单位一致的链，不跨越单位变化猜测换算关系。

例如：

```mlir
%root = arith.select %cond, %lhs, %rhs : !pto.ptr<f32, ub>
%p1 = pto.addptr %root, %a
%p2 = pto.addptr %p1, %b
%v = pto.vlds %p2[%c] : ...
```

在三者元素单位一致时归一化为：

```
rootBase   = %root
baseOffset = %a + %b
```

随后将 `baseOffset` 与指令自身的 `%c` （strideOperand）一起用于有效地址 step 分析。`baseOffset` 和 strideOperand 沿与循环路径相同的 `arith.addi`、`arith.subi`、常量乘法和 index cast 规则构造成 `StrideExpr`；无法继续分解的 SSA 值作为叶子保留。

对同一 op 类型、同一 `rootBase` 的相邻候选 `i-1` 和 `i`，定义：

```
deltaBase   = baseOffset_i - baseOffset_(i-1)       // 元素
deltaStride = strideOperand_i - strideOperand_(i-1) // strideOperand 单位

step = combineStride(deltaBase, deltaStride, elemBytes, unitBytes)
     = (elemBytes/unitBytes)·deltaBase + deltaStride
```

`step` 始终以该 op 的 strideOperand 单位表示。单位换算和精确缩放直接复用 4.2.1 与 4.2.5 的规则，不能证明精确换算时放弃该相邻关系。

单位可精确换算并不等于 `step` 可安全写入 strideOperand 的目标类型。顺序路径在分析和物化时保留原表达式中的 `index_cast` / `index_castui`，不会为了匹配目标类型而消除 cast、再用较窄的输入类型重新计算外层算术。常量 `step` 仍按目标类型的可表示范围检查；对于结果类型宽于目标类型的动态表达式，当前 pass 不做值域证明，因此保守放弃该 `SequentialRun`。

例如 `%k: i16` 经 `arith.index_castui` 零扩展为 index 后，若相邻 f32 地址差为 `16 * zext(%k)` 个元素，则 block step 是 `2 * zext(%k)`。当 `%k = 40000` 时原 step 为 80000；若先消除 cast 并在 i16 中计算 `2 * %k`，结果会回绕成 14464，改变访问地址。因此这种宽动态 step 不会改写为 i16 `repeat_stride`。

为比较两个非常量 `step` 是否相同，将 `StrideExpr` 按加减、常量乘法进行展平和常量折叠，得到规范化仿射形式“常量 + Σ(系数 × SSA 叶子)”。这里的“仿射”是相对于 SSA 叶子而言，并不要求叶子本身是原始输入的仿射函数。相同叶子按 Value 同一性合并，系数为零的项删除。两个 step 的规范化形式完全相同，才视为固定公差；不支持的非仿射运算保守地作为叶子，仅在复用同一 SSA 值时才能匹配。两个独立计算但语义上可能相等的非仿射结果是不同 Value，pass 不尝试证明它们等价。

例如 `%s` 由 `arith.select` 产生，虽然该运算不属于上述仿射语法，但下面三个 offset 的相邻 step 都可规范化为同一个 opaque leaf `%s`，因此仍能形成 `SequentialRun`：

```mlir
%s = arith.select %cond, %lhs, %rhs : index
// offset: 0, %s, %s + %s
```

若相邻 step 分别依赖两个独立的非仿射结果 `%s0` 与 `%s1`，即使两者运行时可能相等，也不会匹配，除非此前的规范化或 CSE 已使它们成为同一个 SSA Value。

该分析可识别 base 与 offset 相互补偿的模式。例如 Element 类指令的三个候选满足：

```
baseOffset:  x,      x + 16, x + 32
offset:     -x,     48 - x, 96 - x
有效偏移:    0,          64,      128
```

虽然 base 和 offset 都不是常量，但相邻地址的规范化 `step` 恒为 64。

#### 4.3.2 分桶与 `SequentialRun`

在一个 block 内，将仍未改写的候选按 `(op 类型, rootBase)` 放入有序桶。不同桶的 op 可以在物理程序序中任意交错；每个桶只保留本桶候选的原始程序序。

`SequentialRun` 对应代码中的同名结构。无法再向后加入同 `step` 候选的 `SequentialRun` 称为最大 `SequentialRun`。

每个桶使用一次确定性的线性扫描，产生互不重叠的最大 `SequentialRun`：

1. 以当前候选作为 `SequentialRun` 起点，下一候选与它形成第一个非零、合法的 `step`。
2. 后续候选与前一候选的 `step` 若和当前 `SequentialRun` 的规范化 step 相同，则加入该 `SequentialRun`。
3. 若 step 不同或无法分析，则当前候选 `SequentialRun` 立即结束；破坏公差的候选不加入该 `SequentialRun`。
4. 尚未形成合法 step 的两个候选若无法配对，丢弃前一个，以后一个重新尝试。
5. `SequentialRun` 长度达到 2 时即可确定 `step`，但长度至少为 3、通过 4.3.3 的合法性检查并满足 4.3.4 的收益性条件后才接受并改写。接受后，破坏公差的候选作为下一个 `SequentialRun` 的起点；拒绝后，从该候选 `SequentialRun` 的最后一个候选（`end - 1`）重新尝试，使其可与破坏公差的候选组成新 `SequentialRun`。
6. 只有已接受 `SequentialRun` 的候选会被消费，且不再参与后续 `SequentialRun`，因此最终接受的 `SequentialRun` 互不重叠。被拒 `SequentialRun` 最多复用一个尾候选，不进行内部回溯或最长子序列搜索。

该规则不追求全局最多命中，而是保证结果简单、线性、确定且符合程序序。

```mlir
// vlds 桶：中间可以穿插其他桶的 op
%v0 = pto.vlds %base0[%off0] : ...  // 本桶地址 0
pto.vsts %x, %other[%c0], %mask : ...
%v1 = pto.vlds %base1[%off1] : ...  // 本桶地址 64
%v2 = pto.vlds %base2[%off2] : ...  // 本桶地址 128
// 本桶形成一个 step = 64 的最大 SequentialRun
```

> **与循环路径的关键区别（链式 vs 共享）。** 循环路径同组 op 共享一个 `iter_arg`、全用 pre-update 指针、**不**链式传递（4.2.7，同迭代内同地址）。顺序路径相反，**必须**链式传递（前一条的 `updated_base` 喂后一条，见 4.3.5）。因此两条交错的序列（vlds 链 + vsts 链）是两条**独立**的链，各自穿针，不能合并；按 `(op 类型, rootBase)` 分桶使每条链自然独立。

#### 4.3.3 合法性检查

1. **op 尚未处于 Post-Update 形式。** 循环路径已经改写的 op 自动排除。
2. **地址可归一到同一 rootBase。** 只穿过 `pto.addptr`；其他指针变换不猜测别名关系。
3. **step 可精确换算、可按目标类型精确物化且非零。** 复用 4.2.5 的单位与常量折叠规则；常量必须能由目标类型表示，动态表达式必须在保留全部 cast 后与 strideOperand 类型一致。结果类型更宽且没有值域证明时放弃，禁止通过消除 cast 把外层算术降到窄类型。
   对 `vstus` 这类 stride 不参与当前地址的 stateful op，run 的地址 step 只来自 base；此外要求除尾 op 外每个候选原有的 offset 都与该 step 的规范化仿射形式完全相同，因为 post 形式会保留这些 offset 作为实际状态推进量。
4. **step 在 `SequentialRun` 头部可用。** 复用 4.2.5 的叶子可用性检查；定义在 `SequentialRun` 头部之后的 pure 定义链（包括产生 opaque leaf 的非仿射运算）可克隆到头部之前，含副作用、不可安全提前或无法支配的定义使该 `SequentialRun` 放弃。
5. **整条 `SequentialRun` 先分析、后物化。** 所有候选通过检查后才创建常量、`pto.addptr` 或新 op，拒绝 `SequentialRun` 不留下残余 IR。

顺序路径不重排 op，也不改变访问地址和访问顺序，因此不同候选之间的内存读写不会截断 `SequentialRun`，无需额外别名分析。

#### 4.3.4 收益性检查

长度至少为 3 只是合法性门槛。合法的 `SequentialRun` 还必须命中下列两类结构信号之一。共享表注册的全部候选都使用相同规则，不按 op 类型建立收益性白名单；`pto.vlds`、`pto.vsts` 和 `pto.vsstb` 只是对应测试中的典型例子。

**动态多层 base chain。** 当规范化后的 `step` 是编译期常量时，从第二个候选开始统计改写后确定死亡的动态 `pto.addptr`。某层 addptr 的 offset 经 4.3.1 的仿射规范化后包含 SSA 项，才视为动态；纯常量 addptr 不计收益。addptr result 必须只有下一层待删除 addptr 或当前候选这一个用户，第一条候选的 base chain 因仍用于构造 `init_ptr` 而不计收益。使用 `DenseSet` 对共享定义去重，并按下式接受：

```
deadDynamicAddPtrs > pointerEdges + initPtrCost

pointerEdges = runLength - 1
initPtrCost  = stride 参与当前地址、存在 strideOperand 且其不为常量零 ? 1 : 0
```

即使尾指令使用 normal 形式，它仍消费倒数第二条返回的指针，因此长度为 `N` 的 run 仍有 `N-1` 条 pointer edge。

**direct symbolic leaf。** 所有候选直接使用同一个 `rootBase`，首条 effective offset 为零，规范化 step 恰为 `1 × SSA atom`（常量项为零、只有一个系数为 1 的 leaf/cast atom），且第三条及以后 offset 的累计 `arith.addi` / `arith.subi` 定义链在改写后确定死亡。定义链若有 run 外用户则拒绝。step 已支配 run 头时直接复用；定义在 run 头之后时，仅当 pure 定义可安全克隆且原定义链随地址 use 一起死亡、不会增加重复计算时才接受。该类别没有额外的长度阈值：任意满足合法性最小长度 `N >= 3` 的 run 都可改写。

除此之外均保守拒绝。当前实现不为一般 `arith.addi`、`arith.muli` 或常量 `pto.addptr` 设置主观权重，也不因这些 op 出现在地址表达式中就假设最终 ISA 一定减少。收益性检查只读，不创建 IR；被拒 run 不留下常量、clone 或 `pto.addptr`。

#### 4.3.5 改写

对每个 `SequentialRun`，`stride_new` 就是 4.3.1 得到的规范化 `step`。初始指针直接从 `SequentialRun` 首条 op 的实际操作数计算：

```
if strideParticipatesInCurrentAddress:
  init_ptr = pto.addptr(first.base,
                        (unitBytes/elemBytes)·first.strideOperand)
else:
  init_ptr = first.base
```

这里复用循环路径的精确单位换算；不需要从 `rootBase` 重新构造绝对地址。对长度为 `N` 的 `SequentialRun`，前 `N-1` 条替换为 post-update 形式，前一条的 `updated_base` 作为后一条的 base。stride 参与当前地址时把 strideOperand 替换为 `stride_new`；`vldus` 追加该 increment；`vstus` 保留已经验证为等于 run step 的原 offset。最后一条只需访问当前指针，不再产生后续用户，因此保留 normal 形式：

```mlir
// vlds 变换后
%v0, %ptr1 = pto.vlds %init_ptr[%c64] : ... -> ..., !pto.ptr<f32, ub>
%v1, %ptr2 = pto.vlds %ptr1[%c64] : ... -> ..., !pto.ptr<f32, ub>
%v2        = pto.vlds %ptr2[%c0] : ... -> ...
```

最后一条 normal op 保留 attributes 和原始 result types，不追加 `updated_base`。对于 stride 参与当前地址的候选，base 改为当前 pointer、strideOperand 改为同类型零值；`vlds`、`vsts` 与 `vsstb` 都属于这种情况。`vldus` 的普通尾 op 不带 increment；`vstus` 的 offset 不参与当前地址，因此普通尾 op 保留原 offset。所有 `SequentialRun` 在只读分析阶段确定后，先物化各自的 `stride_new`、同类型零值和 `init_ptr`，再按原程序序替换候选 op，避免 erase op 使其他 `SequentialRun` 保存的表达式失效。不同桶各自维护独立指针链，即使其 op 在原 block 中交错也互不影响。

## 5. Pass 集成

### 5.1 在 Pipeline 中的位置

独立 normalization 固定运行，optional soft-postupdate 紧随其后；两者位于 `PTOInferVPTOVecScope` 之后、loop invariant code motion 之前。完成 VPTO emission IR 校验后，LLVM emitter 才运行 `PrepareVPTOLLVMLoweringPass` 和 `LowerVPTOOpsPass`：

```
VPTOExpandWrapperOps
PTOInferVPTOVecScope
→ VPTONormalizeAddressRecurrences
→ VPTOSoftPostUpdate
LoopInvariantCodeMotion
PTONarrowVPTOLoopCounters
Canonicalizer
CSE
PTOValidateVPTOEmissionIR
...
PrepareVPTOLLVMLoweringPass
LowerVPTOOpsPass
```

PTOAS 的 VPTO 后端始终运行 normalization，且不提供单独 CLI 开关。`--enable-vpto-soft-postupdate=false` 只关闭 soft-postupdate。由于同一 post-update 优化不应在 MLIR 与 LLVM 层重复执行，PTOAS 调用 Bisheng 编译 VPTO device LLVM IR 时默认显式关闭 Bisheng 公开 LLVM 选项 `-mllvm --cce-vf-enable-auto-postupdate=false` 与 `-mllvm --cce-vf-enable-blockldst-auto-postupdate=false`。只有诊断或对照场景显式指定 `--enable-bisheng-soft-postupdate` 时，PTOAS 才会把上述两个公开选项改为 `true`，重新开启 Bisheng 的普通 vector load/store 与 block load/store 自动 post-update 路径。

默认行为由两层测试约束：lit 直接对 runtime case 的 `kernel.pto` 分别检查 permanent normalization、post form 和拒绝形态，`test/vpto` 再通过 simulator 或 NPU 对完整输出做 `COMPARE_STRICT=1` 比较。重点场景包括源类型/i16 域回绕拒绝、i16/i32 正例、同循环中 post-update 成功与失败候选并存、负向递推、嵌套循环、共享 chain 以及不同地址单位隔离。

该位置确保：
- Wrapper op 已展开（IR 干净）
- **`pto.vecscope` 已存在**。pass 只改写 `pto.vecscope` 内的 op，而多数 kernel 的 vecscope 是由 `PTOInferVPTOVecScope` 创建的——排在它之前会让 pass 对这类输入静默失效。手写 vecscope 的测试用例不会暴露该问题，因此回归测试中必须包含不含手写 vecscope 的输入（`soft_postupdate_inferred-vecscope.pto`）
- Post-Update 结果对 LLVM lowering 可见，后者根据 `getUpdatedBase()` 选择 `vldsx1` 或 `vldsx1.post`

`PTOInferVPTOVecScope` 以整 op 方式把 `scf.for` 搬入 `pto.vecscope`（`wrapCluster` 用 `splice` 移动整个 op），循环体内部结构不变，因此 4.2.5 检查 2「op 直接位于循环体内」的判定不受影响。

### 5.2 Pass 注册

```tablegen
// 在 include/PTO/Transforms/Passes.td 中
def VPTONormalizeAddressRecurrences
    : Pass<"vpto-normalize-address-recurrences", "ModuleOp"> {
  let summary = "Normalize proven VPTO address recurrences to i16";
  let dependentDialects = ["mlir::func::FuncDialect",
                           "mlir::pto::PTODialect",
                           "mlir::arith::ArithDialect",
                           "mlir::scf::SCFDialect"];
}

def VPTOSoftPostUpdate : Pass<"vpto-soft-postupdate", "ModuleOp"> {
  let summary = "Convert fixed-stride VPTO memory ops to post-update form";
  let dependentDialects = ["mlir::func::FuncDialect",
                           "mlir::pto::PTODialect",
                           "mlir::arith::ArithDialect",
                           "mlir::scf::SCFDialect"];
}
```
