# VPTO Soft Post-Update 优化 Pass 设计文档

## 1. 概述

本文档设计一个新的 PTOAS pass——`VPTOSoftPostUpdate`，在 **MLIR 层**（LLVM lowering 之前）将非 Post-Update 形式的 VPTO 访存操作转换为 Post-Update 形式。首期目标指令为已支持 `updated_base` 的三个 op（`pto.vlds`、`pto.vsts`、`pto.vsstb`），后续扩展至更多访存指令（见第 2 节指令全景）。pass 覆盖两种场景：`scf.for` 循环间的固定步长访存模式（循环路径），以及同一 block 单次执行中的 `SequentialRun`（顺序路径）。`SequentialRun` 指同一候选桶中按程序序连续、相邻候选的有效地址差均为同一个非零 `step` 的候选序列。

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

下表列出 bisheng `hiipu-vf-soft-postupdate` 支持的所有指令与 PTOAS 现状的交叉对比。

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

### 2.3 Stateful Post-Update 指令（Mechanism B：align 状态穿针）

这些指令通过显式的 align 寄存器跟踪状态，**始终返回**更新后的 align，没有非 Post 形式。

| 分支 | PTOAS Op | Intrinsic | 输出状态 | 备注 |
|------|----------|-----------|---------|------|
| Soft | `pto.vldus` | `llvm.hivm.vldus.v{N}{ty}` | `updated_align`（+ hidden base ptr） | intrinsic 返回 3 个值，PTOAS 丢弃第 3 个 |

### 2.4 bisheng 支持但 PTOAS 尚无 Op 定义的指令

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
| 嵌套 `scf.for` 循环 | 部分支持（脆弱的 PHI 嵌套） | ✓（递归 `iter_args` 穿针） |
| 带 mask 的 `pto.vsts` | ✓ | ✓ |
| `pto.vsstb` 块步长存储 | 部分支持（独立开关） | ✓（统一框架） |
| AIV 软件循环 | ✗（排除，因为 AIV 软件循环不被 `LoopInfo` 识别为真实硬件循环，SCEV 无法分析） | ✓（`scf.for` 统一表示，无此限制） |
| 非循环顺序访问 | ✗（不支持） | ✓（顺序路径，见 4.3） |
| 循环体内单次迭代的展开访问 | ✗（不支持） | ✓（循环路径未命中的剩余 op 由顺序路径处理） |
| 显式 `arith.addi` 步长模式 | 可能 SCEV 失败 | ✓（直接匹配 `arith.addi`） |

## 4. 算法设计

### 4.1 整体流程

pass 的驱动分为两个阶段。两个阶段都通过 `PostUpdateTable`（static `StringMap<PostUpdateOpInfo>`，维护第 2 章的指令集合及其地址操作数布局）识别候选，并共享地址单位换算、`StrideExpr`、合法性检查和 post-update op 构造逻辑：

```
阶段一 · 循环路径：候选 op 直接位于某个 scf.for 的循环体内，并且在每次迭代中更新。
    以 ForOp 为单位批量处理（同循环内多 op 合并 iter_arg）：
      · 纯符号分析：先做累加器分析，未命中再做 delta 分析。
      · 判定与物化：完成合法性检查后，在唯一插入点物化 stride 并改写；
        若产生新的 iter_arg，则向外层循环传播。

阶段二 · 顺序路径：循环路径完成后，重新收集 vecscope 内的全部 block，
    包括 scf.for body、vecscope 体和 scf.if 分支体。只扫描仍为非 Post-Update
    形式的剩余 op；以 block 为单位分桶，桶内按程序序寻找互不重叠的
    最大 `SequentialRun`，并链式改写。
```

循环路径按内层到外层处理全部 `scf.for`。由于循环改写会 erase 并重建 ForOp，顺序路径必须在全部循环改写结束后重新收集 block 和候选 op，不能跨阶段保存 `Block *` 或 `Operation *`。这样循环路径已改写的 op 会因已有 `updated_base` 被自然跳过；循环路径未命中，以及原本位于 `scf.if` 等嵌套区域中的 op，仍可在单次 block 执行范围内参与顺序路径。

### 4.2 循环路径

循环路径对 base 和 strideOperand **各自独立**分析：每个操作数先试 **累加器分析**（优先），未命中再退到 **delta 分析**（兜底），两者的结果最后按 4.2.1 的公式合并为 `stride_new`。前者处理该操作数已通过 `iter_args` 显式累加的场景（stride 可以是任意已计算的值），后者处理从 IV 全新计算、无累加器的场景（stride 须为循环不变量）。因此同一条指令的 base 走累加器、strideOperand 走 delta 是允许的组合。

各类内存指令的分析和改写通过统一的地址描述符抽象，共享同一套分析流程。

无论走哪条路径，分析都只产出**符号表达式**，不触碰 IR；确认候选可行后才在单一插入点物化（见 4.2.2）。

#### 4.2.1 地址描述符

每个候选 op 由 `PostUpdateOpInfo` 描述：base 与 strideOperand 的操作数下标，以及 strideOperand 的**单位**。

```
enum class StrideUnit { Element, Block, Alignment, Byte };
struct PostUpdateOpInfo {
  int baseOperandIdx;
  int strideOperandIdx;
  StrideUnit strideUnit;
  unsigned minResultsForPost;
};
```

`delta(base)` 与 `strideOperand` 的单位不同，合并前必须先统一到字节。引入两个字节量：

- **`elemBytes`** = base 指针一个 `pto.addptr` 单位的字节数（由 `addPtrUnitBytes(base)` 求得，即 lowering 规约后的 GEP 元素类型宽度；非字节对齐的低精度打包类型无法确定，直接放弃候选）。
- **`unitBytes`** = strideOperand 一个单位的字节数（由 `strideUnitBytes(unit, elemBytes)` 求得）。单位取值来自该 op 的 lowering：

| 指令 | base | strideOperand | strideUnit | unitBytes | 有效地址 |
|------|------|---------------|-----------|-----------|---------|
| vlds/vsts | source/destination | offset (Index) | Element | elemBytes | base + offset |
| vldsx2（Step 4） | source | offset (Index) | Element | elemBytes | base + offset |
| vsstb/vsldb | destination/source | repeat_stride (I16) | Block | 32 | dest + (32/elemBytes)·repeat_stride |
| plds/psts（Step 4） | source/destination | offset (Index) | Byte | 1 | base + offset/elemBytes |
| pldi（Step 4） | source | offset (Index) | Alignment | NORM: VL/8；US: VL/16；DS: min(32, VL/4) | base + (unitBytes/elemBytes)·offset |
| psti（Step 4） | destination | offset (Index) | Alignment | NORM: VL/8；PK: VL/16 | base + (unitBytes/elemBytes)·offset |
| sprsts（Step 4） | destination | offset (I32) | Byte | 1 | dest + offset/elemBytes |
| sprsti（Step 4） | destination | offset (I32) | Alignment | AR: 4 | dest + (4/elemBytes)·offset |
| vstas（Step 4） | destination | offset (I32) | Element | elemBytes | dest + offset |

> intrinsic 参数原样透传不能单独证明硬件地址单位；Step 4 的 immediate/scalar 差异以 CANN 9.1 SIM 的实际更新地址为准。
> 上表中的 VL 以字节计；A5 的 VL 为 256 bytes，因此 pldi 的 NORM/US/DS 分别为 32/16/32 bytes，psti 的 NORM/PK 分别为 32/16 bytes。其他目标必须提供自己的查询结果，否则该候选不改写。

分析和改写的核心公式统一以字节表达：

```
Δbytes     = elemBytes·delta(base) + unitBytes·delta(strideOperand)
stride_new = Δbytes / unitBytes = (elemBytes/unitBytes)·delta(base) + delta(strideOperand)
init_ptr   = pto.addptr(base_0,  (unitBytes/elemBytes)·strideOperand_0)   // 偏移以元素计
```

展开后可见**只有 `delta(base)` 被缩放,`delta(strideOperand)` 恒等透传**——这保证了 stride 保持符号形式,循环变化的增量对所有指令类别都仍受支持;当 `elemBytes == unitBytes`(Element 类)缩放因子为 1,不发射任何 IR,vlds/vsts 的行为与未引入缩放前逐字相同。

约束（由 4.2.5 检查 3 统一裁定）：`(elemBytes/unitBytes)·delta(base)` 须为精确的整数缩放。`unitBytes % elemBytes == 0` 时，将 `delta(base)` 规范化为“常量 + Σ(系数 × SSA 叶子)”，要求常量和每个系数都能被 `unitBytes/elemBytes` 整除；`elemBytes % unitBytes == 0` 时乘 `elemBytes/unitBytes` 恒精确；互不整除则放弃。`init_ptr` 的 `(unitBytes/elemBytes)·strideOperand_0` 对称：Block 类恒精确（elemBytes 整除 32），Byte 类要求 `elemBytes | strideOperand_0`。

#### 4.2.2 分析与物化分离

循环路径分两个阶段。**分析阶段**不产生任何 IR：`decomposeLinear`、`getIterArgIncrement`、`computeDelta` 均为纯函数，返回符号表达式 `StrideExpr`。**物化阶段**在候选通过全部合法性检查之后，由 `materialize` 在唯一插入点一次性发射 IR。

```
StrideExpr := Const(int64)                  // 类型在物化时按上下文决定
            | Leaf(Value)                   // IR 中已存在的叶子值
            | Add(e, e) | Sub(e, e) | Mul(e, e)
            | Cast(op, e)                   // index_cast/index_castui，op 为克隆模板
```

`StrideExpr` 在构造时即做常量折叠（`foldConst`），因此 `delta(base)` 缩放是否精确、`stride_new` 是否恒为零，全部在符号层判定，判定不依赖任何已生成的 IR。

该划分是正确性要求，而非代码组织风格。以下三条性质依赖于它：

1. **支配性由构造保证。** 物化自底向上进行，每个新建 op 的操作数要么是刚发射的子表达式，要么是已通过可用性检查的叶子，因此结果天然支配插入点。

   可用性检查（4.2.5 检查 6）本身也依赖分析不建 IR。分析期间考察的每个 `Value` 都先于本次变换存在；对这样的值，"定义点早于插入点"可**传递地**推出"其操作数也早于插入点"，于是只需检查表达式的叶子。而对于变换过程中新建的 op，这条蕴含关系并不成立——它自身位置合法，不代表它的操作数在该位置可用。把 IR 生成推迟到分析之后，正是为了让被检查的值始终落在前一种情形里。

2. **递归可 memoize。** 纯函数的结果只取决于入参，`decomposeLinear` 与 `computeDelta` 按 `Value` 缓存，共享子表达式只分解一次，分解代价与定义链 DAG 的规模成线性关系。

3. **被拒候选不留残留。** 合法性检查全部先于物化完成，放弃某个候选时 IR 尚未被触碰。`elemBytes/unitBytes` 缩放这类中间运算也只存在于符号层，不会以 op 的形式落地。

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

累加器路径和 delta 路径对 cast 使用同一条正确性条件：必须能够证明 `delta(cast(x)) == cast(delta(x))`。循环不变量的 cast 的 delta 恒为零，可以直接保留；loop-varying cast 则按转换方向分别证明：

- 对 `iN -> index` 扩展，当前 pass 只接受规范 i16 地址域中的直接 IV 或 iter_arg 递推。循环次数、初值和固定增量必须均为常量，并能证明包括最终 backedge 在内的完整源递推不会在相应的有符号或无符号 i16 范围中回绕；其他源宽度、动态递推或复杂递推均返回 `Failed`。
- 对 `index -> iN` 缩窄，必须证明循环中每个被 cast 的源值均可由目标类型表示，使缩窄不会截断地址序列。当前只对具有常量 lower、upper 和正 step 的直接 IV 做此证明；index 类型 iter_arg 的完整运行时值域无法由当前分析确定，因此返回 `Failed`。

上述规则只覆盖 `arith.index_cast` 和 `arith.index_castui`；`arith.trunci` 等其他整数转换不属于当前分析语法，遇到时保守放弃。

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

当 base 和 strideOperand 都不是 iter_arg 时，回退到 delta 分析。

定义 `delta(v)` 为值 `v` 在 `scf.for` 每次迭代间的变化量：

| `v` 的类型 | `delta(v)` 的值 |
|---------|-------|
| `v` 是 IV | `step`（`scf.for` 的步长） |
| `v` 是常量或循环不变量 | `0` |
| `v = arith.addi(a, b)` | `delta(a) + delta(b)` |
| `v = arith.subi(a, b)` | `delta(a) - delta(b)` |
| `v = arith.muli(a, b)`，其中一个循环不变 | `invariant * delta(other)` |
| `v = arith.index_castui(a)` 或 `arith.index_cast(a)` | 循环不变量为零；loop-varying 的 `iN -> index` 扩展仅接受已证明完整递推不回绕的规范 i16 来源；loop-varying 的 `index -> iN` 缩窄仅在能证明所有源值均可由目标类型表示时为 `cast(delta(a))`，否则为 `unknown` |
| 其他 | `unknown`（放弃） |

```
stride_new = (elemBytes/unitBytes)·delta(base) + delta(strideOperand)   // 见 4.2.1
```

`stride_new` 须为循环不变量，`(elemBytes/unitBytes)·delta(base)` 须为精确整数缩放（见 4.2.1 约束）。

**正确性：** delta 表中的操作构成仿射函数的封闭运算集合。定义链仅由这些操作构成时，delta 计算不会遗漏。遇到表外操作时保守放弃。cast 需要额外满足源递推和值域条件（4.2.3）。

delta 分析同样是纯符号的：表中每一行返回 `StrideExpr`，结果按 `Value` 缓存。

#### 4.2.5 合法性检查

无论由累加器分析还是 delta 分析产出 stride，都须满足以下条件。全部检查在物化之前完成，任一不满足即放弃该候选，且不留下任何 IR。

1. **op 尚未处于 Post-Update 形式。** 检查 `op.getUpdatedBase()` 为空。

2. **op 直接位于 `scf.for` 循环体内**（不嵌套在循环内的 `scf.if` 或其他控制流中），避免部分迭代问题。

3. **`(elemBytes/unitBytes)·delta(base)` 为精确整数缩放。** Element 类（elemBytes == unitBytes）自动满足；Block 类（`unitBytes % elemBytes == 0`，如 unitBytes=32）要求规范化 `delta(base)` 的常量项和每个 SSA 叶子系数都能被 `unitBytes/elemBytes` 整除；Byte 类（`elemBytes % unitBytes == 0`）乘 `elemBytes/unitBytes` 恒精确；互不整除则放弃。

   关键在于**只有 `delta(base)` 参与缩放**：`delta(strideOperand)` 恒等透传、不要求整除也不要求是常量，因此循环变化的 strideOperand 增量对各类指令都受支持（如 vsstb 的 `repeat_stride` 为非常量 iter_arg 累加）。仿射规范化使 `8*k` 这类非常量 base delta 也能证明可被 8 精确缩放。

4. **stride_new 为零时跳过。** 地址不前进，post-update 无收益。常量折叠使各项相消、合成结果恒为零的情形（如 base 每轮 +8、strideOperand 每轮 −8）同样能被识别。

5. **类型一致性与目标类型精确性。** stride 表达式各子项须归结为同一类型，否则放弃，以免构造出 `arith.addi(index, i32)` 这类非法 op。非常量表达式的结果类型还必须与 op 的 strideOperand 类型一致；不能通过消除 cast 或改用窄类型算术来强行匹配。`Const` 项不参与子项类型约束，其类型在物化时直接采用 strideOperand 类型。

   此外，各 `Const` 项的数值须能被目标类型表示。`stride_new` 对块步长指令是窄整数（i16），超出范围的 stride 会构造出非法常量，因此在物化前一并检查并放弃。动态宽结果若没有完整值域证明，同样保守放弃。

6. **操作数可用性（支配性）。** stride 表达式的每个叶子须在候选 op 处可用：循环不变量、block argument、或定义点早于候选 op。若叶子定义在候选 op **之后**，仅当其定义链全部为 pure op 时克隆到候选 op 之前（克隆保留原结果序号，多结果 op 亦正确）；否则放弃。该检查以只读方式先行完成，克隆发生在物化阶段。

#### 4.2.6 改写

改写步骤对所有指令统一：

1. **物化 stride。** 叶子全部循环不变时发射到循环外，否则发射到候选 op 之前。常量按 `(值, 类型)` 在同一循环内复用同一个 SSA 值——4.2.7 的分组按 stride 的 **Value 同一性** 判定，重复创建等值常量会把本可共享 `iter_arg` 的 op 拆成多组。

2. 计算初始指针 `init_ptr = pto.addptr(base_0, (unitBytes/elemBytes)·strideOperand_0)`（见 4.2.1；若偏移为零则直接用 `base_0`）。传给 `pto.addptr` 前将最终偏移规范为 `index`；Block 单位保持无符号扩展，其他单位使用有符号扩展，避免丢失 `sprsti` 负立即数的语义。
3. 新增指针类型的 `iter_arg`，初始值为 `init_ptr`。
4. 创建 Post-Update op：将 `strideOperand` 替换为 `stride_new`，base 替换为 iter_arg 的 block argument。其余操作数（block_stride、mask、dist 等）不变。
5. 将 `updated_base` 通过 `scf.yield` 传出。
6. 对改写后的循环做 loop-aware liveness，删除已经被 Post-Update 指针链完全取代的旧 accumulator `iter_arg`、更新表达式、loop result 和 yield 操作数。

第 6 步不能只依赖普通 DCE。旧 accumulator 即使没有真实用户，仍会形成
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

理想的分组键是 `(init_ptr, byte_stride)`。但 `init_ptr` 不适合直接入键：分组按 **Value 同一性** 比较，而 `computeInitialPtr` 可能为每个候选各自物化一个 `pto.addptr`，起点数值相同也未必是同一个 SSA 值。因此改用决定地址序列的原始量：分组键取 `(base, strideOperand, stride_new, unitBytes)`。加入 `unitBytes` 可防止 f32 上数值相同的 Element 与 Byte stride 被误合并；该键可能把本可合并的组拆开，但不会合并字节递推不同的组。

同组的 op 共享一个 `iter_arg`，所有 op 使用同一个 pre-update 指针（block argument），不链式传递 `updated_base`。原因：同一迭代内同组 op 访问相同地址，链式传递会使后续 op 的地址偏移一个 stride。每组只需 yield 一个 `updated_base`。

因为键按 Value 同一性比较，stride 的物化必须保证等值常量复用同一 SSA 值（见 4.2.6 步骤 1）；否则语义相同的 op 会被拆成多组，退化为各自持有一个 `iter_arg`。

#### 4.2.8 嵌套循环

对于嵌套 `scf.for`，在每一层循环添加 `iter_arg` 携带指针，内层的 init 值接外层的当前值。`scf.for` 的 `iter_args` 天然保证 init/yield 的对应关系。

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
4. **step 在 `SequentialRun` 头部可用。** 复用 4.2.5 的叶子可用性检查；定义在 `SequentialRun` 头部之后的 pure 定义链（包括产生 opaque leaf 的非仿射运算）可克隆到头部之前，含副作用、不可安全提前或无法支配的定义使该 `SequentialRun` 放弃。
5. **整条 `SequentialRun` 先分析、后物化。** 所有候选通过检查后才创建常量、`pto.addptr` 或新 op，拒绝 `SequentialRun` 不留下残余 IR。

顺序路径不重排 op，也不改变访问地址和访问顺序，因此不同候选之间的内存读写不会截断 `SequentialRun`，无需额外别名分析。

#### 4.3.4 收益性检查

长度至少为 3 只是合法性门槛。合法的 `SequentialRun` 还必须命中下列两类结构信号之一；`pto.vlds`、`pto.vsts` 和 `pto.vsstb` 使用相同规则，不按 op 类型建立白名单。

**动态多层 base chain。** 当规范化后的 `step` 是编译期常量时，从第二个候选开始统计改写后确定死亡的动态 `pto.addptr`。某层 addptr 的 offset 经 4.3.1 的仿射规范化后包含 SSA 项，才视为动态；纯常量 addptr 不计收益。addptr result 必须只有下一层待删除 addptr 或当前候选这一个用户，第一条候选的 base chain 因仍用于构造 `init_ptr` 而不计收益。使用 `DenseSet` 对共享定义去重，并按下式接受：

```
deadDynamicAddPtrs > pointerEdges + initPtrCost

pointerEdges = runLength - 1
initPtrCost  = first.strideOperand 为常量零 ? 0 : 1
```

即使尾指令使用 normal 形式，它仍消费倒数第二条返回的指针，因此长度为 `N` 的 run 仍有 `N-1` 条 pointer edge。

**direct symbolic leaf。** 所有候选直接使用同一个 `rootBase`，首条 effective offset 为零，规范化 step 恰为 `1 × SSA atom`（常量项为零、只有一个系数为 1 的 leaf/cast atom），且第三条及以后 offset 的累计 `arith.addi` / `arith.subi` 定义链在改写后确定死亡。定义链若有 run 外用户则拒绝。step 已支配 run 头时直接复用；定义在 run 头之后时，仅当 pure 定义可安全克隆且原定义链随地址 use 一起死亡、不会增加重复计算时才接受。该类别没有额外的长度阈值：任意满足合法性最小长度 `N >= 3` 的 run 都可改写。

除此之外均保守拒绝。当前实现不为一般 `arith.addi`、`arith.muli` 或常量 `pto.addptr` 设置主观权重，也不因这些 op 出现在地址表达式中就假设最终 ISA 一定减少。收益性检查只读，不创建 IR；被拒 run 不留下常量、clone 或 `pto.addptr`。

#### 4.3.5 改写

对每个 `SequentialRun`，`stride_new` 就是 4.3.1 得到的规范化 `step`。初始指针直接从 `SequentialRun` 首条 op 的实际操作数计算：

```
init_ptr = pto.addptr(first.base,
                      (unitBytes/elemBytes)·first.strideOperand)
```

这里复用循环路径的精确单位换算；不需要从 `rootBase` 重新构造绝对地址。对长度为 `N` 的 `SequentialRun`，前 `N-1` 条替换为 post-update 形式，前一条的 `updated_base` 作为后一条的 base，strideOperand 替换为 `stride_new`。最后一条只需访问当前指针，不再产生后续用户，因此保留 normal 形式，以前一条的 `updated_base` 为 base，并把 strideOperand 替换为同类型零值：

```mlir
// vlds 变换后
%v0, %ptr1 = pto.vlds %init_ptr[%c64] : ... -> ..., !pto.ptr<f32, ub>
%v1, %ptr2 = pto.vlds %ptr1[%c64] : ... -> ..., !pto.ptr<f32, ub>
%v2        = pto.vlds %ptr2[%c0] : ... -> ...
```

最后一条 normal op 保留原 operands、attributes 和原始 result types，不追加 `updated_base`；`vsts` 与 `vsstb` 同理。所有 `SequentialRun` 在只读分析阶段确定后，先物化各自的 `stride_new`、同类型零值和 `init_ptr`，再按原程序序替换候选 op，避免 erase op 使其他 `SequentialRun` 保存的表达式失效。不同桶各自维护独立指针链，即使其 op 在原 block 中交错也互不影响。

## 5. Pass 集成

### 5.1 在 Pipeline 中的位置

pass 应运行在 VPTO 后端 pipeline 中，位于 `PTOInferVPTOVecScope` 之后、`PrepareVPTOLLVMLoweringPass` 之前：

```
VPTOExpandWrapperOps
CSE
PTOInferVPTOVecScope
→ VPTOSoftPostUpdate（新增）        ← 在此处
...
PrepareVPTOLLVMLoweringPass
LowerVPTOOpsPass
```

该位置确保：
- Wrapper op 已展开（IR 干净）
- **`pto.vecscope` 已存在**。pass 只改写 `pto.vecscope` 内的 op，而多数 kernel 的 vecscope 是由 `PTOInferVPTOVecScope` 创建的——排在它之前会让 pass 对这类输入静默失效。手写 vecscope 的测试用例不会暴露该问题，因此回归测试中必须包含不含手写 vecscope 的输入（`soft_postupdate_inferred-vecscope.pto`）
- Post-Update 结果对 LLVM lowering 可见，后者根据 `getUpdatedBase()` 选择 `vldsx1` 或 `vldsx1.post`

`PTOInferVPTOVecScope` 以整 op 方式把 `scf.for` 搬入 `pto.vecscope`（`wrapCluster` 用 `splice` 移动整个 op），循环体内部结构不变，因此 4.2.5 检查 2「op 直接位于循环体内」的判定不受影响。

该优化在 `ptoas` 的 VPTO pipeline 中默认开启。需要对比或回退到未优化行为时，可显式传入
`--enable-vpto-soft-postupdate=false` 关闭。

### 5.2 Pass 注册

```tablegen
// 在 include/PTO/Transforms/Passes.td 中
def VPTOSoftPostUpdate : Pass<"vpto-soft-postupdate", "ModuleOp"> {
  let summary = "Convert fixed-stride VPTO memory ops to post-update form";
  let dependentDialects = ["pto::PTODialect", "scf::SCFDialect",
                           "arith::ArithDialect"];
}
```
