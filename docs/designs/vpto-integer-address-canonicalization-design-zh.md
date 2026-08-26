# VPTO 整数地址规范化设计

## 1. 结论

Issue #591 暴露的是 pointer SSA 形态问题：地址的循环变化量仍藏在
`pto.castptr` 的整数输入中，而现有 `PTOAddressAnalysis` 只会沿
`pto.addptr` 累积 element offset。

本设计只规范化 pointer，不解释使用该 pointer 的 memory operation：

```mlir
%byte_address = ... : i64
%ptr = pto.castptr %byte_address : i64 -> !pto.ptr<T, ub>
```

规范化的目标形态是「规范根 + addptr(element_offset)」：

```mlir
%root_addr = ...            // 规范根整数输入：常量 0，或不可归零原子（见 3.2）
%base = pto.castptr %root_addr : i64 -> !pto.ptr<T, ub>
%ptr = pto.addptr %base, %element_offset
    : !pto.ptr<T, ub> -> !pto.ptr<T, ub>
```

`%element_offset` 必须是与原字节表达式「可归零部分」等价的精确商，并按
`pto.addptr` 要求物化为 `index`。规范根取 `castptr(0)` 是退化特例：整个字节
表达式全部归入 element offset；含运行时原子基址（如 kernel 参数）时，原子保留
为根（见 3.2 与 C15）。变换不改 memory operation，不感知 LLVM intrinsic、offset
字段宽度、Bisheng post-update 编码、访问 footprint 或 post-update 行为。

设计由两个彼此独立的 canonical rewrite 组成：

1. **整数地址规范化**：`castptr(byte_address)` 变为
   `castptr(规范根) + addptr(element_offset)`（规范根为 `castptr(0)` 或不可归零
   原子，见 3.2/3.5）；这是核心设计。
2. **addptr 吸收**：对本来就有 element offset 的 operation，将
   `op(addptr(base, A), O)` 变为 `op(base, A + O)`；这是面向现有后端形态的普通
   canonical fold，不参与地址等价分析。

第一步已经让 `PTOAddressAnalysis` 和它的 consumers 看见地址递推。第二步是否执行，
不影响第一步的正确性。

## 2. 问题边界

### 2.1 当前主线已有的能力

当前 `PTOAddressAnalysis` 已经统一表示 typed pointer 地址：

- `PTOAddressExpr` 保存 root/base、`pto.addptr` element offset 和 operation offset；
- `getAddresses()` 沿相同 element type 的 `pto.addptr` 链向上遍历；
- `getDeltaBytes()`、`getDifferenceBytes()` 将 pointer offset 和 operation offset
  统一换算为 byte difference；
- `convertDeltaToUnit()` 对线性表达式做精确单位换算，不能整除时返回
  `InexactUnitConversion`；
- `PTOValueEvolutionAnalysis` 已经表示常量、加减、常量乘法和不同种类的整数 cast。

缺口只有一个：`getAddresses()` 到达 `pto.castptr(integer)` 后，把它作为 opaque root，
不会把整数输入中的循环变化量变成 element offset。

因此不新增额外的地址模型、终端 footprint 或 operation adapter。需要新增的是一个使用现有 typed expression
能力的 rewrite，以及它所需的精确商和 SSA 物化 helper。

### 2.2 `pto.addptr` 是 PTO 层的语义目标

`pto.addptr` 是 Pure op，offset 以 pointer element 为单位。规范化在 PTO IR 内证明：

```text
integer byte address
  == zero-origin typed pointer + element offset
```

LLVM 最终使用 GEP、整数加法还是 target intrinsic，不属于该证明。特别是 LLVM lowering
中出现的 `i32 trunc` 只是当前 emitter 的结果，不能成为地址分析的输入、拒绝条件或
正确性依据。若某个 operation 的 lowering 字段过窄，那是 operation verifier、target
legality 或 emitter 的问题。

### 2.3 地址空间前提不是用户配置

该变换只适用于 PTO 语义明确规定为 **zero-origin integral address space** 的空间：

- 地址由编译器或开发者以整数管理；
- 地址 0 有效；
- `castptr(integer)` 按整数 bit pattern 形成地址；
- `addptr` 的 element scaling 与该整数地址使用相同的 byte 单位。

A5 UB 满足这些条件。这个判定应来自 PTO dialect/target 的权威语义，不引入可由用户任意
组合的地址配置对象。其他空间只有在其 PTO 语义同样满足四项条件时才能启用；
LLVM object/provenance 管理的空间和语义未知的空间保持原样。

这一区分也解释了为什么 LLVM 关于 `null + GEP` 的通用限制不直接否定 UB 变换：UB 的
0 是实际地址原点，不是 C/LLVM object model 中不可解引用的 null object。

## 3. 规范化语义

### 3.1 输入和输出

设原 op 为：

```text
P = castptr(B) : integer -> ptr<T, S>
```

其中：

- `B` 是整数 byte-address expression；
- `E = sizeof_storage(T)`，单位为 byte；
- `S` 是已确认的 zero-origin integral address space；
- `Waddr` 是该空间的地址位宽；
- `Windex` 是当前 PTO target 中 `index` 的位宽。

把 `B` 线性化为 `C + sum(Ki * Xi)` 后，规范根与商分别定义：

- **规范根整数输入** `R`：所有系数 `Ki` 都能被 `E` 整除时，`R = 0`（退化特例，
  根为 `castptr(0)`）；否则取「系数为 1 且不可被 `E` 整除的原子叶子」为 `R`
  （见 3.2 的原子根规则，最常见是运行时基址参数；仅支持单个此类原子，多原子
  场景整体拒绝）。
- **商** `Q`：`B - R` 的可整除部分除以 `E` 的精确商，可物化为 `index`，满足：

```text
CastToAddressWidth(B - R)
  == ScaleElementOffset(CastIndex(Q), E)       (mod 2^Waddr)
```

然后生成：

```text
P' = addptr(castptr(R), Q) : ptr<T, S>
```

只要上式成立，`P` 和 `P'` 作为 pointer value 等价，与它们是否被 load、store、pointer
cast、循环递推或多个 operation 使用无关。这就是本设计不需要分析地址终端的原因。

### 3.2 精确商与原子根

对线性 typed expression，默认规范形式是对整个 `B` 求 `E` 的精确商，不先选择 base，
也不把表达式拆成 invariant 和 variant 部分：

```text
B = C + sum(Ki * Xi)
Q = C/E + sum((Ki/E) * Xi)
```

只有常数项 `C` 和每个系数 `Ki` 都能被 `E` 整除时，线性精确商存在，规范根
`R = 0`。cast、block argument 和其他不能穿透的 value 作为原子 `Xi` 保留，因此不会
为了整除而把 cast 沿数据流移动。

**原子根例外**：当存在系数 `Ki` 不可被 `E` 整除的原子叶子时（最常见是系数 1 的
运行时基址参数，如 kernel 参数的 `castptr(%param)`），该原子不能归入 offset，
保留为规范根：

```text
B = R + (C' + sum(Kj * Xj))         // R 是不可归零原子部分
Q = C'/E + sum((Kj/E) * Xj)
P' = addptr(castptr(R), Q)
```

闭合边界：只支持系数为 1 的原子叶子作根（`%param` 本身）；系数 `|Ki| != 1` 且
不可整除的原子（如 `%param * 3` 对 `E = 2`）没有唯一的「根 + 偏移」分离，整体
拒绝，保持原样。多个系数为 1 的原子叶子同时存在时，它们组成的和表达式作为根
（根仍唯一）；该场景当前按保守策略整体拒绝，留给后续扩展。

现有 `normalizePTOLinearExpr()` 与 `dividePTOLinearExprExact()` 已经实现这项代数能力。
新 rewrite 可以复用它们，但还必须完成 typed round-trip proof；仅有 `int64_t` 系数整除
不等于跨位宽转换一定正确。

不允许为了扩大覆盖范围在 IR 中生成动态 `div`/`rem`。规范化的目标是更直接的地址
表达式；新增运行时除法既有成本，也把“静态可证明”变成了运行时条件。

### 3.3 cast 和有限位宽

精确商保留原表达式中的 cast 边界。以下输入可以把 cast 结果作为一个原子：

```mlir
%x64 = arith.extui %x : i32 to i64
%bytes = arith.muli %x64, %c4096 : i64
```

商为 `%x64 * 2048`，没有移动 `%x64`。相反，默认不把：

```text
trunc(mul(x, 4096))
```

改写成：

```text
mul(trunc(x), 2048)
```

因为截断发生点改变了。

从商表达式到 `index` 还必须验证 round trip：

```text
Q-expression -> index -> addptr element scaling -> Waddr
```

保持原地址 bit pattern。`Windex` 和 `Waddr` 来自 PTO target/data-layout 语义，不从
LLVM lowering 产物反推。当前 `PTOValueEvolutionAnalysis` 中固定 64 位的 index 处理是
实现现状，不应被提升为跨 target 合同。

**实现近似（C14 的充分条件）**：第一版实现以「输入位宽 == index 位宽（64）」作为
round-trip 的充分条件——同宽 `CastIndex` 平凡无损。更窄输入（如 i32）即使商在数学上
可无损进入 index（如 `x * 4096` 的商 `x * 2048` 在 `x` 值域内不溢出）也被保守拒绝，
直到 PTO 明确 `castptr` 从窄整数到 64 位地址空间的零/符号扩展语义；完整的商
round-trip proof 列为后续工作。拒绝 message 与 C14 表格行的「不改」行为一致，但
当前实现不区分「商真的不能无损」与「语义未定义导致不能证明」。

带 `nuw`/`nsw` 的原算术只要求在原程序有定义的输入上保持地址，并且新表达式不新增
poison。reifier 不复制无法证明的 overflow flag。无法证明 refinement 时保持原样。

### 3.4 非整除地址

例如：

```mlir
%bytes = arith.addi %multiple_of_2, %c1 : i64
%ptr = pto.castptr %bytes : i64 -> !pto.ptr<f16, ub>
```

不存在 f16 element offset `Q` 使 `2 * Q` 表示所有输入，因此不规范化。

本设计不退化成：

```text
castptr(0) -> ptr<i8>
addptr(byte_offset)
castptr -> ptr<T>
```

该形态虽然可能保持数值地址，却重新引入 ptr-to-ptr cast，并且当前
`PTOAddressAnalysis` 不会透明地穿过最后一个 cast。它不能形成唯一 canonical form，
也不能解决本设计要解决的分析入口问题。拒绝非整除输入是闭合的语义边界。

这里的「非整除」指常数项或系数不可整除且不适用 3.2 的原子根例外：常数项不可整除
（如上面的 `+1`）没有 typed canonical form，整体拒绝；单个系数为 1 的原子叶子不可
归零则是「保留为根」而非拒绝（C15，见 3.2 闭合边界），两者拒绝原因不同（与 6.3
的区分一致）。

### 3.5 收敛性、幂等性和共享

规范化必须收敛到唯一的规范形（NF），不是若干条互不相关的 peephole：

```text
NF = castptr(R)                     // Q 恒为 0 时（纯根：castptr(0) 或 castptr(%原子)）
   | castptr(R) + addptr(Q)         // R 是常量 0 或不可归零原子输入，Q 非平凡
```

rewrite 规则 `R` 只匹配 integer-to-pointer `pto.castptr`，且要求**可归零部分的精确商
`Q = (B - R) / E` 不恒为 0**（即 `B - R` 非平凡）：

- **终止**：每次应用 `R` 消灭一个「Q 非平凡」的 `castptr`；`R` 的产物中唯一的
  `castptr` 输入是常量 0 或不可归零原子，它们的 `B - R = 0`、`Q` 恒为 0，不满足
  `R` 的前提，不会被 canonicalizer worklist 再次命中。应用次数 ≤ 输入中可规范化
  `castptr` 的个数，有限。
  （若只排除常量 0 而不排除纯原子输入，`castptr(%param)` 会被反复包成
  `addptr(castptr(%param), 0)`，永不终止——所以「Q 不恒为 0」才是完整的排除条件，
  它同时覆盖常量 0 与纯原子两类输入。）
- **唯一性**：根由不可归零原子部分唯一确定（无原子时为常量 0；仅支持单个原子，
  见 3.2），`Q` 是 `B - R` 的唯一精确商（3.2）。同一字节表达式只有一种 NF。
- **幂等**：`R` 的输出是 NF，再次运行不匹配、不改 IR。

若原 `%ptr` 有多个 users，统一替换其定义即可；不按 memory operation 分别 clone 地址
表达式。若商表达式不能在原 `castptr` 位置合法物化，整个 rewrite 失败，不在 users 附近
复制计算。

## 4. 实现结构

### 4.1 分析，不新建第二套地址体系

rewrite 使用现有 `PTOValueEvolutionAnalysis`：

```text
castptr integer input
  -> getExpr()
  -> normalizePTOLinearExpr()
  -> dividePTOLinearExprExact(elementBytes)
  -> typed/index round-trip proof
```

分析结果可以是 pass 内部的小型值对象：

```c++
struct ExactElementQuotient {
  PTOTypedExprRef quotient;
  Type sourceIntegerType;
  unsigned addressWidth;
  unsigned indexWidth;
};
```

它不是新的 public address model，也不保存 operation、builder 或待创建 SSA。element
storage size 应复用 `PTOAddressAnalysis` 现有规则；实现时把当前 file-local
`getElementBytes()` 提升为 Analysis 层共享 helper，避免复制类型规则。

PR #1260 增加的 typed address/alignment/contiguity 查询与这里互补，但不替代精确商。
截至本文核对的当前 checkout，它仍不是可直接依赖的主线 API；本设计不以该 PR 合入为
前提，也不复制它的 alignment 或 stream-fusion 能力。PR 合入后，规范化输出自然成为
这些查询的普通 `addptr` 输入。

### 4.2 SSA 物化

reifier 负责：

1. 复用支配原 `castptr` 的 SSA leaves；
2. 按 typed expression 重建常量、加减和常量乘法；
3. 保留原子 cast value，不跨 cast 重排；
4. 生成经过 round-trip proof 的 `index` value；
5. 在原 `castptr` 位置生成规范根整数输入（常量 0 或原子叶子，见 3.2）、同 pointer
   type 的 `castptr(R)` 和 `pto.addptr`；
6. 原子地替换所有 users，失败时不改 IR。

只重建 Pure、可推测执行且已由 proof 覆盖的算术。load、call、带副作用 op 或 region
外不可捕获的值都只可作为支配位置合法的叶子，不能 clone。

**规范根的提升约束**：规范根 `castptr(R)` 是 loop-invariant（`R` 为常量 0 或函数
参数等原子），reifier 把它提升到最近 `scf.for` 之外，使 post-update consumer 看到
循环外 base。提升仅当 `R` 对**所有**被跳过的循环都是 loop-invariant 时合法；若原子根
定义在循环内（如 `index_cast(iv)` 本身不可整除），提升会违反 SSA 支配，此时 `castptr(R)`
留在原位（IR 保持合法，只是不触发 post-update）。商 `Q` 始终在原 `castptr` 位置物化
（依赖循环变量的部分留在循环内）。

### 4.3 与 `PTOAddressAnalysis` 的接入

规范化后无需给 `PTOAddressAnalysis` 增加 raw-integer root：

```text
castptr(integer)                         opaque root
        |
        | canonicalize
        v
castptr(R) -> addptr(element_offset)     existing analyzable form
                                          (R = 常量 0 或不可归零原子)
```

`getAddresses()` 按现有逻辑沿 `addptr` 得到 element offset；
`getDeltaBytes()` 按现有逻辑得到 loop step；SoftPostUpdate、alignment、contiguity 等
consumer 只消费这一套结果。pass 改写 IR 后正常失效并重建 analysis cache。

## 5. 独立的 addptr 吸收 fold

实际 Bisheng 验证表明，下面两种 PTO 形态并不等价地触发后端 post-update：

```mlir
// 地址分析可见，但当前 Bisheng 仍生成 VLDI + SADD。
%ptr = pto.addptr %base, %a
%v = pto.vldsx2 %ptr[%zero], ...

// 当前 Bisheng 能生成 VLDS post-update。
%v = pto.vldsx2 %base[%a], ...
```

因此需要时可再执行一个普通 canonical fold：

```text
op(addptr(base, A), O) -> op(base, A + O)
```

其合法性条件仅来自现有 `VPTOAddressSemantics`：

- operation 当前访问恰有对应 base 和 offset operand；
- offset unit 是 Element，且 element type 与 `addptr` 相同；
- operation 不是已经带 `updatedBase` 的 post-update form；
- `A + O` 能在原 offset 类型中无损物化；
- 替换不改变其他 base users。

这个 fold 不读取 LLVM intrinsic 字段，不推测 post-update step，也不分析 footprint。
它只是把同单位的两级加法折叠到 operation 已有的地址 operand。对于 offset unit 为 Byte、
Block 或 Alignment 的 operation，除非现有 semantics 提供精确的同单位换算，否则不做。

operation 语义建模复用主线的 `VPTOAddressSemanticsOpInterface`（14 个 load/store op
已声明接口：`vlds`/`vldsx2`/`vldus`/`plds`/`pldi`/`vsts`/`vstus`/`psts`/`psti`/
`sprsts`/`sprsti`/`vstas`/`vsldb`/`vsstb`，实现见 `lib/PTO/IR/VPTOAddressSemantics.cpp`
的 `getDefaultVPTOAddressSemantics`；`VPTOSoftPostUpdate` 已按同一接口消费
`postUpdate`）。C11 fold 只读 `currentAccesses` 的 base/offset/unit，不需要新增语义
建模。已知缺口：`vstsx2` 未声明该接口（与 `vldsx2` 不对称），`PTOAddressAnalysis` 与
C11 fold 对它均不可用；实现时补一行接口声明 + `getDefaultVPTOAddressSemantics` case
（与 `vsts` 对称），或在 fold 中显式声明暂不支持。

## 6. Case 驱动的设计收敛

case 不是实现后的补充，而是设计合同的可执行边界。每个接受 case 必须同时检查输出形态
和地址等价；每个拒绝 case 必须检查 IR 保持不变。核心矩阵如下。

| Case | 输入关键形态 | 期望 | 锁定的设计结论 |
| --- | --- | --- | --- |
| C01 | Issue #591: `index_cast(iv) * 4096 -> ptr<f16,ub>` | `castptr(0) + addptr(iv * 2048)` | 整条 byte expression 精确除 element size |
| C02 | `8192 + index_cast(iv) * 4096 -> ptr<f16,ub>` | offset 为 `4096 + iv * 2048` | 不需要挑选 invariant base，所有地址统一归到 offset |
| C03 | `index_cast(iv) * 4096 -> ptr<f32,ub>` | offset 为 `iv * 1024` | element storage size 来自 pointer type |
| C04 | 任意线性 i64 byte address -> `ptr<i8,ub>` | 同值 element offset | element size 1 是恒等换算 |
| C05 | `(extui x) * 4096 -> ptr<f16,ub>` | 保留 `extui x`，系数变 2048 | cast 作为原子，不移动 cast |
| C06 | `trunc(x * 4096) -> ptr<f16,ub>` | 不改 | 不跨截断做代数重排 |
| C07 | `iv * 4096 + 1 -> ptr<f16,ub>` | 不改 | 非整除 byte address 没有 typed element canonical form |
| C08 | 地址空间不是 zero-origin integral | 不改 | 不把 UB 数值地址规则扩散到 object pointer |
| C09 | 已是规范形：`castptr(0) + addptr(E)`、`castptr(%原子) + addptr(E)`，或裸 `castptr(0)` / `castptr(%原子)` | 不改 | 规则要求 `Q` 非平凡；收敛到规范形，无重复包装（见 3.5） |
| C10 | 一个 integer-backed pointer 有多个 users | 生成一个共享 addptr | rewrite pointer producer，不按终端复制 |
| C11 | `%ptr = addptr(base,A); op %ptr[O]` | 可选 fold 为 `op base[A+O]` | 后端形态 fold 与地址规范化分离 |
| C12 | 已有 post-update op | 不做 C11 fold | 当前访问与访问后更新不混合 |
| C13 | `%raw = castptr %p : ptr -> i64; %bytes = raw * 2` | 保持不变 | pointer-derived integer 不进入本规范化；不引入 provenance parser |
| C14 | 商不能无损 round-trip 到 target index | 不改 | index/address width 是 PTO target 证明条件 |
| C15 | `castptr(%param + iv * 4096)`（`%param` 为 kernel 参数） | `castptr(%param)` 为根，offset 只含 `iv * 2048` | 运行时原子基址是规范根；分析与发射以「根循环不变」为前提 |

### 6.1 C01：Issue #591

```mlir
%iv64 = arith.index_cast %iv : index to i64
%bytes = arith.muli %iv64, %c4096 : i64
%ptr = pto.castptr %bytes : i64 -> !pto.ptr<f16, ub>
```

线性式只有一个系数 4096，`sizeof(f16)=2`，精确商是：

```text
Q = index_cast(iv) * 2048
```

A5 的 PTO target 证明该商可无损进入 index 后，输出：

```mlir
%zero = arith.constant 0 : i64
%base = pto.castptr %zero : i64 -> !pto.ptr<f16, ub>
%elements = ... // 与 index_cast(iv) * 2048 bit-equivalent
%ptr = pto.addptr %base, %elements
```

memory operation 原样使用 `%ptr`。如果再需要当前 Bisheng 的 post-update 形态，C11 fold
把这个 `addptr` 吸收到 operation offset；这是第二个独立测试。

### 6.2 C02：常量和动态项一起归一

```text
B = 8192 + iv * 4096
E = 2
Q = 4096 + iv * 2048
```

不需要设计“新 base 选 8192 还是 0”的策略。唯一 canonical base 是 0（常量场景；
运行时原子基址见 6.5/C15），完整商进入 `addptr`。这消除了 invariant-term
selection、profitability search 和多个等价 normal form。

### 6.3 C06/C07：两类拒绝原因不同

C06 可能在数学上总是偶数，但当前表达式模型不能穿过 trunc 证明同一个有限宽函数，属于
**proof unavailable**。C07 的常数项 1 明确不可被 2 整除，属于 **no exact typed form**。
诊断和单元测试应区分两者，避免以后把分析能力不足误认为语义上不可能。

### 6.4 C13：pointer-derived integer

即使某个 PTO 输入在数值上看起来可以整除，也不把 pointer-to-integer-to-pointer
round-trip 纳入本规范化。它需要单独的 PTO provenance/round-trip 合同，而不是依赖
LLVM lowering 的事实。当前设计明确保持该 case 不变；这不会影响 Issue #591，因为
Issue #591 的整数表达式没有 pointer-derived leaf。

### 6.5 C15：运行时原子基址

**转换输入**——`castptr` 的整数输入含不可归零的运行时原子：

```mlir
%param : i64                        // kernel 参数，UB 基址
%bytes = arith.muli %iv, %c4096 : i64
%addr  = arith.addi %param, %bytes : i64
%base  = pto.castptr %addr : i64 -> !pto.ptr<f16, ub>
```

`%param` 是原子叶子（block argument），系数 1 不能被 `E = 2` 整除，按 3.2 的原子根
例外保留为规范根：输出为 `addptr(castptr(%param), iv * 2048)`。该形态的等价性不需要
证明 `%param` 本身可整除；分析与发射都只依赖「根循环不变」：

- `PTOAddressAnalysis::getPointerDelta` 对循环外定义的 root 返回 0
  （`PTOAddressAnalysis.cpp` 的 `isDefinedOutsideOfLoop` 分支）；
- `VPTOSoftPostUpdate` 的 post-update 前提是 base 循环外（`VPTOSoftPostUpdate.cpp`
  的 `isDefinedOutsideOfLoop` 检查），不要求根是常量 0。

这证明规范形是「规范根 + addptr(唯一商)」而不仅是「castptr(0) + addptr」：
`castptr(0)` 是整条可整除时的退化特例，`castptr(%param)` 是纯原子根退化特例。
C15 同时是 §9.4 规范形表述的收口用例。

**C15 实测（2026-08-25，A5 bisheng 工具链）**：以下验证针对**规范形态的发射行为**
（`addptr(castptr(%param), iv*2048)` 经 C11 fold 前/后的两个变体；转换本身由
C01-C14 的 lit 覆盖）。`.work/c15-investigation/` 记录了 `castptr(%param)` 根两种
形态的 LLVM IR 与 device 二进制对比：

- `c15_op_offset.ll`（根 + operation offset，SoftPostUpdate 开启）循环体为
  `@llvm.hivm.vldsx2.post.v128f16(ptr %p, i32 4096, ...)`，返回三元组含更新后指针
  并被 `scf.for` phi 循环携带；与 `castptr(0)` 根对照 `ctrl0_root.ll` 的 intrinsic
  完全同构，仅 base 装载（`inttoptr %param` vs `null`）不同。
- 两者经 bisheng（`--cce-aicore-arch=dav-c310-vec --cce-aicore-only
  -cce-bitcode-is-aicore`）编出的 device `.text` 同为 0x120 字节，prologue 逐字节
  相同，循环体结构一致 → `castptr(%param)` 根 **同样触发 VLDS post-update**。
- 对照 `c15_addptr.ll`（根 + addptr，未 fold）为 `@llvm.hivm.vldsx2.v128f16`
  （非 post）+ 循环内 `mul`/`getelementptr`，device `.text` 为 0x128 字节（多 8
  字节 = 一条地址更新指令，对应 `VLDI + SADD` 形态）→ 不触发 post-update。

结论：post-update 触发只依赖 base 循环不变，与根是否为常量 0 无关；C11 fold
仍是触发 post-update 的必要步骤。

## 7. 测试分层

### 7.1 Analysis/helper 单元测试

直接测试 exact quotient 和 typed round trip，不经过 memory operation：

- 常数、正负系数、多个线性项全部可整除；
- 常数项或任一系数不可整除；
- cast atom 保留；
- 原子根例外：系数 1 的原子叶子不可整除时保留为规范根；系数 `|K| != 1` 且不可
  整除时整体拒绝；
- trunc 外层保持 opaque；
- i32/i64/index 的可证明和不可证明 round trip；
- 带 overflow flag 的 source-defined-domain refinement；
- element size 1、2、4、8；
- pointer-derived leaf 必须保持 opaque 并被 canonicalizer 拒绝。

这些测试锁定证明器，而不是某个后端输出。

### 7.2 Canonicalization lit 测试

实现 pass 时以 C01-C15 构造一个 lit 文件，检查：

```text
accepted:
  integer castptr 消失
  只生成一个规范根 castptr（常量 0 或不可归零原子输入）
  quotient 只生成一次
  addptr element type/address space 不变

rejected:
  原 castptr 和整数表达式逐字形态保持
  不残留部分创建的 zero/base/arithmetic
```

同一文件至少包含一个无 memory user 的 pointer cast round-trip 和一个多 users case，防止
实现重新滑向“按 memory operation 做地址 rebase”。

### 7.3 地址分析测试

规范化后运行 `-pto-print-address-analysis`：

- C01 的 root 是 `castptr(0)`；
- C15 的 root 是 `castptr(%param)`，element offset 是 `iv * 2048`；
- element offset 是 `iv * 2048`；
- f16 byte delta 是 element delta 的 2 倍；
- operation 自身已有的 offset 仍独立存在；
- SoftPostUpdate 不需要 raw-integer matcher。

### 7.4 当前可运行的 baseline

仓库中的 `test/lit/vpto/integer_address_forms_baseline.pto` 已构造并保留三种输入：

1. integer-backed `castptr`；
2. `castptr(0) + typed addptr`；
3. `castptr(0) + operation offset`。

它检查当前 LLVM handoff 的真实差异，并用 store sink 防止被测 load/address chain 被 DCE。
该测试是实现前 baseline，不是地址规范化正确性的证明。

### 7.5 Bisheng 与运行时集成测试

后端观察单独测试，不写入 analysis 合同：

| PTO 输入 | 当前 Bisheng 观察 |
| --- | --- |
| integer-backed `castptr` | `VLDI + SADD` |
| `castptr(0) + addptr` | 仍为 `VLDI + SADD` |
| `castptr(0) + operation offset` | `VLDS ... post-update` |
| `castptr(%param) + operation offset`（C15） | `VLDS ... post-update`（2026-08-25 实测，见 6.5） |
| `castptr(%param) + addptr`（C15 未 fold） | `VLDI + SADD`（2026-08-25 实测，见 6.5） |

此外用边界输入对原/新 kernel 做 differential runtime test：0、1、最大合法 loop index、
含常量 base 的 C02、含运行时基址的 C15、多个 element type。汇编选择是收益回归；
访问结果相同才是语义回归。

## 8. Pass 位置

地址规范化必须在 integer-backed `castptr` 已形成之后、所有 typed-address consumers 之前：

```text
形成 VPTO pointer SSA
  -> integer address canonicalization
  -> optional addptr absorption canonicalization
  -> PTOAddressAnalysis consumers
  -> LLVM lowering
```

这是数据依赖，不把任何具体 post-update pass 写进设计合同。rewrite 修改 pointer SSA 后，
MLIR analysis preservation 应如实声明失效；下游自然重建 `PTOAddressAnalysis`。

闭环 Issue #591 需要 C11 fold 的输出进入 post-update 阶段，因此以
`VPTOSoftPostUpdate` 默认启用为前提（主线 #1330 已将 `--enable-vpto-soft-postupdate`
默认开启）。C11 fold 必须在 post-update pass 之前执行；若 post-update 被显式关闭，
第一步规范化仍改善分析可见性，但不改变发射形态（§7.5 第二行证据）。

## 9. 实现完成条件

实现只有同时满足以下条件才算闭合：

1. C01-C15 均有 executable lit/unit case，正例和反例都有；
2. exact quotient 与 typed round-trip proof 分开测试；
3. canonicalizer 不查询 memory operation 或 LLVM lowering；
4. 输出只有「规范根 + addptr」这一种 pointer normal form（规范根为 `castptr(0)`
   或不可归零原子输入，见 3.2；`Q` 恒为 0 时退化为纯根 `castptr(R)`）；
5. 规则要求 `Q` 非平凡（同时排除常量 0 与纯原子输入），规范化终止于规范形；
   非整除和证明不足均原子失败，不留下半成品 IR；
6. `PTOAddressAnalysis` 无第二套 integer-backed root/matcher；
7. addptr absorption 是独立 pattern，可单独开关和测试；
8. Issue #591 的 PTO IR、address-analysis 输出、Bisheng 汇编和 differential result 四层
   证据均通过；
9. 性能验收与 Issue #591 的目标绑定：rolled tile 循环体内 0 个 `RV_SADD`/`RV_SMOVK`
   标量地址 op（对照 fully-unrolled 形态），且 rolled 形态的 RVEC cycles 不劣于
   unrolled 形态（目标 ≤1.1×）；WHT N=512 的 RVEC cycle 对比作为回归记录。

## 10. 调研依据与事实证据

- LLVM [GetElementPtr FAQ](https://llvm.org/docs/GetElementPtr.html) 说明 GEP 与整数地址算术
  的 object/provenance 边界；本设计据此只在 PTO 已声明的数值地址空间中使用 zero base。
- LLVM [LangRef pointer aliasing rules](https://llvm.org/docs/LangRef.html#pointer-aliasing-rules)
  说明通用 LLVM pointer 不能仅凭数值相等任意重建；这也是 C08 的来源。
- MLIR [Data Layout](https://mlir.llvm.org/docs/DataLayout/) 将 index bitwidth 作为 target
  语义；本设计从 PTO target/data layout 取得位宽，不从 lowering 结果反推。
- LLVM ScalarEvolution 与 `SCEVExpander` 分离符号证明和 SSA 物化；本设计同样把 exact
  quotient proof 与 reifier 分开。
- 当前仓库事实来自 `PTOValueEvolutionAnalysis`、`PTOAddressAnalysis`、
  `VPTOAddressSemantics` 和 `VPTOLLVMEmitterHelper` 的实际实现。
- `.work/issue591-postupdate/` 中的 `original`、`addptr`、`normalized` LLVM/汇编产物记录了
  三种形态的 Bisheng A5 实测结果；它们是诊断证据，不作为提交测试。
- `.work/c15-investigation/` 记录了 C15（`castptr(%param)` 运行时根）两种形态的
  LLVM IR 与 device 二进制对比（2026-08-25 实测，见 6.5）；同为诊断证据。
- 设备编译参数来自 `tools/ptoas/ObjectEmission.cpp` 的
  `compileDeviceLLVMToObject()`（`bisheng --cce-aicore-arch=dav-c310-vec
  --cce-aicore-only -cce-bitcode-is-aicore -c -x ir`）。

设计的最终边界可以概括为：

```text
地址规范化只证明 pointer value 等价；
地址分析只消费 canonical pointer SSA；
operation fold 只合并同单位 offset；
lowering 和 post-update 只作为独立集成观察。
```
