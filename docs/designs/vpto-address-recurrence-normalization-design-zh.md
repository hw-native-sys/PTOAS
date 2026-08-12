# VPTO 地址递推 i16 前置规范化设计

## 1. 要处理的问题

### 1.1 soft-postupdate 需要什么地址信息

[VPTO Soft Post-Update 优化 Pass 设计](vpto-soft-postupdate-design-zh.md)中的 `VPTOSoftPostUpdate` 要把循环中每轮重新计算的访存地址改写成由访存 op 返回、再通过 `scf.for iter_args` 携带的 pointer recurrence。例如普通 `vlds` 每轮用归纳变量重新计算 `base + offset`：

```mlir
scf.for %iv = %c0 to %c4 step %c1 {
  %value = pto.vlds %base[%iv]
      : !pto.ptr<ui8, ub> -> !pto.vreg<256xui8>
}
```

post-update 形式则让访存 op 使用当前指针并返回下一轮指针：

```mlir
scf.for %iv = %c0 to %c4 step %c1 iter_args(%ptr = %base) {
  %value, %next = pto.vlds %ptr[%c1]
      : !pto.ptr<ui8, ub> -> !pto.vreg<256xui8>, !pto.ptr<ui8, ub>
  scf.yield %next
}
```

这个变换需要解决两个不同层次的问题。第一层是单个 loop-varying 地址值能否跨越 `index`、`i32`、`i16` 及 signed/unsigned cast，被安全地解释为固定步长的 i16 地址递推。第二层是一个访存 op 的完整有效地址如何变化，即如何合并 `delta(base)` 与 `delta(offset/stride)`、如何处理 Element、Block、Byte、Alignment 单位，以及最终 post stride 是否满足 `Dynamic`、`Constant` 或 `SignedI8` 约束。

本 pass 只解决第一层，并把证明结果以可逆形式交给 soft-postupdate；soft-postupdate 继续负责第二层的唯一一次完整分析。这样不会在两个 pass 中重复实现 accumulator/delta、单位换算和最终 stride 判断。

### 1.2 原始 VPTO IR 与 post-update 消费形式之间的缺口

实际 VPTO IR 中，loop-varying 地址不只有直接 `index` IV。base 的 `pto.addptr` offset、访存 op 的显式 offset/stride，以及 pointer iter_arg 的 advancement stride，都可能来自 `index`、`i32` 或 `i16` 的固定步长 iter_arg。同一递推还可能先以 i32 传给 `vstus`，再转成 index 用于 base 的 `pto.addptr` backedge。

这里不能把“地址相关 operand”简单等同于 i16。Element 类 offset 多为 index，部分 scalar、predicate 和 unaligned-store 候选使用 i32，`vsldb/vsstb` 的 block/repeat stride 才是 i16，而 post 形式返回的 `updated_base` 是 pointer。原 op 的地址 operand 通常确实是 index、i32 或 i16，但 operand 类型本身不能证明跨 cast 后的每轮地址差仍等于源递推的固定数学增量。

如果 soft-postupdate 直接承担所有窄整数范围证明，它的 accumulator/delta 分析就必须同时理解循环次数、源类型回绕、cast signedness 和 i16 端点范围。把这些证明拆到前置 pass 后，soft-postupdate 只需识别一种有明确 no-wrap 语义的 canonical recurrence，同时仍使用原有三类路径：base accumulator、stride accumulator 和 delta fallback。前置 pass 不替代这些路径，只为它们提供一个安全的 loop-varying integer leaf。

### 1.3 PR #1018 暴露的主要正确性问题：widening cast 不一定保持 loop delta

[PR #1018 的 review](https://github.com/hw-native-sys/PTOAS/pull/1018#discussion_r3683903350) 给出了触发错误代码的最小例子。循环以 i8 iter_arg 保存 offset，初值位模式为 224（写作 `-32 : i8`），每轮执行 `addi 32`，访存前再通过 `arith.index_castui` 扩展到 index：

```mlir
scf.for ... iter_args(%off8 = %cm32_i8) -> i8 {
  %off = arith.index_castui %off8 : i8 to index
  %value = pto.vlds %base[%off]
  %next = arith.addi %off8, %c32_i8 : i8
  scf.yield %next : i8
}
```

原程序的 i8 递推在第一轮 backedge 回绕，因此扩展后的实际 offset 序列是 `224, 0`。旧分析仅根据 `resultWidth >= inputWidth` 认定 widening cast 保持 loop delta，再从 `addi 32` 推导 post-update stride 32，生成的 pointer recurrence 却访问 `224, 256`。该 review 还记录了 simulator 对照结果：关闭优化时输出一致，开启后出现 255 个 mismatch。

因此，引入本 pass 的主要原因不是最后一次 backedge 本身，而是不能把源类型中的模运算递推重新解释成宽地址域中的普通数学递推。对于穿越类型域边界的 loop-varying 值，只有证明完整源递推不回绕，才能认为 `delta(cast(value)) == cast(delta(value))`。本 pass 只接受 index、i32 和 i16；i8 例子用于定义证明合同。

### 1.4 地址数学语义与硬件窄 stride 编码必须分离

[同一 PR 的另一条 review](https://github.com/hw-native-sys/PTOAS/pull/1018#discussion_r3670760383) 指出了相关但不同的问题：地址差原本在 index/宽整数域中计算，却在物化 post stride 时通过消除 cast 把算术下沉到了 i16。例子中 `%k : i16` 先经 `arith.index_castui` 零扩展到 index，f32 `vsstb` 的相邻地址差为 `2 * zext(k)` 个 32-byte block；当 `%k` 的位模式表示无符号 40000 时，原地址差为 80000 blocks，而直接在 i16 中计算 `2 * %k` 会回绕成 14464。

normalizer 因此只能规范化自身能够完整证明的固定步长 leaf，不能通过去掉 cast 或把完整地址表达式提前下沉到 i16 来制造一个表面合法的 stride。soft-postupdate 在更宽的原数学语义中合并 base 和 stride delta，并在最终物化点检查目标 operand 类型、单位换算和硬件 stride 约束。

最终 backedge 是完整证明的一项边界。设 `R(k) = I + k * D`、trip count 为 `T`，实际访存使用 `R(0) .. R(T-1)`，最后一次循环体仍会计算并 yield `R(T)`。即使 `R(T)` 不再用于下一次访存，它仍是原 `scf.for` 的最终 SSA 状态；规范化必须保证这次实际执行的 backedge 在源类型和目标 i16 域中都不回绕。

### 1.5 为什么输出必须可回滚

normalizer 只能证明单个 leaf 的 i16 递推等价，不能据此保证整个访存 op 最终能够 post-update。例如 base delta 和 offset delta 合并后可能超出 `SignedI8`，Block 单位换算可能不整除，`vstus` 的 advancement 可能与 `delta(base)` 不相等，或者 soft-postupdate 的类型与可用性检查可能失败。循环路径中通过通用 `combineStride` 合并地址 delta 的候选还会在最终 stride 为零时拒绝；`vstus` 走独立的 advancement 等价检查。结构性收益检查只用于后续顺序路径。

如果 normalizer 直接把普通访存 operand 永久替换为 canonical shadow，即使 soft-postupdate 最终失败，程序通常仍因前置证明而保持语义正确，但循环会无收益地多出 i16 iter_arg、backedge 算术和扩展，增加寄存器压力与代码体积，并可能妨碍后续循环和地址优化。这正是需要回滚的性能问题。

因此，normalizer 通过内部 `pto.address_recurrence_witness` 同时携带原值和 canonical 值。soft-postupdate 分析时读取 canonical 一侧，在提交任何 post-update 改写前将所有普通 op 恢复为 original 一侧；成功候选随后被 post-update op 取代，失败候选继续使用原地址表达式，死亡的 shadow recurrence 由 loop-aware liveness 删除。一个循环中可以同时提交成功候选并回滚失败候选，witness 不得越过 soft-postupdate。

## 2. 目标与流水线位置

`VPTONormalizeAddressRecurrences` 是 `VPTOSoftPostUpdate` 的保守前置 pass：

```text
... -> vpto-normalize-address-recurrences -> vpto-soft-postupdate -> LICM ...
```

VPTO 后端默认启用 `--enable-vpto-soft-postupdate`；启用时两个 pass 总是按上述顺序运行，显式传入 `--enable-vpto-soft-postupdate=false` 时则同时跳过两者。测试工具可以单独使用 `-vpto-normalize-address-recurrences` 检查 witness 和 canonical shadow，但这种中间 IR 不是可进入 lowering 的稳定形式。

normalizer 的职责是识别并证明简单地址递推、创建 canonical shadow 和可逆 witness。soft-postupdate 的职责是消费 witness、完成 op 级 accumulator/delta、地址单位、最终 stride 与合法性分析，并对顺序路径执行结构性收益检查，然后恢复普通 operand 并提交成功项。normalizer 证明失败时完全不改写候选；normalizer 成功而 soft-postupdate 失败时由 soft 回滚。

两个 pass 共享同一个 `pto.vecscope` 所有权边界。normalizer 只处理 `pto.vecscope` 内的 `scf.for`，因为 soft-postupdate 也只在该边界内进行循环与顺序分析；`pto.vecscope` 外的候选 op 即使递推形状可证明，也保持原样，不创建 canonical shadow 或 witness。这样 paired pipeline 中每个 witness 都必然进入 consumer 的提交或回滚流程，避免 producer 在 consumer 不会访问的区域留下临时 IR 并导致最终 witness 完整性检查失败。

## 3. 共享 op 描述

候选集合和地址语义集中在 `VPTOPostUpdateUtils`，normalization 与 soft-postupdate 不各自维护指令白名单。每个 `PostUpdateOpInfo` 包含：

- base operand 下标；
- 可选 stride operand 下标；
- stride 是否参与本次访问地址；
- stride 的 `Element`、`Block`、`Byte` 或 `Alignment` 单位；
- stride 的 `Signed` 或 `Unsigned` 地址数值域；
- 普通/post 形式的结果数边界；
- post stride 的 `Dynamic`、`Constant` 或 `SignedI8` 最终约束。

两者共享整张描述表，但消费不同字段。normalizer 用候选、base/stride 位置和 stride 地址数值域定位需要证明的 recurrence leaf；`pto.addptr` base offset 固定按 signed 域处理。soft-postupdate 使用全部字段计算有效地址、单位换算和最终约束。描述信息共享不等于两个 pass 重复进行完整 stride 分析。

普通 Auto 类 op 使用 `current_address = base + stride`；`vldus` 没有普通形式的显式 stride，normalizer 从 loop-varying base 的 `pto.addptr` offset 或 pointer advancement 找到 leaf；`vstus` 的 stride 不参与本次访问地址，只推进 unaligned-store 状态和返回 base，因此最终仍由 soft-postupdate 检查 `delta(base) == advancement` 并保留原 stride operand。

## 4. 可接受递推

pass 只接受直接位于候选 op 所在 `scf.for` 的两种整数递推：

1. 直接 induction variable；
2. 初值为常量、backedge 为 `%arg + constant`、`constant + %arg` 或 `%arg - constant` 的 iter_arg。

这个覆盖面故意小于 soft-postupdate 的完整 accumulator/delta 表达式分析。normalizer 只需要把可能跨越窄整数类型域的 leaf 变成可证明的 canonical 输入；base 与 offset 的组合、多项仿射表达式以及 pointer accumulator 仍由 soft-postupdate 处理。纯 index 域内且没有窄整数来源的 sequential 和 loop recurrence 也继续直接由 soft-postupdate 分析。

支持的原 operand 类型为 `index`、`i32` 和 `i16`。Signed 域的 shadow 通过 `arith.index_cast` 或 `arith.extsi` 恢复为 index/i32，unsigned 域通过 `arith.index_castui` 或 `arith.extui` 恢复；目标 operand 为 i16 时直接使用 shadow。paired pipeline 中，原 operand 已是 i16 时 normalizer 也创建独立 shadow 和 witness，使失败候选仍能恢复到完全相同的原递推。soft-postupdate 单独运行时并不要求 canonical i16 必须来自 shadow；它按结构和 overflow flag 识别证明证书。

地址相关 use 包括：

- 候选 op 的显式 offset/stride；
- `pto.addptr` 形式 loop-varying base 的 offset；
- pointer iter_arg backedge `pto.addptr` 的 advancement offset，包括无显式 stride 的 `vldus`。

同一原递推在相同 signed/unsigned 域中被多个地址 use 复用时可以共享 shadow；同一源值被不同地址域消费时分别创建 shadow。原递推始终保留到 soft-postupdate 作出最终决定。

## 5. 完整安全证明

证明要求由候选 op 的地址数值域决定，而不是统一把所有 i16 当作有符号整数。

### 5.1 跨类型域规范化

设常量 trip count 为 `T`，原递推初值为 `I`，每次 backedge 增量为 `D`。证明检查闭区间 `k ∈ [0, T]`：

```text
R(k) = I + k * D
```

由于 `D` 固定，序列单调，只需用 128-bit 中间计算检查两个端点。实现把 `index` 按 64-bit 地址整数域建模。Signed 域必须同时满足：

```text
signed_min(source_type) <= R(k) <= signed_max(source_type)
-32768                  <= R(k) <= 32767
```

第二组约束保证 i16 shadow 不回绕；第一组约束保证规范化没有把依赖 index/i32/i16 源类型回绕的序列误解释成数学整数递推。signed 增量必须能以 i16 常量表达。

Unsigned 域使用对应边界：

```text
unsigned_min(source_type) <= R(k) <= unsigned_max(source_type)
0                         <= R(k) <= 65535
```

Block 类 `vsldb/vsstb` 的 i16 stride 按 unsigned 域处理，因此位模式 40000 不会仅因超过 signed i16 上界而被拒绝；但完整递推超过 65535 时仍保持原 IR。unsigned 增长使用可表示的 i16 位模式常量，unsigned 下降只接受 `%arg - constant` 并把正的 decrement 物化为 `arith.subi`。

trip count、初值和增量必须全为常量。动态边界、零或负 loop step、非线性和多层 backedge 表达式均不推断。

### 5.2 canonical 证明证书与恢复关系

normalizer 使用算术 op 的标准 no-wrap 语义表达证明结果。Signed 增长递推输出：

```mlir
%next = arith.addi %addr16, %step16 overflow<nsw> : i16
```

Unsigned 增长递推使用 `overflow<nuw>`；unsigned 下降递推规范为 `arith.subi ... overflow<nuw>`。overflow flag 对每次实际执行的 backedge 都成立，因此包含最终 backedge 的证明结论。soft-postupdate 只检查 i16 iter_arg、常量步长、对应 signed/unsigned extension 和 `nsw/nuw` 是否组成 canonical 结构，不再重复读取 trip count、初值和端点。

`pto.address_recurrence_witness` 不承担正确性证明，它只记录可恢复的 SSA 对应关系：

```mlir
%canonical = arith.index_cast %addr16 : i16 to index
%address = pto.address_recurrence_witness %original, %canonical : index
%value = pto.vlds %base[%address] : ...
```

normalizer 单独运行后的候选通过 witness 指向 canonical 值。soft-postupdate 的分析递归解开 witness 并读取 canonical 一侧；在改写阶段开始前，所有 witness result 的普通用途先替换回 original 一侧。成功项不再需要普通地址 operand，失败项则精确保留原 operand。之后删除 witness、无用户的 cast、常量以及 shadow iter_arg。

原生 i16 operand 也映射到新的 shadow：如果它按目标域回绕，normalizer 无法合法添加对应 overflow flag，因此不产生 witness。这不是要求所有 i16 在所有语义下都满足 signed no-wrap，而是按实际候选的 signed/unsigned 地址域证明。

## 6. 两阶段拒绝与回滚

normalizer 在以下情况不产生 witness，IR 保持原样：

- 完整递推或最终 backedge 超出源类型或目标 signed/unsigned i16 域；
- iter_arg loop result 在循环外有用户；
- 递推值除已识别地址 use 和自身固定步长 update 外还有难以保持的用户；
- base、offset 或 advancement 经过无法证明等价的算术或控制流；
- trip count、初值或固定增量无法静态证明。

normalizer 成功后，soft-postupdate 仍可能因以下 op 级条件拒绝候选，此时必须恢复 original operand 并清理 shadow：

- Element、Block、Byte 或 Alignment 换算不精确，或目标 alignment 未知；
- `vstus` 的 base delta 与 advancement stride 不相等；
- 合并后的 post stride 不满足 `Constant`、`SignedI8`、目标类型或可用性约束；
- 支配性、类型、零 stride 或其他完整 post-update 合法性检查失败；
- 顺序路径未满足长度和结构性收益条件。

normalizer 按候选和关联 leaf 先整体分析，再重建 `scf.for`，不会留下“只有部分 leaf 有 witness”的候选。soft-postupdate 则按循环形成 rewrite plan，恢复该循环的全部 witness 后只提交成功 plan；即使同一循环同时存在成功和失败候选，失败普通 op 仍使用原地址，成功项共享或建立自己的 pointer chain。

## 7. soft-postupdate 输入契约

soft-postupdate 接受以下 loop-varying canonical form。paired pipeline 中这些值位于 witness canonical 一侧；单独运行 consumer 时也可以直接来自输入 IR：

```mlir
// Signed domain.
%idx = arith.index_cast %addr16 : i16 to index
%i32 = arith.extsi %addr16 : i16 to i32
%next = arith.addi %addr16, %step16 overflow<nsw> : i16

// Unsigned domain.
%uidx = arith.index_castui %uaddr16 : i16 to index
%ui32 = arith.extui %uaddr16 : i16 to i32
%unext = arith.addi %uaddr16, %ustep16 overflow<nuw> : i16
```

consumer 根据共享 op 描述选择预期域；存在 witness 时先解开 canonical 一侧，再匹配对应 extension 与 overflow flag，并读取常量 backedge step。原生 loop-varying i16 iter_arg 只有本身符合相同 canonical 结构、带域匹配的 `nsw`/`nuw` constant-step backedge 时才接受；raw i32 iter_arg、域不匹配的 cast、没有对应 overflow flag 的 i16 递推和动态或复杂窄整数 backedge 均拒绝。纯 index 域内且没有窄整数来源的地址表达式继续使用 soft-postupdate 原有 accumulator/delta 与 sequential 分析，不要求先变成 i16 shadow。

顺序路径不需要另一套 normalization 输出。循环路径完成最终判定并恢复 witness 后，soft-postupdate 才重新收集 block；因此非循环 `SequentialRun` 和循环内未被循环路径消费的普通 op 都看到 original 地址表达式，继续使用原有 sequential 分析。

## 8. 覆盖矩阵

lit 回归覆盖：

- `index/i32/i16` 三类 operand；
- Element、Block、Byte、Alignment 四类地址单位；
- 无显式 stride 的 `vldus`；
- stride 不参与当前地址、且必须匹配 base advancement 的 `vstus`；
- signed/unsigned source wrap、i16 shadow wrap、最终 backedge 和缺失或错误 overflow flag 拒绝；
- `Constant` 与 `SignedI8` 最终约束失败时回滚；
- 同一循环中一个候选成功提交、另一个候选失败回滚；
- `pto.vecscope` 外的候选循环保持原样，不产生无法消费的 witness；
- paired pipeline 结束后不存在 `pto.address_recurrence_witness`、无收益 i16 shadow 或 overflow backedge；
- CLI pipeline 中 normalization 固定先于 soft-postupdate，并在成功时形成 post base chain。

SIM/runtime 回归使用同一份 `kernel.pto` 同时接受 lit 形态检查和 `test/vpto` 严格输出比较，避免“普通形式执行正确但优化其实未触发”的假阳性：

- `soft-post-update-wrap-regressions` 覆盖 i8 unsigned 源回绕、signed i16 源回绕和仅 i16 shadow 回绕，三者均保持普通 load；
- `soft-post-update-normalized-recurrence-types` 覆盖 signed i16、signed i32、unsigned i16 递推，以及 `vsstb`、`vstas` 自动改写；
- `soft-post-update-mixed-commit-rollback` 覆盖同循环中 `vlds` 成功提交而 `sprsti` 因 SignedI8 约束回滚；
- `soft-post-update-descending-recurrence` 覆盖非零高地址上的 signed 负向 stride；
- `soft-post-update-nested-shared-chain` 覆盖 load/store 共享 pointer chain、Element/Byte 单位隔离和内外层循环各自递推。

所有 runtime compare 都按完整 4096-byte 输出执行严格逐字节比较；paired pipeline 的 lit 检查同时要求最终不存在 witness。
