# VPTO 地址递推 i16 规范化设计

## 1. 要处理的问题

### 1.1 VPTO 中存在多种地址递推表示

VPTO 访存 op 的 loop-varying 地址可能来自直接 `index` induction variable、`index`/`i32`/`i16` 固定步长 `scf.for iter_arg`、`pto.addptr` 的 offset，或者 pointer iter_arg backedge 的 advancement。不同 op 又使用 Element、Block、Byte、Alignment 四种地址单位，并且有些 op 没有显式 stride，例如 `vldus`；有些 op 的 stride 只表示状态推进而不参与当前访问地址，例如 `vstus`。

这些地址写法在数学上可能表示同一个固定步长递推，但在 IR 中分散在不同整数类型、cast 和 pointer 表达式中。A5 VPTO 地址生成更偏好窄 i16 递推，因此需要一个独立的 architecture-level canonicalization pass：只要能够证明语义等价，就把地址相关的固定步长递推永久规范到 i16 域，再按原 operand 类型直接使用 i16，或扩展回 i32/index。

例如：

```mlir
// before
scf.for %iv = %c0 to %c4 step %c1 {
  %value = pto.vlds %base[%iv]
      : !pto.ptr<ui8, ub> -> !pto.vreg<256xui8>
}

// after normalization
scf.for %iv = %c0 to %c4 step %c1
    iter_args(%addr16 = %c0_i16) -> i16 {
  %offset = arith.index_cast %addr16 : i16 to index
  %value = pto.vlds %base[%offset]
      : !pto.ptr<ui8, ub> -> !pto.vreg<256xui8>
  %next16 = arith.addi %addr16, %c1_i16 overflow<nsw> : i16
  scf.yield %next16 : i16
}
```

该输出本身就是稳定 canonical form，不以 soft-postupdate 最终成功为前提。

### 1.2 不能直接截断宽递推

[PR #1018 的 review](https://github.com/hw-native-sys/PTOAS/pull/1018#discussion_r3683903350) 给出了源整数类型回绕导致错误 post-update 的例子。循环以 i8 iter_arg 保存 offset，初值位模式为 224，每轮执行 `addi 32`，再通过 `arith.index_castui` 扩展到 index。原程序实际 offset 序列是 `224, 0`；如果忽略 i8 回绕，把它当作普通数学递推，则会错误得到 `224, 256`。

因此，widening cast 本身不能证明 `delta(cast(value)) == cast(delta(value))`。规范化必须证明源类型中的完整递推不回绕，同时证明同一递推在目标 i16 地址域中也不回绕。

[同一 PR 的另一条 review](https://github.com/hw-native-sys/PTOAS/pull/1018#discussion_r3670760383) 还说明不能为了得到窄 stride 而把完整宽地址算术下沉到 i16。normalizer 只规范化能够单独证明的 recurrence leaf，不合并 `delta(base)` 与 `delta(offset)`，不执行地址单位换算，也不计算最终 post stride。

### 1.3 完整证明必须包含最终 backedge

设 `R(k) = I + k * D`，trip count 为 `T`。循环体访存使用 `R(0) .. R(T-1)`，但最后一次循环体仍会实际计算并 yield `R(T)`。即使 `R(T)` 不再用于下一次访存，它仍是原 `scf.for` 的最终 SSA 状态，因此规范化必须保证 `k ∈ [0, T]` 的完整递推在源类型和 i16 目标域中都不回绕。

## 2. 独立 pass 合同

`VPTONormalizeAddressRecurrences` 是独立、永久的 VPTO canonicalization pass，而不是 `VPTOSoftPostUpdate` 的试探性 producer。

它遵循以下合同：

- 假设目标架构偏好 i16 地址递推；只要能够证明安全，就永久改写，不以 soft-postupdate 是否启用或是否成功为条件。
- 不创建 original/canonical 双轨值，不创建 witness，也不存在 commit、rollback 或 consumer 完整性检查。
- 同一 op 的 base、显式 stride 和 pointer advancement 分别尽力规范化；其中一个 leaf 无法证明不会阻止另一个安全 leaf 的改写。
- 如果被替代的宽 iter_arg 及其 update 没有其他语义用户，loop-aware liveness 会把它们从 `scf.for` 中删除，只保留 canonical i16 recurrence。
- 复杂递推、动态 trip count、源类型或 i16 回绕、域不匹配以及存在难以保持的其他用户时保持原 IR。

soft-postupdate 是该 canonical form 的一个可能 consumer，但不是它的所有者。soft-postupdate 失败时，访存 op 保持普通形式并继续使用已经规范化的 i16 地址递推。

## 3. Pipeline 与控制方式

VPTO emission pipeline 固定按以下顺序运行：

```text
VPTOExpandWrapperOps
PTOInferVPTOVecScope
VPTONormalizeAddressRecurrences
[VPTOSoftPostUpdate]
LoopInvariantCodeMotion
PTONarrowVPTOLoopCounters
Canonicalizer
CSE
PTOValidateVPTOEmissionIR
```

`VPTONormalizeAddressRecurrences` 无 CLI 开关，在 VPTO emission pipeline 中始终运行。不是每个 pass 都需要独立开关；这里的 canonicalization 被定义为目标架构的默认 IR 规范。

`--enable-vpto-soft-postupdate` 只控制 `VPTOSoftPostUpdate`。显式传入 `--enable-vpto-soft-postupdate=false` 时，normalizer 仍运行并保留 i16 canonical recurrence，只跳过普通访存到 post-update 形式的转换。

normalizer 不再以 `pto.vecscope` 作为 producer/consumer 所有权边界。独立运行时，它遍历 module 中全部 `scf.for`，并处理共享候选表所描述的地址 operand；`pto.vecscope` 外的安全候选同样可以规范化。soft-postupdate 仍以 `pto.vecscope` 作为自己的循环和顺序分析边界。

## 4. 共享 op 描述与职责边界

候选集合和地址语义集中在 `VPTOPostUpdateUtils` 的 `PostUpdateOpInfo` 表中，normalizer 不维护少数指令的硬编码白名单。该表描述：

- base operand 下标；
- 可选 stride operand 下标；
- stride 是否参与本次访问地址；
- Element、Block、Byte 或 Alignment 地址单位；
- Element 单位对应的元素类型来自 base、某个 operand 或某个 result；
- Signed 或 Unsigned 地址数值域；
- 普通/post 形式的结果数边界；
- Dynamic、Constant 或 SignedI8 最终 post stride 约束。

normalizer 只使用候选集合、base/stride 位置和地址数值域寻找 recurrence leaf。`pto.addptr` offset 按 signed 域处理。它不检查单位和最终 stride 约束，因为 canonicalization 是否有价值不再由 post-update 成功决定。

soft-postupdate 使用完整描述计算有效地址变化：合并 base 与 stride delta、完成单位换算、验证 `vstus` advancement，并检查最终 stride 类型和 Constant/SignedI8 等约束。两个 pass 共享描述和 canonical recurrence 结构，但不会重复完整的 op 级 stride 分析。

共享表覆盖当前 soft-postupdate 的全部候选，包括无显式 stride 的 `vldus` 和 stride 不参与当前地址的 `vstus`。

## 5. 可接受递推与 best-effort 策略

pass 只接受候选 op 所在 `scf.for` 的两种直接整数递推：

1. 直接 induction variable；
2. 初值为常量、backedge 为 `%arg + constant`、`constant + %arg` 或 `%arg - constant` 的 iter_arg。

支持的地址 operand 类型为 `index`、`i32` 和 `i16`。Signed 域通过 `arith.index_cast` 或 `arith.extsi` 从 i16 恢复原 operand 类型；Unsigned 域通过 `arith.index_castui` 或 `arith.extui` 恢复；目标 operand 本身是 i16 时直接使用 canonical iter_arg。

地址相关 use 包括：

- 候选 op 的显式 offset/stride；
- loop-varying base 中 `pto.addptr` 的 offset；
- pointer iter_arg backedge `pto.addptr` 的 advancement offset，包括无显式 stride 的 `vldus`。

同一原递推在相同 signed/unsigned 域中被多个地址 use 复用时共享 canonical recurrence；同一源值被不同地址域消费时分别建立对应域的 recurrence。

normalizer 按 recurrence leaf 独立决策，而不是按最终 post-update candidate 整体决策。例如一个 op 的 base offset 可以安全规范化，而显式 stride 递推过于复杂时，只改写 base offset。最终 op 是否能够 post-update 由后续 pass 独立决定。

为了避免为非地址用户改变数据流或同时保留宽窄双份递推，当前实现要求被规范化的源递推除已识别地址 use 和自身固定步长 update 外没有其他用户，并要求 iter_arg loop result 没有循环外用户。无法满足时保持原 IR。

## 6. 完整安全证明

设常量 trip count 为 `T`，初值为 `I`，每次 backedge 增量为 `D`。实现以 128-bit 中间值检查闭区间 `k ∈ [0, T]` 上的：

```text
R(k) = I + k * D
```

因为 `D` 固定，序列单调，只需检查初值和最终 backedge 两个端点。

Signed 域必须同时满足源类型范围和：

```text
-32768 <= R(k) <= 32767
```

Unsigned 域必须同时满足源类型范围和：

```text
0 <= R(k) <= 65535
```

Block 类 `vsldb/vsstb` 的 i16 stride 按 unsigned 域处理，因此位模式 40000 不会仅因超过 signed i16 上界而被拒绝；完整递推超过 65535 时仍保持原 IR。

trip count、初值和增量必须为常量。动态边界、零或负 loop step、非线性 backedge、多层递推和无法证明的 cast 均不推断。证明还检查 increment 能否以目标 i16 算术准确表达。

canonical recurrence 使用标准 overflow flag 保存证明结论：

```mlir
// Signed growth.
%next = arith.addi %addr16, %step16 overflow<nsw> : i16

// Unsigned growth.
%unext = arith.addi %uaddr16, %ustep16 overflow<nuw> : i16

// Unsigned descent.
%udec = arith.subi %uaddr16, %decrement16 overflow<nuw> : i16
```

这些 flag 是 canonical IR 自身的 no-wrap 语义，不是传给某个特定 consumer 的临时 witness。任何后续分析都可以据此识别固定步长 i16 recurrence。

## 7. 永久输出与 soft-postupdate 输入

normalizer 直接把候选地址 operand 接到 canonical value：

```mlir
%offset = arith.index_cast %addr16 : i16 to index
%value = pto.vlds %base[%offset] : ...
```

旧设计中用于同时保存 original/canonical 的可逆 marker 已删除，最终 IR 只保留直接接入 operand 的 canonical value。

soft-postupdate 对 loop-varying 窄整数只接受以下 canonical 结构：i16 `scf.for iter_arg`、常量步长 backedge、匹配地址域的 `nsw`/`nuw`，以及必要时匹配 signedness 的 widening cast。它读取常量 step，不重复 trip count、初值和端点证明。

如果 soft-postupdate 的完整分析成功，它可以用 pointer recurrence 替代普通访存地址，并通过共享 liveness 删除不再需要的 i16 recurrence。如果完整分析因单位、最终 stride、支配性、零 stride、Constant/SignedI8 或其他条件失败，普通访存 op 和 canonical i16 recurrence 都保留。

非循环 sequential 分析不依赖 normalizer 输出，继续直接分析同一 block 中相邻访存的有效地址差。循环中未形成 post-update 的普通 op 可能已经使用 canonical i16 recurrence，这是独立 normalization 的预期稳定形态。

## 8. 测试覆盖

lit 回归覆盖：

- index、i32、i16 三类 operand；
- Element、Block、Byte、Alignment 四类地址单位；
- 无显式 stride 的 `vldus`；
- stride 不参与当前地址的 `vstus`；
- signed/unsigned source wrap、i16 域 wrap 和最终 backedge wrap 拒绝；
- Constant 或 SignedI8 最终 post stride 拒绝时，普通访存仍保留 canonical i16 recurrence；
- 同一循环中 post-update 成功和失败候选混合时，失败候选不回退；
- `pto.vecscope` 外候选也可由独立 normalizer 规范化；
- `--enable-vpto-soft-postupdate=false` 只关闭 soft-postupdate，不关闭 normalization；
- 最终 IR 不包含 witness，且被完全替代的宽递推由 loop-aware liveness 删除。

SIM/runtime 回归继续验证 source wrap、normalized recurrence 类型、混合成功/失败、下降递推和嵌套共享 chain。由于 normalizer 的 canonical form 现在独立保留，runtime 形态检查应分别验证普通访存地址递推和实际 post-update 是否符合各自合同。
