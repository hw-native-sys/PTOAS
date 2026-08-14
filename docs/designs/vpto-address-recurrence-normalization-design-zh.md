# VPTO 地址递推 i16 类型收窄设计

## 1. 目标

A5 VPTO 地址生成偏好 i16。`VPTONormalizeAddressRecurrences` 因此独立于
soft-postupdate 运行：只要能够证明完整地址递推在 i16 中不回绕，就永久把递推
的存储类型收窄为 i16。

该 pass 的核心约束是：**只改变类型，不改变递推结构**。

- 原来是 `scf.for` induction variable（IV）的地址仍由同一个 IV 驱动；
- 原来是 `scf.for iter_arg` 的地址仍使用原来的 iter_arg 槽位；
- 不为地址创建额外的影子 iter_arg，不把 delta 形式改成 accumulator 形式；
- pointer、align 及其他无关 iter_arg 的数量、顺序和数据流保持不变；
- 原 operand 需要 index 或 i32 时，在循环体入口或地址 use 前从 i16 扩展回原类型。

soft-postupdate 可以消费收窄后的 IR，但不是该 IR 的所有者。即使关闭或拒绝
soft-postupdate，安全的 i16 类型收窄仍然保留。

## 2. 两种结构保持改写

### 2.1 IV 递推

改写前：

```mlir
%c0 = arith.constant 0 : index
%c4 = arith.constant 4 : index
%c1 = arith.constant 1 : index
scf.for %iv = %c0 to %c4 step %c1 {
  %value = pto.vlds %base[%iv]
      : !pto.ptr<ui8, ub> -> !pto.vreg<256xui8>
}
```

改写后仍是 IV/delta 形式，只收窄 loop control：

```mlir
%c0_i16 = arith.constant 0 : i16
%c4_i16 = arith.constant 4 : i16
%c1_i16 = arith.constant 1 : i16
scf.for %iv16 = %c0_i16 to %c4_i16 step %c1_i16 : i16 {
  %iv = arith.index_cast %iv16 : i16 to index
  %value = pto.vlds %base[%iv]
      : !pto.ptr<ui8, ub> -> !pto.vreg<256xui8>
}
```

这里不会创建 `%addr16` iter_arg，也不会显式合成 `%addr16 + step` 的回边。
循环控制仍由 `scf.for` 自身表达，后续 delta 分析仍从 IV 读取 loop step。

### 2.2 iter_arg 递推

改写前：

```mlir
scf.for %iv = %lb to %ub step %step
    iter_args(%offset = %c0_i32) -> i32 {
  pto.sprsts "AR", %base[%offset] : !pto.ptr<ui32, ub>, i32
  %next = arith.addi %offset, %c1_i32 : i32
  scf.yield %next : i32
}
```

改写后仍是同一个 iter_arg 槽位：

```mlir
scf.for %iv = %lb to %ub step %step
    iter_args(%offset16 = %c0_i16) -> i16 {
  %offset = arith.extsi %offset16 : i16 to i32
  pto.sprsts "AR", %base[%offset] : !pto.ptr<ui32, ub>, i32
  %next16 = arith.addi %offset16, %c1_i16 overflow<nsw> : i16
  scf.yield %next16 : i16
}
```

pass 替换原槽位的 init、block argument、backedge 和无外部用户的 loop result
类型，不追加新槽位。Signed/Unsigned 域分别使用 `nsw`/`nuw` 记录已经完成的
no-wrap 证明；同一递推同时满足两个域时可以同时携带两种 flag。

## 3. 安全证明

设固定步长递推为：

```text
R(k) = I + k * D
```

常量 trip count 为 `T`。循环体访问 `R(0)..R(T-1)`，最后一次迭代仍会计算
IV 的退出更新或 iter_arg 的最终 backedge `R(T)`，因此证明覆盖闭区间
`k ∈ [0, T]`。

实现用 128-bit 中间值检查两个端点。因为 `D` 固定，序列单调，端点足以覆盖
整个区间。

- Signed 地址域要求源类型和 i16 中均满足 `-32768 <= R(k) <= 32767`；
- Unsigned 地址域要求源类型和 i16 中均满足 `0 <= R(k) <= 65535`；
- increment 必须能由 i16 加/减准确表达；
- trip count、初值和步长必须为常量，loop step 必须为正；
- 动态边界、非线性 backedge、源类型回绕或 i16 回绕均保持原 IR。

IV 的收窄还必须保持 `scf.for` 自身的有符号循环控制语义。因此 lower、upper、
step 和最终 exit value 都必须落在 signed i16 范围内。地址递推虽然可能满足
unsigned i16，但只要循环控制不能安全收窄，就不会以新增 iter_arg 的方式绕过
该限制。

这也防止了经典的窄整数回绕误判。例如 i8 地址位模式序列 `224, 0` 不能被当作
数学序列 `224, 256`；widening cast 本身不是 no-wrap 证明。

## 4. 可接受输入和 best-effort 策略

pass 只匹配候选所在 `scf.for` 的两种直接整数递推：

1. 直接 IV；
2. 常量初值、回边为 `%arg + constant`、`constant + %arg` 或
   `%arg - constant` 的 iter_arg。

支持 `index`、i32 和 i16 地址 leaf。地址相关 use 包括：

- 候选 op 的显式 offset/stride；
- 已经是 post-update 形式的候选 op 仍按相同 operand 位置处理；
- loop-varying `pto.addptr` 的整数 offset；
- pointer iter_arg 回边 `pto.addptr` 的 advancement，包括 `vldus`；
- `vstus` 中只负责状态推进、不参与当前访问地址的 stride。

Signed 域用 `arith.index_cast`/`arith.extsi` 恢复原 operand 类型，Unsigned 域用
`arith.index_castui`/`arith.extui`。目标 operand 已是 i16 时直接使用原 IV 或
原 iter_arg 槽位。

每个 recurrence leaf 独立决策。一个 op 的 base offset 能安全收窄、显式 stride
过于复杂时，只收窄 base leaf。源值的其他循环体用户通过一次 widening cast
继续观察原类型；iter_arg result 有循环外用户时，也在循环后扩展回原 result
类型。pointer result 的外部用户、`pto.addptr` 的额外用户和 pointer 自身的数据流
只影响 post-update 是否成立，不影响其整数 offset/advancement leaf 的独立收窄。
因此额外用户不会迫使 pass 建立宽窄双份回边，也不会阻止安全收窄。

## 5. 与通用 loop-counter 收窄的关系

`PTONarrowVPTOLoopCounters` 负责 vecscope 中一般常量边界循环的 IV 收窄；地址
normalizer 使用同一个 `canNarrowLoopCounterToI16` 安全判定。两者对 IV 的结构
语义一致：重建同一 `scf.for` 的 i16 control values，并在循环体中恢复原 IV
类型，不增加 iter_arg。

地址 normalizer 仍独立遍历 module 中的候选循环，因此 vecscope 外的安全地址
IV 也可被收窄；通用 pass 自身仍保持 vecscope 范围。

## 6. soft-postupdate 消费合同

soft-postupdate 接受两类收窄后的地址 leaf：

- i16 IV：重新检查常量 bounds、正 step 和最终 exit update 均满足 signed i16；
- i16 iter_arg：检查固定步长回边以及与地址域匹配的 `nsw`/`nuw` flag。

必要的 widening cast 必须与地址域 signedness 一致。动态边界 i16 IV、无 no-wrap
flag 的 i16 iter_arg、i8/i32 loop-varying cast 或 signedness 不匹配都保守拒绝。

分析分支保持原有分类：

- IV 派生地址继续走 delta 分析；
- iter_arg 派生地址继续走 accumulator 分析；
- normalizer 不为了方便 consumer 而在两类结构之间转换。

soft-postupdate 成功时可按自身合同新增 pointer iter_arg，并用 post-update op 的
`updated_base` 形成指针回边。这个结构变化属于 post-update 优化，不属于类型
收窄。若 soft-postupdate 失败，普通访存继续使用已经收窄的原 IV/iter_arg。

## 7. Pipeline

VPTO emission 的相关顺序为：

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

normalizer 无独立 CLI 开关并始终运行。`--enable-vpto-soft-postupdate=false` 只跳过
soft-postupdate，不回退 i16 类型收窄。

候选 op、base/stride 位置、地址单位、Signed/Unsigned 域和最终 stride 约束由
`VPTOPostUpdateUtils` 的共享 `PostUpdateOpInfo` 表描述。normalizer 只使用其中的
候选位置和数值域证明 leaf；地址单位换算、base/stride delta 合并以及
Constant/SignedI8 等最终约束仍由 soft-postupdate 完成。

## 8. 测试覆盖

lit 回归应明确验证：

- index/i32/i16，以及 Element/Block/Byte/Alignment 地址单位；
- direct IV 收窄后仍是 IV，normalizer 不新增 iter_arg；
- i32/index iter_arg 收窄后仍占原槽位，其他 iter_arg 数量和顺序不变；
- signed/unsigned source wrap、i16 域 wrap 和最终 backedge wrap 均拒绝；
- 动态 trip count IV 不因类型恰好为 i16 就被 soft-postupdate 信任；
- `vldus`、`vstus`、混合成功/失败候选和 vecscope 外候选；
- soft-postupdate 同时消费收窄后的 IV/delta 和 iter_arg/accumulator；
- `--enable-vpto-soft-postupdate=false` 时仍保留结构不变的 i16 收窄。

SIM/runtime 回归继续负责验证 source wrap、下降递推、地址单位、payload 元素宽度
和实际 post-update 指针序列；lit 负责锁定本设计最关键的 IR 结构合同。
