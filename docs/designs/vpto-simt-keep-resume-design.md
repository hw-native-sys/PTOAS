# VPTO SIMT `pto.keep` / `pto.resume` 设计

## 1. 背景

当前 VPTO 已支持通过 `pto.store_vfsimt_info` + `func.call @simt_func(...)` + `pto.get_tid_x/y/z` 表达一次 SIMT VF 调用。

硬件还存在一个额外语义：两个连续的 SIMT VF 调用之间，SIMT 寄存器状态不会被自动清零。前一个 SIMT VF 写入的部分逻辑值，可以在后一个 SIMT VF 中稳定读取。

这类语义如果不显式建模，后续 pass 可能把它当成普通无依赖调用重排、复制或删除。

因此需要引入一对新的 VPTO op：

- `pto.keep`
- `pto.resume`

用于显式表达这段跨 SIMT VF 调用的隐式逻辑数据流。

---

## 2. 设计目标

- 显式表达“保留哪些 SSA 值、恢复哪些 SSA 值”。
- 不把这段语义建模成普通可长期持有的 SSA `return` 值，避免误导。
- 不允许 `!pto.vreg` / `!pto.mask` 进入这套机制。
- 让 verifier 能按 slot 和 payload 做强约束。
- 让 lowering 可以先消费语义，再决定是否擦除或映射 intrinsic。

非目标：

- 不做任意 CFG 上的远距离传播。
- 不支持一个 slot 对多个消费者。
- 不支持跨 host / kernel 传播。

---

## 3. 核心语义

`pto.keep` 和 `pto.resume` 通过一个编译期常量 `slot` 配对。

- `pto.keep`
  - 绑定当前 `simtvf` 内显式列出的 SSA 值
  - 把这组值记录到指定 `slot`

- `pto.resume`
  - 从指定 `slot` 恢复之前 `pto.keep` 保存的那组值
  - 重新物化成新的 SSA 值供后续使用

这是一条**逻辑数据流**，不是普通 `return` 值语义。

---

## 4. IR 接口

### 4.1 `pto.keep`

语法：

```mlir
pto.keep %a {slot = 0} : i32
```

语义：

- 显式保存一个 SSA 值到 `slot`
- 支持不超过 64 bit 的整数标量，以及 `f16` / `bf16` / `f32`
- 64-bit 整数值占用 `slot` 和 `slot + 1`，因此 `slot` 必须为偶数

### 4.2 `pto.resume`

语法：

```mlir
%x = pto.resume {slot = 0} : i32
```

语义：

- 从 `slot` 恢复出之前保存的值
- 恢复结果类型必须和 `keep` 的 payload 类型完全一致
- 支持不超过 64 bit 的整数标量，以及 `f16` / `bf16` / `f32`
- 64-bit 整数值占用 `slot` 和 `slot + 1`，因此 `slot` 必须为偶数

### 4.3 示例

```mlir
module attributes {pto.target_arch = "a5"} {
  func.func @kernel(%dst: !pto.ptr<i32, gm>) {
    %c0_i64 = arith.constant 0 : i64
    %c32_i64 = arith.constant 32 : i64
    %c128_i64 = arith.constant 128 : i64
    %dim_z = arith.constant 1 : i32
    %dim_y = arith.constant 32 : i32
    %dim_x = arith.constant 32 : i32
    %ub_out = pto.castptr %c0_i64 : i64 -> !pto.ptr<i32, ub>

    pto.store_vfsimt_info %dim_z, %dim_y, %dim_x : i32, i32, i32
    func.call @simt_stage0(%ub_out) : (!pto.ptr<i32, ub>) -> ()
    func.call @simt_stage1(%ub_out) : (!pto.ptr<i32, ub>) -> ()

    pto.set_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]
    pto.wait_flag["PIPE_V", "PIPE_MTE3", "EVENT_ID0"]
    pto.dma_store %ub_out, %dst, %c128_i64
      nburst(%c32_i64, %c128_i64, %c128_i64)
      : !pto.ptr<i32, ub>, !pto.ptr<i32, gm>, i64, i64, i64, i64
    return
  }

  func.func @simt_stage0(%dst: !pto.ptr<i32, ub>) attributes {pto.simt_entry} {
    %tid = pto.get_tid_x : i32
    pto.keep %tid {slot = 0 : i64} : i32
    return
  }

  func.func @simt_stage1(%dst: !pto.ptr<i32, ub>) attributes {pto.simt_entry} {
    %tid2 = pto.resume {slot = 0 : i64} : i32
    %tid = pto.get_tid_x : i32
    %idx = arith.index_castui %tid : i32 to index
    pto.store %tid2, %dst[%idx] : !pto.ptr<i32, ub>, index, i32
    return
  }
}
```

---

## 5. 放置约束

### 5.1 `pto.keep`

- 只能出现在 `pto.simt_entry` 函数内部。
- 必须显式列出要保存的 SSA 值。
- `slot` 必须是编译期常量。
- payload 不得包含 `!pto.vreg` / `!pto.mask`。
- 当前实现中必须紧邻 `func.return`。

### 5.2 `pto.resume`

- 只能出现在 `pto.simt_entry` 函数内部。
- 必须显式恢复 payload。
- `slot` 必须与对应 `pto.keep` 匹配。
- payload 不得包含 `!pto.vreg` / `!pto.mask`。
- 当前实现中必须是所在 block 的第一条操作。

### 5.3 成对约束

这些约束描述用户必须保持的跨 SIMT entry 语义关系。实现不需要对跨
entry 关系做 verifier 检查。

- 一个 `slot` 在被 `resume` 消费前不能再次被 `keep` 覆盖。
- `keep` 和 `resume` 的 payload 类型必须完全一致。
- 第一版只支持线性调用链，不跨分支/循环。

### 5.4 中间允许的 op

`keep` 和匹配的 `resume` 之间：

- 禁止插入无关的 `pto.simt_entry` 调用
- 禁止新的 `pto.keep` / `pto.resume`
- 允许普通标量算术、地址计算、常量、`arith.*`、以及非 `pto.simt_entry` 的纯 helper call

---

## 6. `store_vfsimt_info` 约束

两次相关 `simtvf` 的 launch 维度必须一致：

- `keep` 所在的 `simtvf`
- `resume` 所在的 `simtvf`

---

## 7. 为什么不用 `return` 语义

如果把这份状态做成普通 `return` 值，会误导成“我可以把它长期拿着，过会再用”。

这不符合这里的时序敏感语义。

所以本设计选择：

- 用 `slot` 表达跨函数/跨调用的逻辑连接
- 用显式 payload 表达“保存了哪些变量”
- 不把它做成普通可长期持有的 return value

---

## 8. Side Effect 建模

`pto.keep` / `pto.resume` 不能是 `Pure` op。

建议把它们建模成访问抽象资源：

- `pto.keep`：`Write<SIMT_PAYLOAD_SLOT>`
- `pto.resume`：`Read<SIMT_PAYLOAD_SLOT>`

这样可避免被 CSE / DCE / 重排误伤。

---

## 9. Verifier 规则

至少检查：

1. `pto.keep` / `pto.resume` 只能出现在 `pto.simt_entry` 内。
2. `keep` / `resume` 的 payload 中不得出现 `!pto.vreg` / `!pto.mask`。
3. `slot` 必须是编译期常量。
4. `slot` 是用户显式分配的槽位；32-bit 及以下值占一个槽，64-bit
   整数占两个槽且起始 slot 必须为偶数。
5. 同一 keep/resume 组内的槽位覆盖范围不得重叠。
6. `resume` 必须是所在 block 的第一条操作，`keep` 必须紧邻 `func.return`。

跨 SIMT entry 的 slot 配对、payload 类型匹配、覆盖关系和 launch 维度
一致性属于用户必须遵守的语义约束，不作为 verifier 的跨 entry 检查项。

---

## 10. Lowering 方案

### 10.1 LLVM 承载方式

VPTO 语义层保持 `pto.keep` / `pto.resume` 不变。LLVM 层不引入可长期持有的 `simt_state` 值，也不新增 intrinsic。`keep/resume` 在 lowering 时映射成空 `llvm.inlineasm` sideeffect call，通过固定 TPER output constraint 表达物理寄存器绑定。

lowering 把 `slot` 直接映射到固定 SIMT 物理寄存器。32-bit 及以下
payload 使用 32-bit TPER 名称：

- `slot = 0` -> `TPER4`
- `slot = 1` -> `TPER5`
- ...
- `slot = 122` -> `TPER126`

64-bit 整数值使用 64-bit 寄存器对名称 `TPERL<n>`。设
`base_reg = 4 + slot`，其 constraint 名称为
`TPERL<base_reg / 2>`，并对应 `TPER<base_reg>` 和
`TPER<base_reg + 1>`。例如：

- `slot = 0` -> `TPERL2` -> `TPER4/TPER5` -> `R4:R5`
- `slot = 2` -> `TPERL3` -> `TPER6/TPER7` -> `R6:R7`
- `slot = 8` -> `TPERL6` -> `TPER12/TPER13` -> `R12:R13`
- `slot = 120` -> `TPERL62` -> `TPER124/TPER125` -> `R124:R125`

因此 64-bit 起始 slot 必须为偶数。slot 是用户显式分配的，不会按
keep/resume 组内出现顺序重新 packed；consumer 只恢复部分 slot 时，
剩余 slot 的物理位置仍然稳定。

两个连续 i32 keep 的 lowering 形态为：

```llvm
%keep = call { i32, i32 } asm sideeffect "",
  "={TPER4},={TPER5},0,1"(i32 %a, i32 %b)
```

对应 resume 的 lowering 形态为：

```llvm
%resume = call { i32, i32 } asm sideeffect "",
  "={TPER4},={TPER5}"()
```

keep 的数字 input constraint 与前面的 fixed output 绑定，使 LLVM 负责将 payload 物化到目标 TPER。resume 直接把 fixed output 物化为新的 SSA 值。asm body 为空，不生成 keep/resume 专用 `MOV` 或 `IMAD.WIDE`。

64-bit 值必须使用 i64 fixed register-pair output。例如 `slot = 8` 的
keep/resume 分别生成：

```llvm
%keep = call i64 asm sideeffect "", "={TPERL6},0"(i64 %value)
%resume = call i64 asm sideeffect "", "={TPERL6}"()
```

`TPER12` 只表示 32-bit `R12`；`TPERL6` 才表示 64-bit
`R12:R13`。两种名称不能仅根据 LLVM operand type 自动互换。

约束：

- `keep` 和 `resume` 必须带 `sideeffect`，不能被当成纯函数删除。
- `resume` 必须产生与 `keep` payload 完全一致的结果类型。
- 不能把它做成普通 return token，也不能让它看起来像可无限期持有的 state 值。
- 不使用 generic `R` / `=R` output，也不使用 `~{R}` / `~{TPER}` clobber 表达 persistent 数据流。
- 32-bit 及以下 payload 必须能映射到 `TPER4~TPER126`，超出范围直接
  在 verifier 中拒绝。
- 64-bit 整数值要求偶数 slot，且必须能映射到 `TPERL2~TPERL62`；
  第二个 32-bit 槽不能超出范围。
- lowering 必须根据 packed payload 宽度选择 `TPER<n>` 或
  `TPERL<n>`，不能只依赖 LLVM operand type 推导寄存器宽度。

### 10.2 执行顺序

1. 在 authoring IR 中保留 `keep/resume`。
2. 在 VPTO verifier 中收紧 slot、payload、边界和插入点约束。
3. 在 `VPTOLLVMEmitter` 中 lower 成 inline asm sideeffect call。
4. 目标后端根据 fixed TPER constraint 完成物理寄存器绑定，空 asm body 不生成额外指令。

### 10.3 一个具体 case

以“`simt_stage0` 保存，`simt_stage1` 恢复”为例，目标形态如下：

```mlir
module attributes {pto.target_arch = "a5"} {
  func.func @kernel(%dst: !pto.ptr<i32, ub>, %src: !pto.ptr<i32, gm>) {
    %c0_i64 = arith.constant 0 : i64
    %c32_i64 = arith.constant 32 : i64
    %c128_i64 = arith.constant 128 : i64
    %dim_z = arith.constant 1 : i32
    %dim_y = arith.constant 32 : i32
    %dim_x = arith.constant 32 : i32
    %ub = pto.castptr %c0_i64 : i64 -> !pto.ptr<i32, ub>

    pto.store_vfsimt_info %dim_z, %dim_y, %dim_x : i32, i32, i32
    func.call @simt_stage0(%ub) : (!pto.ptr<i32, ub>) -> ()

    func.call @simt_stage1(%ub) : (!pto.ptr<i32, ub>) -> ()

    pto.dma_store %ub, %dst, %c128_i64
      nburst(%c32_i64, %c128_i64, %c128_i64)
      : !pto.ptr<i32, ub>, !pto.ptr<i32, gm>, i64, i64, i64, i64
    return
  }

  func.func @simt_stage0(%dst: !pto.ptr<i32, ub>) attributes {pto.simt_entry} {
    %tid = pto.get_tid_x : i32
    // 这里代表 stage0 产生的状态被 keep 到 slot 0。
    pto.keep %tid {slot = 0} : i32
    return
  }

  func.func @simt_stage1(%dst: !pto.ptr<i32, ub>) attributes {pto.simt_entry} {
    // 这里代表 stage1 从 slot 0 恢复上一个 stage 的状态。
    %tid = pto.resume {slot = 0} : i32
    return
  }
}
```

对应到 LLVM lowering 后，`keep/resume` 仍然留在各自函数内部，只是变成内联汇编承载的 sideeffect 操作：

- `simt_stage0` 内的 `pto.keep %tid {slot = 0} : i32`
  - `%keep = call i32 asm sideeffect "", "={TPER4},0"(i32 %tid)`
- `simt_stage1` 内的 `pto.resume {slot = 0} : i32`
  - `%tid2 = call i32 asm sideeffect "", "={TPER4}"()`

这里固定的不只是数据流关系，也包括最终 asm 形态：

- `keep/resume` 都必须留在对应 `simt_entry` 内。
- `slot = 0` 只在这两个函数之间建立关联。
- `slot` 在 lowering 时必须按 payload 宽度映射到对应的
  `TPER4~TPER126` 或 `TPERL2~TPERL62`。

---

## 11. 对现有 pass 的影响

- `PTOValidateVPTOIR` 需要新增 keep/resume 校验。
- canonicalize / CSE 需要因为 side effect 而保留它们。
- `VPTOLLVMEmitter` 需要先消费再擦除。

---

## 12. 结论

这版方案比 `return simt_state` 更准确：

- 不误导成“可长期持有的状态值”
- 仍然能显式表达“保存了哪些变量”
- 还能通过 slot 和 verifier 建立严格逻辑数据流
