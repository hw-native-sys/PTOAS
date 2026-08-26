# VPTO tied-operand copy 物化设计

## 1. 背景

VPTO 使用 SSA 表达 vector value，但部分 A5 vector 指令在硬件层属于 two-address instruction。以 `pto.vmula` 和 `pto.vmadd` 为例，其 accumulator 输入与结果输出具有 tied-operand constraint，必须分配到同一个物理寄存器；指令执行后，该寄存器中的旧值被结果覆盖。

当多个具有 tied-operand constraint 的指令共享同一个 tied operand 时，SSA 仍允许它们独立读取同一个 value：

```mlir
%r0 = pto.vmula %acc, %x0, %y, %mask : ...
%r1 = pto.vmula %acc, %x1, %y, %mask : ...
%r2 = pto.vmula %acc, %x2, %y, %mask : ...
```

LLVM lowering 之后，Bisheng 必须为其中一部分 use 插入真实 VMOV，保证每条破坏性指令得到可独立更新的寄存器副本。当前 PTOAS scheduler 在 VPTO IR 上工作，因而看不到这些后端新增指令，造成以下偏差：

- 调度 DAG 缺少实际存在的 VMOV node 和依赖；
- target resource 和 latency 模型低估实际 vector pipe 工作量；
- register-pressure 模型看不到 VMOV 产生的新物理寄存器；
- PTOAS 与 Bisheng 的 scheduler 实际面对不同的指令序列。

Issue #1327 的 precise FP32 division 修正链中，同一个 `%55` 分别作为三条 `pto.vmula` 的 accumulator，是该问题的直接实例。

## 2. 目标与非目标

### 2.1 目标

本设计引入一个独立于 scheduler 实现的 VPTO pass，在 emission-ready VPTO 上：

1. 显式表达完整物理 vector-register copy；
2. 识别具有 tied-operand constraint 的 operation；
3. 在共享 tied operand 需要独立寄存器时插入 `pto.vmov`；
4. 保证通用 canonicalize、CSE 和 DCE 不会删除或合并已物化的 VMOV；
5. 为后续 scheduler 提供完整、稳定的 operation 视野。

设计还必须允许 `vpto-mov` 先于 `vpto-sched-2` 合入。物化 pass 的正确性和测试不依赖 scheduler 的 DAG、strategy、region 或 pressure tracker。

### 2.2 非目标

本阶段不负责：

- 在 `vpto-mov` 中实现或修改 scheduler 的选点策略；
- 改变现有 canonicalize、CSE、reduction combine 等 pass 的相对顺序；
- 依赖 Bisheng 后端继续为 tied-operand constraint 隐式插入 copy；
- 为所有未来可能具有 tied-operand constraint 的指令一次性建模；
- 跨 block 或跨控制流路径求最少 VMOV；
- 将 `pto.vmov` 暴露为 PTODSL/VMI 的通用用户 API。

首期覆盖已确认具有该约束的 `pto.vmula`、`pto.vmadd`、`pto.vaxpy`、`pto.chistv2`、`pto.dhistv2` 和 `pto.vusqz`。新增指令必须先补齐 tied-operand 接口和目标语义，不能仅在 pass 中按 operation name 特判。

### 2.3 指令审计

不能仅凭 operation 是否有名为 `acc` 的 operand 判断 tied-operand constraint。本设计以 A5 emitter 选择的 intrinsic form 和参数位置为主要依据：使用 merging-form 的指令由第一个向量参数提供 destination/merge carrier，结果会复用并覆盖该参数对应的物理寄存器。历史 `pto.vmov` emitter 会把同一个 input 同时传入 merging operand 和 source operand，也印证了这一调用约定。

当前 VPTO 指令面的审计结果如下：

| operation | tied operand | tied result | 判定依据 |
| --- | --- | --- | --- |
| `pto.vmula` | `acc`（operand 0） | `result`（result 0） | multiply-accumulate 的 accumulator 是 read-modify-write destination，lowering 使用 merging-form |
| `pto.vmadd` | `acc`（operand 0） | `result`（result 0） | `%acc` 是 destination-as-source multiplicand，lowering 使用 merging-form |
| `pto.vaxpy` | `src1`（operand 1） | `result`（result 0） | 硬件契约为 `dst = alpha * src + dst`，emitter 将 `%src1` 调整到第一个 destination/merge 参数 |
| `pto.chistv2` | `acc`（operand 0） | `result`（result 0） | A5 cumulative histogram 语义原地更新 destination，lowering 使用 merging-form |
| `pto.dhistv2` | `acc`（operand 0） | `result`（result 0） | A5 frequency histogram 语义原地更新 destination，lowering 使用 merging-form |
| `pto.vusqz` | `src`（operand 0） | `result`（result 0） | `%src` 在 VPTO 语义中是 carrier，lowering 将其作为 merging destination，指令执行后该物理寄存器被结果覆盖 |

### 2.4 `vpto-mov` PR 的边界

`vpto-mov` 提供：

- `pto.vmov` IR 定义、verifier、文档和两套 VPTO LLVM emitter lowering；
- tied-operand operation interface；
- 与 scheduler 无关的 physical-view root 查询工具；
- copy 物化 pass 及独立 lit 测试；
- 手写 `pto.vmov` 的 parse/verify/emission 测试。

该 PR 注册 pass，使其可由 `pto-test-opt` 独立运行，但不在默认 `ptoas` pipeline 中自动启用。这样它可以在 active scheduler 合入前独立评审，同时不改变当前默认产物。

## 3. `pto.vmov` IR 契约

### 3.1 语法

新增无 mask、单输入、单结果的物理复制 operation：

```mlir
%copy = pto.vmov %input
    : !pto.vreg<NxT> -> !pto.vreg<NxT>
```

verifier 要求：

- input 和 result 都是 `!pto.vreg<...>`；
- input 与 result 类型完全相同；
- 类型是目标支持的单个物理 vector register 表示。

首期不支持 mask operand、跨类型 copy、predicate-register copy 或多寄存器 aggregate。跨类型的同一物理值 view 继续由 `pto.vbitcast` 表达。

### 3.2 语义

`pto.vmov` 表示一次必须保留的完整物理寄存器复制：

- 读取 `%input` 对应的全部物理 lane；
- 产生内容相同、但可由后续 destructive consumer 独立覆盖的物理寄存器；
- 在硬件上产生一条真实 VMOV；
- 不读写普通 memory；
- 在 A5 上占用 vector pipeline。

该 operation 不带 mask。即使后续 tied operation 是 masked form，也必须先复制完整 tied operand，保证该指令执行前获得内容完整且可独立覆盖的物理寄存器。历史上曾存在带 mask 的 `pto.vmov`，其 emitter 使用 merging form；该形式混合了“条件更新 destination”与“创建独立物理副本”两种语义，不能直接恢复为本 pass 的物化 primitive。

两套 emitter 应分别选择各自 CANN 版本的 full-register VMOV intrinsic。预期命名族为：

| emitter | 预期 intrinsic family |
| --- | --- |
| legacy / CANN 9.0.0-beta.1 | `llvm.hivm.vmov.v<lanes><type>` |
| CANN 9.0.0 | `llvm.hivm.vmov.x.v<lanes><type>` |

实现时必须用对应工具链头文件或最小编译用例确认最终 spelling、prototype 和支持类型；不能复用历史 `.m` merging form 的调用约定。

### 3.3 不可折叠性

以下两条 VMOV 即使 source 相同，也代表两个不同的物理副本：

```mlir
%copy0 = pto.vmov %input : !pto.vreg<64xf32> -> !pto.vreg<64xf32>
%copy1 = pto.vmov %input : !pto.vreg<64xf32> -> !pto.vreg<64xf32>
```

因此 `pto.vmov` 不得带 `Pure`，不得提供 fold/canonicalization 将其替换为 input，也不得声明为对通用变换完全无 effect。其 IR effect 契约必须使 `wouldOpBeTriviallyDead` 为 false，并使 CSE 不具备合并两个 VMOV 的依据。

与此同时，`VPTOSchedulingOpInterface` 必须明确报告：

- scheduling class 为 `Schedulable`；
- ordinary `memoryBehavior` 为 `None`；
- execution pipe 为 `PIPE_V`。

当前实现可沿用 `mem_bar`、`sprclr` 和 ctrl-register operation 的处理方式：通用 IR 变换保守保留 operation，而 VPTO scheduling semantics 单独声明其不访问普通 memory。不得为了保护 VMOV 而移动或删除已有 CSE pass。

## 4. Tied-operand 接口

新增 operation interface，例如 `VPTOTiedOperandOpInterface`，用于描述 operation-local 的物理约束。首期每个实现者只支持一组 tied operand/result pair，接口至少提供：

- tied operand index；
- tied result index。

首期登记关系为：

| operation | tied operand | tied result |
| --- | --- | --- |
| `pto.vmula` | `acc`（operand 0） | `result`（result 0） |
| `pto.vmadd` | `acc`（operand 0） | `result`（result 0） |
| `pto.vaxpy` | `src1`（operand 1） | `result`（result 0） |
| `pto.chistv2` | `acc`（operand 0） | `result`（result 0） |
| `pto.dhistv2` | `acc`（operand 0） | `result`（result 0） |
| `pto.vusqz` | `src`（operand 0） | `result`（result 0） |

接口只陈述“结果必须复用哪个输入寄存器”的事实，不决定：

- 是否需要插 VMOV；
- 哪一个 destructive tied use 成为 owner；
- scheduler 应生成哪种 edge；
- operation 在何处调度。

如果未来一个 operation 有多组 tied operand/result pair，应扩展接口返回 pair 列表，而不是继续增加按 operation name 的分支。

## 5. Physical view 等价关系

本节只处理 copy 物化分析所需的物理寄存器同一性，不负责 `pto.vbitcast`/`pto.pbitcast` 自身的调度成本、latency 或 pressure 建模。当 tied operand 经过零开销 view 时，多个不同 SSA value 仍可能引用同一个物理寄存器；pass 必须识别这一关系，避免把它们错误地分别选为 owner。

`vpto-mov` 新增位于 IR/通用 transform support 层的查询工具，例如：

```text
getPhysicalRegisterViewRoot(value)
```

它只沿已确认的零开销 view 向上查找，并使用 visited set 防止异常 IR 形成循环。首期支持：

- `pto.vbitcast`；
- `pto.pbitcast`（供统一工具使用，尽管首期不物化 predicate copy）。

该工具不把以下关系混为 view：

- `pto.vmov` input/result：它们是两个不同物理寄存器；
- tied operand/result：二者在指令执行点复用寄存器，但表示不同 SSA version；
- 普通 elementwise producer/result。

该工具本身不引用 scheduler 类型。`vpto-sched-2` 是否复用该工具属于 scheduler PR 的实现选择，不作为 `vpto-mov` 的合入依赖。

## 6. Copy 物化 pass

### 6.1 名称和输入契约

建议 pass 名称：

```text
pto-vpto-materialize-tied-operand-copies
```

C++ factory 建议为：

```text
createVPTOMaterializeTiedOperandCopiesPass()
```

pass 以 `func::FuncOp` 为根运行，遍历 emission-ready `pto.vecscope` 和 `pto.strict_vecscope` 内的 block。它没有 scheduler mode，不读取 scheduler analysis，也不要求 scheduler 已经运行。

首期目标语义限定为 A5。独立调用时，缺失或不匹配的 target arch 应给出明确诊断，而不是静默生成目标不支持的 VMOV。

### 6.2 基本算法

对每个 block 执行以下步骤：

1. 收集实现 tied-operand 接口的 operation；
2. 对每个 tied operand 求 physical-view root；
3. 收集该 root 及其零开销 view 的所有 material use；
4. 判断是否存在可以直接更新原寄存器的 owner；
5. 为其余 destructive tied use 在对应 operation 前插入 `pto.vmov`；
6. 仅将该 use 的 tied operand 替换为 VMOV result。

material use 指真实读取或破坏物理寄存器的 operation。`pto.vbitcast`/`pto.pbitcast` 本身只传播 view，不作为 material use；它们的下游 use 继续计入 root。

### 6.3 Owner 选择

本文将 tied operand 对应物理寄存器会被结果覆盖的使用点称为 destructive tied use；将直接复用并覆盖原寄存器的 destructive tied use 称为 original-register owner，后文简称 owner。一个 destructive tied use 只有同时满足以下条件才能成为 owner：

- root 的所有 use 和 view use 都可在同一 block 内完整分析；
- root 不 live-out 到其他 block、region 或函数边界；
- 按 block 原始顺序，该 destructive tied use 是 root 的最后一个 material use。

满足条件时，最后一个 destructive tied use 直接使用原 operand，其他 destructive tied use 各得到一个 VMOV。若共有 `N` 个 destructive tied use 且不存在其他更晚 use，物化 `N - 1` 条 VMOV：

```mlir
%copy0 = pto.vmov %acc : !pto.vreg<64xf32> -> !pto.vreg<64xf32>
%r0 = pto.vmula %copy0, %x0, %y, %mask : ...

%copy1 = pto.vmov %acc : !pto.vreg<64xf32> -> !pto.vreg<64xf32>
%r1 = pto.vmula %copy1, %x1, %y, %mask : ...

// 最后一个 material use，可以覆盖原寄存器。
%r2 = pto.vmula %acc, %x2, %y, %mask : ...
```

若最后一个 material use 是普通 read，或存在 cross-block/live-out use，则没有安全 owner，所有 destructive tied use 都必须复制：

```mlir
%copy0 = pto.vmov %acc : !pto.vreg<64xf32> -> !pto.vreg<64xf32>
%r0 = pto.vmula %copy0, %x0, %y, %mask : ...
%copy1 = pto.vmov %acc : !pto.vreg<64xf32> -> !pto.vreg<64xf32>
%r1 = pto.vmula %copy1, %x1, %y, %mask : ...
%later = pto.vabs %acc, %mask : ...
```

这种保守策略可能比全局最优方案多生成 VMOV，但不依赖控制流调度分析，并保证 `vpto-mov` 可以独立正确工作。跨 block 的最小化不在首期范围内。

### 6.4 插入位置

每条 VMOV 紧邻对应 destructive consumer 之前插入。这样即使没有 scheduler：

- 原 block 顺序仍满足所有 read-before-destructive-write 关系；
- VMOV result 到 destructive consumer 存在直接 SSA data dependency；
- pass 输出可直接进入 emitter。

后续 scheduler 可以将 VMOV 提前以隐藏 latency，但必须遵守第 8 节定义的新增物理依赖。

### 6.5 幂等性

pass 必须幂等：

- pass 自己插入的单 use VMOV result 再次运行时不产生额外 copy；
- 用户手写 VMOV 的 result 作为新的 physical root 分析；
- 若一个手写 VMOV result 又被多个 destructive tied use 共享，仍按普通 root 规则物化；
- 第二次运行后的 IR 与第一次相同。

不得通过临时 attribute 标记“已处理”；结构化 IR 本身应足以判断。

## 7. Pipeline 位置

本设计不调整任何已有 pass 的相对顺序。scheduler 集成后的目标位置为：

```text
... soft-lib expansion / inline
  -> canonicalize
  -> CSE
  -> VPTOMaterializeTiedOperandCopiesPass
  -> VPTOSchedulerPass
  -> VPTOCombineReductionsPass
  -> CSE
  -> PTOValidateVPTOEmissionIR
```

选择该位置的原因是：

- 所有可能新增首期 tied operation 的 lowering 已完成；
- 物化之前仍可正常优化 SSA；
- scheduler 能看到完整 VMOV；
- 物化之后的 CSE 不会合并 VMOV，因为不可折叠性由 operation 自身保证。

`vpto-mov` PR 只注册独立 pass，不进行上述 driver 接线。`vpto-sched-2` PR 完成接线。pass 本身不区分 `off`、`analyze` 或 `on`；具体模式是否组合该 pass 是 driver 的 pipeline policy，不进入物化算法。

## 8. Scheduler 后续适配契约

本节定义 `vpto-sched-2` 必须消费的事实，但不属于 `vpto-mov` 的实现范围。

### 8.1 DAG 依赖

VMOV 到其 destructive consumer 已由 SSA 建立 data edge。除此之外，当某个 destructive tied use 被选为 original-register owner 时，scheduler 必须保证 owner 在该 root 的所有其他 material read 之后执行，包括：

- 从原 root 创建其他副本的 VMOV；
- 读取原 root 的普通 vector operation；
- 通过 `pto.vbitcast` view 读取同一物理 root 的 operation。

这些关系应表现为 `Anti/Must` edge，reason 应稳定标识 original-register owner，便于 trace 和测试。不能只依赖原始文本顺序，否则 scheduler 可能把 owner 提前并覆盖仍需复制或读取的 tied operand。

### 8.2 Resource 和 latency

`pto.vmov` 是真实 vector micro-op，不能按零 micro-op 或零 latency 建模。首版可以使用 A5 vector-move 参数；最终 write latency 和 resource occupancy 必须用 CA trace 或工具链资料验证。现有文档中的 9-cycle `RV_VLD` proxy 只能作为待验证初值，不能直接当作 VMOV 实测结论。

### 8.3 Register pressure

- `pto.vmov` result 是新的 vector-register pressure unit；
- VMOV input 的 live range 至少延续到该 VMOV；
- tied operand/result 应按寄存器复用约束处理，不能同时重复计为两个独立寄存器；
- scheduler 提前 VMOV 时必须计入被拉长的 copy-result live range。

### 8.4 Schedule verification

调度结果校验至少检查：

- 每条 VMOV 在其 destructive consumer 之前；
- owner 在同 root 的其他 material read 之后；
- view chain 不绕过 physical-root 依赖；
- schedule apply 后 tied operand/result 约束仍成立。

## 9. 错误处理和保守边界

遇到以下情况时，pass 不应猜测最优 owner：

- tied operand 的 view/use 穿过 block 或嵌套 region；
- use 属于无法识别的 region control flow；
- operand 或 result 不是支持的单物理 `!pto.vreg`；
- tied-operand interface 返回越界或不匹配的 operand/result；
- target 不支持所需 full-register VMOV。

对于可以通过“给所有 destructive tied use 插 copy”保证正确的 cross-block/live-out 情形，采用保守物化。对于 IR/interface/target 契约本身非法的情形，pass 失败并给出包含 operation name 和原因的诊断，不得静默跳过。

## 10. 测试计划

### 10.1 ODS 和 verifier

- `pto.vmov` parse/print round trip；
- input/result 类型相同的合法用例；
- 非 vreg、类型不一致和非单物理寄存器的非法用例；
- 六个首期 operation 的 tied-operand interface 查询，包括 `vaxpy` 的 operand 1 和其余 operation 的 operand 0；
- `vaddcs`/`vsubcs` 不实现 tied-operand interface；
- 当前不存在同名 VPTO operation 的 FMA/MSUB mnemonic 不进入 operation registry 或 pass name-based fallback；未来新增 operation 时验证并登记其 destination/result tied pair；

### 10.2 物化 pass

- 三个共享 `vmula` 生成两条 VMOV，最后一个为 owner；
- 三个共享 `vmadd` 的同类用例；
- `vmula` 与 `vmadd` 混合共享 acc；
- `vaxpy` 共享 `src1` 的用例；
- `chistv2`/`dhistv2` 共享 acc 的用例；
- `vaddcs`/`vsubcs` 构造不会被 CSE 合并的两级 carry/borrow 链，共享 vector input 时不插入 VMOV；
- `vusqz` 共享 carrier 的用例；
- 末尾存在普通 read 时所有 destructive tied use 都复制；
- live-out/cross-block 时所有 destructive tied use 都复制；
- 经一层和多层 `vbitcast` 后仍识别同一 physical root；
- 已有单 use VMOV 不重复物化；
- 手写共享 VMOV result 仍会继续物化；
- pass 连续运行两次保持相同 IR；
- CSE 在 pass 后运行时不合并两个同源 VMOV；
- DCE 不删除无 SSA user 但必须保留的 VMOV。

### 10.3 Emitter

- legacy emitter 生成 full-register VMOV intrinsic；
- CANN 9.0.0 emitter 生成对应 full-register VMOV intrinsic；
- 支持类型矩阵逐项编译；
- 输出交给 Bisheng 后确实保留 VMOV，不重新退化为由后端隐式插入 copy；
- masked tied operation 前的完整复制保持原始 tied operand 内容和结果语义；
- `chistv2` 和 `dhistv2` 分别通过 A5 Bisheng 编译，并在 CA/NPU 上完成真实执行和严格 compare；

### 10.4 Scheduler PR 后续测试

- DAG 包含 VMOV node、SSA edge 和 owner anti-edge；
- `vbitcast` 为零开销 view，VMOV 为真实 pressure unit；
- scheduler apply 后 emission 顺序满足 tied-operand constraint；
- issue #1327 最小 IR 中，PTOAS 与 Bisheng 看到相同 VMOV 数量；
- A5 CA/NPU 上功能 compare 通过，并记录性能相对默认路径和 Bisheng MI scheduler 的对比。

## 11. 验收标准

`vpto-mov` 可以先合入的必要条件：

- 独立 pass 不引用 `VPTOScheduler` 目录下的类型或实现；
- 三个共享 destructive tied use 的基本用例稳定产生 `N - 1` 条 VMOV；
- cross-block/live-out 用例采用正确的保守策略；
- VMOV 经现有 canonicalize/CSE/DCE 后仍保持数量和相互独立性；
- 两套 emitter 均通过 intrinsic 编译验证；
- pass 可由 `pto-test-opt` 独立运行，默认 `ptoas` pipeline 行为不变。

完整功能在 `vpto-sched-2` 合入时的附加标准：

- scheduler 输入包含全部显式 VMOV；
- DAG 和 schedule verifier 防止 owner 覆盖尚未读取的 tied operand；
- pressure tracker 区分 VMOV copy 与 bitcast view；
- issue #1327 的功能结果不变，并获得可解释、可复现的性能对比。
