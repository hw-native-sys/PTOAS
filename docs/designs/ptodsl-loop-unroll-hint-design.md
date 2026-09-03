# PTODSL / PTOAS Loop Unroll Hint 设计文档

> **修订记录（v5）**：新增 `pto-promote-persistent-fragment-loops`——
> persistent fragment materialization 要求访问 persistent buffer 的循环
> 被完全展开，此前依赖调用方手写 `full`；该 pass 沿
> `llvm.alloca {pto.persistent}` 的 pointer use graph（GEP 链，
> 不跟随 load 的数据结果）发现相关循环并提升为
> `full`（覆盖 `enable`），附 `pto.persistent_unroll` marker 让 Pass A
> 的丢弃兜底升级为硬错误。详见
> `docs/designs/ptoas_persistent_simt_fragment_plan.md` 的「自动提升」一
> 节。
>
> **修订记录（v4）**：评审指出，原实现只局部降级带注解 loop 会把新生成的
> cf blocks 留在外层 single-block region（无 hint 的外层 `scf.for`、`scf.if`
> 等）内，触发 SingleBlock verifier 失败。现改为**单 pass 完成整个函数的
> SCF→CF 转换**——上游 conversion patterns + 更高 benefit 的带注解 loop
> pattern,该 pass 因此**替代**两条 emitter pipeline 中的
> `createConvertSCFToCFPass`;pass 名相应改为
> `pto-convert-scf-to-cf-with-loop-hints`（它已不只处理 hint,而是真正承担
> 控制流转换）。另修复 Pass A 中 `enable` 判断排在 native-unroll guard 之后
> 导致空 body / 非 index loop 的 hint 被静默丢弃的问题。
>
> **修订记录（v3）**：v2 移除整个阶段一（Pass B 与 `enable`/`disable`）
> 后，#1242 Req2 的验收标准（"public `pto.for_` 可以携带 unroll-enable
> hint；hint 能到达 LLVM/BiSheng 而不是前端强制 full unroll"）无法闭
> 环。v3 **恢复 `unroll="enable"` 单一 metadata 通道**：精简版
> `pto-convert-scf-to-cf-with-loop-hints` 只负责 `enable` → `!llvm.loop.unroll.enable`，
> `disable` 不再恢复，`full`/`factor` 的丢弃语义不变。
>
> ~~修订记录（v2）~~：初版设计为双阶段——Pass A 原生展开 + Pass B 把残
> 留 hint 透传为 LLVM loop metadata（含 `enable`/`disable`）。评审后决
> 定移除阶段一：`full`/`unroll_factor` 的原生展开已覆盖 unroll 接口能
> 力，`enable`/`disable` 随之删除。v3 重新评估后部分回退了该决定（仅
> 恢复 `enable`）。
>
> 关联 Issue：
> - [Issue #1242](https://github.com/hw-native-sys/PTOAS/issues/1242) Requirement 2 —— `pto.for_` loop-unroll hint（`unroll="enable"` 经 `pto-convert-scf-to-cf-with-loop-hints` 转为 `llvm.loop.unroll.enable`）
> - [Issue #1000](https://github.com/hw-native-sys/PTOAS/issues/1000) —— 支持 Loop Unroll Hint（含 `pto.range`、factor unroll、两阶段计划）
> - [PR #838](https://github.com/hw-native-sys/PTOAS/pull/838) —— `PTOUnrollSIMTForPass`（规避 BiSheng AICore 后端 bug 的临时方案，本次一并重构）

---

## 1. 背景与动机

### 1.1 需求来源

**Issue #1242 Requirement 2**：SIMTVF codegen 需要表达无 factor 的 `#pragma unroll` 语义——保留 device-side loop，将展开意图交给 LLVM/BiSheng 的 cost model 决定 full 或 partial unroll。当前 PTODSL 只有两个极端：

- `pto.static_range`：trace 阶段强制完全展开，增加 trace/编译时间、IR 体积与寄存器压力；
- `pto.for_`：保留 device loop，但无法携带任何 frontend unroll hint。

**Issue #1000**：TileLang-PTO 后端需要将 `T.unroll(..., explicit=False)` lower 成对等语义（编译器侧 unroll，而非 DSL 前端强制展开）。该 issue 给出了两阶段计划：阶段一由 Bisheng/CCE 执行展开（hint 透传），阶段二由 PTOAS 原生展开。

**PR #838（历史背景）**：`PTOUnrollSIMTForPass` 是为了规避 BiSheng AICore 后端 bug 的临时方案——SIMTVF kernel 中 `scf.for` + `scf.if` 常量分支经 SCF→CF→LLVM lowering 后，AICore 后端未正确处理 `SimtEntry` calling convention，给 `END` 生成了带谓词的 `END @!P0`。规避手段是将 SIMT 上下文内标注 `{pto.unroll = "full"}` 的常量循环强制完全展开，由下游 SCCP + canonicalize 消除常量分支，生成无分支直线代码。Issue #1000 的评论明确指出该 pass 是临时方案，本次应一并重构为通用 unroll 能力。

### 1.2 目标

1. `pto.for_` / 新增 `pto.range` 支持 `unroll="full"` 与 `unroll_factor=N` hint：PTOAS 在 LLVM lowering 前原生展开（#1242 的 loop-unroll 接口诉求、#1000 阶段二）。
2. 将 `PTOUnrollSIMTFor` 从"SIMT-only、full-only 的临时 pass"重构为通用的 attr 驱动 unroll pass，同时保持 #838 的 bug 规避语义不回归。
3. 未指定 hint 的循环行为完全不变。

（v1 曾包含 `unroll="enable"`/`"disable"` 的 metadata 透传目标；评审后删除，见文首修订记录与 §5.3。）

### 1.3 非目标（Non-goals）

- 不修改 LLVM/BiSheng 的 unroll cost model；
- 不改变 `pto.static_range` 的 trace-time full-unroll 语义；
- 不支持任意动态循环的 full unroll（动态 bound 的 `"full"` 无法原生展开，丢 hint + remark，loop 保留）；
- 不包含 TileLang codegen 侧的修改。

---

## 2. 现状分析

### 2.1 现有 unroll 能力

`PTOUnrollSIMTForPass`（原 `lib/PTO/Transforms/PTOUnrollSIMTForPass.cpp`，本设计中重构为 `PTOUnrollLoopsPass.cpp`）：

- 只处理显式标注 `{pto.unroll = "full"}` 的 `scf.for`（`PTOUnrollSIMTForPass.cpp:58-66`）；
- 只在 SIMT 上下文内生效（`pto.simt_entry` 函数或 inline `pto.section.simt` 区域）；
- 要求静态 lb/ub/step、正 step，通过 `loopUnrollByFactor(tripCount)` 全展开；
- 在 `prepareVPTOForEmission` 中、SCCP/canonicalize/CSE 之前运行（`tools/ptoas/ptoas.cpp:3074`），展开后常量分支被下游折叠。

PTODSL 目前**没有任何 public API** 设置 `pto.unroll` attr——该 attr 只出现在手写 `.pto` 测试中（`test/test_unroll_annotation.mlir`、`test/lit/vpto/unroll_inline_simt_sections.pto`）。

### 2.2 关键发现：vendored MLIR 已具备完整的 loop-annotation 下传链路

无需对 LLVM/BiSheng 对接层做任何改动：

1. **SCF→CF**（历史记录，v1 依据）：vendored LLVM 19（feature-vpto）的 `SCFToControlFlow` **不会**把 `scf.for` 上的 `llvm.loop_annotation` 拷贝到 latch `cf.br`（该能力只在更新的上游版本存在，`llvm-workspace` 的 LLVM 21 已具备）。v1 因此让 Pass B 自行降级带注解 loop；v2 移除阶段一后，该链路不再被使用；
2. **CF→LLVM**：`BranchOpLowering` / `CondBranchOpLowering` 将全部 attrs 保留到 `llvm.br` / `llvm.cond_br`
   （`llvm-workspace/llvm-project/mlir/lib/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.cpp:130-170`）；
3. **MLIR→LLVM IR**：`translateModuleToLLVMIR` 把 latch 上的 `#llvm.loop_annotation` 翻译为 `!llvm.loop` metadata
   （`llvm-workspace/llvm-project/mlir/lib/Target/LLVMIR/LoopAnnotationTranslation.cpp:124-138`）：

   | MLIR attr | LLVM IR metadata |
   |---|---|
   | `#llvm.loop_annotation<unroll = <disable = false>>` | `!{!"llvm.loop.unroll.enable"}` |
   | `#llvm.loop_annotation<unroll = <disable = true>>` | `!{!"llvm.loop.unroll.disable"}` |
   | `#llvm.loop_annotation<unroll = <full = true>>` | `!{!"llvm.loop.unroll.full"}` |
   | `#llvm.loop_annotation<unroll = <count = N>>` | `!{!"llvm.loop.unroll.count", i32 N}` |

   其中 `!llvm.loop.unroll.enable` 正是无 factor `#pragma unroll` 的等价物（#1242 req2 的目标语义）。

4. **两条 VPTO emission pipeline 结构一致**：均为 `createConvertSCFToCFPass()` → `createConvertControlFlowToLLVMPass()` → `translateModuleToLLVMIR`
   （`lib/PTO/Transforms/VPTOLLVMEmitter.cpp:14195-14200`、`lib/PTO/Transforms/VPTOCANN900LLVMEmitter.cpp:11875` 附近）。
   `prepareVPTOForEmission`（`tools/ptoas/ptoas.cpp:3252`）先于所有 emission 路径执行。

### 2.3 关键发现：上游 unroll 工具能力足够

`loopUnrollByFactor`（`llvm-workspace/llvm-project/mlir/lib/Dialect/SCF/Utils/Utils.cpp:364`）：

- 支持**动态 bounds**：自动生成 trip count 计算与 epilogue remainder loop；
- **live-out carry values 由 epilogue 正确穿线**：main loop results 作为 epilogue 的 init args，uses 自动替换（现有 pass 注释中"live-out 不能展开"的限制在这条路径上不存在）；
- `loopUnrollFull` 是 `loopUnrollByFactor(tripCount)` 的封装；
- 限制：要求正 step，factor 必须为正；动态 bound 路径按 `ceilDivPositive`（`arith.divui`）计算 trip count，运行时 `upper < lower` 会把负差值按无符号解释成极大值——上游注释明确留了 `TODO: Add dynamic asserts for negative lb/ub/step`。Pass A 因此在调用前自行兜底：静态 `ub <= lb` 丢 hint，动态上界插入 `arith.maxsi(ub, lb)` 钳位（`ub >= lb` 时是恒等变换，`ub < lb` 时主循环与 epilogue 同样零次迭代）。

因此 #1000 阶段二中最重的工作（remainder loop 生成、loop-carried SSA 穿线）已由上游 util 完成，本方案只需实现 attr 驱动的调度、边界兜底与丢弃逻辑。

### 2.4 attr 存活风险排查

`prepareVPTOForEmission` 与 emission 之间唯一重建 `scf.for` 的 pass 是 `PTONarrowVPTOLoopCounters`，其实现保留了全部 attrs（`lib/PTO/Transforms/PTONarrowVPTOLoopCounters.cpp:115-117`，`newFor->setAttrs(forOp->getAttrs())`）。discardable attr 在 canonicalize/SCCP/CSE/LICM 中默认保留。方案中仍以 lit 测试锁定该行为。

---

## 3. 总体设计

### 3.1 语义矩阵

一套 attr 编码（`scf.for` 的 discardable attrs），一种消费方式（Pass A 原生展开）：

| 前端写法 | `scf.for` attr | 语义 | Native 处理（Pass A） | 无法原生展开时 |
|---|---|---|---|---|
| （无 hint） | 无 | 现状不变 | 不处理 | — |
| `unroll="enable"` | `pto.unroll = "enable"` | 保留 loop，编译器 cost model 决定 full/partial | 不处理（在所有 native-unroll guard 之前跳过，留给 Pass B） | —（必然转成 metadata） |
| `unroll="full"` | `pto.unroll = "full"` | PTOAS 强制全展开 | `loopUnrollByFactor(tripCount)`，loop 消失 | 动态 trip：丢 hint + remark，保留 loop |
| `unroll_factor=N` | `pto.unroll_factor = N`（i32） | PTOAS 按 N 展开 | `loopUnrollByFactor(N)`，生成 main + epilogue | 动态 step / 超上限 / N=1：丢 hint + remark，保留 loop |

约束：

约束：

- `unroll` 与 `unroll_factor` 互斥；
- `unroll_factor` 必须是 ≥ 1 的 Python 编译期整数常量；
- hint 不改变 `.carry(...)` 的 loop-carried value 语义；
- 未指定 hint 的 `pto.for_` / `range` 行为与 IR 完全不变。

边界情形（实现决策）：

- `unroll_factor=1` 无 native 意义，hint 丢弃 + remark；
- 空 body 的 loop（`loopUnrollByFactor` 对其为 no-op）不 native 展开，hint 丢弃 + remark；
- 只支持 index 类型的 loop：`scf.for` 也允许 signless 整数（i16/i32）边界，但 `loopUnrollByFactor` 一律用 `arith::ConstantIndexOp` 构造 step/offset 运算，展开 i16 loop 会产出 `arith.muli(i16, index)` 以及 step 类型与边界不符的 `scf.for`，直接 verifier 失败。非 index loop 丢 hint + remark；
- 空迭代区间不展开：上游按 `ceilDivPositive`（`arith.divui`）算 trip count，静态 `ub <= lb` 会被展开成上界回绕的循环，运行时 `ub < lb` 更会把负差值解释成极大无符号值。因此 Pass A 静态 `ub <= lb` 丢 hint + remark，动态上界在调用前插 `arith.maxsi(ub, lb)` 钳位（`ub >= lb` 恒等，`ub < lb` 时主循环与 epilogue 同样零次迭代）；
- 带 `break`/`continue`/`else` 的 `pto.range(...)` loop 走 `scf.while` 降级路径，无法承载 hint，前端直接报错；
- `unroll_factor` 必须 ≤ 2^31−1：attr 编码为 signless i32 并按有符号读回，更大的值会回绕成负数。前端在构造期报错；后端同样坚持该契约——Pass A 对类型/范围不合约的 factor（如手写 IR 里的 i64 attr）直接报错，不会截断成负 factor；
- Pass A 的展开 fixpoint 不设轮数上限：每轮重新 walk 拾取外层展开克隆出的内层带注解 loop，直到某轮不再有任何变化为止。固定轮数上限会让超过该深度的嵌套 `full` hint 静默残留并丢失，违反 `full` 对静态循环强制 native 展开的契约；
- Pass A 不使用 greedy pattern driver，而是按 post-order（内层先处理）手动驱动 `loopUnrollByFactor`：该 util 用内部 `IRRewriter` 删除被展开的 loop，绕过 driver 的 listener，会使 driver worklist 中的指针悬空；post-order 保证 erase 外层 loop 时其内层已全部处理完，不存在悬空指针；
- factor 上限：`max-unroll-factor`（默认 1024）限制 native 展开的 factor，超限丢 hint + remark，防止巨大 factor 导致编译器挂死/OOM；
- metadata 通道的两个实现要点：(a) LLVM 19 的 `convert-scf-to-cf` 不会把 `llvm.loop_annotation` 从 `scf.for` 传到 latch `cf.br`（上游新版本才支持），因此 Pass B 用自定义 pattern 降级带注解 loop 并以 ODS 裸名 `loop_annotation` 挂到 latch（MLIR→LLVM IR 翻译经 `BrOp::getLoopAnnotationAttr()` 按裸名查找）;(b) 该降级**必须覆盖整个函数**而非只处理带注解的 loop:只降级内层带注解 loop 会把新生成的 condition/body/latch/exit blocks 留在外层 single-block region 内（无 hint 的外层 `scf.for`、`scf.if`、`scf.while` 等），在 stock 转换运行前就触发该 op 的 SingleBlock verifier 失败。因此 Pass B 组合上游 conversion patterns（处理无 hint 结构）与自定义 pattern(benefit=2，处理带注解 loop),用 `applyPartialConversion` 一次转换完毕；这也顺带满足"不折叠无关 IR"的约束——conversion driver 本就不做 folding。

### 3.2 重复展开的 by-construction 排除

#1000 担心"阶段一 CCE bypass 与阶段二 native unroll 对同一循环重复展开"。本设计中每个 loop 的 attr 只被消费一次：

- `full`/`factor`：只被 Pass A 消费（展开或丢弃），Pass B 只认 `enable`，对其他 attr 一律不触碰；
- `enable`：Pass A 在所有 native-unroll guard 之前直接跳过（不消费），由 Pass B 翻译成 metadata。

factor 展开生成的 epilogue loop 不带任何 hint attr，fixpoint 与后续管线都不会再处理它，天然防止二次展开。

### 3.3 架构总览

```
PTODSL 前端                          PTOAS 后端
─────────────                        ─────────────────────────────────────────────
pto.for_(..., unroll=...)            prepareVPTOForEmission:
pto.range(...)  (AST rewrite)          [Pass A] pto-unroll-loops
        │                                ├─ "full" / factor 可展开 → 原生展开，attr 移除
        ▼                                ├─ 无法展开 → 丢 hint + remark，loop 保留
scf.for {pto.unroll = "full"/"enable",   ├─ "enable" → 跳过（留给 Pass B）
         pto.unroll_factor = N}          └─ 非法 hint → 硬错误
        │                                SCCP / canonicalize / CSE
        └──────────────────────────▶     ─────────────────────────────
                                         VPTO emission pipeline（两个 emitter）:
                                           [Pass B] pto-convert-scf-to-cf-with-loop-hints
                                           ├─ "enable" → #llvm.loop_annotation，
                                           │  自定义 pattern 降级并挂 latch cf.br
                                           └─ 其余 scf.for/if/while → 上游 patterns
                                              （本 pass 替代 convert-scf-to-cf）
                                           convert-cf-to-llvm / translateModuleToLLVMIR
                                               → !llvm.loop.unroll.enable metadata
                                               │
                                               ▼
                                         BiSheng（enable loop 由其 cost model 决定
                                         full/partial；其余 loop 走默认估价）
```

---

## 4. PTODSL 前端设计

### 4.1 `pto.for_` 扩展

签名（`ptodsl/ptodsl/_control_flow.py:168`）：

```python
def for_(start, stop, *, step, unroll=None, unroll_factor=None):
    ...
```

- `unroll`：取值 `None | "full" | "enable"`（`"enable"` 为 metadata hint，见 §5.3；`"disable"` 不支持）；
- `unroll_factor`：`None` 或 ≥ 1 的 `int`；
- 入口参数校验：非法取值 / 互斥冲突 / 非正整数 factor 抛 `TypeError` 或 `ValueError`，诊断信息可定位到调用点；
- hint 沿 `_ForBuilder` → `_ForCM.__enter__`（`_control_flow.py`）传递，`scf.ForOp` 创建后立即通过共享 helper（`_tracing/control_flow.py` 的 `apply_unroll_hint`，配套校验函数 `normalize_unroll_hint`）挂 attr：

```python
def apply_unroll_hint(for_op, unroll, unroll_factor):
    if unroll is not None:
        for_op.operation.attributes["pto.unroll"] = StringAttr.get(unroll)
    if unroll_factor is not None:
        for_op.operation.attributes["pto.unroll_factor"] = IntegerAttr.get(
            IntegerType.get_signless(32), unroll_factor)
```

### 4.2 三条建 loop 路径统一接入

hint 必须在以下所有创建 `scf.for` 的路径上一致生效，全部走 `apply_unroll_hint`：

1. **普通路径**：`_ForCM.__enter__`（`_control_flow.py`）；
2. **carry / session 路径**：`_CarryForCM` → `Session.begin_carry_loop`（`_tracing/session.py`）→ `build_carry_loop_frame`（`_tracing/control_flow.py`），在该处 ForOp 创建后挂 attr；`begin_carry_loop` 签名透传 hint 参数；
3. **tile template tracing 路径**：`_tile_template_tracing.py` 的 `for_` 同步扩展。

### 4.3 新增 `pto.range`（Python 原生 `for` 的 hint carrier）

```python
for i in pto.range(0, N, unroll="full"):
    ...

for i in pto.range(0, N, unroll_factor=4):
    ...
```

- 只在 AST rewrite 场景下有意义；`pto.range(...)` 的 runtime 实现是一个立即报错的 marker（"pto.range 仅可用于被 AST rewrite 的 for 循环"），防止被当作普通 iterable 误用；
- `_ast_rewrite.py` 中扩展 `_range_triplet`（`:934`）与各 `visit_For` / `_rewrite_for`（`:1847`、`:1906`、`:2018`）：识别 `_is_pto_attr_call(stmt.iter, "range")`，抽取 start/stop/step 与 unroll kwargs，规范化为 `pto.for_(start, stop, step=step, unroll=..., unroll_factor=...)` 调用节点，与现有 `range(...)` → `pto.for_` 改写共用同一路径；
- `range`、`pto.range`、`pto.for_` 在无 hint 时生成的 IR 完全一致；
- `pto.static_range` 判别逻辑（`:542` 等）保持精确匹配，不得把 `pto.range` 误识别为 `static_range`；
- `break` / `continue` / loop-carried value 等现有原生控制流限制不变，并对 `pto.range` 提供同样的明确诊断；
- 嵌套循环只影响直接使用 `pto.range` 的层级。

---

## 5. PTOAS 后端设计

### 5.1 共享常量

将 `kUnrollAttrName` / `kUnrollFullValue`（原为 `PTOUnrollSIMTForPass.cpp` 私有）提升为共享常量（`include/PTO/IR/PTO.h`），新增 `kUnrollFactorAttrName = "pto.unroll_factor"` 与 `isValidUnrollFactorAttr` 契约校验，供 pass 与文档共用。

### 5.2 Pass A：`PTOUnrollLoops`（重构 `PTOUnrollSIMTFor`，唯一消费者）

- **新 pass 名**：`pto-unroll-loops`；保留 `pto-unroll-simt-for` 作为 alias（两个现存测试通过 `--mlir-print-ir-after=pto-unroll-simt-for` 引用，行为不变，零回归）；
- **位置不变**：`prepareVPTOForEmission` 内、SCCP/canonicalize/CSE 之前（`tools/ptoas/ptoas.cpp`），保留 #838 "展开后常量分支被折叠"的收益；
- **两阶段结构**：先 walk 全函数校验所有 hint（收集全部诊断后统一失败——函数 pass adaptor 在某个函数失败后可能跳过其余函数，诊断必须函数内完备），合法再进入展开 fixpoint；
- **校验**（硬错误）：`pto.unroll` 非 `"full"`/`"enable"` 值（`"disable"` 及未知值）、两 attr 同现、factor 不符合 signless i32 正数契约（`isValidUnrollFactorAttr`）；`"enable"` 不消费、原样留给 Pass B(该判断排在所有 native-unroll guard 之前，否则空 body / 非 index loop 的 hint 会被静默丢弃);
- **处理逻辑**（校验通过后）：
  - `pto.unroll = "full"`：静态 lb/ub/step、正 step、可计算 trip count → `loopUnrollByFactor(tripCount)` 全展开，loop 与 attr 一并消失；动态 trip 无法展开 → 丢 hint + remark，loop 保留；
  - `pto.unroll_factor = N`：`loopUnrollByFactor(N)`（动态 bounds 同样支持，上游 util 自动生成 epilogue 并穿线 live-out carry）；成功后 attr 移除；N=1、动态 step、超过 `max-unroll-factor`（默认 1024）→ 丢 hint + remark，loop 保留；
  - 空 body loop：丢 hint + remark（`loopUnrollByFactor` 对空 body 是假成功，会让 fixpoint 死循环）；
- **放开 SIMT-context 限制**：#838 的 auto-detect（trip count ≤ 64 自动展开）已移除，现存逻辑只认显式 attr——显式 attr 即用户意图，在非 SIMT 函数中静默忽略反而违反直觉。删除 `isInSIMTContext` 检查，pass 文档同步更新；
- **可选护栏**：full unroll 静态 trip count 超过阈值（默认 1024，可用 pass option 调整）时 emit warning，防止 IR 体积爆炸。

### 5.3 Pass B：`PTOConvertSCFToCFWithLoopHints`（承担 SCF→CF 转换 + `enable` 透传）

v2 曾整体移除该 pass;v3 为满足 #1242 Req2 的 enable 验收标准恢复其最小子集;v4 按评审意见把它从"局部降级"改为**完整的 PTOAS 专用 SCF→CF 转换**,并相应改名（原名 `pto-lower-loop-hints` 已不能反映其职责）:

- **pass 名**：`pto-convert-scf-to-cf-with-loop-hints`,func-level;
- **插入点**：两个 emitter pipeline 中原 `createConvertSCFToCFPass()` 的位置——本 pass **替代**它，两者不可同时运行（会冗余）;必须排在所有 structured-loop 变换之后，确保没有后续 pass 克隆 loop 时丢失 hint;
- **翻译**：`{pto.unroll = "enable"}` → `#llvm.loop_annotation<unroll = <disable = false>>`（即 `!llvm.loop.unroll.enable`,LLVM ForceEnable 语义)；loop 上已有 `llvm.loop_annotation` 时合并 unroll 字段（已有 unroll 条目被覆盖时 warning);其余 attr 一律不触碰(full/factor 归 Pass A,理论上到不了这里);
- **转换机制**:`populateSCFToControlFlowConversionPatterns`（上游全套 for/if/while/forall/parallel）+ 自定义 `LowerAnnotatedForPattern`(benefit=2,覆盖上游 `ForLowering`),经 `applyPartialConversion` 一次完成（scf 系全部标记 illegal),与上游 `SCFToControlFlowPass` 同构。带注解 loop 的注解以 ODS 裸名 `loop_annotation` 挂到 latch `cf.br`(MLIR→LLVM IR 翻译经 `BrOp::getLoopAnnotationAttr()` 按裸名查找,`convert-cf-to-llvm` 原样转发分支属性);
- **为什么必须整体转换**：只降级带注解 loop 会把新生成的 condition/body/latch/exit blocks 留在外层 single-block region 内，触发 `scf.for`/`scf.if`/`scf.while` 的 SingleBlock verifier 失败（emitter 的 PassManager 开启 `enableVerifier()`,失败发生在 stock 转换运行之前）;
- **副作用差异**:conversion driver 不做 folding,因此不存在 greedy driver 那种"顺带折叠无关 op"的问题；空 body 的 enable loop 也按上游行为正常降级（不再被当死代码删除）。

### 5.4 与 #838 bug 规避语义的关系

- SIMT 内显式 `{pto.unroll = "full"}` 的常量循环仍在 SCCP/canonicalize 之前被强制全展开，`END @!P0` 规避路径不回归；
- 现存测试 `test/test_unroll_annotation.mlir` 与 `test/lit/vpto/unroll_inline_simt_sections.pto` 不需要修改（pass alias 保持名字与行为）。

---

## 6. 诊断与错误处理

| 场景 | 行为 |
|---|---|
| `unroll` 与 `unroll_factor` 同时指定 | PTODSL 前端 `ValueError` |
| `unroll` 取值非法 | PTODSL 前端 `ValueError`（列出合法取值） |
| `unroll_factor` 非正整数 / 非编译期常量 | PTODSL 前端 `TypeError` / `ValueError` |
| `unroll_factor` 超过 signless i32 上限（2^31−1） | PTODSL 前端 `ValueError` |
| 普通路径 `range(...)` / `pto.range(...)` 使用常量非正 step | PTODSL 前端 `PTODSLAstRewriteError`（负 step 仅带 break/continue 的 `pto._while` 路径支持） |
| `pto.range` 在非 AST-rewrite 上下文被调用 | `RuntimeError`（提示仅用于 rewrite 场景） |
| 手写 IR 中 attr 种类错误（`pto.unroll` 非 string / `pto.unroll_factor` 非 integer） | Pass A `emitError`（否则 typed getter 返回空，malformed hint 会静默留在 IR 中） |
| 手写 IR 中 `pto.unroll` 未知字符串（含不支持的 `"disable"`；`"enable"` 合法） | Pass A `emitError` |
| 手写 IR 中 `pto.unroll_factor` 类型/范围不合约（非 signless i32 或非正） | Pass A `emitError` |
| 手写 IR 中 `pto.unroll` 与 `pto.unroll_factor` 同时出现在一个 loop 上 | Pass A `emitError`（互斥） |
| `"full"` / factor 无法原生展开（动态 trip / 动态 step / 超 `max-unroll-factor` / factor=1 / 空 body / 静态空迭代区间 `ub <= lb` / 非 index 归纳变量） | Pass A emit remark + 丢 hint，loop 保留，编译继续 |
| full unroll trip count 超过护栏阈值 | Pass A warning，仍执行展开 |

---

## 7. 测试计划

### 7.1 PTODSL 前端测试（`ptodsl/tests/`）

- `for_(..., unroll="full")` / `unroll="enable"` / `unroll_factor=4` 生成的 `scf.for` 携带正确 attr；`unroll="disable"` 被拒绝；
- `for i in pto.range(...)` 与 `with pto.for_(...)` 生成相同 IR（bounds / step / attr / SSA 语义逐字节一致）；
- `.carry(...)` 循环携带 hint 并正确编译（live-out carry 值正确）；
- `range` / `pto.range` / `pto.for_` 无 hint 时 IR 完全一致；
- 互斥参数、非法 `unroll` 取值、非整数 / 动态 factor 的稳定诊断；
- `pto.range` 的 start/stop/step 组合与 Python `range` 语义一致（单参数形式、负数界等）；普通路径（无 break/continue）下降为 `scf.for`，仅支持正 step，常量非正 step 由前端报错（bool 是 int 子类：`step=False` 按 0 拒绝，`step=True` 归一化为 1——下游 index coercion 拒绝 bool）；负 step 仅在带 break/continue 的 `pto._while` 路径受支持；
- 嵌套循环中只有直接使用 `pto.range` 的层级携带 hint。

### 7.2 PTOAS lit 测试（`test/`、`test/lit/vpto/`）

**Native unroll（Pass A，唯一消费者）**：

- factor 整除（无 epilogue）/ 不整除（epilogue 存在且 init args 正确）；
- trip count < factor、0 次、1 次迭代；
- 动态 upper bound 的 factor 展开；
- 带 live-out carry values 的 factor 展开（验证 epilogue 穿线）；
- 嵌套循环（含 `promoteIfSingleIteration` 把带 hint 内层 splice 到父 block 的场景）、循环内含条件分支与同步操作（`pto.barrier`）；
- 非 SIMT 上下文中的显式 attr 循环同样被展开（放开限制后的行为锁定）。

**Hint 丢弃与诊断（Pass A）**：

- 动态 trip 的 `"full"`、动态 step / 超上限 / factor=1 的 factor → remark + 丢 hint、loop 保留；
- 非法 hint（未知值含不支持的 `disable`、互斥、非 i32 factor；`enable` 合法）→ error 而非静默通过；单函数内多个非法 loop 的诊断一次性全部发出（不受函数级并行调度影响）；
- 无 hint 循环的 IR 与最终产物逐字节不变。

**回归**：

- `test/test_unroll_annotation.mlir`、`test/lit/vpto/unroll_inline_simt_sections.pto` 不修改、不回归（#838 规避路径）。

### 7.3 端到端验证

- SIMTVF ST（`test/dsl-st/unroll_hint_numeric.py`）在模拟器与 A5 硬件上验证展开后 kernel 数值正确；
- 带 `.carry()` 的循环与普通循环均完成 PTODSL → PTOAS → BiSheng 闭环。

---

## 8. 跨层同步清单（依据 `.claude/rules/cross-layer-sync.md`）

- [ ] PTODSL：`_control_flow.py`、`_tracing/control_flow.py`、`_tracing/session.py`、`_tile_template_tracing.py`、`_ast_rewrite.py`、`pto.py`（导出 `range`）
- [ ] 共享常量头文件（`pto.unroll` / `pto.unroll_factor` attr 名与取值）
- [ ] Pass A：`include/PTO/Transforms/Passes.td`（`pto-unroll-loops` + `pto-unroll-simt-for` alias）、`lib/PTO/Transforms/PTOUnrollSIMTForPass.cpp`（重构，视情况改名）
- [ ] `tools/ptoas/ptoas.cpp`（Pass A 替换）
- [ ] `docs/`：PTODSL user guide 增加 unroll hint 章节（语义矩阵、无法展开时的丢弃行为、与 `static_range` 的区别）
- [ ] 上述全部测试

## 9. 工作量评估

合并实现（native full/factor unroll + #838 重构）的主体是 Pass A 的 attr 驱动调度、边界兜底与丢弃逻辑（约百余行，重活均在上游 `loopUnrollByFactor`）及对应测试。v2 移除阶段一后，"阶段互斥"这一 #1000 关注点在构造上不存在。

---

---

# PTODSL / PTOAS Loop Unroll Hint — Design Document

> **Revision history (v5)**: adds `pto-promote-persistent-fragment-loops` -
> persistent fragment materialization requires loops touching a persistent
> buffer to be fully unrolled, which previously relied on authors writing
> `full` by hand.  The pass discovers loops via the pointer use graph of
> `llvm.alloca {pto.persistent}` (GEP chains; a load's data result is not
> followed) and promotes them to `full` (overriding
> `enable`), attaching a `pto.persistent_unroll` marker that upgrades Pass
> A's drop-with-remark fallback into a hard error.  See the "automatic
> promotion" section of
> `docs/designs/ptoas_persistent_simt_fragment_plan.md`.
>
> **Revision history (v4)**: review pointed out that lowering only the
> annotated loops leaves the freshly created cf blocks inside an enclosing
> single-block region (an unannotated outer `scf.for`, an `scf.if`, ...) and
> trips its SingleBlock verifier.  The pass now performs the **complete
> SCF-to-CF conversion for the function** - upstream conversion patterns plus
> a higher-benefit pattern for annotated loops - and therefore **replaces**
> `createConvertSCFToCFPass` in both emitter pipelines.  It was renamed to
> `pto-convert-scf-to-cf-with-loop-hints` accordingly (it no longer just
> handles hints; it owns the control-flow conversion).  Also fixes Pass A,
> where the `enable` check sat behind the native-unroll guards and silently
> dropped the hint on empty-body and non-index loops.
>
> **Revision history (v3)**: after v2 removed stage 1 entirely (Pass B and
> `enable`/`disable`), #1242 Req2's acceptance criteria ("public `pto.for_`
> can carry an unroll-enable hint; the hint reaches LLVM/BiSheng instead of
> being force-unrolled by the frontend") could not be satisfied.  v3
> **restores `unroll="enable"` as the single metadata channel**: a slimmed
> `pto-convert-scf-to-cf-with-loop-hints` handles only `enable` →
> `!llvm.loop.unroll.enable`; `disable` stays removed, and the drop semantics
> of `full`/`factor` is unchanged.
>
> ~~Revision history (v2)~~: the initial design had two stages - Pass A
> (native unrolling) plus Pass B forwarding leftover hints as LLVM loop
> metadata.  After review, stage 1 was removed and `enable`/`disable` with
> it.  v3 partially reverted that decision (restoring only `enable`).
>
> Related issues:
> - [Issue #1242](https://github.com/hw-native-sys/PTOAS/issues/1242) Requirement 2 — `pto.for_` loop-unroll hint (v1's `unroll="enable"` was removed with stage 1; the interface need is covered by `unroll="full"` / `unroll_factor=N`)
> - [Issue #1000](https://github.com/hw-native-sys/PTOAS/issues/1000) — Loop Unroll Hint support (incl. `pto.range`, factor unroll, two-phase plan)
> - [PR #838](https://github.com/hw-native-sys/PTOAS/pull/838) — `PTOUnrollSIMTForPass` (temporary workaround for a BiSheng AICore backend bug, refactored by this design)

---

## 1. Background and Motivation

### 1.1 Requirements

**Issue #1242 Requirement 2**: SIMTVF codegen needs the semantics of a no-factor `#pragma unroll` — keep the device-side loop and delegate the unrolling decision to the LLVM/BiSheng cost model (full or partial). PTODSL today offers only two extremes:

- `pto.static_range`: forces full unrolling at trace time, increasing trace/compile time, IR size, and register pressure;
- `pto.for_`: keeps the device loop but cannot carry any frontend unroll hint.

**Issue #1000**: the TileLang-PTO backend needs to lower `T.unroll(..., explicit=False)` into equivalent semantics (compiler-side unrolling rather than DSL frontend forced unrolling). The issue proposed a two-phase plan: phase 1 lets Bisheng/CCE perform the unrolling (hint pass-through); phase 2 unrolls natively in PTOAS.

**PR #838 (historical context)**: `PTOUnrollSIMTForPass` was a temporary workaround for a BiSheng AICore backend bug — after SCF→CF→LLVM lowering of `scf.for` + constant-condition `scf.if` in SIMTVF kernels, the AICore backend mishandles the `SimtEntry` calling convention and emits a predicated `END @!P0`. The workaround force-unrolls constant-trip-count loops annotated `{pto.unroll = "full"}` inside SIMT contexts, letting downstream SCCP + canonicalize eliminate the constant branches and produce branch-free straight-line code. The comment on issue #1000 explicitly calls this pass a temporary workaround to be refactored into a general unroll capability this time.

### 1.2 Goals

1. `pto.for_` / the new `pto.range` support the `unroll="full"` and
   `unroll_factor=N` hints: PTOAS unrolls natively before LLVM lowering
   (#1242's loop-unroll interface request, #1000 phase 2).
2. Refactor `PTOUnrollSIMTFor` from a SIMT-only, full-only stopgap pass into a
   general attribute-driven unroll pass, keeping the #838 bug-workaround
   semantics intact.
3. Loops without a hint behave exactly as before.

(v1 also targeted `unroll="enable"`/`"disable"` metadata forwarding; removed
after review - see the revision note at the top and §5.3.)

### 1.3 Non-goals

- No changes to the LLVM/BiSheng unroll cost model;
- No change to `pto.static_range`'s trace-time full-unroll semantics;
- No full unrolling of arbitrary dynamic loops (a dynamic-bound `"full"` cannot be unrolled natively: the hint is dropped with a remark and the loop is kept);
- No TileLang codegen changes.

---

## 2. Current-State Analysis

### 2.1 Existing unroll capability

`PTOUnrollSIMTForPass` (formerly `lib/PTO/Transforms/PTOUnrollSIMTForPass.cpp`, refactored into `PTOUnrollLoopsPass.cpp` by this design):

- Only handles `scf.for` explicitly annotated `{pto.unroll = "full"}` (`PTOUnrollSIMTForPass.cpp:58-66`);
- Only applies inside SIMT contexts (`pto.simt_entry` functions or inline `pto.section.simt` regions);
- Requires static lb/ub/step and a positive step; fully unrolls via `loopUnrollByFactor(tripCount)`;
- Runs in `prepareVPTOForEmission` before SCCP/canonicalize/CSE (`tools/ptoas/ptoas.cpp:3074`), so constant branches are folded downstream after unrolling.

PTODSL currently has **no public API** that sets the `pto.unroll` attribute — it appears only in hand-written `.pto` tests (`test/test_unroll_annotation.mlir`, `test/lit/vpto/unroll_inline_simt_sections.pto`).

### 2.2 Key finding: the vendored MLIR already has a complete loop-annotation delivery chain

No changes are needed in the LLVM/BiSheng interface layers:

1. **SCF→CF** (historical record, v1 rationale): the vendored LLVM 19 (feature-vpto) `SCFToControlFlow` does **not** copy `llvm.loop_annotation` from `scf.for` to the latch `cf.br` (that support only exists in newer upstream versions; the LLVM 21 in `llvm-workspace` already has it).  v1 therefore had Pass B lower annotated loops itself; v2 removed stage 1 and the chain is no longer used;
2. **CF→LLVM**: `BranchOpLowering` / `CondBranchOpLowering` preserve all attributes onto `llvm.br` / `llvm.cond_br`
   (`llvm-workspace/llvm-project/mlir/lib/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.cpp:130-170`);
3. **MLIR→LLVM IR**: `translateModuleToLLVMIR` translates `#llvm.loop_annotation` on the latch into `!llvm.loop` metadata
   (`llvm-workspace/llvm-project/mlir/lib/Target/LLVMIR/LoopAnnotationTranslation.cpp:124-138`):

   | MLIR attr | LLVM IR metadata |
   |---|---|
   | `#llvm.loop_annotation<unroll = <disable = false>>` | `!{!"llvm.loop.unroll.enable"}` |
   | `#llvm.loop_annotation<unroll = <disable = true>>` | `!{!"llvm.loop.unroll.disable"}` |
   | `#llvm.loop_annotation<unroll = <full = true>>` | `!{!"llvm.loop.unroll.full"}` |
   | `#llvm.loop_annotation<unroll = <count = N>>` | `!{!"llvm.loop.unroll.count", i32 N}` |

   `!llvm.loop.unroll.enable` is exactly the equivalent of a no-factor `#pragma unroll` (the target semantics of #1242 req2).

4. **Both VPTO emission pipelines share the same structure**: `createConvertSCFToCFPass()` → `createConvertControlFlowToLLVMPass()` → `translateModuleToLLVMIR`
   (`lib/PTO/Transforms/VPTOLLVMEmitter.cpp:14195-14200`, near `lib/PTO/Transforms/VPTOCANN900LLVMEmitter.cpp:11875`).
   `prepareVPTOForEmission` (`tools/ptoas/ptoas.cpp:3252`) runs before every emission path.

### 2.3 Key finding: upstream unroll utilities are sufficient

`loopUnrollByFactor` (`llvm-workspace/llvm-project/mlir/lib/Dialect/SCF/Utils/Utils.cpp:364`):

- Supports **dynamic bounds**: automatically generates the trip-count computation and an epilogue remainder loop;
- **Live-out carry values are threaded correctly through the epilogue**: main-loop results become the epilogue's init args and uses are replaced automatically (the "live-out values cannot be unrolled" restriction noted in the current pass does not apply on this path);
- `loopUnrollFull` is a wrapper around `loopUnrollByFactor(tripCount)`;
- Restrictions: a positive step is required and the factor must be positive; on the dynamic-bound path the trip count is computed with `ceilDivPositive` (`arith.divui`), so a runtime `upper < lower` reinterprets the negative difference as a huge unsigned value — upstream explicitly leaves a `TODO: Add dynamic asserts for negative lb/ub/step`.  Pass A therefore guards the call itself: a static `ub <= lb` drops the hint, and a dynamic upper bound is clamped with `arith.maxsi(ub, lb)` (the identity when `ub >= lb`; when `ub < lb` both the unrolled main loop and the epilogue iterate zero times, like the original loop).

The heaviest work of #1000 phase 2 (remainder-loop generation, loop-carried SSA threading) is therefore already done by the upstream utility; this design only needs attribute-driven scheduling, bounds guarding, and hint-dropping logic.

### 2.4 Attribute-survival risk audit

The only pass between `prepareVPTOForEmission` and emission that rebuilds `scf.for` is `PTONarrowVPTOLoopCounters`, which preserves all attributes (`lib/PTO/Transforms/PTONarrowVPTOLoopCounters.cpp:115-117`, `newFor->setAttrs(forOp->getAttrs())`). Discardable attributes survive canonicalize/SCCP/CSE/LICM by default. A lit test locks this behavior in place.

---

## 3. Overall Design

### 3.1 Semantics matrix

One attribute encoding (discardable attrs on `scf.for`), one consumption path (native unroll in Pass A):

| Frontend syntax | `scf.for` attr | Semantics | Native handling (Pass A) | When native unrolling is impossible |
|---|---|---|---|---|
| (no hint) | none | unchanged | not handled | — |
| `unroll="enable"` | `pto.unroll = "enable"` | keep the loop; the compiler's cost model decides full/partial | not handled (skipped before every native-unroll guard, left for Pass B) | — (always becomes metadata) |
| `unroll="full"` | `pto.unroll = "full"` | PTOAS forced full unroll | `loopUnrollByFactor(tripCount)`; the loop disappears | dynamic trip: drop hint + remark, keep the loop |
| `unroll_factor=N` | `pto.unroll_factor = N` (i32) | PTOAS unrolls by N | `loopUnrollByFactor(N)`; main + epilogue loops | dynamic step / over-cap / N=1: drop hint + remark, keep the loop |

Constraints:

- `unroll` and `unroll_factor` are mutually exclusive;
- `unroll_factor` must be a Python compile-time integer constant ≥ 1;
- hints do not change `.carry(...)` loop-carried-value semantics;
- `pto.for_` / `range` without a hint behave identically to today, IR included.

Edge cases (implementation decisions):

- `unroll_factor=1` has no native meaning; the hint is dropped with a remark;
- empty-body loops (a no-op for `loopUnrollByFactor`) are not unrolled
  natively; the hint is dropped with a remark;
- only index-typed loops can be unrolled: `scf.for` also accepts signless
  integer bounds (i16/i32), but `loopUnrollByFactor` always builds its
  step/offset arithmetic with `arith::ConstantIndexOp`, so unrolling an i16
  loop emits `arith.muli(i16, index)` and an `scf.for` whose step type no
  longer matches its bounds - both verifier failures.  Non-index loops drop
  the hint with a remark;
- empty iteration spaces are never unrolled: upstream computes the trip count
  with `ceilDivPositive` (`arith.divui`), so a static `ub <= lb` would unroll
  into a loop with a wrapped-around upper bound, and a runtime `ub < lb`
  would reinterpret the negative difference as a huge unsigned value.  Pass A
  therefore drops the hint with a remark for a static `ub <= lb`, and clamps a
  dynamic upper bound with `arith.maxsi(ub, lb)` before the call (the identity
  when `ub >= lb`; zero-trip for both the main loop and the epilogue when
  `ub < lb`);
- `pto.range(...)` loops with `break`/`continue`/`else` lower through
  `scf.while` and cannot carry a hint, so the frontend reports an error;
- `unroll_factor` must be ≤ 2^31−1: the attribute is encoded as a signless
  i32 and read back as a signed value, so larger values wrap negative.  The
  frontend rejects them at construction time; the backend enforces the same
  contract — Pass A reports an error for an out-of-contract factor (e.g. an
  i64 attribute in handwritten IR) instead of truncating it negative;
- Pass A's unroll fixpoint has no round cap: each round re-walks the function
  to pick up annotated inner loops cloned by an outer unroll, and stops only
  when a round changes nothing.  A fixed round budget would silently leave
  hints behind on loops nested deeper than the budget, and a leftover `full`
  hint being dropped would violate the forced native-unroll contract for
  static loops;
- Pass A does not use the greedy pattern driver; it drives
  `loopUnrollByFactor` manually in post-order (innermost first): the utility
  erases the unrolled loop with its own internal `IRRewriter`, bypassing the
  driver's listener and leaving dangling pointers in a driver worklist.
  Post-order guarantees that by the time an outer loop is unrolled and
  erased, every annotated inner loop has already been processed;
- factor cap: `max-unroll-factor` (default 1024) bounds native factor
  unrolling; beyond it the hint is dropped with a remark, preventing a huge
  factor from hanging/OOMing the compiler;
- two implementation points of the metadata channel: (a) LLVM 19's
  `convert-scf-to-cf` does not propagate `llvm.loop_annotation` from
  `scf.for` to the latch `cf.br` (only newer upstream versions do), so Pass B
  lowers annotated loops with a custom pattern and stores the annotation on
  the latch under the bare ODS name `loop_annotation` (the MLIR-to-LLVM-IR
  translation looks it up via `BrOp::getLoopAnnotationAttr()`); (b) that
  lowering **must cover the whole function** rather than just the annotated
  loops: lowering an inner annotated loop alone leaves its new
  condition/body/latch/exit blocks inside the enclosing single-block region
  (an unannotated outer `scf.for`, `scf.if`, `scf.while`, ...) and fails that
  op's SingleBlock verifier before the stock conversion ever runs.  Pass B
  therefore combines the upstream conversion patterns (for unannotated
  structures) with the custom pattern (benefit=2, for annotated loops) in a
  single `applyPartialConversion`; this also satisfies the "do not fold
  unrelated IR" constraint for free - the conversion driver does no folding

### 3.2 Double unrolling excluded by construction

#1000 worried about "phase-1 CCE bypass and phase-2 native unroll both
unrolling the same loop".  In this design each loop's attribute is consumed
exactly once:

- `full`/`factor`: consumed only by Pass A (unrolled or dropped); Pass B
  only recognizes `enable` and never touches anything else;
- `enable`: skipped by Pass A before every native-unroll guard, and translated into metadata by Pass B.

The epilogue loop produced by a factor unroll carries no hint attribute, so
neither the fixpoint nor any later pass touches it again - re-unrolling is
prevented without any tagging.

### 3.3 Architecture overview

```
PTODSL frontend                      PTOAS backend
─────────────                        ─────────────────────────────────────────────
pto.for_(..., unroll=...)            prepareVPTOForEmission:
pto.range(...)  (AST rewrite)          [Pass A] pto-unroll-loops
        │                                ├─ "full" / factor unrollable → unroll
        ▼                                │   natively, attribute removed
scf.for {pto.unroll = "full"/"enable",   ├─ not unrollable → drop hint + remark
         pto.unroll_factor = N}          ├─ "enable" → skipped (left for Pass B)
        │                                └─ malformed hint → hard error
        └──────────────────────────▶     SCCP / canonicalize / CSE
                                         ─────────────────────────────
                                         VPTO emission pipeline (both emitters):
                                           [Pass B] pto-convert-scf-to-cf-with-loop-hints
                                           ├─ "enable" → #llvm.loop_annotation,
                                           │  custom pattern, annotation on
                                           │  the latch cf.br
                                           └─ every other scf.for/if/while →
                                              upstream patterns (this pass
                                              replaces convert-scf-to-cf)
                                           convert-cf-to-llvm / translateModuleToLLVMIR
                                               → !llvm.loop.unroll.enable metadata
                                               │
                                               ▼
                                         BiSheng (the enable loop's full/partial
                                         unroll is chosen by its cost model;
                                         everything else gets the default
                                         evaluation)
```

---

## 4. PTODSL Frontend Design

### 4.1 `pto.for_` extension

Signature (`ptodsl/ptodsl/_control_flow.py:168`):

```python
def for_(start, stop, *, step, unroll=None, unroll_factor=None):
    ...
```

- `unroll`: one of `None | "full"` (v1's `"enable"`/`"disable"` were removed together with stage 1);
- `unroll_factor`: `None` or an `int` ≥ 1;
- Entry-point validation: illegal values / mutually-exclusive conflicts / non-positive factor raise `TypeError` or `ValueError` with diagnostics that locate the call site;
- The hint flows through `_ForBuilder` → `_ForCM.__enter__` (`_control_flow.py`); the attribute is attached immediately after `scf.ForOp` creation via a shared helper (`apply_unroll_hint` in `_tracing/control_flow.py`, with `normalize_unroll_hint` as the companion validator):

```python
def apply_unroll_hint(for_op, unroll, unroll_factor):
    if unroll is not None:
        for_op.operation.attributes["pto.unroll"] = StringAttr.get(unroll)
    if unroll_factor is not None:
        for_op.operation.attributes["pto.unroll_factor"] = IntegerAttr.get(
            IntegerType.get_signless(32), unroll_factor)
```

### 4.2 Unified integration across all three loop-building paths

The hint must take effect consistently on every path that creates an `scf.for`, all going through `apply_unroll_hint`:

1. **Plain path**: `_ForCM.__enter__` (`_control_flow.py`);
2. **Carry / session path**: `_CarryForCM` → `Session.begin_carry_loop` (`_tracing/session.py`) → `build_carry_loop_frame` (`_tracing/control_flow.py`), attaching the attribute right after the ForOp is created there; `begin_carry_loop` forwards the hint parameters;
3. **Tile-template tracing path**: the `for_` in `_tile_template_tracing.py` is extended in lockstep.

### 4.3 New `pto.range` (hint carrier for native Python `for`)

```python
for i in pto.range(0, N, unroll="full"):
    ...

for i in pto.range(0, N, unroll_factor=4):
    ...
```

- Meaningful only under AST rewrite; the runtime implementation of `pto.range(...)` is a marker that raises immediately ("pto.range may only be used in AST-rewritten for loops"), preventing misuse as a plain iterable;
- In `_ast_rewrite.py`, extend `_range_triplet` (`:934`) and the various `visit_For` / `_rewrite_for` sites (`:1847`, `:1906`, `:2018`): recognize `_is_pto_attr_call(stmt.iter, "range")`, extract start/stop/step and the unroll kwargs, and normalize to a `pto.for_(start, stop, step=step, unroll=..., unroll_factor=...)` call node, sharing the existing `range(...)` → `pto.for_` rewrite path;
- `range`, `pto.range`, and `pto.for_` produce identical IR when no hint is given;
- The `pto.static_range` detection (`:542` etc.) keeps exact matching and must never misidentify `pto.range` as `static_range`;
- Existing native-control-flow restrictions (`break` / `continue` / loop-carried values) are unchanged, with the same clear diagnostics for `pto.range`;
- Nested loops: only the levels directly using `pto.range` carry the hint.

---

## 5. PTOAS Backend Design

### 5.1 Shared constants

Promote `kUnrollAttrName` / `kUnrollFullValue` (previously private in `PTOUnrollSIMTForPass.cpp`) into a shared header (`include/PTO/IR/PTO.h`), and add `kUnrollFactorAttrName = "pto.unroll_factor"` plus the `isValidUnrollFactorAttr` contract check, shared by the pass and the docs.

### 5.2 Pass A: `PTOUnrollLoops` (refactor of `PTOUnrollSIMTFor`, the only consumer)

- **New pass name**: `pto-unroll-loops`; keep `pto-unroll-simt-for` as an alias (two existing tests reference it via `--mlir-print-ir-after=pto-unroll-simt-for`; behavior is unchanged, zero regression);
- **Position unchanged**: inside `prepareVPTOForEmission`, before SCCP/canonicalize/CSE (`tools/ptoas/ptoas.cpp`), preserving the #838 benefit of folding constant branches after unrolling;
- **Two-phase structure**: first walk the function and validate every hint (collecting all diagnostics before failing once - the function pass adaptor may stop scheduling functions after the first failure, so diagnostics must be complete per function), then run the unroll fixpoint;
- **Validation** (hard errors): a `pto.unroll` value other than `"full"`/`"enable"` (`"disable"` and unknown values), both attributes on one loop, or a factor violating the signless-i32 positive-factor contract (`isValidUnrollFactorAttr`); `"enable"` is passed through untouched for Pass B (this check runs before every native-unroll guard - otherwise an empty-body or non-index loop would silently lose the hint);
- **Handling logic** (after validation):
  - `pto.unroll = "full"`: static lb/ub/step, positive step, computable trip count → fully unroll via `loopUnrollByFactor(tripCount)`; the loop and the attribute disappear.  A dynamic trip count cannot be unrolled → drop the hint with a remark and keep the loop;
  - `pto.unroll_factor = N`: unroll via `loopUnrollByFactor(N)` (dynamic bounds supported; the upstream utility generates the epilogue and threads live-out carries); on success the attribute is removed.  N=1, a dynamic step, or a factor above `max-unroll-factor` (default 1024) → drop the hint with a remark and keep the loop;
  - empty-body loops: drop the hint with a remark (`loopUnrollByFactor` reports a no-op success on them, which would loop the fixpoint forever);
- **Lift the SIMT-context restriction**: #838's auto-detection (auto-unroll for trip count ≤ 64) has already been removed; only explicit attributes remain — an explicit attribute is user intent, and silently ignoring it outside SIMT contexts would be counterintuitive.  The `isInSIMTContext` check is removed and the pass documentation updated;
- **Optional guardrail**: warn when a full-unroll static trip count exceeds a threshold (default 1024, tunable via pass option) to prevent IR explosion.

### 5.3 Pass B: `PTOConvertSCFToCFWithLoopHints` (owns SCF-to-CF plus `enable` forwarding)

v2 removed this pass entirely; v3 restored its minimal subset for #1242
Req2's enable criteria; v4 turned it from a local lowering into the
**complete PTOAS-specific SCF-to-CF conversion** per review, and renamed it
accordingly (the old `pto-lower-loop-hints` no longer described its job):

- **Pass name**: `pto-convert-scf-to-cf-with-loop-hints`, func-level;
- **Insertion point**: where `createConvertSCFToCFPass()` used to sit in both
  emitter pipelines - this pass **replaces** it, and the two must not both
  run (that would be redundant).  It must stay after every structured-loop
  transformation so no later pass can clone a loop and lose its hint;
- **Translation**: `{pto.unroll = "enable"}` →
  `#llvm.loop_annotation<unroll = <disable = false>>` (i.e.
  `!llvm.loop.unroll.enable`, LLVM's ForceEnable semantics).  An existing
  `llvm.loop_annotation` is merged (an existing unroll entry is overwritten
  with a warning).  Every other attribute is untouched (`full`/factor belong
  to Pass A and cannot legitimately reach here);
- **Conversion mechanism**:
  `populateSCFToControlFlowConversionPatterns` (the full upstream set:
  for/if/while/forall/parallel) plus a custom `LowerAnnotatedForPattern`
  (benefit=2, overriding the upstream `ForLowering`), driven by a single
  `applyPartialConversion` with the scf ops marked illegal - structurally
  identical to upstream `SCFToControlFlowPass`.  The annotation is stored on
  the latch `cf.br` under the bare ODS name `loop_annotation` (the
  MLIR-to-LLVM-IR translation looks it up via
  `BrOp::getLoopAnnotationAttr()`; `convert-cf-to-llvm` forwards branch
  attributes verbatim);
- **Why the whole function**: lowering only the annotated loops leaves the
  new condition/body/latch/exit blocks inside the enclosing single-block
  region and fails the `scf.for`/`scf.if`/`scf.while` SingleBlock verifier -
  the emitter's PassManager has `enableVerifier()`, so this happens before
  the stock conversion would run;
- **Side-effect difference**: the conversion driver does no folding, so
  there is no "accidentally folds unrelated ops" hazard like the greedy
  driver had, and an empty-body enable loop is lowered like upstream would
  (it is no longer removed as dead code).

### 5.4 Relationship to the #838 bug-workaround semantics

- Constant loops explicitly annotated `{pto.unroll = "full"}` inside SIMT are still force-unrolled before SCCP/canonicalize; the `END @!P0` workaround path does not regress;
- The existing tests `test/test_unroll_annotation.mlir` and `test/lit/vpto/unroll_inline_simt_sections.pto` need no modification (the pass alias keeps both name and behavior).

---

## 6. Diagnostics and Error Handling

| Scenario | Behavior |
|---|---|
| `unroll` and `unroll_factor` given together | PTODSL frontend `ValueError` |
| Illegal `unroll` value | PTODSL frontend `ValueError` (listing legal values) |
| `unroll_factor` not a positive integer / not a compile-time constant | PTODSL frontend `TypeError` / `ValueError` |
| `unroll_factor` exceeds the signless i32 limit (2^31−1) | PTODSL frontend `ValueError` |
| Constant non-positive step on the plain `range(...)` / `pto.range(...)` path | PTODSL frontend `PTODSLAstRewriteError` (negative steps are only supported on the break/continue `pto._while` path) |
| `pto.range` called outside an AST-rewrite context | `RuntimeError` (rewrite-only hint) |
| Hand-written IR with a wrongly *kinded* attribute (`pto.unroll` not a string / `pto.unroll_factor` not an integer) | Pass A `emitError` (otherwise the typed getters return null and the malformed hint survives silently) |
| Hand-written IR with unknown `pto.unroll` string (including the unsupported `"disable"`; `"enable"` is legal) | Pass A `emitError` |
| Hand-written IR with an out-of-contract `pto.unroll_factor` (not signless i32, or non-positive) | Pass A `emitError` |
| Hand-written IR with both `pto.unroll` and `pto.unroll_factor` on one loop | Pass A `emitError` (mutual exclusion) |
| `"full"` / factor cannot be unrolled natively (dynamic trip / dynamic step / above `max-unroll-factor` / factor=1 / empty body / statically empty iteration space `ub <= lb` / non-index induction variable) | Pass A emits a remark + drops the hint; the loop is kept; compilation continues |
| Full-unroll trip count exceeds the guardrail threshold | Pass A warning; unroll still proceeds |

---

## 7. Test Plan

### 7.1 PTODSL frontend tests (`ptodsl/tests/`)

- `for_(..., unroll="full")` / `unroll="enable"` / `unroll_factor=4` produce `scf.for` with the correct attributes; `unroll="disable"` is rejected;
- `for i in pto.range(...)` and `with pto.for_(...)` produce identical IR (bounds / step / attributes / SSA semantics, byte-for-byte);
- `.carry(...)` loops carry the hint and compile correctly (correct live-out carry values);
- `range` / `pto.range` / `pto.for_` without hints produce identical IR;
- Stable diagnostics for mutually-exclusive arguments, illegal `unroll` values, and non-integer / dynamic factors;
- `pto.range` start/stop/step combinations match Python `range` semantics (single-argument form, negative bounds, etc.); the plain path (no break/continue) lowers to `scf.for` and only supports a positive step — a constant non-positive step is rejected by the frontend (bool is an int subclass: `step=False` is rejected as 0, `step=True` is normalized to 1 because downstream index coercion rejects bools); negative steps are only supported on the break/continue `pto._while` path;
- In nested loops, only the levels directly using `pto.range` carry the hint.

### 7.2 PTOAS lit tests (`test/`, `test/lit/vpto/`)

**Native unroll (Pass A, the only consumer)**:

- factor divides / does not divide the trip count (epilogue present with correct init args);
- trip count < factor, zero-trip, single-iteration;
- factor unroll with a dynamic upper bound;
- factor unroll with live-out carry values (epilogue threading);
- nested loops (including `promoteIfSingleIteration` splicing hinted inner loops into the parent block), bodies with conditionals and synchronization ops (`pto.barrier`);
- explicit-attribute loops outside SIMT contexts are unrolled too (locking in the lifted restriction).

**Hint dropping and diagnostics (Pass A)**:

- dynamic-trip `"full"`, dynamic-step / over-cap / factor=1 → remark + hint dropped, loop kept;
- malformed hints (unknown values including the unsupported `disable`, conflicting attrs, non-i32 factor; `enable` is legal) → errors, never silently accepted; multiple malformed loops in one function produce all diagnostics at once (immune to function-level parallel scheduling);
- unhinted loops produce byte-identical IR.

**Regression**:

- `test/test_unroll_annotation.mlir` and `test/lit/vpto/unroll_inline_simt_sections.pto` are unmodified and keep passing (#838 workaround path).

### 7.3 End-to-end validation

- A SIMTVF ST (`test/dsl-st/unroll_hint_numeric.py`) validates the unrolled kernels numerically on the simulator and on A5 hardware;
- Both `.carry()` loops and plain loops complete the PTODSL → PTOAS → BiSheng loop.

---

## 8. Cross-Layer Synchronization Checklist (per `.claude/rules/cross-layer-sync.md`)

- [ ] PTODSL: `_control_flow.py`, `_tracing/control_flow.py`, `_tracing/session.py`, `_tile_template_tracing.py`, `_ast_rewrite.py`, `pto.py` (export `range`)
- [ ] Shared-constants header (`pto.unroll` / `pto.unroll_factor` attr names and values)
- [ ] Pass A: `include/PTO/Transforms/Passes.td` (`pto-unroll-loops` + `pto-unroll-simt-for` alias), `lib/PTO/Transforms/PTOUnrollSIMTForPass.cpp` (refactored, possibly renamed)
- [ ] `tools/ptoas/ptoas.cpp` (Pass A replacement)
- [ ] `docs/`: PTODSL user guide gains an unroll-hint section (semantics matrix, hint-dropping behavior when native unrolling is impossible, difference from `static_range`)
- [ ] All tests listed above

## 9. Effort Estimate

The bulk of the implementation (native full/factor unroll + the #838 refactor) is Pass A's attribute-driven scheduling, bounds guarding, and hint-dropping logic (roughly a hundred lines — the heavy lifting lives in the upstream `loopUnrollByFactor`) plus the corresponding tests. With stage 1 removed in v2, #1000's "phase mutual exclusion" concern does not exist by construction.
