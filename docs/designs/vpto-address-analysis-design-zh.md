# VPTO 通用地址分析设计

## 1. 状态与结论

本文档提出一套面向 PTOAS VPTO 层的通用地址分析框架。它不是
`VPTOSoftPostUpdate` 的私有 helper，也不是把地址统一改写成某种固定类型或
固定 IR 形态的 normalize pass。

首期采用以下结构：

```text
VPTO op
  └── VPTOAddressSemanticsOpInterface
        ├── 当前访存使用的 typed base、optional offset 和 offset unit
        └── post-update 的 typed base/advance、unit、constraint 和结果状态

func::FuncOp
  ├── PTOValueEvolutionAnalysis
  │     ├── typed finite-width expression
  │     ├── range / cast / no-wrap
  │     └── loop recurrence / known step
  │
  └── PTOAddressAnalysis
        ├── 复用 PTOValueEvolutionAnalysis
        ├── 构建 typed AddressExpr
        └── 查询 effective address delta 和地址单位换算

consumer
  ├── VPTOSoftPostUpdate（首期正式 consumer）
  ├── loop counter narrowing（可复用 ValueEvolution）
  └── 其他需要已知步长或地址事实的优化（按需求接入）
```

这里的两个 Analysis 是由 MLIR `AnalysisManager` 缓存的 C++ 分析对象，不是
两个必须依次插入 pass pipeline 的 transform pass。可以额外提供只用于测试和
调试输出的 analysis-print pass，但生产 pipeline 不依赖它。

本设计与 [PR #1280](https://github.com/hw-native-sys/PTOAS/pull/1280)
提出的统一 Typed AddressExpr 方向一致，并进一步明确了在 PTOAS 中的组件边界、
首期 consumer、缓存粒度和迁移方式。

PR #1280 中“统一”指所有 consumer 从同一份 typed address semantics 和同一套查询
规则获得结论，不要求所有求解器必须实现为一个不可拆分的 C++ class。本设计中的
`PTOValueEvolutionAnalysis` 是 AddressExpr 的可复用数学求解层，
`PTOAddressAnalysis` 是对外提供统一 AddressExpr/address query 的组合层；二者不产生
两套相互竞争的地址表示。PR #1280 的概念图把 recurrence、cast 和 `addptr` 都画在
统一 Typed AddressExpr 中；本设计在 C++ 实现上把整数/index 子表达式交给
ValueEvolution、把 pointer root/`addptr` 组合留在 AddressAnalysis，但最终查询仍基于
同一个 AddressExpr。这是缓存与职责拆分，不是语义模型差异。

## 2. 背景与当前问题

历史上的 `VPTOSoftPostUpdate` 在 pass 内部同时完成了以下工作：

- 用 `PostUpdateTable` 描述不同 VPTO memory op 的 base、offset 和 offset unit；
- 将整数、`index`、`pto.addptr` 和简单算术分解为 `StrideExpr`；
- 识别 `scf.for` IV 和 `iter_args` recurrence；
- 计算 range、cast 是否保持循环 delta，以及有限位宽 recurrence 是否回绕；
- 将 base delta 与 offset delta 换算到指令要求的 stride unit；
- 检查目标 post-update 指令的 immediate/类型约束并改写 IR。

其中只有最后一项是 Post-Update 特有逻辑。前五项描述的是值和地址本身的事实，
可以被其他优化复用。

当前 cast 处理暴露了这种耦合的问题：loop-varying integer-to-index cast 只有在
输入是规范的 i16 地址 recurrence，并且 pass 自己能够证明整个循环不回绕时才
容易被接受。i8、i32 或直接为 `index` 的等价地址表达式可能走不同的匹配路径，
即使数学上具有相同的 step，也不能稳定得到相同结论。

根因不是 Post-Update ABI 偏好 i16，而是分析把“consumer 最终允许什么类型”与
“源表达式在其真实类型语义下如何演化”混在了一起。固定 i16 normalization 只能
扩大某些模式匹配，不能作为通用正确性证明：

- 缩窄到 i16 需要先证明值域和 no-wrap；
- i32 地址不应仅因未规范化成 i16 而失去其已知 step；
- `index` 没有显式 cast，也仍然需要分析其表达式与循环演化；
- cast 前后 step 相同与 cast 在所有迭代都保持语义，是两个需要同时证明的问题。

因此本设计保留源 IR 的真实类型和 cast，在查询时给出证明；不通过预先改写 IR
来编码分析结果。

## 3. 目标与非目标

### 3.1 目标

1. 对 VPTO 低层 pointer/integer IR 提供统一、可缓存、只读的数学分析。
2. 用可组合的 typed value/address 模型处理 `iN`、`index`、cast、简单算术、
   `pto.addptr`、`scf.for` IV 和 loop-carried recurrence。
3. 对 range、cast safety、no-wrap、known step 和 effective address delta 给出
   明确的 `Known` 或带原因的 `Unknown`，而不是依赖固定 i16 形态。
4. 让 `VPTOSoftPostUpdate` 从 op interface 读取 post-update 指令约束，只负责应用
   constraint、收益判断和 IR 改写。
5. 让非地址 consumer（例如 loop counter narrowing）可以只复用值与循环演化
   分析，而不依赖 VPTO memory op。
6. 为 uniformity、lane stride、alignment 等未来明确的 consumer 查询保留同一
   TypedExpr/AddressExpr 扩展点，避免再建立第二套地址表示。

### 3.2 非目标

- 不新增强制运行的 address normalization transform pass。
- 不把分析结果物化成固定 i16/i32 IR。
- 不在地址分析中判断 alias、no-alias、disjointness、memory dependence 或
  memory effects。
- 不由公共分析决定某条指令是否值得或允许转换成 post-update。
- 首期不承诺支持任意非线性表达式、不规则 CFG loop 或未知边界 recurrence。
- 首期不具体实现没有明确 consumer 的 uniformity、lane stride、alignment
  传播规则；这些查询按 consumer 需求增加，但复用同一表达式模型。

## 4. 收益与 consumer

### 4.1 VPTOSoftPostUpdate

这是首期唯一要求接入的正式 consumer。收益不是笼统的“识别更多”，而是用一套
类型正确的证明等价替代现有 pass 内部的地址分析：

- `index` IV 直接作为 `vlds`/`vsts` offset 时，即使没有 cast，也通过同一
  TypedExpr 和 loop evolution 查询得到 step；
- i8、i16、i32 和 index recurrence 不再按固定宽度白名单区分，是否可用取决于
  真实范围、cast 语义和 no-wrap 证明；
- `index_cast`/`index_castui` 被保留为 typed 节点，避免把 signed/unsigned 和
  截断/扩展效果丢失；
- base 与 offset 可以分别演化，再按 op 的地址单位合成 effective delta；
- Post-Update 的 immediate 范围、`SignedI8`/`Constant` 等限制仍由 consumer
  最后检查，不污染通用事实。

### 4.2 Loop counter narrowing

`PTONarrowVPTOLoopCounters` 当前以局部规则判断常量边界是否适合 i16。未来可以直接
查询 `PTOValueEvolutionAnalysis` 的 range、cast preservation 和 no-wrap 结果，
等价替代其局部证明。它不需要构建 AddressExpr，也不需要依赖任何 memory op。

### 4.3 其他已知步长需求

其他需要循环已知步长的优化可以复用 `PTOValueEvolutionAnalysis`；需要 effective
address delta 时再复用 `PTOAddressAnalysis`。例如 PR #1260 中出现的连续地址/步长
需求可以作为未来复用场景，但不属于首期交付范围，也不反向扩大首期 API。参见
[PR #1260](https://github.com/hw-native-sys/PTOAS/pull/1260)。

## 5. 为什么是组合分析，而不是一个或三个 pass

### 5.1 两个可缓存分析对象

值域和 recurrence 放在一个 `PTOValueEvolutionAnalysis` 中，因为二者形成闭环：

- loop-carried range 需要 initial value、step 和 iteration domain；
- cast 是否保持 recurrence 需要输入范围和有限位宽 no-wrap；
- recurrence 的 backedge 是否安全又需要整个循环范围。

把二者注册为两个独立缓存分析会形成互相查询或重复求解。实现内部仍可拆成
`TypedExprBuilder`、`RangeEvaluator` 和 `LoopEvolutionEvaluator`，但它们共享缓存
和求解上下文。

地址分析单独作为 `PTOAddressAnalysis`，因为它多了一层 VPTO memory/address
语义：base、offset、pointer element size、offset unit 和 effective delta。非地址
consumer 无需支付或依赖这一层。

### 5.2 不是 pipeline 中的三个 pass

如果把 range、recurrence、address 分别做成 transform/analysis pass 并顺序运行，
就必须处理结果如何跨 pass 持久化、前一 pass 修改 IR 后如何失效，以及后续 pass
是否重复计算。`AnalysisManager` 已经提供按 IR operation 缓存和失效的机制，因此
生产 consumer 应直接查询 C++ analysis object。

可选的 `-pto-print-address-analysis` 仅用于 lit 测试和人工调试。它不拥有分析逻辑，
也不作为其他 pass 的前置条件。

## 6. Op 级地址语义

### 6.1 VPTOAddressSemanticsOpInterface

首期直接新增 `VPTOAddressSemanticsOpInterface`，由具有可分析地址的 VPTO op 实现。
这是 op 自身的 IR 语义，适合与 ODS 定义放在一起维护。它同时是 current access
与 post-update advance 的唯一事实来源；transform 不再按 op name 维护第二张表。

概念接口如下，最终 C++ 名称可在实现阶段遵循现有 ODS interface 风格调整：

```cpp
enum class VPTOAddressUnit {
  Element,
  Block,
  Byte,
  Alignment,
};

struct VPTOAddressOffset {
  OpOperand *operand;
  VPTOAddressUnit unit;
  Value elementTypeSource;
};

struct VPTOAddressAccess {
  OpOperand *baseOperand;
  std::optional<VPTOAddressOffset> offset; // none 表示当前 access 为 base + 0
};

struct VPTOPostUpdateSemantics {
  OpOperand *baseOperand;
  OpOperand *advanceOperand; // 可选 trailing operand 不存在时为 null
  VPTOAddressUnit advanceUnit;
  VPTOAdvanceConstraint constraint;
  Value updatedBase; // normal form 中为 null
  Value elementTypeSource;
};

struct VPTOAddressSemantics {
  SmallVector<VPTOAddressAccess> currentAccesses;
  std::optional<VPTOPostUpdateSemantics> postUpdate;
};

VPTOAddressSemantics getVPTOAddressSemantics();
```

接口返回 ODS 具名 accessor 对应的 `OpOperand *`/`Value`，而不是把物理 operand/result
下标暴露给 consumer，原因是：

- analysis 不应依赖可选 operand 导致的物理下标变化；
- 一个 op 将来可以描述多个实际 memory access；
- op 可以将特殊 assembly/operand 形式规范化为统一语义。

`currentAccesses` 只描述“当前这次访存访问哪个地址”；`postUpdate` 独立描述访问后
如何更新 pointer。该访问是 read 还是 write 仍属于现有 memory-effect/scheduling
语义，不在本接口重复维护。只实现 current access 而不支持 post-update 的 op 显式
返回 `postUpdate = none`。

normal form 例如：

```text
vlds/vsts:  {base, {offset, Element}}
vsstb:      {base, {repeat_stride, Block}}
vldus:      {source, none}
vstus:      {base, none}
```

已经带 `updatedBase` 的 post-update form 则统一将 current access 描述为
`{base, none}`；同一个 offset 只出现在 `postUpdate.advanceOperand` 中。SoftPostUpdate
会先把首次访问的完整地址物化为该 base，再把相邻访问间 stride 作为 advance，因此
不能把 advance 再叠加到当前访问，否则会产生一拍偏移。

对于 stateful op，作为状态输入的 align 或本次访问之后的 advance 不能误报为当前
access offset。例如 `vstus` 的 current access 是 `base + 0`，而 `offset` 只出现在
`postUpdate.advanceOperand` 中。current unit 与 advance unit 是两个独立字段，不要求
相同；不同 unit 可能正是指令有意定义的语义。没有 current offset 时，不为它臆造
unit。

对于 `Element` unit，`elementTypeSource` 进一步指定单位宽度来自哪个 SSA value。
普通 load/store 使用 base pointer；`vlds`、`vldsx2`、`vldus` 使用结果 payload，
`vstus` 使用待写入 value。这样当 base 是 `ptr<ui16>`、payload 是 `vreg<...xui8>`
时，一个 Element 明确表示 1 byte，而不会被误算为 2 bytes。

### 6.2 地址单位

unit 到 byte 的换算由公共 helper 根据 access 与目标信息计算：

| Unit | 一个单位的字节数 |
|------|------------------|
| `Element` | `elementTypeSource` 的 element bytes |
| `Block` | 32 bytes |
| `Byte` | 1 byte |
| `Alignment` | 由 op mode 和目标查询得到，例如现有 `getLoadStoreVecAlignmentSize` |

这里的 `Alignment` 是指令 offset 的计量单位，不代表 analysis 已经证明了地址对齐。
“known alignment” 是未来可在同一 AddressExpr 上增加的独立查询。

VPTO 的逻辑 current-access offset 使用 `index` 类型，并按上述 unit 参与地址计算。
目标 intrinsic 的 i32 offset 参数只是 ABI/编码限制，不构成第二套 VPTO 地址语义。
lowering 只有在完整的 byte offset 可证明能以 i32 表示时，才能直接使用该参数；否则
必须以完整 pointer-width index 调整 base，并向 intrinsic 传递可表示的 residual offset。
post-update 形式同样必须从完整 index 计算 updated base。任何路径都不得通过静默截断
高位来改变 VPTO IR 表示的地址，因而 normal 与 post-update lowering 对同一逻辑地址
保持一致。该规则同样适用于 `plds/pldi/psts/psti`：`plds/psts` 的 index 以 byte
计量，`pldi/psti` 的 index 以目标 alignment 计量；当逻辑 index 无法由 intrinsic
的 i32 参数表示时，lowering 必须按对应 unit 的完整字节数调整 base，并传入零
residual offset。

### 6.3 与现有接口的关系

现有 `VPTOSchedulingOpInterface` 描述 scheduler 需要的 operation-local 语义；
`VPTOMemoryAccess` 主要描述读写、address space 和可选静态 byte range。它们不提供
动态 offset SSA、offset unit 或 recurrence，因此不能替代本接口。

`VPTOAddressSemanticsOpInterface` 也不应重复 read/write、memory effects 或 alias
信息。调度、依赖或 alias consumer 可以组合各自分析与 AddressExpr，而不应把这些
结论写回地址接口。

## 7. PTOValueEvolutionAnalysis

### 7.1 分析粒度和职责

`PTOValueEvolutionAnalysis` 以 `func::FuncOp` 为分析单位，建立函数内整数和 index
值的 typed expression，并按需、memoized 地回答：

```cpp
class PTOValueEvolutionAnalysis {
public:
  const TypedExpr &getExpr(Value value);
  RangeResult getRange(Value value, const IterationDomain &domain = {});
  ProofResult preservesValue(Operation *cast,
                             const IterationDomain &domain = {});
  EvolutionResult getEvolution(Value value, scf::ForOp loop);
  AnalysisResult<TypedExpr> getPointExpression(const TypedExpr &expression);
  ProofResult doesNotWrap(Value value, scf::ForOp loop);
};
```

接口名称是设计示意，核心要求是所有结果保留类型语义并允许返回 `Unknown(reason)`。

### 7.2 TypedExpr

首期表达式至少覆盖：

```text
TypedExpr := Constant(APInt, Type)
           | Opaque(Value, Type)
           | Add(TypedExpr, TypedExpr, Type)
           | Sub(TypedExpr, TypedExpr, Type)
           | MulByConstant(TypedExpr, APInt, Type)
           | Cast(CastKind, TypedExpr, SrcType, DstType)
```

`Opaque` 是保守边界，不是错误。无法继续分解的 SSA value 仍可作为 loop-invariant
symbol 参与仿射 step，例如 `%base + %iv * %invariant_stride`。

每个由 SSA `Value` 构建的 Add/Sub/Mul/Cast 节点都保留其 backing value。对这种
source-backed 节点，expression evolution 必须等同于
`getEvolution(Value, loop)`：复用同一套 range、cast 和 no-wrap transfer，失败后
不得再展开叶子得到更强结论。没有 backing value 的 synthetic 节点只组合已经证明
安全的子 evolution；如果实际有限位宽语义无法证明，则保守返回 Unknown。

表达式节点必须记录真实 source/result type。不能先把所有整数提升为无限精度整数，
再假定算术恒不回绕；有限位宽语义是 range、cast 和 recurrence 证明的一部分。

### 7.3 首期 cast 范围

| Cast | TypedExpr 表示 | 首期证明/传播 |
|------|----------------|---------------|
| `arith.index_cast` | 是 | 是 |
| `arith.index_castui` | 是 | 是 |
| `arith.trunci` | 预留 | 返回 Unsupported/Unknown |
| `arith.extsi` | 预留 | 返回 Unsupported/Unknown |
| `arith.extui` | 预留 | 返回 Unsupported/Unknown |

首期只实现当前实际出现的 `index_cast` 和 `index_castui`，但 `CastKind` 不应设计成
只能表达这两个 op。后续增加其他整数 cast 时，不需要改变 AddressExpr 结构。

### 7.4 Range 与 cast preservation

range 使用 APInt/ConstantRange 等能表达有限位宽 signed/unsigned 语义的表示。至少
需要回答：

- 某个值在指定 loop domain 内的 signed 或 unsigned 范围；
- 该范围是否可由目标整数类型精确表示；
- cast 是否在所有相关迭代上保持原值；
- 某个算术或 recurrence 是否在原类型中不回绕。

分析不必把 `exact value`、任意乘法和 alignment/divisibility 都作为首期独立功能。
首期只实现 Post-Update 正确性所需的 constant、range、cast preservation、简单线性
算术和 no-wrap；以后由明确 consumer 增加查询。

### 7.5 Loop evolution

首期处理结构化 `scf.for` 的两类简单演化：

1. IV：由 lower bound、upper bound 和 step 推导 iteration domain 与每次迭代 delta；
2. iter_arg recurrence：yield 可表示为 `iterArg + invariantStep`，或等价的简单线性式。

概念结果如下：

```cpp
struct EvolutionResult {
  TypedExpr initial;
  TypedExpr step;
  IterationDomain domain;
  ProofResult noWrap;
};
```

no-wrap 必须覆盖循环实际执行的所有值以及最后一次 backedge 更新。只检查 body 中
最后一次被访存使用的值、忽略 yield 更新，可能错误接受最后一次更新发生回绕的
recurrence。

动态 trip count 只有在现有事实足以给出保守范围时才返回 Known；否则返回 Unknown，
由 consumer 放弃依赖该证明的优化。

## 8. PTOAddressAnalysis

### 8.1 AddressExpr

`PTOAddressAnalysis` 以 `func::FuncOp` 为单位，依赖同一函数的
`PTOValueEvolutionAnalysis`，并从 `VPTOAddressSemanticsOpInterface` 构建：

```cpp
struct AddressExpr {
  Value rootOrBase;
  TypedExpr elementOffset; // 沿 pto.addptr 累积，单位为 pointer element
  std::optional<TypedAddressOffset> offset;
  int64_t elementBytes;
};

struct TypedAddressOffset {
  TypedExpr value;
  VPTOAddressUnit unit;
  std::optional<int64_t> unitBytes;
};
```

AddressAnalysis 负责沿 `pto.addptr` 把 pointer 表达式拆成
`rootOrBase + elementOffset`，并识别 `iter_args` 中由 `pto.addptr` 形成的 pointer
recurrence；其中所有整数 offset、step、cast 和 no-wrap 证明都委托给
ValueEvolution。无法继续剥离 pointer 时，当前 pointer SSA value 作为新的
`rootOrBase`，`elementOffset` 为零。这样 ValueEvolution 不需要理解 VPTO pointer
op，非地址 consumer 也不会依赖它们。

概念查询接口如下：

```cpp
class PTOAddressAnalysis {
public:
  AnalysisResult<SmallVector<AddressExpr>> getAddresses(Operation *op);
  AnalysisResult<TypedExpr> getDeltaBytes(const AddressExpr &address,
                                          scf::ForOp loop);
  AnalysisResult<TypedExpr> getDifferenceBytes(const AddressExpr &from,
                                               const AddressExpr &to);
  AnalysisResult<TypedExpr> convertDeltaToUnit(const TypedExpr &deltaBytes,
                                               int64_t targetUnitBytes);
};
```

`getDeltaBytes` 是通用地址查询；`convertDeltaToUnit` 的 target unit 由 consumer
声明。对 Post-Update 而言，它可能来自 Post-Update advance 的编码单位，而不一定
来自 current access offset。

`rootOrBase` 是地址 provenance/base identity 的载体：首期至少保留 root pointer
SSA；如果 IR 或其他语义已经提供 allocation identity，也可以附着在同一 base
描述上。具有相同 root 时，可以在这个共同符号下比较 offset；但该信息本身不产生
以下结论：

- 两个不同 root 必然 no-alias；
- 两个范围不相交；
- 两个 memory op 无依赖。

这些结论属于独立 alias/dependence analysis。未来它们可以消费 AddressExpr 提供的
base symbol、offset 和 range，但不是本分析的职责。

### 8.2 Effective address delta

对于某个 loop，地址分析分别查询 pointer element offset 与 op offset 的 evolution，
再统一换算为 byte：

```text
deltaBytes = elementBytes * delta(elementOffset / pointer recurrence)
           + unitBytes    * delta(offset)
```

如果 current access 没有 offset，第二项就是零，不需要臆造 `unitBytes`。

Post-Update consumer 若需要以原指令 offset unit 表示 stride，则进一步查询：

```text
strideInUnit = deltaBytes / unitBytes
             = (elementBytes / unitBytes) * delta(pointer)
             + delta(offset)
```

除法必须可精确证明；不能通过截断或向下取整产生 stride。分析返回符号 TypedExpr，
不在查询期间创建 `arith` 或 `pto.addptr` op。

`getDifferenceBytes` 是两个地址在同一点的差值查询，没有 loop domain。它可以消去
相同的 SSA point value，也可以通过 ValueEvolution 的 point-expression 查询展开由
原始 SSA operation 的 no-wrap flag 证明安全的 Add/Sub/Mul；没有匹配 flag 时，再由
operand point range 证明结果是否落在类型范围内。对完整输入域保值的 widening cast
也可以展开。两种证明都没有的 source-backed 节点仍保持 opaque，不能借用不存在的
loop range 做重关联。精确结果为零时，两个公共 delta 查询都返回 Known(0)，包括单位
换算后的零。是否接受 zero stride 属于 consumer 策略；SoftPostUpdate 的 loop 与
sequential 路径分别显式拒绝零步长。

对于 offset 直接为 `index` IV 的情况，`delta(offset)` 直接来自 loop evolution；没有
cast 并不会绕过分析。对于 `i32 -> index`，分析先根据 i32 recurrence 的范围和
no-wrap 证明 cast 是否保值，再传播相同 delta；并不要求先变成 i16。

### 8.3 查询结果

公共查询采用可区分原因的保守结果：

```cpp
template <typename T>
struct AnalysisResult {
  std::optional<T> value;
  UnknownReason reason;
};
```

典型 `UnknownReason` 包括：

- unsupported op/cast；
- unknown iteration domain；
- range unavailable；
- possible signed/unsigned wrap；
- non-affine recurrence；
- unknown element/unit size；
- inexact unit conversion；
- different/unrelated base symbols。

这样 Post-Update 的 debug remark 或 analysis-print test 可以区分“指令不支持”与
“数学证明不足”，也便于后续逐项扩展，而不是用一个 `failure()` 隐藏原因。

## 9. AnalysisManager 集成与失效

两个 analysis 都注册在 `func::FuncOp` 层级：

```cpp
class PTOValueEvolutionAnalysis {
public:
  PTOValueEvolutionAnalysis(func::FuncOp func, AnalysisManager &am);
};

class PTOAddressAnalysis {
public:
  PTOAddressAnalysis(func::FuncOp func, AnalysisManager &am)
      : valueEvolution(am.getAnalysis<PTOValueEvolutionAnalysis>()) {}
};
```

函数级 consumer 使用 `getAnalysis<T>()`。当前 `VPTOSoftPostUpdate` 是 ModuleOp pass，
应对每个待处理函数使用 `getChildAnalysis<PTOAddressAnalysis>(func)`。

transform consumer 必须遵循以下顺序：

1. 查询分析；
2. 生成不修改 IR 的 rewrite plan；
3. 完成所有合法性检查；
4. 统一物化并改写 IR；
5. 不声明 preserve 已经因改写失效的 ValueEvolution/AddressAnalysis。

同一次改写中不要在 mutation 之后继续读取旧 analysis result。如果后续阶段仍需分析，
应让 AnalysisManager 在新的 IR 上重新计算。

## 10. VPTOSoftPostUpdate 迁移

迁移后，现有职责按下表归属：

| 现有逻辑 | 新归属 |
|----------|--------|
| current base/offset/unit | `VPTOAddressSemanticsOpInterface::currentAccesses` |
| post-update base/advance/unit/constraint/result | `VPTOAddressSemanticsOpInterface::postUpdate` |
| `StrideExpr`、`decomposeLinear` | `PTOValueEvolutionAnalysis` 的 TypedExpr/线性分解 |
| `getIterArgIncrement`、IV delta | `PTOValueEvolutionAnalysis::getEvolution` |
| `canonicalAddressRecurrenceDoesNotWrap` | range/no-wrap evaluator |
| `castPreservesLoopDelta` | typed cast preservation + evolution propagation |
| `computeDelta` | ValueEvolution/AddressAnalysis 查询 |
| `combineStride`、element/unit 换算 | `getDeltaBytes` + `convertDeltaToUnit` |
| normal/post-update op 构造 | 消费 interface 的 typed operand/result contract |
| profitability、pointer chain/rewrite plan | `VPTOSoftPostUpdate` |

`PostUpdateOpInfo`、`PostUpdateTable`、裸 `advanceOperandIdx` 与按结果数量判断
post-update form 的规则全部删除。`Constant`、`SignedI8`、`Dynamic` constraint 也由
op interface 返回。SoftPostUpdate 只保留通用的 legality、profitability、pointer
chain 与 rewrite plan；构造 normal/post-update 形式时直接替换 interface 返回的
typed operand，并从具名 updated-base accessor 取得结果。

## 11. 扩展查询边界

统一 TypedExpr/AddressExpr 的意义是 future consumer 可以在同一模型上增加查询，
不是首期预先实现所有分析：

| 查询 | 与本设计关系 |
|------|--------------|
| uniformity | 可在 TypedExpr 的 SSA leaf/loop/lane 语义上增加 evaluator |
| lane stride / contiguity | 可查询 AddressExpr 随 lane id 的 delta |
| known alignment | 可由 root alignment、offset range/divisibility 组合 |
| divisibility | 可作为 TypedExpr 的附加数学事实 |
| alias/disjointness | 外部 analysis 消费 root + offset/range，不放入本分析 |
| memory dependence | 外部 analysis 组合 memory effects、alias 和控制流 |

在没有明确 consumer 之前，公共 API 可以暂不暴露对应方法，或返回 Unknown；但新增
实现时必须扩展现有模型，不能另建与 AddressExpr 并行且语义冲突的地址系统。

## 12. 实施阶段

### 阶段一：公共语义与分析骨架

- 新增 `VPTOAddressSemanticsOpInterface` 和 AddressUnit/access 数据结构；
- 为首期 Post-Update 涉及的 VPTO memory op 实现接口；
- 新增 `PTOValueEvolutionAnalysis`、TypedExpr、range/no-wrap/evolution 查询；
- 新增组合式 `PTOAddressAnalysis` 和 effective delta/unit 查询；
- 提供可选 analysis-print pass 或等价测试入口。

### 阶段二：迁移 VPTOSoftPostUpdate

- 先让新旧分析对当前已支持 case 产生一致结果；
- 将 cast、recurrence、delta 和 unit conversion 切换到公共查询；
- 增加 i8/i32/index、`index_cast`/`index_castui`、wrap 边界和 unit conversion 回归；
- 删除 pass 内被完全替代的重复分析 helper；
- 保留 Post-Update 特有的 legality、profitability 和 rewrite。

### 阶段三：按 consumer 扩展

- loop narrowing 按需接入 ValueEvolution；
- 其他 known-step/address-delta consumer 按需接入；
- uniformity、lane stride、alignment/divisibility 只有在 consumer 给出明确查询契约后
  扩展 evaluator。

## 13. 验证要求

分析测试至少覆盖：

1. `index` IV 直接作为 `vlds`/`vsts` offset；
2. i8、i16、i32 recurrence 在各自类型内不回绕与可能回绕的边界；
3. signed `index_cast` 与 unsigned `index_castui` 的不同范围解释；
4. cast 在部分迭代保值、但在最后一次 backedge 更新溢出的反例；
5. base delta、offset delta 及二者同时变化；
6. Element、Block、Byte、Alignment 单位的精确和非精确换算；
7. opaque loop-invariant leaf 与 mul-by-constant；
8. 不支持的 cast、非线性 recurrence、动态未知 domain 返回带原因的 Unknown；
9. stateful op 的 post-access advance 不被错误解释成 current access offset；
10. Post-Update 迁移前后的既有合法 case 等价，新增可证明 cast case 得到正确 stride。

分析查询必须是只读的。测试还应确认被拒查询不产生临时 IR，transform 在改写后不复用
失效结果。

## 14. 设计原则总结

- **保留类型，而不是先归一化类型。** i16/i32/index 是表达式语义的一部分，不是
  地址分析的准入条件。
- **证明数学事实，而不是编码 consumer 决策。** range、no-wrap、step 和 delta 是
  公共事实；post-update immediate 限制是 consumer 约束。
- **组合分析，而不是复制 helper。** 非地址 pass 复用 ValueEvolution，地址 pass
  在其上组合 unit 和 pointer 语义。
- **op 提供语义，analysis 提供推导。** ODS OpInterface 是 base/offset/unit 的单一
  事实源，analysis 不维护易漂移的 op-name descriptor table。
- **Unknown 是合法结果。** 无法证明时保守放弃，不通过 normalization 或默认步长
  猜测结论。
- **地址事实不等于 alias 结论。** root/base symbol 用于表达和 delta 比较，alias、
  disjointness 和 dependence 由外部分析负责。
