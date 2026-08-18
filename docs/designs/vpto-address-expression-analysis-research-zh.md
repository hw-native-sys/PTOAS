# VPTO 地址表达式与整数 Cast 分析调研

## 1. 背景

地址表达式可能包含 `arith.trunci`、`arith.extsi`、`arith.extui` 和
`arith.index_cast`。其中窄化 cast 具有有限位宽回绕语义：

```text
ext(trunc(x)) != x
```

只有在能够证明 `x` 落入目标窄类型的 signed 或 unsigned 值域时，才能把上述
cast 链视为保值变换。因此，地址分析不能默认穿透或删除 cast。

本调研关注两个问题：

1. Triton 和 XLA 如何在地址表达式包含 cast 时继续工作；
2. 这些做法对 VPTO 通用地址分析能力的设计有什么启发。

本文只记录调研结论和设计原则，不预设必须采用固定 i16 normalizer，也不把
现有 post-update 实现视为通用地址分析的最终结构。

## 2. 核心区分

正确生成地址和精确理解地址表达式是两个不同问题。

编译器可以原样保留不能证明的 cast，并按其有限位宽语义生成代码。分析失败只应
减少依赖相关证明的优化机会，不应影响编译正确性。

地址优化依赖的信息可分为两类。

### 2.1 相对地址结构

这类信息通常不要求知道完整绝对地址：

- lane 间地址差和 contiguity；
- stride；
- uniform 与 non-uniform 部分；
- alignment 和 divisibility；
- tile shape 和 allocation identity。

例如：

```text
addr[lane] = base + opaque_runtime_offset + lane
```

即使 `opaque_runtime_offset` 来自无法穿透的 cast，仍可证明相邻 lane 连续。

### 2.2 精确数值关系

以下变换需要 range 或 no-wrap 证明：

- 证明 cast 前后的值相等；
- 消除 `ext(trunc(x))`；
- 在 cast 两侧重组算术表达式；
- 将 pointer offset 收窄到更小位宽；
- 证明固定步长 recurrence 在目标位宽内不回绕；
- 依赖精确 offset 范围的 alias、post-update 或目标指令选择。

不能因为第二类证明失败而丢弃第一类信息。

## 3. Triton 的处理方式

Triton AMD 的 pointer canonicalizer 将 pointer 表示为 `{base, offset}`。它可以分解
常量、`add`、`mul`、broadcast、reshape 等已知结构；遇到不能理解的 cast 时，不把
cast 当作不存在，而是把整个 cast 子表达式保留为 opaque leaf。

例如：

```text
offset = uniform + cast(x) + lane
```

分析仍可能得到：

```text
uniform part     = uniform
non-uniform part = cast(x) + lane
```

或者由其他属性分析继续得知 `lane` 的连续性。它不能据此声称 `cast(x) == x`，但仍
可以进行不依赖该等价关系的 pointer canonicalization、coalescing、vectorization、
layout 和 scheduling。

不同 offset 位宽需要合并时，Triton 显式生成 `extsi` 或 `trunci`。只有
`tt.pointer_range=32`、small tensor 或保守的 `canNarrow` 条件成立时，才允许最终
窄化 offset。

Triton 还专门避免错误地分解有限位宽表达式。原始整体表达式可能不回绕，但将其拆成
uniform 和 non-uniform 部分后，各部分可能独立回绕，再经 signed extension 得到不同
地址。small-tensor 路径因此保留原始 base，先重新组合完整 offset，最后整体窄化。

需要精确数值范围时，Triton 使用 IntegerRangeAnalysis。无法得到范围时，相关窄化
或表达式重组应当退化，而不是假设 cast 保值。

参考：

- [Triton AMD CanonicalizePointers](https://github.com/triton-lang/triton/blob/main/third_party/amd/lib/TritonAMDGPUTransforms/CanonicalizePointers.cpp)
- [Triton AxisInfoAnalysis](https://github.com/triton-lang/triton/blob/main/lib/Analysis/AxisInfo.cpp)
- [Triton AMD IntegerRangeAnalysis](https://github.com/triton-lang/triton/blob/main/third_party/amd/include/Analysis/RangeAnalysis.h)

## 4. XLA 的处理方式

XLA Indexing Analysis 主要根据 HLO op 语义构造 tensor element 之间的 indexing
relation，而不是从低层 pointer arithmetic 中恢复完整地址表达式。

reshape、transpose、broadcast 和静态 slice 等索引主要由编译器根据 shape 和 layout
生成，变量范围也来自 tensor shape。因此，常见静态地址不要求分析任意用户提供的
cast 链。

动态 offset 通常作为 cast 之后的 runtime variable 建模。例如：

```text
x -> convert -> dynamic-slice(input, offset)
```

对应的 indexing relation 可以写成：

```text
input_index = output_index + rt0
rt0 = runtime_value(convert(x))
```

IndexingMap 不假设 `rt0 == x`，也通常不把 `convert(x)` 的生产者表达式展开进
`SymbolicExpr`。cast 的回绕如果发生，已经包含在 `rt0` 的实际运行时值中。代码生成
仍执行原始 convert；分析只使用 cast 后的值。

对于 dynamic-slice，最终有效 offset 还受 op 自身的合法区间语义约束。XLA 当前只
会把少数 runtime value（如 constant 和 iota）进一步内联成符号表达式；复杂生产者
通常保持 runtime symbol。因此它以损失 producer 数值关系为代价，保留正确且可组合
的索引关系。

`convert` 对 Indexing Analysis 可以是 identity map，只表示输出位置 `d` 来自输入
位置 `d`，并不表示 convert 前后的数值相等。

参考：

- [XLA Indexing Analysis](https://github.com/openxla/xla/blob/main/docs/indexing.md)
- [XLA indexing_analysis.cc](https://github.com/openxla/xla/blob/main/xla/hlo/analysis/indexing_analysis.cc)
- [XLA SymbolicExpr](https://github.com/openxla/xla/blob/main/xla/hlo/analysis/symbolic_expr.h)

## 5. 为什么保守处理仍可实现高性能

高性能优化不是一个全有或全无的结论。cast 只应阻断依赖精确数值等价的变换。

例如：

```text
addr[lane] = base + cast(runtime_offset) + lane
```

即使无法证明 cast 保值，仍可知道 cast 后的值对所有 lane 相同，并证明 lane 间地址
差为 1。由此仍可进行连续访存、向量化、tile/layout 选择和 pipeline。

如果 cast 包含 lane-varying 表达式：

```text
addr[lane] = base + sext(trunc(x + lane))
```

且无法证明活动 lane 范围内不回绕，则 cast 可能真的破坏连续性。此时拒绝依赖连续性
的变换是必要的正确性约束，不只是分析能力不足。

Triton 和 XLA 都不要求输入没有 cast。常见输入由前端或编译器生成，地址结构通常
规整；冗余 cast 可以由 canonicalization 消除，剩余 cast 则作为精确分析边界保留。

## 6. 对 VPTO 地址分析设计的启发

### 6.1 优先统一分析表示，而非强制统一 IR 位宽

通用地址能力首先应提供统一的分析结果和查询接口，而不必先把所有原始 IR 永久改写
成 i16。

建议的分析表示至少能表达：

```text
AddressExpr
  base
  uniform offset
  varying offset / lane stride
  bit width and signedness
  value range
  alignment and divisibility
  loop recurrence
  proven no-wrap facts
```

cast 节点必须保留其 signedness、源位宽和目标位宽。无法证明保值时，cast 后的结果
可以成为新的 opaque leaf，但其外围结构仍可继续分析。

### 6.2 分离属性传播和精确证明

分析应分别回答：

- 哪些相对地址属性可以跨 cast 保留；
- cast 是否保值；
- source 和 target 域是否不回绕；
- 某个 consumer 是否具备所需的全部证明。

不能用“无法证明 cast 保值”替代“整个地址表达式不可分析”。

### 6.3 由 consumer 声明所需事实

不同优化可以查询不同事实，但这些事实必须从同一个地址表达式和同一个语义分析器
派生，不能为每个优化维护一套独立的地址表示：

| Consumer | 需要的事实 |
|----------|------------|
| 连续访存和向量化 | lane stride、contiguity、alignment |
| pointer canonicalization | base、uniform/non-uniform offset |
| alias/disjointness | allocation identity、offset range |
| post-update | 精确 recurrence、固定 delta、必要的 cast/no-wrap 证明 |
| i16 指令选择 | 最终 offset 或 delta 可在目标 i16 语义中表达 |
| loop-counter narrowing | loop control 自身满足目标位宽语义 |

一个 consumer 的精确证明失败，不应阻止其他只依赖相对结构的优化。

上表描述的是查询接口，不是不同的分析系统。

### 6.4 统一地址表达式模型

地址分析的核心对象应统一表示地址 provenance、带类型语义的 offset 表达式和迭代
域：

```text
AddressExpr {
  provenance: root pointer / allocation
  offset: TypedExpr
  address unit: element / byte / block
  iteration domain: loop IV, iter_arg, lane and constraints
}
```

`TypedExpr` 保留表达式的语义节点，而不是针对某个优化提前物化结果：

```text
constant
add / sub
mul-by-constant
signed cast / unsigned cast
loop recurrence
addptr
```

循环递推、cast 和有限位宽运算都属于同一表达式语义。分析器在此模型上提供统一的
抽象解释和证明查询：

```text
range(expr)
uniformity(expr)
lane_stride(expr)
alignment(expr)
affine_or_recurrence(expr)
no_wrap(expr)
equivalent(expr1, expr2)
delta(expr, iteration -> iteration + 1)
```

`delta` 不是 post-update 专用字段，而是任意地址表达式上的通用查询。post-update
只查询最终地址的 delta 是否恒定且能够编码；向量化查询 lane stride；窄化查询目标
位宽下的等价性。所有查询共享同一套 cast、位宽、signedness 和循环域语义。

这比按优化分别维护“post-update 地址信息”“窄化信息”“访存连续性信息”更统一，
也避免不同 pass 对同一个 cast 采用不同的数值解释。

### 6.5 把 i16 视为待确认的目标约束

需要区分：

1. 逻辑地址值在语言或 IR 语义上就是 16 位模地址；
2. 只有某个硬件寻址模式、delta 字段或指令选择要求 i16。

若属于第二种情况，更适合保留逻辑地址的原始位宽，由通用分析证明范围，再在需要 i16
的 consumer 或 lowering 处选择窄表示。否则，全局 i16 改写可能需要为大量宽类型
operand 恢复 cast，并把分析、证明、loop 重建和用户修复耦合在一个 pass 中。

### 6.6 从 DSL/MLIR 到 post-update 的具体调用链

以下用一个简单的 DSL 访存循环说明各分析组件之间的职责。DSL：

```python
def kernel(src):
    offset = 0
    for _ in range(4):
        value = pto.vlds(src[offset:])
        offset += 32
```

其关键 IR 结构可以抽象为：

```mlir
scf.for %i = %c0 to %c4 step %c1
    iter_args(%offset = %c0) -> (index) {
  %value = pto.vlds %src[%offset]
  %next = arith.addi %offset, %c32 : index
  scf.yield %next : index
}
```

post-update consumer 不应自行解析上述结构，而是依次调用共享分析接口：

```text
SoftPostUpdatePass
  -> LoopContext::findEnclosingLoop(load)
  -> AddressExprBuilder::build(load, loop)
  -> RecurrenceAnalysis::analyze(iter_arg, loop)
  -> RangeAnalysis::query(recurrence, domain)
  -> EffectiveDeltaAnalysis::compute(address, recurrence, domain)
  -> PostUpdateLegality::check(load, address, delta)
  -> PostUpdateRewriter::apply(load, plan)
```

各接口的输入和输出如下。

1. `LoopContext::findEnclosingLoop(load)` 找到外层 `scf.for`，返回 induction variable、
   lower/upper/step 和可推导的 trip count。示例中为 `k in [0, 4)`。

2. `AddressExprBuilder::build(load, loop)` 从 load 的 source 和 offset 反向追踪 SSA
   定义，返回统一的 `AddressExpr`：

   ```text
   provenance = src
   offset = LoopRecurrenceRef(%offset)
   unit = element
   domain = loop(k)
   ```

   `addptr`、`add`、`mul` 和 lane 偏移会继续作为 `TypedExpr` 节点保留。窄化或扩展
   cast 也保留 source/target 位宽和 signedness，不能在此阶段假设 cast 保值。

3. `RecurrenceAnalysis::analyze(iter_arg, loop)` 读取 `iter_args` 初值和 `scf.yield`
   表达式。示例中得到：

   ```text
   R(0) = 0
   R(k + 1) = R(k) + 32
   ```

   返回值包含 initial value、typed transition expression 和 loop domain。若 yield
   经过未知调用或无法解释的运算，则 transition 为 `Unknown`。

4. `RangeAnalysis::query(recurrence, domain)` 在实际迭代域上计算值域和回绕事实。示例
   中 `R(k)` 的范围为 `[0, 96]`，并可证明每次 backedge update 不回绕。这个证明服务于
   后续变换，不是 DSL IR 本身的合法性要求。

5. `EffectiveDeltaAnalysis::compute(address, recurrence, domain)` 计算最终有效地址的
   差分，而不是直接读取源 `addi` 的常量：

   ```text
   A(k) = base(src) + R(k)
   delta(k) = A(k + 1) - A(k) = 32
   ```

   返回 `Constant(32)`、单位和证明状态。若地址包含 `lane` 偏移，lane 项在相邻 loop
   迭代差分中消去。

6. `PostUpdateLegality::check` 检查 provenance 是否一致、delta 是否恒定、单位和位宽
   是否可被目标 post-update 操作编码，以及相关 cast/no-wrap 证明是否齐全。通过后形成
   `PostUpdatePlan { initial_pointer, increment = 32, unit }`。

7. `PostUpdateRewriter::apply` 将 offset recurrence 和普通访存改写为 pointer
   accumulator 及带 increment 的 `pto.vldus`（或对应 store）形式。改写使用的是地址
   分析得到的 `effective_delta`，而不是未经验证的源 recurrence step。

若 offset 是 `i16` recurrence 并经过 `index_cast`，`EffectiveDeltaAnalysis` 必须计算：

```text
R(k + 1) = trunc_i16(R(k) + 32)
A(k) = base + sext_i16_to_index(R(k))
delta(k) = A(k + 1) - A(k)
```

例如初值 `32736` 时，`R(1) = -32768`，delta 不再是恒定 `32`，接口返回
`Unknown(CastChangesEffectiveDelta)`，post-update 放弃，原始 IR 保持不变。

### 6.7 与现有 post-update 地址分析的关系

这个设计不是完全重写一套无关的分析，也不是把现有函数原样搬到公共目录。更准确的
定位是：复用现有 post-update 中已经验证过的分析内核，同时重新定义分析对象和调用边界。

当前 `VPTOSoftPostUpdate.cpp` 已经包含以下可复用逻辑：

| 当前实现 | 可迁移到统一框架的职责 |
|----------|------------------------|
| `decomposeLinear` | 递归解析 `addi/subi`、常量乘法、`index_cast` 和 `addptr` |
| `getIterArgIncrement` | 从 `iter_args`/`scf.yield` 提取 loop recurrence 的 transition |
| `getConstantTripCount` | 构造 loop iteration domain |
| `canonicalAddressRecurrenceDoesNotWrap` | 在给定 domain 上计算 range 和 no-wrap 事实 |
| `castPreservesLoopDelta` | 判断 cast 是否保持 loop delta，作为 cast 语义分析的一个特例 |
| `combineStride` | 合并 base/offset delta，并完成 element、byte、block 单位转换 |
| `createInitialPtr` | 根据分析结果物化 post-update 的初始地址 |

因此，recurrence 提取、范围计算、回绕检查和 stride 单位换算并不是新框架凭空引入的
能力，现有 pass 已经有一个面向自身需求的实现基础。

但现有分析的抽象边界是 post-update 专用的。`decomposeLinear` 的结果是：

```text
blockArg * coeff + increment
```

它围绕某个 loop block argument 分解一个候选值；`getIterArgIncrement` 也只在追踪当前
post-update 候选时提取增量。这样的结果足以支持当前 fixed-stride 改写，但不是通用的
地址表达式表示。

统一框架需要把它提升为：

```text
AddressExpr {
  provenance
  typed offset expression
  address unit
  iteration domain
}
```

现有逻辑与新设计的主要差异如下：

1. **分析对象不同。** 当前 pass 从一个 post-update 候选出发，围绕一个 block argument
   做线性分解；统一框架从任意 pointer/load/store 的 SSA operand 构造地址表达式，多个
   consumer 共享同一个表达式。

2. **cast 的处理边界不同。** 当前 `castPreservesLoopDelta` 失败后通常使这条候选路径
   失败，并且 loop-varying integer-to-index cast 主要接受 canonical i16、直接常量步长
   且已证明不回绕的情况。统一框架保留 cast 节点，并分别提供 cast 保值、cast 后 range、
   cast 后 delta 等查询；一个查询失败不必让所有地址属性都丢失。

3. **i16 的角色不同。** 当前 post-update 将 canonical i16 作为可消费的地址域，其他宽度
   通常交给前置 normalization。统一框架不把 i16 作为地址分析前提，而是按原始位宽和
   signedness 解释 `i8/i16/i32/index`，在 consumer 需要时再证明目标位宽或目标指令格式
   可以表达该结果。

4. **delta 的定义不同。** 当前实现从线性分解和各 operand stride 合并出 post-update
   stride。统一框架把 `delta(address, iteration -> iteration + 1)` 定义为任意地址表达式
   的通用查询，post-update 只是查询最终有效地址 delta 是否恒定；lane stride、alignment
   和 alias analysis 使用同一个地址表达式但查询不同事实。

5. **单位转换的归属不同。** 当前 `combineStride` 将 element/byte/block 换算直接耦合
   在 post-update。统一框架仍可复用这套换算规则，但应把 address unit 作为表达式语义的
   一部分，使其他 consumer 也能复用，而不是复制一份 stride 计算。

因此，建议的迁移方式是：先把当前 `decomposeLinear`、recurrence 提取、trip-count/range
和单位换算中的可验证逻辑抽成共享组件；再将其返回类型从 `LinearDecomp` 和
`StrideExpr` 扩展为 typed `AddressExpr` 及其查询结果；最后让 post-update 调用
`effective_delta` 查询。现有 post-update 的改写和盈利性判断可以继续保留在 consumer 层。

这是一种“实现内核渐进抽取、语义模型重新设计”的方案，而不是简单的代码重命名，也不
要求一次性重写当前 pass。

## 7. 建议的推进顺序

1. 定义统一的 Typed AddressExpr、迭代域和 cast/有限位宽语义。
2. 建立共享的 AddressExprAnalysis，首期支持 constant、add/sub、
   mul-by-constant、cast 和结构化 loop recurrence。
3. 在同一个语义模型上实现 lane/uniform/alignment、range/no-wrap、recurrence 和
   effective-delta 查询。
4. 让各 consumer 只调用共享查询，并在证明不足时独立退化。
5. 在统一分析接口稳定后，再评估是否需要 IR normalizer，以及它应物化哪些已经证明安全
   的 canonical form。

最终原则是：

> 统一地址分析的表示和查询接口，使优化能够在 cast 存在时继续工作；只有确实依赖
> 精确数值等价的变换，才要求 range 或 no-wrap 证明。
