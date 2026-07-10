# PTODSL TileOp Subkernel Redesign Explainer

本文解释本次 PR 为什么要重做 tile-level
subkernel 的 public surface，以及 PTODSL tracing、PTO IR、PTOAS 如何配合
完成这次修改。

更精确的最终 contract 见
`docs/designs/ptodsl-redesign-of-simd-simt-cube-subkernel.md`。本文侧重背景铺垫
和方案理解。

## 1. 背景：PTODSL 在 PTOAS 中的位置

PTOAS 的底层输入是 PTO IR。PTO IR 描述数据搬运、tile 计算、同步、内存规划
和代码生成所需的信息。PTODSL 是 PTOAS 的 Python 前端：用户写 Python 函数，
PTODSL tracing 把函数体记录成 PTO IR，然后交给 PTOAS 继续做 layout、memory、
sync、EmitC 或 VPTO lowering。

一个典型流程是：

```text
Python @pto.jit function
  -> PTODSL tracing
  -> PTO IR module
  -> PTOAS
  -> EmitC / VPTO output
  -> runtime or simulator validation
```

## 2. 问题定义：旧 API 把 helper 名字和计算风格绑在一起

issue 直接推动的是 PTODSL subkernel public surface 的收敛。修改前，PTODSL
同时暴露 `@pto.simd`、`@pto.cube`、inline `with pto.simd():`、
inline `with pto.cube():` 和 `@pto.simt`。

这里的问题不是“名字太多”本身，而是 `simd` / `cube` 把两件不该耦合的事情绑
在了一起：

- 用户入口（public surface）：用户要定义的是“一个 tile-level helper”，它的
  参数和返回值边界是什么、能不能含 load/compute/store 多个阶段。
- 主计算风格：helper body 里真正的主计算是 Vector 还是 Cube。

这两件事本应分开。一个 tile-level helper 的自然边界是 `Tile` / `TensorView`
/ `PartitionTensorView` / PTO scalar；它的 body 可以是 vector-style，也可以是
cube-style，还可能同时包含 MTE load/store、Scalar 控制和 sync（softmax 就是
典型例子：MTE load → Vector compute → MTE store）。

旧 API 要求用户先用 `@pto.simd` / `@pto.cube` 选主计算风格，于是 API 名字、
参数契约、inline 路径和 PTOAS 后续分析都跟着 `simd` / `cube` 分叉成两套。但
“是不是 vector 主计算”本该由 PTOAS 看 body 自己推导出来，而不是让用户提前
用 decorator 名字声明。

因此本次设计只保留两个用户入口：

```text
tile-level custom helper -> @pto.tileop
launched SIMT helper      -> @pto.simt
```

`@pto.simt` 不合并进 `@pto.tileop`，因为 SIMT helper 面向 launched SIMT 代码，
允许 `pto.ptr(...)` 这类 raw pointer 边界；tileop helper 则是 tile/view/scalar
层面的 helper。旧 `simd/cube` surface 不再作为兼容别名继续使用，而是诊断并提示
迁移到 `@pto.tileop`。

## 3. 一个 tile-level helper 要表达什么

subkernel 是从主 kernel 里抽出来的可复用 helper。它通常不是独立 launched
kernel，而是一个会被主 kernel 调用、之后再由 PTOAS inline 或拆分处理的函数。

以 row-wise softmax 风格的 helper 为例，用户想表达的是：

```python
@pto.tileop
def row_softmax(src: pto.TensorView,
                dst: pto.TensorView,
                scratch: pto.Tile,
                rows: pto.i32,
                cols: pto.i32):
    pto.tile.load(src, scratch)
    # vector compute on scratch
    pto.tile.store(scratch, dst)

@pto.jit
def kernel(src, dst):
    row_softmax(src, dst, scratch, rows, cols)
```

这个 helper 的 public ABI 是 `src/dst/scratch/rows/cols` 这些跨调用边界可见的
值；helper body 里则有实际执行过程：

```text
MTE load -> Vector compute -> MTE store
```


## 4. 方案总览

第 2 节已经说明用户入口的变化。这里从相同层级比较修改前后的职责分工，只关注
tile-level helper；`@pto.simt` 在修改前后都保留独立入口，不在图中重复展示。

修改前，用户必须先选择 `simd` 或 `cube`。PTODSL tracing 再把这个选择记录成
分类标记，并按该分类生成对应的 section 形态：

```mermaid
flowchart LR
  User["用户定义<br/>tile-level helper"]
  Surface["选择 @pto.simd<br/>或 @pto.cube"]
  Trace["PTODSL tracing<br/>记录 simd/cube 分类标记<br/>生成对应 section 形态"]
  IR["PTO IR"]
  PTOAS["PTOAS<br/>消费前端给出的分类"]
  Output["EmitC / VPTO output"]

  User --> Surface --> Trace --> IR --> PTOAS --> Output
```

新方案在相同层级上的流程如下：

```mermaid
flowchart LR
  User["用户定义<br/>tile-level helper"]
  Surface["统一使用 @pto.tileop"]
  Trace["PTODSL tracing<br/>记录 body 和<br/>pto.tileop.helper"]
  IR["PTO IR"]
  PTOAS["PTOAS<br/>从 body 推导分类、阶段和读写<br/>再生成对应 section"]
  Output["EmitC / VPTO output"]

  User --> Surface --> Trace --> IR --> PTOAS --> Output
```

核心变化是分类职责发生了转移：修改前由用户选择 `simd/cube`，PTODSL tracing
把这个选择直接写进 IR；修改后用户统一写 `@pto.tileop`，PTODSL tracing 只记录
helper body，PTOAS 再从 body 推导主计算域、阶段和读写关系。

因此新方案简化的是用户入口和 PTODSL tracing 路径。PTOAS 内部新增的分析和校验会在第 7 节展开。

## 5. Public contract

`@pto.tileop` 的核心含义是：它是 tile/view/scalar 层面的 helper，不是 raw
pointer helper，也不是 launched SIMT kernel。

### 5.1 参数能传什么

`@pto.tileop` 参数只允许表达 tile-level 边界的数据：

- `Tile`：片上 tile buffer。
- `TensorView`：tensor 视图，常用作 load/store 的逻辑输入或输出边界。
- `PartitionTensorView`：已经切分好的 tensor 分区。
- PTO scalar：例如 rows、cols、scale、flag、loop bound。

不允许把这些值作为 `@pto.tileop` 参数：

- `pto.ptr(...)`：raw pointer 只给 `@pto.simt` 使用。
- host tensor / `TensorSpec`：这是 host/kernel entry 层面的整 tensor 或规格描述，
  不是 tileop helper 的边界。
- vector register、mask、pipe handle 等 helper 内部临时值或控制值。

### 5.2 结果怎么传出来

tile/view 结果通过参数传入的可写 tile/view 表达，而不是作为 Python 函数返回值
返回。

例如用户应写成“把结果写进 `out_tile` / `out_view`”：

```python
@pto.tileop
def helper(src: pto.TensorView, out: pto.Tile):
    ...
    pto.tile.store(..., out)
```

不要把 tile/view 当作函数 result 返回：

```python
@pto.tileop
def helper(src: pto.TensorView) -> pto.Tile:  # not allowed
    ...
```

当前版本只允许 scalar result。也就是说，helper 可以返回类似状态值或小标量，
但主要数据输出仍然应该写入传进来的 tile/view。

### 5.3 helper body 的边界

`@pto.tileop` helper body 可以包含数据搬运、主计算、标量控制和同步，但有几个
边界：

- 当前 tileop contract 明确禁止 helper 在函数内部自己申请新的 tile buffer。
  需要的 scratch/output tile 应由 caller 创建后作为参数传进来。实现层面会拒绝
  `alloc_tile`、`reserve_buffer`、`pto.talloc` 这类 helper-local tile allocation。
  这个限制只针对 tileop helper；普通 `@pto.jit` kernel 内仍可以使用
  `pto.alloc_tile(...)`。
- helper 必须有一个主计算域：Vector 或 Cube。MTE load/store、Scalar、sync 可以
  作为辅助阶段存在，但它们不决定主计算域。
- 一个 helper 里不要同时混用 Vector 主计算和 Cube 主计算。当前实现要求一个
  tileop helper 只表达一个主计算域。
- tileop helper 不调用另一个 tileop helper。这样每个 helper 的读写摘要都是
  局部、清楚、可验证的；递归组合多个 helper summary 留给后续设计。

## 6. 新 IR 形态

PTODSL tracing 时不再提前写 `primary_domain`、`phases`、`operand_effects`，
也不预套 `pto.section.vector/cube`。PTODSL tracing 只把 helper 标成
canonical marker：

```mlir
// PTODSL tracing 刚完成后的 helper 形态。
// attributes 里只有 pto.tileop.helper，没有 primary_domain/phases/effects。
func.func private @softmax(%src: !pto.tensor_view<...>,
                           %dst: !pto.tensor_view<...>,
                           %scratch: !pto.tile_buf<...>,
                           %rows: i32,
                           %cols: i32)
    attributes {pto.tileop.helper} {
  // body 里先保留 tracing 得到的原始 op 顺序。
  // 这里还没有 pto.section.vector 或 pto.section.cube。
  pto.tload ... outs(%scratch) : ...     // MTE load
  pto.vadd ... outs(%scratch) : ...      // Vector compute
  pto.tstore ... ins(%scratch) : ...     // MTE store
  return
}
```


PTOAS 后续再基于 helper body 推导这些属性：

```mlir
attributes {
  pto.tileop.helper,
  pto.tileop.primary_domain = #pto.kernel_kind<vector>,
  pto.tileop.phases = [...],
  pto.tileop.operand_effects = [...]
}
```

## 7. lib/PTO 详细设计

前面的章节说明了 public surface 和 PTODSL tracing 后的 IR 形态。本节继续说明
`lib/PTO` 中负责摘要推导、section 物化、contract 校验、同步建模和 VPTO split
的具体修改。

本节涉及的 pass / 流程状态如下：

| pass / 流程 | 状态 | 这次承担的职责 |
|---|---|---|
| helper marker 识别 | 修改 | 新增 canonical `pto.tileop.helper` marker，同时兼容旧 `pto.ptodsl.subkernel_helper = "tileop"` |
| `PTOInferTileOpSummaryPass` | 新增 | 从 helper body 推导 `primary_domain`、`phases`、`operand_effects` |
| `PTOMaterializeTileOpSectionsPass` | 新增 | 根据 summary 把主计算 span 包成 `pto.section.vector/cube` |
| `PTOVerifyTileOpContractPass` | 新增 | 校验 tileop ABI、body 约束和 summary 是否 stale |
| InsertSync 的 `PTOIRTranslator` | 修改 | helper call 不再按单个粗粒度节点建模，而是按 `pto.tileop.phases` 建同步节点 |
| `VPTOSplitCVModule` | 修改 | 在已单域 VPTO module 里也展开同域 section、删除异域 section，避免 section 留到后续 lowering |

### 7.1 Pipeline 位置

修改前，从同一个 PTOAS pipeline 视角看，tile-level helper 更依赖 PTODSL tracing
已经写好的角色标记和 section 形态。PTOAS 不会先从 helper body 推导
`primary_domain`、`phases` 和边界 effect，后续同步建模也更接近按一个 helper
call 整体处理：

```mermaid
flowchart LR
  IR["PTO IR<br/>func.func + 前端角色标记"]

  subgraph PTOAS["PTOAS"]
    Role["消费前端角色标记<br/>simd / cube"]
    Section["使用前端预套<br/>vector/cube section"]
    Sync["InsertSync PTOIRTranslator<br/>按整个 helper call 建模"]
    Inline["helper inline<br/>后续 PTOAS passes"]
    Split["VPTOSplitCVModule<br/>处理 vector/cube section"]
  end

  Lower["EmitC / VPTO lowering"]

  IR --> Role --> Section --> Sync --> Inline --> Split --> Lower
```

PTODSL tracing 的输出是 PTO IR。带 `pto.tileop.helper` marker 的 `func.func` 进入
PTOAS 后，关键数据流如下：

```mermaid
flowchart LR
  IR["PTO IR<br/>func.func + pto.tileop.helper"]

  subgraph PTOAS["PTOAS"]
    Infer["PTOInferTileOpSummaryPass<br/>推导 primary_domain / phases / effects"]
    Materialize["PTOMaterializeTileOpSectionsPass<br/>包主计算 span"]
    Verify["PTOVerifyTileOpContractPass<br/>校验 ABI / body / summary"]
    Sync["InsertSync PTOIRTranslator<br/>按 phase 建模 helper call"]
    Inline["helper inline<br/>后续 PTOAS passes"]
    Split["VPTOSplitCVModule<br/>处理 vector/cube section"]
  end

  Lower["EmitC / VPTO lowering"]

  IR --> Infer --> Materialize --> Verify --> Sync --> Inline --> Split --> Lower
```

### 7.2 如何标记一个 tileop helper

PTODSL tracing 生成 PTO IR 时，需要告诉 PTOAS：“这个函数不是普通函数，而是一个
tileop helper”。做法是在函数上加一个没有额外取值的标签：

```mlir
attributes {pto.tileop.helper}
```

看到这个标签后，后续 PTOAS pass 才会对该函数执行 tileop 专用的摘要推导、
section 物化和 contract 校验。普通函数没有这个标签，也就不会进入这些处理。

修改前，PTODSL 使用一个字符串属性同时表示不同 helper 类型：

```mlir
pto.ptodsl.subkernel_helper = "simd"
pto.ptodsl.subkernel_helper = "cube"
pto.ptodsl.subkernel_helper = "tileop"
```

新生成的 IR 统一使用专门的 `pto.tileop.helper` 标签，不再把 tileop 塞进旧的
`subkernel_helper` 字符串。为了让已有 IR 仍能通过 PTOAS，代码暂时也能识别旧的
`pto.ptodsl.subkernel_helper = "tileop"` 写法。

这两种形式的统一识别封装在 `include/PTO/IR/PTO.h` 中。后续 pass 只需要询问
“这个函数是不是 tileop helper”，不需要分别处理新旧属性格式。

### 7.3 Summary 属性设计

`PTOInferTileOpSummaryPass` 在 tileop helper 函数上生成三类 PTOAS-owned 属性。
下面是示意形态，省略了完整 MLIR attribute assembly 的细节：

```mlir
attributes {
  pto.tileop.helper,
  pto.tileop.primary_domain = #pto.kernel_kind<vector>,
  pto.tileop.phases = [
    {
      pipe = #pto.pipe<MTE2>,
      operand_uses = [0],
      operand_defs = [2],
      result_defs = []
    },
    {
      pipe = #pto.pipe<V>,
      operand_uses = [2],
      operand_defs = [2],
      result_defs = []
    },
    {
      pipe = #pto.pipe<MTE3>,
      operand_uses = [2],
      operand_defs = [1],
      result_defs = []
    }
  ],
  pto.tileop.operand_effects = ["read", "write", "readwrite"]
}
```

这个例子可以对应：

```text
operand #0: src_view
operand #1: dst_view
operand #2: scratch_tile
```

每个属性的含义是：

| 属性 | 粒度 | 用途 |
|---|---|---|
| `pto.tileop.primary_domain` | 整个 helper | 记录主计算域是 `vector` 还是 `cube`，用于 section materialization 和 summary 校验 |
| `pto.tileop.phases` | helper body 内的 pipe phase | 记录每个 phase 的 pipe，以及这个 phase 读写了哪些 helper operand；InsertSync 用它把一个 helper call 拆成多个同步建模节点 |
| `pto.tileop.operand_effects` | helper operand | 记录整个 helper 对每个 operand 的合并 effect，用于 contract 校验和 stale-summary 检查 |

`result_defs` 当前保持为空，因为当前设计里 tile/view 输出不通过 function result
返回；function result 只允许 scalar。

### 7.4 PTOInferTileOpSummaryPass

`PTOInferTileOpSummaryPass` 是 summary-only pass。它只写属性，不改 helper body，
也不插 section。

它的算法可以概括成：

```text
按源代码顺序递归遍历 helper body 里的每个 op：
  pipe = classify(op)                          # 判断 op 属于哪个 pipe
  if pipe 是 Vector 或 Cube 主计算：
      设置或校验 primary_domain
  if pipe 和上一个 body op 不同：
      开始一个新的 phase
  if op 带有 MemoryEffect：
      把 op 实际读写的值追溯回 helper 参数
      在当前 phase 记录这个参数被读(use)还是被写(def)
      合并得到整个 helper 对该参数的总体 read/write effect
```

pipe 分类分两层：

- 优先通过 op 自己声明的 `getPipe()` 获取它所属执行管线，例如 MTE、Vector、
  Cube。`getPipe()` 来自 op 实现的 `OpPipeInterface`。
- 对部分没有明确 pipe interface 的 PTO op，用名字做 fallback 判断。例如
  `pto.v*` 识别为 Vector pipe，`pto.mad*` 识别为 Cube pipe，MTE load/store
  识别为对应 MTE pipe，标量 load/store 或 predicate op 识别为 Scalar pipe。

主计算域只由 Vector/Cube primary compute 决定：

- `PIPE_V` / `PIPE_V2` -> `primary_domain = vector`
- `PIPE_M` -> `primary_domain = cube`
- MTE、Scalar、sync 不决定 `primary_domain`

如果同一个 helper 里同时出现 Vector primary compute 和 Cube primary compute，
summary inference 直接报错。这对应当前“单主计算域”的实现边界。

边界 effect 的关键点是：op 实际读写的值可能是 helper 参数派生出来的内部临时值，
例如 tile slice、address value、cast 后的值或 loop iter_arg。PTOAS 不能只记录
这些内部临时值，因为 helper call 的外部边界只看得到函数参数。

例如：

```python
@pto.tileop
def helper(tile: pto.Tile, row: pto.i32):
    x = tile[row, 0:]
    v = pto.vlds(x)
    pto.vsts(v, tile[row, 0:])
```

在 IR 里，`tile[row, 0:]` 可能会变成 `memref.subview`。`vlds/vsts` 的 memory
effect 挂在这个 subview 上，但 summary 需要继续追溯，知道这个 subview 来自
helper 参数 `tile`。最终 phase 摘要记录的是“读/写了参数 `tile`”，而不是
“读/写了内部临时值 subview”。

当前实现会把 subview、cast、reshape、tile/view address、loop iter_arg 等视为
可透明追溯的中间值。追溯回 helper 参数后，只有 `Tile`、`TensorView`、
`PartitionTensorView` 这类 tileop 边界值会进入 tileop 的公开 effect 摘要。
`ptr` / `memref` 只在内部分析里用于兼容低层 IR 形态；它们不会因此变成 tileop
的合法 public ABI，verifier 仍会拒绝 `ptr` / `memref` 作为 tileop 参数。

### 7.5 PTOMaterializeTileOpSectionsPass

`PTOMaterializeTileOpSectionsPass` 消费 summary，把 helper body 中的主计算段包进
`pto.section.vector` 或 `pto.section.cube`。

它不是简单把整个 helper body 包进 section。原因是 helper 可能长这样：

```text
MTE load
Scalar loop setup
Vector compute
MTE store
```

正确的 IR 形态应该是：

```text
MTE load
Scalar loop setup
pto.section.vector {
  Vector compute
}
MTE store
```

而不是：

```text
pto.section.vector {
  MTE load
  Scalar loop setup
  Vector compute
  MTE store
}
```

materialize pass 的具体规则是：

1. 只处理标记为 tileop helper 的函数。
2. 如果 helper 里已经有 `pto.section.vector/cube`，不重复 materialize。
3. 读取 `pto.tileop.primary_domain` 和 `pto.tileop.phases`，确认存在 primary
   phase。
4. 递归进入普通控制流 region，例如 `scf.for` / `scf.if`。
5. 在每个 block 内收集 PTO body op，找到与 `primary_domain` 匹配的 primary
   compute span。
6. 当前实现要求这个 primary compute span 是连续的。如果写成
   `Vector compute A -> Scalar/MTE/sync -> Vector compute B`，materialize pass
   会报错。
7. 对 vector span，向前扩展包含产生 `mask` / `vreg` 这类 vector-scope local 值的
   producer，避免这些局部值被留在 section 外。
8. 把最终 span 包进 `pto.section.vector` 或 `pto.section.cube`，MTE/S/sync 留在
   section 外。

控制流中的主计算也需要被找到。这里的 helper entry block，指函数 body 最外层的
代码块。用户把计算写在循环里时，最外层直接包含的是 `scf.for`，真正的 MTE 和
Vector op 则在 `scf.for` 的循环体里：

```mlir
scf.for %i = %c0 to %rows step %c1 {
  pto.tload ...       // MTE load
  pto.tadds ...       // Vector compute
  pto.tstore ...      // MTE store
}
```

如果 pass 只检查函数最外层直接包含的 op，它只能看到一个 `scf.for`，看不到循环
体里的 `pto.tadds`，因而会错误地认为这个 helper 没有 Vector 主计算。

当前实现会把 `scf.for` 当作包含内部代码的控制流容器，继续进入它的循环体检查。
找到 `pto.tadds` 后，只在循环体内部给这段 Vector 主计算加 section：

```mlir
scf.for %i = %c0 to %rows step %c1 {
  pto.tload ...       // 仍在 section 外
  pto.section.vector {
    pto.tadds ...
  }
  pto.tstore ...      // 仍在 section 外
}
```

因此这里的“递归”只是指：从函数最外层进入 `scf.for` / `scf.if` 等控制流的内部
代码块，继续寻找主计算段。pass 不会因为主计算写在循环或分支里，就把整个循环或
分支都包进 Vector/Cube section。

### 7.6 PTOVerifyTileOpContractPass

`PTOVerifyTileOpContractPass` 是防线：它不负责推导新信息，而是确认 helper body 和
summary 没有违反 tileop contract。

它检查的内容包括：

- 参数类型只能是 `TileBufType`、`TensorViewType`、`PartitionTensorViewType` 或
  PTO scalar。
- function result 只能是 scalar。
- helper 必须至少包含一个 Vector 或 Cube primary compute op。
- 一个 helper 不能混用 Vector primary compute 和 Cube primary compute。
- helper 内不能出现 SIMT-only op，例如 thread id、launch、vote、shuffle、
  thread fence 等。
- helper 内不能分配 callee-local tile buffer，例如 `pto.alloc_tile`、
  `pto.reserve_buffer`、`pto.talloc`。
- tileop helper 不能调用另一个 tileop helper。
- `pto.tileop.primary_domain`、`pto.tileop.phases`、
  `pto.tileop.operand_effects` 必须和重新扫描 body 得到的结果一致。

最后一条很重要：summary attr 是 PTOAS 后续 pass 的输入，如果 body 被改了但
summary 没更新，InsertSync 或 section materialization 看到的就会是 stale 信息。
verifier 通过重新推导一遍 summary 来发现这种不一致。

### 7.7 InsertSync 如何消费 tileop summary

InsertSync 的任务是根据不同 pipe 对同一份数据的读写关系插入同步。对于直接出现
在 kernel 里的 PTO op，PTOAS 可以从 op 本身知道它属于哪个 pipe、读取什么、写入
什么。但是看到一次 helper 调用时：

```mlir
func.call @row_softmax(%src, %dst, %scratch) : ...
```

这条 `func.call` 只列出了传给 helper 的三个值，看不出 helper 内部先做 MTE load、
再做 Vector compute、最后做 MTE store。因此，PTOAS 需要读取 `@row_softmax`
函数上的 `pto.tileop.phases`，用它补回调用语句中看不到的阶段和读写信息。

一次 tileop helper 调用按下面的步骤处理：

1. `PTOIRTranslator` 找到被调用的 helper，读取它的 `pto.tileop.phases`。
2. 将摘要里的 helper 参数编号，对应到这次调用实际传入的值。
3. 每个读写了 helper 边界参数的 phase，生成一个 InsertSync 建模节点。
4. `InsertSyncAnalysis` 比较这些节点的 pipe 和读写关系，判断哪里需要同步。

转换后，每个建模节点记录所属 pipe、读取的值和写入的值。实现中这个节点对应
`CompoundInstanceElement`，但它不是新的 PTO IR op，只是 InsertSync 分析期间
使用的一条阶段记录。

如果一个 phase 只操作 helper 内部的临时值，没有读写任何 helper 参数，PTOAS
不会为它建立调用边界节点，因为它不会与 helper 外部的操作产生数据依赖。

#### 7.7.1 和旧同步建模的对比

这次修改没有更换 InsertSync 判断同步依赖的算法，改变的是
`PTOIRTranslator` 如何向它描述一次 helper 调用。

旧 `simd/cube` helper：

```mermaid
flowchart TB
  OldCall["func.call @helper(%src, %dst, %scratch)"]
  OldRole["读取前端给出的<br/>simd / cube 分类"]
  OldOperands["收集调用时传入的<br/>tile / view 等值"]
  OldNode["把整个调用表示成 1 个节点<br/>所有边界值保守地视为既读又写"]
  OldHazard["InsertSyncAnalysis<br/>判断同步依赖"]

  OldCall --> OldRole --> OldOperands --> OldNode --> OldHazard
```

当前 `tileop` helper：

```mermaid
flowchart TB
  NewCall["func.call @helper(%src, %dst, %scratch)"]
  NewSummary["读取 helper 的<br/>pto.tileop.phases"]
  NewMap["把 helper 参数编号<br/>对应到这次调用实际传入的值"]
  NewNodes["按 phase 生成节点<br/>分别记录 pipe、读取值和写入值"]
  NewHazard["InsertSyncAnalysis<br/>判断同步依赖"]

  NewCall --> NewSummary --> NewMap --> NewNodes --> NewHazard
```

| 对比项 | 旧 `simd/cube` helper | 当前 `tileop` helper |
|---|---|---|
| 信息来源 | 前端给出的 `simd/cube` 分类和调用参数 | PTOAS 从 helper body 推导的 phase 摘要 |
| 建模粒度 | 整个 helper 调用是一个节点 | 一个有边界读写的 phase 是一个节点 |
| 读写关系 | 边界值保守地视为既读又写 | 分别记录每个 phase 实际读取和写入的参数 |
| pipe 信息 | 整个调用只有一个粗粒度分类 | 每个 phase 保留自己的 MTE、Vector 或 Cube pipe |

例如一个 helper body 是：

```text
MTE load(src -> scratch)
Vector compute(scratch)
MTE store(scratch -> dst)
```

旧方案把整个调用描述成一个节点，无法表达三个阶段各自属于哪个 pipe、读写哪个
参数。当前方案则把它表示成三条阶段记录：

```text
phase 0: pipe = MTE,    use src,     def scratch
phase 1: pipe = Vector, use scratch, def scratch
phase 2: pipe = MTE,    use scratch, def dst
```

当 kernel 中前后出现其他 PTO op 或 helper 调用时，InsertSync 可以用这些阶段记录
判断它们是否在不同 pipe 上读写了同一个值，再决定是否插入同步；不再需要把整个
helper 调用保守地当作一次不可拆分的读写。

### 7.8 VPTOSplitCVModule 的 section rewrite

section 的作用有明确的生命周期：计算域尚未确定时用于区分 Vector/Cube；计算域
确定后就应被移除，因为 section 容器本身不对应后端指令。

```mermaid
flowchart LR
  Input["包含 vector/cube section 的 module"]
  Kind{"已有 pto.kernel_kind?"}
  Split["按 section 拆成<br/>Vector / Cube module"]
  Rewrite["展开同域 section<br/>删除异域 section"]
  Ready["不再包含 section<br/>进入 VPTO LLVM lowering"]

  Input --> Kind
  Kind -- "否" --> Split --> Rewrite --> Ready
  Kind -- "是" --> Rewrite
```

`VPTOSplitCVModule` 的处理可以概括为：

```text
if module 已有 kernel_kind:
    展开同域 section，删除异域 section
else:
    根据 vector/cube section 拆分 module
    在每个新 module 中展开同域 section，删除异域 section
```

主干原有实现会在“已有 `kernel_kind`”时直接返回，导致 helper inline 后的 section
可能遗留到 LLVM lowering。本次修改补上该分支，并增加包含 `scf.for` 的回归测试。

### 7.9 设计边界

这套 `lib/PTO` 设计刻意把几件事分开：

- summary inference 只负责“看懂 helper”，不改 IR body。
- section materialization 只负责“把主计算 span 结构化”，不重新定义 ABI。
- contract verifier 负责“发现非法 helper 或 stale summary”，不修正用户代码。
- InsertSync 只消费 summary 建 dependency graph，不再重新进入 callee body 分析。
- VPTO split 只处理 section sugar 和 module kind，不参与 tileop ABI 判断。

这种拆分让每个 pass 的输入输出都比较明确：前一个 pass 产出的属性或 section，
是后一个 pass 的显式输入；如果中途被改坏，verifier 会尽早报错。

## 8. 相比旧方案新增或收紧的限制

旧 `@pto.simd` / `@pto.cube` 路径主要依赖 frontend role 和预套 section。新
`@pto.tileop` 路径把 helper 交给 PTOAS 推导和校验，因此也把一些边界变成了明确
诊断。先从用户写 PTODSL 时能直接感知到的限制看：

### 8.1 用户写 PTODSL 时会看到的限制

| 用户写法 | 旧方案形态 | 当前 tileop 形态 |
|---|---|---|
| `@pto.simd` / `@pto.cube` | 作为 public decorator 使用 | 作为旧接口诊断报错，tile-level helper 统一写 `@pto.tileop` |
| tileop 参数 | 旧 simd/cube 的参数边界不统一 | `@pto.tileop` 只允许 `Tile`、`TensorView`、`PartitionTensorView`、PTO scalar |
| `pto.ptr(...)` 参数 | 容易被误认为 tile-level helper 也能接收 | 仍然只允许 `@pto.simt`；`@pto.tileop` 不能用 raw pointer 作为 public boundary |
| tile/view 输出 | 容易被误写成 function result | 通过 output operand 传出；function result 只允许 scalar |
| helper 内申请 tile | 普通 kernel 里可以 `pto.alloc_tile(...)` | `@pto.tileop` helper 内不能新建 helper-local tile buffer；scratch/output tile 由 caller 传入 |
| helper body | 由 `simd` / `cube` 名字暗示主域 | body 必须真的包含 Vector 或 Cube primary compute，不能只有 MTE/Scalar/sync |
| 混合 Vector/Cube | 旧名字无法清楚表达混合语义 | 同一个 `@pto.tileop` helper 不能同时写 Vector primary compute 和 Cube primary compute |
| helper 互调 | 没有清晰的 phase summary 组合语义 | `@pto.tileop` helper 不能调用另一个 `@pto.tileop` helper |
| SIMT 操作 | 容易和 tile-level helper 混用 | thread id、launch、vote、shuffle、thread fence 等 SIMT-only op 只放在 `@pto.simt` |

### 8.2 PTOAS 内部额外收紧的检查

还有一些限制用户不一定会在 Python 写法里直接看到，但会影响手写 IR、其他
frontend，或后续 PTOAS pass 的输入：

| 检查项 | 旧方案形态 | 当前 tileop 形态 |
|---|---|---|
| 后端 helper ABI | frontend / backend 边界规则容易分叉 | verifier 在 PTO IR 层拒绝 `ptr` / `memref` 作为 tileop helper 参数 |
| summary 一致性 | 没有 `primary_domain/phases/effects` 这组属性 | verifier 会重新扫描 body，确认 summary 没有 stale |
| 主计算 span | 旧路径常把整个 helper body 预套进一个 section | 当前只 materialize 主计算 span；同一个 block 内 primary compute 中间不能夹非 primary pipe |
| sync 建模输入 | helper call 可被保守看成单个 read/write 节点 | InsertSync 按 `pto.tileop.phases` 建多个 phase 节点，summary 缺失或错误会被拒绝 |
| section rewrite | section 可能留到后续 lowering | 已单域 VPTO module 里也会展开同域 section、删除异域 section |

这些限制不是为了减少 tileop 能力，而是为了让当前版本的 ABI、summary、sync 和
section 语义先保持可验证。后续如果要放开某一项，需要同时定义它对 summary、
InsertSync 和 VPTO section rewrite 的影响。

## 9. 方案同时覆盖的问题和收益

前面第 7 节解释了 `lib/PTO` 如何落地这套设计，第 8 节列出了当前实现收紧的边界。
本节再总结这套实现同时带来的收益和覆盖面。它们不是 issue 一开始逐条提出的问题，
但统一到 `@pto.tileop` 后，这些相关边界可以一起收敛。

### 9.1 tile-level helper 只保留一套 public contract

旧 API 把 helper 名字直接命名成 `simd` 或 `cube`。这看起来直观：做 vector
计算就写 `simd`，做 cube 计算就写 `cube`。但对 PTOAS 来说，这两类 helper
共享同一个更重要的 contract：它们都是 tile-level helper，参数边界应该是
tile/view/scalar，调用边界需要有 read/write effect，body 里也都可能包含
MTE、主计算、Scalar/sync 等多个 phase。

统一成 `@pto.tileop` 后，vector-style 和 cube-style 不再拥有两套 public
boundary 规则、两套诊断文案、两套 inline/decorated 处理路径。区别保留在
`primary_domain = vector/cube`，由 PTOAS 从 body 推导。

### 9.2 phase summary 让跨 helper 调用的同步更清楚

这里要区分两件事：一是 helper body 里实际出现了哪些 pipe（helper 真实做了
什么），二是 PTOAS 在 helper 调用边界能看到的摘要粒度（PTOAS 处理连续 helper
调用时能拿到多少内部信息）。例如面对：

```text
call @row_softmax(...)
call @row_softmax(...)
```

这类连续 helper 调用，PTOAS 能看到多少内部读写信息，就取决于摘要粒度。

旧 `simd`/`cube` 路径会把整个 helper call 压成一个粗粒度节点，例如“这是一个
vector-style call”。这不表示 helper 里面真的只有 Vector pipe，而是表示 PTOAS
只能把这个 call 当成一个整体来建模。它看不到这个 call 内部其实有
`MTE load -> Vector compute -> MTE store` 三个阶段，也就无法分别知道：

- MTE load 阶段读哪个输入 view、写哪个 scratch tile。
- Vector compute 阶段读写哪个 tile。
- MTE store 阶段读哪个 tile、写哪个输出 view。

这就是“跨 call 建同步时丢掉阶段结构”的意思：不是 IR 里的操作消失了，而是
PTOAS 在两个 helper call 之间判断依赖和插同步时，只拿到一个大粒度 call 摘要，
拿不到每个 phase 的读写摘要。

新的 `pto.tileop.phases` 用来表达这些 phase-level 信息。注意这不表示 MTE
load/store 属于 Vector；它们仍是 MTE pipe，只是和 Vector 主计算共同构成一个
`primary_domain = vector` 的 tileop helper。

### 9.3 section 由 PTOAS 物化，避免 PTODSL tracing 过早判断

section 是 PTO IR 里的结构化容器，用来告诉 PTOAS “这一段是 vector 主计算”
或“这一段是 cube 主计算”：

```mlir
pto.section.vector {
  // vector primary compute
}
```

section 不是“整个 helper 的容器”。它只应该包主计算段。MTE load/store 是数据
搬运，Scalar/sync 是辅助控制或同步；这些 op 通常应该留在主计算 section 外面。
也就是说，一个典型 helper 更接近：

```text
MTE load
pto.section.vector {
  Vector compute
}
MTE store
```

新方案让 PTODSL tracing 只标记 `pto.tileop.helper`，不预套 section。PTOAS 先
推导 summary，再递归进入控制流 body，找到连续的主计算 span 后只包这段，
MTE/S/sync 保持在 section 外。这样 decorated helper 和 inline helper 也能走
同一条 contract。

### 9.4 tileop ABI 收敛到 Tile/View/Scalar，ptr 保留给 SIMT

ABI 在这里指 helper 的参数和返回值 contract：哪些值允许跨 helper 边界，哪些
值只能留在 helper 内部。对 tileop 来说，用户关心的是 tile-level 数据结构，而
不是 raw pointer：

- `Tile` 表示片上 tile buffer。
- `TensorView` 表示一个 tensor 视图，可作为 load/store 的逻辑边界。
- `PartitionTensorView` 表示已经切分好的 tensor 分区。
- PTO scalar 表示 rows、cols、scale、loop bound 等标量参数。

这些类型足够表达 tile-level helper 的自然边界。例如 softmax helper 可以读
一个 `TensorView`，使用一个 `Tile` 做 scratch，再写一个 output view。

`pto.ptr(...)` 不同。它是 raw pointer 风格的低层 ABI，适合 SIMT launched
helper。raw pointer 只表达一段地址，不携带 tile/view 的结构化边界语义。因此
tileop contract 收敛为：

```text
tileop: Tile / TensorView / PartitionTensorView / PTO scalar
simt:   pto.ptr(...) and SIMT-specific boundary
```

### 9.5 控制流和 tile slice 的建模更完整

PTODSL 用户写 Python `for`/`if` 时，PTODSL tracing 会把它们记录成设备侧控制流，
而不是在 Python 编译期展开。MLIR 里常见形式是 `scf.for` / `scf.if`。PTODSL
用户写 `tile[row, col:]` 这类切片时，IR 里常见形式是 `memref.subview` 或 PTO
自己的 view/tile address op。

tracing 到 MLIR 后，它们会变成类似这样的结构：

```mlir
scf.for %i = %c0 to %rows step %c1 {
  %slice = memref.subview %tile_addr[%i, 0] [1, 64] [1, 1]
  %v = pto.vlds %slice[%c0] : ...
  pto.vsts %v, %out_slice[%c0], %mask : ...
}
```

对不熟 MLIR 的读者，可以把它理解成：

- `scf.for` 是 IR 里的结构化 loop。
- `memref.subview` 是从一个 tile/view 中取出一段 slice，例如第 `i` 行。
- `vlds/vsts` 可能真正读写的是 subview 结果，而不是函数参数本身。

这套方案要求 PTOAS 递归查看控制流 body，能看到 loop/if 内的主计算 op；同时
识别由 helper 边界 operand 派生出的透明 view、tile address、cast、subview 等
来源，能把 tile slice 的 effect 归回对应 helper 参数。这样 PTOAS 才能为常见
row-wise/tile-slice helper 建出正确的跨 call 依赖。

## 10. 用户迁移方式

用户原来写：

```python
@pto.simd
def helper(...):
    ...

@pto.cube
def matmul_helper(...):
    ...
```

现在统一写：

```python
@pto.tileop
def helper(...):
    ...
```

如果是 launched SIMT helper，仍然写：

```python
@pto.simt
def helper(ptr: pto.ptr(pto.f32, "gm")):
    ...
```

判断标准是：

- 参数是 tile/view/scalar，body 是 tile-level load/compute/store：用
  `@pto.tileop`。
- 参数是 raw pointer，body 是 SIMT-style launched code：用 `@pto.simt`。

旧代码里如果看到 `@pto.simd` 或 `@pto.cube`，迁移时不要按名字一比一替换成
新的 Vector/Cube 专用 decorator，因为新设计没有这两个专用 decorator。统一改
成 `@pto.tileop`，再让 PTOAS 从 body 推导 vector-style 或 cube-style 主域。

## 11. 常见问题

### MTE load/store 属于 vector 还是 cube？

都不属于。MTE load/store 属于 MTE pipe，是数据搬运阶段。一个 helper 的
`primary_domain` 由主计算 phase 决定：

- `MTE load -> Vector compute -> MTE store` 的主域是 `vector`。
- `MTE load -> Cube compute -> MTE store` 的主域是 `cube`。

因此文档中说 “vector-style helper 包含 MTE phase” 时，不是在说 MTE 属于
Vector，而是在说同一个 tileop helper 的执行过程可能包含多个 pipe。

### 为什么 tileop 只能有一个 primary_domain？

这是当前实现边界。一个 helper 如果同时包含 Vector 主计算和 Cube 主计算，PTOAS
需要决定如何拆 section、如何拆 kernel module、如何在两个主域之间建同步。这些
规则比单主域复杂很多。当前 contract 要求一个 tileop helper 只表达一个主计算
域，MTE/S/sync 作为辅助 phase。

### 为什么 `pto.ptr(...)` 不能传给 tileop？

`pto.ptr(...)` 是低层 pointer ABI，适合 SIMT launched helper。tileop 的目标是
让 helper 站在 tile/view/scalar 层面建模，这样 PTOAS 能基于结构化信息做
memory、sync 和 section 处理。允许 raw pointer 进入 tileop ABI 会绕开这些
结构化边界，所以仍保留为 SIMT-only。

### 为什么不把所有 operand 都统一标成 readwrite？

统一 readwrite 最保守，通常不容易漏同步，但会丢掉 phase 级别的真实依赖，
可能引入不必要的同步，也让 PTOAS 无法判断哪些 phase 真的跨 helper 边界产生
effect。`pto.tileop.phases` 的目的就是让 helper call 的同步模型更接近真实
读写行为。

### 为什么要禁止 tileop helper 调用另一个 tileop helper？

如果 helper A 调 helper B，PTOAS 需要递归组合两个 helper 的 summary、phase 和
operand effect，还要处理 inline 后可能产生的跨域 section 嵌套。当前版本先
禁止 tileop helper 互调，保证每个 helper 的 summary 是局部可验证、可消费的。

## 12. 测试覆盖

本次修改对应的测试覆盖分几层：

- PTODSL Python tests：验证 public API、legacy diagnostic、inline/decorated
  tileop 行为。
- PTO lit tests：验证 helper ABI、summary、contract verifier、memory planning
  和 sync。
- VPTO lit tests：验证 section materialization、control-flow body、single-kind
  module section rewrite。
- DSL ST simulator：验证实际 PTODSL runtime/simulator 路径，例如
  `vmulscvt.py`、`predicate_pack.py`、`cube_matrix_pipeline.py`。

关键回归包括：

- tileop 允许 `Tile` / `TensorView` / `PartitionTensorView` / scalar。
- tileop 拒绝 `ptr`；SIMT 仍接受 `ptr`。legacy `simd/cube` public surface 会
  先被 PTODSL frontend 迁移诊断拒绝。
- tile slice 经 `memref.subview` 后仍能被 summary effect 识别。
- control-flow 内的 primary compute 能被 materialize 成 section。
- mask/vreg local producer 不会逃出 vector scope。
- 已单域 VPTO module 中的 section 会在 lowering 前被展开。

## 13. 当前仍需注意的点

第 8 节已经列出相比旧方案新增或收紧的限制。除此之外，使用当前实现时还要注意：

- function result 只允许 scalar；tile/view 输出仍通过 output operand 表达。
- 如果一个 operand 没有被任何 op 显式读写，它的 effect 默认记为 `read`。只有
  实现了 `MemoryEffectOpInterface`、且读写值能追溯回 helper 参数的 op，才会真正
  影响 phase summary 里的 use/def。

这些边界不是因为 PTOAS/VPTO 永远无法支持，而是为了让当前 public contract 和
sync/memory/codegen 行为先稳定下来。
