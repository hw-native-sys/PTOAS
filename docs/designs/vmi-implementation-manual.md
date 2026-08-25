# VMI 实现手册

本文配套 `docs/designs/vmi-introduction.md` 和当前 VMI lowering 设计，回答
“按什么顺序改哪些文件、每一步做到什么程度才算完成”。

本文不替代最终 ODS / C++ verifier / lit 测试。实现时如果发现本文和 ODS 或 verifier 冲突，以
更精确的 verifier 约束为准，并同步刷新本文。

## 0. 当前仓库约束

当前仓库只有一个 MLIR dialect：

```text
dialect name: pto
cpp namespace: ::mlir::pto
```

VPTO 低层 op/type 也在同一个 `pto` dialect 里，通过 `VPTOOps.td`、`VPTOTypeDefs.td` 等文件组织。
因此第一版 VMI 不新建独立 dialect，采用同一 dialect 下的嵌套 mnemonic：

```text
types:
  !pto.vmi.vreg<...>
  !pto.vmi.mask<...>

attrs:
  #pto.vmi.layout<...>

ops:
  pto.vmi.addf
  pto.vmi.subf
  pto.vmi.mulf
  pto.vmi.ensure_layout
```

落地方式是：`PTO_Dialect` 仍是唯一 dialect，VMI 只是 `pto` dialect 内的一组 type/attr/op。
如果后续要拆成真正独立的 `pto.vmi` dialect，必须先保证所有 pass、type converter、parser 测试
和公开文档同步迁移；第一版不要做这个拆分。

风险点：带点 mnemonic 例如 `vmi.vreg`、`vmi.addf` 必须在 Slice 0 先用 parser round-trip 测试
证明。如果 TableGen 的默认 type/attr parser 不接受该 spelling，就在 VMI type/attr 上实现
custom assembly format，而不是改公开 spelling。

## 1. 文件布局

新增文件：

```text
include/PTO/IR/VMIAttrs.td
include/PTO/IR/VMITypeDefs.td
include/PTO/IR/VMIOps.td
lib/PTO/IR/VMI.cpp
lib/PTO/Transforms/VMILayoutAssignment.cpp
lib/PTO/Transforms/VMIToVPTO.cpp
lib/PTO/Transforms/PTOValidateVMIIR.cpp
test/lit/vmi_new/
```

修改文件：

```text
include/PTO/IR/PTOAttrs.td
include/PTO/IR/PTOTypeDefs.td
include/PTO/IR/PTOOps.td
include/PTO/IR/CMakeLists.txt
lib/PTO/IR/CMakeLists.txt
include/PTO/Transforms/Passes.td
lib/PTO/Transforms/CMakeLists.txt
```

推荐 include 关系：

```text
PTOAttrs.td
  include "PTO/IR/VMIAttrs.td"

PTOTypeDefs.td
  include "PTO/IR/VMITypeDefs.td"

PTOOps.td
  include "PTO/IR/VMIOps.td"
```

放置顺序：

```text
VMIAttrs.td:
  include PTODialect.td, AttrTypeBase.td, EnumAttr.td
  must not include PTOAttrs.td

VMITypeDefs.td:
  include PTODialect.td and can rely on PTOAttrs.td having included VMIAttrs.td

VMIOps.td:
  include after PTO_Op is defined in PTOOps.td
  do not include VPTOOps.td from VMIOps.td
```

这样现有 `LLVM_TARGET_DEFINITIONS PTOOps.td` 的 TableGen 生成路径可以继续覆盖 VMI type、attr
和 op。只有当 TableGen 生成目标不能正确收集新增 td 时，才单独新增 `mlir_tablegen` 目标。

`lib/PTO/IR/VMI.cpp` 放 VMI type/attr/op verifier、parse/print helper 和公共 lane-map helper。
不要把 VMI verifier 塞进 `VPTO.cpp`。

Pass 注册要求：

```text
include/PTO/Transforms/Passes.td:
  add VMILayoutAssignment
  add VMIToVPTO
  add PTOValidateVMIIR

include/PTO/Transforms/Passes.h:
  add explicit create*Pass declarations if generated declarations are not enough

lib/PTO/Transforms/CMakeLists.txt:
  add the three new .cpp files to PTOTransforms
  keep DEPENDS PTOPassesIncGen and PTOOpsIncGen
  add missing MLIR dialect libraries only when a new source actually includes them
```

The VPTO backend always runs the VMI semantic pipeline. There is no separate
enable or disable switch. The pipeline is ordered around vecscope inference as
follows:

```text
pto-validate-vmi-ir
vmi-layout-assignment
canonicalize/cse
vmi-layout-fold
canonicalize/cse
vmi-layout-rematerialize
canonicalize/cse
vmi-layout-sink-materialization
canonicalize/cse
vmi-legalize-arith-select
pto-validate-vmi-layout-ir
vmi-to-vpto
SIMT unroll/SCCP/canonicalize/CSE
vpto pointer/wrapper normalization
pto-infer-vpto-vecscope
VMI LICM
canonicalize/CSE
```

The pipeline only applies when the effective backend is VPTO. EmitC does not
run it because the pipeline produces physical VPTO values and ops.

The VPTO/VMI user-facing entry also rejects public functions whose
signature contains `!pto.vmi.*`.
Internal/private VMI-typed functions are materialized at explicit boundary
helpers by baseline `vmi-layout-assignment` and physicalized by `vmi-to-vpto`.
A later optimization pass may specialize private signatures.  A public VMI ABI
requires an explicit materialization plan and must not be inferred from the
layout solver.

CLI coverage:

```text
vmi_ptoas_cli_pipeline.pto:
  --pto-backend=vpto always lowers the VMI pipeline
  pto.backend = "vpto" also selects the always-on VMI path
  f16->f32 store lowers through the fold-consumers path, proving the driver
  uses the optimized pipeline rather than only the hard skeleton

vmi_ptoas_public_abi_invalid.pto / vmi_ptoas_public_result_abi_invalid.pto:
  public VMI argument/result signatures are rejected before layout assignment
```

## MLIR Framework Usage

三个 correctness stage 和若干 layout optimization pass 不应该用同一种 MLIR 机制硬套。
这里先定义实现框架选择，避免后续把 layout 求解、优化重写、结构化控制流改写和 1:N
physicalization 混在一个 pattern pass 里。

当前实现框架按下面的职责切开：

```text
pto-validate-vmi-ir:
  Operation::walk verifier。只看 IR 是否满足阶段不变量，不改 IR，不使用 conversion framework。

vmi-layout-assignment:
  module-level per-SSA-value constraint solver。先收集等价类、producer natural layout 和 consumer request，
  再把结果写回 VMI type/helper op。它可以使用 IRRewriter 改 IR，但不以 TypeConverter 为主模型。

vmi-layout-fold / vmi-layout-rematerialize / vmi-layout-sink-materialization:
  legal-to-legal VMI optimization passes。它们只消费 layout-assigned VMI IR，并继续产出
  layout-assigned VMI IR；所有新选择必须体现在 current op、type 或 helper IR 中。

vmi-legalize-arith-select:
  canonicalize 之后的 hygiene pass。它把 scalar-condition arith.select with VMI result
  恢复成 VMI pipeline 可控的结构化控制流形态。

vmi-to-vpto:
  MLIR OneToNTypeConversion。每个 layout-assigned VMI value 按统一 physical ordering 展开成多个
  VPTO value，并依靠 OneToN structural patterns 重写函数、return、region result、block argument 和
  branch operand。
```

这三个 pass 的边界必须通过 IR 可见状态传递：layout 写在 `!pto.vmi.*` type 上，必要 materialization
写成 `pto.vmi.ensure_*`，physicalization 后不允许残留 `pto.vmi.*`、`!pto.vmi.*` 或
`unrealized_conversion_cast`。不能把 layout 决策藏在 pass-private side table 里让后续 pass 猜。

源码级实现应该进一步拆成七个独立层次：

```text
IR layer:
  include/PTO/IR/VMIAttrs.td
  include/PTO/IR/VMITypeDefs.td
  include/PTO/IR/VMIOps.td
  lib/PTO/IR/VMI.cpp

  只定义语义、parse/print、type/op verifier 和公共 lane-map helper。
  这一层不能知道 layout assignment 的全局选择，也不能直接依赖 VPTO lowering pass。

Semantic validation layer:
  lib/PTO/Transforms/PTOValidateVMIIR.cpp

  只检查阶段输入/输出是否满足 contract。它是 hard gate，不做 repair。

Layout solving layer:
  lib/PTO/Transforms/VMILayoutAssignment.cpp

  负责从 producer/consumer/control-flow/call 关系解出每个 logical value 的 layout，
  然后把结果写回 type 或 ensure_* helper。

Layout support query layer:
  include/PTO/Transforms/VMILayoutSupport.h
  lib/PTO/Transforms/VMILayoutSupport.cpp

  只放跨阶段共享的纯查询：cast layout fact、group_reduce layout fact、
  ensure_* materialization support、layout-aware store support 等。它可以被
  assignment、validation、layout optimization 和 vmi-to-vpto 调用，但不能保存
  per-value 状态，不能返回 VPTO 指令计划，不能决定 clone/rematerialize，也不能
  通过 producer/user/control-flow context 恢复 lowering 决策。

  加新 query 的标准是：至少两个阶段需要同一个语义事实，并且重复实现会导致
  assignment、validation、lowering 对同一个 layout shape 得出不同结论。只有
  一个 lowering pattern 自己使用的分支应该留在该 pattern 内。

Layout optimization layer:
  lib/PTO/Transforms/VMILayoutFold.cpp
  lib/PTO/Transforms/VMILayoutRematerialize.cpp
  lib/PTO/Transforms/VMILayoutSinkMaterialization.cpp
  lib/PTO/Transforms/VMILegalizeArithSelect.cpp

  负责在 layout-assigned VMI IR 内做 legal-to-legal 改写。它可以让公共 canonicalize/cse
  协助清理和合并 IR，但不能把决策藏到 side table 里。

Physicalization layer:
  lib/PTO/Transforms/VMIToVPTO.cpp

  负责把 layout-assigned VMI value 通过 OneToNTypeConversion 展成 VPTO physical values，
  并把每个 pto.vmi.* semantic op 改写成 VPTO op 序列。

Driver/test layer:
  tools/ptoas/ptoas.cpp
  tools/pto-test-opt/
  test/lit/vmi_new/

  ptoas 对 VPTO backend 默认运行完整 pipeline；pto-test-opt 保留单 pass 和中间 IR 的调试入口。
```

每层的 MLIR 框架选择如下：

```text
ODS/TableGen:
  定义 type/attr/op surface 和 verifier hook。

Operation::walk:
  用于 validation 和 layout constraint collection。

Union-find + DenseMap<Value, id>:
  用于 layout assignment 的 per-SSA-value 等价类求解。

IRRewriter/RewriterBase:
  用于 layout assignment 之后的 type rewrite、helper insertion；cheap producer
  rematerialization 属于后续 layout optimization pass。

OneToNTypeConverter + OneToNOpConversionPattern:
  只用于 vmi-to-vpto，把一个 logical VMI value 展成多个 VPTO value。

Upstream OneToN structural helpers:
  func.func / func.call / func.return / common SCF region-result conversion。

Project-local OneToN structural patterns:
  cf.br / cf.cond_br / cf.switch / scf.execute_region / scf.index_switch。
```

不要把这些层次合并成一个万能 pattern pass。特别是：

```text
layout assignment 不能依赖 OneToNTypeConverter:
  因为 layout 不是 type-only 决策，同一个 !pto.vmi.vreg<128xf32> 的不同 SSA value
  可能因 producer/consumer/control-flow 约束得到不同 layout。

vmi-to-vpto 不能重新做 layout solving:
  它只消费已经写在 type/helper 上的 layout 决策。遇到未 assignment 的 VMI type 必须失败。

structural OneToN pattern 不能知道 VMI 语义:
  它们只负责 flatten/rebuild operands、results、successor operands 和 block arguments。
  具体 lane 语义只属于 pto.vmi.* op lowering pattern。

verifier 不能偷偷修 IR:
  否则后续 pass 会依赖 verifier 的隐式 repair 行为，导致 pipeline 顺序不可推理。
```

一个可以直接对照代码的 pass 边界表：

```text
pass                         input                         output
---------------------------  ----------------------------  ----------------------------
pto-validate-vmi-ir          surface VMI IR                same IR, or hard failure
vmi-layout-assignment        surface/layout-partial VMI    layout-assigned VMI IR
layout optimization passes   layout-assigned VMI IR        layout-assigned VMI IR
vmi-legalize-arith-select    layout-assigned VMI IR        layout-assigned VMI IR
pto-validate-vmi-layout-ir   layout-assigned VMI IR        same IR, or hard failure
vmi-to-vpto                  layout-assigned VMI IR        physical VPTO IR
final residual verifier      physical VPTO candidate       no pto.vmi.*, no !pto.vmi.*
```

### 代码级落点

当前实现应该能按文件直接审计。每个 pass 的核心类、MLIR 机制和失败边界如下：

```text
lib/PTO/Transforms/PTOValidateVMIIR.cpp
  pass:
    PTOValidateVMIIRPass
    PTOValidateVMILayoutIRPass
  public helpers:
    validateVMIProducerBoundaryIR
    validateVMILayoutAssignedIR
  MLIR API:
    Operation::walk
    func::FuncOp function type inspection
    recursive TypeAttr / TypedAttr / ArrayAttr / DictionaryAttr scan
  must not:
    rewrite IR
    create unrealized_conversion_cast
    create ConversionTarget
    repair illegal helper/type leakage

lib/PTO/Transforms/VMILayoutAssignment.cpp
  pass:
    VMILayoutAssignmentPass
  core object:
    LayoutSolver
  state:
    DenseMap<Value, unsigned>
    SmallVector<DataNode>
    SmallVector<MaskNode>
    SmallVector<DataUseRequest>
    SmallVector<MaskUseRequest>
  MLIR API:
    Operation::walk for fact collection
    SymbolTable for direct internal calls
    concrete cf/scf handlers for control-flow equivalence
    IRRewriter/OpBuilder only after solving
  must not:
    use TypeConverter as the layout decision model
    rewrite while collecting constraints
    hide chosen layout in a pass-private side table
    infer external VMI ABI

lib/PTO/Transforms/VMILayoutFold.cpp
lib/PTO/Transforms/VMILayoutRematerialize.cpp
lib/PTO/Transforms/VMILayoutSinkMaterialization.cpp
lib/PTO/Transforms/VMILegalizeArithSelect.cpp
  pass:
    VMILayoutFoldPass
    VMILayoutRematerializePass
    VMILayoutSinkMaterializationPass
    VMILegalizeArithSelectPass
  role:
    legal-to-legal layout-assigned VMI optimization and hygiene
  MLIR API:
    Operation::walk for local discovery
    OpBuilder/RewriterBase for explicit IR rewrites
    canonicalize/cse between passes for cleanup and deduplication
  must not:
    introduce physical VPTO register types
    require vmi-to-vpto to inspect producers, users, or CFG
    preserve optimization decisions outside IR

lib/PTO/Transforms/VMIToVPTO.cpp
  pass:
    VMIToVPTOPass
  converter:
    VMIToVPTOTypeConverter : OneToNTypeConverter
  pattern families:
    OneToNOpConversionPattern for pto.vmi.* semantic ops
    upstream func/scf OneToN structural patterns
    project-local cf/scf structural OneToN patterns
  MLIR API:
    populateFuncTypeConversionPatterns
    scf::populateSCFStructuralOneToNTypeConversions
    applyPartialOneToNConversion
    final residual walk
  must not:
    redo layout solving
    inspect defining ops to recover physical parts
    allow pto.vmi.pack/unpack/ensure_* to survive final output
    allow unrealized_conversion_cast to survive final output
```

这里最重要的分界是：`vmi-layout-assignment` 解决的是 value-level layout，`vmi-to-vpto`
解决的是 type/value 1:N physicalization。前者的结果必须已经写回 `!pto.vmi.*` type 或显式
`pto.vmi.ensure_*`；后者只能消费这些 IR-visible facts。

这也回答了“有没有充分利用 MLIR 自带能力”：结构化 1:N signature/control-flow conversion 必须用
MLIR OneToN conversion；layout assignment 则不能强行塞进 converter，因为 converter 看不到
producer natural layout、consumer request、CFG join 和 call-return slot 这些 value-level facts。

### Pass 级实现细则

这几个 pass 对 MLIR 自带能力的使用方式应该是“各用其长”，而不是都套成 converter pattern。
实现时按下面的判断标准拆：

```text
只检查阶段不变量:
  用 Operation::walk。不要创建 ConversionTarget，也不要 rewrite。

需要根据 SSA value、CFG join、call boundary 和 consumer request 决策 layout:
  用 module-level solver。MLIR conversion framework 没有 per-value layout 决策模型。

需要把一个 logical value 展成多个 physical value，并同步改 function/block/control-flow signature:
  用 OneToNTypeConversion。这里是 converter framework 最应该发挥作用的地方。
```

#### Pass 框架细化

第一版实现按下面的源码和 MLIR infra 对齐。这个表是实现时的边界，不只是文档分层：

```text
source file                                pass                         primary MLIR facility
-----------------------------------------  ---------------------------  ---------------------------------------------
lib/PTO/Transforms/PTOValidateVMIIR.cpp    pto-validate-vmi-ir          Operation::walk + recursive type/attr scan
lib/PTO/Transforms/PTOValidateVMIIR.cpp    pto-validate-vmi-layout-ir   Operation::walk + recursive type/attr scan
lib/PTO/Transforms/VMILayoutAssignment.cpp vmi-layout-assignment        module-level union-find solver + IRRewriter
lib/PTO/Transforms/VMILayoutFold.cpp
                                          vmi-layout-fold     Pattern-free local IR rewrite
lib/PTO/Transforms/VMILayoutRematerialize.cpp
                                          vmi-layout-rematerialize      Pattern-free local IR rewrite
lib/PTO/Transforms/VMILayoutSinkMaterialization.cpp
                                          vmi-layout-sink-materialization
                                                                       Pattern-free local IR rewrite
lib/PTO/Transforms/VMILegalizeArithSelect.cpp
                                          vmi-legalize-arith-select     Operation::walk + OpBuilder rewrite
lib/PTO/Transforms/VMIToVPTO.cpp           vmi-to-vpto                  OneToNTypeConverter + OneToNOpConversionPattern
```

这意味着每个 pass 的输入输出 contract 是固定的：

```text
pto-validate-vmi-ir:
  input:
    surface VMI IR
  legal:
    pto.vmi semantic ops
    !pto.vmi.vreg<NxT>
    !pto.vmi.mask<Nxpred>
    func/scf/cf structural ops carrying those types
  illegal:
    layout-assigned !pto.vmi.* type
    physical !pto.vreg / !pto.mask / !pto.align type
    pto.vmi.ensure_* / pack / unpack helper
    VMI or physical type hidden in non-signature attribute
  output:
    exactly the same IR, or failure

vmi-layout-assignment:
  input:
    verifier-clean surface VMI IR
  legal work:
    solve per-SSA layout/granularity constraints
    rewrite VMI value/function/block types with explicit layout
    insert pto.vmi.ensure_* only for use-site materialization
    rematerialize cheap producers instead of inserting ensure_* when semantics are replay-safe
  illegal work:
    physicalize to !pto.vreg / !pto.mask
    introduce pto.vmi.pack / pto.vmi.unpack
    keep layout only in a pass-private side table
  output:
    layout-assigned VMI IR, or failure

pto-validate-vmi-layout-ir:
  input:
    layout-assigned VMI IR
  legal:
    pto.vmi semantic ops
    pto.vmi.ensure_layout / ensure_mask_layout / ensure_mask_granularity
    !pto.vmi.vreg<NxT, layout>
    !pto.vmi.mask<Nxb8/b16/b32, layout>
  illegal:
    surface !pto.vmi.vreg<NxT>
    surface !pto.vmi.mask<Nxpred>
    physical VPTO register types before vmi-to-vpto
    pto.vmi.pack / pto.vmi.unpack
    VMI or physical type hidden in non-signature attribute
  output:
    exactly the same IR, or failure

vmi-to-vpto:
  input:
    layout-assigned VMI IR
  legal work:
    convert each VMI value to an ordered list of physical VPTO values
    rewrite function signatures, block arguments, branch operands, region results and calls
    lower pto.vmi semantic/helper ops to VPTO ops
  illegal work:
    infer missing layouts
    change a chosen layout because one pattern finds a cheaper lowering
    leave pto.vmi.* / !pto.vmi.* / unrealized_conversion_cast in final IR
  output:
    physical VPTO IR, or failure
```

`vmi-layout-assignment` 和 `vmi-to-vpto` 的关键差异是：前者解决“这个 SSA value 应该是什么 layout”，
后者解决“这个已经有 layout 的 SSA value 展开成哪些 physical value”。同一个 surface type 不能用
`TypeConverter` 得到唯一答案：

```mlir
%a = pto.vmi.broadcast %s : f32 -> !pto.vmi.vreg<128xf32>
%b = pto.vmi.extf %x : !pto.vmi.vreg<128xf16> -> !pto.vmi.vreg<128xf32>
%c = scf.if %cond -> !pto.vmi.vreg<128xf32> {
  scf.yield %a : !pto.vmi.vreg<128xf32>
} else {
  scf.yield %b : !pto.vmi.vreg<128xf32>
}
```

这里 `%a` 可以按 consumer 需要 rematerialize 成 contiguous 或 deinterleaved；`%b` 的 natural layout 是
`deinterleaved=2`；`%c` 的 layout 必须由两个 yield 和后续 consumer 共同约束。这个选择依赖 Value、
def-use、control-flow join 和 use-site request，不是 `!pto.vmi.vreg<128xf32> -> ...` 的 type-only 规则。

因此 layout pass 的代码形态应该固定为：

```cpp
LogicalResult LayoutSolver::run() {
  if (failed(collectAllVMIValues()))
    return failure();
  if (failed(collectEquivalenceConstraints()))
    return failure();
  if (failed(collectProducerNaturalLayouts()))
    return failure();
  if (failed(collectConsumerRequests()))
    return failure();
  if (failed(rewriteDataTypes()))
    return failure();
  if (failed(insertDataUseMaterializations()))
    return failure();
  if (failed(inferAndRewriteMaskTypes()))
    return failure();
  if (failed(insertMaskUseMaterializations()))
    return failure();
  rewriteFunctionTypesFromSolvedValues();
  return validateVMILayoutAssignedIR(module);
}
```

其中 `collect*` 阶段只能记录事实，不能边 walk 边改 IR。原因是控制流和 call boundary 会把后面才遇到的
operand/result 合并到前面的 value class；边收集边改 type 会让后续约束看到混合状态，错误诊断也会依赖
walk 顺序。

`vmi-to-vpto` 则必须是 converter pass。第一版使用的是 `OneToNTypeConversion`，因为它要同时处理
value type 和结构签名：

```text
!pto.vmi.vreg<128xf32, #pto.vmi.layout<deinterleaved = 2>>
  -> !pto.vreg<64xf32>, !pto.vreg<64xf32>

func.func @f(%arg0: !pto.vmi.vreg<128xf32, layout>) -> !pto.vmi.vreg<128xf32, layout>
  -> func.func @f(%arg0_0: !pto.vreg<64xf32>, %arg0_1: !pto.vreg<64xf32>)
       -> (!pto.vreg<64xf32>, !pto.vreg<64xf32>)
```

这里不能用普通 1:1 `TypeConverter`，也不能靠每个 VMI op pattern 自己拆 operand。否则 `func.return`、
`cf.br`、`scf.for` iter arg 这种没有 VMI defining op 的边界会漏转换。`OneToN` adaptor 才是 semantic
pattern 获取 physical parts 的唯一来源：

```cpp
ValueRange lhsParts = adaptor.getLhs();
ValueRange rhsParts = adaptor.getRhs();
TypeRange resultTypes = adaptor.getResultMapping().getConvertedTypes(0);
```

结构化转换的实现分工如下：

```text
upstream helper:
  populateFuncTypeConversionPatterns
    covers func.func / func.return / direct func.call signature conversion

  scf::populateSCFStructuralOneToNTypeConversions
    covers common SCF result/yield/block-argument structural conversions

project-local OneToN patterns:
  cf.br
  cf.cond_br
  cf.switch
  scf.execute_region
  scf.index_switch
```

项目内 structural pattern 只能做结构搬运：

```text
1. read OneToNTypeMapping for each original operand/result
2. flatten successor operands or region result types
3. rebuild the same cf/scf op with converted types
4. inline/move original regions when required
```

它们不能做下面这些事：

```text
infer layout from operand defining op
emit vadd/vcvt/vlds/vsts
decide contiguous vs deinterleaved
special-case pto.vmi semantic op
```

VMI 语义只能出现在 `OneToNOpConversionPattern<pto.vmi.*>` 里。这样才能保证 block argument、function
argument、loop-carried value 和 branch target argument 都按同一套 physical ordering 转换。

`vmi-to-vpto` 的 legality 由 preflight + conversion + final gate 三段组成，而不是单靠
`ConversionTarget`：

```text
preflight:
  verifyVMIToVPTOInputIR
    rejects layout-free VMI types
  verifySupportedVMIToVPTOOps
    rejects unsupported semantic/materialization cases before rewrite starts

conversion:
  applyPartialOneToNConversion
    applies structural and semantic OneToN patterns

final gate:
  verifyNoResidualVMIIR
    rejects pto.vmi.*
    rejects !pto.vmi.* in operand/result/block/function/attribute type trees
    rejects pto.vmi.pack/unpack materialization helpers
    rejects unrealized_conversion_cast
```

这比只设置 `ConversionTarget` 更直接，因为当前 OneToN 工具链的重点是 type/value expansion 和 pattern
rewriter；最终合法性必须递归检查 attribute/type tree，防止 VMI type 被藏在 nested attr 里。

#### `pto-validate-vmi-ir` / `pto-validate-vmi-layout-ir`

这两个 pass 是 hard gate，不是 legalization pass。

使用的 MLIR 能力：

```text
Operation::walk:
  遍历 module 内所有 op、region、block argument、operand/result type 和 attribute。

TypeAttr / TypedAttr recursive scan:
  拒绝把 VMI/physical VPTO type 藏在 nested attribute 中。

func::FuncOp function type special case:
  function_type attr 是签名本身，可以按当前阶段规则检查；其它 attr 不能携带 VMI/physical type。
```

不使用 `ConversionTarget` 的原因：

```text
ConversionTarget 适合表达“哪些 op/type legal，哪些 pattern 能改掉”。
这里我们只想回答“当前 IR 是否已经处在某个阶段边界”，失败后必须停机，而不是尝试 repair。
如果 verifier 顺手改 IR，pipeline 的阶段不变量会变成隐式行为，后续 pass 很难审计。
```

这两个 pass 的输出只能是原 IR 或 failure：

```cpp
void runOnOperation() override {
  if (failed(verifyStageInvariant(getOperation())))
    signalPassFailure();
}
```

#### `vmi-layout-assignment`

这个 pass 使用 MLIR 的 IR 遍历和 rewrite 基础设施，但不使用 `TypeConverter` 作为主模型。

核心原因：

```text
TypeConverter 的输入是 Type。
layout assignment 的输入是 Value。

同一个 !pto.vmi.vreg<128xf32> 可以因为不同 producer/consumer 关系得到不同 layout：
  f16->f32 widen result       -> deinterleaved=2
  f8 ->f32 widen result       -> deinterleaved=4
  only contiguous store value -> contiguous
```

实现应拆成两个阶段，不要边 walk 边 rewrite：

```text
collect:
  1. 收集所有 VMI data/mask SSA value 和 block argument。
  2. 用 union-find 合并必须同 layout 的 value。
  3. 记录 producer natural layout。
  4. 记录 consumer layout/granularity request。
  5. 记录 function return slot、call operand/result、branch operand/block argument 关系。

rewrite:
  1. 为每个 equivalence class 选 layout。
  2. 改写 value/function/block/result type。
  3. 对 use-site mismatch 插入 ensure_* 或 rematerialize cheap producer。
  4. 运行 pto-validate-vmi-layout-ir。
```

建议的数据结构边界：

```cpp
struct DataNode {
  Value value;
  VMIVRegType type;
  unsigned parent;
  VMILayoutAttr naturalLayout;
};

struct MaskNode {
  Value value;
  VMIMaskType type;
  unsigned parent;
  VMILayoutAttr requestedLayout;
  std::string requestedGranularity;
};

struct DataUseRequest {
  OpOperand *operand;
  VMILayoutAttr layout;
};

struct MaskUseRequest {
  OpOperand *operand;
  VMILayoutAttr layout;
  std::string granularity;
};
```

这里可以充分使用 MLIR 的接口，但它们只是 constraint source：

```text
BranchOpInterface / concrete cf.* handlers:
  successor operand[i] == destination block argument[i]

RegionBranchOpInterface / concrete scf.* handlers:
  region yield operand[i] == parent result[i]
  loop init/result/iter_arg/yield 同 slot 等价

CallOpInterface + SymbolTable:
  direct internal call operand/result 和 callee argument/return slot 等价
  external/indirect VMI call 先拒绝，因为缺 ABI materialization

IRRewriter:
  只在 solve 完成后统一改 type、插 ensure_*、clone cheap producer。
```

`vmi-layout-assignment` 的 pass invariant 是：所有 layout 决策必须写回 IR。后续 `vmi-to-vpto`
只能读取 `!pto.vmi.*` type 和显式 `pto.vmi.ensure_*`，不能依赖 layout solver 的 side table。

#### `vmi-to-vpto`

这个 pass 应该充分使用 MLIR converter framework，具体是 `OneToNTypeConversion`，不是普通
`DialectConversion`。

普通 1:1 dialect conversion 不够的地方：

```text
!pto.vmi.vreg<128xf32, deinterleaved=2>
  -> !pto.vreg<64xf32>, !pto.vreg<64xf32>

!pto.vmi.vreg<256xf8, deinterleaved=4>
  -> !pto.vreg<256xf8>, !pto.vreg<256xf8>, !pto.vreg<256xf8>, !pto.vreg<256xf8>
```

函数参数、返回值、block argument、branch operand、region result 都必须做同样的 1:N 展开。
这正是 `OneToNTypeConverter`、`OneToNOpConversionPattern` 和结构化 OneToN helper 的职责。

实现骨架：

```cpp
void runOnOperation() override {
  ModuleOp module = getOperation();

  if (failed(verifyVMIToVPTOInputIR(module)) ||
      failed(verifySupportedVMIToVPTOOps(module)))
    return signalPassFailure();

  VMIToVPTOTypeConverter typeConverter;
  RewritePatternSet patterns(&getContext());

  populateFuncTypeConversionPatterns(typeConverter, patterns);
  scf::populateSCFStructuralOneToNTypeConversions(typeConverter, patterns);
  populateProjectLocalCFOneToNPatterns(typeConverter, patterns);
  populateVMISemanticOneToNPatterns(typeConverter, patterns);

  if (failed(applyPartialOneToNConversion(module, typeConverter,
                                          std::move(patterns))) ||
      failed(verifyNoResidualVMIIR(module)))
    signalPassFailure();
}
```

`VMIToVPTOTypeConverter` 只做一种事：把 layout-assigned VMI type 映射到 canonical physical value list。
它不能重新推导 layout。

```text
contiguous:
  chunk0, chunk1, ... in logical order

deinterleaved=2:
  part0 chunks for logical lanes 0,2,4,...
  part1 chunks for logical lanes 1,3,5,...

deinterleaved=4:
  part0 chunks for lanes 0,4,8,...
  part1 chunks for lanes 1,5,9,...
  part2 chunks for lanes 2,6,10,...
  part3 chunks for lanes 3,7,11,...

num_groups=G:
  group-slot reduce result layout
  physical storage is contiguous chunk order
  only canonical group_slot(g) lanes contain semantic values
```

每个 semantic pattern 必须从 adaptor 拿 physical parts，不允许从 defining op 反推：

```cpp
LogicalResult matchAndRewrite(VMIAddFOp op, OpAdaptor adaptor,
                              OneToNPatternRewriter &rewriter) const override {
  ValueRange lhs = adaptor.getLhs();
  ValueRange rhs = adaptor.getRhs();
  TypeRange resultTypes = adaptor.getResultMapping().getConvertedTypes(0);

  if (lhs.size() != rhs.size() || lhs.size() != resultTypes.size())
    return rewriter.notifyMatchFailure(op, "physical arity mismatch");

  SmallVector<Value> results;
  for (auto [i, resultType] : llvm::enumerate(resultTypes)) {
    results.push_back(
        rewriter.create<VaddFOp>(op.getLoc(), resultType, lhs[i], rhs[i])
            .getResult());
  }

  rewriter.replaceOp(op, results, adaptor.getResultMapping());
  return success();
}
```

这个约束对控制流是关键的：`scf.for` iter arg、branch target argument、function argument 都没有可用的
defining op；它们的 physical parts 只能来自 OneToN signature/block argument conversion。

`vmi-to-vpto` 应有三层失败点，诊断不要混在一起：

```text
preflight:
  layout 未 assignment、unsupported semantic op、unsupported materialization path

conversion:
  pattern 缺失、arity mismatch、结构化控制流展开失败

final residual verifier:
  任何 pto.vmi.*、!pto.vmi.*、pto.vmi.pack/unpack/ensure_*、unrealized_conversion_cast 残留
```

### `pto-validate-vmi-ir`

`pto-validate-vmi-ir` 是边界 verifier，不使用 DialectConversion。

推荐使用：

```text
Operation::walk
TypeSwitch / isa / dyn_cast
emitOpError / InFlightDiagnostic
SymbolTable, for function/call boundary checks
CallGraph or manual call graph collection, if recursive SCC needs diagnostics
DominanceInfo, if helper placement or resource dominance is checked
```

这个 pass 只检查 VMI producer boundary 和阶段不变量：

```text
before layout assignment:
  VMI data/mask values use surface type
  no layout-assigned VMI type leaks in unless the test explicitly starts after assignment
  no physical VPTO op appears in the semantic VMI region
  no VMI helper op appears before the pass that is allowed to create it
  no non-signature op/module TypeAttr or TypedAttr payload contains VMI or physical VPTO types

after layout assignment:
  pass: pto-validate-vmi-layout-ir
  every VMI data value has a layout
  every VMI mask has layout and concrete granularity
  control-flow joins have stable type/layout
  no non-signature op/module TypeAttr or TypedAttr payload contains VMI or physical VPTO types

after VMI-to-VPTO:
  no VMI op/type/helper remains
  no unrealized_conversion_cast remains
```

不要把这个 pass 写成 rewrite pass。它可以收集 context 用于诊断，但不能通过局部修补让非法 IR
继续前进；否则后续 pass 会开始依赖 verifier 的隐式 repair 行为。

实现上要扫描的不只是 operand/result/block argument：

```text
func.func function type:
  作为函数签名本身检查，允许出现当前阶段合法的 VMI type。

non-signature attributes:
  module/op attribute 中只要递归包含 VMI type 或 physical VPTO type 都拒绝。这里包括 TypeAttr、
  TypedAttr，以及 ArrayAttr/DictionaryAttr 这类容器中的 nested attribute/type payload。
```

这样可以堵住 hidden-state 形式的 side table，例如把 `!pto.vmi.vreg<...>` 偷存在 module attribute
里。`func.func` 的内建 `function_type` attr 是唯一例外，因为它只是函数签名的 MLIR 表达，不是额外
隐藏状态。

### `vmi-layout-assignment`

`vmi-layout-assignment` 不以 MLIR `TypeConverter` 作为主机制。

原因是 layout 选择不是单纯的 `Type -> TypeRange` 映射：

```text
same surface type:
  !pto.vmi.vreg<128xf32>

possible per-value decisions:
  value produced by f16->f32 widen: deinterleaved=2
  value loaded only for contiguous store: contiguous
  value feeding fp8-like->f32 consumer path: deinterleaved=4
```

两个 SSA value 可以有完全相同的 surface type，但因为 producer natural layout、consumer demand、
控制流 join 和 target capability 不同，得到不同 layout。因此主模型应该是 per-SSA-value 的约束图，
而不是类型转换表。

推荐内部结构：

```text
DenseMap<Value, LayoutNode>
DenseMap<BlockArgument, LayoutNode>
DenseMap<Operation *, OpLayoutTransfer>
SmallVector<LayoutConstraint>
SmallVector<MaterializationRequest>
```

推荐使用的 MLIR 基础能力：

```text
RegionBranchOpInterface:
  collect scf.if/scf.for-like region entry, yield, result relations

BranchOpInterface:
  collect cf.br/cf.cond_br predecessor operand -> block argument relations

CallOpInterface, CallableOpInterface, FunctionOpInterface:
  collect call operand/result and function argument/result relations

SymbolTable:
  resolve direct calls and reject unresolved VMI signature assumptions

DominanceInfo:
  choose legal insertion points for ensure_layout, mask conversion, and rematerialization

IRRewriter / RewriterBase:
  rewrite types, insert helper ops, clone rematerializable producers
```

求解结果必须 materialize 回 IR，不能留在 side table：

```text
1. Rewrite every VMI value type to a layout-assigned type.
2. Rewrite mask type to layout + b8/b16/b32 granularity.
3. Insert pto.vmi.ensure_layout where a consumer requires a different layout.
4. Insert pto.vmi.ensure_mask_layout / ensure_mask_granularity where predicate layout or granularity differs.
5. Clone rematerializable producers such as constant, broadcast, create_mask, iota-like producers when cheaper.
6. Re-run the VMI stage verifier.
```

这个 pass 可以用 `RewritePatternSet` 辅助局部 canonicalization，例如删除同 layout 的
`ensure_layout`，但不能让 greedy pattern driver 决定全局 layout。全局约束必须先收敛，再做改写。

更具体地说，这里不用 `TypeConverter` 的原因不是 MLIR converter 不好用，而是此阶段的问题不是
“一个旧 type 机械变成一个新 type”：

```text
%a : !pto.vmi.vreg<128xf32>  // 只被 contiguous store 消费
%b : !pto.vmi.vreg<128xf32>  // 来自 f16->f32 widen，后续继续 vadd
%c : !pto.vmi.vreg<128xf32>  // 控制流 join，两个 predecessor 必须统一 layout
```

这三个 value 的 surface type 完全相同，但 layout 决策分别可能是 contiguous、deinterleaved=2、
以及由 join 两侧约束共同决定。`TypeConverter` 看不到“这个 SSA value 的 producer/consumer/CFG
关系”，所以它只能作为后续 physicalization 的工具，不能作为 layout assignment 的主算法。

该 pass 对 MLIR 基础能力的使用边界是：

```text
Operation::walk:
  收集所有 VMI SSA value、block argument、函数签名和 op transfer facts。

Union-find / DenseMap<Value, id>:
  表达必须同 layout 的 equivalence class。

SymbolTable:
  解析 direct internal func.call；带 VMI type 的 external/indirect call 先拒绝。

IRRewriter:
  改写 function/block/result type，插入 ensure_*。

verifyLayoutAssignedVMIIR:
  pass 末尾 hard gate，确认所有决策已经 materialize 到 IR。
```

### `vmi-to-vpto`

`vmi-to-vpto` 应该使用 MLIR 的 1:N conversion framework，而不是普通 `DialectConversion`。
这个 pass 的核心问题正是一个 logical VMI value physicalize 成多个 VPTO value：

```text
!pto.vmi.vreg<NxT, layout> -> !pto.vreg<partLanes x T>...
!pto.vmi.mask<NxG, layout> -> !pto.mask<G>...
```

普通 `DialectConversion` 的 `OpConversionPattern` 对 1:N fixed operand/result 支持不够直接：
pattern adaptor 可能拿到 source materialization，也可能拿到 flat converted operands；`func.return`
这类“一个 logical operand 展开成多个 physical operands”的场景也容易出现不完整展开。因此这里采用
MLIR `OneToNTypeConversion` 工具：

推荐组件：

```text
OneToNTypeConverter
OneToNOpConversionPattern
OneToNPatternRewriter
OneToNTypeMapping
populateFuncTypeConversionPatterns
scf::populateSCFStructuralOneToNTypeConversions
applyPartialOneToNConversion
final residual verifier
```

`OneToNTypeConverter` 负责 layout-assigned VMI type 到 ordered physical VPTO value list：

```cpp
typeConverter.addConversion([](VMIVRegType type, SmallVectorImpl<Type> &results) {
  // Use getVMIPhysicalArity(type) and the shared lane-map helper.
  // Append one physical !pto.vreg<lanesPerPart x elementType> per part/chunk.
});

typeConverter.addConversion([](VMIMaskType type, SmallVectorImpl<Type> &results) {
  // Use mask granularity and physical arity helper.
  // Append one physical !pto.mask<granularity> per part/chunk.
});
```

source/target materialization 可以用 VMI helper 承接中间状态：

```text
VMI value -> physical values:
  pto.vmi.unpack

physical values -> VMI value:
  pto.vmi.pack
```

但它们只是 conversion materialization，不是最终 IR 的合法残留。final gate 必须拒绝：

```text
pto.vmi.pack
pto.vmi.unpack
pto.vmi.ensure_layout
pto.vmi.ensure_mask_layout
pto.vmi.ensure_mask_granularity
unrealized_conversion_cast
```

`applyPartialOneToNConversion` 本身不是 legality framework；它负责应用 1:N patterns 并替换内部
`unrealized_conversion_cast`。因此 `vmi-to-vpto` 必须在 conversion 后运行 final residual verifier，
把下面这些全部作为 hard failure：

```text
any pto.vmi.* op
any !pto.vmi.* type
any pto.vmi.pack/unpack materialization helper
any pto.vmi.ensure_* helper
any unrealized_conversion_cast
```

结构转换必须覆盖：

```text
func arguments/results and return operands:
  use populateFuncTypeConversionPatterns

call operands/results:
  convert callee signature and call sites together

block arguments and branch operands:
  convert target block arguments and predecessor operands in the same conversion
  current implementation provides project-local OneToN patterns for cf.br,
  cf.cond_br, and cf.switch because MLIR only provides the generic
  BranchOpInterface helper for ordinary 1:1 dialect conversion, not for VMI
  1:N physicalization.

scf.if/scf.for region yields and results:
  use scf::populateSCFStructuralOneToNTypeConversions
  otherwise write explicit OneToN patterns around RegionBranchOpInterface relations
```

如果当前 LLVM/MLIR 版本没有提供对应 OneToN helper，就补项目内 custom `OneToNConversionPattern`。
选择标准不是“少写代码”，而是能否正确处理 1:N result、block argument、region yield 和
recursive/function SCC。

当前实现的结构转换分工如下：

```text
upstream OneToN helper:
  func.func / func.return / func.call
  scf.if / scf.for / scf.while and common SCF structural cases

project-local OneToN structural patterns:
  cf.br
  cf.cond_br
  cf.switch
  scf.execute_region
  scf.index_switch
```

项目内 structural pattern 只做一件事：按照 `OneToNTypeMapping` 展平/重建 operand、result、
successor operand 和 block argument。它们不能内嵌 VMI layout 语义，也不能通过 defining op
重新推导物理寄存器列表。VMI 语义只出现在各个 `pto.vmi.*` 的 `OneToNOpConversionPattern` 中。

OneToN conversion 的执行顺序：

```text
1. Populate structural conversion patterns.
2. Populate VMI semantic op lowering patterns.
3. Populate helper lowering/materialization patterns.
4. applyPartialOneToNConversion on the module.
5. Run final residual verifier as the hard legality gate.
```

如果 conversion 或 final gate 失败，诊断必须区分：

```text
unsupported VMI semantic op
unsupported layout materialization path
unconverted function/control-flow boundary
unexpected VMI helper residual
unexpected unrealized_conversion_cast
```

这样 pass 边界就是清楚的：

```text
pto-validate-vmi-ir:
  verifier/walk, no conversion

vmi-layout-assignment:
  global per-value layout solver, then IR materialization

vmi-to-vpto:
  OneToNTypeConversion-based 1:N physicalization and final legality gate
```

### Concrete Pass Skeleton

整个 pipeline 按下面的 hard contract 串起来：

```text
raw VMI producer
  -> pto-validate-vmi-ir
  -> vmi-layout-assignment
  -> canonicalize/cse
  -> vmi-layout-fold
  -> canonicalize/cse
  -> vmi-layout-rematerialize
  -> canonicalize/cse
  -> vmi-layout-sink-materialization
  -> canonicalize/cse
  -> vmi-legalize-arith-select
  -> pto-validate-vmi-layout-ir
  -> vmi-to-vpto
  -> SIMT unroll/SCCP/canonicalize/CSE
  -> vpto pointer/wrapper normalization
  -> pto-infer-vpto-vecscope
  -> VMI LICM
  -> canonicalize/CSE
  -> final residual verifier
```

The `ptoas` VPTO driver uses this sequence by default. The test-opt entry
remains useful for isolated pass debugging. Optimization after physicalization
must preserve the inferred resultless vecscope boundary. Pre-emission
canonicalization remains before inference as input normalization; VMI LICM and
the final canonicalize/CSE cleanup run after inference.

各阶段之间只通过 IR 传递状态，不通过 pass-private side table 传递语义。也就是说：

```text
layout assignment output:
  VMI value type already contains layout
  VMI mask type already contains layout + concrete b8/b16/b32 granularity
  required layout conversion already appears as pto.vmi.ensure_* or rematerialized producer

vmi-to-vpto input:
  may contain pto.vmi.* semantic ops and helper ops
  must not contain layout-free VMI type
  function signatures and op/module TypeAttr or TypedAttr payloads are part of this invariant,
  not just SSA operands/results

vmi-to-vpto output:
  must not contain pto.vmi.* op/type/helper
  must not contain unrealized_conversion_cast
  function type attributes and any other op/module TypeAttr or TypedAttr payloads must not contain !pto.vmi.*
```

This prevents a fragile design where `vmi-to-vpto` has to rediscover layout decisions from defining ops. A VMI value
may be a function argument, block argument, `scf.if` result, `scf.for` carried value, or branch target argument; none
of those has a useful defining op.

#### Layout Assignment State

`vmi-layout-assignment` should be implemented as one module-level solver object:

```cpp
struct DataValueState {
  Value value;
  VMIVRegType surfaceType;
  UnionFindNode eqClass;
  VMILayoutAttr naturalLayout;        // producer-preferred layout
  SmallVector<LayoutUseRequest> uses; // consumer requirements
};

struct MaskValueState {
  Value value;
  VMIMaskType surfaceType;
  UnionFindNode eqClass;
  VMILayoutAttr requestedLayout;
  StringRef requestedGranularity;     // b8/b16/b32 after inference
  SmallVector<MaskUseRequest> uses;   // consumer layout/granularity requests
};

struct LayoutUseRequest {
  Operation *consumer;
  VMILayoutAttr layout;
  StringRef reason; // add/select/store/widen-source/etc.
};
```

The solver runs in phases:

```text
1. collect all VMI data/mask SSA values, including block arguments
2. add equivalence constraints
3. add producer natural-layout constraints
4. add consumer layout/granularity requests
5. solve each equivalence class
6. insert ensure_* for non-class-compatible uses
7. rewrite value types and function signatures
8. run pto-validate-vmi-layout-ir
```

Equivalence is only for cases where two logical values must have the same physical lane order:

```text
add/sub/mul:
  lhs == rhs == result

cmpf/cmpi:
  lhs == rhs
  result mask requests lhs layout + element-width granularity

select:
  true_value == false_value == result
  mask operand gets a use-site request for result layout + element-width granularity

scf.if:
  result[i] == then yield[i] == else yield[i]

scf.for:
  init_arg[i] == region_iter_arg[i] == yield[i] == result[i]

cf.br/cf.cond_br:
  successor operand[i] == successor block argument[i]

direct internal func.call:
  call operand[i] == callee argument[i]
  call result[i] == all callee return operand[i]
```

Natural layout is not equivalence. For example:

```text
extf f16 -> f32:
  result natural layout = deinterleaved=2

extf f8 -> f32:
  result natural layout = deinterleaved=4

truncf f32 -> f16:
  result natural layout = contiguous

truncf f32 -> fp8-like:
  result natural layout = contiguous

store:
  consumer requests contiguous externally visible order
```

If one equivalence class has incompatible natural layouts, the pass must diagnose `VMI-LAYOUT-CONTRACT` unless an
explicit use-site `ensure_*` can represent the requested materialization. Baseline layout assignment does not
clone/rematerialize producers. The separate `vmi-layout-rematerialize` optimization may replace an `ensure_*`
with a cloned trivially replayable producer after the materialization request is visible in IR:

```text
constant
broadcast
constant_mask
create_mask
```

For non-rematerializable producers, insert `pto.vmi.ensure_layout` immediately before the consumer that requested the
different layout. This is the conservative first implementation rule. It works for ordinary SSA values, block
arguments, loop-carried values, branch arguments, and call results because the helper is dominated by the value at the
use site and does not need to be hoisted across control flow. `DominanceInfo` may be used later to hoist duplicated
helpers as an optimization, but it must not be required for correctness in the first implementation.

That helper is a real IR marker: if `vmi-to-vpto` cannot lower its requested conversion, the program fails with an
explicit unsupported materialization diagnostic.

#### Layout Assignment Implementation Frame

This pass is a normal `OperationPass<ModuleOp>`. It deliberately does not use `DialectConversion`, because there is
no stable `Type -> Type` rule until the pass has solved producer preference, consumer demand, and control-flow joins.
The implementation should look like this:

```cpp
struct LayoutSolver {
  ModuleOp module;
  MLIRContext *ctx;

  DenseMap<Value, unsigned> dataIds;
  SmallVector<DataNode> dataNodes;
  DenseMap<Value, unsigned> maskIds;
  SmallVector<MaskNode> maskNodes;

  SmallVector<DataUseRequest> dataUseRequests;
  SmallVector<MaskUseRequest> maskUseRequests;
  DenseMap<func::FuncOp, SmallVector<Value>> firstReturnOperandsByFunc;

  LogicalResult collectConstraints();
  LogicalResult rewriteIR();
};
```

The concrete state objects should carry only facts that are materialized back into IR:

```cpp
struct DataNode {
  Value value;
  VMIVRegType surfaceType;
  unsigned parent;
  VMILayoutAttr naturalLayout; // null means no producer preference yet
};

struct MaskNode {
  Value value;
  VMIMaskType surfaceType;
  unsigned parent;
  VMILayoutAttr requestedLayout;
  std::string requestedGranularity; // empty until b8/b16/b32 is known
};

struct DataUseRequest {
  OpOperand *operand;
  VMILayoutAttr layout;
};

struct MaskUseRequest {
  OpOperand *operand;
  VMILayoutAttr layout;
  std::string granularity;
};
```

Do not store hidden layout state that `vmi-to-vpto` must rediscover. After this pass, a debugger should be able to read
the IR and know the chosen layout for every VMI value from its type alone.

The pass body should stay simple:

```cpp
void runOnOperation() override {
  LayoutSolver solver(getOperation());
  if (failed(solver.collectConstraints()) ||
      failed(solver.rewriteIR()) ||
      failed(verifyLayoutAssignedVMIIR(getOperation())))
    signalPassFailure();
}
```

The current implementation should map directly to this phase order:

```cpp
LogicalResult LayoutSolver::run() {
  if (failed(collect()))
    return failure();
  if (failed(addConstraints()))
    return failure();

  rewriteDataTypes();
  if (failed(insertDataUseMaterializations()))
    return failure();

  if (failed(inferMaskRequests()))
    return failure();
  rewriteMaskTypes();
  if (failed(insertMaskUseMaterializations()))
    return failure();

  rewriteFunctionType();
  return validateVMILayoutAssignedIR(module);
}
```

This order is intentional:

```text
collect:
  only discovers VMI values and block arguments.

addConstraints:
  only records equivalence, natural layout and consumer request facts.
  It must not rewrite IR, because later CFG/call constraints may still merge
  two values that were already seen.

rewriteDataTypes:
  commits solved data layouts to !pto.vmi.vreg type.

insertDataUseMaterializations:
  repairs use-site layout mismatch after the producer's committed type is known.

inferMaskRequests:
  uses already committed data layouts and element widths to infer concrete mask
  layout/granularity requests.

rewriteMaskTypes:
  commits mask layout and b8/b16/b32 granularity.

insertMaskUseMaterializations:
  repairs mask layout/granularity mismatch.

rewriteFunctionType:
  updates function signatures last, after argument/result value types have been
  rewritten.
```

Do not move `rewriteFunctionType` before use-site materialization. A function signature is the public shape of the
solved value class; changing it early makes call/return diagnostics depend on walk order and can hide an unresolved
use-site mismatch.

Constraint collection is a module walk with explicit handlers. The important point is that each handler only records
facts; it must not rewrite while walking:

```text
Data equivalence:
  pto.vmi.addf/addi: lhs == rhs == result
  pto.vmi.cmpf/cmpi: lhs == rhs
  pto.vmi.select: true_value == false_value == result
  pto.vmi.ensure_layout: source and result are not equivalent if layouts differ

Data natural layout:
  pto.vmi.extf f16->f32: result natural = deinterleaved=2
  pto.vmi.extf fp8-like->f32: result natural = deinterleaved=4
  pto.vmi.truncf:        result natural = contiguous
  pto.vmi.channel_merge with C inputs: result natural = deinterleaved=C

Data use request:
  pto.vmi.store: value requested as contiguous
  pto.vmi.channel_split with C results: source requested as deinterleaved=C
  op requiring a common operand/result layout: request producer class layout

Mask request:
  cmp result:      same data layout as operands, granularity from element width
  select mask:     same data layout as selected value, granularity from element width
  store mask path: same data layout as stored value, granularity from element width
```

Control flow should be handled as equivalence, not as local op preference:

```text
scf.if:
  result[i] == then yield[i] == else yield[i]

scf.for:
  init_arg[i] == body iter_arg[i] == yield[i] == result[i]

scf.while:
  before argument[i] == condition forwarded operand[i] == after argument[i]
  after yield[i] == result[i]

scf.execute_region:
  every nested scf.yield operand[i] == execute_region result[i]

scf.index_switch:
  every case/default yield operand[i] == index_switch result[i]

cf.br:
  operand[i] == destination block argument[i]

cf.cond_br:
  true operand[i] == true destination block argument[i]
  false operand[i] == false destination block argument[i]

cf.switch:
  default operand[i] == default destination block argument[i]
  case k operand[i] == case k destination block argument[i]

func.call:
  only direct internal callees are supported in the first implementation
  call operand[i] == callee argument[i]
  call result[i] == every corresponding callee return operand[i]
```

Function returns need one extra bookkeeping rule. A function result slot has one public layout in the function type, so
all `func.return` operands at the same index must be equivalent:

```text
first return operand[i] == every later return operand[i]
function result type[i] is rewritten from the solved type of return operand[i]
call result[i] == every corresponding callee return operand[i]
```

If two return paths naturally produce incompatible layouts, the pass should report `VMI-LAYOUT-CONTRACT` instead of
silently choosing one path:

```mlir
^a:
  %x = pto.vmi.extf %f16 : !pto.vmi.vreg<128xf16> -> !pto.vmi.vreg<128xf32>
  return %x : !pto.vmi.vreg<128xf32> // natural deinterleaved=2

^b:
  %y = pto.vmi.extf %f8 : !pto.vmi.vreg<256xf8E4M3FN> -> !pto.vmi.vreg<256xf32>
  return %y : !pto.vmi.vreg<256xf32> // different result shape/layout, invalid by verifier/type first
```

For equal result shape but incompatible producer preferences, the same rule applies:

```text
return slot 0 from f16->f32 path: natural deinterleaved=2
return slot 0 from f8E4M3FN->f32 path with the same logical result shape: natural deinterleaved=4
diagnostic: VMI-LAYOUT-CONTRACT: conflicting natural layouts ...
```

External declarations with VMI types are not a layout problem; they are ABI materialization. The first implementation
must reject them before rewriting:

```text
VMI-LAYOUT-CONTRACT: VMI typed function declaration requires an explicit external ABI materialization plan
```

The rewrite phase has three ordered steps:

```text
1. Rewrite all data SSA value types to !pto.vmi.vreg<NxT, chosen_layout>.
2. Rewrite all mask SSA value types to !pto.vmi.mask<NxG, chosen_layout, concrete_granularity>.
3. Repair use-site mismatches by either rematerializing a cheap producer or inserting an explicit helper.
```

Rematerialization is allowed only when replaying the producer cannot change memory, control flow, or execution count
semantics:

```text
allowed:
  pto.vmi.constant splat
  pto.vmi.broadcast
  pto.vmi.constant_mask
  pto.vmi.create_mask

not allowed in the first implementation:
  load
  arithmetic result
  conversion result
  shuffle/channel_split/channel_merge result
  value crossing a call boundary or block argument
```

If rematerialization is not legal, insert:

```text
pto.vmi.ensure_layout
pto.vmi.ensure_mask_layout
pto.vmi.ensure_mask_granularity
```

These helpers make the unresolved materialization explicit. `vmi-layout-assignment` is allowed to create them;
`vmi-to-vpto` is responsible for proving and lowering them. If lowering cannot prove the physical transform, the final
diagnostic should be an unsupported layout/materialization diagnostic, not silent incorrect code.

Layout assignment completion checks:

```text
1. No surface !pto.vmi.vreg<NxT> remains.
2. No surface !pto.vmi.mask<Nxpred> remains.
3. Every VMI function argument, result, block argument, branch operand, call operand, and return operand has the
   layout-assigned type selected by the solved equivalence class.
4. Every consumer-specific mismatch is represented by an explicit pto.vmi.ensure_* op immediately before that
   consumer. Optional optimization passes may later replace selected helpers with rematerialized cheap producers.
5. External declarations with VMI types are rejected; they are not rewritten into an implicit ABI.
```

#### OneToN Conversion Details

`vmi-to-vpto` should use MLIR `OneToNTypeConversion` for all structural rewriting that involves VMI values:

```text
OneToNTypeConverter:
  !pto.vmi.vreg<NxT, layout> -> !pto.vreg<partLanes x T>...
  !pto.vmi.mask<NxG, layout> -> !pto.mask<G>...

Patterns:
  framework structural OneToN patterns for func/return/scf
  explicit OneToNOpConversionPattern for each pto.vmi semantic op
  explicit helper patterns for pack/unpack/ensure_*

Final gate:
  reject residual pto.vmi.*, !pto.vmi.*, function signatures containing !pto.vmi.*, and unrealized_conversion_cast
```

The implementation is an `OperationPass<ModuleOp>` with this shape:

```cpp
struct VMIToVPTOTypeConverter final : OneToNTypeConverter {
  VMIToVPTOTypeConverter() {
    addConversion([](Type t) { return t; });
    addConversion(convertVMIVRegType);
    addConversion(convertVMIMaskType);

    TypeConverter::addSourceMaterialization(materializeVPTOToVMI);
    TypeConverter::addArgumentMaterialization(materializeVPTOToVMI);
    OneToNTypeConverter::addTargetMaterialization(materializeVMIToVPTO);
  }
};

void runOnOperation() override {
  ModuleOp module = getOperation();
  if (failed(verifyVMIToVPTOInputIR(module)) ||
      failed(verifySupportedVMIToVPTOOps(module)))
    return signalPassFailure();

  VMIToVPTOTypeConverter typeConverter;
  RewritePatternSet patterns(module.getContext());
  populateVMIOneToNConversionPatterns(typeConverter, patterns);

  if (failed(applyPartialOneToNConversion(module, typeConverter,
                                          std::move(patterns))) ||
      failed(verifyNoResidualVMIIR(module)))
    signalPassFailure();
}
```

The type converter must define one canonical physical ordering and every pattern must use that ordering:

```text
!pto.vmi.vreg<NxT, contiguous>
  -> chunks in logical order:
     chunk0 lanes [0..P-1], chunk1 lanes [P..2P-1], ...

!pto.vmi.vreg<NxT, deinterleaved=2>
  -> part-major chunks:
     part0 chunk0 lanes [0,2,4,...]
     part0 chunk1 next even lanes
     part1 chunk0 lanes [1,3,5,...]
     part1 chunk1 next odd lanes

!pto.vmi.vreg<NxT, deinterleaved=4>
  -> part-major chunks:
     part0 lanes [0,4,8,...]
     part1 lanes [1,5,9,...]
     part2 lanes [2,6,10,...]
     part3 lanes [3,7,11,...]

!pto.vmi.vreg<NxT, num_groups=G>
  -> chunks in contiguous physical storage order
     only derived group_slot(g) lanes contain semantic values
     this layout is valid only for group reduce/broadcast exchange values

!pto.vmi.mask<NxG, layout, granularity>
  -> same part/chunk ordering as its data layout, one !pto.mask<granularity> per physical part/chunk
```

`materializeVPTOToVMI` and `materializeVMIToVPTO` should use only `pto.vmi.pack` and `pto.vmi.unpack`. These ops are
conversion scaffolding; they are never valid final output. This makes accidental framework materialization visible in
the IR and easy to reject.

Pattern population should be explicit:

```cpp
void populateVMIOneToNConversionPatterns(VMIToVPTOTypeConverter &converter,
                                         RewritePatternSet &patterns) {
  populateFuncTypeConversionPatterns(converter, patterns);
  scf::populateSCFStructuralOneToNTypeConversions(converter, patterns);

  patterns.add<OneToNCFBranchOpPattern,
               OneToNCFCondBranchOpPattern,
               OneToNCFSwitchOpPattern>(converter, ctx);

  patterns.add<OneToNSCFExecuteRegionOpPattern,
               OneToNSCFIndexSwitchOpPattern>(converter, ctx);

  patterns.add<OneToNVMIPackOpPattern,
               OneToNVMIUnpackOpPattern,
               OneToNVMIEnsureLayoutOpPattern,
               OneToNVMIEnsureMaskLayoutOpPattern,
               OneToNVMIEnsureMaskGranularityOpPattern,
               ... semantic VMI op patterns ...>(converter, ctx);
}
```

Use upstream OneToN helpers where they exist:

```text
func.func / func.return / func.call:
  populateFuncTypeConversionPatterns

scf.if / scf.for / scf.while and common structural SCF:
  scf::populateSCFStructuralOneToNTypeConversions
```

Use project-local OneToN patterns where the current MLIR version does not provide a complete 1:N structural rewrite:

```text
cf.br
cf.cond_br
cf.switch
scf.execute_region
scf.index_switch
```

These project-local structural patterns should not know VMI semantics. They only flatten operands/results according to
`OneToNTypeMapping`, convert successor block argument lists, and rebuild the same control-flow op.

#### Pattern Authoring Checklist

Every new `pto.vmi.*` lowering pattern should answer the same questions before it is added to
`populateVMIOneToNConversionPatterns`:

```text
1. Does the op require all data operands/results to have identical physical arity?
   If yes, check every ValueRange size against the result mapping before emitting VPTO ops.

2. Does the op consume a mask?
   If yes, the mask must already have concrete granularity and the same physical ordering expected by the data
   operand. The pattern must not reinterpret a pred mask by lane count alone.

3. Does the op observe contiguous logical order outside the register file?
   If yes, require contiguous layout or explicitly lower the ensure_layout/materialization before using load/store
   style VPTO ops.

4. Does the op have padding lanes?
   If yes, prove padding is unobservable. For load-like ops this requires a full-read safety proof or a fallback.
   For store-like ops this requires a true predicate that disables padding writes.

5. Does the op have target-specific side effects or ordering, such as squeeze/compact/store coupling?
   If yes, put that check in verifySupportedVMIToVPTOOps before conversion starts, so the pass fails before partial
   rewriting.

6. Can it create pto.vmi.pack/unpack or unrealized_conversion_cast through framework materialization?
   If yes, the semantic pattern still may be correct, but final residual verification must reject any leftover helper.
```

This gives a concrete division of labor:

```text
verifySupportedVMIToVPTOOps:
  shape/target/path support checks that should fail before any rewrite.

OneToNOpConversionPattern:
  mechanical lowering for a preflight-approved case.

verifyNoResidualVMIIR:
  final hard gate for missed patterns, illegal materializations and hidden VMI type payloads.
```

Do not put target capability probing in a structural pattern. For example, a `cf.br` pattern must never ask whether
`deinterleaved=4` can be materialized. It only converts successor operands. The semantic op that created or consumes
the value is responsible for proving the VPTO lowering path.

#### Converter Use By Pass

The implementation should be reviewable with the following rule:

```text
pto-validate-vmi-ir:
  no TypeConverter, no ConversionTarget, no rewrite.

vmi-layout-assignment:
  no TypeConverter for choosing layouts.
  It may use RewriterBase after solving, but not DialectConversion as the solving model.

vmi-to-vpto:
  must use OneToNTypeConverter for VMI types.
  must use OneToNOpConversionPattern for semantic VMI ops.
  should use upstream func/scf OneToN helpers when available.
  may add project-local structural OneToN patterns only for missing framework coverage.
```

The main reason is not style. It is correctness across values without defining ops:

```mlir
^bb0(%x: !pto.vmi.vreg<128xf32, #pto.vmi.layout<deinterleaved = 2>>):
  cf.br ^bb1(%x : !pto.vmi.vreg<128xf32, #pto.vmi.layout<deinterleaved = 2>>)

^bb1(%y: !pto.vmi.vreg<128xf32, #pto.vmi.layout<deinterleaved = 2>>):
  %z = pto.vmi.addf %y, %y
    : !pto.vmi.vreg<128xf32, #pto.vmi.layout<deinterleaved = 2>>
  ...
```

`%y` has no defining VMI op. Its physical values are the converted block arguments produced by OneToN block signature
conversion. Any implementation that tries to recover physical parts from a defining op is therefore incomplete for
control flow, function arguments and loop-carried values.

When writing semantic `OneToNOpConversionPattern`, do not infer physical parts from a defining op. Use the OneToN
adaptor's per-original-operand `ValueRange`:

```cpp
LogicalResult matchAndRewrite(VMIAddFOp op, OpAdaptor adaptor,
                              OneToNPatternRewriter &rewriter) const override {
  ValueRange lhsParts = adaptor.getLhs();
  ValueRange rhsParts = adaptor.getRhs();
  TypeRange resultTypes = adaptor.getResultMapping().getConvertedTypes(0);
  ...
  rewriter.replaceOp(op, physicalResults, adaptor.getResultMapping());
}
```

Every VMI semantic lowering then follows the same shape:

```cpp
ValueRange lhsParts = adaptor.getLhs();
ValueRange rhsParts = adaptor.getRhs();
TypeRange resultTypes = adaptor.getResultMapping().getConvertedTypes(0);

for each physical part index i:
  emit physical VPTO op for lhsParts[i], rhsParts[i] -> resultTypes[i]

replace op with all physical results using adaptor.getResultMapping()
```

This convention is mandatory for values crossing control flow. For example an `scf.for` iter arg has no defining op;
its physical parts are the converted block arguments created by OneToN signature conversion.

The concrete pattern shape is:

```cpp
LogicalResult matchAndRewrite(SourceOp op, OpAdaptor adaptor,
                              OneToNPatternRewriter &rewriter) const override {
  ValueRange in0 = adaptor.getIn0();
  ValueRange in1 = adaptor.getIn1();
  TypeRange outTypes = adaptor.getResultMapping().getConvertedTypes(0);

  if (in0.size() != in1.size() || in0.size() != outTypes.size())
    return rewriter.notifyMatchFailure(op, "physical arity mismatch");

  SmallVector<Value> results;
  for (auto [i, outType] : llvm::enumerate(outTypes)) {
    results.push_back(rewriter.create<PhysicalOp>(op.getLoc(), outType,
                                                  in0[i], in1[i]).getResult());
  }

  rewriter.replaceOp(op, results, adaptor.getResultMapping());
  return success();
}
```

For non-VMI operands, use a helper like `getSingleValue(op, adaptor.getOffset(), "...")` and fail if the framework
unexpectedly expanded them. This catches malformed conversion rules early.

#### Semantic Lowering Buckets

The first implementation should split VMI op lowering into four buckets:

```text
identity/helper:
  pack, unpack, ensure_layout identity/materialization cases, ensure_mask_* identity case

per-part elementwise:
  addf, addi, subf, subi, mulf, muli, divf, minf, maxf, negf, absf, absi, sqrt, exp, ln, relu, andi, ori, xori, shli, shrui, shrsi, not, cmpf, cmpi, select

per-part predicate:
  mask_and, mask_or, mask_xor, mask_not

layout-producing conversion:
  extf, truncf, bitcast

externally ordered memory:
  load, store

value-indexed accumulation:
  dhist, chist
```

Per-part elementwise ops are straightforward only when all operands/results already share the same assigned layout:

```text
logical deinterleaved=2 value:
  part0 contains logical lanes 0, 2, 4, ...
  part1 contains logical lanes 1, 3, 5, ...

vmi.addf/subf/mulf on two such values:
  emit the matching VPTO per-part op for part0_lhs, part0_rhs
  emit the matching VPTO per-part op for part1_lhs, part1_rhs
```

This preserves logical lane semantics because each physical part contains the same logical lane subset for all
operands and the result.

Memory ops are different because their observable semantics are contiguous logical order:

```text
vmi.store of deinterleaved=2:
  cannot blindly store part0 then part1 as the final memory order
  must use a store plan that writes logical lane 0,1,2,3,... order
  or materialize source to contiguous before physical store
```

Therefore `store` lowering must either:

```text
1. consume contiguous layout directly, or
2. lower ensure_layout(deinterleaved -> contiguous), then store, or
3. use target store instructions whose dist mode proves contiguous external order
```

The first implementation uses option 2 for full physical chunks:

```text
vmi.load:
  emit contiguous physical vlds chunks in memory order
  materialize contiguous -> assigned result layout

vmi.masked_load:
  only when the full physical read footprint is proven safe
  emit contiguous physical vlds chunks in memory order
  select loaded lanes against passthru with the VMI mask
  if enable-stable-gather-masked-load is set, reject pto.vmi.masked_load with
  a stable TODO diagnostic until the VGATHER2-based strict no-read path is
  implemented

vmi.store:
  materialize assigned source layout -> contiguous
  emit physical vsts chunks in memory order
```

Current direct memory lowering may only emit VPTO vector memory ops for
UB-backed memory. Concretely, a `!pto.ptr<..., ub>` is legal, a
`!pto.ptr<..., gm>` is not; a memref with `#pto.address_space<vec>` is legal,
and a memref without a memory-space attribute is treated as unknown/local to
this stage to preserve existing local-view tests. A memref explicitly marked
GM or another non-VEC space is rejected by `vmi-to-vpto`.

GM-backed VMI memory is still a valid semantic source/sink before this pass,
but direct lowering does not perform GM<->UB movement. That must be represented
by an earlier/lower memory access plan, scratch materialization, or UB view
normalization before `vmi-to-vpto`; otherwise the diagnostic is
`VMI-UNSUPPORTED` and names the GM-backed source/destination.

For `deinterleaved=2`, `vldsx2 DINTLV_B*` and `vstsx2 INTLV_B*` are valid optimization candidates because the ISA has
an explicit two-stream de/interleave memory distribution mode. This should be implemented only as a peephole inside
`vmi-to-vpto` after the generic plan is correct:

```text
vmi.load result layout deinterleaved=2:
  vldsx2 DINTLV_B* can directly produce part0/part1 chunks

vmi.store source layout deinterleaved=2:
  vstsx2 INTLV_B* can directly store part0/part1 chunks in logical memory order
```

Do not generalize this to `deinterleaved=4` unless the two-level dist composition is proven against the ISA. The
fallback for `deinterleaved=4` remains generic layout materialization plus ordinary memory ops.

Direct `vmi.load` is lowered as full VPTO physical reads when the source memory kind/layout is supported and the
element type has a known physical lane width, even for non-full logical vectors. Masked/expand/gather read-style
operations still require the lowering to prove that the full physical read footprint is safe, or to use a future
true masked/non-faulting fallback. The current proof handles:

```text
source is a statically shaped memref
offset is a constant non-negative index
offset + physical_arity(result) * lanes_per_physical_part <= static memref element count
```

When this proof holds, masked/expand read-style operations may still issue full `pto.vlds` chunks. The extra padding
lanes are not logical VMI lanes and must remain unobservable through later VMI materialization rules. Pointer sources,
dynamic offsets, dynamic memrefs, and insufficient static footprints remain unsupported for those stricter read-style
operations:

```text
VMI-UNSUPPORTED: pto.vmi.<load-op> requires full physical chunks without padding lanes or a statically safe full-read
footprint (...; safe-read proof failed: ...)
VMI-UNSUPPORTED: pto.vmi.<load-op> ... (source is GM-backed, but current direct VMI-to-VPTO memory lowering emits
pto.vlds/pto.vsts and requires UB-backed memory)
```

Store-style ops are different because inactive lanes can be made write-free with true predicates. `vmi.store`,
`vmi.masked_store` therefore support the explicit contiguous/deinterleaved tail-store
materialization paths described below.

## 2. Slice 0: Type / Attr Bootstrap

第一步只实现 VMI type、layout attr 和纯 helper，不实现任何 conversion pass。

### 2.1 `#pto.vmi.layout`

定义 `VMILayoutAttr`：

```mlir
#pto.vmi.layout<contiguous>
#pto.vmi.layout<deinterleaved = 2>
#pto.vmi.layout<deinterleaved = 4>
```

建议内部参数：

```text
kind: enum { contiguous, deinterleaved }
factor: int64_t
```

Verifier：

```text
contiguous:
  factor must be 1

deinterleaved:
  factor must be 2 or 4
```

禁止接受其它 spelling，例如 `stride2`、`stride4`、`parity`、`mod_split`、`blocked`。

### 2.2 `!pto.vmi.vreg`

定义 `VMIVRegType`：

```mlir
!pto.vmi.vreg<128xf32>
!pto.vmi.vreg<128xf32, #pto.vmi.layout<contiguous>>
!pto.vmi.vreg<128xf32, #pto.vmi.layout<deinterleaved = 2>>
```

建议参数：

```text
elementCount: int64_t
elementType: Type
layout: Attribute  // null means surface type before layout assignment
```

Verifier：

```text
elementCount > 0
elementType is scalar-like integer / float / index supported by VMI
layout is null or VMILayoutAttr
deinterleaved=4 only allowed when target registry later supports it; type verifier only checks shape
```

不要要求 `elementCount * bitwidth(elementType)` 是 256B 整数倍。

### 2.3 `!pto.vmi.mask`

定义 `VMIMaskType`：

```mlir
!pto.vmi.mask<128xpred>
!pto.vmi.mask<128xb32, #pto.vmi.layout<contiguous>>
!pto.vmi.mask<128xb32, #pto.vmi.layout<deinterleaved = 2>>
```

建议参数：

```text
elementCount: int64_t
granularity: enum/string { pred, b8, b16, b32 }
layout: Attribute
```

Verifier：

```text
elementCount > 0
surface mask may use pred and no layout
layout-assigned mask must use b8/b16/b32 and must have VMILayoutAttr
pred mask must not carry layout
```

### 2.4 Lane Map Helper

在 C++ 中提供纯函数 helper，供 verifier、layout assignment、VMI-to-VPTO 和测试共用：

```text
getDataLanesPerPart(elementType)
getMaskLanesPerPart(granularity)
getVMIPhysicalArity(type)
mapLogicalLaneToPhysical(type, logicalLane)
mapPhysicalLaneToLogical(type, part, chunk, lane)
isPaddingLane(type, part, chunk, lane)
```

这些 helper 是 hard dependency。任何 pass 不能重新手写一套 arity 公式。

Slice 0 完成条件：

```text
1. VMI type/attr 能 parse/print round-trip。
   Covered by vmi_type_attr_parse.pto.
2. 非法 layout factor、非法 mask granularity、非法 element count 有 verifier diagnostic。
   Covered by vmi_layout_factor_invalid.pto,
   vmi_mask_granularity_invalid.pto, vmi_type_element_count_invalid.pto,
   and vmi_mask_concrete_without_layout_invalid.pto /
   vmi_mask_pred_with_layout_invalid.pto.
3. helper 单测或 lit 测试覆盖 contiguous/deinterleaved=2/deinterleaved=4 和非整 tile。
   Covered by vmi_to_vpto_type_only.pto and
   vmi_to_vpto_type_arity.pto.
```

## 3. Slice 1: Minimal VMI Op Set

不要一次实现 75 个 semantic op。第一批只实现能跑通 widening + elementwise + store 的闭环。

### 3.1 必选 semantic op

Construction：

```text
pto.vmi.constant
pto.vmi.broadcast
pto.vmi.iota
pto.vmi.create_mask
pto.vmi.constant_mask
```

`pto.vmi.from_elements` belongs to the eventual construction surface, but it is
not part of Slice 1. Do not synthesize it from ad hoc scalar lane inserts until
there is an explicit vreg immediate, scalar-insert, or scratch materialization
contract.

Mask：

```text
pto.vmi.mask_and
pto.vmi.mask_or
pto.vmi.mask_xor
pto.vmi.mask_not
```

Arithmetic / conversion：

```text
pto.vmi.addf
pto.vmi.addi
pto.vmi.subf
pto.vmi.subi
pto.vmi.mulf
pto.vmi.muli
pto.vmi.fma
pto.vmi.divf
pto.vmi.minf
pto.vmi.maxf
pto.vmi.negf
pto.vmi.absf
pto.vmi.absi
pto.vmi.sqrt
pto.vmi.exp
pto.vmi.ln
pto.vmi.relu
pto.vmi.andi
pto.vmi.ori
pto.vmi.xori
pto.vmi.shli
pto.vmi.shrui
pto.vmi.shrsi
pto.vmi.not
pto.vmi.cmpf
pto.vmi.cmpi
pto.vmi.select
pto.vmi.extf
pto.vmi.truncf
pto.vmi.bitcast
```

`pto.vmi.shrui` represents logical right shift and lowers to unsigned
`pto.vshr`. `pto.vmi.shrsi` represents arithmetic right shift and lowers to
signed `pto.vshr`; the physical element type selects the VPTO/VISA sign mode.
Integer div/rem, integer casts, int-float casts, and index casts are also
intentionally outside the current VMI surface until signedness, rounding,
saturation, overflow/remainder, and target lowering contracts are explicit.

Memory：

```text
pto.vmi.load
pto.vmi.masked_load
pto.vmi.gather
pto.vmi.expand_load
pto.vmi.store
pto.vmi.masked_store
pto.vmi.scatter
pto.vmi.compress_store
```

Value-indexed accumulation：

```text
pto.vmi.vdhist
pto.vmi.vchist
```

`pto.vmi.vdhist` is a first-stage semantic op when histogram support is enabled.
`pto.vmi.vchist` may share the surface verifier, but its final lowering must be
gated until the target CHISTv2 high-range cumulative semantics are verified.

Current implementation scope note:

```text
pto.vmi.gather / scatter
pto.vmi.active_prefix_index / compress / compress_store
future scan / contract style ops
```

These families are not first-stage completion blockers. The dialect surface may
define them, and the lowering may keep narrow direct paths when the target VPTO
contract is already explicit. Full semantic coverage for these families remains
out of scope until cross-chunk state, duplicate-index ordering, prefix carry,
compaction state, or contraction accumulation contracts are explicitly designed.
Unsupported shapes must fail before OneToN rewrite with `VMI-UNSUPPORTED`; they
must not fall through to residual-op diagnostics.

Permutation：

```text
pto.vmi.shuffle
pto.vmi.channel_split
pto.vmi.channel_merge
```

Internal helper：

```text
pto.vmi.ensure_layout
pto.vmi.ensure_mask_layout
pto.vmi.ensure_mask_granularity
pto.vmi.unpack
pto.vmi.pack
```

### 3.2 Op Verifier Rules

Construction op verifier:

```text
constant value must be a dense elements attr, and its element type/count must match the result vreg
broadcast scalar type must match the result element type
constant_mask value must be a dense elements attr, must have i1 element type, and its element count must match the
result mask
create_mask may produce surface pred mask or concrete layout-assigned mask
mask_and/mask_or/mask_xor/mask_not require all mask operands/results to have the same logical lane count; if any
mask is layout-assigned, all masks must carry the same layout and granularity
```

Elementwise op verifier：

```text
all data operands have same logical lane count
all data operands have same element type except documented conversion op
if any operand has layout, all layouted operands/results must agree
surface op may have no layout before vmi-layout-assignment
```

`select` verifier：

```text
mask lane count == true/false/result lane count
mask layout must match data layout after layout assignment
mask granularity must match selected element width after layout assignment
```

`extf/truncf` verifier：

```text
source/result lane count equal
source/result element types are float
bitwidth changes in the expected direction
truncf rounding attr, when present, must be A/H and currently only applies to
  f32 -> !pto.hif8
```

Memory op verifier:

```text
load memory element type must match result VMI data element type when the source is PtrType or MemRefType
store memory element type must match stored VMI data element type when the destination is PtrType or MemRefType
```

Histogram op verifier:

```text
dhist/chist acc type must be !pto.vmi.vreg<256xui16>
dhist/chist result type must match acc type
source type must be !pto.vmi.vreg<Nxui8>
mask logical lane count must match source logical lane count
surface mask may be pred; after layout assignment it must be b8 contiguous
source/result/acc must not carry layout before vmi-layout-assignment
layout-assigned dhist/chist requires contiguous source, mask, acc, and result
```

`shuffle` verifier：

```text
static mask length == result lane count
each mask index selects an existing source logical lane
result element type == source element type
no padding lane may be selected
```

`channel_split` verifier：

```text
result count C >= 2
input lane count N == C * M
each result is vreg<MxT>
channel c result semantics: out[c][i] = input[i * C + c]
if any source/result carries layout, all must carry layout
for C=2/4, layout-assigned source must be contiguous or deinterleaved=C
layout-assigned results must be contiguous
```

`channel_merge` verifier：

```text
operand count C >= 2
all operands have same M and element type T
result is vreg<C * M x T>
result semantics: result[i * C + c] = input[c][i]
if any input/result carries layout, all must carry layout
layout-assigned inputs must be contiguous
for C=2/4, layout-assigned result must be contiguous or deinterleaved=C
```

`ensure_layout` verifier：

```text
source/result are both VMIVRegType
same elementCount and elementType
source/result both layout-assigned
source layout may equal result layout; that is a canonical no-op
```

`ensure_mask_layout` verifier is identical except it uses `VMIMaskType` and preserves granularity.

`ensure_mask_granularity` verifier：

```text
source/result are both VMIMaskType
same elementCount
same layout
source/result granularity are b8/b16/b32
logical predicate value must be preserved
```

`pack/unpack` verifier：

```text
VMI side must be layout-assigned
physical operand/result count == getVMIPhysicalArity(VMI type)
physical data types are !pto.vreg<lanesPerPart x T>
physical mask types are !pto.mask<G>
ordering is the shared Physical Arity helper order
```

Slice 1 完成条件：

```text
1. Every Slice 1 op parses, prints, and has negative verifier tests.
   Arithmetic/mask/helper verifier coverage includes vmi_elementwise_kind_invalid.pto,
   vmi_mask_logic_invalid.pto, vmi_ensure_layout_surface_invalid.pto,
   vmi_unpack_arity_invalid.pto, and vmi_pack_arity_invalid.pto.
2. Helper ops are marked internal in docs and rejected by final VMI-to-VPTO gate if residual.
3. `channel_split/channel_merge` have tests proving shuffle-equivalent lane order.
```

## 4. Slice 2: VMI Producer Boundary Verifier

VMI core implementation starts from VMI IR. Producer-specific import is outside this manual's core path.

实现 `PTOValidateVMIIR.cpp` 中的 VMI boundary verifier：

```text
recommended pass name: pto-validate-vmi-ir
anchor: func::FuncOp or ModuleOp
source file: lib/PTO/Transforms/PTOValidateVMIIR.cpp
```

Boundary verifier checks:

```text
all logical vector values use !pto.vmi.vreg / !pto.vmi.mask
all logical vector behavior is represented by pto.vmi semantic ops
surface VMI values before layout assignment do not carry layout
no physical VPTO op appears before vmi-to-vpto
no hidden side table is required to interpret VMI values
scalar/tensor/debug/transform boundary has already been resolved by producer
```

Slice 2 完成条件：

```text
1. VMI-native positive tests pass boundary verification.
   Covered by vmi_producer_boundary_valid.pto.
2. Physical VPTO op before VMI-to-VPTO is rejected.
   Covered by vmi_producer_boundary_physical_invalid.pto, including both
   physical function types and physical VPTO ops.
3. Layout-assigned type before layout assignment is rejected unless the test explicitly starts after layout assignment.
   Covered by vmi_producer_boundary_layout_invalid.pto and
   vmi_producer_boundary_mask_layout_invalid.pto.
4. Missing VMI type/op invariants produce `VMI-PASS-INVARIANT` or a more specific diagnostic.
   Covered by vmi_producer_boundary_non_vmi_op_invalid.pto,
   vmi_producer_boundary_helper_invalid.pto, and the producer-boundary
   TypeAttr nested/surface/layout invalid tests.
```

## 5. Slice 3: `vmi-layout-assignment`

推荐实现为 pass：

```text
recommended pass name: vmi-layout-assignment
anchor: ModuleOp
source file: lib/PTO/Transforms/VMILayoutAssignment.cpp
```

`vmi-layout-assignment` 必须是 module 级 pass。函数参数、`func.return` operand、
`func.call` operand/result 和 callee signature 需要在同一个约束图里求解；函数级 pass
只能看到局部 body，无法安全地同步 callsite 和 callee。

### 5.1 Internal Data Model

Build one layout node per VMI SSA value:

```text
Operation result
BlockArgument
Region yield operand
Function argument/result
Call operand/result
```

Each node records:

```text
logical type: VMIVRegType or VMIMaskType
allowed layouts: bitset {contiguous, deinterleaved2, deinterleaved4}
required mask granularity: pred/b8/b16/b32 or unknown
natural layout preference
hard constraints
```

No information required by later passes may live only in this data structure. After the pass, type/attr/op
operands must fully describe the result.

### 5.2 Transfer Functions

Minimum Slice 3 transfer functions：

```text
constant/broadcast/create_mask/constant_mask:
  rematerializable in any legal consumer layout

mask_and/mask_or/mask_xor/mask_not:
  all mask operands/results same layout and granularity

addf/addi/subf/subi/mulf/muli/divf/minf/maxf/negf/absf/absi/sqrt/exp/ln/relu/andi/ori/xori/shli/shrui/shrsi/not/cmpf/cmpi/select:
  all data operands/results same layout
  mask layout follows data layout

extf f16 -> f32:
  result natural layout = deinterleaved=2
  source requires contiguous layout for the direct vcvt part=EVEN/ODD path
  partial/tail source chunks are supported when they still fit in one physical
  source chunk and produce the natural two-part result; source padding lanes map
  only to result padding lanes

extf f8 -> f32:
  result natural layout = deinterleaved=4
  source requires contiguous layout for the direct vcvt part=P0/P1/P2/P3 path
  partial/tail source chunks are supported under the same one-source-chunk
  contract; source padding lanes map only to result padding lanes

truncf f32 -> f16:
  can consume deinterleaved=2 and produce contiguous
  current implementation records a deinterleaved=2 source use-site request and
  inserts pto.vmi.ensure_layout when the source value solved to contiguous.
  partial/tail source pairs are supported when the two deinterleaved source
  parts pack into one contiguous result chunk; source padding lanes map only to
  result padding lanes

truncf f32 -> fp8-like:
  can consume deinterleaved=4 and produce contiguous
  current implementation records a deinterleaved=4 source use-site request and
  inserts pto.vmi.ensure_layout when the source value solved to contiguous.
  The lowering emits four pto.vcvt operations with part=P0/P1/P2/P3, then ORs
  the mutually exclusive partial destination registers into one contiguous fp8
  result. This mirrors the hardware packed-4 contract: each source part owns
  one quarter of the destination byte lanes, so the final externally visible
  vector remains logical lane order 0..N-1 after the merge.
  default round mode is result-type specific: f8E4M3/f8E5M2 use rnd=R, hif8
  uses rnd=A. hif8 may explicitly request hybrid lowering with
  pto.vmi.truncf {rounding = "H"}, which forwards rnd=H to every packed part.

bitcast:
  source and result layouts must match
  source/result total logical bits must match
  current implementation supports contiguous/deinterleaved layouts with identical
  physical arity when every source/result physical chunk carries the same number
  of logical bits. This covers full chunks and partial/tail chunks such as
  65xf32 -> 130xi16, where the second physical chunk carries 32 logical bits on
  both sides, and uneven deinterleaved tails such as 129xf32 -> 129xi32.
  Partial/tail bitcast remains unsupported if source padding bits would become
  result logical bits. group_slots bitcast follows the same rule: it is valid
  only when the source/result group_slots layout is identical and every
  physical group-slot chunk carries the same logical bit footprint.

load:
  baseline result layout is deterministic from explicit layout attrs or the
  producer natural layout; consumer-specific alternatives are represented by
  ensure_layout and optimized later

store:
  baseline requests contiguous source layout
  current implementation records a contiguous use-site request for vmi.store and
  inserts pto.vmi.ensure_layout when the stored value class solved to a
  non-contiguous layout. This makes externally visible memory order explicit in
  IR before vmi-to-vpto. If explicit IR reaches vmi-to-vpto with a
  deinterleaved=2/4 tail value, the direct lowering may still materialize it to
  contiguous physical chunks first, but only when every deinterleaved part has
  the same physical chunk count and therefore forms complete intlv groups.

shuffle/channel_split/channel_merge:
  default result layout contiguous unless the current op explicitly carries a
  supported layout-preserving contract
  current implementation supports pto.vmi.shuffle when every result physical
  chunk forwards one source physical chunk with identical lane positions for
  all non-padding result lanes. Result padding lanes are ignored by the
  forwarding proof and remain unobservable after physicalization. This allows
  whole-chunk projection/reordering under contiguous or explicit deinterleaved
  layouts, including tail-prefix projections such as `[0, 1, 2, 3] ->
  !pto.vmi.vreg<4xf32>`. Arbitrary lane permutation remains unsupported unless
  the vselr index-vector path below can materialize it.
  current implementation supports channel_split/channel_merge for 2 or 4
  channels. channel_split consumes a natural deinterleaved=C source and produces
  contiguous per-channel results; channel_merge consumes contiguous per-channel
  inputs and produces a natural deinterleaved=C result. The direct path also
  accepts partial/tail channel groups when the virtual deinterleaved=C channel
  layout has the same physical arity as the source/result representation, so
  every physical group can be materialized with complete intlv/dintlv pairs.
  Arity-changing partial groups such as splitting 4xf32 into two 2xf32 channels
  remain unsupported. If a producer/consumer
  requires dense contiguous layout, pto.vmi.ensure_layout materializes the
  pto.vdintlv/pto.vintlv tree explicitly. Non-matching layouts and other channel
  counts remain unsupported.
```

### 5.3 Solver Order

Implement deterministic solving:

```text
1. Collect region/SCC constraints, including scf/cf/function/call boundaries.
2. Propagate impossible layouts and required mask granularities.
3. Pick one layout per node using deterministic priority, not a cost model:
   explicit layout already present on the VMI type, then unique natural layout,
   then hard non-contiguous request, then contiguous.
5. Rewrite result/block/function types to layout-assigned VMI types.
6. Insert ensure_layout / ensure_mask_layout / ensure_mask_granularity at uses that need conversion.
7. Run verifier gate.
```

Current implementation status:

```text
implemented:
  extf source -> contiguous use-site request for supported f16/fp8-like to f32 paths
  truncf f32->f16 source -> deinterleaved=2 use-site request
  truncf f32->fp8-like source -> deinterleaved=4 use-site request
  single-use pto.vmi.load results can adopt a consumer-requested
  layout before type rewrite; this covers direct memory producers such as
  load -> truncf without inserting a redundant ensure_layout
  vmi.store data operand -> contiguous use-site request
  explicit VMI vreg layout is preserved as an initial solver constraint
  explicit concrete VMI mask layout/granularity is preserved as an initial solver constraint
  channel_split source -> deinterleaved=C use-site request
  channel_split results -> contiguous natural layout
  channel_merge inputs -> contiguous use-site request
  channel_merge result -> deinterleaved=C natural layout
  shuffle without explicit layouts -> contiguous source use-site request and contiguous result natural layout
  shuffle with explicit source/result layouts -> preserve explicit layouts and let vmi-to-vpto prove chunk forwarding
  pto.vmi.ensure_layout insertion for non-contiguous store operands
  pto.vmi.ensure_layout insertion for truncf source materialization
  pto.vmi.ensure_mask_layout / ensure_mask_granularity insertion for select mask operands
  pto.vmi.create_mask / constant_mask rematerialization for select mask operands when the consumer needs a
  different mask layout/granularity
  splat pto.vmi.constant rematerialization for data operands when the consumer needs
  a different layout
  pto.vmi.broadcast rematerialization for data operands when the consumer needs
  a different layout
  scf.execute_region result/yield layout equivalence
  scf.index_switch result/yield layout equivalence
  scf.while state layout equivalence

not yet implemented:
  generic per-consumer layout request table for every VMI op
  producer rematerialization for non-splat data constants and other cheap producers
  cost model / target capability registry
```

Do not implement a local greedy pattern pass that ignores block arguments or function signatures.

### 5.4 CFG Rules

CFG 处理分两层。第一层是必须做的 layout equivalence：同一个控制流值在
result、yield、region/block argument 之间必须形成同一个 layout/mask 约束组。第二层才是
layout conflict resolution：当同一个 producer 的不同 consumers 希望不同 layout 时，插入
`ensure_layout` 或 `ensure_mask_layout`。后续 `vmi-layout-rematerialize` 可以把部分 helper
替换成重放的纯构造 producer。

当前可落地的最小实现先做第一层。它不尝试在 branch 边界自动插入 conversion，因此下面这些
关系一旦因为 natural layout 或 mask granularity 冲突无法合并，必须报 `VMI-LAYOUT-CONTRACT`，
不能默默选择某一边。

`scf.if` equivalence：

```text
for each result index i:
  scf.if result[i]
  == then scf.yield operand[i]
  == else scf.yield operand[i]
```

如果 value 是 `!pto.vmi.vreg`，合并 data layout 约束；如果 value 是
`!pto.vmi.mask`，合并 mask layout 和 granularity 请求。这样 `%m = scf.if ... ->
!pto.vmi.mask` 后被 `vmi.select` 消费时，select 对 `%m` 推出的 `b8/b16/b32 + layout`
会传播回两边 yield 的 mask producer。

`scf.for` equivalence：

```text
for each iter_arg index i:
  init_arg[i]
  == region_iter_arg[i]
  == scf.yield operand[i]
  == scf.for result[i]
```

这条规则避免 loop-carried value 每次迭代改变 layout。对于 `extf f16->f32` 作为 init、
loop body 内部 `addf` 并 yield 的 case，`extf` 的 natural layout `deinterleaved=2`
必须稳定传递到 `%acc` region arg、`scf.yield` 和 loop result。

`cf.br` / `cf.cond_br` equivalence：

```text
for each successor operand index i:
  branch successor operand[i]
  == successor block argument[i]
```

当前实现覆盖标准 `cf.br`、`cf.cond_br` 和 `cf.switch`。其中 `cf.switch` 的 default operands
与 default destination block arguments 按 index 建 layout 等价关系；每个 case operand segment
与对应 case destination block arguments 按 index 建 layout 等价关系。更泛化的
`BranchOpInterface` op 如果携带 VMI type，后续要么补对应 mapping，要么在 layout assignment
阶段明确 diagnostic，不能让 hidden default layout 穿过去。

当前实现支持携带 VMI value 的 `scf.execute_region`：execute_region result 与直属 region terminator
`scf.yield` operands 按 result index 合并到同一个 layout 等价类。嵌套 region 内属于其他 op 的
`scf.yield` 不参与 execute_region 的等价关系。

当前实现支持携带 VMI value 的 `scf.index_switch`：default/case region `scf.yield` operands 与
index_switch results 按 result index 合并到同一个 layout 等价类。

当前实现支持携带 VMI value 的 `scf.while`：init operand、before region argument、`scf.condition`
forwarded operand、after region argument、after region `scf.yield` operand 和 while result 按状态
index 合并到同一个 layout 等价类。`scf.condition` 的 i1 condition 本身不参与 VMI layout 约束。

Function boundary：

```text
internal functions may get specialized layouted signatures
external ABI must not expose VMI layout
recursive SCC requires fixed-point signature layout
```

当前实现支持 direct `func.call` 到同一 module 内带 body 的 `func.func`：

```text
call operand[i] == callee argument[i]
call result[i] == every callee return operand[i]
same-result-index return operands inside one callee are equivalent
```

如果携带 VMI type 的 call 无法解析到带 body 的 direct callee，layout assignment 必须报
`VMI-LAYOUT-CONTRACT`。后续如需支持 public/external ABI，必须先定义 VMI 值如何在 ABI
边界 materialize，不能把 layouted VMI type 暴露出去。
当前实现明确拒绝携带 VMI type 的 `func.call_indirect`，因为它没有可解析的 direct internal
callee signature/body 可参与 layout constraint solving。

当前实现对携带 VMI type 的 external function declaration 报 `VMI-LAYOUT-CONTRACT`，因为还没有
定义 VMI value 的外部 ABI materialization plan。没有 VMI type 的 external declaration 必须在
`rewriteFunctionType` 中保持原签名，不能因为没有 entry block arguments 被改写成空签名。

`ptoas` 的默认 VPTO/VMI pipeline 拒绝 public `func.func` 的 VMI-typed signature：

```text
VMI-LAYOUT-CONTRACT: public VMI typed function requires an explicit external ABI materialization plan
```

这样 test-opt 仍可覆盖 internal/private function signature physicalization，用户入口则不会把
layout-assigned VMI 值隐式暴露成 public ABI。

Slice 3 完成条件：

```text
1. All VMI values have layout-assigned types after the pass.
2. All masks have b8/b16/b32 granularity after the pass.
3. CFG and call tests prove branch/yield/signature layout equality.
4. Multi-use rematerializable producer tests prove broadcast, constant, iota,
   create_mask, and constant_mask rematerialization vs ensure_layout /
   ensure_mask_* is deterministic.
5. The pass runs the layout-assigned VMI hard gate before returning, including
   recursive TypeAttr/TypedAttr rejection; covered by
   vmi_layout_assignment_post_gate_type_attr_invalid.pto.
```

## 6. Slice 4: `vmi-to-vpto`

推荐实现为 pass：

```text
recommended pass name: vmi-to-vpto
anchor: ModuleOp
source file: lib/PTO/Transforms/VMIToVPTO.cpp
```

第一步实现必须先落地 MLIR OneToN conversion 框架：

```text
VMIToVPTOTypeConverter : OneToNTypeConverter:
  !pto.vmi.vreg<NxT, layout> -> ordered !pto.vreg<lanesPerPart x T> list
  !pto.vmi.mask<Nxb8/b16/b32, layout> -> ordered !pto.mask<G> list

Structural patterns:
  populateFuncTypeConversionPatterns
  scf::populateSCFStructuralOneToNTypeConversions
  project-local OneToN patterns for cf.br/cf.cond_br/cf.switch
  project-local OneToN patterns for scf.execute_region/scf.index_switch

VMI patterns:
  OneToNOpConversionPattern for pack/unpack/ensure_*/semantic ops

Final residual gate:
  reject pto.vmi.*, !pto.vmi.*, unrealized_conversion_cast
  scan SSA types, block argument types, function signatures, and op/module TypeAttr or TypedAttr payloads
```

这一步可以先支持 type-only physicalization 和 `pack/unpack` helper physicalization，但不能让未实现的 VMI semantic op 静默通过。
如果还有 `pto.vmi.*` 或 VMI type 残留，必须报 `VMI-RESIDUAL-OP`。

当前 slice 支持 VMI function/input/block argument 展开成 physical arguments，并支持：

```text
pto.vmi.unpack(layouted VMI aggregate) -> physical parts:
  replace with OneToN adaptor source parts

pto.vmi.pack(physical parts) -> layouted VMI aggregate:
  replace with the physical parts through resultMapping

pto.vmi.ensure_layout / ensure_mask_layout / ensure_mask_granularity:
  ensure_layout must compare the original VMI source/result layout attrs, not only the converted physical type list.
  If source/result layouts are identical, replace with source parts. This identity case supports partial/tail physical
  chunks because no lane reordering or packing is performed.
  If deinterleaved=2 -> contiguous, emit one pto.vintlv.
  If contiguous -> deinterleaved=2, emit one pto.vdintlv.
  If deinterleaved=4 -> contiguous, emit the two-level pto.vintlv tree.
  If contiguous -> deinterleaved=4, emit the reverse two-level pto.vdintlv tree.
  ensure_mask_layout supports the same contiguous <-> deinterleaved=2/4 layout conversions with predicate
  rearrange ops:
    deinterleaved=2 -> contiguous: pto.pintlv_b8/b16/b32
    contiguous -> deinterleaved=2: pto.pdintlv_b8/b16/b32
    deinterleaved=4 -> contiguous: two-level pto.pintlv_b8/b16/b32 tree
    contiguous -> deinterleaved=4: two-level pto.pdintlv_b8/b16/b32 tree
  ensure_mask_granularity supports concrete b8/b16/b32 logical predicate-preserving conversion:
    widening b8 -> b16 -> b32: split each physical chunk with pto.punpack LOWER/HIGHER
    narrowing b32 -> b16 -> b8: pack physical chunk pairs with pto.ppack LOWER/HIGHER and merge halves with pto.por
    b8 <-> b32 conversions are lowered as two adjacent steps through b16.

pto.vmi.broadcast:
  current direct lowering requires the physical result element width to be 8,
  16, or 32 bits, because the vdup is predicated by pto.mask<b8/b16/b32>.
  Other semantic element types need a dedicated materialization contract before
  vmi-to-vpto may lower them.
  for each physical result part:
    materialize pto.pset_b8/b16/b32 "PAT_ALL" from the physical result element width
    emit pto.vdup(scalar, all_true_mask)
  This is layout-independent because every logical lane has the same scalar value. A deinterleaved layout simply
  receives one identical vdup per partition/chunk; no vintlv/vdintlv is needed.

pto.vmi.iota:
  semantics:
    ASC:  result[lane] = base + lane
    DESC: result[lane] = base - lane
  supported element types follow pto.vci:
    integer 8/16/32 and f16/f32
  contiguous full-chunk direct path:
    for each physical chunk c:
      chunk_base = base +/- c * lanes_per_part
      emit pto.vci chunk_base {order = ASC|DESC}
  deinterleaved layout requires strided index materialization because physical part p contains logical lanes:
    p, p + factor, p + 2 * factor, ...
  The required formula is:
    ASC:  base + p + factor * local_lane
    DESC: base - p - factor * local_lane
  The current lowering materializes this per physical chunk:
    local = pto.vci 0
    scaled = pto.vmuls local, factor
    ASC:  result = pto.vadds scaled, base + part_offset
    DESC: result = pto.vsub pto.vdup(base - part_offset), scaled
  Partial/tail chunks are allowed. The physical padding lanes receive the natural continuation of the generated iota
  sequence and remain padding/undef at the VMI semantic level; memory writes, masks, reductions, and other
  externally-visible consumers must still obey the VMI padding rules.

pto.vmi.constant_mask:
  support dense bool constants for concrete b8/b16/b32 masks. For each physical chunk:
    if the active lanes form a prefix:
      emit pto.pset_b8/b16/b32 PAT_ALL, PAT_ALLF, or supported PAT_VL*
      if a prefix count has no supported PAT_VL token, fall back to pto.plt_b8/b16/b32 with a constant i32 count
    otherwise decompose the static bitset into active runs:
      run [lo, hi) = prefix(hi) & ~prefix(lo)
      combine runs with pto.por under an all-true predicate
  pred-only masks remain unsupported until they have a concrete b8/b16/b32 consumer granularity.

pto.vmi.mask_and / mask_or / mask_xor / mask_not:
  for each physical predicate part:
    materialize pto.pset_b8/b16/b32 "PAT_ALL" from the physical mask granularity
    mask_and emits pto.pand(lhs_part, rhs_part, all_true_mask)
    mask_or emits pto.por(lhs_part, rhs_part, all_true_mask)
    mask_xor emits pto.pxor(lhs_part, rhs_part, all_true_mask)
    mask_not emits pto.pnot(source_part, all_true_mask)

pto.vmi.addf / addi / subf / subi / mulf / muli / divf / minf / maxf / negf / absf / absi / sqrt / exp / ln / relu / andi / ori / xori / shli / shrui / shrsi / not:
  current direct lowering requires the physical element width to be 8, 16, or
  32 bits, because every emitted VPTO op is predicated by a materialized
  pto.mask<b8/b16/b32>. VMI types such as index or f64 remain valid semantic
  surface types only after a dedicated lowering contract exists; until then
  vmi-to-vpto must report VMI-UNSUPPORTED before OneToN conversion.
  This common predicate-maskability rule is necessary but not sufficient for
  every target op. Direct lowering must also preflight the concrete VPTO/VISA
  element contract before OneToN rewriting:
    addf/subf/mulf -> pto.vadd/vsub/vmul support f16/bf16/f32 floating types
    divf -> pto.vdiv supports f16/f32 floating types
    minf/maxf -> pto.vmin/vmax support f16/bf16/f32 floating types
    negf/absf/sqrt/exp/ln/relu -> pto.vneg/vabs/vsqrt/vexp/vln/vrelu support f16/f32 floating types
    absi -> pto.vabs supports signless/signed i8/i16/i32 integer types
  bf16/f8 remain legal VMI float-like semantic types for the ops whose VMI
  semantics allow them, but vmi-to-vpto must report VMI-UNSUPPORTED until a
  materialization plan or wider target contract exists.
  for each physical part:
    materialize pto.pset_b8/b16/b32 "PAT_ALL" from the physical element width
    addf/addi emit pto.vadd(lhs_part, rhs_part, all_true_mask)
    subf/subi emit pto.vsub(lhs_part, rhs_part, all_true_mask)
    mulf/muli emit pto.vmul(lhs_part, rhs_part, all_true_mask)
    divf emits pto.vdiv(lhs_part, rhs_part, all_true_mask)
    minf emits pto.vmin(lhs_part, rhs_part, all_true_mask)
    maxf emits pto.vmax(lhs_part, rhs_part, all_true_mask)
    negf emits pto.vneg(source_part, all_true_mask)
    absf/absi emit pto.vabs(source_part, all_true_mask)
    sqrt emits pto.vsqrt(source_part, all_true_mask)
    exp emits pto.vexp(source_part, all_true_mask)
    ln emits pto.vln(source_part, all_true_mask)
    relu emits pto.vrelu(source_part, all_true_mask)
    andi emits pto.vand(lhs_part, rhs_part, all_true_mask)
    ori emits pto.vor(lhs_part, rhs_part, all_true_mask)
    xori emits pto.vxor(lhs_part, rhs_part, all_true_mask)
    shli emits pto.vshl(lhs_part, rhs_part, all_true_mask)
    shrui emits pto.vshr(lhs_part, rhs_part, all_true_mask)
    shrsi emits signed pto.vshr(lhs_part, rhs_part, all_true_mask)
    not emits pto.vnot(source_part, all_true_mask)

pto.vmi.fma:
  semantic:
    result = fused_multiply_add(lhs, rhs, acc)
    It must not be decomposed to pto.vmi.mulf + pto.vmi.addf because VPTO VMULA
    may produce different floating-point results from separate multiply and add.
  layout assignment:
    lhs, rhs, acc, and result belong to one data layout equivalence class.
  verifier contract:
    source/result element type must be f16, bf16, or f32
  current direct lowering:
    for each physical part:
      materialize pto.pset_b16/b32 "PAT_ALL" from the physical element width
      emit pto.vmula(acc_part, lhs_part, rhs_part, all_true_mask)
    The VMI operand order is lhs, rhs, acc; the VPTO operand order is acc, lhs, rhs.

pto.vmi.cmpf / cmpi:
  verifier contract:
    source/result element width must be 8/16/32-bit so the result predicate
    can be materialized as b8/b16/b32.
    cmpf: f16/bf16/f32, matching VISA VCMP floating-point element types
    cmpi: signless/signed/unsigned i8/i16/i32, matching VISA VCMP integer element types
  for each physical part:
    materialize pto.pset_b8/b16/b32 "PAT_ALL" as the seed predicate
    canonicalize predicate to VPTO cmp_mode eq/ne/lt/le/gt/ge
    emit pto.vcmp(lhs_part, rhs_part, seed_mask, cmp_mode)
  supported cmpf predicates:
    eq/ne/lt/le/gt/ge pass through
    oeq -> eq
    one -> ne
    olt -> lt
    ole -> le
    ogt -> gt
    oge -> ge
  supported cmpi predicates:
    eq/ne pass through
    ult -> lt on unsigned integer carriers
    ule -> le on unsigned integer carriers
    ugt -> gt on unsigned integer carriers
    uge -> ge on unsigned integer carriers
    slt -> lt
    sle -> le
    sgt -> gt
    sge -> ge
  if the physical vreg element signedness does not match the predicate, insert pto.vbitcast to the matching
  si/ui integer carrier before pto.vcmp.
  unsupported cmpi bare relational predicates lt/le/gt/ge must emit VMI-UNSUPPORTED because integer signedness
  must be explicit.
  unsupported floating-point predicates such as ord/uno/ult/ule/ugt/uge must emit VMI-UNSUPPORTED until NaN-aware
  predicate construction is designed.

pto.vmi.vcmp / vcmps:
  unified vmi_new integer compare uses type-driven signedness:
    signed/signless integer element types map lt/le/gt/ge to legacy slt/sle/sgt/sge
    unsigned integer element types map lt/le/gt/ge to legacy ult/ule/ugt/uge
    explicit integer predicates slt/sle/sgt/sge/ult/ule/ugt/uge are rejected at the unified op level
  legacy cmpi remains predicate-driven and requires explicit signed/unsigned relational forms.

pto.vmi.active_prefix_index:
  semantic:
    idx[i] = popcount(mask[0 .. i))
  result element type must be signless i8/i16/i32, and concrete mask granularity must match the result element width.
  current direct lowering:
    only contiguous layout
    only one physical result/mask chunk
    result and mask chunks must be full, with no padding logical lanes
    materialize a zero vreg carrier with pto.vdup
    emit pto.vusqz(carrier, mask)
  unsupported cases:
    partial/tail chunks because padding mask lanes could affect the observable prefix
    multi-chunk contiguous values need cross-chunk prefix carry
    deinterleaved layouts need logical-lane-order prefix reconstruction
    both must report VMI-UNSUPPORTED before OneToN conversion

pto.vmi.compress:
  semantic:
    keep source lanes whose mask lane is true and compact them in logical lane order; inactive tail lanes are zero/undef
    at the VMI semantic level unless consumed by an operation that defines them.
  current direct lowering:
    source/result/mask must be contiguous
    source/result/mask must each materialize to one physical chunk
    source chunk must be full, with no padding logical lanes
    emit pto.vsqz(source, mask)
  unsupported cases:
    partial/tail chunks because padding mask lanes could be squeezed into the observable result prefix
    multi-chunk values need cross-chunk compaction and SQZN/carry planning
    deinterleaved layouts need logical-lane-order compaction before physical part placement
    compress_store is not implied by register compress; store-coupled VSQZ #st=1 and VSTUR require a separate
    producer/consumer pairing plan

pto.vmi.compress_store:
  semantic:
    store source lanes whose mask lane is true as a dense logical memory stream:
      k = 0
      for lane in logical order:
        if mask[lane]:
          base[offset + k] = value[lane]
          k += 1
  layout assignment:
    value use is requested as contiguous
    mask use is requested as contiguous with granularity derived from value element width
  current direct lowering:
    value and mask must be contiguous
    value and mask must each materialize to one physical chunk
    the value chunk must be full, with no padding logical lanes
    destination must be a UB !pto.ptr because pto.vstur is pointer-only and UB-only
    lower as:
      store_base = pto.addptr destination, offset
      squeezed = pto.vsqz(value, mask)
      align0 = pto.init_align
      align1 = pto.vstur align0, squeezed, store_base, "POST_UPDATE"
      pto.vstar align1, store_base
    The pto.vstur user is the required consumer that lets the VPTO LLVM emitter
    set VSQZ #st=1. A plain register pto.vsqz must not be assumed to enqueue
    SQZN for store.
  unsupported cases:
    memref or GM destination until an explicit pointer/materialization plan exists
    partial/tail physical chunks, because padding mask lanes could be squeezed into memory
    multi-chunk values, because they need cross-chunk active-count compaction and SQZN/VSTUR state planning
    deinterleaved layouts, because compaction must be in logical lane order

pto.vmi.reduce_addi:
  semantic:
    acc = init[0]
    for lane in logical order:
      if mask[lane]:
        acc = acc + source[lane]    // integer wraparound addition
    result[0] = acc
  layout assignment:
    source use is requested as contiguous
    init use is requested as contiguous
    result natural layout is contiguous
    mask use is requested as contiguous with granularity derived from source element width
  current direct lowering:
    source element width must be 32 bits; narrower vcadd widens its result and needs a separate result type plan
    source must materialize to one or more full physical chunks with no padding logical lanes
    init/result must be 1-lane VMI vectors and each materialize to one physical chunk
    mask must materialize to the same number of physical chunks as source
    when all physical mask chunks are equivalent, lower as:
      first_lane = pto.pge_b32 "PAT_VL1"
      combined = source_chunks[0]
      for each remaining source_chunk:
        combined = pto.vadd(combined, source_chunk, common_mask)
      reduced = pto.vcadd(combined, common_mask)
      result = pto.vadd(reduced, init, first_lane)
    otherwise preserve the per-chunk fallback:
      first_lane = pto.pge_b32 "PAT_VL1"
      acc = init
      for each source_chunk, mask_chunk in physical order:
        reduced = pto.vcadd(source_chunk, mask_chunk)
        acc = pto.vadd(reduced, acc, first_lane)
      result = acc
  unsupported cases:
    i8/i16 until widening result and init conversion are designed
    partial/tail source chunks because padding lanes must not participate
    floating-point add reduction without pto.vmi.reduce_addf {reassoc}

pto.vmi.reduce_addf:
  semantic:
    requires {reassoc}; without it the verifier rejects the op
    acc = init[0]
    for lane in any reassociated tree over active logical lanes:
      acc = acc + source[lane]
    result[0] = acc
  layout assignment:
    source use is requested as contiguous
    init use is requested as contiguous
    result natural layout is contiguous
    mask use is requested as contiguous with granularity derived from source element width
  current direct lowering:
    source element type must be f32
    source must materialize to one or more full physical chunks with no padding logical lanes
    init/result must be 1-lane VMI vectors and each materialize to one physical chunk
    mask must materialize to the same number of b32 physical chunks as source
    when all physical mask chunks are equivalent, combine source chunks with
    pto.vadd under the common mask, execute one pto.vcadd, then combine the
    reduced value with init under PAT_VL1
    otherwise preserve the per-chunk fallback:
      first_lane = pto.pge_b32 "PAT_VL1"
      acc = init
      for each source_chunk, mask_chunk in physical order:
        reduced = pto.vcadd(source_chunk, mask_chunk)
        acc = pto.vadd(reduced, acc, first_lane)
      result = acc
  unsupported cases:
    missing reassoc attr
    f16 until accumulator precision and rounding contract are designed
    partial/tail source chunks because padding lanes must not participate

pto.vmi.group_load / pto.vmi.group_store:
  semantic:
    num_groups is the only static grouping attribute.
    N = logical lane count; G = num_groups; S = N / G.
    group_load reads each logical group as one contiguous row:
      result[g * S + i] = source[offset + g * row_stride + i]
      for 0 <= g < G and 0 <= i < S
    group_store writes the inverse row mapping:
      destination[offset + g * row_stride + i] = value[g * S + i]
    row_stride is an index operand, measured in elements, and may be dynamic.
    Tail/valid-lane information is not an attr; it must be represented by a
    mask in the producing/consuming computation. Direct group_store also
    supports a contiguous tail when each logical group is exactly one 32B
    block; lowering creates a prefix predicate for the final physical vreg.
  layout assignment:
    group_load result natural layout is contiguous
    group_store value use is requested as contiguous
  current direct lowering:
    source/value element width must be maskable by b8/b16/b32
    for contiguous group_store with S equal to one 32B VCG block and a static
    positive 32B-aligned row_stride:
      block_stride = row_stride / S, measured in 32B blocks
      lower each physical vreg with pto.vsstb and repeat_stride = 0
      the physical-part base is offset + first_group_in_part * row_stride
      block_stride must fit the 16-bit pto.vsstb control field
    for contiguous full physical chunks, num_groups must evenly divide N and S
    must be a multiple of the physical lanes per part; lower each chunk with
    pto.vlds/pto.vsts at:
      offset + group * row_stride + chunk_in_group * lanes_per_part
    deinterleaved=2 full chunks lower through pto.vstsx2
  unsupported cases:
    derived group sizes smaller than one 32B block or between one block and a
    full physical vreg, because these need partial-vreg extraction or a more
    general gather/scatter plan
    dynamic or non-32B-aligned row_stride for the one-block pto.vsstb path
    GM-backed direct vector load/store paths not already accepted by the normal
    VMI memory access plan

pto.vmi.group_reduce_addf:
  semantic:
    requires {reassoc}
    N = logical lane count; G = num_groups; S = N / G
    L = physical lanes per 256B chunk for the element type.
    The result carries #pto.vmi.layout<num_groups = G, slots = K>, a group-slot
    group-slot layout. It is not a dense vector layout: only slot lanes have
    semantic values.  Supported K values are:
      K = 8 for VCGADD-style packed results, where group g is stored in
      physical chunk floor(g / 8), lane g % 8.
      K = 1 for row-local VCADD results, where group g is stored in physical
      chunk g, lane 0.
    for each group g:
      result[group_slot(g)] =
          reduce_add(source[g * S .. (g + 1) * S), mask in same range)
    Non-slot lanes are not consumed by pto.vmi.group_broadcast. The current
    direct lowering materializes them as zero where the hardware path does not
    already define them.
    The result remains a VMI vector with the same element type as the source,
    but its logical lane count is G: one scalar result per group.  Its layout
    is an explicit group-slot layout that describes where those G scalars are
    placed in physical registers.
  layout assignment:
    source use is requested as contiguous
    result natural layout is #pto.vmi.layout<num_groups = G, slots = K>
    mask use is requested as contiguous with granularity derived from source
    element width
  current direct lowering:
    source/result element type must be f16 or f32
    source and mask must have compatible full physical chunks. The result is
    `GxT` group-slot data and may have different physical arity from the
    source tile.
    if S=8 for f32, lower each physical chunk with pto.vcgadd. This is the
    hardware 32B VLane group reduction path for f32: each source chunk produces
    eight 8-lane group sums in the low lanes of that physical chunk. The
    lowering preserves this natural no-pack result.
    for multi-part packed and row-local groups whose physical masks are
    equivalent, combine corresponding source parts with pto.vadd under the
    common mask before executing one pto.vcgadd or pto.vcadd per result group
    Otherwise:
    derived group size S must be a multiple of physical lanes per part
    lower each source chunk with pto.vcadd, combine chunks in the same group
    with pto.vadd under PAT_VL1, then place group g in the slot lane defined by
    K. All other result chunks/lane values
    are zero.
  unsupported cases:
    missing reassoc attr
    integer element types, which use the corresponding typed integer op
    derived group size S that neither divides nor is a multiple of L

pto.vmi.group_reduce_addi / group_reduce_maxi / group_reduce_mini:
  semantic:
    source and result use the same i8/i16/i32 element type
    the result has one group-slot value per logical group
    integer addition has same-type wraparound semantics
  layout assignment:
    use the same registered group-block table as floating-point group reduction
    packed 32B-block cases use slots=8
    aligned full-row cases use slots=1
  current direct lowering:
    packed cases use pto.vcgadd/pto.vcgmax/pto.vcgmin and same-type combines
    aligned full-row max/min cases use pto.vcmax/pto.vcmin
    aligned full-row i8/i16 add cases use widening pto.vcadd partials and
    widened pto.vadd combines, then pto.vbitcast the low bits back to the
    declared VMI result type
    when the physical masks for one result group are equivalent, add/min/max
    all combine the source parts elementwise first and execute one group or row
    reduction; narrow integer add combines in the source type before the
    widening pto.vcadd and final pto.vbitcast
    the widening is internal and is not exposed in the VMI type contract
  unsupported cases:
    element types other than i8/i16/i32
    group sizes outside the registered high-performance group-block classes

pto.vmi.group_broadcast:
  semantic:
    source logical lane count is G; result logical lane count is N.
    S = N / G.
    source must carry #pto.vmi.layout<num_groups = G, slots = K>. For each
    group g, the source value is read from the slot lane defined by K. The
    result broadcasts it back to each logical group:
      result[g * S + i] = source[group_slot(g)]
  layout assignment:
    source use is requested as #pto.vmi.layout<num_groups = G, slots = K>
    result is consumer-driven. If no consumer requests another layout, it
    defaults to contiguous.
  current direct lowering:
    source must carry #pto.vmi.layout<num_groups = G, slots = K> with one
    logical lane per group
    result may be contiguous with full physical chunks
    result may also be deinterleaved when S is large enough that every physical
    result chunk stays inside one logical group, for example N=512, G=2, S=256,
    L=64, deinterleaved=4.  If the source is
    #pto.vmi.layout<num_groups = G, slots = 1>, the source physical part is
    selected by group id rather than by source chunk id.
    derived group size S must divide or be a multiple of L for canonical
    group-slot addressing
    if result is contiguous and S < L, each physical chunk contains multiple group
    slots. Lower by
    creating an index vector [0...0, 1...1, ...] and applying pto.vselr to the
    corresponding source chunk.
    if S >= L and each result physical chunk belongs to one group, lower by
    duplicating the first lane of that group's source chunk with pto.vdup LOWEST.
  unsupported cases:
    partial/tail physical chunks
    derived group size S that neither divides nor is a multiple of L
    deinterleaved small-group broadcast where one physical result chunk needs
    values from multiple source chunks

pto.vmi.reduce_maxf / reduce_minf / reduce_maxi / reduce_mini:
  semantic:
    acc = init[0]
    for each active logical lane in logical lane order:
      reduce_max*: acc = max(acc, source[lane])
      reduce_min*: acc = min(acc, source[lane])
    result[0] = acc
    inactive lanes inside each physical chunk follow VPTO identities:
      reduce_maxf uses pto.vcmax, where inactive FP lanes behave as -INF
      reduce_minf uses pto.vcmin, where inactive FP lanes behave as +INF
    NaN and signed-zero behavior follows pto.vcmax/pto.vcmin for the chunk
    reduction and pto.vmax/pto.vmin for serial chunk accumulation. The index
    lane produced by pto.vcmax/pto.vcmin is ignored because VMI exposes only the
    1-lane value result.
  layout assignment:
    source use is requested as contiguous
    init use is requested as contiguous
    result natural layout is contiguous
    mask use is requested as contiguous with granularity derived from source element width
  current direct lowering:
    source element type must be f16/f32 for the floating ops or i8/i16/i32 for
    the integer ops
    source must materialize to one or more full physical chunks with no padding logical lanes
    init/result must be 1-lane VMI vectors and each materialize to one physical chunk
    mask must materialize to the same number of physical chunks as source
    when all physical mask chunks are equivalent, lower reduce_maxf as:
      combined = source_chunks[0]
      for each remaining source_chunk:
        combined = pto.vmax(combined, source_chunk, common_mask)
      reduced = pto.vcmax(combined, common_mask)
      result = pto.vmax(reduced, init, first_lane)
    use the analogous pto.vmin/pto.vcmin chain for reduce_minf
    otherwise preserve the per-chunk fallback, shown for reduce_maxf:
      first_lane = pto.pge_b16/b32 "PAT_VL1"
      acc = init
      for each source_chunk, mask_chunk in physical order:
        reduced = pto.vcmax(source_chunk, mask_chunk)
        acc = pto.vmax(reduced, acc, first_lane)
      result = acc
  unsupported cases:
    bf16/fp8/f64 until VPTO reduction and combine semantics are designed
    partial/tail source chunks because padding lanes must not participate
    integer widths other than i8/i16/i32

pto.vmi.select:
  current direct lowering is a storage-width select rather than a semantic
  arithmetic op: source/result physical elements must be b8/b16/b32-maskable,
  but signedness and float-vs-integer interpretation are not inspected.
  for each physical part:
    consume the corresponding physical predicate part
    emit pto.vsel(true_part, false_part, predicate_part)

pto.vmi.extf, direct path:
  support 16-bit float-like contiguous source part -> f32 deinterleaved=2 result parts
    materialize pto.pset_b16 "PAT_ALL"
    emit pto.vcvt(source_part, mask, part=EVEN/ODD)
    partial/tail is valid when the logical lanes fit in the one physical source
    part; PAT_ALL may convert padding lanes, but those lanes remain padding in
    the deinterleaved result
  support 8-bit contiguous source part -> f32 deinterleaved=4 result parts
    materialize pto.pset_b8 "PAT_ALL"
    emit pto.vcvt(source_part, mask, part=P0/P1/P2/P3)
    the same padding rule applies
  reject other extf width/layout shapes until their exact part plan is implemented

pto.vmi.truncf, direct path:
  support f32 deinterleaved=2 source parts -> 16-bit contiguous result part
    materialize pto.pset_b32 "PAT_ALL" for the source conversion
    emit pto.vcvt(even_f32_part, mask, rnd=R, sat=SAT, part=EVEN)
    emit pto.vcvt(odd_f32_part, mask, rnd=R, sat=SAT, part=ODD)
    materialize pto.pset_b16 "PAT_ALL"
    merge mutually exclusive part results with pto.vor
    partial/tail is valid when the two source parts pack into one physical
    result part; converted padding lanes remain result padding
  support f32 deinterleaved=4 source parts -> 8-bit contiguous result part
    materialize pto.pset_b32 "PAT_ALL" for the source conversion
    emit pto.vcvt(p0_f32_part, mask, rnd=<result round>, sat=SAT, part=P0)
    emit pto.vcvt(p1_f32_part, mask, rnd=<result round>, sat=SAT, part=P1)
    emit pto.vcvt(p2_f32_part, mask, rnd=<result round>, sat=SAT, part=P2)
    emit pto.vcvt(p3_f32_part, mask, rnd=<result round>, sat=SAT, part=P3)
    result round is R for f8E4M3/f8E5M2, A for default hif8, or H for
      hif8 truncf with {rounding = "H"}
    materialize pto.pset_b8 "PAT_ALL"
    merge mutually exclusive part results with pto.vor
    partial/tail is valid when the four source parts pack into one physical
    result part; converted padding lanes remain result padding
  reject other truncf width/layout shapes until their exact pack plan is implemented

pto.vmi.bitcast:
  for each physical part:
    emit pto.vbitcast(source_part) -> result_part_type
  source/result layouts must match, physical arity must match, and every
  corresponding physical chunk must carry the same number of logical bits.
  This includes contiguous, deinterleaved, and identical group_slots layouts.
  Padding bits may map only to result padding bits; any shape where source
  padding would become result logical data remains unsupported.

pto.vmi.channel_split / pto.vmi.channel_merge:
  support 2-way and 4-way channel transforms for contiguous per-channel values
  and matching deinterleaved=C merged values.

  channel_split C=2:
    if the source layout is already deinterleaved=2, forward physical chunks
    directly to the two contiguous channel results.
    if the source layout is contiguous, source logical vector must physicalize
    as 2*N contiguous chunks. For each pair of dense chunks:
      %ch0_i, %ch1_i = pto.vdintlv %dense_2i, %dense_2i_plus_1
    Results are returned in per-channel order:
      channel0 chunks..., channel1 chunks...

  channel_split C=4:
    if the source layout is already deinterleaved=4, forward physical chunks
    directly to the four contiguous channel results.
    if the source layout is contiguous, source logical vector must physicalize
    as 4*N contiguous chunks. The lowering is the same two-level pto.vdintlv
    tree used by contiguous -> deinterleaved=4 materialization, but the
    partition-major output is interpreted as four separate contiguous channel
    results.

  channel_merge C=2/C=4:
    inputs are consumed as per-channel contiguous chunks.
    If the result layout is deinterleaved=C, the physical chunks are forwarded
    directly in partition-major order.
    If the result layout is contiguous, the lowering uses the reverse
    pto.vintlv tree and returns dense contiguous chunks for the merged result.

  Unsupported:
    channel counts other than 2 or 4
    non-matching channel input/result layouts
    arity-changing or uneven partial physical channel groups that cannot form
    complete intlv/dintlv groups

pto.vmi.shuffle:
  first try whole physical chunk forwarding cases:
    source/result layouts are assigned
    every non-padding lane in a result physical chunk maps to the same source physical chunk
    source lane number equals result lane number inside the physical chunk
    result padding lanes are ignored and remain semantically unobservable

  If forwarding fails, try vci-materializable vselr per physical chunk:
    every result physical chunk has no padding lane
    every lane in a result physical chunk maps to the same source physical chunk
    source lane indices inside the chunk form one ASC or DESC consecutive sequence
    materialize the index vector with pto.vci(base_lane, ASC|DESC)
    emit pto.vselr(source_chunk, index_vector)

  Examples:
    identity 128xf32 -> 128xf32:
      indices = [0, 1, ..., 127]
      forward dense chunks 0 and 1

    second physical chunk 128xf32 -> 64xf32:
      indices = [64, 65, ..., 127]
      forward dense chunk 1

    tail prefix 128xf32 -> 4xf32:
      indices = [0, 1, 2, 3]
      forward dense chunk 0
      lanes 4..63 of the physical result are padding lanes and are not part of
      the logical vmi value

    chunk swap 128xf32 -> 128xf32:
      indices = [64, 65, ..., 127, 0, 1, ..., 63]
      forward dense chunks in order 1, 0

    reverse one 64xf32 chunk:
      indices = [63, 62, ..., 0]
      index = pto.vci 63 {order = DESC} : i32 -> !pto.vreg<64xi32>
      result = pto.vselr source_chunk, index

  Unsupported:
    partial physical chunk projection whose observable result lanes are not
    padding-safe forwarding, e.g. [1, 2, 3, 4] -> 4xf32 when it would require
    shifting lanes rather than forwarding a whole physical chunk
    broadcast, duplicate lanes, arbitrary non-affine permutation
    current implementation emits VMI-UNSUPPORTED for these cases before
    OneToN conversion, instead of leaving a generic residual VMI op.
```

`func.return` 携带 VMI operand 时必须通过 OneToN func/return structural pattern 展开成 physical
return operands。不能只取第一个 physical part；这种错误会导致函数类型已经返回两个 physical value，
但 `func.return` 只返回一个 value。

### 6.1 Type Conversion

Use one shared physicalization helper:

```text
VMIVRegType -> N physical !pto.vreg<lanesPerPart x T>
VMIMaskType -> N physical !pto.mask<G>
```

Physical result ordering must be:

```text
contiguous:
  chunk0, chunk1, ...

deinterleaved=K:
  p0_chunk0, p0_chunk1, ..., p1_chunk0, ..., p(K-1)_chunkN
```

### 6.2 Structural Conversion

The pass must convert:

```text
operation results
block arguments
branch operands
cf.br / cf.cond_br successor block signatures
scf.if results and yields
scf.for iter_args and yields
func arguments/results
call operands/results
return operands
cf.br / cf.cond_br / cf.switch block arguments and successor operands
scf.execute_region results and yields:
  current implementation uses a project-local OneToN structural pattern.
scf.index_switch results and yields:
  current implementation uses a project-local OneToN structural pattern.
```

Do not rely on a defining op to recover parts. Any VMI value may come from a block argument or function
argument, so `unpack` must be valid on arbitrary layout-assigned VMI SSA values before final lowering.

### 6.3 Op Lowering

Internal helper lowering：

```text
unpack:
  replace with physical values in helper ordering

pack:
  materialize one logical VMI aggregate before it is immediately consumed by another VMI helper
  must not remain after final gate

ensure_layout:
  preflight:
    source/result must have computable physical arity
    source/result physical arity must match
    identity source/result layouts do not require full chunks
    if source/result layouts differ, either:
      every source/result physical chunk is full, with no padding lanes; or
      source/result both have complete contiguous/deinterleaved=2/4 materialization groups and their materialized
      physical arity still equals the original VMI physical arity
    arity-changing partial/tail layout conversion remains unsupported because it would need an explicit padding
    packing/drop plan
    otherwise report VMI-UNSUPPORTED before OneToN conversion

  compare the original VMI source/result layout attrs:
    same layout:
      forward the converted source parts
    deinterleaved=2 -> contiguous:
      %d0, %d1 = pto.vintlv %p0, %p1
    contiguous -> deinterleaved=2:
      %p0, %p1 = pto.vdintlv %d0, %d1
    deinterleaved=4 -> contiguous:
      %a0, %a1 = pto.vintlv %p0, %p2
      %b0, %b1 = pto.vintlv %p1, %p3
      %d0, %d1 = pto.vintlv %a0, %b0
      %d2, %d3 = pto.vintlv %a1, %b1
    contiguous -> deinterleaved=4:
      %a0, %b0 = pto.vdintlv %d0, %d1
      %a1, %b1 = pto.vdintlv %d2, %d3
      %p0, %p2 = pto.vdintlv %a0, %a1
      %p1, %p3 = pto.vdintlv %b0, %b1

  It is a bug to treat layout conversion as identity merely because both sides convert to the same
  number of physical !pto.vreg values with the same type. For example:
    !pto.vmi.vreg<128xf32, deinterleaved=2>
    !pto.vmi.vreg<128xf32, contiguous>
  both physicalize to two !pto.vreg<64xf32> values, but their logical lane order differs.

ensure_mask_layout:
  preflight:
    source/result must have computable physical arity
    source/result physical arity must match
    if source/result layouts differ, every source/result physical predicate chunk must be full, with no padding lanes
    identity source/result layouts do not require full chunks
    otherwise report VMI-UNSUPPORTED before OneToN conversion

  same-layout:
    forward source parts
  deinterleaved=2 -> contiguous:
    use pto.pintlv_b8/b16/b32 on each partition pair
  contiguous -> deinterleaved=2:
    use pto.pdintlv_b8/b16/b32 on each dense pair
  deinterleaved=4 -> contiguous:
    use the same two-level tree as data layout conversion, replacing pto.vintlv with pto.pintlv_b8/b16/b32
  contiguous -> deinterleaved=4:
    use the reverse two-level tree, replacing pto.vdintlv with pto.pdintlv_b8/b16/b32
  source/result granularity must be identical; granularity conversion belongs to ensure_mask_granularity.

ensure_mask_granularity:
  source/result layout and logical lane count must match.
  source/result granularity must be concrete b8/b16/b32.
  identity conversion forwards physical parts.
  widening conversion:
    b8 -> b16 or b16 -> b32 uses pto.punpack LOWER/HIGHER for each source physical chunk.
    each source physical mask chunk can produce up to two result chunks in logical order.
  narrowing conversion:
    b32 -> b16 or b16 -> b8 uses pto.ppack LOWER for the low source chunk.
    if a high source chunk exists, use pto.ppack HIGHER and merge the two partial masks with pto.por under PAT_ALL.
    this handles odd tail groups because the missing high half is padding and remains zero.
  multi-step conversion:
    b8 -> b32 is b8 -> b16 -> b32.
    b32 -> b8 is b32 -> b16 -> b8.
```

Elementwise lowering：

```text
for each physical part:
  lower add/cmp/select to corresponding VPTO op sequence
  preserve source/result physical ordering
  cmp predicates must be canonicalized before creating pto.vcmp:
    cmpf eq/ne/lt/le/gt/ge pass through
    ordered FP aliases oeq/one/olt/ole/ogt/oge map to eq/ne/lt/le/gt/ge
    cmpi eq/ne pass through
    unsigned integer aliases ult/ule/ugt/uge map to lt/le/gt/ge on unsigned carriers
    signed integer aliases slt/sle/sgt/sge map to lt/le/gt/ge
    insert pto.vbitcast to si/ui integer carriers when the physical vreg element signedness does not match
    cmpi bare relational predicates lt/le/gt/ge are unsupported because integer signedness must be explicit
    unordered/NaN-sensitive FP predicates are unsupported until represented explicitly
```

Producer lowering：

```text
broadcast:
  TypeConverter gives the ordered result physical types.
  For each result physical vreg:
    create all-true mask with the vreg element width
    emit pto.vdup scalar -> that physical vreg

  This is valid for contiguous and deinterleaved layouts because splat has no lane-order dependence.

constant:
  Splat dense constants use the same path as broadcast:
    create scalar arith.constant from the splat attribute
    emit pto.vdup per physical result part
    require the same 8/16/32-bit physical result element-width precondition as
    broadcast
  Non-splat dense constants need an explicit constant materialization strategy or must remain unsupported with a
  precise diagnostic; do not synthesize an arbitrary lane sequence by scalar inserts unless that path is designed.

create_mask / constant_mask:
  constant active_lanes create_mask lowers per physical mask part:
    clamp active_lanes to [0, logical lane count]
    compute active prefix count for each physical mask chunk with the VMI lane-map helper
    emit pto.pge_b8/b16/b32 PAT_ALL, PAT_ALLF, or supported PAT_VL*
    if a chunk prefix count has no supported PAT_VL token, fall back to pto.plt_b8/b16/b32 with a constant i32 count
  Dynamic active_lanes with contiguous layout lowers by chaining pto.plt_b8/b16/b32 over the physical chunks:
    active_i32 = arith.index_cast active_lanes : index to i32
    active_i32 = minui(maxsi(active_i32, 0), logical_lane_count)
    mask0, remaining0 = pto.plt_b* active_i32
    mask1, remaining1 = pto.plt_b* remaining0
    ...
  Dynamic active_lanes with deinterleaved layout remaps one logical prefix into per-part dynamic lane counts before
  chaining pto.plt_b*:
    active_i32 = minui(maxsi(index_cast(active_lanes), 0), logical_lane_count)
    part_count(part) = (active_i32 + factor - 1 - part) / factor
    then chain pto.plt_b* independently for each partition in VMI physical order:
      p0 chunks..., p1 chunks..., ...
  dense constant_mask lowers per physical mask part:
    first map logical lanes to physical predicate lanes using the assigned VMI layout
    prefix chunks emit pto.pset_b8/b16/b32 PAT_ALL, PAT_ALLF, or supported PAT_VL*
    if a prefix count has no supported PAT_VL token, emit pto.plt_b8/b16/b32 with a constant i32 count
    non-prefix chunks are decomposed into static active runs:
      prefix(hi) = pto.pge/plt for the run end
      prefix(lo) = pto.pge/plt for the run begin
      run = prefix(hi) & ~prefix(lo) using pto.pnot + pto.pand
      chunk = run0 | run1 | ... using pto.por

Unsupported diagnostics:
  unexpected residual dynamic pto.vmi.create_mask after OneToN conversion:
    VMI-UNSUPPORTED: dynamic pto.vmi.create_mask active_lanes could not be lowered by the current runtime predicate
    generation plan
    This is a final-gate diagnostic for malformed or newly unsupported dynamic shapes. The supported dynamic
    contiguous/deinterleaved=2/deinterleaved=4 paths above must lower before this residual gate.

  non-splat pto.vmi.constant:
    VMI-UNSUPPORTED: non-splat pto.vmi.constant requires a vreg immediate or scratch materialization plan

  unsupported partial/tail masked/expand read-style op:
    VMI-UNSUPPORTED: pto.vmi.<memory-op> requires full physical chunks without padding lanes or a statically safe
    full-read footprint (...; safe-read proof failed: ...)
  GM-backed direct pto.vmi.load/masked_load/expand_load:
    VMI-UNSUPPORTED: pto.vmi.<memory-op> ... (source is GM-backed, but current direct VMI-to-VPTO memory lowering
    emits pto.vlds/pto.vsts and requires UB-backed memory)
  unsupported partial/tail pto.vmi.store/masked_store:
    VMI-UNSUPPORTED: pto.vmi.<store-op> requires an 8/16/32-bit predicate-maskable element type and either full
    physical chunks or contiguous/deinterleaved tail-store materialization, with UB-backed destination; unsupported
    cases include values such as f64/index that have no b64 predicate representation, GM-backed destinations that
    still need a memory movement/materialization plan, and uneven deinterleaved physical groups that cannot form
    complete intlv groups

  unsupported non-identity partial/tail pto.vmi.ensure_layout:
    VMI-UNSUPPORTED: pto.vmi.ensure_layout cannot materialize the requested data layout conversion; unsupported cases
    include arity-changing partial/tail conversion and uneven deinterleaved groups that cannot form complete intlv
    groups
    If the helper has a single consumer, the main diagnostic is emitted on the
    consumer op and operand, including both the actual operand VMI type and the
    required VMI type. For example, pto.vmi.truncf operand #0 can report
    `!pto.vmi.vreg<128xf32, contiguous>` vs.
    `!pto.vmi.vreg<128xf32, deinterleaved=4>` for f32->fp8. The failed
    pto.vmi.ensure_layout conversion is attached as a note.

  unsupported non-identity partial/tail pto.vmi.ensure_mask_layout:
    VMI-UNSUPPORTED: pto.vmi.ensure_mask_layout cannot materialize the requested mask layout conversion; unsupported
    cases include arity-changing partial/tail conversion and uneven deinterleaved groups that cannot form complete
    predicate intlv groups

  unsupported pto.vmi.ensure_mask_granularity:
    VMI-UNSUPPORTED: non-identity mask granularity materialization requires concrete b8/b16/b32 masks with matching
    lane count and layout (...)

  unsupported pto.vmi.extf direct path shape:
    VMI-UNSUPPORTED: pto.vmi.extf supports only one contiguous 16-bit float-like or fp8-like physical source chunk to f32
    deinterleaved=2/4 results; partial/tail is allowed only when source padding maps to result padding

  unsupported pto.vmi.truncf direct path shape:
    VMI-UNSUPPORTED: pto.vmi.truncf supports only f32 deinterleaved=2 source parts to one contiguous f16 result chunk
    or f32 deinterleaved=4 source parts to one contiguous fp8-like result chunk

  unsupported pto.vmi.bitcast shape:
    VMI-UNSUPPORTED: pto.vmi.bitcast requires matching source/result layouts with identical physical
    arity and matching per-chunk logical bit footprints (...)

  unsupported pto.vmi.channel_split / pto.vmi.channel_merge channel count:
    VMI-UNSUPPORTED: pto.vmi.channel_split supports only 2 or 4 channels
    VMI-UNSUPPORTED: pto.vmi.channel_merge supports only 2 or 4 channels
  unsupported pto.vmi.channel_split / pto.vmi.channel_merge layout:
    VMI-UNSUPPORTED: pto.vmi.channel_split requires source layout to be contiguous or matching deinterleaved channel
    layout, and every result layout to be contiguous
    VMI-UNSUPPORTED: pto.vmi.channel_merge requires every input layout to be contiguous and result layout to be
    contiguous or matching deinterleaved channel layout
```

Width conversion lowering：

```text
f16 -> f32:
  supported direct path when source is contiguous and result is deinterleaved=2:
    pto.vcvt part=EVEN produces logical lanes 0,2,4,...
    pto.vcvt part=ODD produces logical lanes 1,3,5,...
  source/result physical arity must be 1 -> 2

f8 -> f32:
  supported direct path when source is contiguous and result is deinterleaved=4:
    pto.vcvt part=P0/P1/P2/P3 produces the four modulo-4 lane partitions
  source/result physical arity must be 1 -> 4

f32 -> f16:
  supported direct path when source is deinterleaved=2 and result is contiguous:
    pto.vcvt part=EVEN consumes even/source part 0
    pto.vcvt part=ODD consumes odd/source part 1
    pto.vor merges mutually exclusive f16 part results into one contiguous vreg
  source/result physical arity must be 2 -> 1
  current default conversion attrs are rnd=R, sat=SAT

f32 -> 8-bit fp-like:
  supported direct path when source is deinterleaved=4 and result is contiguous:
    pto.vcvt part=P0/P1/P2/P3 consumes the four source partitions
    pto.vor merges mutually exclusive byte-lane part results into one
      contiguous vreg
  source/result physical arity must be 4 -> 1
  current default conversion attrs are rnd=R for f8E4M3/f8E5M2 and rnd=A for
    hif8. pto.vmi.truncf {rounding = "H"} is accepted only for f32 -> hif8
    and forwards rnd=H to the emitted pto.vcvt operations.
```

Memory lowering：

```text
vmi.load:
  current direct memory path first reads contiguous physical chunks. The logical lane count must be an exact multiple
  of the physical vreg lane count.
  For each contiguous physical chunk i:
    offset_i = base_offset + i * lanesPerPart
    dense_i = pto.vlds base[offset_i]

  If the requested VMI result layout is contiguous, return the dense chunks directly.
  If the requested VMI result layout is deinterleaved=2:
    prefer pto.vldsx2 "DINTLV_B8/B16/B32" per physical chunk group:
      %p0_i, %p1_i = pto.vldsx2 base[offset_i], "DINTLV_B*"
    return results in VMI partition-major order:
      p0_chunk0, p0_chunk1, ..., p1_chunk0, p1_chunk1, ...
  If the requested VMI result layout is deinterleaved=4 with exactly four physical parts:
    use dense pto.vlds chunks followed by the reverse two-level pto.vdintlv tree.

  For larger multi-chunk deinterleaved=4 loads, apply the same conversion per contiguous chunk group and return
  physical parts in VMI partition-major order:
    deinterleaved=4: p0_chunks..., p1_chunks..., p2_chunks..., p3_chunks...

vmi.store:
  direct lowering requires value element width to be 8, 16, or 32 bits so the
  emitted pto.vsts/pto.vstsx2 predicate can be materialized as b8/b16/b32.
  contiguous layout with full physical chunks:
    offset_i = base_offset + i * lanesPerPart
    mask_i = pto.pset_b8/b16/b32 "PAT_ALL"
    pto.vsts value_i, base[offset_i], mask_i
  contiguous layout with a final partial physical chunk:
    full chunks still use PAT_ALL
    the final chunk computes valid_lanes = logical_lane_count - chunk_i * lanesPerPart
    tail_mask_i = pto.plt_b8/b16/b32(valid_lanes)
    pto.vsts tail_value_i, base[offset_i], tail_mask_i
    padding lanes therefore have no externally visible store effect.

deinterleaved store:
  deinterleaved=2 with full physical chunks:
    prefer pto.vstsx2 "INTLV_B8/B16/B32" per physical chunk group:
      pto.vstsx2 p0_i, p1_i, base[offset_i], "INTLV_B*", all_true_mask
    offset_i = base_offset + i * 2 * lanesPerPart
    the vstsx2 dist mode writes logical lane 0,1,2,3,... order externally.

  current safe path lowers through proven register materialization before store:
    deinterleaved=4 with exactly four physical parts:
      use the two-level pto.vintlv tree, then store %d0/%d1/%d2/%d3 as contiguous chunks

  Larger multi-chunk deinterleaved=4 values use the same conversion per chunk group. The final store order is dense
  chunk order, so external memory observes logical lane 0,1,2,... order.

vmi.masked_load:
  semantics:
    if mask[lane] is true, result[lane] = memory[base + lane]
    if mask[lane] is false, result[lane] = passthru[lane]
    inactive mask lanes do not by themselves permit unsafe memory reads
  current direct path:
    result, passthru, and mask are requested as contiguous
    full physical chunks can always use pto.vlds because every loaded lane is logical
    partial/tail chunks require the same statically safe full-read proof as vmi.load
    for each contiguous physical chunk i:
      loaded_i = pto.vlds base[offset_i]
      result_i = pto.vsel loaded_i, passthru_i, mask_i
  unsupported cases:
    non-contiguous layouts
    unsafe partial/tail read footprints
    target true masked/non-faulting load and guarded/scratch fallback

vmi.stride_load:
  semantics:
    result lane order is contiguous VMI logical order
    source addresses start at source + offset and advance by block_stride
    32-byte blocks; VMI does not expose a repeat stride
    mask false lanes are inactive for the underlying block-strided load
  layout assignment:
    result natural layout is contiguous
    mask use is requested as contiguous with granularity derived from result element width
  current direct path:
    source must be !pto.ptr<T, ub>
    result and mask must be one contiguous physical chunk
    base = pto.addptr source, offset
    result = pto.vsldb base, block_stride, repeat_stride=0, mask
  unsupported cases:
    multi-chunk result or mask
    non-contiguous layouts
    memref/gm source

vmi.stride_store:
  semantics:
    value lane order is contiguous VMI logical order
    destination addresses start at destination + offset and advance by
    block_stride 32-byte blocks; VMI does not expose a repeat stride
    mask false lanes do not write memory
  layout assignment:
    value use is requested as contiguous
    mask use is requested as contiguous with granularity derived from value element width
  current direct path:
    destination must be !pto.ptr<T, ub>
    value and mask must be one contiguous physical chunk
    base = pto.addptr destination, offset
    pto.vsstb value, base, block_stride, repeat_stride=0, mask
      VMI lowering uses the ordinary resultless VPTO form. The fixed zero
      repeat stride preserves base-start addressing without post-update.
  unsupported cases:
    multi-chunk value or mask
    non-contiguous layouts
    memref/gm destination

vmi.gather:
  semantics:
    if mask[lane] is true, result[lane] = memory[base + indices[lane]]
    if mask[lane] is false, result[lane] = passthru[lane] and no memory read occurs for that lane
    indices are interpreted in element units, not bytes
  layout assignment:
    result natural layout is contiguous
    indices and passthru uses are requested as contiguous
    mask use is requested as contiguous with granularity derived from result element width
  current direct path:
    source must be !pto.ptr<T, ub>
    supported 32-bit mode:
      T must be a 32-bit element type
      indices must be signless or unsigned i32
      result / indices / passthru / mask must be contiguous full physical chunks
      mask granularity must be b32
      for each physical chunk i:
        gathered_i = pto.vgather2_bc source, indices_i, mask_i
        result_i   = pto.vsel gathered_i, passthru_i, mask_i
    supported ui16 mode:
      T must be ui16
      indices must be unsigned i16
      result / indices / passthru / mask must be one contiguous physical chunk
      mask granularity must be b16
      gathered = pto.vgather2 source, indices, mask
      result   = pto.vsel gathered, passthru, mask
      VPTO LLVM emitter bitcasts the physical index register from <128xi16>
      to the installed Bisheng intrinsic ABI <64xi32>; this is the same
      256B register payload viewed as the wrapper-level vector_u16 index
      container.
  reason for vsel:
    VPTO gather false predicate lanes do not read memory but produce zero; VMI false lanes preserve passthru.
  unsupported cases:
    f16/b16/f8/i8 result element types
    partial/tail chunks
    non-contiguous layouts
    memref/gm source
    guarded/scratch fallback

vmi.scatter:
  semantics:
    if mask[lane] is true, memory[base + indices[lane]] = value[lane]
    if mask[lane] is false, no memory write occurs for that lane
    indices are interpreted in element units, not bytes
    all active lanes must have pairwise-distinct indices; duplicate active indices violate the VMI scatter contract
  layout assignment:
    value and indices uses are requested as contiguous
    mask use is requested as contiguous with granularity derived from value element width
  current direct path:
    destination must be !pto.ptr<T, ub>
    T must be a 32-bit element type
    indices must be signless or unsigned i32
    value / indices / mask must be contiguous full physical chunks
    mask granularity must be b32
    for each physical chunk i:
      pto.vscatter value_i, destination, indices_i, mask_i
  unsupported cases:
    f16/b16/f8/i8 value element types
    partial/tail chunks
    non-contiguous layouts
    memref/gm destination
    ordered duplicate-index fallback

vmi.expand_load:
  semantics:
    k = 0
    for lane in logical order:
      if mask[lane]:
        result[lane] = memory[base + k]
        k += 1
      else:
        result[lane] = passthru[lane]
  layout assignment:
    result natural layout is contiguous
    passthru use is requested as contiguous
    mask use is requested as contiguous with granularity derived from result element width
  current direct path:
    static all-active path:
      pto.vmi.create_mask with constant active_lanes >= logical lane count
      dense all-true pto.vmi.constant_mask
    in that case expand_load degenerates to ordinary vmi.load:
      for each contiguous physical chunk i:
        loaded_i = pto.vlds base[offset_i]
        result_i = loaded_i
    partial/tail chunks still require the same statically safe full-read proof as vmi.load.
    runtime-mask path:
      source must be !pto.ptr<T, ub>
      T must be a 32-bit element type
      result / passthru / mask must be contiguous one full physical chunk
      mask granularity must be b32
      base_i    = pto.addptr source, offset
      indices_i = pto.vusqz(zero_i32_carrier, mask_i)
      loaded_i  = pto.vgather2_bc base_i, indices_i, mask_i
      result_i  = pto.vsel loaded_i, passthru_i, mask_i
  unsupported cases:
    runtime masks across multiple physical chunks
    runtime masks on non-32-bit element types
    non-contiguous layouts
    unsafe partial/tail read footprints
    guarded load or scratch fallback

vmi.masked_store:
  semantics:
    if mask[lane] is true, store value[lane]
    if mask[lane] is false, no memory write occurs for that logical lane
  current full-footprint path:
    value and mask are requested as contiguous at the use site
    mask granularity is derived from value element width
    for each contiguous physical chunk i:
      offset_i = base_offset + i * lanesPerPart
      pto.vsts value_i, base[offset_i], mask_i
  contiguous layout with a final partial physical chunk:
    full chunks store with the user mask directly
    the final chunk computes tail_valid_i with pto.plt_b8/b16/b32(valid_lanes)
    store_mask_i = pto.pand user_mask_i, tail_valid_i, all_true_mask_i
    pto.vsts tail_value_i, base[offset_i], store_mask_i
    padding lanes and user-inactive lanes therefore both have no write effect.
  If the incoming value/mask are deinterleaved, layout assignment inserts
  ensure_layout/ensure_mask_layout or the vmi-to-vpto pattern materializes the same contiguous representation before
  emitting stores. This preserves logical memory order and keeps inactive lanes write-free.

non-full chunks:
  vmi.store and vmi.masked_store support contiguous tail chunks by predicating the final pto.vsts with
  a prefix valid mask. masked_store additionally ANDs the user mask with the tail-valid mask.
  deinterleaved=2/4 tail store/masked_store is supported only through explicit layout materialization to
  contiguous chunks first. This requires every deinterleaved part to have the same physical chunk count, so the
  materializer can build complete vintlv/pintlv groups. After materialization, each contiguous chunk is predicated by
  the logical tail-valid mask; chunks whose active logical lane count is zero are not emitted as stores. Uneven
  deinterleaved groups, such as 129xf32 with deinterleaved=2, remain unsupported until a padding/scratch plan can
  assemble only the observable contiguous chunks.
  vmi.load support partial/tail chunks only when the direct full physical read is statically safe:
  statically shaped memref source, constant non-negative offset, and enough elements for the
  whole physical read footprint. Padding lanes must never become observable. Other partial/tail load cases still need
  scratch/guarded/true-masked load planning.
```

Histogram lowering：

```text
vmi.dhist semantics:
  source lanes are ui8 samples
  mask selects active source lanes
  acc/result are complete logical 256-bin ui16 histograms
  result[b] = acc[b] + count(active source lanes whose value equals b)

layout assignment:
  source layout = contiguous
  mask layout = contiguous, granularity b8
  acc/result layout = contiguous !pto.vmi.vreg<256xui16>

physicalization:
  acc/result physical arity is 2 because 256xui16 is 512B
  part0 represents logical bins 0..127
  part1 represents logical bins 128..255
```

`vmi-to-vpto` lowering for `pto.vmi.vdhist` is local and deterministic from the
op and assigned types:

```text
lo = converted acc part0
hi = converted acc part1

for each converted source physical chunk c in logical order:
  chunk_mask = converted b8 mask chunk c

  if source chunk c contains padding lanes because N is not a multiple of 256:
    valid = pto.pge/plt_b8 prefix mask for the valid logical lanes in this chunk
    chunk_mask = pto.pand chunk_mask, valid

  lo = pto.dhistv2 lo, src_c, chunk_mask, #bin=0
  hi = pto.dhistv2 hi, src_c, chunk_mask, #bin=1

return physical result parts [lo, hi]
```

Required preflight:

```text
acc/result element type is ui16 and logical lane count is exactly 256
source element type is ui8
source and mask logical lane counts match
source/mask are contiguous
mask granularity is b8
source physical chunks are 256-lane ui8 chunks; final partial chunk is allowed
only when the lowering can construct the valid-lane prefix mask
```

Diagnostics:

```text
VMI-UNSUPPORTED: pto.vmi.vdhist requires contiguous ui8 source, b8 mask, and
contiguous 256xui16 accumulator/result

VMI-UNSUPPORTED: pto.vmi.vdhist final partial source chunk requires valid-lane
b8 mask materialization
```

`pto.vmi.vchist` has the same verifier and assignment requirements as `pto.vmi.vdhist`.
A5 hardware `chistv2` high-range semantics have been confirmed as **global cumulative**
(bin=1 result automatically accumulates bin0's total count), so `pto.vmi.vchist` lowers
via the same template as `pto.vmi.vdhist` — the only difference is emitting `pto.chistv2`
in place of `pto.dhistv2`.  No software compensation is needed.

If future hardware switches to range-local cumulative semantics, the chist pattern
will need a broadcast+add compensation path.  In that scenario, introduce a
`-vmi-chist-mode` attribute or runtime probe as a separate evolution step.

Reference lit tests: `vmi_to_vpto_chist.pto` (mirrors `vmi_to_vpto_dhist.pto`).

Do not classify histogram as `group_reduce`.  Its result location is selected
by source values, not by lane/group position, and its low/high split is caused
by the physical `128xui16` VPTO result width.

Final hard gate：

```text
no pto.vmi op remains
no !pto.vmi.* type remains, including in function signatures
no UnrealizedConversionCastOp remains
physical arity matches helper for every lowered value
```

Slice 4 完成条件：

```text
1. `f16 -> f32 -> add -> store` lowers with deinterleaved=2 and stores contiguous logical order.
   Covered by vmi_to_vpto_e2e_widen_add_store.pto.
2. `f8 -> f32 -> add -> store` lowers with deinterleaved=4 and stores contiguous logical order.
   Covered by vmi_to_vpto_e2e_widen_add_store.pto.
3. Non-full memory physical arity and valid lane map are tested.
   Covered by vmi_to_vpto_load_nonfull.pto, vmi_to_vpto_load_nonfull_memref.pto,
   vmi_to_vpto_store_deint_invalid.pto,
   vmi_to_vpto_load_safe_tail_memref.pto,
   vmi_to_vpto_load_safe_tail_memref_negative_offset.pto,
   vmi_to_vpto_masked_load_safe_tail_memref.pto,
   vmi_to_vpto_masked_load_safe_tail_memref_negative_offset_invalid.pto,
   vmi_to_vpto_expand_load_all_active.pto,
   vmi_to_vpto_expand_load_all_active_negative_offset_invalid.pto, and multi-chunk load/store layout tests.
4. Full-footprint load/store direct path lowers through pto.vlds/pto.vsts or deinterleaved=2 x2 dist
   instructions with offset 0.
   Covered by the load/store direct-path and layout-folding tests.
5. Internal func.call boundaries expand callee signatures, call operands/results, and returned VMI values together.
   Covered by vmi_layout_assignment_call_boundary.pto, vmi_layout_assignment_indirect_call_invalid.pto,
   and vmi_to_vpto_call_boundary.pto.
6. Structured control-flow carrying VMI values expands iter args, yields, results, masks, and returns together.
   Covered by vmi_layout_assignment_cf_switch.pto,
   vmi_layout_assignment_scf_execute_region.pto,
   vmi_layout_assignment_scf_index_switch.pto,
   vmi_layout_assignment_scf_while.pto, vmi_to_vpto_cf_branch.pto,
   vmi_to_vpto_scf_for.pto, vmi_to_vpto_scf_if.pto, and the user-facing
   vmi_ptoas_cli_control_flow.pto.
7. Final gate rejects residual VMI helper and unrealized casts.
   Covered by vmi_to_vpto_ensure_identity.pto,
   vmi_to_vpto_ensure_layout_partial_invalid.pto,
   vmi_to_vpto_truncf_fp8_128_contiguous_invalid.pto,
   vmi_to_vpto_ensure_mask_layout_partial_invalid.pto,
   vmi_to_vpto_unsupported_op_invalid.pto,
   vmi_to_vpto_unrealized_cast_residual_invalid.pto,
   vmi_to_vpto_type_attr_residual_invalid.pto, and per-feature unsupported
   tests.
8. Same-family indirect memory ops reject unsupported direct-lowering shapes consistently.
   Covered by vmi_to_vpto_gather_scatter_shape_invalid.pto together with the existing gather/scatter positive and
   per-feature negative tests.
9. Same-family reduction ops reject unsupported direct-lowering shapes consistently.
   Covered by vmi_to_vpto_reduce_shape_invalid.pto together with the existing reduce add/min/max positive and
   per-feature tests, including vmi_to_vpto_reduce_addi_i16_invalid.pto for narrow integer rejection and
   vmi_to_vpto_reduce_addf_f16.pto for f16 floating-point reduction lowering.
10. VMI op/type verifiers reject unsupported element types before OneToN rewriting.
    Covered by vmi_to_vpto_bf16_arith.pto, vmi_to_vpto_math_element_type_invalid.pto,
    vmi_to_vpto_cmp_select.pto, vmi_to_vpto_cmp_element_type_invalid.pto,
    vmi_to_vpto_fma.pto, vmi_to_vpto_fma_element_type_invalid.pto, and
    vmi_to_vpto_unary_math.pto for negf/absf/absi/sqrt/exp/ln/relu, plus
    vmi_to_vpto_relu_element_type_invalid.pto.
11. Same-family mask logic ops lower through the physical mask granularity instead of assuming b32 masks.
    Covered by vmi_to_vpto_mask_logic.pto for mask_and/mask_or/mask_xor/mask_not on b32 masks produced by
    cmpf and on direct b8/b16 mask operands.
12. `pto.vmi.vdhist` lowers one logical 256-bin histogram into two VPTO low/high
    bin-range histogram accumulator chains, and tail source chunks are masked
    with a valid-lane b8 prefix. `pto.vmi.vchist` uses the same lowering template
    as `pto.vmi.vdhist` (A5 hardware confirmed global cumulative semantics).
    Covered by vmi_to_vpto_dhist.pto, vmi_to_vpto_dhist_tail_mask.pto, and
    vmi_to_vpto_chist.pto.
```

## 7. Slice 5: Memory Padding

The Slice 4 direct path lowers `pto.vmi.load` through plain `pto.vlds` when the
memory source itself is supported and the element type has a known physical lane
width. This includes non-full logical vectors; the operation is treated as a
direct full physical read of the selected VPTO chunk(s). Masked/expand/gather
read-like operations still use the richer access plan because their masks or
lane maps carry additional semantic constraints.

Implement an internal `VMIMemoryAccessPlan`:

```text
base
logical lane count
logical_shape
permutation_map
lane-to-address map in element units
validMask
paddingValue
safeReadProof
writeMask
target capability decision
fallback resource decision
```

Current implementation status:

```text
lib/PTO/Transforms/VMIToVPTO.cpp
  VMIMemoryAccessPlan
  VMIMemorySafeReadProof
  VMIMemoryLogicalShape
  VMIMemoryLaneAddressMap
  VMIMemoryFallbackDecision

currently routed through the plan:
  contiguous identity logical_shape/permutation/lane-to-address map in element units
  explicit rejection of non-identity memref layouts until subview/affine lane maps are represented
    covered by vmi_to_vpto_memref_layout_invalid.pto, including a memref.subview-produced strided view
    subview diagnostics name the missing normalized base/offset/stride lane-to-address plan
  target true masked/non-faulting load capability query
    current result is missing capability because pto.vlds has no mask operand
    unified masked-load semantics use a full vload followed by vsel
    covered by vmi_to_vpto_masked_load_nonfull.pto and
    vmi_to_vpto_masked_load_safe_tail_memref.pto
  direct pto.vmi.load source/layout capability check for full physical reads
  pto.vmi.masked_load partial/tail safe full-read proof
  pto.vmi.expand_load static all-active safe full-read proof
  VMI-to-VPTO rewrite match guard for supported direct load sources
  pto.vmi.store direct write target decision with all-true writeMask kind
  pto.vmi.masked_store direct write target decision with explicit writeMask kind
  unsafe expand-load partial/tail read fallback decision as RequiredUnavailable diagnostic
    covered by vmi_to_vpto_expand_load_all_active_negative_offset_invalid.pto

currently not implemented by the plan:
  paddingValue materialization (intentionally unsupported in the first implementation stage)
  non-all-true validMask direct masked/non-faulting load lowering
  scratch/guarded fallback lowering or allocation
  lowering for non-identity logical_shape/permutation_map/lane-to-address maps, including subview or affine lane maps
  writeMask fallback planning beyond the existing contiguous tail-store predicate path
```

Important first-stage contract:

```text
VMI physical tail lanes and transfer paddingValue are different concepts.

Physical tail lanes:
  arise because pto.vreg is fixed at 256 bytes
  are outside the logical VMI lane count
  may be read/computed only when the extra lanes remain unobservable

transfer_read-style paddingValue:
  is an observable logical result for invalid/OOB transfer lanes
  cannot be dropped or replaced by arbitrary physical tail contents
  is not materialized by the first-stage VMI implementation

Therefore any frontend path that still needs transfer_read paddingValue
semantics must stop before direct VMI-to-VPTO lowering with VMI-UNSUPPORTED,
unless it has already canonicalized to an all-valid load/masked_load subset
whose invalid lanes are proven absent.
```

Read-like memory decision tree：

```text
safeReadProof full && validMask all true:
  direct load

safeReadProof full && validMask not all true:
  first-stage: VMI-UNSUPPORTED because paddingValue materialization is not implemented
  future: full load + padding materialization + select

target true masked/non-faulting load:
  first-stage: VMI-UNSUPPORTED because true masked/non-faulting load and paddingValue materialization are not implemented
  future: masked load + padding materialization

otherwise:
  first-stage: VMI-UNSUPPORTED with the missing fallback reason
  future: split safe regions, scratch fill/copy/load, guarded fallback, or diagnostic
```

Write-like memory decision tree：

```text
writeMask all true && full footprint safe-writable:
  direct store

target true masked store:
  masked store

otherwise:
  split/guarded/scatter-like fallback or diagnostic
```

Slice 5 完成条件：

```text
1. Unsafe partial/tail read-like ops never lower to a potentially invalid full
   read unless the physical footprint is statically proven safe.
2. PaddingValue materialization is not required in the first implementation
   stage. Any path that would require paddingValue, true masked/non-faulting
   load, scratch fill/copy/load, or guarded fallback must report
   `VMI-UNSUPPORTED` with the missing fallback reason.
3. Non-identity logical_shape/permutation_map/lane-to-address maps, including
   subview or affine lane maps, are explicitly rejected before lowering.
4. Store-like partial/tail writes are supported only by the existing
   full-chunk or contiguous/deinterleaved tail-store predicate paths. Other
   writeMask fallback paths must report `VMI-UNSUPPORTED`.
```

## 8. Layout Fact And Lowering Support Helpers

Keep layout facts separate from layout assignment policy and VPTO lowering
choices.  Shared layout helpers expose legal/preferred layout facts; they do
not select a global lowering plan and are not a target capability registry.

```text
getPreferredCastLayoutFact(sourceType, resultType)
getPreferredGroupReduceLayoutFact(sourceType, numGroups)
canMaterializeDataLayout(sourceType, resultType)
canMaterializeMaskLayout(sourceType, resultType)
supportsMemoryAccessProof(proof)
supportsPrefixPopcount(maskType)
supportsReductionScanContract(op)
getScratchResource(plan)
```

Support and materialization helpers must expose actionable reasons. A pass must
not silently choose scalar fallback when fallback is disabled.

Current implementation status:

```text
include/PTO/Transforms/VMILayoutSupport.h
lib/PTO/Transforms/VMILayoutSupport.cpp
  central table-driven source for legal/preferred layout facts:
    dense/group load and store layouts
    masked load/store data-mask layout relations
    ensure data/mask layout materialization pairs
    cast, bitcast, reduce, group_reduce, group_broadcast, histogram layouts

lib/PTO/Transforms/VMIToVPTO.cpp
  local op lowering support helpers:
    true masked-load and fallback diagnostics for currently unimplemented paths
    pointer-only constraints for concrete VPTO gather/stride/scatter paths
    padding-safety and full-physical-chunk requirements

still legacy helper-based and should migrate into layout/support tables when
they become layout facts:
  full layout materialization plans and padding-safety checks
  adjacent ppack/punpack mask granularity materialization plans
  prefix popcount and full reduction/scan/contract shape checks
```

## 9. Diagnostics

Centralize diagnostic codes in one header or utility file:

```text
VMI-UNSUPPORTED
VMI-LAYOUT-CONTRACT
VMI-PASS-INVARIANT
VMI-RESIDUAL-OP
```

Current implementation defines these codes and their `": "` prefixes in `include/PTO/IR/VMIUtils.h`. Transform and
CLI code must reference those constants instead of spelling the diagnostic code strings locally; a source grep for the
four code strings should find only the central definitions.

Every diagnostic should include:

```text
source op
logical VMI type
producer natural layout, if any
consumer required layout, if any
missing capability or disabled option
available materialization paths, if known
```

## 10. Lit Test Layout

Use a dedicated directory:

```text
test/lit/vmi_new/
```

Minimum test files:

```text
vmi_type_attr_parse.mlir
vmi_type_attr_invalid.mlir
vmi_op_verifier_basic.mlir
vmi_producer_boundary.mlir
vmi_layout_assignment_widen.mlir
vmi_layout_assignment_cfg.mlir
vmi_layout_assignment_broadcast_remat.mlir
vmi_layout_assignment_iota_remat.mlir
vmi_layout_assignment_mask_remat.mlir
vmi_to_vpto_deinterleaved2.mlir
vmi_to_vpto_deinterleaved4.mlir
vmi_to_vpto_compaction_deint_invalid.mlir
vmi_to_vpto_load_safe_tail_memref.mlir
vmi_to_vpto_masked_load_safe_tail_memref.mlir
vmi_to_vpto_store_tail.mlir
vmi_to_vpto_dhist.mlir
vmi_to_vpto_dhist_tail_mask.mlir
vmi_to_vpto_chist.mlir
vmi_pipeline_hard_gates.mlir
```

Each pass test must use `FileCheck` to prove both positive output and negative absence:

```text
CHECK: pto.vmi.addf
CHECK-NOT: pto.vadd
CHECK-NOT: unrealized_conversion_cast
```

Final lowering tests must check:

```text
CHECK-NOT: pto.vmi.
CHECK-NOT: unrealized_conversion_cast
```

## 11. Implementation Order

Recommended merge order:

```text
1. VMI type/attr + helper + parse/verify tests.
2. Slice 1 op shells + verifier tests.
3. VMI producer boundary verifier.
4. layout assignment for straight-line code.
5. layout assignment for scf/cf/function boundaries.
6. vmi-to-vpto type conversion + pack/unpack/unpackable block args.
7. deinterleaved=2 f16 widen end-to-end.
8. deinterleaved=4 f8 widen end-to-end.
9. load/store padding-safe lowering.
10. remaining semantic op families.
```

Do not merge a pass that leaves hidden side tables as a required interpretation mechanism. Temporary internal
analysis structures are fine only if the pass materializes the final state into IR before returning.

## 12. Review Checklist Before Coding Each Slice

Before implementation:

```text
1. Is the op/type syntax written in ODS and tested by parser round-trip?
2. Does every verifier rule have a negative test?
3. Does every pass have a post-pass hard gate?
4. Are CFG block arguments and function signatures covered?
5. Does any lowering rely on a defining op that block arguments do not have?
6. Does memory lowering prove safe footprint separately from valid lane mask?
7. Does mask granularity follow consumer element width?
8. Does final VPTO lowering leave zero VMI op/type/helper or unrealized-cast residuals?
```

If any answer is no, the slice is not ready to be treated as complete.

## 13. Adding One VMI Op End To End

新增一个 `pto.vmi.*` op 时，不要只补 ODS 和 lowering pattern。它必须穿过固定的七个落点，
否则很容易出现 verifier 能过、layout pass 不知道怎么约束、或控制流 physicalization 后残留 VMI type。

```text
1. ODS surface:
   include/PTO/IR/VMIOps.td

2. semantic verifier:
   lib/PTO/IR/VMI.cpp

3. layout assignment facts:
   lib/PTO/Transforms/VMILayoutAssignment.cpp

4. shared layout support, when the fact crosses stages:
   include/PTO/Transforms/VMILayoutSupport.h
   lib/PTO/Transforms/VMILayoutSupport.cpp

5. vmi-to-vpto preflight:
   lib/PTO/Transforms/VMIToVPTO.cpp::verifySupportedVMIToVPTOOps

6. OneToN lowering pattern:
   lib/PTO/Transforms/VMIToVPTO.cpp::populateVMIOneToNConversionPatterns

7. focused lit tests:
   test/lit/vmi_new/
```

这七个落点的职责不同：

```text
ODS:
  只定义 op 形状、operand/result type 类别、assembly format、interface 和 verifier hook。

VMI.cpp verifier:
  检查局部语义，例如元素类型、rank、lane count、predicate 字符串、source/result bit 数关系。
  不能依赖 def-use 图，不能决定 layout。

LayoutAssignment:
  只收集 value-level layout/granularity 事实：
    - producer natural layout
    - operands that must share layout with result
    - consumer required layout
    - mask consumer required granularity
  不能在 collect 阶段改 IR。

VMILayoutSupport:
  只放跨 assignment、validation、optimization、lowering 中至少两个阶段共享的纯查询。
  典型内容是 cast layout fact、group_reduce layout fact、ensure_* materialization support。
  不能返回 VPTO instruction sequence、不能决定 clone/rematerialize、不能读取 producer/user context。
  只有一个 lowering pattern 自己使用的判断不要抽到这里。

VMIToVPTO preflight:
  在 rewrite 前拒绝当前 lowering 不支持但语义合法的 case。
  典型例子是 partial physical chunk、non-prefix mask constant、dynamic create_mask、unsupported shuffle。

OneToN pattern:
  从 adaptor 读取 physical parts，按已经确定的 layout 发 VPTO op。
  不能重新推断 layout，也不能通过 defining op 找 physical parts。

lit:
  至少覆盖 parser/verify、layout assignment、positive lowering、negative unsupported diagnostic。
```

### Layout Fact Template

新增 op 时先给它归类，再写 layout 约束。不要从 VPTO 指令形态反推 VMI layout；layout 的来源必须是
logical vector 语义和当前物理指令的天然限制。

```text
elementwise same-shape op:
  examples:
    addf/addi/subf/mulf/andi/shli/shrui/shrsi/absf/absi/sqrt
  layout rule:
    all data operands and result are in one equivalence class
  lowering rule:
    emit one VPTO op per physical part

compare op:
  examples:
    cmpf/cmpi
  layout rule:
    lhs/rhs data layout unified
    result mask requested to the same data layout
    result mask granularity comes from lhs/rhs element width
  lowering rule:
    emit one vcmp per data part, producing corresponding mask part

mask logical op:
  examples:
    mask_and/mask_or/mask_xor/mask_not
  layout rule:
    all mask operands/results share layout and granularity
  lowering rule:
    emit one predicate op per physical mask part

layout-changing producer:
  examples:
    extf f16->f32, extf f8->f32, truncf f32->f16, truncf f32->fp8-like
  layout rule:
    source/request side follows instruction input contract
    result natural layout follows instruction output contract
  lowering rule:
    emit the instruction sequence that preserves logical lane order under that layout

memory consumer/producer:
  examples:
    load/store/load/store
  layout rule:
    load result natural layout is chosen by memory dist capability
    store value operand requests the layout that memory dist can consume
  lowering rule:
    direct path only when every physical chunk has no padding lane and footprint is safe

structural boundary:
  examples:
    scf.if result/yield, scf.for iter args, cf.br successor operands, func.call
  layout rule:
    semantically identical incoming/outgoing values are unified
  lowering rule:
    handled by OneToN structural patterns, not by op semantic lowering
```

代码里 `LayoutSolver::addConstraints()` 应该只表达上面的事实。例如一个普通 elementwise binary op
只需要：

```cpp
if (auto addf = dyn_cast<VMIAddFOp>(op)) {
  if (failed(unite(addf.getLhs(), addf.getRhs(), op)) ||
      failed(unite(addf.getLhs(), addf.getResult(), op)))
    return WalkResult::interrupt();
  return WalkResult::advance();
}
```

一个 layout-changing op 不应该把 source/result 直接 `unite`，而是明确写 producer/consumer 合同：

```cpp
if (auto extf = dyn_cast<VMIExtFOp>(op)) {
  requestDataUse(extf.getSourceMutable(), getContiguousLayout());
  if (failed(setNaturalLayout(extf.getResult(),
                              VMILayoutAttr::getDeinterleaved(ctx, factor),
                              op)))
    return WalkResult::interrupt();
  return WalkResult::advance();
}
```

### OneToN Pattern Template

`vmi-to-vpto` pattern 的输入不再是 logical VMI value，而是 adaptor 里已经 flatten 好的 physical parts。
pattern 只做三件事：

```text
1. 从 adaptor 取每个 logical operand 的 physical part list。
2. 从 resultMapping 取每个 logical result 对应的 physical result type list。
3. 按 part 顺序创建 VPTO op，并用 resultMapping replace 原 op。
```

普通 elementwise binary op 的代码形态应该接近：

```cpp
LogicalResult matchAndRewrite(VMIAddFOp op, OpAdaptor adaptor,
                              OneToNPatternRewriter &rewriter) const override {
  ValueRange lhsParts = adaptor.getLhs();
  ValueRange rhsParts = adaptor.getRhs();
  TypeRange resultTypes = adaptor.getResultMapping().getConvertedTypes(0);

  if (lhsParts.size() != rhsParts.size() || lhsParts.size() != resultTypes.size())
    return rewriter.notifyMatchFailure(op, "physical arity mismatch");

  SmallVector<Value> results;
  for (auto [lhs, rhs, resultType] : llvm::zip_equal(lhsParts, rhsParts, resultTypes))
    results.push_back(rewriter.create<VaddOp>(op.getLoc(), resultType, lhs, rhs));

  rewriter.replaceOp(op, results, adaptor.getResultMapping());
  return success();
}
```

这里不能调用 `op.getLhs().getDefiningOp()` 去找物理寄存器。原因是 VMI value 可以来自：

```text
function argument
block argument
scf.for iter arg
scf.if result
cf.br successor argument
func.call result
```

这些 value 很多没有 VMI defining op。physical parts 的唯一合法来源是 OneToN adaptor 和
OneToNTypeMapping。

### Control-Flow Checklist

每新增一个 op，不一定要写新的控制流 pattern；但必须检查它的结果或 operand 是否可能跨边界。
如果只是普通 VMI value，那么已有 structural OneToN pattern 应该负责边界 physicalization：

```text
func.func / func.call / func.return:
  upstream func OneToN conversion

scf.if / scf.for / scf.while / scf.yield:
  upstream SCF OneToN structural conversion plus layout solver equivalence constraints

cf.br / cf.cond_br / cf.switch:
  project-local OneToN patterns flatten successor operands and rewrite destination block signatures

scf.execute_region / scf.index_switch:
  project-local OneToN patterns flatten region results
```

新增 op 的测试要至少放一个跨边界用例，证明 op 的 result 不是只在 straight-line IR 中工作：

```mlir
%r = scf.if %cond -> !pto.vmi.vreg<128xf32> {
  %x = pto.vmi.addf %a, %b : ... -> !pto.vmi.vreg<128xf32>
  scf.yield %x : !pto.vmi.vreg<128xf32>
} else {
  scf.yield %c : !pto.vmi.vreg<128xf32>
}
pto.vmi.store %r, %ptr, %off : ...
```

对应 lowering test 必须检查：

```text
CHECK-NOT: pto.vmi.
CHECK-NOT: !pto.vmi.
CHECK-NOT: unrealized_conversion_cast
```

如果这个测试失败，通常不是该 op 的 VPTO pattern 本身错，而是 layout assignment 没有把 yield/result/consumer
约束统一，或者 OneToN structural pattern 漏了某种 region/control-flow op。

### Preflight Versus Pattern Failure

语义合法但当前还没有物理实现的 case，应该在 `verifySupportedVMIToVPTOOps()` 里给稳定 diagnostic，
不要让 pattern 随机 `notifyMatchFailure()` 后落成 generic conversion failure。

```text
use verifier failure:
  op 本身语义非法，任何 target 都不应该接受。
  examples:
    absf on integer element
    shrui on signed integer element
    shrsi on unsigned integer element
    bitcast total bits mismatch

use VMI-LAYOUT-CONTRACT:
  多个 producer/consumer/control-flow 约束互相冲突。
  examples:
    one value simultaneously required as contiguous and deinterleaved=2
    one mask simultaneously required as b16 and b32

use VMI-UNSUPPORTED in preflight:
  VMI semantics are valid, but current VPTO materialization is not implemented.
  examples:
    partial/tail memory access
    pred-only constant mask without concrete b8/b16/b32 granularity
    shuffle that requires vselr index-vector materialization
    bitcast with mismatched layouts or per-chunk logical bit footprints

use VMI-RESIDUAL-OP:
  conversion framework finished but VMI op/type/helper/cast remains.
  This is a pass bug or missing pattern, not a user semantic error.
```

Pattern-local `notifyMatchFailure()` is still useful for debugging competing patterns, but it must not be the only
user-visible explanation for a known unsupported VMI semantic case.
