# PTOAS SIMT 寄存器驻留 fragment 方案

## 背景

RMSNorm 中存在重复的 UB 读取。权重 `W` 已从 GM 搬入 UB，但每个 token 进入 `SimtVF` 时仍需将 `w_ub` 读入寄存器；以 `batch/N_CORES = 64` 为例，每核重复 64 次。目标是在同一 kernel 的多个 `SimtVF` 之间驻留该数据，将 `UB→VRF` 读取从 64 次降为 1 次。

本方案采用**显式 fragment**：

- 前端不新增 API。`alloc_fragment` 声明于 `SimtVF` 外即视为跨 `SimtVF` 驻留的 fragment。
- `pto.persistent` 标记 allocation 是跨 SIMT section 的持久化载体。`residentElements` 由 init section 中确定完成初始化的 element 定义，不由 allocation 的物理 padding 或后续 consumer 的访问集合定义。
- `residentElements` 中每个 element 都分配稳定 slot，并作为一个完整对象穿过所有受 init section 支配的 SIMT section；consumer 访问不用于裁剪。
- 不设只读限制，须支持 read-modify-write（如寄存器上的 reduce 累加）。
- fragment 在不同 `SimtVF` 中的 layout 一致性由 TileLang layout inference 保证。
- PTOAS 侧使用已有的 `pto.keep` / `pto.resume` 表达跨 `SimtVF` 的寄存器保存与恢复。

处理链路为：

```text
TileLang TIR → TileLang PTO codegen → PTODSL Python → PTODSL lowering → PTO IR → PTOAS
```

寄存器驻留优化在 PTOAS 中完成，输入为 codegen 产出的、SIMT 代码仍包裹在内联 section 中、尚未 outline 的 PTO IR。

## 输入表示与前置约束

物化 pass 的输入保持「SIMT 代码包裹在内联 section 中、尚未 outline」的形态，采用以下跨层约定：

1. **PTODSL 侧：保留 SIMT 内联形态、推迟 outline。** `with pto.simt(...)` emit `pto.section.simt`，block 退出时不生成 `pto.simt_entry + pto.simt_launch`。outline 统一在 PTOAS 完成。

2. **PTO IR：使用 SIMT 内联 region op。** `pto.section.simt<<<dimX, dimY, dimZ>>>` 与 `pto.section.cube` / `pto.section.vector` 保持同类 section 表示，并在 op 头部携带静态三维 launch 参数。当前 section region 保持单顶层 block，但该 block 内可以包含 `scf.if` / `scf.for` 等 nested structured control flow。`pto-outline-simt-sections` 在物化后将其转换为 `simt_entry + simt_launch`。

3. **PTOAS 侧：承接 simt 作用域外的 persistent alloca。** persistent fragment 的 `llvm.alloca` 位于 simt 作用域外的 kernel 函数体，且并非真实内存（只有被 keep/resume 替换后才成立）。物化 pass 之前该 alloca 不得被当作栈内存 lower；物化未触发时按 fail-fast 报错（见「资源预算、跨层契约与错误处理」）。

4. **物化 pass 内部：outline 前的 unroll 与 keep/resume 位置。** 现有 keep/resume verifier（`PTOValidateVPTOIR.cpp`）要求 keep 紧邻 `func.return`。`pto-unroll-simt-for`（`Passes.td`）已扩展为同时处理带 `pto.simt_entry` 的函数和 outline 前的 `pto.section.simt`，且不会影响普通函数中 section 外的循环；keep/resume 在 outline 前物化，outline 后自然落在 entry 的入口/出口，从而复用现有 verifier。

## Outline 前置条件

- outline 前，fragment 的外层定义、完整 init 和其余 SIMT section 位于同一函数内，对象校验、slot 分配、物化和 cleanup 均可在一个 `func.func` 内完成。
- outline 后每次 `simt_launch` 是一次不透明调用，fragment 的完整 allocation、初始化 section 和其余 carry section 的关系均不可见，PTOAS 无法可靠完成对象级物化。
- keep/resume 在 outline 前生成、outline 后自然落到各 `simt_entry` 的入口与出口，满足现有 verifier 对位置的要求。

## 输入 IR 形态

内联 SIMT section 的实际 op 名为 `pto.section.simt`。

fragment 在最终 TIR 中是一块 `local` buffer（`T.allocate([32], "float32", "local")`）。TileLang PTO codegen 将其 emit 为 kernel body 作用域（所有 `with pto.simt` 块之外）的 `pto.alloc_buffer`；PTODSL lowering 将其转换为带 `pto.persistent` 属性的 `llvm.alloca`。该属性用于识别 def/use。示意输入：

```mlir
func.func @main_kernel(...) attributes {pto.entry} {
  %w_frag = llvm.alloca %c32 x f32 {pto.persistent} : (i32) -> !llvm.ptr

  // init section：UB → fragment
  pto.section.simt<<<128, 1, 1>>> {
    %w0 = pto.load %w_ub[%idx0] : !pto.ptr<f32, ub> -> f32
    %p0 = llvm.getelementptr %w_frag[%off0] : (!llvm.ptr, i64) -> !llvm.ptr, f32
    llvm.store %w0, %p0 : f32, !llvm.ptr
    // ... 同样定义本次需要驻留的其它 element
  }

  scf.for %t = %c0 to %tokens step %c1 {
    pto.section.simt<<<128, 1, 1>>> {
      %p0 = llvm.getelementptr %w_frag[%off0] : (!llvm.ptr, i64) -> !llvm.ptr, f32
      %w0 = llvm.load %p0 : !llvm.ptr -> f32
      %y = arith.mulf %x, %w0 : f32
      // 可选 read-modify-write：
      // llvm.store %updated, %p0 : f32, !llvm.ptr
    }
  }
  return
}
```

PTOAS 把 `%w_frag` 认成 persistent fragment 的依据：

- defining op 是带 `pto.persistent` 的 `llvm.alloca`；
- 定义点在 `pto.section.simt` 外，支配所有使用它的 section；
- 至少一个 use 在 `pto.section.simt` 内；
- allocation 边界、init 中确定初始化的 element 和访问 offset 静态可解析。

## `pto.persistent` 的来源

`pto.persistent` 由 fragment 的词法作用域表达，PTODSL 前端合成，PTOAS 消费并校验。规则「`alloc_buffer` 在 SIMT region 外即视为跨 VF 驻留」在前端求值。

**DSL 表面。** fragment 写在其自然作用域：跨 VF 复用的 fragment 位于所有 `with pto.simt` 块之外，section-local 的临时 fragment 位于块内。以 `kernel.py` 为例：

```python
w_frag = pto.alloc_buffer((32,), pto.f32)   # SIMT region 外 → persistent
with pto.simt(128, 1, 1):                    # init：UB → w_frag
    for i in pto.static_range(0, 16):
        scalar.store(scalar.load(...), w_frag, i * 2)

with pto.for_(0, 64, step=1) as t:
    with pto.simt(128, 1, 1):                # consume
        x_frag = pto.alloc_buffer((32,), pto.f32)   # SIMT region 内 → 非 persistent
        sum_sq = pto.alloc_buffer((1,), pto.f32)    # 同上
        ...
        scalar.store(... * scalar.load(w_frag, i_2 * 2, contiguous=2), ...)
```

**PTODSL lowering。** `pto.alloc_buffer(...)` 在 trace 时读取所处 subkernel 作用域（`ptodsl/_ops.py`）：位于 `with pto.simt(...)` 外则 persistent，emit 出的 `llvm.alloca` 附加 `pto.persistent`（`UnitAttr`）；位于块内则为普通 local alloca。无需新增前端 API 或显式 `persistent=True` 参数。

**PTOAS。** 消费 `pto.persistent`，并校验对象级契约：定义点在 region 外且支配所有 use、allocation shape 静态、init section 至少确定初始化一个 element、其余 carry section 由 init 支配且 launch 维度一致、所有访问均属于 `residentElements`，且访问 offset 静态并位于 allocation 范围内。不满足则 fail-fast（见「资源预算、跨层契约与错误处理」）。PTOAS 不自行发现 persistent fragment。

## Persistent 对象语义

`pto.persistent` 表示整个 `llvm.alloca` 是 persistent fragment 的存储载体，但 allocation 的物理 element count 不直接等于驻留 slot 数。物化 pass 遵循以下契约：

1. allocation 的 element type 和静态 element count 定义合法地址范围；init section 中确定完成初始化的 element 集合定义 `residentElements`。
2. init section 是访问该 allocation 且支配其它所有访问 section 的唯一 section，不要求它直接位于 kernel entry block。当前规则中，直接位于 init section 顶层 block 中的 store 视为确定初始化；每个 resident element 的首次访问必须是 store。
3. init 未写入的 padding element 不占 slot，也不需要伪造初值。例如 allocation 经 layout inference 为 32 个 `f32`，init 只确定写入 element 0～19，则 `residentElements = {0..19}`，slot 数为 20。
4. `residentElements` 一经 init 确定便保持不变。consumer 是否访问某个 resident element 不参与裁剪；任一访问落在集合外均报错。
5. 所有受 init 支配的其它 SIMT section 都 carry 完整 `residentElements`：入口 resume 全部 resident element，出口 keep 全部 resident element。无本地访问的 section 也不能跳过。
6. 不做跨 section per-element liveness 或末次使用分析。read-modify-write 只改变 element 的值，不改变 `residentElements` 和 slot 集合。
7. 多个 persistent allocation 使用函数级互不重叠的 slot；每个对象可以有自己的 init section，已 active 的对象必须穿过其激活点之后的其它对象 init section。

## 输出 IR 形态

keep/resume 承载跨 section 的寄存器状态。示意输出：

```mlir
func.func @main_kernel(...) attributes {pto.entry} {
  pto.section.simt<<<128, 1, 1>>> {             // init
    %w0 = pto.load %w_ub[%idx0] : !pto.ptr<f32, ub> -> f32
    pto.keep %w0 {slot = 0 : i64} : f32
    // ... keep 完整 residentElements
  }

  scf.for %t = %c0 to %tokens step %c1 {
    pto.section.simt<<<128, 1, 1>>> {           // consume
      %w0 = pto.resume {slot = 0 : i64} : f32
      // ... resume 完整 residentElements
      %y = arith.mulf %x, %w0 : f32
      pto.keep %w0 {slot = 0 : i64} : f32       // 出口 re-keep
      // ... re-keep 完整 residentElements
    }
  }
  return
}
```

read-modify-write 场景下 resume 为携带值、keep 为更新后的值：

```mlir
pto.section.simt<<<128, 1, 1>>> {
  %acc0 = pto.resume {slot = 0 : i64} : f32
  %acc1 = arith.addf %acc0, %partial : f32
  pto.keep %acc1 {slot = 0 : i64} : f32
}
```

出口 re-keep 是完整驻留对象的一部分：resume 之后 body 可能占用其它 SIMT 寄存器、覆盖这些物理 slot，因此 init 之后的每个 SIMT section 都必须在入口 resume、出口 keep 完整 `residentElements`。即使 section 不访问该 fragment，也要用 `resume → keep` 携带该集合；不进行末次使用分析，最后一个 section 同样 re-keep。

outline pass 再把每个 section 转成 `func.func {pto.simt_entry}` + `pto.simt_launch`，keep 落在 return 前、resume 落在入口，正好满足现有 verifier。

## Pass 设计

- 分析实现：`lib/PTO/Transforms/SIMTPersistentFragmentAnalysis.cpp` / `SIMTPersistentFragmentAnalysis.h`
- Analysis-only pass：`lib/PTO/Transforms/PTOAnalyzeSIMTPersistentFragment.cpp`，pass 名 `pto-analyze-simt-persistent-fragment`
- C++：`lib/PTO/Transforms/PTOMaterializeSIMTPersistentFragment.cpp`
- Pass 名：`pto-materialize-simt-persistent-fragment`
- 作用域：`func::FuncOp`（内联 section 在 kernel 函数内，无需跨函数分析）
- 构造：`createPTOMaterializeSIMTPersistentFragmentPass()`

### 1. 收集 SIMT section

遍历 kernel 函数中的全部 `pto.section.simt`，按函数 walk 顺序记录父函数、位置、线程维度和 body region。该列表是函数级计划的一部分，不能只收集访问 persistent alloca 的 section；已 outline 为 `func.func {pto.simt_entry}` 的函数不在本 pass 的处理范围内。

### 2. 定位并校验 persistent fragment

顺 SSA 定位带 `pto.persistent` 的 `llvm.alloca`，并校验结构前提：

- 定义点在 `pto.section.simt` 外，且支配所有 use；
- 至少一个 use 在 `pto.section.simt` 内；
- 访问该 fragment 且支配其它所有访问 section 的唯一 section 是 init section，并由其中确定完成初始化的 element 建立非空 `residentElements`；
- init section 必须支配其余 carry section，同一 fragment 的 carry section 使用相同 launch 维度；
- init 之后的访问必须属于 `residentElements`；allocation 中未被 init 确定初始化的 padding element 不参与驻留；
- 若同时在非 SIMT 代码中被读写，报错，避免跨 section 寄存器状态与普通内存语义混用。

持久化由 `pto.persistent` 表达，PTOAS 不从「被多个 section 使用」等结构特征反推持久化；上述校验用于确认属性的前提成立，不成立则 fail-fast。

RMSNorm 权重驻留的典型形态：kernel 作用域 alloca `w_frag` → init section store → 循环体 consume section load。

### 3. 解析 allocation、residentElements、访问与 slot

对每个 fragment 解析 section 内的 load/store：

- 支持 alloca offset 0 直接访问、常量 GEP、可折叠为常量的 offset；
- 向量访问（`vector<2xf32>` / `vector<4xf32>`）按 lane 展开为连续 scalar element 访问；
- 每个访问须有静态 element offset；动态 offset 直接报错，不做静默回退。

slot 分配由 init-defined `residentElements` 决定。先按 init section 的程序顺序确认每个 element 的首次访问为 store，再按 element offset 升序建立 `(fragment, element_offset) → slot` 映射。consumer 的访问集合不能缩减该映射；allocation 中未被 init 写入且不被访问的 padding 不占 slot。`i1/i8/i16/i32/f16/bf16/f32` 每个 resident element 占 1 个 slot；`i64` 每个 resident element 占 2 个 slot且起始 slot 偶数对齐。vector load/store 只是多个标量 element 的访问形式，不改变已确定的 resident set 和 slot 编号。当前实现按 alloca 出现顺序、resident element offset 升序分配。

**向量访问需特别处理**：生成的 IR 中 fragment 常以 `vector<2xf32>` 经字节 GEP（`getelementptr ... i8`，index 已乘 element 字节数）访问。对「同一 alloca、字节偏移寻址、标量与向量混用」这种形态，标准 mem2reg/SROA 无法提升。此处需按静态字节 offset 做定制切分（SROA 风格），将每个 offset 映射到确定的 slot，而非依赖现成 mem2reg。此为实现中风险最高的一处。

### 4. 边界插 resume/keep + 通用 mem2reg

直接在 load/store 上手动替换只对 write-once-read-many 成立，read-modify-write 会得到错误结果。采用的做法是在 section 边界插入 seed/save，再交由通用 mem2reg 穿线：

- **init section**：不生成 resume；出口对全部 `residentElements` 生成 `pto.keep`。
- **每个 carry section 入口**：对全部 resident slot 插入 `pto.resume`。本地访问的 element 用 resume 结果初始化 section-local proxy；未访问的 resident element 直接把 resume 结果传给出口 keep。
- **每个 carry section 出口**：对全部 `residentElements` 生成 `pto.keep`。本地更新过的 element 保存更新值，未访问的 resident element 原样 re-keep。
- 对 section 中本地访问的 element 创建 scalar proxy，将原 load/store 的地址改写到对应 proxy；未访问的 element 直接连接 resume 与 keep。
- 对每个 proxy 运行 mem2reg：入口 `store resume` 成为初始 SSA def，body 读-改-写成为正常 SSA 链，出口 `load` 取到链尾值并由 keep 保存，proxy 的全部 load/store 被提升。
- `pto.resume` / `pto.keep` 读写的是 slot payload 而非 alloca，mem2reg 不会触及——resume 成为链入口、keep 取到链出口。只读场景下 resume→keep 为自移动（可消除），读-改-写场景下 resume 为携带值、keep 为更新值，二者均正确。

物化按函数级 `sections` 顺序执行。每个 section 的入口操作统一插到 body 开头：先生成 proxy 使用的 count=1 常量，再生成完整 resume group，之后创建全部 proxy 和 seed store；所有 proxy 就绪后，按 section body 的 operation 顺序统一改写原访问。出口先为所有本地 proxy 生成最终 load，再统一生成按 slot 排序的连续 keep group。每个 proxy 单独调用 `tryToPromoteMemorySlots`，避免“多个 allocator 中只提升一部分”被误判为整体成功。

除 init section 外，不根据本地首次访问决定是否 resume；每个 carry section 一律 resume/keep 完整 `residentElements`。当前 persistent load/store 直接位于 section 顶层 block，proxy 的定义和使用保持线性。nested structured control flow 可以执行与 fragment 无关的计算，也可以使用顶层 fragment load 产生的普通 SSA value，但不能在 nested region 内访问或更新 fragment。条件写、循环携带 fragment 状态以及相应的路径合流由 Phase 4 处理；它们不改变边界 slot 集合，只改变每个 slot 在出口处的 reaching definition。

### 5. cleanup

mem2reg 提升后，persistent alloca 的派生 GEP 已无最终 load/store user。物化 pass 从 alloca result 出发递归检查并按叶子到根删除死 GEP；若仍存在非 GEP user 或 GEP 仍有 live use，则直接报错。最后确认 alloca result 无 use 后显式删除 `llvm.alloca {pto.persistent}`。proxy 使用的 count=1 常量在无 use 时同时删除，不依赖 outline 前额外运行 DCE。

### 6. outline

cleanup 后 section 中仅剩 slot-only 的 keep/resume。将每个 `pto.section.simt` outline 成 `func.func {pto.simt_entry}` + `pto.simt_launch`：init section → `@init_wfrag`，其余 section → 对应 SIMT helper。`@init_wfrag` 一次性 keep 完整 slot 集合，每个 helper 入口 resume、出口 re-keep，在调用链上形成 `keep →(resume…re-keep)×N`，中间不再回 UB 重读。

### 7. 两个 pass 的内部流程

`pto-analyze-simt-persistent-fragment` 和 `pto-materialize-simt-persistent-fragment` 共享同一个 `SIMTPersistentFragmentAnalysis`。前者是只读的诊断/checkpoint，后者消费不可变 plan 并修改 IR；analysis-only pass 不是 plan 的传输载体，materialize 即使在 pipeline 中没有显式经过 analysis-only pass，也会通过 `getAnalysis` 按需构造相同结果。

以下图中 pass 名仅作为标题，节点表示各 pass 实际完成的分析或 IR 处理。

#### 7.1 Analysis pass：构造 persistent fragment plan

```text
func::FuncOp / inline PTO IR
              |
              v
收集全部 pto.section.simt 与 pto.persistent llvm.alloca
              |
              v
检查 kernel-entry、词法作用域和 alloca 支配关系
              |
              v
沿 alloca → GEP → load/store use graph 追踪指针
              |
              v
折叠累计 byte offset，归一化 element offset 与 vector lane
              |
              v
按访问和 dominance 确定 init section、residentElements、carry sections
              |
              v
校验访问类型、边界、对齐、launch 维度和 resident 集合
              |
              v
按 fragment/element 顺序分配函数级 keep/resume slot
              |
              v
按 section、operation、lane 顺序将归一化访问写入
`ResidentElementPlan.accesses`（`AccessLane`）
              |
              v
缓存完整 immutable plan
              |
              v
只读 checkpoint 保留 analysis，IR 不发生修改
```

#### 7.2 Materialize pass：将 plan 物化为 slot traffic

```text
func::FuncOp / inline PTO IR
              |
              v
取得或按需构造 immutable persistent fragment plan
              |
              v
按函数 section 顺序建立 transform worklist
              |
              v
把每个 fragment 的完整 residentElements 绑定到 init/carry section
              |
              v
预先校验 section 归属、element 顺序、resident set 完整性、
              access ownership 和 lane 连续性
              |
              v
在 carry section 入口插入完整 resume group
              |
              v
为本地访问 element 创建 scalar proxy
              carry section 用 resume seed，init section 由已验证的首次 store 定义
              |
              v
按原 operation 顺序改写 scalar load/store
              并将 vector access 拆成 lane 级 proxy 读写后重建结果
              |
              v
在 section 出口读取 proxy 最终值，按 slot 顺序生成完整 keep group
              |
              v
逐 proxy 执行 mem2reg，删除无用的临时 count 常量
              |
              v
删除 dead GEP tree 与原 persistent alloca
              |
              v
得到 slot-only SIMT sections，交给后续 outline pass
```

处理边界如下：

- **analysis 构造阶段**只读取 IR。任何一个 alloca、访问、init、resident set、slot 或 lane 校验失败，均不发布部分 plan；`isValid()` 为 false。
- **analysis-only pass**只强制构造并验证 analysis，成功后 `markAllAnalysesPreserved()`，不插入或删除任何操作。
- **materialize preflight**先为全部 section 建立 worklist，校验 access/lane 唯一分配、fragment/element 顺序和完整 resident set；preflight 失败时不应留下部分 resume/keep。
- **materialize body**才创建 proxy、改写 load/store、执行 mem2reg 和生成 keep/resume；cleanup 完成后删除原 persistent alloca 及死 GEP。
- `pto-outline-simt-sections` 不属于上述 materialize pass 的内部步骤，而是消费 slot-only section 的后续独立 pass。

### 8. 分析 plan 到物化工作列表

analysis 对 `func::FuncOp` 发布的稳定结果是 `SIMTPersistentFragmentAnalysis` 中缓存的 `PersistentMaterializationPlan`。analysis-only pass 只强制构造和校验该结果；materialize 通过 `getAnalysis()` 取得同一个不可变 plan，不依赖 analysis-only pass 作为数据传输载体。随后 `buildPersistentTransformWorklist()` 将按 fragment/element 组织的 plan 转成按 section/operation 组织的临时物化工作列表；该转换只建立引用、局部访问集合和改写映射，不修改 IR。

```text
SIMTPersistentFragmentAnalysis (MLIR analysis cache)
└── plan: optional<PersistentMaterializationPlan>
    ├── sections[]: SectionSimtOp               // function walk order
    └── fragments[]: PersistentFragmentAnalysis // alloca walk order
        ├── allocaOp
        ├── initSection
        ├── carrySections[]
        └── residentElements[]                  // element offset order
            └── ResidentElementPlan
                ├── elementOffset
                ├── slot
                └── accesses[]: AccessLane
                    ├── op
                    └── laneIndex
```

顺序和所有权是 plan 的一部分契约：`sections` 按函数 walk 顺序保存，`fragments` 按 persistent alloca walk 顺序保存，`residentElements` 按 element offset 升序保存，`AccessLane` 按 section、operation 和 lane 顺序挂在所属 element 下。`elementOffset` 和 `slot` 由父级 `ResidentElementPlan` 提供，`AccessLane` 只保存原始访问 operation 和 vector lane 序号。

materialize 的 preflight 数据结构为：

```text
PersistentMaterializationPlan
fragment -> resident element -> access lane
                    |
                    | buildPersistentTransformWorklist()
                    v
PersistentTransformWorklist
section -> element work item
section -> operation -> lane rewrite
```

```text
PersistentTransformWorklist
└── sections[]: PersistentSectionWorklist       // 与 plan.sections 顺序一致
    ├── section: SectionSimtOp
    ├── elements[]: PersistentElementWorkItem   // active fragment/element 顺序
    │   ├── fragment*                           // 引用 immutable plan
    │   ├── residentElement*                    // 引用 immutable plan
    │   └── accesses[]: AccessLane              // 仅保留当前 section 的访问
    └── laneRewritesByAccess: Operation* -> PersistentLaneRewrite[]
        └── PersistentLaneRewrite
            ├── laneIndex                       // 原 scalar/vector access 的 lane
            └── elementIndex                    // 当前 elements[] 的下标
```

转换分为三步：

1. 为 `plan.sections` 中的每个 section 创建一个 `PersistentSectionWorklist`，保持函数 walk 顺序；没有 active fragment 的 section 也保留为空 worklist。
2. 对每个 fragment，把完整 `residentElements` 依次加入其 init 和全部 carry section。`PersistentElementWorkItem` 直接引用原 fragment/element，并从 `ResidentElementPlan::accesses` 中筛出位于当前 section 的访问；即使本 section 没有访问某个 resident element，该 element 仍在 `elements[]` 中，用于生成完整 resume/keep 集合。
3. preflight 按 section 校验 element 顺序、完整 resident set、访问归属和 lane 唯一性，再把 element-centric 的 `ResidentElementPlan -> AccessLane` 反向整理为 operation-centric 的 `Operation -> (laneIndex, elementIndex)`。该映射使同一个 vector load/store 可以一次完成所有 lane 的 scalar proxy 改写。跨 section 共享的 `assignedAccessLanes` 记录每个 `(operation, lane)` 至多分配一次，随后与 plan 对照确认没有遗漏；该集合不进入最终 worklist。

`materializeSection()` 随后创建与 `sectionWorklist.elements` 等长且下标平行的临时 rewrite state：

```text
rewrites[]: PersistentElementRewrite
├── proxy                            // 有本地访问时创建的 scalar alloca
└── resumeValue                      // carry section 中该 element 的入口值
```

`laneRewritesByAccess` 中的 `elementIndex` 用来取得 `rewrites[elementIndex].proxy`；它只是当前 section 的临时数组下标，不是原 fragment 的 `elementOffset`，也不是 persistent `slot`。`ResidentElementPlan::slot` 由 analysis 按函数范围分配并跨 section 保持稳定，同一个 element 的 `pto.resume` 和 `pto.keep` 均直接使用该 slot。对没有本地访问的 carry element，不创建 proxy，直接将 `resumeValue` 作为同 slot 的 keep payload。

分析成功由 `isValid()` 表示，materialize 只通过 `getPlan()` 以 `const` 方式读取该结果；analysis 失败时不发布部分 plan。没有 persistent alloca 时发布合法的空 plan；存在 persistent alloca 时，只有完整校验成功才发布非空 plan。

`FragmentShape`、pointer-use graph 的中间记录和 normalized access discovery 只在 analysis 内部用于类型、offset、bounds、resident set 和 lane 校验，plan 发布后销毁。`PersistentTransformWorklist`、section-local work item、lane rewrite 和 element rewrite state 都只属于 materialize，先完成全局 preflight 再修改 IR，物化结束后销毁，不进入 analysis cache。

## 流水接入位置

```text
PTODSL lowering
  → PTO IR（含内联 pto.section.simt，保留 structured control flow）
  → canonicalizer / SCCP / CSE
  → unroll section 内标注的循环
  → canonicalizer / SCCP / CSE
  → pto-analyze-simt-persistent-fragment（显式 analysis checkpoint）
  → pto-materialize-simt-persistent-fragment
  → outline pto.section.simt 成 func.func {pto.simt_entry}
  → SCF-to-CF（在 late outline 之后）
  → 现有 VPTO emission / LLVM lowering
```

流水顺序为“保留 structured control flow → persistent analysis/materialize → late outline → SCF-to-CF”。当前 SCF-to-CF 位于 LLVM emission pipeline 的 outline 之后。persistent pass 不依赖 section 的显式 CFG；`pto.section.simt` 的单顶层 block 可以容纳 nested structured control flow，RMSNorm 不需要多顶层 block section。

unroll 须在物化之前执行。slot 数由 init 确定写入的 `residentElements` 决定，不依赖后续访问 footprint。以 `threads=256`、layout capacity 为 32 的 `d=5120` RMSNorm 为例，init 只确定写入 lane-local element 0～19，因此分配 20 个 slot；allocation 中 element 20～31 是 layout padding，不参与驻留。init/consumer 的 load/store 位于静态循环中，不展开就无法把各访问归一到确定 element offset，也无法建立 resident set。展开并折叠 `%i` 后，物化 pass 才能把各访问连接到稳定 slot。`pto-unroll-loops`（旧名 `pto-unroll-simt-for`）同时处理 outline 前的 `pto.section.simt` 和 outline 后的 `pto.simt_entry`，只展开显式标注 `{pto.unroll = "full"}` 的循环。

### 自动提升：`pto-promote-persistent-fragment-loops`

早期版本要求调用方在访问 persistent buffer 的循环上**手写** `{pto.unroll = "full"}`；漏标会让 materialization 在静态访问集合缺失的情况下失败。`pto-promote-persistent-fragment-loops` 把这个前置条件自动化，插在 `pto-unroll-loops` 之前（`prepareVPTOForEmission` 内）：

- **识别**：以显式 `llvm.alloca {pto.persistent}` 为入口（不做结构推断），沿 alloca 的 pointer use graph（getelementptr 链会被跟随，load/store 及其他直接 consumer 视为终点 access；**不跟随** load 的数据结果——物化只要求每个 access 可归一到稳定 slot，仅消费 load 结果的循环与 buffer 无关）收集每个相关 op 的外层 `scf.for`——section 内直接包裹 access 的循环与多层嵌套的各层级；**例外（依赖感知）**：循环体内包含 `pto.section.simt` region、且其 induction variable 不出现在该循环体内任何 persistent GEP 的 index 中（沿纯整数算术链追踪）时才跳过——即循环只搬运数据、fragment 按 lane 常量索引的形态（rmsnorm persistent 的 tile 循环）。此时展开只会把 section 按 trip 克隆 N 份，SIMT outlining 为每个克隆生成单调用点的 linkonce_odr entry 函数，BiSheng 会把其中部分折叠/内联导致 `_simt_entry` ELF symbol 缺失，VPTO fatobj 发射在 VF_SIMT size patch 处失败（且代码体积按 trip 线性膨胀），而对 access 静态化零贡献。当 induction variable 参与 persistent GEP（按迭代选 fragment slot）时循环**仍被提升**——展开是这类 access 静态化的唯一途径；已端到端验证：无手写 hint 的依赖形态 kernel 由 pass 自动提升、编译通过、且每个克隆正确读取各自驻留 slot（数值逐 slot 校验一致）。静态零 trip（`ub <= lb`）的循环不会执行任何 access，整条 enclosing 链（含动态外层循环）都跳过不提升；
- **提升**：无 hint 或 `pto.unroll = "enable"` 的循环改写为 `pto.unroll = "full"`（覆盖 enable 是刻意的：保持 enable 会让 loop 保留到 metadata 阶段，materialization 看不到确定的静态访问集合）；已是 `"full"` 的保持不变；命中的循环附加内部 marker `pto.persistent_unroll`；
- **fail-fast（硬错误，绝不静默降级）**：persistent 循环上的 `pto.unroll_factor`；`scf.while` 内的 persistent access（该结构无法承载 full-unroll hint）；静态 trip 超过 `max-persistent-unroll-trip-count`（默认 128，pass option 可调，报错含函数名与 trip 值）。动态 trip 在 promotion 时无法检查，由 marker 把 `pto-unroll-loops` 的"丢 hint + remark"兜底升级为 persistent 专属硬错误——persistent loop 静默存活会破坏物化前提，必须 fail-fast。marker 随循环展开一同消失，不会残留在 IR 中。

该 pass 只做识别与标注：slot 分配、keep/resume 生成、SIMT outline、LLVM metadata lowering 仍归各自的 pass。

## 当前支持范围与限制

当前实现已完成内联 SIMT 表示、persistent fragment 分析、init/resident set 与函数级 slot 分配、scalar/vector keep/resume 物化、section-local proxy mem2reg 和 late outline 前 cleanup。当前支持标量以及一维 `vector<2xT>` / `vector<4xT>` 访问。

当前 `pto.section.simt` region 具有一个顶层 block。该 block 可以包含具有自身 region/block 的 `scf.if`、`scf.for`；fragment-transparent 的 nested structured control flow 由当前 pass 和 late outline 保留。

当前 persistent fragment 的 pointer/GEP/load/store 必须直接位于 section 顶层 block。顶层 fragment load 产生的普通 SSA value 可以进入 nested region，nested region 内与 fragment 无关的计算、分支和循环不受此限制。persistent access 位于 nested `scf.if` / `scf.for` 中时，由 Phase 4 处理。section 可以嵌套在外层 `scf.if`、`scf.for` 或普通 CFG block 中，init section 必须支配该 fragment 的全部访问 section。section region 的多顶层 block CFG 不在本方案范围内。

RMSNorm 的 reduce-all 控制流位于 `SimtVF` 对应的 section 顶层 block 内，不访问 persistent weight fragment；weight fragment 的 load/store 仍位于顶层 block。该形态属于当前实现范围。

### 当前非目标

- 不采用 persistent UB buffer 自动 peel 方案。
- 不在 PTOAS 里推断 TileLang fragment layout。
- 不新增 TileLang 前端 API。
- 输出为标量 keep/resume，不引入 `keep_range` / `resume_range`。
- 不处理动态 offset fragment。
- 不做跨复杂 CFG 的通用 mem2reg。
- 不处理 section region 的多顶层 block CFG。
- 不处理 persistent pointer/GEP/load/store 位于 nested `scf.if` / `scf.for` 中的条件访问、循环携带状态或多出口状态；fragment-transparent 的 nested structured control flow 已支持，路径相关 fragment 状态属于 Phase 4。
- 不根据 consumer 访问做 per-element liveness、末次使用分析或 resident set 裁剪。
- 不做优雅回退（超预算即 fail-fast）。
- 不处理已经 outline 后才出现的 persistent 推断。

## 资源预算、跨层契约与错误处理

`R4..R126` 共 123 个 slot。`M` 由所有 persistent allocation 的 `residentElements` 决定：每个 resident element 的 `slot_width(element_type)` 加上 64 位 element 的对齐 padding。consumer 实际访问较少不能降低 `M`，但 init 未写入的 allocation padding 不计入 `M`。理论上需满足 `M + body 峰值 + 余量 ≤ 123` 才执行变换，但 body 峰值取决于寄存器压力，需接近寄存器分配才能得到准确值，TIR 与 PTO 均无法准确估计。当前实现只按 resident set 的 `M` 判定，超出预算即报错。

**回退语义。** codegen 后 consume section 只保留对 `w_frag` 的访问，原 UB 地址表达式已删除，PTOAS 无法恢复 UB load。因此契约违反统一采用 **fail-fast**：

- fragment 定义点不支配 section use → 报错；
- fragment element 数量不可知 → 报错；
- init section 没有确定初始化任何 element，或 resident element 首次访问为 load → 报错；
- init 之后访问不属于 `residentElements` 的 element → 报错；
- 找不到唯一支配全部访问 section 的 init，或 carry section launch 维度不一致 → 报错；
- 访问 offset 不是编译期常量 → 报错；
- 访问类型无法拆成 keep/resume 支持的标量 → 报错；
- slot 数超预算 → 报错；
- persistent access 位于 nested region，或 section region 含多个顶层 block → 报错；
- fragment 同时被 SIMT 和非 SIMT 代码读写 → 报错或明确跳过。

### Layout 一致性

PTO 侧以 alloca 的 element type、element count 和 element offset 定义合法布局，以 init-defined `residentElements` 定义驻留对象。resident `element_offset = k` 在所有 section 中始终映射到同一 slot；consumer 可以只访问对象子集，边界 resume/keep 仍覆盖全部 resident element。各 section 的访问 offset 集合可以不同。

TileLang layout inference 负责保证同一 fragment 在各 `SimtVF` 中具有一致的 lane-to-element 映射。PTOAS 负责可验证的结构约束：init section 确定建立非空 `residentElements`；所有访问类型、对齐和 offset 均落在同一 allocation 内；init 后访问只能落在 resident set 内；carry section 的 `<<<dimX, dimY, dimZ>>>` launch 参数与 init section 一致。违反这些约束时直接报错，避免不同 lane 拓扑或错误 offset 产生静默数值错误。

### Keep/Resume 语义

不新增底层硬件语义，复用现有 `pto.keep` / `pto.resume`（`include/PTO/IR/VPTOOps.td`）：

- `pto.resume {slot = N}`：从固定寄存器 slot 取回一个元素；
- `pto.keep %v {slot = N}`：把一个元素存到固定寄存器 slot；
- slot → 物理寄存器的映射由 PTOAS 现有 lowering 负责（`slot N → R{4+N}`）；
- BiSheng 继续负责普通寄存器分配，PTOAS 只显式表达跨 VF 的 live-in/live-out slot。

当前实现直接生成多条标量 keep/resume；`keep_range` / `resume_range` 不在本方案范围内。

### 跨层对齐约定

| 项目 | 约定 |
|---|---|
| 内联 SIMT section | PTODSL emit `pto.section.simt` 并推迟 outline；PTOAS 在物化后执行 late outline |
| persistent 判定 | PTODSL 依 `alloc_buffer` 作用域合成 `pto.persistent`；PTOAS 消费并校验，不自行发现 |
| 对象边界 | allocation shape 定义合法地址范围；init 中确定完成初始化的 element 定义固定 `residentElements` |
| 初始化 | 唯一支配全部访问 section 的 init 建立非空 resident set；其它访问不得扩展 resident set |
| layout 一致性 | TileLang layout inference 保证 lane-to-element 映射；PTOAS 检查 shape、offset bounds 和 launch 维度 |
| fragment 大小 | PTOAS 需要静态得到 allocation 边界和 init 写入 offset，并按 `residentElements` 计算 slot 预算 |
| 访问约束 | 当前实现要求访问 offset 静态可解析；consumer 可以只访问对象子集 |
| 只读限制 | 不设只读限制，必须支持 read-modify-write |
| outline 顺序 | keep/resume 物化必须在 section outline 之前 |
| SCF-to-CF 时机 | 保留 structured control flow 完成 persistent analysis/materialize，late outline 后再做 SCF-to-CF |
| section 控制流 | region 保持一个顶层 block；fragment-transparent 的 nested `scf.if` / `scf.for` 已支持；persistent access 必须在顶层 block |
| 路径状态合流 | nested 分支/循环中的 fragment state 通过 path-sensitive SSA/block argument 合流；init 使用 definite-assignment；section 保持单一外层出口 |

## Phase 4：路径相关 fragment 状态

`pto.section.simt` 的 region 保持一个顶层 block。顶层 block 可以包含 `scf.if`、`scf.for` 等 nested structured control flow。当前实现要求 persistent fragment 的 pointer/GEP/load/store 直接位于该顶层 block；nested region 可以使用这些访问产生的普通 SSA value，但不能访问或更新 fragment。section 可以嵌套在函数外层的 `scf.if`、`scf.for` 或普通 CFG block 中，前提是 init section 支配该 fragment 的全部访问 section。

RMSNorm 的 reduce-all 控制流属于 fragment-transparent nested structured control flow，persistent weight 的访问仍位于 section 顶层 block；现有 section 表示和 pipeline 顺序适用。

Phase 4 支持 persistent access 位于 nested `scf.if` / `scf.for` 中时的路径状态分析和 SSA 合流。section 外层仍保持单顶层 block、单一外层出口；显式多顶层 block CFG 不在本方案范围内。

### 支持范围

| 场景 | persistent element 语义 |
|---|---|
| carry section 中单分支 `scf.if` | true 分支写入新值；false 路径保留 section 入口的 resume 值 |
| carry section 中完整 `scf.if/else` | 两个分支可以分别写入，也可以只有一个分支写入；merge 后选择对应路径的 reaching definition |
| 分支内 read-modify-write | 分支读取 resume 或前序定义，计算并写回；未执行该分支时保留旧值 |
| 嵌套 `scf.if` | 按每层 merge 关系合并 element 状态，出口得到单一 SSA 值 |
| init section 中条件初始化 | 每条可达路径都定义全部 `residentElements`；任一路径缺失定义即拒绝 |
| `scf.for` 循环携带值 | 循环入口使用 resume 或循环前定义，backedge 携带上一迭代值，循环退出值进入 keep |

循环控制值可以是动态值，但 persistent element offset 必须静态可解析。循环在 analysis 前完整 unroll 后，展开产生的顶层访问按当前规则处理；未展开且访问位于循环 region 内的情况由 Phase 4 处理。

### 实现方案

1. **递归访问发现。** 在 section 顶层 block 内递归遍历已支持的 `scf.if`、`scf.for` region。每个访问继续归一化为 `ResidentElementPlan` / `AccessLane`；控制流位置和路径信息作为 analysis 内部状态，不改变发布 plan 的 element、slot、lane 语义。nested section、`scf.while`、未知 `RegionBranchOpInterface` 和不可归约 CFG 直接报错。
2. **proxy 表达 fragment 状态。** carry section 在入口生成完整 resume group，并将每个 resume 值写入对应 scalar proxy；init section 由经过校验的初始化 store 建立 proxy 的初始定义。nested region 内的 fragment load/store 改写为 proxy load/store，普通 SSA 结果保持原有数据流。section 出口从 proxy 读取最终值并生成完整 keep group。
3. **路径状态合流。** branch merge 使用 block argument/PHI 表达 reaching definition，loop backedge 携带上一迭代值。未执行写入的路径显式传递入口 resume 值，不使用 poison 或 undef。
4. **late outline 后提升 proxy。** 含 nested SCF 的 section 先完成 persistent analysis/materialize，再 outline 为 SIMT helper；SCF-to-CF 在 helper 内执行，随后运行 proxy cleanup 和 mem2reg。简单线性 section 沿用 outline 前的 cleanup。
5. **definite assignment。** carry section 的 resume 为所有 resident element 提供入口定义。init section 使用前向数据流，在每个 block 记录所有到达路径均已定义的 element 集合，merge 使用 predecessor 集合的交集。每个出口必须定义完整 `residentElements`。
6. **边界 slot 保持不变。** 每个 carry section 的入口 resume 和出口 keep 都覆盖完整 `residentElements`，slot、类型和顺序与 analysis plan 一致；不根据控制流或 consumer 访问裁剪 resident set。

### 诊断边界

- persistent access 位于 section 顶层 block 之外；
- init 任一可达路径未完整定义 resident set；
- fragment 访问使用无法静态归一化的动态 element offset；
- branch predecessor 无法补充 block argument 或 branch operand；
- `scf.while`、异常式退出、未知 `RegionBranchOpInterface` 或不可归约 CFG；
- proxy cleanup 后仍存在 proxy alloca 或 persistent fragment load/store。

### 验收条件

- 单分支、双分支、嵌套 `scf.if` 的条件写和 read-modify-write 均得到正确 reaching definition；
- 未执行写入的路径保留 resume 旧值；
- init 的所有路径完整定义 resident set 时通过，缺失任一 element 时失败；
- `scf.for` 的 entry、backedge 和 exit 值正确；
- outline、SCF-to-CF、proxy cleanup 后通过 VPTO verifier 和 LLVM emission；
- resident set、slot 编号和 keep/resume 顺序保持 analysis plan 的定义。

Phase 4 的 pipeline 为：

```text
保留 structured SCF
  -> SIMT unroll / SCCP / canonicalizer / CSE
  -> persistent analysis + materialize
  -> late outline
  -> SCF-to-CF（outlined helper）
  -> persistent proxy mem2reg + cleanup
  -> VPTO validation / lowering
```

## 路线图

Phase 0～3 是当前实现基线，Phase 4 是路径相关 persistent fragment 状态扩展。

| 阶段 | 目标 | 范围 | 产出 |
|---|---|---|---|
| Phase 0 | 打通内联 section + 晚 outline | PTODSL emit 内联 SIMT section；PTOAS 新增 `pto.section.simt` op + outline pass | 前置依赖落地 |
| Phase 1 | outline 前 persistent fragment 物化 | 内联 section、单顶层 block（允许 fragment-transparent nested SCF）、标量 alloca、常量 offset、标量 keep/resume | pass + lit 测试 |
| Phase 2 | 向量访问拆标量 | 字节偏移寻址下的 `vector<2/4xf32>` 定制切分 | 覆盖 RMSNorm 的 vectorized 访问 |
| Phase 3 | slot 预算与工程化 | i64 对齐、超预算诊断、错误边界和端到端验证 | 工程化 |
| Phase 4 | nested access 的路径相关 fragment 状态 | 条件读写、SSA 合流、循环携带值、init definite-assignment；保持单一外层出口 | nested SCF/SSA 扩展 + 完整控制流 lit |

## 测试计划

lit 测试覆盖：

1. **init + 只读 consume**：kernel 作用域定义 fragment，init section store，循环内 consume section load。期望：init 出口生成 keep，consume 入口生成 resume、出口 re-keep，原 load/store 被删除。
2. **read-modify-write**：consume 内 load fragment、算新值、store 回。期望：resume 作为 SSA 链初值，keep 用更新后的值，不要求只读。
3. **多元素 fragment**：`[4 x f32]`，init 确定写入 offset 0/1/2/3。期望：生成连续 slot，每个 resident offset 对应稳定 slot，keep/resume 数量和编号符合预期。
4. **consumer 子集访问**：init 建立 resident set `{0,1,2,3}`，consumer 只访问 element 1。期望：边界仍 carry 全部 4 个 resident element，不根据 consumer 裁剪。
5. **无本地访问的中间 section**：init 与 consumer 之间插入不访问 fragment 的 SIMT section。期望：中间 section 仍生成完整 resume/keep group。
6. **allocation padding**：`[4 x f32]` 的 init 只确定写入 element 0～2，后续也只访问 0～2。期望：resident set 为 `{0,1,2}`，只分配 3 个 slot。
7. **访问非 resident element 报错**：init 只确定写入 element 0～2，后续访问 element 3。期望：在物化前报告 element 3 不属于 resident set。
8. **向量访问拆标量**：`vector<2xf32>` / `vector<4xf32>` 经字节 GEP 访问 fragment。期望：向量 load 由多个 slot assemble，向量 store 拆成多个 slot 更新，keep/resume 仍是标量。
9. **section 内局部 fragment 不处理**：fragment 定义在 section 内。期望：不生成 keep/resume，保持普通局部 fragment 语义。
10. **非法动态 offset 报错**：persistent fragment 用动态 GEP offset。期望：报错，提示 offset 必须静态可解析。
11. **launch 维度不一致报错**：init 与后续 carry section 的 SIMT 维度不同。期望：在生成 keep/resume 前报错。
12. **外层控制流中的 init**：init 和 consumer 位于同一个动态 `scf.if` 分支，且 init 支配 consumer。期望：允许 init 嵌套在外层控制流中。
13. **不存在统一 init**：两个互斥分支分别访问并初始化同一 fragment，没有任何 section 支配全部访问。期望：报告找不到唯一 dominating init。
14. **多个 fragment 与不同 init 点**：section 0 初始化 fragment A，section 1 carry A 并初始化 fragment B，section 2 使用 A/B。期望：各 section 包含正确的 active resident element，空本地访问仍保留，并按全局 slot 排序。
15. **fragment-transparent nested SCF（当前支持）**：单顶层 block 的 init/carry section 中插入不访问 fragment 的 `scf.if` / `scf.for`，并让顶层 fragment load 的普通 SSA 结果进入 nested region。期望：keep/resume 正常生成，nested structured control flow 保持，原 persistent load/store 被删除。
16. **nested persistent access 边界（当前负例）**：把 persistent load/store 放入 `scf.if` / `scf.for` region。期望：当前 pass 在 materialize 前报告 access 必须直接位于 section 顶层 block。
17. **Phase 4 条件状态合流**：carry section 的单分支写、双分支写和 read-modify-write。期望：merge 后得到正确的 reaching definition，未执行写入的路径保留 resume 值。
18. **Phase 4 初始化与出口**：init 的某条路径缺失 resident element 定义时失败；所有合法路径生成完整且顺序一致的 slot keep。

## 验证与性能记录

以下内容是当前实现和 benchmark 环境的验证快照，不改变前述规范性契约。

### 当前验证状态

- `ptoas` 构建通过。
- 标量三元素 init/consume 输入可完成 keep/resume 物化、late outline 和 VPTO pipeline。
- analysis-only、persistent materialize、inline section outline/unroll 相关的 30 个定向 lit 测试通过，包含 fragment-transparent nested SCF 正向和 nested persistent access 负向用例。全量 `check-pto` 当前为 752 个通过、128 个失败；失败均集中在现有 PTODSL daemon 环境的 `ImportError: cannot import name 'ir' from 'mlir'`，不作为本 pass 的回归结论。
- RMSNorm `d=4096, batch=4096` inline 输入可完成 persistent materialization、late outline 和 `--emit-vpto`；输出中不再包含 `pto.persistent`。

### 性能结果

端到端性能使用 TileLang RMSNorm PTO 后端生成代码作为优化前基线，persistent fragment 版本只把每个 token 重复执行的 W UB→寄存器读取提取到独立 init SIMT section，并通过 keep/resume 在后续 token SIMT section 间保持权重。

本节同时保留 PTO 后端修改前后的两组结果。两组 RMSNorm kernel 的算法、shape
和有效带宽计算方式相同，生成代码的差异在于 UB→GM 写回使用的 L2 cache mode：

| 数据组 | PTO `mte_ub_gm` codegen | UB→GM cache mode | GM→UB cache mode |
|:---|:---|:---|:---|
| 此前 cache mode | 未显式传递 `l2_cache`，使用 PTODSL 默认值 | `nmfv`（`NORMAL_FV`，raw value 0） | 不变 |
| 当前 cache mode | 传递 TileLang `l2_cache_ctrl` 默认值 4 | `naci`（`NOTALLOC_CI`，raw value 4） | 不变 |

因此，跨表差值反映 UB→GM cache mode 对端到端性能的影响；persistent fragment
自身的收益应在同一张表内，以对应 cache mode 下的 PTO backend 基线计算。

测试口径如下：

- 数据类型为 FP32，`batch=4096`，grid 为 64；
- 每组基线与两种优化版本均在同一张 Ascend NPU、同一套 CANN/PTOAS 构建上测试；
- 使用 TileLang `do_bench` 的 `msprof` backend，每次测量 warmup 30 次、repeat 20 次，并在采样间执行 L2 cache flush；
- 有效带宽按算法逻辑访存量 `2 * batch * d * sizeof(fp32) + d * sizeof(fp32) + batch * sizeof(fp32)` 计算，分别对应 X 读取、Y 写出、W 读取和 RSTD 写出；
- 正确性阈值为 `max_y_diff < 1e-3` 且 `max_rstd_diff < 1e-3`。

基线使用 `tilelang-ds/examples/ascend/example_rmsnorm.py` 的
`rms_norm_fwd`，通过 `tilelang.compile(..., target="pto")` 生成未做 persistent
fragment 物化的 kernel；优化版本分别使用
`example_rmsnorm_persistent_simtvf.py` 中的 UB → fragment 和 GM → fragment
实现。不同实现与 hidden size 的组合使用独立 TileLang cache，避免复用其他
组合的编译产物。当前 cache mode 的测试使用 TileLang
`92dcb62a`、本地 `/home/qukelin/projects/PTOAS/build/tools/ptoas/ptoas`
（`ptoas 0.55`）和 CANN 9.1；每个 shape 均完成 kernel 编译、warmup 和
`msprof` 采样。

#### 当前 cache mode：`naci`

当前 cache mode 下，每个实现与 shape 的组合启动 5 个独立 Python 进程完成 5 次
测量；同一组合复用已编译 kernel cache。表中延迟为 5 次结果的算术平均值，有效
带宽按平均延迟重新计算。单元格格式为“延迟 / 有效带宽”。

| Shape | resident slots | PTO backend 基线 | UB → fragment | GM → fragment |
|---:|---:|---:|---:|---:|
| `d=4096` | 32 | 120.059 us / 1118.2 GB/s | 115.022 us / 1167.2 GB/s | 117.207 us / 1145.4 GB/s |
| `d=5120` | 20 | 137.213 us / 1223.0 GB/s | 132.188 us / 1269.5 GB/s | 134.469 us / 1247.9 GB/s |
| `d=7168` | 28 | 187.979 us / 1249.7 GB/s | 179.402 us / 1309.5 GB/s | 182.577 us / 1286.7 GB/s |

按当前 cache mode 的测量均值计算，UB → fragment 的延迟降低 3.7%～4.6%、
有效带宽提升 3.8%～4.8%；GM → fragment 的延迟降低 2.0%～2.9%、有效带宽
提升 2.0%～3.0%。

#### 此前 cache mode：`nmfv`

下表保留修改 UB→GM cache mode 前的性能快照；数值沿用当时的测量记录，不按本轮
5 次重复测量重新聚合。

| Shape | resident slots | PTO backend 基线 | UB → fragment | GM → fragment |
|---:|---:|---:|---:|---:|
| `d=4096` | 32 | 80.6 us / 1665 GB/s | 70.625 us / 1900.9 GB/s | 70.342 us / 1908.5 GB/s |
| `d=5120` | 20 | 84.5 us / 1986 GB/s | 79.264 us / 2117.1 GB/s | 79.111 us / 2121.2 GB/s |
| `d=7168` | 28 | 127.3 us / 1845 GB/s | 117.471 us / 1999.9 GB/s | 119.344 us / 1968.5 GB/s |

按此前 cache mode 的测量值计算，UB → fragment 的延迟降低 6.2%～12.4%、
有效带宽提升 6.6%～14.2%；GM → fragment 的延迟降低 6.2%～12.7%、有效
带宽提升 6.7%～14.6%。

#### 正确性

| Shape | `max_y_diff` | `max_rstd_diff` | 正确性 |
|---:|---:|---:|:---:|
| `d=4096` | `1.907e-06` | `1.788e-07` | 通过 |
| `d=5120` | `2.861e-06` | `2.384e-07` | 通过 |
| `d=7168` | `2.861e-06` | `1.788e-07` | 通过 |

两种 cache mode 下的三个 shape 均通过正确性检查；当前 cache mode 的 45 次
测量样本全部通过。
