
# 定稿设计方案（final）

## 1. 目标与用户模型

`pto.tileop` 统一 custom subkernel 标识（取代 `pto.cube`/`pto.simd` 的 public subkernel 入口职责；`pto.simt` 专属 launched SIMT）。建模为 tile-level IR 上以 tile/tensorview/scalar 为 IO、带 phase 摘要的命名 helper + `func.call`，让 `PTOInsertSync`/`PTOPlanMemory` 当一等公民。用户零参数，**摘要全由后端 `PTOInferTileOpSummaryPass` 推导；canonical helper marker 统一收敛到 `pto.tileop.helper` unit attr。**

```python
@pto.tileop
def softmax(src_view: pto.TensorView, out_tile: pto.Tile, scratch_tile: pto.Tile,
            rows: pto.i32, cols: pto.i32):
    # caller 传入 scratch/out tile；body 不新建 tile buffer
    pto.tload(src_view, scratch_tile)        # MTE
    m = pto.vmax(scratch_tile)               # PIPE_V
    e = pto.vexp(pto.vsub(m, scratch_tile))  # PIPE_V
    s = pto.vsum(e)                          # PIPE_V
    r = pto.vdiv(e, s)                       # PIPE_V
    pto.tstore(out_tile, r)                  # MTE

@pto.jit
def kernel(out, x):
    softmax(x, out, scratch, rows, cols)
    softmax(x, out, scratch, rows, cols)     # 复用
```

约束（编译期强制）：
- IO 只允许 Tile/TensorView/PartitionTensorView/PTO scalar。**输出 tile/tensorview 全走 output operand + operand_effects=write/readwrite；func.call results MVP 只允许 scalar。**
- **helper 内禁止 `alloc_tile`/`reserve_buffer`/`TAlloc`/任何需 PlanMemory 为 callee-local 规划的 op；内部 tile buffer 必须来自 caller operand。内部 vreg/mask/scalar 临时可存但不跨边界。**
- body 允许 tload/tstore、vector ops、scalar(PIPE_S) ops、cube ops、`pipe_barrier` 同步。
- 不允许 host tensor/TensorSpec/vreg/mask/pipe_handle 跨边界；不允许 SIMT-only op。
- **tileop helper 之间不互相调用**（避免 helper 摘要和边界 effect 递归组合）。inline 后若形成同主域 `pto.section.*`，由后端 section 规范化展开；跨主域或无法判定语义的 section 混合仍拒绝。
- **负例：tileop 只有 MTE/S/sync、无 vector/cube 主计算证据时报错。**
- MVP 单主计算域（vector **或** cube）+ 多辅助 pipe；reject cube+vector 混算。多 phase 是 correctness 必需。

## 2. 关键后端事实（已核对，含 pipeline 实测）

- `PTOInsertSync`/`PTOPlanMemory` 均 `func.walk` 全递归进 region。
- InsertSync 现有两条 PTODSL subkernel 路径：legacy `simd/cube` 兼容路径仍按 helper role→单 pipe、memory operand **保守建模为 read+write**；`tileop` 路径已读 `primary_domain/phases/operand_effects`，按 **non-empty boundary-effect phase** 拆多 `CompoundInstanceElement`。
- **`CompoundInstanceElement`（`SyncCommon.h:334-341`）单 `kPipeValue` + 单组 def/use**；空 def/use 节点合法但不贡献跨边界依赖。
- **`classifyTileOpByPipe`（`PTONormalizeUncoveredTileSections.cpp:252-254`）把 MTE1 归 Cube**；`inferSegmentKind` 对混合段报错。
- **`normalizeFunction`（:740-764）**：对不带 kernel_kind 且不带 tileop helper marker 的 func 会 `collectUncoveredTopLevelSegments`→`inferSegmentKind`，混合段 `emitSegmentInferenceError` 失败。`hasKnownKernelKindContext` 现已把 canonical `pto.tileop.helper`（并兼容 legacy `pto.ptodsl.subkernel_helper = "tileop"`）视为已知上下文并直接跳过 NormalizeUncovered。
- **`VPTOSplitCVModule`**：`hasSectionKind`（:58-83）检查 func 含 `SectionCubeOp`/`SectionVectorOp`；不带 section 的 split candidate 被 `eraseSectionSplitCandidatesWithoutSectionKind`（:170-175）擦除；`:135` 要求"must contain section"。未带 `pto.kernel_kind` 的多域输入仍拒绝嵌套 section；已单域化的 `pto.kernel_kind` module 会按该主域展开同域 section、删除异域 section，避免 section 留到 VPTO LLVM lowering。
- **实测 pipeline 顺序（`ptoas.cpp:1780-1900`）**：
  ```
  preBackendPM: NormalizeUncoveredTileSections (1786)
  main pm: ... → PTOInferTileOpSummaryPass → PTOMaterializeTileOpSectionsPass
           → PTOVerifyTileOpContractPass → ... → ViewToMemref
           → PlanMemory → ResolveReservedBuffers
           → VerifySubkernelPipeContract → InsertSync → ... →
           MaterializeTileHandles → InlineBackendHelpers
  ```
- MLIR attribute 不能引用 SSA value；PTO 无 ValueAttr 机制；custom attr 需在 `PTOAttrs.td` 注册（`PTO_Attr` 基类 :36）。
- `alloc_tile`（`PTOOps.td:318`）、`reserve_buffer`（:1792）、`TAllocOp`（:2240，PlanMemory:478 处理）真实存在。

## 3. IR 载体：方案 B — named helper + `func.call` + 后端推导的 phase 摘要

复用 `func.call` + 命名 helper + callee 解析。**不复用 `kernel_kind`**，用后端推导的 `pto.tileop.primary_domain`。**不预套 section**，改 verifier + NormalizeUncovered 接受 tileop 裸 body。**前端只标 `pto.tileop.helper`，摘要全后端生成。**

```
// 前端 trace 后：helper 只带统一 marker
func.func @softmax(%src: !pto.tensorview<...>, %out: !pto.tile<...>, %scratch: !pto.tile<...>,
                   %rows: i32, %cols: i32)
    { pto.tileop.helper } {
  pto.tload %src, %scratch
  %m = pto.vmax %scratch
  ...
  pto.tstore %out, %r
  return
}

// PTOInferTileOpSummaryPass 后：补全摘要（真 MLIR attr 结构）
func.func @softmax(%src, %out, %scratch, %rows, %cols)
    { pto.tileop.helper,
      pto.tileop.primary_domain = #pto.kernel_kind<vector>,
      pto.tileop.phases = #array<#dict<{
        pipe = #pto.pipe<MTE1>, operand_uses = [0], operand_defs = [2], result_defs = []
      }, #dict<{
        pipe = #pto.pipe<V>,    operand_uses = [2], operand_defs = [1], result_defs = []
      }, #dict<{
        pipe = #pto.pipe<MTE1>, operand_uses = [1], operand_defs = [1], result_defs = []
      }>>,
      pto.tileop.operand_effects = ["read", "readwrite", "readwrite", "read", "read"]
    } {
  ...
}

func.func @kernel(%out, %x, %scratch, %rows, %cols) {
  func.call @softmax(%x, %out, %scratch, %rows, %cols) : (...) -> ()
  func.call @softmax(%x, %out, %scratch, %rows, %cols) : (...) -> ()   // 复用
}
```

### phase attr schema（真 MLIR 结构，需在 PTOAttrs.td 注册）

- `pto.tileop.phases`: `ArrayAttr<DictionaryAttr>`，每 phase dict：
  - `pipe`: 复用现有 pipe 整数枚举（`PTOAttrs.td:213-227`）或新 `PipeAttr`（需注册），按 op `getPipe()` 推。
  - `operand_uses`: 整数 `ArrayAttr`（operand index 指向函数所有 operands；**InsertSync 只消费 memory-like operand**，scalar 可在 summary 供验证或忽略、不参与建图；可空）。
  - `operand_defs`: 整数 `ArrayAttr`（同上，可空）。
  - `result_defs`: 整数 `ArrayAttr`（**MVP 固定空或仅 scalar result，不参与 memory sync**；复杂语义后置）。
- **effects 可空**：纯内部 phase 可全空，保留在 phases 用于校验/主域推导，**InsertSync 跳过不建 `CompoundInstanceElement`**。有 boundary effect 的 phase 才建节点。是否标 use/def 是 **policy 非 IR 不变量**。
- `pto.tileop.operand_effects`：从 phases 非空 effects 派生（union use→read、def→write）；无显式 boundary effect 的 operand 默认标 read。scalar 标 read 但不建图。
- `pto.tileop.primary_domain`：主计算域 vector/cube（借用枚举值，不挂 `kernel_kind` attr）。
- **去掉 `pipe_footprint`**：body pipe set 由 `phases.pipe` 集合表达。

### operand index 作用域（明确）

- index 指向**函数所有 operands**（含 scalar）。
- InsertSync 只消费 **memory-like operand**（tile/tensorview）的 use/def 建 `CompoundInstanceElement`。
- scalar operand 可在 summary 供验证或忽略；**不参与 def/use 建图**。

### 摘要属性职责划分

| 职责 | 由谁承担 |
|---|---|
| body 出现过的 pipe 集合 | `phases.pipe` 集合（verifier 校验 body op getPipe() ∈ 此集） |
| caller 跨边界 sync 建模 | 有 boundary effect 的 phase（memory-like operand use/def 非空），InsertSync 为其建 `CompoundInstanceElement` |
| 主计算域 | `primary_domain` |
| 每 operand 副作用 | `operand_effects`（从非空 phase effects 派生；scalar 标 read 但不建图） |

> 不保留 `has_sync`：InsertSync 假设 helper 内部自管同步、caller 层只管跨边界。

### 输出/results 边界（MVP 硬约束）

- 输出 tile/tensorview 全走 output operand + operand_effects=write/readwrite。
- **func.call results MVP 只允许 scalar**（alias handle 后置）。
- **helper 内禁 `alloc_tile`/`reserve_buffer`/`TAlloc`**；内部 tile 必须来自 caller operand；内部 vreg/mask/scalar 临时不跨边界。

## 4. MVP 边界

- 单主计算域 + 多辅助 pipe；reject cube+vector 混算。
- **多 phase correctness 必需**，pipe 按 `getPipe()` 推；effects 可空（非空才建 sync 节点），softmax V phase 标 use/def 是 policy。
- MTE1/2/3/4、PIPE_S 归 phase、不参与 primary_domain 判定（tileop 专用规则，不改全局 `classifyTileOpByPipe`）。
- 禁 helper-local tile allocation。
- SIMT-only op 排除。**tileop helper 禁止调用另一 tileop helper**；同域 section 嵌套由后端规范化展开，异域/混合 section 不作为合法用户模型。
- **负例：tileop 只有 MTE/S/sync 无主计算证据报错。**

## 5. 改动点

### 前端（`_subkernels.py` + `_tracing/session.py`）

1. `_create_subkernel_section_op`：tileop **不预套 section**。
2. helper 只附 `pto.tileop.helper`；**不写 primary_domain/phases/operand_effects**。
3. helper 函数类型：输出全走 operand；results 只用于 scalar。
4. 前端 public boundary 契约：保留 vreg/mask 不外逃；results 限 scalar；**禁 tileop helper 互调**。helper body 内 `alloc_tile/reserve_buffer/TAlloc` 等 helper-local 资源分配由后端 `PTOVerifyTileOpContractPass` 兜底拒绝。
5. 装饰器无 `kind` 参数；public surface 仅保留 `@pto.tileop` / `@pto.simt`，legacy `@pto.cube` / `@pto.simd` 前端直接报迁移诊断。

### 后端

1. **verifier 改造**：tile op verifier 把带 `pto.tileop.helper` 的 func 当合法上下文；results 限 scalar；**拒绝 alloc_tile/reserve_buffer/TAlloc**；内部 vreg/mask/scalar 临时不跨边界。
2. **`PTONormalizeUncoveredTileSections` 跳过 tileop**（P0）：`normalizeFunction`/`hasKnownKernelKindContext` 增条件——带 `pto.tileop.helper` 的 func 跳过，避免 preBackendPM:1786 扫到裸 body 混合段报错。
3. **不把 tileop materialize 合并进 `PTONormalizeUncoveredTileSections`**：NormalizeUncovered 面向普通函数的 top-level uncovered tile segment，不消费 tileop summary；tileop helper 需要先由 `PTOInferTileOpSummaryPass` 推导 primary_domain/phases/effects，再只包主计算 span，并保留 MTE/S/sync 在 section 外。因此 tileop 走 helper-local、summary-driven 的 `PTOMaterializeTileOpSectionsPass`，NormalizeUncovered 只负责跳过 tileop helper，避免早期按普通 segment 规则误判。
4. **新增 `PTOInferTileOpSummaryPass`**：扫 helper body 推导 primary_domain + phases（pipe 按 `getPipe()`，effects 可空，operand index 指向所有 operand，memory-like 才建图）+ operand_effects（从非空 effects 派生）。tileop 专用 MTE/S 规则，不改全局 `classifyTileOpByPipe`。
5. **新增 materialize pass**：按 primary_domain+phases 物化 `SectionCubeOp`/`SectionVectorOp`（只包 cube/vector 主段），MTE/S/sync 保持 top-level。**lit case 覆盖两类**：MTE+section.vector+MTE、MTE+section.cube+MTE，验证 `VPTOSplitCVModule`/EmitC/VPTO 接受（注意 `hasSectionKind`:58-83 要求 func 含 section，`eraseSectionSplitCandidatesWithoutSectionKind`:170-175 会擦除无 section 的 candidate）。
6. **`UpdatePTODSLSubkernelCallInfo` 改造**：读 primary_domain+phases；按**有 boundary effect 的 phase**拆多 `CompoundInstanceElement`（空 effect phase 跳过，scalar operand 不建图）；memory operand 副作用从保守全 R+W 改读 operand_effects；支持 callsite scalar results 进依赖图。
7. **新增 `PTOVerifyTileOpContractPass`**：校验 body op `getPipe()` ∈ phases pipe 集合、主域 pipe 与 primary_domain 一致、operand_effects == 非空 phases 派生、SIMT-only op 排除、cube+vector 混算 reject、results 限 scalar、tileop helper 无互调、**拒绝 alloc_tile/reserve_buffer/TAlloc**、**负例（只有 MTE/S/sync 无主计算证据报错）**。旧 `PTOVerifySubkernelPipeContractPass` 保留兼容 cube/simd。
8. 主 pipeline 不再依赖 `PTOWrapFunctionsInSectionsPass` 为 tileop helper 自动套单段；tileop section 形成以 `PTOMaterializeTileOpSectionsPass` 为准。
9. **`PTOInlineBackendHelpers`**：保证不丢围绕 call 的 sync ops；后续 `VPTOSplitCVModule` 会在单域 VPTO module 内展开同域 section、删除异域 section，避免 section 容器遗留到 VPTO LLVM lowering。

### pass 顺序（实测修正，P0）

```
前端 trace → verifier(tileop 裸 body)
→ [preBackendPM] NormalizeUncoveredTileSections (跳过 `pto.tileop.helper`)   ← P0 必须跳过
→ [main pm] PTOInferTileOpSummaryPass
→ PTOMaterializeTileOpSectionsPass
→ PTOVerifyTileOpContractPass
→ ... → ViewToMemref → PlanMemory → ResolveReservedBuffers →
   VerifySubkernelPipeContract → InsertSync → ... →
   MaterializeTileHandles → InlineBackendHelpers
```

### 可选后置（非 MVP）

- helper-local tile allocation 的 callsite clone/inline（放开 result 返回 tile / alloc_tile 等）。
- alias handle result（放开 result 非 scalar / result_defs 复杂语义）。
- phases def/use 细到 UB 子区域。
- 内联 opt pass。

## 6. 当前落地状态

已落地并与本文主设计一致的部分：

1. `NormalizeUncoveredTileSections` 已把 tileop helper marker 视为已知上下文并跳过预归一化。
2. `PTOInferTileOpSummaryPass`、`PTOMaterializeTileOpSectionsPass`、`PTOVerifyTileOpContractPass` 已接入主 pipeline，且都位于 `PlanMemory` 之前。
3. `UpdatePTODSLSubkernelCallInfo` 已能消费 tileop 摘要，按 phase 建模跨 helper 边界的 InsertSync 依赖；legacy `simd/cube` 兼容路径仍保留保守单-pipe 建模。
4. tileop helper ABI 已收敛为 Tile/TensorView/PartitionTensorView/PTO scalar；`ptr` 仍为 SIMT-only。
5. `@pto.tileop` 在 IR 层语义上统一到 tileop helper role，并使用 canonical `pto.tileop.helper` marker；legacy public `@pto.simd` / `@pto.cube` 已前端报错；后端仍兼容 legacy `pto.ptodsl.subkernel_helper = "tileop"`。
6. inline `with pto.tileop()` 已与 decorated `@pto.tileop` 对齐：前端不再预套 section，统一交后端 `PTOMaterializeTileOpSectionsPass` 物化。
7. `VPTOSplitCVModule` 已支持已单域化 `pto.kernel_kind` module 的 section rewrite：同域 section 展开，异域 section 删除，避免 tileop inline 后的 section 容器进入 VPTO LLVM lowering。

## 7. 当前默认 effect 策略

`pto.tileop.operand_effects` 对无显式 boundary effect 的 operand 默认记录为
`"read"`，与当前实现和 verifier 保持一致。

这样做的含义是：如果 helper body 没有明确写某个边界 operand，summary 不凭空为
它制造 write 依赖。真正读写边界 tile/view 的 op 必须通过 MemoryEffect 和
boundary value trace 被记录到 phase 的 `operand_uses` / `operand_defs` 中；如果
真实写入被漏掉，那是 op effect 或 trace 规则缺失，需要补对应 op 的 effect 建模，
而不是靠默认值掩盖。
