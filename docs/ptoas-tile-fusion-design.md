# Tile Fusion 设计说明文档

## 1. 引言

### 1.1 背景与动机

PTO 指令在 Davinci 架构上以 Unified Buffer（UB）中驻留的数据块（Tile）为操作对象。这种模型虽然表达能力很强，但在多条 PTO 指令连续执行时会引入两类开销：

- **访存开销**：每条 PTO 指令（如 `pto.tadd`）将输入从 UB 读入向量寄存器，计算后再将结果写回 UB。当多条指令串联时，中间结果需要经历 UB 的一次完整往返——由生产者写入、再由消费者重新读出——产生冗余的带宽消耗。
- **控制开销**：每条 PTO 指令会展开为一组以 Tile 形状为参数的硬件循环。多条 PTO 指令意味着多组循环初始化、分支预测和指令发射。

**示例：`D = Relu(Add(A, B))`**

*非融合执行：*

1. **循环 1 (Add)**：从 UB 读取 Tile A, B → 执行加法 → 结果 Tile C 写回 UB。
2. **循环 2 (Relu)**：从 UB 读取 Tile C → 执行 ReLU → 结果 Tile D 写回 UB。

*瓶颈*：Tile C 被写入 UB 后立即又被读回——这是一次完全冗余的往返。此外两组独立的循环控制器被分别创建和销毁。

*融合执行：*

1. **单一循环**：从 UB 读取 Tile A, B → 执行加法 → **寄存器级直传** → 执行 ReLU → 结果 Tile D 写回 UB。

*收益*：消除了中间 Tile C 的 UB 读写。两组循环控制器合并为一组，循环管理开销减半。

**数据流对比：**

```text
      [非融合模式]                           [融合模式]
      (循环1: Add)                        (单一循环)
    ┌───────────────┐                     ┌───────────────┐
    │ Read A, B     │                     │ Read A, B     │
    │      │        │                     │      │        │
    │ Compute Add   │                     │      │        │
    │      │        │                     │      │ (Reg)  │
    │ Write C to UB │──┐ (UB 延迟)        │ Compute Relu  │
    └───────────────┘  │                  │      │        │
    (循环2: Relu)      │                  │ Write D to UB │
    ┌───────────────┐  │                  └───────────────┘
    │ Read C fr UB  │◀─┘
    │      │        │               >> 消除中间写回/读取
    │ Compute Relu  │               >> 合并循环控制逻辑
    │      │        │
    │ Write D to UB │
    └───────────────┘
```

**现有瓶颈总结**：串联 PTO 指令间频繁的 UB 读写导致带宽争抢、循环控制指令占比过高、以及计算单元在等待访存时空闲。

### 1.2 设计目标与范围

#### 1.2.1 融合范围

本特性首期聚焦于 `docs/PTO_IR_manual.md` 中所有硬件映射为 **Vector Pipeline（`PIPE_V`）** 的 PTO 指令。

**支持的指令：**

| 类别 | 指令 |
|---|---|
| 向量算术运算（逐点） | `pto.tadd`, `pto.tsub`, `pto.tmul`, `pto.tdiv`, `pto.tmax`, `pto.tprelu` |
| 向量-标量算术运算 | `pto.tadds`, `pto.tsubs`, `pto.tmuls`, `pto.tmaxs` |
| 向量规约与广播 | `pto.trowsum`, `pto.tcolmax`, `pto.trowexpand`, `pto.tcolexpand` |
| 向量位运算 | `pto.tand`, `pto.tor`, `pto.txor` |
| 局部数据搬运 | `pto.tmov`（ACC ↔ VEC 域移动）, `pto.ttrans`（转置） |

**不在范围内：** Matrix Pipeline（`PIPE_M`）的矩阵乘法指令和 DMA Pipeline（`PIPE_MTE`）的数据搬运指令不参与内部融合逻辑，但它们可以作为融合链路的起点或终点。

#### 1.2.2 融合准则

并非任意两条 `PIPE_V` 指令都可以融合。必须满足以下核心准则：

##### 准则一：迭代空间一致性

参与融合的指令必须在相同的逻辑迭代域内执行。对于 PTO Tile 而言，这意味着其逻辑计算形状（有效形状，即 `v_row` 和 `v_col`）必须能够对齐到同一个循环空间。

- **物理与逻辑解耦**：物理形状决定内存分配，可以不同；逻辑形状决定计算范围，必须一致。
- **可融合示例**：
  - *场景 A（完全匹配）*：OP1 和 OP2 的逻辑形状均为 `16×128`。融合后生成单一 `16 × 128` 硬件循环。
  - *场景 B（物理布局不同）*：OP1 的 Tile 物理大小为 `32×128`，有效区域为 `v_row=16, v_col=128`；OP2 的 Tile 物理大小为 `16×128`，有效区域亦为 `16×128`。两者在相同的逻辑域内迭代——可融合。
  - *场景 C（异构输出形状、同域）*：`OP1: C = Add(A, B)`（输出 Tile 16×128）后接 `OP2: d = RowSum(C)`（输出向量 16×1）。迭代域兼容。
- **不可融合示例**：
  - *场景 D（逻辑形状冲突）*：OP1 逻辑形状 `16×128`，OP2 逻辑形状 `8×128`。迭代边界不一致会导致循环次数错误，首期不支持此类非对齐融合。
  - *场景 E（动态形状无法证明）*：OP1 的 `v_row` 为动态值 `%M1`，OP2 的 `v_row` 为动态值 `%M2`。若编译器无法静态证明 `%M1 == %M2`（例如通过符号分析追溯两者来自同一个 `pto.get_tensor_view_dim`），则保守拒绝融合。

##### 准则二：数据依赖与映射规则

- **逐点指令的普适性**：逐点指令（如 `Add`, `Mul`, `Relu`）不改变数据索引映射关系，是构建融合链的最灵活基础单元。
- **弱依赖规则**：融合不强制要求指令间存在数据流依赖。处于同一迭代空间且无依赖的并行独立指令仍可通过"控制流融合"受益——合并循环逻辑以降低控制开销。
- **深度融合规则**：当指令间存在生产者-消费者关系且满足逐点对逐点（1:1）映射时，触发"访存消除融合"，利用寄存器级直传彻底省去中间 Tile 的 UB 读写。
- **规约适配规则**：在 A5 硬件支持下，逐点指令与规约（Reduce）指令的组合同样遵循深度融合规则，支持在寄存器内完成即时累加并消除访存。
- **示例**：
  - *场景 F（并行无依赖融合）*：`OP1: C = Add(A, B)` 和 `OP2: F = Mul(D, E)`。虽无数据依赖，但合并为单一循环可降低控制开销并提升计算单元利用率。
  - *场景 G（逐点深度融合，1:1 映射）*：`OP1: C = Add(A, B)`, `OP2: D = Relu(C)`。最典型的 1:1 场景。`C` 在寄存器中直接被 `Relu` 消费，彻底消除了中间 Tile 的物理内存分配。
  - *场景 H（跨模式融合，逐点 + 规约）*：`OP1: C = Add(A, B)`, `OP2: d = RowSum(C)`。在 A5 上，`RowSum` 指令可在向量寄存器内跨列累加的同时完成加法运算。融合不仅合并了循环，还消除了大尺寸 Tile `C` 在 UB 上的存储和加载开销。

##### 准则三：内存布局兼容性与弹性适配

**标准规则（布局一致）**：参与融合的输入/输出 Tile 在元数据和硬件访问模式上必须完全兼容：

- **元数据对齐**：Tile 的基础布局（`blayout`，如行优先/列优先）、二级布局（`slayout`）、分形大小（`fractal`）以及数据类型（`dtype`）必须完全一致。
- **内存映射一致**：在硬件层面，逻辑索引为 `(i, j)` 的元素在寄存器序列中的物理位置，无论由哪条指令访问都必须相同。
- **不匹配的后果**：若布局不同（例如 OP1 输出为行优先，OP2 要求列优先输入），即使迭代空间相同，元素在寄存器中的排列顺序也是"错位"的。因此标准规则要求物理布局对齐，以支持零开销的寄存器级传参。

**特殊场景适配：**

- **虚拟布局变换**：若中间指令为纯布局转换（如 `pto.ttrans`），编译器可将布局转换开销吸收到计算循环的索引映射中，通过调整向量寄存器访问步长或偏移来消除物理转置代价。
- **硬件重排加速**：利用 A5 硬件在向量寄存器内部的混叠/排列指令，在计算过程中实时调整数据顺序，从而支持不同布局指令的融合。

**示例：**

- *场景 I（转置消除融合）*：`OP1: B = Transpose(A)`（通过 `pto.ttrans`），`OP2: C = Relu(B)`。若单独执行，`Transpose` 需要在 UB 上进行物理搬移。通过融合，编译器可生成一个"按列遍历"的 `Relu` 计算核心直接处理 Tile `A`。逻辑上的中间变量 `B` 消失了，物理转置的冗余访存也被彻底消除。

##### 准则四：寄存器与硬件参数预算约束

- **物理寄存器限制**：Davinci A5 拥有 32 个向量寄存器（`V0`–`V31`）。融合链中所有活跃变量（输入、中间结果、临时变量）总数不得超过此阈值，否则会产生寄存器溢出（Spill），导致数据被迫写回 UB/L1，严重拖慢性能。
- **VF 参数限制**：向量函数（Vector Function, VF）硬件循环的参数列表（包括 Tile 物理地址、步长 Stride、形状 Shape 等元数据）上限为 32。如果融合后的单一大循环涉及过多独立 Tile 对象，将导致 VF 调用参数超限。
- **示例**：
  - *场景 J（参数与寄存器双重超限）*：尝试将一个涉及 18 个不同输入项的复杂融合链（如 18 路 Tile 加法）进行融合。
    - *分析*：(1) **参数溢出**：18 个输入 Tile 地址 + 1 个输出 Tile 地址 + 对应的 Stride 元数据，总参数数量极易触碰 32 个 VF 参数的硬件上限。(2) **寄存器溢出**：若循环展开导致超过 32 个 128-bit 向量值同时处于活跃状态，硬件无法在不写回内存的情况下维持寄存器级传参。
    - *决策*：此时编译器必须在中间点强制截断融合链，将部分结果写回 UB，在两个独立的 VF 循环中执行。

##### 准则五：典型支持模式

- **线性链路**：`A → B → C` 的连续逐点运算。
- **并行独立融合**：将处于同一迭代空间、无直接数据流依赖的多条指令组合进同一硬件循环，降低控制开销。
- **广播融合**：`pto.trowexpand` / `pto.tcolexpand` 后紧跟逐点运算。
- **末端规约融合**：一系列逐点运算后紧跟 `pto.trowsum` 或 `pto.tcolmax`。
- **规约 + 逐点**：规约指令的输出向量直接作为后续逐点指令的输入。
- **逐点 + 广播**：逐点运算的结果 Tile 直接作为广播指令的输入，在寄存器层面完成扩展。
- **规约 + 广播**：常见的 Softmax 结构优化。规约得到的行/列极值直接通过广播扩展回 Tile 形状进行后续计算。

### 1.3 核心设计思想

核心思路是在 MLIR 层面将多条 PTO 指令的循环体进行融合，通过向量寄存器直接传递中间数据。这实现了以下优化目标：

- **减少 UB 访存**：通过寄存器级数据流水化替代物理 Tile 的中间写回与读取，消除冗余带宽消耗。
- **降低控制开销**：将多组硬件循环合并为单一循环结构，减少循环初始化、分支预测及指令发射开销。
- **提升计算效率**：通过指令重排与循环展开，使向量计算单元尽可能保持在满载状态。
- **可预期性**：融合边界显式且可预测，上层编译器框架可以明确预期哪些指令组合能够获得寄存器级加速。

---

## 2. 硬件架构与约束

### 2.1 Davinci A5 存储层次与访存成本

- **Global Memory (GM)**：百 GB/s 级带宽，高访存延迟（数百周期）。
- **Unified Buffer (UB)**：TB/s 级带宽，中等访存延迟。PTO 指令默认以此为输入输出媒介。
- **向量寄存器 (Vector Register)**：指令级单时钟周期访问。

**融合驱动力**：PTO 指令间若通过 UB 传递数据，会受限于 UB 的读写带宽；通过寄存器直传可将数据流提升至寄存器级带宽，消除物理存取延迟。

### 2.2 向量计算流水线与执行机制

- **双发射流水线**：Davinci 支持两组向量计算流水线（Pipe V），通过合理的指令调度可隐藏计算延迟。
- **硬件循环 (Vector Function)**：内置硬件循环控制器，初始化开销不可忽略。融合后的"单大循环"能有效摊薄循环头（Prologue）和循环尾（Epilogue）的指令周期。
- **向量掩码 (VMSK)**：用于处理非对齐边界。融合时需确保不同指令的掩码逻辑在同一个循环迭代内是可组合的。

### 2.3 物理对齐与 Tile 布局约束

- **32B/512B 对齐要求**：Davinci 架构对 UB 地址及 Tile 宽度有严格的对齐要求。
- **Padding 填充**：当逻辑（有效）形状小于物理形状时，硬件会在物理 Tile 边缘填充无效数据。融合时必须严格限定计算边界，防止对无效填充区进行错误累加。
- **分形布局机制**：针对矩阵/分形操作的特殊布局。向量运算在处理此类布局时需要额外的混叠或步长计算。

---

## 3. 设计难点与挑战

### 3.1 寄存器压力

- **溢出风险**：融合多条指令会显著增加活跃变量数量。若所需向量寄存器超过硬件上限，会产生寄存器溢出，导致数据写回 UB 或 L1，抵消融合的访存收益。
- **生命周期重叠**：融合后的长循环会延长中间结果的生命周期，增加寄存器分配算法的复杂度。

### 3.2 内存布局与对齐

- **布局不匹配**：例如生产者指令输出行优先数据，但消费者指令要求列优先输入。融合此类指令可能需要插入昂贵的转置或混排指令，抵消访存收益。
- **非逐点操作**：规约或数据搬运指令的融合涉及复杂的索引变换和同步逻辑。

### 3.3 循环控制与同步

- **循环结构差异**：具有不同遍历范围或步长的指令融合时，需要精细的循环剥离或填充技术。
- **细粒度同步**：在某些流水线架构中，融合后的长指令序列可能导致硬件流水线死锁或违反数据依赖顺序。

### 3.4 成本模型与边界决策

- **贪心策略的局限性**：过度融合可能导致代码膨胀和指令 Cache 缺失。
- **自适应决策**：在不同 Tile Shape 和硬件配置下自动决定最优融合边界，在一般情形下是一个 NP-hard 问题。

---

## 4. IR 表示与流水线集成

### 4.1 PTO Dialect 背景

融合系统工作在 PTO Tile 级 Dialect 之上，包括 `pto.alloc_tile`、`pto.load_tile`、`pto.store_tile` 等核心指令以及 §1.2.1 中列出的计算指令。这些指令现有的 Lowering 路径构成了融合机制的基础。

### 4.2 融合流水线位置

#### 4.2.1 生效条件

融合仅在 `tools/ptoas/ptoas.cpp` 的 A5 EmitC 或 VPTO 后端主线上生效，需同时满足以下全部条件：

- `--pto-backend=emitc` 或 `--pto-backend=vpto`
- `--pto-arch=a5`
- `--pto-level=level2` 或 `--pto-level=level3`
- 显式传入 `--enable-op-fusion=true`；未设置或传入 `false` 时均保持关闭

迭代域推导由 `--enable-shape-inference` 开关控制（同样作用于 EmitC 后端的
`FusionPlan`）：

- **默认开启**：启用 `ShapeConstraintSolver`，可证明结构等价的动态 shape
  表达式（如两个都计算 `minsi(validRow, 32)` 的 SSA 值）属于同一迭代域，
  使计算型动态 shape 的 op 也能融合。
- **`--enable-shape-inference=false` 关闭**：退回静态/直接绑定推导，只有
  valid-shape 为编译期常量且相等的 tile 才被证明同域；动态 shape
  （`valid=?x?`）的 op 各自成独立域，不被融合。行为与引入
  `ShapeConstraintSolver` 之前一致。

`--enable-shape-inference` 只作用于 `FusionPlan`：这是唯一读取 iteration
domain classes、做出融合决策的 pass。`PTOFusionRegionGen` 只消费 `FusionPlan`
产出的 `pto.fusion.group_id`/`pto.fusion.order` 注解并封装成 `pto.fusion_region`。

`OpScheduling` 是 `FusionPlan` metadata 与下游消费者之间的规范化边界：它在最终
物理顺序上把无法压紧的 planned group 降级为独立连续 span，清除 singleton
metadata，并以函数级唯一 ID 重新编号。`RegionGen` 仍保持严格校验；绕过
`OpScheduling` 直接向 `RegionGen` 提供断裂或不完整 metadata 属于非法
pipeline/input，不在 `RegionGen` 中静默修复。

全函数重新编号（而非只给被拆分的段分配新 ID）是有意为之：所有 surviving span
统一重新编号，保证跨 block 与嵌套 region 的 ID 唯一性与确定性，并避免旧 ID
碰撞或 max-old-ID 记账。

#### 4.2.2 输入层级支持

| Level | 状态 | 说明 |
|---|---|---|
| `level2` | 支持 | 融合适配器在 `PlanMemory` 之前运行。 |
| `level3` | 支持 | 融合适配器直接工作在 manual-address tile-native IR 上。 |
| `level1` | N/A | 当前迁移范围内无可支持的输入面。 |

#### 4.2.3 主线 Pass 序列

1. **共享 tile-native 预处理**：
   ```
   PTOAssignDefaultFrontendPipeId → PTOLowerFrontendPipeOps →
   PTOInferValidatePipeInit → LoweringSyncToPipe → InferPTOLayout →
   PTOA5NormalizeTMov
   ```

2. **融合核心**（在 `PlanMemory` 决策点之前插入，当 A5 VPTO 融合条件满足时）：
   ```
   FusionPlan → OpScheduling → PTOFusionRegionGen
   ```

3. **Level 特定适配器**：
   - `level2`：`共享融合核心 → PlanMemory → PTOResolveReservedBuffers → (可选) PTOInsertSync`
   - `level3`：`共享融合核心 → 跳过 PlanMemory → PTOResolveReservedBuffers`
     （若在 `level3` 下显式开启 `--enable-insert-sync`，会打印 warning 并忽略。）

4. **ExpandTileOp 分界**（过渡到 VPTO 后端 Lowering）：
   ```
   ExpandTileOp → PTOInlineLibCall → FoldTileBufIntrinsics → SCCP → Canonicalizer
   ```

5. **Post-lowering 融合生命周期**（仅 fused A5 VPTO 路径）：
   ```
   PTOLowLevelLoopFusion → Canonicalizer → CSE →
   PTOFusionPredicateElision → PTOFusionLoadStoreElision →
   PTOFlattenFusionRegion → CSE
   ```

6. **VPTO 发射准备**：
   ```
   Canonicalizer/CSE → VPTOPtrNormalize → VPTOPtrCastCleanup →
   ReconcileUnrealizedCasts → PTOVPTOExpandBridgeOps →
   PTOInferVPTOVecScope → Canonicalizer → CSE →
   PTOValidateVPTOEmissionIR
   ```

#### 4.2.4 非目标路径

- 未显式传入 `--enable-op-fusion=true` 时，EmitC 不生成 `pto.last_use`，VPTO 不形成 `pto.fusion_region`，两条路径均不进入 TileOp fusion 生命周期。
- 后端分界线已固定为 `ExpandTileOp`；原有的 `View2Memref` / `PTOToA5VM` 主线已移除。

---

## 5. Pass 详细设计

### 5.0 融合转换流水线总览

```text
    [tile-native PTO IR]
               │
               ▼
    5.1  PreFusionAnalysis（纯分析）
               │
               ▼
    5.2  迭代域证明（内嵌于预分析；无独立 ShapeInferencePass）
               │
               ▼
    5.3  FusionPlan
               │
               ▼
    5.4  OpScheduling（组内物理聚拢）
               │
               ▼
    5.5  PTOFusionRegionGen（区域封装）
               │
               ▼
    5.6  Level 特定适配器（ExpandTileOp 之前）
         level2: PlanMemory → ResolveReservedBuffers → 可选 InsertSync
         level3: 跳过 PlanMemory → ResolveReservedBuffers
               │
               ▼
    5.7  VPTO 后端分界
         (ExpandTileOp → InlineLibCall → FoldTileBufIntrinsics →
          SCCP → Canonicalizer)
               │
               ▼
    5.8  PTOLowLevelLoopFusion
               │
               ▼
    5.9  后融合规范化与谓词消除
         (Canonicalizer → CSE → PTOFusionPredicateElision)
               │
               ▼
    5.10 PTOFusionLoadStoreElision（区域内访存消除）
               │
               ▼
    5.11 PTOFlattenFusionRegion → CSE（区域展平）
               │
               ▼
    5.12 VPTO 发射准备
         (ptr 归一化 / bridge 展开 / vecscope 推断 / 发射 IR 校验)
               │
               ▼
    5.13 VPTO 后端发射
         (VPTO 文本 / LLVM 发射)
```

### 5.1 PreFusionAnalysis（纯分析）

**设计动机。** 为 `FusionPlan` 提供可复用的 block-local 分析结果，不修改 IR。将哪些 op 能参与 tile fusion、哪些值会跨越 local/hard boundary 逃逸、哪些 op 处于同一迭代域等所有决策前置到统一分析中，供规划、封装及后续 region-local 清理复用。

**流水线位置。**
- **前置条件**：IR 仍在 tile-native `tile_buf` 语义世界，尚未跨越 `ExpandTileOp` 分界。
- **后置条件**：生成 `PreFusionAnalysis` 结果对象（仅含共享 dataflow graph，不含 iteration-domain classes）。此 pass 不在默认主线中单独插入；`FusionPlanPass` 与 `FusionRegionGenPass` 均通过 `getAnalysis<pto::PreFusionAnalysis>()` 复用同一份缓存的 DFG。迭代域推理被拆为可分离的 `inferIterationDomainClasses()` 步骤：因 `--enable-shape-inference` 依赖 per-pass 开关，无法进入 analysis manager 缓存（其只能以 `PreFusionAnalysis(getOperation())` 这个无开关固定签名构造），故由 `FusionPlanPass` 取缓存 DFG 后在本地副本上按自身开关运行；`FusionRegionGenPass` 不消费 domain classes，直接复用缓存 DFG。中间的 `OpSchedulingPass` 仅在 block 内重排（不跨 block），通过 `markAnalysesPreserved<PreFusionAnalysis>()` 声明 DFG 不失效，使下游 `FusionRegionGenPass` 命中缓存而非重建。

**输入/输出规格。**
- **输入**：`func::FuncOp` 内的 PTO tile-level IR。
- **输出**：每个 basic block 的 `FusionBlockAnalysis`，主要包含：
  - `computeNodes`：可参与预融合建模的 compute node。
  - `edges`：生产者/消费者依赖边。
  - `valueLiveness` / `writeInstances`：值和写实例的生命周期类别与逃逸类别。
  - `iterationDomainClasses`：按 `(v_row, v_col)` 证明结果划分的迭代域等价类。

**核心逻辑与约束。**
- **Op 语义分类**：`FusionOpSemantics` 将 op 分为 `Compute`、`LocalBoundary`、`HardBoundary`。
  - `treshape` 被视为 `LocalBoundary`，会阻断穿越它的局部规划与调度。
  - 无 `OpLibOpInterface`、带 region/call/未知副作用的 op，一律按 `HardBoundary` 处理。
- **当前可识别的 compute family**：
  - `Elementwise`：`tadd/tsub/tmul/tdiv/tmax/tmin` 及对应 scalar 版本、`texp`
  - `ScalarExpand`：`texpands`
  - `RowBroadcastBinary`：`trowexpandmul`、`trowexpanddiv`
  - `ReduceRow/ReduceCol`：`trowsum/trowmax/trowmin`、`tcolsum/tcolmax/tcolmin`
- **依赖与生命周期建模**：
  - 通过 SSA use-def 链和 DPS 输出归一化收集 tile 输入/输出。
  - 记录 block 内 consumer、local boundary user、hard boundary user、块外逃逸信息。
  - 写实例进一步区分为 `Internal`、`LocalBoundaryExternal`、`HardExternal`，供 region output/frontier 计算使用。
- **分析范围**：严格限制在单个 basic block 内，不做跨 block、跨 CFG 的全局规划。

**不变性与副作用。**
- 纯分析，不修改 IR。
- 可通过 `pto-pre-fusion-analysis` / `pto-print-pre-fusion-analysis` 做调试或 lit 检查，但它们不是默认后端主线的一部分。

### 5.2 迭代域证明

**设计动机。** 当前代码中形状推导尚未落成独立 pass。真正被主线依赖的是 `FusionAnalysis.cpp` 中对迭代域的保守证明：只有当参与计算的 anchor tile 能在编译期证明为一致的 rank-2 有效形状时，规划阶段才会继续融合。

**流水线位置。**
- **前置条件**：依赖 §5.1 中的 op 语义分类和 tile 类型/`pto.bind_tile` 元数据。
- **后置条件**：不修改 IR。分析结果中的 `IterationDomainInfo` 会被标注为 `Proven` 或 `Unproven` 并附带失败原因。

**输入/输出规格。**
- **输入**：compute op 的 tile 输入/输出及其有效形状信息。
- **输出**：`IterationDomainInfo { vRow, vCol, proof, unprovenReason }`。

**核心逻辑与约束。**
- **当前证明来源**：
  - 优先读取 `TileBufType::validShape`。
  - 若值来自 `pto.bind_tile`，则用其常量 `validRow/validCol` 覆盖类型上的静态形状。
  - `Elementwise` 聚合输入和输出；`ScalarExpand/RowBroadcastBinary` 用输出域；`ReduceRow/ReduceCol` 用输入域。
- **失败情形**：
  - 任一关键维度为动态值。
  - 同一组 anchor 的 `(v_row, v_col)` 不一致。
  - 缺少可恢复的 tile domain 信息。
- **现实边界**：实现中明确不尝试证明动态符号等价。带动态 `v_row/v_col` 的链路保持 `Unproven`，并在 §5.3 的规划阶段被保守拒绝。

**不变性与副作用。**
- 不生成新 attribute，不重写类型，不引入新的 shape solver。

### 5.3 FusionPlanPass

**设计动机。** 消费 §5.1 和 §5.2 的分析结果，为 block 内 op 打上稳定的组元数据，形成后续调度和区域封装的唯一输入契约。当前实现优先保证保守可用，而非开放式策略插件框架。

**流水线位置。**
- **前置条件**：`PreFusionAnalysis` 有效，op 仍处于 tile-level PTO IR。
- **后置条件**：被接受的组成员获得：
  - `pto.fusion.group_id`
  - `pto.fusion.order`

**输入/输出规格。**
- **输入**：`FusionBlockAnalysis`。
- **输出**：带规划 metadata 的原始 PTO IR。仅组大小 ≥ 2 的 group 才会落盘。

**核心逻辑与约束。**
- **当前策略**：
  - `ConservativeDAGGreedyStrategyEngine`
  - `ConservativeDAGGreedyCostModel`
- **可规划 op 子集**（比 §5.1 中可分析的 compute family 更窄）：
  - `tadd/tsub/tmul/tdiv/tmax/tmin`
  - `tadds/tsubs/tmuls/tdivs/tmaxs/tmins`
  - `texp`
  - `texpands`
  - `trowexpandmul`
  - `trowexpanddiv`
- **Seed 条件**：
  - op 必须属于上述可规划集合。
  - 其迭代域类必须是 `Proven`。
- **Append 条件**：
  - candidate 与当前组首成员属于同一 `iterationDomainClass`。
  - candidate 与当前组至少存在一条直接数据流连接。
  - 成本模型评分：`dependencyBenefit + loopMergeBenefit - liveTilePenalty - vfParameterPenalty > 0`。
- **当前成本模型参数**：
  - `dependencyBenefit = 4 × connectionCount`
  - `loopMergeBenefit = 4`
  - 当 `liveTileCount > 10` 时开始罚分
  - 当 `vfParameterCount > 12` 时开始罚分
- **结果排序**：
  - 组内顺序按 `blockOrder/id` 稳定排序。
  - `group_id` 按组首成员的 block 顺序稳定分配。

**不变性与副作用。**
- 只打 metadata，不移动 op。
- 当前未将策略接口暴露为可插拔配置点。ML/AI 决策接口仍属于未来方向。

### 5.4 OpSchedulingPass

**设计动机。** 将 §5.3 已规划好的 group 压缩成 block 内连续 span，为 `PTOFusionRegionGen` 提供"一组对应一个连续区间"的结构前提。

**流水线位置。**
- **前置条件**：op 已带有完整的 `pto.fusion.group_id/order`。
- **后置条件**：每个 group 在 block 中形成一个连续 span，组成员关系不变。

**输入/输出规格。**
- **输入**：带规划 metadata 的 PTO IR。
- **输出**：物理顺序重排后的 PTO IR。

**核心逻辑与约束。**
- **Barrier 分类**：
  - `Movable`：普通可移动 compute op。
  - `LocalBoundary`：例如 `treshape`，在无 tile 依赖冲突时可跨越。
  - `HardBoundary`：call、region op、未知副作用 op、不可安全移动的边界。
- **调度策略**：
  - 按 `group_id` 收集成员，再按 `pto.fusion.order` 排序。
  - 对组内后续成员，优先尝试前移到当前 `placement` 之后。
  - 若前移受阻，则在不违反 later-crossing 约束时反向尝试将 `placement` 后推。
- **合法性检查**：
  - 不能跨越 operand 的定义点。
  - 不能将 producer 移到某个 consumer 之后。
  - 不能越过 hard boundary。
  - 穿越 local boundary 时，需确认双方不共享 tile input/output 依赖。

**不变性与副作用。**
- 会改写 block 内 op 顺序。
- 不改 group metadata，不改 CFG。

### 5.5 PTOFusionRegionGenPass

**设计动机。** 将连续 span 封装成显式 `pto.fusion_region`，为 VPTO post-lowering 的 region-local 循环融合、谓词清理和 load/store 消除提供容器边界。

**流水线位置。**
- **前置条件**：同一 `group_id` 在 block 中已是单一连续 span。
- **后置条件**：每个 span 被包成一个 `pto.fusion_region`，并用 `pto.yield` 显式声明对外可见的 frontier。

**输入/输出规格。**
- **输入**：已调度好的 block-local group span。
- **输出**：`pto.fusion_region` 包装后的 PTO IR。
  - region 不显式建模输入 block argument。
  - region body 直接隐式捕获父作用域 SSA 值。
  - `pto.yield` 只返回真正对 region 外可见的 value。

**核心逻辑与约束。**
- **Span 识别约束**：
  - 同一 `group_id` 在一个 block 里必须只出现一个连续 span，否则直接报错。
  - 组内 `pto.fusion.order` 必须严格递增。
- **Frontier 计算**：
  - `PTOFusionRegionGen` 结合 use-def 分析和 `PreFusionAnalysis` 的写实例逃逸信息，找出必须在 region result 列表中保留的 escaping value。
  - 仅 region 内使用、或虽有外用但不可被 region result 合法替换的值，不会盲目外提。
- **结构约束**：
  - 一组对应一个 region。
  - 不额外生成 region 输入 operands。
  - 允许空 result / 空 `pto.yield`。

**不变性与副作用。**
- 会引入嵌套 region，显著改变 IR 结构。
- 这是当前主线中真正将"逻辑 fusion group"转成"后续 backend 可识别容器"的分界点。

### 5.6 Level 特定适配器（`ExpandTileOp` 之前）

**设计动机。** 不再为每个 level 维护独立融合实现，而是将差异收敛为适配器：共享融合核心固定在 `ExpandTileOp` 之前，`level2`/`level3` 仅在其前后的约束不同。

**流水线位置。**
- **前置条件**：`pto.fusion_region` 已建立，body 仍是 PTO tile op。
- **后置条件**：
  - `level2`：完成 `PlanMemory → PTOResolveReservedBuffers → 可选 PTOInsertSync`，同时保持 `pto.fusion_region` 不被破坏。
  - `level3`：跳过 `PlanMemory`，仅做 `PTOResolveReservedBuffers`，保持 explicit-address / manual-sync 契约。

**核心逻辑与约束。**
- 共享适配器插入点固定在 `PlanMemory` 决策点之前。
- `level2` 契约：
  - 允许 `PlanMemory` 将 region 内外 `alloc_tile` 改写成 `pto.pointer_cast`。
  - 允许 `PTOInsertSync` 在 region 之后追加 tail barrier。
- `level3` 契约：
  - 不会重新进入 `PlanMemory`。
  - 若显式开启 `--enable-insert-sync`，会打印 warning 并忽略，以保留 manual sync contract。
- `level1`：
  - 架构上应同样遵循"共享融合核心在 PlanMemory 之前"。
  - 当前分支无可用输入面，实现和测试均保持 N/A。

### 5.7 VPTO 后端分界

**设计动机。** `ExpandTileOp` 是当前 PTO tile IR 到 VPTO authoring IR 的硬边界。Tile fusion 必须在此之前完成 group 和 region 建模，同时确保 `pto.fusion_region` 能跨过这个 seam 供后续 post-lowering cleanup 使用。

**流水线位置。**
- **前置条件**：输入为 tile-native PTO IR；`pto.fusion_region` 可能已存在。
- **后置条件**：
  - Tile-level PTO op 被替换为 TileLang helper `call`/`func.call`。
  - Helper body 被 inline 到主函数。
  - `pto.tile_buf_addr` / `pto.tile_valid_rows` / `pto.tile_valid_cols` 等 intrinsic 被折叠为 VPTO 侧的 `memref`/`ptr`/常量。
  - Fused 路径上 `pto.fusion_region` 继续保留。

**核心逻辑与约束。**
- 固定顺序：
  1. `ExpandTileOp`
  2. `PTOInlineLibCall`
  3. `FoldTileBufIntrinsics`
  4. `SCCP`
  5. `Canonicalizer`
- 现实边界：
  - 原有的 `PTOViewToMemref` bridge 已移除，不再是有效 seam 前置条件。
  - `FoldTileBufIntrinsics` 依赖 native tile metadata；非 native producer 的输入（如缺少地址/valid-shape 元数据的 block argument）会导致下游保守失败。

### 5.8 PTOLowLevelLoopFusion

**设计动机。** 真正的"循环融合"现在发生在 `ExpandTileOp` 之后，直接在 `pto.fusion_region` 内处理当前 VPTO post-lowering 的 `scf.for + memref/pto.v*` 结构。

**流水线位置。**
- **前置条件**：输入契约是保留在 `pto.fusion_region` 内的 VPTO post-lowering loop nest。
- **后置条件**：可融合的相邻 loop stage 被聚合成共享 loop-header 的单一 carrier loop。

**核心逻辑与约束。**
- **Stage 识别方式**：
  - 每个 stage 由 setup op、一层或多层同构 `scf.for`、以及叶子 `pto.vlds/vsts/vadd/...` 或相关纯 op 组成。
  - 仅尝试融合彼此相邻、loop header 等价、且中间 prelude/setup 可安全重排的 stage。
- **合法性条件**：
  - `sameForHeader(lhs, rhs)`：下界、上界、步长和 loop attrs 必须完全等价。
  - 本 pass 以已完成 TileOp fusion 规划的 VPTO post-lowering loop nest 为输入。叶子体接受无区域且无内存副作用的 op、`pto.vlds/vsts`，以及经过跨 stage 依赖检查的标量 `memref.load/store`；未知叶子副作用阻止融合。
  - 含标量访问的融合须进一步确认各层循环上下界及步长为同一循环外 SSA 值或同值常量，避免仅凭动态表达式的结构判等。
  - 标量访问对至少一方写内存时，须证明存储不别名，或双方均为叶子访问同一循环外定义的 identity-layout memref、各维下标具有一致的 `iv + 常量` 映射且覆盖所有循环层级；只读访问对无需此限制。
  - 标量访问与其他内存操作的重排也须证明无冲突，检查覆盖 leaf、prelude 和 epilogue。Prelude/epilogue 还须通过下述既有重排检查。
  - Prelude/setup 必须是 side-effect-free 或可分析的内存 prelude。
  - 跨 stage 移动 prelude 时，不能与前一 stage 的内存根产生潜在 alias 冲突。
- **现实边界**：
  - 这是一个非常保守的 matcher，不做激进 loop normalization。
  - 只融合"相邻 stage"，不做跨区域或跨复杂控制流的全局循环拼接。

### 5.9 后融合规范化与谓词消除

**设计动机。** 低层循环融合完成后，主线先做常规 `Canonicalizer + CSE`，再专门清理 fusion-region 内部冗余的 VPTO 谓词物化，减少后续 load/store 消除看到的噪声。

**流水线位置。**
- **前置条件**：`PTOLowLevelLoopFusion` 已完成。
- **后置条件**：重复的 `pto.plt_*` 计算被压缩，后续访存消除看到的 VPTO loop body 更干净。

**核心逻辑与约束。**
- 历史上专门的 `PTOVPTOIfCanonicalize` / `A5VMIfCanonicalize` 已退役，仅使用共享 `Canonicalizer`。
- `PTOFusionPredicateElision` 当前聚焦 `pto::PltB8/B16/B32Op`。
- 在 fusion-region 内做保守 value 等价判断，包括：
  - 纯 op 结果等价。
  - Loop-carried `iter_arg` 与 `plt.scalar_out` 的直接递归等价。
- 仅在能证明等价时复用已有谓词，避免错误跨越 side effect 或复杂循环递归。

### 5.10 PTOFusionLoadStoreElision

**设计动机。** 在已形成稳定 VPTO carrier loop 的前提下，消除 fusion-region 内部仅用于中转的本地 store/load 往返，将 region-local 数据通路收缩到更接近寄存器/向量值直传的形态。

**流水线位置。**
- **前置条件**：低层循环融合、全局规范化和谓词消除已完成。
- **后置条件**：局部 `pto.vsts → pto.vlds` round-trip 被消除；非逃逸尾部 store 也可能被清理。

**核心逻辑与约束。**
- **输入契约**：处理对象是 `pto.fusion_region` 内的 VPTO post-lowering loop body，典型形态为 `scf.for + memref.subview + memref.cast + pto.vlds/vsts/vadd/...`。
- **核心行为**：
  - 归一化 tracked memref 根值，穿透 `bind_tile`、`memref.cast`、`reinterpret_cast`、`transpose` 等包装。
  - 以 `pto.yield` / region result 作为"外部可见性 frontier"，避免误删需要向 region 外暴露的写回。
  - 保守消除匹配的 store/load 对，并做 frontier-aware tail-store cleanup。
- **现实边界**：
  - 遇到无法做别名分析的内存 effect 会直接保守退出。
  - 仅处理 fusion-region 局部模式，不替代通用 DSE。

### 5.11 PTOFlattenFusionRegion

**设计动机。** 一旦 region-local 后端优化全部结束，`pto.fusion_region` 这个结构化容器就不再需要，必须显式展平回父 block，恢复后端发射更容易消费的平面 VPTO IR。

**流水线位置。**
- **前置条件**：§5.10 之后 region 内剩余 op 已是最终 backend-ready 形式。
- **后置条件**：`pto.fusion_region` 和 `pto.yield` 被删除，父 block 仅保留普通低层 VPTO op。

**核心逻辑与约束。**
- 将 region body 中除 terminator 外的 op 全部移到 wrapper 之前。
- 用 `pto.yield` 的 operands 替换 `pto.fusion_region` 的结果。
- 擦除 `pto.yield` 和 wrapper 本身。
- 展平后再跑一轮 `CSE`，去掉 wrapper 生命周期结束后产生的冗余值。

### 5.12 VPTO 发射准备

**设计动机。** Post-lowering 融合生命周期结束后，IR 还需经过一段与发射器强相关的 VPTO emission preparation，才能成为真正可导出的 VPTO 文本或 LLVM 产物。

**流水线位置。**
- **前置条件**：IR 已是展平后的 backend-ready VPTO 形式。
- **后置条件**：完成 ptr 归一化、bridge 展开、vecscope 推断和发射合法性校验。

**核心逻辑与约束。**
- 固定顺序：
  1. `Canonicalizer + CSE`
  2. `VPTOPtrNormalize`
  3. `VPTOPtrCastCleanup`
  4. `ReconcileUnrealizedCasts`
  5. `PTOVPTOExpandBridgeOps`
  6. `CSE`
  7. `PTOInferVPTOVecScope`
  8. `Canonicalizer + CSE`
  9. `PTOValidateVPTOEmissionIR`
- 此阶段已不属于 tile fusion 本身，但它决定了前面产出的 VPTO IR 能否成功发射。

### 5.13 VPTO 后端发射

**设计动机。** 当前 tile fusion 主线的最终目标是进入现有 VPTO 后端发射器：要么输出清理后的 VPTO 文本，要么继续走 LLVM emission 交给后续工具链。

**流水线位置。**
- **前置条件**：IR 已通过发射准备和合法性校验。
- **后置条件**：生成 VPTO 文本输出或 LLVM 级后端产物。

**核心逻辑与约束。**
- `--emit-vpto` 直接输出准备完成后的 VPTO IR。
- `--vpto-emit-hivm-llvm` / `--vpto-emit-hivm-bc` 继续走 LLVM/HIVM 导出。
- 从 tile fusion 的角度，这里关注的是前置 pass 是否产出了结构正确的 VPTO IR，而非后端发射器本身的实现细节。

---

## 6. 关键算法

- **预分析驱动的 block-local DFG 建模**：在 `tile_buf` 世界内提取 compute/local-boundary/hard-boundary 分类、value liveness 和 write escape class，作为所有后续决策的统一依据。
- **保守 DAG 贪心规划**：当前默认策略是 `ConservativeDAGGreedyStrategyEngine + ConservativeDAGGreedyCostModel`，非开放式插件系统。
- **Span 压缩调度**：通过 barrier 分类与双向移动规则，将离散 group 压成连续 span，同时维持 SSA 和 boundary 合法性。
- **Post-lowering stage matcher**：`PTOLowLevelLoopFusion` 在 `scf.for + memref/pto.v*` VPTO 层匹配同头循环和可重排 prelude，执行保守的相邻 stage 融合。
- **Frontier-aware cleanup**：`PTOFusionPredicateElision` 和 `PTOFusionLoadStoreElision` 都依赖 fusion-region frontier，避免把仍需对外可见的值错误消除。

---

## 7. 与系统其他模块的交互

- **共享 pre-backend normalization**：Tile fusion 前半段与 VPTO backend 共用 `LoweringSyncToPipe`、`InferPTOLayout`、`PlanMemory`、`PTOResolveReservedBuffers`、可选 `PTOInsertSync` 等 pass；其中 `level3` 会显式跳过 `PlanMemory`。
- **ExpandTileOp seam 契约**：`ExpandTileOp` 是当前 PTO → VPTO 的硬边界。Tile fusion 必须在 seam 之前形成 `pto.fusion_region`，并保证 wrapper 能跨过 `InlineLibCall` / `FoldTileBufIntrinsics`。
- **Backend emitter 契约**：上游 pass 需将 IR 收敛到 `prepareVPTOForEmission()` 可接受的形态，否则后端发射会直接失败。
- **测试与 OpenSpec**：当前主要行为约束已转移到 `test/lit/tile_fusion/*.pto`、`test/basic/expand_tile_op_tilelang_tadds.pto`、`test/vpto/auto_vecscope_infer_boundary.pto` 以及 `openspec/changes/reintroduce-vpto-tile-fusion/*`。原有的 LibCall/CCE 设计不再是 source of truth。

---

## 8. 验证与测试

### 8.1 功能验证

- 当前功能回归以 `test/lit/tile_fusion/*.pto` 为主。原有的 `test/tile_fusion/*.mlir` 和 test-only CLI 路径已废弃。
- 查看 frontend group/schedule/region 结果时，使用：
  - `--mlir-print-ir-after=pto-fusion-plan`
  - `--mlir-print-ir-after=pto-op-scheduling`
  - `--mlir-print-ir-after=pto-fusion-region-gen`
- 验证 backend 主线 IR 形态时，使用：
  - `--mlir-print-ir-after=pto-expand-tile-op`
  - `--mlir-print-ir-after=pto-low-level-loop-fusion`
  - `--mlir-print-ir-after=pto-fusion-predicate-elision`
  - `--mlir-print-ir-after=pto-fusion-load-store-elision`
  - `--mlir-print-ir-after=pto-flatten-fusion-region`
- 负例应覆盖：动态 shape、local boundary、hard boundary、不可移动副作用 op。

### 8.2 性能度量

- **结构指标**：
  - `pto.fusion.group_id/order` 是否正确。
  - `pto.fusion_region` 是否仅包住一个连续 span。
  - `pto.fusion_region` 是否能够跨过 `PlanMemory`、`ResolveReservedBuffers`、可选 `PTOInsertSync` 和 `ExpandTileOp` seam。
- **低层指标**：
  - 冗余 `pto.plt_*` 是否下降。
  - Region 内 `vlds/vsts` round-trip 是否减少。
- **端到端测试命令**：
  - `llvm-lit -sv test/lit/tile_fusion`
  - 代表性 fused A5 VPTO 样例：`test/lit/tile_fusion/op_fusion_adapter_placement_level2_tadd.pto` 和 `test/lit/tile_fusion/op_fusion_adapter_placement_level3_tadd.pto`
  - 必要时用 `--mlir-print-ir-after=<pass>` 对照关键阶段 IR。

---

## 9. 当前边界与后续方向

### 9.1 当前边界

- 主线仅覆盖 A5 VPTO backend，不覆盖 EmitC。
- 规划仅支持保守的 block-local group，不做跨 basic block 融合。
- 动态迭代域当前无法证明，相关链路会被拒绝。
- `PreFusionAnalysis` 可识别的 compute family 比 `FusionPlan` 当前允许规划的 op 范围更宽。
- `level1` 在当前迁移范围内无可支持输入面，仅保留设计位置，无实现与回归覆盖。

### 9.2 后续可扩展方向

- 将动态 `v_row/v_col` 证明补成独立 shape inference 主线 pass。
- 放宽 planner 对 reduce/broadcast 组合的实际落地支持。
- 在保持 `pto.fusion_region` 契约稳定的前提下，继续增强 post-lowering loop fusion 和 load/store elimination 的覆盖面。
