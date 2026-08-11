# PTOAS Largest-First-Fit 与四道冲突闸门内存规划设计

## 总体方案

PTOAS 当前保留两套 local memory planner：

- legacy memplan：默认启用，保留旧版 SPEC_LEVEL_0/1/2 三级投机复用，内存不够时回滚的策略。
- modern memplan：通过 `--plan-memory-impl=modern` 显式启用，当前实现Largest-First-Fit 与四道冲突闸门内存规划设计

PTOAS 当前已经具备 largest-first 风格的规划开关：`--plan-memory-order-by-size`。该选项会让 planner 在同一 AddressSpace 内优先处理更大的 buffer。当用户显式选择 `--plan-memory-impl=modern` 且未显式指定 `--plan-memory-order-by-size` 时，modern memplan 默认开启该排序；legacy memplan 仍保持默认关闭。

当前 `pto.alloc_tile` 的 lowering 路径保持 tile-native：

```text
pto.alloc_tile(no addr)
  -> PTOViewToMemref 透传，不转换成 memref.alloc
  -> pto-plan-memory 收集为 local allocation root
  -> modern/legacy memplan 按 level 校验并规划 local addr
  -> 直接给 pto.alloc_tile 补常量 addr
  -> 后续 pto.t* tile op 继续使用 !pto.tile_buf 形态
```

用户显式写 `pto.alloc_tile addr` 时，level 语义仍由 memplan 校验：level1/level2 禁止显式 addr，level3 要求显式 local addr。也就是说，`pto.alloc_tile` 不再通过 `memref.alloc -> pto.bind_tile` 这类中间链路表达地址；普通 local tile allocation 由 memplan 直接把常量地址回写到 `pto.alloc_tile addr`。历史 `pto.pointer_cast` bridge 已删除，不再作为 memplan materialize 结果。

## 目标

本设计目标如下：

- 复用 PTOAS 已有 `--plan-memory-order-by-size` largest-first 能力。
- 将复用判定收敛为统一的 `canShare(a, b)` 谓词。
- 引入统一的扁平冲突闸门模型，避免不安全复用。
- 当前实现范围不包含 PyPTO 闸门 4；PTOAS 显式 multibuffer 已覆盖 ping-pong slot 分离语义。
- 保持按 AddressSpace 独立规划。
- 保持不回滚、不降级的 deterministic 策略。
- 保持 `pto.alloc_tile(no addr)` tile-native 路径：memplan 直接补 `addr`。
- 保持 legacy memplan 默认行为不变。

## 非目标

本设计不包含以下内容：

- 不修改 legacy memplan 的 StorageEntry / SPEC_LEVEL_1 / SPEC_LEVEL_2 逻辑。
- 不在 legacy memplan 中实现四道闸门。
- 不恢复旧版回滚式投机规划。
- 不把 modern memplan 作为默认实现。
- 不在本阶段实现跨函数、跨 module 的全局内存规划。

## 核心思想

将每个 plannable local allocation root 抽象成一个待装箱 item。这里的 item 对应当前 modern memplan 中的 `RootInfo`，不是所有 local SSA value：

`pto.bind_tile`、`pto.slot_marker`、`memref.subview`、`memref.cast` 等 alias/view value 不会成为新的 item，而是通过 `valueToRoots` 归属到已有 root。四道闸门需要的额外约束数据不放进 `RootInfo`，而是由 `ConflictFacts` 这类 side table 提供。


## 算法流程

### 1. Root 收集

modern memplan 继续收集以下 local root：

- `pto.alloc_tile(no addr)`。
- plannable local `memref.alloc`。
- 后续如有其它明确的 local address root，可扩展到同一 root 表。

不参与 local memplan 的 root：

- GM space。
- `AddressSpace::Zero`。
- 已经由 level3 显式指定地址的 local `pto.alloc_tile addr`。
- 非静态 shape 或无法计算 element byte size 的 root。

### 2. LifetimeInterval

每个 root 生成一个 `LifetimeInterval`：

```text
  Value root;
  Operation *defOp;
  AddressSpace space;
  uint64_t slotBytes;
  uint64_t totalBytes;
  uint64_t alignmentBytes;
  uint64_t slotCount;
  unsigned allocIndex;
  unsigned freeIndex;
  unsigned stableOrder;
  SmallVector<uint64_t> offsets;
```

现代 planner 中已有线性化 walk 和 root alias 传播，本设计在此基础上补齐：

- loop-aware lifetime extension。
- branch / yield / iter_arg alias family 信息。
- per-root semantic metadata。

### 3. 按 AddressSpace 分桶

复用只允许发生在同一 AddressSpace 内：

```text
Vec 只和 Vec 复用
Mat 只和 Mat 复用
Left/Right/Acc/Bias/Scaling 各自独立
GM 不参与 local memplan
```

分桶是硬约束，在进入装箱前完成。

### 4. Largest-First-Fit 排序与装箱

PTOAS 已有 `--plan-memory-order-by-size` 对应的 largest-first 排序能力。打开该选项后，每个 AddressSpace 桶内排序：

```text
sizeBytes 降序
sizeBytes 相同则按 defIndex 升序
defIndex 相同则按 stableOrder 升序
```

这样先让大 buffer 更早参与规划，再把生命周期不冲突的小 buffer 放入可复用位置，避免 definition-order greedy 只能“小复用先出现的大”的单向问题。

将一个物理 local buffer 抽象成一个 bin：

```text
ReuseGroup
  representative root
  members
  slot size bytes
  address space
  chosen offset
```

因此，本设计不是重新新增一个独立的 largest-first 开关，而是要求四道闸门的 `canShare` 判定接入现有 `--plan-memory-order-by-size` 路径。

特点：

- 不回滚。
- 不降级。
- 装进第一个可容纳 group 后立即停止。
- 后续容量超限直接报错，不为了适配容量放松闸门。

## 四道禁止内存复用的闸门

四道闸门是扁平 AND 关系，任意一道闸门失败，两个 root 就不能进入同一个 `ReuseGroup`，也就不能复用同一个 base offset。

### 闸门 1：生命周期

**目的：** 禁止两个运行时可能同时存活的 local root 复用同一块物理地址。

基础规则：

```text
如果 a 和 b 生命周期重叠，则不能复用。
如果 a.lastUse == b.def 或 b.lastUse == a.def，则认为 touching，可继续检查其它闸门。
```

PTOAS 现有 `lifetimesOverlap` 可扩展为：

```text
overlap = !(a.lastUse <= b.def || b.lastUse <= a.def)
```

`<=` 表示 touching 不算生命周期冲突。

**适用场景 sample：普通直线代码中的临时 buffer 复用。**

```text
%a = alloc vec[1024]
use(%a)              // %a.lastUse

%b = alloc vec[1024] // %b.def
use(%b)
```

如果 `%a.lastUse <= %b.def`，闸门 1 允许 `%a` 和 `%b` 进入后续闸门。只要其它闸门也通过，二者可以复用同一个 offset：

```text
%a.offset = 0
%b.offset = 0
```

反例：

```text
%a = alloc vec[1024]
%b = alloc vec[1024]
use(%a)
use(%b)
```

此时 `%a` 和 `%b` 的生命周期重叠，闸门 1 直接失败，必须分配不同 offset。

### 闸门 2：phi family

**目的：** 处理互斥控制流分支，避免过度保守的 liveness extension 阻止本来安全的复用。

场景：

```text
%r = scf.if %cond -> tile {
  scf.yield %a
} else {
  scf.yield %b
}
```

`%a` 和 `%b` 在程序序生命周期上可能被扩展到 `%r` 的最后使用点，看起来重叠；但运行时两个分支互斥，不会同时存在。因此同一 phi family 的 yield source 可以豁免闸门 1 的生命周期冲突。

需要收集：

```text
phiFamilyIds[root] = set<familyId>
```

规则：

- 同 family 且无其它非 family 真重叠成员：允许共享。
- 如果混入外部 live root 或非 family 真重叠 root：不豁免。

实现建议：

```text
Gate2_LifetimeAndPhi(a, b):
  if !lifetimeOverlap(a, b):
    return true

  if samePhiFamily(a, b):
    return true

  return false
```

因此闸门 2 在实现上可以嵌入闸门 1，也可以保留独立函数名但由闸门 1 调用。

**适用场景 sample：if/else 分支内 local root 互斥复用。**

```text
%r = scf.if %cond -> tile {
  %then_buf = alloc vec[1024]
  produce(%then_buf)
  scf.yield %then_buf
} else {
  %else_buf = alloc vec[1024]
  produce(%else_buf)
  scf.yield %else_buf
}

consume(%r)
```

如果简单地把 `%then_buf` 和 `%else_buf` 的生命周期都延伸到 `consume(%r)`，二者看起来重叠，闸门 1 会失败。但运行时 then/else 互斥，不会同时分配这两个 branch-local root。因此它们属于同一个 phi family 时，可以共享同一 offset：

```text
%then_buf.offset = 0
%else_buf.offset = 0
```

反例：

```text
%outer = alloc vec[1024]

%r = scf.if %cond -> tile {
  %then_buf = alloc vec[1024]
  use(%outer)
  scf.yield %then_buf
} else {
  %else_buf = alloc vec[1024]
  use(%outer)
  scf.yield %else_buf
}
```

`%then_buf` 和 `%else_buf` 之间可以因为 phi family 互斥而复用；但它们不能借这个豁免和 `%outer` 这种外部 live root 错误复用。


### 闸门 3：target-specific load/tpop hazard

**目的：** 表达特定 target 上的局部硬件/后端 hazard。它不是通用生命周期问题，而是某些 load-derived buffer 与 consumes-tpop writer 在 touching 点复用会触发目标相关错误。

参考 PyPTO 的 Ascend910B split-AIV hazard。PTOAS 中设计为 target-gated 闸门：

```text
如果目标架构/后端不存在该 hazard，则闸门恒通过。
如果存在该 hazard，则禁止 load-derived buffer 与 consumes-split-tpop 的 writer 在 touching 点复用。
```

当前 PTOAS modern memplan 的启用条件：
A5 上该闸门恒通过。A3 上也只有识别到 split tpop 派生值时才可能触发。

#### 设计原理

这道闸门保护的是一种普通静态生命周期分析看不到的 target-specific touching 复用风险。普通 memplan 只看 allocation root 的 def/use 区间：如果一个 load-derived buffer 的最后一次使用正好等于另一个 writer output 的写入点，生命周期上属于 touching，通常可以复用同一个 offset。但在特定 target 上，`tload/tprefetch` 产生的 load-derived buffer 与 consuming split-tpop 的 writer 可能在后端流水或指令调度中存在更细粒度的运行时 overlap；如果二者复用同一地址，writer 的写入可能破坏 load-derived input 在该 target 上仍然需要保持稳定的内容。

因此该闸门不把问题抽象成通用“生命周期重叠”，而是显式识别一类 target hazard：

```text
load-derived input 的 last use
touching
consumes split-tpop 的 writer output 写入
```

只有当目标架构确认存在该 hazard 时才禁止复用。当前 PTOAS 中 A3 开启该闸门，A5 恒通过，避免把 A3 的局部硬件/后端约束错误扩散成所有 target 的通用内存规划规则。

实现上使用 writer op index，而不是 allocation root 的 def index，原因是 hazard 发生在“某个 writer op 同时读取 load-derived input 和 split-tpop-derived value，并写出 DPS output”的这个操作点。allocation root 的 def 只表示 buffer 被声明出来，不一定等于真正发生写入和触发 target hazard 的位置；用 writer op index 可以把 touching 条件精确绑定到产生风险的 op 上。

相关事实拆开收集：

```text
loadDerivedRoots:
  标记来自 tload/tprefetch 的 DPS dst root

split-tpop-derived values:
  标记 split tpop 结果、split tpop tile operand 以及它们经过 alias/view op 派生出的 value

tpopConsumerRoots:
  标记同时读取 load-derived root 和 split-tpop-derived value 的 DPS writer output root

tpopConsumerWriteIndices:
  记录上述 writer op 的 op index，用于判断是否正好在 touching 点复用
```

这种拆分可以避免过度保守：不是所有 load buffer 都禁止复用，也不是所有 tpop 相关 buffer 都禁止复用，只有“load-derived input 的最后使用点”碰到“consumes split-tpop writer 的写入点”时才失败。

当前 PTOAS modern memplan 对 DPS output root 使用 writer-def liveness：如果某个 DPS output 是 pure overwrite，即该 output 没有被 memory effects 标记为 Read，则它的 `allocIndex` 会从 `pto.alloc_tile` / `memref.alloc` 收缩到第一次有效 writer op。这样 load-derived input 的 `freeIndex` 与 writer output 的 `allocIndex` 可以形成 touching，是否允许复用继续由 target-specific load/tpop hazard 和 op semantic no-alias 闸门判断。若 output 需要旧值，例如 read-modify-write / accumulate 语义，则保持 allocation-start 的保守生命周期。

需要收集：

```text
loadDerivedRoots
tpopConsumerRoots
tpopConsumerWriteIndices
targetHazardEnabled
```

判定：

```text
input.root in loadDerivedRoots
&& writer.root in tpopConsumerRoots
&& input.lastUseIndex in tpopConsumerWriteIndices[writer.root]
```

双向检查：

```text
hazard(a, b) || hazard(b, a)
```

**适用场景 sample：target-gated touching 复用禁止。**

```text
%load_buf = alloc vec[1024]
load_or_tpop_producer(%load_buf)
last_use(%load_buf)

%writer_dst = alloc vec[1024]
writer_consumes_tpop(...) outs(%writer_dst)
```

从普通生命周期看，`%load_buf.lastUse == %writer_dst.def`，属于 touching，可以复用。但如果 target 标记了：

```text
%load_buf in loadDerivedRoots
%writer_dst in tpopConsumerRoots
tpopConsumerWriteIndices[%writer_dst] contains writer op index
targetHazardEnabled = true
```

则闸门 3 失败，二者不能复用。若目标没有该 hazard，闸门 3 恒通过，不影响复用。

当前实现的 fact 来源：

```text
loadDerivedRoots:
  pto.tload / pto.tprefetch 的 DPS dst root

split tpop derived value:
  A3 上 split != 0 的 pto.tpop_from_aic result
  A3 上 split != 0 的 pto.tpop tile operand
  以及从这些 value 经过 bind_tile / slot_marker / cast / subview / select 等 alias/view op 派生出的 value

tpopConsumerRoots:
  某个 DPS writer 同时读取 split tpop derived value 和 load-derived root 时，
  记录该 writer 的 DPS output root
```


### 闸门 4：op semantic no-alias

**目的：** 表达“从生命周期看可以复用，但从 op 语义看不能 alias”的约束。闸门 4 对应 PyPTO 的 inplace 机制：`not_inplace_safe()` 与 `forbid_output_alias(i)`，并覆盖 PTOAS legacy memplan 中已有的 scratch-output conflict。

在 modern memplan 中，闸门 4 不应再只叫笼统的 `semantic conflict`，而应建模为一张明确的 forbid-alias side table：

```text
forbidAlias[root] = forbiddenRootSet
```

这里的 `root` 是 plannable local allocation root，而不是任意 SSA value。收集时需要先通过 `valueToRoots` 把 operand、view、alias value 归约到 root，再记录 root 与 root 之间的禁止复用关系。

#### 4.1 scratch-output conflict

PTOAS legacy memplan 的 semantic conflict 主要就是 scratch-output conflict。modern memplan 应继续覆盖这类场景：

```text
op implements PTO_DpsInitOpInterface
dpsInits = op outs(...)

effects = MemoryEffectOpInterface::getEffects(op)
scratchOperands =
  Write operand
  && operand in op operands
  && operand not in dpsInits

for scratch in scratchOperands:
  for dst in dpsInits:
    forbidAlias[root(scratch)].insert(root(dst))
    forbidAlias[root(dst)].insert(root(scratch))
```

**适用场景 sample：`tmp` scratch 不能和 output 复用。**

```text
%src = alloc vec[1024]
%tmp = alloc vec[1024]
%dst = alloc vec[1024]

pto.ttrans ins(%src, %tmp) outs(%dst)
```

`%tmp` 是 op 执行过程中的 scratch workspace，`%dst` 是最终 output。即使未来 liveness 把 `%dst` 看作在 op 上定义、从而让 `%tmp.lastUse == %dst.def` 成为 touching，二者也不能复用，否则 scratch 写入可能覆盖 output。闸门 3 记录：

```text
forbidAlias[%tmp].insert(%dst)
forbidAlias[%dst].insert(%tmp)
```

#### 4.2 not_inplace_safe

PyPTO 中 `not_inplace_safe()` 表示该 op 不能做 `src == dst` 的 inplace 执行。映射到 PTOAS 时，规则是：

```text
if opPolicy.notInplaceSafe:
  for operand in op operands excluding dpsInits:
    for dst in dpsInits:
      forbidAlias[root(operand)].insert(root(dst))
      forbidAlias[root(dst)].insert(root(operand))
```

典型 op 包括：

```text
pto.ttrans
pto.tgather
pto.tands / pto.tors / pto.txors
pto.tfillpad // dst physical shape is larger than src
pto.tfmod / pto.tfmods
pto.trecip / pto.trsqrt
pto.trowmax / pto.trowmin / pto.trowsum / pto.trowprod
pto.trowargmax / pto.trowargmin
pto.tcolargmax / pto.tcolargmin
pto.tsort32 / pto.tmrgsort
```

其中 `pto.tands` / `pto.tors` / `pto.txors` 和推导为 expand lowering 的 `pto.tfillpad` 是 PTOAS 侧额外保守标记的 non-inplace-safe op。它们虽然不是 scratch-output conflict，但后端/ISA 语义没有明确承诺 input/output alias 安全，memplan 不应通过地址复用隐式把它们变成 inplace 执行。

**适用场景 sample：算法本身不支持 input/output alias。**

```text
%x = alloc vec[1024]
%y = alloc vec[1024]

pto.tfmod ins(%x, %rhs) outs(%y)
```

`tfmod` 的实现可能会在计算中间覆盖某个源值，但后续仍需要原始源值。因此 `%y` 不能复用 `%x` 的物理地址。即使生命周期或 touching 规则允许，也必须由闸门 3 禁止：

```text
forbidAlias[%x].insert(%y)
```

#### 4.3 forbid_output_alias(i)

PyPTO 中 `forbid_output_alias(i)` 表示 op 整体可以对某些 value operand 做 inplace，但 output 不能 alias 第 `i` 个特定 operand。PTOAS 中需要按 PTO IR 的 DPS operand 布局映射到具体 operand。

典型场景：

```text
pto.tsel:
  forbid mask
  forbid tmp

pto.trowexpand / pto.tcolexpand:
  forbid broadcast source

pto.trowexpand* / pto.tcolexpand*:
  forbid row/column vector operand
```

**适用场景 sample：broadcast vector 不能被 output 覆盖。**

```text
%row = alloc vec[16x1]
%dst = alloc vec[16x64]

pto.trowexpand ins(%row) outs(%dst)
```

`%row` 会被重复读取并广播到 `%dst` 的多个位置。如果 `%dst` 复用 `%row` 的地址，写 output 的过程中可能覆盖后续仍要读取的 broadcast source。因此闸门 3 记录：

```text
forbidAlias[%row].insert(%dst)
```

**适用场景 sample：select 的 mask/tmp 不能和 output alias。**

```text
%mask = alloc vec[1024]
%tmp  = alloc vec[1024]
%dst  = alloc vec[1024]

pto.tsel ins(%mask, %lhs, %rhs, %tmp) outs(%dst)
```

`%lhs` / `%rhs` 是否允许和 `%dst` inplace 取决于 op 语义；但 `%mask` 和 `%tmp` 是被 op 读取或作为 scratch 使用的特殊 operand，不能被 `%dst` 覆盖。闸门 3 记录：

```text
forbidAlias[%mask].insert(%dst)
forbidAlias[%tmp].insert(%dst)
```

#### 4.4 闸门 4 判定

`canShare(a, b)` 中的闸门 4 是双向检查：

```text
Gate4_OpSemanticNoAlias(a, b):
  return !forbidAlias[a].contains(b)
      && !forbidAlias[b].contains(a)
```

由于 PTOAS 中 `pto.bind_tile`、`pto.slot_marker`、`memref.subview` 等 view/alias value 不一定是 root，收集 forbid-alias 时必须先做 root 归约：

```text
for value in op operands / dpsInits:
  roots = valueToRoots[value]
```

如果一个 value 通过 alias closure 对应多个 root，则需要记录所有 root 组合。这样后续 `ReuseGroup` 只需要比较 root 与 root，不需要在装箱阶段重新理解每个 op 的 operand 语义。

## PIPE_V 物理共址性能代价模型

四道闸门只回答“两个 root 复用是否语义安全”。但对 A2/A3 这类 PIPE_V、MTE2、MTE3 可以重叠执行的后端，语义安全不等价于性能最优。若 modern memplan 为了最小化 local footprint，把本来分开的 scratch live range 压到同一小段物理 UB 地址，后续 InsertSync 会基于物理地址重叠补出更强的流水依赖；即使显式同步数量没有增加，MTE3 store 与下一段 MTE2 load、连续 PIPE_V producer/consumer 的可重叠窗口也会变窄。

因此，modern memplan 在通过四道安全闸门之后，还需要引入一个非硬约束的性能代价模型：

```text
canShare(a, b) == true
  只表示允许复用。

reuseCost(a, b, candidateOffset)
  表示复用该 offset 可能带来的流水串行化代价。
```

装箱策略不应只做“第一个可容纳地址即复用”，而应在容量允许时优先选择低代价地址：

```text
1. 先过滤所有违反四道闸门的 candidate。
2. 对剩余 candidate 计算 reuseCost。
3. 优先选择 cost 最小的 candidate。
4. 若存在 cost=0 的 fresh/near-fresh 地址，优先保留流水并行。
5. 只有容量压力确实需要压缩时，才接受高 cost 的复用。
```

该模型不是第五道正确性闸门：容量不足时仍允许选择高 cost 复用，只要四道闸门全部通过；但在容量充足时，它应避免把高频 loop 内的 scratch 过度挤压到同一物理地址。

### 1. 物理区间建模

判断“连续 PIPE_V op 是否共址”不能只看 SSA value 是否相同，而要看 memplan materialize 后的物理 local 区间：

```text
PhysicalInterval {
  AddressSpace space;
  uint64_t startByte;
  uint64_t endByte;
}
```

每个 root 的区间由规划结果得到：

```text
startByte = plannedOffset
endByte   = plannedOffset + slotBytes
```

alias/view value 需要先归约到 root，再把 subview/treshape/bitcast/multi_tile_get 等局部 offset 合入区间。对无法精确计算 byte range 的 alias，性能模型应保守地使用 root 的完整 slot range，不能为了少报冲突而截断未知范围。

当前 modern memplan 的首版实现运行在 materialize 之前，因此用“prospective `ReuseGroup` 共址”作为物理区间代理：如果一个 root 加入已有 group，就认为它和 group 内成员共享同一 local 区间；如果选择 fresh group，就认为它不和已有 group 产生共址代价。后续若 planner 支持更细粒度 subview byte range，可把该代理替换为精确 `PhysicalInterval`。

区间重叠判定：

```text
overlap(a, b):
  a.space == b.space
  && a.startByte < b.endByte
  && b.startByte < a.endByte
```

### 2. PIPE_V access 收集

对每个可能影响流水的 op，收集其 pipe 与内存访问集合：

```text
OpAccess {
  Operation *op;
  Pipe pipe;
  unsigned opIndex;
  SmallVector<PhysicalInterval> reads;
  SmallVector<PhysicalInterval> writes;
}
```

`reads` / `writes` 来自 `MemoryEffectsOpInterface`，并通过 `valueToRoots` 归约到 root。规则与 InsertSync 保持一致：

`opIndex` 使用 pipe/memory-access op 的序号，而不是完整 IR linear op 序号；这样中间夹着 `arith.constant`、`pto.alloc_tile addr`、`TASSIGN` materialization 或结构性 op 时，仍能识别真正相邻的 PIPE_V/MTE 访问。

- DPS input 是 read。
- DPS output 是 write；若 op 明确 read-modify-write，则同时是 read + write。
- scratch/tmp operand 若被 MemoryEffects 建模为 write，则进入 writes；若接口语义要求 scratch 读旧值，则进入 reads + writes。
- 控制流 result、`pto.fusion_region` result、`pto.subview`、`pto.treshape`、`pto.bitcast`、`pto.multi_tile_get` 等 alias value 必须穿透到 root。

只有 `pipe == PIPE_V` 的 op 参与 PIPE_V 共址代价；MTE2/MTE3 相关代价单独建模。

### 3. 连续 PIPE_V 共址判定

两个 PIPE_V op 若在局部 op 序上相邻或近邻，并且存在 write/read-write 区间重叠，就认为该 candidate 会压缩 PIPE_V 流水窗口：

```text
pipeVCoLocated(opA, opB):
  opA.pipe == PIPE_V
  opB.pipe == PIPE_V
  distance(opA, opB) <= pipeVLookahead
  && (
       overlap(any opA.writes, any opB.reads)
    || overlap(any opA.writes, any opB.writes)
    || overlap(any opA.reads,  any opB.writes)
  )
```

`pipeVLookahead` 初始可以取 1，仅覆盖真正连续的 PIPE_V op；若后续发现短距离 MTE/arith op 夹在中间也会造成同类串行化，可以扩展到 2 或 3。该值是性能启发式，不影响四道安全闸门的正确性。

### 4. MTE3 -> MTE2 共址代价

对 store-heavy kernel，还需要识别同一物理 UB 区间被 MTE3 store 源 tile 使用后，又马上作为 MTE2 load 目的 tile 复用的场景。此时 InsertSync 往往需要建立更强的 `MTE3 -> MTE2` 依赖，后续 load 不能像不同地址时那样提前发起：

```text
mteStoreThenLoadCoLocated(opA, opB):
  opA.pipe == PIPE_MTE3
  opB.pipe == PIPE_MTE2
  distance(opA, opB) <= mteLookahead
  && overlap(any opA.reads, any opB.writes)
```

该代价对 `tstore` 后接下一段 `tload` 的 loop 尤其敏感。若 candidate offset 导致 `opA.reads` 与 `opB.writes` 共址，应给该 candidate 加较高 penalty。

### 5. Candidate 评分

建议使用简单可解释的 penalty 累加：

```text
reuseCost(candidate):
  cost = 0

  if creates PIPE_V write/read or write/write co-location:
    cost += pipeVOverlapPenalty

  if creates MTE3-store-source -> MTE2-load-dst co-location:
    cost += mte3ToMte2Penalty

  if joins a hot loop scratch root into an already hot reuse group:
    cost += hotClusterPenalty + rootHotness

  if the candidate implies exact co-location for hot UB/L1 roots:
    cost += sameBankRiskPenalty

  if conflict is inside a loop body:
    cost *= loopWeight

  if root size is small and current AddressSpace has enough remaining capacity:
    prefer fresh offset by subtracting freshAddressBonus
```

推荐初始权重：

```text
pipeVOverlapPenalty = 10
mte3ToMte2Penalty   = 20
hotClusterPenalty   = 6
sameBankRiskPenalty = 4
loopWeight          = 4
freshAddressBonus   = 1
```

这些权重只决定多个可行 candidate 的选择顺序，不改变 safety gate。若没有任何低 cost candidate，planner 仍可选择高 cost candidate 并保持正确性。

`hotClusterPenalty` 使用 root 的访问统计，而不是硬编码 op 名称。一个 root 若在 loop 内被 PIPE_V/MTE 访问、在 loop 内有多次 read/write，或即使没有显式 `scf.for` 但存在多次 PIPE_V/MTE local 访问，就视为 hot scratch。后者覆盖 PTODSL 通过 task/block 并行表达重复工作的 kernel。把 hot root 继续并入已有 hot group 虽然可能语义安全，但容易把多个高频阶段压到同一小段 UB/L1 地址，导致同步分析保守化、bank/cache 热点或流水 overlap 窗口缩小。`sameBankRiskPenalty` 首版只建模最强信号：candidate reuse 会让两个 hot root 精确同址，因此一定落入同一 bank pattern；后续若 planner 在 materialize 前具备精确 offset/stride interval，可扩展为 `offset % bankModulo` 的更细判断。

容量压力仍优先于性能 hint：若 fresh group 会让剩余 local 空间低于 planner 的保守 reserve，则必须选择合法 reuse group，避免为了展开 hot scratch 造成后续 root overflow。

Cube 相关 local space（`MAT` / `LEFT` / `RIGHT` / `ACC`）不使用 largest-first 排序，即使用户显式选择 modern planner 默认的 `orderBySize`。这些空间的性能更依赖计算流附近的 L1/L0A/L0B/ACC 地址规律；把大 tile 全部提前规划会打散 `TLOAD -> TEXTRACT -> TMATMUL` 周边的 operand 地址模式，可能触发不利 bank pattern。`VEC` 仍保留 largest-first，因为 VEC scratch 的主要风险通常是容量压力和跨 PIPE_V/MTE 热点复用，而不是 cube operand 的固定 L0 bank 节奏。

### 6. prefill_c4_state_update 类场景

典型退化模式：

```text
legacy:
%t        addr = 0
%pool_dep addr = 256
%ape_row  addr = 512
%tmp0     addr = 768
%out0     addr = 1024
%tmp1     addr = 1280
%out1     addr = 1536
%tmp2     addr = 1792

modern aggressive:
%t        addr = 0
%pool_dep addr = 0
%ape_row  addr = 256
%tmp0     addr = 512
%out0     addr = 512
%tmp1     addr = 512
%out1     addr = 256
%tmp2     addr = 256
```

从四道闸门看，modern aggressive 规划可能是合法的：这些 scratch 的静态生命周期不重叠，op semantic no-alias 也未禁止它们复用。但从流水性能看，它把多个连续 V 计算和 store/load 交错阶段压到同一物理地址，使后续同步分析必须把本可 overlap 的阶段串起来。

性能模型期望在 Vec 容量充足时选择更接近 legacy 的展开地址；只有当 Vec 容量确实不足时，才逐步接受 `0/256/512` 这类压缩复用。

### 7. 与 largest-first-fit 的关系

Largest-first 仍决定 item 处理顺序；性能代价模型只影响“当前 item 放入哪个 candidate offset”。推荐流程：

```text
for item in largestFirstOrder:
  candidates = collectExistingReuseGroupsAndFreshOffsets(item)
  candidates = filterByFourGates(candidates)
  choose min(reuseCost(candidate), offset, stableOrder)
```

其中 tie-breaker 仍保持 deterministic：

```text
cost 升序
offset 升序
representative stableOrder 升序
```

这样既保留 largest-first 的确定性，也避免“第一个空洞”把高频 scratch 过早压到低地址。

## PTOAS 数据结构建议

### RootInfo

```cpp
struct RootInfo {
  Value root;
  Operation *defOp = nullptr;
  AddressSpace space = AddressSpace::Zero;
  uint64_t slotBytes = 0;
  uint64_t totalBytes = 0;
  uint64_t alignmentBytes = 1;
  uint64_t slotCount = 1;
  unsigned allocIndex = 0;
  unsigned freeIndex = 0;
  unsigned stableOrder = 0;
  SmallVector<uint64_t> offsets;
};
```

本设计保持当前 modern memplan 的 `RootInfo` 字段不变。`RootInfo` 只表达 local allocation root 的基础事实：

```text
root identity
definition op
address space
slot size / total size / slot count
lifetime interval
stable order
planned offsets
```

四道闸门所需的额外事实不直接塞进 `RootInfo`，而是放在 `ConflictFacts` 这类 side table 中。这样可以避免 root 结构随着每个闸门膨胀，也便于后续按阶段启用或删除某个闸门。

### ReuseGroup

```cpp
struct ReuseGroup {
  Value representative;
  SmallVector<unsigned> memberIndices;
  AddressSpace space;
  uint64_t slotSizeBytes;
  uint64_t offsetBytes;
};
```

### ConflictFacts

```cpp
struct ConflictFacts {
  DenseMap<Value, SmallVector<Value>> forbidAlias;
  DenseSet<Value> loadDerivedRoots;
  DenseSet<Value> tpopConsumerRoots;
  DenseMap<Value, SmallVector<unsigned>> phiFamilyIds;

  // Performance-only facts. These do not decide whether reuse is legal; they
  // only rank legal candidate offsets when modern memplan has spare capacity.
  SmallVector<OpAccess> opAccesses;
  DenseMap<Value, SmallVector<PhysicalInterval>> plannedIntervals;

  // Reserved for future implicit pipeline lowering support. Not used by the
  // current PTOAS design because explicit multibuffer owns ping-pong slot
  // separation.
  // DenseMap<Value, SmallVector<PipelineMembership>> pipelineMembership;
  // DenseSet<Value> pipelineLoadRoots;
};
```

### 说明：pipeline stage load conflict（预留，不纳入当前实现）

- 当前不实现 PyPTO 闸门 4。
- 不新增 pipeline stage load 复用负例。
- 保留设计占位：若未来 PTOAS 引入隐式 pipeline lowering，并能稳定提供 `pipelineMembership[root] = (group, stage)`，再接入该闸门。
- 显式 `pto.alloc_multi_tile count=N` 继续由 multibuffer slot 分配保证 ping-pong 正确性。

## 测试计划

### lit 测试

新增或扩展以下测试：

- `plan_memory_order_by_size_*.pto` 继续作为 largest-first 覆盖。
- `plan_memory_five_gates_lifetime_touching.pto`
- `plan_memory_five_gates_phi_family.pto`
- `plan_memory_five_gates_semantic_no_alias.pto`
- `plan_memory_five_gates_target_hazard.pto`
- `plan_memory_pipev_reuse_cost_state_update.pto`：构造多个连续 PIPE_V scratch，验证容量充足时 modern memplan 不把所有 touching live range 压到同一小段 Vec 地址。
- `plan_memory_pipev_reuse_cost_capacity_pressure.pto`：构造 Vec 容量接近上限的场景，验证性能 cost 只影响 candidate 排序，不会因为偏好 fresh address 而错误拒绝合法复用。
- `plan_memory_mte3_mte2_reuse_cost.pto`：构造 `tstore` 源 tile 后接 `tload` 目的 tile 的近邻复用场景，验证 planner 优先选择不会制造 `MTE3 -> MTE2` 共址依赖的地址。
- 暂不新增 `plan_memory_five_gates_pipeline_load.pto`；闸门 4 当前为预留设计。

已有 `plan_memory_*.pto` 应继续保留 legacy + modern 双 RUN。

### 验证命令

```bash
cmake --build build --target ptoas -j8

PATH=/Users/fangrui/workspace/huawei/llvm21-workspace/llvm-project/llvm/build-assert/bin:$PATH \
  /Users/fangrui/workspace/huawei/llvm21-workspace/llvm-project/llvm/build-assert/bin/llvm-lit \
  -sv build/test/lit \
  --filter 'plan_memory'

ctest --test-dir build --output-on-failure -L PTODSL
```

## 风险与注意事项

- `--plan-memory-order-by-size` 本身会改变 modern memplan 的 offset 分配顺序，测试应避免把 order-sensitive 预期错误地复用于默认路径。
- 四道闸门是硬约束，不应为了容量不足而放松。
- target hazard 需要先确认 PTOAS IR 中稳定的标记来源。
- pipeline metadata 闸门当前不实现；如果未来启用，需要先定义稳定的 pipeline membership 来源。
- phi family 豁免必须保守，不能让外部 live alias 借互斥分支错误复用。
- PIPE_V 物理共址模型只是性能排序，不是安全闸门；容量不足时不能因为 cost 高而报错，除非四道安全闸门本身失败。
- cost 权重会改变 modern memplan 的 offset 选择，测试应只检查关键“不应过度压缩”的相对关系，避免过度绑定完整地址布局。
- 物理区间必须和 InsertSync 使用的 root/alias/MemoryEffects 视角保持一致；否则 planner 认为低 cost 的地址，后续同步分析仍可能补出强依赖。
- legacy memplan 不应受该设计影响，默认行为保持不变。
