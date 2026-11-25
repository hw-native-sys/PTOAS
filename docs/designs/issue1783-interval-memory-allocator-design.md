# 区间内存分配器：用 Best-Fit 装箱替换 ReuseGroup 分桶

当前实现状态：P0–P3 代码已实现，默认仅支持 `interval`；`group` 与
`order-by-size` 已移除。下文设计与 P0/P1 记录保留阶段背景，最终状态和验证结果见文末。

关联 issue：[hw-native-sys/PTOAS#1783](https://github.com/hw-native-sys/PTOAS/issues/1783)
下游追踪：[hw-native-sys/pypto#3069](https://github.com/hw-native-sys/pypto/issues/3069)
涉及文件：`lib/PTO/Transforms/Passes/PTOPlanMemoryModern.cpp`

---

## 【背景】

### 1. 现象

```
ptoas --pto-arch=a3 --pto-level=level2 --emit-pto-ir fifo.pto -o /dev/null
→ error: failed to reserve a local memory hole for reserve_buffer
```

用例（issue 1783 最小复现）：a3 的 VEC 容量 196608 B（192 KiB），其中

| buffer | size | 生命周期 |
|---|---|---|
| `fifo`（`reserve_buffer`，auto） | 64 KiB | 整个 function |
| `a` | 64 KiB | t1 – t5 |
| `b` | 64 KiB | t2 – t6 |
| `c` | 128 KiB | t7 – t9 |

peak liveness = `max(fifo+a+b, fifo+c)` = 192 KiB，**理论上刚好放得下**，但规划失败。

### 2. 根因一：阶段顺序 —— `reserve_buffer` 不在容量预算内

`runModernPlanMemory` 的顺序是：

1. `planOneAddressSpace`（`:1764`）给所有 tile 排地址，上界用**完整的** `spec.capacityBytes`
2. 之后才 `planReserveBuffers`（`:1798`），在 tile 留下的空洞里找位置

但 auto `reserve_buffer` 的生命周期是整个 kernel，**永远不能与任何 tile 共享地址**（`validateReserveBufferForPlanMemory`，`:1590`：PlanMemory 里只接受 `auto = true` 且无 `base`），它却完全没参与第 1 步的预算。第 1 步合法地用光 192 KiB，第 2 步必然失败。

### 3. 根因二：ReuseGroup 分桶的表达能力不足（本质原因）

当前布局模型（`assignReuseGroupsAndLayout`，`:1721`）是一维分桶：

- 每个 root 归属**唯一**一个 `ReuseGroup`
- `group.sizeBytes = max(成员 totalBytes)`
- 各 group 从 0 **顺序紧排**，且该 extent **在整个 function 期间被该 group 独占**

group 的地址范围在时间上是静态的，不存在跨 group 的时序共享。本例实际轨迹：

| root | 决策 |
|---|---|
| `a` 64K | 新建 g0 = 64K |
| `b` 64K | 与 `a` 同时 live，`canShare` 失败 → 新建 g1 = 64K |
| `c` 128K | 可并入 g0（`a` 已死）→ g0 撑到 128K |

布局 g0 `[0,131072)` + g1 `[131072,196608)` = **192 KiB 占满，零空洞**。

碎片来源：`a`、`b` 同时存活时，占用是 `[0,64K)` 和 `[128K,192K)`，中间 `[64K,128K)` 的空洞属于 g0、为 `c` 长期预留，`b` 碰不到。**真实峰值 128 KiB，布局却要 192 KiB，碎片率 1.5。**

issue 期望的布局是 `c` 横跨 `a`+`b` 腾出的**连续** 128 KiB —— 分桶模型表达不出来：`c` 只能加入 g0 或 g1 之一。

另一个过度约束：`canJoinReuseGroup`（`:1240`）要求新 root 与 group **全员** `canShare`，实际是在求传递闭包。`b` 只和 `a` 冲突，却因 `a` 在 g0 里而被整个 g0 拒绝。

### 4. 为什么现有 `orderBySize` 无效

两点：

1. **默认关闭**：`Passes.td:207`、`ptoas.cpp:289` 均 `init(false)`，issue 的命令行未开启
2. **开启也修不了**：`order-by-size=true` 时顺序变 `c, a, b` —— `c` 新建 g0=128K，`a` 与 `c` 不冲突并入 g0（g0 仍 128K），`b` 与组内 `a` 冲突 → 新建 g1=64K。**布局字节与默认完全相同**

（`a` 确实会并入 g0 而非另起一组：`preferReuseOverFresh`（`:1434`）在 `freshSlack = 0 < pressureReserve = 65536` 时强制走复用分支。）

`orderBySize` 只能改变 group 的创建顺序和归属偏好，改不了"group extent = max 成员、全程独占"这两条硬性质。本例的限制因素是**跨 group 的时序碎片**，不是排序。

---

## 【业界对标】

静态内存规划的学名是 **DSA（Dynamic Storage Allocation）**：等价于固定 x 轴的二维装箱 —— 每个 buffer 是矩形，宽 = 生命周期（时间轴位置**固定不可平移**），高 = 字节数，要求互不重叠、最小化总高度。NP-hard；**下界 = peak liveness**，衡量指标是碎片率 `plannedBytes / peakLiveBytes`。

| 代际 | 代表实现 | 机制 | 碎片率 |
|---|---|---|---|
| 一代：图着色 / 槽位合并 | LLVM `StackColoring` | 不冲突的 buffer 合并到**同一槽位**（all-or-nothing） | 1.3–2.0+ |
| **二代：降序 + Best-Fit 区间装箱** | XLA `GlobalDecreasingSizeBestFitHeap`、TFLite `GreedyMemoryPlanner`、TVM USMP、MindSpore SOMAS | size 降序，逐个在空闲区间 best-fit，**允许部分重叠** | 1.0–1.1 |
| 三代：求解器辅助 | TelaMalloc (ASPLOS'22)、MiniMalloc (ASPLOS'23, `google/minimalloc`) | 启发式 + 分支限界/约束求解 | ≈1.0 |

PTOAS 现状等价于一代：`ReuseGroup` 的 `size = max(成员)` 就是着色的 all-or-nothing 语义。着色适合等宽的寄存器，不适合**变宽**的 buffer。

**本方案选二代。** 不上三代的理由：求解器的价值在超大规模整图（数百至上千 buffer）；PTOAS 是 kernel 级，单 space 的 root 通常几十个，二代贪心基本即最优，上求解器是负收益（编译时间）。但接口留口子（见下）。

---

## 【设计】

### 1. 总体结构

```
planOneAddressSpace(space)
  ├─ 收集该 space 的 RootInfo            （不变）
  ├─ 为该 space 的 auto reserve_buffer 合成 pinned 伪 root   ← 新增
  ├─ 构造 AllocRequest[] + conflictFn/costFn 闭包            ← 新增
  ├─ IntervalMemoryAllocator::run(...)                       ← 新增（纯算法）
  ├─ verifyAllocationInvariants(...)                         ← 新增（安全网）
  └─ 回写：tile → buffer2Offsets；reserve → base 属性
```

`planReserveBuffers` 的独立阶段**取消**，reserve 并入统一分配。这同时消除根因一。

### 2. 纯算法单元接口

新文件 `lib/PTO/Transforms/Support/IntervalMemoryAllocator.{h,cpp}`（CMake 挂到 `lib/PTO/Transforms/CMakeLists.txt`，与 `Support/Utils.cpp` 同列）。**不依赖 MLIR 类型**，只吃整数和谓词闭包 —— 这样它可被合成用例直接驱动，将来替换为 solver 只改这一块。

```cpp
struct AllocRequest {
  unsigned id = 0;             // 调用方的稳定标识（RootInfo 下标）
  uint64_t sizeBytes = 0;      // 已按 space alignment 对齐的总字节
  uint64_t alignmentBytes = 1;
  uint64_t orderKey = 0;       // 稳定 tie-break（allocIndex）
  bool pinned = false;         // 必须最先放置（reserve_buffer）
};

struct AllocResult {
  SmallVector<uint64_t> offsets;  // 按 request 下标
  uint64_t plannedBytes = 0;      // 实际占用高度
};

struct AllocFailure {
  unsigned failedId = 0;
  uint64_t requestedBytes = 0;
  uint64_t largestFreeGapBytes = 0;
  uint64_t plannedBytes = 0;
};

// conflictFn(i, j)：i 与 j 的地址区间是否必须互斥
// costFn(i, oi, j, oj)：i 放在 oi、与已放置的 j(@oj) 地址重叠时的性能惩罚
using ConflictFn = llvm::function_ref<bool(unsigned, unsigned)>;
using CostFn =
    llvm::function_ref<uint64_t(unsigned, uint64_t, unsigned, uint64_t)>;

enum class OrderPolicy { SizeDescending, ProgramOrder };

FailureOr<AllocResult> allocateIntervals(ArrayRef<AllocRequest> requests,
                                         uint64_t capacityBytes,
                                         OrderPolicy policy,
                                         ConflictFn conflictFn, CostFn costFn,
                                         AllocFailure &failureOut);
```

peak liveness 不由分配器计算（只给 pairwise 谓词时求最大团是 NP-hard）；由 `planOneAddressSpace` 用 `[allocIndex, freeIndex)` 扫一遍算出，仅用于诊断。

### 3. 主循环

```
order = stable_sort(indices) by
          (pinned desc, policy==SizeDescending ? sizeBytes desc : 0,
           orderKey asc, id asc)

placed = []                       # (id, lo, hi)
for i in order:
    blockers = mergeSorted([ (lo,hi) for (j,lo,hi) in placed
                             if conflictFn(i, j) ])        # ← 核心过滤
    gaps = enumerateGaps(blockers, requests[i], capacity)
    if gaps.empty(): return failure(i, largestFreeGap(blockers, capacity))
    best = min(gaps, key = (waste, cost(i, gap.offset), gap.offset))
    placed.append((i, best.offset, best.offset + sizeBytes))
    offsets[i] = best.offset
plannedBytes = max(hi for placed)
```

`blockers` 的过滤是整个修复的机制本身：**只有与当前 root 真正冲突的已放置区间才算障碍物，不冲突的直接忽略，可以放心压在其地址上。** 复用不再靠事先聚类，而是放置时的过滤 —— 从"必须与 group 全员兼容"收窄为"只需与地址真正压上去的那些兼容"，根因二的过度约束随之消失。

若不做这个过滤（所有已放置区间都当障碍物），算法退化为纯 bump 分配、零复用。

### 4. 空隙枚举与 Best-Fit 打分

```
enumerateGaps(blockers, req, capacity):      # blockers 已合并且按起址升序
  gaps = []; cursor = 0
  for (lo, hi) in blockers:
      start = alignUp(cursor, req.alignmentBytes)
      if start + req.sizeBytes <= lo:
          gaps.append({offset: start, extent: lo - start})      # 内部空洞
      cursor = max(cursor, hi)
  start = alignUp(cursor, req.alignmentBytes)
  if start + req.sizeBytes <= capacity:
      gaps.append({offset: start, extent: capacity - start})    # 尾部空间
  return gaps
```

`waste = gap.extent - req.sizeBytes`，取 waste 最小者即 **best-fit**。

相比 lowest-fit（取第一个塞得下的）的优势：尾部空隙的 extent 是 `capacity - start`，waste 天然很大，于是 best-fit **自动偏好把小块塞进内部空洞、把连续的高地址空间留给后续大块**；lowest-fit 则倾向把低地址啃出一堆放不下大块的碎缝。配合 size 降序，这是二代算法碎片率的主要来源。

打分为字典序 `(waste, cost, offset)`：先保证装箱质量，再考性能，最后用地址保证确定性。按 waste 容差分桶后再比 cost 留作后续优化。

### 5. `reserve_buffer` 建模：pinned 伪 root

为该 space 每个 auto `reserve_buffer` 合成一个 `RootInfo`：

| 字段 | 取值 |
|---|---|
| `root` | `reserveOp` 的结果 Value |
| `totalBytes` / `slotBytes` | `alignUp(reserveOp.getSize(), spec.alignmentBytes)` |
| `alignmentBytes` | `spec.alignmentBytes` |
| `slotCount` | 1 |
| `allocIndex` | `0` |
| `freeIndex` | `UINT_MAX` |

对应 `AllocRequest.pinned = true`。

**关键性质：这样建模后，现有 `canShare`（`:1232`）不需要任何特判就得出正确结论。** 逐个 gate 验证：

- `gateLifetimeAndPhi`：`[0, UINT_MAX)` 与一切区间 `lifetimesStrictlyOverlap`（`:1175`）→ 进入 `areBranchExclusive`，`branchExclusiveRoots` 不含伪 root → 返回 `false`
- `gateTargetHazard`：`loadDerivedRoots` / `tpopConsumerRoots` 不含伪 root → 无 hazard → 通过
- `gateSemanticNoAlias`：`forbidAlias` 不含伪 root → 通过

`gateLifetimeAndPhi` 返回 `false` ⇒ `canShare` 为 `false` ⇒ 伪 root **与所有 root 冲突**，正是"全生命周期、不可共享"的语义。

多个 reserve_buffer 之间也互相冲突（同样走上述路径），彼此顺序排列；它们之间按 `(sizeBytes desc, 程序序)` 确定。

### 6. cost model 迁移

当前 `getGroupReuseCost` / `getHotClusterReuseCost`（`:1340`–`:1409`）是 root-vs-group，需改为 root-vs-root 的 `getPairReuseCost(const RootInfo &, const RootInfo &, const ConflictFacts &)`，由 `costFn` 闭包对**地址会重叠的已放置 root** 求和。注意这些 root 按定义都是不冲突的（冲突的已成为 blocker），所以求和对象正是真正的复用伙伴，语义与今天一致。

**bank 冲突建模顺带变精确。** `:1399` 现有注释写着：

> The planner does not materialize final byte intervals yet, so only model the bank pattern that is known for a reuse candidate: the new root would start at the same offset as every member in this group.

区间分配器**确实**产出最终字节偏移，因此可以把"同起始 offset"这个代理条件换成真实判断：

```cpp
(offsetI % kBankConflictModuloBytes) == (offsetJ % kBankConflictModuloBytes)
```

常量 `kBankConflictModuloBytes = 8192` / `kBankConflictPenalty = 4`（`:1382`–`:1383`）不变。

### 7. 不变量校验（安全网）

分桶模型靠结构天然保证"不兼容的 root 不会同址"（不兼容 ⇒ 不同组 ⇒ 不重叠）。区间模型没有这个结构保证，**必须显式断言**：

```cpp
// 恒开（非仅 debug）：n 为单 space root 数，通常几十，O(n²) 可忽略
for each i:          offsets[i] % alignmentBytes == 0
                     offsets[i] + sizeBytes <= capacityBytes
for each pair i<j:   conflictFn(i,j) ⇒ [oi, oi+si) ∩ [oj, oj+sj) == ∅
```

违反时报 internal error 并 `return failure()`，绝不静默产出错地址。这是替代结构保证的核心防线。

`verifySemanticNoAliasRanges`（`Support/Utils.cpp:870`）保持不变并继续依赖 —— 它已是 byte-range 语义（`haveOverlappingTileRanges`），本来就不认 group，自动覆盖部分重叠的情形。

### 8. 诊断信息

现有报错 `failed to reserve a local memory hole for reserve_buffer`（`:1826`）和 `vec overflow, requires N bits while M bits available`（`:1785`）都不可操作。新格式同时给出理论下界，让用户能区分"真放不下"和"碎片化"：

```
error: vec overflow: cannot place a 131072-byte buffer
  capacity              : 196608 bytes
  already planned       : 196608 bytes
  largest free hole     : 65536 bytes
  peak live (lower bound): 131072 bytes
  note: planned footprint exceeds peak liveness; this is allocator
        fragmentation, please attach the IR to an issue
```

当 `plannedBytes > capacityBytes >= peakLiveBytes` 时附上 note —— 这种情况是分配器质量问题而非用户问题，值得被动上报。

### 9. 配置开关

新增 pass option 与 CLI flag：

| 层 | 新增 |
|---|---|
| `include/PTO/Transforms/Passes.td` | `Option<"allocator", "allocator", "std::string", "\"interval\"", "...">` |
| `tools/ptoas/ptoas.cpp` | `--plan-memory-allocator=interval\|group` |

- `interval`（默认）：本方案
- `group`：保留旧分桶模型一个发布周期，供 pypto 做性能 A/B 与紧急回退，标注 deprecated

`order-by-size` 在 `interval` 下**不再生效**：区间分配器以 size 降序为前提（greedy-by-size 的标准假设），非 cube space 恒为 `SizeDescending`。它仅在 `group` 下保留原语义，与 `group` 一同移除。

**cube space 保守处理**：`isCubeLocalSpace`（MAT/LEFT/RIGHT/ACC，`:130`）初期用 `OrderPolicy::ProgramOrder`，与今天 `sizeFirstForSpace = orderBySize && !isCubeLocalSpace`（`:1770`）的豁免保持一致，避免在 L0A/L0B/ACC 这些硬件约束尚未完全梳理的空间上同时改动两个维度。分配器本身对所有 space 生效。

---

## 【改动清单】

### 删除

`PTOPlanMemoryModern.cpp` 中约 100 行：`ReuseGroup`（`:87`）、`canJoinReuseGroup`（`:1240`）、`chooseReuseGroupByCost`（`:1472`）、`getProjectedPackedBytes`（`:1411`）、`preferReuseOverFresh`（`:1434`）、`assignReuseGroupsAndLayout`（`:1721`）、`planReserveBuffers`（`:1798`）、`planReserveBufferBase`（`:1523`）。

`ReuseGroup` **无下游依赖**（仅活在 `planOneAddressSpace` 内部，唯一产物是每个 root 的 `offsets`），删除不牵动其他层。

### 保留不改

`canShare` 及三道 gate、`lifetimesStrictlyOverlap`、`areBranchExclusive`、`hasTargetHazard`、`buildSlotOffsets`、`materializePlannedOffsets`、`verifySemanticNoAliasRanges`、`verifyNoUnplannedAllocTile`、`validateReserveBufferForPlanMemory`。

### multi-tile 槽位

`slotCount > 1` 的 root 按 `totalBytes = slotBytes * slotCount`（`:496`）整段放置，再用 `buildSlotOffsets` 展开。"槽位连续"这个不变量在区间模型下天然成立，不需要 group 维护。

### 跨层同步（按 `.claude/rules/cross-layer-sync.md`）

- [ ] ODS `*.td`：**无 op 变更**，不涉及
- [ ] C++ verifier：无语义约束变更
- [ ] `Passes.td` + `ptoas.cpp`：新增 `allocator` option / flag，option 透传入 pass 构造（不引入全局可变状态）
- [ ] Python bindings：不涉及
- [ ] docs：本文档 + `order-by-size` 的 deprecated 说明
- [ ] tests：见下

---

## 【测试方案】

仓库无 gtest 基础设施（`unittests/` 不存在，CMake 无 GTest 引用），因此纯算法单元以 lit `.pto` 用例驱动。

### 新增

| 用例 | 覆盖点 |
|---|---|
| `test/lit/pto/issue1783_reserve_buffer_interval_alloc.pto` | issue 原始 IR；CHECK `base = 0`、`a@65536`、`b@131072`、`c@65536` |
| `test/lit/pto/plan_memory_interval_big_reuses_union.pto` | 一个大 tile 复用 N 个小 tile 腾出的连续区间（根因二的一般形式） |
| `test/lit/pto/plan_memory_interval_best_fit_vs_lowest_fit.pto` | 构造 lowest-fit 会浪费、best-fit 不浪费的用例，锁定 best-fit 策略 |
| `test/lit/pto/plan_memory_interval_reserve_multi_tile.pto` | `reserve_buffer` + `slotCount > 1` 的槽位连续性 |
| `test/lit/pto/plan_memory_interval_overflow_diag.pto` | 新诊断文本，含 peak/fragmentation note |

本地已有未跟踪的 `test/dsl/issue_1783_repro.pto`，整理后并入上述第一条。

### 现有用例

36 个用例引用 `pto-plan-memory`，其中 **34 个断言具体 `addr`**。切到 interval 后地址普遍变化（方向是占用变紧），需逐一更新 CHECK。这是本方案最大的机械工作量，也是分阶段落地的主要动因。

---

## 【风险】

| 风险 | 说明 | 缓解 |
|---|---|---|
| **部分重叠语义** | 分桶下共享 = 同起始 offset + 完全包含；区间下共享 = **可部分重叠**（本例 `a` 占 `c` 前半、`b` 占 `c` 后半）。对 liveness 精度要求提高：原来 liveness 有偏差要么完全同址（错得整齐、易被 no-alias verifier 抓）要么完全无关，现在会出现"半重叠"，只破坏一部分数据，症状更隐蔽 | §7 恒开不变量断言；`verifySemanticNoAliasRanges` 继续生效；重点验证 multi-buffer / 流水展开（liveness 本就偏保守）场景 |
| **34 个 lit 用例地址 churn** | 工作量集中、易误改 | 分阶段：P1 带 flag 不动默认，P2 统一翻转并一次性更新 |
| **bank 冲突导致性能回退** | lowest/best-fit 可能把热 buffer 挤到同一 bank | cost model 迁移 + 真实 offset 取模判定（反而比今天更准）；P2 跑 pypto-lib 性能 A/B |
| **cube space 未知硬件约束** | L0A/L0B/ACC 约束未完全梳理 | 初期保持 `ProgramOrder`，与今天行为一致 |

---

## 【实施阶段】

| 阶段 | 内容 | 验收 |
|---|---|---|
| **P0** | `IntervalMemoryAllocator.{h,cpp}` + 不变量校验，**不接线** | 编译通过，行为零变化 |
| **P1** | 接入 `planOneAddressSpace`，默认仍 `group`；新增 5 个 lit 用例显式带 `--plan-memory-allocator=interval` | issue 1783 复现用例通过；34 个旧用例不动即通过 |
| **P2** | 默认翻转为 `interval`；更新 34 个旧用例 CHECK；pypto-lib 性能 A/B | 全量 lit 通过；pypto#3069 原始 kernel 编译通过；makespan 不退化 |
| **P3** | 删除 `group` 模型与 `order-by-size` | 代码净减约 100 行 |

P0/P1 可并行推进：纯算法单元与 lit 用例互不依赖。


## 实施记录（P0/P1）

本阶段提供 `--plan-memory-allocator=interval`，默认仍为 `group`；旧地址断言保持不变。
切换默认和删除回退路径分别保留为 P2、P3，需先完成本文所列原始 kernel 和性能验收。

实现相对接口草案的调整：

- 算法单元返回 `std::optional<AllocResult>`，避免依赖 MLIR 的 `FailureOr`。
  谓词接收 request 数组下标，稳定 ID 用于排序与诊断。
- pinned 请求由算法强制与所有非空请求互斥，包括零长度生命周期的 root。
- 所有地址先用剩余容量做减法检查，再进行加法；性能代价求和饱和到 `uint64_t` 上限。
- 诊断输出 `peak live (linear lifetime estimate)`。线性扫描会同时计入分支互斥的 root，
  也不能表达生命周期之外的禁止别名约束，因此不能无条件将其称为真实峰值下界，
  或仅据此断言失败必然由碎片导致。
- 所有空间均检查地址对齐、容量和两两冲突；原有语义禁止别名区间校验继续执行。
- #1784 的多 slot 生命周期分析属于独立问题，本阶段不修改 `markUse`、`markWrite`
  或循环生命周期扩展规则。

### 远端验证结果（2026-11-25）

- 本地 worktree：`codex/issue1783-interval-allocator`，基线 `dc3ecefb9`。
- Worker：`fangrui@123.60.231.48:1211`，源码与独立构建目录：
  `/home/fangrui/PTOAS-remote/issue-1783-memplan/build-remote`。
- GCC 12.3、LLVM/MLIR 19.1.7、Python 3.10.12，ABI 为 `_GLIBCXX_USE_CXX11_ABI=0`。
- 远端构建 `tools/ptoas/all`、`pto-test-opt` 和相关测试工具成功。
- 纯算法边界检查通过：空输入、跨多个小块复用、pinned 互斥、不变量失败、
  对齐、零尺寸、无效对齐、容量不足和 `uint64_t` 边界。
- 7 个新增 lit 用例全部通过，相关回归合计 92/92 通过。
- `cmake --build build-remote --target check-pto -j16`：发现 2568 个用例，
  2566 通过、1 unsupported、1 失败。唯一失败 `deepseek_score_inputs.test`
  使用的系统 `python3` 缺少 numpy；将已配置的 venv 加入 PATH 后，该项单独复测通过。
  未将 unsupported 计入通过，也未将首轮全量运行称为全绿。
- issue1783 最终地址：FIFO=0，a=65536，b=131072，c=65536；
  显式 group 模式仍复现原始 FIFO 分配失败。
- changed-code 预检：8 个代码文件、10 处 `G.FMT.11-CPP` 命中，均为正则将条件中
  内层右括号后的运算符误当成无括号语句体；逐行确认所有实际语句体均有大括号。
  无其他检查发现。远端 `git diff --check` 通过；未运行 EChecker/SecK 专用分析器，
  不宣称完整静态分析合规。
- 未运行设备数值验证、原始 PyPTO kernel 和 pypto-lib 性能 A/B；因此不切换默认。

远端日志位于 `/home/fangrui/PTOAS-remote/`：
`issue1783-interval-rebuild.log`、`issue1783-related-final.log`、
`issue1783-check-pto.log`、`issue1783-numpy-recheck.log`、
`issue1783-layout-final.log`、`issue1783-compliance-scoped.log`。

## 实施记录（P2/P3）

- Pass 和 CLI 默认使用 `interval`，保留 `--plan-memory-allocator=interval`
  作为显式选择；`group` 与其他非法值均给出诊断。
- 删除 ReuseGroup 分桶、旧 reserve-buffer 放置路径以及 `order-by-size` 的
  Pass/CLI 选项和参数透传。非 cube 空间固定按尺寸降序，cube 空间保留程序顺序。
- 更新受地址变化影响的回归断言和用户手册；原 order-by-size 用例改为
  interval 尺寸排序用例，保留复用与禁止复用两条路径。
- 新增非法 allocator、真实容量不足与碎片化诊断覆盖；issue1783、连续区间复用、
  best-fit 选择、reserve-buffer 与 multi-tile 连续槽位用例继续覆盖新分配器。

### 最终远端验证（2026-11-25）

- 沿用上述独立 worker 和 `build-remote`，构建 `tools/ptoas/all`、
  `pto-test-opt` 等相关目标成功。链接器报告串行 LTRANS 提示，未将其解释为源码警告。
- 最终全量 lit：2568 项中 2567 PASS、1 UNSUPPORTED、0 FAIL，耗时 416.24 秒。
  该结果来自最终断言修正后的完整运行，覆盖此前定向回归中发现的失败。
- changed-code 检查扫描 8 个代码文件，报告 9 处 `G.FMT.11-CPP`：
  `PTOPlanMemoryModern.cpp` 的 1352、1551、1607 行，以及
  `IntervalMemoryAllocator.cpp` 的 41、83、101、121、145、157 行。
  逐处阅读确认语句体均有大括号；报告来自检查器对含嵌套括号或多行条件的误判。
  无其他检查发现；未运行 EChecker/SecK 或专用复杂度、重复率分析器，
  不宣称完整静态分析合规。
- 未运行原始 PyPTO kernel、设备数值验证及 pypto-lib 性能 A/B，
  因此 P2 的 kernel 与 makespan 验收仍待完成，不能据 lit 结果宣称性能无退化。
  P3 代码按用户要求一并完成，未等待上述性能验收。

最终日志位于 `/home/fangrui/PTOAS-remote/`：
`issue1783-p3-build-final.log`、`issue1783-p3-lit-verified.log`、
`issue1783-p3-compliance.log`。
