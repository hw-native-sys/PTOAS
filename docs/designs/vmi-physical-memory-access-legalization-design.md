# VMI 物理访存规划与 VPTO 合法化设计

## 1. 状态与结论

本文档定义 VMI 访存在完成 layout assignment 后，如何先形成与具体 VPTO
指令无关的物理访存计划，再根据 A5 dist 约束、地址证明和内存安全条件选择
direct dist、stateful stream 或 point access。

设计采用以下流水：

```text
VMI op + assigned layout
  -> physical access planning
  -> plan normalization and local coalescing
  -> VPTO memory legalization
  -> VPTO memory operations
  -> VPTOStatefulStreamFusion
  -> LLVM emitter
```

本次将 planning 和 legalization 实现为 `vmi-to-vpto` 内部的两个 C++ 阶段，
不增加可见的中间 IR，也不增加新的 pass。只有当计划需要被其他 pass 持久化、
检查或跨 op 改写时，才重新评估是否值得引入中间 dialect。

本设计不引入 `VPTOFallbackKind`。fallback 不是语义事实，不能作为核心模型。
legalizer 应根据以下事实推导实现：

- VMI 操作实际访问的有序内存元素；
- 内存元素与物理寄存器值之间的映射；
- direct dist 的 ISA contract；
- 地址对齐证明；
- 候选实现的真实读取和写入范围；
- predicate 是否生效；
- 相邻访问是否可以形成同一条 stateful stream。

## 2. 背景问题

当前 `VMIToVPTO.cpp` 中，VMI 语义物化、地址分析、dist 选择、寄存器重排和
stateful stream 构造分散在不同 lowering 分支中。例如：

- NORM load/store 分支会检查 32B 对齐，并在无法证明时生成 stateful 访问；
- UNPK/PK 分支会直接生成对应 dist，尚未统一检查 mode-specific alignment；
- DINTLV/INTLV 分支会直接生成 x2 dist；
- E2B 分支会直接生成 E2B load；
- group store 分支分别实现 aligned store、packed stateful store 和 1PT store；
- dist token、element width 和硬件立即数在 verifier、lowering 和 emitter 中重复维护。

这种结构会造成三个问题：

1. 指令选择发生得过早。生成多个 1PT 后，后续 pass 很难恢复它们是否来自同一个
   连续逻辑 payload，以及应从哪些 carrier lane 取值。
2. direct dist 和 fallback 可能使用不同的内存 footprint。例如 E2B 只读取少量
   group scalar，而普通 `vldus` 会形成一个完整 VL 的 load；如果没有额外可读范围
   证明，这不是无条件安全的替换。
3. 相同 ISA 事实可能在不同分支中得到不同解释，尤其是 BRC 的自然对齐、dist 的
   predicate 行为和 x2 操作的双 packet footprint。

因此需要先回答“语义上必须访问哪些字节”，再回答“目标 ISA 用什么指令访问”。

## 3. 目标与非目标

### 3.1 目标

1. 统一描述所有现有 VPTO vector load/store dist 的 token、方向、arity、element
   width、对齐、内存 footprint、predicate 行为和寄存器映射。
2. 让 VMI memory lowering 先产生物理访问事实，再统一做指令合法化。
3. 在仍保留 VMI layout 和 carrier-lane 信息时，合并真正连续的 point access。
4. 对无法证明 direct dist 对齐的访问，在内存安全和语义允许时生成 stateful
   load/store。
5. 在同一个 VMI 访问计划内部直接生成完整 stateful 流：load 使用
   `vldas -> multiple vldus`，store 使用
   `init_align -> multiple vstus -> vstas`。
6. 继续使用 `PTOAddressAnalysis` 的 typed address language 证明对齐、地址差和循环
   迭代间连续性，不建立第二套地址求解器。
7. 保持 `VPTOStatefulStreamFusion` 为后置 VPTO 优化，只处理已经合法化的 stateful
   流。

### 3.2 非目标

- 不让地址分析理解 dist、layout 或成本模型。
- 不让 `VPTOStatefulStreamFusion` 恢复 VMI 语义或选择 dist。
- 不把任意 masked store 转换成无 mask 的 `vstus`。
- 不允许为了使用 `vldus` 而无证明地扩大可观察读取范围。
- 不跨有可观察 memory effect 的 op 重排访问。
- 不在首期重新设计 VMI layout assignment。
- 不假设物理 vector 固定为 256B；所有 VL 相关规则使用实际物理 vector byte size。

## 4. 层次与职责

### 4.1 VMI physical access planner

planner 负责：

- 将已分配 layout 的逻辑 VMI memory op 解释成有序内存 segment；
- 描述每个 segment 的地址、元素类型、访问覆盖形状和逻辑长度；
- 描述有序内存元素与输入/输出物理 values 之间的映射；
- 记录 load 可被证明安全读取的范围；
- 保留生成寄存器 pack/unpack/select 所需的信息。

planner 不负责：

- 选择 `vlds`、`vstus` 或 1PT；
- 判断某个 dist 地址是否合法；
- 创建 VPTO memory op；
- 管理 align SSA state。

planner 可以引用 conversion adaptor 已经提供的物理 values 和 types。为避免提前锁死
direct PK/UNPK 路径，planner 不应先无条件生成 pack/unpack IR。

### 4.2 Plan normalizer

normalizer 只在同一个原始 VMI op 的有序 segment 内工作，负责：

- 删除静态零长度 segment；
- 计算或查询相邻 segment 的 byte difference；
- 在地址相邻、顺序一致、映射可组合、覆盖兼容且容量合法时合并 segment；
- 区分连续访问、连续前缀、任意 predicate 和真正稀疏访问；
- 保留不能证明连续的原始顺序。

它不跨 VMI op 合并，不跨 memory effect 重排，也不因为两个地址“看起来相似”而假设
alias 或连续。

### 4.3 VPTO memory legalizer

legalizer 负责：

- 用共享 dist contract 查找能直接表达 register mapping 的候选指令；
- 使用 `PTOAddressAnalysis` 证明候选指令要求的地址对齐；
- 检查候选路径的读取/写入范围和 predicate 行为是否保持 VMI 语义；
- 选择 direct dist、stateful stream、自然对齐 point access 或发出诊断；
- 仅在 fallback 路径需要时物化显式寄存器重排；
- 为一个 plan 内的连续访问建立一条 align SSA chain。

### 4.4 VPTOStatefulStreamFusion

该 pass 只处理已经生成的 VPTO stateful 流：

- 合并 basic block 内可证明连续且中间无冲突 effect 的流；
- 将可证明迭代间连续的流改写成 loop-carried align/base state；
- 保证 load 和 store align state 的 SSA 顺序及终止操作正确。

该 pass 不理解 UNPK、PK、E2B、DINTLV、INTLV 或 VMI layout。fusion 实现主体位于
`VPTOStatefulStreamFusion.cpp`，与 `VMIToVPTO.cpp` 中的 plan/legalization 职责分离。

## 5. 共享 VPTO Dist Contract

### 5.1 为什么需要共享 contract

共享 contract 的价值不是给分支命名，而是让 verifier、VMI lowering、其他 VPTO
优化和 emitter 从同一份 ISA 事实得到结论。它必须能被通用查询消费，否则只是一层
间接调用。

共享 contract 位于：

```text
include/PTO/IR/VPTOMemoryDist.h
lib/PTO/IR/VPTOMemoryDist.cpp
```

概念接口：

```cpp
enum class VPTOMemoryOpFamily {
  Load,
  LoadX2,
  Store,
  StoreX2,
};

enum class VPTOPredicatePolicy {
  Applied,
  Ignored,
  NotPresent,
};

struct VPTOMemoryDistContract {
  VPTOMemoryOpFamily family;
  VPTOMemoryDist dist;
  unsigned registerArity;
  VPTOPredicatePolicy predicatePolicy;
  VPTOMemoryTransferMapping transfer;

  FailureOr<int64_t>
  getRequiredAlignmentBytes(int64_t vectorBytes) const;

  FailureOr<int64_t>
  getFullActiveFootprintBytes(int64_t vectorBytes,
                              int64_t elementBytes) const;

  FailureOr<int64_t>
  getDependencyGranularityBytes(int64_t vectorBytes,
                                int64_t elementBytes) const;
};
```

contract 必须区分：

```text
required alignment      指令有效地址的合法性要求
semantic footprint      指令语义实际读取或写入的数据范围
dependency granularity  memory dependence 分析使用的覆盖粒度
physical read envelope  实现为得到结果实际可能读取的相对地址区间
```

不能用一个 `maximumMemoryBytes` 同时表示这些概念。semantic footprint 和
dependency granularity 属于 direct dist contract；physical read envelope 还取决于
所选指令序列，例如 stateful load 同时包含 `vldas` 和后续 `vldus`，应由 candidate
implementation 描述。

### 5.2 A5 对齐事实

以下规则来自 A5 `visa.txt` 的 load/store alignment table。令 `V` 为实际物理
vector byte size：

| Family | Dist | Required alignment | Full-active semantic footprint |
|---|---|---:|---:|
| Load | NORM | 32B | `V` |
| Load | BRC_B8/B16/B32 | 1/2/4B | 1/2/4B |
| Load | E2B_B16 | `V / 16` | `V / 16` |
| Load | E2B_B32 | `V / 8` | `V / 8` |
| Load | UNPK_B8/B16/B32 | `min(32, V / 2)` | `V / 2` |
| Load | UNPK4 | `min(32, V / 4)` | `V / 4` |
| LoadX2 | DINTLV_B8/B16/B32 | 32B | `2 * V` |
| Store | NORM_B8/B16/B32 | 32B | up to `V`, controlled by predicate |
| Store | 1PT_B8/B16/B32 | 1/2/4B | 1/2/4B |
| Store | PK_B16/B32/B64 | `min(32, V / 2)` | up to `V / 2` |
| Store | PK4_B32 | `min(32, V / 4)` | up to `V / 4` |
| StoreX2 | INTLV_B8/B16/B32 | 32B | `2 * V` |

`US/DS/BRC_BLK/BDINTLV/SPLT/MRG` 也必须进入共享 registry，避免 verifier 与 emitter
继续漂移；上表只列本次 VMI memory legalization 首期会迁移的 mode。

对典型 `V = 256B`：

```text
E2B_B16 alignment = 16B
E2B_B32 alignment = 32B
UNPK alignment    = 32B
UNPK4 alignment   = 32B
PK alignment      = 32B
PK4 alignment     = 32B
```

BRC 必须保留两个不同事实：

```text
BRC_B32 address validity = 4B alignment
BRC hardware dependency  = containing 32B block
```

因此 4B 对齐但非 32B 对齐的 BRC_B32 仍是合法 direct BRC，不能转成 `vldus`。

### 5.3 Token 与 target encoding

共享层负责：

- token 到 typed dist 的解析；
- dist 是否支持给定 op family 和访问 element width；
- 对齐、footprint、predicate 和 mapping 查询。
- 在显式选择 A5 profile 时，提供 typed dist 到 A5 immediate 的唯一映射。

这里的访问 element width 不是无条件等同于 `vreg` carrier width。尤其 store 的
`NORM_B32`、`PK_B64` 等显式 token 描述物理访问或 lane compact 语义；合法 producer
可以使用更宽的 carrier type。只有省略 store dist、需要选择默认 NORM token 时，
才根据 carrier width 推导 `NORM_B8/B16/B32`。verifier 和 emitter 必须遵守同一边界。

A5 LLVM emitter 不再维护独立的 token parser 或编码 switch，而是显式查询 A5
profile contract 中的 immediate。这样 verifier、lowering 和两个 emitter 使用同一条
registry 记录，同时避免把 A5 immediate 误当成跨 target 的通用语义。

本文 contract 的 alignment 数值是 A5 profile 的事实。typed dist、token 和 observable
mapping 可以共享；未来若其他 target profile 的 alignment 或支持集不同，查询必须显式
选择 profile，不能静默复用 A5 数值。

## 6. Physical Access Plan 数据模型

### 6.1 地址和访问覆盖

概念结构如下，具体 C++ 命名可在实现时按现有风格调整：

```cpp
struct VMIPlannedAddress {
  Value base;
  Value elementOffset;
  Type elementType;
};

enum class VMIMemoryCoverageKind {
  Dense,
  Prefix,
  Predicate,
};

struct VMIMemoryCoverage {
  VMIMemoryCoverageKind kind;
  OpFoldResult elementCount;
  Value predicate;
};

struct VMIPhysicalMemorySegment {
  VMIPlannedAddress address;
  VMIMemoryCoverage coverage;
  VMIRegisterTransfer transfer;
  VMIReadSafety readSafety;
};

struct VMIPhysicalAccessPlan {
  VPTOMemoryDirection direction;
  SmallVector<VMIPhysicalMemorySegment> segments;
};
```

当前 `VMIToVPTO.cpp` 已有一套较窄的 `VMIMemoryAccessPlan`、
`VMIMemoryLaneAddressMap` 和 `VMIMemorySafeReadProof`，主要服务 identity layout 和
static memref full-read 判断。实现时应演进并替换这套结构，而不是在旁边长期保留第二套
同义 plan。旧结构中的 `fallbackDecision` 只表示某个局部 lowering 当前是否缺少实现，
不是本文所需的 legalization 语义，也不应扩展成 dist fallback 枚举。

地址继续保留 typed base、element offset 和 element type。plan normalizer 通过
`PTOAddressAnalysis` 将差值换算成 bytes；不要提前把所有地址改写成无类型的整数
byte arithmetic。

覆盖含义：

- `Dense`：segment 中每个元素都必须访问；
- `Prefix`：只访问从起始地址开始的连续前缀，长度可以是静态或动态值；
- `Predicate`：任意 predicate 决定哪些固定位置被访问，可能存在洞。

稀疏 group store 通常表示为多个单元素 Dense segment，而不是把 stride 隐藏在
register mapping 中。

### 6.2 Register transfer 是语义映射

register transfer 描述“内存序号 k 对应哪个物理 value/lane”，而不是“调用哪个
fallback helper”。建议用带参数的代数数据类型或 `std::variant` 表达：

```cpp
using VMIRegisterTransfer = std::variant<
    IdentityTransfer,
    LaneExpandTransfer,
    LowBitsCompactTransfer,
    GroupRepeatTransfer,
    DeinterleaveTransfer,
    InterleaveTransfer,
    LaneSelectionTransfer>;
```

示例：

```text
UNPK load
  memory: one compact Dense segment
  transfer: LaneExpand{factor = 2 or 4}

PK store
  memory: one compact Dense/Prefix segment
  transfer: LowBitsCompact{factor = 2 or 4}

E2B load
  memory: compact group scalars
  transfer: GroupRepeat{groups, lanesPerGroup}

DINTLV load
  memory: two-vector-width Dense segment
  transfer: Deinterleave{factor = 2}

unit-stride slots=1 group store
  memory: one Dense segment
  transfer: LaneSelection{one source carrier lane per memory ordinal}

sparse slots=1 group store
  memory: multiple one-element Dense segments
  transfer: one selected lane for each segment
```

该模型的约束是：对相同 plan，direct dist 和显式重排后的 fallback 必须实现同一个
transfer。测试应验证 mapping 本身，而不只检查最后出现了哪个 dist token。

### 6.3 Load safety

load plan 必须记录相对有效地址可以证明安全读取的区间，而不只记录 VMI 语义所需
长度：

```cpp
struct VMIByteInterval {
  int64_t begin;
  int64_t end;
};

struct VMIReadSafety {
  std::optional<VMIByteInterval> provenReadableEnvelope;
};
```

这里 `[begin, end)` 相对 semantic effective address 表示，允许 `begin < 0`。具体
实现也可以复用现有 safe-read 判定并按 candidate 查询，而不是把区间存入 plan。
必要的不变量是：

```text
candidate physical read envelope <= proven readable envelope
```

如果没有额外证明，默认只允许读取 semantic footprint。

这是 E2B/UNPK fallback 的关键：

- E2B_B16 direct load 对 `V=256B` 只读取 16B；
- E2B_B32 direct load 只读取 32B；
- UNPK direct load 只读取 `V/2`；
- `vldus` 会形成一个完整 `V` 大小的 load result；
- `vldas` 还会读取包含起始地址的 32B 对齐块。

因此 stateful load candidate 必须检查 `vldas + vldus` 的联合 physical read
envelope，而不只是检查 `[address, address + V)`。对于起始地址 32B remainder 未知的
保守模型，envelope 可能延伸到 semantic address 之前最多 31B，并延伸到完整结果所需
范围之后。只有该联合范围被证明安全时，`vldus + explicit mapping` 才可用。否则
legalizer 必须选择精确的自然对齐访问组合，例如多个 BRC/point load，或在当前目标
没有合理精确路径时给出明确诊断。不能为了优化而静默 overread。

现有 `VMIMemorySafeReadProof` 只证明 static memref 中从 constant offset 开始的 identity
full-vector 范围，尚未覆盖起始地址之前的 aligned-block read，也不能表达 direct
E2B/UNPK 与 stateful candidate 的不同 envelope。它必须在接入 load fallback 前扩展。
此外，physical lane count 可计算只说明结果寄存器形状已知，不构成 safe-read 证明；
proof 失败时不能因为 `lanesPerPart` 可计算而继续选择会扩大读取的 candidate。

store 不允许超出 semantic footprint 写入。`vstus` 的 byte count 由 advance/size
operand 控制，它只存 source vector 的低位连续前缀，因此适合 Dense 或 Prefix store，
不适合任意 Predicate store。

## 7. Plan 规范化与 Point 合并

### 7.1 连续性证明

两个相邻 segment 只有同时满足以下条件才可合并：

1. 地址来自可证明相同的 pointer root/provenance；
2. `next - current` 的 byte difference 可证明等于 current semantic span；
3. memory order 与 register transfer order 一致；
4. 两段覆盖均为 Dense，或可组合成一个无洞 Prefix；
5. 合并后 payload 不超过候选指令和 stateful access 的容量；
6. 不需要重排或跨越其他 memory effect。

证明通过现有 `PTOAddressAnalysis`/`PTOValueEvolutionAnalysis` 完成。ISA dist 规则不进入
地址分析；地址分析只回答 root、byte difference、alignment 和 loop delta 等事实。

### 7.2 1PT_B32 示例

可合并：

```text
base + 0  : one b32 from carrier0 lane0
base + 4  : one b32 from carrier1 lane0
base + 8  : one b32 from carrier2 lane0
base + 12 : one b32 from carrier3 lane0
```

normalizer 得到一个 16B Dense segment，legalizer 先按 LaneSelection 形成一个 vreg 低
16B payload：

```text
32B-aligned and legal masked-prefix store -> one NORM store
otherwise                            -> one 16B vstus stream
```

不可合并：

```text
base + 0  : one b32
base + 32 : one b32
base + 64 : one b32
base + 96 : one b32
```

这些地址不连续，应保留自然 4B 对齐的 1PT_B32。是否地址本身能证明 32B 对齐与是否
连续是两个独立问题。

## 8. Legalization 算法

### 8.1 候选选择顺序

对规范化后的 segment，legalizer 按以下顺序处理：

```text
1. 根据 direction、transfer、physical types 和 coverage 查询 direct dist。
2. 检查 direct dist 的 element type、arity、predicate policy 和 footprint。
3. 查询 required alignment，并用 PTOAddressAnalysis 证明有效地址对齐。
4. direct candidate 全部合法时，发射 direct dist。
5. 否则枚举语义等价的通用实现：
     load  = exact/safe continuous load + explicit register transform
     store = explicit register transform + exact continuous store
6. 对通用实现检查 read envelope、write footprint、coverage 和容量。
7. 相邻连续实现共享同一条 stateful align/base chain。
8. 只有真正稀疏的自然对齐访问保留 point form。
9. 无合法实现时，诊断应包含失败的对齐、footprint 或 safety 条件。
```

direct dist 是合法化候选，不是 plan 中预先指定的结果。可使用 cost/preference 保持
当前目标上的 direct dist 优先级，但成本不能绕过语义和安全检查。

### 8.2 各类访问

#### NORM

```text
known 32B alignment -> direct NORM
otherwise           -> stateful continuous access
```

load stateful 路径仍需满足完整 VL read 的安全条件。store 可以用 `vstus` 精确写入
Dense/Prefix 的低位连续 payload。

#### UNPK

```text
required alignment proven
  -> direct UNPK

alignment not proven and stateful read envelope proven safe
  -> vldus full vector
  -> explicit lane expansion using only semantic input portion

otherwise
  -> exact naturally aligned load sequence if available
  -> or diagnostic
```

不能把 direct UNPK 的 `V/2` read 无条件替换成 `V` read。

#### PK

```text
required alignment proven and predicate semantics match
  -> direct PK

Dense/Prefix coverage
  -> explicit low-bit compact materialization
  -> vstus with exact compact byte count

arbitrary Predicate coverage
  -> keep predicate-aware direct/normal path
  -> do not use vstus unless values and destination addresses can both be
     legally compacted without changing the VMI memory mapping
```

#### E2B

```text
E2B alignment proven
  -> direct E2B

alignment not proven and stateful read envelope proven safe
  -> vldus
  -> explicit group-repeat materialization

stateful read envelope not safe
  -> exact scalar/BRC load sequence plus group-repeat materialization
  -> or diagnostic if the exact path is unsupported/unprofitable by policy
```

E2B_B8 不存在，不能从 element width 推导出该 token。

#### BRC

```text
b8/b16/b32 natural alignment proven
  -> direct BRC
```

BRC 不要求 32B 地址对齐。dependency analyzer 必须按 containing 32B block 建模硬件
访问冲突范围；共享 contract 为该 consumer 提供 granularity 事实。

#### DINTLV

```text
known 32B alignment -> direct DINTLV x2
otherwise:
  vldas
  vldus packet0, advance V
  vldus packet1, advance V
  explicit deinterleave
```

两个 `vldus` 各自读取 V，合计与 direct DINTLV 的 `2*V` semantic footprint 一致；
`vldas` 还会读取起始地址所在的 aligned block，因此仍需按 VMI load safety contract
检查整个 stateful candidate 的联合 physical read envelope。

#### INTLV

```text
known 32B alignment and full-store semantics
  -> direct INTLV x2
otherwise:
  explicit interleave into packet0 and packet1
  init_align
  vstus packet0, size V
  vstus packet1, size V
  vstas
```

ISA 中 INTLV predicate 被忽略，所以只有 VMI 语义要求完整写入两个 packet 时才能选择
direct INTLV。tail 或任意 predicate store 必须走能保持覆盖语义的路径。

### 8.3 Stateful stream 构造

plan 内 stateful store：

```text
init_align
vstus segment0, exact size0
vstus segment1, exact size1
...
vstas
```

`vstus` 的 size/advance 是 byte count 的硬件语义。VPTO 当前用 typed element unit
表达 offset 时，legalizer 必须验证 byte count 可被 pointer/value element byte size
整除，再创建对应 element count；不能混用 bytes 和 elements。

plan 内 stateful load：

```text
vldas base
vldus packet0, advance0
vldus packet1, advance1
...
```

每个 `vldus` 结果大小为 V，advance 只决定下一次 base/alignment state，并不把本次
物理读取缩短为 advance 大小。

## 9. 与 PTOAddressAnalysis 的边界

地址分析提供：

- `base + typed element offset` 的 N-byte alignment 证明；
- 同 root 地址之间的 known byte difference；
- loop iteration 的 address delta；
- cast、constant、affine add/sub/mul 和 no-wrap 所支持的 typed expression 证明。

memory legalizer 提供：

- 某个 dist 需要多少 alignment；
- segment semantic span 是多少；
- 哪些 segment 允许合并；
- 哪种指令和 register transform 可实现该 plan；
- 候选路径是否扩大 read/write envelope。

如果 plan normalizer 需要比较尚未物化为 VPTO op 的 `(base, elementOffset)`，应在
`PTOAddressAnalysis` 中增加基于同一 typed expression core 的地址构造/差值查询，
而不是在 `VMIToVPTO.cpp` 再写 constant-only 地址解析器。该 API 只回答地址事实，
不接受 dist 参数。

## 10. 实现组织

建议分阶段形成以下结构：

```text
include/PTO/IR/VPTOMemoryDist.h
lib/PTO/IR/VPTOMemoryDist.cpp
  shared dist parsing and semantic contract

lib/PTO/Transforms/VMIPhysicalMemoryAccess.h
  private plan/mapping data structures

lib/PTO/Transforms/VMIPhysicalMemoryAccess.cpp
  normalization, direct candidate matching, memory legalization

lib/PTO/Transforms/VMIToVPTO.cpp
  op-specific plan construction and non-memory physicalization

lib/PTO/Transforms/VPTOStatefulStreamFusion.cpp
  complete post-legalization stream fusion implementation
```

为降低首轮重构风险，plan structs 和第一批 legalizer helper 可以先作为
`VMIToVPTO.cpp` 的私有实现，并在第二个 memory family 接入后抽取到独立文件。
验收标准是多个 family 真正调用同一 legalizer，而不是为了文件数量提前抽象。

## 11. 迁移计划

### 阶段 A：共享事实，不改 lowering 行为

1. 增加 typed dist enum、parser 和 contract。
2. 将 VPTO verifier 的 dist token、element type、mask granularity 检查接入 contract。
3. 将 `VPTOCANN900LLVMEmitter` 和仍在构建范围内的 `VPTOLLVMEmitter` string parser
   改为 typed dist 后的 exhaustive encoding。
4. 审计 `VPTOOptimizeVcvt`、`PTOValidateVPTOIR` 等其余 dist string consumer。
5. 为所有现有 token 增加 contract 单元或 lit 覆盖。

### 阶段 B：建立 plan/legalizer 基线

1. 引入 address、coverage、read safety 和 register transfer 数据结构。
2. 将现有窄 `VMIMemoryAccessPlan` 演进到新结构，不保留平行 plan。
3. 修正 safe-read gating：proof unknown/failed 不得退化为“lane count 可计算即安全”。
4. 迁移普通 contiguous NORM load/store。
5. 保持现有 aligned/unaligned 输出不变，验证两阶段框架没有行为漂移。
6. 建立 plan normalization 的 same-root/known-difference 测试。

### 阶段 C：group store 与 point 合并

1. 迁移 unit-stride slots=1 和 compact group store。
2. 将 carrier lane selection 保留到 plan 中。
3. 合并连续 4 x 1PT_B32 为一个 16B Dense payload。
4. 保留非连续 point store。
5. 覆盖 aligned NORM、unaligned stateful 和 sparse 1PT 三种输出。

### 阶段 D：UNPK/PK

1. 迁移 lane-stride dense load/store。
2. 使用实际 V 计算 `min(32, V/2)` 和 `min(32, V/4)`。
3. 增加 UNPK full-VL safe-read 条件。
4. 增加 PK Prefix 与 arbitrary Predicate 的区分。

### 阶段 E：BRC/E2B

1. BRC 使用自然 1/2/4B alignment，单独覆盖 dependency granularity。
2. E2B 使用 `V/16` 或 `V/8` alignment。
3. direct E2B 不满足对齐时，按 read safety 选择 vldus 或 exact scalar/BRC 方案。
4. 保持 assigned physical layout 决定 E2B mapping，不在 lowering 中重新选择 layout。

### 阶段 F：DINTLV/INTLV

1. 迁移 x2 direct path。
2. 增加两个连续 packet 共享 align chain 的 fallback。
3. 验证 INTLV predicate ignored 条件。
4. 覆盖 multichunk 和 partial/tail rejection/fallback。

### 阶段 G：收口与文件边界

1. 审计 `VMIToVPTO.cpp` 中所有 `VldsOp`、`VstsOp`、`Vldsx2Op`、`Vstsx2Op`、
   `VldusOp` 和 `VstusOp` 创建点。
2. 未迁移创建点必须注明为何不属于 vector dist legalization。
3. 将 stateful fusion 实现完整迁移到独立 pass 文件。
4. 删除被统一 legalizer 替代的局部对齐和 stream helper。

## 12. 测试与性能验证

### 12.1 Contract 测试

必须覆盖：

```text
BRC_B32 at 4B but not 32B alignment -> legal direct BRC
E2B_B16 at 16B but not 32B          -> legal direct E2B
E2B_B32 at only 16B                 -> direct E2B illegal
UNPK/PK with non-default V          -> alignment derived from actual V
1PT_B32                             -> 4B natural alignment
DINTLV/INTLV                        -> 32B and arity 2
INTLV/1PT predicate                 -> ignored
```

### 12.2 Plan 测试

测试 planner/normalizer 的事实，而不只检查最终指令：

- unit-stride group store 形成一个 Dense segment；
- non-unit-stride group store 形成多个 point segment；
- 四个连续 b32 point 合并为 16B segment；
- 不同 root、unknown difference 或有洞的 point 不合并；
- tail 是 Prefix，不是 arbitrary Predicate；
- E2B semantic footprint 是 compact group scalar bytes；
- DINTLV/INTLV semantic footprint 是两个物理 packet。

若不增加 debug IR，可用 focused C++ test；若项目继续以 lit 为主，可增加仅用于测试的
plan printer，但生产 pipeline 不依赖该 printer。

### 12.3 End-to-end lit

每个 family 至少覆盖：

- 最小合法对齐；
- 低于最小对齐但自然对齐；
- unknown alignment；
- static UB pointer + aligned element stride；
- multi-packet stream；
- safe overread 与 unsafe overread；
- prefix mask 与 arbitrary predicate；
- loop iteration stream fusion；
- 无残留 VMI type/op 和合法 VPTO verifier/emitter 输出。

测试中非对齐不是默认噪声。若 case 的目的不是测试 non-alignment，应使用静态 UB base
和能证明所需 alignment 的 element-unit stride。

### 12.4 CA 性能验证

功能验证通过后，使用已验证的 CA model 流程比较：

```text
16B contiguous store payload:
  4 x 1PT_B32
  one independent stateful stream
  loop-carried fused stateful stream

dist fallback:
  aligned direct dist
  unaligned exact/safe fallback
```

load side 统一使用语义相同且安全的输入准备，避免把额外 `vldus` 或过量 1PT 混入被测
store 序列。报告 instruction count、busy/latency、循环迭代数和 steady-state 每迭代
开销，不能只给两列汇总而隐藏 warm-up/finalize 成本。

## 13. 正确性不变量

实现和 review 必须逐项满足：

1. **Footprint preservation**：store 不多写一个 byte；load 的扩大读取有明确 safe-read
   证明。
2. **Order preservation**：内存元素顺序和 register transfer 顺序一致。
3. **Coverage preservation**：Dense、Prefix、Predicate 不得互相偷换。
4. **Alignment legality**：每条 direct dist 使用自身 contract，不统一强制 32B，也不
   因 unknown alignment 直接假设合法。
5. **Predicate legality**：predicate ignored 的 dist 只能用于 VMI 语义本来就是 full
   access 的情况。
6. **Typed units**：bytes、pointer elements、vector elements 和 32B blocks 显式换算。
7. **Vector-size independence**：不硬编码 256B。
8. **State ownership**：每条 stateful 流的 align/base state 在 SSA 上线性传递并正确
   终止。
9. **Analysis boundary**：PTOAddressAnalysis 只证明地址事实，不决定指令或收益。
10. **Fusion boundary**：local legalizer 处理单个 plan，StatefulStreamFusion 处理跨
    plan/跨迭代融合。
11. **No recovered semantics**：生成 point op 之前完成可合并性判断；后续 pass 不依赖
    从低层指令猜回 carrier mapping。
12. **Diagnostic completeness**：没有合法路径时说明是 alignment、coverage、read
    envelope、unsupported mapping 还是 target support 导致失败。

## 14. 方案自检

### 14.1 一致性

- 与 VMI layout 边界一致：layout assignment 决定物理 lane mapping，memory planner
  只消费 assigned layout，不暗中重选 layout。
- 与 PTOAddressAnalysis 设计一致：复用 typed expression、pointer provenance 和 byte
  difference，不新增常量专用地址系统。
- 与 StatefulStreamFusion 边界一致：legalizer 创建合法流，fusion 只证明并合并流。
- 与 emitter 边界一致：共享通用 dist 语义；A5 immediate 只存在于显式 A5 profile
  contract 中，target emitter 负责选择并消费该 profile。
- 与当前 pipeline 一致：`vmi-to-vpto` 后继续运行 `vpto-stateful-stream-fusion`，不要求
  新增 pass 顺序。

### 14.2 ISA 准确性

- BRC 使用自然 1/2/4B 地址对齐，不误用 32B；同时保留 32B dependency granularity。
- E2B 只支持 B16/B32，alignment 分别为 `V/16`、`V/8`。
- UNPK/PK alignment 使用 `min(32, V/2)`，UNPK4/PK4 使用 `min(32, V/4)`。
- DINTLV/INTLV 使用两个 register/packet，地址要求 32B。
- 1PT_B32 表示一个 32-bit point，要求 4B，不是 32-byte store。
- `vstus` 写低位 size-byte 连续前缀并更新 base；`vldus` 返回完整 V 大小的数据，
  advance 不限制本次读取大小。
- INTLV 和 1PT 的 predicate ignored 行为进入 contract，不由 lowering 猜测。

### 14.3 语义正确性

- arbitrary masked store 不会被无 mask stateful store 替换。
- E2B/UNPK fallback 不会无证明 overread。
- stateful load safety 同时覆盖 `vldas` 的 containing block 和 `vldus` 的 vector read。
- group store 只有在地址连续且 lane selection 可按内存顺序拼成 payload 时才合并。
- sparse group store 即使含多个 1PT，也不会因“存在 1PT”而被错误视为连续。
- direct 和 fallback 使用相同 register transfer，避免 footprint 与结果 layout 脱节。

### 14.4 可实施性

- 第一阶段只集中现有 dist table，不要求重写 VMI lowering，可单独验证和回退。
- NORM 是最小 plan/legalizer 试点，已有 aligned/stateful 行为可作为等价基线。
- group store、UNPK/PK、E2B、x2 family 可逐个迁移，每一步都有独立 lit 边界。
- plan 首期为 pass 内 C++ 对象，不涉及 dialect、parser、serializer 或跨 pass lifetime。
- 现有窄 `VMIMemoryAccessPlan` 可作为演进起点，但其 identity/static safe-read proof
  不能直接作为完整 legalizer 的正确性依据。
- 现有 `isKnownAddressAligned`、known address difference 和 loop delta 能覆盖首批查询；
  只有 raw planned address pair 查询不足时才扩展同一 analysis framework。
- 最终审计所有 memory op creation site，避免出现新旧决策系统长期并存。

### 14.5 已知风险与控制

| Risk | Consequence | Control |
|---|---|---|
| Mapping model too generic | 变成无实际 consumer 的抽象层 | 每增加一种 mapping，至少两个实现候选或一个 direct/fallback 对必须消费它 |
| Planning emits register IR too early | 丢失 direct dist 机会 | transfer 保持 declarative，legalizer 决定何时 materialize |
| Stateful load overread | 越界或错误 memory effect | 显式 read-envelope check，unknown 默认为不允许扩大 |
| Existing safe-read proof is treated as complete | 忽略 `vldas` 前向块读取或 proof failure | 演进原结构并让 proof 结果直接 gate candidate selection |
| Predicate semantics drift | inactive address 被写入 | predicate policy 进入 shared contract，coverage 进入 plan |
| Address analysis duplication | 等价表达式结论不一致 | 所有新 query 基于 PTOValueEvolution/PTOAddressAnalysis typed expressions |
| One-shot large refactor | 门禁难以定位 | 按 A-G 阶段迁移，每阶段保持全量 `vmi_new` 通过 |
| String token drift remains | verifier/emitter 支持集不一致 | 审计所有 dist string consumer，typed parser 成为唯一入口 |

## 15. 完成标准

设计完成落地需同时满足：

1. 所有现有 vector memory dist 均有共享 contract，verifier 与 emitter 不再分别解析
   支持集。
2. NORM、group store、UNPK/PK、BRC/E2B、DINTLV/INTLV 均通过 plan/legalizer。
3. 每个 direct dist 在发射前完成 mode-specific alignment 和 predicate 检查。
4. 每个 stateful fallback 有 footprint/coverage/read-safety 证明。
5. 连续 point store 在 plan 阶段合并，真正稀疏访问保持 point form。
6. 单 plan 多 packet 使用一条 stateful stream，跨 plan/跨迭代由独立 fusion pass 合并。
7. `VMIToVPTO.cpp` 不再包含 dist-specific 地址判断和重复 align stream 管理分支。
8. 全量 `test/lit/vmi_new`、compliance checker、LLVM emission 和 CA 功能验证通过。
9. 性能报告覆盖真实 16B payload 的 1PT、独立 stateful 和迭代间 fused stateful 三组
   steady-state 对比。
