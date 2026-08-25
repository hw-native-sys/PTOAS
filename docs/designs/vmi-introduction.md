# VMI 介绍

本文介绍 VMI 的设计入口：VMI 解决什么问题，layout 有哪些，pass pipeline
如何分工，以及这些机制分别应对哪些典型场景。更完整的逐 case lowering 结果见
`docs/designs/vmi-layout-lowering-cases.md`。

示例是设计级 IR，保留关键 type、layout、helper op 和 VPTO op 形状，
省略 module wrapper、完整 operand list 和不影响讨论的 SSA 细节。

## 1. VMI 表达什么

VMI 是 VPTO 之前的逻辑向量层。它让前端先表达“我要对 `NxT` 的逻辑向量做什么”，
再由 layout assignment 决定这个逻辑向量如何拆到 256B 物理 vector register 上。
当 VPTO 指令因为物理 register 宽度只能暴露半宽接口时，VMI 也负责提供完整的
逻辑语义。例如 `ui8` histogram 的完整结果是 `256xui16`，物理 VPTO histogram
一次只能返回 `128xui16`；VMI surface 应该表达完整 histogram，low/high bin
range 拆分属于 lowering 细节。

Surface VMI 类型不携带布局：

```mlir
!pto.vmi.vreg<128xf32>
!pto.vmi.mask<128xpred>
```

Layout-assigned VMI 类型携带具体布局和 mask granularity：

```mlir
!pto.vmi.vreg<128xf32, #pto.vmi.layout<deinterleaved = 2>>
!pto.vmi.mask<128xb32, #pto.vmi.layout<deinterleaved = 2>>
```

VMI 的核心约束是：`vmi-to-vpto` 只从当前 op 的 attrs、operands、types、
layouts 和显式 helper ops 做 lowering，不读取隐藏 plan/recipe，也不通过
defining op 或 sibling user 恢复上下文。

## 2. Layout 类型

### 2.1 `contiguous`

```mlir
#pto.vmi.layout<contiguous>
```

含义：logical lane 按顺序落入物理 register list。

```text
logical lanes:  0  1  2 ... 63 | 64 65 ... 127
physical part:  p0             | p1
```

典型场景：

```text
dense load/store
普通 elementwise compute
一个 group 天然适配当前 reduce op 时的 reduction input
caller/callee 约定 dense order 时的 control-flow/function boundary
```

### 2.2 `deinterleaved = F` 与 `block_deinterleaved = F`

```mlir
#pto.vmi.layout<deinterleaved = 2>
#pto.vmi.layout<block_deinterleaved = 4>
```

`deinterleaved` 按 element 拆分：

```text
part p        = i % F
physical lane = i / F
```

`block_deinterleaved` 是独立 layout kind，按固定 32B VCG block 拆分。block
中的 element 数量由 element type 推导，不编码在 layout 中：

```text
f32/b32 -> 8 elements
f16/b16 -> 16 elements
f8/b8   -> 32 elements
```

令 `B = 32B / sizeof(element)`，其逻辑 lane 到物理 part 的映射是：

```text
logical lane i
block q         = i / B
in-block lane r = i % B
part p          = q % F
part block t    = q / F

physical part p, physical lane t * B + r
```

`deinterleaved=2` 的直观例子：

```text
logical lanes:   0 1 2 3 4 5 ...
physical part0:  0   2   4   ...
physical part1:    1   3   5 ...
```

`block_deinterleaved=4` 对 f32 的直观例子（此时 `B=8`）：

```text
logical group S=32:
  lanes  0.. 7 -> part0 lanes 0..7
  lanes  8..15 -> part1 lanes 0..7
  lanes 16..23 -> part2 lanes 0..7
  lanes 24..31 -> part3 lanes 0..7
```

典型场景：

```text
f16 -> f32:
  vcvt 天然产生 even/odd 两个 f32 part，所以结果使用 deinterleaved=2。

f32 -> f16:
  vcvt 需要 f32 source 先拆成 even/odd 两个 part，所以 source 使用
  deinterleaved=2。

S=32 group_reduce f32:
  一个 group 有 32 个 f32 element。高效 reduce path 消费四个 8-lane block，
  所以 source/mask 使用 block_deinterleaved=4。
```

`block_deinterleaved` 表示一种按 32B row fragment 组织的输入形态，不表示
S=32 reduce 只能接受这一种形态。如果同一个 value 还要服务 narrow cast 等
element-parity consumer，assignment 可以选择普通 `deinterleaved=4` 作为共同
layout，再由 lowering 生成对应的物理指令序列。

两种 layout 都只描述最终物理 part 中有哪些 logical lane，不描述这个 layout
由哪条指令生成。它们是不同的 layout kind，不能因为 factor 相同就视为兼容。
不同 producer 可以用不同方式直接产生同一个 layout；如果不能直接产生，后续
lowering 再通过显式 materialization helper 把 source layout 转换成 consumer
需要的 layout。具体 lowering 形状见 case catalog。

### 2.3 `num_groups = G, slots = K`

```mlir
#pto.vmi.layout<num_groups = 8, slots = 8>
#pto.vmi.layout<num_groups = 8, slots = 1>
#pto.vmi.layout<num_groups = 8, slots = 8, lane_stride = 4>
```

这是 group-slot result layout。它不表示全部 `N` 个 logical lane 都有语义值。
只有 `G` 个 group 结果 slot 有语义值。

```text
slot_block(g) = g / K
slot_lane(g)  = (g % K) * lane_stride

physical part slot_block(g) 的 lane slot_lane(g) 保存 group g 的结果
```

`lane_stride` 缺省为 1，单位是 logical element-sized physical slot。
它描述 group result 在物理存储中的固定间距，不改变 VMI 的逻辑元素类型。
例如 `ui8 lane_stride=4` 表示 group slot 存在 byte lane 0, 4, 8, ...
这种形态可以 lower 为 `PK4_B32` store，物理上使用 b32 carrier 的 low byte。

`num_groups=16, slots=8` 的例子：

```text
part0 lane0..7 = group result 0..7
part1 lane0..7 = group result 8..15
other lanes    = 对普通 dense consumer 来说未定义
```

为什么 group 信息也要放进 layout：

```text
group_reduce 自身有 num_groups，但它的结果可能继续跨过 truncf、
group_broadcast、group_store、scf.if、scf.for、function call 或多个 consumer。

这些后续 op 不应该回看 producer attr。value layout 因此需要记录有多少个
group result，以及这些 result 如何 packed 到 physical slot。
```

典型场景：

```text
group_reduce result
group_slot_load result
group_store input
group_broadcast input
group-slot control-flow/function boundary
部分 row-local cast 路径，通常使用 slots=1
```

## 3. Pass Pipeline

```text
pto-validate-vmi-ir
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
```

### 3.1 `pto-validate-vmi-ir`

检查 surface VMI 边界。

合法输入：

```mlir
%x = pto.vmi.load %src[%off]
  : !pto.ptr<f16, ub> -> !pto.vmi.vreg<128xf16>
```

非法输入：

```mlir
%x = pto.vmi.load %src[%off]
  : !pto.ptr<f16, ub>
    -> !pto.vmi.vreg<128xf16, #pto.vmi.layout<contiguous>>
```

原因：具体 layout 由 `vmi-layout-assignment` 产生，不应该由 surface frontend
提前写入。

### 3.2 `vmi-layout-assignment`

这是硬合法化 pass。它选择具体 value layout、具体 mask granularity，
并在 layout 不匹配的 use site 插入显式 helper op。

这个 pass 的工作顺序是固定的：

```text
1. 做少量 VMI 内部规整，让后续 layout 规则面对稳定形态。
2. 为 data value 建 union-find 求解器，并收集 data 约束和 data use request。
3. 把可采纳的 consumer request 提升为 producer/result 的最终 layout。
4. 改写所有 data value type，让 !pto.vmi.vreg 携带具体 layout。
5. 对仍不匹配的 data use 插入 pto.vmi.ensure_layout。
6. 基于已经确定的 data layout 推导 mask layout 和 predicate granularity。
7. 改写所有 mask type，并对不匹配的 mask use 插入 ensure_mask_*。
8. 同步更新 function type、call boundary 和 block argument type。
9. 校验 layout-assigned VMI IR。
```

Data 和 mask 分两轮求解。原因是 mask layout 通常依赖对应 data operand 或 result
的 layout；例如 `cmpf` 产生的 mask 跟比较输入的 data layout 对齐，
`select`/`reduce`/`masked_load` 消费的 mask 也要跟对应 data value 的 lane
layout 和元素 bitwidth 对齐。

Data 求解器为每个 `!pto.vmi.vreg` 建一个节点：

```text
DataNode:
  value         = 对应 SSA value
  original type = surface VMI type
  parent        = union-find parent
  naturalLayout = 当前等价类选择的自然 layout，可能为空
```

遍历 IR 时，每个 op 向 data 求解器贡献三类信息。

第一类是 layout 等价约束。它表示几个 value 必须使用同一个 physical layout，
也就是 union-find 中的同一个等价类。典型来源：

```text
layout-transparent elementwise:
  addf/addi/subf/subi/mulf/muli/fma/divf/minf/maxf/...
  L(operands...) = L(result)

unary elementwise:
  negf/absf/absi/sqrt/exp/ln/relu/not
  L(source) = L(result)

select:
  L(true_value) = L(false_value) = L(result)

bitcast:
  L(source) = L(result)

structured control flow:
  scf.if result     = then/else yield operand
  scf.for result    = init operand = iter_arg = yield operand
  scf.while result  = init/before/condition/after/yield carried value

cf branch:
  branch operand = destination block argument

function boundary:
  call operand = callee argument
  call result  = callee return operand
  multiple returns of the same function agree per result index
```

这一步只说明“这些 value 如果存在布局，就必须一致”。它不等价于把某个
consumer 的 request 无条件推过所有 producer 或控制流。

等价类可以画成“同一个框里的 value 共用一个 layout 变量”。例如普通
elementwise 链：

```text
surface VMI:

  %x = pto.vmi.load ...
  %k = pto.vmi.broadcast ...
  %y = pto.vmi.mulf %x, %k
  %q = pto.vmi.truncf %y

data layout 等价类:

  class C0
  +--------------------------------------+
  | %x        %k        %y               |
  | load      broadcast mulf result      |
  +--------------------------------------+
                         ^
                         |
              use request from truncf source:
              wants deinterleaved=4

若 %y 的 producer chain 可采纳该 request，assignment 可以选择:

  L(C0) = deinterleaved=4
```

控制流 join 也是等价类，但 request adoption 的含义不同：

```text
surface VMI:

  %y = scf.if %c -> !pto.vmi.vreg<128xf32> {
    scf.yield %a
  } else {
    scf.yield %b
  }
  %q = pto.vmi.truncf %y

data layout 等价类:

  class C1
  +--------------------------------------+
  | %a        %b        %y               |
  | then yield else yield if result      |
  +--------------------------------------+
                         ^
                         |
              use request from truncf source:
              wants deinterleaved=4

scf.if result 不是 consumer-driven adoption 的可采纳 producer。
若 C1 不能直接选择 deinterleaved=4，assignment 保持 C1 的布局，
并在 use site materialize:

  %y_for_q = pto.vmi.ensure_layout %y : L(C1) -> deinterleaved=4
  %q = pto.vmi.truncf %y_for_q
```

多 consumer 冲突时，等价类仍然只有一个 layout：

```text
surface VMI:

  %y = pto.vmi.mulf %x, %k
  pto.vmi.store %y, %out0
  %q = pto.vmi.truncf %y

data layout 等价类:

  class C2
  +-----------------------------+
  | %x        %k        %y      |
  +-----------------------------+
                         |\
                         | \ use request from truncf: deinterleaved=4
                         |
                         +--- use request from store: contiguous

两个 use request 不一致时，不能让 %y 同时拥有两个 layout。
baseline assignment 保留 C2 已有的 natural layout；若没有 natural layout，
则使用默认 contiguous。与该 layout 不匹配的 edge 会插 ensure_layout。
```

第二类是 result 自然布局。某些 op 的结果本身有目标相关的自然布局：

```text
普通 reduce / compress / shuffle:
  result 通常是 contiguous。

group_reduce:
  source 需要适配 group reduce 指令形态；
  result 使用 group_slots(num_groups, slots) 描述 group-slot result。

cast:
  widening/narrowing 根据 cast support 决定 source request 和 result layout。

group_load / group_slot_load / group_broadcast_load:
  result 根据 group size、row stride 和目标能力选择 contiguous、deinterleaved
  或 group_slots。group_broadcast_load 表达“每个 logical group load 一个值并
  广播到组内 lanes”的逻辑语义；E2B 只是兼容 layout 下的一种 lowering。

stride_load:
  result 是 contiguous。block/repeat stride 只描述 memory address map，
  不改变 register 内 logical lane order。

active_prefix_index:
  result 使用 contiguous。
```

若同一个等价类已经有自然布局，再设置不同自然布局会报 layout contract 冲突。

第三类是 operand 使用请求。consumer 不直接修改 operand 的 type，而是记录
“这个 use site 希望 operand 是什么 layout”：

```text
store / masked_store value:
  wants contiguous

ordinary reduce source/init:
  wants contiguous

group_reduce source:
  wants preferred group-reduce source layout

group_store value:
  wants preferred group result layout

stride_store value:
  wants contiguous。block/repeat stride 只描述 memory write address map，
  不表示 source vreg 是 lane-strided 或 NZ layout。

truncf/trunci/extf/extsi/extui source:
  wants cast support 给出的 source layout

channel_split / channel_merge / shuffle:
  wants 各自 lowering 需要的 source/input layout
```

收集完这些信息后，assignment 才尝试做 consumer-driven adoption。它逐个查看
use request：如果 operand 的 producer 可以直接用 consumer 需要的 layout 产生
同一个逻辑向量，并且多 use 时所有 use 都请求同一个 layout，那么这个 request
会被提升为该 value 所在 data 等价类的最终 layout。

可采纳 producer 是受限集合：

```text
load
broadcast / constant / iota
layout-transparent elementwise
select
bitcast
```

这就是 request 看起来能穿过 elemwise 的原因：

```mlir
%x = pto.vmi.load ...
%k = pto.vmi.broadcast ...
%y = pto.vmi.mulf %x, %k
%q = pto.vmi.truncf %y
```

`mulf` 先把 `%x`、`%k`、`%y` 合成同一个 data 等价类。`truncf` 对 `%y`
的 source use 请求 `deinterleaved=4` 时，这个 request 作用到 `%y` 所在等价类；
因为 `mulf` 是可采纳 producer，assignment 可以把整个等价类选成
`deinterleaved=4`，从而让 load/broadcast/mulf 直接在这个 layout 下产生数据。

控制流边界也会形成等价类，但它不是任意 request 的自动传播通道：

```mlir
%y = scf.if %c -> !pto.vmi.vreg<128xf32> {
  scf.yield %a
} else {
  scf.yield %b
}
%q = pto.vmi.truncf %y
```

`%y`、`%a`、`%b` 的 layout 必须一致；但 `scf.if` result 本身不是
consumer-driven adoption 的可采纳 producer。若 `%q` 需要的 layout 无法成为
这个等价类的最终布局，assignment 会在 `%q` 的 use site 插
`pto.vmi.ensure_layout`，而不是隐式重写两个 branch 的内部计算。

Data layout 确定后，pass 会把每个 `!pto.vmi.vreg<NxT>` 改写成
`!pto.vmi.vreg<NxT, layout>`。如果某个记录过的 use request 仍然和 operand
当前 layout 不一致，pass 在该 consumer 前插显式 materialization：

```mlir
%x_req = pto.vmi.ensure_layout %x
  : !pto.vmi.vreg<NxT, source_layout>
    -> !pto.vmi.vreg<NxT, requested_layout>
consumer %x_req
```

这个规则也处理多 consumer 冲突：

```mlir
%y = pto.vmi.mulf %x, %k
pto.vmi.store %y, %out0      // wants contiguous
%q = pto.vmi.truncf %y       // wants deinterleaved=4 source
```

一个 SSA value 只能属于一个 data layout 等价类。若两个 use 不能共同满足，
baseline assignment 保留一个等价类 layout，并在不匹配 use 前插
`ensure_layout`。后续 `vmi-layout-fold`、`vmi-layout-rematerialize`
和 `vmi-layout-sink-materialization` 可以在显式 helper op 上做优化，但
`vmi-to-vpto` 不读取隐藏 plan 或 sibling user。

Mask 求解发生在 data type 改写之后。它同样维护 union-find 等价类，但节点记录
两件事：

```text
mask layout
predicate granularity: b8 / b16 / b32
```

mask request 从已经带 layout 的 data value 推导：

```text
cmpf/cmpi result:
  mask layout = lhs data layout
  granularity = lhs element bitwidth 对应的 predicate 粒度

select mask:
  mask layout = result data layout
  granularity = result element bitwidth 对应的 predicate 粒度

reduce / group_reduce / masked_load / expand_load mask:
  mask layout = source/result data layout
  granularity = 对应 data element bitwidth 的 predicate 粒度
```

若 mask use 的 layout 或 granularity 不匹配，pass 显式插
`pto.vmi.ensure_mask_layout` 或 `pto.vmi.ensure_mask_granularity`。

完成 data/mask 改写和 helper 插入后，pass 会同步更新 function type。直接
internal call 会把 call operand/result 与 callee argument/return operand 合成
同一布局约束；带 VMI type 的 external declaration 或 indirect call 没有可见
body，当前需要显式 ABI materialization 设计，因此 layout assignment 会拒绝。
这个阶段之后，IR 不再依赖隐藏 plan；后续 pass 和 `vmi-to-vpto` 都只读取 type
上的 layout 和显式 `ensure_*` helper。

### 3.3 `vmi-layout-fold`

当 consumer 可以直接保持同样的外部效果时，把显式 materialization 折进
consumer。

变换前：

```mlir
%dense = pto.vmi.ensure_layout %x
  : !pto.vmi.vreg<128xf32, #pto.vmi.layout<deinterleaved = 2>>
    -> !pto.vmi.vreg<128xf32, #pto.vmi.layout<contiguous>>
pto.vmi.store %dense, %dst[%off]
```

变换后：

```mlir
pto.vmi.store %x, %dst[%off]
  : !pto.vmi.vreg<128xf32, #pto.vmi.layout<deinterleaved = 2>>, !pto.ptr<f32, ub>
```

可能的 VPTO 形状：

```text
fold 前：vintlv + vsts + vsts
fold 后：vstsx2，使用交错 store mode
```

### 3.4 `vmi-layout-rematerialize`

通过 clone 低成本、layout-polymorphic 的 producer 来替换 `ensure_*`。

变换前：

```mlir
%s = pto.vmi.broadcast %scale
  : f32 -> !pto.vmi.vreg<128xf32, #pto.vmi.layout<contiguous>>
%s_split = pto.vmi.ensure_layout %s
  : !pto.vmi.vreg<128xf32, #pto.vmi.layout<contiguous>>
    -> !pto.vmi.vreg<128xf32, #pto.vmi.layout<deinterleaved = 2>>
```

变换后：

```mlir
%s_split = pto.vmi.broadcast %scale
  : f32 -> !pto.vmi.vreg<128xf32, #pto.vmi.layout<deinterleaved = 2>>
```

预期可 rematerialize 的 producer：

```text
splat constant
broadcast
iota
create_mask
create_group_mask
constant_mask
```

这个 pass 不 rematerialize：

```text
load / masked_load / group_load / group_slot_load / group_broadcast_load
stride_load
reduce / group_reduce
control-flow results
```

### 3.5 `vmi-layout-sink-materialization`

把匹配的 layout 转换跨过 layout-transparent elementwise op。

变换前：

```mlir
%a_dense = pto.vmi.ensure_layout %a : deinterleaved=2 -> contiguous
%b_dense = pto.vmi.ensure_layout %b : deinterleaved=2 -> contiguous
%y_dense = pto.vmi.addf %a_dense, %b_dense : contiguous
```

变换后：

```mlir
%y_split = pto.vmi.addf %a, %b : deinterleaved=2
%y_dense = pto.vmi.ensure_layout %y_split : deinterleaved=2 -> contiguous
```

效果：

```text
两个 input materialization -> 一个 result materialization
```

这个 pass 不会 sink 穿过 cast、load、store、reduce、group_broadcast 或
control-flow op。

### 3.6 `vmi-legalize-arith-select`

Canonicalization 可能把简单的 `scf.if` 折成 `arith.select`。VMI 希望把
control-flow lowering 保持在结构化控制流里，所以这个 pass 会把 VMI value 上的
`arith.select` 改回 `scf.if`。

```mlir
%r = arith.select %cond, %a, %b
  : !pto.vmi.vreg<128xf32, #pto.vmi.layout<contiguous>>
```

改成：

```mlir
%r = scf.if %cond
    -> !pto.vmi.vreg<128xf32, #pto.vmi.layout<contiguous>> {
  scf.yield %a : !pto.vmi.vreg<128xf32, #pto.vmi.layout<contiguous>>
} else {
  scf.yield %b : !pto.vmi.vreg<128xf32, #pto.vmi.layout<contiguous>>
}
```

### 3.7 `pto-validate-vmi-layout-ir`

检查 post-assignment gate：

```text
每个 VMI 数据值都有 concrete layout
每个 VMI mask 都有 concrete granularity 和 layout
helper op 有支持的 materialization path
semantic op/layout 组合有支持的 local lowering
vmi-to-vpto 之前没有物理 VPTO value 泄漏到 VMI IR 中
```

非法例子：

```mlir
%sum = pto.vmi.group_reduce_addf %x, %mask {num_groups = 8, reassoc}
  : ... -> !pto.vmi.vreg<8xf32, #pto.vmi.layout<num_groups = 8, slots = 8>>

pto.vmi.store %sum, %dst[%off]
  : !pto.vmi.vreg<8xf32, #pto.vmi.layout<num_groups = 8, slots = 8>>,
    !pto.ptr<f32, ub>
```

原因：

```text
dense store 不能把 group_slots 当 dense vector 读取。
应使用 group_store、group_broadcast 或显式支持的 group-to-dense op。
```

### 3.8 `vmi-to-vpto`

把 layout-assigned VMI value 转换成有序物理 VPTO value 列表，并对每个
VMI op 做 local lowering。

例子：

```text
!pto.vmi.vreg<128xf32, #pto.vmi.layout<contiguous>>
  -> 两个 physical !pto.vreg<64xf32> part

!pto.vmi.vreg<128xf32, #pto.vmi.layout<deinterleaved = 2>>
  -> 两个 physical !pto.vreg<64xf32> part
     part0 携带 even lanes，part1 携带 odd lanes

!pto.vmi.vreg<32xf32, #pto.vmi.layout<num_groups = 32, slots = 8>>
  -> 四个 physical part
     part0 携带 group 0..7，part1 携带 group 8..15，...
```

`VMILayoutSupport` 不是 pass。它是 assignment、validation、optimization 和
lowering 共享的查询库，用来避免重复实现 layout fact 和 supported
materialization 检查。

## 4. 典型场景

### 4.1 Dense Cast 与 Store

```text
surface:
  load f16，语义上连续
  extf 到 f32
  dense store f32

assignment:
  load result      = contiguous
  extf result      = deinterleaved=2
  store use        = ensure_layout(deinterleaved=2 -> contiguous)

baseline VPTO:
  vlds
  vcvt even / vcvt odd
  vintlv
  vsts + vsts

fold-consumers 后的优化 VPTO:
  vlds
  vcvt even / vcvt odd
  vstsx2，使用 interleaving store
```

这个场景说明为什么需要 `deinterleaved=2`，以及为什么 store-consumer folding
有价值。

### 4.2 Narrow Cast 与 Store

```text
surface:
  load f32
  truncf 到 f16
  dense store f16

assignment:
  load result = deinterleaved=2
  truncf result = contiguous

VPTO:
  vldsx2 deinterleaving load
  vcvt even / vcvt odd
  vor
  vsts
```

这个场景说明 memory op 可以直接产生 consumer 需要的 layout，但不需要保存隐藏
plan。

### 4.3 一个 Producer 同时服务 Dense 和 Group Consumer

```mlir
%x32 = pto.vmi.extf %x16
%sum = pto.vmi.group_reduce_addf %x32, %mask {num_groups = 8, reassoc}
pto.vmi.group_store %sum, %sum_out[%off], %c1 {num_groups = 8}
pto.vmi.store %x32, %dense_out[%off]
```

Assignment 形状：

```text
%x32 layout = deinterleaved=2
group_reduce 直接消费 %x32
dense store 获得 ensure_layout(%x32 -> contiguous)
```

VPTO 形状：

```text
vcvt even/odd
vcgadd + vcgadd + vadd -> group_store result
vintlv + dense stores  -> 产生 dense store 结果
```

这个场景说明为什么需要 use-site materialization。producer 不需要选择一个能同时
满足所有 consumer 的唯一 layout。

### 4.4 按 Group Size 区分的 Group Reduce

对于 `N` 个 f32 lane 和 `G = num_groups`，group size 是 `S = N / G`。

```text
S=8:
  input layout 可以是 contiguous。
  group_reduce result 通常使用 layout<num_groups=G, slots=8>。

S=16:
  如果 input 来自 f16->f32 vcvt，layout 可以是 deinterleaved=2。
  如果 input 从 dense 拆出，layout 可以是 block_deinterleaved=2。
  result 通常使用 layout<num_groups=G, slots=8>。

S=32:
  input layout 使用 block_deinterleaved=4。
  VPTO 形状是四个部分 group reduction 后接 add tree。
  result 通常使用 layout<num_groups=G, slots=8>。

S=64:
  row-local path 在可行时让每个 group 使用一条 physical row。
  result 可以使用 layout<num_groups=G, slots=1>，避免 unsupported packing。
```

S=32 例子：

```text
assignment:
  source/mask = block_deinterleaved=4
  result      = group_slots(num_groups=8, slots=8)

VPTO:
  vdintlv / pdintlv_b32
  vcgadd x4
  使用 PAT_VL8 做 vadd tree
  通过一次 PAT_VL8 store 完成 group_store
```

这个场景说明为什么 `block_deinterleaved` 必须是独立的 layout kind。

### 4.5 Group Result 继续作为 Dense Rows 使用

Surface 意图：

```mlir
%sum32 = pto.vmi.group_reduce_addf %x, %mask {num_groups = 8, reassoc}
%rows32 = pto.vmi.group_broadcast %sum32 {num_groups = 8}
%rows16 = pto.vmi.truncf %rows32
pto.vmi.store %rows16, %dst[%off]
```

支持的 assignment 形状：

```mlir
%sum32 = pto.vmi.group_reduce_addf ...
  -> !pto.vmi.vreg<8xf32, #pto.vmi.layout<num_groups = 8, slots = 8>>

%rows32 = pto.vmi.group_broadcast %sum32 {num_groups = 8}
  -> !pto.vmi.vreg<128xf32, #pto.vmi.layout<contiguous>>

%rows32_split = pto.vmi.ensure_layout %rows32
  : contiguous -> deinterleaved=2

%rows16 = pto.vmi.truncf %rows32_split
  : deinterleaved=2 -> contiguous

pto.vmi.store %rows16, %dst[%off]
```

VPTO 形状：

```text
group_reduce:
  vcgadd partials + vadd tree

group_broadcast:
  vselr 风格 selection，把 group slots 展开到 dense row lanes

truncf:
  vcvt even/odd + merge

store:
  vsts
```

这个场景说明为什么 group 结果 layout 必须挂在 value 上：reduce 之后，
cast 和 broadcast 必须知道 group 结果在哪里，而不能回看 producer。

### 4.6 通过 Mask 表达 Tail

VMI 通过 mask 表达 tail，不通过 padding 表达 tail。

```mlir
%mask = pto.vmi.create_mask %active_lanes
%x = pto.vmi.masked_load %src[%off], %mask
%y = pto.vmi.mulf %x, %scale
pto.vmi.masked_store %y, %dst[%off], %mask
```

Grouped tail：

```mlir
%gmask = pto.vmi.create_group_mask %active_elems_per_group
    {num_groups = 8, group_size = 32}
%sum = pto.vmi.group_reduce_addf %x, %gmask {num_groups = 8, reassoc}
```

同一个 semantic mask 面对 f8/f16/f32 user 时，可能需要不同 concrete
granularity。Assignment 会通过 mask helper op 显式表达这些转换。

### 4.7 控制流和函数边界

Concrete layout 必须显式跨过 CFG 和内部 function boundary。

```mlir
%r = scf.if %cond
    -> !pto.vmi.vreg<128xf32, #pto.vmi.layout<contiguous>> {
  %a_dense = pto.vmi.ensure_layout %a : deinterleaved=2 -> contiguous
  scf.yield %a_dense
} else {
  %b_dense = pto.vmi.ensure_layout %b : deinterleaved=2 -> contiguous
  scf.yield %b_dense
}
```

`vmi-to-vpto` 之后，region result 会变成多个物理 VPTO value：

```text
scf.if -> (!pto.vreg<64xf32>, !pto.vreg<64xf32>)
```

这个场景说明为什么 layout 应该是 type 的一部分，而不是依赖 defining op。

### 4.8 完整 Histogram 语义

VPTO 的 histogram 指令一次读取 `256xui8` source，但结果只能写
`128xui16` accumulator。完整 `ui8` histogram 有 256 个 bin，因此物理 VPTO
接口需要通过 `#bin = 0/1` 分两次统计低半区和高半区。

VMI surface 不暴露这个物理 split：

```mlir
%hist = pto.vmi.vdhist %acc, %src, %mask
  : !pto.vmi.vreg<256xui16>,
    !pto.vmi.vreg<Nxui8>,
    !pto.vmi.mask<Nxpred>
 -> !pto.vmi.vreg<256xui16>
```

语义是完整 256-bin distribution histogram：

```text
for b = 0..255:
  hist[b] = acc[b] + count(i where mask[i] && src[i] == b)
```

Assignment 形状：

```text
src/mask = contiguous, b8 mask granularity
acc/result = contiguous 256xui16 logical value
```

VPTO 形状：

```text
acc/result part0 = bins   0..127
acc/result part1 = bins 128..255

for each 256-lane source chunk:
  part0 = dhistv2(part0, src_chunk, mask_chunk, #bin=0)
  part1 = dhistv2(part1, src_chunk, mask_chunk, #bin=1)
```

这说明 VMI 的易用性不只来自 layout assignment。对于这种 value-indexed
accumulation，VMI 还应该隐藏 VPTO 为了物理 vreg 宽度暴露出来的 range
selector、lo/hi accumulator 和多条物理指令。

`pto.vmi.vchist` 可以使用相同 surface 形状，但当前必须先验证 VPTO `CHISTv2`
在 high range 上返回的是全局累计还是 range-local 累计。这个差异会影响是否需要
额外给 high half 加上 low half 的总计数，因此不能只按 op 名字猜 lowering。

### 4.9 Block-Strided UB Staging

有些 CCE kernel 并不是在 register 内做任意 byte shuffle，而是先把结果写到
UB scratch，再用 block-strided vector load/store materialize 目标 UB layout。
`quant_minimum` 的 MXFP8 NZ case 是典型例子：

```text
compute:
  row-major ND FP8 scratch

row-wise staging:
  for row in 0..31:
    q8_row = vmi.stride_load(nd + row * 64,
                             block_stride=1)
    vmi.stride_store(q8_row, nz + row * 32,
                     block_stride=33)

copy-out:
  2D MTE copies two 1024B NZ planes from UB to GM
```

这里 `q8_row` 的 VMI value 仍然是 contiguous `64xf8` 逻辑向量：

```mlir
%q8_row = pto.vmi.stride_load %nd[%nd_off], %c1_i16, %mask
    : !pto.ptr<f8E4M3FN, ub>, i16, !pto.vmi.mask<64xpred>
    -> !pto.vmi.vreg<64xf8E4M3FN>

pto.vmi.stride_store %q8_row, %nz[%nz_off], %c33_i16, %mask
    : !pto.vmi.vreg<64xf8E4M3FN>, !pto.ptr<f8E4M3FN, ub>, i16,
      !pto.vmi.mask<64xpred>
```

Assignment 形状：

```text
stride_load result = contiguous
stride_load mask   = contiguous, granularity follows result element width
stride_store value = contiguous
stride_store mask  = contiguous, granularity follows value element width
```

VPTO 形状：

```text
base_in  = pto.addptr nd, nd_off
q8_row   = pto.vsldb base_in, block_stride=1, repeat_stride=0, mask

base_out = pto.addptr nz, nz_off
pto.vsstb q8_row, base_out, block_stride=33, repeat_stride=0, mask
```

这个场景说明：memory layout transformation 不一定要变成 VMI data layout。
只要 VMI op 的语义是“从哪些地址读/写哪些 logical lane”，register value
仍然可以保持 contiguous，`vmi-to-vpto` 也仍然是 local lowering。

## 5. 当前边界

当前设计方向：

```text
surface VMI:
  描述不带 layout 的逻辑向量语义。

layout assignment:
  选择 layout、mask granularity 和显式 materialization helper。

optimization:
  只在结果 IR 仍然可以 local lowering 时改写显式 helper。

vmi-to-vpto:
  严格 lower 它看到的 assigned/optimized IR。
```

暂不支持或有意收紧的范围：

```text
group_slots value 的普通 dense store:
  非法，除非先经过 group_broadcast 或其他显式 group-to-dense op。

packed group_slots f32->f16 cast:
  非法，除非 assignment 能把它 commute 到 group_broadcast 之后，或者使用
  支持的 row-local slots=1 path。

FP4 packed input/output:
  packed FP4 不属于当前 VMI surface。PTO/VPTO 已有 !pto.f4E1M2x2
  和 !pto.f4E2M1x2 packed 物理类型，且这些类型的 shape 语义是
  packed pair/byte 数，不是 logical FP4 lane 数。在 VMI 中直接写
  vreg<Nx!pto.f4E2M1x2> 会让 N 表示物理 packed byte 还是逻辑 FP4 元素
  产生歧义，因此 verifier 会直接拒绝
  vmi.vreg<...x!pto.f4E1M2x2/!pto.f4E2M1x2>。

  当前 VMI surface 不包含专用 FP4 packed-memory op。FP4 packed IO
  需要先作为独立语义重新设计，不能进入当前 dialect surface。

extract:
  暂不作为支持的 VMI surface。

padding transfer_read:
  当前 tail 设计不需要；tail 使用 mask。

scan / contract / compress / active_prefix_index:
  dialect surface 中可以存在，但除非补充具体 case，否则不属于第一阶段聚焦的
  layout/lowering 实现集合。

gather / scatter:
  当前只覆盖 UB pointer、contiguous layout 和已明确支持的 element/index 宽度。
  `ui16` gather 可承接 E8M0 byte-pair reorder；它不是通用 byte shuffle。
```

设计目标是优先保证语义完整：只要 VMI 接受某个 case，所需的 layout 沟通就必须
在 IR 中显式表达，并且能被 `vmi-to-vpto` local lowering。
