# VMI Layout Relation-Aware Rematerialization Design

本文描述 VMI layout optimization 中 relation-aware rematerialization 的设计。
目标是让 `vmi-layout-assignment` 只产生 legal baseline IR，把跨 layout
relation 的优化放到显式 `ensure_layout` 上完成。

## 1. Motivation

`vmi-layout-assignment` 已经负责三件 hard legalization 工作：

```text
1. 为每个 VMI value 选择 concrete layout
2. 在不匹配的 use-site 插入 ensure_layout / ensure_mask_layout
3. 保证 vmi-to-vpto 只需要 local lowering information
```

对 `ext` 这类 width-changing op，assignment 的 baseline 可以保守选择：

```text
ext f16 -> f32:
  source = contiguous
  result = deinterleaved=2
```

如果下游 `truncf f32 -> f8` 要求 source 为 `deinterleaved=4`，assignment 会
显式插入：

```text
%e = pto.vmi.extf %x
  : !vreg<..., layout<contiguous>>
 -> !vreg<..., layout<deinterleaved = 2>>

%e4 = pto.vmi.ensure_layout %e
  : !vreg<..., layout<deinterleaved = 2>>
 -> !vreg<..., layout<deinterleaved = 4>>
```

这个 IR 已经合法，但不是最优。优化 pass 可以从显式 helper 出发，把 relation
应用到 producer：

```text
ensure_layout(ext(src), resultLayout)
  => ext(ensure_layout(src, derivedSourceLayout))
```

这样 assignment 不需要做 consumer-driven global propagation，也不需要在多
consumer 冲突时引入 cost model。

## 2. Goals

```text
1. assignment 保持 hard legalization baseline，不做 ext relation propagation。
2. relation-aware optimization 从显式 ensure_layout 出发。
3. 多 consumer 冲突由 use-site helper + rematerialization 解决。
4. vmi-to-vpto 仍只消费当前 op 的 operand/result layout，不扫描上下文。
5. 变换必须是局部、确定、可验证的 IR rewrite。
```

非目标：

```text
1. 不做 ComputeY1 专用 pattern。
2. 不在 assignment 中实现全局 cost model。
3. 不通过 vmi-to-vpto 猜 producer/consumer relation。
4. 第一阶段不做 trunc/narrow relation remat。
```

## 3. Optimization Model

relation-aware remat 以 `ensure_layout` 为唯一触发点：

```text
%wanted = pto.vmi.ensure_layout %source : sourceLayout -> targetLayout
```

如果 `%source` 的 producer 可以在 `targetLayout` 或 relation 派生出的 operand
layout 下重新创建等价结果，则用 cloned producer 替换 helper。

### 3.1 Layout-Transparent Producer Remat

对 layout-transparent elementwise op：

```text
ensure_layout(op(a, b), L)
  => op(ensure_layout(a, L), ensure_layout(b, L))
```

适用对象包括纯 elementwise data ops：

```text
vadd/vsub/vmul/vdiv/vmin/vmax
vand/vor/vxor/vshl/vshr
vneg/vabs/vsqrt/vexp/vln/vrelu/vnot
vmula
select, when data operands and mask layout requirements can be kept explicit
```

第一阶段可以先覆盖 ComputeY1 需要的 `vmul`，但实现形态应按 op family 泛化。

### 3.2 Widen Ext Relation Remat

对 widening `ext`：

```text
ensure_layout(ext(src), resultLayout)
  => ext(ensure_layout(src, sourceLayout))
```

其中：

```text
resultFactor = sourceFactor * widenFactor
```

例子：

```text
ext f16 -> f32, widenFactor = 2
target result layout = deinterleaved=4
derived source layout = deinterleaved=2
```

`deinterleaved=1` 等价于 contiguous。

### 3.3 Producer Fold After Remat

relation remat 可能暴露 producer-side helper：

```text
ensure_layout(load(...), deinterleaved=2)
```

这类 helper 应由 `vmi-layout-fold` 吸收到 producer 或 consumer：

```text
load contiguous + ensure_layout to deinterleaved=2
  => load result deinterleaved=2
```

因此推荐优化 pipeline 在 remat 后再次运行 fold：

```text
vmi-layout-assignment
  -> canonicalize/cse
  -> vmi-layout-rematerialize
  -> canonicalize/cse
  -> vmi-layout-fold
  -> canonicalize/cse
  -> vmi-layout-sink-materialization
  -> canonicalize/cse
```

## 4. Multi-Consumer Conflict

如果一个 `ext` result 有两个 consumer：

```text
consumer A requires deinterleaved=2
consumer B requires deinterleaved=4
```

assignment 不需要判断哪个更优。它可以选择稳定 baseline，例如 `deinterleaved=2`，
并为另一个 use 插入 helper：

```text
%e2 = pto.vmi.extf %x : contiguous -> deinterleaved=2
consumer_a(%e2)

%e4 = pto.vmi.ensure_layout %e2 : deinterleaved=2 -> deinterleaved=4
consumer_b(%e4)
```

remat 再把第二个 use 优化成 cloned producer：

```text
%x2 = pto.vmi.ensure_layout %x : contiguous -> deinterleaved=2
%e4 = pto.vmi.extf %x2 : deinterleaved=2 -> deinterleaved=4
consumer_b(%e4)
```

原 `%e2` 仍服务 `consumer_a`。这样不需要 assignment 做全局 cost selection。

## 5. ComputeY1 Shape

baseline assignment 可能产生：

```text
%x32 = extf %x16              // result deinterleaved=2
%s32 = extf %scale16          // result deinterleaved=2
%m   = vmul %x32, %s32        // result deinterleaved=2
%m4  = ensure_layout %m       // deinterleaved=2 -> deinterleaved=4
%y   = truncf %m4
```

remat/fold 后目标 IR：

```text
%x16_d2 = load ...            // folded deinterleaved=2 load
%x32_d4 = extf %x16_d2        // deinterleaved=2 -> deinterleaved=4

%scale16_d2 = group_broadcast_load ... // folded/assigned deinterleaved=2
%scale32_d4 = extf %scale16_d2         // deinterleaved=2 -> deinterleaved=4

%m4 = vmul %x32_d4, %scale32_d4
%y  = truncf %m4
```

关键点：

```text
1. truncf 只通过 ensure_layout 表达自己的 source layout requirement。
2. remat 不需要识别 quant 语义。
3. ext relation 是 local rule。
4. load/group_broadcast_load 的物理优化由 fold 或 producer capability 处理。
```

## 6. Lowering Contract

`vmi-to-vpto` 的 contract 不变：

```text
1. 不扫描 ext 的 users。
2. 不扫描 producer chain 来猜 layout。
3. 只根据当前 op 的 operand/result layout lower。
```

relation-aware remat 必须在 `vmi-to-vpto` 前把 IR 显式改写为：

```text
%x = pto.vmi.load ... -> !vreg<..., layout<deinterleaved = 2>>
%e = pto.vmi.extf %x
  : !vreg<..., layout<deinterleaved = 2>>
 -> !vreg<..., layout<deinterleaved = 4>>
```

之后 lowering 只消费这个 local shape。
