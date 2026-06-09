# 5.4 地址规划模型

## 1. 概述

`ptoas` 会把“需要哪些本地 / 工作区缓冲”转换成“这些缓冲实际落在什么偏移上”。这是第五章里最工程化的一环，也是把对象模型变成可生成代码表示的关键阶段。

## 2. 入口

## 3. 规划模式

### 3.1 `local-mem-plan` mode

用于本地内存规划，核心目标是给参与本地地址规划的 buffer 分配本地偏移。

### 3.2 `global-work-space-plan` mode

用于全局 workspace 规划，适合需要工作区分配与复用的路径。

## 4. 关键选项

当前实现暴露的主要选项包括：

- `--mem-plan-mode=local-mem-plan`
- `--mem-plan-mode=global-work-space-plan`
- `--enable-global-workspace-reuse`
- `--enable-print-memory-allocated-size`
- `--restrict-inplace-as-isa`

这些选项分别影响：

- 规划对象属于本地地址空间还是全局工作区
- 是否允许 workspace 复用
- 是否打印分配规模
- 是否按更保守的实现边界限制 inplace

## 5. 规划输入

地址规划不是只看 `alloc_tile` 列表，而是组合多类分析结果：

- buffer 信息
- buffer 生命周期
- gen / kill 信息
- 语义冲突对
- inplace 候选对
- stable value order

这些信息由 `MemLivenessAnalysis` 和相关辅助分析阶段构造，再交给 `MemPlan` 真正完成分配。

## 6. 规划过程概览

典型流程可以概括为：

1. 构建线性操作序
2. 识别需要规划的 buffer
3. 计算 buffer 生命周期
4. 生成 inplace / conflict 关系
5. 按地址空间与规划模式进行分配
6. 回写偏移并把地址物化到 IR 中

在本地地址规划模式下，普通本地 buffer 会先参与核心 `MemPlan` 算法；之后 `reserve_buffer` 再从同一地址空间中寻找对齐空洞。

## 7. inplace 与复用

地址规划的目标不是“每个 buffer 一块新地址”，而是尽量在不破坏语义的前提下复用：

- 生命周期不重叠的 buffer 可以候选复用
- 满足语义条件的 buffer 可以形成 inplace 对
- `restrict-inplace-as-isa` 会把某些本可复用的情况收紧为更保守策略

这使得最终的地址规划结果通常取决于：

- 生命周期关系
- 地址空间
- 语义冲突
- 目标相关限制

## 8. `reserve_buffer` 与地址规划的关系

`reserve_buffer` 不进入普通 buffer 的核心 `MemPlan` 分配逻辑，而是在普通本地 buffer 完成规划后，再基于已占用区间找第一个满足对齐要求的空洞。

自动分配模式下：

- `auto = true`
- `base` 必须缺省
- pass 会补出解析后的 `base`

当前 `local-mem-plan` 模式下：

- 显式 `base` 的 `reserve_buffer` 会被拒绝
- 若需要保留显式基址路径，应走更接近手工控制的编译层级

## 9. 结果物化

地址规划完成后，偏移信息会写回到相关对象，使后续阶段不再停留在抽象“待分配”状态。
