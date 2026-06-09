# 2.4 PTO ISA 层级

## 1. 概述

PTO ISA 采用分层模型组织，不同层级暴露不同强度的存储、同步与调度细节。`ptoas` 通过 `--pto-level` 选择当前程序采用的 ISA 层级，并据此决定是否允许自动内存规划、是否要求显式地址，以及哪些操作形式可以被接受。

## 2. 层级模型概述

PTO ISA 不是单一抽象层的表示，而是一组逐步细化的层级接口。各层级共享同一套 PTO 语义对象，但在以下方面的暴露程度不同：

- tile 是否需要显式缓冲对象表示
- 本地地址是否由用户显式提供
- 同步与调度细节是否直接写入 IR
- 是否允许由工具补充部分地址或同步相关信息

这种分层的目的，是在“前端表达便利性”和“后端控制精确性”之间建立清晰边界。

## 3. 层级定义

### 3.1 Level-2

Level-2 是当前 PTO ISA 的主体工作层，也是 `ptoas` 的默认构建层级。

其核心特征是：

- tile 以显式缓冲对象表示，而不是纯 SSA 值
- `pto.tile_buf` 成为主要本地对象类型
- `pto.alloc_tile` 用于显式声明 tile buffer 生命周期
- 数据搬运、计算和多数算子以 destination-passing style 组织
- 布局、有效区域、位置等信息已经进入类型与编译检查约束

这一层强调：

- 上层或用户显式管理 tile 对象
- 地址可以由工具自动安排
- 更适合先表达语义，再逐步细化控制

### 3.2 Level-3

Level-3 是更接近显式资源控制的低层 PTO ISA 形态。

其核心特征是：

- pipeline / event / synchronization 信息更显式
- 本地 tile 地址不再由 `ptoas` 自动规划，而是要求 IR 直接给出
- 更适合手工控制执行顺序、缓冲区地址与同步依赖

在这一层，IR 已经不再假设编译器替用户补足所有资源决策，而是要求输入模块本身携带更完整的执行与存储信息。

## 4. 层级差异

| 维度 | Level-2 | Level-3 |
| --- | --- | --- |
| tile 表示 | 显式 `tile_buf` | 显式 `tile_buf` |
| 分配责任 | 用户声明 tile，对地址仍可自动规划 | 用户声明 tile，且需显式提供地址 |
| 地址语义 | 中 | 强 |
| 同步暴露 | 可配合自动补充 | 更显式、偏手工控制 |
| 典型用途 | 主体编译、验证、自动规划 | 低层调度、显式控制、接近生成 |

## 5. 为什么 Level-2 / Level-3 采用显式 tile buffer

当前 PTO ISA 公开接口重点聚焦在 Level-2 / Level-3，这两个层级都把 tile 建模为 buffer 对象，而不是纯值对象。

这样做的主要原因是：

- tile 的生命周期、地址和复用需要被清楚表达
- 同步与执行顺序在低层场景中需要更明确地写出
- 如果把这些信息完全隐藏，程序可读性和约束边界都会变差

因此，在 Level-2 / Level-3 中：

- `pto.alloc_tile` 负责显式表达 tile buffer 生命周期
- 不同层级决定地址和同步信息由用户写多少

## 6. `ptoas` 的 `--pto-level`

### 6.1 可选值

`ptoas` 当前公开支持以下构建层级：

- `--pto-level=level2`
- `--pto-level=level3`

默认值为：

```bash
--pto-level=level2
```

如果给出非法值，`ptoas` 会直接报错并终止。

### 6.2 Level 选择的作用

`--pto-level` 不只是一个说明性标签，它会直接影响：

- `ptoas` 对输入 IR 的合法性检查
- 是否允许 `pto.alloc_tile` 带显式 `addr`
- 是否执行自动本地内存规划
- 某些低层操作是否可用

## 7. 不同 level 下的常见写法

### 7.1 `level2`

`level2` 是默认模式，也是当前最稳定、最常用的 PTO 编译层级。

在这一模式下，用户通常：

- 显式声明 tile buffer
- 不显式提供 tile 地址
- 让工具负责自动分配本地内存

### 7.2 `level3`

`level3` 面向更低层、更显式控制的 IR。

在这一模式下：

- 每个 `pto.alloc_tile` 都必须提供 `addr`
- 允许使用 `pto.tassign`
- 自动同步相关模式不能和 `pto.tassign` 混用

也就是说，`level3` 下用户需要自己把 tile 的地址与低层调度信息写清楚。

## 8. 与操作/约束相关的显式规则

`--pto-level` 至少会触发以下显式约束：

### 8.1 `pto.alloc_tile` 的 `addr` 约束

- 当 `--pto-level=level3` 时，`pto.alloc_tile` 必须带 `addr`
- 当 `--pto-level=level2` 时，`pto.alloc_tile` 不能带 `addr`

这条规则对应了“地址由谁负责”的层级边界：

- Level-2：地址由编译器规划
- Level-3：地址由输入 IR 明确指定

### 8.2 `pto.tassign` 的层级约束

`pto.tassign` 只允许在：

```bash
--pto-level=level3
```

下使用。

如果在 `level2` 中出现 `pto.tassign`，`ptoas` 会直接报错。

### 8.3 `pto.tassign` 与自动同步模式互斥

当模块中存在 `pto.tassign` 时，以下自动同步模式都必须关闭：

- `--enable-insert-sync`
- `--enable-inject-barrier-all-sync`

这说明 `pto.tassign` 所处的 Level-3 模式，更偏向用户手工管理同步与调度。

## 9. 从使用方式理解各层级

从用户角度看，level 主要决定的是“资源与调度细节由谁负责”：

- `level2`：更适合把重点放在语义表达上
- `level3`：更适合显式控制地址、同步和低层资源

## 10. 选型建议

如果目标是：

- 编写多数 `.pto` 样例
- 使用默认编译路径
- 依赖工具做自动地址规划

应优先选择：

```bash
--pto-level=level2
```

如果目标是：

- 手工控制 tile 地址
- 显式管理更低层同步与调度
- 使用 `pto.tassign` 这一类仅限低层接口的操作

应选择：

```bash
--pto-level=level3
```

## 11. 当前支持情况

从用户使用角度看，可以直接选择的层级只有：

- `level2`
- `level3`

其中：

- `level2` 是默认且最常用的层级
- `level3` 用于需要显式地址和更低层控制的场景

## 12. 示例

### 12.1 默认层级

```bash
ptoas input.pto -o output.cpp
```

等价于：

```bash
ptoas input.pto --pto-level=level2 -o output.cpp
```

### 12.2 选择 Level-3

```bash
ptoas input.pto --pto-level=level3 -o output.cpp
```

此时需要保证：

- `pto.alloc_tile` 带 `addr`
- 模块不依赖自动本地内存规划来补全地址
- 如果使用 `pto.tassign`，自动同步模式全部关闭
