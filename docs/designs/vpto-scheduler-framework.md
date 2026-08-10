# VPTO 调度框架

本文描述 [PTOAS issue #1143](https://github.com/hw-native-sys/PTOAS/issues/1143)
对应的调度基础设施、集成位置和稳定边界。当前范围是建立可验证的调度分析闭环，
不包含指令重排策略。

## 流水线位置与模式

调度驱动是模块级 pass，按 IR 中的函数顺序构建函数内分析；它位于 VPTO 发射
准备流水线的最终 CSE 之后、
`PTOValidateVPTOEmissionIR` 之前。同步插入、wrapper 展开、vecscope 推断、
soft-postupdate、LICM、循环计数器窄化和清理均已完成，因此调度器看到的是最终
发射形态的 VPTO IR。

`ptoas` 提供 `--vpto-scheduler=<off|analyze|on>`：

调度器只分析 A5 Vector kernel。CV 分裂产生的 Cube 子模块会被跳过，不构建
DAG，也不输出调度报告；不带 `pto.kernel_kind` 的模块仍可用于独立调试 pass。
在非 A5 架构上显式选择 `analyze` 或 `on` 时，`ptoas` 会报告
`--vpto-scheduler requires --pto-arch=a5` 并终止编译。

- `off`：默认值，不创建 pass，不输出报告，也不改变 IR。
- `analyze`：构建调度分析并把确定性文本报告写入标准错误，不改变 IR。
- `on`：执行与 `analyze` 相同的分析，不改变 IR。

独立调试 pass 时，可使用
`pto-test-opt input.pto '-pto-vpto-scheduler=mode=analyze'`。

## 组件边界

调度框架按以下单向数据流组织：

```text
操作语义分类
  -> 基本块内区域划分
  -> 带类型和强度的依赖 DAG
  -> 目标调度模型
  -> ready boundary / 资源 / hazard / 寄存器压力
  -> analyze 报告
```

### 操作语义与区域

`VPTOSchedulingOpInterface` 允许操作声明调度类别、隐式读写和是否为调度边界。
没有专用接口的操作通过统一分类器落入 `schedulable`、`structural`、`boundary`
或 `unsupported`。区域只包含同一基本块内连续且可分析的操作；terminator、带
region 的控制流和 unsupported 操作切断区域。报告同时保留区域前后边界原因，
避免后续策略越过不可证明安全的边界。

默认分类不把 `PTO_MicroOp` 本身当作 schedulable 证明。只有具备明确 family
marker、`OpPipeInterface` 或非空调度 effect 的接口 op 才进入调度区域；新增的
generic micro-op 若没有补充分类证据，会保守成为 `scheduling boundary`。
coverage 除四类总数外，还按名称输出 `unsupported-op` 与
`unclassified-op`；后者专指实现了调度接口、但缺少 family/pipe/effect 分类证据
而落到默认 boundary 的 op。真实 `ptoas` A5 Vector 发射流水线测试要求代表性
VPTO block 的报告中 `unsupported=0` 且没有 `unclassified-op`。

### 依赖 DAG

每个区域生成稳定编号的 `VPTOSUnit`。依赖边记录：

- 类型：`data`、`anti`、`output`、`memory`、`control`、`sync`、`artificial`
  或 `cluster`；
- 强度：影响正确性的 `must` 与仅供策略参考的 `weak`；
- latency 和可诊断 reason。

SSA def-use 直接产生 `data/must` 边。内存依赖优先读取
`MemoryEffectOpInterface`；不同地址空间视为不别名，同一 alias root 上能够解析为
常量 element offset 和固定 element size 的标量访问按字节区间判断重叠，其余
同地址空间访问保守排序写相关访问。原子操作由具体 atomic op 类型声明
`AtomicMemory` effect，带 `volatile`/`is_volatile` 属性的 micro-op 声明
`VolatileMemory` effect；DAGBuilder 只消费这些接口语义，不依赖名称猜测。
同步与接口声明的隐式读写产生对应的强依赖。
DAG 构建完成后计算 topological depth/height，并保留 live-in/live-out。
隐式状态以资源名独立跟踪；例如各 SPR 名称以及 CTRL 分别形成自己的
RAW、WAR、WAW 依赖链，互不相关的隐式状态不会被强制串行化。
带可选 `updated_base` 结果的 VPTO memory op 在结果存在时声明 `PostUpdate`
effect；更新后的地址仍通过普通 SSA def-use 建立 `data/must` 边，但报告会明确
标记为 post-update address，便于审计地址演化链。
`pto.mem_bar`、`pto.dsb`、`pto.barrier`、`pto.fence.barrier_all` 以及三类
SIMT fence/barrier op 声明 `Barrier` effect。当前 DAG 以保守的完整调度屏障处理：
屏障前所有节点连向屏障，屏障再连向后续所有节点。
`pto.set_flag`/`pto.wait_flag` 及其 dynamic 变体声明 `Event` effect。静态事件按
source pipe、destination pipe、event-id 精确匹配；dynamic event-id 在同一 pipe
pair 内保守视为可能相同。所有可能匹配的 signal 到后续 wait 建立 `sync/must` 边。
接口同时用 `Pipe` effect 标出操作的执行 pipe。set_flag 必须位于之前的 source
pipe 指令之后，wait_flag 必须位于后续 destination pipe 指令之前；`pto.barrier`
只约束其指定 pipe（`PIPE_ALL` 匹配所有 pipe），避免把无关执行 pipe 串行化。
尚未实现精确 `OpPipeInterface` 的 raw MTE micro-op 以 unknown/`PIPE_ALL` pipe
保守匹配所有同步约束，不会因缺失 pipe 信息错误穿越同步点。
`pto.get_buf`/`pto.rls_buf` 及 dynamic 变体声明 `BufferId` effect。静态 ID 精确
匹配，dynamic ID 保守视为可能匹配；acquire→release 以及 release→后续 acquire
形成 `sync/must` 边。mode 0 acquire/release 同时接入对应 pipe 的前后执行顺序；
非零 mode 的 deferred release 保守等待所有已知 pipe 执行。

### 目标模型与动态状态

`VPTOSchedModel` 是只读目标契约。内置 `generic-a5-v1`：issue width 为 1，
资源契约预留 scalar、vector、MTE、cube、control 和 unknown，压力集合分为
vector、predicate、scalar、address、align 和 special。调度器只在 Vector kernel
上消费该模型，Cube 资源项作为后续扩展保留。模型完整度标记为 `minimal`；
未知调度类必须在覆盖率中显式出现，不能被静默当作精确模型。
操作实现 `OpPipeInterface` 时，模型优先根据 `getPipe()` 选择执行资源；没有显式
pipe 契约时，再退回 Vector、MTE、Cube 或 SIMT micro-op family marker。

方向化 ready boundary 分别维护 top/bottom 的可用、pending 和已提交集合。
资源跟踪器提供最早可发射周期、stall 和逐周期占用；hazard recognizer 是目标
特有限制的扩展点；压力跟踪器同时维护每个压力集合的 delta、current、peak 和
可选超限值。

## Analyze 输出契约

报告统一以 `vpto-scheduler:` 开头，按函数、基本块、区域、节点、边、原始顺序
发射模拟、时间线、覆盖率的顺序输出。节点和边使用区域内稳定 ID，不打印指针
或位置地址。unsupported 操作名按字典序输出，因此相同 IR 的报告可用于回归
测试和版本比较。

发射模拟只评估原始指令顺序，用于联合验证资源和压力状态机；它不是
最终调度结果。任何 DAG 环、非法模型引用或 tracker 拒绝都会输出明确的
`fallback=` 原因。

## 范围外事项

- 不移动、克隆或删除任何 IR 操作；
- 不实现候选优先级、双向选点或束调度策略；
- 不声称 `generic-a5-v1` 是完整的硬件性能模型；
- 不越过基本块或控制流边界调度；
- 不把 weak 边升级为正确性约束。

后续工作调度器可以复用同一 DAG、目标模型和状态跟踪接口，在 `on` 模式中选择
候选并提交重排，同时保留 `analyze` 作为行为与覆盖率审计入口。
