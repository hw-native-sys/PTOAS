# 编译选项总览

## 范围

本章介绍 `ptoas` 中与用户直接相关的常用编译选项。

重点说明：

- 这些选项控制什么用户可见行为
- 默认行为是什么
- 什么时候建议修改
- 遇到问题时如何选择

本章不展开 `ptoas` 内部实现细节，只从使用方式和结果差异的角度说明这些选项。

## 阅读建议

如果你是第一次使用 `ptoas`，建议优先关注：

1. 目标架构选择：`--pto-arch`
2. 自动同步相关选项：`--enable-insert-sync`、`--enable-inject-barrier-all-sync`

如果你正在排查编译或结果问题，再看：

- 自动内存规划：`--enable-plan-memory`（默认开启，当前不支持关闭）；`--plan-memory-order-by-size` 只调整同一地址空间内的规划顺序（默认按 Tile 的声明顺序规划）

以上选项的取值和默认值见[选项总表](#选项总表)。

## 选项总表

下表汇总 `ptoas` 显式提供的主要命令行选项。

说明：

- 这里只列 `ptoas` 自身定义的选项
- 不展开 LLVM / MLIR 通用选项，例如 `--help`、`--version`、`--mlir-print-ir-after`
- 某些选项虽然工具支持，但只适合特定排查场景

| 选项 | 参数 | 默认值 | 作用 |
|---|---|---:|---|
| `<input file>` | 路径 | `-` | 输入的 `.pto` 文件；未指定时从标准输入读取 |
| `-o` | 路径 | `-` | 输出文件名；未指定时输出到标准输出（mixed fatobj 模式要求显式指定 `-o`） |
| `--pto-arch` | `a3` \| `a5` | `a3` | 指定目标架构 |
| `--enable-plan-memory` | 无 | 开启 | 自动内存规划总开关；当前只支持开启，为未显式给出地址的 `pto.alloc_tile` 规划地址 |
| `--plan-memory-order-by-size` | 无 | 关闭 | 同一地址空间内优先规划较大的本地 buffer，再套用基础复用策略 |
| `--enable-insert-sync` | 无 | 关闭 | 启用常规自动同步插入 |
| `--enable-inject-barrier-all-sync` | 无 | 关闭 | 启用更保守的同步插入策略 |
| `-mllvm` | 后端参数（可重复） | 无 | 按 `-mllvm <arg>` 配对把下一个参数透传给设备端编译（bisheng 自身也用这个配对） |
| `-Xbisheng` | 后端参数（可重复） | 无 | 按 clang `-Xclang` 约定把下一个 token 透传给设备端编译 |
| `--bisheng-arg` | 后端参数（可重复） | 无 | 上面两个的显式长别名，每个 occurrence 透传一个 token |
| `--vmi-fastmath` | 无 | 开启 | 记录整条 VMI 流水线的转换状态，同时选择布局关系与可能改变 NaN 结果的转换快速路径 |
| `--vmi-layout-max-frontier-entries` | 正整数 | `288` | 每个 VMI 布局子图的搜索状态数上限 |
| `--vmi-layout-max-transitions` | 正整数 | `16384` | 每个 VMI 布局子图的搜索转移次数上限 |

上表中的两个自动同步选项互斥：`--enable-insert-sync` 和 `--enable-inject-barrier-all-sync` 一次只能启用一个，同时指定会直接报错。

`--vmi-fastmath` 默认开启。它是整条 VMI 流水线的转换状态：`vmi-layout-assignment` 与 `vmi-to-vpto` 读取同一个值，因此该开关同时决定布局关系。单源分片（K=1）的 `f32` 到 `bf16` 窄化在 `Z` 舍入且不饱和时，只有开启才允许使用稠密 `c -> c` 关系；否则使用 `lane_stride = 2` 关系，并在稠密消费者前插入一次打包。开启时这类转换会使用位级截断；例如 `f32` 位模式 `0x7F800001` 可能变为表示正无穷的 `bf16` 位模式 `0x7F80`。如果程序要求保留这类 NaN 结果，请显式传入 `--vmi-fastmath=false`。

遇到 VMI 布局搜索预算耗尽时，可仅对当前编译增大预算，例如同时传入
`--vmi-layout-max-frontier-entries=1024 --vmi-layout-max-transitions=32768`。
两个参数均必须大于零，按子图分别计数；增大预算可能增加编译时间和内存占用。

## 向 Bisheng 透传后端选项

这些参数原样交给 bisheng：`-mllvm <arg>` 按 `-mllvm` 配对交给 LLVM codegen 解析器，
`-Xbisheng <arg>`、`--bisheng-arg=<arg>` 和顶层模块属性 `pto.bisheng_args` 交给 bisheng
驱动。效果等同于你自己敲 `bisheng ...`：`ptoas` 不校验、不改写、不重排，只拒绝一个完全
空的 token。它只负责**位置**——这些 token 插在自己的参数之后、固定的
`-c -x ir - -o <tmp.o>` 尾巴之前，所以带取值的 `-o <path>` 会被自己的 `-o` 覆盖；但一个
**裸露的**取参 token（裸 `-o`、`-x`、`-mllvm`、`--`）会吃掉尾巴里的下一个 token，那是
使用者自己的选择：多数情况 bisheng 会用它自己的报错指出问题，但裸 `-o` 在 CANN 9.2.0-beta.2
上仍是 rc 0（产物与不加参数时等效，只是命令行结构变了）——这类形式请自己确认。

```bash
# issue #1729 的场景：关闭 VF 预留 UBUF，裸的 bisheng 驱动选项
ptoas kernel.pto --pto-arch=a5 --pto-backend=vpto \
  -Xbisheng --cce-disable-vf-stack-reserved-ubuf -o kernel.o
```

`-Xbisheng` 和 `-mllvm` 各吃一个 token，`--bisheng-arg=<arg>` 一次一个；同样的 token
也可以写进输入文件顶层模块的属性，例如
`pto.bisheng_args = ["--cce-disable-vf-stack-reserved-ubuf"]`。四种来源合并成一份列表
（模块属性在前，命令行按出现顺序在后）。生效范围：`--pto-backend=vpto` 的设备
LLVM 编译、`emitc` 设备端 C++（含 fatobj）、VPTO bridge wrapper 的前端编译；host stub
（`bisheng -cc1`）和 target-attribute 探测不接收。只输出文本的模式（`--emit-vpto`、
`--emit-pto-ir`、`emitc -o x.cpp`）不跑设备编译，会提示这些参数未被使用。参数在读取后
即被移除，不会进入导出的 MLIR / C++ / fatobj；需要看合并结果时设
`PTOAS_DEBUG_BISHENG_ARGS=1`。失败时 `ptoas` 会打印完整命令（`Command: ...`），bisheng
自己的报错原样输出。

需要留意：`ptoas` 不会阻止你覆盖它自己的选择，例如 `--cce-aicore-arch=dav-c310` 会改变
kernel metadata（在 a5 vector kernel 上仍然 rc 0），而
`-mllvm -cce-vf-enable-vf-ldst-elimination=true` 因为排在 `ptoas` 的 `=false` 之后而生效，
会重新打开已知的 CANN 误编译类别——这两个都是使用者显式要求的结果，`ptoas` 既不检测也不警告。

在 PTODSL 里用 `pto.jit(ptoas_flags=[...])` 透传，元素必须以 `--` 开头，例如
`ptoas_flags=["--bisheng-arg=--cce-disable-vf-stack-reserved-ubuf"]`；它同时进入 native
build 的缓存键。

## 选项使用顺序

对大多数用户，建议按下面顺序理解和使用这些选项：

1. 先明确 `--pto-arch`
2. 优先保持默认布局相关行为
3. 按需要开启自动同步
4. 只有在排查问题时，再使用更保守的同步选项或中间结果观察选项

## 本章目录

- [目标架构选项](./target_arch_options.md)
- [自动内存规划](./memory_planning.md)
- [自动同步选项](./auto_sync_options.md)

## 使用建议

如果你的目标是先理解 PTO ISA 和写出合法程序，应优先保持输入清晰、语义明确，而不是一开始就调整过多编译选项。
