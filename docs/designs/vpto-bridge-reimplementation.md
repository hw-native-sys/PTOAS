# VPTO C++ Bridge 技术设计

## 概述

VPTO C++ Bridge 为部分 PTO 操作提供一条受控的 C++ wrapper 生成与调用路径。
它允许 PTO IR 保持稳定的语义接口，同时复用 PTO-ISA/CANN wrapper 中已经存在的
设备侧实现。

Bridge 不替代 VPTO 的常规 lowering。一个模块可以同时包含 bridge 操作和普通
VPTO 操作：被路由的操作生成 bridge 调用，其余操作继续使用原有的 VPTO lowering
路径。当前实现覆盖 Pipe、Cube Direct 和 Print 三类 family。

## 整体结构

```text
route policy
    │ 选择 family / op
    ▼
typed lowering
    │ 按 PTO op 语义构造结构化 bridge IR
    ▼
bridge registry ── canonical spec / core ──► instance resolution
    │                                              │
    ├──────────────► C++ wrapper generation ◄──────┘
    │
    └──────────────► bridge call lowering
                           │
                           ▼
                    object emission / linking
```

各层职责如下：

- **Route policy** 只决定某个 family 或操作是否使用 bridge。
- **Bridge registry** 描述可调用 entry 的 ABI、family、core 和 renderer。
- **Typed lowering** 将 PTO 操作转换为带有明确类型和属性的结构化 bridge IR。
- **Instance resolution** 根据 entry、规范化参数和目标 core 选择唯一实例。
- **Wrapper generation** 为实例生成 C++ wrapper 及其声明。
- **Bridge call lowering** 将结构化 bridge 调用转换为最终的函数调用和 LLVM 表示。
- **Object emission** 编译、链接 wrapper，并保留各目标架构原有的发射流程。

这种分层使路由策略、ABI 描述和 C++ 实现细节彼此隔离：策略文件不能定义任意
C++ 符号、include、模板表达式或调用参数布局。

## 路由策略

可以通过 `--vpto-bridge-policy=<file>` 指定 YAML 路由策略。例如：

```yaml
version: 1
families:
  pipe:
    enabled: true
  cube:
    enabled_ops:
      - pto.tmatmul
      - pto.tmatmul.acc
  print:
    enabled_ops:
      - pto.print
      - pto.tprint
```

策略文件只包含版本、family 开关和已注册的操作名。解析器会检查未知版本、未知
family、未知操作、重复项和类型错误。

未被策略选中的 Cube 操作继续使用原有 `ExpandTileOp` 路径。对于没有非 bridge
实现的操作，关闭 bridge 后会得到明确诊断，而不会产生未解析的 wrapper 符号。
操作被选中但没有被对应的 typed lowering 消除时，编译也会在发射前报错。

当前默认路由为：

- Pipe 默认启用。
- Print 默认启用，因为 `pto.print` 和 `pto.tprint` 在 VPTO 上没有可用的非
  bridge 实现。
- Cube 默认不启用，需要在策略中逐个选择操作。

## Registry 与 ABI

所有 family 共用编译器内的强类型 registry。entry descriptor 至少包含以下信息：

```cpp
enum class BridgeFamily { Pipe, Cube, Print };
enum class BridgeCoreKind { Any, Vector, Cube };
enum class BridgeValueKind { Pointer, I32, I64, ObjectHandle };
enum class BridgeRendererKind { Pipe, CubeDirect, Print };

struct BridgeFunctionDesc {
  BridgeEntryId id;
  BridgeFamily family;
  BridgeCoreKind core;
  BridgeRendererKind renderer;
  llvm::StringLiteral symbolBase;
  llvm::ArrayRef<BridgeValueKind> arguments;
  llvm::ArrayRef<BridgeValueKind> results;
};
```

实际 descriptor 还记录指针地址空间、整数 signedness，以及对象存储的 size/alignment
约束。`ObjectHandle` 是 bridge IR 中的逻辑对象句柄，不表示 C++ 初始化函数的
返回值。

典型 entry 的逻辑 ABI 如下：

| Entry | 输入 | 输出 | 语义 |
|---|---|---|---|
| `pipe.init` | 配置、对象存储 | object handle | 在调用方提供的存储中初始化对象 |
| `pipe.push` | object handle、地址 | 无 | 向 pipe 写入 tile 或地址 |
| `pipe.pop` | object handle | 地址 | 从 pipe 取出地址并回绑结果 |
| `cube.tmatmul` | dst、lhs、rhs 地址 | 无 | 调用 cube 矩阵乘 wrapper |
| `cube.tmatmul.acc` | dst、acc、lhs、rhs 地址 | 无 | 调用带累加目标的 cube wrapper |
| `print.f32` | 格式串指针、f32 | 无 | 通过 DebugTunnel 输出标量 |

对象存储由编译器静态分配并校验上界，不依赖运行时 size 查询 entry。registry
同时保证 entry id、操作名和符号基名的唯一性。

## Typed lowering 与实例化

Pipe、Cube Direct 和 Print 共享同一个路由 pass，但各自使用对应的 typed lowering
逻辑。lowering 阶段只生成结构化 bridge IR，不直接拼接 C++ 源码或 LLVM 调用。

Pipe 和 Cube 使用带 verifier 的 tile spec；Pipe、Cube 的对象和 tile 参数通过
模块顶层的 bridge instance symbol 统一管理。bridge 操作使用 symbol reference
引用实例，规范化后的属性身份用于去重。

实例的外部 wrapper 名称由编译单元的 SHA-256 摘要、core 和模块内实例编号
共同限定。摘要基于实例表插入前、去除操作及块参数调试位置的 IR 副本，
使用完整的 MLIR bytecode 序列化，不受全局 IR 打印和省略选项影响。因此名称不依赖
临时文件路径、进程地址或跨编译调用的全局计数器。独立子核即使使用相同的
局部实例编号也不会直接复用同名外部符号。设备调用声明与 C++ 定义均从同一
instance symbol 派生。

entry 的 core 由最近的 `pto.section.vector` 或 `pto.section.cube` 确定；没有局部
section 时，再使用函数或模块上的 `pto.kernel_kind`。声明为具体 core 的 entry
在 core 不匹配或缺少 core 上下文时会报错。拆分 vector/cube 子模块时，只保留
当前 core 仍引用的实例。

Wrapper 按 renderer 和 core 分组生成 translation unit。原型由 registry descriptor
生成，Pipe 初始化使用调用方提供的存储，Cube Direct 使用无状态调用。生成的
wrapper 随后按目标 core 编译并合入设备 LLVM IR。

## Print 支持

Print bridge 同时支持 `pto.print` 和 `pto.tprint`。entry 根据值类型和格式串
conversion 选择，格式串和参数类型不匹配时给出诊断。当前支持的行为包括：

- f16、bf16 扩展到可打印的浮点表示；
- f64 按 DebugTunnel 的 FLOAT 记录格式截取为 f32；
- 窄整数按 signedness 扩展；
- 格式串驻留为地址空间 1 的模块常量；
- 设备侧使用固定原型调用，不生成 variadic device call。

`pto.tprint` 当前要求二维、row-major ND Vec tile，具有已规划地址和静态
`valid_shape`。支持 f16、bf16、f32，以及带 unsigned 标记的 i8/i16/i32。
动态有效尺寸、其他布局、`tmp` tile 和显式 `printFormat` 暂不支持。

DebugTunnel 参数约定通过 `--vpto-print-dtdata=implicit|explicit` 显式选择，
不会从 kernel 签名推断。两种模式都会为 kernel entry 添加隐藏尾参数；host stub
和设备编译必须使用相同约定。

## 管线位置与混合编译

Bridge 路由、Print 参数处理、实例解析和 wrapper 生成发生在 vector/cube module
拆分之前，使中层变换看到完整的结构化 bridge IR。最终的 bridge call lowering
在 VPTO LLVM 发射阶段按 core 子模块执行。

VPTO 的主线入口和无 tile operation 时使用的快路径都接入同一套 bridge 流程，因此
print-only kernel 也可以使用 Print bridge。一个 kernel 中可以同时出现：

- 继续走普通 VPTO lowering 的操作；
- 走 C++ bridge 的 Pipe、Cube 或 Print 操作；
- vector 和 cube 两个 section 中的不同 bridge family。

这种混合是按已注册的 operation 路由实现的，不支持在 PTO IR 中嵌入任意 C++ 源码，
也不支持通过策略文件声明未注册的 C++ ABI。

## 使用方式

```bash
ptoas input.pto --pto-arch=a5 --pto-backend=vpto \
  --vpto-bridge-policy=bridge-policy.yaml -o output.o
```

不指定策略文件时，使用内置默认路由。需要打印时，还应明确选择 DebugTunnel
参数约定，例如模拟器场景通常使用：

```bash
ptoas input.pto --pto-arch=a5 --pto-backend=vpto \
  --vpto-print-dtdata=explicit -o output.o
```

## 当前验证结果

Bridge 的 lit 和 C++ 单元测试覆盖路由策略、ABI、对象生命周期、Pipe/Cube/Print
lowering、wrapper 生成和非法输入。A5 `dav_3510` 模拟器验证覆盖以下端到端场景：

| Case | 验证内容 |
|---|---|
| `fifo-tile-data-consume` | cube/vector 间通过 Pipe bridge 执行 push、pop、free |
| `cube-tmatmul-bridge` | `pto.tmatmul` 经 wrapper 执行并校验矩阵乘结果 |
| `print-bridge` | 标量 print、tprint 和显式 DTData 投递 |

上述模拟器 case 均完成 compare 校验。真实 NPU 尚未完成端到端验证。

## 当前限制

- 真实 NPU 路径尚未完成验证；Print 的隐式 DTData 约定尤其依赖真实运行时投递。
- f64 打印最终只有 f32 精度，因为 DebugTunnel FLOAT 记录为 4 字节。
- `pto.tmatmul` 和 `pto.tmatmul.acc` 当前要求 tile 操作数直接来自带 planned address
  的 `alloc_tile`；view、subview、MX scale、多结果等形式尚未支持。
- pipe 弹出的 tile 不能直接作为当前 bridged cube 操作的输入。
- Cube 默认关闭，以避免未准备 wrapper 工具链的存量 cube kernel 被自动切换到
  bridge 路径。
