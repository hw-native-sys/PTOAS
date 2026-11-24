# Linx 独立前端 IR、校验与 C++ Lowering 设计

> 状态：静态 local f32 的 TLOAD/TADD/TSTORE、TMATMUL 与 softmax 独立链路已实现；其他阶段仍为规划。
> 基线：GitCode `cann/pto-as` master，`64deb0300d`。
> 目标：为 Linx 建立独立的前端语义和编译链，不再通过 A3/A5 的 PTO tile IR 与架构条件分支承载 Linx。

## 1. 目标与边界

新增 Linx 专用的 PTO dialect 实现，独立定义类型、属性、操作、verify、分析和到 EmitC 的转换，最终调用 pto-isa 的 Linx C++ API。**文本命名继续使用 `pto.txxx`、`!pto.tile_buf` 等 `pto` 命名；内部实现与 A3/A5 隔离。** A3/A5 继续使用已有 PTO dialect 和编译链。

这里的“独立”指目标语义和转换实现独立，不要求复制 MLIR、另写 C++ 打印器或另造命令行程序。`func`、`arith`、`scf`、PassManager、诊断和通用 EmitC C++ 翻译设施可以复用；复用前提是它们不包含 Ascend 地址空间、布局、同步或 TileLib 规则。

第一阶段打通 vadd、matmul、softmax 三条端到端路径。此后以两份 Linx 手册为功能清单，按实际 pto-isa API 的可用性逐批扩展。解析某个 op 不代表支持其所有 dtype、layout、动态形状和多 PE 组合。

参考资料：

- `linx-tile-ir-manual.md`：执行模型、布局、容量、矩阵角色与 Shared 协作。
- `linx-tile-ir-op-reference.md`：操作清单、逐 op 约束与待确认问题。
- pto-isa Linx C++ API：作为版本固定的 lowering 契约和编译运行验证后端。

前两份手册使用的 `pto.*` 和 `!pto.*` 拼写保持不变。本方案继承其中经确认的 **Linx 语义**，不承诺直接接受原文全部参数组合，也不把手册中的待确认项当成已确定规范。同名操作在不同目标下可以有独立签名和限制，文本名称相同不意味着共用 ODS、C++ 类或 verifier。

## 2. 为什么独立建模

| 维度 | A3/A5 PTO IR | Linx IR 的建模方式 |
|---|---|---|
| tile 存储 | 有地址的片上 buffer，区分 vec/mat/left/right/acc 等 | 无软件地址的 local/shared tile handle |
| 分配与生命周期 | 地址分配、内存复用等 | 声明寄存器对象及语义生命周期；实际寄存器分配交给后端 |
| 矩阵角色 | 存储层级表达 Left/Right 等 | 前端不暴露角色；按 C++ API 约束推导类型角色 |
| 布局 | Ascend 专用布局及 fractal 约束 | row_major/m16/m32/n8 和 CELL 容量规则 |
| op 签名 | 可能有 tmp、fixpipe 等 Ascend 参数 | 按 Linx API 独立定义；不保留无意义的占位参数 |
| 执行与同步 | Ascend pipeline、同步与事件规则 | 保留 Linx 协作、可见性和副作用；不生成 Ascend flag/barrier |
| GM 描述 | 既有 tensor/view 体系 | 独立二维 GM descriptor 与明确的元素 stride 单位 |

例如 Linx 的行归约无需 Ascend scratch tile，应直接定义不含 tmp 的 op，而不是在共享 op 中将 tmp 设为可选并按目标切换 verifier。前端禁止地址属性，也不通过“地址为零”的占位约定进入通用内存规划。

## 3. 代码与依赖边界

Linx 按与 VMI 类似的功能层级归入 PTO 目录树；目录归属不改变独立 ODS、verify 和 lowering 的边界。建议增加以下目录和库（库名称为拟议名称）：

```text
include/PTO/IR/Linx/
  LinxDialect.td  LinxTypes.td  LinxAttrs.td  LinxOps.td
  LinxDialect.h   LinxTypes.h   LinxOps.h
include/PTO/Transforms/Linx/Passes.td
include/PTO/Conversion/Linx/LinxToEmitC.h
lib/PTO/IR/Linx/                  # LinxIR：parser/printer、类型与 op verifier
lib/PTO/Analysis/Linx/            # LinxAnalysis：状态、别名、角色和 uniformity
lib/PTO/Transforms/Linx/          # LinxTransforms：合法化与语义校验 pass
lib/PTO/Conversion/Linx/LinxToEmitC/
                             # LinxToEmitC：类型转换、patterns、ABI、API 映射
lib/PTO/Pipelines/Linx/           # LinxPipelines：独立 pipeline 构造
test/linx/
  IR/  Verify/  Analysis/  Conversion/  Driver/  Runtime/
```

`LinxIR` 不依赖 `PTOIR`；Linx 转换库不依赖 `PTOToEmitC`、ExpandTileOp、PlanMemory、Ascend 同步或 PTODSL TileLib。工具可同时链接两套实现的库，但必须按目标为每个 MLIRContext 选择其中一套注册；不能把两套同名 `pto` dialect 同时加入同一个 registry/context。

不得在已有 PTO op 的 verifier、PTO 类型、地址空间枚举或逐 op lowering 中新增 Linx 分支。若当前 EmitC 文本翻译入口夹带 PTO 特定逻辑，只抽取与目标无关的打印功能，Linx 的 include、类型别名和 API 调用构造仍归 Linx 转换库。

### 3.1 同名语法与独立内部实现

| 项目 | A3/A5 | Linx |
|---|---|---|
| 文本 dialect namespace | `pto` | `pto` |
| 文本操作名 | `pto.tadd` 等 | `pto.tadd` 等，保持命名 |
| C++ namespace | 保留现有命名空间 | 独立 `mlir::pto::linx` |
| C++ dialect/op/type 类 | 现有类 | 独立类与 TypeID，例如 `linx::PTODialect`、`linx::TAddOp` |
| ODS 与生成文件 | 现有 PTO 定义 | 独立 TableGen 记录名、生成目标及 include 路径 |
| verify/分析/lowering | 现有编译链 | Linx 专用编译链 |

Linx 的 ODS dialect 设置文本名称 `pto`，C++ namespace 设置 `::mlir::pto::linx`。TableGen 记录、生成文件、库符号和 pass 注册名均需隔离；例如 lowering pass 使用 `linx-pto-to-emitc`，不能重复注册原 pass 名。patterns 通过 Linx C++ Op 类型匹配，不通过字符串 `pto.tadd` 猜测目标。

C++ 类完整名称使用 `mlir::pto::linx::PTODialect`、`mlir::pto::linx::TAddOp` 和 `mlir::pto::linx::TileBufType`。类的短名可以与原 PTO 实现相同，靠命名空间和独立 TypeID 区分；不另建大小写不同的 `PTO::Linx` 命名空间。若调用方需要短写，可在局部使用 namespace alias；作用域分隔符始终为 `::`。

### 3.2 Context、前端范围与序列化边界

一个 context 只加载一种 PTO 实现。需要在同一进程处理两个目标时，使用两个独立 context 和 pipeline；不能把两套 IR 对象跨 context 直接传递。首阶段不支持一个 module 混合两种目标，即使函数分别带有 target 属性也不支持。

第一阶段不实现 Python bindings 和 CAPI 前端。测试直接编写 MLIR 层级的文本 IR 文件（扩展名规定为 `.pto`），由原生工具选择 Linx 实现后解析、verify、lowering，再对生成的 C++ 进行远端 CPU-sim 验证。后续如扩展 bindings/CAPI，必须显式选择 context 对应的目标实现，禁止自动注册两套同名 dialect。

同名 dialect 的文本名称和 bytecode 中的 namespace 均不足以区分实现。文本输入依靠解析前的目标选择，module target 用于解析后的交叉检查；工具日志与复现命令必须携带目标。已有 PTOBC 不自动兼容 Linx；定义 dialect 版本、编码和解析前可读取的目标标识之前，Linx 路径明确拒绝该输入。所有独立 opt/翻译工具也遵循同样的注册规则。

## 4. 前端 IR

### 4.1 类型、布局与数据类型

初始静态 local f32 类型形式如下；当前 vadd 路径支持这些类型和文件头类型别名。

```mlir
!X = !pto.tile_buf<local, 16x128xf32, blayout=row_major>
!R = !pto.tile_buf<local, 16x1xf32, blayout=row_major>
!G = !pto.partition_tensor_view<16x128xf32>
```

- `tile_buf` 描述存储类、静态物理形状、元素类型、布局、有效区域及必要的填充语义；没有 addr、Ascend slayout/fractal 和 Left/Right location。
- `partition_tensor_view` 表示二维外部内存窗口；descriptor 携带 base、行列尺寸、元素 stride。第一阶段限制静态尺寸、内维 stride=1，行 stride 允许通过明确 ABI 传入；更窄的 API 限制由 capability 校验拒绝。
- `valid` 默认等于物理形状。第一阶段只支持静态 valid；后续使用显式操作数表示动态 valid，不能依赖解析器全局 target 状态改变类型合法性。
- 容量计算由 Linx 专用工具实现：先按布局计算所需字节，再向上取整到 2 的幂，最小 128 B、最大 256 KiB。不能直接用 `R*C*sizeof(dtype)` 代替所有布局的容量。`16x1xf32` 的逻辑数据为 64 B，物理容量为 128 B。
- `TileBuf` 的 `location` 类型参数使用 Linx 独立 ODS 枚举 `TileLocation`，包含 `local` 和 `shared`；参与类型唯一化、相等比较和 IR 往返打印。当前 C++ lowering 仅支持 `local`，对 `shared` 显式报错，不能隐式映射为 Vec。
- local/shared 是硬件存储类；C++ API 的 Vec/Left/Right 是类型角色，两者不能互相替代。

MLIR builtin 的 `f32/f16/bf16` 等基础类型可复用，但合法 dtype 组合、signedness、packing、C++ 名称与数值语义必须独立映射。不能认为双方都写 `i8` 就可直接无条件复用转换：需明确 signless 整数的有符号解释、无符号类型和标量扩展规则。packed FP4 等还需区分逻辑元素数量、存储字节数和 GM 索引单位。

特殊格式优先使用语义准确的 MLIR builtin 类型；确实无法表达时新增 Linx 自有类型，使用独立 Linx C++ 类型实现，文本仍归属 `!pto.*`，不依赖 A3/A5 的类型类。TF32 等可能属于运算精度模式，必须按后端契约判断，不能一概定义成存储类型。首批只开放经验证的 f32 组合。

### 4.2 操作与状态模型

采用 **SSA handle + 显式 allocation + destination passing**，便于准确对应 C++ Tile 对象和写入 API。SSA handle 标识对象，不表示对象内容不可变；赋值、初始化、视图别名和发布状态必须另行建模。


```mlir
module attributes {pto.target_arch = "linx", pto.linx_profile = "cpu-sim"} {
  func.func @vadd(%ga: !G, %gb: !G, %gc: !G) {
    %a = pto.alloc_tile : !X
    %b = pto.alloc_tile : !X
    %c = pto.alloc_tile : !X
    pto.tload ins(%ga : !G) outs(%a : !X)
    pto.tload ins(%gb : !G) outs(%b : !X)
    pto.tadd ins(%a, %b : !X, !X) outs(%c : !X)
    pto.tstore ins(%c : !X) outs(%gc : !G)
    return
  }
}
```

`partition_tensor_view` 的函数参数是逻辑 descriptor，具体外部 C++ ABI 见第 7 节。每个 op 单独定义跨操作数 shape、dtype、valid 与 layout 规则；不能把全部 op 简化为“三个 tile 类型完全相同”。后端模板确实要求单类型的 op 才施加完整类型一致性约束。

读写操作实现 MLIR 副作用接口：tile 读取/写入、GM 读取/写入与分配应有相应 effect；视图必须追踪到父对象，不能因换了 SSA handle 就被认为不别名。第一阶段采用保守 effects 和优化白名单，禁止在未完成别名/状态分析前对这些 op 做 CSE、LICM、任意复制或删除。仅添加 effect 接口不足以证明所有目标优化都安全。

首批新 op：`alloc_tile`、`tload`、`tstore`、`tadd`、`tmatmul`、`trowmax`、`trowexpandsub`、`texp`、`trowsum`、`trowexpanddiv`。矩阵累加另设 `tmatmul_acc`，以读写累加器表示，不能按普通 matmul 的只写目标处理。

### 4.3 控制流与函数边界

首批限定单个入口函数、直线代码、无 tile handle 返回/逃逸，不承诺接受任意 `scf` 或 `cf`。普通标量运算只开放已有 lowering 的子集。

后续支持 `scf.if/for` 时，必须同时实现状态的路径合并、循环不动点、别名传递与 lowering，才开放对应语法。分支合流采用“所有可能路径均满足”规则；循环可能执行零次，不能把循环体里的写入当作循环后必然初始化。单纯检查写入支配读取不足以处理覆盖、再写和跨迭代状态。

## 5. 分层 verify

| 层次 | 负责内容 | 示例诊断 |
|---|---|---|
| 类型 verifier | 存储类、布局、维度、容量和 valid 的固有合法性 | `valid_col=129 exceeds physical columns=128` |
| Op verifier | 操作签名与局部语义，不依赖特定 API 版本 | matmul 的 K 不匹配；归约输出行数错误 |
| 函数/状态分析 | tile 别名、读前写、初始化覆盖、生命周期与非法逃逸 | tstore 读取未初始化的输出 |
| 目标 capability 校验 | 当前固定后端版本是否能实现该合法 IR 组合 | cpu-sim profile 尚不支持此 dtype/layout |
| 降级合法性检查 | 所有允许的 IR 均已转换，无残留 Linx/PTO op 或未解决类型 | 某 op 没有已注册的 lowering |

局部 verifier 不读取全局可变架构状态；需要目标信息的规则通过 module target/profile 或显式 pass 配置传递。所有入口都强制执行必需的语义和 capability 校验，不提供以关闭通用 verify 为由绕过这些门禁的 C++ 发射路径。

状态分析第一阶段检查直线代码的已定义有效区域与读取区域包含关系。in-place、输入输出重叠、partial write 只按逐 op 已确认规则允许；无法证明安全时拒绝并给出原因。动态范围后续区分编译期可证约束、可生成的运行时检查和调用方契约，不能声称静态 verifier 能证明任意运行时边界。

错误应标出失败 op、相关 tile 定义和冲突使用点，并建议可行修改。不同 profile 支持能力不同应报告“不支持的后端组合”，避免误称为 IR 语法错误。

## 6. Shared、subview 与 assemble

这部分需要独立状态模型，但不是第一阶段闭环的前置条件。

PTOAS 负责表达并保留 PE mask、参与一致性、读写窗口、generation 与发布约束，检查静态可证明的违规。pto-isa/Linx 编译器负责 API 到 intrinsic/指令、寄存器分配、bundle 调度以及硬件可见性机制。PTOAS 不实现硬件调度，也不照搬 Ascend 软件同步。

- Shared 的 `insert` 和 `publish` 必须可区分；“写过”不等于“可读”。状态至少包含 generation、已定义区域/quarter 和是否发布；开始新 generation 时不能沿用旧 mask。
- 多 PE 操作的 collective/convergent 契约不能只靠一个 trait 实现。必须定义参与集合、控制流一致性与 effect，防止移动、复制或让部分参与者跳过。控制流一致性无法证明时，拒绝相关多 PE 组合。
- `subview` 表达只读源窗口，必须保持父对象别名、布局映射、CELL 对齐与可见性；不能绕过父 tile 的初始化/发布限制。
- `assemble` 应体现一段完整写入事务：INIT 创建/开启父对象，MIDDLE 延续，LAST 完成并发布；LAST 前不得读取父对象及其 subview。不能暴露一个可被普通算子读取的“半成品 subtile”。
- 建议独立 assemble region 或不可作为普通 tile 输入的写会话 token；最终形式需先与后端 API/intrinsic 契约对齐，再决定 ODS。不能先按普通纯 SSA 返回值实现，再补 side effect。
- shared slot 是后端接口的资源约束，不默认当成语义 shape 的一部分。若跨 region 需要显式 slot，应引入资源绑定属性并检查活跃区间冲突；不自动继承手册草案中的所有 slot 假设。

手册的 Shared 布局、可用 mask、分段发布和跨 region 生命周期部分尚引用外部待确认评审；群内 assemble 讨论也不是完整规范。以上能力须逐项取得版本化契约和测试后开放。

## 7. 独立 lowering pipeline

```text
CLI 选择 Linx 输入模式
  → 创建只注册 Linx PTO 实现的 context（文本 namespace 为 pto）
  → 解析 Linx PTO + 允许的通用 dialect
  → 类型/op verify + module 边界检查
  → Linx 初始化/别名/生命周期校验
  → 目标 capability 校验
  → Linx 角色推导与 C++ ABI/type 规划
  → LinxToEmitC 全量转换
  → 转换合法性与残留检查
  → 通用 EmitC → C++ 翻译
  → 远端 C++ 编译、链接、CPU-sim 运行
```

不得经过 Ascend tile normalization、TileLib 展开、地址规划、同步插入或既有 PTOToEmitC 的逐 op patterns。每个转换阶段只接受自己声明支持的操作集，禁止依赖 `allowUnregisteredDialects` 透传未知 op。

### 7.1 CLI 与目标选择

沿用一个 `ptoas` 可执行文件，优先沿用现有架构选择选项，例如 `--pto-arch=linx`；实际选项拼写与现有 CLI 对齐。CPU-sim profile 另行显式选择。**架构选择必须发生在创建 registry/context 和 parse/verify 之前**，不能等正常解析完 module target 后才切换 dialect，也不能以修改全局状态的方式让共享 Op verifier 改变语义。未知目标直接报错。

```text
选择 A3/A5 → 原 PTO dialect/context → 原 verifier → 原 pipeline
选择 Linx  → Linx PTO dialect/context → Linx verifier → Linx pipeline
```

工具入口的目标分派是允许的；每个 op 的定义、verify 和 lowering 不再按架构分支。未显式选择架构时保持现有 A3/A5 CLI 默认行为，Linx 输入必须显式选择 Linx；不能仅凭 `pto.*` 名称自动识别。

保留 module 的 `pto.target_arch = "linx"` 作为交叉检查，profile 可用 `pto.linx_profile = "cpu-sim"` 字符串属性描述。module 属性与 CLI 显式值冲突时报错。目标一经用于构造 context 就不可切换；module 属性不驱动第二轮解析。Linx context 按自己的定义拒绝 Ascend 地址、同步和其他不支持的 op/类型组合，不能因为名字都叫 `pto.*` 就放行。`func/arith/scf` 是否允许由阶段支持清单确定。

### 7.2 C++ 类型角色

角色分析以 allocation/别名身份及后端模板类型等价约束为基础；从 matmul 的 lhs/rhs 等使用点收集 Left/Right 要求，通过单类型 op 和视图传播，默认 Vec。目标 API 不约束的角色不能凭名称强行推断。

local 角色要求冲突时诊断两个使用点，不通过字符串替换、伪装 layout、未解释的 `unrealized_conversion_cast` 或隐式数据复制“修复”。Shared 仅在后端确有支持且语义吻合时生成角色视图。控制流开放后，角色求解必须覆盖 block 参数和循环传递。

结果存放在独立 analysis/lowering context 中，由 LinxTypeConverter 构造一次确定的 C++ 类型。前端 tile 类型不加入 Left/Right 字段，也不以私有 PTO 属性作为跨 pass 协议。

### 7.3 ABI 与 API 映射

首批使用 `#include <pto/pto-inst.hpp>`，选择 Linx CPU-sim 的具体宏、编译参数、链接依赖记录在验证配置中。禁止同时包含两个实现相同名称的 TileOP 后端。

外部 kernel ABI 建议是 `extern "C"` 函数：GM 参数展平为 dtype 指针和必要的行 stride（元素单位），静态尺寸在函数内构建 pto-isa `GlobalTensor` descriptor。读指针的 const 规则、参数顺序、整数宽度、符号名和 descriptor stride 填充规则均由 Linx ABI 模块唯一生成，host harness 从同一契约调用。不能把 pto-isa C++ 模板 descriptor 类型误当作稳定的 C ABI。

| Linx IR | C++ API/行为 |
|---|---|
| `alloc_tile` | 声明分析后确定的 local Tile 对象 |
| `tload/tstore` | TLOAD/TSTORE 与二维 GM descriptor |
| `tadd` | TADD(dst, src0, src1) |
| `tmatmul` | TMATMUL(dst, lhs, rhs)，以固定版本真实签名为准 |
| `trowmax/trowsum` | TROWMAX/TROWSUM，无 Ascend tmp |
| `trowexpandsub/trowexpanddiv` | TROWEXPANDSUB/TROWEXPANDDIV |
| `texp` | TEXP |

建立可审查的 API capability 表，逐项记录 op、dtype、布局、valid、in-place、API 符号、profile、后端 commit 和验证用例。不要用一份简单的 op 名称白名单表示完整能力。后端升级需要重跑签名编译和数值用例。

## 8. 迁移与兼容

用户继续使用 `pto.txxx`、`!pto.tile_buf`、`!pto.partition_tensor_view` 和已有类型别名写法，无需进行 `pto` 到 `linx` 的名称迁移。已有 A3/A5 文件继续走原编译链。

| 既有 Linx 表示 | 新链路处理 |
|---|---|
| `!pto.tile_buf<local,...>` | 保留拼写，由 Linx 自有类型 parser/verifier 处理 |
| `!pto.partition_tensor_view<...>` | 保留拼写，按 Linx 二维与 stride 契约检查 |
| `pto.alloc_tile/tadd/...` | 保留名称，由独立 Linx Op 定义与 lowering 处理 |
| `pto.target_arch="linx"` | 保留属性；命令行必须在解析前选择 Linx 实现 |

保持名称不代表全部旧输入天然兼容。需要逐项核对参数、默认值、类型语法和状态语义；例如归约的无 tmp 签名由 Linx 独立定义。确需转换的旧参数提供显式迁移工具和诊断，不通过全局文本替换或复用 Ascend verifier 放行。

新旧 Linx 路径不能在同一 context 同时注册。在新链路达到验收条件之前，如需保留原路径，应通过明确的实验性入口/配置选择不同 context；切换默认入口、弃用/删除旧共享路径另做明确变更。

## 9. 分批实施与验收

| 阶段 | 交付 | 退出条件 |
|---|---|---|
| P0：规范与接口固定 | 新类型/op 签名、CLI/ABI、后端 commit、能力矩阵 | 三条闭环所需组合全部有明确签名与语义，无待确认项阻塞 |
| P1：独立骨架与 vadd | LinxIR、verify、pipeline、LinxToEmitC、GM ABI | 独立 IR 可解析/往返；负例正确拒绝；远端 vadd 编译运行通过 |
| P2：矩阵与角色分析 | m16/m32/n8、TMATMUL、角色冲突诊断 | M16/M32 矩阵用例与布局转换数值验证通过，K/shape/角色负例覆盖 |
| P3：softmax | 五个归约/广播/指数 op | 稳定 softmax 与高精度参考比较，覆盖非平凡 stride 和边界数据 |
| P4：扩展单 PE 能力 | 已实现 API 优先，扩展 dtype、valid、标量和其他 op | 每项均有语义、lowering、编译与数值证据；未支持组合明确拒绝 |
| P5：控制流与协作 | region 数据流、Shared、多 PE、subview/assemble | 后端契约确认，状态与一致性负例完备，匹配执行模型的验证通过 |
| P6：手册清单收口 | 两份手册全量覆盖与绑定/文档 | 每个条目具有状态、限制、测试与责任方；没有仅解析无实现的“完成项” |

P4 内部先接入固定后端已有完整实现和 CPU-sim 入口的 API，再接入仅部分 dtype/layout 实现的 API；尚无 wrapper 或语义未定的项目单独跟踪后端依赖。CPU-sim 未实现不等于 IR 不合法，但不能将该组合宣称为可运行支持。

## 10. 验证计划

所有构建、测试和 CPU-sim 执行都在远端 worker/CI 进行，本地仅编辑、检索和 Git 操作。当前可用 worker 为 `fangrui@123.60.231.48:1211`；执行前重新确认编译器、LLVM/MLIR、CPU 架构、pto-isa commit 与 profile，不把旧运行结果当作新实现的验收结果。

- IR：parse/print、类型别名、非法容量/valid/layout/dtype、签名和不支持组合。
- 状态：读前写、in-place 别名、partial write、角色冲突；开放控制流后增加零次循环和路径合并反例。
- lowering：检查类型与 API 参数顺序、无 tmp、ABI/stride、无残留 Linx/PTO op 或未解决 cast；对输出 C++ 真正编译链接。
- vadd：非恒定输入、非连续行 stride、输出 guard 与输入不变性。
- matmul：M16/M32 与 N8 的布局敏感输入，独立标量参考、非对称矩阵；误差标准按实际运算精度制定。
- softmax：减行最大值的稳定算法；大正负值、重复最大值、随机数据，与 f64 参考逐元素比较，同时检查行和、非负性、越界写。NaN/Inf 单列 API 语义测试，不能默认与有限输入同一标准。
- 隔离：远端检查构建依赖和 pipeline，运行既有 A3/A5 回归，确认默认 CLI 和公开 PTO IR 未改变。补充同名 op 在两种 context 下的签名差异、CLI/module 目标冲突、同一进程双 context、不支持 bytecode 输入的负例；bindings/CAPI 的注册测试留到对应前端实现阶段。
- 多 PE：单 PE 的 CPU 数值通过不能证明 collective/可见性正确；需要具有匹配执行模型的仿真或目标验证，另设验收门禁。

首批验收应同时交付三套 MLIR 文本输入（`.pto`）、生成 C++、host harness、固定依赖信息、远端复现命令和结果日志。只有这三条路径在新 dialect/new pipeline 下重新通过，才能视为独立架构的首个可用版本。

### 10.1 验证配置的位置与状态

首批验证清单已新增到 `test/linx/Runtime/config/cpu-sim.json`，复现命令见 `test/linx/Runtime/README.md`。它记录依赖和编译参数，不是 module 的 `pto.linx_profile` 属性，也不负责放宽编译器 capability 限制。

验证文件按以下位置组织（matmul、softmax 仍为后续规划；编译回归用例放在现有 lit 自动发现的 `test/lit/pto/linx/`）：

- `test/linx/Runtime/config/cpu-sim.json`：版本化的后端仓库/commit、CPU-sim 宏、编译与链接参数、相对 include/library 路径、精度判定标准；参数按数组保存，不把任意 shell 命令放进配置。
- `test/linx/Runtime/README.md`：远端环境准备、运行入口和三条用例的复现命令。
- `test/linx/Runtime/{vadd,matmul,softmax}/`：直接输入的 `.pto`、host harness 与参考结果生成逻辑。

SSH 主机、端口、本地后端 checkout 和工具链绝对路径属于机器环境，通过远端环境变量或不提交的机器配置传入；不在仓库保存密码。编译器内部的 capability 定义由 Linx 库维护，不能由运行配置随意放宽。每次运行日志另记录实际解析后的依赖 commit、编译器版本、参数与结果，保证可复现。

## 11. 需要先确认的事项

1. 已确认：公开命名保留 `pto.*`，内部实现独立；实施前核对 CLI 解析时序、各原生工具的 registry，保证同名 dialect 的互斥注册。
2. pto-isa 哪个 commit 作为实施基线，以及 CPU-sim/真实 Linx 目标的能力差异；不能只依赖一个不断变化的 PR 编号。
3. 特殊 dtype、取模符号和 gather/scatter 索引单位：参考文档已提示冲突，须逐项裁定，不从 A3/A5 继承默认行为。
4. Shared 发布与 assemble 的规范、wrapper 和副作用契约：契约确定前保持未开放。

这些问题不阻塞 P1 的静态 f32 local vadd 骨架设计；但各自对应功能在答案确定前不能进入“支持完成”状态。

## 12. 本次最小链路的实施范围

新增独立 `LinxPTOIR` 和 `LinxPTOToEmitC` 库；dialect、类型和操作分别定义在 `LinxDialect.td`、`LinxTypes.td`、`LinxOps.td`；类型存储、构造器与访问器由 ODS TypeDef 生成，op 直接使用类型约束，不使用 `AnyType`。保留现有语法的类型 parser/printer 和类型参数校验位于 `LinxTypes.cpp`，op 关系校验位于 `LinxDialect.cpp`，状态检查位于 `LinxVerification.cpp`。这些库不依赖 Ascend `PTOIR` 或 `PTOToEmitC`。后续功能增多时再按第 3 节拆分类型、分析和 pass 文件。

`ptoas --pto-arch=linx` 在解析前选择独立 context，保留 `pto.*` 名称。当前 CLI 启动器仍复用已有 Python 包，但 Linx 文本解析和编译使用自己的原生 context，不新增 Linx Python bindings/CAPI。

当前开放 `alloc_tile → tload → tadd/tmatmul → tstore`，静态二维、local、f32、full-valid、直线函数。TADD 支持相同类型的 row-major/CUBE tile；TMATMUL 的 A 为 M×K（m16/m32），B 为 K×N（n8），D 为 M×N，D 与 A 布局相同且不得与输入别名。m16 的 M≤16，m32 的 M≤32。输入输出 GM shape 必须与 tile 匹配；读取前必须初始化。其他属性、dtype、布局、函数调用及 in-place TADD 均拒绝。GM ABI 为每个参数对应 `(float *base, int32_t row_stride_elements)`，内维连续；buffer 有效范围和 GM 输入输出不重叠由调用方保证。

转换先构造独立 EmitC module 并执行 MLIR verifier，再使用通用 EmitC C++ 翻译器。固定后端的基础 API 使用全局 `::TLOAD`、`::TADD`、`::TSTORE`，类型使用 `pto::Tile` 和 `pto::GlobalTensor`。尚未接入 matmul、softmax、Shared、多 PE 或控制流，不能将旧编译链的验证结果计入这些功能。


### 12.1 Local TMATMUL 链路

布局由 Linx 独立 ODS 枚举 `TileLayout` 保存，支持 `row_major/m16/m32/n8`；默认 row-major。类型校验按 f32 CELL 几何计算物理包络（M16=16×2、M32=32×1、N8=4×8），容量不超过 256 KiB。

```mlir
!A = !pto.tile_buf<local, 16x8xf32, blayout=m16>
!B = !pto.tile_buf<local, 8x8xf32, blayout=n8>
!D = !pto.tile_buf<local, 16x8xf32, blayout=m16>
// a/b/d 来自独立 alloc_tile，a/b 已通过 tload 初始化。
pto.tmatmul ins(%a, %b : !A, !B) outs(%d : !D)
```

`LinxPTOAnalysis` 从 TMATMUL 使用点收集 Left/Right 约束，并沿 TADD 的单一 C++ 类型约束传播；未约束的 allocation 默认为 Vec。结果仅用于 C++ 类型选择，不写回 IR、不改变 local/shared，也不插入隐式复制。不同 allocation 即使 IR 类型相同，也可具有不同 C++ 角色。TMATMUL 的 D 没有固定角色要求，允许后续用途推导它的角色。

lowering 生成 `::TMATMUL(dst, lhs, rhs)`，TLOAD/TSTORE 根据 tile 的 CUBE 布局完成 ND 与 CELL 布局转换。状态验证将 A/B 视为读、D 视为写；普通 TMATMUL 不要求 D 已初始化。当前尚不支持 shared/cooperative、group_m、ACC、bias、MX、后处理或非 f32 矩阵。


### 12.2 Softmax 链路

新增 `trowmax`、`trowexpandsub`、`texp`、`trowsum`、`trowexpanddiv`，使用 Linx 独立 ODS 与 verifier。按行计算 `exp(x - rowmax(x)) / rowsum(exp(x - rowmax(x)))`，输出形状与输入相同。

五个 op 均要求 local f32 tile，当前使用独立输出 allocation。行归约的输出是 R×1；行广播的第二个输入是 R×1，完整输入与输出类型相同；归约和广播的行数及布局必须一致。TEXP 要求输入输出类型相同，其 C++ 角色约束与 TADD 一样传播。归约及广播各操作数的角色不强行绑定。支持已有 row-major 和 CUBE 布局类型，不增加 scratch 操作数。

lowering 分别调用 `::TROWMAX(dst, src)`、`::TROWEXPANDSUB(dst, src, row)`、`::TEXP(dst, src)`、`::TROWSUM(dst, src)`、`::TROWEXPANDDIV(dst, src, row)`。所有输入按读处理，输出按写处理，读取前必须初始化。非有限输入的 softmax 行为不额外定义；数值验收使用有限输入。
