# VPTO Pipe PTODSL 与 TileLib 边界设计

**Issue：** [#966](https://github.com/hw-native-sys/PTOAS/issues/966)
**状态：** 编译器侧实现完成；TileLib 实现作为独立后续工作
**目标：** A5 VPTO 后端
**本次变更负责：** PTODSL pipe 表层校验、PTO IR、PTOAS pass、
`ExpandTileOp` 契约生产、VPTO LLVM lowering 和聚焦的前端覆盖
**本次不负责：** `ptodsl/tilelib/**`、TileLib template 测试，以及
TileLib-ST/runtime 覆盖

## 1. 目的与范围

PTODSL 已可将表层 pipe 抽象 lower 为统一的 PTO pipe operation。VPTO 路径不能
保留 EmitC 所使用的 C++ `TPipe` object，因为在进入 LLVM lowering 前，VPTO 需要
在 PTO IR 中显式表示 FIFO 状态、同步和数据搬运。

本文记录当前已经落地的 PTOAS 编译器侧边界。它同时是编译器侧实现设计与 TileLib
owner 的交接契约。本文刻意不宣称：在没有相应 TileLib 改动的源树中，GM 或 local
split pipe 已可端到端运行。

稳定的前端到后端流程如下：

```text
PTODSL Pipe object
  -> frontend pipe operation
  -> 统一 PTO pipe operation
  -> 推导/校验 pipe nosplit 配置
  -> materialize PipeState 与结束 tdrain
  -> 内存规划和 reserved-buffer 解析
  -> pipe metadata/candidate discovery
  -> ExpandTileOp PipeSpec ABI
  -> 外部 PTODSL TileLib template
  -> VPTO LLVM lowering
```

## 2. 目标

- 保持已有 PTODSL Pipe API 及其逐 transaction 的 `split` 值。
- 为 A5 VPTO 路径显式表示可变 FIFO state。
- materialize terminal producer cleanup，且不从任意一条 `tpush` 推导其策略。
- 在内存规划后，向 `ExpandTileOp` 提供完整、已解析的 pipe metadata 与资源 operand。
- 在 A5 VPTO backend 之外，保持 EmitC 的 `TPipe` 路径不变。
- 提供收敛的 ABI，使独立开发的 TileLib 可消费该信息，而不依赖 opaque
  `!pto.pipe` value。

## 3. 非目标

- 替换既有 EmitC pipe 实现。
- 新增一套并行的 PTODSL `pipe.push()` / `pipe.pop()` API，或重新设计已有 API。
  `pto.pipe.c2v(...)`、`pto.pipe.v2c(...)` 和 `pto.pipe.bidirectional(...)`
  已返回带有这些方法的 Pipe object。
- 在本次变更中修改 `ptodsl/tilelib/**`。
- 在 TileLib 实现和 TileLib-ST/simulator 覆盖合入前，宣称 GM/global 或 local
  split 的运行时正确性已经完成。
- 支持 A2/A3 VPTO pipe expansion、`v2c_ctrl`、fixpipe quantization 或
  phase/NZ conversion。

## 4. PTODSL Pipe 表层

### 4.1 已有公开 API

已有 constructor 返回带方向信息的 Pipe object：

```python
pipe = pto.pipe.c2v(
    id=7,
    gm_slot_tensor=slots,
    slot_size=1024,
)
pipe.init_cube()
pipe.init_simd()
entry = pipe.alloc(split=1)
pipe.push(entry, split=1)
entry = pipe.pop(split=1)
pipe.free(entry, split=1)
```

`v2c(...)` 提供相同的 transaction 方法。`bidirectional(...)` 在执行带方向的
transaction 前通过 `.c2v`、`.v2c` 暴露 endpoint。Pipe API 要求稳定、显式的 `id`。

公开方法直接 lower 到相应 frontend operation：

| Pipe 方法 | lower 后的 frontend operation family |
|---|---|
| `alloc(split)` | `talloc_to_aiv` 或 `talloc_to_aic` |
| `push(entry, split)` | `tpush_to_aiv` 或 `tpush_to_aic` |
| `pop(split, ...)` | `tpop_from_aic` 或 `tpop_from_aiv` |
| `free(entry, split)` | `tfree_from_aic` 或 `tfree_from_aiv` |

`split` 是 operation property，不是 Pipe-object property。因此同一个
split-capable pipe 可以先执行 `split=1` transaction，再执行 `split=2`
transaction，但仍须满足既有 pipe configuration 校验。

### 4.2 GM slot-size 规则

对于 GM-entry pipe，只有 `nosplit=True` 时才能无歧义地推导 `slot_size`：此时一个
`gm_slot_tensor` shape 就是一个完整 FIFO slot。对于 split-capable GM pipe，调用方
必须显式传入完整 slot 的 `slot_size`。这避免把 split subregion 误认为 FIFO slot size。

### 4.3 Pipe 操作与 template 边界

PTODSL 源码和 frontend helper 使用公开的 `Pipe` object。其完整 operation surface
如下：

| 类别 | 公开 surface | 契约 |
|---|---|---|
| constructor | `pto.pipe.c2v(...)`、`pto.pipe.v2c(...)`、`pto.pipe.bidirectional(...)` | 创建 C2V、V2C 或双向 logical pipe。`id` 必填，且始终是稳定的 pipe identity。 |
| initialization | `init_cube()`、`init_simd()` | 从 Cube 或 SIMD 侧初始化 pipe。双向 pipe 使用 root object 初始化。 |
| producer transaction | `alloc(split=0)` | 仅适用于 global-entry pipe；返回下一个 FIFO entry 的 descriptor。 |
| producer transaction | `push(entry, split=0)` | 将填充完成的 global entry 或 local tile 发布给 consumer。 |
| consumer transaction | `pop(split=0, result_type=None, valid_shape=None, valid_row=None, valid_col=None)` | 返回下一个 global-entry descriptor 或 local tile。local tile-entry pipe 必须提供 `result_type`；`valid_shape` 与 `valid_row` / `valid_col` 互斥。 |
| consumer transaction | `free(entry=None, split=0)` | 释放已消费的 FIFO entry。global-entry pipe 必须传入相应 `pop` 返回的 entry；local tile-entry pipe 可省略。 |
| read-only property | `id`、`slot_size`、`entry_type` | 暴露 compile-time identity、完整 logical slot 的 byte size，以及适用时的 global-entry descriptor type。 |

`c2v` 和 `v2c` 只暴露方向合法的 transaction。双向 pipe 必须在调用 `alloc`、
`push`、`pop` 或 `free` 前选择 `.c2v` 或 `.v2c`；其 root object 没有无歧义的
transaction direction。`split` 是逐 transaction 的 compile-time value，不是可变的
Pipe-object state：`0` 表示不切分，`1` 表示上下切分，`2` 表示左右切分。

这些是 PTODSL API，并非 TileLib template API。template expansion 时，template 不会
收到 Python `Pipe` object，也不能调用其方法。PTOAS 通过第 7 节 ABI 提供每条
operation 的 `PipeSpec`、有序 `PipeResources`、共享 `PipeState`、存在时的 entry，以及
可选 AIV subblock value。TileLib owner 根据这些 ABI value 实现 transaction 的 FIFO
address、synchronization 和 counter 行为；本次变更既不新增 template-side Pipe wrapper，
也不修改 `ptodsl/tilelib/**`。

## 5. A5 VPTO 默认行为与兼容性

不存在 pipe expansion feature flag。当包含 frontend 或 unified pipe transaction 的 module
使用 `--pto-arch=a5 --pto-backend=vpto` 编译时，driver 会自动执行 pipe-specific validation、
PipeState materialization、candidate discovery 和 expansion preparation。默认 TileLib backend
是 PTODSL，因此默认路径使用 PTODSL metadata 和 expansion。

PipeState materialization pass 只插入到 A5 VPTO backend。所有 EmitC 路径（包括 A5
EmitC）仍使用既有 `TPipe` lowering，且不会 materialize PipeState 或 terminal `tdrain`。
A2/A3 行为同样保持不变。

若所选 TileLib 尚未实现合法 pipe candidate，candidate discovery/expansion 会明确失败。
PTOAS 不得静默回退到 C++ `TPipe` 或另一套 TileLib implementation。

## 6. PTO IR 契约

### 6.1 有状态 operation

内部 `pto.talloc`、`pto.tpush`、`pto.tpop` 与 `pto.tfree` 均有可选的
`pipe_state` operand。legacy source-authored IR 仍可不含该 operand，以保持既有
EmitC 路径的 source compatibility。feature-owned VPTO IR 使用的类型固定为：

```text
!pto.struct<i32, i32>
```

verifier 拒绝其他 state shape。一旦某个 pipe 的任一 stateful user 带有 state，该
pipe 的全部 `talloc`、`tpush`、`tpop`、`tfree` 和 `tdrain` user 都必须携带同一个
SSA state。这样可以在 hand-authored internal IR 中拒绝部分 materialization 或各自
分配 counter 的情况。`tpush`、`tpop` 和 `tfree` 使用 operand segment，使已有的可选
entry/subblock operand 在 assembly form 中仍无歧义。

两个 field 的契约固定如下：

| Field | 名称 | 含义 |
|---|---|---|
| 0 | `prod_index` | 下一个 producer FIFO position。 |
| 1 | `cons_index` | 下一个 consumer FIFO position。 |

state 不包含 pipe handle、物理地址、`flag_base` 或 split mode。这些属于不可变的
pipe configuration 或 runtime resource，会在 expansion 时单独提供。

### 6.2 结束 `tdrain`

`pto.tdrain` 是只由 PipeState 路径插入的内部 operation：

```text
pto.tdrain(%pipe, %state : !pto.pipe, !pto.struct<i32, i32>) { split = <0|1> }
```

它表示此前由 EmitC `TPipe` object 生命周期执行的 producer-side cleanup。对于至少
有一个 producer `tpush` 的 pipe，它在每个可达 `func.return` 前插入一次；没有 producer
的 pipe 不插入 drain。

`tdrain` 是 pipe-level cleanup。其 `split` 在 pipe configuration 解析后推导，不能从
producer operation 推导：

| 已解析 initializer `nosplit` | materialize 的 `tdrain.split` |
|---|---|
| `true` | `0` |
| `false` | `1` |

因此 materialization pass 接受不同的 producer axis，例如
`tpush(split=1)` 后再 `tpush(split=2)`。既有 infer/validate pass 仍负责拒绝非法的
`split` / `nosplit` 组合。IR verifier 还要求 hand-authored `tdrain` 使用该精确推导
的 split，并在 initializer 尚未解析 `nosplit` 时拒绝该 drain。

### 6.3 Materialization pass

`pto-materialize-pipe-state` 在 `pto-infer-validate-pipe-init` 已解析 `nosplit`
之后，以每个 `func.func` 为单位运行。A5 VPTO driver 将其安排在 layout、fusion、
memory planning 和 reserved-buffer resolution 之后、pipe-only candidate discovery 之前。
这样既有 shared pass 继续消费原有 pipe IR，同时 pipe metadata 能同时获得 PipeState、
已解析的 `flag_base` 与资源。

对于每个有 stateful user 的 initialized pipe，它会：

1. 创建一个 `!pto.struct<i32, i32>` 类型的 `pto.declare_struct`。
2. 用 `pto.struct_set` 将两个 field 初始化为零。
3. 将同一个 state 附加到该 pipe 的全部 `talloc`、`tpush`、`tpop` 与 `tfree` user。
4. 标记 feature-owned pipe IR，供后续 expansion cleanup 使用。
5. 只要该 pipe 有 producer，就插入 pipe-level `tdrain`。

initializer 必须支配每个需要插入 drain 的 return。pass 对不能表示的 nested/lifetime
形式给出诊断，而不是跨 region 移动 pipe handle。

## 7. Expansion ABI

### 7.1 值的职责划分

`ExpandTileOp` 将 pipe 信息拆分为四个 logical operand：

| Logical operand | 职责 | Physical helper argument |
|---|---|---|
| entry（存在时） | transaction 使用的 tile 或可变 GM descriptor | 是 |
| PipeSpec | 不可变配置和逐 operation 的 `split` | 否 |
| PipeResources | 有序 runtime address | 每个 resource 一个 argument |
| PipeState | 可变 producer/consumer counter | 一个 struct argument |

`!pto.pipe` 不会成为 template helper argument。直接传递它会使 opaque pipe dependency
留在 VPTO lowering 中，违背 expansion 的目标。

对于 `tpush` 与 `tpop`，ABI 还包含 AIV subblock ID。原 operation 没有该 ID 时，PTOAS
会序列化值为零的 `i64` scalar，以保持 template logical signature 稳定。

### 7.2 PipeSpec producer 契约

在 `pto-resolve-reserved-buffers` 后，编译器从 pipe initializer 和单条 operation
推导 `PipeWireInfo`，并序列化如下 `pipe` operand：

```json
{
  "kind": "pipe",
  "init_kind": "l2g2l",
  "dir_mask": 2,
  "slot_size": 1024,
  "slot_num": 8,
  "local_slot_num": null,
  "flag_base": 0,
  "nosplit": false,
  "split": 2,
  "resource_names": ["gm_addr"]
}
```

`init_kind` 取值为 `l2l` 或 `l2g2l`。source direction encoding 仍是 PTO frontend
encoding（`1`、`2` 或 `3`）；TileLib consumer 不得假设该 field 已被转成
ISA-specific direction。

`split` 从每条 `talloc`、`tpush`、`tpop`、`tfree` 或 `tdrain` 复制而来，并纳入
specialization identity，因此不同 split path 的 helper 不会错误复用。`tdrain` 因而
拿到第 6.2 节确定的 pipe-level 值。

缺少 `flag_base`、无法解析 initializer、没有 PipeState，或使用不支持的
`acc_push_epilogue`，都会在请求 helper 前报错。

### 7.3 Resources、state 和 entry

`PipeResources` 按 `resource_names` 指定的精确顺序保存存在的资源：

```text
gm_addr, local_addr, peer_local_addr
```

`l2l` pipe 可提供 local 和 peer-local resource。`l2g2l` pipe 还可提供 `gm_addr`；
不存在的 resource 不占用 helper argument。

`PipeState` 的序列化形式为：

```json
{"kind": "pipe_state", "fields": ["i32", "i32"]}
```

已声明的 global entry 会序列化为 `pipe_entry`，而不是普通只读 view，因为 TileLib
可能需要重新绑定 caller-owned descriptor。其 helper ABI 是 `!pto.tensor_view<...>`。
tile entry 保持原有 tile metadata。

### 7.4 Specialization 与 cleanup

`ExpandTileOp` 使用 operation name、target、entry metadata、PipeSpec、resource
topology/type metadata、PipeState schema、subblock argument，以及 `kernel_kind` 等
context attribute 构建 specialization key。它转发每条 operation 的 `split`，而不为
整个 pipe 选择单一 split 值。

template expansion 成功后，PTOAS 只删除 use 已消失的 feature-owned unified pipe
operation 和 initializer。若 feature-owned pipe operation 仍存在，则 expansion 报错，
不回退到其他后端。

## 8. VPTO LLVM 边界

PipeState 必须跨 helper creation 和 inlining 存活，因此两个 VPTO LLVM emitter 都对
受支持的 struct subset 进行 lowering：

- `pto.declare_struct` lower 为 function-local storage；
- `pto.struct_get` lower 为 field address calculation 和 load；
- `pto.struct_set` lower 为 field address calculation 和 store。

pipe entry 路径还支持可变的 `!pto.tensor_view` descriptor：
`pto.declare_global`、`pto.tassign` 和 `pto.tensor_view_addr` 通过 descriptor
storage lower，不留下 memref bridge 或 VPTO pipeline 中的
`unrealized_conversion_cast`。这是 TileLib ABI 的编译器支持，不是 FIFO address 或
synchronization 语义的实现。

global pipe entry 可以直接使用 `pto.declare_global` 的结果，也可以使用从该声明
经过一个或多个 `pto.tassign` 得到的结果。传给 pipe op 的仍是重绑定后的结果值。

## 9. TileLib 交接

以下工作明确属于 TileLib owner，不包含在此 PR：

- 消费 `pipe`、`pipe_resources`、`pipe_state`、`pipe_entry` operand 的 metadata
  class、renderer binding 与 candidate constraint；
- `talloc`、`tpush`、`tpop`、`tfree`、`tdrain` 的 A5 template；
- GM FIFO split address offset 与 local split/subblock constraint；
- operation-specific synchronization、FIFO index update 与 terminal drain 行为；
- daemon/unit coverage 以及 TileLib-ST 或 simulator 端到端覆盖。

TileLib 实现必须把 PipeSpec 视为不可变 configuration，按声明顺序消费 resource，并
使用单条 operation 的 `split` 及可选 subblock 值。它不得从 `!pto.pipe` 恢复策略、
重新推导 pipe-wide split，或将 `tdrain` 当作任意 producer transaction。

在该实现合入前，完整 pipeline 可能因没有合法 pipe template candidate，或 template
有意拒绝不支持配置而失败。这是预期的组件边界失败，不能据此认定 PTOAS pass 失败。

## 10. 诊断

编译器为下列边界违规提供可操作的失败信息：

| 条件 | 诊断方向 |
|---|---|
| PipeState 类型非法 | 要求 `!pto.struct<i32, i32>` |
| PipeState 关联不一致 | 要求同一 pipe 的全部 stateful user 共享一个 state |
| hand-authored `tdrain.split` 非法 | 要求使用由已解析 `nosplit` 推导的 split |
| drain 前 `nosplit` 未解析 | 要求在 materialization 前执行 pipe-init validation |
| drain lifetime 不支配 | 标识 initializer 与受影响 return |
| `flag_base` 未解析 | 要求 expansion 前执行 reserved-buffer resolution |
| pipe initializer/state 不可解析 | 标识对应 unified pipe operation |
| `acc_push_epilogue` 不支持 | 在 template invocation 前拒绝 |
| 没有 legal candidate 或遗留 feature-owned pipe IR | expansion 无回退地失败 |

## 11. 验证与后续工作

### 11.1 已包含覆盖

PTODSL surface regression 覆盖显式 full-slot GM construction，以及两条独立的
nonzero transaction（`split=1`、`split=2`）经过 `alloc`、`push`、`pop`、`free` 的
路径。它检查每个 operation 收到自己的 split 值，而不是 pipe-wide default。

本次变更要求的本地检查为：

```text
ninja -C build-local-vpto ptoas PTOPythonModules
ptodsl/tests/test_vector_cube_ops.py -v
```

当前 unit suite 通过 38 个测试，`git diff --check` 也通过。

额外的 focused compiler lit coverage 覆盖 PipeState materialization、PipeSpec RPC
payload（使用位于 `ptodsl/tilelib/**` 之外的 test-only mock daemon）、默认与显式 AIV
subblock operand、有序的 GM 和 local/peer-local resource list、可变 descriptor 的
VPTO LLVM lowering、verifier 诊断、默认 A5 VPTO activation，以及 A5/A3 EmitC
compatibility。mock daemon 不代表 TileLib template 或 FIFO runtime 行为已经得到验证。

### 11.2 TileLib 必需的后续覆盖

TileLib 接入此 ABI 后，必须自行补充：

- `split=0`、`1`、`2` 的 candidate selection 与 specialization separation；
- GM/local resource order、address offset 与 subblock handling；
- PipeState counter transition 与 terminal `tdrain` 行为；
- 不支持 direction/quantization 的诊断；以及
- TileLib-ST/simulator 路径上的端到端 FIFO 正确性。

runtime suite 必须验证输出和同步行为。仅检查能编译或能渲染 IR，不能证明
GM/local FIFO 语义正确。

## 12. 验收边界

当 frontend 保留 split、PipeState/`tdrain` materialization 遵循已解析的 `nosplit`、
编译器产生本文档化的 expansion ABI，且 VPTO LLVM 可 lower 所需 state 和可变
descriptor 形式时，PTOAS 侧工作完成。

只有独立的 TileLib 变更实现该交接契约，并以 TileLib-ST/simulator 覆盖证明支持配置
后，完整功能才完成。PR 状态和 release claim 必须明确区分这两个里程碑。
