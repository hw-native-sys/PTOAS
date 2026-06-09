# **同步操作**

昇腾 AI Core 中各条流水线（如 MTE2、Vector、Cube、MTE3、MTE1、Fixpipe、Scalar）异步执行。这些流水线之间的数据依赖关系需要显式的同步操作来建立执行顺序。本节介绍的同步操作涵盖了事件记录与等待、流水线内存屏障、跨核全局同步等机制，使得用户和编译器能够精细控制硬件流水线的同步行为。

同步操作通常不产生 SSA 返回值，而是作为执行流控制的指令。部分同步操作使用属性（Attribute）来指定操作类型或事件 ID。

---

## 目录

- [`pto.record_event` — 记录同步事件](#ptorecord_event--记录同步事件)
- [`pto.wait_event` — 等待同步事件](#ptowait_event--等待同步事件)
- [`pto.barrier_sync` — 高层同步屏障](#ptobarrier_sync--高层同步屏障)

---

## 操作详解

### `pto.record_event` — 记录同步事件

```
pto.record_event [<src_op>, <dst_op>, <event_id>]
```

**语义：**

为指定的事件 ID 记录一个从源操作类到目的操作类的同步关系。该事件可随后通过 `pto.wait_event` 等待。可以把它理解成一条依赖边：

```text
src_op 完成
  -> 记录 event_id
  -> dst_op 等待这个 event_id
  -> dst_op 继续执行
```

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src_op` | `#pto.sync_op_type<...>` | 产生完成信号的事件类型，例如 `#pto.pipe_event_type<EVENT_LOAD_FROM_GM>`、`#pto.pipe_event_type<EVENT_COMPUTE_MATMUL>` |
| `dst_op` | `#pto.sync_op_type<...>` | 消费该完成信号的目标事件类型，例如 `#pto.pipe_event_type<EVENT_COMPUTE_VEC>` |
| `event_id` | `#pto.event<...>` | 事件 ID，例如 `#pto.event<EVENT_ID0>` |

**返回值：** 无。

**参数详解：**

`#pto.sync_op_type<...>`当前支持以下取值：

| Value | Int | 含义 | 典型用途 |
| ---- | ---- | ---- | -------- |
| `EVENT_LOAD_FROM_GM` | `0` | 从 GM 把数据加载到片上 | 表示一次 load 已完成，后续计算或搬运可以消费数据 |
| `EVENT_STORE_FROM_ACC` | `1` | 从 ACC 写回 | 表示累加器结果已可写出 |
| `EVENT_STORE_FROM_VEC` | `2` | 从 VEC/UB 写回 | 表示向量结果已可写出 |
| `EVENT_MOVE_MAT_TO_LEFT` | `3` | `MAT -> LEFT` 搬运 | 表示左矩阵输入已准备好 |
| `EVENT_MOVE_MAT_TO_SCALAR` | `4` | `MAT -> SCALAR` 搬运 | 表示标量侧可读取对应数据 |
| `EVENT_MOVE_MAT_TO_BIAS` | `5` | `MAT -> BIAS` 搬运 | 表示 bias 缓冲区输入已准备好 |
| `EVENT_MOVE_MAT_TO_VEC` | `6` | `MAT -> VEC` 搬运 | 表示向量流水线可消费来自 MAT 的数据 |
| `EVENT_MOVE_VEC_TO_MAT` | `7` | `VEC -> MAT` 搬运 | 表示矩阵流水线可消费来自 VEC 的数据 |
| `EVENT_COMPUTE_MATMUL` | `8` | MatMul 计算完成 | 表示 Cube/Matmul 结果已准备好，可供后续使用 |
| `EVENT_COMPUTE_VEC` | `9` | Vector 计算完成 | 表示向量计算结果已准备好 |
| `EVENT_VEC_WAITPOINT` | `10` | Vector 等待点 | 用于向量流水线上的显式等待点或阶段边界 |

如果不知道该怎么选，通常按“前驱实际完成了什么”来选 `src_op`，按“后继是谁在等待”来选 `dst_op`：

  - GM 读入完成后给向量计算用：`EVENT_LOAD_FROM_GM -> EVENT_COMPUTE_VEC`
  - MatMul 完成后给写回或后续搬运用：`EVENT_COMPUTE_MATMUL -> ...`
  - 向量计算完成后给后续阶段用：`EVENT_COMPUTE_VEC -> ...`

`#pto.event<...>` 当前支持以下静态取值：

| Value | Int |
| ---- | ---- |
| `EVENT_ID0` | `0` |
| `EVENT_ID1` | `1` |
| `EVENT_ID2` | `2` |
| `EVENT_ID3` | `3` |
| `EVENT_ID4` | `4` |
| `EVENT_ID5` | `5` |
| `EVENT_ID6` | `6` |
| `EVENT_ID7` | `7` |

**约束：**

- **实现检查 (A2A3/A5)**
  - `src_op` 和 `dst_op` 必须使用有效的 `#pto.pipe_event_type<...>` 枚举值。
  - `event_id` 必须使用有效的 `#pto.event<...>` 枚举值。

**示例：**

```mlir
pto.record_event [#pto.pipe_event_type<EVENT_LOAD_FROM_GM>, #pto.pipe_event_type<EVENT_COMPUTE_VEC>, #pto.event<EVENT_ID0>]
```

---

### `pto.wait_event` — 等待同步事件

```
pto.wait_event [<src_op>, <dst_op>, <event_id>]
```

**语义：**

等待由 `pto.record_event` 记录的事件。当指定的事件被记录完成后，此操作才返回，从而确保数据依赖关系得到满足。

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `src_op` | `#pto.sync_op_type<...>` | 与 `pto.record_event` 中一致的源事件类型 |
| `dst_op` | `#pto.sync_op_type<...>` | 与 `pto.record_event` 中一致的目标事件类型 |
| `event_id` | `EventAttr` | 事件 ID |

**返回值：** 无。

**约束：**

- **实现检查 (A2A3/A5)**
  - `src_op` 和 `dst_op` 必须使用有效的 `#pto.pipe_event_type<...>` 枚举值。
  - `event_id` 必须使用有效的 `#pto.event<...>` 枚举值。

**示例：**

```mlir
pto.wait_event [#pto.pipe_event_type<EVENT_LOAD_FROM_GM>, #pto.pipe_event_type<EVENT_COMPUTE_VEC>, #pto.event<EVENT_ID0>]
```

---

### `pto.barrier_sync` — 高层同步屏障

```
pto.barrier_sync [<op_type>]
```

**语义：**

在高层指定一个同步屏障，通过操作类型（而非具体管线）来标识。编译器的降级（lowering）通路会根据操作类型（如 TMATMUL、TVEC 等）将其映射到相应的硬件管线，并生成对应的 `pto.barrier` 指令。

**参数：**

| Name | Type | Description |
| ---- | ---- | ----------- |
| `op_type` | `#pto.sync_op_type<...>` | 同步操作类型，例如 `#pto.sync_op_type<TMATMUL>` |

**返回值：** 无。

**参数详解：**

`op_type` 使用 `#pto.sync_op_type<...>`，它表示“按哪一类高层同步端点来插入屏障”。 `#pto.pipe_event_type<...>` 不同，`#pto.sync_op_type<...>` 更偏向高层操作语义，而不是某个具体完成事件。`pto.barrier_sync` 会根据这个高层类型，把屏障映射到对应的硬件流水线同步点。

当前支持以下取值：

| Value | Int | 含义 | 典型用途 |
| ---- | ---- | ---- | -------- |
| `TLOAD` | `0` | 高层 load 同步端点 | 在 load 相关阶段前后插入同步 |
| `TSTORE_ACC` | `1` | 从 ACC 写回的高层同步端点 | 在累加器结果写回前后插入同步 |
| `TSTORE_VEC` | `2` | 从 VEC/UB 写回的高层同步端点 | 在向量结果写回前后插入同步 |
| `TMOV_M2L` | `3` | `MAT -> LEFT` 搬运端点 | 约束左矩阵输入相关阶段 |
| `TMOV_M2S` | `4` | `MAT -> SCALAR` 搬运端点 | 约束标量相关阶段 |
| `TMOV_M2B` | `5` | `MAT -> BIAS` 搬运端点 | 约束 bias 输入相关阶段 |
| `TMOV_M2V` | `6` | `MAT -> VEC` 搬运端点 | 约束矩阵到向量的搬运阶段 |
| `TMOV_V2M` | `7` | `VEC -> MAT` 搬运端点 | 约束向量到矩阵的搬运阶段 |
| `TMATMUL` | `8` | 矩阵计算端点 | 对 Cube / MatMul 阶段插入同步 |
| `TVEC` | `9` | 向量计算端点 | 对 Vector 阶段插入同步 |
| `TVECWAIT_EVENT` | `10` | Vector wait 端点 | 对 vector wait / waitpoint 类阶段插入同步 |

如果不知道该选哪个值，可以按“你要约束的是哪类高层操作”来判断：

- 要同步 load 阶段，选 `#pto.sync_op_type<TLOAD>`
- 要同步矩阵乘阶段，选 `#pto.sync_op_type<TMATMUL>`
- 要同步向量计算阶段，选 `#pto.sync_op_type<TVEC>`
- 要同步某类片上搬运阶段，选对应的 `TMOV_*`
- 要同步结果写回阶段，选对应的 `TSTORE_*`

**约束：**

- **实现检查 (A2A3/A5)**
  - `op_type` 必须使用有效的 `#pto.sync_op_type<...>` 枚举值。

**示例：**

```mlir
pto.barrier_sync [#pto.sync_op_type<TLOAD>]
pto.barrier_sync [#pto.sync_op_type<TMATMUL>]
pto.barrier_sync [#pto.sync_op_type<TVEC>]
```
