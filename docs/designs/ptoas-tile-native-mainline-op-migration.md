# PTOAS Tile-Native Mainline 按 Op 迁移设计

## 1. 背景

PTOAS 当前主 pipeline 在内存规划前后包含两个 ModuleOp pass：

```text
PTOViewToMemref
  -> implicit tmp materialization
  -> legacy/modern memplan
  -> InsertSync/BufidSync
  -> PTOResolveBufferSelect
  -> PTOMaterializeTileHandles
  -> EmitC/VPTO
```

`PTOViewToMemref` 把函数 ABI、PTO view、部分 local tile 和计算 op 转成
memref 形态；`PTOMaterializeTileHandles` 在规划和同步完成后重新构造
`!pto.tile_buf` handle。普通 `pto.alloc_tile` 已经是例外：它保持
tile-native，由 memplan 直接填写 `addr`。

本设计的目标是逐个 op 消除上述往返转换，最终删除：

- `lib/PTO/Transforms/PTOViewToMemref.cpp`
- `lib/PTO/Transforms/PTOMaterializeTileHandles.cpp`
- `pto-view-to-memref`
- `pto-materialize-tile-handles`

迁移过程中不增加用户可见的双 pipeline 开关。每个 op 在现有 pipeline 中
先补齐 tile-native 分析和 lowering；全部 op 完成后一次性删除两个 pass。

## 2. 目标 IR

```text
PTO tile/view IR
  -> implicit tmp materialization（产生 pto.alloc_tile(no addr)）
  -> memplan（直接填写 alloc_tile/alloc_multi_tile 地址）
  -> sync（直接分析 PTO root、view、slot 和物理地址）
  -> EmitC/VPTO（直接消费 PTO tile/view IR）
```

基本原则：

1. local allocation root 使用 PTO allocation op 表达，不中转 `memref.alloc`。
2. tile view 保持 `!pto.tile_buf`，view op 自身携带 shape/layout/offset 语义。
3. GM tensor view 保持 PTO view op，直到 EmitC/VPTO backend lowering。
4. memplan、InsertSync 和 backend 各只有一套算法主体，通过统一的 root/alias/address
   查询处理不同 PTO op。
5. `pto.bind_tile`、`pto.declare_tile_memref`、`pto.slot_marker` 仅在完成对应
   tile-native 迁移后删除。

## 3. 每个 Op 的完成标准

每个 op 只有同时满足以下条件才算迁移完成：

- verifier 和 parse/print 能表达目标 tile-native 语义；
- legacy 和 modern memplan 能识别其 root/alias/size/address；
- InsertSync、Graph Sync Solver 和 BufidSync 能恢复其物理内存访问；
- EmitC 和 VPTO 都能直接 lowering；
- level1/level2/level3 行为明确；
- A2/A3 和 A5 行为明确；
- 至少有一个正例和必要的 verifier/level 负例；
- 测试不再检查 `bind_tile`、`memref.subview` 等旧中间形态。

## 4. Allocation 与内部 Handle Op

| Op | 当前行为 | Tile-native 目标 | Memplan | InsertSync | Backend 与测试 |
|---|---|---|---|---|---|
| `pto.alloc_tile` | `PTOViewToMemref` 已透传；memplan 填写 `addr` | 保持现状，作为单 slot local root | legacy/modern 均继续收集无地址 root；level3 校验显式地址 | InsertSync 已读取类型、大小和常量地址 | EmitC/VPTO 已有直接 lowering；保留 level1/2 自动地址和 level3 显式地址测试 |
| `pto.alloc_multi_tile` | 已保持 tile-native；memplan 写入内部 N-address 属性，level3 保留用户 base | op 本身作为 N-slot root，最终可由 backend 直接消费 | legacy/modern 已支持 `slotCount=N`、`slotBytes`、`totalBytes` 和每 slot offsets；禁止 sibling slot 互相 alias | InsertSync 已直接识别 multi root，不依赖 `slot_marker -> pointer_cast` | 当前由 `PTOResolveBufferSelect` 展开成带地址的 `alloc_tile`；已覆盖 constant/dynamic slot、loop/non-loop 测试 |
| `pto.multi_tile_get` | 已保持到 sync 和 `PTOResolveBufferSelect` | 最终继续保持到 backend | result 已 alias 到 multi root，并携带 slot expression | constant slot 只访问一个地址；dynamic slot 保守访问全部地址，并保留 dyn event-id 推导 | 当前 resolve pass 根据 slot 生成带单地址的 `alloc_tile`；不再生成 `slot_marker` |
| `pto.declare_tile` | 已保持 tile-native，不参与静态 local allocation | 保持现状，表示地址由 `tpop/tassign` 等运行时操作绑定 | legacy/modern 不收集为 allocation root | InsertSync 直接注册 symbolic root | EmitC 已直接声明 Tile；已覆盖 tpush/tpop 和 interleaved sync |
| `pto.declare_tile_memref` | 仅保留 legacy memref IR 兼容，不再由 `PTOViewToMemref` 生成 | 其余旧兼容用户清除后删除 | legacy 仅跳过，不分配 | InsertSync 暂保兼容入口 | 暂保 legacy EmitC pattern，最终随兼容路径删除 |
| `pto.bind_tile` | memref 与 tile metadata 的桥接和 alias anchor | local tile 不再需要；只允许迁移期兼容，最终删除或限制为外部 ABI bridge | 先统一 `getAliasSource()`，切换后删除 bind 分支 | 先统一 alias tracing，切换后删除 bind 分支 | 删除 materialize/EmitC bind lowering 测试 |
| `pto.pointer_cast` | 表达 memref root 的物理地址或 multi-address root | 仅保留真正的 raw-address ABI bridge；普通 local allocation 改由 allocation op 持有地址 | 不再作为 `alloc_tile` 的 materialization 结果 | 若保留，sync 继续读取其地址；multi-buffer 不再必须依赖它 | 后端保留 raw-address lowering，删除 alloc fallback 用法 |
| `pto.slot_marker` | `multi_tile_get` 在 memref 层的内部 slot 标签 | `multi_tile_get` 直接承担 slot 标签语义，最终删除 | 删除 alias 特判 | 将 slot narrowing 和 dyn event-id 逻辑迁移到 `multi_tile_get` | 删除 `PTOResolveBufferSelect` 对 slot_marker 的入口 |
| `pto.materialize_tile` | 用于显式恢复 tile handle 的兼容 op | 若无独立用户则随 materialize pass 一并删除 | 无 | 无 | 搜索并删除残余 pattern/test |

## 5. Tile View 与 Metadata Op

| Op | 当前行为 | Tile-native 目标 | Memplan / Alias | InsertSync | Backend 与测试 |
|---|---|---|---|---|---|
| `pto.subview` | 已在 memplan 和 sync 主线保持 tile-native，不再生成 `memref.subview + pto.bind_tile` | 同步完成前保留 `pto.subview` | result 与 source 同 root；按 layout/stride/offset 计算 byte range | InsertSync 已精确计算地址范围并通过通用 view alias 回溯 source | `PTOResolveBufferSelect` 在同步后解析为带偏移地址的 `alloc_tile`，供 EmitC/VPTO 共用；覆盖 row/col-major、boxed、dynamic offset 和 valid shape |
| `pto.treshape` | 已保持 tile-native，不再生成 memref view 或 `bind_tile` | 保持原 op | result 与 source 完全 alias，size 不变；legacy/modern memplan 已传播 alias | InsertSync 已直接传播 result-to-source alias | EmitC 已直接 lowering；覆盖 tile 参数、动态 valid shape 和静态 valid shape |
| `pto.bitcast` | 已保持 tile-native，不再生成 memref view 或 `bind_tile` | 保持原 op | result 与 source alias；legacy/modern memplan 已按 byte range 传播同一 root，并校验总容量一致 | InsertSync 已直接传播 alias，不按 element type 分裂 root | EmitC 已直接 lowering；覆盖 tile 参数和同容量 dtype 改变 |
| `pto.set_validshape` | 已保持 tile-native，输入已收紧为 `TileBufType` | 直接更新 tile handle metadata，不改变物理 root | 不产生新 root；物理 size 使用 allocation shape，不使用 valid shape | 对原 root 建模为 metadata Write | EmitC 已直接消费；覆盖 if 和动态 valid shape |
| `pto.get_validshape` | 已保持 tile-native，输入已收紧为 `TileBufType` | 直接读取 tile operand metadata | 不产生 root | 对原 root 建模为 metadata Read | EmitC 已直接 lowering；覆盖 tile 参数和动态值 |
| `pto.tile_buf_addr` | tile-native 输入已保持原 op；仅 legacy memref 输入仍走线性 memref 兼容路径 | 直接从 tile root/view 计算地址；返回 pointer-like PTO 类型 | 不产生新 root；legacy memplan 将 source 记录为 use | 包含该 op 的函数保持 tile ABI，地址结果保留 source provenance | EmitC 直接生成 `tile.data()`，VPTO pointer normalize 保持 typed pointer；覆盖 tile 参数和 alloc root |

## 6. GM Tensor View 与地址 Op

| Op | 当前行为 | Tile-native 目标 | Memplan / Sync | Backend 与测试 |
|---|---|---|---|---|
| `pto.make_tensor_view` | 已保持 PTO 形态穿过 `PTOViewToMemref`、memplan 和 sync | 最终保持到 backend | 不参与 local memplan；InsertSync 已把 result alias 到 pointer source | 当前在 `PTOResolveBufferSelect` 与完整 GM view 链一起转成 `memref.reinterpret_cast`，复用成熟 backend lowering |
| `pto.partition_view` | 已保持 PTO 形态穿过 memplan 和 sync | 最终保持到 backend | 不参与 local memplan；sync result alias source，并根据 offset/size 缩小 GM range | memref-backed source 在 `PTOResolveBufferSelect` 转成 `memref.subview`；`declare_global` 等运行时 source 保持 PTO op 走已有直接 EmitC lowering |
| `pto.get_tensor_view_dim` | 已保持 PTO 形态穿过 memplan 和 sync | 直接读取 PTO view shape | 不影响 memplan/sync | 当前在 `PTOResolveBufferSelect` 随所属 view 转成 `memref.dim` |
| `pto.get_tensor_view_stride` | 已保持 PTO 形态穿过 memplan 和 sync | 直接读取 PTO view stride | 不影响 memplan/sync | 当前在 `PTOResolveBufferSelect` 随所属 view 转成 strided memref metadata |
| `pto.inttoptr` | 结果改成 GM memref，并限制用途 | 保持 PTO pointer-like value | 不参与 local memplan；sync 将其视作 GM provenance | 保留 restricted-use verifier；EmitC/VPTO 直接 lowering |
| `pto.ptrtoint` | 折叠 `addptr` 链并生成 byte offset | 保持 PTO op，或迁移到独立 address-canonicalization pass | 不影响 local memplan；不能丢失 GM provenance | backend 生成整数地址；覆盖 ptr、addptr、view base |
| `pto.addptr` | 折叠进 tensor view、scalar load/store 或 pipe init | 保持到独立 address canonicalization/backend | sync 需要把 result alias 到 base，并记录 offset | EmitC/VPTO 直接 lowering；保留非法 escape 校验 |
| `pto.castptr` | 作为 pointer/memref 适配 op | 保持 PTO pointer cast | result alias source，地址空间变化必须校验 | 两后端直接 lowering |
| `pto.load_scalar` | `addptr` offset 被折叠进 op | op 直接接受 base+offset，或 backend 统一折叠 | InsertSync 记录 GM read；不参与 local memplan | 覆盖 inttoptr/addptr 和动态 offset |
| `pto.store_scalar` | `addptr` offset 被折叠进 op | op 直接接受 base+offset，或 backend 统一折叠 | InsertSync 记录 GM write | 覆盖跨 pipe flush/sync |
| `pto.initialize_l2g2l_pipe` | `gm_addr` 上的 addptr 被提前折叠 | 迁移到独立 address canonicalization 或 op verifier/lowering | sync 保留 GM base provenance | EmitC/VPTO 覆盖动态地址 |

## 7. Control Flow 与函数 ABI

| Op | 当前行为 | Tile-native 目标 | 需要改动 |
|---|---|---|---|
| `func.func` | tile 参数/结果可能改成 memref，入口插入 `bind_tile` | authored tile ABI 保持不变 | backend helper、外部声明和 kernel ABI 明确区分；memplan 仍按函数独立规划 |
| `func.call` | 依赖 ViewToMemref 的签名桥接，materialize 后恢复 helper tile ABI | caller/callee 直接使用一致的 tile/pointer ABI | call verifier、inline helper 和 InsertSync call effects 需要统一；跨函数不做 local allocation 合并 |
| `scf.if` | pass 重建 result type 和两个 `scf.yield` | tile result 原样穿过分支 | memplan 和 sync 合并两个分支的 root family；backend 支持 tile result |
| `scf.for` | pass 重建 iter_arg/result/yield 类型 | tile loop-carried value 原样保持 | memplan 扩展 loop-carried root 生命周期；sync 建立 iter_arg/yield alias 和 back-edge dependency |
| `scf.yield` | 随外层控制流重建 operand type | 直接 yield tile value | 不能把 yield 当新 root；传播到外层 result/下一轮 iter_arg |
| `pto.fusion_region` | 协调 region result 与 `pto.yield` 类型 | 保持 tile-native region ABI | FusionAnalysis、memplan、sync 和 backend 共享同一 root family |
| `pto.yield` | materialize 阶段恢复 tile operand | 直接 yield tile value | 与 `scf.yield` 相同，传播 alias/liveness |

## 8. 特殊后端/同步 Op

| Op | 当前依赖 | 删除两个 pass 后的工作 |
|---|---|---|
| `pto.vlds` | materialize pass 为 view operand 恢复 tile address | backend 和 sync 直接从 tile/view root 计算地址；保持 Read/Write effects |
| `pto.vsts` | 同上 | 直接处理 tile view 地址和 post-update 结果 |
| `pto.vsstb` | 同上 | 直接处理 tile view 地址、packed stride 和 post-update result |
| `pto.mgather` | ViewToMemref 会在无 tile root 的旧路径重建 op | 保持 tile operands；sync macro model 直接读取 index/src/dst/scratch 的 effects |
| `pto.mscatter` | 同上 | 保持 tile operands；直接计算 GM/local ranges 和 coalesce 语义 |
| `pto.tassign` | ViewToMemref 协调 result type，materialize 用它恢复地址 | 明确为 tile handle 地址重绑定；result 类型始终等于 tile operand 类型 | memplan 不把 result 当新 allocation；sync 将其 alias 到 tile root并更新地址 provenance；backend 直接 lowering |

## 9. Compute Op 逐项处理

以下 op 在 `PTOViewToMemref` 的 Stage 3 中主要为了适配 memref operand 而被
重新创建。tile-native 主路径中不应重建它们。每个 op 的共同要求是：

- 保持 authored `!pto.tile_buf` operand/result；
- memplan 只通过 MemoryEffects、DPS output 和 alias side table 使用它，不改 op；
- InsertSync 直接读取 MemoryEffects；tmp/scratch 必须保持正确的 Read/Write；
- EmitC/VPTO 的现有 tile lowering 必须在没有 materialize pass 的情况下通过；
- 每个 op 至少保留一个 end-to-end lit 或 sample。

| Op | 额外检查 |
|---|---|
| `pto.tload` | GM view 到 local tile 的 MTE2 write；dst 是 writer output，地址来自 `alloc_tile`/subview |
| `pto.tstore` | local tile 到 GM view 的 MTE3 read；检查 reused local addr 与后续 writer 的同步 |
| `pto.ttrans` | optional tmp 的条件 materialization、A5 placeholder 和 scratch conflict |
| `pto.texp` | precision attr 原样保留 |
| `pto.tmul` | DPS inplace 规则和 input/output alias |
| `pto.tmuls` | scalar operand 顺序和 precision attr |
| `pto.tadd` | DPS inplace 规则 |
| `pto.taddc` | carry/result operand 顺序和多输出语义 |
| `pto.tadds` | scalar operand和 valid-shape 校验 |
| `pto.taddsc` | scalar/carry 组合和多输出语义 |
| `pto.tmatmul` | LEFT/RIGHT/ACC 地址空间、tile config 和 role 不再依赖 materialize 推断 |
| `pto.tmatmul_acc` | ACC init/output 的 Read+Write 和 inplace 语义 |
| `pto.tmatmul_bias` | bias root/address space 和 scratch/output conflict |
| `pto.tmatmul_mx` | MX scale tile role 和低精度 packed shape |
| `pto.tmatmul_mx_acc` | MX scale + ACC init/output |
| `pto.tmatmul_mx_bias` | MX scale + bias root |
| `pto.tgemv` | vector/matrix role 和 dst writer lifetime |
| `pto.tgemv_acc` | accumulator init/output effects |
| `pto.tgemv_bias` | bias tile role |
| `pto.tgemv_mx` | MX scale role |
| `pto.tgemv_mx_acc` | MX scale + accumulator effects |
| `pto.tgemv_mx_bias` | MX scale + bias effects |
| `pto.tmov` | identity removal、view metadata、src/dst alias，以及 fp/pre-quant optional operands 和地址空间 |
| `pto.tabs` | unary Read(src)/Write(dst) |
| `pto.tand` | binary input/output effects |
| `pto.tands` | scalar form operand 顺序 |
| `pto.tor` | binary input/output effects |
| `pto.tors` | scalar form operand 顺序 |
| `pto.tnot` | unary effects |
| `pto.tneg` | unary effects |
| `pto.tcmp` | mask/output类型和 compare mode |
| `pto.tcmps` | scalar compare operand 顺序 |
| `pto.tconcat` | 多 input root 的 liveness |
| `pto.tconcatidx` | index/output effects |
| `pto.tci` | implicit tmp、A2/A3 scratch、A5 unused placeholder |
| `pto.tcolexpand` | broadcast source/output shape |
| `pto.tcolexpandmul` | binary broadcast effects |
| `pto.tcolexpandmax` | binary broadcast effects |
| `pto.tcolexpandmin` | binary broadcast effects |
| `pto.tcolmax` | reduction tmp/output effects |
| `pto.tcolmin` | reduction tmp/output effects |
| `pto.tcolsum` | `isBinary` 条件 tmp 和 arch 行为 |
| `pto.tcvt` | 条件 tmp、rmode/satmode operand 顺序 |
| `pto.tdiv` | precision attr 和 inplace policy |
| `pto.tdivs` | scalar 顺序和 precision attr |
| `pto.texpands` | shape扩展和 scalar operand |
| `pto.textract` | tile role、offset、fp tile 地址空间和可选 pre-quant |
| `pto.tinsert` | materialize pass 当前对 tile config 有特殊推断；包含 fp/pre-quant tile role，目标是由 result type 完整携带 |
| `pto.tfillpad` | 基于 physical shape 和 PlanMemory 地址推导 lowering、src/dst alias、MemoryEffects 和 A5 MAT/PIPE 选择 |
| `pto.tsetval` | tile writer 和 result type |
| `pto.tgetval` | tile reader和 scalar result |
| `pto.tgather` | optional tmp、compare/index form 和 sync macro model |
| `pto.tgatherb` | mask/index buffer effects |
| `pto.tlog` | precision attr |
| `pto.tlrelu` | unary input/output effects |
| `pto.tmax` | binary inplace policy |
| `pto.tmaxs` | scalar form |
| `pto.tmin` | binary inplace policy |
| `pto.tmins` | scalar form |
| `pto.tquant` | 删除 ViewToMemref 内部 tmp 补全；统一由 `PTOMaterializeImplicitTmp` 负责 |
| `pto.tmrgsort` | format1/format2、optional tmp 和多 source roots |
| `pto.tpartadd` | partition参数和 dst writer |
| `pto.tpartmul` | partition参数和 dst writer |
| `pto.tprint` | optional format tmp、tile read 和无输出语义 |

以下 op 当前没有出现在 Stage 3 的逐 op 重建列表中，通常已经保持 tile-native；
但 `PTOMaterializeTileHandles` 会泛化处理所有 `pto.t*` 的 memref operand，因此
删除 pass 前仍需逐项确认它们不依赖 memref metadata 回溯：

| Op | 额外检查 |
|---|---|
| `pto.trowexpandadd` | optional tmp、A2/A3 scratch 和 shared-tmp PIPE_V dependency |
| `pto.trowexpandsub` | optional tmp 和 scratch conflict |
| `pto.trowexpandmul` | optional tmp 和 inplace dst |
| `pto.trowexpanddiv` | optional tmp、precision attr 和 PIPE_V barrier pruning |
| `pto.trowexpandmax` | optional tmp 和 reduction/broadcast mode |
| `pto.trowexpandmin` | optional tmp 和 reduction/broadcast mode |
| `pto.trowexpandexpdif` | tmp effects 和 dst writer |
| `pto.tcolexpandadd` | broadcast input/output alias |
| `pto.tcolexpandsub` | broadcast input/output alias |
| `pto.tcolexpanddiv` | precision attr 和 broadcast alias |
| `pto.tcolexpandexpdif` | tmp/output effects |
| `pto.trowmax` | A2/A3 tmp、A5 placeholder 和 output lifetime |
| `pto.trowmin` | A2/A3 tmp、A5 placeholder 和 output lifetime |
| `pto.trowsum` | A2/A3 tmp、A5 placeholder 和 output lifetime |
| `pto.trowprod` | A2/A3 tmp、A5 placeholder 和 output lifetime |
| `pto.tcolargmax` | tmp capacity、value/index outputs 和多结果 root |
| `pto.tcolargmin` | tmp capacity、value/index outputs 和多结果 root |
| `pto.trowargmax` | index-only/value+index 模式和条件 tmp |
| `pto.trowargmin` | index-only/value+index 模式和条件 tmp |
| `pto.tprelu` | ui8 tmp、tmp/dst no-alias 和 A5 unused tmp |
| `pto.trem` | 2-row tmp、precision attr 和 scratch effects |
| `pto.trems` | 1-row tmp、scalar operand 和 scratch effects |
| `pto.tsel` | ui32 mask tmp、mask/src/dst effects |
| `pto.tsels` | one-row tmp、scalar select 和 A5 unused tmp |
| `pto.tpow` | A2/A3 floating tmp、integer no-tmp 和 A5 no-tmp |
| `pto.tpows` | A2/A3 floating tmp、scalar exponent 和 A5 no-tmp |
| `pto.trsqrt` | API compatibility tmp 不应触发自动 allocation |
| `pto.tsort32` | 非 32 对齐尾部的条件 tmp |
| `pto.txor` | tmp dtype/shape 和 tmp/output conflict |
| `pto.txors` | scalar operand、tmp dtype/shape 和 conflict |
| `pto.tpartmax` | partition参数和 dst writer |
| `pto.tpartmin` | partition参数和 dst writer |

其它没有在两个 pass 中出现、且 operand/result 始终是 tile-native 的 `pto.t*`
op 默认不需要迁移实现；仍需通过完整 lit 确认它们没有间接依赖 `bind_tile`、
`pointer_cast` 或 memref metadata 回溯。

## 10. InsertSync 专项改造

### 10.1 普通 InsertSync

现有 `PTOIRTranslator` 已直接支持：

- `pto.alloc_tile`
- `pto.make_tensor_view`
- `pto.partition_view`
- `pto.subview`
- `pto.bind_tile`
- `memref.subview`

删除两个 pass 前的支持状态：

1. `pto.treshape` 和 `pto.bitcast` 的 alias 传播。
2. `pto.declare_tile` runtime-bound root：已完成 EmitC、memplan、InsertSync 主路径迁移。
3. `pto.alloc_multi_tile` 的 N-address root：已完成。
4. `pto.multi_tile_get` 的 constant/dynamic slot narrowing：已完成。
5. tile 类型 helper argument 和 `func.call` effects。
6. `scf.yield/iter_arg`、`pto.yield/fusion result` 的 root family 传播检查。
7. `tile_buf_addr` 结果到原 tile root 的 provenance。

`CanPrunePipeVBarrier()` 已支持 `TileBufType`，不需要因删除两个 pass重写；
但新增 view/multi-buffer alias 后必须确认 dependency pair 仍能区分普通
output-input RAW 和 scratch WAW/WAR。

### 10.2 BufidSync 与 Macro Model

检查所有 unwrap helper，去掉对 `bind_tile/memref.subview` 的必需依赖，直接穿透：

- `pto.subview`
- `pto.treshape`
- `pto.bitcast`
- `pto.multi_tile_get`
- `pto.set_validshape`

## 11. 推荐迁移顺序

按照风险从低到高逐 op 处理：

1. `pto.treshape`
2. `pto.bitcast`
3. `pto.set_validshape/get_validshape`
4. `pto.subview`
5. `pto.tile_buf_addr`
6. `pto.declare_tile`
7. `pto.alloc_multi_tile`
8. `pto.multi_tile_get`
9. `pto.make_tensor_view`
10. `pto.partition_view`
11. `pto.get_tensor_view_dim/get_tensor_view_stride`
12. `pto.inttoptr/ptrtoint/addptr/castptr`
13. `pto.load_scalar/store_scalar/initialize_l2g2l_pipe`
14. `scf.if/scf.for/scf.yield`
15. `pto.fusion_region/pto.yield`
16. `func.func/func.call/helper ABI`
17. `vlds/vsts/vsstb`
18. 逐项回归第 9 节 compute op
19. 删除 `bind_tile/declare_tile_memref/slot_marker` 兼容路径
20. 从 `ptoas.cpp` 删除两个 pass，随后删除源文件和 Passes.td 注册

`alloc_multi_tile/multi_tile_get` 迁移已完成：地址规划、slot identity 和动态
event id 均不再依赖 `PTOViewToMemref`。剩余 op 仍按上述顺序逐项迁移。

## 12. 测试矩阵

每批 op 至少运行：

```text
level1 + legacy memplan
level1 + modern memplan
level2 + legacy memplan
level2 + modern memplan
level3 explicit address

A3 + InsertSync
A3 + Graph Sync Solver
A5 + BufidSync

EmitC backend
VPTO backend
```

必须重点恢复/改写以下现有测试类别：

- `materialize_tile_handles_*`
- `multi_tile_*`
- `subview_*`
- `treshape_*`
- `ptr_int_cast.pto`
- PTODSL subkernel helper ABI
- tile fusion control-flow result
- plan memory reused-address sync
- implicit tmp end-to-end

旧测试中检查 `memref.alloc`、`memref.subview`、`pto.bind_tile`、
`pto.pointer_cast` 或 `pto.slot_marker` 的部分，应改为检查 tile-native op、规划后
地址和最终 backend 输出。

## 13. 最终删除条件

只有满足以下条件才能删除两个 pass：

- 主 pipeline 中不再产生需要恢复为 tile 的 local memref；
- legacy/modern memplan 都通过完整测试；
- InsertSync/BufidSync 都能直接分析 tile-native multi-buffer 和 view；
- EmitC/VPTO 不再依赖 bind/memref metadata 回溯；
- helper、控制流、fusion 和 multi-buffer 测试全部通过；
- `rg "PTOViewToMemref|PTOMaterializeTileHandles"` 只剩历史文档；
- `rg "BindTileOp|DeclareTileMemRefOp|SlotMarkerOp"` 不再存在主路径依赖。
