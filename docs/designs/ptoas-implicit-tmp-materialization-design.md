# PTOAS Implicit Tmp Materialization Design

## 背景

PTOAS 前端 IR 中很多 tile op 的 `tmp` operand 是可选的。当前如果用户没有显式写 `tmp`，PTOAS 会继续 lowering 到后端不带 `tmp` 的 C++ 接口。

后续希望改成：前端仍允许省略 `tmp`，但 PTOAS 在内部为需要 tmp-aware 后端接口的 op 自动补充合法 tmp tile，并让这些 tmp tile 和其它 tile buffer 一起进入 memplan 做 local addr 规划。

本文档的目标是给所有类似 `pto.tci` 的 optional tmp op 提供整体改造方案。每个 op 再根据自己的后端 tmp 规格，补充 op-specific 的 tmp requirement、MemoryEffects、verifier 和测试。

## 目标

- 保持前端 IR 兼容：用户仍然可以写不带 `tmp` 的目标 op。
- 在 memplan 之前补齐隐式 tmp，使 tmp 作为普通 local allocation root 参与地址规划。
- 对需要 tmp-aware 后端接口的 op，EmitC lowering 统一走带 tmp 的 C++ overload，避免继续选择 no-tmp 后端接口。
- 每个 op 的 tmp shape、dtype、address space、layout、容量等约束由该 op 的后端接口规格决定。
- 不在 EmitC lowering 中临时分配 tmp 地址。
- 不在 memplan 中特殊创建 tmp；memplan 只负责规划已经存在的 root。

## 非目标

- 本阶段不一次性覆盖所有 optional tmp op。
- 本阶段不改变用户显式提供 tmp 的语义。
- 本阶段不为 level3 自动分配 tmp 地址。
- 本阶段不引入新的全局 workspace 规划。
- 本阶段不把所有 op 的 tmp 规格抽象成完全统一的 shape；不同 op 可以有不同 tmp requirement。

## 总体方案

新增一个 IR 规范化 pass：

```text
pto-materialize-implicit-tmp
```

该 pass 运行在 fusion/调度 pass 之后、`pto-plan-memory` 之前：

```text
PTOFusionRegionGen
  -> pto-materialize-implicit-tmp
  -> PTORematerializeFixpipeVectorQuant
  -> pto-plan-memory (level1/level2 only; skipped at level3)
  -> PTOResolveReservedBuffers
  -> sync passes (InsertSync / GraphSyncSolver / BarrierAll ...)
  -> PTOResolveBufferSelect
  -> EmitPTOManual (PTO -> EmitC lowering)
```

pass 的职责是扫描所有已纳入改造的目标 op。如果 op 没有 tmp operand，就根据该 op 的 `TmpRequirement` 在 op 前插入 `pto.alloc_tile(no addr)`，并重写原 op，使其显式携带 tmp。

抽象流程：

```text
target_op(no tmp)
  -> lookup TmpRequirement(target_op)
  -> create pto.alloc_tile(no addr) tmp
  -> rewrite target_op(with tmp)
  -> memplan assigns addr to tmp
  -> EmitC sees tmp and emits tmp-aware overload
```

`TmpRequirement` 至少应包含：

```text
AddressSpace space;
Type elementType;
StaticShape or MinBytes requirement;
Layout/layout-family requirement;
uint64_t minBytes;
bool requireExplicitAtLevel3;
```

对自动生成的 tmp：

- 使用 tile-native `pto.alloc_tile(no addr)`。
- 不设置 `addr`，由 memplan 统一规划。
- 尽量使用静态 full-valid shape，即 `v_row/v_col` 与 `rows/cols` 一致，不额外携带 `valid_row` / `valid_col` operand。
- 定义位置必须支配目标 op。
- 生命周期由后续 liveness/memplan 根据真实 use 计算。

## Memplan 接入

自动生成的 tmp 是 tile-native `pto.alloc_tile(no addr)`，因此复用当前 memplan 路径：

```text
pto.alloc_tile(no addr)
  -> local allocation root
  -> legacy/modern memplan 分配 offset
  -> pto.alloc_tile addr = ...
```

legacy memplan 和 modern memplan 都应把自动生成的 tmp 当成普通 local allocation root。memplan 不应该知道“这是某个 op 的隐式 tmp”，也不应该在内部临时创建 tmp。

memplan 侧需要依赖 op 的 MemoryEffects / semantic no-alias 信息保证正确复用：

- tmp 如果是 scratch buffer，应通过 `Write(tmp)` 建模，使 scratch-output conflict 能禁止 tmp 与同 op output 错误复用。
- 如果某个 op 的 tmp 与 output 不能 alias，但 tmp 不适合建模成 scratch write，则应在 semantic no-alias side table 中显式加入 `forbidAlias(tmp, output)`。
- 每个 op 的专项改造必须说明 tmp 和 output、input 之间的 alias 约束。

## Level 行为

### level1 / level2

level1/level2 下 memplan 会运行，因此允许省略 tmp：

```text
target_op(no tmp)
  -> pto-materialize-implicit-tmp
  -> pto.alloc_tile(no addr) tmp
  -> pto-plan-memory 补 addr
```

用户显式提供 tmp 时，仍需满足该 op 的 tmp verifier 约束。level1/level2 下用户不应显式指定 local addr，地址由 memplan 统一规划。

### level3

level3 下用户显式管理 local 地址，memplan 通常跳过。pass 通过构造参数 `requireExplicitTmp` 判定 level3（`createPTOMaterializeImplicitTmpPass(effectiveLevel == Level3)`）。

对 **A2/A3 实际使用 tmp 的 op**，level3 不自动创建无地址 tmp，缺省 tmp 直接报错：

```text
level3 + A2/A3 target_op(no tmp) => pass 报错
```

实际诊断字符串（`PTOMaterializeImplicitTmp.cpp`）：

```text
<op> requires explicit tmp when PlanMemory is skipped
```

个别 op 有更具体的变体，例如 binary tcolsum 为 `requires explicit tmp for binary tcolsum when PlanMemory is skipped`，非 32 对齐 tsort32 为 `requires explicit tmp for non-32-aligned tsort32 when PlanMemory is skipped`。

A2/A3 用户在 level3 使用这些 op 时，必须显式提供合法 tmp，并保证 tmp 自身带合法 local addr，或满足现有 level3 显式地址规则。

**A5 例外**：对 A5 只接受但不使用 tmp 的 op，pass 通过 `!isA5` 跳过上述 level3 报错：

- 若后端 C++ 签名仍要求 tmp（row/arg reduction、TXOR/TXORS、TSEL/TSELS、TPRELU、TREM/TREMS 等），pass 即使在 level3 也自动生成固定 32 字节的 ABI placeholder（`makeA5PlaceholderTmpType`，形状 `{1, 32/sizeof(elem)}`）。该 placeholder 不建模 Read/Write MemoryEffects、不参与 memplan、无需回写地址，因此不违反 level3“不自动分配 tmp 地址”的非目标。
- 若后端存在 no-tmp overload（TCI、TROWEXPAND*、TQUANT 等），A5 直接保持 no-tmp 形态，不补 placeholder。

因此当前实现下不存在 “A5 + level3 + 缺省 tmp” 报错的路径：要么 pass 自动补 placeholder，要么保持合法的 no-tmp 形态，verifier 的 no-tmp overload 也无条件接受。

## EmitC Lowering

目标 op 的 EmitC lowering 应保持简单：

- `op.getTmp()` 为空：生成 no-tmp C++ 调用，或者在该 op 改造完成后仅作为未经过 materialize pass 的兜底路径。
- `op.getTmp()` 非空：生成带 tmp 的 C++ 调用。

引入 `pto-materialize-implicit-tmp` 后，level1/level2 的目标 op 在 EmitC 前都会携带 tmp，因此会自然走带 tmp 的 overload。

不建议在 `EmitPTOManual`（PTO -> EmitC lowering）中补 tmp，原因：

- EmitC 阶段已经错过 memplan。
- 临时生成 tmp 无法获得 local addr。
- 会绕过 liveness、sync 和 semantic no-alias 分析。

## TCI 针对性改造

本节描述 `pto.tci` 作为第一批目标 op 的具体落地规则。后续其它 optional tmp op 应新增类似小节，分别说明自己的 tmp 规格、pass 行为、MemoryEffects、verifier 和测试计划。

### TCI Tmp 约束

`pto.tci` 当前 ODS 已经支持可选 tmp：

```td
Optional<PTODpsType>:$tmp
```

`EmitPTOManual`（PTO -> EmitC lowering）也已经根据 `op.getTmp()` 选择带 tmp 或不带 tmp 的 C++ 调用。因此 TCI 改造不需要改 `pto.tci` 的 IR 语法，关键是保证进入 EmitC 前缺省 tmp 已经被显式 materialize。

TCI 后端 C++ 接口存在两类 overload：

```cpp
TCI(dst, start)
TCI(dst, start, tmp)
```

A2/A3 上 no-tmp overload 可能走 scalar loop；带 tmp overload 才能走更优路径。A5 接受 tmp，但 tmp 可以作为兼容占位，不额外引入有效计算约束。

TCI tmp 不应要求固定 shape，应按 PTO-ISA 文档中的精细化 tmp 约束校验容量。PTOAS 对用户显式 tmp 和自动生成 tmp 采用同一组 A2/A3 合法性规则：

```text
loc      = vec
dtype    = 4-byte type: f32 / i32 / ui32
shape    = static shape
layout   = row_major
fractal  = 512
capacity = product(shape) * sizeof(dtype)
```

A2/A3 的最小容量由 dst 元素类型决定：

```text
b32 dst: i32 / ui32  -> tmp capacity >= 768 bytes
b16 dst: i16 / ui16  -> tmp capacity >= 1792 bytes
```

其中 `shape` 可以是任意静态形状，只要总容量满足对应 dst 类型的最小容量。例如 b32 dst 可以使用 `1x192xf32`，b16 dst 可以使用 `1x448xf32`。`Tile<TileType::Vec, float, 1, 512>` 是 PTO-ISA 文档中推荐的方便形状无关分配，容量为 2048 bytes (2KiB)，可以同时覆盖 b32/b16。

A5 上 `tmp` Tile 被接受但不使用；A5 硬件直接使用 `vci` 向量指令，无需临时缓冲区。因此 A5 下 `pto.tci(no tmp)` 不需要自动 materialize tmp，用户显式传 tmp 时也不按 A2/A3 的容量规则校验。

### Pass 行为

对每个 `pto.tci`：

- 如果已经有 `tmp`，pass 不修改。
- A5 如果没有 `tmp`，pass 不修改；A5 后端直接使用 `vci`，不需要 tmp。
- A2/A3 如果没有 `tmp`，且当前 build level 会运行 memplan，则自动补 tmp。
- A2/A3 如果没有 `tmp`，但当前 level3 会跳过 memplan，则报错，要求用户显式提供带地址的 tmp。

重写前：

```mlir
pto.tci ins(%s : i32)
  outs(%dst : !pto.tile_buf<loc=vec, dtype=i32, rows=1, cols=128, ...>)
```

A2/A3 重写后，以下以 b32 dst 自动生成 `f32 1x192` tmp 为例：

```mlir
%tmp = pto.alloc_tile
  : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=192,
                  v_row=1, v_col=192, blayout=row_major,
                  slayout=none_box, fractal=512, pad=0>

pto.tci ins(%s, %tmp : i32,
            !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=192,
                          v_row=1, v_col=192, blayout=row_major,
                          slayout=none_box, fractal=512, pad=0>)
  outs(%dst : !pto.tile_buf<loc=vec, dtype=i32, rows=1, cols=128, ...>)
```

随后 memplan 会把 `%tmp` 当成普通 tile-native local allocation root，和其它 `pto.alloc_tile(no addr)` 一起规划 local address：

```mlir
%tmp = pto.alloc_tile addr = %c4096_i64
  : !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=192, ...>
```

TCI rewrite 需要保留原 op 的：

- scalar operand `S`。
- dst operand。
- `descending` attr。
- location。
- 其它已有属性。

### MemoryEffects

当前 `TCIOp::getEffects()` 只建模为：

```text
Write(dst)
```

A2/A3 自动补 tmp 后，应改为：

```text
Read(tmp) if tmp exists
Write(tmp) if tmp exists
Write(dst)
```

A5 上 tmp 被接受但不使用，因此不应把 tmp 建模为 Read/Write：

```text
Write(dst)
```

原因：

- liveness 需要看到 tmp 在 `pto.tci` 被使用。
- sync pass 需要知道 `pto.tci` 会读 tmp 地址。
- memplan 需要把 tmp 识别为 scratch buffer，避免 tmp 和同 op 的 dst 错误复用。
- modern memplan 的 op semantic no-alias 和 root use 传播需要真实 use 信息。

如果 tmp 不被建模为 Read/Write，tmp 可能被认为没有 use 或不是 scratch，导致生命周期、复用或同步分析不准确。

TCI 也可以在 semantic no-alias side table 中显式加入：

```text
op = pto.tci
forbidAlias(tmp, dst)
```

这不是 scratch conflict 生效的必要条件；A2/A3 只要 `TCIOp::getEffects()` 建模了 `Write(tmp)`，tmp 就会进入 scratch buffer conflict。但显式 side table 能防止未来有人调整 MemoryEffects 后破坏 tmp/dst no-alias 语义。

### Verifier

`TCIOp::verify()` 应检查 tmp 是合法 tile buf，并满足后端接口容量约束：

```text
如果 tmp 存在：
  A5: tmp 被接受但不使用，不执行 A2/A3 tmp 容量校验。
  A2/A3: tmp 必须是 vec tile。
  A2/A3: tmp element type 必须是 4 字节类型（f32 / i32 / ui32）。
  A2/A3: tmp shape 必须是静态 shape。
  A2/A3: tmp layout 必须满足后端 TCI tmp 接口要求。
  A2/A3 b32 dst: tmp capacity 必须大于等于 768 bytes。
  A2/A3 b16 dst: tmp capacity 必须大于等于 1792 bytes。
```

这里的关键是“容量满足接口约束”，而不是“shape 必须等于某个固定值”。TCI 按 dst 元素类型精细化检查：

```text
// b32 dst: 768B 即可
!pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=192, ...>  // 合法
!pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=128, ...>  // 非法，容量不足

// b16 dst: 1792B 即可
!pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=448, ...>  // 合法
!pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=192, ...>  // 非法，容量不足
```

自动生成 tmp 选择满足最小容量的 canonical shape：

- b32 dst: `f32 1x192`。
- b16 dst: `f32 1x448`。
- A5: 不自动生成 tmp。

### 测试计划

#### lit：自动补 tmp

新增用例：

```text
test/lit/pto/tci_implicit_tmp_materialization.pto
```

检查：

```text
CHECK: pto.alloc_tile
CHECK-SAME: dtype=f32
CHECK: pto.tci ins(%{{.*}}, %{{.*}}
```

可选地检查 IR 不打印 `operandSegmentSizes`（可选 tmp 的自定义 assembly printer 已 elide 该属性，保证 round-trip 稳定）。

可以额外检查自动生成 shape：b32 dst 为 `f32 1x192`，b16 dst 为 `f32 1x448`。同时应覆盖 A5 下不自动生成 tmp。

#### lit：memplan 回写 addr

检查 plan memory 后：

```text
CHECK: pto.alloc_tile addr =
CHECK: pto.tci ins(%{{.*}}, %{{.*}}
```

legacy 和 modern 都应覆盖：

```text
// RUN: ptoas --pto-level=level2 --plan-memory-impl=legacy ...
// RUN: ptoas --pto-level=level2 --plan-memory-impl=modern ...
```

#### lit：EmitC 走带 tmp overload

检查 C++ 输出：

```text
CHECK: TCI<
CHECK-SAME: Tile<
CHECK-SAME: float
CHECK: TCI{{.*}}({{.*}}, {{.*}}, {{.*}})
```

#### lit：level3 负例

```text
level3 + pto.tci(no tmp)
```

期望：

```text
expected-error {{pto.tci requires explicit tmp when compiling at level3}}
```

#### lit：verifier 负例

用户显式提供非法 tmp：

- 非 vec space。
- 非 f32 dtype。
- dynamic shape。
- layout 不满足 TCI tmp 接口约束。
- A2/A3 b32 dst 的 tmp capacity 小于 768 bytes。
- A2/A3 b16 dst 的 tmp capacity 小于 1792 bytes。

期望 verifier 报错。

## TROWEXPAND 二元 op 针对性改造

本节覆盖以下 row-expand 二元 op：

```text
pto.trowexpandadd
pto.trowexpandsub
pto.trowexpandmul
pto.trowexpanddiv
pto.trowexpandmax
pto.trowexpandmin
```

这些 op 的 PTO-ISA 文档对 tmp 的描述一致：带 `TileDataTmp &tmp` 的 C++ overload 仅支持模式 1；A2/A3 上 tmp 用作行广播缓冲区；A5 接受 tmp 但不使用。

### RowExpand Tmp 约束

这些 op 有两种 row-broadcast 模式：

- 模式 1：扩展操作数为 `ColMajor`，每行一个标量。带 tmp overload 仅支持该模式。
- 模式 2：扩展操作数为 `RowMajor`，每行一个 32 字节块。该模式不需要 tmp，不应为了 tmp-aware overload 强行改写。

A2/A3 模式 1 下，tmp 作为 `vbrcb` 广播缓冲区使用。扩展操作数的每行标量会广播成一个 32 字节块；`vbrcb` repeat stride 为 8 个块，即 256 字节，每个 repeat 处理 8 行。

tmp 最小容量由 `R = dst.validRow` 决定：

```text
if R < 256:
  tmpBytes = ceil(R / 8) * 256
else:
  tmpBytes = 30 * 256 = 7680
```

说明：

- 当 `R >= 256` 时，后端按循环处理，每次循环最多 30 个 repeat，也就是 240 行；tmp 在循环间复用，因此每次循环只需要 7680 字节。
- 一个紧凑的形状无关上界是 8KB，即 8192 字节。该上界可作为自动 materialize 的保守 canonical tmp 大小。
- 不带 tmp 的 3 参数 overload 支持模式 1 和模式 2；对 A2/A3 的模式 1，后端使用内部 8KB 缓冲区 `TMP_UB_OFFSET`；模式 2 不需要广播缓冲区。
- A5 硬件通过 `vlds` 广播模式原生支持行广播，tmp 被接口接受但不使用。

PTOAS 对用户显式 tmp 的合法性规则：

```text
A2/A3:
  op 必须是模式 1，才能使用显式 tmp。
  tmp 必须是 vec tile。
  tmp shape 必须静态可计算容量，或后续 verifier 能证明容量满足公式。
  tmp capacity >= min(ceil(R / 8) * 256, 7680)。

A5:
  tmp 被接受但不使用，不执行 A2/A3 tmp 容量校验。
```

### Pass 行为

对每个目标 row-expand 二元 op：

- 如果已经有 `tmp`，pass 不修改，但 verifier 需要保证它只用于合法模式。
- A5 如果没有 `tmp`，pass 不修改。
- A2/A3 如果没有 `tmp`，且 op 是模式 1、当前 build level 会运行 memplan，则自动补 tmp。
- A2/A3 如果没有 `tmp`，op 是模式 1、但当前 level3 会跳过 memplan，则保留 no-tmp overload，由后端使用内部 8KB `TMP_UB_OFFSET`，避免 pass 生成无地址 tmp。
- 模式 2 不需要 tmp；pass 不应自动补 tmp，也不应强制改成带 tmp overload。

自动补 tmp 的 canonical shape 建议采用形状无关上界：

```mlir
%tmp = pto.alloc_tile
  : !pto.tile_buf<loc=vec, dtype=<dst element type>,
                  rows=1, cols=<8192 / sizeof(dst element type)>,
                  v_row=1, v_col=<8192 / sizeof(dst element type)>,
                  blayout=row_major, slayout=none_box,
                  fractal=512, pad=0>
```

默认使用 `dst element type` 作为 tmp element type，以贴合 row-expand 后端模板参数；如果后续确认某些后端实现允许更宽松的 tmp dtype，可在对应 op-specific verifier 中放宽。

这样不需要在 materialize pass 中依赖 `dst.validRow` 是否为静态值，也能覆盖 A2/A3 模式 1 的最大每轮 tmp 需求。后续如果希望节省 UB，可以在能静态证明 `R` 时生成更小 tmp：

```text
tmpBytes = min(ceil(R / 8) * 256, 7680)
```

### MemoryEffects

A2/A3 上这些 op 的 tmp 是广播 scratch buffer，应该建模为：

```text
Read(non-tmp inputs)
Read(tmp) if tmp exists
Write(tmp) if tmp exists
Write(dst)
```

其中 `Write(tmp)` 用于让 memplan 的 scratch-output conflict 禁止 tmp 与同 op 的 `dst` 错误复用。

A5 上 tmp 被接受但不使用，因此不应把 tmp 建模为 Read/Write：

```text
Read(non-tmp inputs)
Write(dst)
```

如果未来某个 row-expand op 的 MemoryEffects 不适合用 `Write(tmp)` 表达 scratch 语义，也应在 semantic no-alias side table 中显式加入：

```text
op = pto.trowexpand*
forbidAlias(tmp, dst)
```

### Verifier

这些 op 的 verifier 需要区分模式和 arch：

```text
如果 tmp 存在：
  A5: tmp 被接受但不使用，不执行 A2/A3 tmp 容量校验。
  A2/A3: op 必须是模式 1，即扩展操作数为 ColMajor 每行标量。
  A2/A3: tmp 必须是 vec tile。
  A2/A3: tmp capacity 必须满足 min(ceil(dst.validRow / 8) * 256, 7680)。
```

模式识别规则沿用 ISA 文档：

- `src0` 或 `src1` 中恰好一个与 `dst` 有相同 valid shape，该 operand 是全尺寸操作数。
- 另一个 operand 是扩展操作数。
- 扩展操作数为 `ColMajor` 且每行一个标量时是模式 1。
- 扩展操作数为 `RowMajor` 且每行 `32 / sizeof(T)` 列时是模式 2。

如果 `dst.validRow` 是动态值，verifier 无法精确证明用户显式 tmp 是否足够小时，可以采用保守规则：

- 用户显式 tmp 至少 8192 字节；或
- 后续引入运行时/符号约束证明 tmp capacity 满足公式。

自动生成 tmp 建议先使用 8192 字节 canonical 上界，因此不会受动态 `dst.validRow` 影响。

### 测试计划

#### lit：自动补 tmp

为至少一个代表 op 增加 A2/A3 模式 1 用例，例如 `pto.trowexpandadd(no tmp)`：

```text
CHECK: pto.alloc_tile
CHECK: pto.trowexpandadd ins(%{{.*}}, %{{.*}}, %{{.*}}
```

同时检查 A5 下不自动生成 tmp。

#### lit：模式 2 不补 tmp

构造 RowMajor 扩展操作数的模式 2 用例，确认 pass 不自动补 tmp，并继续走 no-tmp overload。

#### lit：memplan 回写 addr

检查自动生成 tmp 在 plan memory 后带 `addr`：

```text
CHECK: pto.alloc_tile addr =
CHECK: pto.trowexpand{{.*}} ins(%{{.*}}, %{{.*}}, %{{.*}}
```

legacy 和 modern 都应覆盖。

#### lit：level3 负例

A2/A3 level3 + 模式 1 + no tmp 应报错：

```text
expected-error {{requires explicit tmp when compiling at level3}}
```

A5 level3 + no tmp 不应因为 tmp 缺失报错。

#### lit：verifier 负例

需要覆盖：

- A2/A3 显式 tmp 用在模式 2，报错。
- A2/A3 显式 tmp capacity 小于公式要求，报错。
- A2/A3 动态 `dst.validRow` 且显式 tmp 小于 8192 字节，按保守规则报错。
- A5 显式 tmp 不触发 A2/A3 容量校验。

## 批量 optional tmp op 分类设计

本节按 PTO-ISA tmp 行为对后续待改造 op 做分类。输入列表中的重复项只记录一次：

```text
TCOLARGMAX, TCOLARGMIN, TROWARGMAX, TROWARGMIN,
TADDDEQRELU, TGATHER, TTRANS, TXOR, TXORS, TPRELU, TMRGSORT,
TROWPROD, TCOLSUM, TROWSUM, TROWMAX, TROWMIN,
TSEL, TSELS, TRSQRT, TPOW, TPOWS, TREM, TREMS, TCVT,
TSORT32, TQUANT
```

其中 `TADDDEQRELU` 对应 PTO-ISA 文档 `TAddDeqRelu_zh.md`；当前 PTOAS ODS 中未找到同名 op，先标记为待 IR 接入。

### 分类总览

| 分类 | Op / 模式 | 设计结论 |
| --- | --- | --- |
| A2/A3 使用 tmp，A5 接受但不使用 | `TCOLARGMAX`、`TCOLARGMIN`、`TROWARGMAX`、`TROWARGMIN`、`TROWPROD`、`TROWSUM`、`TROWMAX`、`TROWMIN`、`TSEL`、`TSELS`、`TREM`、`TREMS`、`TQUANT`、`TADDDEQRELU` | level1/2 在 A2/A3 生成真实 scratch；若 A5 C++ 签名仍要求 tmp，则生成不带 MemoryEffects 的 ABI placeholder。 |
| A2/A3 和 A5 都可能使用 tmp | `TCOLSUM(isBinary=true)`、`TSORT32` 非 32 对齐尾部、`TMRGSORT` 多列表归并 format2 | tmp 使用由 op 模式决定，不能只按 arch 判断。 |
| 条件性 tmp，不应无条件 materialize | `TTRANS`、`TCVT`、`TPOW`、`TPOWS`、`TRSQRT`、`TMRGSORT`、`TSORT32` | 需要先判断精度、dtype、layout、format 或尾部条件。 |
| 已从 mandatory tmp 改为 optional tmp | `TTRANS`、`TXOR`、`TXORS`、`TPRELU`、`TROWPROD`、`TROWSUM`、`TROWMAX`、`TROWMIN`、`TROWARGMAX`、`TROWARGMIN`、`TCOLARGMAX`、`TCOLARGMIN`、`TSEL`、`TSELS`、`TREM`、`TREMS` | ODS、parse/print、verifier、MemoryEffects、materialize 和 lowering 已接入。 |
| 当前 PTOAS IR 已有 optional tmp | `TCOLSUM`、`TRSQRT`、`TPOW`、`TPOWS`、`TSORT32`、`TQUANT` | 可直接纳入 `pto-materialize-implicit-tmp` 的后续实现。 |
| 已有 optional tmp 但不纳入 implicit-tmp materialize | `TGATHER` | main 分支要求 A2/A3 显式 tmp（verifier 拒绝省略），A5 index-form 设计为无 tmp；`replaceTGatherWithTmp` 实现保留但当前不在 dispatch 中启用。 |
| 当前 PTOAS IR 暂无对应 op | `TADDDEQRELU` | 需先完成 PTOAS IR 接入；`TCVT` 已新增 optional tmp operand。 |

### 通用规则

- level1/level2：只有该 op 在当前 arch / 模式下实际需要 tmp，且 IR 允许省略 tmp 时，才自动 materialize `pto.alloc_tile(no addr)`。
- level3：若该 op 在 A2/A3 当前模式下需要 tmp 且用户省略 tmp，则报错（`requires explicit tmp when PlanMemory is skipped`）。A5 见下一条，即使 level3 也不因缺省 tmp 报错。
- A5 仅接受但不使用 tmp 的 op：若后端存在 no-tmp overload，则不自动补 tmp；若 C++ 签名仍要求 tmp，则自动生成固定 32 字节 ABI placeholder（level1/2/3 一致，通过 `!isA5` 绕过 level3 显式 tmp 检查）。placeholder 不建模 tmp 的 Read/Write（`getEffects` 以 `!tmp.empty() && arch != A5` 守卫）、不参与 memplan、无需地址，用户显式 tmp 也不按 A2/A3 容量规则校验。
- tmp 是 scratch 的 op：MemoryEffects 需要建模为 `Read(tmp) + Write(tmp)`，或在 semantic no-alias side table 中显式加入 `forbidAlias(tmp, dst/output)`。
- 原 mandatory tmp op 已统一改为 optional，并保持显式 tmp 文本格式兼容。
- 容量校验统一走 `verifyTmpCapacityAtLeast`，按 tmp 的**声明 shape** × `sizeof(dtype)` 计算（`getStaticByteSize`，非 valid 区域），A5 分支不执行该校验。对 row/arg reduction 的 32 字节下限：因 `pto.alloc_tile` 已对 row-major none_box tile 强制行 `cols * sizeof(dtype)` 32 字节对齐，而 reduction `src` 必须是这类 tile，故任何合法 src 单行即 ≥32 字节，同形状 tmp 恒满足下限；materialize 无需为 sub-block src 额外兜底容量（<32 字节的合法 reduction src 无法构造）。

### Arg reduction 类

覆盖：

```text
TCOLARGMAX, TCOLARGMIN, TROWARGMAX, TROWARGMIN
```

现状：

- 这些 op 的 tmp 已改为 optional，并接入 `pto-materialize-implicit-tmp`。
- A2/A3 使用 tmp；A5 接受 tmp 但不使用。

TCOLARGMAX / TCOLARGMIN：

- tmp dtype 必须与 `src` 一致。
- tmp 用于索引跟踪和当前比较值临时存储。
- tmp 容量需要按 `tmpGapEles` 和输出模式计算：当 `srcValidCol >= elemPerRpt` 时，`tmpGapEles = elemPerRpt`；否则 `tmpGapEles = ceil(srcValidCol / elemPerBlock) * elemPerBlock`。
- half + 纯索引模式是 tmp 使用量最大的组合；其它类型 / 模式下 tmp 中可能只需要区域 0，但自动 materialize 可先采用覆盖最大需求的保守形状。

TROWARGMAX / TROWARGMIN：

- 仅索引模式在 A2/A3 可能不使用 tmp；值+索引模式和两阶段归约需要 tmp。
- tmp 行数与 `src` 相同；每行 stride 按 PTO-ISA 文档公式计算。
- 当前 PTOAS ODS 只有单输出索引模式；该模式仍需满足后端显式 tmp 参数签名，因此 level1/2 生成保守同形状 tmp。未来接入值+索引模式后，再按输出模式和归约阶段收紧容量。

容量校验：

- A2/A3 verifier 在按 `tmpGapEles` / 布局（DN 1 列、ND 2 列、min stride 等）校验后，统一以 `verifyTmpCapacityAtLeast(op, tmp, 32)` 兜底；容量按声明 shape 计算，同 row reduction 一样被 `pto.alloc_tile` 的 32 字节行对齐恒满足。
- A5 arg-reduction verifier（`verifyTColArgReductionOpA5` / `verifyTRowArgReductionOpA5`）不含任何容量校验。

MemoryEffects / alias：

- A2/A3 实际使用 tmp 时建模 `Read(src) + Read(tmp) + Write(tmp) + Write(dstIdx/dstVal)`。
- A5 placeholder 不建模 tmp 的 Read/Write（`getEffects` 以 `!tmp.empty() && arch != A5` 守卫）。
- tmp 不应与同 op 的输出 alias；如果 MemoryEffects 无法覆盖，应加入 `forbidAlias(tmp, dstIdx)` 和必要的 `forbidAlias(tmp, dstVal)`。

### Row reduction 类

覆盖：

```text
TROWPROD, TROWSUM, TROWMAX, TROWMIN
```

现状：

- 这些 op 的 tmp 已改为 optional，并接入 `pto-materialize-implicit-tmp`。
- A2/A3 使用 tmp；A5 接受 tmp 但不使用。

tmp 规格：

- tmp dtype 与 `src` / `dst` 一致。
- ISA 最小需求为 1 行 1 个 vector block（32 字节）：`int32` 为 8 列，`int16` 为 16 列；浮点二叉树归约同样以 1 个 block 为下限。
- A2/A3 materialize 直接生成与 `src` 同形状的 tmp（`makeSameShapeTmpType`），不做逐 dtype 的特化裁剪；`TROWPROD` 亦同。
- 容量校验：A2/A3 verifier 走 `verifyTmpCapacityAtLeast(op, tmp, 32)`，按声明 shape × `sizeof(dtype)` 计算。**该 32 字节下限对合法 IR 恒被满足**——reduction `src` 必须是 row-major none_box tile，`pto.alloc_tile` 已对其强制行 32 字节对齐，故同形状 tmp 单行即 ≥32 字节，无需为 sub-block src 特殊兜底。

Pass 行为：

- A2/A3 level1/2：若 IR 已支持 optional tmp 且缺省 tmp，则自动生成与 `src` 同形状的 vec row-major none-box tmp。
- A5：生成后端签名需要的固定 32 字节 ABI placeholder，但不为其添加 tmp MemoryEffects，也不跑 32 字节容量校验。
- level3：A2/A3 需要 tmp 时缺省 tmp 报错（`requires explicit tmp when PlanMemory is skipped`）；A5 通过 `!isA5` 跳过报错，仍自动生成 ABI placeholder。

MemoryEffects / alias：

- A2/A3 建模 `Read(src) + Read(tmp) + Write(tmp) + Write(dst)`。
- A5 placeholder 不建模 tmp 的 Read/Write（`getEffects` 以 `!tmp.empty() && arch != A5` 守卫）。
- tmp 与 `dst` 禁止 alias。

### Column sum 类

覆盖：

```text
TCOLSUM
```

现状：

- 当前 PTOAS IR 已支持 optional tmp。
- no-tmp 形式表示顺序累加，不需要 tmp。
- `isBinary=true` 时 A2/A3 和 A5 都使用 tmp 做二叉树累加。

tmp 规格：

- tmp dtype 与 `src` / `dst` 一致。
- tmp 为 vec row-major none-box tile。
- `tmp.validCol >= src.validCol`。
- `tmp.validRow >= ceil(src.validRow / 2)`。

Pass 行为：

- 仅当 `isBinary=true` 且缺省 tmp 时自动 materialize。
- `isBinary=false` 不自动补 tmp。
- level3 下 `isBinary=true` 且缺省 tmp 报错。

MemoryEffects / alias：

- `isBinary=true` 且 tmp 存在时，建模 `Read(src) + Read(tmp) + Write(tmp) + Write(dst)`。
- `isBinary=false` 且 tmp 缺省时，保持 no-tmp 顺序累加语义。

### Elementwise scratch 类

覆盖：

```text
TXOR, TXORS, TPRELU, TREM, TREMS, TADDDEQRELU, TQUANT
```

现状：

- `TXOR`、`TXORS`、`TPRELU`、`TREM`、`TREMS` 的 tmp 均已改为 optional 并接入 materialize pass。
- `TQUANT` 当前 PTOAS IR 已支持 optional tmp。
- `TADDDEQRELU` 当前 PTOAS ODS 中未找到同名 op，需先完成 IR 接入。

tmp 规格：

- `TXOR` / `TXORS`：A2/A3 tmp dtype 与输入输出一致，row-major，容量覆盖 `dst` 有效区域；A5 不使用 tmp。
- `TPRELU`：A2/A3 tmp dtype 为 `uint8_t`，row-major，`tmp.validRow > dst.validRow`，用于 mask buffer；A5 不使用 tmp。
- `TREM`：A2/A3 tmp dtype 与 `dst` 一致，至少 2 行和 `dst.validCol` 列；A5 不使用 tmp。
- `TREMS`：A2/A3 tmp dtype 与 `dst` 一致，至少 1 行和 `dst.validCol` 列；A5 不使用 tmp。
- `TADDDEQRELU`：A2/A3 tmp dtype 为 `int32_t`，容量至少覆盖 `dst` 有效区域；A5 不使用 tmp。
- `TQUANT`：A2/A3 tmp 为 FP32，形状与 `src` 同尺寸，用作 FP32 到 S32 转换中间结果；A5 不使用 tmp。

Pass 行为：

- A2/A3 level1/2：缺省 tmp 且 IR 支持 optional tmp 时自动 materialize。
- A5：`TXOR/TXORS` 因后端签名要求生成 ABI placeholder；其余 op 按各自后端是否存在 no-tmp overload 决定。
- level3：A2/A3 缺省 tmp 报错；A5 不强制 tmp。

MemoryEffects / alias：

- A2/A3 tmp 是 scratch 时建模 `Read(tmp) + Write(tmp)`。
- tmp 与 `dst` 禁止 alias；`TXOR/TXORS/TPRELU/TREM/TREMS/TADDDEQRELU` 还应禁止 tmp 与同 op 输入错误 alias。

### Mask select 类

覆盖：

```text
TSEL, TSELS
```

现状：

- 当前 PTOAS IR 中 tmp 是 mandatory；隐式 tmp 支持前需要先改成 optional tmp。
- A2/A3 使用 tmp；A5 接受 tmp 但不使用。

tmp 规格：

- `TSEL`：tmp dtype 为 `uint32_t`，用于 mask buffer。16 位数据类型的 `cmpmaskLen = 4` 个 `uint32_t`；32 位数据类型的 `cmpmaskLen = 2` 个 `uint32_t`。
- `TSELS`：tmp dtype 与 `src` 一致，至少 1 个元素，用于保存 scalar 和比较 mask。

Pass 行为：

- A2/A3 level1/2：缺省 tmp 时自动 materialize。
- A5：不自动补 tmp。
- level3：A2/A3 缺省 tmp 报错；A5 不强制 tmp。

MemoryEffects / alias：

- A2/A3 建模 `Read(mask/src) + Read(tmp) + Write(tmp) + Write(dst)`。
- tmp 与 `dst` 禁止 alias。

### Data movement / layout 类

覆盖：

```text
TGATHER, TTRANS, TCVT
```

TGATHER：

- 当前 PTOAS IR 已支持 optional tmp，但 **tgather 不纳入 implicit-tmp materialize 范围**。
- main 分支（PR #1080 "Add TGATHER indices and mask"）对 tgather 的 tmp 契约更严：A2/A3 index-form 和所有 compare-form 都要求显式 `tmp`（verifier 报 `index-form tgather expects both indices and tmp` / `compare-form tgather expects dst, cdst, kValue, and tmp`）；A5 index-form 设计为不带 tmp（emit `TGATHER(src, indices, dst)` 三参数）。
- 因此 tgather 省略 tmp 时不由 `pto-materialize-implicit-tmp` 自动补齐，而是由 verifier 直接拒绝（A2/A3）或允许无 tmp（A5 index-form）。
- index form：A2/A3 C++ API 需要 tmp；tmp dtype 与 indices dtype 一致，shape 覆盖 indices；A5 不使用 tmp。
- compare form：A2/A3 tmp 是合并暂存缓冲区，包含 `cmpsTmp`、`indexTmp`、`cvtTmp` 三个区域；最小字节数按 PTO-ISA 文档公式计算；A5 不使用 tmp。
- mask form 不使用 tmp。
- A2/A3 index / compare form 必须显式提供 tmp；A5 index-form 不带 tmp。

TTRANS：

- 当前 PTOAS IR 中 tmp 已改为 optional，并接入 materialize pass。
- tmp 只在满足高效转置路径条件时使用；scalar copy 和部分 layout 转换不需要 tmp。
- 静态满足 stride 条件时生成与 src 同形状的保守 scratch；不满足并走 scalar copy 时生成 32 字节 ABI placeholder。
- 只有真实 scratch 进入 `Read(tmp) + Write(tmp)` MemoryEffects；scalar-copy placeholder 不建模内存访问。

TCVT：

- 当前 PTOAS IR 已新增 optional tmp operand，并完成 parser/printer、verifier、MemoryEffects、materialize 和 EmitC lowering。
- A2/A3 仅在 `SaturationMode::OFF` 的 PyTorch 兼容非饱和窄化路径使用 tmp：`float -> int16`、`half -> int16`、`half -> int8`。
- 其它转换不需要 tmp，不应自动 materialize。
- tmp 按字节规划，容量使用 PTO-ISA 的 `tmpFloatToInt16Bytes`、`tmpHalfToInt16Bytes`、`tmpHalfToInt8Bytes` 公式。
- level3 下上述三种路径缺省 tmp 报错；其它转换继续使用 no-tmp overload。

MemoryEffects / alias：

- 只有实际使用 tmp 的路径建模 `Read(tmp) + Write(tmp)`。
- tmp 与 `dst` 禁止 alias。

### Sort / merge 类

覆盖：

```text
TSORT32, TMRGSORT
```

TSORT32：

- 当前 PTOAS IR 已支持 optional tmp。
- 3 参数形式适用于 `validCol` 已按 32 对齐的路径，不需要 tmp。
- 4 参数形式用于非 32 对齐尾部，通过 tmp 保存填充后的行或尾块副本。
- tmp dtype 与 `src` 一致；容量按 PTO-ISA `tmpSize` 公式计算，不能固定为 8KB。
- level1/2 仅在能静态证明存在非 32 对齐尾部时自动补 tmp；否则保持 no-tmp。
- level3 下非 32 对齐尾部缺省 tmp 报错。

TMRGSORT：

- 当前 PTOAS IR 已支持 optional tmp；缺省 format2 使用显式 `no_tmp` 中间语法消除 src/tmp 个数歧义。
- format1 单输入 block sort 不需要 tmp。
- format2 多列表归并需要 tmp 和 executed list。
- level1/2 对 format2 `no_tmp` 生成一行 row-major tmp，`tmp.cols = sum(src.cols)`；format1 不补 tmp。
- level3 下 format2 `no_tmp` 报错。

MemoryEffects / alias：

- 使用 tmp 的 sort / merge 路径建模 `Read(tmp) + Write(tmp)`。
- tmp 与 `dst` 禁止 alias；`TMRGSORT` 还需考虑 `executed` output 的写 effect。

### Pow / rsqrt 类

覆盖：

```text
TPOW, TPOWS, TRSQRT
```

TPOW / TPOWS：

- 当前 PTOAS IR 已支持 optional tmp。
- A2/A3 浮点路径使用 tmp；整数路径不使用 tmp。
- A5 接受 tmp 但不使用。
- tmp dtype 与 `dst` / `base` 一致，容量覆盖 `dst` 有效区域。
- level1/2：A2/A3 浮点路径缺省 tmp 时自动 materialize；整数路径不补 tmp。
- level3：A2/A3 浮点路径缺省 tmp 报错；整数路径不强制 tmp。

TRSQRT：

- 当前 PTOAS IR 已支持 optional tmp。
- no-tmp 默认实现不需要 tmp。
- 带 tmp overload 当前仅作为 API 兼容 / 未来高精度路径保留，现阶段不自动 materialize。
- 用户显式 tmp 时，A5 不按 A2/A3 scratch 容量规则校验。

MemoryEffects / alias：

- `TPOW/TPOWS` 只有浮点 tmp-backed 路径建模 `Read(tmp) + Write(tmp)`。
- `TRSQRT` 现阶段不因缺省 tmp 增加 MemoryEffects。

### 批量 op 测试策略

后续按类别实现时，每一类至少补以下 lit：

- 自动补 tmp：level1/2 + A2/A3 + 缺省 tmp，检查 `pto.alloc_tile(no addr)` 经 memplan 后带 `addr`。
- A5 不补 tmp：A5 + 缺省 tmp，检查保持 no-tmp 形态。
- level3 负例：A2/A3 + 需要 tmp + 缺省 tmp 报错。
- verifier 负例：显式 tmp dtype / shape / capacity / layout 不满足对应 op 规格时报错。
- EmitC overload：需要 tmp 的路径最终走 tmp-aware C++ 调用；不需要 tmp 的路径不强行切换。

## 后续扩展

后续新增其它 optional tmp op 时，需要补充一个 op-specific 小节，并明确：

- op 的 tmp 后端接口规格。
- 自动生成 tmp 的 canonical shape。
- 用户显式 tmp 的 verifier 规则。
- tmp 的 MemoryEffects。
- tmp 与 input/output 的 alias 约束。
- level3 下是否要求显式 tmp。
- lit 覆盖自动补 tmp、memplan 回写 addr、EmitC overload、level3 负例和 verifier 负例。
