# `mad` 族泛化 lowering 规则设计

## 问题

`mad` 族当前已经拆成 semantic op 和 raw op，但 lowering 仍然不够泛化：

- ordinary MAD 与 MX MAD 的 callee 选择依赖局部 if / fallback，导致 FP8 场景容易串线。
- `X_t`、`CTRL`、bias packing、callee dispatch 分散在多个 helper 中，新增一种类型或模式时不清楚应该改哪一层。
- 部分类型识别依赖字符串匹配，并散落在 emitter 逻辑里。

要解决的问题不是再加一个“更大的 descriptor”，而是定义一组泛化 lowering 规则：

- 不复制 operand。
- 不把可由类型推导的信息再枚举存一份。
- 不让 raw-to-LLVM 重新解释 semantic clause。
- 不允许 ordinary/MX family 互相 fallback。

## 核心原则

### 1. IR 本身是事实源，op interface 是访问入口

lowering 不引入承载 operand 的 descriptor。operand、type、attribute 仍然只存在于原 op 上。
但 lowering 也不应该到处写 `isa<pto::MadOp>` 这种 class 判断。

需要给 semantic MAD 和 raw MAD 各定义一个 op interface，让不同 op class 暴露同一组
accessor 和派生语义：

```c++
enum class MadFamily { Ordinary, Mx };
enum class MadAccumulation { ZeroInit, Accumulate, BiasInit };
enum class MadRawKind { Ordinary, OrdinaryBias, Mx, MxBias };

class MadSemanticOpInterface {
  Value getLhs();
  Value getRhs();
  Value getDst();
  Value getM();
  Value getN();
  Value getK();

  bool hasBiasOperand();
  Value getBiasOrNull();
  bool supportsTf32Mode();
  bool readsAccumulator();
  bool initializesAccumulatorWithZero();
  bool initializesAccumulatorWithBias();

  std::optional<pto::MadUnitFlagMode> getUnitFlagMode();
  bool getDisableGemv();
  std::optional<pto::MadSatMode> getSatMode();
  std::optional<pto::Tf32Mode> getTf32Mode();
  bool getNDir();

  MadFamily getMadFamily();
  MadAccumulation getMadAccumulation();
};

class MadRawOpInterface {
  Value getLhs();
  Value getRhs();
  Value getDst();
  Value getXt();

  bool hasBiasOperand();
  Value getBiasOrNull();

  MadRawKind getMadRawKind();
  MadFamily getMadFamily();
  bool readsAccumulator();
  bool initializesAccumulatorWithZero();
  bool initializesAccumulatorWithBias();
};
```

interface method 可以由 ODS 的 extra class declaration 或 C++ method 实现。关键是：
lowering pattern 只匹配 interface，不直接按 6 个 semantic op class 和 4 个 raw op
class 分别写逻辑。

允许在 interface method 的实现内部有一次 class 分发，因为那是 op 定义层的局部事实；
不允许在 lowering 主流程里散落 class 分发。

MLIR 不会因为原 op 已经有同名 getter 就自动认为它实现了 interface。需要在 ODS
里显式把 interface 加到 op traits 上。interface method 的实现有两种方式：

```tablegen
def MadSemanticOpInterface : OpInterface<"MadSemanticOpInterface"> {
  let cppNamespace = "::mlir::pto";
  let methods = [
    InterfaceMethod<"lhs", "::mlir::Value", "getLhs">,
    ...
  ];
}

def PTO_MadOp : PTO_Op<"mad", [
  MadSemanticOpInterface,
  DeclareOpInterfaceMethods<MemoryEffectsOpInterface>
]> { ... }
```

如果 interface method 没有 default implementation，ODS 只会为实现该 interface 的
op 生成声明，具体定义需要在 C++ 里补齐。若所有实现 op 都有相同名字和相同语义的
generated accessor，可以在 interface method 上写 default implementation：

```tablegen
InterfaceMethod<
  "lhs",
  "::mlir::Value",
  "getMadLhs",
  (ins),
  [{}],
  [{ return $_op.getLhs(); }]
>
```

对 `lhs/rhs/dst/m/n/k` 这类所有 semantic MAD 都同名的字段，可以用 default
implementation 直接转发已有 accessor。对 `bias`、`tf32_mode` 这类并非所有 op
都有的字段，interface 必须提供 capability 方法：

- `getMadFamily()`：ordinary 或 MX，决定 raw family 和 callee lookup family。
- `getMadAccumulation()`：`ZeroInit / Accumulate / BiasInit`，决定 accumulator 初值语义。
- `readsAccumulator()`：只有 acc 模式返回 true；它表示 C 初值来自现有 `%dst`。
- `initializesAccumulatorWithZero()`：zero-init 模式返回 true；它决定 `X_t.c_init = 1`。
- `initializesAccumulatorWithBias()`：bias-init 模式返回 true；它决定 `X_t.c_src = 1`。
- `hasBiasOperand()`：只有 bias-init op 返回 true。
- `getBiasOrNull()`：`hasBiasOperand() == false` 时返回空值。
- `supportsTf32Mode()`：ordinary MAD op 返回 true，MX MAD op 返回 false。
- `getTf32Mode()`：`supportsTf32Mode() == false` 时必须返回 `std::nullopt`。

lowering 只能先看 capability，再使用 optional accessor；不能直接假设所有实现 op
都有 `getBias()` 或 `getTf32ModeAttr()`。

这两个 capability 是正交的，不能用一个 op class 分支同时处理：

| op | family | accumulation | reads acc | zero init | bias init | bias operand | TF32 |
|---|---|---|---:|---:|---:|---:|---:|
| `pto.mad` | Ordinary | ZeroInit | false | true | false | false | true |
| `pto.mad_acc` | Ordinary | Accumulate | true | false | false | false | true |
| `pto.mad_bias` | Ordinary | BiasInit | false | false | true | true | true |
| `pto.mad_mx` | MX | ZeroInit | false | true | false | false | false |
| `pto.mad_mx_acc` | MX | Accumulate | true | false | false | false | false |
| `pto.mad_mx_bias` | MX | BiasInit | false | false | true | true | false |

因此 `mad_bias` 同时是 bias op 和 TF32-capable op；`mad_mx_bias` 是 bias op
但不是 TF32-capable op。`mad_acc` 没有额外 operand，但它是唯一会读取现有 accumulator
作为 C 初值的模式。lowering 不能把 “没有 bias operand” 直接等价成 “zero-init”，
也不能把 “bias op” 和 “不支持 TF32” 绑定在一起。

实现上也不要让 interface default implementation 调用某个并非所有 op 都存在的
generated getter。建议：

- `getLhs/getRhs/getDst` 可以用固定 operand index `0/1/2`。
- `getBiasOrNull` 根据 `hasBiasOperand()` 决定是否返回 operand `3`。
- `getM/getN/getK` 根据 `hasBiasOperand()` 决定从 operand `3/4/5` 或 `4/5/6` 读取。
- `readsAccumulator/initializesAccumulatorWithZero/initializesAccumulatorWithBias`
  从 `getMadAccumulation()` 派生，三者必须互斥且恰好一个为 true。
- `getTf32Mode` 通过通用 attribute lookup 读取 `"tf32_mode"`，而不是调用 generated
  `getTf32Mode()`；MX op 没有这个 attr 时自然返回空。
- verifier 保证 `supportsTf32Mode() == false` 的 op 不能携带 `"tf32_mode"`。

也就是说，interface 的统一性来自“固定 operand organization + capability”，不是来自
假设所有 op 都有完全相同的 generated C++ getter。

因此这里不是“自己写 C++ 继承类”。正确方式是：

1. 在 `PTOInterfaces.td` 定义 op interface。
2. 在每个 MAD ODS op 的 traits 中声明实现该 interface。
3. 能统一转发的 getter 用 interface default implementation。
4. 形态相关的字段，例如 family / accumulation / bias/tf32 capability，用每个 op
   的小实现显式给出。

### 2. family 由 op kind 决定，不由类型猜

ordinary / MX 是 op 语义，不是类型语义：

- `pto.mad*` semantic op 只能 lower 到 ordinary raw family。
- `pto.mad_mx*` semantic op 只能 lower 到 MX raw family。
- `pto.mad_raw` / `pto.mad_bias_raw` 只能发 ordinary MAD。
- `pto.mad_mx_raw` / `pto.mad_mx_bias_raw` 只能发 MX MAD。

类型只用于选择同一 family 内的具体 typed intrinsic。类型不能改变 family。

这是防止普通 FP8 和 MX FP8 串线的关键规则。

### 3. lowering 使用规则函数，不使用大 descriptor

semantic-to-raw 必须有一个统一入口。这个入口负责从 semantic op 生成 raw op
需要的两个运行时值：

- `xt`：raw MAD 的 packed shape/config operand，由 semantic op 的
  `m/n/k` 和 clause 生成。
- `ctrl_for_mad`：本次 MAD 临时使用的控制状态，由 semantic op 的
  numeric/layout clause 和指针类型生成。

规则 helper 只服务这个统一入口，并且接收 interface，而不是裸 `Operation *`：

```c++
MadRawKind deriveRawKind(MadSemanticOpInterface op);
Value buildMadXtFromSemanticOp(MadSemanticOpInterface op,
                               PatternRewriter &rewriter);
Value emitCtrlForMad(MadSemanticOpInterface op, Value ctrlSaved,
                     PatternRewriter &rewriter);
StringRef lookupMadIntrinsic(MadRawOpInterface op);
```

这些函数不返回“重新包装过的 op”。它们只返回当前阶段真正需要的产物。

## semantic-to-raw 规则

semantic-to-raw 的统一入口是：

```c++
LogicalResult lowerMadSemanticOp(MadSemanticOpInterface op,
                                 PatternRewriter &rewriter);
```

这个函数是唯一创建 `xt` 的地方。`xt` 不是外部传入的，也不是 raw-to-LLVM
再生成的；它在 semantic-to-raw 期间由原 semantic op 的 operands/attributes
构造出来，然后作为 operand 传给 raw op。

输入是 semantic op，输出是：

```text
get_ctrl
set_ctrl(ctrl_for_this_mad)
raw op(..., xt)
set_ctrl(ctrl_saved)
```

### raw op 选择

raw op 只由 semantic op 名字决定：

| semantic op | raw op |
|---|---|
| `pto.mad` | `pto.mad_raw` |
| `pto.mad_acc` | `pto.mad_raw` |
| `pto.mad_bias` | `pto.mad_bias_raw` |
| `pto.mad_mx` | `pto.mad_mx_raw` |
| `pto.mad_mx_acc` | `pto.mad_mx_raw` |
| `pto.mad_mx_bias` | `pto.mad_mx_bias_raw` |

这里不需要 descriptor。pattern 使用一个通用
`lowerMadSemanticOp(MadSemanticOpInterface op)`，通过 interface 的
`getMadFamily()` 和 `getMadAccumulation()` 选择 raw op。

### `X_t` 生成

`X_t` 是 raw op 的 packed `xt` operand。它只在
`buildMadXtFromSemanticOp(op)` 中生成，来源是 semantic op 本身：

```text
X_t.M = op.m
X_t.K = op.k
X_t.N = op.n
X_t.unit_flag = op.unit_flag or 0
X_t.disable_gemv = op.has(disable_gemv)
X_t.c_src = op.initializesAccumulatorWithBias()
X_t.c_init = op.initializesAccumulatorWithZero()
```

其中 accumulation 由 semantic op 自己通过 interface 暴露：

```text
mad / mad_mx -> ZeroInit
mad_acc / mad_mx_acc -> Accumulate
mad_bias / mad_mx_bias -> BiasInit
```

这条规则避免把 `c_src/c_init` 存进另一个结构。它们是 op kind 的派生语义。

### `CTRL` 生成

`CTRL` 只由 semantic clause 和指针类型生成：

```text
CTRL[HiF8] = isHiF8(lhs.type, rhs.type)
CTRL[TF32 enable/round] = op.supportsTf32Mode ? op.tf32_mode/default : disabled
CTRL[sat] = op.sat_mode only if explicitly present
CTRL[n_dir] = op.has(n_dir)
```

规则：

- HiF8 必须从 lhs/rhs 指针元素类型推导，不能作为独立 operand 或 enum 保存。
- TF32 只允许 `supportsTf32Mode() == true` 的 ordinary `f32 x f32 -> f32`
  semantic op 使用；MX op 的 `supportsTf32Mode()` 必须为 false，`getTf32Mode()`
  必须返回空值。
- `sat|nosat` 不写时不覆盖对应状态；写了才覆盖。
- `n_dir` 不写时显式设置为默认方向，避免污染后续 MAD。
- semantic-to-raw 必须保存并恢复进入 op 前的 `CTRL`。

### semantic-to-raw 伪码

```c++
LogicalResult lowerMadSemanticOp(MadSemanticOpInterface op,
                                 PatternRewriter &rewriter) {
  // One entry for the entire semantic-to-raw conversion.
  // Every value consumed by the raw op is produced here.
  MadRawKind rawKind = deriveRawKind(op);          // only from interface family/accumulation
  Value xt = buildMadXtFromSemanticOp(op, rewriter); // op.m/n/k + clauses

  Value ctrlSaved = emitGetCtrl();
  Value ctrlForOp = emitCtrlForMad(op, ctrlSaved, rewriter);
  emitSetCtrl(ctrlForOp);

  emitRawOp(rawKind, op, xt, rewriter);       // forwards existing operands

  emitSetCtrl(ctrlSaved);
  erase op;
}
```

注意这里的 `emitRawOp(rawKind, op, xt, rewriter)` 只是把原 op 的现有
data operands 加上刚生成的 `xt` 转发给 raw op，不创建一份新的 operand model。

更具体地说，几个规则函数应当长这样：

```c++
MadRawKind deriveRawKind(MadSemanticOpInterface op) {
  switch (op.getMadFamily()) {
  case MadFamily::Ordinary:
    return op.getMadAccumulation() == MadAccumulation::BiasInit
               ? MadRawKind::OrdinaryBias
               : MadRawKind::Ordinary;
  case MadFamily::Mx:
    return op.getMadAccumulation() == MadAccumulation::BiasInit
               ? MadRawKind::MxBias
               : MadRawKind::Mx;
  }
}

Value buildMadXtFromSemanticOp(MadSemanticOpInterface op,
                               PatternRewriter &rewriter) {
  Value m = op.getM();
  Value n = op.getN();
  Value k = op.getK();

  Value xt = zextOrCastI64(m);
  xt = bitOr(xt, shl(zextOrCastI64(k), 12));
  xt = bitOr(xt, shl(zextOrCastI64(n), 24));

  if (auto mode = op.getUnitFlagMode()) {
    uint64_t bits = *mode == pto::MadUnitFlagMode::CheckOnly ? 2 : 3;
    xt = bitOr(xt, shl(i64(bits), 55));
  }

  if (op.getDisableGemv())
    xt = bitOr(xt, shl(i64(1), 61));

  if (op.initializesAccumulatorWithBias())
    xt = bitOr(xt, shl(i64(1), 62));

  if (op.initializesAccumulatorWithZero())
    xt = bitOr(xt, shl(i64(1), 63));

  return xt;
}

Value emitCtrlForMad(MadSemanticOpInterface op, Value ctrlSaved,
                     PatternRewriter &rewriter) {
  Value ctrl = ctrlSaved;

  // HiF8 is inferred from the existing pointer element types.
  bool hif8 = isHiF8Type(getPtrElementType(op.getLhs()));
  ctrl = setCtrlBit(ctrl, kCtrlHiF8, hif8);

  if (op.supportsTf32Mode()) {
    auto tf32 = op.getTf32Mode();
    ctrl = setCtrlBit(ctrl, kCtrlTf32Enable, true);
    ctrl = setCtrlBit(ctrl, kCtrlTf32RoundAway,
                      tf32 && *tf32 == pto::Tf32Mode::RoundAway);
  } else {
    ctrl = setCtrlBit(ctrl, kCtrlTf32Enable, false);
    ctrl = setCtrlBit(ctrl, kCtrlTf32RoundAway, false);
  }

  // sat/nosat is only an override when the semantic op spells it explicitly.
  if (auto sat = op.getSatMode())
    ctrl = setCtrlBit(ctrl, kCtrlNoSat,
                      *sat == pto::MadSatMode::NoSat);

  ctrl = setCtrlBit(ctrl, kCtrlNDir, op.getNDir());
  return ctrl;
}

void emitRawOp(MadRawKind rawKind, MadSemanticOpInterface op, Value xt,
               PatternRewriter &rewriter) {
  Value lhs = op.getLhs();
  Value rhs = op.getRhs();
  Value dst = op.getDst();

  switch (rawKind) {
  case MadRawKind::Ordinary:
    rewriter.create<pto::MadRawOp>(op.getLoc(), lhs, rhs, dst, xt);
    return;
  case MadRawKind::OrdinaryBias:
    assert(op.hasBiasOperand());
    rewriter.create<pto::MadBiasRawOp>(op.getLoc(), lhs, rhs, dst,
                                       op.getBiasOrNull(), xt);
    return;
  case MadRawKind::Mx:
    rewriter.create<pto::MadMxRawOp>(op.getLoc(), lhs, rhs, dst, xt);
    return;
  case MadRawKind::MxBias:
    assert(op.hasBiasOperand());
    rewriter.create<pto::MadMxBiasRawOp>(op.getLoc(), lhs, rhs, dst,
                                         op.getBiasOrNull(), xt);
    return;
  }
}
```

这里的 `op.getLhs()/op.getM()/op.getUnitFlagMode()` 都来自 interface。它们只从原 op
读取 operand 或 attribute，不缓存、不重组、不创建新的语义对象。

## raw-to-LLVM 规则

raw-to-LLVM 的输入是 raw op。它只做四件事：

1. 从 raw op kind 得到 family。
2. 从 raw op operand type 生成 intrinsic type suffix。
3. 查 family-local intrinsic 表。
4. 发 call。

### family-local dispatch

callee lookup 必须拆成两个互不 fallback 的入口：

```c++
FailureOr<StringRef> lookupOrdinaryMadIntrinsic(Type lhs, Type rhs, Type dst);
FailureOr<StringRef> lookupMxMadIntrinsic(Type lhs, Type rhs, Type dst);
```

调用规则：

```c++
if (op.getMadFamily() == MadFamily::Ordinary)
  callee = lookupOrdinaryMadIntrinsic(lhsElem, rhsElem, dstElem);
else if (op.getMadFamily() == MadFamily::Mx)
  callee = lookupMxMadIntrinsic(lhsElem, rhsElem, dstElem);
```

禁止：

```c++
ordinary lookup failed -> try MX lookup
MX lookup failed -> try ordinary lookup
```

这比 `MadElementFamily` 更直接：类型 suffix 是从 operand type 现场推导的，不需要先存成 enum。

### typed suffix 推导

suffix 推导只回答“这个 type 在当前 family 下叫什么”：

```c++
FailureOr<StringRef> getOrdinaryMadTypeSuffix(Type lhsElem, Type rhsElem,
                                               Type dstElem);

FailureOr<StringRef> getMxMadTypeSuffix(Type lhsElem, Type rhsElem,
                                         Type dstElem);
```

示例：

```text
ordinary:
  f16, f16, f32 -> "f162f32.c310"
  bf16, bf16, f32 -> "bf162f32.c310"
  f32, f32, f32 -> "f322f32.c310"
  e4m3, e4m3, f32 -> "e4m3e4m3.c310"

MX:
  e4m3, e4m3, f32 -> "e4m3e4m3"
  e4m3, e5m2, f32 -> "e4m3e5m2"
```

同样的 FP8 类型组合在 ordinary 和 MX 下可以映射到不同 intrinsic stem，但这个差异由 family-local lookup 决定，不由类型自己决定。

### HiF8 处理

HiF8 不参与 raw-to-LLVM callee 区分：

- HiF8 ordinary MAD 使用 ordinary FP8 typed suffix。
- HiF8 的执行解释由 semantic-to-raw 的 `CTRL` 修改表达。
- raw-to-LLVM 不读取 HiF8 semantic mode，也不设置 `CTRL`。

这保证 HiF8 不会因为 callee 名称选择污染 ordinary FP8。

### bias packing

bias packing 是 raw kind 的机械规则：

```text
mad_raw / mad_mx_raw:
  call dst = dst

mad_bias_raw / mad_mx_bias_raw:
  call dst = pack(dst, bias)
```

它不参与 callee 选择，也不影响 ordinary/MX family。

### raw-to-LLVM 伪码

```c++
LogicalResult emitMadRaw(MadRawOpInterface op,
                         ConversionPatternRewriter &rewriter) {
  Type lhsElem = getPtrElementType(op.getLhs());
  Type rhsElem = getPtrElementType(op.getRhs());
  Type dstElem = getPtrElementType(op.getDst());

  FailureOr<StringRef> callee =
      op.getMadFamily() == MadFamily::Mx
          ? lookupMxMadIntrinsic(lhsElem, rhsElem, dstElem)
          : lookupOrdinaryMadIntrinsic(lhsElem, rhsElem, dstElem);
  if (failed(callee))
    return failure();

  Value lhs = castToLeft(op.getLhs());
  Value rhs = castToRight(op.getRhs());
  Value dst = castToAcc(op.getDst());
  Value callDst = op.hasBiasOperand()
                      ? packDstAndBias(dst, castToBias(op.getBiasOrNull()))
                      : dst;

  emitCall(*callee, callDst, lhs, rhs, op.getXt());
}
```

## 类型识别规则

类型识别不应该在 emitter 中到处 `contains("e4m3")`。需要收敛成 family-local type suffix helper：

```c++
FailureOr<StringRef> getOrdinaryMadElemToken(Type elem);
FailureOr<StringRef> getMxMadElemToken(Type elem);
bool isHiF8Type(Type elem);
```

约束：

- 优先使用 PTO type API。
- 如果某些 FP8/HiF8 类型暂时没有稳定 API，允许在这个 helper 内部有兼容字符串匹配。
- 字符串匹配不得出现在 callee lookup、pattern rewrite、raw lowering 主流程里。
- unsupported target-profile type 在 helper 中失败，不进入 fallback。

这样新增类型只改 type token helper 和对应 family-local suffix 规则。

## 实现组织

建议新增轻量 helper 和 op interface，而不是新增大 descriptor：

```text
include/PTO/IR/PTOInterfaces.td
include/PTO/Transforms/MadLoweringRules.h
lib/PTO/Transforms/MadLoweringRules.cpp
```

放入：

- `MadSemanticOpInterface` / `MadRawOpInterface`
- semantic-to-raw 规则：`deriveRawKind`、`buildMadXtFromSemanticOp`、`emitCtrlForMad`
- raw-to-LLVM 规则：`lookupOrdinaryMadIntrinsic`、`lookupMxMadIntrinsic`
- type token helper：`getOrdinaryMadElemToken`、`getMxMadElemToken`、`isHiF8Type`

不放入：

- operand 副本
- type-family enum 副本
- 与具体 rewriter 强绑定的大型状态对象
- lowering 主流程里的 repeated op class 判断

`VPTOExpandWrapperOps.cpp` 保留 IR 构造和 pattern 注册。
`VPTOCANN900LLVMEmitterTypePatterns.cpp` 与
`VPTOCANN900LLVMEmitterPacking.cpp` 负责 LLVM address-space cast、bias
packing 和 call emission。

## 验收标准

结构验收：

- ordinary raw lowering 只调用 `lookupOrdinaryMadIntrinsic`。
- MX raw lowering 只调用 `lookupMxMadIntrinsic`。
- 两个 lookup 之间没有 fallback。
- semantic-to-raw 主流程匹配 `MadSemanticOpInterface`，不是逐个 op class 模板实例。
- raw-to-LLVM 主流程匹配 `MadRawOpInterface`，不是逐个 raw op class 分支。
- semantic-to-raw 不构造保存 operand 的 descriptor。
- raw-to-LLVM 不读取 semantic clause。
- FP8/HiF8 字符串识别如果存在，只存在于 type token helper。

行为验收：

- MAD SIM 全量通过。
- ordinary FP8 `mad_raw` 静态导向 ordinary `MAD.e4m3e4m3`。
- `mad_mx_raw` / `mad_mx_bias_raw` 静态导向 `MMAD.MX.*`。
- HiF8 + 后续 ordinary FP8 的同 kernel SIM 通过，证明 `CTRL` 不泄漏。
- `sat|nosat`、`tf32_mode`、`n_dir` 的现有 SIM 覆盖仍通过。

## 非目标

本设计不改用户可见 MAD op 语法，不新增 MX scale operand，不改 acc_store 族接口，也不重新定义 `sat|nosat` 数值语义。
