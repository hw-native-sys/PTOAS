# ADR-0003: VMI TileLib 候选 dtype 覆盖扩展至 NUMERIC_DTYPES

- 日期: 2026-08-17
- 状态: Proposed
- 关联: ADR-0001 VMI VF Fusion canonical pipeline、ADR-0002 VMI 资源感知融合与 codegen
- 行号基准: 本文档引用的代码行号基于 commit `5b28aa887`（`main-llvm19-build` HEAD）。后续 commit 可能导致行号漂移；以函数名/符号名为准。

## Context

### 现状：普通 PTODSL 与 VMI 候选的 dtype 覆盖不对齐

普通 PTODSL elementwise 候选使用 `_common.py` 的 `NUMERIC_DTYPES`
（`lib/TileOps/a5/_common.py:17`，共 9 种：`f32/f16/bf16/i8/i16/i32/ui8/ui16/ui32`），
而 VMI elementwise 候选普遍锁定 `f32`。按 family 实测的差距表：

| op family | VMI 候选声明的 dtype | 普通 PTODSL 覆盖 |
|---|---|---|
| elementwise 二元 `tadd/tmul/tsub/tmax/tmin`（`tadd.py:52,63` 等） | 仅 `(("f32","f32","f32"),)` | `NUMERIC_DTYPES` 全 9 种 |
| elementwise 一元 `tneg/tabs`（`tneg.py:53`、`tabs.py:46`） | 仅 `(("f32","f32"),)` | 全 9 种 |
| 向量-标量 `tadds/tmuls/tmaxs/tmins/tsubs`（`tadds.py:48` 等） | 仅 `(("f32","f32","f32"),)` | 全 9 种 |
| 指数/对数/平方根 `texp/tlog/tsqrt/trsqrt`（`texp.py:50` 等） | 仅 `(("f32","f32"),)`，但 emit 内 `allowed_dtypes=FLOAT_DTYPES` | 全 9 种 |
| 倒数 `trecip`（`trecip.py:96`）、除法 `tdiv`（`tdiv.py:103`） | `("f16","f16","f16"),("f32","f32","f32")` | 全 9 种 |
| 搬运 `tmov`（`tmov.py:104`） | `(("f32","f32"),("f16","f16"),("bf16","bf16"))` | 全 9 种 |
| 转换 `tcvt`（`tcvt.py:1861` `vmi_tcvt`） | 按 dtype **对**支持（`f32↔bf16/f16/i32`、`i32→f16` 等多对） | 按 dtype 对 |
| 行规约 `trowmax/trowsum`（`trowmax.py:38,55,71`、`trowsum.py:32,49,65`） | 仅 `(("f32","f32","f32"),)` | 全 9 种 |
| 列规约 `tcolmax/tcolmin/tcolsum`（`tcolmax.py:43` 等） | 仅 `(("f32","f32"),)` | 全 9 种 |
| col/row 广播二元 `tcolexpandadd/trowexpandmul` 等 | 仅 `(("f32","f32","f32"),)` | `NUMERIC_SIGNATURES` |

### 影响

`bf16` 是 A5 大模型主力 dtype（权重/激活），但开 `--enable-vmi` 后，`tadd/tmul` 的
bf16 tile 因 VMI 候选 `dtypes=(("f32",...))` 根本不进候选列表，永远走普通 PTODSL
回退、吃不到 VMI 循环融合 + load/store 消除收益。`i8/i16` 同理。dtype 覆盖不对齐
导致**回退不一致**：相同 shape 下，仅因 dtype 不同而走向不同的优化路径。

### 既有正确基础设施（扩 dtype 不破坏的部分）

本 ADR 的关键前提：**框架已 dtype-agnostic，差距集中在 Python 候选的 `dtypes=` 声明
+ 几处硬编码 `f32.lanes`/`_DTYPE_BYTEWIDTH["f32"]`**。已验证就绪的基础设施：

| 组件 | 位置 | 为何 dtype 无关 |
|---|---|---|
| `ScalarType` 的 `lanes/bytewidth/mask_bits` | `ptodsl/ptodsl/_tile_template_tracing.py:80-85` | 每种 dtype 显式 `lanes*bytewidth==256`（一个 A5 物理 VREG），`mask_bits`==元素位宽 |
| `emit_elementwise_vmi` / `_validate_elementwise_tiles` | `lib/TileOps/a5/_vmi_common.py:1030,1053,1245-1279` | 用 `dst.element_type.lanes`、`allowed_dtypes` 参数，已 dtype-agnostic |
| C++ 资源守卫 `estimateCandidateResource` | `lib/PTO/Transforms/SelectTemplateCandidate.cpp:108-144,150-187` | `getElementBytes` 按位宽 dispatch（f32/i32=4、f16/bf16/i16=2、i8=1、f64/i64=8），按 256 取整，预算按字节算不按 dtype |
| 融合跨迭代 UB 守卫 `getElementBytes` | `lib/PTO/Transforms/PTOVmiLoopFusion.cpp:208-222`、`lib/PTO/Transforms/VmiMemoryLocation.cpp:56-75` | 按整数位宽 dispatch，不区分 signed/unsigned（正确，因位宽共享） |
| load/store elision `getVRegLaneCount` / `LaneRange` | `lib/PTO/Transforms/PTOVmiLoadStoreElision.cpp:468-492` | 按 VREG 实际元素数，不硬编码 64 |
| mask 粒度 per-element + `mask_bits` 校验 | `lib/TileOps/a5/_vmi_common.py:197-204` | mask 前缀 `[0,N)` 以元素计，`mask_bits` 与 dtype 匹配才放行 |
| 已 int-correct 的 reduction 中性元（C++） | `lib/PTO/Transforms/VMILowerUnifiedToLegacy.cpp:148-188` | `createReduceNeutralInit`：int max→`INT_MIN`、min→`INT_MAX`、add→0、unsigned 各自正确 |

### 既有整数 VMI 先例（仅一处）

真正在 VMI 层支持整数元素类型的候选**只有 `vmi_tcvt`**：
`tcvt.py:1861` 的 `vmi_tcvt` + `convert_vmi_constraint`（`_vmi_common.py:851-905`）。
它建立的扩展模式：
1. **per-pair dtype allowlist**（`_vmi_common.py:866-877`）：如 `("f32","i32"): {"TRUNC"}`
   只允许截断，`("i32","f16"): {"ROUND"}` 只允许舍入——按 dtype 对做语义门控。
2. **dtype-aware 字节算**（`:885-892`）：`src_bytewidth = _DTYPE_BYTEWIDTH.get(src_dtype)`，
   `max(cols*src_bytewidth, cols*dst_bytewidth) >= 128`——不硬编码 f32 字节。
3. **混合宽度 chunk**（`emit_convert_vmi` `:2511`）：`chunk_lanes = min(src.element_type.lanes,
   dst.element_type.lanes)`。

> **纠正一个常见误解**：`tmov2bias`/`tmov2left`/`tmov2right`/`tmov2vec`/`tdequant` 虽然在
> dtypes 列表里含 `i32/i8`，但它们是**普通 MTE/vector 模板**（用 `pto.mte_l1_bt`/
> `pto.vlds`/`pto.vsts`），**不是** `@canonical_vmi_template` 装饰的 VMI 候选，不构成
> 整数 VMI 先例。`vmi_tmov`（`tmov.py:99-112`）反而**显式排除整数**（`dtypes` 限
> `f32/f16/bf16`，helper `_vmi_tmov_shape_supported` 硬编码这三种 dtype 的 lanes）——
> 本 ADR 正是要反转这一历史限制。

## Requirements

1. VMI elementwise 候选的 dtype 覆盖对齐普通 PTODSL 的 `NUMERIC_DTYPES`，使相同 shape
   下不再因 dtype 不被 VMI 支持而回退。
2. 候选准入门槛以 "`cols * bytewidth` 凑够 VL" 为准（不再硬编码 f32 字节），门槛与
   dtype 无关。
3. 整数语义必须明确（溢出 wrap vs saturate）；未验完的 dtype/语义组合保守回退，不猜测
   安全。
4. 资源守卫、融合合法性、mask 粒度对全 dtype 正确（已验证基础设施就绪，本 ADR 固化此
   结论，不在实现 PR 中再质疑）。
5. 正确性门槛：新 dtype 跑现有 VMI lit（参数化 dtype）+ `fa-softmax` sim compare PASS。

## Decision

### D1：分两批 dtype 推进（降低风险）

**批 1 — 浮点全 dtype**（`f16`/`bf16` + 既有 `f32`）：elementwise family。
语义风险低——同属浮点，NaN/Inf 传播规则同族，mask/lane 已就绪。改动：
- 各 elementwise op 的 `dtypes=(("f32","f32","f32"),)` →
  `(("f32","f32","f32"),("f16","f16","f16"),("bf16","bf16","bf16"))`（二元），
  一元同理。
- `emit_elementwise_vmi` 的 `allowed_dtypes` 默认从 `(f32,)`（`_vmi_common.py:1030`）
  改为 `(f32,f16,bf16)`，或各 op 显式传 `FLOAT_DTYPES`（`_vmi_common.py:49` 定义
  `(f32,f16)`——需补 `bf16`）。

**批 2 — 整数全 dtype**（`i8/i16/i32/ui8/ui16/ui32`）：elementwise family。
语义风险中——溢出 wrap 默认、需补 `ui8` 的 `ScalarType`/`_pto_dtype`、reduction 中性元
Python 路径要修（见 D3）。本批**先只做 elementwise（非 reduction）**。

### D2：修硬编码 f32 点（全批共用）

以下 `_vmi_common.py` 行号实测硬编码 `f32.lanes` 或 `_DTYPE_BYTEWIDTH["f32"]`，必须改为
`dst.element_type.lanes` / `_DTYPE_BYTEWIDTH[dtype_str]`，对齐既有正确模式
（`_vmi_common.py:1053`、`:1219`、`:2511`）：

| 行号 | 代码 | 修正目标 |
|---|---|---|
| `:133` | `cols * _DTYPE_BYTEWIDTH["f32"] >= 128`（`row_reduce_vmi_constraint`） | `_DTYPE_BYTEWIDTH[src_dtype]` |
| `:167` | `rows * cols * f32.bytewidth > 256`（`row_reduce_streaming_vmi_constraint`） | `src_dtype.bytewidth` |
| `:681` | `dtype == "f32"`（`sinkhorn_compact_elementwise_vmi_constraint`） | 扩成允许 int 或显式按 dtype 分支 |
| `:745, :748` | `_is_safe_static_row_prefix(..., native_lanes=f32.lanes)`（`row_expand_binary_vmi_constraint`） | `dst_dtype.lanes` |
| `:754` | `logical_cols * _DTYPE_BYTEWIDTH["f32"] >= 128` | `_DTYPE_BYTEWIDTH[src_dtype]` |
| `:838` | `cols * _DTYPE_BYTEWIDTH["f32"] >= 128`（`col_expand_vmi_constraint`） | `_DTYPE_BYTEWIDTH[src_dtype]` |
| `:1919` | `safe_read_cols = ((valid_cols + f32.lanes - 1) // f32.lanes) * f32.lanes`（`_validate_row_reduce_tiles`） | `src.element_type.lanes` |
| `:1982` | `if physical_cols < f32.lanes:`（`emit_row_reduce_vmi`） | `src.element_type.lanes` |
| `:2097, :2102` | `_is_safe_static_row_prefix(..., native_lanes=f32.lanes)`（`emit_row_expand_binary_vmi`） | `dst_dtype.lanes` |
| `:2117` | `io_lanes = ((cols + f32.lanes - 1) // f32.lanes) * f32.lanes` | row dtype 的 lanes |

per-op constraint 的 `dtype == "f32"` 字符串门（如 `row_reduce_vmi_constraint`
`:103-105` 要求 `src/workspace/dst dtype == "f32"`）扩成允许目标 dtype 集。

> 注：`_vmi_common.py:1055` 的 `if dst.element_type == f32 and ((rows,cols), valid_shape) in {...}`
> 是**有意的 f32-only Sinkhorn 快速路径**，不是 bug——i8/i16 的 Sinkhorn 形会落到通用路径，
> 可接受。

### D3：整数特化处理

**溢出语义**：明确 VMI `vadd/vmul` 整数为 **wrap**（非 saturate），与 A5 `vadd/vmul`
默认语义一致。compute closure（`_vmi_common.py:1282-1321` 的 `_add/_mul/_max/_min`）调
`_vadd/_vmul`（`:329-349`），无 `sat_mode`。若后续需 saturating int add/mul，新增
`sat_mode` context attr 到 vadd/vmul lowering（当前仅 `vcvt` 有 saturation，
`VMILowerUnifiedToLegacy.cpp:431,439,445,451,472`）——列为 out-of-scope 后续。

**`ui8` 缺失**：`_tile_template_tracing.py:80-85` 定义了 f32/f16/bf16/i32/i16/i8 的
`ScalarType`，但**没有 `ui8`**；`_vmi_common.py:50-51` 只补了 `ui16/ui32`。`_DTYPE_BYTEWIDTH`
（`:604-614`）含 `"ui8": 1`，但 `_pto_dtype`（`:58-72`）无 `"ui8"` 入口。批 2 前需补：
- `_tile_template_tracing.py`（或 `_vmi_common.py` 本地）加
  `ui8 = ScalarType("ui8", lanes=256, mask_bits=8, bytewidth=1)`。
- `_pto_dtype` 加 `"ui8"` → `pto.i8`（或对应 unsigned 类型）入口。

**reduction 中性元**：C++ `createReduceNeutralInit`（`VMILowerUnifiedToLegacy.cpp:148-188`）
已 int-correct。但 Python `emit_col_reduce_vmi`/`emit_row_reduce_vmi` 硬编码
`reduce_identity = {"max": float("-inf"), ...}` + `_vconstant(..., f32, ...)`（
`_vmi_common.py:2300-2305,2317,2343`），对 int reduction 会把 `float("-inf")` 喂给
int literal materializer，出错。**elementwise 批（D1 批 1/批 2）不涉及 reduction**，
但本 ADR 明确标注：若后续扩 reduction 到 int，必须把 Python reduce identity 按 dtype
映射（`-inf→INT_MIN`、`inf→INT_MAX`、`0.0→0`），列为 reduction 扩展（PR3）的前置依赖。
同时 `_vmi_common.py:1964,2129` 的 mask 包裹也用了 `f32`，需一并修。

**先例采纳**：`convert_vmi_constraint` 的 per-dtype allowlist + `_DTYPE_BYTEWIDTH` 字节算
作为 elementwise 扩展的结构模板（elementwise 因 src.dtype==dst.dtype 而简化为单 dtype
allowlist，非 per-pair）。

### D4：资源守卫无需改

确认 `estimateCandidateResource`（`SelectTemplateCandidate.cpp:150-187`）已按
`getElementBytes`（`:108-118` 按 bit-width dispatch）+ 256 取整（`:120-126`），
dtype 扩展**不破坏** 6144B 预算判定（默认 `maxCandidateVectorBytes=6144`，即 24 个物理
向量）。i8 一个 VREG 256 lanes * 1 byte = 256 字节，与 f32 的 64*4=256 一致，预算计算
等价。

## 分 PR 交付（建议）

- **PR1（批 1 浮点）**：elementwise `f16/bf16` 扩 + 修硬编码 f32 点（D2）+ 现有 lit
  参数化 dtype + `fa-softmax` bf16 变体 compare PASS。
- **PR2（批 2 整数 elementwise）**：补 `ui8` `ScalarType`/`_pto_dtype`（D3）+
  elementwise `i8/i16/i32/ui8/ui16/ui32` + 溢出 wrap 语义文档化 + lit。
- **PR3（可选，reduction 扩展）**：修 Python reduce identity int 映射（D3）+
  `tcolmax/tcolsum/trowmax/trowmin/trowsum` 扩 int + `createReduceNeutralInit` 已就绪
  验证。

## 验收门槛

用户已定：**现有 lit + fa-softmax compare PASS**。

- 现有 `test/lit/vpto/vmi_*` + `ptodsl_vmi_*` 测试在参数化 dtype（新增 `f16/bf16/i8/i16/
  i32` 期望行）下全 PASS。
- `fa-softmax-dn-init-rowplusone` case 用 `bf16` 跑 sim，`compare passed`（参照
  `test/vpto/cases/vmi/fa-softmax-dn-init-rowplusone/README.md` §3 的两路 sim 流程）。
- 回归：现有 `f32` 路径不回归（VMI 路 rvec_busy/ticks、hazard 计数、compare PASS 与
  扩展前一致）。

## Alternatives considered

- **A. 固定 24-chunk 阈值拒绝**：拒。ADR-0002 已否，资源按字节算不按固定 chunk 数；
  且 i8 与 f32 一个 VREG 都是 256 字节，固定 chunk 数对 i8 不公允。
- **B. 永久普通回退（只 f32 走 VMI）**：拒。违背本 ADR 目标，失去 `bf16` 融合收益
  （A5 大模型主力 dtype）。
- **C. 运行时 chunk 循环在 TileLib candidate 内**：拒。ADR-0001 已否，破坏 logical-row
  协议与 canonical 模板契约。
- **D. 只扩 `bf16/f16` 不扩整数**：部分采纳为批 1，整数作为批 2 后续（降低首版风险）。
  本 ADR 的最终目标是全 `NUMERIC_DTYPES`，但允许分批落地。

## Open questions（不阻塞本 ADR）

- saturating int `add/mul` 是否需要？若框架语义要求（如某些量化路径），加 `sat_mode`
  context attr 到 vadd/vmul lowering（out-of-scope）。
- reduction 扩 int 的 Python identity 修复时序（PR3）——是否与 elementwise 整数批（PR2）
  合并，取决于实际 workload 中 int reduction 的频率。
- `bf16` 的 `ScalarType` 未在 `_tile_template_tracing.py:80-85` 定义（只有 f32/f16/i32/
  i16/i8），需确认 `bf16` 的 `ScalarType` 来源与 `lanes=128`/`bytewidth=2`/`mask_bits=16`
  一致（`_vmi_common.py` 已用 `bf16`，说明已有定义，实现 PR 时核对位置）。
