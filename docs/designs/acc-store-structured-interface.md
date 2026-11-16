# `acc_store` 统一接口设计

## 1. 目标方案

`acc_store` 的目标接口只保留 `target_profile` 可用的 `L0C -> OUT` 结构化语义：

```mlir
pto.mte_l0c_l1 %src, %dst, %m, %n, %src_stride, %dst_stride,
              unit_flag(check_only | check_and_clear)?,
              pre_quant(%scalar_or_fb_addr, mode = ...)?,
              pre_relu(%alpha_or_fb_addr, mode = ..., clip = %clip_value ?)?,
              nz2nd? | nz2dn(%loop0_src_stride)? | nz2nz(%split)?,
              loop3(%count, %src_stride, %dst_stride)?,
              (sat | nosat)?
```

其中：

- `%src` 必须是 `!pto.ptr<..., l0c>`
- `%dst` 允许是 `!pto.ptr<..., gm>` `!pto.ptr<..., vec>`或 `!pto.ptr<..., l1>`
- 这些扩展字段全部是可选的，未出现就表示不启用
- `nz2nd / nz2dn / nz2nz` 和 `loop3` 是并列的 layout 相关参数

## 2. 字段形态

这些是这版结构化接口里各字段的可选/必填关系：

- `unit_flag(check_only | check_and_clear)?`
  - 不写表示 `off`
  - `check_only` 对应先检查后不清零
  - `check_and_clear` 对应先检查后清零
  - `2'b01` 是 ISA 保留值，不进入合法 keyword
- `pre_quant(..., mode = ...)`
  - `mode` 必填
  - `mode` 可选值：`no_convert`、`f32_f16`、`qf322hif8_pre_vec`、`qf322hif8_pre_scalar`、`qf322hif8_pre_hybrid_vec`、`qf322hif8_pre_hybrid_scalar`、`deqs32_int_vec`、`deqs32_int_scalar`、`req8_vec`、`req8_scalar`、`deqf16_vec`、`deqf16_scalar`、`qf322fp8_pre_vec`、`qf322fp8_pre_scalar`、`qf322f32_pre_vec`、`qf322f32_pre_scalar`、`f32_bf16`、`qf162b8_pre_vec`、`qf162b8_pre_scalar`、`qf162s4_pre_vec`、`qf162s4_pre_scalar`、`req4_vec`、`req4_scalar`、`qf322b8_pre_vec`、`qf322b8_pre_scalar`、`qf322s4_pre_vec`、`qf322s4_pre_scalar`、`deqs16_vec`、`deqs16_scalar`、`qf162s16_pre_vec`、`qf162s16_pre_scalar`、`qf322f16_pre_vec`、`qf322f16_pre_scalar`、`qf322bf16_pre_vec`、`qf322bf16_pre_scalar`、`qs322bf16_pre_vec`、`qs322bf16_pre_scalar`
  - `%scalar_or_fb_addr` 由 `mode` 决定解释方式
  - scalar 类模式下，`%scalar_or_fb_addr` 是量化参数值，允许直接传 `f16`、`bf16`、`f32`
  - `f16`/`bf16` scalar payload 会先扩成 `f32`，再按 `SPR.QUANT_PRE` 需要的 32-bit 浮点 bit pattern 编码
  - `f32` scalar payload 直接按 32-bit 浮点 bit pattern 编码到 `SPR.QUANT_PRE`
  - vector 类模式下，`%scalar_or_fb_addr` 是 FB1 地址，映射到 `SPR.FPC[15:8] / Quant_PRE_ADDR`
  - `mode` 还必须与 `acc_store*` 的源/目的元素类型匹配；例如 `f32 -> f16` 应选 `qf322f16_pre_vec/scalar`，`req8_vec/scalar` 只适用于 `i32 -> i8/u8`
  - 无额外可选子参数
- `pre_relu(%alpha_or_fb_addr, mode = ..., clip = %clip_value)?`
  - `mode` 必填
  - `mode` 可选值：`no_relu`、`normal_relu`、`scalar_relu`、`vector_relu`、`pwl`
  - payload 不是对所有 mode 都必填：
  - `mode = no_relu` 或 `mode = normal_relu` 时，不带 payload
  - `mode = scalar_relu` 时，必须带 `%alpha_or_fb_addr`，允许直接传 `f16`、`bf16`、`f32`
  - `f16`/`bf16` scalar alpha 会先扩成 `f32`，再按 `SPR.RELU_ALPHA` 需要的 32-bit 浮点 bit pattern 编码
  - `f32` scalar alpha 直接按 32-bit 浮点 bit pattern 编码到 `SPR.RELU_ALPHA`
  - `mode = vector_relu` 时，必须带 `%alpha_or_fb_addr`，其值作为 FB1 地址，映射到 `SPR.FPC[7:0] / RELU_PRE_ADDR`
  - `clip = %clip_value` 为可选子句：表示启用 pre-stage clip，并把 `%clip_value` 映射到 `SPR.FIX_CLIP_RELU`
  - `clip` 只允许用于手册明确覆盖的目标类型：`f16`、`ui8`、`s4/s8/s16`
- `nz2nd?`
  - 无额外参数
- `nz2dn(%loop0_src_stride)?`
  - `loop0_src_stride` 必填
- `nz2nz(%split)?`
  - `split` 可选
  - 不写 `split` 表示不做 F32 channel split
- `loop3(%count, %src_stride, %dst_stride)?`
  - 这三个参数都必填
  - 无额外可选子参数
- `(sat | nosat)?`
  - 可选 flag
  - 不写表示不显式配置饱和控制，沿用进入 op 前的状态
  - 写 `sat` 表示本 op 内选择饱和行为
  - 写 `nosat` 表示本 op 内选择非饱和行为
  - `sat` 和 `nosat` 互斥
- `atomic(type = ..., op = ...)?`
  - 仅 `acc_store_gm` 支持
  - `type` 必填，可选值：`f32`、`f16`、`s16`、`s32`、`s8`、`bf16`
  - `op` 必填，可选值：`add`、`max`、`min`
  - 不写 `atomic(...)` 表示普通覆盖写回，不启用 OUT atomic read-modify-write

## 3. 约束

`target_profile` 下，这版接口保留的有效项是：

- `pre_quant(...)`
- `pre_relu(..., clip = %clip_value)?`
- `unit_flag(...)`
- `nz2nd`
- `nz2dn(%loop0_src_stride)`
- `nz2nz(%split)?`
- `loop3(...)`
- `sat` / `nosat`
- `atomic(...)`（仅 `acc_store_gm`）

其中：

- `pre_quant` 的 scalar 类模式走 `SPR.QUANT_PRE`
- `pre_quant` 的 vector 类模式走 `SPR.FPC[15:8] / Quant_PRE_ADDR`，对应 FB1 mem_block0
- `pre_relu(%alpha_or_fb_addr, mode = scalar_relu)` 走 `SPR.RELU_ALPHA[31:13]`，不走 FB1 地址
- `pre_relu(%alpha_or_fb_addr, mode = vector_relu)` 走 FB1 mem_block1，并通过 `SPR.FPC[7:0]` 选择 `RELU_PRE_ADDR`
- `pre_relu(..., clip = %clip_value)` 的 `clip` 子句走 `SPR.FIX_CLIP_RELU`
- `unit_flag` 不走 FB1
- `split` 不走 FB1
- `sat` / `nosat` 走 `SPR.CTRL`
- `atomic` 仅在 `acc_store_gm` 上走 `SPR.CTRL`
- `post-stage`、`element-wise`、`LoopEnhance` 相关扩展不纳入本版接口

注意：`NZ2DN` 和 `unit_flag` 不是无条件兼容的。`loop0_src_stride != 1` 时，`unit_flag` 必须关闭。

`target_profile` 下不是禁止 `NZ2ND / NZ2DN` 的参数。相反，`FIX_L0C_TO_OUT.f32/s32` 明确标了 `NZ2ND Mode` 和 `NZ2DN Mode` valid；其中 `nz2dn(%loop0_src_stride)` 仍然需要把 `loop0_src_stride` 写入 `CHANNEL_PARA[63:48]`，单位是 `C0_SIZE`。

**`nz2nz` 且未启用 `%split` 就是硬件 native normal-mode 写回**（`NZ2ND = NZ2DN = SPLIT_EN = 0`，对应 AscendNPU-IR 的 `fixpipe_normal_*` 模板，例如 `fixpipe_normal_float_to_bfloat16_t_4d_to_4d_cbuf`）。不写 layout 子句**不是**这个模式：lowering 会置 `nz2nd`（`X_t[43] = 1`），不能当作 normal-mode 写回使用。

`nz2nz` 的目的元素类型限制只属于 channel split（`SPLIT_EN`）。`SPLIT_EN = 1` 且输出类型不是 `f32` 时是非法配置；`%split` 不写或为常量 0（`SPLIT_EN = 0`）时，`nz2nz` 只清掉 `NZ2ND` / `NZ2DN` 两个 layout 转换位，就是硬件 native normal-mode 写回，目的元素类型只由 `pre_quant` / `pre_relu` 决定（例如 `f32` 累加器经 `pre_quant(mode = f32_bf16)` 写成 `bf16` L1 块状 tile）。这里的“常量 0”包括 `arith.constant` 和 DSL 实际生成的 `pto.constant` 两种形式；非常量 `%split` 以及低 bit 为 1 的常量仍然要求 `f32` 目的类型。

注意：`%dst_stride` 的单位随 layout 变化。`nz2nd` / `nz2dn` 用目的元素数，normal-mode `nz2nz` 用 `C0_SIZE`（32B）单位（FIXPIPE normal-mode 的 `Loop1_dst_stride` 定义）。操作数原样下发给指令，所以把同一个 store 在 `nz2nd` 和 `nz2nz` 之间切换而不重新缩放 `%dst_stride`，会静默改变含义。

已知未校验项：未写 `pre_quant` 时是 `no_convert`（`quant_pre = 0`），硬件要求源/目的同为 32-bit 同类型（`f32`→`f32`、`i32`→`i32`）。本接口目前不校验这一耦合（`nz2nd` 下同样存在，属既有行为），改 dtype 的 store 必须显式给出 `pre_quant` / `pre_relu`。

`loop3(...)` 不是 `nz2dn` 或 `nz2nd` 的别名，它是单独的参数组，只在 `nz2nd` 或 `nz2dn` 场景下使用。

## 4. 映射

- `pre_quant(%scalar_or_fb_addr, mode = ...)` 映射到 `SPR.QUANT_PRE` 或 `SPR.FPC[15:8] / Quant_PRE_ADDR`
- `pre_relu(%alpha_or_fb_addr, mode = ...)` 映射到 `X_t[41:39] / ReLU_PRE`，并按模式进一步映射到 `SPR.RELU_ALPHA[31:13]` 或 `SPR.FPC[7:0] / RELU_PRE_ADDR`
- `pre_relu(..., clip = %clip_value)?` 映射到 `X_t[31:30] / Clip_ReLU_PRE`（使能）以及 `SPR.FIX_CLIP_RELU[15:0]`
- `unit_flag(check_only | check_and_clear)?` 映射到 `X_t[33:32] / unit_flag`
- `nz2nz(%split)?` 映射到 `X_t[42] / SPLIT_EN`；`SPLIT_EN = 0` 且无其他 layout 位时即 `X_t[43] NZ2ND = 0` / `X_t[62] NZ2DN = 0`，对应硬件 native normal-mode 写回
- `nz2dn(%loop0_src_stride)` 映射到 `CHANNEL_PARA[63:48]`
- `loop3(...)` 映射到 `SPR.LOOP3_PARA`
- `sat` / `nosat` 映射到 `SPR.CTRL[48] / ctrl_sat_ctrl`
- `atomic(type = ..., op = ...)?` 仅对 `acc_store_gm` 有效，映射到 `SPR.CTRL[8:6] / ctrl_atomic_en` 和 `SPR.CTRL[10:9] / ctrl_atomic_op`

## 5. Keyword

当前结构化接口使用语义 keyword，不直接暴露 bit 编码：

- `pre_relu.mode`
  - `no_relu` -> `3'b000`
  - `normal_relu` -> `3'b001`
  - `scalar_relu` -> `3'b010`
  - `vector_relu` -> `3'b011`
  - `pwl` -> `3'b100`
- `pre_quant.mode`
  - `no_convert` -> `6'b000000`
  - `f32_f16` -> `6'b000001`
  - `qf322hif8_pre_vec` -> `6'b000010`
  - `qf322hif8_pre_scalar` -> `6'b000011`
  - `qf322hif8_pre_hybrid_vec` -> `6'b000100`
  - `qf322hif8_pre_hybrid_scalar` -> `6'b000101`
  - `deqs32_int_vec` -> `6'b000110`
  - `deqs32_int_scalar` -> `6'b000111`
  - `req8_vec` -> `6'b001000`
  - `req8_scalar` -> `6'b001001`
  - `deqf16_vec` -> `6'b001010`
  - `deqf16_scalar` -> `6'b001011`
  - `qf322fp8_pre_vec` -> `6'b001100`
  - `qf322fp8_pre_scalar` -> `6'b001101`
  - `qf322f32_pre_vec` -> `6'b001110`
  - `qf322f32_pre_scalar` -> `6'b001111`
  - `f32_bf16` -> `6'b010000`
  - `qf162b8_pre_vec` -> `6'b010001`
  - `qf162b8_pre_scalar` -> `6'b010010`
  - `qf162s4_pre_vec` -> `6'b010011`
  - `qf162s4_pre_scalar` -> `6'b010100`
  - `req4_vec` -> `6'b010101`
  - `req4_scalar` -> `6'b010110`
  - `qf322b8_pre_vec` -> `6'b010111`
  - `qf322b8_pre_scalar` -> `6'b011000`
  - `qf322s4_pre_vec` -> `6'b011001`
  - `qf322s4_pre_scalar` -> `6'b011010`
  - `deqs16_vec` -> `6'b011011`
  - `deqs16_scalar` -> `6'b011100`
  - `qf162s16_pre_vec` -> `6'b011101`
  - `qf162s16_pre_scalar` -> `6'b011110`
  - `qf322f16_pre_vec` -> `6'b011111`
  - `qf322f16_pre_scalar` -> `6'b100000`
  - `qf322bf16_pre_vec` -> `6'b100001`
  - `qf322bf16_pre_scalar` -> `6'b100010`
  - `qs322bf16_pre_vec` -> `6'b100011`
  - `qs322bf16_pre_scalar` -> `6'b100100`
- `pre_quant.scalar`
  - the specific scalar payload is mode-dependent and lives in `SPR.QUANT_PRE`
- `pre_quant.fb_addr`
  - the specific parameter array address is mode-dependent and lives in `SPR.FPC[15:8]`
- `pre_relu.clip`（是否出现 `clip = %clip_value` 子句）
  - 未出现 -> `2'b00`
  - 出现 -> `2'b01`
- `unit_flag`
  - absent -> `2'b00`
  - `check_only` -> `2'b10`
  - `check_and_clear` -> `2'b11`
- `atomic.type`
  - `f32` -> `3'b001`
  - `f16` -> `3'b010`
  - `s16` -> `3'b011`
  - `s32` -> `3'b100`
  - `s8` -> `3'b101`
  - `bf16` -> `3'b110`
- `atomic.op`
  - `add` -> `2'b00`
  - `max` -> `2'b01`
  - `min` -> `2'b10`
- `sat` / `nosat`
  - absent -> no explicit `CTRL[48]` override
  - `sat` -> `CTRL[48] = 1'b0`
  - `nosat` -> `CTRL[48] = 1'b1`

## 6. 说明

这份文档只描述目标方案，不保留旧扁平接口的过渡写法，也不展开 `profile1` 的后处理、element-wise 和 LoopEnhance 字段列表。

这里不再引入 `fixpipe(...)` 大包；这些项直接作为 `acc_store` 的结构化语义字段出现，避免把 source access、layout transform 和 writeback control 都误解成同一个固定 pipeline stage。
