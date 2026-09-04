# A5 LOAD3D / TIMG2COL 接口与 LLVM intrinsic 参考

本文记录安装版 CANN 9.1.0 与 camodel 的实测结果；附件 `disa-cube.json` 仅作 ISA 对照。

## 已确认的 LLVM intrinsic

实际从 A5 CCE kernel 导出的 LLVM IR 中观察到：

```llvm
declare void @llvm.hivm.LOAD.L1.TO.L0A.3DV2.c310.f16.C310.NO.DUAL(
  ptr addrspace(3) nocapture writeonly,
  ptr addrspace(2) nocapture readonly, i64, i64) #7
```

同一家族还生成 `f32`、`s8` 类型后缀。配置展开形式为：

```llvm
declare void @llvm.hivm.LOAD.L1.TO.L0A.3DV2.c310.f16.C310.NO.DUAL.cfg(
  ptr addrspace(3) nocapture writeonly,
  ptr addrspace(2) nocapture readonly,
  i64, i64, i64, i64, i64, i64, i64, i64,
  i64, i64, i64, i64, i64, i64, i64) #7
```

`addrspace(2)` 是 L1/CBUF 源，`addrspace(3)` 是 L0A 目的；最后两个 `i64` 是 wrapper 打包后的控制字。`.cfg` 的 15 个标量参数当前没有被 `TIMG2COL` wrapper 调用，不能据声明猜测逐字段顺序。

已确认的控制 intrinsic：

```llvm
declare void @llvm.hivm.SET.L3D.RPT(i64) #6
declare void @llvm.hivm.SET.L3D.RPT.B(i64) #6
declare void @llvm.hivm.SET.FMATRIX(i64) #6
declare void @llvm.hivm.SET.FMATRIX.B(i64) #6
declare void @llvm.hivm.SET.PADDING(i64) #6
declare void @llvm.hivm.SET.PADDING.B(i64) #6
```

## 入口和语义

## 建议的 PTOAS 接口形态

该设计明确参考现有二维 MTE 接口的两层结构，而不是把硬件 packed
参数直接暴露给 DSL：

```text
pto.mte_l1_l0a_3d       (DSL / wrapper 层)
        |
        v
pto.load_cbuf_to_ca_3d  (具体 lowering 层)
        |
        v
SET.FMATRIX / SET.PADDING / SET.L3D.RPT
        |
        v
llvm.hivm.LOAD.L1.TO.L0A.3DV2...
```

二维接口已经采用同样的分层：`mte_l1_l0a` 根据逻辑矩阵参数推导控制值，
再展开为 `load_cbuf_to_ca`，最后由 LLVM emitter 打包为 intrinsic。3D
接口应保持这个边界。

### Wrapper 层

建议新增 `pto.mte_l1_l0a_3d`，操作数仍然只有 source 和 destination，
几何与 repeat 参数使用有名字的 attributes：

```text
source, destination {
  fmap_h, fmap_w,
  filter_h, filter_w,
  stride_h, stride_w,
  dilation_h, dilation_w,
  pad_top, pad_bottom, pad_left, pad_right,
  channel_size,
  repeat_time, repeat_mode, repeat_source_stride,
  dst_stride, dst_m_position,
  transpose
}
```

该层负责 layout、地址空间、范围和 target availability 检查，并根据
`fmap/filter/stride/dilation` 推导 M/K 结果范围。

### Concrete 层

建议新增 `pto.load_cbuf_to_ca_3d`，沿用 `load_cbuf_to_ca` 的显式控制
operand 风格，但增加 3D 必需参数：

```text
source, destination,
m_start, k_start, m_step, k_step,
stride_w, stride_h, filter_w, filter_h,
dilation_w, dilation_h, channel_size,
repeat_source_stride, repeat_time, repeat_mode,
dst_stride, dst_m_position
{ padding = ..., transpose = ..., fmatrix_mode = ... }
```

Concrete 层不再做高层 shape 推导；LLVM emitter 负责把这些值编码成
`SET.FMATRIX`、`SET.PADDING`、`SET.L3D.RPT` 和最终的两个 packed `i64`。
packed `i64` 不应出现在 DSL 或稳定的 PTO IR 接口中。

### L0B 扩展

当前只建议先实现 L0A。确认 A5 的 L0B CCE wrapper 和 LLVM intrinsic
之后，再平行增加 `pto.mte_l1_l0b_3d` / `pto.load_cbuf_to_cb_3d`；不要
通过 `destination_kind` 把两个有不同布局约束的操作合并成一个接口。

[TImg2col.hpp](/usr/local/CANN/cann-9.1.0/x86_64-linux/include/pto/npu/a5/TImg2col.hpp) 的 A5 实现调用 `img2colv2_cbuf_to_ca(dstAddr, srcAddr, stepK, stepM, posK, posM, strideW, strideH, lowFilterW, lowFilterH, dilationW, dilationH, highFilterW, highFilterH, transpose, fmatrixCtrl, channelSize)`。

当前已验证入口是 `TIMG2COL`/`img2colv2_cbuf_to_ca`，即把 L1 的 `NC1HWC0` image 展开为 cube 的 L0A 矩阵。`strideW/H`、`filterW/H`、`dilationW/H`、`channelSize`、`posM/posK`、`stepM/stepK` 决定滑窗和输出范围；越界位置使用 padding 值，`transpose` 改变目的矩阵组织。

`SetRepeat` 的打包布局（[SetImg2colRpt.hpp](/usr/local/CANN/cann-9.1.0/x86_64-linux/include/pto/npu/a5/SetImg2colRpt.hpp)）：`[15:0] repeatStride`（repeat 间源 image 推进量）、`[23:16] repeatTime`、`[31:24] repeatMode`、`[47:32] dstStride`、`[63:48] dstMposition`。`repeatStride` 只影响启用 repeat 时相邻重复窗口的源图像推进，不构成另一种独立单次寻址模式。

## L0B、旧指令和验证边界

附件列出的旧 `LOAD_L1_TO_L0A_3D`、`LOAD_L1_TO_L0B_3D`、`LOAD_L1_TO_UB_3D` 以及 `LOAD_L1_TO_UB_3Dv2` 标记为 unavailable。本机 A5 `TImg2col.hpp` 固定走 A 侧；本次 module 未观察到 `llvm.hivm.LOAD.L1.TO.L0B.3DV2`，因此不能把 L0B 3Dv2 当成本机已公开的 CCE 入口。

使用 PTO-ISA `tests/npu/a5/src/st/testcase/timg2col` 运行 `case1_bfloat16`、`case2_float16`，两项均通过；camodel 指令日志实际出现 `LOAD_L1_TO_DST_3DV2`。验证链路为：`TIMG2COL` -> CCE wrapper -> `llvm.hivm.LOAD.L1.TO.L0A.3DV2...` -> A5 指令 -> camodel。
