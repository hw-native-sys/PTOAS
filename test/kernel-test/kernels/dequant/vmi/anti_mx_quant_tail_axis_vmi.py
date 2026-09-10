# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""VMI rewrite of AntiMxQuantTailAxis scale/dequant helpers.

Source regions:
  test/kernel-test/kernels/dequant/anti_mx_quant_tail_axis.h:313
  test/kernel-test/kernels/dequant/anti_mx_quant_tail_axis.h:380

Current modeling notes:
  1. ``ComputeScale`` is expressed in terms of the logical e8m0 scale value
     rather than replaying the source's temporary zero-fill + interleave
     choreography. For the fp8 path we materialize `scaleBuffer` as one
     contiguous 16-lane compact FP32 scale vector per loop: the 16 raw
     block-scale bytes are converted to FP32 and kept compact in UB.
  2. The source does ``DIST_DINTLV_B8`` and then immediately interleaves the
     two FP8 registers before FP8-to-FP32 part casts. Logically that is
     equivalent to two contiguous 256-lane FP8 halves, so the VMI rewrite
     models this as two dense loads followed by two `vcvt`s.
  3. ``scaleBufAddr`` therefore holds one contiguous 16-lane compact scale
     vector per loop, and ``ComputeData`` reloads each 8-scale half with a
     grouped ``vload`` before grouped ``vbrc`` expansion and multiply.
  4. The runtime `dequant_vmi_*` kernels now call `ComputeScale` and
     `ComputeData` explicitly, matching the source `Compute(...)` structure.
"""

from ptodsl import pto

_ELEMS_PER_LOOP = 512
_HALF_ELEMS_PER_LOOP = _ELEMS_PER_LOOP // 2
_BLOCK_SIZE = 32
_BLOCKS_PER_DATA_LOOP = _ELEMS_PER_LOOP // _BLOCK_SIZE
_SCALE_LANES_PER_LOOP = _BLOCKS_PER_DATA_LOOP
_COMPACT_SCALE_LANES_PER_LOOP = _SCALE_LANES_PER_LOOP
_SCALE_LANES_PER_HALF = _SCALE_LANES_PER_LOOP // 2
_SCALE_REPEAT_GROUPS = _SCALE_LANES_PER_HALF
_RAW_SCALE_BYTES_PER_LOOP = _BLOCKS_PER_DATA_LOOP

_X_BASE_ADDR = 0
_RAW_SCALE_BASE_ADDR = 0x4000
_SCALE_BASE_ADDR = 0x8000
_Y_BASE_ADDR = 0x10000


def _compute_scale_ub(
    scale_local_addr: pto.i64,
    scale_buf_addr: pto.i64,
    *,
    DST_FMT: pto.const_expr = "f32",
    LOOP_NUM2VF: pto.const_expr = 1,
):
    scale_src_ptr = pto.castptr(scale_local_addr, pto.ptr(pto.ui8, "ub"))

    if pto.const_expr(DST_FMT == "f32"):
        scale_dst_ptr = pto.castptr(scale_buf_addr, pto.ptr(pto.f32, "ub"))
        dst_dtype = pto.f32
    elif pto.const_expr(DST_FMT == "bf16"):
        scale_dst_ptr = pto.castptr(scale_buf_addr, pto.ptr(pto.bf16, "ub"))
        dst_dtype = pto.bf16
    else:
        raise ValueError(f"unsupported ComputeScale DST_FMT specialization: {DST_FMT}")

    shift = pto.vmi.vbrc(pto.i32(23), size=_BLOCKS_PER_DATA_LOOP)
    scale_mask = pto.vmi.create_mask(_SCALE_LANES_PER_LOOP, size=_SCALE_LANES_PER_LOOP)

    for i in range(LOOP_NUM2VF):
        raw_scale_off = i * _RAW_SCALE_BYTES_PER_LOOP
        dst_off = i * _COMPACT_SCALE_LANES_PER_LOOP

        raw_scale = pto.vmi.vload(
            scale_src_ptr,
            raw_scale_off,
            size=_BLOCKS_PER_DATA_LOOP,
        )
        scale_u32 = pto.vmi.vcvt(raw_scale, pto.ui32)
        scale_i32 = pto.vmi.vinterpret_cast(
            scale_u32,
            pto.i32,
        )
        scale_bits = pto.vmi.vshl(scale_i32, shift)
        scale_compact = pto.vmi.vinterpret_cast(
            scale_bits,
            pto.f32,
        )
        if pto.const_expr(DST_FMT == "f32"):
            pto.vmi.vstore(scale_compact, scale_dst_ptr, dst_off, scale_mask)
        else:
            pto.vmi.vstore(pto.vmi.vcvt(scale_compact, dst_dtype), scale_dst_ptr, dst_off, scale_mask)


def _compute_data_ub(
    x_local_addr: pto.i64,
    scale_buf_addr: pto.i64,
    y_local_addr: pto.i64,
    *,
    SRC_FMT: pto.const_expr = "e4m3",
    DST_FMT: pto.const_expr = "f32",
    LOOP_NUM2VF: pto.const_expr = 1,
):
    if pto.const_expr(SRC_FMT == "e4m3"):
        x_ptr = pto.castptr(x_local_addr, pto.ptr(pto.f8e4m3, "ub"))
    elif pto.const_expr(SRC_FMT == "e5m2"):
        x_ptr = pto.castptr(x_local_addr, pto.ptr(pto.f8e5m2, "ub"))
    else:
        raise ValueError(f"unsupported SRC_FMT specialization: {SRC_FMT}")

    scale_ptr = pto.castptr(scale_buf_addr, pto.ptr(pto.f32, "ub"))

    if pto.const_expr(DST_FMT == "f32"):
        y_ptr = pto.castptr(y_local_addr, pto.ptr(pto.f32, "ub"))
        dst_dtype = pto.f32
    elif pto.const_expr(DST_FMT == "bf16"):
        y_ptr = pto.castptr(y_local_addr, pto.ptr(pto.bf16, "ub"))
        dst_dtype = pto.bf16
    elif pto.const_expr(DST_FMT == "f16"):
        y_ptr = pto.castptr(y_local_addr, pto.ptr(pto.f16, "ub"))
        dst_dtype = pto.f16
    else:
        raise ValueError(f"unsupported DST_FMT specialization: {DST_FMT}")

    mask256 = pto.vmi.create_mask(_HALF_ELEMS_PER_LOOP, size=_HALF_ELEMS_PER_LOOP)

    for i in range(LOOP_NUM2VF):
        x_off = i * _ELEMS_PER_LOOP
        scale_off = i * _COMPACT_SCALE_LANES_PER_LOOP
        y_off = i * _ELEMS_PER_LOOP

        x_lo_f8 = pto.vmi.vload(x_ptr, x_off, size=_HALF_ELEMS_PER_LOOP)
        x_hi_f8 = pto.vmi.vload(x_ptr, x_off + _HALF_ELEMS_PER_LOOP, size=_HALF_ELEMS_PER_LOOP)
        x_lo_f32 = pto.vmi.vcvt(x_lo_f8, pto.f32)
        x_hi_f32 = pto.vmi.vcvt(x_hi_f8, pto.f32)

        scale_lo_slots = pto.vmi.vload(
            scale_ptr,
            scale_off,
            size=_SCALE_LANES_PER_HALF,
            stride=1,
            group=_SCALE_REPEAT_GROUPS,
        )
        scale_hi_slots = pto.vmi.vload(
            scale_ptr,
            scale_off + _SCALE_LANES_PER_HALF,
            size=_SCALE_LANES_PER_HALF,
            stride=1,
            group=_SCALE_REPEAT_GROUPS,
        )
        scale_lo_f32 = pto.vmi.vbrc(
            scale_lo_slots,
            size=_HALF_ELEMS_PER_LOOP,
            group=_SCALE_REPEAT_GROUPS,
        )
        scale_hi_f32 = pto.vmi.vbrc(
            scale_hi_slots,
            size=_HALF_ELEMS_PER_LOOP,
            group=_SCALE_REPEAT_GROUPS,
        )
        y_lo_f32 = pto.vmi.vmul(x_lo_f32, scale_lo_f32, mask256)
        y_hi_f32 = pto.vmi.vmul(x_hi_f32, scale_hi_f32, mask256)

        if pto.const_expr(DST_FMT == "f32"):
            pto.vmi.vstore(y_lo_f32, y_ptr, y_off, mask256)
            pto.vmi.vstore(y_hi_f32, y_ptr, y_off + _HALF_ELEMS_PER_LOOP, mask256)
        else:
            pto.vmi.vstore(
                pto.vmi.vcvt(y_lo_f32, dst_dtype),
                y_ptr,
                y_off,
                mask256,
            )
            pto.vmi.vstore(
                pto.vmi.vcvt(y_hi_f32, dst_dtype),
                y_ptr,
                y_off + _HALF_ELEMS_PER_LOOP,
                mask256,
            )


def _runtime_entry(
    x_gm: pto.ptr(pto.ui8, "gm"),
    scale_gm: pto.ptr(pto.ui8, "gm"),
    y_gm,
    *,
    SRC_FMT: pto.const_expr = "e4m3",
    DST_FMT: pto.const_expr = "f32",
    ROW_BLOCK_NUM: pto.const_expr = 4,
    COL_BLOCK_NUM: pto.const_expr = 4,
):
    effective_col_block_num = COL_BLOCK_NUM + (COL_BLOCK_NUM % 2)
    total_scale_num = ROW_BLOCK_NUM * effective_col_block_num
    total_block_num = total_scale_num
    scale_loop_num = (total_scale_num + _BLOCKS_PER_DATA_LOOP - 1) // _BLOCKS_PER_DATA_LOOP
    loop_num2vf = (total_block_num + _BLOCKS_PER_DATA_LOOP - 1) // _BLOCKS_PER_DATA_LOOP
    padded_total_block_num = loop_num2vf * _BLOCKS_PER_DATA_LOOP
    total_elems = padded_total_block_num * _BLOCK_SIZE

    x_ub_ptr = pto.castptr(pto.const(_X_BASE_ADDR, dtype=pto.ui64), pto.ptr(pto.ui8, "ub"))
    scale_ub_ptr = pto.castptr(pto.const(_RAW_SCALE_BASE_ADDR, dtype=pto.ui64), pto.ptr(pto.ui8, "ub"))
    x_bytes = total_elems
    scale_bytes = scale_loop_num * _RAW_SCALE_BYTES_PER_LOOP

    if pto.const_expr(DST_FMT == "f32"):
        y_ub_ptr = pto.castptr(pto.const(_Y_BASE_ADDR, dtype=pto.ui64), pto.ptr(pto.f32, "ub"))
        y_bytes = total_elems * 4
    else:
        y_ub_ptr = pto.castptr(
            pto.const(_Y_BASE_ADDR, dtype=pto.ui64),
            pto.ptr(pto.bf16 if pto.const_expr(DST_FMT == "bf16") else pto.f16, "ub"),
        )
        y_bytes = total_elems * 2

    pto.mte_gm_ub(x_gm, x_ub_ptr, 0, x_bytes, nburst=(1, x_bytes, x_bytes))
    pto.mte_gm_ub(scale_gm, scale_ub_ptr, 0, scale_bytes, nburst=(1, scale_bytes, scale_bytes))

    pto.set_flag(pto.Pipe.MTE2, pto.Pipe.V, event_id=0)
    pto.wait_flag(pto.Pipe.MTE2, pto.Pipe.V, event_id=0)
    _compute_scale_ub(
        pto.const(_RAW_SCALE_BASE_ADDR, dtype=pto.i64),
        pto.const(_SCALE_BASE_ADDR, dtype=pto.i64),
        DST_FMT="f32",
        LOOP_NUM2VF=scale_loop_num,
    )
    _compute_data_ub(
        pto.const(_X_BASE_ADDR, dtype=pto.i64),
        pto.const(_SCALE_BASE_ADDR, dtype=pto.i64),
        pto.const(_Y_BASE_ADDR, dtype=pto.i64),
        SRC_FMT=SRC_FMT,
        DST_FMT=DST_FMT,
        LOOP_NUM2VF=loop_num2vf,
    )

    pto.set_flag(pto.Pipe.V, pto.Pipe.MTE3, event_id=0)
    pto.wait_flag(pto.Pipe.V, pto.Pipe.MTE3, event_id=0)
    pto.mte_ub_gm(y_ub_ptr, y_gm, y_bytes, nburst=(1, y_bytes, y_bytes))
    pto.pipe_barrier(pto.Pipe.ALL)


@pto.jit(
    name="anti_mx_quant_tail_axis_compute_scale_probe",
    target="a5",
    backend="vpto",
    mode="explicit",
    kernel_kind="vector",
    insert_sync=False,
)
def anti_mx_quant_tail_axis_compute_scale_probe(
    scale_local_addr: pto.i64 = _X_BASE_ADDR,
    scale_buf_addr: pto.i64 = _SCALE_BASE_ADDR,
    *,
    DST_FMT: pto.const_expr = "f32",
    LOOP_NUM2VF: pto.const_expr = 1,
):
    _compute_scale_ub(
        scale_local_addr,
        scale_buf_addr,
        DST_FMT=DST_FMT,
        LOOP_NUM2VF=LOOP_NUM2VF,
    )


@pto.jit(
    name="anti_mx_quant_tail_axis_compute_data_probe",
    target="a5",
    backend="vpto",
    mode="explicit",
    kernel_kind="vector",
    insert_sync=False,
)
def anti_mx_quant_tail_axis_compute_data_probe(
    x_local_addr: pto.i64 = _X_BASE_ADDR,
    scale_buf_addr: pto.i64 = _SCALE_BASE_ADDR,
    y_local_addr: pto.i64 = _Y_BASE_ADDR,
    *,
    SRC_FMT: pto.const_expr = "e4m3",
    DST_FMT: pto.const_expr = "f32",
    LOOP_NUM2VF: pto.const_expr = 1,
):
    _compute_data_ub(
        x_local_addr,
        scale_buf_addr,
        y_local_addr,
        SRC_FMT=SRC_FMT,
        DST_FMT=DST_FMT,
        LOOP_NUM2VF=LOOP_NUM2VF,
    )


@pto.jit(
    name="dequant_vmi_f32",
    target="a5",
    backend="vpto",
    mode="explicit",
    kernel_kind="vector",
    insert_sync=False,
)
def dequant_vmi_f32(
    x_gm: pto.ptr(pto.ui8, "gm"),
    scale_gm: pto.ptr(pto.ui8, "gm"),
    y_gm: pto.ptr(pto.f32, "gm"),
    *,
    SRC_FMT: pto.const_expr = "e4m3",
    ROW_BLOCK_NUM: pto.const_expr = 4,
    COL_BLOCK_NUM: pto.const_expr = 4,
):
    _runtime_entry(
        x_gm,
        scale_gm,
        y_gm,
        SRC_FMT=SRC_FMT,
        DST_FMT="f32",
        ROW_BLOCK_NUM=ROW_BLOCK_NUM,
        COL_BLOCK_NUM=COL_BLOCK_NUM,
    )


@pto.jit(
    name="dequant_vmi_bf16",
    target="a5",
    backend="vpto",
    mode="explicit",
    kernel_kind="vector",
    insert_sync=False,
)
def dequant_vmi_bf16(
    x_gm: pto.ptr(pto.ui8, "gm"),
    scale_gm: pto.ptr(pto.ui8, "gm"),
    y_gm: pto.ptr(pto.bf16, "gm"),
    *,
    SRC_FMT: pto.const_expr = "e4m3",
    ROW_BLOCK_NUM: pto.const_expr = 4,
    COL_BLOCK_NUM: pto.const_expr = 4,
):
    _runtime_entry(
        x_gm,
        scale_gm,
        y_gm,
        SRC_FMT=SRC_FMT,
        DST_FMT="bf16",
        ROW_BLOCK_NUM=ROW_BLOCK_NUM,
        COL_BLOCK_NUM=COL_BLOCK_NUM,
    )


@pto.jit(
    name="dequant_vmi_f16",
    target="a5",
    backend="vpto",
    mode="explicit",
    kernel_kind="vector",
    insert_sync=False,
)
def dequant_vmi_f16(
    x_gm: pto.ptr(pto.ui8, "gm"),
    scale_gm: pto.ptr(pto.ui8, "gm"),
    y_gm: pto.ptr(pto.f16, "gm"),
    *,
    SRC_FMT: pto.const_expr = "e4m3",
    ROW_BLOCK_NUM: pto.const_expr = 4,
    COL_BLOCK_NUM: pto.const_expr = 4,
):
    _runtime_entry(
        x_gm,
        scale_gm,
        y_gm,
        SRC_FMT=SRC_FMT,
        DST_FMT="f16",
        ROW_BLOCK_NUM=ROW_BLOCK_NUM,
        COL_BLOCK_NUM=COL_BLOCK_NUM,
    )
