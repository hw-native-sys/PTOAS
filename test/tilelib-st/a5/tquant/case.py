#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from pathlib import Path
import struct
import sys
import zlib

import numpy as np
import ml_dtypes

if __package__ in {None, ""}:
    _case_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(_case_dir))
    sys.path.insert(0, str(_case_dir.parent.parent))

from common import auto_main, golden_output_case
from ptodsl import pto


# ════════════════════════════════════════════════════════════════
#  Helpers
# ════════════════════════════════════════════════════════════════

def _ca(v, bs, to=32):
    n = v * bs
    return ((n + to - 1) // to) * to // bs


def _seed(name):
    return zlib.crc32(name.encode("utf-8")) & 0xFFFFFFFF


_NP_DT = {
    pto.f32: (np.float32, 4),
    pto.f16: (np.float16, 2),
    pto.bf16: (ml_dtypes.bfloat16, 2),
}


# ════════════════════════════════════════════════════════════════
#  Each: (quant_type, dtype_name, shape, scale_alg, mode, name, data_pattern)
# ════════════════════════════════════════════════════════════════

_SPECS = [
    # ── INT8 SYM (3) ──
    ("INT8_SYM", "f32", (64, 128), "N/A", "ND", "int8_sym_fp32_64x128_nd", "normal"),
    ("INT8_SYM", "f32", (128, 128), "N/A", "ND", "int8_sym_fp32_128x128_nd", "normal"),
    ("INT8_SYM", "f32", (256, 128), "N/A", "ND", "int8_sym_fp32_256x128_nd", "normal"),

    # ── INT8 ASYM (3) ──
    ("INT8_ASYM", "f32", (64, 128), "N/A", "ND", "int8_asym_fp32_64x128_nd", "normal"),
    ("INT8_ASYM", "f32", (128, 128), "N/A", "ND", "int8_asym_fp32_128x128_nd", "normal"),
    ("INT8_ASYM", "f32", (256, 128), "N/A", "ND", "int8_asym_fp32_256x128_nd", "normal"),

    # ── MXFP8 FP32 OCP ND (8) ──
    ("MXFP8", "f32", (32, 32), "OCP", "ND", "mxfp8_fp32_32x32_nd", "normal"),
    ("MXFP8", "f32", (32, 64), "OCP", "ND", "mxfp8_fp32_32x64_nd", "normal"),
    ("MXFP8", "f32", (64, 128), "OCP", "ND", "mxfp8_fp32_64x128_nd", "normal"),
    ("MXFP8", "f32", (128, 128), "OCP", "ND", "mxfp8_fp32_128x128_nd", "normal"),
    ("MXFP8", "f32", (15, 32), "OCP", "ND", "mxfp8_fp32_15x32_nd", "normal"),
    ("MXFP8", "f32", (7, 64), "OCP", "ND", "mxfp8_fp32_7x64_nd", "normal"),
    ("MXFP8", "f32", (33, 64), "OCP", "ND", "mxfp8_fp32_33x64_nd", "normal"),
    ("MXFP8", "f32", (13, 192), "OCP", "ND", "mxfp8_fp32_13x192_nd", "normal"),

    # ── MXFP8 FP32 OCP NZ (5) ──
    ("MXFP8", "f32", (32, 64), "OCP", "NZ", "mxfp8_fp32_32x64_nz", "normal"),
    ("MXFP8", "f32", (64, 128), "OCP", "NZ", "mxfp8_fp32_64x128_nz", "normal"),
    ("MXFP8", "f32", (128, 128), "OCP", "NZ", "mxfp8_fp32_128x128_nz", "normal"),
    ("MXFP8", "f32", (64, 256), "OCP", "NZ", "mxfp8_fp32_64x256_nz", "normal"),
    ("MXFP8", "f32", (64, 512), "OCP", "NZ", "mxfp8_fp32_64x512_nz", "normal"),

    # ── MXFP8 FP32 NV ND (2) ──
    ("MXFP8", "f32", (32, 128), "NV", "ND", "mxfp8_nv_fp32_32x128_nd", "normal"),
    ("MXFP8", "f32", (2, 256), "NV", "ND", "mxfp8_nv_fp32_2x256_boundary_nd", "boundary"),

    # ── MXFP8 BF16 OCP ND (14) ──
    ("MXFP8", "bf16", (32, 128), "OCP", "ND", "mxfp8_bf16_32x128_nd", "normal"),
    ("MXFP8", "bf16", (64, 128), "OCP", "ND", "mxfp8_bf16_64x128_nd", "normal"),
    ("MXFP8", "bf16", (128, 128), "OCP", "ND", "mxfp8_bf16_128x128_nd", "normal"),
    ("MXFP8", "bf16", (1, 32), "OCP", "ND", "mxfp8_bf16_1x32_nd", "normal"),
    ("MXFP8", "bf16", (2, 32), "OCP", "ND", "mxfp8_bf16_2x32_nd", "normal"),
    ("MXFP8", "bf16", (3, 32), "OCP", "ND", "mxfp8_bf16_3x32_nd", "normal"),
    ("MXFP8", "bf16", (5, 96), "OCP", "ND", "mxfp8_bf16_5x96_nd", "normal"),
    ("MXFP8", "bf16", (1, 32), "OCP", "ND", "mxfp8_bf16_1x32b_nd", "normal"),
    ("MXFP8", "bf16", (4, 256), "OCP", "ND", "mxfp8_bf16_4x256_nd", "normal"),
    ("MXFP8", "bf16", (4, 512), "OCP", "ND", "mxfp8_bf16_4x512_nd", "normal"),
    ("MXFP8", "bf16", (3, 256), "OCP", "ND", "mxfp8_bf16_3x256_nd", "normal"),
    ("MXFP8", "bf16", (5, 256), "OCP", "ND", "mxfp8_bf16_5x256_nd", "normal"),
    ("MXFP8", "bf16", (1, 192), "OCP", "ND", "mxfp8_bf16_1x192_nd", "normal"),
    ("MXFP8", "bf16", (1, 224), "OCP", "ND", "mxfp8_bf16_1x224_nd", "normal"),

    # ── MXFP8 BF16 OCP NZ (3) ──
    ("MXFP8", "bf16", (32, 128), "OCP", "NZ", "mxfp8_bf16_32x128_nz", "normal"),
    ("MXFP8", "bf16", (64, 128), "OCP", "NZ", "mxfp8_bf16_64x128_nz", "normal"),
    ("MXFP8", "bf16", (128, 128), "OCP", "NZ", "mxfp8_bf16_128x128_nz", "normal"),

    # ── MXFP8 BF16 NV ND (4) ──
    ("MXFP8", "bf16", (32, 128), "NV", "ND", "mxfp8_nv_bf16_32x128_nd", "normal"),
    ("MXFP8", "bf16", (64, 128), "NV", "ND", "mxfp8_nv_bf16_64x128_nd", "normal"),
    ("MXFP8", "bf16", (128, 128), "NV", "ND", "mxfp8_nv_bf16_128x128_nd", "normal"),
    ("MXFP8", "bf16", (2, 256), "NV", "ND", "mxfp8_nv_bf16_2x256_boundary_nd", "boundary"),

    # ── MXFP8 BF16 Exp2D OCP (1) ──
    ("MXFP8", "bf16", (1, 64), "OCP", "Exp2D", "mxfp8_bf16_1x64_static3x128_exp2d_fuzz01_nd", "fuzz01"),

    # ── MXFP8 FP16 OCP ND (5) ──
    ("MXFP8", "f16", (32, 128), "OCP", "ND", "mxfp8_fp16_32x128_nd", "normal"),
    ("MXFP8", "f16", (64, 128), "OCP", "ND", "mxfp8_fp16_64x128_nd", "normal"),
    ("MXFP8", "f16", (128, 128), "OCP", "ND", "mxfp8_fp16_128x128_nd", "normal"),
    ("MXFP8", "f16", (4, 256), "OCP", "ND", "mxfp8_fp16_4x256_nd", "normal"),
    ("MXFP8", "f16", (11, 640), "OCP", "ND", "mxfp8_fp16_11x640_nd", "normal"),

    # ── MXFP8 FP16 NV ND (4) ──
    ("MXFP8", "f16", (32, 128), "NV", "ND", "mxfp8_nv_fp16_32x128_nd", "normal"),
    ("MXFP8", "f16", (64, 128), "NV", "ND", "mxfp8_nv_fp16_64x128_nd", "normal"),
    ("MXFP8", "f16", (128, 128), "NV", "ND", "mxfp8_nv_fp16_128x128_nd", "normal"),
    ("MXFP8", "f16", (2, 256), "NV", "ND", "mxfp8_nv_fp16_2x256_boundary_nd", "boundary"),

    # ── MXFP8 FP16 OCP NZ (3) ──
    ("MXFP8", "f16", (32, 128), "OCP", "NZ", "mxfp8_fp16_32x128_nz", "normal"),
    ("MXFP8", "f16", (64, 128), "OCP", "NZ", "mxfp8_fp16_64x128_nz", "normal"),
    ("MXFP8", "f16", (128, 128), "OCP", "NZ", "mxfp8_fp16_128x128_nz", "normal"),

    # ── MXFP4 E2M1 FP16 OCP ND (9) ──
    ("MXFP4_E2M1", "f16", (2, 128), "OCP", "ND", "mxfp4_e2m1_fp16_2x128_special_nd", "special"),
    ("MXFP4_E2M1", "f16", (2, 128), "OCP", "ND", "mxfp4_e2m1_fp16_2x128_inf_only_nd", "inf_only"),
    ("MXFP4_E2M1", "f16", (2, 128), "OCP", "ND", "mxfp4_e2m1_fp16_2x128_subnormal_nd", "subnormal"),
    ("MXFP4_E2M1", "f16", (2, 128), "OCP", "ND", "mxfp4_e2m1_fp16_2x128_rounding_nd", "rounding"),
    ("MXFP4_E2M1", "f16", (2, 128), "OCP", "ND", "mxfp4_e2m1_fp16_2x128_exp_random_a_nd", "exp_random_a"),
    ("MXFP4_E2M1", "f16", (2, 128), "OCP", "ND", "mxfp4_e2m1_fp16_2x128_exp_random_b_nd", "exp_random_b"),
    ("MXFP4_E2M1", "f16", (2, 128), "OCP", "ND", "mxfp4_e2m1_fp16_2x128_mixed_nd", "mixed"),
    ("MXFP4_E2M1", "f16", (32, 1024), "OCP", "ND", "mxfp4_e2m1_fp16_32x1024_mixed_nd", "mixed"),
    ("MXFP4_E2M1", "f16", (32, 1024), "OCP", "ND", "mxfp4_e2m1_fp16_32x1024_normal_nd", "normal"),

    # ── MXFP4 E2M1 FP16 NV ND (3) ──
    ("MXFP4_E2M1", "f16", (2, 256), "NV", "ND", "mxfp4_e2m1_nv_fp16_2x256_boundary_nd", "boundary"),
    ("MXFP4_E2M1", "f16", (2, 256), "NV", "ND", "mxfp4_e2m1_nv_fp16_2x256_rounding_nd", "rounding"),
    ("MXFP4_E2M1", "f16", (2, 256), "NV", "ND", "mxfp4_e2m1_nv_fp16_2x256_mixed_nd", "mixed"),

    # ── MXFP4 E2M1 BF16 OCP ND (9) ──
    ("MXFP4_E2M1", "bf16", (2, 128), "OCP", "ND", "mxfp4_e2m1_bf16_2x128_special_nd", "special"),
    ("MXFP4_E2M1", "bf16", (2, 128), "OCP", "ND", "mxfp4_e2m1_bf16_2x128_inf_only_nd", "inf_only"),
    ("MXFP4_E2M1", "bf16", (2, 128), "OCP", "ND", "mxfp4_e2m1_bf16_2x128_subnormal_nd", "subnormal"),
    ("MXFP4_E2M1", "bf16", (2, 128), "OCP", "ND", "mxfp4_e2m1_bf16_2x128_rounding_nd", "rounding"),
    ("MXFP4_E2M1", "bf16", (2, 128), "OCP", "ND", "mxfp4_e2m1_bf16_2x128_exp_random_a_nd", "exp_random_a"),
    ("MXFP4_E2M1", "bf16", (2, 128), "OCP", "ND", "mxfp4_e2m1_bf16_2x128_exp_random_b_nd", "exp_random_b"),
    ("MXFP4_E2M1", "bf16", (2, 128), "OCP", "ND", "mxfp4_e2m1_bf16_2x128_mixed_nd", "mixed"),
    ("MXFP4_E2M1", "bf16", (32, 1024), "OCP", "ND", "mxfp4_e2m1_bf16_32x1024_mixed_nd", "mixed"),
    ("MXFP4_E2M1", "bf16", (32, 1024), "OCP", "ND", "mxfp4_e2m1_bf16_32x1024_normal_nd", "normal"),

    # ── MXFP4 E2M1 BF16 NV ND (3) ──
    ("MXFP4_E2M1", "bf16", (2, 256), "NV", "ND", "mxfp4_e2m1_nv_bf16_2x256_boundary_nd", "boundary"),
    ("MXFP4_E2M1", "bf16", (2, 256), "NV", "ND", "mxfp4_e2m1_nv_bf16_2x256_rounding_nd", "rounding"),
    ("MXFP4_E2M1", "bf16", (2, 256), "NV", "ND", "mxfp4_e2m1_nv_bf16_2x256_mixed_nd", "mixed"),

    # ── MXFP4 E2M1 BF16 NV Exp2D (1) ──
    ("MXFP4_E2M1", "bf16", (2, 128), "NV", "Exp2D", "mxfp4_e2m1_nv_bf16_2x128_static4x128_exp2d_nd", "normal"),


    ("MXFP8", "bf16", (64, 128, 128), "OCP", "DN", "bf16_64x128", "normal"),
    ("MXFP4_E2M1", "f16", (64, 128, 128), "OCP", "DN", "mxfp4_fp16_64x128", "normal"),
    ("MXFP8", "bf16", (512, 48, 64, 24), "OCP", "DN_valid_shape", "validshape_bf16_s512x48_v64x24", "normal"),
    ("MXFP8", "f16", (896, 48, 896, 34), "NV", "DN_valid_shape", "nv_validshape_fp16_s896x48_v896x34", "normal"),
    ("MXFP8", "f32", (128, 128, 128), "NV", "DN_interleave", "nv_fp32_128x128_interleaved", "normal"),
    ("MXFP4_E2M1", "bf16", (128, 128, 128), "NV", "DN_interleave", "nv_mxfp4_bf16_128x128_interleaved", "normal"),
]

assert len(_SPECS) == 86, f"Expected 86 specs, got {len(_SPECS)}"



# ════════════════════════════════════════════════════════════════
#  INT8 kernels + golden
# ════════════════════════════════════════════════════════════════

def _sym_kernel(name, r, c):
    pc = _ca(c, 4)
    dc = _ca(c, 1)

    @pto.jit(name=name, target="a5", kernel_kind="vector")
    def _k(s: pto.ptr(pto.f32, "gm"), f: pto.ptr(pto.f32, "gm"),
           d: pto.ptr(pto.i8, "gm")):
        st = pto.alloc_tile(shape=[r, pc], dtype=pto.f32, valid_shape=[r, c])
        ft = pto.alloc_tile(shape=[r, 1], dtype=pto.f32, valid_shape=[r, 1],
                            blayout="ColMajor")
        dt = pto.alloc_tile(shape=[r, dc], dtype=pto.i8, valid_shape=[r, c])
        pto.tile.load(pto.make_tensor_view(s, shape=[r, c], strides=[c, 1]), st)
        pto.tile.load(pto.make_tensor_view(f, shape=[r, 1], strides=[1, 1],
                                            layout="dn"), ft)
        pto.tile.quant(st, ft, dt)
        pto.tile.store(dt, pto.make_tensor_view(d, shape=[r, c], strides=[c, 1]))
        pto.pipe_barrier(pto.Pipe.ALL)
    return _k


def _asym_kernel(name, r, c):
    pc = _ca(c, 4)
    dc = _ca(c, 1)

    @pto.jit(name=name, target="a5", kernel_kind="vector")
    def _k(s: pto.ptr(pto.f32, "gm"), f: pto.ptr(pto.f32, "gm"),
           o: pto.ptr(pto.f32, "gm"), d: pto.ptr(pto.ui8, "gm")):
        st = pto.alloc_tile(shape=[r, pc], dtype=pto.f32, valid_shape=[r, c])
        ft = pto.alloc_tile(shape=[r, 1], dtype=pto.f32, valid_shape=[r, 1],
                            blayout="ColMajor")
        ot = pto.alloc_tile(shape=[r, 1], dtype=pto.f32, valid_shape=[r, 1],
                            blayout="ColMajor")
        dt = pto.alloc_tile(shape=[r, dc], dtype=pto.ui8, valid_shape=[r, c])
        pto.tile.load(pto.make_tensor_view(s, shape=[r, c], strides=[c, 1]), st)
        pto.tile.load(pto.make_tensor_view(f, shape=[r, 1], strides=[1, 1],
                                            layout="dn"), ft)
        pto.tile.load(pto.make_tensor_view(o, shape=[r, 1], strides=[1, 1],
                                            layout="dn"), ot)
        pto.tile.quant(st, ft, dt, offset=ot)
        pto.tile.store(dt, pto.make_tensor_view(d, shape=[r, c], strides=[c, 1]))
        pto.pipe_barrier(pto.Pipe.ALL)
    return _k


def _sym_golden(src, scale):
    return np.clip(np.round(src * scale[:, 0:1]), -128, 127).astype(np.int8)


def _asym_golden(src, scale, offset):
    return np.clip(np.round(src * scale[:, 0:1] + offset[:, 0:1]),
                   0, 255).astype(np.uint8)


# ════════════════════════════════════════════════════════════════
#  MX kernels
# ════════════════════════════════════════════════════════════════

_DTYPE_MAP = {
    "f32": pto.f32,
    "bf16": pto.bf16,
    "f16": pto.f16,
}

_BW = {
    pto.f32: 4,
    pto.f16: 2,
    pto.bf16: 2,
}

_DDT_MAP = {
    "MXFP8": pto.ui8,
    "MXFP4_E2M1": pto.f4e2m1x2,
}


def _mx_nd_kernel(name, r, c, sdt, ddt, quant_type, scale_alg):
    pad_c = _ca(c, 1)
    evc = (r * pad_c) // 32
    ec = _ca(evc, 1)
    bw = _BW[sdt]
    # TQuantMx verifier parity (axis1 flat): f32 scaling must cover
    # alignTo(groups[*2 if unrolled], 64) elements; B16 OCP scaling must
    # cover alignTo(groups, 128) elements (256B either way). unroll kicks in
    # when src physical elems > 1024 and both physical/valid elems %256==0.
    src_elems = r * pad_c
    unroll = src_elems > 1024 and src_elems % 256 == 0 and (r * c) % 256 == 0
    if bw == 4:
        need = evc * (2 if unroll else 1)
        mc = max(_ca(evc, bw), ((need + 63) // 64) * 64)
    else:
        mc = max(_ca(evc, bw), ((evc + 127) // 128) * 128)
    ds = pad_c
    is_fp4 = quant_type == "MXFP4_E2M1"
    dst_valid_c = c // 2 if is_fp4 else c
    dst_ds = _ca(c // 2, 1) if is_fp4 else ds
    groups_per_row = pad_c // 32
    ec_2d = _ca(groups_per_row, 1)

    @pto.jit(name=name, target="a5", kernel_kind="vector")
    def _k(sp: pto.ptr(sdt, "gm"), ep: pto.ptr(pto.ui8, "gm"),
           dp: pto.ptr(ddt, "gm")):
        st = pto.alloc_tile(shape=[r, pad_c], dtype=sdt, valid_shape=[r, c])
        dt = pto.alloc_tile(shape=[r, dst_ds], dtype=ddt, valid_shape=[r, dst_valid_c])
        if is_fp4:
            et = pto.alloc_tile(shape=[r, ec_2d], dtype=pto.ui8, valid_shape=[r, groups_per_row])
        else:
            et = pto.alloc_tile(shape=[1, ec], dtype=pto.ui8, valid_shape=[1, evc])
        mt = pto.alloc_tile(shape=[1, mc], dtype=sdt, valid_shape=[1, evc])
        st2 = pto.alloc_tile(shape=[1, mc], dtype=sdt, valid_shape=[1, evc])
        pto.tile.load(pto.make_tensor_view(sp, shape=[r, c], strides=[c, 1]), st)
        pto.set_flag("MTE2", "V", event_id=0)
        pto.wait_flag("MTE2", "V", event_id=0)
        kwargs = {}
        if scale_alg != "OCP":
            kwargs["quant_scale_alg"] = scale_alg.lower()
        pto.tile.quant.mx(st, dt, et, mt, st2, quant_type=quant_type, **kwargs)
        pto.set_flag("V", "MTE3", event_id=0)
        pto.wait_flag("V", "MTE3", event_id=0)
        pto.tile.store(dt, pto.make_tensor_view(dp, shape=[r, dst_valid_c], strides=[dst_valid_c, 1]))
        if is_fp4:
            pto.tile.store(et, pto.make_tensor_view(ep, shape=[r, groups_per_row], strides=[groups_per_row, 1]))
        else:
            pto.tile.store(et, pto.make_tensor_view(ep, shape=[1, evc], strides=[evc, 1]))
        pto.pipe_barrier(pto.Pipe.ALL)
    return _k


def _mx_nz_kernel(name, r, c, sdt, ddt, quant_type):
    pad_c = _ca(c, 1)
    evc = (r * pad_c) // 32
    ec = _ca(evc, 1)
    bw = _BW[sdt]
    src_elems = r * pad_c
    unroll = src_elems > 1024 and src_elems % 256 == 0 and (r * c) % 256 == 0
    if bw == 4:
        need = evc * (2 if unroll else 1)
        mc = max(_ca(evc, bw), ((need + 63) // 64) * 64)
    else:
        mc = max(_ca(evc, bw), ((evc + 127) // 128) * 128)
    ds = pad_c

    @pto.jit(name=name, target="a5", kernel_kind="vector")
    def _k(sp: pto.ptr(sdt, "gm"), ep: pto.ptr(pto.ui8, "gm"),
           dp: pto.ptr(ddt, "gm")):
        st = pto.alloc_tile(shape=[r, pad_c], dtype=sdt, valid_shape=[r, c])
        dt = pto.alloc_tile(shape=[r, ds], dtype=ddt, valid_shape=[r, c])
        et = pto.alloc_tile(shape=[1, ec], dtype=pto.ui8, valid_shape=[1, evc])
        mt = pto.alloc_tile(shape=[1, mc], dtype=sdt, valid_shape=[1, evc])
        st2 = pto.alloc_tile(shape=[1, mc], dtype=sdt, valid_shape=[1, evc])
        pto.tile.load(pto.make_tensor_view(sp, shape=[r, c], strides=[c, 1]), st)
        pto.set_flag("MTE2", "V", event_id=0)
        pto.wait_flag("MTE2", "V", event_id=0)
        pto.tile.quant.mx(st, dt, et, mt, st2, quant_type=quant_type)
        pto.set_flag("V", "MTE3", event_id=0)
        pto.wait_flag("V", "MTE3", event_id=0)
        pto.tile.store(dt, pto.make_tensor_view(dp, shape=[r, c], strides=[c, 1]))
        pto.tile.store(et, pto.make_tensor_view(ep, shape=[1, evc], strides=[evc, 1]))
        pto.pipe_barrier(pto.Pipe.ALL)
    return _k


def _mx_exp2d_kernel(name, vr, vc, sr, sc, sdt, ddt, quant_type, scale_alg):
    is_fp4 = quant_type == "MXFP4_E2M1"
    dst_valid_c = vc // 2 if is_fp4 else vc
    valid_groups = vc // 32
    ds = sc if not is_fp4 else sc // 2
    # expCols = ceil(staticCols/32, 32): physical cols must cover canonical
    # [M, N/32] and pass the 32-group alignment.
    sec = _ca(_ca(sc // 32, 32), 1)
    bw = _BW[sdt]
    mc = _ca(vr * valid_groups, bw)

    @pto.jit(name=name, target="a5", kernel_kind="vector")
    def _k(sp: pto.ptr(sdt, "gm"), ep: pto.ptr(pto.ui8, "gm"),
           dp: pto.ptr(ddt, "gm")):
        st = pto.alloc_tile(shape=[sr, sc], dtype=sdt, valid_shape=[vr, vc])
        dt = pto.alloc_tile(shape=[sr, ds], dtype=ddt, valid_shape=[vr, dst_valid_c])
        et = pto.alloc_tile(shape=[sr, sec], dtype=pto.ui8, valid_shape=[vr, valid_groups])
        mt = pto.alloc_tile(shape=[1, mc], dtype=sdt, valid_shape=[1, vr * valid_groups])
        st2 = pto.alloc_tile(shape=[1, mc], dtype=sdt, valid_shape=[1, vr * valid_groups])
        pto.tile.load(pto.make_tensor_view(sp, shape=[vr, vc], strides=[vc, 1]), st)
        pto.set_flag("MTE2", "V", event_id=0)
        pto.wait_flag("MTE2", "V", event_id=0)
        kwargs = {}
        if scale_alg != "OCP":
            kwargs["quant_scale_alg"] = scale_alg.lower()
        pto.tile.quant.mx(st, dt, et, mt, st2,
                          quant_type=quant_type, **kwargs)
        pto.set_flag("V", "MTE3", event_id=0)
        pto.wait_flag("V", "MTE3", event_id=0)
        pto.tile.store(dt, pto.make_tensor_view(dp, shape=[vr, dst_valid_c], strides=[dst_valid_c, 1]))
        pto.tile.store(et, pto.make_tensor_view(ep, shape=[vr, valid_groups], strides=[valid_groups, 1]))
        pto.pipe_barrier(pto.Pipe.ALL)
    return _k


def _mx_dn_kernel(name, r, c, npad, sdt, ddt, quant_type, scale_alg="OCP", interleave=False):
    is_fp4 = quant_type == "MXFP4_E2M1"
    dst_valid_c = c // 2 if is_fp4 else c
    ds = npad if not is_fp4 else npad // 2
    hat_m = r // 32
    hat_e = r // 64
    et_cols = _ca(npad * 2, 1) if interleave else npad

    @pto.jit(name=name, target="a5", kernel_kind="vector")
    def _k(sp: pto.ptr(sdt, "gm"), ep: pto.ptr(pto.ui8, "gm"),
           dp: pto.ptr(ddt, "gm")):
        st = pto.alloc_tile(shape=[r, npad], dtype=sdt, valid_shape=[r, c])
        dt = pto.alloc_tile(shape=[r, ds], dtype=ddt, valid_shape=[r, dst_valid_c])
        if interleave:
            et = pto.alloc_tile(shape=[hat_e, et_cols], dtype=pto.ui8, valid_shape=[hat_e, c * 2])
        else:
            et = pto.alloc_tile(shape=[hat_m, npad], dtype=pto.ui8,
                                valid_shape=[hat_m, c])
        mt = pto.alloc_tile(shape=[hat_m, npad], dtype=sdt,
                            valid_shape=[hat_m, c])
        st2 = pto.alloc_tile(shape=[hat_m, npad], dtype=sdt,
                             valid_shape=[hat_m, c])
        pto.tile.load(pto.make_tensor_view(sp, shape=[r, c], strides=[c, 1]), st)
        pto.set_flag("MTE2", "V", event_id=0)
        pto.wait_flag("MTE2", "V", event_id=0)
        kw = {}
        if scale_alg != "OCP":
            kw["quant_scale_alg"] = scale_alg.lower()
        if interleave:
            kw["interleave"] = True
        pto.tile.quant.mx(st, dt, et, mt, st2, quant_type=quant_type, grp_axis=0, **kw)
        pto.set_flag("V", "MTE3", event_id=0)
        pto.wait_flag("V", "MTE3", event_id=0)
        pto.tile.store(dt, pto.make_tensor_view(dp, shape=[r, dst_valid_c], strides=[dst_valid_c, 1]))
        if interleave:
            pto.tile.store(et, pto.make_tensor_view(ep, shape=[hat_e, c * 2], strides=[c * 2, 1]))
        else:
            pto.tile.store(et, pto.make_tensor_view(ep, shape=[hat_m, c],
                                                    strides=[c, 1]))
        pto.pipe_barrier(pto.Pipe.ALL)
    return _k


def _mx_dn_valid_shape_kernel(name, sr, sc, vr, vc, sdt, ddt, quant_type, scale_alg="OCP"):
    hat_m = vr // 32
    # Upstream TQuantMxOp verifier: interleaved exp physical rows must be
    # src PHYSICAL rows / 64 (not valid rows).
    hat_e = sr // 64
    et_cols = _ca(sc * 2, 1)
    is_fp4 = quant_type == "MXFP4_E2M1"
    dst_valid_c = vc // 2 if is_fp4 else vc
    ds = _ca(sc, 1) if not is_fp4 else _ca(sc // 2, 1)

    @pto.jit(name=name, target="a5", kernel_kind="vector")
    def _k(sp: pto.ptr(sdt, "gm"), ep: pto.ptr(pto.ui8, "gm"),
           dp: pto.ptr(ddt, "gm")):
        st = pto.alloc_tile(shape=[sr, sc], dtype=sdt, valid_shape=[vr, vc])
        dt = pto.alloc_tile(shape=[sr, ds], dtype=ddt, valid_shape=[vr, dst_valid_c])
        et = pto.alloc_tile(shape=[hat_e, et_cols], dtype=pto.ui8, valid_shape=[vr // 64, vc * 2])
        mt = pto.alloc_tile(shape=[hat_m, sc], dtype=sdt, valid_shape=[vr // 32, vc])
        st2 = pto.alloc_tile(shape=[hat_m, sc], dtype=sdt, valid_shape=[vr // 32, vc])
        pto.tile.load(pto.make_tensor_view(sp, shape=[vr, vc], strides=[vc, 1]), st)
        pto.set_flag("MTE2", "V", event_id=0)
        pto.wait_flag("MTE2", "V", event_id=0)
        kw = {}
        if scale_alg != "OCP":
            kw["quant_scale_alg"] = scale_alg.lower()
        kw["interleave"] = True
        pto.tile.quant.mx(st, dt, et, mt, st2, quant_type=quant_type, grp_axis=0, **kw)
        pto.set_flag("V", "MTE3", event_id=0)
        pto.wait_flag("V", "MTE3", event_id=0)
        pto.tile.store(dt, pto.make_tensor_view(dp, shape=[vr, dst_valid_c], strides=[dst_valid_c, 1]))
        pto.tile.store(et, pto.make_tensor_view(ep, shape=[vr // 64, vc * 2], strides=[vc * 2, 1]))
        pto.pipe_barrier(pto.Pipe.ALL)
    return _k


# ════════════════════════════════════════════════════════════════
#  MX golden: FP32 → FP8 E4M3 element conversion
# ════════════════════════════════════════════════════════════════

def _fp32_to_fp8_e4m3(val):
    if np.isnan(val):
        return 0x7F
    if np.isinf(val):
        return 0x7E if val > 0 else 0xFE
    sign = 0x80 if val < 0 else 0
    x = abs(val)
    if x == 0:
        return sign
    exp = int(np.floor(np.log2(x)))
    exp = max(-6, min(7, exp))
    mant_frac = (x / (2 ** exp) - 1.0) * 4.0
    mant = int(mant_frac)
    if mant_frac - mant == 0.5:
        mant = (mant + 1) if (mant & 1) else mant
    else:
        mant = int(mant_frac + 0.5)
    if mant >= 4:
        exp += 1
        mant = 0
    if exp > 7:
        return 0x7E | sign
    if exp < -6:
        return sign
    return sign | ((exp + 7) << 2) | mant


# ════════════════════════════════════════════════════════════════
#  MX golden: OCP shared exponent + scaling
# ════════════════════════════════════════════════════════════════

def _ocp_exp_and_scale(max_abs_f32, emax):
    """OCP: shared_exp = max(0, max_exp - emax), scale = 2^(254-shared_exp)."""
    bits = struct.unpack("I", struct.pack("f", float(max_abs_f32)))[0]
    e = (bits >> 23) & 0xFF
    mant = bits & 0x7FFFFF
    if e == 0xFF and mant != 0:
        return 0xFF, float("nan")
    if e <= emax:
        return 0x00, struct.unpack("f", struct.pack("I", 0x7F000000))[0]
    shared_exp = e - emax
    scale_exp = 254 - shared_exp
    scale = struct.unpack("f", struct.pack("I", scale_exp << 23))[0]
    return shared_exp, scale


def _nv_exp_and_scale(max_abs_f32, descale_multiplier):
    """NV: shared_exp = ceil(log2(max * descale)), with rounding."""
    scaled_max = float(max_abs_f32) * descale_multiplier
    bits = struct.unpack("I", struct.pack("f", scaled_max))[0]
    e = (bits >> 23) & 0xFF
    mant = bits & 0x7FFFFF
    if e == 0xFF:
        if mant != 0:
            return 0xFF, float("nan")
        return 0xFE, struct.unpack("f", struct.pack("I", 0x00400000))[0]
    shared_exp = e
    if mant > 0 and e > 0 and e < 0xFE:
        shared_exp = e + 1
    elif mant > 0x00400000 and e == 0:
        shared_exp = 1
    scale_exp = 0xFE - shared_exp
    scale = struct.unpack("f", struct.pack("I", scale_exp << 23))[0]
    return shared_exp, scale


_EMAX = {"MXFP8": 8, "MXFP4_E2M1": 8}
_DESCALE = {"MXFP8": 1.0 / 448.0, "MXFP4_E2M1": 1.0 / 6.0}


from ml_dtypes import float8_e4m3fn as _f8e4m3
from ml_dtypes import bfloat16 as _bf16
from ml_dtypes import float4_e2m1fn as _f4e2m1


def _nv_fp32_to_fp8_element(data_abs_max):

    if np.float32(data_abs_max) == np.float32(0.0):
        return 0x00, np.float32(0x7F000000).view(np.float32)
    descale = np.float32(data_abs_max) * np.float32(1.0 / 448.0)
    bits = np.uint32(np.frombuffer(descale.tobytes(), dtype=np.uint32)[0])
    exponent = int((bits & np.uint32(0x7F800000)) >> np.uint32(23))
    mantissa = int(bits & np.uint32(0x007FFFFF))
    if exponent == 0xFF:
        if mantissa == 0:
            return 0xFE, np.float32(2.0**-127)
        return 0xFF, np.float32(np.nan)
    round_up = mantissa > 0 and exponent != 0xFE and not (exponent == 0 and mantissa <= 0x00400000)
    e8m0 = exponent + (1 if round_up else 0)
    if e8m0 == 0:
        return e8m0, np.float32(0x7F000000).view(np.float32)
    if e8m0 == 0xFE:
        return e8m0, np.float32(2.0**-127)
    scaling_bits = np.uint32((254 - e8m0) << 23)
    return e8m0, scaling_bits.view(np.float32)


def _pad_src(src, r, c, dtype):

    pad_c = _ca(c, 1)
    if pad_c == c:
        return src.reshape(r, c)
    padded = np.zeros((r, pad_c), dtype=dtype)
    padded[:, :c] = src.reshape(r, c)
    return padded


def _mxfp8_dn_golden(src, r, c, scale_alg, dt_name):

    from _gen_data_dn import get_group_max_dn, scale_data_dn, fp32_maxes_to_fp8, nv_maxes_to_mx, fp32_to_e4m3
    pad_c = _ca(c, 1)
    npad = c
    if dt_name == "f32":
        src_padded = _pad_src(src, r, c, np.float32)
        src_fp32 = src_padded.astype(np.float32)
    elif dt_name == "bf16":
        src_padded = _pad_src(src, r, c, _bf16)
        src_fp32 = src_padded.astype(np.float32)
    else:
        src_padded = _pad_src(src, r, c, np.float16)
        src_fp32 = src_padded.astype(np.float32)

    group_max = get_group_max_dn(src_fp32, group_size=32)
    if scale_alg == "NV":
        e8m0, scaling = nv_maxes_to_mx(group_max, 448.0)
    else:
        e8m0, scaling = fp32_maxes_to_fp8(group_max)
    scaled = scale_data_dn(src_fp32, scaling, group_size=32)
    scaled = np.clip(scaled, -448.0, 448.0)
    data_fp8 = fp32_to_e4m3(scaled).reshape(r, pad_c)[:, :c].copy()
    return data_fp8


def _mxfp8_golden(src, r, c, scale_alg):

    src = _pad_src(src, r, c, np.float32)
    pad_c = _ca(c, 1)
    n = r * pad_c
    num_groups = n // 32
    data_abs = np.abs(src).reshape(-1, 32)
    group_max = np.max(data_abs, axis=1)

    if scale_alg == "NV":
        scalings = np.array([_nv_fp32_to_fp8_element(mx)[1] for mx in group_max],
                            dtype=np.float32).reshape(-1, 1)
    else:
        scalings = np.array([_ocp_exp_and_scale(mx, _EMAX["MXFP8"])[1] for mx in group_max],
                            dtype=np.float32).reshape(-1, 1)

    scaled = src.reshape(-1, 32) * scalings
    scaled = np.clip(scaled, -448.0, 448.0)
    data_fp8 = scaled.astype(_f8e4m3).view(np.uint8).reshape(r, pad_c)[:, :c].copy()
    return data_fp8


def _mxfp8_bf16_golden(src, r, c, scale_alg):

    src_bf16 = _pad_src(src, r, c, _bf16)
    pad_c = _ca(c, 1)
    data_abs = np.abs(src_bf16).reshape(-1, 32)
    with np.errstate(invalid="ignore"):
        group_max_bf16 = np.max(data_abs, axis=1)
    group_max_fp32 = group_max_bf16.astype(np.float32)

    if scale_alg == "NV":
        scalings_fp32 = np.array([_nv_fp32_to_fp8_element(mx)[1] for mx in group_max_fp32],
                                 dtype=np.float32).reshape(-1, 1)
    else:
        scalings_fp32 = np.array([_ocp_exp_and_scale(mx, _EMAX["MXFP8"])[1] for mx in group_max_fp32],
                                 dtype=np.float32).reshape(-1, 1)
    scalings_bf16 = scalings_fp32.astype(_bf16)

    scaled = (src_bf16.reshape(-1, 32) * scalings_bf16).astype(_bf16)
    scaled = np.clip(scaled, _bf16(-448), _bf16(448))
    data_fp8 = scaled.astype(_f8e4m3).view(np.uint8).reshape(r, pad_c)[:, :c].copy()
    return data_fp8


def _fp32_to_fp4_e2m1(val):
    """F32 → FP4 E2M1 nibble (0-15), packed two per byte."""
    if np.isnan(val):
        return 0x7
    if np.isinf(val):
        return 0x7 if val > 0 else 0xF
    sign = 1 if val < 0 else 0
    x = abs(val)
    if x == 0:
        return 0x0 | (sign << 3)
    exp = int(np.floor(np.log2(x)))
    exp = max(-1, min(2, exp))
    mant_frac = (x / (2 ** exp) - 1.0) * 2.0
    mant = int(mant_frac + 0.5)
    if mant >= 2:
        exp += 1
        mant = 0
    if exp > 2:
        return 0x6 | (sign << 3)
    if exp < -1:
        return 0x0 | (sign << 3)
    code = (sign << 3) | ((exp + 1) << 1) | mant
    return code


def _mxfp4_golden(src, r, c, scale_alg):
    """MXFP4 E2M1 golden: → packed FP4 bytes."""
    src = _pad_src(src, r, c, np.float32).flatten()
    pad_c = _ca(c, 1)
    n = r * pad_c
    num_groups = n // 32
    codes = np.zeros(n, dtype=np.uint8)
    for g in range(num_groups):
        base = g * 32
        group = src[base:base + 32]
        mx = float(np.max(np.abs(group)))
        if scale_alg == "NV":
            _, scale = _nv_exp_and_scale(mx, _DESCALE["MXFP4_E2M1"])
        else:
            _, scale = _ocp_exp_and_scale(mx, _EMAX["MXFP4_E2M1"])
        for i in range(32):
            codes[base + i] = _fp32_to_fp4_e2m1(group[i] * scale)
    packed = np.zeros(n // 2, dtype=np.uint8)
    for i in range(0, n, 2):
        packed[i // 2] = (codes[i] & 0xF) | ((codes[i + 1] & 0xF) << 4)
    return packed.reshape(r, pad_c // 2)[:, :c // 2].copy()


# ════════════════════════════════════════════════════════════════
#  Input generators with data patterns
# ════════════════════════════════════════════════════════════════

def _gen_pattern(pattern, r, c, dtype_np, seed_k):
    rng = np.random.default_rng(seed_k)
    n = r * c

    if pattern == "normal":
        mags = rng.lognormal(mean=0.0, sigma=2.0, size=(r, c))
        signs = np.where(rng.random((r, c)) < 0.5, -1.0, 1.0)
        data = (mags * signs).astype(np.float32)
        data = np.clip(data, -1e8, 1e8)
    elif pattern == "boundary":
        data = rng.uniform(-20, 20, size=(r, c))
        data.flat[0] = 448.0
        data.flat[1] = -448.0
        if n > 2:
            data.flat[2] = 1.0
        if n > 3:
            data.flat[3] = -1.0
    elif pattern == "special":
        vals = [0.0, float("inf"), -float("inf"), float("nan"),
                448.0, -448.0, 1.0, -1.0, 0.5, -0.5, 256.0, -256.0,
                1e-38, -1e-38, 3.0, -3.0]
        data = np.array(vals * (n // len(vals) + 1))[:n].reshape(r, c)
    elif pattern == "inf_only":
        data = np.zeros((r, c))
        data.flat[::2] = float("inf")
        data.flat[1::2] = -float("inf")
    elif pattern == "subnormal":
        data = rng.uniform(-20, 20, size=(r, c))
        data.flat[::4] = 1e-38
        data.flat[1::4] = -1e-38
        data.flat[2::4] = 1e-40
        data.flat[3::4] = -1e-40
    elif pattern == "rounding":
        vals = [0.5, 1.5, 2.5, 3.5, -0.5, -1.5, -2.5, -3.5,
                0.25, 0.75, 1.25, 1.75, -0.25, -0.75, -1.25, -1.75]
        data = np.array(vals * (n // len(vals) + 1))[:n].reshape(r, c)
    elif pattern == "exp_random_a":
        data = rng.uniform(-20, 20, size=(r, c))
    elif pattern == "exp_random_b":
        data = rng.uniform(-100, 100, size=(r, c))
    elif pattern == "mixed":
        data = rng.uniform(-20, 20, size=(r, c))
        for i in range(0, n, 32):
            if i < n:
                data.flat[i] = float("inf")
            if i + 1 < n:
                data.flat[i + 1] = 0.0
            if i + 2 < n:
                data.flat[i + 2] = 1e-38
            if i + 3 < n:
                data.flat[i + 3] = 448.0
    else:
        data = rng.uniform(-20, 20, size=(r, c))

    return data.astype(dtype_np)


def _mx4_inputs(r, c, dtype_name, pattern, scale_alg, name=""):

    from _gen_data_nd import make_mxfp4_e2m1_fp16_data, make_mxfp4_e2m1_bf16_data
    case_suffix = pattern
    if "static4x128_exp2d" in name:
        case_suffix = "static4x128_exp2d"
    gen_scale_alg = "nv" if scale_alg == "NV" else "ocp"
    if dtype_name == "f16":
        data = make_mxfp4_e2m1_fp16_data(r, c, case_suffix, gen_scale_alg)
    else:
        data = make_mxfp4_e2m1_bf16_data(r, c, case_suffix, gen_scale_alg)
    return [data]


def _mx_inputs(r, c, dtype_name, pattern, seed_k, quant_type=None, scale_alg="OCP", name=""):
    if quant_type == "MXFP4_E2M1":
        return _mx4_inputs(r, c, dtype_name, pattern, scale_alg, name)
    np_dt, _ = _NP_DT[_DTYPE_MAP[dtype_name]]
    return [_gen_pattern(pattern, r, c, np_dt, seed_k)]


def _mxfp8_fp16_golden(src, r, c, scale_alg):

    src_f16 = _pad_src(src, r, c, np.float16)
    pad_c = _ca(c, 1)
    data_abs = np.abs(src_f16).reshape(-1, 32)
    with np.errstate(invalid="ignore"):
        group_max_f16 = np.max(data_abs, axis=1)
    group_max_fp32 = group_max_f16.astype(np.float32)

    if scale_alg == "NV":
        scalings_fp32 = np.array([_nv_fp32_to_fp8_element(mx)[1] for mx in group_max_fp32],
                                 dtype=np.float32).reshape(-1, 1)
    else:
        scalings_fp32 = np.array([_ocp_exp_and_scale(mx, _EMAX["MXFP8"])[1] for mx in group_max_fp32],
                                 dtype=np.float32).reshape(-1, 1)
    scalings_bf16 = scalings_fp32.astype(_bf16).astype(np.float32)

    src_fp32 = src_f16.astype(np.float32).reshape(-1, 32)
    scaled = src_fp32 * scalings_bf16
    scaled = np.clip(scaled, -448.0, 448.0)
    data_fp8 = scaled.astype(_f8e4m3).view(np.uint8).reshape(r, pad_c)[:, :c].copy()
    return data_fp8


def _mxfp4_bf16_golden(src, r, c, scale_alg):

    from _gen_data_nd import fp16_maxes_to_e2m1, nv_maxes_to_e2m1, encode_e2m1_magic_scalar, pack_fp4_e2m1
    src_bf16 = _pad_src(src, r, c, _bf16)
    pad_c = _ca(c, 1)
    src_fp32 = src_bf16.astype(np.float32)

    data_abs = np.abs(src_bf16).reshape(-1, 32)
    with np.errstate(invalid="ignore"):
        group_max_bf16 = np.max(data_abs, axis=1).astype(np.float32)

    if scale_alg == "NV":
        e8m0, scaling_bf16 = nv_maxes_to_e2m1(group_max_bf16)
    else:
        e8m0, scaling_bf16 = fp16_maxes_to_e2m1(group_max_bf16)

    scaling_bf16 = scaling_bf16.astype(_bf16)
    scaled = (src_bf16.reshape(-1, 32) * scaling_bf16).astype(_bf16).reshape(src_bf16.shape)
    codes = np.vectorize(encode_e2m1_magic_scalar, otypes=[np.uint8])(scaled.astype(np.float32))
    packed = pack_fp4_e2m1(codes, r, pad_c)
    return packed[:, :c // 2].copy()


def _mxfp4_fp16_golden(src, r, c, scale_alg):

    from _gen_data_nd import float32_to_bf16_trunc, fp16_maxes_to_e2m1, nv_maxes_to_e2m1, encode_e2m1_magic_scalar, pack_fp4_e2m1
    src_f16 = _pad_src(src, r, c, np.float16)
    pad_c = _ca(c, 1)
    n = r * pad_c
    src_fp32 = src_f16.astype(np.float32)

    if scale_alg == "NV":
        data_abs = np.abs(src_f16).astype(np.float16)
        with np.errstate(invalid="ignore"):
            group_max = np.max(data_abs.reshape(-1, 32), axis=1).astype(np.float32)
        e8m0, scaling_bf16 = nv_maxes_to_e2m1(group_max)
    else:
        src_bf16_for_max = float32_to_bf16_trunc(src_fp32)
        data_abs = np.abs(src_bf16_for_max).astype(np.float32)
        group_max_bf16 = np.max(data_abs.reshape(-1, 32), axis=1)
        e8m0, scaling_bf16 = fp16_maxes_to_e2m1(group_max_bf16)

    scaled = (src_fp32.reshape(-1, 32) * scaling_bf16.astype(np.float32)).reshape(src_f16.shape).astype(np.float32)
    codes = np.vectorize(encode_e2m1_magic_scalar, otypes=[np.uint8])(scaled)
    packed = pack_fp4_e2m1(codes, r, pad_c)
    return packed[:, :c // 2].copy()


def _mx_golden(src, r, c, quant_type, scale_alg, dt_name="f32", dn_mode=False):
    if quant_type == "MXFP8":
        if dn_mode:
            return _mxfp8_dn_golden(src, r, c, scale_alg, dt_name)
        if dt_name == "bf16":
            return _mxfp8_bf16_golden(src, r, c, scale_alg)
        if dt_name == "f16":
            return _mxfp8_fp16_golden(src, r, c, scale_alg)
        return _mxfp8_golden(src, r, c, scale_alg)
    else:
        if dt_name == "bf16":
            return _mxfp4_bf16_golden(src, r, c, scale_alg)
        if dt_name == "f16":
            return _mxfp4_fp16_golden(src, r, c, scale_alg)
        return _mxfp4_golden(src, r, c, scale_alg)


def _sym_inputs(r, c, seed_k):
    rng = np.random.default_rng(seed_k)
    src = rng.uniform(-10, 10, size=(r, c)).astype(np.float32)
    scale = (rng.uniform(0.5, 2.0, size=(r, 1)) * 0.001).astype(np.float32)
    return [src, scale]


def _asym_inputs(r, c, seed_k):
    rng = np.random.default_rng(seed_k)
    src = rng.uniform(-10, 10, size=(r, c)).astype(np.float32)
    scale = (rng.uniform(0.5, 2.0, size=(r, 1)) * 0.001).astype(np.float32)
    offset = rng.uniform(-3, 3, size=(r, 1)).astype(np.float32)
    return [src, scale, offset]


# ════════════════════════════════════════════════════════════════
#  Case registration
# ════════════════════════════════════════════════════════════════

CASES = []

for qt, dt_name, shape, alg, mode, name, pattern in _SPECS:
    kn = f"tquant_{name}"
    r, c = shape[0], shape[1]

    if qt in ("INT8_SYM", "INT8_ASYM"):
        if qt == "INT8_SYM":
            _k = _sym_kernel(kn, r, c)
            CASES.append(golden_output_case(
                kn, _k,
                inputs=lambda r=r, c=c, n=kn: _sym_inputs(r, c, _seed(n)),
                expected=_sym_golden,
                output_shape=(r, c), output_dtype=np.int8,
                rtol=0, atol=0,
            ))
        else:
            _k = _asym_kernel(kn, r, c)
            CASES.append(golden_output_case(
                kn, _k,
                inputs=lambda r=r, c=c, n=kn: _asym_inputs(r, c, _seed(n)),
                expected=_asym_golden,
                output_shape=(r, c), output_dtype=np.uint8,
                rtol=0, atol=0,
            ))
        continue

    sdt = _DTYPE_MAP[dt_name]
    ddt = _DDT_MAP[qt]
    sk = _seed(name)

    if mode == "ND":
        _k = _mx_nd_kernel(kn, r, c, sdt, ddt, qt, alg)
        out_c = c if qt == "MXFP8" else c // 2
        evc = (r * c) // 32
        CASES.append(golden_output_case(
            kn, _k,
            inputs=lambda r=r, c=c, dn=dt_name, p=pattern, n=kn, evc=evc, q=qt, a=alg:
                [*_mx_inputs(r, c, dn, p, _seed(n), q, a, n), np.zeros(evc, dtype=np.uint8)],
            expected=lambda src, *args, r=r, c=c, q=qt, a=alg, dn=dt_name: _mx_golden(src, r, c, q, a, dn),
            output_shape=(r, out_c), output_dtype=np.uint8,
            rtol=0, atol=0,
        ))
    elif mode == "NZ":
        _k = _mx_nz_kernel(kn, r, c, sdt, ddt, qt)
        out_c = c if qt == "MXFP8" else c // 2
        evc = (r * c) // 32
        CASES.append(golden_output_case(
            kn, _k,
            inputs=lambda r=r, c=c, dn=dt_name, p=pattern, n=kn, evc=evc, q=qt, a=alg:
                [*_mx_inputs(r, c, dn, p, _seed(n), q, a, n),
                 np.zeros(evc, dtype=np.uint8)],
            expected=lambda src, *args, r=r, c=c, q=qt, a=alg, dn=dt_name: _mx_golden(src, r, c, q, a, dn),
            output_shape=(r, out_c), output_dtype=np.uint8,
            rtol=0, atol=0,
        ))
    elif mode == "Exp2D":
        if "static3x128" in name:
            sr, sc = 3, 128
        elif "static4x128" in name:
            sr, sc = 4, 128
        else:
            sr, sc = r, c
        _k = _mx_exp2d_kernel(kn, r, c, sr, sc, sdt, ddt, qt, alg)
        out_c = c if qt == "MXFP8" else c // 2
        evc = (r * c) // 32
        CASES.append(golden_output_case(
            kn, _k,
            inputs=lambda r=r, c=c, dn=dt_name, p=pattern, n=kn, evc=evc, q=qt, a=alg:
                [*_mx_inputs(r, c, dn, p, _seed(n), q, a, n),
                 np.zeros(evc, dtype=np.uint8)],
            expected=lambda src, *args, r=r, c=c, q=qt, a=alg, dn=dt_name: _mx_golden(src, r, c, q, a, dn),
            output_shape=(r, out_c), output_dtype=np.uint8,
            rtol=0, atol=0,
        ))
    elif mode == "DN":
        npad = shape[2] if len(shape) > 2 else c
        _k = _mx_dn_kernel(kn, r, c, npad, sdt, ddt, qt, alg)
        out_c = c if qt == "MXFP8" else c // 2
        hat_m = r // 32
        evc = hat_m * c
        CASES.append(golden_output_case(
            kn, _k,
            inputs=lambda r=r, c=c, dn=dt_name, p=pattern, n=kn, evc=evc, q=qt, a=alg:
                [*_mx_inputs(r, c, dn, p, _seed(n), q, a, n), np.zeros(evc, dtype=np.uint8)],
            expected=lambda src, *args, r=r, c=c, q=qt, a=alg, dn=dt_name: _mx_golden(src, r, c, q, a, dn, dn_mode=True),
            output_shape=(r, out_c), output_dtype=np.uint8,
            rtol=0, atol=0,
        ))
    elif mode == "DN_interleave":
        npad = shape[2] if len(shape) > 2 else c
        _k = _mx_dn_kernel(kn, r, c, npad, sdt, ddt, qt, alg, interleave=True)
        out_c = c if qt == "MXFP8" else c // 2
        hat_e = r // 64
        et_cols = _ca(c * 2, 32)
        evc = hat_e * et_cols
        CASES.append(golden_output_case(
            kn, _k,
            inputs=lambda r=r, c=c, dn=dt_name, p=pattern, n=kn, evc=evc, q=qt, a=alg:
                [*_mx_inputs(r, c, dn, p, _seed(n), q, a, n), np.zeros(evc, dtype=np.uint8)],
            expected=lambda src, *args, r=r, c=c, q=qt, a=alg, dn=dt_name: _mx_golden(src, r, c, q, a, dn, dn_mode=True),
            output_shape=(r, out_c), output_dtype=np.uint8,
            rtol=0, atol=0,
        ))
    elif mode == "DN_valid_shape":
        sr, sc_dim, vr, vc = shape[0], shape[1], shape[2], shape[3]
        _k = _mx_dn_valid_shape_kernel(kn, sr, sc_dim, vr, vc, sdt, ddt, qt, alg)
        out_c = vc if qt == "MXFP8" else vc // 2
        hat_e = vr // 64
        et_cols = _ca(vc * 2, 32)
        evc = hat_e * et_cols
        CASES.append(golden_output_case(
            kn, _k,
            inputs=lambda vr=vr, vc=vc, dn=dt_name, p=pattern, n=kn, evc=evc, q=qt, a=alg:
                [*_mx_inputs(vr, vc, dn, p, _seed(n), q, a, n), np.zeros(evc, dtype=np.uint8)],
            expected=lambda src, *args, vr=vr, vc=vc, q=qt, a=alg, dn=dt_name: _mx_golden(src, vr, vc, q, a, dn, dn_mode=True),
            output_shape=(vr, out_c), output_dtype=np.uint8,
            rtol=0, atol=0,
        ))


auto_main(globals())
