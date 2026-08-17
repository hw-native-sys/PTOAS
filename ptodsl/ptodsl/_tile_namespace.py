# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from . import _ops


def _row_reduction_tmp_metadata(src):
    surface_metadata = getattr(src, "surface_metadata", None)
    if surface_metadata is not None:
        shape = surface_metadata.get("shape")
        dtype = surface_metadata.get("dtype")
        memory_space = surface_metadata.get("memory_space")
        if shape is not None and dtype is not None and memory_space is not None:
            return {
                "shape": shape,
                "dtype": dtype,
                "memory_space": memory_space,
                "valid_shape": surface_metadata.get("valid_shape"),
            }

    parsed = _ops.parse_tile_type_metadata(_ops.unwrap_surface_value(src).type)
    if parsed is None:
        return None
    return {
        "shape": parsed["shape_dims"],
        "dtype": parsed["element_type"],
        "memory_space": parsed["memory_space"],
        "valid_shape": parsed["valid_dims"],
    }


def _resolve_row_reduction_tmp(src, tmp):
    if tmp is not None:
        return tmp
    metadata = _row_reduction_tmp_metadata(src)
    if metadata is None:
        return _ops.alloc_tile(tile_type=_ops.unwrap_surface_value(src).type)
    alloc_kwargs = {
        "shape": list(metadata["shape"]),
        "dtype": metadata["dtype"],
        "memory_space": metadata["memory_space"],
        "blayout": "RowMajor",
        "slayout": "NoneBox",
    }
    if metadata["valid_shape"] is not None:
        alloc_kwargs["valid_shape"] = list(metadata["valid_shape"])
    return _ops.alloc_tile(**alloc_kwargs)


class _TileNamespace:
    mov = staticmethod(_ops.tmov)
    transpose = staticmethod(_ops.ttrans)
    extract = staticmethod(_ops.textract)
    insert = staticmethod(_ops.tinsert)
    concat = staticmethod(_ops.tconcat)
    matmul = staticmethod(_ops.tmatmul)
    matmul_acc = staticmethod(_ops.tmatmul_acc)
    matmul_mx = staticmethod(_ops.tmatmul_mx)
    matmul_mx_acc = staticmethod(_ops.tmatmul_mx_acc)
    matmul_mx_bias = staticmethod(_ops.tmatmul_mx_bias)
    gemv_mx = staticmethod(_ops.tgemv_mx)
    gemv_mx_acc = staticmethod(_ops.tgemv_mx_acc)
    gemv_mx_bias = staticmethod(_ops.tgemv_mx_bias)

    @staticmethod
    def load(src, tile, *, offsets=None, sizes=None):
        if offsets is None and sizes is None and _ops._is_partition_tensor_view(src):
            return _ops.tload(src, tile)
        part = _ops._tile_transfer_partition(
            src,
            tile,
            offsets=offsets,
            sizes=sizes,
            context="tile.load(...)",
        )
        return _ops.tload(part, tile)

    @staticmethod
    def store(tile, dst, *, offsets=None, sizes=None, fp=None):
        if offsets is None and sizes is None and _ops._is_partition_tensor_view(dst):
            return _ops.tstore(tile, dst, fp=fp)
        part = _ops._tile_transfer_partition(
            dst,
            tile,
            offsets=offsets,
            sizes=sizes,
            context="tile.store(...)",
        )
        return _ops.tstore(tile, part, fp=fp)

    add = staticmethod(_ops.tadd)
    addrelu = staticmethod(_ops.taddrelu)
    sub = staticmethod(_ops.tsub)
    mul = staticmethod(_ops.tmul)
    div = staticmethod(_ops.tdiv)
    max = staticmethod(_ops.tmax)
    min = staticmethod(_ops.tmin)

    adds = staticmethod(_ops.tadds)
    subs = staticmethod(_ops.tsubs)
    muls = staticmethod(_ops.tmuls)
    divs = staticmethod(_ops.tdivs)
    maxs = staticmethod(_ops.tmaxs)
    mins = staticmethod(_ops.tmins)

    exp = staticmethod(_ops.texp)
    log = staticmethod(_ops.tlog)
    sqrt = staticmethod(_ops.tsqrt)
    rsqrt = staticmethod(_ops.trsqrt)
    recip = staticmethod(_ops.trecip)
    abs = staticmethod(_ops.tabs)
    neg = staticmethod(_ops.tneg)
    dequant = staticmethod(_ops.tdequant)
    print = staticmethod(_ops.tprint)

    relu = staticmethod(_ops.trelu)
    lrelu = staticmethod(_ops.tlrelu)

    @staticmethod
    def rowsum(src, dst, *, tmp=None):
        return _ops.trowsum(src, _resolve_row_reduction_tmp(src, tmp), dst)

    @staticmethod
    def rowmax(src, dst, *, tmp=None):
        return _ops.trowmax(src, _resolve_row_reduction_tmp(src, tmp), dst)

    @staticmethod
    def rowmin(src, dst, *, tmp=None):
        return _ops.trowmin(src, _resolve_row_reduction_tmp(src, tmp), dst)

    @staticmethod
    def rowprod(src, dst, *, tmp=None):
        return _ops.trowprod(src, _resolve_row_reduction_tmp(src, tmp), dst)

    @staticmethod
    def rowargmax(src, dst, *, tmp=None):
        return _ops.trowargmax(src, _resolve_row_reduction_tmp(src, tmp), dst)

    @staticmethod
    def rowargmin(src, dst, *, tmp=None):
        return _ops.trowargmin(src, _resolve_row_reduction_tmp(src, tmp), dst)

    colsum = staticmethod(_ops.tcolsum)
    colmax = staticmethod(_ops.tcolmax)
    colmin = staticmethod(_ops.tcolmin)
    colprod = staticmethod(_ops.tcolprod)
    colargmax = staticmethod(_ops.tcolargmax)
    colargmin = staticmethod(_ops.tcolargmin)

    cmp = staticmethod(_ops.tcmp)
    cmps = staticmethod(_ops.tcmps)

    expands = staticmethod(_ops.texpands)
    reshape = staticmethod(_ops.treshape)
    rowexpand = staticmethod(_ops.trowexpand)
    colexpand = staticmethod(_ops.tcolexpand)

    rowexpandadd = staticmethod(_ops.trowexpandadd)
    rowexpandsub = staticmethod(_ops.trowexpandsub)
    rowexpandmul = staticmethod(_ops.trowexpandmul)
    rowexpanddiv = staticmethod(_ops.trowexpanddiv)
    rowexpandmax = staticmethod(_ops.trowexpandmax)
    rowexpandmin = staticmethod(_ops.trowexpandmin)
    rowexpandexpdif = staticmethod(_ops.trowexpandexpdif)

    colexpandadd = staticmethod(_ops.tcolexpandadd)
    colexpandsub = staticmethod(_ops.tcolexpandsub)
    colexpandmul = staticmethod(_ops.tcolexpandmul)
    colexpanddiv = staticmethod(_ops.tcolexpanddiv)
    colexpandmax = staticmethod(_ops.tcolexpandmax)
    colexpandmin = staticmethod(_ops.tcolexpandmin)
    colexpandexpdif = staticmethod(_ops.tcolexpandexpdif)

    sort32 = staticmethod(_ops.tsort32)
    mrgsort = staticmethod(_ops.tmrgsort)
    gather = staticmethod(_ops.tgather)
    gatherb = staticmethod(_ops.tgatherb)
    scatter = staticmethod(_ops.tscatter)

    tri = staticmethod(_ops.ttri)
    histogram = staticmethod(_ops.tthistogram)

    sel = staticmethod(_ops.tsel)
    sels = staticmethod(_ops.tsels)
    cvt = staticmethod(_ops.tcvt)
    ci = staticmethod(_ops.tci)

    bit_not = staticmethod(_ops.tnot)
    bit_and = staticmethod(_ops.tand)
    bit_ands = staticmethod(_ops.tands)
    bit_or = staticmethod(_ops.tor)
    bit_ors = staticmethod(_ops.tors)
    bit_xor = staticmethod(_ops.txor)
    bit_xors = staticmethod(_ops.txors)
    bit_shl = staticmethod(_ops.tshl)
    bit_shls = staticmethod(_ops.tshls)
    bit_shr = staticmethod(_ops.tshr)
    bit_shrs = staticmethod(_ops.tshrs)

    partadd = staticmethod(_ops.tpartadd)
    partmul = staticmethod(_ops.tpartmul)
    partmax = staticmethod(_ops.tpartmax)
    partmin = staticmethod(_ops.tpartmin)

    fillpad = staticmethod(_ops.tfillpad)


tile = _TileNamespace()
