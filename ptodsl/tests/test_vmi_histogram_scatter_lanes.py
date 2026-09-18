#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


from ptoas.mlir import ir
from ptoas.mlir.dialects import func, pto as dialect
from ptodsl import pto

LEGAL = {1, 2, 4, 8, 64, 128, 256}
LANES = [1, 2, 3, 4, 8, 16, 32, 64, 96, 128, 192, 256, 512]
KINDS = ['vreg', 'mask', 'create_mask', 'vload', 'vbrc', 'vci',
         'vdhist', 'vchist', 'vscatter']


def probe_argument_types(kind, n, mask_n, offsets_n):
    if kind in ('vload', 'create_mask'):
        return [ir.Type.parse('!pto.ptr<ui8, ub>')]
    if kind == 'vbrc':
        return [ir.F32Type.get()]
    if kind == 'vci':
        return [ir.IntegerType.get_signless(32)]
    if kind in ('vdhist', 'vchist'):
        u8, u16 = ir.IntegerType.get_unsigned(8), ir.IntegerType.get_unsigned(16)
        return [dialect.VMIVRegType.get(128, u16),
                dialect.VMIVRegType.get(n, u8),
                dialect.VMIMaskType.get(mask_n, 'pred')]
    f32, i32 = ir.F32Type.get(), ir.IntegerType.get_signless(32)
    return [dialect.VMIVRegType.get(n, f32),
            ir.Type.parse('!pto.ptr<f32, ub>'),
            dialect.VMIVRegType.get(offsets_n, i32),
            dialect.VMIMaskType.get(mask_n, 'pred')]


def emit_probe(kind, args, n, active):
    if kind == 'vload':
        return pto.vmi.vload(args[0], 0, size=n)
    if kind == 'vbrc':
        return pto.vmi.vbrc(args[0], size=n)
    if kind == 'vci':
        return pto.vmi.vci(args[0], size=n)
    if kind == 'create_mask':
        return pto.vmi.create_mask(n if active is None else active, size=n)
    return getattr(pto.vmi, kind)(*args)


def build(kind, n, active=None, mask_n=None, offsets_n=None):
    mask_n = n if mask_n is None else mask_n
    offsets_n = n if offsets_n is None else offsets_n
    if kind == 'vreg':
        return str(pto.vmi.vreg(n, pto.f32)), ''
    if kind == 'mask':
        return str(pto.vmi.mask(n)), ''
    with ir.Context() as ctx, ir.Location.unknown():
        dialect.register_dialect(ctx)
        types = probe_argument_types(kind, n, mask_n, offsets_n)
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            fn = func.FuncOp('probe', (types, []))
        block = fn.add_entry_block()
        with ir.InsertionPoint(block):
            result = emit_probe(kind, block.arguments, n, active)
            func.ReturnOp([])
        module.operation.verify()
        return str(getattr(result, 'type', None)), str(module)



def main():
    for kind in KINDS:
        for lanes in LANES:
            if lanes in LEGAL:
                build(kind, lanes)
                continue
            try:
                build(kind, lanes)
            except ValueError as exc:
                assert "1, 2, 4, 8, 64, 128, 256" in str(exc), str(exc)
            else:
                raise AssertionError(f"{kind} accepted illegal lanes={lanes}")

    for kind in ("vdhist", "vchist", "vscatter"):
        for mask_lanes in (96, 64):
            try:
                build(kind, 128, mask_n=mask_lanes)
            except ValueError as exc:
                assert "mask" in str(exc), str(exc)
            else:
                raise AssertionError(f"{kind} accepted mask lanes={mask_lanes}")
    for offsets in (96, 64):
        try:
            build("vscatter", 128, offsets_n=offsets)
        except ValueError as exc:
            assert "offsets" in str(exc), str(exc)
        else:
            raise AssertionError(f"vscatter accepted offsets lanes={offsets}")

    result_type, text = build("create_mask", 128, active=96)
    assert result_type == "!pto.vmi.mask<128xpred>", result_type
    # The unified scalar surface authors the active prefix through pto.constant.
    assert "pto.constant 96" in text


if __name__ == "__main__":
    main()
