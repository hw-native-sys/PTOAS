#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import logging

from ptoas.mlir import ir
from ptoas.mlir.dialects import func
from ptoas.mlir.dialects import pto as dialect

from ptodsl import pto

LOGGER = logging.getLogger(__name__)


def main() -> None:
    with ir.Context() as context, ir.Location.unknown():
        dialect.register_dialect(context)
        module = ir.Module.create()
        with ir.InsertionPoint(module.body):
            function = func.FuncOp("vmi_value_constructors", ([], []))
        block = function.add_entry_block()
        with ir.InsertionPoint(block):
            vector = pto.vmi.vreg(64, pto.f32)
            mask = pto.vmi.mask(64)
            typed_vector = pto.unspecified_value(pto.vmi.vreg_type(64, pto.f32))
            typed_mask = pto.unspecified_value(pto.vmi.mask_type(64))
            physical_vector = pto.unspecified_value(pto.vreg_type(64, pto.f32))
            func.ReturnOp([])
        module.operation.verify()

        text = str(module)
        assert text.count("pto.unspecified_value : !pto.vmi.vreg<64xf32>") == 2, text
        assert text.count("pto.unspecified_value : !pto.vmi.mask<64xpred>") == 2, text
        assert "pto.unspecified_value : !pto.vreg<64xf32>" in text, text
        assert str(pto.vmi.vreg_type(64, pto.f32).resolve()) == "!pto.vmi.vreg<64xf32>"
        assert str(pto.vmi.mask_type(64).resolve()) == "!pto.vmi.mask<64xpred>"
        assert str(vector.type) == "!pto.vmi.vreg<64xf32>"
        assert str(mask.type) == "!pto.vmi.mask<64xpred>"
        assert str(typed_vector.type) == "!pto.vmi.vreg<64xf32>"
        assert str(typed_mask.type) == "!pto.vmi.mask<64xpred>"
        assert str(physical_vector.type) == "!pto.vreg<64xf32>"

    LOGGER.info("ptodsl_vmi_value_constructors: PASS")


if __name__ == "__main__":
    main()
