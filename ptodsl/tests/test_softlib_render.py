#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import json

from ptodsl._context import make_context
from ptodsl.softlib._compiler_runtime import materialize


def test_i32_vdiv_softop_materializes_from_dsl():
    context = make_context()
    module, entry = materialize(
        "a5",
        "pto.vdiv",
        json.dumps({"dtype": "i32", "lanes": 64, "mask": "b32"}),
        context,
    )
    text = str(module)

    assert entry == "div_i32_soft"
    assert "pto.softlib.instance" in text
    assert "pto.vbitcast" in text
    assert "pto.vdiv" in text
    assert "pto.vcvt" in text
    assert "pto.vmull" in text
    assert "pto.vshr" not in text
    assert "!pto.vreg<64xi32>" in text


def test_signed_i32_softop_preserves_signed_vreg_abi():
    context = make_context()
    module, _ = materialize(
        "a5",
        "pto.vdiv",
        json.dumps({"dtype": "si32", "lanes": 64, "mask": "b32"}),
        context,
    )
    text = str(module)
    assert "!pto.vreg<64xsi32>" in text
    assert "-> !pto.vreg<64xsi32>" in text


def test_i32_softop_materializes_all_integer_corner_constants():
    context = make_context()
    module, _ = materialize(
        "a5",
        "pto.vdiv",
        json.dumps({"dtype": "i32", "lanes": 64, "mask": "b32"}),
        context,
    )
    text = str(module)
    assert "-2147483648" in text
    assert "pto.vbitcast" in text
