# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""The downstream capability-probe contract for runtime mad flags.

Downstream templates (e.g. the tilelang PTO GEMM template) detect whether
the installed ptodsl accepts unit_flag / init / bias_init / disable_gemv as
runtime mad operands through the package-level switch
``ptodsl.MAD_RUNTIME_FLAGS``. A signature-based probe is not a usable
contract: the mad entry points take **kwargs, and inspect.signature does
not expand them.
"""

import unittest

import ptodsl


class MadRuntimeFlagsContractTest(unittest.TestCase):
    def test_capability_switch_is_exported_and_true(self):
        self.assertTrue(getattr(ptodsl, "MAD_RUNTIME_FLAGS", False))


if __name__ == "__main__":
    unittest.main()
