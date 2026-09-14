# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""ptodsl – PTO MLIR DSL package."""

__all__ = ["pto"]

from importlib import import_module

# Capability switch. Downstream templates probe this instead of
# introspecting function signatures: a **kwargs signature hides the named
# keyword arguments from inspect.signature, so a signature-based probe
# would always be False. This switch is set when mad/mad_acc/mad_bias
# accept unit_flag / init / bias_init / disable_gemv as runtime operands
# packed into the mad xt immediate (PTOAS issue #1279).
MAD_RUNTIME_FLAGS = True


def __getattr__(name):
    if name == "pto":
        module = import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
