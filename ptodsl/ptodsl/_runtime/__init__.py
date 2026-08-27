# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Private runtime helpers for ``@pto.jit`` launch.

Launch code talks to MLIR types. Native-build helpers do not. Keep this
package lazy so a test or caller that only wants command construction does
not need ``ptoas.mlir`` bindings.
"""

from importlib import import_module

__all__ = [
    "LaunchHandle",
    "build_and_load_native_library",
    "build_native_library",
]

_EXPORTS = {
    "LaunchHandle": (".launch", "LaunchHandle"),
    "build_and_load_native_library": (".launch", "build_and_load_native_library"),
    "build_native_library": (".native_build", "build_native_library"),
}


def __getattr__(name):
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
