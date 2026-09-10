# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Python package for the PTOAS command-line interface."""

from ._loader import ensure_core, install_online_finder, preload_shared_libs

# Preload the shipped, Python-independent DSOs and install the meta path finder
# that serves the four version-sensitive native extensions (``_core`` plus the
# ``ptoas.mlir`` pybind11 family). This must run before any ``import
# ptoas.mlir.*`` so a mismatching interpreter transparently online-compiles them.
preload_shared_libs()
install_online_finder()

# Acquire the native ``_core`` extension eagerly: the prebuilt binary on a
# matching interpreter, or an online-compiled build otherwise. This registers
# ``sys.modules['ptoas._core']`` so ``from ptoas import _core`` works uniformly.
ensure_core()
