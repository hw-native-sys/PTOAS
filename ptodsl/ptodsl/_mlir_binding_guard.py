# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Keep the build-tree MLIR Python bindings ahead of editable installs.

When a developer has an editable scikit-build install of PTOAS, its import
finder can redirect ptoas.mlir to a previously generated bindings tree that
misses newly added ops. Importing this private module drops those finders, so
callers that must observe the build tree import it before ptoas.mlir.
"""

from __future__ import annotations

import sys


def prefer_build_tree_mlir_bindings() -> None:
    """Drop editable-install import finders from sys.meta_path."""
    sys.meta_path[:] = [
        finder for finder in sys.meta_path if "editable" not in repr(finder).lower()
    ]


prefer_build_tree_mlir_bindings()
