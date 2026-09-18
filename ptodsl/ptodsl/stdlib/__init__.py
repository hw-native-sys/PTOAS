# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""PTODSL first-party standard library.

This package ships reusable source-level helpers with PTODSL itself so that
kernels can call them through the normal ``pto.*`` surface without any
additional import, configuration, or registration.

The public export catalog lives in ``_exports``.  ``ptodsl.pto`` resolves
catalog entries lazily on attribute access, so importing ``ptodsl`` never
eagerly loads the implementation modules.
"""

__all__ = ["_exports"]

from . import _exports
