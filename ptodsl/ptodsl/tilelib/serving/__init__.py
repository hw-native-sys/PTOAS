# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Unix-socket serving layer for the PTODSL TileLib."""

from .client import DaemonClient, DaemonError


def __getattr__(name):
    # Keep daemon.py unloaded when executing it with ``python -m``.
    if name in {"TileLibDaemonServer", "metadata_request", "render_request"}:
        from .daemon import TileLibDaemonServer, metadata_request, render_request

        exports = {
            "TileLibDaemonServer": TileLibDaemonServer,
            "metadata_request": metadata_request,
            "render_request": render_request,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DaemonClient",
    "DaemonError",
    "TileLibDaemonServer",
    "metadata_request",
    "render_request",
]
