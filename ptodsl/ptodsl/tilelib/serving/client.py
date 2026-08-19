# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Synchronous client for the PTODSL TileLib daemon."""

from __future__ import annotations

import socket

from .wire import recv_message, send_message


class DaemonError(Exception):
    """An RPC reached the daemon but the requested operation failed."""


class DaemonClient:
    """Issue one daemon RPC per Unix-socket connection."""

    def __init__(self, socket_path: str):
        self.socket_path = socket_path

    def _call(self, method: str, params: dict | None = None):
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.connect(self.socket_path)
            send_message(sock, {"method": method, "params": params or {}})
            response = recv_message(sock)

        if not response.get("success"):
            raise DaemonError(response.get("error", "unknown daemon error"))
        return response["result"]

    def ping(self):
        return self._call("ping")

    def get_metadata(
        self,
        target,
        op,
        operand_specs,
        context_attrs=None,
        include_vmi_candidates=False,
    ):
        return self._call(
            "get_metadata",
            {
                "target": target,
                "op": op,
                "operand_specs": operand_specs,
                "context_attrs": context_attrs or {},
                "include_vmi_candidates": bool(include_vmi_candidates),
            },
        )

    def instantiate(
        self,
        target,
        op,
        operand_specs,
        context_attrs=None,
        candidate_id=None,
    ):
        return self._call(
            "instantiate",
            {
                "target": target,
                "op": op,
                "operand_specs": operand_specs,
                "context_attrs": context_attrs or {},
                "candidate_id": candidate_id,
            },
        )

    def get_stats(self):
        return self._call("get_stats")

    def clear(self):
        return self._call("clear")


__all__ = ["DaemonClient", "DaemonError"]
