# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Length-prefixed JSON framing for the TileLib daemon RPC."""

from __future__ import annotations

import json


MAX_MESSAGE_SIZE = 64 * 1024 * 1024


def recv_exactly(sock, length: int) -> bytes:
    """Read exactly ``length`` bytes or fail if the peer closes early."""
    chunks = []
    remaining = length
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise ConnectionError("socket closed mid-message")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def send_message(sock, message: dict) -> None:
    """Send one UTF-8 JSON message with a 4-byte big-endian length prefix."""
    payload = json.dumps(message).encode("utf-8")
    if len(payload) > MAX_MESSAGE_SIZE:
        raise ValueError(
            f"message length {len(payload)} exceeds limit {MAX_MESSAGE_SIZE}"
        )
    sock.sendall(len(payload).to_bytes(4, byteorder="big"))
    sock.sendall(payload)


def recv_message(sock) -> dict:
    """Receive one length-prefixed UTF-8 JSON message."""
    length = int.from_bytes(recv_exactly(sock, 4), byteorder="big")
    if length > MAX_MESSAGE_SIZE:
        raise ValueError(
            f"message length {length} exceeds limit {MAX_MESSAGE_SIZE}"
        )
    return json.loads(recv_exactly(sock, length).decode("utf-8"))


__all__ = [
    "MAX_MESSAGE_SIZE",
    "recv_exactly",
    "recv_message",
    "send_message",
]
