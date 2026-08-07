#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Mock TileLib daemon for testing SimtLaunchOp callee renaming.

Returns pre-canned MLIR containing a template function with pto.simt_launch
and a helper function, so the ExpandTileOp pass can be tested without the
real PTODSL template package.
"""

import json
import os
import socket
import struct
import sys

CANNED_MLIR = r"""
func.func @template_entry(%a: !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64, v_row=16, v_col=64, blayout=row_major, slayout=none_box, fractal=512, pad=0>, %b: !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64, v_row=16, v_col=64, blayout=row_major, slayout=none_box, fractal=512, pad=0>, %dst: !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=64, v_row=16, v_col=64, blayout=row_major, slayout=none_box, fractal=512, pad=0>) attributes {pto.tilelang.instance = "mock"} {
  %c1 = arith.constant 1 : i32
  pto.simt_launch @simt_body<<<%c1, %c1, %c1>>>() : () -> ()
  return
}
func.func @simt_body() attributes {pto.simt_entry} {
  return
}
"""


def recv_exactly(sock, n):
    chunks = []
    while n:
        chunk = sock.recv(n)
        if not chunk:
            raise ConnectionError("closed")
        chunks.append(chunk)
        n -= len(chunk)
    return b"".join(chunks)


def recv_msg(sock):
    length = struct.unpack(">I", recv_exactly(sock, 4))[0]
    return json.loads(recv_exactly(sock, length).decode("utf-8"))


def send_msg(sock, msg):
    payload = json.dumps(msg).encode("utf-8")
    sock.sendall(struct.pack(">I", len(payload)))
    sock.sendall(payload)


def main():
    sock_path = sys.argv[1]
    if os.path.exists(sock_path):
        os.unlink(sock_path)
    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(sock_path)
    srv.listen(8)
    try:
        while True:
            conn, _ = srv.accept()
            try:
                req = recv_msg(conn)
                if req.get("method") == "instantiate":
                    send_msg(conn, {"success": True, "result": CANNED_MLIR})
                else:
                    send_msg(conn, {"success": False, "error": "unsupported"})
            except Exception:
                pass
            finally:
                conn.close()
    except (KeyboardInterrupt, OSError):
        pass
    finally:
        srv.close()
        if os.path.exists(sock_path):
            os.unlink(sock_path)


if __name__ == "__main__":
    main()
