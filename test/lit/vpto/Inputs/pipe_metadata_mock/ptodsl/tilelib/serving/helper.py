# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""One-shot client for the test-only PipeSpec metadata daemon."""

import argparse
import json
import socket
import sys


def _recv_exactly(sock, length):
    chunks = []
    while length:
        chunk = sock.recv(length)
        if not chunk:
            raise ConnectionError("socket closed mid-message")
        chunks.append(chunk)
        length -= len(chunk)
    return b"".join(chunks)


def _recv_message(sock):
    length = int.from_bytes(_recv_exactly(sock, 4), byteorder="big")
    return json.loads(_recv_exactly(sock, length).decode("utf-8"))


def _send_message(sock, payload):
    encoded = json.dumps(payload).encode("utf-8")
    sock.sendall(len(encoded).to_bytes(4, byteorder="big") + encoded)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--op", required=True)
    parser.add_argument("--operand-specs", required=True)
    parser.add_argument("--context-attrs", default=None)
    parser.add_argument("--method", required=True)
    parser.add_argument("--candidate-id", default=None)
    args = parser.parse_args(argv)
    if args.method != "get_metadata":
        raise SystemExit("only get_metadata is supported by the test daemon")

    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.connect(args.socket)
        _send_message(
            client,
            {
                "method": args.method,
                "params": {
                    "target": args.target,
                    "op": args.op,
                    "operand_specs": json.loads(args.operand_specs),
                    "context_attrs": json.loads(args.context_attrs)
                    if args.context_attrs
                    else {},
                },
            },
        )
        response = _recv_message(client)
    if not response.get("success"):
        raise SystemExit(response.get("error", "metadata request failed"))
    sys.stdout.write(json.dumps(response["result"]))


if __name__ == "__main__":
    main()
