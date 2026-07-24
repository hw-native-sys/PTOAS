# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Test-only daemon for PipeSpec metadata and compiler expansion ABI."""

import argparse
import json
import os
import signal
import socket


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


def _expect_pipe_abi(params, request_counts):
    target = params.get("target")
    op = params.get("op")
    operands = params.get("operand_specs")
    if target != "a5" or not isinstance(operands, list):
        raise ValueError("expected an A5 pipe metadata request")

    pipe_index = 0 if op == "pto.tdrain" else 1
    if len(operands) <= pipe_index + 2:
        raise ValueError(f"incomplete PipeSpec operands for {op}")

    pipe = operands[pipe_index]
    if pipe.get("init_kind") == "l2l":
        _expect_local_pipe_abi(op, operands, pipe_index, pipe, request_counts)
        return

    expected_splits = {
        "pto.talloc": (1, 0),
        "pto.tpush": (1, 2),
        "pto.tpop": (1, 0),
        "pto.tfree": (1, 0),
        "pto.tdrain": (1,),
    }
    if op not in expected_splits:
        raise ValueError(f"unexpected metadata operation: {op!r}")

    request_key = ("l2g2l", op)
    request_index = request_counts.get(request_key, 0)
    request_counts[request_key] = request_index + 1
    allowed_splits = expected_splits[op]
    if request_index >= len(allowed_splits):
        raise ValueError(f"unexpected duplicate metadata request for {op}")

    if op != "pto.tdrain" and operands[0].get("kind") != "pipe_entry":
        raise ValueError(f"{op} must serialize its entry as pipe_entry")

    is_no_split_pipe = (
        op in ("pto.talloc", "pto.tpop", "pto.tfree") and request_index == 1
    )
    expected_pipe = {
        "kind": "pipe",
        "init_kind": "l2g2l",
        "dir_mask": 1,
        "slot_size": 1024,
        "slot_num": 8,
        "local_slot_num": None,
        "flag_base": 4,
        "nosplit": is_no_split_pipe,
        "split": allowed_splits[request_index],
        "resource_names": ["gm_addr"],
    }
    if pipe != expected_pipe:
        raise ValueError(f"unexpected PipeSpec for {op}: {pipe!r}")

    resources = operands[pipe_index + 1]
    if resources.get("kind") != "pipe_resources" or resources.get("names") != [
        "gm_addr"
    ] or len(resources.get("values", ())) != 1:
        raise ValueError(f"unexpected PipeResources for {op}: {resources!r}")
    if operands[pipe_index + 2] != {
        "kind": "pipe_state",
        "fields": ["i32", "i32"],
    }:
        raise ValueError(f"missing PipeState for {op}")

    if op in ("pto.tpush", "pto.tpop"):
        subblock = operands[pipe_index + 3]
        if subblock != {"kind": "scalar", "dtype": "i64", "value": 0}:
            raise ValueError(f"unexpected default subblock for {op}: {subblock!r}")


def _expect_local_pipe_abi(op, operands, pipe_index, pipe, request_counts):
    if op not in ("pto.tpush", "pto.tpop", "pto.tdrain"):
        raise ValueError(f"unexpected local pipe operation: {op!r}")

    request_key = ("l2l", op)
    if request_counts.get(request_key, 0):
        raise ValueError(f"unexpected duplicate local metadata request for {op}")
    request_counts[request_key] = 1

    if op != "pto.tdrain" and operands[0].get("kind") != "tile":
        raise ValueError(f"{op} must serialize its local entry as tile")

    expected_pipe = {
        "kind": "pipe",
        "init_kind": "l2l",
        "dir_mask": 3,
        "slot_size": 512,
        "slot_num": 4,
        "local_slot_num": None,
        "flag_base": 8,
        "nosplit": False,
        "split": 1,
        "resource_names": ["local_addr", "peer_local_addr"],
    }
    if pipe != expected_pipe:
        raise ValueError(f"unexpected local PipeSpec for {op}: {pipe!r}")

    resources = operands[pipe_index + 1]
    expected_resources = {
        "kind": "pipe_resources",
        "names": ["local_addr", "peer_local_addr"],
        "values": [
            {"kind": "scalar", "dtype": "i32", "value": 17},
            {"kind": "scalar", "dtype": "i32", "value": 29},
        ],
    }
    if resources != expected_resources:
        raise ValueError(f"unexpected local PipeResources for {op}: {resources!r}")
    if operands[pipe_index + 2] != {
        "kind": "pipe_state",
        "fields": ["i32", "i32"],
    }:
        raise ValueError(f"missing local PipeState for {op}")

    if op in ("pto.tpush", "pto.tpop"):
        subblock = operands[pipe_index + 3]
        if subblock != {"kind": "scalar", "dtype": "i64", "value": 7}:
            raise ValueError(f"unexpected explicit subblock for {op}: {subblock!r}")


def _metadata_response():
    return {
        "candidates": {
            "mock_pipe_template": {
                "id": 42,
                "name": "mock_pipe_template",
                "loop_depth": 0,
                "is_post_update": False,
                "has_tail": False,
            }
        }
    }


def _mlir_type_for_spec(spec):
    kind = spec.get("kind")
    if kind in ("pipe_entry", "view"):
        shape = spec.get("shape")
        dtype = spec.get("dtype")
        if not isinstance(shape, list) or not dtype:
            raise ValueError(f"incomplete view spec: {spec!r}")
        dims = "x".join("?" if dim == -1 else str(dim) for dim in shape)
        return f"!pto.tensor_view<{dims}x{dtype}>"
    if kind == "pipe_state":
        return "!pto.struct<i32, i32>"
    if kind == "scalar" and spec.get("dtype"):
        return spec["dtype"]
    raise ValueError(f"unsupported mock helper operand: {spec!r}")


def _instantiate_response(params):
    if params.get("candidate_id") != "mock_pipe_template":
        raise ValueError(f"unexpected candidate: {params.get('candidate_id')!r}")
    operands = params.get("operand_specs")
    if params.get("target") != "a5" or not isinstance(operands, list):
        raise ValueError("expected an A5 pipe instantiate request")

    argument_types = []
    for operand in operands:
        kind = operand.get("kind")
        if kind == "pipe":
            continue
        if kind == "pipe_resources":
            argument_types.extend(_mlir_type_for_spec(value)
                                  for value in operand.get("values", ()))
            continue
        argument_types.append(_mlir_type_for_spec(operand))

    arguments = ", ".join(
        f"%arg{index}: {type_name}"
        for index, type_name in enumerate(argument_types)
    )
    return (
        'module attributes {pto.target_arch = "a5"} {\n'
        f"  func.func @mock_pipe_template({arguments}) "
        "attributes {pto.tilelang.instance} {\n"
        "    return\n"
        "  }\n"
        "}\n"
    )


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", required=True)
    args = parser.parse_args(argv)

    try:
        os.unlink(args.socket)
    except FileNotFoundError:
        pass

    stopping = False

    def stop(*_):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    request_counts = {}
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
        server.bind(args.socket)
        server.listen()
        server.settimeout(0.1)
        try:
            while not stopping:
                try:
                    connection, _ = server.accept()
                except socket.timeout:
                    continue
                with connection:
                    try:
                        request = _recv_message(connection)
                        method = request.get("method")
                        params = request.get("params", {})
                        if method == "get_metadata":
                            _expect_pipe_abi(params, request_counts)
                            result = _metadata_response()
                        elif method == "instantiate":
                            result = _instantiate_response(params)
                        else:
                            raise ValueError(f"unsupported method: {method!r}")
                        _send_message(
                            connection,
                            {"success": True, "result": result},
                        )
                    except Exception as error:
                        _send_message(
                            connection,
                            {"success": False, "error": f"{type(error).__name__}: {error}"},
                        )
        finally:
            try:
                os.unlink(args.socket)
            except FileNotFoundError:
                pass


if __name__ == "__main__":
    main()
