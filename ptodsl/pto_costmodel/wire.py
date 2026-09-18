# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""Versioned JSON and filesystem boundary for the CV planning exchange."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile

MAX_BYTES = 16 * 1024 * 1024
MAX_INTEGER = (1 << 31) - 1


class ContractError(ValueError):
    """An unsupported input or invalid external plan, with a stable category."""

    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def require(condition, code, message):
    if not condition:
        raise ContractError(code, message)


def fields(value, required, optional=()):
    require(isinstance(value, dict), "SCHEMA", "expected an object")
    missing = set(required) - set(value)
    unknown = set(value) - set(required) - set(optional)
    require(not missing, "SCHEMA", f"missing required fields: {sorted(missing)}")
    require(not unknown, "SCHEMA", f"unknown fields: {sorted(map(str, unknown))}")


def integer(value, label, minimum=0):
    require(type(value) is int and minimum <= value <= MAX_INTEGER,
            "RANGE", f"{label} must be an integer in [{minimum}, {MAX_INTEGER}]")
    return value


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, "SCHEMA", f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_text(path):
    with Path(path).open("rb") as stream:
        data = stream.read(MAX_BYTES + 1)
    require(len(data) <= MAX_BYTES, "RANGE", "input exceeds 16 MiB")
    return data.decode("utf-8")


def read_json(path):
    def invalid_constant(value):
        raise ContractError("SCHEMA", f"invalid JSON constant: {value}")
    try:
        return json.loads(read_text(path), object_pairs_hook=unique_object,
                          parse_constant=invalid_constant)
    except (json.JSONDecodeError, RecursionError) as exc:
        raise ContractError("SCHEMA", "invalid JSON document") from exc


def encode(value):
    return json.dumps(value, ensure_ascii=True, sort_keys=True, indent=2, allow_nan=False) + "\n"


def fingerprint(value):
    return hashlib.sha256(encode(value).encode("utf-8")).hexdigest()


def publish(directory, files):
    """Publish a complete fresh package; never overwrite an existing result."""
    destination = Path(directory).resolve()
    require(not destination.exists(), "OUTPUT_EXISTS", str(destination))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".cv-package-", dir=destination.parent) as tmp:
        root = Path(tmp) / "package"
        root.mkdir()
        for name, content in files.items():
            require(len(content.encode("utf-8")) <= MAX_BYTES, "RANGE", "output exceeds 16 MiB")
            (root / name).write_text(content, encoding="utf-8")
        require(not destination.exists(), "OUTPUT_EXISTS", str(destination))
        os.rename(root, destination)
