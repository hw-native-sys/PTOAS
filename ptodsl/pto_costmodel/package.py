# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""Model-side package integrity without compiler/MLIR dependencies."""
from pathlib import Path

from pto_costmodel.contract import SEMANTICS, envelope, identity
from pto_costmodel.wire import fields, fingerprint, read_json, require

FILES = ("program.json", "runtime_bindings.json", "target_profile.json", "canonical.pto")


def digest_text(text):
    import hashlib
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_package(path):
    from pto_costmodel.wire import read_text
    root = Path(path).resolve()
    manifest = read_json(root / "manifest.json")
    envelope(manifest, "package")
    required = ("protocol_version", "kind", "required_features", "program_fingerprint", "bindings_fingerprint",
                "target_fingerprint", "checkpoint_version", "semantics_version", "producer", "files")
    fields(manifest, required, ("extensions",))
    require(manifest["semantics_version"] == SEMANTICS and manifest["checkpoint_version"] == 1,
            "VERSION", "unsupported program/checkpoint semantics")
    fields(manifest["files"], FILES)
    for name in FILES:
        require(digest_text(read_text(root / name)) == manifest["files"][name],
                "PACKAGE_INTEGRITY", f"file digest mismatch: {name}")
    program = read_json(root / "program.json")
    bindings = read_json(root / "runtime_bindings.json")
    target = read_json(root / "target_profile.json")
    for name, content in (("program", program), ("bindings", bindings), ("target", target)):
        require(fingerprint(content) == manifest[name + "_fingerprint"], "PACKAGE_INTEGRITY", name)
    return dict(manifest=manifest, program=program, bindings=bindings, target=target, identity=identity(manifest))
