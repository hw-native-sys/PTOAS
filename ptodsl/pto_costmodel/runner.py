# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""Bounded subprocess adapter transport; argv is user configuration, never model output."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from pto_costmodel.contract import envelope
from pto_costmodel.wire import MAX_BYTES, fields, read_json, require


def invoke(adapter, action, package=None, request=None):
    fields(adapter, ("argv", "cwd"), ("environment",))
    argv = adapter["argv"]
    require(isinstance(argv, list) and argv and all(isinstance(x, str) and x for x in argv),
            "ADAPTER_CONFIG", "argv must be nonempty strings")
    executable = shutil.which(argv[0])
    require(executable is not None, "ADAPTER_CONFIG", "adapter executable not found")
    directory = Path(adapter["cwd"]).resolve()
    require(directory.is_dir(), "ADAPTER_CONFIG", "adapter working directory missing")
    environment = dict(os.environ)
    supplied = adapter.get("environment", {})
    require(isinstance(supplied, dict) and all(isinstance(k, str) and isinstance(v, str)
                                            for k, v in supplied.items()), "ADAPTER_CONFIG", "invalid environment")
    environment.update(supplied)
    command = [executable] + argv[1:] + [action]
    if package is not None:
        command += ["--package", str(Path(package).resolve()), "--request", str(Path(request).resolve())]
    with tempfile.TemporaryDirectory(prefix="pto-model-transport-") as tmp:
        output, error = Path(tmp) / "output.json", Path(tmp) / "stderr.txt"
        with output.open("wb") as stdout, error.open("wb") as stderr:
            result = subprocess.run(command, cwd=directory, env=environment, stdout=stdout, stderr=stderr,
                                    timeout=120, check=False)
        require(output.stat().st_size <= MAX_BYTES and error.stat().st_size <= MAX_BYTES,
                "ADAPTER_OUTPUT", "adapter output exceeds size limit")
        require(result.returncode == 0, "ADAPTER_FAILURE", error.read_text(encoding="utf-8"))
        return read_json(output)


def negotiate(adapter):
    result = invoke(adapter, "capabilities")
    envelope(result, "capabilities")
    require("a5" in result.get("targets", []), "TARGET", "adapter does not support A5")
    require("pto.static_cv.1" in result.get("semantics_versions", [])
            and "prefix_suffix_v1" in result.get("schedule_kinds", []), "UNSUPPORTED_FEATURE",
            "adapter does not support requested operation/schedule semantics")
    fields(result.get("model"), ("name", "revision", "adapter_version", "config_fingerprint"))
    return result
