# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Shared tracing runtime building blocks for PTODSL frontends.

This package intentionally avoids eager cross-imports so that low-level users
such as ``ptodsl._control_flow`` can import ``ptodsl._tracing.active`` without
triggering the heavier ``runtime/session`` stack during package initialization.
"""

from importlib import import_module

__all__ = [
    "activate_runtime",
    "activate_session",
    "current_runtime",
    "current_session",
    "require_active_runtime",
    "require_active_session",
    "ModuleArtifact",
    "KernelModuleSpec",
    "ModuleStyle",
    "create_kernel_module",
    "CallbackTracingRuntime",
    "SignatureTracingRuntime",
    "TracingRuntime",
    "HelperFunctionSpec",
    "SubkernelTraceFrame",
    "TraceSession",
]

_EXPORTS = {
    "activate_runtime": (".active", "activate_runtime"),
    "activate_session": (".active", "activate_session"),
    "current_runtime": (".active", "current_runtime"),
    "current_session": (".active", "current_session"),
    "require_active_runtime": (".active", "require_active_runtime"),
    "require_active_session": (".active", "require_active_session"),
    "ModuleArtifact": (".artifacts", "ModuleArtifact"),
    "KernelModuleSpec": (".module_builder", "KernelModuleSpec"),
    "ModuleStyle": (".module_builder", "ModuleStyle"),
    "create_kernel_module": (".module_builder", "create_kernel_module"),
    "CallbackTracingRuntime": (".runtime", "CallbackTracingRuntime"),
    "SignatureTracingRuntime": (".runtime", "SignatureTracingRuntime"),
    "TracingRuntime": (".runtime", "TracingRuntime"),
    "HelperFunctionSpec": (".session", "HelperFunctionSpec"),
    "SubkernelTraceFrame": (".session", "SubkernelTraceFrame"),
    "TraceSession": (".session", "TraceSession"),
}


def __getattr__(name):
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
