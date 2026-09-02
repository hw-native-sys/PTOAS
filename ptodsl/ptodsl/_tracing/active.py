# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Active tracing-runtime stack shared by PTODSL frontends."""

from __future__ import annotations

from contextlib import contextmanager

import threading

# Per-thread stacks: MLIR pass pipelines may run ExpandTileOp materialization
# concurrently on multiple threads; a process-global stack would interleave
# unrelated runtimes and report spurious corruption.
_LOCAL = threading.local()


def _runtime_stack():
    stack = getattr(_LOCAL, "runtime_stack", None)
    if stack is None:
        stack = []
        _LOCAL.runtime_stack = stack
    return stack


def _session_stack():
    stack = getattr(_LOCAL, "session_stack", None)
    if stack is None:
        stack = []
        _LOCAL.session_stack = stack
    return stack


@contextmanager
def activate_runtime(runtime):
    """Push *runtime* as the current active tracing runtime."""
    stack = _runtime_stack()
    stack.append(runtime)
    try:
        yield runtime
    finally:
        popped = stack.pop()
        if popped is not runtime:
            raise RuntimeError("PTODSL active tracing runtime stack corruption detected")


@contextmanager
def activate_session(session):
    """Push *session* as the current active trace session."""
    stack = _session_stack()
    stack.append(session)
    try:
        yield session
    finally:
        popped = stack.pop()
        if popped is not session:
            raise RuntimeError("PTODSL active trace-session stack corruption detected")


def current_runtime(expected_type=None):
    """Return the current active tracing runtime, or ``None`` if inactive."""
    stack = _runtime_stack()
    if not stack:
        return None
    runtime = stack[-1]
    if expected_type is not None and not isinstance(runtime, expected_type):
        return None
    return runtime


def current_session():
    """Return the current active trace session, or ``None`` if inactive."""
    stack = _session_stack()
    if not stack:
        return None
    return stack[-1]


def require_active_runtime(surface: str, expected_type=None):
    """Return the active runtime or raise a surface-specific error."""
    runtime = current_runtime(expected_type=expected_type)
    if runtime is None:
        raise RuntimeError(
            f"{surface}() may only be used while tracing a compatible PTODSL kernel"
        )
    return runtime


def require_active_session(surface: str):
    """Return the active trace session or raise a surface-specific error."""
    session = current_session()
    if session is None:
        raise RuntimeError(
            f"{surface}() may only be used while tracing a compatible PTODSL kernel"
        )
    return session


__all__ = [
    "activate_runtime",
    "activate_session",
    "current_runtime",
    "current_session",
    "require_active_runtime",
    "require_active_session",
]
