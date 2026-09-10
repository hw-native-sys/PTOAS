# coding=utf-8
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

_ACTIVE_RUNTIME_STACK = []
_ACTIVE_SESSION_STACK = []


@contextmanager
def activate_runtime(runtime):
    """Push *runtime* as the current active tracing runtime."""
    _ACTIVE_RUNTIME_STACK.append(runtime)
    try:
        yield runtime
    finally:
        popped = _ACTIVE_RUNTIME_STACK.pop()
        if popped is not runtime:
            raise RuntimeError("PTODSL active tracing runtime stack corruption detected")


@contextmanager
def activate_session(session):
    """Push *session* as the current active trace session."""
    _ACTIVE_SESSION_STACK.append(session)
    try:
        yield session
    finally:
        popped = _ACTIVE_SESSION_STACK.pop()
        if popped is not session:
            raise RuntimeError("PTODSL active trace-session stack corruption detected")


def current_runtime(expected_type=None):
    """Return the current active tracing runtime, or ``None`` if inactive."""
    if not _ACTIVE_RUNTIME_STACK:
        return None
    runtime = _ACTIVE_RUNTIME_STACK[-1]
    if expected_type is not None and not isinstance(runtime, expected_type):
        return None
    return runtime


def current_session():
    """Return the current active trace session, or ``None`` if inactive."""
    if not _ACTIVE_SESSION_STACK:
        return None
    return _ACTIVE_SESSION_STACK[-1]


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
