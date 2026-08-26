#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""ctypes access to the AsyncWorkspace shim, for engine-path ST cases.

This module deliberately contains no session field indices, slot widths, SQE
constants or channel-record offsets. Every one of those stays in
``include/PTO/Support/AsyncSessionABI.h`` and is reached through
``AsyncWorkspaceShim.cpp``, so this file cannot drift from the ABI: it asks the
shim how large a session template is and hands it named values to fill one.
"""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np

_DEFAULT_LIB_RELPATH = Path("build") / "libpto_async_shim.so"
_LIB_PATH_ENV = "PTO_ASYNC_SHIM"

# The A2/A3 doorbell stages one 32-bit word in UB before an MTE store carries it
# out. Only the address reaches the expansion today, but reserving a whole
# 32-byte block keeps the staging clear of anything a kernel puts next to it.
DOORBELL_STAGE_BYTES = 32


class AsyncWorkspaceError(RuntimeError):
    """A shim call failed, or the shim library could not be loaded."""


def _shim_library_path(explicit: str | os.PathLike[str] | None = None) -> Path:
    """Resolve the shim library path from an argument, the environment, or the default."""
    raw = explicit if explicit is not None else os.environ.get(_LIB_PATH_ENV)
    if raw:
        candidate = Path(raw).expanduser().resolve()
        source = f"{_LIB_PATH_ENV}={raw}" if explicit is None else f"library_path={raw}"
    else:
        candidate = (Path(__file__).resolve().parent / _DEFAULT_LIB_RELPATH).resolve()
        source = "default location"
    if not candidate.is_file():
        raise AsyncWorkspaceError(
            f"AsyncWorkspace shim not found at {candidate} ({source}); "
            "build it with test/comm/build_async_shim.sh"
        )
    return candidate


def _bind_signatures(lib: ctypes.CDLL) -> None:
    """Declare every shim entry point, so ctypes never guesses a width."""
    lib.pto_async_ws_init.argtypes = [ctypes.c_uint]
    lib.pto_async_ws_init.restype = ctypes.c_int

    lib.pto_async_ws_finalize.argtypes = []
    lib.pto_async_ws_finalize.restype = None

    lib.pto_async_ws_error.argtypes = []
    lib.pto_async_ws_error.restype = ctypes.c_char_p

    lib.pto_async_ws_context_gm.argtypes = []
    lib.pto_async_ws_context_gm.restype = ctypes.c_uint64

    lib.pto_async_ws_channel_count.argtypes = []
    lib.pto_async_ws_channel_count.restype = ctypes.c_uint

    lib.pto_async_ws_reported_queue_num.argtypes = []
    lib.pto_async_ws_reported_queue_num.restype = ctypes.c_uint32

    lib.pto_async_ws_sq_tail.argtypes = [ctypes.c_uint, ctypes.c_uint]
    lib.pto_async_ws_sq_tail.restype = ctypes.c_longlong

    for getter in (
        lib.pto_async_session_bytes,
        lib.pto_async_session_qos_default,
        lib.pto_async_ws_min_transfer_bytes,
        lib.pto_async_ws_max_channels,
    ):
        getter.argtypes = []
        getter.restype = ctypes.c_uint32

    lib.pto_async_session_pack.argtypes = [
        ctypes.c_void_p,  # out
        ctypes.c_uint32,  # out_bytes
        ctypes.c_uint32,  # channel_idx
        ctypes.c_uint32,  # channel_num
        ctypes.c_uint64,  # block_bytes
        ctypes.c_uint64,  # tmp_buf_addr
        ctypes.c_uint32,  # tmp_buf_size
        ctypes.c_uint32,  # sync_id
        ctypes.c_uint64,  # comm_block_offset
        ctypes.c_uint32,  # qos
        ctypes.c_uint32,  # dest_rank_id
    ]
    lib.pto_async_session_pack.restype = ctypes.c_int


class AsyncWorkspace:
    """One async workspace on the device the caller has already selected.

    The device must be selected before ``init``. Under the PTODSL ST harness
    that is torch_npu's doing, and whether a workspace can attach to a context
    someone else created is the assumption these cases exist to check.

    The shim can arrive two ways. Pass ``library`` to use one already loaded,
    which is how a kernel that listed the shim in
    ``@pto.jit(native_options={"host_sources": ...})`` reaches it: the shim is
    inside the kernel's own library, so there is one build and one artifact.
    Pass ``library_path``, or neither, to load a standalone build from
    ``build_async_shim.sh`` instead, which needs no kernel and so is also how the
    shim gets exercised on a machine with no card.
    """

    def __init__(
        self,
        library_path: str | os.PathLike[str] | None = None,
        *,
        library: ctypes.CDLL | None = None,
    ) -> None:
        if library is not None and library_path is not None:
            raise AsyncWorkspaceError("pass either library or library_path, not both")
        if library is not None:
            self._library_path: Path | None = None
            self._lib: ctypes.CDLL = library
        else:
            self._library_path = _shim_library_path(library_path)
            self._lib = ctypes.CDLL(str(self._library_path))
        self._channels: int = 0
        self._inited: bool = False
        _bind_signatures(self._lib)

    # --- lifecycle -----------------------------------------------------------

    def init(self, channels: int = 1) -> None:
        if channels <= 0:
            raise AsyncWorkspaceError(f"channel count must be positive, got {channels}")
        limit = self.max_channels
        if channels > limit:
            raise AsyncWorkspaceError(f"channel count {channels} exceeds the shim limit {limit}")
        if self._lib.pto_async_ws_init(ctypes.c_uint(channels)) != 0:
            raise AsyncWorkspaceError(f"AsyncWorkspace init failed: {self._error()}")
        self._channels = channels
        self._inited = True

    def finalize(self) -> None:
        if not self._inited:
            return
        self._lib.pto_async_ws_finalize()
        self._channels = 0
        self._inited = False

    def _error(self) -> str:
        raw = self._lib.pto_async_ws_error()
        if raw is None:
            return "no error reported"
        return raw.decode("utf-8", errors="replace")

    def _require_inited(self) -> None:
        if not self._inited:
            raise AsyncWorkspaceError("workspace is not initialized; call init() first")

    # --- properties ----------------------------------------------------------

    @property
    def library_path(self) -> Path | None:
        """Where the shim was loaded from, or None when it came in already loaded."""
        return self._library_path

    @property
    def context_gm(self) -> int:
        """SessionField::ContextGm, opaque here and only ever passed back to the shim."""
        self._require_inited()
        return int(self._lib.pto_async_ws_context_gm())

    @property
    def channel_count(self) -> int:
        self._require_inited()
        return int(self._lib.pto_async_ws_channel_count())

    @property
    def reported_queue_num(self) -> int:
        """Channels the STARS query reported populating, which bounds valid channel indices."""
        self._require_inited()
        return int(self._lib.pto_async_ws_reported_queue_num())

    @property
    def session_bytes(self) -> int:
        return int(self._lib.pto_async_session_bytes())

    @property
    def qos_default(self) -> int:
        return int(self._lib.pto_async_session_qos_default())

    @property
    def min_transfer_bytes(self) -> int:
        """Shortest transfer the engine accepts, so a block size below it is invalid."""
        return int(self._lib.pto_async_ws_min_transfer_bytes())

    @property
    def max_channels(self) -> int:
        return int(self._lib.pto_async_ws_max_channels())

    # --- observation ---------------------------------------------------------

    def sq_tail(self, channel_idx: int = 0, channel_num: int = 1) -> int:
        """The live queue tail for one channel.

        Reading this before and after a launch is what separates "moved the
        bytes" from "moved them the way the split implies": one oversized entry
        transfers the same bytes as a correct split, and only the tail differs.
        """
        self._require_inited()
        tail = int(self._lib.pto_async_ws_sq_tail(ctypes.c_uint(channel_idx), ctypes.c_uint(channel_num)))
        if tail < 0:
            raise AsyncWorkspaceError(f"reading the queue tail failed: {self._error()}")
        return tail

    # --- session template ----------------------------------------------------

    def pack_session(
        self,
        *,
        block_bytes: int,
        channel_idx: int = 0,
        channel_num: int = 1,
        tmp_buf_addr: int = 0,
        tmp_buf_size: int = DOORBELL_STAGE_BYTES,
        sync_id: int = 0,
        comm_block_offset: int = 0,
        qos: int | None = None,
        dest_rank_id: int = 0,
    ) -> np.ndarray:
        """Return the session image the kernel loads with ``pto.session_init``.

        The shim fills it, so the field order and slot width never appear here.
        ``ContextGm`` is taken from this workspace and is not a parameter.
        """
        self._require_inited()
        size = self.session_bytes
        image = np.zeros(size, dtype=np.uint8)
        rc = self._lib.pto_async_session_pack(
            image.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_uint32(size),
            ctypes.c_uint32(channel_idx),
            ctypes.c_uint32(channel_num),
            ctypes.c_uint64(block_bytes),
            ctypes.c_uint64(tmp_buf_addr),
            ctypes.c_uint32(tmp_buf_size),
            ctypes.c_uint32(sync_id),
            ctypes.c_uint64(comm_block_offset),
            ctypes.c_uint32(self.qos_default if qos is None else qos),
            ctypes.c_uint32(dest_rank_id),
        )
        if rc != 0:
            raise AsyncWorkspaceError(f"packing the session template failed: {self._error()}")
        return image


__all__ = [
    "AsyncWorkspace",
    "AsyncWorkspaceError",
    "DOORBELL_STAGE_BYTES",
]
