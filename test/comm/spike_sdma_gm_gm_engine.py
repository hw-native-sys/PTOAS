#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# ----------------------------------------------------------------------------
# spike: sdma_gm_gm, engine path
# target_ops: pto.session_init, pto.sdma_gm_gm
# scenarios: SDMA engine post, SQE split, ragged tail, session-driven offset
# ----------------------------------------------------------------------------
#
# The engine form of pto.sdma_gm_gm: the kernel writes SQEs, publishes the queue
# tail and rings a doorbell, and the SDMA engine moves the bytes afterwards. Its
# {soft_put} sibling in sdma_gm_gm.py is a synchronous copy and needs none of
# that, which is why the two are separate files rather than separate cases.
#
# This is a spike, and what it is really checking is not the transfer. The IR is
# pinned in test/lit/vpto and the data path is covered by the {soft_put} cases;
# what is unproven is whether the engine path can run under the PTODSL ST harness
# at all. Three assumptions have to hold together, and none is checkable offline:
#
#   1. AsyncWorkspace can attach to the device and context torch_npu already set
#      up, instead of needing to own aclrtSetDevice itself.
#   2. Memory the workspace allocates through the runtime is addressable by a
#      kernel this harness launched, so a raw device address can be handed over
#      inside a session template.
#   3. A destination the engine fills can be observed from the harness, which
#      synchronizes the kernel and knows nothing about the engine.
#
# If all three hold, the engine path needs no framework change and the C++ and
# shell harness these cases used to need has nothing left to do. If one fails,
# this file is where it shows up, and the failure says which.
#
# The host side comes along in the kernel's own shared library. AsyncWorkspaceShim
# is listed in native_options below, so PTODSL compiles and links it beside the
# generated launch code, and this file reaches it through
# compiled.native_library(). One build, one artifact, and the shim is part of the
# build's identity: editing it rebuilds rather than silently reusing.
#
# Requires CANN >= 9.0.0 for the AICPU STARS query and a real device. There is no
# skip path: a spike that quietly passes when it could not run would defeat its
# own purpose.
#
# Which is why it lives here and not under test/vpto/cases/async-comm/ next to its
# {soft_put} sibling. run_host_vpto_validation.sh discovers every .py in that tree
# and runs it, and a case needing a 9.0 toolkit would fail the whole validation
# run on every machine without one. Promoting it once the three assumptions above
# are confirmed is a rename, and that should be a deliberate step rather than
# something a spike does on its way in. Run it directly, naming the generation of
# the card it is pointed at:
#
#   PTO_ASYNC_ARCH=a5 python3 test/comm/spike_sdma_gm_gm_engine.py
#   PTO_ASYNC_ARCH=a3 python3 test/comm/spike_sdma_gm_gm_engine.py
#
# Do not add ``from __future__ import annotations``. ``@pto.jit`` reads the
# entry annotations at decoration time and needs the live ``pto.ptr`` objects,
# not their string forms.

import os
from pathlib import Path
import sys
import time

import numpy as np


def _bootstrap_paths() -> None:
    """Put test/dsl-st and test/comm on sys.path, wherever this file is checked out."""
    here = Path(__file__).resolve()
    for candidate in here.parents:
        common_dir = candidate / "test" / "dsl-st"
        comm_dir = candidate / "test" / "comm"
        if (common_dir / "common.py").exists():
            sys.path.insert(0, str(common_dir))
            sys.path.insert(0, str(comm_dir))
            return
    raise RuntimeError("Unable to locate test/dsl-st/common.py from spike_sdma_gm_gm_engine.py")


_bootstrap_paths()

from async_workspace import DOORBELL_STAGE_BYTES, AsyncWorkspace
from common import auto_main
from ptodsl import pto


SEED = 31

# The doorbell is the one part of the post that differs by generation, so the
# arch is a knob rather than a constant. A2/A3 reaches it only by MTE and stages
# the tail in UB; A5 writes it with st_dev, at a different offset. Everything
# else is shared, and the compiler draws no distinction between a2 and a3.
#
# Required rather than defaulted. This is the only thing here that decides which
# card the result may be run on, and getting it wrong is not a benign mistake: on
# A2/A3 a store aimed at sq_reg_base the wrong way faults the vector unit hard
# enough to leave the card in an unrecoverable RAS state. A card that has to be
# reset because a variable was unset is worth one line of typing.
_SUPPORTED_ARCHES = ("a2", "a3", "a5")
ARCH = os.environ.get("PTO_ASYNC_ARCH")
if ARCH is None:
    raise RuntimeError(
        "PTO_ASYNC_ARCH must name the generation of the card this will run on, one of "
        f"{_SUPPORTED_ARCHES}. There is no default: the doorbell sequence differs "
        "between A2/A3 and A5, and posting the wrong one at a card can leave it in "
        "an unrecoverable state."
    )
if ARCH not in _SUPPORTED_ARCHES:
    raise RuntimeError(f"PTO_ASYNC_ARCH must be one of {_SUPPORTED_ARCHES}, got {ARCH!r}")

# How long to wait for the engine to drain. The harness synchronizes the kernel,
# which returns once the doorbell is rung, so the destination is still being
# written when the check starts.
POLL_TIMEOUT_MS = int(os.environ.get("PTO_ASYNC_POLL_TIMEOUT_MS", "2000"))
POLL_INTERVAL_S = 0.001

# One channel, index zero. A group wide enough to spread a post across is a
# scheduling policy the expansion does not implement yet, and the tail read has
# to name the same geometry the session did.
CHANNEL_IDX = 0
CHANNEL_NUM = 1

# UB base for the A2/A3 doorbell staging. UB is private to a core and these
# kernels touch nothing else in it, so the bottom of the buffer is free.
UB_STAGE_ADDR = 0

# Pipe event id for that staging, chosen so it cannot collide: these kernels
# raise no other MTE3 event.
DOORBELL_SYNC_ID = 0

_SESSION_TYPE = "!pto.struct<i64, i64, i32, i32, i32, i32, i64, i64, i32, i32, i32, i32, i32>"

# The destination is the last pointer so the harness allocates it as the output
# buffer. Copy direction is still dst <- src.
ENGINE_SOURCE = f"""module attributes {{pto.target_arch = "{ARCH}", pto.kernel_kind = #pto.kernel_kind<vector>}} {{
  func.func @sdma_gm_gm_engine(
      %src: !pto.ptr<i8, gm>, %sess_gm: !pto.ptr<i8, gm>,
      %dst: !pto.ptr<i8, gm>, %nbytes: i64) attributes {{pto.kernel}} {{
    %sess = pto.declare_struct -> {_SESSION_TYPE}
    pto.session_init %sess, %sess_gm
      : {_SESSION_TYPE}, !pto.ptr<i8, gm>
    pto.sdma_gm_gm %dst, %src, %nbytes session(%sess)
      : !pto.ptr<i8, gm>, !pto.ptr<i8, gm>, i64, {_SESSION_TYPE}
    return
  }}
}}
"""


@pto.jit(
    name="sdma_gm_gm_engine",
    target=ARCH,
    backend="vpto",
    mode="explicit",
    source=ENGINE_SOURCE,
    native_options={
        # Paths resolve against this file. AsyncWorkspace.h sits beside the shim;
        # AsyncSessionABI.h, which owns every field position the shim uses, comes
        # from the repository include tree.
        "host_sources": ["AsyncWorkspaceShim.cpp"],
        "include_dirs": [".", "../../include"],
        # AsyncWorkspace resolves the CANN runtime with dlopen rather than linking
        # it, so this is the shim's only link dependency.
        "link_libraries": ["dl"],
    },
)
def sdma_gm_gm_engine(
    src: pto.ptr(pto.i8, "gm"),
    sess_gm: pto.ptr(pto.i8, "gm"),
    dst: pto.ptr(pto.i8, "gm"),
    nbytes: pto.i64,
):
    pass


_workspace: AsyncWorkspace | None = None


def workspace() -> AsyncWorkspace:
    """The process-wide workspace, built on first use.

    Built lazily rather than at import: --list and --emit-mlir run before the
    harness selects a device, and neither should need a card.

    Compiling here is what makes the shim reachable, since it lives in the
    library this call produces. The harness compiles the same kernel again before
    it launches; that hits the specialization cache and returns this same handle,
    so the library is built once.
    """
    global _workspace
    if _workspace is None:
        compiled = sdma_gm_gm_engine.compile()
        ws = AsyncWorkspace(library=compiled.native_library())
        ws.init(channels=CHANNEL_NUM)
        reported = ws.reported_queue_num
        if reported <= CHANNEL_IDX:
            raise RuntimeError(
                f"the STARS query reported {reported} populated channels, "
                f"which does not cover channel {CHANNEL_IDX}"
            )
        print(
            f"async workspace ready: arch={ARCH} "
            f"library={compiled.native_library_path()} "
            f"channels={ws.channel_count} reported={reported}"
        )
        _workspace = ws
    return _workspace


def make_source(nbytes: int) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    return rng.integers(0, 256, size=nbytes, dtype=np.uint8)


def _tail_delta(before: int, after: int) -> int:
    """SQEs posted between two tail reads.

    The expansion masks the tail into the ring on every entry, so a delta is only
    a count while the ring did not wrap. These cases post single digits into a
    ring thousands of entries deep, so a wrap here means something else is wrong
    and is worth reporting rather than folding away.
    """
    if after < before:
        raise AssertionError(
            f"the queue tail moved backwards, from {before} to {after}; "
            "the ring wrapped or another writer is sharing the channel"
        )
    return after - before


def _poll_for_match(device_output, golden: np.ndarray) -> tuple[np.ndarray, float]:
    """Read the destination back until it matches the golden or the deadline passes."""
    started = time.monotonic()
    deadline = started + POLL_TIMEOUT_MS / 1000.0
    actual = device_output.cpu().numpy()
    while not np.array_equal(actual, golden):
        if time.monotonic() >= deadline:
            break
        time.sleep(POLL_INTERVAL_S)
        actual = device_output.cpu().numpy()
    return actual, time.monotonic() - started


def engine_case(
    name: str,
    *,
    buffer_bytes: int,
    nbytes: int,
    block_bytes: int,
    comm_block_offset: int = 0,
):
    """One engine transfer, described by the session the host writes for it.

    ``block_bytes`` is what splits the transfer into SQEs, so it is the knob that
    decides how many entries the queue should gain. ``comm_block_offset`` shifts
    both endpoints, so the moved window keeps its position in the buffer.
    """
    if comm_block_offset + nbytes > buffer_bytes:
        raise RuntimeError(
            f"case {name!r} would move past the end of the buffer: "
            f"{comm_block_offset} + {nbytes} > {buffer_bytes}"
        )
    want_entries = -(-nbytes // block_bytes)
    tail_chunk = nbytes - (want_entries - 1) * block_bytes

    def make_case():
        ws = workspace()
        floor = ws.min_transfer_bytes
        # Both the full block and the short last one become an engine transfer,
        # so either being under the floor is a broken case rather than a finding.
        if block_bytes < floor or tail_chunk < floor:
            raise RuntimeError(
                f"case {name!r} splits into chunks of {block_bytes} and {tail_chunk} bytes, "
                f"below the {floor}-byte engine minimum"
            )

        src = make_source(buffer_bytes)
        session = ws.pack_session(
            block_bytes=block_bytes,
            channel_idx=CHANNEL_IDX,
            channel_num=CHANNEL_NUM,
            tmp_buf_addr=UB_STAGE_ADDR,
            tmp_buf_size=DOORBELL_STAGE_BYTES,
            sync_id=DOORBELL_SYNC_ID,
            comm_block_offset=comm_block_offset,
        )

        # Everything outside the moved window stays zero. A session read that
        # dropped CommBlockOffset would copy from the base and shift the window,
        # which shows up here rather than as a byte count that still adds up.
        golden = np.zeros(buffer_bytes, dtype=np.uint8)
        window = slice(comm_block_offset, comm_block_offset + nbytes)
        golden[window] = src[window]

        out = np.zeros(buffer_bytes, dtype=np.uint8)
        expected = {
            "golden": golden,
            "tail_before": ws.sq_tail(CHANNEL_IDX, CHANNEL_NUM),
        }
        return [src, session, out], expected, [nbytes]

    def check_case(device_inputs, expected):
        actual, waited_s = _poll_for_match(device_inputs[-1], expected["golden"])
        entries = _tail_delta(
            expected["tail_before"], workspace().sq_tail(CHANNEL_IDX, CHANNEL_NUM)
        )
        matched = np.array_equal(actual, expected["golden"])

        # Report the post before the data. A correct entry count with an unfilled
        # destination means the kernel did its part and the engine did not, which
        # is a different failure from an entry count that never matched.
        if entries != want_entries:
            raise AssertionError(
                f"{name}: expected {want_entries} SQEs for {nbytes} bytes in "
                f"{block_bytes}-byte blocks, the queue tail gained {entries}"
                + ("" if matched else "; the destination did not match either")
            )
        if not matched:
            differing = int(np.count_nonzero(actual != expected["golden"]))
            first = int(np.flatnonzero(actual != expected["golden"])[0])
            raise AssertionError(
                f"{name}: {want_entries} SQEs were posted, but after {waited_s * 1000:.0f}ms "
                f"{differing} of {actual.size} bytes still differ, first at byte {first}"
            )

    return {
        "name": name,
        "kernel": sdma_gm_gm_engine,
        "make_case": make_case,
        "check": check_case,
    }


CASES = [
    # One SQE. The shortest post that exercises the record read, the SQE write,
    # the tail publish and the doorbell together.
    engine_case(
        "sdma_gm_gm_engine_single_entry",
        buffer_bytes=4096,
        nbytes=4096,
        block_bytes=4096,
    ),
    # An even split, so the loop runs more than once and every entry is full.
    engine_case(
        "sdma_gm_gm_engine_even_split",
        buffer_bytes=4096,
        nbytes=4096,
        block_bytes=1024,
    ),
    # A block size that does not divide the transfer: two full entries and a
    # short one. The tail entry has to carry what is left rather than a full
    # block, which would run past the end of the buffer.
    engine_case(
        "sdma_gm_gm_engine_ragged_tail",
        buffer_bytes=4096,
        nbytes=4096,
        block_bytes=1536,
    ),
    # The session, not the kernel, decides where the transfer lands.
    engine_case(
        "sdma_gm_gm_engine_comm_block_offset",
        buffer_bytes=8192,
        nbytes=2048,
        block_bytes=2048,
        comm_block_offset=2048,
    ),
]


auto_main(globals())
