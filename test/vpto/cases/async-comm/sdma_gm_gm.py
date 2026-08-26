#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# -----------------------------------------------------------------------------
# case: async-comm/sdma_gm_gm
# target_ops: pto.session_init, pto.sdma_gm_gm
# scenarios: A5 {soft_put} GM->UB->GM, chunk tail, session-driven offset, multicore
# -----------------------------------------------------------------------------
#
# Only the A5 {soft_put} form of pto.sdma_gm_gm is exercised here, and that is
# deliberate. That expansion is a synchronous GM->UB->GM copy, so the transfer
# is complete when the kernel returns and the result is an ordinary golden
# comparison. The engine form instead posts SQEs and rings a doorbell, which
# needs an async workspace from the AICPU STARS query and a destination that is
# polled rather than read once; neither fits a golden ST harness. The engine
# path and the two-device window shapes stay in test/lit/vpto/async_*.pto.
#
# What the four cases separate is not four kernels but four things the
# expansion has to get right. Three of them share one kernel and differ only in
# the session template the host writes, which is the point: the transfer is
# described by session data, not by the kernel.

from pathlib import Path
import sys

import numpy as np


def _bootstrap_dsl_st_common() -> None:
    here = Path(__file__).resolve()
    for candidate in here.parents:
        common_dir = candidate / "test" / "dsl-st"
        if (common_dir / "common.py").exists():
            sys.path.insert(0, str(common_dir))
            return
    raise RuntimeError("Unable to locate test/dsl-st/common.py from sdma_gm_gm.py")


_bootstrap_dsl_st_common()

from common import auto_main, golden_output_case
from ptodsl import pto


SEED = 29

# Staging chunk the A5 {soft_put} expansion uses per iteration. It is fixed in
# the expansion rather than read from SessionField::BlockBytes, so a transfer
# longer than this is what makes the copy loop run more than once.
SOFT_PUT_CHUNK_BYTES = 32768

# Field slots and their values match include/PTO/Support/AsyncSessionABI.h.
# The template is one 8-byte slot per field, so a uint64 array indexed by the
# SessionField value is the whole layout.
_SESSION_FIELDS = 13
_TMP_BUF_ADDR = 1
_TMP_BUF_SIZE = 2
_SYNC_ID = 3
_CHANNEL_IDX = 4
_CHANNEL_NUM = 5
_BLOCK_BYTES = 6
_COMM_BLOCK_OFFSET = 7
_ENGINE = 8
_FLAGS = 11
_QOS = 12

_ENGINE_SDMA = 0
_FLAG_VALID = 1
_QOS_DEFAULT = 6

# UB base for the staging buffer. Nothing else in these kernels touches UB, and
# UB is private to a core, so every block can stage at the same address.
_UB_TMP_ADDR = 0

_SESSION_TYPE = (
    "!pto.struct<i64, i64, i32, i32, i32, i32, i64, i64, i32, i32, i32, i32, i32>"
)

# The destination is the last pointer so the harness can allocate it as the
# golden output. Copy direction is still dst <- src.
SOFT_PUT_SOURCE = f"""module attributes {{pto.target_arch = "a5", pto.kernel_kind = #pto.kernel_kind<vector>}} {{
  func.func @sdma_gm_gm_soft_put(
      %src: !pto.ptr<i8, gm>, %sess_gm: !pto.ptr<i8, gm>,
      %dst: !pto.ptr<i8, gm>, %nbytes: i64) attributes {{pto.kernel}} {{
    %sess = pto.declare_struct -> {_SESSION_TYPE}
    pto.session_init %sess, %sess_gm
      : {_SESSION_TYPE}, !pto.ptr<i8, gm>
    pto.sdma_gm_gm %dst, %src, %nbytes session(%sess) {{soft_put}}
      : !pto.ptr<i8, gm>, !pto.ptr<i8, gm>, i64, {_SESSION_TYPE}
    return
  }}
}}
"""

# Each core takes one slice by overwriting SessionField::CommBlockOffset after
# the template load. The offset is written into the session rather than folded
# into the pointers because that is the field the expansion applies to both
# endpoints, so a core that read the template but failed to diverge lands on
# slice zero and is visible as a hole in the destination.
MULTICORE_SOURCE = f"""module attributes {{pto.target_arch = "a5", pto.kernel_kind = #pto.kernel_kind<vector>}} {{
  func.func @sdma_gm_gm_soft_put_multicore(
      %src: !pto.ptr<i8, gm>, %sess_gm: !pto.ptr<i8, gm>,
      %dst: !pto.ptr<i8, gm>, %chunk: i64) attributes {{pto.kernel}} {{
    %sess = pto.declare_struct -> {_SESSION_TYPE}
    pto.session_init %sess, %sess_gm
      : {_SESSION_TYPE}, !pto.ptr<i8, gm>
    %bid = pto.get_block_idx
    %off = arith.muli %bid, %chunk : i64
    pto.struct_set %sess[{_COMM_BLOCK_OFFSET}], %off : {_SESSION_TYPE}, i64
    pto.sdma_gm_gm %dst, %src, %chunk session(%sess) {{soft_put}}
      : !pto.ptr<i8, gm>, !pto.ptr<i8, gm>, i64, {_SESSION_TYPE}
    return
  }}
}}
"""


@pto.jit(
    name="sdma_gm_gm_soft_put",
    target="a5",
    backend="vpto",
    mode="explicit",
    source=SOFT_PUT_SOURCE,
)
def sdma_gm_gm_soft_put(
    src: pto.ptr(pto.i8, "gm"),
    sess_gm: pto.ptr(pto.i8, "gm"),
    dst: pto.ptr(pto.i8, "gm"),
    nbytes: pto.i64,
):
    pass


@pto.jit(
    name="sdma_gm_gm_soft_put_multicore",
    target="a5",
    backend="vpto",
    mode="explicit",
    source=MULTICORE_SOURCE,
)
def sdma_gm_gm_soft_put_multicore(
    src: pto.ptr(pto.i8, "gm"),
    sess_gm: pto.ptr(pto.i8, "gm"),
    dst: pto.ptr(pto.i8, "gm"),
    chunk: pto.i64,
):
    pass


def session_template(*, comm_block_offset: int = 0, sync_id: int = 0):
    """Host-written session template, as one uint64 slot per SessionField."""
    slots = np.zeros(_SESSION_FIELDS, dtype=np.uint64)
    # SessionField::ContextGm stays zero: it names the channel record table, and
    # {soft_put} posts nothing, so no record is ever read. The engine path is
    # what needs a real workspace address there.
    slots[_TMP_BUF_ADDR] = _UB_TMP_ADDR
    slots[_TMP_BUF_SIZE] = SOFT_PUT_CHUNK_BYTES
    slots[_SYNC_ID] = sync_id
    slots[_CHANNEL_IDX] = 0
    slots[_CHANNEL_NUM] = 1
    slots[_BLOCK_BYTES] = SOFT_PUT_CHUNK_BYTES
    slots[_COMM_BLOCK_OFFSET] = comm_block_offset
    slots[_ENGINE] = _ENGINE_SDMA
    slots[_FLAGS] = _FLAG_VALID
    slots[_QOS] = _QOS_DEFAULT
    return slots.view(np.uint8).copy()


def make_source(nbytes: int):
    rng = np.random.default_rng(SEED)
    return rng.integers(0, 256, size=nbytes, dtype=np.uint8)


def whole_buffer_case(name, *, buffer_bytes, grid=1, kernel=sdma_gm_gm_soft_put):
    """A transfer that ends up covering the whole destination buffer."""

    def make_inputs():
        return [make_source(buffer_bytes), session_template()]

    def make_expected(src, _sess):
        return src.copy()

    return golden_output_case(
        name,
        kernel,
        inputs=make_inputs,
        expected=make_expected,
        # The trailing scalar is nbytes for the single-core kernel and the
        # per-core chunk for the multicore one; both are bytes moved per launch
        # of one block.
        launch_args=lambda src, _sess: [int(src.size) // grid],
        grid=grid,
        rtol=0.0,
        atol=0.0,
    )


def offset_case(name, *, buffer_bytes, offset, nbytes):
    """A transfer displaced by SessionField::CommBlockOffset on both endpoints."""

    def make_inputs():
        return [
            make_source(buffer_bytes),
            session_template(comm_block_offset=offset),
        ]

    def make_expected(src, _sess):
        # The offset applies to source and destination alike, so the moved
        # window keeps its position and everything outside it stays zero. A
        # session read that dropped the field would copy from the base instead
        # and shift the window, which this catches.
        expected = np.zeros_like(src)
        expected[offset : offset + nbytes] = src[offset : offset + nbytes]
        return expected

    return golden_output_case(
        name,
        sdma_gm_gm_soft_put,
        inputs=make_inputs,
        expected=make_expected,
        launch_args=[nbytes],
        rtol=0.0,
        atol=0.0,
    )


CASES = [
    # One staging iteration: the shortest thing that proves the template load,
    # the UB round trip, and the MTE2/MTE3 handshake fit together.
    whole_buffer_case("sdma_gm_gm_soft_put_single_chunk", buffer_bytes=4096),
    # Two full chunks plus a short one, so the loop trip count is above one and
    # the tail iteration is clamped to what is left rather than a full chunk.
    whole_buffer_case(
        "sdma_gm_gm_soft_put_chunk_tail",
        buffer_bytes=2 * SOFT_PUT_CHUNK_BYTES + 1024,
    ),
    # The session, not the kernel, decides where the transfer lands.
    offset_case(
        "sdma_gm_gm_soft_put_comm_block_offset",
        buffer_bytes=8192,
        offset=2048,
        nbytes=4096,
    ),
    # Four cores, four disjoint slices, one shared template. The destination is
    # only whole if every core resolved its own offset.
    whole_buffer_case(
        "sdma_gm_gm_soft_put_multicore",
        buffer_bytes=16384,
        grid=4,
        kernel=sdma_gm_gm_soft_put_multicore,
    ),
]


auto_main(globals())
