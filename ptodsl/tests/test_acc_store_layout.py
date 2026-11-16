#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""acc_store layout spellings: bare nz2nz is the native normal-mode writeback.

Issue #1751. nz2nz without a split operand (SPLIT_EN = 0) is the hardware
normal-mode L0C writeback, so the DSL must be able to spell it without inventing
a split value; nz2nd and the tuple spellings stay unchanged.
"""

import logging

from ptoas.mlir import ir
from ptoas.mlir.dialects import pto as dialect

from ptodsl._ops_mte import _acc_store_layout

LOGGER = logging.getLogger(__name__)


def layout_triple(layout):
    with ir.Context() as ctx, ir.Location.unknown():
        dialect.register_dialect(ctx)
        mode, split, loop0_src_stride = _acc_store_layout(layout)
        return (
            None if mode is None else str(mode),
            None if split is None else str(split.type),
            None if loop0_src_stride is None else str(loop0_src_stride.type),
        )


def check_layout_spellings():
    # Bare nz2nz: normal-mode writeback, no SPLIT_EN operand at all.
    assert layout_triple("nz2nz") == ("#pto<acc_store_mode nz2nz>", None, None)
    # Pre-existing spellings keep working.
    assert layout_triple("nz2nd") == ("#pto<acc_store_mode nz2nd>", None, None)
    assert layout_triple(None) == (None, None, None)
    assert layout_triple(("nz2dn", 1)) == ("#pto<acc_store_mode nz2dn>", None, "i64")
    # The tuple form still passes an explicit split through (0 disables it).
    assert layout_triple(("nz2nz", 0)) == ("#pto<acc_store_mode nz2nz>", "i64", None)
    assert layout_triple(("nz2nz", 1)) == ("#pto<acc_store_mode nz2nz>", "i64", None)


def check_invalid_spellings():
    # nz2dn needs its operand and unknown tokens are rejected, as before.
    for layout in ("nz2dn", "bogus"):
        try:
            layout_triple(layout)
        except ValueError as exc:
            assert "nz2nd or nz2nz" in str(exc), exc
        else:
            raise AssertionError(f"{layout!r} should be rejected")


def main():
    check_layout_spellings()
    check_invalid_spellings()
    LOGGER.info("ptodsl_acc_store_layout: PASS")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
