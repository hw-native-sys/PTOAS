#!/usr/bin/env python3
# coding=utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Issue #1690: emitted MTE parameters for the acc2gm NZ paths.

Pins the two template parameters that used to be derived from the valid
region instead of the physical tile geometry:

- template_tload_gm_to_mat_nd2nz: the NZ loop3 destination stride is the
  tile's physical rows, so a 31-of-32 valid region still pitches C0
  fractals at 32 rows;
- template_tstore_acc_to_gm_nz2nz: the destination stride is the 16-row
  aligned row count times C0 (8 for an f32 C0=8 channel-split view), so a
  32x64 store pitches column blocks at 32 * 8 = 256, not n = 64.
"""

import unittest



from ptodsl.tilelib import ScalarType, TileSpec, ViewSpec, select


def _nd_view_spec(rows, cols):
    span = rows * cols
    return ViewSpec(
        shape=(1, 1, 1, rows, cols),
        dtype=ScalarType("f32"),
        strides=(span, span, span, cols, 1),
    )


def _mat_tile(rows, cols, valid_rows, valid_cols):
    return TileSpec(
        shape=(rows, cols),
        dtype=ScalarType("f32"),
        memory_space="mat",
        valid_shape=(valid_rows, valid_cols),
        b_layout="col_major",
        s_layout="row_major",
    )


class Acc2gmNzStrideTest(unittest.TestCase):
    def test_nd2nz_loop3_stride_is_physical_rows(self):
        specs = {
            "src": _nd_view_spec(32, 16),
            "dst": _mat_tile(32, 16, 31, 15),
        }
        selected = select("pto.tload", "a5", specs)
        text = selected.specialize(**specs).mlir_text()
        # The valid region is 31 rows; the emitted NZ loop3 stride constant
        # must still be the physical 32.
        self.assertIn("pto.constant 32 : i64", text)
        self.assertNotIn("pto.constant 31 : i64", text)

    def test_nz2nz_dst_stride_is_aligned_rows_times_c0(self):
        # 32x64 f32 store into a C0=8 NZ view: 16-row aligned rows (32)
        # times C0 (8) pitch the column blocks at 256, not n = 64.
        n_tiles, k_tiles = 32 // 16, 64 // 8
        stride2 = 16 * 8
        stride1 = n_tiles * stride2
        specs = {
            "src": TileSpec(
                shape=(32, 64),
                dtype=ScalarType("f32"),
                memory_space="acc",
                valid_shape=(32, 64),
                b_layout="col_major",
                s_layout="row_major",
            ),
            "dst": ViewSpec(
                shape=(1, k_tiles, n_tiles, 16, 8),
                dtype=ScalarType("f32"),
                strides=(k_tiles * stride1, stride1, stride2, 8, 1),
                layout="nz",
            ),
        }
        selected = select("pto.tstore", "a5", specs)
        text = selected.specialize(**specs).mlir_text()
        self.assertIn("256", text)


if __name__ == "__main__":
    unittest.main()
