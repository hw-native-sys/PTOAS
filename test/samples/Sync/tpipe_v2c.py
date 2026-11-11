# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details on copying and distributing this software.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,

# PyPTO tile-op baseline: V2C pipe matmul.
# Vector loads B (16x16), converts ND->NZ, pushes to cube (tpush_to_aic);
# cube loads A (16x16), pops B (tpop_from_aiv), computes A@B, stores via acc.
# Covers tpush_to_aic, tpop_from_aiv, tfree_from_aiv. Golden = A @ B.

from ptoas.mlir.ir import Context, Location, Module

_MLIR = """module {
  func.func @tpipe_v2c(
      %gm_a: !pto.ptr<f32>,
      %gm_b: !pto.ptr<f32>,
      %gm_o: !pto.ptr<f32>)
      attributes {pto.entry} {
    func.call @tpipe_v2c_cube(%gm_a, %gm_o)
      : (!pto.ptr<f32>, !pto.ptr<f32>) -> ()
    func.call @tpipe_v2c_vector(%gm_b)
      : (!pto.ptr<f32>) -> ()
    return
  }

  func.func private @tpipe_v2c_cube(
      %gm_a: !pto.ptr<f32>,
      %gm_o: !pto.ptr<f32>)
      attributes {pto.kernel_kind = #pto.kernel_kind<cube>} {
    %c0_i32 = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    
    // V2C consumer (cube) owns the local fifo buffer.
    %v2c_reserve = pto.reserve_buffer {
      name = "v2c_fifo",
      size = 8192,
      location = #pto.address_space<mat>,
      auto = true
    } -> i32
    pto.aic_initialize_pipe {id = 0, dir_mask = 2, slot_size = 1024}
      (c2v_consumer_buf = %c0_i32 : i32,
       v2c_consumer_buf = %v2c_reserve : i32)

    %mat_a_tile = pto.alloc_tile : !pto.tile_buf<loc=mat, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=col_major, slayout=row_major, fractal=512, pad=0>
    %left_tile = pto.alloc_tile : !pto.tile_buf<loc=left, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=col_major, slayout=row_major, fractal=512, pad=0>
    %right_tile = pto.alloc_tile : !pto.tile_buf<loc=right, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=col_major, fractal=512, pad=0>
    %acc_tile = pto.alloc_tile : !pto.tile_buf<loc=acc, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=col_major, slayout=row_major, fractal=1024, pad=0>

    %gm_a_view = pto.make_tensor_view %gm_a, shape = [%c16, %c16], strides = [%c16, %c1] : !pto.tensor_view<16x16xf32>
    %gm_a_part = pto.partition_view %gm_a_view, offsets = [%c0, %c0], sizes = [%c16, %c16] : !pto.tensor_view<16x16xf32> -> !pto.partition_tensor_view<16x16xf32>
    pto.tload ins(%gm_a_part : !pto.partition_tensor_view<16x16xf32>) outs(%mat_a_tile : !pto.tile_buf<loc=mat, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=col_major, slayout=row_major, fractal=512, pad=0>)
    pto.tmov ins(%mat_a_tile : !pto.tile_buf<loc=mat, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=col_major, slayout=row_major, fractal=512, pad=0>) outs(%left_tile : !pto.tile_buf<loc=left, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=col_major, slayout=row_major, fractal=512, pad=0>)

    %received = pto.tpop_from_aiv {id = 0, split = 0}
      -> !pto.tile_buf<loc=mat, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=col_major, slayout=row_major, fractal=512, pad=0>
    pto.tmov ins(%received : !pto.tile_buf<loc=mat, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=col_major, slayout=row_major, fractal=512, pad=0>) outs(%right_tile : !pto.tile_buf<loc=right, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=col_major, fractal=512, pad=0>)
    pto.tfree_from_aiv {id = 0, split = 0}

    pto.tmatmul ins(%left_tile, %right_tile : !pto.tile_buf<loc=left, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=col_major, slayout=row_major, fractal=512, pad=0>, !pto.tile_buf<loc=right, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=col_major, fractal=512, pad=0>) outs(%acc_tile : !pto.tile_buf<loc=acc, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=col_major, slayout=row_major, fractal=1024, pad=0>)

    %gm_o_view = pto.make_tensor_view %gm_o, shape = [%c16, %c16], strides = [%c16, %c1] : !pto.tensor_view<16x16xf32>
    %gm_o_part = pto.partition_view %gm_o_view, offsets = [%c0, %c0], sizes = [%c16, %c16] : !pto.tensor_view<16x16xf32> -> !pto.partition_tensor_view<16x16xf32>
    pto.tstore ins(%acc_tile : !pto.tile_buf<loc=acc, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=col_major, slayout=row_major, fractal=1024, pad=0>) outs(%gm_o_part : !pto.partition_tensor_view<16x16xf32>)
    return
  }

  func.func private @tpipe_v2c_vector(
      %gm_b: !pto.ptr<f32>)
      attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {
    %c0_i32 = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    
    %v2c_import = pto.import_reserved_buffer {
      name = "v2c_fifo",
      peer_func = @tpipe_v2c_cube
    } -> i32
    pto.aiv_initialize_pipe {id = 0, dir_mask = 2, slot_size = 1024}
      (c2v_consumer_buf = %c0_i32 : i32,
       v2c_consumer_buf = %v2c_import : i32)

    %vec_nd = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %vec_nz = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=col_major, slayout=row_major, fractal=512, pad=0>

    %gm_b_view = pto.make_tensor_view %gm_b, shape = [%c16, %c16], strides = [%c16, %c1] : !pto.tensor_view<16x16xf32>
    %gm_b_part = pto.partition_view %gm_b_view, offsets = [%c0, %c0], sizes = [%c16, %c16] : !pto.tensor_view<16x16xf32> -> !pto.partition_tensor_view<16x16xf32>
    pto.tload ins(%gm_b_part : !pto.partition_tensor_view<16x16xf32>) outs(%vec_nd : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.tmov ins(%vec_nd : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%vec_nz : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=col_major, slayout=row_major, fractal=512, pad=0>)
    pto.tpush_to_aic(%vec_nz : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=col_major, slayout=row_major, fractal=512, pad=0>) {id = 0, split = 0}
    return
  }
}"""


def build():
    with Context() as ctx:
        from ptoas.mlir.dialects import pto as _pto
        _pto.register_dialect(ctx, load=True)
        with Location.unknown(ctx):
            return Module.parse(_MLIR)


if __name__ == "__main__":
    print(build())
