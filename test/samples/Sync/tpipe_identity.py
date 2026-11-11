# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details on copying and distributing this software.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,

# PyPTO tile-op baseline: C2V pipe identity via pushed matmul result.
# Cube loads A and an identity matrix I, computes A@I into acc and pushes the
# acc tile to the vector core (tpush_to_aiv); the vector core pops it
# (tpop_from_aic) and stores to GM (tfree_from_aic).
# All cube-side DMAs stay on the cbuf/FIX pipes (no ubuf access in the cube
# section). Covers tpush_to_aiv, tpop_from_aic, tfree_from_aic.
# Golden = A @ I = A.

from ptoas.mlir.ir import Context, Location, Module

_MLIR = """module {
  func.func @tpipe_identity(
      %gm_a: !pto.ptr<f32>,
      %gm_i: !pto.ptr<f32>,
      %gm_o: !pto.ptr<f32>)
      attributes {pto.entry} {
    func.call @tpipe_identity_cube(%gm_a, %gm_i)
      : (!pto.ptr<f32>, !pto.ptr<f32>) -> ()
    func.call @tpipe_identity_vector(%gm_o)
      : (!pto.ptr<f32>) -> ()
    return
  }

  func.func private @tpipe_identity_cube(
      %gm_a: !pto.ptr<f32>,
      %gm_i: !pto.ptr<f32>)
      attributes {pto.kernel_kind = #pto.kernel_kind<cube>} {
    %c0_i32 = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index

    // C2V producer (cube) imports the consumer-owned fifo buffer.
    %c2v_import = pto.import_reserved_buffer {
      name = "c2v_fifo",
      peer_func = @tpipe_identity_vector
    } -> i32
    pto.aic_initialize_pipe {id = 0, dir_mask = 1, slot_size = 1024}
      (c2v_consumer_buf = %c2v_import : i32,
       v2c_consumer_buf = %c0_i32 : i32)

    %mat_a = pto.alloc_tile : !pto.tile_buf<loc=mat, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=col_major, slayout=row_major, fractal=512, pad=0>
    %mat_i = pto.alloc_tile : !pto.tile_buf<loc=mat, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=col_major, slayout=row_major, fractal=512, pad=0>
    %left_tile = pto.alloc_tile : !pto.tile_buf<loc=left, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=col_major, slayout=row_major, fractal=512, pad=0>
    %right_tile = pto.alloc_tile : !pto.tile_buf<loc=right, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=col_major, fractal=512, pad=0>
    %acc_tile = pto.alloc_tile : !pto.tile_buf<loc=acc, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=col_major, slayout=row_major, fractal=1024, pad=0>

    %gm_a_view = pto.make_tensor_view %gm_a, shape = [%c16, %c16], strides = [%c16, %c1] : !pto.tensor_view<16x16xf32>
    %gm_a_part = pto.partition_view %gm_a_view, offsets = [%c0, %c0], sizes = [%c16, %c16] : !pto.tensor_view<16x16xf32> -> !pto.partition_tensor_view<16x16xf32>
    pto.tload ins(%gm_a_part : !pto.partition_tensor_view<16x16xf32>) outs(%mat_a : !pto.tile_buf<loc=mat, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=col_major, slayout=row_major, fractal=512, pad=0>)
    %gm_i_view = pto.make_tensor_view %gm_i, shape = [%c16, %c16], strides = [%c16, %c1] : !pto.tensor_view<16x16xf32>
    %gm_i_part = pto.partition_view %gm_i_view, offsets = [%c0, %c0], sizes = [%c16, %c16] : !pto.tensor_view<16x16xf32> -> !pto.partition_tensor_view<16x16xf32>
    pto.tload ins(%gm_i_part : !pto.partition_tensor_view<16x16xf32>) outs(%mat_i : !pto.tile_buf<loc=mat, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=col_major, slayout=row_major, fractal=512, pad=0>)

    pto.tmov ins(%mat_a : !pto.tile_buf<loc=mat, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=col_major, slayout=row_major, fractal=512, pad=0>) outs(%left_tile : !pto.tile_buf<loc=left, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=col_major, slayout=row_major, fractal=512, pad=0>)
    pto.tmov ins(%mat_i : !pto.tile_buf<loc=mat, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=col_major, slayout=row_major, fractal=512, pad=0>) outs(%right_tile : !pto.tile_buf<loc=right, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=col_major, fractal=512, pad=0>)
    pto.tmatmul ins(%left_tile, %right_tile : !pto.tile_buf<loc=left, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=col_major, slayout=row_major, fractal=512, pad=0>, !pto.tile_buf<loc=right, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=col_major, fractal=512, pad=0>) outs(%acc_tile : !pto.tile_buf<loc=acc, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=col_major, slayout=row_major, fractal=1024, pad=0>)

    pto.tpush_to_aiv(%acc_tile : !pto.tile_buf<loc=acc, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=col_major, slayout=row_major, fractal=1024, pad=0>) {id = 0, split = 0}
    return
  }

  func.func private @tpipe_identity_vector(
      %gm_o: !pto.ptr<f32>)
      attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {
    %c0_i32 = arith.constant 0 : i32
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index

    %c2v_local = pto.reserve_buffer {
      name = "c2v_fifo",
      size = 8192,
      location = #pto.address_space<vec>,
      auto = true
    } -> i32
    pto.aiv_initialize_pipe {id = 0, dir_mask = 1, slot_size = 1024}
      (c2v_consumer_buf = %c2v_local : i32,
       v2c_consumer_buf = %c0_i32 : i32)

    %fifo_tile = pto.tpop_from_aic {id = 0, split = 0}
      -> !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>

    %gm_o_view = pto.make_tensor_view %gm_o, shape = [%c16, %c16], strides = [%c16, %c1] : !pto.tensor_view<16x16xf32>
    %gm_o_part = pto.partition_view %gm_o_view, offsets = [%c0, %c0], sizes = [%c16, %c16] : !pto.tensor_view<16x16xf32> -> !pto.partition_tensor_view<16x16xf32>
    pto.tstore ins(%fifo_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%gm_o_part : !pto.partition_tensor_view<16x16xf32>)
    pto.tfree_from_aic {id = 0, split = 0}
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
