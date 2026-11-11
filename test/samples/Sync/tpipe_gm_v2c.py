# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details on copying and distributing this software.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,

# PyPTO tile-op baseline: GM-slot V2C pipe matmul.
# Vector loads IN into a vec tile, tstores it into a talloc_to_aic GM FIFO
# entry and pushes it; cube pops the slot, tloads it into a mat tile (cbuf
# path, no ubuf DMA in the cube section) and computes A @ IN.
# Covers talloc_to_aic (+ tpush_to_aic / tpop_from_aiv / tfree_from_aiv, GM path).
# Golden = A @ IN.

from ptoas.mlir.ir import Context, Location, Module

_MLIR = """module {
  func.func @tpipe_gm_v2c(
      %gm_a: !pto.ptr<f32>,
      %gm_in: !pto.ptr<f32>,
      %gm_slots: !pto.ptr<f32>,
      %gm_o: !pto.ptr<f32>)
      attributes {pto.entry} {
    func.call @tpipe_gm_v2c_cube(%gm_a, %gm_slots, %gm_o)
      : (!pto.ptr<f32>, !pto.ptr<f32>, !pto.ptr<f32>) -> ()
    func.call @tpipe_gm_v2c_vector(%gm_in, %gm_slots)
      : (!pto.ptr<f32>, !pto.ptr<f32>) -> ()
    return
  }

  func.func private @tpipe_gm_v2c_cube(
      %gm_a: !pto.ptr<f32>,
      %gm_slots: !pto.ptr<f32>,
      %gm_o: !pto.ptr<f32>)
      attributes {pto.kernel_kind = #pto.kernel_kind<cube>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index

    %slots_view = pto.make_tensor_view %gm_slots, shape = [%c16, %c16], strides = [%c16, %c1] : !pto.tensor_view<16x16xf32>
    pto.aic_initialize_pipe {id = 0, dir_mask = 2, slot_size = 1024}
      (gm_slot_tensor = %slots_view : !pto.tensor_view<16x16xf32>)

    %mat_a = pto.alloc_tile : !pto.tile_buf<loc=mat, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=col_major, slayout=row_major, fractal=512, pad=0>
    %mat_b = pto.alloc_tile : !pto.tile_buf<loc=mat, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=col_major, slayout=row_major, fractal=512, pad=0>
    %left_tile = pto.alloc_tile : !pto.tile_buf<loc=left, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=col_major, slayout=row_major, fractal=512, pad=0>
    %right_tile = pto.alloc_tile : !pto.tile_buf<loc=right, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=col_major, fractal=512, pad=0>
    %acc_tile = pto.alloc_tile : !pto.tile_buf<loc=acc, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=col_major, slayout=row_major, fractal=1024, pad=0>

    %gm_a_view = pto.make_tensor_view %gm_a, shape = [%c16, %c16], strides = [%c16, %c1] : !pto.tensor_view<16x16xf32>
    %gm_a_part = pto.partition_view %gm_a_view, offsets = [%c0, %c0], sizes = [%c16, %c16] : !pto.tensor_view<16x16xf32> -> !pto.partition_tensor_view<16x16xf32>
    pto.tload ins(%gm_a_part : !pto.partition_tensor_view<16x16xf32>) outs(%mat_a : !pto.tile_buf<loc=mat, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=col_major, slayout=row_major, fractal=512, pad=0>)

    %slot = pto.tpop_from_aiv {id = 0, split = 0}
      -> !pto.tensor_view<16x16xf32>
    %slot_part = pto.partition_view %slot, offsets = [%c0, %c0], sizes = [%c16, %c16] : !pto.tensor_view<16x16xf32> -> !pto.partition_tensor_view<16x16xf32>
    pto.tload ins(%slot_part : !pto.partition_tensor_view<16x16xf32>) outs(%mat_b : !pto.tile_buf<loc=mat, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=col_major, slayout=row_major, fractal=512, pad=0>)
    pto.tfree_from_aiv {id = 0, split = 0}

    pto.tmov ins(%mat_a : !pto.tile_buf<loc=mat, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=col_major, slayout=row_major, fractal=512, pad=0>) outs(%left_tile : !pto.tile_buf<loc=left, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=col_major, slayout=row_major, fractal=512, pad=0>)
    pto.tmov ins(%mat_b : !pto.tile_buf<loc=mat, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=col_major, slayout=row_major, fractal=512, pad=0>) outs(%right_tile : !pto.tile_buf<loc=right, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=col_major, fractal=512, pad=0>)
    pto.tmatmul ins(%left_tile, %right_tile : !pto.tile_buf<loc=left, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=col_major, slayout=row_major, fractal=512, pad=0>, !pto.tile_buf<loc=right, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=col_major, fractal=512, pad=0>) outs(%acc_tile : !pto.tile_buf<loc=acc, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=col_major, slayout=row_major, fractal=1024, pad=0>)

    %gm_o_view = pto.make_tensor_view %gm_o, shape = [%c16, %c16], strides = [%c16, %c1] : !pto.tensor_view<16x16xf32>
    %gm_o_part = pto.partition_view %gm_o_view, offsets = [%c0, %c0], sizes = [%c16, %c16] : !pto.tensor_view<16x16xf32> -> !pto.partition_tensor_view<16x16xf32>
    pto.tstore ins(%acc_tile : !pto.tile_buf<loc=acc, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=col_major, slayout=row_major, fractal=1024, pad=0>) outs(%gm_o_part : !pto.partition_tensor_view<16x16xf32>)
    return
  }

  func.func private @tpipe_gm_v2c_vector(
      %gm_in: !pto.ptr<f32>,
      %gm_slots: !pto.ptr<f32>)
      attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index

    %slots_view = pto.make_tensor_view %gm_slots, shape = [%c16, %c16], strides = [%c16, %c1] : !pto.tensor_view<16x16xf32>
    pto.aiv_initialize_pipe {id = 0, dir_mask = 2, slot_size = 1024}
      (gm_slot_tensor = %slots_view : !pto.tensor_view<16x16xf32>)

    %data_tile = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %gm_in_view = pto.make_tensor_view %gm_in, shape = [%c16, %c16], strides = [%c16, %c1] : !pto.tensor_view<16x16xf32>
    %gm_in_part = pto.partition_view %gm_in_view, offsets = [%c0, %c0], sizes = [%c16, %c16] : !pto.tensor_view<16x16xf32> -> !pto.partition_tensor_view<16x16xf32>
    pto.tload ins(%gm_in_part : !pto.partition_tensor_view<16x16xf32>) outs(%data_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>)

    %entry = pto.talloc_to_aic {id = 0, split = 0}
      -> !pto.tensor_view<16x16xf32>
    %entry_part = pto.partition_view %entry, offsets = [%c0, %c0], sizes = [%c16, %c16] : !pto.tensor_view<16x16xf32> -> !pto.partition_tensor_view<16x16xf32>
    pto.tstore ins(%data_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%entry_part : !pto.partition_tensor_view<16x16xf32>)
    pto.tpush_to_aic(%entry : !pto.tensor_view<16x16xf32>) {id = 0, split = 0}
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
