# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details on copying and distributing this software.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,

# PyPTO tile-op baseline: GM-slot C2V pipe identity (pre-filled slot form).
# The host pre-fills the GM slot buffer; the cube kernel only talloc_to_aiv's
# the current FIFO entry and pushes it (no cube-side DMA at all); the vector
# kernel pops the slot, tloads it into a vec tile and stores to OUT.
# Covers talloc_to_aiv (+ tpush_to_aiv / tpop_from_aic / tfree_from_aic, GM path).
# Golden = identity on the pre-filled slots content.

from ptoas.mlir.ir import Context, Location, Module

_MLIR = """module {
  func.func @tpipe_gm_c2v(
      %gm_slots: !pto.ptr<f32>,
      %gm_o: !pto.ptr<f32>)
      attributes {pto.entry} {
    func.call @tpipe_gm_c2v_cube(%gm_slots)
      : (!pto.ptr<f32>) -> ()
    func.call @tpipe_gm_c2v_vector(%gm_slots, %gm_o)
      : (!pto.ptr<f32>, !pto.ptr<f32>) -> ()
    return
  }

  func.func private @tpipe_gm_c2v_cube(
      %gm_slots: !pto.ptr<f32>)
      attributes {pto.kernel_kind = #pto.kernel_kind<cube>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index

    %slots_view = pto.make_tensor_view %gm_slots, shape = [%c16, %c16], strides = [%c16, %c1] : !pto.tensor_view<16x16xf32>
    pto.aic_initialize_pipe {id = 0, dir_mask = 1, slot_size = 1024}
      (gm_slot_tensor = %slots_view : !pto.tensor_view<16x16xf32>)

    %entry = pto.talloc_to_aiv {id = 0, split = 0}
      -> !pto.tensor_view<16x16xf32>
    pto.tpush_to_aiv(%entry : !pto.tensor_view<16x16xf32>) {id = 0, split = 0}
    return
  }

  func.func private @tpipe_gm_c2v_vector(
      %gm_slots: !pto.ptr<f32>,
      %gm_o: !pto.ptr<f32>)
      attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index

    %slots_view = pto.make_tensor_view %gm_slots, shape = [%c16, %c16], strides = [%c16, %c1] : !pto.tensor_view<16x16xf32>
    pto.aiv_initialize_pipe {id = 0, dir_mask = 1, slot_size = 1024}
      (gm_slot_tensor = %slots_view : !pto.tensor_view<16x16xf32>)

    %slot = pto.tpop_from_aic {id = 0, split = 0}
      -> !pto.tensor_view<16x16xf32>
    %slot_part = pto.partition_view %slot, offsets = [%c0, %c0], sizes = [%c16, %c16] : !pto.tensor_view<16x16xf32> -> !pto.partition_tensor_view<16x16xf32>
    %out_tile = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    pto.tload ins(%slot_part : !pto.partition_tensor_view<16x16xf32>) outs(%out_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>)

    %gm_o_view = pto.make_tensor_view %gm_o, shape = [%c16, %c16], strides = [%c16, %c1] : !pto.tensor_view<16x16xf32>
    %gm_o_part = pto.partition_view %gm_o_view, offsets = [%c0, %c0], sizes = [%c16, %c16] : !pto.tensor_view<16x16xf32> -> !pto.partition_tensor_view<16x16xf32>
    pto.tstore ins(%out_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=16, cols=16, v_row=16, v_col=16, blayout=row_major, slayout=none_box, fractal=512, pad=0>) outs(%gm_o_part : !pto.partition_tensor_view<16x16xf32>)
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
