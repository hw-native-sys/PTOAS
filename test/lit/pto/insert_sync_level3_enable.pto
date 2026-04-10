// RUN: ptoas --pto-level=level3 --pto-arch a5 --enable-insert-sync %s | FileCheck %s

module {
  func.func @run_partition(%arg0: !pto.ptr<f32>, %arg1: !pto.ptr<f32>,
                           %arg2: index, %arg3: index, %arg4: index, %arg5: index, %arg6: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %c1024 = arith.constant 1024 : index
    %c1048576 = arith.constant 1048576 : index
    %c0_i64 = arith.constant 0 : i64

    %base = pto.make_tensor_view %arg0,
            shape = [%c1, %c1, %c16, %c1024, %c1024],
            strides = [%c1048576, %c1048576, %c1048576, %c1024, %c1]
            : !pto.tensor_view<1x1x16x1024x1024xf32>
    %part = pto.partition_view %base,
             offsets = [%c0, %c0, %c0, %c0, %c0],
             sizes = [%arg2, %arg3, %arg4, %arg5, %arg6]
            : !pto.tensor_view<1x1x16x1024x1024xf32>
              -> !pto.partition_tensor_view<?x?x?x?x?xf32>

    %tmp0 = arith.muli %arg2, %arg3 : index
    %tmp1 = arith.muli %tmp0, %arg4 : index
    %tmp2 = arith.muli %tmp1, %arg5 : index

    %tile = pto.alloc_tile addr = %c0_i64 valid_row = %tmp2 valid_col = %arg6
            : !pto.tile_buf<loc=vec, dtype=f32, rows=256, cols=16, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>

    pto.tload ins(%part : !pto.partition_tensor_view<?x?x?x?x?xf32>)
            outs(%tile : !pto.tile_buf<loc=vec, dtype=f32, rows=256, cols=16, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>)

    %dst_view = pto.make_tensor_view %arg1,
                shape = [%c1, %c1, %c16, %c1024, %c1024],
                strides = [%c1048576, %c1048576, %c1048576, %c1024, %c1]
                : !pto.tensor_view<1x1x16x1024x1024xf32>
    %dst_part = pto.partition_view %dst_view,
                offsets = [%c0, %c0, %c0, %c0, %c0],
                sizes = [%arg2, %arg3, %arg4, %arg5, %arg6]
               : !pto.tensor_view<1x1x16x1024x1024xf32>
                 -> !pto.partition_tensor_view<?x?x?x?x?xf32>

    pto.tstore ins(%tile : !pto.tile_buf<loc=vec, dtype=f32, rows=256, cols=16, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
             outs(%dst_part : !pto.partition_tensor_view<?x?x?x?x?xf32>)
    return
  }
}

// CHECK-LABEL: __global__ AICORE void run_partition(
// CHECK: set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
// CHECK: wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
