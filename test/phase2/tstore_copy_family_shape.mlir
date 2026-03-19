// RUN: ./build/tools/ptoas/ptoas --pto-backend=a5vm --a5vm-print-ir %s -o /dev/null 2>&1 | FileCheck %s

// CHECK-LABEL: func.func @tstore_copy_family_shape
// CHECK: a5vm.set_loop_size_ubtoout
// CHECK-SAME: loop2 = 32
// CHECK-SAME: loop1 = 32
// CHECK: a5vm.set_loop1_stride_ubtoout
// CHECK-SAME: src_stride = 1
// CHECK-SAME: dst_stride = 1
// CHECK: a5vm.set_loop2_stride_ubtoout
// CHECK-SAME: src_stride = 32
// CHECK-SAME: dst_stride = 32
// CHECK: a5vm.copy_ubuf_to_gm
// CHECK-SAME: layout = "nd"
// CHECK-SAME: valid_rows = 32
// CHECK-SAME: valid_cols = 32
// CHECK-SAME: burst_count = 32
// CHECK-SAME: burst_len = 32
// CHECK-SAME: gm_stride = 32
// CHECK-SAME: ub_stride = 32
// CHECK-SAME: dst_strides = [32, 1]
// CHECK-SAME: trace_offsets = [0, 0]
// CHECK-SAME: trace_sizes = [32, 32]

module {
  func.func @tstore_copy_family_shape(%dst: !pto.ptr<f32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %tv = pto.make_tensor_view %dst, shape = [%c32, %c32], strides = [%c32, %c1]
      : !pto.tensor_view<?x?xf32>
    %slice = pto.partition_view %tv, offsets = [%c0, %c0], sizes = [%c32, %c32]
      : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<32x32xf32>
    %src = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    pto.tstore ins(%src : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
      outs(%slice : !pto.partition_tensor_view<32x32xf32>)
    return
  }
}
