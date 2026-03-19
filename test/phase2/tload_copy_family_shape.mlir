// RUN: ./build/tools/ptoas/ptoas --pto-backend=a5vm --a5vm-print-ir %s -o /dev/null 2>&1 | FileCheck %s

// CHECK-LABEL: func.func @tload_copy_family_shape
// CHECK: %[[ZERO_I64:.*]] = arith.constant 0 : i64
// CHECK: %[[C32:.*]] = arith.constant 32 : index
// CHECK: %[[VALID:.*]] = arith.index_castui %[[C32]] : index to i64
// CHECK: %[[C1_I64:.*]] = arith.constant 1 : i64
// CHECK: %[[NBURST:.*]] = arith.constant 32 : i64
// CHECK: %[[ELEM_BYTES:.*]] = arith.constant 4 : i64
// CHECK: %[[LOOP_STRIDE:.*]] = arith.constant 4096 : i64
// CHECK: %[[LEN_BURST:.*]] = arith.muli %[[VALID]], %[[ELEM_BYTES]] : i64
// CHECK: %[[STRIDE_BYTES:.*]] = arith.constant 128 : i64
// CHECK: a5vm.set_loop2_stride_outtoub %[[LOOP_STRIDE]], %[[LOOP_STRIDE]]
// CHECK: a5vm.set_loop1_stride_outtoub %[[LOOP_STRIDE]], %[[LOOP_STRIDE]]
// CHECK: a5vm.set_loop_size_outtoub %[[C1_I64]], %[[C1_I64]]
// CHECK: a5vm.copy_gm_to_ubuf %{{.*}}, %{{.*}}, %[[VALID]], %[[VALID]], %[[ZERO_I64]], %[[NBURST]], %[[LEN_BURST]], %[[ZERO_I64]], %[[ZERO_I64]], %[[ZERO_I64]], %[[STRIDE_BYTES]], %[[STRIDE_BYTES]]
// CHECK-SAME: layout = "nd"
// CHECK-SAME: data_select_bit = false
// CHECK-SAME: ub_pad = false
// CHECK-NOT: g_shape =
// CHECK-NOT: g_strides =
// CHECK-NOT: src_strides =
// CHECK-NOT: trace_offsets =
// CHECK-NOT: trace_sizes =

module {
  func.func @tload_copy_family_shape(%src: !pto.ptr<f32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %tv = pto.make_tensor_view %src, shape = [%c32, %c32], strides = [%c32, %c1]
      : !pto.tensor_view<?x?xf32>
    %slice = pto.partition_view %tv, offsets = [%c0, %c0], sizes = [%c32, %c32]
      : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<32x32xf32>
    %dst = pto.alloc_tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    pto.tload ins(%slice : !pto.partition_tensor_view<32x32xf32>)
      outs(%dst : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    return
  }
}
