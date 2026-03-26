// RUN: ./build/tools/ptoas/ptoas --pto-backend=a5vm --emit-a5vm %s -o - 2>/dev/null | FileCheck %s

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
// CHECK: %[[GM_BASE_BYTES:.*]] = pto.castptr %arg0 : !pto.ptr<f32, gm> -> !pto.ptr<i8, gm>
// CHECK: %[[ZERO_INDEX:.*]] = arith.index_castui %[[ZERO_I64]] : i64 to index
// CHECK: %[[GM_OFFSET_PTR:.*]] = pto.addptr %[[GM_BASE_BYTES]], %[[ZERO_INDEX]] : <i8, gm> -> <i8, gm>
// CHECK: a5vm.set_loop2_stride_outtoub %[[LOOP_STRIDE]], %[[LOOP_STRIDE]]
// CHECK: a5vm.set_loop1_stride_outtoub %[[LOOP_STRIDE]], %[[LOOP_STRIDE]]
// CHECK: a5vm.set_loop_size_outtoub %[[C1_I64]], %[[C1_I64]]
// CHECK: %[[GM_TYPED_PTR:.*]] = pto.castptr %[[GM_OFFSET_PTR]] : !pto.ptr<i8, gm> -> !pto.ptr<f32, gm>
// CHECK: %[[FALSE:.*]] = arith.constant false
// CHECK: a5vm.copy_gm_to_ubuf %[[GM_TYPED_PTR]], %{{.*}}, %[[VALID]], %[[VALID]], %[[ZERO_I64]], %[[NBURST]], %[[LEN_BURST]], %[[ZERO_I64]], %[[ZERO_I64]], %[[FALSE]], %[[ZERO_I64]], %[[STRIDE_BYTES]], %[[STRIDE_BYTES]]
// CHECK-SAME: : !pto.ptr<f32, gm>, !pto.ptr<f32, ub>, i64, i64, i64, i64, i64, i64, i64, i1, i64, i64, i64
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
