// RUN: ./build/tools/ptoas/ptoas --pto-backend=a5vm --a5vm-print-ir %s -o /dev/null 2>&1 | FileCheck %s
// RUN: ./build/tools/ptoas/ptoas --pto-arch a5 --pto-backend=a5vm --a5vm-emit-hivm-llvm %s -o - 2>/dev/null | FileCheck --check-prefix=CHECK-HIVM %s

// CHECK-LABEL: func.func @copy_dynamic_transfer_operands
// CHECK: %[[ZERO_I64:.*]] = arith.constant 0 : i64
// CHECK: %[[ROW_I64:.*]] = arith.index_castui %arg2 : index to i64
// CHECK: %[[COL_I64:.*]] = arith.index_castui %arg3 : index to i64
// CHECK: %[[C1_I64:.*]] = arith.constant 1 : i64
// CHECK: %[[NBURST:.*]] = arith.constant 32 : i64
// CHECK: %[[ELEM_BYTES:.*]] = arith.constant 4 : i64
// CHECK: %[[LOOP_STRIDE:.*]] = arith.constant 4096 : i64
// CHECK: %[[LEN_BURST:.*]] = arith.muli %[[COL_I64]], %[[ELEM_BYTES]] : i64
// CHECK: %[[STRIDE_BYTES:.*]] = arith.constant 128 : i64
// CHECK: a5vm.set_loop2_stride_outtoub %[[LOOP_STRIDE]], %[[LOOP_STRIDE]]
// CHECK: a5vm.set_loop1_stride_outtoub %[[LOOP_STRIDE]], %[[LOOP_STRIDE]]
// CHECK: a5vm.set_loop_size_outtoub %[[C1_I64]], %[[C1_I64]]
// CHECK: a5vm.copy_gm_to_ubuf %{{.*}}, %{{.*}}, %[[ROW_I64]], %[[COL_I64]], %[[ZERO_I64]], %[[NBURST]], %[[LEN_BURST]], %[[ZERO_I64]], %[[ZERO_I64]], %[[ZERO_I64]], %[[STRIDE_BYTES]], %[[STRIDE_BYTES]]
// CHECK-SAME: layout = "nd"
// CHECK-SAME: data_select_bit = false
// CHECK-SAME: ub_pad = false
// CHECK: a5vm.set_loop_size_ubtoout %[[C1_I64]], %[[C1_I64]]
// CHECK: a5vm.set_loop1_stride_ubtoout %[[LOOP_STRIDE]], %[[LOOP_STRIDE]]
// CHECK: a5vm.set_loop2_stride_ubtoout %[[LOOP_STRIDE]], %[[LOOP_STRIDE]]
// CHECK: a5vm.copy_ubuf_to_gm %{{.*}}, %{{.*}}, %[[ROW_I64]], %[[COL_I64]], %[[ZERO_I64]], %[[NBURST]], %[[LEN_BURST]], %[[ZERO_I64]], %[[STRIDE_BYTES]], %[[STRIDE_BYTES]]
// CHECK-SAME: layout = "nd"
// CHECK-NOT: valid_rows =
// CHECK-NOT: valid_cols =
// CHECK-HIVM-LABEL: define dso_local void @copy_dynamic_transfer_operands
// CHECK-HIVM: call void @llvm.hivm.SET.LOOP2.STRIDE.OUTTOUB
// CHECK-HIVM: call void @llvm.hivm.SET.LOOP1.STRIDE.OUTTOUB
// CHECK-HIVM: call void @llvm.hivm.SET.LOOP.SIZE.OUTTOUB
// CHECK-HIVM: call void @llvm.hivm.MOV.OUT.TO.UB.ALIGN.V2.f32.DV
// CHECK-HIVM: call void @llvm.hivm.SET.LOOP.SIZE.UBTOOUT
// CHECK-HIVM: call void @llvm.hivm.SET.LOOP1.STRIDE.UBTOOUT
// CHECK-HIVM: call void @llvm.hivm.SET.LOOP2.STRIDE.UBTOOUT
// CHECK-HIVM: call void @llvm.hivm.MOV.UB.TO.OUT.ALIGN.V2.DV
// CHECK-HIVM-NOT: call void @llvm.hivm.SET.LOOP1.STRIDE(i64)

module {
  func.func @copy_dynamic_transfer_operands(%src: !pto.ptr<f32>, %dst: !pto.ptr<f32>, %valid_row: index, %valid_col: index) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %src_view = pto.make_tensor_view %src, shape = [%c32, %c32], strides = [%c32, %c1]
      : !pto.tensor_view<?x?xf32>
    %src_slice = pto.partition_view %src_view, offsets = [%c0, %c0], sizes = [%c32, %c32]
      : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<32x32xf32>
    %tile = pto.alloc_tile valid_row = %valid_row valid_col = %valid_col
      : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
    pto.tload ins(%src_slice : !pto.partition_tensor_view<32x32xf32>)
      outs(%tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>)

    %dst_view = pto.make_tensor_view %dst, shape = [%c32, %c32], strides = [%c32, %c1]
      : !pto.tensor_view<?x?xf32>
    %dst_slice = pto.partition_view %dst_view, offsets = [%c0, %c0], sizes = [%c32, %c32]
      : !pto.tensor_view<?x?xf32> -> !pto.partition_tensor_view<32x32xf32>
    pto.tstore ins(%tile : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>)
      outs(%dst_slice : !pto.partition_tensor_view<32x32xf32>)
    return
  }
}
