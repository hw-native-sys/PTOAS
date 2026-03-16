// RUN: ptoas --pto-arch a5 --enable-insert-sync %s | FileCheck %s

module attributes {"pto.device-spec" = "Ascend950"} {
  // A5: acc->mat TINSERT must use PIPE_FIX (regular A2M path).
  func.func @tinsert_acc_mat_pipeline(%a: memref<32x32xf32, #pto.address_space<gm>>,
                                      %b: memref<32x32xf32, #pto.address_space<gm>>,
                                      %i: memref<32x32xf16, #pto.address_space<gm>>,
                                      %out: memref<32x32xf32, #pto.address_space<gm>>) {
    %c0 = arith.constant 0 : index

    %a_mat = memref.alloc() : memref<32x32xf32, #pto.address_space<mat>>
    %b_mat = memref.alloc() : memref<32x32xf32, #pto.address_space<mat>>
    %a_left = memref.alloc() : memref<32x32xf32, #pto.address_space<left>>
    %b_right = memref.alloc() : memref<32x32xf32, #pto.address_space<right>>
    %src_acc = memref.alloc() : memref<32x32xf32, #pto.address_space<acc>>

    %dst_mat = memref.alloc() : memref<32x32xf16, #pto.address_space<mat>>
    %i_mat = memref.alloc() : memref<32x32xf16, #pto.address_space<mat>>
    %out_left = memref.alloc() : memref<32x32xf16, #pto.address_space<left>>
    %i_right = memref.alloc() : memref<32x32xf16, #pto.address_space<right>>
    %out_acc = memref.alloc() : memref<32x32xf32, #pto.address_space<acc>>

    pto.tload ins(%a : memref<32x32xf32, #pto.address_space<gm>>)
              outs(%a_mat : memref<32x32xf32, #pto.address_space<mat>>)
    pto.tload ins(%b : memref<32x32xf32, #pto.address_space<gm>>)
              outs(%b_mat : memref<32x32xf32, #pto.address_space<mat>>)
    pto.tmov ins(%a_mat : memref<32x32xf32, #pto.address_space<mat>>)
             outs(%a_left : memref<32x32xf32, #pto.address_space<left>>)
    pto.tmov ins(%b_mat : memref<32x32xf32, #pto.address_space<mat>>)
             outs(%b_right : memref<32x32xf32, #pto.address_space<right>>)
    pto.tmatmul ins(%a_left, %b_right : memref<32x32xf32, #pto.address_space<left>>,
                                      memref<32x32xf32, #pto.address_space<right>>)
                outs(%src_acc : memref<32x32xf32, #pto.address_space<acc>>)

    pto.tinsert ins(%src_acc, %c0, %c0 : memref<32x32xf32, #pto.address_space<acc>>, index, index)
               outs(%dst_mat : memref<32x32xf16, #pto.address_space<mat>>)

    pto.tload ins(%i : memref<32x32xf16, #pto.address_space<gm>>)
              outs(%i_mat : memref<32x32xf16, #pto.address_space<mat>>)
    pto.tmov ins(%dst_mat : memref<32x32xf16, #pto.address_space<mat>>)
             outs(%out_left : memref<32x32xf16, #pto.address_space<left>>)
    pto.tmov ins(%i_mat : memref<32x32xf16, #pto.address_space<mat>>)
             outs(%i_right : memref<32x32xf16, #pto.address_space<right>>)
    pto.tmatmul ins(%out_left, %i_right : memref<32x32xf16, #pto.address_space<left>>,
                                        memref<32x32xf16, #pto.address_space<right>>)
                outs(%out_acc : memref<32x32xf32, #pto.address_space<acc>>)
    pto.tstore ins(%out_acc : memref<32x32xf32, #pto.address_space<acc>>)
               outs(%out : memref<32x32xf32, #pto.address_space<gm>>)
    return
  }
}

// CHECK-LABEL: __global__ AICORE void tinsert_acc_mat_pipeline(
// CHECK: set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
// CHECK-NOT: set_flag(PIPE_M, PIPE_MTE3, EVENT_ID0);
// CHECK: wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
// CHECK: TINSERT(
// CHECK: set_flag(PIPE_FIX, PIPE_MTE1, EVENT_ID0);
// CHECK-NOT: set_flag(PIPE_MTE3, PIPE_MTE1, EVENT_ID0);
// CHECK: wait_flag(PIPE_FIX, PIPE_MTE1, EVENT_ID0);
