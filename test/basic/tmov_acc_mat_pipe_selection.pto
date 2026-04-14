// RUN: ptoas --pto-arch a5 --enable-insert-sync %s | FileCheck %s

module attributes {"pto.device-spec" = "Ascend950"} {
  // A5: tmov acc->mat should be treated as PIPE_FIX for sync insertion.
  func.func @tmov_acc_mat_pipeline(%a: memref<32x32xf16, #pto.address_space<gm>>,
                                   %b: memref<32x32xf16, #pto.address_space<gm>>) {
    %a_mat = memref.alloc() : memref<32x32xf16, #pto.address_space<mat>>
    %b_mat = memref.alloc() : memref<32x32xf16, #pto.address_space<mat>>
    %a_left = memref.alloc() : memref<32x32xf16, #pto.address_space<left>>
    %b_right = memref.alloc() : memref<32x32xf16, #pto.address_space<right>>
    %acc = memref.alloc() : memref<32x32xf32, #pto.address_space<acc>>
    %dst_mat = memref.alloc() : memref<32x32xf16, #pto.address_space<mat>>
    %dst_left = memref.alloc() : memref<32x32xf16, #pto.address_space<left>>

    pto.tload ins(%a : memref<32x32xf16, #pto.address_space<gm>>)
              outs(%a_mat : memref<32x32xf16, #pto.address_space<mat>>)
    pto.tload ins(%b : memref<32x32xf16, #pto.address_space<gm>>)
              outs(%b_mat : memref<32x32xf16, #pto.address_space<mat>>)
    pto.tmov ins(%a_mat : memref<32x32xf16, #pto.address_space<mat>>)
             outs(%a_left : memref<32x32xf16, #pto.address_space<left>>)
    pto.tmov ins(%b_mat : memref<32x32xf16, #pto.address_space<mat>>)
             outs(%b_right : memref<32x32xf16, #pto.address_space<right>>)
    pto.tmatmul ins(%a_left, %b_right : memref<32x32xf16, #pto.address_space<left>>,
                                        memref<32x32xf16, #pto.address_space<right>>)
                outs(%acc : memref<32x32xf32, #pto.address_space<acc>>)
    pto.tmov ins(%acc : memref<32x32xf32, #pto.address_space<acc>>)
             outs(%dst_mat : memref<32x32xf16, #pto.address_space<mat>>)
    pto.tmov ins(%dst_mat : memref<32x32xf16, #pto.address_space<mat>>)
             outs(%dst_left : memref<32x32xf16, #pto.address_space<left>>)
    return
  }
}

// CHECK-LABEL: __global__ AICORE void tmov_acc_mat_pipeline(
// CHECK: set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
// CHECK: wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
// CHECK: TMOV(
// CHECK: set_flag(PIPE_FIX, PIPE_MTE1, EVENT_ID0);
// CHECK: wait_flag(PIPE_FIX, PIPE_MTE1, EVENT_ID0);
