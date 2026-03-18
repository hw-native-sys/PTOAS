// RUN: ./build/tools/ptoas/ptoas --pto-backend=a5vm %s -o - | FileCheck %s

// CHECK-LABEL: func.func @abs_kernel_2d
// CHECK: a5vm.copy_gm_to_ubuf
// CHECK: %[[LOAD:.+]] = a5vm.vlds %{{.+}} : memref<256xf32> -> !a5vm.vec<64xf32>
// CHECK: %[[ABS:.+]] = a5vm.vabs %[[LOAD]] : !a5vm.vec<64xf32> -> !a5vm.vec<64xf32>
// CHECK: a5vm.vsts %[[ABS]], %{{.+}} : !a5vm.vec<64xf32>, memref<256xf32>
// CHECK: a5vm.copy_ubuf_to_gm
// CHECK-NOT: llvm.hivm
module {
  func.func @abs_kernel_2d(%base: memref<1024xf32>, %ubuf: memref<256xf32>, %out: memref<1024xf32>, %index: index) {
    a5vm.copy_gm_to_ubuf %base, %ubuf {
      layout = "nd",
      valid_rows = 32 : i64,
      valid_cols = 32 : i64,
      burst_count = 1 : i64,
      burst_len = 128 : i64,
      gm_stride = 32 : i64,
      ub_stride = 64 : i64,
      ub_pad = false
    } : memref<1024xf32>, memref<256xf32>
    %loaded = a5vm.vlds %ubuf[%index] : memref<256xf32> -> !a5vm.vec<64xf32>
    %abs = a5vm.vabs %loaded : !a5vm.vec<64xf32> -> !a5vm.vec<64xf32>
    a5vm.vsts %abs, %ubuf[%index] : !a5vm.vec<64xf32>, memref<256xf32>
    a5vm.copy_ubuf_to_gm %ubuf, %out {
      layout = "nd",
      valid_rows = 32 : i64,
      valid_cols = 32 : i64,
      burst_count = 1 : i64,
      burst_len = 128 : i64,
      gm_stride = 32 : i64,
      ub_stride = 64 : i64,
      ub_pad = false
    } : memref<256xf32>, memref<1024xf32>
    return
  }
}
