// RUN: ./build/tools/ptoas/ptoas %s -o - | FileCheck %s

// CHECK-LABEL: @copy_ubuf_to_gm
// CHECK: a5vm.copy_ubuf_to_gm
// CHECK-SAME: {burst_count = 1 : i64, burst_len = 128 : i64, gm_stride = 32 : i64, layout = "nd", ub_pad = false, ub_stride = 64 : i64, valid_cols = 32 : i64, valid_rows = 32 : i64}
// CHECK-SAME: : memref<256xf32>, memref<1024xf32>
module {
  func.func @copy_ubuf_to_gm(%src: memref<256xf32>, %dst: memref<1024xf32>) {
    a5vm.copy_ubuf_to_gm %src, %dst {
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

// CHECK: error: 'a5vm.copy_ubuf_to_gm' op requires UB source, GM destination, and complete transfer metadata
module {
  func.func @copy_ubuf_to_gm_wrong_direction(%src: memref<256xf32>, %dst: memref<1024xf32>) {
    a5vm.copy_ubuf_to_gm %dst, %src {
      layout = "nd",
      valid_rows = 32 : i64,
      valid_cols = 32 : i64,
      burst_count = 1 : i64,
      burst_len = 128 : i64,
      gm_stride = 32 : i64,
      ub_stride = 64 : i64
    } : memref<1024xf32>, memref<256xf32>
    return
  }
}
