// RUN: ./build/tools/ptoas/ptoas %s -o - | FileCheck %s

// CHECK-LABEL: @copy_gm_to_ubuf
// CHECK: a5vm.copy_gm_to_ubuf
// CHECK-SAME: {burst_count = 1 : i64, burst_len = 128 : i64, gm_stride = 32 : i64, layout = "nd", ub_pad = false, ub_stride = 64 : i64, valid_cols = 32 : i64, valid_rows = 32 : i64}
// CHECK-SAME: : memref<1024xf32>, memref<256xf32>
module {
  func.func @copy_gm_to_ubuf(%src: memref<1024xf32>, %dst: memref<256xf32>) {
    a5vm.copy_gm_to_ubuf %src, %dst {
      layout = "nd",
      valid_rows = 32 : i64,
      valid_cols = 32 : i64,
      burst_count = 1 : i64,
      burst_len = 128 : i64,
      gm_stride = 32 : i64,
      ub_stride = 64 : i64,
      ub_pad = false
    } : memref<1024xf32>, memref<256xf32>
    return
  }
}

// CHECK: error: 'a5vm.copy_gm_to_ubuf' op requires GM source, UB destination, and complete transfer metadata
module {
  func.func @copy_gm_to_ubuf_missing_metadata(%src: memref<1024xf32>, %dst: memref<256xf32>) {
    a5vm.copy_gm_to_ubuf %dst, %src {
      layout = "nd",
      valid_rows = 32 : i64,
      valid_cols = 32 : i64,
      burst_count = 1 : i64,
      gm_stride = 32 : i64,
      ub_stride = 64 : i64
    } : memref<256xf32>, memref<1024xf32>
    return
  }
}
