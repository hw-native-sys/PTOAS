// RUN: ./build/tools/ptoas/ptoas %s -o - | FileCheck %s

// CHECK-LABEL: @load_kernel
// CHECK: = a5vm.load
// CHECK-SAME: {domain = "gm", layout = "nd", valid_cols = 32 : i64, valid_rows = 32 : i64}
// CHECK-SAME: : memref<1024xf32> -> !a5vm.vec<64xf32>
module {
  func.func @load_kernel(%base: memref<1024xf32>, %index: index) -> !a5vm.vec<64xf32> {
    %0 = a5vm.load %base[%index] {
      layout = "nd",
      valid_rows = 32,
      valid_cols = 32,
      domain = "gm"
    } : memref<1024xf32> -> !a5vm.vec<64xf32>
    return %0 : !a5vm.vec<64xf32>
  }
}
