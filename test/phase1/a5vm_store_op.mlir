// RUN: ./build/tools/ptoas/ptoas %s -o - | FileCheck %s

// CHECK-LABEL: @store_kernel
// CHECK: a5vm.store
// CHECK-SAME: {domain = "vec", layout = "nd"} : !a5vm.vec<64xf32>, memref<1024xf32>
// CHECK-NEXT: return
module {
  func.func @store_kernel(%value: !a5vm.vec<64xf32>, %base: memref<1024xf32>, %index: index) {
    a5vm.store %value, %base[%index] {
      layout = "nd",
      domain = "vec"
    } : !a5vm.vec<64xf32>, memref<1024xf32>
    return
  }
}

// CHECK-NOT: valid_rows
// CHECK-NOT: valid_cols
