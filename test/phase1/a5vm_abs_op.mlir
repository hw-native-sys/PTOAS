// RUN: ./build/tools/ptoas/ptoas %s -o - | FileCheck %s

// CHECK-LABEL: @abs_kernel
// CHECK: %[[ABS:.+]] = a5vm.abs %[[IN:.+]] : !a5vm.vec<64xf32> -> !a5vm.vec<64xf32>
module {
  func.func @abs_kernel(%arg0: !a5vm.vec<64xf32>) -> !a5vm.vec<64xf32> {
    %0 = a5vm.abs %arg0 : !a5vm.vec<64xf32> -> !a5vm.vec<64xf32>
    return %0 : !a5vm.vec<64xf32>
  }
}

// CHECK: error: 'a5vm.abs' op mismatched vector types
module {
  func.func @abs_mismatch(%arg0: !a5vm.vec<64xf32>) -> !a5vm.vec<128xi16> {
    %0 = a5vm.abs %arg0 : !a5vm.vec<64xf32> -> !a5vm.vec<128xi16>
    return %0 : !a5vm.vec<128xi16>
  }
}
