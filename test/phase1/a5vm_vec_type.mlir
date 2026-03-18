// RUN: ./build/tools/ptoas/ptoas %s 2>&1 | FileCheck %s

// CHECK-LABEL: @legal_f32
// CHECK: !a5vm.vec<64xf32>
module {
  func.func @legal_f32(%arg0: !a5vm.vec<64xf32>) -> !a5vm.vec<64xf32> {
    return %arg0 : !a5vm.vec<64xf32>
  }
}

// CHECK-LABEL: @legal_i16
// CHECK: !a5vm.vec<128xi16>
module {
  func.func @legal_i16(%arg0: !a5vm.vec<128xi16>) -> !a5vm.vec<128xi16> {
    return %arg0 : !a5vm.vec<128xi16>
  }
}

// The corrected Phase 1 contract keeps the normalized vector spelling while
// shifting the operation surface to copy and register primitives.
// CHECK: error: '!a5vm.vec<32xf32>' expected exactly 256 bytes
module {
  func.func @illegal_f32_width(%arg0: !a5vm.vec<32xf32>) -> !a5vm.vec<32xf32> {
    return %arg0 : !a5vm.vec<32xf32>
  }
}

// CHECK: error: '!a5vm.vec<64xi64>' expected exactly 256 bytes
module {
  func.func @illegal_i64_width(%arg0: !a5vm.vec<64xi64>) -> !a5vm.vec<64xi64> {
    return %arg0 : !a5vm.vec<64xi64>
  }
}
