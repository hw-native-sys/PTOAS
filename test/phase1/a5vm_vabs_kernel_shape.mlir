// RUN: ./build/tools/ptoas/ptoas %s -o - | FileCheck %s

// CHECK-LABEL: @vabs_kernel
// CHECK: %[[LOAD:.+]] = a5vm.vlds %{{.+}} : memref<256xf32> -> !a5vm.vec<64xf32>
// CHECK: %[[ABS:.+]] = a5vm.vabs %[[LOAD]] : !a5vm.vec<64xf32> -> !a5vm.vec<64xf32>
// CHECK: a5vm.vsts %[[ABS]], %{{.+}} : !a5vm.vec<64xf32>, memref<256xf32>
module {
  func.func @vabs_kernel(%src: memref<256xf32>, %dst: memref<256xf32>, %index: index) {
    %tile = a5vm.vlds %src[%index] : memref<256xf32> -> !a5vm.vec<64xf32>
    %abs = a5vm.vabs %tile : !a5vm.vec<64xf32> -> !a5vm.vec<64xf32>
    a5vm.vsts %abs, %dst[%index] : !a5vm.vec<64xf32>, memref<256xf32>
    return
  }
}

// CHECK: error: 'a5vm.vabs' op requires matching register vector shape
module {
  func.func @vabs_shape_mismatch(%src: memref<256xf32>, %dst: memref<256xf32>, %index: index) {
    %tile = a5vm.vlds %src[%index] : memref<256xf32> -> !a5vm.vec<64xf32>
    %abs = a5vm.vabs %tile : !a5vm.vec<64xf32> -> !a5vm.vec<128xi16>
    a5vm.vsts %abs, %dst[%index] : !a5vm.vec<128xi16>, memref<256xf32>
    return
  }
}
