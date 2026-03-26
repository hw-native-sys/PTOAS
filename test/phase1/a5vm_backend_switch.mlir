// RUN: ./build/tools/ptoas/ptoas --pto-backend=a5vm %s -o - | FileCheck %s

// CHECK-LABEL: func.func @abs_kernel_2d
// CHECK: %[[MASK:.+]] = a5vm.pset_b32 "PAT_ALL" : !a5vm.mask
// CHECK: a5vm.copy_gm_to_ubuf
// CHECK: %[[LOAD:.+]] = a5vm.vlds %{{.+}} : !pto.ptr<f32, ub> -> !a5vm.vec<64xf32>
// CHECK: %[[ABS:.+]] = a5vm.vabs %[[LOAD]], %[[MASK]] : !a5vm.vec<64xf32>, !a5vm.mask -> !a5vm.vec<64xf32>
// CHECK: a5vm.vsts %[[ABS]], %{{.+}}, %[[MASK]] : !a5vm.vec<64xf32>, !pto.ptr<f32, ub>, !a5vm.mask
// CHECK: a5vm.copy_ubuf_to_gm
// CHECK-NOT: llvm.hivm
module {
  func.func @abs_kernel_2d(%base: !pto.ptr<f32, gm>, %ubuf: !pto.ptr<f32, ub>, %out: !pto.ptr<f32, gm>, %index: index) {
    %c0_i64 = arith.constant 0 : i64
    %c32_i64 = arith.constant 32 : i64
    %c128_i64 = arith.constant 128 : i64
    %c4_i64 = arith.constant 4 : i64
    %cfalse = arith.constant false
    %mask = a5vm.pset_b32 "PAT_ALL" : !a5vm.mask
    a5vm.copy_gm_to_ubuf %base, %ubuf, %c32_i64, %c32_i64, %c0_i64, %c32_i64, %c128_i64, %c0_i64, %c0_i64, %cfalse, %c0_i64, %c128_i64, %c128_i64 : !pto.ptr<f32, gm>, !pto.ptr<f32, ub>, i64, i64, i64, i64, i64, i64, i64, i1, i64, i64, i64
    %loaded = a5vm.vlds %ubuf[%index] : !pto.ptr<f32, ub> -> !a5vm.vec<64xf32>
    %abs = a5vm.vabs %loaded, %mask : !a5vm.vec<64xf32>, !a5vm.mask -> !a5vm.vec<64xf32>
    a5vm.vsts %abs, %ubuf[%index], %mask : !a5vm.vec<64xf32>, !pto.ptr<f32, ub>, !a5vm.mask
    a5vm.copy_ubuf_to_gm %ubuf, %out, %c32_i64, %c32_i64, %c0_i64, %c32_i64, %c128_i64, %c0_i64, %c4_i64, %c128_i64 : !pto.ptr<f32, ub>, !pto.ptr<f32, gm>, i64, i64, i64, i64, i64, i64, i64, i64
    return
  }
}
