// RUN: ./build/tools/ptoas/ptoas --pto-backend=a5vm --a5vm-print-ir %s -o /dev/null 2>&1 | FileCheck %s

// This fixture pins the expectation that scalar math and loop structure stay
// in shared dialects even when the hardware-facing operation is an a5vm op.
// CHECK: arith.addi
// CHECK: scf.for
// CHECK: a5vm.vabs
// CHECK: scf.yield
module {
  func.func @shared_dialects(%src: memref<256xf32>, %dst: memref<256xf32>, %arg1: index, %arg2: index) -> index {
    %sum = arith.addi %arg1, %arg2 : index
    %loop = scf.for %iv = %arg1 to %arg2 step %arg1 iter_args(%acc = %sum) -> (index) {
      %next = arith.addi %acc, %iv : index
      scf.yield %next : index
    }
    %0 = a5vm.vlds %src[%arg1] : memref<256xf32> -> !a5vm.vec<64xf32>
    %1 = a5vm.vabs %0 : !a5vm.vec<64xf32> -> !a5vm.vec<64xf32>
    a5vm.vsts %1, %dst[%arg1] : !a5vm.vec<64xf32>, memref<256xf32>
    return %loop : index
  }
}

// CHECK-NOT: llvm.hivm
