// Test pto-unroll-loops pass: annotation-driven native unrolling behavior.
//
// Verifies three cases:
//   1. simt_entry + pto.unroll="full" -> fully unrolled (no scf.for)
//   2. simt_entry + no annotation      -> loop NOT unrolled
//   3. pto.unroll="full" outside SIMT  -> fully unrolled (the historical
//      SIMT-only restriction of pto-unroll-simt-for has been lifted: the
//      annotation is explicit user intent in any context)

// RUN: ptoas --pto-arch=a5 --pto-backend=vpto %s -o /dev/null --mlir-print-ir-after=pto-unroll-loops 2>&1 | FileCheck %s

// There should be exactly 1 scf.for in the output (from case 2).
// CHECK-COUNT-1: scf.for
// CHECK-NOT:     scf.for

module attributes {pto.kernel_kind = #pto.kernel_kind<vector>} {
  // Case 1: simt_entry + pto.unroll="full" -> unrolled (no scf.for)
  func.func @annotated_unrolled() attributes {pto.simt_entry} {
    %buf = memref.alloc() : memref<1xindex>
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    "scf.for"(%c0, %c4, %c1) ({
    ^bb0(%i: index):
      %val = arith.addi %i, %i : index
      memref.store %val, %buf[%c0] : memref<1xindex>
      scf.yield
    }) {"pto.unroll" = "full"} : (index, index, index) -> ()
    return
  }

  // Case 2: simt_entry but NO annotation -> loop survives
  func.func @not_annotated_skipped() attributes {pto.simt_entry} {
    %buf = memref.alloc() : memref<1xindex>
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %c4 step %c1 {
      %val = arith.addi %i, %i : index
      memref.store %val, %buf[%c0] : memref<1xindex>
    }
    return
  }

  // Case 3: annotation outside SIMT context -> also unrolled now
  func.func @not_simt_entry_unrolled() {
    %buf = memref.alloc() : memref<1xindex>
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    scf.for %i = %c0 to %c4 step %c1 {
      %val = arith.addi %i, %i : index
      memref.store %val, %buf[%c0] : memref<1xindex>
    } {pto.unroll = "full"}
    return
  }
}
