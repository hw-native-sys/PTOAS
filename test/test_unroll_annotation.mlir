// Test pto-unroll-loops pass: annotation-driven native unrolling behavior.
//
// Verifies three cases:
//   1. simt_entry + pto.unroll="full" -> fully unrolled (no scf.for)
//   2. simt_entry + no annotation      -> loop NOT unrolled
//   3. pto.unroll="full" outside SIMT  -> fully unrolled (the historical
//      SIMT-only restriction of pto-unroll-simt-for has been lifted: the
//      annotation is explicit user intent in any context)

// RUN: ptoas --pto-arch=a5 --pto-backend=vpto --emit-vpto --mlir-print-ir-after=pto-unroll-loops %s -o /dev/null 2>&1 | FileCheck %s --check-prefix=UNROLL
// RUN: ptoas --pto-arch=a5 --pto-backend=vpto --emit-pto-ir %s -o - 2>&1 | FileCheck %s --check-prefix=EMITIR

// The unroll pass runs in the VPTO backend pipeline, so --emit-vpto is required
// to observe it. The annotated simt_entry loop is fully unrolled while an
// unannotated simt_entry loop survives.
// UNROLL-COUNT-1: scf.for
// UNROLL-NOT:     scf.for

// --emit-pto-ir stops before the VPTO backend pipeline, so the module is
// printed as text without requiring the CANN toolchain (regressions here used
// to fail with "CANN toolchain is required").
// EMITIR-LABEL: module attributes
// EMITIR: func.func @annotated_unrolled() -> index
// EMITIR: pto.unroll = "full"

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
