// RUN: not ptoas %s 2>&1 | FileCheck %s

module {
  func.func @invalid_fusion_kind(
      %a: memref<32x32xf32>,
      %b: memref<32x32xf32>,
      %c: memref<32x32xf32>,
      %d: memref<32x32xf32>) {
    pto.tfusion ins(%a, %b, %c : memref<32x32xf32>, memref<32x32xf32>, memref<32x32xf32>)
                outs(%d : memref<32x32xf32>)
                ops = [0, 2],
                prev_pos = [0],
                keep_stage = [],
                fusion_kind = "bad_kind"
    return
  }
}

// CHECK: error: 'pto.tfusion' op expects fusion_kind to be "elemwise_chain"
