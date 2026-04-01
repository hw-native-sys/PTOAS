// RUN: ptoas --pto-level=level3 %s 2>&1 1>/dev/null | FileCheck %s

module {
  func.func @tilebuf_asm_invalid_valid_neg(
      %arg0: !pto.tile_buf<vec, 16x16xf32, valid=-1x8>) {
    return
  }
}

// CHECK: error: valid shape expects exactly 2 dimensions
