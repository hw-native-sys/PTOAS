// RUN: ptoas --pto-level=level3 %s 2>&1 1>/dev/null | FileCheck %s

module {
  func.func @tilebuf_asm_conflicting_positional_shape(
      %arg0: !pto.tile_buf<vec, 16x16xf32, 8x16xf32>) {
    return
  }
}

// CHECK: error: conflicting rows/cols values
