// RUN: ptoas --pto-level=level3 %s 2>&1 1>/dev/null | FileCheck %s

module {
  func.func @tilebuf_asm_conflicting_loc(
      %arg0: !pto.tile_buf<vec, loc=acc, 16x16xf32>) {
    return
  }
}

// CHECK: error: conflicting loc values
