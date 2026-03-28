// RUN: ptoas --pto-level=level3 %s 2>&1 1>/dev/null | FileCheck %s

module {
  func.func @tilebuf_asm_unknown_loc(
      %arg0: !pto.tile_buf<foo, 16x16xf32>) {
    return
  }
}

// CHECK: error: unknown loc: foo
