// RUN: ptoas --pto-level=level3 %s 2>&1 1>/dev/null | FileCheck %s

module {
  func.func @tilebuf_asm_unknown_key(
      %arg0: !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=16, foo=1>) {
    return
  }
}

// CHECK: error: unknown key in tile_buf: foo
