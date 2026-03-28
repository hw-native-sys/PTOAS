// RUN: ptoas --pto-level=level3 %s 2>&1 1>/dev/null | FileCheck %s

module {
  func.func @tilebuf_asm_rows_negative(
      %arg0: !pto.tile_buf<loc=vec, dtype=f32, rows=-1, cols=16>) {
    return
  }
}

// CHECK: error: rows must be non-negative
