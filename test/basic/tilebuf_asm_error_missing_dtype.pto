// RUN: ptoas --pto-level=level3 %s 2>&1 1>/dev/null | FileCheck %s

module {
  func.func @tilebuf_asm_missing_dtype(
      %arg0: !pto.tile_buf<loc=vec, rows=16, cols=16>) {
    return
  }
}

// CHECK: error: missing dtype
