// RUN: ptoas --pto-level=level3 %s 2>&1 1>/dev/null | FileCheck %s

module {
  func.func @tilebuf_asm_missing_rows_cols(
      %arg0: !pto.tile_buf<loc=vec, dtype=f16>) {
    return
  }
}

// CHECK: error: missing rows/cols
