// RUN: ptoas --pto-level=level3 %s 2>&1 1>/dev/null | FileCheck %s

module {
  func.func @tilebuf_asm_invalid_vrow_value(
      %arg0: !pto.tile_buf<vec, 16x16xf32, v_row=-2, v_col=1>) {
    return
  }
}

// CHECK: error: v_row must be '?', -1, or a non-negative integer
