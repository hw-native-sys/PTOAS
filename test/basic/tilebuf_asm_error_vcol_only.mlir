// RUN: ptoas --pto-level=level3 %s 2>&1 1>/dev/null | FileCheck %s

module {
  func.func @tilebuf_asm_vcol_only(
      %arg0: !pto.tile_buf<vec, 16x16xf32, v_col=1>) {
    return
  }
}

// CHECK: error: v_row and v_col must be provided together
