// RUN: ptoas --pto-level=level3 %s 2>&1 1>/dev/null | FileCheck %s

module {
  func.func @tilebuf_asm_invalid_dim_question(
      %arg0: !pto.tile_buf<vec, ?x?xf32, rows=16, cols=16>) {
    return
  }
}

// CHECK: error: conflicting rows values
