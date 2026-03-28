// RUN: ptoas --pto-level=level3 %s 2>&1 1>/dev/null | FileCheck %s

module {
  func.func @tilebuf_asm_conflicting_dtype(
      %arg0: !pto.tile_buf<vec, 16x16xf32, dtype=f16>) {
    return
  }
}

// CHECK: error: conflicting dtype values
