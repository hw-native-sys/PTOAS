// RUN: ptoas --pto-level=level3 %s 2>&1 1>/dev/null | FileCheck %s

module {
  func.func @tilebuf_asm_invalid_blayout(
      %arg0: !pto.tile_buf<vec, 16x16xf32, blayout=bad>) {
    return
  }
}

// CHECK: error: unknown blayout: bad
