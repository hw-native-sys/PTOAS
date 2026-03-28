// RUN: ptoas --pto-level=level3 %s 2>&1 1>/dev/null | FileCheck %s

module {
  func.func @tilebuf_asm_invalid_pad(
      %arg0: !pto.tile_buf<vec, 16x16xf32, pad=9>) {
    return
  }
}

// CHECK: error: unknown pad: 9
