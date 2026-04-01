// RUN: ptoas --pto-level=level3 %s 2>&1 1>/dev/null | FileCheck %s

module {
  func.func @tilebuf_asm_invalid_dim_second_negative(
      %arg0: !pto.tile_buf<vec, 16x-2xf32>) {
    return
  }
}

// CHECK: error: dimension must be '?' or a non-negative integer
