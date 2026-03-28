// RUN: ptoas --pto-level=level3 %s 2>&1 1>/dev/null | FileCheck %s

module {
  func.func @tilebuf_asm_invalid_fractal(
      %arg0: !pto.tile_buf<vec, 16x16xf32, fractal=7>) {
    return
  }
}

// CHECK: error: unsupported s_fractal_size: 7
