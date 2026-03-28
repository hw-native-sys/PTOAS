// RUN: ptoas --print-ir-after-all --print-ir-after-all-func-filter=tilebuf_asm --pto-level=level3 %s 2>&1 1>/dev/null | FileCheck %s

module {
  func.func @tilebuf_asm(
      %arg0: !pto.tile_buf<loc=vec, dtype=f32, rows=1, cols=16, v_row=1, v_col=16,
                           blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %arg1: !pto.tile_buf<vec, 16x128xf32, valid=16x1, blayout=col_major,
                           slayout=row_major, fractal=1024, pad=2>) {
    return
  }

  func.func @tilebuf_asm_dynamic(
      %arg0: !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=32, v_row=?, v_col=?,
                           blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %arg1: !pto.tile_buf<acc, 16x16xbf16, valid=?x8, blayout=col_major>) {
    return
  }
}

// CHECK: func.func @tilebuf_asm
// CHECK: !pto.tile_buf<vec, 1x16xf32>
// CHECK: !pto.tile_buf<vec, 16x128xf32, valid=16x1, blayout=col_major, slayout=row_major, fractal=1024, pad=2>

// CHECK: func.func @tilebuf_asm_dynamic
// CHECK: !pto.tile_buf<vec, 32x32xf16, valid=?x?>
// CHECK: !pto.tile_buf<acc, 16x16xbf16, valid=?x8, blayout=col_major>
