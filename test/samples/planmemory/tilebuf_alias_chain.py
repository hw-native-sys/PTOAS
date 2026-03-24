PTO_IR = r"""
module {
  func.func @tilebuf_alias_chain(%arg0: memref<32x32xf16, #pto.address_space<gm>>,
                                 %arg1: memref<32x16xi16, #pto.address_space<gm>>) {
    %c0 = arith.constant 0 : index

    %base = pto.alloc_tile
      : !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=32, v_row=32, v_col=32,
                      blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %sub = pto.subset %base[%c0, %c0] sizes [16, 32]
      : !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=32, v_row=32, v_col=32,
                      blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %reshape = pto.treshape %sub
      : !pto.tile_buf<loc=vec, dtype=f16, rows=16, cols=32, v_row=16, v_col=32,
                      blayout=row_major, slayout=none_box, fractal=512, pad=0>
        -> !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=16, v_row=32, v_col=16,
                         blayout=row_major, slayout=none_box, fractal=512, pad=0>
    %cast = pto.bitcast %reshape
      : !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=16, v_row=32, v_col=16,
                      blayout=row_major, slayout=none_box, fractal=512, pad=0>
        -> !pto.tile_buf<loc=vec, dtype=i16, rows=32, cols=16, v_row=32, v_col=16,
                         blayout=row_major, slayout=none_box, fractal=512, pad=0>

    pto.tload ins(%arg0 : memref<32x32xf16, #pto.address_space<gm>>)
             outs(%base : !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=32, v_row=32, v_col=32,
                                        blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    pto.tstore ins(%cast : !pto.tile_buf<loc=vec, dtype=i16, rows=32, cols=16, v_row=32, v_col=16,
                                      blayout=row_major, slayout=none_box, fractal=512, pad=0>)
              outs(%arg1 : memref<32x16xi16, #pto.address_space<gm>>)
    return
  }
}
"""

if __name__ == "__main__":
    print(PTO_IR)
