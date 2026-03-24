PTO_IR = r"""
module {
  func.func @tilebuf_planmemory_auto_addr() attributes {pto.entry} {
    %buf = pto.alloc_tile
      : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32,
                      blayout=row_major, slayout=none_box, fractal=512, pad=0>
    pto.tprint ins(%buf : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32,
                                      v_row=32, v_col=32, blayout=row_major,
                                      slayout=none_box, fractal=512, pad=0>)
    return
  }
}
"""

if __name__ == "__main__":
    print(PTO_IR)
