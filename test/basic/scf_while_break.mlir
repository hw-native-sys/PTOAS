// RUN: ptoas %s | FileCheck %s

module {
  func.func @while_break_kernel() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c4 = arith.constant 4 : index
    %c32 = arith.constant 32 : index
    %true = arith.constant true
    %false = arith.constant false
    %one = arith.constant 1.0 : f32
    %ten = arith.constant 10.0 : f32

    %tile = pto.alloc_tile
      : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32,
                      blayout=row_major, slayout=none_box, fractal=512, pad=0>

    %final:2 = scf.while (%i = %c0, %alive = %true)
        : (index, i1) -> (index, i1) {
      %lt = arith.cmpi slt, %i, %c4 : index
      %go = arith.andi %lt, %alive : i1
      scf.condition(%go) %i, %alive : index, i1
    } do {
    ^bb0(%i2: index, %alive2: i1):
      pto.tadds ins(%tile, %one
        : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32,
                        blayout=row_major, slayout=none_box, fractal=512, pad=0>,
          f32)
        outs(%tile
        : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32,
                        blayout=row_major, slayout=none_box, fractal=512, pad=0>)
      %break_now = arith.cmpi eq, %i2, %c2 : index
      %next_i = arith.addi %i2, %c1 : index
      %yield:2 = scf.if %break_now -> (index, i1) {
        scf.yield %next_i, %false : index, i1
      } else {
        pto.tadds ins(%tile, %ten
          : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32,
                          blayout=row_major, slayout=none_box, fractal=512, pad=0>,
            f32)
          outs(%tile
          : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32,
                          blayout=row_major, slayout=none_box, fractal=512, pad=0>)
        scf.yield %next_i, %true : index, i1
      }
      scf.yield %yield#0, %yield#1 : index, i1
    }

    %stopped = arith.xori %final#1, %true : i1
    scf.if %stopped {
      pto.tadds ins(%tile, %one
        : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32,
                        blayout=row_major, slayout=none_box, fractal=512, pad=0>,
          f32)
        outs(%tile
        : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=32, v_col=32,
                        blayout=row_major, slayout=none_box, fractal=512, pad=0>)
    }
    return
  }
}

// CHECK: bool [[ALIVE:v[0-9]+]];
// CHECK: int32_t [[IV:v[0-9]+]];
// CHECK: [[IV]] =
// CHECK: [[ALIVE]] =
// CHECK: goto [[HEADER:label[0-9]+]];
// CHECK: [[HEADER]]:
// CHECK: bool [[COND:v[0-9]+]];
// CHECK: [[COND]] = false;
// CHECK: [[COND]] = [[IV]] < {{.*}} & [[ALIVE]];
// CHECK: if ([[COND]]) {
// CHECK: goto [[BODY:label[0-9]+]];
// CHECK: } else {
// CHECK: goto [[EXIT:label[0-9]+]];
// CHECK: [[BODY]]:
// CHECK: if ([[IV]] ==
// CHECK: [[IV]] =
// CHECK: [[ALIVE]] =
// CHECK: goto [[HEADER]];
// CHECK: [[EXIT]]:
