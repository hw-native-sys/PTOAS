// tilelang.target = a5
 // tilelang.op = pto.tadd
 // tilelang.dtypes = (f32, f32, f32)
 // tilelang.verify = True
 // tilelang.advanced = False
 // tilelang.specialize dst shape=(8, 64) memory_space=ub config=None
 // tilelang.specialize src0 shape=(8, 64) memory_space=ub config=None
 // tilelang.specialize src1 shape=(8, 64) memory_space=ub config=None
 module attributes {pto.target_arch = "a5"} {
 func.func @template_tadd(%arg0: !pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=64, v_row=8, v_col=64, blayout=row_major, slayout=none_box, fractal=512, pad=0>, %arg1: !pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=64, v_row=8, v_col=64, blayout=row_major, slayout=none_box, fractal=512, pad=0>, %arg2: !pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=64, v_row=8, v_col=64, blayout=row_major, slayout=none_box, fractal=512, pad=0>) attributes { pto.tilelang.instance, pto.kernel_kind = #pto.kernel_kind<vector> } {
 %c0 = arith.constant 0 : index
 %c1 = arith.constant 1 : index
 %c64 = arith.constant 64 : index
 %tmp_0 = pto.tile_buf_addr %arg0 : !pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=64, v_row=8, v_col=64, blayout=row_major, slayout=none_box, fractal=512, pad=0> -> !pto.ptr<f32, ub>
 %tmp_1 = pto.tile_buf_addr %arg1 : !pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=64, v_row=8, v_col=64, blayout=row_major, slayout=none_box, fractal=512, pad=0> -> !pto.ptr<f32, ub>
 %tmp_2 = pto.tile_buf_addr %arg2 : !pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=64, v_row=8, v_col=64, blayout=row_major, slayout=none_box, fractal=512, pad=0> -> !pto.ptr<f32, ub>
 %valid_rows_1 = pto.tile_valid_rows %arg2 : !pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=64, v_row=8, v_col=64, blayout=row_major, slayout=none_box, fractal=512, pad=0> -> index
 %valid_cols_2 = pto.tile_valid_cols %arg2 : !pto.tile_buf<loc=vec, dtype=f32, rows=8, cols=64, v_row=8, v_col=64, blayout=row_major, slayout=none_box, fractal=512, pad=0> -> index
 scf.for %row_3 = %c0 to %valid_rows_1 step %c1 {
 %tmp_3 = arith.index_cast %valid_cols_2 : index to i32
 %remained_11:1 = scf.for %col_5 = %c0 to %valid_cols_2 step %c64 iter_args(%remained_iter_0 = %tmp_3) -> (i32) {
 %mask_6, %remained_7 = pto.plt_b32 %remained_iter_0 : i32 -> !pto.mask<b32>, i32
 %row_off_4 = arith.muli %row_3, %c64 : index
 %linear_5 = arith.addi %row_off_4, %col_5 : index
 %lhs_8 = pto.vlds %tmp_0[%linear_5] : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>
 %rhs_9 = pto.vlds %tmp_1[%linear_5] : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>
 %summed_10 = pto.vadd %lhs_8, %rhs_9, %mask_6 : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
 pto.vsts %summed_10, %tmp_2[%linear_5], %mask_6 : !pto.vreg<64xf32>, !pto.ptr<f32, ub>, index, !pto.mask<b32>
 scf.yield %remained_7 : i32
 }
 }
 return
 }
 }
