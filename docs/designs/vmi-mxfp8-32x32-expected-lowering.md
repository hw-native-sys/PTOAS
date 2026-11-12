# VMI MXFP8 32x32 Expected VPTO Lowering

本文记录 `test/vpto/cases/vmi_new/kernels/tquant-mxfp8-32x32-nd/kernel.pto`
的预期 VPTO lower 结果。输入 VMI case 在 `vecscope` 内按 8 行一组循环，
每次处理一个 `256xf32` tile，也就是 8 行 x 32 列。

这里写的是设计目标，不是当前 `--emit-vpto` 的实际输出。重点是把 E8M0
scale 的内存效果写明确：每个 8x32 chunk 产生 8 个 scale byte。lowering
按 CCE 风格先写到 32B 对齐的 padded UB slot，再通过 UB->GM copy 的
`src_stride=32B, dst_stride=8B` 消除 UB padding，使 GM 端仍然连续。

## Complete Expected PTO File

```mlir
module attributes {pto.backend = "vpto", pto.target_arch = "a5"} {
  module attributes {pto.backend = "vpto", pto.kernel_kind = #pto.kernel_kind<vector>, pto.target_arch = "a5"} {
    func.func @vmi_tquant_mxfp8_32x32_nd_kernel(%src_gm: !pto.ptr<f32, gm>,
                                                %out_fp8_gm: !pto.ptr<ui8, gm>,
                                                %out_e8m0_gm: !pto.ptr<ui8, gm>) attributes {pto.kernel} {
      %false = arith.constant false
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c3 = arith.constant 3 : index
      %c4 = arith.constant 4 : index
      %c5 = arith.constant 5 : index
      %c6 = arith.constant 6 : index
      %c7 = arith.constant 7 : index
      %c8 = arith.constant 8 : index
      %c16 = arith.constant 16 : index
      %c24 = arith.constant 24 : index
      %c32 = arith.constant 32 : index
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      %c192 = arith.constant 192 : index

      %c0_i32 = arith.constant 0 : i32
      %c1_i32 = arith.constant 1 : i32
      %c2_i32 = arith.constant 2 : i32
      %c3_i32 = arith.constant 3 : i32
      %c4_i32 = arith.constant 4 : i32
      %c5_i32 = arith.constant 5 : i32
      %c6_i32 = arith.constant 6 : i32
      %c7_i32 = arith.constant 7 : i32
      %c8_i32 = arith.constant 8 : i32
      %c23_i32 = arith.constant 23 : i32
      %c24_i32 = arith.constant 24 : i32
      %c40_i32 = arith.constant 40 : i32
      %c48_i32 = arith.constant 48 : i32
      %c56_i32 = arith.constant 56 : i32
      %c254_i32 = arith.constant 254 : i32
      %c2139095040_i32 = arith.constant 2139095040 : i32

      %c0_i64 = arith.constant 0 : i64
      %c1_i64 = arith.constant 1 : i64
      %c4_i64 = arith.constant 4 : i64
      %c8_i64 = arith.constant 8 : i64
      %c32_i64 = arith.constant 32 : i64
      %c256_i64 = arith.constant 256 : i64
      %c1024_i64 = arith.constant 1024 : i64
      %c4096_i64 = arith.constant 4096 : i64
      %c8192_i64 = arith.constant 8192 : i64
      %c12288_i64 = arith.constant 12288 : i64

      %ub_src = pto.castptr %c0_i64 : i64 -> !pto.ptr<f32, ub>
      %ub_out_fp8_u8 = pto.castptr %c8192_i64 : i64 -> !pto.ptr<ui8, ub>
      %ub_out_fp8_f8 = pto.castptr %c8192_i64 : i64 -> !pto.ptr<f8E4M3FN, ub>
      %ub_out_e8m0 = pto.castptr %c12288_i64 : i64 -> !pto.ptr<ui8, ub>

      pto.copy_gm_to_ubuf %src_gm, %ub_src, %c0_i64, %c1_i64, %c4096_i64, %c0_i64, %c0_i64, %false, %c0_i64, %c4096_i64, %c4096_i64
        : !pto.ptr<f32, gm>, !pto.ptr<f32, ub>, i64, i64, i64, i64, i64, i1, i64, i64, i64
      pto.copy_gm_to_ubuf %out_fp8_gm, %ub_out_fp8_u8, %c0_i64, %c1_i64, %c1024_i64, %c0_i64, %c0_i64, %false, %c0_i64, %c1024_i64, %c1024_i64
        : !pto.ptr<ui8, gm>, !pto.ptr<ui8, ub>, i64, i64, i64, i64, i64, i1, i64, i64, i64
      pto.set_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]
      pto.wait_flag[<PIPE_MTE2>, <PIPE_V>, <EVENT_ID0>]

      pto.vecscope {
        scf.for %row = %c0 to %c32 step %c8 {
          %elem_off = arith.muli %row, %c32 : index
          %elem_off_64 = arith.addi %elem_off, %c64 : index
          %elem_off_128 = arith.addi %elem_off, %c128 : index
          %elem_off_192 = arith.addi %elem_off, %c192 : index

          %x0 = pto.vlds %ub_src[%elem_off] : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>
          %x1 = pto.vlds %ub_src[%elem_off_64] : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>
          %x2 = pto.vlds %ub_src[%elem_off_128] : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>
          %x3 = pto.vlds %ub_src[%elem_off_192] : !pto.ptr<f32, ub>, index -> !pto.vreg<64xf32>

          %d0, %d1 = pto.vdintlv %x0, %x1 : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>, !pto.vreg<64xf32>
          %d2, %d3 = pto.vdintlv %x2, %x3 : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>, !pto.vreg<64xf32>
          %d4, %d5 = pto.vdintlv %d0, %d2 : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>, !pto.vreg<64xf32>
          %d6, %d7 = pto.vdintlv %d1, %d3 : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>, !pto.vreg<64xf32>

          %all_b32 = pto.pset_b32 "PAT_ALL" : !pto.mask<b32>
          %slot8_b32 = pto.pge_b32 "PAT_VL8" : !pto.mask<b32>
          %vl8_b32 = pto.pset_b32 "PAT_VL8" : !pto.mask<b32>
          %vl16_b32 = pto.pset_b32 "PAT_VL16" : !pto.mask<b32>
          %vl24_b32, %unused24 = pto.plt_b32 %c24_i32 : i32 -> !pto.mask<b32>, i32
          %vl32_b32 = pto.pset_b32 "PAT_VL32" : !pto.mask<b32>
          %vl40_b32, %unused40 = pto.plt_b32 %c40_i32 : i32 -> !pto.mask<b32>, i32
          %vl48_b32, %unused48 = pto.plt_b32 %c48_i32 : i32 -> !pto.mask<b32>, i32
          %vl56_b32, %unused56 = pto.plt_b32 %c56_i32 : i32 -> !pto.mask<b32>, i32

          %abs0 = pto.vabs %d4, %all_b32 : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
          %abs1 = pto.vabs %d6, %all_b32 : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
          %abs2 = pto.vabs %d5, %all_b32 : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
          %abs3 = pto.vabs %d7, %all_b32 : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>

          %g0 = pto.vcgmax %abs0, %all_b32 : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
          %g1 = pto.vcgmax %abs1, %all_b32 : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
          %g2 = pto.vcgmax %abs2, %all_b32 : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
          %g3 = pto.vcgmax %abs3, %all_b32 : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
          %g01 = pto.vmax %g0, %g1, %slot8_b32 : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
          %g23 = pto.vmax %g2, %g3, %slot8_b32 : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
          %amax = pto.vmax %g01, %g23, %slot8_b32 : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>

          %amax_i32 = pto.vbitcast %amax : !pto.vreg<64xf32> -> !pto.vreg<64xi32>
          %exp_mask = pto.vdup %c2139095040_i32, %all_b32 : i32, !pto.mask<b32> -> !pto.vreg<64xi32>
          %shift = pto.vdup %c23_i32, %all_b32 : i32, !pto.mask<b32> -> !pto.vreg<64xi32>
          %emax = pto.vdup %c8_i32, %all_b32 : i32, !pto.mask<b32> -> !pto.vreg<64xi32>
          %scale_exp_bias = pto.vdup %c254_i32, %all_b32 : i32, !pto.mask<b32> -> !pto.vreg<64xi32>
          %exp_bits = pto.vand %amax_i32, %exp_mask, %all_b32 : !pto.vreg<64xi32>, !pto.vreg<64xi32>, !pto.mask<b32> -> !pto.vreg<64xi32>
          %exp = pto.vshr %exp_bits, %shift, %all_b32 : !pto.vreg<64xi32>, !pto.vreg<64xi32>, !pto.mask<b32> -> !pto.vreg<64xi32>
          %e8m0_payload_i32 = pto.vsub %exp, %emax, %all_b32 : !pto.vreg<64xi32>, !pto.vreg<64xi32>, !pto.mask<b32> -> !pto.vreg<64xi32>

          %idx0 = pto.vdup %c0_i32, %all_b32 : i32, !pto.mask<b32> -> !pto.vreg<64xi32>
          %idx1 = pto.vdup %c1_i32, %all_b32 : i32, !pto.mask<b32> -> !pto.vreg<64xi32>
          %idx2 = pto.vdup %c2_i32, %all_b32 : i32, !pto.mask<b32> -> !pto.vreg<64xi32>
          %idx3 = pto.vdup %c3_i32, %all_b32 : i32, !pto.mask<b32> -> !pto.vreg<64xi32>
          %idx4 = pto.vdup %c4_i32, %all_b32 : i32, !pto.mask<b32> -> !pto.vreg<64xi32>
          %idx5 = pto.vdup %c5_i32, %all_b32 : i32, !pto.mask<b32> -> !pto.vreg<64xi32>
          %idx6 = pto.vdup %c6_i32, %all_b32 : i32, !pto.mask<b32> -> !pto.vreg<64xi32>
          %idx7 = pto.vdup %c7_i32, %all_b32 : i32, !pto.mask<b32> -> !pto.vreg<64xi32>

          %not_vl8 = pto.pnot %vl8_b32, %all_b32 : !pto.mask<b32>, !pto.mask<b32> -> !pto.mask<b32>
          %range_8_15 = pto.pand %vl16_b32, %not_vl8, %all_b32 : !pto.mask<b32>, !pto.mask<b32>, !pto.mask<b32> -> !pto.mask<b32>
          %broadcast_idx_1 = pto.vsel %idx1, %idx0, %range_8_15 : !pto.vreg<64xi32>, !pto.vreg<64xi32>, !pto.mask<b32> -> !pto.vreg<64xi32>
          %not_vl16 = pto.pnot %vl16_b32, %all_b32 : !pto.mask<b32>, !pto.mask<b32> -> !pto.mask<b32>
          %range_16_23 = pto.pand %vl24_b32, %not_vl16, %all_b32 : !pto.mask<b32>, !pto.mask<b32>, !pto.mask<b32> -> !pto.mask<b32>
          %broadcast_idx_2 = pto.vsel %idx2, %broadcast_idx_1, %range_16_23 : !pto.vreg<64xi32>, !pto.vreg<64xi32>, !pto.mask<b32> -> !pto.vreg<64xi32>
          %not_vl24 = pto.pnot %vl24_b32, %all_b32 : !pto.mask<b32>, !pto.mask<b32> -> !pto.mask<b32>
          %range_24_31 = pto.pand %vl32_b32, %not_vl24, %all_b32 : !pto.mask<b32>, !pto.mask<b32>, !pto.mask<b32> -> !pto.mask<b32>
          %broadcast_idx_3 = pto.vsel %idx3, %broadcast_idx_2, %range_24_31 : !pto.vreg<64xi32>, !pto.vreg<64xi32>, !pto.mask<b32> -> !pto.vreg<64xi32>
          %not_vl32 = pto.pnot %vl32_b32, %all_b32 : !pto.mask<b32>, !pto.mask<b32> -> !pto.mask<b32>
          %range_32_39 = pto.pand %vl40_b32, %not_vl32, %all_b32 : !pto.mask<b32>, !pto.mask<b32>, !pto.mask<b32> -> !pto.mask<b32>
          %broadcast_idx_4 = pto.vsel %idx4, %broadcast_idx_3, %range_32_39 : !pto.vreg<64xi32>, !pto.vreg<64xi32>, !pto.mask<b32> -> !pto.vreg<64xi32>
          %not_vl40 = pto.pnot %vl40_b32, %all_b32 : !pto.mask<b32>, !pto.mask<b32> -> !pto.mask<b32>
          %range_40_47 = pto.pand %vl48_b32, %not_vl40, %all_b32 : !pto.mask<b32>, !pto.mask<b32>, !pto.mask<b32> -> !pto.mask<b32>
          %broadcast_idx_5 = pto.vsel %idx5, %broadcast_idx_4, %range_40_47 : !pto.vreg<64xi32>, !pto.vreg<64xi32>, !pto.mask<b32> -> !pto.vreg<64xi32>
          %not_vl48 = pto.pnot %vl48_b32, %all_b32 : !pto.mask<b32>, !pto.mask<b32> -> !pto.mask<b32>
          %range_48_55 = pto.pand %vl56_b32, %not_vl48, %all_b32 : !pto.mask<b32>, !pto.mask<b32>, !pto.mask<b32> -> !pto.mask<b32>
          %broadcast_idx_6 = pto.vsel %idx6, %broadcast_idx_5, %range_48_55 : !pto.vreg<64xi32>, !pto.vreg<64xi32>, !pto.mask<b32> -> !pto.vreg<64xi32>
          %not_vl56 = pto.pnot %vl56_b32, %all_b32 : !pto.mask<b32>, !pto.mask<b32> -> !pto.mask<b32>
          %range_56_63 = pto.pand %all_b32, %not_vl56, %all_b32 : !pto.mask<b32>, !pto.mask<b32>, !pto.mask<b32> -> !pto.mask<b32>
          %broadcast_idx = pto.vsel %idx7, %broadcast_idx_6, %range_56_63 : !pto.vreg<64xi32>, !pto.vreg<64xi32>, !pto.mask<b32> -> !pto.vreg<64xi32>

          %scale_u16 = pto.vpack %e8m0_payload_i32, "LOWER" : !pto.vreg<64xi32> -> !pto.vreg<128xui16>
          %scale_u8 = pto.vpack %scale_u16, "LOWER" : !pto.vreg<128xui16> -> !pto.vreg<256xui8>
          %scale_slot = arith.divui %row, %c8 : index
          %scale_ub_off = arith.muli %scale_slot, %c32 : index
          %scale8_b8 = pto.pge_b8 "PAT_VL8" : !pto.mask<b8>
          pto.vsts %scale_u8, %ub_out_e8m0[%scale_ub_off], %scale8_b8 {dist = "NORM_B8"} : !pto.vreg<256xui8>, !pto.ptr<ui8, ub>, index, !pto.mask<b8>

          %scale_exp = pto.vsub %scale_exp_bias, %e8m0_payload_i32, %all_b32 : !pto.vreg<64xi32>, !pto.vreg<64xi32>, !pto.mask<b32> -> !pto.vreg<64xi32>
          %scale_bits = pto.vshl %scale_exp, %shift, %all_b32 : !pto.vreg<64xi32>, !pto.vreg<64xi32>, !pto.mask<b32> -> !pto.vreg<64xi32>
          %scale_f32 = pto.vbitcast %scale_bits : !pto.vreg<64xi32> -> !pto.vreg<64xf32>
          %scale_vec = pto.vselr %scale_f32, %broadcast_idx : !pto.vreg<64xf32>, !pto.vreg<64xi32> -> !pto.vreg<64xf32>

          %m0 = pto.vmul %d4, %scale_vec, %all_b32 : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
          %m1 = pto.vmul %d6, %scale_vec, %all_b32 : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
          %m2 = pto.vmul %d5, %scale_vec, %all_b32 : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>
          %m3 = pto.vmul %d7, %scale_vec, %all_b32 : !pto.vreg<64xf32>, !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<64xf32>

          %i0, %i1 = pto.vintlv %m0, %m2 : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>, !pto.vreg<64xf32>
          %i2, %i3 = pto.vintlv %m1, %m3 : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>, !pto.vreg<64xf32>
          %i4, %i5 = pto.vintlv %i0, %i2 : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>, !pto.vreg<64xf32>
          %i6, %i7 = pto.vintlv %i1, %i3 : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>, !pto.vreg<64xf32>
          %r0, %r1 = pto.vdintlv %i4, %i5 : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>, !pto.vreg<64xf32>
          %r2, %r3 = pto.vdintlv %i6, %i7 : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>, !pto.vreg<64xf32>
          %r4, %r5 = pto.vdintlv %r0, %r2 : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>, !pto.vreg<64xf32>
          %r6, %r7 = pto.vdintlv %r1, %r3 : !pto.vreg<64xf32>, !pto.vreg<64xf32> -> !pto.vreg<64xf32>, !pto.vreg<64xf32>

          %all_b8 = pto.pset_b8 "PAT_ALL" : !pto.mask<b8>
          %q0 = pto.vcvt %r4, %all_b32 {part = "P0", rnd = "R", sat = "SAT"} : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<256xf8E4M3FN>
          %q1 = pto.vcvt %r6, %all_b32 {part = "P1", rnd = "R", sat = "SAT"} : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<256xf8E4M3FN>
          %q2 = pto.vcvt %r5, %all_b32 {part = "P2", rnd = "R", sat = "SAT"} : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<256xf8E4M3FN>
          %q3 = pto.vcvt %r7, %all_b32 {part = "P3", rnd = "R", sat = "SAT"} : !pto.vreg<64xf32>, !pto.mask<b32> -> !pto.vreg<256xf8E4M3FN>
          %q01 = pto.vor %q0, %q1, %all_b8 : !pto.vreg<256xf8E4M3FN>, !pto.vreg<256xf8E4M3FN>, !pto.mask<b8> -> !pto.vreg<256xf8E4M3FN>
          %q012 = pto.vor %q01, %q2, %all_b8 : !pto.vreg<256xf8E4M3FN>, !pto.vreg<256xf8E4M3FN>, !pto.mask<b8> -> !pto.vreg<256xf8E4M3FN>
          %q = pto.vor %q012, %q3, %all_b8 : !pto.vreg<256xf8E4M3FN>, !pto.vreg<256xf8E4M3FN>, !pto.mask<b8> -> !pto.vreg<256xf8E4M3FN>
          pto.vsts %q, %ub_out_fp8_f8[%elem_off], %all_b8 : !pto.vreg<256xf8E4M3FN>, !pto.ptr<f8E4M3FN, ub>, index, !pto.mask<b8>
        }
      }

      pto.set_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      pto.wait_flag[<PIPE_V>, <PIPE_MTE3>, <EVENT_ID0>]
      pto.copy_ubuf_to_gm %ub_out_fp8_u8, %out_fp8_gm, %c0_i64, %c1_i64, %c1024_i64, %c0_i64, %c1024_i64, %c1024_i64
        : !pto.ptr<ui8, ub>, !pto.ptr<ui8, gm>, i64, i64, i64, i64, i64, i64
      pto.copy_ubuf_to_gm %ub_out_e8m0, %out_e8m0_gm, %c0_i64, %c4_i64, %c8_i64, %c0_i64, %c8_i64, %c32_i64
        : !pto.ptr<ui8, ub>, !pto.ptr<ui8, gm>, i64, i64, i64, i64, i64, i64
      pto.barrier <PIPE_ALL>
      return
    }
  }
}
```

## Scale Store Contract

上面的 lower 对每次循环执行一条 `NORM_B8` store，写到 32B 对齐的 UB
slot：

```text
row = 0   -> UB[0..7],    UB[8..31] padding
row = 8   -> UB[32..39],  UB[40..63] padding
row = 16  -> UB[64..71],  UB[72..95] padding
row = 24  -> UB[96..103], UB[104..127] padding
```

最终 copy-out 只搬每个 slot 的前 8B：

```text
copy len = 8B
repeat = 4
source stride = 32B
destination stride = 8B
```

因此 GM 端效果仍然是连续 scale 输出：

```text
GM[0..7]   <- UB[0..7]
GM[8..15]  <- UB[32..39]
GM[16..23] <- UB[64..71]
GM[24..31] <- UB[96..103]
```
