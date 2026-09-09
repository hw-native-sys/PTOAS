from ptodsl import pto, scalar
from tilelang.contrib.ptodsl.dcache_bypass import (
  pto_read_gm_bypass_dcache as _tl_pto_read_gm_bypass_dcache,
  pto_write_gm_bypass_dcache as _tl_pto_write_gm_bypass_dcache,
)
from tilelang.contrib.ptodsl.simt import (
  scalar_div as _tl_scalar_div,
  scalar_rsqrt as _tl_scalar_rsqrt,
  simt_allreduce_max as _tl_simt_allreduce_max,
  simt_allreduce_min as _tl_simt_allreduce_min,
  simt_allreduce_sum as _tl_simt_allreduce_sum,
  vectorize_binary_f32x2 as _tl_vectorize_binary_f32x2,
  vectorize_unary_f32x2 as _tl_vectorize_unary_f32x2,
)
from ptodsl._ops import _coerce_i64 as _tl_coerce_i64
from ptodsl._surface_values import wrap_surface_value as _tl_wrap_surface_value

@pto.jit(name="per_channel_rescale_kernel_kernel", kernel_kind="vector", target="a5", mode="explicit")
def per_channel_rescale_kernel_kernel(out: pto.ptr(pto.f8e4m3, "gm"), out_sf: pto.ptr(pto.ui8, "gm"), x: pto.ptr(pto.f8e4m3, "gm"), x_sf_invs: pto.ptr(pto.si16, "gm")):
  buf_dyn_shmem = pto.castptr(pto.const(0, dtype=pto.i64), pto.ptr(pto.ui8, "ub"))
  sf_tma_ub_version_counter_1 = pto.const(0, dtype=pto.int64)
  pto.set_flag("MTE3", "V", event_id=0)
  pto.set_flag("MTE3", "V", event_id=1)
  pto.set_flag("MTE3", "V", event_id=2)
  pto.set_flag("V", "MTE2", event_id=0)
  pto.set_flag("V", "MTE2", event_id=1)
  pto.set_flag("V", "MTE2", event_id=2)
  pto.set_flag("V", "MTE2", event_id=3)
  pto.set_flag("V", "MTE2", event_id=4)
  pto.set_flag("V", "MTE2", event_id=5)
  sf_tma_ub_version_counter_1 = _tl_wrap_surface_value(_tl_coerce_i64(0, context="PTO local.var store"))
  for w in range(0, 4, 1):
    __cond_0 = ((w * 9) + (pto.get_block_idx() // 8)) < 28
    if ((w * 9) + (pto.get_block_idx() // 8)) < 28:
      pto.wait_flag("V", "MTE2", event_id=((3 & ((sf_tma_ub_version_counter_1 % 3) // pto.const(2147483648, dtype=pto.i64))) + (sf_tma_ub_version_counter_1 % 3)) + 3)
      pto.mte_gm_ub(pto.addptr(x_sf_invs, ((((w * 72) + pto.get_block_idx()) % 28) * 2048) + ((((w * 72) + pto.get_block_idx()) // 28) * 64)), pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.si16, "ub")), (((3 & ((sf_tma_ub_version_counter_1 % 3) // pto.const(2147483648, dtype=pto.i64))) * 256) + ((sf_tma_ub_version_counter_1 % 3) * 256)) + 54272), 0, 128, nburst=(4, 1024, 128))
      pto.set_flag("MTE2", "V", event_id=((3 & ((sf_tma_ub_version_counter_1 % 3) // pto.const(2147483648, dtype=pto.i64))) + (sf_tma_ub_version_counter_1 % 3)) + 3)
      pto.wait_flag("V", "MTE2", event_id=(3 & ((sf_tma_ub_version_counter_1 % 3) // pto.const(2147483648, dtype=pto.i64))) + (sf_tma_ub_version_counter_1 % 3))
      pto.mte_gm_ub(pto.addptr(x, ((((w * 72) + pto.get_block_idx()) // 28) * 458752) + ((((w * 72) + pto.get_block_idx()) % 28) * 256)), pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.f8e4m3, "ub")), ((3 & ((sf_tma_ub_version_counter_1 % 3) // pto.const(2147483648, dtype=pto.i64))) * 16384) + ((sf_tma_ub_version_counter_1 % 3) * 16384)), 0, 256, nburst=(64, 7168, 256))
      pto.set_flag("MTE2", "V", event_id=(3 & ((sf_tma_ub_version_counter_1 % 3) // pto.const(2147483648, dtype=pto.i64))) + (sf_tma_ub_version_counter_1 % 3))
      pto.wait_flag("MTE2", "V", event_id=((3 & ((sf_tma_ub_version_counter_1 % 3) // pto.const(2147483648, dtype=pto.i64))) + (sf_tma_ub_version_counter_1 % 3)) + 3)
      with pto.vecscope():
        one = pto.vmi.create_mask(1, size=1)
        ff = pto.vmi.vbrc(pto.ui16(255), size=1)
        for r in range(0, 64, 1):
          for p in range(0, 4, 1):
            packed = pto.vmi.vinterpret_cast(pto.vmi.vload(pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.si16, "ub")), (((3 & ((sf_tma_ub_version_counter_1 % 3) // pto.const(2147483648, dtype=pto.i64))) * 256) + ((sf_tma_ub_version_counter_1 % 3) * 256)) + 54272), ((p * 64) + r), size=1), to_dtype=pto.ui16)
            lo = pto.vmi.vcvt(pto.vmi.vand(packed, ff, one), saturate="SAT", to_dtype=pto.ui8)
            hi = pto.vmi.vcvt(pto.vmi.vshrs(packed, 8, one), saturate="SAT", to_dtype=pto.ui8)
            pto.vmi.vstore(lo, pto.addptr(buf_dyn_shmem, (((3 & ((sf_tma_ub_version_counter_1 % 3) // pto.const(2147483648, dtype=pto.i64))) * 2048) + ((sf_tma_ub_version_counter_1 % 3) * 2048)) + 98304), ((r * 32) + (p * 2)), one)
            pto.vmi.vstore(hi, pto.addptr(buf_dyn_shmem, (((3 & ((sf_tma_ub_version_counter_1 % 3) // pto.const(2147483648, dtype=pto.i64))) * 2048) + ((sf_tma_ub_version_counter_1 % 3) * 2048)) + 98304), (((r * 32) + (p * 2)) + 1), one)
      pto.set_flag("V", "MTE2", event_id=((3 & ((sf_tma_ub_version_counter_1 % 3) // pto.const(2147483648, dtype=pto.i64))) + (sf_tma_ub_version_counter_1 % 3)) + 3)
      with pto.vecscope():
        sf_u16 = pto.vmi.vreg(128, pto.ui16)
        mask_sf = pto.vmi.create_mask(128, size=128)
        for r_1 in range(0, 64, 1):
          sf_u16 = pto.vmi.vcvt(pto.vmi.vload(pto.addptr(buf_dyn_shmem, (((3 & ((sf_tma_ub_version_counter_1 % 3) // pto.const(2147483648, dtype=pto.i64))) * 2048) + ((sf_tma_ub_version_counter_1 % 3) * 2048)) + 98304), (r_1 * 32), size=128), to_dtype=pto.ui16)
          sf_u16 = pto.vmi.vshls(sf_u16, 7, mask_sf)
          pto.vmi.vstore(pto.vmi.vinterpret_cast(sf_u16, to_dtype=pto.bf16), pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.bf16, "ub")), 52224), (r_1 * 32), mask_sf)
      pto.wait_flag("MTE3", "V", event_id=(3 & ((sf_tma_ub_version_counter_1 % 3) // pto.const(2147483648, dtype=pto.i64))) + (sf_tma_ub_version_counter_1 % 3))
      pto.wait_flag("MTE2", "V", event_id=(3 & ((sf_tma_ub_version_counter_1 % 3) // pto.const(2147483648, dtype=pto.i64))) + (sf_tma_ub_version_counter_1 % 3))
      with pto.vecscope():
        amax_ui = pto.vmi.vreg(256, pto.ui16)
        amax_a = pto.vmi.vreg(128, pto.ui16)
        amax_b = pto.vmi.vreg(128, pto.ui16)
        mask128 = pto.vmi.create_mask(128, size=128)
        mask256 = pto.vmi.create_mask(256, size=256)
        zero_bf256 = pto.vmi.vbrc(pto.bf16(float.fromhex('0x0p+0')), size=256)
        zero_bf128 = pto.vmi.vbrc(pto.bf16(float.fromhex('0x0p+0')), size=128)
        zero_ui256 = pto.vmi.vbrc(pto.ui16(0), size=256)
        abs_mask256 = pto.vmi.vbrc(pto.ui16(32767), size=256)
        clamp_f128 = pto.vmi.vbrc(pto.f32(float.fromhex('0x1.a36e2eb1c432dp-14')), size=128)
        inv_qv128 = pto.vmi.vbrc(pto.f32(float.fromhex('0x1.2492492492492p-9')), size=128)
        qmax_v128 = pto.vmi.vbrc(pto.f32(float.fromhex('0x1.cp+8')), size=128)
        u1_128 = pto.vmi.vbrc(pto.ui32(1), size=128)
        u254_128 = pto.vmi.vbrc(pto.ui32(254), size=128)
        mask64 = pto.vmi.create_mask(64, size=64)
        clamp_f64 = pto.vmi.vbrc(pto.f32(float.fromhex('0x1.a36e2eb1c432dp-14')), size=64)
        inv_qv64 = pto.vmi.vbrc(pto.f32(float.fromhex('0x1.2492492492492p-9')), size=64)
        u1_64 = pto.vmi.vbrc(pto.ui32(1), size=64)
        u254_64 = pto.vmi.vbrc(pto.ui32(254), size=64)
        abs_ui128_p2 = pto.vmi.vbrc(pto.ui16(32767), size=128)
        zero_ui128_p2 = pto.vmi.vbrc(pto.ui16(0), size=128)
        for sf_row in range(0, 2, 1):
          amax_ui = zero_ui256
          amax_a = zero_ui128_p2
          amax_b = zero_ui128_p2
          sf_lane_idx128 = pto.vmi.vshrs(pto.vmi.vci(pto.ui16(0), order="ASC", size=128), 5, mask128)
          for row in range(0, 32, 1):
            scale_a = pto.vmi.vgather(pto.addptr(pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.bf16, "ub")), 52224), (sf_row * 1024) + (row * 32)), sf_lane_idx128, mask128)
            f0 = pto.vmi.vcvt(pto.vmi.vload(pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.f8e4m3, "ub")), ((3 & ((sf_tma_ub_version_counter_1 % 3) // pto.const(2147483648, dtype=pto.i64))) * 16384) + ((sf_tma_ub_version_counter_1 % 3) * 16384)), ((sf_row * 8192) + (row * 256)), size=64), to_dtype=pto.f32)
            f1 = pto.vmi.vcvt(pto.vmi.vload(pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.f8e4m3, "ub")), ((3 & ((sf_tma_ub_version_counter_1 % 3) // pto.const(2147483648, dtype=pto.i64))) * 16384) + ((sf_tma_ub_version_counter_1 % 3) * 16384)), (((sf_row * 8192) + (row * 256)) + 64), size=64), to_dtype=pto.f32)
            a = pto.vmi.vinterpret_cast(f0, to_dtype=pto.bf16)
            b = pto.vmi.vinterpret_cast(f1, to_dtype=pto.bf16)
            v = pto.vmi.vdintlv(a, b, mask128)
            __1 = (v)[0]
            bf_a = (v)[1]
            vals_a = pto.vmi.vmul((v)[1], scale_a, mask128)
            pto.vmi.vstore(vals_a, pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.bf16, "ub")), 55040), (row * 256), mask128)
            amax_a = pto.vmi.vmax(amax_a, pto.vmi.vand(pto.vmi.vinterpret_cast(vals_a, to_dtype=pto.ui16), abs_ui128_p2, mask128), mask128)
            scale_b = pto.vmi.vgather(pto.addptr(pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.bf16, "ub")), 52224), ((sf_row * 1024) + (row * 32)) + 4), sf_lane_idx128, mask128)
            f2 = pto.vmi.vcvt(pto.vmi.vload(pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.f8e4m3, "ub")), ((3 & ((sf_tma_ub_version_counter_1 % 3) // pto.const(2147483648, dtype=pto.i64))) * 16384) + ((sf_tma_ub_version_counter_1 % 3) * 16384)), (((sf_row * 8192) + (row * 256)) + 128), size=64), to_dtype=pto.f32)
            f3 = pto.vmi.vcvt(pto.vmi.vload(pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.f8e4m3, "ub")), ((3 & ((sf_tma_ub_version_counter_1 % 3) // pto.const(2147483648, dtype=pto.i64))) * 16384) + ((sf_tma_ub_version_counter_1 % 3) * 16384)), (((sf_row * 8192) + (row * 256)) + 192), size=64), to_dtype=pto.f32)
            c = pto.vmi.vinterpret_cast(f2, to_dtype=pto.bf16)
            d = pto.vmi.vinterpret_cast(f3, to_dtype=pto.bf16)
            v_1 = pto.vmi.vdintlv(c, d, mask128)
            __2 = (v_1)[0]
            bf_b = (v_1)[1]
            vals_b = pto.vmi.vmul((v_1)[1], scale_b, mask128)
            pto.vmi.vstore(vals_b, pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.bf16, "ub")), 55040), ((row * 256) + 128), mask128)
            amax_b = pto.vmi.vmax(amax_b, pto.vmi.vand(pto.vmi.vinterpret_cast(vals_b, to_dtype=pto.ui16), abs_ui128_p2, mask128), mask128)
          pto.vmi.vstore(pto.vmi.vinterpret_cast(amax_a, to_dtype=pto.bf16), pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.bf16, "ub")), 55040), 8192, mask128)
          pto.vmi.vstore(pto.vmi.vinterpret_cast(amax_b, to_dtype=pto.bf16), pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.bf16, "ub")), 55040), 8320, mask128)
          pto.mem_bar(pto.BarrierType.VST_VLD)
          amax_bf = pto.vmi.vload(pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.bf16, "ub")), 55040), 8192, size=256)
          v_2 = pto.vmi.vintlv(zero_bf256, amax_bf, mask256)
          lo_1 = (v_2)[0]
          hi_1 = (v_2)[1]
          amax0 = pto.vmi.vmax(pto.vmi.vinterpret_cast((v_2)[0], to_dtype=pto.f32), clamp_f128, mask128)
          amax1 = pto.vmi.vmax(pto.vmi.vinterpret_cast((v_2)[1], to_dtype=pto.f32), clamp_f128, mask128)
          pto.mem_bar(pto.BarrierType.VST_VLD)
          scale_raw0 = pto.vmi.vmul(amax0, inv_qv128, mask128)
          scale_raw1 = pto.vmi.vmul(amax1, inv_qv128, mask128)
          bits0 = pto.vmi.vinterpret_cast(scale_raw0, to_dtype=pto.ui32)
          bits1 = pto.vmi.vinterpret_cast(scale_raw1, to_dtype=pto.ui32)
          exp0 = pto.vmi.vadds(pto.vmi.vshrs(pto.vmi.vsub(bits0, u1_128, mask128), 23, mask128), 1, mask128)
          exp1 = pto.vmi.vadds(pto.vmi.vshrs(pto.vmi.vsub(bits1, u1_128, mask128), 23, mask128), 1, mask128)
          inverse0 = pto.vmi.vinterpret_cast(pto.vmi.vshls(pto.vmi.vsub(u254_128, exp0, mask128), 23, mask128), to_dtype=pto.f32)
          inverse1 = pto.vmi.vinterpret_cast(pto.vmi.vshls(pto.vmi.vsub(u254_128, exp1, mask128), 23, mask128), to_dtype=pto.f32)
          e0_u8 = pto.vmi.vcvt(exp0, saturate="SAT", to_dtype=pto.ui8)
          e1_u8 = pto.vmi.vcvt(exp1, saturate="SAT", to_dtype=pto.ui8)
          pto.vmi.vstore(e0_u8, pto.addptr(buf_dyn_shmem, (((3 & ((sf_tma_ub_version_counter_1 % 3) // pto.const(2147483648, dtype=pto.i64))) * 512) + ((sf_tma_ub_version_counter_1 % 3) * 512)) + 127488), (sf_row * 256), mask128)
          pto.vmi.vstore(e1_u8, pto.addptr(buf_dyn_shmem, (((3 & ((sf_tma_ub_version_counter_1 % 3) // pto.const(2147483648, dtype=pto.i64))) * 512) + ((sf_tma_ub_version_counter_1 % 3) * 512)) + 127488), ((sf_row * 256) + 128), mask128)
          for row_1 in range(0, 32, 1):
            x_bf0 = pto.vmi.vload(pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.bf16, "ub")), 55040), (row_1 * 256), size=128)
            x_bf1 = pto.vmi.vload(pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.bf16, "ub")), 55040), ((row_1 * 256) + 128), size=128)
            pto.vmi.vstore(pto.vmi.vcvt(pto.vmi.vmul(pto.vmi.vcvt(x_bf0, to_dtype=pto.f32), inverse0, mask128), rounding="R", saturate="SAT", to_dtype=pto.f8e4m3), pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.f8e4m3, "ub")), (((3 & ((sf_tma_ub_version_counter_1 % 3) // pto.const(2147483648, dtype=pto.i64))) * 16384) + ((sf_tma_ub_version_counter_1 % 3) * 16384)) + 49152), ((sf_row * 8192) + (row_1 * 256)), mask128)
            pto.vmi.vstore(pto.vmi.vcvt(pto.vmi.vmul(pto.vmi.vcvt(x_bf1, to_dtype=pto.f32), inverse1, mask128), rounding="R", saturate="SAT", to_dtype=pto.f8e4m3), pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.f8e4m3, "ub")), (((3 & ((sf_tma_ub_version_counter_1 % 3) // pto.const(2147483648, dtype=pto.i64))) * 16384) + ((sf_tma_ub_version_counter_1 % 3) * 16384)) + 49152), (((sf_row * 8192) + (row_1 * 256)) + 128), mask128)
        mask_pk = pto.vmi.create_mask(256, size=256)
        pe0 = pto.vmi.vload(pto.addptr(buf_dyn_shmem, (((3 & ((sf_tma_ub_version_counter_1 % 3) // pto.const(2147483648, dtype=pto.i64))) * 512) + ((sf_tma_ub_version_counter_1 % 3) * 512)) + 127488), 0, size=256)
        pe1 = pto.vmi.vload(pto.addptr(buf_dyn_shmem, (((3 & ((sf_tma_ub_version_counter_1 % 3) // pto.const(2147483648, dtype=pto.i64))) * 512) + ((sf_tma_ub_version_counter_1 % 3) * 512)) + 127488), 256, size=256)
        v_3 = pto.vmi.vintlv(pe0, pe1, mask_pk)
        plow = (v_3)[0]
        phigh = (v_3)[1]
        pto.vmi.vstore((v_3)[0], pto.addptr(buf_dyn_shmem, (((3 & ((sf_tma_ub_version_counter_1 % 3) // pto.const(2147483648, dtype=pto.i64))) * 512) + ((sf_tma_ub_version_counter_1 % 3) * 512)) + 129024), 0, mask_pk)
        pto.vmi.vstore((v_3)[1], pto.addptr(buf_dyn_shmem, (((3 & ((sf_tma_ub_version_counter_1 % 3) // pto.const(2147483648, dtype=pto.i64))) * 512) + ((sf_tma_ub_version_counter_1 % 3) * 512)) + 129024), 256, mask_pk)
      pto.set_flag("V", "MTE3", event_id=(3 & ((sf_tma_ub_version_counter_1 % 3) // pto.const(2147483648, dtype=pto.i64))) + (sf_tma_ub_version_counter_1 % 3))
      pto.set_flag("V", "MTE2", event_id=(3 & ((sf_tma_ub_version_counter_1 % 3) // pto.const(2147483648, dtype=pto.i64))) + (sf_tma_ub_version_counter_1 % 3))
      pto.wait_flag("V", "MTE3", event_id=(3 & ((sf_tma_ub_version_counter_1 % 3) // pto.const(2147483648, dtype=pto.i64))) + (sf_tma_ub_version_counter_1 % 3))
      pto.mte_ub_gm(pto.castptr(_tl_coerce_i64((((3 & ((sf_tma_ub_version_counter_1 % 3) // pto.const(2147483648, dtype=pto.i64))) * 16384) + ((sf_tma_ub_version_counter_1 % 3) * 16384)) + 49152, context="PTO local pointer offset"), pto.ptr(pto.f8e4m3, "ub")), pto.addptr(out, ((((w * 72) + pto.get_block_idx()) // 28) * 458752) + ((((w * 72) + pto.get_block_idx()) % 28) * 256)), 256, nburst=(64, 256, 7168), l2_cache="naci")
      pto.mte_ub_gm(pto.castptr(_tl_coerce_i64((((3 & ((sf_tma_ub_version_counter_1 % 3) // pto.const(2147483648, dtype=pto.i64))) * 512) + ((sf_tma_ub_version_counter_1 % 3) * 512)) + 129024, context="PTO local pointer offset"), pto.ptr(pto.ui8, "ub")), pto.addptr(out_sf, (w * 36864) + (pto.get_block_idx() * 512)), 512, nburst=(1, 512, 512), l2_cache="naci")
      pto.set_flag("MTE3", "V", event_id=(3 & ((sf_tma_ub_version_counter_1 % 3) // pto.const(2147483648, dtype=pto.i64))) + (sf_tma_ub_version_counter_1 % 3))
      sf_tma_ub_version_counter_1 = _tl_wrap_surface_value(_tl_coerce_i64(sf_tma_ub_version_counter_1 + 1, context="PTO local.var store"))
  pto.wait_flag("MTE3", "V", event_id=0)
  pto.wait_flag("MTE3", "V", event_id=1)
  pto.wait_flag("MTE3", "V", event_id=2)
  pto.wait_flag("V", "MTE2", event_id=0)
  pto.wait_flag("V", "MTE2", event_id=1)
  pto.wait_flag("V", "MTE2", event_id=2)
  pto.wait_flag("V", "MTE2", event_id=3)
  pto.wait_flag("V", "MTE2", event_id=4)
  pto.wait_flag("V", "MTE2", event_id=5)

