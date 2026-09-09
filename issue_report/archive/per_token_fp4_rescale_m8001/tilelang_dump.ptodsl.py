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

@pto.jit(name="main_kernel", kernel_kind="vector", target="a5", mode="explicit")
def main_kernel(out: pto.ptr(pto.f8e4m3, "gm"), out_sf: pto.ptr(pto.ui8, "gm"), x: pto.ptr(pto.f8e4m3, "gm"), x_sf: pto.ptr(pto.ui8, "gm")):
  buf_dyn_shmem = pto.castptr(pto.const(0, dtype=pto.i64), pto.ptr(pto.ui8, "ub"))
  sf_in_ub_version_counter_1 = pto.const(0, dtype=pto.int64)
  pto.set_flag("MTE3", "V", event_id=0)
  pto.set_flag("MTE3", "V", event_id=1)
  pto.set_flag("V", "MTE2", event_id=0)
  pto.set_flag("V", "MTE2", event_id=1)
  sf_in_ub_version_counter_1 = _tl_wrap_surface_value(_tl_coerce_i64(0, context="PTO local.var store"))
  with pto.vecscope():
    transpose_mask = pto.vmi.create_mask(128, size=128)
    lane = pto.vmi.vci(pto.ui16(0), size=128)
    source_row = pto.vmi.vshrs(lane, 4, transpose_mask)
    source_pair = pto.vmi.vand(lane, pto.vmi.vbrc(pto.ui16(15), size=128), transpose_mask)
    transpose_indices = pto.vmi.vadd(pto.vmi.vmuls(source_pair, 16, transpose_mask), source_row, transpose_mask)
    pto.vmi.vstore(transpose_indices, pto.castptr(buf_dyn_shmem, pto.ptr(pto.ui16, "ub")), 0, transpose_mask)
  for w in range(0, 2, 1):
    __cond_0 = ((w * 3) + (pto.get_block_idx() // 24)) < 4
    if ((w * 3) + (pto.get_block_idx() // 24)) < 4:
      pto.wait_flag("V", "MTE2", event_id=sf_in_ub_version_counter_1 & 1)
      pto.mte_gm_ub(pto.addptr(x, ((w * 1179648) + ((pto.get_block_idx() // 3) * 49152)) + ((pto.get_block_idx() % 3) * 1024)), pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.f8e4m3, "ub")), ((sf_in_ub_version_counter_1 & 1) * 16384) + 256), 0, 1024, nburst=(16, 3072, 1024))
      pto.mte_gm_ub(pto.addptr(x_sf, (((pto.get_block_idx() % 3) * 512) + (w * 24)) + (((pto.get_block_idx() // 3) // 2) * 2)), pto.addptr(buf_dyn_shmem, ((sf_in_ub_version_counter_1 & 1) * 512) + 136448), 0, 2, nburst=(16, 32, 32))
      pto.set_flag("MTE2", "V", event_id=sf_in_ub_version_counter_1 & 1)
      pto.mem_bar(pto.BarrierType.VST_VLD)
      pto.wait_flag("MTE2", "V", event_id=sf_in_ub_version_counter_1 & 1)
      with pto.vecscope():
        one = pto.vmi.create_mask(1, size=1)
        for si in range(0, 32, 1):
          u16 = pto.vmi.vcvt(pto.vmi.vload(pto.addptr(buf_dyn_shmem, ((sf_in_ub_version_counter_1 & 1) * 512) + 136448), (((si // 2) * 32) + (si & 1)), size=1), to_dtype=pto.ui16)
          shifted = pto.vmi.vshls(u16, 7, one)
          pto.vmi.vstore(pto.vmi.vinterpret_cast(shifted, to_dtype=pto.bf16), pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.bf16, "ub")), 68736), si, one)
      with pto.vecscope():
        mask128 = pto.vmi.create_mask(128, size=128)
        mask256 = pto.vmi.create_mask(256, size=256)
        sf_lane_idx128 = pto.vmi.vshrs(pto.vmi.vci(pto.ui16(0), order="ASC", size=128), 5, mask128)
        for row in range(0, 16, 1):
          for tile in range(0, 4, 1):
            scale_a = pto.vmi.vgather(pto.addptr(pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.bf16, "ub")), 68736), tile * 8), sf_lane_idx128, mask128)
            f0 = pto.vmi.vcvt(pto.vmi.vload(pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.f8e4m3, "ub")), ((sf_in_ub_version_counter_1 & 1) * 16384) + 256), ((row * 1024) + (tile * 256)), size=64), to_dtype=pto.f32)
            f1 = pto.vmi.vcvt(pto.vmi.vload(pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.f8e4m3, "ub")), ((sf_in_ub_version_counter_1 & 1) * 16384) + 256), (((row * 1024) + (tile * 256)) + 64), size=64), to_dtype=pto.f32)
            a = pto.vmi.vinterpret_cast(f0, to_dtype=pto.bf16)
            b = pto.vmi.vinterpret_cast(f1, to_dtype=pto.bf16)
            v = pto.vmi.vdintlv(a, b, mask128)
            __1 = (v)[0]
            bf_a = (v)[1]
            vals_a = pto.vmi.vmul((v)[1], scale_a, mask128)
            pto.vmi.vstore(vals_a, pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.bf16, "ub")), ((sf_in_ub_version_counter_1 & 1) * 16384) + 16512), ((row * 1024) + (tile * 256)), mask128)
            scale_b = pto.vmi.vgather(pto.addptr(pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.bf16, "ub")), 68736), (tile * 8) + 4), sf_lane_idx128, mask128)
            f2 = pto.vmi.vcvt(pto.vmi.vload(pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.f8e4m3, "ub")), ((sf_in_ub_version_counter_1 & 1) * 16384) + 256), (((row * 1024) + (tile * 256)) + 128), size=64), to_dtype=pto.f32)
            f3 = pto.vmi.vcvt(pto.vmi.vload(pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.f8e4m3, "ub")), ((sf_in_ub_version_counter_1 & 1) * 16384) + 256), (((row * 1024) + (tile * 256)) + 192), size=64), to_dtype=pto.f32)
            c = pto.vmi.vinterpret_cast(f2, to_dtype=pto.bf16)
            d = pto.vmi.vinterpret_cast(f3, to_dtype=pto.bf16)
            v_1 = pto.vmi.vdintlv(c, d, mask128)
            __2 = (v_1)[0]
            bf_b = (v_1)[1]
            vals_b = pto.vmi.vmul((v_1)[1], scale_b, mask128)
            pto.vmi.vstore(vals_b, pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.bf16, "ub")), ((sf_in_ub_version_counter_1 & 1) * 16384) + 16512), (((row * 1024) + (tile * 256)) + 128), mask128)
      pto.set_flag("V", "MTE2", event_id=sf_in_ub_version_counter_1 & 1)
      with pto.vecscope():
        amax_f32 = pto.vmi.vreg(8, pto.f32)
        mask256_1 = pto.vmi.create_mask(256, size=256)
        mask8 = pto.vmi.create_mask(8, size=8)
        abs_mask = pto.vmi.vbrc(pto.ui16(32767), size=256)
        clamp_min_f32 = pto.vmi.vbrc(pto.f32(float.fromhex('0x1.a36e2eb1c432dp-14')), size=8)
        fp8_recip_max = pto.vmi.vbrc(pto.f32(float.fromhex('0x1.2492492492492p-9')), size=8)
        u1 = pto.vmi.vbrc(pto.ui32(1), size=8)
        u254 = pto.vmi.vbrc(pto.ui32(254), size=8)
        for row_1 in range(0, 16, 1):
          for tile_1 in range(0, 4, 1):
            x_u = pto.vmi.vinterpret_cast(pto.vmi.vload(pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.bf16, "ub")), ((sf_in_ub_version_counter_1 & 1) * 16384) + 16512), ((row_1 * 1024) + (tile_1 * 256)), size=256), to_dtype=pto.ui16)
            abs_u = pto.vmi.vand(x_u, abs_mask, mask256_1)
            abs_f = pto.vmi.vcvt(pto.vmi.vinterpret_cast(abs_u, to_dtype=pto.bf16), to_dtype=pto.f32)
            amax_f32 = pto.vmi.vcmax(abs_f, mask256_1, group=8)
            amax_f32 = pto.vmi.vmax(amax_f32, clamp_min_f32, mask8)
            raw = pto.vmi.vmul(amax_f32, fp8_recip_max, mask8)
            bits = pto.vmi.vinterpret_cast(raw, to_dtype=pto.ui32)
            exp = pto.vmi.vadds(pto.vmi.vshrs(pto.vmi.vsub(bits, u1, mask8), 23, mask8), 1, mask8)
            ue = pto.vmi.vcvt(exp, to_dtype=pto.ui8)
            recip8 = pto.vmi.vinterpret_cast(pto.vmi.vshls(pto.vmi.vsub(u254, exp, mask8), 23, mask8), to_dtype=pto.f32)
            recip = pto.vmi.vcvt(recip8, to_dtype=pto.bf16)
            pto.vmi.vstore(ue, pto.addptr(buf_dyn_shmem, ((sf_in_ub_version_counter_1 & 1) * 512) + 131328), ((row_1 * 32) + (tile_1 * 8)), group=8, stride=1)
            pto.vmi.vstore(recip, pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.bf16, "ub")), 66176), ((row_1 * 128) + (tile_1 * 32)), group=8, stride=1)
      pto.mem_bar(pto.BarrierType.VST_VLD)
      pto.wait_flag("MTE3", "V", event_id=sf_in_ub_version_counter_1 & 1)
      with pto.vecscope():
        transpose_indices_1 = pto.vmi.vload(pto.castptr(buf_dyn_shmem, pto.ptr(pto.ui16, "ub")), 0, size=128)
        full_mask = pto.vmi.create_mask(128, size=128)
        values = pto.vmi.vload(pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.ui16, "ub")), ((sf_in_ub_version_counter_1 & 1) * 256) + 65664), 0, size=128)
        pto.vmi.vscatter(values, pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.ui16, "ub")), ((sf_in_ub_version_counter_1 & 1) * 256) + 68800), transpose_indices_1, full_mask)
        values_1 = pto.vmi.vload(pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.ui16, "ub")), ((sf_in_ub_version_counter_1 & 1) * 256) + 65664), 128, size=128)
        pto.vmi.vscatter(values_1, pto.addptr(pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.ui16, "ub")), ((sf_in_ub_version_counter_1 & 1) * 256) + 68800), 8), transpose_indices_1, full_mask)
      with pto.vecscope():
        mask256_q = pto.vmi.create_mask(256, size=256)
        mask128_q = pto.vmi.create_mask(128, size=128)
        for row_2 in range(0, 16, 1):
          for tile_2 in range(0, 4, 1):
            recip_v = pto.vmi.vload(pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.bf16, "ub")), 66176), ((row_2 * 128) + (tile_2 * 32)), dist_mode="brc", group=8, size=256, stride=1)
            x_bf = pto.vmi.vload(pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.bf16, "ub")), ((sf_in_ub_version_counter_1 & 1) * 16384) + 16512), ((row_2 * 1024) + (tile_2 * 256)), size=256)
            pto.vmi.vstore(pto.vmi.vcvt(pto.vmi.vmul(pto.vmi.vcvt(x_bf, to_dtype=pto.f32), pto.vmi.vcvt(recip_v, to_dtype=pto.f32), mask256_q), rounding="R", saturate="SAT", to_dtype=pto.f8e4m3), pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.f8e4m3, "ub")), ((sf_in_ub_version_counter_1 & 1) * 16384) + 98560), ((row_2 * 1024) + (tile_2 * 256)), mask256_q)
      pto.set_flag("V", "MTE3", event_id=sf_in_ub_version_counter_1 & 1)
      pto.wait_flag("V", "MTE3", event_id=sf_in_ub_version_counter_1 & 1)
      pto.mte_ub_gm(pto.castptr(_tl_coerce_i64(((sf_in_ub_version_counter_1 & 1) * 16384) + 98560, context="PTO local pointer offset"), pto.ptr(pto.f8e4m3, "ub")), pto.addptr(out, ((w * 1179648) + ((pto.get_block_idx() // 3) * 49152)) + ((pto.get_block_idx() % 3) * 1024)), 1024, nburst=(16, 1024, 3072), l2_cache="naci")
      pto.mte_ub_gm(pto.castptr(_tl_coerce_i64(((sf_in_ub_version_counter_1 & 1) * 512) + 137600, context="PTO local pointer offset"), pto.ptr(pto.ui8, "ub")), pto.addptr(out_sf, (((pto.get_block_idx() % 3) * 16384) + (w * 768)) + ((pto.get_block_idx() // 3) * 32)), 32, nburst=(16, 32, 1024), l2_cache="naci")
      pto.set_flag("MTE3", "V", event_id=sf_in_ub_version_counter_1 & 1)
      sf_in_ub_version_counter_1 = _tl_wrap_surface_value(_tl_coerce_i64(sf_in_ub_version_counter_1 + 1, context="PTO local.var store"))
  pto.wait_flag("MTE3", "V", event_id=0)
  pto.wait_flag("MTE3", "V", event_id=1)
  pto.wait_flag("V", "MTE2", event_id=0)
  pto.wait_flag("V", "MTE2", event_id=1)

