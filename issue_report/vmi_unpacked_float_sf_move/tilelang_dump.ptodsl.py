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
def main_kernel(x: pto.ptr(pto.bf16, "gm"), x_q: pto.ptr(pto.f8e4m3, "gm"), x_sf: pto.ptr(pto.f32, "gm")):
  buf_dyn_shmem = pto.castptr(pto.const(0, dtype=pto.i64), pto.ptr(pto.ui8, "ub"))
  x_ub_version_counter_1 = pto.const(0, dtype=pto.int64)
  pto.set_flag("MTE3", "V", event_id=0)
  pto.set_flag("MTE3", "V", event_id=1)
  pto.set_flag("V", "MTE2", event_id=0)
  pto.set_flag("V", "MTE2", event_id=1)
  pto.set_flag("MTE3", "V", event_id=2)
  x_ub_version_counter_1 = _tl_wrap_surface_value(_tl_coerce_i64(0, context="PTO local.var store"))
  for w in range(0, 2, 1):
    __cond_0 = ((w * 3) + (pto.get_block_idx() // 24)) < 4
    if ((w * 3) + (pto.get_block_idx() // 24)) < 4:
      pto.wait_flag("V", "MTE2", event_id=x_ub_version_counter_1 & 1)
      pto.mte_gm_ub(pto.addptr(x, ((w * 1179648) + ((pto.get_block_idx() // 6) * 98304)) + ((pto.get_block_idx() % 6) * 512)), pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.bf16, "ub")), (x_ub_version_counter_1 & 1) * 16384), 0, 1024, nburst=(32, 6144, 1024))
      pto.set_flag("MTE2", "V", event_id=x_ub_version_counter_1 & 1)
      pto.wait_flag("MTE2", "V", event_id=x_ub_version_counter_1 & 1)
      with pto.vecscope():
        mask64 = pto.vmi.create_mask(64, size=64)
        mask32 = pto.vmi.create_mask(32, size=64)
        mask1 = pto.vmi.create_mask(1, size=1)
        lane64 = pto.vmi.vci(pto.ui32(0), size=64)
        mask_hi32 = pto.vmi.vcmps(lane64, 31, mask64, "gt")
        for row in range(0, 32, 1):
          for pair in range(0, 4, 1):
            x0 = pto.vmi.vcvt(pto.vmi.vload(pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.bf16, "ub")), (x_ub_version_counter_1 & 1) * 16384), ((row * 512) + (pair * 128)), size=64), to_dtype=pto.f32)
            x1 = pto.vmi.vcvt(pto.vmi.vload(pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.bf16, "ub")), (x_ub_version_counter_1 & 1) * 16384), (((row * 512) + (pair * 128)) + 64), size=64), to_dtype=pto.f32)
            abs0 = pto.vmi.vabs(x0, mask64)
            abs1 = pto.vmi.vabs(x1, mask64)
            pto.vmi.vstore(pto.vmi.vcmax(abs0, mask32), pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.f32, "ub")), ((x_ub_version_counter_1 & 1) * 2048) + 28672), ((row * 64) + (pair * 4)), mask1)
            pto.vmi.vstore(pto.vmi.vcmax(abs0, mask_hi32), pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.f32, "ub")), ((x_ub_version_counter_1 & 1) * 2048) + 28672), (((row * 64) + (pair * 4)) + 1), mask1)
            pto.vmi.vstore(pto.vmi.vcmax(abs1, mask32), pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.f32, "ub")), ((x_ub_version_counter_1 & 1) * 2048) + 28672), (((row * 64) + (pair * 4)) + 2), mask1)
            pto.vmi.vstore(pto.vmi.vcmax(abs1, mask_hi32), pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.f32, "ub")), ((x_ub_version_counter_1 & 1) * 2048) + 28672), (((row * 64) + (pair * 4)) + 3), mask1)
      with pto.vecscope():
        group_mask = pto.vmi.create_mask(16, size=64)
        fp8_max = pto.vmi.vbrc(pto.f32(float.fromhex('0x1.cp+8')), size=64)
        clamp_min = pto.vmi.vbrc(pto.f32(float.fromhex('0x1.a36e2eb1c432dp-14')), size=64)
        u1 = pto.vmi.vbrc(pto.ui32(1), size=64)
        u254 = pto.vmi.vbrc(pto.ui32(254), size=64)
        for row_1 in range(0, 32, 1):
          amax = pto.vmi.vmax(pto.vmi.vload(pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.f32, "ub")), ((x_ub_version_counter_1 & 1) * 2048) + 28672), (row_1 * 64), size=64), clamp_min, group_mask)
          raw = pto.vmi.vmuls(amax, float.fromhex('0x1.2492492492492p-9'), group_mask)
          bits = pto.vmi.vinterpret_cast(raw, to_dtype=pto.ui32)
          exp = pto.vmi.vadds(pto.vmi.vshrs(pto.vmi.vsub(bits, u1, group_mask), 23, group_mask), 1, group_mask)
          scale = pto.vmi.vinterpret_cast(pto.vmi.vshls(exp, 23, group_mask), to_dtype=pto.f32)
          inverse = pto.vmi.vinterpret_cast(pto.vmi.vshls(pto.vmi.vsub(u254, exp, group_mask), 23, group_mask), to_dtype=pto.f32)
          pto.vmi.vstore(scale, pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.f32, "ub")), ((x_ub_version_counter_1 & 1) * 2048) + 24576), (row_1 * 64), group_mask)
          pto.vmi.vstore(inverse, pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.f32, "ub")), ((x_ub_version_counter_1 & 1) * 2048) + 32768), (row_1 * 64), group_mask)
      pto.wait_flag("MTE3", "V", event_id=x_ub_version_counter_1 & 1)
      with pto.vecscope():
        mask64_1 = pto.vmi.create_mask(64, size=64)
        lane64_1 = pto.vmi.vci(pto.ui32(0), size=64)
        mask_hi32_1 = pto.vmi.vcmps(lane64_1, 31, mask64_1, "gt")
        for row_2 in range(0, 32, 1):
          for pair_1 in range(0, 4, 1):
            inverse0 = pto.vmi.vload(pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.f32, "ub")), ((x_ub_version_counter_1 & 1) * 2048) + 32768), ((row_2 * 64) + (pair_1 * 4)), dist_mode="brc", size=64)
            inverse1 = pto.vmi.vload(pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.f32, "ub")), ((x_ub_version_counter_1 & 1) * 2048) + 32768), (((row_2 * 64) + (pair_1 * 4)) + 1), dist_mode="brc", size=64)
            inverse2 = pto.vmi.vload(pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.f32, "ub")), ((x_ub_version_counter_1 & 1) * 2048) + 32768), (((row_2 * 64) + (pair_1 * 4)) + 2), dist_mode="brc", size=64)
            inverse3 = pto.vmi.vload(pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.f32, "ub")), ((x_ub_version_counter_1 & 1) * 2048) + 32768), (((row_2 * 64) + (pair_1 * 4)) + 3), dist_mode="brc", size=64)
            scale0 = pto.vmi.vsel(mask_hi32_1, inverse1, inverse0)
            scale1 = pto.vmi.vsel(mask_hi32_1, inverse3, inverse2)
            values0 = pto.vmi.vcvt(pto.vmi.vload(pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.bf16, "ub")), (x_ub_version_counter_1 & 1) * 16384), ((row_2 * 512) + (pair_1 * 128)), size=64), to_dtype=pto.f32)
            values1 = pto.vmi.vcvt(pto.vmi.vload(pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.bf16, "ub")), (x_ub_version_counter_1 & 1) * 16384), (((row_2 * 512) + (pair_1 * 128)) + 64), size=64), to_dtype=pto.f32)
            pto.vmi.vstore(pto.vmi.vcvt(pto.vmi.vmul(values0, scale0, mask64_1), rounding="R", saturate="SAT", to_dtype=pto.f8e4m3), pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.f8e4m3, "ub")), ((x_ub_version_counter_1 & 1) * 16384) + 65536), ((row_2 * 512) + (pair_1 * 128)), mask64_1)
            pto.vmi.vstore(pto.vmi.vcvt(pto.vmi.vmul(values1, scale1, mask64_1), rounding="R", saturate="SAT", to_dtype=pto.f8e4m3), pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.f8e4m3, "ub")), ((x_ub_version_counter_1 & 1) * 16384) + 65536), (((row_2 * 512) + (pair_1 * 128)) + 64), mask64_1)
      pto.set_flag("V", "MTE3", event_id=x_ub_version_counter_1 & 1)
      pto.set_flag("V", "MTE2", event_id=x_ub_version_counter_1 & 1)
      for sf_c in range(0, 16, 1):
        if sf_c == 0:
          pto.wait_flag("MTE3", "V", event_id=2)
        for i in pto.static_range(0, 32, 1):
          scalar.store(scalar.load(pto.castptr(buf_dyn_shmem, pto.ptr(pto.f32, "ub")), ((((x_ub_version_counter_1 & 1) * 2048) + (i * 64)) + sf_c) + 24576), pto.castptr(buf_dyn_shmem, pto.ptr(pto.f32, "ub")), ((sf_c * 32) + i) + 36864)
        if sf_c == 15:
          pto.set_flag("V", "MTE3", event_id=2)
      pto.wait_flag("V", "MTE3", event_id=x_ub_version_counter_1 & 1)
      pto.mte_ub_gm(pto.castptr(_tl_coerce_i64(((x_ub_version_counter_1 & 1) * 16384) + 65536, context="PTO local pointer offset"), pto.ptr(pto.f8e4m3, "ub")), pto.addptr(x_q, ((w * 1179648) + ((pto.get_block_idx() // 6) * 98304)) + ((pto.get_block_idx() % 6) * 512)), 512, nburst=(32, 512, 3072), l2_cache="naci")
      pto.set_flag("MTE3", "V", event_id=x_ub_version_counter_1 & 1)
      x_ub_version_counter_1 = _tl_wrap_surface_value(_tl_coerce_i64(x_ub_version_counter_1 + 1, context="PTO local.var store"))
      pto.wait_flag("V", "MTE3", event_id=2)
      pto.mte_ub_gm(pto.castptr(pto.const(147456, dtype=pto.int64), pto.ptr(pto.f32, "ub")), pto.addptr(x_sf, (((pto.get_block_idx() % 6) * 8192) + (w * 384)) + ((pto.get_block_idx() // 6) * 32)), 128, nburst=(16, 128, 2048), l2_cache="naci")
      pto.set_flag("MTE3", "V", event_id=2)
  pto.wait_flag("MTE3", "V", event_id=0)
  pto.wait_flag("MTE3", "V", event_id=1)
  pto.wait_flag("V", "MTE2", event_id=0)
  pto.wait_flag("V", "MTE2", event_id=1)
  pto.wait_flag("MTE3", "V", event_id=2)