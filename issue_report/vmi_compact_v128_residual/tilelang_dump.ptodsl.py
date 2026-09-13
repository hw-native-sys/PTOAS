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
def main_kernel(x: pto.ptr(pto.bf16, "gm"), x_q: pto.ptr(pto.f8e4m3, "gm"), x_sf: pto.ptr(pto.ui8, "gm")):
  buf_dyn_shmem = pto.castptr(pto.const(0, dtype=pto.i64), pto.ptr(pto.ui8, "ub"))
  x_ub_version_counter_1 = pto.const(0, dtype=pto.int64)
  pto.set_flag("MTE3", "V", event_id=0)
  pto.set_flag("MTE3", "V", event_id=1)
  pto.set_flag("V", "MTE2", event_id=0)
  pto.set_flag("V", "MTE2", event_id=1)
  pto.set_flag("MTE3", "V", event_id=2)
  pto.set_flag("MTE3", "V", event_id=3)
  x_ub_version_counter_1 = _tl_wrap_surface_value(_tl_coerce_i64(0, context="PTO local.var store"))
  for w in range(0, 2, 1):
    __cond_0 = ((w * 9) + (pto.get_block_idx() // 8)) < 16
    if ((w * 9) + (pto.get_block_idx() // 8)) < 16:
      pto.wait_flag("V", "MTE2", event_id=x_ub_version_counter_1 & 1)
      pto.mte_gm_ub(pto.addptr(x, (w * 36864) + (pto.get_block_idx() * 512)), pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.bf16, "ub")), (x_ub_version_counter_1 & 1) * 512), 0, 1024, nburst=(1, 1024, 1024))
      pto.set_flag("MTE2", "V", event_id=x_ub_version_counter_1 & 1)
      pto.wait_flag("MTE3", "V", event_id=(x_ub_version_counter_1 & 1) + 2)
      pto.wait_flag("MTE2", "V", event_id=x_ub_version_counter_1 & 1)
      with pto.vecscope():
        mask256 = pto.vmi.create_mask(64, size=64)
        mask128 = pto.vmi.create_mask(64, size=64)
        mask8 = pto.vmi.create_mask(2, size=2)
        abs_mask = pto.vmi.vbrc(pto.ui16(32767), size=64)
        clamp_min_f32 = pto.vmi.vbrc(pto.f32(float.fromhex('0x1.a36e2eb1c432dp-14')), size=2)
        fp8_recip_max = pto.vmi.vbrc(pto.f32(float.fromhex('0x1.2492492492492p-9')), size=2)
        u1_e4 = pto.vmi.vbrc(pto.ui32(1), size=2)
        u254_e4 = pto.vmi.vbrc(pto.ui32(254), size=2)
        for i in range(0, 4, 1):
          for tile in range(0, 2, 1):
            x_u = pto.vmi.vinterpret_cast(pto.vmi.vload(pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.bf16, "ub")), (x_ub_version_counter_1 & 1) * 512), ((i * 128) + (tile * 64)), size=64), to_dtype=pto.ui16)
            abs_u = pto.vmi.vand(x_u, abs_mask, mask256)
            abs_f = pto.vmi.vcvt(pto.vmi.vinterpret_cast(abs_u, to_dtype=pto.bf16), to_dtype=pto.f32)
            amax_f32 = pto.vmi.vcmax(abs_f, mask256, group=2)
            amax_f32_1 = pto.vmi.vmax(amax_f32, clamp_min_f32, mask8)
            raw = pto.vmi.vmul(amax_f32_1, fp8_recip_max, mask8)
            bits = pto.vmi.vinterpret_cast(raw, to_dtype=pto.ui32)
            exp = pto.vmi.vadds(pto.vmi.vshrs(pto.vmi.vsub(bits, u1_e4, mask8), 23, mask8), 1, mask8)
            ue = pto.vmi.vcvt(exp, to_dtype=pto.ui8)
            recip8 = pto.vmi.vinterpret_cast(pto.vmi.vshls(pto.vmi.vsub(u254_e4, exp, mask8), 23, mask8), to_dtype=pto.f32)
            recip = pto.vmi.vcvt(recip8, to_dtype=pto.bf16)
            pto.vmi.vstore(ue, pto.addptr(buf_dyn_shmem, ((x_ub_version_counter_1 & 1) * 32) + 3072), ((i * 4) + (tile * 2)), group=2, stride=1)
            pto.vmi.vstore(recip, pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.bf16, "ub")), 1568), ((i * 64) + (tile * 32)), group=2, stride=1)
      pto.set_flag("V", "MTE3", event_id=(x_ub_version_counter_1 & 1) + 2)
      pto.wait_flag("V", "MTE3", event_id=(x_ub_version_counter_1 & 1) + 2)
      pto.mte_ub_gm(pto.castptr(_tl_coerce_i64(((x_ub_version_counter_1 & 1) * 32) + 3072, context="PTO local pointer offset"), pto.ptr(pto.ui8, "ub")), pto.addptr(x_sf, (w * 1152) + (pto.get_block_idx() * 16)), 16, nburst=(1, 16, 16), l2_cache="naci")
      pto.set_flag("MTE3", "V", event_id=(x_ub_version_counter_1 & 1) + 2)
      pto.wait_flag("MTE3", "V", event_id=x_ub_version_counter_1 & 1)
      with pto.vecscope():
        mask256_q = pto.vmi.create_mask(64, size=64)
        mask128_1 = pto.vmi.create_mask(64, size=64)
        mask64 = pto.vmi.create_mask(64, size=64)
        for i_1 in range(0, 4, 1):
          for tile_1 in range(0, 2, 1):
            recip_v = pto.vmi.vload(pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.bf16, "ub")), 1568), ((i_1 * 64) + (tile_1 * 32)), dist_mode="brc", group=2, size=64, stride=1)
            x_bf = pto.vmi.vload(pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.bf16, "ub")), (x_ub_version_counter_1 & 1) * 512), ((i_1 * 128) + (tile_1 * 64)), size=64)
            pto.vmi.vstore(pto.vmi.vcvt(pto.vmi.vmul(pto.vmi.vcvt(x_bf, to_dtype=pto.f32), pto.vmi.vcvt(recip_v, to_dtype=pto.f32), mask256_q), rounding="R", saturate="SAT", to_dtype=pto.f8e4m3), pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.f8e4m3, "ub")), ((x_ub_version_counter_1 & 1) * 512) + 2048), ((i_1 * 128) + (tile_1 * 64)), mask256_q)
      pto.set_flag("V", "MTE3", event_id=x_ub_version_counter_1 & 1)
      pto.set_flag("V", "MTE2", event_id=x_ub_version_counter_1 & 1)
      pto.wait_flag("V", "MTE3", event_id=x_ub_version_counter_1 & 1)
      pto.mte_ub_gm(pto.castptr(_tl_coerce_i64(((x_ub_version_counter_1 & 1) * 512) + 2048, context="PTO local pointer offset"), pto.ptr(pto.f8e4m3, "ub")), pto.addptr(x_q, (w * 36864) + (pto.get_block_idx() * 512)), 512, nburst=(1, 512, 512), l2_cache="naci")
      pto.set_flag("MTE3", "V", event_id=x_ub_version_counter_1 & 1)
      x_ub_version_counter_1 = _tl_wrap_surface_value(_tl_coerce_i64(x_ub_version_counter_1 + 1, context="PTO local.var store"))
  pto.wait_flag("MTE3", "V", event_id=0)
  pto.wait_flag("MTE3", "V", event_id=1)
  pto.wait_flag("V", "MTE2", event_id=0)
  pto.wait_flag("V", "MTE2", event_id=1)
  pto.wait_flag("MTE3", "V", event_id=2)
  pto.wait_flag("MTE3", "V", event_id=3)