# Desired VMI for ASC per_block H=384 (no host pad).
#
# ASC: block_k = align(gcd(limit, hidden), 128); ceildiv(hidden, block_k).
# VMI already has a legal 128-lane strip for H=128 (tilelang_dump.ptodsl.py).
# The same strip at hidden=384 (three K tiles) must lower; PTOAS currently
# reports VMI-RESIDUAL-OP.
#
# The remainder of this file is the working H=128 dump once copied from
# tilelang_dump.ptodsl.py.
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

@pto.jit(name="per_block_cast_kernel_kernel", kernel_kind="vector", target="a5", mode="explicit")
def per_block_cast_kernel_kernel(out: pto.ptr(pto.f8e4m3, "gm"), out_sf: pto.ptr(pto.f32, "gm"), x: pto.ptr(pto.bf16, "gm")):
  buf_dyn_shmem = pto.castptr(pto.const(0, dtype=pto.i64), pto.ptr(pto.ui8, "ub"))
  x_ub_version_counter_1 = pto.const(0, dtype=pto.int64)
  x_ub_version_counter_1 = _tl_wrap_surface_value(_tl_coerce_i64(0, context="PTO local.var store"))
  __cond_0 = pto.get_block_idx() < 16
  if pto.get_block_idx() < 16:
    pto.mte_gm_ub(pto.addptr(x, pto.get_block_idx() * 4096), pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.bf16, "ub")), ((x_ub_version_counter_1 & 1) * 4096) + 4128), 0, 8192, nburst=(1, 8192, 8192))
    pto.set_flag("MTE2", "V", event_id=x_ub_version_counter_1 & 1)
    pto.wait_flag("MTE2", "V", event_id=x_ub_version_counter_1 & 1)
    with pto.vecscope():
      acc0 = pto.vmi.vreg(64, pto.f32)
      acc1 = pto.vmi.vreg(64, pto.f32)
      mask64 = pto.vmi.create_mask(64, size=64)
      mask128 = pto.vmi.create_mask(128, size=128)
      mask32 = pto.vmi.create_mask(32, size=64)
      mask1 = pto.vmi.create_mask(1, size=1)
      m_high = pto.vmi.vcmps(pto.vmi.vci(pto.si32(0), size=64), 31, mask64, "gt")
      fp8_max = pto.vmi.vbrc(pto.f32(float.fromhex('0x1.cp+8')), size=1)
      clamp_min = pto.vmi.vbrc(pto.f32(float.fromhex('0x1.a36e2eb1c432dp-14')), size=1)
      u1 = pto.vmi.vbrc(pto.ui32(1), size=1)
      u254 = pto.vmi.vbrc(pto.ui32(254), size=1)
      acc0 = pto.vmi.vbrc(pto.f32(float.fromhex('0x0p+0')), size=64)
      acc1 = pto.vmi.vbrc(pto.f32(float.fromhex('0x0p+0')), size=64)
      for row in range(0, 32, 1):
        values0 = pto.vmi.vload(pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.bf16, "ub")), ((x_ub_version_counter_1 & 1) * 4096) + 4128), (row * 128), size=64)
        values1 = pto.vmi.vload(pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.bf16, "ub")), ((x_ub_version_counter_1 & 1) * 4096) + 4128), ((row * 128) + 64), size=64)
        values0_1 = pto.vmi.vcvt(values0, to_dtype=pto.f32)
        values1_1 = pto.vmi.vcvt(values1, to_dtype=pto.f32)
        acc0 = pto.vmi.vmax(acc0, pto.vmi.vabs(values0_1, mask64), mask64)
        acc1 = pto.vmi.vmax(acc1, pto.vmi.vabs(values1_1, mask64), mask64)
      amax0 = pto.vmi.vmax(pto.vmi.vcmax(acc0, mask32, group=1), clamp_min, mask1)
      amax1 = pto.vmi.vmax(pto.vmi.vcmax(acc0, m_high, group=1), clamp_min, mask1)
      amax2 = pto.vmi.vmax(pto.vmi.vcmax(acc1, mask32, group=1), clamp_min, mask1)
      amax3 = pto.vmi.vmax(pto.vmi.vcmax(acc1, m_high, group=1), clamp_min, mask1)
      raw0 = pto.vmi.vdiv(amax0, fp8_max, mask1)
      raw1 = pto.vmi.vdiv(amax1, fp8_max, mask1)
      raw2 = pto.vmi.vdiv(amax2, fp8_max, mask1)
      raw3 = pto.vmi.vdiv(amax3, fp8_max, mask1)
      bits0 = pto.vmi.vinterpret_cast(raw0, to_dtype=pto.ui32)
      bits1 = pto.vmi.vinterpret_cast(raw1, to_dtype=pto.ui32)
      bits2 = pto.vmi.vinterpret_cast(raw2, to_dtype=pto.ui32)
      bits3 = pto.vmi.vinterpret_cast(raw3, to_dtype=pto.ui32)
      exp0 = pto.vmi.vadds(pto.vmi.vshrs(pto.vmi.vsub(bits0, u1, mask1), 23, mask1), 1, mask1)
      exp1 = pto.vmi.vadds(pto.vmi.vshrs(pto.vmi.vsub(bits1, u1, mask1), 23, mask1), 1, mask1)
      exp2 = pto.vmi.vadds(pto.vmi.vshrs(pto.vmi.vsub(bits2, u1, mask1), 23, mask1), 1, mask1)
      exp3 = pto.vmi.vadds(pto.vmi.vshrs(pto.vmi.vsub(bits3, u1, mask1), 23, mask1), 1, mask1)
      scale0 = pto.vmi.vinterpret_cast(pto.vmi.vshls(exp0, 23, mask1), to_dtype=pto.f32)
      scale1 = pto.vmi.vinterpret_cast(pto.vmi.vshls(exp1, 23, mask1), to_dtype=pto.f32)
      scale2 = pto.vmi.vinterpret_cast(pto.vmi.vshls(exp2, 23, mask1), to_dtype=pto.f32)
      scale3 = pto.vmi.vinterpret_cast(pto.vmi.vshls(exp3, 23, mask1), to_dtype=pto.f32)
      recip0 = pto.vmi.vinterpret_cast(pto.vmi.vshls(pto.vmi.vsub(u254, exp0, mask1), 23, mask1), to_dtype=pto.f32)
      recip1 = pto.vmi.vinterpret_cast(pto.vmi.vshls(pto.vmi.vsub(u254, exp1, mask1), 23, mask1), to_dtype=pto.f32)
      recip2 = pto.vmi.vinterpret_cast(pto.vmi.vshls(pto.vmi.vsub(u254, exp2, mask1), 23, mask1), to_dtype=pto.f32)
      recip3 = pto.vmi.vinterpret_cast(pto.vmi.vshls(pto.vmi.vsub(u254, exp3, mask1), 23, mask1), to_dtype=pto.f32)
      pto.vmi.vstore(scale0, pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.f32, "ub")), ((x_ub_version_counter_1 & 1) * 8) + 2048), 0, group=1, stride=1)
      pto.vmi.vstore(scale1, pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.f32, "ub")), ((x_ub_version_counter_1 & 1) * 8) + 2048), 1, group=1, stride=1)
      pto.vmi.vstore(scale2, pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.f32, "ub")), ((x_ub_version_counter_1 & 1) * 8) + 2048), 2, group=1, stride=1)
      pto.vmi.vstore(scale3, pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.f32, "ub")), ((x_ub_version_counter_1 & 1) * 8) + 2048), 3, group=1, stride=1)
      recip_v0 = pto.vmi.vsel(mask32, pto.vmi.vbrc(recip0, group=1, size=64), pto.vmi.vbrc(recip1, group=1, size=64))
      recip_v1 = pto.vmi.vsel(mask32, pto.vmi.vbrc(recip2, group=1, size=64), pto.vmi.vbrc(recip3, group=1, size=64))
      for row_1 in range(0, 32, 1):
        values0_2 = pto.vmi.vload(pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.bf16, "ub")), ((x_ub_version_counter_1 & 1) * 4096) + 4128), (row_1 * 128), size=64)
        values1_2 = pto.vmi.vload(pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.bf16, "ub")), ((x_ub_version_counter_1 & 1) * 4096) + 4128), ((row_1 * 128) + 64), size=64)
        values0_3 = pto.vmi.vcvt(values0_2, to_dtype=pto.f32)
        values1_3 = pto.vmi.vcvt(values1_2, to_dtype=pto.f32)
        quantized0 = pto.vmi.vcvt(pto.vmi.vmul(values0_3, recip_v0, mask64), rounding="R", saturate="SAT", to_dtype=pto.f8e4m3)
        quantized1 = pto.vmi.vcvt(pto.vmi.vmul(values1_3, recip_v1, mask64), rounding="R", saturate="SAT", to_dtype=pto.f8e4m3)
        pto.vmi.vstore(quantized0, pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.f8e4m3, "ub")), (x_ub_version_counter_1 & 1) * 4096), (row_1 * 128), mask64)
        pto.vmi.vstore(quantized1, pto.addptr(pto.castptr(buf_dyn_shmem, pto.ptr(pto.f8e4m3, "ub")), (x_ub_version_counter_1 & 1) * 4096), ((row_1 * 128) + 64), mask64)
    pto.set_flag("V", "MTE3", event_id=x_ub_version_counter_1 & 1)
    pto.wait_flag("V", "MTE3", event_id=x_ub_version_counter_1 & 1)
    pto.mte_ub_gm(pto.castptr(_tl_coerce_i64((x_ub_version_counter_1 & 1) * 4096, context="PTO local pointer offset"), pto.ptr(pto.f8e4m3, "ub")), pto.addptr(out, pto.get_block_idx() * 4096), 4096, nburst=(1, 4096, 4096), l2_cache="naci")
    pto.mte_ub_gm(pto.castptr(scalar.muli(_tl_coerce_i64(((x_ub_version_counter_1 & 1) * 8) + 2048, context="PTO local pointer offset"), pto.const(4, dtype=pto.int64)), pto.ptr(pto.f32, "ub")), pto.addptr(out_sf, pto.get_block_idx() * 4), 16, nburst=(1, 16, 16), l2_cache="naci")
    x_ub_version_counter_1 = _tl_wrap_surface_value(_tl_coerce_i64(x_ub_version_counter_1 + 1, context="PTO local.var store"))